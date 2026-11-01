"""Adaptive Tool Router - pre-filters tools based on user intent."""

import json
import re
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Keyword-to-tool mapping: maps intent keywords to relevant tool names
DEFAULT_ROUTING_RULES: dict[str, list[str]] = {
    "web_search": [
        "搜索",
        "查找",
        "search",
        "google",
        "查一下",
        "找一下",
        "搜一下",
        "新闻",
        "天气",
        "最新",
        "搜下",
    ],
    "web_fetch": ["打开", "访问", "fetch", "网页", "url", "链接", "http", "https"],
    "read_file": ["读取", "查看文件", "read", "打开文件", "看一下文件", "文件内容"],
    "write_file": ["写入", "创建文件", "write", "保存", "新建文件"],
    "edit_file": ["编辑", "修改文件", "edit", "改一下"],
    "list_dir": ["列出", "目录", "ls", "文件列表", "有哪些文件"],
    "exec": ["执行", "运行", "命令", "exec", "shell", "终端", "pip", "python"],
    "session_manage": ["会话", "session", "切换", "新建会话", "列出会话"],
    "message": ["发送", "消息", "通知", "send", "告诉"],
    "spawn": ["后台", "子任务", "异步", "background"],
    "cron": ["定时", "定期", "cron", "schedule", "每天", "每小时"],
}

# Tools that should always be available regardless of routing
ALWAYS_AVAILABLE_TOOLS = {
    "web_search",
    "read_file",
    "exec",
    "message",
    "session_manage",
    "search_knowledge",
}

TOOL_DEPENDENCIES = {
    "read_file": {"list_dir"},
    "write_file": {"read_file", "list_dir"},
    "edit_file": {"read_file", "list_dir"},
}


class ToolRouter:
    """Pre-filters tools based on user intent to reduce token usage and improve selection."""

    def __init__(
        self,
        routing_rules: dict[str, list[str]] | None = None,
        always_available: set[str] | None = None,
        log_dir: Path | None = None,
        enable_logging: bool = False,
    ):
        rules = DEFAULT_ROUTING_RULES if routing_rules is None else routing_rules
        self.routing_rules = {name: list(keywords) for name, keywords in rules.items()}
        self.always_available = set(
            ALWAYS_AVAILABLE_TOOLS if always_available is None else always_available
        )
        self.enable_logging = enable_logging
        self._log_path = (log_dir if log_dir is not None else Path(".")) / "tool_routing_log.jsonl"
        self._route_count = 0
        self._stats: dict[str, int] = {}  # tool_name -> times_routed
        # Bound both calls per session and the number of retained sessions.
        self._session_tool_history: OrderedDict[str, deque[str]] = OrderedDict()

    def route(
        self,
        user_message: str,
        available_tools: list[str],
        session_key: str | None = None,
    ) -> list[str]:
        """Select relevant tools based on user message and session history.

        Args:
            user_message: The user's input message
            available_tools: All registered tool names
            session_key: Current session key for history-based boosting

        Returns:
            Filtered list of tool names to send to LLM
        """
        available = set(available_tools)
        # Unknown tools (including dynamic MCP tools) must remain discoverable.
        known_tools = set(DEFAULT_ROUTING_RULES) | ALWAYS_AVAILABLE_TOOLS | set(self.routing_rules)
        matched_tools = self.always_available | (available - known_tools)
        keyword_tools: set[str] = set()

        message_lower = user_message.lower()

        for tool_name, keywords in self.routing_rules.items():
            if tool_name not in available:
                continue
            for keyword in keywords:
                if self._matches_keyword(message_lower, keyword):
                    matched_tools.add(tool_name)
                    keyword_tools.add(tool_name)
                    break

        # History-based boosting: if a tool was used recently in this session,
        # include it even if keywords didn't match
        boosted: list[str] = []
        if session_key and session_key in self._session_tool_history:
            self._session_tool_history.move_to_end(session_key)
            recent = list(self._session_tool_history[session_key])[-5:]
            for tool_name in recent:
                if tool_name in available and tool_name not in matched_tools:
                    matched_tools.add(tool_name)
                    boosted.append(tool_name)

        if not keyword_tools and not boosted:
            selected = list(available_tools)
            decision = "fallback_all"
        else:
            for tool_name in tuple(matched_tools):
                matched_tools.update(TOOL_DEPENDENCIES.get(tool_name, ()))
            selected = [t for t in available_tools if t in matched_tools]
            decision = "filtered"

        # Update stats
        self._route_count += 1
        for tool in dict.fromkeys(selected):
            self._stats[tool] = self._stats.get(tool, 0) + 1

        # Log routing decision
        if self.enable_logging:
            self._log_decision(
                available_count=len(available_tools),
                selected_count=len(selected),
                selected_tools=selected,
                matched_tools=[t for t in available_tools if t in keyword_tools],
                boosted_from_history=boosted,
                decision=decision,
            )

        return selected

    @staticmethod
    def _matches_keyword(message: str, keyword: str) -> bool:
        """Match English words without matching fragments such as 'ls' in 'false'."""
        keyword = keyword.lower()
        if not keyword:
            return False
        if keyword.isascii():
            pattern = r"(?<![a-z0-9_])" + re.escape(keyword) + r"(?![a-z0-9_])"
            return re.search(pattern, message) is not None
        return keyword in message

    def record_tool_usage(self, session_key: str, tool_name: str) -> None:
        """Record an executed tool, not a tool merely offered by routing."""
        if session_key not in self._session_tool_history:
            self._session_tool_history[session_key] = deque(maxlen=20)
        self._session_tool_history.move_to_end(session_key)
        self._session_tool_history[session_key].append(tool_name)
        if len(self._session_tool_history) > 128:
            self._session_tool_history.popitem(last=False)

    def get_stats(self) -> dict[str, Any]:
        """Return route calls and per-tool selections, not execution counts."""
        return {
            "total_routes": self._route_count,
            "tool_frequency": dict(sorted(self._stats.items(), key=lambda x: -x[1])),
        }

    def _log_decision(self, **kwargs: Any) -> None:
        """Append metadata only; never persist the user's message."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"timestamp": datetime.now().isoformat(), **kwargs}
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write tool routing log")
