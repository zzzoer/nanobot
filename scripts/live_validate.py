"""Opt-in real-provider smoke test using synthetic documents and a temporary workspace.

This script makes real requests to the model configured by the user. It never
copies provider credentials into the report or modifies the user's workspace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config, set_config_path
from nanobot.nanobot import _make_provider
from nanobot.providers.base import GenerationSettings


class Trace(AgentHook):
    def __init__(self):
        self.iterations = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def after_iteration(self, context: AgentHookContext) -> None:
        self.iterations += 1
        self.prompt_tokens += context.usage.get("prompt_tokens", 0)
        self.completion_tokens += context.usage.get("completion_tokens", 0)


async def validate(config_path: Path, output: Path) -> bool:
    logger.disable("nanobot")
    config = load_config(config_path)
    provider = _make_provider(config)
    provider.generation = GenerationSettings(temperature=0.0, max_tokens=512)
    client = getattr(provider, "_client", None)
    if client is not None:
        client.max_retries = 0
        client.timeout = 60.0
    secret = config.get_api_key() or ""
    trace = Trace()
    started = time.monotonic()
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "provider": config.get_provider_name(),
        "model": config.agents.defaults.model,
        "method": "real provider, real AgentLoop and CLI, synthetic data, temporary workspace",
        "checks": [],
    }

    def clean(text):
        text = str(text).replace(str(Path.home()), "<HOME>")
        return text.replace(secret, "<REDACTED>") if secret else text

    def record(name, passed, **evidence):
        report["checks"].append({"name": name, "passed": bool(passed), **evidence})
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="grounded-nanobot-live-") as temporary:
            workspace = Path(temporary)
            # Isolate legacy lookup as well as knowledge, sessions and memory.
            set_config_path(workspace / "config-context.json")
            loop = AgentLoop(
                MessageBus(), provider, workspace, max_iterations=4,
                restrict_to_workspace=True, hooks=[trace],
            )
            knowledge = workspace / "knowledge"
            knowledge.mkdir()
            notes = knowledge / "validation-notes.md"
            notes.write_text(
                "# Synthetic project notes\n"
                "项目代号是蓝鹭-42。\n"
                "重复实验使用三个随机种子：7、11、19。\n"
                "The release contact is Mira Chen.\n",
                encoding="utf-8",
            )

            async def ask(prompt, scope):
                selected = loop.sessions.get_active_session_key(*scope.split(":", 1)) or scope
                before = len(loop.sessions.get_or_create(selected).messages)
                response = await asyncio.wait_for(
                    loop.process_direct(prompt, session_key=scope), timeout=150,
                )
                new_messages = loop.sessions.get_or_create(selected).messages[before:]
                tool_results = [m for m in new_messages if m.get("role") == "tool"]
                answer = clean(response.content if response else "")
                return answer, tool_results

            answer, results = await ask(
                "请先调用 search_knowledge 检索本地知识库。项目代号和实验随机种子是什么？"
                "只依据工具返回资料回答，并用 knowledge/文件名:起始行-结束行 引用。",
                "cli:live-grounded",
            )
            matches = [json.loads(r["content"]) for r in results if r.get("name") == "search_knowledge"]
            record(
                "chinese_grounded_answer_and_citation",
                bool(matches) and all(t in answer for t in ("蓝鹭-42", "7", "11", "19", "knowledge/validation-notes.md"))
                and any(m.get("matches") for m in matches),
                answer=answer, retrieval_results=matches,
            )

            answer, results = await ask(
                "Use search_knowledge with query unlisted_quasar_phone. What is that phone number? "
                "If no document provides it, reply exactly NOT_IN_KNOWLEDGE. Do not invent a number.",
                "cli:live-unknown",
            )
            record(
                "unsupported_question_abstention",
                "NOT_IN_KNOWLEDGE" in answer and any(
                    r.get("name") == "search_knowledge"
                    and json.loads(r["content"]).get("status") == "no_results" for r in results
                ), answer=answer,
            )

            notes.write_text(
                "# Revised synthetic project notes\n项目代号是蓝鹭-42。\n"
                "重复实验现使用五个随机种子：7、11、19、23、29。\n",
                encoding="utf-8",
            )
            answer, results = await ask(
                "请调用 search_knowledge（refresh=true）查询项目现在的全部随机种子。"
                "只按最新知识库回答，并引用文件行号。",
                "cli:live-updated",
            )
            record(
                "changed_file_refresh",
                all(t in answer for t in ("7", "11", "19", "23", "29"))
                and any(r.get("name") == "search_knowledge" and "29" in r["content"] for r in results),
                answer=answer,
            )

            notes.unlink()
            answer, results = await ask(
                "Use search_knowledge with query 蓝鹭 and refresh=true. "
                "If the knowledge base has no result, reply exactly NOT_IN_KNOWLEDGE.",
                "cli:live-deleted",
            )
            record(
                "deleted_file_refresh",
                "NOT_IN_KNOWLEDGE" in answer and any(
                    r.get("name") == "search_knowledge"
                    and json.loads(r["content"]).get("status") == "no_results" for r in results
                ), answer=answer,
            )

            scope = "cli:live-sessions"
            answer, results = await ask(
                "Use session_manage to create and activate session_key='研究笔记', title='研究笔记'. "
                "Do not write any files with other tools. Confirm when complete.", scope,
            )
            selected = loop.sessions.get_active_session_key("cli", "live-sessions")
            record(
                "named_session_created_by_model",
                bool(selected) and any(r.get("name") == "session_manage" for r in results),
                answer=answer, active_session=selected,
            )
            answer, _ = await ask(
                "Remember this marker only in this conversation: ORBIT-9271. Reply MEMORY_SAVED.", scope,
            )
            named = loop.sessions.get_or_create(selected or scope)
            default = loop.sessions.get_or_create(scope)
            record(
                "next_turn_uses_named_history",
                bool(selected) and any("ORBIT-9271" in str(m.get("content")) for m in named.messages)
                and not any("ORBIT-9271" in str(m.get("content")) for m in default.messages),
                answer=answer,
            )
            await loop.close_mcp()

            # A new OS process exercises the actual CLI and disk-backed history.
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "nanobot", "agent", "--config", str(config_path),
                "--workspace", str(workspace), "--session", scope, "--no-markdown",
                "--message", "What marker did I ask you to remember? Reply only the exact marker. Do not use tools.",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=150)
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                raise RuntimeError("CLI restart validation timed out") from None
            cli_answer = clean(stdout.decode("utf-8", errors="replace"))
            record(
                "new_cli_process_restores_named_history",
                process.returncode == 0 and "ORBIT-9271" in cli_answer,
                exit_code=process.returncode, output=cli_answer,
                stderr=clean(stderr.decode("utf-8", errors="replace"))[:1500],
            )

            # Reload after the separate process so reset cannot overwrite its new turn.
            loop = AgentLoop(
                MessageBus(), provider, workspace, max_iterations=4,
                restrict_to_workspace=True, hooks=[trace],
            )
            answer, results = await ask(
                "Use session_manage action reset to return to the default conversation. "
                "Keep all histories and confirm when complete.", scope,
            )
            record(
                "reset_preserves_named_history",
                loop.sessions.get_active_session_key("cli", "live-sessions") is None
                and bool(selected) and bool(loop.sessions.get_existing(selected).messages)
                and any(r.get("name") == "session_manage" for r in results),
                answer=answer,
            )
            await loop.close_mcp()
    except Exception as exc:
        record("run_completed", False, error_type=type(exc).__name__)
    finally:
        if client is not None:
            await client.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["harness_model_iterations"] = trace.iterations
        report["harness_prompt_tokens"] = trace.prompt_tokens
        report["harness_completion_tokens"] = trace.completion_tokens
        report["usage_note"] = "CLI subprocess and connectivity probe tokens are not included."
        report["all_passed"] = bool(report["checks"]) and all(c["passed"] for c in report["checks"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"all_passed": report["all_passed"], "checks": len(report["checks"]),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report["all_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing provider config; never copied")
    parser.add_argument("--output", type=Path, required=True, help="Synthetic-data-only evidence JSON")
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(validate(args.config.expanduser().resolve(), args.output.resolve())) else 1)
