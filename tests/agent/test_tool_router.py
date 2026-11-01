"""Behavioral checks for optional, conservative keyword tool routing."""

import json
from pathlib import Path

import pytest

from nanobot.agent.tool_router import ToolRouter


@pytest.mark.parametrize("word", ["false", "also", "else", "pulses"])
def test_english_keywords_do_not_match_word_fragments(word):
    router = ToolRouter(always_available=set())

    assert router.route(f"schedule this {word}", ["cron", "list_dir"]) == ["cron"]
    assert router.route("schedule this; LS please", ["cron", "list_dir"]) == [
        "cron",
        "list_dir",
    ]


def test_chinese_keywords_and_https_still_match():
    router = ToolRouter(always_available=set())

    assert router.route("请每天搜索新闻", ["web_search", "cron", "web_fetch"]) == [
        "web_search",
        "cron",
    ]
    assert router.route("https://example.com", ["web_fetch", "cron"]) == ["web_fetch"]


def test_unknown_dynamic_tools_are_preserved_in_registry_order():
    router = ToolRouter(always_available=set())
    available = ["mcp_calendar_find", "web_fetch", "cron", "new_plugin_tool"]

    assert router.route("schedule a reminder", available) == [
        "mcp_calendar_find",
        "cron",
        "new_plugin_tool",
    ]
    assert available == ["mcp_calendar_find", "web_fetch", "cron", "new_plugin_tool"]


@pytest.mark.parametrize("action", ["read", "write", "edit"])
def test_file_operations_keep_registered_discovery_dependencies(action):
    router = ToolRouter(always_available=set())
    available = ["web_fetch", "list_dir", "edit_file", "read_file", "write_file"]
    expected = {"read_file", "list_dir", f"{action}_file"}

    assert router.route(f"{action} this file", available) == [
        name for name in available if name in expected
    ]
    assert router.route(f"{action} this file", [f"{action}_file"]) == [f"{action}_file"]


def test_session_and_knowledge_tools_are_available_during_filtered_routes():
    router = ToolRouter()

    assert router.route(
        "search this", ["edit_file", "session_manage", "web_search", "search_knowledge"]
    ) == ["session_manage", "web_search", "search_knowledge"]


def test_explicit_empty_configuration_is_respected():
    router = ToolRouter(routing_rules={}, always_available=set())
    available = ["web_search", "cron", "edit_file"]

    # With no configured rules, the conservative fallback returns all tools.
    assert router.route("search", available) == available
    router = ToolRouter(routing_rules={"cron": ["timer"]}, always_available=set())
    assert router.route("timer", ["exec", "message", "cron"]) == ["cron"]


def test_configuration_is_copied_and_fallback_returns_a_new_list():
    rules = {"cron": ["timer"]}
    always = set()
    router = ToolRouter(routing_rules=rules, always_available=always)
    rules["cron"].append("greeting")
    always.add("exec")
    available = ["exec", "cron", "web_fetch"]

    assert router.route("timer", available) == ["cron"]
    selected = router.route("greeting", available)
    assert selected == available
    selected.clear()
    assert available == ["exec", "cron", "web_fetch"]
    assert router.route("anything", []) == []


def test_history_boosts_only_recorded_calls_in_the_same_session():
    router = ToolRouter(always_available=set())
    available = ["cron", "web_fetch", "write_file", "read_file", "list_dir"]

    router.route("write this file", available, session_key="one")
    assert router.route("schedule", available, session_key="one") == ["cron"]
    router.record_tool_usage("one", "write_file")
    assert router.route("schedule", available, session_key="one") == [
        "cron",
        "write_file",
        "read_file",
        "list_dir",
    ]
    assert router.route("schedule", available, session_key="two") == ["cron"]
    assert router.route("schedule", ["cron"], session_key="one") == ["cron"]


def test_history_expires_old_calls_and_evicts_old_sessions():
    router = ToolRouter(always_available=set())
    router.record_tool_usage("old", "write_file")
    for _ in range(30):
        router.record_tool_usage("old", "cron")
    assert len(router._session_tool_history["old"]) == 20
    assert router.route("fetch", ["cron", "web_fetch", "write_file"], "old") == [
        "cron",
        "web_fetch",
    ]

    for number in range(128):
        router.record_tool_usage(f"session-{number}", "write_file")
    assert len(router._session_tool_history) == 128
    assert "old" not in router._session_tool_history
    assert router.route("schedule", ["cron", "write_file"], "old") == ["cron"]
    assert router.route("schedule", ["cron", "write_file"], "session-127") == [
        "cron",
        "write_file",
    ]


def test_route_count_is_independent_of_number_of_offered_tools():
    router = ToolRouter(always_available=set())
    router.route("hello", ["cron", "web_fetch"])
    router.route("schedule", ["cron", "web_fetch"])
    router.route("hello", [])
    router.record_tool_usage("one", "cron")

    assert router.get_stats() == {
        "total_routes": 3,
        "tool_frequency": {"cron": 2, "web_fetch": 1},
    }


def test_logging_is_disabled_by_default(tmp_path):
    router = ToolRouter(log_dir=tmp_path)
    router.route("search my private address", ["web_search"])

    assert not (tmp_path / "tool_routing_log.jsonl").exists()


def test_opt_in_logging_creates_directories_without_recording_message(tmp_path):
    log_dir = tmp_path / "nested" / "logs"
    router = ToolRouter(log_dir=log_dir, enable_logging=True)
    router.route("search private@example.com, password secret-value", ["web_search"])

    log_text = (log_dir / "tool_routing_log.jsonl").read_text(encoding="utf-8")
    entry = json.loads(log_text)
    assert entry["selected_tools"] == ["web_search"]
    assert entry["matched_tools"] == ["web_search"]
    assert "user_message" not in entry
    assert "private@example.com" not in log_text
    assert "secret-value" not in log_text


@pytest.mark.parametrize("operation", ["mkdir", "open"])
def test_logging_errors_do_not_interrupt_routing(tmp_path, monkeypatch, operation):
    router = ToolRouter(log_dir=tmp_path, enable_logging=True)

    def fail(*args, **kwargs):
        raise OSError("unwritable routing log")

    if operation == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    else:
        monkeypatch.setattr("builtins.open", fail)

    assert router.route("search", ["web_search", "cron"]) == ["web_search"]
