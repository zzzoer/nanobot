"""Session selection exercised through the real loop, runner, and tool registry."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


def make_loop(tmp_path, monkeypatch, responses, *, hooks=None):
    monkeypatch.setattr("nanobot.session.manager.get_legacy_sessions_dir", lambda: tmp_path / "legacy")
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=32)
    provider.estimate_prompt_tokens.return_value = (10, "test")
    provider.chat_with_retry = AsyncMock(side_effect=responses)
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model", hooks=hooks,
    )
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    # These tests focus on conversation routing; no background memory work is needed.
    loop._schedule_background = lambda coroutine: coroutine.close()
    return loop


def session_call(action, *, key=None, call_id="manage-session"):
    arguments = {"action": action}
    if key is not None:
        arguments["session_key"] = key
    return LLMResponse(None, tool_calls=[ToolCallRequest(call_id, "session_manage", arguments)])


@pytest.mark.asyncio
async def test_next_turn_selection_restart_and_history_are_consistent(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, monkeypatch, [
        session_call("create", key="research"), LLMResponse("created"),
        session_call("current", call_id="inspect-current"), LLMResponse("named reply"),
    ])
    try:
        response = await loop.process_direct("create a research session", session_key="sdk:alice")
        assert response.content == "created"
        key = "sdk:alice:session:research"
        assert loop.sessions.get_active_session_key("sdk", "alice") == key
        default = loop.sessions.get_or_create("sdk:alice")
        assert [m["role"] for m in default.messages] == ["user", "assistant", "tool", "assistant"]
        assert default.messages[-1]["content"] == "created"
        assert loop.sessions.get_or_create(key).messages == []
        assert list(loop.tool_router._session_tool_history["sdk:alice"]) == ["session_manage"]

        response = await loop.process_direct("which session is current?", session_key="sdk:alice")
        assert response.content == "named reply"
        named = loop.sessions.get_or_create(key)
        assert [m["role"] for m in named.messages] == ["user", "assistant", "tool", "assistant"]
        assert key in named.messages[2]["content"]
        assert list(loop.tool_router._session_tool_history[key]) == ["session_manage"]
        assert len(default.messages) == 4
    finally:
        await loop.close_mcp()

    restarted = make_loop(tmp_path, monkeypatch, [LLMResponse("continued")])
    try:
        await restarted.process_direct("continue my research", session_key="sdk:alice")
        assert restarted.sessions.get_active_session_key("sdk", "alice") == key
        named = restarted.sessions.get_or_create(key)
        assert [m["role"] for m in named.messages] == [
            "user", "assistant", "tool", "assistant", "user", "assistant",
        ]
        assert named.messages[-1]["content"] == "continued"
        assert len(restarted.sessions.get_or_create("sdk:alice").messages) == 4
    finally:
        await restarted.close_mcp()


@pytest.mark.parametrize("scope", ["sdk:default", "api:default", "matrix:!room:example.org"])
@pytest.mark.asyncio
async def test_explicit_scopes_keep_transport_context(tmp_path, monkeypatch, scope):
    loop = make_loop(tmp_path, monkeypatch, [session_call("create", key="notes"), LLMResponse("done")])
    try:
        response = await loop.process_direct(
            "create a session", session_key=scope, channel="cli", chat_id="direct",
        )
        assert (response.channel, response.chat_id) == ("cli", "direct")
        channel, chat_id = scope.split(":", 1)
        assert loop.sessions.get_active_session_key(channel, chat_id) == f"{scope}:session:notes"
        assert loop.tools.get("message")._default_channel == "cli"
        assert loop.tools.get("message")._default_chat_id == "direct"
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_thread_override_isolates_selection_and_preserves_delivery_chat(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, monkeypatch, [
        session_call("create", key="notes"), LLMResponse("created in topic one"),
        session_call("current"), LLMResponse("topic two unchanged"),
    ])
    try:
        first = InboundMessage(
            channel="telegram", chat_id="-100123", sender_id="user", content="create session",
            session_key_override="telegram:-100123:topic:1",
        )
        response = await loop._process_message(first)
        assert (response.channel, response.chat_id) == ("telegram", "-100123")
        assert loop.sessions.get_active_session_key("telegram", "-100123:topic:1") == (
            "telegram:-100123:topic:1:session:notes"
        )
        second = InboundMessage(
            channel="telegram", chat_id="-100123", sender_id="user", content="current session",
            session_key_override="telegram:-100123:topic:2",
        )
        await loop._process_message(second)
        other = loop.sessions.get_or_create(second.session_key)
        assert "telegram:-100123:topic:2" in other.messages[2]["content"]
        assert ":session:notes" not in other.messages[2]["content"]
        assert loop.sessions.get_active_session_key("telegram", "-100123:topic:2") is None
        assert loop.sessions.get_active_session_key("telegram", "-100123") is None
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_non_namespaced_sdk_key_preserves_history_and_disables_session_tool(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, monkeypatch, [session_call("create", key="notes"), LLMResponse("done")])
    try:
        await loop.process_direct("create a session", session_key="custom-sdk-key")
        session = loop.sessions.get_or_create("custom-sdk-key")
        assert session.messages[2]["content"].startswith("Error:")
        assert len(session.messages) == 4
        assert not loop.tool_router._session_tool_history
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_parallel_chats_do_not_share_session_tool_context(tmp_path, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    class PauseHook(AgentHook):
        async def before_execute_tools(self, context):
            if any(call.id == "create-first" for call in context.tool_calls):
                entered.set()
                await release.wait()

    async def respond(**kwargs):
        messages = kwargs["messages"]
        if messages[-1]["role"] == "tool":
            return LLMResponse("done")
        if "first chat" in messages[-1]["content"]:
            return LLMResponse(None, tool_calls=[
                ToolCallRequest("create-first", "session_manage", {"action": "create", "session_key": "first"}),
            ])
        return session_call("create", key="second")

    loop = make_loop(tmp_path, monkeypatch, respond, hooks=[PauseHook()])
    first = asyncio.create_task(loop.process_direct("first chat session", session_key="telegram:one"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        await loop.process_direct("second chat session", session_key="telegram:two")
        release.set()
        await asyncio.wait_for(first, timeout=5)

        assert loop.sessions.get_active_session_key("telegram", "one") == "telegram:one:session:first"
        assert loop.sessions.get_active_session_key("telegram", "two") == "telegram:two:session:second"
        assert not loop.sessions.session_exists("telegram:two:session:first")
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await loop.close_mcp()
