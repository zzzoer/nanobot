"""Exercise retrieval through the real agent runner with a local fake provider."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, ToolRoutingConfig
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


def make_loop(tmp_path, **kwargs):
    provider = MagicMock()
    provider.get_default_model.return_value = "local-test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (100, "test")
    loop = AgentLoop(MessageBus(), provider, tmp_path, **kwargs)
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    return loop, provider


@pytest.mark.asyncio
async def test_real_tool_roundtrip_retrieves_evidence_and_saves_legal_history(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "guide.md").write_text(
        "# Experiments\nRepeatability requires fixed seeds and recorded configuration.\n",
        encoding="utf-8",
    )
    loop, provider = make_loop(tmp_path)
    observed = {}

    async def chat(*, messages, tools, **kwargs):
        if messages[-1]["role"] != "tool":
            assert "search_knowledge" in {tool["function"]["name"] for tool in tools}
            return LLMResponse(content="", tool_calls=[ToolCallRequest(
                id="retrieve-1", name="search_knowledge", arguments={"query": "repeatability"},
            )])
        result = json.loads(messages[-1]["content"])
        observed.update(result["matches"][0])
        assert "fixed seeds" in observed["content"]
        return LLMResponse(content=(
            f"Record fixed seeds and configuration. "
            f"[{observed['source']}:{observed['start_line']}-{observed['end_line']}]"
        ))

    provider.chat_with_retry = chat
    response = await loop.process_direct("Search the knowledge base for repeatability")
    await loop.close_mcp()

    assert "knowledge/guide.md:1-2" in response.content
    history = loop.sessions.get_or_create("cli:direct").get_history()
    assert [message["role"] for message in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["id"] == history[2]["tool_call_id"]
    assert list(loop.tool_router._session_tool_history["cli:direct"]) == ["search_knowledge"]
    assert not (tmp_path / "tool_routing_log.jsonl").exists()


@pytest.mark.asyncio
async def test_disabled_routing_preserves_full_registry(tmp_path):
    loop, _ = make_loop(tmp_path, tool_routing_config=ToolRoutingConfig(enabled=False))
    loop.runner.run = AsyncMock(return_value=AgentRunResult("done", []))
    loop.tool_router.route = MagicMock(side_effect=AssertionError("Routing must be disabled"))

    await loop._run_agent_loop([{"role": "user", "content": "write a file"}])

    assert loop.runner.run.call_args.args[0].tools is loop.tools


@pytest.mark.asyncio
async def test_router_uses_text_without_runtime_metadata_or_image_payloads(tmp_path):
    loop, _ = make_loop(tmp_path)
    loop.runner.run = AsyncMock(return_value=AgentRunResult("done", []))
    loop.tool_router.route = MagicMock(return_value=loop.tools.tool_names)
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now"

    await loop._run_agent_loop([
        {"role": "user", "content": [
            {"type": "text", "text": runtime},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}},
            {"type": "text", "text": "read a file"},
        ]},
    ], session_key="cli:actual")

    assert loop.tool_router.route.call_args.args[0] == "read a file"
    assert loop.tool_router.route.call_args.kwargs["session_key"] == "cli:actual"


def test_routing_config_accepts_camel_case_and_defaults():
    config = Config.model_validate({"tools": {"routing": {"logDecisions": True}}})
    assert config.tools.routing.enabled
    assert config.tools.routing.log_decisions
    assert not Config().tools.routing.log_decisions
