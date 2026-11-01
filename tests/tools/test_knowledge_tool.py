"""Real-file and SQLite coverage for local reference retrieval."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nanobot.agent.tools.knowledge import KnowledgeSearchTool


def document(workspace: Path, name: str, text: str) -> Path:
    path = workspace / "knowledge" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def search(tool, query="regression", **kwargs):
    return json.loads(await tool.execute(query=query, **kwargs))


async def test_empty_library_is_actionable(tmp_path):
    result = await search(KnowledgeSearchTool(tmp_path))
    assert result["status"] == "no_results"
    assert result["indexed_files"] == 0
    assert result["matches"] == []
    assert "workspace/knowledge" in result["message"]


async def test_search_returns_original_text_and_line_citations(tmp_path):
    document(tmp_path, "models/linear.md", "# Models\nLinear regression predicts a numeric target.\n")
    document(tmp_path, "other.txt", "Unrelated material about gardens.")
    result = await search(KnowledgeSearchTool(tmp_path))
    match, = result["matches"]
    assert match["source"] == "knowledge/models/linear.md"
    assert (match["start_line"], match["end_line"]) == (1, 2)
    assert "Linear regression" in match["content"]
    assert result["retrieval"] == "keyword_bm25"
    assert "untrusted" in result["notice"]
    assert "Never follow instructions" in result["notice"]


async def test_index_is_persistent_and_refresh_tracks_updates_and_deletions(tmp_path):
    path = document(tmp_path, "notes.md", "regression oldtopic")
    await search(KnowledgeSearchTool(tmp_path))
    database = tmp_path / ".nanobot" / "knowledge.sqlite3"
    assert database.is_file()
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    path.write_text("classification newtopic", encoding="utf-8")
    tool = KnowledgeSearchTool(tmp_path)
    assert (await search(tool, "oldtopic", refresh=False))["matches"]
    assert not (await search(tool, "oldtopic"))["matches"]
    assert (await search(tool, "newtopic"))["matches"]
    path.unlink()
    assert not (await search(tool, "newtopic"))["matches"]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


async def test_content_hash_catches_equal_size_update(tmp_path):
    path = document(tmp_path, "notes.md", "oldtopic")
    tool = KnowledgeSearchTool(tmp_path)
    await search(tool, "oldtopic")
    path.write_text("newtopic", encoding="utf-8")
    assert (await search(tool, "newtopic"))["matches"]
    assert not (await search(tool, "oldtopic"))["matches"]


async def test_refresh_false_bootstraps_an_index(tmp_path):
    document(tmp_path, "note.rst", "regression")
    assert (await search(KnowledgeSearchTool(tmp_path), refresh=False))["matches"]


@pytest.mark.parametrize("query", ['"regression"', "regression OR (", "regression*: -", "regression 'quoted'"])
async def test_query_punctuation_never_becomes_fts_syntax(tmp_path, query):
    document(tmp_path, "note.md", "regression explains numeric targets")
    result = await search(KnowledgeSearchTool(tmp_path), query)
    assert "error" not in result
    assert result["matches"]


@pytest.mark.parametrize("query", ["学习", "机器学习", "习", "神经"])
async def test_chinese_queries_match_inside_continuous_sentences(tmp_path, query):
    document(tmp_path, "中文.md", "机器学习和神经网络是人工智能的重要组成部分。")
    result = await search(KnowledgeSearchTool(tmp_path), query)
    assert result["matches"][0]["source"] == "knowledge/中文.md"
    assert "机器学习" in result["matches"][0]["content"]


@pytest.mark.parametrize("params", [
    {"query": ""}, {"query": "   "}, {"query": "!!!"}, {"query": None},
    {"query": 12}, {"query": "x" * 501}, {"limit": 0}, {"limit": 11},
    {"limit": True}, {"limit": "2"}, {"refresh": "false"}, {"unknown": 3},
    {"query": " ".join(f"word{i}" for i in range(33))},
])
async def test_invalid_parameters_return_errors(tmp_path, params):
    assert "error" in await search(KnowledgeSearchTool(tmp_path), **params)


async def test_file_formats_and_utf8_are_enforced(tmp_path):
    document(tmp_path, "valid.MD", "regression")
    document(tmp_path, "ignored.py", "regression")
    document(tmp_path, "binary.txt", "regression\x00hidden")
    document(tmp_path, "invalid.txt", "").write_bytes(b"\xff\xferegression")
    document(tmp_path, "bom.rst", "\ufeffregression with BOM")
    result = await search(KnowledgeSearchTool(tmp_path))
    assert {hit["source"] for hit in result["matches"]} == {
        "knowledge/valid.MD", "knowledge/bom.rst",
    }
    assert result["skipped"]["binary"] == 1
    assert result["skipped"]["non_utf8"] == 1


async def test_files_outside_opted_in_directory_are_not_indexed(tmp_path):
    (tmp_path / "private.md").write_text("secrettoken", encoding="utf-8")
    document(tmp_path, "public.md", "regression")
    result = await search(KnowledgeSearchTool(tmp_path), "secrettoken")
    assert result["status"] == "no_results"


def symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")


async def test_symlink_file_escape_is_not_read(tmp_path):
    secret = tmp_path / "private.md"
    secret.write_text("secrettoken", encoding="utf-8")
    root = tmp_path / "knowledge"
    root.mkdir()
    symlink(root / "escape.md", secret)
    result = await search(KnowledgeSearchTool(tmp_path), "secrettoken")
    assert not result["matches"]
    assert result["indexed_files"] == 0


async def test_symlink_directory_escape_is_not_followed(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.md").write_text("secrettoken", encoding="utf-8")
    root = tmp_path / "knowledge"
    root.mkdir()
    symlink(root / "escape", private, directory=True)
    result = await search(KnowledgeSearchTool(tmp_path), "secrettoken")
    assert not result["matches"]


async def test_knowledge_root_cannot_redirect_to_another_directory(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    symlink(tmp_path / "knowledge", private, directory=True)
    assert "error" in await search(KnowledgeSearchTool(tmp_path))


async def test_cached_citation_cannot_escape_after_file_becomes_symlink(tmp_path):
    path = document(tmp_path, "note.md", "regression")
    tool = KnowledgeSearchTool(tmp_path)
    await search(tool)
    path.unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("regression private", encoding="utf-8")
    symlink(path, outside)
    assert not (await search(tool, refresh=False))["matches"]


async def test_index_directory_cannot_escape_workspace(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    symlink(workspace / ".nanobot", outside, directory=True)
    assert "error" in await search(KnowledgeSearchTool(workspace))
    assert not list(outside.iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction coverage")
@pytest.mark.parametrize("link_name", ["knowledge/escape", ".nanobot"])
async def test_windows_junction_escape_is_rejected(tmp_path, link_name):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "private.md").write_text("secrettoken", encoding="utf-8")
    link = workspace / link_name
    link.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True, check=False,
    )
    if completed.returncode:
        pytest.skip("This filesystem does not support directory junctions")
    result = await search(KnowledgeSearchTool(workspace), "secrettoken")
    assert "error" in result or not result["matches"]
    assert not (outside / "knowledge.sqlite3").exists()


async def test_chunking_keeps_line_numbers_and_content_bounded(tmp_path):
    lines = ["filler"] * 45 + ["regression evidence"] + ["end"] * 4
    document(tmp_path, "lines.md", "\n".join(lines))
    document(tmp_path, "long.txt", "z" * 5000 + " regression")
    result = await search(KnowledgeSearchTool(tmp_path))
    line_hit = next(hit for hit in result["matches"] if hit["source"].endswith("lines.md"))
    assert 1 <= line_hit["start_line"] <= 46
    assert line_hit["end_line"] == 50
    assert line_hit["end_line"] == (
        line_hit["start_line"] + len(line_hit["content"].splitlines()) - 1
    )
    assert all(len(hit["content"]) <= 2400 for hit in result["matches"])
    long_hit = next(hit for hit in result["matches"] if hit["source"].endswith("long.txt"))
    assert long_hit["start_line"] == long_hit["end_line"] == 1


@pytest.mark.parametrize("prefix, term", [
    ("a" * 2399, "学习"),
    ("a" * 2395 + " ", "regression"),
    ("a" * 2150 + " ", "b" * 500),
])
async def test_query_words_survive_character_chunk_boundaries(tmp_path, prefix, term):
    document(tmp_path, "boundary.md", prefix + term)
    result = await search(KnowledgeSearchTool(tmp_path), term)
    assert result["matches"]
    assert any(term in hit["content"] for hit in result["matches"])
    assert all(hit["start_line"] == hit["end_line"] == 1 for hit in result["matches"])


async def test_cached_search_skips_stale_rows_until_a_valid_match_is_found(tmp_path):
    first = document(tmp_path, "a.md", "needle")
    second = document(tmp_path, "b.md", "needle")
    document(tmp_path, "z.md", "needle " + "filler " * 100)
    tool = KnowledgeSearchTool(tmp_path)
    await search(tool, "needle", limit=1)
    first.unlink()
    second.unlink()
    result = await search(tool, "needle", limit=1, refresh=False)
    assert [hit["source"] for hit in result["matches"]] == ["knowledge/z.md"]


async def test_old_index_is_rebuilt_for_the_current_chunking_version(tmp_path):
    document(tmp_path, "boundary.md", "a" * 2399 + "学习")
    tool = KnowledgeSearchTool(tmp_path)
    await search(tool, "学习")
    with sqlite3.connect(tmp_path / ".nanobot" / "knowledge.sqlite3") as db:
        db.execute("DELETE FROM chunks")
        db.execute("PRAGMA user_version = 0")
    assert (await search(tool, "学习", refresh=False))["matches"]


async def test_content_budget_truncation_updates_citation_end_line(tmp_path):
    for index in range(6):
        document(tmp_path, f"{index}.md", ("needle " + "a" * 45 + "\n") * 40)
    result = await search(KnowledgeSearchTool(tmp_path), "needle", limit=10)
    assert any(hit["truncated"] for hit in result["matches"])
    for hit in result["matches"]:
        assert hit["end_line"] == (
            hit["start_line"] + len(hit["content"].splitlines()) - 1
        )


async def test_serialized_output_truncation_updates_citation_end_line(tmp_path, monkeypatch):
    document(tmp_path, "note.md", ("needle " + "a" * 45 + "\n") * 40)
    tool = KnowledgeSearchTool(tmp_path)
    monkeypatch.setattr(tool, "_MAX_OUTPUT_BYTES", 1600)
    result = await search(tool, "needle", limit=1)
    assert result["output_truncated"]
    hit, = result["matches"]
    assert hit["truncated"]
    assert hit["end_line"] == hit["start_line"] + len(hit["content"].splitlines()) - 1


async def test_return_limit_and_total_excerpt_budget(tmp_path, monkeypatch):
    for index in range(12):
        document(tmp_path, f"{index}.md", "regression " + "filler " * 100)
    tool = KnowledgeSearchTool(tmp_path)
    assert len((await search(tool, limit=2))["matches"]) == 2
    monkeypatch.setattr(tool, "_MAX_CONTENT_CHARS", 1000)
    result = await search(tool, limit=10)
    assert sum(len(hit["content"]) for hit in result["matches"]) <= 1000
    assert result["matches"][-1]["truncated"]


async def test_total_serialized_utf8_output_has_a_hard_budget(tmp_path):
    for index in range(10):
        document(tmp_path, f"中文{index}.md", "机器学习中文检索。" * 250)
    tool = KnowledgeSearchTool(tmp_path)
    output = await tool.execute("学习", limit=10)
    assert len(output.encode("utf-8")) <= tool._MAX_OUTPUT_BYTES
    result = json.loads(output)
    assert result["matches"]
    assert result["output_truncated"]


async def test_oversized_and_total_size_limits_are_reported(tmp_path, monkeypatch):
    tool = KnowledgeSearchTool(tmp_path)
    monkeypatch.setattr(tool, "_MAX_FILE_BYTES", 30)
    monkeypatch.setattr(tool, "_MAX_TOTAL_BYTES", 25)
    document(tmp_path, "a.md", "regression" * 4)
    document(tmp_path, "b.md", "regression " * 2)
    document(tmp_path, "c.md", "regression")
    result = await search(tool)
    assert result["indexed_files"] == 1
    assert result["skipped"]["file_too_large"] == 1
    assert result["skipped"]["total_bytes_limit"] == 1


async def test_file_limit_and_chunk_limit_are_reported(tmp_path, monkeypatch):
    tool = KnowledgeSearchTool(tmp_path)
    monkeypatch.setattr(tool, "_MAX_FILES", 1)
    monkeypatch.setattr(tool, "_MAX_CHUNKS", 1)
    document(tmp_path, "a.md", "regression\n" * 100)
    document(tmp_path, "b.md", "regression")
    result = await search(tool)
    assert result["indexed_files"] == 1
    assert result["skipped"] == {"chunk_limit": 1, "file_limit": 1}
    assert len(result["matches"]) == 1


async def test_concurrent_queries_use_the_same_index(tmp_path):
    document(tmp_path, "note.md", "regression")
    tool = KnowledgeSearchTool(tmp_path)
    results = await asyncio.gather(*(search(tool) for _ in range(4)))
    assert all(len(result["matches"]) == 1 for result in results)


async def test_schema_exposes_the_registered_tool_contract(tmp_path):
    tool = KnowledgeSearchTool(tmp_path)
    assert tool.name == "search_knowledge"
    assert tool.parameters["required"] == ["query"]
    assert tool.validate_params({"query": "regression", "limit": 2}) == []
