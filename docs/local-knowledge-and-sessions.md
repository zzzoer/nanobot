# Local knowledge retrieval, sessions, and tool routing

## Ask questions about your documents

Place UTF-8 `.md`, `.txt`, or `.rst` documents under the configured nanobot
workspace's `knowledge/` directory. This is the runtime workspace, normally
`~/.nanobot/workspace`, not the repository checkout.

```text
workspace/
  knowledge/
    project-notes.md
    reference/
      experiment-guide.txt
```

Then ask nanobot, for example:

```text
根据知识库里的项目笔记，总结实验的复现步骤，并给出来源。
Search my knowledge base for the evaluation protocol and cite the relevant lines.
```

The agent can call `search_knowledge(query, limit=5, refresh=True)`. The tool
retrieves excerpts with a relative source path, starting and ending line numbers,
and a BM25 score. The existing configured model receives those excerpts and
generates the answer; the system prompt asks it to cite sources and state when
the documents do not support an answer. Retrieval and indexing run locally;
excerpts are still sent to your configured model provider as normal tool results.

This implementation uses keyword/BM25 retrieval, with Chinese characters and
overlapping two-character terms. It does not download embedding models, call
an additional API, or require a vector database. PDF, Word, images, and scanned
documents must first be converted to supported text formats. This is a bounded
local knowledge base rather than a semantic search service.

### Index updates and limits

The index is created lazily at `workspace/.nanobot/knowledge.sqlite3`. Searches
refresh changed and deleted files by content hash by default. `refresh=False`
reuses cached text, so edits can remain stale until the next refreshed search.
Retrieval ranks excerpts, not whole documents; overlapping excerpts may refer
to the same source.

The defaults bound local work and model context: 500 documents, 1 MiB per file,
32 MiB per scan, 10,000 chunks, at most 10 returned excerpts, and at most
16,000 UTF-8 bytes in the result JSON. Responses report skipped files and
truncation. Unsupported encodings, binary files, and paths escaping the
knowledge directory are not indexed. Python's SQLite build must support FTS5.

The SQLite file contains document text. Keep it with the same access controls
as the source knowledge base. Repository ignore rules exclude knowledge,
index, and routing-log files. Retrieved text is untrusted reference data;
instructions embedded in a document must not override the conversation.

## Named conversations within one chat

The `session_manage` tool supports:

| Action | Behavior |
|---|---|
| `create` | Create a named session, optionally using `title` or `seed`; activates it by default |
| `switch` | Select an existing named session in the current chat/topic |
| `list` | List sessions and titles belonging to the current scope |
| `current` | Show the active named session or the default conversation |
| `reset` | Return to the default conversation while preserving all histories |

For example, ask “新建一个叫论文阅读的会话” or “列出当前聊天的会话”.
Switches take effect on the **next message**. The message that requested the
switch and its response stay in the original conversation.
Background system/subagent messages keep their existing default-conversation
routing; they do not automatically follow a named-session selection.

Named keys use `channel:chat:session:name`, preserve Unicode names, and store
explicit ownership metadata. The selected session is persisted on the default
conversation and restored after restart. Chats, threads, and explicit API/SDK
session keys remain isolated. Legacy sessions without ownership metadata are
not automatically treated as named sessions. Reset does not delete messages.
Session management requires a scoped `channel:chat` key; arbitrary unscoped SDK
keys continue to hold history but cannot use the named-session tool.

## Adaptive tool routing

Routing is enabled by default and can be disabled in your existing config:

```json
{
  "tools": {
    "routing": {
      "enabled": false,
      "logDecisions": false
    }
  }
}
```

When enabled, explicit intent keywords and recent successful tool calls select
tool definitions for the model. Unknown/MCP tools remain available; read/write
dependencies are retained; unmatched requests fall back to all tools. Session
management and knowledge retrieval remain discoverable. English keyword
matching uses word boundaries to avoid matching `ls` inside `false`.

Routing diagnostics are off by default. Enabling `logDecisions` writes
`workspace/tool_routing_log.jsonl` containing counts, selected tool names, and
decision metadata; user-message text is not written. Routing frequency and
actual tool execution are different statistics. Tool-call history is saved
once through the normal assistant/tool protocol, without synthetic assistant
messages inserted into conversations.

## Reproduce the feature tests

```shell
python -m pip install -e ".[dev]"
python -m pytest tests/agent/test_tool_router.py tests/agent/test_knowledge_integration.py tests/agent/test_loop_session_routing.py tests/tools/test_session_manage.py tests/tools/test_knowledge_tool.py
```

The knowledge integration test drives the real runner, retriever, SQLite index,
and session store with a deterministic fake model provider. It verifies that
retrieved evidence reaches the model and citations can be returned; it does
not measure a live model's answer quality. Some symlink tests require OS
permissions and are skipped when the host does not grant them.
