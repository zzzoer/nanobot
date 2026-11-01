# Grounded Nanobot

**A nanobot fork for asking questions about local documents, keeping named conversations, and selecting relevant tools.**

Grounded Nanobot extends [HKUDS/nanobot](https://github.com/HKUDS/nanobot).
Upstream provides the agent runtime, model providers, tool execution, memory,
channels, Python SDK, and OpenAI-compatible API. This fork adds bounded local
document retrieval with citations, persistent conversation selection, and
conservative keyword-based tool routing. It is an independent extension, not
an official upstream release.

The tested extension is published on the default `grounded` branch, based on
upstream revision `63d646f`. The separate `main` branch retains a newer upstream
snapshot; this project does not claim compatibility with that later refactor.

“Grounded” describes the retrieval workflow: the agent can consult source text
before answering. It does not guarantee factual answers or correct citations.

## What this fork adds

| Feature | Behavior |
|---|---|
| Local document retrieval | Search opted-in UTF-8 Markdown, text, and reStructuredText files with SQLite FTS5/BM25; return excerpts and file/line citations |
| Chinese text search | Index Chinese characters and overlapping two-character terms without an external tokenizer |
| Persistent named conversations | Create, select, list, inspect, and reset conversations within the current chat or topic; restore the selection after restart |
| Adaptive tool routing | Filter tool definitions using intent keywords and recent successful calls; retain unknown/MCP tools and file-operation dependencies |
| Routing diagnostics | Optional metadata-only JSONL records; disabled by default, without raw user-message text |
| Regression coverage | Exercise retrieval, cache updates, path boundaries, routing, and conversation selection through unit and fake-provider integration tests |

Retrieval runs locally and needs no embedding model or vector database.
Answer generation uses the configured language model. When that provider is
remote, retrieved excerpts leave your machine as ordinary tool results.

## Get started

Use Python 3.11 or newer. Install this checkout to get the fork's changes:

```shell
git clone https://github.com/zzzoer/grounded-nanobot.git
cd grounded-nanobot
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```shell
# macOS / Linux
source .venv/bin/activate
```

```shell
python -m pip install -e ".[dev]"
nanobot onboard
```

The command remains `nanobot`, and the Python package remains `nanobot`.
The upstream `nanobot-ai` PyPI package does not include this fork's changes.

### Configure a model

Run `nanobot onboard --wizard` for interactive provider setup, or edit
`~/.nanobot/config.json`. Merge the following fields into your existing
configuration, replacing both placeholders with your provider credentials and
an available model ID that supports tool calling:

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "YOUR_OPENROUTER_API_KEY"
    }
  },
  "agents": {
    "defaults": {
      "provider": "openrouter",
      "model": "YOUR_PROVIDER_MODEL_ID",
      "workspace": "~/.nanobot/workspace"
    }
  }
}
```

Keep credentials in your local configuration. Provider availability and usage
charges depend on your account. Other supported providers and local model
servers are described in the [preserved upstream guide](docs/upstream-readme.md#providers).

```shell
nanobot agent
```

For a separate runtime workspace, initialize and launch with the same override:

```shell
nanobot onboard --workspace ./my-workspace
nanobot agent --workspace ./my-workspace
```

## Ask questions about your documents

Put reference files in the **runtime workspace**, normally
`~/.nanobot/workspace/knowledge/`, rather than the repository checkout:

```text
~/.nanobot/workspace/
  knowledge/
    experiment-notes.md
    reference/
      evaluation-protocol.txt
```

Example document:

```markdown
# Evaluation protocol
Use a held-out test set only after model selection is complete.
Record the random seed, dataset revision, and hyperparameters for each run.
```

Ask in the agent chat:

```text
Search my knowledge base for the evaluation protocol. Explain the steps and cite the source lines.
```

The model can call `search_knowledge(query, limit=5, refresh=True)`. Results
include a relative source such as `knowledge/experiment-notes.md`, start/end
line numbers, the original excerpt, and a BM25 score. The system prompt asks
the model to cite sources and acknowledge missing evidence.

The index is created lazily at
`<workspace>/.nanobot/knowledge.sqlite3`. By default, each search scans for
changed and deleted files using content hashes. `refresh=False` reuses cached
text, so file edits can remain stale until the next refreshed search.

This is keyword retrieval, not embedding-based semantic search. It supports
`.md`, `.txt`, and `.rst`; convert PDFs, Word files, images, or scanned pages to
supported text first. Excerpts are untrusted reference data, not instructions.

## Keep separate conversations

Request session actions in natural language, for example:

```text
Create a conversation named research-notes.
List the conversations in this chat.
Switch to research-notes.
Which conversation is active?
Return to the default conversation without deleting my history.
```

The `session_manage` tool exposes `create`, `switch`, `list`, `current`, and
`reset`. Selection changes apply to the **next user message**; the request
and confirmation stay in the original conversation. Reset selects the default
conversation and preserves all histories.

Named sessions belong to the current chat/topic, use keys such as
`telegram:123:session:research-notes`, and persist selection across restarts.
Unicode names are supported. API/SDK callers need a scoped `channel:chat`
session key for session management; arbitrary unscoped keys still retain
ordinary history. Background system/subagent results continue to use their
existing default-conversation routing, rather than following named sessions.

## Control tool routing

Routing is enabled by default. Merge this into your config to disable it:

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

When enabled, English keywords use word boundaries: `ls` does not match
`false`. Chinese keywords use substring matching. Unknown/MCP tools remain
available; file writes/edits retain read/list dependencies; knowledge search
and session management stay discoverable. Requests without intent matches or
relevant tool history fall back to the complete tool registry.

Set `logDecisions` to `true` to write
`<workspace>/tool_routing_log.jsonl`. These records contain selection counts,
tool names, and routing decisions, without the original user message.
Selection frequency is not a tool-execution count. No token, latency, or
answer-quality improvement is claimed without a measured comparison.

## Validation

The local regression checkpoint recorded **495 passing tests, 5 skipped tests,
and 32 passing subtests** on Windows with Python 3.14. This includes inherited
upstream tests as well as the fork's additions; it is not a claim of 495 newly
authored tests. The four hosted checks passed on Linux Python 3.11/3.12/3.13
and Windows Python 3.12 at the feature checkpoint.

On November 1, 2026, a real-provider smoke test ran on the Windows host using
OpenRouter's `deepseek/deepseek-chat` model. **All eight checks passed**:

- Chinese document retrieval, factual response, and source citation.
- Abstention when the knowledge base had no evidence.
- Refresh after a document edit and removal after deletion.
- Model-driven creation of a Unicode-named conversation.
- Next-turn history isolation in the selected conversation.
- Recovery of the selected history in a new CLI operating-system process.
- Returning to the default conversation while retaining the named history.

The eight scenarios are recorded separately in the
[sanitized evidence report](docs/live-validation-2026-11-01.json). They used
synthetic documents and real model requests, not a fake provider. This smoke
test establishes end-to-end feasibility for that setup; it is not an answer
quality benchmark or a guarantee about other models.

To repeat it with your own configured provider (makes real model requests):

```shell
python scripts/live_validate.py --config ~/.nanobot/config.json --output live-validation.json
```

Reproduce the focused feature checks:

```shell
python -m pytest tests/agent/test_tool_router.py tests/agent/test_knowledge_integration.py tests/agent/test_loop_session_routing.py tests/tools/test_session_manage.py tests/tools/test_knowledge_tool.py
```

Run the broader suite and inspect the dedicated workflow:

```shell
python -m pytest tests/agent tests/tools tests/config tests/cli tests/test_nanobot_facade.py tests/test_openai_api.py -q
```

[Feature CI](https://github.com/zzzoer/grounded-nanobot/actions/workflows/knowledge-sessions.yml)
and the [upstream CI configuration](.github/workflows/ci.yml) are included.
Deterministic regression tests use a fake provider with the real runner,
retriever, and session store; the opt-in live validation above uses the real
provider. Some path/symlink tests require OS permissions and may be skipped.

## Limits and design choices

- Retrieval is lexical: synonyms and paraphrases may be missed, and overlapping chunks can return excerpts from the same document.
- Default bounds are 500 documents, 1 MiB per file, 32 MiB per scan, 10,000 chunks, up to 10 excerpts, and 16,000 UTF-8 bytes per result JSON. Skips and truncation are reported.
- Python's SQLite build must include FTS5. The index stores source text and should have the same access controls as the documents.
- Named sessions isolate conversation histories, not runtime workspaces or long-term memory files. They are not an access-control boundary for tools.
- Routing is a keyword heuristic. Disable it if tool selection is unsuitable for a workflow. Tools remain subject to the existing runtime's permissions and configuration.
- Document instructions can be malicious. The prompt marks excerpts as untrusted; this does not provide a complete prompt-injection defense or a sandbox.

See [local knowledge and sessions](docs/local-knowledge-and-sessions.md) for
the detailed behavior, [Python SDK](docs/PYTHON_SDK.md) for embedding the agent,
and the [preserved upstream README](docs/upstream-readme.md) for inherited
channels, API, deployment, and provider documentation.

## Attribution and license

This project is a derivative of [HKUDS/nanobot](https://github.com/HKUDS/nanobot).
The original README is preserved in [docs/upstream-readme.md](docs/upstream-readme.md),
with relative links adjusted for its new location. Upstream authors and
contributors retain credit for the original implementation. This fork's
contributions are the local retrieval, named-session, routing, integration,
documentation, and associated regression changes described above.

Distributed under the [MIT license](LICENSE). The original copyright and
license notice are retained. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
proposing changes upstream.
