# MCP Server

Built on the open [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), Ariadne's memory graph is exposed as a server callable by any MCP client. See the [MCP Specification](https://spec.modelcontextprotocol.io/) for protocol details.

### Transport modes

| Mode | Usage | Use case |
|------|-------|----------|
| **stdio** (default) | `ariadne-mcp` | clients that spawn a local process, e.g. Claude Desktop |
| **SSE** | `ariadne-mcp --sse --port 8766` | clients connecting via a network URL, e.g. Cursor |

> `--yaml` / `--graph-dir` are both optional: which graph to use comes first from the shared pointer, falling back to `--yaml`, then to the empty placeholder `sample_graph.yaml` inside the graph directory.

> ⚠️ The SSE default port is now `8766` (`ariadne-api` stays on `8765`), so they no longer clash.

### Environment variables (optional)

Besides CLI flags, LLM/Embedding config can live in the project root `.env` (or system env vars) and is loaded at startup; **CLI flags take precedence**.

| CLI flag | Environment variable |
|----------|---------------------|
| `--llm-model` | `OPENAI_MODEL` |
| `--llm-api-key` | `OPENAI_API_KEY` |
| `--llm-base-url` | `OPENAI_API_BASE` |
| `--embedding-model` | `EMBEDDING_MODEL` |
| `--embedding-api-key` | `EMBEDDING_API_KEY` |
| `--embedding-base-url` | `EMBEDDING_API_BASE` |
| `--embedding-local` | `EMBEDDING_LOCAL` |

Maintenance batching (`ARIADNE_MCP_IDLE_TIMEOUT` / `MAX_BATCH_ROUNDS` / `PROACTIVE_*` / `AUTO_SKIP`),
`ARIADNE_MCP_TRIAGE`, source storage (`ARIADNE_SOURCE_STORE`) and idle review (`ARIADNE_REVIEW_*`)
are documented with their defaults in `.env.example` under "MCP 维护 / 检索开关" and "记忆保真".

```bash
# project root .env example
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_LOCAL=true
```

> With `EMBEDDING_LOCAL=true`, install local dependencies first (`pip install -e ".[local]"`); otherwise startup errors out.

### Extra startup flags

| Flag | Description |
|------|-------------|
| `--vector-index` | path to a FAISS index file (optional), to restore an existing vector index |
| `--restore-dir` | fully restore from a checkpoint directory (graph + vectors + builder + scheduler state) |

> Pairs with `dba_checkpoint`: save at runtime with `dba_checkpoint`, restore at startup with `--restore-dir`.

### Run mode

Only the **full DBA mode** is currently supported: an LLM must be configured (`--llm-model` or `OPENAI_MODEL`). On startup, `dba_add_conversation` invokes the internal LLM for memory maintenance: **node extraction (Step 1) and edge linking (Step 2) are two separate LLM calls** — nodes are extracted and applied first, then edges are linked using all of this round's new nodes plus related old nodes / one-hop neighbors / existing edges, avoiding the visibility limits of a single-pass output. Missing LLM / DBA dependencies cause an immediate error exit (no stub-mode fallback).

> The agent's LLM and Ariadne's internal maintenance LLM are **two separate models**: the agent's LLM understands intent and calls tools, while Ariadne's LLM turns conversations into graph facts. They share memory through the graph, not the context window.

### Embedding configurations

| Mode | Parameters | Description |
|------|-----------|-------------|
| **API** (default) | `--embedding-model` | calls any OpenAI-compatible `/v1/embeddings` endpoint |
| **Local** | `--embedding-model ... --embedding-local` | uses sentence-transformers, no network |
| **None** | omit `--embedding-model` | graph-only operations, no vector retrieval |

When `--embedding-base-url` / `--embedding-api-key` are omitted, they fall back to `--llm-base-url` / `--llm-api-key`.

```bash
# API embedding (OpenAI)
ariadne-mcp --yaml data.yaml --llm-model gpt-4o-mini --llm-api-key sk-xxx \
    --llm-base-url https://api.openai.com/v1 --embedding-model text-embedding-3-small

# Local embedding
ariadne-mcp --yaml data.yaml --llm-model gpt-4o-mini --llm-api-key sk-xxx \
    --embedding-model BAAI/bge-m3 --embedding-local
```

### 10 tools

| Tool | Description |
|------|-------------|
| `dba_add_conversation` | append a conversation; queued in the scheduler first and maintained asynchronously once a threshold is reached |
| `dba_query_memory` | purpose-driven associative retrieval (PAR pipeline: Jump Axis + Purpose Regression + Peak Finding) |
| `dba_query_knowledge` | literal keyword screening over **all active nodes**; results contain no edges |
| `dba_temporal_lookup` | standalone single-hop temporal query (only "time → event", reverse direction) for "what happened at a given time" |
| `dba_inspect_graph` | expand a node's 1-hop neighbors |
| `dba_intervene` | manually CRUD nodes and edges |
| `dba_checkpoint` | save a full checkpoint |
| `dba_get_stats` | graph statistics (including the raw-fallback count) |
| `dba_review_graph` | graph health check (read-only): reports likely duplicate pairs, isolated nodes and pending raw fallbacks, never edits the graph |
| `dba_review_sources` | source tracing (read-only): audits **node** extraction against the original conversations; edges are not audited (fix them by re-linking/deleting). Source store is on by default; `ARIADNE_SOURCE_STORE=0` disables |

### Keyword lookup (`dba_query_knowledge`)

- The index covers **all active nodes** (extracted nodes and raw fallbacks included); it is not a separate knowledge base.
- `dba_query_knowledge(query, top_k?)`
  - keyword matching: ASCII tokens + CJK character bigrams;
  - returns `node_id / text / score`, **without edges**;
  - no threshold and no refusal; an empty list simply means no keyword hit.
- `ARIADNE_MCP_RERANK=1` enables reranking for keyword results. It is independent of `ALM_RERANK`; failures fall back to keyword order.
- Historical data may still contain `metadata.channel="note"` nodes (early project notes); the panel can filter by that flag, but the write entry has been removed.

### Retrieval details (`dba_query_memory`)

- Runs the full PAR pipeline: Jump Axis directed expansion + Purpose Regression filtering + Peak Finding termination.
- Reranking and compression happen inside StoryRank, which returns a **single** narrative and never truncates by count. `total_matched` is the true PAR candidate count; `returned` is the number of story segments (always 0 or 1).
- Deprecated/forgotten nodes are filtered out and never appear in results.

### Agent integration

**Claude Desktop (stdio)**:

```json
{
  "mcpServers": {
    "ariadne": {
      "command": "python",
      "args": ["-m", "dba_pipeline.mcp_server", "--yaml", "/path/to/memory_graph.yaml"],
      "cwd": "/path/to/dba_pipeline"
    }
  }
}
```

**Cursor / SSE clients** (start the server first, then set the URL):

```bash
# Recommended: one-command start (panel 8765 + MCP 8766), config read from .env
python start_all.py --yaml /path/to/memory_graph.yaml

# Or start SSE alone (config also read from .env, overridable via CLI)
ariadne-mcp --yaml /path/to/memory_graph.yaml --sse --port 8766
```

```json
{
  "mcpServers": {
    "ariadne": {
      "url": "http://127.0.0.1:8766/sse",
      "transport": "sse",
      "headers": { "Authorization": "Bearer <ARIADNE_MCP_TOKEN>" }
    }
  }
}
```

> The SSE endpoint always requires auth. Without an `Authorization` header it returns
> `401 {"error": "未登录或凭据无效"}`. Clients that support `headers` (e.g. AstrBot) can use the
> config above; clients that only accept a URL need another transport, or fall back to stdio.

### Remote access and security

`--sse` listens on `127.0.0.1:8766` by default (override with `--host`) and **always requires auth**:

- **Same machine**: `http://127.0.0.1:8766/sse` with HTTP Basic, or a Bearer token from `ARIADNE_MCP_TOKEN`;
- **Another machine**: prefer an SSH tunnel (`ssh -L 8766:127.0.0.1:8766 <user>@<host>`); the client still uses the local address, so the plaintext port never leaves the host;
- **Direct public access**: terminate TLS at a front proxy first (this service does not do TLS), then set `--host 0.0.0.0` and restrict the source;
- In containers the host port binds to loopback by default (`ARIADNE_BIND_HOST`) — see "Public exposure and remote access" in the README.

Failed tool calls return `isError: true` with only an `error_id` in the body (the full traceback goes to the server log and operation log), so debugging stays possible without leaking internals.
