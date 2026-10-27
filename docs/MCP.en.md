# MCP Server

Built on the open [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), Ariadne's memory graph is exposed as a server callable by any MCP client. See the [MCP Specification](https://spec.modelcontextprotocol.io/) for protocol details.

### Transport modes

| Mode | Usage | Use case |
|------|-------|----------|
| **stdio** (default) | `ariadne-mcp` | clients that spawn a local process, e.g. Claude Desktop |
| **SSE** | `ariadne-mcp --sse --port 8766` | clients connecting via a network URL, e.g. Cursor |

> `--yaml` / `--graph-dir` are both optional: which graph to use comes first from the shared pointer, falling back to `--yaml`, then to the empty placeholder `sample_graph.yaml` inside the graph directory.

> ⚠️ The SSE default port `8765` clashes with `ariadne-api`; change it (e.g. `--port 8766`) when running both. The one-command start (`start_all.py`) auto-separates to 8766.

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

```bash
# project root .env example
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
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
    --embedding-model BAAI/bge-large-zh-v1.5 --embedding-local
```

### 11 tools

| Tool | Description |
|------|-------------|
| `dba_add_conversation` | append a conversation; queued in the scheduler first and maintained asynchronously once a threshold is reached |
| `dba_query_memory` | purpose-driven associative retrieval (PAR pipeline: Jump Axis + Purpose Regression + Peak Finding) |
| `dba_add_note` | write a project note as an ordinary graph node (it can also be recalled by the memory layer) |
| `dba_query_knowledge` | keyword-exact lookup over project facts/conventions/APIs/error codes; results contain no edges |
| `dba_temporal_lookup` | standalone single-hop temporal query (only "time → event", reverse direction) for "what happened at a given time" |
| `dba_inspect_graph` | expand a node's 1-hop neighbors |
| `dba_intervene` | manually CRUD nodes and edges |
| `dba_checkpoint` | save a full checkpoint |
| `dba_get_stats` | graph statistics (including the note count) |
| `dba_review_graph` | graph health check (read-only): reports likely duplicate pairs and isolated nodes, never edits the graph |
| `dba_review_sources` | source tracing (read-only): checks extraction quality against the original conversations; requires `ARIADNE_SOURCE_STORE=1` |

### Notes and knowledge lookup (`dba_add_note` / `dba_query_knowledge`)

- A note is an ordinary node: `metadata.channel="note"`, indexed by the vector store, and usable as a PAR seed. It is not a separate store.
- `dba_add_note(title, text, observed_at?, source_hint?, fidelity?, tags?)`
  - `title` is required; `text` must be ≤ 4000 chars, otherwise it is rejected and you should split it;
  - `tags` express project ownership (e.g. `project:alpha`); there is no server-side project filter;
  - `source_hint` is a hint only; the server never fetches it;
  - identical `title + text + tags` returns the existing node (`duplicate=true`).
- `dba_query_knowledge(query, top_k?)`
  - keyword matching: ASCII tokens + CJK character bigrams; `title/tags/text` weights are 3/2/1;
  - returns `node_id / text / score / observed_at / source_hint / fidelity`, **without edges**;
  - no threshold and no refusal; an empty list simply means no keyword hit.
- `ARIADNE_MCP_RERANK=1` enables reranking for the knowledge layer. It is independent of `ALM_RERANK`; failures fall back to keyword order.

### Retrieval details (`dba_query_memory`)

- Runs the full PAR pipeline: Jump Axis directed expansion + Purpose Regression filtering + Peak Finding termination.
- `rerank_k` caps the number of returned items (rank-k, default 20); `total_matched` is the true candidate count and `returned` is the number actually returned. If `total_matched > returned`, raise `rerank_k` to retrieve the remaining memories.
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
    "ariadne": { "url": "http://127.0.0.1:8766/sse" }
  }
}
```
