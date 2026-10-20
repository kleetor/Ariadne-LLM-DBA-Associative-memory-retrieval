# ALM Synchronous HTTP Memory Service

Besides the MCP form, the repository also ships a **synchronous HTTP memory service for external callers**
(`alm/`: Add / Search / Health), suited to "an external program writes and retrieves memory in bulk under a
fixed contract".

### Differences from the MCP version

| Aspect | MCP version | ALM version |
| --- | --- | --- |
| Interface | MCP protocol (stdio / SSE), exposes 9 tools to the Agent | Synchronous HTTP: `POST /add`, `POST /search`, `GET /health` |
| Caller | Agent runtime (calls tools on demand during a conversation) | External program / service (calls directly under the contract) |
| Memory organization | A single graph, switched via a shared pointer | **One isolated space per `user_id`** (directory isolation + LRU eviction) |
| Write strategy | Everything is built into the graph | **Route by content shape + decide per space whether to build a graph** (see below) |
| Auth | Panel / MCP credentials (`auth.json` + Cookie / Basic / Bearer) | `ALM_API_KEYS` (`X-Api-Key` or Bearer); **refuses to start** when bound to a non-loopback address with no keys |
| Visualization | 3D panel + offline HTML | None (pure API) |
| Dependencies | LLM + Embedding | LLM + Embedding (both API mode) |

Both **share the same core pipeline** (extraction / storage / retrieval / output), and ALM only swaps the
"access layer" for a synchronous HTTP contract plus per-`user_id` space isolation with an LRU cache.

### Two-Layer Architecture: ALM Does *Not* Build a Graph for Everything

This is the **most easily confused** point versus the MCP version. The MCP version serves a single Agent session
and **builds a graph for everything** by default. ALM, however, takes **heterogeneous corpora** pushed in bulk by
external callers (dialogue mixed with long documents), and empirically "graph everything" is a **net negative** —
so it splits memory into **two layers** and only builds a graph where it pays off.

**What the two layers are**

| Layer | Content | Origin | LLM cost |
| --- | --- | --- | --- |
| **Raw layer (L1)** | Original messages + long-document chunks | Written directly by the caller | **Zero** |
| **Graph layer (L2)** | Extracted typed nodes and edges (6 node roles × 8 relation types) | **Derived** from the raw layer | Used during extraction / linking |

The raw layer is always kept and is the **authoritative content**; the graph layer is just a derived view of it.

**Two-stage routing on write**

1. **Route by content shape** (at the `/add` entry point — only counts and lengths, zero LLM): **documents**
   (long text) → the **document channel**, chunked and embedded into a document store **fully isolated** from the
   graph, **never entering the graph**; **conversations** → the extraction pipeline, producing nodes and edges;
   **noise** → dropped.
2. **Decide per space whether to build a graph** (default `ALM_FORCE_MODE=auto`): a new space starts in
   **`vector`** — zero LLM on the write side, content landing in the raw layer first; on the first 2 questions the
   space's attribution is **decided once and then frozen**; if it resolves to **`graph`** and backfill is enabled
   (`ALM_BACKFILL=1`), the content already in the raw layer is **backfilled into the graph** in the background;
   if it stays `vector`, nothing is graphed for that space.

**Read side**: a `vector` space uses pure-vector recall (optionally with a cross-encoder rerank) or document-channel
retrieval; a `graph` space uses graph expansion + purpose inference and renders as `story` / `evidence`.

> **Why the split is necessary**: the write side (extraction + graph building) is by far the dominant LLM cost
> (measured at roughly 90%). Running node/edge extraction on "knowledge-base-style long documents" is a **category
> error** — extraction yields nothing, the raw text is chopped into hundreds of isolated nodes dumped into the graph
> (with exactly zero edges), and those isolated blocks then get picked as memory seeds, degrading graph expansion to
> zero hops. After the split: no graph means **zero LLM on the write side**, while the raw text is still retrievable
> by vector — **nothing is lost**.

### Deployment

```bash
# 1) Create the ALM-specific config (fully isolated from the repository-root .env)
#    The template lists the required entries (ALM_API_KEYS / LLM / Embedding); add optional ones as needed
cp alm/.env.alm.example alm/.env.alm && vim alm/.env.alm

# 2) Build and start (run from the repository root)
docker compose --env-file alm/.env.alm -f deploy/docker-compose.alm.yml up -d --build

# 3) Read-only self-check (writes no memory)
python -m alm.status --base-url http://127.0.0.1:8770 --api-key <your-key>
```

- Listens on `127.0.0.1:8770` by default; public deployment requires a reverse proxy terminating TLS (the service
  itself does not do TLS).
- Memory is persisted in the named volume `alm-data` (container `/data`) — surviving restarts and image upgrades.
- The repository **does not ship** the image tarball (too large); just build it on site with `--build` above. To
  distribute a prebuilt image, upload the `docker save` output as a **GitHub Release asset** instead of committing it.
- Health check, cleanup and load testing: `GET /health`, `python -m alm.cleanup`, `python -m alm.loadtest`.
