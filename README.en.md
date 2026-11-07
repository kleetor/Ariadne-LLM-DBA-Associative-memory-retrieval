# Ariadne

**LLM DBA Management & Purpose-Driven Associative Memory Retrieval System**

Ariadne applies Database Administration (DBA) ideas — fact extraction, correction, deduplication, deprecation — to
LLM memory, modeling long-term memory as a **directed, typed knowledge graph** and implementing **purpose-driven
associative retrieval (PAR)** on top of it: results are not a flat candidate list but causal story fragments
organized by StoryRank.

```bash
pip install -c constraints.txt -e .
cp .env.example .env        # fill in your LLM / Embedding settings
python start_all.py         # panel 127.0.0.1:8765 + MCP SSE 127.0.0.1:8766
```

The documents under `docs/` are organized by topic and can be handed straight to an AI to grasp the whole system.
`data/` contains an empty `sample` dataset (the default landing spot on first start) and a pre-populated `memory`
dataset for quickly experiencing the system.

The retrieval weight table is tunable: different usage scenarios can adjust the base weights and related parameters to change retrieval behaviour (see [docs/architecture.en.md](docs/architecture.en.md)).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-AGPLv3-blue)](LICENSE)
![Version](https://img.shields.io/badge/Version-0.1.5-lightgrey)

> *"A thread through the labyrinth of memory."*

**Language**: [English](README.en.md) · [中文](README.md)

**Docs**: [Index](docs/README.md) · [ALM](docs/ALM.en.md) · [MCP](docs/MCP.en.md) · [Architecture](docs/architecture.en.md)

---

## Goals & Rationale

> Memory is not a searchable drive, but the thing that makes a person who they are.

Ariadne does not aim to be a better knowledge base. It aims to **give an Agent a memory of its own**: one that remembers who said what to it, how that felt, and who the other person is turning out to be.

### Why we narrowed the scope to "one person"

We started from Tulving's split between semantic and episodic memory, framing memory systems against knowledge bases: stable facts demand accuracy and consistency, while personal experiences allow context, causality and emotion. That layering holds up in engineering, but after building the ALM pipeline and several iterations of the MCP server we reached a plainer conclusion:

**How well a memory system works depends heavily on how well it matches the source data.** Enterprises, teams and individuals differ sharply in corpus density, factual stability and privacy boundaries. The same extraction and retrieval strategy produces very different results — and very different integration costs — across the three, and trying to cover all of them tends to leave each one shallow.

So Ariadne narrows its goal to one thing: **personal conversation and emotional companionship for a single user.**

### What PAR really is: an LLM-like weighting and prediction system

Ariadne's retrieval algorithm (PAR) and the engineering around it are, in essence, an **LLM-like weighting and prediction system**:

- The LLM first turns raw dialogue into **typed nodes** — categorized facts and experiences — and links them with typed relations;
- Node types × relation types form a **weight table** that adds a second dimension to vector recall. What it carries is the causal, situational and interpersonal structure the LLM has compressed from pretraining — priors that the embedding space itself does not encode;
- Guided by the query vector, retrieval does more than "recall nodes": it **predicts further relation-based recall** on top of what has already been retrieved, following causal and temporal chains to bring out what should come to mind next.

The payoff is twofold: the LLM can **model and understand a specific person quickly**, and the user can treat it as a **reliable companion** rather than an assistant that needs to be reintroduced every time.

### Engineering stance: give the memory back to the Agent

Ariadne holds one principle in its implementation: **memory belongs to no user and no session — it belongs to the Agent.**

The common pattern treats memory as a profile maintained *for* a user and filters retrieval by user_id. Ariadne lets the Agent own and maintain the graph itself: what it records is "the conversations I have lived through", not "data about a user". We see this as the more decisive step toward anthropomorphism — only when memory is *mine* can recall have a stance, continuity and character.

### Where this is heading

Ariadne hopes Agents will come to understand complex human emotion: through long-term companionship, to grasp a person's character, flaws and spirit, and to offer guidance and care where it is needed. That is the long road from "remembering facts" to "understanding a person".

---

## Table of Contents

- [Goals & Rationale](#goals--rationale)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Retrieval Method: PAR Baseline](#retrieval-method-par-baseline)
- [MCP Server](#mcp-server)
- [ALM Synchronous HTTP Service](#alm-synchronous-http-service)
- [Data Format and Type System](#data-format-and-type-system)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Overview

Ariadne applies **Database Administration (DBA)** concepts — fact extraction, error correction, deduplication, and deprecation — to LLM memory systems, modeling long-term memory as a **directed, typed knowledge graph**, on top of which it implements a **purpose-driven** associative memory retrieval pipeline (the "PAR" pipeline).

The core thesis is that the value of memory lies not in storing more, but in being recalled with the correct causal structure when needed. To that end, Ariadne provides:

- **Automated memory maintenance**: conversation logs are extracted into nodes and relations via batched, asynchronous DBA scheduling, with continuous correction and deprecation of stale memories;
- **Purpose-driven retrieval**: a three-stage mechanism (Jump Axis, Purpose Regression, Peak Finding) performs bounded expansion along causal/semantic directions in the graph;
- **Story-based output**: retrieval results are organized by StoryRank into story-fragment documents rather than flat candidate lists;
- **Multi-endpoint access**: a 3D visualization panel, an [MCP](https://modelcontextprotocol.io/) server for LLM agents, and an offline HTML export.

## Key Features

| Feature | Description |
|---------|-------------|
| 🧠 Typed knowledge graph | 6 node roles × 8 relation types, with directed, weighted edges |
| 🔧 Automated DBA maintenance | node extraction and edge linking as two separate steps, correction/deprecation + batched async scheduling to cut tokens |
| 🎯 Purpose-driven retrieval | Jump Axis + Purpose Regression + Peak Finding, replacing fixed top-K |
| 📖 StoryRank narrativization | causal chains → story fragments, avoiding context pollution for the chat LLM |
| 🔌 MCP integration | 11 tools (including note-taking and knowledge lookup) over stdio / SSE |
| ☁️ ALM synchronous HTTP | `/add` / `/search` contract for external programs, spaces isolated by `user_id` |
| 🔐 Unified auth | panel and MCP share `auth.json` (Cookie / Basic / Bearer); changing the password revokes old sessions |
| 🖥️ 3D visualization | force-directed graph, layer filtering, focus mode, online CRUD |
| 📄 Offline export | one-command, self-contained HTML, no server required |

## Architecture

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval@main/assets/diagrams/architecture.en.png">
    <img src="https://cdn.jsdelivr.net/gh/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval@main/assets/diagrams/architecture.en.png" width="100%" alt="Ariadne architecture: write path, storage and state, read path">
  </a>
</p>

Writes queue the conversation and let a background worker run batch maintenance (triage → extraction → edge linking → dedupe/deprecation); reads run the full chain "vector seeds → directed expansion → purpose regression → peak termination → StoryRank". MCP and the panel share one YAML file and cross-process lock, while ALM keeps a separate space per `user_id`. Notes enter both the vector store and the keyword index, but only act as retrieval seeds and never take part in edge linking.

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| **Extraction** | conversation → nodes/edges, correction & deprecation | `extraction/dba.py`, `extraction/graph_builder.py` |
| **Storage** | graph + vector index | `graph/memory_graph.py`, `embedding/store.py` |
| **Retrieval** | directed expansion, purpose filtering, peak termination | `core/jump_axis.py`, `core/purpose.py`, `core/peak_find.py` |
| **Story** | causal chains → story fragments | `retrieval/retriever.py` (StoryRank) |
| **Knowledge** | keyword-first note lookup | `kwindex.py` (inverted index, no embedding dependency) |
| **Access** | API / MCP / visualization | `viz/api_server.py`, `mcp_server.py` |

## Installation

**Requires**: Python 3.10+

```bash
# Base install (constraints.txt provides secure version floors)
pip install -c constraints.txt -e .

# With local embedding models (optional)
pip install -c constraints.txt -e ".[local]"

# Development dependencies (optional)
pip install -c constraints.txt -e ".[dev]"
```

## Quick Start

The `data/` directory **is the graph library**: every `*.yaml` in it that parses as a graph shows up under *Settings → Graph library* in the panel, where you can switch, import, export and delete. The repository ships two:

- `sample_graph.yaml` — the empty placeholder (0 nodes / 0 edges), the default landing spot on first start;
- `memory_graph.yaml` — a preset dataset for trying the system out.

**Which graph is in use is recorded in the shared pointer `<graph dir>/active_graph.json`**: once you pick one in the panel, both the panel and MCP return to it after a restart. That makes `--yaml` **optional** — it only acts as a hint for "which one to use the very first time", so day-to-day switching happens in the panel, with no config edit or restart.

### One-command start (recommended)

Run `start_all.py` from the project root to launch the 3D panel and MCP SSE together:

```bash
# 1) Copy the config template and fill in your model info
#    (MCP reads .env automatically, no need to repeat args on the CLI)
cp .env.example .env

# 2) One-command start (panel 8765 + MCP SSE 8766, ports auto-separated)
python start_all.py

# Visualization panel  http://127.0.0.1:8765
# MCP SSE              http://127.0.0.1:8766/sse
```

> `--yaml` picks the graph used on first start (optional, e.g. `--yaml data/memory_graph.yaml`); `--graph-dir` sets the graph library directory. Override default ports/address with `--api-port` / `--mcp-port` / `--host`; `Ctrl+C` stops both services.

### Entry 1: 3D visualization WebUI

View in the browser, perform manual CRUD, and inspect observability data:

```bash
ariadne-api --port 8765          # uses the graph recorded in the shared pointer (empty placeholder on first run)
# or pick one for first start: ariadne-api --yaml data/memory_graph.yaml --port 8765
# open http://127.0.0.1:8765
```

Features:

- **Graph**: 3D force-directed graph (per-role shapes/colors, labels, highlight, focus, fuzzy search), layer filtering with presets, node/edge CRUD (auto-written back to YAML, with undo). Concurrent writes with MCP are serialized via a cross-process file lock (`<yaml>.lock`), so neither side overwrites the other; panel edits are reconciled into the vector index by MCP before retrieval (graph acts as the authority).
- **Graph library** (*Settings → Graph library*): manage and switch between multiple graphs. The list only includes YAMLs in the graph directory that **parse as a graph** (sibling files such as evaluation datasets are not listed, so they cannot be deleted by accident); supports import-as-new / export / delete (the graph currently in use by the panel, and the one MCP is using, cannot be deleted).
  - **Switch** affects the panel only, and records the choice in the shared pointer so restarts return to it;
  - **Load into MCP** is a separate action: it writes a switch request that the MCP process applies on its own (within ~3s by default), **clearing and fully rebuilding the vector index**; the panel shows progress and the result. Without clicking it, MCP stays where it is — each side's "current graph" is independent.
- **Observability**: metrics overview (graph size/orphans/requests/uptime), operation log viewer (LLM + DBA, filterable), runtime logs (in-process logging plus aggregated shared logs from MCP and other processes via `ARIADNE_LOG_FILE`), live log stream (SSE).
- **Settings**: rendering/layout parameters (quality preset, forces, labels, background; persisted in the browser), filter preset management, read-only server config, graph YAML and oplog import/export.

### Entry 2: MCP server

For LLM agents (full DBA mode; requires LLM + Embedding):

```bash
ariadne-mcp \
    --llm-model gpt-4o-mini --llm-api-key sk-xxx --llm-base-url https://api.openai.com/v1 \
    --embedding-model text-embedding-3-small
```

### Entry 3: Offline HTML

Generate a self-contained visualization page without a server (reuses the WebUI frontend, read-only mode):

```bash
ariadne-render --yaml data/sample_graph.yaml -o output.html
```

## Docker Deployment

A single image provides both entry points — the MCP server (`ariadne-mcp`) and the 3D panel (`ariadne-api`) — and they share the same `./data`.

### Option 1: docker compose (recommended)

```bash
cp .env.example .env        # fill in your model settings
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
```

- **Panel**: `http://127.0.0.1:8765` — the host port binds to loopback by default; for remote access use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 <user>@<host>`). Override with `ARIADNE_VIZ_PORT` / `ARIADNE_BIND_HOST`. Default account `ariadne / ariadne`, with a forced password change on first login.
- **MCP**: the host port also binds to loopback by default (`127.0.0.1:8766`); containers on the same compose network reach it at `http://ariadne-mcp:8766/sse`. Panel/MCP speak plain HTTP with password/Basic auth — to expose them publicly set `ARIADNE_BIND_HOST=0.0.0.0` in .env, but only behind a TLS-terminating proxy.
- Both services mount `./data:/app/data`, so **the host's `./data` is the graph library**: switching graphs in the panel creates/updates `active_graph.json` there.

### Option 2: load a prebuilt image

```bash
docker load -i package/ariadne-mcp.tar

docker run -d --name ariadne-mcp -p 127.0.0.1:8766:8766 \
  -e OPENAI_API_KEY=sk-xxx -e OPENAI_API_BASE=https://api.deepseek.com/v1 -e OPENAI_MODEL=deepseek-v4-flash \
  -e EMBEDDING_LOCAL=false \
  -e EMBEDDING_API_KEY=sk-xxx -e EMBEDDING_API_BASE=https://api.siliconflow.cn/v1 -e EMBEDDING_MODEL=BAAI/bge-m3 \
  -v $(pwd)/data:/app/data \
  ariadne-mcp:latest
```

> The tar uses the OCI layout and loads on Docker 25+ / Podman / containerd; for older Docker, convert it once with `skopeo copy docker-archive:ariadne-mcp.tar docker-archive:out.tar`.

### Option 3: ALM synchronous HTTP service

`alm/` is a separate external interface (`POST /add`, `POST /search`, `GET /health`) with config **fully isolated** from the root `.env`:

```bash
cp alm/.env.alm.example alm/.env.alm   # ALM_API_KEYS / LLM / Embedding are required
docker compose --env-file alm/.env.alm -f deploy/docker-compose.alm.yml up -d --build
python -m alm.status --base-url http://127.0.0.1:8770 --api-key <your-key>   # read-only self-check
```

- The host port binds to `127.0.0.1:8770` by default; public access needs a TLS-terminating reverse proxy in front (the service does not do TLS itself);
- With `ALM_API_KEYS` empty and a non-loopback listener, the service **refuses to start** (it does not allow all requests);
- Memories live in the named volume `alm-data` and survive restarts / upgrades;
- Two-layer architecture, `user_id` space isolation and tuning knobs: see [docs/ALM.en.md](docs/ALM.en.md).

### The embedding mode decides the image size

`EMBEDDING_LOCAL` is **both a build arg and a runtime switch — the two must agree** (values `1/true/yes/on` are all accepted):

| Mode | Build | Runtime | Image size |
| --- | --- | --- | --- |
| **API** (default) | `--build-arg EMBEDDING_LOCAL=false` | `EMBEDDING_LOCAL=false` + `EMBEDDING_API_KEY/BASE/MODEL` | ~700 MB |
| **Local** | `--build-arg EMBEDDING_LOCAL=true` | `EMBEDDING_LOCAL=true` | extra CPU-only torch, 2 GB+ |

> If the build only accepted `"true"` while `.env` said `1`, torch would be skipped but the runtime would still take the local path, failing with an ImportError on startup — `deploy/Dockerfile` uses the same truthiness rules.
> Local mode downloads the model on first run; mount the HF cache (compose already defines an `hf-cache` volume) or pre-seed the cache and run offline with `HF_HUB_OFFLINE=1`.

### Graph library inside the container

Identical to a local run: which graph is in use comes from the shared pointer. The graph directory defaults to the directory of `ARIADNE_YAML`, or can be set explicitly:

```bash
-e ARIADNE_GRAPHS_DIR=/app/data
```

### Caveats

- **Never bake `data/auth.json` or `data/.ariadne_secret` into the image or a volume**: the former holds the password hash, the latter is the session-cookie signing key (leaking it lets anyone forge a login). `.dockerignore` already excludes them, and the container generates fresh ones on first start.
- The image ships **no credentials**. So without a `./data` mount, upgrading the image resets the account to the default and invalidates existing sessions (the signing key is regenerated) — this is intentional.
- The panel's **"chain test"** (running a real chain from the panel) needs model settings; compose already injects `.env` into `ariadne-viz` and sets `ARIADNE_MCP_URL=http://ariadne-mcp:8766/sse` so the panel's MCP block can health-check the other container. **The prerequisite is** that the image's embedding mode matches `EMBEDDING_LOCAL` in `.env` (an API-mode image has no torch, so `1` raises an ImportError).

### Public exposure and remote access

Both the panel and MCP speak **plain HTTP with password / Basic auth**, so credentials can be sniffed on the wire. The host ports therefore bind to `127.0.0.1` by default (controlled by `ARIADNE_BIND_HOST`):

| Scenario | How |
| --- | --- |
| Local browser / local IDE | `http://127.0.0.1:8765` and `http://127.0.0.1:8766/sse` directly |
| Remote panel | SSH tunnel `ssh -L 8765:127.0.0.1:8765 <user>@<host>`, then open `http://127.0.0.1:8765` |
| Remote MCP | Add `-L 8766:127.0.0.1:8766` to the same tunnel, or point the client at an HTTPS domain behind a proxy |
| Public service | Terminate TLS at nginx / Caddy, then set `ARIADNE_BIND_HOST=0.0.0.0` and give the panel and MCP a vhost each |

> Container-to-container traffic (the panel's MCP health check) goes over the compose network at `http://ariadne-mcp:8766/sse` and is unaffected by the bind address. ALM is the same: loopback by default, public access needs a reverse proxy plus a non-empty `ALM_API_KEYS`.

### Updating an existing deployment

| Scenario | What to replace |
| --- | --- |
| Using a prebuilt image | `docker load` the new tar; if your `deploy/docker-compose.yml` lacks `env_file` / `ARIADNE_MCP_URL` on `ariadne-viz`, replace it too |
| Building on the deploy host | `deploy/Dockerfile`, `.dockerignore`, `deploy/docker-compose.yml`, `dba_pipeline/`, `data/*.yaml`, `pyproject.toml` |
| ALM service | `deploy/Dockerfile.alm`, `deploy/docker-compose.alm.yml`, `alm/`, `alm/.env.alm.example` |

> **`.dockerignore` must be updated too**, otherwise a freshly built image will bake the credentials and signing key under `data/` back in.

## Retrieval Method: PAR Baseline

The proposed retrieval method **PAR (Proposed)** is the full pipeline, composed of three mechanisms:

1. **Jump Axis**: edges are typed into 8 relation types and nodes classified into 6 roles; a 6×8 weight matrix drives directed expansion, with zero-weight directions blocked outright to avoid undirected diffusion.
2. **Purpose Regression**: an LLM infers the query's implicit purpose and encodes it as a vector; each expansion step filters candidates that deviate from that purpose.
3. **Peak Finding**: the purpose relevance of each round is recorded; when the slope turns negative, the search falls back to the peak tolerance band for output, replacing fixed top-K.

Baseline comparison:

| Method | Composition | Key limitation |
|--------|-------------|----------------|
| **A** Pure vector | semantic embedding matching | no directionality, degrades fast on large graphs |
| **B** Vector + graph hybrid | undirected graph expansion | undirected, no incrementality (A=B in practice) |
| **C** Jump Axis | directed expansion only | no purpose constraint |
| **PAR** Full pipeline | directed + purpose + peak | divergence noise, output volume inflation |

The core advantage is **associative recall breadth**: at 216 nodes, PAR reaches R@all = 0.538 vs A = 0.159 (roughly **3.4×**), and the gap widens with scale. Note that this R@all uses a **wide-output protocol** (PAR emits 6–27 items vs A fixed top-5), measuring breadth via "more output, more hits"; under a fixed top-N, PAR is actually weaker (P@5 = 0.160 vs A = 0.267 at 216). In the quota-aligned cross-system comparison (LiFEMem 322 nodes, candidate quota aligned at 38.3), pure vector (RAG/Mem0) achieves higher expected coverage (0.840) than PAR (0.765) — pure vector is better at focused semantic recall; PAR's unique value lies in the associative/divergence zone (pure-story comparison wins 25:5 / 26:4). The two metrics measure "associative breadth vs focused recall" — two facets of the same system.

### StoryRank: narrativizing the retrieval chain

PAR retrieval produces a **causal chain** of "nodes + relations" rather than a flat candidate list. Before sending it to reply generation, StoryRank understands and organizes this chain into **story-fragment documents**, serving three responsibilities:

1. **Preserve causality**: relation types (`CAUSAL` / `PREFERENCE` / `SCENARIO`, etc.) are naturally woven into story sentences.
2. **Avoid polluting the chat context**: the chat model receives clean stories instead of raw `[id] content` node listings.
3. **Semantic filtering**: when the LLM organizes stories by edge relations, nodes that are clearly incongruent or off-topic are naturally dropped.

The entry point is `retrieve_with_story` (a library method); its output includes `stories` (narrative text), `story_used` (number of nodes the narrative actually covers, **computed backend-side**), and `story_degraded` (whether it fell back to concatenation).

> See also the theory paper: [0828PAR检索理论文档.md](docs/theory/0828PAR检索理论文档.md) *(in Chinese)*.

## MCP Server

The MCP server exposes the memory graph as 11 tools over stdio or SSE.

Two of them form the knowledge layer: `dba_add_note` writes project notes and `dba_query_knowledge` retrieves them by keyword. Notes are ordinary graph nodes (`metadata.channel="note"`), not a separate store, so they can be recalled semantically by the memory layer and screened literally by the knowledge layer.

- Tool reference, environment variables, graph switching and agent setup: [docs/MCP.en.md](docs/MCP.en.md)
- One-command start: `python start_all.py`

## ALM Synchronous HTTP Service

A synchronous HTTP interface for external programs: `POST /add` (write), `POST /search` (retrieve),
`GET /health` (liveness). Spaces are isolated by `user_id`, over a two-layer architecture
(raw-text layer L1 + graph layer L2): long documents are chunked into the raw-text layer without
graph construction, while conversational material is extracted into nodes and edges — keeping LLM
cost where it pays off.

- Architectural differences, two-layer routing and tuning knobs: [docs/ALM.en.md](docs/ALM.en.md)
- Container deployment: see Option 3 under [Docker Deployment](#docker-deployment); config template `alm/.env.alm.example`

## Data Format and Type System

See [docs/architecture.en.md](docs/architecture.en.md#data-format-and-type-system) for the
YAML schema and the full lists of 6 node types and 8 relation types.

## Project Structure

See [docs/architecture.en.md](docs/architecture.en.md) for the full tree and module responsibilities.

- `dba_pipeline/`: retrieval, extraction, graph, LLM, WebUI and MCP server
- `alm/`: synchronous HTTP memory service
- `deploy/`: container files (`Dockerfile` + `docker-compose.yml` for MCP/panel, `Dockerfile.alm` + `docker-compose.alm.yml` for ALM)
- `packaging/`: release packaging (npm shim, quickstart scripts)
- `data/`: graph library and sample data
- `tests/`: pytest suite

## References

- [0828前置数据管理与后处理.md](docs/theory/0828前置数据管理与后处理.md) *(data management / LLM DBA theory, in Chinese)*
- [0828PAR检索理论文档.md](docs/theory/0828PAR检索理论文档.md) *(purpose-driven associative retrieval theory, in Chinese)*
- [0908e2e实验文档.md](docs/theory/0908e2e实验文档.md) *(end-to-end experiments, in Chinese)*

## License

This project is licensed under the [GNU AGPLv3](LICENSE) (`AGPL-3.0-only`). Copyright © 2026 **kleetor**.

> The AGPLv3 **network-use clause (Section 13)** applies: when providing the service remotely to third parties (e.g. MCP SSE / HTTP API / 3D panel), you must offer the complete source code to its users — the source is already public in this repository.
