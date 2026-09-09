# Ariadne

**LLM DBA Management & Purpose-Driven Associative Memory Retrieval System**

A full-pipeline system for LLM-driven memory graph construction, retrieval, and visualization.

All planning & report documents live in the `Plan/` directory; the root-level documents are curated so that handing them to an AI lets it quickly grasp this system's functionality and implementation value. The `data/` directory contains an empty `sample` dataset for first-time use and a pre-populated `memory` dataset for quickly experiencing the system.

Future direction: further research on the retrieval algorithm — in different usage scenarios, the base weight table and other collaborating parts may be tuned via parameters to achieve different retrieval behaviors (e.g. opening up taxonomic weights could enable a knowledge-base-style query mode).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-AGPLv3-blue)](LICENSE)
![Version](https://img.shields.io/badge/Version-0.1.0-lightgrey)

> *"A thread through the labyrinth of memory."*

**Language**: [English](README.en.md) · [中文](README.md)

---

## Goals & Rationale

> Memory is not a single storehouse, but a duet of "facts" and "experiences."

Ariadne's long-term goal is to answer one question: **how should an Agent manage and retrieve memory and knowledge in a way that is as scientific and efficient as a human's?**

Cognitive science distinguishes two types of human long-term memory (Tulving):

- **Semantic Memory** — decontextualized, stable knowledge: facts, concepts, and rules about the world ("The Earth revolves around the Sun", "the user is a backend engineer"). It is like a **knowledge base**, emphasizing accuracy, consistency, and verifiability.
- **Episodic Memory** — personal experiences as time-space-emotion fragments ("last Tuesday I worked overtime until midnight and my neck ached"). It emphasizes **context, causality, sequence, and emotional color**, and tolerates approximation and association.

The human brain processes these two kinds of memory with different mechanisms and lets them **collaborate**: semantic memory provides background knowledge to make sense of new experiences, while new experiences gradually sediment into semantic knowledge. Ariadne argues that an LLM Agent's long-term memory should follow the same principle of separation and collaboration:

- **Separation** — the knowledge base (semantic) and the experience store (episodic) should be stored in layers and retrieved on demand, avoiding cross-contamination between "stable facts" and "personal experiences": knowledge demands accuracy and consistency, while experiences are allowed to evolve over time.
- **Collaboration** — semantic knowledge provides explanatory background for episodic recall; specific experiences in turn validate, correct, and consolidate semantic facts; the two convert into each other through graph relations.

How this thinking lands in Ariadne today and where it is headed:

| Stage | Semantic Memory (Knowledge Base) | Episodic Memory (Experience Store) |
| --- | --- | --- |
| **Today** | A directed, typed graph: nodes + edges carry stable facts, maintained by DBA (extraction / correction / deduplication / deprecation) for consistency | Personal experience nodes in the graph, organized by StoryRank into story fragments preserving causality / sequence / emotion |
| **Future** | An explicit "knowledge layer": independent storage and consistency checking of decontextualized facts | An explicit "experience layer": spatiotemporal-emotional retrieval unitized by experience |

The ultimate goal is to let an Agent **recall an experience like a person remembering, and consult a fact like looking something up in a database** — that is the meaning of Ariadne as a thread through the labyrinth of memory.

---

## Table of Contents

- [Goals & Rationale](#goals--rationale)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Retrieval Method: PAR Baseline](#retrieval-method-par-baseline)
- [MCP Server](#mcp-server)
- [Data Format](#data-format)
- [Type System](#type-system)
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
| 🔌 MCP integration | 7 tools over stdio / SSE transports |
| 🖥️ 3D visualization | force-directed graph, layer filtering, focus mode, online CRUD |
| 📄 Offline export | one-command, self-contained HTML, no server required |

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │              Ariadne Core Pipeline        │
                        └──────────────────────────────────────────┘

Conversation ──► DBA Maintenance ──► MemoryGraph + VectorStore ──► PAR Retrieval ──► StoryRank ──► Reply
    │                │                        │
    │      MaintenanceScheduler               ├──► API Server (HTTP REST + 3D panel)
    │      (batched async)                    ├──► MCP Server (7 tools, stdio / SSE)
    │                                         └──► Offline HTML
    └──► Human intervention (CRUD panel + MCP dba_intervene)
```

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| **Extraction** | conversation → nodes/edges, correction & deprecation | `extraction/dba.py`, `extraction/graph_builder.py` |
| **Storage** | graph + vector index | `graph/memory_graph.py`, `embedding/store.py` |
| **Retrieval** | directed expansion, purpose filtering, peak termination | `core/jump_axis.py`, `core/purpose.py`, `core/peak_find.py` |
| **Story** | causal chains → story fragments | `retrieval/retriever.py` (StoryRank) |
| **Access** | API / MCP / visualization | `viz/api_server.py`, `mcp_server.py` |

## Installation

**Requires**: Python 3.10+

```bash
# Base install
pip install -e .

# With local embedding models (optional)
pip install -e ".[local]"

# Development dependencies (optional)
pip install -e ".[dev]"
```

## Quick Start

The repository ships with a sample graph `data/sample_graph.yaml` (10 nodes, 7 edges).

### One-command start (recommended)

Run `start_all.py` from the project root to launch the 3D panel and MCP SSE together:

```bash
# 1) Copy the config template and fill in your model info
#    (MCP reads .env automatically, no need to repeat args on the CLI)
cp .env.example .env

# 2) One-command start (panel 8765 + MCP SSE 8766, ports auto-separated)
python start_all.py --yaml data/sample_graph.yaml

# Visualization panel  http://127.0.0.1:8765
# MCP SSE              http://127.0.0.1:8766/sse
```

> Override default ports/address with `--api-port` / `--mcp-port` / `--host`; `Ctrl+C` stops both services.

### Entry 1: 3D visualization panel

View in the browser and perform manual CRUD:

```bash
ariadne-api --yaml data/sample_graph.yaml --port 8765
# open http://127.0.0.1:8765
```

Features: 3D force-directed graph, layer filtering, focus mode, fuzzy search, CRUD panel, with changes written back to YAML automatically.

### Entry 2: MCP server

For LLM agents (full DBA mode; requires LLM + Embedding):

```bash
ariadne-mcp --yaml data/sample_graph.yaml \
    --llm-model gpt-4o-mini --llm-api-key sk-xxx --llm-base-url https://api.openai.com/v1 \
    --embedding-model text-embedding-3-small
```

### Entry 3: Offline HTML

Generate a self-contained visualization page without a server:

```bash
ariadne-render --yaml data/sample_graph.yaml -o output.html
```

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

The core advantage is **associative recall breadth**: at 216 nodes, PAR reaches R@all = 0.538 vs A = 0.159 (roughly **3.4×**), and the gap widens with scale. Note that this R@all uses a **wide-output protocol** (PAR emits 6–27 items vs A fixed top-5), measuring breadth via "more output, more hits"; under a fixed top-N, PAR is actually weaker (P@5 = 0.160 vs A = 0.267 at 216). In the quota-aligned cross-system comparison (LiFEMem 322 nodes, candidate quota aligned at 38.3), pure vector (RAG/Mem0) achieves higher expected coverage (0.840) than PAR (0.765) — pure vector is better at focused semantic recall; PAR's unique value lies in the associative/divergence zone (pure-story comparison wins 25:5 / 26:4). The two metrics measure "associative breadth vs focused recall" — two facets of the same system (see evaluation report §4.8).

### StoryRank: narrativizing the retrieval chain

PAR retrieval produces a **causal chain** of "nodes + relations" rather than a flat candidate list. Before sending it to reply generation, StoryRank understands and organizes this chain into **story-fragment documents**, serving three responsibilities:

1. **Preserve causality**: relation types (`CAUSAL` / `PREFERENCE` / `SCENARIO`, etc.) are naturally woven into story sentences.
2. **Avoid polluting the chat context**: the chat model receives clean stories instead of raw `[id] content` node listings.
3. **Semantic filtering**: when the LLM organizes stories by edge relations, nodes that are clearly incongruent or off-topic are naturally dropped.

The entry point is `retrieve_with_story` (a library method); its output includes `stories`, `story_nodes` (adopted nodes), and `discarded_nodes` (dropped nodes).

> See also the theory paper: [0828PAR检索理论文.md](0828PAR检索理论文.md) *(in Chinese)*.

## MCP Server

Built on the open [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), Ariadne's memory graph is exposed as a server callable by any MCP client. See the [MCP Specification](https://spec.modelcontextprotocol.io/) for protocol details.

### Transport modes

| Mode | Usage | Use case |
|------|-------|----------|
| **stdio** (default) | `ariadne-mcp --yaml xxx.yaml` | clients that spawn a local process, e.g. Claude Desktop |
| **SSE** | `ariadne-mcp --yaml xxx.yaml --sse --port 8765` | clients connecting via a network URL, e.g. Cursor |

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

Only the **full DBA mode** is currently supported: an LLM must be configured (`--llm-model` or `OPENAI_MODEL`). On startup, `dba_add_conversation` invokes the internal LLM for memory maintenance: **node extraction (Step 1) and edge linking (Step 2) are two separate LLM calls** — nodes are extracted and applied first, then edges are linked using all of this round's new nodes plus related old nodes / one-hop neighbors / existing edges, avoiding the visibility limits of a single-pass output (see the [0822 extraction evaluation report](Plan/0822——DBA抽取环节评测报告.md)). Missing LLM / DBA dependencies cause an immediate error exit (no stub-mode fallback).

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

### 7 tools

| Tool | Description |
|------|-------------|
| `dba_add_conversation` | append a conversation; queued in the scheduler first and maintained asynchronously once a threshold is reached |
| `dba_query_memory` | purpose-driven associative retrieval (PAR pipeline: Jump Axis + Purpose Regression + Peak Finding) |
| `dba_temporal_lookup` | standalone single-hop temporal query (only "time → event", reverse direction) for "what happened at a given time" |
| `dba_inspect_graph` | expand a node's 1-hop neighbors |
| `dba_intervene` | manually CRUD nodes and edges |
| `dba_checkpoint` | save a full checkpoint |
| `dba_get_stats` | graph statistics |

### Query notes (`dba_query_memory`)

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

## Data Format

The graph persists as YAML, with nodes and edges stored in two top-level lists, `nodes` and `edges`:

```yaml
nodes:
  - id: n1
    content: "The user does backend development at an internet company"
    type: status
    deprecated: false
    forgotten: false
  - id: n2
    content: "The user often works overtime until 10 PM"
    type: reason
    deprecated: false
    forgotten: false

edges:
  - from: n2
    to: n1
    type: causal
    bidirectional: false
```

| Field | Description |
|-------|-------------|
| `nodes[].type` | node role type (see the 6 node types below) |
| `nodes[].deprecated` | whether deprecated (excluded from retrieval) |
| `nodes[].forgotten` | whether forgotten (excluded from retrieval) |
| `edges[].type` | relation type (see the 8 edge types below) |
| `edges[].bidirectional` | whether bidirectional (`SCENARIO` / `SOCIAL` / `ATTRIBUTE` are bidirectional by default) |

## Type System

**Nodes (6 types)**:

| Type | Meaning |
|------|---------|
| `STATUS` | status / current state |
| `REASON` | reason |
| `ACTION` | behavior / action |
| `THING` | thing / object |
| `PERSON` | person |
| `EMOTION` | emotion |

**Edges (8 types)**:

| Type | Meaning | Direction |
|------|---------|-----------|
| `CAUSAL` | causation | points to the result |
| `SCENARIO` | scenario membership | bidirectional |
| `SEQUENCE` | temporal order | points to the successor |
| `PREFERENCE` | attitude / preference | points to the object |
| `SOCIAL` | social relation | bidirectional |
| `ATTRIBUTE` | attribute ownership | bidirectional |
| `TEMPORAL` | temporal positioning | points to the time |
| `TAXONOMIC` | classification / ontology | points to the parent class |

## Project Structure

```
.
├── start_all.py                    # one-command start (panel + MCP SSE)
├── data/
│   ├── sample_graph.yaml           # sample graph
│   └── memory_graph.yaml           # live data
└── dba_pipeline/
    ├── core/                       # retrieval core: Jump Axis, Purpose, Peak Finding
    │   ├── jump_axis.py
    │   ├── purpose.py
    │   ├── peak_find.py
    │   └── path_tracker.py
    ├── graph/                      # graph structure
    │   └── memory_graph.py
    ├── embedding/                  # vector store
    │   └── store.py
    ├── extraction/                 # DBA extraction & batched maintenance
    │   ├── dba.py
    │   ├── graph_builder.py
    │   └── maintenance_scheduler.py
    ├── llm/                        # inference engine
    │   └── inference.py
    ├── retrieval/                  # PAR retrieval + StoryRank
    │   └── retriever.py
    ├── viz/                        # API server, 3D rendering, export
    │   ├── api_server.py
    │   ├── renderer.py
    │   └── exporter.py
    ├── mcp_server.py               # MCP server entry point
    └── loader.py                   # graph / query loading
```

## References

- [0828前置数据管理与后处理.md](0828前置数据管理与后处理.md) *(data management / LLM DBA theory, in Chinese)*
- [0828PAR检索理论文.md](0828PAR检索理论文.md) *(purpose-driven associative retrieval theory, in Chinese)*
- [0828对比评测数据.md](0828对比评测数据.md) *(retrieval & story-quality multi-system comparison, in Chinese)*

## License

This project is licensed under the [GNU AGPLv3](LICENSE) (`AGPL-3.0-only`). Copyright © 2026 **kleetor**.

> The AGPLv3 **network-use clause (Section 13)** applies: when providing the service remotely to third parties (e.g. MCP SSE / HTTP API / 3D panel), you must offer the complete source code to its users — the source is already public in this repository.
