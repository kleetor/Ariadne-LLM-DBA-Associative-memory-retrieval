# Architecture and Project Structure

```
.
├── start_all.py                    # one-command start (panel + MCP SSE)
├── .github/                        # CI and Issue/PR templates
├── alm/                            # synchronous HTTP memory service (Add / Search / Health)
├── assets/                         # icons and diagrams
├── data/                           # graph library directory (panel lists every graph in it)
│   ├── sample_graph.yaml           # empty placeholder (0 nodes, default landing spot on first start)
│   ├── memory_graph.yaml           # preset dataset
│   └── active_graph.json           # shared pointer: each side's current graph (runtime-generated)
├── deploy/                         # Dockerfile, Dockerfile.alm and compose files
├── docs/                           # docs index, ALM/MCP/architecture guides and theory papers
│   ├── theory/                     # 0828 retrieval theory and experiments
│   ├── ALM.en.md / MCP.en.md       # service guides (English)
│   └── architecture.en.md          # this file
├── examples/                       # ALM / MCP quick-start examples
├── tests/                          # pytest suite
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
    ├── viz/                        # WebUI server, static frontend, log bus, export
    │   ├── api_server.py           # Starlette REST + SSE server
    │   ├── logbus.py               # app-log capture + oplog tail + SSE fan-out
    │   ├── chain_runner.py         # reuses the MCP retrieval chain inside the panel
    │   ├── renderer.py             # offline self-contained HTML
    │   ├── exporter.py
    │   └── static/                 # WebUI frontend (index.html / css / js)
    ├── mcp_server.py               # MCP server entry point
    ├── graphlib.py                 # graph library + shared pointer (panel & MCP)
    ├── webauth.py                  # unified auth (session cookie / Basic / Bearer)
    └── loader.py                   # graph / query loading
```

---

## Data Format and Type System

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
