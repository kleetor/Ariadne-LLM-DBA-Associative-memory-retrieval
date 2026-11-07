# ariadne-memory

Anthropomorphic long-term memory for LLM agents — the npm launcher for the **Ariadne** memory server.

Ariadne gives an Agent a memory of its own: an LLM extracts typed facts and experiences from conversation into a
knowledge graph, and purpose-driven associative retrieval (PAR) with StoryRank turns them back into something the
agent can recall. It is built for personal conversation and emotional companionship with a single user.

Design notes, architecture and the 3D panel live in the
[GitHub repository](https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval).

This package is a launcher only — it does **not** bundle Python. The Python package is published on PyPI as
`ariadne-mcp-server`.

```bash
npx ariadne-memory start        # 3D panel + MCP (SSE)
npx ariadne-memory api          # panel only
npx ariadne-memory mcp --sse    # MCP only (SSE); omit --sse for stdio
```

What it does:

1. checks that `uv` is installed
2. runs `uvx --from ariadne-mcp-server ariadne-api` / `ariadne-mcp`
3. forwards all extra arguments unchanged

- `ARIADNE_UVX_FROM` overrides the package spec passed to `uvx`
  (useful before the PyPI release: point it at a local path).
- `--dry-run` prints the commands without executing them.
- `--help` / `--version` are supported.
