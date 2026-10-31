# ariadne-memory

A thin npm launcher for the Ariadne memory server. It does **not** bundle Python.

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

The Python package is published separately as `ariadne-mcp-server`.
