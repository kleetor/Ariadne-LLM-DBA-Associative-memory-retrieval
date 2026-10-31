Ariadne Quickstart
==================

Starts the 3D panel (ariadne-api) and the MCP server (ariadne-mcp --sse)
with a single script. No Python installation is required: the launcher
uses uv / uvx to fetch and run the published package.

Requirements
------------
- uv (https://docs.astral.sh/uv/)
  Windows:     powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

Steps
-----
1. Unzip this package anywhere.
2. Copy .env.example to .env and fill in:
     OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL
     EMBEDDING_API_KEY / EMBEDDING_API_BASE / EMBEDDING_MODEL
3. Run the launcher:
     Windows:     powershell -ExecutionPolicy Bypass -File .\start.ps1
     macOS/Linux: chmod +x start.sh && ./start.sh
4. Open http://127.0.0.1:8765
   Default login: ariadne / ariadne (change it on first login).

Ports
-----
- 3D panel: 8765  (ARIADNE_VIZ_PORT)
- MCP SSE : 8766  (ARIADNE_PORT)
Stop with Ctrl+C; the script also stops the panel process.

Data
----
The graph YAML, auth.json and logs live in ./data.
The service creates an empty graph and credentials on first start.

Notes
-----
- The launcher runs the PyPI package "ariadne-mcp-server".
  Until that package is published, the scripts cannot start the service.
  For local testing set ARIADNE_UVX_FROM to a local path or package spec.
- This package uses API embedding (EMBEDDING_LOCAL=false).
  Local embedding requires a different install with torch.
