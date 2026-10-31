#!/usr/bin/env sh
# Ariadne quickstart launcher (macOS / Linux)
# Starts the 3D panel (ariadne-api) and the MCP server (ariadne-mcp --sse).
set -eu

cd "$(dirname "$0")"

UVX_FROM="${ARIADNE_UVX_FROM:-ariadne-mcp-server}"

if ! command -v uv >/dev/null 2>&1; then
  cat <<'EOF'
[quickstart] uv not found.

Install it first:
  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
  Windows:     powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

Then run this script again.
EOF
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[quickstart] Created .env from .env.example."
  echo "[quickstart] Fill in OPENAI_* and EMBEDDING_* keys, then run this script again."
  exit 1
fi

# Export every KEY=VALUE line from .env into the process environment.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

YAML_PATH="${ARIADNE_YAML:-./data/memory_graph.yaml}"
VIZ_PORT="${ARIADNE_VIZ_PORT:-8765}"
MCP_PORT="${ARIADNE_PORT:-8766}"

mkdir -p ./data

echo "[quickstart] panel : http://127.0.0.1:${VIZ_PORT}"
echo "[quickstart] mcp   : http://127.0.0.1:${MCP_PORT}/sse"
echo "[quickstart] graph : ${YAML_PATH}"

uvx --from "$UVX_FROM" ariadne-api --yaml "$YAML_PATH" --host 127.0.0.1 --port "$VIZ_PORT" &
API_PID=$!

cleanup() {
  if kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

uvx --from "$UVX_FROM" ariadne-mcp --yaml "$YAML_PATH" --sse --host 127.0.0.1 --port "$MCP_PORT"
