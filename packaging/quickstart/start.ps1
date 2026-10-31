# Ariadne quickstart launcher (Windows PowerShell)
# Starts the 3D panel (ariadne-api) and the MCP server (ariadne-mcp --sse).
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$UvxFrom = if ($env:ARIADNE_UVX_FROM) { $env:ARIADNE_UVX_FROM } else { 'ariadne-mcp-server' }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '[quickstart] uv not found.' -ForegroundColor Yellow
    Write-Host 'Install it first:'
    Write-Host '  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host 'Then run this script again.'
    exit 1
}

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host '[quickstart] Created .env from .env.example.'
    Write-Host '[quickstart] Fill in OPENAI_* and EMBEDDING_* keys, then run this script again.'
    exit 1
}

# Load KEY=VALUE lines into the process environment (ignore comments / blanks).
Get-Content '.env' | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim(), 'Process')
    }
}

$YamlPath = if ($env:ARIADNE_YAML) { $env:ARIADNE_YAML } else { './data/memory_graph.yaml' }
$VizPort = if ($env:ARIADNE_VIZ_PORT) { $env:ARIADNE_VIZ_PORT } else { '8765' }
$McpPort = if ($env:ARIADNE_PORT) { $env:ARIADNE_PORT } else { '8766' }

New-Item -ItemType Directory -Force -Path './data' | Out-Null

Write-Host "[quickstart] panel : http://127.0.0.1:$VizPort"
Write-Host "[quickstart] mcp   : http://127.0.0.1:$McpPort/sse"
Write-Host "[quickstart] graph : $YamlPath"

$api = Start-Process -FilePath 'uvx' -NoNewWindow -PassThru -ArgumentList @(
    '--from', $UvxFrom, 'ariadne-api',
    '--yaml', $YamlPath, '--host', '127.0.0.1', '--port', $VizPort
)

try {
    & uvx --from $UvxFrom ariadne-mcp --yaml $YamlPath --sse --host 127.0.0.1 --port $McpPort
}
finally {
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
