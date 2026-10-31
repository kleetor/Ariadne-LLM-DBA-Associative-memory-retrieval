#!/usr/bin/env node
/**
 * Ariadne thin launcher.
 *
 * It does three things only:
 *   1. checks that uv is installed
 *   2. runs `uvx --from ariadne-mcp-server ariadne-api / ariadne-mcp`
 *   3. forwards every extra argument unchanged
 *
 * It deliberately does NOT bundle Python or install PyTorch.
 */
import { spawn, spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = JSON.parse(readFileSync(join(here, '..', 'package.json'), 'utf8'))
const UVX_FROM = process.env.ARIADNE_UVX_FROM || 'ariadne-mcp-server'

const HELP = `ariadne-memory — thin launcher for the Ariadne memory server

Usage:
  ariadne-memory start [args]   Start panel + MCP (SSE)
  ariadne-memory api [args]     Start only the 3D panel (ariadne-api)
  ariadne-memory mcp [args]     Start only the MCP server (stdio unless you pass --sse)

Options:
  -h, --help     Show this help
  -v, --version  Print version
  --dry-run      Print the uvx commands without running them

Requires: uv (https://docs.astral.sh/uv/)
Env: ARIADNE_UVX_FROM  package spec passed to uvx (default: ariadne-mcp-server)
`

const argv = process.argv.slice(2)
if (argv.includes('-h') || argv.includes('--help')) {
  console.log(HELP)
  process.exit(0)
}
if (argv.includes('-v') || argv.includes('--version')) {
  console.log(pkg.version)
  process.exit(0)
}

const dryRun = argv.includes('--dry-run')
const args = argv.filter((arg) => arg !== '--dry-run')
const command = args[0] && !args[0].startsWith('-') ? args[0] : 'start'
const rest = command === args[0] ? args.slice(1) : args

const plans = {
  start: [
    { bin: 'ariadne-api', extra: [] },
    { bin: 'ariadne-mcp', extra: ['--sse'] },
  ],
  api: [{ bin: 'ariadne-api', extra: [] }],
  mcp: [{ bin: 'ariadne-mcp', extra: [] }],
}

const plan = plans[command]
if (!plan) {
  console.error(`[ariadne] unknown command: ${command}\n`)
  console.error(HELP)
  process.exit(1)
}

const uvxArgs = (step) => ['--from', UVX_FROM, step.bin, ...step.extra, ...rest]

if (dryRun) {
  for (const step of plan) {
    console.log(['uvx', ...uvxArgs(step)].join(' '))
  }
  process.exit(0)
}

const probe = spawnSync('uv', ['--version'], { stdio: 'ignore' })
if (probe.error || probe.status !== 0) {
  console.error('[ariadne] uv not found. Install it first:')
  console.error('  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh')
  console.error('  Windows:     powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"')
  process.exit(1)
}

if (command === 'start') {
  const children = plan.map((step) =>
    spawn('uvx', uvxArgs(step), { stdio: 'inherit' }),
  )

  const stop = () => {
    for (const child of children) {
      if (!child.killed) child.kill('SIGTERM')
    }
  }

  process.on('SIGINT', () => {
    stop()
    process.exit(130)
  })
  process.on('SIGTERM', () => {
    stop()
    process.exit(143)
  })

  // MCP runs in the foreground: when it exits, stop the panel too.
  const [, mcp] = children
  mcp.on('exit', (code) => {
    stop()
    process.exit(code ?? 0)
  })
  children[0].on('exit', (code) => {
    if (code && code !== 0) console.error(`[ariadne] panel exited with code ${code}`)
  })
} else {
  const child = spawn('uvx', uvxArgs(plan[0]), { stdio: 'inherit' })
  child.on('exit', (code) => process.exit(code ?? 0))
}
