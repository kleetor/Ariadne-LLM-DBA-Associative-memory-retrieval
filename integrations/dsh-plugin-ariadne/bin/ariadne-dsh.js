#!/usr/bin/env node
/**
 * ariadne-dsh — Ariadne 记忆服务的配置与状态工具（DSH / DeepSeek Harness）。
 *
 *   ariadne-dsh init     交互式写入 ~/.ariadne-dsh/config.json（可同时写 profile 的 .env）
 *   ariadne-dsh env      打印 export 行，供手动 source
 *   ariadne-dsh status   查看配置、uv 与图谱目录概况
 *   ariadne-dsh serve    以 stdio 模式前台拉起 ariadne-mcp（给 MCP client 用）
 *
 * 设计：DSH 的 MCP client 通过 cordis.patch.yml 读取 process.env.ARIADNE_* / OPENAI_* 等，
 * 本工具只负责把这些值收集好，不改变 Ariadne 服务本身。
 */
import { spawn, spawnSync } from 'node:child_process'
import { chmodSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'

const CONFIG_DIR = join(homedir(), '.ariadne-dsh')
const CONFIG_PATH = join(CONFIG_DIR, 'config.json')
const BLOCK_START = '# >>> ariadne-dsh >>>'
const BLOCK_END = '# <<< ariadne-dsh <<<'

const FIELDS = [
  { key: 'OPENAI_API_KEY', label: 'LLM API Key', secret: true },
  { key: 'OPENAI_API_BASE', label: 'LLM API Base', fallback: 'https://api.deepseek.com/v1' },
  { key: 'OPENAI_MODEL', label: 'LLM 模型', fallback: 'deepseek-v4-flash' },
  { key: 'EMBEDDING_API_KEY', label: 'Embedding API Key', secret: true },
  { key: 'EMBEDDING_API_BASE', label: 'Embedding API Base', fallback: 'https://api.siliconflow.cn/v1' },
  { key: 'EMBEDDING_MODEL', label: 'Embedding 模型', fallback: 'BAAI/bge-m3' },
  { key: 'ARIADNE_GRAPHS_DIR', label: '图谱目录', fallback: join(process.cwd(), 'data') },
  { key: 'ARIADNE_LOG_LEVEL', label: '日志级别', fallback: 'INFO' },
]

const HELP = `ariadne-dsh — Ariadne 记忆服务配置与状态

用法:
  ariadne-dsh init [--yes] [--write-env <file>]
  ariadne-dsh env
  ariadne-dsh status
  ariadne-dsh serve [--graph-dir <dir>]

说明:
  init 会把配置写到 ${CONFIG_PATH}
  --write-env <file>  同时把环境变量块写入指定文件（例如 DSH profile 的 .env）
  --yes               不交互，直接使用当前环境变量 / 默认值

依赖: uv（https://docs.astral.sh/uv/）
`

function loadConfig() {
  if (!existsSync(CONFIG_PATH)) return {}
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, 'utf8'))
  } catch (error) {
    console.error(`[ariadne] 无法解析 ${CONFIG_PATH}: ${error.message}`)
    return {}
  }
}

function saveConfig(config) {
  mkdirSync(CONFIG_DIR, { recursive: true })
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 })
  try {
    chmodSync(CONFIG_PATH, 0o600)
  } catch {
    // Windows 上 chmod 语义有限，忽略
  }
}

function mask(value) {
  if (!value) return '(未设置)'
  const text = String(value)
  if (text.length <= 8) return '***'
  return `${text.slice(0, 4)}…${text.slice(-4)}`
}

function toEnv(config) {
  const env = { ...process.env }
  for (const [key, value] of Object.entries(config)) {
    if (value !== undefined && value !== null && value !== '') env[key] = String(value)
  }
  return env
}

function writeEnvBlock(file, config) {
  const target = resolve(file)
  const lines = Object.entries(config)
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}=${value}`)
  const block = [BLOCK_START, ...lines, BLOCK_END].join('\n')

  let text = ''
  if (existsSync(target)) {
    text = readFileSync(target, 'utf8')
    const pattern = new RegExp(`${BLOCK_START}[\\s\\S]*?${BLOCK_END}\\n?`, 'm')
    text = text.replace(pattern, '')
  }
  mkdirSync(dirname(target), { recursive: true })
  writeFileSync(target, `${text.replace(/\s*$/, '')}\n\n${block}\n`, 'utf8')
  console.log(`[ariadne] 已写入 ${target}`)
}

async function init(argv) {
  const nonInteractive = argv.includes('--yes')
  const envIndex = argv.indexOf('--write-env')
  const envFile = envIndex >= 0 ? argv[envIndex + 1] : null

  const config = { ...loadConfig() }
  if (nonInteractive) {
    for (const field of FIELDS) {
      const value = process.env[field.key] || config[field.key] || field.fallback
      if (value) config[field.key] = value
    }
  } else {
    const rl = createInterface({ input: process.stdin, output: process.stdout })
    try {
      console.log('按回车保留中括号内的默认值；密钥输入会回显，注意终端记录。\n')
      for (const field of FIELDS) {
        const current = config[field.key] || process.env[field.key] || field.fallback || ''
        const shown = field.secret && current ? '(已保存，回车保留)' : current
        const answer = (await rl.question(`${field.label}${shown ? ` [${shown}]` : ''}: `)).trim()
        if (answer) config[field.key] = answer
        else if (current && !config[field.key]) config[field.key] = current
      }
      const local = (await rl.question('使用本地 embedding？(y/N): ')).trim().toLowerCase()
      config.EMBEDDING_LOCAL = local === 'y' ? 'true' : 'false'
    } finally {
      rl.close()
    }
  }

  saveConfig(config)
  console.log(`[ariadne] 配置已保存：${CONFIG_PATH}`)
  if (envFile) writeEnvBlock(envFile, config)
  console.log('[ariadne] 下一步：重启 DSH，或先运行 `ariadne-dsh status` 检查')
}

function printEnv() {
  const config = loadConfig()
  if (!Object.keys(config).length) {
    console.error('[ariadne] 还没有配置，先运行 `ariadne-dsh init`')
    process.exit(1)
  }
  const isWindows = process.platform === 'win32'
  for (const [key, value] of Object.entries(config)) {
    if (value === undefined || value === null || value === '') continue
    console.log(isWindows ? `set ${key}=${value}` : `export ${key}=${value}`)
  }
}

function countDirs(path) {
  try {
    return readdirSync(path, { withFileTypes: true }).filter((entry) => entry.isDirectory()).length
  } catch {
    return null
  }
}

function status() {
  const config = loadConfig()
  const uv = spawnSync('uv', ['--version'], { encoding: 'utf8' })
  const graphs = config.ARIADNE_GRAPHS_DIR || join(process.cwd(), 'data')
  const graphFile = config.ARIADNE_YAML || join(graphs, 'memory_graph.yaml')

  console.log('Ariadne · DSH\n')
  console.log(`配置文件    ${existsSync(CONFIG_PATH) ? CONFIG_PATH : `${CONFIG_PATH}（缺失，先 init）`}`)
  console.log(`uv          ${uv.status === 0 ? uv.stdout.trim() : '未找到（安装：https://docs.astral.sh/uv/）'}`)
  console.log(`图谱目录    ${graphs}${existsSync(graphs) ? '' : '（不存在）'}`)
  console.log(`图谱文件    ${existsSync(graphFile) ? graphFile : `${graphFile}（未找到）`}`)

  if (existsSync(graphs)) {
    const users = countDirs(graphs)
    console.log(`用户空间    ${users == null ? '-' : users}`)
  }
  if (existsSync(graphFile)) {
    const size = statSync(graphFile).size
    console.log(`图谱大小    ${(size / 1024).toFixed(1)} KB`)
  }

  console.log('\n配置项')
  for (const field of FIELDS) {
    const value = config[field.key]
    console.log(`  ${field.key.padEnd(22)} ${field.secret ? mask(value) : value || '(未设置)'}`)
  }
  console.log(`  EMBEDDING_LOCAL        ${config.EMBEDDING_LOCAL || 'false'}`)

  if (!config.OPENAI_API_KEY || !config.EMBEDDING_API_KEY) {
    console.log('\n提示：OPENAI_API_KEY / EMBEDDING_API_KEY 还没填全，服务会在首次调用时失败。')
  }
}

function serve(argv) {
  const config = loadConfig()
  const graphIndex = argv.indexOf('--graph-dir')
  if (graphIndex >= 0 && argv[graphIndex + 1]) {
    config.ARIADNE_GRAPHS_DIR = argv[graphIndex + 1]
  }
  if (!existsSync(CONFIG_PATH) && !config.ARIADNE_GRAPHS_DIR) {
    console.error('[ariadne] 还没有配置：先运行 `ariadne-dsh init`')
    process.exit(1)
  }

  const args = ['--from', 'ariadne-mcp-server', 'ariadne-mcp']
  if (config.ARIADNE_GRAPHS_DIR) args.push('--graph-dir', config.ARIADNE_GRAPHS_DIR)
  if (config.ARIADNE_YAML) args.push('--yaml', config.ARIADNE_YAML)

  const child = spawn('uvx', args, { stdio: 'inherit', env: toEnv(config) })
  const forward = (signal) => {
    if (!child.killed) child.kill(signal)
  }
  process.on('SIGINT', () => forward('SIGINT'))
  process.on('SIGTERM', () => forward('SIGTERM'))
  child.on('exit', (code) => process.exit(code ?? 0))
}

const [command, ...rest] = process.argv.slice(2)
if (command === '-h' || command === '--help' || !command) {
  console.log(HELP)
} else if (command === 'init') {
  await init(rest)
} else if (command === 'env') {
  printEnv()
} else if (command === 'status') {
  status()
} else if (command === 'serve') {
  serve(rest)
} else {
  console.error(`[ariadne] 未知命令：${command}\n`)
  console.log(HELP)
  process.exit(1)
}
