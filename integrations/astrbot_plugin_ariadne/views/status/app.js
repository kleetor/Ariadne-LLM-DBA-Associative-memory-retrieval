/**
 * AstrBot 插件视图：状态、启停、日志、配置与 MCP 片段。
 * 所有后端访问都走 window.AstrBotPluginView bridge，不直接 fetch。
 */
let bridge = window.AstrBotPluginView || window.AstrBotPluginPage

function timeout(ms) {
  return new Promise((_, reject) => window.setTimeout(() => reject(new Error('timeout')), ms))
}

async function waitForBridge(limit = 6000) {
  const started = Date.now()
  let current = window.AstrBotPluginView || window.AstrBotPluginPage
  while (!current && Date.now() - started < limit) {
    await new Promise((resolve) => window.setTimeout(resolve, 100))
    current = window.AstrBotPluginView || window.AstrBotPluginPage
  }
  return current
}

const $ = (id) => document.getElementById(id)
const state = {
  timer: null,
  snippet: null,
  configDirty: false,
  busy: false,
}

function toast(message, timeout = 2600) {
  const node = $('toast')
  node.textContent = message
  node.hidden = false
  window.clearTimeout(toast._timer)
  toast._timer = window.setTimeout(() => {
    node.hidden = true
  }, timeout)
}

function setText(id, value) {
  const node = $(id)
  if (node) node.textContent = value == null ? '-' : String(value)
}

function setDot(cardId, status) {
  const dot = $(cardId)?.querySelector('.dot')
  if (!dot) return
  dot.dataset.state = status
}

function renderService(name, info) {
  const cardId = name === 'mcp' ? 'card-mcp' : 'card-panel'
  const metaId = name === 'mcp' ? 'mcp-meta' : 'panel-meta'
  if (!info) {
    setDot(cardId, 'error')
    setText(metaId, '状态未知')
    return
  }
  if (!info.running) {
    setDot(cardId, 'off')
    const suffix = info.enabled === false ? ' · 未启用' : ''
    setText(metaId, `已停止${suffix} · ${info.host}:${info.port}`)
    return
  }
  const portState = info.healthy ? '端口可达' : '端口不可达'
  setDot(cardId, info.healthy ? 'on' : 'error')
  setText(metaId, `pid ${info.pid || '-'} · ${info.host}:${info.port} · ${portState}`)
}

function renderGraph(graph) {
  if (!graph) return
  setText('graph-nodes', graph.nodes ?? '-')
  setText('graph-edges', graph.edges ?? '-')
  setText('graph-path', graph.error || graph.path || '未找到图谱文件')
}

function fillConfig(config) {
  if (!config || state.configDirty) return
  const setValue = (id, value) => {
    const node = $(id)
    if (node && value != null) node.value = String(value)
  }
  const llm = config.llm || {}
  const emb = config.embedding || {}
  const storage = config.storage || {}
  const mcp = config.mcp || {}
  const panel = config.panel || {}

  setValue('llm-api_key', '')
  $('llm-api_key').placeholder = llm.api_key?.set ? '已保存，留空不修改' : '未设置'
  setValue('llm-api_base', llm.api_base)
  setValue('llm-model', llm.model)

  const mode = emb.mode === 'local' ? 'local' : 'api'
  $('embedding-mode').value = mode
  setValue('embedding-api_key', '')
  $('embedding-api_key').placeholder = emb.api_key?.set ? '已保存，留空不修改' : '未设置'
  setValue('embedding-api_base', emb.api_base)
  setValue('embedding-model', emb.model)

  setValue('storage-graphs_dir', storage.graphs_dir)
  setValue('mcp-port', mcp.port)
  setValue('mcp-token', '')
  $('mcp-token').placeholder = mcp.token?.set ? '已保存，留空不修改' : '可选'

  $('panel-enabled').checked = Boolean(panel.enabled)
  setValue('panel-port', panel.port)
}

function renderStatus(payload) {
  if (!payload) return
  setText(
    'subtitle',
    payload.installed
      ? `${payload.runtime || 'uvx'} 已就绪 · ${payload.graphs_dir || ''}`
      : '未找到 uv：插件用 uvx 隔离运行服务，请先安装 uv',
  )
  renderService('mcp', payload.services?.mcp)
  renderService('panel', payload.services?.panel)
  renderGraph(payload.graph)
  fillConfig(payload.config)
}

async function refreshStatus() {
  const payload = await bridge.apiGet('status')
  renderStatus(payload)
}

async function refreshLogs() {
  const payload = await bridge.apiGet('logs', { limit: 120 })
  const lines = payload?.lines || []
  setText('logs', lines.length ? lines.join('\n') : '暂无日志')
}

async function refreshAll(silent = false) {
  try {
    await refreshStatus()
    await refreshLogs()
  } catch (error) {
    if (!silent) toast(`读取状态失败：${error.message || error}`)
  }
}

async function loadSnippet() {
  try {
    state.snippet = await bridge.apiGet('snippet')
    if (state.snippet?.sse) {
      setText('snippet', JSON.stringify(state.snippet.sse, null, 2))
    }
  } catch (error) {
    setText('snippet', `读取片段失败：${error.message || error}`)
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // iframe 沙箱下 clipboard 可能不可用，退回 textarea
    const area = document.createElement('textarea')
    area.value = text
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand('copy')
    area.remove()
    return ok
  }
}

function bindActions() {
  document.querySelectorAll('button[data-action]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (state.busy) return
      state.busy = true
      const { action, target } = button.dataset
      button.disabled = true
      try {
        const result = await bridge.apiPost('action', { action, target })
        toast(result?.message || '操作完成')
        renderStatus(result?.status)
        await refreshLogs()
      } catch (error) {
        toast(`操作失败：${error.message || error}`)
      } finally {
        state.busy = false
        button.disabled = false
      }
    })
  })
}

function bindConfig() {
  document.querySelectorAll('.form-grid input, .form-grid select').forEach((node) => {
    node.addEventListener('input', () => {
      state.configDirty = true
      setText('config-hint', '有未保存的修改')
    })
  })

  $('save').addEventListener('click', async () => {
    if (state.busy) return
    state.busy = true
    $('save').disabled = true
    const payload = {
      llm: {
        api_base: $('llm-api_base').value.trim(),
        model: $('llm-model').value.trim(),
      },
      embedding: {
        mode: $('embedding-mode').value,
        api_base: $('embedding-api_base').value.trim(),
        model: $('embedding-model').value.trim(),
      },
      storage: { graphs_dir: $('storage-graphs_dir').value.trim() },
      mcp: {
        port: $('mcp-port').value,
      },
      panel: {
        enabled: $('panel-enabled').checked,
        port: $('panel-port').value,
      },
    }
    const llmKey = $('llm-api_key').value.trim()
    if (llmKey) payload.llm.api_key = llmKey
    const embKey = $('embedding-api_key').value.trim()
    if (embKey) payload.embedding.api_key = embKey
    const token = $('mcp-token').value.trim()
    if (token) payload.mcp.token = token

    try {
      const result = await bridge.apiPost('config/save', payload)
      const count = result?.changed?.length || 0
      toast(count ? `已保存 ${count} 项配置，重启服务后生效` : '没有需要保存的修改')
      state.configDirty = false
      setText('config-hint', '留空的密钥表示保持原值')
      $('llm-api_key').value = ''
      $('embedding-api_key').value = ''
      $('mcp-token').value = ''
      renderStatus(result?.status)
    } catch (error) {
      toast(`保存失败：${error.message || error}`)
    } finally {
      state.busy = false
      $('save').disabled = false
    }
  })
}

function bindSnippets() {
  $('copy-sse').addEventListener('click', async () => {
    if (!state.snippet) await loadSnippet()
    const text = JSON.stringify(state.snippet?.sse || {}, null, 2)
    setText('snippet', text)
    toast((await copyText(text)) ? '已复制 SSE 配置' : '复制失败，请手动选择文本')
  })

  $('copy-stdio').addEventListener('click', async () => {
    if (!state.snippet) await loadSnippet()
    const text = JSON.stringify(state.snippet?.stdio || {}, null, 2)
    setText('snippet', text)
    toast((await copyText(text)) ? '已复制 stdio 配置' : '复制失败，请手动选择文本')
  })
}

async function boot() {
  bridge = bridge || (await waitForBridge())
  if (!bridge) {
    setText('subtitle', '未检测到插件视图 bridge：请从 AstrBot 插件详情页的视图入口打开')
    return
  }

  let context
  try {
    context = await Promise.race([bridge.ready(), timeout(6000)])
  } catch {
    setText('subtitle', 'bridge.ready 超时：当前页面可能不是由 AstrBot 仪表盘加载的，请从插件详情页重试')
    return
  }
  const dark = Boolean(context?.isDark)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
  document.title = bridge.t('views.status.title', 'Ariadne 长期记忆')
  setText('heading', bridge.t('views.status.heading', '长期记忆运行面板'))

  bridge.onContext((next) => {
    document.documentElement.dataset.theme = next?.isDark ? 'dark' : 'light'
  })

  bindActions()
  bindConfig()
  bindSnippets()

  $('refresh').addEventListener('click', () => refreshAll())
  $('autorefresh').addEventListener('change', (event) => {
    if (event.target.checked) refreshAll()
  })

  await refreshAll()
  await loadSnippet()
  state.timer = window.setInterval(() => {
    if ($('autorefresh').checked) refreshAll(true)
  }, 5000)

  window.addEventListener('beforeunload', () => window.clearInterval(state.timer))
}

boot().catch((error) => {
  setText('subtitle', `加载失败：${error.message || error}`)
})
