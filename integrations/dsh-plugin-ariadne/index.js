/**
 * 可选的服务侧入口。
 *
 * 目前 bundle 只声明了一个 `@deepseek-ai/dsh-mcp-client` 行（见 cordis.patch.yml），
 * 由 DSH 自己拉起 Ariadne MCP 服务；本模块保留为后续接入 DSH WebUI / 状态工具的位置，
 * 不会在加载时做任何重活，避免影响 DSH 启动。
 */
export const name = 'ariadne-memory-control'

export function apply(ctx) {
  const log = ctx?.logger?.info?.bind(ctx.logger) || console.log
  try {
    log('[ariadne] memory bridge loaded; run `ariadne-dsh status` to inspect the service')
  } catch {
    // 日志失败不应影响 harness
  }
}
