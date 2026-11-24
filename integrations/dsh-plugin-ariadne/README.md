# dsh-plugin-ariadne

把 [Ariadne](https://www.ariadne.top/) 长期记忆以 **MCP server** 的形式接进 DeepSeek Harness（DSH）。

它不是新的记忆实现：DSH 官方自带 `@deepseek-ai/dsh-mcp-client`，本包做的是**安装 + 配置 + 状态**。

## 安装

```bash
# DSH 的插件就是 npm 包；本地开发时可以直接装目录
npx @deepseek-ai/dsh plugin add ./integrations/dsh-plugin-ariadne
# 发布后：
npx @deepseek-ai/dsh plugin add dsh-plugin-ariadne
```

`package.json` 里的 `dsh.bundle.patch` 会向 profile 插入一行 MCP client 配置
（见 [cordis.patch.yml](./cordis.patch.yml)），stdio 命令为：

```
uvx --from ariadne-mcp-server ariadne-mcp
```

所以需要先装 [uv](https://docs.astral.sh/uv/)；Python 包由 `uvx` 按需拉取，不需要全局安装。

## 一键配置

```bash
ariadne-dsh init
```

交互式填写 LLM / Embedding 与图谱目录，写入 `~/.ariadne-dsh/config.json`（0600）。

如果希望把变量直接写进 DSH profile 的 `.env`：

```bash
ariadne-dsh init --write-env ~/.dsh/profiles/default/.env
```

（路径按你的 DSH profile 实际位置调整；写入内容在 `# >>> ariadne-dsh >>>` 区块内，可重复执行。）

其他命令：

```bash
ariadne-dsh env      # 打印 export / set 行，手动注入当前 shell
ariadne-dsh status   # 基础视图：uv、配置、图谱目录与规模
ariadne-dsh serve    # 前台以 stdio 拉起 ariadne-mcp（排查时用）
```

## 配置字段

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_API_BASE` / `OPENAI_MODEL` | 记忆抽取与维护使用的 LLM |
| `OPENAI_THINKING` | 内部推理开关，默认 `0` |
| `EMBEDDING_LOCAL` | `false` 走 API；`true` 需要本地 torch 与权重 |
| `EMBEDDING_API_KEY` / `EMBEDDING_API_BASE` / `EMBEDDING_MODEL` | 向量模型（默认 SiliconFlow + `BAAI/bge-m3`） |
| `ARIADNE_GRAPHS_DIR` | 图谱目录（YAML、索引、日志、auth.json） |
| `ARIADNE_YAML` | 可选，起始图谱文件 |
| `ARIADNE_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## 关于“基础视图”

DSH 仍是 developer preview，WebUI 扩展 API 变动频繁；本包把稳定的部分放在 CLI：

- `ariadne-dsh status` 提供配置 / 依赖 / 图谱规模的状态卡；
- `ariadne-dsh init` 就是 env 变量的快捷填入；
- DSH 自带的插件配置页 / 日志仍然可用。

后续 DSH 的 webui 插件接口稳定后，`index.js` 会挂一个状态页，不需要改动 MCP 那一侧。

## 边界

- 不复制、不重写任何 MCP 工具；
- 不管理 DSH 的模型与对话，只负责记忆服务的安装与配置；
- 同一个图谱目录不要同时被 DSH 和别的进程写入（stdio 单实例使用）。
