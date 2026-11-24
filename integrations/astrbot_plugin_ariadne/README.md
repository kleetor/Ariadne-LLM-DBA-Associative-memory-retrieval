# astrbot_plugin_ariadne

把 [Ariadne](https://www.ariadne.top/) 长期记忆接入 AstrBot 的插件。

> 官网：<https://www.ariadne.top/>

## Ariadne 是什么

Ariadne 是一套面向个人陪伴场景的长期记忆服务。它把对话中的事实抽取成一张
**有向、带类型的记忆图谱**，并在需要时沿因果与语义关系联想出相关经历，
整理成连贯的故事片段交给模型，而不是返回一堆互不相关的检索片段。

它主要解决三个问题：

- **聊过就忘**：跨会话记不住用户的偏好、关系、经历与计划；
- **记得太碎**：普通 RAG 只按相似度取片段，缺少因果、时间与上下文连续性；
- **越记越乱**：旧事实被新事实覆盖后仍留在记忆里，逐渐互相矛盾。

Ariadne 的做法：

- **自动维护**：对话批量交给 LLM 抽取成节点与关系，持续纠错、去重、废弃过时记忆；
- **目的驱动检索**：从问题出发，在图上沿因果/语义方向有界扩展，再用 StoryRank 整理成故事；
- **重要性判断**：只沉淀长期有效、值得再次想起的事实，不把一次性琐事写进记忆；
- **可视化**：3D 力导向图谱，支持搜索、聚焦、增删改。

## 这个插件为 AstrBot 带来什么

插件本身不重写记忆逻辑，记忆能力仍由独立的 `ariadne-mcp-server` 提供。
它负责把服务在 AstrBot 里跑起来，并提供：

- **长期记忆工具**：AstrBot 的模型可以记录对话、检索回忆、做时间查询、巡检与人工干预；
- **配置表单**：LLM / Embedding / 存储 / MCP / 面板，保存后即可启动；
- **运行状态**：进程、端口、图谱规模、日志、启停与重启；
- **3D 记忆面板**：在插件视图里直接查看整张记忆图谱；
- **MCP 接入片段**：一键复制 SSE 或 stdio 配置。

在 AstrBot 的 MCP 页面加入插件给出的 SSE 地址后，模型就能在聊天中自动记录
值得长期保留的事，并在回答前检索相关记忆。

## 快速开始

1. 在 AstrBot `插件` 页安装本插件（或把源码放到 `data/plugins/` 后重载）。
2. 确认系统里有 [uv](https://docs.astral.sh/uv/)（AstrBot 官方镜像已自带）。
   插件用 `uvx` 在独立虚拟环境运行服务，不往 AstrBot 的 Python 环境装包。
3. 打开插件详情页 → `配置`，填写：
   - `llm.api_key` / `llm.api_base` / `llm.model`
   - `embedding.api_key` / `embedding.api_base` / `embedding.model`（默认 SiliconFlow + `BAAI/bge-m3`）
   - `storage.graphs_dir`（默认插件数据目录下的 `data`）
4. 保存并重载插件。MCP 服务会按 `mcp.auto_start` 自动启动；首次启动由 uvx
   下载依赖，需要能访问 PyPI（国内可设 `UV_DEFAULT_INDEX` 镜像）。

## 视图

插件详情页 → `Ariadne 长期记忆`：

- **状态卡**：MCP / 面板是否运行、pid、端口是否可达；
- **图谱**：节点数、关系数、当前 YAML 路径；
- **启停 / 重启**：只操作插件托管的子进程；
- **接入片段**：一键复制 SSE 或 stdio 配置，粘贴到 `插件 → MCP → 新增服务器`；
- **连接配置**：与 `_conf_schema.json` 同一份配置，密钥留空表示不修改；
- **最近日志**：stderr/stdout 的环形缓冲，同时落盘到插件数据目录的 `mcp.log`。

插件详情页 → `Ariadne 3D 面板`：直接打开原版 3D 记忆图谱，无需额外域名或端口映射。

聊天指令：

```
/ariadne                  # 状态摘要
/ariadne start [mcp|panel]
/ariadne stop [mcp|panel]
/ariadne restart [mcp|panel]
/ariadne logs
```

## MCP 接入

视图里复制 **SSE** 片段：`{"url": "http://127.0.0.1:8766/sse", "transport": "sse"}`。

如果希望 AstrBot 自己拉起服务（不启用本插件的托管），复制 **stdio** 片段：

```json
{
  "command": "<AstrBot 的 python>",
  "args": ["-m", "dba_pipeline.mcp_server", "--graph-dir", "<插件数据目录>/data"],
  "env": {
    "OPENAI_API_KEY": "<YOUR_KEY>",
    "OPENAI_API_BASE": "https://api.deepseek.com/v1",
    "OPENAI_MODEL": "deepseek-v4-flash",
    "EMBEDDING_LOCAL": "false",
    "EMBEDDING_API_KEY": "<YOUR_KEY>",
    "EMBEDDING_API_BASE": "https://api.siliconflow.cn/v1",
    "EMBEDDING_MODEL": "BAAI/bge-m3"
  }
}
```

> 同一份图谱目录不要同时被两个进程写入：SSE 托管与 stdio 自拉二选一。

## 数据与安全

- 图谱、`auth.json`、日志、原文落盘都在 `storage.graphs_dir`（默认插件数据目录），重装插件不会丢。
- `mcp.token`、`panel.password`、两个 API Key 在表单里是密码框；视图只显示“是否已设置”。
- 默认只绑 `127.0.0.1`。改成 `0.0.0.0` 前请确认已有 TLS 反向代理与鉴权。
- 面板密码只是重置后门；日常凭据以图谱目录下 `auth.json` 为准。

## 3D 面板

插件自带一个 `panel` 视图：面板页面与静态资源随插件放在 `views/panel/`，由 AstrBot
插件视图直接提供；浏览器内的请求经 AstrBot 官方 bridge 交给父页面，再打到插件的
`/proxy`、`/stream`、`/download` 三个端点：

- 浏览器只需要能打开 AstrBot WebUI（IP、内网、域名都行），**不需要额外域名、端口映射或宿主机 nginx**；
- 鉴权直接复用 AstrBot Dashboard 登录态；插件再用自动生成并保存的 `mcp.token`（Bearer）访问容器内的面板服务；
- 面板进程仍然只监听 `127.0.0.1:8765`，不对外暴露。

> 面板 API 的 Bearer 鉴权需要 `ariadne-mcp-server >= 0.1.8`（`ARIADNE_MCP_TOKEN`）。
> PyPI 上的 0.1.7 已被 yank，`uvx` 会自动回退到 0.1.6；请发布并使用 0.1.8。

## 版本与升级

插件默认用 `uvx --from ariadne-mcp-server ...` 取 PyPI 最新版。需要固定版本，或
指向包含修复的源码提交时，在 AstrBot 容器里设置 `ARIADNE_UVX_FROM`，再重启 AstrBot
（或重载插件，让子进程用新值重新启动）：

```bash
# 固定版本（发布后可用）
ARIADNE_UVX_FROM=ariadne-mcp-server==0.1.8

# 或直接指向包含修复的 Git 提交
ARIADNE_UVX_FROM=git+https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval@<commit>
```

> 服务器上手工修改 `uvx` 缓存的做法只对当前容器有效，不能复制给其他人；
> 其他人请使用新版本，或用 `ARIADNE_UVX_FROM` 指向含修复的源码。

## 边界

- 本插件不是 MCP 的替代实现，不注册任何额外工具；
- 不做 A/B 测试或图谱编辑：这些在 Ariadne 面板 / MCP 工具里完成；
- AstrBot 版本要求 `>=4.17.0`（插件视图 API）。
