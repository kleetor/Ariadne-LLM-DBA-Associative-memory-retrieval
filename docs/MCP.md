# MCP 服务详解

基于 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) 开放协议，将 Ariadne 的记忆图谱封装为可被任意 MCP 客户端调用的 Server。协议规范详见 [MCP Specification](https://spec.modelcontextprotocol.io/)。

### 传输模式

| 模式            | 用法                                              | 适用场景                       |
| ------------- | ----------------------------------------------- | -------------------------- |
| **stdio**（默认） | `ariadne-mcp`                                   | Claude Desktop 等本地拉起进程的客户端 |
| **SSE**       | `ariadne-mcp --sse --port 8766`                 | Cursor 等通过网络 URL 连接的客户端    |

> `--yaml` / `--graph-dir` 均为可选：用哪份图谱优先取共享指针里记录的选择，没有记录时才回落到 `--yaml`，再回落到图谱目录内的空图谱占位 `sample_graph.yaml`。

> ⚠️ SSE 默认端口 `8765` 与 `ariadne-api` 相同，同时运行需改端口（如 `--port 8766`）。一键启动（`start_all.py`）会自动错开为 8766。

### 环境变量配置（可选）

LLM / Embedding 配置除命令行传参外，也可写入项目根 `.env`（或系统环境变量），启动时自动读取；**命令行参数优先级更高**。

| 命令行参数                  | 环境变量                 |
| ---------------------- | -------------------- |
| `--llm-model`          | `OPENAI_MODEL`       |
| `--llm-api-key`        | `OPENAI_API_KEY`     |
| `--llm-base-url`       | `OPENAI_API_BASE`    |
| `--embedding-model`    | `EMBEDDING_MODEL`    |
| `--embedding-api-key`  | `EMBEDDING_API_KEY`  |
| `--embedding-base-url` | `EMBEDDING_API_BASE` |
| `--embedding-local`    | `EMBEDDING_LOCAL`    |

```bash
# 项目根 .env 示例
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_LOCAL=true
```

> 设置 `EMBEDDING_LOCAL=true` 需先安装本地依赖 `pip install -e ".[local]"`，否则启动时报错退出。

### 启动参数补充

| 参数                  | 说明 |
| --------------------- | ---- |
| `--yaml`              | 初始图谱路径（**可选**）。只在共享指针没有记录时作为「第一次用哪份」的提示 |
| `--graph-dir`         | 图谱库目录（可选）。默认取 `--yaml` 所在目录，都没有则用 `./data`；也可用 `ARIADNE_GRAPHS_DIR` 指定 |
| `--vector-index`      | FAISS 索引文件路径（可选），用于恢复已有向量索引 |
| `--restore-dir`       | 从 checkpoint 目录完整恢复（图谱 + 向量 + 构建器 + 调度器状态） |

> 与 `dba_checkpoint` 配合使用：运行期用 `dba_checkpoint` 落盘，启动时用 `--restore-dir` 恢复。

### 图谱切换（面板 ↔ MCP）

面板与 MCP 是两个进程：面板的 SSE 服务只暴露 MCP 协议端点，MCP 也不暴露控制接口，因此二者通过**共享指针文件**协作：

```text
<图谱目录>/active_graph.json     # 请求与回执（panel_active / active / generation）
<图谱目录>/active_graph.lock     # 跨进程读写锁
```

- 面板里点「切换」→ 只切面板自己，并记 `panel_active`；
- 面板里点「载入 MCP」→ 递增 `generation` 登记请求，MCP 轮询到后自行切换并**全量重建向量索引**，结果写回同一文件供面板展示；
- 两端启动时都优先读各自字段，因此都跨重启记住上次的选择；轮询间隔可用 `ARIADNE_GRAPH_POLL` 调整（默认 3 秒）。

> 图谱库限制在**同一目录**内：`auth.json`、`.ariadne_secret`、`operations.log`、`ariadne.log`、`retrieval_params.yaml` 都按图谱目录存放，同目录切换因此不会影响登录态与运行时参数。

### 运行模式

当前仅保留**完整 DBA 模式**：需配置 LLM（`--llm-model` 或 `OPENAI_MODEL`）。启动后 `dba_add_conversation` 会调用内部 LLM 完成记忆维护：**节点抽取（Step 1）与边连接（Step 2）是两次独立的 LLM 调用**——先抽节点并落地，再基于「本轮全部新节点 + 相关旧节点/一跳邻居/已有边」连边，避免单次输出受可见节点集合限制。缺少 LLM / DBA 依赖时直接报错退出（不再降级为存根模式）。

> Agent 客户端的 LLM 与 Ariadne 内部的维护 LLM 是**两个独立模型**：Agent 的 LLM 负责理解意图、调用工具；Ariadne 的 LLM 负责把对话变成图谱事实。两者通过图谱（而非上下文窗口）共享记忆。

### Embedding 三种配置

| 模式          | 参数                                        | 说明                                 |
| ----------- | ----------------------------------------- | ---------------------------------- |
| **API**（默认） | `--embedding-model`                       | 调用任意 OpenAI 兼容 `/v1/embeddings` 端点 |
| **本地**      | `--embedding-model ... --embedding-local` | 使用 sentence-transformers，无需网络      |
| **不使用**     | 不传 `--embedding-model`                    | 仅图谱操作，不启用向量检索                      |

`--embedding-base-url` / `--embedding-api-key` 未指定时，自动复用 `--llm-base-url` / `--llm-api-key`。

```bash
# API embedding（OpenAI）
ariadne-mcp --yaml data.yaml --llm-model gpt-4o-mini --llm-api-key sk-xxx \
    --llm-base-url https://api.openai.com/v1 --embedding-model text-embedding-3-small

# 本地 embedding
ariadne-mcp --yaml data.yaml --llm-model gpt-4o-mini --llm-api-key sk-xxx \
    --embedding-model BAAI/bge-large-zh-v1.5 --embedding-local
```

### 9 个 Tool

| Tool                   | 说明                             |
| ---------------------- | ------------------------------ |
| `dba_add_conversation` | 追加对话，优先进入调度器批量累积，达阈值后异步维护 |
| `dba_query_memory`     | 目的驱动联想检索（PAR 链路：跳转轴 + 目的回归 + 寻峰） |
| `dba_temporal_lookup`  | 独立单跳时序查询（仅"时间→事件"反向），用于定位某时间点发生的事 |
| `dba_inspect_graph`    | 展开节点 1-hop 邻居                  |
| `dba_intervene`        | 人工 CRUD 节点和边                   |
| `dba_checkpoint`       | 保存完整检查点                        |
| `dba_get_stats`        | 图谱统计信息                         |
| `dba_review_graph`     | 图谱体检（只读）：报出疑似重复节点对与孤立节点，不改图 |
| `dba_review_sources`   | 溯源巡检（只读）：用原始对话核对抽取质量，需 `ARIADNE_SOURCE_STORE=1` |

### 检索说明（`dba_query_memory`）

- 走完整 PAR 链路：跳转轴有向扩展 + 目的回归过滤 + 寻峰终止。
- `rerank_k` 控制返回条数上限（rank-k，默认 20）；`total_matched` 是真实候选总数，`returned` 是本次实际返回数。若 `total_matched > returned`，可调大 `rerank_k` 取回剩余记忆。
- 已废弃/遗忘的节点会被过滤，不出现在结果中。

### Agent 接入配置

**Claude Desktop（stdio）**：

```json
{
  "mcpServers": {
    "ariadne": {
      "command": "python",
      "args": ["-m", "dba_pipeline.mcp_server", "--yaml", "/path/to/memory_graph.yaml"],
      "cwd": "/path/to/dba_pipeline"
    }
  }
}
```

**Cursor / SSE 客户端**（先启动 Server，再填 URL）：

```bash
# 推荐：一键启动（面板 8765 + MCP 8766），模型配置从 .env 读取
python start_all.py --yaml /path/to/memory_graph.yaml

# 或单独启动 SSE（模型配置同样从 .env 读取，也可用命令行覆盖）
ariadne-mcp --yaml /path/to/memory_graph.yaml --sse --port 8766
```

```json
{
  "mcpServers": {
    "ariadne": { "url": "http://127.0.0.1:8766/sse" }
  }
}
```
