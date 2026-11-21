# MCP 服务详解

基于 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) 开放协议，将 Ariadne 的记忆图谱封装为可被任意 MCP 客户端调用的 Server。协议规范详见 [MCP Specification](https://spec.modelcontextprotocol.io/)。

### 传输模式

| 模式            | 用法                                              | 适用场景                       |
| ------------- | ----------------------------------------------- | -------------------------- |
| **stdio**（默认） | `ariadne-mcp`                                   | Claude Desktop 等本地拉起进程的客户端 |
| **SSE**       | `ariadne-mcp --sse --port 8766`                 | Cursor 等通过网络 URL 连接的客户端    |

> `--yaml` / `--graph-dir` 均为可选：用哪份图谱优先取共享指针里记录的选择，没有记录时才回落到 `--yaml`，再回落到图谱目录内的空图谱占位 `sample_graph.yaml`。

> ⚠️ SSE 默认端口已改为 `8766`（`ariadne-api` 仍为 `8765`），两者不再冲突。

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
EMBEDDING_MODEL=BAAI/bge-m3
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
    --embedding-model BAAI/bge-m3 --embedding-local
```

### 检索参数

运行时参数存放在图谱同目录的 `retrieval_params.yaml`（可用 `--params` 指定），启动时加载，
之后按文件 mtime 热更新；面板与 MCP 共用同一份文件，也可通过 `dba_intervene(action=set_params)` 修改。

| 参数 | 默认 | 说明 |
|---|---|---|
| `seed_k` | 5 | 向量种子数 |
| `max_hops` | 5 | 最大跳数 |
| `expand_k` | 0 | 0 表示与 `seed_k` 相同 |
| `max_peak_nodes` | 0 | 峰值带输出上限，0 表示不限制（作用于检索侧，早于连通性粗筛） |
| `seed_cos_min` | 0 | 种子余弦下限，0 表示不设下限；不足时不补齐 |
| `distance_decay` | 0.85 | 每跳衰减底数 |
| `jump_weight_coef` | 0.5 | 组合分中跳转轴权重 |
| `purpose_weight_coef` | 0.5 | 组合分中目的关联权重 |
| `purpose_filter_threshold` | 0.30 | 目的关联度阈值 |
| `purpose_filter_decay` | 0 | 阈值逐跳衰减，0 表示固定 |
| `peak_tolerance` | 0.10 | 峰值带宽（绝对）：轮均分 ≥ 峰值 − δ 的轮进入输出 |
| `peak_tolerance_ratio` | 0 | 峰值带宽（相对）：>0 时 δ = ratio ×（峰值 − 最低轮均分），与尺度无关 |

两点口径提醒：

- **默认阈值 0.30 在 163 节点主图上不删除任何候选**（0/163），等价于未开启；有效工作点在
  0.45–0.55，且以输出规模换精度（论文 §6.4、§6.10）。不要把它当成"已开启过滤"。
- 论文的实验配置是 `seed_k=5`、`expand_k=5`、`max_hops=5`；ALM 服务使用按图规模自适应的
  seed（`seed_k=40` 上限，163 节点图取 16）与 `expand_k=40`，两者不可直接对比。

结构化目的帧（论文 E11）默认关闭，用环境变量开启：

```bash
ARIADNE_MCP_PURPOSE_FRAME=1 ariadne-mcp --yaml data.yaml ...
# 同时指定三通道权重（嵌入 / 节点类型 / 关系类型）
ARIADNE_MCP_PURPOSE_FRAME=1 ARIADNE_MCP_FRAME_WEIGHTS=0.55,0.35,0.10 \
    ariadne-mcp --yaml data.yaml ...
```

开启后，检索前先推断 purposes 并编译 `PurposeFrame`（内容类型 / 关系类型 / 时段），
显式注入目的会跳过内部推断，不增加 LLM 调用次数。

`ARIADNE_MCP_FRAME_WEIGHTS` 接受逗号或分号分隔的三个非负浮点数，默认 `0.55,0.35,0.10`；
retriever 内部会归一化。格式非法、含负值或和为 0 时告警并退回默认，不会带病启动。

`ARIADNE_MCP_FRAME_ANCHOR`（0–1，默认 0）控制帧的第三维「时段」：0 表示只用目的短语向量，
0.4 表示话题相似 = `0.6 × 目的余弦 + 0.4 × query 原文余弦`。大于 0 时每次检索会多编码一次
query，用于把类型偏好限制在当前情节内（对应论文 E11 的 `a04`）。非法值同样告警退回 0。

`ARIADNE_MCP_FIRST_PEAK=1` 切换寻峰终止：默认（关闭）在显著下降后进入 patience 计时，
期间回升会重置；开启后**首次显著下降立即停**。受控实验（70 条查询，bge-m3）显示输出减少
约 19%、gold recall 降 0.8pp（论文 Limitations 与 `eval/outputs/mechanism/peak_stop_real.json`）。
默认保持关闭，与论文主表口径一致。

### 11 个 Tool

| Tool                   | 说明                             |
| ---------------------- | ------------------------------ |
| `dba_add_conversation` | 追加对话，优先进入调度器批量累积，达阈值后异步维护 |
| `dba_query_memory`     | 目的驱动联想检索（PAR 链路：跳转轴 + 目的回归 + 寻峰） |
| `dba_add_note`         | 写入一条项目笔记（普通节点，可被记忆层召回） |
| `dba_query_knowledge`  | 关键词精确检索项目事实/约定/接口/错误码，返回不含关系边 |
| `dba_temporal_lookup`  | 独立单跳时序查询（仅"时间→事件"反向），用于定位某时间点发生的事 |
| `dba_inspect_graph`    | 展开节点 1-hop 邻居                  |
| `dba_intervene`        | 人工 CRUD 节点和边                   |
| `dba_checkpoint`       | 保存完整检查点                        |
| `dba_get_stats`        | 图谱统计信息（含 notes 计数）           |
| `dba_review_graph`     | 图谱体检（只读）：报出疑似重复节点对与孤立节点，不改图 |
| `dba_review_sources`   | 溯源巡检（只读）：用原始对话核对抽取质量，需 `ARIADNE_SOURCE_STORE=1` |

### 笔记与知识检索（`dba_add_note` / `dba_query_knowledge`）

- 笔记就是普通节点：`metadata.channel="note"`，进向量索引，可被 PAR 当种子；它不是独立存储。
- `dba_add_note(title, text, observed_at?, source_hint?, fidelity?, tags?)`
  - `title` 必填；`text` ≤ 4000 字，超限拒绝并要求拆分；
  - `tags` 用来表达项目归属（如 `project:alpha`），系统不做项目过滤；
  - `source_hint` 只作提示，服务端不会抓取；
  - 完全相同的 `title + text + tags` 返回已有节点（`duplicate=true`）。
- `dba_query_knowledge(query, top_k?)`
  - 关键词匹配：ASCII 词元 + 中文二元组；`title/tags/text` 权重 3/2/1；
  - 返回 `node_id / text / score / observed_at / source_hint / fidelity`，**不含 edge**；
  - 不设阈值、不拒答；无命中返回空列表。
- `ARIADNE_MCP_RERANK=1` 可开启知识层重排，独立于 `ALM_RERANK`；失败静默回退关键词序。

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

### 远程接入与安全

`--sse` 默认只监听 `127.0.0.1:8766`（`--host` 可改），且**始终鉴权**：

- **本机**：`http://127.0.0.1:8766/sse`，用 HTTP Basic 或 `ARIADNE_MCP_TOKEN` 的 Bearer；
- **跨机**：优先走 SSH 隧道 `ssh -L 8766:127.0.0.1:8766 <user>@<host>`，客户端仍填本地地址——明文端口不出网；
- **公网直连**：必须先在前置代理终止 TLS（本服务自身不做 TLS），再把 `--host` 改为 `0.0.0.0` 并限制来源；
- 容器部署时宿主端口默认只绑回环（`ARIADNE_BIND_HOST`），详见 README 的「公网暴露与远程访问」。

工具调用失败时返回 `isError: true`，正文只给 `error_id`（完整堆栈仅进服务端日志与操作日志），便于排障又不泄漏内部信息。
