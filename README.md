# Ariadne

**LLM DBA 管理与目的驱动的联想记忆检索系统**

LLM 驱动的记忆图谱构建、检索、可视化全链路管线。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![Version](https://img.shields.io/badge/Version-0.1.0-lightgrey)

> *"A thread through the labyrinth of memory."*
>
> 记忆迷宫中的阿里阿德涅之线。

***

## 项目的目标与思考

> 记忆不是单一仓库，而是「事实」与「经历」两种记忆的协奏。

Ariadne 的长期目标是回答一个问题：**Agent 应如何像人类一样，科学、高效地管理与检索记忆和知识？**

认知科学将人类长期记忆区分为两类（Tulving）：

- **语义记忆（Semantic Memory）**——去情境化的稳定知识：世界的事实、概念与规则（"地球绕太阳转""用户是后端工程师"）。它像一座**知识库**，强调准确、一致、可核对。
- **情景记忆（Episodic Memory）**——个人经历的时间-空间-情绪片段（"上周二加班到十二点，脖子很酸"）。它强调**情境、因果、时序与情绪色彩**，允许近似与联想。

人类大脑用不同机制处理这两类记忆，并让它们**相互协作**：语义记忆提供背景知识以理解新经历，新经历又逐渐沉淀为语义知识。Ariadne 认为，LLM Agent 的长期记忆也应遵循同样的分离与合作原则：

- **分离**——知识库（语义）与经历库（情景）分层存储、按需检索，避免"稳定事实"与"个人经历"互相污染：知识要求准确一致，经历允许随时间演变。
- **合作**——语义知识为情景回忆提供解释背景；具体经历反过来验证、修正、固化语义事实；二者通过图谱关系双向转化。

这一思考在 Ariadne 中的落地与未来方向：

| 阶段 | 语义记忆（知识库） | 情景记忆（经历库） |
| --- | --- | --- |
| **现状** | 有向类型化图谱：节点 + 边承载稳定事实，DBA 维护（抽取 / 纠错 / 去重 / 废弃）保证一致性 | 图谱中的个人经历节点，经 StoryRank 整理为故事片段，保留因果 / 时序 / 情绪 |
| **未来** | 显式"知识层"：去情境化事实的独立存储与一致性核对 | 显式"经历层"：以经历为单位的时空-情绪情境检索 |

最终目标是让 Agent 能够：**回忆时像人一样"想到某段经历"，分析时像查库一样"调用某条事实"**——这正是 Ariadne 作为一条穿过记忆迷宫之线的意义。

***

## 目录

- [项目的目标与思考](#项目的目标与思考)
- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构](#架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [检索方法：PAR 基线](#检索方法par-基线)
- [MCP 服务详解](#mcp-服务详解)
- [数据格式](#数据格式)
- [类型体系](#类型体系)
- [项目结构](#项目结构)
- [参考与论文](#参考与论文)
- [许可证](#许可证)

***

## 项目简介

Ariadne 将 LLM 对话中的事实抽取、纠错、去重、废弃等数据库管理（DBA, Database Administration）思想引入记忆系统，把长期记忆建模为一张有向类型化知识图谱，并在此基础上实现了一套目的驱动的联想记忆检索链路（PAR 链路）。

系统的核心主张是：记忆的价值不在于"存得多"，而在于"在需要时以正确的因果结构被唤回"。为此，Ariadne 提供了：

- **自动化记忆维护**：对话日志经 DBA 批量异步调度，抽取为节点与关系，持续纠错、废弃旧记忆；
- **目的驱动检索**：以跳转轴、目的回归、寻峰终止三层机制，在图中沿因果/语义方向有界扩展；
- **故事化输出**：检索结果被 StoryRank 整理为故事片段文档，而非扁平候选列表；
- **多端接入**：3D 可视化面板、[MCP](https://modelcontextprotocol.io/) Server（供 LLM Agent 调用）、离线 HTML 三种形态。

## 核心特性

| 特性               | 说明                              |
| ---------------- | ------------------------------- |
| 🧠 类型化知识图谱       | 6 种节点角色 × 8 种关系类型，边带方向与权重       |
| 🔧 DBA 自动维护      | 节点抽取与边连接两步分离，纠错/废弃 + 批量异步调度降 token |
| 🎯 目的驱动检索        | 跳转轴 + 目的回归 + 寻峰终止，替代固定 top-K    |
| 📖 StoryRank 故事化 | 因果链路 → 故事片段，避免污染聊天上下文           |
| 🔌 MCP 集成        | 6 个工具，支持 stdio / SSE 两种传输       |
| 🖥️ 3D 可视化       | 力导向图、图层过滤、聚焦模式、在线 CRUD          |
| 📄 离线导出          | 一键生成自包含 HTML，无需服务器              |

## 架构

```
                        ┌──────────────────────────────────────────┐
                        │              Ariadne 核心链路             │
                        └──────────────────────────────────────────┘

对话日志 ──► DBA 维护 ──► MemoryGraph + VectorStore ──► PAR 检索 ──► StoryRank ──► 回复
   │            │                    │
   │    MaintenanceScheduler        ├──► API Server（HTTP REST + 3D 面板）
   │    （批量异步调度）              ├──► MCP Server（6 tools，stdio / SSE）
   │                                 └──► 离线 HTML
   └──► 人工干预（CRUD 面板 + MCP dba_intervene）
```

| 层       | 职责              | 主要模块                                                      |
| ------- | --------------- | --------------------------------------------------------- |
| **抽取层** | 对话 → 节点/边，纠错与废弃 | `extraction/dba.py`、`extraction/graph_builder.py`         |
| **存储层** | 图谱 + 向量索引       | `graph/memory_graph.py`、`embedding/store.py`              |
| **检索层** | 有向扩展、目的过滤、寻峰终止  | `core/jump_axis.py`、`core/purpose.py`、`core/peak_find.py` |
| **故事层** | 因果链路 → 故事片段     | `retrieval/retriever.py`（StoryRank）                       |
| **接入层** | API / MCP / 可视化 | `viz/api_server.py`、`mcp_server.py`                       |

## 安装

**环境要求**：Python 3.10+

```bash
# 基础安装
pip install -e .

# 需要本地 embedding 模型时（可选）
pip install -e ".[local]"

# 开发依赖（可选）
pip install -e ".[dev]"
```

## 快速开始

仓库自带样例 `data/sample_graph.yaml`（10 个节点、7 条边）。

### 一键启动（推荐）

在项目根目录运行 `start_all.py`，一条命令同时启动 3D 面板与 MCP SSE：

```bash
# 1) 复制配置模板并填入模型信息（MCP 会自动读取 .env，无需在命令行重复传参）
cp .env.example .env

# 2) 一键启动（面板 8765 + MCP SSE 8766，端口自动错开）
python start_all.py --yaml data/sample_graph.yaml

# 可视化面板  http://127.0.0.1:8765
# MCP SSE      http://127.0.0.1:8766/sse
```

> 可用 `--api-port` / `--mcp-port` / `--host` 覆盖默认端口与地址；`Ctrl+C` 同时停止两个服务。

### 入口一：3D 可视化面板

浏览器查看 + 手动 CRUD：

```bash
ariadne-api --yaml data/sample_graph.yaml --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

功能：3D 力导向图、图层过滤、聚焦模式、模糊搜索、CRUD 面板，操作自动写回 YAML。

### 入口二：MCP Server

供 LLM Agent 调用（完整 DBA 模式，需 LLM + Embedding）：

```bash
ariadne-mcp --yaml data/sample_graph.yaml \
    --llm-model gpt-4o-mini --llm-api-key sk-xxx --llm-base-url https://api.openai.com/v1 \
    --embedding-model text-embedding-3-small
```

### 入口三：离线 HTML

无需服务器，直接生成自包含可视化页面：

```bash
ariadne-render --yaml data/sample_graph.yaml -o output.html
```

## 检索方法：PAR 基线

项目提出的检索方法 **PAR（Proposed）** 是完整链路，由三层机制组成：

1. **跳转轴（Jump Axis）**：边按 8 种关系类型化、节点按 6 种角色分类，用 6×8 权重矩阵做有向扩展，权重为 0 的方向直接阻断，避免无方向扩散。
2. **目的回归（Purpose Regression）**：LLM 推断查询隐含目的并编码为向量，每步扩展过滤偏离目的的候选。
3. **寻峰终止（Peak Finding）**：记录每轮目的关联度，斜率转负时回到峰值容忍带输出，替代固定 top-K。

与基线的对比：

| 方法              | 组成              | 关键局限           |
| --------------- | --------------- | -------------- |
| **A** 纯向量       | 语义 embedding 匹配 | 无方向感，大图退化快     |
| **B** 向量 + 图谱混合 | 无向图谱扩展          | 无向、无增量（实测 A=B） |
| **C** 跳转轴       | 仅有向扩展           | 无目的约束          |
| **PAR** 完整链路      | 有向 + 目的 + 寻峰    | 发散噪声、输出量膨胀     |

核心优势在**联想召回广度**：216 节点上 PAR 的 R@all = 0.538 vs A = 0.159（约 **3.4 倍**），且随规模放大。注意该 R@all 为**宽输出口径**（PAR 输出 6–27 条 vs A 固定 5 条），度量"多输出多命中"的联想广度；固定 top-N 下 PAR 反而更弱（P@5 = 0.160 vs A = 0.267@216）。在配额对齐的跨系统对比中（LiFEMem 322 节点，候选配额对齐 38.3），纯向量（RAG/Mem0）expected 覆盖 0.840 > PAR 0.765——聚焦语义召回纯向量更优；PAR 的独特价值在联想/发散区（纯故事对比 25:5 / 26:4 胜出），两口径对应"联想广度 vs 聚焦召回"两个切面（详见评测部分 §4.8）。

### StoryRank：检索链路的故事化

PAR 检索产出的是由「节点 + 关系」构成的**因果链路**，而非扁平候选列表。StoryRank 在送入回复生成前，把这条链路理解并整理成**故事片段文档**，承担三个职责：

1. **保留因果关系**：关系类型（`CAUSAL` / `PREFERENCE` / `SCENARIO` 等）语义自然融入故事句子。
2. **避免污染聊天上下文**：聊天模型只接收干净故事，而非 `[id] content` 节点列举。
3. **语义过滤**：LLM 按边关系组织故事时，明显突兀、与主线无关的节点被自然舍弃。

入口为 `retrieve_with_story`（库内方法），产物含 `stories`、`story_nodes`（采纳节点）、`discarded_nodes`（舍弃节点）。

> 详见 [Ariadne——LLM DBA 管理与目的驱动的联想记忆检索系统（理论部分）](Ariadne——LLM%20DBA管理与目的驱动的联想记忆检索系统%20理论部分.md)

## MCP 服务详解

基于 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) 开放协议，将 Ariadne 的记忆图谱封装为可被任意 MCP 客户端调用的 Server。协议规范详见 [MCP Specification](https://spec.modelcontextprotocol.io/)。

### 传输模式

| 模式            | 用法                                              | 适用场景                       |
| ------------- | ----------------------------------------------- | -------------------------- |
| **stdio**（默认） | `ariadne-mcp --yaml xxx.yaml`                   | Claude Desktop 等本地拉起进程的客户端 |
| **SSE**       | `ariadne-mcp --yaml xxx.yaml --sse --port 8765` | Cursor 等通过网络 URL 连接的客户端    |

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
| `--vector-index`      | FAISS 索引文件路径（可选），用于恢复已有向量索引 |
| `--restore-dir`       | 从 checkpoint 目录完整恢复（图谱 + 向量 + 构建器 + 调度器状态） |

> 与 `dba_checkpoint` 配合使用：运行期用 `dba_checkpoint` 落盘，启动时用 `--restore-dir` 恢复。

### 启动参数补充

| 参数                  | 说明 |
| --------------------- | ---- |
| `--vector-index`      | FAISS 索引文件路径（可选），用于恢复已有向量索引 |
| `--restore-dir`       | 从 checkpoint 目录完整恢复（图谱 + 向量 + 构建器 + 调度器状态） |

> 与 `dba_checkpoint` 配合使用：运行期用 `dba_checkpoint` 落盘，启动时用 `--restore-dir` 恢复。

### 启动参数补充

| 参数                  | 说明 |
| --------------------- | ---- |
| `--vector-index`      | FAISS 索引文件路径（可选），用于恢复已有向量索引 |
| `--restore-dir`       | 从 checkpoint 目录完整恢复（图谱 + 向量 + 构建器 + 调度器状态） |

> 与 `dba_checkpoint` 配合使用：运行期用 `dba_checkpoint` 落盘，启动时用 `--restore-dir` 恢复。

### 运行模式

当前仅保留**完整 DBA 模式**：需配置 LLM（`--llm-model` 或 `OPENAI_MODEL`）。启动后 `dba_add_conversation` 会调用内部 LLM 完成记忆维护：**节点抽取（Step 1）与边连接（Step 2）是两次独立的 LLM 调用**——先抽节点并落地，再基于「本轮全部新节点 + 相关旧节点/一跳邻居/已有边」连边，避免单次输出受可见节点集合限制（详见 [0822 DBA 抽取环节评测报告](Plan/0822——DBA抽取环节评测报告.md)）。缺少 LLM / DBA 依赖时直接报错退出（不再降级为存根模式）。

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

### 6 个 Tool

| Tool                   | 说明                             |
| ---------------------- | ------------------------------ |
| `dba_add_conversation` | 追加对话，优先进入调度器批量累积，达阈值后异步维护 |
| `dba_query_memory`     | 目的驱动联想检索（PAR 链路：跳转轴 + 目的回归 + 寻峰） |
| `dba_inspect_graph`    | 展开节点 1-hop 邻居                  |
| `dba_intervene`        | 人工 CRUD 节点和边                   |
| `dba_checkpoint`       | 保存完整检查点                        |
| `dba_get_stats`        | 图谱统计信息                         |

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

## 数据格式

图谱以 YAML 作为持久化格式，节点与边分别存放在 `nodes` / `edges` 两个顶层列表中：

```yaml
nodes:
  - id: n1
    content: "用户在互联网公司做后端开发"
    type: status
    deprecated: false
    forgotten: false
  - id: n2
    content: "用户经常加班到晚上十点"
    type: reason
    deprecated: false
    forgotten: false

edges:
  - from: n2
    to: n1
    type: causal
    bidirectional: false
```

| 字段                      | 说明                                             |
| ----------------------- | ---------------------------------------------- |
| `nodes[].type`          | 节点角色类型（见下方 6 种节点类型）                            |
| `nodes[].deprecated`    | 是否已废弃（不再参与检索）                                  |
| `nodes[].forgotten`     | 是否已遗忘（不再参与检索）                                  |
| `edges[].type`          | 关系类型（见下方 8 种边类型）                               |
| `edges[].bidirectional` | 是否双向（`SCENARIO` / `SOCIAL` / `ATTRIBUTE` 默认双向） |

## 类型体系

**节点（6 种）**：

| 类型        | 语义      |
| --------- | ------- |
| `STATUS`  | 状态 / 现状 |
| `REASON`  | 原因      |
| `ACTION`  | 行为 / 动作 |
| `THING`   | 事物 / 对象 |
| `PERSON`  | 人物      |
| `EMOTION` | 情绪      |

**边（8 种）**：

| 类型           | 语义      | 方向   |
| ------------ | ------- | ---- |
| `CAUSAL`     | 因果      | 指向结果 |
| `SCENARIO`   | 场景归属    | 双向   |
| `SEQUENCE`   | 时序先后    | 指向后序 |
| `PREFERENCE` | 态度 / 偏好 | 指向对象 |
| `SOCIAL`     | 社交关系    | 双向   |
| `ATTRIBUTE`  | 属性归属    | 双向   |
| `TEMPORAL`   | 时间定位    | 指向时间 |
| `TAXONOMIC`  | 分类 / 本体 | 指向父类 |

## 项目结构

```
.
├── start_all.py                    # 一键启动（面板 + MCP SSE）
├── data/
│   ├── sample_graph.yaml           # 样例图谱
│   └── memory_graph.yaml           # 实际数据
└── dba_pipeline/
    ├── core/                       # 检索核心：跳转轴、目的回归、寻峰
    │   ├── jump_axis.py
    │   ├── purpose.py
    │   ├── peak_find.py
    │   └── path_tracker.py
    ├── graph/                      # 图结构
    │   └── memory_graph.py
    ├── embedding/                  # 向量存储
    │   └── store.py
    ├── extraction/                 # DBA 抽取与批量维护
    │   ├── dba.py
    │   ├── graph_builder.py
    │   └── maintenance_scheduler.py
    ├── llm/                        # 推理引擎
    │   └── inference.py
    ├── retrieval/                  # PAR 检索 + StoryRank
    │   └── retriever.py
    ├── viz/                        # API Server、3D 渲染、导出
    │   ├── api_server.py
    │   ├── renderer.py
    │   └── exporter.py
    ├── mcp_server.py               # MCP Server 入口
    └── loader.py                   # 图 / 查询加载
```

## 参考与论文

- [Ariadne——LLM DBA 管理与目的驱动的联想记忆检索系统（理论部分）](Ariadne——LLM%20DBA管理与目的驱动的联想记忆检索系统%20理论部分.md)
- [Ariadne——LLM DBA 管理与目的驱动的联想记忆检索系统（评测部分）](Ariadne——LLM%20DBA管理与目的驱动的联想记忆检索系统%20评测部分（0818-0822报告整合）.md)

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
