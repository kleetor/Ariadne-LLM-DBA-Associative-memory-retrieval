# Ariadne

**LLM DBA 管理与目的驱动的联想记忆检索系统**

LLM 驱动的记忆图谱构建、检索、可视化全链路管线。

计划与报告等文档均在Plan/目录下,根目录的文档已经过整理，将其交由AI进行分析可快速掌握本系统的功能与实现价值。data/目录下包含一个空数据集sample供初次使用，以及一个预设数据集memory供快速体验系统效果。

后续方向：针对检索算法部分进行深入研究，在不同的使用场景下，基础权重表与其他配合部分或可经过参数调整实现不同的检索效果（如开放分类学权重或可实现知识库查询功能）

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-AGPLv3-blue)](LICENSE)
![Version](https://img.shields.io/badge/Version-0.1.0-lightgrey)

> *"A thread through the labyrinth of memory."*
>
> 记忆迷宫中的阿里阿德涅之线。

**语言**：[中文](README.md) · [English](README.en.md)

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
- [Docker 部署](#docker-部署)
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
| 🔌 MCP 集成        | 9 个工具，支持 stdio / SSE 两种传输       |
| 🖥️ 3D 可视化       | 力导向图、图层过滤、聚焦模式、在线 CRUD          |
| 🗂️ 图谱库          | 多图谱管理与切换，可一键载入 MCP（各自跨重启记住）  |
| 📄 离线导出          | 一键生成自包含 HTML，无需服务器              |

## 架构

```
                        ┌──────────────────────────────────────────┐
                        │              Ariadne 核心链路             │
                        └──────────────────────────────────────────┘

对话日志 ──► DBA 维护 ──► MemoryGraph + VectorStore ──► PAR 检索 ──► StoryRank ──► 回复
   │            │                    │
   │    MaintenanceScheduler        ├──► API Server（HTTP REST + 3D 面板）
   │    （批量异步调度）              ├──► MCP Server（9 tools，stdio / SSE）
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

`data/` 目录就是**图谱库**：目录下所有能被解析成图谱的 `*.yaml` 都会出现在面板的「设置 → 图谱库」中，可切换、导入、导出、删除。仓库自带两份：

- `sample_graph.yaml` —— 空图谱占位（0 节点 / 0 边），首次启动的默认落点；
- `memory_graph.yaml` —— 预设数据集，用于快速体验系统效果。

**用哪份图谱由共享指针 `<图谱目录>/active_graph.json` 记录**：在面板里选过之后，面板与 MCP 重启都会回到那份。所以启动参数 `--yaml` 是**可选**的——它只在没有历史记录时作为「第一次用哪份」的提示，日常换图谱直接在面板里操作即可，不必改配置或重启。

### 一键启动（推荐）

在项目根目录运行 `start_all.py`，一条命令同时启动 3D 面板与 MCP SSE：

```bash
# 1) 复制配置模板并填入模型信息（MCP 会自动读取 .env，无需在命令行重复传参）
cp .env.example .env

# 2) 一键启动（面板 8765 + MCP SSE 8766，端口自动错开）
python start_all.py

# 可视化面板  http://127.0.0.1:8765
# MCP SSE      http://127.0.0.1:8766/sse
```

> 可用 `--yaml` 指定首次启动用哪份图谱（可选，如 `--yaml data/memory_graph.yaml`）；`--graph-dir` 可指定图谱库目录；`--api-port` / `--mcp-port` / `--host` 覆盖默认端口与地址；`Ctrl+C` 同时停止两个服务。

### 入口一：3D 可视化 WebUI

浏览器查看 + 手动 CRUD + 可观测性：

```bash
ariadne-api --port 8765          # 用共享指针里记录的图谱（首次为空图谱占位）
# 或指定首次启动用哪份： ariadne-api --yaml data/memory_graph.yaml --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

功能：

- **图谱**：3D 力导向图（按角色形状/配色、标签、高亮、聚焦、模糊搜索）、图层过滤与过滤预设、节点/边 CRUD（操作自动写回 YAML，支持撤销）。与 MCP 并发写入通过跨进程文件锁（`<yaml>.lock`）串行化「读-改-写」，不会相互覆盖；面板对图谱的增删改会由 MCP 在检索前自动对账到向量索引（以图谱为权威）。
- **图谱库**（设置 → 图谱库）：管理与切换多份图谱。列表只收录图谱目录下**能解析成图谱**的 YAML（评测数据这类同目录 YAML 不会被列出，也就不会被误删）；支持导入为新图谱 / 导出 / 删除（当前正在用的、以及 MCP 正在用的都不允许删）。
  - **切换**只影响面板自身，并把选择记进共享指针，重启后仍回到这份；
  - **载入 MCP**是独立动作：写入切换请求后由 MCP 进程自行应用（默认 3s 内生效），应用过程会**清空并全量重建向量索引**，面板会显示进度与回执。不点这个按钮，MCP 就不会跟着变——两端各自的「当前图谱」互不干扰。
- **可观测**：指标概览（图规模/孤立/请求/运行时长）、操作日志查看器（LLM + DBA，可筛选）、运行日志（面板进程内 logging + MCP 等其它进程的共享日志聚合，见 `ARIADNE_LOG_FILE`）、实时日志流（SSE 推送）。
- **设置**：渲染与布局参数（性能档位/力导向/标签/背景等，浏览器本地持久化）、过滤预设管理、服务端配置只读查看、图谱库、MCP 连接（端点/鉴权/探活/客户端接入片段）、调用测试（在面板内跑一次真实链路：意图识别 → PAR → StoryRank，PAR 经过的节点在 3D 图上以「激活」形式呈现）、图谱 YAML 与操作日志导入导出。

#### 鉴权

面板与 MCP SSE **始终鉴权**，共用同一份凭据文件 `<图谱目录>/auth.json`（随机盐 + PBKDF2-HMAC-SHA256 散列，不可逆；已加入 `.gitignore`）。首次启动自动创建默认账号：

```text
ariadne / ariadne
```

- **首次登录必须改密**：用初始密码（或重置后门）登录后，除改密相关接口外的请求一律 403，前端会自动跳到 `/login?mode=change`。
- **浏览器**：访问受保护页面会 302 到 `/login`，登录成功后下发 7 天有效的签名会话 Cookie（HttpOnly + SameSite=Lax）；顶栏显示当前用户（点击即改密）与登出按钮。登录失败按来源 IP 限速（5 次 / 5 分钟）。
- **改密**：登录页 `/login?mode=change`（顶栏用户名 / 登出按钮旁边进入），可同时改用户名；新密码至少 8 位。
- **脚本 / 第三方客户端**：可直接用 HTTP Basic 调 API，无需走登录页；MCP 还可用 `ARIADNE_MCP_TOKEN` 走 `Authorization: Bearer`。初始密码未修改前 Basic 会被拒（避免绕过强制改密）。
- **MCP SSE**（`--sse`）同样受保护：无凭据连接返回 401，客户端可用 `http://user:pass@host:port/sse` 或 Bearer 头。
- **忘记密码**：用 `.env` 里的 `ARIADNE_VIZ_USER` / `ARIADNE_VIZ_PASS` 登录（重置后门，登录后强制改密），重置完建议把这两行从 `.env` 删掉——它不是日常校验凭据。
- **可选 pepper**：设置 `ARIADNE_AUTH_KEY` 后它参与散列且只存在环境里，`auth.json` 泄漏也无法离线爆破。注意设置后不要随意更改，否则原密码无法校验（走上面的重置后门重设即可）。
- 会话签名密钥默认自动生成在 `<图谱目录>/.ariadne_secret`，也可用 `ARIADNE_SECRET_KEY` 显式指定；密钥不变则重启后登录态保留。

无需鉴权的只有 `/api/health`（探活）与登录流程本身（`/login`、`/api/login`、`/api/logout`、登录页图标）。

### 入口二：MCP Server

供 LLM Agent 调用（完整 DBA 模式，需 LLM + Embedding）：

```bash
ariadne-mcp \
    --llm-model gpt-4o-mini --llm-api-key sk-xxx --llm-base-url https://api.openai.com/v1 \
    --embedding-model text-embedding-3-small
```

> `--yaml` 可选：不传时用共享指针记录的图谱（首次为空图谱占位）。

### 入口三：离线 HTML

无需服务器，直接生成自包含可视化页面（复用 WebUI 前端，只读模式）：

```bash
ariadne-render --yaml data/sample_graph.yaml -o output.html
```

## Docker 部署

同一份镜像同时提供 MCP Server（`ariadne-mcp`）与 3D 面板（`ariadne-api`）两个入口，二者共用同一份 `./data`。

### 方式一：docker compose（推荐）

```bash
cp .env.example .env        # 填入模型信息
docker compose up -d --build
```

- **面板**：`http://<主机>:8765`（`ARIADNE_VIZ_PORT` 可改），默认账号 `ariadne / ariadne`，首次登录强制改密；
- **MCP**：默认**不发布宿主端口**，仅供同一 compose 网络内的容器访问 `http://ariadne-mcp:8766/sse`。若本机 IDE 要直连，把 `docker-compose.yml` 里那段 `ports` 注释打开（已绑定回环，无需开防火墙）；
- 两个服务都挂载 `./data:/app/data`，因此**宿主机的 `./data` 就是图谱库**：在面板里切换图谱会在那里生成/更新 `active_graph.json`。

### 方式二：加载预构建镜像

```bash
docker load -i package/ariadne-mcp.tar

docker run -d --name ariadne-mcp -p 8766:8766 \
  -e OPENAI_API_KEY=sk-xxx -e OPENAI_API_BASE=https://api.deepseek.com/v1 -e OPENAI_MODEL=deepseek-v4-flash \
  -e EMBEDDING_LOCAL=false \
  -e EMBEDDING_API_KEY=sk-xxx -e EMBEDDING_API_BASE=https://api.siliconflow.cn/v1 -e EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5 \
  -v $(pwd)/data:/app/data \
  ariadne-mcp:latest
```

> tar 是 OCI 布局，Docker 25+ / Podman / containerd 均可导入；更老的 Docker 可用 `skopeo copy docker-archive:ariadne-mcp.tar docker-archive:out.tar` 转换一次。

### embedding 模式决定镜像大小

`EMBEDDING_LOCAL` **同时是构建参数与运行时开关，两边必须一致**（取值 `1/true/yes/on` 均认）：

| 模式 | 构建 | 运行时需设置 | 镜像体积 |
| --- | --- | --- | --- |
| **API**（默认） | `--build-arg EMBEDDING_LOCAL=false` | `EMBEDDING_LOCAL=false` + `EMBEDDING_API_KEY/BASE/MODEL` | 约 700 MB |
| **本地** | `--build-arg EMBEDDING_LOCAL=true` | `EMBEDDING_LOCAL=true` | 需额外装 CPU 版 torch，2 GB+ |

> 若只认 `"true"` 而 `.env` 写的是 `1`，构建时会跳过 torch，运行时却按本地模式走，容器一起步就 ImportError——Dockerfile 已按同一套真值判断。
> 本地模式首次运行要下载模型，建议挂载 HF 缓存（compose 已建 `hf-cache` 卷）或预置缓存后设 `HF_HUB_OFFLINE=1` 离线运行。

### 容器内的图谱库

与本地完全一致：用哪份图谱由共享指针决定，图谱目录取 `ARIADNE_YAML` 所在目录，也可用 `ARIADNE_GRAPHS_DIR` 显式指定：

```bash
-e ARIADNE_GRAPHS_DIR=/app/data
```

### 注意

- **不要把 `data/auth.json`、`data/.ariadne_secret` 拷进镜像或卷**：前者是密码散列，后者是会话 Cookie 的签名密钥（泄漏即可伪造登录态）。`.dockerignore` 已排除它们，容器首次启动会自行生成。
- 镜像**不预置凭据**。因此不挂载 `./data` 时，升级镜像会让凭据回到默认账号、并使已有登录态失效（签名密钥重新生成）——这是有意为之。
- 面板的**「调用测试」**（在面板内跑真实链路）需要模型配置，compose 里已给 `ariadne-viz` 注入 `.env`，并设了 `ARIADNE_MCP_URL=http://ariadne-mcp:8766/sse`，让面板的「MCP 连接」区块能探活到另一个容器。**前提是**镜像的 embedding 模式与 `.env` 的 `EMBEDDING_LOCAL` 一致（API 模式的镜像不含 torch，写 `1` 会 ImportError）。

### 更新已有部署

| 场景 | 需要替换 |
| --- | --- |
| 用预构建镜像 | 重新 `docker load` 新 tar；`docker-compose.yml` 若缺 `ariadne-viz` 的 `env_file` / `ARIADNE_MCP_URL`，建议一并替换 |
| 在部署机上构建 | `Dockerfile`、`.dockerignore`、`docker-compose.yml`、`dba_pipeline/`、`data/*.yaml`、`pyproject.toml`、`requirements.txt` |

> **`.dockerignore` 必须一起更新**，否则新构建的镜像会把 `data/` 下的凭据与密钥重新打进去。

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

> 详见 [0828PAR检索理论文.md](0828PAR检索理论文.md)

## MCP 服务详解

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
├── data/                           # 图谱库目录（面板列出其中所有图谱供切换）
│   ├── sample_graph.yaml           # 空图谱占位（0 节点，首次启动的默认落点）
│   ├── memory_graph.yaml           # 预设数据集
│   └── active_graph.json           # 共享指针：面板 / MCP 各自当前的图谱（运行期生成）
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
    ├── viz/                        # WebUI 服务端、静态前端、日志总线、渲染、导出
    │   ├── api_server.py           # Starlette REST + SSE 服务
    │   ├── logbus.py               # 应用日志捕获 + 操作日志 tail + SSE 分发
    │   ├── chain_runner.py         # 面板内复用 MCP 检索链路（调用测试）
    │   ├── renderer.py             # 离线自包含 HTML 生成
    │   ├── exporter.py
    │   └── static/                 # WebUI 前端（index.html / css / js）
    ├── mcp_server.py               # MCP Server 入口
    ├── graphlib.py                 # 图谱库与共享指针（面板与 MCP 共用）
    ├── webauth.py                  # 统一鉴权（会话 Cookie / Basic / Bearer）
    └── loader.py                   # 图 / 查询加载
```

## 参考与论文

- [0828前置数据管理与后处理.md](0828前置数据管理与后处理.md)（前置数据管理 / LLM DBA 理论部分）
- [0828PAR检索理论文.md](0828PAR检索理论文.md)（目的驱动联想记忆检索模型 / 检索理论部分）
- [0828对比评测数据.md](0828对比评测数据.md)（检索层与故事质量层多系统对比测评）

## 许可证

本项目采用 [GNU AGPLv3](LICENSE) 许可证（`AGPL-3.0-only`），版权归 **kleetor** 所有。

> AGPLv3 的**网络使用条款（Section 13）**对本项目适用：当通过远程网络向第三方提供服务（如 MCP SSE / HTTP API / 3D 面板）时，须向服务使用者提供完整源码——本项目源码已在仓库公开。
