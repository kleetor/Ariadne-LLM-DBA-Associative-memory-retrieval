# Ariadne

**LLM DBA 管理与目的驱动的联想记忆检索系统**

Ariadne 把 LLM 对话中的事实抽取、纠错、去重与废弃（DBA）引入记忆系统，将长期记忆建模为**有向类型化知识图谱**，
并在其上实现**目的驱动的联想检索（PAR）**：检索结果不是扁平候选列表，而是经 StoryRank 整理的因果故事片段。

```bash
pip install -c constraints.txt -e .
cp .env.example .env        # 填入 LLM / Embedding 配置
python start_all.py         # 面板 127.0.0.1:8765 + MCP SSE 127.0.0.1:8766
```

`docs/` 已按主题整理，可直接交给 AI 快速掌握系统全貌；`data/` 含空数据集 `sample`（首次启动的默认落点）
与预设数据集 `memory`（用于快速体验效果）。

检索权重表是可调的：不同的使用场景可以通过调整基础权重与配套参数得到不同的检索行为（详见 [docs/architecture.md](docs/architecture.md)）。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-orange)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-AGPLv3-blue)](LICENSE)
![Version](https://img.shields.io/badge/Version-0.1.5-lightgrey)

> *"A thread through the labyrinth of memory."*
>
> 记忆迷宫中的阿里阿德涅之线。

**语言**：[中文](README.md) · [English](README.en.md)

**文档**：[文档索引](docs/README.md) · [ALM](docs/ALM.md) · [MCP](docs/MCP.md) · [架构](docs/architecture.md)

***

## 项目的目标与思考

> 记忆不是一块可检索的硬盘，而是一个人之所以成为他自己的东西。

Ariadne 想做的不是一个更好的知识库，而是**给 Agent 一副属于它自己的记忆**：它记得谁对它说过什么、那件事让它怎么想，并在相处中逐渐认识对面这个人。

### 为什么把目标收窄到「个人」

早期我们用 Tulving 的语义记忆 / 情景记忆二分，对照"知识库"与"记忆系统"：稳定事实要求准确一致，个人经历允许情境、因果与情绪。这套分层在工程上成立，但经过 ALM 链路的开发与 MCP 的几轮更迭，我们得到一个更朴素的结论：

**记忆系统的适配性，与原始数据的相关程度高度相关。** 企业、团队、个人三类场景的语料密度、事实稳定性与隐私边界完全不同，同一套抽取与检索策略落到不同场景，效果与落地难度会拉开显著差距——想同时覆盖三者，往往哪一类都做不深。

因此 Ariadne 把目标收窄到一件事：**单一用户的个人聊天与情感陪伴。**

### PAR 的本质：类 LLM 的加权与预测

Ariadne 的检索算法（PAR）与配套工程实现，本质上是一个**类 LLM 的加权与预测系统**：

- LLM 先把原始对话处理成**分类节点**（类型化的事实与经历），并为它们建立类型化的关系；
- 节点类型 × 边类型构成**权重表**，为向量召回提供了另一个维度——它承载的是 LLM 在海量预训练中理解并浓缩下来的因果关系、场景关联与人物关系，这些先验并不来自 embedding 空间本身；
- 在 query 向量的导向下，检索不止于"召回节点"，还会在已召回的结构上**继续预测关系召回**，沿因果链与时序链把下一步该被想起的东西带出来。

由此换来两件事：LLM 能**快速建模并理解一个具体的人**；用户也能把 LLM 当作**可靠伙伴**，而不是一个每次都要重新自我介绍的客服。

### 工程理念：把记忆交还给 Agent 自己

Ariadne 在实现上坚持一件事：**记忆不属于某个 user，也不属于某个 session，它属于 Agent 自己。**

常见的做法是把记忆做成"为某个用户维护的档案"，检索时按 user_id 过滤；Ariadne 让 Agent 自己持有并维护这份图谱——它记的是"我经历过的对话"，而不是"关于用户的数据"。我们认为这是拟人化更关键的一步：只有记忆成为"我的"，回忆才会有立场、有连续性、有性格。

### 未来

Ariadne 希望 Agent 能逐步理解人类复杂的情感：通过长期陪伴，理解一个人的性格、缺陷与精神，并在对方需要时提供引导与关爱。这是我们从"记住事实"走向"理解人"的长期方向。

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
- [MCP 服务](#mcp-服务)
- [ALM 同步 HTTP 服务](#alm-同步-http-服务)
- [数据格式与类型体系](#数据格式与类型体系)
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
| 🔌 MCP 集成        | 11 个工具（含笔记与知识检索），支持 stdio / SSE 两种传输 |
| ☁️ ALM 同步 HTTP   | `/add` `/search` 契约供外部程序调用，按 `user_id` 隔离空间 |
| 🔐 统一鉴权        | 面板与 MCP 共用 `auth.json`，Cookie / Basic / Bearer；改密即吊销旧会话 |
| 🖥️ 3D 可视化       | 力导向图、图层过滤、聚焦模式、在线 CRUD          |
| 🗂️ 图谱库          | 多图谱管理与切换，可一键载入 MCP（各自跨重启记住）  |
| 📄 离线导出          | 一键生成自包含 HTML，无需服务器              |

## 架构

<p align="center">
  <a href="https://raw.githubusercontent.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/main/assets/diagrams/architecture.png">
    <img src="https://raw.githubusercontent.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/main/assets/diagrams/architecture.png" width="100%" alt="Ariadne 架构：写入链路、存储与状态、读取链路">
  </a>
</p>

写入把对话先放进缓冲，由后台 worker 批量维护（triage → 抽取 → 连边 → 去重/废弃）；读取沿「向量种子 → 有向扩展 → 目的回归 → 寻峰 → StoryRank」走完整条链。MCP 与面板共用同一份 YAML 和跨进程文件锁，ALM 则按 `user_id` 各持一份空间。笔记同时进入向量库与关键词倒排，但只作检索种子，不参与边连接。

| 层       | 职责              | 主要模块                                                      |
| ------- | --------------- | --------------------------------------------------------- |
| **抽取层** | 对话 → 节点/边，纠错与废弃 | `extraction/dba.py`、`extraction/graph_builder.py`         |
| **存储层** | 图谱 + 向量索引       | `graph/memory_graph.py`、`embedding/store.py`              |
| **检索层** | 有向扩展、目的过滤、寻峰终止  | `core/jump_axis.py`、`core/purpose.py`、`core/peak_find.py` |
| **故事层** | 因果链路 → 故事片段     | `retrieval/retriever.py`（StoryRank）                       |
| **知识层** | 笔记关键词精筛          | `kwindex.py`（倒排表，无向量依赖）                              |
| **接入层** | API / MCP / 可视化 | `viz/api_server.py`、`mcp_server.py`                       |

## 安装

**环境要求**：Python 3.10+

```bash
# 基础安装（constraints.txt 提供安全版本下限）
pip install -c constraints.txt -e .

# 需要本地 embedding 模型时（可选）
pip install -c constraints.txt -e ".[local]"

# 开发依赖（可选）
pip install -c constraints.txt -e ".[dev]"
```

> 发布到 PyPI 时包名为 `ariadne-mcp-server`（`pip install ariadne-mcp-server`）；命令仍是 `ariadne-api` / `ariadne-mcp`，不需要改启动方式。

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
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
```

- **面板**：`http://127.0.0.1:8765`（宿主端口默认只绑回环；远程用 SSH 隧道 `ssh -L 8765:127.0.0.1:8765 <user>@<主机>` 访问，`ARIADNE_VIZ_PORT` / `ARIADNE_BIND_HOST` 可改），默认账号 `ariadne / ariadne`，首次登录强制改密；
- **MCP**：宿主端口同样默认只绑回环（`127.0.0.1:8766`），同一 compose 网络内的容器直接访问 `http://ariadne-mcp:8766/sse`。面板/MCP 是 HTTP 明文 + 密码/Basic 鉴权，要暴露公网需先在 .env 设 `ARIADNE_BIND_HOST=0.0.0.0`——仅限前面有 TLS 终止代理的场景；
- 两个服务都挂载 `./data:/app/data`，因此**宿主机的 `./data` 就是图谱库**：在面板里切换图谱会在那里生成/更新 `active_graph.json`。

### 方式二：加载预构建镜像

```bash
docker load -i package/ariadne-mcp.tar

docker run -d --name ariadne-mcp -p 127.0.0.1:8766:8766 \
  -e OPENAI_API_KEY=sk-xxx -e OPENAI_API_BASE=https://api.deepseek.com/v1 -e OPENAI_MODEL=deepseek-v4-flash \
  -e EMBEDDING_LOCAL=false \
  -e EMBEDDING_API_KEY=sk-xxx -e EMBEDDING_API_BASE=https://api.siliconflow.cn/v1 -e EMBEDDING_MODEL=BAAI/bge-m3 \
  -v $(pwd)/data:/app/data \
  ariadne-mcp:latest
```

> tar 是 OCI 布局，Docker 25+ / Podman / containerd 均可导入；更老的 Docker 可用 `skopeo copy docker-archive:ariadne-mcp.tar docker-archive:out.tar` 转换一次。

### 方式三：ALM 同步 HTTP 服务

`alm/` 是独立的对外同步接口（`POST /add`、`POST /search`、`GET /health`），配置与根目录 `.env` **完全隔离**：

```bash
cp alm/.env.alm.example alm/.env.alm   # 必填 ALM_API_KEYS / LLM / Embedding
docker compose --env-file alm/.env.alm -f deploy/docker-compose.alm.yml up -d --build
python -m alm.status --base-url http://127.0.0.1:8770 --api-key <your-key>   # 只读自检
```

- 宿主端口默认只绑 `127.0.0.1:8770`；公网访问需前置反向代理终止 TLS（本服务自身不做 TLS）；
- `ALM_API_KEYS` 为空且监听非回环地址时**拒绝启动**（不是放行所有请求）；
- 记忆落在具名卷 `alm-data`，容器重启 / 升级不丢；
- 两层架构、`user_id` 空间隔离与调参项见 [docs/ALM.md](docs/ALM.md)。

### embedding 模式决定镜像大小

`EMBEDDING_LOCAL` **同时是构建参数与运行时开关，两边必须一致**（取值 `1/true/yes/on` 均认）：

| 模式 | 构建 | 运行时需设置 | 镜像体积 |
| --- | --- | --- | --- |
| **API**（默认） | `--build-arg EMBEDDING_LOCAL=false` | `EMBEDDING_LOCAL=false` + `EMBEDDING_API_KEY/BASE/MODEL` | 约 700 MB |
| **本地** | `--build-arg EMBEDDING_LOCAL=true` | `EMBEDDING_LOCAL=true` | 需额外装 CPU 版 torch，2 GB+ |

> 若只认 `"true"` 而 `.env` 写的是 `1`，构建时会跳过 torch，运行时却按本地模式走，容器一起步就 ImportError——`deploy/Dockerfile` 已按同一套真值判断。
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

### 公网暴露与远程访问

面板与 MCP 都是**HTTP 明文 + 密码 / Basic 鉴权**，凭据在链路上可被嗅探，因此宿主端口**默认只绑 `127.0.0.1`**（由 `ARIADNE_BIND_HOST` 控制）：

| 场景 | 做法 |
| --- | --- |
| 本机浏览器 / 本机 IDE | 直接访问 `http://127.0.0.1:8765`、`http://127.0.0.1:8766/sse` |
| 远程看面板 | SSH 隧道 `ssh -L 8765:127.0.0.1:8765 <user>@<host>`，浏览器开 `http://127.0.0.1:8765` |
| 远程接 MCP | 同一隧道追加 `-L 8766:127.0.0.1:8766`，或让客户端直连反代后的 HTTPS 域名 |
| 对外正式服务 | 前置 nginx / Caddy 终止 TLS，再设 `ARIADNE_BIND_HOST=0.0.0.0`，并为面板与 MCP 各配一个 vhost |

> 容器之间互访（面板的「MCP 连接」探活）走 compose 内部网络 `http://ariadne-mcp:8766/sse`，不受绑定地址影响。ALM 同理：默认只绑回环，公网需反代 + 非空 `ALM_API_KEYS`。

### 更新已有部署

| 场景 | 需要替换 |
| --- | --- |
| 用预构建镜像 | 重新 `docker load` 新 tar；`deploy/docker-compose.yml` 若缺 `ariadne-viz` 的 `env_file` / `ARIADNE_MCP_URL`，建议一并替换 |
| 在部署机上构建 | `deploy/Dockerfile`、`.dockerignore`、`deploy/docker-compose.yml`、`dba_pipeline/`、`data/*.yaml`、`pyproject.toml` |
| ALM 服务 | `deploy/Dockerfile.alm`、`deploy/docker-compose.alm.yml`、`alm/`、`alm/.env.alm.example` |

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

核心优势在**联想召回广度**：216 节点上 PAR 的 R@all = 0.538 vs A = 0.159（约 **3.4 倍**），且随规模放大。注意该 R@all 为**宽输出口径**（PAR 输出 6–27 条 vs A 固定 5 条），度量"多输出多命中"的联想广度；固定 top-N 下 PAR 反而更弱（P@5 = 0.160 vs A = 0.267@216）。在配额对齐的跨系统对比中（LiFEMem 322 节点，候选配额对齐 38.3），纯向量（RAG/Mem0）expected 覆盖 0.840 > PAR 0.765——聚焦语义召回纯向量更优；PAR 的独特价值在联想/发散区（纯故事对比 25:5 / 26:4 胜出），两口径对应"联想广度 vs 聚焦召回"两个切面。

### StoryRank：检索链路的故事化

PAR 检索产出的是由「节点 + 关系」构成的**因果链路**，而非扁平候选列表。StoryRank 在送入回复生成前，把这条链路理解并整理成**故事片段文档**，承担三个职责：

1. **保留因果关系**：关系类型（`CAUSAL` / `PREFERENCE` / `SCENARIO` 等）语义自然融入故事句子。
2. **避免污染聊天上下文**：聊天模型只接收干净故事，而非 `[id] content` 节点列举。
3. **语义过滤**：LLM 按边关系组织故事时，明显突兀、与主线无关的节点被自然舍弃。

入口为 `retrieve_with_story`（库内方法），产物含 `stories`（叙事文本）、`story_used`（正文实际覆盖的节点数，**后端计算**）、`story_degraded`（是否为降级拼接）。

> 详见 [0828PAR检索理论文档.md](docs/theory/0828PAR检索理论文档.md)

## MCP 服务

MCP Server 将记忆图谱暴露为 11 个工具，支持 stdio 与 SSE 两种传输模式。

其中 `dba_add_note` 写入项目笔记、`dba_query_knowledge` 用关键词做精确检索。笔记是普通节点（`metadata.channel="note"`），不是独立存储，因此既能被知识层按字面精筛，也能被记忆层按语义召回。

- 工具清单、环境变量、图谱切换与 Agent 接入：[docs/MCP.md](docs/MCP.md)
- 一键启动：`python start_all.py`

## ALM 同步 HTTP 服务

面向外部程序的同步接口：`POST /add`（写入）、`POST /search`（检索）、`GET /health`（探活）。
按 `user_id` 隔离空间，并采用「原文层 L1 + 图谱层 L2」两层架构：长文档只切块入原文层、不建图，
会话型语料才抽取成节点与边，把 LLM 开销压在划算的地方。

- 架构差异、两层路由与调参项：[docs/ALM.md](docs/ALM.md)
- 容器部署见上文 [Docker 部署](#docker-部署) 的方式三；配置模板 `alm/.env.alm.example`

## 数据格式与类型体系

图谱 YAML 结构、6 类节点与 8 类关系的完整定义见
[docs/architecture.md](docs/architecture.md#数据格式与类型体系)。

## 项目结构

完整目录与模块职责见 [docs/architecture.md](docs/architecture.md)。核心组成：

- `dba_pipeline/`：检索、抽取、图谱、LLM、WebUI 与 MCP Server
- `alm/`：同步 HTTP 记忆服务
- `deploy/`：两套容器化文件（`Dockerfile` + `docker-compose.yml` 供 MCP/面板，`Dockerfile.alm` + `docker-compose.alm.yml` 供 ALM）
- `packaging/`：发行打包（npm 薄壳、quickstart 启动脚本）
- `data/`：图谱库与示例数据
- `tests/`：pytest 测试

## 参考与论文

- [0828前置数据管理与后处理.md](docs/theory/0828前置数据管理与后处理.md)（前置数据管理 / LLM DBA 理论部分）
- [0828PAR检索理论文档.md](docs/theory/0828PAR检索理论文档.md)（目的驱动联想记忆检索模型 / 检索理论部分）
- [0908e2e实验文档.md](docs/theory/0908e2e实验文档.md)（端到端实验记录）

## 许可证

本项目采用 [GNU AGPLv3](LICENSE) 许可证（`AGPL-3.0-only`），版权归 **kleetor** 所有。

> AGPLv3 的**网络使用条款（Section 13）**对本项目适用：当通过远程网络向第三方提供服务（如 MCP SSE / HTTP API / 3D 面板）时，须向服务使用者提供完整源码——本项目源码已在仓库公开。
