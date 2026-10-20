# 架构与项目结构

```
.
├── start_all.py                    # 一键启动（面板 + MCP SSE）
├── .github/                        # CI、Issue/PR 模板
├── alm/                            # 同步 HTTP 记忆服务（Add / Search / Health）
├── assets/                         # 图标与示意图
├── data/                           # 图谱库目录（面板列出其中所有图谱供切换）
│   ├── sample_graph.yaml           # 空图谱占位（0 节点，首次启动的默认落点）
│   ├── memory_graph.yaml           # 预设数据集
│   └── active_graph.json           # 共享指针：面板 / MCP 各自当前的图谱（运行期生成）
├── deploy/                         # Dockerfile、Dockerfile.alm 与 compose 文件
├── docs/                           # 文档索引、ALM/MCP/架构说明与理论文档
│   ├── theory/                     # 0828 检索理论与实验文档
│   ├── ALM.md / MCP.md             # 服务说明
│   └── architecture.md             # 本文件
├── examples/                       # ALM / MCP 快速开始示例
├── tests/                          # pytest 测试
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

---

## 数据格式与类型体系

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
