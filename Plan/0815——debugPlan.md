# Ariadne 系统问题清单与修复计划

> 来源：对当前代码库的一次静态分析。
> 本文档按 P0/P1/P2/P3 优先级排列，记录问题、影响、定位与修复建议，并给出增量实施顺序。

***

## 一、问题总览

| 编号 | 优先级 | 问题                                | 模块                                | 性质           |
| -- | --- | --------------------------------- | --------------------------------- | ------------ |
| 1  | P0  | YAML 加载丢失 deprecated/forgotten 字段 | loader.py                         | 数据一致性 / 状态丢失 |
| 2  | P1  | PathTracker 双因子动态权重未接入运行链路        | path\_tracker.py / mcp\_server.py | 核心机制未生效      |
| 3  | P1  | 调度器自动检查点未真正启动                     | maintenance\_scheduler.py         | 功能未生效        |
| 4  | P2  | 未启用 embedding 时语义去重静默失效           | graph\_builder.py                 | 正确性隐患        |
| 5  | P2  | FAISS L2 距离被近似当作余弦相似度             | graph\_builder.py                 | 算法近似         |
| 6  | P2  | PeakFinder 默认参数与实际使用不一致           | peak\_find.py / retriever.py      | 一致性          |
| 7  | P3  | 可视化 API 使用单线程 HTTPServer          | api\_server.py                    | 工程化 / 并发     |
| 8  | P3  | 缺少测试、评测脚本与实验数据                    | tests/                            | 工程化          |

***

## 二、问题详情

### P0-1：YAML 加载丢失 deprecated/forgotten 字段

- **定位**：[loader.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/loader.py#L38-L64) `load_graph()`
- **现象**：只读取 `id` / `content` / `type` 三个字段，忽略 `deprecated` / `forgotten`。
  - 注：`metadata` 本身不参与 `to_dict()` 序列化（[memory\_graph.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/graph/memory_graph.py#L185-L193) 未写出 `metadata`），故不属于往返丢失；但若手写 YAML 含 `metadata`，`load_graph()` 同样会忽略。
- **影响**：
  - `_save()` 通过 `to_dict()` 会写出 `deprecated` / `forgotten`，但再次加载时被丢弃，导致「废弃 / 遗忘」状态在重启后丢失。
  - API Server、MCP Server、渲染器均使用 `load_graph()`，影响面覆盖所有入口。
- **修复建议**：
  - 在 `load_graph()` 中补齐 `deprecated` / `forgotten` 的读取，或直接复用 `MemoryGraph.from_dict()`（[memory\_graph.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/graph/memory_graph.py#L213-L237) 已支持这些字段）。
- **验收标准**：含 `deprecated: true` 的 YAML 经「加载 → 保存 → 再加载」后，废弃标记保持不变。

### P1-2：PathTracker 双因子动态权重未接入运行链路

- **定位**：[mcp\_server.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/mcp_server.py#L804-L810) 创建 `PurposeDrivenRetriever` 时未传 `path_tracker`
- **现象**：
  - [path\_tracker.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/path_tracker.py) 的「终身累积增强 × 会话饱和抑制」机制已实现，但无任何调用点接入，属于死代码。
  - `PathTrackerConfig.session_boundary_hops` 字段声明了但从未被使用。
- **影响**：检索结果不包含路径级动态权重，与设计文档描述不符。
- **修复建议**：
  - 在 `main()` 中实例化 `PathTracker` 并传入 `PurposeDrivenRetriever`；确认 `start_session()` 的调用时机（如每次 MCP 会话 / 每轮对话）。
  - 或明确决策：若暂不启用，则删除 `path_tracker.py` 及 `retriever.py` 中的相关分支，避免误导。
- **验收标准**：同一路径跨会话被激活后，检索权重体现终身增强；同会话内重复激活体现饱和抑制。

### P1-3：调度器自动检查点未真正启动

- **定位**：[maintenance\_scheduler.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/maintenance_scheduler.py#L186-L204) `_start_auto_checkpoint()`
- **现象**：定义了 `_start_auto_checkpoint()` 与 `_auto_checkpoint()`，但 `start()` 中从未调用，`auto_checkpoint_interval`（默认 300s）形同虚设。
- **影响**：长时间运行时缺少定时持久化保护，异常退出会丢失内存中的图谱变更。
- **修复建议**：在 `start()` 中调用 `self._start_auto_checkpoint()`，并确保 `on_checkpoint` 回调已设置（MCP 中已设置，见 [mcp\_server.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/mcp_server.py#L785)）。
- **验收标准**：启动调度器后，每隔 `auto_checkpoint_interval` 触发一次 `on_checkpoint` 回调。

  【移除该功能】

### P2-4：未启用 embedding 时语义去重静默失效

- **定位**：[graph\_builder.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/graph_builder.py#L371-L384) `_find_duplicate()`
- **现象**：去重依赖 `vector_store.search()`；当未启用 embedding 时 `store` 为 `None`，`search()` 返回空列表，去重直接返回 `None`，且无任何告警。
- **影响**：存根/无 embedding 模式下，重复节点会静默入库。
- **修复建议**：
  - 无向量能力时回退到「精确字符串匹配」去重，并记录日志。
  - 或在 `VectorStore.search()` 返回空时明确区分「索引为空」与「无相似结果」。
- **验收标准**：无 embedding 模式下，重复内容的 `create` 能被拦截（精确匹配）。

### P2-5：FAISS L2 距离被近似当作余弦相似度

- **定位**：[graph\_builder.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/graph_builder.py#L376-L379)
- **现象**：`similarity = 1.0 / (1.0 + score)` 将 FAISS 返回的 L2 距离近似映射为相似度，非严格余弦。
- **影响**：`dedup_threshold=0.85` 的语义去重阈值在 L2 空间下语义不精确，可能误判或漏判。
- **修复建议**：
  - 统一使用余弦距离建索引（FAISS 的 `IndexFlatIP` + 归一化向量），或改用 `_content_vectors` 缓存直接计算余弦相似度。
  - 至少在 `search()` 返回值中明确标注距离类型。
- **验收标准**：去重判定与实际余弦相似度一致（单元测试覆盖）。

### P2-6：PeakFinder 默认参数与实际使用不一致

- **定位**：[peak\_find.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/peak_find.py#L23-L24) vs [retriever.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/retrieval/retriever.py#L45)
- **现象**：`PeakFinder.__init__` 默认 `patience=1, min_delta=0.03`，而 retriever 实际传 `patience=2, min_delta=0.015`，与类 docstring 示例一致但与默认值不一致。
- **影响**：默认值有误导性，且 `peak_tolerance` 在 `PeakFinder` 中定义但 `retriever` 未显式传（依赖默认 0.10）。
- **修复建议**：统一默认值，或在 `PurposeDrivenRetriever` 中显式传入全部关键参数。
- **验收标准**：默认构造与检索实际使用参数一致。

  【以当前使用参数为准】

### P3-7：可视化 API 使用单线程 HTTPServer

- **定位**：[api\_server.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/viz/api_server.py#L339) `HTTPServer`
- **现象**：标准库 `HTTPServer` 为单线程串行处理，CRUD 面板并发请求会阻塞。
- **影响**：多客户端同时操作 3D 面板时响应缓慢；且写回 YAML 无并发保护。
- **修复建议**：改用 `ThreadingHTTPServer`（`from http.server import ThreadingHTTPServer`），并对 `_save()` 加锁；或迁移到已有的 Starlette/uvicorn 栈。
- **验收标准**：并发 CRUD 请求下无阻塞，写回不产生脏数据。

【本地mcp服务，可视化面板只在单机上运行】

### P3-8：缺少测试、评测脚本与实验数据

- **定位**：[tests/](file:///c:/Users/makot/Desktop/Ariadne/tests) 为空（仅 `.gitkeep`）
- **现象**：
  - 无单元测试。
  - README 声称「216 节点 P R\@all=0.538 vs A=0.159」，但仓库无对应数据文件与评测脚本。
- **影响**：重构无保障，核心指标无法复现。
- **修复建议**：
  - 优先为核心纯逻辑补测试：`jump_axis`、`peak_find`、`path_tracker`、`graph_builder` 的 CRUD/去重/事务。
  - 补充评测脚本（`R@all` 计算）与样例数据集，或至少说明实验数据来源。
- **验收标准**：核心模块测试可运行通过；评测指标可复现。

  【前置修复完成后再选择性实施】

***

## 三、增量实施顺序

1. **第一步（P0）**：修复 YAML 加载丢字段（问题 1）——改动小、影响面大，优先保障数据一致性。
2. **第二步（P1）**：接入 PathTracker（问题 2）与启动自动检查点（问题 3）——让已设计的能力真正生效。
3. **第三步（P2）**：修复去重静默失效（问题 4）、统一距离度量（问题 5）、统一 PeakFinder 参数（问题 6）。
4. **第四步（P3）**：可视化 API 并发化（问题 7）、补测试与评测（问题 8）。

> 建议每完成一步就补充对应测试，避免回归。

