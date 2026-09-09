# TEMPORAL 时序功能 · 实现总结报告

> 日期：2026-08-27 · 状态：**已落地，测试全绿** · 关联：`0827——TEMPORAL时序关系专项实验与链路补全.md`（完整过程/实验/决策）

## 0. 一句话结论

时序功能以「**主检索器保留纯基线 + 独立单跳 `temporal_lookup` 工具 + LLM 路由分流**」的方式落地：真正打通了系统此前缺失的「时间→事件」方向（昨天我干了什么），且**主检索零污染、无回归**；功能 / 路由 / 稳定性三层测试全绿。

---

## 1. 背景与目标

- 系统图谱中 `TEMPORAL` 边（事件→时间，**单向**，跳转轴反向权重恒为 0）被系统性忽视：数据集稀疏、抽取层无时间节点引导、检索层反向无法遍历。
- 目标：填补「从时间锚点联想事实」这一真实缺口，同时**避免全局化引入的回归**（此前全局提权已致 R@all −4pp、抽取精度 87%→61%）。

## 2. 诊断出的三个代表性问题（详见主文档 §8）

| 编号 | 问题 | 证据 |
|---|---|---|
| **RC1** | **抽取层是硬瓶颈** | 基线抽取 TEMPORAL 边 = 0 → 全图 `anchorable=0`，时间线无从谈起；temporal 引导到 61.5% 但过度拆分（边精确率下滑） |
| **RC2** | **检索层跨期真弱** | gold 图（数据完美）下 TD R@all 仅 0.17 → boost 0.67 → dba_base 0.83，证明"多期跨越"真实难 |
| **RC3** | **"时间→事件"是词面假象** | gold 图默认跳转下 TB R@all 竟=1.0，但实际靠事件文本里**字面含时间词**被向量命中；纯图结构的时间→事件反向**从未实现**，被词面掩盖 |

## 3. 方向收敛（关键决策链）

1. 全局提权（jump_axis 全表抬 TEMPORAL）→ **回归**（R@all −4pp + 抽取过度拆分）→ 回退。
2. 手术式补全（检索侧，`_is_temporal_query`/真锚点豁免/定向权重）→ 增量复测证实**空操作**（ON/OFF 路径一致、structure 无差）→ 废弃。
3. **收敛为「独立单跳时序工具」**：时序 = 一次符号查找，联想/推理交给上层 LLM。呼应"本系统是模拟联想、agent 无离线经历——不给它装时间记忆体，只给查询原语"。

**分工路由**（§9.6）：

| 查询类型 | 走哪个 |
|---|---|
| 时间→事件（T,B：昨天我做了什么） | `temporal_lookup`（单跳） |
| 事件→时间（TA：什么时候开始学吉他） | 主检索器（正向 TEMPORAL 联想） |
| 跨期/历程（TD：从高中到工作） | 主检索器（多跳联想） |
| 因果/事实/现状 | 主检索器（照旧） |

## 4. 最终架构

- **主检索器 `retrieve()`**：回到**纯基线**（已删除全部旧手术式时序代码），零污染。
- **`temporal_lookup`**（独立工具，不进主检索流程）：
  - 仅做「时间→事件」反向、**单跳**；
  - **多锚点返回**（`matches`，默认 3，按相关性降序），把「三年→大专/工作」这类歧义交给 LLM 挑，不做硬性消歧；
  - **真锚点校验** `_has_reverse_temporal`（只认带反向 TEMPORAL 邻居的候选）→ 挡掉 11 个「含时间词事件节点」误判；
  - 判定 `_tool_is_time_node` + 正则 `TOOL_TIME_ANCHOR_RE`。
- **MCP 工具 `dba_temporal_lookup`**（`DBAServer.temporal_lookup` 包装 + handler + 描述），并由 LLM 在 `dba_query_memory` 与 `dba_temporal_lookup` 间**分流**。

## 5. 测试与结果（全绿）

| 层 | 脚本 | 覆盖 | 结果 |
|---|---|---|---|
| 功能 | `eval/test_temporal_lookup.py` | gold 图 TB/时代锚点/多锚点≤3/TA 拒答 | ✅ |
| 端到端+完整 | `eval/test_temporal_full.py` | 322 图 th0/th1/th2 正确、误判过滤、DBAServer 走通 | ✅ |
| 路由 | `eval/test_tool_dispatch.py` | LLM 在两者间路由 | ✅ 7/7 |
| 稳定性 | `eval/test_temporal_stability.py` | 确定性/防误判/上限/非时序拒答/主检索无回归 | ✅ 18 项 PASS |
| 抽取侧规则 | `tests/test_dba_temporal.py` | 守卫 `has_temporal_signal` 触发/语义边界 + TEMPORAL 规则串完整性 | ✅ 11 项 PASS |
| 闭环+增量 | `eval/test_temporal_closedloop.py` | 路由→工具→取回→生成；工具(时间切片) vs 主检索器(跨期叙事) | ✅ |
| RC3 去词面化确证 | `eval/verify_temporal_delex.py` | 事件去时间词后，反向遍历仍靠 TEMPORAL 结构取到昨日事件 | ✅ |
| 动态更新 | `eval/test_temporal_dynamic.py` | 增量新增时间锚点/事件/边后工具即时反映；删边即时消失 | ✅（修复"这周"漏匹配）|
| 抽取③零回归 | `eval/run_dba_extraction_eval.py` | 当前守卫下 0822 子集：node_prec 91.1%、0 TEMPORAL、未过度拆分 | ✅ |
| 抽取②锚点产出 | `eval/verify_temporal_extract.py` | 时序叙事触发守卫→产出 TEMPORAL 边+时间锚点；非时序 0 边 | ✅ |
| MCP 链路 | `eval/test_mcp_link.py` | 工具注册完整；时序工具经 MCP handler 全链路可用；max_anchors 已透传 | ✅（透传 max_anchors 修复）|

关键样例：
- `temporal_lookup("昨天用户都做了些什么")` → 锚点 `T10(昨天)` → `{e9,e10,e12}`；另一候选 `T11(今天)`。
- `temporal_lookup("用户当前工作时期怎么样")` → 锚点 `th2` → `{w0,w13}`。
- `temporal_lookup("用户是什么时候开始学吉他")` → 空（应走主检索器）。

## 6. 现状

- 主检索器纯基线、零污染；`temporal_lookup` 工具 + 分流已接入；三层测试全绿。
- **关键认识**：`retrieve()` 是**会话态**——`PathTracker` 的 `lifetime_counts` 跨调用累积（终身增强，刻意设计），故同实例多跑不 bit 确定，但关键节点稳定、无回归。

## 7. 遗留 / 待决（承接 §9.4）

| 项 | 说明 |
|---|---|
| **TIME 节点类型**（另议） | 语义最顺（根除内容正则误判），但改动面大（6→7 类/prompt/权重/校验），需干净重测 |
| **细粒度锚点来源** | 322 图仅阶段级 th0/th1/th2，「昨天/上周」无真实节点，需抽取侧引导细粒度时间锚点 |
| **现实/虚拟 present** | 时间基准指针：虚拟时钟由用户显式推动（"三年后"→present 前移），相对时间词统一按 present 解析 |
| **锚点语义定位质量** | "三年"→大专/工作歧义已用多锚点缓解，但非根治；可加查询实体/词面加权 |
| **RC3 确证（✅ 已完）** | 已用去词面化数据集确证：`temporal_lookup` 反向遍历**靠 TEMPORAL 结构**，非词面（见 §5 测试表）；主检索器确含词面/关联成分 |

## 8. 相关文件 / 数据

**测试脚本**：
- `eval/`：`test_temporal_lookup.py`、`test_temporal_full.py`、`test_tool_dispatch.py`、`test_temporal_stability.py`
- `tests/`：`test_dba_temporal.py`（抽取侧守卫规则）

**关键数据集**（`data/dba_eval/`）：
- `lifemem_temporal.yaml` / `_dialogs.yaml` / `_queries.yaml`（TEMPORAL 专项）
- `lifemem_crossperiod.yaml` / `_queries.yaml` / `_dialogs.yaml`（跨期+细时间 gold，诊断 RC1/RC2/RC3）
- `data/LiFEMem.yaml`（322 全量，阶段锚点 th0/th1/th2）

**输出**（`eval/outputs/`）：
- `temporal_rooted.json`（RC1/RC2/RC3 证据）
- `temporal_experiment.json`（17 节点端到端）
- `lifemem_temporal_cmp.json`、`story_alone_inj_temporal.json`（时序故事结构对比）

**实现**：`dba_pipeline/retrieval/retriever.py`（`temporal_lookup`）、`dba_pipeline/mcp_server.py`（`dba_temporal_lookup` 工具）、`dba_pipeline/extraction/dba.py`（时序叙事守卫，抽取侧）。
