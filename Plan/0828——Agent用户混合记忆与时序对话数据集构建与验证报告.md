# 0828——Agent·用户混合记忆与时序对话数据集构建与验证报告

> 背景：系统先后修复了**时序（TEMPORAL）**与**自身记忆（AI 自身经历/行为入图）**两处缺陷。
> 但现有数据集（LiFEMem / lifemem_temporal / lifemem_crossperiod / ai_self_dialogs）均为「用户单主体」视角，
> 无法同时验证**混合记忆（用户+助手）**与**时序信息**在一条数据流里的协同抽取与故事化。
> 本报告记录：新构建的 Agent-用户混合记忆 + 时序数据集，及其抽取 / 检索 / StoryRank / 稳定性全套验证。

***

## 一、数据集构建（96 节点）

| 文件 | 内容 |
|------|------|
| data/dba_eval/agent_user_gold.yaml | gold 图谱：**96 节点 + 104 内部边**，主体分布 **用户 42 / 助手 32 / 时间锚点 22** |
| data/dba_eval/agent_user_dialogs.yaml | 15 批口语对话，`covers` 覆盖 **96/96**，**14/15 批触发时序** |
| data/dba_eval/agent_user_queries.yaml | 6 条 StoryRank 查询（时序还原 / 助手做过的事 / 混合记忆） |

**场景与节点预算**：工作开发（27）/ 日常健康（20）/ 社交（17）/ 学习（21）/ 混合闲聊（11）。
**主体区分**：沿用 content 前缀（「用户…」/「助手…」），时间锚点为 `THING` 且 content 即时间词。
**时序设计**：每批对话 ≥2 个不同日期/时期词，驱动 `has_temporal_signal` → TEMPORAL 链。

**新增脚本**：
- `eval/build_agent_user_gold.py`：gold 构建 + 程序化校验（边引用/类型/主体前缀）。
- `eval/gen_agent_user_dialogs.py`：**双主体生成器**——信息点带 `subject`，assistant 也承载行为信息点（区别于旧的「仅用户自述、助手纯倾听」生成器 `gen_full_dialogs.py`）。

***

## 二、综合验证（三项同时成立）

`eval/run_dba_agent_user_eval.py`（空图逐批 maintain，15 批）：

| 断言 | 结果 | 判定 |
|------|------|:---:|
| 混合记忆 | 助手节点 37，gold 32，召回 0.84 | ✅ |
| 时序 | TEMPORAL 边 28，时间锚 23，召回 0.68 | ✅ |
| 规模 | 活跃 136 / gold 96（1.42×，区间内） | ✅ |

主体召回：**用户 0.93 / 助手 0.84 / 时间锚点 0.68**。

***

## 三、整体抽取质量（batch 模式）

`eval/run_dba_extraction_eval.py`（--batch-size 6 → 3 批，对照 gold）：

| 指标 | 结果 |
|------|:---:|
| 节点召回 | **88.5%** |
| 节点精确 | 64.4% |
| 类型正确 | **84.7%** |
| 边召回 | 45.2% |
| 边精确 | 16.4%（激进连边特性，历史结论：边可多连，检索/故事不受伤） |
| 边类型一致 | 69.2% |

***

## 四、StoryRank 故事化

`eval/run_dba_agent_user_story.py`：DBA 图 **132 节点 / 349 边 / TEMPORAL 24 条** → 6 条查询全部产出故事。
故事文本均包含「用户经历 + 助手做过的事 + 时间词」自然融合。如 q2 清楚串出「助手帮用户做过的具体事」
（排期清单 / 拆每日计划 / 推荐基础课 / 整理 3 项目 / 文档整理），q5 含时序（上周/周五/周六/下周日），
q6 混合记忆 + 时序同入话。

***

## 五、抽取粒度实验（分批 vs 整段一次）

`eval/run_dba_agent_user_fulltext.py`（同一段完整对话 4670 字，一次 maintain）：

| 指标 | 分批（逐批15次） | 整段（一次） |
|------|:---:|:---:|
| 活跃节点 | **136** | 92 |
| 边 | **366** | 136 |
| TEMPORAL 边 | **28** | 13 |
| 用户召回 | **0.93** | 0.88 |
| 助手召回 | **0.84** | 0.72 |
| 时间锚召回 | **0.68** | 0.64 |

**结论**：整段一次抽取压缩更深（边 366→136、TEMPORAL 28→13），主体召回集体略降；StoryRank 产物
`story_nodes` 重叠极低（Jaccard 0.0~0.23），典型 q5「周末朋友安排」整段版跑偏到工作/健康——**整段图边稀疏、
联想路径断裂**。印证 0822「批量文本让提纯更充分 → 压缩更深」。**逐批维持（真实 MCP：每 `add_conversation`
触发一次维护）更优。**

***

## 六、稳定性验证（3 次独立分抽取）

`eval/run_dba_agent_user_stability.py`（复用 run1 + 补跑 run2/run3）：

| 指标 | run1 | run2 | run3 | 均值 | 极差 |
|------|:---:|:---:|:---:|:---:|:---:|
| 活跃节点 | 136 | 123 | 121 | 126.7 | **15** |
| 助手节点 | 37 | 34 | 35 | 35.3 | **3** |
| 时间锚节点 | 23 | 20 | 21 | 21.3 | **3** |
| TEMPORAL 边 | 28 | 25 | 27 | 26.7 | **3** |
| 用户召回 | 0.93 | 0.79 | 0.71 | 0.81 | 0.214 |
| 助手召回 | 0.84 | 0.66 | 0.72 | 0.74 | 0.188 |
| 时间锚召回 | 0.68 | 0.59 | 0.68 | 0.65 | 0.091 |

**结论**：
- **结构类极稳**（活跃节点极差 15、助节点/时间锚/TEMPORAL 边极差≈3）——主体与时序骨架在多次抽取中高度稳定；
- **主体召回是性能敏感项**（单次会飘 0.2 左右），严谨评测需取均值（用户 0.81 / 助手 0.74 / 时间锚 0.65）；
- 边数波动大（极差 76）但非敏感——与 0822「图结构一致性要求低、边类型非性能敏感项」定论一致。

***

## 七、产物清单

| 文件 | 内容 |
|------|------|
| data/dba_eval/agent_user_gold.yaml | gold 图谱（96 节点 + 104 边） |
| data/dba_eval/agent_user_dialogs.yaml | 对话语料（15 批，覆盖 96/96） |
| data/dba_eval/agent_user_queries.yaml | 6 条 StoryRank 查询 |
| eval/build_agent_user_gold.py | gold 构建器 + 校验 |
| eval/gen_agent_user_dialogs.py | 双主体对话生成器 |
| eval/run_dba_agent_user_eval.py | 综合验证（混合记忆+时序+规模） |
| eval/run_dba_agent_user_story.py | StoryRank 故事化 |
| eval/run_dba_agent_user_fulltext.py | 抽取粒度实验 + StoryRank diff |
| eval/run_dba_agent_user_stability.py | 多次抽取稳定性 |
| eval/outputs/dba_graph_agent_user.yaml | 分批抽取 DBA 图（132 节点/349 边） |
| eval/outputs/dba_graph_agent_user_fulltext.yaml | 整段抽取 DBA 图 |
| eval/outputs/dba_agent_user_extraction.json | 综合验证结果 |
| eval/outputs/dba_extraction_agent_user.json | 整体抽取质量 |
| eval/outputs/dba_story_agent_user.json | 分批图 StoryRank 故事 |
| eval/outputs/dba_story_diff_agent_user.json | 分批 vs 整段 StoryRank diff |
| eval/outputs/dba_agent_user_stability.json | 稳定性（3 次） |

***

## 八、总结论

1. **新数据集覆盖双能力**：单条数据流同时蕴含用户事实 + 助手自身行为 + 事件级时间锚点，为「自身记忆」与「时序」修复提供联合验证基座；
2. **两项修复在完整数据集上同时生效**：综合验证三项全过，StoryRank 故事自然融合用户/助手/时间；
3. **逐批维护优于整段一次**：压缩深会伤联想路径（整段版 q5 跑偏、story_nodes 重叠 <0.23），真实 MCP 逐批维护是正确建图方式；
4. **抽取稳定性良好**：主体与时序骨架极稳（极差≈3），主体召回为性能敏感项（均值 用户 0.81 / 助手 0.74 / 时间锚 0.65），评测应多次取均值。
