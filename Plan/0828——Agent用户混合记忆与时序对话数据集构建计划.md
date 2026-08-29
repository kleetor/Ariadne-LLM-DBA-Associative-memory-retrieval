# 0828——Agent·用户混合记忆与时序对话数据集构建计划

> 背景：系统已先后修复**时序（TEMPORAL）**与**自身记忆（AI 自身经历/行为入图）**两处缺陷。
> 但现有数据集（LiFEMem / lifemem_temporal / lifemem_crossperiod / ai_self_dialogs）有一个共性缺口：
> **都是「用户单主体」视角**——对话由用户讲述自己的事、助手仅倾听回应，无法同时验证「混合记忆（用户+助手）」
> 与「时序信息」在同一条对话里的协同抽取。本计划构建一份新的 Agent-用户对话数据集，作为这两个能力联合验证的基座。

***

## 一、目标

构建一份**约 100 节点**的 Agent-用户对话数据集（gold 图谱 + 口语化对话语料），要求：

1. **混合记忆**：单条对话同时蕴含「用户事实」与「助手自身经历/行为」两类节点；
2. **时序信息**：对话内含多个**事件级时间锚点**（上周一晚上 / 这周 / 周三凌晨等），触发 `has_temporal_signal`，
   驱动 TEMPORAL 链，期望抽出「时间锚点 THING 节点 + TEMPORAL 边（事件→时间）+ SEQUENCE 边（事件先后）」；
3. **内容不限**：覆盖开发工作 + 日常寒暄聊天，节点总量 ~100（用户 + 助手 + 时间锚点）。
4. **可评测**：配套 gold 图谱，能跑现有 `run_dba_extraction_eval.py`（抽取质量）+ 新增专项断言（混合记忆 & 时序）。

***

## 二、现状盘点（差距确认）

| 现有产物 | 主体视角 | 时序 | 能否验证「混合记忆」 |
|---------|:---:|:---:|------|
| LiFEMem（322 节点） | 仅用户 | 仅 8 条年代级 temporal | 否（无助手节点概念） |
| lifemem_temporal / crossperiod | 仅用户 | 事件级 TEMPORAL | 否 |
| **ai_self_dialogs**（本修复新增） | 用户 + 助手 | 无 | 是（但无时序，仅 3 批小手例） |

**两者的工具根因**：
- `eval/gen_full_dialogs.py` 的 `GEN_PROMPT` 固定「用户以第一人称自述、assistant 是倾听回应角色」→ 只产出用户信息点，**无法产出助手行为**；
- 时序专项数据集的生成器没有「主体」维度，且时间锚与事件混在同一批。

**结论**：需要一份**双向信息**（用户+助手都承载信息点）且**带事件级时间锚点**的新数据集，并改造生成器。

***

## 三、数据集设计

### 3.1 主体与结构

节点沿用现有 `NodeType`（status/reason/action/thing/person/emotion），**不新增类型**；主体区分沿用已定的「content 前缀」方案
（用户→「用户…」，助手→「助手…」），时间锚点 → `THING` 且 content 即时间词（如「上周一晚上」）。gold 中不额外加字段，保持与现有评测链兼容。

### 3.2 ID 前缀（兼顾场景分组 + 主体 + 复用生成器）

| 主体 | 工作开发 | 日常起居/健康 | 社交/约会 | 学习/探索 | 混合闲聊 |
|------|:---:|:---:|:---:|:---:|:---:|
| 用户 | `uw` | `ud` | `us` | `ul` | `um` |
| 助手 | `aw` | `ad` | `as` | `al` | `am` |
| 时间锚点 | `tw` | `td` | `ts` | `tl` | — |

（`gen_full_dialogs` 的 `_scene_of` 依赖前缀最长匹配，此表可直接替换为新 `SCENE_ORDER` 分组。）

### 3.3 场景与节点预算（目标 ~100）

| 场景 | 用户 | 助手 | 时间锚点 | 小计 |
|------|:---:|:---:|:---:|:---:|
| 工作开发（加班/排期/上线/复盘） | 12 | 10 | 8 | 30 |
| 日常起居/健康（晨跑/晚睡/体检） | 10 | 6 | 6 | 22 |
| 社交/约会（周五聚会/约饭） | 8 | 4 | 5 | 17 |
| 学习/探索（学数据分析/写脚本） | 8 | 8 | 5 | 21 |
| 混合闲聊（跨场景 + 少量寒暄） | 5 | 5 | 0 | 10 |
| **合计** | **43** | **33** | **24** | **100** |

> 时间锚点 24 个：每批对话埋 ≥2 个**不同**日期/时期词，保证 `has_temporal_signal` 命中 → 走 TEMPORAL 增强链。

### 3.4 边类型

与现有体系一致：`temporal`（事件→时间，单向）、`sequence`（事件先后，单向）、`causal`（单向）、
`scenario`（同场景，双向）、`attribute`（双向）。时间锚点集合成 gold 一部分，用于对照 TEMPORAL 边召回。

***

## 四、构建流程

1. **定义 gold 图谱**（`data/dba_eval/agent_user_gold.yaml`）：按 §3.2-3.3 手工设计 100 节点 + 若干边，标注主体与时间锚点；
2. **改造对话生成器**（`eval/gen_agent_user_dialogs.py`，基于 `gen_full_dialogs.py`）：
   - `GEN_PROMPT` 加入**主体字段**：信息点带 `subject: user|assistant`；
   - 用户信息点由 user 引出；**助手信息点由 assistant 说出/做**（「我帮你整理了…」「我给你写了个脚本…」），assistant 不再是纯倾听；
   - 每批强制含 ≥2 个不同时间词（时间锚点节点要自然带出）；
   - `--dry-run` 先打印批次计划与主体分布，再生成。
3. **生成对话语料**：逐批 LLM 还原 → `data/dba_eval/agent_user_dialogs.yaml`；校验 **covers 覆盖 100/100**、主体分布（用户节点 / 助手节点 / 时间锚点）符合预算。
4. **空图 DBA 建图验证**：脚本跑 `MemoryDBA.maintain()`，验证两个能力同时生效（见 §五）。

***

## 五、验证方案

新建综合验证脚本 `eval/run_dba_agent_user_eval.py`，复用 `build_llm/build_embeddings/_load_dotenv`：

| 断言 | 判定 | 说明 |
|------|------|------|
| **混合记忆** | 抽出 content 以「助手」开头的节点数 > 0，且 gold 助手节点召回达标 | 自身记忆入图 |
| **时序** | TEMPORAL 边数 > 0，且时间锚点 THING 节点数 ≥ 阈值 | TEMPORAL 链生效 |
| **规模** | DBA 活跃节点数落在 ~100（±20%） | 规模水位 |

另可复用 `run_dba_extraction_eval.py`（传 `--yaml data/dba_eval/agent_user_gold.yaml --dialogs ...`）做**整体抽取质量**（节点/边召回·精确·类型一致），与 gold 对照。

**产出报告**：`eval/outputs/dba_agent_user_extraction.json` + 简要结论（含各主体召回、时间锚点/ TEMPORAL 边抽取情况）。

***

## 六、产物清单

| 文件 | 内容 |
|------|------|
| data/dba_eval/agent_user_gold.yaml | gold 图谱（~100 节点 + 边，含主体/时间锚点） |
| data/dba_eval/agent_user_dialogs.yaml | 口语化对话语料（覆盖 100 节点） |
| eval/gen_agent_user_dialogs.py | 双向信息对话生成器（dry-run / generate） |
| eval/run_dba_agent_user_eval.py | 混合记忆+时序综合验证脚本 |
| eval/outputs/dba_agent_user_extraction.json | 验证结果 |

***

## 七、风险与权衡

| 风险 | 缓解 |
|------|------|
| 生成器「用户自述」惯性导致助手节点覆盖不足 | GEN_PROMPT 显式强调助理也承载 subject=assistant 的信息点，dry-run 先看主体分布 |
| LLM 抽取随机性 → 单次结论不稳 | 关键断言用多个场景批次聚合判断；必要时多次独立抽取取均值（沿用 0822 做法） |
| 时间词集中在少数批 → 时序验证不显著 | 每批强制 ≥2 个不同时间词，时间锚点按场景铺开 |
| gold 主体「助手」与 DBA content「助手」表述偏差 | 验证以「content 前缀+语义匹配」双口径判定，避免纯字符串漏判 |

***

## 八、验收标准

- [ ] gold 图谱节点数达 100±10，主体覆盖用户 + 助手 + 时间锚点三类；
- [ ] 对话语料 covers 覆盖 100/100 节点，且含 ≥2 个不同时间词的批次占比高；
- [ ] DBA 建图后：抽出「助手」节点、TEMPORAL 边、时间锚点 THING 节点均 > 0（同一条数据流内同时成立）；
- [ ] 复用 `run_dba_extraction_eval.py` 产出整体抽取质量指标，与现有水平对齐。
