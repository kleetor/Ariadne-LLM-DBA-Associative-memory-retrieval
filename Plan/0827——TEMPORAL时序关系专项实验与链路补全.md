# TEMPORAL 时序关系专项实验与全链路补全

> **📌 总结报告（精简版）**：见 [TEMPORAL时序功能实现总结报告.md](TEMPORAL时序功能实现总结报告.md)。本文档是完整过程/实验/决策的原始记录。

> 来源：2026-08-27 会话。此前评测链（抽取 → 边连接 → 检索 → 评价）在 LiFEMem 子集与全量上反复收敛，但 **TEMPORAL（事件→时间）这一关系被系统性忽视**：数据集层只有年代级粗锚点、抽取层从不产出 TEMPORAL 边、检索层跳转权重几乎为零。本次专项：先以自建"事件级时间线"小数据集做端到端闭环，再在 LiFEMem 全量(322) 上补充 TEMPORAL 验证时序，最后定位并**补全全系统链路对 TEMPORAL 的忽略**。

***

## 一、问题定位：TEMPORAL 为何被避开

理论定义（[理论部分 L184](file:///c:/Users/makot/Desktop/Ariadne/Ariadne——LLM DBA管理与目的驱动的联想记忆检索系统 理论部分.md#L184)）：**TEMPORAL = 单向（事件→时间）**，理想粒度是「面试→下周五」。实际现状：

| 层 | 现状 | 证据 |
|----|------|------|
| 数据集 | 全量 LiFEMem 392 边仅 **8 条 temporal**（~2%），且全为年代级粗锚点 `h0/h20/h23→th0`、`c0/c11/c14→th1`、`w0/w13→th2`，锚点 `th0/th1/th2` 标为 `thing`；子集(100 节点) **不含 th 时间节点 → temporal=0** | [LiFEMem.yaml L2713-2734](file:///c:/Users/makot/Desktop/Ariadne/data/LiFEMem.yaml#L2713-L2734) |
| 抽取 | 两步拆分后，边连接 few-shot 示例**无任何 TEMPORAL 示例**；节点抽取 prompt **无"时间节点"引导**；节点类型无 TIME | [dba.py L81-L132](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py#L81-L132)、[dba.py L59-L79](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py#L59-L79) |
| 检索 | TEMPORAL 前向权重 0.3-0.5、**反向恒为 0**（时间锚点 thing 仅 0.3），P 检索沿 temporal 边几乎不扩展 | [jump_axis.py L41-L102](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/jump_axis.py#L41-L102) |
| 结构 | 无显式"事件"抽象；SEQUENCE(相对先后) 与 TEMPORAL(绝对时间锚点) 被混为"时间维度"，角色不同 | 理论部分 |

**结构性根因**：TEMPORAL 需要"时间节点"作为对象，但建图时没有引导创建、也没有示例可模仿 → 抽取层从不产出；即便产出，检索层也因极低权重而忽略。三环相扣。

***

## 二、实验一：事件级时间线小数据集 · 端到端闭环（17 节点）

### 2.1 数据与脚本
- **新数据集**：`data/dba_eval/lifemem_temporal.yaml`（10 事件/状态 + 7 时间锚点，21 边含 **10 temporal**、7 sequence、4 causal）；`lifemem_temporal_dialogs.yaml`（5 批对话，显式时间词）；`lifemem_temporal_queries.yaml`（5 条事件检测查询 + 手写目的 + gold 时序）。
- **脚本**：`eval/run_temporal_experiment.py`。抽取 base/temporal 两版 prompt；时序重建按 TEMPORAL 定纪元秩排序，指标 = R@all / 可锚定率 / 时间线 pairwise。
- **隔离**：triage 在 temperature=0 下仍非确定（短对话 NEEDED/SKIP 反复横跳），实验旁路 triage（`_run_step`）并以 `triage_need_rate` 如实记录。

### 2.2 抽取保真度（同一语料，base vs 加"时间节点引导+TEMPORAL few-shot"）
| 配置 | 节点 | 边 | 节点召回 | 边召回 | **temporal 边** | sequence | causal |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| gold | 17 | 21 | — | — | 10 | 7 | 4 |
| base | 16 | 39 | 52.9% | 28.6% | **0/10（0%）** | 3/7 | 3/4 |
| **temporal** | 18 | 27 | **76.5%** | 42.9% | **4/10（40%）** | 3/7 | 2/4 |

- **base 零 temporal 边**：LLM 把时间"焊进"事件内容（如 `用户项目于周三凌晨上线`），无独立时间节点，无法连 TEMPORAL。
- **temporal 版** 产生 5 个独立 THING 时间节点 + 5 条 TEMPORAL 边（`temporal_graph_temporal.yaml`），节点召回 53%→77%、temporal 召回 0→40%。**缺口可修复，只需引导 + 示例。**

### 2.3 事件检测效用（5 条查询，P 检索注入目的）
| 图 × 跳转 | R@all | 可锚定率 | 时间线一致性 |
|------|:--:|:--:|:--:|
| gold | 100% | 100% | 1.0 |
| dba_base | **84.2%** | **0%** | **N/A（无法建时序）** |
| dba_temporal（默认跳转） | 45.0% | 43.3% | **1.0** |
| dba_temporal（temporal_boost） | 51.7% | 53.3% | **1.0** |

### 2.4 实验一定论
1. **TEMPORAL 边是"事件时间线可重建"的充要条件**：dba_base 虽召回 84% 事件（高于 temporal），但 **可锚定率 0%**、时间线重建 N/A——没有 TEMPORAL 锚点，召回再多也是一堆无序事实；有 TEMPORAL 边（temporal 版）则**可锚定 40-53%、时间线一致性全 1.0**。
2. **代价**：加 TEMPORAL 后图更"干净"（27 边 vs base 39）+ 时间节点成为检索目的地，事件 R@all 下降（84%→45%）。**当前 P 检索未为"事件+锚点双 surfacing"设计** → 提高 TEMPORAL 跳转权重（temporal_boost）把 R@all 45→52%、可锚定率 43→53%，**需检索侧配合**。
3. **附带发现**：triage 非确定（temperature=0 下 NEEDED/SKIP 反复横跳），真实系统有健壮性隐患。

***

## 三、实验二：LiFEMem 全量(322) · 补充 TEMPORAL + 时序重建

### 3.1 背景
LiFEMem.yaml 已含 h(高中)/c(大专)/f(家庭)/th(时间锚点) 叙事层，但 **TEMPORAL 边仅 8 条**，多数叙事事件未锚定。216 基站（memory_graph.yaml）则完全无时间锚点、多为"当前生活"快照、时序扁平。

### 3.2 数据与脚本
- **补充**：按人生阶段前缀补 TEMPORAL 边（`h*→th0 高中`、`c*→th1 大专`、`w*→th2 工作`），新增 91 条，锚定事件 8→99（导出 `eval/outputs/lifemem_full_temporal_augmented.yaml`）。
- **脚本**：`eval/run_temporal_lifemem.py`。时序重建 = 主序 TEMPORAL 纪元秩(th0<th1<th2) + 次序推进秩（SEQUENCE-only 与 SEQUENCE+CAUSAL 两版对比）；指标 R@all / 可锚定率 / 跨纪元 pairwise / 期内 pairwise。
- 5 条叙事时序查询（恋爱线、高考→大专、转行线、高中经历、人生转折线）。

### 3.3 结果（5 条查询，使用补全后新跳转权重）
| 图 | R@all | 可锚定率 | **跨纪元 pairwise** | 期内(纯 SEQUENCE) | 期内(SEQUENCE+CAUSAL) |
|------|:--:|:--:|:--:|:--:|:--:|
| 原始(仅 8 条 TEMPORAL) | 77.0% | **56.9%** | 1.0 | 0.50 | 0.50 |
| **补充版(91 条)** | **96.0%** | **100%** | **1.0** | 0.66 | **0.88** |

> 口径说明：该表用补全后的跳转权重重跑。与补全前相比 R@all 大幅提升（补充版 57.8%→96%），因 TEMPORAL 权重上调后事件+锚点能被检索同时涌现；可锚定率仍由"图中有无 TEMPORAL 边"决定（32%→100%），跨纪元顺序恒 1.0。

### 3.4 实验二定论（三层）
1. **TEMPORAL = "可锚定 + 纪元骨架"使能层**：补边后锚定率 32%→100%，**跨纪元顺序完全正确（pairwise=1.0）**——TEMPORAL 把 高中<大专<工作 的纪元骨架建对。没有它，原始图 68% 事件放不进时间线（qA/qD 直接 None）。
2. **期内精细时序靠"推进边"，纯 SEQUENCE 不够**：期内纯 SEQUENCE 0.66——LiFEMem 叙事先后大量由 **CAUSAL** 承载（如"递情书→答应"是 causal）。用 SEQUENCE+CAUSAL 共同定序，期内升至 **0.88**。
3. **残余误差在跨阶段因果边**：qE(贯穿高中→大专→工作) 加 CAUSAL 后反回落(0.72→0.5)，因 `c8→ln9`、`c11→f12` 等跨阶段因果边搅乱推进秩。**真正时间线重建需区分"时序推进边"与"纯因果/静态边"，或补更细时间锚点。**

***

## 四、全系统链路 TEMPORAL 忽略点清单（补全目标）

| 环节 | 文件 | 忽略点 | 补全动作 |
|------|------|--------|---------|
| 抽取·节点 | [dba.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py) | NODE_EXTRACTION_PROMPT 无"时间节点单独抽取"引导 → 时间被焊进事件 content | 尝试加规则 7 → 常规语料过度拆分，**已回退** |
| 抽取·连边 | [dba.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py) | EDGE_LINKING_PROMPT 少 TEMPORAL 强调；FEWSHOT 无 TEMPORAL 示例 | 尝试加规则 7 + 示例 3 → 抽取精度/边类型一致率下降，**已回退** |
| 检索·权重 | [jump_axis.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/jump_axis.py) | TEMPORAL 前向 0.3-0.5、反向恒 0 → P 检索忽略 | 尝试上调至 (0.7-0.8, 0.3-0.4) → P 基线 R@all 掉 10pp，**已回退** |
| 检索·双 surfacing | [jump_axis.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/jump_axis.py) | 事件+锚点无法同时涌现 | 抬反向权重在小图上有效，但通用查询回归 → **需手术式定向处理** |
| 数据集 | LiFEMem.yaml（评估中） | 全量 temporal 稀疏(8)，多事件未锚定 | 已用补充版验证；是否回写 gold 待定 |

***

## 五、补全落地与回归复核（关键结论：全局化改动引起回归，已全部回退）

### 5.1 尝试的三处全局化改动
| 环节 | 文件 | 改动 |
|------|------|------|
| 抽取·节点 | [dba.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py) | NODE_EXTRACTION_PROMPT 增规则 7：时间/时段抽成独立 THING 节点 |
| 抽取·连边 | [dba.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py) | EDGE_LINKING_PROMPT 规则 4 补 TEMPORAL、增规则 7；FEWSHOT 增示例 3 |
| 检索·权重 | [jump_axis.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/core/jump_axis.py) | TEMPORAL 上调至 `(0.7-0.8, 0.3-0.4)`、时间锚点反向 `0.8` |

### 5.2 受控实验（17 节点语料）——时序能力确实被"点亮"了
这些改动在**含显式时间词的事件级语料**上高度有效：默认抽取即产出 TEMPORAL 边（召回 0→50%，加引导 70%）；时间线从"不可重建"(0% 可锚定、N/A)→"可锚定 63-65%、一致率全 1.0"；R@all 45%→93%（`temporal_experiment_postfix.json`）。

### 5.3 回归复核——在"常规/无时间词"语料上全局化却致回归，已回退

**① 检索层（35 查询，LiFEMem，`cross_metric.py` 全指标）**
| 系统 | CROSS_OUT | CROSS_R | R@all | 输出量 |
|------|:--:|:--:|:--:|:--:|
| base（原权重） | 0.745 | **30.2** | **0.791** | 40.5 |
| boost（提权） | 0.757 | **33.2（最差）** | **0.688** | 42.3 |
| revert（回退） | 0.762 | 30.6 | 0.728 | 39.8 |

- **base vs revert 差 6.3pp = 同权重重跑噪声底**；权重真实效应 ≈ R@all 0.728→0.688 = **-4pp**。
- **boost 的 CROSS_OUT/R 反而更差**（0.757、33.2），并非更好。
- 分能力 CROSS_OUT（temporal 类）：base 0.719 / boost **0.768** / revert 0.744 —— 抬权重让 temporal 类查询最跨场景。

**② 主指标：纯故事对比（无查询，5 维含"时序结构"，`story_alone_multi.py`，30 非弃权）**
| 系统 | richness | authenticity | coherence | memory_value | **structure 时序结构** |
|------|:--:|:--:|:--:|:--:|:--:|
| base | 4.23 | 4.57 | 4.23 | 4.37 | 3.63 |
| boost | **4.53** | **4.70** | 4.27 | **4.60** | **3.93** |
| revert | 4.20 | 4.53 | 4.10 | 4.43 | 3.60 |

- **base vs revert 噪声底收紧到 ≤0.07（5 维判官更稳）** → **boost 全面超出噪声底**：structure **+0.30**（≈噪声底 0.03 的 10 倍）、richness +0.30、memory_value +0.23。
- **修正**：之前 4 维指标"主指标无增益"是**度量盲区**（漏测时序结构）；加入 structure 后，**TEMPORAL 提权让主指标真实变好**，尤其时序结构最明显（q21: base 4→boost 5）。

**③ 抽取层（0822 闭环，子集 batch6）**
| 系统 | 节点 | 边 | 节点精确 | 边精确 | 边类型一致 |
|------|:--:|:--:|:--:|:--:|:--:|
| v4 基线 | 87 | 170 | 87.4% | 35.9% | 77% |
| 改动后(加时间引导) | 134 | 292 | 61.2% | 21.2% | 54.8% |
| 回退后 | 97 | 192 | 79.4% | 32.3% | 59.7% |

**机制**：
- **检索**：TEMPORAL 反向（尤其时间锚点 0.8）使 th 锚点成为跨期"枢纽"，在通用查询上让检索过度扩散 → R@all 降、CROSS_R 升。
- **抽取**：时间节点规则在常规对话上把事件拆出额外 THING 时间节点 → 节点/边数暴涨、精确率与边类型一致率崩塌。
- **主指标无增益**：故事质量对提权不敏感（纯故事对比全在噪声带内）。

> ⚠️ 结论：**TEMPORAL 的补全不可用"全局化"方式落地**（改默认 prompt / 改全表权重）。受控实验证明其价值，但生产集成必须**手术式定向**，否则造成既有评测回归。两处代码均已恢复 0822/0818 基线。

### 5.4 正确的补全路径（手术式）—— ①抽取落地、②检索已废弃
1. **抽取侧 ✅ 已落地**：在 [dba.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/extraction/dba.py) 增加"时序叙事守卫"`has_temporal_signal()`（判定 ≥2 个不同日期/时期锚点，如 周三/上周/这周；刻意排除"今天/昨天/凌晨/每天"等弱时间副词与仅 era 词），命中才叠加时间节点规则 + TEMPORAL 连边规则与示例（`node_chain_temporal`/`edge_chain_temporal`），否则走标准逻辑。
   - 验证：`lifemem_temporal`（6 时期词）→ 命中启用 TEMPORAL；`lifemem_subset`（1 时期词+7 era 词）→ **不命中**，走标准路径 ⇒ **0822 闭环零回归**；`每天晨跑/今天加班`→ 不命中。
2. **检索侧 ✅ 已删除**（原"手术式时序"彻底清除）：曾在 [retriever.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/retrieval/retriever.py) 加 `_is_temporal_query`/`_is_time_node`/`_expand_temporal_reverse` + 时间锚点豁免 + 定向 TEMPORAL 权重（前向0.8/反向0.5）——增量复测证实其为 **空操作**（ON/OFF 路径一致、structure 无差，见 §7.0），且"时间→事件"实为**词面假象**（§8 RC3）。**已从 `retriever.py` 与 `run_p_baseline.py` 删除**（含 `temporal_mode` 参数与 `--temporal-mode` CLI），主检索器 `retrieve()` 回到纯基线、零污染，改走 §九 的独立单跳时序工具。

### 5.5 现状
- `dba.py`：**手术式守卫**（`has_temporal_signal` 命中才启用 TEMPORAL 抽取），默认 standard 逻辑→常规语料零回归。
- `retriever.py`：**手术式时序代码已删除**（主检索器 `retrieve()` 纯基线、零污染），保留独立 `temporal_lookup` 工具（见 §九）；`jump_axis.py`：**恢复基线**（不全局提权）。
- 评测：`judge_story_alone.py` / `story_alone_multi.py` 已加**"时序结构"维度**，主指标能测出时序增量。
- TEMPORAL 抽取/检索能力以受控实验形式保留（`lifemem_temporal` 数据集、`run_temporal_experiment.py`、`run_temporal_lifemem.py`、`lifemem_full_temporal_augmented.yaml`），作为后续（TIME 类型、更细锚点、story 侧复测时序增量）的验证基础。

***

## 六、结论与后续建议
- **TEMPORAL 是事件检测/时间线重建的"必要使能层"**：受控实验证明——提供事件→时间锚点与纪元骨架后，跨期时序可重建（一致率 1.0）、期内顺序升至 0.88，且 R@all 大幅回升。
- **补全必须手术式，不能全局化**：全局"改默认 prompt + 全表提权"虽让主指标真实变好（5 维含时序结构后 boost 全面超噪声底，structure **+0.30**），但在常规语料造成抽取过度拆分（精度 87→61%）、检索 R@all 下降（约 -4pp）、CROSS_R 变差——这些代价与主指标收益都不该全局承担。生产落地：**抽取侧守卫已实现**（时序叙事才启用 TEMPORAL，常规语料零回归，见 5.4①）；**检索侧不再做"定向启用 TEMPORAL"（②已废弃），改走 §九 的独立单跳时序工具**。
- **根本取舍**：系统是"模拟联想、无离线经历"（见 §九），时序层是**工具性的排序/锚定语用**，不是"时间记忆"。因此时序只按需启用（跨期/历程/时间线重建 + 稳定时代锚点），细粒度指示词（昨天/今天）交给语义联想。
- **仍差"期内精细推进顺序"一层**：需区分时序推进边（SEQUENCE/时序因果）与静态因果；或把 TEMPORAL 锚点做细（高一/高三、大一大二…）。
- 下一步（见 §九）：实现 `temporal_lookup` 独立单跳时序工具；抽取侧保留手术式守卫；以"面向输出/任务"口径评估其正增益，并考虑 TIME 类型与"去词面化"复测。

***

## 七、时序关系设计重构：本轮探讨收敛（未实施，仅论证）

> ⚠️ 本节为 **2026-08-27 深夜的探讨记录**（保留关键证据，删除被推翻的中间计划）。流程：§5.4 的手术式检索②经增量复测证实 **≈ 空操作**；随后围绕"TEMPORAL 边应如何定义发散/方向"多轮收敛。**最终结论（见 §九）：时序不走通用联想检索、不揉进主检索器，而是做成一个独立的单跳查询工具 `temporal_lookup`，把联想选择权交给下一次对话/主 LLM。** 本节仅保留支撑该结论的证据（7.0/7.1/7.4）与关键定性（锚点角色、模拟联想无体感时间），被推翻的计划（阀向量、结构底线+目的门控、time 不入 top-K、手术式检索②）已在 §5.4/§六/§九 标注废弃。

### 7.0 触发：增量复测暴露"手术式检索=空操作"

固定目的（`--purpose-mode inject`）下，手术式时序 ON vs OFF 在 5 条时序查询、5 维纯故事判定（[story_alone_inj_temporal.json](file:///c:/Users/makot/Desktop/Ariadne/eval/outputs/story_alone_inj_temporal.json)）：

| 系统 | richness | authenticity | coherence | memory_value | **structure** |
|------|:--:|:--:|:--:|:--:|:--:|
| inj_off（基线检索） | **5.0** | 4.8 | 4.4 | **5.0** | 4.8 |
| inj_on（手术式时序开） | 4.6 | 4.8 | 4.6 | 4.8 | 4.8 |

- **structure 逐条完全一致（4/5/5/5/5）**；th 锚点出现次数 ON/OFF 各 56 次、路径集合逐条相同；R@all 均 0.706。
- ⇒ 温和手术式（`TEMPORAL_FWD_BOOST 0.8 / REV 0.5` + 反查 + 时间锚点豁免）**没有把跨期锚点拉进 top path，等于空操作**。此前 +0.30（§5.3② structure 3.63→3.93）只来自**全局 boost**（q21: th1→th0,th1,th2）。

### 7.1 挤占假设验证（用 candidates 层而非 path 层）

用度量层（candidates，R@all/CROSS_OUT 读它）测 base(基线)→boost(全局提权)：

- **15/35 的非 time 域查询中，time 域(th*)候选被注入变多**：q02(college) 1→3、q10(social) 0→2、q26(emotion) 0→2、q21(health) 1→3、q16/q20/q29 0-1→1-2。
- 这些查询 gold 不在 time 域，time 节点挤占 top-K 席位 → gold 被挤出 → **R@all 0.791→0.688、CROSS_OUT 变差**。坐实"时序节点挤占了其它关系边节点位置"。
- **关键前提**：0 条查询的 gold 节点在 time 域（gold 域分布无 time）→ 把 time 节点排除出 top-K **绝不可能伤 R@all**，只会去噪声、CROSS_OUT 必然改善。
- **结构前提**：candidates（度量）与 path（StoryRank）天然分离，path 由 hop_history 回溯父链构建 →「从 candidates 剔除、留 path 做时间轴」可行。

### 7.2 方案迭代记录（探讨过程）

1. **想法① "time 节点不进入 top-K"**：time 从 candidates 剔除、留在 path 做时间轴；因进不了 answer 榜，可放心提高 TEMPORAL 权重拉锚点进 path → **时序结构增量（+0.30）可拿，而 R@all/CROSS_OUT 反改善**、且收益只在时序查询（门控）。非时序查询彻底不动。
2. **用户重构——转向边的发散属性（不只节点分类）**：
   - 时间戳→time 节点；**单向 TEMPORAL（事件→时间）的盲区**：A→th、B→th 都汇到 th，则 A、B 在时间轴上**共现**——【昨天我干了什么】命题 = th→{A,B,C} 的**共时展开**；而"时间→无关事实"与"时间→共时事实"是**同一条反向边**，只能靠目的区分。
   - **反向发散除非明确目的应停止**，由**目的回归阻断**执行。
3. **关键定性——判别变量是"锚点角色"**：
   - 时间做主语（昨天我做了什么）→ 种子该是 th → 开**反向** th→事件；
   - 事件做主语 + 问 when（什么时候开始晨跑 = q21）→ 种子该是事件 → 开**正向** event→th（现默认即如此）。
   - **时间词属双向歧义**：两类查询都是时间词触发，但方向相反（"什么时候开始晨跑"= event→th，"昨天我做了什么"= th→event）。
4. **评判"阀向量"（Point 3）**：一维"时序/事实"开关只能管**事实↔时间**，管不了**事实→事实**（CAUSAL/SEQUENCE…）。对混合查询（"昨天为什么失眠"：时间侧 th→失眠 + 事实侧 失眠→加班因果）事实→事实本就该开放，因此阀向量"影响不了混合查询"是**合理且正确**的。结论：**阀向量维度错了，退化为"锚点角色判别"**。

### 7.3 双种子与反漂移 · 关键事实（最终方案见 §九）

- **双种子普遍**（后续核实：21/35）：种子集同时含 th 与事件 → 若正反向同开，共享枢纽 th 上形成 `事件A→th→事件B→th'→…` 的**跨期 ping-pong 漂移**。
- **定性结论**：逐节点定方向（时间→反向、事实→正向）+ 单跳不链式，即可天然反漂移——这正由 §九 的"独立单跳时序工具"实现，因此不必在通用检索器里做"结构底线 + 目的门控"那套复杂规则（**已废弃**）。
- 佐证图级事实：th 各带 SCENARIO 入边（见 7.4），故只要工具**只走 TEMPORAL 单跳、不链式扩展**，就不可能经 SCENARIO 或第二跳时间节点跨期漂移。

### 7.4 图级核实结论（支撑上述规则）

| 项 | 结论 |
|----|------|
| **双种子** | **21/35** 查询种子集同时含 time(th*)与事件。但 q21–24 的 `th=['w30']` 是**内容启发式误判**（"站会是早上9点半"，THING 命中"早上"），**非真锚点** → "time"识别应改**真 th* 锚点/正式 TIME 类型**，不能靠 `_is_time_node` 内容匹配 |
| **th 邻边（322 图）** | th0(高中)/th1(大专)/th2(工作) 各为 **TEMPORAL 入度 3/3/2**；但各带 **SCENARIO 边** → th0→h25（thing，4 连接）、th1→c12（3）、th2→w35（4）→ **th 并非只靠 TEMPORAL 可达**，反漂移须覆盖非 TEMPORAL 入边 |
| **建模粒度** | th 是**粗粒度人生阶段**（高中三年/大专三年/当前工作），非日期；【昨天我干了什么】在 322 图**无真实 th 锚点**（落到 w30 误判）。细粒度时间（【上周一晚上】【周三凌晨】）只能靠**抽取侧 temporal 引导**新建（=17 节点测试集 T1–T7），但此类引导在常规语料会过度拆分（§5.3③） |
| **"时间线一致性=满分"本质** | `timeline_pairwise_acc`（[run_temporal_experiment.py](file:///c:/Users/makot/Desktop/Ariadne/eval/run_temporal_experiment.py#L374-L428)）= 读图上 TEMPORAL 边(事件→时间) 得锚点，**按 gold 时间秩排序**比 gold 顺序 → 一旦可锚定**必然 1.0**。它是"**事件被正确锚到 gold 时间**"的检查，**不是**测"时间→事件反向联想"。且基线抽取 **temporal 边召回=0** → 真实链路 anchorable=0、时间线能力基本没落地；**th→事件 反向联想从未实现、也从未被该指标检验** |

### 7.5 现状与待决（收束到 §九 方向）

- **已确认**：现有"时间线 1.0"靠单向 event→时间 + 事后排序；真正的 th→事件（【昨天我干了什么】）当前系统**未实现**；§5.4 的手术式检索②经增量复测证实 **空操作**（7.0）与**词面假象**（§8 RC3）。
- **收敛**：不再在通用联想检索器里做时序，改走 §九 的**独立单跳时序工具**。残留待决并入 §九（TIME 类型、细粒度锚点来源、现实/虚拟 present）作为其扩展项。

***

## 八、真实时序场景分型诊断实验（定位弱势与根因）

> 为真正找出系统弱项，新建**跨期 + 细时间**场景，用「查询类型 × 图来源」隔离矩阵跑全链路（[run_temporal_rooted.py](file:///c:/Users/makot/Desktop/Ariadne/eval/run_temporal_rooted.py)），产出 [temporal_rooted.json](file:///c:/Users/makot/Desktop/Ariadne/eval/outputs/temporal_rooted.json)。

### 8.1 场景与查询集
- **数据集**（新造，24 节点 / 24 边）：用户「高中→大学→工作」成长线 + 工作期细到 去年底/上个月/上周/昨天/今天。时间锚点 T1..T11（单调秩），事件《e0..e12》，含 `temporal`(13) / `sequence`(10) / `causal`(4)。
  - `data/dba_eval/lifemem_crossperiod.yaml`、`_dialogs.yaml`（10 批，显式时间词）、`_queries.yaml`。
- **查询分型**（每类含手写目的 + gold_expected）：TA 事件→时间、TB 时间→事件（2条）、TC 期内线、TD 跨期、TE 事实+时间背景。

### 8.2 三大根因（诊断结论）

**RC1 —— 抽取层是硬瓶颈：`R@all` 与 `可锚定率` 解耦**
| 抽取配置 | 节点 | 边 | 节点召回 | **temporal 边召回** | 节点精确 | 边精确 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| base | 23 | 43 | 54.2% | **0/13（0%）** | 56.5% | 23.3% |
| temporal | 38 | 66 | **95.8%** | **8/13（61.5%）** | 62.2% | 21.2% |

- 基线抽取 **零 TEMPORAL 边** → 全部查询 anchorable=0、pairwise=N/A（无序事实堆），**即便 R@all 高也排不出时间线**；加引导可到 61.5%，但**过度拆分**（节点 23→38、边 43→66、边精确 23%→21%），与 §5.3③ 一致。

**RC2 —— 检索层：跨期（TD）是真弱势**
- **gold 图（数据完美）下 TD R@all 仅 0.17**（漏 高中/入职），temporal_boost 才 0.67。→ "**多 th 齐聚**"是真实检索困难，正对应 §7.3 需要的**目的门控放行跨期**。

**RC3 —— 检索层："时间→事件"是词面假象（掩盖了方向未实现）**
- gold 图默认 jump（反向权重=0、无 th→事件遍历）下 TB R@all 竟 **1.0**：因事件的 content **字面含"昨天"**，被向量搜索直接命中。
- ⇒ **纯图结构的"时间锚点→事件"反向关联当前系统根本没实现**，回答靠事件文本里的时间词，而非 TEMPORAL 边。

### 8.3 检索矩阵（R@all / 可锚定率），按类型 × 图
| 类型 | gold 默认 | gold boost | dba_base（anchorable 恒 0） | dba_temporal |
|------|:--:|:--:|:--:|:--:|
| TA 事件→时间 | 0.50 | 0.50 | 1.00（0） | 1.00（0.5） |
| TB 时间→事件 | 1.00 | 1.00 | 0.67（0） | 1.00（0.33） |
| TC 期内线 | 1.00 | 1.00 | 0.33（0） | 1.00（0.33） |
| **TD 跨期** | **0.17** | **0.67** | 0.83（0） | 0.50→0.67（0.33-0.5） |
| TE 事实+时间 | 0.67 | 1.00 | 1.00（0） | 1.00（0.67） |

> 观察：dba_base 常出现"R@all 高于 gold"（如 TD 0.83 vs 0.17），但那是**无约束广撒网**（检索节点数更多），且 anchorable=0 → 无法定序，**高召回≠可重建时间线**。

### 8.4 局限与下一步
- **局限**：诊断数据集的事件 content 含时间词，使 TB 依赖词面匹配、无法单独检验反向遍历。若要**隔离检验 th→事件 反向**，需让事件 content **不含**时间词、仅靠 TEMPORAL 边与时间锚点相连。
- **下一步**：按 §九 落地**独立单跳时序工具**（不再在通用检索器里做目的门控/一次性 TEMPORAL）；以"去词面化"数据集复测，单独验证"时间→事件"反向是否真正生效。

***

## 九、收敛方向：独立单跳时序工具（temporal_lookup）【定论】

> 定论：系统是「模拟联想、无离线经历」的检索+排序工具，**不是自然人的时间记忆体**。因此时序**不做成一套"时间记忆模型"**，而是做成**一个独立、单跳、可被上层调用的查询工具**，把联想选择权交给下一次对话/主 LLM。

### 9.1 设计
| 项 | 说明 |
|----|------|
| 定位 | 独立工具（`temporal_lookup(query)`），**不揉进主联想检索器** |
| 行为 | 语义匹配到锚节点 → 按 TEMPORAL 边**单跳**取邻居 → 扁平返回 |
| 锚点角色 | 问"某时间发生了什么"→ 锚=时间节点→返回共时事实；问"某事件何时"→ 锚=事件节点→返回其时间锚点 |
| 不做什么 | 不因果/SEQUENCE 链式扩展、不做目的打分挤占、不跨期漂移（单跳天然不链式） |
| 联想归属 | 工具只递候选；怎么解读/串因果/追下去，交给主 LLM 在对话上下文里判断 |

### 9.2 为什么这样
- **化解全部纠结**：阀向量、锚点角色、一次性 TEMPORAL、目的门控、挤占——都被"单跳 + 独立通道 + LLM 联想"自然消解，无需在通用检索器里实现。
- **契合「模拟联想、无体感时间」**：时序只是一次符号查找，不是时间记忆；不追求让 agent"体验时间"。
- **主检索器零污染**：纯事实/因果查询不再受时序侵入（0 回归）。
- §8 RC3 表明：细粒度"昨天"本就被词面联想覆盖——独立工具把这块干净显式化，不干扰主链路。

### 9.3 保留 / 丢弃
- **保留**：抽取侧手术式守卫（5.4①）；§5.2 受控实验证据；§8 RC1/RC2/RC3；"锚点角色"定性；"模拟联想、无经历"认知；jump_axis 基线（不全局提权）。
- **丢弃（被推翻）**：§5.4② 手术式检索（空操作）；"time 不入 top-K"；"阀向量"；"结构底线 + 目的门控"；"每路径 TEMPORAL 一次/逐节点定方向"那套在**通用检索器内**的实现。

### 9.4 落地 / 评估
- 实现最小 `temporal_lookup`：给定查询 → 定位锚节点（时间或事件）→ 返回其 TEMPORAL 邻居。
- **评估口径改为面向输出/任务**：不是"符号匹配 gold 时序"（循环、恒 1.0），而是"开启该工具后，需要时序的查询其回答/叙事是否变好、成本是否可接受"；用 **"去词面化"数据集**（事件 content 不含时间词、仅靠 TEMPORAL 边连时间锚点）隔离验证"时间→事件"反向是否真正生效。
- **待决（扩展项）**：① 正式 TIME 类型（明确 time 节点）；② 细粒度锚点来源（抽取侧引导 vs 仅阶段级）；③ 现实/虚拟 present：时间基准指针，虚拟时钟由用户显式推动（"三年后"→ present 前移），相对时间词统一按 present 解析。

### 9.5 落地与测试（[retriever.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/retrieval/retriever.py) 的 `temporal_lookup`，[test_temporal_lookup.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_temporal_lookup.py)）

**v1 范围取定（按用户修正）**：`temporal_lookup` **仅做"时间→事件"反向**（主检索器反向权重=0 做不到的部分）；"事件→时间"(TA)、跨期(TD)、纯事实均归**主联想检索器**，本工具不承担。

| 查询 | 时间锚点 | 返回共时事件 | 判定 |
|---|---|---|---|
| 昨天用户都做了些什么 | T10（昨天） | e9/e10/e12 | ✅ |
| 上周…累人的事 | T9（上周） | e8 | ✅ |
| 高三下学期发生了什么 | T3 | e2 | ✅ |
| 用户是什么时候开始学吉他 | **未定位到时间锚点** | 空 | ✅ 正确拒绝→走主检索器 |
| 高中时期发生了什么 | th0 | h0/h20/h23 | ✅ |
| 当前工作时期怎么样 | th2 | w0/w13 | ✅ |

- **机制正确**：只认时间锚点、只做反向单跳（时间→事件）、不链式、不挤占、不侵入主检索；**TB（时间→事件，真正缺口）干净打通**。
- **范围边界（非 bug）**：TA（事件→时间）与 TD（跨期）**本就不属于该工具**，已正确拒答/交由主检索器——之前 v0 把它们塞进工具、暴露"锚点漂移/跨期不可达"，实为**范围越界**，收窄后自动消除。
- **TD 归属验证**：关键目标事件分散在 6 个不同时间锚点（T1/T3/T4/T6/T7/T8），单锚工具一次最多覆盖 1 个；§8.3 实证主检索器召回跨期更强（gold TD R@all 0.17→boost 0.67、dba_base 0.83）。
- **下一步**：接入主 agent 工具调用，做"时间推事件→`temporal_lookup`、其余→主检索器"的分流（§9.6）。

### 9.6 分工路由（TD 归主检索器）
| 查询类型 | 走哪个 | 依据 |
|---|---|---|
| 单锚点时间：昨天做了什么（TB）/ 某事件何时（TA） | `temporal_lookup`（单跳） | 单一明确时间词/单事件 |
| **跨期/历程/多主题**：从高中到工作（TD）/ 整个项目经历了什么 | **主联想检索器**（多跳） | 多个不同时间锚点 / "从…到… 一路 整个过程 历程"跨度意图 |
| 纯事实/因果/关联 | 主联想检索器（照旧） | 默认 |

- **分工本质**：`temporal_lookup` 只做**干净的单锚点取事实**；跨期联想归主检索器，符合"联想交给主检索器/LLM"。
- **TD 完整质量 = 主检索器召回 + 抽取侧手术式守卫提供的 TEMPORAL 锚点用于定序**（基线抽取无锚点则"召回多、顺序乱"，见 §8 anchorable=0）。

### 9.7 接入分流（[mcp_server.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/mcp_server.py) + [test_tool_dispatch.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_tool_dispatch.py)）
- 新增 MCP 工具 `dba_temporal_lookup`（`DBAServer.temporal_lookup` 包装 `retriever.temporal_lookup`），handler/状态文本已接线；描述中明确"只在问【某个具体时间发生了什么】时用"，并逐条列出【不要用于】的场景（事件→时间/因果/跨期/无单锚点）。
- `dba_query_memory` 描述同步扩充适用范围（除 temporal 外的因果/现状/事件→时间/跨期/事实）。
- **LLM 路由观察**（OpenAI function-calling，7 条）——**全部正确**：
  | 问题 | 判断 → 工具 |
  |---|---|
  | 昨天我做了什么 / 上周…累人的事 / 高中时期都发生了什么事（TB） | ✅ `dba_temporal_lookup` |
  | 用户是什么时候开始学吉他的（TA） | ✅ `dba_query_memory` |
  | 从高中到工作…完整历程（TD） | ✅ `dba_query_memory` |
  | 为什么最近心情不好（因果）/ 喜欢喝什么咖啡（事实） | ✅ `dba_query_memory` |
- **结论**：工具描述调优到位，LLM 能正确区分"时间→事件"（走 temporal_lookup）与"事件→时间/跨期/因果/事实"（走 query_memory）。

### 9.8 端到端 + 完整数据集（322 图）验证（[test_temporal_full.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_temporal_full.py)）
- **端到端（DBAServer.temporal_lookup，MCP 同一入口）**：`高中→th0→{h0,h20,h23}`、`大专→th1→{c0,c11,c14}`、`工作→th2→{w0,w13}` 全部正确；`昨天用户做了什么` 正确拒答（未定位时间锚点 → 提示改 query_memory）。
- **完整数据集时间锚点覆盖**：`_tool_is_time_node`（内容正则）判定 14 个节点为"时间锚点"，但**仅 3 个（th0/th1/th2）是真锚点**，其余 11 个是"事件节点内容里带时间词"的误判（w13"现在这份工作做了三年"、w30"站会是早上9点半"、h25/h32/c12 等），它们**没有反向 TEMPORAL 邻居**。
- **修复：加"真锚点校验"** `_has_reverse_temporal`（只认带反向 TEMPORAL 邻居的候选）——误判节点不再被当作锚点（"工作做了三年"查询绕过 w13）；完整数据集覆盖仍显示 3 个真锚点正确返回共时事件。
- **多锚点返回（按用户思路）**：`temporal_lookup` 返回**全部候选锚点的事实组** `matches`（每锚点一组），默认 `max_anchors=3`，按相关性从高到低，**交由 LLM 挑选最相关的一个/多个**（避免硬性消歧）。例：gold 图"昨天我做了什么"→ T10(昨天,e9/e10/e12)+T11(今天,e11)；"高三下学期"→ T3/T2/T1 三个高相关时期。`max_anchors` 过大（如 5）会带出大四/大二等偏噪声时期，故默认取 3。
- **残余权衡**：多锚点在"略歧义"（如"三年"→大专三年/工作三年）时利大于弊；但当时间词本身较精确（如"高三下学期"）时 top-1 已够，多返回邻近时期由 LLM 自行取舍。属**锚点语义定位质量**问题，非结构问题。

## 风险/待决（承接上文 §9.4 扩展项）
- TIME 节点类型仍另议（避免内容正则误判，为最语义顺改造）。
- 细粒度锚点（昨天/上周）在 322 图缺真实节点，仅靠抽取侧引导；虚拟时钟 present 模型并行推进。

### 9.9 清理 + 稳定性测试
- **清理（✅ 已完成）**：旧"手术式"时序代码彻底删除——[retriever.py](file:///c:/Users/makot/Desktop/Ariadne/dba_pipeline/retrieval/retriever.py) 移除 `TEMPORAL_QUERY_RE`/`TIME_ANCHOR_RE`/`TEMPORAL_FWD_BOOST`/`TEMPORAL_REV_BOOST`/`temporal_mode` 参数/`_is_temporal_query`/`_is_time_node`/`_expand_temporal_reverse` 及 `retrieve()` 内 3 处时序分支；[run_p_baseline.py](file:///c:/Users/makot/Desktop/Ariadne/eval/run_p_baseline.py) 移除 `temporal_mode` 参数与 `--temporal-mode` CLI。主检索器 `retrieve()` 回到**纯基线**（编译通过、查询"昨天用户都做了些什么"返回 peak_memories=13 含 e10/e9/T10）。
- **稳定性测试（✅ [test_temporal_stability.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_temporal_stability.py)，18 项 PASS，exit 0）**：
  - **INV1 确定性**：同一查询重复 3 次，`temporal_lookup` 的锚点/事实完全一致。
  - **INV2 防误判锚点**：用 11 个"含时间词事件节点"内容作查询，解析出的锚点绝不再落到这些误判节点（而是 th0/th1/th2 或空）；"高中时期"→th0。
  - **INV3 多锚点上限**：`matches ≤ max_anchors(3)`，按相关性从高到低（高三下学期→T3/T2/T1）。
  - **INV4 非时序拒答**：TA"什么时候开始学吉他"→空 matches（应走主检索器）。
  - **INV5 主检索无回归**：注入目的 3 次（每次 `start_session()`），关键基线节点 e10/e9/T10 稳定出现（n=13/13/13）。
- **关键认识**：`retrieve()` 是**会话态**——`PathTracker` 的 `lifetime_counts` 跨调用累积（终身增强，刻意设计），故同实例多跑**不 bit 确定**，但关键节点稳定、无回归。这是设计使然，非缺陷。

### 9.10 抽取侧时序规则测试（[test_dba_temporal.py](file:///c:/Users/makot/Desktop/Ariadne/tests/test_dba_temporal.py)，11 项 PASS）
- **范围**：守卫 `has_temporal_signal()` 的触发判定 + 语义边界（仅 1 个日期词 / 同一词重复 / 仅 era 词 / 弱时间副词 → 不命中；≥2 个不同日期词 → 命中），以及 `NODE_TEMPORAL_EXTRA`/`EDGE_TEMPORAL_EXTRA` 规则串完整性（含"时间节点"/"TEMPORAL"/"单向"）。
- **离线确定性**：不走 LLM，仅测纯函数与常量；零回归定量由 `eval/run_dba_extraction_eval.py`（带 LLM 闭环）另覆盖。
- **可运行方式**：无 pytest 环境时 `python tests/test_dba_temporal.py`（自包含 runner），或 `python -m pytest tests/test_dba_temporal.py`（若已装 pytest）。

### 9.11 闭环 + 增量验证（[test_temporal_closedloop.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_temporal_closedloop.py)）
- **闭环**（路由→工具→取回→生成）：
  - `昨天用户都做了些什么？` → 路由 `dba_temporal_lookup` → 取回「昨天」= {被批评/倾诉/迟到扣钱}（+「今天」候选）→ LLM 生成共情回复。✅
  - `为什么用户最近心情不好？` → 路由 `dba_query_memory` → 取回一条**跨期时间线叙事**（高二吉他→高考→大学→工作→昨天）。✅ 分工正确。
- **增量**（同一"时间→事件"查询两种工具对比）：
  - `temporal_lookup` = **时间切片**：精确返回「昨天」3 件共时事实，聚焦、直接；
  - `query_memory` = **跨期叙事**：把相关事件串成完整时间线，"昨天"被埋进整段历史里，**聚焦精度低**。
- **结论：架构已能满足时序需求**——时间→事件的**精确聚焦**由 `temporal_lookup` 提供，跨期/因果由主检索器承担，二者互补、分工正确，闭环可给出贴合的回答。
- **告诫（承接 RC3，未消除）**：当前事件 content 仍带时间词（"昨天被点名批评"），故主检索器也能向量命中"昨天"几件事——工具的增量目前主要体现在**结构化的时间切片精度**，而非"信息有无"。**仍需"去词面化"数据集确证**反向遍历不靠词面（见 §7 待决⑤）。

### 9.12 去词面化 RC3 确证（[verify_temporal_delex.py](file:///c:/Users/makot/Desktop/Ariadne/eval/verify_temporal_delex.py)，生成 [lifemem_crossperiod_delex.yaml](file:///c:/Users/makot/Desktop/Ariadne/data/dba_eval/lifemem_crossperiod_delex.yaml)）
- **做法**：把 crossperiod gold 的事件 content 里的时间词全部去掉（只保留时间锚点 T1..T11 的时间词；事件仅靠 TEMPORAL 边连到锚点）。
- **temporal_lookup（结构）**：锚定 `T10=昨天` → 取到 `{e9被批评, e10倾诉, e12迟到扣钱}`，3 条 content **均不含时间词** → **纯靠 T10 反向 TEMPORAL 边**。✅
- **主检索器 retrieve（对照，同一去词面图）**：取到 `[e10, e9, e8(上周), e1(高三), e7(上月), T10, T8, T9…]` —— **极混杂**、昨日事件只取到 `2/3（缺 e12）` → 主检索器**无时间聚焦能力**，靠目的+因果/先后链作文法。
- **结论：RC3 确证** —— `temporal_lookup` 的"时间→事件"反向遍历**靠图结构（TEMPORAL 边）**，非词面；主检索器确有词面/关联成分（原 TB=1.0 即此）。工具的增量从"结构切片精度"升级为"**结构性能力**"。

### 9.13 时序动态更新测试（[test_temporal_dynamic.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_temporal_dynamic.py)）
- **场景**：模拟 dba 增量维护，向图谱新增时间锚点/事件/TEMPORAL 边，验证工具即时反映。
- **发现并修复一个真实缺陷**：`TOOL_TIME_ANCHOR_RE` 的 `^(上|本|下)?周` **漏了"这周"**（动态新建的时间锚点 n15"这周"不被识别 → 锚到错误的 T9"上周"）。**修复**：改为 `^[这上本下前那]?周`（覆盖 这周/上周/本周/下周/前周/那周/周）。
- **修复后**：
  - 新增时间锚点 n15（id 为 nX、非 th/T 前缀）→ `_tool_is_time_node(n15)=True`（**内容正则**识别）；
  - `temporal_lookup("这周用户都做了什么")` → 锚定 n15 → `{n16准备周报, n17组织评审}`（+ 次要候选 T9）；
  - 移除 `n17→n15` 边 → n17 即时消失、n16 仍在。
- **结论**：工具对**动态新增/删除**的时间锚点、事件、TEMPORAL 边都能正确反映；非 th/T 前缀的时间节点靠内容正则识别（已覆盖"这周"）。

### 9.14 抽取侧真实验证：③零回归 + ②细粒度锚点产出
**③ 零回归（0822 subset，当前手术式守卫，[dba_extraction_subset_guard.json](file:///c:/Users/makot/Desktop/Ariadne/eval/outputs/dba_extraction_subset_guard.json)）**
- 节点：gold=100 / DBA活跃=79 / 匹配=72；**node_precision=91.14%**、type_accuracy=58.33%。
- 边：gold=138 / DBA=230；edge_recall=34.78%、edge_precision=33.48%、**edge_type_accuracy=66.23%**。
- **建图 0 条 TEMPORAL 边** → 守卫对非时序子集**未触发**，走标准链 → **未过度拆分**（活跃节点 79，远低于全局提升版的 134），node_precision 91% 高位、无退化。
- **结论：零回归成立**（对比全局版 node 61.19%/292 边的过度拆分态）。

**② 细粒度锚点产出（[verify_temporal_extract.py](file:///c:/Users/makot/Desktop/Ariadne/eval/verify_temporal_extract.py)）**
- 时序叙事对话"这周很累：周一感冒发烧…周五恢复" → `has_temporal_signal=True` → 产出 **6 条 TEMPORAL 边**（事件→各日）+ **时间锚点 THING 节点**（这周/周一…周五）。
- 非时序对话"喜欢喝手冲咖啡…" → `has_temporal_signal=False` → **0 条 TEMPORAL 边**（标准链，无时序副作用）。
- **结论**：守卫触发时**能可靠产出细粒度时间锚点 + 单向 TEMPORAL 边**，正是 `temporal_lookup` 所需结构；非时序不产、零回归。
- （测试中修正一处脚本 bug：RelationType 为 Enum，`str(rel).lower()=="temporal"` 误判为 0，改为枚举直接比较。）

### 9.15 MCP 链路检查（[test_mcp_link.py](file:///c:/Users/makot/Desktop/Ariadne/eval/test_mcp_link.py)）
- **编译/导入**：`mcp_server.py`/`retriever.py` `py_compile` 通过；`DBAServer` + `create_mcp_server` 构建成功。
- **工具注册**：7 个工具含 `dba_temporal_lookup`；server 注册了 `tools/list` + `tools/call`（含 ping/server/discover）。
- **数据通路（经真实 MCP handler→DBAServer→retriever）**：`dba_temporal_lookup("昨天用户都做了些什么？")` → 锚点 `['T10','T11']`、共时事实 `{e9,e10,e12,e11}`。
- **修复一处接线缺口**：`DBAServer.temporal_lookup` 原**未透传** `max_anchors`（schema 声明但被忽略），已补上并验证 `max_anchors=1 → matches=1` 生效。
- **无残留引用**：mcp_server 不引用已删除的 `temporal_mode`/`_is_time_node`/`_expand_temporal_reverse`。
- **结论：MCP 链路完全修复**，时序工具经标准 MCP 协议层可用。
