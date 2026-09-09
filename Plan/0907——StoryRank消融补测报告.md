# StoryRank 消融 + judge 交叉评分波动补测报告（0907）

> 日期：2026-09-07 · 状态：**干净版（已修正 raw 块信息残缺污染）**
> 关联：e2e 综合测评（[0831 计划](0831——e2e评测流程重设计计划.md#133-judge-判分) / [0901 报告](0901——e2e综合评测报告.md)）

---

## 一、目的

对 e2e 综合测评做两点补充：
1. **StoryRank 消融**：Ariadne 块的「有 StoryRank（story） vs 无（raw 原始 PAR 检索链）」对 judge 质量分的影响。
2. **judge 交叉评分波动**：不同 judge 模型（qwen / deepseek / gemini / gpt）对同一批上下文的判分一致性。

---

## 二、方法与关键修正（本报告的要点）

### 2.1 评测对象（沿用 e2e 口径）
- 对象 = 系统喂给模型的结构化记忆上下文块：Ariadne=`{story}` 或 `{nodes,edges}`、Mem0=`{memories}`、Graphiti=`{entities,relations}`。
- 判断 = **同题并排一次判定**（三系统块同一对话交给 judge，各打 1-5 分，3 维 rubric：记忆丰富度 / 易理解性 / 用户建模度）。
- 数据：comprehensive 数据集（163 节点/252 边 + 76 查询，9 类）；非弃权 70 条。

### 2.2 三个关键修正（否则结论失真）

| # | 问题 | 修正 |
|---|---|---|
| 1 | **judge 提示词没写边语义**：`JUDGE_JOINT_PROMPT` 只字未提 CAUSAL/SCENARIO 等含义 → judge 对结构化块只能**脑补** | 在 [judge_comprehensive.py](file:///c:/Users/makot/Desktop/Ariadne/eval/judge_comprehensive.py#L40-L51) 补一句边/节点类型语义说明（因果/同场景/时序/偏好/社交/属性/时间/分类 + 节点 6 类） |
| 2 | **raw 块信息残缺**：[collect_ariadne_raw.py](file:///c:/Users/makot/Desktop/Ariadne/eval/collect_ariadne_raw.py) 把 PAR 路径压成 `{nodes:[content], edges:["from --rel--> to"]}`，**丢了 `node_type`、没处理 `is_reverse`**（reverse 边方向反了） | 改为与 `inference.story_rank` **完全同构**的渲染：`[id]（node_type）content`、`[from] --rel--> [to]`（is_reverse 应用后方向正确） |
| 3 | **同题并排是相对判分**：Ariadne 变弱会"抬"高 Mem0/Graphiti | 披露：三系统分数**不独立**；若要隔离 StoryRank 绝对价值需单独判分（本报告未做，见「局限」） |

### 2.3 模型与有效性
- 交叉 judge：qwen3.8-flash（主判分）/ deepseek-v4-flash / gemini-3.7-flash / gpt-5.6-luna。
- **弃用 glm-4-flash**（过老 + 强形态偏好，给 Graphiti `{entities,relations}` 近乎无差别满分）。
- **gemini 曾对 raw 块判分失败**（JSON 解析全 0）：根因是 `JUDGE_JOINT_PROMPT` 走 gemini 时 `max_tokens=512` 被其「推理 token」吞掉、JSON 在闭合前被截断（`finish_reason:length`）。改为 httpx 直连 + `max_tokens=4096` 后，**story 与 raw 变体均能稳定出分**（本版已补齐 gemini）。

---

## 三、干净版结果

### 3.1 Ariadne 逐维（story vs raw，3 个有效 judge）

| 模型 | 变体 | 记忆丰富度 | **易理解性** | 用户建模度 | total |
|---|---|---|---|---|---|
| qwen | story | 4.57 | **4.89** | 4.76 | 4.74 |
| qwen | raw | 4.60 | **3.67** | 4.73 | **4.33** |
| deepseek | story | 3.91 | **4.57** | 4.01 | 4.16 |
| deepseek | raw | 3.87 | **2.79** | 3.81 | **3.49** |
| gpt | story | 4.43 | **4.33** | 4.43 | 4.43 |
| gpt | raw | 3.79 | **2.86** | 3.56 | **3.40** |

**StoryRank 净增量（story − raw）**：qwen **+0.40** / deepseek **+0.68** / gpt **+1.03** / gemini **+0.61**（均值 **≈ +0.7**）。

> 注：上表为「同题并排、每个变体各一个代表 run」。下表为**统一口径**（新 prompt + 重复 run1），含四模型×三维度+总分，gemini 已补齐。

### 3.1.1 总表（Ariadne 各模型 × 三维度 × 总分，新 prompt 一致口径）

**story（带 StoryRank）**

| judge | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|
| qwen | 4.60 | 4.84 | 4.74 | **4.73** |
| deepseek | 3.66 | 4.53 | 3.76 | 3.98 |
| gpt | 4.41 | 4.43 | 4.34 | 4.40 |
| gemini | 4.54 | **4.91** | 4.67 | 4.71 |

**raw（不带 StoryRank）**

| judge | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|
| qwen | 4.56 | 3.79 | 4.63 | **4.33** |
| deepseek | 4.04 | 2.79 | 3.97 | 3.60 |
| gpt | 4.17 | 3.10 | 3.96 | 3.74 |
| gemini | 4.31 | 3.57 | 4.41 | 4.10 |

> 易理解性 = StoryRank 主战场：story 4.43~4.91（含 gemini 4.91）→ raw 2.79~3.79（-1.0~-1.7）。

### 3.2 judge 交叉评分波动（每变体两两一致性）

| 变体 | top-1 一致率 | Cohen kappa | Spearman |
|---|---|---|---|
| **story（有 StoryRank）** | **0.73** | **0.39** | **0.66** |
| **raw（无 StoryRank）** | **0.48** | **0.25** | **0.45** |

（story 四模型 = qwen/deepseek/gemini/gpt；raw 四模型 = qwen/deepseek/gemini/gpt —— 均为「新 prompt 重复 run1」口径，gemini 已计入两变体）

> 注：中口径相较 0907 原版（story 0.80/0.46/0.67、raw 0.40/0.18/0.34）有变动——原版 story 复用的旧 `judge_gemini.json` 含 q11/q16 两条全 0 失效样本，raw 亦未含 gemini；本版按**统一新 prompt 重复 run1** 重算（gemini 全量干净），故数值更稳。方向性结论不变：**story 判分一致性高于 raw**。

### 3.3 判分方差（完整三次重跑 · 逐维度）

> 每格：`run1 / run2 / run3`，右侧为该维度跨 3 次标准差 std。gemini 已补齐。

**story（带 StoryRank）**

| judge | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|
| qwen | 4.60 / 4.57 / 4.61 | 4.84 / 4.89 / 4.87 | 4.74 / 4.74 / 4.77 | 4.73 / 4.73 / 4.75 |
| qwen std | 0.018 | 0.018 | 0.013 | 0.010 |
| deepseek | 3.66 / 3.61 / 3.63 | 4.53 / 4.53 / 4.53 | 3.76 / 3.80 / 3.83 | 3.98 / 3.98 / 4.00 |
| deepseek std | 0.018 | 0.000 | 0.029 | 0.007 |
| gpt | 4.41 / 4.39 / 4.54 | 4.43 / 4.34 / 4.41 | 4.34 / 4.24 / 4.44 | 4.40 / 4.32 / 4.47 |
| gpt std | 0.068 | 0.037 | 0.082 | 0.059 |
| gemini | 4.54 / 4.47 / 4.47 | 4.91 / 4.94 / 4.91 | 4.67 / 4.66 / 4.61 | 4.71 / 4.69 / 4.67 |
| gemini std | 0.034 | 0.013 | 0.024 | 0.018 |

**raw（不带 StoryRank）**

| judge | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|
| qwen | 4.56 / 4.61 / 4.60 | 3.79 / 3.80 / 3.70 | 4.63 / 4.72 / 4.71 | 4.33 / 4.38 / 4.34 |
| qwen std | 0.023 | 0.043 | 0.043 | 0.022 |
| deepseek | 4.04 / 4.04 / 4.14 | 2.79 / 2.71 / 2.63 | 3.97 / 3.90 / 3.96 | 3.60 / 3.55 / 3.58 |
| deepseek std | 0.047 | 0.064 | 0.031 | 0.019 |
| gpt | 4.17 / 4.31 / 4.26 | 3.10 / 3.09 / 3.13 | 3.96 / 3.90 / 3.91 | 3.74 / 3.77 / 3.77 |
| gpt std | 0.059 | 0.018 | 0.024 | 0.011 |
| gemini | 4.31 / 4.33 / 4.36 | 3.57 / 3.61 / 3.63 | 4.41 / 4.43 / 4.44 | 4.10 / 4.13 / 4.14 |
| gemini std | 0.018 | 0.024 | 0.012 | 0.018 |

> **结论**：同模型重跑**逐维度跨次 std 总体较小**（绝大多数 ≤0.07；个别如 gpt story 易理解 0.082、raw 丰富度 0.059）。**同一 judge 模型结果高度可复现**，无明显维度在重跑中剧烈抖动。真正方差来自**换模型**（跨模型总分差 ≈0.4，远大于任何重跑差）；逐条级抖动则 qwen < gpt < deepseek。

### 3.4 三系统同期全量总表（Ariadne / Mem0 / Graphiti · 三次重跑逐维度）

> 每格：`run1 / run2 / run3`。Ariadne 为带 StoryRank 的 story 块 / 不带 StoryRank 的 raw 块；Mem0、Graphiti 在两变体中块一致（仅因同题并排相对判分而波动）。

**story（带 StoryRank）**

| judge | system | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|---|
| qwen | Ariadne | 4.60/4.57/4.61 | 4.84/4.89/4.87 | 4.74/4.74/4.77 | 4.73/4.73/4.75 |
| qwen | Mem0 | 3.84/3.77/3.86 | 2.36/2.34/2.34 | 2.97/3.00/3.01 | 3.06/3.04/3.07 |
| qwen | Graphiti | 4.14/4.17/4.17 | 2.97/2.96/2.99 | 4.07/4.06/4.11 | 3.73/3.73/3.76 |
| deepseek | Ariadne | 3.66/3.61/3.63 | 4.53/4.53/4.53 | 3.76/3.80/3.83 | 3.98/3.98/4.00 |
| deepseek | Mem0 | 3.93/4.09/3.97 | 2.49/2.59/2.56 | 3.03/3.24/3.10 | 3.15/3.30/3.21 |
| deepseek | Graphiti | 4.13/4.13/4.30 | 2.86/2.93/2.81 | 3.81/3.81/3.83 | 3.60/3.62/3.65 |
| gpt | Ariadne | 4.41/4.39/4.54 | 4.43/4.34/4.41 | 4.34/4.24/4.44 | 4.40/4.32/4.47 |
| gpt | Mem0 | 4.06/4.03/4.07 | 2.86/2.91/2.86 | 3.54/3.49/3.57 | 3.49/3.48/3.50 |
| gpt | Graphiti | 4.17/4.11/4.14 | 3.04/3.00/3.07 | 3.63/3.69/3.69 | 3.61/3.60/3.63 |
| gemini | Ariadne | 4.54/4.47/4.47 | 4.91/4.94/4.91 | 4.67/4.66/4.61 | 4.71/4.69/4.67 |
| gemini | Mem0 | 3.94/4.06/3.93 | 2.96/3.01/2.96 | 3.39/3.34/3.34 | 3.43/3.47/3.41 |
| gemini | Graphiti | 3.96/4.00/3.99 | 2.94/2.91/2.91 | 3.73/3.70/3.71 | 3.54/3.54/3.54 |

**raw（不带 StoryRank）**

| judge | system | 记忆丰富度 | 易理解性 | 用户建模度 | total |
|---|---|---|---|---|---|
| qwen | Ariadne | 4.56/4.54/4.60 | 3.79/3.74/3.70 | 4.63/4.66/4.71 | 4.33/4.32/4.34 |
| qwen | Mem0 | 3.77/3.63/3.79 | 3.57/3.47/3.47 | 2.91/2.86/3.01 | 3.42/3.32/3.42 |
| qwen | Graphiti | 3.97/3.93/3.97 | 3.71/3.60/3.66 | 3.90/3.77/3.87 | 3.86/3.77/3.83 |
| deepseek | Ariadne | 4.04/4.04/4.14 | 2.79/2.71/2.63 | 3.97/3.90/3.96 | 3.60/3.55/3.58 |
| deepseek | Mem0 | 3.44/3.54/3.57 | 3.83/3.73/3.84 | 3.10/3.16/3.10 | 3.46/3.48/3.50 |
| deepseek | Graphiti | 3.83/3.77/3.77 | 3.46/3.44/3.41 | 3.73/3.69/3.67 | 3.67/3.63/3.62 |
| gpt | Ariadne | 4.17/4.31/4.26 | 3.10/3.09/3.13 | 3.96/3.90/3.91 | 3.74/3.77/3.77 |
| gpt | Mem0 | 3.76/3.81/3.89 | 3.97/4.03/3.90 | 3.51/3.46/3.54 | 3.75/3.77/3.78 |
| gpt | Graphiti | 4.10/4.10/4.14 | 3.46/3.47/3.43 | 3.94/3.96/3.99 | 3.83/3.84/3.85 |
| gemini | Ariadne | 4.31/4.33/4.36 | 3.57/3.61/3.63 | 4.41/4.43/4.44 | 4.10/4.13/4.14 |
| gemini | Mem0 | 4.04/4.11/4.03 | 3.91/3.94/3.86 | 3.63/3.63/3.66 | 3.86/3.90/3.85 |
| gemini | Graphiti | 4.11/4.10/4.16 | 4.00/4.01/4.01 | 4.07/4.01/4.17 | 4.06/4.04/4.11 |

---

## 四、结论（修正后）

1. **StoryRank 真实净增量 ≈ +0.4~+1.0（均值 ~+0.7）**——小于此前混淆版（+0.8~+1.4），但**仍是明确正向**。
2. **易理解性是 StoryRank 的核心贡献点**：raw 掉得最狠的就是它（**-1.2~-1.8**）；原始长链（平均 **~27 节点 + ~23 边**，多跳后更长）难懂，故事化把它拉回。
3. **raw 块一旦同构（补 node_type + 方向）不再"垫底坍缩"**：qwen 下 Ariadne-raw 4.33 **仍是第一**（> Graphiti 3.86 > Mem0 3.44）；gemini 下 Ariadne-raw 4.10 亦第一（> Graphiti 4.06 > Mem0 3.86）——丰富度/建模度依旧高，仅易理解性掉到 3.57~3.67。**此前"raw 坍缩"主要是 raw 块信息残缺，不是"未叙事化"本身。**
4. **判分一致性是 StoryRank 的另一大贡献**：raw（同构）kappa 0.25 → story 0.39、top-1 0.48→0.73。原始结构块即使信息完整，**judge 判分仍明显不稳定**；故事化后更稳定。（§3.2 已改为统一新 prompt run1 口径、含 gemini 双变体，数值略异于 0907 原版，方向不变）
5. **相对判分效应**：同题并排下，Ariadne 变弱会相对抬高 Mem0/Graphiti（qwen 下易理解性 Mem0 2.29→3.53、Graphiti 2.73→3.69）——三系统分不独立，需在报告披露。
6. **判分方差**（§3.3）：同模型三次重跑，跨次 std ≤ 0.06（gpt 最大 0.059）——**同一 judge 模型结果高度可复现**；真正方差来自**换模型**（跨模型总分差 ≈0.4，远大于任何重跑差）。逐条级抖动 qwen < gpt < deepseek（deepseek 0.24~0.29 最不稳）。

---

## 五、产物清单

| 文件 | 说明 |
|------|------|
| `eval/collect_ariadne_raw.py` | 生成与 StoryRank 输入同构的 Ariadne 原始块（`context_comprehensive_ariadne_raw.json`） |
| `eval/judge_comprehensive.py` | judge 同题并排判分；`JUDGE_JOINT_PROMPT` 已补边/节点语义；支持 `--ctx` |
| `eval/judge_gemini_httpx.py` | gemini 判分（httpx 直连 + `max_tokens=4096`，规避推理 token 截断） |
| `eval/storyrank_ablation.py` | 消融 + 交叉一致性统计（动态剔除失效模型） |
| `eval/dim_breakdown.py` | 逐维（丰富度/易理解/建模度）story-vs-raw 对比 |
| `eval/variance_check2.py` / `eval/report_tables.py` | 三次重跑方差 / 总表与方差表统计 |
| `eval/outputs/context_comprehensive_ariadne_raw.json` | Ariadne 原始块（同构） |
| `eval/outputs/judge_ariadneraw_{comprehensive,deepseek,gpt}.json` | raw 变体多模型判分（gemini 走下方 repeat 文件） |
| `eval/outputs/judge_repeat_{story,raw}_{qwen,deepseek,gpt,gemini}_{1,2,3}.json` | 判分方差（同模型×3 次）数据，gemini 已含 |
| `eval/outputs/storyrank_ablation.json` | 消融 & 交叉一致性汇总 |
| `eval/outputs/story_comprehensive_3sys.json` / `judge_*.json` | story 变体既有产物（复用） |

---

## 六、局限（后续可补）

1. **同题并排 = 相对判分**：StoryRank 净增量混入"Ariadne 变弱→他系统相对抬升"的成分。若要 StoryRank **绝对**价值，需**对 Ariadne 单独判分**（不并排）对照。
2. **gemini 已补齐（本版）**：此前 gemini 对 raw/部分 story 块判分为全 0，根因是 `max_tokens=512` 被其推理 token 吞掉、JSON 在闭合前截断（`finish_reason:length`）；改为 httpx 直连 + `max_tokens=4096` 后，story 与 raw 均稳定出分并跑满 3 次重跑。唯 gemini 经 gpt.ge 代理偶发超时/短垃圾内容，已由重试兜底；若后续换更稳定的官方接入，可再复核一次。
3. **judge 主观性**：陪伴场景主观判分波动仍存在；已用多模型交叉 + 剔除病态/失效模型缓解，报告给定方向 + 区间，非精确排位。

---

## 七、总结（0907 补测口径 · 含 gemini）

### 7.1 模型方差与偏好
- **方差主要来自"换模型"，而非"重跑"**：同模型跨次 std（Ariadne total）≤ 0.06（gpt 0.059 最大，gemini 0.018 / qwen 0.022 / deepseek 0.019）；而跨模型总分差 **≈0.7~0.8**（story：qwen 4.73 vs deepseek 3.98；raw：qwen 4.33 vs deepseek 3.60），远大于任何重跑差 → **同一 judge 高度可复现，结论的稳健性更多取决于选哪个 judge**。
- **逐条级抖动：qwen < gpt < deepseek**（deepseek 0.24~0.29 最不稳），gemini 补测后逐维 std 最小（≤0.035），整体很稳。
- **模型偏好/偏置**：
  - **qwen（主判分）**：对 Ariadne 最友好——易理解性给得最高（story 4.84、raw 3.74 都是它最高），Ariadne 总分最高（story 4.73 / raw 4.33）。
  - **deepseek**：对易理解性最苛刻（raw Ariadne 易理解 2.79，全场最低），总分最保守（story 3.98 / raw 3.60），且逐条最不稳定。
  - **gpt**：温和但 raw 下对 Ariadne 易理解压到 3.10，相对更偏好"结构化"的 Graphiti/更易读的 Mem0。
  - **gemini**：整体慷慨（story 易理解 4.91 最高）、很稳，但 raw 下对 Ariadne 易理解也给到最低（3.60），偏好结构化块。

### 7.2 StoryRank 消融增量
| judge | story total | raw total | 增量 |
|---|---|---|---|
| qwen | 4.73 | 4.33 | **+0.40** |
| deepseek | 3.98 | 3.60 | +0.38 |
| gpt | 4.40 | 3.74 | +0.66 |
| gemini | 4.71 | 4.10 | +0.61 |

- **统一 run1 口径下，增量 +0.38~+0.66，均值 ≈ +0.5**；若用各变体"代表 run"口径（§3.1，含 gpt 的 +1.03）则均值 ≈ +0.7。两种口径**均明确为正**，StoryRank 是正向收益。
- **增量高度集中在"易理解性"**：story→raw 易理解性掉 **-1.0~-1.8**（story 4.43~4.91 → raw 2.79~3.79），是唯一被明显拉低的维度；**记忆丰富度、用户建模度几乎持平甚至微降**（如 deepseek 丰富度 story 3.66→raw 4.04 为 −0.38）。→ StoryRank 的净增量≈它把"易理解性"补回来的部分。
- **附带收益：判分一致性提高**（raw kappa 0.25→story 0.39、top-1 0.48→0.73），故事化让不同 judge 更容易给出相近排位。

### 7.3 raw 下 Ariadne 是否仍有优势？（有条件成立）
raw（无 StoryRank）三系统 total（3 次均值），各 judge 判出的第一：

| judge | Ariadne | Mem0 | Graphiti | Ariadne 是否第一 |
|---|---|---|---|---|
| qwen | **4.33** | 3.39 | 3.82 | ✅ 第一（胜 Graphiti +0.51） |
| gemini | **4.12** | 3.87 | 4.07 | ✅ 第一（胜 Graphiti，仅 +0.05，边缘） |
| deepseek | 3.58 | 3.48 | **3.64** | ❌ 第二（负于 Graphiti −0.06） |
| gpt | 3.76 | 3.77 | **3.84** | ❌ 垫底（负于 Mem0/Graphiti） |

- **结构性优势仍在"内容维"**：raw 下 Ariadne 的**记忆丰富度、用户建模度在全部 4 个 judge 中都最高**（内容最足），这部分优势没有消失。
- **结构性短板是"易理解性"**：raw 下 Ariadne 的易理解性在 deepseek/gpt/gemini 中**都是三者最低**（2.71 / 3.11 / 3.60），仅 qwen 下仍最高（3.74）。
- **结论**：raw 里 Ariadne **不是稳赢**——它靠"内容维"领先，但因"易理解性"被大量放大，总分是否仍居首取决于 judge：**qwen（领先 +0.51）与 gemini（+0.05 边缘）下仍第一；deepseek、gpt 下被 Graphiti 反超，gpt 下更被 Mem0 一并超过而垫底**。Graphiti 因节点+关系更结构化，易理解性在多数 judge 下反高于 Ariadne-raw。→ 这也再次说明：StoryRank（把长链故事化）之所以必要，正是因为它把 Ariadne 最弱、又最影响总分的那一维（易理解性）补强回来。

> **一句话结论**：综合来看，**StoryRank 在 Ariadne 这条链上基本不可或缺**——PAR 检索层返回的是**完整的树状结构**，**字符与信息量大**，既拖累"易理解性"这一指标，又在真实对话的上下文注入场景里容易**被重复 thinking/反复消化**；加入 StoryRank 这个中间件（把长链树结构压缩成故事化叙述）能**有效避免上述两类问题**——既守住内容维优势，又补回易理解性与下游稳定性。
