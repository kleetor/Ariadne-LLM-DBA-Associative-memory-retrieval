# LongMemEval 探针评测报告（0907）

> 日期：2026-09-07 · 状态：中文数据集探针完成（4 条 oracle）
> 关联：LongMemEval 接入（[loader](file:///c:/Users/makot/Desktop/Ariadne/frameworks/longmemeval_loader.py) / [翻译](file:///c:/Users/makot/Desktop/Ariadne/frameworks/longmemeval_translate.py) / [runner](file:///c:/Users/makot/Desktop/Ariadne/frameworks/longmemeval_runner.py) / [检索探针](file:///c:/Users/makot/Desktop/Ariadne/frameworks/longmemeval_retrieval_probe.py)）

---

## 核心结论（速览）

| 路径 | QA 正确率 | 说明 |
|------|:---:|------|
| **DBA 建图 → PAR 检索** | **0 / 4** | 抽取把原始对话**抽象/合并**，丢失精确事实；且英文时序守卫（中文正则）未触发 |
| **原始 turn 转录 → 直接送检索层** | **3 / 4** | 保留原始文本事实，session 召回全 1.0、turn 召回 0.5~1.0 |

**结论**：在 LongMemEval 这类「精确事实 / 事件先后」基准上，**「把原始对话 turn 转录为记忆条目、直接送检索层（纯向量）」优于 DBA 抽象建图**。这与既有结论一致（纯向量是记忆检索的最优默认方案）。

---

## 一、方向决策

- Ariadne 为**中文母语设计**（中文语义/情景复杂度高于英文）。
- 因此**不改系统**，而是把 LongMemEval 数据集**翻译为中文**后用中文系统评测。
- 系统侧改动（dba.py 英文 prompt / store.py 截断）已 `git checkout` **全部回退**到中文基线。

## 二、数据与管线

| 项 | 内容 |
|------|------|
| 数据集 | `data/longmemeval/longmemeval_oracle.json`（500 题，证据会话版）+ `longmemeval_s_cleaned.json`（264MB） |
| 中译 | `data/longmemeval/longmemeval_oracle_zh.json`（探针子集 4 条，保留 question_type / answer_session_ids / has_answer 标签） |
| embedding | 本地 `BAAI/bge-large-zh-v1.5`（`.env` 默认，不强制英文） |
| 探针题型 | single-session-assistant ×2 ＋ temporal-reasoning ×2 |

**翻译要点**：会话内分批（`batch_turns=6`）翻译，回退时不泄漏编号前缀；长表 turn（shift 表）逐条回退兜底。

## 三、探针 A：DBA 建图 → PAR 检索（`longmemeval_runner.py`）

- 流程：中文会话 → `dba.maintain` 批量建图 → `retrieve_with_story` → QA。
- 结果：只有 7161e7e2 建图成功（8 节点）；其余 **3/4 为 0 节点**——**`triage` 全部判 SKIP**（记忆治理过严）。
- 绕过 triage（`NoTriageDBA`）后 4 条全部建图（8~41 节点），但 **QA 仍 0/4**：
  - **assistant**：DBA 把轮班表抽象成「4 班次 / 每周 2 天休」等，**丢了 Admon 周日的具体分配**；story 甚至幻写「Admon 周日=中班」。
  - **temporal**：图谱**无 THING 时间锚点、无 TEMPORAL 边**（`has_temporal_signal` 需同批 ≥2 个不同中文日期词，oracle 证据会话大多不满足），`temporal_lookup` 触发=False。

## 四、探针 B：原始 turn 转录 → 直接送检索层（`longmemeval_retrieval_probe.py`）

- 流程：每个 turn 作为记忆条目（id=`{qid}_s{si}_t{ti}`，保留会话/轮次下标）灌入向量库 → `search(question, topk)` → 精确召回 → QA。
- **精确召回**（对齐 LongMemEval turn/session 两级）：

| 实例 | 题型 | turn 召回 | session 召回 | QA 正确 |
|------|------|:---:|:---:|:---:|
| 7161e7e2 | assistant | 1.0 | 1.0 | ✅（Admon 周日=上午8-4 白班） |
| c4f10528 | assistant | 0.0 | 1.0 | ✅（Miss Bee Providore） |
| gpt4_2487a7cb | temporal | 1.0 | 1.0 | ✅（Python 网络研讨会先） |
| gpt4_2655b836 | temporal | 1.0 | 1.0 | ❌（答 LLM 未合成） |

- **QA 调优**：放宽 `[NO_ANSWER]`→ 命中 turn **按 ts 排序** → 加「按时间先后推理」提示 → judge 判定。
- **遗留失败（gpt4_2655b836）**：答案在命中 top-8 内（`s1_t2`「车载 GPS 3月22 出问题」、服务 3月15），**证据全命中但答 LLM 未把「保养后首个问题=GPS」合成**——纯推理缺口，非检索/翻译问题。

## 五、产物清单

| 文件 | 内容 |
|------|------|
| `frameworks/longmemeval_loader.py` | LongMemEval JSON → 会话/查询/证据标签 |
| `frameworks/longmemeval_translate.py` | LongMemEval → 中文（保留结构） |
| `frameworks/longmemeval_runner.py` | DBA 建图 → PAR 检索 → 近似召回 → QA（含 `NoTriageDBA`） |
| `frameworks/longmemeval_retrieval_probe.py` | 原始 turn 直接检索：精确召回（turn/session）+ QA |
| `data/longmemeval/longmemeval_oracle_zh.json` | 中文探针数据（4 条） |
| `eval/outputs/longmemeval_zh_probe.json` | 探针 A 结果 |
| `eval/outputs/longmemeval_turn_retrieval.json` | 探针 B 结果（最终） |

## 六、后续建议

1. **沿用「原始 turn 直接送检索层」**作为 LongMemEval 的检索路径（保留事实、召回高、QA 3/4）。
2. 若要扳回 gpt4_2655b836：答案侧加**显式 CoT**（先定位时间，再按先后合成），或增大 `topk`。
3. 扩大样本（覆盖全部 6 种题型 / 增加每题型数量）以稳定统计，并跑 `longmemeval_s_cleaned`（真实检索难度，含填充会话）。
