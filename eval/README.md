# 评测复现说明

本目录是论文 **Purpose-Driven Associative Retrieval: Building a User-Modeling Context from a
Memory Graph**（v0.5，2026-11-19）的评测代码与归档产物。论文把评测对象定义为“系统喂给下游
模型的结构化记忆上下文块”，主指标是三个 LLM judge 的同题并排 1–5 分，检索侧指标只作机制诊断。

> 已归档：预印本 DOI [10.5281/zenodo.24372785](https://doi.org/10.5281/zenodo.24372785)；
> 本仓库 v0.5 快照 DOI [10.5281/zenodo.24372787](https://doi.org/10.5281/zenodo.24372787)。

## 1. 环境

- Python >= 3.10（见仓库根目录 `pyproject.toml`）
- 核心包安装：`pip install -c constraints.txt -e .`
- 评测额外依赖：`pip install -r eval/requirements.txt`
- 环境变量：复制仓库根目录 `.env.example` 为 `.env`，至少填 `OPENAI_*`（检索侧 LLM）、
  `EMBEDDING_*`（默认 `BAAI/bge-m3`）、`JUDGE_*`（判分模型）。
- 所有命令在**仓库根目录**执行（脚本用 `python -m eval.<name>`，依赖 `dba_pipeline`）。

模型与版本（论文正文一致）：

| 角色 | 模型 |
|---|---|
| 主 judge | `qwen3.8-flash` |
| 交叉 judge | `deepseek-v4-flash`、`gpt-5.6-luna` |
| 默认向量 | `BAAI/bge-m3` |
| 换向量对照（E15c / zh15） | `BAAI/bge-large-zh-v1.5` |

## 2. 数据（`data/dba_eval/`）

| 文件 | 内容 |
|---|---|
| `comprehensive_gold.yaml` | 主图：163 节点 / 237 边，人工构建 |
| `comprehensive_dialogs.yaml` | 26 批对话，覆盖 7 个场景 |
| `comprehensive_queries.yaml` | 76 条第一人称查询：`category`（9 类）、`purposes`、`gold_expected` |
| `intent_queries.yaml` | E5 / E11 的 10 情景 × 3 意图 |
| `e10_L.yaml` / `e10_M.yaml` / `e10_H.yaml` | E10 三规模图（163 / 253 / 343 节点） |
| `e12a_del30.yaml` / `e12a_rewire30.yaml` | E12a 结构扰动图（seed 20261119） |
| `e12b_auto.yaml` | E12b 由 DBA 管线自动构建的同数据图 |
| `abstain_probe_queries.yaml` | 弃权阈值探测 |

`gold_expected` 是“语义期望”，不是严格对齐的 golden set；论文 §5.2 说明了为什么不用
P@K / R@all 作主指标。

## 3. 实验与命令

通用三步：**组装上下文 → judge 判分 → 统计汇总**。

```bash
# 判分（--ctx 指向上下文块 JSON；--model 覆盖 judge 模型做交叉验证）
python -m eval.judge_comprehensive --ctx <ctx.json> --out <judge.json>
# 统计（bootstrap CI + Wilcoxon + κ/ICC/Spearman）
python -m eval.stats_report --judge <judge1.json> <judge2.json> ... --out <stats.json>
```

| 实验 | 命令 | 归档产物 |
|---|---|---|
| E1 系统级（Ariadne / Mem0 / Graphiti） | 三系统上下文块分别由 `run_p_baseline.py` + `collect_ariadne_raw.py`（Ariadne）、`compare_mem0.py`、`compare_graphiti.py` 产出后合并；再跑 `judge_comprehensive.py` | `outputs/e1/` |
| E2 judge 可靠性 | `python -m eval.cross_model_consistency`；或 `stats_report.py` 直接汇总多 judge | `outputs/e1/stats_e1.json`、`outputs/e3/stats_e3.json` |
| E3 同图隔离（PAR+story / PAR+raw / 向量） | `python -m eval.e3_par_vs_vector`（默认三臂、`--anon-order shuffle`） | `outputs/e3/context_e3_bandfix.json`（主表批次）、`judge_e3_bandfix_*.json`、`stats_e3.json` |
| E4 机制消融 | `python -m eval.e4_ablation`（purpose / structure 两批，`full` 为共用锚点） | `outputs/e4/` |
| E5 意图条件 | `python -m eval.e5_intent`；判分 `python -m eval.judge_e5_intent`；阈值扫描 `python -m eval.e5_tau_sweep` | `outputs/e5/` |
| E7 输出稳定性 | `python -m eval.e7_stability --reps 3` | `outputs/e7/` |
| E9 失败类型 | `python -m eval.e9_failure_taxonomy`（LLM 辅助编码，非人工双标） | `outputs/e9/failure_taxonomy.json` |
| E10 分化度与阈值 | `python -m eval.e10_build_graphs` → `python -m eval.e10_probe` | `outputs/e10/` |
| E11 结构化目的 | `python -m eval.e11_purpose_frame`；过滤对照 `python -m eval.e11_filter_ab` | `outputs/e11/purpose_frame*.json`、`filter_ab_*.json` |
| E12a 结构扰动 | `python -m eval.e12a_perturb` → 在两张扰动图上重跑 E3 + judge → `python -m eval.e12a_report --del ... --rewire ...` | `outputs/e12a/stats.json`、`manifest.json` |
| E12b 自动图 | `python -m eval.e3_par_vs_vector --yaml data/dba_eval/e12b_auto.yaml --out eval/outputs/e12b/context_auto.json` → judge → stats | `outputs/e12b/stats_e3_auto.json` |
| E13 pAUC / 分位阈值 | `python -m eval.e13_quantile`（加 `--cache-suffix zh15` 换向量） | `outputs/e13/quantile*.json` |
| E14 AUC 置信区间 | `python -m eval.e14_auc_ci`（pair bootstrap + node bootstrap） | `outputs/e14/auc_ci*.json` |
| E15c 换 embedding | `python -m eval.e3_par_vs_vector --vector-cache-key comprehensive_zh15 --out eval/outputs/e15c/context_e3_zh15.json` → judge → stats | `outputs/e15c/stats_e3_zh15.json` |
| 0828 附录 D | `python -m eval.e0828_stress_rall` | `outputs/audit/0828_stress_rall.json` |
| 论文图 1–3 | `python -m eval.make_figures` | `docs/paper/figs/` |

## 4. 归档产物与可重跑范围

本仓库只跟踪**统计层**产物（`stats_*.json`、`score_*.json`、`quantile*.json`、`auc_ci*.json`、
`failure_taxonomy.json`、`filter_ab_*.json`）。完整上下文块（`context_*.json`，单份 300–600 KB）
和向量缓存（`eval/.cache/`）不入库：前者可由上述命令重跑生成，后者随 embedding 模型与机器变化。

`outputs/e1`、`outputs/e3` 里的原始 judge 输出保留了逐题打分与 `label_map`（匿名映射），可用于
复核匿名化协议（逐题固定种子打乱，judge 只看“系统A/B/C”）。

## 5. 复现时的口径注意

- **默认目的过滤阈值 0.30 在主图上不删任何候选**（0/163），等价于未开启；有效工作点在
  0.45–0.55，且以输出规模换精度。E4/E5 的负面结果来自目的种子与加权，不来自过滤。
- **E3 主表批次是 `bandfix`**，必须与同批基线比较；不要与更早存档批次的绝对值混比
  （同批 C−D = 1.633，存档批次为 1.296，差异来自代码版本）。
- **E12a 的所有比较都用同批原图基线**，不与存档数字混比。
- **E12b 自动图的绝对分不能与人工图跨图比较**，只用于确认机制排序方向不变。
- **E13 分位阈值只对齐跨主体混入率，不对齐输出预算**；pAUC 的局部可分性是零和的。
- **E9 的 92.1% 出自严格服务标准**（要求覆盖全部期望侧面）；放宽标准后为 0%，两种口径都
  在附录中报告。
- 自动图的抽取层数字（节点召回 81.6%、边精确率 10.6%）只用于图来源替换，不作为检索结果。

## 6. 许可

- 代码：GNU AGPL-3.0（见仓库根目录 `LICENSE`）。
- 数据集与评测 prompt/rubric（`data/dba_eval/`、`eval/judge_*.py` 内的提示词）：CC BY 4.0。
- 如引用，请使用 `CITATION.cff` 或论文条目。
