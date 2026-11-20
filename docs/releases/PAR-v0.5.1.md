# PAR v0.5.1 发布与归档记录

论文：Purpose-Driven Associative Retrieval: Building a User-Modeling Context from a Memory Graph
归档日期：2026-11-20（对应版本日期 2026-11-19）

## 1. 发布位置

| 渠道 | 标识 | 备注 |
|---|---|---|
| Zenodo（预印本） | `10.5281/zenodo.24373015` | concept DOI `10.5281/zenodo.24372784`，v0.5.1 为当前版本 |
| Zenodo（artifact） | `10.5281/zenodo.24373025` | concept DOI `10.5281/zenodo.24372786` |
| ChinaXiv | 待审核 | 以英文全文提交，语种英文；已声明 Zenodo 预印本 DOI |

## 2. 归档文件（`docs/paper/`）

| 文件 | 大小 | SHA256 前 16 位 | 用途 |
|---|---|---|---|
| `PAR-preprint-v0.5.1.pdf` | 403.3 KB | `01B32B8950D8B77B` | Zenodo 线上预印本 |
| `PAR-preprint-chinaxiv.pdf` | 403.3 KB | `FBC108989261792C` | ChinaXiv 提交版 |
| `PAR-overleaf.zip` | 93.5 KB | `A5C5EE7202E2541B` | 源码包（`main.tex` + `chinaxiv.tex` + `refs.bib` + 3 张图） |
| `PAR-data-v0.5.1.zip` | 82.5 KB | `CD6CDEB2BC344B77` | 仅数据集 |
| `PAR-results-v0.5.1.zip` | 181.8 KB | `B4D25940E6377E26` | 仅统计与判分产物 |
| `PAR-supplement-v0.5.1.zip` | 264.5 KB | `300A4212BCBB859A` | 数据集 + 结果二合一（ChinaXiv 推荐附件） |

源码：`docs/paper/latex/main.tex`（`CEEC98A978C7E9A1`）、`chinaxiv.tex`（`30991FF15A700DD2`）、
`refs.bib`（`D4968DDBAE5FFCD1`）。

> 注意：线上 v0.5.1 的 PDF 使用旧署名块（QQ 邮箱、"Independent Researcher"）。此后源文件已改为
> 机构署名、通讯作者脚注与 ChinaXiv 图宽，因此**当前源码构建出的 PDF 与线上 v0.5.1 不同**；
> 这些改动计划在 v0.5.2 统一发布。

## 3. 构建方式

```bash
# 默认（Zenodo / arXiv）
pdflatex main && bibtex main && pdflatex main && pdflatex main

# ChinaXiv（前置部分不同：五段式摘要、5 个关键词、作者贡献、图宽 120mm）
pdflatex chinaxiv && bibtex chinaxiv && pdflatex chinaxiv && pdflatex chinaxiv
```

## 4. ChinaXiv 提交信息

- 语种：英文；题名：目的驱动联想检索：从记忆图构建用户建模上下文
- 作者：陈思宇 / Siyu Chen；单位：重庆电力高等专科学校 / Chongqing Electric Power College
- 分类：计算机科学技术 → 人工智能
- 关键词（中）：联想记忆；Agent 长期记忆；记忆图；LLM 评测；用户建模
- 关键词（英）：Associative memory; Agent long-term memory; Memory graph; LLM evaluation; User modeling
- 许可：CC BY 4.0；版权承诺：作者提交、未向出版社投稿；投稿状态：未投稿
- 附件：`PAR-preprint-chinaxiv.pdf`、`PAR-supplement-v0.5.1.zip`、许可承诺书、身份识别材料

## 5. 复现入口

- 评测说明与逐实验命令：`eval/README.md`
- 数据：`data/dba_eval/`（归档时已纳入版本控制）
- 统计与判分产物：`eval/outputs/`（仅统计层，完整上下文与向量缓存重跑生成）

## 6. 变更记录

- **v0.5**：补 E14（AUC bootstrap CI）、E13（pAUC / 分位阈值）、E12a（结构扰动）、E12b（自动图）、
  E15c（换 embedding）；修正 0.30 过滤的归因、7–9 pp 混入差距、E9 的 92.1% 口径。
- **v0.5.1**：首页补 Zenodo DOI；标题/摘要口径不变。
- **待发布（v0.5.2）**：机构署名、通讯作者邮箱脚注、图宽、ChinaXiv 前置格式；arXiv 待 endorsement。
