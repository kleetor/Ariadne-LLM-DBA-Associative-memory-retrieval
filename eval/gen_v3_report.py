# -*- coding: utf-8 -*-
"""v3 汇总报告生成器（陈述触发式评测）"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml

scores = json.load(open("eval/outputs/story_scores_v3.json", encoding="utf-8"))["results"]
p_cand = json.load(open("eval/outputs/p_candidates_v3.json", encoding="utf-8"))["results"]
rag_cand = json.load(open("eval/outputs/naive_rag_candidates_v3.json", encoding="utf-8"))["results"]
qs = {q["id"]: q for q in yaml.safe_load(open("data/LiFEMem_queries_v3.yaml", encoding="utf-8"))["queries"]}

DIMS = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter", "total"]
SYS = ["p", "naive_rag"]
DISPLAY = {"p": "P（联想检索）", "naive_rag": "Naive RAG"}

L = []
A = L.append


def sys_total(row, s):
    return row["systems"][s]["scores"]["total"] if s in row["systems"] else None


def cv(row, s):
    v = row["systems"][s]
    if "cross_valid" not in v or not v["cross_valid"]:
        return None
    r, n = v["cross_valid"].get("reasonable", 0), v["cross_valid"].get("noise", 0)
    return (r + n) if (r + n) else 0


A("# LiFEMem v3 评测报告（陈述触发式 · 对话记忆范式）")
A("")
A("> 查询形态：用户第一人称陈述（无提问）；35 条 = 30 陈述 + 5 弃权")
A("> 数据：LiFEMem（322 节点/410 边）｜judge：DeepSeek 盲评（五维 + CROSS_VALID）")
A("")

# 1. 总体
A("## 一、总体对比")
A("")
rows = [r for r in scores if r["category"] != "abstention"]
A("| 系统 | relevance | coherence | completeness | faithfulness | noise_filter | total | CROSS_VALID |")
A("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for s in SYS:
    vals = [round(sum(r["systems"][s]["scores"][d] for r in rows) / len(rows), 2) for d in DIMS]
    cvs = [x for r in rows if (x := cv(r, s)) is not None]
    cva = round(sum(cvs) / len(cvs), 3) if cvs else None
    A(f"| **{DISPLAY[s]}** | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | **{vals[5]}** | {cva} |")

# 2. 分情境
A("")
A("## 二、分情境对比（total / CROSS_VALID）")
A("")
cats = ["work_pressure", "health", "emotion", "nostalgia", "relationship", "life", "learning", "social", "future"]
cat_names = {"work_pressure": "工作压力", "health": "健康", "emotion": "情绪", "nostalgia": "怀旧",
             "relationship": "感情", "life": "生活", "learning": "学习", "social": "社交", "future": "未来"}
A("| 情境 | P total | RAG total | P CV | RAG CV |")
A("|:---|:---:|:---:|:---:|:---:|")
for c in cats:
    sub = [r for r in rows if r["category"] == c]
    if not sub:
        continue
    p_t = round(sum(sys_total(r, "p") for r in sub) / len(sub), 2)
    r_t = round(sum(sys_total(r, "naive_rag") for r in sub) / len(sub), 2)
    p_c = round(sum(cv(r, "p") for r in sub) / len(sub), 3)
    r_c = round(sum(cv(r, "naive_rag") for r in sub) / len(sub), 3)
    A(f"| {cat_names[c]}（{len(sub)}） | {p_t} | {r_t} | {p_c} | {r_c} |")

# 3. expected 覆盖（R@all）
A("")
A("## 三、expected 覆盖（联想召回，仅非弃权且无 no_gold）")
A("")
A("| 查询 | 情境 | expected | P 命中 | RAG 命中 |")
A("|:---|:---|:---|:---:|:---:|")
p_cov, r_cov = [], []
for r in scores:
    q = qs[r["query_id"]]
    if r["category"] == "abstention" or q.get("no_gold"):
        continue
    exp = set(q.get("expected", []))
    if not exp:
        continue
    p_ids = {c["id"] for c in next(x for x in p_cand if x["query_id"] == r["query_id"])["candidates"]}
    r_ids = {c["id"] for c in next(x for x in rag_cand if x["query_id"] == r["query_id"])["candidates"]}
    ph = len(exp & p_ids)
    rh = len(exp & r_ids)
    p_cov.append(ph / len(exp))
    r_cov.append(rh / len(exp))
    mark = " P覆盖优于RAG" if ph > rh else (" RAG覆盖优于P" if rh > ph else "")
    A(f"| {r['query_id']} | {r['category']} | {len(exp)} | {ph}/{len(exp)} | {rh}/{len(exp)} |{mark}")
A(f"\n平均覆盖：P={round(sum(p_cov)/len(p_cov),3)} vs RAG={round(sum(r_cov)/len(r_cov),3)}")

# 4. 弃权
A("")
A("## 四、弃权类（陈述隐含图中不存在的信息）")
A("")
A("| 查询 | 陈述 | P 拒答 | RAG 拒答 |")
A("|:---|:---|:---:|:---:|")
abs_rows = [r for r in scores if r["category"] == "abstention"]
p_abs = r_abs = 0
for r in abs_rows:
    pa = r["systems"]["p"]["abstention"]["refused"] if "abstention" in r["systems"].get("p", {}) else None
    ra = r["systems"]["naive_rag"]["abstention"]["refused"] if "abstention" in r["systems"].get("naive_rag", {}) else None
    p_abs += 1 if pa else 0
    r_abs += 1 if ra else 0
    A(f"| {r['query_id']} | {r['query'][:30]}… | {pa} | {ra} |")
A(f"\n正确拒答：P={p_abs}/{len(abs_rows)}，RAG={r_abs}/{len(abs_rows)}")

# 5. 目的推断抽查
A("")
A("## 五、目的推断（陈述→隐含目的，前 10 条抽查）")
A("")
A("| 查询 | 陈述（截断） | 目的推断 |")
A("|:---|:---|:---|")
for r in scores[:10]:
    A(f"| {r['query_id']} | {r['query'][:22]}… | {r['purpose']} |")

out = ROOT / "Plan/0818——LiFEMem_v3报告.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"[ok] {out}（{len(L)} 行）")
