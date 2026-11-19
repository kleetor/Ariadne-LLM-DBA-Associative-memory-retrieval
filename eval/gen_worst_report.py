# -*- coding: utf-8 -*-
"""P 最差四条查询（q17/q27/q18/q19）详细信息文档生成器"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

import yaml

scores = {r["query_id"]: r for r in json.load(open("eval/outputs/story_scores_full2.json", encoding="utf-8"))["results"]}
p_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/p_candidates.json", encoding="utf-8"))["results"]}
rag_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/naive_rag_candidates.json", encoding="utf-8"))["results"]}
stories = json.load(open("eval/outputs/story_texts.json", encoding="utf-8"))
qs = {q["id"]: q for q in yaml.safe_load(open("data/LiFEMem_queries.yaml", encoding="utf-8"))["queries"]}
nodes = {n["id"]: n["content"] for n in yaml.safe_load(open("data/LiFEMem.yaml", encoding="utf-8"))["nodes"]}

WORST = ["q17", "q27", "q18", "q19"]
dims = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter", "total"]

L = []
A = L.append
A("# P 最差四条查询专项分析（q17 / q27 / q18 / q19）")
A("")
A("> P 链路故事质量（total）最低的 4 条非弃权查询。三条为 knowledge_update 类。")
A("")

for qid in WORST:
    row = scores[qid]
    q = qs[qid]
    expected = q.get("expected", [])
    A(f"## {qid} [{q['category']}] {q['text']}")
    A("")
    A(f"- **目的推断**：{row['purpose']}")
    A(f"- **expected（命题所需节点）**：{expected}")
    for e in expected:
        A(f"  - `{e}`：{nodes.get(e, '?')}")

    p = p_cand[qid]
    r = rag_cand[qid]
    out_ids = [c["id"] for c in p["candidates"]]
    hit = set(out_ids) & set(expected)
    A("")
    A("### P 候选完整列表（全部 {} 条）".format(p["total_candidates"]))
    A("")
    A("| 排名 | id | content | score | 命中expected |")
    A("|:---:|---|---|:---:|:---:|")
    for i, c in enumerate(p["candidates"], 1):
        mark = "**命中**" if c["id"] in expected else ""
        A(f"| {i} | `{c['id']}` | {c['content']} | {c['score']:.3f} | {mark} |")

    sc = row["systems"]["p"]["scores"]
    cv = row["systems"]["p"].get("cross_valid", {})
    A("")
    A(f"### P 五维评分与联想合理性")
    A("")
    A("| relevance | coherence | completeness | faithfulness | noise_filter | total |")
    A("|:---:|:---:|:---:|:---:|:---:|:---:|")
    A(f"| {sc['relevance']} | {sc['coherence']} | {sc['completeness']} | {sc['faithfulness']} | {sc['noise_filter']} | **{sc['total']}** |")
    A(f"- CROSS_VALID：reasonable={cv.get('reasonable')} / noise={cv.get('noise')} / 合理率={cv.get('cross_valid')}")
    A(f"- 命题覆盖：命中 {len(hit)}/{len(expected)}（{sorted(hit)}）")

    st = stories[qid]["stories"]["P"]["story"]
    A("")
    A(f"### P 故事全文（{len(st)} 字）")
    A("")
    A(f"> {st}")

    rsc = row["systems"]["naive_rag"]["scores"]
    rcv = row["systems"]["naive_rag"].get("cross_valid", {})
    rst = stories[qid]["stories"]["Naive RAG"]["story"]
    A("")
    A(f"### Naive RAG 对照（total={rsc['total']}，CROSS_VALID={rcv.get('cross_valid')}）")
    A("")
    A("**RAG 候选 top8**：")
    for c in r["candidates"][:8]:
        mark = "（**命中**）" if c["id"] in expected else ""
        A(f"- `{c['id']}` {c['content']}{mark}")
    A("")
    A(f"**RAG 故事（{len(rst)} 字）**：")
    A(f"> {rst}")
    A("")

out = ROOT / "Plan/0818——P最差查询专项分析.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"[ok] {out}（{len(L)} 行）")
