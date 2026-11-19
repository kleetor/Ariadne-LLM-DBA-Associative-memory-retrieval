# -*- coding: utf-8 -*-
"""P 表现最差查询专项分析：故事 + 候选节点 + 目的推断"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

scores = {r["query_id"]: r for r in json.load(open("eval/outputs/story_scores_full2.json", encoding="utf-8"))["results"]}
p_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/p_candidates.json", encoding="utf-8"))["results"]}
rag_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/naive_rag_candidates.json", encoding="utf-8"))["results"]}
stories = json.load(open("eval/outputs/story_texts.json", encoding="utf-8"))
qs = {q["id"]: q for q in yaml.safe_load(open("data/LiFEMem_queries.yaml", encoding="utf-8"))["queries"]}
nodes = {n["id"]: n["content"] for n in yaml.safe_load(open("data/LiFEMem.yaml", encoding="utf-8"))["nodes"]}

# P total 最低的非弃权查询
ranked = sorted(
    [(qid, row["systems"]["p"]["scores"]["total"]) for qid, row in scores.items()
     if row["category"] != "abstention" and "scores" in row["systems"].get("p", {})],
    key=lambda x: x[1],
)
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print("P total 排名（最低）：")
for qid, t in ranked[:TOP_N + 3]:
    print(f"  {qid}: {t}")
print()

dims = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter", "total"]
for qid, ptot in ranked[:TOP_N]:
    row = scores[qid]
    q = qs[qid]
    expected = q.get("expected", [])
    print("=" * 95)
    print(f"[{qid}] ({q['category']}) {q['text']}")
    print(f"P 目的推断: {row['purpose']}")
    print(f"expected: {expected}")
    if expected:
        for e in expected:
            print(f"    {e}: {nodes.get(e, '?')}")

    p = p_cand[qid]
    out_ids = [c["id"] for c in p["candidates"]]
    print(f"\n--- P 候选（{p['total_candidates']}条，top10）---")
    for c in p["candidates"][:10]:
        mark = "  <<<expected" if c["id"] in expected else ""
        print(f"    {c['id']:<6} {c['content'][:46]}{mark}")

    st = stories[qid]["stories"]["P"]["story"]
    sc = row["systems"]["p"]["scores"]
    cv = row["systems"]["p"].get("cross_valid", {})
    print(f"\n--- P 故事（{len(st)}字）---")
    print(f"  五维: " + " ".join(f"{d}={sc[d]}" for d in dims))
    print(f"  cross_valid: reasonable={cv.get('reasonable')} noise={cv.get('noise')} rate={cv.get('cross_valid')}")
    print(f"> {st}")

    r = rag_cand[qid]
    rsc = row["systems"]["naive_rag"]["scores"]
    rcv = row["systems"]["naive_rag"].get("cross_valid", {})
    print(f"\n--- Naive RAG 对照：total={rsc['total']} cross_valid={rcv.get('cross_valid')} 候选top5---")
    for c in r["candidates"][:5]:
        mark = "  <<<expected" if c["id"] in expected else ""
        print(f"    {c['id']:<6} {c['content'][:46]}{mark}")
    print()
