# -*- coding: utf-8 -*-
"""查看 Graphiti 代表查询的候选质量"""
import json

data = json.load(open("eval/outputs/graphiti_candidates.json", encoding="utf-8"))
by = {r["query_id"]: r for r in data["results"]}
for qid in ["s17", "s23", "s14"]:
    r = by[qid]
    print(f"===== {qid}: {r['query']} =====")
    for c in r["candidates"][:6]:
        print(f"  [{c['id']}] {c['content']}")
        print(f"      fact: {c.get('fact', '')}")
    print()
