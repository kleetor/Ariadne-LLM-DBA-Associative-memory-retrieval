# -*- coding: utf-8 -*-
"""查看 P vs Graphiti 纯选择的理由"""
import json

d = json.load(open("eval/outputs/story_pair_pg.json", encoding="utf-8"))
print("===== Graphiti 获胜的查询 =====")
for r in d["results"]:
    if r["winner"] == "graphiti":
        print(f"  [{r['query_id']}] {r['reason']}")
print("\n===== 前 4 条样本（含 P 获胜） =====")
for r in d["results"][:4]:
    print(f"  [{r['query_id']}] winner={r['winner']}: {r['reason']}")
