# -*- coding: utf-8 -*-
"""查看 P vs Graphiti 原始事实 纯选理由"""
import json

d = json.load(open("eval/outputs/story_pair_p_facts.json", encoding="utf-8"))
print("===== 汇总 =====")
print(d["summary"])
print("\n===== graphiti_facts 获胜查询 =====")
for r in d["results"]:
    if r["winner"] == "graphiti_facts":
        print(f"  [{r['query_id']}] {r['reason']}")
print("\n===== P 获胜样本理由（前 4 条） =====")
for r in d["results"][:4]:
    print(f"  [{r['query_id']}] {r['reason']}")
