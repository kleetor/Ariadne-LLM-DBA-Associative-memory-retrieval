# -*- coding: utf-8 -*-
"""
生成"Graphiti 原始事实文本"故事缓存，供纯选对比：
P 的故事（StoryRank 产物） vs Graphiti 检索返回的原始事实文本（未过 StoryRank）。

把 graphiti_candidates.json 每个查询候选的 `fact` 字段按检索顺序拼接为一段文本，
作为该查询的 Graphiti"故事"输入（key = graphiti_facts），复用 judge_story_pair.py。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC = "eval/outputs/graphiti_candidates.json"
OUT = "eval/outputs/story_texts_graphiti_facts.json"

data = json.load(open(ROOT / SRC, encoding="utf-8"))
out = {}
for r in data["results"]:
    qid = r["query_id"]
    facts = []
    for c in r.get("candidates", []):
        f = c.get("fact", "").strip()
        if f:
            facts.append(f)
    text = "\n".join(f"- {f}" for f in facts)
    out[qid] = {"query": r.get("query", ""), "category": r.get("scenario", ""),
                "stories": {"graphiti_facts": {"story": text, "n_facts": len(facts)}}}

with open(ROOT / OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# 摘要
lens = [len(v["stories"]["graphiti_facts"]["story"]) for v in out.values()]
ns = [v["stories"]["graphiti_facts"]["n_facts"] for v in out.values()]
print(f"已生成 {OUT}")
print(f"  查询数: {len(out)}")
print(f"  fact 数均值: {sum(ns)/len(ns):.1f}（范围 {min(ns)}-{max(ns)}）")
print(f"  拼接文本长度均值: {sum(lens)/len(lens):.0f} 字（范围 {min(lens)}-{max(lens)}）")
