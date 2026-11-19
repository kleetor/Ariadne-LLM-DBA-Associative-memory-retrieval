# -*- coding: utf-8 -*-
"""
综合能力测评 · 三系统「结构化记忆上下文块」收集（供 judge 直评）

按 0831 e2e 计划 §4.5.2 口径：评测对象 = 系统喂给模型的结构化记忆上下文块。
- Ariadne   = {story}          （模型实际看到的检索故事，内部已 StoryRank）
- Mem0      = {memories}       （检索返回的记忆文本列表，不做 StoryRank）
- Graphiti  = {entities,relations}  （检索返回的实体+关系事实，不做 StoryRank）

judge 材料统一为「喂给模型的结构化记忆上下文」，同题并排交给 judge（§4.5.3）。
本脚本只负责把三系统的检索候选组装成各自上下文块，不做 StoryRank、不生成回复。

上下文块构造：
- Ariadne：沿用已收集的 story（read story_comprehensive_3sys.json 的 ariadne.story）。
- Mem0：候选 → 去重记忆文本列表（保留 score 降序）。
- Graphiti：候选 content 去重 = entities；候选 fact 去重 = relations（fact 含 rel_type）。

用法:
    python -m eval.collect_context_comprehensive --systems ariadne,mem0,graphiti
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

QUERIES_YAML = "data/dba_eval/comprehensive_queries.yaml"
MEM0_CAND = "eval/outputs/mem0_comprehensive_candidates.json"
GRAPHITI_CAND = "eval/outputs/graphiti_comprehensive_candidates.json"
ARIADNE_STORY = "eval/outputs/story_comprehensive_3sys.json"   # 已收集的 ariadne story
OUT_JSON = "eval/outputs/context_comprehensive_3sys.json"


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def load_queries() -> list:
    with open(ROOT / QUERIES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)["queries"]


def load_ariadne_story() -> dict:
    """读取已收集的 Ariadne story → {qid: story}"""
    p = ROOT / ARIADNE_STORY
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {r["query_id"]: r["systems"].get("ariadne", {}).get("story", "")
            for r in data.get("queries", [])}


def load_mem0_candidates() -> dict:
    """候选 → {qid: [memory_text, ...]}（保留顺序，去重）"""
    with open(ROOT / MEM0_CAND, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data["results"]:
        qid = r["query_id"]
        mems = []
        seen = set()
        for c in r.get("candidates", []):
            t = c.get("content", "").strip()
            if t and t not in seen:
                seen.add(t)
                mems.append({"text": t, "score": c.get("score")})
        out[qid] = mems
    return out


def load_graphiti_candidates() -> dict:
    """候选 → {qid: {entities: [...], relations: [...]}}

    entities = 候选 content 去重；relations = 候选 fact 去重（fact 含 rel_type 三元组）。
    """
    with open(ROOT / GRAPHITI_CAND, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data["results"]:
        qid = r["query_id"]
        entities, relations = [], []
        seen_ent, seen_rel = set(), set()
        for c in r.get("candidates", []):
            ent = c.get("content", "").strip()
            if ent and ent not in seen_ent:
                seen_ent.add(ent)
                entities.append(ent)
            rel = c.get("fact", "").strip()
            if rel and rel not in seen_rel:
                seen_rel.add(rel)
                relations.append(rel)
        out[qid] = {"entities": entities, "relations": relations}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="三系统结构化记忆上下文块收集")
    parser.add_argument("--systems", default="ariadne,mem0,graphiti", help="逗号分隔系统列表")
    args = parser.parse_args()
    systems = [s.strip() for s in args.systems.split(",")]

    _load_dotenv(ROOT / ".env")
    queries = load_queries()
    print(f"[collect] 查询 {len(queries)} 条，系统 {systems}")

    qinfo = {q["id"]: {"category": q.get("category", ""), "text": q["text"]} for q in queries}

    ari_story = load_ariadne_story() if "ariadne" in systems else {}
    mem0_cands = load_mem0_candidates() if "mem0" in systems else {}
    graph_cands = load_graphiti_candidates() if "graphiti" in systems else {}

    collected = {"meta": {"query_set": QUERIES_YAML,
                          "context_format": "ariadne={story}, mem0={memories}, graphiti={entities,relations}",
                          "systems": systems}, "queries": []}
    for q in queries:
        qid = q["id"]
        row = {"query_id": qid, "category": qinfo[qid]["category"], "query": qinfo[qid]["text"],
               "systems": {}}
        if "ariadne" in systems and qid in ari_story:
            row["systems"]["ariadne"] = {"story": ari_story[qid],
                                         "n_candidates": None}
        if "mem0" in systems and qid in mem0_cands:
            mems = mem0_cands[qid]
            row["systems"]["mem0"] = {"memories": [m["text"] for m in mems],
                                      "n_memories": len(mems)}
        if "graphiti" in systems and qid in graph_cands:
            g = graph_cands[qid]
            row["systems"]["graphiti"] = {"entities": g["entities"], "relations": g["relations"],
                                          "n_entities": len(g["entities"]),
                                          "n_relations": len(g["relations"])}
        collected["queries"].append(row)

    out_path = ROOT / OUT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[collect] 上下文块已写: {out_path}（{len(collected['queries'])} 条查询）")


if __name__ == "__main__":
    main()
