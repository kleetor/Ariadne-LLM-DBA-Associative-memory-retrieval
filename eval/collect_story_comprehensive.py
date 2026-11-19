# -*- coding: utf-8 -*-
"""
综合能力测评 · 三系统 StoryRank 故事产物收集

按用户流程：各系统先各自产出【StoryRank 后】的故事产物，统一收集，之后才送 judge。
- Ariadne：build_retriever(comprehensive_gold) → retriever.retrieve_with_story（内部已 StoryRank）
            → 采 story + story_used（0927 审查 C1：输出契约变更，不再有采纳 id 列表）。
- Mem0 / Graphiti：读取各自 candidates（top-N 已达对齐 N）→ story_rank_for_texts
            （外部候选文本列表 → {nodes,edges=[]} → inference.story_rank）→ 采 story + adopted。
- 汇总为 story 产物表（逐 query 含三系统的 story），供后续同题并排 judge。

用法:
    python -m eval.collect_story_comprehensive          # 3 系统
    python -m eval.collect_story_comprehensive --systems ariadne,mem0,graphiti
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
GOLD_YAML = "data/dba_eval/comprehensive_gold.yaml"
MEM0_CAND = "eval/outputs/mem0_comprehensive_candidates.json"
GRAPHITI_CAND = "eval/outputs/graphiti_comprehensive_candidates.json"
P_CAND = "eval/outputs/p_candidates_comprehensive.json"
OUT_JSON = "eval/outputs/story_comprehensive_3sys.json"


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def load_queries() -> list:
    with open(ROOT / QUERIES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)["queries"]


def ariadne_stories(queries: list, retriever) -> dict:
    """Ariadne：retriever.retrieve_with_story（内部已 StoryRank）。返回 {qid: {story, adopted}}。"""
    out = {}
    for q in queries:
        qid = q["id"]
        res = retriever.retrieve_with_story(q["text"], with_response=False)
        stories = res.get("stories", [])
        story = "\n\n".join(stories) if isinstance(stories, list) else str(stories)
        out[qid] = {
            "story": story,
            # 0927 审查 C1 消费者同步：retrieve_with_story 不再返回 story_nodes（采纳 id 列表），
            # 改为 story_used（后端按正文实际覆盖统计的节点数，int），此处仅计数，直接取用。
            "adopted": res.get("story_used", 0),
            "n_candidates": res.get("total_candidates"),
        }
        print(f"[Ariadne] {qid} -> story {len(story)}字 / adopted {out[qid]['adopted']}", flush=True)
    return out


def load_candidates(path: str) -> dict:
    """读取候选文件 → {qid: [content, ...]}"""
    with open(ROOT / path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["query_id"]: [c.get("content", "") for c in r.get("candidates", [])] for r in data["results"]}


def external_stories(queries: list, cands_by_system: dict, inference, systems: list) -> dict:
    """外部框架候选 → story_rank_for_texts。返回 {system: {qid: {story, adopted}}}。"""
    from eval.story_rank_adapter import story_rank_for_texts
    out = {}
    for sysname in systems:
        out[sysname] = {}
        cand_by_qid = cands_by_system.get(sysname, {})
        for q in queries:
            qid = q["id"]
            texts = cand_by_qid.get(qid, [])
            res = story_rank_for_texts(inference, q["text"], texts)
            story = res.get("story", "")
            out[sysname][qid] = {
                "story": story,
                # 0927 审查 C1 消费者同步：story_rank 只返回 {story, used, degraded}，
                # 不再有 adopted_ids（采纳 id 列表）；used 即正文覆盖节点数，直接取用计数。
                "adopted": res.get("used", 0),
                "n_texts": len(texts),
            }
            print(f"[{sysname}] {qid} -> story {len(story)}字 / adopted {out[sysname][qid]['adopted']}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="三系统 StoryRank 故事产物收集")
    parser.add_argument("--systems", default="ariadne,mem0,graphiti", help="逗号分隔系统列表")
    args = parser.parse_args()
    systems = [s.strip() for s in args.systems.split(",")]

    _load_dotenv(ROOT / ".env")
    queries = load_queries()
    print(f"[collect] 查询 {len(queries)} 条，系统 {systems}")

    # 构建同一个 inference（LLM 一致），供外部故事化与 Ariadne 内部共用
    from eval.run_p_baseline import build_retriever
    _, retriever = build_retriever(GOLD_YAML)
    inference = retriever.inference

    # 基底：已存在的产物（本次未跑的系统沿用，避免重复计算）
    old = {}
    out_path = ROOT / OUT_JSON
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            old_data = json.load(f)
        old = {r["query_id"]: r["systems"] for r in old_data.get("queries", [])}
        print(f"[collect] 已加载旧产物 {len(old)} 条（沿用非本次系统的故事）")

    # 回溯每 query 的查询文本/category
    qinfo = {q["id"]: {"category": q.get("category", ""), "text": q["text"]} for q in queries}

    if "ariadne" in systems:
        ari = ariadne_stories(queries, retriever)
    else:
        ari = {}

    ext = {}
    if "mem0" in systems or "graphiti" in systems:
        cands = {}
        if "mem0" in systems:
            cands["mem0"] = load_candidates(MEM0_CAND)
        if "graphiti" in systems:
            cands["graphiti"] = load_candidates(GRAPHITI_CAND)
        ext = external_stories(queries, cands, inference,
                               [s for s in ("mem0", "graphiti") if s in systems])

    # 逐查询组装三系统 story（本次跑到 + 沿用旧产物中未跑的系统）
    all_systems = ["ariadne", "mem0", "graphiti"]
    collected = {"meta": {"query_set": QUERIES_YAML, "gold": GOLD_YAML, "systems": systems},
                 "queries": []}
    for q in queries:
        qid = q["id"]
        row = {"query_id": qid, "category": qinfo[qid]["category"], "query": qinfo[qid]["text"],
               "systems": {}}
        # 沿用旧产物（本次未跑的系统）
        for sysname in all_systems:
            if sysname not in systems and qid in old and sysname in old[qid]:
                row["systems"][sysname] = old[qid][sysname]
        if qid in ari:
            row["systems"]["ariadne"] = ari[qid]
        for sysname in systems:
            if sysname in ("mem0", "graphiti") and qid in ext.get(sysname, {}):
                row["systems"][sysname] = ext[sysname][qid]
        collected["queries"].append(row)

    out_path = ROOT / OUT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[collect] 故事产物已写: {out_path}（{len(collected['queries'])} 条查询）")


if __name__ == "__main__":
    main()
