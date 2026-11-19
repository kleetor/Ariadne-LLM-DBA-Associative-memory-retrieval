# -*- coding: utf-8 -*-
"""
纯故事对比（无查询）—— 对 base / boost / revert 三套候选集
给候选生成故事（story_rank_for_path，复用 base 缓存），再以 STORY_ONLY_PROMPT 盲评。

用途：回归复核——验证 TEMPORAL 权重/抽取改动对「主指标=story 质量」的影响，
并给出 base vs revert（同权重复跑）作为 LLM 噪声底。

用法:
    python -m eval.story_alone_multi
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever
from eval.story_rank_adapter import story_rank_for_path
from eval.judge_story_alone import STORY_ONLY_PROMPT

SYSTEMS = {
    "base": "eval/outputs/p_candidates.json",
    "boost": "eval/outputs/p_candidates_regress.json",
    "revert": "eval/outputs/p_candidates_regress2.json",
}
BASE_CACHE = "eval/outputs/story_texts.json"  # 已有 base("P") 故事缓存
DIMS = ["richness", "authenticity", "coherence", "memory_value", "structure"]


def main() -> None:
    parser = argparse.ArgumentParser(description="纯故事对比（任意候选集，5 维含时序结构）")
    parser.add_argument("--queries", default="data/LiFEMem_queries.yaml")
    parser.add_argument("--out", default="eval/outputs/story_alone_regress.json")
    parser.add_argument("--story-cache", default="eval/outputs/story_alone_regress_stories.json")
    parser.add_argument("--reuse-base", action="store_true", help="复用 story_texts.json 的 base(P) 故事")
    parser.add_argument("--systems", default="base:p_candidates.json,boost:p_candidates_regress.json,revert:p_candidates_regress2.json",
                        help="逗号分隔 name:file（相对 eval/outputs/）")
    parser.add_argument("--only-queries", default="", help="逗号分隔查询 id 子集（默认全部）")
    args = parser.parse_args()

    systems = {}
    for item in args.systems.split(","):
        name, fname = item.strip().split(":", 1)
        systems[name] = fname
    SYSTEMS.update(systems)
    only = [q.strip() for q in args.only_queries.split(",") if q.strip()]

    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever("data/LiFEMem.yaml")
    inference = retriever.inference
    llm = retriever.llm if hasattr(retriever, "llm") else None
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
            temperature=0,
        )

    import yaml
    queries = yaml.safe_load(open(ROOT / args.queries, encoding="utf-8"))["queries"]
    queries = [q for q in queries if q.get("category") != "abstention"]
    if only:
        queries = [q for q in queries if q["id"] in only]

    cands = {}
    for name, fname in SYSTEMS.items():
        cands[name] = {r["query_id"]: r for r in
                       json.load(open(ROOT / fname, encoding="utf-8"))["results"]}

    base_cache = {}
    if args.reuse_base and (ROOT / BASE_CACHE).exists():
        base_cache = json.load(open(ROOT / BASE_CACHE, encoding="utf-8"))

    story_cache = {}
    if (ROOT / args.story_cache).exists():
        story_cache = json.load(open(ROOT / args.story_cache, encoding="utf-8"))

    agg = {name: {d: [] for d in DIMS} for name in SYSTEMS}
    results = []

    for q in queries:
        qid = q["id"]
        for name in SYSTEMS:
            r = cands[name].get(qid)
            if not r:
                continue
            sc = story_cache.setdefault(qid, {"stories": {}})
            story = sc["stories"].get(name, {}).get("story", "")
            if not story:
                if name == "base" and args.reuse_base:
                    story = base_cache.get(qid, {}).get("stories", {}).get("P", {}).get("story", "")
                if not story:
                    story = story_rank_for_path(inference, q["text"], r.get("path", {"nodes": [], "edges": []}))
                    story = story.get("story", "")
                sc["stories"][name] = {"story": story}
                with open(ROOT / args.story_cache, "w", encoding="utf-8") as f:
                    json.dump(story_cache, f, ensure_ascii=False, indent=1)
            if not story:
                continue
            resp = llm.invoke(STORY_ONLY_PROMPT.format(story=story))
            text = resp.content.strip()
            scores, comment = {}, ""
            try:
                s = text.find("{")
                e = text.rfind("}")
                obj = json.loads(text[s:e + 1])
                scores = {d: int(obj.get(d, 0)) for d in DIMS}
                comment = obj.get("comment", "")
            except Exception:
                comment = text[:200]
            for d in DIMS:
                agg[name][d].append(scores.get(d, 0))
            results.append({"query_id": qid, "system": name, "scores": scores,
                            "comment": comment[:150], "story_len": len(story)})
            print(f"[ALONE] {qid} {name} {scores}", flush=True)

    summary = {"systems": {}}
    for name in SYSTEMS:
        summary["systems"][name] = {
            d: round(sum(agg[name][d]) / len(agg[name][d]), 2) for d in DIMS
        }
    summary["count"] = {name: len(agg[name]["richness"]) for name in SYSTEMS}
    out = {"summary": summary, "results": results}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[ALONE] 纯故事评分汇总（各系统）:")
    for name in SYSTEMS:
        print(f"  {name}: {summary['systems'][name]}  n={summary['count'][name]}")


if __name__ == "__main__":
    main()
