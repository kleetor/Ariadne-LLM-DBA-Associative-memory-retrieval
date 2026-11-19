# -*- coding: utf-8 -*-
"""E7：输出稳定性（同一系统重复跑，结果抖多少）。

为什么要单测这个：
E1/E3/E4/E5 全部是**单次生成**，分数里含一个从未量化的重复噪声。E5 已经暴露了线索——
`vector_text` 的排序在 5/10 情景下不是逐位一致（同一文本，两次结果不同），说明
**embedding 接口有微小非确定性**。若这个抖动足够大，单次跑分之间的差异就不能全归给机制。

测量分两层，分开报（混在一起会无法归因）：
1. **检索层**（无 LLM）：同一 query 重跑 R 次，比较输出节点集合的 Jaccard
   —— 这一层只含 embedding 抖动 + FAISS 检索；
2. **生成层**（有 LLM，抽样）：把同一条 path 交给 StoryRank 跑 R 次，比较正文
   —— 这一层含 LLM sampling 抖动。用**字符级相似度**（difflib）粗看，不调用 judge。

臂：`par`（注入目的，排除推断随机性）与 `vector_purpose`（等规模 top-k）。
**必须用 inject**：infer 模式下 LLM 目的推断本身就是随机源，会把两层抖动混在一起。

用法：
    python -m eval.e7_stability --reps 3
    python -m eval.e7_stability --reps 3 --story-sample 10
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.e3_par_vs_vector import _path_from_ids, _vector_topk  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _pairs(xs: List[Any]):
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            yield xs[i], xs[j]


def main() -> int:
    ap = argparse.ArgumentParser(description="E7 输出稳定性")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml")
    ap.add_argument("--out", default="eval/outputs/e7/stability.json")
    ap.add_argument("--reps", type=int, default=3, help="每臂重跑次数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部非弃权 70）")
    ap.add_argument("--story-sample", type=int, default=10,
                    help="生成层抽样条数（0=不测生成层）")
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--vector-cache-key", default="comprehensive")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        queries = [q for q in (yaml.safe_load(f) or {}).get("queries", [])
                   if str(q.get("category") or "") != "abstention"]
    if args.limit:
        queries = queries[: args.limit]
    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model,
        vector_cache_key=args.vector_cache_key)
    print(f"[E7] queries={len(queries)} reps={args.reps} emb={os.environ.get('EMBEDDING_MODEL')}",
          flush=True)

    per_q: Dict[str, Dict[str, List[List[str]]]] = {}
    for rep in range(args.reps):
        for q in queries:
            qid, text = str(q["id"]), str(q["text"])
            purposes = list(q.get("purposes") or [])
            par = retriever.retrieve_with_story(
                text, seed_k=args.seed_k, story=False, with_response=False,
                purpose=purposes or None)
            par_ids = [n["id"] for n in (par.get("path") or {}).get("nodes") or []]
            k = max(len(par_ids), 1)
            vec = _path_from_ids(retriever, _vector_topk(retriever, text + " " + " ".join(purposes), k),
                                 with_edges=False)
            vec_ids = [n["id"] for n in (vec.get("nodes") or [])]
            d = per_q.setdefault(qid, {"par": [], "vector_purpose": []})
            d["par"].append(par_ids)
            d["vector_purpose"].append(vec_ids)
        print(f"  rep {rep+1}/{args.reps} done", flush=True)

    report: Dict[str, Any] = {"retrieval": {}, "per_query": {}}
    for arm in ("par", "vector_purpose"):
        jac_all, size_std, identical = [], [], 0
        for qid, d in per_q.items():
            sets = [set(x) for x in d[arm]]
            jac = [_jaccard(a, b) for a, b in _pairs(sets)]
            jac_all.extend(jac)
            sizes = [len(x) for x in d[arm]]
            size_std.append(max(sizes) - min(sizes))
            if all(s == sets[0] for s in sets):
                identical += 1
            report["per_query"].setdefault(qid, {})[arm] = {
                "jaccard_min": round(min(jac), 4) if jac else None,
                "jaccard_mean": round(statistics.mean(jac), 4) if jac else None,
                "size_range": [min(sizes), max(sizes)],
            }
        report["retrieval"][arm] = {
            "jaccard_mean": round(statistics.mean(jac_all), 4) if jac_all else None,
            "jaccard_p05": round(float(sorted(jac_all)[max(0, int(0.05 * len(jac_all)))]), 4),
            "identical_all_reps": identical,
            "n": len(per_q),
            "size_range_mean": round(statistics.mean(size_std), 2),
        }

    print("\n=== 检索层稳定性（同 query 重跑 {} 次）===".format(args.reps))
    for arm, r in report["retrieval"].items():
        print(f"  {arm:15s} Jaccard 均值={r['jaccard_mean']:.3f}  p05={r['jaccard_p05']:.3f}  "
              f"三次完全一致={r['identical_all_reps']}/{r['n']}  "
              f"节点数极差(平均)={r['size_range_mean']}")

    if args.story_sample and args.reps > 1:
        sample = queries[: args.story_sample]
        print(f"\n=== 生成层稳定性（抽样 {len(sample)} 条，StoryRank 重跑 {args.reps} 次）===")
        sims = []
        for q in sample:
            text = str(q["text"])
            purposes = list(q.get("purposes") or [])
            par = retriever.retrieve_with_story(
                text, seed_k=args.seed_k, story=False, with_response=False,
                purpose=purposes or None)
            path = par.get("path") or {"nodes": [], "edges": []}
            stories = []
            for _ in range(args.reps):
                stories.append(str(retriever.inference.story_rank(text, path).get("story") or ""))
            pair_sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in _pairs(stories)]
            sims.extend(pair_sims)
            print(f"  {q['id']}  相似度={statistics.mean(pair_sims):.3f}  "
                  f"字数={[len(s) for s in stories]}", flush=True)
        report["generation"] = {
            "n": len(sample), "reps": args.reps,
            "char_similarity_mean": round(statistics.mean(sims), 4) if sims else None,
            "char_similarity_min": round(min(sims), 4) if sims else None,
        }
        print(f"  平均字符相似度={report['generation']['char_similarity_mean']:.3f}  "
              f"最低={report['generation']['char_similarity_min']:.3f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"experiment": "E7", "reps": args.reps, "queries": len(queries),
                 "embedding": os.environ.get("EMBEDDING_MODEL"),
                 "note": ("检索层只含 embedding/FAISS 抖动（purpose 注入，排除推断随机性）；"
                          "生成层为 StoryRank LLM sampling 抖动，用字符相似度粗看。")},
        **report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E7] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
