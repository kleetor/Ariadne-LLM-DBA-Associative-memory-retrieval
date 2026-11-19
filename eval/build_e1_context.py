# -*- coding: utf-8 -*-
"""E1 重跑：用当前链路重新生成 Ariadne 的上下文块，与存档的 Mem0 / Graphiti 块拼成三系统。

为什么只重生成 Ariadne：Mem0 / Graphiti 的块来自存档（同一批 query、同一候选配额），
重跑它们需要各自框架环境；E1 要回答的是"当前链路下 Ariadne 是否仍领先"，故固定外部块、
只换 Ariadne，比较口径与 0908 e2e 文档一致。

输出：eval/outputs/e1/context_e1_current.json（judge_comprehensive 可直接消费）
用法：python -m eval.build_e1_context
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

GOLD = "data/dba_eval/comprehensive_gold.yaml"
QUERIES = "data/dba_eval/comprehensive_queries.yaml"
ARCH = "eval/outputs/context_comprehensive_3sys.json"
OUT = "eval/outputs/e1/context_e1_current.json"


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 重跑：重生成 Ariadne 块")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部 76）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    _, retriever = build_retriever(str(ROOT / GOLD), vector_cache_key="comprehensive")
    with open(ROOT / QUERIES, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    if args.limit:
        queries = queries[: args.limit]
    with open(ROOT / ARCH, encoding="utf-8") as f:
        arch = {r["query_id"]: r for r in json.load(f)["queries"]}

    print(f"[E1] queries={len(queries)} commit={_commit()} "
          f"emb={os.environ.get('EMBEDDING_MODEL')} "
          f"tau={retriever.purpose_filter_threshold}", flush=True)

    rows = []
    for q in queries:
        qid = str(q["id"])
        a = arch.get(qid) or {}
        res = retriever.retrieve_with_story(str(q["text"]), with_response=False)
        story = "\n\n".join(res.get("stories") or [])
        systems = dict(a.get("systems") or {})
        systems["ariadne"] = {"story": story, "n_candidates": res.get("total_candidates")}
        rows.append({"query_id": qid, "category": q.get("category", ""),
                     "query": q["text"], "systems": systems})
        print(f"  {qid} [{q.get('category','')}] story={len(story)}字 "
              f"nodes={res.get('path_nodes')} edges={res.get('path_edges')}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {"code": _commit(), "embedding": os.environ.get("EMBEDDING_MODEL"),
                 "purpose_filter_threshold": retriever.purpose_filter_threshold,
                 "query_set": QUERIES, "note": "ariadne=current chain; mem0/graphiti=archive"},
        "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E1] wrote {out_path}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
