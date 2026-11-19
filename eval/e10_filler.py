# -*- coding: utf-8 -*-
"""E10-D：填充位效应（4 臂 × 3 套分化图 × 2 个 τ）。

测的是什么：
当某个意图真正相关的记忆不足时，纯向量的 top-k 会把**凑数**的无关节点一起塞进来；
PAR 有逐跳的绝对阈值挡板，理应混入更少。指标是**跨主体混入率**——
输出节点里属于其他主体（B 语文老师 / C 电商运营）的比例。

四臂（缺了任何一臂都无法归因）：
  par                       完整 PAR（文本 + 目的，逐跳过滤）
  vector_purpose            文本+目的 top-k，**无阈值**
  vector_purpose_thresholded 同上，但 top-k 后再用**同一 τ** 过滤
  vector_text               仅文本 top-k（对目的不可见）

为什么必须同时有 `vector_purpose_thresholded`：
`par` 与 `vector_purpose` 差两个因素——有/无阈值、有/无图扩展。只有加上
"同样加了 τ 的向量臂"，才能把"阈值挡板"的贡献从"图扩展"里分离出来。

τ 取 0.30（现行默认，本图上几乎不筛）与 0.55（本图最优分离点），par 与
`vector_purpose_thresholded` 用**同一个 τ**，保证对比公平。

用法：python -m eval.e10_filler
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eval.e3_par_vs_vector import _cosine, _path_from_ids  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

GRAPHS = {"L": "data/dba_eval/e10_L.yaml",
          "M": "data/dba_eval/e10_M.yaml",
          "H": "data/dba_eval/e10_H.yaml"}
TAUS = (0.30, 0.55)
ARMS = ("par", "vector_purpose", "vector_purpose_thresholded", "vector_text")


def _subject(nid: str) -> str:
    c = nid[0]
    return "A" if c in ("u", "a", "t") else c.upper()


def _topk(retriever, text: str, k: int, cos_min: float | None = None):
    vs, graph = retriever.vector_store, retriever.graph
    qv = vs._embed(text)
    ids = [n for n in graph.graph.nodes() if graph.graph.nodes[n].get("content")]
    vecs = vs.get_content_vectors(ids)
    scored = [(n, _cosine(qv, v)) for n, v in zip(ids, vecs) if v is not None]
    scored.sort(key=lambda p: -p[1])
    if cos_min is not None:
        scored = [p for p in scored if p[1] >= cos_min]
    return scored[:k]


def _flatten(path: Path) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    out = []
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            out.append({"id": f"{prefix}_{it['intent']}", "text": sc["text"],
                        "intent": it["intent"], "purposes": list(it.get("purposes") or [])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E10-D 填充位效应")
    ap.add_argument("--queries", default="data/dba_eval/intent_queries.yaml")
    ap.add_argument("--out", default="eval/outputs/e10/filler.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    queries = _flatten(ROOT / args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print(f"[E10-D] queries={len(queries)} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    report: Dict[str, Any] = {}
    for gname, gpath in GRAPHS.items():
        graph, retriever = build_retriever(str(ROOT / gpath), vector_cache_key=f"e10_{gname}")
        report[gname] = {"nodes": graph.node_count, "by_tau": {}}
        for tau in TAUS:
            retriever.purpose_filter_threshold = tau
            rows: Dict[str, List[Dict[str, float]]] = {a: [] for a in ARMS}
            for q in queries:
                purposes = list(q["purposes"])
                par = retriever.retrieve_with_story(
                    q["text"], seed_k=5, story=False, with_response=False, purpose=purposes)
                par_ids = [n["id"] for n in (par.get("path") or {}).get("nodes") or []]
                k = max(len(par_ids), 1)
                joined = q["text"] + " " + " ".join(purposes)
                arms_ids = {
                    "par": par_ids,
                    "vector_purpose": [n for n, _ in _topk(retriever, joined, k)],
                    "vector_purpose_thresholded": [n for n, _ in _topk(retriever, joined, k, cos_min=tau)],
                    "vector_text": [n for n, _ in _topk(retriever, q["text"], k)],
                }
                for arm, ids in arms_ids.items():
                    off = [i for i in ids if _subject(i) != "A"]
                    rows[arm].append({
                        "n": len(ids),
                        "off": len(off),
                        "ratio": (len(off) / len(ids)) if ids else 0.0,
                    })
            summary = {}
            for arm in ARMS:
                rs = rows[arm]
                summary[arm] = {
                    "mean_nodes": round(statistics.mean(x["n"] for x in rs), 1),
                    "off_ratio": round(statistics.mean(x["ratio"] for x in rs), 4),
                }
            # 逐题配对：par 比各臂混入更少的比例
            for arm in ARMS[1:]:
                w = sum(1 for a, b in zip(rows["par"], rows[arm]) if a["ratio"] < b["ratio"] - 1e-9)
                t = sum(1 for a, b in zip(rows["par"], rows[arm]) if abs(a["ratio"] - b["ratio"]) <= 1e-9)
                summary.setdefault("par_vs", {})[arm] = {"win": w, "tie": t, "n": len(rows["par"])}
            report[gname]["by_tau"][f"{tau:.2f}"] = summary
            line = "  ".join(f"{a}={summary[a]['off_ratio']:.1%}" for a in ARMS)
            print(f"  [{gname}] τ={tau:.2f}  节点≈{summary['par']['mean_nodes']}  {line}", flush=True)

    print("\n=== 跨主体混入率（越低越好；括号内为 par 逐题胜出数）===")
    print(f"{'图':4s} {'τ':>5s} " + " ".join(f"{a[:22]:>24s}" for a in ARMS))
    for gname, g in report.items():
        for tau, s in g["by_tau"].items():
            cells = []
            for a in ARMS:
                w = s.get("par_vs", {}).get(a, {}).get("win")
                extra = f" ({w}/{s['par_vs'][a]['n']})" if w is not None else ""
                cells.append(f"{s[a]['off_ratio']:>17.1%}{extra:>7s}")
            print(f"{gname:4s} {tau:>5s} " + " ".join(cells))

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"experiment": "E10-D", "taus": list(TAUS),
                                      "queries": len(queries),
                                      "embedding": os.environ.get("EMBEDDING_MODEL")},
                             "graphs": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E10-D] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
