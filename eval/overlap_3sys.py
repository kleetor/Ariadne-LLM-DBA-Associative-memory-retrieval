# -*- coding: utf-8 -*-
"""三系统候选重叠分析：逐条 expected 命中 + 两两 Jaccard 重叠"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

FILES = {
    "P": "eval/outputs/p_candidates_v3.json",
    "Naive RAG": "eval/outputs/naive_rag_candidates_v3.json",
    "Graphiti": "eval/outputs/graphiti_candidates.json",
}
QUERIES = "data/LiFEMem_queries_v3.yaml"


def main() -> None:
    syss = {n: {r["query_id"]: {c["id"] for c in r.get("candidates", [])}
                for r in json.load(open(ROOT / p, encoding="utf-8"))["results"]}
            for n, p in FILES.items()}
    with open(ROOT / QUERIES, encoding="utf-8") as f:
        qs = yaml.safe_load(f)["queries"]

    names = list(syss.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    print(f"{'qid':<5}{'cat':<15}" + "".join(f"{n:>10}" for n in names) +
          "".join(f"{a[:3]}&{b[:3]:>8}" for a, b in pairs))
    agg = {n: {"hits": 0, "exp": 0} for n in names}
    jacc = {f"{a}|{b}": [] for a, b in pairs}
    for q in qs:
        qid, cat = q["id"], q.get("scenario", "")
        exp = set(q.get("expected", []))
        row = f"{qid:<5}{cat:<15}"
        for n in names:
            hits = exp & syss[n].get(qid, set())
            agg[n]["exp"] += len(exp)
            agg[n]["hits"] += len(hits)
            row += f"{len(hits)}/{len(exp):>7}"
        for a, b in pairs:
            sa, sb = syss[a].get(qid, set()), syss[b].get(qid, set())
            j = len(sa & sb) / len(sa | sb) if (sa | sb) else 0
            jacc[f"{a}|{b}"].append(j)
            row += f"{j:>10.2f}"
        print(row)

    print("\n===== 汇总 =====")
    for n in names:
        print(f"  {n}: expected 命中 {agg[n]['hits']}/{agg[n]['exp']} "
              f"= {agg[n]['hits']/agg[n]['exp']:.3f}")
    for a, b in pairs:
        js = jacc[f"{a}|{b}"]
        print(f"  {a} ∩ {b} Jaccard: 均值 {sum(js)/len(js):.3f}")


if __name__ == "__main__":
    main()
