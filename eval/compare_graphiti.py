# -*- coding: utf-8 -*-
"""
Graphiti vs P vs Naive RAG vs Mem0 检索层指标对比（LiFEMem v3）

在 compare_mem0.py 基础上加入 Graphiti 候选（frameworks/graphiti_runner.py 产出）。

指标（辅助诊断，非主结论）：
- expected 覆盖：候选集中命中的 expected 节点数 / expected 总数（字面召回）
- 弃权：abstention 查询是否产生候选
- CROSS_OUT：跨场景率（复用 cross_out_v3 的域定义）
- 输出量：候选条数（各框架按 P 输出量 top-N 对齐）

用法:
    python -m eval.compare_graphiti
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

FILES = {
    "P": "eval/outputs/p_candidates_v3.json",
    "RAG": "eval/outputs/naive_rag_candidates_v3.json",
    "Mem0": "eval/outputs/mem0_candidates.json",
    "Graphiti": "eval/outputs/graphiti_candidates.json",
}
QUERIES = "data/LiFEMem_queries_v3.yaml"

# 情境 → 所属域前缀（与 cross_out_v3.py 一致）
DOM = {"work_pressure": {"w"}, "health": {"he"}, "emotion": {"e"}, "nostalgia": {"h", "c"},
       "relationship": {"h"}, "life": {"w", "e", "f", "sh", "t"}, "learning": {"ln"},
       "social": {"so"}, "future": {"f", "w", "ln"}, "abstention": set()}
PFS = ["he", "ln", "sh", "so", "w", "e", "h", "c", "f", "t", "n"]


def prefix(nid):
    for pf in PFS:
        if nid.startswith(pf):
            return pf
    return "?"


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return {r["query_id"]: r for r in json.load(f)["results"]}


def main() -> None:
    systems = {name: load(p) for name, p in FILES.items()}
    with open(ROOT / QUERIES, encoding="utf-8") as f:
        queries = {q["id"]: q for q in yaml.safe_load(f)["queries"]}

    agg = {name: {"exp_hit": 0, "exp_total": 0, "exp_count": 0,
                  "cross_num": 0, "cross_den": 0, "n_total": 0,
                  "abstention_nonempty": 0, "abstention_total": 0} for name in systems}

    header = f"{'qid':<5}{'scenario':<15}" + "".join(f"{n:>12}" for n in systems) + \
             "".join(f"{n:>10}" for n in systems)
    print(header)
    for qid, q in queries.items():
        scen = q.get("scenario", "")
        expected = set(q.get("expected", []))
        home = DOM.get(scen, set())
        row = f"{qid:<5}{scen:<15}"
        per = {}
        for name, by_id in systems.items():
            cands = by_id.get(qid, {}).get("candidates", [])
            ids = [c["id"] for c in cands]
            exp = len(expected & set(ids)) if expected else None
            if exp is not None:
                agg[name]["exp_hit"] += exp
                agg[name]["exp_total"] += len(expected)
                agg[name]["exp_count"] += 1
            if home:
                cross = sum(1 for i in ids if prefix(i) not in home) / len(ids) if ids else 0
                agg[name]["cross_num"] += cross * len(ids)
                agg[name]["cross_den"] += len(ids)
            agg[name]["n_total"] += len(ids)
            if scen == "abstention":
                agg[name]["abstention_total"] += 1
                if ids:
                    agg[name]["abstention_nonempty"] += 1
            per[name] = (f"{exp}/{len(expected)}" if exp is not None else "-", f"{cross:.2f}" if home else "-")
        row += "".join(f"{per[n][0]:>12}" for n in systems)
        row += "".join(f"{per[n][1]:>10}" for n in systems)
        print(row)

    print("\n===== 汇总 =====")
    for name, s in agg.items():
        print(f"\n[{name}] 非弃权查询 {s['exp_count']} 条")
        print(f"  expected 覆盖: {s['exp_hit']}/{s['exp_total']} = "
              f"{s['exp_hit']/s['exp_total']:.3f}")
        print(f"  CROSS_OUT（跨场景率，加权）: {s['cross_num']/s['cross_den']:.3f}  (n={s['cross_den']})")
        print(f"  弃权查询仍有候选: {s['abstention_nonempty']}/{s['abstention_total']}")
        print(f"  平均输出量: {s['n_total']/35:.1f}")


if __name__ == "__main__":
    main()
