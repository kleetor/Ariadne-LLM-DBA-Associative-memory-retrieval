# -*- coding: utf-8 -*-
"""
CROSS 跨场景指标（论文 §4.3：CROSS_OUT / CROSS_R，辅助诊断）

CROSS_OUT = 输出中 domain != query_domain 的节点率
CROSS_R   = 输出中 domain != query_domain 的节点数（独立于 R@all）
另输出 R@all（输出 ∩ expected）/|expected| 作为召回参照。

domain 由节点 id 前缀映射（LiFEMem 的 id 前缀即场景编码）。

用法:
    python -m eval.cross_metric --systems p:p_candidates.json,naive_rag:naive_rag_candidates.json
    python -m eval.cross_metric --systems p:p_candidates.json,p_inject:p_candidates_inject.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

# ---- domain 映射（LiFEMem 节点 id 前缀 → 场景域） ----
PREFIX_DOMAIN = {
    "w": "work", "n": "noise", "he": "health", "e": "emotion",
    "ln": "learn", "so": "social", "h": "highschool", "c": "college",
    "f": "family", "th": "time", "sh": "consume", "t": "travel",
}


def domain_of(node_id: str) -> str:
    """节点 id → 场景域。n31-n37 为工作加班叙事（n 前缀但属 work）"""
    if node_id.startswith("n"):
        try:
            if 31 <= int(node_id[1:]) <= 37:
                return "work"
        except ValueError:
            pass
        return "noise"
    for pref, dom in sorted(PREFIX_DOMAIN.items(), key=lambda x: -len(x[0])):
        if node_id.startswith(pref):
            return dom
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="CROSS 跨场景指标")
    parser.add_argument("--systems", default="p:p_candidates.json,naive_rag:naive_rag_candidates.json",
                        help="逗号分隔 name:file")
    args = parser.parse_args()

    # 查询集 + expected
    queries = yaml.safe_load(open(ROOT / "data/LiFEMem_queries.yaml", encoding="utf-8"))["queries"]
    q_by_id = {q["id"]: q for q in queries}

    systems = {}
    for item in args.systems.split(","):
        name, fname = item.split(":")
        systems[name] = {r["query_id"]: r for r in
                         json.load(open(ROOT / "eval/outputs" / fname, encoding="utf-8"))["results"]}

    # 每系统逐查询统计
    stats = {name: {"cross_out": [], "cross_r": [], "r_all": [], "n_out": []} for name in systems}
    detail = {name: [] for name in systems}

    for q in queries:
        qid = q["id"]
        expected = set(q.get("expected", []))
        # query_domain：expected 多数节点域（弃权 expected=[] 跳过 CROSS）
        doms = [domain_of(e) for e in expected]
        q_dom = max(set(doms), key=doms.count) if doms else None
        for name, by_id in systems.items():
            r = by_id.get(qid)
            if not r:
                continue
            out_ids = [c["id"] for c in r.get("candidates", [])]
            if q_dom:
                cross = [i for i in out_ids if domain_of(i) != q_dom]
                co = len(cross) / len(out_ids) if out_ids else 0
            else:
                cross = []
                co = None  # 弃权类无 query_domain，CROSS 不适用
            r_all = len(set(out_ids) & expected) / len(expected) if expected else None
            if co is not None:
                stats[name]["cross_out"].append(co)
                stats[name]["cross_r"].append(len(cross))
            stats[name]["r_all"].append(r_all)
            stats[name]["n_out"].append(len(out_ids))
            detail[name].append({"qid": qid, "category": q["category"], "query_domain": q_dom,
                                 "cross_out": co, "cross_r": len(cross), "r_all": r_all})

    # 汇总
    print("===== CROSS 跨场景指标汇总（按查询平均）=====")
    print(f"{'系统':<12}{'CROSS_OUT':>10}{'CROSS_R':>10}{'R@all':>10}{'输出量':>8}")
    for name in systems:
        s = stats[name]
        co = sum(s["cross_out"]) / len(s["cross_out"])
        cr = sum(s["cross_r"]) / len(s["cross_r"])
        ra = sum(x for x in s["r_all"] if x is not None) / sum(1 for x in s["r_all"] if x is not None)
        no = sum(s["n_out"]) / len(s["n_out"])
        print(f"{name:<12}{co:>10.3f}{cr:>10.1f}{ra:>10.3f}{no:>8.1f}")

    # 分能力
    print("\n===== 分能力 CROSS_OUT =====")
    cats = {}
    for name in systems:
        cats.setdefault("_order", [])
    for name in systems:
        for d in detail[name]:
            cats.setdefault(d["category"], {}).setdefault(name, []).append(d["cross_out"])
    for cat, bysys in sorted(cats.items()):
        if cat == "_order":
            continue
        line = f"  [{cat}]"
        for name in systems:
            vals = [v for v in bysys.get(name, []) if v is not None]
            line += f" {name}={sum(vals)/len(vals):.3f}" if vals else ""
        print(line)


if __name__ == "__main__":
    main()
