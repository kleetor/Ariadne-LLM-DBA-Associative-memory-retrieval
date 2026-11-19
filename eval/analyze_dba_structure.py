# -*- coding: utf-8 -*-
"""
DBA 多次抽取的结构差异分析：观察 LLM 重复抽取同一语料时图结构的一致性

对多个版本（如 v4/v5/v6/v7）的 DBA 建图：
- 每版基础统计（节点/边/覆盖 gold 节点/gold 边）
- 两两版本的覆盖结构 Jaccard（节点覆盖、边覆盖）
- 与 gold 的重合率

用法:
    python -m eval.analyze_dba_structure --versions v4,v5,v6,v7
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from dba_pipeline.loader import load_graph


def _load_gold_subset(full_yaml: Path, gold_node_ids: set):
    g = load_graph(str(full_yaml))
    gold_edges = []
    for u, v, data in g.graph.edges(data=True):
        if u in gold_node_ids and v in gold_node_ids:
            gold_edges.append((u, v))
    return set(gold_edges)


def _load_version(prefix: str):
    eval_path = ROOT / f"eval/outputs/dba_extraction_subset_b6_{prefix}.json"
    graph_path = ROOT / f"eval/outputs/dba_graph_subset_b6_{prefix}.yaml"
    ev = json.loads(eval_path.read_text(encoding="utf-8"))["eval"]
    d2g = {d["id"]: d["matched_gold_id"] for d in ev["dba_node_detail"] if d.get("matched_gold_id")}
    g = load_graph(str(graph_path))
    dba_edges = set()
    for a, b in g.graph.edges():
        dba_edges.add((a, b))
        dba_edges.add((b, a))
    return {
        "prefix": prefix,
        "nodes": ev["dba_active_nodes"],
        "edges": ev["dba_edges"],
        "node_recall": ev["node_recall"],
        "edge_recall": ev["edge_recall"],
        "d2g": d2g,
        "dba_edges": dba_edges,
    }


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return round(len(a & b) / len(union), 3) if union else 1.0


def _covered_gold_edges(v: dict, gold_edges: set) -> set:
    g2d = {g: d for d, g in v["d2g"].items()}
    cov = set()
    for u, vv in gold_edges:
        if u in g2d and vv in g2d and (g2d[u], g2d[vv]) in v["dba_edges"]:
            cov.add((u, vv))
    return cov


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 多次抽取结构差异分析")
    parser.add_argument("--versions", default="v4,v5,v6,v7", help="逗号分隔版本前缀")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_subset_dialogs.yaml")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml")
    args = parser.parse_args()

    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dlg = yaml.safe_load(f)
    gold_node_ids = set(dlg["subset"]["gold_nodes"])
    gold_edges = _load_gold_subset(ROOT / args.full_yaml, gold_node_ids)

    vers = [args.versions.split(",")[0]] if "," not in args.versions else args.versions.split(",")
    vers = [v.strip() for v in vers if v.strip()]
    data = [_load_version(v) for v in vers]

    # 每版覆盖的 gold 节点集合（映射目标）
    for v in data:
        v["covered_gold_nodes"] = set(v["d2g"].values())
        v["covered_gold_edges"] = _covered_gold_edges(v, gold_edges)
        v["gold_edge_overlap"] = round(len(v["covered_gold_edges"]) / len(gold_edges), 3)

    print("===== 每版基础统计 =====")
    print(f"{'版本':<6}{'节点':>5}{'边':>5}{'节点召回':>8}{'边召回':>8}{'gold边重合':>10}")
    for v in data:
        print(f"{v['prefix']:<6}{v['nodes']:>5}{v['edges']:>5}"
              f"{v['node_recall']:>8.1%}{v['edge_recall']:>8.1%}{v['gold_edge_overlap']:>10.1%}")

    print("\n===== 两两版本结构一致性（覆盖 gold 节点集合 Jaccard）=====")
    names = [v["prefix"] for v in data]
    print("        " + " ".join(f"{n:>7}" for n in names))
    for i, a in enumerate(data):
        row = [f"{a['prefix']:>6}"]
        for j, b in enumerate(data):
            jac = _jaccard(a["covered_gold_nodes"], b["covered_gold_nodes"])
            row.append(f"{jac:>7}")
        print(" ".join(row))

    print("\n===== 两两版本结构一致性（覆盖 gold 边集合 Jaccard）=====")
    print("        " + " ".join(f"{n:>7}" for n in names))
    for i, a in enumerate(data):
        row = [f"{a['prefix']:>6}"]
        for j, b in enumerate(data):
            jac = _jaccard(a["covered_gold_edges"], b["covered_gold_edges"])
            row.append(f"{jac:>7}")
        print(" ".join(row))

    # 边集合差异补充：各版本独有的 gold 边（相对其他版本）
    print("\n===== 各版本独有的 gold 边（相对其他版本并集） =====")
    union_edges = set().union(*[v["covered_gold_edges"] for v in data])
    for v in data:
        others = set().union(*[o["covered_gold_edges"] for o in data if o is not v])
        uniq = v["covered_gold_edges"] - others
        print(f"{v['prefix']}: 覆盖 {len(v['covered_gold_edges'])} 条，独有 {len(uniq)} 条（共覆盖并集 {len(union_edges)} 条）")

    print("\n（结论判据：若各版本结构 Jaccard 低但此前性能相当 → 图结构一致性要求低的鲁棒验证）")


if __name__ == "__main__":
    main()
