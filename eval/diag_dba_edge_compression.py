# -*- coding: utf-8 -*-
"""
DBA 边压缩归因审计：边压缩是节点压缩的果，还是 LLM 独立连边保守？

对每条 gold 子集内部边 (u, v) 分类：
  A. 两端节点都被 DBA 抽到（映射存在）且 DBA 图已连 → 正确连接
  B. 两端节点都被 DBA 抽到，但 DBA 图未连 → LLM 连边判断/可见性问题（非节点缺失）
  C. 至少一端节点未被 DBA 抽到 → 节点压缩导致无法连边（节点缺失是直接原因）

结论判据：B 大 → 连边本身保守；C 大 → 节点压缩导致边压缩（用户假设成立）。

用法:
    python -m eval.diag_dba_edge_compression
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from dba_pipeline.loader import load_graph


def main() -> None:
    full_yaml = ROOT / "data/LiFEMem.yaml"
    dialogs_yaml = ROOT / "data/dba_eval/lifemem_subset_dialogs.yaml"
    dba_graph_yaml = ROOT / "eval/outputs/dba_graph_subset_b6.yaml"
    eval_json = ROOT / "eval/outputs/dba_extraction_subset_b6.json"

    with open(dialogs_yaml, "r", encoding="utf-8") as f:
        dlg = yaml.safe_load(f)
    gold_node_ids = set(dlg["subset"]["gold_nodes"])

    gold = load_graph(str(full_yaml))
    dba = load_graph(str(dba_graph_yaml))
    ev = json.loads(eval_json.read_text(encoding="utf-8"))["eval"]

    # dba→gold 映射（仅已匹配）
    d2g = {d["id"]: d["matched_gold_id"] for d in ev["dba_node_detail"] if d.get("matched_gold_id")}
    g2d = {g: d for d, g in d2g.items()}

    # gold 子集内部边
    gold_edges = []
    for u, v, data in gold.graph.edges(data=True):
        if u in gold_node_ids and v in gold_node_ids:
            gold_edges.append((u, v, str(data.get("rel_type", ""))))

    # DBA 边（无向集合）
    dba_edges = set()
    for a, b in dba.graph.edges():
        dba_edges.add((a, b))
        dba_edges.add((b, a))

    a_ok, b_missing_edge, c_missing_node = [], [], []
    for u, v, rt in gold_edges:
        u_in, v_in = u in g2d, v in g2d
        if u_in and v_in:
            if (g2d[u], g2d[v]) in dba_edges:
                a_ok.append((u, v, rt))
            else:
                b_missing_edge.append((u, v, rt, g2d[u], g2d[v]))
        else:
            c_missing_node.append((u, v, rt,
                                   None if u_in else u, None if v_in else v))

    total = len(gold_edges)
    print(f"gold 子集内部边: {total}")
    print(f"A 正确连接        : {len(a_ok):3d} ({len(a_ok)/total:.1%})")
    print(f"B 两端都有未连    : {len(b_missing_edge):3d} ({len(b_missing_edge)/total:.1%})")
    print(f"C 一端节点缺失    : {len(c_missing_node):3d} ({len(c_missing_node)/total:.1%})")
    print(f"（DBA 图总边 {len(dba.graph.edges())} 条；dba→gold 映射 {len(d2g)} 条）")

    print("\n--- B 全量（两端节点都抽到了，但 LLM 未连）---")
    for u, v, rt, du, dv in b_missing_edge:
        nu = gold.get_node(u); nv = gold.get_node(v)
        du_node = dba.get_node(du); dv_node = dba.get_node(dv)
        print(f"  {u}--[{rt}]-->{v}  ({du},{dv})")
        print(f"    gold: {nu['content']}  |  {nv['content']}")
        print(f"    dba : {du_node['content']}  |  {dv_node['content']}")

    print("\n--- C 样例（一端节点缺失）---")
    for u, v, rt, mu, mv in c_missing_node[:8]:
        nu = gold.get_node(u); nv = gold.get_node(v)
        miss = mu if mu is not None else mv
        print(f"  {u}--[{rt}]-->{v}  缺失端:{miss}  {nu['content'][:18]} → {nv['content'][:18]}")


if __name__ == "__main__":
    main()
