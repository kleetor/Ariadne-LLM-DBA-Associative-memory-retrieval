# -*- coding: utf-8 -*-
"""
查询候选内容对比（离线）：从检索联动 JSON + 图 YAML 读取，输出每条查询
gold 子图 vs DBA 图的候选节点内容，便于直观理解命中/未命中。

用法:
    python -m eval.show_query_cand_cmp --cmp eval/outputs/dba_retrieval_cmp_b6_v2.json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="查询候选内容对比（离线）")
    parser.add_argument("--cmp", default="eval/outputs/dba_retrieval_cmp_b6_v2.json",
                        help="检索联动结果 JSON")
    parser.add_argument("--dba-graph", default="eval/outputs/dba_graph_subset_b6_v2.yaml",
                        help="DBA 图 YAML")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml", help="gold 全图 YAML")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_subset_dialogs.yaml",
                        help="对话语料（取子集节点）")
    parser.add_argument("--only", default="", help="只看指定查询，逗号分隔（默认全部）")
    args = parser.parse_args()

    dba = load_graph(str(ROOT / args.dba_graph))
    gold = load_graph(str(ROOT / args.full_yaml))
    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dlg = yaml.safe_load(f)
    gold_node_ids = set(dlg["subset"]["gold_nodes"])

    def content(g, nid):
        n = g.get_node(nid)
        if not n:
            return "???"
        t = n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])
        return f"[{nid} {t}] {n['content']}"

    data = json.loads((ROOT / args.cmp).read_text(encoding="utf-8"))
    only = {q.strip() for q in args.only.split(",") if q.strip()}

    for row in data["queries"]:
        qid = row["query_id"]
        if only and qid not in only:
            continue
        g = row["gold"]
        d = row["dba"]
        print(f"\n===== {qid} · {row['category']} =====")
        print(f"  {row['text']}")
        print(f"  expected(子集内): {row['expected_eff']}")
        print(f"  R@all gold={g['r_at_all']} dba={d['r_at_all']}")
        print("  --- gold 候选 (前10) ---")
        for nid in g["cand_ids"][:10]:
            mark = " <== HIT" if nid in g["hit_ids"] else ""
            print(f"    {content(gold, nid)}{mark}")
        print("  --- dba 候选 (前10) ---")
        for nid in d["cand_ids"][:10]:
            mark = " <== HIT" if nid in d["hit_ids"] else ""
            print(f"    {content(dba, nid)}{mark}")


if __name__ == "__main__":
    main()
