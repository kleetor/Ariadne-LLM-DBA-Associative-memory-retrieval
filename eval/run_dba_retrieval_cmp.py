# -*- coding: utf-8 -*-
"""
DBA 抽取评测 · 检索联动（DBA 图 vs gold 子图）

两图分别跑 LiFEMem v3 35 条查询（P 检索，purpose 推断），对比：
  - R@all：expected ∩ 子集的召回（gold 子图候选直接用 gold id；
           DBA 图候选 id 经 dba→gold 匹配映射还原后比对，未匹配候选不计命中）
  - 每查询候选量、候选覆盖
只对比"子集内 expected"——子集外 expected 两图均无法覆盖，公平起见排除。

用法:
    python -m eval.run_dba_retrieval_cmp
    python -m eval.run_dba_retrieval_cmp --out eval/outputs/dba_retrieval_cmp.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph
from eval.run_p_baseline import build_retriever, _load_dotenv


def extract_subgraph_yaml(full_yaml: Path, node_ids: set, out_path: Path) -> Path:
    """从全图提取子图并导出 yaml（gold 子图基准）"""
    g = load_graph(str(full_yaml))
    sub = MemoryGraph()
    for nid in node_ids:
        n = g.get_node(nid)
        if n:
            sub.add_memory(nid, n["content"], n["node_type"],
                           n.get("deprecated", False), n.get("forgotten", False))
    for u, v, data in g.graph.edges(data=True):
        if u in node_ids and v in node_ids:
            sub.add_edge(u, v, data["rel_type"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(sub.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path


def load_dba_to_gold_map(eval_json: Path) -> dict:
    """从抽取评测 JSON 读 dba_node_detail，构建 {dba_id: gold_id}（仅已匹配）"""
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    mapping = {}
    for d in data["eval"].get("dba_node_detail", []):
        if d.get("matched_gold_id"):
            mapping[d["id"]] = d["matched_gold_id"]
    return mapping


def per_query(expected_eff: list, cand_ids: list, id_map: dict):
    """R@all per query。id_map 非空时先把候选 id 映射到 gold id"""
    if not expected_eff:
        return None
    if id_map:
        cand_gold = {id_map.get(c) for c in cand_ids}
        cand_gold.discard(None)
    else:
        cand_gold = set(cand_ids)
    hit = len(set(expected_eff) & cand_gold)
    return round(hit / len(expected_eff), 4), hit, len(expected_eff), sorted(set(expected_eff) & cand_gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 抽取评测 · 检索联动")
    parser.add_argument("--dba-graph", default="eval/outputs/dba_graph_subset.yaml", help="DBA 建图 YAML")
    parser.add_argument("--eval-json", default="eval/outputs/dba_extraction_subset.json", help="抽取评测结果 JSON")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_subset_dialogs.yaml", help="对话语料 YAML（取子集节点）")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml", help="gold 全图 YAML")
    parser.add_argument("--queries", default="data/LiFEMem_queries_v3.yaml", help="查询集 YAML")
    parser.add_argument("--out", default="eval/outputs/dba_retrieval_cmp.json", help="输出 JSON 路径")
    parser.add_argument("--limit", type=int, default=10, help="只跑前 N 条查询（快速验证）")
    parser.add_argument("--mix-gold-edges", action="store_true",
                        help="混合图模式：DBA 抽取节点 + gold 边结构（隔离验证边是否为检索瓶颈）")
    parser.add_argument("--purposes", default=None, help="手写目的注入 YAML（{qid: [目的]}，排除目的推断噪声）")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    # 子集节点
    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dlg = yaml.safe_load(f)
    gold_node_ids = set(dlg["subset"]["gold_nodes"])

    # gold 子图（临时导出）
    sub_yaml = ROOT / "eval/outputs/_gold_subgraph.yaml"
    extract_subgraph_yaml(ROOT / args.full_yaml, gold_node_ids, sub_yaml)

    # 两图 retriever
    print("[CMP] 构建 gold 子图 retriever...")
    _, retriever_gold = build_retriever(str(sub_yaml), llm_model=args.llm_model)

    dba_to_gold = load_dba_to_gold_map(ROOT / args.eval_json)
    print(f"[CMP] dba→gold 映射 {len(dba_to_gold)} 条")

    # 混合图模式：DBA 抽取的节点 + gold 边结构（验证"边是否为检索瓶颈"）
    dba_graph_path = ROOT / args.dba_graph
    if args.mix_gold_edges:
        dg = load_graph(str(dba_graph_path))
        gold_sub = load_graph(str(sub_yaml))
        g2d = {g: d for d, g in dba_to_gold.items()}
        added = 0
        for u, v, data in gold_sub.graph.edges(data=True):
            if u in g2d and v in g2d and not dg.graph.has_edge(g2d[u], g2d[v]):
                dg.add_edge(g2d[u], g2d[v], data["rel_type"])
                added += 1
        mix_yaml = ROOT / "eval/outputs/_dba_mixed_goldedges.yaml"
        mix_yaml.write_text(yaml.safe_dump(dg.to_dict(), allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
        dba_graph_path = mix_yaml
        print(f"[CMP] 混合图：向 DBA 图补充 gold 边 {added} 条")

    print("[CMP] 构建 DBA 图 retriever...")
    _, retriever_dba = build_retriever(str(dba_graph_path), llm_model=args.llm_model)

    # 查询
    with open(ROOT / args.queries, "r", encoding="utf-8") as f:
        qs = yaml.safe_load(f)
    queries = qs["queries"][: args.limit]
    print(f"[CMP] 查询数: {len(queries)}（--limit {args.limit}）")

    # 手写目的注入（可选）：排除目的推断噪声
    inject_purposes = {}
    if args.purposes:
        with open(ROOT / args.purposes, "r", encoding="utf-8") as f:
            inject_purposes = yaml.safe_load(f)
        print(f"[CMP] purpose-mode=inject（手写目的 {len(inject_purposes)} 条）")

    rows = []
    tot = {"gold": [0, 0], "dba": [0, 0]}  # [hit, total]
    for q in queries:
        qid = q["id"]
        expected = q.get("expected", []) or []
        expected_eff = [e for e in expected if e in gold_node_ids]
        purpose = inject_purposes.get(qid) if args.purposes else None

        res_gold = retriever_gold.retrieve(q["text"], purpose=purpose)
        res_dba = retriever_dba.retrieve(q["text"], purpose=purpose)
        cand_gold = [mid for mid, _ in res_gold.get("peak_memories", [])]
        cand_dba = [mid for mid, _ in res_dba.get("peak_memories", [])]

        rg = per_query(expected_eff, cand_gold, None)
        rd = per_query(expected_eff, cand_dba, dba_to_gold)
        if rg is not None:
            tot["gold"][0] += rg[1]
            tot["gold"][1] += rg[2]
            tot["dba"][0] += rd[1]
            tot["dba"][1] += rd[2]

        rows.append({
            "query_id": qid,
            "category": q.get("scenario", ""),
            "text": q["text"][:40],
            "expected_eff": expected_eff,
            "gold": {"r_at_all": rg[0] if rg else None, "hit": rg[1] if rg else None,
                     "cand": len(cand_gold), "hit_ids": rg[3] if rg else [], "cand_ids": cand_gold},
            "dba": {"r_at_all": rd[0] if rd else None, "hit": rd[1] if rd else None,
                    "cand": len(cand_dba), "hit_ids": rd[3] if rd else [], "cand_ids": cand_dba},
        })
        print(f"[CMP] {qid} ({q.get('scenario','')}) gold={rg[0] if rg else '-':<6} dba={rd[0] if rd else '-':<6} "
              f"cand gold/dba={len(cand_gold)}/{len(cand_dba)}")

    r_gold = tot["gold"][0] / tot["gold"][1] if tot["gold"][1] else 0.0
    r_dba = tot["dba"][0] / tot["dba"][1] if tot["dba"][1] else 0.0
    summary = {
        "R_at_all_gold": round(r_gold, 4),
        "R_at_all_dba": round(r_dba, 4),
        "expected_effective_total": tot["gold"][1],
        "avg_cand_gold": round(sum(r["gold"]["cand"] for r in rows) / len(rows), 1),
        "avg_cand_dba": round(sum(r["dba"]["cand"] for r in rows) / len(rows), 1),
    }

    out = {"meta": {"dba_graph": args.dba_graph, "gold_subgraph_nodes": len(gold_node_ids),
                    "mix_gold_edges": args.mix_gold_edges},
           "summary": summary, "queries": rows}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 检索联动：DBA 图 vs gold 子图 =====")
    print(f"R@all(子集内 expected): gold={r_gold:.2%}  dba={r_dba:.2%}")
    print(f"平均候选数: gold={summary['avg_cand_gold']}  dba={summary['avg_cand_dba']}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
