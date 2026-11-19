# -*- coding: utf-8 -*-
"""
DBA 第二轮边连接实验：对已建成的 DBA 全量图再做一次边连接（不抽节点），观察变化

输入 dba_graph_full.yaml（261 节点/529 边），以全量对话语料（批量 6 → 9 组）
再跑一遍 Step 2 边连接（edge_chain，激进 prompt），只执行 edge_ops：
- 统计每轮新增/删除边、失败操作
- 与 gold 边重合率（经抽取匹配映射）前后对比
- 导出第二轮图 dba_graph_full_relink.yaml

用法:
    python -m eval.run_dba_relink
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
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.dba import MemoryDBA
from dba_pipeline.loader import load_graph
from eval.run_dba_extraction_eval import build_llm, build_embeddings, _load_dotenv


def _load_gold_edges(full_yaml: Path, gold_node_ids: set) -> set:
    g = load_graph(str(full_yaml))
    gold_edges = set()
    for u, v, data in g.graph.edges(data=True):
        if u in gold_node_ids and v in gold_node_ids:
            gold_edges.add((u, v))
            gold_edges.add((v, u))
    return gold_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 第二轮边连接实验")
    parser.add_argument("--graph", default="eval/outputs/dba_graph_full.yaml")
    parser.add_argument("--eval-json", default="eval/outputs/dba_extraction_full.json")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_full_dialogs.yaml")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--out-graph", default="eval/outputs/dba_graph_full_relink.yaml")
    parser.add_argument("--out", default="eval/outputs/dba_relink_full.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    # 已有图
    graph = load_graph(ROOT / args.graph)
    init_edges = graph.edge_count
    print(f"[RELINK] 初始: {graph.node_count} 节点 / {init_edges} 边")

    # 向量索引（重建：图节点全量入索引）
    embeddings = build_embeddings()
    vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    all_nodes = [(nid, graph.get_node(nid)) for nid in graph.graph.nodes()]
    all_nodes = [(nid, n) for nid, n in all_nodes if n and not n.get("deprecated") and not n.get("forgotten")]
    vector_store.add_memories([nid for nid, _ in all_nodes], [n["content"] for _, n in all_nodes])
    print(f"[RELINK] 向量索引重建: {len(all_nodes)} 节点")

    builder = GraphBuilder(graph, vector_store, orphan_threshold=100)
    llm = build_llm()
    dba = MemoryDBA(llm=llm, graph=graph, vector_store=vector_store, graph_builder=builder)

    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dialogs = data["dialogs"]
    gold_node_ids = set(data["subset"]["gold_nodes"])
    gold_edges = _load_gold_edges(ROOT / args.full_yaml, gold_node_ids)

    # dba→gold 映射（抽取评测）
    d2g = {}
    ev = json.loads((ROOT / args.eval_json).read_text(encoding="utf-8"))["eval"]
    for d in ev.get("dba_node_detail", []):
        if d.get("matched_gold_id"):
            d2g[d["id"]] = d["matched_gold_id"]

    def gold_overlap(gr) -> float:
        if gr.edge_count == 0:
            return 0.0
        hit = 0
        for a, b in gr.graph.edges():
            if a in d2g and b in d2g and (d2g[a], d2g[b]) in gold_edges:
                hit += 1
        return round(hit / gr.edge_count, 4)

    groups = [dialogs[i:i + args.batch_size] for i in range(0, len(dialogs), args.batch_size)]
    round_log = []
    for gi, group in enumerate(groups):
        turns_all = [t for dialog in group for t in dialog["turns"]]
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in turns_all)

        # 只做边连接：不抽取节点（new_ids=[]），复用 Step 2 上下文
        edge_context = dba._build_edge_context(conversation, [])
        raw = dba.edge_chain.invoke(edge_context).content
        edge_ops = dba._parse_response(raw).get("edge_ops", [])
        result = builder.apply_ops([], edge_ops)

        created = sum(1 for op in edge_ops if op.get("action") == "create")
        deleted = sum(1 for op in edge_ops if op.get("action") == "delete")
        round_log.append({
            "round": gi + 1,
            "dialogs": [d["id"] for d in group],
            "edge_ops": len(edge_ops),
            "create_ops": created,
            "delete_ops": deleted,
            "errors": result.get("errors", []),
            "edges_total": graph.edge_count,
            "gold_overlap": gold_overlap(graph),
        })
        print(f"[RELINK] 第{gi + 1}轮（{', '.join(d['id'] for d in group)}）-> "
              f"create={created} delete={deleted} (累计 {graph.edge_count} 边, gold重合 {round_log[-1]['gold_overlap']:.1%})")

    # 结果
    out = {
        "meta": {"initial_nodes": len(all_nodes), "initial_edges": init_edges, "rounds": len(groups)},
        "round_log": round_log,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 导出第二轮图
    graph_path = ROOT / args.out_graph
    graph_path.write_text(yaml.safe_dump(graph.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n[RELINK] 导出: {graph_path}（{graph.node_count} 节点 / {graph.edge_count} 边）")

    final = round_log[-1]
    print(f"=== 第二轮边连接完成 ===")
    print(f"边: {final['edges_total']}（gold 重合 {final['gold_overlap']:.1%}）")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
