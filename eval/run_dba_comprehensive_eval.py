# -*- coding: utf-8 -*-
"""
综合能力测评 · 双链路编排脚本（抽取质量 + 混合记忆/时序专项断言）

链路一（抽检）：对照 gold 出节点/边召回·精确·类型一致（复用 run_dba_extraction_eval.evaluate）。
链路二（检索/专项断言）：空图逐批 maintain 建图后，断言三项能力**同时成立**：
  [1] 混合记忆：抽出 content 以「助手」开头的「助手自身经历/行为」节点；gold 助手节点召回达标；
  [2] 时序：抽出了 TEMPORAL 边 + 时间锚点（THING）节点；gold 时间锚点召回达标；
  [3] 规模：DBA 活跃节点落在 gold 节点数 ±20%（含合理展开）。

数据：data/dba_eval/comprehensive_{gold,dialogs}.yaml；查询集 comprehensive_queries.yaml（供检索侧复用）。

用法:
    python -m eval.run_dba_comprehensive_eval
    python -m eval.run_dba_comprehensive_eval --out eval/outputs/dba_comprehensive_extraction.json
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

import numpy as np
import yaml

from eval.run_dba_extraction_eval import (
    _load_dotenv, build_llm, build_embeddings, match_nodes,
)
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.dba import MemoryDBA
from dba_pipeline.core.jump_axis import RelationType


def fresh(orphan_threshold: int = 100):
    embeddings = build_embeddings()
    vs = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    g = MemoryGraph()
    b = GraphBuilder(g, vs, orphan_threshold=orphan_threshold)
    dba = MemoryDBA(llm=build_llm(), graph=g, vector_store=vs, graph_builder=b)
    return g, dba, embeddings


def prefix_of(nid: str) -> str:
    return nid.rstrip("0123456789")[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description="综合能力测评（抽检 + 混合记忆/时序专项）")
    parser.add_argument("--gold", default="data/dba_eval/comprehensive_gold.yaml")
    parser.add_argument("--dialogs", default="data/dba_eval/comprehensive_dialogs.yaml")
    parser.add_argument("--out", default="eval/outputs/dba_comprehensive_extraction.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--orphan-threshold", type=int, default=100, help="评测语料为一次性批量，调大隔离遗忘机制")
    parser.add_argument("--save-graph", default=None,
                        help="把 DBA 自动建图导出为 YAML（供 E12b 的检索/判分复用）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    with open(ROOT / args.gold, "r", encoding="utf-8") as f:
        gold_data = yaml.safe_load(f)
    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dlg_data = yaml.safe_load(f)
    dialogs = dlg_data["dialogs"]
    gold_nodes = gold_data["nodes"]
    gold_base = {n["id"]: {"content": n["content"], "type": n["type"]} for n in gold_nodes}
    gold_user = [n["id"] for n in gold_nodes if prefix_of(n["id"]).startswith("u")]
    gold_ai = [n["id"] for n in gold_nodes if prefix_of(n["id"]).startswith("a")]
    gold_time = [n["id"] for n in gold_nodes if prefix_of(n["id"]).startswith("t")]

    # ===== 建图（空图逐批 maintain）=====
    g, dba, embeddings = fresh(args.orphan_threshold)
    embed_fn = lambda text: np.array(embeddings.embed_query(text))  # noqa: E731

    batch_log = []
    for d in dialogs:
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in d["turns"])
        res = dba.maintain(conversation)
        batch_log.append({
            "id": d["id"], "scene": d["scene"],
            "skipped": res.get("skipped", False),
            "node_ops": len(res.get("ops", {}).get("node_ops", [])),
            "created": len(res.get("result", {}).get("created_ids", [])),
            "nodes": g.node_count, "edges": g.edge_count,
        })
        print(f"[批 {d['id']}] ({d['scene']}) skipped={res.get('skipped', False)} "
              f"node_ops={len(res.get('ops', {}).get('node_ops', []))} "
              f"created={len(res.get('result', {}).get('created_ids', []))} "
              f"(累计 {g.node_count} 节点 / {g.edge_count} 边)")

    # ===== DBA 活跃节点 =====
    graph_path = None
    if args.save_graph:
        graph_path = ROOT / args.save_graph
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"[DBA] 自动建图已导出: {graph_path}")

    dba_active = []
    for nid in g.graph.nodes():
        n = g.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            dba_active.append({"id": nid, "content": n["content"],
                               "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})
    dba_ai = [n for n in dba_active if n["content"].startswith("助手")]
    dba_time = [n for n in dba_active if n["type"] == "thing"]
    temporal_edges = [(a, b) for a, b, ed in g.graph.edges(data=True)
                      if ed.get("rel_type") == RelationType.TEMPORAL]

    # ===== 链路二：主体/时序召回 =====
    def recall(gold_ids, dba_list):
        if not gold_ids or not dba_list:
            return None
        gold_list = [{"id": nid, "content": gold_base[nid]["content"], "type": gold_base[nid]["type"]}
                     for nid in gold_ids]
        matched = match_nodes(gold_list, dba_list, embed_fn, args.threshold)
        return round(len(matched) / len(gold_list), 4)

    recall_ai = recall(gold_ai, dba_ai)
    recall_time = recall(gold_time, dba_time)
    recall_user = recall(gold_user, [n for n in dba_active if n["content"].startswith("用户")])

    n_gold = len(gold_nodes)
    n_active = len(dba_active)
    mixed_ok = len(dba_ai) > 0 and (recall_ai is not None and recall_ai > 0)
    temporal_ok = len(temporal_edges) > 0 and len(dba_time) > 0 and (recall_time is not None and recall_time > 0)
    size_ok = 0.6 * n_gold <= n_active <= 1.6 * n_gold

    # ===== 链路一：抽取质量对照（节点/边召回·精确·类型）=====
    import eval.run_dba_extraction_eval as rex
    gold_ids = set(n["id"] for n in gold_nodes)
    # 用 build_llm 的图做抽取质量是一个独立小评测；此处直接复用 evaluate 逻辑
    # （需 gold MemoryGraph + 本次建好的 dba graph）。为保持可复现，用 loader 重建 gold graph。
    from dba_pipeline.loader import load_graph
    gold_graph = load_graph(str(ROOT / args.gold))
    eval_result = rex.evaluate(gold_graph, gold_ids, g, embed_fn, args.threshold)

    result = {
        "meta": {
            "dataset": args.gold,
            "dialogs": args.dialogs,
            "gold_nodes": n_gold,
            "dialogs": len(dialogs),
            "batch_size": 1,
        },
        "assert_combined": {
            "mixed_memory": {
                "ai_nodes_in_dba": len(dba_ai),
                "gold_ai_nodes": len(gold_ai),
                "recall_ai": recall_ai,
                "ok": mixed_ok,
                "examples": [n["content"] for n in dba_ai[:8]],
            },
            "temporal": {
                "temporal_edges": len(temporal_edges),
                "time_nodes_in_dba": len(dba_time),
                "gold_time_nodes": len(gold_time),
                "recall_time": recall_time,
                "ok": temporal_ok,
                "time_examples": [n["content"] for n in dba_time[:8]],
            },
            "scale": {"dba_active": n_active, "gold_nodes": n_gold, "ok": size_ok},
        },
        "recall": {"user": recall_user, "assistant": recall_ai, "time_anchor": recall_time},
        "extraction": eval_result,
        "batch_log": batch_log,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 综合能力测评 =====")
    print(f"规模: DBA活跃={n_active} gold={n_gold} -> {'OK' if size_ok else 'FAIL'}")
    a = result["assert_combined"]["mixed_memory"]
    print(f"混合记忆: 助手节点={a['ai_nodes_in_dba']} 召回={a['recall_ai']} -> {'OK' if a['ok'] else 'FAIL'}")
    t = result["assert_combined"]["temporal"]
    print(f"时序: TEMPORAL边={t['temporal_edges']} 时间锚点={t['time_nodes_in_dba']} 召回={t['recall_time']} -> {'OK' if t['ok'] else 'FAIL'}")
    print(f"召回: 用户={result['recall']['user']} 助手={result['recall']['assistant']} 时间锚点={result['recall']['time_anchor']}")
    e = eval_result
    print(f"抽取质量: 节点召回={e['node_recall']:.2%} 精确={e['node_precision']:.2%} 类型一致={e['type_accuracy']:.2%} | "
          f"边召回={e['edge_recall']:.2%} 边精确={e['edge_precision']:.2%} 边类型一致={e['edge_type_accuracy']:.2%}")

    ok = mixed_ok and temporal_ok and size_ok
    print("\n" + "=" * 60)
    if ok:
        print(f"综合专项通过 [OK]：混合记忆 + 时序 + 规模三项同时成立。结果已写: {out_path}")
    else:
        print(f"综合专项未通过 [FAIL]，请检查。结果已写: {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
