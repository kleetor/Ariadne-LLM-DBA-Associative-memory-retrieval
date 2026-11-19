# -*- coding: utf-8 -*-
"""
DBA 抽取图（激进 v4）vs gold 子图：StoryRank 故事化 + 五维盲评对比

流程：两图各跑 P 检索 → build_path（nodes+edges）→ StoryRank 故事化
  → DeepSeek 五维盲评（隐藏来源系统）→ 输出故事与分数对比。

用法:
    python -m eval.run_dba_story_cmp
    python -m eval.run_dba_story_cmp --dba-graph eval/outputs/dba_graph_subset_b6_v4.yaml
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
from eval.run_p_baseline import build_retriever, _load_dotenv, build_path
from eval.judge_story import judge_story


def extract_subgraph_yaml(full_yaml: Path, node_ids: set, out_path: Path) -> Path:
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


def _mean_scores(sc: dict) -> float:
    dims = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter"]
    vals = [sc.get(d, 0) for d in dims]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 图 vs gold 图 StoryRank 对比")
    parser.add_argument("--dba-graph", default="eval/outputs/dba_graph_subset_b6_v4.yaml")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_subset_dialogs.yaml")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml")
    parser.add_argument("--queries", default="data/LiFEMem_queries_v3.yaml")
    parser.add_argument("--out", default="eval/outputs/dba_story_cmp.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip-judge", action="store_true", help="只故事化，不做五维盲评")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dlg = yaml.safe_load(f)
    gold_node_ids = set(dlg["subset"]["gold_nodes"])

    sub_yaml = ROOT / "eval/outputs/_gold_subgraph.yaml"
    extract_subgraph_yaml(ROOT / args.full_yaml, gold_node_ids, sub_yaml)

    print("[STORY] 构建 gold 子图 retriever...")
    _, retriever_gold = build_retriever(str(sub_yaml), llm_model=None)
    print("[STORY] 构建 DBA 图 retriever...")
    _, retriever_dba = build_retriever(str(ROOT / args.dba_graph), llm_model=None)
    llm = retriever_dba.llm

    with open(ROOT / args.queries, "r", encoding="utf-8") as f:
        qs = yaml.safe_load(f)
    queries = qs["queries"][: args.limit]

    results = []
    tot_g = {"len": 0, "score": 0.0, "n": 0}
    tot_d = {"len": 0, "score": 0.0, "n": 0}
    for q in queries:
        qid = q["id"]
        text = q["text"]

        def _run(retriever):
            res = retriever.retrieve(text)
            path = build_path(retriever, res)
            out = retriever.inference.story_rank(text, path)
            return res, path, out.get("story", "")

        res_g, path_g, story_g = _run(retriever_gold)
        res_d, path_d, story_d = _run(retriever_dba)

        row = {
            "query_id": qid,
            "category": q.get("scenario", ""),
            "query": text,
            "gold": {"nodes": len(path_g["nodes"]), "edges": len(path_g["edges"]),
                     "story": story_g},
            "dba": {"nodes": len(path_d["nodes"]), "edges": len(path_d["edges"]),
                    "story": story_d},
        }
        if not args.skip_judge:
            purpose = str(res_g.get("purpose", ""))
            sc_g = judge_story(llm, text, story_g, purpose=purpose)
            sc_d = judge_story(llm, text, story_d, purpose=purpose)
            row["gold"]["scores"] = sc_g
            row["dba"]["scores"] = sc_d
            tg, td = _mean_scores(sc_g), _mean_scores(sc_d)
            tot_g["score"] += tg
            tot_d["score"] += td
            tot_g["n"] += 1
            tot_d["n"] += 1
            print(f"[STORY] {qid} gold总分={tg} dba总分={td}")
        tot_g["len"] += len(story_g)
        tot_d["len"] += len(story_d)
        results.append(row)
        print(f"[STORY] {qid} ({q.get('scenario','')}) gold: {len(path_g['nodes'])}n/{len(path_g['edges'])}e {len(story_g)}字 | "
              f"dba: {len(path_d['nodes'])}n/{len(path_d['edges'])}e {len(story_d)}字")

    summary = {
        "avg_story_len_gold": round(tot_g["len"] / len(results), 1),
        "avg_story_len_dba": round(tot_d["len"] / len(results), 1),
    }
    if tot_g["n"]:
        summary["avg_total_gold"] = round(tot_g["score"] / tot_g["n"], 3)
        summary["avg_total_dba"] = round(tot_d["score"] / tot_d["n"], 3)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n===== StoryRank 对比：gold 子图 vs DBA 图(激进) =====")
    print(f"平均故事长度: gold={summary['avg_story_len_gold']}字  dba={summary['avg_story_len_dba']}字")
    if "avg_total_gold" in summary:
        print(f"五维总分均值: gold={summary['avg_total_gold']}  dba={summary['avg_total_dba']}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
