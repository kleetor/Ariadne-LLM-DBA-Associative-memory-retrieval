# -*- coding: utf-8 -*-
"""
分批抽取稳定性验证（多版本鲁棒性）

复用 run1（eval/outputs/dba_agent_user_extraction.json）作为基准，再补跑 2 次「分批（逐批）抽取」，
共 3 次独立 LLM 抽取，量化节点/边/主体/时序/召回的波动（均值 + 极差），用于判断抽取是否稳定。

用法:
    python -m eval.run_dba_agent_user_stability
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import yaml

from eval.run_dba_agent_user_eval import fresh
from eval.run_p_baseline import _load_dotenv
from eval.run_dba_extraction_eval import match_nodes
from dba_pipeline.core.jump_axis import RelationType

METRICS = [
    ("nodes", "节点总数"), ("edges", "边总数"), ("active", "活跃节点"),
    ("ai", "助手节点"), ("time", "时间锚节点"), ("temporal", "TEMPORAL边"),
    ("recall_user", "用户召回"), ("recall_ai", "助手召回"), ("recall_time", "时间锚召回"),
]


def run_once(dialogs, gold_base, gold_user, gold_ai, gold_time):
    g, dba, embeddings = fresh()
    embed_fn = lambda text: np.array(embeddings.embed_query(text))  # noqa: E731

    for d in dialogs:
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in d["turns"])
        dba.maintain(conversation)

    def recall(gold_ids, dba_list):
        if not gold_ids or not dba_list:
            return None
        gl = [{"id": nid, "content": gold_base[nid]["content"], "type": gold_base[nid]["type"]} for nid in gold_ids]
        m = match_nodes(gl, dba_list, embed_fn, 0.75)
        return round(len(m) / len(gl), 4)

    active = []
    for nid in g.graph.nodes():
        n = g.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            active.append({"id": nid, "content": n["content"],
                           "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})
    ai = [n for n in active if n["content"].startswith("助手")]
    usr = [n for n in active if n["content"].startswith("用户")]
    tm = [n for n in active if n["type"] == "thing"]
    temporal = sum(1 for _, _, ed in g.graph.edges(data=True) if ed.get("rel_type") == RelationType.TEMPORAL)

    return {
        "nodes": g.node_count, "edges": g.edge_count, "active": len(active),
        "ai": len(ai), "time": len(tm), "temporal": temporal,
        "recall_user": recall(gold_user, usr), "recall_ai": recall(gold_ai, ai),
        "recall_time": recall(gold_time, tm),
    }


def main() -> None:
    _load_dotenv(ROOT / ".env")

    with open(ROOT / "data/dba_eval/agent_user_dialogs.yaml", "r", encoding="utf-8") as f:
        dialogs = yaml.safe_load(f)["dialogs"]
    with open(ROOT / "data/dba_eval/agent_user_gold.yaml", "r", encoding="utf-8") as f:
        gold_data = yaml.safe_load(f)
    gold_base = {n["id"]: {"content": n["content"], "type": n["type"]} for n in gold_data["nodes"]}
    gold_ai = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("a")]
    gold_user = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("u")]
    gold_time = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("t")]

    # run1 复用已有结果
    with open(ROOT / "eval/outputs/dba_agent_user_extraction.json", "r", encoding="utf-8") as f:
        j = json.load(f)
    run1 = {
        "nodes": j["stats"]["nodes"], "edges": j["stats"]["edges"], "active": j["stats"]["active"],
        "ai": j["assert"]["mixed_memory"]["ai_nodes_in_dba"],
        "time": j["assert"]["temporal"]["time_nodes_in_dba"],
        "temporal": j["assert"]["temporal"]["temporal_edges"],
        "recall_user": j["recall"]["user"], "recall_ai": j["recall"]["assistant"],
        "recall_time": j["recall"]["time_anchor"],
    }
    runs = [("run1", run1)]
    print("[run1] 复用已有（逐批20轮maintain）")

    for i in (2, 3):
        r = run_once(dialogs, gold_base, gold_user, gold_ai, gold_time)
        runs.append((f"run{i}", r))
        print(f"[run{i}] 完成")

    # 汇总
    print("\n===== 分批抽取稳定性（3 次独立抽取）=====")
    header = f"{'指标':<12}" + "".join(f"{name:<16}" for name, _ in runs) + f"{'均值':<10}{'极差':<10}"
    print(header)
    summary = {}
    for key, lbl in METRICS:
        vals = [r[key] for _, r in runs if r.get(key, -1) is not None]
        mean = round(sum(vals) / len(vals), 4) if vals else "-"
        rng = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0
        summary[key] = {"runs": [r.get(key) for _, r in runs], "mean": mean, "range": rng}
        row = f"{lbl:<12}" + "".join(f"{str(r.get(key, '-')):<16}" for _, r in runs) + f"{str(mean):<10}{str(rng):<10}"
        print(row)

    out_path = ROOT / "eval/outputs/dba_agent_user_stability.json"
    out_path.write_text(json.dumps({
        "meta": {"gold_nodes": 96, "runs": [name for name, _ in runs], "note": "分批=逐批maintain"},
        "metrics": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n稳定性结果已写: {out_path}")


if __name__ == "__main__":
    main()
