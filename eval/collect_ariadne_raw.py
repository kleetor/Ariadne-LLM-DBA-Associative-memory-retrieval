# -*- coding: utf-8 -*-
"""
e2e 补测 · Ariadne「无 StoryRank」上下文块生成（StoryRank 消融）。

背景：原 e2e 里 Ariadne 的块是 retrieve_with_story 产出的 {story}（已 StoryRank）。
本脚本把 Ariadne 的块换成 **PAR 原始检索路径**（nodes+edges，未叙事化），用于对比
「StoryRank 故事化」对 judge 评分的影响。

- Ariadne 块（无 StoryRank） = {nodes: [node content...], edges: ["from --rel--> to", ...]}
  （镜像 Graphiti 的 {entities, relations} 结构，都是"原始结构"，未被故事化）
- 数据来源：eval/outputs/p_candidates_comprehensive.json 的 path（就是 PAR 检索的因果链，与 story 同源）
- Mem0 / Graphiti 块沿用 context_comprehensive_3sys.json（不变）
- 输出：eval/outputs/context_comprehensive_ariadne_raw.json

用法:
    python -m eval.collect_ariadne_raw
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CTX = "eval/outputs/context_comprehensive_3sys.json"       # 原上下文（Ariadne=story + mem0 + graphiti）
P_CAND = "eval/outputs/p_candidates_comprehensive.json"    # Ariadne raw 候选/路径
OUT = "eval/outputs/context_comprehensive_ariadne_raw.json"


def ariadne_raw_block(path: dict) -> dict:
    """PAR 路径 → 与 StoryRank 输入完全同构的原始块（带 node_type + 方向已应用 is_reverse）。

    渲染方式与 inference.story_rank 完全一致（[id]（node_type）content / [from] --rel--> [to]），
    使「raw」与「story」只差"是否被故事化"，不含信息残缺（此前丢了 node_type、没处理 is_reverse）。
    """
    nodes = []
    seen_n = set()
    for n in path.get("nodes", []):
        if isinstance(n, dict):
            nid = n.get("id", "")
            content = n.get("content", "")
            nt = n.get("node_type", "")
            render = f"[{nid}]（{nt}）{content}"
        else:
            render = str(n)
        if render and render not in seen_n:
            seen_n.add(render)
            nodes.append(render)

    edges = []
    for e in path.get("edges", []):
        if isinstance(e, dict):
            rel = e.get("rel_type", "?")
            if e.get("is_reverse"):
                render = f"[{e.get('to', '?')}] --{rel}--> [{e.get('from', '?')}]"
            else:
                render = f"[{e.get('from', '?')}] --{rel}--> [{e.get('to', '?')}]"
        else:
            render = str(e)
        edges.append(render)
    return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}


def main() -> None:
    with open(ROOT / CTX, encoding="utf-8") as f:
        ctx = json.load(f)
    with open(ROOT / P_CAND, encoding="utf-8") as f:
        pdata = json.load(f)
    p_by_qid = {r["query_id"]: r for r in pdata["results"]}

    out = {"meta": {**ctx.get("meta", {}),
                    "note": "Ariadne block = PAR raw path (no StoryRank); mem0/graphiti unchanged"},
           "queries": []}
    miss = []
    for r in ctx["queries"]:
        qid = r["query_id"]
        pr = p_by_qid.get(qid)
        if pr is None:
            miss.append(qid)
            raw = {"nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0}
        else:
            raw = ariadne_raw_block(pr.get("path", {}))
        systems = {k: v for k, v in r.get("systems", {}).items()}
        systems["ariadne"] = raw  # 换成无 StoryRank 的 raw 块
        out["queries"].append({"query_id": qid, "category": r.get("category", ""),
                               "query": r.get("query", ""), "systems": systems})

    out_path = ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] 无 StoryRank Ariadne 上下文块已写: {out_path}（{len(out['queries'])} 条）")
    if miss:
        print(f"[collect] 警告：以下 qid 在 P 候选中缺失（raw 块为空）: {miss}")


if __name__ == "__main__":
    main()
