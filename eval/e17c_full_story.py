# -*- coding: utf-8 -*-
"""E17c：不做数值评测，直接跑完整 PAR（默认参数、故事化输出），看最终效果。

用 E17b 的图（跨人物属性边），一条桥接查询 + 两条对照查询；
不设 max_peak_nodes（默认不截断）、不调任何参数、不注入 purpose（走完整推断链）。

用法：python -m eval.e17c_full_story
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402
from eval.e17b_cross_person import NODES, EDGES, DISTRACT  # noqa: E402

QUERIES = [
    "B最近这么高兴，打算怎么犒劳自己",
    "A那次比赛收获如何",
    "那家新开的川菜馆怎么样",
]


def main() -> int:
    _load_dotenv(ROOT / ".env")
    nodes, edges = list(NODES), list(EDGES)
    for i in range(0, len(DISTRACT), 4):
        chain = DISTRACT[i:i + 4]
        for j, (pre, c, t) in enumerate(chain):
            nid = f"{pre}{i // 4 + 1}_{j + 1}"
            nodes.append((nid, c, t))
            if j:
                edges.append((f"{pre}{i // 4 + 1}_{j}", nid, "causal"))

    data = {"nodes": [{"id": i, "content": c, "type": t} for i, c, t in nodes],
            "edges": [{"from": a, "to": b, "type": r} for a, b, r in edges]}
    p = ROOT / "eval/outputs/e17c_full_story.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    graph, retriever = build_retriever(str(p), vector_cache_key="e17b")
    print(f"[E17c] nodes={graph.node_count}")

    for q in QUERIES:
        res = retriever.retrieve_with_story(q, seed_k=5, story=True, with_response=False)
        purpose = res.get("purpose", {})
        path = res.get("path") or {"nodes": [], "edges": []}
        ids = [n["id"] for n in path.get("nodes", [])]
        print("\n" + "=" * 78)
        print("QUERY:", q)
        print("PURPOSE:", purpose.get("purposes"))
        print(f"NODES ({len(ids)}):", ids)
        for nid in ids:
            nt = graph.get_node_type(nid)
            print(f"  [{nid}] ({str(nt.value if hasattr(nt,'value') else nt)}) "
                  f"{graph.get_content(nid)}")
        st = res.get("stories") or res.get("story") or "(empty)"
        if isinstance(st, list):
            st = "\n".join(s.get("story", str(s)) if isinstance(s, dict) else str(s) for s in st)
        print("\nSTORY:\n" + str(st))
        print("\nEDGES:", len(path.get("edges", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
