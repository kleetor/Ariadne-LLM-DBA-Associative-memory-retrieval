# -*- coding: utf-8 -*-
"""
检索命中节点差异分析：gold 图 vs DBA 图在 35 条查询上命中集合的差异

读 dba_retrieval_full.json（或任意 retrieval_cmp 输出）：
- 每条查询 gold.hit_ids / dba.hit_ids（均为还原为 gold id）
- 统计：命中集合一致数、G 特有 / D 特有命中节点及其场景分布

用法:
    python -m eval.analyze_retrieval_hits
    python -m eval.analyze_retrieval_hits --in eval/outputs/dba_retrieval_full.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _prefix(nid: str) -> str:
    for p in ["he", "so", "ln", "sh", "th"]:
        if nid.startswith(p):
            return p
    return nid[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="eval/outputs/dba_retrieval_full.json")
    args = parser.parse_args()

    data = json.loads((ROOT / args.inp).read_text(encoding="utf-8"))
    rows = [r for r in data["queries"] if r["gold"].get("r_at_all") is not None]

    same, diff_g, diff_d = 0, 0, 0
    g_only = Counter()       # gold 命中 dba 未命中（按场景前缀）
    d_only = Counter()       # dba 命中 gold 未命中
    g_only_detail, d_only_detail = [], []
    per_q = []

    for r in rows:
        gh = set(r["gold"].get("hit_ids", []))
        dh = set(r["dba"].get("hit_ids", []))
        if gh == dh:
            same += 1
        else:
            diff_g += 1 if gh - dh else 0
            diff_d += 1 if dh - gh else 0
        for nid in gh - dh:
            g_only[_prefix(nid)] += 1
            g_only_detail.append((r["query_id"], nid))
        for nid in dh - gh:
            d_only[_prefix(nid)] += 1
            d_only_detail.append((r["query_id"], nid))
        per_q.append((r["query_id"], r["category"], sorted(gh), sorted(dh)))

    print(f"有效查询: {len(rows)}")
    print(f"命中集合完全一致: {same} 条")
    print(f"gold 命中而 dba 未命中: {diff_g} 条查询受影响，共 {sum(g_only.values())} 个节点")
    print(f"dba 命中而 gold 未命中: {diff_d} 条查询受影响，共 {sum(d_only.values())} 个节点")

    print("\n=== G 特有（gold 命中 dba 未命中）场景分布 ===")
    for p, c in g_only.most_common():
        print(f"  {p}: {c}")
    print("示例:", [f"{q}:{n}" for q, n in g_only_detail[:15]])

    print("\n=== D 特有（dba 命中 gold 未命中）场景分布 ===")
    for p, c in d_only.most_common():
        print(f"  {p}: {c}")
    print("示例:", [f"{q}:{n}" for q, n in d_only_detail[:15]])

    print("\n=== 每条查询命中节点（G特有/D特有/一致）===")
    for qid, cat, gh, dh in per_q:
        gs, ds = set(gh), set(dh)
        mark = "==" if gs == ds else (f"G+{sorted(gs-ds)}" if gs - ds else f"D+{sorted(ds-gs)}")
        print(f"{qid} {cat:<16} {mark}  | gold={gh} dba={dh}")


if __name__ == "__main__":
    main()
