# -*- coding: utf-8 -*-
"""
用重标后的 DBA expected 重算 DBA R@all（从既有检索 JSON，不重跑检索）

对每条查询：
  gold 侧：r_at_all 保持原值（gold expected 不变）
  dba 侧 ：用 DBA expected 标注集 ∩ 既有 dba cand_ids 重算 R@all

统计口径与原脚本一致：只计入 gold expected_eff 非空的查询（子集对比的同一查询集）。

用法:
    python -m eval.recompute_dba_ratall --retrieval eval/outputs/dba_retrieval_cmp_b6_v4.json \
        --annotations eval/outputs/dba_expected_v4.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="重标后 DBA R@all 重算")
    parser.add_argument("--retrieval", required=True, help="既有检索对比 JSON")
    parser.add_argument("--annotations", required=True, help="DBA expected 重标注 JSON")
    args = parser.parse_args()

    ret = json.loads((ROOT / args.retrieval).read_text(encoding="utf-8"))
    ann = json.loads((ROOT / args.annotations).read_text(encoding="utf-8"))
    ann_map = ann["annotations"]

    gold_tot = [0, 0]   # [hit, total]
    dba_old_tot = [0, 0]
    dba_new_tot = [0, 0]
    n_queries = 0
    rows = []

    for q in ret["queries"]:
        qid = q["query_id"]
        gold_eff = q.get("expected_eff") or []
        if not gold_eff:
            continue  # 与原脚本口径一致：只统计 gold expected 在子集内的查询
        gold = q.get("gold", {})
        dba = q.get("dba", {})

        gold_tot[0] += gold.get("hit", 0)
        gold_tot[1] += len(gold_eff)

        old_rat = dba.get("r_at_all")
        if old_rat is not None:
            dba_old_tot[0] += dba.get("hit", 0)
            dba_old_tot[1] += len(gold_eff)

        dba_expected = ann_map.get(qid, [])
        dba_cand = set(dba.get("cand_ids", []))
        hit_new = len([i for i in dba_expected if i in dba_cand])
        if dba_expected:
            dba_new_tot[0] += hit_new
            dba_new_tot[1] += len(dba_expected)

        rows.append({
            "query_id": qid,
            "category": q.get("category", ""),
            "expected_eff_gold": gold_eff,
            "dba_expected": dba_expected,
            "gold_r_at_all": gold.get("r_at_all"),
            "dba_r_at_all_old": old_rat,
            "dba_r_at_all_new": round(hit_new / len(dba_expected), 4) if dba_expected else None,
            "dba_hit_new": hit_new,
            "dba_cand": dba.get("cand", 0),
        })
        n_queries += 1

    r_gold = gold_tot[0] / gold_tot[1] if gold_tot[1] else 0.0
    r_dba_old = dba_old_tot[0] / dba_old_tot[1] if dba_old_tot[1] else 0.0
    r_dba_new = dba_new_tot[0] / dba_new_tot[1] if dba_new_tot[1] else 0.0

    print(f"图: {ret['meta'].get('dba_graph')}")
    print(f"有效查询数: {n_queries}")
    print(f"gold R@all     : {r_gold:.2%}  ({gold_tot[0]}/{gold_tot[1]})  [不变]")
    print(f"DBA R@all 旧口径: {r_dba_old:.2%}  ({dba_old_tot[0]}/{dba_old_tot[1]})  [gold expected 映射]")
    print(f"DBA R@all 新口径: {r_dba_new:.2%}  ({dba_new_tot[0]}/{dba_new_tot[1]})  [DBA 独立重标 expected]")
    print(f"\n逐条:")
    for r in rows:
        print(f"  {r['query_id']} gold={r['gold_r_at_all']} dba旧={r['dba_r_at_all_old']} "
              f"dba新={r['dba_r_at_all_new']}  exp={r['dba_expected']}")

    # 分场景汇总（重标口径，hit/total 累加）
    from collections import defaultdict
    cat_hit = defaultdict(lambda: [0, 0])
    for r in rows:
        cat = r.get("category", "?")
        if r["dba_r_at_all_new"] is not None:
            cat_hit[cat][1] += len(r["dba_expected"])
            cat_hit[cat][0] += r["dba_hit_new"]
    print("\n分场景 DBA R@all（重标口径，hit/expected）:")
    for cat, (h, t) in sorted(cat_hit.items()):
        print(f"  {cat}: {h}/{t} = {h/t:.1%}")


if __name__ == "__main__":
    main()
