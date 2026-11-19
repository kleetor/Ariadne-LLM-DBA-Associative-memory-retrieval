# -*- coding: utf-8 -*-
"""
P 链路故事性质分析：
1. 扩展性：每条查询输出中「不在 expected（查询命题所需）的扩展节点」占比
2. 性质差异：聚焦 vs 发散的模式（按类别/查询分布）
3. 命题覆盖：是否包含查询命题所需的全部节点（expected 覆盖完整度）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

p = {r["query_id"]: r for r in json.load(open("eval/outputs/p_candidates.json", encoding="utf-8"))["results"]}
queries = yaml.safe_load(open("data/LiFEMem_queries.yaml", encoding="utf-8"))["queries"]

rows = []
for q in queries:
    qid = q["id"]
    r = p[qid]
    expected = set(q.get("expected", []))
    out_ids = [c["id"] for c in r["candidates"]]
    out = set(out_ids)
    hit = out & expected
    extend = out - expected
    # 覆盖完整度
    missing = expected - out if expected else set()
    rows.append({
        "qid": qid, "cat": q["category"], "query": q["text"],
        "N": len(out), "|expected|": len(expected),
        "覆盖数": len(hit), "完整覆盖": (len(missing) == 0) if expected else None,
        "missing": sorted(missing), "扩展数": len(extend), "扩展率": len(extend) / len(out),
    })

# 1) 总体统计
print("===== 1. 扩展性总览（35 条查询）=====")
exts = [r["扩展率"] for r in rows]
print(f"扩展率: min={min(exts):.3f} max={max(exts):.3f} 均值={sum(exts)/len(exts):.3f}")
focus = [r for r in rows if r["扩展率"] < 0.6]
diverge = [r for r in rows if r["扩展率"] >= 0.9]
print(f"聚焦型（扩展率<0.6）: {len(focus)} 条 -> {[r['qid'] for r in focus]}")
print(f"发散型（扩展率>=0.9）: {len(diverge)} 条 -> {[r['qid'] for r in diverge]}")

# 2) 按类别
print("\n===== 2. 按类别（扩展率 / 完整覆盖）=====")
cats = {}
for r in rows:
    cats.setdefault(r["cat"], []).append(r)
for cat, rs in sorted(cats.items()):
    cov = [r for r in rs if r["|expected|"] > 0]
    full = [r for r in cov if r["完整覆盖"]]
    ex = sum(r["扩展率"] for r in rs) / len(rs)
    print(f"  [{cat:<18}] ({len(rs)}条) 扩展率均值={ex:.3f} | 完整覆盖 {len(full)}/{len(cov)}")

# 3) 未完整覆盖的查询（漏了哪些 expected）
print("\n===== 3. 命题覆盖：未完整覆盖 expected 的查询 =====")
miss_rows = [r for r in rows if r["|expected|"] > 0 and not r["完整覆盖"]]
for r in sorted(miss_rows, key=lambda x: -len(x["missing"])):
    print(f"  {r['qid']} [{r['cat']}] {r['query'][:30]}")
    print(f"    期望={r['|expected|']} 覆盖={r['覆盖数']} 漏={r['missing']}")

# 4) 每条查询明细（扩展率排序）
print("\n===== 4. 逐条明细（按扩展率降序）=====")
print(f"{'qid':<5}{'类别':<18}{'扩展率':>7}{'N':>4}{'覆盖':>4}{'缺失':>6}  查询")
for r in sorted(rows, key=lambda x: -x["扩展率"]):
    print(f"{r['qid']:<5}{r['cat']:<18}{r['扩展率']:>7.2f}{r['N']:>4}{r['覆盖数']:>4}"
          f"{str(len(r['missing'])) if r['missing'] else '-':>6}  {r['query'][:28]}")
