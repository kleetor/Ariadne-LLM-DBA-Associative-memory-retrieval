# -*- coding: utf-8 -*-
"""E6-lite 分析：两名人工标注 vs 三个 LLM judge。

读两份 CSV（问卷导出或模板填写），配合 `e6lite_key.json` 的盲表，报告：
- Krippendorff α（ordinal，逐维度 + 总分）；
- 人-人一致率（完全一致 / 差 ≤1 / 平均绝对差）；
- 人类总分与三家 judge 总分的 Spearman（逐条）；
- 人-人分歧最大的 5 条、人-机分歧最大的 5 条；
- 验收：α_total ≥ 0.6；人与 judge 的相关 ≥ 0.5。

用法：
    python -m eval.e6_lite_analyze --rater-a eval/outputs/e6_lite/e6lite_A.csv \
        --rater-b eval/outputs/e6_lite/e6lite_B.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from eval.stats_report import krippendorff_alpha_ordinal, spearman_between

ROOT = Path(__file__).resolve().parent.parent
DIMS = ("记忆丰富度", "易理解性", "用户建模度")


def _read_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            iid = str(row.get("item_id") or "").strip()
            if not iid:
                continue
            rec: Dict[str, Any] = {"rater": str(row.get("rater_id") or "").strip(),
                                   "note": str(row.get("note") or "").strip()}
            for d in DIMS:
                v = str(row.get(d) or "").strip()
                rec[d] = float(v) if re.fullmatch(r"[1-5](\.0)?", v) else None
            out[iid] = rec
    return out


def _total(rec: Dict[str, Any]) -> float | None:
    vals = [rec.get(d) for d in DIMS]
    return float(np.mean(vals)) if all(v is not None for v in vals) else None


def _agree(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for d in DIMS:
        x, y = a.get(d), b.get(d)
        if x is None or y is None:
            continue
        out[d] = {"exact": int(x == y), "within1": int(abs(x - y) <= 1), "absdiff": abs(x - y)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E6-lite 分析")
    ap.add_argument("--rater-a", required=True)
    ap.add_argument("--rater-b", required=True)
    ap.add_argument("--key", default="eval/outputs/e6_lite/e6lite_key.json")
    ap.add_argument("--out", default="eval/outputs/e6_lite/report.json")
    ap.add_argument("--md", default="eval/outputs/e6_lite/report.md")
    args = ap.parse_args()

    key = json.loads((ROOT / args.key).read_text(encoding="utf-8"))
    items = {it["item_id"]: it for it in key["items"]}
    A = _read_csv(ROOT / args.rater_a)
    B = _read_csv(ROOT / args.rater_b)
    common = sorted(set(A) & set(B) & set(items))

    # ---- α 逐维度 ----
    alpha: Dict[str, float] = {}
    for d in DIMS:
        mat = np.array([[A[i][d], B[i][d]] for i in common if A[i][d] is not None and B[i][d] is not None],
                       dtype=float)
        alpha[d] = krippendorff_alpha_ordinal(mat) if mat.size else float("nan")
    mat_total = np.array([[round(_total(A[i]) * 2) / 2, round(_total(B[i]) * 2) / 2]
                          for i in common if _total(A[i]) is not None and _total(B[i]) is not None],
                         dtype=float)
    alpha["total(0.5)"] = krippendorff_alpha_ordinal(mat_total) if mat_total.size else float("nan")

    # ---- 人-人一致 ----
    agree: Dict[str, Any] = {}
    for d in DIMS:
        vals = [_agree(A[i], B[i]).get(d) for i in common]
        vals = [v for v in vals if v]
        agree[d] = {
            "n": len(vals),
            "exact": round(float(np.mean([v["exact"] for v in vals])), 3) if vals else None,
            "within1": round(float(np.mean([v["within1"] for v in vals])), 3) if vals else None,
            "mean_abs_diff": round(float(np.mean([v["absdiff"] for v in vals])), 3) if vals else None,
        }

    # ---- 人-机相关 ----
    judge_names = sorted({j for it in key["items"] for j in (it.get("judges") or {})})
    corr: Dict[str, Any] = {}
    for j in judge_names:
        xs, ys = [], []
        for i in common:
            ht = _total(A[i])
            ht2 = _total(B[i])
            jt = ((items[i].get("judges") or {}).get(j) or {}).get("total")
            if ht is None or ht2 is None or jt is None:
                continue
            xs.append((ht + ht2) / 2)
            ys.append(float(jt))
        corr[j] = {"n": len(xs), "spearman": round(spearman_between(xs, ys), 3)
                   if len(xs) >= 3 else None}

    # ---- 分歧案例 ----
    def human_gap(i: str) -> float:
        t = [_total(A[i]), _total(B[i])]
        return abs(t[0] - t[1]) if all(x is not None for x in t) else 0.0

    def llm_gap(i: str) -> float:
        ht = _total(A[i]); ht2 = _total(B[i])
        js = [((items[i].get("judges") or {}).get(j) or {}).get("total") for j in judge_names]
        js = [float(x) for x in js if x is not None]
        if ht is None or ht2 is None or not js:
            return 0.0
        return abs((ht + ht2) / 2 - float(np.mean(js)))

    def brief(i: str) -> Dict[str, Any]:
        it = items[i]
        return {
            "item_id": i, "query_id": it["query_id"], "category": it["category"],
            "arm": it["arm"], "query": it["query"],
            "human": {"A": {d: A[i].get(d) for d in DIMS}, "B": {d: B[i].get(d) for d in DIMS},
                      "A_total": _total(A[i]), "B_total": _total(B[i])},
            "judges": {j: (it.get("judges") or {}).get(j) for j in judge_names},
        }

    worst_hh = [brief(i) for i in sorted(common, key=human_gap, reverse=True)[:5]]
    worst_hl = [brief(i) for i in sorted(common, key=llm_gap, reverse=True)[:5]]

    report = {
        "meta": {
            "n_items": len(common),
            "rater_a": args.rater_a,
            "rater_b": args.rater_b,
            "acceptance": {"alpha_total": 0.6, "spearman": 0.5},
        },
        "alpha_ordinal": alpha,
        "human_agreement": agree,
        "human_vs_judge_spearman": corr,
        "worst_human_human": worst_hh,
        "worst_human_llm": worst_hl,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown ----
    lines = [
        "# E6-lite 人工复核报告", "",
        f"- 共同题数：{len(common)}；rater A：`{Path(args.rater_a).name}`；"
        f"rater B：`{Path(args.rater_b).name}`",
        "",
        "## 1. Krippendorff α（ordinal）", "",
        "| 维度 | α |", "|---|:---:|",
    ]
    for d, v in alpha.items():
        lines.append(f"| {d} | {v:.3f} |")
    lines += ["", "验收：α_total ≥ 0.6。", "", "## 2. 人-人一致", "",
              "| 维度 | n | 完全一致 | 差 ≤1 | 平均绝对差 |", "|---|:---:|:---:|:---:|:---:|"]
    for d, v in agree.items():
        lines.append(f"| {d} | {v['n']} | {v['exact']} | {v['within1']} | {v['mean_abs_diff']} |")
    lines += ["", "## 3. 人-机 Spearman（人类均分 vs judge total）", "",
              "| judge | n | ρ |", "|---|:---:|:---:|"]
    for j, v in corr.items():
        lines.append(f"| {j} | {v['n']} | {v['spearman']} |")
    lines += ["", "验收：ρ ≥ 0.5。", "", "## 4. 人-人分歧最大的 5 条", ""]
    for x in worst_hh:
        lines.append(f"- **{x['item_id']}**（{x['category']} / {x['arm']}）"
                     f"A={x['human']['A_total']} B={x['human']['B_total']}：{x['query'][:50]}")
    lines += ["", "## 5. 人-机分歧最大的 5 条", ""]
    for x in worst_hl:
        js = {j: (v or {}).get("total") for j, v in (x["judges"] or {}).items()}
        lines.append(f"- **{x['item_id']}**（{x['category']} / {x['arm']}）"
                     f"人={round((x['human']['A_total'] + x['human']['B_total']) / 2, 2)} "
                     f"judge={js}：{x['query'][:50]}")
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[E6-lite] wrote {out} 与 {md}")


if __name__ == "__main__":
    main()
