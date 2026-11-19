# -*- coding: utf-8 -*-
"""E12a 汇总：扰动图上的 E3 三臂配对统计。

读 `judge_comprehensive` 的产物（逐题打乱、label_map 还原），按计划 §4 的验收：
- C−D = par_raw − vector_raw 的均值差 > 1.0；
- 同一配对的 rank-biserial r > 0.8；
- 相对原图 C−D 的保持率 = 扰动图 C−D / 原图 C−D（§1.3 要求 ≥ 60%）。

原图基线默认取 `judge_e3_bandfix_deepseek.json`（与扰动图同一 judge 模型；批次不同，
仅作量级参照；若给了 --baseline 就用指定文件）。

用法：
    python -m eval.e12a_report \
        --del eval/outputs/e12a/judge_del30_deepseek.json \
        --rewire eval/outputs/e12a/judge_rewire30_deepseek.json \
        --baseline eval/outputs/e3/judge_e3_bandfix_deepseek.json \
        --out eval/outputs/e12a/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.stats_report import load_judge, paired_stats  # noqa: E402


def _compare(path: str) -> Dict:
    model, totals, dims, cats = load_judge(path)
    qids = sorted(totals)
    out: Dict[str, object] = {"judge": model, "n_queries": len(qids), "arms": {}}
    for arm in ("par_story", "par_raw", "vector_raw"):
        vals = [totals[q][arm] for q in qids if arm in totals[q]]
        if vals:
            out["arms"][arm] = round(sum(vals) / len(vals), 4)
    for a, b in (("par_raw", "vector_raw"), ("par_story", "vector_raw")):
        qs = [q for q in qids if a in totals[q] and b in totals[q]]
        if qs:
            st = paired_stats([totals[q][a] for q in qs], [totals[q][b] for q in qs])
            out[f"{a}_minus_{b}"] = {
                "diff": round(st["diff"], 4),
                "ci95": [round(x, 4) for x in st["diff_ci95"]],
                "win_tie_loss": [st["win"], st["tie"], st["loss"]],
                "wilcoxon_p": st.get("wilcoxon_p"),
                "rank_biserial_r": round(st.get("rank_biserial_r", float("nan")), 4),
            }
    out["categories"] = cats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E12a 汇总")
    ap.add_argument("--del", dest="del_path", required=True)
    ap.add_argument("--rewire", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default="eval/outputs/e12a/stats.json")
    args = ap.parse_args()

    payload: Dict[str, Optional[Dict]] = {"graphs": {}}
    base_diff = None
    if args.baseline:
        base = _compare(args.baseline)
        payload["graphs"]["baseline"] = base
        base_diff = base.get("par_raw_minus_vector_raw", {}).get("diff")
    for name, path in (("del30", args.del_path), ("rewire30", args.rewire)):
        payload["graphs"][name] = _compare(path)

    print(f"{'图':10s} {'n':>4s} {'C−D':>7s} {'r_rb':>7s} {'A−D':>7s} "
          f"{'保持率':>8s} {'验收':>6s}")
    for name, g in payload["graphs"].items():
        cd = g.get("par_raw_minus_vector_raw", {})
        ad = g.get("par_story_minus_vector_raw", {})
        diff = cd.get("diff")
        keep = (diff / base_diff) if (diff is not None and base_diff) else None
        ok = diff is not None and diff > 1.0 and cd.get("rank_biserial_r", 0) > 0.8 \
            and (keep is None or keep >= 0.6)
        print(f"{name:10s} {g['n_queries']:4d} {str(diff):>7s} "
              f"{str(cd.get('rank_biserial_r')):>7s} {str(ad.get('diff')):>7s} "
              f"{(f'{keep:.0%}' if keep is not None else '-'):>8s} {'PASS' if ok else 'FAIL':>6s}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E12a] wrote {p}")


if __name__ == "__main__":
    main()
