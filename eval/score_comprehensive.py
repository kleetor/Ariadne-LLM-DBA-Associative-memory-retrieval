# -*- coding: utf-8 -*-
"""
综合能力测评 · judge 判分汇总

按 0831 e2e 计划 §4.5.5：
- 每题标准分 = judge 对该题上下文块的一次判分（同题多系统各一档）。
- 总评 = 所有题目标准分的均值（给出总和 + 按类别分解，报告主看总均分）。
- 弃权/过滤类（abstention）单独呈列，不进主总分（D3：作为 Ariadne 独有能力单独展示）。
- 统计处理（§8）：小样本不给精确排位，给方向 + 区间。

输入：eval/outputs/judge_comprehensive.json（judge_comprehensive.py 产物）。
系统名从判分结果里自动取（默认三系统；E3 的 2×2 四条件也能直接汇总）。

用法:
    python -m eval.score_comprehensive
    python -m eval.score_comprehensive --judge eval/outputs/e3/judge_e3_2x2.json --out eval/outputs/e3/score_e3_2x2.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JUDGE = "eval/outputs/judge_comprehensive.json"
DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
SYSTEMS = ["Ariadne", "Mem0", "Graphiti"]
ABSTAIN_CAT = "abstention"


def main() -> None:
    parser = argparse.ArgumentParser(description="judge 判分汇总（多系统同题并排）")
    parser.add_argument("--judge", default=JUDGE, help="judge 结果 JSON")
    parser.add_argument("--out", default="eval/outputs/score_comprehensive.json")
    args = parser.parse_args()

    global SYSTEMS
    with open(ROOT / args.judge, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    judge_model = data.get("meta", {}).get("judge_model", "?")
    # 系统名以判分结果为准（meta.names 优先，其次首条 scores 的键序）
    names = data.get("meta", {}).get("names")
    if not names and results:
        names = list(results[0].get("scores", {}).keys())
    if names:
        SYSTEMS = list(names)
    print(f"judge_model={judge_model}  查询数={len(results)}")

    # 匿名打乱还原：E3 等实验每行带 label_map（条件→匿名称），逐题映射回条件名再汇总；
    # 旧格式（带真实系统名）没有 label_map，原样使用。judge 结果里缺 label_map 时，
    # 回落到 ctx 文件（judge meta.ctx）里读。
    ctx_maps = {}
    ctx_path = data.get("meta", {}).get("ctx")
    if ctx_path and (ROOT / ctx_path).exists():
        with open(ROOT / ctx_path, encoding="utf-8") as f:
            for row in json.load(f).get("queries", []):
                if row.get("label_map"):
                    ctx_maps[row.get("query_id")] = row["label_map"]
    restored = []
    for r in results:
        lm = r.get("label_map") or ctx_maps.get(r.get("query_id"))
        if lm:
            inv = {lab: cond for cond, lab in lm.items()}
            r = {**r, "scores": {inv.get(k, k): v for k, v in r.get("scores", {}).items()}}
        restored.append(r)
    results = restored
    seen = [cond for r in results for cond in r.get("scores", {})]
    order = ["par_story", "vector_story", "par_raw", "vector_raw"]
    SYSTEMS = sorted(dict.fromkeys(seen),
                     key=lambda s: (order.index(s) if s in order else len(order), s)) or SYSTEMS
    print(f"系统={SYSTEMS}（已按 label_map 还原匿名打乱）")

    # 无效判分剔除：judge 偶发返回全 0（解析失败且重试仍失败），这类行对所有系统同时
    # 记 0，会把均值同比拉低，属于数据质量噪声而非系统差异。
    invalid = [r["query_id"] for r in results
               if not any(float(sc.get("total", 0) or 0) > 0 for sc in r.get("scores", {}).values())]
    if invalid:
        print(f"剔除无效判分（全 0）: {invalid}")
        results = [r for r in results if r["query_id"] not in set(invalid)]

    # 分类
    main_rows = [r for r in results if r["category"] != ABSTAIN_CAT]
    abst_rows = [r for r in results if r["category"] == ABSTAIN_CAT]
    print(f"\n主判分（非弃权）={len(main_rows)}  弃权类={len(abst_rows)}")

    # 主判分：总均分 + 各维均分
    print("\n===== 主判分总均分（非弃权） =====")
    print(f"{'系统':<10}{'n':<4}{'total':<8}{'丰富度':<8}{'易理解':<8}{'用户建模':<8}")
    summary = {}
    for s in SYSTEMS:
        totals, dims = [], {d: [] for d in DIMS}
        for r in main_rows:
            sc = r.get("scores", {}).get(s)
            if not sc:
                continue
            totals.append(sc.get("total", 0))
            for d in DIMS:
                dims[d].append(sc.get(d, 0))
        n = len(totals)
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        summary[s] = {"n": n, "total": round(avg(totals), 2),
                      **{d: round(avg(dims[d]), 2) for d in DIMS}}
        print(f"{s:<10}{n:<4}{avg(totals):<8.2f}{avg(dims['记忆丰富度']):<8.2f}"
              f"{avg(dims['易理解性']):<8.2f}{avg(dims['用户建模度']):<8.2f}")

    # 按类别分解
    print("\n===== 按类别 total 均值 =====")
    cats = []
    for r in main_rows:
        if r["category"] not in cats:
            cats.append(r["category"])
    print(f"{'类别':<18}" + "".join(f"{s:>12}" for s in SYSTEMS))
    cat_summary = {}
    for cat in cats:
        line = f"{cat:<18}"
        cat_summary[cat] = {}
        for s in SYSTEMS:
            vals = [r["scores"][s]["total"] for r in main_rows
                    if r["category"] == cat and r["scores"].get(s)]
            avg = sum(vals) / len(vals) if vals else float("nan")
            cat_summary[cat][s] = round(avg, 2) if vals else None
            line += f"{avg:>12.2f}" if vals else f"{'-':>12}"
        print(line)

    # 弃权类
    print("\n===== 弃权/过滤类（abstention，单独呈列不进总分） =====")
    for r in abst_rows:
        print(f"  {r['query_id']}: " + "  ".join(
            f"{s}={r['scores'][s]['total']:.2f}" for s in SYSTEMS if r["scores"].get(s)))

    # 两两对比（win rate + 均值差）
    print("\n===== 两两对比（total 均值差 + 胜率） =====")
    import itertools
    for a, b in itertools.combinations(SYSTEMS, 2):
        pairs = [(r["scores"][a]["total"], r["scores"][b]["total"]) for r in main_rows
                 if r["scores"].get(a) and r["scores"].get(b)]
        wins_a = sum(1 for x, y in pairs if x > y)
        wins_b = sum(1 for x, y in pairs if x < y)
        draws = len(pairs) - wins_a - wins_b
        d = sum(x - y for x, y in pairs) / len(pairs)
        print(f"  {a} vs {b}: 均值差 {d:+.2f}（{a}胜{wins_a}/平{draws}/{b}胜{wins_b}，共{len(pairs)}条）")

    # 落盘
    payload = {"meta": {"judge_model": judge_model, "method": "joint_1pass",
                        "rubric": DIMS, "abstain_excluded": True},
               "summary": summary, "by_category": cat_summary, "abstention": abst_rows}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
