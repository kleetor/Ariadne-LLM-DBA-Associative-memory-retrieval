# -*- coding: utf-8 -*-
"""通用判分维度汇总：按「系统 × 维度」输出均分与两两胜负。

与 ``eval/dim_breakdown.py`` 的区别：那个脚本把文件与系统名（Ariadne/Mem0/Graphiti）
写死，只服务单一历史实验；本脚本对任意 judge 结果 JSON 通用 —— 例如三路并排
（两臂 + 纯向量块，中性标签 甲/乙/丙）这类非默认系统名的判分结果。

用法：
    python -m eval.summarize_judge_dims <judge1.json> [judge2.json ...]
"""

import json
import sys
from collections import defaultdict

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]


def summarize(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload["results"]
    if not rows:
        print(f"{path}: 无结果")
        return {}
    n = len(rows)
    # 保持文件内的出现顺序（与 blocks 的显示顺序一致），便于和 meta.display_order 对照
    names = list(dict.fromkeys(name for r in rows for name in r["scores"]))

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for name, sc in r["scores"].items():
            for key in DIMS + ["total"]:
                agg[name][key].append(float(sc.get(key, 0.0)))

    meta = payload.get("meta", {})
    print(f"\n=== {path} ===")
    print(f"judge={meta.get('judge_model')}  查询 {n} 条  mode={meta.get('mode')}")
    print(f"{'系统':<10}" + "".join(f"{k:<14}" for k in DIMS) + "total")
    for name in names:
        dims = [sum(agg[name][k]) / n for k in DIMS]
        print(f"{name:<10}" + "".join(f"{v:<14.3f}" for v in dims)
              + f"{sum(agg[name]['total']) / n:.3f}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            win = sum(1 for r in rows if r["scores"][a]["total"] > r["scores"][b]["total"])
            lose = sum(1 for r in rows if r["scores"][a]["total"] < r["scores"][b]["total"])
            print(f"    {a} vs {b}：胜 {win} / 负 {lose} / 平 {n - win - lose}")

    return {name: {k: sum(agg[name][k]) / n for k in DIMS + ["total"]} for name in names}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        summarize(arg)
