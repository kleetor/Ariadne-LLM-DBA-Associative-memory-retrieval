# -*- coding: utf-8 -*-
"""
DBA 多版本基础评测汇总：v4-v7 抽取质量指标对比（vs gold）

用法:
    python -m eval.compare_dba_versions
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VERSIONS = ["v4", "v5", "v6", "v7"]
KEYS = [
    ("dba_active_nodes", "节点", "{:.0f}"),
    ("dba_edges", "边", "{:.0f}"),
    ("node_recall", "节点召回", "{:.1%}"),
    ("node_precision", "节点精确", "{:.1%}"),
    ("type_accuracy", "类型正确", "{:.1%}"),
    ("edge_recall", "边召回", "{:.1%}"),
    ("edge_precision", "边精确", "{:.1%}"),
    ("edge_type_accuracy", "边类型一致", "{:.1%}"),
    ("dba_new_edges", "新边", "{:.0f}"),
]


def main() -> None:
    data = {}
    for v in VERSIONS:
        p = ROOT / f"eval/outputs/dba_extraction_subset_b6_{v}.json"
        data[v] = json.loads(p.read_text(encoding="utf-8"))["eval"]

    print("===== DBA 多版本基础评测（vs gold 100 节点 / 138 边）=====")
    header = f"{'指标':<10}" + "".join(f"{v:>10}" for v in VERSIONS)
    print(header)
    for key, label, fmt in KEYS:
        row = f"{label:<10}"
        for v in VERSIONS:
            val = data[v][key]
            if isinstance(val, float):
                row += f"{fmt.format(val):>10}"
            else:
                row += f"{val:>10}"
        print(row)

    # 波动统计
    print("\n===== 稳定性（4 版指标波动范围）=====")
    for key, label, fmt in KEYS:
        vals = [data[v][key] for v in VERSIONS]
        if isinstance(vals[0], float):
            lo, hi = min(vals), max(vals)
            print(f"{label:<10}: {lo:.1%} ~ {hi:.1%}（跨度 {hi-lo:.1%}）" if key not in ("node_recall",) or lo < 1 else
                  f"{label:<10}: {lo:.2f} ~ {hi:.2f}")


if __name__ == "__main__":
    main()
