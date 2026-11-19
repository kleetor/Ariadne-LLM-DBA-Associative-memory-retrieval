# -*- coding: utf-8 -*-
"""
参数收紧实验：目的回归系数（purpose_weight_coef）× 峰值容忍带（peak_tolerance）

对指定查询（默认 q07，目的推断跑偏典型）跑不同参数组合，
对比输出量 N、候选质量（目标节点命中）与 top 候选。

用法:
    python -m eval.param_sweep q07
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever

# 参数组合：base / 收紧A(目的) / 收紧B(容忍带) / 收紧C(两者)
COMBOS = [
    ("base",              None, None),
    ("A_coef0.3",         0.3,  None),
    ("B_tol0.05",         None, 0.05),
    ("C_tol0.02",         None, 0.02),
    ("D_coef0.3_tol0.02", 0.3,  0.02),
]

QUERIES = {
    "q01": ("用户现在做什么工作？", {"w0"}),
    "q07": ("用户平时喜欢喝什么咖啡？", {"w5", "w6", "w7"}),
    "q17": ("用户高中时期的成绩变化是怎样的？", {"h0", "h3"}),
    "q25": ("用户的职业经历顺序是怎样的？", {"c11", "w0"}),
}


def main() -> None:
    qid = sys.argv[1] if len(sys.argv) > 1 else "q07"
    if qid not in QUERIES:
        print(f"未知查询 {qid}，可选: {list(QUERIES)}")
        return
    query, targets = QUERIES[qid]
    _load_dotenv(ROOT / ".env")

    # 只构建一次，逐组覆盖参数
    graph, retriever = build_retriever("data/LiFEMem.yaml")
    print(f"===== 参数收紧实验：{qid}「{query}」目标 {sorted(targets)} =====")

    for name, coef, tol in COMBOS:
        retriever.purpose_weight_coef = coef if coef is not None else 0.5
        retriever.peak_finder.peak_tolerance = tol if tol is not None else 0.10
        res = retriever.retrieve(query)
        peak = res.get("peak_memories", [])
        ids = [mid for mid, _ in peak]
        hit = set(ids) & targets
        print(f"\n[{name}] coef={retriever.purpose_weight_coef}, tol={retriever.peak_finder.peak_tolerance} "
              f"-> N={len(ids)}, 目标命中 {len(hit)}/{len(targets)} {sorted(hit)}")
        print(f"  目的: {res['purpose']['purposes']}")
        for mid, content in peak[:5]:
            mark = "  <-目标" if mid in targets else ""
            print(f"    {mid}: {content[:42]}{mark}")


if __name__ == "__main__":
    main()
