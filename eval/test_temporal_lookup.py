# -*- coding: utf-8 -*-
"""
独立单跳时序工具 temporal_lookup 的冒烟测试（v1：仅"时间→事件"反向）。

工具范围（按 §9.6 分工路由）：
  · temporal_lookup 只处理"时间推事件"(TB) —— 主检索器反向权重为 0、做不到的部分；
  · "事件推时间"(TA) / 跨期(TD) / 纯事实 均归主联想检索器，本工具不测。

用法:
    python -m eval.test_temporal_lookup
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever


def show(tag, res):
    print(f"\n===== {tag} =====")
    matches = res.get("matches", [])
    if not matches:
        print("  （未定位到可用时间锚点 → 非'时间推事件'型，应走主检索器）")
        return
    for g in matches:
        ta = g.get("time_anchor", {})
        print(f"时间锚点: {ta['id']} ({ta.get('content')!r})")
        for f in g.get("facts", []):
            print(f"  - {f['id']} → {f['content']!r}")
        if not g.get("facts"):
            print("  （无共时事件）")


def main() -> None:
    _load_dotenv(ROOT / ".env")

    print("# 图1：跨期+细时间 gold（T1..T11 时间锚点）")
    _, r1 = build_retriever("data/dba_eval/lifemem_crossperiod.yaml")
    show("TB 时间→事件：昨天用户都做了些什么？", r1.temporal_lookup("昨天用户都做了些什么？"))
    show("TB 时间→事件：上周用户经历了什么累人的事？", r1.temporal_lookup("上周用户经历了什么累人的事？"))
    show("TB 时间→事件：高三下学期发生了什么？", r1.temporal_lookup("高三下学期发生了什么？"))
    show("非TB（事件→时间）该走主检索器：用户是什么时候开始学吉他的？",
         r1.temporal_lookup("用户是什么时候开始学吉他的？"))

    print("\n# 图2：LiFEMem 全量(322)，th0/th1/th2 时代锚点")
    _, r2 = build_retriever("data/LiFEMem.yaml")
    show("时代锚点：用户高中时期发生了什么？", r2.temporal_lookup("用户高中时期发生了什么？"))
    show("时代锚点：用户当前工作时期怎么样？", r2.temporal_lookup("用户当前工作时期怎么样？"))


if __name__ == "__main__":
    main()
