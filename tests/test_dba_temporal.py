# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
DBA 抽取侧时序规则（手术式守卫）的离线单元测试。

覆盖：
  · has_temporal_signal() 触发判定（>=2 个不同日期/时期词才命中）
  · 语义边界：仅 1 个日期词 / 同一词重复 / 仅 era 词 / 弱时间副词 → 不命中
  · 时序增强 prompt 额外规则字符串的完整性（含 TEMPORAL / 时间节点）

说明：抽取的"零回归定量"由 eval/run_dba_extraction_eval.py 在带 LLM 的闭环上验证；
本文件只测**确定性、离线**的守卫逻辑，不走 LLM。
"""

from dba_pipeline.extraction.dba import (
    has_temporal_signal,
    NODE_TEMPORAL_EXTRA,
    EDGE_TEMPORAL_EXTRA,
)


class TestHasTemporalSignal:
    """守卫触发判定：只有 >=2 个**不同**日期/时期词才命中"""

    def test_empty_and_none_false(self):
        assert has_temporal_signal("") is False
        assert has_temporal_signal(None) is False

    def test_only_one_date_word_false(self):
        assert has_temporal_signal("昨天被老板点名批评了") is False
        assert has_temporal_signal("周一下午有个会") is False
        assert has_temporal_signal("上周接了个项目") is False

    def test_duplicate_same_word_false(self):
        # 同一个时期词重复出现，不算">=2 个不同"，不应触发
        assert has_temporal_signal("昨天加班，昨天又晚睡") is False

    def test_era_word_only_false(self):
        # 仅 era 词不触发（避免"大学时学的钢琴"这类设定陈述误拆）
        assert has_temporal_signal("大学时学的钢琴，现在转行健身") is False
        assert has_temporal_signal("高中三年一直是班长") is False

    def test_weak_time_adverb_only_false(self):
        # 凌晨/中午/晚上/每天 等日常时间副词被刻意排除
        assert has_temporal_signal("每晚都在加班，中午也没休息") is False

    def test_two_distinct_week_days_true(self):
        # 周[一二三四五六日天] 两个不同 → 命中
        assert has_temporal_signal("周三凌晨上线，周五做复盘") is True

    def test_two_distinct_periods_true(self):
        # 上周 + 这周（这周/周二 均有效），≥2 个不同 → 命中
        assert has_temporal_signal("上周一临时加需求，这周二赶代码") is True

    def test_two_distinct_year_terms_true(self):
        assert has_temporal_signal("去年报的班，今年还在坚持上") is True

    def test_outer_week_and_day_true(self):
        # 上周 与 周几 混用也算不同时期 → 命中
        assert has_temporal_signal("上周出差，周三才回来") is True


class TestTemporalPromptExtras:
    """时序增强链的额外规则字符串应完整（含 TEMPORAL / 时间节点）"""

    def test_node_extra_mentions_time_node_and_temporal(self):
        assert "时间节点" in NODE_TEMPORAL_EXTRA

    def test_edge_extra_mentions_temporal(self):
        assert "TEMPORAL" in EDGE_TEMPORAL_EXTRA
        # 强调 TEMPORAL 是"事件→时间单向"，与 SEQUENCE（事件先后）区分
        assert "单向" in EDGE_TEMPORAL_EXTRA


def _run_all():
    """无 pytest 时的手动 runner（亦可被 pytest 收集运行）"""
    failures = []
    suites = (TestHasTemporalSignal, TestTemporalPromptExtras)
    for cls in suites:
        for name in sorted(dir(cls)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(cls(), name)()
                print(f"PASS {cls.__name__}.{name}")
            except Exception as e:
                failures.append(f"{cls.__name__}.{name}: {e!r}")
                print(f"FAIL {cls.__name__}.{name}: {e!r}")
    return failures


if __name__ == "__main__":
    import sys
    fails = _run_all()
    if fails:
        print(f"\n失败 {len(fails)} 项: {fails}")
        sys.exit(1)
    print("\n全部通过 [OK]")
    sys.exit(0)
