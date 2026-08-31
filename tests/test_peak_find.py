# SPDX-License-Identifier: AGPL-3.0-only

"""寻峰终止器单元测试"""
from dba_pipeline.core.peak_find import PeakFinder


def test_default_params_match_usage():
    # P2-6：默认值应与 retriever 实际使用一致
    pf = PeakFinder()
    assert pf.patience == 2
    assert pf.min_delta == 0.015


def test_first_round_continue():
    pf = PeakFinder()
    assert pf.add_round(0.5) == "continue"


def test_peak_found_after_decline():
    pf = PeakFinder(patience=2, min_delta=0.015)
    assert pf.add_round(0.5) == "continue"
    assert pf.add_round(0.6) == "continue"
    assert pf.add_round(0.7) == "continue"  # peak_index=2
    assert pf.add_round(0.5) == "continue"  # 首次显著下降
    assert pf.add_round(0.45) == "continue"  # extra_rounds=1
    assert pf.add_round(0.4) == "peak_found"  # extra_rounds=2
    assert pf.peak_index == 2


def test_small_fluctuation_ignored():
    pf = PeakFinder(patience=2, min_delta=0.015)
    pf.add_round(0.5)
    # 波动 0.01 < min_delta，视为持平，不触发下降
    assert pf.add_round(0.51) == "continue"
    assert pf.seen_decline is False
