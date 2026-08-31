# SPDX-License-Identifier: AGPL-3.0-only

"""路径追踪器双因子权重单元测试"""
from dba_pipeline.core.path_tracker import PathTracker, PathTrackerConfig


def test_make_path_key():
    assert PathTracker.make_path_key("a", "b", "causal") == "a→b@causal"


def test_unactivated_edge_multiplier_is_one():
    tracker = PathTracker()
    assert tracker.get_edge_weight_multiplier("a", "b", "causal") == 1.0


def test_double_factor_after_activation():
    tracker = PathTracker()
    tracker.record_activation("a→b@causal")
    # lifetime bonus: 1.05, session satiation: 0.85
    w = tracker.get_dynamic_weight("a→b@causal", base_weight=1.0)
    assert abs(w - 1.05 * 0.85) < 1e-9


def test_lifetime_cap():
    tracker = PathTracker(PathTrackerConfig(lifetime_coef=0.05, lifetime_cap=2.0))
    for _ in range(100):
        tracker.record_activation("a→b@causal")
    stats = tracker.get_path_stats("a→b@causal")
    assert stats["lifetime_bonus"] == 2.0


def test_session_satiation_floor():
    tracker = PathTracker(
        PathTrackerConfig(session_satiation_coef=0.15, session_satiation_floor=0.3)
    )
    for _ in range(10):
        tracker.record_activation("a→b@causal")
    stats = tracker.get_path_stats("a→b@causal")
    assert stats["session_satiation"] == 0.3


def test_start_session_resets_satiation_only():
    tracker = PathTracker()
    tracker.start_session()
    tracker.record_activation("a→b@causal")
    assert tracker.lifetime_counts["a→b@causal"] == 1
    assert tracker.session_counts["a→b@causal"] == 1

    tracker.start_session()  # 新会话
    assert tracker.lifetime_counts["a→b@causal"] == 1  # 终身计数保留
    assert tracker.session_counts.get("a→b@causal", 0) == 0  # 会话计数重置
