# SPDX-License-Identifier: AGPL-3.0-only

"""回补断点推进的回归：长消息被切成多组时，不得把断点直接推到队尾。

背景：`_split_add_batches_span` 对「跨组的长消息」把中间组记为 `n_msgs=0`。
旧实现无条件执行 `conv_items[cursor - 1]`，当 cursor 仍为 0 时取到 `-1`（最后一条），
把 `done` 推到队尾——重启/放弃后，尚未建图的内容会被永久跳过。

这里用最小替身直接跑 `run_backfill`，钉住「只有最后一批（整条消息建完）才推进断点」。
"""

import threading
from types import SimpleNamespace

from alm.space import MemorySpace


class _FakeDBA:
    def __init__(self):
        self.calls = 0

    def maintain(self, conversation, timestamp=None):
        self.calls += 1
        return {"result": {"created_ids": ["n1"]},
                "ops": {"node_ops": [{"action": "create", "content": "x"}]}}


def _space():
    space = MemorySpace.__new__(MemorySpace)
    space.user_id = "u_backfill_test"
    space.config = SimpleNamespace(
        add_batch_size=6,
        add_max_chars=1500,
        backfill_batch_size=24,
        backfill_max_chars=8000,
        backfill_group_seconds=5.0,
        graph_max_pred_min=0,   # 关闭预算闸门
    )
    space._lock = threading.RLock()
    space._masked_id = lambda: "***"
    space._bg_cancel = False
    space.dba = _FakeDBA()
    # 20,000 字的消息会被 8000 预算切成 3 片；后面再跟两条短消息。
    # 组序列 = [(片1, 0), (片2, 0), (片3+短消息1+短消息2, 3)]
    space.doc_msgs = [
        {"role": "user", "content": "a" * 20000, "timestamp": 1, "channel": "conv"},
        {"role": "assistant", "content": "short one", "timestamp": 2, "channel": "conv"},
        {"role": "user", "content": "short two", "timestamp": 3, "channel": "conv"},
    ]
    space._progress = {"done": 0, "status": "pending"}
    space.saved_progress = []

    def _progress():
        return dict(space._progress)

    def _save(done, status):
        space.saved_progress.append((done, status))
        space._progress.update(done=done, status=status)

    space.backfill_progress = _progress
    space._save_backfill_progress = _save
    space.save = lambda: None
    return space


def test_backfill_progress_does_not_jump_to_tail_on_partial_message():
    space = _space()

    extracted = space.run_backfill()

    assert extracted == 3, "三条消息都应被消费"
    assert space.dba.calls == 3, "20k 消息应被切成 3 组"
    # 关键断言：除开跑时的初始占位（done=0），只有**最后一批**（整条消息建完）
    # 才允许写 running 进度。旧实现会在前两组各写一次 done=3（把未建内容标成已消费）。
    advanced = [d for d, status in space.saved_progress if status == "running" and d > 0]
    assert advanced == [3], f"断点在长消息中途被推进了：{space.saved_progress}"
    assert space.saved_progress[-1] == (3, "done")
