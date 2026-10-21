# SPDX-License-Identifier: AGPL-3.0-only

"""ALMEngine 空间注册表：构造必须在全局锁外，且不破坏既有并发不变量。

这里不构造真实 LLM / Embedding：用 `object.__new__` 造一个只含注册表字段的引擎，
再把 `alm.engine.MemorySpace` 换成一个可控的假类，专门验证：

  · 一个 user 的慢构造不会阻塞其它已驻留 user；
  · 同一 user 的并发首次访问只构造一次；
  · 构造失败可重试、不残留占位；
  · `create=False` 会等待正在进行的构造，而不是误判为不存在；
  · `get_space` / `space_count` 不被构造阻塞；
  · 租用期间不会被 LRU 换出，释放后才换出。

所有线程都用 `daemon=True` + 有限 `join`：即使实现回归，测试会**失败**而不是挂死。
"""

import threading
import time
from collections import OrderedDict

import pytest

from alm import engine as engine_mod
from alm.config import ALMConfig
from alm.engine import ALMEngine


class _FakeSpace:
    """最小替身：注册表只依赖 user_id 与 save()（后者在关闭/落盘路径用）"""

    def __init__(self, user_id, **kwargs):
        self.user_id = user_id
        self.saved = 0

    def save(self):
        self.saved += 1


def _engine(tmp_path, max_spaces=16):
    engine = object.__new__(ALMEngine)  # 跳过 LLM / Embedding 初始化
    engine.config = ALMConfig(data_dir=tmp_path / "data", max_spaces=max_spaces)
    engine.embeddings = None
    engine.llm = None
    engine.retrieval_stats = None
    engine.reranker = None
    engine._backfill_gate = None
    engine._spaces = OrderedDict()
    engine._in_use = {}
    engine._registry_lock = threading.RLock()
    engine._constructing = set()
    engine._space_cv = threading.Condition(engine._registry_lock)
    engine._construct_slots = threading.BoundedSemaphore(4)
    return engine


def test_slow_construction_does_not_block_resident_user(tmp_path, monkeypatch):
    """冷构造（分钟级 embedding）不能在全局锁内：驻留 user 必须立刻可用。"""
    engine = _engine(tmp_path)
    resident = _FakeSpace("u_fast")
    engine._spaces["u_fast"] = resident

    started = threading.Event()
    release = threading.Event()
    slow_result = {}

    class SlowSpace(_FakeSpace):
        def __init__(self, user_id, **kwargs):
            started.set()
            assert release.wait(5), "测试超时：慢构造没有被释放"
            super().__init__(user_id, **kwargs)

    monkeypatch.setattr(engine_mod, "MemorySpace", SlowSpace)

    def slow_lease():
        with engine._lease("u_slow", True) as space:
            slow_result["space"] = space

    slow_thread = threading.Thread(target=slow_lease, daemon=True)
    slow_thread.start()
    assert started.wait(5), "慢构造没有开始"

    fast_result = {}

    def fast_lease():
        with engine._lease("u_fast", False) as space:
            fast_result["space"] = space

    t0 = time.monotonic()
    fast_thread = threading.Thread(target=fast_lease, daemon=True)
    fast_thread.start()
    fast_thread.join(1.0)
    elapsed = time.monotonic() - t0
    if fast_thread.is_alive():
        release.set()
        fast_thread.join(5)
        slow_thread.join(5)
        raise AssertionError(f"驻留 user 被冷构造阻塞（>{elapsed:.1f}s）")

    assert fast_result["space"] is resident
    assert not release.is_set(), "驻留请求完成时，慢构造本应仍在进行"

    release.set()
    slow_thread.join(5)
    assert not slow_thread.is_alive()
    assert slow_result["space"].user_id == "u_slow"


def test_same_user_concurrent_lease_constructs_once(tmp_path, monkeypatch):
    """同一 user 的并发首次访问只能构造一次，且拿到同一实例。"""
    engine = _engine(tmp_path)
    counter = {"n": 0}
    counter_lock = threading.Lock()
    start = threading.Event()

    class CountingSpace(_FakeSpace):
        def __init__(self, user_id, **kwargs):
            with counter_lock:
                counter["n"] += 1
            time.sleep(0.2)  # 放大并发窗口
            super().__init__(user_id, **kwargs)

    monkeypatch.setattr(engine_mod, "MemorySpace", CountingSpace)
    got = []
    errors = []

    def worker():
        try:
            start.wait(5)
            with engine._lease("u_same", True) as space:
                got.append(space)
        except BaseException as exc:  # pragma: no cover - 失败时用于报错
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(5)

    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads), "并发租用未在 5s 内完成"
    assert counter["n"] == 1, f"同一 user 被构造了 {counter['n']} 次"
    assert len(got) == 2 and got[0] is got[1]


def test_construction_failure_is_retryable(tmp_path, monkeypatch):
    """构造异常不得留下 `_constructing` 占位，后续请求应能重试成功。"""
    engine = _engine(tmp_path)
    calls = {"n": 0}

    class FlakySpace(_FakeSpace):
        def __init__(self, user_id, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            super().__init__(user_id, **kwargs)

    monkeypatch.setattr(engine_mod, "MemorySpace", FlakySpace)

    with pytest.raises(RuntimeError):
        with engine._lease("u_flaky", True):
            pass
    assert engine._constructing == set(), "失败后仍残留构造占位"

    with engine._lease("u_flaky", True) as space:
        assert space.user_id == "u_flaky"
    assert calls["n"] == 2


def test_create_false_waits_for_inflight_construction(tmp_path, monkeypatch):
    """`create=False` 遇到正在构造的同一 user：应等待并命中实例，而非返回 None。"""
    engine = _engine(tmp_path)
    started = threading.Event()
    release = threading.Event()

    class SlowSpace(_FakeSpace):
        def __init__(self, user_id, **kwargs):
            started.set()
            assert release.wait(5)
            super().__init__(user_id, **kwargs)

    monkeypatch.setattr(engine_mod, "MemorySpace", SlowSpace)
    result = {}

    def creator():
        with engine._lease("u_new", True) as space:
            result["created"] = space

    def reader():
        with engine._lease("u_new", False) as space:
            result["read"] = space

    created = threading.Thread(target=creator, daemon=True)
    read = threading.Thread(target=reader, daemon=True)
    created.start()
    assert started.wait(5)

    read.start()
    try:
        time.sleep(0.1)
        assert read.is_alive(), "create=False 在构造完成前就返回了"
    finally:
        release.set()

    created.join(5)
    read.join(5)
    assert not created.is_alive() and not read.is_alive()
    assert result["read"] is result["created"]


def test_get_space_does_not_hold_registry_lock(tmp_path, monkeypatch):
    """`get_space` 的构造同样在锁外：期间 `space_count()` 必须立刻返回。"""
    engine = _engine(tmp_path)
    started = threading.Event()
    release = threading.Event()
    loaded = {}

    class SlowSpace(_FakeSpace):
        def __init__(self, user_id, **kwargs):
            started.set()
            assert release.wait(5)
            super().__init__(user_id, **kwargs)

    monkeypatch.setattr(engine_mod, "MemorySpace", SlowSpace)

    def load():
        loaded["space"] = engine.get_space("u_slow", True)

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    assert started.wait(5)

    counted = {}

    def count():
        counted["n"] = engine.space_count()

    counter = threading.Thread(target=count, daemon=True)
    t0 = time.monotonic()
    counter.start()
    counter.join(1.0)
    elapsed = time.monotonic() - t0
    if counter.is_alive():
        release.set()
        counter.join(5)
        loader.join(5)
        raise AssertionError(f"构造期间仍持有全局注册表锁（>{elapsed:.1f}s）")

    assert counted["n"] == 0
    release.set()
    loader.join(5)
    assert loaded["space"].user_id == "u_slow"
    assert engine.space_count() == 1


def test_lease_evicts_only_after_release(tmp_path, monkeypatch):
    """租用中的空间跳过换出；释放后按 LRU 正常换出。"""
    engine = _engine(tmp_path, max_spaces=1)
    monkeypatch.setattr(engine_mod, "MemorySpace", _FakeSpace)

    with engine._lease("u_keep", True):
        # 模拟另一个 user 的空间加入，触发超额换出
        other = _FakeSpace("u_other")
        with engine._registry_lock:
            engine._spaces["u_other"] = other
            engine._spaces.move_to_end("u_other")
            engine._evict_locked()
            assert "u_keep" in engine._spaces, "在用空间被换出"
            assert "u_other" not in engine._spaces, "最旧的空闲空间没有被换出"

    # 释放后再进一个新空间：u_keep 已不在用 → 应被换出
    with engine._lease("u_next", True):
        pass
    assert "u_keep" not in engine._spaces
    assert "u_next" in engine._spaces
