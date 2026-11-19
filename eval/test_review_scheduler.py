# -*- coding: utf-8 -*-
"""
空闲巡检调度器的确定性测试（不调 LLM）

覆盖 review_scheduler 的全部并发保护与让路逻辑：
    ① 环境变量配置解析（含"空闲阈值必须大于维护 idle_timeout"的强制提升）
    ② 活动追踪（touch / idle_seconds / is_idle）
    ③ 正常巡检：只读、报告落盘、批次数正确
    ④ **中途让路**：核对过程中出现新活动 → 中止并丢弃结果（不落盘）
    ⑤ 工具级互斥：已有人在巡检 → 本次直接跳过（不排队）
    ⑥ 触发条件：不够空闲 / 间隔未到 → 不跑
    ⑦ 报告轮转：只保留最近 N 份

用法:
    python -m eval.test_review_scheduler
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dba_pipeline.core.jump_axis import NodeType, RelationType
from dba_pipeline.extraction.review_scheduler import (
    IDLE_ENV, INTERVAL_ENV, KEEP_REPORTS, IdleReviewScheduler, ReviewConfig,
)
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.mcp_server import DBAServer
from dba_pipeline.source_store import save_batch

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


class _StubVectorStore:
    _content_vectors = {}


def _make_server():
    g = MemoryGraph()
    g.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    g.add_memory("n2", "用户爱喝咖啡", NodeType.STATUS)
    g.add_edge("n1", "n2", RelationType.SOCIAL)
    return DBAServer(g, yaml_path=None, dba=None, vector_store=_StubVectorStore())


def test_config():
    print("\n① 配置解析")
    os.environ.pop(IDLE_ENV, None)
    check(ReviewConfig.from_env() is None, "未设环境变量时应返回 None（默认关闭）")

    os.environ[IDLE_ENV] = "abc"
    check(ReviewConfig.from_env() is None, "非法值应返回 None")

    os.environ[IDLE_ENV] = "0"
    check(ReviewConfig.from_env() is None, "0 应视为关闭")

    os.environ[IDLE_ENV] = "600"
    os.environ[INTERVAL_ENV] = "900"
    cfg = ReviewConfig.from_env(maintenance_idle_timeout=90)
    check(cfg.idle_seconds == 600, f"正常值不该被改动: {cfg.idle_seconds}")
    check(cfg.min_interval == 900, f"间隔应读环境变量: {cfg.min_interval}")

    # 关键保护：空闲阈值必须大于维护调度器的 idle_timeout，否则两者会同时开跑
    cfg2 = ReviewConfig.from_env(maintenance_idle_timeout=90)
    os.environ[IDLE_ENV] = "60"
    cfg2 = ReviewConfig.from_env(maintenance_idle_timeout=90)
    check(cfg2.idle_seconds > 90, f"过小的空闲阈值应被提升到 >90: {cfg2.idle_seconds}")

    os.environ[IDLE_ENV] = "150"
    cfg3 = ReviewConfig.from_env(maintenance_idle_timeout=None)
    check(cfg3.idle_seconds == 150, "未提供维护超时时不做提升")
    print(f"  ✓ 解析 / 关闭 / 强制提升（60 → {cfg2.idle_seconds}）")


def test_activity():
    print("\n② 活动追踪")
    srv = _make_server()
    check(srv.is_idle(0.0), "刚构造时应算空闲 0 秒")
    check(not srv.is_idle(9999.0), "不该算作空闲 9999 秒")
    time.sleep(0.05)
    check(srv.idle_seconds() >= 0.05, "空闲时长应随时钟增长")
    srv.touch()
    check(srv.idle_seconds() < 0.05, "touch 后空闲计时应清零")
    print("  ✓ touch / idle_seconds / is_idle")


def test_normal_run():
    print("\n③ 正常巡检（只读 + 落盘）")
    tmp = Path(tempfile.mkdtemp(prefix="revsched_"))
    srv = _make_server()
    srv.dba = SimpleNamespace(source_dir=str(tmp / "sources"))
    save_batch(str(tmp / "sources"), "b-test-1", [{"ts": None, "text": "小林在加班"}])
    save_batch(str(tmp / "sources"), "b-test-2", [{"ts": None, "text": "阿哲在赶稿"}])

    calls = []

    def fake_review_sources(batch_id=None, max_batches=3):
        calls.append(batch_id)
        return {"batches": [{"batch_id": batch_id}], "summary": {"batches_checked": 1}}

    srv.review_sources = fake_review_sources

    rs = IdleReviewScheduler(srv, ReviewConfig(max_batches=2), yaml_path=str(tmp / "memory_graph.yaml"))
    nodes_before, edges_before = srv.graph.node_count, srv.graph.edge_count
    report = rs.review_now(trigger="test")

    check("skipped" not in report and "aborted" not in report, f"不该被跳过/中止: {report}")
    # 关键回归：调度器整轮持锁、又去调用同样会取锁的 review_graph，同线程重入必须放行，
    # 否则体检结果会被静默换成 {"error": "已有一次巡检正在进行"}（曾被断言宽松放过）
    check("error" not in (report["graph"] or {}), f"体检不该因重入而报错: {report['graph']}")
    check((report["graph"] or {}).get("scanned") == 2,
          f"应扫描到 2 个节点: {report['graph']}")
    check(len(report["sources"]) == 2, f"应核对 2 批: {len(report['sources'])}")
    check(sorted(calls) == ["b-test-1", "b-test-2"], f"核对的批次不对: {calls}")
    check(srv.graph.node_count == nodes_before and srv.graph.edge_count == edges_before,
          "巡检不得修改图谱")
    files = list(rs.review_dir.glob("review-*.json"))
    check(len(files) == 1, f"应落盘 1 份报告: {len(files)}")
    check(rs.stats["runs"] == 1, f"runs 计数不对: {rs.stats}")
    print(f"  ✓ 体检 + 核对 2 批, 落盘 {len(files)} 份, 图谱未变")


def test_abort_on_activity():
    print("\n④ 中途让路（有人来了 → 中止并丢弃）")
    tmp = Path(tempfile.mkdtemp(prefix="revsched_abort_"))
    srv = _make_server()
    srv.dba = SimpleNamespace(source_dir=str(tmp / "sources"))
    for i in (1, 2, 3):
        save_batch(str(tmp / "sources"), f"b-a{i}", [{"ts": None, "text": f"第{i}条"}])

    calls = []

    def fake_review_sources(batch_id=None, max_batches=3):
        calls.append(batch_id)
        # 模拟：核对第 1 批的过程中，用户发来一个新请求
        srv.touch(source="dba_query_memory")
        return {"batches": [{"batch_id": batch_id}]}

    srv.review_sources = fake_review_sources
    rs = IdleReviewScheduler(srv, ReviewConfig(max_batches=3), yaml_path=str(tmp / "memory_graph.yaml"))
    report = rs.review_now(trigger="test")

    check("aborted" in report, f"检测到新活动时应中止: {report}")
    check(len(calls) == 1, f"应在第 1 批核对后即中止，不该继续: {calls}")
    check(len(list(rs.review_dir.glob('review-*.json'))) == 0, "中止时不得落盘（半截报告会误导）")
    check(rs.stats["aborted"] == 1, f"aborted 计数不对: {rs.stats}")
    # RLock 在 Python<3.14 没有 locked()，直接试获取来判断锁是否已释放
    acquired = srv._review_running.acquire(blocking=False)
    check(acquired, "中止后必须释放互斥锁")
    if acquired:
        srv._review_running.release()
    print(f"  ✓ 核对 1 批后中止, 未落盘, 锁已释放")


def test_mutex():
    print("\n⑤ 工具级互斥")
    srv = _make_server()
    rs = IdleReviewScheduler(srv, ReviewConfig(), yaml_path=str(Path(tempfile.mkdtemp()) / "g.yaml"))
    # 另一线程持锁，模拟"已有一个巡检正在跑"。
    # 注意：_review_running 是 RLock——同线程可重入、**跨线程仍互斥**，所以这里必须换线程持有。
    held = threading.Event()
    release = threading.Event()

    def _hold():
        srv._review_running.acquire()
        held.set()
        release.wait(5)
        srv._review_running.release()

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    held.wait(5)
    try:
        r = rs.review_now(trigger="test")
        check("skipped" in r, f"已有巡检时应返回 skipped: {r}")
        check(rs.stats["skipped_busy"] == 1, f"计数不对: {rs.stats}")
    finally:
        release.set()
        t.join(5)
    print("  ✓ 第二次巡检直接跳过，不排队")


def test_should_run():
    print("\n⑥ 触发条件")
    srv = _make_server()
    rs = IdleReviewScheduler(srv, ReviewConfig(idle_seconds=100, min_interval=1000),
                             yaml_path=str(Path(tempfile.mkdtemp()) / "g.yaml"))
    srv.touch()
    check(not rs._should_run(), "刚活动过不该触发")
    srv._last_activity = time.time() - 200          # 假装已空闲 200s
    check(rs._should_run(), "空闲足够时应触发")
    rs._last_run = time.time()
    check(not rs._should_run(), "距上次巡检不足 min_interval 时不该触发")
    rs._last_run = time.time() - 2000
    check(rs._should_run(), "间隔足够且空闲时应触发")
    print("  ✓ 空闲不足 / 间隔未到 均不触发")


def test_prune():
    print("\n⑦ 报告轮转")
    tmp = Path(tempfile.mkdtemp(prefix="revsched_prune_"))
    srv = _make_server()
    rs = IdleReviewScheduler(srv, ReviewConfig(max_batches=0),
                             yaml_path=str(tmp / "memory_graph.yaml"))
    for i in range(KEEP_REPORTS + 5):
        rs.review_now(trigger="test")
        time.sleep(0.005)
    files = list(rs.review_dir.glob("review-*.json"))
    check(len(files) <= KEEP_REPORTS, f"应只保留 {KEEP_REPORTS} 份: {len(files)}")
    print(f"  ✓ 跑 {KEEP_REPORTS + 5} 次后只保留 {len(files)} 份")


def test_review_tools_share_mutex():
    print("\n⑨ 两个巡检工具共用同一把互斥锁")
    srv = _make_server()

    # 跨线程持锁，模拟"已有一个巡检在跑"
    held = threading.Event()
    release = threading.Event()

    def _hold():
        srv._review_running.acquire()
        held.set()
        release.wait(5)
        srv._review_running.release()

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    held.wait(5)
    try:
        for name in ("review_graph", "review_sources"):
            r = getattr(srv, name)()
            check(r.get("running") is True and "已有一次巡检" in str(r.get("error")),
                  f"{name} 未受 _review_running 保护: {r}")
    finally:
        release.set()
        t.join(5)

    # 锁释放后 review_sources 必须能进入实现体（本例无 dba → 返回"未开启原文存储"的提示，
    # 而不是互斥跳过），证明包装层没有把正常调用也挡掉
    r = srv.review_sources()
    check("未开启原文存储" in str(r.get("error")), f"锁释放后应进入实现体: {r}")
    print("  ✓ review_graph / review_sources 共用锁，跨线程互斥且不误挡正常调用")


def test_thread_lifecycle():
    print("\n⑧ 后台线程生命周期")
    srv = _make_server()
    rs = IdleReviewScheduler(srv, ReviewConfig(idle_seconds=0.01, min_interval=0.01,
                                               poll_seconds=0.05),
                             yaml_path=str(Path(tempfile.mkdtemp()) / "g.yaml"))
    rs.start()
    check(rs._thread is not None and rs._thread.is_alive(), "启动后线程应存活")
    deadline = time.time() + 5
    while time.time() < deadline and rs.stats["runs"] == 0:
        time.sleep(0.05)
    check(rs.stats["runs"] >= 1, f"空闲时应自动跑起来: {rs.stats}")
    t0 = time.time()
    rs.stop()
    check(time.time() - t0 < 5, "stop() 应立即返回，不阻塞在 poll 等待上")
    check(not rs._thread.is_alive(), "停止后线程应退出")
    print(f"  ✓ 自动触发 {rs.stats['runs']} 次，stop() 及时返回")


def main():
    print("# 空闲巡检调度器：确定性测试（无 LLM）")
    for fn in (test_config, test_activity, test_normal_run, test_abort_on_activity,
               test_mutex, test_should_run, test_prune, test_thread_lifecycle,
               test_review_tools_share_mutex):
        fn()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for f in FAILS:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("空闲巡检调度器测试通过 [OK]：配置/活动追踪/让路/互斥/触发/轮转/生命周期均正常。")


if __name__ == "__main__":
    main()
