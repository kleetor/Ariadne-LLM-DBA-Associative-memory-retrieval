# SPDX-License-Identifier: AGPL-3.0-only

"""
空闲时段自动巡检（离线检测 + 并发保护）

为什么要它：`dba_review_graph` / `dba_review_sources` 目前只能等上层 agent 主动调用，
而 agent 只在**用户说了话**之后才有机会调工具——也就是说巡检永远发生在"刚被使用过"的时刻，
既不是空闲时段，也可能与正在进行的检索/维护抢资源。

本模块把巡检挪到**没人在用的时段**跑，并保证：
1. **不抢占在线请求**：距最后一次活动超过 `idle_seconds` 才启动；
2. **一有人来就让路**：逐批核对（review_sources 会调 LLM）之间检查活动时间戳，
   期间若有任何工具被调用就**中止并丢弃本次结果**（不落盘），下次重来；
3. **不重叠**：复用 server 的 `_review_running` 非阻塞互斥——手动调用与自动巡检互相排斥。
   该锁是 `RLock`：本调度器整轮持锁、内部又调用同样会取锁的 `review_graph()`，同线程重入
   必须放行（否则体检结果会被静默替换成"已有一次巡检正在进行"）；跨线程仍然互斥；
4. **不与 DBA 维护撞车**：空闲阈值强制大于维护调度器的 `idle_timeout`
   （维护是在空闲后 90s 触发的，巡检若也用 90s 就会同时开跑）；
5. **不写图**：只读 `review_*`，报告落 `<图谱目录>/reviews/`，落盘仍由上层用 `dba_intervene`。

默认**关闭**（`ARIADNE_REVIEW_IDLE` 未设置时不启动）——自动巡检会消耗 LLM 额度，
是否开启属于部署决策。
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 环境变量：距最后活动多少秒算空闲（0/未设 = 关闭自动巡检）
IDLE_ENV = "ARIADNE_REVIEW_IDLE"
# 环境变量：两次自动巡检的最小间隔（秒）
INTERVAL_ENV = "ARIADNE_REVIEW_INTERVAL"
# 环境变量：单次巡检最多核对多少批原文（0 = 只做确定性体检，不调 LLM）
SOURCES_ENV = "ARIADNE_REVIEW_SOURCES"
# 保留的报告份数
KEEP_REPORTS = 20


@dataclass
class ReviewConfig:
    """空闲巡检配置"""

    # 距最后一次活动超过此秒数才算空闲。**必须大于维护调度器的 idle_timeout**，
    # 否则会和"空闲 90s 触发的 DBA 维护"同时开跑。
    idle_seconds: float = 300.0
    # 两次自动巡检的最小间隔，避免长时间挂机时反复烧 LLM
    min_interval: float = 1800.0
    # 轮询周期
    poll_seconds: float = 30.0
    # 单次巡检最多核对多少批原文；0 = 只做确定性体检（dba_review_graph），不调 LLM
    max_batches: int = 3

    @classmethod
    def from_env(cls, maintenance_idle_timeout: Optional[float] = None) -> Optional["ReviewConfig"]:
        """从环境变量构造；未开启返回 None"""
        raw = os.environ.get(IDLE_ENV, "").strip()
        if not raw:
            return None
        try:
            idle = float(raw)
        except ValueError:
            logger.warning(f"{IDLE_ENV}={raw!r} 不是合法秒数，自动巡检不启动")
            return None
        if idle <= 0:
            return None
        if maintenance_idle_timeout is not None and idle <= maintenance_idle_timeout:
            clamped = maintenance_idle_timeout + 60.0
            logger.warning(
                f"{IDLE_ENV}={idle}s 不大于维护调度器的 idle_timeout="
                f"{maintenance_idle_timeout}s，会与 DBA 维护同时开跑；已提升到 {clamped}s"
            )
            idle = clamped
        try:
            interval = float(os.environ.get(INTERVAL_ENV, "1800"))
        except ValueError:
            interval = 1800.0
        try:
            batches = int(os.environ.get(SOURCES_ENV, "3"))
        except ValueError:
            batches = 3
        return cls(idle_seconds=idle, min_interval=max(interval, 60.0), max_batches=max(batches, 0))


def default_review_dir(yaml_path) -> Path:
    """报告目录：与图谱 YAML 同目录，便于随图谱一起备份/清理"""
    base = Path(yaml_path).resolve().parent if yaml_path else Path("data")
    return base / "reviews"


class IdleReviewScheduler:
    """空闲时自动跑巡检；一有活动就让路。

    使用方式：
        rs = IdleReviewScheduler(server, config, yaml_path=args.yaml)
        rs.start()
        ...
        rs.stop()
    """

    def __init__(self, server, config: ReviewConfig = None, yaml_path=None,
                 on_report: Optional[Callable] = None):
        self.server = server
        self.config = config or ReviewConfig()
        self.review_dir = default_review_dir(yaml_path)
        self.on_report = on_report

        self._running = False
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_run: float = 0.0
        self.stats = {"runs": 0, "aborted": 0, "skipped_busy": 0, "failures": 0}

    # ---- 生命周期 ----

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="idle-review", daemon=True)
        self._thread.start()
        logger.info(
            f"空闲巡检启动: idle>{self.config.idle_seconds}s, "
            f"间隔 {self.config.min_interval}s, 每轮最多核对 {self.config.max_batches} 批, "
            f"报告目录 {self.review_dir}"
        )

    def stop(self, timeout: float = 30.0):
        self._running = False
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            f"空闲巡检停止: 运行 {self.stats['runs']} 次, "
            f"让路中止 {self.stats['aborted']} 次, 互斥跳过 {self.stats['skipped_busy']} 次, "
            f"失败 {self.stats['failures']} 次"
        )

    # ---- 主循环 ----

    def _loop(self):
        while self._running:
            # 可中断等待：stop() 设置 _wake 后立即退出，不必等满 poll_seconds
            self._wake.wait(self.config.poll_seconds)
            self._wake.clear()
            if not self._running:
                break
            try:
                if not self._should_run():
                    continue
                self._last_run = time.time()
                report = self.review_now(trigger="idle")
                if self.on_report:
                    try:
                        self.on_report(report)
                    except Exception as e:  # 回调失败不影响巡检循环
                        logger.warning(f"巡检回调失败: {e}")
            except Exception as e:
                self.stats["failures"] += 1
                logger.warning(f"空闲巡检异常: {e}", exc_info=True)

    def _should_run(self) -> bool:
        if self.server is None:
            return False
        if not self.server.is_idle(self.config.idle_seconds):
            return False
        if self._last_run and (time.time() - self._last_run) < self.config.min_interval:
            return False
        return True

    # ---- 单次巡检 ----

    def review_now(self, trigger: str = "manual") -> dict:
        """跑一次巡检。全程只读，报告落盘；期间若出现活动则中止并丢弃结果。"""
        # 与手动调用 / 另一次自动巡检互斥（非阻塞：拿不到就直接放弃，不排队）
        if not self.server._review_running.acquire(blocking=False):
            self.stats["skipped_busy"] += 1
            return {"skipped": "已有巡检在进行中", "trigger": trigger}

        try:
            started_activity = self.server.last_activity()
            started_at = time.time()
            report = {
                "trigger": trigger,
                "started_at": int(started_at * 1000),
                "idle_seconds_at_start": round(self.server.idle_seconds(), 1),
                "graph": None,
                "sources": [],
            }

            # ① 确定性体检：纯计算，最便宜，先做
            try:
                report["graph"] = self.server.review_graph()
            except Exception as e:
                report["graph"] = {"error": str(e)}
                logger.warning(f"图谱体检失败: {e}")

            # ② 溯源核对：逐批调 LLM。每批前检查活动——一有人来就中止并**丢弃**本次结果，
            #    因为半截报告可能让人误判（比如只核了一半批次就以为"都干净"）。
            if self.config.max_batches > 0:
                for batch in self._pending_batches():
                    if self.server.last_activity() > started_activity:
                        self.stats["aborted"] += 1
                        logger.info(
                            f"空闲巡检让路：核对第 {len(report['sources']) + 1} 批前检测到新活动，"
                            "丢弃本次结果，等下次空闲重来"
                        )
                        return {"aborted": "巡检期间出现新活动，已丢弃本次结果", "trigger": trigger}
                    try:
                        report["sources"].append(
                            self.server.review_sources(batch_id=batch["batch_id"]))
                    except Exception as e:
                        report["sources"].append({"batch_id": batch.get("batch_id"), "error": str(e)})
                        logger.warning(f"溯源核对失败（{batch.get('batch_id')}）: {e}")

            report["finished_at"] = int(time.time() * 1000)
            report["duration_s"] = round(time.time() - started_at, 2)
            self.stats["runs"] += 1
            self._persist(report)
            return report
        finally:
            self.server._review_running.release()

    def _pending_batches(self):
        """待核对的批次（按时间倒序）"""
        source_dir = getattr(getattr(self.server, "dba", None), "source_dir", None)
        if not source_dir:
            return []
        from dba_pipeline.source_store import list_batches
        return list_batches(source_dir, limit=self.config.max_batches)

    # ---- 报告落盘 ----

    def _persist(self, report: dict) -> Optional[Path]:
        """只落**报告**，不碰图谱。失败不抛出（巡检不该影响主流程）。"""
        try:
            self.review_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(report["started_at"] / 1000))
            path = self.review_dir / f"review-{stamp}-{report.get('trigger', 'x')}.json"
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            self._prune()
            logger.info(f"空闲巡检报告已落盘: {path}")
            return path
        except Exception as e:
            logger.warning(f"巡检报告落盘失败: {e}")
            return None

    def _prune(self):
        files = sorted(self.review_dir.glob("review-*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[KEEP_REPORTS:]:
            try:
                old.unlink()
            except OSError:
                continue
