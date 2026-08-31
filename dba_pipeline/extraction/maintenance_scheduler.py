"""
异步维护调度器：限流器，阻止 LLM 逐条处理记忆。

触发出口：
  - 被动出口：空闲超时（用户挂机）、会话结束（意外关闭）
  - 主动出口：DBA 累积判定 E 次"需要维护"后，缓冲区积累 B 条 → 触发一次
  - 自动跳过：连续跳过 A 次后，不再调用 LLM，直接清空缓冲区

与 DBA 判断的配合：
  DBA 判"需要维护" → 计入 meaningful_count → 达到 E 后进入主动模式
  DBA 判"跳过"     → 不计入，连续 A 次后进入自动跳过模式
  主动模式下也不逐条触发，而是积累 B 条后再放行

核心设计：调度器是限流器而非加速器。
  - 被动模式：只靠空闲/会话结束触发
  - 主动模式：不等空闲，批量 B 条触发
  - 自动跳过：连续空转太多 → 直接跳过，不再浪费 LLM 调用
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    PROACTIVE = auto()    # 主动模式批量
    SESSION_END = auto()  # 会话结束
    IDLE_TIMEOUT = auto() # 空闲超时
    AUTO_SKIP = auto()    # 自动跳过（连续空转过多）
    MANUAL = auto()       # 手动触发


@dataclass
class ScheduleConfig:
    """调度配置"""

    # 空闲触发器：用户停止对话超过此秒数后触发（0=关闭）
    idle_timeout: float = 90.0

    # 最大合并轮数：单次维护最多合并多少轮对话
    max_batch_rounds: int = 10

    # 会话结束是否强制触发
    flush_on_session_end: bool = True

    # ── 主动模式 ──
    # DBA 累积判定"需要维护"多少次后，进入主动模式
    proactive_entry_threshold: int = 3
    # 主动模式下缓冲区积累多少条对话后触发一次维护
    proactive_batch_size: int = 6

    # ── 自动跳过 ──
    # DBA 连续判定"跳过"多少次后，不再调用 LLM，直接清空缓冲区
    auto_skip_threshold: int = 5


class MaintenanceScheduler:
    """记忆图谱异步维护调度器

    使用方式：
        scheduler = MaintenanceScheduler(dba, config)
        scheduler.start()
        scheduler.on_conversation("user: 今天好累...")
        scheduler.on_session_end()
        scheduler.stop()
    """

    def __init__(
        self,
        dba,
        config: ScheduleConfig = None,
        on_maintenance_done: Optional[Callable] = None,
    ):
        self.dba = dba
        self.config = config or ScheduleConfig()
        self.on_maintenance_done = on_maintenance_done

        self._buffer: List[str] = []
        self._lock = threading.Lock()

        self._last_conversation_time: float = 0.0
        self._flushing: bool = False
        self._pending_flush: bool = False
        self._running: bool = False

        self._meaningful_count: int = 0
        self._consecutive_skips: int = 0

        self._idle_timer: Optional[threading.Timer] = None
        self._worker_thread: Optional[threading.Thread] = None

        self.stats = {
            "total_flushes": 0,
            "total_maintenances": 0,
            "total_skipped": 0,
            "total_auto_skipped": 0,
            "total_conversations": 0,
            "proactive_triggers": 0,
            "passive_triggers": 0,
            "trigger_reasons": {},
        }

    # ---- 公共接口 ----

    def on_conversation(self, conversation: str):
        """对话完成后调用。"""
        with self._lock:
            self._buffer.append(conversation)
            self._last_conversation_time = time.time()
            self.stats["total_conversations"] += 1
            self._cancel_idle_timer()

            if self._is_proactive():
                self._trigger(TriggerReason.PROACTIVE)
                return

            self._start_idle_timer()

    def on_session_end(self):
        with self._lock:
            if self.config.flush_on_session_end and self._buffer:
                self._trigger(TriggerReason.SESSION_END)

    def flush_now(self):
        with self._lock:
            self._trigger(TriggerReason.MANUAL)

    def start(self):
        self._running = True
        logger.info(
            f"MaintenanceScheduler 启动: "
            f"idle={self.config.idle_timeout}s, "
            f"entry={self.config.proactive_entry_threshold}, "
            f"batch={self.config.proactive_batch_size}, "
            f"auto_skip={self.config.auto_skip_threshold}"
        )

    def stop(self):
        self._running = False
        self._cancel_idle_timer()
        with self._lock:
            if self._buffer:
                self._trigger(TriggerReason.MANUAL)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=120)
        logger.info(
            f"MaintenanceScheduler 停止: "
            f"{self.stats['total_conversations']} 对话, "
            f"{self.stats['total_flushes']} 批 "
            f"(有效{self.stats['total_maintenances']}, "
            f"跳过{self.stats['total_skipped']}, "
            f"自动跳过{self.stats['total_auto_skipped']})"
        )

    # ---- 状态持久化 ----

    def save_state(self) -> dict:
        """导出调度器运行时状态"""
        return {
            "_buffer": list(self._buffer),
            "_meaningful_count": self._meaningful_count,
            "_consecutive_skips": self._consecutive_skips,
            "stats": dict(self.stats),
        }

    def load_state(self, state: dict):
        """恢复调度器运行时状态"""
        self._buffer = list(state.get("_buffer", []))
        self._meaningful_count = state.get("_meaningful_count", 0)
        self._consecutive_skips = state.get("_consecutive_skips", 0)
        if "stats" in state:
            for k in self.stats:
                self.stats[k] = state["stats"].get(k, self.stats[k])

    # ---- 内部方法 ----

    def _is_proactive(self) -> bool:
        """主动模式：已进入 + 缓冲够一批 + 不在维护中"""
        return (
            self.config.proactive_entry_threshold > 0
            and self._meaningful_count >= self.config.proactive_entry_threshold
            and not self._flushing
            and len(self._buffer) >= self.config.proactive_batch_size
        )

    def _trigger(self, reason: TriggerReason):
        if not self._buffer:
            return
        if self._flushing:
            self._pending_flush = True
            return

        conversations = self._buffer[:]
        self._buffer.clear()
        self._pending_flush = False
        self._flushing = True

        self.stats["total_flushes"] += 1
        self.stats["trigger_reasons"][reason.name] = \
            self.stats["trigger_reasons"].get(reason.name, 0) + 1

        if reason == TriggerReason.PROACTIVE:
            self.stats["proactive_triggers"] += 1
        else:
            self.stats["passive_triggers"] += 1

        self._worker_thread = threading.Thread(
            target=self._do_flush,
            args=(conversations, reason),
            daemon=True,
        )
        self._worker_thread.start()

    def _do_flush(self, conversations: List[str], reason: TriggerReason):
        batch_count = len(conversations)
        logger.info(
            f"[DBA 维护] {reason.name}, {batch_count} 轮, "
            f"连续跳过={self._consecutive_skips}"
        )

        try:
            if batch_count > self.config.max_batch_rounds:
                logger.warning(f"截断 {batch_count} → {self.config.max_batch_rounds} 轮")
                conversations = conversations[-self.config.max_batch_rounds:]

            # P3a: 自动跳过 — 连续空转太多，不再调用 LLM
            if reason not in (TriggerReason.SESSION_END, TriggerReason.MANUAL):
                with self._lock:
                    if self._consecutive_skips >= self.config.auto_skip_threshold:
                        self.stats["total_auto_skipped"] += 1
                        logger.info(
                            f"[DBA 维护] 自动跳过（连续 {self._consecutive_skips} 次空转），"
                            f"丢弃 {batch_count} 轮对话"
                        )
                        return

            merged = "\n\n".join(conversations)
            result = self.dba.maintain(merged)

            ops = result.get("ops", {})
            node_ops = ops.get("node_ops", [])
            edge_ops = ops.get("edge_ops", [])
            was_skipped = not node_ops and not edge_ops

            with self._lock:
                if was_skipped:
                    self._consecutive_skips += 1
                    self.stats["total_skipped"] += 1
                    logger.info(
                        f"[DBA 维护] 跳过（连续 {self._consecutive_skips} 次）"
                    )
                else:
                    self._consecutive_skips = 0
                    self.stats["total_maintenances"] += 1
                    self._meaningful_count += 1
                    creates = sum(1 for op in node_ops if op.get("action") == "create")
                    deprecates = sum(1 for op in node_ops if op.get("action") == "deprecate")
                    logger.info(
                        f"[DBA 维护] 有效 ({self._meaningful_count}/{self.config.proactive_entry_threshold}): "
                        f"+{creates}节点 -{deprecates}废弃 "
                        f"+{len([o for o in edge_ops if o.get('action')=='create'])}边 "
                        f"-{len([o for o in edge_ops if o.get('action')=='delete'])}边"
                    )

            if self.on_maintenance_done:
                self.on_maintenance_done(result)

        except Exception as e:
            logger.error(f"[DBA 维护] 失败: {e}", exc_info=True)
        finally:
            self._on_flush_done()

    def _on_flush_done(self):
        """维护完成后的清理"""
        with self._lock:
            self._flushing = False
            if self._pending_flush and self._buffer:
                self._trigger(TriggerReason.PROACTIVE)

    def _start_idle_timer(self):
        if self.config.idle_timeout <= 0:
            return
        self._cancel_idle_timer()
        self._idle_timer = threading.Timer(self.config.idle_timeout, self._on_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle(self):
        with self._lock:
            if not self._running:
                return
            if time.time() - self._last_conversation_time >= self.config.idle_timeout and self._buffer:
                self._trigger(TriggerReason.IDLE_TIMEOUT)

    def _cancel_idle_timer(self):
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    # ---- 状态查询 ----

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_flushing(self) -> bool:
        with self._lock:
            return self._flushing

    @property
    def is_proactive(self) -> bool:
        with self._lock:
            return self._meaningful_count >= self.config.proactive_entry_threshold

    @property
    def meaningful_count(self) -> int:
        with self._lock:
            return self._meaningful_count

    @property
    def consecutive_skips(self) -> int:
        with self._lock:
            return self._consecutive_skips

    def get_stats(self) -> Dict:
        with self._lock:
            return dict(self.stats)
