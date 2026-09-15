# SPDX-License-Identifier: AGPL-3.0-only

"""WebUI 可观测性 — 日志总线。

汇聚两类运行时信息，供面板实时展示：
1. 应用运行日志：通过 ``logging.Handler`` 捕获进程内各模块的日志记录；
2. 操作日志（oplog）：后台线程 tail ``operations.log``，捕捉 MCP 进程等外部写入。

两类记录统一进入 :class:`LogBus` 的环形缓冲，并向所有 SSE 订阅者实时推送。
每条记录都带 ``channel`` 字段（``"log"`` / ``"oplog"``），前端据此分流。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

# 单条日志消息的最大长度，避免异常堆栈撑爆缓冲区
_MAX_MESSAGE = 4000

# 这些 logger 噪声大且无助于排查业务链路，直接忽略
_IGNORED_LOGGERS = ("uvicorn.access",)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def default_log_file_path(yaml_path: Optional[str] = None) -> str:
    """跨进程服务日志文件路径（可用环境变量 ``ARIADNE_LOG_FILE`` 覆盖）。

    服务进程（MCP / ALM 等）把运行日志以 JSONL 追加写入该文件；
    WebUI 面板 tail 该文件即可聚合展示**其它进程**的日志。
    默认与图谱同目录：``<yaml 目录>/ariadne.log``。
    """
    p = os.environ.get("ARIADNE_LOG_FILE")
    if p:
        return p
    if yaml_path:
        return os.path.join(os.path.dirname(os.path.abspath(yaml_path)), "ariadne.log")
    return "ariadne.log"


class LogBus:
    """线程安全的环形日志缓冲 + SSE 订阅分发。

    写入可能来自任意线程（logging handler / tail 线程），而订阅者队列绑定在
    asyncio 事件循环上，因此推送时通过 ``loop.call_soon_threadsafe`` 转交。
    """

    def __init__(self, capacity: int = 1000):
        self._records: deque = deque(maxlen=capacity)
        self._subscribers: set = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """由应用 startup 调用，绑定主事件循环。"""
        self._loop = loop

    # ---- 写入 ----

    def publish(self, record: dict) -> None:
        record.setdefault("ts", _now_iso())
        with self._lock:
            self._records.append(record)
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        loop = self._loop
        for queue in subscribers:
            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(self._offer, queue, record)
                except RuntimeError:
                    pass
            else:
                self._offer(queue, record)

    @staticmethod
    def _offer(queue: asyncio.Queue, record: dict) -> None:
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            # 订阅端消费过慢：丢弃最旧的一条，保证流不阻塞
            try:
                queue.get_nowait()
                queue.put_nowait(record)
            except Exception:
                pass

    # ---- 订阅 ----

    def subscribe(self, maxsize: int = 2000) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._records)

    # ---- 读取 ----

    def snapshot(self, channel: Optional[str] = None, level: Optional[str] = None,
                 query: Optional[str] = None, tail: int = 300) -> list:
        with self._lock:
            rows = list(self._records)[-max(tail, 1):]
        return [r for r in rows if _match(r, channel, level, query)]


def _match(record: dict, channel: Optional[str], level: Optional[str],
           query: Optional[str]) -> bool:
    if channel and record.get("channel") != channel:
        return False
    if level:
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        if order.get(record.get("level", "INFO"), 0) < order.get(level.upper(), 0):
            return False
    if query:
        text = json.dumps(record, ensure_ascii=False).lower()
        if query.lower() not in text:
            return False
    return True


class BusLogHandler(logging.Handler):
    """把 logging 记录汇入 LogBus 的 handler。"""

    def __init__(self, bus: LogBus, level: int = logging.INFO):
        super().__init__(level)
        self.bus = bus
        self._formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.startswith(_IGNORED_LOGGERS):
                return
            message = self._formatter.format(record)
            exc = getattr(record, "exc_info", None)
            if exc:
                message = f"{message} | {self._formatter.formatException(exc)}"
            self.bus.publish({
                "channel": "log",
                "ts": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
                "level": record.levelname,
                "logger": record.name,
                "message": message[:_MAX_MESSAGE],
            })
        except Exception:
            self.handleError(record)


class JsonlTailer(threading.Thread):
    """后台线程 tail 一个 JSONL 文件，把新增记录推入 LogBus。

    - 只推送启动之后新增的记录（可选用 ``seed`` 预读最后 N 条历史进缓冲）；
    - ``channel`` 标记记录来源通道（``oplog`` / ``log``），供前端分流；
    - 文件不存在时等待其出现（服务进程稍后启动也能接上）。
    """

    def __init__(self, bus: LogBus, path: str, channel: str,
                 interval: float = 0.8, seed: int = 0, name: str = "jsonl-tailer"):
        super().__init__(daemon=True, name=name)
        self.bus = bus
        self.path = path
        self.channel = channel
        self.interval = interval
        self.seed = seed
        self._stop_event = threading.Event()
        self._pos = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._pos = self._file_size()
        if self.seed > 0:
            self._seed(self.seed)
        while not self._stop_event.wait(self.interval):
            self._poll()

    def _file_size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def _seed(self, count: int) -> None:
        """把文件末尾 count 行读入缓冲（历史记录，供面板首屏展示）。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in deque(f, maxlen=count):
                    self._publish_line(line)
        except OSError:
            pass

    def _poll(self) -> None:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            self._pos = 0
            return
        if size < self._pos:  # 文件被截断/轮转
            self._pos = 0
        if size == self._pos:
            return
        # 以二进制读取：_pos 是字节偏移（来自 getsize），文本模式的 tell() 语义不同，
        # 混用会在多字节字符处读偏。
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                chunk = f.read()
        except OSError:
            return
        # 只消费到最后一个换行符为止：末尾可能正被写入方写着（半行）。
        # 若此刻推进 _pos，半行会被当解析失败丢掉，写入方补全后也只剩后半截，
        # 这条记录就永久丢了。留在原地，下一轮再读。
        last_newline = chunk.rfind(b"\n")
        if last_newline < 0:
            return
        consumed = chunk[:last_newline + 1]
        self._pos += len(consumed)
        for line in consumed.splitlines():
            self._publish_line(line.decode("utf-8", "replace"))

    def _publish_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(record, dict):
            record["channel"] = self.channel
            self.bus.publish(record)


class OplogTailer(JsonlTailer):
    """专职 tail 操作日志（``operations.log``）的 tailer。"""

    def __init__(self, bus: LogBus, path: str, interval: float = 0.8):
        super().__init__(bus, path, channel="oplog", interval=interval, name="oplog-tailer")


def install_log_capture(bus: LogBus, level: int = logging.INFO) -> BusLogHandler:
    """在 root logger 上挂载捕获 handler。

    为了让 WebUI 能看到 INFO 级业务日志，需要把 root 级别降到 ``level``；同时把
    已有的控制台 handler 抬到 WARNING 以上，避免 INFO 级库日志刷屏终端。
    """
    handler = BusLogHandler(bus, level=level)
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, logging.StreamHandler) and not isinstance(existing, BusLogHandler):
            if existing.level < logging.WARNING:
                existing.setLevel(logging.WARNING)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return handler


class JsonlFileHandler(logging.Handler):
    """把日志记录以 JSONL 追加写入共享文件，供面板跨进程聚合读取。

    与 :class:`BusLogHandler` 的区别：后者只在自己的进程内可见，前者写盘后
    可被 WebUI 面板（另一个进程）tail 到，因此服务进程应使用本 handler。
    """

    def __init__(self, path: str, proc: str, level: int = logging.INFO):
        super().__init__(level)
        self.path = path
        self.proc = proc
        self._formatter = logging.Formatter()
        self._write_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.startswith(_IGNORED_LOGGERS):
                return
            message = self._formatter.format(record)
            exc = getattr(record, "exc_info", None)
            if exc:
                message = f"{message} | {self._formatter.formatException(exc)}"
            entry = {
                "ts": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
                "level": record.levelname,
                "logger": record.name,
                "proc": self.proc,
                "message": message[:_MAX_MESSAGE],
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._write_lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            self.handleError(record)


def install_service_logging(proc: str, path: str, level: int = logging.INFO) -> JsonlFileHandler:
    """服务进程启用共享 JSONL 日志（供 WebUI 面板聚合展示）。

    会把 root 级别降到 ``level``（默认 INFO）以便捕获业务日志，同时把控制台
    handler 抬到 WARNING 以上，避免 INFO 级依赖库日志刷屏终端。
    """
    handler = JsonlFileHandler(path, proc, level=level)
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, logging.StreamHandler) and not isinstance(existing, JsonlFileHandler):
            if existing.level < logging.WARNING:
                existing.setLevel(logging.WARNING)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return handler
