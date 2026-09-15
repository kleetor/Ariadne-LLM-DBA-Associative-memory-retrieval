# SPDX-License-Identifier: AGPL-3.0-only

"""跨进程文件锁。

用于串行化多个进程（WebUI 面板、MCP 等）对同一份 YAML checkpoint 的
「读-改-写」临界区，避免相互覆盖（丢失更新）。

基于 OS advisory lock（POSIX ``fcntl.flock`` / Windows ``msvcrt.locking``）：
- 任一进程退出或崩溃时由内核自动释放，不存在手写 lockfile 的残留死锁问题；
- 同一进程内可重入（同一线程嵌套获取不会自锁），跨线程则互斥。
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class FileLock:
    """基于 ``<path>`` 的跨进程互斥锁。

    Args:
        path: 锁文件路径（通常为 ``<yaml>.lock``）
        timeout: 获取锁的最长等待秒数，超时抛 ``TimeoutError``
        poll: 轮询间隔秒数
    """

    def __init__(self, path: str, timeout: float = 30.0, poll: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._fd = None
        self._depth = 0
        self._local = threading.RLock()  # 保证同进程内的线程互斥 + 可重入

    # ---- 获取/释放 ----

    def acquire(self, timeout: float = None) -> None:
        self._local.acquire()
        if self._depth > 0:          # 同线程重入：只计数
            self._depth += 1
            return
        limit = self.timeout if timeout is None else timeout
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            self._local.release()
            raise

        deadline = time.monotonic() + limit
        while True:
            try:
                self._try_lock(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    self._local.release()
                    raise TimeoutError(f"获取文件锁超时（{limit:.0f}s）: {self.path}")
                time.sleep(self.poll)
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth > 0:
            self._local.release()
            return
        fd, self._fd = self._fd, None
        try:
            if fd is not None:
                try:
                    self._unlock(fd)
                except OSError:
                    pass
                os.close(fd)
        finally:
            self._local.release()

    # ---- 上下文管理器 ----

    @contextmanager
    def hold(self, timeout: float = None):
        self.acquire(timeout)
        try:
            yield self
        finally:
            self.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    # ---- 平台实现 ----

    @staticmethod
    def _try_lock(fd: int) -> None:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(fd: int) -> None:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)


def lock_path_for(yaml_path: str) -> str:
    """返回某份 YAML checkpoint 对应的锁文件路径（统一绝对路径，跨进程一致）。"""
    return os.path.abspath(yaml_path) + ".lock"
