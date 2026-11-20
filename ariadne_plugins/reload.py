# SPDX-License-Identifier: AGPL-3.0-only

"""插件热加载：监听 ``ARIADNE_PLUGINS`` 模块文件变化，原地替换工具表。

默认关闭，设置环境变量 ``ARIADNE_PLUGIN_RELOAD=<秒>`` 开启。热替换通过
:class:`ToolRegistry` 的锁完成，前台 ``tools/list`` / ``tools/call`` 看到的
始终是完整快照，不会读到半更新状态。
"""

import importlib
import logging
import os
import threading

from .registry import ToolRegistry

logger = logging.getLogger("mcp.plugins")


def _env_plugin_modules():
    raw = os.environ.get("ARIADNE_PLUGINS", "")
    return [m.strip() for m in raw.replace(",", " ").split() if m.strip()]


def _fingerprint() -> tuple:
    """探针：环境变量值 + 各插件模块文件 mtime。"""
    stamps = [os.environ.get("ARIADNE_PLUGINS", "")]
    for name in _env_plugin_modules():
        try:
            path = getattr(importlib.import_module(name), "__file__", None)
            stamps.append((name, os.stat(path).st_mtime_ns if path else None))
        except Exception:
            stamps.append((name, None))
    return tuple(stamps)


class PluginWatcher(threading.Thread):
    """轮询插件文件变化并热替换 ``registry``（daemon 线程）。"""

    def __init__(self, registry: ToolRegistry, interval: float = 5.0):
        super().__init__(daemon=True, name="plugin-watcher")
        self.registry = registry
        self.interval = max(1.0, interval)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def reload_now(self) -> dict:
        """立即重载一次并返回新工具表（也供测试/手动调用）。"""
        from .discovery import load_plugin_tools
        new = load_plugin_tools(reload=True)
        self.registry.replace(new)
        return new

    def run(self) -> None:
        last = _fingerprint()
        while not self._stop.wait(self.interval):
            try:
                fp = _fingerprint()
            except Exception as e:
                logger.warning("插件探针失败（忽略）: %s", e)
                continue
            if fp == last:
                continue
            last = fp
            try:
                new = self.reload_now()
                logger.info("插件已热加载: %s", " / ".join(new) or "(无)")
            except Exception as e:
                logger.warning("插件热加载失败（保留旧插件）: %s", e)
