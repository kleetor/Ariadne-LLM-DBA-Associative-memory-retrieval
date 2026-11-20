# SPDX-License-Identifier: AGPL-3.0-only

"""线程安全、可热替换的插件工具表。"""

import threading


class ToolRegistry:
    """插件工具注册表。

    暴露 ``get/snapshot/replace`` 等最小接口，用一把 ``RLock`` 串行化读写，
    从而支持「后台线程热替换 + 前台 MCP 分派并发读取」。``create_mcp_server``
    内部按鸭子类型同时接受普通 dict 与本类。
    """

    def __init__(self, tools=None):
        self._tools = dict(tools or {})
        self._lock = threading.RLock()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._tools)

    def get(self, name):
        with self._lock:
            return self._tools.get(name)

    def __contains__(self, name) -> bool:
        with self._lock:
            return name in self._tools

    def replace(self, tools: dict) -> None:
        """整体替换工具表（原子操作，热加载用）。"""
        with self._lock:
            self._tools = dict(tools or {})

    def names(self) -> list:
        with self._lock:
            return list(self._tools)
