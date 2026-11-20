# SPDX-License-Identifier: AGPL-3.0-only

"""示例插件：演示如何注册一个最小的第 12 个工具，并声明生命周期钩子。

用法::

    ARIADNE_PLUGINS=ariadne_plugins.examples.hello ariadne-mcp ...

然后 Agent 就能在 tools/list 里看到 ``plugin_hello``。设置
``ARIADNE_PLUGIN_RELOAD=3`` 后，修改本文件会在约 3 秒内被热加载。
"""

import logging

from ariadne_plugins import ToolSpec

logger = logging.getLogger("mcp.plugins.example")


def _hello(name: str = "world") -> dict:
    return {"message": f"hello, {name}"}


def on_startup(ctx) -> None:
    logger.info("plugin_hello 已启动（yaml=%s）", getattr(ctx, "yaml_path", None))


def on_shutdown(ctx) -> None:
    logger.info("plugin_hello 已卸载")


REGISTER = [
    ToolSpec(
        name="plugin_hello",
        description="示例插件：回显问候语，用于演示插件工具注册。",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "称呼，默认 world"}},
        },
        handler=_hello,
        redact=False,
    ),
]
