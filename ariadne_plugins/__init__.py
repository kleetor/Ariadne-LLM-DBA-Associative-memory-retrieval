# SPDX-License-Identifier: AGPL-3.0-only

"""Ariadne MCP 插件包。

插件只需实现 :class:`ToolSpec`（同步 handler），通过 ``ARIADNE_PLUGINS``
环境变量或 entry-points 组 ``ariadne.mcp.tools`` 被 ``dba_pipeline.mcp_server``
在启动时发现，无需修改 MCP 核心代码。

可选能力：
- 生命周期钩子 ``on_startup(ctx)`` / ``on_shutdown(ctx)``；
- 热加载 ``ARIADNE_PLUGIN_RELOAD=<秒>``（:class:`PluginWatcher`）。
"""

from .api import PluginContext, ToolSpec
from .discovery import PluginBundle, load_plugin_tools, load_plugins
from .registry import ToolRegistry
from .reload import PluginWatcher

__all__ = [
    "PluginContext",
    "ToolSpec",
    "PluginBundle",
    "load_plugin_tools",
    "load_plugins",
    "ToolRegistry",
    "PluginWatcher",
]
