# SPDX-License-Identifier: AGPL-3.0-only

"""插件工具契约：一个插件工具只需要实现一个「同步、dict 进 dict 出」的 handler。"""

import logging
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolSpec:
    """一个可注册到 Ariadne MCP 服务的工具。

    字段与 MCP 的 Tool 一一对应，外加 `redact` 决定操作日志是否脱敏。
    handler 是**同步阻塞**函数（与内置 ``DBAServer`` 方法同形），
    由 MCP 侧统一丢进线程池执行。
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]
    redact: bool = False


@dataclass
class PluginContext:
    """传给插件生命周期钩子的上下文；只暴露必要能力。"""

    dba: object = None
    yaml_path: str | None = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.plugins"))
