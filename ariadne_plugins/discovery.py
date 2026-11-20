# SPDX-License-Identifier: AGPL-3.0-only

"""插件发现。

两种来源（先环境变量、后 entry-points）：

1. ``ARIADNE_PLUGINS``：逗号/空白分隔的模块路径。每个模块可暴露：
   - ``REGISTER: list[ToolSpec]`` 或 ``register(ctx) / register()``；
   - 可选的 ``on_startup(ctx)`` / ``on_shutdown(ctx)`` 生命周期钩子。
2. entry-points 组 ``ariadne.mcp.tools``：安装态使用，加载结果同样可以是
   ``list[ToolSpec]`` 或返回它的可调用对象。

单个插件失败只告警，不影响其它插件与内置工具；未配置时返回空 bundle。
"""

import importlib
import inspect
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("mcp.plugins")

_ENTRY_POINT_GROUP = "ariadne.mcp.tools"


@dataclass
class PluginBundle:
    tools: dict = field(default_factory=dict)
    startups: list = field(default_factory=list)
    shutdowns: list = field(default_factory=list)


def _env_plugin_modules():
    raw = os.environ.get("ARIADNE_PLUGINS", "")
    return [m.strip() for m in raw.replace(",", " ").split() if m.strip()]


def _merge(specs, tools: dict, source: str) -> None:
    if not specs:
        return
    try:
        iterable = list(specs)
    except TypeError:
        logger.warning("插件 %s 返回的不是可迭代的工具列表，跳过", source)
        return
    for spec in iterable:
        name = getattr(spec, "name", None)
        if not name:
            logger.warning("插件 %s 返回了无 name 的工具，跳过", source)
            continue
        if name in tools:
            logger.warning("插件工具重名 %s（来自 %s），保留先注册的", name, source)
            continue
        tools[name] = spec


def _call_register(module, ctx):
    fn = getattr(module, "register", None)
    if not callable(fn):
        return None
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 0
    try:
        return fn(ctx) if n >= 1 else fn()
    except Exception as e:
        logger.warning("插件 %s 的 register() 失败，跳过: %s", module.__name__, e)
        return None


def _load_module_plugin(mod_path: str, bundle: PluginBundle, ctx, reload: bool = False) -> None:
    try:
        module = importlib.import_module(mod_path)
        if reload:
            module = importlib.reload(module)
    except Exception as e:
        logger.warning("插件模块加载失败，跳过 %s: %s", mod_path, e)
        return

    specs = getattr(module, "REGISTER", None)
    if specs is None:
        specs = _call_register(module, ctx)
    _merge(specs, bundle.tools, mod_path)

    for hook_name, bucket in (("on_startup", bundle.startups),
                              ("on_shutdown", bundle.shutdowns)):
        hook = getattr(module, hook_name, None)
        if callable(hook):
            bucket.append(hook)


def _entry_point_plugins():
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python 3.10+ 一定有
        return []
    try:
        eps = entry_points()
        select = getattr(eps, "select", None)
        if select is not None:
            return list(select(group=_ENTRY_POINT_GROUP))
        return list(eps.get(_ENTRY_POINT_GROUP, []))  # 旧版兼容
    except Exception as e:
        logger.warning("读取 entry-points 失败（忽略）: %s", e)
        return []


def load_plugins(ctx=None, reload: bool = False) -> PluginBundle:
    """发现插件并返回 ``PluginBundle``；无配置/无插件 → 空 bundle。"""
    bundle = PluginBundle()
    for mod_path in _env_plugin_modules():
        _load_module_plugin(mod_path, bundle, ctx, reload=reload)
    for ep in _entry_point_plugins():
        try:
            loaded = ep.load()
            specs = loaded() if callable(loaded) else loaded
            _merge(specs, bundle.tools, f"entry-point:{ep.name}")
        except Exception as e:
            logger.warning("entry-point 插件加载失败，跳过 %s: %s", ep.name, e)
    return bundle


def load_plugin_tools(ctx=None, reload: bool = False) -> dict:
    """兼容旧接口：只返回工具表 ``{name: ToolSpec}``。"""
    return load_plugins(ctx, reload=reload).tools
