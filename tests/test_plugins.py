# SPDX-License-Identifier: AGPL-3.0-only

"""插件化改造测试：ToolSpec / 发现 / 注入分派 / 脱敏 / 状态打印。"""

import asyncio
import json
import threading

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.mcp_server import (
    DBAServer, create_mcp_server, _build_tool_list, _redacted,
    _oplog_request, _oplog_result, _print_status,
)
from ariadne_plugins import (
    PluginContext, PluginWatcher, ToolRegistry,
    ToolSpec, load_plugin_tools, load_plugins,
)


class FakeVectorStore:
    store = None

    def add_memories(self, *a, **k):
        pass

    def update_memories(self, *a, **k):
        pass

    def remove_memories(self, *a, **k):
        pass

    def reconcile(self, desired):
        return {"added": 0, "updated": 0, "removed": 0}


def _server(tmp_path):
    return DBAServer(
        MemoryGraph(),
        yaml_path=str(tmp_path / "g.yaml"),
        vector_store=FakeVectorStore(),
        lock=threading.RLock(),
    )


def _echo(value="x"):
    return {"echo": value}


def _spec(redact=False):
    return ToolSpec(
        name="plugin_echo",
        description="回显",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=_echo,
        redact=redact,
    )


def _call_tool(server, name, arguments):
    from mcp.types import CallToolRequestParams
    handler = server.get_request_handler("tools/call").handler
    return asyncio.run(handler(None, CallToolRequestParams(name=name, arguments=arguments)))


def test_toolspec_defaults():
    spec = ToolSpec("n", "d", {}, _echo)
    assert spec.redact is False


def test_load_plugin_tools_empty_by_default(monkeypatch):
    monkeypatch.delenv("ARIADNE_PLUGINS", raising=False)
    assert load_plugin_tools() == {}


def test_load_plugin_tools_from_env_module(tmp_path, monkeypatch):
    mod_name = "ariadne_test_plugin_" + tmp_path.name.replace("-", "_")
    (tmp_path / f"{mod_name}.py").write_text(
        "from ariadne_plugins import ToolSpec\n"
        "def _h(value='x'):\n"
        "    return {'echo': value}\n"
        "REGISTER = [ToolSpec('plugin_echo', '回显', {'type': 'object'}, _h)]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ARIADNE_PLUGINS", mod_name)
    tools = load_plugin_tools()
    assert set(tools) == {"plugin_echo"}
    assert tools["plugin_echo"].handler(value="hi") == {"echo": "hi"}


def test_build_tool_list_appends_plugins():
    builtin = [t.name for t in _build_tool_list()]
    assert "plugin_echo" not in builtin
    assert len(builtin) == 11

    with_plugin = [t.name for t in _build_tool_list({"plugin_echo": _spec()})]
    assert with_plugin[:-1] == builtin
    assert with_plugin[-1] == "plugin_echo"


def test_redacted_builtin_and_plugin():
    assert _redacted("dba_query_memory") is True
    assert _redacted("dba_get_stats") is False
    assert _redacted("plugin_echo", {"plugin_echo": _spec(redact=True)}) is True
    assert _redacted("plugin_echo", {"plugin_echo": _spec(redact=False)}) is False
    assert _redacted("unknown") is False


def test_oplog_redacts_plugin_payload():
    spec = _spec(redact=True)
    out_req = _oplog_request("plugin_echo", {"value": "秘密" * 100}, {"plugin_echo": spec})
    assert out_req["value"] == "<200 字符>"
    out_res = _oplog_result("plugin_echo", {"text": "秘密" * 100}, {"plugin_echo": spec})
    assert out_res["text"] == "<200 字符>"


def test_plugin_dispatch_end_to_end(tmp_path):
    dba = _server(tmp_path)
    touched = []
    dba.touch = lambda source="": touched.append(source)
    server = create_mcp_server(dba, extra_tools={"plugin_echo": _spec()})

    res = _call_tool(server, "plugin_echo", {"value": "hi"})
    payload = json.loads(res.content[0].text)
    assert payload == {"echo": "hi"}
    assert not res.is_error
    # 插件工具与内置走同一条分派：touch 被调用、oplog 落盘
    assert touched == ["plugin_echo"]
    assert "plugin_echo" in (tmp_path / "operations.log").read_text(encoding="utf-8")


def test_plugin_list_end_to_end(tmp_path):
    dba = _server(tmp_path)
    server = create_mcp_server(dba, extra_tools={"plugin_echo": _spec()})
    from mcp.types import PaginatedRequestParams
    handler = server.get_request_handler("tools/list").handler
    res = asyncio.run(handler(None, PaginatedRequestParams()))
    names = [t.name for t in res.tools]
    assert names[-1] == "plugin_echo"
    assert "dba_add_conversation" in names


def test_builtin_still_works_without_plugins(tmp_path):
    dba = _server(tmp_path)
    server = create_mcp_server(dba)
    res = _call_tool(server, "dba_get_stats", {})
    payload = json.loads(res.content[0].text)
    assert "total_nodes" in payload
    assert not res.is_error


def test_unknown_tool_still_errors(tmp_path):
    dba = _server(tmp_path)
    server = create_mcp_server(dba)
    res = _call_tool(server, "no_such_tool", {})
    payload = json.loads(res.content[0].text)
    assert "Unknown tool" in payload["error"]
    assert res.is_error


def test_print_status_shows_plugin(tmp_path, capsys):
    dba = _server(tmp_path)

    class Args:
        llm_model = "m"
        embedding_model = "e"
        embedding_local = False
        sse = False

    _print_status(dba.graph, dba, None, False, Args(), {"plugin_echo": _spec()})
    err = capsys.readouterr().err
    assert "dba_add_conversation" in err
    assert "plugin_echo" in err


# ---- 生命周期钩子 / 热加载 / 注册表 ----

def test_tool_registry_get_snapshot_replace():
    reg = ToolRegistry({"a": _spec()})
    assert reg.names() == ["a"]
    assert reg.get("a") is not None
    assert "a" in reg
    assert reg.snapshot()["a"].name == "plugin_echo"
    reg.replace({})
    assert reg.get("a") is None
    assert reg.snapshot() == {}


def test_build_tool_list_accepts_registry():
    reg = ToolRegistry({"plugin_echo": _spec()})
    names = [t.name for t in _build_tool_list(reg)]
    assert names[-1] == "plugin_echo"


def test_load_plugins_collects_hooks(tmp_path, monkeypatch):
    mod_name = "ariadne_hook_plugin_" + tmp_path.name.replace("-", "_")
    (tmp_path / f"{mod_name}.py").write_text(
        "from ariadne_plugins import ToolSpec\n"
        "CALLS = []\n"
        "def _h(value='x'):\n"
        "    return {'echo': value}\n"
        "REGISTER = [ToolSpec('plugin_echo', '回显', {'type': 'object'}, _h)]\n"
        "def on_startup(ctx):\n"
        "    CALLS.append('startup')\n"
        "def on_shutdown(ctx):\n"
        "    CALLS.append('shutdown')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ARIADNE_PLUGINS", mod_name)
    bundle = load_plugins(PluginContext(dba=None, yaml_path="/tmp/x.yaml"))
    assert set(bundle.tools) == {"plugin_echo"}
    assert len(bundle.startups) == 1
    assert len(bundle.shutdowns) == 1
    bundle.startups[0](None)
    bundle.shutdowns[0](None)
    import importlib
    assert importlib.import_module(mod_name).CALLS == ["startup", "shutdown"]


def test_watcher_reload_now_updates_registry(tmp_path, monkeypatch):
    reg = ToolRegistry()
    watcher = PluginWatcher(reg, interval=1.0)
    mod_name = "ariadne_watch_plugin_" + tmp_path.name.replace("-", "_")
    (tmp_path / f"{mod_name}.py").write_text(
        "from ariadne_plugins import ToolSpec\n"
        "REGISTER = [ToolSpec('plugin_echo', '回显', {'type': 'object'}, "
        "lambda value='x': {'echo': value})]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ARIADNE_PLUGINS", mod_name)
    new = watcher.reload_now()
    assert set(new) == {"plugin_echo"}
    assert set(reg.snapshot()) == {"plugin_echo"}


def test_entry_point_plugin(monkeypatch):
    """安装态发布路径：entry-points 组 ``ariadne.mcp.tools`` 能被发现。"""
    import importlib.metadata as md

    class FakeEP:
        name = "demo"

        def load(self):
            return lambda: [_spec()]

    class FakeEPs:
        def select(self, group):
            assert group == "ariadne.mcp.tools"
            return [FakeEP()]

    monkeypatch.setattr(md, "entry_points", lambda: FakeEPs())
    monkeypatch.delenv("ARIADNE_PLUGINS", raising=False)
    from ariadne_plugins.discovery import load_plugins
    bundle = load_plugins()
    assert set(bundle.tools) == {"plugin_echo"}


