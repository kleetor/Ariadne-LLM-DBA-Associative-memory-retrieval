# -*- coding: utf-8 -*-
"""
MCP 链路检查：验证时序工具经 MCP 全链路（server handlers → DBAServer → retriever）可用，
并确认工具注册完整、max_anchors 已透传、无残留已删代码引用。

覆盖：
  ① 编译/导入：mcp_server + retriever + create_mcp_server 构建成功；
  ② 工具注册：_build_tool_list 含 dba_temporal_lookup，server 注册了 tools/list 与 tools/call；
  ③ 数据通路：通过 MCP handler 调用 dba_temporal_lookup("昨天用户都做了些什么？")，拿到 T10 的共时事实；
  ④ max_anchors 透传：指定 max_anchors=2 应生效（matches 数 <= 2）；
  ⑤ 无残留引用：mcp_server 不引用已删除的 temporal_mode/_is_time_node/_expand_temporal_reverse。

用法:
    python -m eval.test_mcp_link
"""

import os
import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever
from dba_pipeline.mcp_server import DBAServer, create_mcp_server, TOOL_SCHEMAS


def main() -> None:
    _load_dotenv(ROOT / ".env")
    yaml_path = "data/dba_eval/lifemem_crossperiod.yaml"

    # ① 构建 retriever + DBAServer + MCP server（能构建即验证编译/导入与依赖）
    graph, r = build_retriever(yaml_path)
    dba = DBAServer(graph, yaml_path=yaml_path, retriever=r)
    server = create_mcp_server(dba)
    print("① 编译/导入 + DBAServer + create_mcp_server 构建成功 [OK]")

    # ② 工具注册列表
    tool_names = [t.name for t in _build_tool_list()]
    has_temporal = "dba_temporal_lookup" in tool_names
    print(f"② 工具列表({len(tool_names)}): {tool_names}")
    print(f"   含 dba_temporal_lookup = {has_temporal}")
    handlers = getattr(server, "_request_handlers", {})
    registered = sorted(str(k) for k in handlers.keys()) if handlers else []
    print(f"   注册的 handler: {registered}")
    ok_reg = has_temporal and ("tools/call" in registered and "tools/list" in registered)

    # ③④ 通过 MCP handler 调用时序工具
    def _resolve(callable_obj):
        for attr in ("handler", "fn", "func", "_handler"):
            cand = getattr(callable_obj, attr, None)
            if callable(cand):
                return cand
        return callable_obj if callable(callable_obj) else None

    async def call_temporal(max_anchors=3):
        from mcp.types import CallToolRequestParams
        h = _resolve(handlers.get("tools/call"))
        if h is None:
            raise RuntimeError("tools/call handler 无法解析")
        params = CallToolRequestParams(
            name="dba_temporal_lookup",
            arguments={"query": "昨天用户都做了些什么？", "max_anchors": max_anchors},
        )
        return await h(None, params)

    async def run():
        res = await call_temporal(3)
        text = res.content[0].text
        print("③ [MCP dba_temporal_lookup] 昨天用户都做了些什么？ (max_anchors=3)")
        return res, text

    res, text = asyncio.run(run())
    matches = json.loads(text).get("matches", []) if isinstance(text, str) else []
    anchors = [m["time_anchor"]["id"] for m in matches]
    facts = {f["id"] for m in matches for f in m["facts"]}
    print(f"   锚点={anchors} 共时事实={sorted(facts)}")
    ok_data = ("T10" in anchors) and {"e9", "e10", "e12"} <= facts

    # ④ 验证 max_anchors 透传：设为 1 应只返回 1 个锚点组
    async def run2():
        return await call_temporal(1)
    res1 = asyncio.run(run2())
    matches1 = json.loads(res1.content[0].text).get("matches", [])
    print(f"④ max_anchors=1 -> matches 数量 = {len(matches1)}")
    ok_max = len(matches1) <= 1

    # ⑤ 无残留引用（静态检查 mcp_server 源码）
    src = Path(ROOT / "dba_pipeline" / "mcp_server.py").read_text(encoding="utf-8")
    removed_refs = [x for x in ("temporal_mode", "_is_time_node", "_expand_temporal_reverse")
                    if x in src]
    ok_norem = not removed_refs
    print(f"⑤ 残留已删代码引用: {removed_refs if removed_refs else '无'}")

    print("\n" + "=" * 60)
    if ok_reg and ok_data and ok_max and ok_norem:
        print("MCP 链路检查通过 [OK]：工具注册完整、时序工具经 MCP 全链路可用、max_anchors 已透传、无残留引用。")
    else:
        print(f"MCP 链路检查未通过: reg={ok_reg} data={ok_data} max={ok_max} norem={ok_norem}")
        sys.exit(1)


if __name__ == "__main__":
    from dba_pipeline.mcp_server import _build_tool_list
    main()
