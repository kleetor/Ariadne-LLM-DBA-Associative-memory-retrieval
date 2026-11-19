# -*- coding: utf-8 -*-
"""
端到端验证 + 完整数据集（LiFEMem 322 图）时序工具覆盖。

1) 端到端：DBAServer.temporal_lookup 包装（走 MCP 同一入口）是否正常返回 {time_anchor, facts}。
2) 完整数据集覆盖：枚举整个图上被判定为"时间锚点"的节点（_tool_is_time_node），逐一取共时事实，
   看工具在完整数据集上的时间锚点覆盖率，以及是否存在误判锚点（如含时间词的 THING）。

用法:
    python -m eval.test_temporal_full
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever


def main() -> None:
    _load_dotenv(ROOT / ".env")
    from dba_pipeline.mcp_server import DBAServer

    graph, r = build_retriever("data/LiFEMem.yaml")
    server = DBAServer(graph, retriever=r)

    # ---- 1) 端到端：走 DBAServer 包装入口 ----
    print("=" * 72)
    print("一、端到端（DBAServer.temporal_lookup，MCP 同一入口）")
    print("=" * 72)
    e2e_queries = [
        "用户高中时期发生了什么？",
        "用户大专三年经历了什么？",
        "用户当前工作时期怎么样？",
        "用户现在这份工作做了三年，情况怎么样？",  # 语义贴近误判节点 w13，应绕过并落到真锚点 th2
        "昨天用户做了什么？",
    ]
    for q in e2e_queries:
        res = server.temporal_lookup(q)
        matches = res.get("matches", [])
        if matches:
            print(f"\n  Q: {q}")
            for g in matches:
                ta = g.get("time_anchor", {})
                print(f"     锚点: {ta['id']} ({ta.get('content')!r})")
                for f in g.get("facts", []):
                    print(f"       - {f['id']} → {f['content']!r}")
        else:
            print(f"\n  Q: {q}")
            print(f"     （未定位到可用时间锚点）{res.get('note','')}")

    # ---- 2) 完整数据集：枚举所有"时间锚点"节点 ----
    print("\n" + "=" * 72)
    print("二、完整数据集（322 图）时间锚点覆盖")
    print("=" * 72)
    anchor_nodes = [nid for nid in graph.graph.nodes() if r._tool_is_time_node(nid)]
    print(f"判定为时间锚点的节点数: {len(anchor_nodes)}")
    for nid in anchor_nodes:
        content = graph.graph.nodes[nid].get("content", "")
        res = r.temporal_lookup(content)  # 用锚点自身内容做查询 → 应命中相关时间锚点
        groups = res.get("matches", [])
        total = sum(len(g.get("facts", [])) for g in groups)
        print(f"\n  [{nid}] {content!r}  命中锚点数 x{len(groups)}，共时事件 x{total}")
        for g in groups:
            ta = g.get("time_anchor", {})
            print(f"      → {ta['id']} ({ta.get('content')!r})")
            for f in g.get("facts", []):
                print(f"           - {f['id']} → {f['content']!r}")


if __name__ == "__main__":
    main()
