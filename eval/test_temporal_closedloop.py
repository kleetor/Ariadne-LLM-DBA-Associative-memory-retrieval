# -*- coding: utf-8 -*-
"""
时序功能 闭环 + 增量验证。

闭环：LLM 在 dba_temporal_lookup / dba_query_memory 间路由 → 执行工具 → 取回事实 → LLM 生成回复。
增量：对同一"时间→事件"查询，对比 temporal_lookup（精确定位时间锚点取共时事件）
      与主检索器 query_memory（联想+故事化），看工具是否带来更干净、更贴合时序需求的结果。

用法:
    python -m eval.test_temporal_closedloop
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever
from dba_pipeline.mcp_server import TOOL_SCHEMAS


def main() -> None:
    _load_dotenv(ROOT / ".env")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    _, r = build_retriever("data/dba_eval/lifemem_crossperiod.yaml")

    # ---- 路由工具（dba_temporal_lookup / dba_query_memory）----
    wanted = {"dba_query_memory", "dba_temporal_lookup"}
    tools = []
    for ts in TOOL_SCHEMAS:
        if ts["name"] in wanted:
            tools.append({"type": "function", "function": {
                "name": ts["name"], "description": ts["description"],
                "parameters": ts["inputSchema"]}})
    route_chain = llm.bind_tools(tools)

    def route(query):
        resp = route_chain.invoke([SystemMessage(content="你是记忆检索路由助手，从给定工具中选择最合适的一个。只选一个。"),
                                   HumanMessage(content=query)])
        tc = (getattr(resp, "tool_calls", None) or [None])[0]
        if not tc:
            return None, {}
        return tc.get("name"), tc.get("args", {})

    # ---- 执行工具，返回供 LLM 使用的上下文 ----
    def run_tool(tool, query):
        if tool == "dba_temporal_lookup":
            res = r.temporal_lookup(query)
            ctx = []
            for g in res.get("matches", []):
                ta = g["time_anchor"]
                ctx.append(f"时间点「{ta['content']}」：")
                for f in g["facts"]:
                    ctx.append(f"  - {f['content']}")
            return "\n".join(ctx) if ctx else "（未定位到时间锚点）", res.get("matches", [])
        else:
            res = r.retrieve_with_story(query, with_response=False)
            stories = res.get("stories", [])
            ctx = "\n".join(f"- {s}" for s in stories) if stories else "（无相关故事片段）"
            return ctx, stories

    # ---- 生成回复 ----
    def generate(query, context):
        chain = llm.bind_tools([])  # 纯文本生成
        prompt = [SystemMessage(content="你有共情能力，根据下方记忆自然地回答用户问题，不机械列举。"),
                  HumanMessage(content=f"记忆：\n{context}\n\n问题：{query}")]
        return chain.invoke(prompt).content

    print("=" * 72)
    print("一、闭环验证：路由 → 工具 → 取回 → 生成")
    print("=" * 72)
    for query in ["昨天用户都做了些什么？", "为什么用户最近心情不好？"]:
        tool, args = route(query)
        print(f"\n[Q] {query}")
        print(f"  路由 → {tool}  (args={args})")
        ctx, raw = run_tool(tool if tool else "dba_query_memory", query)
        print(f"  取回上下文:\n{ctx}")
        if tool == "dba_temporal_lookup":
            reply = generate(query, ctx)
            print(f"  生成回复: {reply}")

    print("\n" + "=" * 72)
    print("二、增量验证：同一‘时间→事件’查询，工具 vs 主检索器")
    print("=" * 72)
    q = "昨天用户都做了些什么？"
    tctx, tmatches = run_tool("dba_temporal_lookup", q)
    mctx, mstories = run_tool("dba_query_memory", q)
    print(f"[temporal_lookup]  锚点×{len(tmatches)}:\n{tctx}")
    print(f"\n[query_memory]  故事×{len(mstories)}:\n{mctx}")


if __name__ == "__main__":
    main()
