# SPDX-License-Identifier: AGPL-3.0-only

"""MCP 快速开始：通过 stdio 启动 ariadne-mcp，列出工具并执行一次记忆检索。

前置条件：
    - 已安装项目依赖（`pip install -e .`）
    - 根目录 `.env` 中配置好 LLM / Embedding

用法（在仓库根执行）：
    python examples/quickstart_mcp.py
    python examples/quickstart_mcp.py --query "我最近的压力来自哪里？" --rerank-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]


async def run(query: str, rerank_k: int, yaml_path: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "dba_pipeline.mcp_server", "--yaml", yaml_path],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("可用工具：", ", ".join(tool.name for tool in tools.tools))

            result = await session.call_tool(
                "dba_query_memory",
                {"query": query, "rerank_k": rerank_k},
            )
            for item in result.content:
                text = getattr(item, "text", None)
                if text:
                    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP stdio 快速开始")
    parser.add_argument("--query", default="我最近在做什么项目？")
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument(
        "--yaml",
        default=str(REPO_ROOT / "data" / "memory_graph.yaml"),
        help="图谱 YAML 路径",
    )
    args = parser.parse_args()
    asyncio.run(run(args.query, args.rerank_k, args.yaml))


if __name__ == "__main__":
    main()
