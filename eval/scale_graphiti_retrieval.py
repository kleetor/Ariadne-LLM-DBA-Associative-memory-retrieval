# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Graphiti（Zep 前置）「检索侧」注入量随图谱规模增长趋势。

在 .venv_graphiti 内运行：
  python eval/scale_graphiti_retrieval.py

目的：回答"图谱规模 ↑ → 检索返回多少内容（图遍历子图规模）/ 注入 token 是否爆表"。
- 不施加人为 limit（用较大 limit），暴露真实图遍历返回的节点/边/fact 数量。
- 统计 fact 文本的注入 token（cl100k_base，与 judge 上下文同标尺）。
- Graphiti 检索 = 混合检索，非 LLM（=0 内部 LLM token）；本测量聚焦注入量，非内部成本。

每个规模用独立 group（scale_50/100/200，来自 scale_graphiti_usage.py 已建图）。
"""

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # .venv_graphiti 无 tiktoken 时回退
    _ENC = None

import frameworks.graphiti_runner as gr  # noqa: E402
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF  # noqa: E402

SCALES = [50, 100, 200]
OUT = ROOT / "eval/outputs/scale_graphiti_retrieval.json"
QUERY = "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？"


def _tok(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text)


async def measure_scale(scale: int, llm, embedder) -> dict:
    graphiti = gr.build_graphiti("neo4j", llm, embedder)
    group_id = f"scale_{scale}"

    # 不施加人为小 limit：暴露真实图遍历规模（limit 足够大即可覆盖最大子图）
    cfg = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = 4096
    sr = await graphiti.search_(QUERY, config=cfg, group_ids=[group_id])

    nodes = getattr(sr, "nodes", [])
    edges = sr.edges
    facts = [e for e in edges if getattr(e, "fact", None)]

    # fact 文本注入量
    fact_texts = [getattr(e, "fact", "") for e in facts]
    fact_tokens = sum(_tok(t) for t in fact_texts if t)
    fact_chars = sum(len(t) for t in fact_texts)

    info = {
        "scale": scale,
        "subgraph_nodes": len(nodes),
        "subgraph_edges": len(edges),
        "subgraph_facts": len(facts),
        "fact_tokens": fact_tokens,
        "fact_chars": fact_chars,
        "avg_fact_chars": round(fact_chars / len(facts), 1) if facts else 0,
    }
    print(f"[scale={scale}] 子图 节点={len(nodes)} 边={len(edges)} fact={len(facts)}  "
          f"fact注入token={fact_tokens} (avg_chars={info['avg_fact_chars']})", flush=True)
    return info


async def main() -> None:
    gr._load_dotenv(ROOT / ".env")
    llm, embedder = gr.build_clients()

    results = []
    for s in SCALES:
        try:
            results.append(await measure_scale(s, llm, embedder))
        except Exception as e:
            print(f"[scale={s}] 失败: {type(e).__name__}: {e}", flush=True)
            results.append({"scale": s, "error": f"{type(e).__name__}: {e}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"query": QUERY, "results": results,
                              "note": "检索为混合检索(非LLM)；limit=4096 不截断；注入量用 cl100k_base"},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
