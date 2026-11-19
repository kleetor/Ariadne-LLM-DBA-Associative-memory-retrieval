# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Graphiti（Zep 前置）内部 LLM 抽取 token 随图谱规模增长趋势。

在 .venv_graphiti 内运行（graphiti_core 依赖）：
  python eval/scale_graphiti_usage.py

测量：对每个规模 N ∈ {50, 100, 200}（同源递增），把该规模的原始对话 ingest 进图，
测「合并抽取」消耗的 LLM input+output token；再测一次 search 返回的子图规模。

关键点：Graphiti 的抽取是「侵入式」——每次 add_episode_bulk 会把已有图实体的
消歧/关系合并上下文带进 prompt，因此随着图（规模 N）增大，单次 ingest 的 LLM
input token 会增长，这是本测量关注的「图规模 → 抽取成本膨胀」。

产出：eval/outputs/scale_graphiti_usage.json
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
os.environ.setdefault("SEMAPHORE_LIMIT", "10")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF  # noqa: E402

import frameworks.graphiti_runner as gr  # noqa: E402

SCALES = [50, 100, 200]
OUT = ROOT / "eval/outputs/scale_graphiti_usage.json"
QUERY = "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？"


class Recorder:
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def wrap(self, create_fn):
        rec = self

        async def w_create(*a, **k):
            r = await create_fn(*a, **k)
            u = getattr(r, "usage", None)
            if u is not None:
                rec.prompt += getattr(u, "prompt_tokens", 0) or 0
                rec.completion += getattr(u, "completion_tokens", 0) or 0
                rec.calls += 1
            return r

        return w_create

    @property
    def total(self):
        return self.prompt + self.completion


def load_episodes(n_scale: int) -> list:
    import frameworks.dataset_loader as dl
    conv = dl.load_conversation(f"data/dba_eval/scale_{n_scale}_raw.yaml")
    return [(f"ep{i}", dl.turn_to_text(t)) for i, t in enumerate(conv)]


async def measure_scale(scale: int, llm, embedder, group_id: str) -> dict:
    import frameworks.graphiti_runner as gr2
    graphiti = gr2.build_graphiti("neo4j", llm, embedder)
    episodes = gr2.build_episodes(load_episodes(scale))
    print(f"[scale={scale}] episodes={len(episodes)}", flush=True)

    rec = Recorder()
    completions = llm.client.chat.completions
    orig_create = completions.create
    completions.create = rec.wrap(orig_create)
    try:
        res = await graphiti.add_episode_bulk(episodes, group_id=group_id)
    finally:
        completions.create = orig_create

    cfg = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = 16
    sr = await graphiti.search_(QUERY, config=cfg, group_ids=[group_id])

    info = {
        "scale": scale,
        "n_episodes": len(episodes),
        "llm_calls": rec.calls,
        "prompt_tokens": rec.prompt,
        "completion_tokens": rec.completion,
        "total_tokens": rec.total,
        "extracted_nodes": len(res.nodes),
        "extracted_edges": len(res.edges),
        "search_subgraph_nodes": len(getattr(sr, "nodes", [])),
        "search_subgraph_edges": len(sr.edges),
    }
    print(f"[scale={scale}] 抽取: 节点={info['extracted_nodes']} 边={info['extracted_edges']} "
          f"LLM={rec.calls} in={rec.prompt} out={rec.completion} total={rec.total}", flush=True)
    print(f"[scale={scale}] search: 子图节点={info['search_subgraph_nodes']} 边={info['search_subgraph_edges']}", flush=True)
    return info


async def main() -> None:
    gr._load_dotenv(ROOT / ".env")
    gr.force_combined_extraction()
    llm, embedder = gr.build_clients()

    results = []
    for i, s in enumerate(SCALES):
        try:
            info = await measure_scale(s, llm, embedder, group_id=f"scale_{s}")
            results.append(info)
        except Exception as e:
            print(f"[scale={s}] 失败: {type(e).__name__}: {e}", flush=True)
            results.append({"scale": s, "error": f"{type(e).__name__}: {e}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"query": QUERY, "results": results,
                              "note": "合并抽取(input+output)；search 混合检索非LLM；图规模=episode 数"},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
