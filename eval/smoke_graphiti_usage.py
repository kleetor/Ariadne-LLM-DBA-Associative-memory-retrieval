# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Graphiti（Zep 前置）单次/会话内部 LLM 抽取 token 成本。

在 .venv_graphiti 内运行（graphiti_core 依赖）：
  python eval/smoke_graphiti_usage.py

测量：ingest 4 条原始文本 episode 时，合并抽取（节点+边）消耗的 LLM
input+output token。产出 eval/outputs/graphiti_usage.json。
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

RAW_TEXT = "data/dba_eval/smoke_companion_raw.yaml"
GROUP_ID = "smoke_cmp_usage"  # 独立分组，专用于测量，避免污染
OUT = ROOT / "eval/outputs/graphiti_usage.json"


class Recorder:
    """包裹 AsyncOpenAI chat.completions.create，读 response.usage 累加 input+output token。"""

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


def load_episodes() -> list:
    import frameworks.dataset_loader as dl
    conv = dl.load_conversation(RAW_TEXT)
    return [(f"ep{i}", dl.turn_to_text(t)) for i, t in enumerate(conv)]


async def main() -> None:
    gr._load_dotenv(ROOT / ".env")
    gr.force_combined_extraction()
    llm, embedder = gr.build_clients()
    graphiti = gr.build_graphiti("neo4j", llm, embedder)

    episodes = gr.build_episodes(load_episodes())
    print(f"[graphiti-usage] episodes={len(episodes)}", flush=True)

    rec = Recorder()
    # Graphiti 用 AsyncOpenAI client：包裹 chat.completions.create 统计 token
    completions = llm.client.chat.completions
    orig_create = completions.create
    completions.create = rec.wrap(orig_create)
    try:
        res = await graphiti.add_episode_bulk(episodes, group_id=GROUP_ID)
    finally:
        completions.create = orig_create
    print(f"[graphiti-usage] 抽取: 节点={len(res.nodes)} 边={len(res.edges)}", flush=True)
    print(f"[graphiti-usage] LLM调用={rec.calls}  input={rec.prompt}  output={rec.completion}  "
          f"total={rec.total}", flush=True)

    # 额外测一次 search 返回的子图规模（遍历注入量级）
    cfg = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = 16
    sr = await graphiti.search_("最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？",
                                config=cfg, group_ids=[GROUP_ID])
    print(f"[graphiti-usage] search: 子图节点={len(getattr(sr,'nodes',[]))} 边={len(sr.edges)}  "
          f"facts={sum(1 for e in sr.edges if e.fact)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_episodes": len(episodes),
        "llm_calls": rec.calls,
        "prompt_tokens": rec.prompt,
        "completion_tokens": rec.completion,
        "total_tokens": rec.total,
        "extracted_nodes": len(res.nodes),
        "extracted_edges": len(res.edges),
        "search_subgraph_edges": len(sr.edges),
        "note": "合并抽取(input+output token)；search 为混合检索(无LLM)，子图规模另列",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graphiti-usage] 已写: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
