# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Graphiti（Zep 前置）冒烟：在本地 Neo4j 上完整建图 + 检索，取「模型看到的上下文块」。

在 .venv_graphiti 内运行（graphiti_core 依赖）。
  python eval/smoke_graphiti_block.py

产出：eval/outputs/smoke_graphiti_blocks.json
  { query_id: {"scheme":"graphiti","entities":[...],"relations":[...]} }
"""

import asyncio
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SEMAPHORE_LIMIT", "10")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml  # noqa: E402
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF  # noqa: E402

import frameworks.graphiti_runner as gr  # noqa: E402

GRAPH = "data/dba_eval/smoke_companion.yaml"
RAW_TEXT = "data/dba_eval/smoke_companion_raw.yaml"
import frameworks.dataset_loader as _dl  # noqa: E402
QUERIES = [{"id": q["id"], "text": q["text"]} for q in _dl.load_queries(RAW_TEXT)]
GROUP_ID = "smoke_cmp"  # 独立分组，避免污染 lifemem_eval
TOP_K = 8
OUT = ROOT / "eval/outputs/smoke_graphiti_blocks.json"


def load_entries() -> list:
    """从原始对话文本数据集的 conversation 字段重建 episode（写入侧应消费的原始语料）。"""
    import frameworks.dataset_loader as dl
    conv = dl.load_conversation(RAW_TEXT)
    return [(f"ep{i}", dl.turn_to_text(t)) for i, t in enumerate(conv)]


async def main() -> None:
    gr._load_dotenv(ROOT / ".env")
    # 中文场景下分开抽取弃答率高，强制节点+边「合并单次抽取」（0821 同款稳定方案）
    gr.force_combined_extraction()
    entries = load_entries()
    print(f"[graphiti-smoke] 节点: {len(entries)}")

    llm, embedder = gr.build_clients()
    graphiti = gr.build_graphiti("neo4j", llm, embedder)

    episodes = gr.build_episodes(entries)
    episode_map = {}
    # 入库（独立 group_id）
    total = len(episodes)
    for i in range(0, total, 5):
        batch = episodes[i:i + 5]
        res = await graphiti.add_episode_bulk(batch, group_id=GROUP_ID)
        for raw, node in zip(batch, res.episodes):
            episode_map[node.uuid] = raw.name
        print(f"[graphiti-smoke] ingest {i+len(batch)}/{total} (nodes={len(res.nodes)}, edges={len(res.edges)})", flush=True)

    cfg = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = TOP_K * 2
    blocks = {}
    for q in QUERIES:
        sr = await graphiti.search_(q["text"], config=cfg, group_ids=[GROUP_ID])
        edges = sr.edges
        # 实体名从 sr.nodes 取；关系(边)的事实文本取 sr.edges[].fact
        entities = [e.name for e in getattr(sr, "nodes", []) if getattr(e, "name", None)][:TOP_K]
        relations = []
        for e in edges:
            if e.fact:
                relations.append(e.fact)
            if len(relations) >= TOP_K:
                break
        blocks[q["id"]] = {"scheme": "graphiti", "entities": entities, "relations": relations}
        print(f"[graphiti-smoke] {q['id']} -> relations={len(relations)} entities={len(entities)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[graphiti-smoke] 已写: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
