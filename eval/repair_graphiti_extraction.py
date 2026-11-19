# -*- coding: utf-8 -*-
"""
写入侧修复实验：对 Graphiti 抽取丢失的节点做「重新入库 → 持续抽取/去重合并」修复，复测 expected 覆盖。

流程：
1. 从 Neo4j 找出抽取丢失节点（内容从未被任何 RELATES_TO 边引用）；
2. 将丢失节点的原文 content 作为新 episode 重新入库（Graphiti 增量抽取 + 去重合并，原生修复机制）；
3. 重建 episode uuid → node id 映射（含新 episode）；
4. 重跑 35 条查询检索，输出候选（供 compare_graphiti.py 对比修复前后）。

用法（.venv_graphiti）:
    python -m eval.repair_graphiti_extraction
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from neo4j import GraphDatabase

from frameworks.graphiti_runner import (
    OUT_JSON,
    QUERIES_YAML,
    _load_dotenv,
    build_clients,
    build_episodes,
    build_graphiti,
    force_combined_extraction,
    ingest,
    load_episode_map,
    load_graph_entries,
    search_all,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "lifemem2026")
EPISODE_MAP_JSON = ROOT / "eval" / ".cache" / "graphiti_episode_map.json"


def find_extraction_loss_nodes(driver) -> list:
    """expected 集合中、从未被任何 fact 引用的节点 id"""
    with open(ROOT / "data/LiFEMem_queries_v3.yaml", encoding="utf-8") as f:
        qs = yaml.safe_load(f)["queries"]
    expected_ids = set()
    for q in qs:
        expected_ids.update(q.get("expected", []))

    lost = []
    for nid in sorted(expected_ids):
        res = driver.execute_query(
            "MATCH (ep:Episodic {name: $nid}) WITH ep.uuid AS eu "
            "MATCH ()-[e:RELATES_TO]->() WHERE eu IN e.episodes RETURN count(e) AS c",
            nid=nid,
        )
        if not (res.records and res.records[0]["c"] > 0):
            lost.append(nid)
    return lost


async def main() -> None:
    force_combined_extraction()
    _load_dotenv(ROOT / ".env")
    os.environ.setdefault("NEO4J_PASSWORD", "lifemem2026")

    llm, embedder = build_clients()
    graphiti = build_graphiti("neo4j", llm, embedder)
    await graphiti.build_indices_and_constraints()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    lost = find_extraction_loss_nodes(driver)
    driver.close()
    print(f"[repair] 抽取丢失节点（修复前，expected 集合内）: {len(lost)} 个")

    all_entries = load_graph_entries()
    id2content = {nid: c for nid, c in all_entries}
    lost_entries = [(nid, id2content[nid]) for nid in lost if nid in id2content]
    print(f"[repair] 待重入原文: {len(lost_entries)} 条")

    if lost_entries:
        # 先从 Neo4j 重建既有映射，再增量入库追加新 episode
        episode_map = await load_episode_map(graphiti, EPISODE_MAP_JSON)
        await ingest(graphiti, build_episodes(lost_entries), chunk=16, episode_map=episode_map)
        # 重新从 Neo4j 重建完整映射（含新 episode uuid）
        episode_map = await load_episode_map(graphiti, EPISODE_MAP_JSON)

    # 重跑 35 条查询
    with open(ROOT / "eval/outputs/p_candidates_v3.json", encoding="utf-8") as f:
        p_data = json.load(f)
    top_n = {r["query_id"]: r.get("total_candidates", 40) for r in p_data["results"]}
    with open(ROOT / QUERIES_YAML, encoding="utf-8") as f:
        qs = yaml.safe_load(f)["queries"]

    await search_all(graphiti, qs, top_n, id2content, episode_map, ROOT / OUT_JSON)
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
