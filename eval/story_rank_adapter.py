# -*- coding: utf-8 -*-
"""
T2: StoryRank 故事化适配

将各记忆系统的候选统一转换为 StoryRank 输入并故事化：
- P 候选：带因果边（path 结构：nodes + edges）
- 外部框架候选：记忆文本列表 → 统一转为 {nodes, edges=[]} 结构

关键验证点：外部候选 edges 为空时，StoryRank 应产出合格故事且不编造关系。

用法（验证）:
    python -m eval.story_rank_adapter
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eval.run_p_baseline import _load_dotenv, build_retriever

# 外部框架候选无节点类型信息时的默认标注
DEFAULT_NODE_TYPE = "STATUS"


def story_rank_for_path(inference, query: str, path: dict) -> dict:
    """P 候选（带边 path）→ StoryRank"""
    return inference.story_rank(query, path)


def story_rank_for_texts(inference, query: str, texts) -> dict:
    """外部框架候选（文本列表）→ {nodes, edges=[]} → StoryRank"""
    nodes = [
        {"id": f"c{i}", "content": t, "node_type": DEFAULT_NODE_TYPE}
        for i, t in enumerate(texts)
    ]
    path = {"nodes": nodes, "edges": []}
    return inference.story_rank(query, path)


def _verify() -> None:
    """验证：q01 分别用 P 带边 path 与纯文本输入跑 StoryRank"""
    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever("data/LiFEMem.yaml")
    inference = retriever.inference

    with open(ROOT / "eval/outputs/p_candidates.json", encoding="utf-8") as f:
        data = json.load(f)
    res = {r["query_id"]: r for r in data["results"]}

    for qid in ["q01", "q13"]:
        r = res[qid]
        query = r["query"]
        print(f"\n===== {qid}: {query} =====")

        # 1. P 带边 path
        out_p = story_rank_for_path(inference, query, r["path"])
        story_p = out_p.get("story", "")
        print(f"[P 带边 path] 节点数={len(r['path']['nodes'])}, 边数={len(r['path']['edges'])}")
        print(f"  story({len(story_p)}字): {story_p[:150]}...")
        # 0927 审查 C1 消费者同步：story_rank 只返回 {story, used, degraded}，不再有
        # adopted_ids（采纳 id 列表）；used 为后端按正文实际覆盖统计的节点数。
        print(f"  story_used: {out_p.get('used', 0)}")

        # 2. 纯文本输入（模拟外部框架，取前 10 条候选文本）
        texts = [c["content"] for c in r["candidates"][:10]]
        out_t = story_rank_for_texts(inference, query, texts)
        story_t = out_t.get("story", "")
        print(f"[纯文本输入] 文本数={len(texts)}")
        print(f"  story({len(story_t)}字): {story_t[:150]}...")


if __name__ == "__main__":
    _verify()
