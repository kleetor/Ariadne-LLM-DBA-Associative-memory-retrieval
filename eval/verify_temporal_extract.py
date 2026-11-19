# -*- coding: utf-8 -*-
"""
抽取侧 ② 细粒度时间锚点产出验证。

对一条"时序叙事"对话（含 >=2 个不同日期/时期词）跑 dba.maintain()，
验证手术式守卫 has_temporal_signal 触发 → 走 temporal 增强链 → 产出 TEMPORAL 边 + 时间锚点节点；
对非时序对话不触发 → 不产出 TEMPORAL（零回归的另一面）。

用法:
    python -m eval.verify_temporal_extract
"""

import os
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_dba_extraction_eval import _load_dotenv, build_llm, build_embeddings
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.dba import MemoryDBA, has_temporal_signal
from dba_pipeline.core.jump_axis import RelationType


def fresh():
    embeddings = build_embeddings()
    vs = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    g = MemoryGraph()
    b = GraphBuilder(g, vs, orphan_threshold=100)
    dba = MemoryDBA(llm=build_llm(), graph=g, vector_store=vs, graph_builder=b)
    return g, dba


def count_temporal(g):
    # RelationType 是 Enum，直接按枚举比较（str().lower() 会得到 relationtype.temporal，误判）
    return [(a, b, d) for a, b, d in g.graph.edges(data=True)
            if d.get("rel_type") == RelationType.TEMPORAL]


def main() -> None:
    _load_dotenv(ROOT / ".env")

    temporal_conv = ("用户这周很累：周一感冒发烧，周二还坚持上班，周三去了医院打点滴，"
                     "周四好转，周五已经恢复。用户感叹这周过得太辛苦。")
    plain_conv = "用户喜欢喝手冲咖啡，周末常去书店看书，最近在读一本历史书，觉得挺有意思。"

    print("== 时序叙事对话 ==")
    print(f"  has_temporal_signal = {has_temporal_signal(temporal_conv)}")
    g1, dba1 = fresh()
    dba1.maintain(temporal_conv)
    te1 = count_temporal(g1)
    # 时间锚点 = TEMPORAL 边的终点（事件→时间），直接以结构识别
    time_nodes = sorted({b for _, b, d in g1.graph.edges(data=True)
                         if d.get("rel_type") == RelationType.TEMPORAL})
    print(f"  产出节点数={g1.node_count} 边数={g1.edge_count}")
    print(f"  TEMPORAL 边数={len(te1)}")
    print(f"  时间锚点节点数={len(time_nodes)}: {time_nodes}")
    print("  节点内容:")
    for nid in g1.graph.nodes():
        n = g1.get_node(nid)
        print(f"    {nid}: {n['content']!r} [{n['node_type']}]")
    print("  边(rel_type):")
    for a, b, d in g1.graph.edges(data=True):
        print(f"    {a} -> {b} [{d.get('rel_type')}]")
    for a, b, d in te1:
        print(f"    - {a} -> {b} [{d.get('rel_type')}]")

    print("\n== 非时序对话 ==")
    print(f"  has_temporal_signal = {has_temporal_signal(plain_conv)}")
    g2, dba2 = fresh()
    dba2.maintain(plain_conv)
    te2 = count_temporal(g2)
    print(f"  产出节点数={g2.node_count} 边数={g2.edge_count}")
    print(f"  TEMPORAL 边数={len(te2)}")

    ok = has_temporal_signal(temporal_conv) and len(te1) > 0 and len(time_nodes) > 0 \
        and not has_temporal_signal(plain_conv) and len(te2) == 0
    print("\n" + "=" * 60)
    if ok:
        print("抽取侧② 细粒度锚点产出验证通过 [OK]：时序叙事触发守卫并产出 TEMPORAL 边+时间锚点；非时序不触发。")
    else:
        print("抽取侧② 验证未通过，请检查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
