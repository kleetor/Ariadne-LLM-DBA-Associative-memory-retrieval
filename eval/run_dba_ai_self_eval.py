# -*- coding: utf-8 -*-
"""
AI 自身经历/行为抽取验证。

对 data/dba_eval/ai_self_dialogs.yaml 逐批跑 dba.maintain()，
验证节点抽取器不再只记「用户」事实，也能把「助手」为响应用户请求实际做过的事
抽成独立记忆节点（content 以「助手」开头）。

用法:
    python -m eval.run_dba_ai_self_eval
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from eval.run_dba_extraction_eval import _load_dotenv, build_llm, build_embeddings
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.dba import MemoryDBA


def fresh():
    embeddings = build_embeddings()
    vs = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    g = MemoryGraph()
    b = GraphBuilder(g, vs, orphan_threshold=100)
    dba = MemoryDBA(llm=build_llm(), graph=g, vector_store=vs, graph_builder=b)
    return g, dba


def is_self_node(node):
    """判定是否为「助手自身」节点：content 以「助手」开头"""
    return node.get("content", "").startswith("助手")


def main() -> None:
    _load_dotenv(ROOT / ".env")

    with open(ROOT / "data/dba_eval/ai_self_dialogs.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dialogs = data["dialogs"]

    total_self_nodes = 0
    self_node_examples = []
    print(f"数据集 {data['subset']['name']}：{len(dialogs)} 批对话")
    print("=" * 70)

    for d in dialogs:
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in d["turns"])
        g, dba = fresh()
        res = dba.maintain(conversation)
        skipped = res.get("skipped", False)

        nodes = []
        for nid in g.graph.nodes():
            n = g.get_node(nid)
            n["id"] = nid
            n["self"] = is_self_node(n)
            nodes.append(n)

        self_nodes = [n for n in nodes if n["self"]]
        total_self_nodes += len(self_nodes)
        self_node_examples.extend(f"    {n['id']} [{n['node_type']}]: {n['content']}" for n in self_nodes)

        print(f"[批 {d['id']}] ({len(d['turns'])} 轮) triage_skipped={skipped} "
              f"-> 节点 {len(nodes)} / 助手节点 {len(self_nodes)}")
        print("  产出节点:")
        for n in nodes:
            tag = "  <-- 助手自身" if n["self"] else ""
            print(f"    {n['id']} [{n['node_type']}]: {n['content']}{tag}")
        print("  边:")
        for a, b, ed in g.graph.edges(data=True):
            print(f"    {a} -> {b} [{ed.get('rel_type')}]")
        print("-" * 70)

    print("=" * 70)
    ok = total_self_nodes > 0
    if ok:
        print(f"验证通过 [OK]：总共抽出 {total_self_nodes} 个「助手自身」节点。示例：")
        print("\n".join(self_node_examples))
    else:
        print("验证未通过 [FAIL]：未抽出任何「助手自身」节点——AI 自身经历/行为仍未被记忆。")
        sys.exit(1)


if __name__ == "__main__":
    main()
