# -*- coding: utf-8 -*-
"""
英文子集实验：从 LiFEMem 抽取跨场景子集并转译为英文（Mem0 英文环境对照）

目的：Mem0 的 BM25 在英文下真正生效（fastembed Qdrant/bm25 支持英文词级分词），
spaCy 实体提升也是英文 NLP 启发式。将子集转英文后重跑 Mem0，检验：
- 英文 embedding（bge-large-en-v1.5）下检索是否与中文（退化为纯向量）不同
- BM25 信号是否真正改变排序（qdrant vs faiss 差异）

Mem0 不消费图结构，只消费记忆条目（节点 content），因此子集 = 选定查询的
expected 节点 + 一跳邻居（充当相关但非 expected 的干扰记忆），保持条目数适中。

用法:
    python -m eval.prep_lifemem_en
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from langchain_openai import ChatOpenAI

from eval.run_p_baseline import _load_dotenv

GRAPH_YAML = "data/LiFEMem.yaml"
QUERIES_YAML = "data/LiFEMem_queries_v3.yaml"

# 选定的跨场景查询子集（10 条）
QUERY_IDS = ["s01", "s05", "s07", "s08", "s12", "s16", "s19", "s23", "s26", "s29"]

# 术语表：保证专名/人名翻译一致
GLOSSARY = """术语表（必须严格遵循）：
- 用户 → the user（第三人称记忆条目用）
- 小王 → Xiao Wang；小李 → Xiao Li；小陈 → Xiao Chen
- 豆豆 → Doudou（宠物狗的名字）
- 瑞幸 → Luckin；星巴克 → Starbucks；拿铁 → latte
- 字节跳动 → ByteDance；阿瓦隆 → Avalon（桌游）；吴恩达 → Andrew Ng
- 高中学过、高考、晚自习、传纸条、编程社团、P0 故障、Code Review 等保持自然英文
"""


def load_graph(rel: str):
    with open(ROOT / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries(rel: str):
    with open(ROOT / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)["queries"]


def neighbors(graph, nid: str) -> set:
    """一跳邻居（无向，含 expected 节点自身）"""
    out = {nid}
    for e in graph.get("edges", []):
        if e["from"] == nid:
            out.add(e["to"])
        if e.get("to") == nid:
            out.add(e["from"])
    return out


def translate(chat, items: list, kind: str) -> list:
    """批量翻译 items（list[str]），返回 list[str] 英文"""
    result = []
    B = 20
    for start in range(0, len(items), B):
        batch = items[start:start + B]
        lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(batch))
        if kind == "node":
            inst = ("你是数据标注员。将以下每条中文记忆条目忠实翻译为英文，"
                    "保持第三人称（The user ...）与信息完整性，不增删事实。"
                    f"{GLOSSARY}\n按原样输出每一行：`[序号] 英文翻译`，不要输出其他内容。\n")
        else:
            inst = ("你是数据标注员。将以下每条中文**第一人称口语陈述**翻译为英文，"
                    "保持第一人称（I ...）与口语感，不增删信息。"
                    f"{GLOSSARY}\n按原样输出每一行：`[序号] 英文翻译`，不要输出其他内容。\n")
        msg = inst + lines
        resp = chat.invoke(msg)
        text = resp.content.strip()
        # 解析 "[i] xxx" 行
        for i in range(len(batch)):
            found = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(f"[{i}]"):
                    found = line[len(f"[{i}]"):].strip()
                    break
            if not found:
                raise RuntimeError(f"翻译解析失败 kind={kind} idx={i} item={batch[i]!r}\nresp={text[:500]}")
            result.append(found)
        print(f"[prep] 已翻译 {kind} {min(start + B, len(items))}/{len(items)}", flush=True)
    return result


def main() -> None:
    _load_dotenv(ROOT / ".env")
    chat = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    graph = load_graph(GRAPH_YAML)
    nodes = {n["id"]: n for n in graph["nodes"]}
    queries = load_queries(QUERIES_YAML)

    selected = [q for q in queries if q["id"] in QUERY_IDS]
    print(f"[prep] 选定查询 {len(selected)} 条: {QUERY_IDS}")

    # 子图节点：expected 并集 + 一跳邻居
    sub = set()
    for q in selected:
        for nid in q.get("expected", []):
            sub |= neighbors(graph, nid)
    sub_ids = sorted(sub, key=lambda x: (nodes[x].get("type", ""), x))
    print(f"[prep] 子集节点数（expected+一跳邻居）: {len(sub_ids)}")

    # 翻译节点
    node_items = [nodes[nid]["content"] for nid in sub_ids]
    node_en = translate(chat, node_items, "node")

    # 翻译查询
    q_items = [q["text"] for q in selected]
    q_en = translate(chat, q_items, "query")

    # 写英文节点子集 YAML（保留 type 字段）
    sub_nodes = []
    for nid, en in zip(sub_ids, node_en):
        sub_nodes.append({"id": nid, "content": en, "type": nodes[nid].get("type", "")})
    out_sub = ROOT / "data" / "LiFEMem_en_subset.yaml"
    out_sub.write_text(
        yaml.safe_dump({"meta": {"source": GRAPH_YAML, "language": "en",
                                 "query_ids": QUERY_IDS},
                        "nodes": sub_nodes}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"[prep] 英文节点子集 -> {out_sub}（{len(sub_nodes)} 条）")

    # 写英文查询 YAML
    sub_queries = []
    for q, en in zip(selected, q_en):
        sub_queries.append({"id": q["id"], "scenario": q["scenario"],
                            "text": en, "purpose_hint": q.get("purpose_hint", ""),
                            "expected": q.get("expected", [])})
    out_q = ROOT / "data" / "LiFEMem_en_queries.yaml"
    out_q.write_text(
        yaml.safe_dump({"meta": {"source": QUERIES_YAML, "language": "en"},
                        "queries": sub_queries}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"[prep] 英文查询 -> {out_q}（{len(sub_queries)} 条）")

    # 打印对照供人工核对
    for q, en in zip(selected, q_en):
        print(f"\n  {q['id']} [{q['scenario']}]")
        print(f"    中: {q['text']}")
        print(f"    EN: {en}")
    for nid, zh, en in list(zip(sub_ids, node_items, node_en))[:6]:
        print(f"\n  {nid}: {zh}\n    EN: {en}")
    print(f"\n[prep] 完成")


if __name__ == "__main__":
    main()
