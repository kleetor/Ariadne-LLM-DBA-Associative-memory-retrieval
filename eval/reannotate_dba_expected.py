# -*- coding: utf-8 -*-
"""
DBA 图 expected 重标注（修复 expected 由 gold 图反推导致的口径偏置）

背景：LiFEMem 查询集的 expected 原系从 gold 图反推标注，gold 图在 R@all 对比中
天然占优。本脚本为 DBA 图**独立**重标 expected：对每条非弃权查询，将 DBA 图全部
节点（id: content）交给 LLM，由其选出该陈述应「触发联想」的关键记忆集（2~6 个），
不使用任何图结构（边/邻接）信息——从而消除 gold 结构优势。

用法:
    python -m eval.reannotate_dba_expected --graph eval/outputs/dba_graph_subset_b6_v4.yaml \
        --out eval/outputs/dba_expected_v4.json

参数:
    --graph    DBA 图 YAML
    --out      标注输出 JSON
    --queries  查询集（默认 data/LiFEMem_queries_v3.yaml）
    --workers  并行标注线程数（默认 8）
    --min-n    要求 LLM 选的最小节点数（默认 2）
    --max-n    要求 LLM 选的最大节点数（默认 6）

输出 JSON:
    {
      "graph": "...",
      "prompt": "标注口径说明",
      "annotations": {qid: [dba_id, ...]},
      "pool_sizes": {qid: n},       # 提供给 LLM 的候选节点总数（=图节点数）
      "abstention_skipped": [qid, ...]
    }
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dba_pipeline.loader import load_graph


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆标注员。给定一条用户的陈述，以及一个记忆库中的记忆节点列表（id: 内容），
请选出这条陈述应当「触发联想」的关键记忆集：即系统回忆这些记忆才能给出有记忆、有共鸣的回应。

规则：
- 包括与该陈述**直接相关**的记忆，以及**合理的联想**记忆（不要过度发散）；
- 只从给出的节点中选，不要编造 id；
- 数量控制在 {min_n}~{max_n} 个；
- 只输出 JSON 数组（节点 id 字符串列表），不要输出任何其他内容。

用户陈述：
{text}

记忆节点列表：
{node_lines}"""),
])


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash-0731"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0.0,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},  # api.gpt.ge 必需：关 thinking 直出 content
    )


def annotate_one(q, nodes, llm, min_n, max_n, attempts=3):
    """单查询标注：返回 (qid, ids) 或 (qid, [])。ids 强制 clamp 到 max_n"""
    qid = q["id"]
    text = q["text"]
    node_lines = "\n".join(f"{nid}: {content}" for nid, content in nodes)
    for _ in range(attempts):
        try:
            chain = PROMPT | llm
            resp = chain.invoke({"text": text, "node_lines": node_lines,
                                 "min_n": min_n, "max_n": max_n})
            content = resp.content.strip()
            # 容错：截取第一个 [ ... ] 片段
            start, end = content.find("["), content.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"无 JSON 数组: {content[:200]}")
            ids = json.loads(content[start:end + 1])
            valid = [i for i in ids if isinstance(i, str) and i in dict(nodes)]
            if valid:
                return qid, valid[:max_n]  # 钳制到 max_n
        except Exception as e:  # noqa: BLE001
            last_err = e
    return qid, []


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 图 expected 重标注")
    parser.add_argument("--graph", required=True, help="DBA 图 YAML")
    parser.add_argument("--out", required=True, help="标注输出 JSON")
    parser.add_argument("--queries", default="data/LiFEMem_queries_v3.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=4)
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")

    graph = load_graph(str(ROOT / args.graph))
    nodes = []
    for nid in graph.graph.nodes():
        content = graph.graph.nodes[nid].get("content")
        if content:
            nodes.append((nid, content))
    node_map = dict(nodes)
    print(f"[REANN] 图节点数（有 content）: {len(nodes)}")

    with open(ROOT / args.queries, "r", encoding="utf-8") as f:
        qs = yaml.safe_load(f)

    # 增量：已有标注文件则跳过已标注查询
    out_path = ROOT / args.out
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8")).get("annotations", {})

    tasks = []
    for q in qs["queries"]:
        qid = q["id"]
        expected = q.get("expected", []) or []
        if not expected:  # 弃权/无 expected
            continue
        if qid in existing:
            continue
        tasks.append(q)

    if not tasks:
        print(f"[REANN] 全部 {len(existing)} 条已标注，无需新增。")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(annotate_one, q, nodes, build_llm(),
                              args.min_n, args.max_n): q["id"] for q in tasks}
            for fut in as_completed(futs):
                qid, ids = fut.result()
                existing[qid] = ids
                # 增量落盘
                tmp = {"graph": str(args.graph), "annotations": existing}
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(tmp, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[REANN] {qid}: {len(ids)} 个 → {ids[:8]}")

    # 汇总
    empty = [qid for qid, ids in existing.items() if not ids]
    avg = sum(len(v) for v in existing.values()) / len(existing) if existing else 0
    out = {
        "graph": str(args.graph),
        "prompt": "独立重标：全图节点 + LLM 判断（无图结构信息），该陈述应触发联想的关键记忆集 2~6 个",
        "annotations": existing,
        "pool_sizes": {qid: len(node_map) for qid in existing},
        "abstention_skipped": [q["id"] for q in qs["queries"] if not (q.get("expected") or [])],
        "avg_expected": round(avg, 2),
        "empty_annotations": empty,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[REANN] 完成: {len(existing)} 条标注，平均 {avg:.2f} 个/条，空标注 {len(empty)} 条")
    print(f"[REANN] 输出: {out_path}")


if __name__ == "__main__":
    main()
