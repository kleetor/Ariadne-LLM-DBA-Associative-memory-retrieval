# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
多系统 e2e 冒烟对比（改版：不生成回复，直接把「检索出的原文」交给 judge）

设计变化（去除 reply 模型混淆）：
  query → 检索(候选原文) → 直接把候选原文拼接 → judge 评「这段记忆对当下陪伴诉求的质量」
  不再有真实回复 LLM，避免「判分反映的是回复模型能力而非检索质量」。

系统：PAR / Naive RAG / Mem0(中文补丁)
judge：读 JUDGE_*（qwen），temperature=0（检索为确定态，无回复采样方差，故单次判分）
用法:
    python -m eval.smoke_compare_e2e
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 重型依赖（mem0/fastembed/qdrant 等）在项目隔离依赖目录 pylibs，须优先加入路径
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # bge/bm25 均已缓存，强制离线加载，避免联网挂起

import json  # noqa: E402
import httpx  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402
from dba_pipeline.embedding.store import local_model_path  # noqa: E402
from frameworks import naive_rag  # noqa: E402

GRAPH = "data/dba_eval/smoke_companion.yaml"
QUERIES = [
    {"id": "q1", "query": "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……", "scenario": "work_pressure"},
    {"id": "q2", "query": "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？", "scenario": "health"},
    {"id": "q3", "query": "感觉压力好大，不知道怎么熬过这段日子……", "scenario": "emotion"},
]
TOP_K = 5  # 检索候选数（冒烟用；正式按 PAR 输出量对齐）

# ---- 陪伴场景 judge：直接评价「检索到的记忆原文」----
JUDGE_RAW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与「系统检索到的一段记忆原文」。\n"
               "请评估这段记忆对回应当下陪伴诉求的质量（不是评估回复，而是评估这段记忆作为陪伴素材的优劣），5 个维度各打 1-5 分：\n"
               "1. 相关贴合：记忆是否贴合用户当下的情绪/处境（5=很贴合，1=无关）\n"
               "2. 完整性：是否覆盖了支撑理解/安慰用户所需的关键记忆（5=覆盖充分，1=几乎缺失）\n"
               "3. 真实性：是否忠实于记忆内容、没有编造（5=完全忠实，1=明显编造）\n"
               "4. 连贯组织：记忆是否围绕一个主题连贯呈现、而非碎片堆砌（5=连贯，1=零散）\n"
               "5. 聚焦无冗余：是否贴合当前陪伴诉求、没有无关信息（5=干净，1=大量无关）\n"
               "只输出 JSON：{{\"相关贴合\":0,\"完整性\":0,\"真实性\":0,\"连贯组织\":0,\"聚焦\":0,\"total\":0}}"),
    ("human", "用户的倾诉：{query}\n\n检索到的记忆原文：\n{raw_text}"),
])


def build_judge_llm():
    """判分 judge：JUDGE_*，回退 OPENAI_*；temperature=0（检索确定态，单次判分即可）。"""
    return ChatOpenAI(
        model=os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        http_client=httpx.Client(trust_env=False),  # 绕过系统/环境代理，直连 judge 端点
        temperature=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )


def judge_raw(judge_llm, query: str, raw_text: str) -> dict:
    chain = JUDGE_RAW_PROMPT | judge_llm
    resp = chain.invoke({"query": query, "raw_text": raw_text})
    keys = ["相关贴合", "完整性", "真实性", "连贯组织", "聚焦"]
    try:
        data = json.loads(resp.content)
        vals = [float(data.get(k, 0)) for k in keys]
        data["total"] = round(sum(vals) / len(vals), 2)
        return data
    except json.JSONDecodeError:
        return {**{k: 0 for k in keys}, "total": 0.0}


def retrieve_raw(system, query, retriever=None, id2content=None, rag_store=None, mem=None) -> str:
    if system == "PAR":
        res = retriever.retrieve(query)
        return " ".join(c for _, c in res.get("peak_memories", [])[:TOP_K])
    if system == "RAG":
        hits = rag_store.search(query, k=TOP_K)
        return " ".join(id2content.get(mid, mid) for mid, _, _ in hits)
    if system == "Mem0":
        out = mem.search(query, top_k=TOP_K, filters={"user_id": "cmp"}, threshold=0.0)
        return " ".join(str(x.get("memory", "")) for x in out.get("results", []))
    return ""


def main() -> None:
    _load_dotenv(ROOT / ".env")
    # 保持 HF_HUB_OFFLINE=1（naive_rag/run_p_baseline import 时会置位）：bge/bm25 均已缓存，离线加载避免联网超时
    judge_llm = build_judge_llm()

    # PAR
    graph, retriever = build_retriever(GRAPH)
    # Naive RAG
    id2content = naive_rag.build_memories(GRAPH)
    rag_store = naive_rag.build_store(id2content)
    # Mem0（中文补丁）
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    # 清空持久化 qdrant 目录，避免各轮次重复数据混入（冒烟用全新状态）
    import shutil
    mem_dir = ROOT / "eval/.cache/mem0_qdrant_zh"
    if mem_dir.exists():
        shutil.rmtree(mem_dir, ignore_errors=True)
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "smoke_cmp", "path": str(ROOT / "eval/.cache/mem0_qdrant_zh"),
            "embedding_model_dims": 1024, "on_disk": True}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)
    for c in id2content.values():
        mem.add({"role": "user", "content": c}, user_id="cmp", infer=False)

    systems = {
        "PAR": dict(retriever=retriever),
        "RAG": dict(id2content=id2content, rag_store=rag_store),
        "Mem0": dict(mem=mem),
    }
    labels = {"PAR": "PAR", "RAG": "Naive RAG", "Mem0": "Mem0(zh)"}

    print("\n" + "=" * 60)
    print("各系统「检索原文」直接判分（1-5，judge=qwen）：")
    results = {}
    for sysname, kw in systems.items():
        rows = []
        for q in QUERIES:
            raw = retrieve_raw(sysname, q["query"], **kw)
            sc = judge_raw(judge_llm, q["query"], raw)
            rows.append((q["id"], sc["total"], len(raw)))
        results[labels[sysname]] = rows
        print(f"\n[{labels[sysname]}]")
        for qid, total, nlen in rows:
            print(f"   {qid} total={total}  原文{nlen}字")
        agg = sum(r[1] for r in rows) / len(rows)
        print(f"   系统均分 = {agg:.2f}")

    out = ROOT / "eval" / "outputs" / "smoke_judge_raw_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")


if __name__ == "__main__":
    main()
