# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
e2e 冒烟（v2：整条系统链路对比，而非检索层对比）

对比三组【完整系统】，各系统产出「喂给下游模型的结构化记忆上下文（JSON 块）」，直接送 judge 评估这块记忆作为陪伴素材的质量：
  - Ariadne  ：PAR 检索 + StoryRank → {nodes, story}
  - Mem0     ：记忆层检索 → {memories}
  - Graphiti ：Zep 前置，图检索 → {entities, relations}
（构成 0831 计划 §4.5 的"真实回复层"评审替代：judge 直接看系统组装的记忆上下文块，而非检索字符串拼接。）

用法:
    python -m eval.smoke_system_compare
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 重型依赖（mem0/fastembed/qdrant 等）在项目隔离依赖目录 pylibs，须优先加入路径
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 模型已本地化，离线加载

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
TOP_K = 8

# ---- 陪伴场景 judge：三系统产物同题齐评，便于模型分辨好坏；3 个新维度 ----
JUDGE_JOINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与三个记忆系统（Ariadne / Mem0 / Graphiti）"
               "各自检索并组装、将喂给模型的结构化记忆上下文。请结合同一段倾诉，从下面 3 个维度，"
               "分别给每个系统打 1-5 分（三个放在一起更便于分辨好坏）：\n"
               "1. 记忆丰富度：记忆信息是否足够丰富、能支撑充分的陪伴（5=非常丰富，1=单薄贫乏）\n"
               "2. 易理解性：这段上下文是否条理清晰、便于直接用（5=一目了然，1=晦涩零乱）\n"
               "3. 用户建模度：能否据此准确理解用户、形成用户画像（5=很强，1=很弱）\n"
               "只输出 JSON：{{\"Ariadne\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"Mem0\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"Graphiti\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}}}}"),
    ("human", "用户的倾诉：{query}\n\n{blocks}"),
])


def build_judge_llm():
    return ChatOpenAI(
        model=os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        http_client=httpx.Client(trust_env=False),  # 绕开系统代理直连 judge 端点
        temperature=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _extract_json(text: str):
    """从模型输出中稳健提取 JSON 对象（容忍 markdown 围栏/前后缀文本）。"""
    import re
    if not text:
        return None
    t = text.strip()
    # 去掉 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    # 找到第一个平衡的 {...}
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


_JUDGE_KEYS = ["记忆丰富度", "易理解性", "用户建模度"]

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def count_tokens(text: str) -> int:
    """统一 token 计数（cl100k_base），作为「真实环境 token 用量」的代理，三系统共用同一标尺。"""
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text)  # 退化：近似为字符数


def judge_joint(judge_llm, query: str, blocks: dict) -> dict:
    """一次对话里同时给出三个系统产物，让 judge 在一起比对后各打 1-5 分（3 维）。"""
    rendered = "\n\n".join(f"## {name}\n{json.dumps(blk, ensure_ascii=False)}" for name, blk in blocks.items())
    chain = JUDGE_JOINT_PROMPT | judge_llm
    resp = chain.invoke({"query": query, "blocks": rendered})
    data = _extract_json(resp.content)
    out = {}
    for name in blocks:
        if data is not None:
            d = data.get(name, {}) if isinstance(data, dict) else {}
            vals = [float(d.get(k, 0)) for k in _JUDGE_KEYS]
            out[name] = {**{k: float(d.get(k, 0)) for k in _JUDGE_KEYS}, "total": round(sum(vals) / len(vals), 2)}
        else:
            out[name] = {**{k: 0 for k in _JUDGE_KEYS}, "total": 0.0}
    return out


def main() -> None:
    _load_dotenv(ROOT / ".env")
    os.environ["HF_HUB_OFFLINE"] = "1"
    judge_llm = build_judge_llm()

    # ---- Ariadne（完整链路：PAR 检索 + StoryRank）----
    print("setup: build_retriever ...", flush=True)
    graph, retriever = build_retriever(GRAPH)
    print("setup: retriever ok", flush=True)

    # ---- Mem0（记忆层，中文补丁）----
    import shutil
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    id2content = naive_rag.build_memories(GRAPH)
    mem_dir = ROOT / "eval/.cache/mem0_qdrant_sys"
    if mem_dir.exists():
        shutil.rmtree(mem_dir, ignore_errors=True)
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "sys_cmp", "path": str(mem_dir),
            "embedding_model_dims": 1024, "on_disk": True}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)
    print("setup: mem0 from_config ok", flush=True)
    for c in id2content.values():
        mem.add({"role": "user", "content": c}, user_id="cmp", infer=False)
    print("setup: mem0 add ok", flush=True)

    # ---- Graphiti：读取独立 venv 预生成的结构化上下文块（facts/relations）----
    graphiti_blocks = json.loads((ROOT / "eval/outputs/smoke_graphiti_blocks.json").read_text(encoding="utf-8"))

    # ---- 各系统「模型在上下文中看到的内容」提取函数 ----
    # Ariadne：检索工具 query_memory 返回给模型的核心是 stories（一段记忆故事文本）。
    #            story_nodes/discarded_nodes 等是审计字段，不在模型上下文里——只取 story。
    def ariadne_block(query: str, qid: str = "") -> dict:
        r = retriever.retrieve_with_story(query, with_response=False)
        stories = r.get("stories", [])
        return {"story": (stories[0] if stories else "")}

    def mem0_block(query: str, qid: str = "") -> dict:
        out = mem.search(query, top_k=TOP_K, filters={"user_id": "cmp"}, threshold=0.0)
        mems = [{"content": str(x.get("memory", ""))} for x in out.get("results", [])]
        return {"memories": mems}

    def graphiti_block(query: str, qid: str = "") -> dict:
        gb = graphiti_blocks.get(qid, {})
        return {"entities": gb.get("entities", []), "relations": gb.get("relations", [])}

    systems = {
        "Ariadne": ariadne_block,
        "Mem0": mem0_block,
        "Graphiti": graphiti_block,
    }

    print("\n" + "=" * 66)
    print("整条系统链路对比：三系统产物同题齐评（judge=qwen，3 维 1-5 分）")
    print("外部指标：真实环境 token 用量（cl100k_base，喂给模型的结构化上下文 token 数）")
    score_rows = {name: [] for name in systems}
    token_rows = {name: [] for name in systems}
    for q in QUERIES:
        blocks = {name: fn(q["query"], qid=q["id"]) for name, fn in systems.items()}
        sc = judge_joint(judge_llm, q["query"], blocks)
        toks = {name: count_tokens(json.dumps(blk, ensure_ascii=False)) for name, blk in blocks.items()}
        print(f"\n[{q['id']}] {q['scenario']}")
        for name in systems:
            score_rows[name].append(sc[name]["total"])
            token_rows[name].append(toks[name])
            print(f"  {name:<9} total={sc[name]['total']:.2f}  dims={ {k: sc[name][k] for k in _JUDGE_KEYS} }  上下文token={toks[name]}")

    print("\n" + "=" * 66)
    print("汇总")
    summary = {}
    for name in systems:
        s_avg = sum(score_rows[name]) / len(score_rows[name])
        t_sum = sum(token_rows[name])
        t_avg = sum(token_rows[name]) / len(token_rows[name])
        summary[name] = {"judge_avg": round(s_avg, 2), "token_total": t_sum, "token_avg_per_query": round(t_avg, 1)}
        print(f"  {name:<9} 判分均分={s_avg:.2f}  总token={t_sum}  每题平均token={t_avg:.1f}")

    out = ROOT / "eval/outputs/smoke_system_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_query_scores": score_rows, "per_query_tokens": token_rows, "summary": summary}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")


if __name__ == "__main__":
    main()
