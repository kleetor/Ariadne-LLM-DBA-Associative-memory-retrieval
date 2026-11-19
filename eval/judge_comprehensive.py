# -*- coding: utf-8 -*-
"""
综合能力测评 · 三系统结构化记忆上下文块 judge 判分（同题并排一次判定）

按 0831 e2e 计划 §4.5.3/§4.5.4：
- 评测对象 = 系统喂给模型的结构化记忆上下文块（非 StoryRank 故事、非回复）。
- judge = 同题并排一次判定：每题把三系统上下文块一起交给 judge，各打 1-5 分。
- 维度（陪伴场景 rubric）：记忆丰富度 / 易理解性 / 用户建模度。
- 输入：eval/outputs/context_comprehensive_3sys.json（已组装的上下文块）。

§8.2 独立模型交叉：本脚本支持 --model 指定 judge 模型；主判分与交叉验证分开跑，
再用 eval/score_comprehensive.py 汇总一致性（排位一致率/kappa）。

用法:
    python -m eval.judge_comprehensive                        # 主判分（.env JUDGE_*）
    python -m eval.judge_comprehensive --model gpt-4o ...     # 交叉验证（非 deepseek 系）
    python -m eval.judge_comprehensive --out eval/outputs/judge_comprehensive.json
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CTX = "eval/outputs/context_comprehensive_3sys.json"
OUT = "eval/outputs/judge_comprehensive.json"

# 陪伴场景 judge：三系统产物同题齐评，3 个维度（§4.5.4）
JUDGE_JOINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与三个记忆系统（Ariadne / Mem0 / Graphiti）"
               "各自检索并组装、将喂给模型的结构化记忆上下文。请结合同一段倾诉，从下面 3 个维度，"
               "分别给每个系统打 1-5 分（三个放在一起更便于分辨好坏）：\n"
               "1. 记忆丰富度：记忆信息是否足够丰富、能支撑充分的陪伴（5=非常丰富，1=单薄贫乏）\n"
               "2. 易理解性：这段上下文是否条理清晰、便于直接用（5=一目了然，1=晦涩零乱）\n"
               "3. 用户建模度：能否据此准确理解用户、形成用户画像（5=很强，1=很弱）\n"
               "若上下文中出现关系/边/节点类型标签（如 CAUSAL/SCENARIO/SEQUENCE/PREFERENCE/SOCIAL/"
               "ATTRIBUTE/TEMPORAL/TAXONOMIC，或 STATUS/REASON/ACTION/THING/PERSON/EMOTION），"
               "其含义依次为：因果 / 同场景 / 时序先后 / 偏好 / 社交 / 属性 / 时间定位 / 分类，以及"
               "客观处境 / 原因 / 行为 / 事物 / 人物 / 情绪。请据此理解结构，而非按字面臆测。\n"
               "只输出 JSON：{{\"Ariadne\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"Mem0\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"Graphiti\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}}}}"),
    ("human", "用户的倾诉：{query}\n\n{blocks}"),
])

_JUDGE_KEYS = ["记忆丰富度", "易理解性", "用户建模度"]
_NAMES = ["Ariadne", "Mem0", "Graphiti"]


def build_joint_prompt(names):
    """按**实际系统名**构造同题并排 prompt。

    为什么不直接把上面的 ``JUDGE_JOINT_PROMPT`` 参数化：它的输出 JSON 示例里写死了
    Ariadne / Mem0 / Graphiti，改写成动态构造会让**历史判分的提示词逐字变化**，
    已存档的 judge_*.json 便不再可比。故原常量逐字保留，需要其它系统名时用本函数另建。
    （rubric 三个维度与原文保持一致。）
    """
    listed = " / ".join(names)
    schema = ",".join(
        '"{}":{{"记忆丰富度":0,"易理解性":0,"用户建模度":0,"total":0}}'.format(n)
        for n in names
    )
    body = (
        "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与{}个记忆系统（{}）"
        "各自检索并组装、将喂给模型的结构化记忆上下文。请结合同一段倾诉，从下面 3 个维度，"
        "分别给每个系统打 1-5 分（放在一起更便于分辨好坏）：\n"
        "1. 记忆丰富度：记忆信息是否足够丰富、能支撑充分的陪伴（5=非常丰富，1=单薄贫乏）\n"
        "2. 易理解性：这段上下文是否条理清晰、便于直接用（5=一目了然，1=晦涩零乱）\n"
        "3. 用户建模度：能否据此准确理解用户、形成用户画像（5=很强，1=很弱）\n"
        "若上下文中出现关系/边/节点类型标签（如 CAUSAL/SCENARIO/SEQUENCE/PREFERENCE/SOCIAL/"
        "ATTRIBUTE/TEMPORAL/TAXONOMIC，或 STATUS/REASON/ACTION/THING/PERSON/EMOTION），"
        "其含义依次为：因果 / 同场景 / 时序先后 / 偏好 / 社交 / 属性 / 时间定位 / 分类，以及"
        "客观处境 / 原因 / 行为 / 事物 / 人物 / 情绪。请据此理解结构，而非按字面臆测。\n"
        "只输出 JSON："
    ).format(len(names), listed)
    # langchain 的 ChatPromptTemplate 会把 {..} 当模板变量，故示例 JSON 的花括号必须转义成 {{ }}
    system = (body + "{" + schema + "}").replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([("system", system),
                                             ("human", "用户的倾诉：{query}\n\n{blocks}")])


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def build_judge_llm(model=None, api_key=None, api_base=None, thinking=None,
                    reasoning_effort=None, max_retries=None):
    # thinking=None 时用默认（见 build_judge_extra）；部分模型（如 gpt-5.6-luna）不支持该参数，置 False 关闭
    use_thinking = os.environ.get("JUDGE_THINKING", "1") if thinking is None else ("1" if thinking else "0")
    extra = {"thinking": {"type": "disabled"}} if use_thinking == "1" else {}
    # reasoning_effort：不支持 thinking 开关的模型（gpt-5.6-luna）只能靠它压推理 token；
    # 这类模型每次仍会花 100~300 reasoning tokens，墙钟时间主要由网关排队决定。
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(
        model=model or os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=api_base or os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        # 必须显式设超时：httpx 默认对**读**不设上限，网关偶发 hang 会让整批判分永久卡住
        # （实测 gemini 在 q57 之后停摆 76 分钟、无任何报错）。读超时给 180s，足以覆盖最慢的单题。
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(180.0, connect=15.0)),
        temperature=0,
        max_tokens=512,
        max_retries=max_retries if max_retries is not None else 2,
        extra_body=extra,
    )


def _extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        t = m.group(1)
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


def _norm_name(s: str) -> str:
    """系统名归一化：小写 + leet 数字还原 + 去掉非字母数字。"""
    table = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                           "5": "s", "7": "t", "8": "b", "9": "g"})
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(s).lower().translate(table))


def _align_keys(raw: dict, expected) -> dict:
    """把模型返回的 JSON 键对齐到预期系统名。

    为什么需要：部分 judge 会对系统名做**字符混淆**——实测 gemini-3.x 把
    ``Ariadne`` 返回成 ``4r14dn3``、``Graphiti`` 返回成 ``Gr4ph1t1``（数字替换形近字母），
    严格按 key 取值会把这两个系统判成 0 分（E1 首跑：Ariadne/Graphiti 大面积 0，
    Mem0 正常）。层级：精确匹配 → 归一化（含 leet 还原）→ 相似度 ≥0.8 兜底。
    """
    if not isinstance(raw, dict):
        return {}
    out, used = {}, set()
    for name in expected:
        if name in raw:
            out[name] = raw[name]
            used.add(name)
    norm_map = {}
    for k in raw:
        if k not in used:
            norm_map.setdefault(_norm_name(k), k)
    for name in expected:
        if name in out:
            continue
        k = norm_map.get(_norm_name(name))
        if k is not None and k not in used:
            out[name] = raw[k]
            used.add(k)
    for name in expected:
        if name in out:
            continue
        best, best_r = None, 0.0
        for k in raw:
            if k in used:
                continue
            r = difflib.SequenceMatcher(None, _norm_name(name), _norm_name(k)).ratio()
            if r > best_r:
                best, best_r = k, r
        if best is not None and best_r >= 0.8:
            out[name] = raw[best]
            used.add(best)
    return out


def judge_joint(judge_llm, query: str, blocks: dict, prompt=None) -> dict:
    """同题并排：一次对话里同时给出三系统上下文块，judge 各打 1-5 分（3 维）。

    ``prompt`` 为 None 时用模块级默认 ``JUDGE_JOINT_PROMPT``（历史行为）；
    系统名不是 Ariadne/Mem0/Graphiti 时，用 ``build_joint_prompt(names)`` 另建后传入。

    代理（gpt.ge 等）会间歇返回 401/超时/无效或空内容；这里对「异常」与「解析不出有效分（全0）」都重试。"""
    rendered = "\n\n".join(f"## {name}\n{json.dumps(blk, ensure_ascii=False)}" for name, blk in blocks.items())
    chain = (prompt or JUDGE_JOINT_PROMPT) | judge_llm
    last_data = None
    for att in range(5):
        try:
            resp = chain.invoke({"query": query, "blocks": rendered})
            data = _extract_json(resp.content) if (resp is not None and getattr(resp, "content", None)) else None
            # 有效判分：至少一个系统有非 0 分数（否则视为解析失败/空响应）。
            # 注意先做键名对齐，否则被混淆的键（4r14dn3）会被当成"全 0"而白白重试 5 次。
            aligned = _align_keys(data, list(blocks)) if isinstance(data, dict) else {}
            if any(float((aligned.get(n) or {}).get("total", 0) or 0) > 0 for n in blocks):
                last_data = data
                break
            last_data = data
        except Exception:
            last_data = None
        if att < 4:
            time.sleep(2 * (att + 1))
    data = _align_keys(last_data, list(blocks)) if isinstance(last_data, dict) else {}
    out = {}
    for name in blocks:
        d = data.get(name, {})
        vals = [float(d.get(k, 0)) for k in _JUDGE_KEYS]
        out[name] = {**{k: float(d.get(k, 0)) for k in _JUDGE_KEYS}, "total": round(sum(vals) / len(vals), 2)}
    return out


def load_context(ctx_path: str) -> dict:
    with open(ROOT / ctx_path, encoding="utf-8") as f:
        data = json.load(f)
    # 统一系统名映射（key 可能是 ariadne/mem0/graphiti 或 Ariadne/Mem0/Graphiti）
    keymap = {"ariadne": "Ariadne", "mem0": "Mem0", "graphiti": "Graphiti"}
    queries = []
    for r in data["queries"]:
        row = {"query_id": r["query_id"], "category": r.get("category", ""),
               "query": r["query"], "blocks": {}}
        for k, blk in r.get("systems", {}).items():
            row["blocks"][keymap.get(k, k)] = blk
        queries.append(row)
    return {"queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(description="三系统上下文块 judge 判分（同题并排）")
    parser.add_argument("--model", default=None, help="覆盖 judge 模型（交叉验证用非 deepseek 系）")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--ctx", default=CTX, help="上下文块 JSON 路径（默认 story 版）")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--thinking", type=str, default=None, help="thinking 开关：1=发 disabled(默认)，0=不发（供不支持该参数的模型，如 gpt-5.6-luna）")
    parser.add_argument("--reasoning-effort", default=None,
                        help="透传 reasoning_effort（如 low）：不支持 thinking 开关的模型用它压低推理 token 与延迟")
    parser.add_argument("--max-retries", type=int, default=None, help="SDK 重试次数（默认 2；排障可设 0）")
    parser.add_argument("--limit", type=int, default=None, help="只判前 N 条（冒烟测）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    judge_llm = build_judge_llm(args.model, args.api_key, args.api_base,
                                thinking=bool(args.thinking == "1") if args.thinking is not None else None,
                                reasoning_effort=args.reasoning_effort,
                                max_retries=args.max_retries)
    ctx = load_context(args.ctx)
    if args.limit:
        ctx["queries"] = ctx["queries"][:args.limit]
    # 系统名不是默认三系统（Ariadne/Mem0/Graphiti）时，按实际名字动态构造 prompt
    names = sorted({n for r in ctx["queries"] for n in r["blocks"]})
    prompt = None
    if names and names != sorted(_NAMES):
        prompt = build_joint_prompt(names)
        print(f"[judge] 系统名 {names} ≠ 默认 {_NAMES}，改用动态构造的 prompt")
    print(f"[judge] 查询 {len(ctx['queries'])} 条，judge_model={judge_llm.model_name}, ctx={args.ctx}")

    results = []
    for r in ctx["queries"]:
        qid = r["query_id"]
        sc = judge_joint(judge_llm, r["query"], r["blocks"], prompt=prompt)
        results.append({"query_id": qid, "category": r["category"], "query": r["query"],
                        "scores": sc, "label_map": r.get("label_map")})
        print(f"[judge] {qid} ({r['category']}) -> "
              "  ".join(f"{n}={sc[n]['total']:.2f}" for n in r["blocks"]), flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"judge_model": judge_llm.model_name, "ctx": args.ctx,
                        "rubric": _JUDGE_KEYS, "mode": "joint_1pass", "names": names},
               "results": results}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[judge] 结果已写: {out_path}（{len(results)} 条）")


if __name__ == "__main__":
    main()
