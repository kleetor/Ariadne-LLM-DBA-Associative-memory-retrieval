# -*- coding: utf-8 -*-
"""E5 专用 judge：四维 rubric（含「意图贴合度」），同题并排。

为什么不能复用 `judge_comprehensive` 的三维 rubric：
E1/E3 评的是"这段记忆好不好用"，而 E5 要回答的是"这段记忆是否**切中此刻这个意图**"。
同一个上下文块在"倾诉"意图下可能优秀、在"规划"意图下可能完全跑偏——三维 rubric 里
没有任何一维能表达这种"意图—内容错配"。故新增第四维，且它是 E5 的主指标。

其余协议与 E1/E3 完全一致（同题并排、匿名、temperature=0、JSON 输出），
校准与可比性都继承自 `judge_comprehensive`：直接复用其 `build_judge_llm` /
`_extract_json` / `_align_keys`（含 leet 混淆还原）。

用法：
    python -m eval.judge_e5_intent --ctx eval/outputs/e5/context_e5_intent.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from eval.judge_comprehensive import (  # noqa: E402
    _align_keys, _extract_json, build_judge_llm, _load_dotenv,
)

DIMS = ["意图贴合度", "记忆丰富度", "易理解性", "用户建模度"]

_CTX = "eval/outputs/e5/context_e5_intent.json"
_OUT = "eval/outputs/e5/judge_e5_intent.json"

INTENT_LABEL = {
    "vent": "倾诉 / 共情",
    "plan": "规划 / 行动",
    "recommend": "推荐 / 偏好",
}

_JUDGE_SYSTEM = """你是情感陪伴场景的记忆评测员。下面给出「用户当前的意图」、他的原话，以及两个记忆系统
各自检索并组装、将喂给模型的结构化记忆上下文。请结合同一段话，从下面 4 个维度，
分别给每个系统打 1-5 分（放在一起更便于分辨好坏）：

1. **意图贴合度**（本实验的主维度）：这段上下文是否切中用户**此刻**想做的事——
   - 若意图是「倾诉/共情」：应唤起情绪、状态、同类经历，能支撑共情；
   - 若意图是「规划/行动」：应唤起可执行的动作、时间线、未完成事项，能支撑排计划；
   - 若意图是「推荐/偏好」：应唤起偏好、选项、约束条件，能支撑给建议。
   只在「泛泛相关的旧记忆」上堆量、但没对准当前意图的，本维度低分（5=非常贴合，1=完全跑偏）。
2. 记忆丰富度：记忆信息是否足够丰富、能支撑充分的陪伴（5=非常丰富，1=单薄贫乏）
3. 易理解性：这段上下文是否条理清晰、便于直接用（5=一目了然，1=晦涩零乱）
4. 用户建模度：能否据此准确理解用户、形成用户画像（5=很强，1=很弱）

若上下文中出现关系/边/节点类型标签（如 CAUSAL/SCENARIO/SEQUENCE/PREFERENCE/SOCIAL/
ATTRIBUTE/TEMPORAL/TAXONOMIC，或 STATUS/REASON/ACTION/THING/PERSON/EMOTION），
其含义依次为：因果 / 同场景 / 时序先后 / 偏好 / 社交 / 属性 / 时间定位 / 分类，以及
客观处境 / 原因 / 行为 / 事物 / 人物 / 情绪。请据此理解结构，而非按字面臆测。

只输出 JSON："""

_JUDGE_HUMAN = "用户当前意图：{intent}\n\n用户的原话：{query}\n\n{blocks}"


def build_prompt(names) -> ChatPromptTemplate:
    schema = ",".join(
        '"{}":{{"意图贴合度":0,"记忆丰富度":0,"易理解性":0,"用户建模度":0,"total":0}}'.format(n)
        for n in names
    )
    system = (_JUDGE_SYSTEM + "{" + schema + "}").replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([("system", system), ("human", _JUDGE_HUMAN)])


def judge_row(llm, prompt, row) -> dict:
    rendered = "\n\n".join(
        f"## {name}\n{json.dumps(blk, ensure_ascii=False)}"
        for name, blk in row["blocks"].items())
    intent = INTENT_LABEL.get(row["intent"], row["intent"])
    blocks = row["blocks"]
    last = None
    for att in range(5):
        try:
            resp = prompt.invoke({"intent": intent, "query": row["query"], "blocks": rendered})
            resp = llm.invoke(resp)
            data = _extract_json(resp.content) if resp is not None and resp.content else None
            aligned = _align_keys(data, list(blocks)) if isinstance(data, dict) else {}
            # 有效判分：**意图贴合度**必须非 0（它是主维度），避免解析失败被当成"跑偏"
            if any(float((aligned.get(n) or {}).get("意图贴合度", 0) or 0) > 0 for n in blocks):
                last = data
                break
            last = data
        except Exception:
            last = None
        if att < 4:
            time.sleep(2 * (att + 1))
    data = _align_keys(last, list(blocks)) if isinstance(last, dict) else {}
    out = {}
    for name in blocks:
        d = data.get(name, {})
        vals = [float(d.get(k, 0) or 0) for k in DIMS]
        out[name] = {**{k: float(d.get(k, 0) or 0) for k in DIMS},
                     "total": round(sum(vals) / len(vals), 2)}
    return out


def load_ctx(path: str):
    with open(ROOT / path, encoding="utf-8") as f:
        data = json.load(f)
    keymap = {"par": "PAR", "vector": "Vector"}
    rows = []
    for r in data["queries"]:
        rows.append({
            "query_id": r["query_id"],
            "scenario": r.get("scenario", ""),
            "intent": r.get("intent", ""),
            "query": r["query"],
            "blocks": {keymap.get(k, k): v for k, v in (r.get("systems") or {}).items()},
            "label_map": r.get("label_map"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 意图贴合度 judge（四维）")
    ap.add_argument("--ctx", default=_CTX)
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--thinking", type=str, default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    llm = build_judge_llm(args.model, args.api_key, args.api_base,
                          thinking=bool(args.thinking == "1") if args.thinking is not None else None,
                          reasoning_effort=args.reasoning_effort,
                          max_retries=args.max_retries)
    rows = load_ctx(args.ctx)
    if args.limit:
        rows = rows[: args.limit]
    names = sorted({n for r in rows for n in r["blocks"]})
    prompt = build_prompt(names)
    print(f"[E5-judge] 查询 {len(rows)} 条 judge={llm.model_name} 系统={names}", flush=True)

    results = []
    for r in rows:
        sc = judge_row(llm, prompt, r)
        results.append({**{k: r[k] for k in ("query_id", "scenario", "intent", "query")},
                        "scores": sc, "label_map": r.get("label_map")})
        print(f"[E5-judge] {r['query_id']} ({r['intent']}) -> "
              + "  ".join(f"{n}={sc[n]['total']:.2f}" for n in r["blocks"]), flush=True)

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"judge_model": llm.model_name, "ctx": args.ctx, "rubric": DIMS,
                 "mode": "joint_1pass", "names": names},
        "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E5-judge] 结果已写: {p}（{len(results)} 条）")


if __name__ == "__main__":
    main()
