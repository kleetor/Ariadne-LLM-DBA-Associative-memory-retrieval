#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组装「同题并排」三路判分输入：两臂 story 块 + 一个纯向量（Mem0 形态）块。

为什么这么组
------------
- **只换模型、只看 story 块**：两臂分别是 deepseek（thinking off）与 gpt-4o-mini 在同一份
  原始对话上跑完「抽取 → 建图 → 检索 → 叙事」整条链路的产物；本脚本只取最终
  ``space.search()`` 返回的叙事文本（story 块），不看图连边细节。
- **引入纯向量块**（形态对齐 Mem0 的 ``{memories}``，取 top-8 = Mem0 默认 ``top_k``）：
  三路并排比两路并排更能让 judge 有效区分 —— 两路时 rubric 的相对锚点只有彼此，
  容易把"差不多"判成"差距明显"（见 0908 e2e 文档 §四）。
- **系统名用中性标签（甲/乙/丙）**：0908 §四已实测中文模型对系统名存在可复现的名称反应
  （qwen +0.27~0.31、deepseek +0.49~0.56），故按该文档建议做匿名化；真实映射写入 meta。

块形态（对齐 0908 §2.4 的约定）
------------------------------
    deepseek / gpt-4o-mini → {"story": "<叙事文本>"}
    纯向量基线              → {"memories": ["<记忆条目>", ...]}

用法
----
    python -m eval.build_3way_context --corpus groupchat_mixed
    python -m eval.build_3way_context --reverse          # 交换显示顺序，做位置偏置对照
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from alm.engine import ALMEngine  # noqa: E402
from eval.exp_thinking_tiers import (  # noqa: E402
    DEFAULT_BUDGET,
    OUT_ROOT,
    USER_ID,
    _base_config,
    _load_corpus,
)
from eval.probe_llm_arm_ab import _vector_topk  # noqa: E402
from eval.test_alm_vs_mcp_e2e import match_facts  # noqa: E402

# 中性标签 → (真实说明, 臂目录名)。顺序即默认显示顺序。
ARMS: List[tuple] = [
    ("甲", ("deepseek（thinking off）", "off")),
    ("乙", ("gpt-4o-mini", "gpt4omini")),
]
VEC_LABEL = "丙"
VEC_NAME = "纯向量召回（Mem0 形态）"


def _load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"缺少检索产物：{path}（先跑 eval.exp_thinking_tiers --stage all）")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    if not rows:
        raise SystemExit(f"{path} 里没有逐查询结果")
    return {r["id"]: r for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="组装三路同题并排判分输入")
    parser.add_argument("--corpus", default="groupchat_mixed")
    parser.add_argument("--vec-k", type=int, default=8,
                        help="纯向量块的条数（默认 8 = Mem0 默认 top_k，见 0822 §3.3）")
    parser.add_argument("--ref-arm", default="off",
                        help="取哪一臂的空间做纯向量基线（节点集由该臂决定）")
    parser.add_argument("--reverse", action="store_true",
                        help="反转显示顺序（位置偏置对照）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = OUT_ROOT / args.corpus
    _, queries = _load_corpus(args.corpus)

    arm_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _label, (_name, arm) in ARMS:
        arm_rows[arm] = _load_rows(base / arm / "search.json")

    # 纯向量块：从参考臂的空间取 top-k（只调 embedding，不调 LLM）
    cfg = _base_config(DEFAULT_BUDGET["expand_k"], DEFAULT_BUDGET["max_hops"])
    cfg.data_dir = base / args.ref_arm / "data"
    engine = ALMEngine(cfg)
    try:
        space = engine.get_space(USER_ID, create=False)
        vec_by_id: Dict[str, List[str]] = {}
        for q in queries:
            _, texts = _vector_topk(space, q["query"], args.vec_k)
            vec_by_id[q["id"]] = list(texts)
    finally:
        engine.close()

    order = list(ARMS)
    if args.reverse:
        order = list(reversed(order))

    out_queries = []
    for q in queries:
        systems: Dict[str, Any] = {}
        for label, (_name, arm) in order:
            row = arm_rows[arm].get(q["id"])
            systems[label] = {"story": (row or {}).get("content", "")}
        systems[VEC_LABEL] = {"memories": vec_by_id.get(q["id"], [])}
        out_queries.append({
            "query_id": q["id"],
            "category": q.get("category", ""),
            "query": q["query"],
            "systems": systems,
        })

    mapping = {label: name for label, (name, _arm) in ARMS}
    mapping[VEC_LABEL] = f"{VEC_NAME}，top-{args.vec_k}（参考臂 {args.ref_arm}）"
    coverage = {}
    for _l, (name, arm) in ARMS:
        rows = arm_rows[arm]
        coverage[name] = {
            "gold_hit": sum(len(r.get("gold_hit") or []) for r in rows.values()),
            "gold_total": sum(r.get("gold_total") or 0 for r in rows.values()),
            "chars_mean": round(sum(len(r.get("content") or "") for r in rows.values()) / max(len(rows), 1), 1),
        }
    coverage[VEC_NAME] = {
        # 与两臂同口径的客观覆盖：把纯向量块的条目拼成文本后按同一 match_facts 判定。
        # 有了它才能判断「某臂与纯向量块分数接近」到底是"信息量相当"还是"判分没区分开"。
        "gold_hit": sum(len(match_facts("\n".join(vec_by_id.get(q["id"], [])),
                                        q.get("gold_facts") or []))
                        for q in queries),
        "gold_total": sum(len(q.get("gold_facts") or []) for q in queries),
        "chars_mean": round(
            sum(len("\n".join(v)) for v in vec_by_id.values()) / max(len(vec_by_id), 1), 1),
        "items": args.vec_k,
    }

    payload = {
        "meta": {
            "corpus": args.corpus,
            "display_order": [label for label, _ in order] + [VEC_LABEL],
            "mapping": mapping,
            "coverage": coverage,
            "块形态": {"story": "叙事文本（两臂）", "memories": "扁平记忆条目（纯向量）"},
        },
        "queries": out_queries,
    }

    out_path = Path(args.out) if args.out else (
        base / f"judge_input_3way{'_rev' if args.reverse else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[3way] 查询 {len(out_queries)} 条 | 显示顺序 {payload['meta']['display_order']}")
    for label, name in mapping.items():
        print(f"        {label} = {name}")
    print("[3way] 客观覆盖（供与 judge 结果交叉核对）:")
    for name, cov in coverage.items():
        detail = (f"gold {cov['gold_hit']}/{cov['gold_total']}" if "gold_hit" in cov
                  else f"{cov['items']} 条")
        print(f"        {name:<28} {detail:<16} 平均字符 {cov['chars_mean']}")
    print(f"[3way] 产物：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
