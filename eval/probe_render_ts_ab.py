# -*- coding: utf-8 -*-
"""ALM_RENDER_TIMESTAMPS 的 A/B 补测：只比 ALM 自身。

动机：0920 把 `render_timestamps` 默认从 False 改为 True（官方 Answer 模板第 7 条要求把
yesterday / last month 这类相对时间**换算成日期**，前提是记忆里带时间）。但重跑时观测到
q11 疑似把**记录时间当成了事件时间**（答"这周二提测"，gold 是"周三"）。本探针在同一份记忆
上只切换这一个开关，比对回答级事实覆盖，并把时序题原文打出来供人工判定。

成本控制：**只写入 ALM 一次**；两个引擎（False / True）共用同一 data_dir 做只读检索；
不跑 MCP 侧。

⚠️ 判分列只作参考：官方 `ACCURACY_PROMPT` 的 "Lists of DISTINCT facts" 规则要求多余项算错，
而本地 gold 是 fact 列表，会使判分系统性偏低（见 Plan §7.10.5 E）。主指标是事实覆盖。

用法：python -u -m eval.probe_render_ts_ab [--queries q2,q11,q13]
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_p_baseline import _load_dotenv
from eval.test_alm_vs_mcp_e2e import load_dataset, match_facts
from eval.score_alm_vs_mcp_answer import build_models, invoke, judge_answer, render_answer_prompt

USER_ID = "eval:local:render_ts_ab:conv-0"
SESSION_ID = "eval:local:render_ts_ab:sample:0"
TEMPORAL_QUERIES = {"q2", "q11", "q13"}


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM_RENDER_TIMESTAMPS 的 A/B 补测")
    parser.add_argument("--queries", default=None, help="只跑这些查询 id（逗号分隔）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL", file=sys.stderr)
        return 2

    dialogs, queries = load_dataset()
    queries = [q for q in queries if q.get("gold_facts")]
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]

    from alm.config import ALMConfig
    from alm.contract import AddMessage, AddRequest, SearchRequest
    from alm.engine import ALMEngine

    answer_llm, judge_llm, judge_model, answer_model = build_models()

    data_dir = Path(tempfile.mkdtemp(prefix="render_ts_ab_"))
    print(f"# 数据目录: {data_dir}")
    print(f"# Answer 模型: {answer_model} | Judge: {judge_model}")
    print(f"# 查询 {len(queries)} 条（A/B 各跑一遍，同一份记忆）")

    print("\n===== ① 写入（仅 ALM，一次）=====")
    cfg_off = ALMConfig.from_env()
    cfg_off.data_dir = data_dir
    cfg_off.render_timestamps = False
    engine_off = ALMEngine(cfg_off)

    base_ms = int(time.mktime(time.strptime("2026-09-01 09:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    for idx, dialog in enumerate(dialogs):
        engine_off.add(AddRequest(
            request_id=f"{USER_ID}:{dialog['id']}", user_id=USER_ID, session_id=SESSION_ID,
            messages=[AddMessage(role=t["role"], content=f"[{t['speaker']}] {t['text']}",
                                 timestamp=base_ms + idx * 86_400_000 + i * 300_000)
                      for i, t in enumerate(dialog["turns"])],
        ))
        print(f"  {dialog['id']} 写入完成")

    # 第二个引擎：同 data_dir、只改 render_timestamps（只读检索）
    cfg_on = ALMConfig.from_env()
    cfg_on.data_dir = data_dir
    cfg_on.render_timestamps = True
    engine_on = ALMEngine(cfg_on)
    print(f"\n  render_ts=False -> {engine_off.config.render_timestamps}"
          f" | render_ts=True -> {engine_on.config.render_timestamps}")

    print("\n===== ② 检索 → Answer → Judge =====")
    rows: List[Dict[str, Any]] = []
    for item in queries:
        query, facts = item["query"], item["gold_facts"]
        row: Dict[str, Any] = {"id": item["id"], "category": item["category"],
                               "query": query, "facts": facts}
        for tag, engine in (("off", engine_off), ("on", engine_on)):
            items = engine.search(SearchRequest(query=query, user_id=USER_ID, top_k=100))
            memories = "\n".join(it.get("content") or "" for it in items)
            generated = invoke(answer_llm, render_answer_prompt(memories, query))
            row[tag] = {
                "answer": generated,
                "fact_hits": len(match_facts(generated, facts)),
                "facts_total": len(facts),
                "label": judge_answer(judge_llm, query, "、".join(facts), generated),
                "mem_chars": len(memories),
                "with_ts": sum(1 for it in items if it.get("created_at")),
                "n_items": len(items),
                "recorded": memories.count("记录于"),
            }
        rows.append(row)
        print(f"\n  [{row['id']}] {row['category']}｜{query}")
        print(f"    gold: {facts}")
        for tag, name in (("off", "TS=0"), ("on", "TS=1")):
            d = row[tag]
            print(f"    {name} 覆盖={d['fact_hits']}/{d['facts_total']} 判分={d['label']:<9} "
                  f"条目={d['n_items']} 带时间={d['with_ts']} 叙述含「记录于」={d['recorded']}")
            print(f"         回答: {d['answer'][:120]}")

    print("\n===== ③ 汇总 =====")
    total = len(rows)
    fact_total = sum(len(r["facts"]) for r in rows)
    print(f"  查询 {total} 条 / gold_facts 共 {fact_total} 项")
    for tag, name in (("off", "render_ts=False"), ("on", "render_ts=True ")):
        hits = sum(r[tag]["fact_hits"] for r in rows)
        correct = sum(1 for r in rows if r[tag]["label"] == "CORRECT")
        rec = sum(r[tag]["recorded"] for r in rows)
        print(f"  {name}  事实覆盖 {hits}/{fact_total} = {hits / fact_total:6.1%}"
              f"   |  判分(仅参考) {correct}/{total}"
              f"   |  叙述含「记录于」共 {rec} 次")

    print("\n  逐条（TS=0 覆盖 | TS=1 覆盖）:")
    delta = 0
    for r in rows:
        a, b = r["off"]["fact_hits"], r["on"]["fact_hits"]
        delta += b - a
        flag = "←" if b != a else " "
        print(f"    {r['id']:<5} {a}/{r['off']['facts_total']} | {b}/{r['on']['facts_total']}  {flag} {r['category']}")
    print(f"\n  净变化: {delta:+d} 项")

    print("\n===== ④ 时序题原文（人工判定是否把记录时间当事件时间）=====")
    for r in rows:
        if r["id"] not in TEMPORAL_QUERIES:
            continue
        print(f"\n  [{r['id']}] {r['query']}\n    gold: {r['facts']}")
        for tag, name in (("off", "TS=0"), ("on", "TS=1")):
            print(f"    {name} [{r[tag]['fact_hits']}/{r[tag]['facts_total']}] {r[tag]['answer']}")

    engine_on.close()
    engine_off.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
