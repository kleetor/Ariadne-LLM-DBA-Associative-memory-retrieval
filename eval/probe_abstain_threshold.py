# -*- coding: utf-8 -*-
"""弃权阈值专项探针：在低余弦查询集上标定 ALM_ABSTAIN_COS

做法（关键：弃权只看 best_cos，不调 LLM，因此可以离线扫）：
  1. 令 ALM_ABSTAIN_COS=0（强制不弃权），把 4 段群聊写入 ALM，再逐条跑探针查询；
  2. 每条记录两项：该查询的 best_cos（候选与查询的最大余弦）+ 实际命中的 gold_fact；
  3. 任意阈值 t 下的表现由这两项直接推出：
       · 有答案题：best_cos >= t 才返回内容，否则整条归零（丢 A）
       · 无答案题：best_cos <  t 才算正确弃权（H）
     因此一次运行即可画出整条 A/H 权衡曲线，不需要为每个阈值重跑。

产出：
  · 逐条查询表（类型 / best_cos / 命中）
  · 两类样本的余弦分布（看是否重叠——重叠意味着不存在兼顾 A 与 H 的阈值）
  · 阈值扫描表 + 不同"无答题占比"下的最优阈值

用法：python -u -m eval.probe_abstain_threshold
"""

import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_p_baseline import _load_dotenv

DATA_DIR = ROOT / "data" / "dba_eval"
USER_ID = "eval:local:abstain:conv-0"
SESSION_ID = "eval:local:abstain:sample:0"

# 扫描网格：覆盖 0（不弃权）到 0.60（严于现行 0.52）
GRID = [0.00, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.60]

# 无答案题在整体里的占比（决定 A 与 H 的权重）。自建 comprehensive 集是 6:70 ≈ 8%。
RATIOS = [0.08, 0.20, 0.50]

_PUNCT_RE = re.compile(r"[\s，。、：:；;！!？?…—\-（）()「」『』\"'’‘“”·,.\[\]]")


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text or "").lower()


def match_facts(text: str, facts: List[str]) -> List[str]:
    norm = _norm(text)
    return [f for f in facts if _norm(f) in norm]


def load_all() -> tuple:
    with open(DATA_DIR / "groupchat_mixed_dialogs.yaml", encoding="utf-8") as f:
        dialogs = yaml.safe_load(f)["dialogs"]
    with open(DATA_DIR / "abstain_probe_queries.yaml", encoding="utf-8") as f:
        probes = yaml.safe_load(f)
    return dialogs, probes


def ingest(engine, dialogs: List[Dict[str, Any]]) -> None:
    from alm.contract import AddMessage, AddRequest

    base_ms = int(time.mktime(time.strptime("2026-09-01 09:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    for idx, dialog in enumerate(dialogs):
        messages = [
            AddMessage(
                role=turn["role"],
                content=f"[{turn['speaker']}] {turn['text']}",
                timestamp=base_ms + idx * 86_400_000 + i * 300_000,
            )
            for i, turn in enumerate(dialog["turns"])
        ]
        started = time.time()
        stats = engine.add(AddRequest(
            request_id=f"{USER_ID}:{dialog['id']}",
            user_id=USER_ID,
            session_id=SESSION_ID,
            messages=messages,
        ))
        print(f"  写入 {dialog['id']}: {stats.get('nodes')} 节点（{time.time() - started:.1f}s）")


def main() -> int:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        return 2

    dialogs, probes = load_all()

    from alm.config import ALMConfig
    from alm.contract import SearchRequest
    from alm.engine import ALMEngine

    cfg = ALMConfig.from_env()
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="abstain_probe_"))
    # 关键：强制不弃权，才能拿到每条查询真实的 best_cos 与"本可召回什么"
    cfg.abstain_cosine = 0.0
    engine = ALMEngine(cfg)
    print(f"# 数据目录: {cfg.data_dir}；探针查询 {len(probes)} 条；ALM_ABSTAIN_COS 强制为 0（采数用）")

    print("\n===== ① 写入 =====")
    ingest(engine, dialogs)

    print(f"\n===== ② 逐条采数（阈值 0，不弃权）=====")
    rows = []
    for item in probes:
        items = engine.search(SearchRequest(query=item["query"], user_id=USER_ID, top_k=100))
        best_cos = engine.retrieval_stats.last_cosine()
        text = " ".join(it.get("content") or "" for it in items)
        hits = match_facts(text, item.get("gold_facts") or [])
        rows.append({
            "id": item["id"], "type": item["type"], "note": item.get("note", ""),
            "query": item["query"], "facts": item.get("gold_facts") or [],
            "best_cos": best_cos if best_cos is not None else 0.0,
            "hits": hits, "returned": len(items),
            "top_contents": [(it.get("content") or "") for it in items[:12]],
        })
        print(f"  [{item['id']}] {item['type']:<9} cos={rows[-1]['best_cos']:.4f}  "
              f"命中 {len(hits)}/{len(item.get('gold_facts') or [])}  "
              f"返回 {len(items)} 条  | {item['query']}")

    answer = [r for r in rows if r["type"] == "answer"]
    no_answer = [r for r in rows if r["type"] == "no_answer"]
    facts_total = sum(len(r["facts"]) for r in answer)

    # ---- ③ 两类样本的余弦分布（重叠程度是核心诊断）----
    print("\n===== ③ 余弦分布（有答案 vs 无答案）=====")
    for label, group in (("有答案", answer), ("无答案", no_answer)):
        cos = sorted(r["best_cos"] for r in group)
        print(f"  {label}: n={len(cos)}  min={cos[0]:.4f}  "
              f"中位={cos[len(cos) // 2]:.4f}  max={cos[-1]:.4f}")
    overlap_lo = max(min(r["best_cos"] for r in answer), min(r["best_cos"] for r in no_answer))
    overlap_hi = min(max(r["best_cos"] for r in answer), max(r["best_cos"] for r in no_answer))
    if overlap_lo <= overlap_hi:
        clash = [r["id"] for r in answer if overlap_lo <= r["best_cos"] <= overlap_hi]
        clash += [r["id"] for r in no_answer if overlap_lo <= r["best_cos"] <= overlap_hi]
        print(f"  ⚠ 两类余弦区间重叠于 [{overlap_lo:.4f}, {overlap_hi:.4f}]"
              f"（涉及 {sorted(set(clash))}）→ 不存在能同时兼顾 A 与 H 的阈值")
    else:
        print(f"  ✓ 两类余弦区间不重叠，存在完美阈值区间 "
              f"[{max(r['best_cos'] for r in no_answer):.4f}, "
              f"{min(r['best_cos'] for r in answer):.4f})")

    # ---- ④ 阈值扫描 ----
    print("\n===== ④ 阈值扫描 =====")
    header = f"{'阈值':<7}{'A命中':>10}{'A召回':>9}{'H正确弃权':>12}{'H准确':>9}"
    for ratio in RATIOS:
        header += f"{'综合(' + str(int(ratio * 100)) + '%)':>12}"
    print(header)

    results = []
    for t in GRID:
        kept = sum(len(r["hits"]) for r in answer if r["best_cos"] >= t)
        recall_a = kept / facts_total if facts_total else 0.0
        zeroed = [r["id"] for r in answer if r["best_cos"] < t]
        correct_h = sum(1 for r in no_answer if r["best_cos"] < t)
        acc_h = correct_h / len(no_answer) if no_answer else 0.0
        combined = {
            ratio: (1 - ratio) * recall_a + ratio * acc_h for ratio in RATIOS
        }
        results.append({"t": t, "kept": kept, "recall_a": recall_a,
                        "zeroed": zeroed, "correct_h": correct_h, "acc_h": acc_h,
                        "combined": combined})
        line = (f"{t:<7.2f}{kept:>7}/{facts_total:<3}{recall_a:>8.1%}"
                f"{correct_h:>8}/{len(no_answer):<3}{acc_h:>8.1%}")
        for ratio in RATIOS:
            line += f"{combined[ratio]:>12.1%}"
        print(line)

    print("\n  被整条归零的可作答题（随阈值变化）:")
    for r in results:
        if r["zeroed"]:
            print(f"    阈值 {r['t']:.2f}: {r['zeroed']}")

    # ---- ⑤ 不同权重下的最优阈值 ----
    print("\n===== ⑤ 各权重下的最优阈值 =====")
    for ratio in RATIOS:
        best = max(results, key=lambda r: r["combined"][ratio])
        current = next(r for r in results if abs(r["t"] - 0.52) < 1e-9)
        print(f"  无答题占比 {ratio:.0%}: 最优阈值 {best['t']:.2f}"
              f"（A 召回 {best['recall_a']:.1%} / H 准确 {best['acc_h']:.1%}"
              f" / 综合 {best['combined'][ratio]:.1%}）"
              f"　现行 0.52 综合 {current['combined'][ratio]:.1%}")

    # ---- ⑥ 两段式弃权：模糊带交给 LLM 判定 ----
    # 判定结果只取决于 (查询, top_n)，与阈值无关；而入库/检索只做一次，
    # 因此这里对同一批候选同时扫多个 top_n —— 一次运行即可定参。
    from alm.judge import RelevanceJudge

    TOP_N_LIST = [3, 5, 8, 10]
    print(f"\n===== ⑥ 模糊带二次判定（同一批候选，top_n ∈ {TOP_N_LIST} 各判一次）=====")
    for r in rows:
        candidates = [{"content": c} for c in r["top_contents"]]
        r["judge"] = {}
        for n in TOP_N_LIST:
            judge = RelevanceJudge(engine.llm, top_n=n,
                                   max_chars=cfg.abstain_judge_max_chars)
            r["judge"][n] = judge.is_relevant(r["query"], candidates)
        marks = "  ".join(
            f"n={n}:{'相关' if r['judge'][n] is True else ('无关' if r['judge'][n] is False else '失败')}"
            for n in TOP_N_LIST
        )
        print(f"  [{r['id']}] {r['type']:<9} cos={r['best_cos']:.4f}  {marks}")

    def abstained(r: Dict[str, Any], t_low: float, t_hi: float, top_n: int) -> bool:
        if r["best_cos"] < t_low:
            return True                       # 硬下限：不判直接弃
        if r["best_cos"] < t_hi and r["judge"][top_n] is False:
            return True                       # 模糊带判否才弃（失败按放行）
        return False

    def evaluate(t_low: float, t_hi: float, top_n: int) -> Dict[str, Any]:
        kept = sum(len(r["hits"]) for r in answer
                   if not abstained(r, t_low, t_hi, top_n))
        recall_a = kept / facts_total if facts_total else 0.0
        correct_h = sum(1 for r in no_answer if abstained(r, t_low, t_hi, top_n))
        acc_h = correct_h / len(no_answer) if no_answer else 0.0
        missed = [r["id"] for r in answer if abstained(r, t_low, t_hi, top_n)]
        return {"t_low": t_low, "t_hi": t_hi, "top_n": top_n, "kept": kept,
                "recall_a": recall_a, "correct_h": correct_h, "acc_h": acc_h,
                "missed_answer": missed,
                "combined": {ratio: (1 - ratio) * recall_a + ratio * acc_h
                             for ratio in RATIOS}}

    def print_rows(combos: List[Dict[str, Any]], show_top_n: bool = False) -> None:
        header = f"{'t_low':<7}{'t_hi':<7}"
        if show_top_n:
            header += f"{'top_n':<7}"
        header += f"{'A命中':>10}{'A召回':>9}{'H正确弃权':>12}{'H准确':>9}"
        for ratio in RATIOS:
            header += f"{'综合(' + str(int(ratio * 100)) + '%)':>12}"
        print(header)
        for c in combos:
            line = f"{c['t_low']:<7.2f}{c['t_hi']:<7.2f}"
            if show_top_n:
                line += f"{c['top_n']:<7}"
            line += (f"{c['kept']:>7}/{facts_total:<3}{c['recall_a']:>8.1%}"
                     f"{c['correct_h']:>8}/{len(no_answer):<3}{c['acc_h']:>8.1%}")
            for ratio in RATIOS:
                line += f"{c['combined'][ratio]:>12.1%}"
            print(line)

    print("\n===== ⑦-1 判定宽度扫描（固定 t_low=0.35 / t_hi=0.50）=====")
    by_top_n = [evaluate(0.35, 0.50, n) for n in TOP_N_LIST]
    print_rows(by_top_n, show_top_n=True)
    for c in by_top_n:
        print(f"     top_n={c['top_n']:<3} 被误弃的可作答题: {c['missed_answer'] or '无'}")

    best_top_n = max(by_top_n, key=lambda c: c["combined"][0.20])["top_n"]
    print(f"\n===== ⑦-2 阈值网格（top_n={best_top_n}，按 20% 权重选出）=====")
    grid = [evaluate(t_low, t_hi, best_top_n)
            for t_low in (0.30, 0.35, 0.40)
            for t_hi in (0.45, 0.50, 0.55, 0.60)]
    print_rows(grid)

    print("\n  各权重下的最优组合（并与纯阈值对照）:")
    pure_040 = max(results, key=lambda r: -abs(r["t"] - 0.40))
    pure_052 = max(results, key=lambda r: -abs(r["t"] - 0.52))
    for ratio in RATIOS:
        best = max(by_top_n, key=lambda c: c["combined"][ratio])
        print(f"    无答题占比 {ratio:.0%}: 两段式 t_low=0.35 t_hi=0.50 "
              f"top_n={best['top_n']} → {best['combined'][ratio]:.1%}"
              f"　（纯阈值 0.40: {pure_040['combined'][ratio]:.1%}"
              f" / 现行 0.52: {pure_052['combined'][ratio]:.1%}）")

    # ---- ⑧ 判定调用的额外成本 ----
    band_hits = sum(1 for r in rows
                    if 0.35 <= r["best_cos"] < 0.55)
    print(f"\n===== ⑧ 判定成本 =====")
    print(f"  以 t_low=0.35 / t_hi=0.55 计，{len(rows)} 条查询中有 {band_hits} 条"
          f"（{band_hits / len(rows):.0%}）会触发一次判定 LLM 调用"
          f"（每条约 {cfg.abstain_judge_top_n} 个片段 × {cfg.abstain_judge_max_chars} 字符）")
    print(f"  低于 t_low 的 {sum(1 for r in rows if r['best_cos'] < 0.35)} 条不触发，"
          f"高于 t_hi 的 {sum(1 for r in rows if r['best_cos'] >= 0.55)} 条不触发")

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
