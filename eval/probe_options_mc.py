# -*- coding: utf-8 -*-
"""选择题 options 专项探针：e2e 图上 A/B「弃权保护」与「选项对比聚合」

背景（见 Plan 与 api-guide）：文本赛道含 Streaming 选择题，平台只在选择题的 Search
顶层下发 `options`（不含金标），开放题不下发。ALM 侧此前把 options 解析后直接丢弃，
本探针用于判定两个新增档位是否有增益：

    guard     options 存在时跳过两段式弃权（选项题按定义可回答，题干偏短易被误弃权）
    contrast  把「各选项与候选的最大余弦」并入层内语义分（题干欠定时补判别信号）

数据：与 eval/test_alm_vs_mcp_e2e.py 完全同源——graph 与查询集都用 e2e 那套
（groupchat_mixed_dialogs / groupchat_mixed_queries），因此结论可与 e2e 基线对齐。

两个测量设计要点（都是为了拿到可判读的信号）：

1. **用 hit@k 而不是"返回集合召回"**。本图的候选池已覆盖全图 ~85%，`top_k=100` 与
   `rerank_pool=80` 都不触顶，返回集合几乎与排序无关 —— 用整体召回去衡量 contrast
   会永远看到"无差异"（实测 56 节点图返回 43~50 条）。hit@1/3/5/10 只看排序前几
   位，才是排序敏感的口径，也正是平台 Answer 侧真实消费顺序的方式。

2. **四个格子共享同一次检索**（记忆化 retriever.retrieve）。search 内部的目的推断是
   LLM 调用，同一查询重复跑会命中不同的种子与扩展路径，实测抖动带宽 ±1.7~3 名，
   足以吞掉 contrast 的效应。配对之后同一重复轮次内四格看到完全相同的候选集，
   差异只来自 guard / contrast 两个开关。

用法：python -u -m eval.probe_options_mc [--limit 13] [--repeat 2]
"""

import argparse
import os
import random
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_p_baseline import _load_dotenv

DATA_DIR = ROOT / "data" / "dba_eval"
USER_ID = "eval:local:groupchat:conv-0"   # 与 e2e 一致
SESSION_ID = "eval:local:groupchat:sample:0"

# 泛化题干：不含任何主题词，判别信息全在选项里（对 retrieval 最不利的形态）
GENERIC_STEM = "以下哪一项最符合群里的情况？"

# 干扰项数量（与 PersonaMem 的 4 选 1 对齐）
N_DISTRACTORS = 3

# 排序敏感口径：只看返回序列前 K 条
TOPK_LIST = (1, 3, 5, 10)

# 2×2 配置：(标签, options_abstain_guard, options_contrast)
CELLS: Tuple[Tuple[str, bool, bool], ...] = (
    ("off", False, False),
    ("guard", True, False),
    ("contrast", False, True),
    ("both", True, True),
)

_PUNCT_RE = re.compile(r"[\s，。、：:；;！!？?…—\-（）()「」『』\"'’‘“”·,.\[\]]")


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text or "").lower()


def match_facts(text: str, facts: List[str]) -> List[str]:
    norm = _norm(text)
    return [fact for fact in facts if _norm(fact) in norm]


PARAPHRASE_PROMPT = """群里有人提到过「{fact}」这件事。请把它写成一句自然的、群里闲聊语气的中文句子。

要求：
- 事实与主体必须与原文完全一致，不得增删信息；
- 换用不同的措辞与句式，但意思不变；
- 只输出这一句本身，不要引号、编号、解释，也不要说"无法改写"。

句子："""

# 改写失败/跑偏时的兜底模板（保证选项文本长度与语域一致）
_FALLBACK_TEMPLATE = "群里有人提到过{fact}这件事。"

# 出现这些词说明模型在"讨论任务"而不是"改写内容"，直接退回模板
_META_MARKERS = ("无法", "改写", "陈述句", "原文", "要求", "抱歉", "作为AI")


def paraphrase(llm, fact: str) -> str:
    """把事实改写成一句自然陈述，作为选项文本。

    直接用原文当选项会让选项与记忆节点逐字相同，对比信号被高估到不真实的程度
    （余弦接近 1）；改写成句后才是接近真实数据集的形态。单字/双字事实（如「跑步」）
    容易被模型当成"没法改写"而返回元话语，故加一道 sanity 过滤。
    """
    try:
        response = llm.invoke(PARAPHRASE_PROMPT.format(fact=fact))
        text = (getattr(response, "content", "") or "").strip().strip('"').strip("'")
        text = text.splitlines()[0].strip() if text else ""
    except Exception as exc:
        print(f"    [WARN] 改写失败，退回模板: {exc}", file=sys.stderr)
        text = ""
    if not text or any(marker in text for marker in _META_MARKERS):
        return _FALLBACK_TEMPLATE.format(fact=fact)
    return text


def load_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(DATA_DIR / "groupchat_mixed_dialogs.yaml", encoding="utf-8") as f:
        dialogs = yaml.safe_load(f)["dialogs"]
    with open(DATA_DIR / "groupchat_mixed_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)
    return dialogs, queries


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


def install_retrieval_memo(space) -> Tuple[Dict[Any, Any], Dict[str, int]]:
    """把检索前端（目的推断 + PAR）按 (查询, 参数) 记忆化，使 4 个格子变成配对比较。

    search 内部的目的推断是 LLM 调用，同一查询重复跑会走出不同的种子与扩展路径；
    实测同一配置重复跑的排名抖动达 ±1.7~3 名，足以吞掉待测效应。记忆化后同一重复
    轮次内四个格子看到**完全相同**的候选集，差异只剩 guard / contrast 两个开关。

    返回 (memo, stats)：stats 累计「检索调用次数」与「真实执行次数」，比值即复用倍数，
    用于自查配对是否真的生效（不依赖"题数 == 记忆化条目数"这种会被相似题干误导的口径）。
    只在探针内使用，生产链路不装这个补丁。
    """
    retriever = space.retriever
    memo: Dict[Any, Any] = {}
    stats = {"calls": 0, "real": 0}
    originals: Dict[str, Callable] = {
        "raw": retriever.retrieve,
        "story": retriever.retrieve_with_story,
    }

    def make_cached(name: str, fn: Callable) -> Callable:
        def cached(query, **kwargs):
            stats["calls"] += 1
            key = (name, query, tuple(sorted(kwargs.items())))
            if key not in memo:
                stats["real"] += 1
                memo[key] = fn(query, **kwargs)
            return memo[key]
        return cached

    for name, fn in originals.items():
        setattr(retriever, "retrieve" if name == "raw" else "retrieve_with_story",
                make_cached(name, fn))
    return memo, stats


def build_items(engine, queries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    judged = [q for q in queries if q.get("gold_facts")]
    facts = list(dict.fromkeys(fact for q in judged for fact in q["gold_facts"]))
    print(f"  改写 {len(facts)} 条 gold_fact 作为选项池（约 {len(facts)} 次 LLM）")
    pool = {fact: paraphrase(engine.llm, fact) for fact in facts}

    rng = random.Random(20260919)
    items: List[Dict[str, Any]] = []
    for query in judged:
        own = query["gold_facts"]
        candidates = [f for f in facts if f not in own]
        if len(candidates) < N_DISTRACTORS:
            continue
        options = [pool[own[0]]] + [pool[f] for f in rng.sample(candidates, N_DISTRACTORS)]
        rng.shuffle(options)
        items.append({
            "id": query["id"],
            "category": query.get("category", ""),
            "natural": query["query"],
            "generic": GENERIC_STEM,
            # 命中判定用 gold_fact 原文（与选项措辞无关，故改写不影响判定口径）
            "gold_facts": own,
            "correct_option": pool[own[0]],
            "options": [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)],
        })
        print(f"  [{items[-1]['id']}] {items[-1]['category']:<10} "
              f"正确选项={items[-1]['correct_option']!r}  gold={own}")
        if len(items) >= limit:
            break
    return items


def facts_in_topk(result: List[Dict[str, Any]], facts: List[str], k: int) -> Set[str]:
    """返回序列前 k 条里出现的 gold_fact 集合（排序敏感口径）"""
    found: Set[str] = set()
    for entry in result[:k]:
        found |= set(match_facts(entry.get("content") or "", facts))
    return found


def run_cell(engine, cfg, items: List[Dict[str, Any]], stem_key: str,
             guard: bool, contrast: bool) -> List[Dict[str, Any]]:
    from alm.contract import SearchRequest

    cfg.options_abstain_guard = guard
    cfg.options_contrast = contrast
    rows = []
    for item in items:
        result = engine.search(SearchRequest(
            query=item[stem_key],
            user_id=USER_ID,
            top_k=100,
            options=item["options"],
        ))
        best_cos = engine.retrieval_stats.last_cosine()
        joined = " ".join(it.get("content") or "" for it in result)
        rows.append({
            "id": item["id"],
            "returned": len(result),
            "matched": len(match_facts(joined, item["gold_facts"])),
            "facts_total": len(item["gold_facts"]),
            "topk": {k: len(facts_in_topk(result, item["gold_facts"], k)) for k in TOPK_LIST},
            "best_cos": best_cos if best_cos is not None else 0.0,
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "hit_items": sum(1 for r in rows if r["matched"] > 0),
        "total": len(rows),
        "facts_hit": sum(r["matched"] for r in rows),
        "facts_total": sum(r["facts_total"] for r in rows),
        "zeroed": sum(1 for r in rows if r["returned"] == 0),
        "returned": [r["returned"] for r in rows],
        "topk": {k: sum(r["topk"][k] for r in rows) for k in TOPK_LIST},
        "min_cos": round(min(r["best_cos"] for r in rows), 4) if rows else 0.0,
    }


def _fmt(values: List[Any], total: Optional[int] = None) -> str:
    """中位数[min~max]；给出 total 时附分母（如 34/34[32~34]）"""
    if not values:
        return "-"
    median = statistics.median(values)
    spread = f"[{min(values):g}~{max(values):g}]"
    return f"{median:g}/{total}{spread}" if total is not None else f"{median:g}{spread}"


def main() -> int:
    parser = argparse.ArgumentParser(description="选择题 options A/B 探针（e2e 图）")
    parser.add_argument("--limit", type=int, default=13, help="选择题数量上限")
    parser.add_argument("--repeat", type=int, default=2,
                        help="重复轮次；每轮内四格配对，轮间重新检索")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        return 2

    dialogs, queries = load_data()

    from alm.config import ALMConfig
    from alm.engine import ALMEngine

    cfg = ALMConfig.from_env()
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="options_probe_"))
    cfg.search_shape = "raw"        # 不生成叙述：省一次 LLM，且只看原子节点排序
    cfg.abstain_verify_hi = 0.0     # 只看硬下限，隔离掉模糊带判定的 LLM
    env_flags = ALMConfig.from_env()
    engine = ALMEngine(cfg)
    print(f"# 数据目录: {cfg.data_dir}")
    print(f"# 硬下限 ALM_ABSTAIN_COS={cfg.abstain_cosine}；rerank={cfg.rerank_mode}；"
          f"对比权重={cfg.options_contrast_weight}")
    print(f"# 探针集上限 {args.limit} 条；.env 开关 guard={env_flags.options_abstain_guard} "
          f"contrast={env_flags.options_contrast}（仅影响服务启动，探针按 2×2 覆写）")

    print("\n===== ① 写入语料（与 e2e 同源）=====")
    ingest(engine, dialogs)

    print("\n===== ② 构造选择题（正确选项 = gold_fact 改写，干扰项 = 其它 fact 改写）=====")
    items = build_items(engine, queries, args.limit)
    if not items:
        print("错误: 没有可用的选择题", file=sys.stderr)
        return 1

    space = engine.get_space(USER_ID, create=False)
    node_count = space.graph.node_count if space is not None else 0
    facts_total = sum(len(i["gold_facts"]) for i in items)
    print(f"  图节点数={node_count}  gold_fact 合计={facts_total} 项"
          f"  rerank_pool={cfg.rerank_pool}  max_top_k={cfg.max_top_k}")

    # 配对比较：四格共享同一次检索
    memo, memo_stats = install_retrieval_memo(space)
    print("  已启用检索记忆化：同一轮内四个格子复用同一份候选集（配对比较）")

    print(f"\n===== ③ 2×2 对比（{args.repeat} 轮，取中位数[min~max]）=====")
    print("  口径说明：命中事实 = 全量返回里出现的 gold_fact；"
          "h@k = 返回序列前 k 条里出现的 gold_fact（排序敏感）")
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for stem_key, stem_label in (("natural", "原始题干"), ("generic", "泛化题干")):
        results[stem_key] = {}
        print(f"\n--- 题干形态: {stem_label}（{stem_key}）---")
        print(f"{'配置':<10}{'命中题':>13}{'命中事实':>13}"
              f"{'h@1':>10}{'h@3':>10}{'h@5':>10}{'h@10':>10}"
              f"{'归零':>9}{'返回条数':>13}")
        # 外层轮次、内层格子：同一轮内四个格子共享同一次检索（配对比较）；
        # 每轮开始时清空记忆化，让轮与轮之间保留真实的检索抖动。
        per_label: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _, _ in CELLS}
        for _ in range(args.repeat):
            memo.clear()
            for name, guard, contrast in CELLS:
                per_label[name].append(
                    summarize(run_cell(engine, cfg, items, stem_key, guard, contrast))
                )
        for label, _, _ in CELLS:
            runs = per_label[label]
            results[stem_key][label] = {
                "runs": runs,
                "hits": [r["hit_items"] for r in runs],
                "facts": [r["facts_hit"] for r in runs],
                "zeroed": [r["zeroed"] for r in runs],
                "returned": [n for r in runs for n in r["returned"]],
                "topk": {k: [r["topk"][k] for r in runs] for k in TOPK_LIST},
            }
            cell = results[stem_key][label]
            topk_cells = "".join(f"{_fmt(cell['topk'][k], facts_total):>10}" for k in TOPK_LIST)
            print(f"{label:<10}{_fmt(cell['hits'], len(items)):>13}"
                  f"{_fmt(cell['facts'], facts_total):>13}{topk_cells}"
                  f"{_fmt(cell['zeroed']):>9}{_fmt(cell['returned'], node_count):>13}")

    # 配对自查：复用倍数应等于格子数。不按"记忆化条目数"判断——泛化题干下所有题共用一个
    # 题干，条目数本就只有 1，那个口径会误报。
    calls, real = memo_stats["calls"], memo_stats["real"]
    ratio = calls / real if real else 0.0
    verdict = "配对生效" if ratio >= len(CELLS) * 0.9 else "配对未生效，下面的增量不可信"
    print(f"\n  配对自查: 检索调用 {calls} 次，真实执行 {real} 次 → 复用 {ratio:.1f}×"
          f"（格子数={len(CELLS)}）{verdict}")
    print("  旁证: guard 是空操作（零弃权），其增量应恰为 0；both 应逐字等于 contrast。")

    print("\n===== ④ 配对增量（相对 off，同轮次相减，中位数）=====")
    print("  guard 只可能改善「归零」；contrast 只可能改善 h@k —— 两者互不重叠")
    for stem_key, stem_label in (("natural", "原始题干"), ("generic", "泛化题干")):
        base = results[stem_key]["off"]
        print(f"  [{stem_label}]")
        for label in ("guard", "contrast", "both"):
            cell = results[stem_key][label]
            d_zero = statistics.median(
                [c - b for c, b in zip(cell["zeroed"], base["zeroed"])]
            )
            deltas = {
                k: statistics.median(
                    [c - b for c, b in zip(cell["topk"][k], base["topk"][k])]
                ) for k in TOPK_LIST
            }
            delta_txt = "  ".join(f"h@{k}:{deltas[k]:+.1f}" for k in TOPK_LIST)
            print(f"    {label:<9} 归零 Δ={d_zero:+.1f}   {delta_txt}")

    print("\n===== ⑤ best_cos 分布（off 档，决定 guard 是否有触发机会）=====")
    print(f"  硬下限 ALM_ABSTAIN_COS={cfg.abstain_cosine}；低于它才会整题归零")
    memo.clear()
    for stem_key, stem_label in (("natural", "原始题干"), ("generic", "泛化题干")):
        rows = run_cell(engine, cfg, items, stem_key, False, False)
        cosines = sorted(r["best_cos"] for r in rows)
        below = [r["id"] for r in rows if r["best_cos"] < cfg.abstain_cosine]
        print(f"  [{stem_label}] min={cosines[0]:.4f} 中位={cosines[len(cosines) // 2]:.4f} "
              f"max={cosines[-1]:.4f}  低于硬下限={below or '无'}")

    print("\n===== ⑥ 结论口径 =====")
    print("  · guard 的有效性看「归零 Δ」：本图若一直在硬下限之上，它就没有生效机会；")
    print("  · contrast 的有效性看「h@k Δ」：召回饱和（返回条数≈图节点数）时整体召回无差异，")
    print("    只有 h@k 能反映排序变化；")
    print("  · 若 h@k Δ 的中位数与 [min~max] 跨零，说明变化仍在检索抖动范围内，不能定论；")
    print("  · 本探针只覆盖 answer 型查询，不测 H（有选项但库里没有的题需另测）。")

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
