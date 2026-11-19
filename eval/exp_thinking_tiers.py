# -*- coding: utf-8 -*-
"""三档 thinking × 两阶段对照实验（参照 Plan/0908 e2e 论文文档的实验设计）

问题
----
同一份原始对话、同一条代码路径，只改 deepseek 的 thinking 档位（off / low / high），
对「抽取建图」与「完整检索」各有什么影响。重点是**成本与质量的取舍**，而不是"哪档分高"：
本系统的检索决策由 LLM 承担，非确定性使单轮分数不能逐条对比，只作方向判断。

两阶段（产物分离，第二阶段不重跑抽取）
------------------------------------
  build  : 三臂各自对同一份原始对话做 Add（抽取 → 建图 → 落盘），每臂落到**独立且保留**
           的 data_dir；产出 build.json / graph.yaml
  search : 三臂各自**从已存在的 data_dir 加载**（engine.get_space(create=False) 命中已落盘的
           graph.yaml + 索引，全程不触发抽取），跑全量查询；产出 search.json / blocks.json

  目录（每臂一套，互不覆盖，可复用 / 可回溯 / 可只换检索参数重跑阶段二）
    eval/outputs/thinking_tiers/<corpus>/<arm>/
        data/        ALM 空间本体（<user_id>.yaml + <user_id>.index/）
        build.json   阶段一指标
        search.json  阶段二逐查询结果
        blocks.json  上下文块（同题并排判分所需形态，可直接喂现有判分脚本）
        graph.yaml   便于人工检视的副本

为什么显式收窄检索预算（本实验的关键设计）
---------------------------------------
默认 seed_k=40 / expand_k=40 / max_hops=5 在几十节点的单 user 图上，两跳内就会覆盖整张图
—— 图扩展退化成"把全图概括一遍"，三臂差异被抹平、对照失去意义。故把预算收窄到与图规模
同量级（默认 expand_k=3 / max_hops=2），并**逐查询输出「读取集 / 图规模」覆盖比**，
作为"没有饱和成概括整张图"的证据。

纯向量基线口径
------------
基线宽度取 ``space._effective_seed_k()``（与图路径**同种子数**）：两边种子完全相同，唯一
差别是"要不要走图扩展"，此时 ``★仅图命中`` 才真正度量 PAR 的增量。
（历史教训：早期探针硬写 top-100，在几十节点的图上等于整张图，该指标恒为 0。）

用法
----
    python -m eval.exp_thinking_tiers --stage build                 # 三臂建图（可复用）
    python -m eval.exp_thinking_tiers --stage search                # 三臂检索
    python -m eval.exp_thinking_tiers --stage all
    python -m eval.exp_thinking_tiers --stage build --force         # 丢弃已有产物重建
    python -m eval.exp_thinking_tiers --corpus comprehensive --stage build
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:  # pragma: no cover - 环境里无 dotenv 时按已注入的环境变量走
    pass

import yaml  # noqa: E402

from alm.config import ALMConfig  # noqa: E402
from alm.contract import AddMessage, AddRequest  # noqa: E402
from alm.engine import ALMEngine, space_filename  # noqa: E402
from eval.probe_llm_arm_ab import (  # noqa: E402
    THINKING_MODES,
    _Capture,
    _apply_thinking_mode,
    _take_within,
    _vector_topk,
)
from eval.test_alm_vs_mcp_e2e import graph_stats, match_facts, print_stats  # noqa: E402

DATA_DIR = _ROOT / "data" / "dba_eval"
OUT_ROOT = _ROOT / "eval" / "outputs" / "thinking_tiers"

USER_ID = "eval:exp:tiers:conv-0"
SESSION_ID = "eval:exp:tiers:s0"
BASE_MS = 1_704_067_200_000  # 2024-01-01（对话之间按天递增）

DEFAULT_BUDGET = {"expand_k": 3, "max_hops": 2}


# ---------------------------------------------------------------- 语料

def _load_corpus(name: str) -> Tuple[List[dict], List[dict]]:
    dialogs_path = DATA_DIR / f"{name}_dialogs.yaml"
    queries_path = DATA_DIR / f"{name}_queries.yaml"
    for path in (dialogs_path, queries_path):
        if not path.exists():
            raise SystemExit(f"语料缺失: {path}")
    with dialogs_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    dialogs = data["dialogs"] if isinstance(data, dict) else data
    with queries_path.open("r", encoding="utf-8") as fh:
        queries = yaml.safe_load(fh)
    return dialogs, queries


def _clock_offset_ms(value: Any) -> Optional[int]:
    """把语料里的钟点还原成当日毫秒偏移，认不出就返回 None。

    PyYAML 会把 ``time: 09:00`` 解析成**六十进制整数 540**（9 时 0 分 = 540 分钟），
    故两种形态都要处理，否则这里的取值会静默错位。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value * 60_000                      # 已按分钟计
    text = str(value)
    if ":" not in text:
        return None
    head, _, tail = text.partition(":")
    try:
        return (int(head) * 60 + int(tail[:2])) * 60_000
    except ValueError:
        return None


def _messages_for(dialog: dict, idx: int) -> List[AddMessage]:
    """对话 → Add 消息。两种回合形态都支持：

    · groupchat_mixed：``{speaker, role, text}`` —— 昵称必须留在正文，归属信息在正文里
    · comprehensive  ：``{role, time, text}`` —— 双主体、无昵称，正文即原文

    时间戳优先用语料自带的钟点（有则还原真实时刻），否则退化为「轮次间隔 5 分钟」；
    对话之间固定按天递增，时序锚点依赖这一点。
    """
    messages: List[AddMessage] = []
    for i, turn in enumerate(dialog["turns"]):
        speaker = turn.get("speaker")
        content = f"[{speaker}] {turn['text']}" if speaker else str(turn["text"])
        offset = _clock_offset_ms(turn.get("time"))
        if offset is None:
            offset = i * 300_000
        messages.append(
            AddMessage(role=turn["role"], content=content,
                       timestamp=BASE_MS + idx * 86_400_000 + offset)
        )
    return messages


# ---------------------------------------------------------------- 阶段一：建图

def _base_config(expand_k: int, max_hops: int) -> ALMConfig:
    cfg = ALMConfig.from_env()
    cfg.embedding_local = False          # 与云端部署一致，避免本地加载向量模型
    cfg.expand_k = expand_k
    cfg.max_hops = max_hops
    return cfg


def _arm_dir(corpus: str, arm: str) -> Path:
    return _OUT_ROOT / corpus / arm


_OUT_ROOT = OUT_ROOT      # 可被 --out 覆盖（阶段一阶段二共用同一根）


def _select(items: List[dict], ids: str, limit: Optional[int]) -> List[dict]:
    """按 id 白名单与条数上限裁剪，用于分步/冒烟（不改语料文件）"""
    if ids and ids != "all":
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        items = [x for x in items if x.get("id") in wanted]
    return items[:limit] if limit else items


def stage_build(corpus: str, arms: List[str], dialogs: List[dict],
                expand_k: int, max_hops: int, force: bool,
                patch_thinking: bool = True) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        out_dir = _arm_dir(corpus, arm)
        data_dir = out_dir / "data"
        yaml_path = data_dir / space_filename(USER_ID)
        if yaml_path.exists() and not force:
            print(f"\n===== [build] {arm}：已存在 {yaml_path}，跳过（--force 可重建）=====")
            build_meta = out_dir / "build.json"
            results[arm] = (json.loads(build_meta.read_text(encoding="utf-8"))
                            if build_meta.exists() else {"reused": True})
            continue

        if force and data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        cfg = _base_config(expand_k, max_hops)
        cfg.data_dir = data_dir
        if patch_thinking:
            _apply_thinking_mode(arm)

        cap = _Capture()
        root_logger = logging.getLogger()
        root_logger.addHandler(cap)
        engine = ALMEngine(cfg)
        print(f"\n===== [build] 臂 {arm}：{cfg.llm_model} @ {cfg.llm_base_url} "
              f"| expand_k={expand_k} max_hops={max_hops} =====")
        print(f"      data_dir={data_dir}")
        try:
            per_add: List[Dict[str, Any]] = []
            started = time.time()
            for idx, dialog in enumerate(dialogs):
                result = engine.add(AddRequest(
                    request_id=f"{USER_ID}:{dialog['id']}",
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    messages=_messages_for(dialog, idx),
                ))
                per_add.append({"id": dialog["id"], **result})
            elapsed = time.time() - started
            print(f"      写入耗时 {elapsed:.1f}s  |  每段落地: " + ", ".join(
                f"{r['id']}={r['messages']}msg/{r['nodes']}node" for r in per_add))

            space = engine.get_space(USER_ID, create=False)
            stats = graph_stats(space.graph, arm)
            print_stats(stats)
            token_summary = engine.token_meter.summary()
            print(f"      token: calls={token_summary['calls']} "
                  f"in={token_summary['input_tokens']:,} out={token_summary['output_tokens']:,}")
            degrade = cap.degrade_hits()
            print(f"      日志: WARNING={cap.count_level(logging.WARNING)} "
                  f"ERROR={cap.count_level(logging.ERROR)} | 退化信号={degrade}")
        finally:
            engine.close()                       # 只落盘有改动的空间（见 _dirty）
            root_logger.removeHandler(cap)

        results[arm] = {
            "arm": arm, "corpus": corpus, "model": cfg.llm_model,
            "base_url": cfg.llm_base_url,
            "expand_k": expand_k, "max_hops": max_hops,
            "dialogs": len(dialogs), "add_elapsed_s": round(elapsed, 1),
            "per_add": per_add, "graph": stats, "tokens": token_summary,
            "warnings": cap.count_level(logging.WARNING),
            "errors": cap.count_level(logging.ERROR),
            "degrade": degrade,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "build.json").write_text(
            json.dumps(results[arm], ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(yaml_path, out_dir / "graph.yaml")
        print(f"      产物: {out_dir}（data/ 保留，阶段二直接复用）")
    _apply_thinking_mode(None)
    return results


# ---------------------------------------------------------------- 阶段二：检索

def _capture_read_set(space) -> Dict[str, Any]:
    """包一层 retriever，零额外成本地拿到「读取集」大小。

    为什么需要它：判断有没有退化成"概括整张图"，唯一硬证据是**检索器实际纳入考虑的
    节点数 / 图规模**。``_build_result`` 的 peak_scores 键集正是各跳候选的并集，
    即读取集；而它只在 retriever 内部可见，故在此包一层截获，不额外发 LLM 调用。
    """
    box: Dict[str, Any] = {"read_set": None, "adopted": None}
    original = space.retriever.retrieve_with_story

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            box["read_set"] = len(result.get("peak_scores") or {})
            box["adopted"] = len(result.get("peak_memories") or [])
        except Exception:      # 结构变化时不影响主流程
            box["read_set"] = box["adopted"] = None
        return result

    space.retriever.retrieve_with_story = wrapper
    return box


def stage_search(corpus: str, arms: List[str], queries: List[dict],
                 expand_k: int, max_hops: int,
                 patch_thinking: bool = True) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    blocks: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        out_dir = _arm_dir(corpus, arm)
        data_dir = out_dir / "data"
        yaml_path = data_dir / space_filename(USER_ID)
        if not yaml_path.exists():
            print(f"\n===== [search] {arm}：缺少 {yaml_path}，请先跑 --stage build =====",
                  file=sys.stderr)
            results[arm] = {"error": "missing_build"}
            continue

        cfg = _base_config(expand_k, max_hops)
        cfg.data_dir = data_dir
        if patch_thinking:
            _apply_thinking_mode(arm)

        cap = _Capture()
        root_logger = logging.getLogger()
        root_logger.addHandler(cap)
        engine = ALMEngine(cfg)
        space = engine.get_space(USER_ID, create=False)
        node_count = space.graph.node_count
        vec_k = space._effective_seed_k()
        print(f"\n===== [search] 臂 {arm}：图 {node_count} 节点 "
              f"| 种子预算 seed_k={vec_k} | 基线宽度 top-{vec_k} =====")
        box = _capture_read_set(space)
        try:
            rows: List[Dict[str, Any]] = []
            for query in queries:
                shot = time.time()
                data = space.search(query["query"], top_k=100)
                elapsed = time.time() - shot
                text = "\n".join(item["content"] for item in data)
                gold = query.get("gold_facts") or []
                graph_hit = match_facts(text, gold)
                _, vec_texts = _vector_topk(space, query["query"], vec_k)
                vec_hit = match_facts("\n".join(vec_texts), gold)
                vec_budget_hit = match_facts(
                    _take_within(vec_texts, max(len(text), 1)), gold)
                rows.append({
                    "id": query["id"], "query": query["query"],
                    "count": len(data), "chars": len(text), "content": text,
                    "gold_total": len(gold), "gold_hit": graph_hit,
                    "vec_hit": vec_hit, "vec_budget_hit": vec_budget_hit,
                    "only_graph": sorted(set(graph_hit) - set(vec_hit)),
                    "read_set": box["read_set"], "adopted": box["adopted"],
                    "read_ratio": (round(box["read_set"] / node_count, 3)
                                   if box["read_set"] and node_count else None),
                    "elapsed": elapsed,
                })
            token_summary = engine.token_meter.summary()
        finally:
            engine.close()
            root_logger.removeHandler(cap)

        gold_hit = sum(len(r["gold_hit"]) for r in rows)
        gold_total = sum(r["gold_total"] for r in rows)
        vec_hit_total = sum(len(r["vec_hit"]) for r in rows)
        only_graph = sum(len(r["only_graph"]) for r in rows)
        ratios = [r["read_ratio"] for r in rows if r["read_ratio"] is not None]
        print(f"      检索 {len(rows)} 条 | gold={gold_hit}/{gold_total} "
              f"| 纯向量(top-{vec_k})={vec_hit_total} | ★仅图命中={only_graph}")
        if ratios:
            print(f"      读取集/图规模: 中位 {sorted(ratios)[len(ratios) // 2]:.2f} "
                  f"（max {max(ratios):.2f}）→ {'未饱和' if max(ratios) < 0.6 else '已饱和'}")
        print(f"      token: calls={token_summary['calls']} "
              f"in={token_summary['input_tokens']:,} out={token_summary['output_tokens']:,}")

        results[arm] = {
            "arm": arm, "corpus": corpus, "node_count": node_count, "vec_k": vec_k,
            "queries": len(rows), "gold_hit": gold_hit, "gold_total": gold_total,
            "vec_hit": vec_hit_total, "only_graph": only_graph,
            "read_ratio_max": max(ratios) if ratios else None,
            "tokens": token_summary, "rows": rows,
        }
        blocks[arm] = {r["id"]: {"query": r["query"], "content": r["content"],
                                 "gold_facts": next((q.get("gold_facts") or [] for q in queries
                                                     if q["id"] == r["id"]), [])}
                       for r in rows}
        (out_dir / "search.json").write_text(
            json.dumps(results[arm], ensure_ascii=False, indent=2), encoding="utf-8")
    _apply_thinking_mode(None)

    if blocks:
        first_arm = arms[0] if arms[0] in blocks else next(iter(blocks))
        combined: Dict[str, Any] = {}
        for qid, entry in blocks[first_arm].items():
            combined[qid] = {
                "query": entry["query"],
                "gold_facts": entry["gold_facts"],
                "arms": {arm: blocks[arm][qid]["content"]
                         for arm in blocks if qid in blocks[arm]},
            }
        path = _OUT_ROOT / corpus / "blocks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  同题并排上下文块: {path}（{len(combined)} 条查询 × {len(blocks)} 臂）")
    return results


# ---------------------------------------------------------------- 汇总

def _report(builds: Dict[str, Any], searches: Dict[str, Any], arms: List[str]) -> None:
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    if builds:
        print("\n[阶段一] 建图")
        print(f"  {'臂':<6}{'写入(s)':>9}{'节点':>6}{'边':>7}{'TEMPORAL':>10}"
              f"{'锚点':>6}{'「用户」':>8}{'out tok':>11}")
        for arm in arms:
            b = builds.get(arm) or {}
            g = b.get("graph") or {}
            t = b.get("tokens") or {}
            print(f"  {arm:<6}{b.get('add_elapsed_s', 0):>9.1f}{g.get('nodes', 0):>6}"
                  f"{g.get('edges', 0):>7}{g.get('temporal_edges', 0):>10}"
                  f"{g.get('time_anchor_nodes', 0):>6}{g.get('collapse_user', 0):>8}"
                  f"{t.get('output_tokens', 0):>11,}")
    if searches:
        print("\n[阶段二] 检索")
        print(f"  {'臂':<6}{'gold 命中':>12}{'纯向量':>8}{'★仅图':>8}"
              f"{'读取比max':>10}{'out tok':>11}")
        for arm in arms:
            s = searches.get(arm) or {}
            if s.get("error"):
                print(f"  {arm:<6}  {s['error']}")
                continue
            t = s.get("tokens") or {}
            ratio = s.get("read_ratio_max")
            print(f"  {arm:<6}{s.get('gold_hit', 0):>7}/{s.get('gold_total', 0):<4}"
                  f"{s.get('vec_hit', 0):>8}{s.get('only_graph', 0):>8}"
                  f"{(f'{ratio:.2f}' if ratio is not None else '-'):>10}"
                  f"{t.get('output_tokens', 0):>11,}")
    print("\n判读")
    print("  · ★仅图命中 = 图扩展命中、而**同种子数**的纯向量基线拿不到的事实数，是 PAR 的净增量。")
    print("  · 读取比 = 检索器实际纳入考虑的节点 / 图规模；接近 1 说明退化成「概括整张图」，")
    print("    此时三臂差异不可信，需继续收窄 expand_k / max_hops。")
    print("  · LLM 非确定性：单轮只作方向判断；成本类指标（token）是结构性的，可直接比。")


def main() -> int:
    parser = argparse.ArgumentParser(description="三档 thinking × 两阶段对照实验")
    parser.add_argument("--corpus", default="groupchat_mixed",
                        help="data/dba_eval 下的语料名前缀（默认 groupchat_mixed）")
    parser.add_argument("--stage", choices=("build", "search", "all"), default="all")
    parser.add_argument("--arms", default="off,low,high",
                        help=f"逗号分隔的档位，可选 {THINKING_MODES}")
    parser.add_argument("--expand-k", type=int, default=DEFAULT_BUDGET["expand_k"],
                        help="每跳扩展数（默认 3，收窄以避免概括整张图）")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_BUDGET["max_hops"],
                        help="最大跳数（默认 2，同上）")
    parser.add_argument("--force", action="store_true", help="阶段一：丢弃已有产物重建")
    parser.add_argument("--dialogs", default="all", help="阶段一：仅处理这些对话 id（逗号分隔）")
    parser.add_argument("--dialog-limit", type=int, default=None, help="阶段一：仅处理前 N 段")
    parser.add_argument("--queries", default="all", help="阶段二：仅处理这些查询 id")
    parser.add_argument("--query-limit", type=int, default=None, help="阶段二：仅处理前 N 条")
    parser.add_argument("--out", default=None, help="产物根目录（默认 eval/outputs/thinking_tiers）")
    parser.add_argument("--no-thinking-patch", action="store_true",
                        help="不按臂名 patch thinking（改用配置默认），且臂名可任意命名。"
                             "用于「换模型跑同一套 harness」：如 gpt-4o-mini 不支持请求体 "
                             "thinking 参数，带上会 400（实测 api.gpt.ge）")
    args = parser.parse_args()

    global _OUT_ROOT
    if args.out:
        _OUT_ROOT = Path(args.out).resolve()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not args.no_thinking_patch:
        bad = [a for a in arms if a not in THINKING_MODES]
        if bad:
            print(f"错误：未知档位 {bad}，可选 {THINKING_MODES}", file=sys.stderr)
            return 2

    dialogs, queries = _load_corpus(args.corpus)
    dialogs = _select(dialogs, args.dialogs, args.dialog_limit)
    queries = _select(queries, args.queries, args.query_limit)
    print(f"# 语料 {args.corpus}：{len(dialogs)} 段对话 / {len(queries)} 条查询 "
          f"| gold fact {sum(len(q.get('gold_facts') or []) for q in queries)} 项")
    print(f"# 档位 {arms} | 检索预算 expand_k={args.expand_k} max_hops={args.max_hops} "
          f"| 阶段 {args.stage}")
    print(f"# 产物根目录 {_OUT_ROOT / args.corpus}")

    builds = stage_build(args.corpus, arms, dialogs, args.expand_k, args.max_hops,
                         args.force,
                         patch_thinking=not args.no_thinking_patch)\
        if args.stage in ("build", "all") else {}
    searches = (stage_search(args.corpus, arms, queries, args.expand_k, args.max_hops,
                             patch_thinking=not args.no_thinking_patch)
                if args.stage in ("search", "all") else {})
    _report(builds, searches, arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
