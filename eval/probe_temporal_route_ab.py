# -*- coding: utf-8 -*-
"""`temporal_route`（时序路由）的 ON/OFF A/B —— 临时探针（Plan §3.5 证据）。

背景：/search 的时序路由已实现（Plan §3.2/§3.3），但默认关闭，缺净收益证据。
本探针在**同一份记忆**上只切换 `temporal_route`，比对 gold_facts 命中数与叙事长度，
并把路由日志（判定 / match_type / 锚点 id / 追加条数）打出来。**只在开臂记录路由日志。**

⚠️ 测量设计（判读的前提）：
  叙事生成与目的推断都是 LLM 非确定性，**同一臂两次运行本身就会有差异**。
  故每臂各跑 2 遍：先用「臂内两遍之差」量出自波动量级，再看「臂间差」是否明显大于它。
  若无法区分，结论必须写「信号落在噪声内，无法判定」。

合规：本脚本只打印数值与节点 ID，**不打印 query 正文、记忆正文、gold fact 文本**。

成本控制：**只写入一次**；两臂共用同一 data_dir 做只读检索。

用法：python -u -m eval.probe_temporal_route_ab [--repeats 2]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # 与其它 eval 脚本一致：先从 .env 装载模型与路径配置
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from alm.config import ALMConfig
from alm.contract import AddMessage, AddRequest, SearchRequest
from alm.engine import ALMEngine
from eval.test_alm_vs_mcp_e2e import load_dataset, match_facts

USER_ID = "eval:probe:troute:conv-0"
SESSION_ID = "eval:probe:troute:s0"
BASE_MS = 1_704_067_200_000  # 2024-01-01

# 时序组判据（任务口径）：题干含明确时间词 / 问「某时间点发生了什么」。
#   q2  上周三的需求评审会定了什么？        → 显式周次
#   q8  这周六大家都有什么安排？            → 显式周次
#   q11 A 方案的接口原本什么时候提测？后来为什么推迟？ → 时间疑问
#   q13 A 方案的上线时间定在哪天？…        → 时间疑问
# 其余归非时序组（q1 的「最近」属模糊词，不计入「明确时间词」，归非时序）。
TEMPORAL_IDS = {"q2", "q8", "q11", "q13"}


class _RouteCapture(logging.Handler):
    """抓取 alm.space 的路由/主通道日志（只含数值与 ID，合规）"""

    _KEYS = ("时序路由", "时序查询", "主通道已执行")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(k in msg for k in self._KEYS):
            self.lines.append(msg)

    def take(self) -> List[str]:
        lines, self.lines = self.lines, []
        return lines


def parse_route(lines: List[str]) -> Dict[str, Any]:
    """从日志行解析路由结果（只取数值与 ID）"""
    out: Dict[str, Any] = {
        "verdict": None, "purposes": None, "match_type": None,
        "anchors": None, "facts": None, "appended": None,
        "appended_total": None, "main_channel": False,
    }
    for ln in lines:
        if "主通道已执行" in ln:
            out["main_channel"] = True
        m = re.search(r"时序路由: 判定=(\S+) 目的数=(\d+)", ln)
        if m:
            out["verdict"], out["purposes"] = m.group(1), int(m.group(2))
            continue
        m = re.search(r"时序查询: match_type=(\S+) 锚点=\[([^\]]*)\] 事实数=(\d+) 可并入=(\d+)", ln)
        if m:
            anchors = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
            out["match_type"] = m.group(1)
            out["anchors"] = anchors
            out["facts"] = int(m.group(3))
            out["appended"] = int(m.group(4))
            continue
        m = re.search(r"时序路由追加: 目的 (\d+) → (\d+)（追加 (\d+) 条", ln)
        if m:
            out["appended_total"] = int(m.group(3))
    return out


def ingest(dialogs: List[Dict[str, Any]], data_dir: Path) -> float:
    cfg = ALMConfig.from_env()
    cfg.data_dir = data_dir
    engine = ALMEngine(cfg)
    started = time.time()
    for idx, dialog in enumerate(dialogs):
        engine.add(AddRequest(
            request_id=f"{USER_ID}:{dialog['id']}",
            user_id=USER_ID,
            session_id=SESSION_ID,
            messages=[
                AddMessage(
                    role=turn["role"],
                    content=f"[{turn['speaker']}] {turn['text']}",
                    timestamp=BASE_MS + idx * 86_400_000 + i * 300_000,
                )
                for i, turn in enumerate(dialog["turns"])
            ],
        ))
        print(f"  {dialog['id']} 写入完成（累计 {time.time() - started:.1f}s）")
    elapsed = time.time() - started
    engine.close()
    return elapsed


def run_pass(engine: ALMEngine, cap: _RouteCapture, queries: List[Dict[str, Any]],
             arm: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for item in queries:
        cap.take()  # 清空，只保留本次检索的日志
        started = time.time()
        items = engine.search(SearchRequest(query=item["query"], user_id=USER_ID, top_k=100))
        secs = time.time() - started
        lines = cap.take()

        text = " ".join(it.get("content") or "" for it in items)
        facts = item.get("gold_facts") or []
        hits = match_facts(text, facts)
        cos = engine.retrieval_stats.last_cosine()
        rows[item["id"]] = {
            "hits": len(hits), "total": len(facts), "chars": len(text),
            "n_items": len(items), "cos": float(cos) if cos is not None else 0.0,
            "secs": secs, "route": parse_route(lines),
        }
        print(f"    {arm} {item['id']:<4} 命中={len(hits)}/{len(facts)} "
              f"叙事长={len(text):>4} 条数={len(items)} cos={rows[item['id']]['cos']:.4f} "
              f"({secs:.1f}s)")
    return rows


def _median(values: List[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if not n:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description="temporal_route 的 ON/OFF A/B")
    parser.add_argument("--repeats", type=int, default=2, help="每臂重复遍数（默认 2）")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        return 2

    dialogs, queries = load_dataset()
    queries = [q for q in queries if q.get("gold_facts")]  # 跳过无解答题（gold 为空）
    temporal_qs = [q for q in queries if q["id"] in TEMPORAL_IDS]
    other_qs = [q for q in queries if q["id"] not in TEMPORAL_IDS]

    print("# temporal_route ON/OFF A/B（同一份记忆，只切换该开关）")
    print(f"# 时序组 {len(temporal_qs)} 条 {sorted(q['id'] for q in temporal_qs)}")
    print(f"# 非时序组 {len(other_qs)} 条 {sorted(q['id'] for q in other_qs)}")
    print(f"# 每臂重复 {args.repeats} 遍（量臂内自波动）\n")

    data_dir = Path(tempfile.mkdtemp(prefix="almtr_ab_"))
    print(f"# 数据目录: {data_dir}")

    print("\n===== ① 写入（仅一次，两臂共用）=====")
    elapsed = ingest(dialogs, data_dir)
    print(f"  写入耗时 {elapsed:.1f}s")

    # 两臂：同 data_dir，只改 temporal_route
    cfg_off = ALMConfig.from_env()
    cfg_off.data_dir = data_dir
    cfg_off.temporal_route = False
    engine_off = ALMEngine(cfg_off)

    cfg_on = ALMConfig.from_env()
    cfg_on.data_dir = data_dir
    cfg_on.temporal_route = True
    engine_on = ALMEngine(cfg_on)

    cap = _RouteCapture()
    logging.getLogger("alm.space").addHandler(cap)
    logging.getLogger("alm.space").setLevel(logging.INFO)

    print(f"\n  temporal_route: OFF -> {engine_off.config.temporal_route}"
          f" | ON -> {engine_on.config.temporal_route}")

    print("\n===== ② 检索（每臂各 N 遍）=====")
    results: Dict[str, List[Dict[str, Dict[str, Any]]]] = {"OFF": [], "ON": []}
    for arm, engine in (("OFF", engine_off), ("ON", engine_on)):
        for rep in range(args.repeats):
            print(f"\n  --- 臂 {arm} / 第 {rep + 1} 遍 ---")
            results[arm].append(run_pass(engine, cap, queries, arm))

    print("\n===== ③ 路由日志（仅开臂；两遍各记一次）=====")
    print("  节点 ID 与数值以外不记录（合规）")
    for rep in range(args.repeats):
        print(f"\n  --- ON 第 {rep + 1} 遍 ---")
        for q in temporal_qs + other_qs:
            r = results["ON"][rep][q["id"]]["route"]
            group = "时序" if q["id"] in TEMPORAL_IDS else "非时序"
            print(f"    {q['id']:<4}[{group:<3}] 判定={r['verdict']} 目的数={r['purposes']} "
                  f"match_type={r['match_type']} 锚点={r['anchors']} 事实数={r['facts']} "
                  f"追加={r['appended']}(实际{r['appended_total']}) 主通道={r['main_channel']}")

    print("\n===== ④ 逐查询命中数对照（gold_fact 命中数）=====")
    head = f"  {'id':<5}{'组别':<8}" + "".join(
        f"{'OFF' + str(i + 1):>9}{'ON' + str(i + 1):>9}" for i in range(args.repeats)
    ) + f"{'OFF最长':>9}{'ON最长':>9}"
    print(head)

    def _line(item: Dict[str, Any]) -> str:
        qid = item["id"]
        group = "时序" if qid in TEMPORAL_IDS else "非时序"
        cells = ""
        for rep in range(args.repeats):
            cells += f"{results['OFF'][rep][qid]['hits']}/{results['OFF'][rep][qid]['total']:>5}".rjust(9)
            cells += f"{results['ON'][rep][qid]['hits']}/{results['ON'][rep][qid]['total']:>5}".rjust(9)
        off_max = max(results["OFF"][rep][qid]["chars"] for rep in range(args.repeats))
        on_max = max(results["ON"][rep][qid]["chars"] for rep in range(args.repeats))
        return f"  {qid:<5}{group:<8}{cells}{off_max:>9}{on_max:>9}"

    for item in temporal_qs:
        print(_line(item))
    for item in other_qs:
        print(_line(item))

    print("\n===== ⑤ 分组汇总 =====")
    summary: Dict[str, Dict[str, Any]] = {}
    for group_name, group_qs in (("时序组", temporal_qs), ("非时序组", other_qs)):
        total = sum(len(q["gold_facts"]) for q in group_qs)
        for arm in ("OFF", "ON"):
            per_rep = [sum(results[arm][rep][q["id"]]["hits"] for q in group_qs)
                       for rep in range(args.repeats)]
            chars = [sum(results[arm][rep][q["id"]]["chars"] for q in group_qs)
                     for rep in range(args.repeats)]
            summary[f"{group_name}-{arm}"] = {"hits": per_rep, "total": total, "chars": chars}
            rng = f"[{min(per_rep)}~{max(per_rep)}]" if len(per_rep) > 1 else f"[{per_rep[0]}]"
            print(f"  {group_name} {arm:<4} 命中 {per_rep} {rng} / {total} "
                  f"| 叙事总长 {chars} | 中位命中 {_median(per_rep):.1f}")

    print("\n  ---- 臂间差（ON − OFF，逐遍配对）----")
    for group_name, group_qs in (("时序组", temporal_qs), ("非时序组", other_qs)):
        total = sum(len(q["gold_facts"]) for q in group_qs)
        deltas = []
        char_deltas = []
        for rep in range(args.repeats):
            o = sum(results["OFF"][rep][q["id"]]["hits"] for q in group_qs)
            n = sum(results["ON"][rep][q["id"]]["hits"] for q in group_qs)
            deltas.append(n - o)
            co = sum(results["OFF"][rep][q["id"]]["chars"] for q in group_qs)
            cn = sum(results["ON"][rep][q["id"]]["chars"] for q in group_qs)
            char_deltas.append(cn - co)
        print(f"  {group_name} 命中差 {deltas} 中位 {_median(deltas):+.1f} / {total}"
              f" | 叙事长度差 {char_deltas} 中位 {_median(char_deltas):+.0f} 字符")

    print("\n  ---- 臂内自波动（同一臂两遍之差，|Δ|）----")
    for group_name, group_qs in (("时序组", temporal_qs), ("非时序组", other_qs)):
        for arm in ("OFF", "ON"):
            if args.repeats < 2:
                continue
            base = sum(results[arm][0][q["id"]]["hits"] for q in group_qs)
            spread = [sum(results[arm][rep][q["id"]]["hits"] for q in group_qs) - base
                      for rep in range(args.repeats)]
            cb = sum(results[arm][0][q["id"]]["chars"] for q in group_qs)
            cspread = [sum(results[arm][rep][q["id"]]["chars"] for q in group_qs) - cb
                       for rep in range(args.repeats)]
            print(f"  {group_name} {arm:<4} 遍内命中偏移 {spread} | 叙事长度偏移 {cspread}")

    print("\n===== ⑥ 逐条路由判定稳定性（开臂两遍是否一致）=====")
    flip = []
    for q in queries:
        verdicts = [results["ON"][rep][q["id"]]["route"]["verdict"] for rep in range(args.repeats)]
        if len(set(verdicts)) > 1:
            flip.append((q["id"], verdicts))
    print(f"  判定翻转的查询: {flip or '无'}")
    tempo_hit = [q["id"] for q in temporal_qs
                 if results["ON"][0][q["id"]]["route"]["verdict"] == "时序"]
    print(f"  时序组被判定为「时序」（第 1 遍）: {tempo_hit} / {len(temporal_qs)}")
    other_hit = [q["id"] for q in other_qs
                 if results["ON"][0][q["id"]]["route"]["verdict"] == "时序"]
    print(f"  非时序组被误判为「时序」（第 1 遍）: {other_hit or '无'}")
    no_main = [q["id"] for q in queries
               if not results["ON"][0][q["id"]]["route"]["main_channel"]]
    print(f"  主通道未执行的查询（第 1 遍，应为空）: {no_main or '无'}")

    engine_on.close()
    engine_off.close()

    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)
    print(f"\n# 已清理数据目录 {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
