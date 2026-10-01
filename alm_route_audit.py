# -*- coding: utf-8 -*-
"""Full 日志的**路由可行性测量**：不改线上，只用日志算"若按规则分流，能省多少、会伤多少"。

与 `alm_qa_classify.py` 的分工：那个脚本回答"失败是什么形态"，本脚本回答
"这些题里有多少本来就不该走图谱路线、走它会烧掉多少"。

产出四组数字：

1. **逐题表** —— query / 选项 / 判型 / 检索读数 / 得分（`--dump-csv` 落盘，即探针的训练集骨架）
2. **规则分流的题量占比与得分** —— 每套判据各报一次，用于选阈值
3. **读侧放大器** —— story_rank 的实测调用次数（基准 1 + 超预算重试 + 缩喂重试）
4. **写侧量级** —— 维护组数 / triage 跳过率 / 字符总量

配对方式：**按空间分别排队**。`_search_locked` 全程持空间锁，故同一空间内
「入参 → 检索 → QA 回执」严格有序；跨空间的并发不影响各自队列。

用法::

    python alm_route_audit.py <full.log> [--dump-csv route_rows.csv]
"""

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, deque
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

from alm.route_features import FEATURE_NAMES, extract  # noqa: E402

RE_ENTRY = re.compile(
    r"\[IO\] search 入参 空间=(\S+) top_k=(\d+) 检索用=(\d+)字\(原=(\d+)字\) "
    r"判型=(\S+) 精检=(\S+) 跳上限=(\d+) query=(.*)$"
)
RE_RETRIEVAL = re.compile(r"空间 (\S+) 检索: (.+)$")
RE_QA = re.compile(
    r"\[IO\] qa_feedback 空间=(\S+) qa_id=(\S+)\s+method=(\S+)\s+score=([\d.]+)\s+"
    r"question=(.*?)\s*predicted=(.*)$"
)
RE_WRITE_DONE = re.compile(
    r"写入完成: messages=(\d+) chars=(\d+) max_msg=(\d+) 分批=(\d+) 兜底=(\S+) "
    r"nodes=(\d+) ops_create=(\d+)"
)
RE_KV = re.compile(r"([\u4e00-\u9fa5A-Za-z_]+)=(-?[\d.]+)")

# 读侧的"重试放大器"计数（全局，无法可靠归因到单题——见脚本末尾的局限说明）
RETRY_MARKERS = {
    "超预算重试": "叙事超出篇幅预算",
    "缩喂重试": "叙事仍超预算",
    "缩喂后硬截断": "叙事缩喂至",
    "输出撞顶截断": "StoryRank 输出被 max_tokens 截断",
    "JSON 降级拼接": "StoryRank JSON 解析失败",
}


def _num(d: Dict[str, str], k: str) -> Optional[float]:
    try:
        return float(d.get(k))
    except (TypeError, ValueError):
        return None


def parse(path: str):
    """把日志解析成三份：逐题记录、读侧计数、写侧计数"""
    searches: Dict[str, deque] = {}      # 空间 -> 待配对的检索行
    qa_by_space: Dict[str, deque] = {}   # 空间 -> 待配对的 QA 行
    rows: List[Dict[str, Any]] = []
    retry_counts = Counter()
    write = {"adds": 0, "groups": 0, "chars": 0, "messages": 0,
             "triage_skip": 0, "fallback": Counter()}

    pending: Optional[Dict[str, Any]] = None
    opt_state: Optional[str] = None
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                text = json.loads(raw).get("log", "")
            except Exception:
                text = raw
            text = text.rstrip("\n")

            # `[IO] search 入参` 之后紧跟两行形态：`选项:` 头 + N 行以两个空格开头的选项
            if opt_state == "header":
                opt_state = "opts" if text.strip() == "选项:" else None
                if opt_state is not None:
                    continue
            if opt_state == "opts":
                if text.startswith("  ") and pending is not None:
                    opt = text.strip()
                    if opt and opt != "（无）":
                        pending["options"].append(opt)
                    continue
                opt_state = None  # 选项块结束，本行按常规处理

            if "检索: " in text:
                m = RE_RETRIEVAL.search(text)
                if m:
                    space = m.group(1)
                    kv = {k: v for k, v in RE_KV.findall(m.group(2))}
                    dq = searches.get(space)
                    if dq:
                        entry = dq.popleft()
                        entry["ret"] = kv
                        rows.append(entry)
                continue

            if "[IO] search 入参" in text:
                m = RE_ENTRY.search(text)
                if m:
                    space = m.group(1)
                    entry = {
                        "space": space,
                        "检索用字": int(m.group(3)),
                        "原字": int(m.group(4)),
                        "判型": m.group(5),
                        "精检": m.group(6),
                        "跳上限": int(m.group(7)),
                        "query": m.group(8),
                        "options": [],
                        "ret": {},
                        "score": None,
                        "qa_id": None,
                    }
                    searches.setdefault(space, deque()).append(entry)
                    pending = entry
                    opt_state = "header"
                continue

            if "[IO] qa_feedback" in text:
                m = RE_QA.search(text)
                if m:
                    qa_by_space.setdefault(m.group(1), deque()).append(
                        (m.group(2), float(m.group(4)))
                    )
                continue

            if "写入完成: messages=" in text:
                m = RE_WRITE_DONE.search(text)
                if m:
                    write["adds"] += 1
                    write["messages"] += int(m.group(1))
                    write["chars"] += int(m.group(2))
                    write["groups"] += int(m.group(4))
                    if m.group(5) != "无":
                        for reason in m.group(5).split(","):
                            write["fallback"][reason] += 1
                continue

            for label, marker in RETRY_MARKERS.items():
                if marker in text:
                    retry_counts[label] += 1

    # QA 回执按空间顺序回填（数量可能与检索数差 1~2，见脚本末尾局限）
    for space, dq in qa_by_space.items():
        done = [r for r in rows if r["space"] == space]
        for rec, (qa_id, score) in zip(done, dq):
            rec["qa_id"] = qa_id
            rec["score"] = score

    return rows, retry_counts, write


def _tracks(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out = {}
    for r in rows:
        qa = r.get("qa_id") or ""
        out[id(r)] = qa.split(":")[0] if qa else "?"
    return out


def _rate(group: List[Dict[str, Any]]) -> str:
    scored = [r for r in group if r["score"] is not None]
    if not scored:
        return "n/a"
    hit = sum(1 for r in scored if r["score"] >= 1.0)
    return "%d/%d = %.1f%%" % (hit, len(scored), 100.0 * hit / len(scored))


def report(rows: List[Dict[str, Any]], retry: Counter, write: Dict[str, Any]):
    total = len(rows)
    print("==" * 34)
    print("检索记录 %d 条" % total)

    # ---- 3. 读侧放大器 ----
    print("\n[1] 读侧 story_rank 放大器（全局实测）")
    base = total
    extra = retry["超预算重试"] + retry["缩喂重试"]
    print("    基准调用 %d + 超预算重试 %d + 缩喂重试 %d = %d 次"
          % (base, retry["超预算重试"], retry["缩喂重试"], base + extra))
    if base:
        print("    → 平均 %.2f 次/检索（比基准高 %.0f%%）"
              % ((base + extra) / base, 100.0 * extra / base))
    print("    输出撞顶截断 %d 次 → 其中 JSON 降级拼接 %d 次（降级 = 质量退化为按相关度平铺）"
          % (retry["输出撞顶截断"], retry["JSON 降级拼接"]))
    print("    缩喂后仍超限、硬截断 %d 次" % retry["缩喂后硬截断"])

    # ---- 4. 写侧 ----
    print("\n[2] 写侧量级（全局实测）")
    g = write["groups"]
    print("    Add %d 次 → maintain 组 %d（%.1f 组/Add），字符总量 %d"
          % (write["adds"], g, g / write["adds"] if write["adds"] else 0, write["chars"]))
    skips = write["fallback"].get("triage", 0)
    print("    triage 跳过组 %d / %d = %.1f%%" % (skips, g, 100.0 * skips / g if g else 0))
    print("    兜底原因分布: %s"
          % ", ".join("%s=%d" % (k, v) for k, v in write["fallback"].most_common()))
    if g:
        print("    → 每组非跳过走 节点抽取+边连接 = 2 次 LLM，按此估写侧调用 ≈ %d 次"
              % (g + (g - skips) * 2))

    # ---- 2. 规则分流 ----
    print("\n[3] 规则分流的题量占比与得分")
    print("    判据来自日志自带的 `选项:` 与 `判型=`，全部零 LLM；得分来自 QA 回执")
    tracks = _tracks(rows)

    defs = {
        "A 契约 options 非空": lambda r: bool(r["options"]),
        "B 契约选项 或 作答指令（=判型）": lambda r: r["判型"] == "是" or bool(r["options"]),
        "C B 或 时间型问法": None,  # 需要特征，下面单独算
    }
    feats = [extract(r["query"], r["options"]) for r in rows]
    defs["C B 或 时间型问法"] = lambda r, i: (
        r["判型"] == "是" or bool(r["options"]) or feats[i]["is_temporal"] == 1.0
    )

    for name in defs:
        if name.startswith("C"):
            sel = [rows[i] for i in range(total) if defs[name](rows[i], i)]
        else:
            sel = [r for r in rows if defs[name](r)]
        rest = [r for r in rows if id(r) not in {id(x) for x in sel}]
        print("    %-28s 命中 %4d (%5.1f%%)  得分 %-14s | 其余 %-14s"
              % (name, len(sel), 100.0 * len(sel) / total if total else 0,
                 _rate(sel), _rate(rest)))

    # 分赛道细化（用最强的那套判据 B）
    print("\n[4] 按判据 B 分层（分赛道）")
    for tr in sorted({tracks[id(r)] for r in rows}):
        g_all = [r for r in rows if tracks[id(r)] == tr]
        g_v = [r for r in g_all if r["判型"] == "是" or r["options"]]
        g_g = [r for r in g_all if not (r["判型"] == "是" or r["options"])]
        print("    %-22s 全部 %-16s | 走 V %-16s | 走 G %-16s"
              % (tr, _rate(g_all), _rate(g_v), _rate(g_g)))

    # ---- 1. 逐题表 ----
    print("\n[5] 逐题样例（前 5 条）")
    for r in rows[:5]:
        print("    [%s] 判型=%s 精检=%s 选项=%d 检索用=%d字(原=%d字) score=%s"
              % (r.get("qa_id") or "-", r["判型"], r["精检"], len(r["options"]),
                 r["检索用字"], r["原字"], r["score"]))
        print("        %s" % r["query"][:120])


def dump_csv(rows: List[Dict[str, Any]], path: str):
    tracks = _tracks(rows)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qa_id", "track", "space", "score", "判型", "精检",
                    "检索用字", "原字", "n_options",
                    "best_cos", "峰值", "跳", "正文", "文档块", "喂入点"]
                   + ["f_" + n for n in FEATURE_NAMES] + ["query"])
        for r in rows:
            feats = extract(r["query"], r["options"])
            ret = r["ret"]
            w.writerow([
                r.get("qa_id") or "", tracks[id(r)], r["space"],
                "" if r["score"] is None else r["score"],
                r["判型"], r["精检"], r["检索用字"], r["原字"], len(r["options"]),
                ret.get("best_cos", ""), ret.get("峰值", ""), ret.get("跳", ""),
                ret.get("正文", ""), ret.get("文档块", ""), ret.get("喂入点", ""),
            ] + [("%g" % feats[n]) for n in FEATURE_NAMES] + [r["query"]])
    print("\n逐题表已写出: %s（%d 行）" % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser(description="Full 日志的路由可行性测量")
    ap.add_argument("log", help="容器 stdout 日志（JSON 行）")
    ap.add_argument("--dump-csv", default=None, help="把逐题表（含特征列）写到该 CSV")
    args = ap.parse_args()

    rows, retry, write = parse(args.log)
    if not rows:
        print("未解析到任何检索记录；确认日志里含 `[IO] search 入参`（需 ALM_TRACE_IO=1）")
        return 1
    report(rows, retry, write)
    if args.dump_csv:
        dump_csv(rows, args.dump_csv)

    print("\n" + "==" * 34)
    print("局限（读数时须知道）：")
    print("  1. 读侧重试是**全局计数**，无法可靠归因到单题——`叙事超出篇幅预算` 等行不带"
          "空间/题号，而跨空间并发下按时间窗切分会把别的题算进来。故第 [1] 段只给全局均值。")
    print("  2. QA 回执与检索按空间顺序配对，两者条数可能差 1~2（容器重启会打断在途请求，"
          "见 版本变更记录 §4.19 同期的日志发现），故少数题 score 为空。")
    print("  3. 本脚本只做**规则分流**；真正的判定（探针 / 阈值）不在这里，"
          "`--dump-csv` 出的特征列就是那一步的输入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
