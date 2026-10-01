# -*- coding: utf-8 -*-
"""Full 日志失败形态分类：缺料式 vs 答错式。

数据源：容器 stdout 日志（JSON 行，log 字段为文本）。
判据：
  - score=0 且 predicted 命中"无信息"话术  -> 缺料式（检索没给到料）
  - score=0 且 predicted 给出了具体内容    -> 答错式（有料但答错）
顺带把每个问题与它前面最近一条 `检索:` 行配对，比较两类的检索侧信号。
"""
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\makot\Desktop\Ariadne\f4a76242e0944ad40c0e5fb3bc16bb36c5c48155a6ca72040c47719d3793b36d-json (1).log"

QA_RE = re.compile(r"\[IO\] qa_feedback 空间=(\S+) qa_id=(\S+)  method=(\S+)  score=([\d.]+)  question=(.*)  predicted=(.*)$")
RET_RE = re.compile(r"空间 (\S+) 检索: (.*)$")
KV_RE = re.compile(r"([\u4e00-\u9fa5A-Za-z_]+)=(-?[\d.]+)")

NOMEM = re.compile(
    r"(?i)("
    r"do(es)?\s+not\s+(mention|specify|provide|state|include|contain|indicate|suggest|detail|clarify|record|give|say)"
    r"|don'?t\s+(mention|specify|provide|state|include|contain|indicate|suggest)"
    r"|no\s+(information|mention|record|details?|evidence|data|answer)"
    r"|not\s+(mentioned|specified|provided|stated|available|enough|sufficient|clear|present)"
    r"|cannot\s+(be\s+)?(determine|answer|infer|find|locate|identify)"
    r"|can'?t\s+(determine|answer|find|tell)"
    r"|unclear|unknown|insufficient"
    r"|i\s+don'?t\s+know"
    r"|none\s+of\s+the\s+(above|memories|options)"
    r"|the\s+(memories|memory|provided|context|information|conversation|text|excerpt)"
    r"|based\s+on\s+the\s+(provided|given)"
    r"|无法(确定|回答|判断|得知)|没有(提到|提及|记录|相关|明确)|未(提及|记录|说明|给出|包含)|不清楚|不知道"
    r")"
)

fields = ["best_cos", "候选", "峰值", "跳", "正文", "返回", "喂入点", "喂入边", "文档块", "直命中", "文cos", "图cos"]


def num(d, k):
    v = d.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    n_lines = 0
    qas = []          # dict per question
    last_ret = {}     # 空间 -> 检索字段 dict
    per_space_ret = Counter()

    with io.open(LOG, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if "[IO] qa_feedback" not in raw and "检索:" not in raw:
                continue
            n_lines += 1
            try:
                obj = json.loads(raw)
                text = obj.get("log", "")
            except Exception:
                text = raw
            text = text.rstrip("\n")

            if "检索:" in text:
                m = RET_RE.search(text)
                if m:
                    space, rest = m.group(1), m.group(2)
                    d = {k: v for k, v in KV_RE.findall(rest)}
                    last_ret[space] = d
                    per_space_ret[space] += 1
                continue

            m = QA_RE.search(text)
            if not m:
                # 退化匹配：字段间可能有单个空格
                m2 = re.search(r"\[IO\] qa_feedback 空间=(\S+) qa_id=(\S+)\s+method=(\S+)\s+score=([\d.]+)\s+question=(.*?)\s*predicted=(.*)$", text)
                if not m2:
                    continue
                m = m2
            space, qa_id, method, score, question, predicted = m.groups()
            track = qa_id.split(":")[0]
            rec = {
                "space": space,
                "qa_id": qa_id,
                "track": track,
                "method": method,
                "score": float(score),
                "question": question,
                "predicted": predicted.strip(),
                "ret": last_ret.get(space, {}),
            }
            rec["nomem"] = bool(NOMEM.search(rec["predicted"])) or rec["predicted"] in ("", "( )", "()")
            qas.append(rec)

    print("扫描命中行: %d   解析出 [QA] 记录: %d" % (n_lines, len(qas)))
    if not qas:
        return

    total = len(qas)
    correct = sum(1 for r in qas if r["score"] >= 1.0)
    wrong = [r for r in qas if r["score"] < 1.0]
    print("总计 %d  正确 %d (%.1f%%)  错误 %d" % (total, correct, 100.0 * correct / total, len(wrong)))

    # ---- 总体分类 ----
    nomem = [r for r in wrong if r["nomem"]]
    answ = [r for r in wrong if not r["nomem"]]
    print("\n== 错误题分类（总体）==")
    print("  缺料式 %d (%.1f%% of 错误, %.1f%% of 全部)" % (len(nomem), 100.0 * len(nomem) / len(wrong), 100.0 * len(nomem) / total))
    print("  答错式 %d (%.1f%% of 错误, %.1f%% of 全部)" % (len(answ), 100.0 * len(answ) / len(wrong), 100.0 * len(answ) / total))

    # 阳性对照：正确题里有多少被判成缺料式（应接近 0）
    fp = [r for r in qas if r["score"] >= 1.0 and r["nomem"]]
    print("  阳性对照: 正确题中被判缺料式 %d 条（越少越好）" % len(fp))
    for r in fp[:5]:
        print("     ! %s  pred=%s" % (r["qa_id"], r["predicted"][:110]))

    # ---- 分赛道 ----
    print("\n== 分赛道 ==")
    tracks = Counter(r["track"] for r in qas)
    print("%-22s %6s %6s %7s %8s %8s %8s" % ("赛道", "题数", "正确", "正确率", "缺料式", "答错式", "缺料占比"))
    for tr in sorted(tracks, key=lambda t: -tracks[t]):
        g = [r for r in qas if r["track"] == tr]
        gc = sum(1 for r in g if r["score"] >= 1.0)
        gw = [r for r in g if r["score"] < 1.0]
        gn = [r for r in gw if r["nomem"]]
        ga = [r for r in gw if not r["nomem"]]
        ratio = 100.0 * len(gn) / len(gw) if gw else 0.0
        print("%-22s %6d %6d %6.1f%% %8d %8d %7.1f%%" % (tr, len(g), gc, 100.0 * gc / len(g), len(gn), len(ga), ratio))

    # ---- 方法维度（判分器）----
    print("\n== 判分方法 ==")
    for mth, c in Counter(r["method"] for r in qas).most_common():
        g = [r for r in qas if r["method"] == mth]
        gw = [r for r in g if r["score"] < 1.0]
        gn = sum(1 for r in gw if r["nomem"])
        print("  %-26s n=%d 错误=%d 缺料=%d (%.0f%%)" % (mth, len(g), len(gw), gn, 100.0 * gn / len(gw) if gw else 0))

    # ---- 检索侧信号对比 ----
    print("\n== 检索侧信号（均值）==")

    def agg(group, key):
        vals = [num(r["ret"], key) for r in group]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def show(label, group):
        pieces = []
        for k in ["best_cos", "正文", "文档块", "喂入点", "返回", "跳"]:
            v, n = agg(group, k)
            pieces.append("%s=%s" % (k, ("%.3f" % v) if v is not None and k in ("best_cos",) else ("%.1f(%d)" % (v, n) if v is not None else "-")))
        print("  %-10s n=%-5d %s" % (label, len(group), "  ".join(pieces)))

    show("正确", [r for r in qas if r["score"] >= 1.0])
    show("缺料式", nomem)
    show("答错式", answ)
    show("全部错误", wrong)

    # 无料（返回=0 / 正文=0）即使答对的情况
    zero_ret = [r for r in wrong if (num(r["ret"], "返回") or 0) == 0 or (num(r["ret"], "正文") or 0) == 0]
    print("\n  其中 [检索返回=0 或 正文=0] 且答错: %d 条" % len(zero_ret))

    # ---- 缺料式的问题形态 ----
    print("\n== 缺料式的问答形态 ==")
    TEMP_RE = re.compile(r"(?i)^\s*(when\b|how long\b|how many (days|weeks|months|years|times|hours|minutes)\b|what time\b|what date\b|how much time\b|since when\b|as of when\b)")
    FIRST_RE = re.compile(r"^\s*([A-Za-z]+)")

    def shape(group, label):
        temporal = sum(1 for r in group if TEMP_RE.search(r["question"] or ""))
        heads = Counter((FIRST_RE.search(r["question"] or "").group(1).lower() if FIRST_RE.search(r["question"] or "") else "?") for r in group)
        top = ", ".join("%s=%d" % (w, c) for w, c in heads.most_common(6))
        print("  %-8s n=%-5d 时间型=%-4d (%.1f%%)   首词Top: %s" % (label, len(group), temporal, 100.0 * temporal / len(group) if group else 0, top))

    shape([r for r in qas if r["score"] >= 1.0], "正确")
    shape(nomem, "缺料式")
    shape(answ, "答错式")

    # MCQ 赛道里 predicted 只是选项字母，单独看
    mcq = [r for r in wrong if re.fullmatch(r"\([A-Z, ]*\)", r["predicted"].strip())]
    print("  其中 MCQ 形态（predicted 仅选项字母）: %d 条，全部落在: %s" % (len(mcq), sorted(set(r["track"] for r in mcq))))

    # ---- 上限估算 ----
    print("\n== 上限估算（二值）==")
    print("  当前正确 %d (%.1f%%)" % (correct, 100.0 * correct / total))
    print("  若把 缺料式 全部救回: %d (%.1f%%)  [+%.1f pt]" % (correct + len(nomem), 100.0 * (correct + len(nomem)) / total, 100.0 * len(nomem) / total))
    print("  若把 答错式 全部救回: %d (%.1f%%)  [+%.1f pt]" % (correct + len(answ), 100.0 * (correct + len(answ)) / total, 100.0 * len(answ) / total))

    # ---- 正文长短分布（缺料式 vs 答错式）----
    def hist(group, label):
        buckets = Counter()
        for r in group:
            v = num(r["ret"], "正文")
            if v is None:
                buckets["无数据"] += 1
            elif v < 400:
                buckets["<400"] += 1
            elif v < 800:
                buckets["400-800"] += 1
            elif v < 1200:
                buckets["800-1200"] += 1
            elif v < 1600:
                buckets["1200-1600"] += 1
            else:
                buckets[">=1600"] += 1
        order = ["<400", "400-800", "800-1200", "1200-1600", ">=1600", "无数据"]
        print("  %-8s %s" % (label, "  ".join("%s=%d" % (b, buckets.get(b, 0)) for b in order)))

    print("\n== 正文长度分布 ==")
    hist(nomem, "缺料式")
    hist(answ, "答错式")

    # ---- 样例 ----
    def dump(title, group, k=8):
        print("\n== 样例：%s ==" % title)
        for r in group[:k]:
            print("  [%s] sc=%.1f cos=%s 正文=%s 文档块=%s" % (
                r["qa_id"], r["score"], r["ret"].get("best_cos", "-"), r["ret"].get("正文", "-"), r["ret"].get("文档块", "-")))
            print("     Q: %s" % r["question"][:150])
            print("     A: %s" % r["predicted"][:150])

    dump("缺料式", nomem, 10)
    dump("答错式", answ, 10)


if __name__ == "__main__":
    main()
