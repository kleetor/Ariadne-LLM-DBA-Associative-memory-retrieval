# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 一轮冒烟日志的机械分析（不调 LLM、不联网）。

用法：
    python eval/analyze_alm_round.py <docker-json.log> [more.log ...]

产出四段，对应四类判断：
  1. 机制层      —— 弃权 / 返回 / 降级 / out=8192，判断护栏是否生效、读数是否干净
  2. 逐题得分表  —— 平台回写的 `[IO] qa_feedback`（qa_id/method/score）与我们的返回对齐
  3. 得分 vs 长度—— 为「answer 模型侧有输入上限」这一假设提供可读的相关性
  4. B2 嫌疑题   —— 多选题（≥3 个选项行）逐题打印我们的返回开头，人工核关键场景

为什么要单独做第 3 段：ALM 只返回 1 条记忆，平台 answer 模型看到的就是这条
`content`。若 `score=0` 集中在**长返回**上，才谈得上"输入被截断"；
若长短都有 0，则与长度无关，方向应回到内容本身（缺失关键场景）。
必须在跑之前定下这个判据，否则事后无法区分这两种解释。
"""

import collections
import json
import re
import statistics
import sys
from pathlib import Path

REC_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \w+ ")
OPTION_LINE = re.compile(r"^\s*([A-H])\s*[.、)．）]\s*\S", re.MULTILINE)


def load(path: str):
    """把 docker json-file 日志还原成逻辑记录（长消息被 16KB 切开，需拼回）"""
    items, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line).get("log", "")
        except Exception:
            continue
        if REC_PREFIX.match(payload):
            if cur:
                items.append(cur)
            cur = {"text": payload, "iso": "", "bytes": len(payload.encode())}
        elif cur is not None:
            cur["text"] += payload
            cur["bytes"] += len(payload.encode())
    if cur:
        items.append(cur)
    return items


SEARCH_IN = re.compile(r"\[IO\] search 入参 空间=(\S+) top_k=\d+ query=(.*?)\n选项:\n", re.S)
# 0927 起 `[IO] search 返回` 不再带 `采纳=`（覆盖率度量在句式重复的数据集上饱和，不可信），
# 故该组改为可选：旧日志仍解析、新日志留空，组号不变。
SEARCH_OUT = re.compile(
    r"\[IO\] search 返回 空间=(\S+) 条数=\d+ (?:采纳=(\d+) )?喂入点=(\d+) 喂入边=(\d+) 边型=\{.*?\}:\n",
    re.S)
# `检索:` 行第 6 个字段在 0927 前是 `采纳=`、之后是 `正文=`（正文字符数）——两者都收，组号不变。
SEARCH_LOG = re.compile(
    r"空间 (\S+) 检索: best_cos=([\d.]+) 候选=(\d+) 峰值=(\d+) 跳=(\d+) (?:采纳|正文)=(\d+) 返回=(\d+) "
    r"路径=(\S+) 喂入点=(\d+) 喂入边=(\d+) 图cos=([\d.]+) 文cos=([\d.]+) 文档块=(\d+)")
TOKENS = re.compile(r"LLM #(\d+): in=(\d+) out=(\d+)")
# 0927 起另有一条不受 `trace_io` 控制的 `[QA] qa_feedback`（只有 qa_id/method/score）；
# 两条同轮同题会各出一行，故按 (空间, qa_id) 去重（见 collect）。
FEEDBACK = re.compile(r"\[(?:IO|QA)\] qa_feedback 空间=(\S+) (.*)$")
FEED_FIELD = re.compile(r"(qa_id|method|score|question|predicted)=([^\n]*?)(?=\s{2}\w+=|$)")


def collect(items):
    """顺序扫描日志，把每次检索的三件事**按先后配对**成一条记录。

    必须顺序配对：同一空间一轮里可能有多个问题（`u_26***626f` 有 3 个），
    按空间取第一条会把它们的返回张冠李戴。
    顺序恒为 `search 入参` → `检索:` → `search 返回`。
    """
    pending = collections.defaultdict(collections.deque)   # 空间 → 待配对的 query
    recs, feedback, toks, all_queries = [], [], [], collections.defaultdict(list)
    seen_fb = set()          # (空间, qa_id) → 去重 [QA]/[IO] 两条同源留痕
    for it in items:
        text = it["text"].strip()
        m = SEARCH_IN.search(text)
        if m:
            q = m.group(2).strip()
            pending[m.group(1)].append(q)
            all_queries[m.group(1)].append(q)
            continue
        m = FEEDBACK.search(text)
        if m:
            fields = dict(FEED_FIELD.findall(m.group(2)))
            key = (m.group(1), fields.get("qa_id"))
            if key not in seen_fb:
                seen_fb.add(key)
                feedback.append((m.group(1), fields))
            continue
        m = SEARCH_LOG.search(text)
        if m:
            sp = m.group(1)
            q = pending[sp][0] if pending[sp] else ""
            recs.append(dict(sp=sp, query=q, best=float(m.group(2)), cand=int(m.group(3)),
                             peak=int(m.group(4)), hop=int(m.group(5)),
                             returned=int(m.group(7)), path=m.group(8), inj_n=int(m.group(9))))
            continue
        m = SEARCH_OUT.search(text)
        if m:
            sp = m.group(1)
            body = text[m.end():]
            rec = next((r for r in reversed(recs) if r["sp"] == sp and "out_chars" not in r), None)
            if rec is not None:
                rec["out_chars"] = len(body)
                rec["out_body"] = body
                if pending[sp]:
                    pending[sp].popleft()
            continue
        m = TOKENS.search(text)
        if m:
            toks.append((int(m.group(2)), int(m.group(3))))
    return all_queries, recs, feedback, toks


def section1(items, recs, toks):
    print("=" * 96)
    print("【1】机制层：护栏是否生效、读数是否干净")
    print("=" * 96)
    counts = {
        "弃权": len(re.findall(r"弃权:", "\n".join(i["text"] for i in items))),
        "返回": len(recs),
        "返回=0": sum(1 for r in recs if r.get("returned") == 0),
        "路径=拼接": sum(1 for r in recs if r.get("path") == "拼接"),
        "StoryRank 解析失败": len(re.findall(r"StoryRank JSON 解析失败",
                                            "\n".join(i["text"] for i in items))),
        "finish_reason=length": len(re.findall(r"finish_reason=length",
                                               "\n".join(i["text"] for i in items))),
        # 计量器侧的 finish=length：覆盖**写入 + 检索两侧**（story_rank 的告警只管检索侧，
        # 0927 实测写入侧有一次 in=4533 out=32768 撞满上限，当时完全隐形）
        "finish=length(计量器)": len(re.findall(r"finish=length",
                                               "\n".join(i["text"] for i in items))),
        "out>=8192": sum(1 for _, o in toks if o >= 8192),
        "丢弃平台 QA 反馈": len(re.findall(r"丢弃平台 QA 反馈",
                                          "\n".join(i["text"] for i in items))),
        # 0927 新增：篇幅预算是否被模型接受。非 0 说明规则 6 的"不超过 _STORY_MAX_CHARS"
        # 只是倾向而非约束，会触发一次强制压缩重试。
        "叙事超预算(已重试)": len(re.findall(r"叙事超出篇幅预算",
                                             "\n".join(i["text"] for i in items))),
        "超预算硬截断": len(re.findall(r"叙事重试后仍超出预算",
                                       "\n".join(i["text"] for i in items))),
    }
    for k, v in counts.items():
        flag = ""
        if k in ("弃权", "返回=0", "路径=拼接", "finish_reason=length", "out>=8192") and v:
            flag = "   ← 非 0，读数被污染"
        print(f"  {k:<22}: {v}{flag}")
    # 正文长度只在**新口径**（0927 起的 `正文=` 字段）里；旧日志没有该字段 → 显示「—」，
    # 避免与旧的 `采纳=` 计数并排成同一列（见 SEARCH_LOG 的注释）。
    _txt = "\n".join(it["text"] for it in items)
    bl = sorted(int(x) for x in re.findall(
        r"检索: best_cos=[\d.]+ 候选=\d+ 峰值=\d+ 跳=\d+ 正文=(\d+)", _txt))
    oc = sorted(r["out_chars"] for r in recs if "out_chars" in r)
    if bl:
        # ≥1900 即接近 2000 字硬预算
        print(f"  正文(检索行) 中位={bl[len(bl)//2]} max={bl[-1]} ≥1900 的次数="
              f"{sum(1 for x in bl if x >= 1900)}")
    if oc:
        print(f"  正文字符(IO转储) 中位={oc[len(oc)//2]} max={oc[-1]}")
    if toks:
        o = sorted(x[1] for x in toks)
        print(f"  LLM out   中位={o[len(o)//2]} p90={o[int(len(o)*0.9)]} max={o[-1]}")


def section2(recs, feedback):
    print()
    print("=" * 96)
    print("【2】逐题得分表（平台回写 qa_id / method / score ↔ 我们的返回）")
    print("=" * 96)
    if not feedback:
        print("  本轮没有任何 [IO] qa_feedback —— 说明该轮平台未回写，或护栏未生效（看第 1 段）")
        return []
    pairs = []
    for sp, fields in feedback:
        q = fields.get("question", "")
        probe = " ".join(q.split())[:60].lower()
        # 用题面前 60 字符把回写对到当次 search；对不上则退回该空间的顺序配对
        match = next((r for r in recs if r["sp"] == sp
                      and " ".join(r["query"].split())[:60].lower() == probe), None)
        if match is None:
            match = next((r for r in recs if r["sp"] == sp), None)
        pairs.append(dict(sp=sp, qa_id=fields.get("qa_id", "?"), method=fields.get("method", "?"),
                          score=fields.get("score", "?"), question=q,
                          predicted=fields.get("predicted", ""),
                          chars=match.get("out_chars") if match else None,
                          adopted=match.get("adopted") if match else None,
                          hop=match.get("hop") if match else None))
    print(f"  {'qa_id':<30}{'score':>6}{'正文':>7}{'采纳':>6}{'跳':>4}  predicted / question")
    for p in pairs:
        print(f"  {p['qa_id']:<30}{p['score']:>6}{str(p['chars']):>7}{str(p['adopted']):>6}"
              f"{str(p['hop']):>4}  predicted={p['predicted']!r}")
        print(f"  {'':<46}question={p['question'][:96]!r}")
    return pairs


def section3(pairs):
    print()
    print("=" * 96)
    print("【3】得分 vs 返回长度 —— 「answer 模型侧有输入上限」假设的判据")
    print("=" * 96)
    usable = [p for p in pairs if p["chars"] is not None and p["score"] not in (None, "?")]
    if not usable:
        print("  无可用配对（缺 score 或缺正文长度）")
        return
    for p in sorted(usable, key=lambda x: x["score"]):
        print(f"  score={p['score']:>5}  正文={p['chars']:>7} 字符  跳={p['hop']}  {p['qa_id']}")
    zero = [p["chars"] for p in usable if float(p["score"]) == 0.0]
    pos = [p["chars"] for p in usable if float(p["score"]) > 0.0]
    print(f"\n  score=0 组: n={len(zero)}  正文 "
          f"{('中位=' + str(statistics.median(zero)) + ' 区间=[' + str(min(zero)) + ',' + str(max(zero)) + ']') if zero else '（无）'}")
    print(f"  score>0 组: n={len(pos)}  正文 "
          f"{('中位=' + str(statistics.median(pos)) + ' 区间=[' + str(min(pos)) + ',' + str(max(pos)) + ']') if pos else '（无）'}")
    print("\n  判读：若 score=0 组与 score>0 组的正文长度**没有量级差**，则与长度无关，")
    print("        应回到内容（关键场景是否被选中），而不是去查 answer 模型的输入上限。")


def section4(recs):
    print()
    print("=" * 96)
    print("【4】B2 嫌疑题（多选题：≥3 个选项行）—— 逐题看我们的返回开头")
    print("=" * 96)
    hits = 0
    for r in recs:
        n_opts = len({m.group(1) for m in OPTION_LINE.finditer(r["query"])})
        if n_opts < 3:
            continue
        hits += 1
        print(f"\n  --- {r['sp']}  选项 {n_opts} 个  正文={r.get('out_chars')}  "
              f"喂入点={r.get('inj_n')}  跳={r.get('hop')}  路径={r.get('path')} ---")
        print(f"  问：{r['query'][:240]}")
        body = " ".join((r.get("out_body") or "").split())
        print(f"  答：{body[:560]}")
    if hits == 0:
        print("  本轮没识别出多选题（平台每轮题目不同，属正常）")


def main(paths):
    for path in paths:
        print(f"\n\n########## {Path(path).name} ##########")
        items = load(path)
        _, recs, feedback, toks = collect(items)
        print(f"逻辑记录 {len(items)} 条；检索 {len(recs)} 次；平台回写 {len(feedback)} 条\n")
        section1(items, recs, toks)
        pairs = section2(recs, feedback)
        section3(pairs)
        section4(recs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
