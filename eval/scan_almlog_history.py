# SPDX-License-Identifier: AGPL-3.0-only

"""跨轮 almlog 全量扫描：时间线 + 指标对比 + WARNING/ERROR 聚类 + 逐题得分 + 语言/零边批。

用法：
    python eval/scan_almlog_history.py [日志目录]        # 默认 ./almlog，另含根目录的最新一份

为什么需要它：单轮读数噪声大（同一道题在不同轮 1.0/1.0/0.0），只有把全部日志并排看，
才能区分"改动带来的变化"与"轮间抖动"，也才能发现只在某一轮出现过的异常。
本脚本产出的六段对应 `Plan/alm/0927——全链路审查与历史日志发现.md` §6 的全部表格。

解析口径（重要，读结果时注意）：
- 早期日志的 `检索:` 行没有 `图cos=/文cos=/文档块=` 三列，`写入完成:` 行没有
  `分批=/兜底=` 两列，也没有 `[IO] search 返回` 的正文转储 → 相应列为「—」而**不是 0**。
- `finish=length` 字段只在最新一轮才有，旧轮显示 0 **不代表没有截断**。
- **`正文中位/正文max` 在 0927 起是「正文字符数」**（此前这两列是 `采纳=` 计数，更早的
  `拼接` 列同理）：`采纳` 由后端按「二元组覆盖率」算，在句式重复的数据集上会**饱和**
  （实测 529/529 全过、恒等于喂入点），已停用改打正文长度。
- 节点语言统计的是 `[IO] add 落库` 里 `[n1](type) 内容` 形态的行；**不能**当作"语言错配率"
  （平台上确实有中文语料，而日志里没有"入参语言"这一列可对照）。
- `分批=` 与 `max(ceil(条/6), ceil(字/1500))` 的不一致是**预期**：贪心分组只会 ≥ 该式（下界），
  不一致的批都含单条超预算消息（由 `_split_by_chars` 再切）。
"""

import collections
import glob
import io
import json
import os
import re
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "almlog"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = sorted(glob.glob(os.path.join(ROOT, "*-json.log")))

out = io.StringIO()
def P(*a): print(*a, file=out)

REC = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} ")
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} (\w+)\s+([\w.]+): (.*)$", re.S)
SEARCH = re.compile(
    r"检索: best_cos=([\d.]+) 候选=(\d+) 峰值=(\d+) 跳=(\d+) (?:采纳|正文)=(\d+) 返回=(\d+) 路径=(\S+)")
SEARCH_TAIL = re.compile(r"喂入点=(\d+) 喂入边=(\d+) 图cos=([\d.]+) 文cos=([\d.]+) 文档块=(\d+)")
SEARCH_OUT = re.compile(r"\[IO\] search 返回 空间=(\S+) 条数=(\d+) (?:采纳=(\d+) )?喂入点=(\d+) "
                        r"喂入边=(\d+) 边型=(\{.*?\}):\n", re.S)
WRITE = re.compile(r"写入完成: messages=(\d+) chars=(\d+) max_msg=(\d+) 分批=(\d+) 兜底=(\S+) nodes=(\d+)")
FB = re.compile(r"qa_feedback 空间=(\S+)\s+qa_id=(\S*)\s+method=(\S*)\s+score=([\d.]+)")
TOK = re.compile(r"LLM #(\d+): in=(\d+) out=(\d+)")
FALLBACK = re.compile(r"兜底触发: reason=(\S+)")
SKIP = re.compile(r"triage 判定跳过")
NODE_LINE = re.compile(r"^\s*\[(\S+)\]\((action|status|reason|emotion|thing|person)\) (.*)$")
# 边不落日志，但「边连边分布」每次维护都打一行，可用来统计"零边批"（只建节点不建边 = 图退化）
EDGE_DIST = re.compile(r"边连边分布: 批内=(\d+) 跨批=(\d+)（本轮新节点 (\d+) 个）")
CJK = re.compile(r"[\u4e00-\u9fff]")


def load(path):
    """把 docker json-file 的物理行重组成逻辑记录（Docker 会在 16KB 处切分，不丢内容）"""
    recs, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("log", "")
            if REC.match(t):
                if cur is not None:
                    recs.append(cur)
                cur = t
            elif cur is not None:
                cur += t
    if cur is not None:
        recs.append(cur)
    return recs


def scan(path):
    recs = load(path)
    texts = "\n".join(recs)
    d = {"path": os.path.basename(path), "n_rec": len(recs), "recs": recs,
         "size": os.path.getsize(path) / 1024}
    ts = [REC.match(r).group(0).strip() for r in recs if REC.match(r)]
    d["first"] = ts[0] if ts else "?"
    d["last"] = ts[-1] if ts else "?"
    d["searches"] = SEARCH.findall(texts)
    # 正文长度只从**新口径** `正文=` 取（`采纳=` 已判定不可信）；旧轮为空 → 表里显示「—」，
    # 避免把两种口径的数并排成同一列。
    d["bodylens"] = [int(x) for x in re.findall(
        r"检索: best_cos=[\d.]+ 候选=\d+ 峰值=\d+ 跳=\d+ 正文=(\d+)", texts)]
    d["tails"] = SEARCH_TAIL.findall(texts)
    d["writes"] = WRITE.findall(texts)
    # 0927 起同一次回写会同时留两行（`[QA]` 只含 qa_id/method/score，`[IO]` 还含题面），
    # 两侧都命中 FB → 按 (空间, qa_id) 去重，否则逐题得分会被重复计数（0.0/8 而非 0.0/4）。
    _seen_fb, _fb = set(), []
    for sp, qa, m, s in FB.findall(texts):
        if (sp, qa) in _seen_fb:
            continue
        _seen_fb.add((sp, qa))
        _fb.append((sp, qa, m, s))
    d["fb"] = _fb
    d["fallback"] = collections.Counter(FALLBACK.findall(texts))
    d["skip"] = len(SKIP.findall(texts))
    d["abstain"] = len(re.findall(r"弃权:", texts))
    outs = []
    for r in recs:
        m = SEARCH_OUT.search(r)
        if m:
            # 第三项（旧 `采纳=`）已在 0927 取消；正文长度是唯一可信且不依赖 trace_io 的口径
            outs.append((m.group(1), len(r[m.end():])))
    d["outs"] = outs
    ntypes = collections.Counter()
    cjk = lat = 0
    for r in recs:
        if "](" not in r:
            continue
        for line in r.split("\n"):
            m = NODE_LINE.match(line)
            if not m:
                continue
            ntypes[m.group(2)] += 1
            c = m.group(3)
            if CJK.search(c):
                cjk += 1
            elif c.strip():
                lat += 1
    d["ntypes"], d["cjk"], d["lat"] = ntypes, cjk, lat
    dist = EDGE_DIST.findall(texts)
    d["edist"] = (sum(1 for a, b, _ in dist if int(a) + int(b) == 0),
                  len(dist), sum(int(a) + int(b) for a, b, _ in dist))
    return d


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*-json.log"))) + EXTRA
    if not files:
        print(f"未找到日志：{LOG_DIR}/*-json.log 与 {ROOT}/*-json.log")
        return
    rows = sorted([scan(p) for p in files], key=lambda r: r["first"])

    P("=" * 122)
    P("【A】日志时间线（共 %d 份）" % len(rows))
    P("=" * 122)
    for i, r in enumerate(rows, 1):
        P(f"  {i:>3} {r['first'][:19]:<21}{r['last'][:19]:<21}{r['size']:>6.0f}KB{r['n_rec']:>6}  "
          f"{r['path'][:16]}")

    P("\n" + "=" * 122)
    P("【B】跨轮指标对比（「—」= 该轮尚无此列/此功能，不是 0）")
    P("=" * 122)
    P(f"  {'#':>3} {'起':<12}{'写入':>5}{'分批':>5}{'兜底':>5}{'SKIP':>6}{'检索':>5}{'弃权':>5}"
      f"{'拼接':>5}{'SR失败':>7}{'截断':>5}{'正文中位':>9}{'正文max':>8}"
      f"{'文档块':>7}{'得分':>8}")
    for i, r in enumerate(rows, 1):
        # 正文长度只统计**新口径**（`正文=`）；旧轮无此字段 → 显示「—」
        bl = r["bodylens"]
        docblk = sum(int(t[4]) for t in r["tails"]) if r["tails"] else None
        sc = [float(x[3]) for x in r["fb"]]
        texts = "\n".join(r["recs"])
        P(f"  {i:>3} {r['first'][5:16]:<12}{len(r['writes']):>5}"
          f"{sum(int(x[3]) for x in r['writes']):>5}"
          f"{sum(1 for x in r['writes'] if x[4] != '无'):>5}{r['skip']:>6}"
          f"{len(r['searches']):>5}{r['abstain']:>5}"
          f"{sum(1 for x in r['searches'] if x[6] == '拼接'):>5}"
          f"{len(re.findall('StoryRank JSON 解析失败', texts)):>7}"
          f"{len(re.findall('finish=length', texts)):>5}"
          f"{(med(bl) if bl else '—'):>9}{max(bl) if bl else '—':>8}"
          f"{(docblk if docblk is not None else '—'):>7}"
          f"{(f'{sum(sc):.1f}/{len(sc)}' if sc else '—'):>8}")

    P("\n" + "=" * 122)
    P("【C】WARNING/ERROR 聚类（数字归一化为 #，按总次数降序）")
    P("=" * 122)
    lvl = collections.Counter()
    tax = collections.defaultdict(lambda: {"n": 0, "rounds": set(), "sample": ""})
    for r in rows:
        for t in r["recs"]:
            m = TS.match(t)
            if not m or m.group(2) not in ("WARNING", "ERROR", "CRITICAL"):
                continue
            lvl[m.group(2)] += 1
            msg = m.group(4)
            e = tax[re.sub(r"\d+", "#", msg)[:100]]
            e["n"] += 1
            e["rounds"].add(r["first"][5:16])
            if not e["sample"]:
                e["sample"] = msg[:150].replace("\n", " ")
    P(f"  按级别：{dict(lvl)}\n")
    for key, e in sorted(tax.items(), key=lambda kv: -kv[1]["n"]):
        span = f"{min(e['rounds'])}~{max(e['rounds'])}" if len(e["rounds"]) > 1 else min(e["rounds"])
        P(f"  ×{e['n']:<5} {len(e['rounds'])}轮 ({span})  {e['sample']}")

    P("\n" + "=" * 122)
    P("【D】逐题得分（同一 qa_id 跨轮对比）")
    P("=" * 122)
    per_q = collections.defaultdict(list)
    for r in rows:
        for _sp, qa, _m, s in r["fb"]:
            per_q[qa].append((r["first"][5:16], float(s)))
    for qa in sorted(per_q):
        P(f"  {qa:<36}" + "  ".join(f"{t}:{s:g}" for t, s in sorted(per_q[qa])))
    tot = [s for v in per_q.values() for _, s in v]
    if tot:
        P(f"\n  回写 {len(tot)} 条，平均 {sum(tot)/len(tot):.3f}"
          f"（每轮仅 4 条 → 单轮差异不可当结论）")

    P("\n" + "=" * 122)
    P("【E】落库节点 / 语言 / 零边批")
    P("=" * 122)
    P(f"  {'#':>3} {'起':<12}{'节点':>7}{'中文节点':>9}{'中文占比':>9}{'维护':>7}{'零边批':>8}"
      f"{'建边':>7}  节点类型 top5")
    for i, r in enumerate(rows, 1):
        zero, n_dist, sum_e = r["edist"]
        top = " ".join(f"{k}={v}" for k, v in r["ntypes"].most_common(5))
        P(f"  {i:>3} {r['first'][5:16]:<12}{sum(r['ntypes'].values()):>7}{r['cjk']:>9}"
          f"{pct(r['cjk'], r['cjk'] + r['lat']):>9}{n_dist:>7}{zero:>8}{sum_e:>7}  {top[:60]}")

    P("\n" + "=" * 122)
    P("【F】兜底 reason 分布 / 分批与下界公式的差异")
    P("=" * 122)
    for i, r in enumerate(rows, 1):
        if not r["fallback"] and not r["writes"]:
            continue
        mismatch = sum(
            1 for msg, chars, _mx, grp, _fb, _n in r["writes"]
            if max(-(-int(msg) // 6), -(-int(chars) // 1500)) != int(grp))
        P(f"  [{i:>2}] {r['first'][5:16]}  兜底={dict(r['fallback'])}  "
          f"写入 {len(r['writes'])} 批，分批与下界不一致 {mismatch}"
          f"（预期：含单条超预算消息时贪心分组会更细）")

    sys.stdout.buffer.write(out.getvalue().encode("utf-8"))


main()
