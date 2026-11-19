# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""换 embedding 模型后的**余弦阈值重标**探针。

问题
----
链路上有四个阈值本质是「余弦阈值」，它们**与 embedding 模型强绑定**，换模型必然失效：

| 阈值 | 位置 | 默认 | 语义 |
|---|---|---|---|
| `purpose_filter_threshold` | retriever | 0.2 | 候选节点 ↔ **目的向量** 的余弦下限 |
| `abstain_cosine` | alm config | 0.35 | 查询 ↔ 最佳候选 的余弦硬下限（低于即弃权） |
| `abstain_verify_hi` | alm config | 0.50 | 模糊带上界（以下交 LLM 判定） |
| `dedup_threshold` | graph_builder | 0.85 | 节点 ↔ 节点 的余弦（达此值判为同一条记忆） |

0.2 的来历见 `0828PAR检索理论文.md` §3.3.2：**bge-large-zh-v1.5** 下「缓解 vs 喝咖啡」
余弦落在 0.2–0.3，取 0.3 会误杀跳转轴扩展出的功能等价结果。也就是说该值是**为那个模型量的**，
不是通用常数。

方法
----
用**同一个 API 端点、同一批文本**分别跑新旧模型，量三组两两余弦的分布：

  ① 目的过滤：`cos(目的短语, 相关节点)` vs `cos(目的短语, 无关节点)`
     相关节点里刻意含「**功能等价**」样本（动机→行为，如 "缓解压力" ↔ "喝咖啡"）——
     这正是 0828 放宽到 0.2 的原因，新阈值必须仍能放过它们。

  ② 弃权：`cos(查询, 有答案节点)` vs `cos(查询, 无关节点)`
     有答案样本含「措辞贴合」与「措辞距离远」两档（后者是模糊带存在的理由）。

  ③ 语义去重：`cos(同事实不同措辞)` vs `cos(同事实改了数值)` vs `cos(不同事实)`
     第二条是**必须不去重**的（D1 新值覆盖依赖它）。

输出每个模型的分布与「最优分开点」，两模型并排对比，给出建议新值。

合规：探针文本为**本仓自写**的中英对照样本，不取自平台评测数据。

用法：
    python -m eval.probe_threshold_retune
    python -m eval.probe_threshold_retune --models BAAI/bge-large-zh-v1.5 BAAI/bge-m3
"""

import argparse
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_p_baseline import _load_dotenv  # noqa: E402

DEFAULT_MODELS = ["BAAI/bge-large-zh-v1.5", "BAAI/bge-m3"]

# ── ① 目的过滤：目的短语 → 相关 / 无关节点 ──────────────────────────────────
# "functional" 标记的是「功能等价」样本：目的描述动机、节点描述行为或缓解手段，
# 表面语义距离远但**必须保留**（0828 §3.3.2 放宽阈值的直接原因）。
PURPOSE_CASES: List[Tuple[str, List[str], List[str], List[str]]] = [
    # (目的短语, 直接相关, 功能等价(也必须保留), 无关)
    ("了解工作压力", ["用户最近项目很赶，每天加班到十一点"],
     ["用户下午得靠咖啡续命", "用户颈椎经常酸痛"],
     ["用户喜欢看纪录片", "用户养了两只猫"]),
    ("理解情绪状态", ["用户对临时加需求感到烦躁"],
     ["用户晚上常常睡不着", "用户开始每天晨跑"],
     ["用户的公司在杭州", "用户用的数据库是 PostgreSQL"]),
    ("restore the timeline of what happened", ["The user joined the company in March"],
     ["The user moved to Hangzhou", "The user finished the onboarding training"],
     ["The user prefers oat milk", "The user owns a red bicycle"]),
    ("understand work stress", ["The user worked overtime until 11pm every day this week"],
     ["The user drinks coffee to stay awake", "The user's neck hurts from long hours"],
     ["The user watched a documentary last night", "The user has two cats"]),
]

# ── ② 弃权：查询 → 有答案 / 无答案 ─────────────────────────────────────────
# 每项: (查询, 有答案节点, 无答案节点)
ABSTAIN_CASES: List[Tuple[str, List[str], List[str]]] = [
    ("用户平时喜欢喝什么咖啡？", ["用户最喜欢拿铁"], ["用户在一家互联网公司做后端开发", "用户养了两只猫"]),
    ("用户高考考得怎么样？", ["用户高考失利，没考上理想的大学"], ["用户喜欢看纪录片", "用户最近在学吉他"]),
    ("用户的颈椎怎么了？", ["用户加班太多导致颈椎经常酸痛"], ["用户喜欢喝咖啡", "用户周末常去 Fox Hollow 徒步"]),
    ("What kind of coffee does the user like?",
     ["The user's favourite coffee is a latte"],
     ["The user works as a backend developer", "The user has two cats"]),
    ("How did the user do in the college entrance exam?",
     ["The user failed the college entrance exam and did not get into the desired university"],
     ["The user likes documentaries", "The user started learning guitar recently"]),
    ("What happened to the user's neck?",
     ["The user's neck often hurts because of too much overtime"],
     ["The user likes coffee", "The user hikes at Fox Hollow on weekends"]),
]

# ── ③ 语义去重：三档，第二档必须**不**去重 ─────────────────────────────────
# (基准文本, 应去重(同事实不同措辞), 不应去重(改了数值/不同事实))
DEDUP_CASES: List[Tuple[str, List[str], List[str]]] = [
    ("用户在一家互联网公司做后端开发",
     ["用户是某互联网公司的后端工程师", "该用户从事后端开发工作"],
     ["用户在一家互联网公司做前端开发", "用户已经离职，目前待业"]),
    ("用户今年 3 月入职了现在的公司",
     ["用户是今年 3 月加入现在这家公司的"],
     ["用户是今年 7 月入职的", "用户去年 3 月就入职了"]),
    ("The user works as a backend developer at an internet company",
     ["The user is a backend engineer at an internet company"],
     ["The user works as a frontend developer at an internet company",
      "The user left the company and is currently unemployed"]),
    ("The user joined the current company in March this year",
     ["The user started at the current company in March this year"],
     ["The user joined in July this year", "The user joined in March last year"]),
]


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def _best_split(pos: List[float], neg: List[float]) -> Tuple[float, float]:
    """最大化平衡准确率的阈值（阈值取两分布之间），返回 (阈值, 平衡准确率)"""
    if not pos or not neg:
        return float("nan"), float("nan")
    cands = sorted(set(pos + neg))
    best_t, best_acc = float("nan"), -1.0
    for t in cands:
        tpr = sum(1 for v in pos if v >= t) / len(pos)
        tnr = sum(1 for v in neg if v < t) / len(neg)
        acc = (tpr + tnr) / 2
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def _fmt(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    return (f"p10={_quantile(vals, .1):.3f} p50={_quantile(vals, .5):.3f} "
            f"p90={_quantile(vals, .9):.3f} min={min(vals):.3f}")


def measure(model: str) -> Dict[str, Dict]:
    from dba_pipeline.embedding.store import OpenAIEmbeddings

    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    api_base = os.environ.get("EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE") or ""
    if not api_key or not api_base:
        raise SystemExit("缺少 EMBEDDING_API_KEY / EMBEDDING_API_BASE（见 .env）")

    emb = OpenAIEmbeddings(api_key=api_key, api_base=api_base, model=model)

    def vecs(texts: List[str]) -> Dict[str, List[float]]:
        out = emb.embed_documents(texts)
        return {t: v for t, v in zip(texts, out)}

    res: Dict[str, Dict] = {}

    # ① 目的过滤
    p_pos, p_pos_direct, p_pos_func, p_neg = [], [], [], []
    for purpose, direct, func, neg in PURPOSE_CASES:
        all_t = [purpose] + direct + func + neg
        v = vecs(all_t)
        pv = v[purpose]
        for t in direct:
            c = _cos(pv, v[t]); p_pos.append(c); p_pos_direct.append(c)
        for t in func:
            c = _cos(pv, v[t]); p_pos.append(c); p_pos_func.append(c)
        for t in neg:
            p_neg.append(_cos(pv, v[t]))
    t_p, acc_p = _best_split(p_pos, p_neg)
    res["purpose"] = {"pos": p_pos, "pos_direct": p_pos_direct, "pos_func": p_pos_func,
                      "neg": p_neg, "split": t_p, "acc": acc_p}

    # ② 弃权
    a_pos, a_neg = [], []
    for query, ans, non in ABSTAIN_CASES:
        all_t = [query] + ans + non
        v = vecs(all_t)
        qv = v[query]
        for t in ans:
            a_pos.append(_cos(qv, v[t]))
        for t in non:
            a_neg.append(_cos(qv, v[t]))
    t_a, acc_a = _best_split(a_pos, a_neg)
    res["abstain"] = {"pos": a_pos, "neg": a_neg, "split": t_a, "acc": acc_a}

    # ③ 去重
    d_dup, d_nodup = [], []
    for base, dups, nodups in DEDUP_CASES:
        all_t = [base] + dups + nodups
        v = vecs(all_t)
        bv = v[base]
        for t in dups:
            d_dup.append(_cos(bv, v[t]))
        for t in nodups:
            d_nodup.append(_cos(bv, v[t]))
    t_d, acc_d = _best_split(d_dup, d_nodup)
    res["dedup"] = {"dup": d_dup, "nodup": d_nodup, "split": t_d, "acc": acc_d}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="换 embedding 后的余弦阈值重标探针")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", default="eval/outputs/threshold_retune.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    # 探针自带模型名，不受 .env 里 EMBEDDING_MODEL 影响
    all_res = {}
    for m in args.models:
        print(f"\n{'=' * 88}\n模型 {m}\n{'=' * 88}", flush=True)
        r = measure(m)
        all_res[m] = r

        print(f"\n① 目的过滤  purpose_filter_threshold")
        print(f"   相关(直接) {_fmt(r['purpose']['pos_direct'])}")
        print(f"   相关(功能等价，必须放过) {_fmt(r['purpose']['pos_func'])}")
        print(f"   无关       {_fmt(r['purpose']['neg'])}")
        print(f"   → 最优分开点 = {r['purpose']['split']:.3f}（平衡准确率 {r['purpose']['acc']:.0%}）")

        print(f"\n② 弃权      abstain_cosine / abstain_verify_hi")
        print(f"   有答案 {_fmt(r['abstain']['pos'])}")
        print(f"   无答案 {_fmt(r['abstain']['neg'])}")
        print(f"   → 硬下限候选 = {r['abstain']['split']:.3f}（平衡准确率 {r['abstain']['acc']:.0%}）")

        print(f"\n③ 语义去重  dedup_threshold")
        print(f"   同事实不同措辞(应去重) {_fmt(r['dedup']['dup'])}")
        print(f"   改了数值/不同事实(不应去重) {_fmt(r['dedup']['nodup'])}")
        print(f"   → 最优分开点 = {r['dedup']['split']:.3f}（平衡准确率 {r['dedup']['acc']:.0%}）")

    # 对比表
    print(f"\n{'=' * 88}\n两模型并排对比（阈值该往哪边走）\n{'=' * 88}")
    keys = [("purpose", "purpose_filter_threshold", "0.2"),
            ("abstain", "abstain_cosine", "0.35"),
            ("dedup", "dedup_threshold", "0.85")]
    head = f"{'阈值':<28}{'现默认':>8}" + "".join(f"{m.split('/')[-1]:>22}" for m in args.models)
    print(head)
    print("-" * 88)
    for k, label, cur in keys:
        row = f"{label:<28}{cur:>8}"
        for m in args.models:
            row += f"{all_res[m][k]['split']:>22.3f}"
        print(row)

    # 功能等价样本是否被误杀
    print(f"\n【关键校验】功能等价样本的最低余弦（低于新阈值即被误杀）")
    for m in args.models:
        f = all_res[m]["purpose"]["pos_func"]
        print(f"  {m:<28} min={min(f):.3f} p50={_quantile(f, .5):.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
