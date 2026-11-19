# -*- coding: utf-8 -*-
import json
from statistics import mean, pstdev

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
SYS = ["Ariadne", "Mem0", "Graphiti"]


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        return json.load(f)["results"]


def vals(results, system, dim=None):
    out = {}
    for r in results:
        if r.get("category") == "abstention":
            continue
        sc = r.setdefault("scores", {}).get(system, {})
        if dim:
            out[r["query_id"]] = sc.get(dim)
        else:
            out[r["query_id"]] = sc.get("total") if sc.get("total") is not None else (
                mean([sc[d] for d in DIMS]) if all(sc.get(d) is not None for d in DIMS) else None)
    return {k: v for k, v in out.items() if v is not None}


def dims(r, system):
    sc = r.scores.get(system, {}) if hasattr(r, "scores") else r.setdefault("scores", {}).get(system, {})
    return {d: sc.get(d) for d in DIMS}


# ---------- gemini 通不通 ----------
print("===== gemini raw 是否通过 =====")
g = load("judge_raw_gemini_v2.json")
na = [r for r in g if r.get("category") != "abstention"]
ari_gem = [r.setdefault("scores", {}).get("Ariadne", {}).get("total") for r in na]
non_zero = [x for x in ari_gem if x is not None and x > 0]
zero = [x for x in ari_gem if x == 0]
print(f"非弃权 {len(na)}; Ariadne total 非0={len(non_zero)}, 全0={len(zero)}")
print(f"非弃权样例 Ariadne total: {[r.setdefault('scores',{}).get('Ariadne',{}).get('total') for r in na[:6]]}")

# ---------- qwen 逐次判分方差 ----------
print("\n===== Ariadne total 跨三次判分方差 =====")
for label, base in [("story(带StoryRank)", "judge_repeat_story_qwen_{}.json"),
                     ("raw(不带StoryRank)", "judge_repeat_raw_qwen_{}.json")]:
    runs = [vals(load(base.format(i)), "Ariadne") for i in (1, 2, 3)]
    qids = sorted(runs[0].keys())
    # 每次总均分
    means = [mean([r[q] for q in qids]) for r in runs]
    # 逐条标准差（每条3次值的std，再平均）
    perq_std = mean([pstdev([r[q] for r in runs]) for q in qids])
    print(f"\n[{label}] 3 次均分: {['%.3f' % m for m in means]}  总均分=%.3f  std=%.3f  逐条平均std=%.3f"
          % (mean(means), pstdev(means), perq_std))

    # 逐维跨次
    for dim in DIMS:
        druns = [vals(load(base.format(i)), "Ariadne", dim) for i in (1, 2, 3)]
        dmean = mean([mean([r[q] for q in qids]) for r in druns])
        dstd = pstdev([mean([r[q] for q in qids]) for r in druns])
        print(f"    {dim}: 均=%.3f 跨次std=%.3f  逐条std=%.3f" % (
            dmean, dstd, mean([pstdev([r[q] for r in druns]) for q in qids])))

# ---------- 三系统排名跨次稳定性 ----------
print("\n===== 同题 Ariadne 排名（vs Mem0/Graphiti）跨三次是否稳定 =====")
for label, base in [("story", "judge_repeat_story_qwen_{}.json"), ("raw", "judge_repeat_raw_qwen_{}.json")]:
    runs = [load(base.format(i)) for i in (1, 2, 3)]
    na_runs = [[r for r in rr if r.get("category") != "abstention"] for rr in runs]
    qids = sorted(r[0]["query_id"] for r in na_runs) if na_runs[0] else []
    stable = 0
    total = 0
    for q in qids:
        ranks = []
        for rr in na_runs:
            for r in rr:
                if r["query_id"] == q:
                    t = {s: r.setdefault("scores", {}).get(s, {}).get("total") for s in SYS}
                    t = {s: v for s, v in t.items() if v is not None}
                    if len(t) == 3:
                        ranks.append(sorted(t, key=t.get, reverse=True))
                    break
        if len(ranks) >= 2:
            total += 1
            if all(rk == ranks[0] for rk in ranks):
                stable += 1
    # 单次 rank-for-rank 对比
    from itertools import combinations as comb
    agree = 0
    cnt = 0
    for q in qids:
        for rr in na_runs:
            for r in rr:
                if r["query_id"] == q:
                    t = {s: r.setdefault("scores", {}).get(s, {}).get("total") for s in SYS}
                    t = {s: v for s, v in t.items() if v is not None}
                    if len(t) == 3:
                        cnt += 1
                        if sorted(t, key=t.get, reverse=True)[0] == "Ariadne":
                            agree += 1
                    break
    print(f"[{label}] 跨3次排名完全一致率={stable}/{total}；Ariadne 为单次第一的次数={agree}/{cnt}")
