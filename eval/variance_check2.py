# -*- coding: utf-8 -*-
import json
from statistics import mean, pstdev

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
SYS = ["Ariadne", "Mem0", "Graphiti"]
MODELS = ["qwen", "deepseek", "gpt"]


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        return json.load(f)["results"]


def ari_data(results, dim=None):
    """非弃权且 Ariadne 有有效分数 → {qid: score}; 另统计全0(解析失败)行数"""
    out, zero = {}, 0
    for r in results:
        if r.get("category") == "abstention":
            continue
        sc = r.setdefault("scores", {}).get("Ariadne", {})
        if not sc:
            zero += 1
            continue
        if dim:
            v = sc.get(dim)
        else:
            v = sc.get("total") if sc.get("total") is not None else (
                mean([sc[d] for d in DIMS]) if all(sc.get(d) is not None for d in DIMS) else None)
        if v is None:
            zero += 1
            continue
        if v == 0:
            zero += 1
            continue
        out[r["query_id"]] = v
    return out, zero


for base_name, lab in [("story", "story(带StoryRank)"), ("raw", "raw(不带StoryRank)")]:
    print(f"\n================= {lab} =================")
    for model in MODELS:
        runs = []
        zeros = 0
        for i in (1, 2, 3):
            d, z = ari_data(load(f"judge_repeat_{base_name}_{model}_{i}.json"))
            runs.append(d)
            zeros += z
        qids = sorted(runs[0].keys())
        qcomm = [q for q in qids if all(q in r for r in runs)]
        means = [mean([r[q] for q in qcomm]) for r in runs]
        perq = mean([pstdev([r[q] for r in runs]) for q in qcomm])
        print(f"[{model:>8}] 3次均分={['%.3f' % m for m in means]}  总均=%.3f 跨次std=%.3f  逐条std=%.3f  (行数{len(qcomm)}, 全0/无效累计{zeros})"
              % (mean(means), pstdev(means), perq))
