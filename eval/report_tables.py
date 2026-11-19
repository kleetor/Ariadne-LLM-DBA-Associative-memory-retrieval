# -*- coding: utf-8 -*-
import json
from statistics import mean, pstdev

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        return json.load(f)["results"]


def dims_avg(results, dim=None):
    vs, tots = {d: [] for d in DIMS}, []
    for r in results:
        if r.get("category") == "abstention":
            continue
        sc = r.setdefault("scores", {}).get("Ariadne", {})
        if not all(sc.get(d) is not None for d in DIMS) or sc.get("总", None) == 0:
            if all(sc.get(d) == 0 for d in DIMS):
                continue  # 解析失败/全0，剔除
            if not all(sc.get(d) is not None for d in DIMS):
                continue
        for d in DIMS:
            vs[d].append(sc[d])
        tots.append(sc.get("total") if sc.get("total") is not None else mean([sc[d] for d in DIMS]))
    if not tots:
        return None
    return {d: mean(vs[d]) for d in DIMS} | {"total": mean(tots)}


MODELS = ["qwen", "deepseek", "gpt"]
print("===== 总表（Ariadne 各维度均分，重复 run1，新prompt）=====")
for variant, lab in [("story", "story"), ("raw", "raw")]:
    print(f"\n[{lab}]")
    print(f"{'judge':<10}" + "".join(f"{d:<10}" for d in DIMS) + "total")
    for m in MODELS:
        d = dims_avg(load(f"judge_repeat_{variant}_{m}_1.json"))
        if d:
            print(f"{m:<10}" + "".join(f"{d[k]:.2f}".ljust(10) for k in DIMS) + f"{d['total']:.2f}")

print("\n===== 三次方差表（Ariadne total）=====")
for variant, lab in [("story", "story"), ("raw", "raw")]:
    print(f"\n[{lab}]")
    for m in MODELS:
        vals = []
        for i in (1, 2, 3):
            d = dims_avg(load(f"judge_repeat_{variant}_{m}_{i}.json"))
            vals.append(d["total"] if d else None)
        print(f"{m:<10} 3次={['%.3f' % v if v is not None else '-' for v in vals]}  均={mean([v for v in vals if v is not None]):.3f}  std={pstdev([v for v in vals if v is not None]):.3f}")
