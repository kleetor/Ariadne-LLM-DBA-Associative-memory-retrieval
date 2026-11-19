# -*- coding: utf-8 -*-
import json
from statistics import mean, pstdev

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
MODELS = ["qwen", "deepseek", "gpt", "claude"]


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        return json.load(f)["results"]


def dims_avg(results):
    vs = {d: [] for d in DIMS}
    tots = []
    for r in results:
        if r.get("category") == "abstention":
            continue
        sc = r.setdefault("scores", {}).get("Ariadne", {})
        for d in DIMS:
            v = sc.get(d)
            if v is None or v == 0:
                break
            vs[d].append(v)
        else:
            tots.append(sc.get("total") if sc.get("total") is not None else mean([sc[d] for d in DIMS]))
    if not tots:
        return None
    return {d: mean(vs[d]) for d in DIMS} | {"total": mean(tots)}


for variant in ["story", "raw"]:
    print(f"\n===== {variant} =====  (表头: run1/run2/run3 ; std 为跨3次标准差)")
    for m in MODELS:
        runs = [dims_avg(load(f"judge_repeat_{variant}_{m}_{i}.json")) for i in (1, 2, 3)]
        print(f"\n[{m}]")
        for dim in DIMS + ["total"]:
            vals = [r[dim] for r in runs]
            print(f"  {dim:<8} " + " / ".join(f"{v:.2f}" for v in vals) + f"   | std={pstdev(vals):.3f}")
