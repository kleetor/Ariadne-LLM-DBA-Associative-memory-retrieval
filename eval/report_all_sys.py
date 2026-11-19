# -*- coding: utf-8 -*-
import json
from statistics import mean, pstdev

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
SYS = ["Ariadne", "Mem0", "Graphiti"]
MODELS = ["qwen", "deepseek", "gpt", "claude"]


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        return json.load(f)["results"]


def sys_avg(results, system):
    vs = {d: [] for d in DIMS}
    tots = []
    for r in results:
        if r.get("category") == "abstention":
            continue
        sc = r.setdefault("scores", {}).get(system, {})
        if not sc:
            continue
        ok = True
        for d in DIMS:
            v = sc.get(d)
            if v is None:
                ok = False
                break
            vs[d].append(v)
        if ok:
            tots.append(sc.get("total") if sc.get("total") is not None else mean([sc[d] for d in DIMS]))
    if not tots:
        return None
    return {d: mean(vs[d]) for d in DIMS} | {"total": mean(tots)}


for variant in ["story", "raw"]:
    print(f"\n########## {variant} ##########")
    for m in MODELS:
        runs = {s: [sys_avg(load(f"judge_repeat_{variant}_{m}_{i}.json"), s) for i in (1, 2, 3)] for s in SYS}
        print(f"\n=== {variant} / {m} ===")
        for s in SYS:
            if all(r is None for r in runs[s]):
                print(f"  {s:<10} (无有效数据)")
                continue
            print(f"  {s:<10}")
            for dim in DIMS + ["total"]:
                vals = [r[dim] for r in runs[s] if r is not None]
                if not vals:
                    continue
                print(f"      {dim:<8} " + " / ".join(f"{v:.2f}" for v in vals) + f"   | std={pstdev(vals):.3f}")
