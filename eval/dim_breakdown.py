# -*- coding: utf-8 -*-
import json
from collections import defaultdict

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
SYSTEMS = ["Ariadne", "Mem0", "Graphiti"]

FILES = {
    ("qwen", "story"): "judge_comprehensive.json",
    ("qwen", "raw"): "judge_ariadneraw_comprehensive.json",
    ("deepseek", "story"): "judge_deepseek.json",
    ("deepseek", "raw"): "judge_ariadneraw_deepseek.json",
    ("gemini", "story"): "judge_gemini.json",
    ("gemini", "raw"): "judge_ariadneraw_gemini.json",
    ("gpt", "story"): "judge_gpt.json",
    ("gpt", "raw"): "judge_ariadneraw_gpt.json",
}


def load(path):
    with open("eval/outputs/" + path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d["results"]:
        sc = r.setdefault("scores", {})
        out[r["query_id"]] = {"category": r.get("category", ""),
                              "dim": {s: {k: v for k, v in sc[s].items()} for s in SYSTEMS if sc.get(s)}}
    return out


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


# 共享非弃权（以 story/qwen 为准）
story_qwen = load(FILES[("qwen", "story")])
raw_qwen = load(FILES[("qwen", "raw")])
ids = sorted(q for q in story_qwen if story_qwen[q]["category"] != "abstention" and q in raw_qwen)
print("非弃权查询:", len(ids))

MODELS = ["qwen", "deepseek", "gpt"]  # gemini-raw 失效，剔除；glm 弃用


def dim_avg(variant, model, system, dim):
    mm = load(FILES[(model, variant)])
    vals = [mm[q]["dim"][system][dim] for q in ids if system in mm[q]["dim"]]
    return avg(vals)


print("\n===== Ariadne 各维度（story vs raw）=====")
print(f"{'模型':<12}{'变体':<7}" + "".join(f"{d:<12}" for d in DIMS) + "total")
for model in MODELS:
    for v in ("story", "raw"):
        dims = [dim_avg(v, model, "Ariadne", d) for d in DIMS]
        print(f"{model:<12}{v:<7}" + "".join(f"{x!s:<12}" for x in dims) +
              f"{avg(dims)}")

print("\n===== 三系统『易理解性』（story vs raw）=====")
print(f"{'模型':<12}{'变体':<7}" + "".join(f"{s:<12}" for s in SYSTEMS))
for model in MODELS:
    for v in ("story", "raw"):
        print(f"{model:<12}{v:<7}" + "".join(f"{dim_avg(v, model, s, '易理解性')!s:<12}" for s in SYSTEMS))

print("\n===== qwen 主判分：Ariadne 易理解性 逐维(首5条) story vs raw =====")
for q in ids[:5]:
    s = story_qwen[q]["dim"]["Ariadne"]
    r = raw_qwen[q]["dim"]["Ariadne"]
    print(f"  {q}: story 丰富{s['记忆丰富度']}/易理解{s['易理解性']}/建模{s['用户建模度']}"
          f"   vs  raw 丰富{r['记忆丰富度']}/易理解{r['易理解性']}/建模{r['用户建模度']}")
