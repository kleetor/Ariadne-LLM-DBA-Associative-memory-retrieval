# -*- coding: utf-8 -*-
"""
e2e 补测 · StoryRank 消融 + judge 交叉评分波动分析。

对比 Ariadne 块 = {story}(有 StoryRank) vs {nodes,edges}(无 StoryRank) 下的 judge 表现：
1. 每个 judge 模型给 Ariadne/Mem0/Graphiti 的 total 均分（两变体各一份）
2. Ariadne story vs raw 的评分差（StoryRank 的增量）
3. judge 交叉评分波动：每变体内，各 judge 模型间的排位(top-1)一致率 / Cohen kappa / Spearman
4. 区域(zh=en)分组

数据有效性说明：
- gemini 对 raw 变体（{nodes,edges} 结构）判分大量失败(全 0，重跑亦如此) → raw 变体剔除 gemini。
- glm 对 story 变体为病态判分（评分通胀/形态偏好，0901 已披露）→ story 变体区域对照剔除 glm，
  但其 Ariadne 分数仍可用于 story-vs-raw 增量比较。

用法：python -m eval.storyrank_ablation
"""

import itertools
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYSTEMS = ["Ariadne", "Mem0", "Graphiti"]
ABSTAIN_CAT = "abstention"
MODELS = ["qwen", "deepseek", "gemini", "gpt"]
REGION = {"qwen": "zh", "deepseek": "zh", "gemini": "en", "gpt": "en"}

# 各变体用于"交叉评分波动"的有效模型（排除失效/病态；glm-4 太老且有形态偏好，已弃用）
VALID_FOR_CROSS = {"story": ["qwen", "deepseek", "gemini", "gpt"],
                   "raw": ["qwen", "deepseek", "gemini", "gpt"]}
# Ariadne story-vs-raw 增量：需两变体都存在且有效
DELTA_MODELS = ["qwen", "deepseek", "gemini", "gpt"]

FNAME = {
    ("qwen", "story"): "judge_comprehensive.json",
    ("deepseek", "story"): "judge_deepseek.json",
    ("gemini", "story"): "judge_gemini.json",
    ("gpt", "story"): "judge_gpt.json",
    ("glm", "story"): "judge_glm.json",
    ("qwen", "raw"): "judge_ariadneraw_comprehensive.json",
    ("deepseek", "raw"): "judge_ariadneraw_deepseek.json",
    ("gemini", "raw"): "judge_ariadneraw_gemini.json",
    ("gpt", "raw"): "judge_ariadneraw_gpt.json",
    ("glm", "raw"): "judge_ariadneraw_glm.json",
}


def load(path: str) -> dict:
    p = ROOT / "eval/outputs" / path
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d["results"]:
        sc = r.get("scores", {})
        out[r["query_id"]] = {"category": r["category"],
                              "totals": {s: sc[s]["total"] for s in SYSTEMS if sc.get(s)}}
    return out


def top1(totals: dict) -> str:
    if not totals:
        return "NA"
    mx = max(totals.values())
    leaders = [s for s, v in totals.items() if v == mx]
    return leaders[0] if len(leaders) == 1 else "TIE"


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def per_system_avg(models: dict, ids) -> dict:
    return {s: avg([models[qid]["totals"][s] for qid in ids if s in models[qid]["totals"]])
            for s in SYSTEMS}


def main():
    variants = {"story": {}, "raw": {}}
    for m in MODELS:
        for v in ("story", "raw"):
            variants[v][m] = load(FNAME[(m, v)])
    avail = {v: [m for m in MODELS if variants[v].get(m)] for v in ("story", "raw")}
    print("判分文件可用模型:", avail)

    # 共同非弃权查询（以 story qwen 的 category 为准）
    base = variants["story"]["qwen"]
    shared_ids = sorted(qid for qid in base
                        if base[qid]["category"] != ABSTAIN_CAT
                        and qid in variants["raw"]["qwen"])
    print(f"共同非弃权查询: {len(shared_ids)}")

    # 动态判定每变体有效模型：Ariadne 得分全 0 → judge 判分失效（如 gemini 对 raw 块系统性解析失败）
    def ari_avg(mm, ids):
        vals = [mm[q]["totals"]["Ariadne"] for q in ids if "Ariadne" in mm[q]["totals"]]
        return avg(vals) if vals else 0.0

    valid = {v: [m for m in MODELS if variants[v].get(m) and (ari_avg(variants[v][m], shared_ids) > 0.5)]
             for v in ("story", "raw")}
    delta_models = [m for m in MODELS if m in valid["story"] and m in valid["raw"]]
    print("各变体有效模型:", valid)

    print("\n===== 每模型系统 total 均分（StoryRank 消融） =====")
    print(f"{'模型':<11}{'变体':<7}{'Ariadne':<9}{'Mem0':<9}{'Graphiti':<9}")
    for m in delta_models:
        for v in ("story", "raw"):
            if not variants[v].get(m):
                continue
            agg = per_system_avg(variants[v][m], shared_ids)
            print(f"{m:<11}{v:<7}{agg['Ariadne']!s:<9}{agg['Mem0']!s:<9}{agg['Graphiti']!s:<9}")
    print("\nAriadne story - raw 评分差（StoryRank 增量，正值=故事化加分）:")
    deltas = {}
    for m in delta_models:
        if variants["story"].get(m) and variants["raw"].get(m):
            s = per_system_avg(variants["story"][m], shared_ids)["Ariadne"]
            r = per_system_avg(variants["raw"][m], shared_ids)["Ariadne"]
            deltas[m] = round(s - r, 2)
            print(f"  {m:<11} story={s:>5.2f} raw={r:>5.2f}  差={s - r:+.2f}")

    # 交叉评分波动
    def pairwise(models: dict, ids):
        out = {}
        for a, b in itertools.combinations(models.keys(), 2):
            ma, mb = models[a], models[b]
            ta, tb = [], []
            for qid in ids:
                ta.append(top1(ma[qid]["totals"]))
                tb.append(top1(mb[qid]["totals"]))
            agree = sum(1 for x, y in zip(ta, tb) if x == y)
            try:
                kappa = cohen_kappa_score(ta, tb)
            except Exception:
                kappa = float("nan")
            va, vb = [], []
            for qid in ids:
                for s in SYSTEMS:
                    if s in ma[qid]["totals"] and s in mb[qid]["totals"]:
                        va.append(ma[qid]["totals"][s])
                        vb.append(mb[qid]["totals"][s])
            rho, pval = spearmanr(va, vb) if len(va) > 2 else (float("nan"), float("nan"))
            out[f"{a}__vs__{b}"] = {"top1": round(agree / len(ids), 3),
                                    "kappa": round(float(kappa), 3),
                                    "spearman": round(float(rho), 3)}
        return out

    print("\n===== judge 交叉评分波动（每变体内两两） =====")
    for v in ("story", "raw"):
        mm = {m: variants[v][m] for m in valid[v] if variants[v].get(m)}
        pw = pairwise(mm, shared_ids)
        label = "story(有 StoryRank)" if v == "story" else "raw(无 StoryRank)"
        print(f"\n--- {label}（模型={list(mm.keys())}） ---")
        if pw:
            print(f"  均值: top1 一致={avg([x['top1'] for x in pw.values()])}  "
                  f"kappa={avg([x['kappa'] for x in pw.values()])}  "
                  f"spearman={avg([x['spearman'] for x in pw.values()])}")
            for k, x in pw.items():
                print(f"    {k}: top1={x['top1']} kappa={x['kappa']:+.3f} spearman={x['spearman']:+.3f}")

    # 区域 zh/en
    print("\n===== 区域(zh/en) Ariadne 均分 =====")
    for v in ("story", "raw"):
        valid_v = [m for m in valid[v] if variants[v].get(m)]
        zh = [m for m in valid_v if REGION[m] == "zh"]
        en = [m for m in valid_v if REGION[m] == "en"]
        print(f"\n--- {v} 变体 ---  有效={valid_v}")
        for zone, ks in (("zh", zh), ("en", en)):
            vals = [variants[v][m][qid]["totals"]["Ariadne"] for m in ks for qid in shared_ids
                    if "Ariadne" in variants[v][m][qid]["totals"]]
            print(f"  [{zone}] 模型={ks}  Ariadne avg={avg(vals)}")

    payload = {"shared_non_abstain": len(shared_ids), "ariadne_story_vs_raw": deltas}
    out = ROOT / "eval/outputs/storyrank_ablation.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
