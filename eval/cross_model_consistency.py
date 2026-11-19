# -*- coding: utf-8 -*-
"""
综合能力测评 · judge 交叉一致性（§8.2 独立模型交叉验证）

对同一批上下文块，用多套 judge 模型各自判分，计算模型间一致性：
1. 系统排位一致率：每题"哪系统 total 最高"（top-1 归属）两两一致率
2. Cohen kappa：基于 top-1 类别（三系统 + 平局）评分者间一致度
3. Spearman 秩相关：三系统 total 分数（逐题）的秩相关

输入：judge_comprehensive.json（主判分 qwen）+ judge_gemini.json / judge_gpt.json（交叉判定）
用法：
    python -m eval.cross_model_consistency
"""
import json
import itertools
import sys
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FILES = {
    "qwen3.8-flash": "eval/outputs/judge_comprehensive.json",
    "deepseek-v4-flash": "eval/outputs/judge_deepseek.json",
    "gemini-3.7-flash": "eval/outputs/judge_gemini.json",
    "gpt-5.6-luna": "eval/outputs/judge_gpt.json",
    "glm-4-flash": "eval/outputs/judge_glm.json",
}
# 带区域标注：zh=中文/国产模型, en=国外模型（用于分析语言/文化偏差）
REGION = {
    "qwen3.8-flash": "zh",
    "deepseek-v4-flash": "zh",
    "gemini-3.7-flash": "en",
    "gpt-5.6-luna": "en",
    "glm-4-flash": "zh",
}
SYSTEMS = ["Ariadne", "Mem0", "Graphiti"]
ABSTAIN_CAT = "abstention"

# 判分病态模型（评分通胀/形态偏好），仅作案例披露，不计入有效区域对照
INVALID_MODELS = {"glm-4-flash"}


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d["results"]:
        sc = r.get("scores", {})
        out[r["query_id"]] = {
            "category": r["category"],
            "totals": {s: sc[s]["total"] for s in SYSTEMS if sc.get(s)},
        }
    return out


def top1(totals: dict) -> str:
    """返回 total 最高的系统；平局且相同则返回 'TIE'"""
    if not totals:
        return "NA"
    mx = max(totals.values())
    leaders = [s for s, v in totals.items() if v == mx]
    return leaders[0] if len(leaders) == 1 else "TIE"


def main() -> None:
    # 加载三模型判分
    models = {name: load(ROOT / p) for name, p in FILES.items()}
    print("加载判分:", {k: len(v) for k, v in models.items()})

    # 只取三模型共有的查询（避免弃权分歧），并保证非弃权
    shared_ids = set.intersection(*(set(m.keys()) for m in models.values()))
    # 用主判分的 category 判断是否弃权
    shared_ids = [qid for qid in shared_ids if models["qwen3.8-flash"][qid]["category"] != ABSTAIN_CAT]
    print(f"共同非弃权查询: {len(shared_ids)}")

    print("\n===== 各模型系统总均分（非弃权，三模型全展示） =====")
    for name, m in models.items():
        avg = {}
        for s in SYSTEMS:
            vals = [m[qid]["totals"][s] for qid in shared_ids if s in m[qid]["totals"]]
            avg[s] = round(sum(vals) / len(vals), 2) if vals else None
        vals_avail = [s for s in SYSTEMS if avg.get(s) is not None]
        order = " > ".join(s for s in sorted(vals_avail, key=lambda s: -avg[s]))
        print(f"  {name:<16} {avg}   排位: {order}")

    # top-1 归属 + kappa
    print("\n===== 两两一致性 ===")
    for a, b in itertools.combinations(models.keys(), 2):
        ta = [top1(models[a][qid]["totals"]) for qid in shared_ids]
        tb = [top1(models[b][qid]["totals"]) for qid in shared_ids]
        agree = sum(1 for x, y in zip(ta, tb) if x == y)
        rate = agree / len(shared_ids)
        try:
            kappa = cohen_kappa_score(ta, tb)
        except Exception:
            kappa = float("nan")
        # Spearman：三系统分数并排成向量后算秩相关
        va, vb = [], []
        for qid in shared_ids:
            for s in SYSTEMS:
                if s in models[a][qid]["totals"] and s in models[b][qid]["totals"]:
                    va.append(models[a][qid]["totals"][s])
                    vb.append(models[b][qid]["totals"][s])
        rho, pval = spearmanr(va, vb) if len(va) > 2 else (float("nan"), float("nan"))
        print(f"\n  {a}  vs  {b}")
        print(f"    top-1 一致率: {rate:.3f} ({agree}/{len(shared_ids)})")
        print(f"    Cohen kappa (top-1 类别): {kappa:+.3f}")
        print(f"    Spearman 秩相关 (三系统分数): {rho:+.3f} (p={pval:.3f})")

    # 区域（语言/文化）偏差分析：中文/国产模型组 vs 国外模型组
    # 排除病态模型（评分通胀，如 glm-4-flash 对 Graphiti 形态无差别满分）：
    # 它会让"zh 组"失真，故仅作案例披露，不计入有效区域对照。
    VALID = [k for k in models if k not in INVALID_MODELS]
    print(f"有效对照模型={VALID}  排除病态={[k for k in models if k in INVALID_MODELS]}")
    print("\n===== 区域（语言/文化）分组分析（仅有效模型） =====")
    zh = [k for k in VALID if REGION.get(k) == "zh"]
    en = [k for k in VALID if REGION.get(k) == "en"]
    zones = {}
    for zone, ks in (("zh", zh), ("en", en)):
        agg = {}
        for s in SYSTEMS:
            vals = []
            for qid in shared_ids:
                for k in ks:
                    if s in models[k][qid]["totals"]:
                        vals.append(models[k][qid]["totals"][s])
            agg[s] = round(sum(vals) / len(vals), 2) if vals else None
        avg_all = sum(v for v in agg.values() if v is not None) / len([v for v in agg.values() if v is not None])
        order = " > ".join(s for s in sorted([s for s in SYSTEMS if agg.get(s) is not None],
                                             key=lambda s: -agg[s]))
        print(f"  [{zone}] 模型={ks}  系统总均分={agg}  (三系统均值={avg_all:.2f})")
        print(f"      排位: {order}")
        zones[zone] = agg
        # 组内两两一致率（如 zh 有两个模型、en 有两个模型）
        if len(ks) > 1:
            for a, b in itertools.combinations(ks, 2):
                ta = [top1(models[a][qid]["totals"]) for qid in shared_ids]
                tb = [top1(models[b][qid]["totals"]) for qid in shared_ids]
                agree = sum(1 for x, y in zip(ta, tb) if x == y)
                print(f"      内部一致 {a} vs {b}: {agree/len(shared_ids):.3f} ({agree}/{len(shared_ids)})")

    # 跨区域差异：zh 组均值 vs en 组均值（对每个系统）
    print("\n===== 中文组 vs 国外组（系统均分差 zh-en） =====")
    for s in SYSTEMS:
        z = zones.get("zh", {}).get(s)
        e = zones.get("en", {}).get(s)
        if z is not None and e is not None:
            d = z - e
            print(f"  {s:<10} zh={z:>5.2f}  en={e:>5.2f}  差={d:+.2f}  {'→ 中文评分偏高' if d>0 else ('→ 国外评分偏高' if d<0 else '→ 一致')}")

    # 落盘
    payload = {"method": "cross_model_consistency", "shared_non_abstain": len(shared_ids),
               "models_regions": REGION,
               "zone_avg": {z: zones.get(z, {}) for z in ("zh", "en")},
               "pairwise": {}}
    for a, b in itertools.combinations(models.keys(), 2):
        ta = [top1(models[a][qid]["totals"]) for qid in shared_ids]
        tb = [top1(models[b][qid]["totals"]) for qid in shared_ids]
        agree = sum(1 for x, y in zip(ta, tb) if x == y)
        va, vb = [], []
        for qid in shared_ids:
            for s in SYSTEMS:
                if s in models[a][qid]["totals"] and s in models[b][qid]["totals"]:
                    va.append(models[a][qid]["totals"][s])
                    vb.append(models[b][qid]["totals"][s])
        rho, _ = spearmanr(va, vb) if len(va) > 2 else (float("nan"), float("nan"))
        kappa = cohen_kappa_score(ta, tb)
        payload["pairwise"][f"{a}__vs__{b}"] = {
            "top1_agree_rate": round(agree / len(shared_ids), 3),
            "cohen_kappa": round(float(kappa), 3),
            "spearman_rho": round(float(rho), 3),
            "n": len(shared_ids),
        }
    out = ROOT / "eval/outputs/cross_model_consistency.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
