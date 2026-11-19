# -*- coding: utf-8 -*-
"""生成论文三张图（英文标签，PDF + 300dpi PNG）。

Fig 1  机制示意：目的池化成一个向量 vs 三项要求 + 两轴门控（含 E5 的类型分布）
Fig 2  E3 配对斜率图（70 条查询，三 judge，par_story vs vector_raw）
Fig 3  E10 (a) 分化度 AUC；(b) 跨主体混入率 × 输出规模

数据全部来自已固化产物，不重跑检索：
    eval/outputs/e5/inspect.json                     -> Fig 1 类型分布
    eval/outputs/e3/judge_e3_bandfix_*.json           -> Fig 2
    eval/outputs/e3/context_e3_bandfix.json           -> Fig 2 的 label_map
    eval/outputs/e10/probe.json                       -> Fig 3a
    eval/outputs/e10/filler.json                      -> Fig 3b

用法：python -m eval.make_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / "docs/theory/figs"

C_ACTION, C_THING, C_STATUS, C_EMOTION = "#4C78A8", "#F2A65A", "#6BA368", "#B279A2"
C_PAR, C_VP, C_VT = "#1F6F4A", "#E08A3C", "#B9BCC4"
C_GRAY, C_TEXT, C_GRID = "#8A8F98", "#22252A", "#E3E5E9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": "#B9BCC4",
    "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT/name}.pdf / .png")


def _rbox(ax, x, y, w, h, label, fc="#FFFFFF", ec="#B9BCC4", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=0.8, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- Fig 1
def fig1() -> None:
    e5 = json.load(open(ROOT / "eval/outputs/e5/inspect.json", encoding="utf-8"))
    dist = e5["type_dist"]                      # {intent: {type: share}}
    intents = [("vent", "comfort me"), ("plan", "plan my week"),
               ("recommend", "suggest options")]
    order = ["action", "thing", "status", "emotion"]
    colors = {"action": C_ACTION, "thing": C_THING,
              "status": C_STATUS, "emotion": C_EMOTION}

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.1, 9.75, "(a) One pooled vector loses \u201cwhat kind\u201d",
            fontsize=8.5, fontweight="bold", va="top")

    # 三条目的 -> 一个点（盒子宽度按最长标签留足，避免文字压到边框）
    for i, (_, en) in enumerate(intents):
        y = 7.55 - i * 1.35
        _rbox(ax, 0.15, y, 3.45, 0.9, en, fs=7.5)
        ax.annotate("", xy=(4.45, 6.98), xytext=(3.72, y + 0.45),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_GRAY,
                                    shrinkA=0, shrinkB=2))
    ax.add_patch(plt.Circle((4.85, 6.98), 0.33, color="#5A5F66", zorder=3))
    ax.text(5.32, 6.98, "one vector", ha="left", va="center", fontsize=7.5,
            color=C_GRAY)

    # E5 实测类型占比（四类归一化，四类合计约 93%）
    ax.text(0.1, 4.55, "retrieved node types (E5, n=30; four largest)",
            fontsize=7.5, color=C_GRAY, va="top")
    x0, width = 1.95, 7.6
    for i, (intent, en) in enumerate(intents):
        y = 3.45 - i * 0.72
        ax.text(1.82, y + 0.2, en.split()[0].capitalize(), ha="right",
                va="center", fontsize=7)
        left = x0
        vals = {t: float(dist[intent].get(t, 0.0)) for t in order}
        tot = sum(vals.values()) or 1.0
        for t in order:
            w = width * vals[t] / tot
            ax.add_patch(plt.Rectangle((left, y), w, 0.4, color=colors[t],
                                       linewidth=0))
            left += w
    # 图例（单行，只标类型名）
    lx = [1.95, 3.8, 5.65, 7.5]
    for j, t in enumerate(order):
        ax.add_patch(plt.Rectangle((lx[j], 1.02), 0.22, 0.22, color=colors[t], linewidth=0))
        ax.text(lx[j] + 0.32, 1.13, t, va="center", fontsize=6.8, color=C_GRAY)
    ax.text(0.1, 0.35, "three purposes, same mix (TVD 0.04\u20130.07)",
            fontsize=7, color=C_GRAY, va="bottom")

    # (b) 两轴门控
    bx = fig.add_subplot(gs[0, 1])
    bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
    bx.text(0.1, 9.75, "(b) Frame: two gated axes", fontsize=8.5,
            fontweight="bold", va="top")
    cells = [
        (1.35, 5.55, "type \u2713 \u00b7 topic \u2717", "dropped", False),
        (5.45, 5.55, "type \u2713 \u00b7 topic \u2713", "kept", True),
        (1.35, 3.15, "both low", "dropped", False),
        (5.45, 3.15, "topic \u2713 \u00b7 type \u2717", "dropped", False),
    ]
    for x, y, l1, l2, hi in cells:
        _rbox(bx, x, y, 3.7, 2.15, "", fc="#EAF3EC" if hi else "#F6F7F8",
              ec=C_PAR if hi else "#C9CCD2")
        bx.text(x + 1.85, y + 1.36, l1, ha="center", va="center", fontsize=7.6,
                fontweight="bold" if hi else "normal",
                color=C_PAR if hi else C_TEXT)
        bx.text(x + 1.85, y + 0.78, l2, ha="center", va="center", fontsize=6.8,
                color=C_GRAY)
    bx.annotate("", xy=(8.85, 2.85), xytext=(1.35, 2.85),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_GRAY))
    bx.text(5.1, 2.42, "topic similarity \u2192", ha="center", va="top",
            fontsize=7, color=C_GRAY)
    bx.annotate("", xy=(1.1, 7.9), xytext=(1.1, 2.95),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_GRAY))
    bx.text(0.85, 7.95, "type\naffinity", ha="right", va="center", fontsize=7,
            color=C_GRAY)
    bx.text(0.1, 1.55, "score = topic \u00d7 gate", fontsize=7.4, va="center")
    bx.text(0.1, 0.98, "gate = 0.55 + 0.35\u00b7type + 0.10\u00b7relation",
            fontsize=7.2, va="center")
    bx.text(0.1, 0.30, "additive scoring would also keep the top-left cell",
            fontsize=7, color=C_GRAY, va="center")

    _save(fig, "fig1_mechanism")


# ---------------------------------------------------------------- Fig 2
def _arm_scores(judge_path: Path, ctx_path: Path):
    ctx = json.load(open(ctx_path, encoding="utf-8"))
    lmap = {q["query_id"]: q.get("label_map") or {} for q in ctx["queries"]}
    data = json.load(open(judge_path, encoding="utf-8"))
    rows = []
    for r in data["results"]:
        lm = lmap.get(r["query_id"]) or {}
        inv = {v: k for k, v in lm.items()}
        if not inv:
            continue
        vals = {}
        for label, sc in r["scores"].items():
            arm = inv.get(label)
            if arm:
                vals[arm] = float(sc.get("total", 0))
        if "par_story" in vals and "vector_raw" in vals:
            rows.append((r["query_id"], vals["par_story"], vals["vector_raw"]))
    return rows


def fig2() -> None:
    judges = [("qwen3.8-flash", "judge_e3_bandfix_qwen.json"),
              ("deepseek-v4-flash", "judge_e3_bandfix_deepseek.json"),
              ("gpt-5.6-luna", "judge_e3_bandfix_gpt.json")]
    ctx = ROOT / "eval/outputs/e3/context_e3_bandfix.json"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    for ax, (name, fn) in zip(axes, judges):
        rows = _arm_scores(ROOT / "eval/outputs/e3" / fn, ctx)
        a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
        d = a - b
        for ya, yb in zip(a, b):
            ax.plot([0, 1], [ya, yb], color=C_GRAY, lw=0.5, alpha=0.22, zorder=1)
        ax.plot([0, 1], [a.mean(), b.mean()], color=C_PAR, lw=2.2, zorder=3)
        ax.scatter([0] * len(a), a, s=5, color=C_PAR, alpha=0.55, zorder=2)
        ax.scatter([1] * len(b), b, s=5, color=C_VP, alpha=0.55, zorder=2)
        ax.set_title(f"{name}  (+{d.mean():.2f}, {int((d > 0).sum())}/{len(d)})",
                     fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PAR", "vector"], fontsize=7.5)
        ax.set_xlim(-0.25, 1.25); ax.set_ylim(0.8, 5.25)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("rubric total (1\u20135)", fontsize=8)
    axes[0].set_yticks([1, 2, 3, 4, 5])
    fig.suptitle("E3: same graph, same assembler, only the selection differs",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_e3_paired")


# ---------------------------------------------------------------- Fig 3
def fig3() -> None:
    probe = json.load(open(ROOT / "eval/outputs/e10/probe.json", encoding="utf-8"))["graphs"]
    filler = json.load(open(ROOT / "eval/outputs/e10/filler.json", encoding="utf-8"))["graphs"]

    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    keys = ["L", "M", "H"]
    auc = [probe[k]["auc"] for k in keys]
    gap = [probe[k]["gap"] for k in keys]
    bars = ax.bar(keys, auc, width=0.55, color=["#C9CCD2", "#7FAF92", C_PAR])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC AUC (0.5 = chance)", fontsize=7.6)
    ax.set_xlabel("graph (163 / 253 / 343 nodes)", fontsize=7.6)
    ax.axhline(0.5, color=C_GRAY, lw=0.6, ls=":")
    for b, v, g in zip(bars, auc, gap):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=7)
        ax.text(b.get_x() + b.get_width() / 2, v + 0.062, f"\u0394cos {g:.3f}",
                ha="center", fontsize=6.2, color=C_GRAY)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=7.4)
    ax.set_title("(a) Separability saturates at 0.77", fontsize=8.5)
    ax.grid(axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    groups = [("M", "0.30"), ("M", "0.55"), ("H", "0.30"), ("H", "0.55")]
    arms = [("par", "PAR", C_PAR), ("vector_purpose", "vector+purpose", C_VP),
            ("vector_text", "vector (text only)", C_VT)]
    x = np.arange(len(groups)); w = 0.26
    for j, (arm, label, color) in enumerate(arms):
        vals = [float(filler[g]["by_tau"][t][arm]["off_ratio"]) for g, t in groups]
        ax2.bar(x + (j - 1) * w, vals, width=w, color=color, label=label)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax2.text(xi, v + 0.006, f"{v*100:.0f}", ha="center", fontsize=6.2)
    labels = [f"{g}, \u03c4={t}" for g, t in groups]
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.8)
    ax2.set_xlabel("mean output nodes: "
                   + " / ".join(f"{filler[g]['by_tau'][t]['par']['mean_nodes']:.1f}"
                                for g, t in groups), fontsize=6.8)
    ax2.set_ylabel("off-subject share of output", fontsize=7.6)
    ax2.set_ylim(0, 0.40)
    ax2.legend(fontsize=6.2, frameon=True, facecolor="white", edgecolor="none",
               loc="upper left", ncol=1, handlelength=1.1, borderpad=0.4)
    ax2.set_title("(b) Lower contamination comes with a smaller output", fontsize=8.5)
    ax2.grid(axis="y", color=C_GRID, lw=0.6); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig3_e10_tradeoff")


if __name__ == "__main__":
    print("[figs] ->", OUT)
    fig1()
    fig2()
    fig3()
    print("[figs] done")
