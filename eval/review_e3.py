# -*- coding: utf-8 -*-
"""把 E3 的 2×2 产物导出成可人工复核的单页 HTML（PAR+story vs 向量+story）。

输入：context_e3_2x2_noedge.json（A=par_story, B=vector_story, C=par_raw, D=vector_raw）
      judge_e3_shuf_qwen.json / judge_e3_shuf_gemini.json（逐题打乱，按 label_map 还原）
输出：eval/outputs/e3/review_e3.html

用法:
    python -m eval.review_e3
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTX = "eval/outputs/e3/context_e3_2x2_noedge.json"
SHUF = "eval/outputs/e3/context_e3_2x2_noedge_shuf.json"
JUDGES = {
    "qwen": "eval/outputs/e3/judge_e3_shuf_qwen.json",
    "gemini": "eval/outputs/e3/judge_e3_shuf_gemini.json",
}
OUT = "eval/outputs/e3/review_e3.html"

STYLE = """
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;margin:24px;background:#f6f7f9;color:#1c1e21}
h1{font-size:20px}.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 16px;margin:14px 0}
h2{font-size:15px;margin:0 0 6px}h2 small{color:#6b7280;font-weight:400}
h3{font-size:13px;margin:0 0 6px;color:#374151}
.q{margin:0 0 10px;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.story{font-size:13px;line-height:1.65;margin:6px 0 8px;white-space:pre-wrap}
.chip{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:1px 8px;font-size:12px;margin-right:6px}
details{font-size:12px;color:#4b5563}li{margin:1px 0}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _scores(shuf_maps, judge_path):
    with open(ROOT / judge_path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data["results"]:
        lm = r.get("label_map") or shuf_maps.get(r["query_id"], {})
        inv = {lab: cond for cond, lab in lm.items()}
        out[r["query_id"]] = {inv.get(k, k): v for k, v in r["scores"].items()}
    return out


def main() -> None:
    with open(ROOT / CTX, encoding="utf-8") as f:
        ctx = json.load(f)
    with open(ROOT / SHUF, encoding="utf-8") as f:
        shuf_maps = {r["query_id"]: r["label_map"] for r in json.load(f)["queries"]}
    judges = {name: _scores(shuf_maps, path) for name, path in JUDGES.items()}

    cards = []
    for r in ctx["queries"]:
        qid, st = r["query_id"], r["stats"]

        def chip(cond: str) -> str:
            return "".join(
                f'<span class="chip">{name} '
                f'{judges[name].get(qid, {}).get(cond, {}).get("total", "-")}</span>'
                for name in judges
            )

        def nodes(blk) -> str:
            return "".join(f"<li>{_esc(n)}</li>" for n in blk.get("nodes", []))

        cards.append(f"""
<section class="card">
  <h2>{qid} <small>{_esc(r["category"])}</small></h2>
  <p class="q">{_esc(r["query"])}</p>
  <div class="grid">
    <div class="col">
      <h3>PAR + story · {st["par_nodes"]}节点/{st["par_edges"]}边 · 因果边 {st["par_causal_edges"]}</h3>
      {chip("par_story")}
      <p class="story">{_esc(r["systems"]["系统A"]["story"])}</p>
      <details><summary>PAR 选中节点</summary><ul>{nodes(r["systems"]["系统C"])}</ul></details>
    </div>
    <div class="col">
      <h3>向量 + story · {st["vec_nodes"]}节点/0边 · 因果边 0</h3>
      {chip("vector_story")}
      <p class="story">{_esc(r["systems"]["系统B"]["story"])}</p>
      <details><summary>向量选中节点</summary><ul>{nodes(r["systems"]["系统D"])}</ul></details>
    </div>
  </div>
</section>""")

    page = (
        '<!doctype html><meta charset="utf-8"><title>E3 人工复核</title>'
        f"<style>{STYLE}</style>"
        "<h1>E3 人工复核：PAR+story vs 向量+story（无边纯向量，逐题打乱判分）</h1>"
        f'<p style="font-size:13px;color:#6b7280">共 {len(cards)} 条；分数为 total（1–5），'
        "点开可看各自选中的节点。</p>" + "".join(cards)
    )
    out = ROOT / OUT
    out.write_text(page, encoding="utf-8")
    print(f"[review] wrote {out}（{len(cards)} 条）")


if __name__ == "__main__":
    main()
