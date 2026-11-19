# -*- coding: utf-8 -*-
"""E6-lite 问卷生成：从 E3 三臂产物中分层抽 50 条，生成两份盲评 HTML + 分析用盲表。

设计：
- 层 = 8 个非弃权能力维度，按 50/70 比例分配（single_fact 10、preference 6、
  multi_hop 6、knowledge_update 4、temporal 7、cross_scene 5、assistant_behavior 6、
  mixed_memory 6）；
- 每条从 par_story / par_raw / vector_raw 中随机取一臂，三臂总量 17/17/16；
- 问卷不显示系统名、条件名、judge 分数；只给 item_id、用户原话、记忆上下文；
- 两位标注者使用不同题序（A/B），但 item_id 相同，便于计算 α。

输出（eval/outputs/e6_lite/）：
- e6lite_raterA.html / e6lite_raterB.html：自包含问卷，可导出 CSV/JSON；
- e6lite_key.json：item_id → query_id / category / arm / 三家 judge 分数（**不要给标注者**）；
- e6lite_blank_A.csv / e6lite_blank_B.csv：离线填写模板。

用法：python -m eval.e6_lite_build
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
CTX = "eval/outputs/e3/context_e3_bandfix.json"
JUDGES = {
    "qwen": "eval/outputs/e3/judge_e3_bandfix_qwen.json",
    "deepseek": "eval/outputs/e3/judge_e3_bandfix_deepseek.json",
    "gpt": "eval/outputs/e3/judge_e3_bandfix_gpt.json",
}
DIMS = ("记忆丰富度", "易理解性", "用户建模度")
N_ITEMS = 50
SEED = 20261119
ARM_POOL = ["par_story"] * 17 + ["par_raw"] * 17 + ["vector_raw"] * 16


def _load_json(path: str) -> Dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return json.load(f)


def _render_block(block: Dict[str, Any]) -> tuple[str, str]:
    """返回 (html, plain_text)。raw 块把关系折叠，减少阅读负担。"""
    if block.get("story"):
        story = html.escape(str(block["story"]))
        return f'<div class="story">{story}</div>', str(block["story"])
    nodes = block.get("nodes") or []
    edges = block.get("edges") or []
    node_html = "".join(f"<li>{html.escape(str(n))}</li>" for n in nodes)
    edge_html = "".join(f"<li>{html.escape(str(e))}</li>" for e in edges)
    story_html = (
        f'<div class="meta">记忆条目（{len(nodes)} 条）</div>'
        f'<ul class="nodes">{node_html}</ul>'
        f'<details><summary>关系（{len(edges)} 条，点击展开）</summary>'
        f'<ul class="edges">{edge_html}</ul></details>'
    )
    plain = "\n".join([f"- {n}" for n in nodes] + [f"  {e}" for e in edges])
    return story_html, plain


def _alloc(categories: List[str], counts: Dict[str, int], n: int) -> Dict[str, int]:
    """每个能力维度先取 6 条（共 48），余下 2 条给剩余最多的大类。

    严格按计划的「每层 5–6 条」做近似：8 个非弃权层各 6 条，再给 single_fact /
    temporal 各 +1。弃权类不在 E3 主判分里，没有对应块，故不纳入。
    """
    out = {c: min(6, counts[c]) for c in categories}
    rest = n - sum(out.values())
    order = sorted(categories, key=lambda c: (-(counts[c] - out[c]), -counts[c], c))
    for c in order[: max(0, rest)]:
        if out[c] < counts[c]:
            out[c] += 1
    return out


def _html_page(items: List[Dict[str, Any]], order: List[str], rater: str) -> str:
    cards = []
    for idx, iid in enumerate(order, 1):
        it = next(x for x in items if x["item_id"] == iid)
        radios = []
        for dim in DIMS:
            opts = "".join(
                f'<label><input type="radio" name="{iid}_{dim}" value="{v}"> {v}</label>'
                for v in range(1, 6)
            )
            radios.append(f'<div class="dim"><span>{dim}</span><div class="opts">{opts}</div></div>')
        cards.append(f"""
<section class="card" data-item="{iid}">
  <div class="card-head"><span class="iid">{iid}</span>
    <span class="qidx">第 {idx} / {len(order)} 题</span></div>
  <div class="query"><b>用户原话：</b>{html.escape(it['query'])}</div>
  <div class="block">{it['block_html']}</div>
  <div class="ratings">{''.join(radios)}</div>
  <textarea class="note" placeholder="可选：本条第 X 维为什么给这个分（可不填）"></textarea>
</section>""")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E6-lite 人工盲评问卷 · 标注者 {rater}</title>
<style>
:root{{--line:#e3e6ea;--muted:#6b7280;--bg:#f6f7f9;--accent:#3730a3;--ok:#047857}}
*{{box-sizing:border-box}} body{{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
margin:0;background:var(--bg);color:#1c1e21;line-height:1.6}}
header{{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line);
padding:14px 22px;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
header h1{{font-size:18px;margin:0 0 4px}} header .sub{{font-size:12px;color:var(--muted)}}
.wrap{{max-width:980px;margin:18px auto 80px;padding:0 18px}}
.intro{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}}
.intro h2{{font-size:15px;margin:0 0 8px}} .intro p,.intro li{{font-size:13px;margin:4px 0}}
.intro code{{background:#eef2ff;padding:1px 5px;border-radius:4px}}
.rubric{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}}
.rubric div{{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:10px}}
.rubric b{{font-size:13px}} .rubric p{{font-size:12px;color:#374151;margin:4px 0 0}}
.raterbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}}
.raterbar input{{width:90px;padding:6px 8px;border:1px solid var(--line);border-radius:7px}}
.progress{{font-size:12px;color:var(--muted)}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}}
.card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.iid{{display:inline-block;background:var(--accent);color:#fff;border-radius:999px;padding:2px 10px;
font-size:12px;font-weight:600;letter-spacing:.5px}} .qidx{{font-size:12px;color:var(--muted)}}
.query{{font-size:14px;margin:6px 0 10px;font-weight:600}}
.block{{background:#fbfbfc;border:1px dashed var(--line);border-radius:9px;padding:10px 12px;
font-size:13px;margin-bottom:10px}} .story{{white-space:pre-wrap}}
.meta{{font-size:12px;color:var(--muted);margin-bottom:4px}}
.nodes,.edges{{margin:4px 0 0 18px;padding:0}} .nodes li,.edges li{{margin:2px 0}}
details{{margin-top:6px;font-size:12px;color:#4b5563}}
.dim{{display:flex;gap:10px;align-items:center;margin:7px 0;flex-wrap:wrap}}
.dim>span{{width:100px;font-size:13px;font-weight:600}}
.opts{{display:flex;gap:4px;flex-wrap:wrap}}
.opts label{{border:1px solid var(--line);border-radius:7px;padding:4px 10px;background:#fff;
font-size:13px;cursor:pointer}} .opts label:has(input:checked){{background:#eef2ff;border-color:#6366f1;
color:#3730a3;font-weight:600}}
textarea.note{{width:100%;min-height:42px;border:1px solid var(--line);border-radius:8px;padding:7px;
font-size:12px;resize:vertical}}
footer{{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid var(--line);
padding:10px 18px;display:flex;gap:10px;align-items:center;justify-content:flex-end;z-index:9}}
button{{border:0;border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer}}
#export{{background:var(--accent);color:#fff}} #clear{{background:#eef2ff;color:#3730a3}}
#msg{{font-size:12px;color:var(--muted);margin-right:auto}}
@media(max-width:700px){{.rubric{{grid-template-columns:1fr}}}}
</style></head><body>
<header>
  <h1>E6-lite 人工盲评问卷 · 标注者 {rater}</h1>
  <div class="sub">共 {len(order)} 题；三维独立评分；所有内容均为匿名上下文块。填写会自动保存在本机浏览器。</div>
</header>
<div class="wrap">
  <div class="intro">
    <h2>怎么评</h2>
    <p>下面每一题都是：<b>一句用户原话</b> + <b>某个记忆系统为这句话检索/组装的上下文</b>。
    请只根据“这段上下文对当前这句话是否好用”打分，不要猜测它来自哪个系统。</p>
    <div class="rubric">
      <div><b>记忆丰富度</b><p>是否覆盖了理解用户所需的状态、经历、偏好、关系、约束。5=非常丰富，1=单薄贫乏。</p></div>
      <div><b>易理解性</b><p>是否条理清晰、能直接拿来用。5=一目了然，1=晦涩零乱。</p></div>
      <div><b>用户建模度</b><p>能否据此准确理解用户、形成用户画像。5=很强，1=很弱。</p></div>
    </div>
    <p>评分用 1–5 的整数；三个维度都要填。若某维确实无法判断，可留空，但请尽量填。</p>
    <p><b>盲评规则：</b>不要与其他标注者讨论具体题目；不要参考模型输出；按你自己的判断打。</p>
    <div class="raterbar">
      <label>标注者 ID：<input id="rater" value="{rater}"></label>
      <span class="progress" id="progress">已填 0 / {len(order)} 题</span>
      <span id="msg"></span>
    </div>
  </div>
  {''.join(cards)}
</div>
<footer>
  <button id="clear">清空本题卷</button>
  <button id="export">导出 CSV</button>
  <button id="exportJson">导出 JSON</button>
</footer>
<script>
const DIMS = {json.dumps(list(DIMS), ensure_ascii=False)};
const STORE_KEY = 'e6lite_{rater}';
function val(item, dim) {{
  const el = document.querySelector(`input[name="${{item}}_${{dim}}"]:checked`);
  return el ? el.value : '';
}}
function note(item) {{
  const card = document.querySelector(`.card[data-item="${{item}}"]`);
  return card ? card.querySelector('textarea.note').value : '';
}}
function updateProgress() {{
  let done = 0;
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.item;
    if (DIMS.every(d => val(id, d))) done++;
  }});
  document.getElementById('progress').textContent = `已填 ${{done}} / {len(order)} 题`;
}}
function save() {{
  const data = {{}};
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.item;
    data[id] = {{}};
    DIMS.forEach(d => data[id][d] = val(id, d));
    data[id].note = note(id);
  }});
  localStorage.setItem(STORE_KEY, JSON.stringify({{rater: document.getElementById('rater').value, data}}));
}}
function restore() {{
  const raw = localStorage.getItem(STORE_KEY);
  if (!raw) return;
  try {{
    const {{
      rater, data
    }} = JSON.parse(raw);
    if (rater) document.getElementById('rater').value = rater;
    Object.entries(data).forEach(([id, rec]) => {{
      DIMS.forEach(d => {{
        if (rec[d]) {{
          const el = document.querySelector(`input[name="${{id}}_${{d}}"][value="${{rec[d]}}"]`);
          if (el) el.checked = true;
        }}
      }});
      const card = document.querySelector(`.card[data-item="${{id}}"]`);
      if (card && rec.note) card.querySelector('textarea.note').value = rec.note;
    }});
  }} catch (e) {{}}
  updateProgress();
}}
function rows() {{
  const out = [];
  const rid = document.getElementById('rater').value || '{rater}';
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.item;
    out.push([rid, id, ...DIMS.map(d => val(id, d)), note(id)]);
  }});
  return out;
}}
function csvEscape(s) {{
  s = String(s ?? '');
  return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}}
function download(name, text, type) {{
  const blob = new Blob([text], {{type}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}}
document.getElementById('export').onclick = () => {{
  const r = rows();
  const missing = r.filter(x => x.slice(2, 5).some(v => !v)).length;
  if (missing && !confirm(`还有 ${{missing}} 题未填完，仍然导出？`)) return;
  const header = ['rater_id', 'item_id', ...DIMS, 'note'];
  const csv = '\\ufeff' + [header, ...r].map(row => row.map(csvEscape).join(',')).join('\\n');
  download(`e6lite_${{document.getElementById('rater').value || '{rater}'}}.csv`, csv, 'text/csv;charset=utf-8');
  document.getElementById('msg').textContent = '已导出 CSV';
}};
document.getElementById('exportJson').onclick = () => {{
  const rec = {{rater: document.getElementById('rater').value || '{rater}', rows: rows()}};
  download(`e6lite_${{rec.rater}}.json`, JSON.stringify(rec, null, 2), 'application/json');
}};
document.getElementById('clear').onclick = () => {{
  if (!confirm('确定清空本题卷的本地记录？')) return;
  localStorage.removeItem(STORE_KEY); location.reload();
}};
document.addEventListener('change', () => {{ save(); updateProgress(); }});
document.addEventListener('input', () => save());
restore();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="E6-lite 问卷生成")
    ap.add_argument("--ctx", default=CTX)
    ap.add_argument("--out-dir", default="eval/outputs/e6_lite")
    ap.add_argument("--n", type=int, default=N_ITEMS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    ctx = _load_json(args.ctx)
    rows = ctx["queries"]
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_cat[str(r["category"])].append(r)
    categories = sorted(by_cat)
    counts = {c: len(by_cat[c]) for c in categories}
    alloc = _alloc(categories, counts, args.n)

    rng = random.Random(args.seed)
    sampled: List[Dict] = []
    for c in categories:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        sampled.extend(pool[: alloc[c]])
    rng.shuffle(sampled)

    # 三臂平衡
    arms = list(ARM_POOL[: args.n])
    rng.shuffle(arms)

    # 三家 judge 的分数：{qid: {arm: {dim: score, total}}}
    judge_scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for name, path in JUDGES.items():
        data = _load_json(path)
        per_q: Dict[str, Dict[str, Dict[str, float]]] = {}
        for r in data["results"]:
            lm = r.get("label_map") or {}
            if not lm:
                # judge 行里没有 label_map 时，用 ctx 的映射
                lm = next((x["label_map"] for x in rows if x["query_id"] == r["query_id"]), {})
            per_q[r["query_id"]] = {
                arm: r["scores"][sys_name] for arm, sys_name in lm.items()
                if sys_name in r["scores"]
            }
        judge_scores[name] = per_q

    items: List[Dict[str, Any]] = []
    for i, (r, arm) in enumerate(zip(sampled, arms), 1):
        lm = r["label_map"]
        sys_name = lm[arm]
        block = r["systems"][sys_name]
        block_html, block_text = _render_block(block)
        iid = f"E{i:02d}"
        items.append({
            "item_id": iid,
            "query_id": r["query_id"],
            "category": r["category"],
            "arm": arm,
            "query": r["query"],
            "block_html": block_html,
            "block_text": block_text,
            "judges": {name: judge_scores[name].get(r["query_id"], {}).get(arm)
                       for name in JUDGES},
        })

    ids = [it["item_id"] for it in items]
    order_a = list(ids)
    random.Random(args.seed + 101).shuffle(order_a)
    order_b = list(ids)
    random.Random(args.seed + 202).shuffle(order_b)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "e6lite_raterA.html").write_text(_html_page(items, order_a, "A"), encoding="utf-8")
    (out / "e6lite_raterB.html").write_text(_html_page(items, order_b, "B"), encoding="utf-8")

    key = {
        "meta": {
            "experiment": "E6-lite",
            "ctx": args.ctx,
            "n_items": len(items),
            "seed": args.seed,
            "rubric": list(DIMS),
            "categories": {c: alloc[c] for c in categories},
            "arms": dict(Counter(it["arm"] for it in items)),
            "note": "盲表：不要给标注者。judges 为三家 LLM 对该 item 对应块的原始分数。",
        },
        "orders": {"A": order_a, "B": order_b},
        "items": items,
    }
    (out / "e6lite_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    for rater, order in (("A", order_a), ("B", order_b)):
        path = out / f"e6lite_blank_{rater}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rater_id", "item_id", *DIMS, "note"])
            for iid in order:
                w.writerow([rater, iid, "", "", "", ""])

    print(f"[E6-lite] items={len(items)} categories={ {c: alloc[c] for c in categories} }")
    print(f"[E6-lite] arms={dict(Counter(it['arm'] for it in items))}")
    print(f"[E6-lite] wrote {out / 'e6lite_raterA.html'}")
    print(f"[E6-lite] wrote {out / 'e6lite_raterB.html'}")
    print(f"[E6-lite] wrote {out / 'e6lite_key.json'}（仅供分析，勿发给标注者）")


if __name__ == "__main__":
    main()
