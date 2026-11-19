# -*- coding: utf-8 -*-
"""节点写法对可分性的影响：原句 vs 扩写 vs 口语化（角色扮演口吻）。

同一批节点、同一场景标签，只改文字写法，比较「同场景 vs 跨场景」的余弦落差与 AUC。
目的：验证"数据同质"是否主要来自节点太短（12 字），而不是场景设计。

用法：python -m eval.probe_node_style [--n 40]
输出：eval/outputs/node_style_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

PROMPT = """下面是若干条"记忆节点"（id + 内容）。请为每条生成两个改写版本：
- expanded：扩写到 25–35 个汉字，补充时间、对象、数量、地点等细节，**不改变事实**；
- colloquial：改写成第一人称口语化的自述（像角色扮演里人物讲自己的事），保留事实，
  语气自然、可带一点情绪，20–35 字。

只输出 JSON：{{"items":[{{"id":"uw1","expanded":"...","colloquial":"..."}}]}}

节点：
{items}
"""


def _sample(nodes, per_scene=6, seed=19):
    by = {}
    for nid, c in nodes:
        sc = nid[1] if len(nid) > 1 else "?"
        by.setdefault(sc, []).append((nid, c))
    rng = random.Random(seed)
    out = []
    for sc, arr in sorted(by.items()):
        rng.shuffle(arr)
        out += arr[:per_scene]
    return out


def _rewrite(llm, batch):
    payload = "\n".join(f"{nid}: {c}" for nid, c in batch)
    resp = llm.invoke(PROMPT.format(items=payload))
    txt = getattr(resp, "content", "") or ""
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return {}
    try:
        items = json.loads(m.group(0)).get("items") or []
    except Exception:
        return {}
    return {it["id"]: it for it in items if it.get("id")}


def _stats(mat, scenes):
    iu = np.triu_indices(len(scenes), k=1)
    c = mat[iu]
    sc = np.array(scenes)
    same = sc[iu[0]] == sc[iu[1]]
    w, a = c[same], c[~same]
    try:
        auc = roc_auc_score(same.astype(int), c)
    except Exception:
        auc = float("nan")
    return {
        "within": round(float(w.mean()), 3),
        "across": round(float(a.mean()), 3),
        "gap": round(float(w.mean() - a.mean()), 3),
        "auc": round(float(auc), 3),
        "p50": round(float(np.percentile(c, 50)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="每场景取几条（0=每场景 6 条）")
    ap.add_argument("--out", default="eval/outputs/node_style_probe.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever(str(ROOT / "data/dba_eval/comprehensive_gold.yaml"),
                                       vector_cache_key="comprehensive")
    nodes = [(nid, graph.graph.nodes[nid]["content"]) for nid in graph.graph.nodes()
             if graph.graph.nodes[nid].get("content")]
    sample = _sample(nodes, per_scene=args.n or 6)
    print(f"sample={len(sample)} scenes="
          f"{ {nid[1]: sum(1 for x,_ in sample if x[1]==nid[1]) for nid,_ in sample} }", flush=True)

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    )

    rewritten = {}
    for i in range(0, len(sample), 8):
        batch = sample[i:i + 8]
        got = _rewrite(llm, batch)
        rewritten.update(got)
        print(f"  rewrote {len(rewritten)}/{len(sample)}", flush=True)

    keep = [(nid, c) for nid, c in sample if nid in rewritten]
    scenes = [nid[1] for nid, _ in keep]
    emb = retriever._embed
    variants = {
        "original": [c for _, c in keep],
        "expanded": [rewritten[nid]["expanded"] for nid, _ in keep],
        "colloquial": [rewritten[nid]["colloquial"] for nid, _ in keep],
    }
    report = {}
    for name, texts in variants.items():
        M = np.array([emb(t) for t in texts], dtype="float32")
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        report[name] = {**_stats(M @ M.T, scenes),
                        "len_mean": round(statistics.mean(len(t) for t in texts), 1)}

    print("\n=== 同一批节点，只改写法 ===")
    print(f"{'写法':12s} {'within':>7s} {'across':>7s} {'gap':>7s} {'AUC':>6s} {'p50':>6s} {'字数':>5s}")
    for k, v in report.items():
        print(f"{k:12s} {v['within']:7.3f} {v['across']:7.3f} {v['gap']:+7.3f} "
              f"{v['auc']:6.3f} {v['p50']:6.3f} {v['len_mean']:5.1f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"embedding": os.environ.get("EMBEDDING_MODEL"),
                                      "n_nodes": len(keep)},
                             "variants": report,
                             "samples": [{"id": nid, **rewritten[nid]} for nid, _ in keep]},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[wrote] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
