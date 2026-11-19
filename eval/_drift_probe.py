# -*- coding: utf-8 -*-
"""探针：这道题的查询向量把语义漂到哪去了（enemy q0000）

用法：python -u eval/_drift_probe.py
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

_load_dotenv(ROOT / ".env")

YAML = (ROOT / "eval/.cache/alm_data_r3/"
        "u_886f4595eb0ad67ff1d6d33535f90310ba1c58dfb77a64b31f00da9d66572c39.yaml")
PKL = ROOT / "eval/.cache/alm_r3_u_886f4595eb_vectors.pkl"

STEM = ("Taken together, Billing’s repeated oath-heavy outbursts most strongly suggest "
        "what underlying shift in his stance toward the Stockmanns over the course of the exchange?")
OPTS = (
    "A. He remains consistently supportive, using profanity mainly to signal camaraderie "
    "rather than disapproval.\n"
    "B. He moves from vague suspicion to firm moral condemnation, ending in a decision to "
    "distance himself socially from them.\n"
    "C. He shifts from being shocked by gossip to deciding he must privately reconcile with "
    "the Stockmanns to verify the truth.\n"
    "D. He is primarily angry at himself for being naive, and his final refusal is directed at "
    "his own habits rather than at the Stockmanns.\n"
    "E. He moves from incredulous disbelief to harsh moral condemnation, but his final outburst "
    "reflects only momentary frustration—he is still willing to keep up friendly social contact "
    "with the Stockmanns.\n"
    "F. He moves from initial skepticism to impressed approval of what he hears, and his last "
    "oath signals a renewed resolve to join the Stockmanns socially rather than distance himself."
)
TAIL = ("Please provide the option corresponding to the only correct answer, enclosed in "
        "parentheses, e.g., (X).")
FULL = f"{STEM}\n\n{OPTS}\n\n{TAIL}"

OPT = {k: v for k, v in zip("ABCDEF", OPTS.split("\n"))}

EVID_F = ["foremost man", "testimonial", "speak to Aslaksen", "good health", "Your health",
          "pagan", "bother the means", "Absolutely incredible", "country patient"]
EVID_OTHER = ["pest-house", "conduit-pipes", "roast beef", "report", "Baths"]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    graph, r = build_retriever(str(YAML), vector_cache_key="alm_r3_u_886f4595eb")
    vecs = pickle.load(open(PKL, "rb"))
    ids = [nid for nid in graph.graph.nodes() if nid in vecs]
    M = np.array([vecs[nid] for nid in ids], dtype=float)
    norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    contents = {nid: (graph.graph.nodes[nid].get("content") or "") for nid in ids}
    print(f"图节点 {graph.node_count}，有向量的 {len(ids)}")

    pcache = {}
    pf = ROOT / "eval/.cache/purposes_cache.json"
    if pf.exists():
        pcache = json.loads(pf.read_text(encoding="utf-8"))
    purposes = pcache.get(FULL) or pcache.get(STEM)
    if not purposes:
        purposes = (r.inference.infer_purpose(FULL).get("purposes") or [])
        pcache[FULL] = purposes
        pf.write_text(json.dumps(pcache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"目的（{len(purposes)} 条）: {purposes}")

    Q = {"FULL(平台实际下发)": FULL, "STEM(只题干)": STEM}
    for k in "ABCDEF":
        Q[f"OPT_{k}"] = OPT[k]
    Q["PURPOSE"] = " ".join(purposes)

    qv = {}
    for name, text in Q.items():
        v = np.asarray(r._embed(text), dtype=float)
        qv[name] = v

    print("\n== 查询变体两两余弦 ==")
    names = list(qv)
    print(f"{'':<20}" + "".join(f"{n[:10]:>12}" for n in names))
    for a in names:
        row = "".join(f"{cos(qv[a], qv[b]):>12.3f}" for b in names)
        print(f"{a:<20}{row}")

    pos = {nid: k for k, nid in enumerate(ids)}

    def report(name):
        v = qv[name]
        s = norm @ (v / (np.linalg.norm(v) + 1e-9))
        order = np.argsort(-s)
        rank = {ids[k]: r for r, k in enumerate(order)}
        print(f"\n===== {name}  top12 =====")
        for r, k in enumerate(order[:12]):
            print(f"  {r + 1:>3} {ids[k]:>5} cos={s[k]:.3f}  {contents[ids[k]][:88]}")
        print(f"  -- 分位: p50={np.percentile(s, 50):.3f} p90={np.percentile(s, 90):.3f} "
              f"p99={np.percentile(s, 99):.3f} max={s.max():.3f}")
        for label, keys in (("F 侧证据节点", EVID_F), ("多数派（Stockmann/发现）侧", EVID_OTHER)):
            print(f"  -- {label} --")
            for key in keys:
                hit = [nid for nid in ids if key.lower() in contents[nid].lower()][:3]
                for nid in hit:
                    print(f"     rank={rank[nid] + 1:>4}/{len(ids)} cos={s[pos[nid]]:.3f}  "
                          f"«{key}» {contents[nid][:70]}")
        return s

    s_full = report("FULL(平台实际下发)")
    s_stem = report("STEM(只题干)")
    report("PURPOSE")
    for k in "ABCDEF":
        report(f"OPT_{k}")

    pv = qv["PURPOSE"]
    pn = pv / (np.linalg.norm(pv) + 1e-9)
    ps = norm @ pn
    print("\n===== 目的向量回归（阈值 0.30 / 0.38）=====")
    for thr in (0.30, 0.38):
        keep = ps >= thr
        print(f"  thr={thr}: 通过 {int(keep.sum())}/{len(ids)}")
    for key in EVID_F:
        for nid in [nid for nid in ids if key.lower() in contents[nid].lower()]:
            p = ps[pos[nid]]
            flag = "留" if p >= 0.38 else ("留" if p >= 0.30 else "切")
            print(f"  {key:<22} purpose_cos={p:.3f} @0.30/0.38 → {flag}  {contents[nid][:60]}")


if __name__ == "__main__":
    main()
