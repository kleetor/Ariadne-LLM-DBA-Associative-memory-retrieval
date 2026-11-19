# -*- coding: utf-8 -*-
"""英文子集 Mem0 结果对比：qdrant(BM25) vs faiss(纯向量)，及与中文实验结果对照"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "outputs"


def load(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))


def main() -> None:
    q = load("mem0_candidates_en_qdrant.json")["results"]
    f = load("mem0_candidates_en_faiss.json")["results"]
    by_qid = {r["query_id"]: r for r in q}
    by_fid = {r["query_id"]: r for r in f}

    print("qid      exp    qdrant-hit          faiss-hit          排序一致(命中子集)")
    total = {"qdrant": 0, "faiss": 0, "exp_total": 0}
    for qid in by_qid:
        exp = set(by_qid[qid]["expected"])
        total["exp_total"] += len(exp)
        q_ids = [c["id"] for c in by_qid[qid]["candidates"]]
        f_ids = [c["id"] for c in by_fid[qid]["candidates"]]
        q_hit = set(q_ids) & exp
        f_hit = set(f_ids) & exp
        total["qdrant"] += len(q_hit)
        total["faiss"] += len(f_hit)
        q_rank = {i: n for n, i in enumerate(q_ids) if i in exp}
        f_rank = {i: n for n, i in enumerate(f_ids) if i in exp}
        same = all(q_rank.get(i) == f_rank.get(i) for i in exp)
        print(f"{qid:<6} {len(exp):<4} {str(sorted(q_hit)):<20} {str(sorted(f_hit)):<20} {same}")

    print(f"\n汇总: expected 覆盖 qdrant={total['qdrant']}/{total['exp_total']} "
          f"faiss={total['faiss']}/{total['exp_total']}")

    # 排序整体一致性（全部 20 候选逐条比对）
    same_rank = 0
    diff = []
    for qid in by_qid:
        q_ids = [c["id"] for c in by_qid[qid]["candidates"]]
        f_ids = [c["id"] for c in by_fid[qid]["candidates"]]
        if q_ids == f_ids:
            same_rank += 1
        else:
            diff.append(qid)
    print(f"top-20 候选逐条同序: {same_rank}/10 条；有差异: {diff or '无'}")

    # 展示一条 BM25 改变排序的实例
    for qid in ["s01", "s05", "s16"]:
        print(f"\n[{qid}] qdrant 前10 vs faiss 前10")
        q_ids = [c["id"] for c in by_qid[qid]["candidates"]][:10]
        f_ids = [c["id"] for c in by_fid[qid]["candidates"]][:10]
        print("  qdrant:", " ".join(q_ids))
        print("  faiss :", " ".join(f_ids))


if __name__ == "__main__":
    main()
