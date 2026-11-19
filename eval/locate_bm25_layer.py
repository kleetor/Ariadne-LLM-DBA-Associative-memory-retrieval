# -*- coding: utf-8 -*-
"""BM25 作用层定位：qdrant(混合) vs faiss(纯向量) 的 top-20 完整集合/顺序对比
回答：BM25 是否"只在排序层生效"？"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "outputs"


def load(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))["results"]


def main() -> None:
    q = {r["query_id"]: r for r in load("mem0_candidates_en_qdrant.json")}
    f = {r["query_id"]: r for r in load("mem0_candidates_en_faiss.json")}

    same_set = same_seq = 0
    print(f"{'qid':<6}{'集合一致':<8}{'顺序一致':<8}{'qdrant独有':<22}{'faiss独有':<22}")
    for qid in q:
        q_ids = [c["id"] for c in q[qid]["candidates"]]
        f_ids = [c["id"] for c in f[qid]["candidates"]]
        s_q, s_f = set(q_ids), set(f_ids)
        same_set += (s_q == s_f)
        same_seq += (q_ids == f_ids)
        only_q = sorted(s_q - s_f)
        only_f = sorted(s_f - s_q)
        print(f"{qid:<6}{str(s_q==s_f):<8}{str(q_ids==f_ids):<8}{','.join(only_q)[:20]:<22}{','.join(only_f)[:20]:<22}")

    n = len(q)
    print(f"\ntop-20 完整集合一致: {same_set}/{n}；顺序一致: {same_seq}/{n}")

    # 平均 Jaccard（集合相似度）
    js = [len(set([c['id'] for c in q[i]['candidates']]) & set([c['id'] for c in f[i]['candidates']]))
          / len(set([c['id'] for c in q[i]['candidates']]) | set([c['id'] for c in f[i]['candidates']]))
          for i in q]
    print(f"top-20 平均 Jaccard 相似度: {sum(js)/len(js):.3f}")


if __name__ == "__main__":
    main()
