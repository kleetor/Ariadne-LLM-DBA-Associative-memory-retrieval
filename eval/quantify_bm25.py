# -*- coding: utf-8 -*-
"""BM25 改变方向与程度量化（英文子集实验）

用 embedding API 计算查询与候选的语义相似度，对比 qdrant(混合) vs faiss(纯向量)：
- 集合层面：qdrant 独有节点 vs faiss 独有节点的平均相似度 → BM25 带入/挤出的是更相关还是更不相关
- 排序层面：同节点在 qdrant/faiss 中的位置位移；按相似度相关性判断排序是更优还是更劣
- 改变程度：位置位移均值、集合重叠度

用法:
    python -m eval.quantify_bm25
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "outputs"

# 加载 .env（EMBEDDING_API_*）
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def load(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))["results"]


def main() -> None:
    q = {r["query_id"]: r for r in load("mem0_candidates_en_qdrant.json")}
    f = {r["query_id"]: r for r in load("mem0_candidates_en_faiss.json")}
    nodes = {c["id"]: c["content"] for r in q.values() for c in r["candidates"]}

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["EMBEDDING_API_KEY"],
                    base_url=os.environ["EMBEDDING_API_BASE"])
    model = "BAAI/bge-large-en-v1.5"

    # 预取所有涉及的文本 embedding（查询 + 全部候选节点，qdrant/faiss 两池并集）
    texts = {c["id"]: c["content"] for r in q.values() for c in r["candidates"]}
    texts |= {c["id"]: c["content"] for r in f.values() for c in r["candidates"]}
    texts |= {r["query_id"]: r["query"] for r in q.values()}
    id_list = list(texts.keys())
    vecs = {}
    B = 50
    for i in range(0, len(id_list), B):
        ids = id_list[i:i + B]
        chunk = [texts[k] for k in ids]
        resp = client.embeddings.create(input=chunk, model=model)
        for item in sorted(resp.data, key=lambda x: x.index):
            vecs[ids[item.index]] = item.embedding
    print("[quant] embedding 计算完成", len(vecs))

    import numpy as np

    def cos(a, b):
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"\n{'qid':<6}{'qdrant独有<sim>':<14}{'faiss独有<sim>':<14}{'同节点均位位移':<12}{'q_corr':<8}{'f_corr':<8}{'方向'}")
    agg_only_q, agg_only_f, agg_disp = [], [], []
    n_improve = n_worsen = 0
    for qid in q:
        q_ids = [c["id"] for c in q[qid]["candidates"]]
        f_ids = [c["id"] for c in f[qid]["candidates"]]
        qv = vecs[qid]
        only_q = [c["id"] for c in q[qid]["candidates"] if c["id"] not in set(f_ids)]
        only_f = [c["id"] for c in f[qid]["candidates"] if c["id"] not in set(q_ids)]
        sim_q = [cos(qv, vecs[n]) for n in only_q] if only_q else []
        sim_f = [cos(qv, vecs[n]) for n in only_f] if only_f else []
        agg_only_q += sim_q
        agg_only_f += sim_f
        # 同节点位置位移
        qr = {n: i for i, n in enumerate(q_ids)}
        fr = {n: i for i, n in enumerate(f_ids)}
        disp = [abs(qr[n] - fr[n]) for n in q_ids if n in fr]
        agg_disp += disp
        # 方向：相似度与排名相关性（期望负相关，越靠前相似度越高）
        q_corr = np.corrcoef([cos(qv, vecs[n]) for n in q_ids], range(len(q_ids)))[0, 1]
        f_corr = np.corrcoef([cos(qv, vecs[n]) for n in f_ids], range(len(f_ids)))[0, 1]
        # 负相关越强=排序越贴近语义；delta<0 表示 qdrant 更优
        delta = q_corr - f_corr
        n_improve += (delta < 0)
        n_worsen += (delta > 0)
        print(f"{qid:<6}{sum(sim_q)/len(sim_q) if sim_q else float('nan'):<14.3f}"
              f"{sum(sim_f)/len(sim_f) if sim_f else float('nan'):<14.3f}"
              f"{sum(disp)/len(disp) if disp else 0:<12.1f}"
              f"{q_corr:<8.3f}{f_corr:<8.3f}"
              f"{'正向' if delta<0 else '负向' if delta>0 else '持平'}")

    print(f"\n===== 汇总 =====")
    print(f"qdrant 独有节点平均相似度: {sum(agg_only_q)/len(agg_only_q):.4f} (n={len(agg_only_q)})")
    print(f"faiss  独有节点平均相似度: {sum(agg_only_f)/len(agg_only_f):.4f} (n={len(agg_only_f)})")
    print(f"同节点平均位置位移: {sum(agg_disp)/len(agg_disp):.2f} (top-20 内, n={len(agg_disp)})")
    print(f"按相似度相关性衡量排序质量: 正向 {n_improve} 条 / 负向 {n_worsen} 条")


if __name__ == "__main__":
    main()
