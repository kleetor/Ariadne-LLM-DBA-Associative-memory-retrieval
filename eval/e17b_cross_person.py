# -*- coding: utf-8 -*-
"""E17b：跨人物桥接（按实际图的形态修正）。

关键修正：真实图里没有 "person 节点" 也没有 "事件节点"，人物与事件都写在普通节点里，
桥是一条**跨人物的属性边**。例：
  A: "A 在市级编程比赛拿了不错的名次" (status)
  A: "A 觉得这次比赛收获很大" (emotion)
  B: "B 在市级编程比赛拿了第一" (status)
  B: "B 特别开心" (emotion)
  B: "B 想去新开的川菜馆吃一顿庆祝" (action)
  edges: a1 -causal-> a2 ; b1 -causal-> b2 ; b2 -causal-> b3 ;
         a2 -attribute-> b1        ← 公共属性（同一场比赛）的跨人物桥
query: 不提"比赛"，只问 B 的高兴与庆祝。若 PAR 能从 b2/b3 走到 b1、再沿属性边到 a2，
       就实现了"以共同经历为桥、把另一个人带出来"。

假设：这条跨人物属性边由抽取器给出（LLM 是否愿意连边不在本实验范围）。

用法：python -m eval.e17b_cross_person
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402
from eval.e3_par_vs_vector import _cosine  # noqa: E402

NODES = [
    # 核心：A 与 B 共享"市级编程比赛"
    ("a1", "A在市级编程比赛里拿了不错的名次", "status"),
    ("a2", "A把书架第二层重新整理了一遍", "action"),
    ("b1", "B在市级编程比赛拿了第一", "status"),
    ("b2", "B特别开心", "emotion"),
    ("b3", "B想去那家新开的川菜馆吃一顿庆祝", "action"),
    # 控制：C 的无关联故事
    ("c1", "C上个月从外包公司跳到了甲方", "action"),
    ("c2", "C对新公司的节奏还在适应", "status"),
    ("c3", "C说通勤比之前多半小时", "action"),
]
EDGES = [
    ("a1", "a2", "causal"),
    ("b1", "b2", "causal"),
    ("b2", "b3", "causal"),
    ("a2", "b1", "attribute"),   # 跨人物桥：同一场比赛
    ("c1", "c2", "causal"),
    ("c2", "c3", "causal"),
]

# 干扰记忆：8 条链 × 4 条 + 10 条孤立，确保 top-8 不会覆盖全图
DISTRACT = [
    ("d", "说周末去爬山", "action"), ("d", "爬到一半下雨了", "action"),
    ("d", "回家煮了姜汤", "action"), ("d", "第二天有点感冒", "status"),
    ("e", "报了个烘焙班", "action"), ("e", "第一次烤戚风塌了", "action"),
    ("e", "老师说温度要低一点", "action"), ("e", "第二次成功", "emotion"),
    ("f", "在准备驾照科目二", "action"), ("f", "倒车入库总是压线", "action"),
    ("f", "教练让他多练半小时", "action"), ("f", "考试前一晚没睡好", "status"),
    ("g", "家里换了台洗衣机", "action"), ("g", "安装师傅迟到两小时", "action"),
    ("g", "旧机器折了三百块", "action"), ("g", "新机器很安静", "status"),
    ("h", "在追一部年代剧", "action"), ("h", "每天看两集", "action"),
    ("h", "说结局有点仓促", "emotion"), ("h", "又开始看第二遍", "action"),
    ("i", "同事推荐了一家面包店", "action"), ("i", "周末去买了可颂", "action"),
    ("i", "排队排了四十分钟", "action"), ("i", "觉得不太值", "emotion"),
    ("j", "想把阳台改成小书房", "action"), ("j", "量了尺寸", "action"),
    ("j", "预算大概两万", "status"), ("j", "还在犹豫", "emotion"),
    ("k", "养的多肉开花了", "action"), ("k", "拍了好几张照片", "action"),
    ("k", "发到群里炫耀", "action"),
    ("x", "说楼下的便利店换了老板", "action"), ("x", "东西贵了一块", "status"),
    ("x", "但关东煮更好吃", "emotion"), ("y", "在学吉他", "action"),
    ("y", "手指按得有点疼", "status"), ("y", "会弹一首简单的歌", "action"),
    ("z", "上周去看了牙医", "action"), ("z", "补了一颗牙", "action"),
    ("z", "花了一千二", "status"), ("w", "想把旧手机以旧换新", "action"),
]

QUERIES = [
    ("q1", "B最近这么高兴，打算怎么犒劳自己", ["B的高兴和庆祝安排"],
     ["b2", "b3", "a2"], ["a2"]),
    ("q2", "A那次比赛收获如何", ["A对比赛的感受"], ["a1"], []),
    ("q3", "B比赛拿了什么名次", ["B的比赛结果"], ["b1"], []),
    ("q4", "那家新开的川菜馆", ["川菜馆的信息"], ["b3"], []),
    ("q5", "C最近的工作变动", ["C的跳槽近况"], ["c1", "c2"], []),
]


def main() -> int:
    _load_dotenv(ROOT / ".env")
    nodes = list(NODES)
    edges = list(EDGES)
    # 干扰链：每 4 条按顺序连 causal
    for i in range(0, len(DISTRACT), 4):
        chain = DISTRACT[i:i + 4]
        for j, (pre, c, t) in enumerate(chain):
            nid = f"{pre}{i // 4 + 1}_{j + 1}"
            nodes.append((nid, c, t))
            if j:
                edges.append((f"{pre}{i // 4 + 1}_{j}", nid, "causal"))

    data = {"nodes": [{"id": i, "content": c, "type": t} for i, c, t in nodes],
            "edges": [{"from": a, "to": b, "type": r} for a, b, r in edges]}
    p = ROOT / "eval/outputs/e17b_cross_person.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    graph, retriever = build_retriever(str(p), vector_cache_key="e17b")
    ids = [n for n in graph.graph.nodes() if graph.graph.nodes[n].get("content")]
    vs = retriever.vector_store
    vecs = vs.get_content_vectors(ids)

    def topk(text, k=5):
        qv = vs._embed(text)
        sc = sorted(((n, _cosine(qv, v)) for n, v in zip(ids, vecs) if v is not None),
                    key=lambda x: -x[1])
        return [n for n, _ in sc[:k]]

    rows = []
    print(f"nodes={len(ids)}")
    for qid, text, purposes, expected, bridged in QUERIES:
        res = retriever.retrieve(text, purpose=purposes, seed_k=5, max_peak_nodes=5)
        par = [m[0] for m in res.get("peak_memories", [])]
        vec = topk(text, 5)

        def rec(ids_, exp):
            return round(len(set(ids_) & set(exp)) / len(exp), 3), sorted(set(ids_) & set(exp))

        pr, ph = rec(par, expected)
        vr, vh = rec(vec, expected)
        br = None if not bridged else round(len(set(par) & set(bridged)) / len(bridged), 3)
        vb = None if not bridged else round(len(set(vec) & set(bridged)) / len(bridged), 3)
        rows.append({"qid": qid, "par_recall": pr, "vec_recall": vr,
                     "par_bridge": br, "vec_bridge": vb,
                     "par_hits": ph, "vec_hits": vh, "par_nodes": len(par)})
        print(f"{qid:3s} par={pr:.2f} {ph}  vec={vr:.2f} {vh}  bridge par={br} vec={vb}",
              flush=True)

    def agg(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(sum(v) / len(v), 3) if v else None

    summary = {k: agg(k) for k in ("par_recall", "vec_recall", "par_bridge", "vec_bridge")}
    print("\n=== summary ===", summary)
    out = ROOT / "eval/outputs/e17b_cross_person.json"
    out.write_text(json.dumps({"meta": {"nodes": len(ids)}, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("[wrote]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
