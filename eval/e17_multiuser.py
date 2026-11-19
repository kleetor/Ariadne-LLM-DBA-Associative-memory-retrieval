# -*- coding: utf-8 -*-
"""E17：群聊/私聊场景下的多人物记忆（合成小图，验证两个想法）。

背景：记忆主体是"用户"，对话发生在私聊与群聊里，由用户方主动提出并被记录。
人物：用户 U、同事老王 W、朋友小李 L、群友阿May M（共 4 人，含 3 个他人）。
两类共享事件：周六团建（U/W/L/M 都提过）、年终奖（U/W/L 提过）。

验证：
  (a) 名称锚：带人名的记忆是否按人物分化；
  (b) 事件桥：查询共享事件时，PAR 能否同时带出多个人的记忆（向量 top-k 通常只出一侧）。

用法：python -m eval.e17_multiuser
输出：eval/outputs/e17_multiuser.json + 图 YAML
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

# ---- 人物与记忆（内容按真实聊天语感写：倾诉 / 日常 / 问答 / 群聊） ----
NODES = [
    # 人物节点
    ("u0", "用户，后端开发，最近项目很赶", "status"),
    ("w0", "同事老王，技术组长，对代码质量要求严格", "person"),
    ("l0", "朋友小李，正在准备考研", "person"),
    ("m0", "群友阿May，做数据平台，已换工作", "person"),
    # 用户 U（私聊/群聊里主动说的）
    ("u1", "用户在私聊里跟老王说，他带两瓶酒", "action"),
    ("u2", "用户跟小李吐槽最近加班到十一点，整个人很累", "action"),
    ("u3", "用户问阿May有没有好用的数据库迁移工具", "action"),
    ("u4", "用户中午和老王在公司楼下吃了牛肉面", "action"),
    ("u5", "用户跟小李说，年终奖不到三个月就考虑跳槽", "action"),
    ("u6", "用户在群里说那天他要晚到一小时", "action"),
    ("u7", "用户问老王团建的场地订在哪了", "action"),
    ("u8", "用户告诉阿May，她上次推荐的工具已经用上了", "action"),
    # 老王 W
    ("w1", "老王说场地由他来订", "action"),
    ("w2", "老王在群里说年终奖大概年前发", "action"),
    ("w3", "老王中午常去公司楼下那家牛肉面馆", "action"),
    ("w4", "老王负责后端代码评审，要求很严格", "status"),
    ("w5", "老王说他车后座还能坐两个人", "action"),
    ("w6", "老王劝用户先别急着跳槽，等年终奖定了再说", "action"),
    # 小李 L
    ("l1", "小李听用户吐槽加班，劝他早点睡", "action"),
    ("l2", "小李说他最近在研究光圈和快门", "action"),
    ("l3", "小李和用户约好散场后去吃烧烤", "action"),
    ("l4", "小李最近在准备考研，每天复习到很晚", "action"),
    ("l5", "小李说他们公司年终奖发得早", "action"),
    # 阿May M
    ("m1", "阿May推荐用 Flyway 做数据库迁移", "action"),
    ("m2", "阿May说她那天请假不来", "action"),
    ("m3", "阿May换了新工作，在做数据平台", "status"),
    ("m4", "阿May问有没有人玩狼人杀", "action"),
    ("m5", "阿May提醒用户，工具要先在测试库试一遍", "action"),
    # 共享事件
    ("e1", "周六团建", "thing"),
    ("e2", "年终奖", "thing"),
    ("e3", "公司楼下的牛肉面馆", "thing"),
]

EDGES = [
    # 人物归属
    ("u1", "w0", "social"), ("u4", "w0", "social"), ("u7", "w0", "social"),
    ("u2", "l0", "social"), ("u5", "l0", "social"),
    ("u3", "m0", "social"), ("u8", "m0", "social"),
    ("w1", "w0", "attribute"), ("w4", "w0", "attribute"),
    ("l4", "l0", "attribute"), ("m3", "m0", "attribute"),
    # 共享事件：团建 e1
    ("u1", "e1", "scenario"), ("u6", "e1", "scenario"), ("u7", "e1", "scenario"),
    ("w1", "e1", "scenario"), ("w5", "e1", "scenario"),
    ("l2", "e1", "scenario"), ("l3", "e1", "scenario"),
    ("m2", "e1", "scenario"), ("m4", "e1", "scenario"),
    # 共享事件：年终奖 e2
    ("u5", "e2", "scenario"), ("w2", "e2", "scenario"), ("l5", "e2", "scenario"),
    ("w6", "u5", "causal"),
    # 共享事件：牛肉面馆 e3
    ("u4", "e3", "scenario"), ("w3", "e3", "scenario"),
]

QUERIES = [
    ("q1", "周六团建都有谁去、怎么安排", ["周六团建都有谁参加", "团建的时间和场地怎么定"],
     ["u1", "u6", "u7", "w1", "w5", "l2", "m4"], ["u1", "u6", "w1", "w5", "l2", "m4"]),
    ("q2", "年终奖什么时候发", ["年终奖的发放时间", "年终奖与跳槽的关系"],
     ["u5", "w2", "l5", "w6"], []),
    ("q3", "老王对代码评审有什么要求", ["老王的评审标准", "代码质量要求"],
     ["w4"], []),
    ("q4", "小李最近在忙什么", ["小李的近况"], ["l4"], []),
    ("q5", "阿May推荐的数据库工具", ["数据库迁移工具的推荐"], ["m1", "m5"], []),
    ("q6", "团建要带什么、能带家属吗", ["团建的携带物品与家属安排"],
     ["w5", "m4", "l2"], ["w5", "m4", "l2"]),
    ("q7", "公司楼下那家面馆", ["公司附近的午饭去处"], ["u4", "w3"], []),
    ("q8", "用户为什么想跳槽", ["想跳槽的原因和时机"], ["u5", "w6"], []),
    ("q9", "小李怎么劝用户的", ["小李给的作息建议"], ["l1"], []),
    ("q10", "阿May为什么不来团建", ["阿May缺席团建的原因"], ["m2"], ["m2"]),
    ("d1", "阿凯说想学游泳的事", ["阿凯的健身安排"], ["k12"], []),
    ("d2", "老赵公司裁员", ["老赵的近况"], ["z13"], []),
    ("d3", "小雨老家拆迁", ["小雨家里的事"], ["y16"], []),
    ("d4", "小林学英语口语", ["小林的学习计划"], ["q7"], []),
]

PARTY = {"u": "U", "w": "W", "l": "L", "m": "M", "e": "E"}

# ---- 放大到 ~200 节点：9 个干扰人物 × 16 条 + 8 个话题枢纽 ----
D_PERSONS = [
    ("z0", "大学室友老赵，在国企做运维", "老赵", "z"),
    ("k0", "健身教练阿凯", "阿凯", "k"),
    ("o0", "用户的妈妈，在老家", "妈妈", "o"),
    ("r0", "前同事老周，去了大厂", "老周", "r"),
    ("y0", "表妹小雨，在读大四", "小雨", "y"),
    ("n0", "邻居王姐，做外贸", "王姐", "n"),
    ("c0", "高中同学陈浩，爱开玩笑", "陈浩", "c"),
    ("q0", "同事小林，做前端", "小林", "q"),
    ("x0", "朋友徐倩，在医院工作", "徐倩", "x"),
]
FACTS = [
    "周末去爬山，路线选在城郊",
    "最近在学做饭，拿手菜是番茄牛腩",
    "换了新工作，通勤要一个多小时",
    "养了一只橘猫，叫团子",
    "体检发现血脂偏高",
    "打算十一去云南",
    "报名了英语口语班",
    "在看房，首付还差一点",
    "家里安排了相亲",
    "最近在打一款新游戏",
    "追完了一部悬疑剧",
    "说想学游泳",
    "聊到公司裁员",
    "说孩子上幼儿园了",
    "买了新车，还在磨合",
    "说老家可能要拆迁",
]
HUBS = {1: ("t1", "爬山"), 2: ("t2", "做饭"), 3: ("t3", "换工作"),
        4: ("t4", "养猫"), 6: ("t5", "旅行"), 7: ("t6", "学英语"),
        8: ("t7", "买房"), 10: ("t8", "游戏")}

EXTRA_NODES, EXTRA_EDGES = [], []
for pid, desc, nm, pre in D_PERSONS:
    EXTRA_NODES.append((pid, desc, "person"))
    for i, fact in enumerate(FACTS, 1):
        mid = f"{pre}{i}"
        EXTRA_NODES.append((mid, f"{nm}说{fact}", "action"))
        EXTRA_EDGES.append((mid, pid, "attribute"))
        if i in HUBS:
            EXTRA_EDGES.append((mid, HUBS[i][0], "scenario"))
for hid, label in HUBS.values():
    EXTRA_NODES.append((hid, label, "thing"))

# 再补 8 条核心人物记忆，凑到 200 节点
EXTRA_NODES += [
    ("u9", "用户跟老王说，周五要发版，别安排别的", "action"),
    ("u10", "用户问小李考研要不要辞职", "action"),
    ("u11", "用户在群里问团建吃什么", "action"),
    ("u12", "用户跟阿May说，迁移脚本已经跑通", "action"),
    ("w7", "老王说发版前先把回滚方案写好", "action"),
    ("w8", "老王问用户周五能不能留守", "action"),
    ("l6", "小李说考研报名截止在月底", "action"),
    ("l7", "小李问用户团建能不能早点走", "action"),
]
EXTRA_EDGES += [
    ("u9", "w0", "social"), ("u10", "l0", "social"), ("u11", "e1", "scenario"),
    ("u12", "m0", "social"), ("w7", "w0", "attribute"), ("w8", "w0", "attribute"),
    ("l6", "l0", "attribute"), ("l7", "l0", "attribute"),
    ("w7", "u9", "causal"), ("u10", "l6", "scenario"), ("l7", "e1", "scenario"),
]

ALL_NODES = NODES + EXTRA_NODES
ALL_EDGES = EDGES + EXTRA_EDGES


def _party(nid: str) -> str:
    return PARTY.get(nid[0], "?")


def main() -> int:
    _load_dotenv(ROOT / ".env")
    data = {"nodes": [{"id": i, "content": c, "type": t} for i, c, t in ALL_NODES],
            "edges": [{"from": a, "to": b, "type": r} for a, b, r in ALL_EDGES]}
    p = ROOT / "eval/outputs/e17_multiuser.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    graph, retriever = build_retriever(str(p), vector_cache_key="e17")
    vs = retriever.vector_store
    ids = [n for n in graph.graph.nodes() if graph.graph.nodes[n].get("content")]
    vecs = vs.get_content_vectors(ids)

    def vec_topk(text, k):
        qv = vs._embed(text)
        sc = [(n, _cosine(qv, v)) for n, v in zip(ids, vecs) if v is not None]
        sc.sort(key=lambda x: -x[1])
        return [n for n, _ in sc[:k]]

    rows = []
    for qid, text, purposes, expected, bridged in QUERIES:
        res = retriever.retrieve(text, purpose=purposes, seed_k=6, max_peak_nodes=8)
        par_ids = [m[0] for m in res.get("peak_memories", [])]
        v_ids = vec_topk(text, 8)

        def parties(ids_):
            return {_party(i) for i in ids_ if _party(i) != "E"}

        def bridge_recall(ids_):
            if not bridged:
                return None
            return round(len(set(ids_) & set(bridged)) / len(bridged), 3)

        exp_parties = {_party(i) for i in expected}
        rows.append({
            "qid": qid, "n_expected": len(expected), "n_bridged": len(bridged),
            "par_nodes": len(par_ids), "vec_nodes": len(v_ids),
            "par_party_hit": len(parties(par_ids) & exp_parties),
            "vec_party_hit": len(parties(v_ids) & exp_parties),
            "par_recall": round(len(set(par_ids) & set(expected)) / len(expected), 3),
            "vec_recall": round(len(set(v_ids) & set(expected)) / len(expected), 3),
            "par_bridge": bridge_recall(par_ids), "vec_bridge": bridge_recall(v_ids),
            "par_parties": sorted(parties(par_ids)), "vec_parties": sorted(parties(v_ids)),
        })
        print(f"{qid:4s} exp={len(expected)} bridged={len(bridged)} "
              f"par={rows[-1]['par_recall']:.2f}/{rows[-1]['par_bridge']} "
              f"vec={rows[-1]['vec_recall']:.2f}/{rows[-1]['vec_bridge']}", flush=True)

    def agg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {"n_queries": len(rows), "par_party_hit": agg("par_party_hit"),
               "vec_party_hit": agg("vec_party_hit"),
               "par_recall": agg("par_recall"), "vec_recall": agg("vec_recall"),
               "par_bridge": agg("par_bridge"), "vec_bridge": agg("vec_bridge")}
    print("\n=== summary ===")
    print(f"  party hit (人数覆盖): par={summary['par_party_hit']} vec={summary['vec_party_hit']}")
    print(f"  node recall:          par={summary['par_recall']} vec={summary['vec_recall']}")
    print(f"  bridge recall:        par={summary['par_bridge']} vec={summary['vec_bridge']}")

    out = ROOT / "eval/outputs/e17_multiuser.json"
    out.write_text(json.dumps({"meta": {"embedding": os.environ.get("EMBEDDING_MODEL"),
                                        "nodes": len(ids)},
                               "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("[wrote]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
