# -*- coding: utf-8 -*-
"""
抽取粒度实验：整段完整对话 vs 分批抽取 + StoryRank 产物差异比对

实验设计：
  A. 抽取效果：把 agent_user_dialogs 的全部 turns 拼接为「同一段完整对话文本」，
     一次 dba.maintain() 抽取（整段图）；与之前「分批（逐批）抽取」对比节点/边/主体召回/时序。
  B. StoryRank 产物 diff：分别用「分批图」与「整段图」对 agent_user_queries 逐条跑
     retrieve_with_story，比对两类故事产物（story_used 计数、文本长度、助手/时序覆盖）。
     0927 审查 C1：StoryRank 输出契约变更后不再有采纳 id 列表，原按 story_nodes 重叠对齐两臂
     的对比已不可行，此处只保留 story_used 计数等可得的产物指标。

用法:
    python -m eval.run_dba_agent_user_fulltext
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import yaml

from eval.run_dba_agent_user_eval import fresh
from eval.run_p_baseline import build_retriever, _load_dotenv
from eval.run_dba_extraction_eval import match_nodes
from dba_pipeline.core.jump_axis import RelationType

# 时间词（用于判断故事是否含时序信息）
TIME_WORDS_RE = ("昨天", "明天", "前天", "上周", "这周", "本周", "下周",
                 "周一", "周二", "周三", "周四", "周五", "周六", "周日",
                 "去年", "今年", "明年", "去年", "上个月", "这月", "下个月")


def build_full_conversation(dialogs) -> str:
    """把全部批次 turns 按顺序拼接为同一段完整对话文本"""
    return "\n".join(f"{t['role']}({t['time']}): {t['text']}" for d in dialogs for t in d["turns"])


def node_metrics(g):
    """返回整段图的规模/主体/时序统计"""
    active = []
    for nid in g.graph.nodes():
        n = g.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            active.append({"id": nid, "content": n["content"],
                           "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})
    ai = [n for n in active if n["content"].startswith("助手")]
    user = [n for n in active if n["content"].startswith("用户")]
    time = [n for n in active if n["type"] == "thing"]
    temporal = [(a, b) for a, b, ed in g.graph.edges(data=True)
                if ed.get("rel_type") == RelationType.TEMPORAL]
    return {"active": len(active), "nodes": g.node_count, "edges": g.edge_count,
            "user": len(user), "ai": len(ai), "time": len(time), "temporal": len(temporal),
            "_active": active}


def main() -> None:
    _load_dotenv(ROOT / ".env")

    with open(ROOT / "data/dba_eval/agent_user_dialogs.yaml", "r", encoding="utf-8") as f:
        dlg_data = yaml.safe_load(f)
    dialogs = dlg_data["dialogs"]

    # gold 主体集
    with open(ROOT / "data/dba_eval/agent_user_gold.yaml", "r", encoding="utf-8") as f:
        gold_data = yaml.safe_load(f)
    gold_base = {n["id"]: {"content": n["content"], "type": n["type"]} for n in gold_data["nodes"]}
    gold_ai = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("a")]
    gold_user = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("u")]
    gold_time = [n["id"] for n in gold_data["nodes"] if n["id"].rstrip("0123456789")[:2].startswith("t")]

    # ===== A. 整段一次抽取 =====
    full_conv = build_full_conversation(dialogs)
    print(f"[A] 完整对话文本长度={len(full_conv)} 字符")
    g, dba, embeddings = fresh()
    embed_fn = lambda text: np.array(embeddings.embed_query(text))  # noqa: E731
    res = dba.maintain(full_conv)
    skipped = res.get("skipped", False)
    print(f"[A] 整段一次 maintain: triage_skipped={skipped} "
          f"node_ops={len(res.get('ops', {}).get('node_ops', []))} "
          f"created={len(res.get('result', {}).get('created_ids', []))}")

    def recall(gold_ids, dba_list):
        if not gold_ids or not dba_list:
            return None
        gold_list = [{"id": nid, "content": gold_base[nid]["content"], "type": gold_base[nid]["type"]} for nid in gold_ids]
        m = match_nodes(gold_list, dba_list, embed_fn, 0.75)
        return round(len(m) / len(gold_list), 4)

    fm = node_metrics(g)
    r_ai = recall(gold_ai, [n for n in fm["_active"] if n["content"].startswith("助手")])
    r_user = recall(gold_user, [n for n in fm["_active"] if n["content"].startswith("用户")])
    r_time = recall(gold_time, [n for n in fm["_active"] if n["type"] == "thing"])
    full_meta = {**{k: v for k, v in fm.items() if k != "_active"},
                 "recall_user": r_user, "recall_ai": r_ai, "recall_time": r_time}

    # 分批 baseline
    with open(ROOT / "eval/outputs/dba_agent_user_extraction.json", "r", encoding="utf-8") as f:
        batch_json = json.load(f)
    batch_meta = {
        "nodes": batch_json["stats"]["nodes"], "edges": batch_json["stats"]["edges"],
        "active": batch_json["stats"]["active"], "user": None, "ai": batch_json["assert"]["mixed_memory"]["ai_nodes_in_dba"],
        "time": batch_json["assert"]["temporal"]["time_nodes_in_dba"], "temporal": batch_json["assert"]["temporal"]["temporal_edges"],
        "recall_user": batch_json["recall"]["user"], "recall_ai": batch_json["recall"]["assistant"],
        "recall_time": batch_json["recall"]["time_anchor"],
    }

    # 导出整段图
    graph_yaml = ROOT / "eval/outputs/dba_graph_agent_user_fulltext.yaml"
    graph_yaml.parent.mkdir(parents=True, exist_ok=True)
    graph_yaml.write_text(yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[A] 整段图导出: {graph_yaml}")

    print("\n===== A. 抽取效果对比（gold=96 节点 / 104 边）=====")
    print(f"{'指标':<14}{'分批(逐批)':<18}{'整段(一次)':<18}")
    for k, lbl in [("nodes", "节点总数"), ("edges", "边总数"), ("active", "活跃节点"),
                   ("ai", "助手节点"), ("time", "时间锚节点"), ("temporal", "TEMPORAL边"),
                   ("recall_user", "用户召回"), ("recall_ai", "助手召回"), ("recall_time", "时间锚召回")]:
        bv = batch_meta.get(k, "-")
        fv = full_meta.get(k, "-")
        print(f"{lbl:<14}{str(bv):<18}{str(fv):<18}")

    # ===== B. StoryRank 产物 diff =====
    _, retriever_full = build_retriever(str(graph_yaml))
    with open(ROOT / "data/dba_eval/agent_user_queries.yaml", "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)
    with open(ROOT / "eval/outputs/dba_story_agent_user.json", "r", encoding="utf-8") as f:
        story_batch = json.load(f)
    batch_by_qid = {r["query_id"]: r for r in story_batch["results"]}

    def _story_flags(txt):
        return {"has_assistant": "助手" in txt, "has_time": any(w in txt for w in TIME_WORDS_RE),
                "len": len(txt)}

    diff_rows = []
    for q in queries:
        qid = q["id"]
        r_full = retriever_full.retrieve_with_story(q["query"], with_response=False)
        full_story = r_full.get("stories", [])
        # 0927 审查 C1 消费者同步：StoryRank 输出契约变更，retrieve_with_story 不再返回采纳
        # id 列表（story_nodes），改为 story_used（后端按正文实际覆盖统计的节点数）。原本"分批 vs
        # 整段"依赖两臂采纳 id 集合的 Jaccard 对齐，现无 id 列表，该对比已不可行——改为只比
        # story_used 计数（不编造 id）。
        full_used = r_full.get("story_used", 0)
        full_text = "\n\n".join(full_story) if isinstance(full_story, list) else str(full_story)
        b = batch_by_qid.get(qid)
        batch_text = b["story"] if b else ""
        # 历史产物 dba_story_agent_user.json 为旧契约（含 story_nodes id 列表），无 story_used，
        # 此处不编造：取不到就记为 None。
        batch_used = (b.get("story_used") if b else None)

        diff_rows.append({
            "query_id": qid, "query": q["query"],
            "batch": {"story_used": batch_used, "story": batch_text,
                      "flags": _story_flags(batch_text)},
            "fulltext": {"story_used": full_used, "story": full_text,
                         "flags": _story_flags(full_text)},
            "note": "0927 起无采纳 id 列表，原 story_nodes 重叠/Jaccard 对比已不可行，"
                    "仅保留 story_used 计数",
        })
        print(f"\n[STORY diff] {qid} | story_used 分批{batch_used} vs 整段{full_used}")

    out_path = ROOT / "eval/outputs/dba_story_diff_agent_user.json"
    out_path.write_text(json.dumps({
        "meta": {"gold_nodes": 96, "mapping": "分批=逐批15次maintain；整段=同一完整对话一次maintain"},
        "extraction": {"batch": batch_meta, "fulltext": full_meta},
        "story_diff": diff_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== B. StoryRank 产物差异比对 =====\n")
    for r in diff_rows:
        bf, ff = r["batch"]["flags"], r["fulltext"]["flags"]
        print(f"[{r['query_id']}] story_used 分批={r['batch']['story_used']} 整段={r['fulltext']['story_used']} "
              f"| 分批 len={bf['len']} 助手={'是' if bf['has_assistant'] else '否'} 时序={'是' if bf['has_time'] else '否'} "
              f"| 整段 len={ff['len']} 助手={'是' if ff['has_assistant'] else '否'} 时序={'是' if ff['has_time'] else '否'}")
    print(f"\n差异比对产出: {out_path}")


if __name__ == "__main__":
    main()
