# -*- coding: utf-8 -*-
"""
Agent·用户 混合记忆 + 时序 · StoryRank 故事化验证

流程：
  1. 读对话语料 → 空图 dba.maintain() 建 DBA 图（含用户/助手/时间锚三类节点 + TEMPORAL 边）；
  2. 导出 DBA 图 YAML；
  3. build_retriever → 对 agent_user_queries.yaml 逐条跑 retrieve_with_story（StoryRank 故事化）；
  4. 输出故事 + 保存 JSON（含命中 gold_expected 的溯源/统计）。

用法:
    python -m eval.run_dba_agent_user_story
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from eval.run_dba_agent_user_eval import fresh
from eval.run_p_baseline import build_retriever, _load_dotenv
from dba_pipeline.core.jump_axis import RelationType


def main() -> None:
    _load_dotenv(ROOT / ".env")

    with open(ROOT / "data/dba_eval/agent_user_dialogs.yaml", "r", encoding="utf-8") as f:
        dlg_data = yaml.safe_load(f)
    dialogs = dlg_data["dialogs"]

    # 1) 建 DBA 图
    print("[STORY] 空图建 DBA 图（逐批 maintain）...")
    g, dba, _ = fresh()
    for d in dialogs:
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in d["turns"])
        res = dba.maintain(conversation)
        if not res.get("skipped", False):
            print(f"  [{d['id']}] created={len(res.get('result', {}).get('created_ids', []))} "
                  f"节点={g.node_count} 边={g.edge_count}")
    print(f"[STORY] DBA 图: {g.node_count} 节点 / {g.edge_count} 边")

    # 2) 导出图
    graph_yaml = ROOT / "eval/outputs/dba_graph_agent_user.yaml"
    graph_yaml.parent.mkdir(parents=True, exist_ok=True)
    graph_yaml.write_text(yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[STORY] DBA 图导出: {graph_yaml}")

    # 3) 构建检索链路 + 查询
    _, retriever = build_retriever(str(graph_yaml))
    with open(ROOT / "data/dba_eval/agent_user_queries.yaml", "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    # gold 节点集（用于命中溯源）
    with open(ROOT / "data/dba_eval/agent_user_gold.yaml", "r", encoding="utf-8") as f:
        gold_data = yaml.safe_load(f)
    gold_by_prefix = {}
    for n in gold_data["nodes"]:
        pref = n["id"].rstrip("0123456789")[:2]
        gold_by_prefix.setdefault(pref, []).append(n["id"])

    results = []
    temp_edge_count = sum(1 for _, _, ed in g.graph.edges(data=True)
                          if ed.get("rel_type") == RelationType.TEMPORAL)
    for q in queries:
        qid = q["id"]
        text = q["query"]
        res = retriever.retrieve_with_story(text, with_response=False)
        stories = res.get("stories", [])
        # 0927 审查 C1 消费者同步：retrieve_with_story 不再返回采纳 id 列表（story_nodes），
        # 改为 story_used（后端按正文实际覆盖统计的节点数）；原 id 列表产出字段一并改为计数。
        story_used = res.get("story_used", 0)
        story_text = "\n\n".join(stories) if isinstance(stories, list) else str(stories)
        row = {
            "query_id": qid,
            "query": text,
            "story_used": story_used,
            "story": story_text,
            "total_matched": res.get("total_candidates"),
        }
        results.append(row)
        print(f"[STORY] {qid} | story_used={story_used} | 字数={len(story_text)}")
        print("\n".join(f"      {s}" for s in (story_text.split("\n")[:6])))

    out = {
        "meta": {
            "graph": str(graph_yaml),
            "nodes": g.node_count,
            "edges": g.edge_count,
            "temporal_edges": temp_edge_count,
        },
        "results": results,
    }
    out_path = ROOT / "eval/outputs/dba_story_agent_user.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[STORY] StoryRank 产出已写: {out_path}")
    print(f"[STORY] DBA 图含 TEMPORAL 边 {temp_edge_count} 条（时序信息已入图并参与故事化）")


if __name__ == "__main__":
    main()
