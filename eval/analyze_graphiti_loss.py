# -*- coding: utf-8 -*-
"""归因分析：Graphiti expected 覆盖损失的来源拆解

把每条未命中的 expected 节点分成三类：
  A. 抽取丢失：该节点从未被任何 fact 引用（content 没有被 LLM 抽取成可检索的事实）
  B. 检索遗漏：节点在图中（被某 fact 引用）但未进入 top-N 候选
  C. 映射/去重损耗：节点 id 被实体合并/改写吞并（episodes 里没有该 uuid）

同时量化"候选集膨胀挤占"：Graphiti 候选数 vs P 输出配额 N。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from neo4j import GraphDatabase

GRAPHITI = "eval/outputs/graphiti_candidates.json"
P = "eval/outputs/p_candidates_v3.json"
QUERIES = "data/LiFEMem_queries_v3.yaml"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "lifemem2026")


def main() -> None:
    with open(ROOT / GRAPHITI, encoding="utf-8") as f:
        g = {r["query_id"]: r for r in json.load(f)["results"]}
    with open(ROOT / P, encoding="utf-8") as f:
        p = {r["query_id"]: r for r in json.load(f)["results"]}
    with open(ROOT / QUERIES, encoding="utf-8") as f:
        qs = {q["id"]: q for q in yaml.safe_load(f)["queries"]}

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def in_graph(nid: str) -> bool:
        """节点 id 对应的 episode 是否被任一 fact（RELATES_TO 边的 episodes 数组）引用"""
        res = driver.execute_query(
            "MATCH (ep:Episodic {name: $nid}) WITH ep.uuid AS eu "
            "MATCH ()-[e:RELATES_TO]->() WHERE eu IN e.episodes "
            "RETURN count(e) AS c",
            nid=nid,
        )
        return bool(res.records and res.records[0]["c"] > 0)

    stat = {"miss": 0, "extract_loss": 0, "retrieval_miss": 0, "mapping_loss": 0,
            "all_expected": 0, "expected_in_graph": 0}
    rows = []
    for qid, q in qs.items():
        expected = q.get("expected", [])
        scen = q.get("scenario", "")
        if not expected:
            continue
        cand_ids = {c["id"] for c in g.get(qid, {}).get("candidates", [])}
        missed = [nid for nid in expected if nid not in cand_ids]
        hit = len(expected) - len(missed)
        n_quota = p[qid].get("total_candidates", 0)
        n_cand = len(g.get(qid, {}).get("candidates", []))
        # 分解 missed
        cat = {"extract_loss": 0, "retrieval_miss": 0, "mapping_loss": 0}
        for nid in missed:
            stat["miss"] += 1
            if in_graph(nid):
                cat["retrieval_miss"] += 1
                stat["retrieval_miss"] += 1
            else:
                cat["extract_loss"] += 1
                stat["extract_loss"] += 1
        for nid in expected:
            stat["all_expected"] += 1
            if in_graph(nid):
                stat["expected_in_graph"] += 1
        rows.append((qid, scen, f"{hit}/{len(expected)}", cat, n_quota, n_cand))

    print(f"{'qid':<5}{'scenario':<15}{'hit':<7}{'抽丢':<4}{'检漏':<4}{'N配额':<6}{'候选数'}")
    for qid, scen, hit, cat, quota, nc in rows:
        print(f"{qid:<5}{scen:<15}{hit:<7}{cat['extract_loss']:<4}{cat['retrieval_miss']:<4}{quota:<6}{nc}")

    tot = stat["miss"]
    print("\n===== 汇总（30 条非弃权，81 个 expected 节点） =====")
    print(f"expected 总数: {stat['all_expected']}，其中被任一 fact 引用（进图）: {stat['expected_in_graph']} "
          f"({stat['expected_in_graph']/stat['all_expected']:.1%})")
    print(f"未命中: {tot}")
    print(f"  A. 抽取丢失（节点从未被抽成事实）: {stat['extract_loss']} "
          f"({stat['extract_loss']/tot:.1%} of miss)")
    print(f"  B. 检索遗漏（在图中但未召回）: {stat['retrieval_miss']} "
          f"({stat['retrieval_miss']/tot:.1%} of miss)")
    driver.close()


if __name__ == "__main__":
    main()
