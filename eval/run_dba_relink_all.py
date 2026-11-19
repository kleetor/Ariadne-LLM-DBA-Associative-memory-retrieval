# -*- coding: utf-8 -*-
"""
DBA 遍历联想连边实验（第二轮边连接·补全版，按内容场景分类）

对 DBA 全量图（dba_graph_full.yaml，261 节点/529 边）做**遍历式记忆联想连边**：
- 覆盖对象包括旧节点（不依赖对话/新节点）
- 由于 DBA 节点 id 为 n1/n2...（自动编号），按**节点内容语义**做场景前缀分类
  （w/he/so/e/h/ln/c/f/t/sh/th/n），再按场景分批联想连边
- 统计新增边、gold 命中绝对数/重合率前后变化，导出第二轮图

用法:
    python -m eval.run_dba_relink_all --classify-batch 45
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from dba_pipeline.loader import load_graph
from eval.run_dba_extraction_eval import build_llm, build_embeddings, _load_dotenv

RELINK_ASSOC_PROMPT = """你是记忆图谱的「联想连接器」。下面是某用户记忆图谱中一个场景子图的全部节点与已有边。

请你进行**记忆联想遍历连边**：
1. 审查本批【全部节点】（包括旧节点），找出所有存在联想关系的节点对；
2. 联想关系包括：CAUSAL=因→果、SCENARIO=同场景并存、SEQUENCE=先→后、PREFERENCE=偏好、SOCIAL=人物/社交关系、ATTRIBUTE=属性/特征、TEMPORAL=事件→时间、TAXONOMIC=子→父；
3. **宁多勿漏**：只要两个事实之间存在合理联想（同主题相关、因果、场景共享、人物关联、偏好从属、时序先后等）就创建边；
4. 已存在于「已有边」中的关系不要重复创建；
5. 边类型要准确（参照定义）。

## 本批节点

{nodes_text}

## 本批已有边

{edges_text}

## 输出

严格输出 JSON：{{"edge_ops":[{{"action":"create","from":"n1","to":"n2","rel_type":"CAUSAL"}}]}}
只输出 JSON，不要任何额外文字。"""

RELINK_CONSERVATIVE_PROMPT = """你是记忆图谱的「联想连接器」。下面是某用户记忆图谱中一个场景子图的全部节点与已有边。

请你进行**保守的记忆联想连边**：
1. 审查本批【全部节点】（包括旧节点），只对**语义明确、高置信**的联想关系连边；
2. 联想关系包括：CAUSAL=因→果、SCENARIO=同场景并存、SEQUENCE=先→后、PREFERENCE=偏好、SOCIAL=人物/社交关系、ATTRIBUTE=属性/特征、TEMPORAL=事件→时间、TAXONOMIC=子→父；
3. **宁缺毋滥**：只连明确的关系——明显的因果（如「加班导致颈椎不适」）、明确的人物关系（如「同事小王一起加班」）、明确的偏好从属（如「拿铁是最爱的咖啡」）、明确的同场景共存。模糊的、牵强的、仅是"相关"的不要连；
4. 已存在于「已有边」中的关系不要重复创建；
5. 边类型要准确（参照定义）。

## 本批节点

{nodes_text}

## 本批已有边

{edges_text}

## 输出

严格输出 JSON：{{"edge_ops":[{{"action":"create","from":"n1","to":"n2","rel_type":"CAUSAL"}}]}}
只输出 JSON，不要任何额外文字。"""

RELINK_BALANCED_PROMPT = """你是记忆图谱的「联想连接器」。下面是某用户记忆图谱中一个场景子图的全部节点与已有边。

请你进行**克制的记忆联想连边**：
1. 审查本批【全部节点】（包括旧节点），对存在**明确或较强联想**的节点对连边；
2. 联想关系包括：CAUSAL=因→果、SCENARIO=同场景并存、SEQUENCE=先→后、PREFERENCE=偏好、SOCIAL=人物/社交关系、ATTRIBUTE=属性/特征、TEMPORAL=事件→时间、TAXONOMIC=子→父；
3. **连边标准（介于宁缺毋滥与宁多勿漏之间）**：
   - 明确因果、明确人物关系、明确偏好从属、明确同场景强相关（同一主题下的直接关联）——**都要连**；
   - 仅"主题相关"但关系牵强、或可通过其他节点间接表达的弱联想——**不要连**；
4. 已存在于「已有边」中的关系不要重复创建；
5. 边类型要准确（参照定义）。

## 本批节点

{nodes_text}

## 本批已有边

{edges_text}

## 输出

严格输出 JSON：{{"edge_ops":[{{"action":"create","from":"n1","to":"n2","rel_type":"CAUSAL"}}]}}
只输出 JSON，不要任何额外文字。"""

CLASSIFY_PROMPT = """你是记忆整理员。下面是用户记忆图谱中的一批节点（id + 内容）。请为每个节点判断所属场景，输出 JSON 映射 {{node_id: 场景前缀}}。

场景前缀定义（必须从中选择）：
- w = 工作（加班/代码/项目/公司/同事/绩效）
- he = 健康（体检/运动/颈椎/咖啡/饮食/睡眠）
- so = 社交（朋友/聚会/关系/社团）
- e = 情绪（心情/压力/低落/开心/焦虑）
- h = 学生时代·高中（高中/高考/成绩/早恋/晚自习）
- ln = 学生时代·学习（自学/转码/编程学习/课程）
- c = 学生时代·大专（大专/大学/宿舍/社团/毕业）
- f = 家庭（家人/父母/妹妹/家庭）
- t = 旅行（旅行/城市/出行）
- sh = 购物消费（购物/消费/网购）
- th = 人生阶段总结（人生/阶段/时期总结）
- n = 噪声/无关见闻（他人事情/天气/短信/路况等与用户自身无关的见闻）

## 节点

{node_lines}

## 输出

只输出 JSON，如 {{"n1": "w", "n2": "he"}}，不要任何额外文字。"""


def _parse_json_obj(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no json in: {text[:200]}")
    return json.loads(text[start:end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 遍历联想连边实验（内容场景分类）")
    parser.add_argument("--graph", default="eval/outputs/dba_graph_full.yaml")
    parser.add_argument("--eval-json", default="eval/outputs/dba_extraction_full.json")
    parser.add_argument("--full-yaml", default="data/LiFEMem.yaml")
    parser.add_argument("--classify-batch", type=int, default=45, help="场景分类每批节点数")
    parser.add_argument("--mode", choices=["aggressive", "conservative", "balanced"], default="balanced",
                        help="联想连边模式：aggressive=宁多勿漏 / conservative=宁缺毋滥 / balanced=克制")
    parser.add_argument("--out-graph", default=None, help="导出图路径（默认按 mode）")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认按 mode）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mode = args.mode
    relink_prompt = {"aggressive": RELINK_ASSOC_PROMPT,
                     "conservative": RELINK_CONSERVATIVE_PROMPT,
                     "balanced": RELINK_BALANCED_PROMPT}[mode]
    suffix = {"aggressive": "_agg", "conservative": "_cons", "balanced": "_bal"}[mode]
    args.out_graph = args.out_graph or f"eval/outputs/dba_graph_full_relink{suffix}.yaml"
    args.out = args.out or f"eval/outputs/dba_relink{suffix}.json"
    print(f"[RELINK-ALL] 模式: {mode}（{{'aggressive': '宁多勿漏', 'conservative': '宁缺毋滥', 'balanced': '克制'}}[mode]）")

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    graph = load_graph(ROOT / args.graph)
    init_edges = graph.edge_count
    print(f"[RELINK-ALL] 初始: {graph.node_count} 节点 / {init_edges} 边")

    # dba→gold 映射（抽取评测）
    d2g = {}
    ev = json.loads((ROOT / args.eval_json).read_text(encoding="utf-8"))["eval"]
    for d in ev.get("dba_node_detail", []):
        if d.get("matched_gold_id"):
            d2g[d["id"]] = d["matched_gold_id"]

    # gold 全图边（仅内部）
    g_full = load_graph(str(ROOT / args.full_yaml))
    gold_edges = set()
    for u, v in g_full.graph.edges():
        gold_edges.add((u, v))
        gold_edges.add((v, u))

    def gold_stats(gr):
        hit = 0
        for a, b in gr.graph.edges():
            if a in d2g and b in d2g and (d2g[a], d2g[b]) in gold_edges:
                hit += 1
        return hit, round(hit / gr.edge_count, 4) if gr.edge_count else 0.0

    llm = build_llm()
    builder = __import__("dba_pipeline.extraction.graph_builder", fromlist=["GraphBuilder"]).GraphBuilder(
        graph, __import__("dba_pipeline.embedding.store", fromlist=["VectorStore"]).VectorStore(
            embeddings=build_embeddings(), backend="faiss", persist_dir=None), orphan_threshold=100)

    # ---- Step A：按节点内容做场景前缀分类 ----
    active = [nid for nid in graph.graph.nodes()
              if (n := graph.get_node(nid)) and not n.get("deprecated") and not n.get("forgotten")]
    scene_of = {}
    node_lines = [f"{nid} | {graph.get_node(nid)['content']}" for nid in active]
    groups = [node_lines[i:i + args.classify_batch] for i in range(0, len(node_lines), args.classify_batch)]
    for gi, grp in enumerate(groups, 1):
        prompt = CLASSIFY_PROMPT.format(node_lines="\n".join(grp))
        try:
            raw = llm.invoke(prompt).content
            mapping = _parse_json_obj(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[RELINK-ALL] 分类批{gi}: 解析失败，跳过（{exc}）")
            continue
        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, str) and v in (
                    "w", "he", "so", "e", "h", "ln", "c", "f", "t", "sh", "th", "n"):
                scene_of[k] = v
        print(f"[RELINK-ALL] 分类批{gi}: 分类 {len(mapping)} 节点")
    unclassified = [nid for nid in active if nid not in scene_of]
    for nid in unclassified:
        scene_of[nid] = "n"
    print(f"[RELINK-ALL] 场景分类完成: {len(scene_of)} 节点")
    from collections import Counter
    print(f"[RELINK-ALL] 分布: {dict(Counter(scene_of.values()))}")

    # ---- Step B：按场景分批联想连边 ----
    by_scene = {}
    for nid in active:
        by_scene.setdefault(scene_of[nid], []).append(nid)

    batch_log = []
    total_created = 0
    for scene, nids in by_scene.items():
        if len(nids) <= 1:
            continue
        lines = []
        for nid in nids:
            n = graph.get_node(nid)
            tv = n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])
            lines.append(f"[{nid} {tv}] {n['content'][:120]}")
        nodes_text = "\n".join(lines)

        edge_lines = []
        seen = set()
        nid_set = set(nids)
        for a, b, data in graph.graph.edges(data=True):
            if a in nid_set and b in nid_set:
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    edge_lines.append(f"{a} --{data.get('rel_type','')}--> {b}")
        edges_text = "\n".join(edge_lines) if edge_lines else "（无）"

        prompt = relink_prompt.format(nodes_text=nodes_text, edges_text=edges_text)
        raw = llm.invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            edge_ops = json.loads(raw).get("edge_ops", [])
        except json.JSONDecodeError:
            print(f"[RELINK-ALL] {scene}: JSON 解析失败，跳过")
            edge_ops = []
        edge_ops = [op for op in edge_ops
                    if isinstance(op, dict) and op.get("action") in ("create", "delete")
                    and op.get("from") and op.get("to")]
        if not edge_ops:
            print(f"[RELINK-ALL] {scene}: 无有效 op")

        result = builder.apply_ops([], edge_ops)
        created = sum(1 for op in edge_ops if op.get("action") == "create")
        total_created += created
        gh, go = gold_stats(graph)
        batch_log.append({
            "scene": scene,
            "nodes": len(nids),
            "ops": len(edge_ops),
            "created": created,
            "errors": result.get("errors", []),
            "edges_total": graph.edge_count,
            "gold_hit": gh,
            "gold_overlap": go,
        })
        print(f"[RELINK-ALL] {scene}（{len(nids)}节点）-> create={created} "
              f"(累计 {graph.edge_count} 边, gold命中 {gh} / 重合 {go:.1%})")

    gh, go = gold_stats(graph)
    out = {"meta": {"initial_nodes": len(active), "initial_edges": init_edges,
                    "classify_batch": args.classify_batch, "mode": mode},
           "scene_of": scene_of,
           "round_log": batch_log,
           "total_created": total_created,
           "final": {"edges": graph.edge_count, "gold_hit": gh, "gold_overlap": go}}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    graph_path = ROOT / args.out_graph
    graph_path.write_text(yaml.safe_dump(graph.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"\n=== 遍历联想连边（内容分类）完成 ===")
    print(f"新增边: {total_created}，边: {init_edges} -> {graph.edge_count}")
    print(f"gold 命中: {gh}（重合率 {go:.1%}）")
    print(f"导出: {graph_path}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
