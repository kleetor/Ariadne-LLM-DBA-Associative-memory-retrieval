# SPDX-License-Identifier: AGPL-3.0-only

"""
完整检索入口：串联跳转轴 + 目的回归 + 寻峰终止

论文第六章 Algorithm: Purpose-Driven Associative Retrieval
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

from dba_pipeline.core.jump_axis import get_jump_weight, NodeType, RelationType
from dba_pipeline.core.purpose import PurposeModel
from dba_pipeline.core.peak_find import PeakFinder
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.llm.inference import InferenceEngine
from dba_pipeline.embedding.store import VectorStore


# 独立时序工具 temporal_lookup 用的时间锚点判定：覆盖绝对时期 + 指示词 + 人生阶段。
TOOL_TIME_ANCHOR_RE = re.compile(
    r"^[这上本下前那]?周|昨天|今天|明天|前天|去年|今年|明年|前年|上个月|这个月|下个月|"
    r"\d{1,4}年|\d{1,2}月|\d{1,2}号|\d{1,2}日|周[一二三四五六日天]|"
    r"凌晨|早上|上午|中午|下午|晚上|那段时间|当时|近年来|"
    r"高中|大学|大专|工作|毕业|入职|暑假|寒假|上学期|下学期|上半年|下半年|年底|年初|"
    r"高三|高二|高一|大四|大三|大二|大一"
)


class PurposeDrivenRetriever:
    """目的驱动的联想记忆检索器（完整链路）"""

    def __init__(
        self,
        llm: ChatOpenAI,
        embeddings: Embeddings,
        graph: MemoryGraph,
        vector_store: VectorStore,
        inference: InferenceEngine,
        distance_decay: float = 0.85,
        jump_weight_coef: float = 0.5,
        purpose_weight_coef: float = 0.5,
        purpose_filter_threshold: float = 0.2,
        purpose_filter_decay: float = 0.0,
        path_tracker=None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.graph = graph
        self.vector_store = vector_store
        self.inference = inference

        self.purpose_model = PurposeModel(llm, self._embed)
        self.peak_finder = PeakFinder(patience=2, min_delta=0.015)

        self.distance_decay = distance_decay
        self.jump_weight_coef = jump_weight_coef
        self.purpose_weight_coef = purpose_weight_coef
        self.purpose_filter_threshold = purpose_filter_threshold
        self.purpose_filter_decay = purpose_filter_decay
        self.path_tracker = path_tracker  # 可选：PathTracker 实例

    def _get_hop_threshold(self, hop: int) -> float:
        """计算当前跳数的有效过滤阈值

        若 purpose_filter_decay > 0，使用递减公式：
            threshold_hop = purpose_filter_threshold × purpose_filter_decay^hop
        否则使用固定阈值。
        """
        if self.purpose_filter_decay > 0.0:
            return self.purpose_filter_threshold * (self.purpose_filter_decay ** hop)
        return self.purpose_filter_threshold

    def _embed(self, text: str) -> np.ndarray:
        return self.vector_store._embed(text)

    def _is_active(self, node_id: str) -> bool:
        """节点是否仍参与检索：必须存在，且未被废弃/遗忘。

        检索链路需与 DBA 维护（deprecate/forget）保持一致，避免过期或孤立记忆进入结果。
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return False
        if node.get("deprecated") or node.get("forgotten"):
            return False
        return True

    # ---- 独立单跳时序工具 temporal_lookup ----
    # 定位：独立查询工具，不进入主检索流程。按"时间→事件"反向单跳取共时事实，返回多锚点供上层 LLM 挑选。

    def _tool_is_time_node(self, memory_id: str) -> bool:
        """工具判定时间锚点：th/T+数字 前缀（数据集约定），或 THING 且内容含时间/时期词。"""
        if memory_id.startswith("th"):
            return True
        if len(memory_id) > 1 and memory_id[0] == "T" and memory_id[1:].isdigit():
            return True
        try:
            if self.graph.get_node_type(memory_id) != NodeType.THING:
                return False
        except Exception:
            return False
        content = self.graph.graph.nodes[memory_id].get("content", "")
        return bool(TOOL_TIME_ANCHOR_RE.search(content))

    def _has_reverse_temporal(self, memory_id: str) -> bool:
        """是否为"真时间锚点"：存在反向 TEMPORAL 邻居（有事件锚定到它）。"""
        try:
            for nb, rel_type, is_reverse in self.graph.get_neighbors(memory_id):
                if rel_type == RelationType.TEMPORAL and is_reverse:
                    return True
        except Exception:
            return False
        return False

    def temporal_lookup(
        self,
        query: str,
        k_seed: int = 8,
        max_facts: int = 8,
        max_anchors: int = 3,
    ) -> Dict:
        """独立单跳时序工具——**仅做"时间→事件"反向**（主检索器反向权重为 0、做不到的部分）。

        语义匹配到时间锚点，沿 TEMPORAL 边**反向**取共时事件。
        · 不做正向（事件→时间，那是主检索器的活）、不链式扩展、不做目的打分挤占、不跨期漂移。
        · 查询可能命中多个时间锚点（如"三年"可指大专三年/工作三年），**返回全部候选锚点的事实组，
          交由上层 LLM 判断最相关的一个/多个**，而不是硬性消歧。
        · 只返回"真锚点"（带反向 TEMPORAL 邻居）的事实组；无可用锚点时返回空 matches。

        Returns:
            {"query", "matches":[{"time_anchor":{"id","content"}, "facts":[{"id","content","from"}...]}...],
             "count"}
        """
        try:
            hits = self.vector_store.search(query, k=k_seed)  # [(id, score, meta)]
        except Exception:
            hits = []
        matched = [(h[0], h[1]) if len(h) >= 2 else (h[0], 1.0) for h in hits]

        # 只定位"时间锚点"（时间推事件型查询的锚必须是时间节点）；跳过废弃/遗忘节点
        time_cands = [m for m in matched if self._is_active(m[0]) and self._tool_is_time_node(m[0])]
        # 过滤真锚点：只有带反向 TEMPORAL 邻居的才是时间锚点（排除误判的"含时间词事件节点"）。
        usable = [m for m in time_cands if self._has_reverse_temporal(m[0])]

        matches: List[Dict] = []
        for m in usable[:max_anchors]:
            anchor_id = m[0]
            anchor_content = self.graph.get_content(anchor_id) or ""
            # 时间 → 共时事件（反向，单跳）
            facts: List[Dict] = []
            seen = set([anchor_id])
            for nb, rel_type, is_reverse in self.graph.get_neighbors(anchor_id):
                if rel_type == RelationType.TEMPORAL and is_reverse and nb not in seen:
                    if not self._is_active(nb):
                        continue
                    seen.add(nb)
                    facts.append({"id": nb, "content": self.graph.get_content(nb) or "", "from": anchor_id})
            if max_facts:
                facts = facts[:max_facts]
            matches.append({"time_anchor": {"id": anchor_id, "content": anchor_content}, "facts": facts})

        return {"query": query, "matches": matches, "count": len(matches)}

    # ---- 算法主流程 ----

    def retrieve(
        self,
        query: str,
        seed_k: int = 5,
        max_hops: int = 5,
        expand_k: int = None,
        purpose: Optional[List[str]] = None,
    ) -> Dict:
        """执行完整的目的驱动联想检索

        Args:
            query: 用户查询
            seed_k: 种子数量
            max_hops: 最大跳数
            expand_k: 每轮扩展数（默认=seed_k）
            purpose: 可选，调用方（主聊天 LLM/agent）注入的目的列表。
                     提供时跳过独立的目的推断（infer_purpose），检索意图由主 LLM
                     在对话上下文理解中给出；None 时退化到系统内部推断。

        Returns:
            {
                "peak_memories": [(memory_id, content), ...],
                "peak_scores": {memory_id: combined_score, ...},
                "purpose": {"status": str, "purposes": [str]},
                "hop_history": [{"hop": int, "candidates": [...], "mean_score": float}, ...],
                "total_candidates": int,
            }
        """
        # Step 1: 推断状态和目的（可注入：主 LLM 提供则跳过独立推断）
        if purpose is not None:
            purpose_info = {"status": "injected", "purposes": list(purpose)}
        else:
            purpose_info = self.inference.infer_purpose(query)
        purposes = purpose_info.get("purposes", [])
        purpose_vec = self.purpose_model.get_purpose_vector(purposes)

        # Step 2: 向量搜索种子记忆（混合 query text + 目的向量）
        half_k = max(1, seed_k // 2)
        text_seeds = self.vector_store.search(query, k=half_k)
        purpose_seeds = self.vector_store.search_by_vector(purpose_vec, k=seed_k - half_k)

        # 合并去重，保持 text 种子优先顺序，purpose 种子补充；跳过废弃/遗忘节点
        seen = set()
        seed_ids = []
        for r in text_seeds:
            if r[0] not in seen and self._is_active(r[0]):
                seen.add(r[0])
                seed_ids.append(r[0])
        for r in purpose_seeds:
            if r[0] not in seen and len(seed_ids) < seed_k and self._is_active(r[0]):
                seen.add(r[0])
                seed_ids.append(r[0])
        # 不足时用更多 text 种子补齐
        if len(seed_ids) < seed_k:
            extra = self.vector_store.search(query, k=seed_k * 2)
            for r in extra:
                if r[0] not in seen and len(seed_ids) < seed_k and self._is_active(r[0]):
                    seen.add(r[0])
                    seed_ids.append(r[0])
        # 计算种子轮目的关联度（使用预缓存向量，无 API 调用）
        seed_vectors = self.vector_store.get_content_vectors(seed_ids)
        seed_scores = self.purpose_model.compute_purpose_score_from_vectors(
            seed_vectors, purpose_vec
        )
        # 种子轮没有跳转轴扩展（jw=0），使用与后续轮次一致的得分公式
        mu_0 = float(np.mean(seed_scores)) * self.purpose_weight_coef

        self.peak_finder.reset()
        decision = self.peak_finder.add_round(mu_0)
        if decision == "peak_found":
            return self._build_result(seed_ids, purpose_info, [])

        # Step 3-6: 循环扩展
        current_ids = list(seed_ids)
        seed_contents = self.graph.get_contents(seed_ids)
        hop_history: List[Dict] = [{
            "hop": 0,
            "candidates": [
                {"id": mid, "content": content, "purpose_score": float(s),
                 "combined_score": float(s) * self.purpose_weight_coef}
                for mid, content, s in zip(seed_ids, seed_contents, seed_scores)
            ],
            "mean_score": mu_0,
        }]

        for hop in range(1, max_hops + 1):
            # 跳转轴扩展（含可选 PathTracker 动态权重 + 边来源追踪）
            expanded = self.graph.expand_with_trace(current_ids, path_tracker=self.path_tracker)

            # 记录路径激活（供 PathTracker 分析）
            if self.path_tracker is not None:
                for seed_id in current_ids:
                    source_type = self.graph.get_node_type(seed_id)
                    if source_type is None:
                        continue
                    for neighbor_id, rel_type, is_reverse in self.graph.get_neighbors(seed_id):
                        if neighbor_id in seed_ids:
                            continue
                        w = get_jump_weight(source_type, rel_type, is_reverse)
                        if w > 0 and neighbor_id in expanded:
                            pk = self.path_tracker.make_path_key(
                                seed_id, neighbor_id, rel_type.value
                            )
                            self.path_tracker.record_activation(pk)
            if not expanded:
                result_ids = current_ids
                break

            # 目的过滤（使用预缓存向量，无 API 调用）；跳过废弃/遗忘节点
            candidate_ids = [mid for mid in expanded.keys() if self._is_active(mid)]
            candidate_contents = self.graph.get_contents(candidate_ids)
            candidate_vectors = self.vector_store.get_content_vectors(candidate_ids)
            candidate_scores = self.purpose_model.compute_purpose_score_from_vectors(
                candidate_vectors, purpose_vec
            )

            filtered: Dict[str, Dict] = {}
            # {mem_id: {content, jump_weight, from, rel_type, is_reverse, purpose_score}}
            hop_threshold = self._get_hop_threshold(hop)
            for i, mid in enumerate(candidate_ids):
                ps = float(candidate_scores[i])
                trace = expanded[mid]
                rel_type = trace["rel_type"]
                is_rev = trace["is_reverse"]
                if ps < hop_threshold:
                    continue
                jw = trace["weight"]
                filtered[mid] = {
                    "content": candidate_contents[i],
                    "jump_weight": jw,
                    "from": trace["from"],
                    "rel_type": rel_type,
                    "is_reverse": is_rev,
                    "purpose_score": ps,
                }

            if not filtered:
                result_ids = current_ids
                break

            # 距离衰减 + 组合得分
            decay = self.distance_decay ** hop
            combined = {}
            for mid, info in filtered.items():
                jw = info["jump_weight"]
                ps = info["purpose_score"]
                combined[mid] = {
                    "id": mid,
                    "content": info["content"],
                    "jump_weight": jw,
                    "purpose_score": ps,
                    "from": info["from"],
                    "rel_type": info["rel_type"],
                    "is_reverse": info["is_reverse"],
                    "combined_score": jw * decay * self.jump_weight_coef
                                      + ps * self.purpose_weight_coef,
                }

            # 当前轮综合得分均值（跳转轴 + 目的，用于寻峰）
            mu_hop = float(np.mean([v["combined_score"] for v in combined.values()]))

            # 寻峰判断
            decision = self.peak_finder.add_round(mu_hop)

            hop_history.append({
                "hop": hop,
                "candidates": sorted(
                    combined.values(),
                    key=lambda x: x["combined_score"],
                    reverse=True,
                ),
                "mean_score": mu_hop,
            })

            if decision == "peak_found":
                result_ids = self._collect_peak_tolerance(hop_history)
                break

            # 当前轮候选用作下一轮扩展
            top_k = expand_k if expand_k is not None else seed_k
            current_ids = [
                c["id"]
                for c in sorted(
                    combined.values(),
                    key=lambda x: x["combined_score"],
                    reverse=True,
                )[:top_k]
            ]

        else:
            # 达到 max_hops 仍未找到峰值，用峰值容忍带
            result_ids = self._collect_peak_tolerance(hop_history)

        return self._build_result(result_ids, purpose_info, hop_history)

    def _collect_peak_tolerance(self, hop_history: List[Dict]) -> List[str]:
        """峰值容忍带：取 μ ≥ peak_mean - tolerance 的所有轮候选"""
        peak_mean = self.peak_finder.history[self.peak_finder.peak_index]
        threshold = peak_mean - self.peak_finder.peak_tolerance
        result_ids = []
        seen = set()
        for h in hop_history:
            mu = h.get("mean_score", 0)
            if mu >= threshold:
                for c in h.get("candidates", []):
                    cid = c.get("id", "")
                    if cid not in seen:
                        seen.add(cid)
                        result_ids.append(cid)
        return result_ids

    def _build_result(
        self,
        result_ids: List[str],
        purpose_info: dict,
        hop_history: List[Dict],
    ) -> Dict:
        # 从 hop_history 收集每条的 combined_score
        id_score = {}
        for h in hop_history:
            for c in h.get("candidates", []):
                cid = c.get("id", "")
                cs = c.get("combined_score", 0)
                if cid not in id_score or cs > id_score[cid]:
                    id_score[cid] = cs

        memories = []
        for mid in result_ids:
            if not self._is_active(mid):
                continue
            content = self.graph.get_content(mid)
            if content:
                memories.append((mid, content))

        # 按 combined_score 降序排列（作为 LLM rerank 前的默认排序）
        memories.sort(key=lambda x: id_score.get(x[0], 0), reverse=True)
        return {
            "peak_memories": memories,  # [(id, content), ...] 保持向后兼容
            "peak_scores": id_score,    # {id: combined_score} 供 rerank 使用
            "purpose": purpose_info,
            "hop_history": hop_history,
            "total_candidates": len(memories),
        }

    # ---- StoryRank：链路理解 → 故事片段 ----

    @staticmethod
    def _collect_trace_nodes(hop_history: List[Dict]) -> Dict:
        """从 hop_history 提取节点信息与父子关系（树结构）

        Returns:
            {"parent": {child: parent}, "score": {id: combined_score},
             "content": {id: content}, "edge": {child: (rel_type, is_reverse)}}
        """
        parent = {}
        score = {}
        content = {}
        edge = {}
        seen = set()
        for h in hop_history:
            for c in h.get("candidates", []):
                cid = c.get("id")
                if not cid:
                    continue
                score[cid] = c.get("combined_score", 0)
                content[cid] = c.get("content", "")
                # 节点首次出现时才记录父边；回访（如双向边折返）不更新，避免 parent 成环
                if cid in seen:
                    continue
                seen.add(cid)
                frm = c.get("from")
                if frm is not None:
                    parent[cid] = frm
                    edge[cid] = (c.get("rel_type"), c.get("is_reverse", False))
        return {"parent": parent, "score": score, "content": content, "edge": edge}

    def select_core_nodes(
        self,
        hop_history: List[Dict],
        result_ids: List[str],
    ) -> List[List[str]]:
        """按连通性拆分核心节点组（粗筛）

        每个结果节点沿 from 回溯到种子（根），共享根的结果归为一组；
        组内节点按 hop 顺序（根 → 叶）排列。

        Returns:
            List[List[str]]：每个元素是一个片段的核心节点 id（从根到叶）
        """
        trace = self._collect_trace_nodes(hop_history)
        parent = trace["parent"]
        score = trace["score"]

        def find_root(nid: str) -> str:
            seen = set()
            while nid in parent and nid not in seen:
                seen.add(nid)
                nid = parent[nid]
            return nid

        # 按根分组结果节点
        groups: Dict[str, List[str]] = {}
        for rid in result_ids:
            if rid not in score:
                continue
            root = find_root(rid)
            groups.setdefault(root, []).append(rid)

        # 回溯收集每个分组的路径节点（结果 → 根），再按 hop 顺序排列（根 → 叶）
        core_groups: List[List[str]] = []
        for _, rids in groups.items():
            core = set()
            for rid in rids:
                nid = rid
                seen_loop = set()
                while nid is not None and nid not in seen_loop:
                    seen_loop.add(nid)
                    core.add(nid)
                    nid = parent.get(nid)
            ordered = []
            seen = set()
            for h in hop_history:
                for c in h.get("candidates", []):
                    cid = c.get("id")
                    if cid in core and cid not in seen:
                        seen.add(cid)
                        ordered.append(cid)
            if ordered:
                core_groups.append(ordered)

        return core_groups

    def _build_path(
        self,
        group: List[str],
        hop_history: List[Dict],
    ) -> Dict:
        """把核心节点组构建成 StoryRank 的 path 结构（nodes + edges）"""
        trace = self._collect_trace_nodes(hop_history)
        parent = trace["parent"]
        content = trace["content"]
        edge = trace["edge"]
        group_set = set(group)

        nodes = []
        for nid in group:
            nt = self.graph.get_node_type(nid)
            nt_val = nt.value if hasattr(nt, "value") else str(nt)
            nodes.append({
                "id": nid,
                "content": content.get(nid) or self.graph.get_content(nid) or "",
                "node_type": nt_val,
            })

        edges = []
        for nid in group:
            p = parent.get(nid)
            if p is not None and p in group_set:
                rel_type, is_reverse = edge.get(nid, (None, False))
                rel_val = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                edges.append({
                    "from": p,
                    "to": nid,
                    "rel_type": rel_val,
                    "is_reverse": is_reverse,
                })

        return {"nodes": nodes, "edges": edges}

    def retrieve_with_story(
        self,
        query: str,
        seed_k: int = 5,
        with_response: bool = True,
        max_hops: Optional[int] = None,
        expand_k: Optional[int] = None,
    ) -> Dict:
        """检索 → 连通性粗筛 → StoryRank 故事化 →（可选）生成回复

        把检索得到的记忆因果链路理解成故事片段，替代原 rerank 的扁平重排序。
        max_hops / expand_k 为 None 时沿用 retrieve() 的默认值。
        """
        retrieve_kwargs: Dict[str, Any] = {"seed_k": seed_k}
        if max_hops is not None:
            retrieve_kwargs["max_hops"] = max_hops
        if expand_k is not None:
            retrieve_kwargs["expand_k"] = expand_k
        result = self.retrieve(query, **retrieve_kwargs)
        peak_memories = result.get("peak_memories", [])
        hop_history = result.get("hop_history", [])
        result_ids = [mid for mid, _ in peak_memories]

        # 粗筛：按连通性拆分核心节点组
        core_groups = self.select_core_nodes(hop_history, result_ids)

        # 合并所有核心节点（去重、保持 hop 顺序），一次性生成一段完整故事
        all_nodes = []
        seen = set()
        for group in core_groups:
            for nid in group:
                if nid not in seen:
                    seen.add(nid)
                    all_nodes.append(nid)

        stories = []
        story_nodes = []
        discarded_nodes = []
        if all_nodes:
            path = self._build_path(all_nodes, hop_history)
            out = self.inference.story_rank(query, path)
            story = out.get("story", "")
            adopted = out.get("adopted_ids", [])
            if story:
                stories.append(story)
            story_nodes = adopted
            discarded_nodes = [nid for nid in all_nodes if nid not in adopted]

        result["stories"] = stories
        result["story_nodes"] = story_nodes
        result["discarded_nodes"] = discarded_nodes

        # 生成回复：聊天 LLM 只接收干净故事，替代 [id] content 列举
        if with_response:
            context_items = [f"[记忆片段] {s}" for s in stories]
            result["response"] = self.inference.generate_response(query, context_items)
        return result
