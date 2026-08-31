# SPDX-License-Identifier: AGPL-3.0-only

"""
LLM DBA：记忆图谱的自主维护者

每次对话后异步触发，LLM 收到三层上下文（对话、记忆、结构），
自主决定对图谱执行 CRUD 操作（新增/更新/修正/废弃节点，新增/删除边）。
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ---- Step 1：节点抽取 Prompt（2026-08-22 起，节点抽取与边连接拆成两步）----

NODE_EXTRACTION_PROMPT = """你是记忆图谱的「节点抽取器」：从对话中抽取独立事实——既包括关于「用户」的事实，也包括「助手」自身经历/行为的事实，只输出节点维护操作，不负责连边。

## 节点类型（6）
STATUS=客观处境 | REASON=导致状态/行为的原因 | ACTION=主动行为 | THING=物品/地点 | PERSON=社交关系人 | EMOTION=主观情绪
边界：客观处境→STATUS/主观感受→EMOTION；用户做的事→ACTION/导致原因→REASON；物品本身→THING/互动→ACTION

## 操作
create(新事实) | update(事实变化) | fix_type(修正类型) | deprecate(标记过时,不物理删除)

## 抽取原则（重要）
1. 只抽事实，来源有二：用户的事实、助手自身经历/行为（助手为响应用户请求实际做过的事，如「助手帮用户整理了一份项目清单」「助手用Python写了个排序示例」）；忽略无信息量的寒暄/客套/共情
2. 一个事实一个节点：一句话含多个事实时必须拆成多个节点，禁止合并
   （如「加班又总吃外卖」→ 加班节点 + 外卖节点；「小王一起加班还帮分担任务」→ 两人同加班 + 小王帮分担 两个节点）
3. content 为消解指代、补全省略后的完整陈述：指用户时转成「用户」，指助手自身时转成「助手」
4. 保留时间维度：区分长期习惯（「用户每天晨跑半小时」）与单次事件（「用户今早跑了一次步」），不得混淆
5. 与已有节点语义重复时用 update 更新而非新建；与已有节点矛盾时 deprecate 旧节点 + create 新节点
6. 每条独立可读

## 输出
严格输出 JSON：{{"node_ops":[{{"action":"create","content":"...","node_type":"ACTION"}},{{"action":"update","target_id":"n3","content":"...","reason":"..."}},{{"action":"fix_type","target_id":"n7","node_type":"STATUS","reason":"..."}},{{"action":"deprecate","target_id":"n3","reason":"..."}}]}}
无需抽取时返回空 node_ops。只输出 JSON。"""

NODE_EXTRACTION_USER_PROMPT = """── 对话上下文 ──

{conversation}

── 已有相关节点 ──

{current_nodes}

── 请输出节点维护操作 ──"""


# ---- Step 2：边连接 Prompt（与节点抽取拆分，2026-08-22）----

EDGE_LINKING_PROMPT = """你是记忆图谱的「边连接器」：节点已经抽取完成，你只负责为节点建立语义关系边。

## 边类型（8）
CAUSAL=因→果 | SCENARIO=同场景 | SEQUENCE=先→后 | PREFERENCE=偏好 | SOCIAL=社交 | ATTRIBUTE=属性 | TEMPORAL=事件→时间 | TAXONOMIC=子→父
方向：CAUSAL/SEQUENCE/PREFERENCE/TEMPORAL/TAXONOMIC 单向；SCENARIO/SOCIAL/ATTRIBUTE 双向
边界：A不发生B还会发生?不会=CAUSAL,可能=SEQUENCE；B是环境=SCENARIO/自身属性=ATTRIBUTE；喜欢=PREFERENCE/需要=CAUSAL

## 操作
create(连边) | delete(删错误/过时的边)

## 连边原则（激进版：宁可多连，不可漏连）
1. **尽量全连**：同一场景内（工作/健康/社交/情绪/学生时代）所有存在关系的事实之间都要连边——场景内高相关节点是检索枢纽，应连接尽可能多的节点（如「后端开发」连接项目加班、同事、主管、工作年限、换工作意向等）
2. **跨场景大胆桥接**：工作→健康→情绪→社交 之间，只要找得到因果/场景/时序/偏好/人物关系之一就连——跨场景联想是检索的价值所在
3. 连接「本轮新节点」与「已有相关节点」：新事实必须尽量接入已有图谱结构，能连则连
4. 边类型要准确：喜欢/偏好→PREFERENCE、自身属性→ATTRIBUTE、同场景并存→SCENARIO、先后来→SEQUENCE、因果→CAUSAL、人物关系→SOCIAL
5. 保留时间维度：长期习惯与单次事件按语义连（CAUSAL/SEQUENCE）
6. 边审计：删除已过时/被新事实覆盖的边；本轮未提到但可能仍成立的保留

## 输出
严格输出 JSON：{{"edge_ops":[{{"action":"create","from":"n1","to":"n2","rel_type":"CAUSAL"}},{{"action":"delete","from":"n3","to":"n5","reason":"..."}}]}}
无需连边时返回空 edge_ops。只输出 JSON。"""

EDGE_LINKING_FEWSHOT_EXAMPLE = """参考示例（激进连边模式）：

示例 1 —— 同一场景内全连 + 跨场景也连（工作场景）：

对话：
user(09:00): 今天又要加班，这个项目真的排得太满了，天天搞到十一二点。
user(09:05): 我在这家互联网公司做后端开发，项目一上线就得赶进度。脖子又酸了，加班太多颈椎扛不住。
user(09:10): 同事小王跟我一样惨，天天一起加班，不过他有时候会帮我分担点任务。下午得靠咖啡续命，最爱拿铁。

本轮新抽取的节点：
[n1 STATUS] 用户在一家互联网公司做后端开发
[n2 ACTION] 用户项目上线时经常加班到十一二点
[n3 STATUS] 用户加班导致颈椎经常酸痛
[n5 ACTION] 用户喜欢喝咖啡提神
[n6 THING] 拿铁是用户最常点的咖啡
[n8 PERSON] 同事小王经常和用户一起加班

正确输出（场景内全连 + 跨场景桥接）：
{{"edge_ops":[
  {{"action":"create","from":"n1","to":"n2","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n1","to":"n8","rel_type":"SOCIAL"}},
  {{"action":"create","from":"n1","to":"n3","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n2","to":"n3","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n2","to":"n5","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n2","to":"n8","rel_type":"SOCIAL"}},
  {{"action":"create","from":"n5","to":"n6","rel_type":"PREFERENCE"}},
  {{"action":"create","from":"n3","to":"n5","rel_type":"SCENARIO"}}
]}}

示例 2 —— 跨场景大胆桥接：

对话：
user(20:00): 复查结果出来了，指标降了，还挺欣慰的。
user(20:07): 就是戒不掉烧烤，那家店的烤羊排真是我的最爱。

本轮新抽取的节点：
[n20 ACTION] 用户每天早上绕小区晨跑半小时
[n27 EMOTION] 用户看到复查指标下降很欣慰
[n40 THING] 烧烤店的烤羊排是用户的最爱

已有相关节点：
[n2 ACTION] 用户项目上线时经常加班到十一二点
[e0 STATUS] 用户今天心情很低落

正确输出（晨跑接入工作/情绪/饮食多个场景）：
{{"edge_ops":[
  {{"action":"create","from":"n2","to":"e0","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n20","to":"n27","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n2","to":"n20","rel_type":"SEQUENCE"}},
  {{"action":"create","from":"n20","to":"e0","rel_type":"CAUSAL"}},
  {{"action":"create","from":"n20","to":"n40","rel_type":"SCENARIO"}}
]}}"""

EDGE_LINKING_USER_PROMPT = """── 对话上下文 ──

{conversation}

── 本轮新抽取的节点 ──

{new_nodes}

── 相关已有节点 ──

{current_nodes}

已有边：
{current_edges}

── 结构上下文（相关节点一跳邻居）──

{neighbor_info}

图谱概况：总节点 {total_nodes} 个，总边 {total_edges} 条

── 请输出边维护操作 ──"""


# ---- 时序叙事（TEMPORAL）增强：仅在“时序叙事守卫”命中时叠加，避免常规语料过度拆分 ----

NODE_TEMPORAL_EXTRA = """
7. 时间节点单独抽取：对话中出现的明确时间/时段（如「上周一晚上」「周二」「周三凌晨」「这周」）要抽成独立的 THING 节点，content 就是该时间词本身；不要把时间写进事件 content。一个事件若发生在某个时间，需同时抽出「事件节点」和「时间节点」两个节点。"""

EDGE_TEMPORAL_EXTRA = """
8. 务必连 TEMPORAL 边（事件→时间，单向）：本轮新抽出的事件节点，若其发生时间也是一个已抽取的时间节点，用 TEMPORAL 单向连边（from=事件, to=时间）。事件之间的先后关系用 SEQUENCE（先→后），不要与「时间定位」（TEMPORAL）混为一谈：SEQUENCE 是事件 A 在事件 B 之前；TEMPORAL 是事件挂在某个时间锚点上。"""

EDGE_TEMPORAL_EXAMPLE = """

示例 3 —— TEMPORAL 事件→时间 + SEQUENCE 事件先后：

对话：
user(周一 20:30): 上周一晚上临时又甩过来一个紧急需求。
user(周二 13:20): 这周二白天我一直在赶代码。
user(周三 01:20): 周三凌晨上线，结果出了个 P0 故障。

本轮新抽取的节点（事件节点与时间节点分开）：
[e0 ACTION] 用户接到紧急需求
[e1 ACTION] 用户加班赶代码
[e2 STATUS] 项目上线出P0故障
[t1 THING] 上周一晚上
[t2 THING] 上周二白天
[t3 THING] 上周三凌晨

正确输出（事件→时间用 TEMPORAL；事件之间先后用 SEQUENCE；因果用 CAUSAL）：
{{"edge_ops":[
  {{"action":"create","from":"e0","to":"t1","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e1","to":"t2","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e2","to":"t3","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e0","to":"e1","rel_type":"SEQUENCE"}},
  {{"action":"create","from":"e1","to":"e2","rel_type":"SEQUENCE"}},
  {{"action":"create","from":"e0","to":"e2","rel_type":"CAUSAL"}}
]}}"""


TIME_PERIOD_RE = re.compile(
    r"昨天|明天|前天|上周|这周|本周|下周|周[一二三四五六日天]|"
    r"\d{1,2}月(\d{1,2}[日号])?|\d{1,2}号|\d{1,2}日|"
    r"去年|今年|明年|前年|寒假|暑假|开学|年底|年初|年初|"
    r"春天|夏天|秋天|冬天|上半年|下半年"
)


def has_temporal_signal(text: str) -> bool:
    """时序叙事守卫：仅当出现 >=2 个**不同的日期/时期词**（昨天/上周/周三/这周/去年等）时判定为时序叙事。

    刻意排除「凌晨/中午/晚上/每天/近来/有时」这类日常时间副词，避免常规生活语料过度拆分；
    也仅凭「高中/大学」等 era 词**不**触发——它们常出现在非时序的设定陈述里（如「大学时学的钢琴」），
    单靠 era 会让大量常规语料误触发（0822 子集即如此）。真正的时序叙事靠「多个具体日期锚点」识别。
    """
    if not text:
        return False
    periods = set(m.group(0) for m in TIME_PERIOD_RE.finditer(text))
    return len(periods) >= 2


# ---- 维护判断（triage）Prompt ----

DBA_TRIAGE_PROMPT = """判断对话是否包含值得长期记忆的新事实或助手自身经历/行为，只回答 NEEDED 或 SKIP。

SKIP：纯寒暄/打招呼、共情安慰、追问细节无新信息、仅延续话题。
NEEDED：用户的新事实、事实变化、与已有记忆矛盾、话题切换；或助手做了值得记住的具体产出/行为（如生成文件、写代码、整理清单、给出可执行建议），这类「助手做过的事」同样需要记录。

对话：
{conversation}

回答（NEEDED 或 SKIP）："""


# ---- DBA 类 ----

class MemoryDBA:
    """记忆图谱的 LLM 数据库管理员"""

    def __init__(
        self,
        llm: ChatOpenAI,
        graph: MemoryGraph,
        vector_store: VectorStore,
        graph_builder: GraphBuilder,
        current_state_k: int = 15,
    ):
        """
        Args:
            llm: LangChain ChatOpenAI 实例
            graph: 目标 MemoryGraph
            vector_store: 向量存储
            graph_builder: GraphBuilder 实例
            current_state_k: 记忆上下文检索多少条相关节点
        """
        self.llm = llm
        self.graph = graph
        self.vector_store = vector_store
        self.builder = graph_builder
        self.current_state_k = current_state_k

        # 组装 Prompt 链：节点抽取与边连接拆成两步（2026-08-22）
        self.node_prompt = ChatPromptTemplate.from_messages([
            ("system", NODE_EXTRACTION_PROMPT),
            ("human", NODE_EXTRACTION_USER_PROMPT),
        ])
        self.node_chain = self.node_prompt | self.llm
        self.edge_prompt = ChatPromptTemplate.from_messages([
            ("system", EDGE_LINKING_PROMPT),
            ("human", EDGE_LINKING_FEWSHOT_EXAMPLE),
            ("human", EDGE_LINKING_USER_PROMPT),
        ])
        self.edge_chain = self.edge_prompt | self.llm

        # 时序叙事增强链：仅当 has_temporal_signal(conversation) 命中才启用，
        # 把时间节点抽取与 TEMPORAL 连边规则叠加到标准 prompt，避免常规语料过度拆分。
        self.node_chain_temporal = ChatPromptTemplate.from_messages([
            ("system", NODE_EXTRACTION_PROMPT.replace("\n## 输出", NODE_TEMPORAL_EXTRA + "\n\n## 输出")),
            ("human", NODE_EXTRACTION_USER_PROMPT),
        ]) | self.llm
        self.edge_chain_temporal = ChatPromptTemplate.from_messages([
            ("system", EDGE_LINKING_PROMPT.replace("\n## 输出", EDGE_TEMPORAL_EXTRA + "\n\n## 输出")),
            ("human", EDGE_LINKING_FEWSHOT_EXAMPLE + EDGE_TEMPORAL_EXAMPLE),
            ("human", EDGE_LINKING_USER_PROMPT),
        ]) | self.llm

        # 维护判断前置（triage）：先用极小 prompt 判断是否值得维护
        self.triage_chain = ChatPromptTemplate.from_messages([
            ("system", DBA_TRIAGE_PROMPT),
        ]) | self.llm

    # ---- 主入口 ----

    def maintain(
        self,
        conversation: str,
    ) -> Dict:
        """对话完成后执行一次数据库维护（两步：节点抽取 → 边连接）

        Args:
            conversation: 本轮对话的完整文本（含角色和时间戳）

        Returns:
            {
                "ops": {"node_ops": [...], "edge_ops": [...]},  # LLM 原始输出
                "result": {"created_ids": [...], "skipped": [...], "errors": [...]},  # 执行结果
                "context": {...},  # 组装上下文（调试用）
            }
        """
        # 0. 维护判断前置：明显无需维护的对话直接跳过，省掉完整 prompt
        if not self._should_maintain(conversation):
            logger.info("DBA 维护: triage 判定跳过")
            return {
                "ops": {"node_ops": [], "edge_ops": []},
                "result": {"created_ids": [], "skipped": [], "errors": []},
                "context": None,
                "skipped": True,
            }

        # Step 1:节点抽取(对话 + 相关旧节点 → node_ops → 执行)。时序叙事守卫命中则叠加 TEMPORAL 时间节点规则。
        temporal = has_temporal_signal(conversation)
        node_context = self._build_node_context(conversation)
        node_chain = self.node_chain_temporal if temporal else self.node_chain
        node_ops = self._parse_response(node_chain.invoke(node_context).content).get("node_ops", [])
        result1 = self.builder.apply_ops(node_ops, [])

        # Step 2:边连接(对话 + 本轮新节点 + 相关旧节点/一跳邻居/已有边 → edge_ops → 执行)。守卫同样驱动 TEMPORAL 连边规则与示例。
        new_ids = result1.get("created_ids", [])
        edge_context = self._build_edge_context(conversation, new_ids)
        edge_chain = self.edge_chain_temporal if temporal else self.edge_chain
        edge_ops = self._parse_response(edge_chain.invoke(edge_context).content).get("edge_ops", [])
        result2 = self.builder.apply_ops([], edge_ops)

        logger.info(
            f"DBA 维护完成: 节点创建 {len(result1.get('created_ids', []))} 更新 {len(node_ops)} "
            f"边创建/删除 {len(edge_ops)} (累计 {self.graph.node_count} 节点 / {self.graph.edge_count} 边)"
        )

        return {
            "ops": {"node_ops": node_ops, "edge_ops": edge_ops},
            "result": {
                "created_ids": result1.get("created_ids", []),
                "skipped": result1.get("skipped", []) + result2.get("skipped", []),
                "errors": result1.get("errors", []) + result2.get("errors", []),
            },
            "context": {"node": node_context, "edge": edge_context},
        }

    # ---- 上下文组装 ----

    def _build_context(self, conversation: str) -> Dict:
        """组装三层上下文"""
        # 第一层：对话上下文（直接使用传入的完整对话）
        conversation_text = conversation

        # 第二层：记忆上下文（语义检索 + 已有边）
        current_nodes_text, current_edges_text = self._build_memory_context(conversation)

        # 第三层：结构上下文（一跳邻居 + 图谱统计）
        neighbor_text = self._build_structure_context(conversation)

        return {
            "conversation": conversation_text,
            "current_nodes": current_nodes_text,
            "current_edges": current_edges_text,
            "neighbor_info": neighbor_text,
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
        }

    def _build_node_context(self, conversation: str) -> Dict:
        """Step 1 节点抽取上下文：对话 + 相关旧节点（用于去重/更新判断）"""
        if self.graph.node_count == 0:
            nodes_text = "（暂无已有记忆）"
        else:
            nodes_text, _ = self._build_memory_context(conversation)
        return {
            "conversation": conversation,
            "current_nodes": nodes_text,
        }

    def _build_edge_context(self, conversation: str, new_ids: List[str]) -> Dict:
        """Step 2 边连接上下文：对话 + 本轮新节点(全) + 相关旧节点 + 一跳邻居 + 已有边"""
        if self.graph.node_count == 0:
            current_nodes_text = current_edges_text = "（暂无已有记忆）"
        else:
            current_nodes_text, current_edges_text = self._build_memory_context(conversation)
        neighbor_text = self._build_structure_context(conversation)

        new_lines = []
        for nid in new_ids:
            node = self.graph.get_node(nid)
            if node is None:
                continue
            type_val = node["node_type"].value if hasattr(node["node_type"], "value") else str(node["node_type"])
            new_lines.append(f"  [{nid}] {type_val}: {node['content'][:120]}")
        new_nodes_text = "\n".join(new_lines) if new_lines else "（本轮无新节点）"

        return {
            "conversation": conversation,
            "new_nodes": new_nodes_text,
            "current_nodes": current_nodes_text,
            "current_edges": current_edges_text,
            "neighbor_info": neighbor_text,
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
        }

    def _build_memory_context(self, conversation: str) -> Tuple[str, str]:
        """构建记忆上下文：语义相关节点 + 已有边"""
        if self.graph.node_count == 0:
            return "（暂无已有记忆）", "（暂无已有边）"

        results = self.vector_store.search(conversation, k=self.current_state_k)

        node_lines = []
        shown_ids = set()
        for mid, score, _ in results:
            node = self.graph.get_node(mid)
            if node is None or node.get("deprecated"):
                continue
            type_val = node["node_type"].value if hasattr(node["node_type"], "value") else str(node["node_type"])
            content = node["content"][:100]
            node_lines.append(f"  [{mid}] {type_val}: {content}")
            shown_ids.add(mid)

        # 收集这些节点之间的边
        edge_lines = []
        shown_edges = set()
        for u in shown_ids:
            for v in shown_ids:
                if u == v:
                    continue
                if self.graph.graph.has_edge(u, v):
                    edge_key = (u, v)
                    if edge_key in shown_edges:
                        continue
                    shown_edges.add(edge_key)
                    edge_data = self.graph.graph.edges[u, v]
                    rel_type = edge_data.get("rel_type")
                    rel_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                    edge_lines.append(f"  {u} --[{rel_str}]--> {v}")

        nodes_text = "\n".join(node_lines) if node_lines else "（未找到相关已有记忆）"
        edges_text = "\n".join(edge_lines) if edge_lines else "（相关节点间暂无直接边）"

        return nodes_text, edges_text

    def _build_structure_context(self, conversation: str) -> str:
        """构建结构上下文：相关节点的一跳邻居"""
        if self.graph.node_count == 0:
            return "（暂无图谱结构）"

        results = self.vector_store.search(conversation, k=5)
        neighbor_set = set()
        for mid, _, _ in results:
            node = self.graph.get_node(mid)
            if node is None or node.get("deprecated"):
                continue
            for neighbor_id, rel_type, is_reverse in self.graph.get_neighbors(mid):
                neighbor_node = self.graph.get_node(neighbor_id)
                if neighbor_node and not neighbor_node.get("deprecated"):
                    direction = "←" if is_reverse else "→"
                    rel_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                    neighbor_set.add(f"  {mid} --{direction}[{rel_str}]-- {neighbor_id}")

        if neighbor_set:
            return "\n".join(sorted(neighbor_set)[:20])
        return "（相关节点暂无邻居）"

    # ---- LLM 调用 ----

    def _should_maintain(self, conversation: str) -> bool:
        """维护判断前置：极小 prompt 判断是否值得维护"""
        snippet = conversation[-500:]
        resp = self.triage_chain.invoke({"conversation": snippet})
        return "NEEDED" in resp.content.upper()

    # ---- 响应解析 ----

    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 输出，提取 node_ops 和 edge_ops"""
        # 尝试直接解析 JSON
        cleaned = response.strip()

        # 去掉可能的 markdown 代码块包裹
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # 去掉第一行 ```json 和最后一行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            ops = json.loads(cleaned)
            return {
                "node_ops": ops.get("node_ops", []),
                "edge_ops": ops.get("edge_ops", []),
            }
        except json.JSONDecodeError:
            logger.warning(f"LLM 输出 JSON 解析失败，原始响应前 200 字符: {response[:200]}")

            # 尝试提取 JSON 片段
            return self._extract_json_fallback(response)

    def _extract_json_fallback(self, response: str) -> Dict:
        """从非标准格式的 LLM 输出中尝试提取 JSON"""
        # 尝试找到第一个 { 和最后一个 }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        logger.error(f"无法从 LLM 输出中提取有效的 JSON 操作指令")
        return {"node_ops": [], "edge_ops": []}

    # ---- 检查点 ----

    def save_checkpoint(self, save_dir: str):
        """保存完整检查点（图谱 + 向量 + 构建器状态）"""
        import yaml
        import json
        import os
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)

        # 1. 图谱 → YAML
        graph_path = os.path.join(save_dir, "memory_graph.yaml")
        with open(graph_path, "w", encoding="utf-8") as f:
            yaml.dump(self.graph.to_dict(), f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)

        # 2. 向量存储
        vs_path = os.path.join(save_dir, "faiss_index")
        if self.vector_store and self.vector_store.has_vectors():
            self.vector_store.save(vs_path)

        # 3. 构建器状态 → JSON
        builder_path = os.path.join(save_dir, "builder_state.json")
        with open(builder_path, "w", encoding="utf-8") as f:
            json.dump(self.builder.save_state(), f, ensure_ascii=False)

        # 4. 检查点元数据
        checkpoint_path = os.path.join(save_dir, "checkpoint.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1",
                "timestamp": datetime.now().isoformat(),
                "nodes": self.graph.node_count,
                "edges": self.graph.edge_count,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"检查点已保存: {save_dir} ({self.graph.node_count} 节点, {self.graph.edge_count} 边)")
        return save_dir

    def restore_checkpoint(self, save_dir: str) -> bool:
        """从检查点恢复全部状态"""
        import yaml
        import json
        import os

        # 1. 图谱
        graph_path = os.path.join(save_dir, "memory_graph.yaml")
        if os.path.exists(graph_path):
            restored = MemoryGraph.from_dict(
                yaml.safe_load(open(graph_path, encoding="utf-8"))
            )
            self.graph.graph = restored.graph  # 替换内部 nx 图
            logger.info(f"图谱已恢复: {self.graph.node_count} 节点, {self.graph.edge_count} 边")
        else:
            logger.warning("未找到图谱检查点")

        # 2. 向量存储
        vs_path = os.path.join(save_dir, "faiss_index")
        if os.path.exists(vs_path):
            self.vector_store.load(vs_path, embeddings=self.vector_store.embeddings)

        # 3. 构建器状态
        builder_path = os.path.join(save_dir, "builder_state.json")
        if os.path.exists(builder_path):
            with open(builder_path, encoding="utf-8") as f:
                self.builder.load_state(json.load(f))
            logger.info("构建器状态已恢复")

        return True
