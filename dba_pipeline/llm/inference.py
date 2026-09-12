# SPDX-License-Identifier: AGPL-3.0-only

"""
LangChain LLM 封装：状态推断 + 目的推断 + 回复生成
"""

from datetime import datetime
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def format_ts_ms(ts) -> str:
    """Unix 毫秒 → 本地日期字符串；非整数/越界返回空串。

    注意语义：节点 timestamp 是**记录时间**（写入图谱的时刻），不是事件发生时间。
    渲染时必须显式标注「记录于」，否则会被误当成事件日期。
    """
    if not isinstance(ts, int) or isinstance(ts, bool):
        return ""
    try:
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""


# ---- 状态推断 Prompt ----

STATUS_INFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个心理状态分析助手。分析用户当前消息，推断其状态。

输出格式（严格 JSON）：
{{
    "status": "当前状态",
    "emotion": "情绪标签"
}}

状态示例：压力大、疲惫、开心、困惑、焦虑、放松
情绪示例：负面、正面、中性
只输出 JSON。"""),
    ("human", "{query}"),
])

# ---- 目的推断 Prompt ----

PURPOSE_INFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是对话意图分析助手。根据用户消息，先判断查询类型，再推断用户的目的。

查询类型：
- information：信息查询（询问事实/偏好/经历/细节，如「用户喜欢喝什么咖啡？」「他高考考得怎么样？」）
- emotional：情绪表达（倾诉、吐槽、寻求安慰、分享感受）

输出格式（严格 JSON）：
{{
    "status": "用户当前状态",
    "query_type": "information 或 emotional",
    "purposes": ["目的1", "目的2", "目的3"]
}}

规则：
- information 类型：目的必须是信息获取类（如「获取信息」「了解事实」「确认细节」「比较选择」「还原经历」），禁止社交/情绪类目的（如「社交互动」「建立关系」「倾诉」）
- emotional 类型：目的可以是「倾诉」「寻求建议」「理解原因」「寻求安慰」等
只输出 JSON。"""),
    ("human", "{query}"),
])

# ---- 记忆故事整理 StoryRank Prompt ----

_STORY_RANK_SYSTEM = """你是一个记忆故事整理助手。你会收到用户当前消息，以及一组「记忆节点与关系」（可能包含多条因果链路）。

你的任务：
1. 理解这些节点与关系：每个节点是一个记忆片段（带类型），边表示节点之间的关系（带关系类型）。
2. 把所有记忆整合成一段连贯、自然的故事，让读者能顺畅理解用户的状态与经历。只输出一段文字，不要分段列举。
3. 关系语义要自然融入句子，例如：
   - CAUSAL（因果）→ 因为 / 导致 / 所以（**只在边上确实标了 CAUSAL 时才可用**）
   - SEQUENCE（时序）→ 然后 / 之后 / 接着
   - PREFERENCE（偏好）→ 偏爱 / 喜欢
   - SCENARIO（场景）→ 在…场景下 / 常去
   - ATTRIBUTE（属性）→ 具有…特征
   - TEMPORAL（时间）→ 时间上
   - SOCIAL（社交）→ 与…有关
   - TAXONOMIC（分类）→ 属于 / 是一种
4. 因果要克制：只有边类型为 CAUSAL 的两点之间才写成因果；相邻、并列或仅有时序关系的节点，
   只能平铺成"同时/随后"，**不得自行推断出原因、动机或后果**。
5. 主体不能串：多方对话里，每条事实的动作主体必须与该节点内容一致。
   不得把甲的遭遇安到乙头上，也不得用"这也让某人…"把两个人的事件缝成一条因果。
6. 只采纳与链路连贯的记忆；明显突兀、与主线无关的节点应舍弃，不出现在故事中。
7. 忠实于节点内容：节点里没写的动机、原因、结果一律不补；宁可少写，不可写错。
8. adopted_ids 必须与 story 段落严格对应：正文里写到的每条事实，其节点都要列入
   adopted_ids；没写进正文的节点一律不得列入。二者不一致视为输出错误。

输出格式（严格 JSON）：
{{
    "story": "一段连贯的故事文本",
    "adopted_ids": ["被采纳进故事的节点id", "..."]
}}
只输出 JSON。"""

_STORY_RANK_HUMAN = """用户当前消息：
{query}

记忆因果链路：
{path}"""

# 仅当渲染记录时间时追加：时间戳是"写入图谱的时刻"，不是事件发生时间，必须显式区分。
_STORY_RANK_TS_RULE = """

9. 节点若标注「记录于 YYYY-MM-DD」，那是这条记忆的**记录时间**（写入图谱的时刻），
   **不是事件发生时间**。讲述事件经过时不要把它当作事情发生的日期；
   只有当用户问"什么时候记下的/什么时候知道的"时才可以引用它。"""

STORY_RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM),
    ("human", _STORY_RANK_HUMAN),
])

# 带记录时间的变体（MCP 侧使用；ALM 侧仍走上面的 STORY_RANK_PROMPT，prompt 保持逐字不变）
STORY_RANK_PROMPT_TS = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM + _STORY_RANK_TS_RULE),
    ("human", _STORY_RANK_HUMAN),
])

# ---- 溯源核对 Prompt（任务 C3）----

SOURCE_AUDIT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆核查员。下面给你一段**原始对话**，以及从这段对话抽取出的**若干记忆节点**。
逐条核对，只输出 JSON。

核对三件事：
1. unsupported：哪些节点的内容在原文里**找不到支撑**（推断、过度概括、或凭空补出的信息）。
   注意：节点允许"消解指代、补全省略"，但**不得引入原文没有的事实**。
   找不到支撑的节点通常是"把推测写成了事实"，这是最重要的一类问题。
2. missing：原文里有、但**没有被抽出来**的关键事实——重点看"为什么做某个选择"：
   被放弃的选项、取舍依据（成本/工期/偏好/外部约束）有没有留下。
3. granularity：粒度不对的节点。**只报两类**：
   - 把**无法定位的模糊表述**单独抽成了节点——**仅限这一类词**：「过两天」「过几天」「过些天」
     「过阵子」「回头」「改天」「以后」「晚点」「有空」「再说」「将来」等**既落不到具体日期、
     也无法与库中时段对应**的说法，脱离上下文就会漂；
   - 把本应拆开的多件事合并成了一条（**这条与时间词无关，该报照报，不要漏**）。
   **下面"不要报"的限定只针对第一类（时间词独立成节点）**：除第一类词之外，任何时间表述
   单独成节点都不算粒度问题，一律不要报。包括但不限于：
   - 具体日期，与裸周 / 裸月 / 裸年 / 学期（「上周五」「下周一」「明天」「上周」「这周」
     「下周」「上个月」「下个月」「去年」「今年」「寒假」「暑假」「上学期」「上半年」）；
   - 已被系统纳入时段映射、或仅作锚点的相对/指示表述（「前几天」「前两天」「这两天」
     「这几天」「那天」「当天」「当时」「后来」「之后」「第二天」「次日」「最近」「近来」
     「这段时间」「那段时间」「近年来」「毕业」「入职」「凌晨」「早上」「晚上」）。
   这些要么能落到具体时段，要么是系统时序检索的**既定设计**，独立成节点是正确的。

判定纪律：
- 只依据给定的原文，不要用常识补全、不要假设原文之外的信息；
- **逐条节点都要过一遍**，不要跳读；**发现可疑就报**——这份报告是给人工/上层 agent
  复核用的线索，漏报的代价高于误报。但每条都必须给出判断依据（why），
  好让复核者能快速否掉；
- 不确定"算不算问题"时，宁可按更轻的类别报出来（例如先报 granularity），也不要直接放过。

输出格式（严格 JSON）：
{{
  "unsupported": [{{"id": "n1", "why": "..."}}],
  "missing": [{{"quote": "原文片段", "why": "..."}}],
  "granularity": [{{"id": "n1", "why": "..."}}]
}}
没有问题的项返回空数组。只输出 JSON。"""),
    ("human", """── 原始对话 ──
{batch_text}

── 抽取出的节点 ──
{nodes_text}"""),
])

# ---- 回复生成 Prompt ----

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个有共情能力的对话助手。根据以下信息生成自然回复。

{context}

要求：
1. 回复自然流畅，如朋友间聊天
2. 如果有相关记忆，自然地融入回复
3. 不要机械地列举信息
4. 保持简洁温暖"""),
    ("human", "{query}"),
])


class InferenceEngine:
    """基于 LangChain ChatOpenAI 的推理引擎"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def infer_status(self, query: str) -> dict:
        """推断用户状态和情绪"""
        import json
        import logging
        chain = STATUS_INFERENCE_PROMPT | self.llm
        response = chain.invoke({"query": query})
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logging.warning(f"状态推断 JSON 解析失败，原始响应: {response.content[:200]}")
            return {"status": "unknown", "emotion": "中性"}

    def infer_purpose(self, query: str) -> dict:
        """推断用户隐含目的"""
        import json
        import logging
        chain = PURPOSE_INFERENCE_PROMPT | self.llm
        response = chain.invoke({"query": query})
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logging.warning(f"目的推断 JSON 解析失败，原始响应: {response.content[:200]}")
            return {"status": "unknown", "purposes": ["理解"]}

    def story_rank(self, query: str, path: dict, render_timestamps: bool = False) -> dict:
        """把一条检索因果链路理解成故事片段

        Args:
            query: 用户当前消息
            path: {"nodes": [{"id", "content", "node_type", "timestamp"}],
                   "edges": [{"from", "to", "rel_type", "is_reverse"}]}
            render_timestamps: 是否把节点的记录时间渲染进 prompt（MCP 侧开启；
                ALM 侧保持 False，prompt 与既有行为逐字一致）

        Returns:
            {"story": str, "adopted_ids": [id, ...]}
        """
        import json
        import logging

        nodes = path.get("nodes", [])
        if not nodes:
            return {"story": "", "adopted_ids": []}

        # 格式化链路
        lines = ["节点："]
        for n in nodes:
            nt = n.get("node_type", "")
            tag = ""
            if render_timestamps:
                ts_text = format_ts_ms(n.get("timestamp"))
                if ts_text:
                    tag = f"(记录于 {ts_text}) "
            lines.append(f'  [{n["id"]}]（{nt}）{tag}{n.get("content", "")}')
        lines.append("关系：")
        for e in path.get("edges", []):
            rel = e.get("rel_type", "")
            if e.get("is_reverse"):
                lines.append(f'  [{e["to"]}] --{rel}--> [{e["from"]}]')
            else:
                lines.append(f'  [{e["from"]}] --{rel}--> [{e["to"]}]')
        path_text = "\n".join(lines)

        prompt = STORY_RANK_PROMPT_TS if render_timestamps else STORY_RANK_PROMPT
        chain = prompt | self.llm
        response = chain.invoke({"query": query, "path": path_text})

        all_ids = [n["id"] for n in nodes]
        try:
            cleaned = response.content.strip()
            if cleaned.startswith("```"):
                lines_ = cleaned.split("\n")
                lines_ = [l for l in lines_ if not l.startswith("```")]
                cleaned = "\n".join(lines_)
            data = json.loads(cleaned)
            story = data.get("story", "")
            adopted = [str(a) for a in data.get("adopted_ids", []) if str(a) in all_ids]
        except (json.JSONDecodeError, AttributeError):
            logging.warning("StoryRank JSON 解析失败，降级为拼接节点内容")
            story = " ".join(n.get("content", "") for n in nodes)
            adopted = list(all_ids)

        return {"story": story, "adopted_ids": adopted}

    def audit_source(self, batch_text: str, nodes_text: str) -> dict:
        """溯源核对（任务 C3）：把"该批原文 + 该批产出的节点"交给 LLM，只报判断结果。

        Returns:
            {"unsupported": [{"id","why"}], "missing": [{"quote","why"}],
             "granularity": [{"id","why"}]}；解析失败时返回 {"error": ...}
        """
        import json
        import logging

        if not batch_text or not nodes_text:
            return {"unsupported": [], "missing": [], "granularity": []}

        chain = SOURCE_AUDIT_PROMPT | self.llm
        response = chain.invoke({"batch_text": batch_text, "nodes_text": nodes_text})
        raw = (response.content or "").strip()
        # 容错：模型可能套 ```json 代码块
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            logging.warning(f"溯源核对 JSON 解析失败，原始响应: {raw[:200]}")
            return {"error": "JSON 解析失败", "raw": raw[:500]}

        out = {}
        for key in ("unsupported", "missing", "granularity"):
            items = data.get(key)
            out[key] = items if isinstance(items, list) else []
        return out

    def generate_response(
        self,
        query: str,
        context_items: List[str],
    ) -> str:
        """注入检索到的记忆上下文，生成回复"""
        if context_items:
            context_text = "【关于用户的已知信息】\n" + "\n".join(
                f"- {item}" for item in context_items
            )
        else:
            context_text = "【关于用户的已知信息】\n暂无相关记忆。"

        chain = RESPONSE_PROMPT | self.llm
        response = chain.invoke({
            "query": query,
            "context": context_text,
        })
        return response.content
