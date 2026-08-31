# SPDX-License-Identifier: AGPL-3.0-only

"""
LangChain LLM 封装：状态推断 + 目的推断 + 回复生成
"""

from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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

STORY_RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个记忆故事整理助手。你会收到用户当前消息，以及一组「记忆节点与关系」（可能包含多条因果链路）。

你的任务：
1. 理解这些节点与关系：每个节点是一个记忆片段（带类型），边表示节点之间的关系（带关系类型）。
2. 把所有记忆整合成一段连贯、自然的故事，让读者能顺畅理解用户的状态与经历。只输出一段文字，不要分段列举。
3. 关系语义要自然融入句子，例如：
   - CAUSAL（因果）→ 因为 / 导致 / 所以
   - SEQUENCE（时序）→ 然后 / 之后 / 接着
   - PREFERENCE（偏好）→ 偏爱 / 喜欢
   - SCENARIO（场景）→ 在…场景下 / 常去
   - ATTRIBUTE（属性）→ 具有…特征
   - TEMPORAL（时间）→ 时间上
   - SOCIAL（社交）→ 与…有关
   - TAXONOMIC（分类）→ 属于 / 是一种
4. 只采纳与链路连贯的记忆；明显突兀、与主线无关的节点应舍弃，不出现在故事中。
5. 忠实于节点内容，不编造未出现的事实。

输出格式（严格 JSON）：
{{
    "story": "一段连贯的故事文本",
    "adopted_ids": ["被采纳进故事的节点id", "..."]
}}
只输出 JSON。"""),
    ("human", """用户当前消息：
{query}

记忆因果链路：
{path}"""),
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

    def story_rank(self, query: str, path: dict) -> dict:
        """把一条检索因果链路理解成故事片段

        Args:
            query: 用户当前消息
            path: {"nodes": [{"id", "content", "node_type"}],
                   "edges": [{"from", "to", "rel_type", "is_reverse"}]}

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
            lines.append(f'  [{n["id"]}]（{nt}）{n.get("content", "")}')
        lines.append("关系：")
        for e in path.get("edges", []):
            rel = e.get("rel_type", "")
            if e.get("is_reverse"):
                lines.append(f'  [{e["to"]}] --{rel}--> [{e["from"]}]')
            else:
                lines.append(f'  [{e["from"]}] --{rel}--> [{e["to"]}]')
        path_text = "\n".join(lines)

        chain = STORY_RANK_PROMPT | self.llm
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
