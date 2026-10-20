# SPDX-License-Identifier: AGPL-3.0-only

"""空间归属判定器（B 项 + T3）：这个空间值不值得付「扩散召回」的代价。

问什么：把「**语料样本 + 最早的 N 道题**」给模型，让它判该空间该走纯向量还是联想路线。

问法为什么长这样——四版实测：

    v1 「这题是否需要联想路线」        → 放行 82%；剧本 / 论文 / 纯计算等**与用户记忆无关**
                                         的题全被放行（模型答的是"题本身难不难"）
    v2 「答案是否存在于该用户记忆里」  → 挡住 71%；但中文病历、Audrey 的狗等**真记忆题**
                                         被系统性误杀（它答的是"我看到的这段里有没有"）
    v3 「是否需牺牲精度扩大召回」      → 两边各错一半（换了个说法，仍答的是"题本身"）
    v4 「v3 + **明确扩散会引入大量噪声**」→ A 组（应挡）挡住 **96%**、B 组（应放）放行 **100%**

v4 的关键差别只有一处：**把"扩散"描述成一个有代价的动作**，让模型做成本权衡——

    答案就在一条里  → 扩散只有噪声、没有收益 → 低
    答案要靠相邻几条拼 → 收益 > 噪声           → 高

另外两条实测口径（**不要改**）：

- 输出呈**双峰**：取值只落在 `{0.0, 0.1, 0.2, 0.3, 0.8, 0.9}`，0.3~0.8 之间一个都没有。
  因此判定按 `p >= 0.5` 取布尔即可，**门槛不需要标定**（0.4/0.5/0.6/0.7 结果完全相同）。
- 同一输入重复 5 次：数值一致 76%，**二分一致 100%** → **不需要多数投票**。

失败策略：调用失败或输出不可解析时返回 None。调用方按**成本优先**兜底——判不出来就
按 `vector`（少花钱；代价是可能的静默丢分，这是本轮的显式决策）。
"""

import json
import logging
import re
from typing import List, Optional

from dba_pipeline.llm.roles import with_role

logger = logging.getLogger(__name__)

ROUTE_PROMPT = """你是记忆系统的路由判定员。下面给你**某个用户空间开头的一段语料**（已截断）与该空间**最早的几道查询题**。

本系统有两条检索路线，请判断这个空间更适合哪一条。

【路线 A：纯向量直查】
直接用问题去向量库找最相似的内容返回。**没有额外代价，也不会引入噪声。**

【路线 B：扩散召回（联想）】
从命中的内容出发，扩展出周边相关内容再返回。
- 代价：**会引入大量噪声**——返回里可能混进与当前问题不相关、只与命中内容相邻的信息；
- 收益：当答案需要由**相邻的多条内容拼合**才能得出时（时间推移、多个细节共同决定一个结论），
  路线 A 会漏掉，只有路线 B 拿得到。

判定规则：
- 答案很可能就在**最相似的那一条内容**里 → 选 A（扩散只会加噪声，没有收益）；
- 答案需要**跨片段拼合**才能得出 → 选 B；
- 与这个用户自身的经历 / 状态**无关**的题（通用知识、纯计算、文档本身的内容、任务索引）→ 选 A。

只输出 JSON，不要解释：
{{"p": 0.0 到 1.0 之间的一个数}}

其中 p 的含义：**该空间越该走路线 B，p 越接近 1；越该走路线 A，p 越接近 0。**

语料（截断）：
{corpus}

最早的查询题：
{questions}"""

_P_RE = re.compile(r'"p"\s*:\s*(-?[0-9]*\.?[0-9]+)')


def parse_p(text: str) -> Optional[float]:
    """从 LLM 输出里解析 p；解析不出来返回 None（调用方按 vector 兜底）"""
    if not text:
        return None
    match = _P_RE.search(text)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return None
    # 容错：模型可能只回一个裸数字
    stripped = text.strip()
    try:
        return max(0.0, min(1.0, float(stripped)))
    except ValueError:
        pass
    try:
        payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
        value = payload.get("p")
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    except Exception:
        pass
    return None


class AssociationJudge:
    """用**一次**轻量 LLM 调用判定空间归属（每个空间只发生一次）"""

    def __init__(self, llm, sample_chars: int = 700):
        self.llm = llm
        self.sample_chars = max(100, sample_chars)

    def association_p(self, corpus: str, questions: List[str]) -> Optional[float]:
        """返回 p ∈ [0,1]；调用失败或解析失败返回 None（调用方按 vector 兜底）"""
        corpus = (corpus or "").strip()
        questions = [q.strip() for q in (questions or []) if q and q.strip()]
        if not questions:
            return None
        listing = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        prompt = ROUTE_PROMPT.format(corpus=corpus[: self.sample_chars] or "（空）",
                                     questions=listing)
        try:
            response = with_role(self.llm, "route").invoke(prompt)
        except Exception as exc:
            logger.warning("归属判定调用失败，按 vector 兜底: %s", exc)
            return None
        score = parse_p(getattr(response, "content", "") or "")
        if score is None:
            logger.warning("归属判定输出无法解析，按 vector 兜底")
        return score
