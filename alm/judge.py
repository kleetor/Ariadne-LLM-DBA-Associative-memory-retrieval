# SPDX-License-Identifier: AGPL-3.0-only

"""检索可用性判定：弃权「模糊带」的二次判定。

动机（实测，见 eval/probe_abstain_threshold.py）：

    有答案查询: min=0.3954  中位=0.5064  max=0.5931
    无答案查询: min=0.2667  中位=0.3494  max=0.4701
    → 两类区间重叠于 [0.3954, 0.4701]

即「库里确实没有相关记忆」与「有，但提问措辞与节点原文距离远」在余弦上分不开，
任何单一阈值都会错一边。故本模块只处理**模糊带**内的查询：

    best_cos <  abstain_cosine     → 硬下限，直接弃权（省掉这次调用）
    abstain_cosine ≤ best_cos < abstain_verify_hi → 本模块判定
    best_cos ≥ abstain_verify_hi   → 直接放行

失败策略：调用失败或输出不可解析时返回 None，调用方按「不弃权」处理。理由与兜底
落库一致——误弃权会让该题整条归零，代价远大于多返回几条无关内容。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dba_pipeline.llm.roles import with_role

logger = logging.getLogger(__name__)

ABSTAIN_JUDGE_PROMPT = """你是记忆检索的可用性判定员。下面给你一个问题，以及检索系统召回的若干条记忆片段（已按相关度从高到低排序）。

请判断：**这些片段里是否包含回答问题所必需的信息？**

判定规则：
- 只要有一条片段直接给出了问题的答案，或提供了定位答案的关键线索，判 true；
- 片段与问题只是措辞相近、但并未包含问题所问的信息时，判 false；
- 问题涉及的人物、时间、事件、属性在片段里完全找不到时，判 false；
- 问题本身是寒暄、闲聊或与个人记忆无关的通用知识，而片段里没有对应内容时，判 false。

只输出 JSON，不要解释：
{{"relevant": true}}

问题：
{query}

记忆片段：
{listing}"""

_RELEVANT_RE = re.compile(r'"relevant"\s*:\s*(true|false)', re.IGNORECASE)


def parse_verdict(text: str) -> Optional[bool]:
    """从 LLM 输出中解析判定结果；解析不出来返回 None（调用方按放行处理）"""
    if not text:
        return None
    match = _RELEVANT_RE.search(text)
    if match:
        return match.group(1).lower() == "true"
    # 容错：模型可能只回一个裸布尔或中文
    stripped = text.strip().lower()
    if stripped in ("true", "false"):
        return stripped == "true"
    try:
        payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
        if isinstance(payload.get("relevant"), bool):
            return payload["relevant"]
    except Exception:
        pass
    return None


class RelevanceJudge:
    """用一次轻量 LLM 调用判定「召回片段是否够回答问题」"""

    def __init__(self, llm, top_n: int = 5, max_chars: int = 200):
        self.llm = llm
        self.top_n = max(1, top_n)
        self.max_chars = max(50, max_chars)

    def _listing(self, items: List[Dict[str, Any]]) -> str:
        lines = []
        for item in items[: self.top_n]:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if len(content) > self.max_chars:
                content = content[: self.max_chars] + "…"
            lines.append(f"- {content}")
        return "\n".join(lines)

    def is_relevant(self, query: str, items: List[Dict[str, Any]]) -> Optional[bool]:
        """True=片段够用（放行）；False=确实无相关信息（弃权）；None=判定失败（放行）"""
        listing = self._listing(items)
        if not listing:
            return False  # 没有任何可展示的候选内容，等价于无信息
        prompt = ABSTAIN_JUDGE_PROMPT.format(query=query, listing=listing)
        try:
            response = with_role(self.llm, "abstain").invoke(prompt)
        except Exception as exc:
            logger.warning("弃权判定调用失败，按放行处理: %s", exc)
            return None
        verdict = parse_verdict(getattr(response, "content", "") or "")
        if verdict is None:
            logger.warning("弃权判定输出无法解析，按放行处理")
        return verdict
