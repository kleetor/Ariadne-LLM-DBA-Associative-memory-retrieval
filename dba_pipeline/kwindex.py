# SPDX-License-Identifier: AGPL-3.0-only

"""关键词倒排索引：给知识层做“字面精筛”。

设计取舍：

- **不引新依赖**。仓库里没有 jieba / rank_bm25，中文用字符二元组、ASCII 用词元，
  对“接口名 / 错误码 / 项目名”这类查询足够；同义改写的召回交给记忆层。
- **字段加权**。笔记有 `title / tags / text` 三个字段，权重 3/2/1；普通节点只有
  `content`，按 text 计。分母固定 `3 × |query_terms|`，所以标题命中的分数上限是 1，
  仅正文命中的上限是 1/3——这样笔记会自然排在普通节点前面。
- **索引是纯内存投影**。权威数据始终在 `memory_graph.yaml`；索引脏了整表重建即可，
  不落盘、不参与持久化。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

_ASCII_RE = re.compile(r"[a-z0-9_./:-]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

# 只挡掉最常见、信息量几乎为零的词元；不追求完整停用词表。
_STOP = {
    "的", "了", "是", "在", "和", "与", "就", "都", "也", "很",
    "我们", "你们", "他们", "这个", "那个", "什么", "怎么",
    "可以", "一个", "不是", "没有", "就是",
}

_FIELD_WEIGHTS = {"title": 3, "tags": 2, "text": 1}
_MAX_TOP_K = 50

# 截断时优先在这些字符之后断开，避免把一句话/代码行从中间切断
_SENTENCE_ENDS = "。！？；\n.!?;"


def truncate_for_return(text: str, limit: int) -> "tuple[str, bool]":
    """按返回预算截断正文，优先在句末/换行处断开。

    返回 ``(text, truncated)``。截断后长度不超过 ``limit``（省略号计入）。
    找不到合适断点时退化为硬截断——比在句子中间切要差，但仍然可控。
    """
    s = text or ""
    if limit <= 0 or len(s) <= limit:
        return s, False

    cut = s[:limit]
    # 只在尾部 200 字内找断点：更早的断点会让返回过短、信息量不足
    window_start = max(0, limit - 200)
    pos = max(cut.rfind(ch) for ch in _SENTENCE_ENDS)
    if pos >= window_start and pos + 1 < limit:
        out = cut[:pos + 1].rstrip() + "…"
    else:
        out = cut.rstrip()[: max(0, limit - 1)] + "…"
    return out, True


def tokenize(text: str) -> Set[str]:
    """把文本切成关键词集合。

    - ASCII：连续 `[a-z0-9_./:-]`，长度 ≥ 2（保留接口名、错误码、版本号）；
    - 中文：连续 CJK 段取字符二元组，单字段退化为该字符；
    - 统一小写，去掉停用词。
    """
    s = (text or "").lower()
    tokens: Set[str] = set(_ASCII_RE.findall(s))
    for seg in _CJK_RE.findall(s):
        if len(seg) == 1:
            tokens.add(seg)
        else:
            tokens.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return {t for t in tokens if t and t not in _STOP}


class KeywordIndex:
    """倒排表 + 字段命中打分。"""

    def __init__(self):
        self._postings: Dict[str, Set[str]] = {}
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def clear(self) -> None:
        self._postings.clear()
        self._nodes.clear()

    def build(self, nodes: Iterable[Dict[str, Any]]) -> None:
        """整表重建。`nodes` 每项：id / content / metadata / timestamp。"""
        self.clear()
        for node in nodes:
            self.add(node)

    def add(self, node: Dict[str, Any]) -> None:
        """加入或替换一个节点。已废弃/遗忘的节点直接跳过。"""
        if node.get("deprecated") or node.get("forgotten"):
            return
        nid = str(node.get("id") or "")
        content = node.get("content") or ""
        if not nid or not content.strip():
            return

        meta = node.get("metadata") or {}
        title = meta.get("title") or ""
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tag_text = " ".join(str(t) for t in tags)

        fields = {
            "title": tokenize(str(title)),
            "tags": tokenize(tag_text),
            "text": tokenize(content),
        }
        self._nodes[nid] = {
            "id": nid,
            "content": content,
            "metadata": meta,
            "timestamp": node.get("timestamp"),
            "fields": fields,
        }
        for term in set().union(*fields.values()):
            self._postings.setdefault(term, set()).add(nid)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """按关键词命中率返回候选（不截断正文）。

        返回项：node_id / score / content / metadata / observed_at。
        排序：score 降序 → observed_at 降序 → node_id 升序（稳定）。
        """
        q = tokenize(query)
        if not q or not self._nodes:
            return []
        try:
            top_k = max(1, min(int(top_k), _MAX_TOP_K))
        except (TypeError, ValueError):
            top_k = 8

        candidates: Set[str] = set()
        for term in q:
            candidates |= self._postings.get(term, set())
        if not candidates:
            return []

        denom = _FIELD_WEIGHTS["title"] * len(q)
        merged: List[Dict[str, Any]] = []
        for nid in candidates:
            rec = self._nodes.get(nid)
            if rec is None:
                continue
            fields = rec["fields"]
            # 命中任意一个 term 即保留（score > 0），不设覆盖率阈值：
            # 知识层是“精确查找”，部分命中交给 LLM 看分数判断，不在这里筛掉。
            score = (
                _FIELD_WEIGHTS["title"] * len(q & fields["title"])
                + _FIELD_WEIGHTS["tags"] * len(q & fields["tags"])
                + _FIELD_WEIGHTS["text"] * len(q & fields["text"])
            ) / denom
            # 标题与正文可能同时命中；上限仍是 1，避免分数越界。
            score = min(1.0, score)
            if score <= 0:
                continue
            meta = rec["metadata"] or {}
            observed = meta.get("observed_at")
            if not isinstance(observed, int):
                observed = rec.get("timestamp")
            merged.append({
                "node_id": nid,
                "score": round(float(score), 4),
                "content": rec["content"],
                "metadata": meta,
                "observed_at": observed,
            })

        merged.sort(
            key=lambda x: (
                -x["score"],
                -(x["observed_at"] or 0),
                x["node_id"],
            )
        )
        return merged[:top_k]
