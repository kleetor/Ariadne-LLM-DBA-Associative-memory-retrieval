# SPDX-License-Identifier: AGPL-3.0-only

"""
目的回归模型：保持检索方向不偏离出发动机

论文第四章：从用户当前消息推断隐含目的，
每步扩展后用目的向量校验候选记忆的相关性。
"""

import numpy as np
from typing import List

from langchain_openai import ChatOpenAI


class PurposeModel:
    """目的向量构建 + 回归校验"""

    def __init__(self, llm: ChatOpenAI, embedding_fn):
        """
        Args:
            llm: LangChain ChatOpenAI 实例
            embedding_fn: 将文本转为向量的函数，签名为 (text: str) -> np.ndarray
        """
        self.llm = llm
        self.embedding_fn = embedding_fn

    def get_purpose_vector(self, purposes: List[str]) -> np.ndarray:
        """将目的列表转为向量（拼接后编码）"""
        text = " ".join(purposes)
        return self.embedding_fn(text)

    def compute_purpose_score(
        self,
        candidates: List[str],
        purpose_vec: np.ndarray,
    ) -> np.ndarray:
        """计算候选记忆与目的向量的余弦相似度（API 嵌入版本）

        Returns:
            shape (n,) 的 score 数组
        """
        if not candidates:
            return np.array([])

        candidate_vecs = np.array([self.embedding_fn(c) for c in candidates])
        # 归一化
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)
        purpose_norm = purpose_vec / (np.linalg.norm(purpose_vec) + 1e-8)
        scores = np.dot(candidate_norms, purpose_norm)
        return scores

    def compute_purpose_score_from_vectors(
        self,
        candidate_vectors: List[np.ndarray],
        purpose_vec: np.ndarray,
    ) -> np.ndarray:
        """计算候选向量与目的向量的余弦相似度（预缓存向量版本，无 API 调用）"""
        if not candidate_vectors:
            return np.array([])

        valid = [(i, v) for i, v in enumerate(candidate_vectors) if v is not None]
        if not valid:
            return np.zeros(len(candidate_vectors))

        vecs = np.array([v for _, v in valid])
        norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        purpose_norm = purpose_vec / (np.linalg.norm(purpose_vec) + 1e-8)
        valid_scores = np.dot(norms, purpose_norm)

        # 还原完整数组，缺失向量返回 0
        scores = np.zeros(len(candidate_vectors))
        for j, (i, _) in enumerate(valid):
            scores[i] = valid_scores[j]
        return scores

    def filter_by_purpose(
        self,
        candidates: List[str],
        purpose_vec: np.ndarray,
        threshold: float = 0.3,
    ) -> List[tuple]:
        """目的回归过滤：丢掉目的关联度低于阈值的候选

        Returns:
            [(记忆文本, purpose_score), ...] 保留的候选项
        """
        if not candidates:
            return []

        scores = self.compute_purpose_score(candidates, purpose_vec)
        kept = [
            (c, float(scores[i]))
            for i, c in enumerate(candidates)
            if scores[i] >= threshold
        ]
        return kept
