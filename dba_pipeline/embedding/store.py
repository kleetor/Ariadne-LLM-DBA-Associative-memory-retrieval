# SPDX-License-Identifier: AGPL-3.0-only

"""
LangChain 向量存储封装

统一向量检索接口，底层可切换 FAISS / ChromaDB。
"""

from typing import Dict, List, Tuple, Optional
import logging
import threading

import numpy as np
import requests
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(Embeddings):
    """OpenAI 兼容 Embedding API

    支持 OpenAI、SiliconFlow、DeepSeek 等任何兼容 /v1/embeddings 端点的服务。
    使用轻量 requests 直调，避免 openai SDK 依赖。

    Args:
        api_key: API 密钥
        api_base: API Base URL（如 https://api.openai.com/v1）
        model: 模型名（如 text-embedding-3-small）
    """

    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    # bge 系列最大序列 512 token（本地 sentence-transformers 会自动截断，API 不会）。
    # 为与本地行为一致并避免长输入 400，发送前先截断到保守字符预算；
    # 若仍返回 400（数字/符号密集文本 token 偏多），渐进减半重试。
    _MAX_CHARS = 2000

    def _call(self, input_texts: List[str]) -> List[List[float]]:
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        texts, budget = [t[: self._MAX_CHARS] for t in input_texts], self._MAX_CHARS
        while True:
            payload = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                break
            if resp.status_code == 400 and budget > 400:
                budget //= 2
                texts = [t[:budget] for t in texts]
                continue
            resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


# 向后兼容别名
SiliconFlowEmbeddings = OpenAIEmbeddings


class LocalEmbeddings(Embeddings):
    """本地 Embedding 模型（sentence-transformers），消除 API 依赖"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            # 网络不通时降级为仅用缓存
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text, normalize_embeddings=True,
        )
        return embedding.tolist()


class VectorStore:
    """向量存储抽象层 + 内容向量缓存"""

    def __init__(
        self,
        embeddings: Embeddings,
        backend: str = "faiss",
        persist_dir: Optional[str] = None,
    ):
        """        
        Args:
            embeddings: LangChain Embeddings 实例（HuggingFace/OpenAI 等均可）
            backend: "faiss" 或 "chroma"
            persist_dir: 持久化目录（仅 Chroma 支持）
        """
        self.embeddings = embeddings
        self.backend = backend
        self.persist_dir = persist_dir
        self.store = None
        # 内容向量缓存: {memory_id: np.ndarray}，避免重复逐条 API 嵌入
        self._content_vectors: dict = {}
        # 内容缓存: {memory_id: content}，用于全量重建 FAISS 索引
        self._contents: dict = {}
        # 串行化对内部状态（缓存字典 + FAISS store）的访问。
        # 检索读线程与 DBA 维护写线程（maintenance_scheduler）共享同一实例，
        # FAISS 为 C++ 原生实现，读写并发不受 GIL 保护，须加锁避免竞态/崩溃。
        self._lock = threading.RLock()

    def _embed(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        return np.array(self.embeddings.embed_query(text))

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取文本向量"""
        return np.array(self.embeddings.embed_documents(texts))

    def add_memories(
        self,
        memory_ids: List[str],
        contents: List[str],
        metadatas: Optional[List[dict]] = None,
    ):
        """批量添加记忆到向量存储（预缓存向量 + 增量写入，不重复 embedding）"""
        # 无 embedding 配置时降级：仅维护内容映射，不构建向量索引
        if self.embeddings is None:
            with self._lock:
                for mid, content in zip(memory_ids, contents):
                    self._contents[mid] = content
            logger.warning("embeddings 未配置，跳过向量写入（节点仅存在于图谱）")
            return

        # 锁外批量嵌入（网络/CPU 开销大），避免阻塞并发检索线程
        with self._lock:
            new_ids = [mid for mid in memory_ids if mid not in self._content_vectors]
        new_vectors = self.embed_batch(
            [c for mid, c in zip(memory_ids, contents) if mid in new_ids]
        ) if new_ids else []
        vec_map = dict(zip(new_ids, new_vectors))

        # 用预缓存的向量直接写入 FAISS，避免 from_documents/add_documents 重复 embedding
        text_embeddings = []
        embed_contents = []
        doc_metadatas = []
        for i, (mid, content) in enumerate(zip(memory_ids, contents)):
            vec = vec_map.get(mid)
            if vec is None:
                vec = self._content_vectors.get(mid)
            if vec is None:
                logger.warning(f"节点 {mid} 缺少向量，跳过写入")
                continue
            text_embeddings.append((content, vec.tolist()))
            embed_contents.append(content)
            doc_metadatas.append({
                "memory_id": mid,
                **(metadatas[i] if metadatas else {}),
            })

        with self._lock:
            for mid, vec in vec_map.items():
                self._content_vectors[mid] = vec
            # 同步内容映射（用于全量重建）
            for mid, content in zip(memory_ids, contents):
                self._contents[mid] = content

            if not text_embeddings:
                return
            if self.store is None:
                if self.backend == "faiss":
                    self.store = FAISS.from_embeddings(
                        text_embeddings, self.embeddings, metadatas=doc_metadatas,
                    )
                elif self.backend == "chroma":
                    if self.persist_dir is None:
                        self.persist_dir = "./chroma_db"
                    documents = [
                        Document(page_content=content, metadata=meta)
                        for content, meta in zip(embed_contents, doc_metadatas)
                    ]
                    self.store = Chroma.from_documents(
                        documents, self.embeddings, persist_directory=self.persist_dir,
                    )
            else:
                if self.backend == "faiss":
                    self.store.add_embeddings(text_embeddings, metadatas=doc_metadatas)
                elif self.backend == "chroma":
                    documents = [
                        Document(page_content=content, metadata=meta)
                        for content, meta in zip(embed_contents, doc_metadatas)
                    ]
                    self.store.add_documents(documents)

    def update_memories(
        self,
        memory_ids: List[str],
        contents: List[str],
    ):
        """更新已有节点的向量（重算向量 + 全量重建索引）

        FAISS 不支持原地更新，故采用「重算 → 更新权威映射 → 全量重建」。
        适用于低频的 update 操作（DBA 维护 / 人工干预）。
        """
        if not memory_ids:
            return

        # 无 embedding 配置时降级：仅更新内容映射
        if self.embeddings is None:
            for mid, content in zip(memory_ids, contents):
                self._contents[mid] = content
            logger.warning("embeddings 未配置，跳过向量更新")
            return

        vectors = self.embed_batch(contents)
        with self._lock:
            for mid, content, vec in zip(memory_ids, contents, vectors):
                self._content_vectors[mid] = vec
                self._contents[mid] = content

            self._rebuild_index()

    def remove_memories(self, memory_ids: List[str]):
        """从向量库移除节点（更新缓存后全量重建索引）"""
        with self._lock:
            changed = False
            for mid in memory_ids:
                if mid in self._content_vectors:
                    del self._content_vectors[mid]
                    changed = True
                if mid in self._contents:
                    del self._contents[mid]
                    changed = True
            if changed:
                self._rebuild_index()

    def reconcile(self, desired: Dict[str, str]) -> dict:
        """把向量库对齐到期望集合 ``{memory_id: content}``。

        用于图谱被**其它进程**（如 WebUI 面板）修改后，把向量库补齐/更新/清理：
        - 期望里有、库里没有 → 新增（批量嵌入）
        - 两边都有但内容不同 → 更新向量
        - 库里有、期望里没有 → 删除

        与逐条 add/update/remove 的区别：一次完成，且**只重建一次**索引
        （FAISS 不支持原地更新，重建是主要成本）。

        Returns:
            {"added": n, "updated": n, "removed": n}
        """
        # 无 embedding 配置时降级：仅维护内容映射
        if self.embeddings is None:
            with self._lock:
                self._contents = dict(desired)
            logger.warning("embeddings 未配置，仅同步内容映射（未构建向量索引）")
            return {"added": 0, "updated": 0, "removed": 0, "degraded": True}

        with self._lock:
            known = set(self._content_vectors) | set(self._contents)
            to_remove = [mid for mid in known if mid not in desired]
            to_add = [mid for mid in desired if mid not in known]
            to_update = [
                mid for mid in desired
                if mid in known and self._contents.get(mid) != desired[mid]
            ]
            need = to_add + to_update

        # 锁外批量嵌入（网络/CPU 开销大），避免阻塞并发检索线程
        vectors = self.embed_batch([desired[mid] for mid in need]) if need else []

        with self._lock:
            for mid in to_remove:
                self._content_vectors.pop(mid, None)
                self._contents.pop(mid, None)
            for mid, vec in zip(need, vectors):
                self._content_vectors[mid] = np.asarray(vec)
                self._contents[mid] = desired[mid]
            if to_remove or need:
                self._rebuild_index()
        return {"added": len(to_add), "updated": len(to_update), "removed": len(to_remove)}

    def clear_vectors(self):
        """清空向量缓存与索引（用于以图谱为权威全量重建）"""
        with self._lock:
            self._content_vectors.clear()
            self._contents.clear()
            self.store = None

    def _rebuild_index(self):
        """从权威映射（_content_vectors + _contents）全量重建 FAISS 索引"""
        with self._lock:
            if self.backend != "faiss":
                logger.warning("全量重建仅支持 FAISS 后端")
                return
            if not self._content_vectors:
                self.store = None
                return
            text_embeddings = []
            doc_metadatas = []
            for mid, vec in self._content_vectors.items():
                content = self._contents.get(mid, "")
                text_embeddings.append((content, np.asarray(vec).tolist()))
                doc_metadatas.append({"memory_id": mid})
            self.store = FAISS.from_embeddings(
                text_embeddings, self.embeddings, metadatas=doc_metadatas,
            )
            logger.info(f"FAISS 索引已重建: {len(text_embeddings)} 条向量")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        """向量检索 Top-K

        Returns:
            [(memory_id, score, metadata), ...]
        """
        with self._lock:
            if self.store is None:
                return []

            docs_with_scores = self.store.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs_with_scores:
                mid = doc.metadata.get("memory_id", "")
                results.append((mid, float(score), doc.metadata))
            return results

    def search_by_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """按向量检索（用于目的向量匹配）

        返回相关性分数，由距离换算：relevance = 1 / (1 + distance)，
        距离 0 → 1.0，越远越趋近 0，保持「越大越相关」的语义。

        注：本版本 FAISS 没有 similarity_search_by_vector_with_relevance_scores，
        历史上在此处降级为占位分 1.0，使该分数不可用（调用方只能依赖返回顺序）。
        """
        with self._lock:
            if self.store is None:
                return []
            vec = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            try:
                docs_with_scores = self.store.similarity_search_with_score_by_vector(
                    vec, k=k
                )
            except AttributeError:
                logger.warning(
                    "FAISS 不支持 similarity_search_with_score_by_vector，"
                    "退回无分数检索"
                )
                return [
                    (doc.metadata.get("memory_id", ""), 1.0)
                    for doc in self.store.similarity_search_by_vector(vec, k=k)
                ]
            return [
                (
                    doc.metadata.get("memory_id", ""),
                    1.0 / (1.0 + max(0.0, float(score))),
                )
                for doc, score in docs_with_scores
            ]

    @property
    def embedding_dim(self) -> int:
        """向量维度"""
        with self._lock:
            if self._content_vectors:
                return len(list(self._content_vectors.values())[0])
        if self.embeddings is None:
            return 0
        test_vec = self._embed("test")
        return len(test_vec)

    def get_content_vectors(self, memory_ids: List[str]) -> List[np.ndarray]:
        """获取预缓存的节点向量（无 API 调用）"""
        with self._lock:
            return [self._content_vectors.get(mid) for mid in memory_ids]

    def has_vectors(self) -> bool:
        """是否已有向量缓存"""
        with self._lock:
            return len(self._content_vectors) > 0

    # ---- P3b: 快照持久化 ----

    def save(self, path: str):
        """将 FAISS 索引和向量缓存保存到磁盘"""
        with self._lock:
            if self.store is None or self.backend != "faiss":
                logger.warning("无可保存的 FAISS 索引")
                return
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.store.save_local(path)
            # 保存 _content_vectors 缓存到 .npz（与 FAISS 索引同目录）
            if self._content_vectors:
                import numpy as np
                vec_path = os.path.join(os.path.dirname(path) or ".", "content_vectors.npz")
                ids = list(self._content_vectors.keys())
                vectors = np.stack([self._content_vectors[k] for k in ids])
                contents = np.array([self._contents.get(k, "") for k in ids])
                # 实测：此处压缩是索引落盘的主要成本——163 节点 / 1024 维下
                # savez_compressed 耗时 0.325s，savez 仅 0.006s（54 倍差距），
                # 代价是体积 711KB → 1323KB。该文件每次写入都会重写，故取速度。
                np.savez(vec_path, ids=np.array(ids), vectors=vectors, contents=contents)
                logger.info(f"向量缓存已保存: {len(ids)} 个向量")
            logger.info(f"FAISS 索引已保存: {path} ({self.store.index.ntotal} 向量)")

    def load(self, path: str, embeddings: "Embeddings" = None):
        """从磁盘加载 FAISS 索引和向量缓存"""
        with self._lock:
            if self.backend != "faiss":
                raise ValueError("仅 FAISS 支持加载")
            import os
            if not os.path.exists(path):
                raise FileNotFoundError(f"FAISS 索引文件不存在: {path}")
            emb = embeddings or self.embeddings
            # 注意：FAISS 索引基于 pickle 反序列化，仅应从可信的本地 checkpoint 加载
            self.store = FAISS.load_local(
                path, emb, allow_dangerous_deserialization=True,
            )
            # 恢复 _content_vectors 缓存
            vec_path = os.path.join(os.path.dirname(path) or ".", "content_vectors.npz")
            if os.path.exists(vec_path):
                import numpy as np
                # 缓存仅含常规 dtype 数组，禁用 pickle 以降低反序列化风险
                data = np.load(vec_path, allow_pickle=False)
                ids = data["ids"]
                vectors = data["vectors"]
                self._content_vectors = {str(k): v for k, v in zip(ids, vectors)}
                # 恢复内容映射（兼容旧格式：无 contents 字段时尝试从 docstore 补全）
                if "contents" in data:
                    contents = data["contents"]
                    self._contents = {str(k): str(v) for k, v in zip(ids, contents)}
                else:
                    self._contents = {}
                    try:
                        docstore = self.store.docstore
                        for doc in docstore._dict.values():
                            mid = doc.metadata.get("memory_id")
                            if mid in self._content_vectors:
                                self._contents[mid] = doc.page_content
                    except Exception:
                        pass
            logger.info(f"向量缓存已恢复: {len(self._content_vectors)} 个向量")
        logger.info(f"FAISS 索引已加载: {path} ({self.store.index.ntotal} 向量)")
