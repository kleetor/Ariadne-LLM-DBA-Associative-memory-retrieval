# SPDX-License-Identifier: AGPL-3.0-only

"""
LangChain 向量存储封装

统一向量检索接口，底层可切换 FAISS / ChromaDB。
"""

from typing import Dict, List, Tuple, Optional
import logging
import os
import random
import threading
import time

import numpy as np
import requests
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---- Embedding API 的瞬时故障重试 ----
# 云主机上共享 embedding 端点出现 429（限流）/ 连接超时是常态，而这条链路对两端都是
# 关键路径：Add 侧建图要嵌入全部节点，Search 侧拿不到查询向量就无法判定相关性。
# 所以在网络层先做有界重试，重试耗尽才向上抛错——ALM 的 /search 会把异常转成 5xx，
# 交给平台重试，而不是静默降级成「没有相关记忆」。
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# 总尝试次数（含首次）。可用 EMBEDDING_MAX_ATTEMPTS 覆盖。
_EMBED_MAX_ATTEMPTS = max(1, int(os.environ.get("EMBEDDING_MAX_ATTEMPTS", "4")))
_RETRY_BASE = 0.5        # 指数退避基数（秒）：0.5 → 1 → 2 → 4
_RETRY_CAP = 4.0         # 单次退避上限
_RETRY_AFTER_CAP = 10.0  # 尊重 Retry-After，但不超过该上限


def _sleep_before_retry(attempt: int, reason: str, resp=None) -> bool:
    """退避后返回 True（可以再试）；已达次数上限则返回 False（由调用方抛错）"""
    if attempt >= _EMBED_MAX_ATTEMPTS - 1:
        return False
    delay = min(_RETRY_CAP, _RETRY_BASE * (2 ** attempt))
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, min(_RETRY_AFTER_CAP, float(retry_after)))
            except (TypeError, ValueError):
                pass
    delay += random.uniform(0, 0.3)  # 抖动，避免多线程同时重试再次撞限流
    logger.warning(
        "Embedding 调用失败，%.1fs 后重试（第 %d/%d 次）: %s",
        delay, attempt + 1, _EMBED_MAX_ATTEMPTS, reason,
    )
    time.sleep(delay)
    return True


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

    # 单条文本送进 embedding 前的字符预算（超限的**尾部会被丢弃**，只嵌前缀）。
    #
    # 语义要说清：这不是"切块"，是**静默截断**——超过它的部分不参与向量，也就检索不到。
    #
    # 历史：这个值随模型上限变过两次。
    #   · bge-large-zh-v1.5（512 token）时代标定为 **500**（实测 500/550 字符 → 200，
    #     600 字符 → 400 code=20015；512 token ≈ 550 汉字，取 500 留余量）。
    #     当时还刻意收窄过：写 2000 时每个长文本要走 2000(400)→1000(400)→500(200)
    #     三次往返，而 `_call` 是按批重试，同批短文本被迫跟着重嵌，成本成倍放大。
    #   · 改用 `BAAI/bge-m3`（8192 token）后，**实测端点真实上限**
    #     （eval/probe_m3_limit.py，SiliconFlow，直接打 API 绕过本截断）：
    #       英文 16000 字符 → 200；中文 8000 字符 → 200、16000 字符 → 400。
    #     故 500 已严重过窄，取 **2000**：对中英都安全（中文 2000 字 ≈ 1333 token，
    #     离 8192 上限很远），且**高于 `space.doc_chunk_chars=1800`**——文档块必须
    #     整块进向量，否则"放大块以保留语义完整性"这个设计会被这一步静默截掉一半。
    #
    # 为什么不再等于 500：500 时会连带伤害两处
    #   ① 文档块（1800 字）只嵌了前 500 字，尾部永远检索不到；
    #   ② 抽取出的长节点（节点膨胀时可能 >500 字）向量只覆盖前缀，语义被截。
    #
    # 若仍返回 400（数字/符号密集文本 token 偏多），下面的渐进减半会兜住。
    # 注意：改本值会改变长文本的向量，故已纳入 space 的索引指纹（换值即自动重建索引）。
    _MAX_CHARS = 2000


    def _call(self, input_texts: List[str]) -> List[List[float]]:
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        texts, budget = [t[: self._MAX_CHARS] for t in input_texts], self._MAX_CHARS
        attempt = 0
        while True:
            payload = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            }
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            except requests.exceptions.RequestException as exc:
                # 连接失败 / 超时：属瞬时故障，退避重试；次数耗尽才抛出
                if not _sleep_before_retry(attempt, f"{type(exc).__name__}: {exc}"):
                    raise
                attempt += 1
                continue
            if resp.status_code == 200:
                break
            if resp.status_code == 400 and budget > 400:
                budget //= 2
                texts = [t[:budget] for t in texts]
                continue
            if resp.status_code in _RETRY_STATUS:
                if not _sleep_before_retry(attempt, f"HTTP {resp.status_code}", resp):
                    resp.raise_for_status()
                attempt += 1
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

    # 与远端 `OpenAIEmbeddings` 对齐的"单条预算"字段：`space._index_digest()` 会读它，
    # 用于判断"改预算后落盘向量是否还与新查询同语义"。本类此前**没有该属性**，指纹里
    # 该字段恒为 None，与远端路径不一致（0927 审查 D6）。本地路径不按字符截断（交给
    # 模型的序列上限），故这里用模型的 token 上限作代理值：换模型即指纹变化 → 自动重建。
    _MAX_CHARS = 8192

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
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
        # 覆写为模型自身的序列上限：换模型 → 指纹变化 → 索引自动重建
        try:
            self._MAX_CHARS = int(self.model.max_seq_length)
        except Exception:
            pass

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


# ---- 向量落盘：追加式分片（2026-10-01 改）----
#
# 为什么不再走 LangChain 的 `save_local` / `load_local`：
#   ① `save_local` 会把 **docstore 里的全部正文**一并 pickle 到 `index.pkl`，而正文在
#      YAML / docs.yaml 里已有一份 → 每个空间在**盘上存两份**；`save` 里又把 `_contents`
#      写进 npz，是第三份。线上实测单空间索引 33~131 MB。
#   ② 它是**全量重写**：每次 Add 都要重写整个索引 + 整个向量缓存，而一次 Add 通常只新增
#      个位数向量 —— 线上实测 BLOCK I/O 14 GB / 12% 进度。
#   ③ docstore 走 pickle 反序列化，攻击面比只含数组的 npz 大。
#
# 改为：**索引一律不落盘**，只把向量的**变更**按 append-only 分片落盘；`load` 时用分片
# 里的向量在内存里重建索引（重建是 O(N) 的内存拷贝，3 万×1024 float32 ≈ 120 MB，毫秒级）。
# 检索仍是 IndexFlat 精确检索，recall 不变。
#
# 分片是**日志**不是快照：删除写墓碑（`deleted=1`），更新写新行，同 id **后写覆盖先写**。
# 分片文件数超过 `_VEC_SHARD_MAX_FILES` 时 compact 成一个，避免碎片无限增长。
_VEC_SUBDIR = "vectors"
_VEC_SHARD_SIZE = 2048
_VEC_SHARD_MAX_FILES = 16
# 旧格式（LangChain save_local 落下的压缩包 + 前几版的 npz）文件名，仅用于一次性迁移
_LEGACY_NPZ = "content_vectors.npz"


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
        # 待落盘的**向量变更日志**：[(memory_id, vec 或 None)]，None 表示删除（墓碑）。
        # 只记"自上次 save 以来"的变更，故 save 的写量与本次变更条数成正比，
        # 而不是与全库条数成正比（原来每次 Add 都全量重写）。
        self._vec_log: list = []
        # clear_vectors() 之后必须把盘上分片整体作废，否则 load 会把旧向量"复活"
        self._cleared: bool = False
        # 串行化对内部状态（缓存字典 + FAISS store）的访问。
        # 检索读线程与 DBA 维护写线程（maintenance_scheduler）共享同一实例，
        # FAISS 为 C++ 原生实现，读写并发不受 GIL 保护，须加锁避免竞态/崩溃。
        self._lock = threading.RLock()

    def _log_vec(self, mid: str, vec) -> None:
        """登记一次向量变更，供 `save` 增量落盘。调用方须持有 `_lock`。

        统一转 float32：分片落盘只需要单精度（与 FAISS 内部一致），存双精度等于
        白白多占一倍磁盘与内存。
        """
        self._vec_log.append(
            (mid, None if vec is None else np.asarray(vec, dtype=np.float32))
        )

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
                self._log_vec(mid, vec)
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
                self._content_vectors[mid] = np.asarray(vec, dtype=np.float32)
                self._contents[mid] = content
                self._log_vec(mid, vec)

            self._rebuild_index()

    def remove_memories(self, memory_ids: List[str]):
        """从向量库移除节点（更新缓存后全量重建索引）"""
        with self._lock:
            changed = False
            for mid in memory_ids:
                if mid in self._content_vectors:
                    del self._content_vectors[mid]
                    self._log_vec(mid, None)
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
                self._log_vec(mid, None)
            for mid, vec in zip(need, vectors):
                self._content_vectors[mid] = np.asarray(vec, dtype=np.float32)
                self._contents[mid] = desired[mid]
                self._log_vec(mid, vec)
            if to_remove or need:
                self._rebuild_index()
        return {"added": len(to_add), "updated": len(to_update), "removed": len(to_remove)}

    def clear_vectors(self):
        """清空向量缓存与索引（用于以图谱为权威全量重建）"""
        with self._lock:
            self._content_vectors.clear()
            self._contents.clear()
            self.store = None
            # 盘上分片必须整体作废：否则下次 load 会把"已清空"的向量又读回来
            self._vec_log.clear()
            self._cleared = True

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
                # 统一 float32：与 faiss 内部精度一致，避免每次重建都做一次 float64→float32
                text_embeddings.append((content, np.asarray(vec, dtype=np.float32).tolist()))
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
        """按向量检索（用于目的向量匹配 / 文档通道）

        返回**真余弦**（`_relevance`），与图侧的 `_cosine()` 同一把尺，保证
        `max(图余弦, 文档分)` 是合法比较。

        0927 审查 B2：此前返回 `1 / (1 + distance)`。FAISS 返回的是**平方 L2 距离**，
        故那个公式等价于 `1/(3−2cos)`——与余弦**不是同一把尺**，且在 cos≈0.5 处交叉
        （cos<0.5 被高估、cos>0.5 被低估），两方向都会错。原因是文档通道的分数会与图
        余弦一起进入 `best_cos`，进而污染弃权阈值标定。

        注：本版本 FAISS 没有 similarity_search_by_vector_with_relevance_scores；
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
            query_vec = np.asarray(vec, dtype=float)
            return [
                (
                    doc.metadata.get("memory_id", ""),
                    self._relevance(query_vec, doc.metadata.get("memory_id", ""),
                                    float(score)),
                )
                for doc, score in docs_with_scores
            ]

    def _relevance(self, query_vec: np.ndarray, mid: str, distance: float) -> float:
        """把一次向量命中换算成**与图侧同尺度的余弦**（0927 审查 B2）。

        优先用缓存向量精确计算余弦（与 embedding 提供方是否归一化无关）；缓存缺失时
        退回距离换算——仅当向量已归一化时 `cos = 1 − 平方L2/2` 才等价。
        """
        stored = self._content_vectors.get(mid)
        if stored is not None:
            s = np.asarray(stored, dtype=float)
            denom = float(np.linalg.norm(query_vec) * np.linalg.norm(s))
            if denom > 1e-12:
                return float(np.dot(query_vec, s) / denom)
        return float(1.0 - max(0.0, distance) / 2.0)

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
        """把**本次变更的向量**追加成新分片（不再全量重写索引）

        `path` 是索引目录，分片落在 `path/vectors/{seq:04d}.npz`。无变更时不写任何文件
        —— 原来每次 Add 都要重写整个索引（含 docstore 里的全部正文）与整个向量缓存。

        分片是**日志**：行格式 `ids / vectors / deleted`，删除写墓碑（`deleted=1`），
        更新写新行，同 id 后写覆盖先写，故只要按文件名升序回放就能得到权威状态。
        """
        with self._lock:
            if self.backend != "faiss":
                logger.warning("无可保存的 FAISS 索引")
                return
            shard_dir = os.path.join(path, _VEC_SUBDIR)
            if self._cleared:
                # 全量重建过 → 旧分片整体作废，否则 load 会把已清空的向量读回来
                self._wipe_shards(shard_dir)
                self._cleared = False
            if not self._vec_log:
                return
            os.makedirs(shard_dir, exist_ok=True)
            pending = list(self._vec_log)
            for i in range(0, len(pending), _VEC_SHARD_SIZE):
                self._write_shard(shard_dir, self._next_shard_seq(shard_dir),
                                  pending[i:i + _VEC_SHARD_SIZE])
            self._vec_log.clear()
            self._compact_if_needed(shard_dir)
            logger.info(
                "向量分片已落盘: 本次变更 %d 条 → %s（共 %d 个分片）",
                len(pending), shard_dir, len(self._shard_files(shard_dir)),
            )

    # ---- 分片读写辅助 ----

    @staticmethod
    def _shard_files(shard_dir: str) -> List[str]:
        """按序号升序返回分片文件名（回放顺序即权威顺序）"""
        if not os.path.isdir(shard_dir):
            return []
        return sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))

    def _next_shard_seq(self, shard_dir: str) -> int:
        files = self._shard_files(shard_dir)
        if not files:
            return 0
        try:
            return int(os.path.splitext(files[-1])[0]) + 1
        except ValueError:  # 文件名被外部改过：退回按个数续号，不崩
            return len(files)

    @staticmethod
    def _wipe_shards(shard_dir: str) -> None:
        for name in VectorStore._shard_files(shard_dir):
            try:
                os.remove(os.path.join(shard_dir, name))
            except OSError:
                pass

    def _known_dim(self) -> int:
        for vec in self._content_vectors.values():
            return int(np.asarray(vec).shape[0])
        return 0

    def _write_shard(self, shard_dir: str, seq: int, chunk: list) -> None:
        """原子写一个分片。`chunk` 为 [(memory_id, vec 或 None)]。

        墓碑行没有向量，用等长零向量占位 —— npz 是定长数组，不能跳行。
        正文随行落盘（而不是另存 docstore pickle）：这样盘上正文只有
        「YAML / docs.yaml」与分片两份，去掉了 `index.pkl` 这第三份。
        """
        dim = next((np.asarray(v).shape[0] for _, v in chunk if v is not None), 0) or self._known_dim()
        ids = np.array([mid for mid, _ in chunk], dtype=object).astype(str)
        vectors = np.stack([
            np.asarray(v, dtype=np.float32) if v is not None
            else np.zeros(dim, dtype=np.float32)
            for _, v in chunk
        ]) if chunk else np.zeros((0, dim), dtype=np.float32)
        deleted = np.array([v is None for _, v in chunk], dtype=bool)
        contents = np.array([self._contents.get(mid, "") for mid, _ in chunk], dtype=str)

        target = os.path.join(shard_dir, f"{seq:04d}.npz")
        tmp = target + ".tmp"
        # 用文件对象交给 savez：传路径时 numpy 会自作主张补 `.npz` 后缀
        with open(tmp, "wb") as f:
            np.savez(f, ids=ids, vectors=vectors, deleted=deleted, contents=contents)
        os.replace(tmp, target)

    def _compact_if_needed(self, shard_dir: str) -> None:
        """碎片过多时把权威状态合并成单文件，避免分片与墓碑无限累积。

        **只在碎片占比高时才合并**：合并本身是一次全量重写，若分片里大多是存活行，
        合并比继续追加更贵。
        """
        files = self._shard_files(shard_dir)
        if len(files) <= _VEC_SHARD_MAX_FILES:
            return
        total = 0
        for name in files:
            try:
                with open(os.path.join(shard_dir, name), "rb") as f:
                    total += int(np.load(f, allow_pickle=False)["ids"].shape[0])
            except Exception:
                return  # 读不动就别动，宁可留碎片
        if total < 2 * len(self._content_vectors):
            return
        self._wipe_shards(shard_dir)
        live = [(mid, np.asarray(vec, dtype=np.float32))
                for mid, vec in self._content_vectors.items()]
        for i in range(0, len(live), _VEC_SHARD_SIZE):
            self._write_shard(shard_dir, i // _VEC_SHARD_SIZE, live[i:i + _VEC_SHARD_SIZE])
        logger.info("向量分片已合并: %d 条存活（合并前 %d 行）→ %s",
                    len(live), total, shard_dir)

    def load(self, path: str, embeddings: "Embeddings" = None):
        """从分片恢复向量，并**在内存里重建索引**

        **索引不落盘**是这次改动的核心：盘上只需要存向量本身，索引结构与 docstore
        都不必重复存。重建是 O(N) 的内存拷贝（3 万 × 1024 float32 ≈ 120 MB，毫秒级），
        换来的是"每次 Add 不再全量重写"。
        """
        with self._lock:
            if self.backend != "faiss":
                raise ValueError("仅 FAISS 支持加载")
            if embeddings is not None:
                self.embeddings = embeddings
            shard_dir = os.path.join(path, _VEC_SUBDIR)
            if self._shard_files(shard_dir):
                self._load_shards(shard_dir)
                return
            # 旧格式一次性迁移入口：老版本把 `content_vectors.npz` 放在索引目录的**上级**
            legacy = os.path.join(os.path.dirname(path) or ".", _LEGACY_NPZ)
            if os.path.exists(legacy):
                self._load_legacy_npz(legacy)
                logger.info(
                    "向量缓存已按旧格式恢复: %d 条（下次 save 即转为分片）",
                    len(self._content_vectors),
                )
                return
            raise FileNotFoundError(f"向量分片目录不存在: {shard_dir}")

    def _load_shards(self, shard_dir: str) -> None:
        """按序号回放所有分片（后写覆盖先写，墓碑即删）"""
        self._content_vectors = {}
        self._contents = {}
        files = self._shard_files(shard_dir)
        for name in files:
            with open(os.path.join(shard_dir, name), "rb") as f:
                data = np.load(f, allow_pickle=False)
                ids, vectors = data["ids"], data["vectors"]
                deleted = data["deleted"] if "deleted" in data else np.zeros(len(ids), dtype=bool)
                contents = data["contents"] if "contents" in data else None
            for i, (mid, vec, dead) in enumerate(zip(ids, vectors, deleted)):
                mid = str(mid)
                if bool(dead):
                    self._content_vectors.pop(mid, None)
                    self._contents.pop(mid, None)
                else:
                    self._content_vectors[mid] = np.asarray(vec, dtype=np.float32)
                    if contents is not None:
                        self._contents[mid] = str(contents[i])
        self._rebuild_index()
        logger.info("向量分片已恢复: %d 条向量 / %d 个分片", len(self._content_vectors), len(files))

    def _load_legacy_npz(self, vec_path: str) -> None:
        """读取旧格式 `content_vectors.npz`（含 ids/vectors/contents），仅为兼容存量数据"""
        data = np.load(vec_path, allow_pickle=False)
        ids, vectors = data["ids"], data["vectors"]
        self._content_vectors = {
            str(k): np.asarray(v, dtype=np.float32) for k, v in zip(ids, vectors)
        }
        if "contents" in data:
            self._contents = {str(k): str(v) for k, v in zip(ids, data["contents"])}
        else:
            self._contents = {}
        self._rebuild_index()
