# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 引擎：user_id 空间注册表 + 同步 Add / Search 编排。

- LLM 与 Embedding 实例全局共享，避免每个 user_id 空间重复初始化
- 空间按 user_id 懒加载并做 LRU 缓存，磁盘上每个 user_id 一份 YAML
"""

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from alm.config import ALMConfig
from alm.contract import AddRequest, SearchRequest
from alm.space import MemorySpace, RetrievalStats
from alm.tokens import TokenMeter

try:
    from langchain_openai import ChatOpenAI
    from dba_pipeline.embedding.store import LocalEmbeddings, OpenAIEmbeddings

    HAS_DEPS = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover
    HAS_DEPS = False
    _IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

# YAML 文件名的安全字符集：仅保留字母、数字、下划线、连字符。
# 其余字符（含 ':'、'.'、'/'、'\'、'%' 及非 ASCII）一律按 UTF-8 字节做百分号转义。
# 这样 user_id ↔ 文件名一一对应（可逆、无碰撞），并天然规避路径穿越、
# Windows 非法字符（':' 等）与结尾点号等问题。
_SAFE_FILENAME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


def space_filename(user_id: str) -> str:
    """把 user_id 映射为与其匹配的 YAML 文件名。

    例：eval:run_abc:locomo:conv-0 → eval%3Arun_abc%3Alocomo%3Aconv-0.yaml
    """
    escaped = "".join(
        chr(byte) if chr(byte) in _SAFE_FILENAME_CHARS else f"%{byte:02X}"
        for byte in user_id.encode("utf-8")
    )
    return f"{escaped}.yaml"


class ALMEngine:
    """ALM 契约的同步记忆引擎"""

    # 幂等记录的容量上限（按 request_id 记录已处理请求的返回结果）
    MAX_REQUEST_LOG = 8192

    def __init__(self, config: ALMConfig):
        if not HAS_DEPS:
            raise RuntimeError(f"ALM 依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e .")

        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        # token 计量挂在共享 LLM 实例上，Add / Search 的全部调用都被覆盖
        self.token_meter = TokenMeter.from_env()
        self.llm = self._build_llm(config, self.token_meter)
        self.embeddings = self._build_embeddings(config)
        # 检索余弦统计：跨 user 空间累计，用于标定弃权阈值（见 space.RetrievalStats）
        self.retrieval_stats = RetrievalStats()

        self._spaces: "OrderedDict[str, MemorySpace]" = OrderedDict()
        # 正在被使用的空间（user_id → 租用计数）。换出必须跳过这些，
        # 否则同一 user_id 会出现两个实例并发覆盖写同一个 YAML（见 _lease）。
        self._in_use: Dict[str, int] = {}
        # request_id → Add 结果。平台对 5xx 有重试，而 add 不是幂等的：
        # 重跑维护会重复写入（重复写入的成本见 Plan 里 §7.8 的记录）。
        self._request_log: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
        self._registry_lock = threading.RLock()

    # ---- 初始化 ----

    @staticmethod
    def _build_llm(config: ALMConfig, meter: Optional[TokenMeter] = None):
        if not config.llm_model:
            raise RuntimeError("缺少 LLM 配置（OPENAI_MODEL）")
        if not config.llm_api_key and not config.llm_base_url:
            raise RuntimeError("缺少 LLM 凭据（OPENAI_API_KEY 或 OPENAI_API_BASE）")
        return ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key or "not-needed",
            base_url=config.llm_base_url,
            temperature=0,
            callbacks=[meter] if meter is not None else None,
        )

    @staticmethod
    def _build_embeddings(config: ALMConfig):
        if not config.embedding_model:
            raise RuntimeError("缺少 Embedding 配置（EMBEDDING_MODEL），Search 无法工作")
        if config.embedding_local:
            return LocalEmbeddings(model_name=config.embedding_model)
        return OpenAIEmbeddings(
            api_key=config.embedding_api_key or config.llm_api_key or "",
            api_base=config.embedding_base_url or config.llm_base_url or "",
            model=config.embedding_model,
        )

    # ---- 空间管理 ----

    def _yaml_path(self, user_id: str) -> Path:
        """user_id 直接映射为文件名，与检索作用域一一对应"""
        return self.config.data_dir / space_filename(user_id)

    def _get_or_create_locked(self, user_id: str, create: bool) -> Optional[MemorySpace]:
        """注册表内查找或构造空间（调用方须已持有 _registry_lock）。不做换出。"""
        space = self._spaces.get(user_id)
        if space is not None:
            self._spaces.move_to_end(user_id)
            return space

        yaml_path = self._yaml_path(user_id)
        if not create and not yaml_path.exists():
            return None

        space = MemorySpace(
            user_id=user_id,
            config=self.config,
            embeddings=self.embeddings,
            llm=self.llm,
            yaml_path=yaml_path,
            stats=self.retrieval_stats,
        )
        self._spaces[user_id] = space
        return space

    @contextmanager
    def _lease(self, user_id: str, create: bool):
        """租用某个 user_id 的空间：租用期内该空间不会被 LRU 换出。

        为什么需要租约：换出只把空间从注册表移除，不检查它是否正被别的线程使用。
        若无租约，下面的序列会丢数据——
            线程 A 取到 user U 的空间 S1，正在写（含 LLM 维护，实测数十秒）；
            同期其它 user 的空间数超过上限，U 被换出；
            线程 B 再请求 U，注册表已无 U → 从磁盘重建出**第二个实例 S2**；
            S1 完成后仍会整图覆盖写 YAML，把 S2 期间写入的内容抹掉。
        租约让换出跳过在用空间（全部在用时不换出，宁可控不住内存也不丢数据）。
        """
        with self._registry_lock:
            space = self._get_or_create_locked(user_id, create)
            if space is None:
                yield None
                return
            self._in_use[user_id] = self._in_use.get(user_id, 0) + 1

        try:
            yield space
        finally:
            with self._registry_lock:
                remaining = self._in_use.get(user_id, 1) - 1
                if remaining > 0:
                    self._in_use[user_id] = remaining
                else:
                    self._in_use.pop(user_id, None)
                self._evict_locked()

    def get_space(self, user_id: str, create: bool = True) -> Optional[MemorySpace]:
        """获取（或懒加载）某个 user_id 的记忆空间。

        返回的对象**不享受换出保护**，仅供只读排查使用；写操作请走 add()。
        """
        with self._registry_lock:
            space = self._get_or_create_locked(user_id, create)
            self._evict_locked()
            return space

    def _evict_locked(self):
        """LRU 换出：跳过正在被使用的空间（Add 成功后已落盘，换出不会丢数据）"""
        while len(self._spaces) > self.config.max_spaces:
            victim = next((uid for uid in self._spaces if uid not in self._in_use), None)
            if victim is None:
                logger.warning(
                    "空间数 %d 超过上限 %d，但全部在用，暂不换出",
                    len(self._spaces), self.config.max_spaces,
                )
                return
            del self._spaces[victim]
            logger.info("空间 LRU 换出: %s", victim[:4] + "***")

    # ---- 幂等 ----

    def _cached_add(self, request_id: str) -> Optional[Dict[str, int]]:
        with self._registry_lock:
            cached = self._request_log.get(request_id)
            if cached is not None:
                self._request_log.move_to_end(request_id)
                return dict(cached)
        return None

    def _remember_add(self, request_id: str, stats: Dict[str, int]) -> None:
        with self._registry_lock:
            self._request_log[request_id] = dict(stats)
            self._request_log.move_to_end(request_id)
            while len(self._request_log) > self.MAX_REQUEST_LOG:
                self._request_log.popitem(last=False)

    # ---- 契约操作 ----

    def add(self, request: AddRequest) -> Dict[str, int]:
        """同步写入：返回时记忆必须已持久化且可被检索。

        幂等：平台对 5xx 有有限次重试，而重跑维护会重复写入记忆，故按 request_id
        记录已处理的返回结果，重复请求直接回放。记录驻留内存并限定容量（见
        MAX_REQUEST_LOG），进程重启后不保留——重试通常紧跟在失败之后，够用。
        """
        cached = self._cached_add(request.request_id)
        if cached is not None:
            logger.info("Add 重复请求，跳过维护: request_id 已处理")
            return cached

        with self._lease(request.user_id, create=True) as space:
            stats = space.add(request.messages)
            space.save()
        self._remember_add(request.request_id, stats)
        return stats

    def search(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """同步检索：只在该 user_id 的空间内检索。

        `options` 是选择题（含 Streaming）才有的顶层选项，透传给检索层参与选择题保护
        与选项对比聚合；开放题为 None。
        """
        with self._lease(request.user_id, create=False) as space:
            if space is None:
                return []
            return space.search(request.query, request.top_k, request.options)

    # ---- 运维 ----

    def space_count(self) -> int:
        with self._registry_lock:
            return len(self._spaces)

    def close(self):
        """退出前落盘所有驻留空间，避免批次缓冲丢失"""
        with self._registry_lock:
            for space in self._spaces.values():
                try:
                    space.save()
                except Exception as exc:  # pragma: no cover - 退出路径尽力而为
                    logger.error("空间落盘失败: %s", exc)
            self._spaces.clear()
