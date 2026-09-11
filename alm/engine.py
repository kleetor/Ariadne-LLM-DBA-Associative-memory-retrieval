# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 引擎：user_id 空间注册表 + 同步 Add / Search 编排。

- LLM 与 Embedding 实例全局共享，避免每个 user_id 空间重复初始化
- 空间按 user_id 懒加载并做 LRU 缓存，磁盘上每个 user_id 一份 YAML
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from alm.config import ALMConfig
from alm.contract import AddRequest, SearchRequest
from alm.space import MemorySpace

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

    def __init__(self, config: ALMConfig):
        if not HAS_DEPS:
            raise RuntimeError(f"ALM 依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e .")

        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.llm = self._build_llm(config)
        self.embeddings = self._build_embeddings(config)

        self._spaces: "OrderedDict[str, MemorySpace]" = OrderedDict()
        self._registry_lock = threading.RLock()

    # ---- 初始化 ----

    @staticmethod
    def _build_llm(config: ALMConfig):
        if not config.llm_model:
            raise RuntimeError("缺少 LLM 配置（OPENAI_MODEL）")
        if not config.llm_api_key and not config.llm_base_url:
            raise RuntimeError("缺少 LLM 凭据（OPENAI_API_KEY 或 OPENAI_API_BASE）")
        return ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key or "not-needed",
            base_url=config.llm_base_url,
            temperature=0,
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

    def get_space(self, user_id: str, create: bool = True) -> Optional[MemorySpace]:
        """获取（或懒加载）某个 user_id 的记忆空间"""
        with self._registry_lock:
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
            )
            self._spaces[user_id] = space
            self._evict_locked()
            return space

    def _evict_locked(self):
        """LRU 换出：Add 成功后已落盘，换出不会丢数据"""
        while len(self._spaces) > self.config.max_spaces:
            evicted_id, _ = self._spaces.popitem(last=False)
            logger.info("空间 LRU 换出: %s", evicted_id[:4] + "***")

    # ---- 契约操作 ----

    def add(self, request: AddRequest) -> Dict[str, int]:
        """同步写入：返回时记忆必须已持久化且可被检索"""
        space = self.get_space(request.user_id, create=True)
        stats = space.add(request.messages)
        space.save()
        return stats

    def search(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """同步检索：只在该 user_id 的空间内检索"""
        space = self.get_space(request.user_id, create=False)
        if space is None:
            return []
        return space.search(request.query, request.top_k)

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
