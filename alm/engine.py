# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 引擎：user_id 空间注册表 + 同步 Add / Search 编排。

- LLM 与 Embedding 实例全局共享，避免每个 user_id 空间重复初始化
- 空间按 user_id 懒加载并做 LRU 缓存，磁盘上**每个 user_id 独占一个目录**
  （目录名由 user_id 转义而来，目录内是该空间的全部产物，见 `space_dirname`）
"""

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from alm.config import ALMConfig
from alm.contract import AddRequest, SearchRequest
from alm.space import MemorySpace, RetrievalStats, _BackfillGate
from alm.tokens import TokenMeter
from dba_pipeline import throttle
from dba_pipeline.llm.provider import thinking_kwargs

try:
    from langchain_openai import ChatOpenAI
    from dba_pipeline.embedding.store import LocalEmbeddings, OpenAIEmbeddings

    HAS_DEPS = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover
    HAS_DEPS = False
    _IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

# 空间目录名的安全字符集：仅保留字母、数字、下划线、连字符。
# 其余字符（含 ':'、'.'、'/'、'\'、'%' 及非 ASCII）一律按 UTF-8 字节做百分号转义。
# 这样 user_id ↔ 目录名一一对应（可逆、无碰撞），并天然规避路径穿越、
# Windows 非法字符（':' 等）与结尾点号等问题。
_SAFE_FILENAME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)

# 空间目录内的固定文件名。全部产物都在**同一个目录**里，故名字只需在该目录内唯一，
# 不再需要 `<user_id>.xxx` 那种前缀（原来五份产物平铺在 data_dir 下，
# 一个空间一个小目录后 `ls data_dir` 才可读，留存清理也退化成"删一个目录"）。
GRAPH_FILENAME = "graph.yaml"


def space_dirname(user_id: str) -> str:
    """把 user_id 映射为与其一一对应的**空间目录名**（不含扩展名）。

    例：eval:run_abc:locomo:conv-0 → eval%3Arun_abc%3Alocomo%3Aconv-0

    转义规则与"% 编码"同构：`%2E`/`%3A` 这些序列在结果里不可能由别的字符产生，
    故映射可逆、无碰撞，且 `..` → `%2E%2E`（不再是上跳目录）。
    """
    return "".join(
        chr(byte) if chr(byte) in _SAFE_FILENAME_CHARS else f"%{byte:02X}"
        for byte in user_id.encode("utf-8")
    )


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
        # 出站配额节流（字符速率、等待式）**必须先于 LLM/embedding 构造**：LLM 侧靠回调
        # 在 on_llm_start 取令牌，构造时要拿到桶。默认关（两个 env 都为 0 即不生效）。
        throttle.configure(
            embed_chars_per_min=config.embed_chars_per_min,
            llm_chars_per_min=config.llm_chars_per_min,
            rerank_chars_per_min=config.rerank_chars_per_min,
            max_wait=config.throttle_max_wait,
        )
        self.llm = self._build_llm(config, self.token_meter)
        self.embeddings = self._build_embeddings(config)
        # Cross-encoder 重排器（仅纯向量通道；见 config.rerank_enabled）。
        # 构造失败/缺配置时 from_env 返回 None → 该空间不重排（等同关闭），不阻断启动。
        self.reranker = None
        if config.rerank_enabled:
            from dba_pipeline.rerank import Reranker
            self.reranker = Reranker.from_env()
        # 检索余弦统计：跨 user 空间累计，用于标定弃权阈值（见 space.RetrievalStats）
        self.retrieval_stats = RetrievalStats()

        self._spaces: "OrderedDict[str, MemorySpace]" = OrderedDict()
        # 回补槽位闸门（全进程共享，按"最近被检索"优先放行）。放在 Engine 上而不是
        # 类级变量：并发度属于本次运行配置（`ALM_BACKFILL_SLOTS`），不该跨实例泄漏。
        self._backfill_gate = _BackfillGate(config.backfill_slots)
        # 正在被使用的空间（user_id → 租用计数）。换出必须跳过这些，
        # 否则同一 user_id 会出现两个实例并发覆盖写同一个 YAML（见 _lease）。
        self._in_use: Dict[str, int] = {}
        # request_id → Add 结果。调用方对 5xx 有重试，而 add 不是幂等的：
        # 重跑维护会重复写入（并重复消耗抽取/连边的 LLM 成本）。
        self._request_log: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
        self._registry_lock = threading.RLock()

    # ---- 初始化 ----

    @staticmethod
    def _build_llm(config: ALMConfig, meter: Optional[TokenMeter] = None):
        if not config.llm_model:
            raise RuntimeError("缺少 LLM 配置（OPENAI_MODEL）")
        if not config.llm_api_key and not config.llm_base_url:
            raise RuntimeError("缺少 LLM 凭据（OPENAI_API_KEY 或 OPENAI_API_BASE）")
        # 思考模式默认关闭（实测关闭后 output token 约降到 1/16、写入快约 2.5 倍）。该参数只有
        # deepseek 系端点认，其它模型带上会 400，故由 provider.thinking_kwargs 统一判定。
        body = thinking_kwargs(config.llm_model, config.llm_base_url, config.llm_thinking)
        # 输出上限必须显式给：deepseek 端点在 thinking disabled 时不传 max_tokens 就只有
        # 8192，StoryRank 在最难的题上会被截断（0926 实测见 config.llm_max_tokens 注释）。
        #
        # ⚠️ 必须走 `extra_body`，**不能**用 `ChatOpenAI(max_tokens=...)`：langchain-openai
        # 会把该参数**改名成 `max_completion_tokens`**（OpenAI o 系的名字），而 deepseek
        # 端点不认它，于是静默退回自带的 8192 上限——表现为 banner 显示 max_tokens=32768
        # 却依旧 `out=8192` + `finish_reason=length`。实测依据（0926，本地复现）：
        #   ChatOpenAI(max_tokens=32768)._get_request_payload() →
        #     {'model':…, 'stream':False, 'temperature':0.0, 'max_completion_tokens':32768}
        #   同一 prompt 实际 invoke 仍返回 out_tokens=8192 / finish_reason=length。
        # `extra_body` 是**逐字透传**到请求体的，故放在这里。
        if config.llm_max_tokens > 0:
            body.setdefault("extra_body", {})["max_tokens"] = config.llm_max_tokens
        callbacks: List[Any] = []
        if meter is not None:
            callbacks.append(meter)
        _bucket = throttle.llm_bucket()
        if _bucket is not None and throttle.ThrottleCallback is not None:
            # LLM 侧不做 AIMD（回调层拿不到响应状态码），只用配置速率 + 等待。
            callbacks.append(throttle.ThrottleCallback(_bucket))
        return ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key or "not-needed",
            base_url=config.llm_base_url,
            temperature=0,
            # 显式提高瞬时故障重试次数（SDK 默认 2）。云主机到 LLM 端点的连接抖动与
            # 429 较常见，而 Add/Search 各自只允许一次上游重试机会：这里失败会把整个
            # 请求变成 5xx，代价远高于多试一次。
            max_retries=3,
            # 单次请求超时（默认 120s，见 config.llm_timeout）。**不设会吃 SDK 默认的 600s**，
            # 配上上面 3 次重试，一个挂死的上游能把 /search 拖住 40 分钟并一直占着空间锁。
            timeout=config.llm_timeout,
            callbacks=callbacks or None,
            **body,
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

    def _graph_path(self, user_id: str) -> Path:
        """空间内图谱 YAML 的路径：`<data_dir>/<转义 user_id>/graph.yaml`

        索引与文档区由 `MemorySpace` 从该路径的**父目录**派生（见 `_index_paths` /
        `_doc_paths`），故这里只需要给出图谱文件本身。
        """
        return self.config.data_dir / space_dirname(user_id) / GRAPH_FILENAME

    def _get_or_create_locked(self, user_id: str, create: bool) -> Optional[MemorySpace]:
        """注册表内查找或构造空间（调用方须已持有 _registry_lock）。不做换出。"""
        space = self._spaces.get(user_id)
        if space is not None:
            self._spaces.move_to_end(user_id)
            return space

        graph_path = self._graph_path(user_id)
        if not create and not graph_path.exists():
            return None

        space = MemorySpace(
            user_id=user_id,
            config=self.config,
            embeddings=self.embeddings,
            llm=self.llm,
            yaml_path=graph_path,
            stats=self.retrieval_stats,
            reranker=self.reranker,
            backfill_gate=self._backfill_gate,
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
        """LRU 换出：跳过正在被使用的空间（Add 成功后已落盘，换出不会丢数据）

        同时跳过**有后台任务在跑**的空间（`bg_active`，见 `space.start_backfill`）：
        后台回补线程不持租约，若被换出，它会继续在旧实例上整图覆盖写 YAML，
        把换出后重建实例期间写入的内容抹掉。
        """
        while len(self._spaces) > self.config.max_spaces:
            victim = next(
                (uid for uid, sp in self._spaces.items()
                 if uid not in self._in_use and not getattr(sp, "bg_active", False)),
                None,
            )
            if victim is None:
                logger.warning(
                    "空间数 %d 超过上限 %d，但全部在用，暂不换出",
                    len(self._spaces), self.config.max_spaces,
                )
                return
            del self._spaces[victim]
            logger.info("空间 LRU 换出: %s", victim[:4] + "***")

    # ---- 幂等 ----

    def _cached_add(self, user_id: str, request_id: str) -> Optional[Dict[str, int]]:
        with self._registry_lock:
            cached = self._request_log.get((user_id, request_id))
            if cached is not None:
                self._request_log.move_to_end((user_id, request_id))
                return dict(cached)
        return None

    def _remember_add(self, user_id: str, request_id: str, stats: Dict[str, int]) -> None:
        with self._registry_lock:
            key = (user_id, request_id)
            self._request_log[key] = dict(stats)
            self._request_log.move_to_end(key)
            while len(self._request_log) > self.MAX_REQUEST_LOG:
                self._request_log.popitem(last=False)

    # ---- 契约操作 ----

    def add(self, request: AddRequest) -> Dict[str, int]:
        """同步写入：返回时记忆必须已持久化且可被检索。

        幂等：平台对 5xx 有有限次重试，而重跑维护会重复写入记忆，故按 (user_id,
        request_id) 记录已处理的返回结果，重复请求直接回放。键里带 user_id 是为了
        避免「同一 request_id 被两个 user_id 复用」时第二个请求被误判为重复、
        从而静默跳过写入。记录驻留内存并限定容量（见 MAX_REQUEST_LOG），进程重启后
        不保留——重试通常紧跟在失败之后，够用。
        """
        cached = self._cached_add(request.user_id, request.request_id)
        if cached is not None:
            logger.info("Add 重复请求，跳过维护: request_id 已处理")
            return cached

        with self._lease(request.user_id, create=True) as space:
            stats = space.add(request.messages, request.request_id)
            space.save()
        self._remember_add(request.user_id, request.request_id, stats)
        return stats

    def search(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """同步检索：只在该 user_id 的空间内检索。

        `options` 是选择题（含 Streaming）才有的顶层选项；契约允许该字段，故一并透传，
        但**不参与检索**——本系统不在 Search 里做任何面向评分形态的处理（见 space.search）。
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
        """退出前落盘有改动的驻留空间，避免批次缓冲丢失

        只写自上次落盘后被改动的空间（见 MemorySpace._dirty）：无条件重写会把
        所有驻留空间的 mtime 刷成停服时刻，导致留存期清理按 mtime 判定时永远
        找不到过期空间。
        """
        with self._registry_lock:
            for space in self._spaces.values():
                try:
                    space.save()
                except Exception as exc:  # pragma: no cover - 退出路径尽力而为
                    logger.error("空间落盘失败: %s", exc)
            self._spaces.clear()
