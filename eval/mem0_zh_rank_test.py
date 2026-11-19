# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Mem0 中文补丁 §6.3.2「排序分化」硬门槛：qdrant(BM25+向量) 是否不再等于 faiss(纯向量)。

设计（同一进程内切换补丁，三个独立存储）：
- A qdrant（补丁 OFF）：入库时 lemmatize=英文原样 → 中文整句 → BM25 稀疏 1 维 → 检索 = 纯向量，
  应等于 faiss（复现 0819「35/35 同序」现象）；
- B qdrant（补丁 ON）：入库时 lemmatize=jieba → 中文词级 → BM25 稀疏多维 → 检索混合，
  应与 A/faiss 不同（BM25 信号真实参与）；
- faiss（无 BM25，纯向量）：作为纯向量基线。

断言：
  ids_A == ids_faiss   （无补丁 qdrant == 纯向量：bm25 静默失效的复现）
  ids_B != ids_faiss   （有补丁 qdrant 与纯向量分化：bm25 生效）

注意：bm25 稀疏向量在「入库(add)」时按 lemmatize 后的 text_lemmatized 预计算，
所以必须在不同 collection 上分别用「补丁 ON/OFF」入库，才能体现差异。

用法:
    python -m eval.mem0_zh_rank_test
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import mem0 之前
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(ROOT / "eval" / ".cache" / "mem0_home"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import yaml  # noqa: E402

import frameworks.mem0_runner_zh_patch as zh  # noqa: E402
import mem0.memory.main as _mm  # noqa: E402
import mem0.utils.lemmatization as _lem  # noqa: E402
from mem0 import Memory  # noqa: E402

# 补丁前捕获原始 lemmatize（用于回切到"无补丁"态）
_ORIG_LEMMATIZE = _lem.lemmatize_for_bm25

EMB_MODEL = "BAAI/bge-large-zh-v1.5"
EMB_DIMS = 1024
USER_ID = "rank_test"

# 一组强语义关联的中文记忆 + 一条能触发 BM25 词面信号的中文查询
MEMORIES = [
    "用户在一家互联网公司做后端开发",
    "项目经常需要加班赶进度",
    "用户经常加班到晚上11点",
    "加班导致颈椎经常酸痛",
    "用户喜欢喝咖啡提神",
    "拿铁是用户最常点的咖啡",
    "公司附近有一家24小时便利店",
]
QUERY = "用户喜欢喝什么咖啡提神"


def build_config(store: str, collection: str) -> dict:
    if store == "qdrant":
        vstore = {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "path": str(ROOT / "eval" / ".cache" / "mem0_qdrant_zh"),
                "embedding_model_dims": EMB_DIMS,
                "on_disk": True,
            },
        }
    else:
        vstore = {
            "provider": "faiss",
            "config": {
                "collection_name": collection,
                "path": str(ROOT / "eval" / ".cache" / "mem0_faiss_zh"),
                "embedding_model_dims": EMB_DIMS,
                "distance_strategy": "cosine",
            },
        }
    # mem0 即使 infer=False 也强制要求 llm 配置（不用其生成，仅满足工厂）
    llm = {
        "provider": "openai",
        "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
            "temperature": 0,
            "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
            "openai_base_url": os.environ.get("OPENAI_API_BASE"),
        },
    }
    return {
        "vector_store": vstore,
        "llm": llm,
        "embedder": {"provider": "huggingface", "config": {"model": EMB_MODEL, "embedding_dims": EMB_DIMS}},
    }


def add_and_search(config: dict, memory_texts: list, query: str):
    m = Memory.from_config(config)
    for t in memory_texts:
        m.add({"role": "user", "content": t}, user_id=USER_ID, infer=False)
    out = m.search(query, top_k=5, filters={"user_id": USER_ID}, threshold=0.0)
    ids = [str(x.get("memory", "")) for x in out.get("results", [])]
    return ids, [round(float(x.get("score", 0.0)), 4) for x in out.get("results", [])]


def _patch(on: bool):
    if on:
        zh.apply_patch()
    else:
        # 回切到补丁前的原始函数（覆盖源模块 + main 命名空间）
        _lem.lemmatize_for_bm25 = _ORIG_LEMMATIZE
        _mm.lemmatize_for_bm25 = _ORIG_LEMMATIZE


def _load_dotenv_local(env_path: Path) -> None:
    """本地加载 .env（不触发 run_p_baseline 的 HF_HUB_OFFLINE=1 全局离线）。"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        key, val = k.strip(), v.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def main() -> None:
    # 确保不做全局离线：清除 run_p_baseline 可能遗留的 HF_HUB_OFFLINE
    os.environ.pop("HF_HUB_OFFLINE", None)
    _load_dotenv_local(ROOT / ".env")

    print("=" * 62)
    print("§6.3.2 排序分化验证（qdrant 无补丁 / qdrant 有补丁 / faiss）")
    print("=" * 62)

    # A: qdrant 无补丁（复现 0819：bm25 中文失效 → == 纯向量）
    _patch(False)
    idsA, _ = add_and_search(build_config("qdrant", "col_a"), MEMORIES, QUERY)
    print(f"\nA qdrant(无补丁) : {idsA}")

    # B: qdrant 有补丁（bm25 中文生效 → 与纯向量分化）
    _patch(True)
    idsB, _ = add_and_search(build_config("qdrant", "col_b"), MEMORIES, QUERY)
    print(f"B qdrant(有补丁) : {idsB}")

    # faiss（纯向量基线；faiss 无 bm25，补丁对它无影响）
    idsF, _ = add_and_search(build_config("faiss", "col_f"), MEMORIES, QUERY)
    print(f"F faiss(纯向量)  : {idsF}")

    print("\n" + "=" * 62)
    okA = idsA == idsF
    okB = idsB != idsF
    print(f"[复现] A(无补丁 qdrant) == F(faiss)  -> {'PASS' if okA else 'FAIL'}")
    print(f"[分化] B(有补丁 qdrant) != F(faiss)  -> {'PASS' if okB else 'FAIL'}")
    print(f"§6.3.2 总判定: {'PASS' if (okA and okB) else 'FAIL'}")


if __name__ == "__main__":
    main()
