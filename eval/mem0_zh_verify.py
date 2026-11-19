# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Mem0 中文补丁「硬门槛」验证脚本（§6.3）

验证不依赖 BM25 模型的机制层（§6.3.1 / §6.3.3），以及补丁是否正确挂到
mem0.memory.main 命名空间（主检索真正调用的位置）：

1. §6.3.1 词级验证（机制层）：原始 `lemmatize_for_bm25` 对中文整句返回单 token；
   补丁后 jieba 输出经 fastembed SimpleTokenizer（按空白切分）产生 ≥3 token。
2. §6.3.3 补丁挂载断言：`mem0.memory.main.lemmatize_for_bm25 / extract_entities`
   已是中文版函数（否则 main.search() 仍走旧函数）。
3. 实体命中（需 zh_core_web_sm，未装则降级为「spaCy 不可用 → extract_entities 返回 []，
   实体 boost 无法验证」，如实标注）。

模型依赖：Qdrant/bm25 的稀疏向量"非零维度数"需下载 BM25 模型（网络受限时跳到 §6.3.1 的
tokenizer 级证明）。本脚本不含模型下载。

用法:
    python -m eval.mem0_zh_verify
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(ROOT / "eval" / ".cache" / "mem0_home"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_PYLIBS = ROOT / "eval" / ".cache" / "pylibs"
if _PYLIBS.exists() and str(_PYLIBS) not in sys.path:
    sys.path.insert(0, str(_PYLIBS))

from fastembed.sparse.utils.tokenizer import SimpleTokenizer  # noqa: E402

import frameworks.mem0_runner_zh_patch as zh  # noqa: E402
import mem0.utils.lemmatization as _lem  # noqa: E402
import mem0.memory.main as _mm  # noqa: E402
import mem0.utils.entity_extraction as _ee  # noqa: E402

# 补丁前捕获原始行为（apply_patch 之前）
_ORIG_LEMMATIZE = _lem.lemmatize_for_bm25

SAMPLES = ["用户喜欢喝咖啡提神", "项目经常需要加班赶进度", "周末和朋友玩桌游"]


def main() -> None:
    # 先挂补丁（覆盖源模块 + mem0.memory.main 命名空间）
    zh.apply_patch()

    print("=" * 60)
    print("§6.3.1 词级验证（机制层：lemmatize → SimpleTokenizer 按空白分词）")
    print("=" * 60)
    ok = True
    for s in SAMPLES:
        before = SimpleTokenizer.tokenize(_ORIG_LEMMATIZE(s))
        after = SimpleTokenizer.tokenize(zh.lemmatize_for_bm25_zh(s))
        status = "PASS" if len(after) >= 3 else "FAIL"
        ok = ok and (len(after) >= 3)
        print(f"  {s}")
        print(f"    BEFORE tokens={len(before):>2}  {before}")
        print(f"    AFTER  tokens={len(after):>2}  {after}  -> {status}")
    print(f"  §6.3.1 总判定: {'PASS' if ok else 'FAIL'}\n")

    print("=" * 60)
    print("§6.3.3 补丁挂载断言（main 命名空间 = 主检索实际调用处）")
    print("=" * 60)
    assert _mm.lemmatize_for_bm25 is zh.lemmatize_for_bm25_zh, "main.lemmatize 未覆盖!"
    assert _mm.extract_entities is zh.extract_entities_zh, "main.extract_entities 未覆盖!"
    assert _lem.lemmatize_for_bm25 is zh.lemmatize_for_bm25_zh, "源模块 lemmatize 未覆盖!"
    assert _ee.extract_entities is zh.extract_entities_zh, "源模块 extract_entities 未覆盖!"
    print("  PASS: mem0.memory.main.lemmatize_for_bm25 / extract_entities 均为中文版\n")

    print("=" * 60)
    print("§6.3.2/§6.3.3 实体命中（需 zh_core_web_sm）")
    print("=" * 60)
    try:
        import spacy  # noqa: F401
    except ImportError:
        print("  spaCy 未安装 -> extract_entities_zh 返回 []（实体 boost 无法实测，见计划风险表）")
    else:
        ents = zh.extract_entities_zh("用户和恋人苏婉婷去过瑞幸咖啡")
        print(f"  实体抽取: {ents}  ({'PASS' if ents else '无命中，需检查 zh model'})")


if __name__ == "__main__":
    main()
