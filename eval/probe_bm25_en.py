# -*- coding: utf-8 -*-
"""英文 BM25 稀疏向量验证：对照中文整句单 token 失效机制（报告 §3.1）"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_PYLIBS = ROOT / "eval" / ".cache" / "pylibs"
if _PYLIBS.exists():
    sys.path.insert(0, str(_PYLIBS))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastembed import SparseTextEmbedding

model = SparseTextEmbedding("Qdrant/bm25")
ZH = "用户喜欢喝咖啡提神"
EN = "The user likes drinking coffee to stay awake"
for label, text in [("中文", ZH), ("英文", EN)]:
    vec = list(model.embed(text))[0]
    n_tokens = len(vec.values)
    top = sorted(zip(vec.indices, vec.values), key=lambda x: -x[1])[:8]
    print(f"[{label}] 稀疏 token 数={n_tokens}")
    print(f"    top tokens: {top}")
