# SPDX-License-Identifier: AGPL-3.0-only

"""ALM（Agent Memory Leaderboard）专项接入模块。

对外只提供 ALM 契约要求的同步 Add / Search HTTP 接口，
内部复用 Ariadne 的图谱、DBA 维护与 PAR 检索链路。
"""

import os

# 避免 torch 与 FAISS 的 OpenMP 运行时冲突导致进程 Aborted（与 mcp_server 保持一致）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
