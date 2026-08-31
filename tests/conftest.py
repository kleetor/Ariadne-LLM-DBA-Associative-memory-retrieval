# SPDX-License-Identifier: AGPL-3.0-only

import os
import sys
from pathlib import Path

# 避免 torch 与 FAISS 的 OpenMP 运行时冲突导致进程 Aborted
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 确保项目根目录在 sys.path 中，使 `import dba_pipeline` 在未安装时也可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
