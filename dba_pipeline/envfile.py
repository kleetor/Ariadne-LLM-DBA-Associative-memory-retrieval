# SPDX-License-Identifier: AGPL-3.0-only

"""最小 ``.env`` 加载器（不依赖 python-dotenv）。

面板与 MCP 都需要读取项目根目录的 ``.env``（模型密钥、鉴权凭据等），
故抽成公共模块，避免两处各写一份、只生效一半。

规则：

- **已存在的环境变量优先，不覆盖**——命令行/系统里显式设置的值不会被文件里的值顶掉；
- 编码按 ``utf-8-sig``（顺带吃掉 BOM）→ ``gbk`` → ``latin-1`` 依次尝试，
  单个字符解不出来也不会中断启动（Windows 下 .env 常见 GBK）；
- 行尾 ``#`` 注释按标准 dotenv 行为剥离：``ARIADNE_PORT=8766  # MCP 端口``
  的值是 ``8766``，而不是把整段注释当成值。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# utf-8-sig 顺带吃掉 BOM（否则首个键会变成 "\ufeffKEY" 而永远读不到）；
# latin-1 兜底，保证任何字节序列都能读出一个（可能乱码但不报错的）文本。
_ENCODINGS = ("utf-8-sig", "gbk", "latin-1")


def find_dotenv() -> Optional[Path]:
    """从本文件向上查找项目根目录的 .env。"""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _read_text(path: Path) -> str:
    """按候选编码读取 .env；非 UTF-8 时告警（否则只会静默读出乱码）。"""
    for encoding in _ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
        if encoding != "utf-8-sig":
            print(f"提示: {path} 不是 UTF-8，已按 {encoding} 读取", file=sys.stderr)
        return text
    return ""


def _parse_value(raw: str) -> str:
    """解析等号右侧：去引号、去行尾注释。

    只有「前面是空白」的 ``#`` 才算注释起点，因此 ``sk-a#b`` 这种值不会被截断。
    """
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1].isspace():
            return value[:i].strip()
    return value


def load_dotenv() -> None:
    """把 .env 灌入环境变量（只填空缺，不覆盖已有值）。"""
    path = find_dotenv()
    if path is None:
        return
    text = _read_text(path)
    if not text:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, _parse_value(raw_value))
