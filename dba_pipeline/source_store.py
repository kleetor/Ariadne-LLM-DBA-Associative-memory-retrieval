# SPDX-License-Identifier: AGPL-3.0-only

"""
批次原文存储（opt-in）——为「溯源巡检」保留"节点是从哪段话抽出来的"

为什么需要它：抽取是有损压缩，图上只剩结论。一旦怀疑某条节点是幻觉（比如 txt 里的
`n22`「群可能有多用户共用同一 bot」这种推测被写成事实），没有原文就无法核对，
只能被动提醒或在错误节点上继续抽取。

隐私与体积是它的固有代价，因此：
- **默认关闭**（`ARIADNE_SOURCE_STORE=1` 或 `MemoryDBA(enable_source_store=True)` 才落盘）
- **默认脱敏**：邮箱、手机号、长数字 ID（QQ 号 / User ID）落盘前掩码
- **默认 TTL 30 天**：写入时顺手清理过期批次文件
- **仅批次级**：不做句子级 span（LLM 产出的是消解指代后的陈述，本就不是原文子串）
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

SOURCE_STORE_ENV = "ARIADNE_SOURCE_STORE"
DEFAULT_TTL_DAYS = 30

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")
_AT_ID_RE = re.compile(r"@\d{4,}")


def source_store_enabled() -> bool:
    """环境变量开关（默认关闭）"""
    return os.environ.get(SOURCE_STORE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def default_source_dir(yaml_path) -> Path:
    """与图谱 YAML 同目录的 sources/，便于随图谱一起备份/清理"""
    base = Path(yaml_path).resolve().parent if yaml_path else Path("data")
    return base / "sources"


def redact_identifiers(text: str) -> str:
    """掩掉常见身份标识。

    注意：脱敏只作用于**落盘的原文**，不影响抽取（抽取用的是未脱敏的输入，
    evidence 子串校验也在抽取阶段完成）。因此巡检时看到的原文可能带 `***`。
    """
    if not text:
        return ""
    out = _EMAIL_RE.sub("***@***", text)
    out = _MOBILE_RE.sub("1**********", out)
    out = _AT_ID_RE.sub("@***", out)
    out = _LONG_DIGITS_RE.sub("***", out)
    return out


def new_batch_id() -> str:
    """批次 ID：时间可排序 + 短随机后缀防同秒碰撞"""
    return time.strftime("b%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:4]}"


def save_batch(source_dir, batch_id: str, rounds: List[Dict],
               redact: bool = True, ttl_days: Optional[int] = DEFAULT_TTL_DAYS) -> Optional[Path]:
    """落盘一批原文。

    Args:
        source_dir: 批次目录（由 `default_source_dir(yaml_path)` 得到）
        rounds: [{"ts": Unix毫秒或None, "text": 原文}]，按轮次保留到达时间
    """
    if not batch_id or not rounds:
        return None
    target_dir = Path(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_id": batch_id,
        "created_at": int(time.time() * 1000),
        "redacted": bool(redact),
        "rounds": [
            {
                "ts": r.get("ts"),
                "text": redact_identifiers(r.get("text") or "") if redact else (r.get("text") or ""),
            }
            for r in rounds
        ],
    }
    out_path = target_dir / f"{batch_id}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    if ttl_days:
        purge_expired(source_dir, ttl_days)
    return out_path


def load_batch(source_dir, batch_id: str) -> Optional[Dict]:
    if not batch_id:
        return None
    path = Path(source_dir) / f"{batch_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_batches(source_dir, limit: int = 50) -> List[Dict]:
    """按时间倒序列出已有批次（只读元信息，不含正文）"""
    target_dir = Path(source_dir)
    if not target_dir.exists():
        return []
    batches = []
    for p in sorted(target_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        batches.append({
            "batch_id": data.get("batch_id") or p.stem,
            "created_at": data.get("created_at"),
            "rounds": len(data.get("rounds") or []),
        })
        if limit and len(batches) >= limit:
            break
    return batches


def purge_expired(source_dir, ttl_days: int = DEFAULT_TTL_DAYS) -> int:
    """删除超过 TTL 的批次文件，返回删除数量"""
    target_dir = Path(source_dir)
    if not target_dir.exists() or ttl_days <= 0:
        return 0
    deadline = time.time() - ttl_days * 86400
    removed = 0
    for p in target_dir.glob("*.json"):
        try:
            if p.stat().st_mtime < deadline:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def render_batch_text(batch: Dict) -> str:
    """把一批原文渲染成带轮次时间的文本（供巡检 prompt 使用）"""
    if not batch:
        return ""
    lines = []
    for r in batch.get("rounds") or []:
        ts = r.get("ts")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1000)) if ts else "时间未知"
        lines.append(f"[{stamp}] {r.get('text') or ''}")
    return "\n".join(lines)
