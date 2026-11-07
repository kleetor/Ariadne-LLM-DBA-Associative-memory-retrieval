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
from typing import Dict, List, Optional, Tuple

SOURCE_STORE_ENV = "ARIADNE_SOURCE_STORE"
DEFAULT_TTL_DAYS = 30

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")
_AT_ID_RE = re.compile(r"@\d{4,}")
# 批次 ID 白名单：只允许字母数字与 . _ -，且不得为 ..（配合下面的显式排除防路径穿越）
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# 对话文本的说话人标注：`[昵称] 原话`
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[([^\[\]\n]{1,40})\]\s*")
# 标注里的 QQ 号：`（QQ 3512593260，群「…」成员）`——全角/半角括号都认
_INLINE_QQ_RE = re.compile(r"[（(]\s*QQ\s*[:：]?\s*(\d{5,12})[^）)]*[）)]", re.IGNORECASE)
# 形如时间的前缀（`[2026-11-04 14:56:52]` / `[14:56]`）不是说话人
_TIMESTAMP_LIKE_RE = re.compile(r"\d{1,2}:\d{2}|^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")


def _valid_batch_id(batch_id: str) -> bool:
    return bool(batch_id) and bool(_BATCH_ID_RE.match(batch_id)) and ".." not in batch_id


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


def _parse_speaker_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """解析单行标注，返回 (昵称, QQ号)；不是说话人标注时返回 (None, None)。

    可识别：
        [零] （QQ 3512593260，群「ai、零、Shark」成员）666
        [零（QQ 3512593260）] 666
        [Shark] 你的记忆目前是空的
    """
    m = _SPEAKER_PREFIX_RE.match(line)
    if not m:
        return None, None
    label = m.group(1).strip()
    rest = line[m.end():]

    qq = None
    # QQ 写在方括号内：`[零（QQ 123）]`
    inner = _INLINE_QQ_RE.search(label)
    if inner:
        qq = inner.group(1)
        label = (label[:inner.start()] + label[inner.end():]).strip()
    else:
        # QQ 写在紧随其后的圆括号里：`[零] （QQ 123…）原话`
        outer = _INLINE_QQ_RE.match(rest)
        if outer:
            qq = outer.group(1)

    if not label or _TIMESTAMP_LIKE_RE.search(label):
        return None, None
    return label, qq


def extract_speaker_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从一轮对话的标注里解析 (说话人昵称, 稳定 ID)。

    说话人是**原文层属性**（谁说了这句话），不写进节点：一个节点可能涉及多个
    主体（如「Shark 和零下周末去漫展」），节点级 speaker 没有意义。

    只在这一轮所有非空行的解析结果**一致**时返回对应值；多人混排、缺标注、
    或标注其实是时间戳（`[14:56]`）时返回 None——宁可不记，也不要记错归属。
    昵称与 ID 各自独立判定：多人混排但 ID 相同（同一人换了昵称）仍能拿到 ID。
    """
    if not text:
        return None, None
    speakers: List[str] = []
    ids: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, qq = _parse_speaker_line(line)
        if speaker is None:
            return None, None
        speakers.append(speaker)
        if qq:
            ids.append(qq)
    if not speakers:
        return None, None
    speaker = speakers[0] if len(set(speakers)) == 1 else None
    unique_ids = set(ids)
    speaker_id = unique_ids.pop() if len(unique_ids) == 1 else None
    return speaker, speaker_id


def save_batch(source_dir, batch_id: str, rounds: List[Dict],
               redact: bool = True, ttl_days: Optional[int] = DEFAULT_TTL_DAYS) -> Optional[Path]:
    """落盘一批原文。

    Args:
        source_dir: 批次目录（由 `default_source_dir(yaml_path)` 得到）
        rounds: [{"ts": Unix毫秒或None, "text": 原文}]，按轮次保留到达时间；
            speaker / speaker_id 由文本标注自动解析（`[昵称]（QQ 号）原话`），
            也可显式给出同名字段覆盖。

    注意：`text` 默认脱敏（QQ 号会被掩码），但 `speaker_id` 是**结构化字段**、
    按原样落盘——它存在的意义就是稳定标识，掩码后就没有价值了。批次目录与
    图谱同处一份存储，TTL 到期一起清理。
    """
    if not batch_id or not rounds:
        return None
    # batch_id 会拼进文件名：非法/带路径分隔符的值直接拒绝，防目录穿越
    if not _valid_batch_id(batch_id):
        raise ValueError(f"非法批次 ID: {batch_id!r}")
    target_dir = Path(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_rounds = []
    for r in rounds:
        raw_text = r.get("text") or ""
        speaker, speaker_id = extract_speaker_info(raw_text)
        stored_rounds.append({
            "ts": r.get("ts"),
            "speaker": r.get("speaker") or speaker,
            "speaker_id": r.get("speaker_id") or speaker_id,
            "text": redact_identifiers(raw_text) if redact else raw_text,
        })
    payload = {
        "batch_id": batch_id,
        "created_at": int(time.time() * 1000),
        "redacted": bool(redact),
        "rounds": stored_rounds,
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
    if not _valid_batch_id(batch_id):
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
    """删除超过 TTL 的批次文件，返回删除数量

    同时清理写盘中断残留的 `*.json.tmp`：`list_batches` 与下面的 `*.json` 都匹配不到它，
    不主动清就会永久堆积。两者共用同一个 TTL 判定，正在写入的临时文件不会被误删。
    """
    target_dir = Path(source_dir)
    if not target_dir.exists() or ttl_days <= 0:
        return 0
    deadline = time.time() - ttl_days * 86400
    removed = 0
    for pattern in ("*.json", "*.json.tmp"):
        for p in target_dir.glob(pattern):
            try:
                if p.stat().st_mtime < deadline:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


def render_batch_text(batch: Dict) -> str:
    """把一批原文渲染成带轮次时间与说话人的文本（供巡检 prompt 使用）

    说话人必须渲染出来：巡检要判断的正是"这条节点是不是这个人说的、有没有被
    安到别人头上"，只给原文的话 LLM 只能从正文里的自称去猜。
    """
    if not batch:
        return ""
    lines = []
    for r in batch.get("rounds") or []:
        ts = r.get("ts")
        # ts 若被误写成字符串，ts / 1000 会抛 TypeError；非数值一律按"时间未知"处理
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            stamp = "时间未知"
        else:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1000)) if ts else "时间未知"
        text = r.get("text") or ""
        speaker = r.get("speaker")
        # 正文通常已自带 `[昵称]` 标注，此时不再重复前缀
        if speaker and not text.lstrip().startswith(f"[{speaker}]"):
            stamp = f"{stamp} [{speaker}]"
        lines.append(f"[{stamp}] {text}")
    return "\n".join(lines)
