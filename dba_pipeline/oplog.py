# SPDX-License-Identifier: AGPL-3.0-only

"""
可追溯操作日志（Operation Log）

统一记录 LLM（通过 MCP 工具）与 DBA 人工（通过 3D 面板 CRUD）对记忆图谱的操作，
以 JSON Lines（每行一条 JSON 记录）追加写入共享日志文件，便于追溯与审计。

- 默认日志路径：<图谱 YAML 所在目录>/operations.log（MCP 与 3D 面板写入同一文件）
- 可通过环境变量 ARIADNE_OPLOG 覆盖路径。

记录结构（每行）：
{
  "ts": "2026-08-29T20:00:00+08:00",      # 操作时间（本地时区，ISO8601）
  "event_id": "hex",                       # 唯一事件 ID
  "source": "llm" | "dba" | "system",      # 操作来源
  "actor": {"type": "...", "...": "..."},  # 操作主体（llm 的 model / dba 的 user）
  "op": "create_node",                     # 操作类型
  "tool": "dba_intervene",                 # MCP 工具名（仅 llm 来源）
  "request": {...},                        # 请求参数（压缩）
  "result": {...},                         # 返回结果摘要（压缩）
}
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone


def default_oplog_path(yaml_path=None) -> str:
    """返回操作日志文件路径（可用环境变量 ARIADNE_OPLOG 覆盖）。"""
    p = os.environ.get("ARIADNE_OPLOG")
    if p:
        return p
    if yaml_path:
        return os.path.join(os.path.dirname(os.path.abspath(yaml_path)), "operations.log")
    return "data/operations.log"


def _summarize(value, depth=0, max_list=4, max_str=200):
    """递归压缩请求/返回值，避免日志文件膨胀（保留关键字段与条数预览）。"""
    if depth > 3:
        return "..."
    if isinstance(value, dict):
        return {k: _summarize(v, depth + 1, max_list, max_str) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > max_list:
            return {
                "count": len(value),
                "preview": [_summarize(v, depth + 1, max_list, max_str) for v in value[:max_list]],
            }
        return [_summarize(v, depth + 1, max_list, max_str) for v in value]
    if isinstance(value, str):
        return value[:max_str] + ("..." if len(value) > max_str else "")
    return value


def log_operation(source, op, request=None, result=None, actor=None,
                  tool=None, path=None):
    """追加一条操作日志。

    - source: "llm" | "dba" | "system"
    - op: 操作类型，如 create_node / dba_intervene / dba_query_memory
    - request: 请求参数 dict
    - result: 返回结果 dict（会被压缩）
    - actor: 操作主体 dict，如 {"type":"llm","model":...} 或 {"type":"dba","user":...}
    - tool: MCP 工具名（可选）
    - path: 日志文件路径；默认 default_oplog_path()
    """
    record = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "event_id": uuid.uuid4().hex,
        "source": source,
        "actor": actor or {},
        "op": op,
        "request": _summarize(request) if request is not None else {},
        "result": _summarize(result) if result is not None else {},
    }
    if tool:
        record["tool"] = tool

    out_path = path or default_oplog_path()
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        # 日志失败不应阻塞主流程
        sys.stderr.write(f"[oplog] 写入失败 {out_path}: {e}\n")


def read_oplog(tail=200, path=None):
    """读取最近的操作日志（JSONL 转 list），供追溯接口使用。"""
    out_path = path or default_oplog_path()
    if not os.path.exists(out_path):
        return []
    rows = []
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-tail:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except OSError:
        return []
    return rows
