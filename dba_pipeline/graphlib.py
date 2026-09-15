# SPDX-License-Identifier: AGPL-3.0-only

"""图谱库 —— 图谱目录内的多份图谱文件，以及两个进程共享的「当前图谱」指针。

**为什么约定「图谱库 = 某目录下的 *.yaml」**

面板（``viz``）与 MCP 是两个独立进程，都可能需要切换正在使用的图谱。二者的
运行期状态全部按**图谱所在目录**取，与文件名无关：

    auth.json / .ariadne_secret / operations.log / ariadne.log /
    retrieval_params.yaml           → 目录固定，与图谱文件名无关
    <yaml>.lock                     → 随图谱文件名变（每份图谱一把写锁）

因此只要在**同一目录内**切换图谱，登录凭据、会话密钥、日志与检索参数都不受影响，
是零迁移操作。这也是本模块把图谱库定义为「一个目录」的原因。

**两个进程怎么协作**

MCP 的 SSE 服务不暴露控制端点，也没有会话签名器，面板无法直接调用它。故二者通过
``<图谱目录>/active_graph.json`` 交换「切换意图」与「应用回执」：

    面板：request_switch()  → generation + 1, status = pending
    MCP ：pending_request() → 发现有未应用的请求 → 应用 → report_applied()

指针文件的读-改-写统一在 ``<图谱目录>/active_graph.lock`` 内完成，并原子替换。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yaml

from dba_pipeline import filelock

POINTER_FILENAME = "active_graph.json"
LOCK_FILENAME = "active_graph.lock"
GRAPH_SUFFIXES = (".yaml", ".yml")

# 指针文件里两个方向各自的字段：
#   panel_active —— 面板自己当前用的那份（面板内切换时写）
#   active       —— MCP 的切换目标（配合 generation/applied_generation）
PANEL_FIELD = "panel_active"
MCP_FIELD = "active"

# 兜底的空图谱占位文件：首次启动、指针与启动参数都没给出可用图谱时用它
DEFAULT_GRAPH_NAME = "sample_graph.yaml"
EMPTY_GRAPH_TEXT = "nodes: []\nedges: []\n"

# 除路径分隔符外，还要挡掉 Windows 的盘符/ADS 写法（`C:foo.yaml` 会被解释成
# `C` 的备用数据流），故一并禁止冒号。
_INVALID_NAME_CHARS = re.compile(r"[\\/:\x00-\x1f]")


def graph_dir(yaml_path: Optional[str]) -> str:
    """图谱目录（绝对路径）。所有共享运行期状态都以它为锚点。"""
    return os.path.dirname(os.path.abspath(yaml_path)) if yaml_path else "."


def validate_name(name: str) -> Optional[str]:
    """校验库内图谱名（纯文件名）；合法返回 None，否则返回错误文案。"""
    if not name or name != os.path.basename(name):
        return "图谱名只能是文件名，不能包含路径"
    if _INVALID_NAME_CHARS.search(name):
        return "图谱名不能包含路径分隔符或冒号"
    if name.startswith("."):
        return "图谱名不能以点开头"
    if not name.lower().endswith(GRAPH_SUFFIXES):
        return "图谱文件必须以 .yaml 或 .yml 结尾"
    return None


def graph_path(yaml_path: Optional[str], name: str) -> str:
    """库内图谱的绝对路径（调用前应先用 validate_name 校验）。"""
    return graph_path_in(graph_dir(yaml_path), name)


def graph_path_in(directory: str, name: str) -> str:
    return os.path.join(os.path.abspath(directory), name)


# ---------------------------------------------------------------------------
# 启动时用哪份图谱
# ---------------------------------------------------------------------------

def resolve_startup(graph_dir: str, field: str, hint: Optional[str] = None) -> tuple:
    """决定启动时载入哪份图谱，返回 ``(绝对路径, 来源说明)``。

    优先级：**共享指针里记的上次选择 → 启动参数 hint → 空图谱占位文件**。

    指针优先是刻意的：启动参数（本地是命令行 ``--yaml``、Docker 是 ``ARIADNE_YAML``）
    只是「第一次用哪份」的提示，不该每次重启都把用户的面板选择顶掉，否则面板与
    MCP 会各自回到命令行指定的那份，出现静默分歧。
    """
    directory = os.path.abspath(graph_dir)
    name = str(read_pointer_in(directory).get(field) or "")
    if name and os.path.isfile(graph_path_in(directory, name)):
        if inspect_graph(graph_path_in(directory, name)) is not None:
            return graph_path_in(directory, name), "指针记录"

    if hint:
        hint_path = os.path.abspath(hint)
        if inspect_graph(hint_path) is not None:
            return hint_path, "启动参数"

    return graph_path_in(directory, DEFAULT_GRAPH_NAME), "空图谱占位"


def ensure_placeholder(path: str) -> str:
    """确保空图谱占位文件存在（面板的增删改需要一个真实可写的落点）。"""
    if not os.path.isfile(path):
        _write_text_atomic(path, EMPTY_GRAPH_TEXT)
    return path


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# 指针文件
# ---------------------------------------------------------------------------

def pointer_path(yaml_path: Optional[str]) -> str:
    return pointer_path_in(graph_dir(yaml_path))


def pointer_path_in(directory: str) -> str:
    return os.path.join(os.path.abspath(directory), POINTER_FILENAME)


def read_pointer(yaml_path: Optional[str]) -> Dict:
    return read_pointer_in(graph_dir(yaml_path))


def read_pointer_in(directory: str) -> Dict:
    try:
        with open(pointer_path_in(directory), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def patch_pointer(yaml_path: Optional[str], **fields) -> Dict:
    """在跨进程锁内「读-改-写」指针文件。"""
    return patch_pointer_in(graph_dir(yaml_path), **fields)


def patch_pointer_in(directory: str, **fields) -> Dict:
    lock = filelock.FileLock(os.path.join(os.path.abspath(directory), LOCK_FILENAME),
                             timeout=10.0)
    with lock.hold():
        data = read_pointer_in(directory)
        data.update(fields)
        data.setdefault("version", 1)
        _atomic_write_json(pointer_path_in(directory), data)
    return data


def list_graphs(yaml_path: Optional[str]) -> List[Dict]:
    """列出图谱目录下的**图谱**文件。

    同目录里常混着别的 YAML（如 LiFEMem_queries.yaml 这类评测数据），它们既不是
    图谱、也不该被面板的「删除」按钮碰到，故按「能被解析成图谱」过滤——否则用户
    一次误点就可能删掉评测数据。解析结果按 (mtime, size) 缓存，避免每次开面板
    都重新解析全部文件。
    """
    directory = graph_dir(yaml_path)
    current = os.path.abspath(yaml_path) if yaml_path else ""
    items: List[Dict] = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return items
    for entry in entries:
        if not entry.lower().endswith(GRAPH_SUFFIXES):
            continue
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        nodes = inspect_graph(full, st)
        if nodes is None:
            continue
        items.append({
            "name": entry,
            "size_bytes": st.st_size,
            "mtime": _iso(st.st_mtime),
            "nodes": nodes,
            "current": os.path.abspath(full) == current,
        })
    return items


# abspath -> (mtime_ns, size, nodes|None)；按文件 mtime 门控，键数受目录内文件数约束
_INSPECT_CACHE: Dict[str, tuple] = {}


def inspect_graph(path: str, st=None) -> Optional[int]:
    """返回该 YAML 的节点数；不是合法图谱（解析失败 / 没有 nodes 列表）返回 None。"""
    full = os.path.abspath(path)
    if st is None:
        try:
            st = os.stat(full)
        except OSError:
            return None
    hit = _INSPECT_CACHE.get(full)
    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    nodes = _count_nodes(full)
    _INSPECT_CACHE[full] = (st.st_mtime_ns, st.st_size, nodes)
    return nodes


def _count_nodes(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        return None
    return len(data["nodes"])


# ---------------------------------------------------------------------------
# 切换请求与回执（面板写请求，MCP 应用后写回执）
# ---------------------------------------------------------------------------

def request_switch(yaml_path: Optional[str], name: str, by: str = "") -> Dict:
    """面板侧：登记一次「请 MCP 切换到 name」的请求。"""
    error = validate_name(name)
    if error:
        raise ValueError(error)
    if not os.path.isfile(graph_path(yaml_path, name)):
        raise ValueError(f"图谱不存在: {name}")
    generation = _as_int(read_pointer(yaml_path).get("generation")) + 1
    return patch_pointer(
        yaml_path,
        active=name,
        generation=generation,
        requested_at=_now(),
        requested_by=by or "",
        status="pending",
        detail="",
    )


def pending_request(yaml_path: Optional[str]) -> Optional[Dict]:
    """MCP 侧：取出尚未应用的请求；没有则返回 None。"""
    data = read_pointer(yaml_path)
    if not data.get("active"):
        return None
    if _as_int(data.get("generation")) > _as_int(data.get("applied_generation")):
        return data
    return None


def report_applied(yaml_path: Optional[str], generation, *, status: str,
                   detail: str = "", graph: Optional[Dict] = None,
                   vector: Optional[Dict] = None) -> Dict:
    """MCP 侧：回执本次应用结果（成功或失败原因）。"""
    return patch_pointer(
        yaml_path,
        applied_generation=_as_int(generation),
        status=status,
        detail=detail or "",
        applied_at=_now(),
        graph=graph or {},
        vector=vector or {},
    )


def state(yaml_path: Optional[str]) -> Dict:
    """给面板展示用的指针状态摘要。"""
    data = read_pointer(yaml_path)
    if not data:
        return {}
    return {
        "panel_active": data.get(PANEL_FIELD) or "",
        "active": data.get("active") or "",
        "generation": _as_int(data.get("generation")),
        "applied_generation": _as_int(data.get("applied_generation")),
        "status": data.get("status") or ("pending" if pending_request(yaml_path) else ""),
        "detail": data.get("detail") or "",
        "requested_at": data.get("requested_at") or "",
        "requested_by": data.get("requested_by") or "",
        "applied_at": data.get("applied_at") or "",
        "graph": data.get("graph") or {},
        "vector": data.get("vector") or {},
    }


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------

def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _now() -> str:
    return _iso(None)


def _iso(ts: Optional[float]) -> str:
    dt = datetime.fromtimestamp(ts) if ts is not None else datetime.now()
    return dt.astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: str, data: Dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
