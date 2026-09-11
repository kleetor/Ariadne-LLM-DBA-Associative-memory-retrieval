# SPDX-License-Identifier: AGPL-3.0-only

"""ALM Add / Search 契约：请求校验与响应组装。

严格对齐官方规范（https://agentmemoryleaderboard.ai/api-guide）：

- Add：`request_id / messages / user_id / session_id`，响应必须原样回显三个 ID
- Search：`query / options? / user_id / top_k`，响应为顶层 `data` 数组，
  元素含必填 `id`、`content`，可选 `score`、`created_at`
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ALLOWED_ROLES = ("user", "assistant")


class ContractError(Exception):
    """请求不符合 ALM 契约（对应 HTTP 400 / 422）"""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class AddMessage:
    role: str
    content: str
    timestamp: Optional[int] = None


@dataclass
class AddRequest:
    request_id: str
    user_id: str
    session_id: str
    messages: List[AddMessage]


@dataclass
class SearchRequest:
    query: str
    user_id: str
    top_k: int
    options: Optional[List[str]] = None


# ---- 校验 ----

def _require_non_empty_str(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"字段 {key} 必须为非空字符串")
    return value


def _parse_messages(raw: Any) -> List[AddMessage]:
    if not isinstance(raw, list) or not raw:
        raise ContractError("字段 messages 必须为非空数组")

    messages: List[AddMessage] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ContractError(f"messages[{idx}] 必须为对象")

        role = item.get("role")
        if role not in ALLOWED_ROLES:
            raise ContractError(f"messages[{idx}].role 必须为 {' / '.join(ALLOWED_ROLES)}")

        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ContractError(f"messages[{idx}].content 必须为非空字符串")

        timestamp = item.get("timestamp")
        # bool 是 int 子类，需显式排除
        if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, int)):
            raise ContractError(f"messages[{idx}].timestamp 必须为 Unix 毫秒整数")

        messages.append(AddMessage(role=role, content=content, timestamp=timestamp))

    return messages


def parse_add_request(body: Any) -> AddRequest:
    """解析并校验 Add 请求体"""
    if not isinstance(body, dict):
        raise ContractError("请求体必须为 JSON 对象", status=400)

    return AddRequest(
        request_id=_require_non_empty_str(body, "request_id"),
        user_id=_require_non_empty_str(body, "user_id"),
        session_id=_require_non_empty_str(body, "session_id"),
        messages=_parse_messages(body.get("messages")),
    )


def parse_search_request(body: Any, max_top_k: int = 100) -> SearchRequest:
    """解析并校验 Search 请求体"""
    if not isinstance(body, dict):
        raise ContractError("请求体必须为 JSON 对象", status=400)

    query = _require_non_empty_str(body, "query")
    user_id = _require_non_empty_str(body, "user_id")

    top_k = body.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ContractError("字段 top_k 必须为正整数")
    # 返回条数不得超过请求值，同时封顶防止被要求超大返回
    top_k = min(top_k, max_top_k)

    options = body.get("options")
    if options is not None:
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            raise ContractError("字段 options 必须为字符串数组")

    return SearchRequest(query=query, user_id=user_id, top_k=top_k, options=options)


# ---- 响应组装 ----

def build_add_response(request_id: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Add 成功响应：success 必须为布尔 true，三个 ID 原样回显"""
    return {
        "success": True,
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
    }


def build_search_response(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Search 响应：顶层 data 数组（不额外包 items 层）"""
    return {"data": items}
