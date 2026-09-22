# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 契约层测试：请求校验、响应结构与得分归一化（不依赖 LLM / 网络）"""

import pytest

from alm.contract import (
    ContractError,
    build_add_response,
    build_search_response,
    parse_add_request,
    parse_search_request,
)
from alm.engine import space_filename
from alm.space import MemorySpace


def _valid_add_body() -> dict:
    return {
        "request_id": "eval:run_abc:locomo_refined:conv-0:chunk-0",
        "user_id": "eval:run_abc:locomo:conv-0",
        "session_id": "eval:run_abc:sample:0",
        "messages": [
            {"role": "user", "content": "memory text", "timestamp": 1704067200000},
            {"role": "assistant", "content": "reply text"},
        ],
    }


class TestAddContract:
    def test_valid_request(self):
        req = parse_add_request(_valid_add_body())
        assert req.request_id == "eval:run_abc:locomo_refined:conv-0:chunk-0"
        assert req.user_id == "eval:run_abc:locomo:conv-0"
        assert req.session_id == "eval:run_abc:sample:0"
        assert len(req.messages) == 2
        assert req.messages[0].timestamp == 1704067200000
        assert req.messages[1].timestamp is None

    def test_body_must_be_object(self):
        with pytest.raises(ContractError) as exc:
            parse_add_request([1, 2, 3])
        assert exc.value.status == 400

    @pytest.mark.parametrize("field", ["request_id", "user_id", "session_id"])
    def test_missing_id_fields(self, field):
        body = _valid_add_body()
        body.pop(field)
        with pytest.raises(ContractError) as exc:
            parse_add_request(body)
        assert exc.value.status == 422
        assert field in exc.value.message

    def test_messages_must_be_non_empty_list(self):
        for value in (None, [], "text"):
            body = _valid_add_body()
            body["messages"] = value
            with pytest.raises(ContractError):
                parse_add_request(body)

    def test_role_must_be_allowed(self):
        body = _valid_add_body()
        body["messages"][0]["role"] = "system"
        with pytest.raises(ContractError) as exc:
            parse_add_request(body)
        assert "role" in exc.value.message

    def test_content_must_be_non_empty(self):
        body = _valid_add_body()
        body["messages"][1]["content"] = "   "
        with pytest.raises(ContractError):
            parse_add_request(body)

    def test_timestamp_must_be_int_millis(self):
        body = _valid_add_body()
        body["messages"][0]["timestamp"] = "1704067200000"
        with pytest.raises(ContractError):
            parse_add_request(body)

    def test_bool_timestamp_rejected(self):
        body = _valid_add_body()
        body["messages"][0]["timestamp"] = True
        with pytest.raises(ContractError):
            parse_add_request(body)

    def test_response_echoes_ids(self):
        resp = build_add_response("req-1", "user-1", "session-1")
        assert resp == {
            "success": True,
            "request_id": "req-1",
            "user_id": "user-1",
            "session_id": "session-1",
        }
        assert resp["success"] is True  # 必须是布尔 true


class TestSearchContract:
    def test_valid_request(self):
        req = parse_search_request(
            {"query": "Which answer best matches?", "user_id": "u1", "top_k": 100}
        )
        assert req.query == "Which answer best matches?"
        assert req.user_id == "u1"
        assert req.top_k == 100
        assert req.options is None

    def test_options_accepted(self):
        req = parse_search_request(
            {"query": "q", "user_id": "u1", "top_k": 10, "options": ["A. a", "B. b"]}
        )
        assert req.options == ["A. a", "B. b"]

    def test_options_must_be_string_list(self):
        with pytest.raises(ContractError):
            parse_search_request(
                {"query": "q", "user_id": "u1", "top_k": 10, "options": [1, 2]}
            )

    @pytest.mark.parametrize("top_k", [None, 0, -1, "100", 1.5, True])
    def test_top_k_must_be_positive_int(self, top_k):
        with pytest.raises(ContractError):
            parse_search_request({"query": "q", "user_id": "u1", "top_k": top_k})

    def test_top_k_clamped_to_max(self):
        req = parse_search_request({"query": "q", "user_id": "u1", "top_k": 500}, max_top_k=100)
        assert req.top_k == 100

    def test_query_and_user_id_required(self):
        with pytest.raises(ContractError):
            parse_search_request({"query": "", "user_id": "u1", "top_k": 10})
        with pytest.raises(ContractError):
            parse_search_request({"query": "q", "user_id": "", "top_k": 10})

    def test_response_is_top_level_data_array(self):
        resp = build_search_response([{"id": "n1", "content": "text", "score": 0.9}])
        assert set(resp.keys()) == {"data"}
        assert isinstance(resp["data"], list)
        assert resp["data"][0]["id"] == "n1"


class TestScoreNormalization:
    def test_normalized_to_unit_range(self):
        items = [
            {"id": "a", "content": "x", "score": 4.0},
            {"id": "b", "content": "y", "score": 2.0},
        ]
        result = MemorySpace._normalize_scores(items)
        assert result[0]["score"] == 1.0
        assert result[1]["score"] == 0.5
        # 顺序保持（PAR 已按 combined_score 降序返回）
        assert [item["id"] for item in result] == ["a", "b"]

    def test_fallback_scores_are_monotonic(self):
        items = [
            {"id": "a", "content": "x", "score": 0.0},
            {"id": "b", "content": "y", "score": 0.0},
        ]
        result = MemorySpace._normalize_scores(items)
        assert result[0]["score"] > result[1]["score"]

    def test_empty_input(self):
        assert MemorySpace._normalize_scores([]) == []


class TestSpaceFilename:
    """user_id → YAML 文件名映射：可读、可逆、无碰撞，且不含路径分隔符"""

    SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

    def test_readable_mapping(self):
        assert space_filename("eval:run_abc:locomo:conv-0") == (
            "eval%3Arun_abc%3Alocomo%3Aconv-0.yaml"
        )

    def test_only_safe_characters(self):
        name = space_filename('a/b\\c:d*e?f"g<h>i|j k')
        assert name.endswith(".yaml")
        stem = name[: -len(".yaml")]
        assert all(ch in self.SAFE or ch == "%" for ch in stem)
        assert "/" not in name and "\\" not in name

    def test_no_path_traversal(self):
        name = space_filename("../../etc/passwd")
        assert "/" not in name and "\\" not in name
        assert ".." not in name

    def test_dot_only_id_is_escaped(self):
        assert space_filename("..") == "%2E%2E.yaml"
        assert space_filename(".") == "%2E.yaml"

    def test_trailing_dot_is_escaped(self):
        # Windows 不允许文件名以点结尾
        assert space_filename("abc.") == "abc%2E.yaml"

    def test_non_ascii_is_escaped(self):
        name = space_filename("用户1")
        stem = name[: -len(".yaml")]
        assert all(ch in self.SAFE or ch == "%" for ch in stem)
        assert name.endswith("1.yaml")

    def test_no_collision_between_distinct_ids(self):
        ids = [
            "eval:run:locomo:conv-0",
            "eval:run:locomo:conv-1",
            "a:b",
            "a_b",
            "eval%3Arun",
            "eval:run",
        ]
        names = [space_filename(i) for i in ids]
        assert len(set(names)) == len(ids)
