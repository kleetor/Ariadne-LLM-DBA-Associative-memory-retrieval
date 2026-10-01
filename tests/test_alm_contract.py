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
from alm.space import (
    MemorySpace, _clip_for_log, _has_answer_instruction, _is_precision_query,
    _is_qa_feedback, _qa_feedback_summary, _retrieval_text, _strip_option_block,
)
from alm.tokens import TokenMeter, extract_finish_reason


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


class TestQaFeedbackGuard:
    """平台在每次 search 之后会回写一条 QA 反馈（评测元数据）。

    它含「题目 + 我们上一轮的答案 + 得分」，必须拦在入图之前（否则它以近乎 1.0 的
    余弦抢走同题种子位，并把上一次的错误答案写进记忆）；但其 `qa_id` / `method` /
    `score` 是平台唯一主动给出的「题目归属」信道，故要留痕。
    """

    REAL = ('Streaming online QA feedback:\n'
            '{"method": "locomo_llm_judge", "predicted_answer": "Yesterday.", '
            '"qa_id": "locomo1:q0000:step01", '
            '"question": "When Jon has lost his job as a banker?", '
            '"schema_version": "streaming-online-feedback-v1", "score": 0.0}')

    def test_detects_platform_feedback(self):
        assert _is_qa_feedback(self.REAL)

    def test_schema_marker_alone_is_enough(self):
        assert _is_qa_feedback('{"schema_version": "streaming-online-feedback-v1"}')

    def test_ordinary_memory_is_not_matched(self):
        # 真实记忆消息不应被误伤：实测 1491 条线上记忆消息零命中
        assert not _is_qa_feedback("Gina: Hey Jon! Good to see you. What's up?")
        assert not _is_qa_feedback("用户想给产品提 feedback，希望加个入口。")
        assert not _is_qa_feedback("")

    def test_summary_extracts_mapping_fields(self):
        summary = _qa_feedback_summary(self.REAL)
        assert "qa_id=locomo1:q0000:step01" in summary
        assert "method=locomo_llm_judge" in summary
        assert "score=0.0" in summary
        assert "question=When Jon has lost his job as a banker?" in summary

    def test_summary_survives_missing_fields(self):
        assert _qa_feedback_summary("Streaming online QA feedback:") == "（字段未解析出）"


class TestStripOptionBlock:
    """平台把选择题的「选项 + 作答指令」内联进 query，检索侧只该消费题干。

    实测 `scriptMem-enemy:q0000` 的 query 全长 1147 字 = 题干 168 + 选项块 876 +
    作答指令 103。整坨进向量会把该题答案节点（`n385` "foremost man"）从第 1 名挤到
    第 20 名，并把目的推断退化成 "select the best-supported option" 这类元目的；
    选项与作答指令本是留给下游 answer 模型的，故此处在进检索前剥离。
    """

    STEM = ("Taken together, Billing’s repeated oath-heavy outbursts most strongly suggest "
            "what underlying shift in his stance toward the Stockmanns over the course of "
            "the exchange?")
    OPTS = ("A. He remains consistently supportive, using profanity mainly to signal "
            "camaraderie rather than disapproval. B. He moves from vague suspicion to firm "
            "moral condemnation, ending in a decision to distance himself socially from them. "
            "C. He shifts from being shocked by gossip to deciding he must privately reconcile "
            "with the Stockmanns to verify the truth. D. He is primarily angry at himself for "
            "being naive, and his final refusal is directed at his own habits rather than at "
            "the Stockmanns. E. He moves from incredulous disbelief to harsh moral condemnation, "
            "but his final outburst reflects only momentary frustration—he is still willing to "
            "keep up friendly social contact with the Stockmanns. F. He moves from initial "
            "skepticism to impressed approval of what he hears, and his last oath signals a "
            "renewed resolve to join the Stockmanns socially rather than distance himself.")
    TAIL = ("Please provide the option corresponding to the only correct answer, enclosed in "
            "parentheses, e.g., (X).")

    def test_strips_inlined_options_and_instruction(self):
        assert _strip_option_block(f"{self.STEM} {self.OPTS} {self.TAIL}") == self.STEM

    def test_strips_newline_variant(self):
        q = f"{self.STEM}\n\nA. one\nB. two\nC. three\n\n{self.TAIL}"
        assert _strip_option_block(q) == self.STEM

    def test_plain_question_untouched(self):
        assert _strip_option_block("我在杭州做什么工作？") == "我在杭州做什么工作？"
        assert _strip_option_block("") == ""

    def test_two_letter_reference_is_not_an_option_block(self):
        # 题干自身提到 A. / B. 不足以判定：需要连续三个同序标号
        q = "文档里 A. 项与 B. 项有什么区别？"
        assert _strip_option_block(q) == q

    def test_cuts_at_last_valid_block_start(self):
        # 题干先出现一个孤立 A.，真正的选项块在其后 → 从最后一个成立处截断
        keep = f"{self.STEM} 参考 A. 的说法。"
        assert _strip_option_block(f"{keep} {self.OPTS}") == keep


class TestPrecisionRoute:
    """精确证据路由的判型：这条 query 是不是在要一个唯一答案。

    判型只看**整条 query 的形态**（内联选项块 / 作答指令），零 LLM 调用；被判定为精确时，
    检索层才把跳数收拢（见 config.precision_route）。刻意只收这几类表述：宁可漏判，
    不可误判——误判会把"丰富联想"类的问题也压成一条。
    """

    REAL = ("Taken together, Billing’s repeated oath-heavy outbursts most strongly suggest "
            "what underlying shift in his stance toward the Stockmanns over the course of "
            "the exchange? A. one B. two C. three D. four E. five F. six "
            "Please provide the option corresponding to the only correct answer, "
            "enclosed in parentheses, e.g., (X).")

    def test_inlined_options_are_precision(self):
        assert _is_precision_query(self.REAL)

    def test_instruction_alone_is_enough(self):
        assert _is_precision_query("Which one is right? Only the correct answer, please.")
        assert _is_precision_query("请只回答正确选项。")

    def test_stripped_stem_is_not_precision(self):
        # 剥掉选项块之后剩下的题干本身不再是"精确题形态"（免得二次判型又收拢一次）
        assert not _is_precision_query(_strip_option_block(self.REAL))

    def test_ordinary_questions_are_not_precision(self):
        for q in ("我在杭州做什么工作？",
                  "This article lists several options; summarise them.",
                  "这篇文章列了哪些选项？",
                  ""):
            assert not _is_precision_query(q), q

    def test_listed_markers_without_instruction_are_not_precision(self):
        # 有连续 A./B./C. 标号但**没有作答指令** → 不是精确题。判据必须落在"指令"上：
        # 只看标号会误伤这类普通列举句（`_strip_option_block` 会把它整体截断）。
        listed = "Compare version A. alpha, B. beta, and C. gamma for me."
        assert not _is_precision_query(listed)
        assert _retrieval_text(listed) == listed

    def test_retrieval_text_strips_only_when_instructed(self):
        got = _retrieval_text(self.REAL)
        assert got == _strip_option_block(self.REAL)
        # 题干留下、选项与作答指令都不在检索文本里
        assert got.startswith("Taken together")
        assert "A. one" not in got and "enclosed in parentheses" not in got

    def test_retrieval_text_never_empty(self):
        # 整条 query 就是选项块（剥离后为空）→ 必须退回原文，否则空串进 embedding 会 400
        only_opts = "A. one B. two C. three 请只回答正确选项。"
        assert _strip_option_block(only_opts) == ""
        assert _retrieval_text(only_opts) == only_opts


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


class TestTokenMeterTruncation:
    """撞输出上限必须可观测。

    0927 线上有一次**写入侧**抽取 `in=4533 out=32768`——撞满上限、输出畸形，而
    `finish_reason` 当时只在 story_rank 里记录，所以这条完全隐形，只能靠"out 恰好
    等于上限"反推。把 finish_reason 挂到计量器上，写入/检索两侧一并覆盖。
    """

    @staticmethod
    def _result(finish=None, via="generation_info"):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, Generation, LLMResult

        info = {"finish_reason": finish} if finish else {}
        if via == "generation_info":
            gen = Generation(text="x", generation_info=info)
        else:
            gen = ChatGeneration(message=AIMessage(content="x", response_metadata=info))
        return LLMResult(
            generations=[[gen]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20,
                                        "total_tokens": 30}},
        )

    def test_finish_reason_from_generation_info(self):
        assert extract_finish_reason(self._result("length")) == "length"

    def test_finish_reason_from_response_metadata(self):
        assert extract_finish_reason(self._result("length", via="metadata")) == "length"

    def test_finish_reason_absent_is_none(self):
        assert extract_finish_reason(self._result()) is None

    def test_meter_counts_truncated_calls_only(self):
        meter = TokenMeter(price_input=0.15, price_output=0.60, verbose=False)
        meter.on_llm_end(self._result("length"))
        meter.on_llm_end(self._result("length"))
        meter.on_llm_end(self._result("stop"))
        meter.on_llm_end(self._result())            # 取不到 finish_reason 的不算截断
        summary = meter.summary()
        assert summary["calls"] == 4
        assert summary["truncated_calls"] == 2
        assert "被截断" in meter.report()


class TestTraceClip:
    """留痕截断：单条日志记录必须有界（0930 实测单文件 84 MB 的成因之一）"""

    def test_short_text_untouched(self):
        assert _clip_for_log("hello", 100) == "hello"

    def test_long_text_clipped_with_marker(self):
        text = "x" * 500
        out = _clip_for_log(text, 100)
        assert out.startswith("x" * 100)
        assert "留痕截断" in out
        assert "500" in out            # 标注原长，便于区分"本来就短"与"被截了"

    def test_zero_limit_disables_clipping(self):
        text = "x" * 500
        assert _clip_for_log(text, 0) == text

    def test_none_and_empty_are_safe(self):
        assert _clip_for_log(None, 10) == ""
        assert _clip_for_log("", 10) == ""

    def test_exact_limit_not_clipped(self):
        text = "x" * 100
        assert _clip_for_log(text, 100) == text
