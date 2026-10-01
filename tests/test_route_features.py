# SPDX-License-Identifier: AGPL-3.0-only

"""路由特征管道的契约测试。

这里的断言分两类：
  ① **行为**：特征抽得对不对（判型、时间型、长度、选项剥离）；
  ② **契约**：`FEATURE_NAMES` 的顺序与 `vectorize` 的列序——它是落盘契约，
     改了会让已训练的探针与旧特征文件静默失配，故必须有测试钉住。
"""

import numpy as np
import pytest

from alm.route_features import (
    FEATURE_NAMES,
    extract,
    numeric_row,
    vectorize,
)

# 平台真实的两种形态（取自 Full 日志）
PLAIN_Q = "When was Jon in Rome?"
MCQ_INLINE = (
    "Which of the following characters explicitly moved to or toward the water cooler?\n\n"
    "A. Foreman\nB. Juror #7\nC. Juror #3\n\n"
    "Answer with the option letter only, enclosed in parentheses, e.g., (X)."
)
MCQ_CONTRACT = "Which option best describes the verdict?"


def test_plain_question_is_not_precision_shape():
    f = extract(PLAIN_Q)
    assert f["has_options"] == 0.0
    assert f["has_answer_instruction"] == 0.0
    assert f["is_precision_shape"] == 0.0
    assert f["stripped_ratio"] == 0.0  # 没剥离，检索文本与原文等长


def test_contract_options_mark_precision_shape():
    f = extract(MCQ_CONTRACT, ["A. yes", "B. no"])
    assert f["has_options"] == 1.0
    assert f["n_options_raw"] == 2.0
    assert f["is_precision_shape"] == 1.0


def test_inline_options_with_instruction_are_stripped():
    """内联选项 + 作答指令 → 判型命中且检索文本被剥离"""
    f = extract(MCQ_INLINE)
    assert f["has_answer_instruction"] == 1.0
    assert f["is_precision_shape"] == 1.0
    assert f["stripped_ratio"] > 0.5
    assert f["query_chars"] > f["retrieval_chars"]


def test_plain_enumeration_is_not_stripped():
    """有 A./B./C. 标号但**没有**作答指令 → 不得误判（宁可漏判，不可误判）"""
    f = extract("Compare version A. alpha, B. beta, and C. gamma for me.")
    assert f["has_answer_instruction"] == 0.0
    assert f["stripped_ratio"] == 0.0


@pytest.mark.parametrize("q", [
    "When was Jon in Rome?",
    "How long did it take for Jon to open his studio?",
    "how many months did Gina work there",
])
def test_temporal_questions_detected(q):
    assert extract(q)["is_temporal"] == 1.0


@pytest.mark.parametrize("q", [
    "What is Jon's favorite style of dance?",
    "Why did Jon shut down his bank account?",
])
def test_non_temporal_not_flagged(q):
    assert extract(q)["is_temporal"] == 0.0


def test_wh_head_is_captured():
    assert extract("When did he leave?")["is_when"] == 1.0
    assert extract("Why did he leave?")["is_why"] == 1.0
    assert extract("Do Jon and Gina start businesses?")["is_yesno"] == 1.0


def test_year_and_date_words_are_detected():
    f = extract("What happened on January 20, 2023?")
    assert f["has_year"] == 1.0
    assert f["has_date_word"] == 1.0


def test_feature_names_are_stable_and_ordered():
    """落盘契约：名字集合与顺序都不能悄悄改"""
    expected_head = ["has_options", "n_options", "n_options_raw", "has_answer_instruction",
                     "stripped_ratio", "query_chars", "retrieval_chars"]
    assert list(FEATURE_NAMES[:len(expected_head)]) == expected_head
    assert "is_precision_shape" in FEATURE_NAMES
    assert "is_temporal" in FEATURE_NAMES
    assert "embedding" not in FEATURE_NAMES  # 向量不是标量特征列


def test_numeric_row_follows_feature_names_order():
    f = extract("When was Jon in Rome?")
    row = numeric_row(f)
    assert len(row) == len(FEATURE_NAMES)
    assert row[list(FEATURE_NAMES).index("is_temporal")] == 1.0
    assert all(isinstance(x, float) for x in row)


def test_vectorize_puts_embedding_first_and_appends_scalars():
    """列序契约：embedding 在前、标量在尾——加特征只往尾部追加，不移动 embedding 段"""
    emb = [0.1, 0.2, 0.3]
    vec = vectorize(extract("When was Jon in Rome?"), embedding=emb)
    assert vec.shape == (len(emb) + len(FEATURE_NAMES),)
    assert vec.dtype == np.float32
    assert np.allclose(vec[:len(emb)], np.asarray(emb, dtype=np.float32))


def test_vectorize_without_embedding_is_scalars_only():
    vec = vectorize(extract("When was Jon in Rome?"))
    assert vec.shape == (len(FEATURE_NAMES),)


def test_extract_is_deterministic():
    a = extract(MCQ_INLINE)
    b = extract(MCQ_INLINE)
    assert numeric_row(a) == numeric_row(b)
