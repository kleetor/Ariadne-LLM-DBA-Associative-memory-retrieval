# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 留存清理（`alm/cleanup.py`）的归组与删除。

为什么单独钉这两条：清理是**合规**动作（AML 要求 30 天内删除评测数据），
失效模式是"删了图谱、留了原文"——服务看起来一切正常，但从日志/盘上都看不出来。
新布局下"一个空间 = 一个目录"，故要守住的只有两件事：目录被识别为空间、
非目录条目一律不碰。
"""

import os
import time

from alm.cleanup import collect_spaces, purge


def _make_space(root, name="u_abc"):
    """按生产布局造一个空间目录：目录内是该空间的全部产物"""
    space = root / name
    space.mkdir()
    (space / "graph.yaml").write_text("nodes: []", encoding="utf-8")
    (space / "docs.yaml").write_text("docs: []", encoding="utf-8")
    (space / "docs.log").write_text("", encoding="utf-8")
    (space / "graph.index").mkdir()
    (space / "docs.index").mkdir()
    return space


def test_each_directory_is_one_space(tmp_path):
    """一个顶层目录 = 一个空间；目录内的产物不再需要按后缀去认领"""
    _make_space(tmp_path, "u_abc")
    _make_space(tmp_path, "u_def")

    spaces, ignored = collect_spaces(tmp_path)

    assert set(spaces) == {"u_abc", "u_def"}
    assert spaces["u_abc"] == [tmp_path / "u_abc"]
    assert ignored == []


def test_loose_files_are_reported_not_deleted(tmp_path):
    """顶层散落的文件一律只报告（可能是挂进来的别的东西），不进任何空间"""
    _make_space(tmp_path, "u_abc")
    (tmp_path / "README.txt").write_text("x", encoding="utf-8")

    spaces, ignored = collect_spaces(tmp_path)

    assert set(spaces) == {"u_abc"}
    assert [p.name for p in ignored] == ["README.txt"]


def test_legacy_flat_layout_is_still_collected(tmp_path):
    """旧布局（五份产物平铺）的历史遗留必须仍被识别，否则会被永久跳过、违反留存条款"""
    for name in ("u_old.yaml", "u_old.docs.yaml", "u_old.docs.log"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "u_old.index").mkdir()
    (tmp_path / "u_old.docs.index").mkdir()
    _make_space(tmp_path, "u_new")

    spaces, ignored = collect_spaces(tmp_path)

    assert set(spaces) == {"u_old", "u_new"}
    assert len(spaces["u_old"]) == 5, "旧布局产物没有被完整认领"
    assert ignored == []


def test_legacy_and_new_layout_share_a_stem(tmp_path):
    """同名（同一 user_id）的旧遗留与新目录归到同一个空间，一轮删除"""
    legacy = tmp_path / "u_abc.yaml"
    legacy.write_text("x", encoding="utf-8")
    space = _make_space(tmp_path, "u_abc")

    spaces, _ = collect_spaces(tmp_path)

    assert set(spaces) == {"u_abc"}
    assert set(spaces["u_abc"]) == {legacy, space}


def test_purge_deletes_legacy_leftovers(tmp_path):
    """旧布局遗留（装的是记忆正文）必须随留存期一起删掉"""
    legacy = tmp_path / "u_old.yaml"
    legacy.write_text("x", encoding="utf-8")
    old = time.time() - 40 * 86400
    os.utime(legacy, (old, old))

    purge(tmp_path, older_than_days=30, apply=True)

    assert not legacy.exists(), "旧布局遗留没有被清理（留存条款要求 30 天内删除）"


def test_purge_deletes_the_whole_space_directory(tmp_path):
    """超过留存期时整目录删除：图谱、文档区、日志一个都不能留"""
    space = _make_space(tmp_path, "u_abc")
    old = time.time() - 40 * 86400
    # 目录自身的 mtime 也算"最后写入"（建/删文件会刷新它），故最后单独压一次
    for entry in space.rglob("*"):
        os.utime(entry, (old, old))
    os.utime(space, (old, old))

    purge(tmp_path, older_than_days=30, apply=True)

    assert not space.exists(), "空间目录没有整目录删除（留存条款要求 30 天内删除）"
