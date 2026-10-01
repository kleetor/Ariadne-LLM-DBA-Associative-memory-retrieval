# SPDX-License-Identifier: AGPL-3.0-only

"""VectorStore 单元测试：原地更新 + 追加式分片落盘"""
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from dba_pipeline.embedding.store import VectorStore


class FakeEmbeddings(Embeddings):
    """返回 [文本长度, 1.0] 的固定维度向量，便于断言重算。"""

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def _store():
    return VectorStore(embeddings=FakeEmbeddings(), backend="faiss")


def test_update_memories_recomputes_vector():
    vs = _store()
    vs.add_memories(["n1"], ["aaa"])
    assert np.asarray(vs._content_vectors["n1"]).tolist() == [3.0, 1.0]

    vs.update_memories(["n1"], ["aaaaa"])
    assert np.asarray(vs._content_vectors["n1"]).tolist() == [5.0, 1.0]
    assert vs._contents["n1"] == "aaaaa"


def test_update_memories_rebuilds_index_without_duplication():
    vs = _store()
    vs.add_memories(["n1", "n2"], ["aaa", "bbbb"])

    vs.update_memories(["n1"], ["aaaaa"])

    # 重建后 n1 只应出现一条（不再膨胀），且新向量使 n1 排最前
    results = vs.search("aaaaa", k=10)
    mids = [r[0] for r in results]
    assert mids.count("n1") == 1
    assert mids[0] == "n1"


def test_save_load_roundtrip_contents(tmp_path):
    vs = _store()
    vs.add_memories(["n1"], ["你好"])
    path = str(tmp_path / "faiss_index")
    vs.save(path)

    vs2 = _store()
    vs2.load(path)
    assert vs2._contents.get("n1") == "你好"
    assert "n1" in vs2._content_vectors


def test_two_spaces_do_not_share_vector_cache(tmp_path):
    """两个空间各自落盘索引时，向量分片必须互不覆盖。

    ALM 侧每个空间有自己独立的目录 `<user_id>/graph.index/`（见 `MemorySpace._index_paths`），
    所以两个空间必须各写各的。若退化成固定路径，后写的空间会覆盖先写的，load 回来的
    就是**别人空间**的向量——检索结果张冠李戴且完全静默。
    """
    a_dir = tmp_path / "space_a" / "graph.index" / "faiss"
    b_dir = tmp_path / "space_b" / "graph.index" / "faiss"

    va = _store()
    va.add_memories(["a1"], ["aaa"])
    va.save(str(a_dir))

    vb = _store()
    vb.add_memories(["b1", "b2"], ["bbbb", "bbbbb"])
    vb.save(str(b_dir))

    assert list((a_dir / "vectors").glob("*.npz"))
    assert list((b_dir / "vectors").glob("*.npz"))

    va2 = _store()
    va2.load(str(a_dir))
    assert set(va2._content_vectors) == {"a1"}
    assert set(va2._contents) == {"a1"}


# ---- 追加式分片落盘 ----


def test_save_writes_only_new_vectors_incrementally(tmp_path):
    """增量落盘：第二次 save 只追加新分片，不重写已有分片"""
    path = str(tmp_path / "idx" / "faiss")
    shard_dir = tmp_path / "idx" / "faiss" / "vectors"

    vs = _store()
    vs.add_memories(["n1", "n2"], ["aaa", "bbbb"])
    vs.save(path)
    first = sorted(p.name for p in shard_dir.glob("*.npz"))
    assert len(first) == 1

    vs.add_memories(["n3"], ["ccccc"])
    vs.save(path)
    second = sorted(p.name for p in shard_dir.glob("*.npz"))
    assert len(second) == 2
    assert second[:1] == first  # 旧分片原样保留（未被重写）

    vs2 = _store()
    vs2.load(path)
    assert set(vs2._content_vectors) == {"n1", "n2", "n3"}


def test_save_without_changes_writes_nothing(tmp_path):
    path = str(tmp_path / "idx" / "faiss")
    shard_dir = tmp_path / "idx" / "faiss" / "vectors"

    vs = _store()
    vs.add_memories(["n1"], ["aaa"])
    vs.save(path)
    before = sorted(p.name for p in shard_dir.glob("*.npz"))

    vs.save(path)  # 无变更
    assert sorted(p.name for p in shard_dir.glob("*.npz")) == before


def test_removed_vector_does_not_come_back_after_load(tmp_path):
    """删除写墓碑：重新载入后不能复活"""
    path = str(tmp_path / "idx" / "faiss")
    vs = _store()
    vs.add_memories(["n1", "n2"], ["aaa", "bbbb"])
    vs.save(path)

    vs.remove_memories(["n2"])
    vs.save(path)

    vs2 = _store()
    vs2.load(path)
    assert set(vs2._content_vectors) == {"n1"}
    assert set(vs2._contents) == {"n1"}


def test_updated_vector_wins_after_load(tmp_path):
    """更新写新行，回放后后写覆盖先写"""
    path = str(tmp_path / "idx" / "faiss")
    vs = _store()
    vs.add_memories(["n1"], ["aaa"])
    vs.save(path)
    vs.update_memories(["n1"], ["aaaaa"])
    vs.save(path)

    vs2 = _store()
    vs2.load(path)
    assert np.asarray(vs2._content_vectors["n1"]).tolist() == [5.0, 1.0]
    assert vs2._contents["n1"] == "aaaaa"


def test_shards_are_float32_on_disk(tmp_path):
    """落盘向量必须是 float32（原实现是 float64，白占一倍磁盘）"""
    path = str(tmp_path / "idx" / "faiss")
    vs = _store()
    vs.add_memories(["n1"], ["aaa"])
    vs.save(path)

    shard = sorted((tmp_path / "idx" / "faiss" / "vectors").glob("*.npz"))[0]
    # np.load 对 .npz 是惰性的：必须在句柄仍打开时取值
    data = np.load(str(shard), allow_pickle=False)
    assert data["vectors"].dtype == np.float32
    data.close()


def test_clear_vectors_invalidates_disk_shards(tmp_path):
    """clear_vectors 之后旧分片必须作废，否则 load 会把已清空的向量复活"""
    path = str(tmp_path / "idx" / "faiss")
    vs = _store()
    vs.add_memories(["n1"], ["aaa"])
    vs.save(path)

    vs.clear_vectors()
    vs.save(path)

    vs2 = _store()
    with pytest.raises(FileNotFoundError):
        vs2.load(path)


def test_compact_merges_fragmented_shards(tmp_path):
    """分片数与碎片行都超限时合并，且合并前后权威状态一致"""
    path = str(tmp_path / "idx" / "faiss")
    shard_dir = tmp_path / "idx" / "faiss" / "vectors"

    vs = _store()
    prev = None
    for i in range(20):
        mid = f"n{i}"
        vs.add_memories([mid], ["x" * (i + 1)])
        if prev is not None:
            vs.remove_memories([prev])
        prev = mid
        vs.save(path)

    # 不合并的话会有 20 个分片；合并后应远小于此
    assert len(list(shard_dir.glob("*.npz"))) <= 5

    vs2 = _store()
    vs2.load(path)
    assert set(vs2._content_vectors) == {prev}
    assert vs2._contents[prev] == "x" * 20


def test_compaction_triggers_on_file_count_without_tombstones(tmp_path):
    """追加型负载（无墓碑）到硬上限也必须合并，否则文件数无上限 → 每 Add 全目录扫描

    这一档此前会**抛 NameError**：墓碑占比那条判据只定义在 (16, 64] 区间内，
    超过硬上限时 `total` 未赋值就被日志行引用。所以本用例同时钉两件事：
    合并发生了，且没有异常。
    """
    from dba_pipeline.embedding import store as store_mod

    path = str(tmp_path / "idx" / "faiss")
    shard_dir = tmp_path / "idx" / "faiss" / "vectors"

    vs = _store()
    for i in range(store_mod._VEC_SHARD_HARD_MAX + 5):
        vs.add_memories([f"n{i}"], ["x" * (i + 1)])
        vs.save(path)  # 每次只追加一行，全程没有墓碑

    files = list(shard_dir.glob("*.npz"))
    assert len(files) <= store_mod._VEC_SHARD_HARD_MAX, f"分片数无上限：{len(files)}"

    vs2 = _store()
    vs2.load(path)
    assert set(vs2._content_vectors) == {f"n{i}" for i in range(store_mod._VEC_SHARD_HARD_MAX + 5)}


def test_load_legacy_npz_format(tmp_path):
    """旧格式 `content_vectors.npz`（位于索引目录的上级）仍可载入"""
    idx = tmp_path / "space.index"
    idx.mkdir()
    np.savez(
        str(idx / "content_vectors.npz"),
        ids=np.array(["n1"]),
        vectors=np.array([[3.0, 1.0]]),
        contents=np.array(["aaa"]),
    )

    vs = _store()
    vs.load(str(idx / "faiss"))
    assert np.asarray(vs._content_vectors["n1"]).tolist() == [3.0, 1.0]
    assert vs._contents["n1"] == "aaa"
