# SPDX-License-Identifier: AGPL-3.0-only

"""VectorStore 原地更新（update_memories）单元测试"""
import numpy as np
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
