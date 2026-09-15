# SPDX-License-Identifier: AGPL-3.0-only

"""MCP Server 单元测试：干预 CRUD、ID 冲突回归、checkpoint、dotenv"""
import os
import threading

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline import envfile
from dba_pipeline.mcp_server import DBAServer, MAX_CONVERSATION_LENGTH


class FakeVectorStore:
    store = None

    def add_memories(self, memory_ids, contents):
        pass

    def update_memories(self, memory_ids, contents):
        pass

    def remove_memories(self, memory_ids):
        pass


def _server(tmp_path, vector_store=None):
    return DBAServer(
        MemoryGraph(),
        yaml_path=str(tmp_path / "g.yaml"),
        vector_store=vector_store if vector_store is not None else FakeVectorStore(),
        lock=threading.RLock(),
    )


def test_create_node_id_avoids_existing_nodes(tmp_path):
    """C2 回归：DBA 已创建 n5 后，intervene 创建节点必须跳过 n5"""
    server = _server(tmp_path)
    server.graph.graph.add_node(
        "n5", content="dba 创建", node_type=NodeType.STATUS,
        deprecated=False, forgotten=False,
    )
    result = server.intervene("create_node", {"node_type": "STATUS", "content": "人工创建"})
    assert result.get("success") is True
    assert result["node"]["id"] == "n6"
    assert server.graph.graph.nodes["n5"]["content"] == "dba 创建"


def test_intervene_create_edge_bidirectional(tmp_path):
    server = _server(tmp_path)
    server.intervene("create_node", {"node_type": "STATUS", "content": "a"})
    server.intervene("create_node", {"node_type": "STATUS", "content": "b"})
    result = server.intervene("create_edge", {
        "source": "n1", "target": "n2", "rel_type": "social",
    })
    assert result.get("success") is True
    assert server.graph.graph.has_edge("n2", "n1")


def test_intervene_rejects_zero_weight_edge(tmp_path):
    server = _server(tmp_path)
    server.intervene("create_node", {"node_type": "STATUS", "content": "a"})
    server.intervene("create_node", {"node_type": "THING", "content": "b"})
    result = server.intervene("create_edge", {
        "source": "n1", "target": "n2", "rel_type": "taxonomic",
    })
    assert "error" in result


def test_intervene_update_node_rollback_on_vector_error(tmp_path):
    class BrokenStore(FakeVectorStore):
        def update_memories(self, memory_ids, contents):
            raise RuntimeError("embedding api down")

    server = _server(tmp_path, vector_store=BrokenStore())
    server.intervene("create_node", {"node_type": "STATUS", "content": "旧内容"})
    result = server.intervene("update_node", {"node_id": "n1", "content": "新内容"})
    assert "error" in result
    assert server.graph.graph.nodes["n1"]["content"] == "旧内容"


def test_checkpoint_fallback_saves_yaml(tmp_path):
    server = _server(tmp_path)
    server.intervene("create_node", {"node_type": "STATUS", "content": "hello"})
    save_dir = str(tmp_path / "snap")
    result = server.checkpoint(save_dir)
    assert result.get("nodes") == 1
    assert (tmp_path / "snap" / "memory_graph.yaml").exists()


def test_add_conversation_rejects_oversized_input(tmp_path):
    server = _server(tmp_path)
    result = server.add_conversation("x" * (MAX_CONVERSATION_LENGTH + 1))
    assert "error" in result


def test_load_dotenv_prefers_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO_BAR=from_file\n", encoding="utf-8")
    monkeypatch.delenv("FOO_BAR", raising=False)
    monkeypatch.setattr(envfile, "find_dotenv", lambda: env_file)
    envfile.load_dotenv()
    assert os.environ.get("FOO_BAR") == "from_file"

    # 已存在的环境变量优先，不被文件里的值覆盖
    monkeypatch.setenv("FOO_BAR", "from_env")
    envfile.load_dotenv()
    assert os.environ.get("FOO_BAR") == "from_env"
