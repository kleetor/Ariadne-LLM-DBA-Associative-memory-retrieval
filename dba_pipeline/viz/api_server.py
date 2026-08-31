"""
DBA CRUD API Server

提供 MemoryGraph 的 CRUD REST API，支持自动持久化到 YAML checkpoint。
所有修改自动写回 YAML 文件，确保重启不丢失数据。

Usage:
    python -m src.viz.api_server --yaml your_memory_graph.yaml --port 8765
"""

import base64
import json
import os
import argparse
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

import yaml

# Allow running as module
# Package installed via pip

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph
from dba_pipeline.core.jump_axis import NodeType, RelationType, get_jump_weight
from dba_pipeline import oplog


NODE_TYPES = {t.value.upper(): t for t in NodeType}
REL_TYPES = {t.value: t for t in RelationType}


class MemoryGraphAPI:
    """封装 MemoryGraph 的 CRUD 操作，支持自动持久化"""

    def __init__(self, graph: MemoryGraph, yaml_path: str = None, actor_user: str = None):
        self.graph = graph
        self.yaml_path = yaml_path
        self._next_node_id = self._compute_next_id()
        # 记录初始文件 mtime，用于检测 YAML 被外部（如 MCP）修改后自动重载
        self._last_mtime = self._yaml_mtime()
        # 操作日志：DBA 人工操作的主体（Basic 鉴权用户名）与日志路径
        self.actor_user = actor_user or "local"
        self.op_log_path = oplog.default_oplog_path(self.yaml_path)

    def _log(self, op, request=None, result=None):
        """记录一次 DBA 人工操作到操作日志（与 MCP 的 LLM 操作同文件，可追溯）"""
        oplog.log_operation(
            "dba", op, request=request, result=result,
            actor={"type": "dba", "user": self.actor_user},
            path=self.op_log_path,
        )

    def read_oplog(self, tail: int = 200):
        """读取最近操作日志（JSONL 转 list），供追溯接口使用"""
        rows = oplog.read_oplog(tail=tail, path=self.op_log_path)
        # 按时间正序返回（旧→新），前端滚动展示
        return rows

    def _yaml_mtime(self) -> int:
        """返回 YAML 文件的 mtime（纳秒）；文件不存在时返回 0"""
        if not self.yaml_path:
            return 0
        try:
            return os.stat(self.yaml_path).st_mtime_ns
        except OSError:
            return 0

    def _reload_if_changed(self):
        """YAML 被外部修改后自动重新加载，无需重启容器"""
        if not self.yaml_path:
            return
        mtime = self._yaml_mtime()
        if mtime == self._last_mtime:
            return
        try:
            self.graph = load_graph(self.yaml_path)
            self._next_node_id = self._compute_next_id()
            self._last_mtime = mtime  # 仅成功后更新；失败则保留旧 mtime，下次继续重试
        except Exception as e:
            sys.stderr.write(f"[API] 自动重新加载失败: {e}\n")

    def _save(self):
        """自动持久化：将当前图写回 YAML 文件"""
        if not self.yaml_path:
            return
        try:
            data = self.graph.to_dict()
            with open(self.yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            sys.stderr.write(f"[API] 自动保存失败: {e}\n")

    def _compute_next_id(self) -> int:
        """计算下一个可用节点 ID"""
        max_id = 0
        for nid in self.graph.graph.nodes():
            try:
                num = int(nid[1:])
                if num > max_id:
                    max_id = num
            except (ValueError, IndexError):
                pass
        return max_id + 1

    def get_graph_data(self) -> dict:
        """导出完整图数据（同 exporter 格式）"""
        self._reload_if_changed()
        from dba_pipeline.viz.exporter import to_3dforcegraph
        return to_3dforcegraph(self.graph)

    def create_node(self, node_type: str, content: str) -> dict:
        """创建新节点"""
        self._reload_if_changed()
        if not node_type or not content:
            raise ValueError("node_type 和 content 不能为空")
        nt = NODE_TYPES.get(node_type.upper())
        if not nt:
            raise ValueError(f"无效的节点类型: {node_type}")

        nid = f"n{self._next_node_id}"
        self._next_node_id += 1

        self.graph.graph.add_node(
            nid,
            content=content,
            node_type=nt,
            metadata={},
            deprecated=False,
            forgotten=False,
        )
        self._save()
        result = {
            "id": nid,
            "node_type": node_type.upper(),
            "content": content,
            "deprecated": False,
            "forgotten": False,
            "in_degree": 0,
            "out_degree": 0,
        }
        self._log("create_node", request={"node_type": node_type.upper(), "content": content}, result=result)
        return result

    def update_node(self, nid: str, data: dict) -> dict:
        """更新节点属性"""
        self._reload_if_changed()
        if nid not in self.graph.graph.nodes:
            raise ValueError(f"节点不存在: {nid}")

        node = self.graph.graph.nodes[nid]
        if "content" in data:
            node["content"] = data["content"]
        if "node_type" in data:
            nt = NODE_TYPES.get(data["node_type"].upper())
            if not nt:
                raise ValueError(f"无效的节点类型: {data['node_type']}")
            node["node_type"] = nt
        if "deprecated" in data:
            node["deprecated"] = bool(data["deprecated"])
        self._save()
        result = {
            "id": nid,
            "node_type": node["node_type"].value.upper() if hasattr(node["node_type"], "value") else str(node["node_type"]),
            "content": node["content"],
            "deprecated": node.get("deprecated", False),
        }
        self._log("update_node", request={"node_id": nid, **{k: v for k, v in data.items()}}, result=result)
        return result

    def delete_node(self, nid: str) -> dict:
        """删除节点及关联边"""
        self._reload_if_changed()
        if nid not in self.graph.graph.nodes:
            raise ValueError(f"节点不存在: {nid}")

        node = self.graph.graph.nodes[nid]
        # 记录关联边信息用于前端恢复
        in_edges = [(u, v, self.graph.graph.edges[u, v]) for u, v in self.graph.graph.in_edges(nid)]
        out_edges = [(u, v, self.graph.graph.edges[u, v]) for u, v in self.graph.graph.out_edges(nid)]

        self.graph.graph.remove_node(nid)
        self._save()
        result = {
            "deleted": nid,
            "removed_edges": len(in_edges) + len(out_edges),
        }
        self._log("delete_node", request={"node_id": nid}, result=result)
        return result

    def create_edge(self, source: str, target: str, rel_type: str) -> dict:
        """创建边"""
        self._reload_if_changed()
        if not source or not target:
            raise ValueError("source 和 target 不能为空")
        if source == target:
            raise ValueError("不能创建自环边")
        if source not in self.graph.graph.nodes:
            raise ValueError(f"源节点不存在: {source}")
        if target not in self.graph.graph.nodes:
            raise ValueError(f"目标节点不存在: {target}")

        rt = REL_TYPES.get(rel_type.lower())
        if not rt:
            raise ValueError(f"无效的边类型: {rel_type}")

        # 跳转轴权重校验：权重为 0 的方向不允许建边（与 MCP/GraphBuilder 对齐）
        src_type = self.graph.get_node_type(source)
        if src_type and get_jump_weight(src_type, rt, is_reverse=False) <= 0:
            raise ValueError(f"边 {source}--[{rt.value}]-->{target} 在该节点类型上权重为 0，不允许创建")

        if self.graph.graph.has_edge(source, target):
            raise ValueError(f"边已存在: {source} -> {target}")

        self.graph.graph.add_edge(source, target, rel_type=rt)
        # 双向类型自动补反向边
        if rt in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
            if not self.graph.graph.has_edge(target, source):
                self.graph.graph.add_edge(target, source, rel_type=rt)
        self._save()
        result = {
            "source": source,
            "target": target,
            "rel_type": rel_type.lower(),
        }
        self._log("create_edge", request={"source": source, "target": target, "rel_type": rel_type.lower()}, result=result)
        return result

    def delete_edge(self, source: str, target: str) -> dict:
        """删除边"""
        self._reload_if_changed()
        if not self.graph.graph.has_edge(source, target):
            raise ValueError(f"边不存在: {source} -> {target}")

        edge_data = self.graph.graph.edges[source, target]
        self.graph.graph.remove_edge(source, target)
        self._save()
        result = {
            "deleted": f"{source} -> {target}",
            "rel_type": edge_data["rel_type"].value if hasattr(edge_data["rel_type"], "value") else str(edge_data["rel_type"]),
        }
        self._log("delete_edge", request={"source": source, "target": target}, result=result)
        return result

    def export_yaml(self) -> str:
        """导出真实 YAML checkpoint"""
        self._reload_if_changed()
        data = self.graph.to_dict()
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


class APIHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    api: Optional[MemoryGraphAPI] = None  # 由工厂函数设置
    auth_user: Optional[str] = None       # Basic 鉴权用户名（None 表示不开启鉴权）
    auth_pass: Optional[str] = None       # Basic 鉴权密码

    def log_message(self, format, *args):
        """精简日志"""
        sys.stderr.write(f"[API] {args[0]}\n")

    def _send_auth_required(self):
        """返回 401，触发浏览器 Basic 认证弹窗"""
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="ariadne"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _check_auth(self) -> bool:
        """校验 Basic Auth；未配置鉴权时放行。返回 True 表示允许继续处理。"""
        if not (self.auth_user and self.auth_pass):
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8")
                user, _, pwd = decoded.partition(":")
                if user == self.auth_user and pwd == self.auth_pass:
                    return True
            except Exception:
                return False
        self._send_auth_required()
        return False

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        """提供 3D 可视化面板 HTML（数据由前端动态拉取 /api/graph）"""
        template_path = Path(__file__).parent / "templates" / "dashboard_3d.html"
        try:
            html = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._send_json({"error": "dashboard template not found"}, 500)
            return
        html = html.replace("__DATA_PLACEHOLDER__", '{"nodes": [], "links": []}')
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _parse_path(self):
        """解析 URL 路径，提取节点/边 ID"""
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        return parts, parsed.query

    def do_OPTIONS(self):
        self._send_json({}, 204)

    def do_GET(self):
        if not self._check_auth():
            return
        parts, query = self._parse_path()
        try:
            if parts == [] or parts == ["index.html"]:
                self._send_html()
            elif parts == ["api", "graph"]:
                self._send_json(self.api.get_graph_data())
            elif parts == ["api", "oplog"]:
                self._send_json({"ops": self.api.read_oplog()})
            elif parts == ["api", "export", "yaml"]:
                yaml_data = self.api.export_yaml()
                body = yaml_data.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-yaml; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Disposition", "attachment; filename=memory_graph.yaml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"error": "Not Found"}, 404)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        if not self._check_auth():
            return
        parts, query = self._parse_path()
        try:
            body = self._read_body()
            if parts == ["api", "nodes"]:
                result = self.api.create_node(
                    node_type=body.get("node_type", ""),
                    content=body.get("content", ""),
                )
                self._send_json(result, 201)
            elif parts == ["api", "edges"]:
                result = self.api.create_edge(
                    source=body.get("source", ""),
                    target=body.get("target", ""),
                    rel_type=body.get("rel_type", ""),
                )
                self._send_json(result, 201)
            else:
                self._send_json({"error": "Not Found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_PUT(self):
        if not self._check_auth():
            return
        parts, query = self._parse_path()
        try:
            body = self._read_body()
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "nodes":
                nid = parts[2]
                result = self.api.update_node(nid, body)
                self._send_json(result)
            else:
                self._send_json({"error": "Not Found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_DELETE(self):
        if not self._check_auth():
            return
        parts, query = self._parse_path()
        try:
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "nodes":
                nid = parts[2]
                result = self.api.delete_node(nid)
                self._send_json(result)
            elif len(parts) >= 4 and parts[0] == "api" and parts[1] == "edges":
                source = parts[2]
                target = parts[3]
                result = self.api.delete_edge(source, target)
                self._send_json(result)
            else:
                self._send_json({"error": "Not Found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def make_handler(api: MemoryGraphAPI, auth_user: str = None, auth_pass: str = None):
    """工厂函数：创建绑定了 api 实例的 handler"""
    class BoundHandler(APIHandler):
        pass
    BoundHandler.api = api
    BoundHandler.auth_user = auth_user
    BoundHandler.auth_pass = auth_pass
    return BoundHandler


def main():
    parser = argparse.ArgumentParser(description="DBA Mock API Server")
    parser.add_argument("--yaml", required=True, help="Path to YAML checkpoint")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, 容器内请用 0.0.0.0)")
    args = parser.parse_args()

    # Basic 鉴权（默认开启：仅当两者都在 .env 中配置时才开启）
    auth_user = os.environ.get("ARIADNE_VIZ_USER")
    auth_pass = os.environ.get("ARIADNE_VIZ_PASS")
    if bool(auth_user) != bool(auth_pass):
        print("警告: ARIADNE_VIZ_USER 与 ARIADNE_VIZ_PASS 需同时设置，鉴权不会生效", file=sys.stderr)

    print(f"加载图数据: {args.yaml}")
    graph = load_graph(args.yaml)
    # actor_user 作为 DBA 人工操作日志的主体（Basic 鉴权用户名）
    api = MemoryGraphAPI(graph, yaml_path=args.yaml, actor_user=auth_user or "local")

    print(f"节点: {graph.node_count}, 边: {graph.edge_count}")

    handler = make_handler(api, auth_user=auth_user, auth_pass=auth_pass)
    server = HTTPServer((args.host, args.port), handler)
    auth_state = "Basic 鉴权: 开启" if (auth_user and auth_pass) else "Basic 鉴权: 未开启（仅限本地）"
    print(f"API 服务器启动: http://{args.host}:{args.port}/api/graph  ({auth_state})")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
