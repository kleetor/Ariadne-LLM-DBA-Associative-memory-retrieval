# SPDX-License-Identifier: AGPL-3.0-only

"""
DBA 3D 可视化 — 渲染器

从 YAML checkpoint 或 live MemoryGraph 生成自包含的 3D 可视化 HTML。
"""

import argparse
import json
import os
from pathlib import Path
from dba_pipeline.loader import load_graph
from dba_pipeline.viz.exporter import to_3dforcegraph_json

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_PATH = _TEMPLATE_DIR / "dashboard_3d.html"


def render_from_yaml(yaml_path: str, output: str = None) -> str:
    """从 YAML 图谱生成可视化 HTML

    Args:
        yaml_path: 输入的 memory_graph.yaml 路径
        output: 输出 HTML 路径（默认与 yaml 同目录同名 .html）

    Returns:
        输出文件路径
    """
    graph = load_graph(yaml_path)
    data_json = to_3dforcegraph_json(graph)

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DATA_PLACEHOLDER__", data_json)

    if output is None:
        output = str(Path(yaml_path).with_suffix(".html"))
    Path(output).write_text(html, encoding="utf-8")
    print(f"生成: {output}")
    print(f"  节点: {graph.node_count}, 边: {graph.edge_count}")
    return output


def main():
    parser = argparse.ArgumentParser(description="DBA 3D 可视化面板生成器")
    parser.add_argument("--yaml", required=True, help="输入的 memory_graph.yaml 路径")
    parser.add_argument("--output", "-o", default=None, help="输出 HTML 路径")
    args = parser.parse_args()

    render_from_yaml(args.yaml, args.output)


if __name__ == "__main__":
    main()
