# SPDX-License-Identifier: AGPL-3.0-only

"""DBA 3D 可视化 — 离线渲染器。

复用 WebUI 的静态前端（``viz/static``），把 CSS/JS 内联进单个自包含 HTML，
并把图谱数据预注入 ``__ARIADNE_BOOT__``。产物无需服务器即可打开，
前端会自动进入只读离线模式（可观测/数据操作降级为提示）。
"""

import argparse
import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dba_pipeline.loader import load_graph
from dba_pipeline.viz.exporter import to_3dforcegraph

_STATIC_DIR = Path(__file__).parent / "static"
_JS_SRC_RE = re.compile(r'<script src="(/static/js/[^"]+)"></script>')


def render_from_yaml(yaml_path: str, output: str = None) -> str:
    """从 YAML 图谱生成自包含可视化 HTML

    Args:
        yaml_path: 输入的 memory_graph.yaml 路径
        output: 输出 HTML 路径（默认与 yaml 同目录同名 .html）

    Returns:
        输出文件路径
    """
    graph = load_graph(yaml_path)
    data = to_3dforcegraph(graph)
    html = build_offline_html(data)

    if output is None:
        output = str(Path(yaml_path).with_suffix(".html"))
    Path(output).write_text(html, encoding="utf-8")
    print(f"生成: {output}")
    print(f"  节点: {graph.node_count}, 边: {graph.edge_count}")
    return output


def build_offline_html(data: dict) -> str:
    """把静态前端与图数据打包为单文件 HTML。"""
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    css = (_STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="/static/css/app.css">',
        "<style>\n" + css + "\n</style>",
    )

    # 站点图标与顶栏 logo 引用的是同一张 PNG，统一内联为 data URI，
    # 保证单文件离线打开时图标不丢
    icon = _STATIC_DIR / "icon.png"
    if icon.exists():
        encoded = base64.b64encode(icon.read_bytes()).decode("ascii")
        html = html.replace("/static/icon.png", "data:image/png;base64," + encoded)

    def _inline(match: "re.Match") -> str:
        rel = match.group(1).replace("/static/", "")
        content = (_STATIC_DIR / rel).read_text(encoding="utf-8")
        return "<script>\n" + content + "\n</script>"

    html = _JS_SRC_RE.sub(_inline, html)

    boot = {
        "mode": "offline",
        "data": data,
        "generatedAt": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    payload = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("/*__ARIADNE_BOOT__*/null", payload)


def main():
    parser = argparse.ArgumentParser(description="DBA 3D 可视化面板生成器")
    parser.add_argument("--yaml", required=True, help="输入的 memory_graph.yaml 路径")
    parser.add_argument("--output", "-o", default=None, help="输出 HTML 路径")
    args = parser.parse_args()

    render_from_yaml(args.yaml, args.output)


if __name__ == "__main__":
    main()
