# SPDX-License-Identifier: AGPL-3.0-only

"""
Ariadne 一键启动脚本

同时启动 3D 可视化面板（ariadne-api）与 MCP 服务器（SSE 模式），
端口默认错开：面板 8765，MCP 8766。

模型配置（LLM / Embedding）由 mcp_server 从 .env 或环境变量自动读取，
无需在命令行重复传入。

用法：
    python start_all.py --yaml data/sample_graph.yaml
    python start_all.py --yaml data/sample_graph.yaml --api-port 9000 --mcp-port 9001
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 脚本所在目录即 dba_pipeline 包根（确保子进程可 `-m dba_pipeline.xxx` 导入）
PKG_ROOT = Path(__file__).resolve().parent


def _child_env() -> dict:
    """构造子进程环境，确保 dba_pipeline 包可被 `-m` 导入"""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _stop(procs) -> None:
    """终止所有子进程：先 terminate，超时后 kill"""
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Ariadne 一键启动（面板 + MCP SSE）")
    parser.add_argument("--yaml", required=True, help="YAML checkpoint 文件路径")
    parser.add_argument("--api-port", type=int, default=8765, help="可视化面板端口（默认 8765）")
    parser.add_argument("--mcp-port", type=int, default=8766, help="MCP SSE 端口（默认 8766）")
    parser.add_argument("--host", default="127.0.0.1", help="MCP 绑定地址（面板固定 127.0.0.1）")
    args = parser.parse_args()

    yaml_path = Path(args.yaml).resolve()
    if not yaml_path.is_file():
        print(f"[ariadne] 错误: YAML 文件不存在: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    commands = [
        [sys.executable, "-m", "dba_pipeline.viz.api_server",
         "--yaml", str(yaml_path), "--port", str(args.api_port)],
        [sys.executable, "-m", "dba_pipeline.mcp_server",
         "--yaml", str(yaml_path), "--sse", "--host", args.host, "--port", str(args.mcp_port)],
    ]

    print(f"[ariadne] 可视化面板: http://127.0.0.1:{args.api_port}")
    print(f"[ariadne] MCP SSE:    http://{args.host}:{args.mcp_port}/sse")
    print("[ariadne] 按 Ctrl+C 停止所有服务", flush=True)

    procs = []
    try:
        for cmd in commands:
            procs.append(subprocess.Popen(cmd, env=_child_env()))

        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    print(f"[ariadne] 某服务提前退出 (code={code})，正在停止其它服务...", flush=True)
                    _stop(procs)
                    sys.exit(code or 1)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[ariadne] 收到 Ctrl+C，正在停止...", flush=True)
        _stop(procs)


if __name__ == "__main__":
    main()
