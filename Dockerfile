# Ariadne MCP Server — Docker 部署镜像
# 启动方式：默认以 SSE 网络模式运行，供远程 LLM Agent 通过 http://<host>:8766/sse 调用。
#
# 构建：
#   docker build -t ariadne-mcp .
#   # 默认使用 API embedding（镜像小、构建快，适合服务器部署）；
#   # 如需本地 embedding（torch + sentence-transformers，镜像更大），
#   # 用 --build-arg EMBEDDING_LOCAL=true 构建：
#   docker build --build-arg EMBEDDING_LOCAL=true -t ariadne-mcp .
#
# 运行：
#   docker run --rm -p 8766:8766 \
#     -e OPENAI_API_KEY=sk-xxx \
#     -e OPENAI_API_BASE=https://api.deepseek.com/v1 \
#     -e OPENAI_MODEL=deepseek-v4-flash \
#     -e EMBEDDING_LOCAL=false \
#     -e EMBEDDING_API_KEY=sk-xxx \
#     -e EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5 \
#     ariadne-mcp

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 避免 torch 与 FAISS 的 OpenMP 运行时冲突导致进程 Aborted（mcp_server 内也会设置）
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# 基础依赖层（先只拷 requirements，便于利用 docker 层缓存）
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 项目代码
COPY pyproject.toml README.md LICENSE ./
COPY dba_pipeline ./dba_pipeline
COPY data ./data

RUN pip install --no-cache-dir .

# 可选：本地 embedding（CPU 版 torch + sentence-transformers）。默认关闭（API 模式），
# 避免服务器虚拟硬件上跑本地模型效果差；如需本地 embedding，构建时传 --build-arg EMBEDDING_LOCAL=true
ARG EMBEDDING_LOCAL=false
RUN if [ "$EMBEDDING_LOCAL" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir "sentence-transformers>=2.2"; \
    fi

# 本地 embedding 模型下载缓存（运行期第一次使用模型时下载；挂载卷可持久化加速）
VOLUME ["/root/.cache/huggingface"]

EXPOSE 8766

# --yaml 路径与端口可由环境变量覆盖，便于运行时从外部挂载不同的图谱 YAML，无需重建镜像：
#   docker run -v /host/data:/data -e ARIADNE_YAML=/data/my_memory.yaml ...
#   docker run -p 9000:8766 -e ARIADNE_PORT=9000 ...
CMD ["sh", "-c", "ariadne-mcp --yaml \"${ARIADNE_YAML:-/app/data/memory_graph.yaml}\" --sse --host 0.0.0.0 --port \"${ARIADNE_PORT:-8766}\""]
