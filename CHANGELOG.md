# Changelog

本文件格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.5] - 2026-11-07

### Added

- `ARIADNE_BIND_HOST`：面板与 MCP 的宿主端口绑定地址，默认 `127.0.0.1`（仅回环）。
- 容器健康检查：compose 级 MCP / 面板探活与镜像级 `HEALTHCHECK`；`stop_grace_period: 20s`。
- `ALM_MAX_BODY_BYTES`：`/add`、`/search` 的请求体上限，默认 8 MB，超限返回 413。
- `Server/connect.py -L`：SSH 本地端口转发，便于访问只绑回环的面板 / MCP。
- 源层说话人：`source_store` 从对话标注 `[昵称]（QQ 号）原话` 解析并逐轮落盘 `speaker` / `speaker_id`（`extract_speaker_info`），供溯源巡检按人核对；正文照旧脱敏，`speaker_id` 作为结构字段保留。
- 中英双语架构图（`assets/diagrams/architecture{,.en}.{svg,png}`），替换 README 中原有的纯文本架构图。

### Changed

- README 的目标叙事从「知识库与记忆系统的对照」收窄为「拟人化的个人记忆」，明确面向单用户聊天与情感陪伴。
- MCP `dba_add_conversation` 的工具描述补充流水格式约定（每行 `[昵称]（QQ 号）原话`），禁止追加旁白或时间说明。
- npm 包 `ariadne-memory` 的包页说明改为项目简介 + GitHub 链接，不再只是「thin launcher」。
- 面板与 MCP 的宿主端口默认不再暴露公网（明文 HTTP + 密码 / Basic 鉴权），远程访问改走 SSH 隧道或前置 TLS 代理。
- MCP SSE 默认端口改为 `8766`（与面板 `8765` 错开），并在退出时保留 8 秒宽限以 flush 调度器缓冲。
- MCP / 面板容器日志启用轮转（20 MB × 5），避免长期驻留撑满磁盘。

### Fixed

- `deploy/docker-compose.yml` 端口映射两侧统一使用同一变量：自定义端口不再出现「容器 Up 但外部打不通」。
- MCP 工具调用失败返回 `isError: true`，正文只含 `error_id`，堆栈仅进服务端日志。
- 修改密码后此前签发的会话 Cookie 立即失效（`session_epoch`），旧 3 段格式 Cookie 自动作废。
- ALM 增加容器内请求体上限，不再依赖前置 nginx 的 `client_max_body_size`。
- 笔记节点（`channel=note`）不再参与边连接：连边 prompt 增加禁令，`GraphBuilder._create_edge` 补代码层护栏，`_find_duplicate` 跳过笔记，避免被合并进 person / thing。
- 节点不再写入 `speaker` / `speaker_id`：说话人是原文层属性（谁说了这句话），节点级没有唯一说话人。

## [0.1.4] - 2026-10-31

### Added

- MCP 知识层：`dba_add_note` 写入项目笔记、`dba_query_knowledge` 按关键词精筛；笔记作为普通节点（`metadata.channel="note"`）参与向量召回与倒排索引，不引入第二套存储。
- `kwindex.py`：无第三方依赖的关键词倒排索引（ASCII 词元 + 中文二元组，`title/tags/text` 权重 3/2/1）。
- `ARIADNE_MCP_RERANK=1`：知识层可独立开启 rerank，不读取也不回落 `ALM_RERANK`。
- 文档分层：`docs/ALM.md`、`docs/MCP.md`、`docs/architecture.md`、`docs/README.md` 与 `docs/theory/`。
- 快速开始示例：`examples/quickstart_alm.py`、`examples/quickstart_mcp.py`。
- 工程化配置：`.gitattributes`、`.editorconfig`、`.pre-commit-config.yaml`、coverage 配置。
- CI 打包冒烟测试：非 editable 安装后校验全部 console scripts。
- `CHANGELOG.md` 与 `CITATION.cff`。
- npm 薄壳 `ariadne-memory`（`packages/npm/`）与 `packaging/quickstart/` 一键脚本包。
- 本地生图脚本 `imagegen/generate.py`（仅素材生成，两个服务都不读取）。

### Changed

- 根目录瘦身：理论文档移入 `docs/theory/`，图片移入 `assets/`，Docker 文件移入 `deploy/`。
- README 中英双版拆分长章节，补充文档索引与示例入口。
- 安装命令统一使用 `constraints.txt` 约束传递依赖的安全下限。
- 发布流水线改为 tag 驱动：`v*` 同时发布 PyPI 与 npm（trusted publishing，无需长期 token）。

### Fixed

- `alm/space.py` 的 `mode.yaml` 与索引 stamp 改为原子写（临时文件 + `os.replace`），避免半截文件让空间归属判定被兜底覆盖、或触发整空间重新 embedding。

### Security

- 启用 Dependabot、漏洞告警与自动安全修复，并为 `main` 配置分支保护（CI 必过、1 次 review、线性历史）。
- `mcp` 最低版本提升至 `2.2.0`；直接依赖增加上界，避免无界升级。
- 新增 `constraints.txt`，将 `urllib3`、`pillow`、`aiohttp`、`setuptools` 等传递依赖锁定到已修复版本；由 Dependabot 组更新继续抬升安全下限。

### Removed

- 移除 `requirements.txt`（与 `pyproject.toml` 重复且已不一致）。

## [0.1.2] - 2026-10-20

### Added

- ALM 同步 HTTP 记忆服务：原文层（L1）+ 图谱层（L2）两层架构，支持按形态/空间路由、`auto` 模式与图谱回补。
- GitHub Actions CI：Ruff 检查 + pytest（Python 3.10 / 3.11 / 3.12）。
- 社区健康文件：`SECURITY.md`、`CODE_OF_CONDUCT.md`、Issue 模板与 PR 模板。

### Changed

- 版本改为从 `dba_pipeline.__version__` 动态读取，统一为 `0.1.2`。
- README 中英双版修复失效链接，并补充 `alm/`、`tests/`、`docs/` 结构说明。

### Removed

- 移除临时巡检脚本 `alm_inspect_tmp.sh`。

## [0.1.1] - 2026-08-25

### Changed

- 阶段性文档更新与缺陷修复。

## [0.1.0] - 2026-08-14

### Added

- 首个公开版本：LLM DBA 图谱构建、PAR 检索、StoryRank、3D 可视化面板与 MCP Server。

[Unreleased]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/releases/tag/v0.1.0
