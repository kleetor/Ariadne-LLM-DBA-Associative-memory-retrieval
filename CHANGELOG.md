# Changelog

本文件格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- MCP 知识层：`dba_add_note` 写入项目笔记、`dba_query_knowledge` 按关键词精筛；笔记作为普通节点（`metadata.channel="note"`）参与向量召回与倒排索引，不引入第二套存储。
- `kwindex.py`：无第三方依赖的关键词倒排索引（ASCII 词元 + 中文二元组，`title/tags/text` 权重 3/2/1）。
- `ARIADNE_MCP_RERANK=1`：知识层可独立开启 rerank，不读取也不回落 `ALM_RERANK`。
- 文档分层：`docs/ALM.md`、`docs/MCP.md`、`docs/architecture.md`、`docs/README.md` 与 `docs/theory/`。
- 快速开始示例：`examples/quickstart_alm.py`、`examples/quickstart_mcp.py`。
- 工程化配置：`.gitattributes`、`.editorconfig`、`.pre-commit-config.yaml`、coverage 配置。
- CI 打包冒烟测试：非 editable 安装后校验全部 console scripts。
- `CHANGELOG.md` 与 `CITATION.cff`。

### Changed

- 根目录瘦身：理论文档移入 `docs/theory/`，图片移入 `assets/`，Docker 文件移入 `deploy/`。
- README 中英双版拆分长章节，补充文档索引与示例入口。
- 安装命令统一使用 `constraints.txt` 约束传递依赖的安全下限。

### Security

- 启用 Dependabot、漏洞告警与自动安全修复，并为 `main` 配置分支保护（CI 必过、1 次 review、线性历史）。
- `mcp` 最低版本提升至 `2.2.0`；直接依赖增加上界，避免无界升级。
- 新增 `constraints.txt`，将 `urllib3`、`pillow`、`aiohttp`、`setuptools` 等传递依赖锁定到已修复版本。

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

[Unreleased]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kleetor/Ariadne-LLM-DBA-Associative-memory-retrieval/releases/tag/v0.1.0
