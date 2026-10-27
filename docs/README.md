# 文档索引

本目录按用途组织 Ariadne 的文档。英文文档以 `.en.md` 结尾。

## 快速开始

- [README](../README.md)：项目定位、安装、快速开始与入口
- [ALM 同步 HTTP 服务](ALM.md)：Add / Search / Health 契约、两层架构与部署
- [MCP 服务详解](MCP.md)：11 个工具、传输模式、环境变量与 Agent 接入
- [架构与项目结构](architecture.md)：目录树与模块职责

## 理论与实验

- [0828 检索理论文档](theory/0828PAR检索理论文档.md)：目的驱动联想记忆检索模型
- [0828 前置数据管理与后处理](theory/0828前置数据管理与后处理.md)：LLM DBA 图谱构建
- [0908 e2e 实验文档](theory/0908e2e实验文档.md)：端到端实验记录

## 部署与运维

- [ALM 回补修复记录](ALM_BACKFILL_FIX_20261018.md)
- [ALM 空间注册表锁重构计划](ALM_REGISTRY_LOCK_PLAN_20261020.md)
- 数据卷清理方案：已移至私有目录 `alm/almplan/`（不入库；含生产主机与卷名等运行信息）

## 项目时间线

- [TIMELINE.md](TIMELINE.md)
- [TIMELINE.en.md](TIMELINE.en.md)

## 参与贡献

- [CONTRIBUTING](../CONTRIBUTING.md)
- [SECURITY](../SECURITY.md)
- [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md)
- [CHANGELOG](../CHANGELOG.md)
- [CITATION](../CITATION.cff)

## 示例

- [examples/quickstart_alm.py](../examples/quickstart_alm.py)
- [examples/quickstart_mcp.py](../examples/quickstart_mcp.py)
