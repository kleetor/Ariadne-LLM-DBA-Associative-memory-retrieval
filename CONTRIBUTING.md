# Contributing to Ariadne

感谢你对 Ariadne 的关注！本文档说明如何参与贡献。

## 许可证

本项目采用 [GNU AGPLv3](LICENSE)（`AGPL-3.0-only`）许可证，版权归 **kleetor** 所有。

- 所有贡献者提交的代码默认同意以 `AGPL-3.0-only` 发布。
- 每个源码文件顶部带有 `# SPDX-License-Identifier: AGPL-3.0-only` 标识，请勿移除。

## 提交规范（DCO）

本项目采用 **Developer Certificate of Origin（DCO，开发者原创证书）** 来确认贡献者身份与版权归属。

每个提交（commit）的 commit message 末尾必须包含签名行：

```
Signed-off-by: 你的名字 <你的邮箱>
```

例如：

```
git commit -sm "feat: add temporal_lookup to MCP tools"

Signed-off-by: kleetor <you@example.com>
```

- 使用 `git commit -s` 会自动追加该行。
- 未包含 `Signed-off-by` 的提交将不被合并。

## 开发流程

1. **Fork** 本仓库并克隆到你本地。
2. 创建**特性分支**：`git checkout -b feat/your-feature`。
3. 修改代码，保持改动聚焦、可读性优先。
4. 提交时遵守 DCO 签名（见上）。
5. 推送并提交 **Pull Request**。

## 编码约定

- 目标 Python 版本：`3.10+`。
- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 风格。
- 注释与 commit message 使用**中文**（与本仓库保持一致）。
- 新功能请附带对应测试（`tests/` 目录），并保持原有测试通过。
- 尽量避免不必要的新依赖；确需添加时同步更新 `pyproject.toml` / `requirements.txt`。

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

> 部分测试（如涉及 LLM / embedding 的端到端用例）需要配置 `.env` 与 API Key。
