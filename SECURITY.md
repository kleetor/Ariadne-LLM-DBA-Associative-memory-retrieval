# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a Vulnerability

请**不要**通过公开 Issue 报告安全问题。

请优先使用 GitHub 的私密漏洞报告入口：

1. 打开仓库的 **Security** 标签页；
2. 选择 **Advisories → Report a vulnerability**；
3. 描述复现步骤、影响范围与可能的修复方向。

如果无法使用 Security Advisories，可通过 GitHub 私信联系维护者 [@kleetor](https://github.com/kleetor)。

我们会在以下时限内响应：

- 3 个工作日内确认收到报告；
- 7 个工作日内给出初步评估与处理计划；
- 修复发布后，在 Release Notes 中致谢报告者（如报告者同意）。

## Scope

以下问题属于本策略范围：

- API Key、会话 Cookie 签名密钥、鉴权绕过等凭据与访问控制问题；
- 通过 MCP / HTTP API / 可视化面板触发的远程代码执行、路径穿越、注入；
- 记忆数据越权读取或跨 `user_id` 空间泄漏；
- 依赖链中可被本项目直接利用的已知漏洞。

以下问题通常不在范围内：

- 需要攻击者已拥有主机 root / Docker 控制权的场景；
- 仅影响本地开发环境、且未在文档中承诺安全边界的问题；
- 社会工程、物理访问与拒绝服务（除非可稳定远程触发）。

## Safe Harbor

在遵守本策略、不访问他人数据、不破坏服务可用性的前提下进行的安全研究，我们不会采取法律行动。
