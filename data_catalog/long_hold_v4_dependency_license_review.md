# Long Hold V4 依赖与许可证复核

复核日期：2026-07-23。版本以各 `requirements-long-hold-v4*.txt` 的固定版本为准。本表记录包元数据和公开许可证边界，不替代法律意见，也不授予任何数据使用权。

| 依赖组 | 包 | 许可证/风险摘要 |
|---|---|---|
| 核心运行 | NumPy、pandas | BSD 系宽松许可证；NumPy wheel 同时列出若干兼容的第三方组件许可证。 |
| 公共采集 | AKShare、Requests、Beautiful Soup、lxml、charset-normalizer、BaoStock | 代码许可证以 MIT、Apache-2.0、BSD 系为主。接口返回的数据仍受来源网站条款、版权和频率限制约束，代码开源不等于数据可再分发。 |
| 授权数据 | jqdatasdk | SDK 元数据为 Apache-2.0；实际数据访问需要用户自行取得聚宽授权，凭据和下载结果不得进入公开仓库。 |
| PDF 证据 | pdfplumber、PyMuPDF | pdfplumber 项目采用 MIT；PyMuPDF 为 AGPL-3.0 或商业授权双许可，是本仓库最明确的许可证风险点。它已隔离为非默认 PDF 依赖，分发集成产品前需要重新确认适用许可。 |
| 开发测试 | pytest | MIT；只用于测试。开发依赖文件会显式汇总其他可选组，以便完整测试导入采集模块。 |

默认安装文件只含 NumPy 与 pandas。公共采集、授权源、PDF 和开发测试分别使用独立文件，避免在只运行本地合成 replay 时安装网络采集、付费 SDK 或 AGPL 组件。

仓库不携带第三方行情、指数、公告 PDF、授权数据或缓存。研究者仍需逐项遵守数据来源条款，并自行判断抓取、存储、展示和再分发权限。
