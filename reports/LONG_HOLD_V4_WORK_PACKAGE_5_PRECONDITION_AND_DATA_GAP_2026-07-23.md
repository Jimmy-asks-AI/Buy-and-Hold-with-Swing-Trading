# 工作包 5：前置门槛与数据缺口审计

审计时间：2026-07-23 12:03:20 +08:00

审计结论：`PRECONDITIONS_PASSED_DATA_ACQUISITION_BLOCKED`

研究状态：`RESEARCH_ONLY`

晋级状态：`promotion_allowed=false`

## 前置门槛

| 门槛 | 证据 | 结论 |
|---|---|---|
| 工作包 1、2、3、4 全部合并到 main | main 依次包含 PR #1、#2、#3、#4 的合并提交；当前 HEAD 为 `965bb6bf5f8e31d7e4cf7b120d616ea2fee777d5` | PASS |
| main 完整 CI 通过 | GitHub Actions run `29977982519` 在同一 HEAD 上完成 pytest、unittest、compileall 和 synthetic replay，结论为 success | PASS |
| 合成 replay 干净复现且哈希一致 | 从远端 main 独立克隆，新建 Python 3.14.3 虚拟环境，连续运行两次；预期与两次实际 `bundle_sha256` 均为 `5120887f9565b0ace2e846aec1101c3b5a95ba3ce34d39e93d7ddabe2a25ab1e` | PASS |
| 订单、账户、清单和回测正确性测试通过 | 本地针对核心、属性、执行、运行清单、公开 replay、PIT Gate 和 ETF 的测试为 `237 passed, 53 subtests passed` | PASS |
| 从最新 main 创建独立分支 | `agent/work-package-5-pit-walk-forward` 从上述 HEAD 创建 | PASS |

前置门槛通过只允许开始工作包 5 的工程工作，不表示 PIT 数据已经合格，也不表示任何收益结论成立。

## 十类正式数据的当前结论

逐行机器清单位于 `data_catalog/long_hold_v4_work_package_5_dataset_gap_inventory.csv`。本轮没有把候选供应商、当前最终快照或观察链改写成已授权 PIT 数据。

| 数据集 | 当前证据 | PIT 结论 |
|---|---|---|
| 历史行业分类 | 当前工作簿可检查区间和代码迁移，但含事后整理 | BLOCKED |
| 逐版本财务数据 | 最终快照观察可检查字段，缺三张主表逐次修订 | BLOCKED |
| 股票历史估值 | 活跃候选观察存在，全生命周期、退市股与历史版本未闭合 | BLOCKED |
| ETF 基准及变更 | 部分公告和短期观察存在，不能用当前基准回填 | BLOCKED |
| ETF 全收益价格 | 当前最终价格/净值和事件候选存在，历史可得性与部分退市链未闭合 | BLOCKED |
| ETF 历史规模和流动性 | 未取得正式 PIT 数据 | BLOCKED |
| ETF 历史费率 | 未取得正式 PIT 数据 | BLOCKED |
| ETF 历史跟踪误差 | 未取得正式 PIT 数据 | BLOCKED |
| 指数全收益 | 未取得正式授权历史 | BLOCKED |
| 指数历史估值 | 未取得正式授权历史与方法版本 | BLOCKED |

## 授权与费用边界

授权接口队列位于 `data_catalog/long_hold_v4_work_package_5_authorization_interfaces.csv`，逐项记录候选提供方、接口或文件、所需权限、费用状态和替代方案。

费用没有可靠报价的项目统一登记为 `NOT_VERIFIED` 或 `PROVIDER_QUOTE_REQUIRED`，未填写猜测数字。公开公告可访问也不自动取得结构化保存、衍生、展示或再分发权；正式接入前仍需逐项核对来源条款和采购合同。授权数据、账号、Token、原始缓存和第三方文件不得进入公开仓库。

## 当前允许与禁止

允许：

- 完成版本化 PIT Gate 和 walk-forward 的代码、配置、合成夹具和失败路径测试；
- 让门禁对缺失、过期、越界、许可不明、当前快照回填、当前成分回填和修订覆盖旧版本逐项失败；
- 输出 `BLOCKED`、`RESEARCH_ONLY` 或 `NO_ACTION`。

禁止：

- 用当前行业、当前成分、当前估值或当前 ETF 信息补齐历史；
- 将公告后的修订财务数据提前使用；
- 为了生成绩效降低门禁；
- 在十类正式数据闭合前运行或报告正式 walk-forward 绩效；
- 自动晋级或给出实盘建议。

因此，后续工作包 5 只会用合成数据验证流程正确性。正式数据运行保持阻断，`promotion_allowed` 始终为 `false`。
