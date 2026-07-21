# Long Hold V4 ETF 全生命周期总收益观察链

日期：2026-07-18  
市场截止日：2026-07-17  
运行编号：`20260717_20260718T193335462592+0800_4fbf151bf6a9`  
资格状态：`COLLECTION_IN_PROGRESS_CURRENT_FINAL_SNAPSHOT`  
历史回测：`historical_backtest_allowed=false`

## 1. 结论

ETF 原始行情采集已经覆盖全状态主表中的 1,701 只 ETF，包括 1,578 只在市和 123 只退市标的。全量缓存均可读取；其中 1,697 只形成观察价格，4 只因退市尾部冲突继续隔离。

这解决了“只从当前 ETF 名单回看历史”的直接幸存者偏差，但没有把数据升级为正式 PIT 历史。供应商提供的是 2026-07-18 所见的终态历史快照，分红和份额动作缺少完整的首次披露版本链，因此正式文件 `data_raw/long_hold_v4/pit_history/etf_total_return_prices.csv` 没有生成。

## 2. 覆盖与产物

| 项目 | 结果 |
|---|---:|
| 主表 ETF | 1,701 |
| 原始缓存成功 | 1,701 |
| 观察可用 | 1,697 |
| 隔离 | 4 |
| 总收益价格 | 1,463,741 行 |
| 分红/零标记事件 | 1,019 行，271 只 ETF |
| 覆盖区间 | 2005-02-23 至 2026-07-17 |
| 已登记份额动作 | 23 次，21 只 ETF |
| 启发式份额动作 | 129 次，120 只 ETF |
| 全部应用动作 | 152 次 |

主要产物：

- `data_raw/long_hold_v4/pit_history/observations/etf_total_return_prices_lifecycle_observation_latest.csv.gz`
- `data_raw/long_hold_v4/pit_history/observations/etf_dividend_events_lifecycle_observation_latest.csv.gz`
- `data_raw/long_hold_v4/pit_history/observations/etf_total_return_lifecycle_observation_latest_status.csv`
- `data_raw/long_hold_v4/manifests/etf_total_return_lifecycle_observation_latest.json`
- `data_catalog/long_hold_v4_etf_share_action_evidence_queue.csv`

## 3. 公司行动治理

份额拆分、合并和折算优先使用 `configs/long_hold_v4_etf_corporate_actions.json` 中的交易所、基金公司、巨潮或法定披露媒体证据。登记动作无论原始价格跳变是否超过异常阈值，都会按生效日强制应用并校验观察价格比例。

供应商的零增量分红标记仍可用于识别常见比例动作，但这类动作只记为 `zero_marker_common_factor_inference`。逐只状态文件新增：

- `registered_corporate_actions`
- `applied_corporate_actions`
- `governed_corporate_actions`
- `inferred_corporate_actions`
- `corporate_action_evidence_status`
- `corporate_action_evidence_detail_json`

当前 120 只 ETF 至少含一次启发式动作，合计 129 次；公司行动证据门因此为失败。补证队列逐事件保存生效日、推定比例、观察价格比例和来源运行哈希，正式 PIT 回测不得读取这些观察结果。

本轮额外发现并修正 `510950`：2026-04-27 的原始价格下跌 22.45% 实际来自份额拆分，交易所公告比例为 `1.28769271`。修正后该日总收益变动为约 -0.14%，不再被误记为市场损失。

## 4. 数值与血缘检查

| 检查 | 结果 |
|---|---:|
| 价格 `(asset, date)` 重复 | 0 |
| 事件 `(asset, event_date)` 重复 | 0 |
| 早于上市 / 晚于退市 | 0 / 0 |
| 非有限数值 | 0 |
| 非正价格或因子 | 0 |
| OHLC 关系错误 | 0 |
| 调整后单日绝对收益超过 21% | 0 |
| `pit_actionable=true` | 0 |
| `historical_backtest_allowed=true` | 0 |
| `model_promotion_allowed=true` | 0 |
| 运行清单记录 | 6,812 |
| 缺失文件 / SHA-256 不匹配 | 0 / 0 |

最大一组正常价格变动约为 20.08%，集中在 2024-09-30 和 2024-10-08 的科创板、创业板 ETF 涨停附近，没有超过 21% 的剩余跳变。

## 5. 隔离资产

| ETF | 原因 |
|---|---|
| `159927` | 供应商历史延伸到治理退市日之后，疑似代码复用或错误拼接 |
| `159942` | 供应商历史延伸到治理退市日之后，疑似代码复用或错误拼接 |
| `511210` | 终止前分配事件无法对齐到有效交易日 |
| `511230` | 终止前分配事件无法对齐到有效交易日 |

这 4 只不会通过截断、删除事件或沿用新证券价格的方式强行修复。需要基金清算公告、交易所退市安排和独立净值历史共同确认。

## 6. 证据边界

当前观察链可用于：

- 检查全状态 ETF 生命周期覆盖；
- 发现退市尾部、代码复用和公司行动异常；
- 建立后续人工补证清单；
- 验证复权算法与数据治理逻辑。

当前观察链不可用于：

- 声称历史可交易时点已经 PIT 化；
- 运行可晋级的 ETF 回测或 walk-forward；
- 报告年化收益、Sharpe、回撤改善或未来预期收益；
- 用终态分红和份额动作反推历史当时已经知道的信息。

正式晋级至少还需要：分红与份额动作的首次公告/修订版本、跟踪指数历史变更、历史 AUM 与流动性、费率生效区间、指数全收益与历史估值，以及对 129 次启发式动作逐条取得原始证据。
