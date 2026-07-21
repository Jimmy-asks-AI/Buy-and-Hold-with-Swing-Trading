# Long Hold V4：ETF 历史观察链审查

审查日期：2026-07-18  
市场数据截止日：2026-07-17  
资格状态：`OBSERVATION_ONLY_CURRENT_UNIVERSE_FINAL_SNAPSHOT`

## 1. 结论

当前筛选的 30 只境内红利 ETF 已全部形成可复核的自身行情、分红与净值观察链：共 29,685 条日频价格、216 条分红记录，价格覆盖 2007-01-18 至 2026-07-17。30 只全部通过本轮内部质量检查，24 只存在至少一条分红记录。

这批数据仍然**不得用于正式历史回测或模型晋级**。ETF 名单来自 2026 年当前截面，存在幸存者偏差；累计分红是当前最终快照，没有历史首次公告时间、修订版本和经独立核验的除息日。系统固定写入：

```text
pit_actionable=false
historical_backtest_allowed=false
model_promotion_allowed=false
```

## 2. 构建内容

构建命令：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_history_observation_builder --as-of 2026-07-17
```

输出：

- `data_raw/long_hold_v4/pit_history/observations/etf_total_return_prices_current_universe.csv`
- `data_raw/long_hold_v4/pit_history/observations/etf_dividend_events_current_universe.csv`
- `data_raw/long_hold_v4/pit_history/observations/etf_history_asset_status_current_universe.csv`
- `data_raw/long_hold_v4/manifests/etf_history_observation_latest.json`

价格表同时保留原始收盘价、总回报复权 OHLC、来源现金分红、份额折算后现金分红、份额折算因子、总复权因子、单期总回报和以 1 为起点的总回报指数。`market_available_date`只描述原始市场价格可见日；复权结果的`available_date`统一使用本次最终快照观察日，不能伪装成历史 PIT 可得日。

## 3. 本轮发现并修复的问题

初次构建隔离了两只 ETF，复核后确认是公共复权函数的两类缺陷，而非应删除的数据：

| ETF | 原现象 | 核验结果 | 修复 |
|---|---|---|---|
| `515100` | 2023-12-21 原始收盘价下跌 25.64% | 同日每份现金分红 0.437 元可完整解释除息跳变 | 在公司行动拦截前先计算含现金分红总回报 |
| `562060` | 2025-08-11 原始价格约减半 | 分红表同日累计值不变，净值表给出`FHFCZ=2.0`，属于 2:1 份额折算 | 用零增量事件识别份额折算，并将折算前现金分红换算到当前份额口径 |

修复后 30/30 通过。单期总回报只由已经包含现金流的复权收盘价递推，避免再次叠加现金分红造成双重计数。

## 4. 验证证据

- 资产/日期主键重复：0
- 分红事件主键重复：0
- OHLC 边界异常：0
- 总回报指数最大递推误差：`3.34e-13`
- 输入哈希：93/93 一致
- 代码哈希：3/3 一致
- 输出哈希：3/3 一致
- 全项目测试：140项及5个子测试全部通过
- Gate E2：本专项完成时为5/15；股票交易状态后续通过后更新为6/15，ETF六类阻断不变

## 5. 尚未关闭的正式缺口

要把观察文件升级为正式 `etf_total_return_prices` 和 `etf_dividend_events`，至少还需要：ETF 全生命周期名单、跟踪指数变更公告、每次分红的公告日/除息日/派息日、份额折算公告及其首次可得日、历史版本化事件表，以及第二来源核验。当前结果适合研究复权逻辑和数据覆盖，不适合报告收益、Sharpe、回撤改善或未来预期收益。
