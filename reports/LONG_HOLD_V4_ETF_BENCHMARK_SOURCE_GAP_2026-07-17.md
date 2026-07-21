# ETF跟踪指数历史数据源审查

日期：2026-07-17  
用途：判断当前聚宽权限下的`FUND_INVEST_TARGET`能否进入Long Hold V4正式历史回测。

## 结论

不能。当前可见数据只有461条、461只ETF，公告日期从2025-04-08开始，有效日期从2025-04-14开始；相对于1,701只ETF的全状态主表，资产覆盖率仅27.1%。该切片不足以重建2005年以来ETF跟踪指数及其变更历史，也无法完整覆盖退市产品。

数据被保存为观察集：

`data_raw/long_hold_v4/pit_history/observations/etf_benchmark_history_joinquant_limited.csv`

它没有写入正式门禁目标`etf_benchmark_history.csv`，其manifest固定：

```text
qualification_status=BLOCKED_INSUFFICIENT_HISTORY_AND_UNIVERSE_COVERAGE
historical_backtest_allowed=false
model_promotion_allowed=false
```

## 实测结果

| 项目 | 结果 |
|---|---:|
| 原始记录 | 461 |
| 可用记录 | 461 |
| ETF数量 | 461 |
| 全状态ETF主表 | 1,701 |
| 资产覆盖率 | 27.1% |
| 公告日期 | 2025-04-08至2026-06-23 |
| 有效日期 | 2025-04-14至2026-07-01 |
| 历史回测资格 | 否 |

## 为什么单独拆表

ETF生命周期和跟踪指数不是同一种事实。上市、退市属于产品生命周期；跟踪标的可能在存续期间变更，应以`effective_from/effective_to`表达。若把当前跟踪指数直接写入ETF主表，会把后来的映射回填到过去，并且一只ETF无法表示多段跟踪关系。

正式数据集至少需要：

`asset, index_code, index_name, effective_from, effective_to, announcement_date, available_date, data_source, source_vintage`

## 后续获取要求

1. 覆盖2005年以来全部存续及退市ETF，不只覆盖当前产品。
2. 保留每次跟踪标的变更的公告日和生效区间。
3. 与基金合同、招募说明书或交易所公告抽样核对。
4. 不得用当前基金档案或当前指数名称回填历史。

在上述条件满足前，`etf_benchmark_history`继续阻断，ETF历史回测不得晋级。
