# Factor: ETF 60 日动量

Domain: momentum, ETF rotation
Universe: liquid ETF pool
Frequency: daily calculation, monthly rebalance candidate
Data Fields: trade_date, etf_code, adj_close, amount, listed_date

## Rationale

过去 60 个交易日表现较强的 ETF，可能因为趋势延续、资金跟随和信息扩散继续表现较好。

## Formula

```text
momentum_60d = adj_close_today / adj_close_60_trading_days_ago - 1
```

方向：越高越好。

## Universe Filter

- 上市时间足够长，至少覆盖动量窗口和后续持有期。
- 日均成交额足够高。
- 跟踪指数清楚。
- 同类 ETF 只保留流动性较好的代表。
- 暂不纳入杠杆、反向、复杂跨市场产品。

## Strategy Candidate

```text
调仓频率：月度
调仓日：每月最后一个交易日
成交日：下一交易日
买入：动量排名前 3
权重：等权
成本：单边 0.05%-0.10%
基准：ETF 池等权，另看宽基指数
```

## Evaluation

- 年化收益。
- 最大回撤。
- 年化波动。
- 夏普。
- 超额收益。
- 换手。
- 成本敏感性。
- 分年度表现。
- 持仓集中度。

## Failure Modes

- 趋势反转。
- ETF 上市时间短导致样本不稳。
- 行业集中。
- 产品幸存者偏差。
- 流动性不足。
- 拥挤交易。
- 基准错配。

## Implementation Notes

月末收盘计算信号后，成交应滞后到下一交易日。ETF 池等权是检验轮动价值的第一基准。

