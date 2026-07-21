# Factor Factory Data Contract

更新时间：2026-05-23

## 目标

本数据契约定义因子工厂的最小输入。任何真实 A 股多因子实验都必须先满足这里的字段语义，再运行 `strategy_lab/factor_factory_runner.py`。

## Panel 表

最小字段：

| 字段 | 含义 | 要求 |
|---|---|---|
| `date` | 调仓日或截面日期 | 必须是可交易日或月/周频调仓日 |
| `asset` | 股票代码 | 必须包含历史退市股票，不能只用当前成分 |
| `fwd_return` | 下一持有期收益 | 只能作为标签，不能参与因子计算 |
| `industry` | 历史行业分类 | 必须使用当期可得分类 |
| `log_mkt_cap` | 对数市值 | 用于规模中性化 |
| 因子原始列 | 注册表中 `column` 指向的字段 | 必须在预测时点可得 |

推荐字段：

- `amount`：成交额，用于容量和流动性约束。
- `is_tradeable`：是否可交易，过滤停牌、涨跌停、一字板等。
- `listed_days`：上市天数，过滤新股。
- `is_st`：ST 状态。
- `float_mkt_cap`：流通市值。
- `benchmark_weight`：基准权重，用于指数增强。
- `next_open_return` / `next_vwap_return`：更接近可交易价格的标签。

## Registry 表

模板：`data_catalog/factor_registry_template.csv`

必填字段：

| 字段 | 含义 |
|---|---|
| `factor_id` | 因子唯一编号 |
| `column` | panel 中的原始因子列名 |
| `family` | 因子家族 |
| `direction` | 因子方向，`1` 表示越大越好，`-1` 表示越小越好 |
| `horizon` | 目标持有期 |
| `data_type` | 数据类型 |
| `availability_col` | 可得日期字段，可为空 |
| `cost_tier` | 成本层级，`low/medium/high` |
| `description` | 定义说明 |

## 可得日期

财报、分析师、公告、文本、基金、融资融券、Level2、宏观数据必须有可得日期或入库日期。规则：

```text
availability_col <= date
```

违反该规则的因子不得进入模型。

## 频率

因子工厂默认按低频截面工作：

- 月频：`periods_per_year = 12`
- 周频：`periods_per_year = 52`
- 日频：`periods_per_year = 252`

日内和 Level2 因子必须先聚合到调仓截面，不应把未来日内数据拼入当期因子。

## 输出

标准输出目录包含：

- `meta.csv`
- `availability_audit.csv`
- `ic_by_date.csv`
- `ic_summary.csv`
- `quantile_returns.csv`
- `turnover.csv`
- `factor_correlation.csv`
- `clusters.csv`
- `quality.csv`
- `selected.csv`
- `prepared_panel.csv`
- `weights.csv`
- `portfolio_returns.csv`
- `performance.csv`
- `run_manifest.json`

实验账本：

- `logs/factor_factory_experiment_ledger.csv`

