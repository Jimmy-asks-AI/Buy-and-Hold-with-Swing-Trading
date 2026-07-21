# 宏观T值筛选、异质动量与盈利趋势因子框架

## 因子卡 1：T 值约束 MacroBeta

- 类型：宏观敏感性、状态条件选股、风险控制。
- 原始数据：个股收益、宏观变量变化、市场收益、风格控制因子。
- 定义：滚动回归 `ret ~ macro_change + controls`，记录宏观项 beta 和 t-stat。
- 信号：当 `abs(t-stat) >= 1.65/1.96` 时，才允许把宏观方向映射为多空信号。
- 方向：取决于 `macro_forecast_direction * beta_sign`。
- 失败场景：宏观方向不可预测、指标发布滞后、t 值不稳定、样本期宏观制度切换。
- 代码：`macro_sensitivity_models.rolling_macro_sensitivity_stats`、`macro_tvalue_signal`。

## 因子卡 2：异质动量 IMom

- 类型：残差动量、行为延续、风险剥离后的趋势。
- 原始数据：个股收益、共同因子收益。
- 默认窗口：12 个月，最少 8 个有效样本。
- 定义：对每只股票滚动回归历史收益，剥离共同因子后取残差均值；可用残差波动率做风险调整。
- 方向：越高越好。
- 必做控制：市值、行业、换手率、特质波动、短期反转、停牌和流动性。
- 失败场景：市场从下跌快速反弹、金融行业共同因子解释不足、残差估计样本太短。
- 代码：`short_horizon_factor_models.idiosyncratic_momentum_factor`、`market_state_for_momentum`。

## 因子卡 3：GP 盈利趋势

- 类型：质量趋势、基本面动量。
- 原始数据：单季 Gross Profitability 或可比盈利能力指标。
- 默认窗口：4 个可比季度。
- 定义：对同一股票的历史盈利能力序列做 OLS，斜率作为趋势因子。
- 推荐口径：`seasonal_lag=4` 的同比趋势优先于 `seasonal_lag=1` 的环比趋势。
- 方向：越高越好。
- 必做控制：当前 GP、GP 同比增速、市值、估值、行业、ROE、反转、波动。
- 失败场景：公告日处理错误、单季异常项目、行业季节性强、样本点过少。
- 代码：`fundamental_factor_portfolios.rolling_profitability_trend`。

## 数据卡：一致预期底层财年

- 类型：预期数据治理。
- 目标：在构建预期 ROE、NP、NPG、G 等时间序列前，先统一底层财年口径。
- 方法：
  - `current_year`：取当前财年预测。
  - `smooth_next_year`：按报告季节混合当前财年和下一财年。
  - `locked_fiscal_year`：锁定目标财年。
- 默认建议：ROE、NP、NPG 优先测试锁定财年；G 同时关注覆盖率和远期预测可靠性。
- 风险：预测类型、供应商覆盖、极端预测、财年切换、预测入库滞后。
- 代码：`analyst_expectation_factors.select_consensus_fiscal_year_series`。

## 入库标准

1. MacroBeta 类信号必须同时报告 beta、t-stat、显著股票比例和宏观方向来源。
2. IMom 必须与 raw momentum 对照，并报告共同因子剥离版本差异。
3. GP 趋势必须区分环比趋势、同比趋势和原始 GP。
4. 预期因子必须在时间序列标准化前锁定底层财年选择。
5. 所有因子必须输出覆盖率、换手、成本后收益、分股票池表现和失败状态。
