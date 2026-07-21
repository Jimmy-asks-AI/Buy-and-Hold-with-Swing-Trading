# 短周期价量、动态反转与 Level2 因子框架

## 因子组 1：短周期价量 alpha

### 价量背离

- 定义：`-corr(vwap, volume, d)`
- 直觉：价格上涨同时成交放大可能反映短期拥挤或追涨；负相关结构可能代表更健康的价格推进。
- 关键参数：`d=5-20` 个交易日，至少覆盖一周与两周窗口。
- 风险控制：行业、规模、短期反转、换手率中性化。

### 开盘跳空

- 定义：`open_t / close_{t-1} - 1`
- 直觉：隔夜信息冲击、情绪跳空和流动性缺口会影响短期延续或反转。
- 风险控制：必须使用下一可交易价格，避免用当日无法成交的开盘信息回填。

### 异常成交量

- 定义：`-volume_t / mean(volume, d)`，实证时推荐使用滞后滚动均量作为分母。
- 直觉：异常放量可能对应短期分歧、冲击交易或拥挤。
- 风险控制：必须扣除停复牌、涨跌停和极端成交异常。

### 量幅背离

- 定义：`-corr(high / low, volume, d)`
- 直觉：放量但振幅扩张可能代表交易冲击和不稳定；量幅背离捕捉微观交易结构变化。
- 风险控制：与波动率、换手率和流动性因子做残差化复核。

## 因子组 2：动态反转

- 市场波段指数：优先用全市场宽基指数定义，例如 Wind 全 A 或可替代全 A 指数。
- 波段阈值：可用固定涨跌幅，也可用日收益波动率乘数转换成动态阈值。
- 窗口规则：从最近一次市场高低点到当前日期的交易日数。
- 因子定义：`- return(stock, dynamic_window)`。
- 最小窗口：低于 20 个交易日时不输出反转因子，或单独归为短窗口状态。
- 复核：按窗口长度分组检验，避免短窗口动量污染反转结论。

## 因子组 3：Level2 成交结构

### 成交占比类

- 定义：各类订单成交金额 / 总成交金额。
- 重点：连续竞价成交占比、中单/小单占比、超大单/大单占比。
- 实证倾向：成交占比类比净买入比例类更稳定，但容易携带规模、反转和换手暴露。

### 主动净买入比例类

- 定义：`(active_buy_amount - active_sell_amount) / (active_buy_amount + active_sell_amount)`。
- 重点：按订单规模拆分后分别评估。
- 实证倾向：材料中整体较弱，不能因为直觉强就放入 alpha 库。

## 组合构建规则

- 预测目标：因子测试的收益定义必须与组合优化的持有期、成交价、行业/风格约束一致。
- 成本目标：使用 `expected_alpha @ weight - cost_rate * turnover / 2` 一类目标函数。
- 容量约束：按个股成交额参与率和组合持仓权重估算尾部分位容量。
- 样本外：至少做滚动训练/验证，短周期因子尤其需要年度、牛熊和成交成本敏感性拆分。

## 代码入口

- `time_series_factor_exposure`
- `cross_sectional_factor_returns`
- `price_volume_divergence`
- `opening_gap`
- `abnormal_volume`
- `volume_amplitude_divergence`
- `rolling_factor_return_forecast`
- `transaction_cost_adjusted_objective`
- `rolling_swing_threshold`
- `market_swing_windows`
- `dynamic_reversal_factor`
- `level2_trade_share_factors`
- `level2_net_buy_ratio_factors`
