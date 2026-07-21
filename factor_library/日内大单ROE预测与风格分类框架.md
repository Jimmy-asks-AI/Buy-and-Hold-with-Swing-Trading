# 日内大单、ROE预测与风格分类框架

## 模型卡 1：日内时间窗口选择

- 类型：高频因子数据切片。
- 窗口：`open30`、`middle`、`close30`、`exclude_open30`。
- 规则：
  - 知情交易类因子优先 `open30`。
  - 过度反应类因子优先 `exclude_open30`。
  - 收盘前窗口单独测试。
- 代码：`intraday_segment_label`、`intraday_microstructure_summary`、`aggregate_intraday_factor`。

## 因子卡 1：大单净买入

- 类型：高频资金行为、知情交易。
- 阈值：多日单成交金额对数分布的 `mean + N * std`，N 默认 1。
- 保险：可加入绝对金额阈值，低 N 时更有意义。
- 因子：
  - 大单买入占比。
  - 大单净买入占比。
  - 大单买入强度。
  - 大单净买入强度。
- 推荐窗口：开盘后 30 分钟。
- 必做控制：行业、市值、估值、换手、反转、波动、流动性。
- 代码：`refined_large_order_factors`。

## 因子卡 2：预测当期 ROE

- 类型：基本面预测、质量因子改进。
- 输入：最新披露 ROE、一致预期 ROE、历史真实 ROE。
- 方法：滚动回归预测当期真实 ROE。
- 风险：一致预期覆盖率低、分析师关注偏差、数据可得日。
- 代码：`rolling_current_roe_prediction`。

## 因子卡 3：ROE 波动率置信权重

- 类型：基本面信号可靠性。
- 定义：历史 ROE 波动率倒数的横截面标准化权重。
- 用途：把高波动 ROE 信号向横截面均值收缩。
- 代码：`roe_volatility_weight`、`volatility_adjusted_roe_factor`。

## 模型卡 2：风格特征重新分类

- 类型：横截面聚类、风格风险控制。
- 特征：市值、估值、盈利、关注度。
- 方法：K-means，可用行业均值作为初始质心。
- 应用：
  - 类别中性化因子。
  - 类别偏离风控。
  - 风格分类动量溢出。
- 风险：小市值暴露提高、分类稳定性不足、行业强分化年份可能弱于行业中性。
- 代码：`kmeans_style_classification`、`category_neutralize_factor`、`style_momentum_spillover`。

## 入库标准

1. 高频因子必须测试全天、开盘后、剔除开盘后、收盘前四类窗口。
2. 大单因子必须报告阈值敏感性、绝对金额阈值敏感性和指数股票池分层。
3. ROE 改进因子必须用可得日数据滚动预测，不能使用未来当期真实 ROE。
4. 风格分类必须报告分类稳定性、与行业相似度、小市值暴露和风格动量独立性。
