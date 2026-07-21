# AI 因子挖掘与端到端 AlphaNet 框架

## 目标

建立一套可复用的 AI 因子挖掘流程，用于从量价、基本面、情绪、资金流等原始数据中挖掘候选 alpha，并用严格验证区分增量信息和历史过拟合。

## 模块 1：遗传规划公式挖掘

- 输入：`return1/open/close/high/low/volume/vwap/turn/free_turn` 等原始量价序列。
- 标签：未来 5/10/20 个交易日标准化收益。
- 搜索对象：由基础算子和时序算子组成的公式树。
- 常用算子：`add/sub/mul/div/rank/delay/ts_corr/ts_cov/delta/decay_linear/ts_min/ts_max/ts_rank/ts_sum/ts_stddev/ts_zscore/rank_sub/rank_div/sigmoid`。
- 预处理：MAD 去极值、行业中性、市值中性、短期收益/换手/波动中性、z-score。
- 复杂度约束：公式长度、树深度、节俭系数、重复公式去重、与已有因子相关性上限。
- 输出：公式表达式、训练适应度、验证适应度、RankIC、ICIR、分层收益、IC 衰减、相关性和解释。

## 模块 2：适应度函数

### RankIC 适应度

- 适用：单调线性或近似单调因子。
- 定义：每个截面因子值与未来收益 Spearman 相关系数的均值。
- 优点：稳定、可解释、与传统单因子测试兼容。
- 缺点：忽略非线性关系。

### 互信息适应度

- 适用：非线性因子、机器学习合成模型。
- 定义：因子与未来收益共享信息量，可用分位离散化估计。
- 优点：能捕捉中间分层最优等非单调结构。
- 缺点：不直接给出排序方向，样本少时估计不稳。

### 多头超额收益适应度

- 适用：只做多或指数增强策略。
- 定义：分层后 Top 或 Bottom 组合相对基准的较大超额收益。
- 优点：贴近投资目标。
- 缺点：对样本期、费用、换仓路径敏感，容易过拟合。

## 模块 3：非线性因子处理

- 分层曲线单调：保留原始排序方向。
- 中间层最优：使用机器学习模型直接合成，或做三次方残差/多项式变换。
- 两端同时有效：拆为极端暴露因子或构造 U 型/倒 U 型风险暴露。
- 质量门槛：转换后必须重新做中性化、IC、分层和样本外测试。

## 模块 4：AlphaNet 端到端网络

### AlphaNet-v1

- 输入：9 个原始量价特征，30 日历史窗口。
- 结构：特征提取层、池化层、全连接层、线性输出。
- 自定义层：`ts_corr/ts_cov/ts_stddev/ts_zscore/ts_return/ts_decaylinear/ts_mean`，每层后接 BN。
- 训练：过去 1500 个交易日，时间顺序 1:1 划分训练/验证，每半年滚动训练，10 个随机种子平均。

### AlphaNet-v2

- 输入：v1 特征 + 6 个比率类特征，形成 15 x 30 数据矩阵。
- 结构：特征提取层 + LSTM + BN + 输出层。
- 训练：时间顺序 4:1 划分训练/验证，验证集更偏近期样本。
- 改进逻辑：特征提取后的结果仍有时序结构，LSTM 比直接池化更合适。

### AlphaNet-v3

- 输入：v2 特征。
- 结构：10 日和 5 日两组特征提取层 + GRU + BN + 输出层。
- 改进逻辑：多周期特征提取增加表达能力，GRU 减少参数量。

## 模块 5：验证与复核

- 样本切分：训练、验证、样本外必须按时间顺序，不能随机切。
- 滚动训练：每半年或每季度重新训练，记录训练窗口、验证窗口、预测窗口。
- 随机种子：神经网络至少重复多次训练，报告预测均值和离散度。
- 增量信息：对合成因子做行业、市值、短期收益、短期换手、短期波动五因子中性化。
- 交易约束：涨跌停、停牌、VWAP、手续费、换手率限制、容量。
- 衰减监控：按年份、牛熊、大小盘、流动性、发布前后分段评估。

## 代码映射

- `GeneticProgrammingConfig`：GP 默认参数。
- `default_gp_function_set`：GP 算子库说明。
- `preprocess_factor_by_date`：截面预处理。
- `rank_ic_fitness`：RankIC 适应度。
- `mutual_information_fitness`：互信息适应度。
- `long_excess_fitness`：多头超额收益适应度。
- `cubic_residual_transform`：三次方残差非线性变换。
- `fit_polynomial_transform` / `polynomial_factor_transform`：滚动多项式变换基础组件。
- `validation_convergence`：验证集收敛检测。
- `alphanet_feature_list` / `alphanet_architecture`：AlphaNet 版本配置。
- `rolling_train_validation_windows`：滚动训练和验证窗口。
- `make_panel_image`：量价数据图片构造。

## 使用边界

AI 因子挖掘只能提供候选 alpha，不直接等于可交易策略。任何进入组合的候选因子都必须经过传统因子检验、样本外检验、交易成本检验、容量检验和经济解释复核。

