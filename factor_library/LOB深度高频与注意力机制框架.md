# LOB深度高频与注意力机制框架

## 模型卡 1：LOB 还原

- 输入：标准化逐笔委托、成交、撤单事件。
- 输出：逐事件或逐时间片订单簿快照。
- 关键字段：订单编号、方向、价格、数量、事件类型、时间戳。
- 风险：交易所字段差异、市价单价格补全、撤单归属、全成交流水缺失。
- 代码：`reconstruct_lob_from_events`。

## 因子卡 1：LOB 相对强弱

- 订单簿相对强弱：买盘深度与卖盘深度之差除以总深度。
- 订单流相对强弱：买侧成交/挂单/撤单与卖侧对应量之差除以总量。
- 用途：预测短期限价单成交概率、刻画盘口压力。
- 代码：`lob_relative_strength`、`order_flow_relative_strength`、`limit_order_execution_probability_target`。

## 因子卡 2：买入意愿 LOB 分解

- 组件：净挂单、净撤单、净成交、被动净买入。
- 复合方式：等权、逐次正交、IC 加权。
- 风险：组件相关性高，线性复合增量有限。
- 代码：`decompose_buying_intention_lob`、`ic_weighted_composite`。

## 模型卡 2：深度学习高频因子

- 输入张量：`N x T x F`，如股票数 x 20 日序列 x 高频特征数。
- 目标：未来 5 日收益、超额收益或风险调整后收益。
- 损失：IC 或 RankIC。
- 切分：滚动训练/验证，不随机打乱。
- 必做：输出正交层、训练多随机种子、报告样本外 IC 和组合收益。
- 代码：`make_highfreq_sequence_dataset`、`rolling_train_validation_splits`、`rank_ic_loss`、`orthogonalization_layer`。

## 模型卡 3：注意力与残差注意力

- 问题：输入频率提升、序列变长后，GRU/LSTM 遗忘早期信息。
- 注意力：对历史隐含状态分配权重，提高长序列信息保留。
- 残差注意力：`last_state + attention_pool(history)`，缓解简单注意力在旧 regime 的失效。
- 评价：IC、ICIR、多头超额、因子自相关、Top 组合换手、分年度稳定性。
- 代码：`attention_pool`、`residual_attention_pool`、`factor_autocorrelation`。

## 入库标准

1. LOB 因子必须先通过交易所字段标准化和撮合逻辑校验。
2. 算法交易测试必须包含延迟和强制成交。
3. 深度学习因子必须正交已知因子并报告增量 IC。
4. 注意力模型必须比较 10 分钟与 30 分钟输入频率。
5. 模型复杂度提升必须带来样本外收益、换手或回撤改善，不能只看训练集 IC。
