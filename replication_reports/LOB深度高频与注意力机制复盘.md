# LOB深度高频与注意力机制复盘

## 复盘对象

- LOB 还原和 TWAP/成交概率应用。
- LOB 买入意愿分解。
- RNN+NN 深度学习高频因子。
- 注意力机制与残差注意力机制。

## 可复现假设

1. 逐笔还原 LOB 能提供快照以外的信息，并改善执行成本估计。
2. LOB 组件的线性复合增量有限，非线性模型更适合高频特征。
3. RNN+NN 可以从高频序列中提取周度选股 alpha。
4. 注意力机制改善长序列遗忘，残差注意力比简单注意力更稳健。

## 已固化代码

- `reconstruct_lob_from_events`
- `lob_relative_strength`
- `order_flow_relative_strength`
- `limit_order_execution_probability_target`
- `decompose_buying_intention_lob`
- `ic_weighted_composite`
- `make_highfreq_sequence_dataset`
- `rolling_train_validation_splits`
- `rank_ic_loss`
- `orthogonalization_layer`
- `attention_pool`
- `residual_attention_pool`
- `factor_autocorrelation`

## 验证

- `python -X utf8 -m py_compile` 已通过。
- 小样本烟测覆盖：
  - 40 行 LOB 快照。
  - 40 行订单簿相对强弱。
  - 4 个 LOB 买入意愿组件。
  - 96 行 IC 加权复合。
  - 高频序列张量形状 `(250, 5, 2)`。
  - 4 个滚动训练/验证切分。
  - 300 行正交层输出。
  - 注意力权重和为 1，残差注意力输出维度正确。

## 反证清单

- LOB 还原若无法复现交易所快照，后续成交概率和 TWAP 测试无效。
- 深度学习因子若正交后失效，说明只是已知风格的非线性表达。
- 单一随机种子或单一切分表现不能证明模型有效。
- 注意力机制若只降低换手但降低 IC，需要从成本后收益判断是否值得。
- 残差注意力仍需检查模型拥挤和样本外衰减。

## 修正记录

- 第七十七篇 PDF 文件损坏，未标记为已学习，进入待修复队列。
- 根据第七十九篇引用，仅记录第七十七篇的三项明确改进方向：正交层、训练/验证切分重设、预测目标调整。
- 将深度学习高频框架从“模型结构崇拜”修正为数据切分、目标函数、正交层、注意力、换手共同治理。
