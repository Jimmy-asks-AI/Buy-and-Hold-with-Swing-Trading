# 日内大单、ROE预测与风格分类复盘

## 复盘对象

- 日内微观结构与高频因子时间窗口。
- 大单阈值精细化与大单净买入因子。
- 预测当期 ROE 与波动率加权。
- 风格特征 K-means 重新分类与类别中性化。

## 可复现假设

1. 因子逻辑决定最佳日内窗口，知情交易类因子应聚焦开盘后 30 分钟。
2. 大单净买入因子比单纯大单买入因子更稳健。
3. 预测当期 ROE 比最新披露 ROE 更接近真实经营状态。
4. 风格分类能补充行业分类，提高因子稳定性和动量溢出解释力。

## 已固化代码

- `intraday_segment_label`
- `intraday_microstructure_summary`
- `aggregate_intraday_factor`
- `refined_large_order_factors`
- `rolling_current_roe_prediction`
- `roe_volatility_weight`
- `volatility_adjusted_roe_factor`
- `kmeans_style_classification`
- `category_neutralize_factor`
- `style_momentum_spillover`

## 验证

- `python -X utf8 -m py_compile` 已通过。
- 小样本烟测覆盖：
  - 3 类日内时段。
  - 960 行开盘后高频因子聚合。
  - 960 行大单因子，其中 912 行大单净买入占比非空。
  - 576 行 ROE 预测，504 行 ROE 波动率权重。
  - 6 个风格聚类，576 行类别中性化因子，544 行风格动量。

## 反证清单

- 若开盘后窗口没有优于全天，不能盲目缩短窗口。
- 大单阈值若对 N 极端敏感，说明订单分布或复权口径有问题。
- 一致预期 ROE 若覆盖率过低，预测模型可能只适用于分析师关注股票。
- ROE 波动率权重若过度压缩周期行业，应按行业调整。
- 风格分类若造成小市值暴露过高，组合层必须加市值约束。

## 修正记录

- 将高频因子从全日默认改为按因子逻辑选择窗口。
- 将大单阈值改为多日对数订单金额分布。
- 将一致预期 ROE 从直接因子修正为当期 ROE 预测输入。
- 将 ROE 高波动处理从剔除修正为置信度收缩。
- 将风格分类从线性正交修正为类别中性化与风格动量两类用途。
