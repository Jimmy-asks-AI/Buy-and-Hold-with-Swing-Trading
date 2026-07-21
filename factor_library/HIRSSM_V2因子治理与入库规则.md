# HIRSSM V2 因子治理与入库规则

日期：2026-05-24

## 目标

约束 HIRSSM V2 中的行业轮动、大小盘切换和状态识别因子，避免高相关指标重复计票、未来函数、过拟合和不可解释的黑箱权重。

## 因子入库字段

每个因子必须记录：

```text
factor_id
factor_name
family
asset_scope
formula
direction
availability_rule
point_in_time_safe
current_snapshot_only
winsorize_rule
standardization_rule
neutralization_rule
expected_failure_mode
validation_required
```

## 因子分层

### 原始因子

直接由价格、成交量、估值或状态计算。

示例：

- `ret_60`
- `ma_gap_120`
- `vol_60`
- `pb_percentile`

### 家族因子

同一家族内因子先聚类，再合成。

示例：

- `trend_momentum_family_score`
- `risk_compression_family_score`
- `liquidity_crowding_family_score`

### 专家因子

多个家族因子进入专家模型。

示例：

- `trend_expert_score`
- `valuation_repair_expert_score`
- `defensive_expert_score`

### 元模型因子

专家得分按状态条件化集成。

示例：

- `state_conditioned_style_score`
- `state_conditioned_industry_score`

## 去冗余规则

同一日期、同一资产池内：

1. 计算因子相关矩阵。
2. 过去 756 个交易日滚动估计相关性。
3. 相关性高于 0.75 的因子进入同一簇。
4. 每簇只保留一个代表因子，或对簇内因子等权平均。
5. 不允许同一信息源在总分里重复计票。

## 正交规则

行业因子需要尽量剥离市场 beta：

```text
industry_return = alpha + beta_market * market_return + residual
```

残差动量使用 `residual` 的滚动累计收益。

大小盘风格因子需要区分：

- 绝对收益。
- 相对收益。
- 估值 spread。
- 风险 spread。

不能把这些混为同一个动量因子。

## 状态使用规则

状态因子不直接排序资产。

允许用途：

- 调整专家权重。
- 调整 sleeve 风险预算。
- 调整风控阈值。
- 调整调仓频率。

禁止用途：

- 单独作为买入/卖出信号。
- 为每个状态单独过度优化参数。

## 估值规则

宽基估值：

- 可以用于历史回测。
- PE/PB 分位必须只使用当日以前数据。

行业估值：

- 当前只有快照。
- 只能用于当前解释报告。
- 不能进入历史回测。

估值因子的通用规则：

- 低估不是买入条件。
- 低估 + 趋势企稳，或低估 + 风险收缩，才允许加分。

## ML 专家规则

ML 排序专家必须满足：

- walk-forward。
- 多随机种子。
- 特征重要性稳定性。
- PBO 检查。
- Deflated Sharpe 检查。
- 权重上限 15%。

禁止：

- 单次全样本训练后直接回测。
- 用未来归一化。
- 让 ML 输出直接变成最终权重。

## 必做验证

每个因子家族必须输出：

- 覆盖率。
- RankIC。
- ICIR。
- 分年度 RankIC。
- 分状态 RankIC。
- 与其他家族相关性。
- 换手贡献。
- 成本后收益贡献。
- ablation 后组合表现变化。

## 晋级规则

因子从候选到生产研究分三层：

1. `candidate`：有定义和经济解释，但未验证。
2. `research_validated`：通过样本外和 ablation。
3. `paper_trading`：进入纸面跟踪，但不能实盘授权。

当前 HIRSSM V2 只允许进入 `research_validated` 或以下状态，不允许直接标记为实盘可用。
