# HIRSSM V2.0 专家剪枝评估报告

日期：2026-05-24

## 目标

按照因子有效性标准，对 HIRSSM V2.0 中的专家/因子族进行评价，并测试移除表现糟糕专家后，对最终收益率、回撤和 Sharpe 的影响。

## 使用标准

本轮不只看历史收益，而是综合：

- 专家 RankIC 方向和稳定性
- 专家消融后对组合收益、波动、回撤、Sharpe 的影响
- 是否存在明确经济解释
- 是否可能是全样本事后调参
- 是否应进入默认模型，还是仅作为观察候选

## 已确认应移除的专家

### 1. `range_reversal`

结论：默认禁用。

证据：

- style RankIC：-0.0476
- industry RankIC：-0.0213
- 旧模型中移除该专家后，年化收益和 Sharpe 均提高

解释：当前反转实现无法有效区分“超跌修复”和“下跌趋势延续”，方向性证据为负。

### 2. `style_trend_continuation`

结论：默认禁用。

证据：

- style `trend_expert_score` RankIC：-0.0295
- style `relative_strength_score` RankIC：-0.0372
- 禁用 `style_trend_continuation` 后，收益、Sharpe、最大回撤同时改善

对比 10bps 情景：

| 模型 | 默认禁用专家 | 年化收益 | Sharpe | 最大回撤 |
|---|---|---:|---:|---:|
| 修复后基准 | `range_reversal` | 7.26% | 0.368 | -54.81% |
| 保守剪枝后 | `range_reversal`, `style_trend_continuation` | 8.67% | 0.455 | -54.55% |

解释：A 股宽基/风格指数层面的月频趋势和相对强弱，在当前样本中更像追涨杀跌，不能继续作为 style 层默认正向专家。

## 观察候选：暂不默认移除

### `liquidity_overlay`

证据混合：

- industry `liquidity_score` RankIC 为正：0.0184
- style `liquidity_score` RankIC 为负：-0.0469
- 在保守剪枝基础上再禁用全局 `liquidity_overlay`，年化收益提升到 9.29%，Sharpe 提升到 0.489，但这会同时删掉行业层可能有用的信息。

结论：不直接默认全局禁用。下一步应拆分成 `style_liquidity_overlay` 和 `industry_liquidity_overlay` 的独立门控，再做 walk-forward 验证。

## 全组合剪枝实验结果

实验文件：

- `outputs/hirssm_v2_pruning_full/expert_pruning_variants.csv`
- `outputs/hirssm_v2_pruning_granular/expert_pruning_variants.csv`

较优但需谨慎的组合：

| 组合 | 年化收益 | Sharpe | 最大回撤 | 判断 |
|---|---:|---:|---:|---|
| 禁用 `style_trend_continuation` | 8.67% | 0.455 | -54.55% | 保守采用 |
| 禁用 `style_trend_continuation` + 全局 `liquidity_overlay` | 9.29% | 0.489 | -54.76% | 观察候选 |
| 禁用 `style_trend_continuation` + `industry_liquidity_overlay` | 9.39% | 0.493 | -54.88% | 疑似组合过拟合，暂不采用 |

## 当前正式默认模型

`configs/hirssm_v2_default.json` 已将以下专家默认禁用：

- `range_reversal`
- `style_trend_continuation`

修复后正式输出：

- `outputs/hirssm_v2_0/HIRSSM_V2_MODEL_RUN_REPORT.md`
- `outputs/hirssm_v2_0/cost_sensitivity_summary.csv`
- `outputs/hirssm_v2_0/latest_target_weights.csv`

10bps 成本情景：

- 总收益：649.53%
- 年化收益：8.67%
- 年化波动：19.05%
- Sharpe no RF：0.455
- 最大回撤：-54.55%
- 平均现金权重：23.14%

## 不应过度解释的地方

- 这些剪枝结果仍是全样本诊断，不能直接证明未来有效。
- `style_trend_continuation` 被禁用是因为 RankIC 和消融共同失败，证据较强。
- `liquidity_overlay` 虽然全局禁用后收益更高，但行业层 RankIC 为正，因此暂不全局删除。
- 组合剪枝越多，越容易变成事后调参；正式晋级必须做 walk-forward。

## 下一步

1. 实现 walk-forward 专家门控，按过去窗口 RankIC 决定专家是否启用。
2. 拆分 `style_liquidity_overlay` 与 `industry_liquidity_overlay`。
3. 对 `style_trend_continuation` 做样本外观察，若未来恢复有效，再通过门控动态启用。
4. 为所有默认禁用专家保留观察输出，不从代码中删除，避免失去后续再评估能力。
