# HIRSSM V2.4 稳定嵌套选择与自检报告

日期：2026-05-24

## 本轮目标

V2.4 不继续扩大参数搜索，而是把 V2.3 的嵌套选择收缩为更稳健的研究版本：

- 只保留 3 个预先声明的稳定收缩参数族。
- 使用 10/20/30bps 多成本情景共同评分，避免只适配单一成本。
- 加入年度参数切换惩罚，降低频繁换模型的自由度。
- 每轮版本交付必须输出自检结果，不能只给收益表。

## 新增/修改文件

- `strategy_lab/hirssm_v2_4_stable_nested_selection.py`
- `configs/hirssm_v2_default.json`
- `outputs/hirssm_v2_4_stable_nested_selection/`

## 关键输出

- `oos_performance.csv`
- `same_period_baseline_comparison.csv`
- `stable_nested_variant_selection.csv`
- `stable_nested_selection_scores.csv`
- `stable_variant_summary.csv`
- `pbo_cscv_report.csv`
- `smoke_test_results.csv`
- `self_check_results.csv`
- `SELF_CHECK_REPORT.md`
- `WALK_FORWARD_REPORT.md`
- `FACTOR_GATE_REPORT.md`
- `MODEL_CHANGELOG.md`

## 主要结果

V2.4 stable nested selection，样本外区间 2007-02-01 至 2026-05-22：

| 成本 | 年化收益 | Sharpe | 最大回撤 | 平均现金 | 平均换手 |
|---:|---:|---:|---:|---:|---:|
| 5bps | 8.16% | 0.418 | -54.47% | 17.23% | 3.26% |
| 10bps | 7.73% | 0.397 | -54.54% | 17.23% | 3.26% |
| 20bps | 6.88% | 0.353 | -54.67% | 17.23% | 3.26% |
| 30bps | 6.04% | 0.310 | -54.80% | 17.23% | 3.26% |

同期间 V2.0 正式基线 10bps：

- V2.0 基线：年化 7.53%，Sharpe 0.374，最大回撤 -54.64%。
- V2.4：年化 7.73%，Sharpe 0.397，最大回撤 -54.54%。
- 增量：年化 +0.20pct，Sharpe +0.023，最大回撤改善约 +0.11pct。

## 选型稳定性

V2.4 实际年度参数切换次数为 1 次：

- 2007-2021：`stable_anchor`
- 2022-2026：`stable_balanced`

本轮修复了报告层错误：最初将第一年从空状态进入 `stable_anchor` 误算为一次切换，现已排除初始状态，只统计真实参数族变更。

## 自检结果

通过：

- Python 语法编译：通过。
- JSON 配置解析：通过。
- Smoke test：目标权重非空、无负权重、权重和正常、无缺失资产和日期。
- 5/10/20/30bps 输出齐全。
- 10bps Sharpe 优于同期间 V2.0 基线。
- 10bps 最大回撤未劣于同期间 V2.0 基线。
- 必要治理报告输出齐全。

未通过：

- PBO 阈值：CSCV PBO = 0.353，高于 0.20 的稳健阈值。

## 结论

V2.4 是 V2.3 之后更稳的研究版本，但不能晋级默认生产模型。

理由：

- 工程链路和自检链路通过。
- 收益、Sharpe、回撤相对同期间 V2.0 小幅改善。
- 参数切换次数显著受控。
- 但 PBO 仍未达标，说明样本内选型与样本外排名之间仍有不稳定风险。

后续 V2.5 应优先减少“年度选择参数族”这类模型自由度，转向更可解释的机制改进，例如 drawdown brake、现金替代门槛、状态条件化目标波动，而不是继续增加 shrinkage 参数网格。
