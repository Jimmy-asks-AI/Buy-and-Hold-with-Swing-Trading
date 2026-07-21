# 交易约束、纸面跟踪与 Panel 构建器复盘

日期：2026-05-23

## 目标

上一阶段已经形成一键运行的 walk-forward 因子工厂，但仍有三个关键缺口：

1. 目标权重不能绕开停牌、不可交易、成交额容量等执行约束。
2. `promote_to_paper` 之后必须生成纸面跟踪状态，而不是被误读为实盘许可。
3. 真实 A 股数据通常分散在行情、财务、行业等多张表中，需要 point-in-time panel 构建器。

## 新增产物

- `strategy_lab/factor_factory_walk_forward.py`：新增 `tradeable_col`、`amount_col`、`fund_size`、`max_participation_rate`，并在测试期目标权重生成后应用交易可行性和容量约束。
- `strategy_lab/paper_trading_monitor.py`：从最新目标权重生成 `paper_state.json`、`target_weights_latest.csv`、`paper_monitoring_checklist.csv`，且默认 `live_trading_allowed=false`。
- `strategy_lab/a_share_panel_builder.py`：把行情、财务、行业表按 `asset` 和可得日期做 backward as-of 合并，生成可回测的 point-in-time panel。
- `strategy_lab/quant_model_system.py`：在 demo 和真实 walk-forward 路径中自动初始化纸面跟踪状态。

## 验证结果

- walk-forward 执行约束 demo：25200 行、140 只资产、180 个日期、9 个 split 全部完成，单边成本 20 bps，实验账本决策为 `promote_to_paper`。
- 纸面跟踪：生成 16 个最新目标持仓，总敞口 1.0，资金规模 1,000,000，状态为 `paper_tracking`，实盘许可为 `false`。
- Panel 构建器 demo：生成 25600 行、80 只资产、320 个日期的 point-in-time panel。
- 全部 `strategy_lab` 模块通过 `compileall` 语法检查。

## 复习与纠错

- 纠错 1：目标权重不是可交易订单。现在权重生成后会先过滤不可交易证券，再按成交额参与率和资金规模做容量缩放。
- 纠错 2：`promote_to_paper` 不等于实盘可用。纸面状态文件强制写入 `live_trading_allowed=false`。
- 纠错 3：真实数据不能假设已经是 panel。新增构建器要求财务和行业字段通过可得日期向后合并，避免财报期末日期回填。
- 纠错 4：`merge_asof` 对排序敏感。实现已改为按单个 `asset` 分组做 as-of 合并，再拼回全表，避免跨资产排序污染。

## 仍未解决

- 需要真实 A 股行情、财务公告日、行业分类、停牌涨跌停、ST、新股和复权数据来替换合成 demo。
- 容量约束当前是成交额参与率近似，后续仍需引入冲击成本、滑点和订单层模拟。
- 纸面跟踪目前只生成目标权重与检查清单，后续需要加入真实成交记录、漂移监控和再平衡差异归因。
