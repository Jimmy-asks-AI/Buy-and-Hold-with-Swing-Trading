# 量化模型系统 Smoke Test 复盘

日期：2026-05-23

## 目标

把“系统能运行”固化为可重复验证的 smoke test，覆盖执行约束、point-in-time panel 构建和一键模型系统输出。

## 新增产物

- `strategy_lab/run_quant_system_smoke_tests.py`
- `outputs/quant_model_system_smoke_test/smoke_test_results.csv`
- `outputs/quant_model_system_smoke_test/SYSTEM_RUN_SUMMARY.md`
- `outputs/quant_model_system_smoke_test/paper_tracking/paper_state.json`

## 测试覆盖

1. `test_execution_constraints_respect_tradeability_and_capacity`
   - 验证不可交易资产权重为零。
   - 验证成交额容量约束不会被后续归一化破坏。
   - 验证容量不足时组合保留现金敞口，而不是强制满仓。

2. `test_real_data_adapter_maps_aliases_and_bool_strings`
   - 验证真实数据别名字段可映射到 canonical 字段。
   - 验证 `"False"` 这类字符串不会被错误当成 True。

3. `test_point_in_time_panel_builder`
   - 验证分表合并后无重复 `date-asset`。
   - 验证财务字段可得日期不晚于交易日期。
   - 验证 forward return 覆盖率足够。

4. `test_data_quality_report_gates`
   - 验证数据质量报告包含 summary、字段覆盖、因子覆盖、日期健康度和 gates。
   - 验证合成 panel 不存在重复键。
   - 验证注册表因子至少有一部分可用。

5. `test_data_quality_report_blocks_future_availability`
   - 验证未来可得日期泄漏会触发 fail gate。

6. `test_paper_drift_report`
   - 验证当前持仓可以转换为权重。
   - 验证目标权重与当前持仓之间的买入、卖出、退出和新开仓统计。

7. `test_preflight_outputs_without_backtest`
   - 验证 `preflight` 可以只输出质量报告、注册表审计和 panel validation，不启动回测。

8. `test_one_command_system_demo_outputs`
   - 验证一键 demo 输出 panel validation、walk-forward performance、weights、账本行和纸面跟踪状态。
   - 验证 `paper_state.json` 中 `live_trading_allowed=false`。

## 发现并修正的问题

- 问题 1：执行约束合并时，权重表日期可能是字符串，panel 日期可能是 datetime，真实 CSV 输入中会触发 merge 类型错误。
  - 修正：在 `apply_execution_constraints` 中合并前统一转换日期键为 datetime。
- 问题 2：成交额容量先限幅再强制归一化，会把权重重新放大，突破单资产容量上限。
  - 修正：新增 `_scale_weights_to_caps`，在容量约束下进行带上限的比例缩放；当总容量不足时允许组合保留现金。

## 验证结果

```text
test_execution_constraints_respect_tradeability_and_capacity   pass
test_real_data_adapter_maps_aliases_and_bool_strings           pass
test_point_in_time_panel_builder                               pass
test_data_quality_report_gates                                 pass
test_data_quality_report_blocks_future_availability            pass
test_paper_drift_report                                        pass
test_preflight_outputs_without_backtest                        pass
test_one_command_system_demo_outputs                           pass
```

`compileall` 语法检查通过。实验账本新增 `quant_model_system_smoke_test`，决策为 `promote_to_paper`，仅代表 smoke demo 可进入纸面状态，不代表真实 alpha。

## 后续要求

- 每次修改 `factor_factory_walk_forward.py`、`quant_model_system.py`、`a_share_panel_builder.py` 或 `paper_trading_monitor.py` 后，必须运行 smoke test。
- 真实数据接入后，新增 real-data smoke test：小样本真实 A 股 panel、真实注册表、真实停牌/ST/涨跌停字段和真实成交额容量字段。
