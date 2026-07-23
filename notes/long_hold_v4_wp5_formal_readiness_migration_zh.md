# 工作包 5 正式评估入口迁移说明

本次加固修改了工作包 5 的正式输入契约。旧的 PASS Gate 和窗口调用不能直接复用，需要重新生成 target manifest 和 Gate run。

## 破坏性变更

1. PIT 数据行必须包含 `source_row_key`、`asset`、`available_date`。
2. `point_in_time_usage.csv` 中的资产和可得日必须与版本化源行完全一致。
3. 正式输入从6份增加到8份，新增：
   - `validation_benchmark_returns`
   - `independent_benchmark_returns`
4. 候选登记表删除：
   - `validation_score`
   - `validation_p_value`
5. 候选目标必须覆盖“所有候选 × 所有验证窗口”。
6. 独立目标文件只能包含验证后冻结的一个候选。
7. 独立测试消费凭证改为固定路径，不能通过更换 `run_id` 重复读取。
8. `write_window_bundle` 必须重新验证 PIT Gate、代码、配置和正式输入绑定，旧的只传哈希调用方式已失效。

## 新增门槛

- 验证选择使用20bps额外滑点后的主动收益；
- 基准固定为中证全指全收益 `000985.CSI`；
- 块级显著性经 Holm 校正后不高于0.05；
- PBO 不高于0.35；单候选记为不适用；
- 冻结候选的 Deflated Sharpe probability 不低于0.95。

任一门槛失败，独立测试不被读取。

## 迁移步骤

1. 把目标生成代码和配置提交到 Git，确认 tracked worktree 干净。
2. 按新契约重新标准化10类数据。
3. 重新生成 PIT 使用清单、8份正式输入和候选登记表。
4. 使用 `formal_data_intake` 写入新的数据修订和 target manifest。
5. 使用新的 `run_id` 重跑 PIT Gate。
6. 先运行 validation，复核候选、成本、基准和失败窗口。
7. 经人工批准后，再一次性消费独立测试。
8. 使用 `formal_walk_forward verify` 复核正式运行。

旧缺口清单保留为2026-07-23的阻断证据，不修改为通过状态。
