# Long Hold V4 正式历史绩效数据契约 v1

状态：`DATA_REQUIRED`

用途：工作包 5 的 PIT Gate、purged/embargoed walk-forward、独立测试和历史绩效复核。

本契约只定义“哪些数据可以进入正式评估”。通过契约不等于策略有效，也不等于允许投入真实资金。

## 1. 必需数据集

PIT Gate 要求以下 10 类数据全部通过，不能用当前快照回填历史：

| dataset_id | 最低时间语义 |
|---|---|
| `stock_industry_classification_history` | 行业生效起止日期、发布日、历史分类版本 |
| `stock_financial_statement_vintages` | 报告期、首次披露日、修订披露日、版本号 |
| `stock_historical_valuation` | 交易日、PE/PB/股息率及当日可得日 |
| `etf_benchmark_history` | 跟踪标的变更公告日、生效日、旧/新基准 |
| `etf_total_return_prices` | 交易价格、分红、份额调整、终止现金事件 |
| `etf_historical_aum_liquidity` | 历史份额、规模、成交额、换手率及可得日 |
| `etf_historical_fees` | 管理费/托管费/其他费率的生效区间 |
| `etf_historical_tracking_error` | 计算窗口、方法版本、披露日 |
| `index_total_return_history` | 全收益指数，不得用价格指数替代 |
| `index_historical_valuation` | 历史 PE/PB/股息率及方法版本 |

每个标准化 CSV 至少必须包含：

| 字段 | 要求 |
|---|---|
| `source_row_key` | 同一数据修订内唯一、稳定，可追到供应商原始记录 |
| `asset` | 股票、ETF、指数或行业的标准代码 |
| `available_date` | 该行首次可用于研究决策的日期 |

PIT 使用清单会把每次历史决策绑定到这三个字段。清单声明的资产和可得日必须与版本化源行完全一致；源行缺失、重复或日期不一致时，Gate 失败。

## 2. PIT 使用清单

`point_in_time_usage.csv` 必须包含：

```text
dataset_id,source_revision_id,source_row_key,asset,decision_date,available_date
```

硬约束：

- `available_date <= decision_date`；
- `source_revision_id` 必须等于 target manifest 绑定的修订；
- 每个必需数据集都必须至少有一条实际使用记录；
- 清单中的 `available_date` 和 `asset` 必须与源数据行一致；
- 同一数据行可在多个决策日复用，但同一决策使用记录不能重复。

## 3. 正式评估输入

target manifest 必须绑定以下 8 份文件：

| role | 内容 |
|---|---|
| `validation_execution_states` | 验证期逐资产可执行行情和交易状态 |
| `validation_target_weights` | 所有预登记候选在全部验证窗口的完整目标快照 |
| `validation_benchmark_returns` | 验证期中证全指全收益日收益 |
| `independent_execution_states` | 一次性独立测试期逐资产执行状态 |
| `independent_target_weights` | 仅冻结候选的独立测试目标 |
| `independent_benchmark_returns` | 一次性独立测试期中证全指全收益日收益 |
| `trading_calendar` | 严格递增、无重复的资产级评估交易日历 |
| `candidate_registry` | 预登记候选参数与训练信息 |

验证文件不得包含独立测试期日期。独立测试文件在验证门槛通过并写入全局消费凭证后才会解析。

## 4. 候选登记

`candidate_registry.csv` 必须包含：

```text
candidate_id,parameters_json,train_score,split_roles_used
```

禁止提供 `validation_score` 或 `validation_p_value`。正式运行器会对每个候选重新回测，按最严格的 20bps 额外滑点计算相对中证全指全收益的主动收益，再计算：

- 主动收益 Sharpe；
- 20 个交易日块级单侧显著性；
- Holm 多重检验校正；
- CSCV/PBO；
- Deflated Sharpe probability。

默认门槛：

- Holm 调整后 `p <= 0.05`；
- PBO `<= 0.35`；单一预登记候选记为不适用；
- Deflated Sharpe probability `>= 0.95`。

验证门槛不通过时，独立测试不会被读取。

## 5. 基准收益

验证期和独立期的基准文件必须包含：

```text
date,benchmark_id,total_return,available_date,return_basis,historical_backtest_allowed
```

要求：

- `benchmark_id=000985.CSI`；
- `return_basis=total_return`；
- `total_return` 为日收益小数，不是指数点位；
- `historical_backtest_allowed=true`；
- `available_date <= date`；
- 每个策略估值日都有一条基准收益。

价格指数、事后拼接的全收益代理和当前成分回填都不能替代正式基准。

## 6. 入库与运行顺序

代码和配置先提交到 Git；正式入库要求 tracked worktree 干净。

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.formal_data_intake `
  --project-root . `
  --intake-config data_raw/long_hold_v4/formal_intake.json

python -X utf8 -m strategy_lab.long_hold_v4.pit_gate_v2 `
  --project-root . `
  --config configs/long_hold_v4_work_package_5_pit_gate.json `
  --target-manifest data_raw/long_hold_v4/formal_target_manifest.json `
  --run-id <pit_gate_run_id>

python -X utf8 -m strategy_lab.long_hold_v4.formal_walk_forward run `
  --project-root . `
  --pit-gate-run-directory <pit_gate_run_directory> `
  --strategy-config configs/long_hold_v4.json `
  --walk-forward-config configs/long_hold_v4_work_package_5_walk_forward.json `
  --run-id <formal_run_id>
```

先运行 validation，不加 `--consume-independent-test`。确认候选、PBO、DSR、成本和失败窗口后，经过人工批准，才可用一个新的 `run_id` 增加：

```text
--consume-independent-test
```

同一研究输出根目录只允许生成一份 `holdout_consumption/independent-test.json`。更换 `run_id` 不能再次读取同一独立测试期。

## 7. 输出边界

正式运行会生成：

- 每个验证/独立窗口的订单、成交、未成交、账户、NAV、风险和核心/T 仓归因；
- 0/5/10/20bps 的策略、基准和主动收益；
- 候选验证收益、显著性、PBO、DSR；
- 全部输入、代码、配置和输出哈希；
- `promotion_allowed=false`；
- `live_trading_allowed=false`；
- `manual_review_required=true`。

历史评估完成后仍需独立代码复核、至少 60 个交易日纸面跟踪、实盘前风险审批和券商执行链测试。
