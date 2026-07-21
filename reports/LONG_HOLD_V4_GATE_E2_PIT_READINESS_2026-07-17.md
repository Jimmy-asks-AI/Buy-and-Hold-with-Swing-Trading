# Long Hold V4 Gate E2：历史 PIT 数据资格门禁

> 本报告已由 `reports/LONG_HOLD_V4_GATE_E2_PIT_READINESS_2026-07-18.md` 替代。本文仅保留首次 Gate E2 运行的历史状态。

日期：2026-07-17  
范围：股票、ETF、指数及中美利率历史数据是否具备正式 walk-forward 的输入资格。  
边界：本阶段只验收数据，不运行策略、不计算收益、不允许模型晋级。

## 1. 实现结果

新增：

- `strategy_lab/long_hold_v4/pit_history_gate.py`
- `strategy_lab/long_hold_v4/pit_macro_adapter.py`
- `strategy_lab/long_hold_v4/pit_stock_master_builder.py`
- `strategy_lab/long_hold_v4/pit_stock_fundamentals_builder.py`
- `strategy_lab/long_hold_v4/pit_stock_dividend_builder.py`
- `strategy_lab/long_hold_v4/pit_etf_master_builder.py`
- `strategy_lab/long_hold_v4/pit_etf_benchmark_probe.py`
- `configs/long_hold_v4_pit_gate.json`
- `tests/test_long_hold_v4_pit_gate.py`

运行命令：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_master_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_fundamentals_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_dividend_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_master_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_benchmark_probe --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_macro_adapter --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-17
```

当前状态：

```text
system_status=BLOCKED_MISSING_OR_INVALID_PIT_DATA
historical_inputs_ready=false
stock_history_ready=false
etf_history_ready=false
macro_history_ready=true
walk_forward_completed=false
promotion_allowed=false
```

15类必需数据中，5类通过、10类阻断。

## 2. 已通过的数据

| 数据集 | 行数 | 资产/序列 | 覆盖 | 重要边界 |
|---|---:|---:|---|---|
| `stock_security_master` | 5,876 | 5,538 | 1990-12-01至2026-07-10 | 上交所、深交所和聚宽交叉核对；排除北交所 |
| `stock_fundamentals_pit` | 261,979 | 5,217 | 2004-03-31至2026-06-30 | 最终快照延迟到三表最晚公告/更新日，安全但偏保守；不是完整修订版本库 |
| `stock_dividend_events` | 51,914 | 5,380 | 2000-08-23至2026-07-17 | 只纳入最终现金分配方案；来源缺少可靠派息日 |
| `etf_security_master` | 1,824 | 1,701 | 2005-02-23至2026-07-17 | 事件化上市/退市；123个退市事件，跟踪指数另表治理 |
| `macro_rate_history` | 18,391 | 3 | 2000-01-03至2026-05-26 | 中国10Y、美国10Y和中美10Y利差 |

股票主表采用事件化结构：上市事件在上市日可得，退市事件只在退市日可得，不把未来退市日写回早期上市行。交易所A股日期优先；当聚宽起始日晚于交易所日期超过一年时，使用聚宽日期防止代码迁移或复用被错误回填。当前发现并保护了`300114`迁移到`302132`以及`600018`代码历史边界。

ETF主表同样采用事件化结构，聚宽全状态起止日期与东方财富当前名单的重合率分别为96.64%和98.45%。两源差异保留在manifest中，没有用当前名单覆盖历史记录。

所有五类数据均带内容哈希、构建代码哈希、原始输入哈希和`source_vintage`。任一输出、代码或原始输入被修改，门禁都会失败关闭。

### 中美利率历史

现有`data_raw/macro/macro_pit_panel.csv`包含中国10Y、美国10Y和中美10Y利差。适配器只读取三条利率序列，不修改原始宏观面板，并用输入文件SHA-256作为`source_vintage`。

| 项目 | 结果 |
|---|---:|
| 行数 | 18,391 |
| 序列 | `CN10Y`、`US10Y`、`CN_US_10Y_SPREAD` |
| 覆盖开始 | 2000-01-03 |
| 覆盖结束 | 2026-05-26 |
| 主键重复 | 0 |
| 无效日期/数值 | 0 |
| `available_date < observation_date` | 0 |
| 未来可得日 | 0 |
| 数据资格 | `pass` |

该面板可作为历史利率状态输入，但不能单独证明利率因子有效，也不能批准策略晋级。

## 3. 仍然阻断的数据

### 股票链：4类

| 优先级 | 数据集 | 主要缺口 |
|---|---|---|
| P0 | `stock_industry_history` | 有生效区间的历史行业成员 |
| P0 | `stock_adjustment_factor` | 可核验复权因子和公司行动 |
| P0 | `stock_trade_state` | 停牌、ST、涨跌停状态 |
| P0 | `stock_valuation_history` | 历史PE、PB、股息率和市值截面 |

### ETF/指数链：6类

| 优先级 | 数据集 | 主要缺口 |
|---|---|---|
| P0 | `etf_benchmark_history` | 有公告日和生效区间的ETF跟踪指数变更历史 |
| P0 | `etf_total_return_prices` | 经审计的分红/份额转换全收益行情 |
| P1 | `etf_aum_liquidity_history` | 历史规模、份额和流动性 |
| P1 | `etf_fee_tracking_history` | 生效区间化费率和跟踪误差 |
| P0 | `etf_dividend_events` | 分红和份额转换事件 |
| P0 | `index_total_return_valuation` | 指数全收益、PE和股息率历史 |

ETF/指数门禁不强制历史PB。当前来源没有可靠PB历史时，系统继续禁止用当前快照回填或制造历史PB。

## 4. 门禁规则

每个数据集至少检查：

1. 文件存在、CSV可读、最低行数和资产数。
2. 必需字段、业务主键完整且唯一。
3. 日期可解析，`available_date`不早于事件日且不晚于本次`as_of`。
4. 覆盖期满足2005年以来的长周期要求。
5. 复权/总收益口径只能是明确允许值，`none_raw`直接失败。
6. 证券主表必须包含退市或清盘样本，当前产品名单不能通过。
7. 行业、费率等生效区间不能重叠。
8. 数值有限，来源和`source_vintage`不能为空。

即使15类数据全部通过，状态也只能升级为`PIT_INPUTS_READY_FOR_WALK_FORWARD`。只有另行完成 walk-forward 后才可能讨论晋级；本门禁自身永远不把`promotion_allowed`设为`true`。

## 5. 反例测试

专项测试共17项，覆盖：

- 缺失数据集必须阻断。
- 全部输入合格时只允许进入walk-forward，不能直接晋级。
- 未复权`none_raw`收益口径必须阻断。
- 未来`available_date`必须阻断。
- 没有退市资产的当前证券主表必须阻断。
- 重叠的历史行业/费率区间必须阻断。
- 中美利率三序列映射和`source_vintage`必须保留。
- 上游manifest中的输出、代码或任一原始输入哈希不匹配必须阻断。
- 股票上市与退市必须按事件可得，不能提前暴露未来退市日。
- 证券生命周期冲突必须失败，代码迁移不能把新代码回填到旧历史。
- 分红只使用最终公告，季度查询不能越过`as_of`。
- 财务最终快照必须延迟到最晚来源日期，供应商占位日期不能污染可得日。
- ETF上市与退市必须按事件可得，未来上市产品不能提前进入主表。
- ETF供应商重复代码必须失败。
- 受限跟踪指数切片必须按公告日可得并过滤未来记录。

## 6. 获取优先级

1. `stock_adjustment_factor + stock_trade_state`：先补足股票可投资总收益和真实可交易性。
2. `stock_valuation_history`：建立不使用当前截面回填的深度低估历史信号。
3. `stock_industry_history`：恢复历史行业相对估值和行业仓位约束。
4. ETF跟踪指数变更、全收益价格和分红事件。
5. ETF规模、费率、跟踪误差及指数估值历史。

当前Tushare权限只能提供未复权股票日线。聚宽全状态证券接口已补齐股票和ETF生命周期，但`FUND_INVEST_TARGET`可见部分只有461条、覆盖ETF主表27.1%，已另存观察集并固定禁止回测。现有权限不能绕过其余10项阻断，缺失数据继续记录在`outputs/long_hold_v4/pit_gate/missing_data_queue.csv`。

## 7. 结论

Gate E2 已从文档要求升级为可执行、可测试、失败关闭的数据资格系统。股票生命周期、财务、现金分红、ETF生命周期和宏观利率基础链已经合格，但股票价格/可交易性/估值/行业链以及ETF跟踪指数、全收益和运营历史仍未就绪，因此仍不能运行正式V4 walk-forward，也不能引用任何V4预期收益。
