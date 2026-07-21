# Long Hold V4 Gate E2：历史 PIT 数据资格门禁

日期：2026-07-18  
范围：股票、ETF、指数及中美利率历史数据是否具备正式 walk-forward 的输入资格。  
边界：只验收数据，不运行策略、不计算收益、不允许模型晋级。

## 1. 当前结论

```text
system_status=BLOCKED_MISSING_OR_INVALID_PIT_DATA
historical_inputs_ready=false
stock_history_ready=false
etf_history_ready=false
macro_history_ready=true
walk_forward_completed=false
promotion_allowed=false
```

15类必需数据中，**6类通过、9类阻断**。新增通过项是`stock_trade_state`；`stock_fundamentals_pit`因缺逐版本报表时间戳被主动降级，因此股票和ETF仍未形成完整历史链。

## 2. 已通过的数据

| 数据集 | 行数 | 资产/序列 | 覆盖 | 重要边界 |
|---|---:|---:|---|---|
| `stock_security_master` | 5,875 | 5,537 | 1990-12-01至2026-07-10 | 事件化上市/退市；排除 CDR；代码迁移单独登记 |
| `stock_adjustment_factor` | 63,951 | 5,537 | 1990-12-01至2026-07-17 | 稀疏后复权累计因子事件；通过分红对齐和价格跳变双重校验 |
| `stock_trade_state` | 17,429,361 | 5,536 | 2000-01-04至2026-07-17 | 生命周期内停牌、ST及涨跌停；未知执行规则0.4589% |
| `stock_dividend_events` | 49,874 | 5,083 | 2000-08-23至2026-07-17 | 按最后交易日重建的生命周期现金分配事件；派息日和版本史仍不完整 |
| `etf_security_master` | 1,824 | 1,701 | 2005-02-23至2026-07-17 | 事件化上市/退市，包含123个退市事件 |
| `macro_rate_history` | 18,391 | 3 | 2000-01-03至2026-05-26 | 中国10Y、美国10Y及中美10Y利差 |

所有通过项都必须同时满足字段、主键、覆盖、`available_date`、来源版本、输出哈希、构建代码哈希和原始输入哈希检查。门禁通过只表示数据可以进入后续验证，不表示因子有效。

## 3. 股票复权因子专项验收

构建命令：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_dividend_builder --as-of 2026-07-17 --reuse-latest
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_builder --as-of 2026-07-17 --collect-limit 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_validator
```

### 3.1 公司行动对齐

| 项目 | 结果 |
|---|---:|
| 证券生命周期资产 | 5,537 |
| 因子覆盖资产 | 5,537 |
| 因子事件 | 63,951 |
| 合格现金分红除权日 | 49,776 |
| 同日因子匹配 | 49,769 |
| 匹配率 | 99.9859% |
| 固定最低门槛 | 99.5% |
| 未匹配事件 | 7 |

7 个未匹配事件保留在构建 manifest 中，不能通过降低阈值或静默删除绕过。构建器同时约束因子事件不得早于上市、晚于退市或越过本次 `as_of`。

### 3.2 独立价格连续性验证

验证器不依赖单个样本，而是读取已接受的全部 Tushare 日线源：

| 项目 | 结果 |
|---|---:|
| 源文件哈希 | 6,889 |
| 非空 / 空占位文件 | 6,395 / 494 |
| 扫描原始行 | 16,988,946 |
| 因子跳变检查 | 53,051 |
| 覆盖资产 | 5,375 |
| 调整后价格绝对跳变超过30% | 169 |
| 其中超过100% | 43 |
| 连续交易区间硬异常 | 0 |
| 超过10个自然日交易间隔的复核项 | 169 |

资格状态为 `PASS`，复核状态为 `LONG_SUSPENSION_REVIEW_REQUIRED`。所有大跳变都发生在长交易间隔后，因此没有触发连续交易硬失败；但长停牌、重组、重新上市和公司行动仍需人工或第二来源复核。

Tushare `pre_close` 只作诊断，不作为除权连续性真值，因为它并非始终使用除权调整口径。验证器以因子调整后的相邻实际收盘价跳变为硬检查。

### 3.3 防篡改条件

门禁要求二次验证 manifest：

- 验证状态必须为 `PASS`，且 `model_promotion_allowed=false`。
- 连续交易硬异常必须为 0。
- 必须明确保留长停牌复核状态。
- 至少包含 6,000 个唯一输入哈希，并与因子数据输入匹配。
- 验证代码、检查明细、异常表和长停牌复核表哈希必须一致。
- 连续交易异常 CSV 必须为空。

## 4. 仍然阻断的数据

### 股票链：3类

| 优先级 | 数据集 | 缺口 |
|---|---|---|
| P0 | `stock_industry_history` | 现有12,454行 observation 缺官方历史发布版本和可信下载证据 |
| P0 | `stock_fundamentals_pit` | 现有261,885行 observation 缺资产负债表和现金流量表逐版本修订时间 |
| P0 | `stock_valuation_history` | PIT PE、PB、股息率和市值截面 |

行业observation覆盖5,537/5,537只证券和338/338只退市证券，8,915/12,454行在有效区间内具有可操作可得日；当前工作簿仍是现时快照，不能证明历史发布版本。交易状态已由6,430个Tushare交易日文件构建为正式数据，并通过BaoStock停牌及聚宽停牌/ST/涨跌停验证。历史估值仍只采集230/5,512只、55,667行；独立三源验证固定96只，退市股第二PIT估值源和全市场覆盖没有解决。

### ETF/指数链：6 类

| 优先级 | 数据集 | 缺口 |
|---|---|---|
| P0 | `etf_benchmark_history` | 跟踪指数变更及公告/生效区间 |
| P0 | `etf_total_return_prices` | 经审计的分红和份额转换全收益行情 |
| P1 | `etf_aum_liquidity_history` | 历史规模、份额和流动性 |
| P1 | `etf_fee_tracking_history` | 生效区间化费率和跟踪误差 |
| P0 | `etf_dividend_events` | 分红和份额转换事件 |
| P0 | `index_total_return_valuation` | 指数全收益、PE和股息率历史 |

聚宽当前可见 ETF 跟踪指数切片只有 461 条、覆盖生命周期主表 27.1%，且起始时间在 2025 年 4 月以后。它只保存在 observation 目录，固定 `historical_backtest_allowed=false`。

ETF自身价格、分红与净值已经与基准映射解耦采集。当前30只红利ETF形成29,685条总回报价格观察和216条分红事件，覆盖2007-01-18至2026-07-17；30/30通过内部质量检查。由于名单来自当前截面，分红表又是缺公告版本的最终快照，这两份输出仅为`OBSERVATION_ONLY_CURRENT_UNIVERSE_FINAL_SNAPSHOT`，不能替代上表中的`etf_total_return_prices`或`etf_dividend_events`。专项报告见`reports/LONG_HOLD_V4_ETF_HISTORY_OBSERVATION_2026-07-18.md`。

全状态 ETF 生命周期采集已完成1,701/1,701只缓存，1,697只形成1,463,741条价格和1,019条事件，覆盖2005-02-23至2026-07-17；4只退市尾部继续隔离。23次份额动作有登记证据，129次启发式动作涉及120只ETF，逐事件记录于`data_catalog/long_hold_v4_etf_share_action_evidence_queue.csv`。数值形状门通过，公司行动证据门和来源质量门失败；结果固定为`COLLECTION_IN_PROGRESS_CURRENT_FINAL_SNAPSHOT`，没有写入正式`etf_total_return_prices.csv`。详见`reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_LIFECYCLE_OBSERVATION_2026-07-18.md`。

当前30只ETF的17个唯一跟踪指数已进入证据注册表：8个映射通过价格、全收益、估值身份、本地缓存和至少1,000个有效PE交易日门槛，可用于当前快照；5个新指数因PE历史不足被降为observation；`930955`因官方静态源超时仍待缓存；另有1个国证指数缺历史估值、2个标普指数仍未解决。登记候选代码不会自动进入快照；即使状态为`active`也固定禁止历史PIT回测，该注册表不能替代ETF全生命周期基准变更。详见`reports/LONG_HOLD_V4_ETF_INDEX_REGISTRY_2026-07-18.md`。

## 5. 运行顺序

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_master_builder --as-of 2026-07-17 --reuse-latest
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_industry_history_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_fundamentals_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_dividend_builder --as-of 2026-07-17 --reuse-latest
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_builder --as-of 2026-07-17 --collect-limit 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_validator
python -X utf8 -m strategy_lab.long_hold_v4.pit_sse_status_announcement_collector --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_sse_factbook_status_collector --as-of 2026-07-17 --workers 4
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_status_event_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_status_event_reconciler --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_status_event_validator --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_tushare_trade_state_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_market_history_builder --as-of 2026-07-17 --skip-collect
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_market_history_validator --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_master_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_benchmark_probe --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_history_observation_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_total_return_collector --as-of 2026-07-17 --selection-mode all --attempts 1 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_macro_adapter --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-17
```

机器可读结果位于 `outputs/long_hold_v4/pit_gate/`，缺口队列为 `missing_data_queue.csv`。

## 6. 验证边界

当前全项目测试为199项全部通过，pytest另报告5个子测试通过；包含代码迁移生命周期、行业可得日、基本面降级、ETF指数注册表激活门、供应商熔断、复权门槛、全市场交易状态、跨块主键、状态验证去循环、验证输入输出哈希、连续交易异常必须为0及绩效证据注册表等反例。

即使 15 类数据全部通过，Gate E2 也只能将状态升级为 `PIT_INPUTS_READY_FOR_WALK_FORWARD`。只有另行完成正式、净化边界的 walk-forward 后，才可以讨论模型晋级和收益表现。

## 7. 结论

股票生命周期、复权因子、每日交易状态、现金分红、ETF生命周期和中美利率基础链已经取得历史数据资格。ETF全状态观察链已消除当前名单选择偏差，但终态来源、129次启发式公司行动及4只退市尾部问题仍未关闭。股票行业版本、逐版本基本面、估值和完整ETF PIT全收益链仍未就绪，因此系统继续失败关闭，不能运行可晋级回测，也不能报告预期收益。
