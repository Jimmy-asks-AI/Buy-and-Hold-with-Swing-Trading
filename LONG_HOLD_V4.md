# Long Hold Dividend V4

这是本项目唯一的默认投资研究入口。目标不是预测所有A股，而是等待一批可以持有多年的银行、保险、公用事业及高股息ETF，在基本面可靠、估值极低、价格停止快速下跌后分批建立核心仓；核心仓建立后，才允许使用小比例中低频T仓降低成本。

> 数据契约已升级到 v2。目标表、价格口径、年度指标和字段迁移的硬规则见 `data_catalog/long_hold_v4_contract_v2.md` 与 `notes/long_hold_v4_v2_migration_zh.md`。本次升级不调整估值、入场、仓位或风控参数，也不产生新的收益率结论。

## 当前账户

- 初始现金：`500,000 CNY`
- 当前持仓：空仓
- 股票佣金：`0.008%`，无最低佣金
- ETF佣金：`0.005%`，无最低佣金
- 融资和做空：禁止
- 默认行为：数据缺失、陈旧或不满足条件时保持现金

## 决策顺序

1. **长期可持有性**：连续盈利、连续分红、合理派息率、ROE和利润稳定；银行、保险、公用事业分别使用行业会计门槛。
2. **价值陷阱否决**：不良率、偿付能力、杠杆、利息覆盖、自由现金流覆盖等硬门槛任何一项失败即停止。
3. **深度低估**：自身历史估值、行业相对估值和股息率相对中国10年国债利差必须同时具有安全边际。
4. **大幅下跌后企稳**：三年回撤达到阈值，但不在加速创新低；20日波动收缩、均线斜率和路径效率确认盘整。
5. **三批核心仓**：目标持仓的`30% -> 60% -> 100%`，信号在收盘后形成，下一交易日开盘执行。
6. **T仓**：实际核心仓达到完整目标的80%后才启用；T仓不超过单标的完整仓位20%且不超过组合10%，股票不模拟当日回转。

## 组合边界

| 约束 | 默认值 |
|---|---:|
| 核心仓总上限 | 80% |
| 最低现金 | 10% |
| 单只股票 | 12% |
| 单只ETF | 25% |
| 单行业 | 30% |
| 标的数量 | 最多10只 |
| T仓组合上限 | 10% |

单只股票和ETF上限约束核心仓与T仓合计。为T仓预留完整核心仓的20%后，单股完整核心仓默认最多10%，叠加T仓后仍不超过12%。仓位达到上限后不会重新归一化，无法分配的资金留在现金中。

## 数据更新与运行

更新当前股票估值观察与交叉验证链：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.stock_active_valuation_observation_collector --as-of 2026-07-17 --max-fetch 0 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.stock_active_valuation_observation_validator --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.stock_snapshot_builder --as-of 2026-07-19 --sleep-seconds 0
```

当前股票估值观察覆盖178/178只候选、339,927行。聚宽完成2,134次交叉检查并覆盖全部候选，BaoStock完成1,545次检查、覆盖15只候选；33只股票保留PB尾部差异警告。该序列来自2026-07-19实际观察到的最终历史快照，所有历史行共享真实观察可得日，固定`historical_backtest_allowed=false`。候选池来自稳定的`data_catalog/long_hold_v4_watchlist.csv`，不再依赖会被下游重写的`research_snapshot.csv`。

更新当前红利ETF数据链：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.etf_snapshot_builder --as-of 2026-07-19 --max-assets 30
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_history_observation_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_total_return_collector --as-of 2026-07-17 --selection-mode all --attempts 1 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_coverage_collector --as-of 2026-07-17 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_validator
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_events_promoter
```

当前快照构建器按规模和成交额处理红利ETF，重建现金分红和可确认份额转换后的全收益价格，再与股票快照原子合并。历史观察构建器独立处理30只ETF的自身价格、分红和净值，现有29,685条价格和216条分红事件。全生命周期采集器读取1,701只全状态ETF缓存，1,701只全部形成观察结果，共1,466,663条真实价格和1,140条事件/现金账记录，隔离资产为0。正式终止现金表包含118笔、覆盖102只ETF，全部只进入现金账，不补造清算日OHLC；73只事件链完整，29只链未完整，另有21只退市ETF仍为证据不足。原有129次启发式份额动作已经全部取得交易所或巨潮证据，连同既有事件形成152次受治理份额动作，当前推断动作归零。价格缓存仍是当前最终快照，故全收益观察表继续固定`historical_backtest_allowed=false`。

ETF现金分红已单独完成正式PIT链。采集器对全状态主表1,701只ETF逐只查询交易所或巨潮，下载并认证930份官方文件；独立验证器解决19份歧义文件和65份非现金事件文件，未留下未解释文件。正式表包含863条现金分红、147只ETF，公告覆盖2006-05-13至2026-07-16，使用公告日作为`available_date`。该资格只适用于现金分红事件表，不会自动晋级ETF全收益价格或策略模型。详见`reports/LONG_HOLD_V4_ETF_DIVIDEND_PIT_2026-07-19.md`。

官方事件候选表使用863条正式现金分红、152次受治理份额动作和118笔正式终止现金事件重建，覆盖1,701只ETF、1,466,663条价格，0只隔离；859条已生效普通分红进入价格调整，4条未来除息公告保留但未提前应用，118笔终止现金只进入现金账。981条事件使用记录全部对账，事件对齐、终止事件链和收益恒等式失败均为0，价格主键集合与原行情完全一致。聚宽在2025-05-06起的窗口内对285,353条真实成交行情和5,978条净值逐值匹配；腾讯另行取得1,701只、1,469,617条当前最终价格，东方财富取得1,701只、1,483,802条当前最终净值，两者都覆盖2005-02-23至2026-07-17。价格审计单列8条重大价差，`511230`因腾讯遗漏74个微量成交日而只有63.90%日期覆盖。全生命周期当前最终内容通过，价格和净值的采集版本深度仍为0%，历史行只具有2026-07-19的真实采集可得日。因此候选表和独立审计继续固定`historical_backtest_allowed=false`、`formal_table_promotion_allowed=false`。详见`reports/LONG_HOLD_V4_ETF_TERMINAL_EVENT_2026-07-19.md`、`reports/LONG_HOLD_V4_ETF_PRICE_NAV_AUDIT_2026-07-19.md`和`reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_CANDIDATE_2026-07-19.md`。

17个跟踪指数已纳入证据注册表，8个满足身份、缓存和至少1,000个有效PE交易日门槛的映射处于`active`；5个新指数因PE历史不足被降为observation，`930955`因官方静态源超时仍待缓存。探针支持按价格码定向运行，但不会自动激活映射。详见`reports/LONG_HOLD_V4_ETF_HISTORY_OBSERVATION_2026-07-18.md`、`reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_LIFECYCLE_OBSERVATION_2026-07-18.md`和`reports/LONG_HOLD_V4_ETF_INDEX_REGISTRY_2026-07-18.md`。

生成当前决策：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.cli --as-of 2026-07-19
```

每次完整运行发布到`outputs/long_hold_v4/runs/<run_id>/`，`outputs/long_hold_v4/current`仅保存指向最后一次完整运行的JSON指针：

- `readiness.json`：能否开展当前研究及阻断原因
- `candidate_decisions.csv`：长期质量、估值、陷阱、建仓阶段
- `target_weights.csv`：本次运行的目标权重快照
- `order_intents.csv`：仅为下一开盘研究意向，不连接券商
- `timing_proxy_latest.csv`：红利、银行、公用事业、非银指数的价格代理观察
- `agent_decision_log.csv`：九个规则角色的状态日志；当前不是独立上下文运行的真实多 Agent
- `run_manifest.json`：配置、输入、代码和全部业务产物的路径、大小、schema版本与SHA-256
- `run_manifest_seal.json`：对最终`run_manifest.json`进行独立封印

校验当前运行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.run_artifacts verify --output-root outputs/long_hold_v4
```

崩溃留下的`runs/<run_id>.tmp/`不会覆盖`current`，必须通过显式恢复命令移入隔离区。完整威胁模型和迁移步骤见`data_catalog/long_hold_v4_run_artifact_integrity.md`。

## 账户与成交回执

当前账户 schema 为V3，包含单调递增的`state_version`和全账户`state_sha256`。权威文件为`portfolio_lab/long_hold_v4/account.json`；缺失时研究链硬阻断，必须显式初始化或迁移。研究入口只生成意图和订单生命周期状态，不修改账户，也不连接券商。

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.execution --initialize-account --initial-as-of 2026-07-22
# 已有V1/V2账户使用：
python -X utf8 -m strategy_lab.long_hold_v4.execution --migrate-account-state
```

实际或纸面成交发生后，将尚未处理的成交回执填入`portfolio_lab/long_hold_v4/pending_fills.csv`，再运行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.execution --apply
```

执行前还需准备含`asset,price,as_of_date`的全持仓估值快照，并通过`--valuation-prices`传入；缺少任一持仓价格时阻断。执行器重新计算完整订单哈希，校验运行清单、配置、账户版本、交易日历、生命周期、有效期、剩余股数和成交价偏离，再由交易日历重算T仓持有日。模拟成交后会复查现金、单资产、行业、核心仓、T仓和最大持仓数量。账户、成交账本与订单状态使用同一可恢复事务提交。

- `fee_mode=model`：用于纸面成交，按配置佣金、税费和滑点估算。
- `fee_mode=actual`：用于真实交割，必须填写四项实际费用。
- 买入必须匹配可执行研究意图，不能用`manual_approval`绕过。
- 核心仓卖出只允许信封预先授权的人工批准，`manual_approval`本身不能绕过订单。
- 当前股票型ETF和股票都按T+1处理。

完整字段、威胁模型和迁移说明见`data_catalog/long_hold_v4_order_execution_security.md`。

## 分红、公司行为与日终净值

将券商已确认但尚未处理的事件写入`portfolio_lab/long_hold_v4/pending_account_events.json`。字段示例见同目录`account_events.example.json`。处理事件并写入当日净值：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.accounting --mark --as-of 2026-07-17 --apply
```

支持现金分红、后续红利税追缴和核心/T同比例送转拆股。事件必须有唯一ID及券商来源编号；重复导入不会重复记账。当前账户按固定50万元本金运行，不支持中途新增或提取资金后的份额化收益率。

组合日终回撤状态：

- 高于`-12%`：`NORMAL`，正常执行核心仓和T规则。
- 不高于`-12%`：`REVIEW`，允许严格的深度低估核心建仓，但暂停新增T仓。
- 不高于`-20%`：`BRAKE`，停止所有新增风险，退出T仓，核心仓只做人工复核。

当前净值基准为`500,000 CNY`，峰值`500,000 CNY`，回撤`0%`，状态`NORMAL`。完整记录见`reports/LONG_HOLD_V4_GATE_D_ACCOUNTING_2026-07-17.md`。

## 当前状态

截至`2026-07-19`，当前合并快照包含178只银行、保险和公用事业股票，以及17只具备完整当前指数映射链的红利ETF。全体195只中，85只通过数据门、110只阻断，36只通过长期耐久性门槛；动作分布为`KEEP_CASH=178`、`WAIT_DEEP_VALUE=10`、`WAIT_DEEP_DRAWDOWN=6`、`WAIT_STABILIZATION=1`，没有`BUILD_*`。

ETF侧54只关键词候选中深度采集30只，17只形成完整当前快照，13只因指数来源、历史估值或有效PE深度不足而登记为数据缺口。进入快照的17只中，6只通过模型数据门；其余11只主要因ETF自身历史不足三年而阻断。`510880`、`515080`、`515180`通过长期耐久性门，但当前指数PE五年分位约为83%-88%，最终分数低于65，均为`KEEP_CASH`。

系统因此保持`500,000 CNY`现金、零持仓、零订单。红利指数代理更新至`2026-07-17`，银行、公用事业和非银金融代理更新至`2026-07-16`，四条均通过新鲜度检查；ETF自身价格也更新至当前交易日附近，但仍没有满足深度低估、充分回撤和跌后企稳组合条件的标的。

当前股票源清单覆盖`1,079`个输入文件和`2`个构建代码文件，ETF源清单覆盖`166`个输入文件和`4`个构建代码文件；当前run manifest覆盖`204`个直接输入和`75`个V4代码文件，合计`279`项。本轮把ETF分红、终止事件发现/原文/OCR/验证/晋级、生命周期观察、官方事件候选、候选验证、PIT Gate和当前决策纳入16份关键清单，逐项复核22,519个声明哈希，0缺失、0不匹配。20,612个V4配置、代码、原始证据、清单和输出JSON也已严格解析，失败0。采集器、验证器和依赖代码另按内容哈希封存，旧运行不再依赖工作区源码永久不变。源清单只证明对应产物可追踪，不构成因子有效或策略可投证据。

这份快照只允许当前研究，不是历史PIT股票池。V4尚未生成可晋级的历史持仓路径或walk-forward证据，所有V4回测的`promotion_allowed`默认保持`false`。

## 历史条件诊断

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.historical_diagnostic --start 2020-01-31 --end 2026-06-30
```

输出位于`outputs/long_hold_v4/historical_diagnostic/`，包括月度PIT截面、`BUILD_1`筛选和条件T形态。它使用2026年当前成分回看历史，QFQ价格也没有PIT复权版本，因此只能研究规则行为，不能引用为策略收益或晋级证据。完整边界和结果见`reports/LONG_HOLD_V4_GATE_B_DIAGNOSTIC_2026-07-17.md`。

## Gate E2：正式历史数据资格

先构建当前能够取得的历史层，再运行15类历史数据门禁。任何 observation 文件都不能代替正式目标文件：

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
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_market_history_orchestrator --as-of 2026-07-17 --workers 1 --collect-limit 256
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_market_history_builder --as-of 2026-07-17 --skip-collect
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_market_history_validator --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_master_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_benchmark_probe --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_history_observation_builder --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_total_return_collector --as-of 2026-07-17 --selection-mode all --attempts 1 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_coverage_collector --as-of 2026-07-17 --sleep-seconds 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_validator
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_dividend_universe_events_promoter
python -X utf8 -m strategy_lab.long_hold_v4.pit_macro_adapter --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-19
```

当前15类数据中7类通过：全状态股票主表、股票复权因子、全市场股票交易状态、股票现金分红事件、全状态ETF主表、ETF现金分红事件和中美利率；8类仍阻断。股票行业观察表覆盖5,537只证券、12,454行，其中71.58%在有效区间内具有可操作可得日，但当前快照和未验证TLS来源不足以支持历史晋级。基本面观察表按统一市场截止日收敛为261,885行、5,201只证券，因资产负债表和现金流量表缺少逐版本修订时间戳而被降级。

股票交易状态已用2000年以来6,430个交易日文件构建为17,429,361行、5,536个回测期内生命周期资产。未知执行规则占0.4589%；BaoStock 139.35万次停牌检查匹配99.9991%，聚宽停牌、可比ST和涨跌停分别匹配100%、99.9784%和100%。该数据只描述交易状态与原始参考价，不是复权收益。月度估值仍只采集230只、55,667行；剩余股票阻断集中在历史行业版本、逐版本基本面、全市场历史估值和退市股第二PIT估值源。复权因子覆盖5,537只生命周期资产，现金分红除权日同日匹配率为99.9859%；二次验证扫描6,889个Tushare日线源，连续交易区间硬异常为0，另有169个长停牌间隔事件保留复核。

ETF跟踪指数历史观察集只有2025年4月以后461条、覆盖主表27.1%；当前30只红利ETF观察链有29,685条价格和216条分红事件。全状态ETF观察链已覆盖1,701只缓存并形成1,466,663条真实价格；152次份额动作和118笔终止现金均已由正式事件表治理，推断动作归零、隔离资产归零。123只退市ETF全部完成官方标题查询，102只取得正式事件，73只链完整，21只仍证据不足。使用正式事件重建的候选全收益序列已通过确定性检查；腾讯价格和东方财富净值又把当前最终独立内容覆盖扩展至全部1,701只和全部123只退市ETF，统一资格为`PASS_CURRENT_FINAL_PRICE_NAV_CONTENT_PIT_BLOCKED`。两条历史是在2026-07-19事后一次性取得，版本深度为0%，也没有逐交易日的历史可得性证据，故正式`etf_total_return_prices.csv`仍未生成。整体状态仍为`BLOCKED_MISSING_OR_INVALID_PIT_DATA`，7/15类数据通过，`promotion_allowed=false`。当前8项阻断是历史行业、逐版本基本面、股票历史估值、ETF基准变更、ETF全收益价格、ETF历史规模流动性、ETF费率跟踪以及指数全收益估值。通过门禁只表示数据格式、覆盖、可得日和血缘满足当前契约，不表示因子有效或策略可投。详细结果见`reports/LONG_HOLD_V4_GATE_E2_PIT_READINESS_2026-07-19.md`、`reports/LONG_HOLD_V4_ETF_DIVIDEND_PIT_2026-07-19.md`、`reports/LONG_HOLD_V4_ETF_TERMINAL_EVENT_2026-07-19.md`、`reports/LONG_HOLD_V4_ETF_PRICE_NAV_AUDIT_2026-07-19.md`、`reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_CANDIDATE_2026-07-19.md`、`reports/LONG_HOLD_V4_STOCK_MARKET_HISTORY_PILOT_2026-07-18.md`、`reports/LONG_HOLD_V4_STOCK_INDUSTRY_HISTORY_OBSERVATION_2026-07-18.md`和`reports/LONG_HOLD_V4_STOCK_FUNDAMENTALS_OBSERVATION_2026-07-18.md`。

BaoStock估值采集器采用全局限额后分片，每个子进程只写自己的标的缓存，主进程在全部分片结束后统一重建观察输出。BaoStock 是有状态 C/S 协议；8进程压力试验触发过供应商黑名单，生产配置因此硬限制为单连接、单进程。通讯层遇到对端空读取会立即失败并重连，遇到黑名单则熔断剩余请求。正式交易状态改由Tushare逐日文件、交易日历和已验证状态事件流式生成，并由BaoStock、聚宽独立校验；Gate按25万行分块复核主键、日期、枚举、数值关系和全链哈希，避免一次性加载1,743万行。

## 历史系统边界

- `HIRSSM V3.10`：保留为指数轮动治理基线，不再代表本策略。
- `multi_factor_research_framework.py`及旧因子工厂：在重叠收益复利和仓位上限问题修复并重新验证前，不得引用其绩效。
- V2.5至V3.8中使用全面板`crowding_score`的结果：隔离为不可晋级历史产物。
- 所有绩效证据状态统一登记在`data_catalog/performance_evidence_registry.csv`；当前没有任何条目允许收益引用或模型晋级。

详细字段定义见[data contract](./data_catalog/long_hold_v4_contract.md)，当前完整审查见[final audit](./reports/PROJECT_AUDIT_FINAL_2026-07-19.md)。
