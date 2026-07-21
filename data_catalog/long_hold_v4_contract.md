# Long Hold V4 Data Contract

## 1. 当前研究快照

路径：`data_raw/long_hold_v4/research_snapshot.csv`

每个标的必须恰好一行。所有比例字段使用小数，例如`6% = 0.06`。`available_date`是该行所有衍生指标底层数据中最晚的真实可得日；不能使用报告期代替公告日，也不能把当前截面回填到历史。财务记录存在`UPDATE_DATE`时，PIT可得日取`max(NOTICE_DATE, UPDATE_DATE)`，防止使用后来修订值回测更早日期。

### 公共字段

| 字段 | 含义 |
|---|---|
| `as_of_date` | 快照生成日期 |
| `available_date` | 底层信息最晚可得日期，必须不晚于研究日 |
| `asset` / `name` | 证券代码和名称 |
| `asset_type` | `stock`或`etf` |
| `sector` | `bank`、`insurance`、`utility`等配置允许值 |
| `is_tradeable` / `is_st` | 未知可交易状态按不可交易处理 |
| `history_years` | 可用历史年数 |
| `annual_vol_3y` / `max_drawdown_3y` | 使用复权或全收益序列计算 |
| `pe_percentile_5y` | 标的或ETF跟踪指数自身历史分位，0最便宜 |
| `yield_spread_cn10y` | 股息率减中国10年国债收益率 |

### 股票字段

`positive_profit_years_5y, dividend_years_5y, dividend_yield, dividend_cagr_5y, dividend_cut_count_5y, payout_ratio, roe_mean_5y, roe_std_5y, revenue_cagr_5y, profit_cagr_5y, profit_cv_5y`

股票估值还必须提供`pb_percentile_5y, sector_pe_percentile, sector_pb_percentile`。同行业分位只能使用同一PIT截面。

行业专用字段：

- 银行：`npl_ratio, provision_coverage, core_tier1_ratio`
- 保险：`solvency_ratio, new_business_value_cagr_3y`
- 公用事业及一般企业：`debt_to_assets, interest_coverage, fcf_dividend_coverage`

经营现金流和自由现金流不用于银行硬门槛，因为金融企业的现金流量表含义与工业企业不同。

### ETF字段

`aum_cny, avg_daily_amount_cny, expense_ratio, tracking_error_1y, index_history_years, distribution_years_5y, index_dividend_yield, index_earnings_cagr_5y, pe_percentile_5y, total_return_history_ready`

ETF字段日期必须单独提供：

`price_available_date, nav_available_date, valuation_available_date, aum_available_date, distribution_available_date, expense_available_date, index_available_date, total_return_available_date`

每个字段分别检查未来值和最大日龄；最新价格不能掩盖陈旧估值。ETF历史PB目前没有可靠官方序列，因此ETF不要求PB，也不允许用当前PB快照回填历史。指数PE和股息率使用中证指数“计算用股本”口径；跟踪误差使用基金披露日增长率对官方指数全收益日收益。

ETF只有在底层指数全收益历史可用时才能进入可投资回测。价格指数只能做技术状态观察。

### ETF当前构建边界

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.etf_snapshot_builder --as-of 2026-07-17 --max-assets 30
```

构建器输出`etf_research_snapshot.csv`，再与`stock_research_snapshot.csv`原子合并为`research_snapshot.csv`。ETF可投资价格由未复权OHLC、现金分红和可确认的份额转换重建为`total_return`口径。无事件证据的25%以上跳变会失败。

退市 ETF 的终止现金、继受份额和持仓延续不得由最后收盘价推断；三态证据结论、PIT 日期和晋级门槛见[`long_hold_v4_etf_terminal_event_contract.md`](long_hold_v4_etf_terminal_event_contract.md)。

该构建器使用当前ETF列表和当前抓取的完整历史，因此所有行固定`current_universe_only=true, historical_backtest_allowed=false`。当前数据链可用于今日筛选，不能据此声称历史回测无偏。

### ETF Gate E2历史契约

ETF生命周期与跟踪指数历史必须分表保存。

生命周期主表路径：`data_raw/long_hold_v4/pit_history/etf_security_master.csv`

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_master_builder --as-of 2026-07-17
```

每只ETF至少有一条`listing`事件；已退市ETF另有一条`delisting`事件。上市行的`delist_date`必须为空，退市日只在退市事件发生时可得。业务主键为`asset + available_date + list_status`。必需字段为：

`asset, asset_name, list_date, delist_date, list_status, event_type, exchange, fund_type, available_date, data_source, source_vintage`

当前主表包含1,701只ETF、1,824个事件和123个退市事件。聚宽提供全状态起止日期，东方财富当前列表只做在市集合交叉核验；名称是当前供应商标签，不得作为历史信号。

跟踪指数历史目标路径：`data_raw/long_hold_v4/pit_history/etf_benchmark_history.csv`

必需字段为：

`asset, index_code, index_name, effective_from, effective_to, announcement_date, available_date, data_source, source_vintage`

当前聚宽权限只暴露2025年4月以后的461条记录，资产覆盖率27.1%。它们保存于`pit_history/observations/`且固定`historical_backtest_allowed=false`，不能进入正式目标表。详细边界见`reports/LONG_HOLD_V4_ETF_BENCHMARK_SOURCE_GAP_2026-07-17.md`。

### 股票 Gate E2复权因子契约

目标路径：`data_raw/long_hold_v4/pit_history/stock_adjustment_factor.csv`

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_builder --as-of 2026-07-17 --collect-limit 0
python -X utf8 -m strategy_lab.long_hold_v4.pit_stock_adjustment_validator
```

每行代表一次稀疏因子事件，而不是把未来已知的最终因子回填到每个交易日。必需字段为：

`asset, effective_date, adj_factor, adjustment_basis, available_date, data_source, source_vintage`

固定约束：

- `adjustment_basis=hfq_cumulative`，因子必须为有限正数。
- 事件必须处于证券上市至退市的生命周期内，`available_date=effective_date`且不晚于本次`as_of`。
- 资产覆盖必须为生命周期主表的100%。
- 合格现金分红除权日与因子事件同日匹配率不得低于99.5%；阈值不能通过配置静默放宽。
- 未匹配分红事件必须完整写入构建manifest，不得删除后重新计算通过率。
- 二次验证必须读取全部已接受Tushare日线源并逐文件记录SHA-256；输入不得少于6,000个唯一哈希。
- 因子调整后的相邻收盘价在连续交易区间出现30%以上绝对跳变时，资格失败。
- 超过10个自然日交易间隔的大跳变进入长停牌复核表，不静默视为正常公司行动。
- 验证代码和全部输出必须有哈希，连续交易异常表必须为空。

当前结果为63,970个事件、5,537个资产；49,776个合格分红除权日中49,769个同日匹配。二次验证完成53,063次跳变检查，连续交易硬异常为0，169个长交易间隔事件保留复核。数据层`historical_backtest_allowed=true`只表示该因子表可以作为后续历史构建输入，`model_promotion_allowed`仍固定为`false`。

## 2. 可投资价格文件

路径：`data_raw/long_hold_v4/prices/{asset}.csv`

必需字段：`date, open, high, low, close, return_basis`。价格必须是前复权、后复权或全收益口径；允许值为`qfq_adjusted, hfq_adjusted, total_return`。原始未复权收盘价和`price_index_proxy`会被当前投资入口拒绝。

历史回测额外要求：`asset, asset_type`。回测只使用收盘后信号并在下一可交易日开盘执行，逐日计算真实持仓，不使用重叠未来收益标签。

## 2A. 历史条件诊断

路径：`outputs/long_hold_v4/historical_diagnostic/`

该目录使用当前申万银行、保险、公用事业成分，按每个历史月末可得的公告、更正、分红、估值、利率和过去价格重建截面。财务信息使用`max(NOTICE_DATE, UPDATE_DATE)`；中国10年国债使用`available_date <= as_of_date`；滚动价格特征只使用当月末以前的行。

这不是历史股票池回测。所有行必须固定携带：

- `diagnostic_scope=current_2026_constituents_conditioned_history`
- `universe_history_status=current_constituents_only_survivorship_bias`
- `price_history_status=qfq_latest_vintage_not_pit_adjustment_history`
- `historical_backtest_allowed=false`
- `promotion_allowed=false`

`BUILD_1`只表示假设每个月都从空仓开始时满足入场筛选；它不表示实际持仓路径。条件T信号还必须通过耐久性门槛，并假设核心仓已经建立。该目录不得用于报告年化收益、Sharpe、回撤或策略晋级。

## 3. 账户文件

路径：`portfolio_lab/long_hold_v4/account.json`

```json
{
  "schema_version": 2,
  "account_id": "long_hold_v4_primary",
  "base_currency": "CNY",
  "as_of_date": "2026-07-17",
  "cash_cny": 500000.0,
  "holdings": [],
  "realized_pnl_cny": 0.0,
  "gross_dividend_cny": 0.0,
  "dividend_tax_cny": 0.0,
  "net_dividend_cny": 0.0,
  "processed_fills": [],
  "fill_history": [],
  "processed_events": [],
  "event_history": [],
  "nav_history": []
}
```

持仓行固定字段：

`asset, name, asset_type, sector, core_shares, core_average_cost_cny, core_open_date, t_shares, t_average_cost_cny, t_open_date, full_target_shares_reference, realized_pnl_cny`

约束：

- 代码必须是六位A股或ETF代码，股数必须为非负整数。
- 非空持仓必须有名称、行业、正成本和有效开仓日。
- T仓必须依附核心仓；核心仓不得在T仓存在时被减到配置要求以下。
- `full_target_shares_reference`是分批核心仓和T仓比例的分母，不再依赖手工维护比例。
- 账户可实现盈亏允许为负；现金不得为负。
- `processed_fills`是成交幂等索引，`fill_history`是账户内权威成交历史。
- `processed_events`和`event_history`分别是公司事件幂等索引和权威事件历史。
- `nav_history`保存日终现金、市值、净值、峰值、回撤和风险状态。

### 3A. 订单意图

路径：`outputs/long_hold_v4/current/order_intents.csv`

每个可执行意图必须包含：

`order_id, signal_date, valid_through_date, asset, name, asset_type, sector, sleeve, side, shares, indicative_price, target_core_weight, target_t_weight_cap, full_target_weight, full_target_shares_reference, core_fraction_at_signal, t_holding_sessions, risk_override_allowed, status, reason`

只有`status=RESEARCH_INTENT_REPRICE_NEXT_OPEN`可以匹配成交。`review`行不可成交，只表示需要人工风险复核。

### 3B. 成交回执

输入：`portfolio_lab/long_hold_v4/pending_fills.csv`  
输出：`portfolio_lab/long_hold_v4/fill_ledger.csv`

回执必须提供唯一`fill_id`、订单ID、成交日、证券元数据、核心/T子账、方向、整数股数、正成交价、费用模式、人工许可及风险覆盖字段。

- `model`费用模式按配置估算，适合纸面成交。
- `actual`费用模式必须提供`commission_cny, stamp_duty_cny, transfer_fee_cny, other_fees_cny`。
- 收盘信号成交日必须晚于信号日且不超过订单有效期。
- T买入在成交时再次校验核心仓比例和T仓股数上限。
- 配置中的单股票/ETF上限约束核心仓和T仓合计；分配器会预留T仓容量。
- T卖出校验结算规则和最短持有交易日；风险覆盖必须由订单授权。
- 核心仓卖出必须`manual_approval=true`且提供`manual_reason`。

### 3C. 分红与公司行为

输入：`portfolio_lab/long_hold_v4/pending_account_events.json`  
输出：`portfolio_lab/long_hold_v4/account_event_ledger.csv`

支持事件：

- `cash_dividend`：必须提供权益股数、每股现金、毛分红和税费，毛分红需在0.05元容差内核对。
- `dividend_tax`：记录券商后续追缴的差别化红利税，现金不足时阻断。
- `share_adjustment`：送转拆股后的核心仓、T仓和参考股数必须使用相同比例，调整前后账面总成本不变。

事件必须按时间顺序处理，并提供唯一`event_id`和非空`source_ref`。重复事件幂等，修改已处理事件会失败。

### 3D. 净值与回撤状态

输出：`portfolio_lab/long_hold_v4/nav_ledger.csv`

日终净值等于现金加核心/T持仓市值。每日最多一条记录，同日重算替换旧记录。峰值必须单调不降，回撤固定为`nav_cny / peak_nav_cny - 1`。

- `NORMAL`：回撤高于`drawdown_review_trigger`。
- `REVIEW`：回撤不高于复核阈值但高于风险刹车阈值；暂停新增T仓。
- `BRAKE`：回撤不高于风险刹车阈值；停止新增核心/T仓，已有T仓退出，核心仓只人工复核。

当前净值口径假定本金固定，不支持外部资金申购赎回后的份额化净值。

## 4. 成本口径

- 用户账户股票佣金`0.008%`、ETF佣金`0.005%`、无最低佣金。
- A股卖出印花税`0.05%`，股票过户费双向`0.001%`。
- ETF二级市场交易不收印花税。
- 经手费和证管费按券商佣金已包含处理；模型另设每边2bps滑点。

官方参考：[上交所股票交易费用](https://one.sse.com.cn/onething/gptz/)、[中国结算过户费](https://www.chinaclear.cn/zdjs/editor_file/20220701154723234.pdf)、[上交所ETF说明](https://etf.sse.com.cn/fund/learning/knowledge/c/5704298.shtml)。实际交割单与券商口径不一致时，以交割单更新配置。

## 5. 历史验证硬门槛

1. 历史股票池包含退市、暂停上市和曾经ST标的。
2. 财务数据按真实公告/更正时间构建`available_date`。
3. 分红进入全收益或复权回报；高股息策略不得只用价格指数评价收益。
4. 无因子通过的测试期必须记现金收益，不能删除。
5. 5/10/20bps情景、年度结果、回撤、换手、现金和状态分组必须同时输出。
6. 价格代理、当前成分和当前估值快照不能晋级默认模型。
