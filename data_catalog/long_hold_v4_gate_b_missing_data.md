# Long Hold V4 Gate B 数据缺口

更新日期：2026-07-17  
用途：记录从历史条件诊断升级为可晋级股票/ETF回测仍需取得的数据。本文不保存账号、密码或Token。

## 当前边界

- Tushare当前权限已验证可获取股票未复权日线，但`stock_basic`、`adj_factor`、`daily_basic`及财务等接口未形成可用生产链路。
- JoinQuant全状态证券接口已用于股票和ETF生命周期；试用权限下的财务/基金专题表覆盖仍短，不能支持完整牛熊周期验证。
- 本地178只股票的财务、分红、估值和QFQ价格足以做当前名单条件化诊断，但不能消除幸存者偏差或复权版本泄漏。
- 中美利率PIT面板已经通过Gate E2：18,391行，覆盖2000-01-03至2026-05-26，包含中国10Y、美国10Y和中美10Y利差。该项不再列为缺失，但不能替代股票/ETF历史数据。
- 股票全状态主表、复权因子、PIT财务、股票现金分红事件和ETF全状态主表已经通过；数据通过不代表因子或策略有效。

## P0 必需数据

| 数据集 | 最低字段 | 需要解决的问题 | 当前状态 | 可接受来源 |
|---|---|---|---|---|
| 全状态证券主表 | `asset, list_date, delist_date, list_status, event_type, exchange, available_date` | 纳入退市和历史在市股票，退市日不提前泄漏 | 已通过：5,537资产、5,875事件；排除北交所和CDR | 交易所历史清单+JoinQuant交叉核对 |
| 历史申万行业成员 | `date/effective_from, effective_to, asset, sw_code, available_date` | 避免使用2026年行业成分回填历史 | 缺失 | 申万历史成分、JoinQuant按日期行业成员或商业数据源 |
| 复权因子或全收益价格 | `date, asset, adj_factor`，或明确`total_return` OHLC | 正确处理分红、送转、拆并股和长期收益 | 已通过：63,970个稀疏因子事件、5,537资产；分红同日匹配99.9859%；169个长停牌事件待复核 | Sina后复权因子事件+Tushare原始日线独立校验；后续补第二来源 |
| 历史交易状态 | `date, asset, is_paused, is_st, limit_up, limit_down, available_date` | 模拟停牌、ST、涨跌停和可成交性 | 缺失 | Tushare停复牌/涨跌停接口、JoinQuant价格与状态接口或商业源 |
| 全历史PIT财务 | 原始报表字段、`report_date, ann_date, update_date, asset` | 对所有历史在市股票计算质量与价值陷阱门槛 | 已通过基础链：261,979行；最终快照保守延迟，尚非逐版本修订库 | 东方财富报表接口；后续需版本化公告源 |
| 历史估值截面 | `date, asset, pe_ttm, pb, dividend_yield, market_cap, available_date` | 自身估值分位和行业横截面估值 | 仅当前178只股票 | Tushare `daily_basic`、JoinQuant估值表或商业源 |
| 股票现金分红事件 | `announcement_date, ex_date, pay_date, cash_per_share` | 重建现金分红和总收益 | 已通过49,874行；生命周期和代码迁移已校正，派息日仍缺可靠来源 | 东方财富分配方案；后续以公告源补派息日 |

## ETF 必需数据

| 数据集 | 最低字段 | 验收要求 |
|---|---|---|
| ETF生命周期主表 | `asset, list_date, delist_date, list_status, event_type, exchange, fund_type` | 已通过：1,701资产、1,824事件、123个退市事件 |
| ETF跟踪指数历史 | `asset, index_code, effective_from, effective_to, announcement_date` | 当前仅461条、覆盖27.1%，继续阻断；不得用当前映射回填 |
| ETF复权/全收益行情 | `date, open, high, low, close, volume, amount, return_basis` | 分红必须进入收益；停牌日有明确标志 |
| 规模与流动性 | `date, asset, aum_cny, shares, amount` | 每个值有PIT可得日，不用当前规模回填 |
| 费用与跟踪误差 | `effective_from, expense_ratio, tracking_error_1y` | 费率变更按生效区间处理 |
| 指数全收益与估值 | `date, index_code, total_return_level, pe, pb, dividend_yield` | 价格指数不能替代全收益指数评价高股息策略 |
| ETF分红 | `announcement_date, ex_date, pay_date, cash_per_share` | 可重建全收益且公告日不晚于信号日 |

## 接入验收

1. 原始数据单独保存，不覆盖现有文件；每批记录来源、抓取时间和SHA256。
2. `available_date`必须来自公告、生效或交易可得时间，不能统一填写抓取日期后声称历史PIT。
3. `asset + date`或规定业务主键唯一，修订记录必须保留版本而不是静默覆盖。
4. 抽查公司行动前后收益连续性，并与至少一个独立来源核对。
5. 历史每月股票池必须能解释新增、退市、行业迁移、ST和停牌变化。
6. 上述P0数据全部通过前，`promotion_allowed`保持`false`，只能输出条件诊断。

## 手工获取优先级

1. 首先获取`历史交易状态 + 历史行业成员`，补足真实可成交性和行业归属；同时复核现有因子的169个长停牌事件。
2. 然后获取`daily_basic`历史估值截面，并把财务基础链升级为逐版本修订记录、复权链补充第二来源。
3. ETF侧优先补跟踪指数变更、全收益价格和分红事件。
4. 再补ETF规模、费率和跟踪误差历史，不允许用当前值回填。

## 可执行门禁

运行`python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-17`生成机器可读资格结果。当前6/15类数据通过、9类阻断；完整报告见`reports/LONG_HOLD_V4_GATE_E2_PIT_READINESS_2026-07-18.md`。
