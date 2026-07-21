# Long Hold V4 续跑检查点

检查点时间：2026-07-19  
当前工作包：退市ETF终止事件全量取证、多轮现金链和官方事件全收益候选  
状态：123只退市ETF已全部查询和取证；历史PIT可得性、21只证据不足资产和29只未完整事件链仍阻断。

## 已确认状态

- 当前账户：500,000元现金，0持仓，0订单。
- 当前决策：`CASH_NO_ENTRY_SIGNAL`。
- 当前候选：195只；动作分布为`178/10/6/1`，分别对应`KEEP_CASH`、`WAIT_DEEP_VALUE`、`WAIT_DEEP_DRAWDOWN`、`WAIT_STABILIZATION`。
- PIT Gate：7/15通过，8项阻断，`promotion_allowed=false`。
- ETF现金分红：863条、147只ETF，正式事件表已通过。
- ETF份额动作：152次受治理动作；原129行补证队列全部解决，推断动作0。
- ETF终止事件：123只全部完成官方查询，456份PDF入库；118笔正式现金事件覆盖102只，73只事件链完整、29只链未完整，另有21只保持`evidence_insufficient`，6笔候选被隔离。
- ETF全生命周期观察：1,701/1,701只、1,466,663条真实价格、1,140条事件/现金账记录、0只隔离；118笔终止现金全部入账且不生成合成OHLC。
- 聚宽近期独立源：285,353条真实成交行情和5,978条净值均100%匹配；独立窗口始于2025-05-06，仅覆盖6/123只退市ETF。
- 腾讯全生命周期价格：1,701/1,701只、1,469,617行，覆盖2005-02-23至2026-07-17；Sina行覆盖99.962773%，8条重大价差单列，`511230`日期覆盖63.90%。
- 东方财富全生命周期净值：1,701/1,701只、1,483,802行，覆盖2005-02-23至2026-07-17；聚宽5,978行逐值一致。
- ETF官方事件候选：1,701只、1,466,663条价格、981条事件使用记录，其中863条普通分红、118笔终止现金；事件对齐失败0、收益恒等式失败0。
- 统一来源资格：`PASS_CURRENT_FINAL_PRICE_NAV_CONTENT_PIT_BLOCKED`；价格和净值版本深度均为0%，正式`etf_total_return_prices.csv`未生成。
- 全量测试：`pytest` 344项和13个子测试通过；`unittest` 287项通过；编译通过。
- JSON：20,612个V4配置、代码、原始证据、清单和输出JSON严格解析，失败0。
- 哈希：16份关键清单22,519项，缺失0、不匹配0。
- Agent治理：12个角色、95份运行清单，active error为0；旧债务727项、35个白名单路径。

## 权威文件

- 当前决策：`outputs/long_hold_v4/current/readiness.json`
- 当前账户：`outputs/long_hold_v4/current/account_summary.json`
- PIT Gate：`outputs/long_hold_v4/pit_gate/readiness.json`
- ETF分红正式表：`data_raw/long_hold_v4/pit_history/etf_dividend_events.csv`
- ETF份额动作解决表：`data_catalog/long_hold_v4_etf_share_action_resolution_ledger.csv`
- ETF终止现金事件：`data_raw/long_hold_v4/pit_history/etf_terminal_cash_events.csv`
- ETF终止事件三态结算：`data_raw/long_hold_v4/pit_history/observations/etf_terminal_event_settled_coverage_registry.csv`
- ETF终止事件独立验证：`outputs/long_hold_v4/pit_validation/etf_terminal_event_v2/run_manifest.json`
- ETF全生命周期观察：`data_raw/long_hold_v4/manifests/etf_total_return_lifecycle_observation_latest.json`
- 聚宽独立源清单：`data_raw/long_hold_v4/manifests/joinquant_etf_price_nav_validation_latest.json`
- 价格净值审计：`outputs/long_hold_v4/pit_validation/etf_price_nav/run_manifest.json`
- 腾讯价格清单：`data_raw/long_hold_v4/manifests/tencent_etf_price_validation_latest.json`
- 东方财富净值清单：`data_raw/long_hold_v4/manifests/eastmoney_etf_nav_validation_latest.json`
- 统一来源资格：`outputs/long_hold_v4/pit_validation/etf_source_qualification/run_manifest.json`
- 官方事件候选清单：`data_raw/long_hold_v4/manifests/etf_total_return_official_event_candidate_latest.json`
- 候选验证：`outputs/long_hold_v4/pit_validation/etf_total_return_candidate/run_manifest.json`
- 人工接口队列：`data_catalog/long_hold_v4_manual_data_interface_queue.md`

## 下一工作包

1. 按不同采集日期继续保存腾讯价格和东方财富净值不可变快照，建立修订监控；另行获取带真实历史发布版本的授权PIT数据，不能把2026-07-19采集日回填到历史。
2. 对21只`evidence_insufficient`资产和29只未完整链继续补证，优先处理6笔被隔离候选、后续份额承接和非现金终止情形；没有官方证据时不得改写为“无事件”。
3. 关闭ETF跟踪基准变更、历史份额/AUM/流动性、费率/跟踪误差，以及指数全收益和历史估值。
4. 只有独立源、退市覆盖、来源版本深度和全部P0数据均通过后，才从候选生成正式`etf_total_return_prices.csv`并重跑Gate。
5. 正式全收益表通过后，再建立Long Hold V4的purged/embargo walk-forward与成本后绩效；在此之前不输出可投收益指标。

## 2026-07-20暂停增量

ETF历史跟踪基准工作包已经启动，但尚未完成：

- 新增全市场官方公告目录采集器：`strategy_lab/long_hold_v4/pit_etf_official_announcement_catalog_collector.py`。
- 新增证据契约：`data_catalog/long_hold_v4_etf_benchmark_history_contract.md`。
- 无关键词完整分页试验确认上交所`510050`有505条、巨潮`159901`有583条官方公告。
- 发现巨潮`totalpages`在583条记录时返回19，但实际需要请求1至20页；采集器现按`totalAnnouncement / 30`独立计算页数，缺页会失败关闭。
- 首个全市场100只续跑批次已经自然结束，没有残留采集进程。累计完成102/1,701只，目录17,642条，基准候选文件4,467条；其余1,599只保持`query_incomplete`。
- 当前资格为`PARTIAL_AUTHENTICATED_MASTER_TITLE_CATALOG`，所有资产仍是发现层，`historical_backtest_allowed=false`。
- 新模块定向测试7项通过；加入新模块后的完整回归、哈希审计和正式报告尚未执行。

下次重启后从同一不可变缓存续跑：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_official_announcement_catalog_collector --as-of 2026-07-19 --max-assets 100 --sleep-seconds 0 --max-consecutive-failures 5
```

每批会复用已完成资产。不得删除`data_raw/long_hold_v4/pit_history/raw_etf_official_announcement_catalog/queries/`，也不得把部分标题目录晋级为正式`etf_benchmark_history.csv`。

## 快速复核命令

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_source_qualification
python -X utf8 -m strategy_lab.long_hold_v4.pit_etf_total_return_candidate_validator --as-of 2026-07-17
python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-19
python -X utf8 -m strategy_lab.long_hold_v4.cli --as-of 2026-07-19
python -X utf8 -m pytest -q
python -B -X utf8 strategy_lab/agent_framework_check.py
```

## 禁止事项

- 不用当前最终快照回填历史`available_date`。
- 不把全生命周期当前最终内容通过解释为历史PIT通过。
- 不把123只查询完成或102只已识别事件解释为全体终止链完整；21只仍是证据不足，29只已识别链仍未完整。
- 不为了产生回测曲线降低PIT、退市覆盖或独立来源门槛。
- 不引用旧HIRSSM绩效作为Long Hold V4的样本外表现。
