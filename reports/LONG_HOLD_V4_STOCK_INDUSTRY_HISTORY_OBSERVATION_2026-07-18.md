# Long Hold V4 股票行业历史观察链审查

审查日期：2026-07-18  
资格结论：`OBSERVATION_ONLY`  
使用边界：可以检查分类区间、证券代码迁移和数据契约；不得用于正式历史回测、因子检验或模型晋级。

## 1. 来源与证据

- 官方下载入口：[申万宏源研究行业分类下载中心](https://www.swsresearch.com/institute_sw/allIndex/downloadCenter/industryType)
- 工作簿标称地址：[StockClassifyUse_stock.xls](https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls)
- 本地原始文件：`data_raw/long_hold_v4/pit_history/raw_sws_stock_industry/2026-07-12/StockClassifyUse_stock.xls`
- 文件大小：1,161,216 字节
- SHA-256：`8da3f757895a6d77a19d8f690cca3b3022fa0da56533cdb4769f5939b4bc49d2`

审查日访问官方页面时，下载接口出现证书失效或 502 响应，未能重新取得带可验证 TLS 的副本。当前文件来自另一个本地项目的缓存，原采集脚本关闭了 TLS 校验。文件哈希和内容可复核，但传输来源不能据此升级为可信历史数据源。

## 2. 构建方法

构建器读取股票代码、计入日期、行业代码和更新日期，并执行以下约束：

1. `available_date` 取计入日期、更新日后一个自然日和证券上市日三者中的最晚值，避免同日提前使用。
2. 同一证券的行业记录展开为左闭右开的有效区间，下一条分类生效时关闭上一条区间。
3. 区间裁剪到证券上市、退市生命周期，不保留生命周期外记录。
4. `300114 -> 302132`、`600849 -> 601607` 按官方代码迁移登记重分配；迁移前归旧代码，迁移日给新代码建立可用快照。
5. `available_date < effective_to` 的记录标为 `pit_actionable=true`；其余记录保留为回溯信息，不允许驱动当时决策。

代码：`strategy_lab/long_hold_v4/pit_stock_industry_history_builder.py`  
输出：`data_raw/long_hold_v4/pit_history/stock_industry_history.csv`  
清单：`data_raw/long_hold_v4/manifests/stock_industry_builder_latest.json`

## 3. 结果

| 项目 | 结果 |
|---|---:|
| 输出行数 | 12,454 |
| 覆盖证券 | 5,537 / 5,537 |
| 退市证券覆盖 | 338 / 338 |
| 分类代码数 | 553 |
| 有效区间 | 1990-12-01 至 2026-07-10 |
| 可得日期区间 | 2015-10-27 至 2026-07-10 |
| 区间内可操作记录 | 8,915 |
| 回溯但不可操作记录 | 3,539 |
| 可操作比例 | 71.5834% |
| 前序代码重分配 | 1 行 |
| 新代码承接快照 | 1 行 |

完整覆盖不等于历史可用。3,539 行的更新时间已经晚于该行业区间结束日，证明当前工作簿含有事后整理；即使其余 8,915 行时间关系合理，也无法证明审查日前每个历史时点实际收到过同一版本。

## 4. 门禁结论

Gate E2 对该表的 schema、主键、有效区间、资产覆盖、退市股覆盖、代码迁移、可得日和哈希检查均通过；唯一失败项是：

`lineage_historical_use_approved: historical_backtest_allowed=False`

这项失败是主动保护，不是构建器故障。当前状态保持：

- `qualification_status=OBSERVATION_ONLY`
- `historical_backtest_allowed=false`
- `model_promotion_allowed=false`

## 5. 升级条件

要升级为正式历史行业数据，至少需要：

1. 获得按发布日期保存的官方历史工作簿或授权 PIT 行业库。
2. 每个版本保留下载时间、来源 URL、TLS 验证结果和文件哈希。
3. 用版本发布日期而非当前行更新时间构造 `available_date`。
4. 覆盖退市股、行业变更、分类标准切换和全部已知代码迁移。
5. 对随机历史截面与独立供应商做日期级成员核对。

在这些条件满足前，该表只能帮助发现数据问题和设计行业契约，不能用于行业中性化、行业轮动、行业留出验证或历史选股。
