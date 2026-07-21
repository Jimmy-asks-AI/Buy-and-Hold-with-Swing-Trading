# ETF退市终止事件全量盘点报告

审查日期：2026-07-19  
对象：当前已认证ETF主表中的123只退市ETF  
结论：102只取得可记账的正式终止现金事件，21只证据不足；没有资产被认定为“确认无价值转移”。

## 盘点结果

| 项目 | 结果 |
|---|---:|
| 官方查询完成 | 123/123只 |
| 官方公告记录 | 831条 |
| 入选原始PDF | 456份 |
| 原生文本 / OCR侧车 | 454 / 2份 |
| 正式现金事件 | 118笔，覆盖102只ETF |
| 多轮分配 | 15只，单只最多3轮 |
| 完整终止链 | 73只 |
| 已有现金事件但链未完整 | 29只 |
| 证据不足 | 21只 |
| 验证失败候选 | 6笔 |

118笔事件中，88笔直接从官方正文复核每份金额，30笔由官方可分配总额除以原生清算文件中的可信份额复算。每份金额范围为`0.0004656988410043`至`117.725`元。73笔链尾事件有原份额灭失证据；中间分配不会提前注销持仓。

6笔失败候选继续隔离：`159924`缺少可独立复核的金额；`159969`、`159987`、`159990`、`159999`缺少可信份额分母；`512110`正文支付年份为`20162`，验证器不猜测正确年份。其余证据不足资产没有取得可晋级的现金或继受份额机制。`159917`、`159927`、`510820`等继受份额线索也没有在证据不足时被转换成收益。

## PIT与事件链

- 发现层不读取正式事件表。纯发现结果仍是123只全部`evidence_insufficient`，避免正式结果反向污染搜索覆盖。
- 公告保存`published_at`、`available_at`和`available_trade_date`；日频策略最早从公告后的下一日使用，现金最早在实际发放日与可得日两者较晚者入账。
- 同一ETF的多次分配按`event_id + distribution_sequence + holder_scope + pay_date + source_text_sha256`保留，不覆盖成一笔。
- OCR只写独立侧车。2份OCR文本标记为`ocr_derived_unvalidated`，没有改写原生文档元数据，也没有单独提供晋级字段。
- 最终分配和“仍可能追加分配”由独立验证器从同一份哈希认证正文重算；两者冲突数为0。

## 总回报处理

全生命周期观察重新应用118笔终止现金流，覆盖1,701只ETF和1,466,663条真实价格，0只隔离。终止分配只进入现金账：

- 不把清算款改成收盘价；
- 不在公告日、发放日或终止上市日生成OHLC；
- 有供应商现金标记时只移除精确匹配的一行，防止重复复权；
- 没有供应商标记时生成正式现金账事件，不生成市场行。

独立候选验证确认118笔事件全部进入现金账、价格键集合与原行情完全一致、事件链失败0。`511210`的112.79元/份只是其中一笔，仍按2018-01-23实际发放日入账。

## 资格边界

正式事件行允许按各自PIT日期进行对应现金会计处理，`model_promotion_allowed=false`。当前结算仍有21只`evidence_insufficient`，且退市母集只相对于当前已认证ETF主表完整，因此`scope_complete=false`。

正式事件表不能单独晋级ETF全收益价格。当前行情和净值仍是2026-07-19取得的最终历史快照，版本深度为0；`etf_total_return_prices.csv`继续不生成，历史回测资格继续阻断。

## 权威产物

- 纯发现：`data_raw/long_hold_v4/manifests/etf_terminal_event_universe_collector_latest.json`
- 文档合并：`data_raw/long_hold_v4/manifests/etf_terminal_event_document_merge_latest.json`
- 独立验证：`outputs/long_hold_v4/pit_validation/etf_terminal_event_v2/run_manifest.json`
- 正式事件：`data_raw/long_hold_v4/pit_history/etf_terminal_cash_events.csv`
- 正式清单：`data_raw/long_hold_v4/manifests/pit_etf_terminal_cash_events_builder_latest.json`
- 下游结算：`data_raw/long_hold_v4/manifests/etf_terminal_event_coverage_settlement_latest.json`
