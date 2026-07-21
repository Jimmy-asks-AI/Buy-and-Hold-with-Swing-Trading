# ETF跟踪基准历史契约

## 目标

`etf_benchmark_history.csv`描述每只ETF在不同生效区间采用的跟踪基准或业绩比较基准。它是事件版本表，不是当前基金档案快照。货币市场ETF等非指数基金不得被强行赋予指数代码。

正式表至少包含：

`asset, reference_type, index_code, index_name, performance_benchmark_text, effective_from, effective_to, announcement_date, available_date, data_source, source_vintage`

`reference_type`限定为：

- `tracked_index`：基金合同或招募说明书明确采用被动指数化投资并跟踪标的指数。
- `enhanced_index`：官方投资目标明确在跟踪标的指数基础上进行主动增强并争取超额收益；不得与纯复制型指数ETF合并评估。
- `commodity_spot_reference`：黄金等商品ETF明确跟踪交易所现货合约、集中定价合约或现货价格，不得强行映射为证券指数代码。
- `non_index_reference`：官方文件明确基金不跟踪指数，只使用文字型业绩比较基准或不设指数基准。
- `unknown`：正文、条款或基金类型尚未确认。

`non_index_reference`的`index_code/index_name`必须为空，不能从基金名称、当前网站快照或风险披露模板推断。

建议同时保留：

`event_id, benchmark_sequence, source_pdf_path, source_pdf_sha256, validation_status, is_initial_mapping, is_change_event`

## 证据层级

1. 交易所或巨潮完整公告目录只证明标题查询范围完整，不证明正文事实。
2. 上市交易公告书、初始基金合同或初始招募说明书用于确认第一段跟踪基准，同时用于识别货币市场ETF等非指数基金。
3. 标的指数变更公告、基金合同修订、持有人大会决议及其生效公告用于确认后续区间。
4. 指数简称或名称变化必须区分“同一指数更名”和“更换为另一指数”。
5. 正式晋级必须由独立验证器复核基金类型、指数代码、名称、生效日、公告日和前后区间连续性。
6. “标的指数”“货币市场基金”等词在风险披露、定义或引用材料中出现，不能单独作为分类证据；必须定位投资目标、投资策略或业绩比较基准条款。

## 三态

- `benchmark_history_identified`：起始基准和全部后续区间均有正式证据，区间无重叠且覆盖生命周期。
- `official_non_index_classification`：官方正文已确认该基金不跟踪指数；若存在业绩比较基准，还必须闭合其历史变化。
- `official_no_benchmark_change`：起始基准已确认，完整公告目录和相关合同修订正文均已检查，没有发现更换标的指数；标题零结果本身不能触发该状态。
- `evidence_insufficient`：查询、正文、指数身份、生效日或区间链任一项未闭合。

发现阶段所有资产都保持`evidence_insufficient`。只有独立验证后的正式事件才能改变状态。

## PIT边界

- `available_date`不得早于官方公告日；日频策略保守从公告次日使用。
- `effective_from`早于`available_date`时，不得把事件回填到公告前的交易决策。
- 当前跟踪指数不能写回成立日。
- 当前最终公告目录可以用于证据发现，不能自动取得历史回测资格。
- 没有完整基准链的资产在历史截面中必须阻断，不能静默沿用上一段或当前映射。

## 发现产物

- 完整标题目录：`data_raw/long_hold_v4/pit_history/observations/etf_official_announcement_catalog.csv`
- 基准候选文件：`data_raw/long_hold_v4/pit_history/observations/etf_benchmark_document_candidates.csv`
- 三态覆盖登记：`data_raw/long_hold_v4/pit_history/observations/etf_benchmark_discovery_coverage_registry.csv`
- 人工复核队列：`data_catalog/long_hold_v4_etf_benchmark_review_queue.csv`

这些文件均为观察层，固定`historical_backtest_allowed=false`、`model_promotion_allowed=false`。

## 正文选择层

正文采集不按标题候选全量盲抓，使用固定策略生成可审计队列：

1. 每只ETF选择一份初始主证据，优先完整初始基金合同，其次初始招募说明书、基金合同摘要；只有缺少初始法律文件时才用上市交易公告书兜底。
2. “基金合同生效公告”不作为初始基准正文。
3. 缺少初始文件的资产可用最早更新招募说明书、基金合同或产品资料概要回退，但必须标记`fallback_update_candidate`，不能据此确认初始基准。
4. 每只有上市交易公告书的ETF另保留一份最早上市公告书作为上市时点旁证；它不能替代主证据中的投资目标、标的指数和业绩比较基准条款。
5. 每只有初始招募说明书的ETF另保留一份最早初始招募说明书，用于补充指数代码、投资策略，并为扫描版或失效合同提供独立旁证。
6. 所有标题路由的基准变更文件、基金合同修订和持有人大会决议都进入采集队列。
7. 选择成功只表示文件待采集，所有资产仍为`evidence_insufficient`，也不得声明`official_no_benchmark_change`。

对应产物：

- 正文选择表：`data_raw/long_hold_v4/pit_history/observations/etf_benchmark_document_selection.csv`
- 选择覆盖登记：`data_raw/long_hold_v4/pit_history/observations/etf_benchmark_document_selection_coverage_registry.csv`
- 正文采集队列：`data_catalog/long_hold_v4_etf_benchmark_document_collection_queue.csv`
