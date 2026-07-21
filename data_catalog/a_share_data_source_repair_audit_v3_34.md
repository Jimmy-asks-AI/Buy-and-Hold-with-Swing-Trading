# A-share Data Source Repair Audit V3.34

Updated: `2026-05-27T20:03:57`

This catalog note records data-steward decisions after V3.33 found no implementation-ready independent signal.

## Point-in-time Status

| Dataset | Status | Strict PIT Backtest | Restriction |
|---|---:|---:|---|
| `csindex_daily_index_prices` | `approved` | `True` | signal_validation,index_level_backtest |
| `sw_industry_daily_prices` | `approved` | `True` | signal_validation,index_level_backtest |
| `csindex_constituents_current` | `research_only` | `False` | historical_constituent_or_weight_backtest |
| `csindex_weights_latest` | `research_only` | `False` | historical_constituent_or_weight_backtest |
| `sw_industry_components_current` | `research_only` | `False` | historical_constituent_or_weight_backtest |
| `stock_daily_qfq_sample` | `research_only` | `False` | production_broad_stock_factor_backtest,raw_trade_execution_simulation |
| `stock_financial_indicator_sample` | `blocked` | `False` | historical_fundamental_factor_backtest_without_announcement_date |
| `macro_pit_panel` | `approved` | `True` | signal_validation,index_level_backtest |

## Required Repairs

| Priority | Repair | Required Fields | Validation Gates |
|---:|---|---|---|
| 1 | `historical_index_constituents_weights` | index_code,asset,weight,effective_date,available_date,source_vintage,data_source | no current snapshot backfill; date+index+asset unique; weights sum near 100 by effective_date; available_date <= model_asof_date |
| 2 | `broad_stock_daily_raw_and_adjusted` | date,asset,open,high,low,close,volume,amount,adj_factor,raw_close,adjusted_close,is_paused,is_st,listed_days,delist_date | include delisted stocks; raw and adjusted prices separated; no forward-filled halted bars without flag; asset-date unique |
| 3 | `financial_indicator_point_in_time` | asset,report_period,statement_type,metric,value,announcement_date,available_date,data_source,revision_flag | available_date mandatory; no restated current values backfilled into old asof dates; duplicate revision policy explicit |
| 4 | `historical_industry_classification` | asset,industry_code,industry_name,level,effective_date,end_date,available_date,classification_standard | one active classification per asset-level-asof; no latest classification backfill; delisted securities retained |
| 5 | `macro_release_calendar_vintage_upgrade` | series_id,period_date,value,release_date,available_date,revision_timestamp,vintage_id | release lag explicit; revised values separated from initial vintage when available |
