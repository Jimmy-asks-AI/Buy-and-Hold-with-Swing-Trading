# A-share PIT Data Acquisition Contract V3.35

Updated: `2026-05-27T20:25:59`

This catalog note defines the next production-grade data contract. It is not evidence that the data has been acquired.

## Provider Readiness

| Provider | Execute Ready | Notes |
|---|---:|---|
| Tushare | `False` | Needs SDK plus token. |
| JoinQuant | `False` | Needs jqdatasdk plus username/password. |

## Contract

| Priority | Dataset | Provider | Endpoint | PIT Gate |
|---:|---|---|---|---|
| 1 | `historical_index_weights` | `tushare` | `index_weight` | date+index+asset unique; no current snapshot backfill; weights sum near 100 by effective_date; available_date <= asof_date |
| 1 | `historical_index_membership` | `joinquant` | `get_index_stocks` | date+index+asset unique; query_date recorded; membership not used as weight history without separate weights |
| 2 | `stock_universe_all_status` | `tushare` | `stock_basic` | include L,D,P statuses; no current-only stock list; asset unique by status snapshot |
| 2 | `stock_daily_raw` | `tushare` | `daily` | date+asset unique; positive OHLC when traded; no silent forward fill; raw_close preserved |
| 2 | `stock_daily_basic` | `tushare` | `daily_basic` | date+asset unique; nonnegative market caps; no merge before available_date |
| 2 | `stock_adj_factor` | `tushare` | `adj_factor` | date+asset unique; adj_factor positive; raw and adjusted price columns never overwritten |
| 2 | `stock_tradeability_flags` | `tushare` | `suspend_d|stk_limit|namechange` | date+asset unique; flags explicit; no hidden forward fill across missing records |
| 3 | `stock_financial_pit` | `tushare` | `fina_indicator|income|balancesheet|cashflow` | available_date mandatory; restatement policy explicit; no current restated values backfilled into old asof dates |
| 4 | `historical_industry_classification` | `joinquant` | `get_industry` | one active classification per asset-level-asof; no latest classification backfill |

## Execution Boundary

- Current snapshots remain disallowed for historical backtests.
- Dry-run rows are `planned_not_acquired` and cannot be used by factor researchers.
- Actual acquired tables must pass duplicate, missingness, PIT, adjustment, and lifecycle checks before promotion to research inputs.
- Planned harvest rows: `154`.
