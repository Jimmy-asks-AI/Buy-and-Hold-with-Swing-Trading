# A-share PIT Data Pilot V3.36

Updated: `2026-05-28T21:02:51`

This catalog entry records the controlled pilot-readiness state. It is not approval for factor research.

## Execution Flags

- `execute`: `True`
- `allow_network`: `True`

## Provider Readiness

- Tushare ready: `True`
- JoinQuant ready: `False`

## Dataset Registry

| Dataset | Provider | Status | Research Use |
|---|---|---:|---|
| `stock_universe_all_status` | `tushare` | `error` | blocked |
| `historical_index_weights` | `tushare` | `error` | blocked |
| `stock_daily_raw` | `tushare` | `acquired` | pilot_research_only |
| `stock_daily_basic` | `tushare` | `error` | blocked |
| `stock_adj_factor` | `tushare` | `error` | blocked |
| `historical_index_membership` | `joinquant` | `not_acquired` | blocked |

## PIT Rule

Even when a pilot dataset is acquired, it remains pilot research-only. Full historical use requires broader coverage, delisted-name coverage, duplicate checks, available-date checks, raw/adjusted separation checks, and downstream data-steward approval.

PIT rows: `6`.
