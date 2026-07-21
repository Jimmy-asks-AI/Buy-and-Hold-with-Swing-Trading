# A-share Raw Daily Adjustment Guard V3.44

Updated: `2026-05-29T01:56:25`

## Status

- Raw daily data is complete for the configured date range, but it is not adjusted-return data.
- The adapter `strategy_lab/raw_daily_guard.py` must be used by downstream code.
- Adjusted-return and valuation research remains blocked.

## Evidence

- Discontinuity scan rows: `16988946`
- Flagged raw-return discontinuities: `11353`
- Assets with discontinuity flags: `4326`

## Still Needed

- `adj_factor`
- `stock_basic`
- `daily_basic`
- `index_weight`
- `index_dailybasic`
