# A-share Daily-Only Data Acceptance V3.45.1

Updated: `2026-05-29T02:10:01`

## Acceptance

- Acceptance passed after quarantine: `True`
- Raw rows: `16988946`
- Source OHLC anomalies: `2`
- Quarantined OHLC rows: `2`
- Accepted processed rows: `16988944`
- Data range: `20000101` to `20260528`

## Boundary

- Data is raw daily OHLCV only.
- The accepted processed scope excludes row-level OHLC anomalies.
- V3.44 guard blocks adjusted-return and valuation misuse.
- Missing interfaces remain unresolved.

## Still Needed

- `adj_factor`
- `stock_basic`
- `daily_basic`
- `index_weight`
- `index_dailybasic`
