# A-share Daily-Only Permission Data V3.38

Updated: `2026-05-29T01:47:32`

## Scope

- Provider: `tushare`
- Allowed endpoint: `daily`
- Data type: stock non-adjusted daily OHLCV
- Configured date range: `20000101` to `20260528`
- Acquired/cached dates with rows: `6395`

## Permission Profile

| Endpoint | Status | Research Use | Limitation |
|---|---|---|---|
| `daily` | `allowed` | raw_price_volume_only | non-adjusted OHLCV; no valuation, no adjusted return, no historical universe lifecycle |
| `stock_basic` | `permission_blocked` | all-status universe, delisting lifecycle, listing dates | derive traded universe from daily rows by trade_date; no listing lifecycle yet |
| `daily_basic` | `permission_blocked` | valuation, turnover, market cap, PE/PB factors | defer valuation factors; use price/volume-only factors |
| `adj_factor` | `permission_blocked` | 复权收益, split/dividend-adjusted prices | use raw close only; do not compute total-return or qfq factors |
| `index_weight` | `permission_blocked` | historical index constituents and PIT index weights | defer index constituent PIT research |
| `index_dailybasic` | `permission_blocked` | index valuation and index-level PE/PB | defer index valuation repair |

## Model Use Constraints

| Research Area | Allowed Now | Constraint |
|---|---:|---|
| `price_volume_factor` | `True` | raw close only; use forward raw close returns with explicit warning |
| `adjusted_return_factor` | `False` | requires adj_factor or trusted adjusted price history |
| `valuation_factor` | `False` | requires daily_basic or historical valuation/market-cap source |
| `dividend_total_return` | `False` | requires dividends and adjustment history |
| `index_constituent_pit` | `False` | requires historical index constituents and weights |
| `industry_rotation_pit` | `False` | requires historical industry classification or industry index series |

## Governance

- Do not use this dataset as adjusted-return data.
- Do not infer valuation, market cap, dividend yield, or index membership from this dataset.
- Before full backtests, upgrade data layer or explicitly constrain the model to raw-price/volume research.
