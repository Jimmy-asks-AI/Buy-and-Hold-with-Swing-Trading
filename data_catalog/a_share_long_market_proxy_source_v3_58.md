# A-share Long MARKET Price Proxy Source V3.58

## Dataset Role

This dataset is a governed long-history price-index proxy for MARKET-level diagnostics.

## Source

- Primary index: `000985`
- Rows: `5191`
- Date range: `20050104` to `20260522`
- Proxy file: `data_raw/market_labels/market_price_index_proxy_000985.csv`

## Governance

- Allowed use: price-index proxy labels, state diagnostics, and non-official validation experiments after an explicit guarded importer exists.
- Forbidden use: official total-return labels, dividend-inclusive performance claims, direct model promotion, or portfolio backtest promotion.
- `data_raw/market_labels/market_total_return_index.csv` is intentionally not written by V3.58.
