# A-share State-Stratified Proxy Validation V3.60

## Dataset Role

V3.60 evaluates V3.50 MARKET signals against V3.59 non-official price-index proxy labels.

## Governance

- Return basis: `price_index_return` only.
- Official total-return evidence: false.
- Portfolio backtest: not produced.
- Default model promotion: not allowed.

## Inputs

- Proxy labels: `data_raw/market_labels/market_price_proxy_forward_labels_000985.csv`

## Produced Shape

- Signal-horizon summary rows: `44`
- Proxy-positive observation rows: `12`
