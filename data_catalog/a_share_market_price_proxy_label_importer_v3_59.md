# A-share MARKET Price-Proxy Label Importer V3.59

## Dataset Role

V3.59 creates MARKET-level forward labels from the V3.58 long price-index proxy.

## Governance

- Return basis: `price_index_return` only.
- Official total-return status: false.
- Allowed use: proxy-labelled diagnostics and future guarded state validation.
- Forbidden use: official total-return validation, dividend-inclusive performance claims, portfolio promotion, or default model promotion.

## Produced Shape

- Label rows: `20678`
- Label date range: `20050104` to `20260521`
- Canonical label file: `data_raw/market_labels/market_price_proxy_forward_labels_000985.csv`
