# A-share Lag-Safe Proxy Validation V3.63

## Dataset Role

V3.63 validates V3.62 lag-safe non-price MARKET candidate signals against non-official price-index proxy labels.

## Governance

- Return basis: `price_index_return` proxy evidence only.
- Official total-return evidence: false.
- Portfolio backtest: not produced.
- Default model promotion: not allowed.

## Inputs

- Repaired signals: `outputs/agent_runs/v3_62/lag_safe_signal_repair/repaired_signal_panel.csv`
- Proxy labels: `data_raw/market_labels/market_price_proxy_forward_labels_000985.csv`

## Produced Shape

- Signal-horizon rows: `24`
- Proxy-positive rows: `12`
- Walk-forward proxy-review candidates: `12`
