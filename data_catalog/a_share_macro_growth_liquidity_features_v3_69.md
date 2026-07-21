# A-share Macro Growth-Liquidity Feature Layer V3.69

## Dataset Role

V3.69 converts macro PIT rows into a daily as-of feature panel for growth, liquidity, inflation, and external-pressure diagnostics.

## Governance

- Every macro value is joined by `available_date <= trade_date`.
- Macro vintage limitations are retained as dataset metadata.
- Portfolio outputs and default model promotion are blocked.
- Feature timing is after-close for next trade-date research.

## Produced Shape

- Feature rows: `6395`
- Date range: `20000104` to `20260528`
- History-sufficient rows: `5116`
- Trailing window: `252`