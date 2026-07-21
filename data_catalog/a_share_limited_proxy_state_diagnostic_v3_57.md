# A-share Limited Proxy State Diagnostic V3.57

## Dataset Role

V3.57 derives limited-window proxy states from the V3.56 JoinQuant adjusted proxy and joins them to limited MARKET signal labels.

## Governance

- Scope: limited-window smoke diagnostic only.
- Allowed use: schema, availability, state coverage, and join diagnostics.
- Forbidden use: IC, hit-rate, backtest, NAV, Sharpe, drawdown, or model promotion.
- Official source file remains absent unless separately acquired and validated.

## Produced Shapes

- proxy_state_panel rows: `248`
- joined_contract_panel rows: `8272`