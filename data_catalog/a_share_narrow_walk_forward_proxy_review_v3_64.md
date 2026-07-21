# A-share Narrow Walk-Forward Proxy Review V3.64

## Dataset Role

V3.64 reviews V3.63 survivor MARKET signal-horizon rows with rolling train/OOS proxy diagnostics.

## Governance

- Return basis: `price_index_return` proxy evidence only.
- Official total-return evidence: false.
- Portfolio backtest: not produced.
- Default model promotion: not allowed.

## Configuration

- Train years: `5`
- Test years: `1`

## Produced Shape

- Summary rows: `12`
- Rows passing narrow proxy walk-forward: `0`