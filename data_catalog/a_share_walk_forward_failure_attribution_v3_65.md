# A-share Walk-Forward Failure Attribution V3.65

## Dataset Role

V3.65 explains V3.64 narrow walk-forward proxy review failures by year, state, signal horizon, and train-to-OOS drift.

## Governance

- Return basis: price-index proxy diagnostics inherited from V3.64.
- Portfolio harness: not produced.
- Default model change: not allowed.
- Official dividend-inclusive evidence: not produced.

## Configuration

- Broad failure pass-rate threshold: `0.1`
- Strong proxy pass-rate threshold: `0.5`
- Long-horizon retirement set: `20,60`

## Produced Shape

- Year rows: `17`
- Signal-horizon rows: `12`
- Broad failure years: `12`
- Retire rows: `6`