# A-share Industry Breadth And Dispersion Feature Layer V3.68

## Dataset Role

V3.68 converts SW industry index daily history into industry breadth, dispersion, liquidity, and rotation-persistence features.

## Governance

- Uses industry index price and amount history only.
- Current component snapshots and latest weights are blocked.
- Portfolio outputs and default model promotion are blocked.
- Feature timing is after-close for next trade-date research.

## Produced Shape

- Feature rows: `6374`
- Date range: `20000104` to `20260522`
- History-sufficient rows: `6314`
- Industry level: `first`
- Trailing window: `252`