# A-share Guarded Feature-Label Validation V3.71

## Dataset Role

V3.71 joins the V3.70 combined feature registry to V3.59 price-index proxy labels for guarded feature diagnostics.

## Governance

- Label source is non-official price-index proxy only.
- No official total-return validation, portfolio backtest, or model promotion is produced.
- Macro stale rows are excluded for macro features when configured.

## Produced Shape

- Validation universe rows: `20290`
- Tested numeric features: `202`
- Feature-horizon summary rows: `808`
- Proxy pass rows for stricter review: `87`
- Horizons: `1,5,20,60`