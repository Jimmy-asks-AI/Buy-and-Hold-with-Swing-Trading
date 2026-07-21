# A-share Proxy Artifact Audit V3.61

## Dataset Role

V3.61 audits V3.60 proxy-positive MARKET signal observations for likely artifacts.

## Governance

- Return basis: `price_index_return` proxy evidence only.
- Official total-return evidence: false.
- Portfolio backtest: not produced.
- Default model promotion: not allowed.

## Inputs

- Joined panel: `outputs/agent_runs/v3_60/state_stratified_proxy_validation/joined_proxy_validation_panel.csv`
- Signal summary: `outputs/agent_runs/v3_60/state_stratified_proxy_validation/signal_proxy_validation_summary.csv`
- Negative control: `outputs/agent_runs/v3_60/state_stratified_proxy_validation/negative_control_summary.csv`
- Market proxy source: `data_raw/market_labels/market_price_index_proxy_000985.csv`

## Produced Shape

- Candidate decisions: `12`
- Plausible for stricter proxy review: `0`
- Artifact-risk blocked: `11`
