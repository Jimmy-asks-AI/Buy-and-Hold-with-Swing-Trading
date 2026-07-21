# HIRSSM V3.86-V3.92 Quality Review

This review was triggered because the first V3.86-V3.92 pass was too shallow.

## Findings And Fixes

| Version | Severity | Finding | Fix | Status |
|---|---:|---|---|---:|
| V3.86 | high | research assistant agents were referenced before formal roster registration | added AGENT.md specs, roster entries, RACI notes, I/O contract boundary, and assigned-agent framework check | fixed |
| V3.87 | medium | sample research universe did not materialize required asof_date and used primary_horizon while schema required horizon | added asof_date, renamed sample output to horizon, expanded required schema fields, added research_object_contract_check.csv, and split valuation coverage sources | fixed |
| V3.88 | medium | technical method note was too thin for reproducibility, volatility was not structured as evidence, and V3.87 input contract was not checked | added formula overlays, volatility state, confidence caps, panel component fields, and technical_input_contract_check.csv | fixed |
| V3.89 | medium | fundamental scores lacked a formula/recompute audit, 000985 missing PB was not recorded, and macro PIT boundaries were implicit | added formula spec, score reconciliation, PB gap rows, current-snapshot backtest bans, and macro PIT checks | fixed |
| V3.90 | high | technical/fundamental disagreement could still emit directional views and synthesized outputs did not explicitly block trade/backtest use | added hard-conflict neutral score caps, research-only output boundaries, synthesis formula/reconciliation files, and input contract checks | fixed |
| V3.91 | medium | report/dashboard omitted V3.90 research-only boundaries, hard-conflict state, and macro PIT context from the visible layer | added report/dashboard content checks, visible no-order/no-backtest boundaries, conflict tags, macro PIT table, and dashboard visual static checks | fixed |
| V3.92 | high | sample end-to-end run only checked broad pipeline existence and did not aggregate upstream gates or research-only boundaries | added cross-version gate audit, sample boundary audit, and explicit research-only decision report | fixed |
| Governance | medium | new next_handoff roster check initially treated older narrative handoff text as active errors | limited strict next_handoff roster enforcement to V3.86+ tasks and parsed version numbers instead of string-ordering versions | fixed |

## Boundary

These outputs are research-assistant capabilities only. They do not claim validated alpha, portfolio performance, or trade instructions.
