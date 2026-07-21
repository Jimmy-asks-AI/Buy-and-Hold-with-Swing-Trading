# Quant Research RACI Matrix

Use this matrix to keep agent responsibilities separate. `A` means accountable and final owner; `R` means responsible for producing work; `C` means consulted through artifacts; `I` means informed through final reports.

| Work Item | Chief | Data | Factor | Regime | Portfolio | Validation | Cost | Reporter | Code Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Version objective and task split | A/R | C | C | C | C | C | C | I | C |
| Data source approval | I | A/R | C | C | I | C | C | I | C |
| Point-in-time and availability audit | I | A/R | C | C | I | C | I | I | C |
| Factor hypothesis and definition | C | C | A/R | C | I | C | I | I | I |
| Regime and timing signal definition | C | C | C | A/R | I | C | I | I | I |
| Portfolio construction and constraints | C | I | C | C | A/R | C | C | I | C |
| Cost, turnover, and capacity analysis | C | C | I | I | C | C | A/R | I | C |
| Leakage and overfit validation | C | C | C | C | C | A/R | C | I | C |
| Reproducibility and smoke testing | C | I | I | I | I | C | I | I | A/R |
| Version report and dashboard | C | I | I | I | I | C | C | A/R | C |
| Promotion, rejection, or observation decision | A/R | I | I | I | I | C | C | I | C |

## Research Assistant RACI Extension

These roles extend the framework for object-level analysis. They do not bypass the model-promotion path above.

| Work Item | Chief | Data | Technical | Fundamental | Synthesizer | Validation | Reporter | Code Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Research object intake | A/R | C | C | C | I | I | I | C |
| Price-action technical view | I | C | A/R | I | C | I | I | C |
| Fundamental and valuation view | I | C | I | A/R | C | I | I | C |
| PIT data-gap registration | I | A/R | C | R | I | C | I | C |
| Cross-evidence synthesis | A | I | C | C | R | C | I | C |
| Research report rendering | C | I | I | I | C | I | A/R | C |
| Research-output consistency check | C | C | C | C | C | A/R | I | C |

## Decision Authority

- Only `chief_quant_orchestrator` can promote a model, factor, data source, or overlay to default.
- `backtest_validation_auditor` can block promotion but cannot promote.
- `backtest_validation_auditor` should not be the primary designer of portfolio construction rules; if it implements a harness for speed, the manifest must mark the construction assumption as predeclared and the next review must check role-overlap risk.
- `code_quality_engineer` can block promotion for reproducibility or integrity failures but cannot validate alpha.
- `portfolio_risk_engineer` can block invalid weights but cannot approve model validity.
- `research_reporter` records decisions but cannot create them.
- `technical_market_analyst`, `fundamental_equity_analyst`, and `investment_view_synthesizer` produce research views only; none can promote a model, factor, or order rule.

## Conflict Resolution

- If validation and portfolio results conflict, the candidate stays `observation` until the orchestrator records a decision.
- If a data timing issue appears after model testing, downstream evidence is downgraded to `blocked` or `research_only`.
- If full-sample diagnostics and nested validation disagree, nested validation controls promotion.
