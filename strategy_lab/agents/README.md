# Subagent Quant Research Framework

This directory defines the long-lived role memory for the quant research subagents. The default operating rule is context isolation: an agent receives only its own `AGENT.md`, the current task brief, and explicitly allowed artifacts. Agents share conclusions through files, not through the full conversation history.

## Baseline

- Current governance baseline: `HIRSSM V3.10 Clean Rank-Vol Core`
- Baseline script: `strategy_lab/hirssm_v3_10_clean_baseline.py`
- Baseline outputs: `outputs/hirssm_v3_10_clean_baseline`
- Candidate validation harness: `strategy_lab/hirssm_v3_11_nested_candidate_harness.py`
- Default comparison costs: `5bps`, `10bps`, `20bps`, `30bps`
- Promotion note: V3.10 is a governance control baseline, not an alpha target; V3.11 is the required candidate gate.

## Canonical Governance Files

- `AGENT_WORKFLOW.md`: phase gates and promotion states.
- `RACI_MATRIX.md`: role accountability and decision boundaries.
- `AGENT_IO_CONTRACT.md`: required task fields, artifact names, status semantics, and promotion boundary.

## Context Policy

- Spawn subagents with isolated context by default. Do not fork the full parent conversation unless the task explicitly requires it.
- Each task prompt must include only: the agent spec, task objective, allowed read paths, allowed write paths, expected outputs, baseline, and deadline.
- Agents do not read other agents' scratch work unless it is explicitly listed as an allowed input artifact.
- Cross-agent communication happens through structured artifacts: `agent_run_manifest.json`, `agent_report.md`, `metrics.csv`, decision logs, and review reports.
- The orchestrator is the only role allowed to merge a candidate into the default model path.

## Agent Roster

- `chief_quant_orchestrator`: version owner, decision authority, scope control.
- `data_steward`: data source, point-in-time, coverage, and quality owner.
- `factor_researcher`: factor hypothesis, evidence, and factor card owner.
- `regime_timing_researcher`: market state, style switch, industry rotation, and macro timing owner.
- `technical_market_analyst`: object-level trend, momentum, drawdown, volatility, and liquidity view owner.
- `fundamental_equity_analyst`: object-level valuation, dividend, quality, growth, and PIT data-gap view owner.
- `investment_view_synthesizer`: combines finalized technical and fundamental evidence into research-only views.
- `portfolio_risk_engineer`: portfolio construction, risk budget, exposure, and drawdown controls owner.
- `backtest_validation_auditor`: independent leakage, overfit, and robustness auditor.
- `execution_cost_analyst`: cost, turnover, capacity, and implementation friction owner.
- `research_reporter`: readable research reports, dashboards, and changelogs owner.
- `code_quality_engineer`: smoke tests, output integrity, and repository quality owner.

## Research Assistant Extension

The research assistant roles are analysis roles, not strategy-promotion roles.

- `technical_market_analyst` can say price evidence is bullish, neutral, bearish, or blocked, but cannot approve a trade.
- `fundamental_equity_analyst` can score valuation and fundamentals only when source timing is explicit; current snapshots are confidence-capped and research-only.
- `investment_view_synthesizer` can combine finalized views, but must preserve disagreement, confidence caps, data gaps, and invalidation conditions.

## Promotion Rule

A model, factor, data source, or overlay may move from experiment to default only when:

- the responsible agent has produced the required artifacts;
- the validation auditor has reviewed leakage, overfit, and robustness risks;
- the code quality engineer has passed smoke checks;
- the reporter has recorded the change;
- the orchestrator has written the decision into `reports/MODEL_DECISION_LOG.md`.

## Default Work Order

1. `chief_quant_orchestrator` creates a task brief and assigns one accountable owner.
2. `data_steward` approves any new historical data before it is used.
3. `technical_market_analyst`, `fundamental_equity_analyst`, `factor_researcher`, or `regime_timing_researcher` creates observation evidence.
4. `investment_view_synthesizer` combines research-only views when the task is analysis-oriented.
5. `portfolio_risk_engineer` converts approved signals into constrained weights only for model tasks.
6. `execution_cost_analyst` tests net performance, turnover, and capacity for model tasks.
7. `backtest_validation_auditor` runs leakage, nested validation, PBO/DSR, robustness, or research-output consistency checks.
8. `code_quality_engineer` validates scripts, outputs, schemas, and manifests.
9. `research_reporter` records final reports and dashboard updates.
10. `chief_quant_orchestrator` records promote, observe, reject, or block in the decision log.
