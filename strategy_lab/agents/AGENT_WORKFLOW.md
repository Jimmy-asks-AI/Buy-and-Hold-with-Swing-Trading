# Quant Research Agent Workflow

This workflow is the canonical handoff path for model, factor, data, and portfolio research. It is designed to prevent context bleed, full-sample promotion, and unclear ownership.

## Operating States

| State | Meaning | Promotion Allowed |
| --- | --- | --- |
| `idea` | Hypothesis or task not yet tested. | No |
| `observation` | Evidence exists but is incomplete, unstable, or not investable. | No |
| `candidate` | Predeclared variant ready for governed validation. | No |
| `gated` | Candidate passed assigned validation gates but is not default. | Orchestrator only |
| `baseline` | Governance comparison model. | Orchestrator only |
| `default` | Current promoted model path. | Orchestrator only |
| `rejected` | Failed gate but retained for evidence. | No |
| `blocked` | Missing required data, code, manifest, or validation. | No |

## Phase Gates

| Phase | Owner | Required Input | Required Output | Exit Gate |
| --- | --- | --- | --- | --- |
| 0. Intake | `chief_quant_orchestrator` | User objective, active baseline, task board | task brief | Exact owner, inputs, writes, and acceptance criteria are defined. |
| 1. Data Approval | `data_steward` | raw data, source notes | `data_quality_report.csv`, `point_in_time_check.csv` | Dataset is approved or marked `research_only`. |
| 2. Research View | `technical_market_analyst`, `fundamental_equity_analyst`, `investment_view_synthesizer` | approved object data and finalized upstream views | technical, fundamental, and synthesized research views | View is `research_only`; conflicts and data gaps remain visible. |
| 3. Signal Research | `factor_researcher`, `regime_timing_researcher` | approved data, hypothesis | factor/regime diagnostics | Signal is `observation` or `candidate`, never default. |
| 4. Portfolio Construction | `portfolio_risk_engineer` | approved signals, constraints | target weights, exposure, constraints | No invalid weights, leverage, cap, or cash violations. |
| 5. Cost and Capacity | `execution_cost_analyst` | trades, weights, liquidity | cost sensitivity, capacity flags | Net performance survives required cost cases. |
| 6. Validation | `backtest_validation_auditor` | candidate artifacts, baseline | leakage, robustness, PBO/DSR or equivalent | Evidence is pass, observation, fail, or blocked. |
| 7. Code Quality | `code_quality_engineer` | code, config, outputs, manifests | smoke and integrity reports | Scripts reproduce outputs and manifests pass. |
| 8. Reporting | `research_reporter` | finalized artifacts | version report and changelog | Metrics reconcile to source artifacts. |
| 9. Decision | `chief_quant_orchestrator` | all finalized artifacts | decision log entry | Candidate is promoted, observed, rejected, or blocked. |
| 10. Iteration Self-Review | `chief_quant_orchestrator`, `code_quality_engineer` | decision log, failed checks, corrected artifacts | correction note, framework check | Any bug found during the iteration is converted into a reusable check or a documented exception. |

## Non-Negotiable Rules

- A task has one accountable owner and one output directory.
- Subagents receive isolated context by default.
- Failed, rejected, or blocked work is preserved as evidence.
- Task acceptance is not model promotion. Every model-facing report must separate `task_status` from `model_decision`.
- Research assistant outputs are research-only unless a later model task validates alpha, costs, turnover, and risk.
- Full-sample ranking can produce diagnostics but cannot promote a candidate.
- Full-sample forward-label signal validation can produce diagnostics but cannot authorize implementation without a separate holdout gate.
- Any model-facing result without a strict manifest is blocked from promotion.
- Data without point-in-time availability is `research_only` for historical tests.
- Gate metrics must reconcile to their source fold or trade rows. A metric bug discovered during self-review requires code correction, output regeneration, and a decision-log note.
- Manifest `artifacts`, `outputs`, and `changed_files` must point to files that exist at acceptance time.
- A generated agent manifest must already be registered in the task board before final acceptance checks; task board status cannot remain `backlog`.
- Model manifests cannot depend on their own manifest or manifest-check outputs.
- A task brief is required before accepting any new governed task after `20260527_subagent_effectiveness_critique_v4`.
- Attribution can create a hypothesis brief, but cannot directly authorize threshold, amplitude, or universe parameter tuning.

## Minimum Promotion Evidence

- Predeclared candidate registry.
- If the candidate came from forward-label signal research, `signal_gate_holdout_validation.csv` must show holdout eligibility before implementation.
- Baseline comparison on same dates, benchmark, costs, and rebalance assumptions.
- Walk-forward or nested validation.
- Leakage checklist and purge/embargo audit when labels or forward returns are used.
- Cost sensitivity at `5/10/20/30bps` unless the orchestrator states a stricter grid.
- Constraint checks for weights, cash, turnover, concentration, and leverage.
- Strict model run manifest with code, config, data, environment, metrics, checks, and artifacts.
- Source-row reconciliation for gate metrics such as selection rate, fallback rate, PBO, turnover, and cost sensitivity.

## Research Yield Stop-Loss

Governance should prevent bad promotions, but it should not become an endless serial loop around weak ideas.

If five consecutive model-producing versions fail default promotion, the orchestrator must pause new implementation and choose one of these paths:

1. Return to source discovery with new data or a clearly different economic hypothesis.
2. Run a cross-version failure review and retire repeated branches to observation.
3. Open a task brief for a broader research batch, with multiple independent hypotheses and predeclared kill criteria.

Parameter changes to the same branch do not satisfy this stop-loss unless a new out-of-sample validation design is recorded first.

## Task Brief Discipline

Task briefs are the machine-readable boundary for context isolation.

- The brief must exist before final task acceptance.
- It must list exact allowed inputs and writes.
- It must name forbidden actions and failure conditions.
- It must be validated against `_templates/task_brief.schema.json`.
- Historical tasks without briefs remain valid evidence, but new governed tasks should not use that exception.

## Iteration Defect Handling

When an iteration exposes a process defect, use this order:

1. Identify whether the defect is model logic, validation logic, reporting logic, or framework governance.
2. Correct the source script or framework checker, not only the generated report.
3. Regenerate affected outputs before using them in the next task.
4. Record the correction in `MODEL_DECISION_LOG.md` and `AGENT_REVIEW_SUMMARY.md`.
5. Add or extend a reusable check so the same defect is caught before a future task is accepted.

## Iteration Acceptance Order

Use this order for each governed iteration:

1. Create or update the task board row with exact inputs, outputs, owner, and required checks.
2. Run the assigned agent work in its own output directory.
3. Capture runtime warnings, failed commands, and any output regeneration decisions.
4. Generate task artifacts and manifests after the final artifact list is stable.
5. Run syntax, schema, manifest, and framework checks.
6. Update decision and review logs with the final status.

If a task is design-only, the manifest and report must state that it is not promotion-eligible. If a model-producing task emits warnings that may affect evidence, it remains blocked until the affected outputs are regenerated or the warning is proved irrelevant.
