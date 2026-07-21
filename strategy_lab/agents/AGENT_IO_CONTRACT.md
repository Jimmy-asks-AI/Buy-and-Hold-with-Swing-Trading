# Agent I/O Contract

This contract standardizes what every quant research subagent receives and returns.

## Task Brief Required Fields

- `task_id`
- `parent_id`
- `version`
- `assigned_agent`
- `objective`
- `baseline`
- `allowed_inputs`
- `allowed_writes`
- `dependencies`
- `required_outputs`
- `forbidden`
- `acceptance_criteria`
- `failure_conditions`
- `quality_gates`
- `next_handoff`

Task briefs are mandatory for governed tasks accepted after `20260527_subagent_effectiveness_critique_v4`. Historical rows may remain task-board only, but new accepted work should not rely on chat context as the sole task definition.

## Required Manifest Fields

Every completed task must emit `agent_run_manifest.json` with the fields required by `_templates/agent_run_manifest.schema.json`.

Every model-producing task must also emit `model_run_manifest.json` that validates against `_templates/model_run_manifest.schema.json`.

Manifest paths are not advisory text. `artifacts`, `outputs`, and `changed_files` must resolve to existing files or explicit glob patterns that match existing files. A task is not accepted until these paths pass `strategy_lab/agent_framework_check.py`.

Model manifests must not list their own `model_run_manifest.json` or generated `model_run_manifest_check.csv` as required artifacts. Those files are validation products, not dependencies of the model run.

Every emitted `agent_run_manifest.json` must have a matching row in `reports/AGENT_TASK_BOARD.md`. A manifest cannot remain tied to a `backlog` task; the task board must be updated to the appropriate accepted, rejected, blocked, or observation workflow status before final framework acceptance.

For model-facing tasks, the manifest or report must distinguish:

- `task_status`: whether the assigned subagent work completed.
- `model_decision`: whether the model is promoted, rejected, observation-only, or blocked.

The word `accepted` means the task artifact is accepted, not that the model is promoted.

## Standard Artifact Names

| Artifact | Owner | Purpose |
| --- | --- | --- |
| `agent_report.md` | all agents | Human-readable summary and handoff. |
| `agent_run_manifest.json` | all agents | Reproducible task-level metadata. |
| `data_quality_report.csv` | `data_steward` | Dataset quality and coverage. |
| `point_in_time_check.csv` | `data_steward` | PIT availability evidence. |
| `factor_metrics.csv` | `factor_researcher` | IC, ICIR, quantile, turnover, and stability. |
| `regime_performance.csv` | `regime_timing_researcher` | State-level performance and sample evidence. |
| `technical_signal_latest.csv` | `technical_market_analyst` | Latest object-level technical view. |
| `technical_signal_panel.csv` | `technical_market_analyst` | Historical technical evidence panel without forward labels. |
| `technical_formula_spec.csv` | `technical_market_analyst` | Reproducible technical component definitions and weights. |
| `fundamental_signal_latest.csv` | `fundamental_equity_analyst` | Latest object-level valuation and fundamental view. |
| `fundamental_data_gap_register.csv` | `fundamental_equity_analyst` | Missing, snapshot-only, or non-PIT fundamental data gaps. |
| `synthesized_research_views.csv` | `investment_view_synthesizer` | Research-only final view with confidence and invalidation. |
| `decision_trace.csv` | `investment_view_synthesizer` | Evidence, conflict, confidence cap, and failure scenario trace. |
| `target_weights.csv` | `portfolio_risk_engineer` | Portfolio construction output. |
| `constraint_check.csv` | `portfolio_risk_engineer` or model script | Weight, cash, leverage, turnover, and cap checks. |
| `cost_sensitivity.csv` | `execution_cost_analyst` | Net performance across cost assumptions. |
| `validation_findings.csv` | `backtest_validation_auditor` | Leakage, overfit, robustness, and promotion findings. |
| `leakage_checklist.csv` | `backtest_validation_auditor` | Future-function and timing checklist. |
| `robustness_summary.csv` | `backtest_validation_auditor` | PBO/DSR, fold stability, and stress evidence. |
| `smoke_test_results.csv` | `code_quality_engineer` | Commands and pass/fail status. |
| `output_integrity_check.csv` | `code_quality_engineer` | Required files and schema checks. |

## Gate Metric Reconciliation

Any agent that emits promotion or rejection metrics must also provide the source rows needed to recompute them.

- `candidate_gate_decision.csv` must reconcile with `nested_selection_by_fold.csv`.
- `nonbaseline_selection_rate` must be recomputable from train-sufficient prior-window folds, including baseline fallback rows.
- If a selector has fallback behavior, the fold log must include `baseline_fallback_used`.
- Full-sample diagnostic files must be explicitly marked diagnostic and cannot be the source of promotion metrics.
- Code-quality review must run `strategy_lab/agent_framework_check.py` after every accepted model or governance task.

## Effectiveness Review

Every five model-producing versions, or after any repeated default rejection chain, the orchestrator should run an effectiveness review before more implementation.

Required outputs:

- `subagent_effectiveness_scorecard.csv`
- `model_iteration_yield.csv`
- `process_defect_log.csv`
- `optimization_backlog.csv`
- `agent_report.md`

The review must answer:

- Did the framework block bad candidates?
- Did it produce any investable improvement?
- Which role is overloaded or violating boundaries?
- Which defect should become a reusable check?
- Should the next step be implementation, source discovery, or retirement?

## Forward-Label Signal Gate Policy

- A signal-research task that uses forward returns, future labels, or any column marked `forward`/`_fwd` must emit `signal_gate_holdout_validation.csv`.
- `signal_validation.csv` with forward labels is diagnostic unless the same variant passes the holdout implementation gate.
- `implementation_candidate_spec.csv` cannot set `implementation_allowed=true` unless the variant has `eligible_for_implementation=true` on the holdout split.
- `candidate_registry.csv` cannot mark a signal-gated row as `candidate` unless the holdout gate passes.
- A later model harness built from a signal that failed holdout is diagnostic history only and cannot be used as promotion evidence.

## Run Hygiene and Warning Policy

- Long-running scripts should preserve enough stdout/stderr evidence to diagnose warnings without overwhelming reports.
- Runtime warnings are `observation` when they do not change outputs, but they must be logged, fixed, or documented before the next accepted version.
- If warning fixes may change numeric evidence, regenerate affected outputs before handoff.
- Design-only versions must state that they are not promotion-eligible unless a later model-producing task validates them.

## Status Semantics

- `pass`: The assigned gate passed, within that agent's authority.
- `observation`: Evidence exists but is not enough for default promotion.
- `fail`: A required gate failed.
- `blocked`: Required input, data, code, or reproducibility evidence is missing.

## Cross-Agent Boundaries

- Agents may cite finalized artifacts from other agents only when listed in `allowed_inputs`.
- A task brief's `assigned_agent` and `next_handoff` must be in the registered agent roster unless the handoff is `none` or `n/a`.
- Agents must not infer missing metrics from chat context.
- Agents must not overwrite another agent's output directory.
- Agents must not change another role's decision field.
- Agents must not silently change metric definitions across versions; if a status or denominator changes, the output must include a compatibility field and a decision-log note.

## Promotion Boundary

Promotion requires all of the following:

- Data is approved or explicitly not needed.
- Candidate is predeclared.
- Portfolio constraints pass.
- Cost sensitivity is acceptable.
- Validation status is not `fail` or `blocked`.
- Code-quality status is `pass`.
- Reporter recorded the result.
- Orchestrator wrote the decision log entry.

## Research Assistant Boundary

Object-level research outputs are not promotion evidence by themselves.

- `technical_market_analyst` can produce price-action views but cannot authorize trading.
- `fundamental_equity_analyst` can produce valuation/fundamental views but must cap confidence when PIT history is incomplete.
- `investment_view_synthesizer` can produce `bullish`, `neutral`, `bearish`, or `blocked` research views, but those views remain `research_only` until a separate governed strategy task validates alpha, costs, turnover, and risk.
