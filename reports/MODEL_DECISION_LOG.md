# Model Decision Log

This log records decisions by the chief quant orchestrator. It is the only place where an experiment can be promoted, rejected, or moved into observation.

## 2026-05-26 - V3.9 Subagent Governance Framework

- Baseline: `HIRSSM V3.8`
- Decision: establish subagent governance before the next alpha-seeking model iteration.
- Rationale: the research system now needs durable role memory, context isolation, artifact-based handoff, and independent validation before future versions can be promoted.
- Promotion impact: no strategy candidate promoted.
- Evidence: `strategy_lab/agent_framework_check.py` returned `self_check_pass=true` for 9 agents, 7 required governance files, and 9 required sections.
- Subagent review incorporated: data/factor PIT rules, validation/risk/cost non-optimization boundaries, and task-board/manifest/decision-log audit fields.
- Required before next production promotion:
  - agent-specific `AGENT.md` is present;
  - task board entry is present;
  - run manifest is present;
  - validation and code-quality evidence are present;
  - metrics are compared to the active baseline.

## 2026-05-26 - V3.9 Baseline Validation Audit

- Baseline audited: `HIRSSM V3.8`
- Agent: `backtest_validation_auditor`
- Evidence status: `observation`
- Decision: keep V3.8 as the continuity comparison baseline, but do not treat V3.8 as fully validation-promoted production evidence.
- Key findings:
  - no blocking future-function evidence found in the V3.8 overlay layer;
  - selected V3.8 overlay improves only marginally versus the V3.6 exact control;
  - candidate selection still uses full-sample OOS summary ranking;
  - no V3.8-layer PBO/DSR, purged/embargo, or parameter sensitivity output found;
  - reproducibility manifest lacks full command, environment, data snapshot/hash, and git commit.
- Required next work:
  - add PBO/DSR or equivalent nested robustness evidence;
  - improve run manifest reproducibility fields;
  - keep future model promotion blocked until independent validation is stronger.

## 2026-05-26 - V3.9 PBO/DSR and Reproducibility Gap Closure

- Tasks:
  - `20260526_v3_9_pbo_dsr_gap`
  - `20260526_v3_9_repro_manifest_gap`
- Decision: V3.8 stays as an observation/continuity baseline, not a validation-promoted model.
- PBO/DSR result:
  - 10bps PBO `0.7659`, DSR-style worst selected probability `0.7006`;
  - 20bps PBO `0.7421`, DSR-style worst selected probability `0.6050`;
  - 30bps PBO `0.6944`, DSR-style worst selected probability `0.4976`.
- Paired V3.6 stability:
  - yearly hit rate is only `55%-60%`;
  - mean yearly delta is approximately flat;
  - regime delta is negative on average, with `risk_on_overheat` as the worst regime.
- Reproducibility:
  - original V3.8 manifest lacks command, argv, environment, git commit/status, script/config hashes, input hashes, output hashes, and dependency versions;
  - a retrospective enhanced manifest template was generated, but it is post-run evidence and cannot prove the original run environment.
- Promotion rule update: future candidate promotion is blocked if either strict PBO/DSR evidence fails or model-run manifest fields are incomplete.

## 2026-05-26 - V3.9 V3.6 Upstream Audit and V3.10 Governance Design

- V3.6 upstream audit status: `fail` for use as V3.8 default upstream baseline source.
- Decision:
  - V3.6 can remain only as an experimental comparison snapshot;
  - V3.8 cannot be treated as clean because it depends on V3.6 targets;
  - V3.10 must first rebuild a governance-clean baseline before seeking new alpha.
- Reasons:
  - V3.6 was selected from same-period OOS candidate results;
  - V3.6 state gating uses ex-post attribution logic from V3.5;
  - V3.2 and V3.4 upstream artifacts include failed self-check or promotion-style checks;
  - V3.6 manifest lacks strict reproducibility fields.
- V3.10 governance accepted:
  - `model_run_manifest.v1` schema and validator added;
  - future model promotion requires strict command, argv, environment, code/config/input/output hash, artifact inventory, metrics, self-check, and handoff fields;
  - candidate selection must be nested/purged and cannot use full-sample OOS ranking.

## 2026-05-26 - V3.10 Clean Baseline Accepted

- Model: `HIRSSM V3.10 Clean Rank-Vol Core`
- Script: `strategy_lab/hirssm_v3_10_clean_baseline.py`
- Output: `outputs/hirssm_v3_10_clean_baseline`
- Decision: accept as the new governance control baseline, not as an alpha-promoted production model.
- Construction:
  - uses predeclared `rank_plus_volatility_scaling` rules from config;
  - uses fixed state sleeve budgets, caps, defensive cash substitution, explicit `CASH`, and a monthly turnover cap;
  - keeps `range_reversal` and `style_trend_continuation` disabled by default;
  - does not inherit V3.6/V3.8 selected weights and does not use same-period OOS candidate selection.
- 10bps result:
  - annual return `8.76%`;
  - Sharpe `0.455`;
  - max drawdown `-54.56%`;
  - benchmark annual return `8.02%`;
  - benchmark max drawdown `-71.48%`;
  - annual excess versus `000985` benchmark `0.74%`;
  - average trade turnover `0.584`;
  - average cash `0.195`.
- Cost sensitivity:
  - 5bps annual excess `1.10%`;
  - 10bps annual excess `0.74%`;
  - 20bps annual excess `0.02%`;
  - 30bps annual excess `-0.70%`.
- Checks:
  - script self-check passed;
  - strict model manifest check passed with only dirty-worktree warning;
  - agent framework check passed.
- Next decision: build V3.11 as a nested candidate-validation harness; no new alpha candidate may be promoted from full-sample ranking.

## 2026-05-26 - V3.11 Nested Candidate Harness Accepted, Candidates Rejected

- Model system: `HIRSSM V3.11 Nested Candidate Harness`
- Script: `strategy_lab/hirssm_v3_11_nested_candidate_harness.py`
- Output: `outputs/hirssm_v3_11_nested_candidate_harness`
- Decision:
  - accept the V3.11 harness as the candidate governance path;
  - reject the current predeclared candidate set for default promotion;
  - keep V3.10 Clean Rank-Vol Core as active governance baseline.
- Required evidence now emitted:
  - candidate registry;
  - split manifest;
  - embargo/purge audit;
  - inner candidate scores;
  - nested selection by fold;
  - outer-fold OOS results;
  - same-period V3.10 baseline comparison;
  - cost sensitivity;
  - PBO CSCV summary and split detail;
  - candidate gate decision;
  - validation, leakage, and robustness reports.
- 10bps nested OOS result:
  - selected annual return `11.82%`;
  - same-period V3.10 annual return `12.07%`;
  - annual delta vs V3.10 `-0.25%`;
  - selected Sharpe `0.581`;
  - same-period V3.10 Sharpe `0.595`;
  - max drawdown roughly unchanged at `-54.56%`.
- PBO:
  - 5bps `0.361`;
  - 10bps `0.369`;
  - 20bps `0.381`;
  - 30bps `0.365`;
  - all are `fail` under the current PBO gate.
- Gate result: `reject_for_default_observation_only`.
- Next decision: V3.12 should redesign candidates to improve nested same-period performance and PBO stability, not tune from full-sample rankings.

## 2026-05-26 - Governance Schema Upgrade Accepted

- Decision: accept the governance schema upgrade.
- Added schemas:
  - `task_brief.schema.json`
  - `candidate_registry.schema.json`
  - `split_manifest.schema.json`
  - `candidate_gate_decision.schema.json`
- Updated `agent_framework_check.py` to check:
  - required governance schemas exist and parse;
  - accepted task board output refs exist;
  - V3.11 candidate registry, split manifest, and gate decision include required fields.
- Result: framework check passed with `required_file_count=15`.

## 2026-05-26 - V3.12 Candidate Improvement Design Accepted

- Decision: accept V3.12 design artifacts as `design_only`; no model or candidate is promoted.
- Source evidence: V3.11 rejected candidates because nested selected candidates did not beat V3.10 same-period baseline and PBO failed across the cost grid.
- Proposed candidate directions:
  - guarded industry trend with lower turnover;
  - baseline blend with confidence gate;
  - valuation/risk repair defensive guard;
  - PBO stability-penalized selector.
- Required next step: implement the designed candidates and rerun nested/purged validation before any promotion decision.

## 2026-05-26 - V3.12 Candidate Implementation Harness Accepted, Candidates Rejected

- Model system: `HIRSSM V3.12 Candidate Implementation Harness`
- Script: `strategy_lab/hirssm_v3_12_candidate_implementation_harness.py`
- Output: `outputs/hirssm_v3_12_candidate_implementation_harness`
- Decision:
  - accept the V3.12 implementation harness and its governance artifacts;
  - reject V3.12 candidates for default promotion;
  - keep V3.10 Clean Rank-Vol Core as the active governance baseline.
- Implemented candidates:
  - `guarded_industry_trend_low_turnover`;
  - `baseline_blend_confidence_gate`;
  - `valuation_risk_repair_defensive_guard`;
  - `pbo_stability_penalized_selector`.
- 10bps nested OOS result:
  - annual return `12.07%`;
  - Sharpe `0.594`;
  - max drawdown `-54.56%`;
  - annual delta versus same-period V3.10 `-0.003%`.
- Cost sensitivity versus same-period V3.10:
  - 5bps annual delta `-0.032%`;
  - 10bps annual delta `-0.003%`;
  - 20bps annual delta `0.025%`;
  - 30bps annual delta `0.034%`.
- PBO:
  - 5bps `0.579` fail;
  - 10bps `0.484` fail;
  - 20bps `0.266` observation;
  - 30bps `0.099` pass.
- Self-review correction:
  - V3.12 initially reported candidate gate `nonbaseline_selection_rate` incorrectly because baseline fallback rows used a different `selection_status`;
  - the harness was corrected to keep all train-sufficient folds under `selected_by_prior_window` and mark fallback with `baseline_fallback_used`;
  - corrected nonbaseline selection rates are 5bps `14.3%`, 10bps `33.3%`, 20bps `42.9%`, and 30bps `47.6%`;
  - this correction did not change NAV, returns, drawdown, PBO, or the rejection decision.
- Checks:
  - script self-check passed;
  - strict model manifest passed with only dirty-worktree warning;
  - agent framework check passed.
- Interpretation: V3.12 made the selector safer and almost baseline-equivalent, but it did not create enough robust alpha. The 5/10bps PBO failures block default promotion.
- Next decision: V3.13 should diagnose whether the failure comes from candidate similarity to baseline, turnover friction, or unstable industry-trend timing before any new implementation.

## 2026-05-26 - V3.13 Failure Revision Design Accepted

- Task: `20260526_v3_13_failure_revision_design`
- Script: `strategy_lab/hirssm_v3_13_failure_revision_design.py`
- Output: `outputs/agent_runs/v3_13/failure_revision_design`
- Decision:
  - accept V3.13 as a design and governance task;
  - no new model is promoted;
  - V3.10 remains the active governance baseline.
- Diagnosis:
  - V3.12 has 9 fail-level findings;
  - 10bps PBO remains `0.484`, which fails the low-cost gate;
  - 10bps nonbaseline selection rate is `33.3%`;
  - 10bps baseline fallback rate is `14.3%`;
  - V3.12 is effectively baseline-equivalent and does not justify more selector complexity by itself.
- V3.14 candidate directions:
  - `orthogonal_breadth_regime_overlay`;
  - `residual_industry_momentum_low_corr`;
  - `value_quality_defensive_barbell`;
  - `cost_aware_no_trade_band_overlay`;
  - `candidate_diversity_selector`.
- New rule: V3.14 must perform pre-backtest signal validation and candidate similarity checks before any portfolio-level implementation.

## 2026-05-26 - Subagent Iteration Guardrails Accepted

- Task: `20260526_subagent_iteration_guardrails`
- Output: `outputs/agent_runs/governance/subagent_iteration_guardrails`
- Decision: accept the governance guardrail update.
- Problems corrected:
  - manifest completeness problems must be caught as part of framework checks, not only during manual review;
  - gate metrics such as `nonbaseline_selection_rate` must reconcile to source fold logs;
  - structured candidate output checks must apply to every candidate harness, not only V3.11.
- Code and docs updated:
  - `strategy_lab/agent_framework_check.py`;
  - `strategy_lab/agents/AGENT_IO_CONTRACT.md`;
  - `strategy_lab/agents/AGENT_WORKFLOW.md`;
  - `strategy_lab/agents/code_quality_engineer/AGENT.md`.
- New checks:
  - every `agent_run_manifest.json` must point `artifacts`, `outputs`, and `changed_files` to existing files or matching globs;
  - `candidate_registry.csv`, `split_manifest.csv`, and `candidate_gate_decision.csv` are schema-checked for every candidate harness output directory;
  - `candidate_gate_decision.csv` `nonbaseline_selection_rate` is recomputed from `nested_selection_by_fold.csv`.
- Latest framework check passed with `run_manifest_count=15`.

## 2026-05-27 - V3.14 to V3.18 Five-Version Iteration Completed

- Scope: five consecutive versions from V3.14 through V3.18.
- V3.14 `orthogonal_candidate_research`:
  - accepted as pre-backtest signal research;
  - tested 3 factor directions;
  - only `orthogonal_breadth_regime_overlay` passed the pre-backtest gate;
  - V3.12 raw candidates had max absolute active-return correlation `0.999`.
- V3.15 `breadth_overlay_harness`:
  - implemented only the V3.14-passing breadth signal;
  - 10bps nested annual return `11.92%`;
  - 10bps annual delta versus same-period V3.10 `-0.154%`;
  - 10bps PBO `0.611`, fail;
  - candidates rejected for default promotion.
- V3.16 `cost_aware_stability_design`:
  - accepted as execution-cost design only;
  - produced 4 high-priority no-trade-band specs;
  - no model promotion allowed from cost-only design.
- V3.17 `candidate_diversity_governance`:
  - accepted as selector governance;
  - detected 3 near-duplicate active-return candidate pairs;
  - filtered next-PBO candidate count to 1 included candidate.
- V3.18 `five_version_review`:
  - accepted as review and decision artifact;
  - final decision: keep V3.10 as active governance baseline;
  - no new default model promoted.
- Optional next task: V3.19 may implement the V3.17-filtered candidate plus V3.16 no-trade overlay, but only as a new governed candidate.

## 2026-05-27 - Subagent Iteration Guardrails V2 Accepted

- Task: `20260527_subagent_iteration_guardrails_v2`
- Output: `outputs/agent_runs/governance/subagent_iteration_guardrails_v2`
- Decision:
  - accept this as a governance-only optimization;
  - no HIRSSM model logic, weights, or backtest output is changed;
  - V3.10 remains the active governance baseline.
- Problems converted into durable checks:
  - runtime warnings must be captured, classified, and either fixed or documented;
  - model manifests cannot list their own `model_run_manifest.json` or generated `model_run_manifest_check.csv` as required artifacts;
  - every `agent_run_manifest.json` must have a task-board row and cannot remain tied to `backlog`;
  - design-only tasks must explicitly state that they are not promotion-eligible.
- Code and docs updated:
  - `strategy_lab/agent_framework_check.py`;
  - `strategy_lab/agents/AGENT_IO_CONTRACT.md`;
  - `strategy_lab/agents/AGENT_WORKFLOW.md`;
  - `strategy_lab/agents/code_quality_engineer/AGENT.md`.
- Latest expected framework check: `self_check_pass=true`, `run_manifest_count=21`.

## 2026-05-27 - V3.19 to V3.23 Five-Version Iteration Completed

- Scope: five consecutive versions from V3.19 through V3.23.
- V3.19 `filtered_no_trade_candidate`:
  - accepted as blocked evidence;
  - V3.17 filtering left no nonbaseline candidate eligible for V3.16 no-trade implementation;
  - no filtered-out candidate was forced into a backtest.
- V3.20 `rescue_signal_research`:
  - tested 3 drawdown/reentry signals;
  - only `vol_compression_reentry` passed the full-sample pre-backtest signal gate;
  - post-review holdout validation passed `0` signals, so no V3.20 signal is eligible for implementation under the corrected source-gate rule;
  - this remained signal research only, not promotion evidence.
- V3.21 `vol_compression_harness`:
  - implemented the then-allowed V3.20 full-sample-passing signal plus a 3pct no-trade variant;
  - after source-gate correction, V3.21 is diagnostic history only because the source signal failed holdout eligibility;
  - 10bps nested annual return was `12.14%`;
  - 10bps annual delta versus same-period V3.10 was `0.07%`;
  - 10bps PBO was `0.286` observation, but 20bps PBO was `0.421` fail;
  - default promotion rejected because the annual delta did not clear the 50bps gate and cost robustness was weak.
- V3.22 `failure_attribution`:
  - accepted as attribution only;
  - diagnosed marginal alpha too small, 20bps cost-stress PBO failure, and cash-release overlay similarity to baseline.
- V3.23 `five_version_review`:
  - accepted as review and decision artifact;
  - final decision: keep V3.10 as active governance baseline;
  - V3.21 remains observation-only evidence.
- Next allowed work:
  - V3.24 should introduce a genuinely orthogonal information source, such as macro liquidity, rate spread, valuation spread, or constituent breadth;
  - do not continue by tuning small cash-release percentages.

## 2026-05-27 - Subagent Iteration Guardrails V3 Accepted

- Task: `20260527_subagent_iteration_guardrails_v3`
- Output: `outputs/agent_runs/governance/subagent_iteration_guardrails_v3`
- Decision:
  - accept this as a governance-only source-gate correction;
  - no HIRSSM model is promoted;
  - V3.10 remains the active governance baseline.
- Problem corrected:
  - V3.20 used forward 63-day returns for full-sample signal triage;
  - V3.21 implemented the full-sample-passing signal before a separate holdout implementation gate existed.
- Corrections:
  - V3.20 now emits `signal_gate_holdout_validation.csv`;
  - V3.20 now has `holdout_passed_signal_count=0` and `candidate_count=0`;
  - V3.21 is retained only as diagnostic rejected history, not promotion evidence;
  - V3.23 was regenerated to record the source holdout failure.
- New framework check:
  - any `signal_validation.csv` using forward-label evidence must have `signal_gate_holdout_validation.csv`;
  - `implementation_candidate_spec.csv` cannot allow implementation without holdout eligibility;
  - signal-gated `candidate_registry.csv` rows cannot be `candidate` unless the holdout gate passes.
- Latest expected framework check: `self_check_pass=true`, `run_manifest_count=27`.

## 2026-05-27 - V3.24 Valuation Source Research Accepted

- Task: `20260527_v3_24_valuation_source_research`
- Script: `strategy_lab/hirssm_v3_24_valuation_source_research.py`
- Output: `outputs/agent_runs/v3_24/valuation_source_research`
- Decision:
  - accept V3.24 as source research only;
  - no model harness is authorized;
  - V3.10 remains the active governance baseline.
- New information source:
  - historical index valuation spread from `data_raw/index/akshare_csindex/daily_csindex`;
  - tested dividend versus broad, broad-market deep value repair, and large versus mid valuation spread.
- Results:
  - signals tested: `3`;
  - full-sample diagnostic pass count: `0`;
  - holdout implementation pass count: `0`;
  - implementation candidate count: `0`.
- Orthogonality:
  - valuation signals were not highly correlated with V3.20 vol-compression or V3.10 cash exposure;
  - the source was independent enough, but not effective enough.
- Next decision:
  - do not implement V3.25 from V3.24 because no signal passed the source gate;
  - next source research should use macro liquidity/rate spread, constituent breadth, or industry-internal structure rather than simple valuation spread rules.

## 2026-05-27 - V3.25 Industry Structure Source Research Accepted

- Task: `20260527_v3_25_industry_structure_source_research`
- Script: `strategy_lab/hirssm_v3_25_industry_structure_source_research.py`
- Output: `outputs/agent_runs/v3_25/industry_structure_source_research`
- Decision:
  - accept V3.25 as source research only;
  - no model harness is authorized;
  - V3.10 remains the active governance baseline.
- New information source:
  - SW level-1 industry internal structure from `data_raw/index/akshare_sw_industry/daily_sw`;
  - tested industry breadth repair, narrow-leadership overheat defense, and dispersion rotation opportunity.
- Results:
  - signals tested: `3`;
  - full-sample diagnostic pass count: `0`;
  - holdout implementation pass count: `0`;
  - implementation candidate count: `0`.
- Orthogonality:
  - industry-structure signals had low correlation with V3.20 vol compression, V3.24 broad value, and V3.10 cash exposure;
  - the source was independent enough, but not effective enough to implement.
- Next decision:
  - do not implement V3.26 from V3.25 because no signal passed the source gate;
  - next source research should use macro liquidity/rate spread data if available, or move to more granular constituent/stock-level breadth.

## 2026-05-27 - V3.26 Macro Data Readiness Accepted

- Task: `20260527_v3_26_macro_data_readiness`
- Script: `strategy_lab/hirssm_v3_26_macro_data_readiness.py`
- Output: `outputs/agent_runs/v3_26/macro_data_readiness`
- Decision:
  - accept V3.26 as data-readiness research only;
  - block macro signal research until point-in-time macro/rate data is available;
  - V3.10 remains the active governance baseline.
- Data scan:
  - local CSV files scanned: `160`;
  - required macro/rate series: `10`;
  - usable point-in-time series: `0`;
  - missing core series: `china_pmi`, `cn_10y_gov_bond_yield`, `us_10y_treasury_yield`, `usdcny`.
- Blocked candidate families:
  - `cn_us_rate_spread_risk_budget`;
  - `macro_liquidity_repair`;
  - `inflation_policy_constraint_defense`.
- Governance reason:
  - macro data must include `date`, `available_date`, `value`, `source`, `frequency`, and `revision_policy`;
  - no macro factor can enter a historical backtest using current snapshots or revised series without publication-date controls.
- Next decision:
  - V3.27 should first fetch/build point-in-time macro/rate tables;
  - only after data readiness passes should macro signal validation run.

## 2026-05-27 - V3.27 Macro Data Ingestion Accepted

- Task: `20260527_v3_27_macro_data_ingestion`
- Script: `strategy_lab/hirssm_v3_27_macro_data_ingestion.py`
- Output: `outputs/agent_runs/v3_27/macro_data_ingestion`
- Data output: `data_raw/macro`
- Decision:
  - accept V3.27 as a data-layer upgrade;
  - allow only gated macro families to enter later signal validation;
  - no model harness, implementation, or default promotion is authorized by this version.
- Ingestion result:
  - AkShare source success count: `8`;
  - source failure count: `1`;
  - required macro/rate series: `10`;
  - usable series: `8`;
  - limited-history series: `1`;
  - missing series: `1`.
- New local PIT tables:
  - `china_10y_yield.csv`;
  - `us_10y_yield.csv`;
  - `cn_us_10y_rate_spread.csv`;
  - `cny_fx.csv`;
  - `pmi.csv`;
  - `m2_social_financing.csv`;
  - `cpi_ppi.csv`;
  - `commodity_index.csv`;
  - `macro_pit_panel.csv`.
- Gate decisions:
  - `cn_us_rate_spread_risk_budget`: `allow_signal_validation`;
  - `macro_liquidity_repair`: `block_signal_validation` because `china_tsf_yoy` is missing;
  - `inflation_policy_constraint_defense`: `allow_limited_signal_validation` because commodity history starts in 2011.
- Failure note:
  - `akshare.macro_china_shrzgm` failed because the upstream Mofcom SSL endpoint rejected the request;
  - `china_new_financial_credit_yoy` was stored as a substitute series but does not satisfy the TSF requirement.
- Next decision:
  - V3.28 should run holdout-gated macro signal validation for `cn_us_rate_spread_risk_budget` first;
  - do not use `macro_liquidity_repair` until TSF is available;
  - treat inflation/commodity research as limited-history only.

## 2026-05-27 - V3.28 Macro Rate/FX Signal Validation Accepted

- Task: `20260527_v3_28_macro_rate_fx_signal_validation`
- Script: `strategy_lab/hirssm_v3_28_macro_rate_fx_signal_validation.py`
- Output: `outputs/agent_runs/v3_28/macro_rate_fx_signal_validation`
- Decision:
  - accept V3.28 as macro signal validation;
  - allow only holdout-qualified signals to enter a later implementation harness;
  - no default model promotion is authorized by this version.
- Method:
  - macro features are aligned by `available_date <= signal_date`;
  - monthly signal rows use a one-period feature lag before forward labels are evaluated;
  - validation uses full-sample diagnostic plus holdout implementation gate.
- Results:
  - signals tested: `3`;
  - full-sample diagnostic pass count: `3`;
  - holdout implementation pass count: `2`;
  - implementation candidate count: `2`.
- Accepted for later implementation validation:
  - `us_rate_shock_fx_stress_defense`;
  - `spread_repair_risk_on`.
- Rejected for implementation:
  - `rate_fx_stress_defense`, because holdout returns were not worse than unconditional returns.
- Orthogonality:
  - all tested macro rate/FX signals passed low-correlation checks versus V3.20 vol-compression, V3.24 valuation repair, V3.25 industry breadth, and V3.10 cash exposure.
- Next decision:
  - V3.29 should implement only the two holdout-qualified candidates in a constrained harness;
  - the harness must compare against V3.10 at 5/10/20bps and still require nested/PBO-style rejection control before any default change.

## 2026-05-27 - V3.29 Macro Rate/FX Harness Accepted

- Task: `20260527_v3_29_macro_rate_fx_harness`
- Script: `strategy_lab/hirssm_v3_29_macro_rate_fx_harness.py`
- Model output: `outputs/hirssm_v3_29_macro_rate_fx_harness`
- Agent output: `outputs/agent_runs/v3_29/macro_rate_fx_harness`
- Decision:
  - accept V3.29 as constrained implementation validation;
  - reject for default promotion;
  - keep the two macro candidates as observation/research candidates only.
- Implemented candidates:
  - `us_rate_shock_fx_stress_defense`;
  - `spread_repair_risk_on`.
- 10bps nested OOS result:
  - annual return: `11.8650%`;
  - Sharpe: `0.5784`;
  - max drawdown: `-56.2903%`;
  - annual delta versus V3.10 same-period baseline: `-0.2083%`;
  - drawdown delta versus V3.10: `-1.7336%`.
- Cost-gate result:
  - 5bps annual delta versus V3.10: `-0.2351%`;
  - 10bps annual delta versus V3.10: `-0.2083%`;
  - 20bps annual delta versus V3.10: `-0.1553%`;
  - all cost scenarios fail the annual-delta gate.
- PBO result:
  - 5bps PBO: `0.5198`, fail;
  - 10bps PBO: `0.6111`, fail;
  - 20bps PBO: `0.7421`, fail.
- Selection behavior:
  - each cost scenario had 21 train-sufficient selected years;
  - V3.10 baseline selected 11 years;
  - `spread_repair_risk_on` selected 7 years;
  - `us_rate_shock_fx_stress_defense` selected 3 years.
- Trigger behavior:
  - stress-defense trigger: 20 of 292 months;
  - spread-repair trigger: 58 of 292 months.
- Interpretation:
  - V3.28 source signal quality did not survive portfolio constraints, costs, and nested/PBO governance;
  - rate/FX information is still useful as a research source, but current implementation is not investable enough.
- Next decision:
  - V3.30 should perform failure attribution before any further implementation;
  - likely review whether cash cap, trigger sparsity, and candidate interaction dilute signal value.

## 2026-05-27 - V3.30 Macro Rate/FX Failure Attribution Accepted

- Task: `20260527_v3_30_macro_rate_fx_failure_attribution`
- Script: `strategy_lab/hirssm_v3_30_macro_rate_fx_failure_attribution.py`
- Output: `outputs/agent_runs/v3_30/macro_rate_fx_failure_attribution`
- Decision:
  - accept V3.30 as a failure-attribution report;
  - keep V3.29 rejected for default promotion;
  - keep the macro rate/FX branch observation-only;
  - do not tune candidate amplitudes or thresholds before selected-year/regime attribution.
- Main evidence:
  - 10bps PBO: `0.6111`;
  - 10bps annual delta versus V3.10: `-0.2083%`;
  - 10bps drawdown delta versus V3.10: `-1.7336%`;
  - average trigger rate across implemented macro candidates: `13.36%`;
  - average absolute cash delta across implemented macro candidates: `1.16%`;
  - 5-to-20bps candidate-minus-baseline cost drag: `0.1492%`.
- Root causes:
  - `pbo_instability`: dominant hard failure;
  - `negative_marginal_oos_performance`: the selected macro harness reduced OOS annual return and worsened drawdown;
  - `portfolio_overlay_dilution`: triggers and cash deltas were too small to preserve source signal value;
  - `candidate_selection_not_decisive`: the V3.10 baseline was still selected in most years;
  - `cost_drag_not_primary`: transaction cost tuning is not the first-order issue.
- Next decision:
  - V3.31 should split candidate contribution by selected year, regime, and trigger month before any new parameter change;
  - V3.32 can test macro signals as gates on existing risk budget rather than small additive overlays.

## 2026-05-27 - V3.31 Selected-Year/Regime Attribution Accepted

- Task: `20260527_v3_31_selected_year_regime_attribution`
- Script: `strategy_lab/hirssm_v3_31_selected_year_regime_attribution.py`
- Output: `outputs/agent_runs/v3_31/selected_year_regime_attribution`
- Decision:
  - accept V3.31 as selected-year, regime, and trigger-month attribution;
  - keep V3.29 macro candidates observation-only;
  - keep V3.10 clean rank-vol core as the active baseline;
  - do not promote, retune, or amplitude-sweep macro candidates from this attribution alone.
- 10bps selected-year result:
  - macro candidates were selected in `10` years;
  - macro positive-delta rate was `50.00%`;
  - macro average selected-year delta was `-0.0833%`;
  - worst macro year was `2010` from `spread_repair_risk_on`, with `-1.9340%` annual delta.
- Candidate split:
  - `spread_repair_risk_on`: selected `7` years, positive-delta rate `42.86%`, average delta `-0.4702%`;
  - `us_rate_shock_fx_stress_defense`: selected `3` years, positive-delta rate `66.67%`, average delta `0.8193%`.
- Worst 10bps state/trigger buckets:
  - `spread_repair_risk_on` / `risk_on_overheat` / trigger false: annualized delta `-4.57%`;
  - `spread_repair_risk_on` / `risk_off_decline` / trigger true: annualized delta `-3.66%`;
  - `spread_repair_risk_on` / `risk_on_trend` / trigger false: annualized delta `-1.54%`.
- Interpretation:
  - V3.29's aggregate failure is concentrated more in `spread_repair_risk_on` than in the stress-defense branch;
  - the additive risk-on release overlay can hurt during crash-rebound and risk-off windows, including 2008 and 2010;
  - `us_rate_shock_fx_stress_defense` remains interesting as a defensive gate hypothesis but still has too few selected years for promotion.
- Next decision:
  - V3.32 may test a predeclared macro risk-budget gate;
  - V3.32 must rerun nested selection, cost scenarios, and PBO checks before any default-model change.

## 2026-05-27 - V3.32 Macro Risk-Budget Gate Accepted

- Task: `20260527_v3_32_macro_risk_budget_gate`
- Script: `strategy_lab/hirssm_v3_32_macro_risk_budget_gate.py`
- Model output: `outputs/hirssm_v3_32_macro_risk_budget_gate`
- Agent output: `outputs/agent_runs/v3_32/macro_risk_budget_gate`
- Decision:
  - accept V3.32 as a predeclared macro risk-budget gate harness;
  - reject for default promotion;
  - keep V3.10 clean rank-vol core as the active baseline.
- Implemented candidates:
  - `stress_budget_gate`: stress trigger cuts 25% of current risky budget into cash, capped at 55% cash;
  - `state_confirmed_dual_budget_gate`: same stress gate plus conservative repair release only in `range_bound` and `risk_on_trend`.
- 10bps nested OOS result:
  - annual return: `11.9782%`;
  - Sharpe: `0.5904`;
  - max drawdown: `-54.5567%`;
  - annual delta versus V3.10: `-0.0951%`;
  - drawdown delta versus V3.10: `0.0000%`.
- Cost and PBO result:
  - 5bps annual delta: `-0.1176%`, PBO `0.2381`, observation;
  - 10bps annual delta: `-0.0951%`, PBO `0.3294`, observation;
  - 20bps annual delta: `-0.0507%`, PBO `0.4405`, fail.
- Selection behavior:
  - `stress_budget_gate` selected `6` years per cost scenario;
  - `state_confirmed_dual_budget_gate` selected `1` year per cost scenario;
  - V3.10 baseline selected `14` years per cost scenario.
- Gate exposure:
  - `stress_budget_gate`: 18 active months, average active cash delta `19.27%`;
  - `state_confirmed_dual_budget_gate`: 45 active months, average active cash delta `10.13%`.
- Interpretation:
  - V3.32 materially improves the V3.29 failure profile: PBO improves from fail to observation at 5/10bps, and drawdown no longer worsens;
  - the strategy still fails the annual-delta promotion gate and 20bps PBO gate;
  - the stress-only branch appears cleaner than the dual repair branch.
- Next decision:
  - V3.33 should perform candidate-level attribution for V3.32, especially separating stress-only versus dual-gate effects;
  - do not tune the 25% budget cut or 55% cap until failure attribution confirms the mechanism.

## 2026-05-27 - Subagent Effectiveness Critique V4 Accepted

- Task: `20260527_subagent_effectiveness_critique_v4`
- Script: `strategy_lab/subagent_effectiveness_review.py`
- Output: `outputs/agent_runs/governance/subagent_effectiveness_critique_v4`
- Framework report: `reports/AGENT_FRAMEWORK_EFFECTIVENESS_REVIEW.md`
- Decision:
  - accept the critique as a governance optimization;
  - no model is promoted;
  - future model work must separate task acceptance from model decision.
- Evidence:
  - model harnesses with gate decisions: `6`;
  - promoted models: `0`;
  - default-rejected models: `6`;
  - average 10bps annual delta versus V3.10: `-0.1071%`;
  - best 10bps annual delta versus V3.10: `+0.0686%`;
  - average 10bps PBO: `0.4484`.
- Critique:
  - the subagent framework is good at blocking weak candidates and catching process defects;
  - it has not yet produced a better default model;
  - recent versions are too serial and branch-local;
  - validation-agent overload risks weakening independent review.
- Optimizations:
  - added machine-readable `task_briefs/`;
  - added `strategy_lab/subagent_effectiveness_review.py`;
  - extended `agent_framework_check.py` to validate present task briefs;
  - added workflow stop-loss rules after repeated non-promoted model versions;
  - updated RACI to reduce validation-agent role overlap.
- Next decision:
  - next model work should start from a task brief and should prefer source discovery or independent signal batches over small parameter repairs.

## 2026-05-27 - V3.33 Independent Signal Source Discovery Accepted

- Task: `20260527_v3_33_independent_signal_source_discovery`
- Script: `strategy_lab/hirssm_v3_33_independent_signal_source_discovery.py`
- Output: `outputs/agent_runs/v3_33/independent_signal_source_discovery`
- Decision:
  - accept V3.33 as broad independent source discovery;
  - no portfolio harness was run;
  - no model is promoted;
  - all tested signals remain observation-only.
- Scope:
  - evaluated `12` independent signal/data-source hypotheses;
  - used existing style index, industry index, valuation, and non-rate macro PIT data;
  - explicitly avoided further macro rate/FX gate parameter repair.
- Holdout result:
  - implementation candidates after stricter holdout and orthogonality gates: `0`;
  - `trend_breakout_continuation` was closest but failed the stricter gate because holdout RankIC was only `0.0016`;
  - `style_valuation_repair` had only `18` holdout observations and failed.
- Data-source blockers:
  - current industry components are blocked for historical backtest without historical component dates;
  - latest index weights are blocked without historical weight vintages;
  - sample stock QFQ data is too narrow for broad stock-factor discovery.
- Next decision:
  - hand off to `data_steward` for data-source repair or expansion;
  - do not open a V3.34 implementation harness until a source has a stronger holdout gate or new PIT data is added.

## 2026-05-27 - V3.34 Data Source Repair Audit Accepted

- Task: `20260527_v3_34_data_source_repair_audit`
- Script: `strategy_lab/hirssm_v3_34_data_source_repair_audit.py`
- Output: `outputs/agent_runs/v3_34/data_source_repair_audit`
- Catalog update: `data_catalog/a_share_data_source_repair_audit_v3_34.md`
- Decision:
  - accept V3.34 as data governance and repair planning;
  - no portfolio harness was run;
  - no signal or model is promoted;
  - V3.34 does not change the default model.
- Audit result:
  - audited datasets: `8`;
  - strict PIT backtest approved: `3`;
  - research-only: `4`;
  - blocked: `1`;
  - current snapshot datasets restricted: `3`.
- Important restrictions:
  - current index constituents, latest index weights, and current industry components cannot be used as historical point-in-time data;
  - stock QFQ daily data remains sample-only and not production-grade broad stock data;
  - financial indicators are blocked for historical factor use until announcement or available_date fields are added.
- Repair queue:
  - `historical_index_constituents_weights`;
  - `broad_stock_daily_raw_and_adjusted`;
  - `financial_indicator_point_in_time`;
  - `historical_industry_classification`;
  - `macro_release_calendar_vintage_upgrade`.
- Next decision:
  - open a data acquisition task before another implementation harness;
  - the most valuable next acquisition is historical constituents/weights plus a broad stock daily/raw-adjustment panel with delisted names retained.

## 2026-05-27 - V3.35 PIT Data Acquisition Contract Accepted

- Task: `20260527_v3_35_pit_data_acquisition_contract`
- Script: `strategy_lab/hirssm_v3_35_pit_data_acquisition_contract.py`
- Config: `configs/pit_data_acquisition_v3_35.json`
- Output: `outputs/agent_runs/v3_35/pit_data_acquisition_contract`
- Catalog update: `data_catalog/a_share_pit_data_acquisition_contract_v3_35.md`
- Decision:
  - accept V3.35 as a dry-run acquisition contract;
  - no live data download was executed;
  - no factor, portfolio, or model harness was run;
  - no signal or model is promoted.
- Contract scope:
  - dataset contracts: `9`;
  - provider endpoint mappings: `11`;
  - harvest-plan rows: `154`;
  - priority 1: historical index weights and date-specific index membership;
  - priority 2: all-status stock universe, raw daily bars, daily basic fields, adjustment factors, and tradeability flags.
- Credential readiness:
  - Tushare execute-ready: `False`;
  - JoinQuant execute-ready: `False`;
  - Chrome login state is not treated as a safe API credential for the Python data pipeline.
- Next decision:
  - do not hand this to `factor_researcher` yet;
  - next task should be SDK/API credential setup or a controlled pilot acquisition;
  - first pilot should fetch stock universe, one index weight file, and one small stock-daily yearly partition, then rerun PIT validation.

## 2026-05-27 - V3.36 PIT Data Pilot Readiness Accepted

- Task: `20260527_v3_36_pit_data_pilot_readiness`
- Script: `strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py`
- Config: `configs/pit_data_pilot_v3_36.json`
- Output: `outputs/agent_runs/v3_36/pit_data_pilot_readiness`
- Catalog update: `data_catalog/a_share_pit_data_pilot_v3_36.md`
- Decision:
  - accept V3.36 as SDK/credential readiness and pilot-control workflow;
  - no live pilot data was acquired;
  - no factor, portfolio, or model harness was run;
  - no signal or model is promoted.
- Readiness result:
  - `tushare` SDK installed and detected;
  - `jqdatasdk` SDK installed and detected;
  - Tushare API token is missing;
  - JoinQuant username/password are missing;
  - execute flag is `False`;
  - network-allowed flag is `False`.
- Pilot result:
  - planned pilot tasks: `6`;
  - acquired datasets: `0`;
  - blocked tasks: `6`;
  - blocked tasks are not marked as acquired.
- Next decision:
  - provide API credentials through environment variables or local untracked `configs/data_credentials.json`;
  - then rerun V3.36 with explicit pilot execution enabled;
  - do not send V3.36 outputs to `factor_researcher` until at least one pilot dataset is acquired and PIT-validated.

## 2026-05-27 - V3.37 Credential Bootstrap Accepted

- Task: `20260527_v3_37_credential_bootstrap`
- Script: `strategy_lab/hirssm_v3_37_credential_bootstrap.py`
- Output: `outputs/agent_runs/v3_37/credential_bootstrap`
- Catalog update: `data_catalog/a_share_credential_bootstrap_v3_37.md`
- Decision:
  - accept V3.37 as credential bootstrap and safety guard;
  - no API secrets were written or printed;
  - no live data acquisition was run;
  - no factor, portfolio, or model harness was run.
- Changes:
  - `.gitignore` now excludes `configs/data_credentials.json`, `configs/*.local.json`, and `*.env.local`;
  - added placeholder-only local credential template `configs/data_credentials.local.template.json`;
  - added explicit pilot execution example `configs/pit_data_pilot_v3_36_execute.example.json`;
  - added machine-readable credential policy and unblock checklist.
- Readiness result:
  - Tushare SDK installed: `True`;
  - Tushare token present: `False`;
  - JoinQuant SDK installed: `True`;
  - JoinQuant username/password present: `False`;
  - V3.36 pilot remains blocked.
- Next decision:
  - supply local credentials without committing them;
  - rerun V3.36 explicit pilot only after credentials are present;
  - keep factor research blocked until pilot data is acquired and PIT-validated.

## 2026-05-27 - V3.36 Explicit Pilot Entry Rerun

- Task: `20260527_v3_36_pit_data_pilot_readiness`
- Command: `python -X utf8 strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py --config configs/pit_data_pilot_v3_36_execute.example.json --execute`
- Output: `outputs/agent_runs/v3_36/pit_data_pilot_readiness`
- Decision:
  - explicit pilot entry was attempted;
  - execute flag was `True`;
  - network-allowed flag was `True`;
  - no live data was acquired because API credentials were still missing;
  - no factor, portfolio, or model harness was run.
- Result:
  - planned pilot tasks: `6`;
  - acquired datasets: `0`;
  - blocked tasks: `6`;
  - Tushare ready: `False`;
  - JoinQuant ready: `False`.
- Blockers:
  - Tushare SDK is installed, but token is missing;
  - JoinQuant SDK is installed, but username/password are missing.
- Next decision:
  - there is no productive model or data step left until credentials are supplied;
  - after credentials are available, rerun the same explicit pilot command and validate acquired pilot data before factor research.

## 2026-06-02 - V3.86-V3.92 Quality Rework Accepted

- Task range: `20260602_v3_86_quant_research_assistant_architecture` through `20260602_v3_92_sample_research_run`.
- Script: `strategy_lab/hirssm_v3_86_to_v3_92_quant_research_assistant.py`.
- Quality review: `reports/HIRSSM_V3_86_TO_V3_92_QUALITY_REVIEW.md`.
- Decision:
  - accept the reworked V3.86-V3.92 as a research-assistant capability layer;
  - no alpha, strategy, portfolio, or trade rule is promoted;
  - research views remain `research_only`.
- Corrections:
  - registered `technical_market_analyst`, `fundamental_equity_analyst`, and `investment_view_synthesizer` as formal agents with `AGENT.md` specs;
  - added roster validation for new task-brief owners and V3.86+ handoffs;
  - added `asof_date` to sample research objects;
  - added `technical_formula_spec.csv`;
  - added daily index `pe_ttm` fallback for market-index valuation;
  - added conflict-aware synthesis with score pull-to-neutral and confidence caps;
  - added `html_static_check.csv` and quality rework findings.
- Verification:
  - `python -m py_compile strategy_lab\agent_framework_check.py strategy_lab\quant_research_assistant_framework.py strategy_lab\hirssm_v3_86_to_v3_92_quant_research_assistant.py` passed;
  - all V3.86-V3.92 `self_check.csv` files have `0` fail rows;
  - `python strategy_lab\agent_framework_check.py` passed with active errors `0`.
