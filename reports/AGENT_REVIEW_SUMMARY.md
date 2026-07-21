# Agent Review Summary

## V3.9 Governance Setup

The subagent framework is designed around isolated context and artifact sharing only.

Core requirements:

- Each agent has a durable `AGENT.md` role spec.
- Each task brief must explicitly list allowed inputs and allowed writes.
- Each completed task must create `agent_run_manifest.json` and `agent_report.md`.
- The orchestrator is the only role allowed to promote a candidate to default.
- Validation and code-quality reports are required before model promotion.

Open next step:

- Use the framework to launch the first isolated V3.9 audit task against HIRSSM V3.8.
- Keep the next audit read-only and assign it to `backtest_validation_auditor`.

## Subagent Review Incorporated

- Data/factor review: enforce point-in-time availability, no current snapshots in history, no full-sample standardization, and factor evidence beyond headline returns.
- Validation/risk/cost review: auditors must not optimize parameters, portfolio risk must not reweight for returns, and cost analysis must reject models that only work under unrealistic friction.
- Orchestration/reporting/quality review: task board is the state source, run manifests capture reproducible facts, decision logs preserve rejected options, and agents share artifacts only.

## V3.8 Baseline Audit Result

- Status: `observation`
- No blocking future-function evidence was found in the V3.8 overlay layer.
- Promotion is not allowed from this audit alone because V3.8 used full-sample OOS candidate ranking, the increment over V3.6 is small, and PBO/DSR/parameter-sensitivity evidence is missing.
- V3.8 remains the continuity comparison baseline until a replacement is accepted, but its evidence quality is explicitly downgraded to observation.

## V3.9 PBO/DSR Result

- Status: `fail for promotion`, `observation for continuity baseline`.
- Block CSCV/PBO failed at 10/20/30bps with PBO around `0.69-0.77`.
- DSR-style selected probability under conservative `N_eff=18` failed at 10/20/30bps.
- Selected V3.8 does not show enough paired stability versus V3.6; yearly delta is near zero and regime delta is negative on average.
- Future model promotion must use nested or purged validation instead of full-sample candidate ranking.

## V3.9 Manifest Result

- Current V3.8 manifest is insufficient for future promotion standards.
- Required going forward: command, argv, code/input/output hashes, environment, git state, artifact inventory, and self-check details.
- Missing strict reproducibility fields should block future production promotion, even if headline performance improves.

## V3.6 Upstream Audit Result

- Status: `fail` for use as V3.8 default upstream baseline source.
- V3.6 uses same-period OOS selection and ex-post state gating based on V3.5 attribution.
- V3.2/V3.4 upstream checks are not clean enough to be silently inherited.
- V3.6 may only be used as an experimental comparison snapshot.

## V3.10 Governance Design

- V3.10 starts by rebuilding a clean baseline, not by adding alpha.
- Full-sample candidate ranking is disallowed for promotion.
- `model_run_manifest.v1` is now the strict manifest standard for future model scripts.

## V3.10 Clean Baseline Result

- Status: accepted as governance control baseline.
- The clean baseline uses predeclared rank-plus-volatility construction, fixed state sleeve budgets, explicit cash, caps, and a turnover cap.
- It excludes V3.6/V3.8 selected target inheritance, same-period OOS candidate selection, and ex-post state gating.
- At 10bps it produced annual return `8.76%`, Sharpe `0.455`, max drawdown `-54.56%`, annual excess versus `000985` of `0.74%`, average trade turnover `0.584`, and average cash `0.195`.
- It passed script self-check, strict model manifest validation, and the agent framework check.
- Remaining warning: the workspace was dirty when the manifest was generated, so promotion remains governance-only until a clean repo state is used for formal release tagging.

## V3.11 Nested Candidate Harness Result

- Status: harness accepted; candidate set rejected for default promotion.
- V3.11 now emits the required validation artifacts: registry, split manifest, purge/embargo audit, inner candidate scores, nested selection history, outer OOS results, same-period baseline comparison, cost sensitivity, PBO splits, gate decision, validation findings, leakage checklist, and robustness summary.
- Selection uses inner validation windows before each outer test year; full-sample candidate metrics are diagnostic only.
- At 10bps, nested selected candidates returned `11.82%` annualized, but same-period V3.10 returned `12.07%`, so the annual delta was `-0.25%`.
- PBO failed at all cost levels, including 10bps PBO `0.369`.
- Current gate decision is `reject_for_default_observation_only`; V3.10 remains the active baseline.

## Subagent Framework Optimization

- Status: accepted.
- Added governance files: `AGENT_WORKFLOW.md`, `RACI_MATRIX.md`, and `AGENT_IO_CONTRACT.md`.
- Updated all 9 agent specs with `Interface Contract` and `Quality Gates` sections.
- Updated active baseline references from V3.8 to V3.10 Clean Rank-Vol Core and made V3.11 the required candidate validation path.
- Clarified decision authority: validation, portfolio, cost, and code-quality agents can block; only chief orchestrator can promote.
- Strengthened `agent_framework_check.py`: it now checks 11 required governance files and 11 required sections per agent.
- Latest framework check: `self_check_pass=true`, `agent_count=9`, `required_file_count=11`, `required_section_count=11`, `run_manifest_count=10`.

## Governance Schema Upgrade

- Status: accepted.
- Added schemas for task briefs, candidate registry, split manifest, and candidate gate decisions.
- Upgraded `model_run_manifest.schema.json` to a standard typed schema file.
- Extended `agent_framework_check.py` to verify accepted task output references and V3.11 structured outputs.
- Latest framework check: `self_check_pass=true`, `agent_count=9`, `required_file_count=15`, `required_section_count=11`.

## V3.12 Candidate Improvement Design

- Status: accepted as design only; no new model promoted.
- Generated 4 candidate directions:
  - `guarded_industry_trend_low_turnover`
  - `baseline_blend_confidence_gate`
  - `valuation_risk_repair_defensive_guard`
  - `pbo_stability_penalized_selector`
- Design response to V3.11 failures:
  - require baseline fallback before deviating;
  - require inner validation margin;
  - reduce turnover and aggressive industry trend tilt;
  - penalize unstable candidates rather than rewarding one-window return.
- Next required task: implement these candidates and rerun V3.11-style nested/purged gates before any default promotion.

## V3.12 Candidate Implementation Harness

- Status: harness accepted; candidates rejected for default promotion.
- Added `strategy_lab/hirssm_v3_12_candidate_implementation_harness.py`.
- The harness implements the V3.12 candidate set, applies a stability-penalized prior-window selector, and keeps full-sample candidate metrics diagnostic only.
- At 10bps, the selector produced annual return `12.07%`, Sharpe `0.594`, max drawdown `-54.56%`, and annual delta versus same-period V3.10 of roughly `-0.003%`.
- PBO remained unacceptable at the important low-cost settings: 5bps `0.579` fail and 10bps `0.484` fail.
- The 20bps and 30bps settings improved marginally versus same-period V3.10, but the edge is too small to justify promotion.
- Self-review fixed a gate-statistic bug: baseline fallback rows are now counted in train-sufficient folds, so nonbaseline selection rates are correctly reported as 5bps `14.3%`, 10bps `33.3%`, 20bps `42.9%`, and 30bps `47.6%`.
- Manifest validation passed with only a dirty-worktree warning; agent framework self-check passed with `run_manifest_count=13`.
- Next required task: V3.13 should perform a failure-revision design pass before writing more alpha code.

## V3.13 Failure Revision Design

- Status: accepted as design only; no model promoted.
- Added `strategy_lab/hirssm_v3_13_failure_revision_design.py`.
- Output directory: `outputs/agent_runs/v3_13/failure_revision_design`.
- V3.13 diagnosis:
  - V3.12 failed because marginal alpha versus V3.10 is too small;
  - low-cost PBO remains unacceptable, especially 10bps PBO `0.484`;
  - selector complexity mostly made the model baseline-equivalent rather than creating independent edge;
  - small expert multiplier variants should be deprioritized.
- V3.14 must start with pre-backtest signal validation and candidate diversity checks.
- Proposed V3.14 candidates:
  - `orthogonal_breadth_regime_overlay`
  - `residual_industry_momentum_low_corr`
  - `value_quality_defensive_barbell`
  - `cost_aware_no_trade_band_overlay`
  - `candidate_diversity_selector`
- Latest framework check: `self_check_pass=true`, `run_manifest_count=14`.

## Subagent Iteration Guardrails

- Status: accepted.
- Root cause addressed:
  - V3.12 exposed that gate metrics could drift from source fold logs when selector statuses changed;
  - V3.13 exposed that a new agent manifest could miss standard fields until the framework check ran.
- Strengthened `strategy_lab/agent_framework_check.py`:
  - validates `artifacts`, `outputs`, and `changed_files` paths in every `agent_run_manifest.json`;
  - validates structured candidate outputs across all candidate harness directories, not only V3.11;
  - recomputes `nonbaseline_selection_rate` from `nested_selection_by_fold.csv` and compares it to `candidate_gate_decision.csv`.
- Updated governance docs:
  - `AGENT_IO_CONTRACT.md` now requires gate metric reconciliation from source rows;
  - `AGENT_WORKFLOW.md` now includes an iteration self-review phase and defect handling procedure;
  - `code_quality_engineer/AGENT.md` now owns source-row metric reconciliation checks.
- Output directory: `outputs/agent_runs/governance/subagent_iteration_guardrails`.
- Latest framework check: `self_check_pass=true`, `run_manifest_count=15`.

## V3.14-V3.18 Five-Version Iteration

- Status: completed; no new default model promoted.
- V3.14:
  - pre-backtest signal research accepted;
  - only `orthogonal_breadth_regime_overlay` passed;
  - residual industry momentum and style barbell stayed observation-only;
  - candidate active-return similarity remained extremely high.
- V3.15:
  - breadth overlay implementation harness accepted;
  - candidates rejected for default promotion;
  - 10bps annual delta versus same-period V3.10 was `-0.154%`;
  - 10bps PBO was `0.611`, fail.
- V3.16:
  - cost-aware stability design accepted;
  - produced 4 high-priority no-trade-band execution overlay specs;
  - explicitly not alpha and not promotable alone.
- V3.17:
  - candidate diversity governance accepted;
  - found 3 near-duplicate active-return pairs;
  - next PBO candidate set should be filtered before implementation.
- V3.18:
  - five-version review accepted;
  - final decision is to keep V3.10 Clean Rank-Vol Core as active governance baseline.
- Optional next step:
  - V3.19 can implement the filtered breadth candidate with no-trade overlay, but must rerun nested/PBO gates from scratch.

## Subagent Iteration Guardrails V2

- Status: accepted as governance-only optimization.
- Root causes addressed:
  - V3.14 runtime warnings were not part of the formal acceptance checklist;
  - V3.15 exposed that model manifests need an explicit self-reference guard;
  - V3.15-V3.18 showed that task-board registration must be checked for every emitted manifest;
  - V3.16-V3.18 showed that design-only versions need a stronger no-promotion boundary.
- Strengthened `strategy_lab/agent_framework_check.py`:
  - every `agent_run_manifest.json` task ID must exist in `AGENT_TASK_BOARD.md`;
  - a manifest tied to a `backlog` task row is a fail-level error;
  - `status=pass` with positive `fail_count` is a fail-level error;
  - model manifests cannot list their own manifest or generated manifest-check file as artifacts.
- Updated governance docs:
  - `AGENT_IO_CONTRACT.md` now defines warning hygiene and model-manifest artifact boundaries;
  - `AGENT_WORKFLOW.md` now defines task-board-first acceptance order;
  - `code_quality_engineer/AGENT.md` now owns warning classification and manifest self-reference checks.
- Output directory: `outputs/agent_runs/governance/subagent_iteration_guardrails_v2`.
- Latest expected framework check: `self_check_pass=true`, `run_manifest_count=21`.

## V3.19-V3.23 Five-Version Iteration

- Status: completed; no new default model promoted.
- V3.19:
  - accepted as blocked governance evidence;
  - strict V3.17 diversity filtering left no nonbaseline candidate for the V3.16 no-trade overlay;
  - implementation was blocked instead of weakening the filter.
- V3.20:
  - accepted as signal research;
  - tested 3 drawdown/reentry signals;
  - only `vol_compression_reentry` passed the full-sample pre-backtest gate;
  - after post-review holdout validation, `holdout_passed_signal_count=0`, so no signal remains implementation-eligible.
- V3.21:
  - implemented `vol_compression_reentry` and `vol_compression_reentry_no_trade_3pct`;
  - after source-gate correction, this harness is diagnostic rejected history only because the source signal failed holdout eligibility;
  - 10bps nested annual return was `12.14%`;
  - 10bps annual delta versus same-period V3.10 was only `0.07%`;
  - 10bps PBO was `0.286`, while 20bps PBO failed at `0.421`;
  - rejected for default promotion.
- V3.22:
  - accepted as failure attribution;
  - concluded that small cash-release overlays are too close to the baseline and too low effect-size for promotion.
- V3.23:
  - accepted as the block review;
  - final decision is to retain V3.10 Clean Rank-Vol Core as the active governance baseline.
- Next step:
  - V3.24 should use a new orthogonal information source before another model harness is allowed.

## Subagent Iteration Guardrails V3

- Status: accepted as governance-only source-gate optimization.
- Root cause addressed:
  - V3.20 forward-label signal validation used 63-day future returns as full-sample diagnostic evidence;
  - V3.21 implementation started from that full-sample-passing signal before a separate holdout gate existed.
- Strengthened `strategy_lab/agent_framework_check.py`:
  - forward-label `signal_validation.csv` now requires `signal_gate_holdout_validation.csv`;
  - `implementation_candidate_spec.csv` cannot set `implementation_allowed=true` unless holdout eligibility is true;
  - signal-gated `candidate_registry.csv` rows cannot remain `candidate` unless the holdout gate passes.
- Updated V3.20:
  - added `signal_gate_holdout_validation.csv`;
  - `holdout_passed_signal_count=0`;
  - `candidate_count=0`;
  - all implementation specs are now `implementation_allowed=false`.
- Updated V3.23:
  - final decision now records that V3.21 is diagnostic-only because its source signal failed holdout.
- Output directory: `outputs/agent_runs/governance/subagent_iteration_guardrails_v3`.
- Latest expected framework check: `self_check_pass=true`, `run_manifest_count=27`.

## V3.24 Valuation Source Research

- Status: accepted as source research only; no implementation candidate.
- Script: `strategy_lab/hirssm_v3_24_valuation_source_research.py`.
- Output directory: `outputs/agent_runs/v3_24/valuation_source_research`.
- New source:
  - historical index valuation spread from CSI index daily data;
  - designed to be independent from the rejected V3.20/V3.21 volatility-compression chain.
- Signals tested:
  - `dividend_valuation_repair`;
  - `broad_market_deep_value_repair`;
  - `large_vs_mid_valuation_spread`.
- Result:
  - full-sample diagnostic pass count: `0`;
  - holdout implementation pass count: `0`;
  - candidate count: `0`.
- Interpretation:
  - the valuation source is orthogonal enough, but the simple valuation-spread triggers do not have enough predictive evidence;
  - V3.25 should not implement V3.24;
  - the next research branch should use a different source family such as macro liquidity, rate spread, constituent breadth, or industry-internal structure.

## V3.25 Industry Structure Source Research

- Status: accepted as source research only; no implementation candidate.
- Script: `strategy_lab/hirssm_v3_25_industry_structure_source_research.py`.
- Output directory: `outputs/agent_runs/v3_25/industry_structure_source_research`.
- New source:
  - SW level-1 industry internal breadth, dispersion, and turnover concentration;
  - designed to be independent from V3.20 volatility compression and V3.24 valuation-spread research.
- Signals tested:
  - `industry_breadth_repair_thrust`;
  - `narrow_leadership_overheat_defense`;
  - `dispersion_rotation_opportunity`.
- Result:
  - full-sample diagnostic pass count: `0`;
  - holdout implementation pass count: `0`;
  - candidate count: `0`.
- Interpretation:
  - the industry-structure source is orthogonal enough, but current trigger definitions are too sparse or weak;
  - `industry_breadth_repair_thrust` had a better holdout profile but failed full-sample sample-size governance;
  - V3.26 should not implement V3.25;
  - the next branch should prioritize macro liquidity/rate spread if usable point-in-time data exists, otherwise move to stock-level breadth instead of level-1 industry aggregates.

## V3.26 Macro Data Readiness

- Status: accepted as data-readiness research only; macro signal research blocked.
- Script: `strategy_lab/hirssm_v3_26_macro_data_readiness.py`.
- Output directory: `outputs/agent_runs/v3_26/macro_data_readiness`.
- Local data scan:
  - CSV files scanned: `160`;
  - required macro/rate series: `10`;
  - usable point-in-time series: `0`.
- Missing core series:
  - `cn_10y_gov_bond_yield`;
  - `us_10y_treasury_yield`;
  - `usdcny`;
  - `china_pmi`.
- Blocked candidate families:
  - `cn_us_rate_spread_risk_budget`;
  - `macro_liquidity_repair`;
  - `inflation_policy_constraint_defense`.
- Interpretation:
  - the framework correctly refused to mine macro factors without `available_date`;
  - this prevents publication-lag leakage, revision bias, and current-snapshot backfill;
  - V3.27 should be a data-steward task to build point-in-time macro/rate tables before any macro factor test.

## V3.27 Macro Data Ingestion

- Status: accepted as data-layer upgrade; no model promotion.
- Script: `strategy_lab/hirssm_v3_27_macro_data_ingestion.py`.
- Output directory: `outputs/agent_runs/v3_27/macro_data_ingestion`.
- Data directory: `data_raw/macro`.
- Ingestion result:
  - source success count: `8`;
  - source failure count: `1`;
  - usable required series: `8`;
  - limited-history required series: `1`;
  - missing required series: `1`.
- Built PIT tables:
  - China 10Y yield, US 10Y yield, China-US 10Y spread, USDCNY;
  - PMI, M2, CPI, PPI;
  - commodity index with limited history;
  - combined `macro_pit_panel.csv`.
- Gate result:
  - `cn_us_rate_spread_risk_budget`: allowed for V3.28 signal validation;
  - `macro_liquidity_repair`: blocked because `china_tsf_yoy` is still missing;
  - `inflation_policy_constraint_defense`: allowed only as limited-history validation because commodity history starts in 2011.
- Interpretation:
  - V3.27 fixed the macro data-layer blocker for the rate/FX branch;
  - it did not authorize any trading model change;
  - V3.28 should validate rate-spread/FX risk-budget signals with the same full-sample diagnostic plus holdout gate used in V3.24/V3.25.

## V3.28 Macro Rate/FX Signal Validation

- Status: accepted as macro signal validation; no model promotion.
- Script: `strategy_lab/hirssm_v3_28_macro_rate_fx_signal_validation.py`.
- Output directory: `outputs/agent_runs/v3_28/macro_rate_fx_signal_validation`.
- Data alignment:
  - uses `data_raw/macro/macro_pit_panel.csv`;
  - merges macro values by `available_date <= signal_date`;
  - uses a one-period monthly feature lag before evaluating forward labels.
- Signals tested:
  - `rate_fx_stress_defense`;
  - `us_rate_shock_fx_stress_defense`;
  - `spread_repair_risk_on`.
- Result:
  - full-sample diagnostic pass count: `3`;
  - holdout implementation pass count: `2`;
  - candidate count: `2`.
- Candidate signals:
  - `us_rate_shock_fx_stress_defense`: holdout forward 63-day mean `-1.13%` versus unconditional `2.89%`;
  - `spread_repair_risk_on`: holdout forward 63-day mean `4.09%` versus unconditional `2.89%`.
- Rejected for implementation:
  - `rate_fx_stress_defense`, because holdout effect reversed and positive-share rose above the defense gate.
- Interpretation:
  - the rate/FX branch is the first recent source branch with holdout-qualified implementation candidates;
  - V3.29 should be a constrained backtest harness, not a default promotion;
  - implementation must still pass cost, turnover, drawdown, nested selection, and PBO-style checks.

## V3.29 Macro Rate/FX Harness

- Status: accepted as constrained implementation validation; rejected for default promotion.
- Script: `strategy_lab/hirssm_v3_29_macro_rate_fx_harness.py`.
- Model output directory: `outputs/hirssm_v3_29_macro_rate_fx_harness`.
- Agent output directory: `outputs/agent_runs/v3_29/macro_rate_fx_harness`.
- Implemented only V3.28 holdout-qualified signals:
  - `us_rate_shock_fx_stress_defense`;
  - `spread_repair_risk_on`.
- 10bps nested OOS:
  - annual return: `11.8650%`;
  - Sharpe: `0.5784`;
  - max drawdown: `-56.2903%`;
  - annual delta versus V3.10: `-0.2083%`;
  - drawdown delta versus V3.10: `-1.7336%`.
- Gate result:
  - 5/10/20bps all fail annual-delta gate;
  - 5/10/20bps all fail PBO gate;
  - overall decision: `reject_for_default_observation_only`.
- Trigger diagnostics:
  - `us_rate_shock_fx_stress_defense`: triggered 20 of 292 months;
  - `spread_repair_risk_on`: triggered 58 of 292 months.
- Interpretation:
  - the macro rate/FX source passed signal validation but did not survive implementation governance;
  - current implementation slightly reduced annual return and worsened drawdown versus V3.10;
  - V3.30 should diagnose whether signal dilution came from cash cap, turnover cap, trigger sparsity, or interaction with V3.10's existing cash budget.

## V3.30 Macro Rate/FX Failure Attribution

- Status: accepted as failure-attribution report; no model promotion.
- Script: `strategy_lab/hirssm_v3_30_macro_rate_fx_failure_attribution.py`.
- Output directory: `outputs/agent_runs/v3_30/macro_rate_fx_failure_attribution`.
- Scope:
  - reads V3.28 signal validation and V3.29 harness outputs;
  - creates bridge, cost-drag, overlay-dilution, selection, PBO, and yearly attribution tables;
  - does not add candidates, tune amplitudes, or change default weights.
- Main findings:
  - 10bps PBO remains high at `0.6111`;
  - V3.29 10bps annual delta versus V3.10 is `-0.2083%`;
  - V3.29 10bps drawdown delta versus V3.10 is `-1.7336%`;
  - macro trigger rate averaged only `13.36%`;
  - average absolute cash delta was only `1.16%`;
  - cost drag from 5bps to 20bps was `0.1492%`, so execution cost is not the main failure source.
- Root-cause ranking:
  - `pbo_instability`;
  - `negative_marginal_oos_performance`;
  - `portfolio_overlay_dilution`;
  - `candidate_selection_not_decisive`;
  - `cost_drag_not_primary`.
- Next experiments:
  - V3.31 selected-year/regime attribution;
  - V3.32 macro risk-budget gate design;
  - repair TSF data before broader macro-liquidity work;
  - block V3.29 amplitude sweeps until a new holdout gate exists.

## V3.31 Selected-Year/Regime Attribution

- Status: accepted as attribution-only; no model promotion.
- Script: `strategy_lab/hirssm_v3_31_selected_year_regime_attribution.py`.
- Output directory: `outputs/agent_runs/v3_31/selected_year_regime_attribution`.
- Scope:
  - aligns V3.29 nested-selected daily NAV with same-day V3.10 baseline returns;
  - maps each day to the selected candidate's latest target-weight state snapshot;
  - decomposes macro-candidate contribution by selected year, regime, and trigger month;
  - does not create a new candidate, tune thresholds, or change default weights.
- 10bps summary:
  - macro candidates selected years: `10`;
  - macro positive-delta rate: `50.00%`;
  - macro average selected-year delta: `-0.0833%`;
  - worst macro selected year: `2010`, `spread_repair_risk_on`, `-1.9340%`.
- Candidate attribution:
  - `spread_repair_risk_on`: selected `7` years, positive-delta rate `42.86%`, average delta `-0.4702%`;
  - `us_rate_shock_fx_stress_defense`: selected `3` years, positive-delta rate `66.67%`, average delta `0.8193%`.
- Failure localization:
  - `spread_repair_risk_on` drove most of the loss;
  - worst years were 2010 and 2008;
  - weak buckets included risk-on-overheat without trigger, risk-off-decline with trigger, and risk-on-trend without trigger.
- Interpretation:
  - V3.31 supports testing macro signals as risk-budget gates rather than small additive overlays;
  - stress-defense may be more promising than risk-on release, but sample count is still too low for promotion;
  - V3.32 must be a predeclared harness with nested/PBO validation, not a parameter sweep.

## V3.32 Macro Risk-Budget Gate

- Status: accepted as validation harness; rejected for default promotion.
- Script: `strategy_lab/hirssm_v3_32_macro_risk_budget_gate.py`.
- Model output directory: `outputs/hirssm_v3_32_macro_risk_budget_gate`.
- Agent output directory: `outputs/agent_runs/v3_32/macro_risk_budget_gate`.
- Candidate design:
  - `stress_budget_gate`: cuts 25% of current risky budget into cash when the rate/FX stress trigger is active;
  - `state_confirmed_dual_budget_gate`: adds conservative repair release only in `range_bound` and `risk_on_trend`;
  - both are predeclared hypotheses, not optimized parameter sweeps.
- 10bps nested OOS:
  - annual return: `11.9782%`;
  - Sharpe: `0.5904`;
  - max drawdown: `-54.5567%`;
  - annual delta versus V3.10: `-0.0951%`;
  - drawdown delta versus V3.10: `0.0000%`.
- Gate result:
  - rejected because annual delta remains negative;
  - 5bps and 10bps PBO improved to observation;
  - 20bps PBO still fails.
- Selection:
  - V3.10 baseline selected `14` years per cost;
  - `stress_budget_gate` selected `6` years per cost;
  - `state_confirmed_dual_budget_gate` selected `1` year per cost.
- Interpretation:
  - risk-budget gating is a better direction than the V3.29 fixed 4pct overlay;
  - it improves PBO and drawdown behavior but still does not beat V3.10;
  - stress-only looks cleaner than the dual repair branch and should be attributed separately in V3.33.

## Subagent Effectiveness Critique V4

- Status: accepted as governance optimization; no model promotion.
- Script: `strategy_lab/subagent_effectiveness_review.py`.
- Output directory: `outputs/agent_runs/governance/subagent_effectiveness_critique_v4`.
- Framework report: `reports/AGENT_FRAMEWORK_EFFECTIVENESS_REVIEW.md`.
- Critical assessment:
  - the multi-agent framework is effective at governance, falsification, and reproducibility;
  - it is not yet effective enough at producing investable alpha;
  - 6 model-producing harnesses were rejected for default promotion;
  - accepted task status can be misread as model promotion if not separated.
- Quantitative scorecard:
  - promoted models: `0`;
  - default-rejected model harnesses: `6`;
  - average 10bps annual delta versus V3.10: `-0.1071%`;
  - best 10bps annual delta versus V3.10: `+0.0686%`;
  - average 10bps PBO: `0.4484`.
- Optimizations implemented:
  - machine-readable task brief directory and current task brief;
  - repeatable subagent effectiveness review script;
  - task brief validation in `agent_framework_check.py`;
  - workflow stop-loss rule after repeated failed model versions;
  - RACI clarification that validation should not own portfolio construction design.
- Next operating rule:
  - use task briefs before more implementation;
  - prefer source/data discovery or independent signal batches if five more model versions fail.

## V3.33 Independent Signal Source Discovery

- Status: accepted as signal/data-source discovery; no model promotion.
- Script: `strategy_lab/hirssm_v3_33_independent_signal_source_discovery.py`.
- Output directory: `outputs/agent_runs/v3_33/independent_signal_source_discovery`.
- Task brief: `strategy_lab/agents/task_briefs/20260527_v3_33_independent_signal_source_discovery.json`.
- Source batch:
  - `12` independent source hypotheses tested;
  - `7` data sources inventoried;
  - `3` data sources blocked for historical backtesting;
  - `0` implementation candidates after strict holdout and orthogonality gates.
- Important near-miss:
  - `trend_breakout_continuation` had positive holdout spread but holdout RankIC was only `0.0016`;
  - it was kept as observation rather than pushed into a harness.
- Blocked data sources:
  - current industry components;
  - latest index weights;
  - limited stock QFQ sample.
- Interpretation:
  - existing index-level data does not currently supply a robust new implementation candidate;
  - the next productive step is data-source repair or expansion, not another portfolio implementation version.

## V3.34 Data Source Repair Audit

- Status: accepted as data governance; no model promotion.
- Script: `strategy_lab/hirssm_v3_34_data_source_repair_audit.py`.
- Output directory: `outputs/agent_runs/v3_34/data_source_repair_audit`.
- Task brief: `strategy_lab/agents/task_briefs/20260527_v3_34_data_source_repair_audit.json`.
- Dataset audit:
  - `8` datasets audited;
  - `3` strict PIT approved datasets;
  - `4` research-only datasets;
  - `1` blocked dataset;
  - `3` current snapshot datasets explicitly restricted from historical backtests.
- Data-steward decisions:
  - index and industry daily price series remain usable for index-level signal validation;
  - macro PIT panel remains usable because it exposes `available_date`;
  - current constituents/latest weights/current industry components remain current-only;
  - sample QFQ stock data is useful for schema and smoke tests, not broad production factor research;
  - sample financial indicators are blocked until announcement or available-date fields are added.
- Repair contracts queued:
  - historical index constituents and weights;
  - broad stock daily raw and adjusted panel;
  - financial indicator point-in-time panel;
  - historical industry classification;
  - macro release calendar and vintage upgrade.
- Interpretation:
  - the bottleneck after V3.33 is not another macro-gate or portfolio parameter tweak;
  - the next meaningful alpha search needs broader and stricter PIT data before factor_researcher opens a new implementation batch.

## V3.35 PIT Data Acquisition Contract

- Status: accepted as dry-run data acquisition contract; no model promotion.
- Script: `strategy_lab/hirssm_v3_35_pit_data_acquisition_contract.py`.
- Config: `configs/pit_data_acquisition_v3_35.json`.
- Output directory: `outputs/agent_runs/v3_35/pit_data_acquisition_contract`.
- Task brief: `strategy_lab/agents/task_briefs/20260527_v3_35_pit_data_acquisition_contract.json`.
- Contract result:
  - `9` dataset contracts;
  - `11` provider endpoint mappings;
  - `154` harvest-plan rows;
  - priority 1 covers historical index weights and index membership;
  - priority 2 covers all-status stock universe, raw daily bars, daily basic fields, adjustment factors, and tradeability flags.
- Credential result:
  - Tushare execute-ready: `False`;
  - JoinQuant execute-ready: `False`;
  - no credentials or tokens are written to outputs.
- Governance boundary:
  - every dataset is `planned_not_acquired`;
  - no dry-run row can enter factor research;
  - acquired data must later pass duplicate, missingness, PIT, raw/adjusted price, lifecycle, and tradeability checks.
- Interpretation:
  - this version converts the V3.34 data bottleneck into an executable acquisition blueprint;
  - the next useful version is a credential/SDK readiness task or a very small live pilot, not another alpha harness.

## V3.36 PIT Data Pilot Readiness

- Status: accepted as readiness and pilot-control workflow; acquisition blocked; no model promotion.
- Script: `strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py`.
- Config: `configs/pit_data_pilot_v3_36.json`.
- Output directory: `outputs/agent_runs/v3_36/pit_data_pilot_readiness`.
- Task brief: `strategy_lab/agents/task_briefs/20260527_v3_36_pit_data_pilot_readiness.json`.
- Environment result:
  - `tushare` installed and import-detected;
  - `jqdatasdk` installed and import-detected;
  - Tushare token not available;
  - JoinQuant username/password not available.
- Pilot-control result:
  - `6` pilot tasks planned;
  - `0` datasets acquired;
  - `6` tasks blocked;
  - execute/network flags remain off;
  - no blocked task is recorded as acquired.
- Pilot scope:
  - Tushare all-status stock universe;
  - Tushare `000300.SH` historical index weight pilot;
  - Tushare raw daily, daily_basic, and adj_factor for `000001.SZ`, `000002.SZ`, `600000.SH`;
  - JoinQuant `000300.XSHG` membership cross-check.
- Interpretation:
  - the code path for a safe pilot is now present;
  - the remaining blocker is credential setup, not model research;
  - factor_researcher should still not consume these outputs.

## V3.37 Credential Bootstrap

- Status: accepted as credential bootstrap and safety guard; no acquisition; no model promotion.
- Script: `strategy_lab/hirssm_v3_37_credential_bootstrap.py`.
- Output directory: `outputs/agent_runs/v3_37/credential_bootstrap`.
- Task brief: `strategy_lab/agents/task_briefs/20260527_v3_37_credential_bootstrap.json`.
- Security changes:
  - `.gitignore` now protects `configs/data_credentials.json`;
  - `.gitignore` now protects `configs/*.local.json` and `*.env.local`;
  - placeholder-only template added at `configs/data_credentials.local.template.json`;
  - explicit pilot execution example added at `configs/pit_data_pilot_v3_36_execute.example.json`.
- Readiness:
  - Tushare SDK installed: pass;
  - Tushare token: blocked;
  - JoinQuant SDK installed: pass;
  - JoinQuant username: blocked;
  - JoinQuant password: blocked.
- Governance boundary:
  - output artifacts contain boolean readiness only;
  - no secrets are printed or stored in reports;
  - no live data acquisition was attempted;
  - V3.36 remains the execution entrypoint after credentials are supplied.
- Interpretation:
  - the system is now ready to accept credentials safely;
  - the remaining work is external credential entry, then a small pilot rerun;
  - there is still no new data source available for factor_researcher.

## V3.36 Explicit Pilot Entry Rerun

- Status: accepted as explicit pilot attempt; acquisition blocked; no model promotion.
- Command: `python -X utf8 strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py --config configs/pit_data_pilot_v3_36_execute.example.json --execute`.
- Output directory: `outputs/agent_runs/v3_36/pit_data_pilot_readiness`.
- Execution flags:
  - `execute = True`;
  - `allow_network = True`.
- Result:
  - `6` pilot tasks planned;
  - `0` datasets acquired;
  - `6` tasks blocked;
  - no blocked task is marked acquired.
- Blocking reason:
  - Tushare SDK installed but token missing;
  - JoinQuant SDK installed but username/password missing.
- Interpretation:
  - the live pilot code path has been exercised up to the credential gate;
  - the remaining blocker is external credentials only;
  - no additional alpha or factor task should start before pilot acquisition succeeds.

## V3.86-V3.92 Quality Rework

- Status: accepted as corrected research-assistant capability layer; no model promotion.
- Quality review report: `reports/HIRSSM_V3_86_TO_V3_92_QUALITY_REVIEW.md`.
- Main defects found:
  - new research-assistant agents were referenced before being registered as formal agents;
  - V3.87 sample objects did not materialize `asof_date`;
  - V3.88 technical method evidence lacked a durable formula spec;
  - V3.89 missed available 000985 daily `pe_ttm` valuation history;
  - V3.90 synthesized technical/fundamental conflicts too aggressively;
  - V3.91 lacked a durable HTML static check artifact;
  - the first new handoff-roster check accidentally treated historical narrative handoffs as active errors.
- Fixes:
  - added three formal AGENT specs and governance docs for research assistant roles;
  - expanded framework checks for registered `assigned_agent` and V3.86+ exact `next_handoff`;
  - regenerated V3.86-V3.92 outputs with corrected schema, valuation fallback, technical formula spec, conflict caps, HTML static checks, and quality findings.
- Verification:
  - syntax checks passed;
  - V3.86-V3.92 self-check fail counts are all `0`;
  - global agent framework check passed with active errors `0`.
- Interpretation:
  - the framework is now better as a quant research assistant;
  - the sample views are more conservative and better traced;
  - next useful work is adding stock-level PIT fundamentals and a dedicated analyst-style report template, not treating V3.92 views as alpha.
