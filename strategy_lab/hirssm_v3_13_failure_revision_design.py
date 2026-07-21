#!/usr/bin/env python
"""HIRSSM V3.13 failure revision design.

This is a chief-orchestrator design pass after V3.12. It does not backtest or
promote a model. It diagnoses why V3.12 remained baseline-equivalent and why
low-cost PBO failed, then emits predeclared V3.14 research briefs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
V312_DIR = ROOT / "outputs" / "hirssm_v3_12_candidate_implementation_harness"
V312_AGENT_DIR = ROOT / "outputs" / "agent_runs" / "v3_12" / "candidate_implementation_harness"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_13" / "failure_revision_design"
TASK_ID = "20260526_v3_13_failure_revision_design"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def candidate_registry() -> pd.DataFrame:
    rows = [
        {
            "variant": BASELINE_VARIANT,
            "role": "control",
            "description": "Frozen V3.10 clean rank-vol governance baseline.",
            "multipliers_json": "{}",
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "existing_control",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": False,
            "implementation_status": "existing_control",
            "fixed_inputs": "outputs/hirssm_v3_10_clean_baseline; configs/hirssm_v2_default.json",
            "fixed_outputs": "same-period baseline comparison only",
            "forbidden": "do not retune baseline after seeing candidate results",
            "acceptance_standard": "control reference, not promotion candidate",
        },
        {
            "variant": "orthogonal_breadth_regime_overlay",
            "role": "candidate",
            "description": "Use market and industry breadth repair/deterioration to change risk budget instead of small expert multipliers.",
            "multipliers_json": json.dumps(
                {
                    "signal_family": "breadth_dispersion_regime",
                    "budget_overlay": True,
                    "threshold_source": "predeclared_or_nested_only",
                    "max_cash": 0.40,
                    "turnover_cap": 0.55,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_13_failure_revision_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
            "fixed_inputs": "historical index prices; industry index prices; regime table; no components required",
            "fixed_outputs": "breadth_state.csv; target_weights; nested/PBO reports",
            "forbidden": "no full-sample threshold search; no current constituent weights; no macro data without available_date",
            "acceptance_standard": "10bps annual_delta_vs_v310 >= 0.50pct or drawdown improves >= 3pct with annual loss <= 1pct; PBO <= 0.35",
        },
        {
            "variant": "residual_industry_momentum_low_corr",
            "role": "candidate",
            "description": "Rank industries by residual momentum versus broad beta with breadth confirmation and low turnover.",
            "multipliers_json": json.dumps(
                {
                    "signal_family": "residual_industry_momentum",
                    "residual_benchmark": "000985",
                    "lookbacks": [60, 120],
                    "max_industry_holdings": 2,
                    "turnover_cap": 0.50,
                    "active_return_corr_max_vs_baseline": 0.85,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_13_failure_revision_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
            "fixed_inputs": "industry daily returns; 000985 daily returns; prior-window breadth confirmation",
            "fixed_outputs": "factor_validation.csv; target_weights; nested_fold_scores.csv",
            "forbidden": "no same-year OOS ranking; no selecting lookback from full sample; no leverage",
            "acceptance_standard": "positive fold hit rate > 55pct, 10bps annual_delta_vs_v310 >= 0.50pct, 20bps delta >= 0",
        },
        {
            "variant": "value_quality_defensive_barbell",
            "role": "candidate",
            "description": "Shift from broad industry timing to a style barbell: dividend/SSE50 defense plus CSI500/1000 only after valuation repair.",
            "multipliers_json": json.dumps(
                {
                    "signal_family": "style_valuation_risk_barbell",
                    "defensive_assets": ["000922", "000016"],
                    "growth_assets": ["000905", "000852"],
                    "valuation_inputs": ["style_pe", "style_pb"],
                    "industry_weight_cap": 0.15,
                    "turnover_cap": 0.45,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_13_failure_revision_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
            "fixed_inputs": "style daily returns; style PE/PB history; regime table",
            "fixed_outputs": "style_barbell_scores.csv; target_weights; cost_sensitivity.csv",
            "forbidden": "no industry valuation snapshots; no full-sample valuation percentile retuning",
            "acceptance_standard": "max drawdown improves >= 3pct or Calmar improves, while 10bps annual_delta_vs_v310 >= -0.50pct",
        },
        {
            "variant": "cost_aware_no_trade_band_overlay",
            "role": "candidate",
            "description": "Execution overlay that skips marginal trades unless expected edge clears cost and stability bands.",
            "multipliers_json": json.dumps(
                {
                    "signal_family": "execution_cost_control",
                    "min_weight_change_grid": [0.03, 0.05],
                    "grid_selection": "nested_only",
                    "monthly_turnover_target_cap": 0.45,
                    "not_alpha_candidate": True,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_13_failure_revision_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
            "fixed_inputs": "V3.10 targets; V3.14 candidate targets; cost assumptions",
            "fixed_outputs": "turnover_diagnostics.csv; cost_sensitivity.csv; target_weights",
            "forbidden": "do not claim alpha from cost-only improvement; do not optimize on 30bps only",
            "acceptance_standard": "20/30bps improve without making 5/10bps materially worse; turnover falls at least 15pct",
        },
        {
            "variant": "candidate_diversity_selector",
            "role": "candidate",
            "description": "Selector-level governance: remove near-duplicate active-return candidates before PBO and require independent evidence.",
            "multipliers_json": json.dumps(
                {
                    "signal_family": "selector_governance",
                    "active_return_corr_max": 0.90,
                    "min_independent_candidate_count": 2,
                    "baseline_fallback": True,
                    "pbo_target": 0.35,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_13_failure_revision_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
            "fixed_inputs": "candidate navs; active return correlation matrix; nested fold scores",
            "fixed_outputs": "candidate_similarity_matrix.csv; selector_decision_log.csv; pbo_report.csv",
            "forbidden": "do not increase candidate count without diversity evidence; do not use PBO result to choose the same run's candidates",
            "acceptance_standard": "PBO <= 0.35 at 5/10/20bps and nonbaseline selections retain positive same-period delta",
        },
    ]
    return pd.DataFrame(rows)


def raw_candidate_edge(v312_dir: Path) -> pd.DataFrame:
    metrics = read_csv(v312_dir / "candidate_full_sample_diagnostic_metrics.csv")
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for cost, group in metrics.groupby("cost_bps"):
        baseline = group[group["variant"].eq(BASELINE_VARIANT)]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            rows.append(
                {
                    "cost_bps": float(cost),
                    "variant": row["variant"],
                    "diagnostic_full_sample_only": True,
                    "annual_delta_vs_raw_baseline": float(row["annual_return"] - base["annual_return"]),
                    "sharpe_delta_vs_raw_baseline": float(row["sharpe_no_rf"] - base["sharpe_no_rf"]),
                    "drawdown_delta_vs_raw_baseline": float(row["max_drawdown"] - base["max_drawdown"]),
                    "turnover_delta_vs_raw_baseline": float(row["avg_trade_turnover"] - base["avg_trade_turnover"]),
                    "cash_delta_vs_raw_baseline": float(row["avg_cash_weight"] - base["avg_cash_weight"]),
                }
            )
    return pd.DataFrame(rows)


def fold_stability(v312_dir: Path) -> pd.DataFrame:
    scores = read_csv(v312_dir / "nested_fold_scores.csv")
    if scores.empty:
        return pd.DataFrame()
    baseline = scores[scores["variant"].eq(BASELINE_VARIANT)][
        [
            "cost_bps",
            "test_year",
            "inner_validation_score",
            "oos_score",
            "inner_validation_annual_return",
            "oos_annual_return",
            "oos_max_drawdown",
        ]
    ].rename(
        columns={
            "inner_validation_score": "baseline_inner_score",
            "oos_score": "baseline_oos_score",
            "inner_validation_annual_return": "baseline_inner_annual",
            "oos_annual_return": "baseline_oos_annual",
            "oos_max_drawdown": "baseline_oos_drawdown",
        }
    )
    merged = scores.merge(baseline, on=["cost_bps", "test_year"], how="left")
    merged["inner_score_delta"] = merged["inner_validation_score"] - merged["baseline_inner_score"]
    merged["oos_score_delta"] = merged["oos_score"] - merged["baseline_oos_score"]
    merged["inner_annual_delta"] = merged["inner_validation_annual_return"] - merged["baseline_inner_annual"]
    merged["oos_annual_delta"] = merged["oos_annual_return"] - merged["baseline_oos_annual"]
    merged["oos_drawdown_delta"] = merged["oos_max_drawdown"] - merged["baseline_oos_drawdown"]
    rows = []
    for (cost, variant), group in merged.groupby(["cost_bps", "variant"]):
        if variant == BASELINE_VARIANT:
            continue
        rows.append(
            {
                "cost_bps": float(cost),
                "variant": variant,
                "fold_count": int(group.shape[0]),
                "inner_positive_rate": float((group["inner_annual_delta"] > 0).mean()),
                "oos_positive_rate": float((group["oos_annual_delta"] > 0).mean()),
                "inner_oos_sign_agreement_rate": float(((group["inner_annual_delta"] > 0) == (group["oos_annual_delta"] > 0)).mean()),
                "mean_inner_annual_delta": float(group["inner_annual_delta"].mean()),
                "mean_oos_annual_delta": float(group["oos_annual_delta"].mean()),
                "median_oos_annual_delta": float(group["oos_annual_delta"].median()),
                "worst_oos_annual_delta": float(group["oos_annual_delta"].min()),
                "mean_oos_drawdown_delta": float(group["oos_drawdown_delta"].mean()),
                "selected_rate": float(pd.Series(group.get("selected", False)).astype(bool).mean()),
                "proposed_rate": float(pd.Series(group.get("proposed", False)).astype(bool).mean()),
                "mean_stability_penalty": float(group.get("stability_penalty", pd.Series(dtype=float)).mean()),
            }
        )
    return pd.DataFrame(rows)


def selector_behavior(v312_dir: Path) -> pd.DataFrame:
    selection = read_csv(v312_dir / "nested_selection_by_fold.csv")
    if selection.empty:
        return pd.DataFrame()
    usable = selection[selection["selection_status"].eq("selected_by_prior_window")].copy()
    rows = []
    for cost, group in usable.groupby("cost_bps"):
        selected_counts = group["selected_variant"].value_counts()
        proposed_counts = group["proposed_variant"].value_counts() if "proposed_variant" in group else pd.Series(dtype=int)
        fallback_used = pd.Series(group.get("baseline_fallback_used", False)).fillna(False).astype(bool)
        rows.append(
            {
                "cost_bps": float(cost),
                "eligible_years": int(group.shape[0]),
                "nonbaseline_selection_rate": float((group["selected_variant"] != BASELINE_VARIANT).mean()),
                "baseline_fallback_rate": float(fallback_used.mean()),
                "selected_distribution_json": json.dumps(selected_counts.to_dict(), sort_keys=True),
                "proposed_distribution_json": json.dumps(proposed_counts.to_dict(), sort_keys=True),
                "mean_selected_oos_minus_baseline_annual": float(group["selected_oos_minus_baseline_annual"].mean()),
                "positive_oos_delta_rate": float((group["selected_oos_minus_baseline_annual"] > 0).mean()),
                "mean_selected_oos_minus_baseline_drawdown": float(group["selected_oos_minus_baseline_drawdown"].mean()),
            }
        )
    return pd.DataFrame(rows)


def failure_diagnosis(v312_dir: Path) -> pd.DataFrame:
    gate = read_csv(v312_dir / "candidate_gate_decision.csv")
    pbo = read_csv(v312_dir / "pbo_cscv_summary.csv")
    behavior = selector_behavior(v312_dir)
    stability = fold_stability(v312_dir)
    rows = []
    if not gate.empty:
        for _, row in gate.iterrows():
            annual_delta = float(row.get("annual_delta_vs_v310", np.nan))
            rows.append(
                {
                    "source": "candidate_gate_decision",
                    "cost_bps": row.get("cost_bps"),
                    "issue": "marginal_alpha_too_small",
                    "severity": "fail" if annual_delta < 0.005 else "observation",
                    "evidence": f"annual_delta_vs_v310={annual_delta:.6f}; sharpe_delta={float(row.get('sharpe_delta_vs_v310', np.nan)):.6f}; decision={row.get('decision')}",
                    "design_response": "stop small multiplier variants; require orthogonal signal family or explicit risk-control objective",
                }
            )
            if bool(row.get("pbo_not_fail")) is False:
                rows.append(
                    {
                        "source": "candidate_gate_decision",
                        "cost_bps": row.get("cost_bps"),
                        "issue": "promotion_blocked_by_pbo",
                        "severity": "fail",
                        "evidence": f"pbo={float(row.get('pbo', np.nan)):.6f}",
                        "design_response": "candidate set must be smaller, less correlated, and prefiltered for active-return diversity before PBO",
                    }
                )
    if not pbo.empty:
        low_cost = pbo[pbo["cost_bps"].astype(float).isin([5.0, 10.0])]
        if not low_cost.empty and (low_cost["pbo"].astype(float) > 0.35).any():
            rows.append(
                {
                    "source": "pbo_cscv_summary",
                    "cost_bps": "5/10",
                    "issue": "low_cost_pbo_failure",
                    "severity": "fail",
                    "evidence": "; ".join(f"{float(r.cost_bps):.0f}bps={float(r.pbo):.3f}" for r in low_cost.itertuples()),
                    "design_response": "treat 5/10bps as primary; 20/30bps pass is not enough because alpha should not appear only after high costs",
                }
            )
    if not behavior.empty:
        for _, row in behavior.iterrows():
            rows.append(
                {
                    "source": "nested_selection_by_fold",
                    "cost_bps": row["cost_bps"],
                    "issue": "selector_falls_back_often_and_edge_is_flat",
                    "severity": "observation",
                    "evidence": f"nonbaseline_selection_rate={row['nonbaseline_selection_rate']:.3f}; fallback_rate={row['baseline_fallback_rate']:.3f}; mean_oos_delta={row['mean_selected_oos_minus_baseline_annual']:.6f}",
                    "design_response": "separate alpha discovery from selector governance; measure factor edge before adding selector complexity",
                }
            )
    if not stability.empty:
        weak = stability[(stability["cost_bps"].astype(float).eq(10.0)) & (stability["oos_positive_rate"] < 0.55)]
        for _, row in weak.iterrows():
            rows.append(
                {
                    "source": "nested_fold_scores",
                    "cost_bps": row["cost_bps"],
                    "issue": "candidate_fold_hit_rate_below_threshold",
                    "severity": "fail",
                    "evidence": f"{row['variant']} oos_positive_rate={row['oos_positive_rate']:.3f}; mean_oos_delta={row['mean_oos_annual_delta']:.6f}",
                    "design_response": "reject minor expert multiplier candidate; require a new economic signal and fold hit-rate gate",
                }
            )
    return pd.DataFrame(rows)


def candidate_hypotheses(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in registry[registry["role"].eq("candidate")].iterrows():
        rows.append(
            {
                "variant": row["variant"],
                "economic_hypothesis": {
                    "orthogonal_breadth_regime_overlay": "A-share index returns are regime-convex; breadth deterioration and repair can control beta exposure better than tiny factor-weight changes.",
                    "residual_industry_momentum_low_corr": "Industry leadership persists after removing broad beta, but only when confirmed by breadth and controlled turnover.",
                    "value_quality_defensive_barbell": "Valuation repair works more reliably as a style allocation and drawdown-control sleeve than as a broad industry multiplier.",
                    "cost_aware_no_trade_band_overlay": "Many monthly reallocations are too small to survive costs; a no-trade band can improve after-cost robustness without claiming alpha.",
                    "candidate_diversity_selector": "PBO improves only if candidate alternatives are independent; near-duplicate variants create false selection confidence.",
                }.get(row["variant"], "control"),
                "primary_failure_addressed": {
                    "orthogonal_breadth_regime_overlay": "low marginal alpha from expert multipliers",
                    "residual_industry_momentum_low_corr": "candidate similarity and unstable industry trend timing",
                    "value_quality_defensive_barbell": "drawdown/risk objective not separated from alpha objective",
                    "cost_aware_no_trade_band_overlay": "edge disappears under realistic costs",
                    "candidate_diversity_selector": "PBO failure from correlated candidate pool",
                }.get(row["variant"], "baseline"),
                "first_test": {
                    "orthogonal_breadth_regime_overlay": "standalone breadth state RankIC and regime return attribution before full backtest",
                    "residual_industry_momentum_low_corr": "cross-sectional industry residual momentum IC and fold hit rate",
                    "value_quality_defensive_barbell": "style valuation spread repair test and drawdown attribution",
                    "cost_aware_no_trade_band_overlay": "turnover/cost ablation on unchanged alpha weights",
                    "candidate_diversity_selector": "active-return correlation matrix and PBO with duplicate filtering",
                }.get(row["variant"], "same-period baseline"),
                "kill_rule": row["acceptance_standard"],
            }
        )
    return pd.DataFrame(rows)


def validation_plan() -> pd.DataFrame:
    rows = [
        {
            "gate": "pre_backtest_signal_validation",
            "owner": "factor_researcher",
            "required_artifact": "factor_validation.csv",
            "pass_rule": "signal has prior-window IC or economic risk-control evidence before portfolio backtest",
        },
        {
            "gate": "candidate_diversity",
            "owner": "chief_quant_orchestrator",
            "required_artifact": "candidate_similarity_matrix.csv",
            "pass_rule": "active-return correlation between non-control candidates <= 0.90 or one is observation-only",
        },
        {
            "gate": "nested_oos",
            "owner": "backtest_validation_auditor",
            "required_artifact": "nested_selection_by_fold.csv",
            "pass_rule": "10bps annual_delta_vs_v310 >= 0.005, or drawdown improves >= 0.03 with annual_delta >= -0.01",
        },
        {
            "gate": "low_cost_pbo",
            "owner": "backtest_validation_auditor",
            "required_artifact": "pbo_cscv_summary.csv",
            "pass_rule": "5bps and 10bps PBO <= 0.35; promotion target <= 0.20",
        },
        {
            "gate": "cost_robustness",
            "owner": "execution_cost_analyst",
            "required_artifact": "cost_sensitivity.csv",
            "pass_rule": "20bps annual_delta_vs_v310 >= 0 and 30bps not worse by more than 0.50pct",
        },
        {
            "gate": "risk_budget_integrity",
            "owner": "portfolio_risk_engineer",
            "required_artifact": "constraint_check.csv",
            "pass_rule": "no negative weights; no leverage; cash cap respected; single style/industry caps respected",
        },
        {
            "gate": "manifest_and_reproducibility",
            "owner": "code_quality_engineer",
            "required_artifact": "model_run_manifest.json",
            "pass_rule": "strict manifest validation passes; dirty-worktree is warning only for research, blocker for formal release",
        },
    ]
    return pd.DataFrame(rows)


def self_check(
    *,
    registry: pd.DataFrame,
    diagnosis: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    required_inputs: list[Path],
) -> pd.DataFrame:
    rows = [
        {
            "check": "required_inputs_exist",
            "status": "pass" if all(path.exists() for path in required_inputs) else "fail",
            "detail": "; ".join(str(path) for path in required_inputs if not path.exists()) or "all inputs exist",
        },
        {
            "check": "candidate_registry_has_control_and_candidates",
            "status": "pass" if BASELINE_VARIANT in set(registry["variant"]) and int((registry["role"] == "candidate").sum()) >= 3 else "fail",
            "detail": f"rows={registry.shape[0]}; candidates={int((registry['role'] == 'candidate').sum())}",
        },
        {
            "check": "failure_diagnosis_not_empty",
            "status": "pass" if not diagnosis.empty else "fail",
            "detail": f"rows={diagnosis.shape[0]}",
        },
        {
            "check": "validation_plan_has_low_cost_pbo_gate",
            "status": "pass" if "low_cost_pbo" in set(validation["gate"]) else "fail",
            "detail": ",".join(validation["gate"].astype(str).tolist()),
        },
        {
            "check": "design_only_no_promotion",
            "status": "pass",
            "detail": "V3.13 emits designs and task briefs only",
        },
        {
            "check": "output_dir_scoped",
            "status": "pass" if str(output_dir.as_posix()).endswith("outputs/agent_runs/v3_13/failure_revision_design") else "fail",
            "detail": str(output_dir.as_posix()),
        },
    ]
    return pd.DataFrame(rows)


def make_agent_report(
    *,
    diagnosis: pd.DataFrame,
    registry: pd.DataFrame,
    behavior: pd.DataFrame,
    pbo: pd.DataFrame,
) -> str:
    fail_count = int((diagnosis["severity"] == "fail").sum()) if not diagnosis.empty else 0
    candidate_count = int((registry["role"] == "candidate").sum()) if not registry.empty else 0
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
    behavior10 = behavior[behavior["cost_bps"].astype(float).eq(10.0)].head(1) if not behavior.empty else pd.DataFrame()
    lines = [
        "# HIRSSM V3.13 Failure Revision Design",
        "",
        "## Purpose",
        "",
        "Diagnose V3.12 failure and define the next candidate research brief before implementation.",
        "",
        "## Findings",
        "",
        f"- Failure findings: {fail_count}",
        "- V3.12 is effectively baseline-equivalent at 10bps and does not create a robust alpha edge.",
        "- Low-cost PBO remains the main blocker.",
    ]
    if not pbo10.empty:
        lines.append(f"- 10bps PBO: {float(pbo10['pbo'].iloc[0]):.6f} ({pbo10['pbo_status'].iloc[0]})")
    if not behavior10.empty:
        lines.append(f"- 10bps nonbaseline selection rate: {float(behavior10['nonbaseline_selection_rate'].iloc[0]):.6f}")
        lines.append(f"- 10bps baseline fallback rate: {float(behavior10['baseline_fallback_rate'].iloc[0]):.6f}")
    lines.extend(
        [
            "",
            "## V3.14 Direction",
            "",
            f"- Candidate designs: {candidate_count}",
            "- Move away from minor expert multiplier tweaks.",
            "- Require orthogonal signal validation before full backtest.",
            "- Add candidate diversity checks before PBO.",
            "- Keep V3.10 as active baseline until a candidate passes low-cost PBO and same-period OOS gates.",
        ]
    )
    return "\n".join(lines)


def make_task_brief() -> str:
    return "\n".join(
        [
            "# V3.14 Task Brief - Orthogonal Candidate Research",
            "",
            "- Task ID: `20260526_v3_14_orthogonal_candidate_research`",
            "- Owner: `factor_researcher`",
            "- Dependency: `20260526_v3_13_failure_revision_design`",
            "- Baseline: `HIRSSM V3.10 Clean Rank-Vol Core`",
            "",
            "## Fixed Inputs",
            "",
            "- `outputs/agent_runs/v3_13/failure_revision_design/candidate_registry.csv`",
            "- `outputs/agent_runs/v3_13/failure_revision_design/failure_diagnosis.csv`",
            "- `outputs/hirssm_v3_12_candidate_implementation_harness/`",
            "- `configs/hirssm_v2_default.json`",
            "- historical style and industry index data under `data_raw/index/`",
            "",
            "## Required Outputs",
            "",
            "- `factor_validation.csv`",
            "- `candidate_similarity_matrix.csv`",
            "- `candidate_implementation_spec.csv`",
            "- `agent_report.md`",
            "- `agent_run_manifest.json`",
            "",
            "## Forbidden",
            "",
            "- Do not backtest a portfolio candidate before pre-backtest signal validation.",
            "- Do not use full-sample thresholds, same-period OOS ranking, or current constituents.",
            "- Do not promote a cost-only overlay as alpha.",
            "",
            "## Acceptance Criteria",
            "",
            "- At least two candidate signals have explicit economic logic and measurable orthogonality to V3.10.",
            "- Every implementation spec includes fixed inputs, outputs, prohibited data, and kill rules.",
            "- Low-cost PBO and same-period V3.10 comparison remain mandatory for later implementation.",
        ]
    )


def make_agent_manifest(
    *,
    start_time: str,
    output_dir: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "chief_quant_orchestrator",
        "version": "V3.13",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_13_failure_revision_design.py",
        "config": {
            "config_path": "configs/hirssm_v2_default.json",
            "baseline_variant": BASELINE_VARIANT,
            "design_only": True,
        },
        "data_refs": [
            "outputs/hirssm_v3_12_candidate_implementation_harness",
            "outputs/agent_runs/v3_12/candidate_implementation_harness",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_13_failure_revision_design.py",
            "strategy_lab/hirssm_v3_12_candidate_implementation_harness.py",
            "strategy_lab/agent_framework_check.py",
        ],
        "allowed_inputs": [
            "outputs/hirssm_v3_12_candidate_implementation_harness",
            "outputs/agent_runs/v3_12/candidate_implementation_harness",
        ],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Design-only pass; no new model return is produced.",
            "Candidate hypotheses require V3.14 validation before implementation.",
        ],
        "risk_flags": ["v3_12_low_cost_pbo_failed", "v3_12_alpha_marginal_vs_v310"],
        "next_decision": "Assign V3.14 to factor_researcher for pre-backtest orthogonal signal validation.",
        "handoff_summary": "V3.13 diagnosed V3.12 failure and produced candidate designs plus a V3.14 task brief.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HIRSSM V3.13 failure revision design artifacts.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--v312-dir", default=str(V312_DIR))
    parser.add_argument("--v312-agent-dir", default=str(V312_AGENT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    root = Path(args.root)
    v312_dir = Path(args.v312_dir)
    v312_agent_dir = Path(args.v312_agent_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = now_text()

    registry = candidate_registry()
    raw_edge = raw_candidate_edge(v312_dir)
    stability = fold_stability(v312_dir)
    behavior = selector_behavior(v312_dir)
    diagnosis = failure_diagnosis(v312_dir)
    hypotheses = candidate_hypotheses(registry)
    validation = validation_plan()
    pbo = read_csv(v312_dir / "pbo_cscv_summary.csv")

    required_inputs = [
        v312_dir / "candidate_gate_decision.csv",
        v312_dir / "pbo_cscv_summary.csv",
        v312_dir / "nested_fold_scores.csv",
        v312_agent_dir / "agent_run_manifest.json",
    ]
    checks = self_check(
        registry=registry,
        diagnosis=diagnosis,
        validation=validation,
        output_dir=output_dir,
        required_inputs=required_inputs,
    )

    registry_path = output_dir / "candidate_registry.csv"
    registry_json_path = output_dir / "candidate_registry.json"
    raw_edge_path = output_dir / "raw_candidate_edge_by_cost.csv"
    stability_path = output_dir / "fold_stability_by_variant.csv"
    behavior_path = output_dir / "selector_behavior_summary.csv"
    diagnosis_path = output_dir / "failure_diagnosis.csv"
    hypotheses_path = output_dir / "candidate_hypotheses.csv"
    validation_path = output_dir / "validation_plan.csv"
    task_brief_path = output_dir / "task_brief_v3_14.md"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    self_check_report_path = output_dir / "SELF_CHECK_REPORT.md"
    changed_files_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"

    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    write_json({"candidates": registry.to_dict(orient="records")}, registry_json_path)
    raw_edge.to_csv(raw_edge_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
    behavior.to_csv(behavior_path, index=False, encoding="utf-8-sig")
    diagnosis.to_csv(diagnosis_path, index=False, encoding="utf-8-sig")
    hypotheses.to_csv(hypotheses_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    write_text(make_task_brief(), task_brief_path)
    write_text(make_agent_report(diagnosis=diagnosis, registry=registry, behavior=behavior, pbo=pbo), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    write_text(
        "\n".join(
            [
                "# HIRSSM V3.13 Self Check",
                "",
                f"- Failures: {fail_count}",
                f"- Warnings: {warn_count}",
                "",
                "## Checks",
                "",
                *[f"- {row.check}: {row.status} ({row.detail})" for row in checks.itertuples()],
            ]
        ),
        self_check_report_path,
    )

    artifacts = [
        registry_path,
        registry_json_path,
        raw_edge_path,
        stability_path,
        behavior_path,
        diagnosis_path,
        hypotheses_path,
        validation_path,
        task_brief_path,
        report_path,
        checks_path,
        self_check_report_path,
        changed_files_path,
        manifest_path,
    ]
    write_text("\n".join(str(path.relative_to(root).as_posix()) for path in artifacts), changed_files_path)
    metrics = {
        "candidate_design_count": int((registry["role"] == "candidate").sum()),
        "failure_count": int((diagnosis["severity"] == "fail").sum()) if not diagnosis.empty else 0,
        "pbo_10bps": float(pbo[pbo["cost_bps"].astype(float).eq(10.0)]["pbo"].iloc[0]) if not pbo.empty and pbo["cost_bps"].astype(float).eq(10.0).any() else np.nan,
        "nonbaseline_selection_rate_10bps": float(behavior[behavior["cost_bps"].astype(float).eq(10.0)]["nonbaseline_selection_rate"].iloc[0]) if not behavior.empty and behavior["cost_bps"].astype(float).eq(10.0).any() else np.nan,
        "baseline_fallback_rate_10bps": float(behavior[behavior["cost_bps"].astype(float).eq(10.0)]["baseline_fallback_rate"].iloc[0]) if not behavior.empty and behavior["cost_bps"].astype(float).eq(10.0).any() else np.nan,
    }
    manifest = make_agent_manifest(
        start_time=start_time,
        output_dir=output_dir,
        artifacts=artifacts,
        metrics=metrics,
        fail_count=fail_count,
        warn_count=warn_count,
    )
    write_json(manifest, manifest_path)
    result = {
        "task_id": TASK_ID,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
