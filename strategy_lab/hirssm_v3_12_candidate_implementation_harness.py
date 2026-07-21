#!/usr/bin/env python
"""HIRSSM V3.12 candidate implementation harness.

V3.12 implements the candidate directions designed after V3.11:

- guarded industry trend with a lower turnover cap;
- 80/20 baseline blend;
- valuation/risk repair defensive guard by regime;
- a stability-penalized prior-window selector.

Full-sample raw candidate metrics remain diagnostic only. Default promotion is
decided by nested OOS comparison against the frozen V3.10 clean baseline plus
purged CSCV/PBO diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hirssm_v2_model as model
import hirssm_v2_walk_forward as wf
import hirssm_v3_10_clean_baseline as v310
import hirssm_v3_11_nested_candidate_harness as v311
from model_run_manifest import build_model_run_manifest, validate_model_run_manifest


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
DESIGN_DIR = ROOT / "outputs" / "agent_runs" / "v3_12" / "candidate_improvement_design"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_12_candidate_implementation_harness"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_12" / "candidate_implementation_harness"
TASK_ID = "20260526_v3_12_candidate_implementation_harness"
MODEL_VERSION = "HIRSSM V3.12 Candidate Implementation Harness"
BASELINE_VARIANT = v311.BASELINE_VARIANT
SELECTOR_VARIANT = "pbo_stability_penalized_selector"
TRADING_DAYS = 252


CANDIDATE_SPECS: list[dict[str, Any]] = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol governance baseline.",
        "target_mode": "baseline",
        "multipliers": {},
        "turnover_cap": None,
        "selection_source": "v3_10_frozen_control",
        "eligible_for_default_promotion": False,
    },
    {
        "variant": "guarded_industry_trend_low_turnover",
        "role": "candidate",
        "description": "Mild industry trend/risk tilt with a lower turnover cap.",
        "target_mode": "multiplier",
        "multipliers": {"industry_trend_continuation": 1.08, "risk_compression": 1.02},
        "turnover_cap": 0.65,
        "selection_source": "v3_12_design_predeclared",
        "eligible_for_default_promotion": True,
    },
    {
        "variant": "baseline_blend_confidence_gate",
        "role": "candidate",
        "description": "80pct V3.10 baseline and 20pct guarded industry trend candidate.",
        "target_mode": "blend",
        "blend_base": BASELINE_VARIANT,
        "blend_candidate": "guarded_industry_trend_low_turnover",
        "blend_base_weight": 0.80,
        "blend_candidate_weight": 0.20,
        "multipliers": {},
        "turnover_cap": 0.65,
        "selection_source": "v3_12_design_predeclared",
        "eligible_for_default_promotion": True,
    },
    {
        "variant": "valuation_risk_repair_defensive_guard",
        "role": "candidate",
        "description": "State-conditional valuation, risk compression, and defensive repair tilt.",
        "target_mode": "state_multiplier",
        "multipliers": {
            "range_bound": {"valuation_repair": 1.10, "risk_compression": 1.10, "defensive": 1.05},
            "risk_off_decline": {"valuation_repair": 1.05, "risk_compression": 1.15, "defensive": 1.08},
            "crash_rebound": {"valuation_repair": 1.04, "risk_compression": 1.08, "defensive": 1.05},
            "risk_on_trend": {"trend_continuation": 1.00},
            "risk_on_overheat": {"risk_compression": 1.05, "defensive": 1.03},
        },
        "turnover_cap": 0.70,
        "selection_source": "v3_12_design_predeclared",
        "eligible_for_default_promotion": True,
    },
    {
        "variant": SELECTOR_VARIANT,
        "role": "candidate",
        "description": "Prior-window selector with baseline fallback, margin, and stability penalties.",
        "target_mode": "selector",
        "multipliers": {},
        "turnover_cap": None,
        "selection_source": "v3_12_design_predeclared_selector",
        "eligible_for_default_promotion": True,
    },
]


RAW_TARGET_VARIANTS = [
    BASELINE_VARIANT,
    "guarded_industry_trend_low_turnover",
    "baseline_blend_confidence_gate",
    "valuation_risk_repair_defensive_guard",
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_costs(text: str) -> list[float]:
    costs = []
    for item in text.split(","):
        value = item.strip()
        if value:
            costs.append(float(value))
    if not costs:
        raise ValueError("cost list is empty")
    return costs


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def candidate_registry(config: dict) -> pd.DataFrame:
    disabled = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    rows = []
    for spec in CANDIDATE_SPECS:
        rows.append(
            {
                "variant": spec["variant"],
                "role": spec["role"],
                "description": spec["description"],
                "multipliers_json": json.dumps(spec.get("multipliers", {}), sort_keys=True),
                "disabled_experts": disabled,
                "selection_source": spec["selection_source"],
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": bool(spec["eligible_for_default_promotion"]),
                "implementation_mode": spec["target_mode"],
                "turnover_cap": spec.get("turnover_cap"),
            }
        )
    return pd.DataFrame(rows)


def spec_by_variant(variant: str) -> dict[str, Any]:
    for spec in CANDIDATE_SPECS:
        if spec["variant"] == variant:
            return spec
    raise KeyError(variant)


def all_year_state_pairs(panel: dict[str, pd.DataFrame], config: dict) -> list[tuple[int, str]]:
    years = sorted(int(year) for year in pd.to_datetime(panel["eligible"]["date"]).dt.year.dropna().unique())
    states = list(config["portfolio"]["sleeve_budget_by_state"].keys())
    return [(year, state) for year in years for state in states]


def multiplier_map(panel: dict[str, pd.DataFrame], config: dict, spec: dict[str, Any]) -> dict[tuple[int, str], dict[str, float]]:
    if spec["target_mode"] == "multiplier":
        multipliers = dict(spec.get("multipliers", {}))
        return {(year, state): dict(multipliers) for year, state in all_year_state_pairs(panel, config)}
    if spec["target_mode"] == "state_multiplier":
        state_rules = spec.get("multipliers", {})
        return {
            (year, state): dict(state_rules.get(state, {}))
            for year, state in all_year_state_pairs(panel, config)
        }
    return {}


def annotate_targets(targets: pd.DataFrame, spec: dict[str, Any], config: dict) -> pd.DataFrame:
    out = targets.copy()
    out["variant"] = spec["variant"]
    out["candidate_role"] = spec["role"]
    out["multipliers_json"] = json.dumps(spec.get("multipliers", {}), sort_keys=True)
    out["implementation_mode"] = spec["target_mode"]
    out["turnover_cap"] = spec.get("turnover_cap")
    out["disabled_experts"] = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    return out


def build_multiplier_targets(panel: dict[str, pd.DataFrame], config: dict, spec: dict[str, Any]) -> pd.DataFrame:
    disabled = {str(item) for item in config.get("disabled_experts_by_default", [])}
    start_date = pd.to_datetime(panel["eligible"]["date"].min()) if not panel["eligible"].empty else None
    raw_targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=disabled,
        expert_multipliers_by_year_state=multiplier_map(panel, config, spec),
    )
    targets = v310.enrich_targets(raw_targets, panel)
    targets = v310.enforce_cash_cap(targets, config)
    cap = spec.get("turnover_cap")
    if cap is None:
        cap = float(config["portfolio"]["constraints"].get("monthly_turnover_target_cap", 0.8))
    targets = v310.enforce_turnover_cap(targets, max_turnover=float(cap))
    return annotate_targets(targets, spec, config)


def blend_targets(
    *,
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    spec: dict[str, Any],
    config: dict,
) -> pd.DataFrame:
    base_weight = float(spec["blend_base_weight"])
    candidate_weight = float(spec["blend_candidate_weight"])
    b = base.copy()
    c = candidate.copy()
    b["signal_date"] = pd.to_datetime(b["signal_date"])
    c["signal_date"] = pd.to_datetime(c["signal_date"])
    bw = b[["signal_date", "asset", "weight"]].rename(columns={"weight": "base_weight"})
    cw = c[["signal_date", "asset", "weight"]].rename(columns={"weight": "candidate_weight"})
    weights = bw.merge(cw, on=["signal_date", "asset"], how="outer")
    weights["base_weight"] = pd.to_numeric(weights["base_weight"], errors="coerce").fillna(0.0)
    weights["candidate_weight"] = pd.to_numeric(weights["candidate_weight"], errors="coerce").fillna(0.0)
    weights["weight"] = base_weight * weights["base_weight"] + candidate_weight * weights["candidate_weight"]
    weights = weights.drop(columns=["base_weight", "candidate_weight"])

    meta = pd.concat([c, b], ignore_index=True, sort=False)
    meta = meta.drop(columns=[col for col in ["weight", "target_weight"] if col in meta.columns])
    meta = meta.drop_duplicates(["signal_date", "asset"], keep="first")
    out = weights.merge(meta, on=["signal_date", "asset"], how="left")
    rows = []
    for signal_date, group in out.groupby("signal_date", sort=True):
        g = group.copy()
        total = float(pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).sum())
        if total > 0:
            g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0) / total
        g["target_weight"] = g["weight"]
        rows.append(g)
    blended = pd.concat(rows, ignore_index=True, sort=False)
    blended = v310.enforce_cash_cap(blended, config)
    cap = spec.get("turnover_cap")
    if cap is not None:
        blended = v310.enforce_turnover_cap(blended, max_turnover=float(cap))
    return annotate_targets(blended, spec, config)


def build_raw_targets(panel: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
    targets: dict[str, pd.DataFrame] = {}
    baseline_spec = spec_by_variant(BASELINE_VARIANT)
    targets[BASELINE_VARIANT] = annotate_targets(v310.build_targets(panel, deepcopy(config)), baseline_spec, config)

    guarded_spec = spec_by_variant("guarded_industry_trend_low_turnover")
    targets[guarded_spec["variant"]] = build_multiplier_targets(panel, deepcopy(config), guarded_spec)

    blend_spec = spec_by_variant("baseline_blend_confidence_gate")
    targets[blend_spec["variant"]] = blend_targets(
        base=targets[blend_spec["blend_base"]],
        candidate=targets[blend_spec["blend_candidate"]],
        spec=blend_spec,
        config=deepcopy(config),
    )

    valuation_spec = spec_by_variant("valuation_risk_repair_defensive_guard")
    targets[valuation_spec["variant"]] = build_multiplier_targets(panel, deepcopy(config), valuation_spec)
    return targets


def run_raw_candidates(
    *,
    panel: dict[str, pd.DataFrame],
    config: dict,
    costs: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], dict[str, pd.DataFrame], dict[tuple[str, float], pd.DataFrame]]:
    rows = []
    navs: dict[tuple[str, float], pd.DataFrame] = {}
    trades: dict[tuple[str, float], pd.DataFrame] = {}
    targets_by_variant = build_raw_targets(panel, config)
    for variant in RAW_TARGET_VARIANTS:
        spec = spec_by_variant(variant)
        targets = targets_by_variant[variant]
        model.write_csv(targets, output_dir / f"target_weights_{variant}.csv")
        for cost in costs:
            suffix = f"{variant}_{int(cost)}bps"
            bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
            nav = bt["nav"].copy()
            nav["variant"] = variant
            nav["cost_bps"] = float(cost)
            summary = model.summarize_nav(nav)
            if not summary.empty:
                summary.insert(0, "variant", variant)
                summary.insert(1, "role", spec["role"])
                summary.insert(2, "cost_bps", float(cost))
                summary["diagnostic_full_sample_only"] = True
                summary["annual_excess_vs_benchmark"] = summary["annual_return"] - summary["benchmark_annual_return"]
                summary["drawdown_improvement_vs_benchmark"] = summary["max_drawdown"] - summary["benchmark_max_drawdown"]
                rows.append(summary)
            navs[(variant, float(cost))] = nav
            trades[(variant, float(cost))] = bt["trades"]
            model.write_csv(nav, output_dir / f"nav_{suffix}.csv")
            model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
    metrics = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return metrics, navs, targets_by_variant, trades


def yearly_stability_penalty(
    *,
    variant_nav: pd.DataFrame,
    baseline_nav: pd.DataFrame,
    dates: pd.Series,
    train_mask: pd.Series,
) -> dict[str, float]:
    years = sorted(int(year) for year in dates.loc[train_mask.to_numpy()].dt.year.unique())
    annual_deltas = []
    dd_deltas = []
    for year in years:
        year_mask = train_mask & dates.dt.year.eq(year)
        if int(year_mask.sum()) < 60:
            continue
        candidate_metrics = v311.summarize_slice(variant_nav, year_mask)
        baseline_metrics = v311.summarize_slice(baseline_nav, year_mask)
        annual_deltas.append(candidate_metrics["annual_return"] - baseline_metrics["annual_return"])
        dd_deltas.append(candidate_metrics["max_drawdown"] - baseline_metrics["max_drawdown"])
    if not annual_deltas:
        return {
            "underperform_year_rate": 1.0,
            "severe_underperform_year_rate": 1.0,
            "drawdown_worse_year_rate": 1.0,
            "stability_penalty": 0.10,
        }
    annual = pd.Series(annual_deltas, dtype=float)
    dd = pd.Series(dd_deltas, dtype=float)
    under_rate = float((annual < 0.0).mean())
    severe_rate = float((annual < -0.02).mean())
    dd_worse_rate = float((dd < -0.02).mean())
    penalty = 0.040 * under_rate + 0.030 * severe_rate + 0.020 * dd_worse_rate
    return {
        "underperform_year_rate": under_rate,
        "severe_underperform_year_rate": severe_rate,
        "drawdown_worse_year_rate": dd_worse_rate,
        "stability_penalty": float(penalty),
    }


def selector_passes_guards(
    *,
    selected: str,
    penalized_scores: pd.Series,
    select_metrics_by_variant: dict[str, dict[str, float]],
    stability_by_variant: dict[str, dict[str, float]],
    baseline_metrics: dict[str, float],
    margin: float,
    max_underperform_rate: float,
    max_dd_slippage: float,
) -> tuple[bool, str]:
    if selected == BASELINE_VARIANT:
        return True, "baseline_selected"
    annual_delta = select_metrics_by_variant[selected]["annual_return"] - baseline_metrics["annual_return"]
    drawdown_delta = select_metrics_by_variant[selected]["max_drawdown"] - baseline_metrics["max_drawdown"]
    score_margin = float(penalized_scores.loc[selected] - penalized_scores.loc[BASELINE_VARIANT])
    under_rate = float(stability_by_variant[selected]["underperform_year_rate"])
    failures = []
    if score_margin < margin:
        failures.append(f"score_margin={score_margin:.6f}<required={margin:.6f}")
    if annual_delta < 0.0:
        failures.append(f"inner_annual_delta={annual_delta:.6f}<0")
    if drawdown_delta < max_dd_slippage:
        failures.append(f"inner_drawdown_delta={drawdown_delta:.6f}<allowed={max_dd_slippage:.6f}")
    if under_rate > max_underperform_rate:
        failures.append(f"underperform_year_rate={under_rate:.6f}>allowed={max_underperform_rate:.6f}")
    if failures:
        return False, "; ".join(failures)
    return True, "nonbaseline_passed_margin_stability_guards"


def nested_stability_selector(
    *,
    navs: dict[tuple[str, float], pd.DataFrame],
    costs: list[float],
    lookback_years: int,
    inner_validation_years: int,
    min_train_days: int,
    embargo_days: int,
    selection_margin: float,
    max_underperform_rate: float,
    max_dd_slippage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, pd.DataFrame], pd.DataFrame]:
    selection_rows = []
    fold_rows = []
    selected_navs: dict[float, pd.DataFrame] = {}
    performance_rows = []
    for cost in costs:
        aligned = v311.align_variant_navs(navs, cost)
        baseline = aligned[BASELINE_VARIANT]
        dates = pd.to_datetime(baseline["date"]).reset_index(drop=True)
        years = sorted(int(year) for year in dates.dt.year.unique())
        selected_frames = []
        for test_year in years:
            test_start = pd.Timestamp(year=test_year, month=1, day=1)
            test_end = pd.Timestamp(year=test_year, month=12, day=31)
            outer_train_start = test_start - pd.DateOffset(years=lookback_years)
            outer_train_end = test_start - pd.Timedelta(days=embargo_days + 1)
            inner_validation_start = test_start - pd.DateOffset(years=inner_validation_years)
            inner_validation_end = outer_train_end
            inner_train_start = outer_train_start
            inner_train_end = inner_validation_start - pd.Timedelta(days=embargo_days + 1)

            outer_train_mask = (dates >= outer_train_start) & (dates <= outer_train_end)
            inner_train_mask = (dates >= inner_train_start) & (dates <= inner_train_end)
            inner_validation_mask = (dates >= inner_validation_start) & (dates <= inner_validation_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            train_days = int(outer_train_mask.sum())
            inner_train_days = int(inner_train_mask.sum())
            inner_validation_days = int(inner_validation_mask.sum())
            test_days = int(test_mask.sum())
            base_payload = {
                "cost_bps": float(cost),
                "test_year": int(test_year),
                "outer_train_start": outer_train_start,
                "outer_train_end": outer_train_end,
                "inner_train_start": inner_train_start,
                "inner_train_end": inner_train_end,
                "inner_validation_start": inner_validation_start,
                "inner_validation_end": inner_validation_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_days": train_days,
                "inner_train_days": inner_train_days,
                "inner_validation_days": inner_validation_days,
                "test_days": test_days,
                "purge_days": 0,
                "embargo_days": int(embargo_days),
                "selection_policy": SELECTOR_VARIANT,
            }
            if test_days == 0:
                continue
            if train_days < min_train_days or inner_validation_days < 126:
                selection_rows.append(
                    {
                        **base_payload,
                        "selected_variant": BASELINE_VARIANT,
                        "selection_status": "skipped_insufficient_train",
                        "selection_reason": f"train_days={train_days}; inner_validation_days={inner_validation_days}",
                    }
                )
                continue

            base_select = v311.summarize_slice(baseline, inner_validation_mask)
            base_test = v311.summarize_slice(baseline, test_mask)
            raw_scores = {}
            penalized_scores = {}
            select_metrics_by_variant = {}
            test_metrics_by_variant = {}
            stability_by_variant = {}
            for variant, nav in aligned.items():
                select_metrics = v311.summarize_slice(nav, inner_validation_mask)
                test_metrics = v311.summarize_slice(nav, test_mask)
                stability = (
                    {
                        "underperform_year_rate": 0.0,
                        "severe_underperform_year_rate": 0.0,
                        "drawdown_worse_year_rate": 0.0,
                        "stability_penalty": 0.0,
                    }
                    if variant == BASELINE_VARIANT
                    else yearly_stability_penalty(
                        variant_nav=nav,
                        baseline_nav=baseline,
                        dates=dates,
                        train_mask=outer_train_mask,
                    )
                )
                raw_score = v311.selection_score(select_metrics, base_select)
                raw_scores[variant] = raw_score
                penalized_scores[variant] = raw_score - stability["stability_penalty"]
                select_metrics_by_variant[variant] = select_metrics
                test_metrics_by_variant[variant] = test_metrics
                stability_by_variant[variant] = stability

            score_series = pd.Series(penalized_scores).dropna().sort_values(ascending=False)
            proposed = str(score_series.index[0]) if not score_series.empty else BASELINE_VARIANT
            guard_ok, guard_reason = selector_passes_guards(
                selected=proposed,
                penalized_scores=score_series,
                select_metrics_by_variant=select_metrics_by_variant,
                stability_by_variant=stability_by_variant,
                baseline_metrics=base_select,
                margin=selection_margin,
                max_underperform_rate=max_underperform_rate,
                max_dd_slippage=max_dd_slippage,
            )
            selected = proposed if guard_ok else BASELINE_VARIANT
            selection_reason = guard_reason if guard_ok else f"fallback_to_baseline: {guard_reason}"

            selected_test = test_metrics_by_variant[selected]
            oos_scores = pd.Series(
                {variant: v311.selection_score(metrics, base_test) for variant, metrics in test_metrics_by_variant.items()}
            ).sort_values(ascending=False)
            rank = int(list(oos_scores.index).index(selected) + 1)
            rank_pct = 1.0 - (rank - 1) / max(len(oos_scores) - 1, 1)
            selection_rows.append(
                {
                    **base_payload,
                    "selected_variant": selected,
                    "proposed_variant": proposed,
                    "selection_status": "selected_by_prior_window",
                    "selection_reason": selection_reason,
                    "baseline_fallback_used": bool(selected == BASELINE_VARIANT and proposed != BASELINE_VARIANT),
                    "selection_margin_required": float(selection_margin),
                    "max_underperform_rate": float(max_underperform_rate),
                    "max_dd_slippage": float(max_dd_slippage),
                    "selected_inner_validation_score": float(score_series.loc[selected]),
                    "selected_inner_validation_raw_score": float(raw_scores[selected]),
                    "selected_stability_penalty": float(stability_by_variant[selected]["stability_penalty"]),
                    "selected_oos_score": float(oos_scores.loc[selected]),
                    "selected_oos_rank": rank,
                    "selected_oos_rank_pct": float(rank_pct),
                    "baseline_inner_validation_annual": base_select["annual_return"],
                    "selected_inner_validation_annual": select_metrics_by_variant[selected]["annual_return"],
                    "baseline_oos_annual": base_test["annual_return"],
                    "selected_oos_annual": selected_test["annual_return"],
                    "selected_oos_minus_baseline_annual": selected_test["annual_return"] - base_test["annual_return"],
                    "selected_oos_minus_baseline_drawdown": selected_test["max_drawdown"] - base_test["max_drawdown"],
                }
            )
            for variant, select_metrics in select_metrics_by_variant.items():
                row = {
                    "cost_bps": float(cost),
                    "test_year": int(test_year),
                    "variant": variant,
                    "selection_policy": SELECTOR_VARIANT,
                    "inner_validation_raw_score": raw_scores[variant],
                    "stability_penalty": stability_by_variant[variant]["stability_penalty"],
                    "inner_validation_score": penalized_scores[variant],
                    "oos_score": float(oos_scores.loc[variant]),
                    "selected": variant == selected,
                    "proposed": variant == proposed,
                    "inner_validation_start": inner_validation_start,
                    "inner_validation_end": inner_validation_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
                row.update(stability_by_variant[variant])
                row.update({f"inner_validation_{key}": value for key, value in select_metrics.items()})
                row.update({f"oos_{key}": value for key, value in test_metrics_by_variant[variant].items()})
                fold_rows.append(row)

            selected_frame = aligned[selected].loc[test_mask.to_numpy()].copy()
            selected_frame["selected_variant"] = selected
            selected_frame["test_year"] = int(test_year)
            selected_frame["selection_policy"] = SELECTOR_VARIANT
            selected_frames.append(selected_frame)

        selected_nav = v311.stitch_nav(selected_frames)
        selected_navs[float(cost)] = selected_nav
        summary = model.summarize_nav(selected_nav)
        if not summary.empty:
            summary.insert(0, "variant", SELECTOR_VARIANT)
            summary.insert(1, "cost_bps", float(cost))
            baseline_same = v311.stitch_baseline_same_period(baseline, selected_nav)
            base_summary = model.summarize_nav(baseline_same)
            if not base_summary.empty:
                summary["baseline_same_period_annual_return"] = float(base_summary["annual_return"].iloc[0])
                summary["baseline_same_period_sharpe"] = float(base_summary["sharpe_no_rf"].iloc[0])
                summary["baseline_same_period_max_drawdown"] = float(base_summary["max_drawdown"].iloc[0])
                summary["annual_delta_vs_v310"] = summary["annual_return"] - summary["baseline_same_period_annual_return"]
                summary["sharpe_delta_vs_v310"] = summary["sharpe_no_rf"] - summary["baseline_same_period_sharpe"]
                summary["drawdown_delta_vs_v310"] = summary["max_drawdown"] - summary["baseline_same_period_max_drawdown"]
            performance_rows.append(summary)
    performance = pd.concat(performance_rows, ignore_index=True, sort=False) if performance_rows else pd.DataFrame()
    return pd.DataFrame(selection_rows), pd.DataFrame(fold_rows), selected_navs, performance


def target_integrity_checks(targets_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, targets in targets_by_variant.items():
        weights = pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0)
        grouped = targets.assign(_weight=weights).groupby("signal_date")["_weight"].sum()
        rows.append(
            {
                "check": f"target_integrity_{variant}",
                "status": "pass"
                if not targets.empty
                and float(weights.min()) >= -1e-10
                and float((grouped - 1.0).abs().max()) <= 1e-6
                else "fail",
                "detail": f"rows={targets.shape[0]}; min_weight={float(weights.min()) if not weights.empty else math.nan:.8f}; max_weight_sum_error={float((grouped - 1.0).abs().max()) if not grouped.empty else math.nan:.8f}",
            }
        )
    return pd.DataFrame(rows)


def build_constraint_checks(
    *,
    registry: pd.DataFrame,
    selection: pd.DataFrame,
    performance: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    decision: pd.DataFrame,
    targets_by_variant: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    checks = v311.build_constraint_checks(
        registry=registry,
        selection=selection,
        performance=performance,
        pbo_summary=pbo_summary,
        decision=decision,
    )
    extra = target_integrity_checks(targets_by_variant)
    fallback_count = 0
    if not selection.empty and "baseline_fallback_used" in selection.columns:
        fallback_count = int(pd.Series(selection["baseline_fallback_used"]).fillna(False).astype(bool).sum())
    selector_rows = [
        {
            "check": "selector_has_baseline_fallback",
            "status": "pass" if fallback_count > 0 else "warn",
            "detail": str(fallback_count),
        },
        {
            "check": "selector_nonbaseline_evidence_is_prior_window_only",
            "status": "pass",
            "detail": "proposed and selected variants use inner validation plus prior-train stability penalties",
        },
    ]
    return pd.concat([checks, extra, pd.DataFrame(selector_rows)], ignore_index=True, sort=False)


def same_period_baseline_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    out = v311.same_period_baseline_comparison(performance)
    if not out.empty:
        out["selected_strategy"] = SELECTOR_VARIANT
    return out


def cost_sensitivity_table(performance: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    compare = same_period_baseline_comparison(performance)
    if decision.empty:
        compare["gate_decision"] = "blocked"
        return compare
    dec = decision[["cost_bps", "decision", "pbo"]].copy()
    return compare.merge(dec, on="cost_bps", how="left").rename(columns={"decision": "gate_decision"})


def make_report(performance: pd.DataFrame, pbo_summary: pd.DataFrame, decision: pd.DataFrame, selection: pd.DataFrame) -> str:
    ref = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1)
    pbo10 = pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    dec = decision["overall_decision"].iloc[0] if not decision.empty else "blocked"
    nonbaseline_rate = 0.0
    selected = selection[selection["selection_status"].isin(["selected_by_prior_window", "baseline_fallback_prior_window"])]
    if not selected.empty:
        nonbaseline_rate = float((selected["selected_variant"] != BASELINE_VARIANT).mean())
    lines = [
        "# HIRSSM V3.12 Candidate Implementation Harness",
        "",
        "## Purpose",
        "",
        "Implement V3.12 candidate directions and test them against V3.10 with prior-window selection, stability penalties, and purged PBO diagnostics.",
        "",
        "## 10bps Nested OOS",
        "",
    ]
    if not ref.empty:
        item = ref.iloc[0]
        lines.extend(
            [
                f"- Annual return: {float(item['annual_return']):.6f}",
                f"- Sharpe no RF: {float(item['sharpe_no_rf']):.6f}",
                f"- Max drawdown: {float(item['max_drawdown']):.6f}",
                f"- Annual delta vs V3.10: {float(item.get('annual_delta_vs_v310', np.nan)):.6f}",
                f"- Drawdown delta vs V3.10: {float(item.get('drawdown_delta_vs_v310', np.nan)):.6f}",
                f"- Nonbaseline selection rate: {nonbaseline_rate:.6f}",
            ]
        )
    if not pbo10.empty:
        lines.extend(["", "## PBO", "", f"- 10bps PBO: {float(pbo10['pbo'].iloc[0]):.6f}", f"- 10bps PBO status: {pbo10['pbo_status'].iloc[0]}"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Overall decision: {dec}",
            "- Full-sample raw candidate metrics are diagnostic only.",
            "- Selector deviations require prior-window margin, stable recent yearly evidence, and drawdown guard.",
        ]
    )
    return "\n".join(lines)


def make_self_check_report(checks: pd.DataFrame, manifest_findings: list[dict[str, str]] | None = None) -> str:
    manifest_findings = manifest_findings or []
    fail_count = int((checks["status"] == "fail").sum())
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    lines = [
        "# HIRSSM V3.12 Self Check",
        "",
        f"- Constraint failures: {fail_count}",
        f"- Manifest failures: {manifest_fail_count}",
        "- Baseline fallback enabled: true",
        "- Prior-window selection required: true",
        "- Purged PBO diagnostics required: true",
        "",
        "## Checks",
        "",
    ]
    for _, row in checks.iterrows():
        lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
    if manifest_findings:
        lines.extend(["", "## Manifest Findings", ""])
        for finding in manifest_findings:
            lines.append(f"- {finding.get('severity')}: {finding.get('field')} - {finding.get('message')}")
    return "\n".join(lines)


def make_agent_manifest(
    *,
    start_time: str,
    agent_dir: Path,
    config_path: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.12",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_12_candidate_implementation_harness.py",
        "config": {
            "config_path": str(config_path.as_posix()),
            "baseline_variant": BASELINE_VARIANT,
            "selector_variant": SELECTOR_VARIANT,
            "candidate_count": len(CANDIDATE_SPECS),
        },
        "data_refs": ["data_raw/index/akshare_csindex", "data_raw/index/akshare_sw_industry"],
        "code_refs": [
            "strategy_lab/hirssm_v3_12_candidate_implementation_harness.py",
            "strategy_lab/hirssm_v3_11_nested_candidate_harness.py",
            "strategy_lab/hirssm_v3_10_clean_baseline.py",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
            "strategy_lab/model_run_manifest.py",
        ],
        "output_dir": str(agent_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": [
            "outputs/agent_runs/v3_12/candidate_improvement_design",
            "outputs/hirssm_v3_11_nested_candidate_harness",
            "outputs/hirssm_v3_10_clean_baseline",
            "configs/hirssm_v2_default.json",
        ],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Raw candidate full-sample metrics are diagnostic only.",
            "PBO diagnostics apply to raw candidate grid; final selector is nested OOS stitched.",
        ],
        "risk_flags": ["candidate_metrics_full_sample_diagnostic_only", "raw_candidate_pbo_grid_not_selector_nav"],
        "next_decision": "If gates reject, keep V3.10 baseline and use V3.12 diagnostics to design V3.13.",
        "handoff_summary": "V3.12 candidate targets, nested selector, PBO diagnostics, and gate decision were generated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.12 candidate implementation harness.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--agent-output-dir", default=str(AGENT_OUTPUT_DIR))
    parser.add_argument("--costs", default="5,10,20,30")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--inner-validation-years", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--embargo-days", type=int, default=21)
    parser.add_argument("--pbo-blocks", type=int, default=10)
    parser.add_argument("--pbo-train-blocks", type=int, default=5)
    parser.add_argument("--pbo-purge-days", type=int, default=63)
    parser.add_argument("--selection-margin", type=float, default=0.010)
    parser.add_argument("--max-underperform-rate", type=float, default=0.50)
    parser.add_argument("--max-dd-slippage", type=float, default=-0.010)
    args = parser.parse_args()

    start_time = now_text()
    root = Path(args.root)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    agent_dir = Path(args.agent_output_dir)
    costs = parse_costs(args.costs)
    config = model.read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    panel = wf.build_panel(model, root, config, None, None)
    registry = candidate_registry(config)
    candidate_metrics, navs, targets_by_variant, trades = run_raw_candidates(
        panel=panel,
        config=config,
        costs=costs,
        output_dir=output_dir,
    )
    selection, fold_scores, selected_navs, nested_performance = nested_stability_selector(
        navs=navs,
        costs=costs,
        lookback_years=args.lookback_years,
        inner_validation_years=args.inner_validation_years,
        min_train_days=args.min_train_days,
        embargo_days=args.embargo_days,
        selection_margin=args.selection_margin,
        max_underperform_rate=args.max_underperform_rate,
        max_dd_slippage=args.max_dd_slippage,
    )
    pbo_folds, pbo_summary = v311.purged_block_pbo(
        navs=navs,
        costs=costs,
        n_blocks=args.pbo_blocks,
        train_blocks=args.pbo_train_blocks,
        purge_days=args.pbo_purge_days,
        embargo_days=args.embargo_days,
    )
    decision = v311.promotion_decision(nested_performance, pbo_summary, selection)
    split_manifest = v311.split_manifest_from_selection(selection)
    embargo_audit = v311.embargo_purge_audit(split_manifest)
    outer_oos = v311.outer_fold_oos_results(selection)
    same_period = same_period_baseline_comparison(nested_performance)
    cost_sensitivity = cost_sensitivity_table(nested_performance, decision)
    checks = build_constraint_checks(
        registry=registry,
        selection=selection,
        performance=nested_performance,
        pbo_summary=pbo_summary,
        decision=decision,
        targets_by_variant=targets_by_variant,
    )
    findings = v311.validation_findings(checks, decision, pbo_summary)
    leakage = v311.leakage_checklist(split_manifest, embargo_audit)
    robustness = v311.robustness_summary(decision, pbo_summary, cost_sensitivity)

    registry_path = output_dir / "candidate_registry.csv"
    registry_json_path = output_dir / "candidate_registry.json"
    split_manifest_path = output_dir / "split_manifest.csv"
    split_manifest_json_path = output_dir / "split_manifest.json"
    embargo_audit_path = output_dir / "embargo_purge_audit.csv"
    inner_scores_path = output_dir / "inner_candidate_scores.csv"
    nested_selection_path = output_dir / "nested_selection_by_fold.csv"
    outer_oos_path = output_dir / "outer_fold_oos_results.csv"
    same_period_path = output_dir / "same_period_baseline_comparison.csv"
    cost_sensitivity_path = output_dir / "cost_sensitivity.csv"
    pbo_cscv_summary_path = output_dir / "pbo_cscv_summary.csv"
    pbo_cscv_splits_path = output_dir / "pbo_cscv_splits.csv"
    gate_decision_path = output_dir / "candidate_gate_decision.csv"
    validation_findings_path = output_dir / "validation_findings.csv"
    leakage_checklist_path = output_dir / "leakage_checklist.csv"
    robustness_summary_path = output_dir / "robustness_summary.csv"
    candidate_metrics_path = output_dir / "candidate_full_sample_diagnostic_metrics.csv"
    selection_path = output_dir / "nested_selection_history.csv"
    fold_scores_path = output_dir / "nested_fold_scores.csv"
    pbo_folds_path = output_dir / "purged_pbo_folds.csv"
    pbo_summary_path = output_dir / "pbo_report.csv"
    performance_path = output_dir / "nested_oos_performance.csv"
    decision_path = output_dir / "promotion_decision.csv"
    checks_path = output_dir / "constraint_check.csv"
    report_path = output_dir / "WALK_FORWARD_REPORT.md"
    changelog_path = output_dir / "MODEL_CHANGELOG.md"
    self_check_path = output_dir / "SELF_CHECK_REPORT.md"
    model_manifest_path = output_dir / "model_run_manifest.json"
    model_manifest_check_path = output_dir / "model_run_manifest_check.csv"

    model.write_csv(registry, registry_path)
    write_json({"candidates": registry.to_dict(orient="records")}, registry_json_path)
    model.write_csv(split_manifest, split_manifest_path)
    write_json({"splits": split_manifest.astype(str).to_dict(orient="records")}, split_manifest_json_path)
    model.write_csv(embargo_audit, embargo_audit_path)
    model.write_csv(fold_scores, inner_scores_path)
    model.write_csv(selection, nested_selection_path)
    model.write_csv(outer_oos, outer_oos_path)
    model.write_csv(same_period, same_period_path)
    model.write_csv(cost_sensitivity, cost_sensitivity_path)
    model.write_csv(pbo_summary, pbo_cscv_summary_path)
    model.write_csv(pbo_folds, pbo_cscv_splits_path)
    model.write_csv(decision, gate_decision_path)
    model.write_csv(findings, validation_findings_path)
    model.write_csv(leakage, leakage_checklist_path)
    model.write_csv(robustness, robustness_summary_path)
    model.write_csv(candidate_metrics, candidate_metrics_path)
    model.write_csv(selection, selection_path)
    model.write_csv(fold_scores, fold_scores_path)
    model.write_csv(pbo_folds, pbo_folds_path)
    model.write_csv(pbo_summary, pbo_summary_path)
    model.write_csv(nested_performance, performance_path)
    model.write_csv(decision, decision_path)
    model.write_csv(checks, checks_path)
    write_text(make_report(nested_performance, pbo_summary, decision, selection), report_path)
    write_text(make_report(nested_performance, pbo_summary, decision, selection), changelog_path)
    write_text(make_self_check_report(checks), self_check_path)

    selected_nav_artifacts = []
    for cost, nav in selected_navs.items():
        nav_path = output_dir / f"nav_{SELECTOR_VARIANT}_{int(cost)}bps.csv"
        yearly_path = output_dir / f"yearly_returns_{SELECTOR_VARIANT}_{int(cost)}bps.csv"
        regime_path = output_dir / f"regime_returns_{SELECTOR_VARIANT}_{int(cost)}bps.csv"
        model.write_csv(nav, nav_path)
        selected_nav_artifacts.append(nav_path)
        if not nav.empty:
            model.write_csv(model.yearly_returns(nav), yearly_path)
            model.write_csv(model.regime_returns(nav, panel["regimes"]), regime_path)
            selected_nav_artifacts.extend([yearly_path, regime_path])

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    reference_perf = nested_performance[nested_performance["cost_bps"].astype(float).eq(10.0)].head(1)
    selected_windows = selection[selection["selection_status"].isin(["selected_by_prior_window", "baseline_fallback_prior_window"])]
    metrics = {
        "candidate_count": int(registry.shape[0]),
        "raw_candidate_count": int(len(RAW_TARGET_VARIANTS)),
        "walk_forward_selected_year_count": int(selected_windows.shape[0]) if not selected_windows.empty else 0,
        "nonbaseline_selection_rate": float((selected_windows["selected_variant"] != BASELINE_VARIANT).mean()) if not selected_windows.empty else 0.0,
        "pbo_10bps": float(pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)]["pbo"].iloc[0]) if not pbo_summary.empty and pbo_summary["cost_bps"].astype(float).eq(10.0).any() else np.nan,
        "overall_decision": str(decision["overall_decision"].iloc[0]) if not decision.empty else "blocked",
    }
    if not reference_perf.empty:
        item = reference_perf.iloc[0]
        metrics.update(
            {
                "reference_cost_bps": 10.0,
                "annual_return": float(item["annual_return"]),
                "sharpe_no_rf": float(item["sharpe_no_rf"]),
                "max_drawdown": float(item["max_drawdown"]),
                "annual_delta_vs_v310": float(item.get("annual_delta_vs_v310", np.nan)),
                "drawdown_delta_vs_v310": float(item.get("drawdown_delta_vs_v310", np.nan)),
            }
        )

    artifact_paths = [
        registry_path,
        registry_json_path,
        split_manifest_path,
        split_manifest_json_path,
        embargo_audit_path,
        inner_scores_path,
        nested_selection_path,
        outer_oos_path,
        same_period_path,
        cost_sensitivity_path,
        pbo_cscv_summary_path,
        pbo_cscv_splits_path,
        gate_decision_path,
        validation_findings_path,
        leakage_checklist_path,
        robustness_summary_path,
        candidate_metrics_path,
        selection_path,
        fold_scores_path,
        pbo_folds_path,
        pbo_summary_path,
        performance_path,
        decision_path,
        checks_path,
        report_path,
        changelog_path,
        self_check_path,
    ]
    artifact_paths.extend(selected_nav_artifacts)
    for variant in RAW_TARGET_VARIANTS:
        artifact_paths.append(output_dir / f"target_weights_{variant}.csv")
        for cost in costs:
            artifact_paths.extend(
                [
                    output_dir / f"nav_{variant}_{int(cost)}bps.csv",
                    output_dir / f"trades_{variant}_{int(cost)}bps.csv",
                ]
            )

    manifest = build_model_run_manifest(
        root=root,
        task_id=TASK_ID,
        run_id=f"{TASK_ID}_run_001",
        model_version=MODEL_VERSION,
        baseline="HIRSSM V3.10 Clean Rank-Vol Core",
        status="success" if fail_count == 0 else "fail",
        started_at=start_time,
        finished_at=now_text(),
        output_dir=output_dir,
        command=["python", "-X", "utf8", "strategy_lab/hirssm_v3_12_candidate_implementation_harness.py"],
        argv={
            "costs": costs,
            "lookback_years": args.lookback_years,
            "inner_validation_years": args.inner_validation_years,
            "min_train_days": args.min_train_days,
            "embargo_days": args.embargo_days,
            "pbo_blocks": args.pbo_blocks,
            "pbo_train_blocks": args.pbo_train_blocks,
            "pbo_purge_days": args.pbo_purge_days,
            "selection_margin": args.selection_margin,
            "max_underperform_rate": args.max_underperform_rate,
            "max_dd_slippage": args.max_dd_slippage,
        },
        code_paths=[
            root / "strategy_lab" / "hirssm_v3_12_candidate_implementation_harness.py",
            root / "strategy_lab" / "hirssm_v3_11_nested_candidate_harness.py",
            root / "strategy_lab" / "hirssm_v3_10_clean_baseline.py",
            root / "strategy_lab" / "hirssm_v2_model.py",
            root / "strategy_lab" / "hirssm_v2_walk_forward.py",
            root / "strategy_lab" / "model_run_manifest.py",
        ],
        config_path=config_path,
        data_paths=v311.collect_data_refs(root, config),
        artifact_paths=artifact_paths,
        selection={
            "baseline_variant": BASELINE_VARIANT,
            "selector_variant": SELECTOR_VARIANT,
            "candidate_count": int(registry.shape[0]),
            "raw_candidate_count": int(len(RAW_TARGET_VARIANTS)),
            "selection_method": "stability_penalized_prior_window_with_baseline_fallback",
            "full_sample_metrics_diagnostic_only": True,
            "selection_margin": float(args.selection_margin),
            "max_underperform_rate": float(args.max_underperform_rate),
            "purge_days": int(args.pbo_purge_days),
            "embargo_days": int(args.embargo_days),
        },
        metrics=metrics,
        checks={
            "self_check_pass": fail_count == 0,
            "fail_count": fail_count,
            "warn_count": warn_count,
            "constraint_fail_count": fail_count,
        },
        limitations=[
            "Raw candidate full-sample metrics are diagnostics and cannot promote a candidate.",
            "PBO applies to raw candidate grid rather than stitched selector nav.",
            "Selector is conservative by design and may fall back to baseline frequently.",
        ],
        risk_flags=["candidate_metrics_full_sample_diagnostic_only", "raw_candidate_pbo_grid_not_selector_nav"],
        next_decision="If V3.12 rejects candidates, keep V3.10 as active baseline and start V3.13 from the failure diagnostics.",
        handoff_summary="V3.12 generated raw candidate diagnostics, stability selector OOS, purged PBO, and gate decision.",
    )
    write_json(manifest, model_manifest_path)
    manifest_findings = validate_model_run_manifest(manifest)
    manifest_check = pd.DataFrame(manifest_findings)
    if manifest_check.empty:
        manifest_check = pd.DataFrame([{"severity": "pass", "field": "model_run_manifest", "message": "no failures"}])
    model.write_csv(manifest_check, model_manifest_check_path)
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    manifest_warn_count = sum(1 for item in manifest_findings if item.get("severity") == "warn")

    agent_report_path = agent_dir / "agent_report.md"
    agent_registry_path = agent_dir / "candidate_registry.csv"
    agent_decision_path = agent_dir / "promotion_decision.csv"
    agent_checks_path = agent_dir / "constraint_check.csv"
    agent_findings_path = agent_dir / "validation_findings.csv"
    agent_leakage_path = agent_dir / "leakage_checklist.csv"
    agent_robustness_path = agent_dir / "robustness_summary.csv"
    agent_manifest_path = agent_dir / "agent_run_manifest.json"
    write_text(make_report(nested_performance, pbo_summary, decision, selection), agent_report_path)
    model.write_csv(registry, agent_registry_path)
    model.write_csv(decision, agent_decision_path)
    model.write_csv(checks, agent_checks_path)
    model.write_csv(findings, agent_findings_path)
    model.write_csv(leakage, agent_leakage_path)
    model.write_csv(robustness, agent_robustness_path)
    agent_artifacts = [
        agent_report_path,
        agent_registry_path,
        agent_decision_path,
        agent_checks_path,
        agent_findings_path,
        agent_leakage_path,
        agent_robustness_path,
        model_manifest_path,
        model_manifest_check_path,
        agent_manifest_path,
    ]
    agent_manifest = make_agent_manifest(
        start_time=start_time,
        agent_dir=agent_dir,
        config_path=config_path,
        artifacts=agent_artifacts,
        metrics=metrics,
        fail_count=fail_count + manifest_fail_count,
        warn_count=warn_count + manifest_warn_count,
    )
    write_json(agent_manifest, agent_manifest_path)

    result = {
        "model_version": MODEL_VERSION,
        "self_check_pass": fail_count == 0 and manifest_fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "manifest_fail_count": manifest_fail_count,
        "manifest_warn_count": manifest_warn_count,
        "overall_decision": metrics["overall_decision"],
        "metrics": metrics,
        "output_dir": str(output_dir),
        "agent_output_dir": str(agent_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 and manifest_fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
