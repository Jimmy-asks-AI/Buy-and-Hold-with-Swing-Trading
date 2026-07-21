#!/usr/bin/env python
"""HIRSSM V2.1 state-conditional walk-forward expert gates.

V2.1 upgrades V2.0 by gating experts at expert x market-state granularity.
The default variant is predeclared in config. A small predeclared grid is
evaluated with CSCV/PBO diagnostics, but grid results do not automatically
promote a parameter set to production.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
MODEL_PATH = ROOT / "strategy_lab" / "hirssm_v2_model.py"
WF_PATH = ROOT / "strategy_lab" / "hirssm_v2_walk_forward.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_1_walk_forward"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def active_specs(wf, config: dict) -> dict:
    gate_cfg = config.get("expert_state_gates", {})
    candidates = gate_cfg.get("candidate_experts", wf.EXPERT_SPECS.keys())
    return {name: wf.EXPERT_SPECS[name] for name in candidates if name in wf.EXPERT_SPECS}


def clean_monthly_ic(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    out = monthly_ic.copy()
    if out.empty:
        return out
    out["state"] = out["state"].replace("", np.nan).fillna("unknown").astype(str)
    return out


def thresholds_from_config(config: dict, override: dict | None = None) -> dict:
    gate_cfg = config.get("expert_state_gates", {})
    src = dict(gate_cfg)
    if override:
        src.update(override)
    return {
        "gate_mode": str(src.get("gate_mode", "positive_only")),
        "min_observations": int(src.get("min_observations", 8)),
        "rank_ic_mean_min": float(src.get("rank_ic_mean_min", 0.0)),
        "positive_ic_rate_min": float(src.get("positive_ic_rate_min", 0.52)),
        "icir_min": float(src.get("icir_min", 0.0)),
        "negative_rank_ic_mean_max": float(src.get("negative_rank_ic_mean_max", -0.02)),
        "negative_positive_ic_rate_max": float(src.get("negative_positive_ic_rate_max", 0.48)),
        "negative_icir_max": float(src.get("negative_icir_max", -0.05)),
    }


def state_allowed(config: dict, expert: str, state: str) -> bool:
    whitelist = config.get("expert_state_gates", {}).get("state_whitelist_by_expert", {})
    allowed_states = whitelist.get(expert)
    if not allowed_states:
        return True
    return state in set(str(item) for item in allowed_states)


def observation_requirement(config: dict, expert: str) -> int:
    gate_cfg = config.get("expert_state_gates", {})
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    if expert not in observation_experts:
        return 1
    by_expert = gate_cfg.get("consecutive_pass_windows_by_expert", {})
    return int(by_expert.get(expert, gate_cfg.get("observation_consecutive_pass_windows", 2)))


def gate_reason(
    stats: dict,
    thresholds: dict,
    raw_pass: bool,
    negative_gate_hit: bool,
    portfolio_enabled: bool,
    requires_positive_gate: bool,
    structurally_allowed: bool,
    consecutive_pass: int,
    consecutive_required: int,
) -> str:
    if not structurally_allowed:
        return "blocked by expert-state whitelist"
    if thresholds["gate_mode"] == "hybrid_bad_filter" and portfolio_enabled and not requires_positive_gate:
        return "enabled by baseline unless negative state evidence appears"
    if thresholds["gate_mode"] == "hybrid_bad_filter" and negative_gate_hit:
        return "disabled by negative state evidence"
    if raw_pass and consecutive_pass >= consecutive_required:
        return "passes state-specific past-window RankIC gates"
    reasons = []
    if stats["observations"] < thresholds["min_observations"]:
        reasons.append(f"observations {stats['observations']} < {thresholds['min_observations']}")
    if pd.isna(stats["rank_ic_mean"]) or stats["rank_ic_mean"] <= thresholds["rank_ic_mean_min"]:
        reasons.append(f"rank_ic_mean {stats['rank_ic_mean']:.4f} <= {thresholds['rank_ic_mean_min']:.4f}" if pd.notna(stats["rank_ic_mean"]) else "rank_ic_mean missing")
    if pd.isna(stats["positive_ic_rate"]) or stats["positive_ic_rate"] <= thresholds["positive_ic_rate_min"]:
        reasons.append(f"positive_ic_rate {stats['positive_ic_rate']:.2%} <= {thresholds['positive_ic_rate_min']:.2%}" if pd.notna(stats["positive_ic_rate"]) else "positive_ic_rate missing")
    if pd.isna(stats["rank_icir"]) or stats["rank_icir"] <= thresholds["icir_min"]:
        reasons.append(f"rank_icir {stats['rank_icir']:.4f} <= {thresholds['icir_min']:.4f}" if pd.notna(stats["rank_icir"]) else "rank_icir missing")
    if raw_pass and consecutive_pass < consecutive_required:
        reasons.append(f"consecutive_pass_windows {consecutive_pass} < {consecutive_required}")
    return "; ".join(reasons)


def build_state_gate_history(
    monthly_ic: pd.DataFrame,
    config: dict,
    wf,
    specs: dict,
    threshold_override: dict | None = None,
    variant: str = "predeclared_default",
) -> pd.DataFrame:
    gate_cfg = config.get("expert_state_gates", {})
    thresholds = thresholds_from_config(config, threshold_override)
    train_years = int(gate_cfg.get("train_years", 5))
    states = [str(item) for item in config.get("regime_model", {}).get("states", [])]
    candidates = [item for item in gate_cfg.get("candidate_experts", specs.keys()) if item in specs]
    default_disabled_experts = set(str(item) for item in config.get("disabled_experts_by_default", []))
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    if monthly_ic.empty:
        return pd.DataFrame()
    clean_ic = clean_monthly_ic(monthly_ic)
    years = sorted(int(y) for y in clean_ic["year"].dropna().unique())
    first_year = min(years) + train_years
    consecutive_pass = {(expert, state): 0 for expert in candidates for state in states}
    rows = []
    for test_year in [year for year in years if year >= first_year]:
        train_start = pd.Timestamp(year=test_year - train_years, month=1, day=1)
        train_end = pd.Timestamp(year=test_year, month=1, day=1)
        for expert in candidates:
            spec = specs[expert]
            applies_to_portfolio = bool(spec.get("applies_to_portfolio", True))
            for state in states:
                train_ic = clean_ic[
                    clean_ic["expert"].eq(expert)
                    & clean_ic["state"].eq(state)
                    & (clean_ic["date"] >= train_start)
                    & (clean_ic["date"] < train_end)
                ]
                stats = wf.summarize_ic(train_ic)
                structurally_allowed = state_allowed(config, expert, state)
                raw_pass = (
                    structurally_allowed
                    and stats["observations"] >= thresholds["min_observations"]
                    and pd.notna(stats["rank_ic_mean"])
                    and stats["rank_ic_mean"] > thresholds["rank_ic_mean_min"]
                    and pd.notna(stats["positive_ic_rate"])
                    and stats["positive_ic_rate"] > thresholds["positive_ic_rate_min"]
                    and pd.notna(stats["rank_icir"])
                    and stats["rank_icir"] > thresholds["icir_min"]
                )
                negative_gate_hit = (
                    structurally_allowed
                    and stats["observations"] >= thresholds["min_observations"]
                    and pd.notna(stats["rank_ic_mean"])
                    and stats["rank_ic_mean"] < thresholds["negative_rank_ic_mean_max"]
                    and pd.notna(stats["positive_ic_rate"])
                    and stats["positive_ic_rate"] < thresholds["negative_positive_ic_rate_max"]
                    and pd.notna(stats["rank_icir"])
                    and stats["rank_icir"] < thresholds["negative_icir_max"]
                )
                key = (expert, state)
                consecutive_pass[key] = consecutive_pass[key] + 1 if raw_pass else 0
                consecutive_required = observation_requirement(config, expert)
                requires_positive_gate = (
                    thresholds["gate_mode"] == "positive_only"
                    or expert in default_disabled_experts
                    or expert in observation_experts
                )
                if thresholds["gate_mode"] == "hybrid_bad_filter" and not requires_positive_gate:
                    portfolio_enabled = bool(structurally_allowed and not negative_gate_hit)
                else:
                    portfolio_enabled = bool(raw_pass and consecutive_pass[key] >= consecutive_required)
                rows.append(
                    {
                        "variant": variant,
                        "gate_mode": thresholds["gate_mode"],
                        "test_year": int(test_year),
                        "state": state,
                        "train_start": train_start.date().isoformat(),
                        "train_end": (train_end - pd.Timedelta(days=1)).date().isoformat(),
                        "expert": expert,
                        "asset_type_scope": spec["asset_type"] or "all",
                        "score_col": spec["score_col"],
                        "observations": stats["observations"],
                        "rank_ic_mean": stats["rank_ic_mean"],
                        "rank_ic_std": stats["rank_ic_std"],
                        "rank_icir": stats["rank_icir"],
                        "positive_ic_rate": stats["positive_ic_rate"],
                        "structurally_allowed": structurally_allowed,
                        "raw_gate_pass": raw_pass,
                        "negative_gate_hit": negative_gate_hit,
                        "requires_positive_gate": requires_positive_gate,
                        "consecutive_pass_windows": consecutive_pass[key],
                        "consecutive_required": consecutive_required,
                        "portfolio_enabled": portfolio_enabled,
                        "applies_to_portfolio": applies_to_portfolio,
                        "disabled_for_year_state": bool(applies_to_portfolio and not portfolio_enabled),
                        "reason": gate_reason(
                            stats,
                            thresholds,
                            raw_pass,
                            negative_gate_hit,
                            portfolio_enabled,
                            requires_positive_gate,
                            structurally_allowed,
                            consecutive_pass[key],
                            consecutive_required,
                        ),
                        "economic_logic": spec["economic_logic"],
                        "failure_scenario": spec["failure_scenario"],
                    }
                )
    history = pd.DataFrame(rows)
    if history.empty:
        return history
    test_stats = (
        clean_ic.groupby(["year", "state", "expert"], dropna=False)
        .agg(
            test_rank_ic_mean=("rank_ic", "mean"),
            test_positive_ic_rate=("rank_ic", lambda s: float((s > 0).mean())),
            test_observations=("rank_ic", "count"),
        )
        .reset_index()
        .rename(columns={"year": "test_year"})
    )
    return history.merge(test_stats, on=["test_year", "state", "expert"], how="left")


def build_disabled_by_year_state(gate_history: pd.DataFrame) -> dict[tuple[int, str], set[str]]:
    disabled: dict[tuple[int, str], set[str]] = {}
    if gate_history.empty:
        return disabled
    applied = gate_history[gate_history["applies_to_portfolio"].astype(bool)]
    for (year, state), group in applied.groupby(["test_year", "state"]):
        disabled[(int(year), str(state))] = set(group.loc[group["disabled_for_year_state"].astype(bool), "expert"].astype(str))
    return disabled


def add_state_gate_columns(targets: pd.DataFrame, gate_history: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or gate_history.empty:
        return targets
    out = targets.copy()
    out["test_year"] = pd.to_datetime(out["signal_date"]).dt.year
    gate_summary = []
    applied = gate_history[gate_history["applies_to_portfolio"].astype(bool)]
    for (year, state), group in applied.groupby(["test_year", "state"]):
        enabled = sorted(group.loc[group["portfolio_enabled"].astype(bool), "expert"].astype(str).tolist())
        disabled = sorted(group.loc[group["disabled_for_year_state"].astype(bool), "expert"].astype(str).tolist())
        gate_summary.append(
            {
                "test_year": int(year),
                "state": str(state),
                "enabled_experts": ",".join(enabled),
                "disabled_experts": ",".join(disabled),
            }
        )
    return out.merge(pd.DataFrame(gate_summary), on=["test_year", "state"], how="left")


def run_state_gate_variant(
    model,
    wf,
    panel: dict,
    config: dict,
    specs: dict,
    monthly_ic: pd.DataFrame,
    threshold_override: dict | None,
    variant: str,
    cost_bps: float = 10.0,
) -> dict:
    gate_history = build_state_gate_history(monthly_ic, config, wf, specs, threshold_override, variant=variant)
    disabled_map = build_disabled_by_year_state(gate_history)
    first_test_year = int(gate_history["test_year"].min()) if not gate_history.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_test_year, month=1, day=1)
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=set(),
        disabled_experts_by_year_state=disabled_map,
    )
    targets = add_state_gate_columns(targets, gate_history)
    if not targets.empty and "test_year" in targets.columns:
        targets = targets[targets["test_year"].isin(gate_history["test_year"].unique())]
    bt = model.run_backtest(panel["returns"], targets, cost_bps, panel["broad_code"])
    summary = model.summarize_nav(bt["nav"])
    yearly = model.yearly_returns(bt["nav"])
    if not summary.empty:
        summary.insert(0, "variant", variant)
        summary.insert(1, "cost_bps", float(cost_bps))
        summary["target_rows"] = int(targets.shape[0])
        summary["oos_start"] = bt["nav"]["date"].min() if not bt["nav"].empty else pd.NaT
        summary["oos_end"] = bt["nav"]["date"].max() if not bt["nav"].empty else pd.NaT
        summary["oos_years"] = (
            (pd.to_datetime(bt["nav"]["date"].max()) - pd.to_datetime(bt["nav"]["date"].min())).days / 365.25
            if not bt["nav"].empty
            else np.nan
        )
    return {
        "variant": variant,
        "gate_history": gate_history,
        "disabled_map": disabled_map,
        "targets": targets,
        "bt": bt,
        "summary": summary,
        "yearly": yearly.assign(variant=variant) if not yearly.empty else yearly,
        "start_date": start_date,
    }


def run_same_period_baseline(model, panel: dict, config: dict, start_date: pd.Timestamp, cost_scenarios: list[float]) -> dict:
    default_disabled = set(str(item) for item in config.get("disabled_experts_by_default", []))
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=default_disabled,
    )
    summaries = []
    nav_by_cost = {}
    for cost in cost_scenarios:
        bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
        nav_by_cost[float(cost)] = bt["nav"]
        summary = model.summarize_nav(bt["nav"])
        if not summary.empty:
            summary.insert(0, "variant", "same_period_v2_0_baseline")
            summary.insert(1, "cost_bps", float(cost))
            summary["target_rows"] = int(targets.shape[0])
            summaries.append(summary)
    return {
        "targets": targets,
        "summary": pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame(),
        "nav_by_cost": nav_by_cost,
    }


def sharpe_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.shape[0] < 2:
        return np.nan
    std = values.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return float(values.mean() / std)


def run_cscv_pbo(variant_yearly: pd.DataFrame, variant_summaries: pd.DataFrame, selected_variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if variant_yearly.empty:
        return pd.DataFrame(), pd.DataFrame()
    yearly = variant_yearly[["variant", "year", "strategy_return"]].dropna().copy()
    variants = sorted(yearly["variant"].unique())
    years = sorted(int(year) for year in yearly["year"].unique())
    if len(variants) < 2 or len(years) < 6:
        return pd.DataFrame(), pd.DataFrame()
    n_blocks = min(10, len(years) if len(years) % 2 == 0 else len(years) - 1)
    n_blocks = max(6, n_blocks)
    year_blocks = {}
    for block_id, block_years in enumerate(np.array_split(years, n_blocks), start=1):
        for year in block_years:
            year_blocks[int(year)] = block_id
    blocks = sorted(set(year_blocks.values()))
    split_rows = []
    for split_id, is_blocks in enumerate(itertools.combinations(blocks, len(blocks) // 2), start=1):
        is_blocks = set(is_blocks)
        is_years = {year for year, block in year_blocks.items() if block in is_blocks}
        oos_years = set(year_blocks) - is_years
        scores = []
        for variant in variants:
            group = yearly[yearly["variant"].eq(variant)]
            is_score = sharpe_from_returns(group[group["year"].isin(is_years)]["strategy_return"])
            oos_score = sharpe_from_returns(group[group["year"].isin(oos_years)]["strategy_return"])
            scores.append({"variant": variant, "is_score": is_score, "oos_score": oos_score})
        score_df = pd.DataFrame(scores).dropna(subset=["is_score", "oos_score"])
        if score_df.empty or score_df["variant"].nunique() < 2:
            continue
        best = score_df.sort_values("is_score", ascending=False).iloc[0]
        score_df["oos_rank"] = score_df["oos_score"].rank(ascending=False, method="average")
        n_variants = score_df.shape[0]
        best_rank = float(score_df.loc[score_df["variant"].eq(best["variant"]), "oos_rank"].iloc[0])
        lambda_rank = 1.0 - (best_rank - 1.0) / max(n_variants - 1, 1)
        lambda_rank = min(max(lambda_rank, 1e-6), 1 - 1e-6)
        split_rows.append(
            {
                "split_id": split_id,
                "best_is_variant": best["variant"],
                "best_is_score": float(best["is_score"]),
                "best_oos_score": float(best["oos_score"]),
                "best_oos_rank": best_rank,
                "variant_count": n_variants,
                "lambda_rank": lambda_rank,
                "logit_lambda": math.log(lambda_rank / (1.0 - lambda_rank)),
                "oos_under_median": bool(lambda_rank < 0.5),
                "is_years": ",".join(str(year) for year in sorted(is_years)),
                "oos_years": ",".join(str(year) for year in sorted(oos_years)),
            }
        )
    splits = pd.DataFrame(split_rows)
    if splits.empty:
        return pd.DataFrame(), splits
    pbo = float((splits["logit_lambda"] < 0).mean())
    summary_row = {
        "metric": "cscv_pbo",
        "value": pbo,
        "pass": bool(pbo < 0.2),
        "method": "CSCV annual-return Sharpe ranking over predeclared state-gate grid",
        "interpretation": "Fraction of splits where the in-sample best variant ranks below median out of sample.",
    }
    selected = variant_summaries[variant_summaries["variant"].eq(selected_variant)].head(1)
    if selected.empty:
        dsr_proxy = np.nan
        selected_sharpe = np.nan
        years_count = len(years)
    else:
        selected_sharpe = float(selected["sharpe_no_rf"].iloc[0])
        years_count = max(1, len(years))
        dsr_proxy = selected_sharpe - math.sqrt(2.0 * math.log(len(variants)) / years_count)
    report = pd.DataFrame(
        [
            summary_row,
            {
                "metric": "deflated_sharpe_proxy",
                "value": dsr_proxy,
                "pass": bool(pd.notna(dsr_proxy) and dsr_proxy > 0),
                "method": "selected_variant_sharpe_minus_multiple_testing_penalty",
                "interpretation": f"Selected variant {selected_variant}; raw Sharpe {selected_sharpe:.3f}.",
            },
            {
                "metric": "cscv_split_count",
                "value": float(splits.shape[0]),
                "pass": bool(splits.shape[0] > 0),
                "method": "chronological annual blocks",
                "interpretation": "Number of CSCV train/test combinations evaluated.",
            },
        ]
    )
    return report, splits


def make_reports(
    output_dir: Path,
    selected_variant: dict,
    baseline_summary: pd.DataFrame,
    variant_summaries: pd.DataFrame,
    gate_history: pd.DataFrame,
    pbo_report: pd.DataFrame,
    smoke: pd.DataFrame,
    data_audit: pd.DataFrame,
    nav10: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_summary = selected_variant["summary"]
    selected_10 = selected_summary[selected_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    baseline_10 = baseline_summary[baseline_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    comparison = pd.concat([baseline_10, selected_10], ignore_index=True, sort=False) if not selected_10.empty else pd.DataFrame()
    lines = [
        "# HIRSSM V2.1 Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- State-conditional expert gates: expert x regime x asset type.",
        "- Default gate mode is hybrid: keep the V2.0 core unless an expert-state pair has clear negative IC evidence; observation experts still require positive gates.",
        "- `industry_trend_continuation` is structurally restricted to crash_rebound and range_bound.",
        "- `range_reversal` is structurally restricted to risk_off_decline and crash_rebound.",
        "- Same-period V2.0 baseline is included to prevent start-date illusion.",
        "- PBO uses a predeclared state-gate parameter grid; grid search does not automatically promote defaults.",
        "",
        "## Selected Predeclared Variant",
        "",
        selected_summary.to_markdown(index=False) if not selected_summary.empty else "No selected summary.",
        "",
        "## Same-Period Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Predeclared Grid Summary",
        "",
        variant_summaries.sort_values(["sharpe_no_rf", "annual_return"], ascending=False).to_markdown(index=False) if not variant_summaries.empty else "No variants.",
        "",
        "## CSCV / PBO",
        "",
        pbo_report.to_markdown(index=False) if not pbo_report.empty else "No PBO report.",
        "",
        "## Smoke Tests",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke tests.",
        "",
        "## Data Contract Audit",
        "",
        data_audit.to_markdown(index=False) if not data_audit.empty else "No data audit.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    gate_summary = pd.DataFrame()
    if not gate_history.empty:
        gate_summary = (
            gate_history[gate_history["applies_to_portfolio"].astype(bool)]
            .groupby(["expert", "state"], dropna=False)
            .agg(
                windows=("test_year", "count"),
                enabled_rate=("portfolio_enabled", "mean"),
                raw_pass_rate=("raw_gate_pass", "mean"),
                avg_train_rank_ic=("rank_ic_mean", "mean"),
                avg_test_rank_ic=("test_rank_ic_mean", "mean"),
                avg_positive_ic_rate=("positive_ic_rate", "mean"),
            )
            .reset_index()
            .sort_values(["enabled_rate", "avg_train_rank_ic"], ascending=False)
        )
    factor_lines = [
        "# HIRSSM V2.1 Factor Gate Report",
        "",
        "## Expert x State Gate Summary",
        "",
        gate_summary.to_markdown(index=False) if not gate_summary.empty else "No gate summary.",
        "",
        "## Latest Decisions",
        "",
        gate_history.sort_values(["test_year", "state", "expert"]).tail(60).to_markdown(index=False) if not gate_history.empty else "No gate history.",
    ]
    (output_dir / "FACTOR_GATE_REPORT.md").write_text("\n".join(factor_lines), encoding="utf-8")

    changelog = [
        "# HIRSSM V2.1 Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Added expert x market-state walk-forward gates.",
        "- Added state-level disabled expert maps in target construction.",
        "- Added same-period V2.0 baseline comparison.",
        "- Added CSCV/PBO over a predeclared state-gate parameter grid.",
        "- Kept V2.0 outputs intact; V2.1 writes to a separate output directory.",
        "",
        "## Governance",
        "",
        "- Do not promote a variant solely because it wins the grid.",
        "- Require OOS performance, PBO, cost robustness, and failure-case review before default promotion.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")

    if nav10.empty:
        failure_lines = ["# Regime Failure Cases", "", "No 10bps NAV to analyze."]
    else:
        dd = nav10.copy()
        dd["drawdown"] = dd["nav"] / dd["nav"].cummax() - 1.0
        dd = dd.merge(regimes[["date", "state"]], on="date", how="left")
        worst_days = dd.sort_values("drawdown").head(20)[["date", "state", "portfolio_return", "nav", "drawdown", "cash_weight"]]
        by_state = (
            dd.groupby("state", dropna=False)
            .agg(
                worst_drawdown=("drawdown", "min"),
                avg_return=("portfolio_return", "mean"),
                days=("date", "count"),
                avg_cash_weight=("cash_weight", "mean"),
            )
            .reset_index()
            .sort_values("worst_drawdown")
        )
        failure_lines = [
            "# HIRSSM V2.1 Regime Failure Cases",
            "",
            "## Worst Drawdown Days",
            "",
            worst_days.to_markdown(index=False),
            "",
            "## Drawdown By Regime",
            "",
            by_state.to_markdown(index=False),
        ]
    (output_dir / "REGIME_FAILURE_CASES.md").write_text("\n".join(failure_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    specs = active_specs(wf, config)
    gate_cfg = config.get("expert_state_gates", {})

    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    horizon = int(gate_cfg.get("horizon_days", 21))
    monthly_ic = wf.compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, horizon)
    monthly_ic = clean_monthly_ic(monthly_ic)

    default_variant = "predeclared_default"
    selected = run_state_gate_variant(model, wf, panel, config, specs, monthly_ic, None, default_variant, cost_bps=10.0)
    start_date = selected["start_date"]

    cost_scenarios = [float(item) for item in config["validation"]["cost_bps_scenarios"]]
    selected_summaries = []
    nav_by_cost = {}
    trade_by_cost = {}
    for cost in cost_scenarios:
        cost_run = run_state_gate_variant(model, wf, panel, config, specs, monthly_ic, None, default_variant, cost_bps=cost)
        selected_summaries.append(cost_run["summary"])
        nav_by_cost[cost] = cost_run["bt"]["nav"]
        trade_by_cost[cost] = cost_run["bt"]["trades"]
        suffix = f"{int(cost)}bps"
        model.write_csv(cost_run["bt"]["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(cost_run["bt"]["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(cost_run["bt"]["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(cost_run["bt"]["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        if cost == 10.0:
            selected = cost_run
    selected["summary"] = pd.concat(selected_summaries, ignore_index=True, sort=False) if selected_summaries else pd.DataFrame()

    baseline = run_same_period_baseline(model, panel, config, start_date, cost_scenarios)

    variant_rows = []
    variant_yearly = []
    grid = gate_cfg.get("cscv_pbo_grid", [])
    for item in grid:
        variant = str(item.get("variant", f"grid_{len(variant_rows) + 1}"))
        run = run_state_gate_variant(model, wf, panel, config, specs, monthly_ic, item, variant, cost_bps=10.0)
        if not run["summary"].empty:
            variant_rows.append(run["summary"])
        if not run["yearly"].empty:
            variant_yearly.append(run["yearly"])
    variant_summaries = pd.concat(variant_rows, ignore_index=True, sort=False) if variant_rows else pd.DataFrame()
    variant_yearly_df = pd.concat(variant_yearly, ignore_index=True, sort=False) if variant_yearly else pd.DataFrame()
    pbo_report, pbo_splits = run_cscv_pbo(variant_yearly_df, variant_summaries, selected_variant=grid[0]["variant"] if grid else default_variant)

    selected_10 = selected["summary"][selected["summary"]["cost_bps"].astype(float).eq(10.0)].head(1)
    baseline_10 = baseline["summary"][baseline["summary"]["cost_bps"].astype(float).eq(10.0)].head(1)
    comparison = pd.DataFrame()
    if not selected_10.empty and not baseline_10.empty:
        comparison = pd.concat([baseline_10, selected_10], ignore_index=True, sort=False)
        base = baseline_10.iloc[0]
        comparison["delta_sharpe_vs_same_period_baseline"] = comparison["sharpe_no_rf"] - float(base["sharpe_no_rf"])
        comparison["delta_mdd_vs_same_period_baseline"] = comparison["max_drawdown"] - float(base["max_drawdown"])
        comparison["delta_annual_return_vs_same_period_baseline"] = comparison["annual_return"] - float(base["annual_return"])

    smoke = wf.smoke_test_targets(selected["targets"])
    data_audit = wf.data_contract_audit(root, config)
    gate_history = selected["gate_history"]

    model.write_csv(monthly_ic, output_dir / "monthly_expert_rank_ic.csv")
    model.write_csv(gate_history, output_dir / "expert_state_gate_history.csv")
    model.write_csv(gate_history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(selected["targets"], output_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected["summary"], output_dir / "oos_performance.csv")
    model.write_csv(baseline["summary"], output_dir / "same_period_baseline_summary.csv")
    model.write_csv(comparison, output_dir / "same_period_baseline_comparison.csv")
    model.write_csv(variant_summaries, output_dir / "state_gate_variant_summary.csv")
    model.write_csv(variant_yearly_df, output_dir / "state_gate_variant_yearly_returns.csv")
    model.write_csv(pbo_report, output_dir / "pbo_cscv_report.csv")
    model.write_csv(pbo_splits, output_dir / "pbo_cscv_splits.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(data_audit, output_dir / "data_contract_audit.csv")

    make_reports(
        output_dir,
        selected,
        baseline["summary"],
        variant_summaries,
        gate_history,
        pbo_report,
        smoke,
        data_audit,
        nav_by_cost.get(10.0, pd.DataFrame()),
        panel["regimes"],
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "gate_rows": int(gate_history.shape[0]),
                "target_rows": int(selected["targets"].shape[0]),
                "variant_rows": int(variant_summaries.shape[0]),
                "pbo_rows": int(pbo_report.shape[0]),
                "smoke_pass": bool(smoke["pass"].all()) if not smoke.empty else False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
