#!/usr/bin/env python
"""HIRSSM V2.4 stable nested shrinkage selection.

V2.4 reduces the shrinkage grid to three precommitted families, selects with
10/20/30bps objectives, and penalizes switching the selected family from one
year to the next. It keeps V2.3's no-look-ahead nested selection discipline.
"""

from __future__ import annotations

import argparse
import importlib.util
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
V21_PATH = ROOT / "strategy_lab" / "hirssm_v2_1_walk_forward.py"
V22_PATH = ROOT / "strategy_lab" / "hirssm_v2_2_walk_forward.py"
V23_PATH = ROOT / "strategy_lab" / "hirssm_v2_3_nested_walk_forward.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_4_stable_nested_selection"


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


def variant_names(stable_cfg: dict) -> list[str]:
    return [str(item["variant"]) for item in stable_cfg.get("stable_variant_grid", [])]


def objective_score(prior: pd.DataFrame, baseline_prior: pd.DataFrame, objective: dict) -> dict:
    merged = prior.merge(
        baseline_prior[["year", "strategy_return"]].rename(columns={"strategy_return": "baseline_return"}),
        on="year",
        how="left",
    )
    mean_return = float(merged["strategy_return"].mean())
    excess_return = float((merged["strategy_return"] - merged["baseline_return"]).mean()) if merged["baseline_return"].notna().any() else 0.0
    vol = float(merged["strategy_return"].std(ddof=1)) if merged.shape[0] > 1 else 0.0
    worst_drawdown = float(merged["max_drawdown"].min()) if "max_drawdown" in merged.columns else 0.0
    avg_turnover = float(merged["avg_trade_turnover"].mean()) if "avg_trade_turnover" in merged.columns else 0.0
    underperformance_rate = float((merged["strategy_return"] < merged["baseline_return"]).mean()) if merged["baseline_return"].notna().any() else 0.0
    score = (
        float(objective.get("return_weight", 1.0)) * mean_return
        + float(objective.get("excess_return_weight", 0.6)) * excess_return
        - float(objective.get("volatility_penalty", 0.25)) * vol
        - float(objective.get("drawdown_penalty", 0.12)) * abs(min(worst_drawdown, 0.0))
        - float(objective.get("turnover_penalty", 0.03)) * avg_turnover
        - float(objective.get("underperformance_penalty", 0.03)) * underperformance_rate
    )
    return {
        "selection_score_raw": float(score),
        "train_mean_return": mean_return,
        "train_excess_return": excess_return,
        "train_return_vol": vol,
        "train_worst_drawdown": worst_drawdown,
        "train_avg_trade_turnover": avg_turnover,
        "train_underperformance_rate": underperformance_rate,
        "train_years_count": int(merged.shape[0]),
    }


def build_stable_selection(
    variant_yearly_by_cost: pd.DataFrame,
    baseline_yearly_by_cost: pd.DataFrame,
    config: dict,
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stable_cfg = config.get("expert_state_stable_selection", {})
    lookback = int(stable_cfg.get("selection_lookback_years", 5))
    min_years = int(stable_cfg.get("min_selection_years", 3))
    fallback = str(stable_cfg.get("fallback_variant", variants[0] if variants else ""))
    if fallback not in variants and variants:
        fallback = variants[0]
    switch_penalty = float(stable_cfg.get("switch_penalty", 0.015))
    objective = stable_cfg.get("objective", {})
    cost_weights = {float(cost): float(weight) for cost, weight in stable_cfg.get("objective_cost_weights", {}).items()}
    selection_costs = [float(item) for item in stable_cfg.get("selection_cost_bps_scenarios", [10, 20, 30])]
    if not cost_weights:
        equal = 1.0 / max(len(selection_costs), 1)
        cost_weights = {cost: equal for cost in selection_costs}
    years = sorted(int(year) for year in variant_yearly_by_cost["year"].dropna().unique())
    selections = []
    score_rows = []
    previous_variant = ""
    for test_year in years:
        prior_years = [year for year in years if test_year - lookback <= year < test_year]
        enough = len(prior_years) >= min_years
        if not enough:
            selected = previous_variant or fallback
            selections.append(
                {
                    "test_year": test_year,
                    "selected_variant": selected,
                    "previous_variant": previous_variant,
                    "selection_reason": f"fallback: only {len(prior_years)} prior years < {min_years}",
                    "prior_years": ",".join(str(year) for year in prior_years),
                }
            )
            previous_variant = selected
            continue
        for variant in variants:
            weighted_score = 0.0
            metrics_by_cost = {}
            valid_costs = 0
            for cost in selection_costs:
                prior = variant_yearly_by_cost[
                    variant_yearly_by_cost["variant"].eq(variant)
                    & variant_yearly_by_cost["cost_bps"].astype(float).eq(cost)
                    & variant_yearly_by_cost["year"].isin(prior_years)
                ]
                baseline_prior = baseline_yearly_by_cost[
                    baseline_yearly_by_cost["cost_bps"].astype(float).eq(cost)
                    & baseline_yearly_by_cost["year"].isin(prior_years)
                ]
                if prior.shape[0] < min_years:
                    continue
                metrics = objective_score(prior, baseline_prior, objective)
                weighted_score += cost_weights.get(cost, 0.0) * metrics["selection_score_raw"]
                metrics_by_cost[f"score_{int(cost)}bps"] = metrics["selection_score_raw"]
                metrics_by_cost[f"excess_{int(cost)}bps"] = metrics["train_excess_return"]
                metrics_by_cost[f"turnover_{int(cost)}bps"] = metrics["train_avg_trade_turnover"]
                valid_costs += 1
            if valid_costs == 0:
                continue
            switch_cost = switch_penalty if previous_variant and variant != previous_variant else 0.0
            selection_score = weighted_score - switch_cost
            score_rows.append(
                {
                    "test_year": test_year,
                    "variant": variant,
                    "previous_variant": previous_variant,
                    "prior_years": ",".join(str(year) for year in prior_years),
                    "weighted_score_before_switch": weighted_score,
                    "switch_penalty": switch_cost,
                    "selection_score": selection_score,
                    "valid_costs": valid_costs,
                    **metrics_by_cost,
                }
            )
        year_scores = pd.DataFrame([row for row in score_rows if row["test_year"] == test_year])
        if year_scores.empty:
            selected = previous_variant or fallback
            reason = "fallback: no variant had enough prior observations"
        else:
            best = year_scores.sort_values(
                ["selection_score", "weighted_score_before_switch"],
                ascending=False,
            ).iloc[0]
            selected = str(best["variant"])
            reason = (
                f"stable nested selected {selected}; score={float(best['selection_score']):.4f}; "
                f"switch_penalty={float(best['switch_penalty']):.4f}; prior_years={best['prior_years']}"
            )
        selections.append(
            {
                "test_year": test_year,
                "selected_variant": selected,
                "previous_variant": previous_variant,
                "selection_reason": reason,
                "prior_years": ",".join(str(year) for year in prior_years),
            }
        )
        previous_variant = selected
    return pd.DataFrame(selections), pd.DataFrame(score_rows)


def precompute_stable_variants(
    model,
    wf,
    v22,
    panel: dict,
    config: dict,
    specs: dict,
    monthly_ic: pd.DataFrame,
    stable_grid: list[dict],
    costs: list[float],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    variant_histories: dict[str, pd.DataFrame] = {}
    yearly_rows = []
    summary_rows = []
    for item in stable_grid:
        variant = str(item["variant"])
        for cost in costs:
            run = v22.run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, item, variant, cost_bps=float(cost))
            if variant not in variant_histories:
                variant_histories[variant] = run["history"]
            yearly = v23_yearly_diagnostics(run["bt"]["nav"], variant)
            if not yearly.empty:
                yearly["cost_bps"] = float(cost)
                yearly_rows.append(yearly)
            if not run["summary"].empty:
                summary_rows.append(run["summary"])
    yearly_df = pd.concat(yearly_rows, ignore_index=True, sort=False) if yearly_rows else pd.DataFrame()
    summary_df = pd.concat(summary_rows, ignore_index=True, sort=False) if summary_rows else pd.DataFrame()
    return variant_histories, yearly_df, summary_df


def v23_yearly_diagnostics(nav: pd.DataFrame, variant: str) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    df = nav.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    rows = []
    for year, group in df.groupby("year"):
        curve = (1.0 + group["portfolio_return"]).cumprod()
        bench_curve = (1.0 + group["benchmark_return"]).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        traded_turnover = group.loc[group["turnover"] > 0, "turnover"]
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "strategy_return": float(curve.iloc[-1] - 1.0),
                "benchmark_return": float(bench_curve.iloc[-1] - 1.0),
                "annual_vol": float(group["portfolio_return"].std(ddof=1) * np.sqrt(252)),
                "max_drawdown": float(drawdown.min()),
                "avg_turnover": float(group["turnover"].mean()),
                "avg_trade_turnover": float(traded_turnover.mean()) if not traded_turnover.empty else 0.0,
                "avg_cash_weight": float(group["cash_weight"].mean()),
                "days": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def baseline_yearly_by_cost(v23, baseline: dict) -> pd.DataFrame:
    rows = []
    for cost, nav in baseline.get("nav_by_cost", {}).items():
        yearly = v23.yearly_nav_diagnostics(nav, "same_period_v2_0_baseline")
        if not yearly.empty:
            yearly["cost_bps"] = float(cost)
            rows.append(yearly)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def pbo_report_for_primary_cost(v21, v23, variant_yearly_by_cost: pd.DataFrame, variant_summaries: pd.DataFrame, nested_summary: pd.DataFrame, primary_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly = variant_yearly_by_cost[variant_yearly_by_cost["cost_bps"].astype(float).eq(float(primary_cost))].copy()
    summaries = variant_summaries[variant_summaries["cost_bps"].astype(float).eq(float(primary_cost))].copy()
    pbo_report, pbo_splits = v21.run_cscv_pbo(yearly, summaries, selected_variant=str(summaries["variant"].iloc[0]) if not summaries.empty else "")
    nested_row = nested_summary[nested_summary["cost_bps"].astype(float).eq(float(primary_cost))].head(1)
    if not nested_row.empty and not yearly.empty:
        years = max(1, int(yearly["year"].nunique()))
        trial_count = max(1, int(yearly["variant"].nunique()))
        sharpe = float(nested_row["sharpe_no_rf"].iloc[0])
        dsr_proxy = sharpe - math.sqrt(2.0 * math.log(trial_count) / years)
        pbo_report = pd.concat(
            [
                pbo_report,
                pd.DataFrame(
                    [
                        {
                            "metric": "stable_nested_deflated_sharpe_proxy",
                            "value": dsr_proxy,
                            "pass": bool(dsr_proxy > 0),
                            "method": f"stable_nested_sharpe_minus_small_grid_penalty_{int(primary_cost)}bps",
                            "interpretation": f"Stable nested raw Sharpe {sharpe:.3f}; grid count {trial_count}.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    return pbo_report, pbo_splits


def make_reports(
    output_dir: Path,
    summaries: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    score_table: pd.DataFrame,
    variant_summaries: pd.DataFrame,
    pbo_report: pd.DataFrame,
    smoke: pd.DataFrame,
    monthly_underperf: pd.DataFrame,
    annual_underperf: pd.DataFrame,
    data_audit: pd.DataFrame,
    self_check: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_rate = float(monthly_underperf["underperformed_baseline"].mean()) if not monthly_underperf.empty else np.nan
    annual_rate = float(annual_underperf["underperformed_baseline"].mean()) if not annual_underperf.empty else np.nan
    if not selection.empty and "previous_variant" in selection.columns:
        previous = selection["previous_variant"]
        real_switch = previous.notna() & (previous.astype(str).str.len() > 0) & (selection["selected_variant"] != previous)
        switch_count = int(real_switch.sum())
    else:
        switch_count = 0
    lines = [
        "# HIRSSM V2.4 Stable Nested Selection Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- Three precommitted shrinkage families.",
        "- 10/20/30bps multi-cost selection objective.",
        "- Annual variant switching penalty.",
        "- Same-period V2.0 baseline and underperformance diagnostics remain mandatory.",
        "",
        "## OOS Performance",
        "",
        summaries.to_markdown(index=False) if not summaries.empty else "No summary.",
        "",
        "## Same-Period Baseline Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Selection Stability",
        "",
        f"- Variant switches: {switch_count}",
        "",
        selection.to_markdown(index=False) if not selection.empty else "No selection history.",
        "",
        "## Precomputed Stable Grid Summary",
        "",
        variant_summaries.sort_values(["cost_bps", "sharpe_no_rf"], ascending=[True, False]).to_markdown(index=False) if not variant_summaries.empty else "No variants.",
        "",
        "## PBO / DSR",
        "",
        pbo_report.to_markdown(index=False) if not pbo_report.empty else "No PBO report.",
        "",
        "## Underperformance Diagnostics",
        "",
        f"- Monthly underperformance rate vs V2.0 baseline: {monthly_rate:.2%}" if pd.notna(monthly_rate) else "- Monthly underperformance rate unavailable.",
        f"- Annual underperformance rate vs V2.0 baseline: {annual_rate:.2%}" if pd.notna(annual_rate) else "- Annual underperformance rate unavailable.",
        "",
        "## Self Check",
        "",
        "See `SELF_CHECK_REPORT.md` and `self_check_results.csv`.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    decision_lines = [
        "# HIRSSM V2.4 Selection Report",
        "",
        "## Selection History",
        "",
        selection.to_markdown(index=False) if not selection.empty else "No selection history.",
        "",
        "## Selection Scores",
        "",
        score_table.to_markdown(index=False) if not score_table.empty else "No score table.",
    ]
    (output_dir / "FACTOR_GATE_REPORT.md").write_text("\n".join(decision_lines), encoding="utf-8")

    changelog = [
        "# HIRSSM V2.4 Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Reduced shrinkage selection grid to three precommitted families.",
        "- Added 10/20/30bps multi-cost nested selection objective.",
        "- Added annual variant switching penalty.",
        "- Added self-check report as a required output.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        "# HIRSSM V2.4 Self Check\n\n" + (self_check.to_markdown(index=False) if not self_check.empty else "No self check."),
        encoding="utf-8",
    )


def make_self_check(smoke: pd.DataFrame, summaries: pd.DataFrame, comparison: pd.DataFrame, pbo_report: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    rows.append({"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""})
    rows.append({"check": "required_cost_rows", "pass": bool(set([5.0, 10.0, 20.0, 30.0]).issubset(set(summaries["cost_bps"].astype(float)))) if not summaries.empty else False, "detail": str(sorted(summaries["cost_bps"].astype(float).tolist())) if not summaries.empty else ""})
    row10 = comparison[comparison["variant"].eq("stable_nested_selection") & comparison["cost_bps"].astype(float).eq(10.0)] if not comparison.empty and "variant" in comparison.columns else pd.DataFrame()
    rows.append({"check": "beats_baseline_sharpe_10bps", "pass": bool(not row10.empty and float(row10["delta_sharpe_vs_same_period_baseline"].iloc[0]) > 0), "detail": f"{float(row10['delta_sharpe_vs_same_period_baseline'].iloc[0]):.6f}" if not row10.empty else ""})
    rows.append({"check": "max_drawdown_not_worse_10bps", "pass": bool(not row10.empty and float(row10["delta_mdd_vs_same_period_baseline"].iloc[0]) >= -0.03), "detail": f"{float(row10['delta_mdd_vs_same_period_baseline'].iloc[0]):.6f}" if not row10.empty else ""})
    pbo_row = pbo_report[pbo_report["metric"].eq("cscv_pbo")].head(1) if not pbo_report.empty else pd.DataFrame()
    rows.append({"check": "pbo_below_0_20", "pass": bool(not pbo_row.empty and float(pbo_row["value"].iloc[0]) < 0.20), "detail": f"{float(pbo_row['value'].iloc[0]):.6f}" if not pbo_row.empty else ""})
    for name in ["WALK_FORWARD_REPORT.md", "FACTOR_GATE_REPORT.md", "MODEL_CHANGELOG.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


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
    v21 = load_module("hirssm_v2_1_walk_forward", V21_PATH)
    v22 = load_module("hirssm_v2_2_walk_forward", V22_PATH)
    v23 = load_module("hirssm_v2_3_nested_walk_forward", V23_PATH)
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    stable_cfg = config.get("expert_state_stable_selection", {})
    stable_grid = stable_cfg.get("stable_variant_grid", [])
    variants = variant_names(stable_cfg)
    specs = v22.active_specs(wf, config)

    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    shrink_cfg = config.get("expert_state_shrinkage", {})
    monthly_ic = wf.compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, int(shrink_cfg.get("horizon_days", 21)))
    monthly_ic = v22.clean_monthly_ic(monthly_ic)

    selection_costs = [float(item) for item in stable_cfg.get("selection_cost_bps_scenarios", [10, 20, 30])]
    output_costs = [float(item) for item in stable_cfg.get("cost_bps_scenarios", [5, 10, 20, 30])]
    all_costs = sorted(set(selection_costs + output_costs))
    variant_histories, variant_yearly, variant_summaries = precompute_stable_variants(
        model, wf, v22, panel, config, specs, monthly_ic, stable_grid, all_costs
    )

    first_year = int(variant_yearly["year"].min()) if not variant_yearly.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_year, month=1, day=1)
    baseline = v21.run_same_period_baseline(model, panel, config, start_date, output_costs)
    selection_baseline = v21.run_same_period_baseline(model, panel, config, start_date, selection_costs)
    baseline_yearly = baseline_yearly_by_cost(v23, selection_baseline)

    selection, score_table = build_stable_selection(variant_yearly, baseline_yearly, config, variants)
    selected_history = v23.combine_selected_history(selection, variant_histories)

    summaries = []
    nav_by_cost = {}
    for cost in output_costs:
        run = v23.run_nested_portfolio(model, v22, panel, config, selected_history, cost_bps=cost)
        if not run["summary"].empty:
            run["summary"]["variant"] = "stable_nested_selection"
            summaries.append(run["summary"])
        nav_by_cost[cost] = run["bt"]["nav"]
        suffix = f"{int(cost)}bps"
        model.write_csv(run["bt"]["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(run["bt"]["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(run["bt"]["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(run["bt"]["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        if cost == 10.0:
            selected_run = run
    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()

    primary_cost = 20.0 if 20.0 in selection_costs else selection_costs[0]
    pbo_report, pbo_splits = pbo_report_for_primary_cost(v21, v23, variant_yearly, variant_summaries, summary_df, primary_cost)

    baseline_10 = baseline["summary"][baseline["summary"]["cost_bps"].astype(float).eq(10.0)].head(1)
    nested_10 = summary_df[summary_df["cost_bps"].astype(float).eq(10.0)].head(1)
    comparison = pd.DataFrame()
    if not baseline_10.empty and not nested_10.empty:
        comparison = pd.concat([baseline_10, nested_10], ignore_index=True, sort=False)
        base = baseline_10.iloc[0]
        comparison["delta_sharpe_vs_same_period_baseline"] = comparison["sharpe_no_rf"] - float(base["sharpe_no_rf"])
        comparison["delta_mdd_vs_same_period_baseline"] = comparison["max_drawdown"] - float(base["max_drawdown"])
        comparison["delta_annual_return_vs_same_period_baseline"] = comparison["annual_return"] - float(base["annual_return"])

    nested_nav10 = nav_by_cost.get(10.0, pd.DataFrame())
    baseline_nav10 = baseline["nav_by_cost"].get(10.0, pd.DataFrame())
    monthly_underperf, annual_underperf = v23.underperformance_reports(nested_nav10, baseline_nav10)
    smoke = wf.smoke_test_targets(selected_run["targets"])
    data_audit = wf.data_contract_audit(root, config)

    model.write_csv(monthly_ic, output_dir / "monthly_expert_rank_ic.csv")
    model.write_csv(selected_history, output_dir / "selected_expert_state_multiplier_history.csv")
    model.write_csv(selected_history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(selected_run["targets"], output_dir / "walk_forward_target_weights.csv")
    model.write_csv(selection, output_dir / "stable_nested_variant_selection.csv")
    model.write_csv(score_table, output_dir / "stable_nested_selection_scores.csv")
    model.write_csv(summary_df, output_dir / "oos_performance.csv")
    model.write_csv(baseline["summary"], output_dir / "same_period_baseline_summary.csv")
    model.write_csv(comparison, output_dir / "same_period_baseline_comparison.csv")
    model.write_csv(variant_summaries, output_dir / "stable_variant_summary.csv")
    model.write_csv(variant_yearly, output_dir / "stable_variant_yearly_diagnostics.csv")
    model.write_csv(monthly_underperf, output_dir / "monthly_underperformance_vs_baseline.csv")
    model.write_csv(annual_underperf, output_dir / "annual_underperformance_vs_baseline.csv")
    model.write_csv(pbo_report, output_dir / "pbo_cscv_report.csv")
    model.write_csv(pbo_splits, output_dir / "pbo_cscv_splits.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(data_audit, output_dir / "data_contract_audit.csv")

    self_check = make_self_check(smoke, summary_df, comparison, pbo_report, output_dir)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    make_reports(
        output_dir,
        summary_df,
        baseline["summary"],
        comparison,
        selection,
        score_table,
        variant_summaries,
        pbo_report,
        smoke,
        monthly_underperf,
        annual_underperf,
        data_audit,
        self_check,
    )
    self_check = make_self_check(smoke, summary_df, comparison, pbo_report, output_dir)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        "# HIRSSM V2.4 Self Check\n\n" + self_check.to_markdown(index=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "selection_years": int(selection.shape[0]),
                "score_rows": int(score_table.shape[0]),
                "target_rows": int(selected_run["targets"].shape[0]),
                "summary_rows": int(summary_df.shape[0]),
                "pbo_rows": int(pbo_report.shape[0]),
                "self_check_pass": bool(self_check["pass"].all()) if not self_check.empty else False,
                "smoke_pass": bool(smoke["pass"].all()) if not smoke.empty else False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
