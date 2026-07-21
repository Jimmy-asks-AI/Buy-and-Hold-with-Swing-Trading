#!/usr/bin/env python
"""HIRSSM V2.3 nested walk-forward shrinkage selection.

V2.3 prevents full-OOS grid selection by choosing the shrinkage variant for
each test year using only prior realized walk-forward years. The selected
annual multiplier histories are then stitched into one portfolio and tested
against the same-period V2.0 baseline.
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
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_3_nested_walk_forward"


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


def yearly_nav_diagnostics(nav: pd.DataFrame, variant: str) -> pd.DataFrame:
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


def monthly_returns(nav: pd.DataFrame, variant: str) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    df = nav.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        rows.append(
            {
                "variant": variant,
                "month": month,
                "strategy_return": float((1.0 + group["portfolio_return"]).prod() - 1.0),
                "benchmark_return": float((1.0 + group["benchmark_return"]).prod() - 1.0),
                "avg_cash_weight": float(group["cash_weight"].mean()),
                "avg_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
        + float(objective.get("excess_return_weight", 0.5)) * excess_return
        - float(objective.get("volatility_penalty", 0.25)) * vol
        - float(objective.get("drawdown_penalty", 0.10)) * abs(min(worst_drawdown, 0.0))
        - float(objective.get("turnover_penalty", 0.02)) * avg_turnover
        - float(objective.get("underperformance_penalty", 0.02)) * underperformance_rate
    )
    return {
        "selection_score": float(score),
        "train_mean_return": mean_return,
        "train_excess_return": excess_return,
        "train_return_vol": vol,
        "train_worst_drawdown": worst_drawdown,
        "train_avg_trade_turnover": avg_turnover,
        "train_underperformance_rate": underperformance_rate,
        "train_years_count": int(merged.shape[0]),
    }


def build_nested_selection(
    variant_yearly: pd.DataFrame,
    baseline_yearly: pd.DataFrame,
    config: dict,
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nested_cfg = config.get("expert_state_nested_selection", {})
    lookback = int(nested_cfg.get("selection_lookback_years", 5))
    min_years = int(nested_cfg.get("min_selection_years", 3))
    fallback = str(nested_cfg.get("fallback_variant", variants[0] if variants else ""))
    if fallback not in variants and variants:
        fallback = variants[0]
    objective = nested_cfg.get("objective", {})
    years = sorted(int(year) for year in variant_yearly["year"].dropna().unique())
    selections = []
    score_rows = []
    for test_year in years:
        prior_years = [year for year in years if test_year - lookback <= year < test_year]
        enough = len(prior_years) >= min_years
        if not enough:
            selections.append(
                {
                    "test_year": test_year,
                    "selected_variant": fallback,
                    "selection_reason": f"fallback: only {len(prior_years)} prior years < {min_years}",
                    "prior_years": ",".join(str(year) for year in prior_years),
                }
            )
            continue
        baseline_prior = baseline_yearly[baseline_yearly["year"].isin(prior_years)]
        for variant in variants:
            prior = variant_yearly[variant_yearly["variant"].eq(variant) & variant_yearly["year"].isin(prior_years)]
            if prior.shape[0] < min_years:
                continue
            metrics = objective_score(prior, baseline_prior, objective)
            score_rows.append({"test_year": test_year, "variant": variant, "prior_years": ",".join(str(year) for year in prior_years), **metrics})
        year_scores = pd.DataFrame([row for row in score_rows if row["test_year"] == test_year])
        if year_scores.empty:
            selected = fallback
            reason = "fallback: no variant had enough prior observations"
        else:
            best = year_scores.sort_values(
                ["selection_score", "train_excess_return", "train_mean_return"],
                ascending=False,
            ).iloc[0]
            selected = str(best["variant"])
            reason = (
                f"nested objective selected {selected}; score={float(best['selection_score']):.4f}; "
                f"prior_years={best['prior_years']}"
            )
        selections.append(
            {
                "test_year": test_year,
                "selected_variant": selected,
                "selection_reason": reason,
                "prior_years": ",".join(str(year) for year in prior_years),
            }
        )
    return pd.DataFrame(selections), pd.DataFrame(score_rows)


def combine_selected_history(selection: pd.DataFrame, variant_histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for _, row in selection.iterrows():
        variant = str(row["selected_variant"])
        year = int(row["test_year"])
        hist = variant_histories.get(variant, pd.DataFrame())
        if hist.empty:
            continue
        selected = hist[hist["test_year"].eq(year)].copy()
        selected["selected_variant"] = variant
        selected["selection_reason"] = row["selection_reason"]
        frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def run_nested_portfolio(
    model,
    v22,
    panel: dict,
    config: dict,
    selected_history: pd.DataFrame,
    cost_bps: float,
) -> dict:
    multiplier_map = v22.build_multiplier_map(selected_history)
    first_year = int(selected_history["test_year"].min()) if not selected_history.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_year, month=1, day=1)
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=set(),
        expert_multipliers_by_year_state=multiplier_map,
    )
    targets = v22.add_multiplier_columns(targets, selected_history)
    if not targets.empty:
        targets["test_year"] = pd.to_datetime(targets["signal_date"]).dt.year
        targets = targets[targets["test_year"].isin(selected_history["test_year"].unique())]
    bt = model.run_backtest(panel["returns"], targets, cost_bps, panel["broad_code"])
    summary = model.summarize_nav(bt["nav"])
    if not summary.empty:
        summary.insert(0, "variant", "nested_selection")
        summary.insert(1, "cost_bps", float(cost_bps))
        summary["target_rows"] = int(targets.shape[0])
        summary["oos_start"] = bt["nav"]["date"].min() if not bt["nav"].empty else pd.NaT
        summary["oos_end"] = bt["nav"]["date"].max() if not bt["nav"].empty else pd.NaT
        summary["oos_years"] = (
            (pd.to_datetime(bt["nav"]["date"].max()) - pd.to_datetime(bt["nav"]["date"].min())).days / 365.25
            if not bt["nav"].empty
            else np.nan
        )
    return {"targets": targets, "bt": bt, "summary": summary, "start_date": start_date}


def nested_pbo_report(v21, variant_yearly: pd.DataFrame, variant_summaries: pd.DataFrame, nested_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbo_report, pbo_splits = v21.run_cscv_pbo(
        variant_yearly[["year", "strategy_return", "benchmark_return", "variant"]],
        variant_summaries,
        selected_variant=str(variant_summaries["variant"].iloc[0]) if not variant_summaries.empty else "",
    )
    nested_10 = nested_summary[nested_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    if not nested_10.empty and not variant_yearly.empty:
        years = max(1, int(variant_yearly["year"].nunique()))
        trial_count = max(1, int(variant_yearly["variant"].nunique()))
        sharpe = float(nested_10["sharpe_no_rf"].iloc[0])
        dsr_proxy = sharpe - math.sqrt(2.0 * math.log(trial_count) / years)
        nested_rows = pd.DataFrame(
            [
                {
                    "metric": "nested_selection_deflated_sharpe_proxy",
                    "value": dsr_proxy,
                    "pass": bool(dsr_proxy > 0),
                    "method": "nested_selection_sharpe_minus_grid_multiple_testing_penalty",
                    "interpretation": f"Nested selection raw Sharpe {sharpe:.3f}; grid count {trial_count}.",
                }
            ]
        )
        pbo_report = pd.concat([pbo_report, nested_rows], ignore_index=True, sort=False)
    return pbo_report, pbo_splits


def underperformance_reports(nested_nav: pd.DataFrame, baseline_nav: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nested_monthly = monthly_returns(nested_nav, "nested_selection")
    baseline_monthly = monthly_returns(baseline_nav, "same_period_v2_0_baseline").rename(
        columns={"strategy_return": "baseline_strategy_return", "benchmark_return": "baseline_benchmark_return"}
    )
    monthly = nested_monthly.merge(
        baseline_monthly[["month", "baseline_strategy_return", "baseline_benchmark_return"]],
        on="month",
        how="left",
    )
    monthly["excess_vs_baseline"] = monthly["strategy_return"] - monthly["baseline_strategy_return"]
    monthly["underperformed_baseline"] = monthly["excess_vs_baseline"] < 0
    nested_yearly = yearly_nav_diagnostics(nested_nav, "nested_selection")
    baseline_yearly = yearly_nav_diagnostics(baseline_nav, "same_period_v2_0_baseline").rename(
        columns={"strategy_return": "baseline_strategy_return", "max_drawdown": "baseline_max_drawdown"}
    )
    annual = nested_yearly.merge(
        baseline_yearly[["year", "baseline_strategy_return", "baseline_max_drawdown"]],
        on="year",
        how="left",
    )
    annual["excess_vs_baseline"] = annual["strategy_return"] - annual["baseline_strategy_return"]
    annual["drawdown_delta_vs_baseline"] = annual["max_drawdown"] - annual["baseline_max_drawdown"]
    annual["underperformed_baseline"] = annual["excess_vs_baseline"] < 0
    return monthly, annual


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
    nav10: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_rate = float(monthly_underperf["underperformed_baseline"].mean()) if not monthly_underperf.empty else np.nan
    annual_rate = float(annual_underperf["underperformed_baseline"].mean()) if not annual_underperf.empty else np.nan
    lines = [
        "# HIRSSM V2.3 Nested Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- Nested yearly shrinkage variant selection.",
        "- Variant choice for a test year uses only prior realized OOS years.",
        "- Objective includes return, excess return, volatility, drawdown, turnover, and underperformance penalties.",
        "- Same-period V2.0 baseline, cost robustness, underperformance diagnostics, and PBO remain mandatory.",
        "",
        "## Nested OOS Performance",
        "",
        summaries.to_markdown(index=False) if not summaries.empty else "No summary.",
        "",
        "## Same-Period Baseline Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Selected Variants By Year",
        "",
        selection.to_markdown(index=False) if not selection.empty else "No selection history.",
        "",
        "## Precomputed Grid Summary",
        "",
        variant_summaries.sort_values(["sharpe_no_rf", "annual_return"], ascending=False).to_markdown(index=False) if not variant_summaries.empty else "No variants.",
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
        "## Smoke Tests",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke tests.",
        "",
        "## Data Contract Audit",
        "",
        data_audit.to_markdown(index=False) if not data_audit.empty else "No data audit.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    decision_lines = [
        "# HIRSSM V2.3 Selection Report",
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
        "# HIRSSM V2.3 Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Added nested yearly selection of predeclared shrinkage variants.",
        "- Added turnover-aware training objective.",
        "- Added monthly and annual underperformance diagnostics versus same-period V2.0.",
        "- Added 30bps cost scenario for robustness.",
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
            .agg(worst_drawdown=("drawdown", "min"), avg_return=("portfolio_return", "mean"), days=("date", "count"), avg_cash_weight=("cash_weight", "mean"))
            .reset_index()
            .sort_values("worst_drawdown")
        )
        failure_lines = [
            "# HIRSSM V2.3 Regime Failure Cases",
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
    v21 = load_module("hirssm_v2_1_walk_forward", V21_PATH)
    v22 = load_module("hirssm_v2_2_walk_forward", V22_PATH)
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    nested_cfg = config.get("expert_state_nested_selection", {})
    shrink_cfg = config.get("expert_state_shrinkage", {})
    specs = v22.active_specs(wf, config)
    grid = shrink_cfg.get("cscv_pbo_grid", [])
    variants = [str(item["variant"]) for item in grid]

    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    monthly_ic = wf.compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, int(shrink_cfg.get("horizon_days", 21)))
    monthly_ic = v22.clean_monthly_ic(monthly_ic)

    selection_cost = float(nested_cfg.get("selection_cost_bps", 10.0))
    variant_histories: dict[str, pd.DataFrame] = {}
    variant_yearly_rows = []
    variant_summary_rows = []
    for item in grid:
        variant = str(item["variant"])
        run = v22.run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, item, variant, cost_bps=selection_cost)
        variant_histories[variant] = run["history"]
        diagnostics = yearly_nav_diagnostics(run["bt"]["nav"], variant)
        if not diagnostics.empty:
            variant_yearly_rows.append(diagnostics)
        if not run["summary"].empty:
            variant_summary_rows.append(run["summary"])
    variant_yearly = pd.concat(variant_yearly_rows, ignore_index=True, sort=False) if variant_yearly_rows else pd.DataFrame()
    variant_summaries = pd.concat(variant_summary_rows, ignore_index=True, sort=False) if variant_summary_rows else pd.DataFrame()

    first_year = int(variant_yearly["year"].min()) if not variant_yearly.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_year, month=1, day=1)
    cost_scenarios = [float(item) for item in nested_cfg.get("cost_bps_scenarios", [5, 10, 20, 30])]
    baseline = v21.run_same_period_baseline(model, panel, config, start_date, cost_scenarios)
    baseline_selection_nav = baseline["nav_by_cost"].get(selection_cost, pd.DataFrame())
    baseline_yearly = yearly_nav_diagnostics(baseline_selection_nav, "same_period_v2_0_baseline")

    selection, score_table = build_nested_selection(variant_yearly, baseline_yearly, config, variants)
    selected_history = combine_selected_history(selection, variant_histories)

    summaries = []
    nav_by_cost = {}
    for cost in cost_scenarios:
        run = run_nested_portfolio(model, v22, panel, config, selected_history, cost_bps=cost)
        nav_by_cost[cost] = run["bt"]["nav"]
        if not run["summary"].empty:
            summaries.append(run["summary"])
        suffix = f"{int(cost)}bps"
        model.write_csv(run["bt"]["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(run["bt"]["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(run["bt"]["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(run["bt"]["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        if cost == selection_cost:
            selected_run = run
    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()

    pbo_report, pbo_splits = nested_pbo_report(v21, variant_yearly, variant_summaries, summary_df)
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
    monthly_underperf, annual_underperf = underperformance_reports(nested_nav10, baseline_nav10)
    smoke = wf.smoke_test_targets(selected_run["targets"])
    data_audit = wf.data_contract_audit(root, config)

    model.write_csv(monthly_ic, output_dir / "monthly_expert_rank_ic.csv")
    model.write_csv(selected_history, output_dir / "selected_expert_state_multiplier_history.csv")
    model.write_csv(selected_history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(selected_run["targets"], output_dir / "walk_forward_target_weights.csv")
    model.write_csv(selection, output_dir / "nested_variant_selection.csv")
    model.write_csv(score_table, output_dir / "nested_selection_scores.csv")
    model.write_csv(summary_df, output_dir / "oos_performance.csv")
    model.write_csv(baseline["summary"], output_dir / "same_period_baseline_summary.csv")
    model.write_csv(comparison, output_dir / "same_period_baseline_comparison.csv")
    model.write_csv(variant_summaries, output_dir / "precomputed_variant_summary.csv")
    model.write_csv(variant_yearly, output_dir / "precomputed_variant_yearly_diagnostics.csv")
    model.write_csv(monthly_underperf, output_dir / "monthly_underperformance_vs_baseline.csv")
    model.write_csv(annual_underperf, output_dir / "annual_underperformance_vs_baseline.csv")
    model.write_csv(pbo_report, output_dir / "pbo_cscv_report.csv")
    model.write_csv(pbo_splits, output_dir / "pbo_cscv_splits.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(data_audit, output_dir / "data_contract_audit.csv")

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
        nested_nav10,
        panel["regimes"],
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
                "smoke_pass": bool(smoke["pass"].all()) if not smoke.empty else False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
