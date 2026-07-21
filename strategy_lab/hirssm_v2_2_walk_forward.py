#!/usr/bin/env python
"""HIRSSM V2.2 continuous expert shrinkage.

V2.2 replaces hard state gates with expert x state multipliers. Core experts
are anchored to the V2.0 prior and only shrunk when past state evidence is
weak or negative. Observation experts still require positive evidence before
their multiplier becomes non-zero.
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
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_2_walk_forward"


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
    cfg = config.get("expert_state_shrinkage", {})
    candidates = cfg.get("candidate_experts", wf.EXPERT_SPECS.keys())
    return {name: wf.EXPERT_SPECS[name] for name in candidates if name in wf.EXPERT_SPECS}


def merged_shrinkage_config(config: dict, override: dict | None = None) -> dict:
    src = dict(config.get("expert_state_shrinkage", {}))
    if override:
        src.update(override)
    return src


def clean_monthly_ic(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    out = monthly_ic.copy()
    if out.empty:
        return out
    out["state"] = out["state"].replace("", np.nan).fillna("unknown").astype(str)
    return out


def state_allowed(config: dict, expert: str, state: str) -> bool:
    whitelist = config.get("expert_state_shrinkage", {}).get("state_whitelist_by_expert", {})
    allowed = whitelist.get(expert)
    if not allowed:
        return True
    return state in set(str(item) for item in allowed)


def observation_requirement(config: dict, expert: str, shrink_cfg: dict) -> int:
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    if expert not in observation_experts and expert not in set(str(item) for item in config.get("disabled_experts_by_default", [])):
        return 1
    by_expert = shrink_cfg.get("consecutive_pass_windows_by_expert", {})
    return int(by_expert.get(expert, shrink_cfg.get("observation_consecutive_pass_windows", 2)))


def clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def evidence_score(stats: dict, shrink_cfg: dict) -> float:
    pieces = []
    if pd.notna(stats.get("rank_ic_mean")):
        pieces.append(clip(float(stats["rank_ic_mean"]) / float(shrink_cfg.get("rank_ic_scale", 0.08)), -1.0, 1.0))
    if pd.notna(stats.get("positive_ic_rate")):
        pieces.append(clip((float(stats["positive_ic_rate"]) - 0.5) / float(shrink_cfg.get("positive_rate_scale", 0.15)), -1.0, 1.0))
    if pd.notna(stats.get("rank_icir")):
        pieces.append(clip(float(stats["rank_icir"]) / float(shrink_cfg.get("icir_scale", 0.35)), -1.0, 1.0))
    if not pieces:
        return 0.0
    return float(np.mean(pieces))


def positive_observation_pass(stats: dict, shrink_cfg: dict) -> bool:
    return (
        stats["observations"] >= int(shrink_cfg.get("min_observations", 8))
        and pd.notna(stats["rank_ic_mean"])
        and stats["rank_ic_mean"] > float(shrink_cfg.get("observation_rank_ic_mean_min", 0.0))
        and pd.notna(stats["positive_ic_rate"])
        and stats["positive_ic_rate"] > float(shrink_cfg.get("observation_positive_ic_rate_min", 0.52))
        and pd.notna(stats["rank_icir"])
        and stats["rank_icir"] > float(shrink_cfg.get("observation_icir_min", 0.0))
    )


def negative_evidence_hit(stats: dict, shrink_cfg: dict) -> bool:
    return (
        stats["observations"] >= int(shrink_cfg.get("min_observations", 8))
        and pd.notna(stats["rank_ic_mean"])
        and stats["rank_ic_mean"] < float(shrink_cfg.get("negative_rank_ic_mean_max", -0.02))
        and pd.notna(stats["positive_ic_rate"])
        and stats["positive_ic_rate"] < float(shrink_cfg.get("negative_positive_ic_rate_max", 0.48))
        and pd.notna(stats["rank_icir"])
        and stats["rank_icir"] < float(shrink_cfg.get("negative_icir_max", -0.05))
    )


def multiplier_reason(
    structural: bool,
    requires_positive: bool,
    raw_pass: bool,
    negative_hit: bool,
    observations: int,
    multiplier: float,
) -> str:
    if not structural:
        return "multiplier 0: blocked by expert-state whitelist"
    if requires_positive and multiplier <= 0:
        return "multiplier 0: observation expert lacks positive state evidence"
    if requires_positive and raw_pass:
        return "observation expert enabled with positive state evidence"
    if observations <= 0:
        return "baseline multiplier: no state evidence"
    if negative_hit:
        return "shrunk by persistent negative state evidence"
    if multiplier > 1:
        return "tilted up by positive state evidence"
    if multiplier < 1:
        return "shrunk by weak state evidence"
    return "baseline multiplier"


def build_multiplier_history(
    monthly_ic: pd.DataFrame,
    config: dict,
    wf,
    specs: dict,
    override: dict | None = None,
    variant: str = "predeclared_default",
) -> pd.DataFrame:
    shrink_cfg = merged_shrinkage_config(config, override)
    train_years = int(shrink_cfg.get("train_years", 5))
    states = [str(item) for item in config.get("regime_model", {}).get("states", [])]
    candidates = [item for item in shrink_cfg.get("candidate_experts", specs.keys()) if item in specs]
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    default_disabled = set(str(item) for item in config.get("disabled_experts_by_default", []))
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
            requires_positive = expert in observation_experts or expert in default_disabled
            for state in states:
                train_ic = clean_ic[
                    clean_ic["expert"].eq(expert)
                    & clean_ic["state"].eq(state)
                    & (clean_ic["date"] >= train_start)
                    & (clean_ic["date"] < train_end)
                ]
                stats = wf.summarize_ic(train_ic)
                structural = state_allowed(config, expert, state)
                raw_pass = bool(structural and positive_observation_pass(stats, shrink_cfg))
                key = (expert, state)
                consecutive_pass[key] = consecutive_pass[key] + 1 if raw_pass else 0
                consecutive_required = observation_requirement(config, expert, shrink_cfg)
                evidence = evidence_score(stats, shrink_cfg)
                negative_hit = bool(structural and negative_evidence_hit(stats, shrink_cfg))
                if not structural:
                    multiplier = 0.0
                elif requires_positive:
                    if raw_pass and consecutive_pass[key] >= consecutive_required:
                        entry = float(shrink_cfg.get("observation_entry_multiplier", 0.35))
                        obs_max = float(shrink_cfg.get("observation_max_multiplier", 0.75))
                        multiplier = clip(entry + max(evidence, 0.0) * (obs_max - entry), entry, obs_max)
                    else:
                        multiplier = 0.0
                elif stats["observations"] < int(shrink_cfg.get("min_observations", 8)):
                    multiplier = float(shrink_cfg.get("base_multiplier", 1.0))
                else:
                    base = float(shrink_cfg.get("base_multiplier", 1.0))
                    strength = float(shrink_cfg.get("shrink_strength", 0.22))
                    multiplier = base + strength * evidence
                    multiplier = clip(multiplier, float(shrink_cfg.get("min_multiplier", 0.55)), float(shrink_cfg.get("max_multiplier", 1.15)))
                    if negative_hit:
                        multiplier = min(multiplier, float(shrink_cfg.get("bad_evidence_multiplier", 0.45)))
                rows.append(
                    {
                        "variant": variant,
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
                        "evidence_score": evidence,
                        "negative_evidence_hit": negative_hit,
                        "requires_positive_gate": requires_positive,
                        "raw_observation_pass": raw_pass,
                        "consecutive_pass_windows": consecutive_pass[key],
                        "consecutive_required": consecutive_required,
                        "structurally_allowed": structural,
                        "applies_to_portfolio": applies_to_portfolio,
                        "multiplier": float(multiplier),
                        "reason": multiplier_reason(structural, requires_positive, raw_pass, negative_hit, stats["observations"], float(multiplier)),
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


def build_multiplier_map(history: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    out: dict[tuple[int, str], dict[str, float]] = {}
    if history.empty:
        return out
    for (year, state), group in history.groupby(["test_year", "state"]):
        out[(int(year), str(state))] = {
            str(row["expert"]): float(row["multiplier"])
            for _, row in group.iterrows()
            if bool(row.get("applies_to_portfolio", True))
        }
    return out


def add_multiplier_columns(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or history.empty:
        return targets
    out = targets.copy()
    out["test_year"] = pd.to_datetime(out["signal_date"]).dt.year
    summaries = []
    applied = history[history["applies_to_portfolio"].astype(bool)]
    for (year, state), group in applied.groupby(["test_year", "state"]):
        compact = [
            f"{row['expert']}={float(row['multiplier']):.2f}"
            for _, row in group.sort_values("expert").iterrows()
        ]
        summaries.append({"test_year": int(year), "state": str(state), "expert_multipliers": ",".join(compact)})
    return out.merge(pd.DataFrame(summaries), on=["test_year", "state"], how="left")


def run_shrinkage_variant(
    model,
    wf,
    panel: dict,
    config: dict,
    specs: dict,
    monthly_ic: pd.DataFrame,
    override: dict | None,
    variant: str,
    cost_bps: float = 10.0,
) -> dict:
    history = build_multiplier_history(monthly_ic, config, wf, specs, override=override, variant=variant)
    multiplier_map = build_multiplier_map(history)
    first_test_year = int(history["test_year"].min()) if not history.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_test_year, month=1, day=1)
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=set(),
        expert_multipliers_by_year_state=multiplier_map,
    )
    targets = add_multiplier_columns(targets, history)
    if not targets.empty and "test_year" in targets.columns:
        targets = targets[targets["test_year"].isin(history["test_year"].unique())]
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
        "history": history,
        "targets": targets,
        "bt": bt,
        "summary": summary,
        "yearly": yearly.assign(variant=variant) if not yearly.empty else yearly,
        "start_date": start_date,
    }


def make_reports(
    output_dir: Path,
    selected: dict,
    baseline_summary: pd.DataFrame,
    variant_summaries: pd.DataFrame,
    multiplier_history: pd.DataFrame,
    pbo_report: pd.DataFrame,
    smoke: pd.DataFrame,
    data_audit: pd.DataFrame,
    nav10: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_summary = selected["summary"]
    selected_10 = selected_summary[selected_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    baseline_10 = baseline_summary[baseline_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    comparison = pd.concat([baseline_10, selected_10], ignore_index=True, sort=False) if not selected_10.empty else pd.DataFrame()
    lines = [
        "# HIRSSM V2.2 Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- Continuous expert x state multipliers replace hard gates.",
        "- Core experts are anchored to the V2.0 prior and shrunk only by weak or negative state evidence.",
        "- Observation experts require positive state evidence before their multiplier becomes non-zero.",
        "- Same-period V2.0 baseline and CSCV/PBO remain mandatory.",
        "",
        "## Selected Predeclared Variant",
        "",
        selected_summary.to_markdown(index=False) if not selected_summary.empty else "No selected summary.",
        "",
        "## Same-Period Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Predeclared Shrinkage Grid",
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

    multiplier_summary = pd.DataFrame()
    if not multiplier_history.empty:
        multiplier_summary = (
            multiplier_history[multiplier_history["applies_to_portfolio"].astype(bool)]
            .groupby(["expert", "state"], dropna=False)
            .agg(
                windows=("test_year", "count"),
                avg_multiplier=("multiplier", "mean"),
                min_multiplier=("multiplier", "min"),
                max_multiplier=("multiplier", "max"),
                negative_hit_rate=("negative_evidence_hit", "mean"),
                avg_train_rank_ic=("rank_ic_mean", "mean"),
                avg_test_rank_ic=("test_rank_ic_mean", "mean"),
            )
            .reset_index()
            .sort_values(["avg_multiplier", "avg_train_rank_ic"], ascending=False)
        )
    factor_lines = [
        "# HIRSSM V2.2 Expert Multiplier Report",
        "",
        "## Expert x State Multiplier Summary",
        "",
        multiplier_summary.to_markdown(index=False) if not multiplier_summary.empty else "No multiplier summary.",
        "",
        "## Latest Multiplier Decisions",
        "",
        multiplier_history.sort_values(["test_year", "state", "expert"]).tail(60).to_markdown(index=False) if not multiplier_history.empty else "No multiplier history.",
    ]
    (output_dir / "FACTOR_GATE_REPORT.md").write_text("\n".join(factor_lines), encoding="utf-8")

    changelog = [
        "# HIRSSM V2.2 Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Added expert x state continuous multipliers.",
        "- Added multiplier support to target construction.",
        "- Added shrinkage grid and CSCV/PBO diagnostics.",
        "- Kept V2.0 and V2.1 outputs intact.",
        "",
        "## Governance",
        "",
        "- Multipliers are predeclared and evaluated out of sample.",
        "- Promotion still requires same-period baseline comparison and PBO review.",
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
            "# HIRSSM V2.2 Regime Failure Cases",
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
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    specs = active_specs(wf, config)
    shrink_cfg = config.get("expert_state_shrinkage", {})

    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    horizon = int(shrink_cfg.get("horizon_days", 21))
    monthly_ic = wf.compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, horizon)
    monthly_ic = clean_monthly_ic(monthly_ic)

    default_variant = "predeclared_default"
    selected = run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, None, default_variant, cost_bps=10.0)
    start_date = selected["start_date"]

    cost_scenarios = [float(item) for item in config["validation"]["cost_bps_scenarios"]]
    selected_summaries = []
    nav_by_cost = {}
    for cost in cost_scenarios:
        cost_run = run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, None, default_variant, cost_bps=cost)
        selected_summaries.append(cost_run["summary"])
        nav_by_cost[cost] = cost_run["bt"]["nav"]
        suffix = f"{int(cost)}bps"
        model.write_csv(cost_run["bt"]["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(cost_run["bt"]["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(cost_run["bt"]["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(cost_run["bt"]["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        if cost == 10.0:
            selected = cost_run
    selected["summary"] = pd.concat(selected_summaries, ignore_index=True, sort=False) if selected_summaries else pd.DataFrame()

    baseline = v21.run_same_period_baseline(model, panel, config, start_date, cost_scenarios)

    variant_rows = []
    variant_yearly = []
    grid = shrink_cfg.get("cscv_pbo_grid", [])
    for item in grid:
        variant = str(item.get("variant", f"grid_{len(variant_rows) + 1}"))
        run = run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, item, variant, cost_bps=10.0)
        if not run["summary"].empty:
            variant_rows.append(run["summary"])
        if not run["yearly"].empty:
            variant_yearly.append(run["yearly"])
    variant_summaries = pd.concat(variant_rows, ignore_index=True, sort=False) if variant_rows else pd.DataFrame()
    variant_yearly_df = pd.concat(variant_yearly, ignore_index=True, sort=False) if variant_yearly else pd.DataFrame()
    pbo_report, pbo_splits = v21.run_cscv_pbo(
        variant_yearly_df,
        variant_summaries,
        selected_variant=grid[0]["variant"] if grid else default_variant,
    )

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
    history = selected["history"]

    model.write_csv(monthly_ic, output_dir / "monthly_expert_rank_ic.csv")
    model.write_csv(history, output_dir / "expert_state_multiplier_history.csv")
    model.write_csv(history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(selected["targets"], output_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected["summary"], output_dir / "oos_performance.csv")
    model.write_csv(baseline["summary"], output_dir / "same_period_baseline_summary.csv")
    model.write_csv(comparison, output_dir / "same_period_baseline_comparison.csv")
    model.write_csv(variant_summaries, output_dir / "shrinkage_variant_summary.csv")
    model.write_csv(variant_yearly_df, output_dir / "shrinkage_variant_yearly_returns.csv")
    model.write_csv(pbo_report, output_dir / "pbo_cscv_report.csv")
    model.write_csv(pbo_splits, output_dir / "pbo_cscv_splits.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(data_audit, output_dir / "data_contract_audit.csv")

    make_reports(
        output_dir,
        selected,
        baseline["summary"],
        variant_summaries,
        history,
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
                "multiplier_rows": int(history.shape[0]),
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
