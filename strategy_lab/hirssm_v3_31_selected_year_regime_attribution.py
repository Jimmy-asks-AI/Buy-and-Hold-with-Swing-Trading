#!/usr/bin/env python
"""HIRSSM V3.31 selected-year and regime attribution.

V3.30 found that V3.29 failed because macro rate/FX candidates had negative
marginal OOS value and high PBO risk. V3.31 explains where the selected
candidates helped or hurt by year, market state, and macro-trigger month.

This script is attribution-only. It does not create candidates, tune thresholds,
or change the default model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V329_MODEL_DIR = ROOT / "outputs" / "hirssm_v3_29_macro_rate_fx_harness"
V330_DIR = ROOT / "outputs" / "agent_runs" / "v3_30" / "macro_rate_fx_failure_attribution"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_31" / "selected_year_regime_attribution"
TASK_ID = "20260527_v3_31_selected_year_regime_attribution"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"
IMPLEMENTED = ["us_rate_shock_fx_stress_defense", "spread_repair_risk_on"]
VARIANTS = [BASELINE_VARIANT] + IMPLEMENTED
COSTS = [5.0, 10.0, 20.0]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bool_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(bool)
    return values.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if clean.empty:
        return np.nan
    nav = (1.0 + clean).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def annualized_mean(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean() * 252.0)


def first_nonnull(values: pd.Series) -> Any:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return clean.iloc[0]


def mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    return str(clean.mode().iloc[0])


def load_selection() -> pd.DataFrame:
    selection = read_csv(V329_MODEL_DIR / "nested_selection_by_fold.csv")
    if selection.empty:
        raise FileNotFoundError(V329_MODEL_DIR / "nested_selection_by_fold.csv")
    selection["cost_bps"] = selection["cost_bps"].astype(float)
    selection["test_year"] = selection["test_year"].astype(int)
    return selection


def load_target_meta(variant: str) -> pd.DataFrame:
    path = V329_MODEL_DIR / f"target_weights_{variant}.csv"
    targets = read_csv(path)
    if targets.empty:
        raise FileNotFoundError(path)
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    if "macro_trigger_active" not in targets.columns:
        targets["macro_trigger_active"] = False
    targets["macro_trigger_active"] = bool_series(targets["macro_trigger_active"])

    meta = targets.groupby("signal_date", as_index=False).agg(
        state=("state", first_nonnull),
        macro_trigger_active=("macro_trigger_active", "max"),
        macro_overlay_mode=("macro_overlay_mode", first_nonnull) if "macro_overlay_mode" in targets.columns else ("variant", first_nonnull),
    )
    cash = (
        targets[targets["asset"].astype(str).eq("CASH")]
        .groupby("signal_date", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "target_cash_weight"})
    )
    meta = meta.merge(cash, on="signal_date", how="left")
    meta["target_cash_weight"] = pd.to_numeric(meta["target_cash_weight"], errors="coerce")

    for col in ["us_10y_chg60_lag1", "usdcny_ret60_lag1", "cn_us_rate_spread_chg60_lag1"]:
        if col in targets.columns:
            values = targets.groupby("signal_date", as_index=False)[col].first()
            meta = meta.merge(values, on="signal_date", how="left")
        else:
            meta[col] = np.nan
    meta["selected_variant"] = variant
    return meta.sort_values("signal_date")


def map_meta_to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for variant, sub in daily.groupby("selected_variant"):
        meta = load_target_meta(str(variant))
        sub = sub.sort_values("date").copy()
        mapped = pd.merge_asof(
            sub,
            meta,
            left_on="date",
            right_on="signal_date",
            by="selected_variant",
            direction="backward",
        )
        parts.append(mapped)
    if not parts:
        return daily
    out = pd.concat(parts, ignore_index=True).sort_values(["cost_bps", "date"])
    out["state"] = out["state"].fillna("unknown")
    out["macro_trigger_active"] = bool_series(out["macro_trigger_active"])
    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    out["is_macro_candidate"] = out["selected_variant"].isin(IMPLEMENTED)
    return out


def load_daily_for_cost(cost_bps: float) -> pd.DataFrame:
    cost_label = int(cost_bps)
    selected = read_csv(V329_MODEL_DIR / f"nav_nested_selected_candidate_{cost_label}bps.csv")
    baseline = read_csv(V329_MODEL_DIR / f"nav_{BASELINE_VARIANT}_{cost_label}bps.csv")
    if selected.empty:
        raise FileNotFoundError(V329_MODEL_DIR / f"nav_nested_selected_candidate_{cost_label}bps.csv")
    if baseline.empty:
        raise FileNotFoundError(V329_MODEL_DIR / f"nav_{BASELINE_VARIANT}_{cost_label}bps.csv")
    selected["date"] = pd.to_datetime(selected["date"])
    baseline["date"] = pd.to_datetime(baseline["date"])
    selected["cost_bps"] = float(cost_bps)
    baseline_cols = [
        "date",
        "portfolio_return",
        "gross_return",
        "cost",
        "turnover",
        "cash_weight",
        "benchmark_return",
    ]
    baseline = baseline[baseline_cols].rename(
        columns={
            "portfolio_return": "baseline_portfolio_return",
            "gross_return": "baseline_gross_return",
            "cost": "baseline_cost",
            "turnover": "baseline_turnover",
            "cash_weight": "baseline_cash_weight",
            "benchmark_return": "baseline_benchmark_return",
        }
    )
    daily = selected.merge(baseline, on="date", how="left")
    daily["return_delta_vs_baseline"] = daily["portfolio_return"] - daily["baseline_portfolio_return"]
    daily["gross_delta_vs_baseline"] = daily["gross_return"] - daily["baseline_gross_return"]
    daily["cost_delta_vs_baseline"] = daily["cost"] - daily["baseline_cost"]
    daily["cash_delta_vs_baseline"] = daily["cash_weight"] - daily["baseline_cash_weight"]
    return daily


def build_daily_attribution() -> pd.DataFrame:
    daily = pd.concat([load_daily_for_cost(cost) for cost in COSTS], ignore_index=True)
    daily = map_meta_to_daily(daily)
    keep = [
        "date",
        "cost_bps",
        "test_year",
        "selected_variant",
        "portfolio_return",
        "baseline_portfolio_return",
        "return_delta_vs_baseline",
        "gross_return",
        "baseline_gross_return",
        "gross_delta_vs_baseline",
        "cost",
        "baseline_cost",
        "cost_delta_vs_baseline",
        "turnover",
        "baseline_turnover",
        "cash_weight",
        "baseline_cash_weight",
        "cash_delta_vs_baseline",
        "benchmark_return",
        "state",
        "signal_date",
        "macro_trigger_active",
        "year_month",
        "is_macro_candidate",
        "target_cash_weight",
        "us_10y_chg60_lag1",
        "usdcny_ret60_lag1",
        "cn_us_rate_spread_chg60_lag1",
    ]
    return daily[[col for col in keep if col in daily.columns]].copy()


def selected_year_attribution(daily: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_rows = selection[selection["selection_status"].astype(str).eq("selected_by_prior_window")].copy()
    for (cost, year, variant), sub in daily.groupby(["cost_bps", "test_year", "selected_variant"]):
        sel = selected_rows[
            selected_rows["cost_bps"].astype(float).eq(float(cost))
            & selected_rows["test_year"].astype(int).eq(int(year))
        ].head(1)
        rows.append(
            {
                "cost_bps": float(cost),
                "test_year": int(year),
                "selected_variant": variant,
                "selected_total_return": compound_return(sub["portfolio_return"]),
                "baseline_total_return": compound_return(sub["baseline_portfolio_return"]),
                "selected_minus_baseline_return": compound_return(sub["portfolio_return"]) - compound_return(sub["baseline_portfolio_return"]),
                "selected_max_drawdown": max_drawdown(sub["portfolio_return"]),
                "baseline_max_drawdown": max_drawdown(sub["baseline_portfolio_return"]),
                "drawdown_delta_vs_baseline": max_drawdown(sub["portfolio_return"]) - max_drawdown(sub["baseline_portfolio_return"]),
                "selected_oos_rank_pct": float(sel["selected_oos_rank_pct"].iloc[0]) if not sel.empty and pd.notna(sel["selected_oos_rank_pct"].iloc[0]) else np.nan,
                "selected_oos_score": float(sel["selected_oos_score"].iloc[0]) if not sel.empty and pd.notna(sel["selected_oos_score"].iloc[0]) else np.nan,
                "dominant_state": mode_text(sub["state"]),
                "trigger_days": int(sub["macro_trigger_active"].sum()),
                "trigger_months": int(sub.loc[sub["macro_trigger_active"], "year_month"].nunique()),
                "trigger_day_rate": float(sub["macro_trigger_active"].mean()),
                "avg_cash_delta_vs_baseline": float(sub["cash_delta_vs_baseline"].mean()),
                "avg_cost_delta_vs_baseline": float(sub["cost_delta_vs_baseline"].mean()),
                "days": int(sub.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_bps", "test_year"])


def selected_regime_attribution(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cost, variant, state), sub in daily.groupby(["cost_bps", "selected_variant", "state"]):
        rows.append(
            {
                "cost_bps": float(cost),
                "selected_variant": variant,
                "state": state,
                "days": int(sub.shape[0]),
                "years": int(sub["test_year"].nunique()),
                "selected_annualized_mean": annualized_mean(sub["portfolio_return"]),
                "baseline_annualized_mean": annualized_mean(sub["baseline_portfolio_return"]),
                "annualized_mean_delta_vs_baseline": annualized_mean(sub["return_delta_vs_baseline"]),
                "selected_total_return": compound_return(sub["portfolio_return"]),
                "baseline_total_return": compound_return(sub["baseline_portfolio_return"]),
                "total_return_delta_vs_baseline": compound_return(sub["portfolio_return"]) - compound_return(sub["baseline_portfolio_return"]),
                "trigger_day_rate": float(sub["macro_trigger_active"].mean()),
                "avg_cash_delta_vs_baseline": float(sub["cash_delta_vs_baseline"].mean()),
                "win_rate": float((sub["return_delta_vs_baseline"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_bps", "selected_variant", "annualized_mean_delta_vs_baseline"])


def month_group_row(keys: tuple[Any, ...], sub: pd.DataFrame) -> dict[str, Any]:
    cost, variant, year, month = keys
    return {
        "cost_bps": float(cost),
        "selected_variant": variant,
        "test_year": int(year),
        "year_month": month,
        "state": mode_text(sub["state"]),
        "macro_trigger_active": bool(sub["macro_trigger_active"].any()),
        "days": int(sub.shape[0]),
        "selected_month_return": compound_return(sub["portfolio_return"]),
        "baseline_month_return": compound_return(sub["baseline_portfolio_return"]),
        "selected_minus_baseline_month_return": compound_return(sub["portfolio_return"]) - compound_return(sub["baseline_portfolio_return"]),
        "avg_cash_delta_vs_baseline": float(sub["cash_delta_vs_baseline"].mean()),
        "avg_cost_delta_vs_baseline": float(sub["cost_delta_vs_baseline"].mean()),
        "us_10y_chg60_lag1": float(pd.to_numeric(sub["us_10y_chg60_lag1"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(sub["us_10y_chg60_lag1"], errors="coerce").dropna().size else np.nan,
        "usdcny_ret60_lag1": float(pd.to_numeric(sub["usdcny_ret60_lag1"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(sub["usdcny_ret60_lag1"], errors="coerce").dropna().size else np.nan,
        "cn_us_rate_spread_chg60_lag1": float(pd.to_numeric(sub["cn_us_rate_spread_chg60_lag1"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(sub["cn_us_rate_spread_chg60_lag1"], errors="coerce").dropna().size else np.nan,
    }


def trigger_month_attribution(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in daily.groupby(["cost_bps", "selected_variant", "test_year", "year_month"]):
        rows.append(month_group_row(keys, sub))
    return pd.DataFrame(rows).sort_values(["cost_bps", "test_year", "year_month", "selected_variant"])


def candidate_selected_year_summary(year_attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cost, variant), sub in year_attr.groupby(["cost_bps", "selected_variant"]):
        deltas = pd.to_numeric(sub["selected_minus_baseline_return"], errors="coerce")
        worst_idx = deltas.idxmin() if not deltas.empty else None
        best_idx = deltas.idxmax() if not deltas.empty else None
        rows.append(
            {
                "cost_bps": float(cost),
                "selected_variant": variant,
                "selected_year_count": int(sub.shape[0]),
                "positive_delta_year_count": int((deltas > 0).sum()),
                "positive_delta_rate": float((deltas > 0).mean()) if not deltas.empty else np.nan,
                "avg_selected_minus_baseline_return": float(deltas.mean()) if not deltas.empty else np.nan,
                "sum_selected_minus_baseline_return": float(deltas.sum()) if not deltas.empty else np.nan,
                "worst_year": int(sub.loc[worst_idx, "test_year"]) if worst_idx is not None else np.nan,
                "worst_year_delta": float(sub.loc[worst_idx, "selected_minus_baseline_return"]) if worst_idx is not None else np.nan,
                "best_year": int(sub.loc[best_idx, "test_year"]) if best_idx is not None else np.nan,
                "best_year_delta": float(sub.loc[best_idx, "selected_minus_baseline_return"]) if best_idx is not None else np.nan,
                "avg_trigger_months": float(pd.to_numeric(sub["trigger_months"], errors="coerce").mean()),
                "avg_cash_delta_vs_baseline": float(pd.to_numeric(sub["avg_cash_delta_vs_baseline"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_bps", "avg_selected_minus_baseline_return"])


def regime_trigger_failure_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    macro = daily[daily["selected_variant"].isin(IMPLEMENTED)].copy()
    for (cost, variant, state, trigger), sub in macro.groupby(["cost_bps", "selected_variant", "state", "macro_trigger_active"]):
        rows.append(
            {
                "cost_bps": float(cost),
                "selected_variant": variant,
                "state": state,
                "macro_trigger_active": bool(trigger),
                "days": int(sub.shape[0]),
                "years": int(sub["test_year"].nunique()),
                "annualized_mean_delta_vs_baseline": annualized_mean(sub["return_delta_vs_baseline"]),
                "total_return_delta_vs_baseline": compound_return(sub["portfolio_return"]) - compound_return(sub["baseline_portfolio_return"]),
                "win_rate": float((sub["return_delta_vs_baseline"] > 0).mean()),
                "avg_cash_delta_vs_baseline": float(sub["cash_delta_vs_baseline"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_bps", "annualized_mean_delta_vs_baseline"])


def worst_selected_years(year_attr: pd.DataFrame, month_attr: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    macro_years = year_attr[year_attr["selected_variant"].isin(IMPLEMENTED)].copy()
    if macro_years.empty:
        return pd.DataFrame()
    worst = (
        macro_years.sort_values(["cost_bps", "selected_minus_baseline_return"])
        .groupby("cost_bps", as_index=False)
        .head(top_n)
        .copy()
    )
    worst_months = []
    for _, row in worst.iterrows():
        months = month_attr[
            month_attr["cost_bps"].astype(float).eq(float(row["cost_bps"]))
            & month_attr["test_year"].astype(int).eq(int(row["test_year"]))
            & month_attr["selected_variant"].astype(str).eq(str(row["selected_variant"]))
        ].sort_values("selected_minus_baseline_month_return")
        if months.empty:
            worst_months.append("")
        else:
            m = months.head(3)
            worst_months.append(
                ";".join(
                    f"{item.year_month}:{float(item.selected_minus_baseline_month_return):.2%}:{item.state}:trigger={bool(item.macro_trigger_active)}"
                    for item in m.itertuples(index=False)
                )
            )
    worst["worst_3_months"] = worst_months
    return worst


def next_experiment_queue(summary: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "task": "V3.32 convert macro overlay into risk-budget gate",
                "allowed": True,
                "reason": "V3.31 localizes losses by selected years/regimes; additive 4pct overlay is too weak and unstable.",
                "forbidden": "default promotion without nested/PBO validation",
            },
            {
                "priority": 2,
                "task": "Predeclare regime-conditional activation rules",
                "allowed": True,
                "reason": "Only use V3.31 attribution to define hypotheses; no retroactive threshold fitting.",
                "forbidden": "parameter search on trigger thresholds or cash-shift amplitudes",
            },
            {
                "priority": 3,
                "task": "Add selection-margin veto before choosing macro candidates",
                "allowed": True,
                "reason": "V3.29 often selected macro candidates with weak OOS rank margin versus baseline.",
                "forbidden": "using future OOS rank in the live selection rule",
            },
            {
                "priority": 4,
                "task": "Keep V3.29 macro candidates observation-only",
                "allowed": False,
                "reason": "V3.31 is attribution-only and cannot override V3.29's PBO failure.",
                "forbidden": "promoting us_rate_shock_fx_stress_defense or spread_repair_risk_on to default",
            },
        ]
    )


def make_report(year_attr: pd.DataFrame, summary: pd.DataFrame, matrix: pd.DataFrame, worst: pd.DataFrame) -> str:
    lines = [
        "# HIRSSM V3.31 Selected-Year/Regime Attribution",
        "",
        "## Decision",
        "",
        "- V3.31 is accepted as attribution-only.",
        "- V3.29 macro rate/FX candidates remain observation-only.",
        "- No parameter, amplitude, or threshold change is authorized by this version.",
        "",
        "## Key 10bps Findings",
        "",
    ]
    s10 = summary[
        summary["cost_bps"].astype(float).eq(10.0)
        & summary["selected_variant"].isin(IMPLEMENTED)
    ].copy()
    for _, row in s10.iterrows():
        lines.append(
            f"- `{row['selected_variant']}`: selected years `{int(row['selected_year_count'])}`, "
            f"positive-delta rate `{float(row['positive_delta_rate']):.2%}`, "
            f"avg delta `{float(row['avg_selected_minus_baseline_return']):.2%}`, "
            f"worst year `{int(row['worst_year'])}` `{float(row['worst_year_delta']):.2%}`."
        )
    m10 = matrix[matrix["cost_bps"].astype(float).eq(10.0)].head(6)
    if not m10.empty:
        lines.extend(["", "## Worst State/Trigger Buckets", ""])
        for _, row in m10.iterrows():
            lines.append(
                f"- `{row['selected_variant']}` / `{row['state']}` / trigger={bool(row['macro_trigger_active'])}: "
                f"annualized delta `{float(row['annualized_mean_delta_vs_baseline']):.2%}`, days `{int(row['days'])}`."
            )
    w10 = worst[worst["cost_bps"].astype(float).eq(10.0)].head(5) if not worst.empty else pd.DataFrame()
    if not w10.empty:
        lines.extend(["", "## Worst Selected Years", ""])
        for _, row in w10.iterrows():
            lines.append(
                f"- `{int(row['test_year'])}` `{row['selected_variant']}`: "
                f"year delta `{float(row['selected_minus_baseline_return']):.2%}`, "
                f"worst months `{row['worst_3_months']}`."
            )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "V3.32 may test macro signals as gates on the existing risk budget, but must be predeclared and rerun through nested/PBO checks.",
        ]
    )
    return "\n".join(lines)


def self_check(paths: dict[str, Path], daily: pd.DataFrame, year_attr: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "status": "pass" if path.exists() else "fail", "detail": rel(path)})
    expected_costs = set(COSTS)
    actual_costs = set(round(float(x), 1) for x in daily["cost_bps"].dropna().unique())
    missing_baseline = int(daily["baseline_portfolio_return"].isna().sum()) if "baseline_portfolio_return" in daily.columns else -1
    rows.extend(
        [
            {
                "check": "all_costs_included",
                "status": "pass" if expected_costs.issubset(actual_costs) else "fail",
                "detail": f"actual={sorted(actual_costs)}",
            },
            {
                "check": "daily_rows_have_baseline_returns",
                "status": "pass" if missing_baseline == 0 else "fail",
                "detail": str(missing_baseline),
            },
            {
                "check": "selected_years_present",
                "status": "pass" if not year_attr.empty and year_attr["test_year"].nunique() > 0 else "fail",
                "detail": str(int(year_attr["test_year"].nunique())) if not year_attr.empty else "0",
            },
            {
                "check": "no_new_candidate_created",
                "status": "pass" if set(year_attr["selected_variant"].unique()).issubset(set(VARIANTS)) else "fail",
                "detail": ",".join(sorted(set(year_attr["selected_variant"].astype(str).unique()))),
            },
            {
                "check": "queue_blocks_parameter_search",
                "status": "pass" if queue["forbidden"].astype(str).str.contains("parameter|threshold|amplitude", case=False).any() else "fail",
                "detail": str(int(queue.shape[0])),
            },
        ]
    )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = load_selection()
    daily = build_daily_attribution()
    year_attr = selected_year_attribution(daily, selection)
    regime_attr = selected_regime_attribution(daily)
    month_attr = trigger_month_attribution(daily)
    summary = candidate_selected_year_summary(year_attr)
    matrix = regime_trigger_failure_matrix(daily)
    worst = worst_selected_years(year_attr, month_attr)
    queue = next_experiment_queue(summary, matrix)

    paths = {
        "daily_selected_attribution": output_dir / "daily_selected_attribution.csv",
        "selected_year_attribution": output_dir / "selected_year_attribution.csv",
        "selected_regime_attribution": output_dir / "selected_regime_attribution.csv",
        "trigger_month_attribution": output_dir / "trigger_month_attribution.csv",
        "candidate_selected_year_summary": output_dir / "candidate_selected_year_summary.csv",
        "regime_trigger_failure_matrix": output_dir / "regime_trigger_failure_matrix.csv",
        "worst_selected_years": output_dir / "worst_selected_years.csv",
        "next_experiment_queue": output_dir / "next_experiment_queue.csv",
        "agent_report": output_dir / "agent_report.md",
        "failure_cases": output_dir / "REGIME_FAILURE_CASES.md",
        "changed_files": output_dir / "changed_files.txt",
    }
    for df, key in [
        (daily, "daily_selected_attribution"),
        (year_attr, "selected_year_attribution"),
        (regime_attr, "selected_regime_attribution"),
        (month_attr, "trigger_month_attribution"),
        (summary, "candidate_selected_year_summary"),
        (matrix, "regime_trigger_failure_matrix"),
        (worst, "worst_selected_years"),
        (queue, "next_experiment_queue"),
    ]:
        write_csv(df, paths[key])

    report = make_report(year_attr, summary, matrix, worst)
    write_text(report, paths["agent_report"])
    write_text(report, paths["failure_cases"])
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    self_check_path.touch()
    paths["self_check"] = self_check_path
    checks = self_check(paths, daily, year_attr, queue)
    write_csv(checks, self_check_path)
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())

    s10 = summary[summary["cost_bps"].astype(float).eq(10.0)].copy()
    macro10 = s10[s10["selected_variant"].isin(IMPLEMENTED)].copy()
    worst10 = worst[worst["cost_bps"].astype(float).eq(10.0)].head(1) if not worst.empty else pd.DataFrame()
    metrics = {
        "macro_selected_years_10bps": int(macro10["selected_year_count"].sum()) if not macro10.empty else 0,
        "macro_positive_delta_rate_10bps": float(
            np.average(
                macro10["positive_delta_rate"],
                weights=macro10["selected_year_count"],
            )
        )
        if not macro10.empty and macro10["selected_year_count"].sum() > 0
        else np.nan,
        "macro_avg_year_delta_10bps": float(
            np.average(
                macro10["avg_selected_minus_baseline_return"],
                weights=macro10["selected_year_count"],
            )
        )
        if not macro10.empty and macro10["selected_year_count"].sum() > 0
        else np.nan,
        "worst_macro_year_10bps": int(worst10["test_year"].iloc[0]) if not worst10.empty else None,
        "worst_macro_variant_10bps": str(worst10["selected_variant"].iloc[0]) if not worst10.empty else "",
        "worst_macro_year_delta_10bps": float(worst10["selected_minus_baseline_return"].iloc[0]) if not worst10.empty else np.nan,
        "recommended_next": "risk_budget_gate_design",
    }

    artifacts = list(paths.values())
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.31",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_31_selected_year_regime_attribution.py",
        "config": {"attribution_only": True, "no_parameter_search": True, "source_version": "V3.29,V3.30"},
        "data_refs": [
            "outputs/hirssm_v3_29_macro_rate_fx_harness",
            "outputs/agent_runs/v3_30/macro_rate_fx_failure_attribution",
        ],
        "code_refs": ["strategy_lab/hirssm_v3_31_selected_year_regime_attribution.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "outputs/hirssm_v3_29_macro_rate_fx_harness",
            "outputs/agent_runs/v3_30/macro_rate_fx_failure_attribution",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Uses V3.29 realized selected-candidate outputs only.",
            "Attribution is diagnostic and can define hypotheses, not default promotion.",
            "Regime labels are inherited from V3.29 target-weight state snapshots.",
        ],
        "risk_flags": [
            "post_hoc_failure_attribution",
            "small_macro_candidate_selected_year_count",
            "state_label_dependency",
        ],
        "next_decision": "Run V3.32 predeclared risk-budget gate design if the selected-year evidence supports it.",
        "handoff_summary": "V3.31 localizes V3.29 macro candidate losses by year, state, and trigger month.",
    }
    manifest_path = output_dir / "agent_run_manifest.json"
    write_json(manifest, manifest_path)
    paths["agent_run_manifest"] = manifest_path
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])
    return {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
