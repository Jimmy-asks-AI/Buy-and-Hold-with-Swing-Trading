#!/usr/bin/env python
"""Walk-forward expert gating for HIRSSM V2.0.

This script upgrades HIRSSM validation from full-sample expert pruning to
past-only expert gates. Each test year uses only the previous five years of
monthly RankIC evidence to decide which experts can affect target weights.
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
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_walk_forward"
TRADING_DAYS = 252


EXPERT_SPECS = {
    "trend_continuation": {
        "score_col": "trend_state_score",
        "asset_type": "",
        "applies_to_portfolio": False,
        "economic_logic": "Persistent price trend and relative strength should continue in broad risk-on regimes.",
        "failure_scenario": "Trend crashes after crowding, sharp policy reversal, or style leadership rotation.",
    },
    "style_trend_continuation": {
        "score_col": "trend_state_score",
        "asset_type": "style",
        "applies_to_portfolio": True,
        "economic_logic": "Size-style indices can exhibit medium-term leadership when breadth confirms risk appetite.",
        "failure_scenario": "Style trend is mature or valuation/risk compression overwhelms momentum.",
    },
    "industry_trend_continuation": {
        "score_col": "trend_state_score",
        "asset_type": "industry",
        "applies_to_portfolio": True,
        "economic_logic": "Industry leadership often persists when earnings or policy catalysts diffuse slowly.",
        "failure_scenario": "Crowded hot industries mean-revert or parent-child industry overlap distorts signals.",
    },
    "valuation_repair": {
        "score_col": "valuation_repair_score",
        "asset_type": "",
        "applies_to_portfolio": True,
        "economic_logic": "Low relative valuation can repair after trend stabilization or volatility compression.",
        "failure_scenario": "Cheap assets remain value traps when fundamentals deteriorate or liquidity exits.",
    },
    "risk_compression": {
        "score_col": "risk_compression_score",
        "asset_type": "",
        "applies_to_portfolio": True,
        "economic_logic": "Lower downside volatility and drawdown pressure can signal investable stabilization.",
        "failure_scenario": "Volatility compression precedes another break lower or misses high-beta recoveries.",
    },
    "range_reversal": {
        "score_col": "range_reversal_score",
        "asset_type": "",
        "applies_to_portfolio": True,
        "economic_logic": "Short-term overextension can reverse in range-bound or crash-rebound states.",
        "failure_scenario": "Falling knives and strong trend regimes make reversal signals structurally weak.",
    },
    "defensive": {
        "score_col": "defensive_score",
        "asset_type": "",
        "applies_to_portfolio": True,
        "economic_logic": "Low-risk dividend/large-cap sleeves can protect capital in risk-off regimes.",
        "failure_scenario": "Defensive indices underperform when rates rise or market beta rapidly recovers.",
    },
    "liquidity_overlay": {
        "score_col": "liquidity_score",
        "asset_type": "",
        "applies_to_portfolio": False,
        "economic_logic": "Volume and amount confirmation can separate real leadership from stale price moves.",
        "failure_scenario": "Liquidity spikes are exhaustion/crowding rather than informed accumulation.",
    },
    "style_liquidity_overlay": {
        "score_col": "liquidity_score",
        "asset_type": "style",
        "applies_to_portfolio": True,
        "economic_logic": "Style-level turnover confirmation can validate broad allocation shifts.",
        "failure_scenario": "Index ETF flow creates noisy turnover unrelated to next-period returns.",
    },
    "industry_liquidity_overlay": {
        "score_col": "liquidity_score",
        "asset_type": "industry",
        "applies_to_portfolio": True,
        "economic_logic": "Industry amount confirmation can validate catalyst-driven sector rotation.",
        "failure_scenario": "Hot-sector crowding and late-cycle volume spikes reverse quickly.",
    },
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_model():
    spec = importlib.util.spec_from_file_location("hirssm_v2_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_panel(model, root: Path, config: dict, start_date: str | None, end_date: str | None) -> dict:
    style = model.load_style_daily(root, config)
    industry = model.load_industry_daily(root, config)
    if start_date:
        start = pd.to_datetime(start_date)
        style = style[style["date"] >= start]
        industry = industry[industry["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        style = style[style["date"] <= end]
        industry = industry[industry["date"] <= end]

    broad_code = model.normalize_code(config["asset_universe"]["style"].get("broad_market", "000985"))
    style_raw = style.sort_values(["asset", "date"]).copy()
    style_raw["ret_1d"] = style_raw.groupby("asset")["close"].pct_change()
    market_returns = style_raw[style_raw["asset"].eq(broad_code)].set_index("date")["ret_1d"]

    style_features = model.add_features(style, market_returns=market_returns)
    industry_features = model.add_features(industry, market_returns=market_returns)
    valuation = model.load_style_valuation(root, config, style_features["asset"].drop_duplicates())
    style_features = model.add_valuation_scores(style_features, valuation)
    industry_features["valuation_score"] = 0.0

    style_scores = model.score_assets(style_features, is_style=True)
    industry_scores = model.score_assets(industry_features, is_style=False)
    scored = pd.concat([style_scores, industry_scores], ignore_index=True, sort=False)
    regimes = model.assign_regime(style_scores, industry_scores, config)

    min_history_days = int(config["feature_pipeline"].get("min_history_days", 504))
    eligible = scored.copy()
    eligible["history_count"] = eligible.groupby("asset").cumcount() + 1
    eligible = eligible[eligible["history_count"] >= min_history_days]
    returns = scored[["date", "asset", "ret_1d"]].dropna().copy()
    return {
        "scored": scored,
        "eligible": eligible,
        "regimes": regimes,
        "returns": returns,
        "broad_code": broad_code,
    }


def prepare_expert_panel(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.sort_values(["asset", "date"]).copy()
    if "trend_state_score" not in out.columns:
        out["trend_state_score"] = out[["trend_expert_score", "relative_strength_score"]].mean(axis=1)
    return out


def spearman_ic(group: pd.DataFrame, score_col: str, return_col: str) -> float:
    x = pd.to_numeric(group[score_col], errors="coerce")
    y = pd.to_numeric(group[return_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return np.nan
    xr = x[valid].rank()
    yr = y[valid].rank()
    if xr.nunique(dropna=True) <= 1 or yr.nunique(dropna=True) <= 1:
        return np.nan
    return float(xr.corr(yr))


def compute_monthly_expert_ic(model, scored: pd.DataFrame, regimes: pd.DataFrame, specs: dict, horizon: int) -> pd.DataFrame:
    panel = prepare_expert_panel(scored)
    panel[f"fwd_ret_{horizon}"] = panel.groupby("asset")["close"].shift(-horizon) / panel["close"] - 1
    signal_dates = set(model.month_end_dates(panel["date"]))
    panel = panel[panel["date"].isin(signal_dates)].merge(regimes[["date", "state"]], on="date", how="left")
    rows = []
    return_col = f"fwd_ret_{horizon}"
    for expert, spec in specs.items():
        score_col = spec["score_col"]
        if score_col not in panel.columns:
            continue
        scoped = panel.copy()
        if spec["asset_type"]:
            scoped = scoped[scoped["asset_type"].astype(str).eq(spec["asset_type"])]
        for signal_date, day in scoped.groupby("date"):
            ic = spearman_ic(day, score_col, return_col)
            if pd.isna(ic):
                continue
            rows.append(
                {
                    "date": pd.to_datetime(signal_date),
                    "year": int(pd.to_datetime(signal_date).year),
                    "expert": expert,
                    "asset_type_scope": spec["asset_type"] or "all",
                    "score_col": score_col,
                    "state": str(day["state"].dropna().iloc[0]) if day["state"].notna().any() else "",
                    "rank_ic": ic,
                    "asset_count": int(day[["asset", score_col, return_col]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_ic(train_ic: pd.DataFrame) -> dict:
    if train_ic.empty:
        return {"observations": 0, "rank_ic_mean": np.nan, "rank_ic_std": np.nan, "rank_icir": np.nan, "positive_ic_rate": np.nan}
    mean = float(train_ic["rank_ic"].mean())
    std = float(train_ic["rank_ic"].std(ddof=1))
    return {
        "observations": int(train_ic.shape[0]),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_icir": mean / std if std and not pd.isna(std) else np.nan,
        "positive_ic_rate": float((train_ic["rank_ic"] > 0).mean()),
    }


def format_reason(stats: dict, thresholds: dict, raw_pass: bool, observation_blocked: bool) -> str:
    if raw_pass and not observation_blocked:
        return "passes past-window RankIC, positive-IC-rate, and ICIR gates"
    reasons = []
    if stats["observations"] < thresholds["min_observations"]:
        reasons.append(f"observations {stats['observations']} < {thresholds['min_observations']}")
    if pd.isna(stats["rank_ic_mean"]) or stats["rank_ic_mean"] <= thresholds["rank_ic_mean_min"]:
        reasons.append(f"rank_ic_mean {stats['rank_ic_mean']:.4f} <= {thresholds['rank_ic_mean_min']:.4f}" if pd.notna(stats["rank_ic_mean"]) else "rank_ic_mean missing")
    if pd.isna(stats["positive_ic_rate"]) or stats["positive_ic_rate"] <= thresholds["positive_ic_rate_min"]:
        reasons.append(f"positive_ic_rate {stats['positive_ic_rate']:.2%} <= {thresholds['positive_ic_rate_min']:.2%}" if pd.notna(stats["positive_ic_rate"]) else "positive_ic_rate missing")
    if pd.isna(stats["rank_icir"]) or stats["rank_icir"] <= thresholds["icir_min"]:
        reasons.append(f"rank_icir {stats['rank_icir']:.4f} <= {thresholds['icir_min']:.4f}" if pd.notna(stats["rank_icir"]) else "rank_icir missing")
    if observation_blocked:
        reasons.append("observation expert requires consecutive passing windows before portfolio use")
    return "; ".join(reasons)


def build_gate_history(monthly_ic: pd.DataFrame, config: dict, specs: dict) -> pd.DataFrame:
    gate_cfg = config.get("expert_gates", {})
    thresholds = {
        "min_observations": int(gate_cfg.get("min_observations", 24)),
        "rank_ic_mean_min": float(gate_cfg.get("rank_ic_mean_min", 0.0)),
        "positive_ic_rate_min": float(gate_cfg.get("positive_ic_rate_min", 0.52)),
        "icir_min": float(gate_cfg.get("icir_min", 0.05)),
    }
    train_years = int(gate_cfg.get("train_years", 5))
    observation_experts = set(str(item) for item in gate_cfg.get("observation_experts", []))
    consecutive_required = int(gate_cfg.get("observation_consecutive_pass_windows", 2))
    candidates = [item for item in gate_cfg.get("candidate_experts", specs.keys()) if item in specs]
    if monthly_ic.empty:
        return pd.DataFrame()
    years = sorted(int(y) for y in monthly_ic["year"].dropna().unique())
    first_year = min(years) + train_years
    consecutive_pass = {expert: 0 for expert in candidates}
    rows = []
    for test_year in [year for year in years if year >= first_year]:
        train_start = pd.Timestamp(year=test_year - train_years, month=1, day=1)
        train_end = pd.Timestamp(year=test_year, month=1, day=1)
        for expert in candidates:
            spec = specs[expert]
            train_ic = monthly_ic[
                monthly_ic["expert"].eq(expert)
                & (monthly_ic["date"] >= train_start)
                & (monthly_ic["date"] < train_end)
            ]
            stats = summarize_ic(train_ic)
            raw_pass = (
                stats["observations"] >= thresholds["min_observations"]
                and pd.notna(stats["rank_ic_mean"])
                and stats["rank_ic_mean"] > thresholds["rank_ic_mean_min"]
                and pd.notna(stats["positive_ic_rate"])
                and stats["positive_ic_rate"] > thresholds["positive_ic_rate_min"]
                and pd.notna(stats["rank_icir"])
                and stats["rank_icir"] > thresholds["icir_min"]
            )
            consecutive_pass[expert] = consecutive_pass[expert] + 1 if raw_pass else 0
            observation_blocked = expert in observation_experts and consecutive_pass[expert] < consecutive_required
            portfolio_enabled = bool(raw_pass and not observation_blocked)
            applies_to_portfolio = bool(spec.get("applies_to_portfolio", True))
            rows.append(
                {
                    "test_year": test_year,
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
                    "raw_gate_pass": raw_pass,
                    "consecutive_pass_windows": consecutive_pass[expert],
                    "portfolio_enabled": portfolio_enabled,
                    "applies_to_portfolio": applies_to_portfolio,
                    "disabled_for_year": bool(applies_to_portfolio and not portfolio_enabled),
                    "reason": format_reason(stats, thresholds, raw_pass, observation_blocked),
                    "economic_logic": spec["economic_logic"],
                    "failure_scenario": spec["failure_scenario"],
                }
            )
    history = pd.DataFrame(rows)
    if history.empty:
        return history
    test_stats = (
        monthly_ic.groupby(["year", "expert"], dropna=False)
        .agg(test_rank_ic_mean=("rank_ic", "mean"), test_positive_ic_rate=("rank_ic", lambda s: float((s > 0).mean())), test_observations=("rank_ic", "count"))
        .reset_index()
        .rename(columns={"year": "test_year"})
    )
    return history.merge(test_stats, on=["test_year", "expert"], how="left")


def build_disabled_by_year(gate_history: pd.DataFrame) -> dict[int, set[str]]:
    disabled_by_year: dict[int, set[str]] = {}
    if gate_history.empty:
        return disabled_by_year
    applied = gate_history[gate_history["applies_to_portfolio"].astype(bool)]
    for year, group in applied.groupby("test_year"):
        disabled_by_year[int(year)] = set(group.loc[group["disabled_for_year"].astype(bool), "expert"].astype(str))
    return disabled_by_year


def add_gate_columns(targets: pd.DataFrame, gate_history: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or gate_history.empty:
        return targets
    out = targets.copy()
    out["test_year"] = pd.to_datetime(out["signal_date"]).dt.year
    gate_summary = []
    applied = gate_history[gate_history["applies_to_portfolio"].astype(bool)]
    for year, group in applied.groupby("test_year"):
        enabled = sorted(group.loc[group["portfolio_enabled"].astype(bool), "expert"].astype(str).tolist())
        disabled = sorted(group.loc[group["disabled_for_year"].astype(bool), "expert"].astype(str).tolist())
        gate_summary.append({"test_year": int(year), "enabled_experts": ",".join(enabled), "disabled_experts": ",".join(disabled)})
    return out.merge(pd.DataFrame(gate_summary), on="test_year", how="left")


def data_contract_audit(root: Path, config: dict) -> pd.DataFrame:
    data_contract = config.get("data_contract", {})
    rows = []
    macro_paths = data_contract.get("macro_data_paths", {})
    for name, rel_path in macro_paths.items():
        path = root / rel_path
        exists = path.exists()
        has_available_date = False
        row_count = 0
        if exists:
            try:
                sample = pd.read_csv(path, encoding="utf-8-sig", nrows=100)
                has_available_date = "available_date" in sample.columns
                row_count = int(sample.shape[0])
            except Exception:
                has_available_date = False
        rows.append(
            {
                "dataset": name,
                "path": str(path),
                "exists": exists,
                "row_count_sample": row_count,
                "has_available_date": has_available_date,
                "point_in_time_pass": bool(exists and has_available_date),
                "backtest_status": "eligible" if exists and has_available_date else "observation_only",
            }
        )
    rows.append(
        {
            "dataset": "current_components",
            "path": "data_contract.current_components_backtest_allowed",
            "exists": True,
            "row_count_sample": 0,
            "has_available_date": False,
            "point_in_time_pass": not bool(data_contract.get("current_components_backtest_allowed", False)),
            "backtest_status": "blocked_for_historical_backtest",
        }
    )
    rows.append(
        {
            "dataset": "industry_snapshot",
            "path": "data_contract.industry_snapshot_backtest_allowed",
            "exists": True,
            "row_count_sample": 0,
            "has_available_date": False,
            "point_in_time_pass": not bool(data_contract.get("industry_snapshot_backtest_allowed", False)),
            "backtest_status": "blocked_for_historical_backtest",
        }
    )
    return pd.DataFrame(rows)


def expert_state_ic_summary(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    if monthly_ic.empty:
        return pd.DataFrame()
    return (
        monthly_ic.groupby(["expert", "asset_type_scope", "state"], dropna=False)
        .agg(
            observations=("rank_ic", "count"),
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            positive_ic_rate=("rank_ic", lambda s: float((s > 0).mean())),
            avg_asset_count=("asset_count", "mean"),
        )
        .reset_index()
        .sort_values(["expert", "rank_ic_mean"], ascending=[True, False])
    )


def run_walk_forward_ablation(
    model,
    panel: dict,
    config: dict,
    specs: dict,
    disabled_by_year: dict[int, set[str]],
    start_date: pd.Timestamp,
    base_summary: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    rows = []
    base = base_summary[base_summary["cost_bps"].astype(float).eq(float(cost_bps))].head(1)
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0].to_dict()
    base_row["variant"] = "base_walk_forward"
    base_row["extra_disabled"] = ""
    base_row["target_rows"] = np.nan
    rows.append(base_row)

    applied_experts = sorted(name for name, spec in specs.items() if spec.get("applies_to_portfolio", True))
    for expert in applied_experts:
        variant_disabled = {year: set(disabled) | {expert} for year, disabled in disabled_by_year.items()}
        targets = model.build_targets(
            panel["eligible"],
            panel["regimes"],
            config,
            start_date=start_date,
            disabled_experts=set(),
            disabled_experts_by_year=variant_disabled,
        )
        if not targets.empty:
            targets["test_year"] = pd.to_datetime(targets["signal_date"]).dt.year
            targets = targets[targets["test_year"].isin(variant_disabled.keys())]
        bt = model.run_backtest(panel["returns"], targets, cost_bps, panel["broad_code"])
        summary = model.summarize_nav(bt["nav"])
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["cost_bps"] = float(cost_bps)
        row["variant"] = f"without_{expert}"
        row["extra_disabled"] = expert
        row["target_rows"] = int(targets.shape[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_values = out[out["variant"].eq("base_walk_forward")].iloc[0]
    for col in ["annual_return", "sharpe_no_rf", "max_drawdown", "avg_cash_weight", "avg_trade_turnover"]:
        if col in out.columns:
            out[f"delta_{col}_vs_base"] = out[col] - base_values[col]
    first_cols = ["variant", "extra_disabled", "cost_bps", "annual_return", "sharpe_no_rf", "max_drawdown", "avg_cash_weight", "avg_trade_turnover"]
    first_cols = [col for col in first_cols if col in out.columns]
    other_cols = [col for col in out.columns if col not in first_cols]
    return out[first_cols + other_cols]


def smoke_test_targets(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"check": "target_rows_non_empty", "pass": bool(not targets.empty), "detail": str(int(targets.shape[0]))})
    if targets.empty:
        return pd.DataFrame(rows)
    weight_sum = targets.groupby("signal_date")["weight"].sum()
    min_weight = float(targets["weight"].min())
    rows.extend(
        [
            {"check": "no_negative_weights", "pass": bool(min_weight >= -1e-10), "detail": f"min_weight={min_weight:.8f}"},
            {"check": "weight_sum_not_above_one", "pass": bool((weight_sum <= 1.000001).all()), "detail": f"max_sum={float(weight_sum.max()):.8f}"},
            {"check": "weight_sum_near_one", "pass": bool((weight_sum >= 0.999).all()), "detail": f"min_sum={float(weight_sum.min()):.8f}"},
            {"check": "no_missing_asset", "pass": bool(targets["asset"].notna().all()), "detail": ""},
            {"check": "no_missing_signal_date", "pass": bool(targets["signal_date"].notna().all()), "detail": ""},
        ]
    )
    return pd.DataFrame(rows)


def make_pbo_dsr_report(
    summaries: pd.DataFrame,
    gate_history: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    gate_cfg = config.get("expert_gates", {})
    baseline_sharpe = float(gate_cfg.get("baseline_10bps_sharpe", 0.455))
    baseline_mdd = float(gate_cfg.get("baseline_10bps_max_drawdown", -0.5455))
    allowed_mdd = baseline_mdd - float(gate_cfg.get("max_drawdown_allowed_slippage", 0.03))
    row10 = summaries[summaries["cost_bps"].astype(float).eq(10.0)].head(1)
    sharpe = float(row10["sharpe_no_rf"].iloc[0]) if not row10.empty else np.nan
    max_drawdown = float(row10["max_drawdown"].iloc[0]) if not row10.empty else np.nan
    trial_count = max(1, int(gate_history["expert"].nunique()) if not gate_history.empty else 1)
    enabled = gate_history[
        gate_history["portfolio_enabled"].astype(bool) & gate_history["applies_to_portfolio"].astype(bool)
    ] if not gate_history.empty else pd.DataFrame()
    enabled_failure_rate = float((enabled["test_rank_ic_mean"] <= 0).mean()) if not enabled.empty and "test_rank_ic_mean" in enabled.columns else np.nan
    years = int(summaries.get("oos_years", pd.Series([0])).max()) if "oos_years" in summaries.columns and not summaries.empty else max(1, gate_history["test_year"].nunique() if not gate_history.empty else 1)
    noise_penalty = math.sqrt(2.0 * math.log(trial_count) / max(years, 1)) if trial_count > 1 else 0.0
    deflated_sharpe_proxy = sharpe - noise_penalty if pd.notna(sharpe) else np.nan
    pbo_proxy_flag = bool(pd.notna(enabled_failure_rate) and enabled_failure_rate > 0.5)
    acceptance = bool(
        pd.notna(sharpe)
        and sharpe >= baseline_sharpe
        and pd.notna(max_drawdown)
        and max_drawdown >= allowed_mdd
        and not pbo_proxy_flag
    )
    return pd.DataFrame(
        [
            {"metric": "formal_pbo_available", "value": 0.0, "pass": False, "method": "not_computed", "interpretation": "Formal CSCV PBO requires a tested variant grid; this run records a proxy only."},
            {"metric": "pbo_proxy_enabled_failure_rate", "value": enabled_failure_rate, "pass": bool(not pbo_proxy_flag), "method": "enabled_experts_with_next_year_rank_ic_le_zero", "interpretation": "High value means gates selected experts whose next-year IC failed."},
            {"metric": "deflated_sharpe_proxy", "value": deflated_sharpe_proxy, "pass": bool(pd.notna(deflated_sharpe_proxy) and deflated_sharpe_proxy > 0), "method": "sharpe_minus_multiple_testing_noise_penalty", "interpretation": "Conservative proxy, not a formal Bailey deflated Sharpe implementation."},
            {"metric": "oos_10bps_sharpe", "value": sharpe, "pass": bool(pd.notna(sharpe) and sharpe >= baseline_sharpe), "method": "walk_forward_oos_backtest", "interpretation": f"Baseline Sharpe is {baseline_sharpe:.3f}."},
            {"metric": "oos_10bps_max_drawdown", "value": max_drawdown, "pass": bool(pd.notna(max_drawdown) and max_drawdown >= allowed_mdd), "method": "walk_forward_oos_backtest", "interpretation": f"Allowed floor is {allowed_mdd:.2%}."},
            {"metric": "overall_acceptance_proxy", "value": float(acceptance), "pass": acceptance, "method": "combined_proxy_gate", "interpretation": "Requires Sharpe, drawdown, and PBO proxy gates."},
        ]
    )


def write_markdown_reports(
    output_dir: Path,
    summaries: pd.DataFrame,
    gate_history: pd.DataFrame,
    pbo_dsr: pd.DataFrame,
    smoke: pd.DataFrame,
    data_audit: pd.DataFrame,
    state_ic: pd.DataFrame,
    ablation: pd.DataFrame,
    nav10: pd.DataFrame,
    regimes: pd.DataFrame,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = config.get("expert_gates", {})
    lines = [
        "# HIRSSM V2.0 Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Baseline",
        "",
        f"- Baseline 10bps annual return: {float(baseline.get('baseline_10bps_annual_return', 0.0867)):.2%}",
        f"- Baseline 10bps Sharpe: {float(baseline.get('baseline_10bps_sharpe', 0.455)):.3f}",
        f"- Baseline 10bps max drawdown: {float(baseline.get('baseline_10bps_max_drawdown', -0.5455)):.2%}",
        "",
        "## OOS Cost Sensitivity",
        "",
        summaries.to_markdown(index=False) if not summaries.empty else "No OOS summary.",
        "",
        "## PBO / DSR Proxy",
        "",
        pbo_dsr.to_markdown(index=False) if not pbo_dsr.empty else "No PBO/DSR report.",
        "",
        "## Smoke Tests",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke tests.",
        "",
        "## Data Contract Audit",
        "",
        data_audit.to_markdown(index=False) if not data_audit.empty else "No data audit.",
        "",
        "## Walk-Forward Expert Ablation",
        "",
        ablation.to_markdown(index=False) if not ablation.empty else "No ablation report.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    gate_summary = pd.DataFrame()
    if not gate_history.empty:
        gate_summary = (
            gate_history.groupby("expert")
            .agg(
                windows=("test_year", "count"),
                portfolio_enabled_rate=("portfolio_enabled", "mean"),
                raw_pass_rate=("raw_gate_pass", "mean"),
                avg_train_rank_ic=("rank_ic_mean", "mean"),
                avg_test_rank_ic=("test_rank_ic_mean", "mean"),
                avg_positive_ic_rate=("positive_ic_rate", "mean"),
            )
            .reset_index()
            .sort_values(["portfolio_enabled_rate", "avg_train_rank_ic"], ascending=False)
        )
    factor_lines = [
        "# Factor Gate Report",
        "",
        "This report is based on past-only walk-forward gates. Full-sample ablation can reject weak experts but cannot promote new defaults.",
        "",
        "## Expert Gate Summary",
        "",
        gate_summary.to_markdown(index=False) if not gate_summary.empty else "No gate summary.",
        "",
        "## Latest Gate Decisions",
        "",
        gate_history.sort_values(["test_year", "expert"]).tail(30).to_markdown(index=False) if not gate_history.empty else "No gate history.",
        "",
        "## Expert State RankIC",
        "",
        state_ic.to_markdown(index=False) if not state_ic.empty else "No state IC report.",
    ]
    (output_dir / "FACTOR_GATE_REPORT.md").write_text("\n".join(factor_lines), encoding="utf-8")

    changelog = [
        "# Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Added walk-forward expert gating with five-year training and one-year test windows.",
        "- Split trend and liquidity overlays into style and industry scoped gate decisions.",
        "- Added defensive cash substitution when defensive candidates have negative final score and no risk-compression support.",
        "- Added point-in-time macro data contract audit; missing available_date datasets remain observation-only.",
        "- Added OOS cost sensitivity, gate decision log, and PBO/deflated-Sharpe proxy reporting.",
        "",
        "## Governance Decision",
        "",
        "- New experts are not promoted by full-sample performance.",
        "- Default production promotion requires walk-forward evidence and data contract pass.",
        "- Formal PBO is not claimed until a parameter/variant grid is evaluated with CSCV.",
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
            "# Regime Failure Cases",
            "",
            "## Worst Drawdown Days",
            "",
            worst_days.to_markdown(index=False),
            "",
            "## Drawdown By Regime",
            "",
            by_state.to_markdown(index=False),
            "",
            "## Review Notes",
            "",
            "- Use this file to inspect whether losses concentrate in specific regimes before changing default gates.",
            "- A drawdown improvement caused only by permanent high cash should not be accepted as model improvement.",
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

    model = load_model()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    gate_cfg = config.get("expert_gates", {})
    specs = {name: EXPERT_SPECS[name] for name in gate_cfg.get("candidate_experts", EXPERT_SPECS.keys()) if name in EXPERT_SPECS}

    panel = build_panel(model, root, config, args.start_date, args.end_date)
    horizon = int(gate_cfg.get("horizon_days", 21))
    monthly_ic = compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, horizon)
    gate_history = build_gate_history(monthly_ic, config, specs)
    disabled_by_year = build_disabled_by_year(gate_history)
    first_test_year = int(gate_history["test_year"].min()) if not gate_history.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_test_year, month=1, day=1)

    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=set(),
        disabled_experts_by_year=disabled_by_year,
    )
    targets = add_gate_columns(targets, gate_history)
    targets = targets[targets["test_year"].isin(disabled_by_year.keys())] if "test_year" in targets.columns else targets

    summaries = []
    nav_by_cost: dict[float, pd.DataFrame] = {}
    for cost_bps in config["validation"]["cost_bps_scenarios"]:
        cost = float(cost_bps)
        bt = model.run_backtest(panel["returns"], targets, cost, panel["broad_code"])
        nav = bt["nav"]
        nav_by_cost[cost] = nav
        summary = model.summarize_nav(nav)
        if not summary.empty:
            summary.insert(0, "cost_bps", cost)
            summary["oos_start"] = nav["date"].min()
            summary["oos_end"] = nav["date"].max()
            summary["oos_years"] = (pd.to_datetime(nav["date"].max()) - pd.to_datetime(nav["date"].min())).days / 365.25
            summaries.append(summary)
        suffix = f"{int(cost)}bps"
        model.write_csv(nav, output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(nav), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(nav, panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")

    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    if not summary_df.empty:
        baseline_sharpe = float(gate_cfg.get("baseline_10bps_sharpe", 0.455))
        baseline_mdd = float(gate_cfg.get("baseline_10bps_max_drawdown", -0.5455))
        baseline_ann = float(gate_cfg.get("baseline_10bps_annual_return", 0.0867))
        summary_df["delta_sharpe_vs_10bps_baseline"] = np.where(summary_df["cost_bps"].eq(10.0), summary_df["sharpe_no_rf"] - baseline_sharpe, np.nan)
        summary_df["delta_mdd_vs_10bps_baseline"] = np.where(summary_df["cost_bps"].eq(10.0), summary_df["max_drawdown"] - baseline_mdd, np.nan)
        summary_df["delta_annual_return_vs_10bps_baseline"] = np.where(summary_df["cost_bps"].eq(10.0), summary_df["annual_return"] - baseline_ann, np.nan)

    pbo_dsr = make_pbo_dsr_report(summary_df, gate_history, config)
    smoke = smoke_test_targets(targets)
    data_audit = data_contract_audit(root, config)
    state_ic = expert_state_ic_summary(monthly_ic)
    ablation = run_walk_forward_ablation(
        model,
        panel,
        config,
        specs,
        disabled_by_year,
        start_date,
        summary_df,
        cost_bps=10.0,
    )

    model.write_csv(monthly_ic, output_dir / "monthly_expert_rank_ic.csv")
    model.write_csv(state_ic, output_dir / "expert_state_rank_ic.csv")
    model.write_csv(gate_history, output_dir / "expert_gate_history.csv")
    model.write_csv(gate_history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(targets, output_dir / "walk_forward_target_weights.csv")
    model.write_csv(summary_df, output_dir / "oos_performance.csv")
    model.write_csv(pbo_dsr, output_dir / "pbo_dsr_report.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(data_audit, output_dir / "data_contract_audit.csv")
    model.write_csv(ablation, output_dir / "walk_forward_expert_ablation.csv")

    write_markdown_reports(
        output_dir,
        summary_df,
        gate_history,
        pbo_dsr,
        smoke,
        data_audit,
        state_ic,
        ablation,
        nav_by_cost.get(10.0, pd.DataFrame()),
        panel["regimes"],
        config,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "gate_rows": int(gate_history.shape[0]),
                "target_rows": int(targets.shape[0]),
                "summary_rows": int(summary_df.shape[0]),
                "smoke_pass": bool(smoke["pass"].all()) if not smoke.empty else False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
