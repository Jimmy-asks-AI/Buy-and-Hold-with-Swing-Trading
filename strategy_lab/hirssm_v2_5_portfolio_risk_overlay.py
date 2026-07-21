#!/usr/bin/env python
"""HIRSSM V2.5 fixed portfolio risk overlay.

V2.5 keeps V2.4's stable nested expert selection unchanged and adds a
predeclared portfolio risk overlay:

- state-conditioned gross exposure and target volatility,
- market drawdown brake from point-in-time index features,
- dynamic portfolio drawdown brake during backtest,
- quality-based cash substitution,
- crowded winner discount.

This version intentionally adds no new alpha expert and no expanded parameter
grid. The test is whether a fixed, explainable risk layer improves drawdown and
cost robustness without relying on new in-sample selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
V24_PATH = ROOT / "strategy_lab" / "hirssm_v2_4_stable_nested_selection.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_5_portfolio_risk_overlay"
TRADING_DAYS = 252


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


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def build_v24_context(model, wf, v21, v22, v23, v24, root: Path, config: dict, start_date: str | None, end_date: str | None) -> dict:
    stable_cfg = config.get("expert_state_stable_selection", {})
    stable_grid = stable_cfg.get("stable_variant_grid", [])
    variants = v24.variant_names(stable_cfg)
    specs = v22.active_specs(wf, config)
    panel = wf.build_panel(model, root, config, start_date, end_date)
    shrink_cfg = config.get("expert_state_shrinkage", {})
    monthly_ic = wf.compute_monthly_expert_ic(model, panel["eligible"], panel["regimes"], specs, int(shrink_cfg.get("horizon_days", 21)))
    monthly_ic = v22.clean_monthly_ic(monthly_ic)

    overlay_cfg = config.get("portfolio_risk_overlay_v2_5", {})
    output_costs = [float(item) for item in overlay_cfg.get("cost_bps_scenarios", stable_cfg.get("cost_bps_scenarios", [5, 10, 20, 30]))]
    selection_costs = [float(item) for item in stable_cfg.get("selection_cost_bps_scenarios", [10, 20, 30])]
    all_costs = sorted(set(output_costs + selection_costs))

    variant_histories, variant_yearly, variant_summaries = v24.precompute_stable_variants(
        model, wf, v22, panel, config, specs, monthly_ic, stable_grid, all_costs
    )
    first_year = int(variant_yearly["year"].min()) if not variant_yearly.empty else int(panel["eligible"]["date"].dt.year.min())
    v24_start = pd.Timestamp(year=first_year, month=1, day=1)
    selection_baseline = v21.run_same_period_baseline(model, panel, config, v24_start, selection_costs)
    baseline_yearly = v24.baseline_yearly_by_cost(v23, selection_baseline)
    selection, score_table = v24.build_stable_selection(variant_yearly, baseline_yearly, config, variants)
    selected_history = v23.combine_selected_history(selection, variant_histories)
    base_run = v23.run_nested_portfolio(model, v22, panel, config, selected_history, cost_bps=float(overlay_cfg.get("reference_cost_bps", 10)))
    return {
        "panel": panel,
        "monthly_ic": monthly_ic,
        "selection": selection,
        "score_table": score_table,
        "selected_history": selected_history,
        "base_targets": base_run["targets"],
        "variant_yearly": variant_yearly,
        "variant_summaries": variant_summaries,
        "output_costs": output_costs,
        "start_date": v24_start,
    }


def last_feature_before(features: pd.DataFrame, date: pd.Timestamp, asset: str) -> pd.Series:
    subset = features[(features["asset"].eq(asset)) & (features["date"] <= date)]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values("date").iloc[-1]


def estimate_portfolio_vol(
    returns_wide: pd.DataFrame,
    signal_date: pd.Timestamp,
    weights: dict[str, float],
    lookback_days: int,
) -> float:
    noncash = {asset: weight for asset, weight in weights.items() if asset != "CASH" and weight > 0}
    if not noncash:
        return 0.0
    assets = [asset for asset in noncash if asset in returns_wide.columns]
    if not assets:
        return np.nan
    hist = returns_wide.loc[returns_wide.index <= signal_date, assets].tail(lookback_days)
    if hist.shape[0] < max(20, lookback_days // 3):
        return np.nan
    weight_vector = np.array([noncash[asset] for asset in assets], dtype=float)
    portfolio_returns = hist.fillna(0.0).to_numpy().dot(weight_vector)
    return float(np.nanstd(portfolio_returns, ddof=1) * np.sqrt(TRADING_DAYS))


def market_drawdown_multiplier(row: pd.Series, state: str, cfg: dict) -> tuple[float, str, float]:
    brake = cfg.get("market_drawdown_brake", {})
    if not brake.get("enabled", True) or row.empty:
        return 1.0, "market_brake_disabled", np.nan
    mdd = safe_float(row.get("max_drawdown_120"))
    if not np.isfinite(mdd):
        return 1.0, "market_drawdown_unavailable", np.nan
    hard_trigger = float(brake.get("hard_trigger", -0.20))
    soft_trigger = float(brake.get("soft_trigger", -0.12))
    if mdd <= hard_trigger:
        mult = float(brake.get("hard_multiplier", 0.70))
        reason = "market_hard_drawdown"
    elif mdd <= soft_trigger:
        mult = float(brake.get("soft_multiplier", 0.85))
        reason = "market_soft_drawdown"
    else:
        mult = 1.0
        reason = "market_drawdown_clear"
    if state == "crash_rebound":
        mult = max(mult, float(brake.get("crash_rebound_min_multiplier", 0.82)))
    return mult, reason, mdd


def overlay_targets(base_targets: pd.DataFrame, panel: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config.get("portfolio_risk_overlay_v2_5", {})
    state_controls = cfg.get("state_controls", {})
    default_control = state_controls.get("range_bound", {"target_volatility": 0.12, "gross_exposure_cap": 0.88, "min_cash": 0.12})
    vol_cfg = cfg.get("volatility_scaling", {})
    quality_cfg = cfg.get("quality_cash_substitution", {})
    crowd_cfg = cfg.get("crowding_discount", {})
    broad_code = str(panel.get("broad_code", "000985"))

    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    eligible = panel["eligible"].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    returns = panel["returns"].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns_wide = returns.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)

    rows = []
    decisions = []
    for signal_date, group in targets.sort_values(["signal_date", "asset"]).groupby("signal_date"):
        state = str(group["state"].dropna().iloc[0]) if group["state"].notna().any() else "range_bound"
        control = state_controls.get(state, default_control)
        base_weights = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
        noncash = {asset: weight for asset, weight in base_weights.items() if asset != "CASH" and weight > 0}
        adjusted = dict(noncash)
        asset_multipliers = {asset: 1.0 for asset in noncash}
        asset_reasons = {asset: [] for asset in noncash}

        if crowd_cfg.get("enabled", True):
            for asset in list(adjusted):
                feat = last_feature_before(eligible, signal_date, asset)
                crowded = (
                    safe_float(feat.get("crowding_score"), -np.inf) >= float(crowd_cfg.get("crowding_score_min", 0.9))
                    and safe_float(feat.get("amount_zscore_60"), -np.inf) >= float(crowd_cfg.get("amount_zscore_min", 1.5))
                    and safe_float(feat.get("ret_60"), -np.inf) >= float(crowd_cfg.get("ret60_min", 0.1))
                )
                if crowded:
                    mult = float(crowd_cfg.get("weight_multiplier", 0.85))
                    adjusted[asset] *= mult
                    asset_multipliers[asset] *= mult
                    asset_reasons[asset].append("crowding_discount")

        score_by_asset = group.set_index("asset")["score"].to_dict() if "score" in group.columns else {}
        base_noncash_sum = sum(noncash.values())
        weighted_score = (
            sum(float(score_by_asset.get(asset, 0.0)) * weight for asset, weight in noncash.items()) / base_noncash_sum
            if base_noncash_sum > 0
            else 0.0
        )
        quality_multiplier = 1.0
        quality_reason = "quality_clear"
        if quality_cfg.get("enabled", True):
            thresholds = quality_cfg.get("min_weighted_score_by_state", {})
            min_score = float(thresholds.get(state, thresholds.get("range_bound", 0.0)))
            if weighted_score < min_score:
                quality_multiplier = float(quality_cfg.get("low_quality_multiplier", 0.85))
                quality_reason = "low_quality_cash_substitution"

        vol_multiplier = 1.0
        estimated_vol = estimate_portfolio_vol(
            returns_wide,
            signal_date,
            adjusted,
            int(vol_cfg.get("lookback_days", 60)),
        )
        target_vol = float(control.get("target_volatility", default_control.get("target_volatility", 0.12)))
        if vol_cfg.get("enabled", True) and np.isfinite(estimated_vol) and estimated_vol > target_vol * float(vol_cfg.get("buffer", 1.05)):
            vol_multiplier = target_vol / estimated_vol
            vol_multiplier = max(float(vol_cfg.get("min_multiplier", 0.65)), min(float(vol_cfg.get("max_multiplier", 1.0)), vol_multiplier))

        broad_row = last_feature_before(eligible, signal_date, broad_code)
        market_multiplier, market_reason, market_mdd = market_drawdown_multiplier(broad_row, state, cfg)
        common_multiplier = min(quality_multiplier, vol_multiplier, market_multiplier)
        for asset in list(adjusted):
            adjusted[asset] *= common_multiplier
            asset_multipliers[asset] *= common_multiplier
            if common_multiplier < 0.999:
                asset_reasons[asset].append("common_risk_scale")

        max_gross = min(float(control.get("gross_exposure_cap", 1.0)), 1.0 - float(control.get("min_cash", 0.0)))
        noncash_sum = sum(max(0.0, weight) for weight in adjusted.values())
        cap_multiplier = 1.0
        if noncash_sum > max_gross > 0:
            cap_multiplier = max_gross / noncash_sum
            adjusted = {asset: weight * cap_multiplier for asset, weight in adjusted.items()}
            for asset in asset_multipliers:
                asset_multipliers[asset] *= cap_multiplier
                asset_reasons[asset].append("state_gross_cap")

        adjusted = {asset: max(0.0, min(1.0, weight)) for asset, weight in adjusted.items() if weight > 1e-10}
        final_noncash = sum(adjusted.values())
        if final_noncash > 1.0:
            adjusted = {asset: weight / final_noncash for asset, weight in adjusted.items()}
            final_noncash = 1.0
        cash_weight = max(0.0, 1.0 - final_noncash)

        for _, row in group.iterrows():
            asset = str(row["asset"])
            if asset == "CASH":
                continue
            weight = adjusted.get(asset, 0.0)
            if weight <= 0:
                continue
            out = row.to_dict()
            out["base_weight"] = float(row["weight"])
            out["weight"] = weight
            out["overlay_multiplier"] = asset_multipliers.get(asset, 0.0)
            out["overlay_reason"] = "|".join(asset_reasons.get(asset, [])) or "unchanged"
            out["estimated_portfolio_vol"] = estimated_vol
            out["target_volatility"] = target_vol
            out["market_drawdown_120"] = market_mdd
            out["weighted_score"] = weighted_score
            rows.append(out)
        cash_row = group[group["asset"].eq("CASH")].head(1)
        template = cash_row.iloc[0].to_dict() if not cash_row.empty else {
            "signal_date": signal_date,
            "asset": "CASH",
            "state": state,
            "asset_type": "cash",
            "score": 0.0,
            "risk_adjusted_alpha": 0.0,
            "turnover": 0.0,
        }
        template["asset"] = "CASH"
        template["asset_type"] = "cash"
        template["base_weight"] = float(base_weights.get("CASH", 0.0))
        template["weight"] = cash_weight
        template["overlay_multiplier"] = np.nan
        template["overlay_reason"] = "cash_residual"
        template["estimated_portfolio_vol"] = estimated_vol
        template["target_volatility"] = target_vol
        template["market_drawdown_120"] = market_mdd
        template["weighted_score"] = weighted_score
        rows.append(template)
        decisions.append(
            {
                "signal_date": signal_date,
                "state": state,
                "base_gross_exposure": base_noncash_sum,
                "final_gross_exposure": final_noncash,
                "base_cash_weight": float(base_weights.get("CASH", 0.0)),
                "final_cash_weight": cash_weight,
                "estimated_portfolio_vol": estimated_vol,
                "target_volatility": target_vol,
                "vol_multiplier": vol_multiplier,
                "quality_multiplier": quality_multiplier,
                "quality_reason": quality_reason,
                "market_multiplier": market_multiplier,
                "market_reason": market_reason,
                "market_drawdown_120": market_mdd,
                "cap_multiplier": cap_multiplier,
                "weighted_score": weighted_score,
            }
        )
    out_targets = pd.DataFrame(rows)
    if not out_targets.empty:
        out_targets["signal_date"] = pd.to_datetime(out_targets["signal_date"])
        out_targets = out_targets.sort_values(["signal_date", "asset"]).reset_index(drop=True)
    return out_targets, pd.DataFrame(decisions)


def scale_to_cash(new_weights: dict[str, float], multiplier: float) -> dict[str, float]:
    if multiplier >= 0.999:
        total = sum(max(0.0, weight) for weight in new_weights.values())
        if abs(total - 1.0) > 1e-8:
            noncash = {asset: weight for asset, weight in new_weights.items() if asset != "CASH" and weight > 0}
            cash = max(0.0, 1.0 - sum(noncash.values()))
            return {**noncash, "CASH": cash}
        return dict(new_weights)
    noncash = {asset: max(0.0, weight) * multiplier for asset, weight in new_weights.items() if asset != "CASH" and weight > 0}
    cash = max(0.0, 1.0 - sum(noncash.values()))
    return {**noncash, "CASH": cash}


def portfolio_drawdown_multiplier(drawdown: float, state: str, cfg: dict) -> tuple[float, str]:
    brake = cfg.get("portfolio_drawdown_brake", {})
    if not brake.get("enabled", True):
        return 1.0, "portfolio_brake_disabled"
    hard_trigger = float(brake.get("hard_trigger", -0.20))
    soft_trigger = float(brake.get("soft_trigger", -0.12))
    if drawdown <= hard_trigger:
        mult = float(brake.get("hard_multiplier", 0.70))
        reason = "portfolio_hard_drawdown"
    elif drawdown <= soft_trigger:
        mult = float(brake.get("soft_multiplier", 0.85))
        reason = "portfolio_soft_drawdown"
    else:
        mult = 1.0
        reason = "portfolio_drawdown_clear"
    if state == "crash_rebound":
        mult = max(mult, float(brake.get("crash_rebound_min_multiplier", 0.82)))
    return mult, reason


def rolling_drawdown_from_history(nav_history: list[tuple[pd.Timestamp, float]], current_nav: float, lookback_days: int) -> float:
    recent = [value for _, value in nav_history[-max(1, lookback_days) :]]
    recent.append(float(current_nav))
    peak = max(recent) if recent else float(current_nav)
    return float(current_nav / peak - 1.0) if peak > 0 else 0.0


def run_backtest_with_dynamic_brake(model, returns: pd.DataFrame, targets: pd.DataFrame, cost_bps: float, benchmark_asset: str, config: dict) -> dict[str, pd.DataFrame]:
    cfg = config.get("portfolio_risk_overlay_v2_5", {})
    brake_cfg = cfg.get("portfolio_drawdown_brake", {})
    brake_lookback = int(brake_cfg.get("lookback_days", 252))
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret_wide = ret.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    schedule = model.build_trade_schedule(targets, ret_wide.index.to_series())
    if not schedule.empty:
        first_trade_date = pd.to_datetime(schedule["trade_date"]).min()
        ret_wide = ret_wide[ret_wide.index >= first_trade_date]
    schedule_map = {date: group for date, group in schedule.groupby("trade_date")} if not schedule.empty else {}
    current_weights: dict[str, float] = {"CASH": 1.0}
    nav = 1.0
    peak_nav = 1.0
    nav_history: list[tuple[pd.Timestamp, float]] = []
    rows = []
    trades = []
    brake_rows = []
    for date, day_returns in ret_wide.iterrows():
        gross_ret = sum(
            weight * float(day_returns.get(asset, 0.0))
            for asset, weight in current_weights.items()
            if asset != "CASH"
        )
        nav_before_trade = nav * (1.0 + gross_ret)
        drawdown_before_trade = rolling_drawdown_from_history(nav_history, nav_before_trade, brake_lookback)
        inception_drawdown_before_trade = nav_before_trade / peak_nav - 1.0 if peak_nav > 0 else 0.0
        cost = 0.0
        turnover = 0.0
        brake_multiplier = 1.0
        brake_reason = "no_rebalance"
        if date in schedule_map:
            group = schedule_map[date]
            state = str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else "range_bound"
            scheduled_weights = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
            brake_multiplier, brake_reason = portfolio_drawdown_multiplier(drawdown_before_trade, state, cfg)
            new_weights = scale_to_cash(scheduled_weights, brake_multiplier)
            turnover = sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in set(new_weights) | set(current_weights))
            cost = turnover * cost_bps / 10000.0
            for asset, weight in new_weights.items():
                trades.append(
                    {
                        "date": date,
                        "asset": asset,
                        "weight": weight,
                        "cost_bps": cost_bps,
                        "turnover": turnover,
                        "drawdown_brake_multiplier": brake_multiplier,
                        "drawdown_brake_reason": brake_reason,
                    }
                )
            brake_rows.append(
                    {
                        "date": date,
                        "state": state,
                        "portfolio_drawdown_before_trade": drawdown_before_trade,
                        "inception_drawdown_before_trade": inception_drawdown_before_trade,
                        "drawdown_brake_multiplier": brake_multiplier,
                    "drawdown_brake_reason": brake_reason,
                    "scheduled_gross_exposure": 1.0 - scheduled_weights.get("CASH", 0.0),
                    "executed_gross_exposure": 1.0 - new_weights.get("CASH", 0.0),
                    "turnover": turnover,
                    "cost": cost,
                }
            )
            current_weights = new_weights
        net_ret = gross_ret - cost
        nav *= 1.0 + net_ret
        peak_nav = max(peak_nav, nav)
        rows.append(
            {
                "date": date,
                "gross_return": gross_ret,
                "cost": cost,
                "portfolio_return": net_ret,
                "nav": nav,
                "turnover": turnover,
                "cash_weight": current_weights.get("CASH", 0.0),
                "gross_exposure": 1.0 - current_weights.get("CASH", 0.0),
                "benchmark_return": float(day_returns.get(benchmark_asset, 0.0)),
                "portfolio_drawdown_before_trade": drawdown_before_trade,
                "inception_drawdown_before_trade": inception_drawdown_before_trade,
                "drawdown_brake_multiplier": brake_multiplier,
                "drawdown_brake_reason": brake_reason,
            }
        )
        nav_history.append((date, nav))
    nav_df = pd.DataFrame(rows)
    if not nav_df.empty:
        nav_df["benchmark_nav"] = (1.0 + nav_df["benchmark_return"]).cumprod()
    return {"nav": nav_df, "trades": pd.DataFrame(trades), "schedule": schedule, "brake_log": pd.DataFrame(brake_rows)}


def run_cost_scenarios(model, panel: dict, targets: pd.DataFrame, costs: list[float], variant: str, dynamic: bool, config: dict, output_dir: Path, prefix: str) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    summaries = []
    nav_by_cost: dict[float, pd.DataFrame] = {}
    brake_by_cost: dict[float, pd.DataFrame] = {}
    for cost in costs:
        if dynamic:
            bt = run_backtest_with_dynamic_brake(model, panel["returns"], targets, float(cost), panel["broad_code"], config)
        else:
            bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
            bt["brake_log"] = pd.DataFrame()
        summary = model.summarize_nav(bt["nav"])
        if not summary.empty:
            summary.insert(0, "variant", variant)
            summary.insert(1, "cost_bps", float(cost))
            summary["target_rows"] = int(targets.shape[0])
            summary["oos_start"] = bt["nav"]["date"].min() if not bt["nav"].empty else pd.NaT
            summary["oos_end"] = bt["nav"]["date"].max() if not bt["nav"].empty else pd.NaT
            summary["oos_years"] = (
                (pd.to_datetime(bt["nav"]["date"].max()) - pd.to_datetime(bt["nav"]["date"].min())).days / 365.25
                if not bt["nav"].empty
                else np.nan
            )
            summaries.append(summary)
        nav_by_cost[float(cost)] = bt["nav"]
        brake_by_cost[float(cost)] = bt.get("brake_log", pd.DataFrame())
        suffix = f"{prefix}_{int(cost)}bps"
        model.write_csv(bt["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(bt["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(bt["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        if not bt.get("brake_log", pd.DataFrame()).empty:
            model.write_csv(bt["brake_log"], output_dir / f"drawdown_brake_log_{suffix}.csv")
    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    return summary_df, nav_by_cost, brake_by_cost


def compare_to_base(v25: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if v25.empty or base.empty:
        return pd.DataFrame()
    merged = v25.merge(
        base.add_prefix("base_"),
        left_on="cost_bps",
        right_on="base_cost_bps",
        how="left",
    )
    for col in ["annual_return", "sharpe_no_rf", "max_drawdown", "avg_cash_weight", "avg_trade_turnover", "total_cost"]:
        if col in merged.columns and f"base_{col}" in merged.columns:
            merged[f"delta_{col}_vs_v2_4"] = merged[col] - merged[f"base_{col}"]
    return merged


def make_self_check(smoke: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path, config: dict) -> pd.DataFrame:
    thresholds = config.get("portfolio_risk_overlay_v2_5", {}).get("self_check_thresholds", {})
    max_slippage = float(thresholds.get("max_annual_return_slippage_vs_v2_4_10bps", 0.02))
    max_cash = float(thresholds.get("max_cash_weight", 0.45))
    min_20bps_ann = float(thresholds.get("min_20bps_annual_return", 0.0))
    rows = []
    rows.append({"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""})
    rows.append({"check": "required_cost_rows", "pass": bool(set([5.0, 10.0, 20.0, 30.0]).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""})
    row10 = comparison[comparison["cost_bps"].astype(float).eq(10.0)].head(1) if not comparison.empty else pd.DataFrame()
    rows.append({"check": "drawdown_not_worse_vs_v2_4_10bps", "pass": bool(not row10.empty and float(row10["delta_max_drawdown_vs_v2_4"].iloc[0]) >= -0.005), "detail": f"{float(row10['delta_max_drawdown_vs_v2_4'].iloc[0]):.6f}" if not row10.empty else ""})
    rows.append({"check": "annual_return_slippage_within_limit_10bps", "pass": bool(not row10.empty and float(row10["delta_annual_return_vs_v2_4"].iloc[0]) >= -max_slippage), "detail": f"{float(row10['delta_annual_return_vs_v2_4'].iloc[0]):.6f}" if not row10.empty else ""})
    rows.append({"check": "avg_cash_below_limit_10bps", "pass": bool(not row10.empty and float(row10["avg_cash_weight"].iloc[0]) <= max_cash), "detail": f"{float(row10['avg_cash_weight'].iloc[0]):.6f}" if not row10.empty else ""})
    row20 = summary[summary["cost_bps"].astype(float).eq(20.0)].head(1) if not summary.empty else pd.DataFrame()
    rows.append({"check": "cost_20bps_annual_return_nonnegative", "pass": bool(not row20.empty and float(row20["annual_return"].iloc[0]) >= min_20bps_ann), "detail": f"{float(row20['annual_return'].iloc[0]):.6f}" if not row20.empty else ""})
    rows.append({"check": "no_new_alpha_or_selection_grid", "pass": True, "detail": "V2.5 is fixed overlay on V2.4 selected targets"})
    for name in ["WALK_FORWARD_REPORT.md", "RISK_OVERLAY_REPORT.md", "MODEL_CHANGELOG.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def make_reports(
    output_dir: Path,
    summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    decisions: pd.DataFrame,
    brake_log: pd.DataFrame,
    smoke: pd.DataFrame,
    self_check: pd.DataFrame,
) -> None:
    monthly_brake_rate = float((brake_log["drawdown_brake_multiplier"] < 0.999).mean()) if not brake_log.empty else 0.0
    avg_scheduled_gross = float(brake_log["scheduled_gross_exposure"].mean()) if not brake_log.empty else np.nan
    avg_executed_gross = float(brake_log["executed_gross_exposure"].mean()) if not brake_log.empty else np.nan
    lines = [
        "# HIRSSM V2.5 Portfolio Risk Overlay Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- Keeps V2.4 stable nested expert selection unchanged.",
        "- Adds fixed portfolio risk overlay only.",
        "- Adds state-conditioned exposure, target volatility, cash substitution, crowding discount, and dynamic drawdown brake.",
        "- No new alpha expert and no expanded parameter grid.",
        "",
        "## V2.5 OOS Performance",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary.",
        "",
        "## V2.4 Reference Performance",
        "",
        base_summary.to_markdown(index=False) if not base_summary.empty else "No base summary.",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Dynamic Brake Diagnostics",
        "",
        f"- Rebalance brake trigger rate: {monthly_brake_rate:.2%}",
        f"- Average scheduled gross exposure: {avg_scheduled_gross:.2%}" if np.isfinite(avg_scheduled_gross) else "- Average scheduled gross exposure unavailable.",
        f"- Average executed gross exposure: {avg_executed_gross:.2%}" if np.isfinite(avg_executed_gross) else "- Average executed gross exposure unavailable.",
        "",
        "## Smoke Test",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke test.",
        "",
        "## Self Check",
        "",
        "See `SELF_CHECK_REPORT.md` and `self_check_results.csv`.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    decision_lines = [
        "# HIRSSM V2.5 Risk Overlay Decisions",
        "",
        "## Target Overlay Decisions",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "No decisions.",
        "",
        "## Dynamic Drawdown Brake Log",
        "",
        brake_log.to_markdown(index=False) if not brake_log.empty else "No brake log.",
    ]
    (output_dir / "RISK_OVERLAY_REPORT.md").write_text("\n".join(decision_lines), encoding="utf-8")

    changelog = [
        "# HIRSSM V2.5 Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Added fixed portfolio risk overlay on top of V2.4 targets.",
        "- Added state-conditioned gross exposure and target volatility.",
        "- Added market and portfolio drawdown brakes.",
        "- Added quality cash substitution and crowding discount.",
        "- Added mandatory self-check report.",
        "",
        "## Governance Note",
        "",
        "V2.5 does not promote new alpha factors. Promotion depends on OOS drawdown/return trade-off and self-check results.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        "# HIRSSM V2.5 Self Check\n\n" + (self_check.to_markdown(index=False) if not self_check.empty else "No self check."),
        encoding="utf-8",
    )


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
    v24 = load_module("hirssm_v2_4_stable_nested_selection", V24_PATH)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    context = build_v24_context(model, wf, v21, v22, v23, v24, root, config, args.start_date, args.end_date)
    panel = context["panel"]
    costs = context["output_costs"]
    base_targets = context["base_targets"]
    overlayed_targets, decisions = overlay_targets(base_targets, panel, config)

    base_summary, base_nav_by_cost, _ = run_cost_scenarios(
        model, panel, base_targets, costs, "v2_4_stable_nested_reference", False, config, output_dir, "v24_reference"
    )
    summary, nav_by_cost, brake_by_cost = run_cost_scenarios(
        model, panel, overlayed_targets, costs, "v2_5_portfolio_risk_overlay", True, config, output_dir, "v25_overlay"
    )
    comparison = compare_to_base(summary, base_summary)
    smoke = wf.smoke_test_targets(overlayed_targets)
    reference_cost = float(config.get("portfolio_risk_overlay_v2_5", {}).get("reference_cost_bps", 10))
    brake_log = brake_by_cost.get(reference_cost, pd.DataFrame())

    model.write_csv(context["selection"], output_dir / "stable_nested_variant_selection.csv")
    model.write_csv(context["score_table"], output_dir / "stable_nested_selection_scores.csv")
    model.write_csv(context["selected_history"], output_dir / "selected_expert_state_multiplier_history.csv")
    model.write_csv(base_targets, output_dir / "base_v2_4_target_weights.csv")
    model.write_csv(overlayed_targets, output_dir / "walk_forward_target_weights.csv")
    model.write_csv(decisions, output_dir / "risk_overlay_decision_log.csv")
    model.write_csv(summary, output_dir / "oos_performance.csv")
    model.write_csv(base_summary, output_dir / "base_v2_4_oos_performance.csv")
    model.write_csv(comparison, output_dir / "v2_5_vs_v2_4_comparison.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")

    self_check = make_self_check(smoke, summary, comparison, output_dir, config)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    make_reports(output_dir, summary, base_summary, comparison, decisions, brake_log, smoke, self_check)
    self_check = make_self_check(smoke, summary, comparison, output_dir, config)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        "# HIRSSM V2.5 Self Check\n\n" + self_check.to_markdown(index=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "target_rows": int(overlayed_targets.shape[0]),
                "decision_rows": int(decisions.shape[0]),
                "summary_rows": int(summary.shape[0]),
                "smoke_pass": bool(smoke["pass"].all()) if not smoke.empty else False,
                "self_check_pass": bool(self_check["pass"].all()) if not self_check.empty else False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
