#!/usr/bin/env python
"""HIRSSM V2.6-V2.9 risk-overlay iteration engine.

This script reuses one V2.4 stable nested-selection context, then evaluates
V2.6, V2.7, V2.8, and V2.9 as separate governed model versions.

Version intent:

- V2.6: local sleeve risk overlay and attribution.
- V2.7: V2.6 plus rule-based recovery re-entry.
- V2.8: sleeve-specific repair to preserve style core exposure.
- V2.9: fixed blend of V2.4 return baseline and V2.8 risk overlay.

Each version writes a dedicated output directory with smoke checks, self checks,
performance, attribution, and governance reports.
"""

from __future__ import annotations

import argparse
import copy
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
V25_PATH = ROOT / "strategy_lab" / "hirssm_v2_5_portfolio_risk_overlay.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_ROOT = ROOT / "outputs"
TRADING_DAYS = 252


VERSION_KEYS = [
    "portfolio_risk_overlay_v2_6",
    "portfolio_risk_overlay_v2_7",
    "portfolio_risk_overlay_v2_8",
    "portfolio_risk_overlay_v2_9",
]


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


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolved_version_config(config: dict, key: str) -> dict:
    raw = copy.deepcopy(config.get(key, {}))
    parent_key = raw.get("inherits")
    if parent_key:
        parent = resolved_version_config(config, parent_key)
        raw = deep_merge(parent, raw)
    return raw


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


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
    assets = [asset for asset in noncash if asset in returns_wide.columns]
    if not assets:
        return 0.0
    hist = returns_wide.loc[returns_wide.index <= signal_date, assets].tail(lookback_days)
    if hist.shape[0] < max(20, lookback_days // 3):
        return np.nan
    weights_array = np.array([noncash[asset] for asset in assets], dtype=float)
    returns_array = hist.fillna(0.0).to_numpy().dot(weights_array)
    return float(np.nanstd(returns_array, ddof=1) * np.sqrt(TRADING_DAYS))


def market_brake_multiplier(row: pd.Series, state: str, cfg: dict, reentry_active: bool) -> tuple[float, str, float]:
    brake = cfg.get("market_drawdown_brake", {})
    if not brake.get("enabled", True) or row.empty:
        return 1.0, "market_brake_disabled", np.nan
    mdd = safe_float(row.get("max_drawdown_120"))
    if not np.isfinite(mdd):
        return 1.0, "market_drawdown_unavailable", np.nan
    if mdd <= float(brake.get("hard_trigger", -0.28)):
        mult = float(brake.get("hard_multiplier", 0.88))
        reason = "market_hard_drawdown"
    elif mdd <= float(brake.get("soft_trigger", -0.18)):
        mult = float(brake.get("soft_multiplier", 0.94))
        reason = "market_soft_drawdown"
    else:
        mult = 1.0
        reason = "market_drawdown_clear"
    if state == "crash_rebound":
        mult = max(mult, float(brake.get("crash_rebound_min_multiplier", 0.92)))
    if reentry_active:
        min_mult = float(cfg.get("reentry_rules", {}).get("market_brake_min_multiplier", mult))
        mult = max(mult, min_mult)
        reason = f"{reason}|reentry_floor"
    return mult, reason, mdd


def is_reentry_active(row: pd.Series, state: str, cfg: dict) -> bool:
    rules = cfg.get("reentry_rules", {})
    if not rules.get("enabled", False) or state not in set(rules.get("states", [])) or row.empty:
        return False
    ret20 = safe_float(row.get("ret_20"))
    ma60 = safe_float(row.get("ma_gap_60"))
    mdd = safe_float(row.get("max_drawdown_120"))
    return (
        np.isfinite(ret20)
        and np.isfinite(ma60)
        and np.isfinite(mdd)
        and ret20 >= float(rules.get("market_ret20_min", 0.02))
        and ma60 >= float(rules.get("market_ma60_gap_min", -0.03))
        and mdd >= float(rules.get("market_drawdown_floor", -0.25))
    )


def asset_local_multiplier(row: pd.Series, asset: str, asset_type: str, state: str, cfg: dict) -> tuple[float, list[str]]:
    local = cfg.get("asset_local_risk_discount", {})
    if not local.get("enabled", True) or row.empty:
        return 1.0, []
    mult = 1.0
    reasons: list[str] = []
    vol = safe_float(row.get("vol_60"))
    mdd = safe_float(row.get("max_drawdown_120"))
    crowd = safe_float(row.get("crowding_score"))
    amount_z = safe_float(row.get("amount_zscore_60"))
    ret60 = safe_float(row.get("ret_60"))
    if np.isfinite(vol) and vol >= float(local.get("vol_60_high", 0.36)):
        mult *= float(local.get("high_vol_multiplier", 0.94))
        reasons.append("high_vol")
    if np.isfinite(mdd) and mdd <= float(local.get("max_drawdown_120_hard", -0.24)):
        mult *= float(local.get("deep_drawdown_multiplier", 0.92))
        reasons.append("asset_deep_drawdown")
    crowded = (
        np.isfinite(crowd)
        and np.isfinite(amount_z)
        and np.isfinite(ret60)
        and crowd >= float(local.get("crowding_score_min", 0.9))
        and amount_z >= float(local.get("amount_zscore_min", 1.6))
        and ret60 >= float(local.get("ret60_min", 0.12))
    )
    if crowded:
        mult *= float(local.get("crowding_multiplier", 0.9))
        reasons.append("crowding")
    preservation = cfg.get("sleeve_preservation", {})
    if preservation.get("enabled", False):
        core_assets = set(str(item) for item in preservation.get("style_core_assets", []))
        if asset_type == "style" and asset in core_assets:
            mult = max(mult, float(preservation.get("style_core_min_multiplier", 0.94)))
            reasons.append("style_core_floor")
        if asset_type == "industry":
            mult *= float(preservation.get("industry_risk_budget_multiplier", 1.0))
            reasons.append("industry_budget")
            if state == "risk_off_decline":
                mult *= float(preservation.get("risk_off_industry_multiplier", 1.0))
                reasons.append("risk_off_industry_cut")
    return max(0.0, min(1.0, mult)), reasons


def sleeve_multiplier(
    sleeve: str,
    state: str,
    weights: dict[str, float],
    returns_wide: pd.DataFrame,
    signal_date: pd.Timestamp,
    cfg: dict,
) -> tuple[float, float, float]:
    vol_cfg = cfg.get("sleeve_volatility_scaling", {})
    sleeve_cfg = vol_cfg.get(sleeve, {})
    if not vol_cfg.get("enabled", True) or not weights:
        return 1.0, np.nan, np.nan
    vol = estimate_portfolio_vol(returns_wide, signal_date, weights, int(vol_cfg.get("lookback_days", 60)))
    target = float(sleeve_cfg.get("target_volatility_by_state", {}).get(state, np.nan))
    if not np.isfinite(vol) or not np.isfinite(target) or vol <= target * float(vol_cfg.get("buffer", 1.08)):
        return 1.0, vol, target
    mult = target / vol
    mult = max(float(sleeve_cfg.get("min_multiplier", 0.8)), min(1.0, mult))
    return mult, vol, target


def scale_to_caps(adjusted: dict[str, float], state: str, cfg: dict, reentry_active: bool) -> tuple[dict[str, float], float, float]:
    gross_cap = float(cfg.get("state_gross_cap", {}).get(state, 1.0))
    if reentry_active:
        gross_cap = min(1.0, gross_cap + float(cfg.get("reentry_rules", {}).get("gross_cap_add", 0.0)))
    min_cash = float(cfg.get("min_cash_by_state", {}).get(state, 0.0))
    gross_cap = min(gross_cap, 1.0 - min_cash)
    gross = sum(max(0.0, weight) for weight in adjusted.values())
    cap_mult = 1.0
    if gross > gross_cap > 0:
        cap_mult = gross_cap / gross
        adjusted = {asset: weight * cap_mult for asset, weight in adjusted.items()}
        gross = gross_cap
    return adjusted, cap_mult, gross


def overlay_targets_local(base_targets: pd.DataFrame, panel: dict, cfg: dict, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    eligible = panel["eligible"].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    returns = panel["returns"].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns_wide = returns.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    broad_code = str(panel.get("broad_code", "000985"))
    rows = []
    decisions = []

    for signal_date, group in targets.sort_values(["signal_date", "asset"]).groupby("signal_date"):
        state = str(group["state"].dropna().iloc[0]) if group["state"].notna().any() else "range_bound"
        base_weights = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
        noncash = {asset: weight for asset, weight in base_weights.items() if asset != "CASH" and weight > 0}
        meta = group.set_index("asset").to_dict("index")
        broad_row = last_feature_before(eligible, signal_date, broad_code)
        reentry = is_reentry_active(broad_row, state, cfg)
        adjusted = dict(noncash)
        multipliers = {asset: 1.0 for asset in noncash}
        reasons = {asset: [] for asset in noncash}

        for asset in list(adjusted):
            asset_type = str(meta.get(asset, {}).get("asset_type", ""))
            feat = last_feature_before(eligible, signal_date, asset)
            mult, asset_reasons = asset_local_multiplier(feat, asset, asset_type, state, cfg)
            adjusted[asset] *= mult
            multipliers[asset] *= mult
            reasons[asset].extend(asset_reasons)

        for sleeve in ["style", "industry"]:
            sleeve_assets = [asset for asset in adjusted if str(meta.get(asset, {}).get("asset_type", "")) == sleeve]
            sleeve_weights = {asset: adjusted[asset] for asset in sleeve_assets}
            mult, sleeve_vol, sleeve_target = sleeve_multiplier(sleeve, state, sleeve_weights, returns_wide, signal_date, cfg)
            for asset in sleeve_assets:
                adjusted[asset] *= mult
                multipliers[asset] *= mult
                if mult < 0.999:
                    reasons[asset].append(f"{sleeve}_vol_scale")
            decisions.append(
                {
                    "signal_date": signal_date,
                    "version": version,
                    "state": state,
                    "scope": f"{sleeve}_sleeve",
                    "sleeve_multiplier": mult,
                    "sleeve_vol": sleeve_vol,
                    "sleeve_target_vol": sleeve_target,
                    "reentry_active": reentry,
                }
            )

        market_mult, market_reason, market_mdd = market_brake_multiplier(broad_row, state, cfg, reentry)
        for asset in list(adjusted):
            adjusted[asset] *= market_mult
            multipliers[asset] *= market_mult
            if market_mult < 0.999:
                reasons[asset].append(market_reason)

        adjusted, cap_mult, gross = scale_to_caps(adjusted, state, cfg, reentry)
        if cap_mult < 0.999:
            for asset in adjusted:
                multipliers[asset] *= cap_mult
                reasons[asset].append("state_gross_cap")
        cash = max(0.0, 1.0 - sum(adjusted.values()))

        for _, row in group.iterrows():
            asset = str(row["asset"])
            if asset == "CASH":
                continue
            weight = adjusted.get(asset, 0.0)
            if weight <= 1e-10:
                continue
            out = row.to_dict()
            out["version"] = version
            out["base_weight"] = float(row["weight"])
            out["weight"] = weight
            out["overlay_multiplier"] = multipliers.get(asset, 0.0)
            out["overlay_reason"] = "|".join(reasons.get(asset, [])) or "unchanged"
            out["reentry_active"] = reentry
            out["market_drawdown_120"] = market_mdd
            rows.append(out)
        template = group[group["asset"].eq("CASH")].head(1)
        cash_row = template.iloc[0].to_dict() if not template.empty else {
            "signal_date": signal_date,
            "asset": "CASH",
            "state": state,
            "asset_type": "cash",
            "score": 0.0,
            "risk_adjusted_alpha": 0.0,
            "turnover": 0.0,
        }
        cash_row["version"] = version
        cash_row["base_weight"] = float(base_weights.get("CASH", 0.0))
        cash_row["weight"] = cash
        cash_row["overlay_multiplier"] = np.nan
        cash_row["overlay_reason"] = "cash_residual"
        cash_row["reentry_active"] = reentry
        cash_row["market_drawdown_120"] = market_mdd
        rows.append(cash_row)
        decisions.append(
            {
                "signal_date": signal_date,
                "version": version,
                "state": state,
                "scope": "portfolio",
                "base_gross_exposure": sum(noncash.values()),
                "final_gross_exposure": gross,
                "base_cash_weight": float(base_weights.get("CASH", 0.0)),
                "final_cash_weight": cash,
                "market_multiplier": market_mult,
                "market_reason": market_reason,
                "market_drawdown_120": market_mdd,
                "cap_multiplier": cap_mult,
                "reentry_active": reentry,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["signal_date"] = pd.to_datetime(out["signal_date"])
        out = out.sort_values(["signal_date", "asset"]).reset_index(drop=True)
    return out, pd.DataFrame(decisions)


def blend_targets(base_targets: pd.DataFrame, overlay_targets: pd.DataFrame, cfg: dict, version: str) -> pd.DataFrame:
    blend = cfg.get("blend", {})
    risk_weight = float(blend.get("risk_overlay_weight", 0.65))
    base_weight = float(blend.get("base_v2_4_weight", 1.0 - risk_weight))
    frames = []
    base = base_targets.copy()
    risk = overlay_targets.copy()
    base["signal_date"] = pd.to_datetime(base["signal_date"])
    risk["signal_date"] = pd.to_datetime(risk["signal_date"])
    dates = sorted(set(base["signal_date"].dropna()) | set(risk["signal_date"].dropna()))
    for signal_date in dates:
        b = base[base["signal_date"].eq(signal_date)].set_index("asset")
        r = risk[risk["signal_date"].eq(signal_date)].set_index("asset")
        assets = sorted(set(b.index) | set(r.index))
        state = str(r["state"].dropna().iloc[0]) if not r.empty and r["state"].notna().any() else (str(b["state"].dropna().iloc[0]) if not b.empty and b["state"].notna().any() else "range_bound")
        noncash_weights = {}
        for asset in assets:
            if asset == "CASH":
                continue
            bw = float(b.loc[asset, "weight"]) if asset in b.index else 0.0
            rw = float(r.loc[asset, "weight"]) if asset in r.index else 0.0
            weight = base_weight * bw + risk_weight * rw
            if weight > 1e-10:
                noncash_weights[asset] = weight
        total = sum(noncash_weights.values())
        if total > 1.0:
            noncash_weights = {asset: weight / total for asset, weight in noncash_weights.items()}
            total = 1.0
        cash = max(0.0, 1.0 - total)
        for asset, weight in noncash_weights.items():
            template = r.loc[asset].to_dict() if asset in r.index else b.loc[asset].to_dict()
            template["version"] = version
            template["signal_date"] = signal_date
            template["asset"] = asset
            template["weight"] = weight
            template["base_weight"] = float(b.loc[asset, "weight"]) if asset in b.index else 0.0
            template["overlay_reason"] = "fixed_v2_4_v2_8_blend"
            frames.append(template)
        cash_template = r.loc["CASH"].to_dict() if "CASH" in r.index else (b.loc["CASH"].to_dict() if "CASH" in b.index else {"asset": "CASH", "asset_type": "cash"})
        cash_template["version"] = version
        cash_template["signal_date"] = signal_date
        cash_template["asset"] = "CASH"
        cash_template["asset_type"] = "cash"
        cash_template["state"] = state
        cash_template["weight"] = cash
        cash_template["base_weight"] = float(b.loc["CASH", "weight"]) if "CASH" in b.index else 0.0
        cash_template["overlay_reason"] = "blend_cash_residual"
        frames.append(cash_template)
    out = pd.DataFrame(frames)
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.sort_values(["signal_date", "asset"]).reset_index(drop=True)


def rolling_drawdown(nav_history: list[float], current_nav: float, lookback_days: int) -> float:
    recent = nav_history[-max(1, lookback_days) :] + [float(current_nav)]
    peak = max(recent) if recent else float(current_nav)
    return float(current_nav / peak - 1.0) if peak > 0 else 0.0


def portfolio_brake_multiplier(drawdown: float, state: str, cfg: dict, reentry_active: bool) -> tuple[float, str]:
    brake = cfg.get("portfolio_drawdown_brake", {})
    if not brake.get("enabled", True):
        return 1.0, "portfolio_brake_disabled"
    if drawdown <= float(brake.get("hard_trigger", -0.28)):
        mult = float(brake.get("hard_multiplier", 0.88))
        reason = "portfolio_hard_drawdown"
    elif drawdown <= float(brake.get("soft_trigger", -0.18)):
        mult = float(brake.get("soft_multiplier", 0.94))
        reason = "portfolio_soft_drawdown"
    else:
        mult = 1.0
        reason = "portfolio_drawdown_clear"
    if state == "crash_rebound":
        mult = max(mult, float(brake.get("crash_rebound_min_multiplier", 0.92)))
    if reentry_active:
        floor = float(cfg.get("reentry_rules", {}).get("portfolio_brake_min_multiplier", mult))
        mult = max(mult, floor)
        reason = f"{reason}|reentry_floor"
    return mult, reason


def apply_cash_scale(weights: dict[str, float], multiplier: float) -> dict[str, float]:
    noncash = {asset: max(0.0, weight) * multiplier for asset, weight in weights.items() if asset != "CASH" and weight > 0}
    total = sum(noncash.values())
    if total > 1.0:
        noncash = {asset: weight / total for asset, weight in noncash.items()}
        total = 1.0
    noncash["CASH"] = max(0.0, 1.0 - total)
    return noncash


def run_backtest_dynamic(model, panel: dict, targets: pd.DataFrame, cost_bps: float, cfg: dict) -> dict[str, pd.DataFrame]:
    ret = panel["returns"].copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret_wide = ret.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    schedule = model.build_trade_schedule(targets, ret_wide.index.to_series())
    if not schedule.empty:
        ret_wide = ret_wide[ret_wide.index >= pd.to_datetime(schedule["trade_date"]).min()]
    schedule_map = {date: group for date, group in schedule.groupby("trade_date")} if not schedule.empty else {}
    current_weights = {"CASH": 1.0}
    nav = 1.0
    peak_nav = 1.0
    nav_history: list[float] = []
    rows = []
    trades = []
    brake_rows = []
    lookback = int(cfg.get("portfolio_drawdown_brake", {}).get("lookback_days", 252))

    for date, day_returns in ret_wide.iterrows():
        gross_ret = sum(weight * float(day_returns.get(asset, 0.0)) for asset, weight in current_weights.items() if asset != "CASH")
        nav_before_trade = nav * (1.0 + gross_ret)
        dd = rolling_drawdown(nav_history, nav_before_trade, lookback)
        inception_dd = nav_before_trade / peak_nav - 1.0 if peak_nav > 0 else 0.0
        cost = 0.0
        turnover = 0.0
        brake_mult = 1.0
        brake_reason = "no_rebalance"
        if date in schedule_map:
            group = schedule_map[date]
            state = str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else "range_bound"
            reentry = bool(group["reentry_active"].fillna(False).astype(bool).any()) if "reentry_active" in group.columns else False
            scheduled = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
            brake_mult, brake_reason = portfolio_brake_multiplier(dd, state, cfg, reentry)
            new_weights = apply_cash_scale(scheduled, brake_mult)
            turnover = sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in set(new_weights) | set(current_weights))
            cost = turnover * cost_bps / 10000.0
            for asset, weight in new_weights.items():
                trades.append({"date": date, "asset": asset, "weight": weight, "cost_bps": cost_bps, "turnover": turnover, "drawdown_brake_multiplier": brake_mult, "drawdown_brake_reason": brake_reason})
            brake_rows.append(
                {
                    "date": date,
                    "state": state,
                    "portfolio_drawdown_before_trade": dd,
                    "inception_drawdown_before_trade": inception_dd,
                    "drawdown_brake_multiplier": brake_mult,
                    "drawdown_brake_reason": brake_reason,
                    "scheduled_gross_exposure": 1.0 - scheduled.get("CASH", 0.0),
                    "executed_gross_exposure": 1.0 - new_weights.get("CASH", 0.0),
                    "turnover": turnover,
                    "cost": cost,
                    "reentry_active": reentry,
                }
            )
            current_weights = new_weights
        net_ret = gross_ret - cost
        nav *= 1.0 + net_ret
        peak_nav = max(peak_nav, nav)
        nav_history.append(nav)
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
                "benchmark_return": float(day_returns.get(panel["broad_code"], 0.0)),
                "portfolio_drawdown_before_trade": dd,
                "inception_drawdown_before_trade": inception_dd,
                "drawdown_brake_multiplier": brake_mult,
                "drawdown_brake_reason": brake_reason,
            }
        )
    nav_df = pd.DataFrame(rows)
    if not nav_df.empty:
        nav_df["benchmark_nav"] = (1.0 + nav_df["benchmark_return"]).cumprod()
    return {"nav": nav_df, "trades": pd.DataFrame(trades), "schedule": schedule, "brake_log": pd.DataFrame(brake_rows)}


def run_costs(model, panel: dict, targets: pd.DataFrame, cfg: dict, version: str, output_dir: Path) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    summaries = []
    nav_by_cost: dict[float, pd.DataFrame] = {}
    brake_by_cost: dict[float, pd.DataFrame] = {}
    for cost in [float(item) for item in cfg.get("cost_bps_scenarios", [5, 10, 20, 30])]:
        bt = run_backtest_dynamic(model, panel, targets, cost, cfg)
        summary = model.summarize_nav(bt["nav"])
        if not summary.empty:
            summary.insert(0, "variant", version)
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
        suffix = f"{version.lower()}_{int(cost)}bps"
        model.write_csv(bt["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(bt["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(bt["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        model.write_csv(bt["brake_log"], output_dir / f"drawdown_brake_log_{suffix}.csv")
        nav_by_cost[cost] = bt["nav"]
        brake_by_cost[cost] = bt["brake_log"]
    return pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame(), nav_by_cost, brake_by_cost


def run_base_costs(model, panel: dict, targets: pd.DataFrame, costs: list[float], output_dir: Path) -> pd.DataFrame:
    rows = []
    for cost in costs:
        bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
        summary = model.summarize_nav(bt["nav"])
        if not summary.empty:
            summary.insert(0, "variant", "v2_4_stable_nested_reference")
            summary.insert(1, "cost_bps", float(cost))
            summary["target_rows"] = int(targets.shape[0])
            summary["oos_start"] = bt["nav"]["date"].min() if not bt["nav"].empty else pd.NaT
            summary["oos_end"] = bt["nav"]["date"].max() if not bt["nav"].empty else pd.NaT
            summary["oos_years"] = (pd.to_datetime(bt["nav"]["date"].max()) - pd.to_datetime(bt["nav"]["date"].min())).days / 365.25 if not bt["nav"].empty else np.nan
            rows.append(summary)
        suffix = f"v24_reference_{int(cost)}bps"
        model.write_csv(bt["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def compare_to_base(summary: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or base.empty:
        return pd.DataFrame()
    out = summary.merge(base.add_prefix("base_"), left_on="cost_bps", right_on="base_cost_bps", how="left")
    for col in ["annual_return", "sharpe_no_rf", "max_drawdown", "avg_cash_weight", "avg_trade_turnover", "annual_vol", "total_cost"]:
        if col in out.columns and f"base_{col}" in out.columns:
            out[f"delta_{col}_vs_v2_4"] = out[col] - out[f"base_{col}"]
    return out


def period_sleeve_attribution(model, panel: dict, targets: pd.DataFrame, version: str) -> pd.DataFrame:
    returns = panel["returns"].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    ret_wide = returns.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    schedule = model.build_trade_schedule(targets, ret_wide.index.to_series())
    if schedule.empty:
        return pd.DataFrame()
    dates = sorted(pd.to_datetime(schedule["trade_date"]).unique())
    rows = []
    for idx, trade_date in enumerate(dates):
        next_date = dates[idx + 1] if idx + 1 < len(dates) else ret_wide.index.max() + pd.Timedelta(days=1)
        period = ret_wide[(ret_wide.index >= trade_date) & (ret_wide.index < next_date)]
        if period.empty:
            continue
        group = schedule[pd.to_datetime(schedule["trade_date"]).eq(trade_date)]
        for sleeve, sleeve_group in group.groupby("asset_type"):
            contribution = 0.0
            gross = 0.0
            for _, row in sleeve_group.iterrows():
                asset = str(row["asset"])
                if asset == "CASH" or asset not in period.columns:
                    continue
                weight = float(row["weight"])
                asset_return = float((1.0 + period[asset]).prod() - 1.0)
                contribution += weight * asset_return
                gross += weight
            rows.append(
                {
                    "version": version,
                    "trade_date": trade_date,
                    "next_trade_date": next_date,
                    "state": str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else "",
                    "asset_type": sleeve,
                    "gross_weight": gross,
                    "period_contribution": contribution,
                    "days": int(period.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def make_self_check(smoke: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, attribution: pd.DataFrame, output_dir: Path, cfg: dict) -> pd.DataFrame:
    thresholds = cfg.get("self_check_thresholds", {})
    row10 = comparison[comparison["cost_bps"].astype(float).eq(10.0)].head(1) if not comparison.empty else pd.DataFrame()
    row20 = summary[summary["cost_bps"].astype(float).eq(20.0)].head(1) if not summary.empty else pd.DataFrame()
    rows = [
        {"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""},
        {"check": "required_cost_rows", "pass": bool(set([5.0, 10.0, 20.0, 30.0]).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""},
        {"check": "annual_return_slippage_within_limit_10bps", "pass": bool(not row10.empty and float(row10["delta_annual_return_vs_v2_4"].iloc[0]) >= -float(thresholds.get("max_annual_return_slippage_vs_v2_4_10bps", 0.02))), "detail": f"{float(row10['delta_annual_return_vs_v2_4'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "drawdown_improvement_10bps", "pass": bool(not row10.empty and float(row10["delta_max_drawdown_vs_v2_4"].iloc[0]) >= float(thresholds.get("min_drawdown_improvement_vs_v2_4_10bps", 0.05))), "detail": f"{float(row10['delta_max_drawdown_vs_v2_4'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "sharpe_not_much_worse_10bps", "pass": bool(not row10.empty and float(row10["delta_sharpe_no_rf_vs_v2_4"].iloc[0]) >= float(thresholds.get("min_delta_sharpe_vs_v2_4_10bps", -0.01))), "detail": f"{float(row10['delta_sharpe_no_rf_vs_v2_4'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "avg_cash_below_limit_10bps", "pass": bool(not row10.empty and float(row10["avg_cash_weight"].iloc[0]) <= float(thresholds.get("max_cash_weight", 0.4))), "detail": f"{float(row10['avg_cash_weight'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "cost_20bps_annual_return_floor", "pass": bool(not row20.empty and float(row20["annual_return"].iloc[0]) >= float(thresholds.get("min_20bps_annual_return", 0.0))), "detail": f"{float(row20['annual_return'].iloc[0]):.6f}" if not row20.empty else ""},
        {"check": "attribution_non_empty", "pass": bool(not attribution.empty), "detail": str(attribution.shape[0]) if not attribution.empty else ""},
    ]
    for name in ["WALK_FORWARD_REPORT.md", "ATTRIBUTION_REPORT.md", "MODEL_CHANGELOG.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def make_reports(
    output_dir: Path,
    version: str,
    summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    decisions: pd.DataFrame,
    brake_log: pd.DataFrame,
    attribution: pd.DataFrame,
    smoke: pd.DataFrame,
    self_check: pd.DataFrame,
) -> None:
    trigger_rate = float((brake_log["drawdown_brake_multiplier"] < 0.999).mean()) if not brake_log.empty else 0.0
    attr_state = (
        attribution.groupby(["state", "asset_type"], dropna=False)
        .agg(periods=("period_contribution", "count"), total_contribution=("period_contribution", "sum"), avg_gross_weight=("gross_weight", "mean"))
        .reset_index()
        if not attribution.empty
        else pd.DataFrame()
    )
    lines = [
        f"# HIRSSM {version} Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Performance",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary.",
        "",
        "## V2.4 Reference",
        "",
        base_summary.to_markdown(index=False) if not base_summary.empty else "No base summary.",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison.",
        "",
        "## Brake Diagnostics",
        "",
        f"- Rebalance brake trigger rate: {trigger_rate:.2%}",
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
    attr_lines = [
        f"# HIRSSM {version} Attribution Report",
        "",
        "## State x Sleeve Attribution",
        "",
        attr_state.to_markdown(index=False) if not attr_state.empty else "No attribution.",
        "",
        "## Overlay Decisions",
        "",
        decisions.head(300).to_markdown(index=False) if not decisions.empty else "No decisions.",
    ]
    (output_dir / "ATTRIBUTION_REPORT.md").write_text("\n".join(attr_lines), encoding="utf-8")
    changelog = [
        f"# HIRSSM {version} Model Changelog",
        "",
        f"Run time: {now_text()}",
        "",
        "## Implemented Changes",
        "",
        "- Evaluated through the V2.6-V2.9 risk iteration engine.",
        "- No new alpha factor is promoted by this script.",
        "- Version must pass self-check before being considered a candidate.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        f"# HIRSSM {version} Self Check\n\n" + (self_check.to_markdown(index=False) if not self_check.empty else "No self check."),
        encoding="utf-8",
    )


def run_one_version(model, wf, panel: dict, base_targets: pd.DataFrame, base_summary: pd.DataFrame, config: dict, key: str, output_root: Path, prior_targets: dict[str, pd.DataFrame]) -> dict:
    cfg = resolved_version_config(config, key)
    version = str(cfg.get("version", key.upper()))
    output_dir = output_root / version.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("blend", {}).get("enabled", False):
        source = prior_targets.get("HIRSSM_V2_8")
        if source is None:
            source, _ = overlay_targets_local(base_targets, panel, resolved_version_config(config, "portfolio_risk_overlay_v2_8"), "HIRSSM_V2_8")
        targets = blend_targets(base_targets, source, cfg, version)
        decisions = pd.DataFrame([{"version": version, "scope": "portfolio", "decision": "fixed_blend", **cfg.get("blend", {})}])
    else:
        targets, decisions = overlay_targets_local(base_targets, panel, cfg, version)
    summary, nav_by_cost, brake_by_cost = run_costs(model, panel, targets, cfg, version, output_dir)
    comparison = compare_to_base(summary, base_summary)
    smoke = wf.smoke_test_targets(targets)
    reference_cost = float(cfg.get("reference_cost_bps", 10))
    brake_log = brake_by_cost.get(reference_cost, pd.DataFrame())
    attribution = period_sleeve_attribution(model, panel, targets, version)
    self_check = make_self_check(smoke, summary, comparison, attribution, output_dir, cfg)

    model.write_csv(targets, output_dir / "walk_forward_target_weights.csv")
    model.write_csv(decisions, output_dir / "risk_overlay_decision_log.csv")
    model.write_csv(summary, output_dir / "oos_performance.csv")
    model.write_csv(base_summary, output_dir / "base_v2_4_oos_performance.csv")
    model.write_csv(comparison, output_dir / f"{version.lower()}_vs_v2_4_comparison.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")
    model.write_csv(attribution, output_dir / "period_sleeve_attribution.csv")
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    make_reports(output_dir, version, summary, base_summary, comparison, decisions, brake_log, attribution, smoke, self_check)
    self_check = make_self_check(smoke, summary, comparison, attribution, output_dir, cfg)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    (output_dir / "SELF_CHECK_REPORT.md").write_text(
        f"# HIRSSM {version} Self Check\n\n" + self_check.to_markdown(index=False),
        encoding="utf-8",
    )
    return {
        "version": version,
        "output_dir": output_dir,
        "targets": targets,
        "summary": summary,
        "comparison": comparison,
        "self_check": self_check,
        "self_check_pass": bool(self_check["pass"].all()) if not self_check.empty else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v21 = load_module("hirssm_v2_1_walk_forward", V21_PATH)
    v22 = load_module("hirssm_v2_2_walk_forward", V22_PATH)
    v23 = load_module("hirssm_v2_3_nested_walk_forward", V23_PATH)
    v24 = load_module("hirssm_v2_4_stable_nested_selection", V24_PATH)
    v25 = load_module("hirssm_v2_5_portfolio_risk_overlay", V25_PATH)

    root = Path(args.root)
    output_root = Path(args.output_root)
    config = model.read_json(Path(args.config))
    context = v25.build_v24_context(model, wf, v21, v22, v23, v24, root, config, args.start_date, args.end_date)
    panel = context["panel"]
    base_targets = context["base_targets"]
    costs = [5.0, 10.0, 20.0, 30.0]
    base_summary = run_base_costs(model, panel, base_targets, costs, output_root / "hirssm_v2_6_to_v2_9_reference")
    prior_targets: dict[str, pd.DataFrame] = {}
    results = []
    for key in VERSION_KEYS:
        result = run_one_version(model, wf, panel, base_targets, base_summary, config, key, output_root, prior_targets)
        prior_targets[result["version"]] = result["targets"]
        results.append(result)

    summary_rows = []
    for result in results:
        row10 = result["summary"][result["summary"]["cost_bps"].astype(float).eq(10.0)].head(1)
        comp10 = result["comparison"][result["comparison"]["cost_bps"].astype(float).eq(10.0)].head(1)
        summary_rows.append(
            {
                "version": result["version"],
                "self_check_pass": result["self_check_pass"],
                "annual_return_10bps": float(row10["annual_return"].iloc[0]) if not row10.empty else np.nan,
                "sharpe_10bps": float(row10["sharpe_no_rf"].iloc[0]) if not row10.empty else np.nan,
                "max_drawdown_10bps": float(row10["max_drawdown"].iloc[0]) if not row10.empty else np.nan,
                "avg_cash_10bps": float(row10["avg_cash_weight"].iloc[0]) if not row10.empty else np.nan,
                "delta_annual_return_vs_v2_4": float(comp10["delta_annual_return_vs_v2_4"].iloc[0]) if not comp10.empty else np.nan,
                "delta_sharpe_vs_v2_4": float(comp10["delta_sharpe_no_rf_vs_v2_4"].iloc[0]) if not comp10.empty else np.nan,
                "delta_mdd_vs_v2_4": float(comp10["delta_max_drawdown_vs_v2_4"].iloc[0]) if not comp10.empty else np.nan,
                "output_dir": str(result["output_dir"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    iteration_dir = output_root / "hirssm_v2_6_to_v2_9_iteration"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    model.write_csv(summary, iteration_dir / "iteration_summary.csv")
    (iteration_dir / "ITERATION_REPORT.md").write_text(
        "# HIRSSM V2.6-V2.9 Iteration Report\n\n"
        + f"Run time: {now_text()}\n\n"
        + summary.to_markdown(index=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "iteration_dir": str(iteration_dir.resolve()),
                "versions": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
