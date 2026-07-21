"""Market state monitor for the accepted daily-only feature layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyOnlyStateConfig:
    trailing_window: int = 252
    min_history: int = 60
    stress_pctile: float = 0.2
    recovery_pctile: float = 0.8
    crowding_pctile: float = 0.85
    calm_pctile: float = 0.15
    min_active_asset_ratio: float = 0.95
    breadth_stress_advance_ratio: float = 0.35
    breadth_recovery_advance_ratio: float = 0.55
    limit_crowding_pctile: float = 0.9
    minimum_limit_share: float = 0.003


def trailing_percentile(values: pd.Series, window: int, min_history: int) -> pd.Series:
    """Point-in-time percentile of current value versus prior observations only."""

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    out = np.full(len(arr), np.nan, dtype="float64")
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        start = max(0, idx - window)
        history = arr[start:idx]
        history = history[np.isfinite(history)]
        if len(history) < min_history:
            continue
        out[idx] = float((history <= value).mean())
    return pd.Series(out, index=values.index)


def enrich_state_inputs(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "asset_count",
        "active_asset_ratio",
        "sum_amount_raw",
        "sum_volume_raw",
        "top_amount_share",
        "advance_decline_balance",
        "advance_ratio",
        "decline_ratio",
        "limit_like_up_count",
        "limit_like_down_count",
        "median_intraday_strength_raw",
        "median_range_ratio_raw",
        "low_amount_count",
        "low_volume_count",
        "bad_ohlc_after_adapter",
        "data_scope",
        "price_adjustment",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing feature columns for state monitor: {sorted(missing)}")

    out = features.sort_values("trade_date").copy()
    asset_count = pd.to_numeric(out["asset_count"], errors="coerce").replace(0, np.nan)
    out["amount_per_asset_raw"] = pd.to_numeric(out["sum_amount_raw"], errors="coerce") / asset_count
    out["volume_per_asset_raw"] = pd.to_numeric(out["sum_volume_raw"], errors="coerce") / asset_count
    out["limit_up_share"] = pd.to_numeric(out["limit_like_up_count"], errors="coerce") / asset_count
    out["limit_down_share"] = pd.to_numeric(out["limit_like_down_count"], errors="coerce") / asset_count
    out["low_amount_share"] = pd.to_numeric(out["low_amount_count"], errors="coerce") / asset_count
    out["low_volume_share"] = pd.to_numeric(out["low_volume_count"], errors="coerce") / asset_count
    out["feature_scope"] = "daily_only_market_state_monitor"
    out["state_basis"] = "trailing_window_prior_observations_only"
    return out


def add_trailing_percentiles(data: pd.DataFrame, config: DailyOnlyStateConfig) -> pd.DataFrame:
    out = data.copy()
    percentile_features = [
        "active_asset_ratio",
        "amount_per_asset_raw",
        "volume_per_asset_raw",
        "top_amount_share",
        "advance_decline_balance",
        "median_intraday_strength_raw",
        "median_range_ratio_raw",
        "limit_up_share",
        "limit_down_share",
        "low_amount_share",
    ]
    for feature in percentile_features:
        out[f"{feature}_trailing_pctile"] = trailing_percentile(
            out[feature],
            window=config.trailing_window,
            min_history=config.min_history,
        )
    out["history_available"] = out["amount_per_asset_raw_trailing_pctile"].notna()
    return out


def classify_liquidity(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    if row["active_asset_ratio"] < config.min_active_asset_ratio:
        return "liquidity_quality_alert"
    if (
        row["amount_per_asset_raw_trailing_pctile"] <= config.stress_pctile
        or row["volume_per_asset_raw_trailing_pctile"] <= config.stress_pctile
        or row["low_amount_share_trailing_pctile"] >= config.recovery_pctile
    ):
        return "liquidity_dry_up"
    if row["amount_per_asset_raw_trailing_pctile"] >= config.recovery_pctile:
        return "liquidity_expansion"
    return "liquidity_neutral"


def classify_breadth(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    if (
        row["advance_decline_balance_trailing_pctile"] <= config.stress_pctile
        and row["advance_ratio"] <= config.breadth_stress_advance_ratio
    ):
        return "breadth_stress"
    if (
        row["advance_decline_balance_trailing_pctile"] >= config.recovery_pctile
        and row["advance_ratio"] >= config.breadth_recovery_advance_ratio
    ):
        return "breadth_recovery"
    return "breadth_neutral"


def classify_activity(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    if row["median_range_ratio_raw_trailing_pctile"] >= config.crowding_pctile:
        return "high_range_activity"
    if row["median_range_ratio_raw_trailing_pctile"] <= config.calm_pctile:
        return "low_range_activity"
    return "normal_range_activity"


def classify_concentration(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    if row["top_amount_share_trailing_pctile"] >= config.crowding_pctile:
        return "turnover_concentrated"
    if row["top_amount_share_trailing_pctile"] <= config.calm_pctile:
        return "turnover_diffuse"
    return "turnover_normal"


def classify_limit_crowding(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    up = (
        row["limit_up_share_trailing_pctile"] >= config.limit_crowding_pctile
        and row["limit_up_share"] >= config.minimum_limit_share
    )
    down = (
        row["limit_down_share_trailing_pctile"] >= config.limit_crowding_pctile
        and row["limit_down_share"] >= config.minimum_limit_share
    )
    if up and down:
        return "two_sided_limit_crowding"
    if down:
        return "down_limit_crowding"
    if up:
        return "up_limit_crowding"
    return "no_limit_crowding"


def classify_data_quality(row: pd.Series, config: DailyOnlyStateConfig) -> str:
    if row["data_scope"] != "accepted_processed_daily_only":
        return "wrong_data_scope"
    if row["price_adjustment"] != "none_raw":
        return "wrong_price_adjustment"
    if int(row["bad_ohlc_after_adapter"]) != 0:
        return "bad_ohlc_after_adapter"
    if row["active_asset_ratio"] < config.min_active_asset_ratio:
        return "low_active_asset_ratio"
    return "quality_ok"


def classify_composite(row: pd.Series) -> str:
    if not bool(row["history_available"]):
        return "insufficient_history"
    quality = row["data_quality_state"]
    if quality != "quality_ok":
        return "data_quality_alert"
    liquidity = row["liquidity_state"]
    breadth = row["breadth_state"]
    activity = row["activity_state"]
    concentration = row["concentration_state"]
    limit_state = row["limit_crowding_state"]
    if breadth == "breadth_stress" and liquidity in {"liquidity_dry_up", "liquidity_quality_alert"}:
        return "stress_liquidity_breadth"
    if breadth == "breadth_stress" and limit_state in {"down_limit_crowding", "two_sided_limit_crowding"}:
        return "stress_limit_breadth"
    if breadth == "breadth_recovery" and liquidity == "liquidity_expansion":
        return "risk_on_breadth_liquidity"
    if activity == "high_range_activity" and concentration == "turnover_concentrated":
        return "crowded_high_activity"
    if liquidity == "liquidity_dry_up":
        return "liquidity_soft_patch"
    if breadth == "breadth_recovery":
        return "breadth_recovery_only"
    if breadth == "breadth_stress":
        return "breadth_stress_only"
    return "neutral_diagnostic"


def build_state_panel(features: pd.DataFrame, config: DailyOnlyStateConfig) -> pd.DataFrame:
    out = enrich_state_inputs(features)
    out = add_trailing_percentiles(out, config)
    out["liquidity_state"] = out.apply(classify_liquidity, axis=1, config=config)
    out["breadth_state"] = out.apply(classify_breadth, axis=1, config=config)
    out["activity_state"] = out.apply(classify_activity, axis=1, config=config)
    out["concentration_state"] = out.apply(classify_concentration, axis=1, config=config)
    out["limit_crowding_state"] = out.apply(classify_limit_crowding, axis=1, config=config)
    out["data_quality_state"] = out.apply(classify_data_quality, axis=1, config=config)
    out["composite_state"] = out.apply(classify_composite, axis=1)
    out["model_usage_allowed"] = False
    out["backtest_usage_allowed"] = False
    return out


STATE_DICTIONARY_ROWS = [
    {
        "state": "liquidity_state",
        "definition": "Trailing-window raw amount, volume, active ratio, and low-amount diagnostics.",
        "allowed_use": "market condition monitoring and future research stratification",
        "forbidden_use": "not a trading signal or return forecast by itself",
    },
    {
        "state": "breadth_state",
        "definition": "Trailing-window advance/decline balance plus current advance ratio.",
        "allowed_use": "market breadth regime diagnostics",
        "forbidden_use": "not an adjusted market index return",
    },
    {
        "state": "activity_state",
        "definition": "Trailing-window percentile of median raw intraday range ratio.",
        "allowed_use": "raw activity and volatility-pressure diagnostics",
        "forbidden_use": "not split/dividend-adjusted volatility",
    },
    {
        "state": "concentration_state",
        "definition": "Trailing-window percentile of top amount share.",
        "allowed_use": "turnover concentration diagnostics",
        "forbidden_use": "not a capacity or crowding estimate without constituents and float data",
    },
    {
        "state": "limit_crowding_state",
        "definition": "Trailing-window percentile of limit-like up/down raw pct_chg shares.",
        "allowed_use": "extreme same-day breadth pressure diagnostics",
        "forbidden_use": "not exchange-rule exact limit classification",
    },
    {
        "state": "composite_state",
        "definition": "Rule-based summary of quality, liquidity, breadth, activity, concentration, and limit-crowding states.",
        "allowed_use": "diagnostic segmentation and research queueing",
        "forbidden_use": "not model output, portfolio allocation, or backtest decision",
    },
]
