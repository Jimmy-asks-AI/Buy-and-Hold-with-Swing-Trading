"""Market participation breadth feature layer for HIRSSM V3.67.

The feature layer is built from the accepted daily-only raw OHLCV scope. It
uses raw daily stock data only for cross-sectional participation diagnostics:
advancer/decliner shares, activity breadth, amount concentration, and limit-like
pressure. It does not create stock return labels, adjusted returns, portfolio
returns, or model promotion evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class MarketParticipationBreadthConfig:
    v3_66_manifest_path: Path
    v3_66_queue_path: Path
    v3_66_inventory_path: Path
    v3_44_guard_manifest_path: Path
    v3_44_capability_matrix_path: Path
    v3_47_manifest_path: Path
    v3_47_feature_panel_path: Path
    v3_48_manifest_path: Path
    v3_48_state_panel_path: Path
    raw_partition_dir: Path
    output_dir: Path
    catalog_path: Path
    min_history: int
    trailing_window: int
    breadth_short_window: int
    breadth_medium_window: int
    breadth_long_window: int
    min_feature_rows: int
    min_latest_asset_count: int


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_ratio(num: pd.Series | float, den: pd.Series | float) -> pd.Series | float:
    if isinstance(num, pd.Series) or isinstance(den, pd.Series):
        n = pd.to_numeric(num, errors="coerce")
        d = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
        return n / d
    if den == 0 or pd.isna(den):
        return np.nan
    return float(num) / float(den)


def _trailing_percentile(values: pd.Series, window: int, min_history: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    out = np.full(len(arr), np.nan)
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


def _rolling_zscore(values: pd.Series, window: int, min_history: int) -> pd.Series:
    data = pd.to_numeric(values, errors="coerce")
    mean = data.rolling(window, min_periods=min_history).mean()
    std = data.rolling(window, min_periods=min_history).std(ddof=0).replace(0, np.nan)
    return (data - mean) / std


def _clip_score(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(-3, 3) / 3.0


def _score_component(values: pd.Series) -> pd.Series:
    return _clip_score(values).fillna(0.0)


def normalize_trade_date(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str[:8]
    )


def validate_inputs(
    config: MarketParticipationBreadthConfig,
    v3_66_manifest: dict[str, Any],
    v3_44_manifest: dict[str, Any],
    v3_47_manifest: dict[str, Any],
    v3_48_manifest: dict[str, Any],
    v3_66_queue: pd.DataFrame,
    capability_matrix: pd.DataFrame,
    source_inventory: pd.DataFrame,
) -> pd.DataFrame:
    matrix = capability_matrix.copy()
    allowed_breadth = False
    hard_blocked_return = False
    if not matrix.empty and {"research_use", "status"}.issubset(matrix.columns):
        allowed_breadth = bool(
            matrix.loc[
                matrix["research_use"].astype(str).eq("market_breadth_using_sign_or_pct_chg_with_raw_boundary")
                & matrix["status"].astype(str).eq("allowed_with_boundary")
            ].shape[0]
        )
        hard_blocked_return = bool(
            matrix.loc[
                matrix["research_use"].astype(str).isin(["adjusted_return", "total_return", "portfolio_backtest_performance"])
                & matrix["status"].astype(str).eq("hard_blocked")
            ].shape[0]
            >= 3
        )
    queue_ok = False
    if not v3_66_queue.empty and {"version_hint", "source_id"}.issubset(v3_66_queue.columns):
        queue_ok = bool(
            v3_66_queue["version_hint"].astype(str).eq("V3.67").any()
            and v3_66_queue["source_id"].astype(str).eq("tushare_daily_raw_market_partitions").any()
        )
    source_ready = False
    if not source_inventory.empty and {"source_id", "allowed_stage"}.issubset(source_inventory.columns):
        source_ready = bool(
            source_inventory.loc[
                source_inventory["source_id"].astype(str).eq("tushare_daily_raw_market_partitions")
                & source_inventory["allowed_stage"].astype(str).eq("ready_for_nonreturn_breadth_feature_layer")
            ].shape[0]
        )
    return pd.DataFrame(
        [
            {
                "check": "v3_66_manifest_self_check_passed",
                "status": _status(bool(v3_66_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_66_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_66_queue_contains_v3_67_daily_breadth",
                "status": _status(queue_ok),
                "detail": f"queue_rows={len(v3_66_queue)}",
            },
            {
                "check": "v3_66_inventory_marks_daily_partitions_ready",
                "status": _status(source_ready),
                "detail": "tushare_daily_raw_market_partitions",
            },
            {
                "check": "v3_44_guard_self_check_passed",
                "status": _status(bool(v3_44_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_44_manifest.get('self_check_pass')}",
            },
            {
                "check": "raw_breadth_use_allowed_with_boundary",
                "status": _status(allowed_breadth),
                "detail": "market_breadth_using_sign_or_pct_chg_with_raw_boundary",
            },
            {
                "check": "raw_return_uses_hard_blocked",
                "status": _status(hard_blocked_return),
                "detail": "adjusted_return,total_return,portfolio_backtest_performance",
            },
            {
                "check": "v3_47_feature_layer_self_check_passed",
                "status": _status(bool(v3_47_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_47_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_48_state_monitor_self_check_passed",
                "status": _status(bool(v3_48_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_48_manifest.get('self_check_pass')}",
            },
            {
                "check": "raw_partition_directory_exists",
                "status": _status(config.raw_partition_dir.exists()),
                "detail": str(config.raw_partition_dir),
            },
        ]
    )


def build_participation_feature_panel(
    daily_features: pd.DataFrame,
    state_panel: pd.DataFrame,
    config: MarketParticipationBreadthConfig,
) -> pd.DataFrame:
    required = {
        "trade_date",
        "asset_count",
        "active_asset_ratio",
        "sum_volume_raw",
        "sum_amount_raw",
        "top_amount_share",
        "advancer_count",
        "decliner_count",
        "flat_count",
        "advance_decline_balance",
        "advance_ratio",
        "decline_ratio",
        "limit_like_up_count",
        "limit_like_down_count",
        "low_amount_count",
        "low_volume_count",
        "median_intraday_strength_raw",
        "median_range_ratio_raw",
        "bad_ohlc_after_adapter",
        "data_scope",
        "price_adjustment",
    }
    missing = sorted(required.difference(daily_features.columns))
    if missing:
        raise ValueError(f"daily feature panel missing columns: {missing}")

    features = daily_features.copy()
    features["trade_date"] = normalize_trade_date(features["trade_date"])
    features = features.sort_values("trade_date").reset_index(drop=True)
    states = state_panel.copy()
    states["trade_date"] = normalize_trade_date(states["trade_date"])
    state_cols = [
        "trade_date",
        "history_available",
        "liquidity_state",
        "breadth_state",
        "activity_state",
        "concentration_state",
        "limit_crowding_state",
        "data_quality_state",
        "composite_state",
    ]
    states = states.loc[:, [col for col in state_cols if col in states.columns]].drop_duplicates("trade_date")
    out = features.merge(states, on="trade_date", how="left")

    asset_count = pd.to_numeric(out["asset_count"], errors="coerce").replace(0, np.nan)
    out["source_trade_date"] = out["trade_date"]
    out["available_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    next_trade_date = out["trade_date"].shift(-1)
    out["signal_use_date"] = pd.to_datetime(next_trade_date, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    out["signal_timing"] = "after_close_for_next_trade_date"
    out["amount_per_asset_raw"] = pd.to_numeric(out["sum_amount_raw"], errors="coerce") / asset_count
    out["volume_per_asset_raw"] = pd.to_numeric(out["sum_volume_raw"], errors="coerce") / asset_count
    out["low_amount_share"] = pd.to_numeric(out["low_amount_count"], errors="coerce") / asset_count
    out["low_volume_share"] = pd.to_numeric(out["low_volume_count"], errors="coerce") / asset_count
    out["limit_up_share"] = pd.to_numeric(out["limit_like_up_count"], errors="coerce") / asset_count
    out["limit_down_share"] = pd.to_numeric(out["limit_like_down_count"], errors="coerce") / asset_count
    out["limit_pressure_total"] = out["limit_up_share"] + out["limit_down_share"]
    out["limit_pressure_balance"] = out["limit_up_share"] - out["limit_down_share"]
    out["flat_ratio"] = pd.to_numeric(out["flat_count"], errors="coerce") / asset_count
    out["up_down_ratio_raw_breadth"] = (
        pd.to_numeric(out["advancer_count"], errors="coerce") / pd.to_numeric(out["decliner_count"], errors="coerce").replace(0, np.nan)
    )
    out["up_down_ratio_raw_breadth"] = out["up_down_ratio_raw_breadth"].replace([np.inf, -np.inf], np.nan)
    out["amount_per_asset_log"] = np.log1p(pd.to_numeric(out["amount_per_asset_raw"], errors="coerce").clip(lower=0))
    out["volume_per_asset_log"] = np.log1p(pd.to_numeric(out["volume_per_asset_raw"], errors="coerce").clip(lower=0))

    for col in [
        "advance_ratio",
        "decline_ratio",
        "advance_decline_balance",
        "active_asset_ratio",
        "amount_per_asset_log",
        "volume_per_asset_log",
        "top_amount_share",
        "low_amount_share",
        "limit_up_share",
        "limit_down_share",
        "limit_pressure_total",
        "limit_pressure_balance",
        "median_intraday_strength_raw",
        "median_range_ratio_raw",
    ]:
        out[f"{col}_prior_pctile_{config.trailing_window}d"] = _trailing_percentile(out[col], config.trailing_window, config.min_history)
        out[f"{col}_z_{config.trailing_window}d"] = _rolling_zscore(out[col], config.trailing_window, config.min_history)

    for window in [config.breadth_short_window, config.breadth_medium_window, config.breadth_long_window]:
        out[f"advance_ratio_ma{window}"] = pd.to_numeric(out["advance_ratio"], errors="coerce").rolling(window, min_periods=max(3, window // 3)).mean()
        out[f"decline_ratio_ma{window}"] = pd.to_numeric(out["decline_ratio"], errors="coerce").rolling(window, min_periods=max(3, window // 3)).mean()
        out[f"limit_pressure_total_ma{window}"] = out["limit_pressure_total"].rolling(window, min_periods=max(3, window // 3)).mean()

    out["breadth_thrust_5v20"] = out[f"advance_ratio_ma{config.breadth_short_window}"] - out[f"advance_ratio_ma{config.breadth_medium_window}"]
    out["breadth_thrust_20v60"] = out[f"advance_ratio_ma{config.breadth_medium_window}"] - out[f"advance_ratio_ma{config.breadth_long_window}"]
    out["liquidity_expansion_score"] = (
        _score_component(out[f"amount_per_asset_log_z_{config.trailing_window}d"])
        + _score_component(out[f"volume_per_asset_log_z_{config.trailing_window}d"])
        + _score_component(out[f"active_asset_ratio_z_{config.trailing_window}d"])
        - _score_component(out[f"low_amount_share_z_{config.trailing_window}d"])
    ) / 4.0
    out["breadth_participation_score"] = (
        _score_component(out[f"advance_decline_balance_z_{config.trailing_window}d"])
        + _score_component(out[f"advance_ratio_z_{config.trailing_window}d"])
        - _score_component(out[f"decline_ratio_z_{config.trailing_window}d"])
        + _score_component(out["breadth_thrust_20v60"] * 5.0)
    ) / 4.0
    out["crowding_risk_score"] = (
        _score_component(out[f"top_amount_share_z_{config.trailing_window}d"])
        + _score_component(out[f"limit_pressure_total_z_{config.trailing_window}d"])
        + _score_component(out[f"median_range_ratio_raw_z_{config.trailing_window}d"])
    ) / 3.0
    out["market_participation_breadth_score"] = (
        0.45 * out["breadth_participation_score"]
        + 0.35 * out["liquidity_expansion_score"]
        - 0.20 * out["crowding_risk_score"]
    )
    out["risk_off_breadth_pressure_score"] = (
        _score_component(out[f"decline_ratio_z_{config.trailing_window}d"])
        + _score_component(out[f"limit_down_share_z_{config.trailing_window}d"])
        - _score_component(out[f"advance_ratio_z_{config.trailing_window}d"])
    ) / 3.0

    out["history_sufficient"] = out[f"advance_ratio_prior_pctile_{config.trailing_window}d"].notna()
    out["raw_adjustment_guard_status"] = "pass_nonreturn_breadth_only"
    out["stock_return_label_generated"] = False
    out["adjustment_based_label_generated"] = False
    out["model_promotion_allowed"] = False
    out["portfolio_harness_allowed"] = False
    out["official_label_evidence"] = False
    out["feature_family"] = "market_participation_breadth"
    out["feature_scope"] = "daily_market_participation_breadth_nonreturn"
    out["data_scope"] = "accepted_processed_tushare_daily_only"
    out["price_adjustment"] = "none_raw"

    preferred = [
        "trade_date",
        "source_trade_date",
        "available_date",
        "signal_use_date",
        "signal_timing",
        "asset_count",
        "active_asset_ratio",
        "advance_ratio",
        "decline_ratio",
        "flat_ratio",
        "advance_decline_balance",
        "up_down_ratio_raw_breadth",
        "amount_per_asset_raw",
        "volume_per_asset_raw",
        "amount_per_asset_log",
        "volume_per_asset_log",
        "top_amount_share",
        "low_amount_share",
        "low_volume_share",
        "limit_up_share",
        "limit_down_share",
        "limit_pressure_total",
        "limit_pressure_balance",
        "median_intraday_strength_raw",
        "median_range_ratio_raw",
        "advance_ratio_ma5",
        "advance_ratio_ma20",
        "advance_ratio_ma60",
        "decline_ratio_ma5",
        "decline_ratio_ma20",
        "decline_ratio_ma60",
        "breadth_thrust_5v20",
        "breadth_thrust_20v60",
        "liquidity_expansion_score",
        "breadth_participation_score",
        "crowding_risk_score",
        "market_participation_breadth_score",
        "risk_off_breadth_pressure_score",
        "history_sufficient",
        "history_available",
        "liquidity_state",
        "breadth_state",
        "activity_state",
        "concentration_state",
        "limit_crowding_state",
        "data_quality_state",
        "composite_state",
        "raw_adjustment_guard_status",
        "stock_return_label_generated",
        "adjustment_based_label_generated",
        "model_promotion_allowed",
        "portfolio_harness_allowed",
        "official_label_evidence",
        "feature_family",
        "feature_scope",
        "data_scope",
        "price_adjustment",
    ]
    extra = [col for col in out.columns if col.endswith(f"_prior_pctile_{config.trailing_window}d") or col.endswith(f"_z_{config.trailing_window}d")]
    columns = [col for col in preferred + extra if col in out.columns]
    return out.loc[:, columns].reset_index(drop=True)


def build_raw_adjustment_guard_checks(panel: pd.DataFrame, capability_matrix: pd.DataFrame) -> pd.DataFrame:
    lower_columns = " ".join(panel.columns).lower()
    hard_blocked_terms = ["stock_forward_return", "adjusted_return", "total_return", "portfolio_return"]
    hard_hits = [term for term in hard_blocked_terms if term in lower_columns]
    stock_label_any = _bool_series(panel["stock_return_label_generated"]).any() if "stock_return_label_generated" in panel else True
    adjusted_any = _bool_series(panel["adjustment_based_label_generated"]).any() if "adjustment_based_label_generated" in panel else True
    model_any = _bool_series(panel["model_promotion_allowed"]).any() if "model_promotion_allowed" in panel else True
    portfolio_any = _bool_series(panel["portfolio_harness_allowed"]).any() if "portfolio_harness_allowed" in panel else True
    raw_status_ok = set(panel["raw_adjustment_guard_status"].astype(str).unique()) == {"pass_nonreturn_breadth_only"}
    allowed_breadth = False
    if not capability_matrix.empty and {"research_use", "status"}.issubset(capability_matrix.columns):
        allowed_breadth = bool(
            capability_matrix.loc[
                capability_matrix["research_use"].astype(str).eq("market_breadth_using_sign_or_pct_chg_with_raw_boundary")
                & capability_matrix["status"].astype(str).eq("allowed_with_boundary")
            ].shape[0]
        )
    return pd.DataFrame(
        [
            {
                "check": "raw_breadth_capability_allowed_with_boundary",
                "status": _status(allowed_breadth),
                "detail": "market_breadth_using_sign_or_pct_chg_with_raw_boundary",
            },
            {
                "check": "no_stock_return_label_generated",
                "status": _status(not stock_label_any),
                "detail": f"stock_return_label_any={bool(stock_label_any)}",
            },
            {
                "check": "no_adjustment_based_label_generated",
                "status": _status(not adjusted_any),
                "detail": f"adjustment_based_label_any={bool(adjusted_any)}",
            },
            {
                "check": "no_portfolio_or_model_promotion_flags",
                "status": _status(not model_any and not portfolio_any),
                "detail": f"model_any={bool(model_any)};portfolio_any={bool(portfolio_any)}",
            },
            {
                "check": "no_hard_blocked_output_columns",
                "status": _status(not hard_hits),
                "detail": ",".join(hard_hits),
            },
            {
                "check": "raw_guard_status_consistent",
                "status": _status(raw_status_ok),
                "detail": "|".join(sorted(panel["raw_adjustment_guard_status"].astype(str).unique())),
            },
        ]
    )


def build_feature_dictionary(config: MarketParticipationBreadthConfig) -> pd.DataFrame:
    rows = [
        {
            "feature": "advance_ratio",
            "definition": "Advancer count divided by valid raw daily pct_chg rows.",
            "allowed_use": "cross-sectional market breadth and participation diagnostics",
            "forbidden_use": "stock-level return labels or portfolio performance",
        },
        {
            "feature": "breadth_thrust_5v20",
            "definition": "5-day average advance ratio minus 20-day average advance ratio, available after close.",
            "allowed_use": "short-term participation acceleration feature",
            "forbidden_use": "same-day executable signal or tuned trading rule",
        },
        {
            "feature": "breadth_thrust_20v60",
            "definition": "20-day average advance ratio minus 60-day average advance ratio, available after close.",
            "allowed_use": "medium-term participation acceleration feature",
            "forbidden_use": "performance claim without later walk-forward validation",
        },
        {
            "feature": "amount_per_asset_log",
            "definition": "log1p of raw amount per accepted stock row.",
            "allowed_use": "market activity and liquidity expansion diagnostics",
            "forbidden_use": "precise CNY turnover or capacity estimate without provider unit review",
        },
        {
            "feature": "top_amount_share",
            "definition": "Share of raw amount contributed by the largest amount rows from V3.47.",
            "allowed_use": "turnover concentration diagnostic",
            "forbidden_use": "float-adjusted crowding or capacity estimate",
        },
        {
            "feature": "limit_pressure_total",
            "definition": "Limit-like up share plus limit-like down share based on raw same-day pct_chg thresholds.",
            "allowed_use": "extreme participation pressure diagnostic",
            "forbidden_use": "official limit-up/down event label without exchange-rule handling",
        },
        {
            "feature": "market_participation_breadth_score",
            "definition": "Composite of breadth participation, liquidity expansion, and crowding-risk features.",
            "allowed_use": "candidate feature for later signal validation only",
            "forbidden_use": "default model, portfolio harness, or performance claim in V3.67",
        },
    ]
    for row in rows:
        row["source_scope"] = "accepted_processed_tushare_daily_only"
        row["price_adjustment"] = "none_raw"
        row["timing"] = "after_close_for_next_trade_date"
        row["trailing_window"] = config.trailing_window
    return pd.DataFrame(rows)


def build_feature_contract(config: MarketParticipationBreadthConfig) -> str:
    return "\n".join(
        [
            "# V3.67 Market Participation Breadth Feature Contract",
            "",
            "## Scope",
            "",
            "- Source: accepted Tushare daily-only raw OHLCV partitions via V3.47/V3.48 lineage.",
            "- Timing: features are available after the source trade-date close and may only be used from the next trade date.",
            "- Use: cross-sectional market participation, breadth, liquidity, and crowding diagnostics.",
            "",
            "## Allowed",
            "",
            "- Count or share of advancers, decliners, flat rows, active rows, and limit-like rows.",
            "- Aggregate raw amount and volume diagnostics after daily-only acceptance checks.",
            "- Prior-observation trailing percentiles and rolling transformations.",
            "",
            "## Forbidden",
            "",
            "- Stock-level forward returns from raw close.",
            "- Adjusted return, total return, dividend return, or portfolio performance outputs.",
            "- Same-branch MARKET proxy threshold tuning.",
            "- Default model promotion before a separate walk-forward validation task.",
            "",
            "## Required Next Gate",
            "",
            "A later validation task must join these features to a permitted label source with explicit point-in-time rules. V3.67 itself is not a strategy.",
        ]
    )


def build_feature_quality_checks(
    panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    state_panel: pd.DataFrame,
    raw_guard: pd.DataFrame,
    config: MarketParticipationBreadthConfig,
) -> pd.DataFrame:
    forbidden_output_terms = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(panel.columns).lower()})
    score_cols = ["liquidity_expansion_score", "breadth_participation_score", "crowding_risk_score", "market_participation_breadth_score"]
    finite_scores = panel.loc[panel["history_sufficient"].astype(bool), score_cols].replace([np.inf, -np.inf], np.nan)
    latest = panel.sort_values("trade_date").tail(1)
    rows = [
        {
            "check": "panel_rows_match_daily_features",
            "status": _status(len(panel) == len(daily_features) == len(state_panel)),
            "detail": f"panel={len(panel)};daily={len(daily_features)};state={len(state_panel)}",
        },
        {
            "check": "minimum_feature_rows_present",
            "status": _status(len(panel) >= config.min_feature_rows),
            "detail": f"rows={len(panel)};min={config.min_feature_rows}",
        },
        {
            "check": "latest_asset_count_large_enough",
            "status": _status(not latest.empty and int(latest["asset_count"].iloc[0]) >= config.min_latest_asset_count),
            "detail": f"latest_asset_count={int(latest['asset_count'].iloc[0]) if not latest.empty else 0}",
        },
        {
            "check": "history_sufficient_after_min_history",
            "status": _status(int(panel["history_sufficient"].sum()) >= len(panel) - config.min_history - 2),
            "detail": f"history_sufficient={int(panel['history_sufficient'].sum())}",
        },
        {
            "check": "feature_timing_has_next_trade_date",
            "status": _status(panel["signal_use_date"].astype(str).str.len().gt(0).sum() >= len(panel) - 1),
            "detail": f"missing={int((~panel['signal_use_date'].astype(str).str.len().gt(0)).sum())}",
        },
        {
            "check": "score_columns_finite_when_history_sufficient",
            "status": _status(not finite_scores.empty and finite_scores.notna().all().all()),
            "detail": ",".join(score_cols),
        },
        {
            "check": "price_adjustment_is_none_raw",
            "status": _status(set(panel["price_adjustment"].astype(str).unique()) == {"none_raw"}),
            "detail": "|".join(sorted(panel["price_adjustment"].astype(str).unique())),
        },
        {
            "check": "raw_guard_checks_passed",
            "status": _status(raw_guard["status"].eq("pass").all()),
            "detail": ";".join(raw_guard.loc[raw_guard["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "promotion_columns_absent",
            "status": _status(not forbidden_output_terms),
            "detail": ",".join(forbidden_output_terms),
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "market_participation_breadth_feature_layer",
                "produced": True,
                "blocked": False,
                "reason": "V3.67 builds non-return breadth features only",
            },
            {
                "result_type": "stock_return_label",
                "produced": False,
                "blocked": True,
                "reason": "raw unadjusted daily prices cannot produce stock return labels",
            },
            {
                "result_type": "portfolio_harness",
                "produced": False,
                "blocked": True,
                "reason": "feature layer has no positions, trades, or performance series",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "requires a later validation task",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 24) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    raw_guard: pd.DataFrame,
    quality: pd.DataFrame,
    input_checks: pd.DataFrame,
    no_promotion: pd.DataFrame,
) -> str:
    latest = panel.sort_values("trade_date").tail(1).iloc[0]
    valid = panel.loc[panel["history_sufficient"].astype(bool)]
    lines = [
        "# V3.67 Market Participation Breadth Feature Layer",
        "",
        "## Decision",
        "",
        "- V3.67 builds a non-return market participation breadth feature layer.",
        "- It uses Tushare daily-only raw data only through accepted V3.47/V3.48 lineage and the V3.44 raw-adjustment guard.",
        "- It does not produce stock return labels, portfolio outputs, or model promotion evidence.",
        "",
        "## Coverage",
        "",
        f"- Feature rows: `{len(panel)}`",
        f"- History-sufficient rows: `{len(valid)}`",
        f"- Date range: `{panel['trade_date'].min()}` to `{panel['trade_date'].max()}`",
        f"- Latest asset count: `{int(latest.asset_count)}`",
        "",
        "## Latest Snapshot",
        "",
        f"- Latest trade date: `{latest.trade_date}`",
        f"- Signal use date: `{latest.signal_use_date}`",
        f"- Advance ratio: `{float(latest.advance_ratio):.4f}`",
        f"- Decline ratio: `{float(latest.decline_ratio):.4f}`",
        f"- Limit pressure total: `{float(latest.limit_pressure_total):.4f}`",
        f"- Participation breadth score: `{float(latest.market_participation_breadth_score):.4f}`",
        f"- Composite state: `{latest.composite_state}`",
        "",
        "## Feature Sample",
        "",
    ]
    lines.extend(
        markdown_table(
            panel.sort_values("trade_date").tail(12),
            [
                "trade_date",
                "signal_use_date",
                "advance_ratio",
                "decline_ratio",
                "breadth_thrust_5v20",
                "breadth_thrust_20v60",
                "liquidity_expansion_score",
                "breadth_participation_score",
                "crowding_risk_score",
                "market_participation_breadth_score",
            ],
            max_rows=12,
        )
    )
    lines.extend(["", "## Feature Dictionary", ""])
    lines.extend(markdown_table(feature_dictionary, ["feature", "definition", "allowed_use", "forbidden_use"], max_rows=16))
    lines.extend(["", "## Raw Adjustment Guard", ""])
    lines.extend(markdown_table(raw_guard, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Quality Checks", ""])
    lines.extend(markdown_table(quality, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## No Promotion Guard", ""])
    lines.extend(markdown_table(no_promotion, ["result_type", "produced", "blocked", "reason"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.68 can build the industry breadth and dispersion feature layer.",
            "- A later validation task may test V3.67 features against a permitted label source with purged/walk-forward rules.",
            "- Do not use this feature layer as a trading strategy by itself.",
        ]
    )
    return "\n".join(lines)


def build_catalog(panel: pd.DataFrame, config: MarketParticipationBreadthConfig) -> str:
    return "\n".join(
        [
            "# A-share Market Participation Breadth Feature Layer V3.67",
            "",
            "## Dataset Role",
            "",
            "V3.67 converts accepted Tushare daily-only raw OHLCV diagnostics into market participation breadth features.",
            "",
            "## Governance",
            "",
            "- Raw daily prices are used only for cross-sectional breadth and activity diagnostics.",
            "- Stock return labels, adjusted returns, portfolio outputs, and model promotion are blocked.",
            "- Feature timing is after-close for next trade-date research.",
            "",
            "## Produced Shape",
            "",
            f"- Feature rows: `{len(panel)}`",
            f"- Date range: `{panel['trade_date'].min()}` to `{panel['trade_date'].max()}`",
            f"- History-sufficient rows: `{int(panel['history_sufficient'].sum())}`",
            f"- Trailing window: `{config.trailing_window}`",
        ]
    )
