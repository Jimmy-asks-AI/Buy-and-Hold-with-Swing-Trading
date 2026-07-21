"""Macro growth-liquidity feature layer for HIRSSM V3.69.

V3.69 converts the local macro point-in-time panel into a daily as-of feature
panel. It uses only `available_date <= trade_date` joins, records the source
availability date for every series, and does not produce portfolio outputs or
model-promotion evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class MacroGrowthLiquidityConfig:
    v3_66_manifest_path: Path
    v3_66_queue_path: Path
    v3_66_inventory_path: Path
    macro_panel_path: Path
    trade_calendar_path: Path
    output_dir: Path
    catalog_path: Path
    start_date: str
    min_history: int
    trailing_window: int
    short_window: int
    medium_window: int
    long_window: int
    min_feature_rows: int
    min_history_sufficient_rows: int
    min_required_series: int
    stale_monthly_warn_days: int
    stale_daily_warn_days: int


REQUIRED_SERIES = [
    "china_pmi",
    "china_m2_yoy",
    "china_new_financial_credit_yoy",
    "china_cpi_yoy",
    "china_ppi_yoy",
    "cn_10y_gov_bond_yield",
    "us_10y_treasury_yield",
    "cn_us_10y_rate_spread",
    "usdcny",
    "commodity_index",
]

MONTHLY_SERIES = {
    "china_pmi",
    "china_m2_yoy",
    "china_new_financial_credit_yoy",
    "china_cpi_yoy",
    "china_ppi_yoy",
}

DAILY_SERIES = {
    "cn_10y_gov_bond_yield",
    "us_10y_treasury_yield",
    "cn_us_10y_rate_spread",
    "usdcny",
    "commodity_index",
}


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _rolling_zscore(values: pd.Series, window: int, min_history: int) -> pd.Series:
    data = pd.to_numeric(values, errors="coerce")
    mean = data.rolling(window, min_periods=min_history).mean()
    std = data.rolling(window, min_periods=min_history).std(ddof=0).replace(0, np.nan)
    return (data - mean) / std


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


def _clip_score(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(-3, 3) / 3.0


def _score_component(values: pd.Series) -> pd.Series:
    return _clip_score(values).fillna(0.0)


def normalize_trade_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    out = parsed.dt.strftime("%Y%m%d")
    fallback = (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str[:8]
    )
    return out.fillna(fallback)


def validate_inputs(
    config: MacroGrowthLiquidityConfig,
    v3_66_manifest: dict[str, Any],
    v3_66_queue: pd.DataFrame,
    source_inventory: pd.DataFrame,
    macro_panel: pd.DataFrame,
    trade_calendar: pd.DataFrame,
) -> pd.DataFrame:
    required_cols = {
        "date",
        "available_date",
        "series_id",
        "value",
        "source",
        "frequency",
        "revision_policy",
        "pit_quality",
    }
    missing_cols = sorted(required_cols.difference(macro_panel.columns))
    queue_ok = False
    queue_rows = 0
    if not v3_66_queue.empty and {"version_hint", "source_id"}.issubset(v3_66_queue.columns):
        selected = v3_66_queue[
            v3_66_queue["version_hint"].astype(str).eq("V3.69")
            & v3_66_queue["source_id"].astype(str).eq("macro_pit_panel")
        ]
        queue_rows = int(selected.shape[0])
        queue_ok = queue_rows >= 1
    source_ready = False
    if not source_inventory.empty and {"source_id", "allowed_stage"}.issubset(source_inventory.columns):
        source_ready = bool(
            source_inventory.loc[
                source_inventory["source_id"].astype(str).eq("macro_pit_panel")
                & source_inventory["allowed_stage"].astype(str).eq("ready_for_feature_layer")
            ].shape[0]
        )
    series_present = set(macro_panel["series_id"].astype(str).unique()) if "series_id" in macro_panel.columns else set()
    missing_required = sorted(set(REQUIRED_SERIES).difference(series_present))
    available_dates = pd.to_datetime(macro_panel.get("available_date", pd.Series(dtype=str)), errors="coerce")
    values = pd.to_numeric(macro_panel.get("value", pd.Series(dtype=float)), errors="coerce")
    return pd.DataFrame(
        [
            {
                "check": "v3_66_manifest_self_check_passed",
                "status": _status(bool(v3_66_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_66_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_66_queue_contains_v3_69_macro_task",
                "status": _status(queue_ok),
                "detail": f"queue_rows={queue_rows}",
            },
            {
                "check": "v3_66_inventory_marks_macro_ready",
                "status": _status(source_ready),
                "detail": "macro_pit_panel",
            },
            {
                "check": "macro_panel_required_columns_present",
                "status": _status(not missing_cols),
                "detail": ",".join(missing_cols),
            },
            {
                "check": "macro_required_series_present",
                "status": _status(len(missing_required) == 0),
                "detail": ",".join(missing_required),
            },
            {
                "check": "macro_available_date_present",
                "status": _status(available_dates.notna().all() and not macro_panel.empty),
                "detail": f"nulls={int(available_dates.isna().sum())}",
            },
            {
                "check": "macro_values_present",
                "status": _status(values.notna().all() and not macro_panel.empty),
                "detail": f"nulls={int(values.isna().sum())}",
            },
            {
                "check": "trade_calendar_present",
                "status": _status("date" in trade_calendar.columns and not trade_calendar.empty),
                "detail": str(config.trade_calendar_path),
            },
        ]
    )


def build_trade_dates(config: MacroGrowthLiquidityConfig, trade_calendar: pd.DataFrame, macro_panel: pd.DataFrame) -> pd.DataFrame:
    cal = trade_calendar.copy()
    if "date" not in cal.columns:
        raise ValueError("trade calendar missing date column")
    cal["trade_dt"] = pd.to_datetime(cal["date"], errors="coerce")
    cal = cal.dropna(subset=["trade_dt"]).sort_values("trade_dt").drop_duplicates("trade_dt")
    start_dt = pd.to_datetime(config.start_date)
    max_avail = pd.to_datetime(macro_panel["available_date"], errors="coerce").max()
    cal = cal[(cal["trade_dt"] >= start_dt) & (cal["trade_dt"] <= max_avail)].copy()
    if cal.empty:
        raise ValueError("no trade dates after start_date and before max available_date")
    return pd.DataFrame({"trade_dt": cal["trade_dt"].reset_index(drop=True)})


def build_series_coverage_report(macro_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, sub in macro_panel.groupby("series_id", sort=True):
        dates = pd.to_datetime(sub["date"], errors="coerce")
        available = pd.to_datetime(sub["available_date"], errors="coerce")
        rows.append(
            {
                "series_id": series_id,
                "frequency": str(sub["frequency"].iloc[0]) if "frequency" in sub.columns and not sub.empty else "",
                "rows": int(sub.shape[0]),
                "start_date": dates.min().date().isoformat() if dates.notna().any() else "",
                "end_date": dates.max().date().isoformat() if dates.notna().any() else "",
                "available_start": available.min().date().isoformat() if available.notna().any() else "",
                "available_end": available.max().date().isoformat() if available.notna().any() else "",
                "pit_quality": str(sub["pit_quality"].iloc[0]) if "pit_quality" in sub.columns and not sub.empty else "",
                "required_for_v3_69": series_id in REQUIRED_SERIES,
            }
        )
    return pd.DataFrame(rows)


def _asof_one_series(trade_dates: pd.DataFrame, macro_panel: pd.DataFrame, series_id: str) -> pd.DataFrame:
    sub = macro_panel[macro_panel["series_id"].astype(str).eq(series_id)].copy()
    sub["available_dt"] = pd.to_datetime(sub["available_date"], errors="coerce")
    sub["observation_dt"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["available_dt", "observation_dt", "value"]).sort_values(["available_dt", "observation_dt"])
    sub = sub.drop_duplicates("available_dt", keep="last")
    if sub.empty:
        out = trade_dates.copy()
        out[series_id] = np.nan
        out[f"{series_id}_available_dt"] = pd.NaT
        out[f"{series_id}_observation_dt"] = pd.NaT
        out[f"{series_id}_source"] = ""
        out[f"{series_id}_frequency"] = ""
        return out
    right = sub[["available_dt", "observation_dt", "value", "source", "frequency", "revision_policy", "pit_quality"]].copy()
    merged = pd.merge_asof(
        trade_dates.sort_values("trade_dt"),
        right.sort_values("available_dt"),
        left_on="trade_dt",
        right_on="available_dt",
        direction="backward",
    )
    return merged.rename(
        columns={
            "value": series_id,
            "available_dt": f"{series_id}_available_dt",
            "observation_dt": f"{series_id}_observation_dt",
            "source": f"{series_id}_source",
            "frequency": f"{series_id}_frequency",
            "revision_policy": f"{series_id}_revision_policy",
            "pit_quality": f"{series_id}_pit_quality",
        }
    )


def build_macro_asof_panel(macro_panel: pd.DataFrame, trade_calendar: pd.DataFrame, config: MacroGrowthLiquidityConfig) -> pd.DataFrame:
    trade_dates = build_trade_dates(config, trade_calendar, macro_panel)
    out = trade_dates.copy()
    for series_id in REQUIRED_SERIES:
        series_asof = _asof_one_series(trade_dates, macro_panel, series_id)
        keep_cols = [
            "trade_dt",
            series_id,
            f"{series_id}_available_dt",
            f"{series_id}_observation_dt",
            f"{series_id}_source",
            f"{series_id}_frequency",
            f"{series_id}_revision_policy",
            f"{series_id}_pit_quality",
        ]
        out = out.merge(series_asof[keep_cols], on="trade_dt", how="left")
    out["trade_date"] = out["trade_dt"].dt.strftime("%Y%m%d")
    out["available_date"] = out["trade_dt"].dt.strftime("%Y-%m-%d")
    out["signal_use_date"] = out["trade_dt"].shift(-1).dt.strftime("%Y-%m-%d").fillna("")
    out["signal_timing"] = "macro_asof_available_by_trade_date_close_for_next_trade_date"
    out["macro_required_series_available_count"] = out[REQUIRED_SERIES].notna().sum(axis=1)
    for series_id in REQUIRED_SERIES:
        out[f"{series_id}_staleness_days"] = (out["trade_dt"] - out[f"{series_id}_available_dt"]).dt.days
    return out


def add_macro_features(asof_panel: pd.DataFrame, config: MacroGrowthLiquidityConfig) -> pd.DataFrame:
    out = asof_panel.copy()
    for series_id in REQUIRED_SERIES:
        out[series_id] = pd.to_numeric(out[series_id], errors="coerce")

    w_short = config.short_window
    w_medium = config.medium_window
    w_long = config.long_window
    trailing = config.trailing_window
    min_history = config.min_history

    out["china_pmi_gap"] = out["china_pmi"] - 50.0
    out["china_pmi_chg_short"] = out["china_pmi"] - out["china_pmi"].shift(w_short)
    out["china_pmi_chg_medium"] = out["china_pmi"] - out["china_pmi"].shift(w_medium)
    out["china_m2_yoy_chg_medium"] = out["china_m2_yoy"] - out["china_m2_yoy"].shift(w_medium)
    out["china_credit_yoy_chg_medium"] = out["china_new_financial_credit_yoy"] - out["china_new_financial_credit_yoy"].shift(w_medium)
    out["china_cpi_ppi_gap"] = out["china_cpi_yoy"] - out["china_ppi_yoy"]
    out["china_ppi_yoy_chg_medium"] = out["china_ppi_yoy"] - out["china_ppi_yoy"].shift(w_medium)
    out["cn_10y_yield_chg_medium"] = out["cn_10y_gov_bond_yield"] - out["cn_10y_gov_bond_yield"].shift(w_medium)
    out["us_10y_yield_chg_medium"] = out["us_10y_treasury_yield"] - out["us_10y_treasury_yield"].shift(w_medium)
    out["cn_us_rate_spread_chg_medium"] = out["cn_us_10y_rate_spread"] - out["cn_us_10y_rate_spread"].shift(w_medium)
    out["usdcny_ret_short"] = out["usdcny"] / out["usdcny"].shift(w_short) - 1.0
    out["usdcny_ret_medium"] = out["usdcny"] / out["usdcny"].shift(w_medium) - 1.0
    out["commodity_ret_short"] = out["commodity_index"] / out["commodity_index"].shift(w_short) - 1.0
    out["commodity_ret_medium"] = out["commodity_index"] / out["commodity_index"].shift(w_medium) - 1.0
    out["commodity_ret_long"] = out["commodity_index"] / out["commodity_index"].shift(w_long) - 1.0

    transform_cols = [
        "china_pmi_gap",
        "china_pmi_chg_medium",
        "china_m2_yoy",
        "china_m2_yoy_chg_medium",
        "china_new_financial_credit_yoy",
        "china_credit_yoy_chg_medium",
        "china_cpi_yoy",
        "china_ppi_yoy",
        "china_cpi_ppi_gap",
        "china_ppi_yoy_chg_medium",
        "cn_10y_gov_bond_yield",
        "cn_10y_yield_chg_medium",
        "us_10y_treasury_yield",
        "us_10y_yield_chg_medium",
        "cn_us_10y_rate_spread",
        "cn_us_rate_spread_chg_medium",
        "usdcny",
        "usdcny_ret_medium",
        "commodity_index",
        "commodity_ret_medium",
        "commodity_ret_long",
    ]
    for col in transform_cols:
        out[f"{col}_prior_pctile_{trailing}d"] = _trailing_percentile(out[col], trailing, min_history)
        out[f"{col}_z_{trailing}d"] = _rolling_zscore(out[col], trailing, min_history)

    out["macro_growth_momentum_score"] = (
        _score_component(out[f"china_pmi_gap_z_{trailing}d"])
        + _score_component(out[f"china_pmi_chg_medium_z_{trailing}d"])
        + _score_component(out[f"china_ppi_yoy_chg_medium_z_{trailing}d"])
        + _score_component(out[f"commodity_ret_medium_z_{trailing}d"])
    ) / 4.0
    out["macro_liquidity_impulse_score"] = (
        _score_component(out[f"china_m2_yoy_z_{trailing}d"])
        + _score_component(out[f"china_m2_yoy_chg_medium_z_{trailing}d"])
        + _score_component(out[f"china_new_financial_credit_yoy_z_{trailing}d"])
        + _score_component(out[f"china_credit_yoy_chg_medium_z_{trailing}d"])
        - _score_component(out[f"cn_10y_yield_chg_medium_z_{trailing}d"])
    ) / 5.0
    out["macro_inflation_policy_constraint_score"] = (
        _score_component(out[f"china_cpi_yoy_z_{trailing}d"])
        + _score_component(out[f"china_ppi_yoy_z_{trailing}d"])
        + _score_component(out[f"commodity_ret_long_z_{trailing}d"])
        + _score_component(out[f"cn_10y_yield_chg_medium_z_{trailing}d"])
    ) / 4.0
    out["macro_external_pressure_score"] = (
        _score_component(out[f"us_10y_yield_chg_medium_z_{trailing}d"])
        - _score_component(out[f"cn_us_10y_rate_spread_z_{trailing}d"])
        - _score_component(out[f"cn_us_rate_spread_chg_medium_z_{trailing}d"])
        + _score_component(out[f"usdcny_ret_medium_z_{trailing}d"])
    ) / 4.0
    out["macro_growth_liquidity_mix_score"] = (
        0.35 * out["macro_growth_momentum_score"]
        + 0.35 * out["macro_liquidity_impulse_score"]
        - 0.20 * out["macro_inflation_policy_constraint_score"]
        - 0.10 * out["macro_external_pressure_score"]
    )
    out["macro_risk_pressure_score"] = (
        0.45 * out["macro_external_pressure_score"]
        + 0.35 * out["macro_inflation_policy_constraint_score"]
        - 0.15 * out["macro_growth_momentum_score"]
        - 0.05 * out["macro_liquidity_impulse_score"]
    )

    out["growth_state"] = np.select(
        [
            (out["china_pmi"] >= 50.0) & (out["macro_growth_momentum_score"] > 0.15),
            (out["china_pmi"] < 49.0) | (out["macro_growth_momentum_score"] < -0.25),
        ],
        ["growth_expansion", "growth_slowdown"],
        default="growth_mixed",
    )
    out["liquidity_state"] = np.select(
        [
            out["macro_liquidity_impulse_score"] >= 0.25,
            out["macro_liquidity_impulse_score"] <= -0.25,
        ],
        ["liquidity_support", "liquidity_tight"],
        default="liquidity_neutral",
    )
    out["inflation_state"] = np.select(
        [
            out["macro_inflation_policy_constraint_score"] >= 0.35,
            out["macro_inflation_policy_constraint_score"] <= -0.35,
        ],
        ["inflation_constraint", "disinflation_relief"],
        default="inflation_neutral",
    )
    out["external_state"] = np.select(
        [
            out["macro_external_pressure_score"] >= 0.35,
            out["macro_external_pressure_score"] <= -0.35,
        ],
        ["external_pressure", "external_relief"],
        default="external_neutral",
    )
    out["composite_macro_state"] = np.select(
        [
            (out["growth_state"] == "growth_slowdown") & (out["liquidity_state"] == "liquidity_support"),
            (out["liquidity_state"] == "liquidity_tight") & (out["growth_state"] == "growth_slowdown"),
            (out["inflation_state"] == "inflation_constraint") | (out["external_state"] == "external_pressure"),
            (out["growth_state"] == "growth_expansion") & (out["liquidity_state"] == "liquidity_support"),
        ],
        ["policy_support_window", "growth_liquidity_headwind", "macro_pressure", "growth_liquidity_tailwind"],
        default="neutral_macro_mix",
    )
    score_cols = [
        "macro_growth_momentum_score",
        "macro_liquidity_impulse_score",
        "macro_inflation_policy_constraint_score",
        "macro_external_pressure_score",
        "macro_growth_liquidity_mix_score",
        "macro_risk_pressure_score",
    ]
    out["history_sufficient"] = (
        out["macro_required_series_available_count"].ge(config.min_required_series)
        & out[score_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & out[f"cn_us_10y_rate_spread_z_{trailing}d"].notna()
        & out[f"china_pmi_gap_z_{trailing}d"].notna()
    )
    out["asof_join_policy"] = "available_date_less_than_or_equal_trade_date"
    out["feature_family"] = "macro_growth_liquidity"
    out["feature_scope"] = "macro_pit_asof_daily_feature_layer"
    out["portfolio_harness_allowed"] = False
    out["model_promotion_allowed"] = False
    out["official_total_return_evidence"] = False
    out["macro_vintage_limitation"] = "vendor_snapshot_or_release_event_without_full_vintage_history"

    preferred = [
        "trade_date",
        "available_date",
        "signal_use_date",
        "signal_timing",
        "macro_required_series_available_count",
        *REQUIRED_SERIES,
        "china_pmi_gap",
        "china_pmi_chg_medium",
        "china_m2_yoy_chg_medium",
        "china_credit_yoy_chg_medium",
        "china_cpi_ppi_gap",
        "china_ppi_yoy_chg_medium",
        "cn_10y_yield_chg_medium",
        "us_10y_yield_chg_medium",
        "cn_us_rate_spread_chg_medium",
        "usdcny_ret_medium",
        "commodity_ret_medium",
        "commodity_ret_long",
        "macro_growth_momentum_score",
        "macro_liquidity_impulse_score",
        "macro_inflation_policy_constraint_score",
        "macro_external_pressure_score",
        "macro_growth_liquidity_mix_score",
        "macro_risk_pressure_score",
        "history_sufficient",
        "growth_state",
        "liquidity_state",
        "inflation_state",
        "external_state",
        "composite_macro_state",
        "asof_join_policy",
        "feature_family",
        "feature_scope",
        "portfolio_harness_allowed",
        "model_promotion_allowed",
        "official_total_return_evidence",
        "macro_vintage_limitation",
    ]
    staleness = [f"{series_id}_staleness_days" for series_id in REQUIRED_SERIES]
    asof_cols = []
    for series_id in REQUIRED_SERIES:
        asof_cols.extend(
            [
                f"{series_id}_available_dt",
                f"{series_id}_observation_dt",
                f"{series_id}_frequency",
                f"{series_id}_pit_quality",
            ]
        )
    derived = [
        col
        for col in out.columns
        if col.endswith(f"_prior_pctile_{trailing}d") or col.endswith(f"_z_{trailing}d")
    ]
    selected = [col for col in preferred + staleness + asof_cols + derived if col in out.columns]
    result = out[selected].copy()
    for col in result.columns:
        if col.endswith("_available_dt") or col.endswith("_observation_dt"):
            result[col] = pd.to_datetime(result[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return result.reset_index(drop=True)


def build_asof_join_checks(asof_panel: pd.DataFrame, config: MacroGrowthLiquidityConfig) -> pd.DataFrame:
    rows = []
    trade_dt = pd.to_datetime(asof_panel["trade_date"], format="%Y%m%d", errors="coerce")
    for series_id in REQUIRED_SERIES:
        available_col = f"{series_id}_available_dt"
        observation_col = f"{series_id}_observation_dt"
        staleness_col = f"{series_id}_staleness_days"
        available = pd.to_datetime(asof_panel[available_col], errors="coerce") if available_col in asof_panel else pd.Series(pd.NaT, index=asof_panel.index)
        observation = pd.to_datetime(asof_panel[observation_col], errors="coerce") if observation_col in asof_panel else pd.Series(pd.NaT, index=asof_panel.index)
        values = pd.to_numeric(asof_panel[series_id], errors="coerce") if series_id in asof_panel else pd.Series(np.nan, index=asof_panel.index)
        staleness = pd.to_numeric(asof_panel[staleness_col], errors="coerce") if staleness_col in asof_panel else pd.Series(np.nan, index=asof_panel.index)
        future_available = int((available > trade_dt).sum())
        frequency = "monthly" if series_id in MONTHLY_SERIES else "daily"
        warn_days = config.stale_monthly_warn_days if frequency == "monthly" else config.stale_daily_warn_days
        latest_stale = float(staleness.dropna().iloc[-1]) if staleness.notna().any() else np.nan
        status = "pass"
        if future_available > 0 or values.notna().sum() == 0:
            status = "fail"
        elif np.isfinite(latest_stale) and latest_stale > warn_days:
            status = "warn"
        rows.append(
            {
                "series_id": series_id,
                "frequency": frequency,
                "status": status,
                "asof_policy": "available_date <= trade_date",
                "future_available_date_count": future_available,
                "nonnull_value_rows": int(values.notna().sum()),
                "first_nonnull_trade_date": trade_dt[values.notna()].min().date().isoformat() if values.notna().any() else "",
                "latest_available_date": available.dropna().max().date().isoformat() if available.notna().any() else "",
                "latest_observation_date": observation.dropna().max().date().isoformat() if observation.notna().any() else "",
                "latest_staleness_days": latest_stale,
                "stale_warn_threshold_days": warn_days,
            }
        )
    return pd.DataFrame(rows)


def build_feature_dictionary(config: MacroGrowthLiquidityConfig) -> pd.DataFrame:
    rows = [
        {
            "feature": "macro_growth_momentum_score",
            "definition": "Composite of PMI gap, PMI change, PPI change, and commodity momentum after rolling normalization.",
            "allowed_use": "growth-cycle state feature for later validation",
            "forbidden_use": "standalone trading strategy or performance claim",
        },
        {
            "feature": "macro_liquidity_impulse_score",
            "definition": "Composite of M2 growth, credit growth, their changes, and China 10Y yield change.",
            "allowed_use": "liquidity support or tightening state feature",
            "forbidden_use": "TSF claim when TSF series is unavailable",
        },
        {
            "feature": "macro_inflation_policy_constraint_score",
            "definition": "Composite of CPI, PPI, commodity momentum, and China 10Y yield pressure.",
            "allowed_use": "policy constraint and inflation pressure diagnostic",
            "forbidden_use": "policy forecast without validation",
        },
        {
            "feature": "macro_external_pressure_score",
            "definition": "Composite of US rate change, China-US rate-spread weakness, and RMB depreciation pressure.",
            "allowed_use": "external discount-rate and FX pressure diagnostic",
            "forbidden_use": "default risk-budget gate before later validation",
        },
        {
            "feature": "macro_growth_liquidity_mix_score",
            "definition": "Weighted blend of growth and liquidity support minus inflation and external pressure.",
            "allowed_use": "candidate feature for later guarded validation",
            "forbidden_use": "model promotion in V3.69",
        },
    ]
    for row in rows:
        row["source_scope"] = "macro_pit_panel_available_date_asof"
        row["timing"] = "available_date <= trade_date; signal_use_date is next trade date"
        row["trailing_window"] = config.trailing_window
    return pd.DataFrame(rows)


def build_signal_validation_plan(config: MacroGrowthLiquidityConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "v3_66_macro_growth_liquidity_mix_v1",
                "feature_columns": "macro_growth_liquidity_mix_score,macro_growth_momentum_score,macro_liquidity_impulse_score",
                "hypothesis": "Growth stabilization plus liquidity impulse may identify policy-support windows after macro slowdown.",
                "next_validation": "Join with permitted market or style label source using signal_use_date and purged walk-forward splits.",
                "blocked_now": "portfolio_harness;model_promotion;official_total_return_claim;same_branch_threshold_tuning",
            },
            {
                "signal_id": "macro_pressure_risk_filter_v1",
                "feature_columns": "macro_risk_pressure_score,macro_inflation_policy_constraint_score,macro_external_pressure_score",
                "hypothesis": "Inflation and external pressure can reduce risk budget only after state-conditioned validation.",
                "next_validation": "Validate as risk filter, not return-seeking alpha, with state and stale-data diagnostics.",
                "blocked_now": "portfolio_harness;model_promotion;parameter_sweep",
            },
            {
                "signal_id": "policy_support_window_v1",
                "feature_columns": "composite_macro_state,liquidity_state,growth_state",
                "hypothesis": "Macro slowdown with liquidity support can be a different state than pure growth deterioration.",
                "next_validation": "Test state labels against later market labels without changing portfolio weights in this task.",
                "blocked_now": "portfolio_harness;model_promotion;use_without_available_date",
            },
        ]
    )


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "macro_growth_liquidity_feature_layer",
                "produced": True,
                "blocked": False,
                "reason": "V3.69 creates available-date as-of macro features only",
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
                "reason": "requires a later guarded validation task",
            },
            {
                "result_type": "macro_without_available_date",
                "produced": False,
                "blocked": True,
                "reason": "all macro joins are checked by available_date <= trade_date",
            },
        ]
    )


def build_feature_quality_checks(
    panel: pd.DataFrame,
    asof_checks: pd.DataFrame,
    input_checks: pd.DataFrame,
    config: MacroGrowthLiquidityConfig,
) -> pd.DataFrame:
    forbidden_output_terms = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(panel.columns).lower()})
    score_cols = [
        "macro_growth_momentum_score",
        "macro_liquidity_impulse_score",
        "macro_inflation_policy_constraint_score",
        "macro_external_pressure_score",
        "macro_growth_liquidity_mix_score",
        "macro_risk_pressure_score",
    ]
    finite_scores = panel.loc[panel["history_sufficient"].astype(bool), score_cols].replace([np.inf, -np.inf], np.nan)
    latest = panel.sort_values("trade_date").tail(1)
    model_any = _bool_series(panel["model_promotion_allowed"]).any() if "model_promotion_allowed" in panel else True
    portfolio_any = _bool_series(panel["portfolio_harness_allowed"]).any() if "portfolio_harness_allowed" in panel else True
    official_any = _bool_series(panel["official_total_return_evidence"]).any() if "official_total_return_evidence" in panel else True
    asof_fail = asof_checks["status"].eq("fail").any() if not asof_checks.empty else True
    asof_warn = asof_checks["status"].eq("warn").any() if not asof_checks.empty else False
    rows = [
        {
            "check": "minimum_feature_rows_present",
            "status": _status(len(panel) >= config.min_feature_rows),
            "detail": f"rows={len(panel)};min={config.min_feature_rows}",
        },
        {
            "check": "minimum_history_sufficient_rows_present",
            "status": _status(int(panel["history_sufficient"].sum()) >= config.min_history_sufficient_rows),
            "detail": f"history_sufficient={int(panel['history_sufficient'].sum())};min={config.min_history_sufficient_rows}",
        },
        {
            "check": "latest_required_series_count_large_enough",
            "status": _status(not latest.empty and int(latest["macro_required_series_available_count"].iloc[0]) >= config.min_required_series),
            "detail": f"latest_count={int(latest['macro_required_series_available_count'].iloc[0]) if not latest.empty else 0}",
        },
        {
            "check": "asof_join_has_no_future_available_date",
            "status": _status(not asof_fail),
            "detail": ";".join(asof_checks.loc[asof_checks["status"].eq("fail"), "series_id"].astype(str)) if not asof_checks.empty else "missing_asof_checks",
        },
        {
            "check": "asof_staleness_warning_recorded",
            "status": "warn" if asof_warn else "pass",
            "detail": ";".join(asof_checks.loc[asof_checks["status"].eq("warn"), "series_id"].astype(str)) if not asof_checks.empty else "",
        },
        {
            "check": "input_checks_have_no_failures",
            "status": _status(not input_checks["status"].eq("fail").any()),
            "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
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
            "check": "promotion_and_official_evidence_blocked",
            "status": _status(not model_any and not portfolio_any and not official_any),
            "detail": f"model={bool(model_any)};portfolio={bool(portfolio_any)};official={bool(official_any)}",
        },
        {
            "check": "forbidden_promotion_columns_absent",
            "status": _status(not forbidden_output_terms),
            "detail": ",".join(forbidden_output_terms),
        },
    ]
    return pd.DataFrame(rows)


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    asof_checks: pd.DataFrame,
    quality_checks: pd.DataFrame,
    no_promotion: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    column_text = " ".join(output_column_names).lower()
    hits = sorted(term for term in forbidden if term in column_text)
    blocked_result_types = {"portfolio_harness", "model_promotion", "macro_without_available_date"}
    blocked_correctly = not no_promotion.loc[no_promotion["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    quality_failures = quality_checks["status"].eq("fail") if not quality_checks.empty else pd.Series([True])
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if not input_checks["status"].eq("fail").any() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "asof_join_checks_passed",
                "status": "pass" if not asof_checks["status"].eq("fail").any() else "fail",
                "detail": ";".join(asof_checks.loc[asof_checks["status"].eq("fail"), "series_id"].astype(str)),
            },
            {
                "check": "feature_quality_checks_passed",
                "status": "pass" if not quality_failures.any() else "fail",
                "detail": ";".join(quality_checks.loc[quality_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "promotion_outputs_blocked",
                "status": "pass" if blocked_correctly and not hits else "fail",
                "detail": ",".join(hits),
            },
        ]
    )


def build_feature_contract(config: MacroGrowthLiquidityConfig) -> str:
    return "\n".join(
        [
            "# V3.69 Macro Growth-Liquidity Feature Contract",
            "",
            "## Scope",
            "",
            "- Source: `data_raw/macro/macro_pit_panel.csv`.",
            "- Timing: every macro value is selected with `available_date <= trade_date`.",
            "- Signal use: features are after-close research inputs for the next trade date.",
            "",
            "## Allowed",
            "",
            "- Growth, liquidity, inflation, external pressure, and macro state diagnostics.",
            "- Rolling z-scores and trailing percentiles based only on prior as-of values.",
            "- Staleness diagnostics for monthly and daily macro sources.",
            "",
            "## Forbidden",
            "",
            "- Macro values without `available_date`.",
            "- Portfolio harness, NAV, Sharpe, drawdown, annualized performance, or model promotion.",
            "- Official total-return claims.",
            "- Parameter tuning against V3.64/V3.65 proxy outcomes.",
            "",
            "## Required Next Gate",
            "",
            "A later validation task must join V3.67, V3.68, and V3.69 features to a permitted label source with explicit lag, purged walk-forward splits, and stale-data diagnostics.",
            "",
            f"Configured trailing window: {config.trailing_window} trading days.",
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
    series_coverage: pd.DataFrame,
    asof_checks: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    validation_plan: pd.DataFrame,
    input_checks: pd.DataFrame,
    quality_checks: pd.DataFrame,
    no_promotion: pd.DataFrame,
) -> str:
    latest = panel.sort_values("trade_date").tail(1).iloc[0]
    valid = panel.loc[panel["history_sufficient"].astype(bool)]
    stale_warn = asof_checks[asof_checks["status"].astype(str).eq("warn")]["series_id"].astype(str).tolist()
    lines = [
        "# V3.69 Macro Growth-Liquidity Feature Layer",
        "",
        "## Decision",
        "",
        "- V3.69 builds a point-in-time macro as-of feature layer.",
        "- It covers growth, liquidity, inflation policy constraint, external pressure, and composite macro state.",
        "- It does not produce portfolio outputs, official total-return evidence, or model promotion evidence.",
        "",
        "## Coverage",
        "",
        f"- Feature rows: `{len(panel)}`",
        f"- History-sufficient rows: `{len(valid)}`",
        f"- Date range: `{panel['trade_date'].min()}` to `{panel['trade_date'].max()}`",
        f"- Latest required series count: `{int(latest.macro_required_series_available_count)}`",
        f"- Stale warning series: `{', '.join(stale_warn) if stale_warn else 'none'}`",
        "",
        "## Latest Snapshot",
        "",
        f"- Latest trade date: `{latest.trade_date}`",
        f"- Signal use date: `{latest.signal_use_date}`",
        f"- PMI: `{float(latest.china_pmi):.4f}`",
        f"- M2 YoY: `{float(latest.china_m2_yoy):.4f}`",
        f"- China-US 10Y spread: `{float(latest.cn_us_10y_rate_spread):.4f}`",
        f"- USD/CNY: `{float(latest.usdcny):.4f}`",
        f"- Macro growth-liquidity mix score: `{float(latest.macro_growth_liquidity_mix_score):.4f}`",
        f"- Macro risk pressure score: `{float(latest.macro_risk_pressure_score):.4f}`",
        f"- Composite macro state: `{latest.composite_macro_state}`",
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
                "china_pmi",
                "china_m2_yoy",
                "cn_us_10y_rate_spread",
                "usdcny",
                "macro_growth_momentum_score",
                "macro_liquidity_impulse_score",
                "macro_external_pressure_score",
                "macro_growth_liquidity_mix_score",
                "composite_macro_state",
            ],
            max_rows=12,
        )
    )
    lines.extend(["", "## Series Coverage", ""])
    lines.extend(markdown_table(series_coverage, ["series_id", "frequency", "rows", "available_start", "available_end", "pit_quality"], max_rows=16))
    lines.extend(["", "## As-Of Join Checks", ""])
    lines.extend(markdown_table(asof_checks, ["series_id", "status", "future_available_date_count", "latest_available_date", "latest_staleness_days", "stale_warn_threshold_days"], max_rows=16))
    lines.extend(["", "## Feature Dictionary", ""])
    lines.extend(markdown_table(feature_dictionary, ["feature", "definition", "allowed_use", "forbidden_use"], max_rows=12))
    lines.extend(["", "## Signal Validation Plan", ""])
    lines.extend(markdown_table(validation_plan, ["signal_id", "feature_columns", "next_validation", "blocked_now"], max_rows=12))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Quality Checks", ""])
    lines.extend(markdown_table(quality_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## No Promotion Guard", ""])
    lines.extend(markdown_table(no_promotion, ["result_type", "produced", "blocked", "reason"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.70 can build a combined feature registry joining V3.67, V3.68, and V3.69 by next-trade signal date.",
            "- A later guarded validation task can test the combined features against a permitted label source.",
            "- Do not use this feature layer as a trading strategy by itself.",
        ]
    )
    return "\n".join(lines)


def build_catalog(panel: pd.DataFrame, config: MacroGrowthLiquidityConfig) -> str:
    return "\n".join(
        [
            "# A-share Macro Growth-Liquidity Feature Layer V3.69",
            "",
            "## Dataset Role",
            "",
            "V3.69 converts macro PIT rows into a daily as-of feature panel for growth, liquidity, inflation, and external-pressure diagnostics.",
            "",
            "## Governance",
            "",
            "- Every macro value is joined by `available_date <= trade_date`.",
            "- Macro vintage limitations are retained as dataset metadata.",
            "- Portfolio outputs and default model promotion are blocked.",
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
