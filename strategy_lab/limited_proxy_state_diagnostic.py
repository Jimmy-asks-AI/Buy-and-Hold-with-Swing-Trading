"""Limited-window state diagnostics for HIRSSM V3.57.

This module deliberately produces smoke-test diagnostics only. It can check
whether proxy states, V3.50 candidate signals, and V3.56 limited labels join
cleanly, but it must not emit IC, hit-rate, NAV, Sharpe, or portfolio results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LimitedStateDiagnosticConfig:
    source_path: Path
    signal_panel_path: Path
    label_path: Path
    v3_56_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    trend_window: int
    volatility_window: int
    drawdown_window: int
    trend_up_threshold: float
    trend_down_threshold: float
    low_volatility_threshold: float
    high_volatility_threshold: float
    moderate_drawdown_threshold: float
    deep_drawdown_threshold: float
    label_abs_outlier_threshold: float
    min_source_rows: int
    min_signal_rows: int
    min_label_rows: int
    min_joined_rows: int
    min_state_date_coverage_ratio: float
    min_history_available_signal_date_ratio: float
    min_state_bucket_signal_dates: int


STATE_COLUMNS = [
    "proxy_market_state",
    "trend_state",
    "volatility_state",
    "drawdown_state",
]

FORBIDDEN_PERFORMANCE_TERMS = {
    "rank_ic",
    "icir",
    "hit_rate",
    "sharpe",
    "nav",
    "annualized_return",
    "portfolio_return",
    "max_drawdown",
    "pbo",
    "deflated_sharpe",
}


def normalize_date_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("-", "", regex=False).str.replace(".0", "", regex=False).str[:8]


def _status(ok: bool, fail_status: str = "blocked") -> str:
    return "pass" if ok else fail_status


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def validate_source(source: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    required = {
        "date",
        "asset_or_index",
        "total_return_index_or_adjusted_close",
        "available_date",
        "data_source",
        "source_vintage",
    }
    missing = sorted(required.difference(source.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "source_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        },
        {
            "check": "source_minimum_rows",
            "status": _status(len(source) >= config.min_source_rows),
            "detail": f"rows={len(source)};min={config.min_source_rows}",
        },
    ]
    if missing:
        return pd.DataFrame(rows)
    dates = normalize_date_series(source["date"])
    available = normalize_date_series(source["available_date"])
    levels = _finite_numeric(source["total_return_index_or_adjusted_close"])
    lineage = " ".join(source["data_source"].astype(str).unique()).lower()
    duplicate_rows = int(source.assign(_date=dates).duplicated(["_date", "asset_or_index"]).sum())
    forbidden_lineage = any(term in lineage for term in ["unadjusted", "raw_close", "none_raw", "price_only"])
    rows.extend(
        [
            {
                "check": "source_dates_parseable",
                "status": _status(pd.to_datetime(dates, format="%Y%m%d", errors="coerce").notna().all()),
                "detail": f"bad_rows={int(pd.to_datetime(dates, format='%Y%m%d', errors='coerce').isna().sum())}",
            },
            {
                "check": "source_available_date_not_before_date",
                "status": _status((available >= dates).all()),
                "detail": f"bad_rows={int((available < dates).sum())}",
            },
            {
                "check": "source_level_positive_finite",
                "status": _status(levels.notna().all() and bool((levels > 0).all())),
                "detail": f"bad_rows={int((levels.isna() | (levels <= 0)).sum())}",
            },
            {
                "check": "source_no_duplicate_date_asset",
                "status": _status(duplicate_rows == 0),
                "detail": f"duplicates={duplicate_rows}",
            },
            {
                "check": "source_lineage_is_limited_adjusted_proxy",
                "status": _status("approved_adjusted_proxy" in lineage),
                "detail": "requires approved_adjusted_proxy",
            },
            {
                "check": "source_lineage_has_no_forbidden_raw_terms",
                "status": _status(not forbidden_lineage),
                "detail": "forbidden terms: unadjusted/raw_close/none_raw/price_only",
            },
        ]
    )
    return pd.DataFrame(rows)


def validate_signal_panel(signals: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    required = {"signal_id", "signal_date", "asset", "signal_value", "signal_direction", "available_date"}
    missing = sorted(required.difference(signals.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "signal_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        },
        {
            "check": "signal_minimum_rows",
            "status": _status(len(signals) >= config.min_signal_rows),
            "detail": f"rows={len(signals)};min={config.min_signal_rows}",
        },
    ]
    if missing:
        return pd.DataFrame(rows)
    signal_dates = normalize_date_series(signals["signal_date"])
    available = normalize_date_series(signals["available_date"])
    model_allowed = bool(signals.get("model_promotion_allowed", pd.Series(False, index=signals.index)).astype(bool).any())
    perf_allowed = bool(signals.get("performance_claim_allowed", pd.Series(False, index=signals.index)).astype(bool).any())
    assets = set(signals["asset"].astype(str).unique())
    rows.extend(
        [
            {
                "check": "signal_dates_parseable",
                "status": _status(pd.to_datetime(signal_dates, format="%Y%m%d", errors="coerce").notna().all()),
                "detail": f"bad_rows={int(pd.to_datetime(signal_dates, format='%Y%m%d', errors='coerce').isna().sum())}",
            },
            {
                "check": "signal_available_date_not_after_signal_date",
                "status": _status((available <= signal_dates).all()),
                "detail": f"bad_rows={int((available > signal_dates).sum())}",
            },
            {
                "check": "signal_scope_is_market_only",
                "status": _status(assets == {"MARKET"}),
                "detail": ",".join(sorted(assets)),
            },
            {
                "check": "signal_model_and_performance_flags_false",
                "status": _status(not model_allowed and not perf_allowed),
                "detail": f"model_allowed={model_allowed};performance_allowed={perf_allowed}",
            },
        ]
    )
    return pd.DataFrame(rows)


def validate_labels(labels: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    required = {
        "signal_date",
        "asset",
        "horizon",
        "forward_adjusted_return",
        "return_basis",
        "label_available_date",
        "price_adjustment_source",
    }
    missing = sorted(required.difference(labels.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "label_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        },
        {
            "check": "label_minimum_rows",
            "status": _status(len(labels) >= config.min_label_rows),
            "detail": f"rows={len(labels)};min={config.min_label_rows}",
        },
    ]
    if missing:
        return pd.DataFrame(rows)
    signal_dates = normalize_date_series(labels["signal_date"])
    available = normalize_date_series(labels["label_available_date"])
    returns = _finite_numeric(labels["forward_adjusted_return"])
    horizons = set(pd.to_numeric(labels["horizon"], errors="coerce").dropna().astype(int))
    expected_horizons = set(config.horizons)
    basis = set(labels["return_basis"].astype(str).unique())
    lineage = " ".join(labels["price_adjustment_source"].astype(str).unique()).lower()
    duplicate_rows = int(labels.assign(_date=signal_dates).duplicated(["_date", "asset", "horizon"]).sum())
    rows.extend(
        [
            {
                "check": "label_dates_parseable",
                "status": _status(pd.to_datetime(signal_dates, format="%Y%m%d", errors="coerce").notna().all()),
                "detail": f"bad_rows={int(pd.to_datetime(signal_dates, format='%Y%m%d', errors='coerce').isna().sum())}",
            },
            {
                "check": "label_available_date_after_signal_date",
                "status": _status((available > signal_dates).all()),
                "detail": f"bad_rows={int((available <= signal_dates).sum())}",
            },
            {
                "check": "label_returns_finite",
                "status": _status(returns.notna().all()),
                "detail": f"bad_rows={int(returns.isna().sum())}",
            },
            {
                "check": "label_horizons_match_config",
                "status": _status(horizons == expected_horizons),
                "detail": f"actual={sorted(horizons)};expected={sorted(expected_horizons)}",
            },
            {
                "check": "label_return_basis_allowed",
                "status": _status(basis.issubset({"adjusted_return", "total_return"})),
                "detail": ",".join(sorted(basis)),
            },
            {
                "check": "label_lineage_is_limited_adjusted_proxy",
                "status": _status("approved_adjusted_proxy" in lineage),
                "detail": "requires approved_adjusted_proxy",
            },
            {
                "check": "label_no_duplicate_date_asset_horizon",
                "status": _status(duplicate_rows == 0),
                "detail": f"duplicates={duplicate_rows}",
            },
        ]
    )
    return pd.DataFrame(rows)


def _classify_trend(value: float, config: LimitedStateDiagnosticConfig) -> str:
    if pd.isna(value):
        return "insufficient_history"
    if value >= config.trend_up_threshold:
        return "positive_trend"
    if value <= config.trend_down_threshold:
        return "negative_trend"
    return "range_trend"


def _classify_volatility(value: float, config: LimitedStateDiagnosticConfig) -> str:
    if pd.isna(value):
        return "insufficient_history"
    if value >= config.high_volatility_threshold:
        return "high_volatility"
    if value <= config.low_volatility_threshold:
        return "low_volatility"
    return "normal_volatility"


def _classify_drawdown(value: float, config: LimitedStateDiagnosticConfig) -> str:
    if pd.isna(value):
        return "insufficient_history"
    if value <= config.deep_drawdown_threshold:
        return "deep_drawdown"
    if value <= config.moderate_drawdown_threshold:
        return "moderate_drawdown"
    return "shallow_drawdown_or_high"


def _classify_proxy_market_state(row: pd.Series) -> str:
    if not bool(row["history_available"]):
        return "insufficient_proxy_history"
    if row["drawdown_state"] in {"deep_drawdown", "moderate_drawdown"} and row["trend_state"] == "negative_trend":
        return "stress_decline"
    if row["volatility_state"] == "high_volatility":
        return "high_volatility_transition"
    if row["trend_state"] == "positive_trend" and row["drawdown_state"] == "shallow_drawdown_or_high":
        return "risk_on_or_recovery"
    if row["trend_state"] == "range_trend" and row["drawdown_state"] == "shallow_drawdown_or_high":
        return "range_normal"
    return "mixed_proxy_state"


def build_proxy_state_panel(source: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    out = source.copy()
    out["date"] = normalize_date_series(out["date"])
    out["available_date"] = normalize_date_series(out["available_date"])
    out["level"] = _finite_numeric(out["total_return_index_or_adjusted_close"])
    out = out.sort_values("date").drop_duplicates(["date", "asset_or_index"], keep="last")
    out["daily_return"] = out["level"].pct_change()
    out["trend_return"] = out["level"] / out["level"].shift(config.trend_window) - 1.0
    min_vol_periods = min(10, config.volatility_window)
    min_drawdown_periods = min(20, config.drawdown_window)
    out["realized_volatility"] = out["daily_return"].rolling(config.volatility_window, min_periods=min_vol_periods).std() * np.sqrt(252)
    trailing_max = out["level"].rolling(config.drawdown_window, min_periods=min_drawdown_periods).max()
    out["drawdown_to_trailing_max"] = out["level"] / trailing_max - 1.0
    out["trend_state"] = out["trend_return"].apply(lambda value: _classify_trend(value, config))
    out["volatility_state"] = out["realized_volatility"].apply(lambda value: _classify_volatility(value, config))
    out["drawdown_state"] = out["drawdown_to_trailing_max"].apply(lambda value: _classify_drawdown(value, config))
    out["history_available"] = ~out[["trend_return", "realized_volatility", "drawdown_to_trailing_max"]].isna().any(axis=1)
    out["proxy_market_state"] = out.apply(_classify_proxy_market_state, axis=1)
    out["state_available_date"] = out["available_date"]
    out["state_source"] = out["data_source"].astype(str) + "|derived_limited_proxy_state_v3_57"
    out["diagnostic_scope"] = "limited_window_smoke_only"
    out["model_promotion_allowed"] = False
    out["performance_claim_allowed"] = False
    columns = [
        "date",
        "asset_or_index",
        "level",
        "daily_return",
        "trend_return",
        "realized_volatility",
        "drawdown_to_trailing_max",
        "history_available",
        *STATE_COLUMNS,
        "state_available_date",
        "state_source",
        "diagnostic_scope",
        "model_promotion_allowed",
        "performance_claim_allowed",
    ]
    return out[columns]


def build_state_coverage(state_panel: pd.DataFrame, signals: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    signal_dates = pd.DataFrame({"date": normalize_date_series(signals["signal_date"]).drop_duplicates()})
    merged = signal_dates.merge(state_panel, on="date", how="left")
    rows: list[dict[str, Any]] = []
    total_dates = max(int(signal_dates["date"].nunique()), 1)
    for column in STATE_COLUMNS:
        counts = merged.groupby(column, dropna=False)["date"].nunique().reset_index(name="signal_dates")
        for row in counts.itertuples(index=False):
            state_value = str(getattr(row, column))
            dates = int(row.signal_dates)
            if "insufficient" in state_value:
                diagnostic_role = "monitor_only_insufficient_history"
            elif dates >= config.min_state_bucket_signal_dates:
                diagnostic_role = "usable_smoke_bucket"
            else:
                diagnostic_role = "monitor_only_low_sample"
            rows.append(
                {
                    "state_column": column,
                    "state_value": state_value,
                    "signal_dates": dates,
                    "share_of_signal_dates": dates / total_dates,
                    "history_available_signal_dates": int(
                        merged.loc[(merged[column].astype(str) == state_value) & (merged["history_available"].fillna(False).astype(bool)), "date"].nunique()
                    ),
                    "diagnostic_role": diagnostic_role,
                }
            )
    return pd.DataFrame(rows).sort_values(["state_column", "signal_dates", "state_value"], ascending=[True, False, True])


def build_state_transition_summary(state_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    ordered = state_panel.sort_values("date").copy()
    for column in STATE_COLUMNS:
        tmp = ordered[["date", column]].copy()
        tmp["state_column"] = column
        tmp["prior_state_value"] = tmp[column].shift(1).fillna("start")
        tmp["state_value"] = tmp[column].astype(str)
        counts = tmp.groupby(["state_column", "prior_state_value", "state_value"], dropna=False).size().reset_index(name="transition_count")
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_label_contract_diagnostics(labels: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    out = labels.copy()
    out["signal_date"] = normalize_date_series(out["signal_date"])
    out["forward_adjusted_return"] = _finite_numeric(out["forward_adjusted_return"])
    rows: list[dict[str, Any]] = []
    for horizon, group in out.groupby("horizon", dropna=False):
        values = group["forward_adjusted_return"].dropna()
        rows.append(
            {
                "horizon": int(horizon),
                "label_rows": int(len(group)),
                "unique_signal_dates": int(group["signal_date"].nunique()),
                "finite_label_rows": int(values.shape[0]),
                "nonfinite_label_rows": int(len(group) - values.shape[0]),
                "return_min": float(values.min()) if not values.empty else np.nan,
                "return_p01": float(values.quantile(0.01)) if not values.empty else np.nan,
                "return_p99": float(values.quantile(0.99)) if not values.empty else np.nan,
                "return_max": float(values.max()) if not values.empty else np.nan,
                "abs_return_p99": float(values.abs().quantile(0.99)) if not values.empty else np.nan,
                "abs_outlier_rows": int((values.abs() > config.label_abs_outlier_threshold).sum()),
                "diagnostic_type": "label_distribution_sanity_only",
            }
        )
    return pd.DataFrame(rows).sort_values("horizon")


def build_joined_contract_panel(signals: pd.DataFrame, labels: pd.DataFrame, state_panel: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [
        "signal_id",
        "signal_date",
        "asset",
        "signal_value",
        "signal_direction",
        "available_date",
        "candidate_status",
        "signal_family",
        "formula_name",
        "composite_state",
        "liquidity_state",
        "breadth_state",
        "activity_state",
        "concentration_state",
        "limit_crowding_state",
    ]
    keep_signal_cols = [col for col in signal_cols if col in signals.columns]
    sig = signals[keep_signal_cols].copy()
    sig["signal_date"] = normalize_date_series(sig["signal_date"])
    sig["available_date"] = normalize_date_series(sig["available_date"])
    lab = labels.copy()
    lab["signal_date"] = normalize_date_series(lab["signal_date"])
    lab["label_available_date"] = normalize_date_series(lab["label_available_date"])
    states = state_panel[["date", "history_available", *STATE_COLUMNS, "state_available_date", "diagnostic_scope"]].copy()
    joined = sig.merge(lab, on=["signal_date", "asset"], how="inner").merge(states, left_on="signal_date", right_on="date", how="left")
    joined["state_joined"] = joined["proxy_market_state"].notna()
    joined["diagnostic_usage"] = "contract_join_smoke_only_no_performance_claim"
    return joined.sort_values(["signal_date", "horizon", "signal_id"]).reset_index(drop=True)


def build_joined_contract_coverage(joined: pd.DataFrame, config: LimitedStateDiagnosticConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "state_column",
                "state_value",
                "joined_rows",
                "unique_signal_dates",
                "unique_signal_ids",
                "finite_label_rows",
                "history_available_rows",
                "coverage_role",
            ]
        )
    joined = joined.copy()
    joined["forward_adjusted_return"] = _finite_numeric(joined["forward_adjusted_return"])
    for horizon in sorted(pd.to_numeric(joined["horizon"], errors="coerce").dropna().astype(int).unique()):
        horizon_frame = joined.loc[pd.to_numeric(joined["horizon"], errors="coerce").astype("Int64") == horizon]
        for column in STATE_COLUMNS:
            for state_value, group in horizon_frame.groupby(column, dropna=False):
                rows.append(
                    {
                        "horizon": int(horizon),
                        "state_column": column,
                        "state_value": str(state_value),
                        "joined_rows": int(len(group)),
                        "unique_signal_dates": int(group["signal_date"].nunique()),
                        "unique_signal_ids": int(group["signal_id"].nunique()) if "signal_id" in group.columns else 0,
                        "finite_label_rows": int(group["forward_adjusted_return"].notna().sum()),
                        "history_available_rows": int(group["history_available"].fillna(False).astype(bool).sum()),
                        "coverage_role": "coverage_only_not_performance",
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon", "state_column", "joined_rows"], ascending=[True, True, False])


def build_readiness_checks(
    source_checks: pd.DataFrame,
    signal_checks: pd.DataFrame,
    label_checks: pd.DataFrame,
    state_panel: pd.DataFrame,
    signals: pd.DataFrame,
    joined: pd.DataFrame,
    v3_56_manifest: dict[str, Any],
    config: LimitedStateDiagnosticConfig,
) -> pd.DataFrame:
    non_null_state_dates = int(
        state_panel.loc[state_panel["date"].isin(set(normalize_date_series(signals["signal_date"]))), "proxy_market_state"].notna().sum()
    )
    total_signal_dates = int(normalize_date_series(signals["signal_date"]).nunique())
    state_date_coverage = non_null_state_dates / max(total_signal_dates, 1)
    history_available_dates = int(
        state_panel.loc[
            state_panel["date"].isin(set(normalize_date_series(signals["signal_date"]))) & state_panel["history_available"].astype(bool),
            "date",
        ].nunique()
    )
    history_ratio = history_available_dates / max(total_signal_dates, 1)
    rows = [
        {
            "check": "v3_56_manifest_acceptance_passed",
            "status": _status(bool(v3_56_manifest.get("acceptance_pass")) and bool(v3_56_manifest.get("self_check_pass"))),
            "detail": f"acceptance={v3_56_manifest.get('acceptance_pass')};self_check={v3_56_manifest.get('self_check_pass')}",
        },
        {
            "check": "source_checks_passed",
            "status": _status(source_checks["status"].eq("pass").all()),
            "detail": ";".join(source_checks.loc[source_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "signal_checks_passed",
            "status": _status(signal_checks["status"].eq("pass").all()),
            "detail": ";".join(signal_checks.loc[signal_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "label_checks_passed",
            "status": _status(label_checks["status"].eq("pass").all()),
            "detail": ";".join(label_checks.loc[label_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "state_panel_built",
            "status": _status(len(state_panel) >= config.min_source_rows),
            "detail": f"rows={len(state_panel)};min={config.min_source_rows}",
        },
        {
            "check": "state_date_coverage_passed",
            "status": _status(state_date_coverage >= config.min_state_date_coverage_ratio),
            "detail": f"coverage={state_date_coverage:.4f};min={config.min_state_date_coverage_ratio:.4f}",
        },
        {
            "check": "history_available_signal_date_ratio_passed",
            "status": _status(history_ratio >= config.min_history_available_signal_date_ratio),
            "detail": f"history_ratio={history_ratio:.4f};min={config.min_history_available_signal_date_ratio:.4f}",
        },
        {
            "check": "joined_contract_panel_has_rows",
            "status": _status(len(joined) >= config.min_joined_rows),
            "detail": f"rows={len(joined)};min={config.min_joined_rows}",
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "blocked",
            "detail": "limited-window proxy smoke test only; no IC, hit-rate, backtest, or model promotion",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "limited_state_coverage_diagnostic",
                "produced": True,
                "blocked": False,
                "reason": "coverage and schema smoke checks only",
            },
            {
                "result_type": "limited_joined_contract_panel",
                "produced": True,
                "blocked": False,
                "reason": "joins signals, labels, and states without ranking or scoring signals",
            },
            {
                "result_type": "state_stratified_signal_performance",
                "produced": False,
                "blocked": True,
                "reason": "short-window adjusted proxy cannot support IC, hit-rate, or effect-size claims",
            },
            {
                "result_type": "portfolio_backtest_or_model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "V3.57 is smoke validation only",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    guard: pd.DataFrame,
    state_panel: pd.DataFrame,
    joined: pd.DataFrame,
    official_source_exists: bool,
    output_column_names: list[str],
) -> pd.DataFrame:
    unexpected_blocked = readiness.loc[
        (readiness["status"] == "blocked") & (~readiness["check"].isin(["performance_validation_allowed_now"]))
    ]
    forbidden_metric_columns = sorted({term for term in FORBIDDEN_PERFORMANCE_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(["state_stratified_signal_performance", "portfolio_backtest_or_model_promotion"]),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "all_non_performance_readiness_checks_passed",
                "status": "pass" if unexpected_blocked.empty else "fail",
                "detail": ";".join(unexpected_blocked["check"].astype(str)),
            },
            {
                "check": "state_panel_has_limited_scope_flags",
                "status": "pass"
                if len(state_panel) > 0
                and state_panel["diagnostic_scope"].astype(str).eq("limited_window_smoke_only").all()
                and not state_panel["model_promotion_allowed"].astype(bool).any()
                and not state_panel["performance_claim_allowed"].astype(bool).any()
                else "fail",
                "detail": f"rows={len(state_panel)}",
            },
            {
                "check": "joined_panel_has_state_and_label_contract",
                "status": "pass"
                if len(joined) > 0
                and joined["state_joined"].astype(bool).all()
                and joined["forward_adjusted_return"].notna().all()
                else "fail",
                "detail": f"joined_rows={len(joined)}",
            },
            {
                "check": "performance_outputs_not_produced",
                "status": "pass" if not forbidden_produced and not forbidden_metric_columns else "fail",
                "detail": ",".join(forbidden_metric_columns),
            },
            {
                "check": "official_market_total_return_file_not_created",
                "status": "pass" if not official_source_exists else "fail",
                "detail": "data_raw/market_labels/market_total_return_index.csv",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    rendered = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rendered.append("| " + " | ".join(values) + " |")
    return rendered


def build_report(
    source: pd.DataFrame,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    state_panel: pd.DataFrame,
    state_coverage: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    joined: pd.DataFrame,
    joined_coverage: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    lines = [
        "# V3.57 Limited Proxy State Diagnostic",
        "",
        "## Decision",
        "",
        "- V3.57 is a short-window smoke diagnostic using the V3.56 limited adjusted proxy labels.",
        "- It builds proxy market states and joins them to the limited signal and label contracts.",
        "- It does not run IC, hit rate, effect-size tests, portfolio NAV, Sharpe, max-drawdown metrics, or model promotion.",
        "",
        "## Input Scope",
        "",
        f"- Source rows: `{len(source)}`",
        f"- Signal rows: `{len(signals)}`",
        f"- Label rows: `{len(labels)}`",
        f"- State rows: `{len(state_panel)}`",
        f"- Joined contract rows: `{len(joined)}`",
        "",
        "## State Coverage",
        "",
    ]
    lines.extend(markdown_table(state_coverage, ["state_column", "state_value", "signal_dates", "share_of_signal_dates", "diagnostic_role"]))
    lines.extend(
        [
            "",
            "## Label Sanity",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            label_diagnostics,
            ["horizon", "label_rows", "unique_signal_dates", "finite_label_rows", "return_min", "return_p99", "return_max", "abs_outlier_rows"],
        )
    )
    lines.extend(
        [
            "",
            "## Joined Coverage",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            joined_coverage,
            ["horizon", "state_column", "state_value", "joined_rows", "unique_signal_dates", "unique_signal_ids", "coverage_role"],
        )
    )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
        ]
    )
    lines.extend(markdown_table(readiness, ["check", "status", "detail"], max_rows=20))
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
        ]
    )
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=20))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Use V3.57 to verify whether the short-window data plumbing can support future state-stratified validation.",
            "- Do not treat any V3.57 distribution table as strategy performance evidence.",
            "- A longer point-in-time total-return or explicitly approved adjusted proxy is still required before real validation.",
        ]
    )
    return "\n".join(lines)


def build_catalog(state_panel: pd.DataFrame, joined: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# A-share Limited Proxy State Diagnostic V3.57",
            "",
            "## Dataset Role",
            "",
            "V3.57 derives limited-window proxy states from the V3.56 JoinQuant adjusted proxy and joins them to limited MARKET signal labels.",
            "",
            "## Governance",
            "",
            "- Scope: limited-window smoke diagnostic only.",
            "- Allowed use: schema, availability, state coverage, and join diagnostics.",
            "- Forbidden use: IC, hit-rate, backtest, NAV, Sharpe, drawdown, or model promotion.",
            "- Official source file remains absent unless separately acquired and validated.",
            "",
            "## Produced Shapes",
            "",
            f"- proxy_state_panel rows: `{len(state_panel)}`",
            f"- joined_contract_panel rows: `{len(joined)}`",
        ]
    )
