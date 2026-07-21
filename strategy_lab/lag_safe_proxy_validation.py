"""Lag-safe proxy validation and artifact screening for HIRSSM V3.63.

This module pairs the V3.62 repaired non-price signal panel with V3.59
price-index proxy labels. It is a guarded diagnostic step: no official
total-return evidence, portfolio backtest, or default model promotion is
produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS, _corr, normalize_date


STATE_COLUMNS = [
    "composite_state",
    "liquidity_state",
    "breadth_state",
    "activity_state",
    "concentration_state",
    "limit_crowding_state",
]


@dataclass(frozen=True)
class LagSafeProxyValidationConfig:
    repaired_signal_panel_path: Path
    price_proxy_label_path: Path
    market_proxy_source_path: Path
    v3_62_manifest_path: Path
    v3_59_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    state_columns: tuple[str, ...]
    min_joined_rows: int
    min_signal_observations: int
    min_state_observations: int
    top_quantile: float
    bottom_quantile: float
    min_abs_proxy_spearman: float
    min_proxy_qspread: float
    min_top_directional_alignment: float
    negative_control_shift: int
    min_control_degradation: float
    max_abs_same_day_corr: float
    max_same_day_corr_ratio: float
    max_future_lead_corr_excess: float
    min_primary_state_support: int
    max_bull_state_share: float
    trend_window: int
    bull_return_threshold: float
    bear_return_threshold: float


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_div(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def validate_repaired_signal_panel(signals: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    required = {
        "signal_id",
        "signal_date",
        "source_trade_date",
        "asset",
        "signal_value",
        "signal_direction",
        "available_date",
        "trade_lag_steps",
        "source_component_type",
        "model_promotion_allowed",
        "performance_claim_allowed",
        "portfolio_backtest_allowed",
        "official_total_return_evidence",
        *config.state_columns,
    }
    missing = sorted(required.difference(signals.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "signal_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        }
    ]
    if missing:
        return pd.DataFrame(rows)

    signal_dates = normalize_date(signals["signal_date"])
    source_dates = normalize_date(signals["source_trade_date"])
    available = normalize_date(signals["available_date"])
    assets = set(signals["asset"].astype(str).unique())
    source_types = set(signals["source_component_type"].astype(str).unique())
    model_allowed = _bool_series(signals["model_promotion_allowed"])
    performance_allowed = _bool_series(signals["performance_claim_allowed"])
    portfolio_allowed = _bool_series(signals["portfolio_backtest_allowed"])
    official = _bool_series(signals["official_total_return_evidence"])
    rows.extend(
        [
            {
                "check": "signal_dates_parseable",
                "status": _status(signal_dates.notna().all() and source_dates.notna().all() and available.notna().all()),
                "detail": f"bad_signal={int(signal_dates.isna().sum())};bad_source={int(source_dates.isna().sum())};bad_available={int(available.isna().sum())}",
            },
            {
                "check": "source_trade_date_before_signal_date",
                "status": _status((source_dates < signal_dates).all()),
                "detail": f"bad_rows={int((source_dates >= signal_dates).sum())}",
            },
            {
                "check": "available_date_equals_signal_date",
                "status": _status((available == signal_dates).all()),
                "detail": f"bad_rows={int((available != signal_dates).sum())}",
            },
            {
                "check": "trade_lag_steps_one",
                "status": _status(set(pd.to_numeric(signals["trade_lag_steps"], errors="coerce").dropna().astype(int)) == {1}),
                "detail": ",".join(map(str, sorted(pd.to_numeric(signals["trade_lag_steps"], errors="coerce").dropna().astype(int).unique()))),
            },
            {
                "check": "signal_scope_is_market_only",
                "status": _status(assets == {"MARKET"}),
                "detail": ",".join(sorted(assets)),
            },
            {
                "check": "source_components_non_price_only",
                "status": _status(source_types == {"non_price_only"}),
                "detail": ",".join(sorted(source_types)),
            },
            {
                "check": "signal_no_promotion_flags",
                "status": _status(not model_allowed.any() and not performance_allowed.any() and not portfolio_allowed.any() and not official.any()),
                "detail": f"model={bool(model_allowed.any())};performance={bool(performance_allowed.any())};portfolio={bool(portfolio_allowed.any())};official={bool(official.any())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def validate_proxy_labels(labels: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    required = {
        "signal_date",
        "asset",
        "horizon",
        "forward_price_index_return",
        "return_basis",
        "label_available_date",
        "official_total_return",
        "proxy_label_generation_allowed",
        "official_label_generation_allowed",
        "model_promotion_allowed",
        "performance_claim_allowed",
        "diagnostic_usage",
    }
    missing = sorted(required.difference(labels.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "label_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        }
    ]
    if missing:
        return pd.DataFrame(rows)

    signal_dates = normalize_date(labels["signal_date"])
    label_dates = normalize_date(labels["label_available_date"])
    returns = pd.to_numeric(labels["forward_price_index_return"], errors="coerce")
    basis = set(labels["return_basis"].astype(str).unique())
    horizons = set(pd.to_numeric(labels["horizon"], errors="coerce").dropna().astype(int))
    official = _bool_series(labels["official_total_return"])
    proxy_allowed = _bool_series(labels["proxy_label_generation_allowed"])
    official_allowed = _bool_series(labels["official_label_generation_allowed"])
    model_allowed = _bool_series(labels["model_promotion_allowed"])
    performance_allowed = _bool_series(labels["performance_claim_allowed"])
    usage = set(labels["diagnostic_usage"].astype(str).unique())
    rows.extend(
        [
            {
                "check": "label_dates_parseable",
                "status": _status(signal_dates.notna().all() and label_dates.notna().all()),
                "detail": f"bad_signal={int(signal_dates.isna().sum())};bad_label={int(label_dates.isna().sum())}",
            },
            {
                "check": "label_available_date_after_signal_date",
                "status": _status((label_dates > signal_dates).all()),
                "detail": f"bad_rows={int((label_dates <= signal_dates).sum())}",
            },
            {
                "check": "label_return_basis_is_price_proxy",
                "status": _status(basis == {"price_index_return"}),
                "detail": ",".join(sorted(basis)),
            },
            {
                "check": "label_horizons_match_config",
                "status": _status(horizons == set(config.horizons)),
                "detail": f"actual={sorted(horizons)};expected={sorted(config.horizons)}",
            },
            {
                "check": "label_returns_finite",
                "status": _status(returns.notna().all() and np.isfinite(returns).all()),
                "detail": f"bad_rows={int((returns.isna() | ~np.isfinite(returns)).sum())}",
            },
            {
                "check": "label_proxy_only_no_promotion",
                "status": _status(proxy_allowed.all() and not official.any() and not official_allowed.any() and not model_allowed.any() and not performance_allowed.any()),
                "detail": f"official={bool(official.any())};model={bool(model_allowed.any())};performance={bool(performance_allowed.any())}",
            },
            {
                "check": "label_usage_non_official",
                "status": _status(usage == {"non_official_price_proxy_label_only"}),
                "detail": ",".join(sorted(usage)),
            },
        ]
    )
    return pd.DataFrame(rows)


def load_market_context(source: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    data = source.copy()
    data["signal_date"] = normalize_date(data["date"])
    data["market_close"] = pd.to_numeric(data.get("close", data.get("market_level")), errors="coerce")
    data["same_day_price_return"] = pd.to_numeric(data.get("pct_chg"), errors="coerce") / 100.0
    if data["same_day_price_return"].isna().all():
        data["same_day_price_return"] = data["market_close"].pct_change()
    data["prev_day_price_return"] = data["same_day_price_return"].shift(1)
    data["rolling_trend_return"] = data["market_close"] / data["market_close"].shift(config.trend_window) - 1.0
    data["market_trend_state"] = "range_or_transition"
    data.loc[data["rolling_trend_return"] >= config.bull_return_threshold, "market_trend_state"] = "bull_price_proxy_state"
    data.loc[data["rolling_trend_return"] <= config.bear_return_threshold, "market_trend_state"] = "bear_price_proxy_state"
    keep = [
        "signal_date",
        "same_day_price_return",
        "prev_day_price_return",
        "rolling_trend_return",
        "market_trend_state",
    ]
    return data.loc[:, keep].dropna(subset=["signal_date"]).drop_duplicates("signal_date")


def build_joined_panel(signals: pd.DataFrame, labels: pd.DataFrame, market_context: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    signal_cols = [
        "signal_id",
        "signal_date",
        "source_trade_date",
        "asset",
        "signal_value",
        "signal_direction",
        "available_date",
        "trade_lag_steps",
        "signal_lag_rule",
        "source_component_type",
        "source_columns",
        "candidate_status",
        "signal_family",
        "formula_name",
        "repair_source",
        "state_reference_date",
        "stratification_state_source",
        "data_quality_state",
        "model_promotion_allowed",
        "performance_claim_allowed",
        "portfolio_backtest_allowed",
        "official_total_return_evidence",
        *config.state_columns,
    ]
    keep_cols = [col for col in signal_cols if col in signals.columns]
    sig = signals.loc[signals["asset"].astype(str) == "MARKET", keep_cols].copy()
    sig["signal_date"] = normalize_date(sig["signal_date"])
    sig["source_trade_date"] = normalize_date(sig["source_trade_date"])
    sig["available_date"] = normalize_date(sig["available_date"])
    sig["signal_value"] = pd.to_numeric(sig["signal_value"], errors="coerce")
    lab = labels.copy()
    lab["signal_date"] = normalize_date(lab["signal_date"])
    lab["horizon"] = pd.to_numeric(lab["horizon"], errors="coerce").astype("Int64")
    lab["forward_price_index_return"] = pd.to_numeric(lab["forward_price_index_return"], errors="coerce")
    joined = sig.merge(lab, on=["signal_date", "asset"], how="inner", suffixes=("_signal", "_label"))
    direction = joined["signal_direction"].astype(str).str.lower().map({"positive": 1.0, "negative": -1.0}).fillna(0.0)
    joined["direction_multiplier"] = direction
    joined["expected_signed_proxy_return"] = joined["forward_price_index_return"] * joined["direction_multiplier"]
    joined["active_directional_alignment"] = joined["expected_signed_proxy_return"] > 0
    joined = joined.merge(market_context, on="signal_date", how="left")
    joined["same_day_return_signed"] = joined["same_day_price_return"] * joined["direction_multiplier"]
    joined["prev_day_return_signed"] = joined["prev_day_price_return"] * joined["direction_multiplier"]
    joined = joined.sort_values(["signal_id", "horizon", "signal_date"]).reset_index(drop=True)
    joined["signal_prev_observation"] = joined.groupby(["signal_id", "horizon"])["signal_value"].shift(1)
    joined["signal_next_observation_for_screen"] = joined.groupby(["signal_id", "horizon"])["signal_value"].shift(-1)
    joined["validation_scope"] = "lag_safe_price_proxy_validation_only"
    joined["default_model_allowed"] = False
    joined["portfolio_backtest_allowed"] = False
    joined["official_total_return_evidence"] = False
    return joined


def _bucket_summary(group: pd.DataFrame, config: LagSafeProxyValidationConfig) -> dict[str, Any]:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    observations = int(len(clean))
    if observations == 0:
        return {
            "observations": 0,
            "unique_signal_dates": 0,
            "proxy_pearson_corr": np.nan,
            "proxy_spearman_corr": np.nan,
            "expected_signed_proxy_return_mean": np.nan,
            "directional_alignment_share": np.nan,
            "bottom_bucket_expected_signed_return_mean": np.nan,
            "top_bucket_expected_signed_return_mean": np.nan,
            "proxy_qspread_top_minus_bottom": np.nan,
            "top_bucket_directional_alignment_share": np.nan,
            "top_bucket_rows": 0,
            "bottom_bucket_rows": 0,
        }
    top_cut = clean["signal_value"].quantile(config.top_quantile)
    bottom_cut = clean["signal_value"].quantile(config.bottom_quantile)
    top = clean.loc[clean["signal_value"] >= top_cut]
    bottom = clean.loc[clean["signal_value"] <= bottom_cut]
    top_mean = float(top["expected_signed_proxy_return"].mean()) if not top.empty else np.nan
    bottom_mean = float(bottom["expected_signed_proxy_return"].mean()) if not bottom.empty else np.nan
    return {
        "observations": observations,
        "unique_signal_dates": int(clean["signal_date"].nunique()),
        "proxy_pearson_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "pearson"),
        "proxy_spearman_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "spearman"),
        "expected_signed_proxy_return_mean": float(clean["expected_signed_proxy_return"].mean()),
        "directional_alignment_share": float(clean["active_directional_alignment"].mean()),
        "bottom_bucket_expected_signed_return_mean": bottom_mean,
        "top_bucket_expected_signed_return_mean": top_mean,
        "proxy_qspread_top_minus_bottom": top_mean - bottom_mean if pd.notna(top_mean) and pd.notna(bottom_mean) else np.nan,
        "top_bucket_directional_alignment_share": float(top["active_directional_alignment"].mean()) if not top.empty else np.nan,
        "top_bucket_rows": int(len(top)),
        "bottom_bucket_rows": int(len(bottom)),
    }


def build_signal_validation_summary(joined: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in joined.groupby(["signal_id", "horizon"], dropna=False):
        summary = _bucket_summary(group, config)
        status = "proxy_observation_only"
        if (
            summary["observations"] >= config.min_signal_observations
            and pd.notna(summary["proxy_spearman_corr"])
            and abs(summary["proxy_spearman_corr"]) >= config.min_abs_proxy_spearman
            and pd.notna(summary["proxy_qspread_top_minus_bottom"])
            and summary["proxy_qspread_top_minus_bottom"] > config.min_proxy_qspread
            and pd.notna(summary["top_bucket_directional_alignment_share"])
            and summary["top_bucket_directional_alignment_share"] >= config.min_top_directional_alignment
        ):
            status = "proxy_positive_observation"
        first = group.iloc[0]
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "signal_direction": first.get("signal_direction", ""),
                "signal_family": first.get("signal_family", ""),
                "source_component_type": first.get("source_component_type", ""),
                "repair_source": first.get("repair_source", ""),
                **summary,
                "proxy_evidence_status": status,
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "proxy_evidence_status", "proxy_spearman_corr"], ascending=[True, False, False]).reset_index(drop=True)


def build_state_stratified_summary(joined: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in config.state_columns:
        for (signal_id, horizon, state_value), group in joined.groupby(["signal_id", "horizon", column], dropna=False):
            summary = _bucket_summary(group, config)
            rows.append(
                {
                    "state_column": column,
                    "state_value": str(state_value),
                    "signal_id": signal_id,
                    "horizon": int(horizon),
                    **summary,
                    "validation_role": "primary_proxy_stratum" if summary["observations"] >= config.min_state_observations else "monitor_only_low_sample",
                    "default_model_allowed": False,
                    "official_total_return_evidence": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "state_column", "observations"], ascending=[True, True, False]).reset_index(drop=True)


def build_negative_control_summary(joined: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    control = joined.sort_values(["signal_id", "horizon", "signal_date"]).copy()
    control["control_expected_signed_proxy_return"] = control.groupby(["signal_id", "horizon"])["expected_signed_proxy_return"].shift(config.negative_control_shift)
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in control.groupby(["signal_id", "horizon"], dropna=False):
        real = _bucket_summary(group, config)
        clean = group.dropna(subset=["signal_value", "control_expected_signed_proxy_return"]).copy()
        control_corr = _corr(clean["signal_value"], clean["control_expected_signed_proxy_return"], "spearman")
        degradation = (
            abs(real["proxy_spearman_corr"]) - abs(control_corr)
            if pd.notna(real["proxy_spearman_corr"]) and pd.notna(control_corr)
            else np.nan
        )
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "real_proxy_spearman_corr": real["proxy_spearman_corr"],
                "lag_broken_control_spearman_corr": control_corr,
                "abs_corr_degradation": degradation,
                "negative_control_degraded": bool(pd.notna(degradation) and degradation >= config.min_control_degradation),
                "negative_control_artifact_flag": bool(pd.isna(degradation) or degradation < config.min_control_degradation),
                "control_shift_rows": config.negative_control_shift,
                "control_role": "diagnostic_only_not_gate",
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_temporal_artifact_audit(joined: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in joined.groupby(["signal_id", "horizon"], dropna=False):
        real_corr = _corr(group["signal_value"], group["expected_signed_proxy_return"], "spearman")
        prev_corr = _corr(group["signal_prev_observation"], group["expected_signed_proxy_return"], "spearman")
        next_corr = _corr(group["signal_next_observation_for_screen"], group["expected_signed_proxy_return"], "spearman")
        same_day_corr = _corr(group["signal_value"], group["same_day_return_signed"], "spearman")
        prior_market_corr = _corr(group["signal_value"], group["prev_day_return_signed"], "spearman")
        same_day_ratio = _safe_div(abs(same_day_corr), abs(real_corr))
        next_excess = abs(next_corr) - abs(real_corr) if pd.notna(next_corr) and pd.notna(real_corr) else np.nan
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "observations": int(len(group)),
                "real_proxy_spearman_corr": real_corr,
                "previous_signal_proxy_spearman_corr": prev_corr,
                "next_signal_proxy_spearman_corr": next_corr,
                "same_day_price_signed_spearman_corr": same_day_corr,
                "prior_day_price_signed_spearman_corr": prior_market_corr,
                "same_day_to_forward_abs_corr_ratio": same_day_ratio,
                "next_signal_abs_corr_excess": next_excess,
                "same_day_artifact_flag": bool(
                    pd.notna(same_day_corr)
                    and abs(same_day_corr) > config.max_abs_same_day_corr
                    and (pd.isna(same_day_ratio) or same_day_ratio > config.max_same_day_corr_ratio)
                ),
                "future_signal_artifact_flag": bool(pd.notna(next_excess) and next_excess > config.max_future_lead_corr_excess),
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_state_dependence_audit(state_summary: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    data = state_summary.loc[state_summary["validation_role"].astype(str).eq("primary_proxy_stratum")].copy()
    data["proxy_spearman_corr"] = pd.to_numeric(data["proxy_spearman_corr"], errors="coerce")
    data["proxy_qspread_top_minus_bottom"] = pd.to_numeric(data["proxy_qspread_top_minus_bottom"], errors="coerce")
    data["top_bucket_directional_alignment_share"] = pd.to_numeric(data["top_bucket_directional_alignment_share"], errors="coerce")
    data["observations"] = pd.to_numeric(data["observations"], errors="coerce")
    data["supportive_state"] = (
        (data["proxy_spearman_corr"] >= config.min_abs_proxy_spearman)
        & (data["proxy_qspread_top_minus_bottom"] > config.min_proxy_qspread)
        & (data["top_bucket_directional_alignment_share"] >= config.min_top_directional_alignment)
    )
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in data.groupby(["signal_id", "horizon"], dropna=False):
        composite = group.loc[group["state_column"].eq("composite_state")].copy()
        bull_mask = composite["state_value"].astype(str).str.contains("risk_on|breadth_recovery|liquidity", case=False, regex=True)
        bull_obs = float(composite.loc[bull_mask & composite["supportive_state"], "observations"].sum())
        support_obs = float(composite.loc[composite["supportive_state"], "observations"].sum())
        bull_share = _safe_div(bull_obs, support_obs)
        column_support = group.groupby("state_column")["supportive_state"].sum().reset_index(name="support_count")
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "primary_state_rows": int(len(group)),
                "supportive_primary_state_rows": int(group["supportive_state"].sum()),
                "state_columns_with_support": int((column_support["support_count"] > 0).sum()),
                "composite_supportive_state_rows": int(composite["supportive_state"].sum()) if not composite.empty else 0,
                "composite_bull_support_observation_share": bull_share,
                "state_support_too_sparse_flag": bool(group["supportive_state"].sum() < config.min_primary_state_support),
                "bull_state_proxy_flag": bool(pd.notna(bull_share) and bull_share > config.max_bull_state_share),
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_market_trend_audit(joined: pd.DataFrame, config: LagSafeProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon, trend_state), group in joined.groupby(["signal_id", "horizon", "market_trend_state"], dropna=False):
        bucket = _bucket_summary(group, config)
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "market_trend_state": str(trend_state),
                "observations": bucket["observations"],
                "proxy_spearman_corr": bucket["proxy_spearman_corr"],
                "proxy_qspread_top_minus_bottom": bucket["proxy_qspread_top_minus_bottom"],
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    total = result.groupby(["signal_id", "horizon"])["observations"].transform("sum")
    result["observation_share"] = result["observations"] / total
    return result.sort_values(["horizon", "signal_id", "market_trend_state"]).reset_index(drop=True)


def build_candidate_decision(
    summary: pd.DataFrame,
    temporal: pd.DataFrame,
    controls: pd.DataFrame,
    state_audit: pd.DataFrame,
) -> pd.DataFrame:
    merged = summary.merge(
        temporal.loc[
            :,
            [
                "signal_id",
                "horizon",
                "same_day_artifact_flag",
                "future_signal_artifact_flag",
                "same_day_price_signed_spearman_corr",
                "next_signal_abs_corr_excess",
            ],
        ],
        on=["signal_id", "horizon"],
        how="left",
    ).merge(
        controls.loc[:, ["signal_id", "horizon", "abs_corr_degradation", "negative_control_artifact_flag"]],
        on=["signal_id", "horizon"],
        how="left",
    ).merge(
        state_audit.loc[
            :,
            [
                "signal_id",
                "horizon",
                "supportive_primary_state_rows",
                "state_columns_with_support",
                "state_support_too_sparse_flag",
                "bull_state_proxy_flag",
            ],
        ],
        on=["signal_id", "horizon"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        flags = []
        if row.proxy_evidence_status != "proxy_positive_observation":
            status = "no_proxy_edge_observed"
            next_action = "keep_observation_or_retire_if_repeatedly_null"
        else:
            if bool(row.same_day_artifact_flag):
                flags.append("same_day_price_artifact_risk")
            if bool(row.future_signal_artifact_flag):
                flags.append("future_signal_serial_state_risk")
            if bool(row.negative_control_artifact_flag):
                flags.append("negative_control_not_degraded")
            if bool(row.state_support_too_sparse_flag):
                flags.append("state_support_too_sparse")
            if bool(row.bull_state_proxy_flag):
                flags.append("bull_state_proxy_dependence")
            if flags:
                status = "artifact_risk_blocks_escalation"
                next_action = "do_not_escalate_until_signal_or_label_source_repaired"
            else:
                status = "survives_for_walk_forward_proxy_review"
                next_action = "carry_to_narrow_walk_forward_proxy_review_only"
        rows.append(
            {
                "signal_id": row.signal_id,
                "horizon": int(row.horizon),
                "proxy_evidence_status": row.proxy_evidence_status,
                "artifact_review_status": status,
                "artifact_flags": ";".join(flags) if flags else "none",
                "next_action": next_action,
                "default_model_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "artifact_review_status", "signal_id"]).reset_index(drop=True)


def build_readiness_checks(
    signal_checks: pd.DataFrame,
    label_checks: pd.DataFrame,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    v3_62_manifest: dict[str, Any],
    v3_59_manifest: dict[str, Any],
    config: LagSafeProxyValidationConfig,
) -> pd.DataFrame:
    rows = [
        {
            "check": "v3_62_manifest_accepted",
            "status": _status(bool(v3_62_manifest.get("self_check_pass"))),
            "detail": f"self_check={v3_62_manifest.get('self_check_pass')}",
        },
        {
            "check": "v3_59_manifest_accepted",
            "status": _status(bool(v3_59_manifest.get("self_check_pass")) and bool(v3_59_manifest.get("metrics", {}).get("price_proxy_labels_written"))),
            "detail": f"self_check={v3_59_manifest.get('self_check_pass')};labels_written={v3_59_manifest.get('metrics', {}).get('price_proxy_labels_written')}",
        },
        {
            "check": "signal_contract_checks_passed",
            "status": _status(signal_checks["status"].eq("pass").all()),
            "detail": ";".join(signal_checks.loc[signal_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "price_proxy_label_checks_passed",
            "status": _status(label_checks["status"].eq("pass").all()),
            "detail": ";".join(label_checks.loc[label_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "joined_panel_has_minimum_rows",
            "status": _status(len(joined) >= config.min_joined_rows),
            "detail": f"rows={len(joined)};min={config.min_joined_rows}",
        },
        {
            "check": "proxy_signal_summary_produced",
            "status": _status(not summary.empty),
            "detail": f"rows={len(summary)}",
        },
        {
            "check": "portfolio_or_model_promotion_allowed_now",
            "status": "blocked",
            "detail": "V3.63 is proxy diagnostics only",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard(summary: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    survivors = int(decisions["artifact_review_status"].astype(str).eq("survives_for_walk_forward_proxy_review").sum()) if not decisions.empty else 0
    return pd.DataFrame(
        [
            {
                "result_type": "lag_safe_price_proxy_validation",
                "produced": not summary.empty,
                "blocked": summary.empty,
                "reason": "non-official price-index proxy signal diagnostics",
            },
            {
                "result_type": "walk_forward_queue_candidate",
                "produced": survivors > 0,
                "blocked": survivors == 0,
                "reason": "candidate queue only; not model promotion",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.63 does not use official total-return labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.63 does not create positions, trades, NAV, Sharpe, or drawdown",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "proxy diagnostics cannot promote default model",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    unexpected = readiness.loc[
        (readiness["status"] != "pass") & (~readiness["check"].isin(["portfolio_or_model_promotion_allowed_now"]))
    ]
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(["official_total_return_validation", "portfolio_backtest", "model_promotion"]),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "readiness_checks_passed",
                "status": "pass" if unexpected.empty else "fail",
                "detail": ";".join(unexpected["check"].astype(str)),
            },
            {
                "check": "summary_and_decisions_produced",
                "status": "pass" if not summary.empty and not decisions.empty else "fail",
                "detail": f"summary={len(summary)};decisions={len(decisions)}",
            },
            {
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not decisions["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"rows={len(decisions)}",
            },
            {
                "check": "portfolio_or_promotion_outputs_not_produced",
                "status": "pass" if not forbidden_produced and not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
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
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    temporal: pd.DataFrame,
    controls: pd.DataFrame,
    state_audit: pd.DataFrame,
    trend_audit: pd.DataFrame,
    decisions: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    positive = int((summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not summary.empty else 0
    survivors = int(decisions["artifact_review_status"].astype(str).eq("survives_for_walk_forward_proxy_review").sum()) if not decisions.empty else 0
    blocked = int(decisions["artifact_review_status"].astype(str).eq("artifact_risk_blocks_escalation").sum()) if not decisions.empty else 0
    lines = [
        "# V3.63 Lag-Safe Proxy Validation",
        "",
        "## Decision",
        "",
        "- V3.63 validates V3.62 lag-safe non-price signals against V3.59 `price_index_return` proxy labels.",
        "- It reruns temporal, negative-control, state-dependence, and trend-regime artifact screening.",
        "- It does not run a portfolio backtest, write NAV, or promote any default model.",
        "",
        "## Scope",
        "",
        f"- Joined rows: `{len(joined)}`",
        f"- Signal-horizon rows: `{len(summary)}`",
        f"- Proxy-positive observations: `{positive}`",
        f"- Survives for walk-forward proxy review: `{survivors}`",
        f"- Artifact-risk blocked rows: `{blocked}`",
        "",
        "## Candidate Decisions",
        "",
    ]
    lines.extend(markdown_table(decisions, ["signal_id", "horizon", "proxy_evidence_status", "artifact_review_status", "artifact_flags", "next_action"], max_rows=30))
    lines.extend(["", "## Signal-Horizon Summary", ""])
    lines.extend(
        markdown_table(
            summary,
            [
                "signal_id",
                "horizon",
                "observations",
                "proxy_spearman_corr",
                "proxy_qspread_top_minus_bottom",
                "top_bucket_directional_alignment_share",
                "proxy_evidence_status",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Temporal Artifact Audit", ""])
    lines.extend(
        markdown_table(
            temporal,
            [
                "signal_id",
                "horizon",
                "real_proxy_spearman_corr",
                "same_day_price_signed_spearman_corr",
                "same_day_artifact_flag",
                "next_signal_abs_corr_excess",
                "future_signal_artifact_flag",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Negative Control Audit", ""])
    lines.extend(markdown_table(controls, ["signal_id", "horizon", "real_proxy_spearman_corr", "lag_broken_control_spearman_corr", "abs_corr_degradation", "negative_control_degraded"], max_rows=30))
    lines.extend(["", "## State Dependence Audit", ""])
    lines.extend(markdown_table(state_audit, ["signal_id", "horizon", "supportive_primary_state_rows", "state_columns_with_support", "state_support_too_sparse_flag", "bull_state_proxy_flag"], max_rows=30))
    lines.extend(["", "## Trend Regime Sample", ""])
    lines.extend(markdown_table(trend_audit, ["signal_id", "horizon", "market_trend_state", "observations", "proxy_spearman_corr", "observation_share"], max_rows=30))
    lines.extend(["", "## Readiness", ""])
    lines.extend(markdown_table(readiness, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=16))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Use surviving rows only as a narrow proxy-review queue, not as investable evidence.",
            "- If no rows survive, return to signal design or acquire better independent labels.",
            "- A future walk-forward test still needs official total-return labels before any default model decision.",
        ]
    )
    return "\n".join(lines)


def build_catalog(summary: pd.DataFrame, decisions: pd.DataFrame, config: LagSafeProxyValidationConfig) -> str:
    positive = int((summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not summary.empty else 0
    survivors = int(decisions["artifact_review_status"].astype(str).eq("survives_for_walk_forward_proxy_review").sum()) if not decisions.empty else 0
    return "\n".join(
        [
            "# A-share Lag-Safe Proxy Validation V3.63",
            "",
            "## Dataset Role",
            "",
            "V3.63 validates V3.62 lag-safe non-price MARKET candidate signals against non-official price-index proxy labels.",
            "",
            "## Governance",
            "",
            "- Return basis: `price_index_return` proxy evidence only.",
            "- Official total-return evidence: false.",
            "- Portfolio backtest: not produced.",
            "- Default model promotion: not allowed.",
            "",
            "## Inputs",
            "",
            f"- Repaired signals: `{config.repaired_signal_panel_path}`",
            f"- Proxy labels: `{config.price_proxy_label_path}`",
            "",
            "## Produced Shape",
            "",
            f"- Signal-horizon rows: `{len(summary)}`",
            f"- Proxy-positive rows: `{positive}`",
            f"- Walk-forward proxy-review candidates: `{survivors}`",
        ]
    )
