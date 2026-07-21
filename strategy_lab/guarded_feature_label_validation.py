"""Guarded feature-label validation for HIRSSM V3.71.

V3.71 joins the V3.70 combined feature registry to V3.59 non-official
price-index proxy labels. It runs feature-level and walk-forward diagnostics
only. It does not create positions, NAV, portfolio metrics, official total
return evidence, or default-model promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class GuardedFeatureLabelValidationConfig:
    combined_manifest_path: Path
    combined_panel_path: Path
    feature_registry_path: Path
    label_path: Path
    v3_59_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    train_years: int
    test_years: int
    min_full_sample_observations: int
    min_train_observations: int
    min_test_observations: int
    min_abs_train_spearman: float
    min_abs_train_qspread: float
    min_signed_oos_spearman: float
    min_signed_oos_qspread: float
    min_gated_windows: int
    min_oos_pass_rate: float
    min_oos_median_signed_spearman: float
    min_oos_positive_qspread_share: float
    top_quantile: float
    bottom_quantile: float
    negative_control_shift: int
    fdr_alpha: float
    exclude_stale_macro_rows: bool


NUMERIC_FEATURE_TYPES = {
    "numeric_score",
    "numeric_rolling_zscore",
    "numeric_trailing_percentile",
    "numeric_ratio_or_share",
    "numeric_or_categorical_feature",
    "count_feature",
}


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def normalize_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.isin(["", "nan", "NaN", "None", "NaT"]), "")
    yyyymmdd = text.str.fullmatch(r"\d{8}", na=False)
    parsed_8 = pd.to_datetime(text.where(yyyymmdd), format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")
    parsed_other = pd.to_datetime(text.where(~yyyymmdd), errors="coerce").dt.strftime("%Y%m%d")
    fallback = text.str.replace("-", "", regex=False).str.replace("/", "", regex=False).str[:8]
    return parsed_8.fillna(parsed_other).fillna(fallback)


def _corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(clean["x"].rank().corr(clean["y"].rank(), method="pearson"))
    return float(clean["x"].corr(clean["y"], method="pearson"))


def _approx_corr_pvalue(corr: float, observations: int) -> float:
    if observations < 4 or pd.isna(corr):
        return np.nan
    corr = max(min(float(corr), 0.999999), -0.999999)
    z = abs(corr) * sqrt(max(1, observations - 3))
    return float(erfc(z / sqrt(2.0)))


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce")
    q = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return q
    ranked = valid.reset_index()
    ranked.columns = ["orig_index", "pvalue"]
    ranked["rank"] = np.arange(1, m + 1)
    ranked["qvalue"] = (ranked["pvalue"] * m / ranked["rank"]).clip(upper=1.0)
    ranked["qvalue"] = ranked["qvalue"][::-1].cummin()[::-1]
    q.loc[ranked["orig_index"]] = ranked["qvalue"].to_numpy()
    return q


def _signed_direction(value: float, fallback: float = np.nan) -> int:
    if pd.notna(value) and value > 0:
        return 1
    if pd.notna(value) and value < 0:
        return -1
    if pd.notna(fallback) and fallback > 0:
        return 1
    if pd.notna(fallback) and fallback < 0:
        return -1
    return 0


def _bucket_metrics(group: pd.DataFrame, feature_col: str, target_col: str, config: GuardedFeatureLabelValidationConfig) -> dict[str, Any]:
    clean = group.loc[:, [feature_col, target_col]].copy()
    clean.columns = ["feature", "target"]
    clean["feature"] = pd.to_numeric(clean["feature"], errors="coerce")
    clean["target"] = pd.to_numeric(clean["target"], errors="coerce")
    clean = clean.dropna()
    observations = int(len(clean))
    if observations == 0:
        return {
            "observations": 0,
            "spearman_corr": np.nan,
            "pearson_corr": np.nan,
            "qspread_top_minus_bottom": np.nan,
            "top_bucket_mean": np.nan,
            "bottom_bucket_mean": np.nan,
            "top_bucket_rows": 0,
            "bottom_bucket_rows": 0,
            "target_mean": np.nan,
            "target_positive_share": np.nan,
        }
    top_cut = clean["feature"].quantile(config.top_quantile)
    bottom_cut = clean["feature"].quantile(config.bottom_quantile)
    top = clean.loc[clean["feature"] >= top_cut]
    bottom = clean.loc[clean["feature"] <= bottom_cut]
    top_mean = float(top["target"].mean()) if not top.empty else np.nan
    bottom_mean = float(bottom["target"].mean()) if not bottom.empty else np.nan
    return {
        "observations": observations,
        "spearman_corr": _corr(clean["feature"], clean["target"], "spearman"),
        "pearson_corr": _corr(clean["feature"], clean["target"], "pearson"),
        "qspread_top_minus_bottom": top_mean - bottom_mean if pd.notna(top_mean) and pd.notna(bottom_mean) else np.nan,
        "top_bucket_mean": top_mean,
        "bottom_bucket_mean": bottom_mean,
        "top_bucket_rows": int(len(top)),
        "bottom_bucket_rows": int(len(bottom)),
        "target_mean": float(clean["target"].mean()),
        "target_positive_share": float((clean["target"] > 0).mean()),
    }


def validate_inputs(
    combined_manifest: dict[str, Any],
    v3_59_manifest: dict[str, Any],
    combined: pd.DataFrame,
    registry: pd.DataFrame,
    labels: pd.DataFrame,
    config: GuardedFeatureLabelValidationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combined_required = {
        "signal_date",
        "signal_use_date",
        "combined_feature_validation_ready",
        "macro_any_stale_warning",
        "combined_model_promotion_allowed",
        "combined_portfolio_harness_allowed",
    }
    registry_required = {"feature_id", "source_version", "source_family", "feature_type", "combined_column", "model_promotion_allowed"}
    label_required = {
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
    combined_missing = sorted(combined_required.difference(combined.columns))
    registry_missing = sorted(registry_required.difference(registry.columns))
    label_missing = sorted(label_required.difference(labels.columns))
    rows.extend(
        [
            {
                "check": "v3_70_manifest_passed",
                "status": _status(bool(combined_manifest.get("self_check_pass")) and str(combined_manifest.get("status", "")).lower() == "pass"),
                "detail": f"status={combined_manifest.get('status')};self_check={combined_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_59_label_manifest_passed",
                "status": _status(bool(v3_59_manifest.get("self_check_pass")) and str(v3_59_manifest.get("status", "")).lower() == "pass"),
                "detail": f"status={v3_59_manifest.get('status')};self_check={v3_59_manifest.get('self_check_pass')}",
            },
            {
                "check": "combined_required_columns_present",
                "status": _status(not combined_missing),
                "detail": ",".join(combined_missing),
            },
            {
                "check": "registry_required_columns_present",
                "status": _status(not registry_missing),
                "detail": ",".join(registry_missing),
            },
            {
                "check": "label_required_columns_present",
                "status": _status(not label_missing),
                "detail": ",".join(label_missing),
            },
        ]
    )
    if combined_missing or registry_missing or label_missing:
        return pd.DataFrame(rows)

    ready = _bool_series(combined["combined_feature_validation_ready"])
    model_allowed = _bool_series(combined["combined_model_promotion_allowed"])
    portfolio_allowed = _bool_series(combined["combined_portfolio_harness_allowed"])
    feature_model_allowed = _bool_series(registry["model_promotion_allowed"])
    label_model_allowed = _bool_series(labels["model_promotion_allowed"])
    label_performance_allowed = _bool_series(labels["performance_claim_allowed"])
    official = _bool_series(labels["official_total_return"])
    official_allowed = _bool_series(labels["official_label_generation_allowed"])
    proxy_allowed = _bool_series(labels["proxy_label_generation_allowed"])
    label_basis = set(labels["return_basis"].astype(str).unique())
    label_usage = set(labels["diagnostic_usage"].astype(str).unique())
    label_horizons = set(pd.to_numeric(labels["horizon"], errors="coerce").dropna().astype(int))
    label_signal = normalize_date(labels["signal_date"])
    label_available = normalize_date(labels["label_available_date"])
    rows.extend(
        [
            {
                "check": "combined_ready_rows_present",
                "status": _status(int(ready.sum()) >= config.min_full_sample_observations),
                "detail": f"ready_rows={int(ready.sum())};min={config.min_full_sample_observations}",
            },
            {
                "check": "source_and_registry_no_model_promotion",
                "status": _status(not model_allowed.any() and not portfolio_allowed.any() and not feature_model_allowed.any()),
                "detail": f"combined_model={bool(model_allowed.any())};portfolio={bool(portfolio_allowed.any())};registry_model={bool(feature_model_allowed.any())}",
            },
            {
                "check": "label_source_is_price_proxy_only",
                "status": _status(label_basis == {"price_index_return"} and label_usage == {"non_official_price_proxy_label_only"}),
                "detail": f"basis={sorted(label_basis)};usage={sorted(label_usage)}",
            },
            {
                "check": "label_horizons_match_config",
                "status": _status(label_horizons == set(config.horizons)),
                "detail": f"actual={sorted(label_horizons)};expected={sorted(config.horizons)}",
            },
            {
                "check": "label_available_after_signal_date",
                "status": _status((label_available > label_signal).all()),
                "detail": f"bad_rows={int((label_available <= label_signal).sum())}",
            },
            {
                "check": "label_no_official_or_promotion_flags",
                "status": _status(not official.any() and not official_allowed.any() and proxy_allowed.all() and not label_model_allowed.any() and not label_performance_allowed.any()),
                "detail": f"official={bool(official.any())};official_allowed={bool(official_allowed.any())};proxy_false={int((~proxy_allowed).sum())};model={bool(label_model_allowed.any())};performance={bool(label_performance_allowed.any())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def select_validation_features(registry: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    data = registry.copy()
    data = data.loc[data["feature_type"].astype(str).isin(NUMERIC_FEATURE_TYPES)].copy()
    data = data.loc[~data["feature_type"].astype(str).eq("staleness_diagnostic")].copy()
    data = data.loc[data["combined_column"].astype(str).isin(combined.columns)].copy()
    data["model_promotion_allowed"] = False
    data["validation_scope"] = "proxy_feature_label_validation_only"
    return data.sort_values(["source_version", "source_family", "feature_type", "feature_id"]).reset_index(drop=True)


def build_validation_universe(combined: pd.DataFrame, labels: pd.DataFrame, config: GuardedFeatureLabelValidationConfig) -> pd.DataFrame:
    feature_dates = combined.copy()
    feature_dates["signal_date"] = normalize_date(feature_dates["signal_date"])
    feature_dates["combined_feature_validation_ready"] = _bool_series(feature_dates["combined_feature_validation_ready"])
    feature_dates["macro_any_stale_warning"] = _bool_series(feature_dates["macro_any_stale_warning"])
    feature_dates = feature_dates.loc[feature_dates["combined_feature_validation_ready"]].copy()
    label = labels.copy()
    label["signal_date"] = normalize_date(label["signal_date"])
    label["label_available_date"] = normalize_date(label["label_available_date"])
    label["horizon"] = pd.to_numeric(label["horizon"], errors="coerce").astype("Int64")
    label = label.loc[label["horizon"].isin(config.horizons)].copy()
    label["forward_price_index_return"] = pd.to_numeric(label["forward_price_index_return"], errors="coerce")
    universe = feature_dates.merge(
        label.loc[
            :,
            [
                "signal_date",
                "asset",
                "horizon",
                "forward_price_index_return",
                "return_basis",
                "label_available_date",
                "official_total_return",
                "model_promotion_allowed",
                "performance_claim_allowed",
                "diagnostic_usage",
            ],
        ],
        on="signal_date",
        how="inner",
    )
    universe["signal_year"] = pd.to_datetime(universe["signal_date"], format="%Y%m%d", errors="coerce").dt.year
    universe["validation_scope"] = "non_official_price_proxy_guarded_feature_validation"
    universe["official_total_return_evidence"] = False
    universe["portfolio_backtest_allowed"] = False
    universe["default_model_allowed"] = False
    return universe.sort_values(["horizon", "signal_date"]).reset_index(drop=True)


def feature_sample(universe: pd.DataFrame, feature: pd.Series, config: GuardedFeatureLabelValidationConfig) -> pd.DataFrame:
    data = universe.copy()
    if config.exclude_stale_macro_rows and str(feature["source_family"]) == "macro_growth_liquidity":
        data = data.loc[~_bool_series(data["macro_any_stale_warning"])].copy()
    return data


def build_full_sample_feature_stats(
    universe: pd.DataFrame,
    features: pd.DataFrame,
    config: GuardedFeatureLabelValidationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features.itertuples(index=False):
        feature_row = pd.Series(feature._asdict())
        feature_id = feature.feature_id
        feature_col = feature.combined_column
        for horizon in config.horizons:
            sample = feature_sample(universe.loc[universe["horizon"].astype(int).eq(int(horizon))], feature_row, config)
            metrics = _bucket_metrics(sample, feature_col, "forward_price_index_return", config)
            negative = sample.sort_values("signal_date").copy()
            negative["negative_control_return"] = negative["forward_price_index_return"].shift(config.negative_control_shift)
            neg_metrics = _bucket_metrics(negative, feature_col, "negative_control_return", config)
            direction = _signed_direction(metrics["spearman_corr"], metrics["qspread_top_minus_bottom"])
            rows.append(
                {
                    "feature_id": feature_id,
                    "combined_column": feature_col,
                    "source_version": feature.source_version,
                    "source_family": feature.source_family,
                    "feature_type": feature.feature_type,
                    "horizon": int(horizon),
                    "observations": metrics["observations"],
                    "spearman_corr": metrics["spearman_corr"],
                    "pearson_corr": metrics["pearson_corr"],
                    "qspread_top_minus_bottom": metrics["qspread_top_minus_bottom"],
                    "full_sample_direction": direction,
                    "signed_spearman_corr": direction * metrics["spearman_corr"] if direction else np.nan,
                    "signed_qspread": direction * metrics["qspread_top_minus_bottom"] if direction else np.nan,
                    "approx_spearman_pvalue": _approx_corr_pvalue(metrics["spearman_corr"], metrics["observations"]),
                    "negative_control_spearman_corr": neg_metrics["spearman_corr"],
                    "negative_control_qspread": neg_metrics["qspread_top_minus_bottom"],
                    "negative_control_abs_spearman_ratio": abs(neg_metrics["spearman_corr"]) / max(abs(metrics["spearman_corr"]), 1e-12)
                    if pd.notna(neg_metrics["spearman_corr"]) and pd.notna(metrics["spearman_corr"])
                    else np.nan,
                    "macro_stale_rows_excluded": bool(config.exclude_stale_macro_rows and str(feature.source_family) == "macro_growth_liquidity"),
                    "official_total_return_evidence": False,
                    "portfolio_backtest_allowed": False,
                    "default_model_allowed": False,
                }
            )
    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    stats["spearman_qvalue_bh"] = _bh_qvalues(stats["approx_spearman_pvalue"])
    return stats.sort_values(["horizon", "source_family", "feature_id"]).reset_index(drop=True)


def build_walk_forward_feature_results(
    universe: pd.DataFrame,
    features: pd.DataFrame,
    config: GuardedFeatureLabelValidationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if universe.empty or features.empty:
        return pd.DataFrame(rows)
    min_year = int(universe["signal_year"].min())
    max_year = int(universe["signal_year"].max())
    first_test_year = min_year + config.train_years
    for feature in features.itertuples(index=False):
        feature_row = pd.Series(feature._asdict())
        feature_col = feature.combined_column
        for horizon in config.horizons:
            base = feature_sample(universe.loc[universe["horizon"].astype(int).eq(int(horizon))], feature_row, config)
            base = base.loc[pd.to_numeric(base[feature_col], errors="coerce").notna()].copy()
            if base.empty:
                continue
            for test_year in range(first_test_year, max_year + 1, config.test_years):
                train_start = test_year - config.train_years
                train_end = test_year - 1
                test_end = test_year + config.test_years - 1
                train = base.loc[(base["signal_year"] >= train_start) & (base["signal_year"] <= train_end)].copy()
                test = base.loc[(base["signal_year"] >= test_year) & (base["signal_year"] <= test_end)].copy()
                if test.empty:
                    continue
                train_metrics = _bucket_metrics(train, feature_col, "forward_price_index_return", config)
                test_metrics = _bucket_metrics(test, feature_col, "forward_price_index_return", config)
                direction = _signed_direction(train_metrics["spearman_corr"], train_metrics["qspread_top_minus_bottom"])
                train_gate_reasons: list[str] = []
                if train_metrics["observations"] < config.min_train_observations:
                    train_gate_reasons.append("insufficient_train_observations")
                if pd.isna(train_metrics["spearman_corr"]) or abs(float(train_metrics["spearman_corr"])) < config.min_abs_train_spearman:
                    train_gate_reasons.append("train_abs_spearman_below_threshold")
                if pd.isna(train_metrics["qspread_top_minus_bottom"]) or abs(float(train_metrics["qspread_top_minus_bottom"])) < config.min_abs_train_qspread:
                    train_gate_reasons.append("train_abs_qspread_below_threshold")
                if direction == 0:
                    train_gate_reasons.append("train_direction_unavailable")
                train_gate = len(train_gate_reasons) == 0
                signed_oos_spearman = direction * test_metrics["spearman_corr"] if direction and pd.notna(test_metrics["spearman_corr"]) else np.nan
                signed_oos_qspread = direction * test_metrics["qspread_top_minus_bottom"] if direction and pd.notna(test_metrics["qspread_top_minus_bottom"]) else np.nan
                oos_reasons: list[str] = []
                if test_metrics["observations"] < config.min_test_observations:
                    oos_reasons.append("insufficient_test_observations")
                if pd.isna(signed_oos_spearman) or float(signed_oos_spearman) < config.min_signed_oos_spearman:
                    oos_reasons.append("signed_oos_spearman_below_threshold")
                if pd.isna(signed_oos_qspread) or float(signed_oos_qspread) < config.min_signed_oos_qspread:
                    oos_reasons.append("signed_oos_qspread_below_threshold")
                oos_gate = len(oos_reasons) == 0
                rows.append(
                    {
                        "feature_id": feature.feature_id,
                        "combined_column": feature_col,
                        "source_version": feature.source_version,
                        "source_family": feature.source_family,
                        "feature_type": feature.feature_type,
                        "horizon": int(horizon),
                        "train_start_year": int(train_start),
                        "train_end_year": int(train_end),
                        "test_start_year": int(test_year),
                        "test_end_year": int(test_end),
                        "train_observations": train_metrics["observations"],
                        "train_spearman_corr": train_metrics["spearman_corr"],
                        "train_qspread_top_minus_bottom": train_metrics["qspread_top_minus_bottom"],
                        "train_direction": direction,
                        "train_gate_pass": bool(train_gate),
                        "train_gate_reason": ";".join(train_gate_reasons) if train_gate_reasons else "passed",
                        "test_observations": test_metrics["observations"],
                        "oos_spearman_corr": test_metrics["spearman_corr"],
                        "oos_qspread_top_minus_bottom": test_metrics["qspread_top_minus_bottom"],
                        "signed_oos_spearman_corr": signed_oos_spearman,
                        "signed_oos_qspread": signed_oos_qspread,
                        "oos_gate_pass": bool(oos_gate),
                        "oos_gate_reason": ";".join(oos_reasons) if oos_reasons else "passed",
                        "train_gate_and_oos_pass": bool(train_gate and oos_gate),
                        "macro_stale_rows_excluded": bool(config.exclude_stale_macro_rows and str(feature.source_family) == "macro_growth_liquidity"),
                        "official_total_return_evidence": False,
                        "portfolio_backtest_allowed": False,
                        "default_model_allowed": False,
                    }
                )
    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values(["horizon", "source_family", "feature_id", "test_start_year"]).reset_index(drop=True)


def build_feature_horizon_summary(full_stats: pd.DataFrame, windows: pd.DataFrame, config: GuardedFeatureLabelValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if full_stats.empty:
        return pd.DataFrame(rows)
    for key, stat_group in full_stats.groupby(["feature_id", "horizon"], dropna=False):
        feature_id, horizon = key
        stat = stat_group.iloc[0]
        win = windows.loc[(windows["feature_id"].astype(str).eq(str(feature_id))) & (windows["horizon"].astype(int).eq(int(horizon)))] if not windows.empty else pd.DataFrame()
        gated = win.loc[win["train_gate_pass"].astype(bool)].copy() if not win.empty else pd.DataFrame()
        gated_count = int(len(gated))
        oos_pass_count = int(gated["oos_gate_pass"].astype(bool).sum()) if gated_count else 0
        pass_rate = float(oos_pass_count / gated_count) if gated_count else np.nan
        median_signed_spearman = float(gated["signed_oos_spearman_corr"].median()) if gated_count else np.nan
        positive_qspread_share = float((gated["signed_oos_qspread"] > 0).mean()) if gated_count else np.nan
        median_signed_qspread = float(gated["signed_oos_qspread"].median()) if gated_count else np.nan
        negative_ratio = float(stat.get("negative_control_abs_spearman_ratio", np.nan))
        negative_flag = bool(pd.notna(negative_ratio) and negative_ratio >= 0.8 and abs(float(stat.get("negative_control_spearman_corr", 0))) >= 0.02)
        qualifies = (
            gated_count >= config.min_gated_windows
            and pd.notna(pass_rate)
            and pass_rate >= config.min_oos_pass_rate
            and pd.notna(median_signed_spearman)
            and median_signed_spearman >= config.min_oos_median_signed_spearman
            and pd.notna(positive_qspread_share)
            and positive_qspread_share >= config.min_oos_positive_qspread_share
            and not negative_flag
        )
        rows.append(
            {
                "feature_id": feature_id,
                "combined_column": stat["combined_column"],
                "source_version": stat["source_version"],
                "source_family": stat["source_family"],
                "feature_type": stat["feature_type"],
                "horizon": int(horizon),
                "full_sample_observations": int(stat["observations"]),
                "full_sample_spearman_corr": stat["spearman_corr"],
                "full_sample_signed_spearman_corr": stat["signed_spearman_corr"],
                "full_sample_qspread": stat["qspread_top_minus_bottom"],
                "spearman_qvalue_bh": stat.get("spearman_qvalue_bh", np.nan),
                "negative_control_spearman_corr": stat.get("negative_control_spearman_corr", np.nan),
                "negative_control_abs_spearman_ratio": negative_ratio,
                "negative_control_artifact_flag": negative_flag,
                "total_windows": int(len(win)),
                "train_gated_windows": gated_count,
                "oos_gate_pass_windows": oos_pass_count,
                "oos_gate_pass_rate_on_gated_windows": pass_rate,
                "oos_median_signed_spearman_corr": median_signed_spearman,
                "oos_median_signed_qspread": median_signed_qspread,
                "oos_positive_signed_qspread_share": positive_qspread_share,
                "proxy_validation_status": "passes_proxy_walk_forward_for_stricter_review" if qualifies else "observe_or_repair",
                "next_action": "queue_for_official_or_higher_quality_label_review_not_model_promotion" if qualifies else "keep_observation_or_repair",
                "official_total_return_evidence": False,
                "portfolio_backtest_allowed": False,
                "default_model_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["proxy_validation_status", "horizon", "source_family", "feature_id"]).reset_index(drop=True)


def build_source_family_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(["source_family", "horizon"], as_index=False)
        .agg(
            feature_count=("feature_id", "count"),
            proxy_pass_count=("proxy_validation_status", lambda x: int((x == "passes_proxy_walk_forward_for_stricter_review").sum())),
            median_full_sample_abs_spearman=("full_sample_spearman_corr", lambda x: float(pd.to_numeric(x, errors="coerce").abs().median())),
            median_oos_signed_spearman=("oos_median_signed_spearman_corr", "median"),
            median_oos_pass_rate=("oos_gate_pass_rate_on_gated_windows", "median"),
            negative_control_flag_count=("negative_control_artifact_flag", "sum"),
        )
        .sort_values(["horizon", "source_family"])
        .reset_index(drop=True)
    )


def build_candidate_decision(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    data = summary.loc[
        :,
        [
            "feature_id",
            "source_family",
            "feature_type",
            "horizon",
            "proxy_validation_status",
            "next_action",
            "train_gated_windows",
            "oos_gate_pass_windows",
            "oos_gate_pass_rate_on_gated_windows",
            "oos_median_signed_spearman_corr",
            "oos_positive_signed_qspread_share",
            "negative_control_artifact_flag",
        ],
    ].copy()
    data["decision"] = "proxy_validation_candidate_only_no_default_promotion"
    data["reason"] = "non-official price-index proxy labels are not investable evidence"
    data["official_total_return_evidence"] = False
    data["portfolio_backtest_allowed"] = False
    data["default_model_allowed"] = False
    return data.sort_values(["proxy_validation_status", "horizon", "source_family", "feature_id"]).reset_index(drop=True)


def build_multiple_testing_report(summary: pd.DataFrame, config: GuardedFeatureLabelValidationConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for horizon, group in summary.groupby("horizon"):
        q = pd.to_numeric(group["spearman_qvalue_bh"], errors="coerce")
        rows.append(
            {
                "horizon": int(horizon),
                "feature_horizon_tests": int(len(group)),
                "bh_qvalue_le_alpha_count": int((q <= config.fdr_alpha).sum()),
                "proxy_walk_forward_pass_count": int(group["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()),
                "negative_control_artifact_flag_count": int(group["negative_control_artifact_flag"].astype(bool).sum()),
                "fdr_alpha": config.fdr_alpha,
                "default_model_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def build_no_promotion_guard(summary: pd.DataFrame) -> pd.DataFrame:
    passed = int(summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()) if not summary.empty else 0
    return pd.DataFrame(
        [
            {
                "result_type": "guarded_proxy_feature_label_validation",
                "produced": True,
                "blocked": False,
                "reason": "non-official price-index proxy feature diagnostics",
            },
            {
                "result_type": "stricter_label_review_queue",
                "produced": passed > 0,
                "blocked": passed == 0,
                "reason": "candidate queue only; not default model promotion",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.71 uses only V3.59 price-index proxy labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.71 produces no positions, trades, NAV, or portfolio metrics",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "proxy validation cannot promote the default model",
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    universe: pd.DataFrame,
    features: pd.DataFrame,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
    config: GuardedFeatureLabelValidationConfig,
) -> pd.DataFrame:
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(["official_total_return_validation", "portfolio_backtest", "model_promotion"]),
            "produced",
        ].astype(bool).any()
    )
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": _status(not input_checks["status"].eq("fail").any()),
                "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "validation_universe_large_enough",
                "status": _status(len(universe) >= config.min_full_sample_observations * len(config.horizons)),
                "detail": f"rows={len(universe)};min={config.min_full_sample_observations * len(config.horizons)}",
            },
            {
                "check": "validation_features_present",
                "status": _status(not features.empty),
                "detail": f"rows={len(features)}",
            },
            {
                "check": "walk_forward_windows_produced",
                "status": _status(not windows.empty),
                "detail": f"rows={len(windows)}",
            },
            {
                "check": "feature_horizon_summary_produced",
                "status": _status(not summary.empty),
                "detail": f"rows={len(summary)}",
            },
            {
                "check": "promotion_outputs_blocked",
                "status": _status(not forbidden_produced),
                "detail": "guard blocks official validation, portfolio backtest, and model promotion",
            },
            {
                "check": "forbidden_performance_columns_absent",
                "status": _status(not forbidden_columns),
                "detail": ",".join(forbidden_columns),
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
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
    universe: pd.DataFrame,
    features: pd.DataFrame,
    full_stats: pd.DataFrame,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    multiple_testing: pd.DataFrame,
    input_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
    guard: pd.DataFrame,
) -> str:
    pass_count = int(summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()) if not summary.empty else 0
    top = summary.loc[summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review")].copy()
    if top.empty and not summary.empty:
        top = summary.sort_values(["oos_median_signed_spearman_corr", "oos_gate_pass_rate_on_gated_windows"], ascending=False).head(20)
    else:
        top = top.sort_values(["oos_median_signed_spearman_corr", "oos_gate_pass_rate_on_gated_windows"], ascending=False)
    lines = [
        "# V3.71 Guarded Feature-Label Validation",
        "",
        "## Decision",
        "",
        "- V3.71 validates V3.70 registered numeric features against V3.59 non-official price-index proxy labels.",
        "- Results are proxy diagnostics and candidate queues only.",
        "- No official total-return evidence, portfolio backtest, NAV, or default-model promotion is produced.",
        "",
        "## Coverage",
        "",
        f"- Validation universe rows: `{len(universe)}`",
        f"- Registered numeric features tested: `{len(features)}`",
        f"- Full-sample feature-horizon rows: `{len(full_stats)}`",
        f"- Walk-forward window rows: `{len(windows)}`",
        f"- Feature-horizon proxy passes for stricter review: `{pass_count}`",
        f"- Signal date range: `{universe['signal_date'].min() if not universe.empty else ''}` to `{universe['signal_date'].max() if not universe.empty else ''}`",
        "",
        "## Top Proxy Diagnostics",
        "",
    ]
    lines.extend(
        markdown_table(
            top,
            [
                "feature_id",
                "source_family",
                "horizon",
                "proxy_validation_status",
                "train_gated_windows",
                "oos_gate_pass_rate_on_gated_windows",
                "oos_median_signed_spearman_corr",
                "oos_positive_signed_qspread_share",
                "negative_control_artifact_flag",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Source Family Summary", ""])
    lines.extend(
        markdown_table(
            source_summary,
            [
                "source_family",
                "horizon",
                "feature_count",
                "proxy_pass_count",
                "median_full_sample_abs_spearman",
                "median_oos_signed_spearman",
                "negative_control_flag_count",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Multiple Testing", ""])
    lines.extend(markdown_table(multiple_testing, ["horizon", "feature_horizon_tests", "bh_qvalue_le_alpha_count", "proxy_walk_forward_pass_count", "negative_control_artifact_flag_count"], max_rows=12))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=24))
    lines.extend(["", "## Acceptance Checks", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## No Promotion Guard", ""])
    lines.extend(markdown_table(guard, ["result_type", "produced", "blocked", "reason"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Passing rows can only enter a stricter label-source review queue.",
            "- Any later strategy harness must use a separate task with official or higher-quality label governance.",
            "- Proxy-positive results here are not investable performance evidence.",
        ]
    )
    return "\n".join(lines)


def build_catalog(universe: pd.DataFrame, features: pd.DataFrame, summary: pd.DataFrame, config: GuardedFeatureLabelValidationConfig) -> str:
    pass_count = int(summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()) if not summary.empty else 0
    return "\n".join(
        [
            "# A-share Guarded Feature-Label Validation V3.71",
            "",
            "## Dataset Role",
            "",
            "V3.71 joins the V3.70 combined feature registry to V3.59 price-index proxy labels for guarded feature diagnostics.",
            "",
            "## Governance",
            "",
            "- Label source is non-official price-index proxy only.",
            "- No official total-return validation, portfolio backtest, or model promotion is produced.",
            "- Macro stale rows are excluded for macro features when configured.",
            "",
            "## Produced Shape",
            "",
            f"- Validation universe rows: `{len(universe)}`",
            f"- Tested numeric features: `{len(features)}`",
            f"- Feature-horizon summary rows: `{len(summary)}`",
            f"- Proxy pass rows for stricter review: `{pass_count}`",
            f"- Horizons: `{','.join(str(h) for h in config.horizons)}`",
        ]
    )
