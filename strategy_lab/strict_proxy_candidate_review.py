"""Strict review for V3.71 proxy-positive feature candidates.

V3.72 reviews only V3.71 proxy-positive feature-horizon rows. It adds
redundancy clustering, extended negative controls, alternate price-index proxy
validation, and market-trend proxy diagnostics. Outputs are governance and
research queues only: no portfolio backtest, NAV, official total-return
validation, or model promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class StrictProxyCandidateReviewConfig:
    v3_71_manifest_path: Path
    v3_71_summary_path: Path
    v3_71_candidate_decision_path: Path
    v3_71_walk_forward_path: Path
    combined_panel_path: Path
    primary_label_path: Path
    cross_index_paths: dict[str, Path]
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    correlation_cluster_threshold: float
    min_cluster_score_gap: float
    negative_control_shifts: tuple[int, ...]
    negative_control_ratio_threshold: float
    negative_control_abs_threshold: float
    min_cross_index_observations: int
    min_cross_index_signed_spearman: float
    min_cross_index_signed_qspread: float
    min_cross_index_year_positive_share: float
    min_cross_index_pass_count: int
    market_trend_proxy_corr_threshold: float
    trend_windows: tuple[int, ...]
    top_quantile: float
    bottom_quantile: float


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


def _signed_direction(row: pd.Series) -> int:
    corr = pd.to_numeric(pd.Series([row.get("full_sample_spearman_corr", np.nan)]), errors="coerce").iloc[0]
    signed = pd.to_numeric(pd.Series([row.get("full_sample_signed_spearman_corr", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(corr) or pd.isna(signed) or abs(corr) < 1e-12:
        median = pd.to_numeric(pd.Series([row.get("oos_median_signed_spearman_corr", np.nan)]), errors="coerce").iloc[0]
        return 1 if pd.notna(median) and median >= 0 else -1
    return 1 if signed * corr >= 0 else -1


def _bucket_metrics(data: pd.DataFrame, feature_col: str, target_col: str, config: StrictProxyCandidateReviewConfig) -> dict[str, Any]:
    clean = data.loc[:, [feature_col, target_col]].copy()
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
            "year_positive_share": np.nan,
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
        "year_positive_share": np.nan,
    }


def validate_inputs(
    v3_71_manifest: dict[str, Any],
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    windows: pd.DataFrame,
    combined: pd.DataFrame,
    primary_labels: pd.DataFrame,
    cross_indices: dict[str, pd.DataFrame],
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "check": "v3_71_manifest_passed",
            "status": _status(bool(v3_71_manifest.get("self_check_pass")) and str(v3_71_manifest.get("status", "")).lower() == "pass"),
            "detail": f"status={v3_71_manifest.get('status')};self_check={v3_71_manifest.get('self_check_pass')}",
        },
        {
            "check": "v3_71_proxy_positive_rows_present",
            "status": _status(int(summary.get("proxy_validation_status", pd.Series(dtype=str)).astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()) > 0),
            "detail": f"pass_rows={int(summary.get('proxy_validation_status', pd.Series(dtype=str)).astype(str).eq('passes_proxy_walk_forward_for_stricter_review').sum())}",
        },
    ]
    required_summary = {
        "feature_id",
        "combined_column",
        "source_family",
        "feature_type",
        "horizon",
        "full_sample_spearman_corr",
        "full_sample_signed_spearman_corr",
        "oos_gate_pass_rate_on_gated_windows",
        "oos_median_signed_spearman_corr",
        "negative_control_artifact_flag",
        "proxy_validation_status",
    }
    required_combined = {"signal_date", "combined_feature_validation_ready", "macro_any_stale_warning"}
    required_primary = {"signal_date", "horizon", "forward_price_index_return", "return_basis", "diagnostic_usage"}
    rows.extend(
        [
            {
                "check": "summary_required_columns_present",
                "status": _status(not required_summary.difference(summary.columns)),
                "detail": ",".join(sorted(required_summary.difference(summary.columns))),
            },
            {
                "check": "combined_required_columns_present",
                "status": _status(not required_combined.difference(combined.columns)),
                "detail": ",".join(sorted(required_combined.difference(combined.columns))),
            },
            {
                "check": "primary_label_required_columns_present",
                "status": _status(not required_primary.difference(primary_labels.columns)),
                "detail": ",".join(sorted(required_primary.difference(primary_labels.columns))),
            },
            {
                "check": "walk_forward_rows_present",
                "status": _status(not windows.empty),
                "detail": f"rows={len(windows)}",
            },
            {
                "check": "candidate_decisions_present",
                "status": _status(not decisions.empty),
                "detail": f"rows={len(decisions)}",
            },
        ]
    )
    if required_primary.issubset(primary_labels.columns):
        basis = set(primary_labels["return_basis"].astype(str).unique())
        usage = set(primary_labels["diagnostic_usage"].astype(str).unique())
        rows.append(
            {
                "check": "primary_label_is_non_official_price_proxy",
                "status": _status(basis == {"price_index_return"} and usage == {"non_official_price_proxy_label_only"}),
                "detail": f"basis={sorted(basis)};usage={sorted(usage)}",
            }
        )
    rows.append(
        {
            "check": "cross_index_sources_present",
            "status": _status(len(cross_indices) >= config.min_cross_index_pass_count),
            "detail": f"symbols={','.join(sorted(cross_indices))}",
        }
    )
    for symbol, frame in cross_indices.items():
        missing = sorted({"date", "close"}.difference(frame.columns))
        close = pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "check": f"cross_index_{symbol}_source_usable",
                "status": _status(not missing and close.notna().sum() >= config.min_cross_index_observations),
                "detail": f"missing={','.join(missing)};close_rows={int(close.notna().sum())}",
            }
        )
    return pd.DataFrame(rows)


def select_proxy_positive_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.loc[summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review")].copy()
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").astype(int)
    data["v3_71_direction"] = data.apply(_signed_direction, axis=1)
    data["review_score"] = (
        pd.to_numeric(data["oos_median_signed_spearman_corr"], errors="coerce").fillna(0.0)
        * pd.to_numeric(data["oos_gate_pass_rate_on_gated_windows"], errors="coerce").fillna(0.0)
    )
    data["official_total_return_evidence"] = False
    data["portfolio_backtest_allowed"] = False
    data["default_model_allowed"] = False
    return data.sort_values(["horizon", "source_family", "feature_id"]).reset_index(drop=True)


def prepare_feature_frame(combined: pd.DataFrame) -> pd.DataFrame:
    data = combined.copy()
    data["signal_date"] = normalize_date(data["signal_date"])
    data["combined_feature_validation_ready"] = _bool_series(data["combined_feature_validation_ready"])
    data["macro_any_stale_warning"] = _bool_series(data["macro_any_stale_warning"])
    data = data.loc[data["combined_feature_validation_ready"]].copy()
    return data.sort_values("signal_date").reset_index(drop=True)


def build_index_forward_labels(index_frame: pd.DataFrame, symbol: str, horizons: tuple[int, ...]) -> pd.DataFrame:
    data = index_frame.copy()
    data["signal_date"] = normalize_date(data["date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.loc[data["signal_date"].astype(str).str.len().eq(8) & data["close"].notna()].copy()
    data = data.drop_duplicates("signal_date", keep="last").sort_values("signal_date").reset_index(drop=True)
    rows = []
    for horizon in horizons:
        label = data.loc[:, ["signal_date", "close"]].copy()
        label["horizon"] = int(horizon)
        label["forward_price_index_return"] = data["close"].shift(-horizon) / data["close"] - 1.0
        label["label_available_date"] = data["signal_date"].shift(-horizon)
        label["cross_symbol"] = symbol
        label["return_basis"] = "price_index_return"
        label["diagnostic_usage"] = "alternate_non_official_price_proxy_label_only"
        label["official_total_return_evidence"] = False
        label["portfolio_backtest_allowed"] = False
        label["default_model_allowed"] = False
        rows.append(label.drop(columns=["close"]))
    out = pd.concat(rows, ignore_index=True)
    return out.loc[out["forward_price_index_return"].notna()].reset_index(drop=True)


def build_cross_label_coverage(cross_labels: dict[str, pd.DataFrame], feature_dates: pd.Series) -> pd.DataFrame:
    rows = []
    feature_date_set = set(feature_dates.astype(str))
    for symbol, labels in cross_labels.items():
        for horizon, group in labels.groupby("horizon"):
            overlap = group.loc[group["signal_date"].astype(str).isin(feature_date_set)]
            rows.append(
                {
                    "cross_symbol": symbol,
                    "horizon": int(horizon),
                    "label_rows": int(len(group)),
                    "overlap_rows": int(len(overlap)),
                    "first_overlap_signal_date": str(overlap["signal_date"].min()) if not overlap.empty else "",
                    "last_overlap_signal_date": str(overlap["signal_date"].max()) if not overlap.empty else "",
                    "return_basis": "price_index_return",
                    "official_total_return_evidence": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["cross_symbol", "horizon"]).reset_index(drop=True)


def candidate_sample(features: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    sample = features.copy()
    if str(candidate["source_family"]) == "macro_growth_liquidity":
        sample = sample.loc[~_bool_series(sample["macro_any_stale_warning"])].copy()
    return sample


def build_extended_negative_controls(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    primary_labels: pd.DataFrame,
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    labels = primary_labels.copy()
    labels["signal_date"] = normalize_date(labels["signal_date"])
    labels["horizon"] = pd.to_numeric(labels["horizon"], errors="coerce").astype(int)
    rows = []
    for row in candidates.itertuples(index=False):
        candidate = pd.Series(row._asdict())
        feature_col = str(row.combined_column)
        base_features = candidate_sample(features, candidate)
        label = labels.loc[labels["horizon"].eq(int(row.horizon)), ["signal_date", "forward_price_index_return"]].copy()
        label = label.sort_values("signal_date").reset_index(drop=True)
        joined = base_features.merge(label, on="signal_date", how="inner").sort_values("signal_date")
        true_corr = _corr(joined[feature_col], joined["forward_price_index_return"], "spearman") if feature_col in joined else np.nan
        max_abs_neg = 0.0
        worst_shift = None
        for shift in config.negative_control_shifts:
            control_col = f"negative_control_shift_{shift}"
            joined[control_col] = joined["forward_price_index_return"].shift(int(shift))
            neg_corr = _corr(joined[feature_col], joined[control_col], "spearman") if feature_col in joined else np.nan
            neg_qspread = _bucket_metrics(joined, feature_col, control_col, config)["qspread_top_minus_bottom"] if feature_col in joined else np.nan
            abs_neg = abs(float(neg_corr)) if pd.notna(neg_corr) else np.nan
            if pd.notna(abs_neg) and abs_neg >= max_abs_neg:
                max_abs_neg = abs_neg
                worst_shift = int(shift)
            rows.append(
                {
                    "feature_id": row.feature_id,
                    "source_family": row.source_family,
                    "horizon": int(row.horizon),
                    "negative_control_shift": int(shift),
                    "observations": int(joined[[feature_col, control_col]].dropna().shape[0]) if feature_col in joined else 0,
                    "true_spearman_corr": true_corr,
                    "negative_control_spearman_corr": neg_corr,
                    "negative_control_qspread": neg_qspread,
                    "abs_negative_to_true_ratio": abs(float(neg_corr)) / max(abs(float(true_corr)), 1e-12) if pd.notna(neg_corr) and pd.notna(true_corr) else np.nan,
                    "official_total_return_evidence": False,
                    "portfolio_backtest_allowed": False,
                    "default_model_allowed": False,
                }
            )
        artifact_flag = bool(
            pd.notna(true_corr)
            and max_abs_neg >= max(config.negative_control_abs_threshold, abs(float(true_corr)) * config.negative_control_ratio_threshold)
        )
        for item in rows[-len(config.negative_control_shifts) :]:
            item["max_abs_negative_control_corr"] = max_abs_neg
            item["worst_negative_control_shift"] = worst_shift
            item["extended_negative_control_flag"] = artifact_flag
    return pd.DataFrame(rows).sort_values(["horizon", "source_family", "feature_id", "negative_control_shift"]).reset_index(drop=True)


def build_cross_index_validation(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    cross_labels: dict[str, pd.DataFrame],
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    rows = []
    for row in candidates.itertuples(index=False):
        candidate = pd.Series(row._asdict())
        feature_col = str(row.combined_column)
        base_features = candidate_sample(features, candidate)
        direction = int(row.v3_71_direction)
        for symbol, labels in cross_labels.items():
            label = labels.loc[labels["horizon"].astype(int).eq(int(row.horizon))].copy()
            joined = base_features.merge(label, on="signal_date", how="inner").sort_values("signal_date")
            if feature_col not in joined:
                continue
            metrics = _bucket_metrics(joined, feature_col, "forward_price_index_return", config)
            joined["signal_year"] = pd.to_datetime(joined["signal_date"], format="%Y%m%d", errors="coerce").dt.year
            yearly = []
            for _, group in joined.groupby("signal_year"):
                if len(group) < 100:
                    continue
                corr = _corr(group[feature_col], group["forward_price_index_return"], "spearman")
                qspread = _bucket_metrics(group, feature_col, "forward_price_index_return", config)["qspread_top_minus_bottom"]
                if pd.notna(corr) and pd.notna(qspread):
                    yearly.append((direction * corr > 0) and (direction * qspread > 0))
            signed_spearman = direction * metrics["spearman_corr"] if pd.notna(metrics["spearman_corr"]) else np.nan
            signed_qspread = direction * metrics["qspread_top_minus_bottom"] if pd.notna(metrics["qspread_top_minus_bottom"]) else np.nan
            year_positive_share = float(np.mean(yearly)) if yearly else np.nan
            pass_flag = bool(
                metrics["observations"] >= config.min_cross_index_observations
                and pd.notna(signed_spearman)
                and signed_spearman >= config.min_cross_index_signed_spearman
                and pd.notna(signed_qspread)
                and signed_qspread >= config.min_cross_index_signed_qspread
                and pd.notna(year_positive_share)
                and year_positive_share >= config.min_cross_index_year_positive_share
            )
            rows.append(
                {
                    "feature_id": row.feature_id,
                    "combined_column": row.combined_column,
                    "source_family": row.source_family,
                    "feature_type": row.feature_type,
                    "horizon": int(row.horizon),
                    "cross_symbol": symbol,
                    "observations": metrics["observations"],
                    "v3_71_direction": direction,
                    "cross_spearman_corr": metrics["spearman_corr"],
                    "cross_signed_spearman_corr": signed_spearman,
                    "cross_qspread_top_minus_bottom": metrics["qspread_top_minus_bottom"],
                    "cross_signed_qspread": signed_qspread,
                    "year_positive_share": year_positive_share,
                    "cross_index_pass": pass_flag,
                    "return_basis": "price_index_return",
                    "official_total_return_evidence": False,
                    "portfolio_backtest_allowed": False,
                    "default_model_allowed": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "source_family", "feature_id", "cross_symbol"]).reset_index(drop=True)


def build_redundancy_clusters(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    rows = []
    cluster_seq = 0
    for (horizon, family), group in candidates.groupby(["horizon", "source_family"], dropna=False):
        group = group.sort_values("review_score", ascending=False).reset_index(drop=True)
        assigned: dict[int, dict[str, Any]] = {}
        for i, row in group.iterrows():
            if i in assigned:
                continue
            cluster_seq += 1
            rep_col = str(row["combined_column"])
            rep_id = str(row["feature_id"])
            cluster_id = f"v3_72_c{cluster_seq:03d}"
            assigned[i] = {
                "cluster_id": cluster_id,
                "representative_feature_id": rep_id,
                "representative_combined_column": rep_col,
                "corr_to_representative": 1.0,
                "is_cluster_representative": True,
            }
            for j, other in group.iterrows():
                if j in assigned or j == i:
                    continue
                other_col = str(other["combined_column"])
                corr = _corr(features[rep_col], features[other_col], "spearman") if rep_col in features and other_col in features else np.nan
                if pd.notna(corr) and abs(float(corr)) >= config.correlation_cluster_threshold:
                    assigned[j] = {
                        "cluster_id": cluster_id,
                        "representative_feature_id": rep_id,
                        "representative_combined_column": rep_col,
                        "corr_to_representative": corr,
                        "is_cluster_representative": False,
                    }
        for i, row in group.iterrows():
            info = assigned[i]
            rows.append(
                {
                    "feature_id": row["feature_id"],
                    "combined_column": row["combined_column"],
                    "source_family": row["source_family"],
                    "feature_type": row["feature_type"],
                    "horizon": int(row["horizon"]),
                    "review_score": float(row["review_score"]),
                    **info,
                    "official_total_return_evidence": False,
                    "portfolio_backtest_allowed": False,
                    "default_model_allowed": False,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sizes = out.groupby("cluster_id")["feature_id"].count().rename("cluster_size")
    out = out.merge(sizes, on="cluster_id", how="left")
    return out.sort_values(["horizon", "source_family", "cluster_id", "is_cluster_representative"], ascending=[True, True, True, False]).reset_index(drop=True)


def build_market_trend_proxy_report(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    market_index: pd.DataFrame,
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    market = market_index.copy()
    market["signal_date"] = normalize_date(market["date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.loc[market["close"].notna()].drop_duplicates("signal_date", keep="last").sort_values("signal_date")
    for window in config.trend_windows:
        market[f"market_return_{window}d"] = market["close"].pct_change(int(window))
    market["market_vol_20d"] = market["close"].pct_change().rolling(20, min_periods=10).std()
    trend_cols = [f"market_return_{window}d" for window in config.trend_windows] + ["market_vol_20d"]
    base = features.merge(market.loc[:, ["signal_date", *trend_cols]], on="signal_date", how="left")
    rows = []
    for row in candidates.itertuples(index=False):
        feature_col = str(row.combined_column)
        if feature_col not in base:
            continue
        corr_values = {col: _corr(base[feature_col], base[col], "spearman") for col in trend_cols}
        abs_items = {col: abs(val) for col, val in corr_values.items() if pd.notna(val)}
        max_col = max(abs_items, key=abs_items.get) if abs_items else ""
        max_corr = corr_values.get(max_col, np.nan) if max_col else np.nan
        flag = bool(
            str(row.source_family) != "market_participation_breadth"
            and pd.notna(max_corr)
            and abs(float(max_corr)) >= config.market_trend_proxy_corr_threshold
        )
        rows.append(
            {
                "feature_id": row.feature_id,
                "combined_column": row.combined_column,
                "source_family": row.source_family,
                "feature_type": row.feature_type,
                "horizon": int(row.horizon),
                **{f"corr_{col}": corr_values[col] for col in trend_cols},
                "max_abs_market_trend_corr_column": max_col,
                "max_abs_market_trend_corr": abs(float(max_corr)) if pd.notna(max_corr) else np.nan,
                "market_trend_proxy_flag": flag,
                "official_total_return_evidence": False,
                "portfolio_backtest_allowed": False,
                "default_model_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "source_family", "feature_id"]).reset_index(drop=True)


def build_strict_candidate_decision(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    negative: pd.DataFrame,
    cross: pd.DataFrame,
    trend: pd.DataFrame,
    config: StrictProxyCandidateReviewConfig,
) -> pd.DataFrame:
    cluster_cols = ["feature_id", "horizon", "cluster_id", "representative_feature_id", "is_cluster_representative", "cluster_size"]
    data = candidates.merge(clusters.loc[:, cluster_cols], on=["feature_id", "horizon"], how="left")
    neg_summary = (
        negative.groupby(["feature_id", "horizon"], as_index=False)
        .agg(
            max_abs_negative_control_corr=("max_abs_negative_control_corr", "max"),
            worst_negative_control_shift=("worst_negative_control_shift", "first"),
            extended_negative_control_flag=("extended_negative_control_flag", "max"),
        )
        if not negative.empty
        else pd.DataFrame(columns=["feature_id", "horizon"])
    )
    cross_summary = (
        cross.groupby(["feature_id", "horizon"], as_index=False)
        .agg(
            cross_index_pass_count=("cross_index_pass", "sum"),
            cross_index_test_count=("cross_symbol", "count"),
            median_cross_signed_spearman=("cross_signed_spearman_corr", "median"),
            median_cross_signed_qspread=("cross_signed_qspread", "median"),
            median_cross_year_positive_share=("year_positive_share", "median"),
        )
        if not cross.empty
        else pd.DataFrame(columns=["feature_id", "horizon"])
    )
    trend_summary = trend.loc[:, ["feature_id", "horizon", "max_abs_market_trend_corr", "max_abs_market_trend_corr_column", "market_trend_proxy_flag"]] if not trend.empty else pd.DataFrame(columns=["feature_id", "horizon"])
    data = data.merge(neg_summary, on=["feature_id", "horizon"], how="left")
    data = data.merge(cross_summary, on=["feature_id", "horizon"], how="left")
    data = data.merge(trend_summary, on=["feature_id", "horizon"], how="left")
    rows = []
    for row in data.itertuples(index=False):
        reasons = []
        if not bool(row.is_cluster_representative):
            reasons.append("redundant_non_representative")
        if bool(row.extended_negative_control_flag):
            reasons.append("extended_negative_control_artifact")
        if int(row.cross_index_pass_count) < config.min_cross_index_pass_count:
            reasons.append("insufficient_cross_index_confirmation")
        if bool(row.market_trend_proxy_flag):
            reasons.append("likely_market_trend_proxy")
        strict_pass = len(reasons) == 0
        rows.append(
            {
                "feature_id": row.feature_id,
                "combined_column": row.combined_column,
                "source_family": row.source_family,
                "feature_type": row.feature_type,
                "horizon": int(row.horizon),
                "v3_71_direction": int(row.v3_71_direction),
                "review_score": float(row.review_score),
                "cluster_id": row.cluster_id,
                "representative_feature_id": row.representative_feature_id,
                "is_cluster_representative": bool(row.is_cluster_representative),
                "cluster_size": int(row.cluster_size),
                "extended_negative_control_flag": bool(row.extended_negative_control_flag),
                "max_abs_negative_control_corr": row.max_abs_negative_control_corr,
                "worst_negative_control_shift": row.worst_negative_control_shift,
                "cross_index_pass_count": int(row.cross_index_pass_count),
                "cross_index_test_count": int(row.cross_index_test_count),
                "median_cross_signed_spearman": row.median_cross_signed_spearman,
                "median_cross_signed_qspread": row.median_cross_signed_qspread,
                "median_cross_year_positive_share": row.median_cross_year_positive_share,
                "market_trend_proxy_flag": bool(row.market_trend_proxy_flag),
                "max_abs_market_trend_corr": row.max_abs_market_trend_corr,
                "max_abs_market_trend_corr_column": row.max_abs_market_trend_corr_column,
                "strict_review_status": "strict_proxy_survivor_for_label_review" if strict_pass else "blocked_or_observation",
                "block_reasons": ";".join(reasons) if reasons else "passed_strict_proxy_review",
                "next_action": "queue_for_higher_quality_label_review_not_portfolio" if strict_pass else "keep_observation_or_repair",
                "official_total_return_evidence": False,
                "portfolio_backtest_allowed": False,
                "default_model_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["strict_review_status", "horizon", "source_family", "feature_id"]).reset_index(drop=True)


def build_source_family_strict_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["source_family", "horizon"], as_index=False)
        .agg(
            candidate_count=("feature_id", "count"),
            strict_survivor_count=("strict_review_status", lambda x: int((x == "strict_proxy_survivor_for_label_review").sum())),
            redundancy_block_count=("block_reasons", lambda x: int(x.astype(str).str.contains("redundant_non_representative").sum())),
            negative_control_block_count=("block_reasons", lambda x: int(x.astype(str).str.contains("extended_negative_control_artifact").sum())),
            cross_index_block_count=("block_reasons", lambda x: int(x.astype(str).str.contains("insufficient_cross_index_confirmation").sum())),
            market_trend_proxy_block_count=("block_reasons", lambda x: int(x.astype(str).str.contains("likely_market_trend_proxy").sum())),
            median_cross_signed_spearman=("median_cross_signed_spearman", "median"),
            median_market_trend_corr=("max_abs_market_trend_corr", "median"),
        )
        .sort_values(["horizon", "source_family"])
        .reset_index(drop=True)
    )


def build_no_promotion_guard(decisions: pd.DataFrame) -> pd.DataFrame:
    survivors = int(decisions["strict_review_status"].astype(str).eq("strict_proxy_survivor_for_label_review").sum()) if not decisions.empty else 0
    return pd.DataFrame(
        [
            {
                "result_type": "strict_proxy_candidate_review",
                "produced": True,
                "blocked": False,
                "reason": "reviews proxy-positive rows only",
            },
            {
                "result_type": "higher_quality_label_review_queue",
                "produced": survivors > 0,
                "blocked": survivors == 0,
                "reason": "candidate queue only; not portfolio or model promotion",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.72 uses only price-index proxy labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.72 creates no positions, trades, NAV, or portfolio metrics",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "strict proxy review cannot promote the default model",
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    negative: pd.DataFrame,
    cross: pd.DataFrame,
    trend: pd.DataFrame,
    decisions: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
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
                "check": "proxy_positive_candidates_present",
                "status": _status(not candidates.empty),
                "detail": f"rows={len(candidates)}",
            },
            {
                "check": "redundancy_clusters_produced",
                "status": _status(not clusters.empty),
                "detail": f"rows={len(clusters)}",
            },
            {
                "check": "extended_negative_controls_produced",
                "status": _status(not negative.empty),
                "detail": f"rows={len(negative)}",
            },
            {
                "check": "cross_index_validation_produced",
                "status": _status(not cross.empty),
                "detail": f"rows={len(cross)}",
            },
            {
                "check": "market_trend_proxy_report_produced",
                "status": _status(not trend.empty),
                "detail": f"rows={len(trend)}",
            },
            {
                "check": "strict_decisions_do_not_promote",
                "status": _status(not decisions["default_model_allowed"].astype(bool).any() and not decisions["portfolio_backtest_allowed"].astype(bool).any()),
                "detail": f"decision_rows={len(decisions)}",
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
    candidates: pd.DataFrame,
    label_coverage: pd.DataFrame,
    clusters: pd.DataFrame,
    negative: pd.DataFrame,
    cross: pd.DataFrame,
    trend: pd.DataFrame,
    decisions: pd.DataFrame,
    source_summary: pd.DataFrame,
    input_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
    guard: pd.DataFrame,
) -> str:
    survivor_count = int(decisions["strict_review_status"].astype(str).eq("strict_proxy_survivor_for_label_review").sum()) if not decisions.empty else 0
    top = decisions.loc[decisions["strict_review_status"].astype(str).eq("strict_proxy_survivor_for_label_review")].copy()
    if top.empty and not decisions.empty:
        top = decisions.sort_values(["cross_index_pass_count", "median_cross_signed_spearman", "review_score"], ascending=False).head(20)
    else:
        top = top.sort_values(["median_cross_signed_spearman", "cross_index_pass_count", "review_score"], ascending=False)
    negative_flags = int(negative.drop_duplicates(["feature_id", "horizon"])["extended_negative_control_flag"].astype(bool).sum()) if not negative.empty else 0
    market_flags = int(trend["market_trend_proxy_flag"].astype(bool).sum()) if not trend.empty else 0
    lines = [
        "# V3.72 Strict Proxy Candidate Review",
        "",
        "## Decision",
        "",
        "- V3.72 reviews only V3.71 proxy-positive feature-horizon rows.",
        "- It adds redundancy clustering, extended negative controls, alternate-index proxy validation, and market-trend proxy diagnostics.",
        "- It produces only a higher-quality label-review queue. It does not produce official total-return validation, portfolio backtests, or model promotion.",
        "",
        "## Coverage",
        "",
        f"- V3.71 proxy-positive candidates reviewed: `{len(candidates)}`",
        f"- Redundancy cluster rows: `{len(clusters)}`",
        f"- Extended negative-control feature-horizon flags: `{negative_flags}`",
        f"- Market-trend proxy flags: `{market_flags}`",
        f"- Strict proxy survivors for higher-quality label review: `{survivor_count}`",
        "",
        "## Strict Survivors Or Best Remaining Rows",
        "",
    ]
    lines.extend(
        markdown_table(
            top,
            [
                "feature_id",
                "source_family",
                "horizon",
                "strict_review_status",
                "cross_index_pass_count",
                "median_cross_signed_spearman",
                "is_cluster_representative",
                "extended_negative_control_flag",
                "market_trend_proxy_flag",
                "block_reasons",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Source Family Strict Summary", ""])
    lines.extend(
        markdown_table(
            source_summary,
            [
                "source_family",
                "horizon",
                "candidate_count",
                "strict_survivor_count",
                "redundancy_block_count",
                "negative_control_block_count",
                "cross_index_block_count",
                "market_trend_proxy_block_count",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Alternate Label Coverage", ""])
    lines.extend(markdown_table(label_coverage, ["cross_symbol", "horizon", "label_rows", "overlap_rows", "first_overlap_signal_date", "last_overlap_signal_date"], max_rows=16))
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
            "- Strict survivors can only enter a higher-quality label-source review.",
            "- Blocked rows remain observation or repair candidates.",
            "- No row is allowed to move directly into a portfolio harness from V3.72.",
        ]
    )
    return "\n".join(lines)


def build_catalog(candidates: pd.DataFrame, decisions: pd.DataFrame, config: StrictProxyCandidateReviewConfig) -> str:
    survivors = int(decisions["strict_review_status"].astype(str).eq("strict_proxy_survivor_for_label_review").sum()) if not decisions.empty else 0
    return "\n".join(
        [
            "# A-share Strict Proxy Candidate Review V3.72",
            "",
            "## Dataset Role",
            "",
            "V3.72 reviews V3.71 proxy-positive feature-horizon rows using redundancy, negative-control, alternate-index, and market-trend diagnostics.",
            "",
            "## Governance",
            "",
            "- All labels are non-official price-index proxies.",
            "- No official total-return validation, portfolio backtest, NAV, or model promotion is produced.",
            "- Survivors are only a queue for higher-quality label-source review.",
            "",
            "## Produced Shape",
            "",
            f"- Reviewed candidates: `{len(candidates)}`",
            f"- Strict survivors: `{survivors}`",
            f"- Cross-index sources: `{','.join(sorted(config.cross_index_paths))}`",
        ]
    )
