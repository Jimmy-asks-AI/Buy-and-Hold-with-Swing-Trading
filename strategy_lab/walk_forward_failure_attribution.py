"""Walk-forward failure attribution for HIRSSM V3.65.

V3.65 is a diagnostic step after V3.64. It decomposes failed narrow
walk-forward proxy evidence by year, state, signal-horizon row, and
train-to-OOS drift. It deliberately avoids portfolio outputs and model
promotion.
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
    "market_trend_state",
]


@dataclass(frozen=True)
class WalkForwardFailureAttributionConfig:
    window_results_path: Path
    survivor_panel_path: Path
    oos_summary_path: Path
    v3_64_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    broad_failure_pass_rate: float
    strong_proxy_pass_rate: float
    long_horizons: tuple[int, ...]
    retire_long_horizon_median_spearman: float
    min_state_rows: int
    top_quantile: float
    bottom_quantile: float
    state_min_spearman: float
    state_min_qspread: float
    state_min_top_alignment: float
    max_broad_failure_row_share_for_state_retest: float
    min_strong_proxy_row_share_for_state_retest: float


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _as_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def _safe_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else np.nan


def _reason_summary(values: pd.Series, max_items: int = 4) -> str:
    counts: dict[str, int] = {}
    for raw in values.dropna().astype(str):
        if raw == "passed":
            continue
        for item in raw.split(";"):
            reason = item.strip()
            if not reason or reason == "passed":
                continue
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_items]
    return ";".join(f"{reason}:{count}" for reason, count in ordered)


def _pass_rate_class(pass_rate: float, config: WalkForwardFailureAttributionConfig) -> str:
    if pd.isna(pass_rate):
        return "no_train_gated_windows"
    if pass_rate <= config.broad_failure_pass_rate:
        return "broad_failure_year"
    if pass_rate >= config.strong_proxy_pass_rate:
        return "strong_proxy_year"
    return "partial_success_year"


def _metric_group(group: pd.DataFrame, config: WalkForwardFailureAttributionConfig) -> dict[str, Any]:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    observations = int(len(clean))
    if observations == 0:
        return {
            "row_count": 0,
            "proxy_spearman_corr": np.nan,
            "proxy_qspread_top_minus_bottom": np.nan,
            "top_bucket_directional_alignment_share": np.nan,
            "directional_alignment_share": np.nan,
            "expected_signed_proxy_mean": np.nan,
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
        "row_count": observations,
        "proxy_spearman_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "spearman"),
        "proxy_qspread_top_minus_bottom": top_mean - bottom_mean if pd.notna(top_mean) and pd.notna(bottom_mean) else np.nan,
        "top_bucket_directional_alignment_share": float(top["active_directional_alignment"].mean()) if not top.empty else np.nan,
        "directional_alignment_share": float(clean["active_directional_alignment"].mean()),
        "expected_signed_proxy_mean": float(clean["expected_signed_proxy_return"].mean()),
        "top_bucket_rows": int(len(top)),
        "bottom_bucket_rows": int(len(bottom)),
    }


def normalize_inputs(windows: pd.DataFrame, panel: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = windows.copy()
    panel = panel.copy()
    summary = summary.copy()

    for col in [
        "horizon",
        "train_start_year",
        "train_end_year",
        "test_start_year",
        "test_end_year",
        "train_observations",
        "test_observations",
    ]:
        if col in windows.columns:
            windows[col] = pd.to_numeric(windows[col], errors="coerce")
    for col in [
        "train_proxy_spearman_corr",
        "train_proxy_qspread_top_minus_bottom",
        "train_top_bucket_directional_alignment_share",
        "oos_proxy_spearman_corr",
        "oos_proxy_qspread_top_minus_bottom",
        "oos_top_bucket_directional_alignment_share",
        "oos_directional_alignment_share",
        "oos_expected_signed_proxy_mean",
    ]:
        if col in windows.columns:
            windows[col] = pd.to_numeric(windows[col], errors="coerce")
    for col in ["train_gate_pass", "oos_gate_pass", "train_gate_and_oos_pass"]:
        if col in windows.columns:
            windows[col] = _bool_series(windows[col])

    if "signal_date" in panel.columns:
        panel["signal_date"] = normalize_date(panel["signal_date"])
        panel["signal_year"] = pd.to_datetime(panel["signal_date"], format="%Y%m%d", errors="coerce").dt.year
    elif "signal_year" in panel.columns:
        panel["signal_year"] = pd.to_numeric(panel["signal_year"], errors="coerce")
    for col in ["horizon", "signal_value", "expected_signed_proxy_return", "active_directional_alignment"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    if "horizon" in summary.columns:
        summary["horizon"] = pd.to_numeric(summary["horizon"], errors="coerce")
    for col in [
        "total_windows",
        "train_gated_windows",
        "oos_gate_pass_windows",
        "oos_gate_pass_rate_on_gated_windows",
        "oos_median_proxy_spearman_corr",
        "oos_positive_qspread_share",
        "oos_median_top_bucket_directional_alignment_share",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return windows, panel, summary


def validate_inputs(
    windows: pd.DataFrame,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    v3_64_manifest: dict[str, Any],
) -> pd.DataFrame:
    required_windows = {
        "signal_id",
        "horizon",
        "test_start_year",
        "train_gate_pass",
        "oos_gate_pass",
        "train_gate_and_oos_pass",
        "oos_gate_reason",
        "train_proxy_spearman_corr",
        "oos_proxy_spearman_corr",
    }
    required_panel = {
        "signal_id",
        "horizon",
        "signal_date",
        "signal_value",
        "expected_signed_proxy_return",
        "active_directional_alignment",
        "market_trend_state",
    }
    required_summary = {"signal_id", "horizon", "walk_forward_proxy_review_status"}
    missing_windows = sorted(required_windows.difference(windows.columns))
    missing_panel = sorted(required_panel.difference(panel.columns))
    missing_summary = sorted(required_summary.difference(summary.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "v3_64_manifest_self_check_passed",
            "status": _status(bool(v3_64_manifest.get("self_check_pass"))),
            "detail": f"self_check={v3_64_manifest.get('self_check_pass')}",
        },
        {
            "check": "window_required_columns_present",
            "status": _status(not missing_windows),
            "detail": ",".join(missing_windows),
        },
        {
            "check": "panel_required_columns_present",
            "status": _status(not missing_panel),
            "detail": ",".join(missing_panel),
        },
        {
            "check": "summary_required_columns_present",
            "status": _status(not missing_summary),
            "detail": ",".join(missing_summary),
        },
        {
            "check": "input_rows_present",
            "status": _status(not windows.empty and not panel.empty and not summary.empty),
            "detail": f"windows={len(windows)};panel={len(panel)};summary={len(summary)}",
        },
    ]
    if not windows.empty and "train_gate_and_oos_pass" in windows.columns:
        rows.append(
            {
                "check": "v3_64_has_oos_pass_evidence_to_explain",
                "status": "warn" if int(windows["train_gate_and_oos_pass"].sum()) > 0 else "pass",
                "detail": f"train_and_oos_pass_rows={int(windows['train_gate_and_oos_pass'].sum())}",
            }
        )
    return pd.DataFrame(rows)


def build_yearly_failure_attribution(windows: pd.DataFrame, config: WalkForwardFailureAttributionConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if windows.empty:
        return pd.DataFrame(rows)
    for year, group in windows.groupby("test_start_year", dropna=False):
        gated = group.loc[group["train_gate_pass"].astype(bool)].copy()
        gated_count = int(len(gated))
        pass_count = int(gated["oos_gate_pass"].astype(bool).sum()) if gated_count else 0
        pass_rate = float(pass_count / gated_count) if gated_count else np.nan
        rows.append(
            {
                "test_start_year": int(year),
                "candidate_window_rows": int(len(group)),
                "train_gated_window_rows": gated_count,
                "gate_pass_window_rows": pass_count,
                "oos_pass_rate_on_gated": pass_rate,
                "median_oos_spearman_on_gated": _safe_median(gated["oos_proxy_spearman_corr"]) if gated_count else np.nan,
                "median_oos_qspread_on_gated": _safe_median(gated["oos_proxy_qspread_top_minus_bottom"]) if gated_count else np.nan,
                "median_top_alignment_on_gated": _safe_median(gated["oos_top_bucket_directional_alignment_share"]) if gated_count else np.nan,
                "dominant_train_fail_reasons": _reason_summary(group.loc[~group["train_gate_pass"].astype(bool), "train_gate_reason"]),
                "dominant_oos_fail_reasons_on_gated": _reason_summary(gated.loc[~gated["oos_gate_pass"].astype(bool), "oos_gate_reason"]),
                "year_failure_class": _pass_rate_class(pass_rate, config),
            }
        )
    return pd.DataFrame(rows).sort_values("test_start_year").reset_index(drop=True)


def build_signal_horizon_failure_attribution(
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    config: WalkForwardFailureAttributionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if windows.empty:
        return pd.DataFrame(rows)
    summary_key = summary.set_index(["signal_id", "horizon"], drop=False) if not summary.empty else pd.DataFrame()
    for (signal_id, horizon), group in windows.groupby(["signal_id", "horizon"], dropna=False):
        horizon_int = int(horizon)
        gated = group.loc[group["train_gate_pass"].astype(bool)].copy()
        gated_count = int(len(gated))
        pass_count = int(gated["oos_gate_pass"].astype(bool).sum()) if gated_count else 0
        pass_rate = float(pass_count / gated_count) if gated_count else np.nan
        median_oos_spearman = _safe_median(gated["oos_proxy_spearman_corr"]) if gated_count else np.nan
        median_oos_qspread = _safe_median(gated["oos_proxy_qspread_top_minus_bottom"]) if gated_count else np.nan
        status = "not_in_summary"
        if not summary.empty and (signal_id, horizon_int) in summary_key.index:
            status = str(summary_key.loc[(signal_id, horizon_int), "walk_forward_proxy_review_status"])
        if horizon_int in config.long_horizons and (pd.isna(median_oos_spearman) or median_oos_spearman < config.retire_long_horizon_median_spearman) and (pd.isna(pass_rate) or pass_rate < 0.30):
            decision = "retire_long_horizon_proxy_branch"
            next_action = "do_not_repair_until_official_label_or_new_source_exists"
        elif gated_count == 0:
            decision = "retire_insufficient_train_gate"
            next_action = "replace_source_definition_before_retesting"
        elif pass_count == 0 and pd.notna(median_oos_spearman) and median_oos_spearman < 0:
            decision = "retire_or_rebuild_from_new_source"
            next_action = "do_not_state_tune_failed_proxy_relation"
        elif horizon_int not in config.long_horizons and pass_count > 0:
            decision = "state_conditioned_repair_candidate"
            next_action = "test only predeclared state filters in next version"
        else:
            decision = "observation_only_repair_required"
            next_action = "keep out of default model and require independent source"
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": horizon_int,
                "signal_family": group["signal_family"].dropna().astype(str).iloc[0] if "signal_family" in group and group["signal_family"].notna().any() else "",
                "total_window_rows": int(len(group)),
                "train_gated_window_rows": gated_count,
                "gate_pass_window_rows": pass_count,
                "oos_pass_rate_on_gated": pass_rate,
                "median_train_spearman": _safe_median(gated["train_proxy_spearman_corr"]) if gated_count else np.nan,
                "median_oos_spearman": median_oos_spearman,
                "median_oos_qspread": median_oos_qspread,
                "median_top_alignment": _safe_median(gated["oos_top_bucket_directional_alignment_share"]) if gated_count else np.nan,
                "median_spearman_drift": _safe_median(gated["oos_proxy_spearman_corr"] - gated["train_proxy_spearman_corr"]) if gated_count else np.nan,
                "positive_qspread_share": float((gated["oos_proxy_qspread_top_minus_bottom"] > 0).mean()) if gated_count else np.nan,
                "dominant_train_fail_reasons": _reason_summary(group.loc[~group["train_gate_pass"].astype(bool), "train_gate_reason"]),
                "dominant_oos_fail_reasons_on_gated": _reason_summary(gated.loc[~gated["oos_gate_pass"].astype(bool), "oos_gate_reason"]),
                "v3_64_review_status": status,
                "repair_or_retire_decision": decision,
                "next_action": next_action,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_train_oos_drift(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    drift = windows.loc[
        :,
        [
            "signal_id",
            "horizon",
            "test_start_year",
            "train_gate_pass",
            "oos_gate_pass",
            "train_proxy_spearman_corr",
            "oos_proxy_spearman_corr",
            "train_proxy_qspread_top_minus_bottom",
            "oos_proxy_qspread_top_minus_bottom",
            "train_top_bucket_directional_alignment_share",
            "oos_top_bucket_directional_alignment_share",
            "oos_gate_reason",
        ],
    ].copy()
    drift["spearman_drift_oos_minus_train"] = drift["oos_proxy_spearman_corr"] - drift["train_proxy_spearman_corr"]
    drift["qspread_drift_oos_minus_train"] = drift["oos_proxy_qspread_top_minus_bottom"] - drift["train_proxy_qspread_top_minus_bottom"]
    drift["top_alignment_drift_oos_minus_train"] = (
        drift["oos_top_bucket_directional_alignment_share"] - drift["train_top_bucket_directional_alignment_share"]
    )
    conditions = [
        (~drift["train_gate_pass"].astype(bool)),
        (drift["train_gate_pass"].astype(bool) & drift["oos_gate_pass"].astype(bool)),
        (drift["spearman_drift_oos_minus_train"] < -0.10),
        (drift["qspread_drift_oos_minus_train"] < -0.003),
    ]
    choices = [
        "not_train_gated",
        "oos_gate_passed",
        "large_spearman_decay",
        "large_qspread_decay",
    ]
    drift["drift_class"] = np.select(conditions, choices, default="mixed_or_threshold_failure")
    return drift.sort_values(["horizon", "signal_id", "test_start_year"]).reset_index(drop=True)


def build_year_regime_attribution(
    panel: pd.DataFrame,
    yearly: pd.DataFrame,
    config: WalkForwardFailureAttributionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if panel.empty or "market_trend_state" not in panel.columns:
        return pd.DataFrame(rows)
    year_map = yearly.set_index("test_start_year") if not yearly.empty else pd.DataFrame()
    for (year, trend_state, horizon), group in panel.groupby(["signal_year", "market_trend_state", "horizon"], dropna=False):
        metrics = _metric_group(group, config)
        failure_class = ""
        pass_rate = np.nan
        if not yearly.empty and year in year_map.index:
            failure_class = str(year_map.loc[year, "year_failure_class"])
            pass_rate = _as_float(year_map.loc[year, "oos_pass_rate_on_gated"])
        rows.append(
            {
                "signal_year": int(year) if pd.notna(year) else -1,
                "market_trend_state": str(trend_state),
                "horizon": int(horizon),
                "row_count": metrics["row_count"],
                "proxy_spearman_corr": metrics["proxy_spearman_corr"],
                "proxy_qspread_top_minus_bottom": metrics["proxy_qspread_top_minus_bottom"],
                "top_bucket_directional_alignment_share": metrics["top_bucket_directional_alignment_share"],
                "directional_alignment_share": metrics["directional_alignment_share"],
                "expected_signed_proxy_mean": metrics["expected_signed_proxy_mean"],
                "year_failure_class": failure_class,
                "year_oos_pass_rate_on_gated": pass_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["signal_year", "horizon", "market_trend_state"]).reset_index(drop=True)


def build_state_failure_attribution(
    panel: pd.DataFrame,
    yearly: pd.DataFrame,
    config: WalkForwardFailureAttributionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if panel.empty:
        return pd.DataFrame(rows)
    yearly_classes = yearly.set_index("test_start_year")["year_failure_class"].to_dict() if not yearly.empty else {}
    work = panel.copy()
    work["year_failure_class"] = work["signal_year"].map(yearly_classes)
    for state_col in [col for col in STATE_COLUMNS if col in work.columns]:
        for (state_value, signal_id, horizon), group in work.groupby([state_col, "signal_id", "horizon"], dropna=False):
            metrics = _metric_group(group, config)
            classified_years = group["year_failure_class"].dropna()
            broad_share = float((classified_years == "broad_failure_year").mean()) if len(classified_years) else np.nan
            strong_share = float((classified_years == "strong_proxy_year").mean()) if len(classified_years) else np.nan
            if metrics["row_count"] < config.min_state_rows:
                state_decision = "insufficient_state_rows"
            elif (
                pd.notna(metrics["proxy_spearman_corr"])
                and metrics["proxy_spearman_corr"] >= config.state_min_spearman
                and pd.notna(metrics["proxy_qspread_top_minus_bottom"])
                and metrics["proxy_qspread_top_minus_bottom"] > config.state_min_qspread
                and pd.notna(metrics["top_bucket_directional_alignment_share"])
                and metrics["top_bucket_directional_alignment_share"] >= config.state_min_top_alignment
                and pd.notna(broad_share)
                and broad_share <= config.max_broad_failure_row_share_for_state_retest
                and pd.notna(strong_share)
                and strong_share >= config.min_strong_proxy_row_share_for_state_retest
            ):
                state_decision = "potential_state_condition_retest"
            elif (
                pd.notna(metrics["proxy_spearman_corr"])
                and metrics["proxy_spearman_corr"] >= config.state_min_spearman
                and pd.notna(metrics["proxy_qspread_top_minus_bottom"])
                and metrics["proxy_qspread_top_minus_bottom"] > config.state_min_qspread
                and pd.notna(metrics["top_bucket_directional_alignment_share"])
                and metrics["top_bucket_directional_alignment_share"] >= config.state_min_top_alignment
            ):
                state_decision = "state_metric_pass_but_broad_failure_exposed"
            elif pd.notna(metrics["proxy_spearman_corr"]) and metrics["proxy_spearman_corr"] < 0:
                state_decision = "weak_or_inverted_state_relation"
            else:
                state_decision = "observation_only_state_condition"
            rows.append(
                {
                    "state_column": state_col,
                    "state_value": str(state_value),
                    "signal_id": signal_id,
                    "horizon": int(horizon),
                    "row_count": metrics["row_count"],
                    "years_covered": int(group["signal_year"].nunique()),
                    "broad_failure_year_row_share": broad_share,
                    "strong_proxy_year_row_share": strong_share,
                    "proxy_spearman_corr": metrics["proxy_spearman_corr"],
                    "proxy_qspread_top_minus_bottom": metrics["proxy_qspread_top_minus_bottom"],
                    "top_bucket_directional_alignment_share": metrics["top_bucket_directional_alignment_share"],
                    "directional_alignment_share": metrics["directional_alignment_share"],
                    "state_decision": state_decision,
                }
            )
    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values(["state_decision", "horizon", "signal_id", "state_column", "state_value"]).reset_index(drop=True)


def build_retire_repair_decisions(
    signal_attr: pd.DataFrame,
    state_attr: pd.DataFrame,
    yearly: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if signal_attr.empty:
        return pd.DataFrame(rows)
    state_candidates = set()
    if not state_attr.empty:
        candidates = state_attr.loc[state_attr["state_decision"].eq("potential_state_condition_retest"), ["signal_id", "horizon"]]
        state_candidates = {(row.signal_id, int(row.horizon)) for row in candidates.itertuples(index=False)}
    strong_years = yearly.loc[yearly["year_failure_class"].eq("strong_proxy_year"), "test_start_year"].astype(int).tolist() if not yearly.empty else []
    partial_years = yearly.loc[yearly["year_failure_class"].eq("partial_success_year"), "test_start_year"].astype(int).tolist() if not yearly.empty else []
    for row in signal_attr.itertuples(index=False):
        key = (row.signal_id, int(row.horizon))
        if row.repair_or_retire_decision.startswith("retire"):
            final_action = row.repair_or_retire_decision
        elif key in state_candidates:
            final_action = "repair_with_predeclared_state_filter_only"
        else:
            final_action = "keep_observation_only"
        rows.append(
            {
                "signal_id": row.signal_id,
                "horizon": int(row.horizon),
                "failure_diagnosis": row.repair_or_retire_decision,
                "final_action": final_action,
                "allowed_next_test": row.next_action,
                "supporting_years": ",".join(str(year) for year in strong_years + partial_years[:4]),
                "evidence_basis": "V3.64 rolling proxy OOS windows plus state attribution",
                "promotion_block_reason": "non_official_price_proxy_and_failed_walk_forward_stability",
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "failure_attribution_diagnostics",
                "produced": True,
                "blocked": False,
                "reason": "V3.65 explains V3.64 proxy OOS failure only",
            },
            {
                "result_type": "investable_model_change",
                "produced": False,
                "blocked": True,
                "reason": "diagnostic step cannot modify default model",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "no positions, trades, or performance series are generated",
            },
            {
                "result_type": "official_total_return_review",
                "produced": False,
                "blocked": True,
                "reason": "input evidence remains price-index proxy only",
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    yearly: pd.DataFrame,
    signal_attr: pd.DataFrame,
    state_attr: pd.DataFrame,
    decisions: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    blocked_types = {"investable_model_change", "portfolio_backtest", "official_total_return_review"}
    blocked_correctly = not guard.loc[guard["result_type"].isin(blocked_types), "produced"].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "input_checks_block_no_hard_failures",
                "status": "pass" if not input_checks["status"].eq("fail").any() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "yearly_failure_attribution_produced",
                "status": _status(not yearly.empty),
                "detail": f"rows={len(yearly)}",
            },
            {
                "check": "signal_horizon_attribution_produced",
                "status": _status(not signal_attr.empty),
                "detail": f"rows={len(signal_attr)}",
            },
            {
                "check": "state_attribution_produced",
                "status": _status(not state_attr.empty),
                "detail": f"rows={len(state_attr)}",
            },
            {
                "check": "retire_repair_decisions_produced",
                "status": _status(not decisions.empty),
                "detail": f"rows={len(decisions)}",
            },
            {
                "check": "promotion_outputs_blocked",
                "status": _status(blocked_correctly and not forbidden_columns),
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
    yearly: pd.DataFrame,
    signal_attr: pd.DataFrame,
    drift: pd.DataFrame,
    state_attr: pd.DataFrame,
    decisions: pd.DataFrame,
    input_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    broad_years = yearly.loc[yearly["year_failure_class"].eq("broad_failure_year"), "test_start_year"].astype(int).tolist() if not yearly.empty else []
    strong_years = yearly.loc[yearly["year_failure_class"].eq("strong_proxy_year"), "test_start_year"].astype(int).tolist() if not yearly.empty else []
    state_retests = int(state_attr["state_decision"].eq("potential_state_condition_retest").sum()) if not state_attr.empty else 0
    final_state_retests = int(decisions["final_action"].eq("repair_with_predeclared_state_filter_only").sum()) if not decisions.empty else 0
    retire_rows = int(decisions["final_action"].astype(str).str.startswith("retire").sum()) if not decisions.empty else 0
    lines = [
        "# V3.65 Walk-Forward Failure Attribution",
        "",
        "## Decision",
        "",
        "- V3.65 diagnoses why V3.64 survivor rows failed rolling OOS proxy review.",
        "- It does not run a portfolio harness or promote any signal.",
        "- The main research decision is whether to retire this branch or only allow tightly predeclared state-condition repairs.",
        "",
        "## Headline Findings",
        "",
        f"- Broad failure years: `{','.join(str(year) for year in broad_years)}`",
        f"- Strong proxy years: `{','.join(str(year) for year in strong_years)}`",
        f"- Signal-horizon rows marked retire: `{retire_rows}`",
        f"- State-condition diagnostic rows: `{state_retests}`",
        f"- Final state-filter signal rows: `{final_state_retests}`",
        "",
        "## Yearly Attribution",
        "",
    ]
    lines.extend(
        markdown_table(
            yearly,
            [
                "test_start_year",
                "candidate_window_rows",
                "train_gated_window_rows",
                "gate_pass_window_rows",
                "oos_pass_rate_on_gated",
                "median_oos_spearman_on_gated",
                "median_oos_qspread_on_gated",
                "year_failure_class",
                "dominant_oos_fail_reasons_on_gated",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Signal-Horizon Decisions", ""])
    lines.extend(
        markdown_table(
            signal_attr,
            [
                "signal_id",
                "horizon",
                "train_gated_window_rows",
                "gate_pass_window_rows",
                "oos_pass_rate_on_gated",
                "median_oos_spearman",
                "median_spearman_drift",
                "repair_or_retire_decision",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Drift Sample", ""])
    lines.extend(
        markdown_table(
            drift,
            [
                "signal_id",
                "horizon",
                "test_start_year",
                "train_gate_pass",
                "oos_gate_pass",
                "spearman_drift_oos_minus_train",
                "qspread_drift_oos_minus_train",
                "drift_class",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## State Retest Candidates", ""])
    state_candidates = state_attr.loc[state_attr["state_decision"].eq("potential_state_condition_retest")] if not state_attr.empty else state_attr
    lines.extend(
        markdown_table(
            state_candidates,
            [
                "state_column",
                "state_value",
                "signal_id",
                "horizon",
                "row_count",
                "proxy_spearman_corr",
                "proxy_qspread_top_minus_bottom",
                "top_bucket_directional_alignment_share",
                "state_decision",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Final Repair/Retire Queue", ""])
    lines.extend(markdown_table(decisions, ["signal_id", "horizon", "failure_diagnosis", "final_action", "allowed_next_test"], max_rows=30))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=16))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.66 should avoid broad parameter tuning on the same proxy branch.",
            "- If continuing this branch, only test predeclared state filters from this report.",
            "- The safer research route is to discover independent PIT sources rather than keep tuning the same MARKET proxy relation.",
        ]
    )
    return "\n".join(lines)


def build_catalog(
    yearly: pd.DataFrame,
    signal_attr: pd.DataFrame,
    decisions: pd.DataFrame,
    config: WalkForwardFailureAttributionConfig,
) -> str:
    broad_years = int(yearly["year_failure_class"].eq("broad_failure_year").sum()) if not yearly.empty else 0
    retire_rows = int(decisions["final_action"].astype(str).str.startswith("retire").sum()) if not decisions.empty else 0
    return "\n".join(
        [
            "# A-share Walk-Forward Failure Attribution V3.65",
            "",
            "## Dataset Role",
            "",
            "V3.65 explains V3.64 narrow walk-forward proxy review failures by year, state, signal horizon, and train-to-OOS drift.",
            "",
            "## Governance",
            "",
            "- Return basis: price-index proxy diagnostics inherited from V3.64.",
            "- Portfolio harness: not produced.",
            "- Default model change: not allowed.",
            "- Official dividend-inclusive evidence: not produced.",
            "",
            "## Configuration",
            "",
            f"- Broad failure pass-rate threshold: `{config.broad_failure_pass_rate}`",
            f"- Strong proxy pass-rate threshold: `{config.strong_proxy_pass_rate}`",
            f"- Long-horizon retirement set: `{','.join(str(x) for x in config.long_horizons)}`",
            "",
            "## Produced Shape",
            "",
            f"- Year rows: `{len(yearly)}`",
            f"- Signal-horizon rows: `{len(signal_attr)}`",
            f"- Broad failure years: `{broad_years}`",
            f"- Retire rows: `{retire_rows}`",
        ]
    )
