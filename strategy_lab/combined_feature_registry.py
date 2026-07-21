"""Combined feature registry for HIRSSM V3.70.

V3.70 joins V3.67 market participation, V3.68 industry structure, and V3.69
macro as-of feature layers by next-trade signal date. It is a registry and
governance layer only: it does not create labels, positions, trades, portfolio
outputs, or model-promotion evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class CombinedFeatureRegistryConfig:
    v3_67_manifest_path: Path
    market_panel_path: Path
    v3_68_manifest_path: Path
    industry_panel_path: Path
    v3_69_manifest_path: Path
    macro_panel_path: Path
    output_dir: Path
    catalog_path: Path
    min_combined_rows: int
    min_full_source_rows: int
    min_validation_ready_rows: int
    min_feature_registry_rows: int
    stale_monthly_warn_days: int
    stale_daily_warn_days: int


SOURCE_SPECS = [
    {
        "source_id": "v3_67_market_participation",
        "version": "V3.67",
        "prefix": "market",
        "family": "market_participation_breadth",
        "panel_key": "market",
    },
    {
        "source_id": "v3_68_industry_structure",
        "version": "V3.68",
        "prefix": "industry",
        "family": "industry_breadth_dispersion",
        "panel_key": "industry",
    },
    {
        "source_id": "v3_69_macro_growth_liquidity",
        "version": "V3.69",
        "prefix": "macro",
        "family": "macro_growth_liquidity",
        "panel_key": "macro",
    },
]

SOURCE_PREFIXES = [spec["prefix"] for spec in SOURCE_SPECS]

GOVERNANCE_COLUMN_FRAGMENTS = [
    "model_promotion_allowed",
    "portfolio_harness_allowed",
    "official_total_return_evidence",
    "official_label_evidence",
    "stock_return_label_generated",
    "adjustment_based_label_generated",
    "component_snapshot_used",
]

FEATURE_EXCLUDE_EXACT = {
    "trade_date",
    "source_trade_date",
    "available_date",
    "signal_use_date",
    "signal_timing",
    "feature_family",
    "feature_scope",
    "source_scope",
    "data_scope",
    "price_adjustment",
    "price_adjustment_boundary",
    "asof_join_policy",
    "macro_vintage_limitation",
    "raw_adjustment_guard_status",
}

FEATURE_EXCLUDE_SUFFIXES = (
    "_available_dt",
    "_observation_dt",
    "_source",
    "_frequency",
    "_pit_quality",
    "_revision_policy",
)

MONTHLY_STALENESS_COLUMNS = [
    "macro_china_pmi_staleness_days",
    "macro_china_m2_yoy_staleness_days",
    "macro_china_new_financial_credit_yoy_staleness_days",
    "macro_china_cpi_yoy_staleness_days",
    "macro_china_ppi_yoy_staleness_days",
]

DAILY_STALENESS_COLUMNS = [
    "macro_cn_10y_gov_bond_yield_staleness_days",
    "macro_us_10y_treasury_yield_staleness_days",
    "macro_cn_us_10y_rate_spread_staleness_days",
    "macro_usdcny_staleness_days",
    "macro_commodity_index_staleness_days",
]


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _clean_date_text(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.where(~text.isin(["", "nan", "NaN", "None", "NaT"]), "")


def normalize_date_text(values: pd.Series) -> pd.Series:
    raw = _clean_date_text(values)
    yyyymmdd = raw.str.fullmatch(r"\d{8}", na=False)
    parsed_8 = pd.to_datetime(raw.where(yyyymmdd), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    parsed_other = pd.to_datetime(raw.where(~yyyymmdd), errors="coerce").dt.strftime("%Y-%m-%d")
    out = parsed_8.fillna(parsed_other)
    return out.fillna(raw)


def normalize_date_key(values: pd.Series) -> pd.Series:
    raw = _clean_date_text(values)
    yyyymmdd = raw.str.fullmatch(r"\d{8}", na=False)
    parsed_8 = pd.to_datetime(raw.where(yyyymmdd), format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")
    parsed_other = pd.to_datetime(raw.where(~yyyymmdd), errors="coerce").dt.strftime("%Y%m%d")
    out = parsed_8.fillna(parsed_other)
    fallback = raw.str.replace("-", "", regex=False).str.replace("/", "", regex=False).str[:8]
    return out.fillna(fallback)


def validate_inputs(
    manifests: dict[str, dict[str, Any]],
    panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        key = spec["panel_key"]
        manifest = manifests[key]
        panel = panels[key]
        missing = sorted({"trade_date", "signal_use_date", "history_sufficient"}.difference(panel.columns))
        rows.append(
            {
                "check": f"{key}_manifest_self_check_passed",
                "status": _status(bool(manifest.get("self_check_pass")) and str(manifest.get("status", "")).lower() == "pass"),
                "detail": f"status={manifest.get('status')};self_check={manifest.get('self_check_pass')}",
            }
        )
        rows.append(
            {
                "check": f"{key}_panel_required_columns_present",
                "status": _status(not missing),
                "detail": ",".join(missing),
            }
        )
        rows.append(
            {
                "check": f"{key}_panel_has_rows",
                "status": _status(not panel.empty),
                "detail": f"rows={len(panel)}",
            }
        )
        model_any = _bool_series(panel["model_promotion_allowed"]).any() if "model_promotion_allowed" in panel else False
        portfolio_any = _bool_series(panel["portfolio_harness_allowed"]).any() if "portfolio_harness_allowed" in panel else False
        rows.append(
            {
                "check": f"{key}_source_no_model_or_portfolio_flags",
                "status": _status(not model_any and not portfolio_any),
                "detail": f"model={bool(model_any)};portfolio={bool(portfolio_any)}",
            }
        )
    return pd.DataFrame(rows)


def prepare_source_panel(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "signal_use_date" not in panel.columns:
        raise ValueError(f"{prefix} panel missing signal_use_date")
    data = panel.copy()
    data["signal_use_date"] = normalize_date_text(data["signal_use_date"])
    data = data.loc[data["signal_use_date"].astype(str).str.len().gt(0)].copy()
    data["signal_use_key"] = normalize_date_key(data["signal_use_date"])
    data = data.drop_duplicates("signal_use_key", keep="last").sort_values("signal_use_key")
    rename = {col: f"{prefix}_{col}" for col in data.columns if col not in {"signal_use_date", "signal_use_key"}}
    data = data.rename(columns=rename)
    data[f"{prefix}_source_row_present"] = True
    return data.reset_index(drop=True)


def build_combined_feature_panel(panels: dict[str, pd.DataFrame], config: CombinedFeatureRegistryConfig) -> pd.DataFrame:
    prepared = {}
    for spec in SOURCE_SPECS:
        key = spec["panel_key"]
        prepared[key] = prepare_source_panel(panels[key], spec["prefix"])
    combined: pd.DataFrame | None = None
    for spec in SOURCE_SPECS:
        frame = prepared[spec["panel_key"]]
        if combined is None:
            combined = frame
        else:
            combined = combined.merge(frame, on=["signal_use_date", "signal_use_key"], how="outer")
    if combined is None:
        raise RuntimeError("no source panels to combine")
    combined = combined.sort_values("signal_use_key").reset_index(drop=True)
    combined["signal_date"] = combined["signal_use_key"]
    combined["feature_registry_version"] = "V3.70"

    for prefix in SOURCE_PREFIXES:
        present_col = f"{prefix}_source_row_present"
        combined[present_col] = combined[present_col].fillna(False).astype(bool) if present_col in combined else False
    combined["source_count_available"] = combined[[f"{prefix}_source_row_present" for prefix in SOURCE_PREFIXES]].sum(axis=1).astype(int)
    combined["all_core_sources_available"] = combined["source_count_available"].eq(len(SOURCE_PREFIXES))

    for prefix in SOURCE_PREFIXES:
        history_col = f"{prefix}_history_sufficient"
        if history_col in combined:
            combined[history_col] = _bool_series(combined[history_col])
        else:
            combined[history_col] = False
    combined["history_sufficient_all_sources"] = combined[[f"{prefix}_history_sufficient" for prefix in SOURCE_PREFIXES]].all(axis=1)

    governance_flags = []
    for prefix in SOURCE_PREFIXES:
        for fragment in GOVERNANCE_COLUMN_FRAGMENTS:
            col = f"{prefix}_{fragment}"
            if col in combined:
                flag_col = f"{col}_bool"
                combined[flag_col] = _bool_series(combined[col])
                governance_flags.append((fragment, flag_col))
    blocked_model = pd.Series(False, index=combined.index)
    blocked_portfolio = pd.Series(False, index=combined.index)
    bad_source = pd.Series(False, index=combined.index)
    for fragment, col in governance_flags:
        if fragment in {"model_promotion_allowed", "portfolio_harness_allowed", "official_total_return_evidence", "official_label_evidence", "stock_return_label_generated", "adjustment_based_label_generated", "component_snapshot_used"}:
            bad_source = bad_source | combined[col]
        if fragment == "model_promotion_allowed":
            blocked_model = blocked_model | combined[col]
        if fragment == "portfolio_harness_allowed":
            blocked_portfolio = blocked_portfolio | combined[col]

    monthly_stale = pd.Series(0, index=combined.index, dtype="int64")
    daily_stale = pd.Series(0, index=combined.index, dtype="int64")
    for col in MONTHLY_STALENESS_COLUMNS:
        if col in combined:
            monthly_stale += (pd.to_numeric(combined[col], errors="coerce") > config.stale_monthly_warn_days).fillna(False).astype(int)
    for col in DAILY_STALENESS_COLUMNS:
        if col in combined:
            daily_stale += (pd.to_numeric(combined[col], errors="coerce") > config.stale_daily_warn_days).fillna(False).astype(int)
    combined["macro_monthly_stale_warning_count"] = monthly_stale
    combined["macro_daily_stale_warning_count"] = daily_stale
    combined["macro_any_stale_warning"] = (monthly_stale + daily_stale).gt(0)

    combined["source_governance_block_flag"] = bad_source
    combined["combined_model_promotion_allowed"] = False
    combined["combined_portfolio_harness_allowed"] = False
    combined["combined_feature_validation_ready"] = (
        combined["all_core_sources_available"]
        & combined["history_sufficient_all_sources"]
        & ~combined["source_governance_block_flag"]
    )
    combined["combined_row_status"] = np.select(
        [
            combined["combined_feature_validation_ready"],
            combined["all_core_sources_available"],
            combined["source_count_available"].gt(0),
        ],
        ["ready_for_guarded_validation", "full_source_history_pending_or_warn", "partial_source_available"],
        default="no_source_available",
    )
    governance_bool_cols = [col for _, col in governance_flags]
    if governance_bool_cols:
        combined = combined.drop(columns=governance_bool_cols)
    leading_cols = [
        "signal_date",
        "signal_use_date",
        "feature_registry_version",
        "source_count_available",
        "all_core_sources_available",
        "history_sufficient_all_sources",
        "combined_feature_validation_ready",
        "combined_row_status",
        "macro_monthly_stale_warning_count",
        "macro_daily_stale_warning_count",
        "macro_any_stale_warning",
        "source_governance_block_flag",
        "combined_model_promotion_allowed",
        "combined_portfolio_harness_allowed",
    ]
    return combined[[col for col in leading_cols if col in combined.columns] + [col for col in combined.columns if col not in leading_cols]]


def _is_feature_column(original_col: str) -> bool:
    if original_col in FEATURE_EXCLUDE_EXACT:
        return False
    if original_col.endswith(FEATURE_EXCLUDE_SUFFIXES):
        return False
    if any(fragment == original_col for fragment in GOVERNANCE_COLUMN_FRAGMENTS):
        return False
    if original_col in {"history_sufficient", "history_available"}:
        return True
    return True


def classify_feature(original_col: str) -> str:
    lower = original_col.lower()
    if "state" in lower:
        return "categorical_state"
    if lower.endswith("_score"):
        return "numeric_score"
    if lower.endswith("_z_252d") or "_z_" in lower:
        return "numeric_rolling_zscore"
    if "pctile" in lower:
        return "numeric_trailing_percentile"
    if "ratio" in lower or "share" in lower:
        return "numeric_ratio_or_share"
    if "staleness_days" in lower:
        return "staleness_diagnostic"
    if lower.startswith("history_"):
        return "readiness_flag"
    if lower.endswith("_count") or lower.endswith("_rows"):
        return "count_feature"
    return "numeric_or_categorical_feature"


def build_feature_registry(panels: dict[str, pd.DataFrame], combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        key = spec["panel_key"]
        prefix = spec["prefix"]
        panel = panels[key]
        for col in panel.columns:
            if col in {"signal_use_date"} or not _is_feature_column(col):
                continue
            combined_col = f"{prefix}_{col}"
            if combined_col not in combined.columns:
                continue
            nonnull = combined[combined_col].notna()
            rows.append(
                {
                    "feature_id": combined_col,
                    "source_version": spec["version"],
                    "source_id": spec["source_id"],
                    "source_family": spec["family"],
                    "source_column": col,
                    "combined_column": combined_col,
                    "feature_type": classify_feature(col),
                    "nonnull_rows": int(nonnull.sum()),
                    "coverage_ratio": float(nonnull.mean()) if len(combined) else np.nan,
                    "first_signal_date": combined.loc[nonnull, "signal_date"].min() if nonnull.any() else "",
                    "last_signal_date": combined.loc[nonnull, "signal_date"].max() if nonnull.any() else "",
                    "allowed_next_stage": "guarded_validation_only",
                    "forbidden_use": "no portfolio harness, no model promotion, no official total-return claim",
                    "model_promotion_allowed": False,
                }
            )
    registry = pd.DataFrame(rows)
    return registry.sort_values(["source_version", "feature_type", "feature_id"]).reset_index(drop=True)


def build_source_alignment_checks(
    manifests: dict[str, dict[str, Any]],
    panels: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    config: CombinedFeatureRegistryConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        key = spec["panel_key"]
        prefix = spec["prefix"]
        panel = panels[key].copy()
        signal = normalize_date_text(panel["signal_use_date"])
        nonblank = signal.astype(str).str.len().gt(0)
        duplicate_count = int(signal[nonblank].duplicated().sum())
        trade_dt = pd.to_datetime(normalize_date_text(panel["trade_date"]), errors="coerce")
        signal_dt = pd.to_datetime(signal, errors="coerce")
        future_source_count = int((trade_dt > signal_dt).sum())
        rows.extend(
            [
                {
                    "check": f"{key}_manifest_pass",
                    "status": _status(bool(manifests[key].get("self_check_pass")) and str(manifests[key].get("status", "")).lower() == "pass"),
                    "detail": f"version={manifests[key].get('version')}",
                },
                {
                    "check": f"{key}_nonblank_signal_dates_present",
                    "status": _status(int(nonblank.sum()) > 0),
                    "detail": f"nonblank={int(nonblank.sum())};rows={len(panel)}",
                },
                {
                    "check": f"{key}_signal_dates_unique",
                    "status": _status(duplicate_count == 0),
                    "detail": f"duplicates={duplicate_count}",
                },
                {
                    "check": f"{key}_source_trade_not_after_signal_use_date",
                    "status": _status(future_source_count == 0),
                    "detail": f"future_source_rows={future_source_count}",
                },
            ]
        )
        present_col = f"{prefix}_source_row_present"
        rows.append(
            {
                "check": f"{key}_rows_in_combined_panel",
                "status": _status(present_col in combined.columns and int(combined[present_col].sum()) == int(nonblank.sum())),
                "detail": f"combined_present={int(combined[present_col].sum()) if present_col in combined else 0};source_nonblank={int(nonblank.sum())}",
            }
        )
    full_rows = int(combined["all_core_sources_available"].sum())
    ready_rows = int(combined["combined_feature_validation_ready"].sum())
    rows.extend(
        [
            {
                "check": "combined_panel_minimum_rows",
                "status": _status(len(combined) >= config.min_combined_rows),
                "detail": f"rows={len(combined)};min={config.min_combined_rows}",
            },
            {
                "check": "combined_panel_full_source_rows",
                "status": _status(full_rows >= config.min_full_source_rows),
                "detail": f"full_source_rows={full_rows};min={config.min_full_source_rows}",
            },
            {
                "check": "combined_panel_validation_ready_rows",
                "status": _status(ready_rows >= config.min_validation_ready_rows),
                "detail": f"ready_rows={ready_rows};min={config.min_validation_ready_rows}",
            },
            {
                "check": "macro_stale_warnings_recorded",
                "status": "warn" if bool(combined["macro_any_stale_warning"].any()) else "pass",
                "detail": f"rows={int(combined['macro_any_stale_warning'].sum())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_feature_coverage_report(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    return (
        registry.groupby(["source_version", "source_family", "feature_type"], as_index=False)
        .agg(
            feature_count=("feature_id", "count"),
            median_coverage=("coverage_ratio", "median"),
            min_coverage=("coverage_ratio", "min"),
            max_coverage=("coverage_ratio", "max"),
        )
        .sort_values(["source_version", "source_family", "feature_type"])
        .reset_index(drop=True)
    )


def build_readiness_by_date(combined: pd.DataFrame) -> pd.DataFrame:
    data = combined.copy()
    data["year"] = data["signal_date"].astype(str).str[:4]
    return (
        data.groupby("year", as_index=False)
        .agg(
            rows=("signal_date", "count"),
            full_source_rows=("all_core_sources_available", "sum"),
            ready_rows=("combined_feature_validation_ready", "sum"),
            macro_stale_rows=("macro_any_stale_warning", "sum"),
            median_source_count=("source_count_available", "median"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "combined_feature_registry",
                "produced": True,
                "blocked": False,
                "reason": "V3.70 joins governed feature layers only",
            },
            {
                "result_type": "label_layer",
                "produced": False,
                "blocked": True,
                "reason": "labels require a separate permitted source and validation task",
            },
            {
                "result_type": "portfolio_harness",
                "produced": False,
                "blocked": True,
                "reason": "feature registry has no positions, trades, or performance series",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "requires later guarded validation",
            },
        ]
    )


def build_quality_checks(
    combined: pd.DataFrame,
    registry: pd.DataFrame,
    alignment: pd.DataFrame,
    no_promotion: pd.DataFrame,
    config: CombinedFeatureRegistryConfig,
) -> pd.DataFrame:
    forbidden_output_terms = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(combined.columns).lower()})
    alignment_failures = alignment["status"].eq("fail") if not alignment.empty else pd.Series([True])
    blocked_result_types = {"label_layer", "portfolio_harness", "model_promotion"}
    blocked_correctly = not no_promotion.loc[no_promotion["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "combined_rows_above_threshold",
                "status": _status(len(combined) >= config.min_combined_rows),
                "detail": f"rows={len(combined)};min={config.min_combined_rows}",
            },
            {
                "check": "full_source_rows_above_threshold",
                "status": _status(int(combined["all_core_sources_available"].sum()) >= config.min_full_source_rows),
                "detail": f"rows={int(combined['all_core_sources_available'].sum())};min={config.min_full_source_rows}",
            },
            {
                "check": "validation_ready_rows_above_threshold",
                "status": _status(int(combined["combined_feature_validation_ready"].sum()) >= config.min_validation_ready_rows),
                "detail": f"rows={int(combined['combined_feature_validation_ready'].sum())};min={config.min_validation_ready_rows}",
            },
            {
                "check": "feature_registry_rows_above_threshold",
                "status": _status(len(registry) >= config.min_feature_registry_rows),
                "detail": f"rows={len(registry)};min={config.min_feature_registry_rows}",
            },
            {
                "check": "alignment_checks_have_no_failures",
                "status": _status(not alignment_failures.any()),
                "detail": ";".join(alignment.loc[alignment["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "promotion_outputs_blocked",
                "status": _status(blocked_correctly),
                "detail": "feature registry only",
            },
            {
                "check": "combined_promotion_flags_false",
                "status": _status(
                    not _bool_series(combined["combined_model_promotion_allowed"]).any()
                    and not _bool_series(combined["combined_portfolio_harness_allowed"]).any()
                ),
                "detail": "combined flags",
            },
            {
                "check": "forbidden_promotion_columns_absent",
                "status": _status(not forbidden_output_terms),
                "detail": ",".join(forbidden_output_terms),
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    alignment: pd.DataFrame,
    quality: pd.DataFrame,
    no_promotion: pd.DataFrame,
) -> pd.DataFrame:
    quality_failures = quality["status"].eq("fail") if not quality.empty else pd.Series([True])
    alignment_failures = alignment["status"].eq("fail") if not alignment.empty else pd.Series([True])
    input_failures = input_checks["status"].eq("fail") if not input_checks.empty else pd.Series([True])
    blocked_result_types = {"label_layer", "portfolio_harness", "model_promotion"}
    blocked_correctly = not no_promotion.loc[no_promotion["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if not input_failures.any() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "alignment_checks_passed",
                "status": "pass" if not alignment_failures.any() else "fail",
                "detail": ";".join(alignment.loc[alignment["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "quality_checks_passed",
                "status": "pass" if not quality_failures.any() else "fail",
                "detail": ";".join(quality.loc[quality["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "promotion_outputs_blocked",
                "status": "pass" if blocked_correctly else "fail",
                "detail": "feature registry only",
            },
        ]
    )


def build_feature_contract(config: CombinedFeatureRegistryConfig) -> str:
    return "\n".join(
        [
            "# V3.70 Combined Feature Registry Contract",
            "",
            "## Scope",
            "",
            "- Sources: V3.67 market participation, V3.68 industry structure, and V3.69 macro as-of feature layers.",
            "- Join key: `signal_use_date`, the next trade date when each source feature may be used.",
            "- Output role: feature registry and combined feature panel for later guarded validation.",
            "",
            "## Allowed",
            "",
            "- Source-aligned feature joins with source presence and history-sufficient flags.",
            "- Feature metadata, coverage diagnostics, and stale-data warnings.",
            "- Readiness flags for later validation tasks.",
            "",
            "## Forbidden",
            "",
            "- Return labels, label imports, IC, hit rate, NAV, drawdown, annualized performance, portfolio trades, or model promotion.",
            "- Current constituent or latest weight snapshots.",
            "- Treating stale macro series as fresh observations.",
            "",
            "## Required Next Gate",
            "",
            "A later validation task must join this registry to a permitted label source with explicit lag, purged walk-forward splits, cost-aware diagnostics, and stale-data handling.",
            "",
            f"Minimum full-source rows: {config.min_full_source_rows}.",
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
    combined: pd.DataFrame,
    registry: pd.DataFrame,
    coverage: pd.DataFrame,
    readiness: pd.DataFrame,
    input_checks: pd.DataFrame,
    alignment: pd.DataFrame,
    quality: pd.DataFrame,
    no_promotion: pd.DataFrame,
) -> str:
    latest = combined.sort_values("signal_date").tail(1).iloc[0]
    full_rows = int(combined["all_core_sources_available"].sum())
    ready_rows = int(combined["combined_feature_validation_ready"].sum())
    warn_rows = int(combined["macro_any_stale_warning"].sum())
    lines = [
        "# V3.70 Combined Feature Registry",
        "",
        "## Decision",
        "",
        "- V3.70 joins V3.67, V3.68, and V3.69 feature layers by `signal_use_date`.",
        "- It creates a combined feature panel, feature registry, source alignment checks, and readiness diagnostics.",
        "- It does not produce labels, portfolio outputs, official total-return evidence, or model promotion evidence.",
        "",
        "## Coverage",
        "",
        f"- Combined rows: `{len(combined)}`",
        f"- Full-source rows: `{full_rows}`",
        f"- Validation-ready rows: `{ready_rows}`",
        f"- Registered features: `{len(registry)}`",
        f"- Signal date range: `{combined['signal_date'].min()}` to `{combined['signal_date'].max()}`",
        f"- Macro stale warning rows: `{warn_rows}`",
        "",
        "## Latest Snapshot",
        "",
        f"- Latest signal date: `{latest.signal_date}`",
        f"- Source count available: `{int(latest.source_count_available)}`",
        f"- Row status: `{latest.combined_row_status}`",
        f"- Market score: `{float(latest.market_market_participation_breadth_score):.4f}`" if pd.notna(latest.get("market_market_participation_breadth_score", np.nan)) else "- Market score: `missing`",
        f"- Industry score: `{float(latest.industry_industry_breadth_dispersion_score):.4f}`" if pd.notna(latest.get("industry_industry_breadth_dispersion_score", np.nan)) else "- Industry score: `missing`",
        f"- Macro score: `{float(latest.macro_macro_growth_liquidity_mix_score):.4f}`" if pd.notna(latest.get("macro_macro_growth_liquidity_mix_score", np.nan)) else "- Macro score: `missing`",
        "",
        "## Recent Rows",
        "",
    ]
    lines.extend(
        markdown_table(
            combined.sort_values("signal_date").tail(12),
            [
                "signal_date",
                "source_count_available",
                "all_core_sources_available",
                "history_sufficient_all_sources",
                "combined_feature_validation_ready",
                "combined_row_status",
                "market_market_participation_breadth_score",
                "industry_industry_breadth_dispersion_score",
                "macro_macro_growth_liquidity_mix_score",
                "macro_any_stale_warning",
            ],
            max_rows=12,
        )
    )
    lines.extend(["", "## Feature Coverage", ""])
    lines.extend(markdown_table(coverage, ["source_version", "source_family", "feature_type", "feature_count", "median_coverage", "min_coverage"], max_rows=24))
    lines.extend(["", "## Readiness By Year", ""])
    lines.extend(markdown_table(readiness.tail(12), ["year", "rows", "full_source_rows", "ready_rows", "macro_stale_rows", "median_source_count"], max_rows=12))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## Alignment Checks", ""])
    lines.extend(markdown_table(alignment, ["check", "status", "detail"], max_rows=28))
    lines.extend(["", "## Quality Checks", ""])
    lines.extend(markdown_table(quality, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## No Promotion Guard", ""])
    lines.extend(markdown_table(no_promotion, ["result_type", "produced", "blocked", "reason"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.71 can run guarded feature-label validation using this registry and a permitted label source.",
            "- Validation must explicitly handle V3.69 stale macro warnings and V3.68's shorter latest coverage.",
            "- Do not use V3.70 as a trading strategy by itself.",
        ]
    )
    return "\n".join(lines)


def build_catalog(combined: pd.DataFrame, registry: pd.DataFrame, config: CombinedFeatureRegistryConfig) -> str:
    return "\n".join(
        [
            "# A-share Combined Feature Registry V3.70",
            "",
            "## Dataset Role",
            "",
            "V3.70 joins governed market, industry, and macro feature layers by next-trade signal date.",
            "",
            "## Governance",
            "",
            "- No labels, trades, portfolio outputs, or model promotion are produced.",
            "- Source coverage and history-sufficient flags are preserved.",
            "- Macro stale warnings are retained for later validation.",
            "",
            "## Produced Shape",
            "",
            f"- Combined rows: `{len(combined)}`",
            f"- Registered features: `{len(registry)}`",
            f"- Full-source rows: `{int(combined['all_core_sources_available'].sum())}`",
            f"- Validation-ready rows: `{int(combined['combined_feature_validation_ready'].sum())}`",
            f"- Minimum full-source rows threshold: `{config.min_full_source_rows}`",
        ]
    )
