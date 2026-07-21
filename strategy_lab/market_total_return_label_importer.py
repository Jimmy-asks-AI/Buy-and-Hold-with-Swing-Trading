"""Strict MARKET total-return source importer and label builder for V3.53."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]

LABEL_COLUMNS = [
    "signal_date",
    "asset",
    "horizon",
    "forward_adjusted_return",
    "return_basis",
    "label_available_date",
    "price_adjustment_source",
]

FORBIDDEN_LINEAGE_TERMS = {
    "backtest",
    "benchmark",
    "strategy_nav",
    "portfolio_nav",
    "raw_close",
    "none_raw",
    "unadjusted",
    "price_only",
}


@dataclass(frozen=True)
class MarketLabelImportConfig:
    source_path: Path
    signal_panel_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    market_proxy_codes: tuple[str, ...]
    source_asset_priority: tuple[str, ...]
    min_source_rows: int
    min_signal_coverage_ratio: float


def contract_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "date",
                "required": True,
                "type": "YYYYMMDD or YYYY-MM-DD",
                "rule": "observation date of total-return index level",
                "example": "20000104",
            },
            {
                "column": "asset_or_index",
                "required": True,
                "type": "string",
                "rule": "MARKET or approved broad-market index code",
                "example": "MARKET",
            },
            {
                "column": "total_return_index_or_adjusted_close",
                "required": True,
                "type": "positive numeric",
                "rule": "total-return index level or explicitly adjusted market close",
                "example": "1000.0",
            },
            {
                "column": "available_date",
                "required": True,
                "type": "YYYYMMDD or YYYY-MM-DD",
                "rule": "date when the observation became available; must be >= date",
                "example": "20000105",
            },
            {
                "column": "data_source",
                "required": True,
                "type": "string",
                "rule": "provider and endpoint/file lineage; cannot be backtest or raw price lineage",
                "example": "provider.total_return_index",
            },
            {
                "column": "source_vintage",
                "required": True,
                "type": "string",
                "rule": "provider file vintage or ingestion batch id",
                "example": "vendor_file_20260529",
            },
        ]
    )


def source_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "20000104",
                "asset_or_index": "MARKET",
                "total_return_index_or_adjusted_close": "1000.0",
                "available_date": "20000105",
                "data_source": "example_provider.total_return_index",
                "source_vintage": "example_vintage_YYYYMMDD",
            },
            {
                "date": "20000105",
                "asset_or_index": "MARKET",
                "total_return_index_or_adjusted_close": "1001.5",
                "available_date": "20000106",
                "data_source": "example_provider.total_return_index",
                "source_vintage": "example_vintage_YYYYMMDD",
            },
        ]
    )


def label_schema_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "YYYYMMDD",
                "asset": "MARKET",
                "horizon": 20,
                "forward_adjusted_return": "future_total_return_level/current_total_return_level - 1",
                "return_basis": "total_return",
                "label_available_date": "available_date of horizon-end source observation",
                "price_adjustment_source": "data_source|source_vintage|asset_or_index",
            }
        ],
        columns=LABEL_COLUMNS,
    )


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def normalize_date(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
    )
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def format_date(values: pd.Series) -> pd.Series:
    parsed = normalize_date(values)
    return parsed.dt.strftime("%Y%m%d")


def read_optional_csv(path: Path) -> tuple[bool, pd.DataFrame, str]:
    if not path.exists():
        return False, pd.DataFrame(), f"missing_source_path={path}"
    try:
        return True, pd.read_csv(path, encoding="utf-8-sig", low_memory=False), ""
    except Exception as exc:  # pragma: no cover - defensive file handling.
        return True, pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().ne("") & series.notna()


def validate_market_source(source: pd.DataFrame, config: MarketLabelImportConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in source.columns]
    rows.append(
        {
            "check": "required_columns_present",
            "status": "pass" if not missing else "fail",
            "detail": ",".join(missing),
        }
    )
    if missing:
        return pd.DataFrame(rows)

    work = source.copy()
    work["_date"] = normalize_date(work["date"])
    work["_available_date"] = normalize_date(work["available_date"])
    work["_value"] = pd.to_numeric(work["total_return_index_or_adjusted_close"], errors="coerce")
    source_text = (
        work["data_source"].astype(str)
        + " "
        + work["source_vintage"].astype(str)
        + " "
        + work["asset_or_index"].astype(str)
    ).str.lower()
    forbidden_hit = source_text.apply(lambda text: any(term in text for term in FORBIDDEN_LINEAGE_TERMS))
    duplicate_count = int(work.duplicated(["date", "asset_or_index"]).sum())
    approved_assets = set(config.market_proxy_codes)
    asset_ok = work["asset_or_index"].astype(str).isin(approved_assets)

    rows.extend(
        [
            {
                "check": "minimum_rows",
                "status": "pass" if len(work) >= config.min_source_rows else "fail",
                "detail": f"rows={len(work)};min={config.min_source_rows}",
            },
            {
                "check": "date_parseable",
                "status": "pass" if work["_date"].notna().all() else "fail",
                "detail": f"bad_rows={int(work['_date'].isna().sum())}",
            },
            {
                "check": "available_date_parseable",
                "status": "pass" if work["_available_date"].notna().all() else "fail",
                "detail": f"bad_rows={int(work['_available_date'].isna().sum())}",
            },
            {
                "check": "available_date_not_before_observation_date",
                "status": "pass" if (work["_available_date"] >= work["_date"]).all() else "fail",
                "detail": f"bad_rows={int((work['_available_date'] < work['_date']).sum())}",
            },
            {
                "check": "positive_numeric_total_return_level",
                "status": "pass" if np.isfinite(work["_value"]).all() and (work["_value"] > 0).all() else "fail",
                "detail": f"bad_rows={int((~np.isfinite(work['_value']) | (work['_value'] <= 0)).sum())}",
            },
            {
                "check": "approved_market_asset_identifier",
                "status": "pass" if asset_ok.all() else "fail",
                "detail": f"bad_rows={int((~asset_ok).sum())}",
            },
            {
                "check": "no_duplicate_date_asset",
                "status": "pass" if duplicate_count == 0 else "fail",
                "detail": f"duplicates={duplicate_count}",
            },
            {
                "check": "data_source_nonempty",
                "status": "pass" if _nonempty_text(work["data_source"]).all() else "fail",
                "detail": f"bad_rows={int((~_nonempty_text(work['data_source'])).sum())}",
            },
            {
                "check": "source_vintage_nonempty",
                "status": "pass" if _nonempty_text(work["source_vintage"]).all() else "fail",
                "detail": f"bad_rows={int((~_nonempty_text(work['source_vintage'])).sum())}",
            },
            {
                "check": "forbidden_backtest_or_price_only_lineage_absent",
                "status": "pass" if not forbidden_hit.any() else "fail",
                "detail": f"bad_rows={int(forbidden_hit.sum())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def source_validation_passed(checks: pd.DataFrame) -> bool:
    return bool(not checks.empty and checks["status"].eq("pass").all())


def select_source_asset(source: pd.DataFrame, config: MarketLabelImportConfig) -> tuple[str, pd.DataFrame, str]:
    if source.empty:
        return "", pd.DataFrame(), "empty_source"
    for asset in config.source_asset_priority:
        subset = source.loc[source["asset_or_index"].astype(str) == asset].copy()
        if not subset.empty:
            return asset, subset, "selected_by_priority"
    unique_assets = sorted(source["asset_or_index"].astype(str).unique())
    if len(unique_assets) == 1:
        asset = unique_assets[0]
        return asset, source.copy(), "selected_single_available_asset"
    return "", pd.DataFrame(), "no_priority_asset_available"


def prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    work = source.copy()
    work["_date"] = normalize_date(work["date"])
    work["_available_date"] = normalize_date(work["available_date"])
    work["_date_str"] = work["_date"].dt.strftime("%Y%m%d")
    work["_available_date_str"] = work["_available_date"].dt.strftime("%Y%m%d")
    work["_value"] = pd.to_numeric(work["total_return_index_or_adjusted_close"], errors="coerce")
    work = work.sort_values("_date").reset_index(drop=True)
    return work


def build_forward_labels(signal_panel: pd.DataFrame, source: pd.DataFrame, config: MarketLabelImportConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_panel.empty or source.empty:
        return empty_labels(), pd.DataFrame()
    market_signals = signal_panel.loc[signal_panel["asset"].astype(str) == "MARKET"].copy()
    unique_signal_dates = sorted(market_signals["signal_date"].astype(str).unique())
    prepared = prepare_source(source)
    pos_by_date = {date: idx for idx, date in enumerate(prepared["_date_str"].tolist())}
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    lineage_asset = str(prepared["asset_or_index"].iloc[0])
    lineage_source = str(prepared["data_source"].iloc[0])
    lineage_vintage = str(prepared["source_vintage"].iloc[0])
    lineage = f"{lineage_source}|{lineage_vintage}|{lineage_asset}"
    for horizon in config.horizons:
        matched = 0
        enough_future = 0
        label_available_ok = 0
        for signal_date in unique_signal_dates:
            pos = pos_by_date.get(signal_date)
            if pos is None:
                continue
            matched += 1
            future_pos = pos + int(horizon)
            if future_pos >= len(prepared):
                continue
            enough_future += 1
            current_value = float(prepared.loc[pos, "_value"])
            future_value = float(prepared.loc[future_pos, "_value"])
            label_available_date = str(prepared.loc[future_pos, "_available_date_str"])
            if label_available_date <= signal_date:
                continue
            label_available_ok += 1
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": "MARKET",
                    "horizon": int(horizon),
                    "forward_adjusted_return": future_value / current_value - 1.0,
                    "return_basis": "total_return",
                    "label_available_date": label_available_date,
                    "price_adjustment_source": lineage,
                }
            )
        coverage_rows.append(
            {
                "horizon": int(horizon),
                "unique_signal_dates": len(unique_signal_dates),
                "matched_source_dates": matched,
                "enough_future_dates": enough_future,
                "label_available_date_ok": label_available_ok,
                "coverage_ratio": label_available_ok / len(unique_signal_dates) if unique_signal_dates else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS), pd.DataFrame(coverage_rows)


def build_import_readiness(
    source_exists: bool,
    source_error: str,
    validation_checks: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: pd.DataFrame,
    config: MarketLabelImportConfig,
) -> pd.DataFrame:
    validation_ok = source_validation_passed(validation_checks)
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    rows = [
        {
            "check": "source_file_exists",
            "status": "pass" if source_exists else "blocked",
            "detail": "" if source_exists else source_error,
        },
        {
            "check": "source_file_readable",
            "status": "pass" if source_exists and not source_error else "blocked",
            "detail": source_error,
        },
        {
            "check": "source_validation_passed",
            "status": "pass" if validation_ok else "blocked",
            "detail": ";".join(validation_checks.loc[validation_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "label_rows_produced",
            "status": "pass" if len(labels) > 0 else "blocked",
            "detail": f"label_rows={len(labels)}",
        },
        {
            "check": "minimum_signal_date_coverage",
            "status": "pass" if min_coverage >= config.min_signal_coverage_ratio else "blocked",
            "detail": f"min_coverage={min_coverage:.4f};required={config.min_signal_coverage_ratio:.4f}",
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "pass" if validation_ok and len(labels) > 0 and min_coverage >= config.min_signal_coverage_ratio else "blocked",
            "detail": "requires valid source, generated labels, and coverage threshold",
        },
    ]
    return pd.DataFrame(rows)


def build_no_label_guard(labels: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    produced = len(labels) > 0
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return pd.DataFrame(
        [
            {
                "result_type": "market_forward_total_return_labels",
                "produced": produced,
                "blocked": not produced,
                "reason": "labels_ready" if produced else "source missing or validation blocked",
            },
            {
                "result_type": "state_stratified_signal_validation",
                "produced": False,
                "blocked": perf_status != "pass",
                "reason": f"performance_validation_allowed_now={perf_status}; V3.49 must run validation separately",
            },
            {
                "result_type": "portfolio_backtest_or_model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "V3.53 imports labels only; it never promotes a model.",
            },
        ]
    )


def build_acceptance_checks(
    source_exists: bool,
    validation_checks: pd.DataFrame,
    labels: pd.DataFrame,
    readiness: pd.DataFrame,
    guard: pd.DataFrame,
) -> pd.DataFrame:
    validation_ok = source_validation_passed(validation_checks)
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    label_sources = set(labels["price_adjustment_source"].astype(str).str.lower()) if not labels.empty else set()
    forbidden_source = any(any(term in item for term in FORBIDDEN_LINEAGE_TERMS) for item in label_sources)
    return pd.DataFrame(
        [
            {
                "check": "missing_source_blocks_labels",
                "status": "pass" if source_exists or labels.empty else "fail",
                "detail": f"source_exists={source_exists};label_rows={len(labels)}",
            },
            {
                "check": "invalid_source_blocks_performance_validation",
                "status": "pass" if validation_ok or perf_status == "blocked" else "fail",
                "detail": f"validation_ok={validation_ok};performance_status={perf_status}",
            },
            {
                "check": "forbidden_lineage_not_used_in_labels",
                "status": "pass" if not forbidden_source else "fail",
                "detail": "label price_adjustment_source checked",
            },
            {
                "check": "state_validation_not_run_here",
                "status": "pass" if not bool(guard.loc[guard["result_type"] == "state_stratified_signal_validation", "produced"].iloc[0]) else "fail",
                "detail": "V3.53 is importer only",
            },
            {
                "check": "portfolio_backtest_not_run_here",
                "status": "pass" if not bool(guard.loc[guard["result_type"] == "portfolio_backtest_or_model_promotion", "produced"].iloc[0]) else "fail",
                "detail": "No NAV, drawdown, Sharpe, or model promotion generated.",
            },
        ]
    )


def build_report(
    config: MarketLabelImportConfig,
    source_exists: bool,
    selected_asset: str,
    selected_reason: str,
    validation_checks: pd.DataFrame,
    coverage: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
    labels: pd.DataFrame,
) -> str:
    lines = [
        "# V3.53 Market Total-Return Label Importer",
        "",
        "## Decision",
        "",
        "- V3.53 implements the strict importer for `data_raw/market_labels/market_total_return_index.csv`.",
        "- If the source file is absent or fails validation, no labels or performance validation are produced.",
        "- If a valid source exists, this importer can generate MARKET forward total-return labels for V3.49/V3.51.",
        "- It does not run IC, hit rate, NAV, drawdown, Sharpe, portfolio backtest, or model promotion.",
        "",
        "## Source",
        "",
        f"- Source path: `{config.source_path.as_posix()}`",
        f"- Source exists: `{source_exists}`",
        f"- Selected asset: `{selected_asset}`",
        f"- Selection reason: `{selected_reason}`",
        "",
        "## Validation Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for row in validation_checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Label Coverage",
            "",
            "| horizon | unique_signal_dates | matched_source_dates | enough_future_dates | label_available_date_ok | coverage_ratio |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {int(row.unique_signal_dates)} | {int(row.matched_source_dates)} | {int(row.enough_future_dates)} | {int(row.label_available_date_ok)} | {float(row.coverage_ratio):.4f} |"
        )
    if coverage.empty:
        lines.append("|  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- Label rows produced: `{len(labels)}`",
            f"- Label columns: `{','.join(LABEL_COLUMNS)}`",
            "",
            "## Next Use",
            "",
            "- Place a compliant total-return file at the configured source path, then rerun V3.53.",
            "- Once labels are produced, feed `market_forward_labels.csv` into V3.49 state-stratified validation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame, labels: pd.DataFrame) -> str:
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return "\n".join(
        [
            "# A-share Market Total-Return Label Importer V3.53",
            "",
            "## Dataset Decision",
            "",
            "- Importer ready: `true`",
            f"- Labels produced: `{len(labels) > 0}`",
            f"- Label rows: `{len(labels)}`",
            f"- Performance validation status: `{perf_status}`",
            "",
            "## Required Input",
            "",
            "`data_raw/market_labels/market_total_return_index.csv` with columns:",
            "",
            "`date, asset_or_index, total_return_index_or_adjusted_close, available_date, data_source, source_vintage`",
            "",
        ]
    )
