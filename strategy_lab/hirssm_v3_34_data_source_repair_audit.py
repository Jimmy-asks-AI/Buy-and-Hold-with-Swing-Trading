#!/usr/bin/env python
"""HIRSSM V3.34 data-source repair audit.

This is a data-steward task. It audits source eligibility and repair needs
after V3.33 found no implementation-ready independent signals. It does not run
portfolio backtests, tune signals, or promote any model.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_34" / "data_source_repair_audit"
TASK_ID = "20260527_v3_34_data_source_repair_audit"
VERSION = "V3.34"
BASELINE = "V3.33 Independent Signal Source Discovery"
REPORT_PATH = ROOT / "data_catalog" / "a_share_data_source_repair_audit_v3_34.md"
FINANCIAL_REPORT_DATE_COL = "\u62a5\u544a\u671f"


DATASET_SPECS: list[dict[str, Any]] = [
    {
        "dataset_id": "csindex_daily_index_prices",
        "source_family": "index_market_observation",
        "path": "data_raw/index/akshare_csindex/daily_csindex",
        "frequency": "daily",
        "date_col": "date",
        "index_col": "index_code",
        "asset_col": "",
        "weight_col": "",
        "critical_cols": ["date", "index_code", "close"],
        "availability_col": "",
        "current_snapshot": False,
        "adjustment_method": "index_level",
        "baseline_status": "approved",
        "status_reason": "Historical index levels are observable market data and can be used for index-level return studies.",
    },
    {
        "dataset_id": "sw_industry_daily_prices",
        "source_family": "industry_market_observation",
        "path": "data_raw/index/akshare_sw_industry/daily_sw",
        "frequency": "daily",
        "date_col": "date",
        "index_col": "index_code",
        "asset_col": "",
        "weight_col": "",
        "critical_cols": ["date", "index_code", "close"],
        "availability_col": "",
        "current_snapshot": False,
        "adjustment_method": "index_level",
        "baseline_status": "approved",
        "status_reason": "Historical industry index levels are observable market data and can support index-level rotation research.",
    },
    {
        "dataset_id": "csindex_constituents_current",
        "source_family": "current_constituent_snapshot",
        "path": "data_raw/index/akshare_csindex/constituents_current",
        "frequency": "snapshot",
        "date_col": "date",
        "index_col": "index_code",
        "asset_col": "asset",
        "weight_col": "",
        "critical_cols": ["date", "index_code", "asset"],
        "availability_col": "fetched_at",
        "current_snapshot": True,
        "adjustment_method": "not_applicable",
        "baseline_status": "research_only",
        "status_reason": "Current constituents are usable for current exposure inspection but not historical point-in-time membership.",
    },
    {
        "dataset_id": "csindex_weights_latest",
        "source_family": "latest_weight_snapshot",
        "path": "data_raw/index/akshare_csindex/weights_latest",
        "frequency": "snapshot",
        "date_col": "date",
        "index_col": "index_code",
        "asset_col": "asset",
        "weight_col": "weight",
        "critical_cols": ["date", "index_code", "asset", "weight"],
        "availability_col": "fetched_at",
        "current_snapshot": True,
        "adjustment_method": "not_applicable",
        "baseline_status": "research_only",
        "status_reason": "Latest weights are usable for current exposure inspection but not historical point-in-time weight history.",
    },
    {
        "dataset_id": "sw_industry_components_current",
        "source_family": "current_industry_component_snapshot",
        "path": "data_raw/index/akshare_sw_industry/components_current",
        "frequency": "snapshot",
        "date_col": "inclusion_date",
        "index_col": "index_code",
        "asset_col": "asset",
        "weight_col": "weight",
        "critical_cols": ["index_code", "asset", "weight"],
        "availability_col": "fetched_at",
        "current_snapshot": True,
        "adjustment_method": "not_applicable",
        "baseline_status": "research_only",
        "status_reason": "Current industry membership lacks historical removal and effective-vintage records.",
    },
    {
        "dataset_id": "stock_daily_qfq_sample",
        "source_family": "sample_stock_market_data",
        "path": "data_raw/akshare/daily_qfq",
        "frequency": "daily",
        "date_col": "date",
        "index_col": "",
        "asset_col": "asset",
        "weight_col": "",
        "critical_cols": ["date", "asset", "close", "amount"],
        "availability_col": "fetched_at",
        "current_snapshot": False,
        "adjustment_method": "qfq_back_adjusted",
        "baseline_status": "research_only",
        "status_reason": "Only a small sample is present and QFQ embeds later corporate-action adjustment into earlier levels.",
    },
    {
        "dataset_id": "stock_financial_indicator_sample",
        "source_family": "sample_stock_fundamental_data",
        "path": "data_raw/akshare/financial_indicator",
        "frequency": "report_period",
        "date_col": FINANCIAL_REPORT_DATE_COL,
        "index_col": "",
        "asset_col": "asset",
        "weight_col": "",
        "critical_cols": [FINANCIAL_REPORT_DATE_COL, "asset"],
        "availability_col": "",
        "current_snapshot": False,
        "adjustment_method": "not_applicable",
        "baseline_status": "blocked",
        "status_reason": "Report-period values lack announcement or available_date fields required for PIT fundamental factors.",
    },
    {
        "dataset_id": "macro_pit_panel",
        "source_family": "macro_point_in_time",
        "path": "data_raw/macro/macro_pit_panel.csv",
        "frequency": "mixed",
        "date_col": "date",
        "index_col": "",
        "asset_col": "",
        "weight_col": "",
        "critical_cols": ["date", "available_date"],
        "availability_col": "available_date",
        "current_snapshot": False,
        "adjustment_method": "not_applicable",
        "baseline_status": "approved",
        "status_reason": "Macro panel has available_date and can be used in as-of signal validation with release-lag joins.",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", dtype=str, nrows=nrows, low_memory=False)


def csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return []


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_code(series: pd.Series) -> pd.Series:
    code = series.astype(str).str.extract(r"(\d{1,6})", expand=False)
    return code.str.zfill(6)


def missing_any_rate(data: pd.DataFrame, cols: list[str]) -> float:
    present = [col for col in cols if col in data.columns]
    if not present:
        return 1.0
    return float(data[present].isna().any(axis=1).mean()) if len(data) else 0.0


def unique_count(frames: list[pd.DataFrame], col: str) -> int:
    if not col:
        return 0
    values: list[pd.Series] = []
    for frame in frames:
        if col in frame.columns:
            values.append(normalize_code(frame[col]) if col in {"asset", "index_code"} else frame[col].astype(str))
    if not values:
        return 0
    return int(pd.concat(values, ignore_index=True).dropna().nunique())


def dataset_status(spec: dict[str, Any], quality: dict[str, Any]) -> tuple[str, bool, str, str]:
    if quality["file_count"] == 0:
        return "blocked", False, "missing_source_files", "No local source files are available."
    if spec["current_snapshot"]:
        return spec["baseline_status"], False, "current_snapshot_only", spec["status_reason"]
    if spec["dataset_id"] == "stock_financial_indicator_sample":
        return "blocked", False, "missing_announcement_available_date", spec["status_reason"]
    if spec["dataset_id"] == "stock_daily_qfq_sample":
        return "research_only", False, "sample_and_qfq_adjusted", spec["status_reason"]
    if spec["availability_col"] and not quality["availability_col_present"]:
        return "blocked", False, "missing_available_date_column", "Required availability column is missing."
    if quality["critical_missing_rate"] > 0.05:
        return "research_only", False, "critical_missing_rate_high", "Critical field missingness is above the data-steward threshold."
    return spec["baseline_status"], spec["baseline_status"] == "approved", "passes_declared_dataset_gate", spec["status_reason"]


def audit_dataset(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base_path = ROOT / spec["path"]
    files = csv_files(base_path)
    frames: list[pd.DataFrame] = []
    date_values: list[pd.Series] = []
    available_values: list[pd.Series] = []
    row_count = 0
    duplicate_count = 0
    nonpositive_close_count = 0
    negative_weight_count = 0
    weight_sum_min = np.nan
    weight_sum_max = np.nan
    weight_sums: list[float] = []
    critical_missing_weighted = 0.0
    sample_columns: list[str] = []

    for path in files:
        data = read_csv(path)
        frames.append(data)
        row_count += int(data.shape[0])
        if not sample_columns and not data.empty:
            sample_columns = list(data.columns)
        critical_missing_weighted += missing_any_rate(data, spec["critical_cols"]) * int(data.shape[0])
        key_cols = [col for col in [spec["date_col"], spec["index_col"], spec["asset_col"]] if col and col in data.columns]
        if len(key_cols) >= 2:
            duplicate_count += int(data.duplicated(key_cols).sum())
        if spec["date_col"] in data.columns:
            parsed = parse_dates(data[spec["date_col"]])
            date_values.append(parsed)
        if spec["availability_col"] and spec["availability_col"] in data.columns:
            available_values.append(parse_dates(data[spec["availability_col"]]))
        if "close" in data.columns:
            close = to_number(data["close"])
            nonpositive_close_count += int((close <= 0).sum())
        if spec["weight_col"] and spec["weight_col"] in data.columns:
            weight = to_number(data[spec["weight_col"]])
            negative_weight_count += int((weight < 0).sum())
            group_cols = [col for col in [spec["date_col"], spec["index_col"]] if col and col in data.columns]
            if group_cols:
                for value in data.assign(_weight=weight).groupby(group_cols, dropna=False)["_weight"].sum(min_count=1).dropna().tolist():
                    weight_sums.append(float(value))

    all_dates = pd.concat(date_values, ignore_index=True).dropna() if date_values else pd.Series(dtype="datetime64[ns]")
    all_available = pd.concat(available_values, ignore_index=True).dropna() if available_values else pd.Series(dtype="datetime64[ns]")
    critical_missing_rate = float(critical_missing_weighted / row_count) if row_count else 1.0
    if weight_sums:
        weight_sum_min = float(np.nanmin(weight_sums))
        weight_sum_max = float(np.nanmax(weight_sums))
    availability_col_present = bool(spec["availability_col"] and any(spec["availability_col"] in frame.columns for frame in frames))
    invalid_available_date_rate = 0.0
    if spec["availability_col"] and availability_col_present:
        total_avail_rows = 0
        invalid_avail_rows = 0
        for frame in frames:
            if spec["availability_col"] in frame.columns:
                avail = parse_dates(frame[spec["availability_col"]])
                total_avail_rows += int(frame.shape[0])
                invalid_avail_rows += int(avail.isna().sum())
        invalid_available_date_rate = float(invalid_avail_rows / total_avail_rows) if total_avail_rows else 1.0

    quality_base = {
        "dataset_id": spec["dataset_id"],
        "source_family": spec["source_family"],
        "path": spec["path"],
        "file_count": int(len(files)),
        "row_count": int(row_count),
        "start_date": str(all_dates.min().date()) if not all_dates.empty else "",
        "end_date": str(all_dates.max().date()) if not all_dates.empty else "",
        "available_date_start": str(all_available.min().date()) if not all_available.empty else "",
        "available_date_end": str(all_available.max().date()) if not all_available.empty else "",
        "asset_count": unique_count(frames, spec["asset_col"]),
        "index_count": unique_count(frames, spec["index_col"]),
        "critical_missing_rate": critical_missing_rate,
        "duplicate_key_count": int(duplicate_count),
        "nonpositive_close_count": int(nonpositive_close_count),
        "negative_weight_count": int(negative_weight_count),
        "weight_sum_min": weight_sum_min,
        "weight_sum_max": weight_sum_max,
        "availability_col_present": bool(availability_col_present),
        "invalid_available_date_rate": invalid_available_date_rate,
        "current_snapshot": bool(spec["current_snapshot"]),
        "adjustment_method": spec["adjustment_method"],
    }
    status, strict_pit_allowed, pit_gate, reason = dataset_status(spec, quality_base)
    quality = {
        **quality_base,
        "dataset_status": status,
        "strict_pit_backtest_allowed": bool(strict_pit_allowed),
        "pit_gate": pit_gate,
        "status_reason": reason,
    }

    dictionary_rows = []
    for column in sample_columns:
        dictionary_rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "column": column,
                "role": infer_column_role(column, spec),
                "frequency": spec["frequency"],
                "source_path": spec["path"],
                "availability_rule": availability_rule(spec),
                "downstream_status": status,
            }
        )

    pit_row = {
        "dataset_id": spec["dataset_id"],
        "source_family": spec["source_family"],
        "path": spec["path"],
        "status": status,
        "strict_pit_backtest_allowed": bool(strict_pit_allowed),
        "current_snapshot": bool(spec["current_snapshot"]),
        "available_date_col": spec["availability_col"],
        "available_date_col_present": bool(availability_col_present),
        "invalid_available_date_rate": invalid_available_date_rate,
        "adjustment_method": spec["adjustment_method"],
        "allowed_downstream": downstream_rule(spec["dataset_id"], status, strict_pit_allowed),
        "blocked_use": blocked_use(spec["dataset_id"], status, strict_pit_allowed),
        "pit_gate": pit_gate,
        "reason": reason,
    }
    return quality, dictionary_rows, pit_row


def infer_column_role(column: str, spec: dict[str, Any]) -> str:
    if column == spec["date_col"]:
        return "observation_or_period_date"
    if column == spec["availability_col"]:
        return "available_date_or_fetch_time"
    if column == spec["asset_col"]:
        return "security_identifier"
    if column == spec["index_col"]:
        return "index_identifier"
    if column == spec["weight_col"]:
        return "constituent_weight"
    if column in {"open", "high", "low", "close", "volume", "amount"}:
        return "market_bar"
    if column in {"fetched_at", "data_source"}:
        return "source_provenance"
    return "source_field"


def availability_rule(spec: dict[str, Any]) -> str:
    if spec["current_snapshot"]:
        return "current snapshot only; fetched_at is not a historical effective-date series"
    if spec["availability_col"]:
        return f"use only when asof_date >= {spec['availability_col']}"
    if spec["dataset_id"] == "stock_financial_indicator_sample":
        return "blocked until announcement available_date is added"
    if spec["dataset_id"] == "stock_daily_qfq_sample":
        return "sample-only; QFQ is not a strict raw PIT price level"
    return "observable market series after close"


def downstream_rule(dataset_id: str, status: str, strict_pit_allowed: bool) -> str:
    if strict_pit_allowed:
        return "signal_validation,index_level_backtest"
    if dataset_id.endswith("_current") or "weights_latest" in dataset_id:
        return "current_exposure_snapshot,research_notes"
    if dataset_id == "stock_daily_qfq_sample":
        return "smoke_test,field_mapping,prototype_only"
    if dataset_id == "stock_financial_indicator_sample":
        return "schema_learning_only"
    if status == "research_only":
        return "research_only"
    return "blocked"


def blocked_use(dataset_id: str, status: str, strict_pit_allowed: bool) -> str:
    if strict_pit_allowed:
        return ""
    if dataset_id.endswith("_current") or "weights_latest" in dataset_id:
        return "historical_constituent_or_weight_backtest"
    if dataset_id == "stock_daily_qfq_sample":
        return "production_broad_stock_factor_backtest,raw_trade_execution_simulation"
    if dataset_id == "stock_financial_indicator_sample":
        return "historical_fundamental_factor_backtest_without_announcement_date"
    return "strict_historical_backtest"


def build_repair_plan(pit: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "repair_id": "historical_index_constituents_weights",
            "blocked_datasets": "csindex_constituents_current|csindex_weights_latest",
            "target_sources": "Tushare Pro index_weight; JoinQuant date-specific index constituents; Wind/CSIndex historical files",
            "required_fields": "index_code,asset,weight,effective_date,available_date,source_vintage,data_source",
            "validation_gates": "no current snapshot backfill; date+index+asset unique; weights sum near 100 by effective_date; available_date <= model_asof_date",
            "handoff_agent": "data_steward",
            "downstream_unlock": "index-enhancement factors, constituent aggregation, historical benchmark exposure",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "blocked_datasets": "stock_daily_qfq_sample",
            "target_sources": "Tushare Pro daily/daily_basic/adj_factor; JoinQuant price API with paused/ST flags; high-quality vendor lake",
            "required_fields": "date,asset,open,high,low,close,volume,amount,adj_factor,raw_close,adjusted_close,is_paused,is_st,listed_days,delist_date",
            "validation_gates": "include delisted stocks; raw and adjusted prices separated; no forward-filled halted bars without flag; asset-date unique",
            "handoff_agent": "data_steward",
            "downstream_unlock": "broad stock factor discovery, liquidity/capacity factors, index constituent aggregation",
        },
        {
            "priority": 3,
            "repair_id": "financial_indicator_point_in_time",
            "blocked_datasets": "stock_financial_indicator_sample",
            "target_sources": "Tushare Pro disclosure/financial endpoints; JoinQuant finance tables with publication dates; exchange filings",
            "required_fields": "asset,report_period,statement_type,metric,value,announcement_date,available_date,data_source,revision_flag",
            "validation_gates": "available_date mandatory; no restated current values backfilled into old asof dates; duplicate revision policy explicit",
            "handoff_agent": "data_steward",
            "downstream_unlock": "quality,profitability,growth,dividend and value factors",
        },
        {
            "priority": 4,
            "repair_id": "historical_industry_classification",
            "blocked_datasets": "sw_industry_components_current",
            "target_sources": "JoinQuant industry by date; Wind/SW historical classification; internally versioned classification snapshots",
            "required_fields": "asset,industry_code,industry_name,level,effective_date,end_date,available_date,classification_standard",
            "validation_gates": "one active classification per asset-level-asof; no latest classification backfill; delisted securities retained",
            "handoff_agent": "data_steward",
            "downstream_unlock": "industry-neutral factors, industry rotation, style/industry attribution",
        },
        {
            "priority": 5,
            "repair_id": "macro_release_calendar_vintage_upgrade",
            "blocked_datasets": "",
            "target_sources": "official macro calendars; data vendor vintage tables",
            "required_fields": "series_id,period_date,value,release_date,available_date,revision_timestamp,vintage_id",
            "validation_gates": "release lag explicit; revised values separated from initial vintage when available",
            "handoff_agent": "data_steward",
            "downstream_unlock": "macro regimes with stronger leakage control",
        },
    ]
    blocked = set(pit.loc[~pit["strict_pit_backtest_allowed"], "dataset_id"].astype(str))
    for row in rows:
        members = {item for item in row["blocked_datasets"].split("|") if item}
        row["currently_needed"] = bool(not members or members & blocked)
    return pd.DataFrame(rows)


def build_report(quality: pd.DataFrame, pit: pd.DataFrame, repair: pd.DataFrame) -> str:
    approved = int((pit["strict_pit_backtest_allowed"] == True).sum())
    research_only = int((pit["status"] == "research_only").sum())
    blocked = int((pit["status"] == "blocked").sum())
    current_snapshot_blocked = pit.loc[pit["current_snapshot"] == True, "dataset_id"].tolist()
    repair_needed = repair.loc[repair["currently_needed"] == True, "repair_id"].tolist()
    return "\n".join(
        [
            "# V3.34 Data Source Repair Audit",
            "",
            f"Generated at: `{now_text()}`",
            "",
            "## Decision",
            "",
            "- Task accepted as data governance only.",
            "- No portfolio harness was run.",
            "- No signal or model was promoted.",
            "- Current constituents and latest weights remain blocked for historical point-in-time backtests.",
            "",
            "## Dataset Status",
            "",
            f"- Audited datasets: `{len(quality)}`",
            f"- Strict PIT backtest approved: `{approved}`",
            f"- Research-only: `{research_only}`",
            f"- Blocked: `{blocked}`",
            f"- Current snapshot datasets restricted: `{', '.join(current_snapshot_blocked)}`",
            "",
            "## Repair Queue",
            "",
            *[f"- `{item}`" for item in repair_needed],
            "",
            "## Downstream Rule",
            "",
            "Factor researchers may use approved market or macro PIT datasets for signal validation. They must not use current constituents, latest weights, sample QFQ data, or financial indicators for strict historical stock-level backtests until the repair queue is completed.",
        ]
    )


def build_catalog_update(quality: pd.DataFrame, pit: pd.DataFrame, repair: pd.DataFrame) -> str:
    lines = [
        "# A-share Data Source Repair Audit V3.34",
        "",
        f"Updated: `{now_text()}`",
        "",
        "This catalog note records data-steward decisions after V3.33 found no implementation-ready independent signal.",
        "",
        "## Point-in-time Status",
        "",
        "| Dataset | Status | Strict PIT Backtest | Restriction |",
        "|---|---:|---:|---|",
    ]
    for row in pit.itertuples(index=False):
        lines.append(
            f"| `{row.dataset_id}` | `{row.status}` | `{bool(row.strict_pit_backtest_allowed)}` | {row.blocked_use or row.allowed_downstream} |"
        )
    lines.extend(
        [
            "",
            "## Required Repairs",
            "",
            "| Priority | Repair | Required Fields | Validation Gates |",
            "|---:|---|---|---|",
        ]
    )
    for row in repair.itertuples(index=False):
        lines.append(f"| {row.priority} | `{row.repair_id}` | {row.required_fields} | {row.validation_gates} |")
    return "\n".join(lines) + "\n"


def build_self_check(paths: dict[str, Path], pit: pd.DataFrame, repair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    current_snapshot_bad = pit.loc[(pit["current_snapshot"] == True) & (pit["strict_pit_backtest_allowed"] == True), "dataset_id"].tolist()
    rows.append(
        {
            "check": "no_current_snapshot_approved_for_historical_backtest",
            "status": "pass" if not current_snapshot_bad else "fail",
            "detail": ",".join(current_snapshot_bad),
        }
    )
    financial_status = pit.loc[pit["dataset_id"] == "stock_financial_indicator_sample", "status"].iloc[0]
    rows.append(
        {
            "check": "financial_indicator_requires_available_date",
            "status": "pass" if financial_status in {"blocked", "research_only"} else "fail",
            "detail": financial_status,
        }
    )
    qfq_status = pit.loc[pit["dataset_id"] == "stock_daily_qfq_sample", "status"].iloc[0]
    rows.append(
        {
            "check": "qfq_sample_not_production_approved",
            "status": "pass" if qfq_status == "research_only" else "fail",
            "detail": qfq_status,
        }
    )
    macro = pit.loc[pit["dataset_id"] == "macro_pit_panel"].iloc[0]
    rows.append(
        {
            "check": "macro_pit_has_available_date",
            "status": "pass" if bool(macro.available_date_col_present) else "fail",
            "detail": str(macro.available_date_col),
        }
    )
    rows.append(
        {
            "check": "repair_queue_present",
            "status": "pass" if int(repair["currently_needed"].sum()) >= 4 else "fail",
            "detail": str(int(repair["currently_needed"].sum())),
        }
    )
    rows.append(
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "data governance only",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quality_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    pit_rows: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        quality, dictionary, pit = audit_dataset(spec)
        quality_rows.append(quality)
        dictionary_rows.extend(dictionary)
        pit_rows.append(pit)

    quality_df = pd.DataFrame(quality_rows)
    dictionary_df = pd.DataFrame(dictionary_rows)
    pit_df = pd.DataFrame(pit_rows)
    repair_df = build_repair_plan(pit_df)

    artifacts = {
        "data_quality_report": OUTPUT_DIR / "data_quality_report.csv",
        "data_dictionary": OUTPUT_DIR / "data_dictionary.csv",
        "point_in_time_check": OUTPUT_DIR / "point_in_time_check.csv",
        "data_source_repair_plan": OUTPUT_DIR / "data_source_repair_plan.csv",
        "agent_report": OUTPUT_DIR / "agent_report.md",
        "catalog_update": REPORT_PATH,
        "changed_files": OUTPUT_DIR / "changed_files.txt",
    }
    write_csv(quality_df, artifacts["data_quality_report"])
    write_csv(dictionary_df, artifacts["data_dictionary"])
    write_csv(pit_df, artifacts["point_in_time_check"])
    write_csv(repair_df, artifacts["data_source_repair_plan"])
    write_text(build_report(quality_df, pit_df, repair_df), artifacts["agent_report"])
    write_text(build_catalog_update(quality_df, pit_df, repair_df), artifacts["catalog_update"])

    changed_files = [rel(path) for path in artifacts.values()]
    write_text("\n".join(changed_files) + "\n", artifacts["changed_files"])

    self_check_path = OUTPUT_DIR / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check_df = build_self_check(artifacts, pit_df, repair_df)
    write_csv(self_check_df, self_check_path)

    fail_count = int((self_check_df["status"] == "fail").sum())
    warn_count = int((pit_df["status"] == "research_only").sum())
    metrics = {
        "audited_dataset_count": int(len(quality_df)),
        "strict_pit_approved_count": int(pit_df["strict_pit_backtest_allowed"].sum()),
        "research_only_count": int((pit_df["status"] == "research_only").sum()),
        "blocked_count": int((pit_df["status"] == "blocked").sum()),
        "current_snapshot_restricted_count": int((pit_df["current_snapshot"] == True).sum()),
        "repair_queue_count": int(repair_df["currently_needed"].sum()),
        "model_decision": "no_model_promotion_data_governance_only",
    }
    manifest_path = OUTPUT_DIR / "agent_run_manifest.json"
    manifest = {
        "run_id": "20260527_v3_34_data_source_repair_audit_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": VERSION,
        "baseline": BASELINE,
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_34_data_source_repair_audit.py",
        "config": {
            "data_governance_only": True,
            "no_portfolio_harness": True,
            "no_model_promotion": True,
        },
        "data_refs": [
            "outputs/agent_runs/v3_33/independent_signal_source_discovery/next_research_queue.csv",
            "data_raw/index/akshare_csindex",
            "data_raw/index/akshare_sw_industry",
            "data_raw/akshare",
            "data_raw/macro/macro_pit_panel.csv",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_34_data_source_repair_audit.py",
            "strategy_lab/agents/data_steward/AGENT.md",
        ],
        "output_dir": rel(OUTPUT_DIR),
        "allowed_inputs": [
            "outputs/agent_runs/v3_33/independent_signal_source_discovery/next_research_queue.csv",
            "outputs/agent_runs/v3_33/independent_signal_source_discovery/data_source_inventory.csv",
            "data_raw/index/akshare_csindex/",
            "data_raw/index/akshare_sw_industry/",
            "data_raw/akshare/daily_qfq/",
            "data_raw/akshare/financial_indicator/",
            "data_raw/macro/macro_pit_panel.csv",
            "data_catalog/",
        ],
        "artifacts": [*changed_files, rel(self_check_path)],
        "outputs": [*changed_files, rel(self_check_path)],
        "changed_files": [*changed_files, rel(self_check_path), rel(manifest_path)],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "No external data was downloaded in this task.",
            "Current constituents and latest weights remain current-only snapshots.",
            "Stock-level data remains a five-symbol sample and is not a broad factor universe.",
        ],
        "risk_flags": [
            "current_snapshot_bias",
            "qfq_adjustment_lookahead_for_price_levels",
            "missing_fundamental_available_date",
            "limited_stock_universe_breadth",
        ],
        "next_decision": "Open a data acquisition task for historical constituents/weights and broad PIT stock data before another model harness.",
        "handoff_summary": "V3.34 converts V3.33 data blockers into explicit PIT status, repair contracts, and downstream restrictions.",
    }
    write_json(manifest, manifest_path)

    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "self_check_pass": fail_count == 0,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
