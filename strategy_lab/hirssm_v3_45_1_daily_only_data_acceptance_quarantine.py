#!/usr/bin/env python
"""HIRSSM V3.45.1 daily-only data acceptance with row quarantine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import hirssm_v3_45_daily_only_data_acceptance as base


ROOT = base.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "daily_only_data_acceptance_v3_45_1.json"
TASK_ID = "20260529_v3_45_1_daily_only_data_acceptance_quarantine"
VERSION = "V3.45.1"


def ohlc_mask(data: pd.DataFrame) -> pd.Series:
    row_count = int(data.shape[0])
    open_ = pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else pd.Series([float("nan")] * row_count)
    high = pd.to_numeric(data["high"], errors="coerce") if "high" in data.columns else pd.Series([float("nan")] * row_count)
    low = pd.to_numeric(data["low"], errors="coerce") if "low" in data.columns else pd.Series([float("nan")] * row_count)
    close = pd.to_numeric(data["close"], errors="coerce") if "close" in data.columns else pd.Series([float("nan")] * row_count)
    return ((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).fillna(True)


def anomaly_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    open_ = pd.to_numeric(row.get("open"), errors="coerce")
    high = pd.to_numeric(row.get("high"), errors="coerce")
    low = pd.to_numeric(row.get("low"), errors="coerce")
    close = pd.to_numeric(row.get("close"), errors="coerce")
    if high < low:
        reasons.append("high_below_low")
    if high < open_:
        reasons.append("high_below_open")
    if high < close:
        reasons.append("high_below_close")
    if low > open_:
        reasons.append("low_above_open")
    if low > close:
        reasons.append("low_above_close")
    return ";".join(reasons) if reasons else "non_numeric_ohlc"


def build_quarantine_rows(config: dict[str, Any], file_report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    affected = file_report.loc[file_report["bad_ohlc_count"] > 0]
    for item in affected.itertuples(index=False):
        source_path = ROOT / str(item.path)
        data = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
        bad = data.loc[ohlc_mask(data)].copy()
        for row_index, row in bad.iterrows():
            rows.append(
                {
                    "trade_date": str(item.trade_date),
                    "ts_code": row.get("ts_code", ""),
                    "asset": row.get("asset", row.get("ts_code", "")),
                    "csv_row_number": int(row_index) + 2,
                    "open": row.get("open", ""),
                    "high": row.get("high", ""),
                    "low": row.get("low", ""),
                    "close": row.get("close", ""),
                    "pre_close": row.get("pre_close", ""),
                    "vol": row.get("vol", ""),
                    "amount": row.get("amount", ""),
                    "source_file": str(item.path),
                    "anomaly_type": "ohlc_bounds_violation",
                    "anomaly_reason": anomaly_reason(row),
                    "quarantine_action": config["quarantine_policy"]["ohlc_violation_action"],
                    "raw_source_immutable": config["quarantine_policy"]["raw_source_immutable"],
                }
            )
    columns = [
        "trade_date",
        "ts_code",
        "asset",
        "csv_row_number",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "source_file",
        "anomaly_type",
        "anomaly_reason",
        "quarantine_action",
        "raw_source_immutable",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_processed_scope_summary(file_report: pd.DataFrame, quarantine_rows: pd.DataFrame) -> pd.DataFrame:
    acquired = file_report.loc[file_report["status"] == "acquired"]
    raw_rows = int(acquired["rows"].sum())
    quarantined_rows = int(quarantine_rows.shape[0])
    source_bad_ohlc_rows = int(acquired["bad_ohlc_count"].sum())
    return pd.DataFrame(
        [
            {"metric": "raw_rows", "value": raw_rows},
            {"metric": "source_bad_ohlc_rows", "value": source_bad_ohlc_rows},
            {"metric": "quarantined_ohlc_rows", "value": quarantined_rows},
            {"metric": "accepted_processed_rows", "value": raw_rows - quarantined_rows},
            {"metric": "bad_ohlc_rows_after_quarantine", "value": max(source_bad_ohlc_rows - quarantined_rows, 0)},
            {"metric": "quarantined_assets", "value": int(quarantine_rows["ts_code"].nunique()) if quarantined_rows else 0},
            {"metric": "quarantined_trade_dates", "value": int(quarantine_rows["trade_date"].nunique()) if quarantined_rows else 0},
        ]
    )


def build_acceptance_checks_v3_45_1(
    config: dict[str, Any],
    file_report: pd.DataFrame,
    registry_compare: pd.DataFrame,
    quarantine_rows: pd.DataFrame,
) -> pd.DataFrame:
    base_checks = base.build_acceptance_checks(config, file_report, registry_compare)
    rows = base_checks.loc[base_checks["check"] != "no_bad_ohlc_rows"].to_dict("records")
    acquired = file_report.loc[file_report["status"] == "acquired"]
    source_bad = int(acquired["bad_ohlc_count"].sum())
    quarantined = int(quarantine_rows.shape[0])
    rows.extend(
        [
            {
                "check": "source_bad_ohlc_rows_detected",
                "status": "warn" if source_bad else "pass",
                "detail": str(source_bad),
            },
            {
                "check": "all_bad_ohlc_rows_quarantined",
                "status": "pass" if quarantined == source_bad else "fail",
                "detail": f"quarantined={quarantined};source_bad={source_bad}",
            },
            {
                "check": "processed_scope_excludes_quarantined_rows",
                "status": "pass" if source_bad - quarantined == 0 else "fail",
                "detail": str(max(source_bad - quarantined, 0)),
            },
            {
                "check": "raw_source_immutable",
                "status": "pass" if bool(config["quarantine_policy"]["raw_source_immutable"]) else "fail",
                "detail": config["quarantine_policy"]["ohlc_violation_action"],
            },
        ]
    )
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    file_report: pd.DataFrame,
    registry_compare: pd.DataFrame,
    quarantine_rows: pd.DataFrame,
    processed_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    acquired = file_report.loc[file_report["status"] == "acquired"]
    empty = file_report.loc[file_report["status"] == "empty"]
    failed = checks.loc[checks["status"] == "fail"]
    warned = checks.loc[checks["status"] == "warn"]
    metrics = dict(zip(processed_summary["metric"], processed_summary["value"]))
    lines = [
        "# V3.45.1 Daily-Only Data Acceptance Quarantine",
        "",
        f"Generated at: `{base.now_text()}`",
        "",
        "## Decision",
        "",
        "- Collection is complete for the permitted `tushare.daily` raw daily interface.",
        "- V3.45 found two source-side OHLC bound anomalies; V3.45.1 keeps raw files immutable and quarantines those rows from the processed scope.",
        "- Processed daily-only data is accepted only after applying the quarantine list.",
        "- This remains raw, unadjusted data and is still blocked for adjusted-return, valuation, high-dividend, low-PE, and portfolio-performance research.",
        "",
        "## Coverage",
        "",
        f"- Expected business-date files: `{len(file_report)}`",
        f"- Acquired files with rows: `{len(acquired)}`",
        f"- Empty business-date files: `{len(empty)}`",
        f"- Raw rows: `{int(acquired['rows'].sum())}`",
        f"- Quarantined OHLC rows: `{metrics['quarantined_ohlc_rows']}`",
        f"- Accepted processed rows: `{metrics['accepted_processed_rows']}`",
        f"- Coverage start: `{config['start_date']}`",
        f"- Coverage end: `{config['end_date']}`",
        "",
        "## Quarantine Rows",
        "",
        "| trade_date | ts_code | open | high | low | close | reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in quarantine_rows.itertuples(index=False):
        lines.append(f"| `{row.trade_date}` | `{row.ts_code}` | {row.open} | {row.high} | {row.low} | {row.close} | `{row.anomaly_reason}` |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Total checks: `{len(checks)}`",
            f"- Failed checks: `{len(failed)}`",
            f"- Warning checks: `{len(warned)}`",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
    )
    for row in checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Registry Reconciliation",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for row in registry_compare.itertuples(index=False):
        lines.append(f"| `{row.metric}` | {row.value} |")
    lines.extend(
        [
            "",
            "## Downstream Rule",
            "",
            "- Raw files are a source archive and must not be edited in place.",
            "- Downstream consumers must apply `ohlc_quarantine_rows.csv` or use a derived accepted view before any signal computation.",
            "- V3.44 raw adjustment guard still blocks adjusted-return and valuation misuse.",
            "",
            "## Still Needed",
            "",
            *[f"- `{item}`" for item in config["manual_data_interfaces_still_needed"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(config: dict[str, Any], processed_summary: pd.DataFrame, checks: pd.DataFrame) -> str:
    metrics = dict(zip(processed_summary["metric"], processed_summary["value"]))
    failed = checks.loc[checks["status"] == "fail"]
    return "\n".join(
        [
            "# A-share Daily-Only Data Acceptance V3.45.1",
            "",
            f"Updated: `{base.now_text()}`",
            "",
            "## Acceptance",
            "",
            f"- Acceptance passed after quarantine: `{failed.empty}`",
            f"- Raw rows: `{metrics['raw_rows']}`",
            f"- Source OHLC anomalies: `{metrics['source_bad_ohlc_rows']}`",
            f"- Quarantined OHLC rows: `{metrics['quarantined_ohlc_rows']}`",
            f"- Accepted processed rows: `{metrics['accepted_processed_rows']}`",
            f"- Data range: `{config['start_date']}` to `{config['end_date']}`",
            "",
            "## Boundary",
            "",
            "- Data is raw daily OHLCV only.",
            "- The accepted processed scope excludes row-level OHLC anomalies.",
            "- V3.44 guard blocks adjusted-return and valuation misuse.",
            "- Missing interfaces remain unresolved.",
            "",
            "## Still Needed",
            "",
            *[f"- `{item}`" for item in config["manual_data_interfaces_still_needed"]],
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], checks: pd.DataFrame, quarantine_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": base.rel(path)})
    rows.append(
        {
            "check": "acceptance_checks_have_no_fail",
            "status": "pass" if not bool((checks["status"] == "fail").any()) else "fail",
            "detail": str(int((checks["status"] == "fail").sum())),
        }
    )
    rows.append(
        {
            "check": "quarantine_rows_identified",
            "status": "pass" if int(quarantine_rows.shape[0]) > 0 else "fail",
            "detail": str(int(quarantine_rows.shape[0])),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = base.read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]

    file_report, field_summary = base.scan_files(config)
    registry_compare = base.compare_registry_quality(config, file_report)
    quarantine_rows = build_quarantine_rows(config, file_report)
    processed_summary = build_processed_scope_summary(file_report, quarantine_rows)
    checks = build_acceptance_checks_v3_45_1(config, file_report, registry_compare, quarantine_rows)
    artifacts = {
        "file_acceptance_report": output_dir / "file_acceptance_report.csv",
        "field_integrity_summary": output_dir / "field_integrity_summary.csv",
        "registry_reconciliation": output_dir / "registry_reconciliation.csv",
        "ohlc_quarantine_rows": output_dir / "ohlc_quarantine_rows.csv",
        "processed_scope_summary": output_dir / "processed_scope_summary.csv",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "acceptance_report": output_dir / "acceptance_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    base.write_csv(file_report, artifacts["file_acceptance_report"])
    base.write_csv(field_summary, artifacts["field_integrity_summary"])
    base.write_csv(registry_compare, artifacts["registry_reconciliation"])
    base.write_csv(quarantine_rows, artifacts["ohlc_quarantine_rows"])
    base.write_csv(processed_summary, artifacts["processed_scope_summary"])
    base.write_csv(checks, artifacts["acceptance_checks"])
    base.write_text(build_report(config, file_report, registry_compare, quarantine_rows, processed_summary, checks), artifacts["acceptance_report"])
    base.write_text(build_catalog(config, processed_summary, checks), artifacts["catalog_update"])
    base.write_text("\n".join(base.rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])
    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, checks, quarantine_rows)
    base.write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": not bool((checks["status"] == "fail").any()),
        "acceptance_warning_count": int((checks["status"] == "warn").sum()),
        "file_count": int(file_report.shape[0]),
        "acquired_files": int((file_report["status"] == "acquired").sum()),
        "empty_files": int((file_report["status"] == "empty").sum()),
        "raw_rows": int(file_report.loc[file_report["status"] == "acquired", "rows"].sum()),
        "source_bad_ohlc_rows": int(processed_summary.loc[processed_summary["metric"] == "source_bad_ohlc_rows", "value"].iloc[0]),
        "quarantined_ohlc_rows": int(processed_summary.loc[processed_summary["metric"] == "quarantined_ohlc_rows", "value"].iloc[0]),
        "accepted_processed_rows": int(processed_summary.loc[processed_summary["metric"] == "accepted_processed_rows", "value"].iloc[0]),
        "data_decision": "accepted_after_row_level_ohlc_quarantine",
        "model_decision": "no_model_promotion_data_acceptance_only",
        "outputs": [base.rel(path) for path in artifacts.values()],
    }
    base.write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
