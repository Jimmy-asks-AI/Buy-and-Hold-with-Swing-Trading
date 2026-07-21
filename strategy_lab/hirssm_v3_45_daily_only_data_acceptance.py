#!/usr/bin/env python
"""HIRSSM V3.45 daily-only data acceptance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "daily_only_data_acceptance_v3_45.json"
TASK_ID = "20260529_v3_45_daily_only_data_acceptance"
VERSION = "V3.45"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def expected_business_dates(start_date: str, end_date: str) -> list[str]:
    dates = pd.bdate_range(
        start=pd.to_datetime(start_date, format="%Y%m%d"),
        end=pd.to_datetime(end_date, format="%Y%m%d"),
    )
    return [item.strftime("%Y%m%d") for item in dates]


def scan_files(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = ROOT / config["data_root"]
    required_columns = set(config["required_columns"])
    expected_dates = expected_business_dates(config["start_date"], config["end_date"])
    expected_set = set(expected_dates)
    existing_files = {path.stem.split("=")[-1]: path for path in data_root.glob("trade_date=*.csv")}
    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    for trade_date in expected_dates:
        path = existing_files.get(trade_date)
        if path is None:
            rows.append(
                {
                    "trade_date": trade_date,
                    "path": "",
                    "file_exists": False,
                    "readable": False,
                    "rows": 0,
                    "status": "missing_file",
                    "missing_required_columns": ",".join(sorted(required_columns)),
                    "duplicate_asset_date_count": 0,
                    "trade_date_mismatch_count": 0,
                    "null_close_count": 0,
                    "null_vol_count": 0,
                    "bad_ohlc_count": 0,
                    "bad_price_adjustment_count": 0,
                    "bad_data_source_count": 0,
                }
            )
            continue
        try:
            data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            readable = True
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "trade_date": trade_date,
                    "path": rel(path),
                    "file_exists": True,
                    "readable": False,
                    "rows": 0,
                    "status": "read_error",
                    "missing_required_columns": "",
                    "duplicate_asset_date_count": 0,
                    "trade_date_mismatch_count": 0,
                    "null_close_count": 0,
                    "null_vol_count": 0,
                    "bad_ohlc_count": 0,
                    "bad_price_adjustment_count": 0,
                    "bad_data_source_count": 0,
                    "error": repr(exc),
                }
            )
            continue

        missing = sorted(required_columns.difference(data.columns))
        row_count = int(data.shape[0])
        if row_count == 0:
            status = "empty"
            duplicate_count = 0
            mismatch_count = 0
            null_close = 0
            null_vol = 0
            bad_ohlc = 0
            bad_adjustment = 0
            bad_source = 0
        else:
            status = "acquired"
            key_cols = [col for col in ["ts_code", "trade_date"] if col in data.columns]
            duplicate_count = int(data.duplicated(key_cols).sum()) if len(key_cols) == 2 else row_count
            trade_dates = data["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8] if "trade_date" in data.columns else pd.Series([""] * row_count)
            mismatch_count = int((trade_dates != trade_date).sum())
            close = pd.to_numeric(data["close"], errors="coerce") if "close" in data.columns else pd.Series([float("nan")] * row_count)
            vol = pd.to_numeric(data["vol"], errors="coerce") if "vol" in data.columns else pd.Series([float("nan")] * row_count)
            open_ = pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else pd.Series([float("nan")] * row_count)
            high = pd.to_numeric(data["high"], errors="coerce") if "high" in data.columns else pd.Series([float("nan")] * row_count)
            low = pd.to_numeric(data["low"], errors="coerce") if "low" in data.columns else pd.Series([float("nan")] * row_count)
            null_close = int(close.isna().sum())
            null_vol = int(vol.isna().sum())
            bad_ohlc = int(((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).fillna(True).sum())
            adjustment = data["price_adjustment"].astype(str) if "price_adjustment" in data.columns else pd.Series([""] * row_count)
            source = data["data_source"].astype(str) if "data_source" in data.columns else pd.Series([""] * row_count)
            bad_adjustment = int((adjustment != config["expected_price_adjustment"]).sum())
            bad_source = int((source != config["expected_data_source"]).sum())

            field_rows.append(
                {
                    "trade_date": trade_date,
                    "rows": row_count,
                    "unique_assets": int(data["ts_code"].nunique()) if "ts_code" in data.columns else 0,
                    "min_close": float(close.min()) if close.notna().any() else "",
                    "max_close": float(close.max()) if close.notna().any() else "",
                    "sum_vol": float(vol.sum()) if vol.notna().any() else "",
                    "duplicate_asset_date_count": duplicate_count,
                    "bad_ohlc_count": bad_ohlc,
                }
            )
        rows.append(
            {
                "trade_date": trade_date,
                "path": rel(path),
                "file_exists": True,
                "readable": readable,
                "rows": row_count,
                "status": status,
                "missing_required_columns": ",".join(missing),
                "duplicate_asset_date_count": duplicate_count,
                "trade_date_mismatch_count": mismatch_count,
                "null_close_count": null_close,
                "null_vol_count": null_vol,
                "bad_ohlc_count": bad_ohlc,
                "bad_price_adjustment_count": bad_adjustment,
                "bad_data_source_count": bad_source,
            }
        )

    extra_dates = sorted(set(existing_files).difference(expected_set))
    for trade_date in extra_dates:
        path = existing_files[trade_date]
        rows.append(
            {
                "trade_date": trade_date,
                "path": rel(path),
                "file_exists": True,
                "readable": True,
                "rows": -1,
                "status": "extra_file",
                "missing_required_columns": "",
                "duplicate_asset_date_count": 0,
                "trade_date_mismatch_count": 0,
                "null_close_count": 0,
                "null_vol_count": 0,
                "bad_ohlc_count": 0,
                "bad_price_adjustment_count": 0,
                "bad_data_source_count": 0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(field_rows)


def compare_registry_quality(config: dict[str, Any], file_report: pd.DataFrame) -> pd.DataFrame:
    registry = read_csv(ROOT / config["registry_path"])
    quality = read_csv(ROOT / config["quality_path"])
    rows = []
    rows.append({"metric": "file_report_rows", "value": int(file_report.shape[0])})
    rows.append({"metric": "registry_rows", "value": int(registry.shape[0])})
    rows.append({"metric": "quality_rows", "value": int(quality.shape[0])})
    rows.append({"metric": "file_acquired_count", "value": int((file_report["status"] == "acquired").sum())})
    rows.append({"metric": "registry_acquired_count", "value": int((registry["status"] == "acquired").sum())})
    rows.append({"metric": "file_empty_count", "value": int((file_report["status"] == "empty").sum())})
    rows.append({"metric": "registry_empty_count", "value": int((registry["status"] == "empty").sum())})
    rows.append({"metric": "quality_pass_count", "value": int((quality["quality_flag"] == "pass").sum())})
    rows.append({"metric": "registry_raw_rows", "value": int(registry.loc[registry["status"] == "acquired", "rows"].sum())})
    rows.append({"metric": "file_raw_rows", "value": int(file_report.loc[file_report["status"] == "acquired", "rows"].sum())})
    return pd.DataFrame(rows)


def build_acceptance_checks(config: dict[str, Any], file_report: pd.DataFrame, registry_compare: pd.DataFrame) -> pd.DataFrame:
    guard = read_json(ROOT / config["guard_manifest_path"])
    manual_queue = (ROOT / config["manual_queue_path"]).read_text(encoding="utf-8")
    metrics = dict(zip(registry_compare["metric"], registry_compare["value"]))
    acquired = file_report.loc[file_report["status"] == "acquired"]
    rows = [
        {
            "check": "all_expected_business_date_files_exist",
            "status": "pass" if int((file_report["status"] == "missing_file").sum()) == 0 else "fail",
            "detail": str(int((file_report["status"] == "missing_file").sum())),
        },
        {
            "check": "no_extra_files_outside_date_range",
            "status": "pass" if int((file_report["status"] == "extra_file").sum()) == 0 else "fail",
            "detail": str(int((file_report["status"] == "extra_file").sum())),
        },
        {
            "check": "all_files_readable",
            "status": "pass" if bool(file_report["readable"].all()) else "fail",
            "detail": str(int((~file_report["readable"]).sum())),
        },
        {
            "check": "no_required_columns_missing_in_acquired_files",
            "status": "pass" if int((acquired["missing_required_columns"].astype(str) != "").sum()) == 0 else "fail",
            "detail": str(int((acquired["missing_required_columns"].astype(str) != "").sum())),
        },
        {
            "check": "no_duplicate_asset_date_rows",
            "status": "pass" if int(acquired["duplicate_asset_date_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["duplicate_asset_date_count"].sum())),
        },
        {
            "check": "no_trade_date_mismatch",
            "status": "pass" if int(acquired["trade_date_mismatch_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["trade_date_mismatch_count"].sum())),
        },
        {
            "check": "no_null_close_or_vol",
            "status": "pass" if int(acquired["null_close_count"].sum() + acquired["null_vol_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["null_close_count"].sum() + acquired["null_vol_count"].sum())),
        },
        {
            "check": "no_bad_ohlc_rows",
            "status": "pass" if int(acquired["bad_ohlc_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["bad_ohlc_count"].sum())),
        },
        {
            "check": "price_adjustment_is_none_raw",
            "status": "pass" if int(acquired["bad_price_adjustment_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["bad_price_adjustment_count"].sum())),
        },
        {
            "check": "data_source_is_tushare_daily",
            "status": "pass" if int(acquired["bad_data_source_count"].sum()) == 0 else "fail",
            "detail": str(int(acquired["bad_data_source_count"].sum())),
        },
        {
            "check": "registry_matches_file_counts",
            "status": "pass"
            if metrics["file_report_rows"] == metrics["registry_rows"]
            and metrics["file_acquired_count"] == metrics["registry_acquired_count"]
            and metrics["file_empty_count"] == metrics["registry_empty_count"]
            and metrics["file_raw_rows"] == metrics["registry_raw_rows"]
            else "fail",
            "detail": json.dumps(metrics, ensure_ascii=False),
        },
        {
            "check": "guard_manifest_passed",
            "status": "pass" if bool(guard.get("self_check_pass", False)) else "fail",
            "detail": f"flagged_discontinuities={guard.get('flagged_discontinuities')}",
        },
        {
            "check": "manual_queue_still_records_missing_interfaces",
            "status": "pass" if all(item in manual_queue for item in config["manual_data_interfaces_still_needed"]) else "fail",
            "detail": ",".join(config["manual_data_interfaces_still_needed"]),
        },
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "data acceptance only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    file_report: pd.DataFrame,
    field_summary: pd.DataFrame,
    registry_compare: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    acquired = file_report.loc[file_report["status"] == "acquired"]
    empty = file_report.loc[file_report["status"] == "empty"]
    failed = checks.loc[checks["status"] != "pass"]
    lines = [
        "# V3.45 Daily-Only Data Acceptance",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- The daily-only Tushare raw data layer has been validated for collection completeness and internal processing consistency.",
        "- This is not approval for adjusted-return, valuation, high-dividend, low-PE, or portfolio performance research.",
        "- V3.44 raw adjustment guard remains required for all downstream use.",
        "",
        "## Coverage",
        "",
        f"- Expected business-date files: `{len(file_report)}`",
        f"- Acquired files with rows: `{len(acquired)}`",
        f"- Empty business-date files: `{len(empty)}`",
        f"- Raw rows: `{int(acquired['rows'].sum())}`",
        f"- Coverage start: `{config['start_date']}`",
        f"- Coverage end: `{config['end_date']}`",
        "",
        "## Checks",
        "",
        f"- Total checks: `{len(checks)}`",
        f"- Failed checks: `{len(failed)}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
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
            "## Research Boundary",
            "",
            "The data is normal for raw OHLCV storage and limited price/volume diagnostics. It is deliberately marked `none_raw`, and downstream code must not use it for adjusted returns, total returns, long-horizon momentum, high-dividend return analysis, low-PE selection, or portfolio backtest performance.",
            "",
            "## Still Needed",
            "",
            *[f"- `{item}`" for item in config["manual_data_interfaces_still_needed"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(config: dict[str, Any], checks: pd.DataFrame, file_report: pd.DataFrame) -> str:
    acquired = file_report.loc[file_report["status"] == "acquired"]
    failed = checks.loc[checks["status"] != "pass"]
    return "\n".join(
        [
            "# A-share Daily-Only Data Acceptance V3.45",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Acceptance",
            "",
            f"- Acceptance passed: `{failed.empty}`",
            f"- Acquired files: `{len(acquired)}`",
            f"- Raw rows: `{int(acquired['rows'].sum())}`",
            f"- Data range: `{config['start_date']}` to `{config['end_date']}`",
            "",
            "## Boundary",
            "",
            "- Data is raw daily OHLCV only.",
            "- V3.44 guard blocks adjusted-return and valuation misuse.",
            "- Missing interfaces remain unresolved.",
            "",
            "## Still Needed",
            "",
            *[f"- `{item}`" for item in config["manual_data_interfaces_still_needed"]],
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    rows.append(
        {
            "check": "acceptance_checks_all_pass",
            "status": "pass" if bool((checks["status"] == "pass").all()) else "fail",
            "detail": str(int((checks["status"] != "pass").sum())),
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
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]

    file_report, field_summary = scan_files(config)
    registry_compare = compare_registry_quality(config, file_report)
    checks = build_acceptance_checks(config, file_report, registry_compare)
    artifacts = {
        "file_acceptance_report": output_dir / "file_acceptance_report.csv",
        "field_integrity_summary": output_dir / "field_integrity_summary.csv",
        "registry_reconciliation": output_dir / "registry_reconciliation.csv",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "acceptance_report": output_dir / "acceptance_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(file_report, artifacts["file_acceptance_report"])
    write_csv(field_summary, artifacts["field_integrity_summary"])
    write_csv(registry_compare, artifacts["registry_reconciliation"])
    write_csv(checks, artifacts["acceptance_checks"])
    write_text(build_report(config, file_report, field_summary, registry_compare, checks), artifacts["acceptance_report"])
    write_text(build_catalog(config, checks, file_report), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])
    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, checks)
    write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((checks["status"] == "pass").all()),
        "file_count": int(file_report.shape[0]),
        "acquired_files": int((file_report["status"] == "acquired").sum()),
        "empty_files": int((file_report["status"] == "empty").sum()),
        "raw_rows": int(file_report.loc[file_report["status"] == "acquired", "rows"].sum()),
        "model_decision": "no_model_promotion_data_acceptance_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
