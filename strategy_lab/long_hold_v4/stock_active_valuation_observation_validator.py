"""Validate active-candidate valuation observations against cached providers.

The validation establishes source consistency for current research.  It does
not convert current-final-snapshot histories into point-in-time backtest data,
and it contains no model-performance or alpha evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_stock_market_history_validator import (
    compare_eastmoney_valuation,
    compare_joinquant_valuation,
)
from .stock_active_valuation_observation_collector import (
    MANIFEST_PATH as OBSERVATION_MANIFEST_PATH,
    OBSERVATION_PATH,
    QUALIFICATION_STATUS,
    ROOT,
    STATUS_PATH,
    WATCHLIST_PATH,
    _atomic_csv,
    _atomic_json,
    _relative,
    _sha256,
)


BAOSTOCK_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_market_history_builder_latest.json"
)
JOINQUANT_DIR = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "validation_sources" / "joinquant_valuation"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "stock_active_valuation_observation_validation"
JOINQUANT_CHECKS_PATH = OUTPUT_DIR / "joinquant_cross_source_checks.csv"
BAOSTOCK_CHECKS_PATH = OUTPUT_DIR / "baostock_cross_source_checks.csv"
ASSET_METRICS_PATH = OUTPUT_DIR / "cross_source_asset_metrics.csv"
OUTLIERS_PATH = OUTPUT_DIR / "cross_source_outliers.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

THRESHOLDS = {
    "joinquant_min_assets": 100,
    "joinquant_min_asset_coverage": 0.95,
    "joinquant_min_checks": 1_000,
    "joinquant_pe_median_abs_relative_max": 0.02,
    "joinquant_pe_p95_abs_relative_max": 0.10,
    "joinquant_pb_median_abs_relative_max": 0.02,
    "joinquant_pb_p95_abs_relative_max": 0.10,
    "joinquant_cap_median_abs_relative_max": 0.01,
    "joinquant_cap_p95_abs_relative_max": 0.03,
    "baostock_min_assets": 10,
    "baostock_min_checks": 1_000,
    "baostock_pe_median_abs_relative_max": 0.02,
    "baostock_pe_p95_abs_relative_max": 0.10,
    "baostock_pb_median_abs_relative_max": 0.05,
    "baostock_pb_p95_abs_relative_max": 0.15,
    "baostock_cap_median_abs_relative_max": 0.01,
    "baostock_cap_p95_abs_relative_max": 0.03,
}

REQUIRED_OBSERVATION_COLUMNS = {
    "date",
    "asset",
    "pe_ttm",
    "pb_mrq",
    "float_market_cap",
    "available_date",
    "data_source",
    "current_final_snapshot",
    "pit_actionable",
    "qualification_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def validate_observation_integrity(
    observation: pd.DataFrame,
    status: pd.DataFrame,
    manifest: dict[str, Any],
    as_of: str | pd.Timestamp,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_OBSERVATION_COLUMNS.difference(observation.columns))
    dates = pd.to_datetime(observation.get("date"), errors="coerce")
    available = pd.to_datetime(observation.get("available_date"), errors="coerce")
    target_assets = int(manifest.get("target_assets", -1))
    completed_assets = int(manifest.get("completed_assets", -1))
    checks = {
        "required_columns": not missing,
        "manifest_row_count": len(observation) == int(manifest.get("observation_rows", -1)),
        "manifest_asset_count": observation["asset"].nunique() == completed_assets == target_assets,
        "status_asset_count": len(status) == target_assets,
        "all_status_completed": bool(status["collection_status"].eq("completed").all()),
        "status_has_no_errors": bool(status["error"].fillna("").eq("").all()),
        "valid_keys": bool(dates.notna().all() and observation["asset"].notna().all()),
        "unique_keys": not observation.assign(date=dates).duplicated(["date", "asset"]).any(),
        "no_future_market_dates": bool(dates.le(pd.Timestamp(as_of).normalize()).all()),
        "available_not_before_market_date": bool(available.notna().all() and available.ge(dates).all()),
        "source_is_current_final_snapshot": bool(_as_bool(observation["current_final_snapshot"]).eq(True).all()),
        "pit_actionable_is_false": bool(_as_bool(observation["pit_actionable"]).eq(False).all()),
        "historical_backtest_is_false": bool(
            _as_bool(observation["historical_backtest_allowed"]).eq(False).all()
        ),
        "model_promotion_is_false": bool(_as_bool(observation["model_promotion_allowed"]).eq(False).all()),
        "qualification_status_is_observation": bool(
            observation["qualification_status"].eq(QUALIFICATION_STATUS).all()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "missing_columns": missing,
        "rows": int(len(observation)),
        "assets": int(observation["asset"].nunique()),
        "date_start": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().date().isoformat() if dates.notna().any() else None,
    }


def load_joinquant_history(path: Path = JOINQUANT_DIR) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    files = sorted(path.glob("date=*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no JoinQuant valuation cache files under {path}")
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    for file_path in files:
        frame = pd.read_csv(file_path, compression="gzip", dtype={"asset": str}, low_memory=False)
        required = {"date", "asset", "pe_ratio", "pb_ratio", "circulating_market_cap"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"JoinQuant cache {file_path.name} missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
        expected_date = pd.Timestamp(file_path.name.removeprefix("date=").removesuffix(".csv.gz"))
        if frame["date"].isna().any() or not frame["date"].eq(expected_date).all():
            raise ValueError(f"JoinQuant cache {file_path.name} contains an unexpected date")
        if frame["asset"].duplicated().any():
            raise ValueError(f"JoinQuant cache {file_path.name} contains duplicate assets")
        frames.append(frame)
        inputs.append({"path": _relative(file_path), "sha256": _sha256(file_path)})
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["date", "asset"]).any():
        raise ValueError("JoinQuant valuation history contains duplicate keys")
    return output, inputs


def load_baostock_valuation(
    manifest_path: Path = BAOSTOCK_MANIFEST_PATH,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valuation_manifest = manifest.get("valuation", {})
    relative_path = valuation_manifest.get("output_path")
    if not relative_path:
        raise ValueError("BaoStock builder manifest has no valuation output path")
    output_path = ROOT / relative_path
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    if _sha256(output_path) != valuation_manifest.get("output_sha256"):
        raise ValueError("BaoStock valuation output hash does not match its builder manifest")
    frame = pd.read_csv(output_path, dtype={"asset": str}, parse_dates=["date"], low_memory=False)
    required = {"date", "asset", "pe_ttm", "pb", "market_cap"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BaoStock valuation output missing columns: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("BaoStock valuation output contains duplicate keys")
    return frame, output_path, valuation_manifest


def _source_checks(
    source: str,
    metrics: dict[str, Any],
    asset_count: int,
    asset_coverage: float | None = None,
) -> dict[str, bool]:
    prefix = "joinquant" if source == "joinquant" else "baostock"
    checks = {
        "minimum_assets": asset_count >= THRESHOLDS[f"{prefix}_min_assets"],
        "minimum_checks": int(metrics.get("checks", 0)) >= THRESHOLDS[f"{prefix}_min_checks"],
        "pe_agreement": metrics.get("pe_median_abs_relative_error") is not None
        and metrics["pe_median_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_pe_median_abs_relative_max"]
        and metrics["pe_p95_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_pe_p95_abs_relative_max"],
        "pb_agreement": metrics.get("pb_median_abs_relative_error") is not None
        and metrics["pb_median_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_pb_median_abs_relative_max"]
        and metrics["pb_p95_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_pb_p95_abs_relative_max"],
        "cap_agreement": metrics.get("cap_median_abs_relative_error") is not None
        and metrics["cap_median_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_cap_median_abs_relative_max"]
        and metrics["cap_p95_abs_relative_error"]
        <= THRESHOLDS[f"{prefix}_cap_p95_abs_relative_max"],
    }
    if source == "joinquant":
        checks["minimum_asset_coverage"] = bool(
            asset_coverage is not None
            and asset_coverage >= THRESHOLDS["joinquant_min_asset_coverage"]
        )
    return checks


def build_asset_metrics(
    checks: pd.DataFrame,
    source: str,
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "source",
        "asset",
        "sector",
        "checks",
        "pe_median_abs_relative_error",
        "pe_p95_abs_relative_error",
        "pb_median_abs_relative_error",
        "pb_p95_abs_relative_error",
        "cap_median_abs_relative_error",
        "cap_p95_abs_relative_error",
        "warning_flags",
    ]
    if checks.empty:
        return pd.DataFrame(columns=columns)
    grouped = checks.groupby("asset", as_index=False).agg(
        checks=("date", "size"),
        pe_median_abs_relative_error=("pe_abs_relative_error", "median"),
        pe_p95_abs_relative_error=("pe_abs_relative_error", lambda values: values.quantile(0.95)),
        pb_median_abs_relative_error=("pb_abs_relative_error", "median"),
        pb_p95_abs_relative_error=("pb_abs_relative_error", lambda values: values.quantile(0.95)),
        cap_median_abs_relative_error=("cap_abs_relative_error", "median"),
        cap_p95_abs_relative_error=("cap_abs_relative_error", lambda values: values.quantile(0.95)),
    )
    grouped.insert(0, "source", source)
    grouped = grouped.merge(sectors, on="asset", how="left", validate="one_to_one")
    grouped["sector"] = grouped["sector"].fillna("unknown")
    prefix = "joinquant" if source == "joinquant" else "baostock"

    def warnings(row: pd.Series) -> str:
        flags: list[str] = []
        if row["pe_p95_abs_relative_error"] > THRESHOLDS[f"{prefix}_pe_p95_abs_relative_max"]:
            flags.append("pe_tail")
        if row["pb_p95_abs_relative_error"] > THRESHOLDS[f"{prefix}_pb_p95_abs_relative_max"]:
            flags.append("pb_tail")
        if row["cap_p95_abs_relative_error"] > THRESHOLDS[f"{prefix}_cap_p95_abs_relative_max"]:
            flags.append("cap_tail")
        return "|".join(flags)

    grouped["warning_flags"] = grouped.apply(warnings, axis=1)
    return grouped[columns].sort_values(["source", "asset"]).reset_index(drop=True)


def _sector_metrics(checks: pd.DataFrame, sectors: pd.DataFrame) -> list[dict[str, Any]]:
    if checks.empty:
        return []
    frame = checks.merge(sectors, on="asset", how="left", validate="many_to_one")
    frame["sector"] = frame["sector"].fillna("unknown")
    rows: list[dict[str, Any]] = []
    for sector, group in frame.groupby("sector"):
        rows.append(
            {
                "sector": sector,
                "assets": int(group["asset"].nunique()),
                "checks": int(len(group)),
                "pe_median_abs_relative_error": round(float(group["pe_abs_relative_error"].median()), 8),
                "pe_p95_abs_relative_error": round(float(group["pe_abs_relative_error"].quantile(0.95)), 8),
                "pb_median_abs_relative_error": round(float(group["pb_abs_relative_error"].median()), 8),
                "pb_p95_abs_relative_error": round(float(group["pb_abs_relative_error"].quantile(0.95)), 8),
                "cap_median_abs_relative_error": round(float(group["cap_abs_relative_error"].median()), 8),
                "cap_p95_abs_relative_error": round(float(group["cap_abs_relative_error"].quantile(0.95)), 8),
            }
        )
    return sorted(rows, key=lambda row: row["sector"])


def run_validation(
    as_of: str,
    observation_path: Path = OBSERVATION_PATH,
    observation_manifest_path: Path = OBSERVATION_MANIFEST_PATH,
    status_path: Path = STATUS_PATH,
    candidate_path: Path = WATCHLIST_PATH,
    baostock_manifest_path: Path = BAOSTOCK_MANIFEST_PATH,
    joinquant_dir: Path = JOINQUANT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    observation_manifest = json.loads(observation_manifest_path.read_text(encoding="utf-8"))
    if _sha256(observation_path) != next(
        item["sha256"]
        for item in observation_manifest["outputs"]
        if item["role"] == "observation"
    ):
        raise ValueError("Eastmoney observation hash does not match its manifest")
    observation = pd.read_csv(
        observation_path,
        compression="gzip",
        dtype={"asset": str},
        parse_dates=["date"],
        low_memory=False,
    )
    observation["asset"] = observation["asset"].astype(str).str.zfill(6)
    status = pd.read_csv(status_path, dtype={"asset": str}, low_memory=False)
    status["asset"] = status["asset"].astype(str).str.zfill(6)
    integrity = validate_observation_integrity(observation, status, observation_manifest, as_of)
    if not integrity["pass"]:
        raise ValueError(f"Eastmoney observation integrity failed: {integrity['checks']}")

    candidates = pd.read_csv(candidate_path, dtype={"asset": str}, low_memory=False)
    required_candidate_columns = {"asset", "sector"}
    if not required_candidate_columns.issubset(candidates.columns):
        missing = sorted(required_candidate_columns.difference(candidates.columns))
        raise ValueError(f"candidate watchlist missing columns: {missing}")
    sectors = candidates[["asset", "sector"]].copy()
    sectors["asset"] = sectors["asset"].astype(str).str.zfill(6)
    if sectors["asset"].duplicated().any():
        raise ValueError("candidate watchlist contains duplicate assets")
    sectors = sectors.drop_duplicates("asset")
    candidate_assets = set(observation["asset"])
    base = observation.rename(columns={"pb_mrq": "pb", "float_market_cap": "market_cap"})[
        ["date", "asset", "pe_ttm", "pb", "market_cap"]
    ].copy()

    joinquant, joinquant_inputs = load_joinquant_history(joinquant_dir)
    jq_checks, jq_metrics = compare_joinquant_valuation(base, joinquant)
    jq_assets = int(jq_checks["asset"].nunique())
    jq_coverage = jq_assets / len(candidate_assets) if candidate_assets else 0.0
    jq_gate = _source_checks("joinquant", jq_metrics, jq_assets, jq_coverage)

    baostock, baostock_path, baostock_manifest = load_baostock_valuation(baostock_manifest_path)
    eastmoney_for_baostock = observation.rename(
        columns={
            "pe_ttm": "em_pe_ttm",
            "pb_mrq": "em_pb",
            "float_market_cap": "em_float_market_cap",
        }
    )[["date", "asset", "em_pe_ttm", "em_pb", "em_float_market_cap"]]
    baostock_checks, baostock_metrics = compare_eastmoney_valuation(
        baostock, eastmoney_for_baostock
    )
    baostock_assets = int(baostock_checks["asset"].nunique())
    baostock_gate = _source_checks("baostock", baostock_metrics, baostock_assets)

    jq_asset_metrics = build_asset_metrics(jq_checks, "joinquant", sectors)
    baostock_asset_metrics = build_asset_metrics(baostock_checks, "baostock", sectors)
    asset_metrics = pd.concat([jq_asset_metrics, baostock_asset_metrics], ignore_index=True)
    outliers = asset_metrics[asset_metrics["warning_flags"].ne("")].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    jq_checks_path = output_dir / JOINQUANT_CHECKS_PATH.name
    baostock_checks_path = output_dir / BAOSTOCK_CHECKS_PATH.name
    asset_metrics_path = output_dir / ASSET_METRICS_PATH.name
    outliers_path = output_dir / OUTLIERS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    manifest_path = output_dir / MANIFEST_PATH.name
    _atomic_csv(jq_checks, jq_checks_path)
    _atomic_csv(baostock_checks, baostock_checks_path)
    _atomic_csv(asset_metrics, asset_metrics_path)
    _atomic_csv(outliers, outliers_path)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    evidence_pass = integrity["pass"] and all(jq_gate.values()) and all(baostock_gate.values())
    report = {
        "created_at": created_at,
        "as_of_date": pd.Timestamp(as_of).date().isoformat(),
        "dataset_id": "stock_active_valuation_history_eastmoney_observation",
        "integrity": integrity,
        "joinquant_validation": {
            **jq_metrics,
            "assets": jq_assets,
            "candidate_asset_coverage": round(jq_coverage, 8),
            "dates": int(jq_checks["date"].nunique()),
            "date_start": jq_checks["date"].min().date().isoformat() if len(jq_checks) else None,
            "date_end": jq_checks["date"].max().date().isoformat() if len(jq_checks) else None,
            "checks_by_rule": jq_gate,
            "status": "PASS_EXISTING_METRIC_THRESHOLDS" if all(jq_gate.values()) else "WARNING",
            "sector_metrics": _sector_metrics(jq_checks, sectors),
        },
        "baostock_validation": {
            **baostock_metrics,
            "assets": baostock_assets,
            "candidate_asset_coverage": round(
                baostock_assets / len(candidate_assets) if candidate_assets else 0.0, 8
            ),
            "dates": int(baostock_checks["date"].nunique()),
            "date_start": baostock_checks["date"].min().date().isoformat()
            if len(baostock_checks)
            else None,
            "date_end": baostock_checks["date"].max().date().isoformat()
            if len(baostock_checks)
            else None,
            "checks_by_rule": baostock_gate,
            "status": "SUPPORTING_SAMPLE_PASS_LIMITED_COVERAGE"
            if all(baostock_gate.values())
            else "WARNING",
            "builder_asset_coverage": baostock_manifest.get("asset_coverage"),
            "sector_metrics": _sector_metrics(baostock_checks, sectors),
        },
        "asset_level_warnings": {
            "warning_assets": int(outliers["asset"].nunique()),
            "warning_rows": int(len(outliers)),
            "joinquant_warning_assets": int(
                outliers.loc[outliers["source"].eq("joinquant"), "asset"].nunique()
            ),
            "baostock_warning_assets": int(
                outliers.loc[outliers["source"].eq("baostock"), "asset"].nunique()
            ),
        },
        "thresholds": THRESHOLDS,
        "threshold_provenance": (
            "metric thresholds reuse the existing stock-market-history validator; new asset-count and coverage "
            "checks are post-observation diagnostics and have no independent holdout"
        ),
        "cross_source_diagnostic_status": (
            "PASS_WITHOUT_INDEPENDENT_HOLDOUT" if evidence_pass else "WARNING_REVIEW_REQUIRED"
        ),
        "current_cross_section_research_allowed": bool(integrity["pass"]),
        "current_historical_percentile_diagnostic_allowed": bool(evidence_pass),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "remaining_blockers": [
            "current-universe selection creates survivorship bias for historical research",
            "all historical Eastmoney rows share the actual 2026-07 observation vintage",
            "BaoStock overlap covers only a limited, code-ordered 15-asset candidate sample",
            "no independent holdout remains after this diagnostic review",
            "delisted-security valuation coverage is still incomplete",
        ],
        "interpretation": (
            "PE and float-market-cap histories are strongly consistent across providers; PB has material asset- and "
            "date-specific tail disagreements that must remain visible. The evidence supports current descriptive "
            "valuation percentiles, not historical strategy performance or alpha claims."
        ),
    }
    _atomic_json(report, report_path)
    inputs = [
        {"path": _relative(observation_path), "sha256": _sha256(observation_path)},
        {"path": _relative(observation_manifest_path), "sha256": _sha256(observation_manifest_path)},
        {"path": _relative(status_path), "sha256": _sha256(status_path)},
        {"path": _relative(candidate_path), "sha256": _sha256(candidate_path)},
        {"path": _relative(baostock_manifest_path), "sha256": _sha256(baostock_manifest_path)},
        {"path": _relative(baostock_path), "sha256": _sha256(baostock_path)},
        *joinquant_inputs,
    ]
    outputs = [
        {"role": "joinquant_checks", "path": _relative(jq_checks_path), "sha256": _sha256(jq_checks_path)},
        {"role": "baostock_checks", "path": _relative(baostock_checks_path), "sha256": _sha256(baostock_checks_path)},
        {"role": "asset_metrics", "path": _relative(asset_metrics_path), "sha256": _sha256(asset_metrics_path)},
        {"role": "outliers", "path": _relative(outliers_path), "sha256": _sha256(outliers_path)},
        {"role": "report", "path": _relative(report_path), "sha256": _sha256(report_path)},
    ]
    run_manifest = {
        "created_at": created_at,
        "validation_schema": "stock_active_valuation_cross_source_observation_v2",
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "cross_source_diagnostic_status": report["cross_source_diagnostic_status"],
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(run_manifest, manifest_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_validation(args.as_of)
    print(
        json.dumps(
            {
                "cross_source_diagnostic_status": report["cross_source_diagnostic_status"],
                "integrity_pass": report["integrity"]["pass"],
                "joinquant": {
                    key: report["joinquant_validation"][key]
                    for key in ["assets", "checks", "candidate_asset_coverage", "status"]
                },
                "baostock": {
                    key: report["baostock_validation"][key]
                    for key in ["assets", "checks", "candidate_asset_coverage", "status"]
                },
                "warning_assets": report["asset_level_warnings"]["warning_assets"],
                "historical_backtest_allowed": report["historical_backtest_allowed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
