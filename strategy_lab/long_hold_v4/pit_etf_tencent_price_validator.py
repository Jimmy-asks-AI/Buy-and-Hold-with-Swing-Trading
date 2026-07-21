"""Validate Tencent ETF raw prices against authenticated Sina lifecycle caches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .pit_etf_tencent_price_collector import _source_set_sha256
from .pit_source_code_archive import authenticate_current_or_archive


ROOT = Path(__file__).resolve().parents[2]
TENCENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tencent_etf_price_validation_latest.json"
)
TENCENT_RUN_ROOT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "tencent_etf_price"
)
LIFECYCLE_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_tencent_price"
THRESHOLDS = {
    "full_master_assets": 1_701,
    "full_delisted_assets": 123,
    "ready_asset_coverage_min": 0.98,
    "delisted_ready_coverage_min": 1.0,
    "overall_primary_row_coverage_min": 0.99,
    "assets_with_95pct_row_coverage_ratio_min": 0.99,
    "open_within_one_tick_min": 0.995,
    "close_within_one_tick_min": 0.995,
    "high_within_five_ticks_min": 0.99,
    "low_within_five_ticks_min": 0.99,
    "volume_relative_error_p95_max": 0.02,
    "tick_tolerance": 0.001000001,
    "five_tick_tolerance": 0.005000001,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if compressed else None,
    )
    temporary.replace(path)


def _load_authenticated_tencent_manifest(
    latest_path: Path = TENCENT_MANIFEST_PATH,
) -> tuple[dict[str, Any], Path, pd.DataFrame, pd.DataFrame]:
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    immutable_path = ROOT / str(latest.get("immutable_manifest_path", ""))
    if not immutable_path.is_file() or _sha256(immutable_path) != str(latest.get("immutable_manifest_sha256", "")):
        raise ValueError("Tencent ETF immutable manifest authentication failed")
    manifest = json.loads(immutable_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != latest.get("run_id"):
        raise ValueError("Tencent ETF latest and immutable manifest run ids differ")
    code_path = ROOT / str(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError as exc:
        raise ValueError("Tencent ETF collector code hash mismatch")
    outputs = {str(item["role"]): item for item in manifest.get("outputs", [])}
    required_outputs = {"tencent_raw_prices", "asset_status", "raw_response_inventory", "run_state"}
    if not required_outputs.issubset(outputs):
        raise ValueError("Tencent ETF manifest omits required outputs")
    for role in required_outputs:
        item = outputs[role]
        path = ROOT / str(item["path"])
        if not path.is_file() or _sha256(path) != str(item["sha256"]):
            raise ValueError(f"Tencent ETF output hash mismatch: {role}")
    status_path = ROOT / str(outputs["asset_status"]["path"])
    inventory_path = ROOT / str(outputs["raw_response_inventory"]["path"])
    status = pd.read_csv(status_path, dtype={"asset": str})
    inventory = pd.read_csv(inventory_path, dtype={"asset": str})
    status["asset"] = status["asset"].str.zfill(6)
    inventory["asset"] = inventory["asset"].str.zfill(6)
    required_inventory = {"asset", "page_number", "path", "sha256", "uncompressed_sha256"}
    if not required_inventory.issubset(inventory.columns) or inventory.duplicated(["asset", "page_number"]).any():
        raise ValueError("Tencent ETF raw-response inventory is malformed")
    rows = inventory.to_dict("records")
    for item in rows:
        path = ROOT / str(item["path"])
        if not path.is_file() or _sha256(path) != str(item["sha256"]):
            raise ValueError(f"Tencent ETF raw page hash mismatch: {item['asset']} page {item['page_number']}")
        try:
            content = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError(f"Tencent ETF raw page is not valid gzip: {path}") from exc
        if _bytes_sha256(content) != str(item["uncompressed_sha256"]):
            raise ValueError(f"Tencent ETF raw source hash mismatch: {item['asset']} page {item['page_number']}")
    if _source_set_sha256(rows) != str(manifest.get("raw_response_set_sha256", "")):
        raise ValueError("Tencent ETF raw-response source-set hash mismatch")
    return manifest, immutable_path, status, inventory


def _authenticate_sina_price_map(
    lifecycle_manifest_path: Path = LIFECYCLE_MANIFEST_PATH,
) -> tuple[dict[str, tuple[Path, str]], dict[str, Any]]:
    manifest = json.loads(lifecycle_manifest_path.read_text(encoding="utf-8"))
    immutable_value = str(manifest.get("immutable_manifest_path", ""))
    immutable_path = ROOT / immutable_value
    if not immutable_value or not immutable_path.is_file():
        raise ValueError("Sina ETF lifecycle immutable manifest is absent")
    if _sha256(immutable_path) != _sha256(lifecycle_manifest_path):
        raise ValueError("Sina ETF lifecycle latest and immutable manifests differ")
    code_files = manifest.get("code_files")
    if not isinstance(code_files, list) or not code_files:
        raise ValueError("Sina ETF lifecycle code bundle is absent")
    for item in code_files:
        path = ROOT / str(item.get("path", ""))
        try:
            authenticate_current_or_archive(path, str(item.get("sha256", "")))
        except ValueError as exc:
            raise ValueError(f"Sina ETF lifecycle code hash mismatch: {item.get('path')}") from exc
    if manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("Sina ETF lifecycle source violates observation-only boundary")
    price_map: dict[str, tuple[Path, str]] = {}
    for item in manifest.get("inputs", []):
        if item.get("role") != "etf_raw_price":
            continue
        asset = str(item.get("asset", "")).zfill(6)
        if asset in price_map:
            raise ValueError(f"Sina ETF lifecycle manifest contains duplicate raw prices for {asset}")
        path = ROOT / str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Sina ETF raw price hash mismatch for {asset}")
        price_map[asset] = (path, expected)
    if len(price_map) != int(manifest.get("selected_assets", -1)):
        raise ValueError("Sina ETF lifecycle manifest raw-price population is incomplete")
    return price_map, manifest


def compare_asset_prices(
    tencent: pd.DataFrame,
    sina: pd.DataFrame,
    *,
    asset: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, int], dict[str, np.ndarray]]:
    left = tencent.copy()
    right = sina.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    for frame, source in [(left, "Tencent"), (right, "Sina")]:
        if frame["date"].isna().any() or frame["date"].duplicated().any():
            raise ValueError(f"{source} ETF prices have invalid or duplicate dates for {asset}")
    required_left = {"open", "high", "low", "close", "volume_shares"}
    required_right = {"open", "high", "low", "close", "volume"}
    if not required_left.issubset(left.columns) or not required_right.issubset(right.columns):
        raise ValueError(f"ETF comparison inputs are incomplete for {asset}")
    merged = left[["date", *sorted(required_left)]].merge(
        right[["date", *sorted(required_right)]],
        on="date",
        how="outer",
        suffixes=("_tencent", "_sina"),
        indicator=True,
    )
    overlap = merged["_merge"].eq("both")
    errors: dict[str, np.ndarray] = {}
    for column in ["open", "high", "low", "close"]:
        values = (
            merged.loc[overlap, f"{column}_tencent"] - merged.loc[overlap, f"{column}_sina"]
        ).abs().to_numpy(dtype=float)
        errors[column] = values
        merged[f"{column}_absolute_error"] = np.nan
        merged.loc[overlap, f"{column}_absolute_error"] = values
    volume_error = (
        merged.loc[overlap, "volume_shares"] - merged.loc[overlap, "volume"]
    ).abs().to_numpy(dtype=float)
    volume_denominator = merged.loc[overlap, "volume"].replace(0, np.nan).to_numpy(dtype=float)
    volume_relative = np.divide(
        volume_error,
        volume_denominator,
        out=np.full_like(volume_error, np.nan),
        where=np.isfinite(volume_denominator) & (volume_denominator != 0),
    )
    errors["volume_relative"] = volume_relative
    merged["volume_relative_error"] = np.nan
    merged.loc[overlap, "volume_relative_error"] = volume_relative
    one_tick = THRESHOLDS["tick_tolerance"]
    five_ticks = THRESHOLDS["five_tick_tolerance"]
    overlap_rows = int(overlap.sum())
    primary_rows = int(len(right))
    independent_rows = int(len(left))

    def ratio(values: np.ndarray, tolerance: float) -> float:
        return float(np.mean(values <= tolerance)) if len(values) else 0.0

    summary: dict[str, Any] = {
        "asset": asset,
        "tencent_rows": independent_rows,
        "sina_rows": primary_rows,
        "overlap_rows": overlap_rows,
        "primary_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "independent_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "primary_row_coverage": overlap_rows / primary_rows if primary_rows else 0.0,
        "coverage_start_tencent": left["date"].min().date().isoformat() if len(left) else None,
        "coverage_end_tencent": left["date"].max().date().isoformat() if len(left) else None,
        "coverage_start_sina": right["date"].min().date().isoformat() if len(right) else None,
        "coverage_end_sina": right["date"].max().date().isoformat() if len(right) else None,
    }
    for column in ["open", "high", "low", "close"]:
        summary[f"{column}_exact_ratio"] = ratio(errors[column], 1e-12)
        summary[f"{column}_within_one_tick_ratio"] = ratio(errors[column], one_tick)
        summary[f"{column}_within_five_ticks_ratio"] = ratio(errors[column], five_ticks)
        summary[f"{column}_maximum_absolute_error"] = float(np.max(errors[column])) if len(errors[column]) else None
    finite_volume = volume_relative[np.isfinite(volume_relative)]
    summary["volume_relative_error_median"] = float(np.median(finite_volume)) if len(finite_volume) else None
    summary["volume_relative_error_p95"] = float(np.quantile(finite_volume, 0.95)) if len(finite_volume) else None
    exact_difference = overlap.copy()
    for column in ["open", "high", "low", "close"]:
        exact_difference &= merged[f"{column}_absolute_error"].fillna(0).le(1e-12)
    mismatch = merged[
        merged["_merge"].ne("both")
        | ~exact_difference
        | merged["volume_relative_error"].fillna(0).gt(1e-12)
    ].copy()
    mismatch.insert(0, "asset", asset)
    counts = {
        "tencent_rows": independent_rows,
        "sina_rows": primary_rows,
        "overlap_rows": overlap_rows,
        "primary_only_rows": summary["primary_only_rows"],
        "independent_only_rows": summary["independent_only_rows"],
    }
    return summary, mismatch, counts, errors


def _version_inventory(selected_assets: set[str]) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(TENCENT_RUN_ROOT.glob("*")):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        observed_date = str(manifest.get("created_at", ""))[:10]
        for metadata_path in (run_dir / "assets").glob("*/metadata.json"):
            asset = metadata_path.parent.name.zfill(6)
            if asset not in selected_assets:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if metadata.get("status") != "ready":
                continue
            source_hashes = sorted(str(item.get("uncompressed_sha256", "")) for item in metadata.get("source_pages", []))
            source_set = hashlib.sha256("\n".join(source_hashes).encode("ascii")).hexdigest()
            rows.append(
                {
                    "asset": asset,
                    "run_id": manifest.get("run_id"),
                    "observed_date": observed_date,
                    "source_response_set_sha256": source_set,
                }
            )
    details = pd.DataFrame(rows)
    if details.empty:
        return pd.DataFrame(
            columns=["asset", "run_count", "distinct_observation_dates", "distinct_source_versions", "version_depth_ready"]
        ), 0.0
    inventory = (
        details.groupby("asset")
        .agg(
            run_count=("run_id", "nunique"),
            distinct_observation_dates=("observed_date", "nunique"),
            distinct_source_versions=("source_response_set_sha256", "nunique"),
        )
        .reset_index()
    )
    inventory["version_depth_ready"] = inventory["distinct_observation_dates"].ge(2)
    inventory = pd.DataFrame({"asset": sorted(selected_assets)}).merge(inventory, on="asset", how="left").fillna(
        {"run_count": 0, "distinct_observation_dates": 0, "distinct_source_versions": 0, "version_depth_ready": False}
    )
    coverage = float(inventory["version_depth_ready"].mean()) if len(inventory) else 0.0
    return inventory, coverage


def _check(name: str, value: Any, operator: str, threshold: Any, passed: bool, note: str) -> dict[str, Any]:
    return {"check": name, "value": value, "operator": operator, "threshold": threshold, "passed": bool(passed), "note": note}


def identify_material_close_mismatches(mismatches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "asset",
        "date",
        "close_tencent",
        "close_sina",
        "close_absolute_error",
        "close_relative_error",
        "volume_relative_error",
    ]
    if mismatches.empty:
        return pd.DataFrame(columns=columns)
    frame = mismatches.copy()
    both = frame["_merge"].astype(str).eq("both")
    denominator = pd.to_numeric(frame["close_sina"], errors="coerce").abs().replace(0, np.nan)
    absolute_error = pd.to_numeric(frame["close_absolute_error"], errors="coerce")
    frame["close_relative_error"] = absolute_error / denominator
    material_threshold = np.maximum(0.05, denominator * 0.01)
    return frame.loc[both & absolute_error.ge(material_threshold - 1e-12), columns].sort_values(
        "close_absolute_error", ascending=False
    ).reset_index(drop=True)


def validate(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    manifest, immutable_path, status, inventory = _load_authenticated_tencent_manifest()
    sina_map, lifecycle_manifest = _authenticate_sina_price_map()
    selected_assets = set(status["asset"])
    ready_status = status[status["status"].eq("ready")].copy()
    asset_summaries: list[dict[str, Any]] = []
    mismatches: list[pd.DataFrame] = []
    totals = {key: 0 for key in ["tencent_rows", "sina_rows", "overlap_rows", "primary_only_rows", "independent_only_rows"]}
    error_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["open", "high", "low", "close", "volume_relative"]}
    for row in ready_status.itertuples(index=False):
        asset = str(row.asset).zfill(6)
        metadata_path = immutable_path.parent / "assets" / asset / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tencent_path = ROOT / str(metadata.get("price_path", ""))
        if not tencent_path.is_file() or _sha256(tencent_path) != str(metadata.get("price_sha256", "")):
            raise ValueError(f"Tencent ETF per-asset price hash mismatch for {asset}")
        if asset not in sina_map:
            raise ValueError(f"Sina ETF raw price is absent for {asset}")
        tencent = pd.read_csv(tencent_path, dtype={"asset": str})
        sina = pd.read_csv(sina_map[asset][0])
        summary, mismatch, counts, errors = compare_asset_prices(tencent, sina, asset=asset)
        summary["lifecycle_status"] = "delisted" if pd.notna(row.delist_date) else "listed"
        asset_summaries.append(summary)
        if not mismatch.empty:
            mismatches.append(mismatch)
        for key, value in counts.items():
            totals[key] += int(value)
        for key, values in errors.items():
            if len(values):
                error_arrays[key].append(values)
    asset_summary = pd.DataFrame(asset_summaries).sort_values("asset").reset_index(drop=True)
    mismatch_frame = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()
    material_mismatches = identify_material_close_mismatches(mismatch_frame)
    combined_errors = {
        key: np.concatenate(values) if values else np.array([], dtype=float) for key, values in error_arrays.items()
    }

    def ratio(values: np.ndarray, tolerance: float) -> float:
        return float(np.mean(values <= tolerance)) if len(values) else 0.0

    finite_volume = combined_errors["volume_relative"][np.isfinite(combined_errors["volume_relative"])]
    ready_ratio = len(ready_status) / len(status) if len(status) else 0.0
    delisted = status[status["delist_date"].notna()]
    delisted_ready_ratio = float(delisted["status"].eq("ready").mean()) if len(delisted) else 0.0
    overall_coverage = totals["overlap_rows"] / totals["sina_rows"] if totals["sina_rows"] else 0.0
    asset_coverage_ratio = float(asset_summary["primary_row_coverage"].ge(0.95).mean()) if len(asset_summary) else 0.0
    tencent_coverage_start = pd.to_datetime(
        asset_summary["coverage_start_tencent"], errors="coerce"
    ).min()
    tencent_coverage_end = pd.to_datetime(
        asset_summary["coverage_end_tencent"], errors="coerce"
    ).max()
    low_coverage_assets = asset_summary.loc[
        asset_summary["primary_row_coverage"].lt(0.95), "asset"
    ].astype(str).tolist()
    version_inventory, version_depth_coverage = _version_inventory(selected_assets)
    scope_full_master = len(status) == THRESHOLDS["full_master_assets"]
    scope_full_delisted = int(status["delist_date"].notna().sum()) == THRESHOLDS["full_delisted_assets"]
    checks = [
        _check("collector_errors", int(status["status"].eq("error").sum()), "==", 0, not status["status"].eq("error").any(), "collection attempts"),
        _check("ready_asset_coverage", ready_ratio, ">=", THRESHOLDS["ready_asset_coverage_min"], ready_ratio >= THRESHOLDS["ready_asset_coverage_min"], "all selected assets"),
        _check("delisted_population", int(status["delist_date"].notna().sum()), "==", THRESHOLDS["full_delisted_assets"], scope_full_delisted, "governed delisted ETF population"),
        _check("delisted_ready_coverage", delisted_ready_ratio, ">=", THRESHOLDS["delisted_ready_coverage_min"], delisted_ready_ratio >= THRESHOLDS["delisted_ready_coverage_min"], "all delisted ETFs"),
        _check("overall_primary_row_coverage", overall_coverage, ">=", THRESHOLDS["overall_primary_row_coverage_min"], overall_coverage >= THRESHOLDS["overall_primary_row_coverage_min"], "Tencent overlap relative to Sina rows"),
        _check("assets_with_95pct_row_coverage", asset_coverage_ratio, ">=", THRESHOLDS["assets_with_95pct_row_coverage_ratio_min"], asset_coverage_ratio >= THRESHOLDS["assets_with_95pct_row_coverage_ratio_min"], "per-asset date coverage"),
        _check("open_within_one_tick", ratio(combined_errors["open"], THRESHOLDS["tick_tolerance"]), ">=", THRESHOLDS["open_within_one_tick_min"], ratio(combined_errors["open"], THRESHOLDS["tick_tolerance"]) >= THRESHOLDS["open_within_one_tick_min"], "overlapping rows"),
        _check("close_within_one_tick", ratio(combined_errors["close"], THRESHOLDS["tick_tolerance"]), ">=", THRESHOLDS["close_within_one_tick_min"], ratio(combined_errors["close"], THRESHOLDS["tick_tolerance"]) >= THRESHOLDS["close_within_one_tick_min"], "overlapping rows"),
        _check("high_within_five_ticks", ratio(combined_errors["high"], THRESHOLDS["five_tick_tolerance"]), ">=", THRESHOLDS["high_within_five_ticks_min"], ratio(combined_errors["high"], THRESHOLDS["five_tick_tolerance"]) >= THRESHOLDS["high_within_five_ticks_min"], "Tencent historical rounding allowed but not rewritten"),
        _check("low_within_five_ticks", ratio(combined_errors["low"], THRESHOLDS["five_tick_tolerance"]), ">=", THRESHOLDS["low_within_five_ticks_min"], ratio(combined_errors["low"], THRESHOLDS["five_tick_tolerance"]) >= THRESHOLDS["low_within_five_ticks_min"], "Tencent historical rounding allowed but not rewritten"),
        _check("volume_relative_error_p95", float(np.quantile(finite_volume, 0.95)) if len(finite_volume) else None, "<=", THRESHOLDS["volume_relative_error_p95_max"], bool(len(finite_volume)) and float(np.quantile(finite_volume, 0.95)) <= THRESHOLDS["volume_relative_error_p95_max"], "Tencent lots converted to shares"),
        _check("full_master_scope", len(status), "==", THRESHOLDS["full_master_assets"], scope_full_master, "required before full-market source qualification"),
        _check("source_version_monitoring_depth", version_depth_coverage, "==", 1.0, version_depth_coverage == 1.0, "two collection dates monitor future revisions but do not create historical PIT availability"),
    ]
    checks_frame = pd.DataFrame(checks)
    source_content_checks = checks_frame[~checks_frame["check"].isin({"full_master_scope", "source_version_monitoring_depth"})]
    cross_source_content_passed = bool(source_content_checks["passed"].all())
    full_market_current_final_source_passed = bool(cross_source_content_passed and scope_full_master)
    version_monitoring_ready = version_depth_coverage == 1.0
    historical_source_qualified = False
    if full_market_current_final_source_passed and version_monitoring_ready:
        qualification = "PASS_FULL_MARKET_CURRENT_FINAL_WITH_DISCLOSED_TAILS_MONITORING_READY_PIT_BLOCKED"
    elif full_market_current_final_source_passed:
        qualification = "PASS_FULL_MARKET_CURRENT_FINAL_WITH_DISCLOSED_TAILS_VERSION_DEPTH_BLOCKED"
    elif cross_source_content_passed and scope_full_delisted:
        qualification = "PASS_DELISTED_CURRENT_FINAL_FULL_MARKET_PENDING"
    else:
        qualification = "BLOCKED_INDEPENDENT_PRICE_SOURCE_QUALITY_OR_COVERAGE"
    summary = {
        "as_of_date": manifest.get("as_of_date"),
        "qualification_status": qualification,
        "source_run_id": manifest.get("run_id"),
        "selected_assets": int(len(status)),
        "selected_delisted_assets": int(status["delist_date"].notna().sum()),
        "ready_assets": int(len(ready_status)),
        "error_assets": int(status["status"].eq("error").sum()),
        "no_data_assets": int(status["status"].eq("no_data").sum()),
        "tencent_rows": totals["tencent_rows"],
        "sina_rows": totals["sina_rows"],
        "overlap_rows": totals["overlap_rows"],
        "primary_only_rows": totals["primary_only_rows"],
        "independent_only_rows": totals["independent_only_rows"],
        "coverage_start": tencent_coverage_start.date().isoformat() if pd.notna(tencent_coverage_start) else None,
        "coverage_end": tencent_coverage_end.date().isoformat() if pd.notna(tencent_coverage_end) else None,
        "overall_primary_row_coverage": overall_coverage,
        "assets_with_95pct_row_coverage_ratio": asset_coverage_ratio,
        "open_within_one_tick_ratio": ratio(combined_errors["open"], THRESHOLDS["tick_tolerance"]),
        "close_within_one_tick_ratio": ratio(combined_errors["close"], THRESHOLDS["tick_tolerance"]),
        "high_within_five_ticks_ratio": ratio(combined_errors["high"], THRESHOLDS["five_tick_tolerance"]),
        "low_within_five_ticks_ratio": ratio(combined_errors["low"], THRESHOLDS["five_tick_tolerance"]),
        "maximum_close_absolute_error": float(np.max(combined_errors["close"])) if len(combined_errors["close"]) else None,
        "material_close_mismatch_rows": int(len(material_mismatches)),
        "material_close_mismatch_assets": int(material_mismatches["asset"].nunique()),
        "assets_below_95pct_row_coverage": int(len(low_coverage_assets)),
        "assets_below_95pct_row_coverage_codes": low_coverage_assets,
        "minimum_asset_row_coverage": float(asset_summary["primary_row_coverage"].min()),
        "volume_relative_error_p95": float(np.quantile(finite_volume, 0.95)) if len(finite_volume) else None,
        "ohlc_relationship_invalid_rows_disclosed": int(status["ohlc_relationship_invalid_rows"].sum()),
        "maximum_ohlc_rounding_gap": float(status["maximum_ohlc_rounding_gap"].max()),
        "cross_source_content_passed": cross_source_content_passed,
        "full_market_current_final_source_passed": full_market_current_final_source_passed,
        "version_depth_coverage": version_depth_coverage,
        "version_monitoring_ready": version_monitoring_ready,
        "historical_source_qualified": historical_source_qualified,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "formal_table_promotion_allowed": False,
        "boundary": "independent current-final price source validation; no PIT or performance claim",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_summary_path = output_dir / "asset_summary.csv"
    mismatch_path = output_dir / "cross_source_mismatches.csv.gz"
    material_mismatch_path = output_dir / "material_close_mismatches.csv"
    checks_path = output_dir / "qualification_checks.csv"
    versions_path = output_dir / "source_version_inventory.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "ETF_TENCENT_PRICE_AUDIT.md"
    _atomic_csv(asset_summary, asset_summary_path)
    _atomic_csv(mismatch_frame, mismatch_path, compressed=True)
    _atomic_csv(material_mismatches, material_mismatch_path)
    _atomic_csv(checks_frame, checks_path)
    _atomic_csv(version_inventory, versions_path)
    _atomic_json(summary, summary_path)
    worst_coverage = asset_summary.nsmallest(10, "primary_row_coverage")[
        ["asset", "tencent_rows", "sina_rows", "overlap_rows", "primary_row_coverage", "close_within_one_tick_ratio"]
    ]
    report_lines = [
        "# ETF Tencent Price Audit",
        "",
        f"Qualification: `{qualification}`",
        "",
        f"- Selected / ready: {len(status):,} / {len(ready_status):,}",
        f"- Delisted selected: {int(status['delist_date'].notna().sum()):,}",
        f"- Tencent / Sina / overlap rows: {totals['tencent_rows']:,} / {totals['sina_rows']:,} / {totals['overlap_rows']:,}",
        f"- Overall Sina-row coverage: {overall_coverage:.6%}",
        f"- Close within one tick: {summary['close_within_one_tick_ratio']:.6%}",
        f"- Material close mismatches: {summary['material_close_mismatch_rows']:,} rows across {summary['material_close_mismatch_assets']:,} assets",
        f"- Assets below 95% date coverage: {summary['assets_below_95pct_row_coverage']:,} ({', '.join(low_coverage_assets) or 'none'})",
        f"- Volume relative error p95: {summary['volume_relative_error_p95']:.6%}" if summary["volume_relative_error_p95"] is not None else "- Volume relative error p95: unavailable",
        f"- Source-version depth: {version_depth_coverage:.6%}",
        "",
        "## Worst Date Coverage",
        "",
        worst_coverage.to_markdown(index=False),
        "",
        "## Material Close Mismatches",
        "",
        material_mismatches.to_markdown(index=False),
        "",
        "These rows are disclosed source disagreements. They are not rewritten, excluded from the mismatch file, or treated as historical PIT evidence.",
        "",
        "## Boundary",
        "",
        "The Tencent source is a current-final observation. It can challenge Sina rows and monitor later revisions, but repeated observations cannot retroactively establish what a historical researcher saw.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    code_path = Path(__file__).resolve()
    outputs = [
        {"role": "asset_summary", "path": _relative(asset_summary_path), "sha256": _sha256(asset_summary_path), "rows": len(asset_summary)},
        {"role": "cross_source_mismatches", "path": _relative(mismatch_path), "sha256": _sha256(mismatch_path), "rows": len(mismatch_frame)},
        {"role": "material_close_mismatches", "path": _relative(material_mismatch_path), "sha256": _sha256(material_mismatch_path), "rows": len(material_mismatches)},
        {"role": "qualification_checks", "path": _relative(checks_path), "sha256": _sha256(checks_path), "rows": len(checks_frame)},
        {"role": "source_version_inventory", "path": _relative(versions_path), "sha256": _sha256(versions_path), "rows": len(version_inventory)},
        {"role": "summary", "path": _relative(summary_path), "sha256": _sha256(summary_path), "rows": 1},
        {"role": "report", "path": _relative(report_path), "sha256": _sha256(report_path), "rows": len(report_lines)},
    ]
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        **summary,
        "thresholds": THRESHOLDS,
        "inputs": [
            {"role": "tencent_latest_manifest", "path": _relative(TENCENT_MANIFEST_PATH), "sha256": _sha256(TENCENT_MANIFEST_PATH)},
            {"role": "tencent_immutable_manifest", "path": _relative(immutable_path), "sha256": _sha256(immutable_path)},
            {"role": "sina_lifecycle_manifest", "path": _relative(LIFECYCLE_MANIFEST_PATH), "sha256": _sha256(LIFECYCLE_MANIFEST_PATH)},
        ],
        "outputs": outputs,
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "sina_lifecycle_run_id": lifecycle_manifest.get("run_id"),
    }
    _atomic_json(run_manifest, output_dir / "run_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = validate(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
