"""Validate Eastmoney ETF NAV history without promoting current-final rows to PIT data."""

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

from .pit_etf_eastmoney_nav_collector import _source_set_sha256
from .pit_source_code_archive import authenticate_current_or_archive


ROOT = Path(__file__).resolve().parents[2]
EASTMONEY_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "eastmoney_etf_nav_validation_latest.json"
)
EASTMONEY_RUN_ROOT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "eastmoney_etf_nav"
)
JOINQUANT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "joinquant_etf_price_nav_validation_latest.json"
)
TERMINAL_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_cash_event_collector_latest.json"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_eastmoney_nav"
THRESHOLDS = {
    "full_master_assets": 1_701,
    "full_delisted_assets": 123,
    "ready_asset_coverage_min": 0.98,
    "delisted_ready_coverage_min": 1.0,
    "joinquant_nav_assets": 28,
    "joinquant_asset_coverage_min": 1.0,
    "joinquant_row_coverage_min": 0.995,
    "unit_nav_within_tolerance_min": 0.999,
    "cumulative_nav_comparable_coverage_min": 0.995,
    "cumulative_nav_within_tolerance_min": 0.999,
    "nav_tolerance": 1e-8,
}
NAV_VALIDATION_COLUMNS = [
    "date",
    "asset",
    "list_date",
    "delist_date",
    "unit_nav",
    "cumulative_nav",
    "available_date",
    "pit_actionable",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]


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


def _output_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["role"]): item for item in manifest.get("outputs", [])}


def _authenticate_output(item: dict[str, Any], label: str) -> Path:
    path = ROOT / str(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")
    return path


def _authenticate_code_bundle(manifest: dict[str, Any]) -> None:
    code_path = ROOT / str(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError as exc:
        raise ValueError("Eastmoney NAV collector code hash mismatch")
    files = manifest.get("code_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Eastmoney NAV collector dependency bundle is absent")
    for item in files:
        path = ROOT / str(item.get("path", ""))
        try:
            authenticate_current_or_archive(path, str(item.get("sha256", "")))
        except ValueError as exc:
            raise ValueError(f"Eastmoney NAV dependency hash mismatch: {item.get('path')}")
    bundle_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if bundle_hash != str(manifest.get("code_bundle_sha256", "")):
        raise ValueError("Eastmoney NAV dependency bundle hash mismatch")


def _authenticate_declared_code(manifest: dict[str, Any], label: str) -> None:
    code_path = ROOT / str(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError as exc:
        raise ValueError(f"{label} code hash mismatch") from exc
    files = manifest.get("code_files")
    if files is None:
        return
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} code bundle is malformed")
    for item in files:
        path = ROOT / str(item.get("path", ""))
        try:
            authenticate_current_or_archive(path, str(item.get("sha256", "")))
        except ValueError as exc:
            raise ValueError(f"{label} dependency hash mismatch: {item.get('path')}") from exc


def load_authenticated_eastmoney(
    latest_path: Path = EASTMONEY_MANIFEST_PATH,
) -> tuple[dict[str, Any], Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    immutable_path = ROOT / str(latest.get("immutable_manifest_path", ""))
    if not immutable_path.is_file() or _sha256(immutable_path) != str(latest.get("immutable_manifest_sha256", "")):
        raise ValueError("Eastmoney NAV immutable manifest authentication failed")
    manifest = json.loads(immutable_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != latest.get("run_id"):
        raise ValueError("Eastmoney NAV latest and immutable run ids differ")
    _authenticate_code_bundle(manifest)
    outputs = _output_map(manifest)
    required = {"eastmoney_nav", "asset_status", "raw_response_inventory", "run_state"}
    if not required.issubset(outputs):
        raise ValueError("Eastmoney NAV manifest omits required outputs")
    paths = {role: _authenticate_output(outputs[role], f"Eastmoney NAV {role}") for role in required}
    nav = pd.read_csv(
        paths["eastmoney_nav"],
        dtype={"asset": str},
        usecols=NAV_VALIDATION_COLUMNS,
        low_memory=False,
    )
    status = pd.read_csv(paths["asset_status"], dtype={"asset": str})
    inventory = pd.read_csv(paths["raw_response_inventory"], dtype={"asset": str})
    for frame in [nav, status, inventory]:
        if "asset" in frame.columns:
            frame["asset"] = frame["asset"].str.zfill(6)
    if len(nav) != int(outputs["eastmoney_nav"].get("rows", -1)):
        raise ValueError("Eastmoney NAV combined row count differs from manifest")
    if len(status) != int(outputs["asset_status"].get("rows", -1)):
        raise ValueError("Eastmoney NAV status row count differs from manifest")
    required_inventory = {"asset", "sequence", "path", "sha256", "uncompressed_sha256"}
    if not required_inventory.issubset(inventory.columns) or inventory.duplicated(["asset", "sequence"]).any():
        raise ValueError("Eastmoney NAV raw-response inventory is malformed")
    rows = inventory.to_dict("records")
    for item in rows:
        path = ROOT / str(item["path"])
        if not path.is_file() or _sha256(path) != str(item["sha256"]):
            raise ValueError(f"Eastmoney NAV raw response hash mismatch: {item['asset']} {item['sequence']}")
        try:
            content = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError(f"Eastmoney NAV raw response is not valid gzip: {path}") from exc
        if _bytes_sha256(content) != str(item["uncompressed_sha256"]):
            raise ValueError(f"Eastmoney NAV source-content hash mismatch: {item['asset']} {item['sequence']}")
    if _source_set_sha256(rows) != str(manifest.get("raw_response_set_sha256", "")):
        raise ValueError("Eastmoney NAV raw-response source-set hash mismatch")
    run_state = json.loads(paths["run_state"].read_text(encoding="utf-8"))
    if run_state.get("status") != "completed" or run_state.get("run_id") != manifest.get("run_id"):
        raise ValueError("Eastmoney NAV run state is not a matching completed run")
    if status["asset"].duplicated().any() or len(status) != int(manifest.get("selected_assets", -1)):
        raise ValueError("Eastmoney NAV selected population is inconsistent")
    if nav.duplicated(["asset", "date"]).any():
        raise ValueError("Eastmoney NAV combined output has duplicate keys")
    ready_rows = 0
    for row in status.itertuples(index=False):
        metadata_path = immutable_path.parent / "assets" / str(row.asset).zfill(6) / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != row.status:
            raise ValueError(f"Eastmoney NAV metadata status mismatch for {row.asset}")
        if row.status == "ready":
            nav_path = ROOT / str(metadata.get("nav_path", ""))
            if not nav_path.is_file() or _sha256(nav_path) != str(metadata.get("nav_sha256", "")):
                raise ValueError(f"Eastmoney NAV per-asset hash mismatch for {row.asset}")
            ready_rows += int(metadata.get("normalised_rows", -1))
    if ready_rows != len(nav):
        raise ValueError("Eastmoney NAV per-asset and combined row counts differ")
    return manifest, immutable_path, nav, status, inventory


def load_authenticated_joinquant(
    manifest_path: Path = JOINQUANT_MANIFEST_PATH,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _authenticate_declared_code(manifest, "JoinQuant ETF validation collector")
    outputs = _output_map(manifest)
    item = outputs.get("joinquant_etf_raw_nav")
    if item is None:
        raise ValueError("JoinQuant ETF NAV output is absent")
    path = _authenticate_output(item, "JoinQuant ETF NAV")
    nav = pd.read_csv(
        path,
        dtype={"asset": str},
        usecols=["date", "asset", "unit_nav", "cumulative_nav"],
        low_memory=False,
    )
    nav["asset"] = nav["asset"].str.zfill(6)
    if len(nav) != int(item.get("rows", -1)) or nav.duplicated(["asset", "date"]).any():
        raise ValueError("JoinQuant ETF NAV row contract failed")
    if manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("JoinQuant limited-window NAV source has an invalid qualification flag")
    return manifest, nav


def load_authenticated_terminal_event(
    manifest_path: Path = TERMINAL_MANIFEST_PATH,
) -> tuple[dict[str, Any], pd.Series]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _authenticate_declared_code(manifest, "ETF terminal-event collector")
    for document in manifest.get("documents", []):
        for path_key, hash_key in [("pdf_path", "pdf_sha256"), ("text_path", "text_sha256")]:
            path = ROOT / str(document.get(path_key, ""))
            if not path.is_file() or _sha256(path) != str(document.get(hash_key, "")):
                raise ValueError(f"ETF terminal-event document hash mismatch: {path_key}")
    outputs = _output_map(manifest)
    item = outputs.get("candidates")
    if item is None:
        raise ValueError("ETF terminal-event candidate output is absent")
    path = _authenticate_output(item, "ETF terminal-event candidates")
    events = pd.read_csv(path, dtype={"asset": str})
    events["asset"] = events["asset"].str.zfill(6)
    selected = events[events["asset"].eq("511210")]
    if len(selected) != 1:
        raise ValueError("Expected one authenticated 511210 terminal event")
    return manifest, selected.iloc[0]


def compare_nav_sources(
    eastmoney: pd.DataFrame,
    joinquant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    columns = ["asset", "date", "unit_nav", "cumulative_nav"]
    left = eastmoney[columns].copy()
    right = joinquant[columns].copy()
    for frame, source in [(left, "Eastmoney"), (right, "JoinQuant")]:
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        if frame[["asset", "date"]].isna().any(axis=None) or frame.duplicated(["asset", "date"]).any():
            raise ValueError(f"{source} NAV comparison keys are invalid")
    merged = right[["asset", "date", "unit_nav", "cumulative_nav"]].merge(
        left[["asset", "date", "unit_nav", "cumulative_nav"]],
        on=["asset", "date"],
        how="left",
        suffixes=("_joinquant", "_eastmoney"),
        indicator=True,
        validate="one_to_one",
    )
    both = merged["_merge"].eq("both")
    merged["unit_nav_absolute_error"] = (
        merged["unit_nav_joinquant"] - merged["unit_nav_eastmoney"]
    ).abs()
    cumulative_both = both & merged[["cumulative_nav_joinquant", "cumulative_nav_eastmoney"]].notna().all(axis=1)
    merged["cumulative_nav_absolute_error"] = np.nan
    merged.loc[cumulative_both, "cumulative_nav_absolute_error"] = (
        merged.loc[cumulative_both, "cumulative_nav_joinquant"]
        - merged.loc[cumulative_both, "cumulative_nav_eastmoney"]
    ).abs()
    tolerance = THRESHOLDS["nav_tolerance"]
    mismatch = merged[
        ~both
        | merged["unit_nav_absolute_error"].fillna(np.inf).gt(tolerance)
        | (cumulative_both & merged["cumulative_nav_absolute_error"].gt(tolerance))
    ].copy()
    summaries: list[dict[str, Any]] = []
    for asset, group in merged.groupby("asset", sort=True):
        asset_both = group["_merge"].eq("both")
        asset_cumulative = asset_both & group[["cumulative_nav_joinquant", "cumulative_nav_eastmoney"]].notna().all(axis=1)
        summaries.append(
            {
                "asset": asset,
                "joinquant_rows": int(len(group)),
                "overlap_rows": int(asset_both.sum()),
                "row_coverage": float(asset_both.mean()),
                "unit_nav_within_tolerance_ratio": float(
                    group.loc[asset_both, "unit_nav_absolute_error"].le(tolerance).mean()
                )
                if asset_both.any()
                else 0.0,
                "cumulative_nav_comparable_rows": int(asset_cumulative.sum()),
                "cumulative_nav_within_tolerance_ratio": float(
                    group.loc[asset_cumulative, "cumulative_nav_absolute_error"].le(tolerance).mean()
                )
                if asset_cumulative.any()
                else 0.0,
            }
        )
    asset_summary = pd.DataFrame(summaries)
    comparable_unit = merged.loc[both, "unit_nav_absolute_error"]
    comparable_cumulative = merged.loc[cumulative_both, "cumulative_nav_absolute_error"]
    joinquant_cumulative_rows = int(merged["cumulative_nav_joinquant"].notna().sum())
    totals = {
        "joinquant_assets": int(right["asset"].nunique()),
        "overlap_assets": int(merged.loc[both, "asset"].nunique()),
        "joinquant_rows": int(len(merged)),
        "overlap_rows": int(both.sum()),
        "row_coverage": float(both.mean()) if len(merged) else 0.0,
        "unit_nav_within_tolerance_ratio": float(comparable_unit.le(tolerance).mean()) if len(comparable_unit) else 0.0,
        "unit_nav_maximum_absolute_error": float(comparable_unit.max()) if len(comparable_unit) else None,
        "cumulative_nav_comparable_rows": int(cumulative_both.sum()),
        "joinquant_cumulative_nav_rows": joinquant_cumulative_rows,
        "cumulative_nav_comparable_coverage": (
            float(cumulative_both.sum() / joinquant_cumulative_rows) if joinquant_cumulative_rows else 0.0
        ),
        "cumulative_nav_within_tolerance_ratio": float(comparable_cumulative.le(tolerance).mean())
        if len(comparable_cumulative)
        else 0.0,
        "cumulative_nav_maximum_absolute_error": float(comparable_cumulative.max())
        if len(comparable_cumulative)
        else None,
    }
    return asset_summary, mismatch, totals


def assess_terminal_boundary(nav: pd.DataFrame, event: pd.Series) -> dict[str, Any]:
    asset_nav = nav[nav["asset"].astype(str).str.zfill(6).eq("511210")].copy()
    if asset_nav.empty:
        return {
            "asset_present": False,
            "boundary_passed": False,
            "last_nav_date": None,
            "last_unit_nav": None,
            "liquidation_start_date": str(event["liquidation_start_date"]),
        }
    asset_nav["date"] = pd.to_datetime(asset_nav["date"], errors="coerce").dt.normalize()
    last = asset_nav.sort_values("date").iloc[-1]
    last_date = pd.Timestamp(last["date"]).normalize()
    operation_date = pd.Timestamp(event["last_operation_date"]).normalize()
    liquidation_start = pd.Timestamp(event["liquidation_start_date"]).normalize()
    post_operation_rows = int(asset_nav["date"].gt(operation_date).sum())
    boundary_passed = bool(last_date == operation_date and last_date < liquidation_start and post_operation_rows == 0)
    return {
        "asset_present": True,
        "boundary_passed": boundary_passed,
        "last_nav_date": last_date.date().isoformat(),
        "last_unit_nav": float(last["unit_nav"]),
        "last_operation_date": operation_date.date().isoformat(),
        "liquidation_start_date": liquidation_start.date().isoformat(),
        "post_operation_nav_rows": post_operation_rows,
        "liquidation_nav": float(event["liquidation_nav"]),
        "cash_distribution_per_share": float(event["cash_per_share"]),
        "cash_distribution_minus_last_unit_nav": float(event["cash_per_share"]) - float(last["unit_nav"]),
    }


def _version_inventory(selected_assets: set[str]) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(EASTMONEY_RUN_ROOT.glob("*")):
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
            hashes = sorted(
                str(item.get("uncompressed_sha256", "")) for item in metadata.get("source_responses", [])
            )
            rows.append(
                {
                    "asset": asset,
                    "run_id": manifest.get("run_id"),
                    "observed_date": observed_date,
                    "source_response_set_sha256": hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest(),
                }
            )
    columns = ["asset", "run_count", "distinct_observation_dates", "distinct_source_versions", "version_depth_ready"]
    if not rows:
        return pd.DataFrame(columns=columns), 0.0
    details = pd.DataFrame(rows)
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
    return inventory, float(inventory["version_depth_ready"].mean()) if len(inventory) else 0.0


def _check(name: str, value: Any, operator: str, threshold: Any, passed: bool, note: str) -> dict[str, Any]:
    return {
        "check": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
        "note": note,
    }


def validate(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    manifest, immutable_path, nav, status, inventory = load_authenticated_eastmoney()
    joinquant_manifest, joinquant_nav = load_authenticated_joinquant()
    terminal_manifest, terminal_event = load_authenticated_terminal_event()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["list_date"] = pd.to_datetime(nav["list_date"], errors="coerce").dt.normalize()
    nav["delist_date"] = pd.to_datetime(nav["delist_date"], errors="coerce").dt.normalize()
    nav["available_date"] = pd.to_datetime(nav["available_date"], errors="coerce").dt.normalize()
    if nav[["date", "asset", "unit_nav", "list_date", "available_date"]].isna().any(axis=None):
        raise ValueError("Eastmoney NAV contains missing required values")
    if (pd.to_numeric(nav["unit_nav"], errors="coerce") <= 0).any():
        raise ValueError("Eastmoney NAV contains non-positive unit NAV")
    cumulative_numeric = pd.to_numeric(nav["cumulative_nav"], errors="coerce")
    if (cumulative_numeric.dropna() <= 0).any():
        raise ValueError("Eastmoney NAV contains non-positive cumulative NAV")
    lifecycle_valid = nav["date"].ge(nav["list_date"]) & (
        nav["delist_date"].isna() | nav["date"].le(nav["delist_date"])
    )
    availability_date = pd.Timestamp(str(manifest["created_at"])[:10])
    current_final_flags_valid = bool(
        nav["available_date"].eq(availability_date).all()
        and ~nav["pit_actionable"].astype(bool).any()
        and ~nav["historical_backtest_allowed"].astype(bool).any()
        and ~nav["model_promotion_allowed"].astype(bool).any()
    )
    cumulative_missing = int(cumulative_numeric.isna().sum())
    unit_missing_source_rows_dropped = int(
        pd.to_numeric(status["unit_nav_missing_source_rows_dropped"]).sum()
    )
    unit_missing_disclosed = (
        unit_missing_source_rows_dropped
        == int(manifest.get("unit_nav_missing_source_rows_dropped", -1))
    )
    missing_disclosed = bool(
        cumulative_missing == int(manifest.get("cumulative_nav_missing_rows", -1))
        and cumulative_missing == int(pd.to_numeric(status["cumulative_nav_missing_rows"]).sum())
    )
    nav_asset_summary, cross_source_mismatches, comparison = compare_nav_sources(nav, joinquant_nav)
    terminal = assess_terminal_boundary(nav, terminal_event)
    selected_assets = set(status["asset"])
    versions, version_depth_coverage = _version_inventory(selected_assets)
    ready = status[status["status"].eq("ready")]
    delisted = status[status["delist_date"].notna()]
    ready_ratio = len(ready) / len(status) if len(status) else 0.0
    delisted_ready_ratio = float(delisted["status"].eq("ready").mean()) if len(delisted) else 0.0
    full_master_scope = len(status) == THRESHOLDS["full_master_assets"]
    full_delisted_scope = int(status["delist_date"].notna().sum()) == THRESHOLDS["full_delisted_assets"]
    joinquant_asset_coverage = (
        comparison["overlap_assets"] / comparison["joinquant_assets"] if comparison["joinquant_assets"] else 0.0
    )
    checks = [
        _check("collector_errors", int(status["status"].eq("error").sum()), "==", 0, not status["status"].eq("error").any(), "selected assets"),
        _check("ready_asset_coverage", ready_ratio, ">=", THRESHOLDS["ready_asset_coverage_min"], ready_ratio >= THRESHOLDS["ready_asset_coverage_min"], "selected assets"),
        _check("delisted_population", int(status["delist_date"].notna().sum()), "==", THRESHOLDS["full_delisted_assets"], full_delisted_scope, "governed delisted ETF population"),
        _check("delisted_ready_coverage", delisted_ready_ratio, ">=", THRESHOLDS["delisted_ready_coverage_min"], delisted_ready_ratio >= THRESHOLDS["delisted_ready_coverage_min"], "delisted ETFs"),
        _check("lifecycle_boundaries", int(lifecycle_valid.sum()), "==", len(nav), lifecycle_valid.all(), "NAV rows stay inside list/delist boundaries"),
        _check("current_final_flags", current_final_flags_valid, "==", True, current_final_flags_valid, "availability uses collection date"),
        _check("unit_nav_missing_source_rows_disclosed", unit_missing_source_rows_dropped, "==", int(manifest.get("unit_nav_missing_source_rows_dropped", -1)), unit_missing_disclosed, "unusable source rows dropped without filling"),
        _check("cumulative_nav_missing_rows_disclosed", cumulative_missing, "==", int(manifest.get("cumulative_nav_missing_rows", -1)), missing_disclosed, "missing values retained, never filled"),
        _check("joinquant_nav_population", comparison["joinquant_assets"], "==", THRESHOLDS["joinquant_nav_assets"], comparison["joinquant_assets"] == THRESHOLDS["joinquant_nav_assets"], "authenticated limited-window source"),
        _check("joinquant_asset_coverage", joinquant_asset_coverage, ">=", THRESHOLDS["joinquant_asset_coverage_min"], joinquant_asset_coverage >= THRESHOLDS["joinquant_asset_coverage_min"], "assets with at least one overlap"),
        _check("joinquant_row_coverage", comparison["row_coverage"], ">=", THRESHOLDS["joinquant_row_coverage_min"], comparison["row_coverage"] >= THRESHOLDS["joinquant_row_coverage_min"], "recent independent NAV rows"),
        _check("unit_nav_within_tolerance", comparison["unit_nav_within_tolerance_ratio"], ">=", THRESHOLDS["unit_nav_within_tolerance_min"], comparison["unit_nav_within_tolerance_ratio"] >= THRESHOLDS["unit_nav_within_tolerance_min"], "overlapping rows"),
        _check("cumulative_nav_comparable_coverage", comparison["cumulative_nav_comparable_coverage"], ">=", THRESHOLDS["cumulative_nav_comparable_coverage_min"], comparison["cumulative_nav_comparable_coverage"] >= THRESHOLDS["cumulative_nav_comparable_coverage_min"], "JoinQuant rows with cumulative NAV available in both sources"),
        _check("cumulative_nav_within_tolerance", comparison["cumulative_nav_within_tolerance_ratio"], ">=", THRESHOLDS["cumulative_nav_within_tolerance_min"], comparison["cumulative_nav_within_tolerance_ratio"] >= THRESHOLDS["cumulative_nav_within_tolerance_min"], "comparable overlapping rows"),
        _check("terminal_event_separation_511210", terminal["boundary_passed"], "==", True, terminal["boundary_passed"], "regular NAV stops before separately authenticated liquidation distribution"),
        _check("full_master_scope", len(status), "==", THRESHOLDS["full_master_assets"], full_master_scope, "required for full-lifecycle qualification"),
        _check("source_version_monitoring_depth", version_depth_coverage, "==", 1.0, version_depth_coverage == 1.0, "two collection dates monitor revisions but do not retroactively create PIT history"),
    ]
    checks_frame = pd.DataFrame(checks)
    version_check = "source_version_monitoring_depth"
    full_market_current_final_source_passed = bool(
        checks_frame.loc[~checks_frame["check"].eq(version_check), "passed"].all()
    )
    delisted_integrity_checks = {
        "collector_errors",
        "ready_asset_coverage",
        "delisted_population",
        "delisted_ready_coverage",
        "lifecycle_boundaries",
        "current_final_flags",
        "unit_nav_missing_source_rows_disclosed",
        "cumulative_nav_missing_rows_disclosed",
        "terminal_event_separation_511210",
    }
    delisted_integrity_passed = bool(
        checks_frame.loc[checks_frame["check"].isin(delisted_integrity_checks), "passed"].all()
        and full_delisted_scope
    )
    if full_market_current_final_source_passed:
        qualification = "PASS_FULL_LIFECYCLE_NAV_CURRENT_FINAL_VERSION_DEPTH_BLOCKED"
    elif delisted_integrity_passed:
        qualification = "PASS_DELISTED_NAV_CURRENT_FINAL_FULL_MARKET_PENDING"
    else:
        qualification = "BLOCKED_NAV_SOURCE_QUALITY_OR_COVERAGE"
    summary = {
        "as_of_date": manifest.get("as_of_date"),
        "qualification_status": qualification,
        "source_run_id": manifest.get("run_id"),
        "selected_assets": int(len(status)),
        "selected_delisted_assets": int(status["delist_date"].notna().sum()),
        "ready_assets": int(len(ready)),
        "no_data_assets": int(status["status"].eq("no_data").sum()),
        "error_assets": int(status["status"].eq("error").sum()),
        "nav_rows": int(len(nav)),
        "nav_assets": int(nav["asset"].nunique()),
        "coverage_start": nav["date"].min().date().isoformat() if len(nav) else None,
        "coverage_end": nav["date"].max().date().isoformat() if len(nav) else None,
        "unit_nav_missing_source_rows_dropped": unit_missing_source_rows_dropped,
        "cumulative_nav_missing_rows": cumulative_missing,
        **comparison,
        "terminal_event_boundary": terminal,
        "full_market_current_final_source_passed": full_market_current_final_source_passed,
        "delisted_integrity_passed": delisted_integrity_passed,
        "version_depth_coverage": version_depth_coverage,
        "historical_source_qualified": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "formal_table_promotion_allowed": False,
        "boundary": "current-final independent NAV history; NAV cannot replace execution prices or historical available-date evidence",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_summary_path = output_dir / "nav_asset_summary.csv"
    mismatch_path = output_dir / "nav_cross_source_mismatches.csv.gz"
    checks_path = output_dir / "qualification_checks.csv"
    versions_path = output_dir / "source_version_inventory.csv"
    terminal_path = output_dir / "terminal_event_boundary.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "ETF_EASTMONEY_NAV_AUDIT.md"
    _atomic_csv(nav_asset_summary, asset_summary_path)
    _atomic_csv(cross_source_mismatches, mismatch_path, compressed=True)
    _atomic_csv(checks_frame, checks_path)
    _atomic_csv(versions, versions_path)
    _atomic_json(terminal, terminal_path)
    _atomic_json(summary, summary_path)
    report_lines = [
        "# ETF Eastmoney NAV Audit",
        "",
        f"Qualification: `{qualification}`",
        "",
        f"- Selected / ready: {len(status):,} / {len(ready):,}",
        f"- Delisted selected / ready: {len(delisted):,} / {int(delisted['status'].eq('ready').sum()):,}",
        f"- NAV rows / assets: {len(nav):,} / {nav['asset'].nunique():,}",
        f"- Missing unit-NAV source rows dropped: {unit_missing_source_rows_dropped:,}",
        f"- Missing cumulative NAV rows retained: {cumulative_missing:,}",
        f"- JoinQuant rows / overlap: {comparison['joinquant_rows']:,} / {comparison['overlap_rows']:,}",
        f"- Unit NAV within tolerance: {comparison['unit_nav_within_tolerance_ratio']:.6%}",
        f"- Cumulative NAV comparable coverage: {comparison['cumulative_nav_comparable_coverage']:.6%}",
        f"- Cumulative NAV within tolerance: {comparison['cumulative_nav_within_tolerance_ratio']:.6%}",
        f"- Source-version monitoring depth: {version_depth_coverage:.6%}",
        "",
        "## 511210 Boundary",
        "",
        f"Regular NAV ends on `{terminal.get('last_nav_date')}` at `{terminal.get('last_unit_nav')}`. The separately authenticated liquidation distribution is `{terminal.get('cash_distribution_per_share')}` per share; no synthetic NAV or market bar is created for that payment.",
        "",
        "## Evidence Boundary",
        "",
        "These are current-final observations collected after the fact. Cross-source agreement validates content, not historical availability. Every historical row keeps the collection date as available_date and remains barred from historical backtests and model promotion.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    code_path = Path(__file__).resolve()
    outputs = [
        {"role": "nav_asset_summary", "path": _relative(asset_summary_path), "sha256": _sha256(asset_summary_path), "rows": len(nav_asset_summary)},
        {"role": "nav_cross_source_mismatches", "path": _relative(mismatch_path), "sha256": _sha256(mismatch_path), "rows": len(cross_source_mismatches)},
        {"role": "qualification_checks", "path": _relative(checks_path), "sha256": _sha256(checks_path), "rows": len(checks_frame)},
        {"role": "source_version_inventory", "path": _relative(versions_path), "sha256": _sha256(versions_path), "rows": len(versions)},
        {"role": "terminal_event_boundary", "path": _relative(terminal_path), "sha256": _sha256(terminal_path), "rows": 1},
        {"role": "summary", "path": _relative(summary_path), "sha256": _sha256(summary_path), "rows": 1},
        {"role": "report", "path": _relative(report_path), "sha256": _sha256(report_path), "rows": len(report_lines)},
    ]
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        **summary,
        "thresholds": THRESHOLDS,
        "inputs": [
            {"role": "eastmoney_latest_manifest", "path": _relative(EASTMONEY_MANIFEST_PATH), "sha256": _sha256(EASTMONEY_MANIFEST_PATH)},
            {"role": "eastmoney_immutable_manifest", "path": _relative(immutable_path), "sha256": _sha256(immutable_path)},
            {"role": "joinquant_manifest", "path": _relative(JOINQUANT_MANIFEST_PATH), "sha256": _sha256(JOINQUANT_MANIFEST_PATH)},
            {"role": "terminal_event_manifest", "path": _relative(TERMINAL_MANIFEST_PATH), "sha256": _sha256(TERMINAL_MANIFEST_PATH)},
        ],
        "outputs": outputs,
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "joinquant_run_id": joinquant_manifest.get("run_id"),
        "terminal_event_created_at": terminal_manifest.get("created_at"),
        "raw_response_inventory_rows": int(len(inventory)),
    }
    _atomic_json(run_manifest, output_dir / "run_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
