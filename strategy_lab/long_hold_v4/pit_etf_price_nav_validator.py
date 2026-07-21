"""Audit ETF raw prices, NAV, source versions, and independent agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
JQ_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "joinquant_etf_price_nav_validation_latest.json"
)
LIFECYCLE_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
)
LIFECYCLE_RUN_DIR = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_runs"
NAV_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_source_manifest_20260717.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_price_nav"
PRICE_CHECKS_PATH = OUTPUT_DIR / "price_cross_source_checks.csv.gz"
PRICE_ASSET_PATH = OUTPUT_DIR / "price_asset_summary.csv"
NAV_CHECKS_PATH = OUTPUT_DIR / "nav_cross_source_checks.csv.gz"
NAV_ASSET_PATH = OUTPUT_DIR / "nav_asset_summary.csv"
VERSION_PATH = OUTPUT_DIR / "source_version_inventory.csv"
CHECKS_PATH = OUTPUT_DIR / "qualification_checks.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "ETF_PRICE_NAV_AUDIT.md"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

THRESHOLDS = {
    "recent_price_asset_coverage_min": 0.995,
    "recent_price_row_coverage_min": 0.999,
    "recent_price_ohlc_match_min": 0.9999,
    "recent_price_activity_match_min": 0.999,
    "recent_nav_row_coverage_min": 0.999,
    "recent_nav_value_match_min": 0.9999,
    "full_history_start_max": "2005-02-23",
    "full_delisted_assets_min": 123,
    "source_versions_per_asset_min": 2,
    "source_version_asset_coverage_min": 0.95,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if gzip else None,
    )
    temporary.replace(path)


def _authenticate_declared_file(item: dict[str, Any], role: str) -> Path:
    path = _resolve(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError(f"ETF price/NAV audit input failed hash authentication: {role}")
    return path


def _authenticate_joinquant_manifest() -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = json.loads(JQ_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "LIMITED_WINDOW_INDEPENDENT_VALIDATION_SOURCE"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
    ):
        raise ValueError("JoinQuant ETF validation source violates its observation-only contract")
    code_path = _resolve(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("JoinQuant ETF validation collector code hash mismatch")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    required = ("joinquant_etf_raw_prices", "joinquant_etf_raw_nav", "joinquant_etf_asset_status")
    if any(role not in outputs for role in required):
        raise ValueError("JoinQuant ETF validation manifest is missing required outputs")
    return manifest, {role: _authenticate_declared_file(outputs[role], role) for role in required}


def _authenticate_lifecycle_manifest() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(LIFECYCLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "COLLECTION_IN_PROGRESS_CURRENT_FINAL_SNAPSHOT"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
    ):
        raise ValueError("ETF lifecycle source is not the governed observation-only build")
    for item in manifest.get("code_files", []):
        _authenticate_declared_file(item, "lifecycle_code")
    for item in manifest.get("outputs", []):
        _authenticate_declared_file(item, str(item.get("role", "lifecycle_output")))
    prices: dict[str, dict[str, Any]] = {}
    for item in manifest.get("inputs", []):
        if item.get("role") == "etf_raw_price" and item.get("asset") and item.get("sha256"):
            asset = str(item["asset"]).zfill(6)
            if asset in prices:
                raise ValueError(f"ETF lifecycle manifest has duplicate raw price declarations: {asset}")
            prices[asset] = item
    if len(prices) != int(manifest.get("selected_assets", -1)):
        raise ValueError("ETF lifecycle manifest raw price population is incomplete")
    return manifest, prices


def _authenticate_nav_manifest(nav_assets: Iterable[str]) -> dict[str, Path]:
    manifest = json.loads(NAV_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("ETF NAV source manifest is not observation-only")
    for item in manifest.get("code_files", []):
        _authenticate_declared_file(item, "nav_source_code")
    requested = {str(asset).zfill(6) for asset in nav_assets}
    paths: dict[str, Path] = {}
    for item in manifest.get("input_files", []):
        normalised = str(item.get("path", "")).replace("\\", "/")
        if "/nav/" not in normalised:
            continue
        asset = Path(normalised).stem.zfill(6)
        if asset in requested:
            paths[asset] = _authenticate_declared_file(item, f"eastmoney_nav:{asset}")
    missing = sorted(requested.difference(paths))
    if missing:
        raise ValueError(f"ETF NAV source manifest is missing requested assets: {missing}")
    return paths


def compare_price_panels(joinquant: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    jq = joinquant.copy()
    sina = source.copy()
    jq["date"] = pd.to_datetime(jq["date"], errors="coerce").dt.normalize()
    sina["date"] = pd.to_datetime(sina["date"], errors="coerce").dt.normalize()
    jq["asset"] = jq["asset"].astype(str).str.zfill(6)
    sina["asset"] = sina["asset"].astype(str).str.zfill(6)
    jq["paused"] = pd.to_numeric(jq["paused"], errors="coerce").fillna(0)
    market = ["open", "high", "low", "close", "volume", "amount"]
    jq[market] = jq[market].apply(pd.to_numeric, errors="coerce")
    sina[market] = sina[market].apply(pd.to_numeric, errors="coerce")
    # With fill_paused=False, JoinQuant can still emit a carried OHLC row with
    # zero volume and amount. It is a no-trade marker, not an exchange print.
    traded = jq["paused"].eq(0) & jq["volume"].gt(0) & jq["amount"].gt(0)
    jq = jq[traded].copy()
    checks = jq[["date", "asset", *market]].merge(
        sina[["date", "asset", *market]],
        on=["date", "asset"],
        how="left",
        suffixes=("_joinquant", "_sina"),
        validate="one_to_one",
    )
    checks["source_row_present"] = checks["close_sina"].notna()
    price_flags: list[str] = []
    for column in ("open", "high", "low", "close"):
        error = (checks[f"{column}_sina"] - checks[f"{column}_joinquant"]).abs()
        tolerance = np.maximum(0.0011, checks[f"{column}_joinquant"].abs() * 5e-5)
        checks[f"{column}_absolute_error"] = error
        flag = f"{column}_match"
        checks[flag] = checks["source_row_present"] & error.le(tolerance)
        price_flags.append(flag)
    checks["ohlc_match"] = checks[price_flags].all(axis=1)
    volume_error = (checks["volume_sina"] - checks["volume_joinquant"]).abs()
    volume_tolerance = np.maximum(1.0, checks["volume_joinquant"].abs() * 1e-8)
    checks["volume_absolute_error"] = volume_error
    checks["volume_match"] = checks["source_row_present"] & volume_error.le(volume_tolerance)
    amount_error = (checks["amount_sina"] - checks["amount_joinquant"]).abs()
    amount_tolerance = np.maximum(1.0, checks["amount_joinquant"].abs() * 1e-6)
    checks["amount_absolute_error"] = amount_error
    checks["amount_match"] = checks["source_row_present"] & amount_error.le(amount_tolerance)
    checks["activity_match"] = checks["volume_match"] & checks["amount_match"]
    return checks.sort_values(["asset", "date"]).reset_index(drop=True)


def compare_nav_panels(joinquant: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    jq = joinquant.copy()
    east = source.copy()
    jq["date"] = pd.to_datetime(jq["date"], errors="coerce").dt.normalize()
    east["date"] = pd.to_datetime(east["date"], errors="coerce").dt.normalize()
    jq["asset"] = jq["asset"].astype(str).str.zfill(6)
    east["asset"] = east["asset"].astype(str).str.zfill(6)
    columns = ["unit_nav", "cumulative_nav"]
    jq[columns] = jq[columns].apply(pd.to_numeric, errors="coerce")
    east[columns] = east[columns].apply(pd.to_numeric, errors="coerce")
    checks = jq[["date", "asset", *columns]].merge(
        east[["date", "asset", *columns]],
        on=["date", "asset"],
        how="left",
        suffixes=("_joinquant", "_eastmoney"),
        validate="one_to_one",
    )
    checks["source_row_present"] = checks[[f"{column}_eastmoney" for column in columns]].notna().any(axis=1)
    flags: list[str] = []
    for column in columns:
        comparable = checks[f"{column}_joinquant"].notna()
        source_present = checks[f"{column}_eastmoney"].notna()
        error = (checks[f"{column}_eastmoney"] - checks[f"{column}_joinquant"]).abs()
        checks[f"{column}_comparable"] = comparable
        checks[f"{column}_absolute_error"] = error
        flag = f"{column}_match"
        checks[flag] = ~comparable | (source_present & error.le(0.00011))
        flags.append(flag)
    checks["nav_match"] = checks[flags].all(axis=1)
    return checks.sort_values(["asset", "date"]).reset_index(drop=True)


def build_version_inventory(manifest_paths: Iterable[Path]) -> pd.DataFrame:
    versions: dict[str, set[str]] = defaultdict(set)
    run_ids: dict[str, set[str]] = defaultdict(set)
    created: dict[str, list[str]] = defaultdict(list)
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id", path.stem))
        created_at = str(manifest.get("created_at", ""))
        for item in manifest.get("inputs", []):
            if item.get("role") != "etf_raw_price" or not item.get("asset") or not item.get("sha256"):
                continue
            asset = str(item["asset"]).zfill(6)
            versions[asset].add(str(item["sha256"]))
            run_ids[asset].add(run_id)
            created[asset].append(created_at)
    rows = [
        {
            "asset": asset,
            "declared_run_count": len(run_ids[asset]),
            "distinct_source_price_versions": len(hashes),
            "first_manifest_created_at": min(created[asset]),
            "last_manifest_created_at": max(created[asset]),
            "multiple_source_versions_available": len(hashes) >= THRESHOLDS["source_versions_per_asset_min"],
            "source_hashes_json": json.dumps(sorted(hashes), separators=(",", ":")),
        }
        for asset, hashes in versions.items()
    ]
    return pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)


def _asset_summary(checks: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "price":
        return checks.groupby("asset", as_index=False).agg(
            joinquant_rows=("date", "size"),
            matched_source_rows=("source_row_present", "sum"),
            ohlc_match_rows=("ohlc_match", "sum"),
            activity_match_rows=("activity_match", "sum"),
            coverage_start=("date", "min"),
            coverage_end=("date", "max"),
            maximum_close_absolute_error=("close_absolute_error", "max"),
            maximum_amount_absolute_error=("amount_absolute_error", "max"),
        )
    return checks.groupby("asset", as_index=False).agg(
        joinquant_rows=("date", "size"),
        matched_source_rows=("source_row_present", "sum"),
        nav_match_rows=("nav_match", "sum"),
        coverage_start=("date", "min"),
        coverage_end=("date", "max"),
        maximum_unit_nav_absolute_error=("unit_nav_absolute_error", "max"),
        maximum_cumulative_nav_absolute_error=("cumulative_nav_absolute_error", "max"),
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate(as_of: str | pd.Timestamp) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    jq_manifest, jq_paths = _authenticate_joinquant_manifest()
    lifecycle_manifest, lifecycle_price_records = _authenticate_lifecycle_manifest()
    jq_prices = pd.read_csv(jq_paths["joinquant_etf_raw_prices"], dtype={"asset": str}, low_memory=False)
    jq_nav = pd.read_csv(jq_paths["joinquant_etf_raw_nav"], dtype={"asset": str}, low_memory=False)
    jq_status = pd.read_csv(jq_paths["joinquant_etf_asset_status"], dtype={"asset": str}, low_memory=False)
    jq_prices["asset"] = jq_prices["asset"].astype(str).str.zfill(6)
    jq_nav["asset"] = jq_nav["asset"].astype(str).str.zfill(6)

    source_frames: list[pd.DataFrame] = []
    for asset in sorted(jq_prices["asset"].unique()):
        item = lifecycle_price_records.get(asset)
        if item is None:
            raise ValueError(f"ETF lifecycle source omitted JoinQuant validation asset: {asset}")
        path = _authenticate_declared_file(item, f"sina_raw_price:{asset}")
        frame = pd.read_csv(path, compression="gzip", low_memory=False)
        frame["asset"] = asset
        source_frames.append(frame)
    source_prices = pd.concat(source_frames, ignore_index=True)
    jq_paused_rows = int(pd.to_numeric(jq_prices["paused"], errors="coerce").fillna(0).eq(1).sum())
    jq_zero_activity_rows = int(
        (
            pd.to_numeric(jq_prices["paused"], errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(jq_prices["volume"], errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(jq_prices["amount"], errors="coerce").fillna(0).eq(0)
        ).sum()
    )
    price_checks = compare_price_panels(jq_prices, source_prices)
    price_asset = _asset_summary(price_checks, "price")

    nav_paths = _authenticate_nav_manifest(jq_nav["asset"].unique())
    nav_frames: list[pd.DataFrame] = []
    for asset, path in sorted(nav_paths.items()):
        frame = pd.read_csv(path, low_memory=False)
        frame["asset"] = asset
        nav_frames.append(frame)
    eastmoney_nav = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()
    nav_checks = compare_nav_panels(jq_nav, eastmoney_nav)
    nav_asset = _asset_summary(nav_checks, "nav")

    archive_manifests = sorted(LIFECYCLE_RUN_DIR.glob("*.json"))
    version_inventory = build_version_inventory(archive_manifests)
    selected_assets = int(jq_manifest["selected_assets"])
    recent_price_asset_coverage = _ratio(jq_prices["asset"].nunique(), selected_assets)
    price_row_coverage = float(price_checks["source_row_present"].mean())
    ohlc_match = float(price_checks["ohlc_match"].mean())
    activity_match = float(price_checks["activity_match"].mean())
    nav_row_coverage = float(nav_checks["source_row_present"].mean()) if not nav_checks.empty else 0.0
    nav_match = float(nav_checks["nav_match"].mean()) if not nav_checks.empty else 0.0
    version_depth_coverage = float(version_inventory["multiple_source_versions_available"].mean())
    validation_start = pd.Timestamp(jq_prices["date"].min()).normalize()
    recent_delisted_with_rows = int(
        jq_status[jq_status["lifecycle_status"].eq("delisted") & jq_status["price_rows"].gt(0)]["asset"].nunique()
    )

    check_rows = [
        ("recent_price_asset_coverage", recent_price_asset_coverage >= THRESHOLDS["recent_price_asset_coverage_min"], recent_price_asset_coverage, THRESHOLDS["recent_price_asset_coverage_min"]),
        ("recent_price_row_coverage", price_row_coverage >= THRESHOLDS["recent_price_row_coverage_min"], price_row_coverage, THRESHOLDS["recent_price_row_coverage_min"]),
        ("recent_price_ohlc_match", ohlc_match >= THRESHOLDS["recent_price_ohlc_match_min"], ohlc_match, THRESHOLDS["recent_price_ohlc_match_min"]),
        ("recent_price_activity_match", activity_match >= THRESHOLDS["recent_price_activity_match_min"], activity_match, THRESHOLDS["recent_price_activity_match_min"]),
        ("recent_nav_row_coverage", nav_row_coverage >= THRESHOLDS["recent_nav_row_coverage_min"], nav_row_coverage, THRESHOLDS["recent_nav_row_coverage_min"]),
        ("recent_nav_value_match", nav_match >= THRESHOLDS["recent_nav_value_match_min"], nav_match, THRESHOLDS["recent_nav_value_match_min"]),
        ("full_history_independent_start", validation_start <= pd.Timestamp(THRESHOLDS["full_history_start_max"]), validation_start.date().isoformat(), THRESHOLDS["full_history_start_max"]),
        ("full_delisted_independent_coverage", recent_delisted_with_rows >= THRESHOLDS["full_delisted_assets_min"], recent_delisted_with_rows, THRESHOLDS["full_delisted_assets_min"]),
        ("source_version_depth", version_depth_coverage >= THRESHOLDS["source_version_asset_coverage_min"], version_depth_coverage, THRESHOLDS["source_version_asset_coverage_min"]),
        ("historical_available_date_evidence", False, "collection-date availability only", "source evidence available on or before each historical trade date"),
    ]
    checks = pd.DataFrame(check_rows, columns=["check", "passed", "observed", "threshold"])
    recent_cross_source_passed = bool(checks.iloc[:6]["passed"].all())
    formal_promotion_allowed = False
    qualification_status = (
        "PASS_RECENT_CROSS_SOURCE_FULL_HISTORY_BLOCKED"
        if recent_cross_source_passed
        else "FAIL_CROSS_SOURCE_VALIDATION"
    )
    summary = {
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": qualification_status,
        "recent_cross_source_passed": recent_cross_source_passed,
        "formal_table_promotion_allowed": formal_promotion_allowed,
        "selected_assets": selected_assets,
        "joinquant_price_assets_with_rows": int(jq_prices["asset"].nunique()),
        "joinquant_price_rows": int(len(jq_prices)),
        "joinquant_paused_rows_excluded": jq_paused_rows,
        "joinquant_zero_activity_rows_excluded": jq_zero_activity_rows,
        "joinquant_traded_rows_compared": int(len(price_checks)),
        "recent_price_asset_coverage": recent_price_asset_coverage,
        "price_matched_rows": int(price_checks["source_row_present"].sum()),
        "price_row_coverage": price_row_coverage,
        "ohlc_match_ratio": ohlc_match,
        "activity_match_ratio": activity_match,
        "maximum_close_absolute_error": float(price_checks["close_absolute_error"].max()),
        "joinquant_nav_assets": int(jq_nav["asset"].nunique()),
        "joinquant_nav_rows": int(len(jq_nav)),
        "nav_matched_rows": int(nav_checks["source_row_present"].sum()),
        "nav_row_coverage": nav_row_coverage,
        "nav_match_ratio": nav_match,
        "maximum_unit_nav_absolute_error": float(nav_checks["unit_nav_absolute_error"].max()),
        "maximum_cumulative_nav_absolute_error": float(nav_checks["cumulative_nav_absolute_error"].max()),
        "independent_coverage_start": validation_start.date().isoformat(),
        "independent_coverage_end": pd.Timestamp(jq_prices["date"].max()).date().isoformat(),
        "recent_delisted_assets_with_rows": recent_delisted_with_rows,
        "lifecycle_delisted_assets": int(lifecycle_manifest["selected_delisted_assets"]),
        "lifecycle_assets": int(lifecycle_manifest["selected_assets"]),
        "archived_lifecycle_run_manifests": int(len(archive_manifests)),
        "version_inventory_assets": int(len(version_inventory)),
        "assets_with_multiple_source_price_versions": int(version_inventory["multiple_source_versions_available"].sum()),
        "source_version_depth_coverage": version_depth_coverage,
        "historical_available_date_evidence_passed": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "recent independent agreement passed or failed separately from full-history source qualification",
    }

    _atomic_csv(price_checks, PRICE_CHECKS_PATH, gzip=True)
    _atomic_csv(price_asset, PRICE_ASSET_PATH)
    _atomic_csv(nav_checks, NAV_CHECKS_PATH, gzip=True)
    _atomic_csv(nav_asset, NAV_ASSET_PATH)
    _atomic_csv(version_inventory, VERSION_PATH)
    _atomic_csv(checks, CHECKS_PATH)
    _atomic_json(summary, SUMMARY_PATH)
    report = "\n".join(
        [
            "# ETF Price/NAV Source Audit",
            "",
            f"As of: {summary['as_of_date']}",
            f"Qualification: `{qualification_status}`",
            f"Formal promotion allowed: `{str(formal_promotion_allowed).lower()}`",
            "",
            "## Recent independent comparison",
            "",
            f"- Price assets: {summary['joinquant_price_assets_with_rows']}/{selected_assets}",
            f"- Price rows: {summary['price_matched_rows']}/{summary['joinquant_price_rows']}",
            f"- Compared traded rows: {summary['joinquant_traded_rows_compared']} (excluded {jq_zero_activity_rows} zero-activity markers)",
            f"- OHLC match ratio: {ohlc_match:.8%}",
            f"- Volume/amount match ratio: {activity_match:.8%}",
            f"- NAV assets/rows: {summary['joinquant_nav_assets']}/{summary['joinquant_nav_rows']}",
            f"- NAV row coverage: {nav_row_coverage:.8%}",
            f"- NAV value match ratio: {nav_match:.8%}",
            "",
            "## Full-history blockers",
            "",
            f"- Independent source begins {summary['independent_coverage_start']}, not 2005.",
            f"- Independent delisted coverage is {recent_delisted_with_rows}/{summary['lifecycle_delisted_assets']} assets.",
            f"- {summary['assets_with_multiple_source_price_versions']}/{summary['version_inventory_assets']} assets have two source price versions.",
            "- A recent-window match cannot authenticate old delisted histories or prove that the current-final source was never revised.",
            "- Repeated current-final snapshots cannot establish what was available on historical trade dates.",
            "- The audited panel remains validation evidence only; it is not a backtest input.",
            "",
        ]
    )
    _atomic_text(report, REPORT_PATH)
    input_items = [
        {"role": "joinquant_validation_manifest", "path": _relative(JQ_MANIFEST_PATH), "sha256": _sha256(JQ_MANIFEST_PATH)},
        {"role": "lifecycle_observation_manifest", "path": _relative(LIFECYCLE_MANIFEST_PATH), "sha256": _sha256(LIFECYCLE_MANIFEST_PATH)},
        {"role": "nav_source_manifest", "path": _relative(NAV_MANIFEST_PATH), "sha256": _sha256(NAV_MANIFEST_PATH)},
        *[
            {"role": "lifecycle_run_manifest", "path": _relative(path), "sha256": _sha256(path)}
            for path in archive_manifests
        ],
    ]
    output_paths = [
        ("price_checks", PRICE_CHECKS_PATH),
        ("price_asset_summary", PRICE_ASSET_PATH),
        ("nav_checks", NAV_CHECKS_PATH),
        ("nav_asset_summary", NAV_ASSET_PATH),
        ("source_version_inventory", VERSION_PATH),
        ("qualification_checks", CHECKS_PATH),
        ("summary", SUMMARY_PATH),
        ("report", REPORT_PATH),
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **summary,
        "thresholds": THRESHOLDS,
        "inputs": input_items,
        "outputs": [
            {"role": role, "path": _relative(path), "sha256": _sha256(path)} for role, path in output_paths
        ],
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.as_of)
    keys = (
        "qualification_status",
        "recent_cross_source_passed",
        "formal_table_promotion_allowed",
        "joinquant_price_assets_with_rows",
        "joinquant_price_rows",
        "ohlc_match_ratio",
        "joinquant_nav_assets",
        "nav_match_ratio",
        "assets_with_multiple_source_price_versions",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
