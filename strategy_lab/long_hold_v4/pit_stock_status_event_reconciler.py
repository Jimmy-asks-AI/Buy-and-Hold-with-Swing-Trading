"""Reconcile stock execution-status events with market-wide SSE evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_stock_market_history_builder import ROOT, _sha256
from .pit_stock_name_history_collector import classify_security_name
from .pit_stock_status_event_builder import EXECUTION_STATUSES, OUTPUT_COLUMNS


BASE_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "stock_execution_status_events.csv"
)
BASE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_status_event_builder_latest.json"
SSE_CANDIDATE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "sse_official_status_candidate_events.csv"
)
SSE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_status_full_collection_latest.json"
FACTBOOK_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "sse_factbook_restoration_events.csv"
)
FACTBOOK_REFERENCE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "sse_factbook_status_reference_events.csv"
)
FACTBOOK_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_factbook_status_latest.json"
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
OUTPUT_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "stock_execution_status_events_reconciled.csv"
)
RECONCILIATION_LOG_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "stock_status_event_reconciliation" / "asset_reconciliation_log.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_status_event_reconciler_latest.json"
RECONCILER_VERSION = "stock_status_event_reconciliation_v3"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    temporary.replace(path)


def _normalise_status(value: Any) -> str:
    status = str(value)
    if status == "listing_marker":
        return "normal"
    return status if status in EXECUTION_STATUSES else "unknown"


def _load_manifest(path: Path, output_path: Path, as_of: pd.Timestamp) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("as_of_date") != as_of.date().isoformat():
        raise ValueError(f"manifest as-of mismatch: {_relative(path)}")
    if manifest.get("output_path") != _relative(output_path):
        raise ValueError(f"manifest output path mismatch: {_relative(path)}")
    if manifest.get("output_sha256") != _sha256(output_path):
        raise ValueError(f"manifest output hash mismatch: {_relative(path)}")
    code_path = manifest.get("code_path")
    if code_path and manifest.get("code_sha256") != _sha256(ROOT / str(code_path)):
        raise ValueError(f"manifest code hash mismatch: {_relative(path)}")
    return manifest


def _normalise_base(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["effective_date"] = pd.to_datetime(output["effective_date"], errors="coerce").dt.normalize()
    output["available_date"] = pd.to_datetime(output["available_date"], errors="coerce").dt.normalize()
    output["execution_status"] = output["execution_status"].map(_normalise_status)
    output["evidence_priority"] = 0
    output["evidence_role"] = "base_governed_name_status"
    return output


def _normalise_official(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "evidence_priority", "evidence_role"])
    output = frame.copy()
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["effective_date"] = pd.to_datetime(output["effective_date"], errors="coerce").dt.normalize()
    output["available_date"] = pd.to_datetime(output["available_date"], errors="coerce").dt.normalize()
    output["execution_status"] = output["execution_status"].map(_normalise_status)
    output["is_st"] = output["execution_status"].isin({"risk_warning", "special_transfer"})
    output["source_coverage_mode"] = "official_sse_marketwide_announcement_candidate"
    output["evidence_priority"] = 1
    output["evidence_role"] = "official_sse_company_announcement"
    return output[[*OUTPUT_COLUMNS, "evidence_priority", "evidence_role"]]


def _normalise_factbook(
    frame: pd.DataFrame,
    *,
    evidence_priority: int,
    evidence_role: str,
    coverage_mode: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, "evidence_priority", "evidence_role"])
    output = frame.copy()
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["effective_date"] = pd.to_datetime(output["effective_date"], errors="coerce").dt.normalize()
    output["available_date"] = pd.to_datetime(output["available_date"], errors="coerce").dt.normalize()
    output["execution_status"] = output["execution_status"].map(_normalise_status)
    output["is_st"] = output["execution_status"].isin({"risk_warning", "special_transfer"})
    output["source_coverage_mode"] = coverage_mode
    output["evidence_priority"] = evidence_priority
    output["evidence_role"] = evidence_role
    return output[[*OUTPUT_COLUMNS, "evidence_priority", "evidence_role"]]


def select_factbook_reference_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"binary_state_change", "used_in_reconciliation", "execution_status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SSE factbook reference output missing columns: {missing}")
    binary_state_change = frame["binary_state_change"].astype(str).str.lower().isin({"true", "1"})
    used_in_reconciliation = frame["used_in_reconciliation"].astype(str).str.lower().isin({"true", "1"})
    # A factbook "termination" row is a legal lifecycle event, not evidence of
    # a tradable delisting period. Suspension is an execution state and remains
    # eligible; legal termination stays governed by the security master.
    execution_state = frame["execution_status"].eq("listing_suspended")
    return frame[~used_in_reconciliation & (binary_state_change | execution_state)].copy()


def reconcile_asset_events(
    base: pd.DataFrame,
    official: pd.DataFrame,
    factbook: pd.DataFrame | None = None,
    *,
    exchange: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if base.empty:
        raise ValueError("base status-event history cannot be empty")
    sources = [base.copy()]
    if exchange == "SSE" and not official.empty:
        sources.append(official.copy())
    if exchange == "SSE" and factbook is not None and not factbook.empty:
        sources.append(factbook.copy())
    combined = pd.concat(sources, ignore_index=True)
    combined = combined.sort_values(["effective_date", "evidence_priority", "available_date"])
    same_day_conflicts = int(combined.groupby("effective_date")["execution_status"].nunique().gt(1).sum())
    combined = combined.drop_duplicates("effective_date", keep="last")
    rows: list[dict[str, Any]] = []
    current: str | None = None
    for event in combined.itertuples(index=False):
        status = _normalise_status(event.execution_status)
        if status == "unknown":
            raise ValueError(f"unknown execution status for {event.asset} at {event.effective_date}")
        if status == current:
            continue
        row = event._asdict()
        row["execution_status"] = status
        row["is_st"] = status in {"risk_warning", "special_transfer"}
        rows.append(row)
        current = status
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("status reconciliation unexpectedly produced no events")
    return output, {
        "base_rows": int(len(base)),
        "official_candidate_rows": int(len(official)) if exchange == "SSE" else 0,
        "factbook_candidate_rows": int(len(factbook)) if exchange == "SSE" and factbook is not None else 0,
        "reconciled_rows": int(len(output)),
        "same_day_conflicts_resolved": same_day_conflicts,
        "official_rows_added_after_collapse": int(output["evidence_role"].eq("official_sse_company_announcement").sum()),
        "factbook_rows_added_after_collapse": int(
            output["evidence_role"].astype(str).str.startswith("official_sse_factbook").sum()
        ),
        "factbook_reference_rows_added_after_collapse": int(
            output["evidence_role"].eq("official_sse_factbook_status_reference").sum()
        ),
        "factbook_restoration_rows_added_after_collapse": int(
            output["evidence_role"].eq("official_sse_factbook_restoration").sum()
        ),
    }


def _expected_current_status(asset_name: Any) -> str:
    status = _normalise_status(classify_security_name(asset_name))
    return status


def run_reconciliation(as_of: str) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    base_manifest = _load_manifest(BASE_MANIFEST_PATH, BASE_PATH, as_of_date)
    sse_manifest = _load_manifest(SSE_MANIFEST_PATH, SSE_CANDIDATE_PATH, as_of_date)
    factbook_manifest = _load_manifest(FACTBOOK_MANIFEST_PATH, FACTBOOK_PATH, as_of_date)
    if sse_manifest.get("qualification_status") != "READY_FOR_RECONCILIATION":
        raise ValueError("market-wide SSE status collection is incomplete")
    if factbook_manifest.get("qualification_status") != "READY_FOR_RECONCILIATION":
        raise ValueError("SSE factbook status collection is incomplete")
    if factbook_manifest.get("reference_output_path") != _relative(FACTBOOK_REFERENCE_PATH):
        raise ValueError("SSE factbook reference output path mismatch")
    if factbook_manifest.get("reference_output_sha256") != _sha256(FACTBOOK_REFERENCE_PATH):
        raise ValueError("SSE factbook reference output hash mismatch")

    base = _normalise_base(pd.read_csv(BASE_PATH, dtype={"asset": str}))
    official = _normalise_official(pd.read_csv(SSE_CANDIDATE_PATH, dtype={"asset": str}))
    factbook_restoration = _normalise_factbook(
        pd.read_csv(FACTBOOK_PATH, dtype={"asset": str}),
        evidence_priority=3,
        evidence_role="official_sse_factbook_restoration",
        coverage_mode="official_sse_factbook_restoration_table",
    )
    factbook_reference_raw = select_factbook_reference_candidates(
        pd.read_csv(FACTBOOK_REFERENCE_PATH, dtype={"asset": str})
    )
    factbook_reference = _normalise_factbook(
        factbook_reference_raw,
        evidence_priority=2,
        evidence_role="official_sse_factbook_status_reference",
        coverage_mode="official_sse_factbook_status_table",
    )
    factbook = pd.concat([factbook_reference, factbook_restoration], ignore_index=True)
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    listings = master[master["event_type"].eq("listing")][
        ["asset", "asset_name", "exchange", "list_date", "delist_date"]
    ].copy()
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    listings = listings.drop(columns="delist_date").merge(exits, on="asset", how="left", validate="one_to_one")
    listings["asset"] = listings["asset"].astype(str).str.zfill(6)
    listings["list_date"] = pd.to_datetime(listings["list_date"], errors="coerce").dt.normalize()
    listings["delist_date"] = pd.to_datetime(listings["delist_date"], errors="coerce").dt.normalize()
    listings = listings[listings["list_date"].le(as_of_date)].sort_values("asset")
    base_groups = {asset: group.copy() for asset, group in base.groupby("asset", sort=False)}
    official_groups = {asset: group.copy() for asset, group in official.groupby("asset", sort=False)}
    factbook_groups = {asset: group.copy() for asset, group in factbook.groupby("asset", sort=False)}

    frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    unresolved: dict[str, str] = {}
    for lifecycle in listings.itertuples(index=False):
        asset = str(lifecycle.asset)
        asset_base = base_groups.get(asset, pd.DataFrame()).copy()
        if asset_base.empty:
            unresolved[asset] = "missing base governed status-event seed"
            continue
        asset_official = official_groups.get(asset, pd.DataFrame()).copy()
        asset_factbook = factbook_groups.get(asset, pd.DataFrame()).copy()
        try:
            reconciled, metrics = reconcile_asset_events(
                asset_base,
                asset_official,
                asset_factbook,
                exchange=str(lifecycle.exchange),
            )
        except ValueError as exc:
            unresolved[asset] = str(exc)
            continue
        lifecycle_end = min(pd.Timestamp(lifecycle.delist_date), as_of_date) if pd.notna(lifecycle.delist_date) else as_of_date
        reconciled = reconciled[
            reconciled["effective_date"].between(pd.Timestamp(lifecycle.list_date), lifecycle_end)
        ].copy()
        if reconciled.empty or reconciled.iloc[0]["effective_date"] != pd.Timestamp(lifecycle.list_date):
            unresolved[asset] = "reconciled history does not resolve listing-date status"
            continue
        final_status = str(reconciled.iloc[-1]["execution_status"])
        lifecycle_terminated = pd.notna(lifecycle.delist_date) and pd.Timestamp(lifecycle.delist_date) <= as_of_date
        expected_current = "lifecycle_terminated" if lifecycle_terminated else _expected_current_status(lifecycle.asset_name)
        current_match = (
            final_status in EXECUTION_STATUSES and final_status != "unknown"
            if lifecycle_terminated
            else expected_current != "unknown" and final_status == expected_current
        )
        if not current_match:
            unresolved[asset] = f"final status {final_status} does not match master-name status {expected_current}"
            continue
        if str(lifecycle.exchange) == "SSE":
            evidence = ["name_history"]
            if not asset_official.empty:
                evidence.append("marketwide_official_announcements")
            if not asset_factbook.empty:
                evidence.append("official_factbook_status_tables")
            mode = "sse_" + "_plus_".join(evidence)
            reconciled["source_coverage_mode"] = mode
        reconciled["available_date"] = pd.to_datetime(reconciled["available_date"]).dt.normalize()
        frames.append(reconciled)
        logs.append(
            {
                "asset": asset,
                "exchange": lifecycle.exchange,
                "expected_current_status": expected_current,
                "final_reconciled_status": final_status,
                "current_status_match": current_match,
                "source_coverage_mode": reconciled.iloc[-1]["source_coverage_mode"],
                **metrics,
            }
        )

    reconciled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    logs_frame = pd.DataFrame(logs).sort_values("asset").reset_index(drop=True)
    inputs = [
        {"role": "base_status_events", "path": _relative(BASE_PATH), "sha256": _sha256(BASE_PATH)},
        {"role": "base_status_manifest", "path": _relative(BASE_MANIFEST_PATH), "sha256": _sha256(BASE_MANIFEST_PATH)},
        {"role": "sse_official_candidates", "path": _relative(SSE_CANDIDATE_PATH), "sha256": _sha256(SSE_CANDIDATE_PATH)},
        {"role": "sse_candidate_manifest", "path": _relative(SSE_MANIFEST_PATH), "sha256": _sha256(SSE_MANIFEST_PATH)},
        {"role": "sse_factbook_restorations", "path": _relative(FACTBOOK_PATH), "sha256": _sha256(FACTBOOK_PATH)},
        {"role": "sse_factbook_status_tables", "path": _relative(FACTBOOK_REFERENCE_PATH), "sha256": _sha256(FACTBOOK_REFERENCE_PATH)},
        {"role": "sse_factbook_manifest", "path": _relative(FACTBOOK_MANIFEST_PATH), "sha256": _sha256(FACTBOOK_MANIFEST_PATH)},
        {"role": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
    ]
    bundle_hash = hashlib.sha256(
        "|".join(f"{item['role']}:{item['sha256']}" for item in inputs).encode()
    ).hexdigest()
    reconciled["data_source"] = "governed_exchange_status_event_reconciliation"
    reconciled["source_vintage"] = f"stock_status_reconciliation_bundle_sha256:{bundle_hash}"
    reconciled = reconciled[OUTPUT_COLUMNS].sort_values(["asset", "effective_date"]).reset_index(drop=True)
    if reconciled.duplicated(["asset", "effective_date"]).any():
        raise ValueError("reconciled status output contains duplicate asset dates")
    if (pd.to_datetime(reconciled["available_date"]) > pd.to_datetime(reconciled["effective_date"])).any():
        raise ValueError("reconciled status output contains future availability")
    _atomic_csv(reconciled, OUTPUT_PATH)
    _atomic_csv(logs_frame, RECONCILIATION_LOG_PATH)

    target_assets = int(len(listings))
    covered_assets = int(reconciled["asset"].nunique()) if not reconciled.empty else 0
    ready = covered_assets == target_assets and not unresolved
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reconciler_version": RECONCILER_VERSION,
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "source_vintage": f"stock_status_reconciliation_bundle_sha256:{bundle_hash}",
        "output_path": _relative(OUTPUT_PATH),
        "output_sha256": _sha256(OUTPUT_PATH),
        "reconciliation_log_path": _relative(RECONCILIATION_LOG_PATH),
        "reconciliation_log_sha256": _sha256(RECONCILIATION_LOG_PATH),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(reconciled)),
        "assets": covered_assets,
        "target_assets": target_assets,
        "asset_coverage": round(covered_assets / target_assets, 8) if target_assets else 0.0,
        "official_sse_candidate_rows": int(len(official)),
        "official_sse_rows_retained_after_state_collapse": int(
            logs_frame["official_rows_added_after_collapse"].sum()
        )
        if not logs_frame.empty
        else 0,
        "factbook_candidate_rows": int(len(factbook)),
        "factbook_reference_candidate_rows": int(len(factbook_reference)),
        "factbook_restoration_candidate_rows": int(len(factbook_restoration)),
        "factbook_rows_retained_after_state_collapse": int(
            logs_frame["factbook_rows_added_after_collapse"].sum()
        )
        if not logs_frame.empty
        else 0,
        "factbook_reference_rows_retained_after_state_collapse": int(
            logs_frame["factbook_reference_rows_added_after_collapse"].sum()
        )
        if not logs_frame.empty
        else 0,
        "factbook_restoration_rows_retained_after_state_collapse": int(
            logs_frame["factbook_restoration_rows_added_after_collapse"].sum()
        )
        if not logs_frame.empty
        else 0,
        "status_counts": reconciled["execution_status"].value_counts().sort_index().to_dict(),
        "source_coverage_counts": reconciled.drop_duplicates("asset")["source_coverage_mode"].value_counts().sort_index().to_dict(),
        "unresolved_assets": unresolved,
        "qualification_status": "READY_FOR_CROSS_SOURCE_VALIDATION" if ready else "RECONCILIATION_INCOMPLETE",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "qualification_blockers": [
            "reconciled events require independent daily validation",
        ],
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_reconciliation(args.as_of)
    keys = (
        "qualification_status",
        "rows",
        "assets",
        "target_assets",
        "official_sse_candidate_rows",
        "official_sse_rows_retained_after_state_collapse",
        "factbook_candidate_rows",
        "factbook_rows_retained_after_state_collapse",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
