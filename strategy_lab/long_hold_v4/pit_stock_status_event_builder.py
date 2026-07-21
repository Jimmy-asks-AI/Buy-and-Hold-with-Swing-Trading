"""Build execution-relevant stock status events from governed name history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_stock_name_history_collector import classify_security_name


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
NAME_EVENT_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "stock_name_history_progress.csv"
)
NAME_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_name_history_probe_latest.json"
SSE_SUPPLEMENT_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "sse_official_status_events.csv"
)
SSE_SUPPLEMENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_status_announcement_collector_latest.json"
)
SSE_META_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_name_history" / "sse"
OUTPUT_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "stock_execution_status_events.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_status_event_builder_latest.json"
OUTPUT_COLUMNS = [
    "asset",
    "effective_date",
    "execution_status",
    "is_st",
    "available_date",
    "source_coverage_mode",
    "data_source",
    "source_vintage",
]
EXECUTION_STATUSES = {
    "normal",
    "risk_warning",
    "delisting",
    "special_transfer",
    "listing_suspended",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def _execution_status(value: Any) -> str:
    classified = str(value)
    if classified in {"normal", "listing_marker"}:
        return "normal"
    return classified if classified in EXECUTION_STATUSES else "unknown"


def _load_lifecycles(as_of: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(MASTER_PATH, dtype={"asset": str, "predecessor_asset": str})
    listed = frame[frame["event_type"].eq("listing")][
        ["asset", "asset_name", "exchange", "list_date", "predecessor_asset"]
    ].copy()
    exits = frame[frame["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    output = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce")
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce")
    output = output[output["list_date"].le(as_of)].copy()
    if output[["asset", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("stock lifecycle master has incomplete status-event keys")
    return output.sort_values("asset").reset_index(drop=True)


def _coverage_modes(
    lifecycles: pd.DataFrame,
    name_manifest: dict[str, Any],
    supplement_manifest: dict[str, Any] | None = None,
) -> dict[str, str]:
    coverage: dict[str, str] = {}
    if name_manifest.get("szse_official_workbook_ready") is True:
        coverage.update(
            {str(asset): "official_dated_history" for asset in lifecycles.loc[lifecycles["exchange"].eq("SZSE"), "asset"]}
        )
    for asset in lifecycles.loc[lifecycles["exchange"].eq("SSE"), "asset"]:
        meta_path = SSE_META_DIR / f"{asset}.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if meta.get("status") == "success" and meta.get("coverage_mode") in {
            "dated_history",
            "no_name_changes",
            "undated_non_status_names",
        }:
            coverage[str(asset)] = str(meta["coverage_mode"])
    if supplement_manifest:
        for asset in supplement_manifest.get("resolved_assets", []):
            coverage[str(asset).zfill(6)] = "official_sse_announcement_status"
    return coverage


def official_status_events_as_name_events(official: pd.DataFrame) -> pd.DataFrame:
    if official.empty:
        return pd.DataFrame(columns=["asset", "effective_date", "old_status", "new_status"])
    frame = official.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.normalize()
    if frame[["asset", "effective_date", "execution_status"]].isna().any(axis=None):
        raise ValueError("official SSE status supplement contains incomplete event keys")
    if frame.duplicated(["asset", "effective_date"]).any():
        raise ValueError("official SSE status supplement contains duplicate asset dates")
    rows: list[dict[str, Any]] = []
    for asset, group in frame.groupby("asset", sort=False):
        current = "normal"
        for event in group.sort_values("effective_date").itertuples(index=False):
            new_status = _execution_status(event.execution_status)
            if new_status == "unknown":
                raise ValueError(f"official SSE status supplement has unknown status for {asset}")
            if new_status == current:
                continue
            rows.append(
                {
                    "asset": asset,
                    "effective_date": pd.Timestamp(event.effective_date).normalize(),
                    "old_status": current,
                    "new_status": new_status,
                }
            )
            current = new_status
    return pd.DataFrame(rows, columns=["asset", "effective_date", "old_status", "new_status"])


def build_status_events(
    lifecycles: pd.DataFrame,
    name_events: pd.DataFrame,
    coverage_modes: dict[str, str],
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, str]]:
    as_of_date = pd.Timestamp(as_of).normalize()
    events = name_events.copy()
    if not events.empty:
        events["asset"] = events["asset"].astype(str).str.zfill(6)
        events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
        if events[["asset", "effective_date", "old_status", "new_status"]].isna().any(axis=None):
            raise ValueError("name-history events contain incomplete status keys")
        if events.duplicated(["asset", "effective_date"]).any():
            raise ValueError("name-history events contain duplicate asset dates")
        grouped = {asset: group.sort_values("effective_date") for asset, group in events.groupby("asset", sort=False)}
    else:
        grouped = {}

    rows: list[dict[str, Any]] = []
    unresolved: dict[str, str] = {}
    for lifecycle in lifecycles.itertuples(index=False):
        asset = str(lifecycle.asset).zfill(6)
        coverage_mode = coverage_modes.get(asset)
        if not coverage_mode:
            unresolved[asset] = "missing governed name-history source"
            continue
        start = pd.Timestamp(lifecycle.list_date).normalize()
        end = min(pd.Timestamp(lifecycle.delist_date).normalize(), as_of_date) if pd.notna(lifecycle.delist_date) else as_of_date
        asset_events = grouped.get(asset, pd.DataFrame()).copy()
        if not asset_events.empty:
            asset_events = asset_events[asset_events["effective_date"].between(start, end)].copy()

        if asset_events.empty:
            current = _execution_status(classify_security_name(lifecycle.asset_name))
            if current != "normal":
                unresolved[asset] = f"non-normal current name without a dated transition: {current}"
                continue
        else:
            first = asset_events.iloc[0]
            if first["effective_date"] == start:
                current = _execution_status(first["new_status"])
            else:
                current = _execution_status(first["old_status"])
            if current == "unknown":
                unresolved[asset] = "earliest name event does not resolve listing-date execution status"
                continue

        asset_rows = [
            {
                "asset": asset,
                "effective_date": start,
                "execution_status": current,
                "is_st": current in {"risk_warning", "special_transfer"},
                "available_date": start,
                "source_coverage_mode": coverage_mode,
            }
        ]
        asset_failed = False
        for event in asset_events.itertuples(index=False):
            new_status = _execution_status(event.new_status)
            if new_status == "unknown":
                unresolved[asset] = f"unknown status at {pd.Timestamp(event.effective_date).date()}"
                asset_failed = True
                break
            effective = pd.Timestamp(event.effective_date).normalize()
            if effective == start:
                asset_rows[0]["execution_status"] = new_status
                asset_rows[0]["is_st"] = new_status in {"risk_warning", "special_transfer"}
                current = new_status
                continue
            if new_status == current:
                continue
            asset_rows.append(
                {
                    "asset": asset,
                    "effective_date": effective,
                    "execution_status": new_status,
                    "is_st": new_status in {"risk_warning", "special_transfer"},
                    "available_date": effective,
                    "source_coverage_mode": coverage_mode,
                }
            )
            current = new_status
        if not asset_failed:
            rows.extend(asset_rows)

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=OUTPUT_COLUMNS[:-2])
    output = output.sort_values(["asset", "effective_date"]).reset_index(drop=True)
    if output.duplicated(["asset", "effective_date"]).any():
        raise ValueError("status-event output contains duplicate asset dates")
    return output, unresolved


def run_builder(as_of: str) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    name_manifest = json.loads(NAME_MANIFEST_PATH.read_text(encoding="utf-8"))
    if name_manifest.get("as_of_date") != as_of_date.date().isoformat():
        raise ValueError("name-history manifest as-of date mismatch")
    if name_manifest.get("output_path") != _relative(NAME_EVENT_PATH):
        raise ValueError("name-history manifest output path mismatch")
    if _sha256(NAME_EVENT_PATH) != name_manifest.get("output_sha256"):
        raise ValueError("name-history observation hash mismatch")
    supplement_manifest: dict[str, Any] | None = None
    supplement_events = pd.DataFrame()
    if SSE_SUPPLEMENT_PATH.is_file() and SSE_SUPPLEMENT_MANIFEST_PATH.is_file():
        supplement_manifest = json.loads(SSE_SUPPLEMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
        if supplement_manifest.get("as_of_date") != as_of_date.date().isoformat():
            raise ValueError("SSE status supplement as-of date mismatch")
        if supplement_manifest.get("output_path") != _relative(SSE_SUPPLEMENT_PATH):
            raise ValueError("SSE status supplement output path mismatch")
        if _sha256(SSE_SUPPLEMENT_PATH) != supplement_manifest.get("output_sha256"):
            raise ValueError("SSE status supplement output hash mismatch")
        supplement_events = pd.read_csv(SSE_SUPPLEMENT_PATH, dtype={"asset": str})
    lifecycles = _load_lifecycles(as_of_date)
    name_events = pd.read_csv(NAME_EVENT_PATH, dtype={"asset": str})
    official_name_events = official_status_events_as_name_events(supplement_events)
    if not official_name_events.empty:
        name_events = pd.concat([name_events, official_name_events], ignore_index=True)
    coverage_modes = _coverage_modes(lifecycles, name_manifest, supplement_manifest)
    events, unresolved = build_status_events(lifecycles, name_events, coverage_modes, as_of_date)

    inputs = [
        {"role": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"role": "stock_name_history", "path": _relative(NAME_EVENT_PATH), "sha256": _sha256(NAME_EVENT_PATH)},
        {"role": "stock_name_history_manifest", "path": _relative(NAME_MANIFEST_PATH), "sha256": _sha256(NAME_MANIFEST_PATH)},
    ]
    if supplement_manifest is not None:
        inputs.extend(
            [
                {"role": "sse_official_status_events", "path": _relative(SSE_SUPPLEMENT_PATH), "sha256": _sha256(SSE_SUPPLEMENT_PATH)},
                {
                    "role": "sse_official_status_manifest",
                    "path": _relative(SSE_SUPPLEMENT_MANIFEST_PATH),
                    "sha256": _sha256(SSE_SUPPLEMENT_MANIFEST_PATH),
                },
            ]
        )
    bundle_hash = hashlib.sha256(
        "|".join(f"{item['role']}:{item['sha256']}" for item in inputs).encode()
    ).hexdigest()
    events["data_source"] = "governed_security_name_status_events"
    events["source_vintage"] = f"stock_status_event_bundle_sha256:{bundle_hash}"
    events = events[OUTPUT_COLUMNS]
    _atomic_csv(events, OUTPUT_PATH)
    covered_assets = int(events["asset"].nunique()) if not events.empty else 0
    target_assets = int(len(lifecycles))
    ready = bool(covered_assets == target_assets and not unresolved and len(coverage_modes) == target_assets)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "source_vintage": f"stock_status_event_bundle_sha256:{bundle_hash}",
        "output_path": _relative(OUTPUT_PATH),
        "output_sha256": _sha256(OUTPUT_PATH),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(events)),
        "assets": covered_assets,
        "target_assets": target_assets,
        "asset_coverage": round(covered_assets / target_assets, 6) if target_assets else 0.0,
        "status_counts": events["execution_status"].value_counts().sort_index().to_dict(),
        "source_coverage_counts": events.drop_duplicates("asset")["source_coverage_mode"].value_counts().sort_index().to_dict(),
        "unresolved_assets": unresolved,
        "qualification_status": "READY_FOR_CROSS_SOURCE_VALIDATION" if ready else "COLLECTION_IN_PROGRESS",
        "qualification_blockers": [
            "status events have not passed historical BaoStock and JoinQuant day-level validation",
            "SSE status evidence remains secondary-source data",
        ],
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_builder(args.as_of)
    keys = ("qualification_status", "rows", "assets", "target_assets", "asset_coverage", "historical_backtest_allowed")
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
