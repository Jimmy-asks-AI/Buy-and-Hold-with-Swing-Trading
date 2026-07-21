"""Collect official SSE execution-status announcement candidates market-wide.

This collector does not promote the parsed announcements directly.  It builds
an immutable, resumable official evidence layer for every SSE A-share
lifecycle, which is reconciled with early name history and independent daily
status panels by a later builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .pit_sse_status_announcement_collector import (
    CALENDAR_PATH,
    KEYWORDS,
    MASTER_PATH,
    OUTPUT_COLUMNS,
    RAW_DIR,
    _cache_paths,
    _session,
    _sha256,
    _valid_cache,
    parse_asset_events,
    query_asset,
)
from .pit_stock_market_history_builder import ROOT


STATUS_PATH = RAW_DIR / "all_sse_collection_status.json"
INVENTORY_PATH = RAW_DIR / "all_sse_asset_inventory.csv"
OUTPUT_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "sse_official_status_candidate_events.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_status_full_collection_latest.json"
COLLECTOR_VERSION = "sse_marketwide_status_candidates_v1"


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


def target_sse_assets(master: pd.DataFrame, as_of: str | pd.Timestamp) -> list[str]:
    as_of_date = pd.Timestamp(as_of).normalize()
    frame = master.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce").dt.normalize()
    frame = frame[
        frame["event_type"].eq("listing")
        & frame["exchange"].eq("SSE")
        & frame["list_date"].le(as_of_date)
    ]
    return sorted(frame["asset"].dropna().unique())


def select_shard(assets: list[str], shard_count: int, shard_index: int) -> list[str]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be inside [0, shard_count)")
    return [asset for position, asset in enumerate(sorted(assets)) if position % shard_count == shard_index]


def _collect_pending(
    assets: list[str],
    as_of: pd.Timestamp,
    workers: int,
    sleep_seconds: float,
) -> tuple[dict[str, dict[str, Any]], float]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    thread_local = threading.local()

    def collect_one(asset: str) -> tuple[str, dict[str, Any]]:
        if not hasattr(thread_local, "session"):
            thread_local.session = _session()
        try:
            result = query_asset(thread_local.session, asset, as_of)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return asset, result
        except (OSError, ValueError, requests.RequestException) as exc:
            return asset, {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one, asset): asset for asset in assets}
        for position, future in enumerate(as_completed(futures), start=1):
            asset, result = future.result()
            results[asset] = result
            if position % 25 == 0 or position == len(futures):
                _atomic_json(
                    {
                        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "collector_version": COLLECTOR_VERSION,
                        "as_of_date": as_of.date().isoformat(),
                        "selected_assets": len(assets),
                        "completed_in_run": position,
                        "assets": results,
                    },
                    STATUS_PATH,
                )
    return results, time.perf_counter() - started


def build_candidate_output(as_of: pd.Timestamp, targets: list[str]) -> dict[str, Any]:
    calendar_frame = pd.read_csv(CALENDAR_PATH, usecols=["date"])
    calendar = pd.DatetimeIndex(
        pd.to_datetime(calendar_frame["date"], errors="coerce").dropna().sort_values().unique()
    )
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = [
        {"role": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"role": "trade_calendar", "path": _relative(CALENDAR_PATH), "sha256": _sha256(CALENDAR_PATH)},
    ]
    parse_exceptions: dict[str, list[dict[str, str]]] = {}
    for asset in targets:
        if not _valid_cache(asset):
            inventory_rows.append(
                {
                    "asset": asset,
                    "cache_status": "missing",
                    "candidate_events": 0,
                    "parse_exception_count": 0,
                    "first_event_date": pd.NaT,
                    "last_event_date": pd.NaT,
                }
            )
            continue
        data_path, meta_path = _cache_paths(asset)
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        events, unresolved = parse_asset_events(artifact, calendar, collapse_state_changes=False)
        if not events.empty:
            frames.append(events)
        if unresolved:
            parse_exceptions[asset] = unresolved
        inventory_rows.append(
            {
                "asset": asset,
                "cache_status": "ready",
                "candidate_events": int(len(events)),
                "parse_exception_count": int(len(unresolved)),
                "first_event_date": events["effective_date"].min() if not events.empty else pd.NaT,
                "last_event_date": events["effective_date"].max() if not events.empty else pd.NaT,
            }
        )
        inputs.extend(
            [
                {"role": "sse_status_artifact", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"role": "sse_status_metadata", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )

    inventory = pd.DataFrame(inventory_rows).sort_values("asset").reset_index(drop=True)
    _atomic_csv(inventory, INVENTORY_PATH)
    inputs.append({"role": "asset_inventory", "path": _relative(INVENTORY_PATH), "sha256": _sha256(INVENTORY_PATH)})
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS[:-2])
    bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['role']}:{item['path']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    output["data_source"] = "sse_official_company_announcements_marketwide_candidates"
    output["source_vintage"] = f"sse_marketwide_status_bundle_sha256:{bundle_hash}"
    output = output[OUTPUT_COLUMNS].sort_values(["asset", "effective_date", "execution_status"]).reset_index(drop=True)
    _atomic_csv(output, OUTPUT_PATH)

    ready_assets = int(inventory["cache_status"].eq("ready").sum())
    complete = ready_assets == len(targets)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "collector_version": COLLECTOR_VERSION,
        "query_keywords": list(KEYWORDS),
        "as_of_date": as_of.date().isoformat(),
        "target_assets": len(targets),
        "cache_ready_assets": ready_assets,
        "cache_coverage": round(ready_assets / len(targets), 8) if targets else 0.0,
        "assets_with_candidate_events": int(output["asset"].nunique()) if not output.empty else 0,
        "candidate_event_rows": int(len(output)),
        "parse_exception_assets": parse_exceptions,
        "parse_exception_count": int(sum(len(rows) for rows in parse_exceptions.values())),
        "inventory_path": _relative(INVENTORY_PATH),
        "inventory_sha256": _sha256(INVENTORY_PATH),
        "output_path": _relative(OUTPUT_PATH),
        "output_sha256": _sha256(OUTPUT_PATH),
        "source_vintage": f"sse_marketwide_status_bundle_sha256:{bundle_hash}",
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "inputs": inputs,
        "qualification_status": "READY_FOR_RECONCILIATION" if complete else "COLLECTION_IN_PROGRESS",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "method_boundary": (
            "classified announcement candidates are preserved without assuming an initial state; "
            "a separate reconciler must combine them with early official factbooks/name history and independent daily status"
        ),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def run_collection(
    as_of: str,
    *,
    collect_limit: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    workers: int = 4,
    sleep_seconds: float = 0.05,
    collect_only: bool = False,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    targets = target_sse_assets(master, as_of_date)
    shard_targets = select_shard(targets, shard_count, shard_index)
    pending = [asset for asset in shard_targets if not _valid_cache(asset)]
    selected = pending[: max(0, collect_limit)] if collect_limit is not None else pending
    results: dict[str, dict[str, Any]] = {}
    elapsed = 0.0
    if selected:
        results, elapsed = _collect_pending(selected, as_of_date, workers, sleep_seconds)
    collection_summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "collector_version": COLLECTOR_VERSION,
        "as_of_date": as_of_date.date().isoformat(),
        "target_assets": len(targets),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "shard_assets": len(shard_targets),
        "pending_before": len(pending),
        "selected_assets": len(selected),
        "successful_in_run": sum(result.get("status") == "success" for result in results.values()),
        "failed_in_run": sum(result.get("status") != "success" for result in results.values()),
        "elapsed_seconds": round(elapsed, 3),
        "assets": results,
    }
    _atomic_json(collection_summary, STATUS_PATH)
    if collect_only:
        return {"collection": collection_summary, "build_outputs_skipped": True}
    manifest = build_candidate_output(as_of_date, targets)
    manifest["collection_run"] = {
        key: collection_summary[key]
        for key in (
            "shard_count",
            "shard_index",
            "shard_assets",
            "pending_before",
            "selected_assets",
            "successful_in_run",
            "failed_in_run",
            "elapsed_seconds",
        )
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--collect-limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_collection(
        args.as_of,
        collect_limit=args.collect_limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        workers=args.workers,
        sleep_seconds=args.sleep_seconds,
        collect_only=args.collect_only,
    )
    if result.get("build_outputs_skipped"):
        print(json.dumps(result, ensure_ascii=False))
        return
    keys = (
        "qualification_status",
        "target_assets",
        "cache_ready_assets",
        "cache_coverage",
        "assets_with_candidate_events",
        "candidate_event_rows",
        "parse_exception_count",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
