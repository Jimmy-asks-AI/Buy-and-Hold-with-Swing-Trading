"""Freeze positive ETF distribution markers into an immutable discovery queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_events_lifecycle_observation_latest.csv.gz"
LIFECYCLE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_dividend_evidence_queue.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_evidence_queue_latest.json"
QUEUE_COLUMNS = [
    "priority",
    "asset",
    "asset_name",
    "exchange",
    "lifecycle_status",
    "source_event_date",
    "inferred_cash_per_share",
    "cumulative_dividend",
    "source_observed_at",
    "discovery_source_vintage",
    "source_event_date_basis",
    "required_evidence",
    "review_status",
    "historical_backtest_allowed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def _manifest_authenticates_output(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def build_evidence_queue(observation: pd.DataFrame) -> pd.DataFrame:
    required = {
        "asset",
        "asset_name",
        "lifecycle_status",
        "event_date",
        "cumulative_dividend",
        "cash_distribution",
        "source_observed_at",
        "source_vintage",
        "event_type",
        "historical_backtest_allowed",
    }
    missing = sorted(required.difference(observation.columns))
    if missing:
        raise ValueError(f"ETF dividend lifecycle observation misses columns: {missing}")
    frame = observation.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    frame["cash_distribution"] = pd.to_numeric(frame["cash_distribution"], errors="coerce")
    frame["cumulative_dividend"] = pd.to_numeric(frame["cumulative_dividend"], errors="coerce")
    frame = frame[frame["event_type"].eq("cash_distribution") & frame["cash_distribution"].gt(0)].copy()
    if frame.empty or frame[["event_date", "cash_distribution", "cumulative_dividend"]].isna().any().any():
        raise ValueError("ETF dividend discovery queue has no valid positive distributions")
    if frame["historical_backtest_allowed"].astype(str).str.lower().ne("false").any():
        raise ValueError("ETF dividend discovery source must remain historical-use disabled")
    if frame.duplicated(["asset", "event_date"]).any():
        raise ValueError("ETF dividend discovery source has duplicate asset/event dates")
    exchange = np.where(
        frame["asset"].str.startswith("5"),
        "SSE",
        np.where(frame["asset"].str.startswith("1"), "SZSE", ""),
    )
    if (exchange == "").any():
        unknown = sorted(frame.loc[exchange == "", "asset"].unique())[:10]
        raise ValueError(f"ETF dividend discovery source has unknown exchange codes: {unknown}")
    output = pd.DataFrame(
        {
            "priority": "P0",
            "asset": frame["asset"],
            "asset_name": frame["asset_name"].astype(str),
            "exchange": exchange,
            "lifecycle_status": frame["lifecycle_status"].astype(str),
            "source_event_date": frame["event_date"],
            "inferred_cash_per_share": frame["cash_distribution"],
            "cumulative_dividend": frame["cumulative_dividend"],
            "source_observed_at": frame["source_observed_at"].astype(str),
            "discovery_source_vintage": frame["source_vintage"].astype(str),
            "source_event_date_basis": "sina_cumulative_distribution_marker",
            "required_evidence": "official announcement with announcement date, exchange ex-date, pay date, and cash per share",
            "review_status": "pending_primary_evidence",
            "historical_backtest_allowed": False,
        }
    ).reindex(columns=QUEUE_COLUMNS)
    if not np.isfinite(output["inferred_cash_per_share"].to_numpy(dtype=float)).all():
        raise ValueError("ETF dividend discovery queue has non-finite cash values")
    return output.sort_values(["asset", "source_event_date"]).reset_index(drop=True)


def run_builder() -> dict[str, Any]:
    lifecycle_manifest = json.loads(LIFECYCLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not _manifest_authenticates_output(lifecycle_manifest, OBSERVATION_PATH):
        raise ValueError("ETF lifecycle manifest does not authenticate the dividend observation")
    if lifecycle_manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("ETF lifecycle discovery source unexpectedly allows historical use")
    observation = pd.read_csv(OBSERVATION_PATH, dtype={"asset": str}, compression="gzip")
    queue = build_evidence_queue(observation)
    _atomic_csv(queue, QUEUE_PATH)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": [
            {"path": _relative(OBSERVATION_PATH), "sha256": _sha256(OBSERVATION_PATH)},
            {"path": _relative(LIFECYCLE_MANIFEST_PATH), "sha256": _sha256(LIFECYCLE_MANIFEST_PATH)},
        ],
        "outputs": [
            {"role": "etf_dividend_evidence_queue", "path": _relative(QUEUE_PATH), "sha256": _sha256(QUEUE_PATH), "rows": int(len(queue))}
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "queue_rows": int(len(queue)),
        "queue_assets": int(queue["asset"].nunique()),
        "coverage_start": pd.Timestamp(queue["source_event_date"].min()).date().isoformat(),
        "coverage_end": pd.Timestamp(queue["source_event_date"].max()).date().isoformat(),
        "qualification_status": "DISCOVERY_QUEUE_PRIMARY_EVIDENCE_REQUIRED",
        "current_final_snapshot": True,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "method_boundary": (
            "Sina cumulative distributions are used only to discover official documents. Event dates and cash amounts "
            "must be replaced and validated from exchange or CNInfo announcements before historical use."
        ),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    manifest = run_builder()
    keys = (
        "qualification_status",
        "queue_rows",
        "queue_assets",
        "coverage_start",
        "coverage_end",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
