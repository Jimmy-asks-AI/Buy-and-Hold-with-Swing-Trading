"""Persist stock and ETF snapshot parts without overwriting each other."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
COMBINED_PATH = RAW_ROOT / "research_snapshot.csv"
PART_PATHS = {
    "stock": RAW_ROOT / "stock_research_snapshot.csv",
    "etf": RAW_ROOT / "etf_research_snapshot.csv",
}


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"asset": str})
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    return frame


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def write_snapshot_part(asset_type: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Write one asset-type part and atomically rebuild the combined snapshot."""
    if asset_type not in PART_PATHS:
        raise ValueError(f"unsupported snapshot asset type: {asset_type}")
    if not frame.empty:
        required = {"asset", "asset_type"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"snapshot part missing columns: {missing}")
        if not frame["asset_type"].astype(str).str.lower().eq(asset_type).all():
            raise ValueError(f"snapshot part contains rows other than {asset_type}")
        if frame["asset"].astype(str).duplicated().any():
            raise ValueError(f"duplicate assets in {asset_type} snapshot part")

    current = frame.copy()
    if not current.empty:
        current["asset"] = current["asset"].astype(str).str.zfill(6)
    _write_atomic(current, PART_PATHS[asset_type])

    old_combined = _read(COMBINED_PATH) if COMBINED_PATH.exists() else pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for kind, path in PART_PATHS.items():
        if path.exists():
            part = _read(path)
        elif not old_combined.empty and "asset_type" in old_combined.columns:
            part = old_combined[old_combined["asset_type"].astype(str).str.lower().eq(kind)].copy()
            if not part.empty:
                _write_atomic(part, path)
        else:
            part = pd.DataFrame()
        if not part.empty:
            parts.append(part)

    combined = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if not combined.empty:
        if combined["asset"].astype(str).duplicated().any():
            duplicates = sorted(combined.loc[combined["asset"].astype(str).duplicated(False), "asset"].astype(str).unique())
            raise ValueError(f"duplicate assets across snapshot parts: {duplicates}")
        sort_columns = [column for column in ["asset_type", "sector", "asset"] if column in combined.columns]
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    _write_atomic(combined, COMBINED_PATH)
    return combined
