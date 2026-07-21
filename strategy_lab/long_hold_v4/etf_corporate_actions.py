"""Governed ETF share-conversion evidence used by the observation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = ROOT / "configs" / "long_hold_v4_etf_corporate_actions.json"
REQUIRED_COLUMNS = (
    "asset",
    "action_type",
    "event_date",
    "price_effective_date",
    "shares_after_per_share_before",
    "announcement_date",
    "source_document_title",
    "source_url",
    "source_type",
    "review_status",
)


def load_corporate_action_registry(path: Path = DEFAULT_REGISTRY_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ETF corporate-action registry schema")
    if payload.get("factor_definition") != "shares_after_per_share_before":
        raise ValueError("ETF corporate-action registry has an unsupported factor definition")
    frame = pd.DataFrame(payload.get("actions", []))
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"ETF corporate-action registry is incomplete: {missing}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["asset"] = frame["asset"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    for column in ("event_date", "price_effective_date", "announcement_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["shares_after_per_share_before"] = pd.to_numeric(
        frame["shares_after_per_share_before"], errors="coerce"
    )
    if frame[["asset", "event_date", "price_effective_date", "announcement_date"]].isna().any(axis=None):
        raise ValueError("ETF corporate-action registry contains invalid keys or dates")
    factors = frame["shares_after_per_share_before"]
    if factors.isna().any() or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("ETF corporate-action registry contains invalid conversion factors")
    if frame.duplicated(["asset", "price_effective_date"]).any():
        raise ValueError("ETF corporate-action registry contains duplicate effective actions")
    if (~frame["review_status"].eq("verified")).any():
        raise ValueError("ETF corporate-action registry contains unverified actions")
    if (~frame["source_url"].astype(str).str.startswith("https://")).any():
        raise ValueError("ETF corporate-action registry requires HTTPS evidence URLs")
    if frame["event_date"].gt(frame["price_effective_date"]).any():
        raise ValueError("ETF corporate-action registry has an event after its price-effective date")
    return frame.sort_values(["asset", "price_effective_date"]).reset_index(drop=True)


def conversion_factors_for_asset(
    registry: pd.DataFrame,
    asset: Any,
    as_of: str | pd.Timestamp | None = None,
) -> dict[pd.Timestamp, float]:
    code = str(asset).strip()
    if code.endswith(".0"):
        code = code[:-2]
    code = code.zfill(6)
    selected = registry[registry["asset"].eq(code)].copy()
    if as_of is not None:
        as_of_date = pd.Timestamp(as_of).normalize()
        selected = selected[
            selected["price_effective_date"].le(as_of_date)
            & selected["announcement_date"].le(as_of_date)
        ]
    return {
        pd.Timestamp(row.price_effective_date).normalize(): float(row.shares_after_per_share_before)
        for row in selected.itertuples(index=False)
    }
