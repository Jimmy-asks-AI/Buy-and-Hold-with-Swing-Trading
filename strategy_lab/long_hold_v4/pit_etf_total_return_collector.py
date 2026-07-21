"""Collect and quarantine full-lifecycle ETF total-return observations.

The provider exposes current final historical snapshots.  This module fixes the
current-universe survivorship problem by selecting from the governed all-status
ETF master, but it does not turn the source into point-in-time data.  Every
output therefore remains observation-only until distribution and corporate-
action events have been independently validated with effective dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .etf_corporate_actions import (
    DEFAULT_REGISTRY_PATH,
    conversion_factors_for_asset,
    load_corporate_action_registry,
)
from .etf_snapshot_builder import fetch_dividends, fetch_price, total_return_adjusted_prices


ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = ROOT / "data_raw" / "long_hold_v4" / "pit_history"
DEFAULT_MASTER = PIT_ROOT / "etf_security_master.csv"
DEFAULT_CACHE_DIR = PIT_ROOT / "raw_etf_total_return"
DEFAULT_OBSERVATION_DIR = PIT_ROOT / "observations"
DEFAULT_PRICE_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_total_return_prices_lifecycle_observation_latest.csv.gz"
DEFAULT_DIVIDEND_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_dividend_events_lifecycle_observation_latest.csv.gz"
DEFAULT_STATUS_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_total_return_lifecycle_observation_latest_status.csv"
DEFAULT_MANIFEST = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
)
DEFAULT_ARCHIVE_DIR = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_runs"
DEFAULT_TERMINAL_EVENT_PATH = PIT_ROOT / "etf_terminal_cash_events.csv"
DEFAULT_TERMINAL_EVENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_terminal_cash_events_builder_latest.json"
)

QUALIFICATION_STATUS = "COLLECTION_IN_PROGRESS_CURRENT_FINAL_SNAPSHOT"
SOURCE_NAME = "Sina ETF OHLC and cumulative distributions via AkShare"
REQUIRED_PRICE_COLUMNS = {"date", "open", "high", "low", "close", "volume", "amount"}
REQUIRED_DIVIDEND_COLUMNS = {"date", "cumulative_dividend"}
REQUIRED_TERMINAL_EVENT_COLUMNS = {
    "asset",
    "event_type",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "termination_date",
    "extinguishes_position",
    "available_date",
    "source_vintage",
    "historical_backtest_allowed",
    "model_promotion_allowed",
}
REQUIRED_TERMINAL_EVENT_V2_COLUMNS = {
    "event_id",
    "asset",
    "event_type",
    "distribution_sequence",
    "holder_scope",
    "announcement_date",
    "available_trade_date",
    "available_date",
    "entitlement_date",
    "record_date",
    "pay_date",
    "accounting_date",
    "cash_per_share",
    "is_final_distribution",
    "additional_distribution_expected",
    "extinguishes_position",
    "source_pdf_sha256_set",
    "source_vintage",
    "historical_backtest_allowed",
    "model_promotion_allowed",
    "validation_status",
}
TERMINAL_EVENT_V2_REQUIRED_DATE_COLUMNS = (
    "announcement_date",
    "available_trade_date",
    "available_date",
    "entitlement_date",
    "record_date",
    "pay_date",
    "accounting_date",
)
TERMINAL_EVENT_V2_OPTIONAL_DATE_COLUMNS = ("ex_date", "termination_date")
TERMINAL_EVENT_DATE_COLUMNS = (
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "termination_date",
    "available_date",
)

PRICE_COLUMNS = [
    "date",
    "asset",
    "asset_name",
    "asset_type",
    "exchange",
    "lifecycle_status",
    "list_date",
    "delist_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "raw_close",
    "source_cash_distribution",
    "cash_distribution",
    "share_adjustment_factor",
    "adjustment_factor",
    "return_basis",
    "market_available_date",
    "source_observed_at",
    "available_date",
    "pit_actionable",
    "data_source",
    "source_vintage",
    "qualification_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

DIVIDEND_COLUMNS = [
    "event_date",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "asset",
    "asset_name",
    "asset_type",
    "lifecycle_status",
    "list_date",
    "delist_date",
    "cumulative_dividend",
    "cash_distribution",
    "event_type",
    "extinguishes_position",
    "source_observed_at",
    "available_date",
    "pit_actionable",
    "data_source",
    "source_vintage",
    "qualification_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

STATUS_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "selection_group",
    "lifecycle_status",
    "list_date",
    "delist_date",
    "collection_status",
    "collection_action",
    "build_status",
    "error",
    "price_rows",
    "dividend_event_rows",
    "registered_corporate_actions",
    "applied_corporate_actions",
    "governed_corporate_actions",
    "inferred_corporate_actions",
    "corporate_action_evidence_status",
    "corporate_action_evidence_detail_json",
    "coverage_start",
    "coverage_end",
    "listing_start_gap_days",
    "delisting_end_gap_days",
    "provider_tail_duplicate_rows_removed",
    "provider_tail_duplicate_detail_json",
    "maximum_distribution_alignment_lag_days",
    "distribution_alignment_detail_json",
    "terminal_cash_events",
    "terminal_cash_evidence_status",
    "terminal_cash_evidence_detail_json",
    "source_observed_at",
    "qualification_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]


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


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression=compression,
    )
    temporary.replace(path)


def _normalise_asset(value: Any) -> str:
    asset = str(value).strip()
    if asset.endswith(".0"):
        asset = asset[:-2]
    if not asset.isdigit() or len(asset) > 6:
        raise ValueError(f"invalid ETF asset code: {value}")
    return asset.zfill(6)


def _resolve_declared_path(value: Any) -> Path:
    declared = Path(str(value))
    return declared.resolve() if declared.is_absolute() else (ROOT / declared).resolve()


def _strict_boolean(series: pd.Series, column: str) -> pd.Series:
    normalised = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalised).difference({"true", "false"}))
    if unknown:
        raise ValueError(f"ETF terminal cash-event registry has invalid {column}: {unknown}")
    return normalised.eq("true")


def load_terminal_cash_event_registry(
    event_path: Path = DEFAULT_TERMINAL_EVENT_PATH,
    manifest_path: Path = DEFAULT_TERMINAL_EVENT_MANIFEST_PATH,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Authenticate and load the formally promoted known terminal events."""

    if not event_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("formal ETF terminal cash-event table or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification = str(manifest.get("qualification_status", ""))
    is_v2 = qualification == "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2"
    if (
        qualification
        not in {"PROMOTED_KNOWN_TERMINAL_CASH_EVENT", "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2"}
        or manifest.get("historical_backtest_allowed") is not True
        or manifest.get("model_promotion_allowed") is not False
        or manifest.get("scope_complete") is not False
    ):
        raise ValueError("ETF terminal cash-event manifest has not passed formal known-exception promotion")

    promoter_path = _resolve_declared_path(manifest.get("code_path", ""))
    if not promoter_path.is_file() or _sha256(promoter_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF terminal cash-event promoter code hash mismatch")
    outputs = [
        item for item in manifest.get("outputs", []) if item.get("role") == "pit_etf_terminal_cash_events"
    ]
    if len(outputs) != 1:
        raise ValueError("ETF terminal cash-event manifest has an invalid formal output declaration")
    declared_output = _resolve_declared_path(outputs[0].get("path", ""))
    if declared_output != event_path.resolve() or _sha256(event_path) != str(outputs[0].get("sha256", "")):
        raise ValueError("ETF terminal cash-event table path or hash mismatch")

    frame = pd.read_csv(event_path, dtype={"asset": str}, low_memory=False)
    required_columns = REQUIRED_TERMINAL_EVENT_V2_COLUMNS if is_v2 else REQUIRED_TERMINAL_EVENT_COLUMNS
    missing = sorted(required_columns.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"ETF terminal cash-event registry is incomplete: {missing}")
    if int(manifest.get("rows", -1)) != len(frame) or int(outputs[0].get("rows", -1)) != len(frame):
        raise ValueError("ETF terminal cash-event row count does not match its manifest")

    frame = frame.copy()
    frame["asset"] = frame["asset"].map(_normalise_asset)
    date_columns = (
        (*TERMINAL_EVENT_V2_REQUIRED_DATE_COLUMNS, *TERMINAL_EVENT_V2_OPTIONAL_DATE_COLUMNS)
        if is_v2
        else TERMINAL_EVENT_DATE_COLUMNS
    )
    for column in date_columns:
        if column not in frame.columns:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["cash_per_share"] = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    frame["extinguishes_position"] = _strict_boolean(frame["extinguishes_position"], "extinguishes_position")
    if is_v2:
        frame["is_final_distribution"] = _strict_boolean(
            frame["is_final_distribution"], "is_final_distribution"
        )
        frame["additional_distribution_expected"] = _strict_boolean(
            frame["additional_distribution_expected"], "additional_distribution_expected"
        )
    frame["historical_backtest_allowed"] = _strict_boolean(
        frame["historical_backtest_allowed"], "historical_backtest_allowed"
    )
    frame["model_promotion_allowed"] = _strict_boolean(
        frame["model_promotion_allowed"], "model_promotion_allowed"
    )
    required_values = [
        "asset",
        "event_type",
        *(
            TERMINAL_EVENT_V2_REQUIRED_DATE_COLUMNS
            if is_v2
            else TERMINAL_EVENT_DATE_COLUMNS
        ),
        "cash_per_share",
        "source_vintage",
    ]
    if frame[required_values].isna().any(axis=None):
        raise ValueError("ETF terminal cash-event registry contains missing formal fields")
    if not np.isfinite(frame["cash_per_share"]).all() or frame["cash_per_share"].le(0).any():
        raise ValueError("ETF terminal cash-event registry contains invalid cash values")
    if not frame["event_type"].eq("liquidation_distribution").all():
        raise ValueError("ETF terminal cash-event registry contains an unsupported event type")
    if not frame["historical_backtest_allowed"].all() or frame["model_promotion_allowed"].any():
        raise ValueError("ETF terminal cash-event registry violates formal promotion boundaries")
    if is_v2:
        chronology = (
            frame["entitlement_date"].le(frame["pay_date"])
            & frame["pay_date"].le(frame["accounting_date"])
            & frame["available_trade_date"].le(frame["accounting_date"])
            & frame["available_date"].eq(frame["available_trade_date"])
        )
        if not chronology.all():
            raise ValueError("ETF terminal cash-event V2 registry contains invalid PIT chronology")
        if frame["event_id"].astype(str).duplicated().any():
            raise ValueError("ETF terminal cash-event V2 registry contains duplicate event IDs")
        if not frame["validation_status"].astype(str).eq("pass").all():
            raise ValueError("ETF terminal cash-event V2 registry contains unvalidated rows")
        if (
            (frame["is_final_distribution"] & frame["additional_distribution_expected"]).any()
            or (frame["extinguishes_position"] & ~frame["is_final_distribution"]).any()
        ):
            raise ValueError("ETF terminal cash-event V2 registry has contradictory final-event semantics")
        economic_key = ["asset", "event_type", "holder_scope", "pay_date", "source_pdf_sha256_set"]
        if frame.duplicated(economic_key, keep=False).any():
            raise ValueError("ETF terminal cash-event V2 registry contains duplicate economic keys")
        for (_, _), group in frame.groupby(["asset", "holder_scope"]):
            ordered = group.sort_values("distribution_sequence")
            if ordered["distribution_sequence"].astype(int).tolist() != list(range(1, len(ordered) + 1)):
                raise ValueError("ETF terminal cash-event V2 sequence is not contiguous")
            extinguishing = ordered["extinguishes_position"]
            if int(extinguishing.sum()) > 1 or (extinguishing.any() and not bool(extinguishing.iloc[-1])):
                raise ValueError("ETF terminal cash-event V2 extinguishment is not final")
    else:
        if not frame["extinguishes_position"].all():
            raise ValueError("ETF terminal cash-event registry contains a non-extinguishing event")
        if not frame["available_date"].eq(frame["announcement_date"]).all():
            raise ValueError("ETF terminal cash-event available date must equal its announcement date")
        chronology = (
            frame["announcement_date"].lt(frame["record_date"])
            & frame["record_date"].lt(frame["ex_date"])
            & frame["ex_date"].lt(frame["pay_date"])
            & frame["pay_date"].lt(frame["termination_date"])
        )
        if not chronology.all():
            raise ValueError("ETF terminal cash-event registry contains invalid chronology")
        if frame["asset"].duplicated().any():
            raise ValueError("legacy ETF terminal cash-event registry contains duplicate assets")
    if int(manifest.get("assets", -1)) != frame["asset"].nunique():
        raise ValueError("ETF terminal cash-event registry asset count is undeclared")
    if not frame["source_vintage"].astype(str).str.startswith("official_terminal_event_pdf_set_sha256:").all():
        raise ValueError("ETF terminal cash-event registry has an unauthenticated source vintage")

    metadata = {
        "qualification_status": qualification,
        "scope_complete": False,
        "scope_boundary": str(manifest.get("scope_boundary", "")),
        "formal_validated_event_rows": int(manifest.get("rows", len(frame))),
        "formal_validated_event_assets": int(manifest.get("assets", frame["asset"].nunique())),
        "formal_complete_event_chain_assets": int(manifest.get("complete_chain_assets", 0)),
        "quarantined_candidate_rows": int(manifest.get("quarantined_candidate_rows", 0)),
        "table_sha256": _sha256(event_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    metadata["bundle_sha256"] = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        if is_v2:
            frame = frame[
                frame["available_trade_date"].le(cutoff)
                & frame["accounting_date"].le(cutoff)
            ].copy()
        else:
            frame = frame[frame["available_date"].le(cutoff) & frame["ex_date"].le(cutoff)].copy()
    metadata["validated_event_rows"] = int(len(frame))
    metadata["validated_event_assets"] = int(frame["asset"].nunique())
    metadata["complete_event_chain_assets"] = int(
        frame.loc[frame["extinguishes_position"], "asset"].nunique()
    )
    sort_columns = ["asset", "holder_scope", "distribution_sequence"] if is_v2 else ["asset", "ex_date"]
    return frame.sort_values(sort_columns).reset_index(drop=True), metadata


def collapse_lifecycles(master: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Collapse event rows without exposing delistings unavailable at ``as_of``."""

    required = {
        "asset",
        "asset_name",
        "list_date",
        "delist_date",
        "event_type",
        "exchange",
        "available_date",
    }
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"ETF security master is missing columns: {missing}")
    cutoff = pd.Timestamp(as_of).normalize()
    data = master.copy()
    data["asset"] = data["asset"].map(_normalise_asset)
    data["available_date"] = pd.to_datetime(data["available_date"], errors="coerce").dt.normalize()
    data["list_date"] = pd.to_datetime(data["list_date"], errors="coerce").dt.normalize()
    data["delist_date"] = pd.to_datetime(data["delist_date"], errors="coerce").dt.normalize()
    if data[["asset", "event_type", "available_date"]].isna().any(axis=None):
        raise ValueError("ETF security master contains incomplete event keys")
    data = data[data["available_date"].le(cutoff)].copy()

    listings = data[data["event_type"].eq("listing")][
        ["asset", "asset_name", "exchange", "list_date"]
    ].copy()
    exits = data[data["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listings["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("ETF security master contains duplicate lifecycle events")
    if listings[["asset", "asset_name", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("ETF security master contains incomplete listing events")
    if exits["delist_date"].isna().any():
        raise ValueError("ETF security master contains a delisting without delist_date")

    lifecycle = listings.merge(exits, on="asset", how="left", validate="one_to_one")
    lifecycle = lifecycle[lifecycle["list_date"].le(cutoff)].copy()
    invalid_exit = lifecycle["delist_date"].notna() & lifecycle["delist_date"].lt(lifecycle["list_date"])
    if invalid_exit.any():
        assets = lifecycle.loc[invalid_exit, "asset"].tolist()
        raise ValueError(f"ETF lifecycle delists before listing: {assets[:5]}")
    lifecycle["lifecycle_status"] = np.where(lifecycle["delist_date"].notna(), "delisted", "active")
    return lifecycle.sort_values(["list_date", "asset"]).reset_index(drop=True)


def load_lifecycles(
    path: Path = DEFAULT_MASTER,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    cutoff = pd.Timestamp(as_of or datetime.now().date()).normalize()
    return collapse_lifecycles(pd.read_csv(path, dtype={"asset": str}, low_memory=False), cutoff)


def select_lifecycles(
    lifecycles: pd.DataFrame,
    *,
    mode: str = "pilot",
    explicit_assets: Iterable[str] | None = None,
    earliest_limit: int = 10,
    delisted_limit: int = 10,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select a deterministic pilot or a resumable full-lifecycle slice."""

    required = {"asset", "list_date", "delist_date", "lifecycle_status"}
    missing = sorted(required.difference(lifecycles.columns))
    if missing:
        raise ValueError(f"lifecycle table is missing columns: {missing}")
    if earliest_limit < 0 or delisted_limit < 0 or offset < 0 or (limit is not None and limit < 1):
        raise ValueError("selection limits must be non-negative and limit must be positive")
    data = lifecycles.sort_values(["list_date", "asset"]).reset_index(drop=True).copy()
    data["asset"] = data["asset"].map(_normalise_asset)

    if explicit_assets:
        requested = [_normalise_asset(value) for value in explicit_assets]
        if len(requested) != len(set(requested)):
            raise ValueError("explicit ETF assets contain duplicates")
        indexed = data.set_index("asset", drop=False)
        missing_assets = [asset for asset in requested if asset not in indexed.index]
        if missing_assets:
            raise ValueError(f"ETF assets are absent from the governed master: {missing_assets}")
        output = indexed.loc[requested].reset_index(drop=True)
        output["selection_group"] = "explicit"
        return output

    if mode == "pilot":
        active = data[data["lifecycle_status"].eq("active")].head(earliest_limit).copy()
        active["selection_group"] = "active_earliest"
        delisted = data[data["lifecycle_status"].eq("delisted")].sort_values(
            ["delist_date", "asset"]
        ).head(delisted_limit).copy()
        delisted["selection_group"] = "delisted_earliest"
        return pd.concat([active, delisted], ignore_index=True).drop_duplicates("asset").reset_index(drop=True)
    if mode == "all":
        output = data.iloc[offset : None if limit is None else offset + limit].copy()
        output["selection_group"] = "full_lifecycle"
        return output.reset_index(drop=True)
    raise ValueError("selection mode must be 'pilot' or 'all'")


def _cache_paths(cache_dir: Path, asset: str) -> dict[str, Path]:
    directory = cache_dir / _normalise_asset(asset)
    return {
        "directory": directory,
        "price": directory / "price.csv.gz",
        "dividend": directory / "dividend.csv.gz",
        "metadata": directory / "metadata.json",
        "failure": directory / "last_failure.json",
    }


def _normalise_source_prices(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_PRICE_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF price source is missing columns for {asset}: {missing}")
    prices = raw[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    prices[numeric] = prices[numeric].apply(pd.to_numeric, errors="coerce")
    if prices.empty or prices[["date", *numeric]].isna().any(axis=None):
        raise ValueError(f"ETF price source has incomplete rows for {asset}")
    if prices["date"].duplicated().any():
        raise ValueError(f"ETF price source has duplicate dates for {asset}")
    if (prices[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError(f"ETF price source has non-positive OHLC for {asset}")
    if (prices[["volume", "amount"]] < 0).any(axis=None):
        raise ValueError(f"ETF price source has negative volume/amount for {asset}")
    invalid_bounds = (
        prices["high"].lt(prices[["open", "close", "low"]].max(axis=1))
        | prices["low"].gt(prices[["open", "close", "high"]].min(axis=1))
    )
    if invalid_bounds.any():
        raise ValueError(f"ETF price source has invalid OHLC bounds for {asset}")
    return prices.sort_values("date").reset_index(drop=True)


def _normalise_source_dividends(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "cumulative_dividend"])
    missing = sorted(REQUIRED_DIVIDEND_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF dividend source is missing columns for {asset}: {missing}")
    events = raw[["date", "cumulative_dividend"]].copy()
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
    events["cumulative_dividend"] = pd.to_numeric(events["cumulative_dividend"], errors="coerce")
    if events[["date", "cumulative_dividend"]].isna().any(axis=None):
        raise ValueError(f"ETF dividend source has incomplete rows for {asset}")
    if events["date"].duplicated().any():
        raise ValueError(f"ETF dividend source has duplicate dates for {asset}")
    events = events.sort_values("date").reset_index(drop=True)
    if (events["cumulative_dividend"] < 0).any() or (
        events["cumulative_dividend"].diff().dropna() < -1e-10
    ).any():
        raise ValueError(f"ETF cumulative dividends are invalid for {asset}")
    return events


def _cache_is_valid(cache_dir: Path, lifecycle: Any, as_of: pd.Timestamp) -> bool:
    asset = _normalise_asset(lifecycle.asset)
    paths = _cache_paths(cache_dir, asset)
    if not all(paths[key].is_file() for key in ("price", "dividend", "metadata")):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("status") != "success" or metadata.get("asset") != asset:
            return False
        if _sha256(paths["price"]) != metadata.get("price_sha256"):
            return False
        if _sha256(paths["dividend"]) != metadata.get("dividend_sha256"):
            return False
        requested_end = pd.Timestamp(metadata["requested_lifecycle_end"]).normalize()
        lifecycle_end = (
            min(pd.Timestamp(lifecycle.delist_date).normalize(), as_of)
            if pd.notna(lifecycle.delist_date)
            else as_of
        )
        if requested_end < lifecycle_end:
            return False
        _normalise_source_prices(pd.read_csv(paths["price"], compression="gzip"), asset)
        _normalise_source_dividends(pd.read_csv(paths["dividend"], compression="gzip"), asset)
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


PriceFetcher = Callable[[str], pd.DataFrame]
DividendFetcher = Callable[[str], pd.DataFrame]


def collect_asset(
    lifecycle: Any,
    as_of: str | pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    refresh: bool = False,
    attempts: int = 2,
    retry_sleep_seconds: float = 1.0,
    price_fetcher: PriceFetcher = fetch_price,
    dividend_fetcher: DividendFetcher = fetch_dividends,
) -> dict[str, Any]:
    """Fetch one asset atomically and retain a machine-verifiable cache."""

    if attempts < 1:
        raise ValueError("attempts must be at least one")
    cutoff = pd.Timestamp(as_of).normalize()
    asset = _normalise_asset(lifecycle.asset)
    start = pd.Timestamp(lifecycle.list_date).normalize()
    end = min(pd.Timestamp(lifecycle.delist_date).normalize(), cutoff) if pd.notna(lifecycle.delist_date) else cutoff
    paths = _cache_paths(cache_dir, asset)
    if not refresh and _cache_is_valid(cache_dir, lifecycle, cutoff):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        return {**metadata, "collection_action": "cache_reused"}

    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            prices = _normalise_source_prices(price_fetcher(asset), asset)
            dividends = _normalise_source_dividends(dividend_fetcher(asset), asset)
            fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
            _atomic_csv(prices, paths["price"], compression="gzip")
            _atomic_csv(dividends, paths["dividend"], compression="gzip")
            metadata = {
                "status": "success",
                "asset": asset,
                "asset_name": str(lifecycle.asset_name),
                "lifecycle_status": str(lifecycle.lifecycle_status),
                "list_date": start.date().isoformat(),
                "delist_date": pd.Timestamp(lifecycle.delist_date).date().isoformat()
                if pd.notna(lifecycle.delist_date)
                else None,
                "requested_lifecycle_end": end.date().isoformat(),
                "provider_price_rows": int(len(prices)),
                "provider_dividend_rows": int(len(dividends)),
                "provider_coverage_start": prices["date"].min().date().isoformat(),
                "provider_coverage_end": prices["date"].max().date().isoformat(),
                "fetched_at": fetched_at,
                "data_source": SOURCE_NAME,
                "price_path": _relative(paths["price"]),
                "price_sha256": _sha256(paths["price"]),
                "dividend_path": _relative(paths["dividend"]),
                "dividend_sha256": _sha256(paths["dividend"]),
                "collection_action": "fetched",
            }
            _atomic_json(metadata, paths["metadata"])
            return metadata
        except Exception as exc:  # noqa: BLE001 - provider errors are recorded, not hidden
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if attempt < attempts and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds * attempt)

    failure = {
        "status": "failed",
        "asset": asset,
        "asset_name": str(lifecycle.asset_name),
        "requested_lifecycle_end": end.date().isoformat(),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "attempts": attempts,
        "error": last_error,
        "collection_action": "failed",
    }
    failure_path = paths["metadata"]
    if paths["metadata"].is_file():
        try:
            existing = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                failure_path = paths["failure"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    _atomic_json(failure, failure_path)
    return failure


def load_cached_asset(cache_dir: Path, lifecycle: Any, as_of: str | pd.Timestamp) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    asset = _normalise_asset(lifecycle.asset)
    if not _cache_is_valid(cache_dir, lifecycle, cutoff):
        raise ValueError(f"validated ETF cache is unavailable for {asset}")
    paths = _cache_paths(cache_dir, asset)
    return {
        "price": pd.read_csv(paths["price"], compression="gzip", low_memory=False),
        "dividend": pd.read_csv(paths["dividend"], compression="gzip", low_memory=False),
        "metadata": json.loads(paths["metadata"].read_text(encoding="utf-8")),
    }


def _source_vintage(
    asset: str,
    metadata: dict[str, Any],
    registry_sha256: str,
    terminal_registry_sha256: str = "fixture:none",
) -> str:
    material = {
        "asset": asset,
        "fetched_at": metadata.get("fetched_at"),
        "price_sha256": metadata.get("price_sha256"),
        "dividend_sha256": metadata.get("dividend_sha256"),
        "corporate_action_registry_sha256": registry_sha256,
        "terminal_cash_event_registry_bundle_sha256": terminal_registry_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sina_current_final_snapshot_bundle_sha256:{digest}"


def _trim_verified_stale_post_delisting_tail(
    prices: pd.DataFrame,
    lifecycle_end: pd.Timestamp,
    asset: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    trailing = prices[prices["date"].gt(lifecycle_end)].copy()
    if trailing.empty:
        return prices, []
    inside = prices[prices["date"].le(lifecycle_end)]
    if len(trailing) == 1 and not inside.empty:
        artifact = trailing.iloc[0]
        last_valid = inside.iloc[-1]
        market_columns = ["open", "high", "low", "close", "volume", "amount"]
        values_match = bool(
            np.allclose(
                artifact[market_columns].to_numpy(dtype=float),
                last_valid[market_columns].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
        artifact_date = pd.Timestamp(artifact["date"]).normalize()
        if artifact_date.dayofweek >= 5 and values_match:
            detail = {
                "resolution": "removed_exact_weekend_duplicate_after_governed_delisting",
                "artifact_date": artifact_date.date().isoformat(),
                "matched_last_valid_date": pd.Timestamp(last_valid["date"]).date().isoformat(),
            }
            return inside.copy(), [detail]
    last = trailing["date"].max().date()
    raise ValueError(f"ETF provider history continues after governed delisting for {asset}: {last}")


def build_lifecycle_observation(
    raw_prices: pd.DataFrame,
    raw_dividends: pd.DataFrame,
    lifecycle: Any,
    as_of: str | pd.Timestamp,
    source_observed_at: Any,
    source_vintage: str,
    share_conversion_factors: dict[pd.Timestamp, float] | None = None,
    terminal_cash_event: pd.DataFrame | pd.Series | dict[str, Any] | list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one lifecycle while failing closed on evidence of code reuse."""

    cutoff = pd.Timestamp(as_of).normalize()
    asset = _normalise_asset(lifecycle.asset)
    start = pd.Timestamp(lifecycle.list_date).normalize()
    end = min(pd.Timestamp(lifecycle.delist_date).normalize(), cutoff) if pd.notna(lifecycle.delist_date) else cutoff
    observed = pd.Timestamp(source_observed_at)
    if pd.isna(observed):
        raise ValueError(f"ETF source observation timestamp is missing for {asset}")
    prices = _normalise_source_prices(raw_prices, asset)
    dividends = _normalise_source_dividends(raw_dividends, asset)

    if prices["date"].lt(start).any():
        first = prices.loc[prices["date"].lt(start), "date"].min().date()
        raise ValueError(f"ETF provider history precedes governed listing for {asset}: {first}")
    provider_tail_resolutions: list[dict[str, Any]] = []
    if pd.notna(lifecycle.delist_date):
        prices, provider_tail_resolutions = _trim_verified_stale_post_delisting_tail(prices, end, asset)
    if pd.notna(lifecycle.delist_date) and not dividends.empty and dividends["date"].gt(end).any():
        raise ValueError(f"ETF dividend history continues after governed delisting for {asset}")

    prices = prices[prices["date"].between(start, end)].copy()
    dividends = dividends[dividends["date"].between(start, end)].copy()
    if prices.empty:
        raise ValueError(f"ETF provider has no rows inside governed lifecycle for {asset}")

    events = dividends.copy()
    events["cash_distribution"] = events["cumulative_dividend"].diff().fillna(
        events["cumulative_dividend"]
    )
    if (events["cash_distribution"] < -1e-10).any():
        raise ValueError(f"ETF cumulative dividends decreased inside lifecycle for {asset}")
    events.loc[events["cash_distribution"].abs().le(1e-10), "cash_distribution"] = 0.0

    terminal_details: list[dict[str, Any]] = []
    adjustment_dividends = dividends
    terminal_rows: list[dict[str, Any]] = []
    if terminal_cash_event is not None:
        if isinstance(terminal_cash_event, pd.DataFrame):
            terminal_rows = terminal_cash_event.to_dict(orient="records")
        elif isinstance(terminal_cash_event, pd.Series):
            terminal_rows = [terminal_cash_event.to_dict()]
        elif isinstance(terminal_cash_event, dict):
            terminal_rows = [dict(terminal_cash_event)]
        elif isinstance(terminal_cash_event, list):
            terminal_rows = [dict(item) for item in terminal_cash_event]
        else:
            raise TypeError(f"ETF terminal cash event has an unsupported row type for {asset}")
        if str(lifecycle.lifecycle_status) != "delisted" or pd.isna(lifecycle.delist_date):
            raise ValueError(f"ETF terminal cash events require a governed delisted lifecycle: {asset}")
        claimed_marker_indices: set[Any] = set()
        for position, terminal_row in enumerate(terminal_rows, start=1):
            is_v2 = "event_id" in terminal_row and "accounting_date" in terminal_row
            required_terminal = (
                REQUIRED_TERMINAL_EVENT_V2_COLUMNS if is_v2 else REQUIRED_TERMINAL_EVENT_COLUMNS
            )
            missing_terminal = sorted(required_terminal.difference(terminal_row))
            if missing_terminal:
                raise ValueError(
                    f"ETF terminal cash event is incomplete for {asset}: {missing_terminal}"
                )
            terminal_asset = _normalise_asset(terminal_row["asset"])
            if terminal_asset != asset:
                raise ValueError(f"ETF terminal cash event asset mismatch: {asset} != {terminal_asset}")
            if str(terminal_row["event_type"]) != "liquidation_distribution":
                raise ValueError(f"ETF terminal cash event has an unsupported type for {asset}")
            terminal_cash = float(terminal_row["cash_per_share"])
            terminal_extinguishes = str(terminal_row["extinguishes_position"]).strip().lower() == "true"
            terminal_historical = str(terminal_row["historical_backtest_allowed"]).strip().lower() == "true"
            terminal_model = str(terminal_row["model_promotion_allowed"]).strip().lower() == "true"
            if not np.isfinite(terminal_cash) or terminal_cash <= 0:
                raise ValueError(f"ETF terminal cash event has invalid cash for {asset}")
            if not terminal_historical or terminal_model:
                raise ValueError(f"ETF terminal cash event violates promotion boundaries for {asset}")
            if is_v2:
                terminal_dates = {
                    column: pd.Timestamp(terminal_row[column]).normalize()
                    for column in TERMINAL_EVENT_V2_REQUIRED_DATE_COLUMNS
                }
                for column in TERMINAL_EVENT_V2_OPTIONAL_DATE_COLUMNS:
                    value = pd.to_datetime(terminal_row.get(column), errors="coerce")
                    terminal_dates[column] = pd.Timestamp(value).normalize() if pd.notna(value) else pd.NaT
                if not (
                    terminal_dates["entitlement_date"] <= terminal_dates["pay_date"]
                    <= terminal_dates["accounting_date"]
                    and terminal_dates["available_trade_date"] <= terminal_dates["accounting_date"]
                    and terminal_dates["available_date"] == terminal_dates["available_trade_date"]
                    and terminal_dates["accounting_date"] <= cutoff
                ):
                    raise ValueError(f"ETF terminal cash event V2 chronology is invalid for {asset}")
                candidate_dates = {
                    terminal_dates["entitlement_date"],
                    terminal_dates["record_date"],
                    terminal_dates["pay_date"],
                    terminal_dates["accounting_date"],
                }
                if pd.notna(terminal_dates["ex_date"]):
                    candidate_dates.add(terminal_dates["ex_date"])
                event_id = str(terminal_row["event_id"])
                evidence_basis = "formal_validated_terminal_event_chain_v2"
                output_event_date = terminal_dates["accounting_date"]
            else:
                terminal_dates = {
                    column: pd.Timestamp(terminal_row[column]).normalize()
                    for column in TERMINAL_EVENT_DATE_COLUMNS
                }
                if not terminal_extinguishes or not (
                    terminal_dates["announcement_date"]
                    < terminal_dates["record_date"]
                    < terminal_dates["ex_date"]
                    < terminal_dates["pay_date"]
                    < terminal_dates["termination_date"]
                ):
                    raise ValueError(f"legacy ETF terminal cash event chronology is invalid for {asset}")
                candidate_dates = {terminal_dates["ex_date"]}
                event_id = f"legacy:{asset}:{position}"
                evidence_basis = "formal_known_exception_registry"
                output_event_date = terminal_dates["ex_date"]
            marker = events[
                events["date"].isin(candidate_dates)
                & np.isclose(
                    events["cash_distribution"].astype(float),
                    terminal_cash,
                    rtol=1e-10,
                    atol=1e-8,
                )
            ]
            marker = marker[~marker.index.isin(claimed_marker_indices)]
            if len(marker) > 1 or (not is_v2 and len(marker) != 1):
                raise ValueError(f"ETF terminal cash event provider marker is ambiguous for {asset}")
            marker_index = marker.index[0] if len(marker) == 1 else None
            marker_cash = float(marker.iloc[0]["cash_distribution"]) if len(marker) == 1 else None
            marker_date = pd.Timestamp(marker.iloc[0]["date"]).normalize() if len(marker) == 1 else None
            if marker_index is not None:
                claimed_marker_indices.add(marker_index)
            terminal_details.append(
                {
                    "event_id": event_id,
                    "event_type": "liquidation_distribution",
                    "announcement_date": terminal_dates["announcement_date"].date().isoformat(),
                    "record_date": terminal_dates["record_date"].date().isoformat(),
                    "ex_date": terminal_dates["ex_date"].date().isoformat()
                    if pd.notna(terminal_dates["ex_date"])
                    else None,
                    "pay_date": terminal_dates["pay_date"].date().isoformat(),
                    "accounting_date": output_event_date.date().isoformat(),
                    "official_termination_date": terminal_dates["termination_date"].date().isoformat()
                    if pd.notna(terminal_dates.get("termination_date"))
                    else None,
                    "cash_per_share": terminal_cash,
                    "provider_marker_cash": marker_cash,
                    "provider_marker_date": marker_date.date().isoformat() if marker_date is not None else None,
                    "extinguishes_position": terminal_extinguishes,
                    "distribution_sequence": int(terminal_row["distribution_sequence"])
                    if is_v2
                    else position,
                    "holder_scope": str(terminal_row["holder_scope"])
                    if is_v2
                    else "all_registered_holders",
                    "provider_marker_matched": marker_index is not None,
                    "available_date": terminal_dates["available_date"].date().isoformat(),
                    "source_vintage": str(terminal_row["source_vintage"]),
                    "qualification_status": (
                        "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2"
                        if is_v2
                        else "PROMOTED_KNOWN_TERMINAL_CASH_EVENT"
                    ),
                    "evidence_basis": evidence_basis,
                }
            )
        if claimed_marker_indices:
            adjustment_dividends = dividends[~dividends.index.isin(claimed_marker_indices)].copy()
    governed_factors = share_conversion_factors or {}
    adjusted = total_return_adjusted_prices(
        prices,
        adjustment_dividends,
        asset,
        end,
        share_conversion_factors=governed_factors,
    )
    applied_actions = list(adjusted.attrs.get("applied_share_actions", []))
    distribution_alignments = list(adjusted.attrs.get("distribution_alignments", []))
    governed_actions = sum(item.get("evidence_basis") == "governed_registry" for item in applied_actions)
    inferred_actions = sum(
        item.get("evidence_basis") == "zero_marker_common_factor_inference" for item in applied_actions
    )
    if inferred_actions:
        action_evidence_status = "heuristic_inference_present"
    elif governed_actions:
        action_evidence_status = "governed_registry_only"
    else:
        action_evidence_status = "no_share_adjustment"
    adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce").dt.normalize()
    if adjusted["date"].isna().any() or adjusted["date"].duplicated().any():
        raise ValueError(f"ETF adjusted output has invalid dates for {asset}")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    if not np.isfinite(adjusted[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"ETF adjusted output has non-finite market fields for {asset}")

    available = observed.tz_localize(None).normalize() if observed.tzinfo is not None else observed.normalize()
    output = pd.DataFrame(
        {
            "date": adjusted["date"],
            "asset": asset,
            "asset_name": str(lifecycle.asset_name),
            "asset_type": "etf",
            "exchange": str(lifecycle.exchange),
            "lifecycle_status": str(lifecycle.lifecycle_status),
            "list_date": start,
            "delist_date": pd.Timestamp(lifecycle.delist_date).normalize()
            if pd.notna(lifecycle.delist_date)
            else pd.NaT,
            "open": adjusted["open"],
            "high": adjusted["high"],
            "low": adjusted["low"],
            "close": adjusted["close"],
            "volume": adjusted["volume"],
            "amount": adjusted["amount"],
            "raw_close": adjusted["raw_close"],
            "source_cash_distribution": adjusted["source_cash_distribution"],
            "cash_distribution": adjusted["cash_distribution"],
            "share_adjustment_factor": adjusted["share_adjustment_factor"],
            "adjustment_factor": adjusted["adjustment_factor"],
            "return_basis": "total_return_pre_terminal_cash_event" if terminal_rows else "total_return",
            "market_available_date": adjusted["date"],
            "source_observed_at": observed.isoformat(),
            "available_date": available,
            "pit_actionable": False,
            "data_source": SOURCE_NAME + "; current final snapshot",
            "source_vintage": source_vintage,
            "qualification_status": QUALIFICATION_STATUS,
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
    )[PRICE_COLUMNS]

    dividend_output = pd.DataFrame(columns=DIVIDEND_COLUMNS)
    if not events.empty:
        dividend_output = pd.DataFrame(
            {
                "event_date": events["date"],
                "announcement_date": pd.NaT,
                "record_date": pd.NaT,
                "ex_date": events["date"],
                "pay_date": pd.NaT,
                "asset": asset,
                "asset_name": str(lifecycle.asset_name),
                "asset_type": "etf",
                "lifecycle_status": str(lifecycle.lifecycle_status),
                "list_date": start,
                "delist_date": pd.Timestamp(lifecycle.delist_date).normalize()
                if pd.notna(lifecycle.delist_date)
                else pd.NaT,
                "cumulative_dividend": events["cumulative_dividend"],
                "cash_distribution": events["cash_distribution"],
                "event_type": np.where(events["cash_distribution"] > 0, "cash_distribution", "zero_marker"),
                "extinguishes_position": False,
                "source_observed_at": observed.isoformat(),
                "available_date": available,
                "pit_actionable": False,
                "data_source": SOURCE_NAME + "; current final snapshot",
                "source_vintage": source_vintage,
                "qualification_status": QUALIFICATION_STATUS,
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )[DIVIDEND_COLUMNS]
    terminal_output_rows: list[dict[str, Any]] = []
    claimed_output_indices: set[Any] = set()
    for detail in terminal_details:
        marker_date = pd.to_datetime(detail["provider_marker_date"], errors="coerce")
        marker_cash = detail["provider_marker_cash"]
        terminal_index = None
        if pd.notna(marker_date) and marker_cash is not None and not dividend_output.empty:
            terminal_mask = (
                dividend_output["event_date"].eq(pd.Timestamp(marker_date).normalize())
                & np.isclose(
                    pd.to_numeric(dividend_output["cash_distribution"], errors="coerce"),
                    float(marker_cash),
                    rtol=1e-10,
                    atol=1e-8,
                )
                & ~dividend_output.index.isin(claimed_output_indices)
            )
            if int(terminal_mask.sum()) != 1:
                raise ValueError(f"ETF terminal cash event output marker is ambiguous for {asset}")
            terminal_index = dividend_output.index[terminal_mask][0]
            claimed_output_indices.add(terminal_index)

        official_values = {
            "event_date": pd.Timestamp(detail["accounting_date"]).normalize(),
            "announcement_date": pd.Timestamp(detail["announcement_date"]).normalize(),
            "record_date": pd.Timestamp(detail["record_date"]).normalize(),
            "ex_date": pd.to_datetime(detail["ex_date"], errors="coerce"),
            "pay_date": pd.Timestamp(detail["pay_date"]).normalize(),
            "asset": asset,
            "asset_name": str(lifecycle.asset_name),
            "asset_type": "etf",
            "lifecycle_status": str(lifecycle.lifecycle_status),
            "list_date": start,
            "delist_date": pd.Timestamp(lifecycle.delist_date).normalize(),
            "cash_distribution": float(detail["cash_per_share"]),
            "event_type": "liquidation_distribution",
            "extinguishes_position": bool(detail["extinguishes_position"]),
            "source_observed_at": observed.isoformat(),
            "available_date": pd.Timestamp(detail["available_date"]).normalize(),
            "pit_actionable": True,
            "data_source": (
                "Official liquidation distribution evidence; provider marker cross-check"
                if terminal_index is not None
                else "Official liquidation distribution evidence; cash ledger only"
            ),
            "source_vintage": str(detail["source_vintage"]),
            "qualification_status": str(detail["qualification_status"]),
            "historical_backtest_allowed": True,
            "model_promotion_allowed": False,
        }
        if terminal_index is not None:
            for column, value in official_values.items():
                dividend_output.loc[terminal_index, column] = value
        else:
            terminal_output_rows.append({**official_values, "cumulative_dividend": np.nan})

    if terminal_output_rows:
        dividend_output = pd.concat(
            [dividend_output, pd.DataFrame(terminal_output_rows)[DIVIDEND_COLUMNS]],
            ignore_index=True,
        )
    if not dividend_output.empty:
        for column in ("event_date", "announcement_date", "record_date", "ex_date", "pay_date", "available_date"):
            dividend_output[column] = pd.to_datetime(dividend_output[column], errors="coerce").dt.normalize()
        dividend_output = dividend_output.sort_values(
            ["event_date", "event_type", "cash_distribution"],
            kind="mergesort",
        ).reset_index(drop=True)

    coverage_start = pd.Timestamp(output["date"].min()).normalize()
    coverage_end = pd.Timestamp(output["date"].max()).normalize()
    diagnostics = {
        "coverage_start": coverage_start.date().isoformat(),
        "coverage_end": coverage_end.date().isoformat(),
        "listing_start_gap_days": int((coverage_start - start).days),
        "delisting_end_gap_days": int((end - coverage_end).days)
        if pd.notna(lifecycle.delist_date)
        else None,
        "provider_tail_duplicate_rows_removed": int(len(provider_tail_resolutions)),
        "provider_tail_duplicate_detail_json": json.dumps(
            provider_tail_resolutions,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "maximum_distribution_alignment_lag_days": max(
            (int(item["calendar_lag_days"]) for item in distribution_alignments),
            default=0,
        ),
        "distribution_alignment_detail_json": json.dumps(
            distribution_alignments,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "registered_corporate_actions": int(len(governed_factors)),
        "applied_corporate_actions": int(len(applied_actions)),
        "governed_corporate_actions": int(governed_actions),
        "inferred_corporate_actions": int(inferred_actions),
        "corporate_action_evidence_status": action_evidence_status,
        "corporate_action_evidence_detail_json": json.dumps(
            applied_actions,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "terminal_cash_events": int(len(terminal_details)),
        "terminal_cash_evidence_status": (
            "formal_validated_event_chain_v2"
            if any(
                item["qualification_status"] == "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2"
                for item in terminal_details
            )
            else "governed_known_exception"
            if terminal_details
            else "none_registered"
        ),
        "terminal_cash_evidence_detail_json": json.dumps(
            terminal_details,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    return output, dividend_output, diagnostics


AssetLoader = Callable[[Any], dict[str, Any]]


def provider_circuit_breaker_reason(error: Any) -> str | None:
    text = str(error).lower()
    patterns = {
        "http_403": ("403", "forbidden"),
        "http_429": ("429", "too many requests"),
        "connection_rejected": ("connection reset", "remote disconnected", "max retries exceeded"),
    }
    for reason, tokens in patterns.items():
        if any(token in text for token in tokens):
            return reason
    return None


def process_lifecycle_batch(
    selected: pd.DataFrame,
    loader: AssetLoader,
    as_of: str | pd.Timestamp,
    action_registry: pd.DataFrame | None = None,
    action_registry_sha256: str = "fixture:none",
    terminal_events: pd.DataFrame | None = None,
    terminal_registry_sha256: str = "fixture:none",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build assets independently so one bad lifecycle cannot poison the panel."""

    price_frames: list[pd.DataFrame] = []
    dividend_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    governed_terminal_events = (
        terminal_events.copy() if terminal_events is not None else pd.DataFrame(columns=["asset"])
    )
    if "asset" not in governed_terminal_events.columns:
        raise ValueError("ETF terminal cash-event frame is missing asset")
    if not governed_terminal_events.empty:
        governed_terminal_events["asset"] = governed_terminal_events["asset"].map(_normalise_asset)
        if "event_id" in governed_terminal_events.columns and governed_terminal_events["event_id"].astype(str).duplicated().any():
            raise ValueError("ETF terminal cash-event frame has duplicate event IDs")
    for lifecycle in selected.itertuples(index=False):
        asset = _normalise_asset(lifecycle.asset)
        status: dict[str, Any] = {
            "asset": asset,
            "asset_name": str(lifecycle.asset_name),
            "exchange": str(lifecycle.exchange),
            "selection_group": str(getattr(lifecycle, "selection_group", "unspecified")),
            "lifecycle_status": str(lifecycle.lifecycle_status),
            "list_date": pd.Timestamp(lifecycle.list_date).date().isoformat(),
            "delist_date": pd.Timestamp(lifecycle.delist_date).date().isoformat()
            if pd.notna(lifecycle.delist_date)
            else None,
            "collection_status": "failed",
            "collection_action": "unavailable",
            "build_status": "quarantined",
            "error": "",
            "price_rows": 0,
            "dividend_event_rows": 0,
            "registered_corporate_actions": 0,
            "applied_corporate_actions": 0,
            "governed_corporate_actions": 0,
            "inferred_corporate_actions": 0,
            "corporate_action_evidence_status": "not_built",
            "corporate_action_evidence_detail_json": "[]",
            "coverage_start": None,
            "coverage_end": None,
            "listing_start_gap_days": None,
            "delisting_end_gap_days": None,
            "provider_tail_duplicate_rows_removed": 0,
            "provider_tail_duplicate_detail_json": "[]",
            "maximum_distribution_alignment_lag_days": 0,
            "distribution_alignment_detail_json": "[]",
            "terminal_cash_events": 0,
            "terminal_cash_evidence_status": "not_built",
            "terminal_cash_evidence_detail_json": "[]",
            "source_observed_at": None,
            "qualification_status": QUALIFICATION_STATUS,
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
        try:
            payload = loader(lifecycle)
            metadata = payload["metadata"]
            if metadata.get("status") != "success":
                status.update(
                    {
                        "collection_status": str(metadata.get("status", "failed")),
                        "collection_action": str(metadata.get("collection_action", "failed")),
                    }
                )
                raise ValueError(str(metadata.get("error", "ETF collection failed")))
            observed = metadata.get("fetched_at")
            status.update(
                {
                    "collection_status": "completed",
                    "collection_action": str(metadata.get("collection_action", "cache_loaded")),
                    "source_observed_at": observed,
                }
            )
            terminal_matches = governed_terminal_events[governed_terminal_events["asset"].eq(asset)]
            vintage = _source_vintage(
                asset,
                metadata,
                action_registry_sha256,
                terminal_registry_sha256,
            )
            factors = (
                conversion_factors_for_asset(action_registry, asset, as_of)
                if action_registry is not None
                else {}
            )
            prices, dividends, diagnostics = build_lifecycle_observation(
                payload["price"],
                payload["dividend"],
                lifecycle,
                as_of,
                observed,
                vintage,
                factors,
                terminal_matches if not terminal_matches.empty else None,
            )
            price_frames.append(prices)
            if not dividends.empty:
                dividend_frames.append(dividends)
            status.update(
                {
                    "build_status": "ready_observation",
                    "price_rows": int(len(prices)),
                    "dividend_event_rows": int(len(dividends)),
                    **diagnostics,
                }
            )
        except Exception as exc:  # noqa: BLE001 - quarantine is the data contract
            status["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        statuses.append(status)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame(columns=PRICE_COLUMNS)
    dividends = (
        pd.concat(dividend_frames, ignore_index=True)
        if dividend_frames
        else pd.DataFrame(columns=DIVIDEND_COLUMNS)
    )
    status_frame = pd.DataFrame(statuses, columns=STATUS_COLUMNS)
    if prices.duplicated(["date", "asset"]).any():
        raise ValueError("ETF lifecycle observation contains duplicate asset/date rows")
    if dividends.duplicated(["event_date", "asset"]).any():
        raise ValueError("ETF lifecycle dividends contain duplicate asset/event rows")
    return prices, dividends, status_frame


def _copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def run(
    *,
    as_of: str | pd.Timestamp,
    master_path: Path = DEFAULT_MASTER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    price_output_path: Path = DEFAULT_PRICE_OUTPUT,
    dividend_output_path: Path = DEFAULT_DIVIDEND_OUTPUT,
    status_output_path: Path = DEFAULT_STATUS_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    action_registry_path: Path = DEFAULT_REGISTRY_PATH,
    terminal_event_path: Path = DEFAULT_TERMINAL_EVENT_PATH,
    terminal_event_manifest_path: Path = DEFAULT_TERMINAL_EVENT_MANIFEST_PATH,
    selection_mode: str = "pilot",
    explicit_assets: Iterable[str] | None = None,
    earliest_limit: int = 10,
    delisted_limit: int = 10,
    offset: int = 0,
    limit: int | None = None,
    refresh: bool = False,
    attempts: int = 2,
    sleep_seconds: float = 0.5,
    archive: bool = True,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    explicit_list = list(explicit_assets) if explicit_assets is not None else None
    action_registry = load_corporate_action_registry(action_registry_path)
    action_registry_sha256 = _sha256(action_registry_path)
    terminal_events, terminal_registry_metadata = load_terminal_cash_event_registry(
        terminal_event_path,
        terminal_event_manifest_path,
        cutoff,
    )
    terminal_registry_sha256 = str(terminal_registry_metadata["bundle_sha256"])
    lifecycles = load_lifecycles(master_path, cutoff)
    selected = select_lifecycles(
        lifecycles,
        mode=selection_mode,
        explicit_assets=explicit_list,
        earliest_limit=earliest_limit,
        delisted_limit=delisted_limit,
        offset=offset,
        limit=limit,
    )
    if selected.empty:
        raise ValueError("ETF lifecycle selection is empty")

    collection_results: dict[str, dict[str, Any]] = {}
    selected_rows = list(selected.itertuples(index=False))
    circuit_breaker: dict[str, Any] | None = None
    for position, lifecycle in enumerate(selected_rows):
        result = collect_asset(
            lifecycle,
            cutoff,
            cache_dir,
            refresh=refresh,
            attempts=attempts,
        )
        collection_results[_normalise_asset(lifecycle.asset)] = result
        reason = provider_circuit_breaker_reason(result.get("error")) if result.get("status") == "failed" else None
        if reason is not None:
            circuit_breaker = {
                "reason": reason,
                "trigger_asset": _normalise_asset(lifecycle.asset),
                "error": str(result.get("error", ""))[:500],
            }
            for deferred in selected_rows[position + 1 :]:
                deferred_asset = _normalise_asset(deferred.asset)
                collection_results[deferred_asset] = {
                    "status": "deferred",
                    "asset": deferred_asset,
                    "error": f"provider circuit breaker triggered by {_normalise_asset(lifecycle.asset)}: {reason}",
                    "collection_action": "deferred",
                }
            break
        if sleep_seconds > 0 and position + 1 < len(selected):
            time.sleep(sleep_seconds)

    def loader(lifecycle: Any) -> dict[str, Any]:
        asset = _normalise_asset(lifecycle.asset)
        result = collection_results[asset]
        if result.get("status") != "success":
            return {"price": pd.DataFrame(), "dividend": pd.DataFrame(), "metadata": result}
        payload = load_cached_asset(cache_dir, lifecycle, cutoff)
        payload["metadata"]["collection_action"] = result.get("collection_action", "cache_loaded")
        return payload

    prices, dividends, statuses = process_lifecycle_batch(
        selected,
        loader,
        cutoff,
        action_registry,
        action_registry_sha256,
        terminal_events,
        terminal_registry_sha256,
    )
    _atomic_csv(prices, price_output_path, compression="gzip")
    _atomic_csv(dividends, dividend_output_path, compression="gzip")
    _atomic_csv(statuses, status_output_path)

    successful = statuses[statuses["build_status"].eq("ready_observation")]
    master_hash = _sha256(master_path)
    input_records: list[dict[str, Any]] = [
        {"role": "etf_security_master", "path": _relative(master_path), "sha256": master_hash},
        {
            "role": "etf_corporate_action_registry",
            "path": _relative(action_registry_path),
            "sha256": action_registry_sha256,
        },
        {
            "role": "pit_etf_terminal_cash_events",
            "path": _relative(terminal_event_path),
            "sha256": str(terminal_registry_metadata["table_sha256"]),
        },
        {
            "role": "pit_etf_terminal_cash_events_manifest",
            "path": _relative(terminal_event_manifest_path),
            "sha256": str(terminal_registry_metadata["manifest_sha256"]),
        },
    ]
    for lifecycle in selected.itertuples(index=False):
        asset = _normalise_asset(lifecycle.asset)
        paths = _cache_paths(cache_dir, asset)
        for role in ("price", "dividend", "metadata", "failure"):
            path = paths[role]
            record: dict[str, Any] = {"role": f"etf_raw_{role}", "asset": asset, "path": _relative(path)}
            if path.is_file():
                record["sha256"] = _sha256(path)
                record["bytes"] = path.stat().st_size
            else:
                record["missing"] = True
            input_records.append(record)
    input_bundle_hash = hashlib.sha256(
        json.dumps(input_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    created_at = datetime.now().astimezone()
    run_id = f"{cutoff.strftime('%Y%m%d')}_{created_at.strftime('%Y%m%dT%H%M%S%f%z')}_{input_bundle_hash[:12]}"
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("etf_snapshot_builder.py").resolve(),
        Path(__file__).with_name("etf_corporate_actions.py").resolve(),
    ]
    outputs = [
        {"role": "etf_total_return_prices_observation", "path": _relative(price_output_path), "sha256": _sha256(price_output_path)},
        {"role": "etf_dividend_events_observation", "path": _relative(dividend_output_path), "sha256": _sha256(dividend_output_path)},
        {"role": "etf_lifecycle_asset_status", "path": _relative(status_output_path), "sha256": _sha256(status_output_path)},
    ]
    selected_terminal_events = terminal_events[terminal_events["asset"].isin(selected["asset"])].copy()
    applied_terminal_rows = int(successful["terminal_cash_events"].sum())
    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "selection_mode": selection_mode,
        "selection": {
            "explicit_assets": explicit_list or [],
            "earliest_limit": earliest_limit,
            "delisted_limit": delisted_limit,
            "offset": offset,
            "limit": limit,
        },
        "inputs": input_records,
        "input_bundle_sha256": input_bundle_hash,
        "code_files": [{"path": _relative(path), "sha256": _sha256(path)} for path in code_paths],
        "outputs": outputs,
        "selected_assets": int(len(selected)),
        "selected_active_assets": int(selected["lifecycle_status"].eq("active").sum()),
        "selected_delisted_assets": int(selected["lifecycle_status"].eq("delisted").sum()),
        "collection_successful_assets": int(sum(item.get("status") == "success" for item in collection_results.values())),
        "fetched_assets": int(sum(item.get("collection_action") == "fetched" for item in collection_results.values())),
        "reused_cache_assets": int(sum(item.get("collection_action") == "cache_reused" for item in collection_results.values())),
        "deferred_assets": int(sum(item.get("status") == "deferred" for item in collection_results.values())),
        "provider_circuit_breaker": circuit_breaker,
        "observation_ready_assets": int(len(successful)),
        "quarantined_assets": int(statuses["build_status"].eq("quarantined").sum()),
        "quarantined_asset_codes": statuses.loc[statuses["build_status"].eq("quarantined"), "asset"].tolist(),
        "price_rows": int(len(prices)),
        "dividend_event_rows": int(len(dividends)),
        "registered_corporate_action_rows": int(len(action_registry)),
        "registered_corporate_action_assets": int(action_registry["asset"].nunique()),
        "registered_terminal_cash_event_rows": int(len(terminal_events)),
        "registered_terminal_cash_event_assets": int(terminal_events["asset"].nunique()),
        "selected_registered_terminal_cash_event_rows": int(len(selected_terminal_events)),
        "applied_terminal_cash_event_rows": applied_terminal_rows,
        "assets_with_governed_terminal_cash_events": int(successful["terminal_cash_events"].gt(0).sum()),
        "known_terminal_cash_event_application_gate_passed": bool(
            applied_terminal_rows == len(selected_terminal_events)
        ),
        "terminal_cash_event_scope_complete": bool(terminal_registry_metadata["scope_complete"]),
        "terminal_cash_event_scope_boundary": str(terminal_registry_metadata["scope_boundary"]),
        "applied_corporate_action_rows": int(successful["applied_corporate_actions"].sum()),
        "governed_corporate_action_rows": int(successful["governed_corporate_actions"].sum()),
        "inferred_corporate_action_rows": int(successful["inferred_corporate_actions"].sum()),
        "assets_with_governed_corporate_actions": int(successful["governed_corporate_actions"].gt(0).sum()),
        "assets_with_inferred_corporate_actions": int(successful["inferred_corporate_actions"].gt(0).sum()),
        "corporate_action_evidence_gate_passed": bool(successful["inferred_corporate_actions"].eq(0).all()),
        "provider_tail_duplicate_rows_removed": int(successful["provider_tail_duplicate_rows_removed"].sum()),
        "assets_with_provider_tail_duplicate_resolution": int(
            successful["provider_tail_duplicate_rows_removed"].gt(0).sum()
        ),
        "maximum_distribution_alignment_lag_days": int(
            successful["maximum_distribution_alignment_lag_days"].max()
        ),
        "coverage_start": pd.Timestamp(prices["date"].min()).date().isoformat() if not prices.empty else None,
        "coverage_end": pd.Timestamp(prices["date"].max()).date().isoformat() if not prices.empty else None,
        "numeric_gate_thresholds": {
            "minimum_start_date": "2005-02-23",
            "minimum_end_date": "2026-06-30",
            "minimum_rows": 100000,
            "minimum_assets": 300,
            "shape_thresholds_met": bool(
                not prices.empty
                and pd.Timestamp(prices["date"].min()) <= pd.Timestamp("2005-02-23")
                and pd.Timestamp(prices["date"].max()) >= pd.Timestamp("2026-06-30")
                and len(prices) >= 100000
                and prices["asset"].nunique() >= 300
            ),
        },
        "qualification_status": QUALIFICATION_STATUS,
        "pit_actionable": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "source_quality_gate_passed": False,
        "limitations": [
            "selection uses an all-status lifecycle master, but the provider files are current final snapshots",
            "cumulative distribution rows lack independently verified first-publication and ex-date evidence",
            "share conversions use governed evidence where available; zero-marker common-factor inferences are explicitly tagged per asset",
            "assets with heuristic share-action inference remain observation-only and fail the corporate-action evidence gate",
            "exact weekend duplicates after governed delisting are removed with row-level diagnostics; other lifecycle conflicts are quarantined",
            "ordinary cash distributions align to the next observed trade within 31 calendar days",
            "governed terminal cash events are excluded from adjusted OHLC and represented separately as position-extinguishing events",
            "formal terminal cash-event coverage currently contains one known exception and is not exhaustive for all delisted ETFs",
            "unresolved price jumps and other lifecycle conflicts are quarantined per asset",
            "available_date is the source observation date, so these reconstructed histories are not historical signals",
            "formal promotion requires independent corporate-action validation and a frozen point-in-time source contract",
        ],
    }
    _atomic_json(payload, manifest_path)

    if archive:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_price = archive_dir / f"{run_id}_prices.csv.gz"
        archived_dividend = archive_dir / f"{run_id}_dividends.csv.gz"
        archived_status = archive_dir / f"{run_id}_status.csv"
        _copy_immutable(price_output_path, archived_price)
        _copy_immutable(dividend_output_path, archived_dividend)
        _copy_immutable(status_output_path, archived_status)
        payload["immutable_outputs"] = [
            {"role": "etf_total_return_prices_observation", "path": _relative(archived_price), "sha256": _sha256(archived_price)},
            {"role": "etf_dividend_events_observation", "path": _relative(archived_dividend), "sha256": _sha256(archived_dividend)},
            {"role": "etf_lifecycle_asset_status", "path": _relative(archived_status), "sha256": _sha256(archived_status)},
        ]
        immutable_manifest = archive_dir / f"{run_id}.json"
        payload["immutable_manifest_path"] = _relative(immutable_manifest)
        _atomic_json(payload, manifest_path)
        _atomic_json(payload, immutable_manifest)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--price-output", type=Path, default=DEFAULT_PRICE_OUTPUT)
    parser.add_argument("--dividend-output", type=Path, default=DEFAULT_DIVIDEND_OUTPUT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--action-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--terminal-events", type=Path, default=DEFAULT_TERMINAL_EVENT_PATH)
    parser.add_argument("--terminal-event-manifest", type=Path, default=DEFAULT_TERMINAL_EVENT_MANIFEST_PATH)
    parser.add_argument("--selection-mode", choices=("pilot", "all"), default="pilot")
    parser.add_argument("--assets", help="comma-separated governed ETF codes")
    parser.add_argument("--earliest-limit", type=int, default=10)
    parser.add_argument("--delisted-limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit = [item.strip() for item in args.assets.split(",") if item.strip()] if args.assets else None
    result = run(
        as_of=args.as_of,
        master_path=args.master,
        cache_dir=args.cache_dir,
        price_output_path=args.price_output,
        dividend_output_path=args.dividend_output,
        status_output_path=args.status_output,
        manifest_path=args.manifest,
        archive_dir=args.archive_dir,
        action_registry_path=args.action_registry,
        terminal_event_path=args.terminal_events,
        terminal_event_manifest_path=args.terminal_event_manifest,
        selection_mode=args.selection_mode,
        explicit_assets=explicit,
        earliest_limit=args.earliest_limit,
        delisted_limit=args.delisted_limit,
        offset=args.offset,
        limit=args.limit,
        refresh=args.refresh,
        attempts=args.attempts,
        sleep_seconds=args.sleep_seconds,
        archive=not args.no_archive,
    )
    keys = (
        "run_id",
        "selected_assets",
        "selected_active_assets",
        "selected_delisted_assets",
        "collection_successful_assets",
        "fetched_assets",
        "reused_cache_assets",
        "observation_ready_assets",
        "quarantined_assets",
        "price_rows",
        "dividend_event_rows",
        "coverage_start",
        "coverage_end",
        "qualification_status",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
