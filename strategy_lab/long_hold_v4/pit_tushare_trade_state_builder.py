"""Build and independently validate full-market stock trade state from Tushare daily files.

Tushare ``daily`` contains traded rows only. This builder expands each governed
security lifecycle over the exchange calendar, interprets missing rows as full-
session suspensions, applies validated status events, and derives explicit price
limit rules. The output is promoted only after BaoStock pause-state and JoinQuant
pause/ST/limit checks pass fixed thresholds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from .pit_stock_market_history_builder import (
    CHINEXT_20_PERCENT_EFFECTIVE_DATE,
    MAIN_BOARD_REGISTRATION_EFFECTIVE_DATE,
    ROOT,
    _board,
    _sha256,
)
from .pit_stock_market_history_validator import compare_joinquant_state


BACKTEST_START = pd.Timestamp("2000-01-01")
DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
CALENDAR_PATH = ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv"
STATUS_EVENTS_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "stock_execution_status_events_reconciled.csv"
)
STATUS_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_status_event_reconciler_latest.json"
STATUS_VALIDATION_MANIFEST_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "stock_status_event_validation" / "run_manifest.json"
)
HISTORICAL_ACCEPTANCE_DIR = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine"
)
HISTORICAL_ACCEPTANCE_MANIFEST = HISTORICAL_ACCEPTANCE_DIR / "agent_run_manifest.json"
HISTORICAL_ACCEPTANCE_REPORT = HISTORICAL_ACCEPTANCE_DIR / "file_acceptance_report.csv"
TUSHARE_REFRESH_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tushare_daily_refresh_latest.json"
BAOSTOCK_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_baostock_stock_daily"
BAOSTOCK_SUMMARY_PATH = BAOSTOCK_DIR / "derived_asset_summary.csv"
JQ_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "joinquant_trade_state"
)
DERIVATION_HELPER_PATH = Path(__file__).with_name("pit_stock_market_history_builder.py")
VALIDATION_HELPER_PATH = Path(__file__).with_name("pit_stock_market_history_validator.py")

FORMAL_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_trade_state.csv.gz"
OBSERVATION_OUTPUT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "stock_trade_state_tushare_progress.csv.gz"
)
RAW_INVENTORY_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tushare_trade_state_raw_inventory.csv"
BUILDER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_trade_state_builder_latest.json"

VALIDATION_DIR = ROOT / "outputs" / "long_hold_v4" / "tushare_trade_state_validation"
SAMPLE_OUTPUT = VALIDATION_DIR / "trade_state_validation_sample.csv.gz"
BAOSTOCK_CHECKS_OUTPUT = VALIDATION_DIR / "baostock_pause_checks.csv.gz"
JQ_CHECKS_OUTPUT = VALIDATION_DIR / "joinquant_trade_state_checks.csv.gz"
SOURCE_OBSERVATIONS_OUTPUT = VALIDATION_DIR / "source_observations.csv"
WARNINGS_OUTPUT = VALIDATION_DIR / "validation_warnings.csv"
EXCEPTIONS_OUTPUT = VALIDATION_DIR / "validation_exceptions.csv"
VALIDATION_REPORT_PATH = VALIDATION_DIR / "validation_report.json"
VALIDATION_MANIFEST_PATH = VALIDATION_DIR / "run_manifest.json"

BUILDER_VERSION = "tushare_full_market_trade_state_v1"
VALIDATION_SCHEMA = "cross_provider_tushare_trade_state_v2"
SOURCE_NAME = "tushare.daily+governed_exchange_status_events+explicit_exchange_rules"

OUTPUT_COLUMNS = [
    "date",
    "asset",
    "is_paused",
    "is_st",
    "pre_close",
    "has_price_limit",
    "limit_up",
    "limit_down",
    "price_limit_rate",
    "limit_rule",
    "execution_state_known",
    "available_date",
    "data_source",
    "source_vintage",
]

THRESHOLDS: dict[str, float | int] = {
    "maximum_unknown_execution_state_ratio": 0.03,
    "baostock_min_pause_checks": 750_000,
    "baostock_pause_match_min": 0.995,
    "joinquant_min_state_checks": 15_000,
    "joinquant_min_st_checks": 15_000,
    "joinquant_paused_match_min": 0.995,
    "joinquant_st_match_min": 0.995,
    "joinquant_min_limit_checks": 10_000,
    "joinquant_limit_match_min": 0.995,
}


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


class _DeterministicGzipCsvWriter:
    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self._raw = self.temporary.open("wb")
        self._gzip = gzip.GzipFile(filename="", fileobj=self._raw, mode="wb", compresslevel=6, mtime=0)
        self._text: TextIO = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="")
        self._header = True
        self._closed = False
        self.rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed gzip writer")
        if frame.empty:
            return
        missing = sorted(set(self.columns).difference(frame.columns))
        if missing:
            raise ValueError(f"gzip writer missing columns: {missing}")
        frame[self.columns].to_csv(
            self._text,
            index=False,
            header=self._header,
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
        self._header = False
        self.rows += len(frame)

    def close(self, destination: Path | None = None) -> Path:
        if self._closed:
            raise RuntimeError("gzip writer is already closed")
        if self._header:
            pd.DataFrame(columns=self.columns).to_csv(self._text, index=False, lineterminator="\n")
        self._text.flush()
        self._text.close()
        self._raw.close()
        self._closed = True
        target = destination or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.replace(target)
        return target

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._text.close()
        finally:
            self._raw.close()
            self._closed = True
            self.temporary.unlink(missing_ok=True)


def load_calendar(as_of: str | pd.Timestamp) -> pd.DatetimeIndex:
    as_of_date = pd.Timestamp(as_of).normalize()
    frame = pd.read_csv(CALENDAR_PATH)
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values().unique())
    dates = dates[(dates >= BACKTEST_START) & (dates <= as_of_date)]
    if dates.empty or dates[-1] != as_of_date:
        raise ValueError(f"as-of date is not the final governed market session: {as_of_date.date()}")
    return dates


def load_lifecycles(as_of: str | pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of).normalize()
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    listings = master[master["event_type"].eq("listing")][
        ["asset", "exchange", "list_date", "predecessor_asset"]
    ].copy()
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listings["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("security master contains duplicate lifecycle events")
    frame = listings.merge(exits, on="asset", how="left", validate="one_to_one")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce").dt.normalize()
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce").dt.normalize()
    frame = frame[
        frame["list_date"].le(as_of_date)
        & (frame["delist_date"].isna() | frame["delist_date"].ge(BACKTEST_START))
    ].copy()
    if frame[["asset", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("security master contains incomplete lifecycle keys")
    frame["start_index"] = np.searchsorted(
        calendar.values,
        frame["list_date"].clip(lower=BACKTEST_START).to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    lifecycle_end = frame["delist_date"].fillna(as_of_date).clip(upper=as_of_date)
    frame["end_index"] = (
        np.searchsorted(calendar.values, lifecycle_end.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    )
    predecessor = frame["predecessor_asset"].fillna("").astype(str).str.strip()
    frame["is_ipo"] = predecessor.eq("")
    frame = frame[frame["end_index"].ge(frame["start_index"])].copy()
    return frame.sort_values("asset").reset_index(drop=True)


def load_validated_status_events(as_of: str | pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    as_of_date = pd.Timestamp(as_of).normalize()
    status_manifest = json.loads(STATUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    validation_manifest = json.loads(STATUS_VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if status_manifest.get("as_of_date") != as_of_date.date().isoformat():
        raise ValueError("status reconciler as-of mismatch")
    if status_manifest.get("qualification_status") != "READY_FOR_CROSS_SOURCE_VALIDATION":
        raise ValueError("status reconciler is incomplete")
    if status_manifest.get("output_path") != _relative(STATUS_EVENTS_PATH):
        raise ValueError("status-event output path mismatch")
    if status_manifest.get("output_sha256") != _sha256(STATUS_EVENTS_PATH):
        raise ValueError("status-event output hash mismatch")
    if validation_manifest.get("as_of_date") != as_of_date.date().isoformat():
        raise ValueError("status validation as-of mismatch")
    if validation_manifest.get("qualification_status") != "PASS":
        raise ValueError("status events have not passed independent validation")
    if validation_manifest.get("historical_backtest_allowed") is not True:
        raise ValueError("status validation does not permit historical use")
    validation_inputs = {
        (str(item.get("role")), str(item.get("path"))): str(item.get("sha256"))
        for item in validation_manifest.get("inputs", [])
        if isinstance(item, dict)
    }
    if validation_inputs.get(("status_events", _relative(STATUS_EVENTS_PATH))) != _sha256(STATUS_EVENTS_PATH):
        raise ValueError("status validation does not bind the current event output")
    events = pd.read_csv(STATUS_EVENTS_PATH, dtype={"asset": str})
    events["asset"] = events["asset"].astype(str).str.zfill(6)
    events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
    events["available_date"] = pd.to_datetime(events["available_date"], errors="coerce").dt.normalize()
    if events[["asset", "effective_date", "execution_status", "available_date"]].isna().any(axis=None):
        raise ValueError("validated status events contain incomplete keys")
    if events.duplicated(["asset", "effective_date"]).any():
        raise ValueError("validated status events contain duplicate asset dates")
    if events["available_date"].gt(events["effective_date"]).any():
        raise ValueError("validated status events contain future availability")
    inputs = [
        {"role": "status_events", "path": _relative(STATUS_EVENTS_PATH), "sha256": _sha256(STATUS_EVENTS_PATH)},
        {"role": "status_reconciler_manifest", "path": _relative(STATUS_MANIFEST_PATH), "sha256": _sha256(STATUS_MANIFEST_PATH)},
        {"role": "status_validation_manifest", "path": _relative(STATUS_VALIDATION_MANIFEST_PATH), "sha256": _sha256(STATUS_VALIDATION_MANIFEST_PATH)},
    ]
    return events.sort_values(["effective_date", "asset"]).reset_index(drop=True), inputs


def build_status_schedule(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> dict[int, list[tuple[str, str]]]:
    schedule: dict[int, list[tuple[str, str]]] = defaultdict(list)
    positions = np.searchsorted(
        calendar.values,
        events["effective_date"].to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    for event, position in zip(events.itertuples(index=False), positions, strict=True):
        if int(position) < len(calendar):
            schedule[int(position)].append((str(event.asset).zfill(6), str(event.execution_status)))
    return schedule


def build_raw_inventory(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date in calendar:
        trade_date = pd.Timestamp(date).strftime("%Y%m%d")
        path = DAILY_DIR / f"trade_date={trade_date}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "trade_date": trade_date,
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, RAW_INVENTORY_PATH)
    return frame


def validate_raw_governance(
    calendar: pd.DatetimeIndex,
    inventory: pd.DataFrame,
) -> list[dict[str, str]]:
    acceptance = json.loads(HISTORICAL_ACCEPTANCE_MANIFEST.read_text(encoding="utf-8"))
    if acceptance.get("acceptance_pass") is not True or acceptance.get("self_check_pass") is not True:
        raise ValueError("historical Tushare daily acceptance did not pass")
    report = pd.read_csv(HISTORICAL_ACCEPTANCE_REPORT, dtype={"trade_date": str})
    report["trade_date"] = report["trade_date"].astype(str).str.zfill(8)
    historical_end = report["trade_date"].max()
    required_historical = {date.strftime("%Y%m%d") for date in calendar if date.strftime("%Y%m%d") <= historical_end}
    accepted = set(
        report.loc[
            report["status"].eq("acquired")
            & report["readable"].astype(str).str.lower().eq("true")
            & pd.to_numeric(report["duplicate_asset_date_count"], errors="coerce").fillna(1).eq(0),
            "trade_date",
        ]
    )
    missing_historical = sorted(required_historical.difference(accepted))
    if missing_historical:
        raise ValueError(f"historical Tushare acceptance misses governed sessions: {missing_historical[:10]}")

    refresh = json.loads(TUSHARE_REFRESH_MANIFEST.read_text(encoding="utf-8"))
    if refresh.get("qualification_status") != "REFRESH_COMPLETE_RAW_DAILY_ONLY":
        raise ValueError("Tushare recent daily refresh is incomplete")
    refresh_hashes = {str(item["trade_date"]): str(item["sha256"]) for item in refresh.get("outputs", [])}
    inventory_hashes = inventory.set_index("trade_date")["sha256"].astype(str).to_dict()
    refresh_failures = [date for date, digest in refresh_hashes.items() if inventory_hashes.get(date) != digest]
    if refresh_failures:
        raise ValueError(f"Tushare refresh hashes do not match raw inventory: {refresh_failures[:10]}")
    final_session = calendar[-1].strftime("%Y%m%d")
    if final_session not in refresh_hashes:
        raise ValueError("Tushare refresh does not cover the final governed session")
    return [
        {"role": "historical_daily_acceptance_manifest", "path": _relative(HISTORICAL_ACCEPTANCE_MANIFEST), "sha256": _sha256(HISTORICAL_ACCEPTANCE_MANIFEST)},
        {"role": "historical_daily_acceptance_report", "path": _relative(HISTORICAL_ACCEPTANCE_REPORT), "sha256": _sha256(HISTORICAL_ACCEPTANCE_REPORT)},
        {"role": "tushare_daily_refresh_manifest", "path": _relative(TUSHARE_REFRESH_MANIFEST), "sha256": _sha256(TUSHARE_REFRESH_MANIFEST)},
        {"role": "tushare_trade_state_raw_inventory", "path": _relative(RAW_INVENTORY_PATH), "sha256": _sha256(RAW_INVENTORY_PATH)},
    ]


def normalise_tushare_session(raw: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    required = ["ts_code", "trade_date", "close", "pre_close"]
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise ValueError(f"Tushare daily session missing columns: {missing}")
    frame = raw[required].copy()
    frame["asset"] = frame["ts_code"].astype(str).str.extract(r"^(\d{6})\.(?:SH|SZ)$", expand=False)
    frame = frame.dropna(subset=["asset"]).copy()
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    expected_date = date.strftime("%Y%m%d")
    if not frame["trade_date"].eq(expected_date).all():
        raise ValueError(f"Tushare file contains a different trade date: {expected_date}")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    if frame[["close", "pre_close"]].isna().any(axis=None) or (frame[["close", "pre_close"]] <= 0).any(axis=None):
        raise ValueError(f"Tushare session contains invalid close/pre_close: {expected_date}")
    if frame["asset"].duplicated().any():
        raise ValueError(f"Tushare session contains duplicate A-share assets: {expected_date}")
    return frame[["asset", "pre_close", "close"]].sort_values("asset").reset_index(drop=True)


def _round_half_up_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return np.floor(numeric * 100.0 + 0.5 + 1e-10) / 100.0


def derive_session_trade_state(
    active: pd.DataFrame,
    traded: pd.DataFrame,
    *,
    date: pd.Timestamp,
    session_index: int,
    current_status: dict[str, str],
    previous_paused: set[str],
    last_close: dict[str, float],
    source_vintage: str,
) -> pd.DataFrame:
    frame = active[["asset", "list_date", "start_index", "is_ipo"]].copy()
    frame = frame.merge(traded, on="asset", how="left", validate="one_to_one")
    frame["status"] = frame["asset"].map(current_status).fillna("unknown")
    frame["is_paused"] = frame["close"].isna()
    frame["is_st"] = frame["status"].isin({"risk_warning", "special_transfer"})
    paused_preclose = frame["asset"].map(last_close)
    frame["pre_close"] = frame["pre_close"].where(~frame["is_paused"], paused_preclose)
    frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["listing_market_session"] = session_index - frame["start_index"].astype(int) + 1
    frame["previous_paused"] = frame["asset"].isin(previous_paused)
    frame["board"] = frame["asset"].map(_board)

    frame["has_price_limit"] = False
    frame["limit_up"] = np.nan
    frame["limit_down"] = np.nan
    frame["price_limit_rate"] = np.nan
    frame["limit_rule"] = "paused"
    frame["execution_state_known"] = True

    traded_mask = ~frame["is_paused"]
    status_unknown = frame["status"].eq("unknown")
    frame.loc[status_unknown, ["limit_rule", "execution_state_known"]] = ["status_unknown", False]
    status_trade_conflict = traded_mask & frame["status"].eq("listing_suspended")
    frame.loc[status_trade_conflict, ["limit_rule", "execution_state_known"]] = ["status_trade_conflict", False]
    special_transfer = traded_mask & frame["status"].eq("special_transfer")
    frame.loc[special_transfer, ["limit_rule", "execution_state_known"]] = [
        "legacy_special_transfer_unknown",
        False,
    ]
    delisting = traded_mask & frame["status"].eq("delisting")
    frame.loc[delisting, ["limit_rule", "execution_state_known"]] = ["delisting_limit_unknown", False]

    unresolved = status_unknown | status_trade_conflict | special_transfer | delisting
    no_limit_window = (
        traded_mask
        & ~unresolved
        & frame["is_ipo"].astype(bool)
        & frame["listing_market_session"].le(5)
        & (
            frame["board"].eq("star")
            | (frame["board"].eq("chinext") & frame["list_date"].ge(CHINEXT_20_PERCENT_EFFECTIVE_DATE))
            | (frame["board"].eq("main") & frame["list_date"].ge(MAIN_BOARD_REGISTRATION_EFFECTIVE_DATE))
        )
    )
    frame.loc[no_limit_window, "limit_rule"] = "no_price_limit_listing_window"

    first_special = (
        traded_mask
        & ~unresolved
        & ~no_limit_window
        & frame["listing_market_session"].eq(1)
    )
    frame.loc[first_special, ["limit_rule", "execution_state_known"]] = ["first_session_special_unknown", False]
    resumption = (
        traded_mask
        & ~unresolved
        & ~no_limit_window
        & ~first_special
        & frame["previous_paused"]
    )
    frame.loc[resumption, ["limit_rule", "execution_state_known"]] = ["resumption_limit_unknown", False]
    invalid_preclose = (
        traded_mask
        & ~unresolved
        & ~no_limit_window
        & ~first_special
        & ~resumption
        & (~np.isfinite(frame["pre_close"]) | frame["pre_close"].le(0))
    )
    frame.loc[invalid_preclose, ["limit_rule", "execution_state_known"]] = ["invalid_preclose", False]

    regular = traded_mask & ~unresolved & ~no_limit_window & ~first_special & ~resumption & ~invalid_preclose
    growth_20 = frame["board"].eq("star") | (
        frame["board"].eq("chinext") & (date >= CHINEXT_20_PERCENT_EFFECTIVE_DATE)
    )
    rate = pd.Series(np.where(growth_20, 0.20, np.where(frame["is_st"], 0.05, 0.10)), index=frame.index)
    rule = pd.Series(
        np.where(
            growth_20,
            "regular_growth_20",
            np.where(frame["is_st"], "regular_st_5", "regular_main_or_pre_reform_growth_10"),
        ),
        index=frame.index,
    )
    frame.loc[regular, "has_price_limit"] = True
    frame.loc[regular, "price_limit_rate"] = rate.loc[regular]
    frame.loc[regular, "limit_rule"] = rule.loc[regular]
    frame.loc[regular, "limit_up"] = _round_half_up_positive(
        frame.loc[regular, "pre_close"] * (1.0 + rate.loc[regular])
    )
    frame.loc[regular, "limit_down"] = _round_half_up_positive(
        frame.loc[regular, "pre_close"] * (1.0 - rate.loc[regular])
    )

    frame["date"] = date
    frame["available_date"] = date
    frame["data_source"] = SOURCE_NAME
    frame["source_vintage"] = source_vintage
    output = frame[OUTPUT_COLUMNS].sort_values("asset").reset_index(drop=True)
    if output.duplicated(["date", "asset"]).any():
        raise ValueError(f"derived trade-state session contains duplicate keys: {date.date()}")
    return output


def _load_validation_assets() -> tuple[set[str], set[str]]:
    summary = pd.read_csv(BAOSTOCK_SUMMARY_PATH, dtype={"asset": str})
    baostock = set(summary.loc[summary["trade_status"].eq("completed"), "asset"].astype(str).str.zfill(6))
    joinquant = {path.name[:6] for path in JQ_DIR.glob("[0-9][0-9][0-9][0-9][0-9][0-9]_*.csv.gz")}
    if not baostock or not joinquant:
        raise ValueError("cross-provider validation assets are unavailable")
    return baostock, joinquant


def _load_baostock_pause(assets: set[str]) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for asset in sorted(assets):
        path = BAOSTOCK_DIR / f"{asset}.csv.gz"
        try:
            frame = pd.read_csv(path, compression="gzip", usecols=["date", "tradestatus"])
            frame.insert(0, "asset", asset)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame["actual_is_paused"] = pd.to_numeric(frame["tradestatus"], errors="coerce").eq(0)
            frames.append(frame[["date", "asset", "actual_is_paused"]])
            inputs.append({"role": "baostock_pause_history", "path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append({"source": "baostock", "asset": asset, "error": f"{type(exc).__name__}: {exc}"})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs, errors


def _load_joinquant_state(assets: set[str]) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(JQ_DIR.glob("[0-9][0-9][0-9][0-9][0-9][0-9]_*.csv.gz")):
        asset = path.name[:6]
        if asset not in assets:
            continue
        try:
            frame = pd.read_csv(path, compression="gzip", dtype={"asset": str})
            frame["asset"] = frame["asset"].astype(str).str.zfill(6)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frames.append(frame)
            inputs.append({"role": "joinquant_trade_state", "path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append({"source": "joinquant", "asset": asset, "error": f"{type(exc).__name__}: {exc}"})
    if not frames:
        return pd.DataFrame(), inputs, errors
    output = pd.concat(frames, ignore_index=True)
    duplicate = output.duplicated(["date", "asset"], keep=False)
    if duplicate.any():
        columns = ["close", "pre_close", "high_limit", "low_limit", "paused", "is_st"]
        conflicts = output.loc[duplicate].groupby(["date", "asset"])[columns].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError("JoinQuant validation caches contain conflicting duplicate keys")
        output = output.drop_duplicates(["date", "asset"], keep="last")
    return output.sort_values(["date", "asset"]).reset_index(drop=True), inputs, errors


def validate_cross_provider_sample(
    sample_path: Path,
    baostock_assets: set[str],
    jq_assets: set[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    sample = pd.read_csv(sample_path, compression="gzip", dtype={"asset": str}, low_memory=False)
    sample["asset"] = sample["asset"].astype(str).str.zfill(6)
    sample["date"] = pd.to_datetime(sample["date"], errors="coerce").dt.normalize()
    baostock, baostock_inputs, baostock_errors = _load_baostock_pause(baostock_assets)
    baostock_checks = sample[sample["asset"].isin(baostock_assets)][["date", "asset", "is_paused"]].merge(
        baostock,
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    baostock_checks["pause_match"] = (
        baostock_checks["is_paused"].astype(str).str.lower().isin({"true", "1"})
        == baostock_checks["actual_is_paused"].astype(bool)
    )
    baostock_metrics = {
        "assets": int(baostock_checks["asset"].nunique()) if not baostock_checks.empty else 0,
        "pause_checks": int(len(baostock_checks)),
        "paused_rows": int(baostock_checks["actual_is_paused"].sum()) if not baostock_checks.empty else 0,
        "pause_match_ratio": round(float(baostock_checks["pause_match"].mean()), 8) if len(baostock_checks) else 0.0,
    }
    _atomic_csv(baostock_checks, BAOSTOCK_CHECKS_OUTPUT)

    jq, jq_inputs, jq_errors = _load_joinquant_state(jq_assets)
    jq_checks, jq_metrics = compare_joinquant_state(sample[sample["asset"].isin(jq_assets)], jq)
    _atomic_csv(jq_checks, JQ_CHECKS_OUTPUT)
    return {
        "baostock": baostock_metrics,
        "joinquant": jq_metrics,
        "source_errors": [*baostock_errors, *jq_errors],
    }, [*baostock_inputs, *jq_inputs]


def _qualification_checks(
    *,
    rows: int,
    expected_rows: int,
    assets: int,
    target_assets: int,
    unknown_ratio: float,
    paused_rows: int,
    st_rows: int,
    limited_rows: int,
    cross: dict[str, Any],
) -> pd.DataFrame:
    baostock = cross["baostock"]
    jq = cross["joinquant"]
    checks = {
        "row_count_exact": rows == expected_rows,
        "asset_coverage_exact": assets == target_assets,
        "unknown_execution_ratio": math.isfinite(unknown_ratio)
        and unknown_ratio <= THRESHOLDS["maximum_unknown_execution_state_ratio"],
        "paused_population": paused_rows > 0,
        "st_population": st_rows > 0,
        "price_limit_population": limited_rows > 0,
        "baostock_pause_population": baostock["pause_checks"] >= THRESHOLDS["baostock_min_pause_checks"],
        "baostock_pause_match": baostock["pause_match_ratio"] >= THRESHOLDS["baostock_pause_match_min"],
        "joinquant_state_population": jq["state_checks"] >= THRESHOLDS["joinquant_min_state_checks"],
        "joinquant_paused_match": jq["paused_match_ratio"] >= THRESHOLDS["joinquant_paused_match_min"],
        "joinquant_st_population": jq["st_checks"] >= THRESHOLDS["joinquant_min_st_checks"],
        "joinquant_st_match": jq["st_match_ratio"] >= THRESHOLDS["joinquant_st_match_min"],
        "joinquant_limit_population": jq["limit_checks"] >= THRESHOLDS["joinquant_min_limit_checks"],
        "joinquant_limit_match": jq["limit_match_ratio"] >= THRESHOLDS["joinquant_limit_match_min"],
        "source_read_errors": len(cross["source_errors"]) == 0,
    }
    return pd.DataFrame(
        [{"check_id": check_id, "passed": bool(passed)} for check_id, passed in checks.items()]
    )


def run_build(as_of: str) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    calendar = load_calendar(as_of_date)
    lifecycles = load_lifecycles(as_of_date, calendar)
    events, status_inputs = load_validated_status_events(as_of_date)
    inventory = build_raw_inventory(calendar)
    raw_governance_inputs = validate_raw_governance(calendar, inventory)
    fixed_inputs = [
        {"role": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"role": "trade_calendar", "path": _relative(CALENDAR_PATH), "sha256": _sha256(CALENDAR_PATH)},
        {
            "role": "trade_state_derivation_helper",
            "path": _relative(DERIVATION_HELPER_PATH),
            "sha256": _sha256(DERIVATION_HELPER_PATH),
        },
        *status_inputs,
        *raw_governance_inputs,
    ]
    bundle_hash = hashlib.sha256(
        "|".join(f"{item['role']}:{item['sha256']}" for item in fixed_inputs).encode()
    ).hexdigest()
    source_vintage = f"tushare_trade_state_bundle_sha256:{bundle_hash}"

    status_schedule = build_status_schedule(events, calendar)
    current_status: dict[str, str] = {}
    previous_paused: set[str] = set()
    last_close: dict[str, float] = {}
    baostock_assets, jq_assets = _load_validation_assets()
    sample_assets = baostock_assets | jq_assets
    expected_rows = int((lifecycles["end_index"] - lifecycles["start_index"] + 1).sum())
    target_assets = int(len(lifecycles))

    output_writer = _DeterministicGzipCsvWriter(OBSERVATION_OUTPUT, OUTPUT_COLUMNS)
    sample_writer = _DeterministicGzipCsvWriter(SAMPLE_OUTPUT, OUTPUT_COLUMNS)
    assets_seen: set[str] = set()
    rule_counts: Counter[str] = Counter()
    paused_rows = 0
    st_rows = 0
    limited_rows = 0
    unknown_rows = 0
    raw_outside_lifecycle_rows = 0
    try:
        for index, date in enumerate(calendar):
            for asset, status in status_schedule.get(index, []):
                current_status[asset] = status
            active = lifecycles[
                lifecycles["start_index"].le(index) & lifecycles["end_index"].ge(index)
            ].copy()
            path = DAILY_DIR / f"trade_date={date.strftime('%Y%m%d')}.csv"
            raw = pd.read_csv(
                path,
                usecols=["ts_code", "trade_date", "close", "pre_close"],
                dtype={"ts_code": str, "trade_date": str},
                low_memory=False,
            )
            traded_all = normalise_tushare_session(raw, pd.Timestamp(date))
            traded = traded_all[traded_all["asset"].isin(set(active["asset"]))].copy()
            raw_outside_lifecycle_rows += len(traded_all) - len(traded)
            session = derive_session_trade_state(
                active,
                traded,
                date=pd.Timestamp(date),
                session_index=index,
                current_status=current_status,
                previous_paused=previous_paused,
                last_close=last_close,
                source_vintage=source_vintage,
            )
            output_writer.append(session)
            sample_writer.append(session[session["asset"].isin(sample_assets)])
            assets_seen.update(session["asset"])
            paused_rows += int(session["is_paused"].astype(bool).sum())
            st_rows += int(session["is_st"].astype(bool).sum())
            limited_rows += int(session["has_price_limit"].astype(bool).sum())
            unknown_rows += int((~session["execution_state_known"].astype(bool)).sum())
            rule_counts.update(session["limit_rule"].astype(str))
            previous_paused = set(session.loc[session["is_paused"].astype(bool), "asset"].astype(str))
            for row in traded.itertuples(index=False):
                last_close[str(row.asset)] = float(row.close)
        sample_writer.close(SAMPLE_OUTPUT)
        cross, validation_inputs = validate_cross_provider_sample(
            SAMPLE_OUTPUT,
            baostock_assets,
            jq_assets,
        )
        unknown_ratio = unknown_rows / output_writer.rows if output_writer.rows else 1.0
        qualification = _qualification_checks(
            rows=output_writer.rows,
            expected_rows=expected_rows,
            assets=len(assets_seen),
            target_assets=target_assets,
            unknown_ratio=unknown_ratio,
            paused_rows=paused_rows,
            st_rows=st_rows,
            limited_rows=limited_rows,
            cross=cross,
        )
        passed = bool(qualification["passed"].all())
        destination = FORMAL_OUTPUT if passed else OBSERVATION_OUTPUT
        output_writer.close(destination)
    except BaseException:
        output_writer.abort()
        sample_writer.abort()
        raise

    warnings = pd.DataFrame(
        [
            {
                "warning_id": "unknown_execution_rules",
                "count": unknown_rows,
                "detail": json.dumps(
                    {key: value for key, value in sorted(rule_counts.items()) if "unknown" in key},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "warning_id": "raw_rows_outside_governed_lifecycle",
                "count": raw_outside_lifecycle_rows,
                "detail": "excluded from formal stock lifecycle output",
            },
        ]
    )
    exceptions = pd.DataFrame(
        [
            {"exception_id": row.check_id, "detail": "qualification check failed"}
            for row in qualification.loc[~qualification["passed"]].itertuples(index=False)
        ]
        + [
            {"exception_id": f"{item['source']}:{item['asset']}", "detail": item["error"]}
            for item in cross["source_errors"]
        ],
        columns=["exception_id", "detail"],
    )
    observations = pd.DataFrame(
        [
            {"metric": "rows", "value": output_writer.rows},
            {"metric": "expected_rows", "value": expected_rows},
            {"metric": "assets", "value": len(assets_seen)},
            {"metric": "target_assets", "value": target_assets},
            {"metric": "paused_rows", "value": paused_rows},
            {"metric": "st_rows", "value": st_rows},
            {"metric": "limited_rows", "value": limited_rows},
            {"metric": "unknown_rows", "value": unknown_rows},
            {"metric": "unknown_ratio", "value": unknown_ratio},
            {"metric": "raw_outside_lifecycle_rows", "value": raw_outside_lifecycle_rows},
        ]
    )
    _atomic_csv(observations, SOURCE_OBSERVATIONS_OUTPUT)
    _atomic_csv(warnings, WARNINGS_OUTPUT)
    _atomic_csv(exceptions, EXCEPTIONS_OUTPUT)

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "validation_schema": VALIDATION_SCHEMA,
        "qualification_status": "PASS" if passed else "FAIL",
        "historical_backtest_allowed": passed,
        "model_promotion_allowed": False,
        "thresholds": THRESHOLDS,
        "rows": output_writer.rows,
        "expected_rows": expected_rows,
        "assets": len(assets_seen),
        "target_assets": target_assets,
        "coverage_start": calendar[0].date().isoformat(),
        "coverage_end": calendar[-1].date().isoformat(),
        "paused_rows": paused_rows,
        "st_rows": st_rows,
        "limited_rows": limited_rows,
        "unknown_execution_rows": unknown_rows,
        "unknown_execution_ratio": round(unknown_ratio, 8),
        "limit_rule_counts": dict(sorted(rule_counts.items())),
        "raw_outside_lifecycle_rows": raw_outside_lifecycle_rows,
        "baostock": cross["baostock"],
        "joinquant": cross["joinquant"],
        "failed_checks": qualification.loc[~qualification["passed"], "check_id"].tolist(),
        "source_errors": cross["source_errors"],
        "output_path": _relative(destination),
        "output_sha256": _sha256(destination),
        "evidence_boundary": {
            "price_source": "unadjusted Tushare daily is used only for execution state and exchange reference-price limits",
            "suspension": "missing active-lifecycle row on a governed exchange session, independently checked against BaoStock and JoinQuant",
            "status": "uses the separately qualified status-event chain",
            "promotion": "PASS qualifies stock trade state only; it does not qualify valuation, fundamentals, returns, or the model",
        },
    }
    _atomic_json(report, VALIDATION_REPORT_PATH)
    _atomic_csv(qualification, VALIDATION_DIR / "qualification_checks.csv")

    output_roles = {
        "baostock_checks": BAOSTOCK_CHECKS_OUTPUT,
        "joinquant_checks": JQ_CHECKS_OUTPUT,
        "source_observations": SOURCE_OBSERVATIONS_OUTPUT,
        "warnings": WARNINGS_OUTPUT,
        "exceptions": EXCEPTIONS_OUTPUT,
        "report": VALIDATION_REPORT_PATH,
        "qualification_checks": VALIDATION_DIR / "qualification_checks.csv",
        "validation_sample": SAMPLE_OUTPUT,
    }
    validation_manifest = {
        "created_at": report["created_at"],
        "as_of_date": report["as_of_date"],
        "validation_schema": VALIDATION_SCHEMA,
        "qualification_status": report["qualification_status"],
        "historical_backtest_allowed": passed,
        "model_promotion_allowed": False,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "inputs": [
            {
                "role": "stock_trade_state",
                "path": _relative(destination),
                "sha256": _sha256(destination),
            },
            *fixed_inputs,
            {
                "role": "cross_provider_validation_helper",
                "path": _relative(VALIDATION_HELPER_PATH),
                "sha256": _sha256(VALIDATION_HELPER_PATH),
            },
            *validation_inputs,
        ],
        "outputs": [
            {"role": role, "path": _relative(path), "sha256": _sha256(path)}
            for role, path in output_roles.items()
        ],
        "failed_checks": report["failed_checks"],
        "exception_count": int(len(exceptions)),
    }
    _atomic_json(validation_manifest, VALIDATION_MANIFEST_PATH)

    builder_manifest = {
        "created_at": report["created_at"],
        "builder_version": BUILDER_VERSION,
        "as_of_date": report["as_of_date"],
        "effective_market_date": report["coverage_end"],
        "source_vintage": source_vintage,
        "inputs": fixed_inputs,
        "output_path": _relative(destination),
        "output_sha256": report["output_sha256"],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": output_writer.rows,
        "assets": len(assets_seen),
        "target_assets": target_assets,
        "asset_coverage": round(len(assets_seen) / target_assets, 8) if target_assets else 0.0,
        "internal_quality_pass": passed,
        "qualification_status": "PASS" if passed else "COLLECTION_OR_VALIDATION_INCOMPLETE",
        "historical_backtest_allowed": passed,
        "model_promotion_allowed": False,
        "validation_manifest_path": _relative(VALIDATION_MANIFEST_PATH),
        "validation_manifest_sha256": _sha256(VALIDATION_MANIFEST_PATH),
        "output_compression": "deterministic_gzip_mtime_0",
        "output_sort_order": ["date", "asset"],
        "qualification_blockers": report["failed_checks"],
    }
    _atomic_json(builder_manifest, BUILDER_MANIFEST_PATH)
    return {"builder": builder_manifest, "validation": report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_build(args.as_of)
    print(
        json.dumps(
            {
                "qualification_status": result["validation"]["qualification_status"],
                "historical_backtest_allowed": result["validation"]["historical_backtest_allowed"],
                "rows": result["validation"]["rows"],
                "assets": result["validation"]["assets"],
                "unknown_execution_ratio": result["validation"]["unknown_execution_ratio"],
                "baostock": result["validation"]["baostock"],
                "joinquant": result["validation"]["joinquant"],
                "failed_checks": result["validation"]["failed_checks"],
                "output_path": result["validation"]["output_path"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
