"""Collect and normalize lifecycle-bounded stock market-state history.

BaoStock supplies long-history raw daily records with PE/PB, ST and trading
status. The collector keeps one compressed raw file per security so collection
can be resumed or sharded. Partial runs only write observation outputs. Formal
Gate E2 targets are written after complete lifecycle coverage and internal
quality checks; a separate cross-provider validator is still required by the
gate before the datasets can pass.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import socket
import time
import zlib
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
DIVIDEND_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_dividend_events.csv"
TRADE_CALENDAR_PATH = ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_baostock_stock_daily"
COLLECTION_INVENTORY = RAW_DIR / "collection_inventory.csv"
ASSET_SUMMARY = RAW_DIR / "derived_asset_summary.csv"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
TRADE_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_trade_state.csv"
VALUATION_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_valuation_history.csv"
TRADE_OBSERVATION = OBSERVATION_DIR / "stock_trade_state_baostock_progress.csv"
VALUATION_OBSERVATION = OBSERVATION_DIR / "stock_valuation_history_baostock_progress.csv"
COMBINED_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_market_history_builder_latest.json"
TRADE_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_trade_state_builder_latest.json"
VALUATION_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_valuation_builder_latest.json"

SOURCE_HOST = "public-api.baostock.com"
SOURCE_PORT = 10030
SOURCE_NAME = "baostock.query_history_k_data_plus"
PYPI_PACKAGE_VERSION = "0.8.9"
SERVER_PROTOCOL_VERSION = "00.9.10"
SOURCE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)
REQUIRED_HISTORY_COLUMNS = set(SOURCE_FIELDS.split(","))
BACKTEST_START = pd.Timestamp("2005-01-01")
MAX_UNKNOWN_EXECUTION_RATIO = 0.03
MIN_VALUATION_COVERAGE = 0.95
MAX_SAFE_COLLECTION_WORKERS = 1
MAX_SOURCE_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_HISTORY_ROWS = 15_000
TRADE_OUTPUT_COLUMNS = [
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
VALUATION_OUTPUT_COLUMNS = [
    "date",
    "asset",
    "pe_ttm",
    "pb",
    "dividend_yield",
    "market_cap",
    "market_cap_basis",
    "dividend_yield_basis",
    "available_date",
    "data_source",
    "source_vintage",
]


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
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        compression=compression,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    temporary.replace(path)


class _CsvStreamWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        self.wrote_rows = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        missing = sorted(set(self.columns).difference(frame.columns))
        if missing:
            raise ValueError(f"stream output missing columns: {missing}")
        frame[self.columns].to_csv(
            self.path,
            mode="a" if self.wrote_rows else "w",
            header=not self.wrote_rows,
            index=False,
            encoding="utf-8" if self.wrote_rows else "utf-8-sig",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
        self.wrote_rows = True

    def ensure_file(self) -> None:
        if not self.wrote_rows:
            pd.DataFrame(columns=self.columns).to_csv(
                self.path,
                index=False,
                encoding="utf-8-sig",
                lineterminator="\n",
            )


def load_lifecycles(path: Path = MASTER_PATH, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"asset": str})
    listed = frame[frame["event_type"].eq("listing")][["asset", "exchange", "list_date"]].copy()
    exits = frame[frame["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listed["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("stock master contains duplicate lifecycle events")
    output = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce")
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce")
    if output[["asset", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("stock master contains incomplete lifecycle keys")
    if as_of is not None:
        as_of_date = pd.Timestamp(as_of).normalize()
        output = output[output["list_date"].le(as_of_date)].copy()
    return output.sort_values("asset").reset_index(drop=True)


def relevant_lifecycles(lifecycles: pd.DataFrame) -> pd.DataFrame:
    return lifecycles[
        lifecycles["delist_date"].isna() | lifecycles["delist_date"].ge(BACKTEST_START)
    ].copy().reset_index(drop=True)


def resolve_public_source_ip(timeout: float = 15.0) -> str:
    response = requests.get(
        "https://dns.google/resolve",
        params={"name": SOURCE_HOST, "type": "A"},
        timeout=timeout,
        headers={"Accept": "application/dns-json", "User-Agent": "long-hold-v4-data-steward/1.0"},
    )
    response.raise_for_status()
    answers = response.json().get("Answer", [])
    for answer in answers:
        if int(answer.get("type", 0)) != 1:
            continue
        address = ipaddress.ip_address(str(answer.get("data", "")))
        if address.version == 4 and address.is_global:
            return str(address)
    raise RuntimeError(f"no public IPv4 address resolved for {SOURCE_HOST}")


def _safe_baostock_send_msg(message: str) -> str:
    import baostock.common.contants as constants
    import baostock.common.context as context

    default_socket = getattr(context, "default_socket", None)
    if default_socket is None:
        raise ConnectionError("BaoStock socket is not connected")
    default_socket.sendall((message + "\n").encode("utf-8"))
    receive = bytearray()
    terminator = b"<![CDATA[]]>\n"
    while not receive.endswith(terminator):
        chunk = default_socket.recv(8192)
        if not chunk:
            raise ConnectionError("BaoStock peer closed the socket before a complete response")
        receive.extend(chunk)
        if len(receive) > MAX_SOURCE_MESSAGE_BYTES:
            raise RuntimeError(f"BaoStock response exceeds {MAX_SOURCE_MESSAGE_BYTES} bytes")
    if len(receive) < constants.MESSAGE_HEADER_LENGTH:
        raise RuntimeError("BaoStock response is shorter than the protocol header")
    head_bytes = bytes(receive[: constants.MESSAGE_HEADER_LENGTH])
    head = head_bytes.decode("utf-8")
    header_parts = head.split(constants.MESSAGE_SPLIT)
    if len(header_parts) < 3:
        raise RuntimeError("BaoStock response has an invalid protocol header")
    if header_parts[1] in constants.COMPRESSED_MESSAGE_TYPE_TUPLE:
        body_length = int(header_parts[2])
        body = bytes(
            receive[
                constants.MESSAGE_HEADER_LENGTH : constants.MESSAGE_HEADER_LENGTH + body_length
            ]
        )
        return head + zlib.decompress(body).decode("utf-8")
    return bytes(receive).decode("utf-8")


def _provider_code(asset: str, exchange: str) -> str:
    prefix = {"SSE": "sh", "SZSE": "sz"}.get(str(exchange))
    if prefix is None:
        raise ValueError(f"unsupported exchange for {asset}: {exchange}")
    return f"{prefix}.{str(asset).zfill(6)}"


def _cache_paths(asset: str) -> tuple[Path, Path]:
    return RAW_DIR / f"{asset}.csv.gz", RAW_DIR / f"{asset}.json"


def effective_market_date(
    as_of: str | pd.Timestamp,
    calendar_path: Path | None = None,
) -> pd.Timestamp:
    as_of_date = pd.Timestamp(as_of).normalize()
    calendar_path = calendar_path or TRADE_CALENDAR_PATH
    if not calendar_path.is_file():
        return as_of_date
    calendar = pd.read_csv(calendar_path, usecols=["date"])
    dates = pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.normalize()
    eligible = dates[dates.le(as_of_date)]
    if eligible.empty:
        raise ValueError(f"trade calendar has no date on or before {as_of_date.date()}")
    return pd.Timestamp(eligible.max()).normalize()


def _valid_cache(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    data_path, meta_path = _cache_paths(asset)
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "success" or meta.get("asset") != asset:
            return False
        if pd.Timestamp(meta["query_start"]) > start or pd.Timestamp(meta["query_end"]) < end:
            return False
        if _sha256(data_path) != meta.get("sha256"):
            return False
        sample = pd.read_csv(data_path, compression="gzip", nrows=5)
        return REQUIRED_HISTORY_COLUMNS.issubset(sample.columns)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def _cache_artifact_state(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, str]:
    data_path, meta_path = _cache_paths(asset)
    if not data_path.exists() and not meta_path.exists():
        return "missing", "not_collected"
    if _valid_cache(asset, start, end):
        return "completed", "validated_cache"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "failed":
                return "failed", str(metadata.get("error", "provider_collection_failed"))[:300]
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return "failed", "cache_artifacts_incomplete_or_invalid"


class BaoStockSession:
    def __init__(self, server_ip: str, timeout: float = 30.0):
        self.server_ip = server_ip
        self.timeout = timeout
        self.bs: Any | None = None

    def login(self) -> None:
        socket.setdefaulttimeout(self.timeout)
        import baostock as bs
        import baostock.common.contants as constants
        import baostock.util.socketutil as socketutil

        constants.BAOSTOCK_SERVER_IP = self.server_ip
        constants.BAOSTOCK_SERVER_PORT = SOURCE_PORT
        constants.BAOSTOCK_CLIENT_VERSION = SERVER_PROTOCOL_VERSION
        socketutil.send_msg = _safe_baostock_send_msg
        result = bs.login()
        if result.error_code != "0":
            raise ConnectionError(f"BaoStock login failed: {result.error_code} {result.error_msg}")
        self.bs = bs

    def close(self) -> None:
        if self.bs is not None:
            try:
                self.bs.logout()
            except Exception:
                pass
            finally:
                self.bs = None

    def query(self, provider_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.bs is None:
            self.login()
        result = self.bs.query_history_k_data_plus(
            provider_code,
            SOURCE_FIELDS,
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
            if len(rows) > MAX_SOURCE_HISTORY_ROWS:
                raise RuntimeError(f"BaoStock response exceeds {MAX_SOURCE_HISTORY_ROWS} rows")
        if result.error_code != "0":
            raise RuntimeError(
                f"BaoStock pagination failed: {result.error_code} {result.error_msg}"
            )
        return pd.DataFrame(rows, columns=result.fields or SOURCE_FIELDS.split(","))


def normalise_baostock_history(
    raw: pd.DataFrame,
    asset: str,
    list_date: str | pd.Timestamp,
    delist_date: str | pd.Timestamp | None,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_HISTORY_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"BaoStock history missing columns: {missing}")
    asset = str(asset).zfill(6)
    start = pd.Timestamp(list_date).normalize()
    as_of_date = pd.Timestamp(as_of).normalize()
    end = min(pd.Timestamp(delist_date).normalize(), as_of_date) if pd.notna(delist_date) else as_of_date
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    provider_assets = frame["code"].astype(str).str.extract(r"\.(\d{6})$", expand=False)
    if provider_assets.notna().any() and not provider_assets.dropna().eq(asset).all():
        raise ValueError(f"BaoStock response contains a different security code for {asset}")
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "tradestatus",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["date"].between(start, end)].copy()
    if frame.empty:
        raise ValueError(f"BaoStock history has no lifecycle-bounded rows for {asset}")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError(f"BaoStock history contains invalid or duplicate dates for {asset}")
    for column in ("tradestatus", "isST"):
        if not frame[column].dropna().isin([0, 1]).all() or frame[column].isna().any():
            raise ValueError(f"BaoStock history contains invalid {column} for {asset}")
    positive_close = frame["close"].gt(0)
    traded = frame["tradestatus"].eq(1)
    if (traded & ~positive_close).any():
        raise ValueError(f"BaoStock traded rows contain non-positive close for {asset}")
    frame.insert(0, "asset", asset)
    return frame.sort_values("date").reset_index(drop=True)


def _round_tick(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


CHINEXT_20_PERCENT_EFFECTIVE_DATE = pd.Timestamp("2020-08-24")
MAIN_BOARD_REGISTRATION_EFFECTIVE_DATE = pd.Timestamp("2023-04-10")


def _board(asset: str) -> str:
    if asset.startswith(("300", "301", "302")):
        return "chinext"
    if asset.startswith(("688", "689")):
        return "star"
    return "main"


def _is_no_limit_listing_window(
    asset: str,
    list_date: pd.Timestamp,
    listing_market_session: int,
    is_ipo: bool,
) -> bool:
    if not is_ipo or listing_market_session > 5:
        return False
    board = _board(asset)
    if board == "star":
        return True
    if board == "chinext" and list_date >= CHINEXT_20_PERCENT_EFFECTIVE_DATE:
        return True
    return board == "main" and list_date >= MAIN_BOARD_REGISTRATION_EFFECTIVE_DATE


def _regular_price_limit(asset: str, date: pd.Timestamp, is_st: bool) -> tuple[float, str]:
    board = _board(asset)
    if board == "star" or (board == "chinext" and date >= CHINEXT_20_PERCENT_EFFECTIVE_DATE):
        return 0.20, "regular_growth_20"
    if is_st:
        return 0.05, "regular_st_5"
    return 0.10, "regular_main_or_pre_reform_growth_10"


def build_trade_state(
    history: pd.DataFrame,
    list_date: str | pd.Timestamp,
    *,
    is_ipo: bool = True,
) -> pd.DataFrame:
    required = {"asset", "date", "preclose", "tradestatus", "isST"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"trade-state input missing columns: {missing}")
    frame = history.sort_values("date").copy()
    asset_values = frame["asset"].astype(str).str.zfill(6).unique()
    if len(asset_values) != 1:
        raise ValueError("trade-state input must contain exactly one asset")
    asset = asset_values[0]
    listed = pd.Timestamp(list_date).normalize()
    starts_on_listing_date = pd.Timestamp(frame.iloc[0]["date"]).normalize() == listed
    previous_paused = frame["tradestatus"].shift(1).eq(0)
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        date = pd.Timestamp(row.date).normalize()
        paused = int(row.tradestatus) == 0
        is_st = int(row.isST) == 1
        # Exchange rules refer to market trading days after listing, not the
        # security's cumulative traded days. A full-day suspension must not
        # extend a five-session no-limit window.
        listing_market_session = position + 1 if starts_on_listing_date else position + 6
        preclose = float(row.preclose) if pd.notna(row.preclose) else math.nan
        has_limit = False
        execution_known = True
        limit_pct = math.nan
        rule = "paused"
        limit_up = math.nan
        limit_down = math.nan
        if not paused:
            if _is_no_limit_listing_window(asset, listed, listing_market_session, is_ipo):
                rule = "no_price_limit_listing_window"
            elif listing_market_session == 1:
                rule = "first_session_special_unknown"
                execution_known = False
            elif bool(previous_paused.iloc[position]):
                rule = "resumption_limit_unknown"
                execution_known = False
            elif not math.isfinite(preclose) or preclose <= 0:
                rule = "invalid_preclose"
                execution_known = False
            else:
                has_limit = True
                limit_pct, rule = _regular_price_limit(asset, date, is_st)
                limit_up = _round_tick(preclose * (1.0 + limit_pct))
                limit_down = _round_tick(preclose * (1.0 - limit_pct))
        rows.append(
            {
                "date": date,
                "asset": asset,
                "is_paused": paused,
                "is_st": is_st,
                "pre_close": preclose,
                "has_price_limit": has_limit,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "price_limit_rate": limit_pct,
                "limit_rule": rule,
                "execution_state_known": execution_known,
                "available_date": date,
            }
        )
    return pd.DataFrame(rows)


def _asof_share_estimate(history: pd.DataFrame) -> pd.Series:
    volume = pd.to_numeric(history["volume"], errors="coerce")
    turnover = pd.to_numeric(history["turn"], errors="coerce")
    direct = volume / (turnover / 100.0)
    direct = direct.where(volume.gt(0) & turnover.gt(0) & direct.gt(0) & direct.map(math.isfinite))
    # A trailing median damps the provider's four-decimal turnover rounding without using future observations.
    estimate = direct.rolling(20, min_periods=1).median().ffill()
    return estimate


def build_monthly_valuation(
    history: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"asset", "date", "close", "volume", "turn", "peTTM", "pbMRQ"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"valuation input missing columns: {missing}")
    frame = history.sort_values("date").copy()
    asset_values = frame["asset"].astype(str).str.zfill(6).unique()
    if len(asset_values) != 1:
        raise ValueError("valuation input must contain exactly one asset")
    asset = asset_values[0]
    frame["float_shares_estimate"] = _asof_share_estimate(frame)
    frame["market_cap"] = pd.to_numeric(frame["close"], errors="coerce") * frame["float_shares_estimate"]
    frame["pe_ttm"] = pd.to_numeric(frame["peTTM"], errors="coerce")
    frame["pb"] = pd.to_numeric(frame["pbMRQ"], errors="coerce")

    payout_events: list[tuple[pd.Timestamp, float]] = []
    if dividends is not None and not dividends.empty:
        events = dividends.copy()
        events["asset"] = events["asset"].astype(str).str.zfill(6)
        events = events[events["asset"].eq(asset)].copy()
        events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce")
        events["available_date"] = pd.to_datetime(events["available_date"], errors="coerce")
        events["cash_per_share"] = pd.to_numeric(events["cash_per_share"], errors="coerce")
        events = events.dropna(subset=["ex_date", "available_date", "cash_per_share"])
        for event in events.itertuples(index=False):
            if event.available_date > event.ex_date or event.cash_per_share <= 0:
                continue
            prior = frame[frame["date"].le(event.ex_date) & frame["float_shares_estimate"].notna()].tail(1)
            if prior.empty:
                continue
            payout_events.append((event.ex_date, float(event.cash_per_share) * float(prior.iloc[0]["float_shares_estimate"])))

    frame["month"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby("month", sort=True).tail(1).copy()
    yields: list[float] = []
    for row in monthly.itertuples(index=False):
        date = pd.Timestamp(row.date)
        start = date - pd.Timedelta(days=365)
        payout = sum(amount for event_date, amount in payout_events if start < event_date <= date)
        market_cap = float(row.market_cap) if pd.notna(row.market_cap) else math.nan
        yields.append(payout / market_cap if payout >= 0 and math.isfinite(market_cap) and market_cap > 0 else math.nan)
    monthly["dividend_yield"] = yields
    monthly["available_date"] = monthly["date"]
    monthly["market_cap_basis"] = "float_cap_from_trailing_turnover_implied_shares"
    monthly["dividend_yield_basis"] = "trailing_365d_cash_payout_over_float_market_cap"
    return monthly[
        [
            "date",
            "asset",
            "pe_ttm",
            "pb",
            "dividend_yield",
            "market_cap",
            "market_cap_basis",
            "dividend_yield_basis",
            "available_date",
        ]
    ].reset_index(drop=True)


def _fetch_one(
    session: BaoStockSession,
    lifecycle: Any,
    as_of: pd.Timestamp,
    server_ip: str,
    attempts: int = 3,
) -> dict[str, Any]:
    asset = str(lifecycle.asset).zfill(6)
    start = pd.Timestamp(lifecycle.list_date).normalize()
    end = min(pd.Timestamp(lifecycle.delist_date).normalize(), as_of) if pd.notna(lifecycle.delist_date) else as_of
    data_path, meta_path = _cache_paths(asset)
    if _valid_cache(asset, start, end):
        return {**json.loads(meta_path.read_text(encoding="utf-8")), "collection_action": "cache_reused"}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = session.query(_provider_code(asset, lifecycle.exchange), start, end)
            normalise_baostock_history(raw, asset, start, end, as_of)
            fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
            raw["fetched_at"] = fetched_at
            raw["data_source"] = SOURCE_NAME
            raw["source_server_ip"] = server_ip
            raw["server_protocol_version"] = SERVER_PROTOCOL_VERSION
            _atomic_csv(raw, data_path, compression="gzip")
            metadata = {
                "status": "success",
                "asset": asset,
                "provider_code": _provider_code(asset, lifecycle.exchange),
                "query_start": start.date().isoformat(),
                "query_end": end.date().isoformat(),
                "rows": int(len(raw)),
                "coverage_start": str(raw["date"].min()) if not raw.empty else None,
                "coverage_end": str(raw["date"].max()) if not raw.empty else None,
                "fetched_at": fetched_at,
                "source_host": SOURCE_HOST,
                "source_server_ip": server_ip,
                "source_port": SOURCE_PORT,
                "python_package_version": PYPI_PACKAGE_VERSION,
                "server_protocol_version": SERVER_PROTOCOL_VERSION,
                "collection_action": "fetched",
                "fields": SOURCE_FIELDS.split(","),
                "path": _relative(data_path),
                "sha256": _sha256(data_path),
            }
            _atomic_json(metadata, meta_path)
            return metadata
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:400]}"
            session.close()
            if attempt < attempts:
                time.sleep(float(attempt))
    failure = {
        "status": "failed",
        "asset": asset,
        "query_start": start.date().isoformat(),
        "query_end": end.date().isoformat(),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": last_error,
    }
    _atomic_json(failure, meta_path)
    return failure


def collect_raw_history(
    lifecycles: pd.DataFrame,
    as_of: pd.Timestamp,
    server_ip: str,
    shard_count: int = 1,
    shard_index: int = 0,
    collect_limit: int | None = None,
    sleep_seconds: float = 0.05,
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be within shard_count")
    if shard_count > MAX_SAFE_COLLECTION_WORKERS:
        raise ValueError(
            f"shard_count exceeds provider-safe maximum {MAX_SAFE_COLLECTION_WORKERS}"
        )
    market_end = effective_market_date(as_of)
    pending_mask = []
    for row in lifecycles.itertuples(index=False):
        asset = str(row.asset).zfill(6)
        start = pd.Timestamp(row.list_date).normalize()
        end = (
            min(pd.Timestamp(row.delist_date).normalize(), market_end)
            if pd.notna(row.delist_date)
            else market_end
        )
        pending_mask.append(not _valid_cache(asset, start, end))
    pending = lifecycles.loc[pending_mask].copy().reset_index(drop=True)
    selected = select_collection_shard(
        pending,
        shard_count=shard_count,
        shard_index=shard_index,
        collect_limit=collect_limit,
    )
    statuses: list[dict[str, Any]] = []
    deferred_assets: list[str] = []
    provider_blacklist_triggered = False
    selected_rows = list(selected.itertuples(index=False))
    if selected_rows:
        session = BaoStockSession(server_ip)
        try:
            for position, row in enumerate(selected_rows):
                result = _fetch_one(session, row, market_end, server_ip)
                statuses.append(result)
                if "10001011" in str(result.get("error", "")):
                    provider_blacklist_triggered = True
                    deferred_assets = [
                        str(item.asset).zfill(6) for item in selected_rows[position + 1 :]
                    ]
                    break
                if sleep_seconds > 0 and position + 1 < len(selected):
                    time.sleep(sleep_seconds)
        finally:
            session.close()
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of.date().isoformat(),
        "effective_market_date": market_end.date().isoformat(),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "target_assets": int(len(lifecycles)),
        "already_cached_assets": int(len(lifecycles) - len(pending)),
        "pending_assets_before_limit": int(len(pending)),
        "selected_assets": int(len(selected)),
        "successful_assets": int(sum(item.get("status") == "success" for item in statuses)),
        "fetched_assets": int(sum(item.get("collection_action") == "fetched" for item in statuses)),
        "reused_cache_assets": int(sum(item.get("collection_action") == "cache_reused" for item in statuses)),
        "failed_assets": [item.get("asset") for item in statuses if item.get("status") != "success"],
        "deferred_assets": deferred_assets,
        "circuit_breaker": "provider_blacklist" if provider_blacklist_triggered else None,
    }
    _atomic_json(report, RAW_DIR / f"collection_status_shard_{shard_index}_of_{shard_count}.json")
    return report


def select_collection_shard(
    lifecycles: pd.DataFrame,
    shard_count: int,
    shard_index: int,
    collect_limit: int | None = None,
) -> pd.DataFrame:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be within shard_count")
    bounded = lifecycles.head(max(0, collect_limit)) if collect_limit is not None else lifecycles
    return bounded.iloc[shard_index::shard_count].copy().reset_index(drop=True)


def filter_lifecycles_by_asset_file(lifecycles: pd.DataFrame, asset_file: str | Path) -> pd.DataFrame:
    path = Path(asset_file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    requested = pd.read_csv(path, dtype={"asset": str})
    if "asset" not in requested.columns:
        raise ValueError("asset file must contain an asset column")
    assets = requested["asset"].dropna().astype(str).str.zfill(6).drop_duplicates()
    if assets.empty:
        raise ValueError("asset file contains no assets")
    available = set(lifecycles["asset"].astype(str).str.zfill(6))
    unknown = sorted(set(assets).difference(available))
    if unknown:
        raise ValueError(f"asset file contains assets outside governed lifecycles: {unknown[:10]}")
    order = {asset: position for position, asset in enumerate(assets)}
    output = lifecycles[lifecycles["asset"].astype(str).str.zfill(6).isin(order)].copy()
    output["_asset_file_order"] = output["asset"].astype(str).str.zfill(6).map(order)
    return output.sort_values("_asset_file_order").drop(columns="_asset_file_order").reset_index(drop=True)


def _load_cached_history(
    lifecycle: Any,
    as_of: pd.Timestamp,
    validate_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    asset = str(lifecycle.asset).zfill(6)
    start = pd.Timestamp(lifecycle.list_date).normalize()
    end = min(pd.Timestamp(lifecycle.delist_date).normalize(), as_of) if pd.notna(lifecycle.delist_date) else as_of
    data_path, meta_path = _cache_paths(asset)
    if validate_cache and not _valid_cache(asset, start, end):
        raise FileNotFoundError(f"valid BaoStock cache missing for {asset}")
    if not data_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"BaoStock cache artifacts missing for {asset}")
    raw = pd.read_csv(data_path, compression="gzip", low_memory=False)
    history = normalise_baostock_history(raw, asset, start, end, as_of)
    return history, json.loads(meta_path.read_text(encoding="utf-8"))


def _write_output_manifest(
    output_path: Path,
    manifest_path: Path,
    source_vintage: str,
    inputs: list[dict[str, str]],
    rows: int,
    assets: int,
    target_assets: int,
    formal: bool,
    as_of: pd.Timestamp,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    code_path = Path(__file__).resolve()
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of.date().isoformat(),
        "inputs": inputs,
        "source_vintage": source_vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "rows": rows,
        "assets": assets,
        "target_assets": target_assets,
        "asset_coverage": round(assets / target_assets, 6) if target_assets else 0.0,
        "qualification_status": "READY_FOR_CROSS_SOURCE_VALIDATION" if formal else "COLLECTION_IN_PROGRESS",
        "historical_backtest_allowed": bool(formal),
        "model_promotion_allowed": False,
        **metrics,
    }
    _atomic_json(payload, manifest_path)
    return payload


def build_outputs(as_of: str | pd.Timestamp) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    market_date = effective_market_date(as_of_date)
    lifecycles = load_lifecycles(as_of=as_of_date)
    relevant = relevant_lifecycles(lifecycles)
    dividends = pd.read_csv(DIVIDEND_PATH, dtype={"asset": str}) if DIVIDEND_PATH.is_file() else pd.DataFrame()
    if not dividends.empty:
        dividends["asset"] = dividends["asset"].astype(str).str.zfill(6)
        dividends_by_asset = {asset: group.copy() for asset, group in dividends.groupby("asset", sort=False)}
    else:
        dividends_by_asset = {}
    cache_candidates: list[tuple[Any, dict[str, Any]]] = []
    raw_inputs: list[dict[str, str]] = []
    missing_assets: list[str] = []
    failures: dict[str, str] = {}
    inventory_by_asset: dict[str, dict[str, Any]] = {}
    summary_by_asset: dict[str, dict[str, Any]] = {}
    asset_order: list[str] = []
    for row in relevant.itertuples(index=False):
        asset = str(row.asset).zfill(6)
        asset_order.append(asset)
        start = pd.Timestamp(row.list_date).normalize()
        end = min(pd.Timestamp(row.delist_date).normalize(), market_date) if pd.notna(row.delist_date) else market_date
        state, detail = _cache_artifact_state(asset, start, end)
        summary_by_asset[asset] = {
            "asset": asset,
            "status": state,
            "trade_status": state,
            "valuation_status": state,
            "trade_rows": 0,
            "unknown_execution_rows": 0,
            "valuation_candidate_rows": 0,
            "valuation_valid_rows": 0,
            "detail": detail,
            "trade_detail": detail,
            "valuation_detail": detail,
        }
        if state == "missing":
            missing_assets.append(asset)
            inventory_by_asset[asset] = {"asset": asset, "status": state, "detail": detail}
            continue
        if state == "failed":
            failures[asset] = detail
            inventory_by_asset[asset] = {"asset": asset, "status": state, "detail": detail}
            continue
        try:
            data_path, meta_path = _cache_paths(asset)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_inputs.append(
                {"source_id": asset, "path": _relative(data_path), "sha256": str(meta["sha256"])}
            )
            cache_candidates.append((row, meta))
            inventory_by_asset[asset] = {
                "asset": asset,
                "status": "completed",
                "detail": "validated_cache_pending_derivation",
            }
        except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures[asset] = f"{type(exc).__name__}: {str(exc)[:300]}"
            inventory_by_asset[asset] = {"asset": asset, "status": "failed", "detail": failures[asset]}
            summary_by_asset[asset].update(
                status="failed",
                trade_status="failed",
                valuation_status="failed",
                detail=failures[asset],
                trade_detail=failures[asset],
                valuation_detail=failures[asset],
            )

    master_hash = _sha256(MASTER_PATH)
    dividend_hash = _sha256(DIVIDEND_PATH) if DIVIDEND_PATH.is_file() else "missing"
    calendar_hash = _sha256(TRADE_CALENDAR_PATH) if TRADE_CALENDAR_PATH.is_file() else "missing"
    bundle_hash = hashlib.sha256(
        "|".join(
            [
                master_hash,
                dividend_hash,
                calendar_hash,
                *sorted(f"{x['source_id']}:{x['sha256']}" for x in raw_inputs),
            ]
        ).encode()
    ).hexdigest()
    source_vintage = f"baostock_market_history_bundle_sha256:{bundle_hash}"
    trade_temp = OBSERVATION_DIR / f".stock_trade_state_build_{os.getpid()}.csv.tmp"
    valuation_temp = OBSERVATION_DIR / f".stock_valuation_build_{os.getpid()}.csv.tmp"
    trade_writer = _CsvStreamWriter(trade_temp, TRADE_OUTPUT_COLUMNS)
    valuation_writer = _CsvStreamWriter(valuation_temp, VALUATION_OUTPUT_COLUMNS)
    trade_successful_assets: set[str] = set()
    valuation_processed_assets: set[str] = set()
    valuation_assets: set[str] = set()
    trade_failures: dict[str, str] = {}
    valuation_failures: dict[str, str] = {}
    trade_rows = 0
    unknown_execution_rows = 0
    paused_rows = 0
    st_rows = 0
    trade_duplicate_keys = False
    valuation_candidate_rows = 0
    valuation_valid_rows = 0
    valuation_duplicate_keys = False
    valuation_market_cap_positive = True
    required_valuation = ["pe_ttm", "pb", "market_cap", "dividend_yield"]
    try:
        for row, _ in cache_candidates:
            asset = str(row.asset).zfill(6)
            try:
                history, _ = _load_cached_history(row, market_date, validate_cache=False)
            except (FileNotFoundError, OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                failures[asset] = detail
                trade_failures[asset] = detail
                valuation_failures[asset] = detail
                inventory_by_asset[asset] = {"asset": asset, "status": "failed", "detail": detail}
                summary_by_asset[asset].update(
                    status="failed",
                    trade_status="failed",
                    valuation_status="failed",
                    detail=detail,
                    trade_detail=detail,
                    valuation_detail=detail,
                )
                continue

            trade_error = ""
            valuation_error = ""
            asset_trade_rows = 0
            asset_unknown_rows = 0
            asset_candidate_rows = 0
            asset_valid_rows = 0
            try:
                predecessor = getattr(row, "predecessor_asset", None)
                if pd.isna(predecessor) or not str(predecessor).strip():
                    trade_asset = build_trade_state(history, row.list_date)
                else:
                    trade_asset = build_trade_state(history, row.list_date, is_ipo=False)
                trade_asset = trade_asset[trade_asset["date"].ge(BACKTEST_START)].copy()
                if trade_asset.duplicated(["date", "asset"]).any():
                    raise ValueError(f"duplicate trade-state keys for {asset}")
                asset_trade_rows = int(len(trade_asset))
                if asset_trade_rows <= 0:
                    raise ValueError(f"empty trade-state history for {asset}")
                trade_asset["data_source"] = SOURCE_NAME
                trade_asset["source_vintage"] = source_vintage
                trade_writer.append(trade_asset)
                asset_unknown_rows = int((~trade_asset["execution_state_known"].astype(bool)).sum())
                trade_rows += asset_trade_rows
                unknown_execution_rows += asset_unknown_rows
                paused_rows += int(trade_asset["is_paused"].astype(bool).sum())
                st_rows += int(trade_asset["is_st"].astype(bool).sum())
                trade_successful_assets.add(asset)
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                trade_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                trade_failures[asset] = trade_error

            try:
                valuation_candidates = build_monthly_valuation(
                    history,
                    dividends_by_asset.get(asset),
                )
                valuation_candidates = valuation_candidates[
                    valuation_candidates["date"].ge(BACKTEST_START)
                ].copy()
                if valuation_candidates.duplicated(["date", "asset"]).any():
                    raise ValueError(f"duplicate valuation keys for {asset}")
                valuation_valid_mask = valuation_candidates[required_valuation].notna().all(axis=1)
                valuation_asset = valuation_candidates.loc[valuation_valid_mask].copy()
                if not valuation_asset.empty and not valuation_asset["market_cap"].gt(0).all():
                    raise ValueError(f"non-positive valid market cap for {asset}")
                valuation_asset["data_source"] = SOURCE_NAME
                valuation_asset["source_vintage"] = source_vintage
                valuation_writer.append(valuation_asset)
                asset_candidate_rows = int(len(valuation_candidates))
                asset_valid_rows = int(len(valuation_asset))
                valuation_candidate_rows += asset_candidate_rows
                valuation_valid_rows += asset_valid_rows
                valuation_processed_assets.add(asset)
                if asset_valid_rows:
                    valuation_assets.add(asset)
                else:
                    valuation_error = f"no complete valuation rows for {asset}"
                    valuation_failures[asset] = valuation_error
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                valuation_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                valuation_failures[asset] = valuation_error

            trade_ok = asset in trade_successful_assets
            valuation_ok = asset in valuation_assets
            status = "completed" if trade_ok and valuation_ok else "partial" if trade_ok or valuation_ok else "failed"
            details = []
            if trade_error:
                details.append(f"trade={trade_error}")
            if valuation_error:
                details.append(f"valuation={valuation_error}")
            detail = ";".join(details) if details else "validated_cache_and_derivation"
            inventory_by_asset[asset] = {"asset": asset, "status": status, "detail": detail}
            summary_by_asset[asset].update(
                status=status,
                trade_status="completed" if trade_ok else "failed",
                valuation_status="completed" if valuation_ok else "failed",
                trade_rows=asset_trade_rows,
                unknown_execution_rows=asset_unknown_rows,
                valuation_candidate_rows=asset_candidate_rows,
                valuation_valid_rows=asset_valid_rows,
                detail=detail,
                trade_detail="validated_cache_and_derivation" if trade_ok else trade_error,
                valuation_detail="validated_cache_and_derivation" if valuation_ok else valuation_error,
            )

        trade_writer.ensure_file()
        valuation_writer.ensure_file()
        completed_assets = len(trade_successful_assets.intersection(valuation_assets))
        trade_completed_assets = len(trade_successful_assets)
        valuation_completed_assets = len(valuation_assets)
        target_assets = len(relevant)
        cache_complete = len(cache_candidates) == target_assets and not missing_assets and not failures
        trade_complete = cache_complete and trade_completed_assets == target_assets and not trade_failures
        valuation_complete = (
            cache_complete
            and valuation_completed_assets == target_assets
            and len(valuation_processed_assets) == target_assets
            and not valuation_failures
        )
        unknown_ratio = unknown_execution_rows / trade_rows if trade_rows else 1.0
        valuation_coverage = (
            valuation_valid_rows / valuation_candidate_rows if valuation_candidate_rows else 0.0
        )
        trade_internal_pass = bool(
            trade_complete
            and trade_rows > 0
            and not trade_duplicate_keys
            and unknown_ratio <= MAX_UNKNOWN_EXECUTION_RATIO
            and paused_rows > 0
            and st_rows > 0
        )
        valuation_internal_pass = bool(
            valuation_complete
            and valuation_valid_rows > 0
            and len(valuation_assets) == target_assets
            and not valuation_duplicate_keys
            and valuation_coverage >= MIN_VALUATION_COVERAGE
            and valuation_market_cap_positive
        )
        formal = trade_internal_pass and valuation_internal_pass
        trade_path = TRADE_OUTPUT if trade_internal_pass else TRADE_OBSERVATION
        valuation_path = VALUATION_OUTPUT if valuation_internal_pass else VALUATION_OBSERVATION
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        valuation_path.parent.mkdir(parents=True, exist_ok=True)
        trade_temp.replace(trade_path)
        valuation_temp.replace(valuation_path)
    except BaseException:
        trade_temp.unlink(missing_ok=True)
        valuation_temp.unlink(missing_ok=True)
        raise

    inventory_rows = [inventory_by_asset[asset] for asset in asset_order]
    summary_rows = [summary_by_asset[asset] for asset in asset_order]
    _atomic_csv(pd.DataFrame(inventory_rows, columns=["asset", "status", "detail"]), COLLECTION_INVENTORY)
    _atomic_csv(
        pd.DataFrame(
            summary_rows,
            columns=[
                "asset",
                "status",
                "trade_status",
                "valuation_status",
                "trade_rows",
                "unknown_execution_rows",
                "valuation_candidate_rows",
                "valuation_valid_rows",
                "detail",
                "trade_detail",
                "valuation_detail",
            ],
        ),
        ASSET_SUMMARY,
    )
    inputs = [
        {"source_id": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": master_hash},
        *(
            [{"source_id": "trade_calendar", "path": _relative(TRADE_CALENDAR_PATH), "sha256": calendar_hash}]
            if TRADE_CALENDAR_PATH.is_file()
            else []
        ),
        *(
            [{"source_id": "stock_dividend_events", "path": _relative(DIVIDEND_PATH), "sha256": dividend_hash}]
            if DIVIDEND_PATH.is_file()
            else []
        ),
        *raw_inputs,
    ]
    trade_manifest = _write_output_manifest(
        trade_path,
        TRADE_MANIFEST,
        source_vintage,
        inputs,
        trade_rows,
        trade_completed_assets,
        target_assets,
        trade_internal_pass,
        as_of_date,
        {
            "internal_quality_pass": trade_internal_pass,
            "unknown_execution_state_ratio": round(unknown_ratio, 6),
            "maximum_unknown_execution_state_ratio": MAX_UNKNOWN_EXECUTION_RATIO,
            "paused_rows": paused_rows,
            "st_rows": st_rows,
            "derivation_failure_count": len(trade_failures),
            "derivation_failures": trade_failures,
            "build_mode": "two_pass_asset_streaming_v1",
            "effective_market_date": market_date.date().isoformat(),
            "output_sort_order": ["asset", "date"],
            "qualification_blockers": ([] if trade_internal_pass else ["coverage_or_trade_state_quality_incomplete"]),
        },
    )
    valuation_manifest = _write_output_manifest(
        valuation_path,
        VALUATION_MANIFEST,
        source_vintage,
        inputs,
        valuation_valid_rows,
        len(valuation_assets),
        target_assets,
        valuation_internal_pass,
        as_of_date,
        {
            "internal_quality_pass": valuation_internal_pass,
            "complete_valuation_row_ratio": round(valuation_coverage, 6),
            "minimum_complete_valuation_row_ratio": MIN_VALUATION_COVERAGE,
            "candidate_rows_before_missing_value_filter": valuation_candidate_rows,
            "rows_excluded_for_missing_required_values": valuation_candidate_rows - valuation_valid_rows,
            "processed_assets": len(valuation_processed_assets),
            "derivation_failure_count": len(valuation_failures),
            "derivation_failures": valuation_failures,
            "build_mode": "two_pass_asset_streaming_v1",
            "effective_market_date": market_date.date().isoformat(),
            "output_sort_order": ["asset", "date"],
            "qualification_blockers": ([] if valuation_internal_pass else ["coverage_or_valuation_quality_incomplete"]),
        },
    )
    if not trade_internal_pass:
        _atomic_json(trade_manifest, COMBINED_MANIFEST.with_name("stock_trade_state_probe_latest.json"))
    if not valuation_internal_pass:
        _atomic_json(valuation_manifest, COMBINED_MANIFEST.with_name("stock_valuation_probe_latest.json"))
    combined = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "effective_market_date": market_date.date().isoformat(),
        "target_assets": target_assets,
        "completed_assets": completed_assets,
        "cache_ready_assets": len(cache_candidates),
        "trade_completed_assets": trade_completed_assets,
        "valuation_processed_assets": len(valuation_processed_assets),
        "valuation_completed_assets": valuation_completed_assets,
        "missing_asset_count": len(missing_assets),
        "missing_asset_sample": missing_assets[:100],
        "failed_asset_count": len(failures),
        "failed_assets": failures,
        "trade_derivation_failure_count": len(trade_failures),
        "trade_derivation_failures": trade_failures,
        "valuation_derivation_failure_count": len(valuation_failures),
        "valuation_derivation_failures": valuation_failures,
        "collection_inventory_path": _relative(COLLECTION_INVENTORY),
        "collection_inventory_sha256": _sha256(COLLECTION_INVENTORY),
        "asset_summary_path": _relative(ASSET_SUMMARY),
        "asset_summary_sha256": _sha256(ASSET_SUMMARY),
        "source_vintage": source_vintage,
        "trade_state": trade_manifest,
        "valuation": valuation_manifest,
        "dataset_qualification": {
            "stock_trade_state": trade_manifest["qualification_status"],
            "stock_valuation_history": valuation_manifest["qualification_status"],
        },
        "qualification_status": "READY_FOR_CROSS_SOURCE_VALIDATION" if formal else "COLLECTION_IN_PROGRESS",
        "build_mode": "two_pass_asset_streaming_v1",
        "historical_backtest_allowed": bool(formal),
        "model_promotion_allowed": False,
    }
    _atomic_json(combined, COMBINED_MANIFEST)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--server-ip")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--collect-limit", type=int)
    parser.add_argument("--asset-file", help="CSV containing an asset column for governed targeted collection")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_collect and args.collect_only:
        raise SystemExit("--skip-collect and --collect-only cannot be used together")
    as_of = pd.Timestamp(args.as_of).normalize()
    lifecycles = load_lifecycles(as_of=as_of)
    if args.asset_file:
        collection_lifecycles = filter_lifecycles_by_asset_file(lifecycles, args.asset_file)
    else:
        collection_lifecycles = relevant_lifecycles(lifecycles)
    collection: dict[str, Any] | None = None
    if not args.skip_collect:
        server_ip = args.server_ip or resolve_public_source_ip()
        collection = collect_raw_history(
            collection_lifecycles,
            as_of,
            server_ip,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            collect_limit=args.collect_limit,
            sleep_seconds=args.sleep_seconds,
        )
    if args.collect_only:
        print(json.dumps({"collection": collection, "build_outputs_skipped": True}, ensure_ascii=False))
        return
    result = build_outputs(as_of)
    print(
        json.dumps(
            {
                "collection": collection,
                "qualification_status": result["qualification_status"],
                "completed_assets": result["completed_assets"],
                "target_assets": result["target_assets"],
                "trade_rows": result["trade_state"]["rows"],
                "valuation_rows": result["valuation"]["rows"],
                "historical_backtest_allowed": result["historical_backtest_allowed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
