"""Collect raw Tencent ETF daily prices as an independent validation source.

Tencent retains unadjusted daily bars for many delisted ETFs.  The source is
captured as an immutable current-final observation.  Historical rows therefore
share the real collection date as ``available_date`` and are never promoted to
PIT backtest inputs by this collector.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .pit_etf_total_return_collector import load_lifecycles


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
OUTPUT_ROOT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "tencent_etf_price"
)
LATEST_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tencent_etf_price_validation_latest.json"
)
SOURCE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SOURCE_NAME = "tencent.ifzq.raw_day_kline"
PAGE_SIZE = 2_000
MAX_PAGES = 20
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.+-]{8,160}$")
PRICE_COLUMNS = [
    "date",
    "asset",
    "asset_name",
    "exchange",
    "lifecycle_status",
    "list_date",
    "delist_date",
    "open",
    "high",
    "low",
    "close",
    "volume_lots",
    "volume_shares",
    "ohlc_relationship_valid",
    "ohlc_rounding_gap",
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
    "symbol",
    "status",
    "error",
    "list_date",
    "delist_date",
    "query_end",
    "raw_rows",
    "normalised_rows",
    "coverage_start",
    "coverage_end",
    "page_count",
    "ohlc_relationship_invalid_rows",
    "maximum_ohlc_rounding_gap",
    "source_observed_at",
    "elapsed_seconds",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]


@dataclass(frozen=True)
class SourcePage:
    page_number: int
    query_end: str
    content: bytes
    rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if compressed else None,
    )
    temporary.replace(path)


def _authenticate_master(
    master_path: Path = MASTER_PATH,
    manifest_path: Path = MASTER_MANIFEST_PATH,
) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not True
        or manifest.get("model_promotion_allowed") is not False
        or Path(str(manifest.get("output_path", ""))).as_posix() != _relative(master_path)
        or _sha256(master_path) != str(manifest.get("output_sha256", ""))
    ):
        raise ValueError("ETF security master failed source authentication")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF security-master builder hash mismatch")
    return {"master_sha256": _sha256(master_path), "manifest_sha256": _sha256(manifest_path)}


def tencent_symbol(asset: str, exchange: str) -> str:
    prefix = {"SSE": "sh", "SZSE": "sz"}.get(str(exchange))
    if prefix is None:
        raise ValueError(f"unsupported ETF exchange for {asset}: {exchange}")
    return f"{prefix}{str(asset).zfill(6)}"


def select_assets(
    lifecycles: pd.DataFrame,
    *,
    selection: str = "all",
    explicit_assets: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    selected = lifecycles.copy()
    if selection == "active":
        selected = selected[selected["delist_date"].isna()].copy()
    elif selection == "delisted":
        selected = selected[selected["delist_date"].notna()].copy()
    elif selection != "all":
        raise ValueError(f"unsupported ETF selection: {selection}")
    if explicit_assets:
        requested = {str(value).strip().replace(".0", "").zfill(6) for value in explicit_assets}
        selected = selected[selected["asset"].isin(requested)].copy()
        missing = sorted(requested.difference(selected["asset"]))
        if missing:
            raise ValueError(f"ETF price probe assets are outside the governed selection: {missing}")
    selected["selection_priority"] = selected["delist_date"].isna().astype(int)
    selected = selected.sort_values(["selection_priority", "list_date", "asset"]).drop(columns="selection_priority")
    if offset < 0:
        raise ValueError("selection offset must be non-negative")
    selected = selected.iloc[offset:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("selection limit must be positive")
        selected = selected.iloc[:limit]
    return selected.reset_index(drop=True)


def parse_tencent_page(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    if payload.get("code") not in (0, "0", None):
        raise ValueError(f"Tencent price response error: {payload.get('code')} {payload.get('msg', '')}")
    node = (payload.get("data") or {}).get(symbol) or {}
    rows = node.get("day") or []
    parsed: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("Tencent ETF price response contains a malformed daily row")
        parsed.append(row[:6])
    if not parsed:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume_lots",
                "ohlc_relationship_valid",
                "ohlc_rounding_gap",
            ]
        )
    frame = pd.DataFrame(parsed, columns=["date", "open", "close", "high", "low", "volume_lots"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    numeric = ["open", "close", "high", "low", "volume_lots"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame["date"].isna().any() or frame[numeric].isna().any(axis=None):
        raise ValueError("Tencent ETF price response contains invalid dates or numeric values")
    if (frame[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("Tencent ETF price response contains non-positive prices")
    if (frame["volume_lots"] < 0).any():
        raise ValueError("Tencent ETF price response contains negative volume")
    upper_gap = (frame[["open", "close"]].max(axis=1) - frame["high"]).clip(lower=0.0)
    lower_gap = (frame["low"] - frame[["open", "close"]].min(axis=1)).clip(lower=0.0)
    frame["ohlc_rounding_gap"] = pd.concat([upper_gap, lower_gap], axis=1).max(axis=1)
    frame["ohlc_relationship_valid"] = frame["ohlc_rounding_gap"].le(1e-12)
    if frame["high"].lt(frame["low"]).any() or frame["ohlc_rounding_gap"].gt(0.005000001).any():
        raise ValueError("Tencent ETF price response has an OHLC violation beyond the 0.005 rounding bound")
    if frame["date"].duplicated().any():
        raise ValueError("Tencent ETF price page contains duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def fetch_tencent_history(
    session: requests.Session,
    *,
    symbol: str,
    list_date: str | pd.Timestamp,
    query_end: str | pd.Timestamp,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    timeout: float = 30.0,
) -> tuple[pd.DataFrame, list[SourcePage]]:
    if page_size <= 0 or page_size > PAGE_SIZE:
        raise ValueError(f"Tencent page size must be in 1..{PAGE_SIZE}")
    lifecycle_start = pd.Timestamp(list_date).normalize()
    end = pd.Timestamp(query_end).normalize()
    pages: list[SourcePage] = []
    batches: list[pd.DataFrame] = []
    previous_first: pd.Timestamp | None = None
    for page_number in range(1, max_pages + 1):
        params = {"param": f"{symbol},day,,{end.date().isoformat()},{page_size},"}
        response = session.get(
            SOURCE_URL,
            params=params,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=timeout,
        )
        response.raise_for_status()
        content = bytes(response.content)
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Tencent ETF price response is not valid UTF-8 JSON") from exc
        frame = parse_tencent_page(payload, symbol)
        pages.append(SourcePage(page_number, end.date().isoformat(), content, len(frame)))
        if frame.empty:
            break
        if frame["date"].gt(end).any():
            raise ValueError("Tencent ETF price page exceeds its requested end date")
        first = pd.Timestamp(frame["date"].min()).normalize()
        if previous_first is not None and first >= previous_first:
            raise ValueError("Tencent ETF price pagination made no backward progress")
        previous_first = first
        batches.append(frame)
        if first <= lifecycle_start or len(frame) < page_size:
            break
        end = first - pd.Timedelta(days=1)
    else:
        raise ValueError(f"Tencent ETF price pagination exceeded {max_pages} pages")
    if not batches:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume_lots",
                "ohlc_relationship_valid",
                "ohlc_rounding_gap",
            ]
        ), pages
    combined = pd.concat(batches, ignore_index=True)
    combined = combined.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return combined, pages


def normalise_history(
    raw: pd.DataFrame,
    lifecycle: pd.Series,
    *,
    as_of: str | pd.Timestamp,
    observed_at: str,
    source_vintage: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    asset = str(lifecycle["asset"]).zfill(6)
    start = pd.Timestamp(lifecycle["list_date"]).normalize()
    delist = pd.to_datetime(lifecycle.get("delist_date"), errors="coerce")
    end = min(pd.Timestamp(as_of).normalize(), delist.normalize()) if pd.notna(delist) else pd.Timestamp(as_of).normalize()
    data = raw[raw["date"].between(start, end)].copy()
    if data.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    if "ohlc_rounding_gap" not in data.columns or "ohlc_relationship_valid" not in data.columns:
        upper_gap = (data[["open", "close"]].max(axis=1) - data["high"]).clip(lower=0.0)
        lower_gap = (data["low"] - data[["open", "close"]].min(axis=1)).clip(lower=0.0)
        data["ohlc_rounding_gap"] = pd.concat([upper_gap, lower_gap], axis=1).max(axis=1)
        data["ohlc_relationship_valid"] = data["ohlc_rounding_gap"].le(1e-12)
    if data["date"].duplicated().any():
        raise ValueError(f"Tencent ETF history contains duplicate dates for {asset}")
    data["asset"] = asset
    data["asset_name"] = str(lifecycle.get("asset_name", ""))
    data["exchange"] = str(lifecycle["exchange"])
    data["lifecycle_status"] = "delisted" if pd.notna(delist) else "listed"
    data["list_date"] = start
    data["delist_date"] = delist.normalize() if pd.notna(delist) else pd.NaT
    data["volume_shares"] = data["volume_lots"] * 100.0
    data["source_observed_at"] = observed_at
    data["available_date"] = pd.Timestamp(observed_at).normalize()
    data["pit_actionable"] = False
    data["data_source"] = SOURCE_NAME
    data["source_vintage"] = source_vintage
    data["qualification_status"] = "CURRENT_FINAL_INDEPENDENT_PRICE_OBSERVATION"
    data["historical_backtest_allowed"] = False
    data["model_promotion_allowed"] = False
    data = data.rename(columns={"volume_lots": "volume_lots"})
    return data[PRICE_COLUMNS].sort_values("date").reset_index(drop=True)


def _source_set_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "asset": row["asset"],
            "page_number": int(row["page_number"]),
            "uncompressed_sha256": row["uncompressed_sha256"],
        }
        for row in sorted(rows, key=lambda item: (item["asset"], int(item["page_number"])))
    ]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_id(as_of: pd.Timestamp, observed_at: datetime) -> str:
    stamp = observed_at.strftime("%Y%m%dT%H%M%S%f%z")
    nonce = hashlib.sha256(f"{as_of.date()}|{stamp}|{os.getpid()}".encode()).hexdigest()[:12]
    return f"{as_of.strftime('%Y%m%d')}_{stamp}_{nonce}"


def _selection_sha256(selected: pd.DataFrame) -> str:
    rows = [
        {
            "asset": str(row.asset).zfill(6),
            "exchange": str(row.exchange),
            "list_date": pd.Timestamp(row.list_date).date().isoformat(),
            "delist_date": pd.Timestamp(row.delist_date).date().isoformat() if pd.notna(row.delist_date) else None,
        }
        for row in selected.itertuples(index=False)
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resume_asset(
    asset_dir: Path,
    *,
    asset: str,
    observed_at: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]] | None:
    metadata_path = asset_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("asset") != asset
        or metadata.get("source_observed_at") != observed_at
        or metadata.get("status") not in {"ready", "no_data"}
    ):
        return None
    source_pages = metadata.get("source_pages")
    if not isinstance(source_pages, list) or not source_pages:
        return None
    inventory: list[dict[str, Any]] = []
    for entry in source_pages:
        path = ROOT / str(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != str(entry.get("sha256", "")):
            return None
        try:
            content = gzip.decompress(path.read_bytes())
        except (OSError, EOFError):
            return None
        if _bytes_sha256(content) != str(entry.get("uncompressed_sha256", "")):
            return None
        inventory.append(dict(entry))
    prices = pd.DataFrame(columns=PRICE_COLUMNS)
    if metadata["status"] == "ready":
        price_path = ROOT / str(metadata.get("price_path", ""))
        if not price_path.is_file() or _sha256(price_path) != str(metadata.get("price_sha256", "")):
            return None
        prices = pd.read_csv(price_path, dtype={"asset": str}, parse_dates=["date", "list_date", "delist_date", "available_date"])
        if len(prices) != int(metadata.get("normalised_rows", -1)):
            return None
    status = {key: metadata.get(key) for key in STATUS_COLUMNS}
    status["elapsed_seconds"] = 0.0
    return prices, status, inventory


def collect(
    *,
    as_of: str | pd.Timestamp,
    selection: str = "all",
    explicit_assets: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    sleep_seconds: float = 0.02,
    timeout: float = 30.0,
    run_id: str | None = None,
    resume: bool = False,
    output_root: Path = OUTPUT_ROOT,
    latest_manifest_path: Path = LATEST_MANIFEST_PATH,
) -> dict[str, Any]:
    as_of_ts = pd.Timestamp(as_of).normalize()
    observed = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = run_id or _run_id(as_of_ts, observed)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid Tencent ETF price run id")
    master_auth = _authenticate_master()
    collector_code_sha256 = _sha256(Path(__file__).resolve())
    lifecycles = load_lifecycles(MASTER_PATH, as_of_ts)
    selected = select_assets(
        lifecycles,
        selection=selection,
        explicit_assets=explicit_assets,
        offset=offset,
        limit=limit,
    )
    if selected.empty:
        raise ValueError("Tencent ETF price selection is empty")
    run_root = output_root / run_id
    state_path = run_root / "run_state.json"
    final_manifest_path = run_root / "run_manifest.json"
    selection_sha256 = _selection_sha256(selected)
    if run_root.exists() and any(run_root.iterdir()):
        if not resume:
            raise FileExistsError(f"Tencent ETF price run already exists: {run_root}")
        if final_manifest_path.is_file():
            raise FileExistsError(f"completed Tencent ETF price run is immutable: {run_root}")
        if not state_path.is_file():
            raise ValueError("Tencent ETF price partial run has no authenticated run state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("run_id") != run_id
            or state.get("as_of_date") != as_of_ts.date().isoformat()
            or state.get("selection_sha256") != selection_sha256
            or state.get("master_sha256") != master_auth["master_sha256"]
            or state.get("collector_code_sha256") != collector_code_sha256
        ):
            raise ValueError("Tencent ETF price resume contract differs from the partial run")
        observed_at = str(state["source_observed_at"])
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        observed_at = observed.isoformat(timespec="seconds")
        _atomic_json(
            {
                "schema_version": 1,
                "run_id": run_id,
                "as_of_date": as_of_ts.date().isoformat(),
                "source_observed_at": observed_at,
                "selection": selection,
                "offset": offset,
                "limit": limit,
                "selected_assets": selected["asset"].astype(str).str.zfill(6).tolist(),
                "selection_sha256": selection_sha256,
                "master_sha256": master_auth["master_sha256"],
                "collector_code_sha256": collector_code_sha256,
                "status": "in_progress",
            },
            state_path,
        )
    session = requests.Session()
    price_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    consecutive_errors = 0
    for lifecycle in selected.itertuples(index=False):
        row = pd.Series(lifecycle._asdict())
        asset = str(row["asset"]).zfill(6)
        symbol = tencent_symbol(asset, str(row["exchange"]))
        query_end = min(as_of_ts, pd.Timestamp(row["delist_date"]).normalize()) if pd.notna(row["delist_date"]) else as_of_ts
        asset_dir = run_root / "assets" / asset
        if resume:
            restored = _resume_asset(asset_dir, asset=asset, observed_at=observed_at)
            if restored is not None:
                restored_prices, restored_status, restored_inventory = restored
                if not restored_prices.empty:
                    price_frames.append(restored_prices)
                status_rows.append(restored_status)
                inventory_rows.extend(restored_inventory)
                print(
                    json.dumps(
                        {
                            "asset": asset,
                            "status": restored_status["status"],
                            "rows": int(restored_status["normalised_rows"] or 0),
                            "pages": int(restored_status["page_count"] or 0),
                            "reused": True,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
        error = ""
        pages: list[SourcePage] = []
        raw = pd.DataFrame()
        normalised = pd.DataFrame(columns=PRICE_COLUMNS)
        started = time.monotonic()
        try:
            raw, pages = fetch_tencent_history(
                session,
                symbol=symbol,
                list_date=row["list_date"],
                query_end=query_end,
                timeout=timeout,
            )
            page_hashes = [_bytes_sha256(page.content) for page in pages]
            vintage_hash = hashlib.sha256("\n".join(sorted(page_hashes)).encode("ascii")).hexdigest()
            normalised = normalise_history(
                raw,
                row,
                as_of=as_of_ts,
                observed_at=observed_at,
                source_vintage=f"tencent_raw_page_set_sha256:{vintage_hash}",
            )
            consecutive_errors = 0
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            consecutive_errors += 1
        asset_inventory: list[dict[str, Any]] = []
        for page in pages:
            path = asset_dir / f"page_{page.page_number:03d}.json.gz"
            _atomic_bytes(gzip.compress(page.content, compresslevel=9, mtime=0), path)
            inventory_entry = {
                "asset": asset,
                "symbol": symbol,
                "page_number": page.page_number,
                "query_end": page.query_end,
                "response_rows": page.rows,
                "path": _relative(path),
                "sha256": _sha256(path),
                "uncompressed_sha256": _bytes_sha256(page.content),
                "bytes": path.stat().st_size,
            }
            inventory_rows.append(inventory_entry)
            asset_inventory.append(inventory_entry)
        if not normalised.empty:
            asset_price_path = asset_dir / "prices.csv.gz"
            _atomic_csv(normalised, asset_price_path, compressed=True)
            price_frames.append(normalised)
        else:
            asset_price_path = asset_dir / "prices.csv.gz"
        status = "ready" if not normalised.empty and not error else ("no_data" if not error else "error")
        status_row = {
            "asset": asset,
            "symbol": symbol,
            "status": status,
            "error": error,
            "list_date": pd.Timestamp(row["list_date"]).date().isoformat(),
            "delist_date": pd.Timestamp(row["delist_date"]).date().isoformat() if pd.notna(row["delist_date"]) else None,
            "query_end": query_end.date().isoformat(),
            "raw_rows": int(len(raw)),
            "normalised_rows": int(len(normalised)),
            "coverage_start": normalised["date"].min().date().isoformat() if len(normalised) else None,
            "coverage_end": normalised["date"].max().date().isoformat() if len(normalised) else None,
            "page_count": len(pages),
            "ohlc_relationship_invalid_rows": int((~normalised["ohlc_relationship_valid"]).sum())
            if len(normalised)
            else 0,
            "maximum_ohlc_rounding_gap": float(normalised["ohlc_rounding_gap"].max())
            if len(normalised)
            else 0.0,
            "source_observed_at": observed_at,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
        metadata = {
            **status_row,
            "price_path": _relative(asset_price_path) if not normalised.empty else None,
            "price_sha256": _sha256(asset_price_path) if not normalised.empty else None,
            "source_pages": asset_inventory,
        }
        _atomic_json(metadata, asset_dir / "metadata.json")
        status_rows.append(status_row)
        print(
            json.dumps(
                {"asset": asset, "status": status, "rows": int(len(normalised)), "pages": len(pages), "error": error},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if consecutive_errors >= 5:
            raise RuntimeError("Tencent ETF price provider circuit breaker opened after 5 consecutive errors")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame(columns=PRICE_COLUMNS)
    if not prices.empty:
        prices = prices.sort_values(["asset", "date"]).reset_index(drop=True)
        if prices.duplicated(["date", "asset"]).any():
            raise ValueError("Tencent ETF combined price output contains duplicate keys")
        numeric = ["open", "high", "low", "close", "volume_lots", "volume_shares", "ohlc_rounding_gap"]
        if not np.isfinite(prices[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Tencent ETF combined price output contains non-finite values")
    status_frame = pd.DataFrame(status_rows, columns=STATUS_COLUMNS).sort_values("asset").reset_index(drop=True)
    inventory = pd.DataFrame(inventory_rows).sort_values(["asset", "page_number"]).reset_index(drop=True)
    price_path = run_root / "tencent_etf_raw_prices.csv.gz"
    status_path = run_root / "asset_status.csv"
    inventory_path = run_root / "raw_response_inventory.csv"
    _atomic_csv(prices, price_path, compressed=True)
    _atomic_csv(status_frame, status_path)
    _atomic_csv(inventory, inventory_path)
    code_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": observed_at,
        "as_of_date": as_of_ts.date().isoformat(),
        "qualification_status": "CURRENT_FINAL_INDEPENDENT_PRICE_OBSERVATION",
        "selection": selection,
        "offset": offset,
        "limit": limit,
        "selected_assets": int(len(selected)),
        "selected_delisted_assets": int(selected["delist_date"].notna().sum()),
        "ready_assets": int(status_frame["status"].eq("ready").sum()),
        "no_data_assets": int(status_frame["status"].eq("no_data").sum()),
        "error_assets": int(status_frame["status"].eq("error").sum()),
        "price_rows": int(len(prices)),
        "coverage_start": prices["date"].min().date().isoformat() if len(prices) else None,
        "coverage_end": prices["date"].max().date().isoformat() if len(prices) else None,
        "raw_response_pages": int(len(inventory)),
        "ohlc_relationship_invalid_rows": int((~prices["ohlc_relationship_valid"]).sum()) if len(prices) else 0,
        "maximum_ohlc_rounding_gap": float(prices["ohlc_rounding_gap"].max()) if len(prices) else 0.0,
        "raw_response_set_sha256": _source_set_sha256(inventory_rows),
        "query_contract": {
            "url": SOURCE_URL,
            "frequency": "day",
            "adjustment": "raw",
            "page_size": PAGE_SIZE,
            "pagination": "exclusive end date moved to one calendar day before prior page minimum",
            "volume_unit": "Tencent lots; volume_shares equals lots multiplied by 100",
        },
        "inputs": [
            {"role": "etf_security_master", "path": _relative(MASTER_PATH), "sha256": master_auth["master_sha256"]},
            {
                "role": "etf_security_master_manifest",
                "path": _relative(MASTER_MANIFEST_PATH),
                "sha256": master_auth["manifest_sha256"],
            },
        ],
        "outputs": [
            {"role": "tencent_raw_prices", "path": _relative(price_path), "sha256": _sha256(price_path), "rows": len(prices)},
            {"role": "asset_status", "path": _relative(status_path), "sha256": _sha256(status_path), "rows": len(status_frame)},
            {
                "role": "raw_response_inventory",
                "path": _relative(inventory_path),
                "sha256": _sha256(inventory_path),
                "rows": len(inventory),
            },
        ],
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "formal_table_promotion_allowed": False,
        "limitations": [
            "one current-final Tencent observation is not a historical source-version archive",
            "the source has no traded amount field",
            "some dates and historical OHLC values may differ from Sina and require explicit cross-source diagnostics",
            "bounded high/low rounding violations are preserved and flagged rather than silently repaired",
            "collection proves source availability only; it does not qualify a total-return backtest",
        ],
    }
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "completed"
    state["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    _atomic_json(state, state_path)
    manifest["outputs"].append(
        {"role": "run_state", "path": _relative(state_path), "sha256": _sha256(state_path), "rows": 1}
    )
    manifest_path = final_manifest_path
    _atomic_json(manifest, manifest_path)
    latest = {**manifest, "immutable_manifest_path": _relative(manifest_path), "immutable_manifest_sha256": _sha256(manifest_path)}
    _atomic_json(latest, latest_manifest_path)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-07-17")
    parser.add_argument("--selection", choices=["all", "active", "delisted"], default="all")
    parser.add_argument("--assets", nargs="*")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = collect(
        as_of=args.as_of,
        selection=args.selection,
        explicit_assets=args.assets,
        offset=args.offset,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        run_id=args.run_id,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "run_id",
                    "qualification_status",
                    "selected_assets",
                    "selected_delisted_assets",
                    "ready_assets",
                    "no_data_assets",
                    "error_assets",
                    "price_rows",
                    "coverage_start",
                    "coverage_end",
                    "historical_backtest_allowed",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
