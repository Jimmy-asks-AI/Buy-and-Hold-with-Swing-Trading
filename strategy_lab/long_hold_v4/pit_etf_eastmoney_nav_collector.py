"""Collect full-lifecycle Eastmoney ETF NAV observations with raw evidence.

The compact ``pingzhongdata`` source is preferred because one response carries
the complete unit and cumulative NAV series.  Delisted pages that no longer
publish that JavaScript fall back to the paginated F10 NAV API.  Both methods
produce current-final observations only; historical rows retain the real
collection date as ``available_date``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
from .pit_etf_tencent_price_collector import RUN_ID_PATTERN, select_assets


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
OUTPUT_ROOT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "eastmoney_etf_nav"
)
LATEST_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "eastmoney_etf_nav_validation_latest.json"
)
JS_URL = "https://fund.eastmoney.com/pingzhongdata/{asset}.js"
API_URL = "https://api.fund.eastmoney.com/f10/lsjz"
SOURCE_NAME = "eastmoney.pingzhongdata_or_f10_lsjz"
API_PAGE_SIZE = 20
API_MAX_PAGES = 400
NET_WORTH_PATTERN = re.compile(r"var\s+Data_netWorthTrend\s*=\s*(\[.*?\]);", re.DOTALL)
ACC_WORTH_PATTERN = re.compile(r"var\s+Data_ACWorthTrend\s*=\s*(\[.*?\]);", re.DOTALL)
DEPENDENCY_CODE_PATHS = [
    Path(__file__).resolve(),
    ROOT / "strategy_lab" / "long_hold_v4" / "pit_etf_tencent_price_collector.py",
    ROOT / "strategy_lab" / "long_hold_v4" / "pit_etf_total_return_collector.py",
]
NAV_COLUMNS = [
    "date",
    "asset",
    "asset_name",
    "exchange",
    "lifecycle_status",
    "list_date",
    "delist_date",
    "unit_nav",
    "cumulative_nav",
    "daily_growth_pct",
    "unit_money",
    "source_method",
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
    "status",
    "error",
    "source_method",
    "list_date",
    "delist_date",
    "query_end",
    "normalised_rows",
    "coverage_start",
    "coverage_end",
    "unit_nav_missing_source_rows_dropped",
    "cumulative_nav_missing_rows",
    "raw_response_count",
    "api_page_count",
    "source_observed_at",
    "elapsed_seconds",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
INVENTORY_COLUMNS = [
    "asset",
    "source_kind",
    "sequence",
    "query_page",
    "response_rows",
    "path",
    "sha256",
    "uncompressed_sha256",
    "bytes",
]


@dataclass(frozen=True)
class SourceResponse:
    source_kind: str
    sequence: int
    query_page: int | None
    content: bytes
    rows: int


class EastmoneySourceParseError(ValueError):
    def __init__(self, message: str, responses: list[SourceResponse]):
        super().__init__(message)
        self.responses = responses


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _code_files() -> list[dict[str, str]]:
    return [{"path": _relative(path), "sha256": _sha256(path)} for path in DEPENDENCY_CODE_PATHS]


def _code_bundle_sha256(files: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _authenticate_master() -> dict[str, str]:
    manifest = json.loads(MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not True
        or manifest.get("model_promotion_allowed") is not False
        or Path(str(manifest.get("output_path", ""))).as_posix() != _relative(MASTER_PATH)
        or _sha256(MASTER_PATH) != str(manifest.get("output_sha256", ""))
    ):
        raise ValueError("ETF security master failed source authentication")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF security-master builder hash mismatch")
    return {"master_sha256": _sha256(MASTER_PATH), "manifest_sha256": _sha256(MASTER_MANIFEST_PATH)}


def _timestamp_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, unit="ms", errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None)


def parse_pingzhongdata(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Eastmoney pingzhongdata response is not UTF-8") from exc
    net_match = NET_WORTH_PATTERN.search(text)
    acc_match = ACC_WORTH_PATTERN.search(text)
    if net_match is None or acc_match is None:
        return pd.DataFrame(columns=["date", "unit_nav", "cumulative_nav", "daily_growth_pct", "unit_money"])
    try:
        net_rows = json.loads(net_match.group(1))
        acc_rows = json.loads(acc_match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Eastmoney pingzhongdata NAV arrays are invalid JSON") from exc
    if not isinstance(net_rows, list) or not isinstance(acc_rows, list):
        raise ValueError("Eastmoney pingzhongdata NAV arrays are not lists")
    unit = pd.DataFrame(net_rows)
    cumulative = pd.DataFrame(acc_rows, columns=["x", "cumulative_nav"]) if acc_rows else pd.DataFrame(columns=["x", "cumulative_nav"])
    if unit.empty and cumulative.empty:
        return pd.DataFrame(columns=["date", "unit_nav", "cumulative_nav", "daily_growth_pct", "unit_money"])
    required = {"x", "y"}
    if not required.issubset(unit.columns):
        raise ValueError("Eastmoney pingzhongdata unit-NAV array is incomplete")
    unit = unit.rename(columns={"y": "unit_nav", "equityReturn": "daily_growth_pct", "unitMoney": "unit_money"})
    if "daily_growth_pct" not in unit.columns:
        unit["daily_growth_pct"] = np.nan
    if "unit_money" not in unit.columns:
        unit["unit_money"] = ""
    unit["date"] = _timestamp_dates(unit["x"])
    cumulative["date"] = _timestamp_dates(cumulative["x"])
    output = unit[["date", "unit_nav", "daily_growth_pct", "unit_money"]].merge(
        cumulative[["date", "cumulative_nav"]], on="date", how="outer", validate="one_to_one"
    )
    return _validate_nav_frame(output)


def parse_f10_page(payload: dict[str, Any]) -> tuple[pd.DataFrame, int, int]:
    if payload.get("ErrCode") not in (0, "0", None):
        raise ValueError(f"Eastmoney F10 NAV response error: {payload.get('ErrCode')} {payload.get('ErrMsg', '')}")
    rows = ((payload.get("Data") or {}).get("LSJZList") or [])
    frame = pd.DataFrame(rows).rename(
        columns={
            "FSRQ": "date",
            "DWJZ": "unit_nav",
            "LJJZ": "cumulative_nav",
            "JZZZL": "daily_growth_pct",
            "FHSP": "unit_money",
        }
    )
    if frame.empty:
        frame = pd.DataFrame(columns=["date", "unit_nav", "cumulative_nav", "daily_growth_pct", "unit_money"])
    else:
        required = {"date", "unit_nav"}
        if not required.issubset(frame.columns):
            raise ValueError("Eastmoney F10 NAV rows are incomplete")
        if "cumulative_nav" not in frame.columns:
            frame["cumulative_nav"] = np.nan
        for column in ["daily_growth_pct", "unit_money"]:
            if column not in frame.columns:
                frame[column] = np.nan if column == "daily_growth_pct" else ""
        frame = _validate_nav_frame(frame[["date", "unit_nav", "cumulative_nav", "daily_growth_pct", "unit_money"]])
    return frame, int(payload.get("TotalCount") or 0), int(payload.get("PageSize") or API_PAGE_SIZE)


def _validate_nav_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["unit_nav", "cumulative_nav", "daily_growth_pct"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["unit_money"] = data["unit_money"].fillna("").astype(str)
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("Eastmoney ETF NAV response has invalid or duplicate dates")
    unit_present = data["unit_nav"].notna()
    if (data.loc[unit_present, "unit_nav"] <= 0).any():
        raise ValueError("Eastmoney ETF NAV response has non-positive unit NAV")
    cumulative_present = data["cumulative_nav"].notna()
    if (data.loc[cumulative_present, "cumulative_nav"] <= 0).any():
        raise ValueError("Eastmoney ETF NAV response has non-positive cumulative NAV")
    return data.sort_values("date").reset_index(drop=True)


def fetch_eastmoney_nav(
    session: requests.Session,
    *,
    asset: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    timeout: float = 30.0,
) -> tuple[pd.DataFrame, str, list[SourceResponse]]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://fund.eastmoney.com/{asset}.html"}
    js_response = session.get(JS_URL.format(asset=asset), headers=headers, timeout=timeout)
    js_response.raise_for_status()
    js_content = bytes(js_response.content)
    try:
        js_frame = parse_pingzhongdata(js_content)
    except ValueError as exc:
        responses = [SourceResponse("pingzhongdata", 1, None, js_content, 0)]
        raise EastmoneySourceParseError(str(exc), responses) from exc
    responses = [SourceResponse("pingzhongdata", 1, None, js_content, len(js_frame))]
    if not js_frame.empty:
        return js_frame, "pingzhongdata", responses
    base_params = {
        "fundCode": asset,
        "pageSize": str(API_PAGE_SIZE),
        "startDate": pd.Timestamp(start_date).date().isoformat(),
        "endDate": pd.Timestamp(end_date).date().isoformat(),
    }
    batches: list[pd.DataFrame] = []
    page_count: int | None = None
    for page in range(1, API_MAX_PAGES + 1):
        response = session.get(
            API_URL,
            params={**base_params, "pageIndex": str(page)},
            headers={"User-Agent": "Mozilla/5.0", "Referer": f"https://fundf10.eastmoney.com/jjjz_{asset}.html"},
            timeout=timeout,
        )
        response.raise_for_status()
        content = bytes(response.content)
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failed = SourceResponse("f10_lsjz", len(responses) + 1, page, content, 0)
            raise EastmoneySourceParseError(
                "Eastmoney F10 NAV response is not valid UTF-8 JSON",
                [*responses, failed],
            ) from exc
        try:
            frame, total, page_size = parse_f10_page(payload)
        except ValueError as exc:
            failed = SourceResponse("f10_lsjz", len(responses) + 1, page, content, 0)
            raise EastmoneySourceParseError(str(exc), [*responses, failed]) from exc
        responses.append(SourceResponse("f10_lsjz", len(responses) + 1, page, content, len(frame)))
        if page_count is None:
            page_count = math.ceil(total / page_size) if total else 0
            if page_count > API_MAX_PAGES:
                raise ValueError(f"Eastmoney F10 NAV pagination exceeds {API_MAX_PAGES} pages")
        if not frame.empty:
            batches.append(frame)
        if page >= (page_count or 0):
            break
    if not batches:
        return pd.DataFrame(columns=["date", "unit_nav", "cumulative_nav", "daily_growth_pct", "unit_money"]), "f10_lsjz", responses
    combined = pd.concat(batches, ignore_index=True).drop_duplicates("date", keep="last")
    return _validate_nav_frame(combined), "f10_lsjz", responses


def normalise_nav(
    raw: pd.DataFrame,
    lifecycle: pd.Series,
    *,
    as_of: str | pd.Timestamp,
    observed_at: str,
    source_method: str,
    source_vintage: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=NAV_COLUMNS)
    start = pd.Timestamp(lifecycle["list_date"]).normalize()
    delist = pd.to_datetime(lifecycle.get("delist_date"), errors="coerce")
    end = min(pd.Timestamp(as_of).normalize(), delist.normalize()) if pd.notna(delist) else pd.Timestamp(as_of).normalize()
    data = raw[raw["date"].between(start, end)].copy()
    data = data[data["unit_nav"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=NAV_COLUMNS)
    data["asset"] = str(lifecycle["asset"]).zfill(6)
    data["asset_name"] = str(lifecycle.get("asset_name", ""))
    data["exchange"] = str(lifecycle["exchange"])
    data["lifecycle_status"] = "delisted" if pd.notna(delist) else "listed"
    data["list_date"] = start
    data["delist_date"] = delist.normalize() if pd.notna(delist) else pd.NaT
    data["source_method"] = source_method
    data["source_observed_at"] = observed_at
    data["available_date"] = pd.Timestamp(observed_at).normalize()
    data["pit_actionable"] = False
    data["data_source"] = SOURCE_NAME
    data["source_vintage"] = source_vintage
    data["qualification_status"] = "CURRENT_FINAL_INDEPENDENT_NAV_OBSERVATION"
    data["historical_backtest_allowed"] = False
    data["model_promotion_allowed"] = False
    return data[NAV_COLUMNS].sort_values("date").reset_index(drop=True)


def _run_id(as_of: pd.Timestamp, observed_at: datetime) -> str:
    stamp = observed_at.strftime("%Y%m%dT%H%M%S%f%z")
    nonce = hashlib.sha256(f"eastmoney|{as_of.date()}|{stamp}|{os.getpid()}".encode()).hexdigest()[:12]
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
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_set_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "asset": row["asset"],
            "sequence": int(row["sequence"]),
            "uncompressed_sha256": row["uncompressed_sha256"],
        }
        for row in sorted(rows, key=lambda item: (item["asset"], int(item["sequence"])))
    ]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    if metadata.get("asset") != asset or metadata.get("source_observed_at") != observed_at or metadata.get("status") not in {"ready", "no_data"}:
        return None
    raw_inventory = metadata.get("source_responses")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        return None
    for item in raw_inventory:
        path = Path(str(item.get("path", "")))
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            return None
        if _bytes_sha256(gzip.decompress(path.read_bytes())) != str(item.get("uncompressed_sha256", "")):
            return None
    nav = pd.DataFrame(columns=NAV_COLUMNS)
    if metadata["status"] == "ready":
        nav_path = Path(str(metadata.get("nav_path", "")))
        nav_path = nav_path if nav_path.is_absolute() else ROOT / nav_path
        if not nav_path.is_file() or _sha256(nav_path) != str(metadata.get("nav_sha256", "")):
            return None
        nav = pd.read_csv(nav_path, dtype={"asset": str}, parse_dates=["date", "list_date", "delist_date", "available_date"])
        if len(nav) != int(metadata.get("normalised_rows", -1)):
            return None
    status = {key: metadata.get(key) for key in STATUS_COLUMNS}
    status["elapsed_seconds"] = 0.0
    return nav, status, [dict(item) for item in raw_inventory]


def collect(
    *,
    as_of: str | pd.Timestamp,
    selection: str = "all",
    explicit_assets: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    sleep_seconds: float = 0.05,
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
        raise ValueError("invalid Eastmoney ETF NAV run id")
    master_auth = _authenticate_master()
    code_path = Path(__file__).resolve()
    code_sha256 = _sha256(code_path)
    code_files = _code_files()
    code_bundle_sha256 = _code_bundle_sha256(code_files)
    selected = select_assets(
        load_lifecycles(MASTER_PATH, as_of_ts),
        selection=selection,
        explicit_assets=explicit_assets,
        offset=offset,
        limit=limit,
    )
    if selected.empty:
        raise ValueError("Eastmoney ETF NAV selection is empty")
    selection_hash = _selection_sha256(selected)
    run_root = output_root / run_id
    state_path = run_root / "run_state.json"
    final_manifest_path = run_root / "run_manifest.json"
    if run_root.exists() and any(run_root.iterdir()):
        if not resume:
            raise FileExistsError(f"Eastmoney ETF NAV run already exists: {run_root}")
        if final_manifest_path.is_file():
            raise FileExistsError(f"completed Eastmoney ETF NAV run is immutable: {run_root}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("run_id") != run_id
            or state.get("as_of_date") != as_of_ts.date().isoformat()
            or state.get("selection_sha256") != selection_hash
            or state.get("master_sha256") != master_auth["master_sha256"]
            or state.get("code_bundle_sha256") != code_bundle_sha256
        ):
            raise ValueError("Eastmoney ETF NAV resume contract differs from the partial run")
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
                "selection_sha256": selection_hash,
                "master_sha256": master_auth["master_sha256"],
                "code_bundle_sha256": code_bundle_sha256,
                "status": "in_progress",
            },
            state_path,
        )
    session = requests.Session()
    nav_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    consecutive_errors = 0
    for lifecycle in selected.itertuples(index=False):
        row = pd.Series(lifecycle._asdict())
        asset = str(row["asset"]).zfill(6)
        asset_dir = run_root / "assets" / asset
        if resume:
            restored = _resume_asset(asset_dir, asset=asset, observed_at=observed_at)
            if restored is not None:
                restored_nav, restored_status, restored_inventory = restored
                if not restored_nav.empty:
                    nav_frames.append(restored_nav)
                status_rows.append(restored_status)
                inventory_rows.extend(restored_inventory)
                print(json.dumps({"asset": asset, "status": restored_status["status"], "rows": int(restored_status["normalised_rows"] or 0), "reused": True}, ensure_ascii=False), flush=True)
                continue
        delist = pd.to_datetime(row["delist_date"], errors="coerce")
        query_end = min(as_of_ts, delist.normalize()) if pd.notna(delist) else as_of_ts
        started = time.monotonic()
        raw = pd.DataFrame()
        normalised = pd.DataFrame(columns=NAV_COLUMNS)
        source_method = ""
        responses: list[SourceResponse] = []
        unit_nav_missing_source_rows_dropped = 0
        error = ""
        try:
            raw, source_method, responses = fetch_eastmoney_nav(
                session,
                asset=asset,
                start_date=row["list_date"],
                end_date=query_end,
                timeout=timeout,
            )
            unit_nav_missing_source_rows_dropped = int(raw["unit_nav"].isna().sum())
            response_hashes = [_bytes_sha256(item.content) for item in responses]
            source_hash = hashlib.sha256("\n".join(sorted(response_hashes)).encode("ascii")).hexdigest()
            normalised = normalise_nav(
                raw,
                row,
                as_of=as_of_ts,
                observed_at=observed_at,
                source_method=source_method,
                source_vintage=f"eastmoney_nav_response_set_sha256:{source_hash}",
            )
            consecutive_errors = 0
        except EastmoneySourceParseError as exc:
            responses = exc.responses
            error = f"{type(exc).__name__}: {exc}"[:500]
            consecutive_errors += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            consecutive_errors += 1
        asset_inventory: list[dict[str, Any]] = []
        for response in responses:
            path = asset_dir / f"response_{response.sequence:03d}_{response.source_kind}.bin.gz"
            _atomic_bytes(gzip.compress(response.content, compresslevel=9, mtime=0), path)
            entry = {
                "asset": asset,
                "source_kind": response.source_kind,
                "sequence": response.sequence,
                "query_page": response.query_page,
                "response_rows": response.rows,
                "path": _relative(path),
                "sha256": _sha256(path),
                "uncompressed_sha256": _bytes_sha256(response.content),
                "bytes": path.stat().st_size,
            }
            inventory_rows.append(entry)
            asset_inventory.append(entry)
        nav_path = asset_dir / "nav.csv.gz"
        if not normalised.empty:
            _atomic_csv(normalised, nav_path, compressed=True)
            nav_frames.append(normalised)
        status = "ready" if not normalised.empty and not error else ("no_data" if not error else "error")
        status_row = {
            "asset": asset,
            "status": status,
            "error": error,
            "source_method": source_method,
            "list_date": pd.Timestamp(row["list_date"]).date().isoformat(),
            "delist_date": delist.date().isoformat() if pd.notna(delist) else None,
            "query_end": query_end.date().isoformat(),
            "normalised_rows": int(len(normalised)),
            "coverage_start": normalised["date"].min().date().isoformat() if len(normalised) else None,
            "coverage_end": normalised["date"].max().date().isoformat() if len(normalised) else None,
            "unit_nav_missing_source_rows_dropped": unit_nav_missing_source_rows_dropped,
            "cumulative_nav_missing_rows": int(normalised["cumulative_nav"].isna().sum()),
            "raw_response_count": len(responses),
            "api_page_count": sum(item.source_kind == "f10_lsjz" for item in responses),
            "source_observed_at": observed_at,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
        metadata = {
            **status_row,
            "nav_path": _relative(nav_path) if not normalised.empty else None,
            "nav_sha256": _sha256(nav_path) if not normalised.empty else None,
            "source_responses": asset_inventory,
        }
        _atomic_json(metadata, asset_dir / "metadata.json")
        status_rows.append(status_row)
        print(json.dumps({"asset": asset, "status": status, "rows": len(normalised), "method": source_method, "responses": len(responses), "error": error}, ensure_ascii=False), flush=True)
        if consecutive_errors >= 5:
            raise RuntimeError("Eastmoney ETF NAV provider circuit breaker opened after 5 consecutive errors")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    nav = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame(columns=NAV_COLUMNS)
    if not nav.empty:
        nav = nav.sort_values(["asset", "date"]).reset_index(drop=True)
        if nav.duplicated(["date", "asset"]).any() or not np.isfinite(nav["unit_nav"].to_numpy(dtype=float)).all():
            raise ValueError("Eastmoney ETF NAV combined output failed numeric or key validation")
    status_frame = pd.DataFrame(status_rows, columns=STATUS_COLUMNS).sort_values("asset").reset_index(drop=True)
    inventory = (
        pd.DataFrame(inventory_rows, columns=INVENTORY_COLUMNS)
        .sort_values(["asset", "sequence"])
        .reset_index(drop=True)
    )
    nav_output = run_root / "eastmoney_etf_nav.csv.gz"
    status_output = run_root / "asset_status.csv"
    inventory_output = run_root / "raw_response_inventory.csv"
    _atomic_csv(nav, nav_output, compressed=True)
    _atomic_csv(status_frame, status_output)
    _atomic_csv(inventory, inventory_output)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "completed"
    state["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    _atomic_json(state, state_path)
    qualification = (
        "CURRENT_FINAL_INDEPENDENT_NAV_OBSERVATION"
        if not status_frame["status"].eq("error").any()
        else "PARTIAL_CURRENT_FINAL_INDEPENDENT_NAV_OBSERVATION"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": observed_at,
        "as_of_date": as_of_ts.date().isoformat(),
        "qualification_status": qualification,
        "selection": selection,
        "offset": offset,
        "limit": limit,
        "selected_assets": int(len(selected)),
        "selected_delisted_assets": int(selected["delist_date"].notna().sum()),
        "ready_assets": int(status_frame["status"].eq("ready").sum()),
        "no_data_assets": int(status_frame["status"].eq("no_data").sum()),
        "error_assets": int(status_frame["status"].eq("error").sum()),
        "pingzhongdata_assets": int(status_frame["source_method"].eq("pingzhongdata").sum()),
        "f10_fallback_assets": int(status_frame["source_method"].eq("f10_lsjz").sum()),
        "nav_rows": int(len(nav)),
        "unit_nav_missing_source_rows_dropped": int(
            pd.to_numeric(status_frame["unit_nav_missing_source_rows_dropped"]).sum()
        ),
        "cumulative_nav_missing_rows": int(nav["cumulative_nav"].isna().sum()),
        "coverage_start": nav["date"].min().date().isoformat() if len(nav) else None,
        "coverage_end": nav["date"].max().date().isoformat() if len(nav) else None,
        "raw_response_count": int(len(inventory)),
        "raw_response_set_sha256": _source_set_sha256(inventory_rows),
        "query_contract": {
            "preferred_url": JS_URL,
            "fallback_url": API_URL,
            "fallback_page_size": API_PAGE_SIZE,
            "values": "unit NAV and cumulative NAV; daily growth retained as diagnostic",
        },
        "inputs": [
            {"role": "etf_security_master", "path": _relative(MASTER_PATH), "sha256": master_auth["master_sha256"]},
            {"role": "etf_security_master_manifest", "path": _relative(MASTER_MANIFEST_PATH), "sha256": master_auth["manifest_sha256"]},
        ],
        "outputs": [
            {"role": "eastmoney_nav", "path": _relative(nav_output), "sha256": _sha256(nav_output), "rows": len(nav)},
            {"role": "asset_status", "path": _relative(status_output), "sha256": _sha256(status_output), "rows": len(status_frame)},
            {"role": "raw_response_inventory", "path": _relative(inventory_output), "sha256": _sha256(inventory_output), "rows": len(inventory)},
            {"role": "run_state", "path": _relative(state_path), "sha256": _sha256(state_path), "rows": 1},
        ],
        "code_path": _relative(code_path),
        "code_sha256": code_sha256,
        "code_files": code_files,
        "code_bundle_sha256": code_bundle_sha256,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "formal_table_promotion_allowed": False,
        "limitations": [
            "one current-final NAV observation does not establish historical source versions",
            "NAV is an independent economic-value series and cannot replace exchange execution prices",
            "liquidation distributions after the last regular NAV require separate official terminal-event accounting",
        ],
    }
    _atomic_json(manifest, final_manifest_path)
    latest = {**manifest, "immutable_manifest_path": _relative(final_manifest_path), "immutable_manifest_sha256": _sha256(final_manifest_path)}
    _atomic_json(latest, latest_manifest_path)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-07-17")
    parser.add_argument("--selection", choices=["all", "active", "delisted"], default="all")
    parser.add_argument("--assets", nargs="*")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
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
                    "pingzhongdata_assets",
                    "f10_fallback_assets",
                    "nav_rows",
                    "unit_nav_missing_source_rows_dropped",
                    "cumulative_nav_missing_rows",
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
