"""Collect official SSE announcements for unresolved historical stock status changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .pit_stock_name_history_collector import classify_security_name


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
NAME_STATUS_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_name_history" / "collection_status.json"
CALENDAR_PATH = ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv"
TUSHARE_DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_sse_status_announcements"
STATUS_PATH = RAW_DIR / "collection_status.json"
OUTPUT_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "sse_official_status_events.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_status_announcement_collector_latest.json"
QUERY_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
KEYWORDS = ("风险警示", "特别处理", "退市整理")
OUTPUT_COLUMNS = [
    "asset",
    "effective_date",
    "execution_status",
    "is_st",
    "announcement_date",
    "available_date",
    "announcement_title",
    "source_url",
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


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"})
    return session


def classify_announcement_title(title: str) -> str | None:
    compact = re.sub(r"\s+", "", str(title)).replace("撤消", "撤销")
    if re.search(r"(?:申请|提交|递交|上报|报送).*?(?:撤销|解除|取消)", compact):
        return None
    if re.search(r"(?:撤销|解除|取消).*?申请", compact):
        return None
    if "可能继续" in compact or re.search(r"存在(?:被)?实施.*?风险警示.*?可能", compact):
        return None
    if re.search(r"存在被实施.*?(?:风险警示|特别处理)", compact):
        return None
    if "退市整理期" in compact and "交易" in compact:
        return "delisting_date_requires_document"
    if any(
        token in compact
        for token in (
            "可能被",
            "可能实施",
            "暂不",
            "申请撤销",
            "不能撤销",
            "不予撤销",
            "未获撤销",
            "撤回",
            "进展",
            "风险提示",
            "预审计",
            "专项说明",
            "问询函",
            "回复",
        )
    ):
        return None
    removal = any(token in compact for token in ("撤销", "解除", "取消"))
    status_term = "风险警示" in compact or "特别处理" in compact
    if removal and status_term:
        if re.search(r"(?:并|暨|及)(?:将)?(?:继续)?(?:被)?实施(?:退市|其他)?风险警示", compact):
            return "risk_warning"
        if re.search(r"(?:并|暨|及)(?:将)?(?:继续)?(?:被)?(?:实施|实行)(?:其他)?特别处理", compact):
            return "risk_warning"
        if re.search(r"继续(?:被)?实施(?:退市|其他)?风险警示", compact):
            return "risk_warning"
        return "normal"
    if "风险警示" in compact and any(
        token in compact
        for token in (
            "实施",
            "被实施",
            "实行",
            "将实行",
            "将实施",
            "特别处理",
        )
    ):
        return "risk_warning"
    if "特别处理" in compact and any(
        token in compact for token in ("实施", "被实施", "实行", "被实行", "将实行", "将实施")
    ):
        return "risk_warning"
    if "退市风险警示提示性公告" in compact:
        return "risk_warning"
    return None


def next_market_session(date: str | pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    announcement = pd.Timestamp(date).normalize()
    eligible = calendar[calendar > announcement]
    if eligible.empty:
        raise ValueError(f"trade calendar has no session after {announcement.date()}")
    return pd.Timestamp(eligible[0]).normalize()


def market_session_on_or_after(date: str | pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    announcement = pd.Timestamp(date).normalize()
    eligible = calendar[calendar >= announcement]
    if eligible.empty:
        raise ValueError(f"trade calendar has no session on or after {announcement.date()}")
    return pd.Timestamp(eligible[0]).normalize()


def _target_assets() -> list[str]:
    status = json.loads(NAME_STATUS_PATH.read_text(encoding="utf-8"))
    assets = []
    for asset, entry in status.get("assets", {}).items():
        if entry.get("status") == "failed" and "undated execution-relevant names" in str(entry.get("error", "")):
            assets.append(str(asset).zfill(6))
    return sorted(set(assets))


def _cache_paths(asset: str) -> tuple[Path, Path]:
    return RAW_DIR / f"{asset}.json", RAW_DIR / f"{asset}.meta.json"


def _valid_cache(asset: str) -> bool:
    data_path, meta_path = _cache_paths(asset)
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        return bool(
            meta.get("status") == "success"
            and _sha256(data_path) == meta.get("sha256")
            and artifact.get("query_keywords") == list(KEYWORDS)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def query_asset(session: requests.Session, asset: str, as_of: pd.Timestamp) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        params = {
            "isPagination": "true",
            "productId": asset,
            "keyWord": keyword,
            "securityType": "0101,120100,020100,020200,120200",
            "reportType": "ALL",
            "beginDate": "1990-01-01",
            "endDate": as_of.date().isoformat(),
            "pageHelp.pageSize": "200",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.endPage": "5",
        }
        response = session.get(QUERY_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("pageHelp", {}).get("data", [])
        if not isinstance(rows, list):
            raise ValueError(f"SSE announcement response has invalid rows for {asset}")
        responses.append({"keyword": keyword, "request_url": response.url, "rows": rows})
    artifact = {
        "asset": asset,
        "as_of_date": as_of.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "responses": responses,
    }
    data_path, meta_path = _cache_paths(asset)
    _atomic_json(artifact, data_path)
    meta = {
        "status": "success",
        "asset": asset,
        "path": _relative(data_path),
        "sha256": _sha256(data_path),
        "rows": int(sum(len(item["rows"]) for item in responses)),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(meta, meta_path)
    return meta


def parse_asset_events(
    artifact: dict[str, Any],
    calendar: pd.DatetimeIndex,
    *,
    collapse_state_changes: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    asset = str(artifact["asset"]).zfill(6)
    bulletins: dict[str, dict[str, Any]] = {}
    for response in artifact.get("responses", []):
        for row in response.get("rows", []):
            url = str(row.get("URL", ""))
            if url:
                bulletins[url] = row
    events: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    delisting_first_notices: list[tuple[pd.Timestamp, str, str]] = []
    delisting_titles_seen = False
    for url, row in bulletins.items():
        title = str(row.get("TITLE", ""))
        transition = classify_announcement_title(title)
        if transition is None:
            continue
        announcement = pd.to_datetime(row.get("SSEDATE"), errors="coerce")
        if pd.isna(announcement):
            unresolved.append({"asset": asset, "title": title, "reason": "missing announcement date"})
            continue
        if transition == "delisting_date_requires_document":
            delisting_titles_seen = True
            compact = re.sub(r"\s+", "", title)
            if "首日" in compact or "第一次" in compact:
                delisting_first_notices.append((pd.Timestamp(announcement).normalize(), title, url))
            continue
        effective = next_market_session(announcement, calendar)
        events.append(
            {
                "asset": asset,
                "effective_date": effective,
                "execution_status": transition,
                "is_st": transition == "risk_warning",
                "announcement_date": pd.Timestamp(announcement).normalize(),
                "available_date": pd.Timestamp(announcement).normalize(),
                "announcement_title": title,
                "source_url": "https://www.sse.com.cn" + url,
            }
        )
    if delisting_first_notices:
        announcement, title, url = sorted(delisting_first_notices, key=lambda item: item[0])[0]
        effective = market_session_on_or_after(announcement, calendar)
        events.append(
            {
                "asset": asset,
                "effective_date": effective,
                "execution_status": "delisting",
                "is_st": False,
                "announcement_date": announcement,
                "available_date": announcement,
                "announcement_title": title,
                "source_url": "https://www.sse.com.cn" + url,
            }
        )
    elif delisting_titles_seen:
        unresolved.append(
            {
                "asset": asset,
                "title": "",
                "reason": "delisting announcements lack a first-day or first-risk notice",
            }
        )
    frame = pd.DataFrame(events)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS[:-2]), unresolved
    frame = frame.sort_values(["effective_date", "announcement_date", "source_url"])
    # Multiple announcements can describe the same resulting state. Preserve
    # the latest official announcement for an identical effective date/state.
    frame = frame.drop_duplicates(["asset", "effective_date", "execution_status"], keep="last")
    if not collapse_state_changes:
        return frame.reset_index(drop=True), unresolved
    state_changes = []
    current = "normal"
    for row in frame.itertuples(index=False):
        if row.execution_status == current:
            continue
        state_changes.append(row._asdict())
        current = row.execution_status
    return pd.DataFrame(state_changes, columns=OUTPUT_COLUMNS[:-2]), unresolved


def _master_current_status(as_of: pd.Timestamp) -> dict[str, str]:
    frame = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    frame = frame[frame["event_type"].eq("listing")].copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame = frame[frame["list_date"].le(as_of)]
    return {str(row.asset): classify_security_name(row.asset_name) for row in frame.itertuples(index=False)}


def _validate_delisting_trade_row(asset: str, date: pd.Timestamp) -> tuple[bool, Path]:
    path = TUSHARE_DAILY_DIR / f"trade_date={date:%Y%m%d}.csv"
    if not path.is_file():
        return False, path
    frame = pd.read_csv(path, usecols=["ts_code"], dtype={"ts_code": str})
    return bool(frame["ts_code"].str[:6].eq(asset).any()), path


def run_collection(as_of: str, collect_limit: int | None = None, sleep_seconds: float = 0.15) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    target_assets = _target_assets()
    pending = [asset for asset in target_assets if not _valid_cache(asset)]
    selected = pending[: max(0, collect_limit)] if collect_limit is not None else pending
    session = _session()
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "target_assets": len(target_assets),
        "cached_before": len(target_assets) - len(pending),
        "selected_assets": len(selected),
        "assets": {},
    }
    for position, asset in enumerate(selected):
        try:
            status["assets"][asset] = query_asset(session, asset, as_of_date)
        except (OSError, ValueError, requests.RequestException) as exc:
            status["assets"][asset] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        _atomic_json(status, STATUS_PATH)
        if sleep_seconds > 0 and position + 1 < len(selected):
            time.sleep(sleep_seconds)

    calendar_frame = pd.read_csv(CALENDAR_PATH, usecols=["date"])
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_frame["date"], errors="coerce").dropna().sort_values().unique())
    current_status = _master_current_status(as_of_date)
    frames: list[pd.DataFrame] = []
    unresolved_by_asset: dict[str, list[dict[str, str]]] = {}
    inputs = [
        {"source_id": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"source_id": "trade_calendar", "path": _relative(CALENDAR_PATH), "sha256": _sha256(CALENDAR_PATH)},
        {"source_id": "name_collection_status", "path": _relative(NAME_STATUS_PATH), "sha256": _sha256(NAME_STATUS_PATH)},
    ]
    resolved_assets: list[str] = []
    for asset in target_assets:
        if not _valid_cache(asset):
            continue
        data_path, meta_path = _cache_paths(asset)
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        events, unresolved = parse_asset_events(artifact, calendar)
        for event in events[events["execution_status"].eq("delisting")].itertuples(index=False):
            matched, daily_path = _validate_delisting_trade_row(asset, pd.Timestamp(event.effective_date))
            if not matched:
                unresolved.append(
                    {
                        "asset": asset,
                        "title": str(event.announcement_title),
                        "reason": "official delisting start has no matching Tushare traded row",
                    }
                )
            elif not any(item.get("path") == _relative(daily_path) for item in inputs):
                inputs.append(
                    {
                        "source_id": f"tushare_delisting_trade_row:{event.effective_date:%Y%m%d}",
                        "path": _relative(daily_path),
                        "sha256": _sha256(daily_path),
                    }
                )
        expected = current_status.get(asset, "unknown")
        expected_execution = "risk_warning" if expected == "risk_warning" else "normal" if expected in {"normal", "listing_marker"} else expected
        final = str(events.iloc[-1]["execution_status"]) if not events.empty else "normal"
        if final != expected_execution:
            unresolved.append(
                {"asset": asset, "title": "", "reason": f"final official status {final} does not match current master status {expected_execution}"}
            )
        if unresolved:
            unresolved_by_asset[asset] = unresolved
        else:
            resolved_assets.append(asset)
            frames.append(events)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        inputs.append({"source_id": f"sse_status_announcements:{asset}", "path": str(meta["path"]), "sha256": str(meta["sha256"])})

    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS[:-2])
    bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    output["data_source"] = "sse_official_company_announcements"
    output["source_vintage"] = f"sse_status_announcement_bundle_sha256:{bundle_hash}"
    output = output[OUTPUT_COLUMNS].sort_values(["asset", "effective_date"]).reset_index(drop=True)
    _atomic_csv(output, OUTPUT_PATH)
    complete = len(resolved_assets) == len(target_assets) and not unresolved_by_asset
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "source_vintage": f"sse_status_announcement_bundle_sha256:{bundle_hash}",
        "output_path": _relative(OUTPUT_PATH),
        "output_sha256": _sha256(OUTPUT_PATH),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(output)),
        "target_assets": len(target_assets),
        "resolved_assets": resolved_assets,
        "resolved_asset_count": len(resolved_assets),
        "unresolved_assets": unresolved_by_asset,
        "qualification_status": "READY_FOR_CROSS_SOURCE_VALIDATION" if complete else "PARTIAL_OFFICIAL_SUPPLEMENT",
        "method_boundary": (
            "risk-warning events use the first exchange trading session after the official announcement date; "
            "delisting starts use the official first-day/first-risk notice and require a matching Tushare traded row"
        ),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update({"resolved_asset_count": len(resolved_assets), "unresolved_asset_count": len(unresolved_by_asset)})
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--collect-limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_collection(args.as_of, args.collect_limit, args.sleep_seconds)
    keys = ("qualification_status", "rows", "target_assets", "resolved_asset_count", "historical_backtest_allowed")
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
