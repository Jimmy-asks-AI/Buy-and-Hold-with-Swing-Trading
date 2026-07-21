"""Discover official terminal-event evidence for every delisted A-share ETF.

The output is an observation-only coverage registry. Announcement titles are
useful for routing documents to the correct parser, but they do not establish a
cash amount, successor-share ratio, or absence of a terminal value transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import pandas as pd
import requests

from . import pit_etf_cninfo_share_action_announcement_collector as cninfo_api
from . import pit_etf_dividend_announcement_collector as official
from . import pit_etf_sse_share_action_announcement_collector as sse_api
from .pit_etf_dividend_universe_coverage_collector import recover_delisted_cninfo_identity


ROOT = Path(__file__).resolve().parents[2]
ETF_MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
ETF_MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
TENCENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tencent_etf_price_validation_latest.json"
)
EASTMONEY_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "eastmoney_etf_nav_validation_latest.json"
)
FORMAL_EVENT_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_terminal_cash_events.csv"
FORMAL_EVENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_terminal_cash_events_builder_latest.json"
)
DIVIDEND_QUERY_DIR = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_dividend_announcements" / "queries"
)

RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_terminal_event_universe"
QUERY_DIR = RAW_DIR / "queries"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
LIFECYCLE_PATH = OBSERVATION_DIR / "etf_terminal_event_lifecycle_inventory.csv"
ANNOUNCEMENT_PATH = OBSERVATION_DIR / "etf_terminal_event_official_announcements.csv"
COVERAGE_PATH = OBSERVATION_DIR / "etf_terminal_event_coverage_registry.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_terminal_event_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_universe" / "discovery_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_universe_collector_latest.json"
)

SCHEMA_VERSION = 2
QUERY_CONTRACT_VERSION = 2
CHINA_TIMEZONE = "Asia/Shanghai"
PAGE_SIZE_LIMIT = 100
KEYWORDS = (
    "终止上市",
    "基金合同终止",
    "终止基金合同",
    "清算",
    "剩余财产",
    "转型",
    "份额转换",
    "份额折算",
    "合并",
)
FINAL_STATES = {
    "terminal_event_identified",
    "official_no_terminal_value_transfer",
    "evidence_insufficient",
}
ANNOUNCEMENT_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "published_at",
    "announcement_title",
    "source_url",
    "source_type",
    "matched_keywords_json",
    "title_tags_json",
    "candidate_event_types_json",
    "source_observed_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "data_source",
    "source_vintage",
    "document_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
LIFECYCLE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "master_delist_date",
    "last_price_date",
    "last_close",
    "last_volume_shares",
    "last_nav_date",
    "last_unit_nav",
    "last_cumulative_nav",
    "delist_minus_last_price_days",
    "delist_minus_last_nav_days",
    "last_nav_minus_last_price_days",
    "lifecycle_snapshot_available_date",
    "lifecycle_snapshot_is_current_final",
    "historical_backtest_allowed",
]
COVERAGE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "master_delist_date",
    "query_complete",
    "official_announcement_count",
    "first_announcement_date",
    "last_announcement_date",
    "candidate_event_types_json",
    "candidate_event_type_latest_dates_json",
    "primary_candidate_class",
    "discovery_state",
    "final_evidence_state",
    "final_event_type",
    "formal_event_count",
    "formal_event_chain_complete",
    "terminal_value_amount_known",
    "successor_asset_known",
    "position_extinguishment_known",
    "document_validation_required",
    "review_priority",
    "review_reason",
    "terminal_event_historical_backtest_allowed",
    "universe_terminal_coverage_complete",
    "model_promotion_allowed",
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


def _clean_title(value: Any) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()


def _china_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(CHINA_TIMEZONE)
    return timestamp.tz_convert(CHINA_TIMEZONE)


def _query_contract_fingerprint(exchange: str) -> str:
    payload = {
        "contract_version": QUERY_CONTRACT_VERSION,
        "exchange": exchange,
        "keywords": list(KEYWORDS),
        "endpoint": sse_api.QUERY_URL if exchange == "SSE" else cninfo_api.QUERY_URL,
        "sql_id": sse_api.SQL_ID if exchange == "SSE" else "cninfo_announcement_query",
        "page_size_limit": PAGE_SIZE_LIMIT,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _query_dependency_hashes() -> dict[str, str]:
    return {
        _relative(Path(module.__file__).resolve()): _sha256(Path(module.__file__).resolve())
        for module in (official, sse_api, cninfo_api)
    }


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(str(value) for value in values if str(value))), ensure_ascii=False)


def classify_title(title: str) -> tuple[list[str], list[str]]:
    """Return routing tags and candidate event types without final classification."""

    compact = re.sub(r"\s+", "", str(title))
    tags: list[str] = []
    if "终止上市" in compact or "终止上市交易" in compact:
        tags.append("delisting")
    if ("基金合同" in compact and "终止" in compact) or "终止基金合同" in compact:
        tags.append("fund_contract_termination")
    if "清算" in compact:
        tags.append("liquidation")
    if "剩余财产" in compact or "清算款" in compact:
        tags.append("terminal_distribution")
    if "转型" in compact or "变更注册" in compact or "变更基金类别" in compact:
        tags.append("transformation")
    if any(marker in compact for marker in ("份额转换", "变更登记", "确权登记", "份额折算")):
        tags.append("successor_share_conversion")
    if "合并" in compact and "份额合并" not in compact:
        tags.append("fund_merger")
    if "持有人大会" in compact and any(marker in compact for marker in ("表决结果", "决议生效")):
        tags.append("holder_resolution")

    candidates: list[str] = []
    if "fund_merger" in tags:
        candidates.append("merger_or_share_exchange")
    if "transformation" in tags or "successor_share_conversion" in tags:
        candidates.append("conversion_to_successor")
    if any(tag in tags for tag in ("liquidation", "terminal_distribution", "fund_contract_termination")):
        candidates.append("cash_liquidation_or_extinguishment")
    if "delisting" in tags:
        candidates.append("exchange_delisting")
    return sorted(set(tags)), sorted(set(candidates))


def _primary_candidate_class(
    event_types: set[str],
    latest_dates: dict[str, pd.Timestamp] | None = None,
) -> str:
    classes = {
        "merger_or_share_exchange": "merger_or_share_exchange_candidate",
        "conversion_to_successor": "successor_share_candidate",
        "cash_liquidation_or_extinguishment": "cash_or_extinguishment_candidate",
    }
    economic_types = event_types.intersection(classes)
    if economic_types and latest_dates:
        latest = max(latest_dates[event_type] for event_type in economic_types)
        tied = {event_type for event_type in economic_types if latest_dates[event_type] == latest}
        if len(tied) > 1:
            return "mixed_terminal_mechanism_candidate"
        return classes[next(iter(tied))]
    if economic_types:
        return "mixed_terminal_mechanism_candidate" if len(economic_types) > 1 else classes[next(iter(economic_types))]
    if "exchange_delisting" in event_types:
        return "delisting_evidence_only"
    return "no_terminal_announcement_match"


def _query_paths(exchange: str, asset: str) -> tuple[Path, Path]:
    directory = QUERY_DIR / exchange.lower()
    return directory / f"{asset}.json", directory / f"{asset}.meta.json"


def _valid_query_cache(exchange: str, asset: str, as_of_date: pd.Timestamp) -> bool:
    data_path, meta_path = _query_paths(exchange, asset)
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        responses = artifact.get("responses", [])
        response_keywords = [str(item.get("keyword", "")) for item in responses]
        response_groups_complete = bool(
            len(responses) == len(KEYWORDS)
            and sorted(response_keywords) == sorted(KEYWORDS)
            and all(
                int(item.get("page_count", 0)) >= 1
                and len(item.get("requests", item.get("request_urls", [])))
                == int(item.get("page_count", 0))
                and isinstance(item.get("rows"), list)
                for item in responses
            )
        )
        return bool(
            metadata.get("status") == "success"
            and metadata.get("sha256") == _sha256(data_path)
            and metadata.get("producer_code_sha256") == _sha256(Path(__file__).resolve())
            and metadata.get("dependency_hashes") == _query_dependency_hashes()
            and metadata.get("request_contract_fingerprint") == _query_contract_fingerprint(exchange)
            and artifact.get("schema_version") == SCHEMA_VERSION
            and artifact.get("query_contract_version") == QUERY_CONTRACT_VERSION
            and artifact.get("asset") == asset
            and artifact.get("exchange") == exchange
            and artifact.get("as_of_date") == as_of_date.date().isoformat()
            and artifact.get("query_keywords") == list(KEYWORDS)
            and artifact.get("request_contract_fingerprint") == _query_contract_fingerprint(exchange)
            and response_groups_complete
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _save_query_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    exchange, asset = str(artifact["exchange"]), str(artifact["asset"])
    data_path, meta_path = _query_paths(exchange, asset)
    _atomic_json(artifact, data_path)
    metadata = {
        "status": "success",
        "asset": asset,
        "exchange": exchange,
        "path": _relative(data_path),
        "sha256": _sha256(data_path),
        "producer_code_path": _relative(Path(__file__).resolve()),
        "producer_code_sha256": _sha256(Path(__file__).resolve()),
        "dependency_hashes": _query_dependency_hashes(),
        "request_contract_fingerprint": _query_contract_fingerprint(exchange),
        "rows": int(sum(len(item.get("rows", [])) for item in artifact.get("responses", []))),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(metadata, meta_path)
    return metadata


def load_targets(
    *,
    assets: list[str] | None = None,
    exchange: str = "all",
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    manifest = json.loads(ETF_MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not True
        or str(manifest.get("output_path", "")).replace("\\", "/") != _relative(ETF_MASTER_PATH)
        or str(manifest.get("output_sha256", "")) != _sha256(ETF_MASTER_PATH)
    ):
        raise ValueError("ETF master lineage does not authenticate the delisted universe")
    master = pd.read_csv(ETF_MASTER_PATH, dtype={"asset": str})
    required = {"asset", "asset_name", "exchange", "list_status", "event_type", "list_date", "delist_date"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"ETF master misses lifecycle columns: {missing}")
    listed = master[
        master["list_status"].eq("listed") & master["event_type"].eq("listing")
    ][["asset", "asset_name", "exchange", "list_date"]].copy()
    delisted = master[
        master["list_status"].eq("delisted") & master["event_type"].eq("delisting")
    ][["asset", "delist_date"]].copy()
    targets = listed.merge(delisted, on="asset", how="inner", validate="one_to_one")
    targets["asset"] = targets["asset"].astype(str).str.zfill(6)
    targets["list_date"] = pd.to_datetime(targets["list_date"], errors="coerce").dt.normalize()
    targets["delist_date"] = pd.to_datetime(targets["delist_date"], errors="coerce").dt.normalize()
    if targets[["list_date", "delist_date"]].isna().any().any():
        raise ValueError("delisted ETF universe has invalid lifecycle dates")
    if as_of_date is not None:
        targets = targets[targets["delist_date"].le(pd.Timestamp(as_of_date).normalize())]
    if targets["asset"].duplicated().any() or not set(targets["exchange"]).issubset({"SSE", "SZSE"}):
        raise ValueError("delisted ETF universe has duplicates or unsupported exchanges")
    if exchange != "all":
        targets = targets[targets["exchange"].eq(exchange)]
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        missing_assets = requested.difference(set(targets["asset"]))
        if missing_assets:
            raise ValueError(f"requested assets are not delisted ETFs: {sorted(missing_assets)}")
        targets = targets[targets["asset"].isin(requested)]
    if targets.empty:
        raise ValueError("terminal-event discovery selected no targets")
    return targets.sort_values(["exchange", "asset"]).reset_index(drop=True)


def _load_cached_cninfo_identity(asset: str, as_of_date: pd.Timestamp) -> dict[str, Any] | None:
    data_path = DIVIDEND_QUERY_DIR / "szse" / f"{asset}.json"
    meta_path = DIVIDEND_QUERY_DIR / "szse" / f"{asset}.meta.json"
    if not data_path.is_file() or not meta_path.is_file():
        return None
    try:
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    identity = artifact.get("identity", {})
    if not (
        metadata.get("status") == "success"
        and metadata.get("sha256") == _sha256(data_path)
        and artifact.get("asset") == asset
        and artifact.get("as_of_date") == as_of_date.date().isoformat()
        and str(identity.get("org_id", "")).strip()
    ):
        return None
    return {**identity, "resolution_source_path": _relative(data_path), "resolution_source_sha256": _sha256(data_path)}


def _resolve_cninfo_identity(
    session: requests.Session,
    target: Any,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    asset = str(target.asset)
    cached = _load_cached_cninfo_identity(asset, as_of_date)
    if cached is not None:
        return cached
    response = session.post(cninfo_api.IDENTITY_URL, data={"keyWord": asset, "maxNum": "20"}, timeout=30)
    response.raise_for_status()
    rows = response.json()
    if isinstance(rows, list):
        try:
            return {**cninfo_api.select_fund_identity(asset, rows), "resolution": "direct_exact_fund_identity"}
        except ValueError:
            pass
    return recover_delisted_cninfo_identity(
        session,
        asset=asset,
        asset_name=str(target.asset_name),
        list_date=pd.Timestamp(target.list_date),
        delist_date=pd.Timestamp(target.delist_date),
        as_of_date=as_of_date,
    )


def _fetch_sse_query(session: requests.Session, target: Any, as_of_date: pd.Timestamp) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        requests_made: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            parameters = sse_api._query_params(str(target.asset), keyword, as_of_date, page_number)
            response = session.get(
                sse_api.QUERY_URL,
                params=parameters,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result", [])
            if not isinstance(result, list):
                raise ValueError(f"SSE terminal-event query returned invalid rows for {target.asset}/{keyword}")
            rows.extend(result)
            requests_made.append(
                {"method": "GET", "url": response.url, "page_number": page_number, "parameters": parameters}
            )
            page_count = max(1, int((payload.get("pageHelp") or {}).get("pageCount") or 1))
            if page_count > PAGE_SIZE_LIMIT:
                raise ValueError(f"SSE terminal-event query page count is implausible: {target.asset}/{keyword}/{page_count}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "requests": requests_made, "rows": rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "query_contract_version": QUERY_CONTRACT_VERSION,
        "asset": str(target.asset),
        "asset_name": str(target.asset_name),
        "exchange": "SSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "request_contract_fingerprint": _query_contract_fingerprint("SSE"),
        "sql_id": sse_api.SQL_ID,
        "responses": responses,
    }


def _fetch_cninfo_query(session: requests.Session, target: Any, as_of_date: pd.Timestamp) -> dict[str, Any]:
    identity = _resolve_cninfo_identity(session, target, as_of_date)
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        requests_made: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            parameters = cninfo_api._query_params(
                str(target.asset), str(identity["org_id"]), keyword, as_of_date, page_number
            )
            response = session.post(
                cninfo_api.QUERY_URL,
                data=parameters,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo terminal-event query returned invalid rows for {target.asset}/{keyword}")
            rows.extend(announcements)
            requests_made.append(
                {
                    "method": "POST",
                    "url": cninfo_api.QUERY_URL,
                    "page_number": page_number,
                    "parameters": parameters,
                }
            )
            page_count = max(1, int(payload.get("totalpages") or 1))
            if page_count > PAGE_SIZE_LIMIT:
                raise ValueError(f"CNInfo terminal-event query page count is implausible: {target.asset}/{keyword}/{page_count}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "requests": requests_made, "rows": rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "query_contract_version": QUERY_CONTRACT_VERSION,
        "asset": str(target.asset),
        "asset_name": str(target.asset_name),
        "exchange": "SZSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "request_contract_fingerprint": _query_contract_fingerprint("SZSE"),
        "identity": identity,
        "responses": responses,
    }


def parse_query_artifacts(artifacts: list[dict[str, Any]]) -> pd.DataFrame:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        asset = str(artifact["asset"]).zfill(6)
        exchange = str(artifact["exchange"])
        artifact_cutoff = pd.Timestamp(str(artifact["as_of_date"])).normalize()
        query_path, _ = _query_paths(exchange, asset)
        query_hash = _sha256(query_path)
        for response in artifact.get("responses", []):
            keyword = str(response.get("keyword", ""))
            for raw in response.get("rows", []):
                if exchange == "SSE":
                    raw_asset = str(raw.get("SECURITY_CODE", "")).zfill(6)
                    title = _clean_title(raw.get("TITLE", ""))
                    raw_url = str(raw.get("URL", ""))
                    source_url = urljoin("https://www.sse.com.cn", raw_url)
                    announcement_date = pd.to_datetime(raw.get("SSEDATE"), errors="coerce")
                    published_at: pd.Timestamp | None = None
                    source_type = str(raw.get("BULLETIN_TYPE", "SSE fund announcement"))
                    data_source = "SSE official fund announcement query"
                else:
                    raw_asset = str(raw.get("secCode", "")).zfill(6)
                    title = _clean_title(raw.get("announcementTitle", ""))
                    raw_url = str(raw.get("adjunctUrl", ""))
                    source_url = urljoin(cninfo_api.STATIC_BASE_URL, raw_url)
                    announcement_date = cninfo_api.parse_announcement_time(raw.get("announcementTime"))
                    published_at = (
                        _china_timestamp(announcement_date) if pd.notna(announcement_date) else None
                    )
                    source_type = str(raw.get("announcementTypeName", "CNInfo fund announcement"))
                    data_source = "CNInfo official fund announcement query"
                if raw_asset != asset or not title or not raw_url or pd.isna(announcement_date):
                    continue
                date = pd.Timestamp(announcement_date).normalize()
                # Daily strategies may only consume an announcement from the next
                # calendar day. This is conservative for date-only SSE records and
                # prevents same-day use of after-close CNInfo publications.
                available_trade_date = date + pd.Timedelta(days=1)
                if available_trade_date > artifact_cutoff:
                    continue
                available_at = _china_timestamp(available_trade_date)
                tags, candidate_types = classify_title(title)
                key = (asset, source_url)
                existing = records.get(key)
                if existing is None:
                    existing = {
                        "asset": asset,
                        "asset_name": str(artifact.get("asset_name", "")),
                        "exchange": exchange,
                        "announcement_date": date,
                        "published_at": published_at.isoformat() if published_at is not None else "",
                        "announcement_title": title,
                        "source_url": source_url,
                        "source_type": source_type,
                        "matched_keywords": set(),
                        "title_tags": set(),
                        "candidate_event_types": set(),
                        "source_observed_at": str(artifact.get("fetched_at", "")),
                        "available_at": available_at.isoformat(),
                        "available_trade_date": available_trade_date,
                        "available_date": available_trade_date,
                        "data_source": data_source,
                        "source_vintage": f"official_terminal_query_sha256:{query_hash}",
                        "document_validation_status": "not_started",
                        "historical_backtest_allowed": False,
                        "model_promotion_allowed": False,
                    }
                    records[key] = existing
                existing["matched_keywords"].add(keyword)
                existing["title_tags"].update(tags)
                existing["candidate_event_types"].update(candidate_types)
    rows: list[dict[str, Any]] = []
    for record in records.values():
        row = dict(record)
        row["matched_keywords_json"] = _json_list(row.pop("matched_keywords"))
        row["title_tags_json"] = _json_list(row.pop("title_tags"))
        row["candidate_event_types_json"] = _json_list(row.pop("candidate_event_types"))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    frame = pd.DataFrame(rows, columns=ANNOUNCEMENT_COLUMNS)
    if (pd.to_datetime(frame["available_trade_date"]) > pd.to_datetime(frame["available_date"])).any():
        raise ValueError("terminal announcement availability columns are inconsistent")
    return frame.sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def _manifest_output(manifest_path: Path, role: str) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("outputs", []) if str(item.get("role")) == role]
    if len(matches) != 1:
        raise ValueError(f"manifest does not identify one {role} output: {manifest_path}")
    path = ROOT / str(matches[0].get("path", ""))
    if not path.is_file() or _sha256(path) != str(matches[0].get("sha256", "")):
        raise ValueError(f"manifest output hash mismatch: {manifest_path}/{role}")
    return path, manifest


def _tail_snapshot(
    path: Path,
    target_assets: set[str],
    *,
    value_columns: list[str],
) -> pd.DataFrame:
    columns = ["date", "asset", *value_columns]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, dtype={"asset": str}, chunksize=200_000):
        chunk["asset"] = chunk["asset"].astype(str).str.zfill(6)
        selected = chunk[chunk["asset"].isin(target_assets)].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(pieces, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    return frame.sort_values(["asset", "date"]).groupby("asset", as_index=False).tail(1).reset_index(drop=True)


def build_lifecycle_inventory(targets: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    price_path, price_manifest = _manifest_output(TENCENT_MANIFEST_PATH, "tencent_raw_prices")
    nav_path, nav_manifest = _manifest_output(EASTMONEY_MANIFEST_PATH, "eastmoney_nav")
    target_assets = set(targets["asset"])
    price = _tail_snapshot(price_path, target_assets, value_columns=["close", "volume_shares"]).rename(
        columns={"date": "last_price_date", "close": "last_close", "volume_shares": "last_volume_shares"}
    )
    nav = _tail_snapshot(nav_path, target_assets, value_columns=["unit_nav", "cumulative_nav"]).rename(
        columns={"date": "last_nav_date", "unit_nav": "last_unit_nav", "cumulative_nav": "last_cumulative_nav"}
    )
    lifecycle = targets.rename(columns={"delist_date": "master_delist_date"}).merge(
        price, on="asset", how="left", validate="one_to_one"
    ).merge(nav, on="asset", how="left", validate="one_to_one")
    for column in ("master_delist_date", "last_price_date", "last_nav_date"):
        lifecycle[column] = pd.to_datetime(lifecycle[column], errors="coerce").dt.normalize()
    lifecycle["delist_minus_last_price_days"] = (
        lifecycle["master_delist_date"] - lifecycle["last_price_date"]
    ).dt.days
    lifecycle["delist_minus_last_nav_days"] = (
        lifecycle["master_delist_date"] - lifecycle["last_nav_date"]
    ).dt.days
    lifecycle["last_nav_minus_last_price_days"] = (
        lifecycle["last_nav_date"] - lifecycle["last_price_date"]
    ).dt.days
    observed_dates = [
        pd.to_datetime(price_manifest.get("created_at"), errors="coerce"),
        pd.to_datetime(nav_manifest.get("created_at"), errors="coerce"),
    ]
    observed = max(value for value in observed_dates if pd.notna(value))
    lifecycle["lifecycle_snapshot_available_date"] = pd.Timestamp(observed).normalize()
    lifecycle["lifecycle_snapshot_is_current_final"] = True
    lifecycle["historical_backtest_allowed"] = False
    inputs = [
        {"role": "tencent_price_manifest", "path": _relative(TENCENT_MANIFEST_PATH), "sha256": _sha256(TENCENT_MANIFEST_PATH)},
        {"role": "tencent_price", "path": _relative(price_path), "sha256": _sha256(price_path)},
        {"role": "eastmoney_nav_manifest", "path": _relative(EASTMONEY_MANIFEST_PATH), "sha256": _sha256(EASTMONEY_MANIFEST_PATH)},
        {"role": "eastmoney_nav", "path": _relative(nav_path), "sha256": _sha256(nav_path)},
    ]
    return lifecycle.reindex(columns=LIFECYCLE_COLUMNS).sort_values("asset").reset_index(drop=True), inputs


def _eligible_formal_events(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {
        "asset",
        "event_type",
        "pay_date",
        "available_date",
        "cash_per_share",
        "extinguishes_position",
        "source_pdf_sha256_set",
        "historical_backtest_allowed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"formal ETF terminal-event table misses PIT fields: {missing}")
    eligible = frame.copy()
    eligible["asset"] = eligible["asset"].astype(str).str.zfill(6)
    for column in ("pay_date", "available_date"):
        eligible[column] = pd.to_datetime(eligible[column], errors="coerce").dt.normalize()
    if eligible[["pay_date", "available_date"]].isna().any().any():
        raise ValueError("formal ETF terminal-event table has invalid PIT dates")
    if not eligible["historical_backtest_allowed"].astype(str).str.lower().eq("true").all():
        raise ValueError("formal ETF terminal-event table contains disabled rows")
    duplicate_key = ["asset", "event_type", "pay_date", "source_pdf_sha256_set"]
    if eligible.duplicated(duplicate_key, keep=False).any():
        raise ValueError("formal ETF terminal-event table has duplicate economic event keys")
    if "event_id" not in eligible.columns:
        eligible["event_id"] = [
            hashlib.sha256("|".join(str(row[column]) for column in duplicate_key).encode("utf-8")).hexdigest()
            for _, row in eligible.iterrows()
        ]
    if eligible["event_id"].astype(str).duplicated().any():
        raise ValueError("formal ETF terminal-event table has duplicate event_id values")
    eligible = eligible[
        eligible["available_date"].le(pd.Timestamp(cutoff).normalize())
        & eligible["pay_date"].le(pd.Timestamp(cutoff).normalize())
    ].copy()
    if "distribution_sequence" not in eligible.columns:
        eligible = eligible.sort_values(["asset", "pay_date", "event_id"])
        eligible["distribution_sequence"] = eligible.groupby("asset").cumcount() + 1
    return eligible.reset_index(drop=True)


def _load_formal_events(cutoff: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not FORMAL_EVENT_PATH.is_file() or not FORMAL_EVENT_MANIFEST_PATH.is_file():
        return pd.DataFrame(), []
    manifest = json.loads(FORMAL_EVENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    outputs = [item for item in manifest.get("outputs", []) if item.get("role") == "pit_etf_terminal_cash_events"]
    if (
        len(outputs) != 1
        or str(outputs[0].get("path", "")).replace("\\", "/") != _relative(FORMAL_EVENT_PATH)
        or str(outputs[0].get("sha256", "")) != _sha256(FORMAL_EVENT_PATH)
    ):
        raise ValueError("formal ETF terminal-event manifest does not authenticate its table")
    frame = _eligible_formal_events(
        pd.read_csv(FORMAL_EVENT_PATH, dtype={"asset": str}), pd.Timestamp(cutoff)
    )
    inputs = [
        {"role": "formal_terminal_event_manifest", "path": _relative(FORMAL_EVENT_MANIFEST_PATH), "sha256": _sha256(FORMAL_EVENT_MANIFEST_PATH)},
        {"role": "formal_terminal_event_table", "path": _relative(FORMAL_EVENT_PATH), "sha256": _sha256(FORMAL_EVENT_PATH)},
    ]
    return frame, inputs


def build_coverage_registry(
    targets: pd.DataFrame,
    announcements: pd.DataFrame,
    query_complete_assets: set[str],
    formal_events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff).normalize()
    if (pd.to_datetime(targets["delist_date"]).dt.normalize() > cutoff).any():
        raise ValueError("terminal-event target universe includes delist dates after cutoff")
    eligible_formal = _eligible_formal_events(formal_events, cutoff)
    formal_by_asset = (
        {
            str(asset).zfill(6): group.sort_values(
                ["pay_date", "distribution_sequence", "event_id"]
            )
            for asset, group in eligible_formal.groupby("asset", sort=False)
        }
        if not eligible_formal.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        asset = str(target.asset)
        selected = announcements[announcements["asset"].eq(asset)]
        event_types: set[str] = set()
        latest_dates: dict[str, pd.Timestamp] = {}
        for value in selected.get("candidate_event_types_json", pd.Series(dtype=str)).dropna():
            event_types.update(json.loads(str(value)))
        for announcement in selected.itertuples(index=False):
            date = pd.Timestamp(announcement.announcement_date)
            for event_type in json.loads(str(announcement.candidate_event_types_json)):
                previous = latest_dates.get(event_type)
                if previous is None or date > previous:
                    latest_dates[event_type] = date
        primary = _primary_candidate_class(event_types, latest_dates)
        query_complete = asset in query_complete_assets
        formal = formal_by_asset.get(asset)
        if formal is not None and not formal.empty:
            extinguishment = formal["extinguishes_position"].astype(str).str.lower().eq("true")
            extinguishing_rows = int(extinguishment.sum())
            latest_event_id = str(formal.iloc[-1]["event_id"])
            extinguishing_event_ids = set(formal.loc[extinguishment, "event_id"].astype(str))
            chain_complete = extinguishing_rows == 1 and latest_event_id in extinguishing_event_ids
            formal_event_types = sorted(set(formal["event_type"].astype(str)))
            final_state = "terminal_event_identified"
            final_event_type = (
                formal_event_types[0]
                if len(formal_event_types) == 1
                else _json_list(formal_event_types)
            )
            amount_known = bool(formal["cash_per_share"].notna().all())
            successor_known = False
            extinguishment_known = bool(chain_complete)
            document_required = not chain_complete
            discovery_state = (
                "validated_formal_terminal_event_chain"
                if chain_complete
                else "validated_formal_interim_events_chain_incomplete"
            )
            priority = "DONE" if chain_complete else "P1"
            reason = (
                "Independent validator promoted a complete ordered official terminal-event chain."
                if chain_complete
                else "Validated interim events exist, but the final position-extinguishment event is not established."
            )
            historical_allowed = bool(
                formal["historical_backtest_allowed"].astype(str).str.lower().eq("true").all()
            )
            formal_event_count = int(len(formal))
        else:
            final_state = "evidence_insufficient"
            final_event_type = ""
            amount_known = False
            successor_known = False
            extinguishment_known = False
            document_required = True
            historical_allowed = False
            formal_event_count = 0
            chain_complete = False
            if not query_complete:
                discovery_state, priority = "official_query_incomplete", "P0"
                reason = "Official exchange query is missing or unauthenticated."
            elif primary == "no_terminal_announcement_match":
                discovery_state, priority = "official_query_no_match_requires_no_event_proof", "P0"
                reason = "A zero-title query result is not proof that no terminal value transfer occurred."
            elif primary == "delisting_evidence_only":
                discovery_state, priority = "delisting_evidence_only", "P0"
                reason = "Delisting evidence does not establish cash, successor shares, continuation, or position extinguishment."
            else:
                discovery_state, priority = "terminal_event_candidate_found", "P1"
                reason = "Candidate documents must be downloaded, parsed, reconciled, and independently validated."
        if final_state not in FINAL_STATES:
            raise ValueError(f"invalid terminal-event final state for {asset}: {final_state}")
        rows.append(
            {
                "asset": asset,
                "asset_name": str(target.asset_name),
                "exchange": str(target.exchange),
                "list_date": pd.Timestamp(target.list_date),
                "master_delist_date": pd.Timestamp(target.delist_date),
                "query_complete": query_complete,
                "official_announcement_count": int(len(selected)),
                "first_announcement_date": selected["announcement_date"].min() if not selected.empty else pd.NaT,
                "last_announcement_date": selected["announcement_date"].max() if not selected.empty else pd.NaT,
                "candidate_event_types_json": _json_list(event_types),
                "candidate_event_type_latest_dates_json": json.dumps(
                    {key: value.date().isoformat() for key, value in sorted(latest_dates.items())},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "primary_candidate_class": primary,
                "discovery_state": discovery_state,
                "final_evidence_state": final_state,
                "final_event_type": final_event_type,
                "formal_event_count": formal_event_count,
                "formal_event_chain_complete": bool(chain_complete),
                "terminal_value_amount_known": bool(amount_known),
                "successor_asset_known": bool(successor_known),
                "position_extinguishment_known": bool(extinguishment_known),
                "document_validation_required": bool(document_required),
                "review_priority": priority,
                "review_reason": reason,
                "terminal_event_historical_backtest_allowed": bool(historical_allowed),
                "universe_terminal_coverage_complete": False,
                "model_promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS).sort_values(["review_priority", "asset"]).reset_index(drop=True)


def run_collection(
    as_of: str | pd.Timestamp,
    *,
    assets: list[str] | None = None,
    exchange: str = "all",
    max_assets: int | None = None,
    sleep_seconds: float = 0.1,
    max_consecutive_failures: int = 5,
) -> dict[str, Any]:
    cutoff = _china_timestamp(as_of).normalize().tz_localize(None)
    targets = load_targets(assets=assets, exchange=exchange, as_of_date=cutoff)
    pending = [
        row
        for row in targets.itertuples(index=False)
        if not _valid_query_cache(str(row.exchange), str(row.asset), cutoff)
    ]
    selected = pending[:max_assets] if max_assets is not None else pending
    if STATUS_PATH.is_file():
        try:
            previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            previous = {}
    else:
        previous = {}
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "target_assets": int(len(targets)),
        "pending_assets_at_start": int(len(pending)),
        "selected_uncached_assets": int(len(selected)),
        "assets": previous.get("assets", {}) if previous.get("as_of_date") == cutoff.date().isoformat() else {},
    }
    sessions = {
        "SSE": official._session("https://www.sse.com.cn/disclosure/fund/announcement/index.shtml"),
        "SZSE": official._session("https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"),
    }
    failures = 0
    for position, target in enumerate(selected, start=1):
        asset, target_exchange = str(target.asset), str(target.exchange)
        try:
            artifact = (
                _fetch_sse_query(sessions["SSE"], target, cutoff)
                if target_exchange == "SSE"
                else _fetch_cninfo_query(sessions["SZSE"], target, cutoff)
            )
            status["assets"][asset] = _save_query_artifact(artifact)
            failures = 0
        except (OSError, ValueError, requests.RequestException) as exc:
            status["assets"][asset] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            failures += 1
        status["queries_processed"] = position
        _atomic_json(status, STATUS_PATH)
        if failures >= max_consecutive_failures:
            status["circuit_breaker"] = f"stopped_after_{failures}_consecutive_failures"
            _atomic_json(status, STATUS_PATH)
            break
        if sleep_seconds > 0 and position < len(selected):
            time.sleep(sleep_seconds)

    artifacts: list[dict[str, Any]] = []
    query_inputs: list[dict[str, Any]] = []
    query_complete_assets: set[str] = set()
    for target in targets.itertuples(index=False):
        target_exchange, asset = str(target.exchange), str(target.asset)
        if not _valid_query_cache(target_exchange, asset, cutoff):
            continue
        query_complete_assets.add(asset)
        data_path, meta_path = _query_paths(target_exchange, asset)
        artifacts.append(json.loads(data_path.read_text(encoding="utf-8")))
        query_inputs.extend(
            [
                {"role": f"official_terminal_query:{target_exchange}:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"role": f"official_terminal_query_meta:{target_exchange}:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )

    announcements = parse_query_artifacts(artifacts)
    lifecycle, lifecycle_inputs = build_lifecycle_inventory(targets)
    # Discovery is deliberately independent from the promoted event registry.
    # A separate settlement builder consumes both outputs, preventing a lineage
    # cycle where promotion rewrites an input of its own evidence chain.
    formal_events, formal_inputs = pd.DataFrame(), []
    coverage = build_coverage_registry(
        targets,
        announcements,
        query_complete_assets,
        formal_events,
        cutoff=cutoff,
    )
    review = coverage[coverage["review_priority"].ne("DONE")].copy()

    _atomic_csv(lifecycle, LIFECYCLE_PATH)
    _atomic_csv(announcements, ANNOUNCEMENT_PATH)
    _atomic_csv(coverage, COVERAGE_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)
    states = coverage["final_evidence_state"].value_counts().to_dict()
    classes = coverage["primary_candidate_class"].value_counts().to_dict()
    query_complete = len(query_complete_assets)
    qualification = (
        "FULL_RELATIVE_TO_AUTHENTICATED_MASTER_OFFICIAL_DISCOVERY_REQUIRES_DOCUMENT_VALIDATION"
        if query_complete == len(targets)
        else "PARTIAL_DELISTED_UNIVERSE_DISCOVERY"
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": qualification,
        "target_assets": int(len(targets)),
        "universe_basis": "authenticated_joinquant_etf_master_current_final",
        "universe_exhaustiveness_status": "full_relative_to_authenticated_master_not_cross_source_exhaustive",
        "query_complete_assets": int(query_complete),
        "official_announcement_rows": int(len(announcements)),
        "formal_terminal_events": int(len(formal_events)),
        "final_evidence_state_counts": {str(key): int(value) for key, value in states.items()},
        "primary_candidate_class_counts": {str(key): int(value) for key, value in classes.items()},
        "lifecycle_missing_price_tails": int(lifecycle["last_price_date"].isna().sum()),
        "lifecycle_missing_nav_tails": int(lifecycle["last_nav_date"].isna().sum()),
        "universe_terminal_coverage_complete": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Official title discovery plus current-final lifecycle-tail inventory only. "
            "This layer never reads promoted terminal events; every asset remains evidence_insufficient until the separate settlement builder joins validated events."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    inputs: list[dict[str, Any]] = [
        {"role": "etf_master", "path": _relative(ETF_MASTER_PATH), "sha256": _sha256(ETF_MASTER_PATH)},
        {"role": "etf_master_manifest", "path": _relative(ETF_MASTER_MANIFEST_PATH), "sha256": _sha256(ETF_MASTER_MANIFEST_PATH)},
        *lifecycle_inputs,
        *formal_inputs,
        *query_inputs,
    ]
    outputs = [
        {"role": "lifecycle_inventory", "path": _relative(LIFECYCLE_PATH), "sha256": _sha256(LIFECYCLE_PATH), "rows": int(len(lifecycle))},
        {"role": "official_announcements", "path": _relative(ANNOUNCEMENT_PATH), "sha256": _sha256(ANNOUNCEMENT_PATH), "rows": int(len(announcements))},
        {"role": "coverage_registry", "path": _relative(COVERAGE_PATH), "sha256": _sha256(COVERAGE_PATH), "rows": int(len(coverage))},
        {"role": "review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review))},
        {"role": "discovery_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    dependencies = [
        Path(official.__file__).resolve(),
        Path(sse_api.__file__).resolve(),
        Path(cninfo_api.__file__).resolve(),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(path), "sha256": _sha256(path)} for path in dependencies
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(report)
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--exchange", choices=("all", "SSE", "SZSE"), default="all")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_collection(
        args.as_of,
        assets=args.asset,
        exchange=args.exchange,
        max_assets=args.max_assets,
        sleep_seconds=args.sleep_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    keys = (
        "qualification_status",
        "target_assets",
        "query_complete_assets",
        "official_announcement_rows",
        "final_evidence_state_counts",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
