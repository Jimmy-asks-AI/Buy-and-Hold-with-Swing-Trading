"""Build a resumable official announcement catalogue for the full A-share ETF universe.

The catalogue is a discovery layer. Complete title pagination is useful for
routing benchmark evidence, but titles alone cannot prove an initial benchmark,
a benchmark change, or the absence of a change.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from . import pit_etf_cninfo_share_action_announcement_collector as cninfo_api
from . import pit_etf_dividend_announcement_collector as official
from . import pit_source_code_archive as source_code_archive
from . import pit_etf_sse_share_action_announcement_collector as sse_api
from .pit_etf_dividend_universe_coverage_collector import recover_delisted_cninfo_identity


ROOT = Path(__file__).resolve().parents[2]
ETF_MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
ETF_MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
DIVIDEND_QUERY_DIR = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_dividend_announcements" / "queries"
)

RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_official_announcement_catalog"
QUERY_DIR = RAW_DIR / "queries"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
CATALOG_PATH = OBSERVATION_DIR / "etf_official_announcement_catalog.csv"
BENCHMARK_CANDIDATE_PATH = OBSERVATION_DIR / "etf_benchmark_document_candidates.csv"
COVERAGE_PATH = OBSERVATION_DIR / "etf_benchmark_discovery_coverage_registry.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_review_queue.csv"
REPORT_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_benchmark_discovery" / "discovery_report.json"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_official_announcement_catalog_latest.json"

SCHEMA_VERSION = 1
QUERY_CONTRACT_VERSION_BY_EXCHANGE = {"SSE": 1, "SZSE": 2}
CHINA_TIMEZONE = "Asia/Shanghai"
PAGE_COUNT_LIMIT = 200
CNINFO_MIN_PASSES = 2
CNINFO_MAX_PASSES = 5
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05

CATALOG_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "published_at",
    "announcement_title",
    "source_url",
    "source_type",
    "source_category",
    "source_observed_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "data_source",
    "source_vintage",
    "query_path",
    "query_sha256",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

CANDIDATE_COLUMNS = [
    *CATALOG_COLUMNS[:-2],
    "title_tags_json",
    "candidate_roles_json",
    "document_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

COVERAGE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "delist_date",
    "query_complete",
    "official_announcement_count",
    "first_announcement_date",
    "last_announcement_date",
    "baseline_candidate_count",
    "change_candidate_count",
    "contract_review_candidate_count",
    "candidate_document_count",
    "discovery_state",
    "initial_benchmark_evidence_state",
    "change_history_evidence_state",
    "formal_history_rows",
    "benchmark_history_complete",
    "review_priority",
    "review_reason",
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
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(ATOMIC_REPLACE_SLEEP_SECONDS)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def _china_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(CHINA_TIMEZONE)
    return timestamp.tz_convert(CHINA_TIMEZONE)


def _clean_title(value: Any) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()


def _json_list(values: set[str] | list[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def summarize_cninfo_reconciliation(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize provider instability without treating reported totals as authoritative."""

    multi_pass_assets = 0
    extra_pass_assets = 0
    reconciled_assets = 0
    union_increment_rows = 0
    union_rows_above_reported_total = 0
    for artifact in artifacts:
        if str(artifact.get("exchange")) != "SZSE":
            continue
        reconciliation = artifact.get("pagination_reconciliation", {})
        passes = int(reconciliation.get("passes", 0))
        union_rows = int(reconciliation.get("union_rows", 0))
        reported_total = int(reconciliation.get("maximum_observed_total", 0))
        pass_summaries = reconciliation.get("pass_summaries", [])
        if not isinstance(pass_summaries, list):
            pass_summaries = []
        increment = sum(
            max(0, int(summary.get("new_union_rows", 0)))
            for summary in pass_summaries
            if isinstance(summary, dict) and int(summary.get("pass_number", 0)) > 1
        )
        multi_pass_assets += int(passes >= CNINFO_MIN_PASSES)
        extra_pass_assets += int(passes > CNINFO_MIN_PASSES)
        reconciled_assets += int(increment > 0)
        union_increment_rows += increment
        union_rows_above_reported_total += max(0, union_rows - reported_total)
    return {
        "cninfo_multi_pass_assets": multi_pass_assets,
        "cninfo_assets_requiring_more_than_minimum_passes": extra_pass_assets,
        "cninfo_assets_requiring_union_reconciliation": reconciled_assets,
        "cninfo_union_increment_rows": union_increment_rows,
        "cninfo_union_rows_above_maximum_reported_total": union_rows_above_reported_total,
    }


def classify_benchmark_title(title: str) -> tuple[list[str], list[str]]:
    """Classify a title for document routing without making an evidence decision."""

    compact = re.sub(r"\s+", "", str(title))
    tags: set[str] = set()
    roles: set[str] = set()

    if "上市交易公告书" in compact or "上市公告书" in compact:
        tags.add("listing_document")
        roles.add("initial_benchmark_candidate")
    if "基金合同" in compact:
        tags.add("fund_contract")
        if any(marker in compact for marker in ("修订", "修改", "变更", "更新")):
            tags.add("contract_amendment")
            roles.add("contract_content_review_candidate")
        else:
            roles.add("initial_benchmark_candidate")
    if "招募说明书" in compact:
        tags.add("prospectus")
        if "更新" in compact:
            tags.add("prospectus_update")
            roles.add("contract_content_review_candidate")
        else:
            roles.add("initial_benchmark_candidate")
    if "基金产品资料概要" in compact:
        tags.add("product_summary")
        if "更新" in compact:
            roles.add("contract_content_review_candidate")

    benchmark_terms = ("标的指数", "跟踪标的", "业绩比较基准")
    change_terms = ("变更", "更换", "调整", "修改", "修订")
    if any(term in compact for term in benchmark_terms):
        tags.add("benchmark_named_in_title")
    if any(term in compact for term in benchmark_terms) and any(term in compact for term in change_terms):
        tags.add("explicit_benchmark_change")
        roles.add("benchmark_change_candidate")
    if "指数" in compact and "更名" in compact:
        tags.add("index_name_change")
        roles.add("benchmark_change_candidate")
    if "持有人大会" in compact and any(term in compact for term in ("表决结果", "决议生效", "议案")):
        tags.add("holder_resolution")
        roles.add("contract_content_review_candidate")

    return sorted(tags), sorted(roles)


def _query_contract_fingerprint(exchange: str) -> str:
    payload = {
        "contract_version": QUERY_CONTRACT_VERSION_BY_EXCHANGE[exchange],
        "exchange": exchange,
        "search_keyword": "",
        "endpoint": sse_api.QUERY_URL if exchange == "SSE" else cninfo_api.QUERY_URL,
        "sql_id": sse_api.SQL_ID if exchange == "SSE" else "cninfo_full_announcement_query",
        "page_count_limit": PAGE_COUNT_LIMIT,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _query_dependency_hashes() -> dict[str, str]:
    return {
        _relative(Path(module.__file__).resolve()): _sha256(Path(module.__file__).resolve())
        for module in (official, sse_api, cninfo_api)
    }


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
        page_count = int(artifact.get("page_count", 0))
        pages = artifact.get("pages", [])
        total_rows = int(artifact.get("total_rows", -1))
        rows = artifact.get("rows", [])
        requests_made = artifact.get("requests", [])
        expected_contract_version = QUERY_CONTRACT_VERSION_BY_EXCHANGE[exchange]
        if exchange == "SZSE":
            reconciliation = artifact.get("pagination_reconciliation", {})
            pass_summaries = reconciliation.get("pass_summaries", [])
            row_identities = [_cninfo_row_identity(row) for row in rows]
            request_structure_complete = bool(
                reconciliation.get("complete") is True
                and int(reconciliation.get("passes", 0)) >= CNINFO_MIN_PASSES
                and int(reconciliation.get("maximum_observed_total", -1)) <= total_rows
                and int(reconciliation.get("union_rows", -1)) == total_rows
                and isinstance(pass_summaries, list)
                and len(pass_summaries) == int(reconciliation.get("passes", 0))
                and int(pass_summaries[-1].get("new_union_rows", -1)) == 0
                and len(requests_made) == sum(int(item.get("page_count", 0)) for item in pass_summaries)
                and all(row_identities)
                and len(set(row_identities)) == len(row_identities)
            )
        else:
            request_structure_complete = len(requests_made) == page_count
        producer_code_path = ROOT / str(metadata.get("producer_code_path", ""))
        producer_code_sha256 = str(metadata.get("producer_code_sha256", "")).lower()
        if producer_code_path.resolve() != Path(__file__).resolve() or not producer_code_sha256:
            return False
        source_code_archive.authenticate_current_or_archive(producer_code_path, producer_code_sha256)
        return bool(
            metadata.get("status") == "success"
            and metadata.get("sha256") == _sha256(data_path)
            and metadata.get("dependency_hashes") == _query_dependency_hashes()
            and metadata.get("request_contract_fingerprint") == _query_contract_fingerprint(exchange)
            and artifact.get("schema_version") == SCHEMA_VERSION
            and artifact.get("query_contract_version") == expected_contract_version
            and artifact.get("asset") == asset
            and artifact.get("exchange") == exchange
            and artifact.get("as_of_date") == pd.Timestamp(as_of_date).date().isoformat()
            and artifact.get("search_keyword") == ""
            and artifact.get("request_contract_fingerprint") == _query_contract_fingerprint(exchange)
            and 1 <= page_count <= PAGE_COUNT_LIMIT
            and pages == list(range(1, page_count + 1))
            and request_structure_complete
            and isinstance(rows, list)
            and len(rows) == total_rows
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
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
        "rows": int(len(artifact.get("rows", []))),
        "pages": int(artifact.get("page_count", 0)),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(metadata, meta_path)
    return metadata


def load_targets(
    *,
    as_of_date: pd.Timestamp,
    assets: list[str] | None = None,
    exchange: str = "all",
) -> pd.DataFrame:
    manifest = json.loads(ETF_MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not True
        or str(manifest.get("output_path", "")).replace("\\", "/") != _relative(ETF_MASTER_PATH)
        or str(manifest.get("output_sha256", "")) != _sha256(ETF_MASTER_PATH)
    ):
        raise ValueError("ETF master lineage does not authenticate the announcement universe")
    master = pd.read_csv(ETF_MASTER_PATH, dtype={"asset": str})
    required = {"asset", "asset_name", "exchange", "event_type", "list_date", "delist_date"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"ETF master misses announcement-universe fields: {missing}")
    listing = master[master["event_type"].eq("listing")][
        ["asset", "asset_name", "exchange", "list_date"]
    ].copy()
    delisting = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    listing["asset"] = listing["asset"].astype(str).str.zfill(6)
    delisting["asset"] = delisting["asset"].astype(str).str.zfill(6)
    if listing["asset"].duplicated().any() or delisting["asset"].duplicated().any():
        raise ValueError("ETF master contains duplicate lifecycle event assets")
    targets = listing.merge(delisting, on="asset", how="left", validate="one_to_one")
    targets["list_date"] = pd.to_datetime(targets["list_date"], errors="coerce").dt.normalize()
    targets["delist_date"] = pd.to_datetime(targets["delist_date"], errors="coerce").dt.normalize()
    targets = targets[targets["list_date"].notna() & targets["list_date"].le(pd.Timestamp(as_of_date).normalize())]
    if not set(targets["exchange"]).issubset({"SSE", "SZSE"}):
        raise ValueError("ETF master contains unsupported exchanges")
    if exchange != "all":
        targets = targets[targets["exchange"].eq(exchange)]
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        missing_assets = requested.difference(set(targets["asset"]))
        if missing_assets:
            raise ValueError(f"requested assets are not in the ETF master: {sorted(missing_assets)}")
        targets = targets[targets["asset"].isin(requested)]
    if targets.empty:
        raise ValueError("announcement catalogue selected no ETF targets")
    return targets.sort_values(["exchange", "asset"]).reset_index(drop=True)


def _load_cached_cninfo_identity(asset: str) -> dict[str, Any] | None:
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
        and str(identity.get("org_id", "")).strip()
    ):
        return None
    return {
        **identity,
        "resolution_source_path": _relative(data_path),
        "resolution_source_sha256": _sha256(data_path),
    }


def _resolve_cninfo_identity(session: requests.Session, target: Any, as_of_date: pd.Timestamp) -> dict[str, Any]:
    asset = str(target.asset)
    cached = _load_cached_cninfo_identity(asset)
    if cached is not None:
        return cached
    response = session.post(cninfo_api.IDENTITY_URL, data={"keyWord": asset, "maxNum": "20"}, timeout=30)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError(f"CNInfo identity response is invalid for {asset}")
    try:
        identity = cninfo_api.select_fund_identity(asset, rows)
        return {
            **identity,
            "resolution_source_path": "",
            "resolution_source_sha256": "",
        }
    except ValueError:
        end_date = pd.Timestamp(target.delist_date) if pd.notna(target.delist_date) else as_of_date
        return recover_delisted_cninfo_identity(
            session=session,
            asset=asset,
            asset_name=str(target.asset_name),
            list_date=pd.Timestamp(target.list_date),
            delist_date=pd.Timestamp(end_date),
            as_of_date=as_of_date,
        )


def _fetch_sse_catalog(session: requests.Session, target: Any, as_of_date: pd.Timestamp) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    requests_made: list[dict[str, Any]] = []
    page_number = 1
    page_count = 1
    total_rows: int | None = None
    while page_number <= page_count:
        parameters = sse_api._query_params(str(target.asset), "", as_of_date, page_number)
        response = session.get(sse_api.QUERY_URL, params=parameters, timeout=30)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", [])
        if not isinstance(result, list):
            raise ValueError(f"SSE announcement catalogue returned invalid rows for {target.asset}")
        page_help = payload.get("pageHelp") or {}
        observed_page_count = max(1, int(page_help.get("pageCount") or 1))
        observed_total = int(page_help.get("total") or len(result))
        if total_rows is None:
            total_rows = observed_total
            page_count = observed_page_count
        elif total_rows != observed_total or page_count != observed_page_count:
            raise ValueError(f"SSE announcement pagination changed during collection for {target.asset}")
        if page_count > PAGE_COUNT_LIMIT:
            raise ValueError(f"SSE announcement page count is implausible: {target.asset}/{page_count}")
        rows.extend(result)
        requests_made.append(
            {"method": "GET", "url": response.url, "page_number": page_number, "parameters": parameters}
        )
        page_number += 1
    if total_rows is None or len(rows) != total_rows:
        raise ValueError(f"SSE announcement pagination is incomplete for {target.asset}: {len(rows)}/{total_rows}")
    return {
        "schema_version": SCHEMA_VERSION,
        "query_contract_version": QUERY_CONTRACT_VERSION_BY_EXCHANGE["SSE"],
        "asset": str(target.asset),
        "asset_name": str(target.asset_name),
        "exchange": "SSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "search_keyword": "",
        "request_contract_fingerprint": _query_contract_fingerprint("SSE"),
        "sql_id": sse_api.SQL_ID,
        "page_count": page_count,
        "pages": list(range(1, page_count + 1)),
        "total_rows": total_rows,
        "requests": requests_made,
        "rows": rows,
    }


def _cninfo_row_identity(row: dict[str, Any]) -> str:
    announcement_id = str(row.get("announcementId", "")).strip()
    if announcement_id:
        return f"id:{announcement_id}"
    adjunct_url = str(row.get("adjunctUrl", "")).strip()
    if adjunct_url:
        return f"url:{adjunct_url}"
    return ""


def _fetch_cninfo_catalog(session: requests.Session, target: Any, as_of_date: pd.Timestamp) -> dict[str, Any]:
    identity = _resolve_cninfo_identity(session, target, as_of_date)
    union_rows: dict[str, dict[str, Any]] = {}
    requests_made: list[dict[str, Any]] = []
    pass_summaries: list[dict[str, Any]] = []
    maximum_observed_total = 0
    maximum_page_count = 1
    complete = False
    for pass_number in range(1, CNINFO_MAX_PASSES + 1):
        page_number = 1
        pass_page_count = 1
        pass_rows: dict[str, dict[str, Any]] = {}
        observed_totals: list[int] = []
        while page_number <= pass_page_count:
            parameters = cninfo_api._query_params(
                str(target.asset), str(identity["org_id"]), "", as_of_date, page_number
            )
            response = session.post(cninfo_api.QUERY_URL, data=parameters, timeout=30)
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo announcement catalogue returned invalid rows for {target.asset}")
            observed_total = int(payload.get("totalAnnouncement") or len(announcements))
            observed_totals.append(observed_total)
            derived_page_count = max(1, int(math.ceil(observed_total / cninfo_api.PAGE_SIZE)))
            pass_page_count = max(pass_page_count, derived_page_count)
            if pass_page_count > PAGE_COUNT_LIMIT:
                raise ValueError(f"CNInfo announcement page count is implausible: {target.asset}/{pass_page_count}")
            for row in announcements:
                row_identity = _cninfo_row_identity(row)
                if not row_identity:
                    raise ValueError(f"CNInfo announcement row lacks a stable identity for {target.asset}")
                pass_rows[row_identity] = row
            requests_made.append(
                {
                    "method": "POST",
                    "url": cninfo_api.QUERY_URL,
                    "pass_number": pass_number,
                    "page_number": page_number,
                    "parameters": parameters,
                }
            )
            page_number += 1

        before = len(union_rows)
        union_rows.update(pass_rows)
        new_union_rows = len(union_rows) - before
        maximum_observed_total = max(maximum_observed_total, *observed_totals)
        maximum_page_count = max(maximum_page_count, pass_page_count)
        pass_summaries.append(
            {
                "pass_number": pass_number,
                "page_count": pass_page_count,
                "observed_totals": observed_totals,
                "unique_rows": len(pass_rows),
                "new_union_rows": new_union_rows,
                "row_identity_sha256": hashlib.sha256(
                    "|".join(sorted(pass_rows)).encode("utf-8")
                ).hexdigest(),
            }
        )
        if (
            pass_number >= CNINFO_MIN_PASSES
            and new_union_rows == 0
            and len(union_rows) >= maximum_observed_total
        ):
            complete = True
            break
    if not complete:
        raise ValueError(
            f"CNInfo announcement pagination did not stabilize for {target.asset}: "
            f"union={len(union_rows)}/observed={maximum_observed_total}/passes={len(pass_summaries)}"
        )
    rows = sorted(
        union_rows.values(),
        key=lambda row: (int(row.get("announcementTime") or 0), _cninfo_row_identity(row)),
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "query_contract_version": QUERY_CONTRACT_VERSION_BY_EXCHANGE["SZSE"],
        "asset": str(target.asset),
        "asset_name": str(target.asset_name),
        "exchange": "SZSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "search_keyword": "",
        "request_contract_fingerprint": _query_contract_fingerprint("SZSE"),
        "identity": identity,
        "page_count": maximum_page_count,
        "pages": list(range(1, maximum_page_count + 1)),
        "total_rows": len(rows),
        "requests": requests_made,
        "rows": rows,
        "pagination_reconciliation": {
            "complete": complete,
            "passes": len(pass_summaries),
            "minimum_required_passes": CNINFO_MIN_PASSES,
            "maximum_observed_total": maximum_observed_total,
            "union_rows": len(rows),
            "pass_summaries": pass_summaries,
        },
    }


def parse_query_artifacts(artifacts: list[dict[str, Any]]) -> pd.DataFrame:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        asset = str(artifact["asset"]).zfill(6)
        exchange = str(artifact["exchange"])
        cutoff = pd.Timestamp(str(artifact["as_of_date"])).normalize()
        query_path, _ = _query_paths(exchange, asset)
        query_hash = _sha256(query_path)
        for raw in artifact.get("rows", []):
            if exchange == "SSE":
                raw_asset = str(raw.get("SECURITY_CODE", "")).zfill(6)
                title = _clean_title(raw.get("TITLE", ""))
                raw_url = str(raw.get("URL", ""))
                source_url = urljoin("https://www.sse.com.cn", raw_url)
                announcement_date = pd.to_datetime(raw.get("SSEDATE"), errors="coerce")
                published_at: pd.Timestamp | None = None
                source_type = str(raw.get("BULLETIN_TYPE_DESC") or raw.get("BULLETIN_TYPE") or "SSE fund announcement")
                source_category = str(raw.get("ORG_BULLETIN_TYPE_DESC") or "")
                data_source = "SSE official complete fund announcement catalogue"
            else:
                raw_asset = str(raw.get("secCode", "")).zfill(6)
                title = _clean_title(raw.get("announcementTitle", ""))
                raw_url = str(raw.get("adjunctUrl", ""))
                source_url = urljoin(cninfo_api.STATIC_BASE_URL, raw_url)
                announcement_date = cninfo_api.parse_announcement_time(raw.get("announcementTime"))
                published_at = _china_timestamp(announcement_date) if pd.notna(announcement_date) else None
                source_type = str(raw.get("announcementTypeName") or "CNInfo fund announcement")
                source_category = str(raw.get("announcementType") or "")
                data_source = "CNInfo official complete fund announcement catalogue"
            if raw_asset != asset or not title or not raw_url or pd.isna(announcement_date):
                continue
            date = pd.Timestamp(announcement_date).normalize()
            available_trade_date = date + pd.Timedelta(days=1)
            if available_trade_date > cutoff:
                continue
            available_at = _china_timestamp(available_trade_date)
            key = (asset, source_url)
            records[key] = {
                "asset": asset,
                "asset_name": str(artifact.get("asset_name", "")),
                "exchange": exchange,
                "announcement_date": date,
                "published_at": published_at.isoformat() if published_at is not None else "",
                "announcement_title": title,
                "source_url": source_url,
                "source_type": source_type,
                "source_category": source_category,
                "source_observed_at": str(artifact.get("fetched_at", "")),
                "available_at": available_at.isoformat(),
                "available_trade_date": available_trade_date,
                "available_date": available_trade_date,
                "data_source": data_source,
                "source_vintage": f"official_etf_announcement_catalog_sha256:{query_hash}",
                "query_path": _relative(query_path),
                "query_sha256": query_hash,
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
    if not records:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    frame = pd.DataFrame(records.values(), columns=CATALOG_COLUMNS)
    if frame.duplicated(["asset", "source_url"]).any():
        raise ValueError("official ETF announcement catalogue contains duplicate source URLs")
    return frame.sort_values(["asset", "announcement_date", "source_url"]).reset_index(drop=True)


def build_benchmark_candidates(catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in catalog.to_dict("records"):
        tags, roles = classify_benchmark_title(str(record["announcement_title"]))
        if not roles:
            continue
        row = {key: record[key] for key in CATALOG_COLUMNS[:-2]}
        row.update(
            {
                "title_tags_json": _json_list(tags),
                "candidate_roles_json": _json_list(roles),
                "document_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def build_coverage_registry(
    targets: pd.DataFrame,
    catalog: pd.DataFrame,
    candidates: pd.DataFrame,
    query_complete_assets: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        asset = str(target.asset)
        announcements = catalog[catalog["asset"].eq(asset)]
        selected = candidates[candidates["asset"].eq(asset)]
        roles: list[set[str]] = []
        for value in selected.get("candidate_roles_json", pd.Series(dtype=str)):
            try:
                roles.append(set(json.loads(str(value))))
            except (TypeError, ValueError, json.JSONDecodeError):
                roles.append(set())
        baseline = sum("initial_benchmark_candidate" in value for value in roles)
        change = sum("benchmark_change_candidate" in value for value in roles)
        review = sum("contract_content_review_candidate" in value for value in roles)
        complete = asset in query_complete_assets
        if not complete:
            discovery_state = "query_incomplete"
            priority = "P0_QUERY"
            reason = "official_announcement_catalog_query_incomplete"
        elif selected.empty:
            discovery_state = "evidence_insufficient_no_candidate_document"
            priority = "P0_BASELINE"
            reason = "no_baseline_or_change_candidate_identified_in_complete_title_catalog"
        else:
            discovery_state = "candidate_documents_identified"
            priority = "P0_DOCUMENT"
            reason = "download_and_validate_candidate_documents"
        rows.append(
            {
                "asset": asset,
                "asset_name": str(target.asset_name),
                "exchange": str(target.exchange),
                "list_date": pd.Timestamp(target.list_date),
                "delist_date": pd.Timestamp(target.delist_date) if pd.notna(target.delist_date) else pd.NaT,
                "query_complete": complete,
                "official_announcement_count": int(len(announcements)),
                "first_announcement_date": announcements["announcement_date"].min() if not announcements.empty else pd.NaT,
                "last_announcement_date": announcements["announcement_date"].max() if not announcements.empty else pd.NaT,
                "baseline_candidate_count": int(baseline),
                "change_candidate_count": int(change),
                "contract_review_candidate_count": int(review),
                "candidate_document_count": int(len(selected)),
                "discovery_state": discovery_state,
                "initial_benchmark_evidence_state": "evidence_insufficient",
                "change_history_evidence_state": "evidence_insufficient",
                "formal_history_rows": 0,
                "benchmark_history_complete": False,
                "review_priority": priority,
                "review_reason": reason,
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS).sort_values(
        ["review_priority", "asset"]
    ).reset_index(drop=True)


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
    targets = load_targets(as_of_date=cutoff, assets=assets, exchange=exchange)
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
        "selection_mode": "explicit" if assets else "authenticated_master",
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
                _fetch_sse_catalog(sessions["SSE"], target, cutoff)
                if target_exchange == "SSE"
                else _fetch_cninfo_catalog(sessions["SZSE"], target, cutoff)
            )
            status["assets"][asset] = _save_query_artifact(artifact)
            failures = 0
        except (OSError, TypeError, ValueError, requests.RequestException) as exc:
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
                {"role": f"official_announcement_catalog:{target_exchange}:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"role": f"official_announcement_catalog_meta:{target_exchange}:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )

    catalog = parse_query_artifacts(artifacts)
    candidates = build_benchmark_candidates(catalog)
    coverage = build_coverage_registry(targets, catalog, candidates, query_complete_assets)
    review_queue = coverage[coverage["review_priority"].ne("DONE")].copy()
    _atomic_csv(catalog, CATALOG_PATH)
    _atomic_csv(candidates, BENCHMARK_CANDIDATE_PATH)
    _atomic_csv(coverage, COVERAGE_PATH)
    _atomic_csv(review_queue, REVIEW_QUEUE_PATH)

    full_master_selection = assets is None and exchange == "all"
    query_complete = len(query_complete_assets)
    if full_master_selection and query_complete == len(targets):
        qualification = "FULL_AUTHENTICATED_MASTER_TITLE_CATALOG_DOCUMENT_VALIDATION_REQUIRED"
    elif full_master_selection:
        qualification = "PARTIAL_AUTHENTICATED_MASTER_TITLE_CATALOG"
    else:
        qualification = "EXPLICIT_SELECTION_TITLE_CATALOG"
    cninfo_reconciliation = summarize_cninfo_reconciliation(artifacts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": qualification,
        "selection_mode": "authenticated_master" if full_master_selection else "explicit_or_exchange_subset",
        "target_assets": int(len(targets)),
        "query_complete_assets": int(query_complete),
        "official_announcement_rows": int(len(catalog)),
        "benchmark_candidate_document_rows": int(len(candidates)),
        "assets_with_baseline_candidates": int((coverage["baseline_candidate_count"] > 0).sum()),
        "assets_with_change_candidates": int((coverage["change_candidate_count"] > 0).sum()),
        "assets_with_contract_review_candidates": int((coverage["contract_review_candidate_count"] > 0).sum()),
        **cninfo_reconciliation,
        "discovery_state_counts": {
            str(key): int(value) for key, value in coverage["discovery_state"].value_counts().items()
        },
        "formal_history_rows": 0,
        "benchmark_history_complete_assets": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Complete official title pagination relative to the authenticated ETF master. "
            "Every asset remains evidence_insufficient until document content is collected, parsed, and independently validated."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "official_announcement_catalog", "path": _relative(CATALOG_PATH), "sha256": _sha256(CATALOG_PATH), "rows": int(len(catalog))},
        {"role": "benchmark_document_candidates", "path": _relative(BENCHMARK_CANDIDATE_PATH), "sha256": _sha256(BENCHMARK_CANDIDATE_PATH), "rows": int(len(candidates))},
        {"role": "benchmark_discovery_coverage", "path": _relative(COVERAGE_PATH), "sha256": _sha256(COVERAGE_PATH), "rows": int(len(coverage))},
        {"role": "benchmark_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review_queue))},
        {"role": "discovery_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    dependencies = [
        Path(official.__file__).resolve(),
        Path(sse_api.__file__).resolve(),
        Path(cninfo_api.__file__).resolve(),
        Path(source_code_archive.__file__).resolve(),
    ]
    manifest = {
        **report,
        "inputs": [
            {"role": "etf_master", "path": _relative(ETF_MASTER_PATH), "sha256": _sha256(ETF_MASTER_PATH)},
            {"role": "etf_master_manifest", "path": _relative(ETF_MASTER_MANIFEST_PATH), "sha256": _sha256(ETF_MASTER_MANIFEST_PATH)},
            *query_inputs,
        ],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [{"path": _relative(path), "sha256": _sha256(path)} for path in dependencies],
        "current_final_snapshot": True,
        "contains_current_benchmark_backfill": False,
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
        "benchmark_candidate_document_rows",
        "discovery_state_counts",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
