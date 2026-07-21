"""Collect official CNInfo disclosures for unresolved SZSE ETF share actions."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from pdfminer.pdfparser import PDFSyntaxError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .pit_etf_sse_share_action_announcement_collector import (
    ANNOUNCEMENT_COLUMNS,
    MATCH_COLUMNS,
    REGISTRY_CANDIDATE_COLUMNS,
    _annotate_match_from_document,
    _empty_document_annotation,
    build_registry_candidates,
    classify_action_title,
    match_queue_events,
)


ROOT = Path(__file__).resolve().parents[2]
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_evidence_queue.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_cninfo_etf_share_action_announcements"
QUERY_DIR = RAW_DIR / "queries"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
ANNOUNCEMENT_PATH = OBSERVATION_DIR / "cninfo_etf_share_action_announcements.csv"
MATCH_PATH = OBSERVATION_DIR / "cninfo_etf_share_action_queue_matches.csv"
REGISTRY_CANDIDATE_PATH = OBSERVATION_DIR / "cninfo_etf_share_action_registry_candidates.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "cninfo_etf_share_action_collector_latest.json"
PARSER_DEPENDENCY_PATH = ROOT / "strategy_lab" / "long_hold_v4" / "pit_etf_sse_share_action_announcement_collector.py"

IDENTITY_URL = "https://www.cninfo.com.cn/new/information/topSearch/query"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_BASE_URL = "https://static.cninfo.com.cn/"
KEYWORDS = ("份额拆分", "份额合并", "份额折算")
PAGE_SIZE = 30
EARLIEST_DATE = "1990-01-01"
QUEUE_REQUIRED_COLUMNS = {
    "asset",
    "asset_name",
    "price_effective_date",
    "inferred_factor",
    "observed_price_ratio",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    return session


def load_szse_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    missing = sorted(QUEUE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ETF share-action evidence queue misses columns: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame = frame[frame["asset"].str.startswith("1")].copy()
    frame["price_effective_date"] = pd.to_datetime(frame["price_effective_date"], errors="coerce")
    frame["inferred_factor"] = pd.to_numeric(frame["inferred_factor"], errors="coerce")
    frame["observed_price_ratio"] = pd.to_numeric(frame["observed_price_ratio"], errors="coerce")
    if frame["price_effective_date"].isna().any() or frame["inferred_factor"].isna().any():
        raise ValueError("SZSE ETF share-action queue has invalid event dates or factors")
    if frame.duplicated(["asset", "price_effective_date"], keep=False).any():
        raise ValueError("SZSE ETF share-action queue has duplicate event keys")
    return frame.sort_values(["asset", "price_effective_date"]).reset_index(drop=True)


def select_fund_identity(asset: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    exact = [row for row in rows if str(row.get("code", "")).zfill(6) == asset and row.get("type") == "fund"]
    if len(exact) != 1 or not str(exact[0].get("orgId", "")).strip():
        raise ValueError(f"CNInfo fund identity is ambiguous or missing for {asset}: {len(exact)} matches")
    row = exact[0]
    return {
        "code": asset,
        "org_id": str(row["orgId"]),
        "fund_name": str(row.get("zwjc", "")),
        "category": str(row.get("category", "")),
        "delisted": str(row.get("delisted", "")).lower() == "true",
    }


def _query_paths(asset: str) -> tuple[Path, Path]:
    return QUERY_DIR / f"{asset}.json", QUERY_DIR / f"{asset}.meta.json"


def _valid_query_cache(asset: str, as_of_date: pd.Timestamp) -> bool:
    data_path, meta_path = _query_paths(asset)
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return bool(
            metadata.get("status") == "success"
            and metadata.get("sha256") == _sha256(data_path)
            and artifact.get("asset") == asset
            and artifact.get("as_of_date") == as_of_date.date().isoformat()
            and artifact.get("query_keywords") == list(KEYWORDS)
            and artifact.get("identity", {}).get("org_id")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _query_params(asset: str, org_id: str, keyword: str, as_of_date: pd.Timestamp, page_number: int) -> dict[str, str]:
    return {
        "pageNum": str(page_number),
        "pageSize": str(PAGE_SIZE),
        "column": "fund",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{asset},{org_id}",
        "searchkey": keyword,
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{EARLIEST_DATE}~{as_of_date.date().isoformat()}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def fetch_query_artifact(session: requests.Session, asset: str, as_of_date: pd.Timestamp) -> dict[str, Any]:
    identity_response = session.post(IDENTITY_URL, data={"keyWord": asset, "maxNum": "20"}, timeout=30)
    identity_response.raise_for_status()
    identity_rows = identity_response.json()
    if not isinstance(identity_rows, list):
        raise ValueError(f"CNInfo identity response is invalid for {asset}")
    identity = select_fund_identity(asset, identity_rows)
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            response = session.post(
                QUERY_URL,
                data=_query_params(asset, identity["org_id"], keyword, as_of_date, page_number),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo announcement response is invalid for {asset}/{keyword}")
            rows.extend(announcements)
            page_count = max(1, int(payload.get("totalpages") or 1))
            if page_count > 100:
                raise ValueError(f"CNInfo query has implausible page count for {asset}/{keyword}: {page_count}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "rows": rows})
    return {
        "asset": asset,
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "identity": identity,
        "responses": responses,
    }


def save_query_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    asset = str(artifact["asset"])
    data_path, meta_path = _query_paths(asset)
    _atomic_json(artifact, data_path)
    metadata = {
        "status": "success",
        "asset": asset,
        "path": _relative(data_path),
        "sha256": _sha256(data_path),
        "rows": int(sum(len(item.get("rows", [])) for item in artifact.get("responses", []))),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(metadata, meta_path)
    return metadata


def clean_announcement_title(value: Any) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()


def parse_announcement_time(value: Any) -> pd.Timestamp:
    """Convert CNInfo epoch milliseconds to a timezone-naive China date/time."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return pd.NaT
    timestamp = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).tz_convert("Asia/Shanghai").tz_localize(None)


def parse_announcements(artifacts: list[dict[str, Any]], asset_names: dict[str, str]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        asset = str(artifact["asset"]).zfill(6)
        for response in artifact.get("responses", []):
            for row in response.get("rows", []):
                title = clean_announcement_title(row.get("announcementTitle", ""))
                action_type, role = classify_action_title(title)
                adjunct_url = str(row.get("adjunctUrl", ""))
                announcement_date = parse_announcement_time(row.get("announcementTime"))
                if action_type is None or not adjunct_url or pd.isna(announcement_date):
                    continue
                source_url = urljoin(STATIC_BASE_URL, adjunct_url)
                records[source_url] = {
                    "asset": asset,
                    "asset_name": asset_names.get(asset, ""),
                    "announcement_date": pd.Timestamp(announcement_date).normalize(),
                    "action_type": action_type,
                    "document_role": role,
                    "announcement_title": title,
                    "source_url": source_url,
                    "bulletin_type": str(row.get("announcementTypeName", "")),
                    "original_bulletin_type": str(row.get("announcementType", "")),
                }
    if not records:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS[:-2])
    return pd.DataFrame(records.values()).sort_values(["asset", "announcement_date", "source_url"]).reset_index(drop=True)


def _document_meta_path(source_url: str) -> Path:
    return DOCUMENT_DIR / f"{hashlib.sha256(source_url.encode()).hexdigest()}.meta.json"


def _valid_document_cache(source_url: str) -> dict[str, Any] | None:
    meta_path = _document_meta_path(source_url)
    if not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        pdf_path = ROOT / str(metadata["path"])
        if (
            metadata.get("status") == "success"
            and metadata.get("source_url") == source_url
            and pdf_path.is_file()
            and metadata.get("sha256") == _sha256(pdf_path)
        ):
            return metadata
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def download_document(session: requests.Session, source_url: str) -> dict[str, Any]:
    cached = _valid_document_cache(source_url)
    if cached is not None:
        return {**cached, "cache_hit": True}
    response = session.get(source_url, timeout=60)
    response.raise_for_status()
    payload = response.content
    if not payload.lstrip().startswith(b"%PDF"):
        raise ValueError(f"CNInfo ETF share-action document is not a PDF: {source_url}")
    digest = _sha256_bytes(payload)
    pdf_path = DOCUMENT_DIR / f"{digest}.pdf"
    if not pdf_path.exists():
        _atomic_bytes(payload, pdf_path)
    metadata = {
        "status": "success",
        "source_url": source_url,
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path": _relative(pdf_path),
        "sha256": digest,
        "bytes": len(payload),
    }
    _atomic_json(metadata, _document_meta_path(source_url))
    return {**metadata, "cache_hit": False}


def extract_document_text(metadata: dict[str, Any]) -> dict[str, Any]:
    pdf_path = ROOT / str(metadata["path"])
    pdf_sha256 = str(metadata["sha256"])
    text_path = TEXT_DIR / f"{pdf_sha256}.txt"
    if text_path.is_file():
        return {"status": "success", "path": _relative(text_path), "sha256": _sha256(text_path), "cache_hit": True}
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_path.read_bytes())) as document:
        for page in document.pages:
            pages.append(page.extract_text() or "")
    _atomic_bytes("\n\n".join(pages).encode("utf-8"), text_path)
    return {"status": "success", "path": _relative(text_path), "sha256": _sha256(text_path), "cache_hit": False}


def run_collection(
    as_of: str,
    *,
    assets: list[str] | None = None,
    max_assets: int | None = None,
    sleep_seconds: float = 0.15,
    max_consecutive_failures: int = 5,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    queue = load_szse_queue()
    all_assets = sorted(queue["asset"].unique())
    requested = sorted({str(asset).zfill(6) for asset in assets}) if assets else all_assets
    unknown = sorted(set(requested).difference(all_assets))
    if unknown:
        raise ValueError(f"Requested assets are absent from the SZSE evidence queue: {unknown}")
    queue = queue[queue["asset"].isin(requested)].copy()
    target_assets = requested
    pending = [asset for asset in target_assets if not _valid_query_cache(asset, as_of_date)]
    selected = pending if max_assets is None else pending[: max(0, max_assets)]
    session = _session()
    status: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "target_assets": len(target_assets),
        "target_events": int(len(queue)),
        "cached_before": len(target_assets) - len(pending),
        "selected_assets": len(selected),
        "assets": {},
    }
    failures = 0
    for position, asset in enumerate(selected):
        try:
            status["assets"][asset] = save_query_artifact(fetch_query_artifact(session, asset, as_of_date))
            failures = 0
        except (OSError, ValueError, requests.RequestException) as exc:
            status["assets"][asset] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            failures += 1
        _atomic_json(status, STATUS_PATH)
        if failures >= max_consecutive_failures:
            status["circuit_breaker"] = f"stopped_after_{failures}_consecutive_failures"
            break
        if sleep_seconds > 0 and position + 1 < len(selected):
            time.sleep(sleep_seconds)

    artifacts: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = [
        {"source_id": "etf_share_action_evidence_queue", "path": _relative(QUEUE_PATH), "sha256": _sha256(QUEUE_PATH)}
    ]
    for asset in target_assets:
        if not _valid_query_cache(asset, as_of_date):
            continue
        data_path, meta_path = _query_paths(asset)
        artifacts.append(json.loads(data_path.read_text(encoding="utf-8")))
        inputs.extend(
            [
                {"source_id": f"cninfo_etf_announcement_query:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"source_id": f"cninfo_etf_announcement_query_meta:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )
    asset_names = queue.drop_duplicates("asset").set_index("asset")["asset_name"].astype(str).to_dict()
    announcements = parse_announcements(artifacts, asset_names)
    query_bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    announcements["data_source"] = "cninfo_official_fund_announcements"
    announcements["source_vintage"] = f"cninfo_etf_share_action_query_bundle_sha256:{query_bundle_hash}"
    announcements = announcements.reindex(columns=ANNOUNCEMENT_COLUMNS)
    _atomic_csv(announcements, ANNOUNCEMENT_PATH)

    matches = match_queue_events(queue, announcements)
    document_cache: dict[str, dict[str, Any]] = {}
    final_matches: list[dict[str, Any]] = []
    document_failures: dict[str, str] = {}
    for row in matches.to_dict("records"):
        source_url = str(row.get("source_url", ""))
        if not source_url:
            final_matches.append(_empty_document_annotation(row, "not_applicable"))
            continue
        if source_url not in document_cache:
            try:
                document = download_document(session, source_url)
                text_meta = extract_document_text(document)
                document_cache[source_url] = {"document": document, "text": text_meta}
                token = hashlib.sha256(source_url.encode()).hexdigest()[:16]
                inputs.extend(
                    [
                        {"source_id": f"cninfo_etf_share_action_pdf:{token}", "path": str(document["path"]), "sha256": str(document["sha256"])},
                        {"source_id": f"cninfo_etf_share_action_text:{token}", "path": str(text_meta["path"]), "sha256": str(text_meta["sha256"])},
                    ]
                )
            except (OSError, ValueError, requests.RequestException, PDFSyntaxError) as exc:
                document_failures[source_url] = f"{type(exc).__name__}: {str(exc)[:400]}"
        if source_url in document_cache:
            bundle = document_cache[source_url]
            final_matches.append(_annotate_match_from_document(row, bundle["document"], bundle["text"]))
        else:
            final_matches.append(_empty_document_annotation(row, "failed"))
    match_frame = pd.DataFrame(final_matches).reindex(columns=MATCH_COLUMNS)
    _atomic_csv(match_frame, MATCH_PATH)
    candidates = build_registry_candidates(match_frame)
    candidates["source_type"] = "regulatory_filing"
    candidates = candidates.reindex(columns=REGISTRY_CANDIDATE_COLUMNS)
    _atomic_csv(candidates, REGISTRY_CANDIDATE_PATH)

    cached_assets = sum(_valid_query_cache(asset, as_of_date) for asset in target_assets)
    candidate_count = int(match_frame["match_status"].eq("candidate_found").sum())
    official_factor_count = int(match_frame["best_parsed_factor"].notna().sum())
    exact_factor_count = int(pd.to_numeric(match_frame["factor_relative_error"], errors="coerce").le(0.001).sum())
    correction_count = int(
        match_frame["evidence_review_status"].eq("official_pdf_corrects_heuristic_review_required").sum()
    )
    complete = cached_assets == len(target_assets)
    qualification = (
        "OFFICIAL_DISCOVERY_COMPLETE_REVIEW_REQUIRED"
        if complete and candidate_count == len(queue) and not document_failures
        else "PARTIAL_OFFICIAL_DISCOVERY"
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "outputs": [
            {"path": _relative(ANNOUNCEMENT_PATH), "sha256": _sha256(ANNOUNCEMENT_PATH), "rows": int(len(announcements))},
            {"path": _relative(MATCH_PATH), "sha256": _sha256(MATCH_PATH), "rows": int(len(match_frame))},
            {"path": _relative(REGISTRY_CANDIDATE_PATH), "sha256": _sha256(REGISTRY_CANDIDATE_PATH), "rows": int(len(candidates))},
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {
                "role": "shared_share_action_parser",
                "path": _relative(PARSER_DEPENDENCY_PATH),
                "sha256": _sha256(PARSER_DEPENDENCY_PATH),
            }
        ],
        "target_assets": len(target_assets),
        "target_events": int(len(queue)),
        "query_complete_assets": int(cached_assets),
        "announcement_candidates": int(len(announcements)),
        "queue_events_with_candidate": candidate_count,
        "queue_events_with_official_factor": official_factor_count,
        "queue_events_with_exact_factor_match": exact_factor_count,
        "queue_events_with_heuristic_correction": correction_count,
        "registry_candidate_rows": int(len(candidates)),
        "document_failure_count": len(document_failures),
        "document_failures": document_failures,
        "qualification_status": qualification,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "method_boundary": (
            "CNInfo query responses and matched PDFs are preserved with hashes. Automated parsing remains discovery-only "
            "until independent validation authorizes registry promotion."
        ),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(
        {
            "query_complete_assets": int(cached_assets),
            "announcement_candidates": int(len(announcements)),
            "queue_events_with_candidate": candidate_count,
            "registry_candidate_rows": int(len(candidates)),
            "document_failure_count": len(document_failures),
            "qualification_status": qualification,
        }
    )
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_collection(
        args.as_of,
        assets=args.asset,
        max_assets=args.max_assets,
        sleep_seconds=args.sleep_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    keys = (
        "qualification_status",
        "target_assets",
        "target_events",
        "query_complete_assets",
        "announcement_candidates",
        "queue_events_with_candidate",
        "queue_events_with_exact_factor_match",
        "queue_events_with_heuristic_correction",
        "registry_candidate_rows",
        "document_failure_count",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
