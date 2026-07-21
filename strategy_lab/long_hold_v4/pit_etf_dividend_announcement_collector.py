"""Collect official SSE and CNInfo evidence for discovered ETF cash distributions."""

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

from . import pit_etf_cninfo_share_action_announcement_collector as cninfo_api
from . import pit_etf_sse_share_action_announcement_collector as sse_api


ROOT = Path(__file__).resolve().parents[2]
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_dividend_evidence_queue.csv"
QUEUE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_evidence_queue_latest.json"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_dividend_announcements"
QUERY_DIR = RAW_DIR / "queries"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
ANNOUNCEMENT_PATH = OBSERVATION_DIR / "etf_dividend_official_announcements.csv"
MATCH_PATH = OBSERVATION_DIR / "etf_dividend_official_queue_matches.csv"
CANDIDATE_PATH = OBSERVATION_DIR / "etf_dividend_registry_candidates.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_announcement_collector_latest.json"

KEYWORDS = ("利润分配", "收益分配", "分红")
MAX_ANNOUNCEMENT_LEAD_DAYS = 120
MAX_ANNOUNCEMENT_LAG_DAYS = 7
ANNOUNCEMENT_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "announcement_title",
    "source_url",
    "source_type",
    "matched_keywords_json",
    "data_source",
    "source_vintage",
]
MATCH_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "source_event_date",
    "inferred_cash_per_share",
    "match_status",
    "candidate_document_count",
    "announcement_date",
    "announcement_title",
    "source_url",
    "source_type",
    "pdf_status",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "cash_candidates_json",
    "best_cash_per_share",
    "cash_relative_error",
    "ex_date_candidates_json",
    "best_ex_date",
    "ex_date_distance_days",
    "record_date_candidates_json",
    "best_record_date",
    "pay_date_candidates_json",
    "best_pay_date",
    "review_status",
    "historical_backtest_allowed",
]
CANDIDATE_COLUMNS = [
    "asset",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "source_document_title",
    "source_url",
    "source_type",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "source_event_date",
    "cash_relative_error_to_discovery",
    "ex_date_distance_days",
    "review_status",
    "historical_backtest_allowed",
]


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


def _session(referer: str) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": referer,
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    return session


def load_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    required = {"asset", "asset_name", "exchange", "source_event_date", "inferred_cash_per_share"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ETF dividend evidence queue misses columns: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["source_event_date"] = pd.to_datetime(frame["source_event_date"], errors="coerce").dt.normalize()
    frame["inferred_cash_per_share"] = pd.to_numeric(frame["inferred_cash_per_share"], errors="coerce")
    if frame[["source_event_date", "inferred_cash_per_share"]].isna().any().any():
        raise ValueError("ETF dividend evidence queue has invalid dates or cash values")
    if frame.duplicated(["asset", "source_event_date"]).any():
        raise ValueError("ETF dividend evidence queue has duplicate event keys")
    if not set(frame["exchange"]).issubset({"SSE", "SZSE"}):
        raise ValueError("ETF dividend evidence queue has unsupported exchanges")
    return frame.sort_values(["asset", "source_event_date"]).reset_index(drop=True)


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
        return bool(
            metadata.get("status") == "success"
            and metadata.get("sha256") == _sha256(data_path)
            and artifact.get("asset") == asset
            and artifact.get("exchange") == exchange
            and artifact.get("as_of_date") == as_of_date.date().isoformat()
            and artifact.get("query_keywords") == list(KEYWORDS)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _fetch_sse_query(session: requests.Session, asset: str, as_of_date: pd.Timestamp) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            response = session.get(
                sse_api.QUERY_URL,
                params=sse_api._query_params(asset, keyword, as_of_date, page_number),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result", [])
            if not isinstance(result, list):
                raise ValueError(f"SSE dividend query returned invalid rows for {asset}/{keyword}")
            rows.extend(result)
            page_count = max(1, int(payload.get("pageHelp", {}).get("pageCount") or 1))
            if page_count > 100:
                raise ValueError(f"SSE dividend query has implausible page count for {asset}/{keyword}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "rows": rows})
    return {
        "asset": asset,
        "exchange": "SSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "responses": responses,
    }


def _fetch_cninfo_query(session: requests.Session, asset: str, as_of_date: pd.Timestamp) -> dict[str, Any]:
    identity_response = session.post(cninfo_api.IDENTITY_URL, data={"keyWord": asset, "maxNum": "20"}, timeout=30)
    identity_response.raise_for_status()
    identity = cninfo_api.select_fund_identity(asset, identity_response.json())
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            response = session.post(
                cninfo_api.QUERY_URL,
                data=cninfo_api._query_params(asset, identity["org_id"], keyword, as_of_date, page_number),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo dividend query returned invalid rows for {asset}/{keyword}")
            rows.extend(announcements)
            page_count = max(1, int(payload.get("totalpages") or 1))
            if page_count > 100:
                raise ValueError(f"CNInfo dividend query has implausible page count for {asset}/{keyword}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "rows": rows})
    return {
        "asset": asset,
        "exchange": "SZSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "identity": identity,
        "responses": responses,
    }


def _save_query_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    exchange = str(artifact["exchange"])
    asset = str(artifact["asset"])
    data_path, meta_path = _query_paths(exchange, asset)
    _atomic_json(artifact, data_path)
    metadata = {
        "status": "success",
        "asset": asset,
        "exchange": exchange,
        "path": _relative(data_path),
        "sha256": _sha256(data_path),
        "rows": int(sum(len(item.get("rows", [])) for item in artifact.get("responses", []))),
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(metadata, meta_path)
    return metadata


def is_distribution_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", str(title))))
    return "公告" in compact and any(token in compact for token in ("利润分配", "收益分配", "分红"))


def parse_query_artifacts(artifacts: list[dict[str, Any]], asset_names: dict[str, str]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        asset = str(artifact["asset"]).zfill(6)
        exchange = str(artifact["exchange"])
        for response in artifact.get("responses", []):
            keyword = str(response.get("keyword", ""))
            for row in response.get("rows", []):
                if exchange == "SSE":
                    title = str(row.get("TITLE", ""))
                    date = pd.to_datetime(row.get("SSEDATE"), errors="coerce")
                    relative_url = str(row.get("URL", ""))
                    source_url = urljoin("https://www.sse.com.cn/", relative_url)
                    source_type = "exchange_announcement"
                else:
                    title = cninfo_api.clean_announcement_title(row.get("announcementTitle", ""))
                    date = cninfo_api.parse_announcement_time(row.get("announcementTime"))
                    source_url = urljoin(cninfo_api.STATIC_BASE_URL, str(row.get("adjunctUrl", "")))
                    source_type = "regulatory_filing"
                if not is_distribution_title(title) or pd.isna(date) or not source_url:
                    continue
                existing = records.get(source_url)
                if existing is None:
                    records[source_url] = {
                        "asset": asset,
                        "asset_name": asset_names.get(asset, ""),
                        "exchange": exchange,
                        "announcement_date": pd.Timestamp(date).normalize(),
                        "announcement_title": title,
                        "source_url": source_url,
                        "source_type": source_type,
                        "matched_keywords": {keyword},
                    }
                else:
                    existing["matched_keywords"].add(keyword)
    if not records:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS[:-2])
    rows = []
    for record in records.values():
        record["matched_keywords_json"] = json.dumps(sorted(record.pop("matched_keywords")), ensure_ascii=False)
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["asset", "announcement_date", "source_url"]).reset_index(drop=True)


def build_event_document_edges(queue: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    columns = ["asset", "source_event_date", "source_url"]
    if announcements.empty:
        return pd.DataFrame(columns=columns)
    merged = queue[["asset", "source_event_date"]].merge(
        announcements[["asset", "announcement_date", "source_url"]],
        on="asset",
        how="left",
    )
    distance = (merged["source_event_date"] - merged["announcement_date"]).dt.days
    mask = merged["source_url"].notna() & distance.between(-MAX_ANNOUNCEMENT_LAG_DAYS, MAX_ANNOUNCEMENT_LEAD_DAYS)
    return merged.loc[mask, columns].drop_duplicates().reset_index(drop=True)


def _document_meta_path(source_url: str) -> Path:
    return DOCUMENT_DIR / f"{hashlib.sha256(source_url.encode()).hexdigest()}.meta.json"


def _valid_document_cache(source_url: str) -> dict[str, Any] | None:
    meta_path = _document_meta_path(source_url)
    if not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        path = ROOT / str(metadata["path"])
        if metadata.get("status") == "success" and metadata.get("source_url") == source_url and path.is_file() and metadata.get("sha256") == _sha256(path):
            return metadata
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def download_document(session: requests.Session, source_url: str) -> dict[str, Any]:
    cached = _valid_document_cache(source_url)
    if cached:
        return cached
    response = session.get(source_url, timeout=45)
    response.raise_for_status()
    payload = response.content
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"official ETF dividend document is not a PDF: {source_url}")
    digest = _sha256_bytes(payload)
    path = DOCUMENT_DIR / f"{digest}.pdf"
    if not path.exists():
        _atomic_bytes(payload, path)
    metadata = {
        "status": "success",
        "source_url": source_url,
        "path": _relative(path),
        "sha256": digest,
        "bytes": len(payload),
        "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(metadata, _document_meta_path(source_url))
    return metadata


def extract_document_text(document: dict[str, Any]) -> dict[str, Any]:
    pdf_path = ROOT / str(document["path"])
    text_path = TEXT_DIR / f"{document['sha256']}.txt"
    if text_path.is_file():
        return {"path": _relative(text_path), "sha256": _sha256(text_path)}
    with pdfplumber.open(io.BytesIO(pdf_path.read_bytes())) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if not text.strip():
        raise ValueError(f"official ETF dividend PDF has no extractable text: {document['source_url']}")
    _atomic_bytes(text.encode("utf-8"), text_path)
    return {"path": _relative(text_path), "sha256": _sha256(text_path)}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).replace(",", "").replace("：", ":").replace("（", "(").replace("）", ")")


def _cash_record(denominator: float, amount: float, method: str, match: re.Match[str]) -> dict[str, Any] | None:
    if denominator <= 0 or amount < 0:
        return None
    value = amount / denominator
    if not (0 < value < 1000):
        return None
    start = max(0, match.start() - 50)
    end = min(len(match.string), match.end() + 50)
    return {
        "cash_per_share": float(value),
        "denominator_shares": float(denominator),
        "stated_cash_amount": float(amount),
        "method": method,
        "snippet": match.string[start:end][:240],
    }


def _table_line_cash_candidates(text: str) -> list[dict[str, Any]]:
    lines = str(text).splitlines()
    records: list[dict[str, Any]] = []
    for position, line in enumerate(lines):
        compact_line = re.sub(r"\s+", "", line)
        if not any(token in compact_line for token in ("本次分红方案", "本次基金分红方案", "本次收益分配方案", "本次利润分配方案")):
            continue
        window_lines = lines[position : position + 5]
        window = "\n".join(window_lines)
        denominator_match = re.search(r"元\s*/\s*(\d+(?:\.\d+)?)", window)
        if denominator_match is None:
            denominator_match = re.search(r"/\s*(\d+(?:\.\d+)?)\s*份", window)
        if not denominator_match:
            continue
        denominator = float(denominator_match.group(1))
        amount: float | None = None
        denominator_span = denominator_match.span(1)
        for number_match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", window):
            overlaps_denominator = number_match.start() < denominator_span[1] and number_match.end() > denominator_span[0]
            value = float(number_match.group(1))
            if not overlaps_denominator and 0 < value < 1000:
                amount = value
                break
        if amount is None or denominator <= 0:
            continue
        value = amount / denominator
        if not (0 < value < 1000):
            continue
        snippet = "|".join(window_lines)[:240]
        records.append(
            {
                "cash_per_share": float(value),
                "denominator_shares": denominator,
                "stated_cash_amount": amount,
                "method": "table_line_anchored_distribution",
                "snippet": snippet,
            }
        )
    return records


def extract_cash_candidates(text: str) -> list[dict[str, Any]]:
    compact = _compact(text)
    records: list[dict[str, Any]] = _table_line_cash_candidates(text)
    for match in re.finditer(
        r"本次(?:基金)?(?:分红|收益分配|利润分配)方案[^\n]{0,120}?元\s*/\s*(\d+(?:\.\d+)?)\s*\n\s*(\d+(?:\.\d+)?)\s*\n?\s*份基金份额",
        str(text),
    ):
        record = _cash_record(
            float(match.group(1)),
            float(match.group(2)),
            "multiline_table_distribution_per_n_shares",
            match,
        )
        if record:
            records.append(record)
    for match in re.finditer(
        r"本次(?:基金)?(?:分红|收益分配|利润分配)方案[^\n]{0,120}?元\s*/\s*(\d+(?:\.\d+)?)\s*份\s*\n\s*(\d+(?:\.\d+)?)\s*\n?\s*基金份额",
        str(text),
    ):
        record = _cash_record(
            float(match.group(1)),
            float(match.group(2)),
            "multiline_table_amount_after_share_unit",
            match,
        )
        if record:
            records.append(record)
    patterns = [
        (
            r"每(\d+(?:\.\d+)?)份[^。；;]{0,100}?(?:派发|派送|发放|分配)(?:现金收益|现金红利|红利|现金)?(?:人民币)?(\d+(?:\.\d+)?)元",
            "per_n_shares_distribution",
        ),
        (
            r"本次(?:基金)?(?:分红|收益分配|利润分配)方案[^。；;]{0,100}?元/(\d+(?:\.\d+)?)份基金份额\)?[:：]?(\d+(?:\.\d+)?)",
            "table_distribution_per_n_shares",
        ),
        (
            r"(?:每1份|每份)基金份额[^。；;]{0,80}?(?:现金红利|现金收益|派发红利|分配)(?:为|人民币)?(\d+(?:\.\d+)?)元",
            "per_share_distribution",
        ),
    ]
    for pattern, method in patterns:
        for match in re.finditer(pattern, compact):
            if method == "per_share_distribution":
                denominator, amount = 1.0, float(match.group(1))
            else:
                denominator, amount = float(match.group(1)), float(match.group(2))
            record = _cash_record(denominator, amount, method, match)
            if record:
                records.append(record)
    unique: dict[tuple[float, str], dict[str, Any]] = {}
    for record in records:
        unique[(round(float(record["cash_per_share"]), 12), str(record["method"]))] = record
    return list(unique.values())


def _date_records(text: str, patterns: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
    compact = _compact(text)
    date_token = r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日"
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for prefix, method, rank in patterns:
        for match in re.finditer(prefix.format(date=date_token), compact):
            year, month, day = map(int, match.groups()[-3:])
            try:
                value = pd.Timestamp(year=year, month=month, day=day).date().isoformat()
            except ValueError:
                continue
            start = max(0, match.start() - 45)
            end = min(len(compact), match.end() + 45)
            records[(value, method)] = {
                "date": value,
                "method": method,
                "rank": rank,
                "snippet": compact[start:end][:220],
            }
    return sorted(records.values(), key=lambda item: (int(item["rank"]), str(item["date"]), str(item["method"])))


def extract_ex_date_candidates(text: str) -> list[dict[str, Any]]:
    return _date_records(
        text,
        [
            (r"(?:场内)?除息日(?:\(场内\))?[:：]?{date}", "explicit_exchange_ex_date", 0),
            (r"权益登记日(?:、|和)除息日(?:和发放日)?[:：]?{date}", "combined_record_and_ex_date", 1),
            (r"权益登记日、除息日[:：]?{date}", "combined_record_ex_date", 1),
        ],
    )


def extract_record_date_candidates(text: str) -> list[dict[str, Any]]:
    return _date_records(
        text,
        [
            (r"(?:场内份额)?权益登记日(?:\(场内\))?[:：]?{date}", "explicit_exchange_record_date", 0),
            (r"权益登记日(?:、|和)除息日(?:和发放日)?[:：]?{date}", "combined_record_and_ex_date", 1),
        ],
    )


def extract_pay_date_candidates(text: str) -> list[dict[str, Any]]:
    return _date_records(
        text,
        [
            (r"(?:场内)?现金红利发放日(?:\(场内\))?[:：]?{date}", "explicit_exchange_cash_pay_date", 0),
            (r"(?:场内)?红利发放日(?:\(场内\))?[:：]?{date}", "explicit_exchange_pay_date", 0),
            (r"收益发放日[:：]?{date}", "explicit_income_pay_date", 1),
        ],
    )


def parse_document_text(text: str) -> dict[str, Any]:
    return {
        "cash_candidates": extract_cash_candidates(text),
        "ex_date_candidates": extract_ex_date_candidates(text),
        "record_date_candidates": extract_record_date_candidates(text),
        "pay_date_candidates": extract_pay_date_candidates(text),
    }


def _select_date(candidates: list[dict[str, Any]], target: pd.Timestamp, *, on_or_after: bool | None = None) -> Any:
    eligible = []
    for candidate in candidates:
        date = pd.Timestamp(candidate["date"]).normalize()
        if on_or_after is True and date < target:
            continue
        if on_or_after is False and date > target:
            continue
        eligible.append((int(candidate["rank"]), abs((date - target).days), date))
    return pd.NaT if not eligible else min(eligible)[2]


def match_events(
    queue: pd.DataFrame,
    announcements: pd.DataFrame,
    document_parses: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    announcement_lookup = {str(row.source_url): row for row in announcements.itertuples(index=False)}
    edges = build_event_document_edges(queue, announcements)
    urls_by_key: dict[tuple[str, pd.Timestamp], list[str]] = {}
    for row in edges.itertuples(index=False):
        key = (str(row.asset), pd.Timestamp(row.source_event_date).normalize())
        urls_by_key.setdefault(key, []).append(str(row.source_url))
    rows: list[dict[str, Any]] = []
    for event in queue.itertuples(index=False):
        asset = str(event.asset)
        source_event_date = pd.Timestamp(event.source_event_date).normalize()
        inferred_cash = float(event.inferred_cash_per_share)
        urls = sorted(set(urls_by_key.get((asset, source_event_date), [])))
        options: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for source_url in urls:
            bundle = document_parses.get(source_url)
            if not bundle or bundle.get("pdf_status") != "success":
                continue
            parsed = bundle["parsed"]
            announcement = announcement_lookup[source_url]
            for cash in parsed["cash_candidates"]:
                cash_value = float(cash["cash_per_share"])
                cash_error = abs(cash_value / inferred_cash - 1.0)
                for ex_candidate in parsed["ex_date_candidates"]:
                    ex_date = pd.Timestamp(ex_candidate["date"]).normalize()
                    ex_distance = abs((ex_date - source_event_date).days)
                    chronology_bad = pd.Timestamp(announcement.announcement_date).normalize() > ex_date
                    score = (
                        1 if chronology_bad else 0,
                        1 if ex_distance > 7 else 0,
                        1 if cash_error > 0.02 else 0,
                        ex_distance,
                        cash_error,
                        int(ex_candidate["rank"]),
                        pd.Timestamp(announcement.announcement_date),
                    )
                    options.append(
                        (
                            score,
                            {
                                "announcement": announcement,
                                "bundle": bundle,
                                "cash_value": cash_value,
                                "cash_error": cash_error,
                                "ex_date": ex_date,
                                "ex_distance": ex_distance,
                            },
                        )
                    )
        base = {
            "asset": asset,
            "asset_name": str(event.asset_name),
            "exchange": str(event.exchange),
            "source_event_date": source_event_date,
            "inferred_cash_per_share": inferred_cash,
            "match_status": "no_parseable_official_candidate",
            "candidate_document_count": len(urls),
            "announcement_date": pd.NaT,
            "announcement_title": "",
            "source_url": "",
            "source_type": "",
            "pdf_status": "not_applicable",
            "pdf_path": "",
            "pdf_sha256": "",
            "text_path": "",
            "text_sha256": "",
            "cash_candidates_json": "[]",
            "best_cash_per_share": pd.NA,
            "cash_relative_error": pd.NA,
            "ex_date_candidates_json": "[]",
            "best_ex_date": pd.NaT,
            "ex_date_distance_days": pd.NA,
            "record_date_candidates_json": "[]",
            "best_record_date": pd.NaT,
            "pay_date_candidates_json": "[]",
            "best_pay_date": pd.NaT,
            "review_status": "official_pdf_parse_inconclusive",
            "historical_backtest_allowed": False,
        }
        if options:
            _, selected = min(options, key=lambda item: item[0])
            announcement = selected["announcement"]
            bundle = selected["bundle"]
            parsed = bundle["parsed"]
            ex_date = selected["ex_date"]
            record_date = _select_date(parsed["record_date_candidates"], ex_date, on_or_after=False)
            pay_date = _select_date(parsed["pay_date_candidates"], ex_date, on_or_after=True)
            chronology_ok = pd.Timestamp(announcement.announcement_date).normalize() <= ex_date
            complete_dates = pd.notna(record_date) and pd.notna(pay_date) and record_date <= ex_date <= pay_date
            exact = selected["cash_error"] <= 0.001 and selected["ex_distance"] <= 3
            close = selected["cash_error"] <= 0.02 and selected["ex_distance"] <= 7
            if exact and chronology_ok and complete_dates:
                review_status = "official_pdf_cash_and_dates_found_review_required"
            elif close and chronology_ok and complete_dates:
                review_status = "official_pdf_corrects_discovery_review_required"
            else:
                review_status = "official_pdf_parse_inconclusive"
            base.update(
                {
                    "match_status": "candidate_found",
                    "announcement_date": pd.Timestamp(announcement.announcement_date).normalize(),
                    "announcement_title": str(announcement.announcement_title),
                    "source_url": str(announcement.source_url),
                    "source_type": str(announcement.source_type),
                    "pdf_status": "success",
                    "pdf_path": bundle["document"]["path"],
                    "pdf_sha256": bundle["document"]["sha256"],
                    "text_path": bundle["text"]["path"],
                    "text_sha256": bundle["text"]["sha256"],
                    "cash_candidates_json": json.dumps(parsed["cash_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "best_cash_per_share": selected["cash_value"],
                    "cash_relative_error": selected["cash_error"],
                    "ex_date_candidates_json": json.dumps(parsed["ex_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "best_ex_date": ex_date,
                    "ex_date_distance_days": selected["ex_distance"],
                    "record_date_candidates_json": json.dumps(parsed["record_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "best_record_date": record_date,
                    "pay_date_candidates_json": json.dumps(parsed["pay_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "best_pay_date": pay_date,
                    "review_status": review_status,
                }
            )
        rows.append(base)
    return pd.DataFrame(rows).reindex(columns=MATCH_COLUMNS).sort_values(["asset", "source_event_date"]).reset_index(drop=True)


def build_registry_candidates(matches: pd.DataFrame) -> pd.DataFrame:
    allowed = {
        "official_pdf_cash_and_dates_found_review_required",
        "official_pdf_corrects_discovery_review_required",
    }
    rows = []
    for match in matches[matches["review_status"].isin(allowed)].itertuples(index=False):
        rows.append(
            {
                "asset": str(match.asset).zfill(6),
                "announcement_date": pd.Timestamp(match.announcement_date).normalize(),
                "record_date": pd.Timestamp(match.best_record_date).normalize(),
                "ex_date": pd.Timestamp(match.best_ex_date).normalize(),
                "pay_date": pd.Timestamp(match.best_pay_date).normalize(),
                "cash_per_share": float(match.best_cash_per_share),
                "source_document_title": str(match.announcement_title),
                "source_url": str(match.source_url),
                "source_type": str(match.source_type),
                "pdf_path": str(match.pdf_path),
                "pdf_sha256": str(match.pdf_sha256),
                "text_path": str(match.text_path),
                "text_sha256": str(match.text_sha256),
                "source_event_date": pd.Timestamp(match.source_event_date).normalize(),
                "cash_relative_error_to_discovery": float(match.cash_relative_error),
                "ex_date_distance_days": int(match.ex_date_distance_days),
                "review_status": str(match.review_status),
                "historical_backtest_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).sort_values(["asset", "ex_date"]).reset_index(drop=True)


def run_collection(
    as_of: str,
    *,
    assets: list[str] | None = None,
    exchange: str = "all",
    max_assets: int | None = None,
    sleep_seconds: float = 0.1,
    max_consecutive_failures: int = 5,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    queue_manifest = json.loads(QUEUE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if queue_manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("ETF dividend evidence queue unexpectedly permits historical use")
    queue = load_queue()
    if exchange != "all":
        queue = queue[queue["exchange"].eq(exchange)].copy()
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        queue = queue[queue["asset"].isin(requested)].copy()
    if queue.empty:
        raise ValueError("ETF dividend announcement collection selected no queue events")
    targets = queue[["asset", "exchange"]].drop_duplicates().sort_values(["exchange", "asset"])
    pending = [
        (row.exchange, row.asset)
        for row in targets.itertuples(index=False)
        if not _valid_query_cache(str(row.exchange), str(row.asset), as_of_date)
    ]
    selected = pending[:max_assets] if max_assets is not None else pending
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "target_assets": int(len(targets)),
        "target_events": int(len(queue)),
        "selected_uncached_assets": len(selected),
        "assets": {},
    }
    sessions = {
        "SSE": _session("https://www.sse.com.cn/disclosure/fund/announcement/index.shtml"),
        "SZSE": _session("https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"),
    }
    failures = 0
    for position, (target_exchange, asset) in enumerate(selected):
        try:
            artifact = (
                _fetch_sse_query(sessions["SSE"], asset, as_of_date)
                if target_exchange == "SSE"
                else _fetch_cninfo_query(sessions["SZSE"], asset, as_of_date)
            )
            status["assets"][asset] = _save_query_artifact(artifact)
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

    artifacts = []
    inputs: list[dict[str, str]] = [
        {"source_id": "etf_dividend_evidence_queue", "path": _relative(QUEUE_PATH), "sha256": _sha256(QUEUE_PATH)},
        {"source_id": "etf_dividend_evidence_queue_manifest", "path": _relative(QUEUE_MANIFEST_PATH), "sha256": _sha256(QUEUE_MANIFEST_PATH)},
    ]
    for row in targets.itertuples(index=False):
        target_exchange, asset = str(row.exchange), str(row.asset)
        if not _valid_query_cache(target_exchange, asset, as_of_date):
            continue
        data_path, meta_path = _query_paths(target_exchange, asset)
        artifacts.append(json.loads(data_path.read_text(encoding="utf-8")))
        inputs.extend(
            [
                {"source_id": f"etf_dividend_query:{target_exchange}:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"source_id": f"etf_dividend_query_meta:{target_exchange}:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )
    asset_names = queue.drop_duplicates("asset").set_index("asset")["asset_name"].astype(str).to_dict()
    announcements = parse_query_artifacts(artifacts, asset_names)
    query_bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    if not announcements.empty:
        announcements["data_source"] = "SSE and CNInfo official ETF distribution announcements"
        announcements["source_vintage"] = f"official_etf_dividend_query_bundle_sha256:{query_bundle_hash}"
    announcements = announcements.reindex(columns=ANNOUNCEMENT_COLUMNS)

    edges = build_event_document_edges(queue, announcements)
    needed_urls = sorted(edges["source_url"].dropna().unique()) if not edges.empty else []
    document_parses: dict[str, dict[str, Any]] = {}
    document_failures: dict[str, str] = {}
    for source_url in needed_urls:
        try:
            session = sessions["SSE"] if str(source_url).startswith("https://www.sse.com.cn/") else sessions["SZSE"]
            document = download_document(session, str(source_url))
            text_meta = extract_document_text(document)
            text = (ROOT / str(text_meta["path"])).read_text(encoding="utf-8")
            document_parses[str(source_url)] = {
                "pdf_status": "success",
                "document": document,
                "text": text_meta,
                "parsed": parse_document_text(text),
            }
            token = hashlib.sha256(str(source_url).encode()).hexdigest()[:16]
            inputs.extend(
                [
                    {"source_id": f"etf_dividend_pdf:{token}", "path": str(document["path"]), "sha256": str(document["sha256"])},
                    {"source_id": f"etf_dividend_text:{token}", "path": str(text_meta["path"]), "sha256": str(text_meta["sha256"])},
                ]
            )
        except (OSError, ValueError, requests.RequestException, PDFSyntaxError) as exc:
            document_failures[str(source_url)] = f"{type(exc).__name__}: {str(exc)[:400]}"

    matches = match_events(queue, announcements, document_parses)
    candidates = build_registry_candidates(matches)
    _atomic_csv(announcements, ANNOUNCEMENT_PATH)
    _atomic_csv(matches, MATCH_PATH)
    _atomic_csv(candidates, CANDIDATE_PATH)

    complete_queries = sum(
        _valid_query_cache(str(row.exchange), str(row.asset), as_of_date)
        for row in targets.itertuples(index=False)
    )
    reviewed = int(matches["review_status"].isin({
        "official_pdf_cash_and_dates_found_review_required",
        "official_pdf_corrects_discovery_review_required",
    }).sum())
    qualification = "OFFICIAL_DISCOVERY_COMPLETE_REVIEW_REQUIRED" if complete_queries == len(targets) else "PARTIAL_OFFICIAL_DISCOVERY"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "outputs": [
            {"path": _relative(ANNOUNCEMENT_PATH), "sha256": _sha256(ANNOUNCEMENT_PATH), "rows": int(len(announcements))},
            {"path": _relative(MATCH_PATH), "sha256": _sha256(MATCH_PATH), "rows": int(len(matches))},
            {"path": _relative(CANDIDATE_PATH), "sha256": _sha256(CANDIDATE_PATH), "rows": int(len(candidates))},
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"role": "sse_query_contract", "path": _relative(Path(sse_api.__file__).resolve()), "sha256": _sha256(Path(sse_api.__file__).resolve())},
            {"role": "cninfo_query_contract", "path": _relative(Path(cninfo_api.__file__).resolve()), "sha256": _sha256(Path(cninfo_api.__file__).resolve())},
        ],
        "target_assets": int(len(targets)),
        "target_events": int(len(queue)),
        "query_complete_assets": int(complete_queries),
        "announcement_candidates": int(len(announcements)),
        "candidate_documents": int(len(needed_urls)),
        "document_failure_count": int(len(document_failures)),
        "document_failures": document_failures,
        "queue_events_with_parseable_official_record": reviewed,
        "registry_candidate_rows": int(len(candidates)),
        "qualification_status": qualification,
        "current_final_snapshot": True,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "method_boundary": (
            "Official PDFs are preserved and parsed, but automated matching remains discovery-only until an independent "
            "validator authenticates every row and authorizes a PIT dividend table."
        ),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(
        {
            "query_complete_assets": int(complete_queries),
            "announcement_candidates": int(len(announcements)),
            "candidate_documents": int(len(needed_urls)),
            "document_failure_count": int(len(document_failures)),
            "registry_candidate_rows": int(len(candidates)),
            "qualification_status": qualification,
        }
    )
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
    manifest = run_collection(
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
        "target_events",
        "query_complete_assets",
        "announcement_candidates",
        "candidate_documents",
        "document_failure_count",
        "queue_events_with_parseable_official_record",
        "registry_candidate_rows",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
