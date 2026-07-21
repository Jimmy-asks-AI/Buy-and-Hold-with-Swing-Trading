"""Collect official SSE ETF share-action announcements for unresolved price jumps.

The collector discovers and preserves primary-source evidence. It never promotes
parsed ratios into the governed corporate-action registry automatically.
"""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[2]
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_evidence_queue.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_sse_etf_share_action_announcements"
QUERY_DIR = RAW_DIR / "queries"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
ANNOUNCEMENT_PATH = OBSERVATION_DIR / "sse_etf_share_action_announcements.csv"
MATCH_PATH = OBSERVATION_DIR / "sse_etf_share_action_queue_matches.csv"
REGISTRY_CANDIDATE_PATH = OBSERVATION_DIR / "sse_etf_share_action_registry_candidates.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_etf_share_action_collector_latest.json"

QUERY_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_BASE_URL = "https://www.sse.com.cn"
SQL_ID = "COMMON_PL_JJXX_JJGG_NEW_L"
KEYWORDS = ("份额拆分", "份额合并", "份额折算")
PAGE_SIZE = 200
EARLIEST_DATE = "1990-01-01"
QUEUE_REQUIRED_COLUMNS = {
    "asset",
    "asset_name",
    "price_effective_date",
    "inferred_factor",
    "observed_price_ratio",
    "review_status",
}
ANNOUNCEMENT_COLUMNS = [
    "asset",
    "asset_name",
    "announcement_date",
    "action_type",
    "document_role",
    "announcement_title",
    "source_url",
    "bulletin_type",
    "original_bulletin_type",
    "data_source",
    "source_vintage",
]
MATCH_COLUMNS = [
    "asset",
    "asset_name",
    "price_effective_date",
    "inferred_factor",
    "observed_price_ratio",
    "match_status",
    "candidate_action_type",
    "candidate_document_role",
    "candidate_announcement_date",
    "announcement_distance_days",
    "candidate_title",
    "source_url",
    "pdf_status",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "parsed_factor_candidates_json",
    "best_parsed_factor",
    "factor_relative_error",
    "parsed_dates_json",
    "nearest_document_date_distance_days",
    "parsed_action_date_candidates_json",
    "best_action_event_date",
    "action_event_date_distance_days",
    "evidence_review_status",
    "historical_backtest_allowed",
]
REGISTRY_CANDIDATE_COLUMNS = [
    "asset",
    "action_type",
    "event_date",
    "price_effective_date",
    "shares_after_per_share_before",
    "announcement_date",
    "source_document_title",
    "source_url",
    "source_type",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "factor_relative_error_to_inference",
    "factor_relative_error_to_observed_price_ratio",
    "normalized_price_ratio_residual",
    "action_event_date_distance_days",
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


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.sse.com.cn/disclosure/fund/announcement/index.shtml",
        }
    )
    return session


def load_sse_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    missing = sorted(QUEUE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ETF share-action evidence queue misses columns: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame = frame[frame["asset"].str.startswith("5")].copy()
    frame["price_effective_date"] = pd.to_datetime(frame["price_effective_date"], errors="coerce")
    frame["inferred_factor"] = pd.to_numeric(frame["inferred_factor"], errors="coerce")
    frame["observed_price_ratio"] = pd.to_numeric(frame["observed_price_ratio"], errors="coerce")
    if frame["price_effective_date"].isna().any() or frame["inferred_factor"].isna().any():
        raise ValueError("SSE ETF share-action queue has invalid event dates or factors")
    duplicate = frame.duplicated(["asset", "price_effective_date"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["asset", "price_effective_date"]].astype(str).to_dict("records")
        raise ValueError(f"SSE ETF share-action queue has duplicate event keys: {keys[:5]}")
    return frame.sort_values(["asset", "price_effective_date"]).reset_index(drop=True)


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
            and artifact.get("sql_id") == SQL_ID
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _query_params(asset: str, keyword: str, as_of_date: pd.Timestamp, page_number: int) -> dict[str, str]:
    return {
        "isPagination": "true",
        "pageHelp.pageSize": str(PAGE_SIZE),
        "pageHelp.pageNo": str(page_number),
        "pageHelp.beginPage": str(page_number),
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": str(page_number),
        "type": "inParams",
        "sqlId": SQL_ID,
        "TITLE": keyword,
        "SECURITY_CODE": asset,
        "BULLETIN_TYPE": "",
        "START_DATE": EARLIEST_DATE,
        "END_DATE": as_of_date.date().isoformat(),
        "DATE_DESC": "1",
        "DATE_ASC": "",
        "CODE_DESC": "",
        "CODE_ASC": "",
    }


def fetch_query_artifact(
    session: requests.Session,
    asset: str,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        rows: list[dict[str, Any]] = []
        request_urls: list[str] = []
        page_count = 1
        page_number = 1
        while page_number <= page_count:
            response = session.get(QUERY_URL, params=_query_params(asset, keyword, as_of_date, page_number), timeout=30)
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result", [])
            if not isinstance(result, list):
                raise ValueError(f"SSE ETF announcement query returned invalid rows for {asset}/{keyword}")
            rows.extend(result)
            request_urls.append(response.url)
            page_help = payload.get("pageHelp", {})
            page_count = max(1, int(page_help.get("pageCount") or 1))
            if page_count > 100:
                raise ValueError(f"SSE ETF announcement query has implausible page count for {asset}/{keyword}: {page_count}")
            page_number += 1
        responses.append(
            {
                "keyword": keyword,
                "page_count": page_count,
                "request_urls": request_urls,
                "rows": rows,
            }
        )
    return {
        "asset": asset,
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(KEYWORDS),
        "sql_id": SQL_ID,
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


def classify_action_title(title: str) -> tuple[str | None, str]:
    compact = re.sub(r"\s+", "", str(title))
    if "份额拆分" in compact:
        action_type = "share_split"
    elif "份额合并" in compact:
        action_type = "share_merger"
    elif "份额折算" in compact:
        action_type = "share_conversion"
    else:
        return None, "irrelevant"
    if "结果" in compact:
        role = "result"
    elif "实施" in compact or "业务安排" in compact:
        role = "implementation"
    elif "提示" in compact:
        role = "reminder"
    else:
        role = "announcement"
    return action_type, role


def parse_announcements(artifacts: list[dict[str, Any]], asset_names: dict[str, str]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        asset = str(artifact["asset"]).zfill(6)
        for response in artifact.get("responses", []):
            for row in response.get("rows", []):
                title = str(row.get("TITLE", ""))
                action_type, role = classify_action_title(title)
                relative_url = str(row.get("URL", ""))
                announcement_date = pd.to_datetime(row.get("SSEDATE"), errors="coerce")
                if action_type is None or not relative_url or pd.isna(announcement_date):
                    continue
                source_url = urljoin(SSE_BASE_URL, relative_url)
                records[source_url] = {
                    "asset": asset,
                    "asset_name": asset_names.get(asset, ""),
                    "announcement_date": pd.Timestamp(announcement_date).normalize(),
                    "action_type": action_type,
                    "document_role": role,
                    "announcement_title": title,
                    "source_url": source_url,
                    "bulletin_type": str(row.get("BULLETIN_TYPE_DESC", "")),
                    "original_bulletin_type": str(row.get("ORG_BULLETIN_TYPE_DESC", "")),
                }
    if not records:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS[:-2])
    return pd.DataFrame(records.values()).sort_values(["asset", "announcement_date", "source_url"]).reset_index(drop=True)


def _expected_action_types(inferred_factor: float) -> set[str]:
    if inferred_factor > 1:
        return {"share_split", "share_conversion"}
    return {"share_merger", "share_conversion"}


def match_queue_events(queue: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    role_rank = {"result": 0, "implementation": 1, "announcement": 2, "reminder": 3}
    rows: list[dict[str, Any]] = []
    for event in queue.itertuples(index=False):
        candidates = announcements[announcements["asset"].eq(event.asset)].copy()
        target_date = pd.Timestamp(event.price_effective_date).normalize()
        if not candidates.empty:
            candidates["announcement_distance_days"] = (candidates["announcement_date"] - target_date).dt.days
            candidates = candidates[candidates["announcement_distance_days"].between(-120, 15)].copy()
        if not candidates.empty:
            expected = _expected_action_types(float(event.inferred_factor))
            candidates["type_rank"] = (~candidates["action_type"].isin(expected)).astype(int)
            candidates["role_rank"] = candidates["document_role"].map(role_rank).fillna(4).astype(int)
            candidates["distance_rank"] = candidates["announcement_distance_days"].abs()
            candidate = candidates.sort_values(["type_rank", "role_rank", "distance_rank", "announcement_date"]).iloc[0]
            rows.append(
                {
                    "asset": event.asset,
                    "asset_name": event.asset_name,
                    "price_effective_date": target_date,
                    "inferred_factor": float(event.inferred_factor),
                    "observed_price_ratio": float(event.observed_price_ratio),
                    "match_status": "candidate_found",
                    "candidate_action_type": candidate["action_type"],
                    "candidate_document_role": candidate["document_role"],
                    "candidate_announcement_date": candidate["announcement_date"],
                    "announcement_distance_days": int(candidate["announcement_distance_days"]),
                    "candidate_title": candidate["announcement_title"],
                    "source_url": candidate["source_url"],
                }
            )
        else:
            rows.append(
                {
                    "asset": event.asset,
                    "asset_name": event.asset_name,
                    "price_effective_date": target_date,
                    "inferred_factor": float(event.inferred_factor),
                    "observed_price_ratio": float(event.observed_price_ratio),
                    "match_status": "no_candidate_in_120d_window",
                    "candidate_action_type": "",
                    "candidate_document_role": "",
                    "candidate_announcement_date": pd.NaT,
                    "announcement_distance_days": pd.NA,
                    "candidate_title": "",
                    "source_url": "",
                }
            )
    return pd.DataFrame(rows)


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
        raise ValueError(f"SSE ETF share-action document is not a PDF: {source_url}")
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
        return {
            "status": "success",
            "path": _relative(text_path),
            "sha256": _sha256(text_path),
            "cache_hit": True,
        }
    payload = pdf_path.read_bytes()
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(payload)) as document:
        for page in document.pages:
            pages.append(page.extract_text() or "")
    text = "\n\n".join(pages)
    _atomic_bytes(text.encode("utf-8"), text_path)
    return {
        "status": "success",
        "path": _relative(text_path),
        "sha256": _sha256(text_path),
        "cache_hit": False,
    }


def _factor_record(value: float, method: str, match: re.Match[str]) -> dict[str, Any] | None:
    if not (0.000001 <= value <= 1000000):
        return None
    start = max(0, match.start() - 45)
    end = min(len(match.string), match.end() + 45)
    snippet = re.sub(r"\s+", "", match.string[start:end])
    return {"factor": float(value), "method": method, "snippet": snippet[:220]}


def extract_factor_candidates(text: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", "", str(text)).replace(",", "").replace("：", ":")
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(
        r"每\s*(\d+(?:\.\d+)?)\s*份[^。；;\n]{0,160}?(?:拆分|拆成|折算|合并|变更)(?:为|成|后为)?\s*(\d+(?:\.\d+)?)\s*份",
        normalized,
    ):
        before, after = float(match.group(1)), float(match.group(2))
        record = _factor_record(after / before, "explicit_shares_before_after", match)
        if record:
            candidates.append(record)
    for match in re.finditer(
        r"(?:拆分|合并|折算)(?:的)?比例(?:为|是|按)?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)",
        normalized,
    ):
        before, after = float(match.group(1)), float(match.group(2))
        record = _factor_record(after / before, "explicit_ratio_before_after", match)
        if record:
            candidates.append(record)
    for match in re.finditer(
        r"按?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)的(?:份额)?(?:拆分|合并|折算)比例",
        normalized,
    ):
        before, after = float(match.group(1)), float(match.group(2))
        record = _factor_record(after / before, "explicit_ratio_before_action", match)
        if record:
            candidates.append(record)
    for match in re.finditer(
        r"(?:拆分|合并|折算)后[^。；;\n]{0,100}?=(?:[^。；;\n]{0,100}?)[×xX*]\s*(\d+(?:\.\d+)?)",
        normalized,
    ):
        record = _factor_record(float(match.group(1)), "explicit_formula_multiplier", match)
        if record:
            candidates.append(record)
    for match in re.finditer(
        r"(?:份额)?(?:拆分|合并|折算)比例\s*(?:为|是)?\s*(0?\.\d+|\d+(?:\.\d+)?)",
        normalized,
    ):
        record = _factor_record(float(match.group(1)), "explicit_scalar_ratio", match)
        if record:
            candidates.append(record)
    unique: dict[tuple[float, str], dict[str, Any]] = {}
    for candidate in candidates:
        unique[(round(float(candidate["factor"]), 12), str(candidate["method"]))] = candidate
    return list(unique.values())


def extract_document_dates(text: str) -> list[str]:
    dates: set[str] = set()
    for match in re.finditer(r"(20\d{2}|19\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", str(text)):
        try:
            dates.add(pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3))).date().isoformat())
        except ValueError:
            continue
    for match in re.finditer(r"(?<!\d)((?:20|19)\d{2})[-./](\d{1,2})[-./](\d{1,2})(?!\d)", str(text)):
        try:
            dates.add(pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3))).date().isoformat())
        except ValueError:
            continue
    return sorted(dates)


def extract_action_date_candidates(text: str, action_type: str) -> list[dict[str, Any]]:
    action_word = {
        "share_split": "拆分",
        "share_merger": "合并",
        "share_conversion": "折算",
    }.get(action_type)
    if action_word is None:
        return []
    compact = re.sub(r"\s+", "", str(text)).replace("(", "（").replace(")", "）")
    date_token = r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日"
    patterns = [
        (
            rf"{date_token}（(?:份额)?{action_word}(?:权益登记)?日(?:、除权日)?）",
            "date_followed_by_action_day",
            0,
        ),
        (
            rf"(?:份额)?{action_word}(?:权益登记)?日（{date_token}）",
            "action_day_followed_by_date",
            0,
        ),
        (
            rf"(?:份额)?{action_word}日（[^）]{{0,40}}?即{date_token}）",
            "action_day_parenthetical_date",
            0,
        ),
        (
            rf"权益登记日（{action_word}当日，即{date_token}）",
            "registration_day_is_action_day",
            0,
        ),
        (
            rf"(?:份额)?{action_word}(?:权益登记)?日(?:为|是|即){date_token}",
            "action_day_explicit_date",
            1,
        ),
        (
            rf"已于{date_token}（[^）]{{0,40}}?{action_word}[^）]{{0,20}}?）(?:进行|实施)",
            "completed_action_on_date",
            1,
        ),
        (
            rf"(?:已)?于{date_token}(?:进行|实施)(?:了)?(?:基金)?份额{action_word}",
            "action_performed_on_date",
            1,
        ),
        (
            rf"以{date_token}为[^。；]{{0,360}}?权益登记日[^。；]{{0,260}}?(?:进行|实施)(?:了)?(?:基金)?份额{action_word}",
            "registration_day_declared_before_completed_action",
            1,
        ),
        (
            rf"权益登记日（{date_token}）",
            "registration_day_fallback",
            2,
        ),
        (
            rf"于{date_token}完成(?:了)?(?:份额)?{action_word}",
            "action_completed_on_date",
            2,
        ),
    ]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for pattern, method, rank in patterns:
        for match in re.finditer(pattern, compact):
            groups = match.groups()
            year, month, day = map(int, groups[-3:])
            try:
                date_value = pd.Timestamp(year=year, month=month, day=day).date().isoformat()
            except ValueError:
                continue
            start = max(0, match.start() - 45)
            end = min(len(compact), match.end() + 45)
            records[(date_value, method)] = {
                "date": date_value,
                "method": method,
                "rank": rank,
                "snippet": compact[start:end][:220],
            }
    return sorted(records.values(), key=lambda item: (int(item["rank"]), str(item["date"]), str(item["method"])))


def select_action_event_date(candidates: list[dict[str, Any]], price_effective_date: Any) -> tuple[Any, Any]:
    if not candidates:
        return pd.NaT, pd.NA
    target = pd.Timestamp(price_effective_date).normalize()
    eligible = []
    for candidate in candidates:
        date_value = pd.Timestamp(candidate["date"]).normalize()
        distance = int((target - date_value).days)
        eligible.append((int(candidate["rank"]), 0 if 0 <= distance <= 15 else 1, abs(distance), date_value))
    _, _, _, selected = min(eligible)
    return selected, int((target - selected).days)


def _annotate_match_from_document(row: dict[str, Any], document: dict[str, Any], text_meta: dict[str, Any]) -> dict[str, Any]:
    text = (ROOT / str(text_meta["path"])).read_text(encoding="utf-8")
    factors = extract_factor_candidates(text)
    target_factor = float(row["inferred_factor"])
    best_factor = pd.NA
    factor_error = pd.NA
    if factors:
        best = min(factors, key=lambda item: abs(float(item["factor"]) / target_factor - 1.0))
        best_factor = float(best["factor"])
        factor_error = abs(best_factor / target_factor - 1.0)
    parsed_dates = extract_document_dates(text)
    date_distance = pd.NA
    if parsed_dates:
        target_date = pd.Timestamp(row["price_effective_date"])
        date_distance = min(abs((pd.Timestamp(value) - target_date).days) for value in parsed_dates)
    factor_matched = pd.notna(factor_error) and float(factor_error) <= 0.001
    date_near = pd.notna(date_distance) and int(date_distance) <= 7
    action_date_candidates = extract_action_date_candidates(text, str(row["candidate_action_type"]))
    action_event_date, action_event_distance = select_action_event_date(
        action_date_candidates,
        row["price_effective_date"],
    )
    observed_ratio = float(row["observed_price_ratio"])
    observed_error = (
        abs(float(best_factor) / observed_ratio - 1.0)
        if pd.notna(best_factor) and observed_ratio > 0
        else pd.NA
    )
    if factor_matched and date_near and pd.notna(action_event_date):
        review_status = "official_pdf_factor_and_near_date_found_review_required"
    elif factor_matched and pd.notna(action_event_date):
        review_status = "official_pdf_factor_found_date_unresolved"
    elif pd.notna(best_factor) and pd.notna(action_event_date) and pd.notna(observed_error) and float(observed_error) <= 0.10:
        review_status = "official_pdf_corrects_heuristic_review_required"
    else:
        review_status = "official_pdf_parse_inconclusive"
    return {
        **row,
        "pdf_status": "success",
        "pdf_path": document["path"],
        "pdf_sha256": document["sha256"],
        "text_path": text_meta["path"],
        "text_sha256": text_meta["sha256"],
        "parsed_factor_candidates_json": json.dumps(factors, ensure_ascii=False, separators=(",", ":")),
        "best_parsed_factor": best_factor,
        "factor_relative_error": factor_error,
        "parsed_dates_json": json.dumps(parsed_dates, ensure_ascii=False, separators=(",", ":")),
        "nearest_document_date_distance_days": date_distance,
        "parsed_action_date_candidates_json": json.dumps(
            action_date_candidates,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "best_action_event_date": action_event_date,
        "action_event_date_distance_days": action_event_distance,
        "evidence_review_status": review_status,
        "historical_backtest_allowed": False,
    }


def _empty_document_annotation(row: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        **row,
        "pdf_status": status,
        "pdf_path": "",
        "pdf_sha256": "",
        "text_path": "",
        "text_sha256": "",
        "parsed_factor_candidates_json": "[]",
        "best_parsed_factor": pd.NA,
        "factor_relative_error": pd.NA,
        "parsed_dates_json": "[]",
        "nearest_document_date_distance_days": pd.NA,
        "parsed_action_date_candidates_json": "[]",
        "best_action_event_date": pd.NaT,
        "action_event_date_distance_days": pd.NA,
        "evidence_review_status": "no_official_candidate" if not row.get("source_url") else "official_pdf_unavailable",
        "historical_backtest_allowed": False,
    }


def build_registry_candidates(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in matches.itertuples(index=False):
        if match.pdf_status != "success" or pd.isna(match.best_parsed_factor) or pd.isna(match.best_action_event_date):
            continue
        factor = float(match.best_parsed_factor)
        observed_ratio = float(match.observed_price_ratio)
        rows.append(
            {
                "asset": str(match.asset).zfill(6),
                "action_type": str(match.candidate_action_type),
                "event_date": pd.Timestamp(match.best_action_event_date).normalize(),
                "price_effective_date": pd.Timestamp(match.price_effective_date).normalize(),
                "shares_after_per_share_before": factor,
                "announcement_date": pd.Timestamp(match.candidate_announcement_date).normalize(),
                "source_document_title": str(match.candidate_title),
                "source_url": str(match.source_url),
                "source_type": "exchange_announcement",
                "pdf_path": str(match.pdf_path),
                "pdf_sha256": str(match.pdf_sha256),
                "text_path": str(match.text_path),
                "text_sha256": str(match.text_sha256),
                "factor_relative_error_to_inference": abs(factor / float(match.inferred_factor) - 1.0),
                "factor_relative_error_to_observed_price_ratio": abs(factor / observed_ratio - 1.0),
                "normalized_price_ratio_residual": abs(observed_ratio / factor - 1.0),
                "action_event_date_distance_days": int(match.action_event_date_distance_days),
                "review_status": str(match.evidence_review_status),
                "historical_backtest_allowed": False,
            }
        )
    if not rows:
        return pd.DataFrame(columns=REGISTRY_CANDIDATE_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=REGISTRY_CANDIDATE_COLUMNS).sort_values(
        ["asset", "price_effective_date"]
    ).reset_index(drop=True)


def run_collection(
    as_of: str,
    *,
    assets: list[str] | None = None,
    max_assets: int | None = None,
    sleep_seconds: float = 0.15,
    max_consecutive_failures: int = 5,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    queue = load_sse_queue()
    all_queue_assets = sorted(queue["asset"].unique())
    requested_assets = sorted({str(asset).zfill(6) for asset in assets}) if assets else all_queue_assets
    unknown_assets = sorted(set(requested_assets).difference(all_queue_assets))
    if unknown_assets:
        raise ValueError(f"Requested assets are absent from the SSE evidence queue: {unknown_assets}")
    queue = queue[queue["asset"].isin(requested_assets)].copy()
    target_assets = requested_assets
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
    consecutive_failures = 0
    for position, asset in enumerate(selected):
        try:
            metadata = save_query_artifact(fetch_query_artifact(session, asset, as_of_date))
            status["assets"][asset] = metadata
            consecutive_failures = 0
        except (OSError, ValueError, requests.RequestException) as exc:
            status["assets"][asset] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            consecutive_failures += 1
        _atomic_json(status, STATUS_PATH)
        if consecutive_failures >= max_consecutive_failures:
            status["circuit_breaker"] = f"stopped_after_{consecutive_failures}_consecutive_failures"
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
                {"source_id": f"sse_etf_announcement_query:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"source_id": f"sse_etf_announcement_query_meta:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )
    asset_names = queue.drop_duplicates("asset").set_index("asset")["asset_name"].astype(str).to_dict()
    announcements = parse_announcements(artifacts, asset_names)
    query_bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    announcements["data_source"] = "sse_official_fund_announcements"
    announcements["source_vintage"] = f"sse_etf_share_action_query_bundle_sha256:{query_bundle_hash}"
    announcements = announcements.reindex(columns=ANNOUNCEMENT_COLUMNS)
    _atomic_csv(announcements, ANNOUNCEMENT_PATH)

    matches = match_queue_events(queue, announcements)
    annotated_by_url: dict[str, dict[str, Any]] = {}
    final_matches: list[dict[str, Any]] = []
    document_failures: dict[str, str] = {}
    for row in matches.to_dict("records"):
        source_url = str(row.get("source_url", ""))
        if not source_url:
            final_matches.append(_empty_document_annotation(row, "not_applicable"))
            continue
        if source_url not in annotated_by_url:
            try:
                document = download_document(session, source_url)
                text_meta = extract_document_text(document)
                annotated_by_url[source_url] = {"document": document, "text": text_meta}
                inputs.extend(
                    [
                        {
                            "source_id": f"sse_etf_share_action_pdf:{hashlib.sha256(source_url.encode()).hexdigest()[:16]}",
                            "path": str(document["path"]),
                            "sha256": str(document["sha256"]),
                        },
                        {
                            "source_id": f"sse_etf_share_action_text:{hashlib.sha256(source_url.encode()).hexdigest()[:16]}",
                            "path": str(text_meta["path"]),
                            "sha256": str(text_meta["sha256"]),
                        },
                    ]
                )
            except (OSError, ValueError, requests.RequestException, PDFSyntaxError) as exc:
                document_failures[source_url] = f"{type(exc).__name__}: {str(exc)[:400]}"
        if source_url in annotated_by_url:
            bundle = annotated_by_url[source_url]
            final_matches.append(_annotate_match_from_document(row, bundle["document"], bundle["text"]))
        else:
            final_matches.append(_empty_document_annotation(row, "failed"))
    match_frame = pd.DataFrame(final_matches).reindex(columns=MATCH_COLUMNS)
    _atomic_csv(match_frame, MATCH_PATH)
    registry_candidates = build_registry_candidates(match_frame)
    _atomic_csv(registry_candidates, REGISTRY_CANDIDATE_PATH)

    cached_assets = sum(_valid_query_cache(asset, as_of_date) for asset in target_assets)
    candidate_count = int(match_frame["match_status"].eq("candidate_found").sum())
    parsed_factor_count = int(match_frame["best_parsed_factor"].notna().sum())
    exact_factor_count = int(pd.to_numeric(match_frame["factor_relative_error"], errors="coerce").le(0.001).sum())
    official_factor_count = int(match_frame["best_parsed_factor"].notna().sum())
    heuristic_correction_count = int(
        match_frame["evidence_review_status"].eq("official_pdf_corrects_heuristic_review_required").sum()
    )
    complete_queries = cached_assets == len(target_assets)
    all_candidates = candidate_count == len(queue)
    qualification = (
        "OFFICIAL_DISCOVERY_COMPLETE_REVIEW_REQUIRED"
        if complete_queries and all_candidates and not document_failures
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
            {
                "path": _relative(REGISTRY_CANDIDATE_PATH),
                "sha256": _sha256(REGISTRY_CANDIDATE_PATH),
                "rows": int(len(registry_candidates)),
            },
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "target_assets": len(target_assets),
        "target_events": int(len(queue)),
        "query_complete_assets": int(cached_assets),
        "announcement_candidates": int(len(announcements)),
        "queue_events_with_candidate": candidate_count,
        "queue_events_with_parsed_factor": parsed_factor_count,
        "queue_events_with_official_factor": official_factor_count,
        "queue_events_with_exact_factor_match": exact_factor_count,
        "queue_events_with_heuristic_correction": heuristic_correction_count,
        "registry_candidate_rows": int(len(registry_candidates)),
        "document_failure_count": len(document_failures),
        "document_failures": document_failures,
        "qualification_status": qualification,
        "method_boundary": (
            "Official SSE query responses and matched PDFs are preserved with hashes. Automated text parsing is discovery-only; "
            "no factor is written to the governed corporate-action registry without document review and price-series validation."
        ),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(
        {
            "query_complete_assets": int(cached_assets),
            "announcement_candidates": int(len(announcements)),
            "queue_events_with_candidate": candidate_count,
            "document_failure_count": len(document_failures),
            "qualification_status": qualification,
        }
    )
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--asset", action="append", help="Restrict collection to one queued SSE ETF code; repeat as needed")
    parser.add_argument("--max-assets", type=int, help="Fetch at most this many uncached SSE ETF codes; omit for all")
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
