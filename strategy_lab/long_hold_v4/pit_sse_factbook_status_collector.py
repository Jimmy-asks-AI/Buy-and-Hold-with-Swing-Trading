"""Collect SSE factbook restoration events as governed official evidence.

The annual factbooks contain retrospective tables for listing restorations that
cannot be recovered reliably from title-only company-announcement searches.
Raw PDFs are kept immutably; parsed events remain observation-only until the
cross-source status validator passes.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .pit_stock_market_history_builder import ROOT
from .pit_stock_name_history_collector import classify_security_name


RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_sse_factbooks"
OUTPUT_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "sse_factbook_restoration_events.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_factbook_status_latest.json"
REFERENCE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "sse_factbook_status_reference_events.csv"
)
COLLECTOR_VERSION = "sse_factbook_restoration_events_v2"

# The SSE archive has no 2007 edition entry. Each URL is an official exchange PDF.
FACTBOOK_URLS = {
    2001: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170580/files/264f684a05a74342a4ea941e3f1f169b.pdf",
    2002: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170579/files/e849c3d34ad145adbc995a6648ff9371.pdf",
    2003: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170578/files/0b7476e76eb045b7ba52ced90434575d.pdf",
    2004: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170577/files/ae3c4a6d91b74aacbddc96a4d600f06f.pdf",
    2005: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170576/files/efc5dc34faad42ce929c3eff0368e701.pdf",
    2006: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170575/files/c8fb197bd52a4a578c678c9c25d23807.pdf",
    2008: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170574/files/977f9c4f930746c18732b7d063f5676a.pdf",
    2009: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170573/files/073e042b8b87407b919e3d8e75ee91a5.pdf",
    2010: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170572/files/36d0635dee474838943e931bcc04c3df.pdf",
    2011: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170571/files/f43f33c247f242d48780c3097548281b.pdf",
    2012: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170570/files/9a7b8e0d00e84d358d02fbba056f7bda.pdf",
    2013: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170569/files/cdccf6c0b46a4c0cadc0a531ecc3aad7.pdf",
    2014: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170568/files/7ff41b8a306a4376aa785aec7bb1fc34.pdf",
    2015: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170567/files/cafa59ab90014dd98a4b3de148a75a5d.pdf",
    2016: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170566/files/23ad0cdde73846fa88b1ee417d306eb9.pdf",
    2017: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170565/files/42dc9278c7a14ff9b2d887ff0fbf4dfb.pdf",
    2018: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170564/files/300a2dcef0f0452794683bd1bfbf65b0.pdf",
    2019: "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170563/files/9f33f39ecb1c4546a88ec88b81105434.pdf",
}

RESTORATION_HEADING = re.compile(r"(?P<year>\d{4})\s*年\s*恢复\s*上市\s*的\s*公司")
NEXT_COMPANY_SECTION = re.compile(r"\n\s*\d{4}\s*年[^\n]{0,30}的\s*公司")
STATUS_SECTION_HEADING = re.compile(
    r"(?P<year>\d{4})\s*年\s*"
    r"(?P<label>实施退市风险警示|实施其他风险警示|取消特别处理|撤销风险警示|"
    r"特别处理|暂停上市|终止上市|除牌|恢复上市)\s*(?:的)?\s*公司"
)
NEXT_FACTBOOK_SECTION = re.compile(
    r"\n\s*(?:\d{4}\s*年|上市公司(?:名称|简称)|债券市场|权证|基金|会员)"
)
ROW_START = re.compile(r"^\s*(?:\d+\s+)?(?P<asset>6\d{5})\s+(?P<body>.+?)\s*$")
DATE_TOKEN = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}")
SUBSECTION_HEADING = re.compile(r"^\s*[（(][一二三四五六七八九十]+[)）]\s*(?P<label>.+?)\s*$")
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
    "factbook_edition",
    "factbook_event_year",
    "factbook_page",
    "pre_restoration_name",
    "restored_name",
    "availability_basis",
]
REFERENCE_COLUMNS = [
    *OUTPUT_COLUMNS,
    "event_class",
    "event_subclass",
    "binary_state_change",
    "used_in_reconciliation",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _atomic_gzip_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as archive:
        archive.write(encoded)
    _atomic_bytes(buffer.getvalue(), path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    temporary.replace(path)


def _session() -> requests.Session:
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"})
    return session


def _cached_factbook(edition: int, source_url: str) -> tuple[bytes, dict[str, Any]] | None:
    metadata_path = RAW_DIR / f"{edition}_latest.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_path = ROOT / str(metadata["raw_path"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    if metadata.get("source_url") != source_url or not raw_path.exists():
        return None
    payload = raw_path.read_bytes()
    if not payload.startswith(b"%PDF") or _sha256_bytes(payload) != metadata.get("sha256"):
        return None
    return payload, metadata


def load_or_download_factbook(edition: int, source_url: str) -> tuple[bytes, dict[str, Any]]:
    cached = _cached_factbook(edition, source_url)
    if cached is not None:
        return cached
    with _session() as session:
        response = session.get(source_url, timeout=(15, 180))
        response.raise_for_status()
        payload = response.content
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"SSE factbook {edition} response is not a PDF")
    digest = _sha256_bytes(payload)
    raw_path = RAW_DIR / f"sse_factbook_{edition}_{digest[:12]}.pdf"
    if not raw_path.exists():
        _atomic_bytes(payload, raw_path)
    metadata = {
        "edition": edition,
        "source_url": source_url,
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_path": _relative(raw_path),
        "sha256": digest,
        "bytes": len(payload),
    }
    _atomic_json(metadata, RAW_DIR / f"{edition}_latest.json")
    return payload, metadata


def _status_from_restored_name(name: str) -> str:
    status = classify_security_name(name)
    return "normal" if status == "listing_marker" else status


def load_or_extract_page_texts(
    payload: bytes,
    *,
    edition: int,
    source_sha256: str,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    cache_path = RAW_DIR / f"sse_factbook_{edition}_{source_sha256[:12]}.pages.json.gz"
    if cache_path.exists():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as archive:
                cached = json.load(archive)
            if (
                cached.get("source_sha256") == source_sha256
                and cached.get("pdfplumber_version") == pdfplumber.__version__
                and isinstance(cached.get("page_texts"), list)
            ):
                return cached["page_texts"], cached.get("page_extraction_errors", []), {
                    "path": _relative(cache_path),
                    "sha256": _sha256(cache_path),
                    "cache_hit": True,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    page_texts: list[str] = []
    extraction_errors: list[dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(payload)) as document:
        for page_number, page in enumerate(document.pages, start=1):
            try:
                page_texts.append(page.extract_text() or "")
            except Exception as exc:  # pragma: no cover - corrupt vendor-page fallback
                page_texts.append("")
                extraction_errors.append({"page": page_number, "error": str(exc)})
    _atomic_gzip_json(
        {
            "edition": edition,
            "source_sha256": source_sha256,
            "pdfplumber_version": pdfplumber.__version__,
            "page_texts": page_texts,
            "page_extraction_errors": extraction_errors,
        },
        cache_path,
    )
    return page_texts, extraction_errors, {
        "path": _relative(cache_path),
        "sha256": _sha256(cache_path),
        "cache_hit": False,
    }


def _document_text(page_texts: list[str]) -> tuple[str, list[int]]:
    starts: list[int] = []
    cursor = 0
    for text in page_texts:
        starts.append(cursor)
        cursor += len(text) + 1
    return "\n".join(page_texts), starts


def _parse_restoration_line(
    line: str,
    *,
    edition: int,
    event_year: int,
    page_number: int,
    source_url: str,
    source_sha256: str,
) -> dict[str, Any] | None:
    start = ROW_START.match(line)
    if start is None:
        return None
    body = start.group("body")
    dates = list(DATE_TOKEN.finditer(body))
    if not dates:
        return None
    before = body[: dates[0].start()].strip().split()
    if not before:
        return None
    if len(dates) >= 2:
        announcement_date = pd.Timestamp(dates[0].group().replace(".", "-").replace("/", "-")).normalize()
        effective_date = pd.Timestamp(dates[1].group().replace(".", "-").replace("/", "-")).normalize()
        after = body[dates[1].end() :].strip().split()
        availability_basis = "official_table_announcement_date"
    else:
        effective_date = pd.Timestamp(dates[0].group().replace(".", "-").replace("/", "-")).normalize()
        announcement_date = effective_date
        after = body[dates[0].end() :].strip().split()
        availability_basis = "effective_date_proxy_no_announcement_column"
    restored_name = after[0] if after else before[0]
    status = _status_from_restored_name(restored_name)
    if status not in {"normal", "risk_warning", "special_transfer", "delisting", "listing_suspended"}:
        return None
    return {
        "asset": start.group("asset"),
        "effective_date": effective_date,
        "execution_status": status,
        "is_st": status in {"risk_warning", "special_transfer"},
        "announcement_date": announcement_date,
        "available_date": announcement_date,
        "announcement_title": f"{event_year}年恢复上市的公司（上交所市场资料{edition}卷）",
        "source_url": source_url,
        "data_source": "sse_official_factbook_restoration_table",
        "source_vintage": f"sse_factbook_{edition}_sha256:{source_sha256}",
        "factbook_edition": edition,
        "factbook_event_year": event_year,
        "factbook_page": page_number,
        "pre_restoration_name": before[0] if after else "",
        "restored_name": restored_name,
        "availability_basis": availability_basis,
    }


def extract_restoration_events(
    payload: bytes,
    *,
    edition: int,
    source_url: str,
    source_sha256: str,
    as_of: str | pd.Timestamp,
    page_texts: list[str] | None = None,
    extraction_errors: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if page_texts is None:
        page_texts, extraction_errors, _ = load_or_extract_page_texts(
            payload,
            edition=edition,
            source_sha256=source_sha256,
        )
    extraction_errors = extraction_errors or []
    full_text, starts = _document_text(page_texts)
    rows: list[dict[str, Any]] = []
    heading_count = 0
    for heading in RESTORATION_HEADING.finditer(full_text):
        heading_count += 1
        event_year = int(heading.group("year"))
        section_end_match = NEXT_COMPANY_SECTION.search(full_text, heading.end())
        section_end = section_end_match.start() if section_end_match else len(full_text)
        segment = full_text[heading.end() : section_end]
        segment_start = heading.end()
        running_offset = 0
        for line in segment.splitlines(keepends=True):
            absolute_offset = segment_start + running_offset
            page_number = bisect.bisect_right(starts, absolute_offset)
            parsed = _parse_restoration_line(
                line.rstrip("\r\n"),
                edition=edition,
                event_year=event_year,
                page_number=max(page_number, 1),
                source_url=source_url,
                source_sha256=source_sha256,
            )
            if parsed is not None:
                rows.append(parsed)
            running_offset += len(line)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame = frame[pd.to_datetime(frame["effective_date"]).le(pd.Timestamp(as_of).normalize())].copy()
        frame = frame.drop_duplicates(["asset", "effective_date"], keep="last")
        frame = frame.sort_values(["effective_date", "asset"]).reset_index(drop=True)
    return frame, {
        "edition": edition,
        "pages": len(page_texts),
        "restoration_headings": heading_count,
        "restoration_rows": int(len(frame)),
        "page_extraction_errors": extraction_errors,
    }


def _section_default_status(label: str) -> str:
    if label in {"实施退市风险警示", "实施其他风险警示", "特别处理"}:
        return "risk_warning"
    if label in {"取消特别处理", "撤销风险警示"}:
        return "normal"
    if label == "暂停上市":
        return "listing_suspended"
    if label in {"终止上市", "除牌"}:
        return "delisting"
    if label == "恢复上市":
        return "normal"
    raise ValueError(f"unsupported factbook status section: {label}")


def _subsection_semantics(line: str) -> str | None:
    match = SUBSECTION_HEADING.match(line.strip())
    if match is None:
        return None
    label = re.sub(r"\s+", "", match.group("label"))
    if "撤销退市风险警示" in label and "实施其他" in label:
        return "remove_delisting_warning_keep_other_warning"
    if "撤销退市风险警示" in label:
        return "remove_delisting_warning"
    if "撤销其他风险警示" in label:
        return "remove_other_warning"
    if "取消特别处理" in label:
        return "cancel_special_treatment"
    return f"unclassified:{label}"


def _reference_dates(label: str, dates: list[re.Match[str]]) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    parsed = [pd.Timestamp(item.group().replace(".", "-").replace("/", "-")).normalize() for item in dates]
    if len(parsed) == 1:
        return parsed[0], parsed[0], "effective_date_proxy_no_announcement_column"
    if label in {"取消特别处理", "撤销风险警示"}:
        # These tables print the implementation start before the announcement
        # date. Some older PDFs contain an extra text-extraction token before
        # the two governed date columns, so the final two dates are canonical.
        return parsed[-2], parsed[-1], "official_table_announcement_date"
    return parsed[-1], parsed[0], "official_table_announcement_date"


def _reference_status(label: str, event_subclass: str, post_event_name: str) -> str:
    default = _section_default_status(label)
    if label not in {"取消特别处理", "撤销风险警示"}:
        return default
    if event_subclass == "remove_delisting_warning_keep_other_warning":
        return "risk_warning"
    if label == "取消特别处理" and event_subclass == "remove_delisting_warning":
        # In the older tables this subsection records *ST -> ST: the delisting
        # warning is removed while special treatment remains in force.
        return "risk_warning"
    if post_event_name:
        inferred = _status_from_restored_name(post_event_name)
        if inferred in {"risk_warning", "special_transfer"}:
            return inferred
    return default


def _binary_state_change(label: str, event_subclass: str, before_status: str, after_status: str) -> bool:
    if label in {"暂停上市", "终止上市", "除牌"}:
        return False
    if label == "取消特别处理" and event_subclass in {
        "remove_delisting_warning",
        "remove_delisting_warning_keep_other_warning",
    }:
        return False
    if label == "撤销风险警示" and event_subclass == "remove_delisting_warning_keep_other_warning":
        return False
    if label == "恢复上市":
        before_is_st = before_status in {"risk_warning", "special_transfer"}
        after_is_st = after_status in {"risk_warning", "special_transfer"}
        return before_is_st != after_is_st
    return True


def _parse_reference_line(
    line: str,
    *,
    label: str,
    edition: int,
    event_year: int,
    page_number: int,
    source_url: str,
    source_sha256: str,
    event_subclass: str = "",
) -> dict[str, Any] | None:
    if label == "恢复上市":
        restored = _parse_restoration_line(
            line,
            edition=edition,
            event_year=event_year,
            page_number=page_number,
            source_url=source_url,
            source_sha256=source_sha256,
        )
        if restored is None:
            return None
        before_status = _status_from_restored_name(str(restored.get("pre_restoration_name", "")))
        return {
            **restored,
            "event_class": label,
            "event_subclass": "restoration",
            "binary_state_change": _binary_state_change(
                label,
                "restoration",
                before_status,
                str(restored["execution_status"]),
            ),
            "used_in_reconciliation": True,
        }
    start = ROW_START.match(line)
    if start is None:
        return None
    body = start.group("body")
    dates = list(DATE_TOKEN.finditer(body))
    if not dates:
        return None
    non_date_body = DATE_TOKEN.sub(" ", body)
    name_tokens = [
        token
        for token in non_date_body.strip().split()
        if token not in {"后A股简称", "后B股简称", "A股简称", "B股简称", "备注"}
    ]
    if not name_tokens:
        return None
    effective_date, announcement_date, availability_basis = _reference_dates(label, dates)
    available_date = announcement_date
    if announcement_date > effective_date and label in {"终止上市", "除牌"}:
        # A small number of legacy tables record the newspaper publication on
        # the following non-trading day. The terminal execution state was
        # already observable on its effective date; retain both dates and mark
        # the conservative execution-availability cap explicitly.
        available_date = effective_date
        availability_basis = "official_terminal_announcement_after_effective_capped_to_effective"
    post_event_name = name_tokens[-1] if len(name_tokens) >= 2 else ""
    status = _reference_status(label, event_subclass, post_event_name)
    before_status = _status_from_restored_name(name_tokens[0])
    return {
        "asset": start.group("asset"),
        "effective_date": effective_date,
        "execution_status": status,
        "is_st": status in {"risk_warning", "special_transfer"},
        "announcement_date": announcement_date,
        "available_date": available_date,
        "announcement_title": f"{event_year}年{label}的公司（上交所市场资料{edition}卷）",
        "source_url": source_url,
        "data_source": "sse_official_factbook_status_reference_table",
        "source_vintage": f"sse_factbook_{edition}_sha256:{source_sha256}",
        "factbook_edition": edition,
        "factbook_event_year": event_year,
        "factbook_page": page_number,
        "pre_restoration_name": name_tokens[0],
        "restored_name": post_event_name,
        "availability_basis": availability_basis,
        "event_class": label,
        "event_subclass": event_subclass,
        "binary_state_change": _binary_state_change(label, event_subclass, before_status, status),
        "used_in_reconciliation": False,
    }


def extract_status_reference_events(
    *,
    page_texts: list[str],
    edition: int,
    source_url: str,
    source_sha256: str,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_text, starts = _document_text(page_texts)
    rows: list[dict[str, Any]] = []
    heading_counts: dict[str, int] = {}
    for heading in STATUS_SECTION_HEADING.finditer(full_text):
        label = heading.group("label")
        event_year = int(heading.group("year"))
        heading_counts[label] = heading_counts.get(label, 0) + 1
        next_heading = NEXT_FACTBOOK_SECTION.search(full_text, heading.end())
        section_end = next_heading.start() if next_heading else len(full_text)
        segment = full_text[heading.end() : section_end]
        running_offset = 0
        event_subclass = ""
        for line in segment.splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            subsection = _subsection_semantics(stripped)
            if subsection is not None:
                event_subclass = subsection
            absolute_offset = heading.end() + running_offset
            page_number = max(bisect.bisect_right(starts, absolute_offset), 1)
            parsed = _parse_reference_line(
                stripped,
                label=label,
                edition=edition,
                event_year=event_year,
                page_number=page_number,
                source_url=source_url,
                source_sha256=source_sha256,
                event_subclass=event_subclass,
            )
            if parsed is not None:
                rows.append(parsed)
            running_offset += len(line)
    frame = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    if not frame.empty:
        frame = frame[pd.to_datetime(frame["effective_date"]).le(pd.Timestamp(as_of).normalize())].copy()
        frame = frame.drop_duplicates(["asset", "effective_date", "execution_status"], keep="last")
        frame = frame.sort_values(["effective_date", "asset"]).reset_index(drop=True)
    return frame, {
        "edition": edition,
        "status_section_headings": heading_counts,
        "status_reference_rows": int(len(frame)),
    }


def run_collection(as_of: str, workers: int = 4) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    as_of_date = pd.Timestamp(as_of).normalize()
    fetched: dict[int, tuple[bytes, dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_or_download_factbook, edition, url): edition
            for edition, url in FACTBOOK_URLS.items()
        }
        for future in as_completed(futures):
            edition = futures[future]
            try:
                fetched[edition] = future.result()
            except Exception as exc:
                failures[str(edition)] = str(exc)

    page_results: dict[int, tuple[list[str], list[dict[str, Any]], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                load_or_extract_page_texts,
                payload,
                edition=edition,
                source_sha256=str(metadata["sha256"]),
            ): edition
            for edition, (payload, metadata) in fetched.items()
        }
        for future in as_completed(futures):
            edition = futures[future]
            try:
                page_results[edition] = future.result()
            except Exception as exc:
                failures[str(edition)] = f"text extraction: {type(exc).__name__}: {exc}"

    frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for edition in sorted(page_results):
        payload, metadata = fetched[edition]
        page_texts, extraction_errors, text_cache = page_results[edition]
        frame, diagnostic = extract_restoration_events(
            payload,
            edition=edition,
            source_url=FACTBOOK_URLS[edition],
            source_sha256=str(metadata["sha256"]),
            as_of=as_of_date,
            page_texts=page_texts,
            extraction_errors=extraction_errors,
        )
        reference_frame, reference_diagnostic = extract_status_reference_events(
            page_texts=page_texts,
            edition=edition,
            source_url=FACTBOOK_URLS[edition],
            source_sha256=str(metadata["sha256"]),
            as_of=as_of_date,
        )
        frames.append(frame)
        reference_frames.append(reference_frame)
        diagnostics.append({**diagnostic, **reference_diagnostic})
        inputs.append(
            {
                "edition": edition,
                "source_url": FACTBOOK_URLS[edition],
                "path": metadata["raw_path"],
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
                "pages": diagnostic["pages"],
                "restoration_rows": diagnostic["restoration_rows"],
                "status_reference_rows": reference_diagnostic["status_reference_rows"],
                "text_cache_path": text_cache["path"],
                "text_cache_sha256": text_cache["sha256"],
                "text_cache_hit": text_cache["cache_hit"],
            }
        )
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["asset", "effective_date", "factbook_edition"])
        events = events.drop_duplicates(["asset", "effective_date"], keep="last").reset_index(drop=True)
    if events.duplicated(["asset", "effective_date"]).any():
        raise ValueError("factbook restoration output contains duplicate asset dates")
    if not events.empty and (
        pd.to_datetime(events["announcement_date"]) > pd.to_datetime(events["effective_date"])
    ).any():
        raise ValueError("factbook restoration has announcement dates after effective dates")
    _atomic_csv(events, OUTPUT_PATH)

    reference_events = (
        pd.concat(reference_frames, ignore_index=True)
        if reference_frames
        else pd.DataFrame(columns=REFERENCE_COLUMNS)
    )
    if not reference_events.empty:
        reference_events = reference_events.sort_values(
            ["asset", "effective_date", "execution_status", "factbook_edition"]
        )
        reference_events = reference_events.drop_duplicates(
            ["asset", "effective_date", "execution_status"], keep="last"
        ).reset_index(drop=True)
        invalid_availability = pd.to_datetime(reference_events["available_date"]).gt(
            pd.to_datetime(reference_events["effective_date"])
        )
        if invalid_availability.any():
            examples = reference_events.loc[
                invalid_availability,
                ["asset", "effective_date", "announcement_date", "available_date", "event_class", "event_subclass"],
            ].head(10)
            raise ValueError(f"factbook references are unavailable on their effective dates: {examples.to_dict('records')}")
    _atomic_csv(reference_events, REFERENCE_PATH)

    references = {
        "600057": {"effective_date": "2011-08-29", "execution_status": "normal"},
        "600633": {"effective_date": "2011-09-29", "execution_status": "normal"},
    }
    reference_checks: dict[str, dict[str, Any]] = {}
    for asset, expected in references.items():
        matched = events[
            events["asset"].astype(str).eq(asset)
            & pd.to_datetime(events["effective_date"]).eq(pd.Timestamp(expected["effective_date"]))
            & events["execution_status"].eq(expected["execution_status"])
        ]
        reference_checks[asset] = {**expected, "matched": len(matched) == 1}
    all_pages_clean = not any(item["page_extraction_errors"] for item in diagnostics)
    ready = (
        len(page_results) == len(FACTBOOK_URLS)
        and not failures
        and all_pages_clean
        and all(item["matched"] for item in reference_checks.values())
    )
    independent_reference = reference_events[~reference_events["used_in_reconciliation"].astype(bool)].copy()
    reference_statuses = set(independent_reference["execution_status"].astype(str))
    reference_ready = (
        50 <= len(independent_reference) <= 2_000
        and {"normal", "risk_warning", "listing_suspended", "delisting"}.issubset(reference_statuses)
    )
    bundle_hash = hashlib.sha256(
        "|".join(f"{item['edition']}:{item['sha256']}" for item in inputs).encode()
    ).hexdigest()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "collector_version": COLLECTOR_VERSION,
        "as_of_date": as_of_date.date().isoformat(),
        "archive_note": "The official SSE factbook index has no 2007 edition entry.",
        "requested_editions": sorted(FACTBOOK_URLS),
        "downloaded_editions": sorted(fetched),
        "failed_editions": failures,
        "inputs": inputs,
        "diagnostics": diagnostics,
        "reference_checks": reference_checks,
        "output_path": _relative(OUTPUT_PATH),
        "output_sha256": _sha256(OUTPUT_PATH),
        "reference_output_path": _relative(REFERENCE_PATH),
        "reference_output_sha256": _sha256(REFERENCE_PATH),
        "source_vintage": f"sse_factbook_bundle_sha256:{bundle_hash}",
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(events)),
        "assets": int(events["asset"].nunique()) if not events.empty else 0,
        "event_year_min": int(events["factbook_event_year"].min()) if not events.empty else None,
        "event_year_max": int(events["factbook_event_year"].max()) if not events.empty else None,
        "reference_rows": int(len(reference_events)),
        "independent_reference_rows": int(len(independent_reference)),
        "reference_status_counts": independent_reference["execution_status"].value_counts().sort_index().to_dict(),
        "reference_qualification_status": "READY_FOR_HOLDOUT" if reference_ready else "REFERENCE_INCOMPLETE",
        "qualification_status": "READY_FOR_RECONCILIATION" if ready else "FACTBOOK_COLLECTION_INCOMPLETE",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "qualification_blockers": ["factbook-derived events require independent daily-state validation"],
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_collection(args.as_of, workers=args.workers)
    keys = (
        "qualification_status",
        "downloaded_editions",
        "failed_editions",
        "rows",
        "assets",
        "event_year_min",
        "event_year_max",
        "reference_checks",
        "reference_rows",
        "independent_reference_rows",
        "reference_qualification_status",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
