"""Collect and hash the selected official ETF benchmark documents."""

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
from urllib.parse import urlparse

import pandas as pd
import pymupdf
import requests
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

from . import pit_etf_benchmark_document_selector as selector
from . import pit_etf_dividend_announcement_collector as official
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
SELECTOR_MANIFEST_PATH = selector.MANIFEST_PATH
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
DOCUMENT_META_DIR = RAW_DIR / "document_metadata"
STATUS_PATH = RAW_DIR / "document_collection_status.json"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
DOCUMENT_INDEX_PATH = OBSERVATION_DIR / "etf_benchmark_document_index.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_document_collection_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_benchmark_documents" / "collection_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_document_collector_latest.json"

SCHEMA_VERSION = 1
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
SUPPORTED_DOMAINS = {"www.sse.com.cn", "static.sse.com.cn", "static.cninfo.com.cn"}
TEXT_AVAILABLE_STATUSES = {"success"}
COLLECTION_COLUMNS = [
    "collection_status",
    "document_format",
    "raw_document_path",
    "raw_document_sha256",
    "raw_document_bytes",
    "pdf_path",
    "pdf_sha256",
    "pdf_bytes",
    "page_count",
    "text_status",
    "text_path",
    "text_sha256",
    "text_characters",
    "extraction_engine",
    "source_text_encoding",
    "http_content_type",
    "http_etag",
    "http_last_modified",
    "collected_at",
    "error",
]
DOCUMENT_INDEX_COLUMNS = [*selector.SELECTION_COLUMNS, *COLLECTION_COLUMNS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    payload = frame.to_csv(
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    ).encode("utf-8-sig")
    _atomic_bytes(payload, path)


def _content_snapshot(path: Path) -> Path:
    digest = _sha256(path)
    snapshot = LINEAGE_DIR / f"{digest}{path.suffix.lower()}"
    if not snapshot.is_file():
        _atomic_bytes(path.read_bytes(), snapshot)
    if _sha256(snapshot) != digest:
        raise ValueError(f"ETF benchmark document lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _meta_path(document_key: str) -> Path:
    if len(document_key) != 64 or any(character not in "0123456789abcdef" for character in document_key.lower()):
        raise ValueError(f"invalid ETF benchmark document key: {document_key}")
    return DOCUMENT_META_DIR / f"{document_key.lower()}.json"


def _valid_document_cache(row: Any) -> dict[str, Any] | None:
    document_key = str(row.document_key)
    meta_path = _meta_path(document_key)
    if not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("collection_status") == "success"
            and metadata.get("document_key") == document_key
            and metadata.get("asset") == str(row.asset)
            and metadata.get("source_url") == str(row.source_url)
        ):
            return None
        producer_path = ROOT / str(metadata.get("producer_code_path", ""))
        producer_hash = str(metadata.get("producer_code_sha256", ""))
        source_code_archive.authenticate_current_or_archive(producer_path, producer_hash)
        document_format = str(metadata.get("document_format") or "pdf").lower()
        if document_format not in {"pdf", "html"}:
            return None
        raw_path_value = str(metadata.get("raw_document_path") or metadata.get("pdf_path", ""))
        raw_hash_value = str(metadata.get("raw_document_sha256") or metadata.get("pdf_sha256", ""))
        raw_path = ROOT / raw_path_value
        if not raw_path.is_file() or _sha256(raw_path) != raw_hash_value:
            return None
        if document_format == "pdf":
            pdf_path = ROOT / str(metadata.get("pdf_path", ""))
            if not pdf_path.is_file() or _sha256(pdf_path) != str(metadata.get("pdf_sha256", "")):
                return None
        if metadata.get("text_status") not in {"success", "no_extractable_text"}:
            return None
        if metadata.get("text_status") in TEXT_AVAILABLE_STATUSES:
            text_path = ROOT / str(metadata.get("text_path", ""))
            if not text_path.is_file() or _sha256(text_path) != str(metadata.get("text_sha256", "")):
                return None
        return {
            **metadata,
            "document_format": document_format,
            "raw_document_path": raw_path_value,
            "raw_document_sha256": raw_hash_value,
            "raw_document_bytes": int(metadata.get("raw_document_bytes") or metadata.get("pdf_bytes", 0)),
            "source_text_encoding": str(metadata.get("source_text_encoding", "")),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _session_for_exchange(exchange: str) -> requests.Session:
    referer = (
        "https://www.sse.com.cn/disclosure/fund/announcement/index.shtml"
        if exchange == "SSE"
        else "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
    )
    return official._session(referer)


def _extract_html_text(payload: bytes) -> tuple[str, str]:
    declared = re.search(br"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", payload[:8192], flags=re.IGNORECASE)
    decoded = ""
    encoding = ""
    if declared:
        encoding = declared.group(1).decode("ascii", errors="ignore")
        try:
            decoded = payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            decoded = ""
            encoding = ""
    if not decoded:
        match = from_bytes(payload).best()
        if match is None or not match.encoding:
            raise ValueError("official ETF benchmark HTML encoding cannot be detected")
        decoded = str(match)
        encoding = str(match.encoding)
    soup = BeautifulSoup(decoded, "lxml")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    lines = []
    for line in soup.get_text("\n").splitlines():
        cleaned = re.sub(r"[ \t\u3000]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines), encoding


def collect_document(session: requests.Session, row: Any) -> dict[str, Any]:
    source_url = str(row.source_url)
    hostname = (urlparse(source_url).hostname or "").lower()
    if hostname not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported official ETF benchmark document domain: {source_url}")
    response = session.get(source_url, timeout=90)
    response.raise_for_status()
    payload = response.content
    content_type = str(response.headers.get("Content-Type", ""))
    if payload.lstrip().startswith(b"%PDF"):
        document_format = "pdf"
        raw_hash = _sha256_bytes(payload)
        raw_path = DOCUMENT_DIR / f"{raw_hash}.pdf"
        if not raw_path.is_file():
            _atomic_bytes(payload, raw_path)
        if _sha256(raw_path) != raw_hash:
            raise ValueError(f"content-addressed ETF benchmark PDF hash mismatch: {raw_path}")
        try:
            with pymupdf.open(stream=payload, filetype="pdf") as document:
                page_count = int(document.page_count)
                text = "\n".join(page.get_text("text", sort=True) or "" for page in document)
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise ValueError(f"official ETF benchmark PDF cannot be parsed: {type(exc).__name__}: {exc}") from exc
        if page_count <= 0:
            raise ValueError(f"official ETF benchmark PDF contains no pages: {source_url}")
        extraction_engine = f"PyMuPDF {pymupdf.__version__}"
        source_text_encoding = ""
        pdf_path_value = _relative(raw_path)
        pdf_hash_value = raw_hash
        pdf_bytes = len(payload)
    elif "html" in content_type.lower() or payload.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        document_format = "html"
        raw_hash = _sha256_bytes(payload)
        raw_path = DOCUMENT_DIR / f"{raw_hash}.html"
        if not raw_path.is_file():
            _atomic_bytes(payload, raw_path)
        if _sha256(raw_path) != raw_hash:
            raise ValueError(f"content-addressed ETF benchmark HTML hash mismatch: {raw_path}")
        text, source_text_encoding = _extract_html_text(payload)
        page_count = 0
        extraction_engine = "BeautifulSoup 4 + lxml + charset-normalizer"
        pdf_path_value = ""
        pdf_hash_value = ""
        pdf_bytes = 0
    else:
        raise ValueError(f"official ETF benchmark document format is unsupported: {source_url}")
    if text.strip():
        text_payload = text.encode("utf-8")
        text_hash = _sha256_bytes(text_payload)
        text_path = TEXT_DIR / f"{text_hash}.txt"
        if not text_path.is_file():
            _atomic_bytes(text_payload, text_path)
        text_status = "success"
        text_path_value = _relative(text_path)
        text_hash_value = text_hash
    else:
        text_status = "no_extractable_text"
        text_path_value = ""
        text_hash_value = ""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "document_key": str(row.document_key),
        "asset": str(row.asset),
        "source_url": source_url,
        "producer_code_path": _relative(Path(__file__).resolve()),
        "producer_code_sha256": _sha256(Path(__file__).resolve()),
        "collection_status": "success",
        "document_format": document_format,
        "raw_document_path": _relative(raw_path),
        "raw_document_sha256": raw_hash,
        "raw_document_bytes": len(payload),
        "pdf_path": pdf_path_value,
        "pdf_sha256": pdf_hash_value,
        "pdf_bytes": pdf_bytes,
        "page_count": page_count,
        "text_status": text_status,
        "text_path": text_path_value,
        "text_sha256": text_hash_value,
        "text_characters": len(text),
        "extraction_engine": extraction_engine,
        "source_text_encoding": source_text_encoding,
        "http_content_type": content_type,
        "http_etag": str(response.headers.get("ETag", "")),
        "http_last_modified": str(response.headers.get("Last-Modified", "")),
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": "" if text_status == "success" else f"{document_format.upper()} has no extractable text; review required.",
    }
    _atomic_json(metadata, _meta_path(str(row.document_key)))
    return metadata


def _unavailable_metadata(row: Any, *, status: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document_key": str(row.document_key),
        "asset": str(row.asset),
        "source_url": str(row.source_url),
        "producer_code_path": _relative(Path(__file__).resolve()),
        "producer_code_sha256": _sha256(Path(__file__).resolve()),
        "collection_status": status,
        "document_format": "",
        "raw_document_path": "",
        "raw_document_sha256": "",
        "raw_document_bytes": 0,
        "pdf_path": "",
        "pdf_sha256": "",
        "pdf_bytes": 0,
        "page_count": 0,
        "text_status": "not_available",
        "text_path": "",
        "text_sha256": "",
        "text_characters": 0,
        "extraction_engine": f"PyMuPDF {pymupdf.__version__}",
        "source_text_encoding": "",
        "http_content_type": "",
        "http_etag": "",
        "http_last_modified": "",
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": error,
    }


def _validate_selection(selection: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(set(selector.SELECTION_COLUMNS).difference(selection.columns))
    if missing:
        raise ValueError(f"ETF benchmark document selection misses columns: {missing}")
    frame = selection.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    if frame["document_key"].duplicated().any():
        raise ValueError("ETF benchmark document selection has duplicate document keys")
    if not frame["selection_policy_version"].eq(selector.SELECTION_POLICY_VERSION).all():
        raise ValueError("ETF benchmark document selection policy version mismatch")
    if frame["historical_backtest_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("ETF benchmark selection unexpectedly authorizes historical use")
    if frame["model_promotion_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("ETF benchmark selection unexpectedly authorizes model promotion")
    if int(manifest.get("selected_documents", 0)) != len(frame):
        raise ValueError("ETF benchmark selector manifest row count mismatch")
    baseline = frame["baseline_selection_state"].ne("not_baseline")
    if not baseline.groupby(frame["asset"]).sum().eq(1).all():
        raise ValueError("ETF benchmark selection does not retain exactly one baseline per asset")
    unsupported = sorted(
        {
            (urlparse(str(url)).hostname or "").lower()
            for url in frame["source_url"]
            if (urlparse(str(url)).hostname or "").lower() not in SUPPORTED_DOMAINS
        }
    )
    if unsupported:
        raise ValueError(f"ETF benchmark selection contains unsupported domains: {unsupported}")
    return frame.sort_values(
        ["selection_priority_rank", "asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def _authenticate_selection() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(SELECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "FULL_AUTHENTICATED_DOCUMENT_ROUTING_COLLECTION_REQUIRED"
        or int(manifest.get("target_assets", 0)) != 1701
        or manifest.get("selection_policy_version") != selector.SELECTION_POLICY_VERSION
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF benchmark selector does not authorize full document collection")
    producer_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        producer_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    item = outputs.get("benchmark_document_selection", {})
    selection_path = ROOT / str(item.get("path", ""))
    if not selection_path.is_file() or _sha256(selection_path) != str(item.get("sha256", "")):
        raise ValueError("ETF benchmark selector output hash mismatch")
    manifest_snapshot = _content_snapshot(SELECTOR_MANIFEST_PATH)
    selection_snapshot = _content_snapshot(selection_path)
    inputs = [
        {"role": "selector_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)},
        {"role": "selector_output_snapshot", "path": _relative(selection_snapshot), "sha256": _sha256(selection_snapshot)},
        {"role": "authenticated_selector_code", "path": _relative(authenticated_code), "sha256": _sha256(authenticated_code)},
    ]
    selection = pd.read_csv(selection_snapshot, dtype={"asset": str}, low_memory=False)
    return _validate_selection(selection, manifest), inputs


def _selection_reason_values(value: Any) -> set[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("ETF benchmark selection contains invalid selection reasons JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("ETF benchmark selection reasons must be a JSON list")
    return {str(item) for item in parsed if str(item)}


def _filter_collection_scope(
    selection: pd.DataFrame,
    *,
    priorities: list[str] | None = None,
    assets: list[str] | None = None,
    selection_reasons: list[str] | None = None,
) -> pd.DataFrame:
    scope = selection.copy()
    if priorities:
        requested_priorities = {str(priority).upper() for priority in priorities}
        unknown = requested_priorities.difference({"P0", "P1"})
        if unknown:
            raise ValueError(f"unsupported ETF benchmark document priorities: {sorted(unknown)}")
        scope = scope[scope["selection_priority"].isin(requested_priorities)].copy()
    if assets:
        requested_assets = {str(asset).zfill(6) for asset in assets}
        unknown = requested_assets.difference(set(selection["asset"]))
        if unknown:
            raise ValueError(f"requested assets are outside ETF benchmark selection: {sorted(unknown)}")
        scope = scope[scope["asset"].isin(requested_assets)].copy()
    if selection_reasons:
        requested_reasons = {str(reason).strip() for reason in selection_reasons if str(reason).strip()}
        available_reasons = {
            reason
            for value in selection["selection_reasons_json"]
            for reason in _selection_reason_values(value)
        }
        unknown = requested_reasons.difference(available_reasons)
        if unknown:
            raise ValueError(f"requested selection reasons are outside ETF benchmark selection: {sorted(unknown)}")
        scope = scope[
            scope["selection_reasons_json"].map(
                lambda value: bool(_selection_reason_values(value).intersection(requested_reasons))
            )
        ].copy()
    return scope


def run_collection(
    *,
    priorities: list[str] | None = None,
    assets: list[str] | None = None,
    selection_reasons: list[str] | None = None,
    max_documents: int | None = None,
    sleep_seconds: float = 0.05,
    max_consecutive_failures: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    selection, inputs = _authenticate_selection()
    scope = _filter_collection_scope(
        selection,
        priorities=priorities,
        assets=assets,
        selection_reasons=selection_reasons,
    )

    pending = [row for row in scope.itertuples(index=False) if force or _valid_document_cache(row) is None]
    selected_pending = pending[:max_documents] if max_documents is not None else pending
    cached_at_start = int(sum(_valid_document_cache(row) is not None for row in selection.itertuples(index=False)))
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_documents": int(len(selection)),
        "scope_documents": int(len(scope)),
        "cached_documents_at_start": cached_at_start,
        "pending_documents_in_scope_at_start": int(len(pending)),
        "attempted_documents": int(len(selected_pending)),
        "requested_priorities": sorted({str(priority).upper() for priority in priorities or []}),
        "requested_assets": sorted({str(asset).zfill(6) for asset in assets or []}),
        "requested_selection_reasons": sorted(
            {str(reason).strip() for reason in selection_reasons or [] if str(reason).strip()}
        ),
        "documents": {},
    }
    sessions = {
        exchange: _session_for_exchange(exchange)
        for exchange in sorted(set(selected_pending_row.exchange for selected_pending_row in selected_pending))
    }
    consecutive_failures = 0
    for position, row in enumerate(selected_pending, start=1):
        previous_metadata = _valid_document_cache(row)
        try:
            metadata = collect_document(sessions[str(row.exchange)], row)
            status["documents"][str(row.document_key)] = metadata
            consecutive_failures = 0
        except (OSError, TypeError, ValueError, requests.RequestException) as exc:
            failure = _unavailable_metadata(
                row,
                status="failed",
                error=f"{type(exc).__name__}: {str(exc)[:800]}",
            )
            if previous_metadata is None:
                _atomic_json(failure, _meta_path(str(row.document_key)))
                metadata = failure
            else:
                metadata = previous_metadata
            status["documents"][str(row.document_key)] = {
                **metadata,
                "latest_attempt_status": "failed",
                "latest_attempt_error": failure["error"],
            }
            consecutive_failures += 1
        status["documents_processed"] = position
        _atomic_json(status, STATUS_PATH)
        if consecutive_failures >= max_consecutive_failures:
            status["circuit_breaker"] = f"stopped_after_{consecutive_failures}_consecutive_failures"
            _atomic_json(status, STATUS_PATH)
            break
        if sleep_seconds > 0 and position < len(selected_pending):
            time.sleep(sleep_seconds)

    index_rows: list[dict[str, Any]] = []
    document_inputs: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        metadata = _valid_document_cache(row)
        meta_path = _meta_path(str(row.document_key))
        if metadata is None and meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                metadata = None
        if metadata is None:
            metadata = _unavailable_metadata(row, status="pending", error="Document collection has not completed.")
        base = {column: getattr(row, column) for column in selector.SELECTION_COLUMNS}
        index_rows.append({**base, **{column: metadata.get(column, "") for column in COLLECTION_COLUMNS}})
        role_suffix = str(row.document_key)[:16]
        if meta_path.is_file():
            document_inputs.append(
                {"role": f"document_metadata:{row.asset}:{role_suffix}", "path": _relative(meta_path), "sha256": _sha256(meta_path)}
            )
        if metadata.get("collection_status") == "success":
            raw_document_path = ROOT / str(metadata["raw_document_path"])
            document_inputs.append(
                {
                    "role": f"official_{metadata['document_format']}:{row.asset}:{role_suffix}",
                    "path": _relative(raw_document_path),
                    "sha256": _sha256(raw_document_path),
                }
            )
            if metadata.get("text_status") in TEXT_AVAILABLE_STATUSES:
                text_path = ROOT / str(metadata["text_path"])
                document_inputs.append(
                    {"role": f"official_text:{row.asset}:{role_suffix}", "path": _relative(text_path), "sha256": _sha256(text_path)}
                )

    document_index = pd.DataFrame(index_rows).reindex(columns=DOCUMENT_INDEX_COLUMNS).sort_values(
        ["selection_priority_rank", "asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)
    review_queue = document_index[
        document_index["collection_status"].ne("success") | document_index["text_status"].ne("success")
    ].copy()
    _atomic_csv(document_index, DOCUMENT_INDEX_PATH)
    _atomic_csv(review_queue, REVIEW_QUEUE_PATH)

    collected = document_index["collection_status"].eq("success")
    native_text = document_index["text_status"].eq("success")
    no_text = document_index["text_status"].eq("no_extractable_text")
    pdf_documents = collected & document_index["document_format"].eq("pdf")
    html_documents = collected & document_index["document_format"].eq("html")
    p0 = document_index["selection_priority"].eq("P0")
    p1 = document_index["selection_priority"].eq("P1")
    if collected.all():
        qualification = (
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_COMPLETE"
            if native_text.all()
            else "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_OCR_REQUIRED"
        )
    else:
        qualification = "PARTIAL_SELECTED_OFFICIAL_DOCUMENT_SET"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "selected_documents": int(len(document_index)),
        "collected_documents": int(collected.sum()),
        "native_text_documents": int(native_text.sum()),
        "no_extractable_text_documents": int(no_text.sum()),
        "pdf_documents": int(pdf_documents.sum()),
        "html_documents": int(html_documents.sum()),
        "failed_documents": int(document_index["collection_status"].eq("failed").sum()),
        "pending_documents": int(document_index["collection_status"].eq("pending").sum()),
        "p0_selected_documents": int(p0.sum()),
        "p0_collected_documents": int((p0 & collected).sum()),
        "p1_selected_documents": int(p1.sum()),
        "p1_collected_documents": int((p1 & collected).sum()),
        "assets_with_collected_baseline": int(
            document_index[collected & document_index["baseline_selection_state"].ne("not_baseline")]["asset"].nunique()
        ),
        "document_collection_complete": bool(collected.all()),
        "document_content_validation_complete": False,
        "formal_history_rows": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Official PDFs or archived HTML and extracted text only. Collection success does not prove benchmark facts, "
            "effective dates, complete change history, or absence of changes."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "benchmark_document_index", "path": _relative(DOCUMENT_INDEX_PATH), "sha256": _sha256(DOCUMENT_INDEX_PATH), "rows": int(len(document_index))},
        {"role": "benchmark_document_collection_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review_queue))},
        {"role": "collection_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        **report,
        "inputs": [*inputs, *document_inputs],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(selector.__file__).resolve()), "sha256": _sha256(Path(selector.__file__).resolve())},
            {"path": _relative(Path(official.__file__).resolve()), "sha256": _sha256(Path(official.__file__).resolve())},
            {"path": _relative(Path(source_code_archive.__file__).resolve()), "sha256": _sha256(Path(source_code_archive.__file__).resolve())},
        ],
        "current_final_snapshot": True,
        "contains_benchmark_facts": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(report)
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", action="append", choices=("P0", "P1"))
    parser.add_argument("--asset", action="append")
    parser.add_argument("--selection-reason", action="append")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_collection(
        priorities=args.priority,
        assets=args.asset,
        selection_reasons=args.selection_reason,
        max_documents=args.max_documents,
        sleep_seconds=args.sleep_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        force=args.force,
    )
    keys = (
        "qualification_status",
        "selected_documents",
        "collected_documents",
        "native_text_documents",
        "failed_documents",
        "pending_documents",
        "p0_collected_documents",
        "p1_collected_documents",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
