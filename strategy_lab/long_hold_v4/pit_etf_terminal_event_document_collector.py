"""Download and hash the official document chain for delisted ETF terminal events."""

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
from urllib.parse import urlparse

import pandas as pd
import pdfplumber
import requests
from pdfminer.pdfparser import PDFSyntaxError

from . import pit_etf_dividend_announcement_collector as official
from . import pit_etf_terminal_event_universe_collector as universe


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_universe_collector_latest.json"
)
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_terminal_event_universe"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
DOCUMENT_META_DIR = RAW_DIR / "document_metadata"
STATUS_PATH = RAW_DIR / "document_collection_status.json"
LINEAGE_DIR = RAW_DIR / "document_lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
SELECTION_PATH = OBSERVATION_DIR / "etf_terminal_event_document_selection.csv"
DOCUMENT_INDEX_PATH = OBSERVATION_DIR / "etf_terminal_event_document_index.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_terminal_event_document_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_documents" / "collection_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_document_collector_latest.json"
)

SCHEMA_VERSION = 2
TEXT_AVAILABLE_STATUSES = {"success"}
HIGH_VALUE_PATTERN = re.compile(
    r"清算报告|剩余财产|清算资金|清算款|处置.*资金|财产分配|追加分配|二次分配|补充分配|"
    r"份额转换|变更登记|确权|转型|合并生效|换份额"
)
SUCCESSOR_PATTERN = re.compile(r"份额转换|变更登记|确权|转型|变更注册|合并生效|换份额")
PROMPT_PATTERN = re.compile(r"提示性|第一次提示|第二次提示|第三次提示|风险提示")
SELECTION_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "master_delist_date",
    "announcement_date",
    "announcement_title",
    "source_url",
    "source_type",
    "title_tags_json",
    "candidate_event_types_json",
    "selection_reasons_json",
    "document_role_candidates_json",
    "published_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "historical_backtest_allowed",
]
DOCUMENT_INDEX_COLUMNS = [
    *SELECTION_COLUMNS,
    "collection_status",
    "pdf_path",
    "pdf_sha256",
    "pdf_bytes",
    "page_count",
    "text_status",
    "text_path",
    "text_sha256",
    "text_characters",
    "source_observed_at",
    "error",
    "document_validation_status",
    "model_promotion_allowed",
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


def _content_snapshot(path: Path) -> Path:
    digest = _sha256(path)
    snapshot = LINEAGE_DIR / f"{digest}{path.suffix.lower()}"
    if not snapshot.is_file():
        _atomic_bytes(path.read_bytes(), snapshot)
    if _sha256(snapshot) != digest:
        raise ValueError(f"terminal document lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _json_values(value: Any) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, received {type(parsed).__name__}")
    return {str(item) for item in parsed}


def _json_list(values: set[str] | list[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False)


def _role_candidates(title: str, tags: set[str]) -> set[str]:
    roles: set[str] = set()
    if "清算报告" in title:
        roles.add("liquidation_report")
    if any(marker in title for marker in ("剩余财产", "清算资金", "清算款", "财产分配", "资金发放")):
        roles.add("cash_distribution")
    if "fund_contract_termination" in tags or "liquidation" in tags:
        roles.add("termination_mechanism")
    if "delisting" in tags:
        roles.add("exchange_delisting")
    if "transformation" in tags or "successor_share_conversion" in tags:
        roles.add("successor_share_registration")
    if "fund_merger" in tags:
        roles.add("merger_or_share_exchange")
    if "holder_resolution" in tags:
        roles.add("holder_resolution")
    return roles


def _best_non_prompt(frame: pd.DataFrame) -> int:
    if frame.empty:
        raise ValueError("cannot select from an empty document frame")
    ranked = frame.assign(
        _is_prompt=frame["announcement_title"].astype(str).str.contains(PROMPT_PATTERN, regex=True),
        _date=pd.to_datetime(frame["announcement_date"], errors="coerce"),
    ).sort_values(["_is_prompt", "_date", "source_url"], ascending=[True, False, True])
    return int(ranked.index[0])


def _is_mechanism_candidate(row: pd.Series) -> bool:
    tags = _json_values(row["title_tags_json"])
    primary = str(row["primary_candidate_class"])
    title = str(row["announcement_title"])
    if primary == "cash_or_extinguishment_candidate":
        return "fund_contract_termination" in tags or (
            "liquidation" in tags and not HIGH_VALUE_PATTERN.search(title)
        )
    if primary == "successor_share_candidate":
        return bool(tags.intersection({"transformation", "successor_share_conversion"}))
    if primary == "merger_or_share_exchange_candidate":
        return "fund_merger" in tags
    return bool(
        tags.intersection(
            {
                "fund_contract_termination",
                "liquidation",
                "transformation",
                "successor_share_conversion",
                "fund_merger",
            }
        )
    ) and not HIGH_VALUE_PATTERN.search(title)


def select_documents(announcements: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    required_announcements = set(universe.ANNOUNCEMENT_COLUMNS)
    required_coverage = {"asset", "master_delist_date", "primary_candidate_class"}
    missing = sorted(required_announcements.difference(announcements.columns))
    if missing:
        raise ValueError(f"terminal announcement inventory misses columns: {missing}")
    missing = sorted(required_coverage.difference(coverage.columns))
    if missing:
        raise ValueError(f"terminal coverage registry misses columns: {missing}")
    frame = announcements.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"], errors="coerce").dt.normalize()
    coverage_dates = coverage[["asset", "master_delist_date", "primary_candidate_class"]].copy()
    coverage_dates["asset"] = coverage_dates["asset"].astype(str).str.zfill(6)
    coverage_dates["master_delist_date"] = pd.to_datetime(
        coverage_dates["master_delist_date"], errors="coerce"
    ).dt.normalize()
    frame = frame.merge(coverage_dates, on="asset", how="left", validate="many_to_one")
    if frame[["announcement_date", "master_delist_date"]].isna().any().any():
        raise ValueError("terminal document selection has invalid announcement or delisting dates")

    reasons: dict[int, set[str]] = {int(index): set() for index in frame.index}
    for index, row in frame.iterrows():
        title = str(row["announcement_title"])
        tags = _json_values(row["title_tags_json"])
        if HIGH_VALUE_PATTERN.search(title):
            reasons[int(index)].add("all_high_value_terminal_documents")
        if "holder_resolution" in tags:
            reasons[int(index)].add("all_holder_resolutions")
        successor_window = (
            row["primary_candidate_class"] == "successor_share_candidate"
            and abs((row["announcement_date"] - row["master_delist_date"]).days) <= 370
            and bool(SUCCESSOR_PATTERN.search(title))
        )
        if successor_window:
            reasons[int(index)].add("all_successor_documents_in_terminal_window")

    for asset, selected in frame.groupby("asset", sort=True):
        mechanism = selected[selected.apply(_is_mechanism_candidate, axis=1)]
        if not mechanism.empty:
            reasons[_best_non_prompt(mechanism)].add("one_canonical_terminal_mechanism_document")
        delisting = selected[
            selected["title_tags_json"].map(lambda value: "delisting" in _json_values(value))
        ]
        if not delisting.empty:
            reasons[_best_non_prompt(delisting)].add("one_canonical_delisting_document")

    selected_indices = [index for index, values in reasons.items() if values]
    selected = frame.loc[selected_indices].copy()
    if set(coverage["asset"].astype(str).str.zfill(6)).difference(set(selected["asset"])):
        missing_assets = sorted(set(coverage["asset"].astype(str).str.zfill(6)).difference(set(selected["asset"])))
        raise ValueError(f"terminal document policy selected no evidence for assets: {missing_assets}")
    selected["selection_reasons_json"] = [_json_list(reasons[int(index)]) for index in selected.index]
    selected["document_role_candidates_json"] = [
        _json_list(_role_candidates(str(row.announcement_title), _json_values(row.title_tags_json)))
        for row in selected.itertuples(index=False)
    ]
    selected["historical_backtest_allowed"] = False
    return selected.reindex(columns=SELECTION_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def _meta_path(asset: str, source_url: str) -> Path:
    url_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
    return DOCUMENT_META_DIR / f"{asset}_{url_hash}.json"


def _valid_document_cache(asset: str, source_url: str) -> dict[str, Any] | None:
    meta_path = _meta_path(asset, source_url)
    if not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("collection_status") == "success"
        and metadata.get("asset") == asset
        and metadata.get("source_url") == source_url
        and metadata.get("producer_code_sha256") == _sha256(Path(__file__).resolve())
    ):
        return None
    pdf_path = ROOT / str(metadata.get("pdf_path", ""))
    if not pdf_path.is_file() or _sha256(pdf_path) != str(metadata.get("pdf_sha256", "")):
        return None
    if metadata.get("text_status") not in {"success", "no_extractable_text"}:
        return None
    if metadata.get("text_status") in TEXT_AVAILABLE_STATUSES:
        text_path = ROOT / str(metadata.get("text_path", ""))
        if not text_path.is_file() or _sha256(text_path) != str(metadata.get("text_sha256", "")):
            return None
    return metadata


def _session_for_exchange(exchange: str) -> requests.Session:
    referer = (
        "https://www.sse.com.cn/disclosure/fund/announcement/index.shtml"
        if exchange == "SSE"
        else "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
    )
    return official._session(referer)


def collect_document(session: requests.Session, row: Any) -> dict[str, Any]:
    source_url = str(row.source_url)
    hostname = (urlparse(source_url).hostname or "").lower()
    if hostname not in {"www.sse.com.cn", "static.cninfo.com.cn"}:
        raise ValueError(f"unsupported official document domain: {source_url}")
    response = session.get(source_url, timeout=90)
    response.raise_for_status()
    payload = response.content
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"official terminal document is not a PDF: {source_url}")
    pdf_hash = _sha256_bytes(payload)
    pdf_path = DOCUMENT_DIR / f"{pdf_hash}.pdf"
    _atomic_bytes(payload, pdf_path)
    try:
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            page_count = len(pdf.pages)
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except (PDFSyntaxError, ValueError) as exc:
        raise ValueError(f"official terminal PDF cannot be parsed: {type(exc).__name__}: {exc}") from exc
    if text.strip():
        text_payload = text.encode("utf-8")
        text_hash = _sha256_bytes(text_payload)
        text_path = TEXT_DIR / f"{pdf_hash}.txt"
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
        "asset": str(row.asset),
        "source_url": source_url,
        "producer_code_path": _relative(Path(__file__).resolve()),
        "producer_code_sha256": _sha256(Path(__file__).resolve()),
        "collection_status": "success",
        "pdf_path": _relative(pdf_path),
        "pdf_sha256": pdf_hash,
        "pdf_bytes": len(payload),
        "page_count": page_count,
        "text_status": text_status,
        "text_path": text_path_value,
        "text_sha256": text_hash_value,
        "text_characters": len(text),
        "source_observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": "" if text_status == "success" else "PDF has no extractable text; OCR review required.",
    }
    _atomic_json(metadata, _meta_path(str(row.asset), source_url))
    return metadata


def _failed_metadata(row: Any, error: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "asset": str(row.asset),
        "source_url": str(row.source_url),
        "producer_code_path": _relative(Path(__file__).resolve()),
        "producer_code_sha256": _sha256(Path(__file__).resolve()),
        "collection_status": "failed",
        "pdf_path": "",
        "pdf_sha256": "",
        "pdf_bytes": 0,
        "page_count": 0,
        "text_status": "not_available",
        "text_path": "",
        "text_sha256": "",
        "text_characters": 0,
        "source_observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": error,
    }


def _authenticate_discovery() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(DISCOVERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        != "FULL_RELATIVE_TO_AUTHENTICATED_MASTER_OFFICIAL_DISCOVERY_REQUIRES_DOCUMENT_VALIDATION"
        or int(manifest.get("target_assets", 0)) != 123
        or int(manifest.get("query_complete_assets", 0)) != 123
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("terminal-event discovery does not authorize full-universe document collection")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal-event discovery code hash mismatch")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    paths: dict[str, Path] = {}
    manifest_snapshot = _content_snapshot(DISCOVERY_MANIFEST_PATH)
    inputs = [
        {"role": "discovery_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)}
    ]
    for role in ("official_announcements", "coverage_registry"):
        item = outputs.get(role, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"terminal-event discovery output hash mismatch: {role}")
        snapshot = _content_snapshot(path)
        paths[role] = snapshot
        inputs.append({"role": f"{role}_snapshot", "path": _relative(snapshot), "sha256": _sha256(snapshot)})
    announcements = pd.read_csv(paths["official_announcements"], dtype={"asset": str})
    coverage = pd.read_csv(paths["coverage_registry"], dtype={"asset": str})
    return announcements, coverage, inputs


def run_collection(
    *,
    assets: list[str] | None = None,
    max_documents: int | None = None,
    sleep_seconds: float = 0.05,
    max_consecutive_failures: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    announcements, coverage, inputs = _authenticate_discovery()
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        unknown = requested.difference(set(coverage["asset"].astype(str).str.zfill(6)))
        if unknown:
            raise ValueError(f"requested assets are outside terminal-event coverage: {sorted(unknown)}")
        announcements = announcements[announcements["asset"].astype(str).str.zfill(6).isin(requested)].copy()
        coverage = coverage[coverage["asset"].astype(str).str.zfill(6).isin(requested)].copy()
    selection = select_documents(announcements, coverage)
    _atomic_csv(selection, SELECTION_PATH)
    pending = [
        row
        for row in selection.itertuples(index=False)
        if force or _valid_document_cache(str(row.asset), str(row.source_url)) is None
    ]
    selected_pending = pending[:max_documents] if max_documents is not None else pending
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_documents": int(len(selection)),
        "pending_documents_at_start": int(len(pending)),
        "attempted_documents": int(len(selected_pending)),
        "documents": {},
    }
    sessions = {"SSE": _session_for_exchange("SSE"), "SZSE": _session_for_exchange("SZSE")}
    consecutive_failures = 0
    for position, row in enumerate(selected_pending, start=1):
        key = f"{row.asset}:{hashlib.sha256(str(row.source_url).encode()).hexdigest()[:16]}"
        previous_metadata = _valid_document_cache(str(row.asset), str(row.source_url))
        try:
            metadata = collect_document(sessions[str(row.exchange)], row)
            status["documents"][key] = metadata
            consecutive_failures = 0
        except (OSError, ValueError, requests.RequestException) as exc:
            failure = _failed_metadata(row, f"{type(exc).__name__}: {str(exc)[:600]}")
            if previous_metadata is None:
                _atomic_json(failure, _meta_path(str(row.asset), str(row.source_url)))
                metadata = failure
            else:
                metadata = previous_metadata
            status["documents"][key] = {
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
        metadata = _valid_document_cache(str(row.asset), str(row.source_url))
        if metadata is None:
            meta_path = _meta_path(str(row.asset), str(row.source_url))
            if meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    metadata = None
        if metadata is None:
            metadata = _failed_metadata(row, "Document collection has not completed.")
            metadata["collection_status"] = "pending"
        base = {column: getattr(row, column) for column in SELECTION_COLUMNS}
        index_rows.append(
            {
                **base,
                **metadata,
                "document_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
        meta_path = _meta_path(str(row.asset), str(row.source_url))
        role_suffix = hashlib.sha256(str(row.source_url).encode("utf-8")).hexdigest()[:16]
        if meta_path.is_file():
            document_inputs.append(
                {"role": f"document_metadata:{row.asset}:{role_suffix}", "path": _relative(meta_path), "sha256": _sha256(meta_path)}
            )
        if metadata.get("collection_status") == "success":
            pdf_path = ROOT / str(metadata["pdf_path"])
            document_inputs.append(
                {"role": f"official_pdf:{row.asset}:{role_suffix}", "path": _relative(pdf_path), "sha256": _sha256(pdf_path)}
            )
            if metadata.get("text_status") in TEXT_AVAILABLE_STATUSES:
                text_path = ROOT / str(metadata["text_path"])
                document_inputs.append(
                    {"role": f"official_text:{row.asset}:{role_suffix}", "path": _relative(text_path), "sha256": _sha256(text_path)}
                )
    document_index = pd.DataFrame(index_rows).reindex(columns=DOCUMENT_INDEX_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)
    review = document_index[
        document_index["collection_status"].ne("success") | document_index["text_status"].ne("success")
    ].copy()
    _atomic_csv(document_index, DOCUMENT_INDEX_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)
    collected = int(document_index["collection_status"].eq("success").sum())
    text_success = int(document_index["text_status"].eq("success").sum())
    text_available = int(document_index["text_status"].isin(TEXT_AVAILABLE_STATUSES).sum())
    no_text = int(document_index["text_status"].eq("no_extractable_text").sum())
    qualification = (
        (
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_COMPLETE"
            if no_text == 0
            else "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_GAPS"
        )
        if collected == len(selection)
        else "PARTIAL_OFFICIAL_DOCUMENT_SET"
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "covered_assets": int(selection["asset"].nunique()),
        "selected_documents": int(len(selection)),
        "collected_documents": collected,
        "text_extraction_success": text_success,
        "text_available_documents": text_available,
        "no_extractable_text_documents": no_text,
        "failed_or_pending_documents": int(len(selection) - collected),
        "review_queue_rows": int(len(review)),
        "document_selection_policy": (
            "all high-value cash/liquidation/successor documents, all holder resolutions, all successor-window documents, "
            "one canonical mechanism document and one canonical delisting document per asset"
        ),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "Raw official PDFs and native extracted text only. OCR is an immutable sidecar and event economics require independent validation.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "document_selection", "path": _relative(SELECTION_PATH), "sha256": _sha256(SELECTION_PATH), "rows": int(len(selection))},
        {"role": "document_index", "path": _relative(DOCUMENT_INDEX_PATH), "sha256": _sha256(DOCUMENT_INDEX_PATH), "rows": int(len(document_index))},
        {"role": "document_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review))},
        {"role": "collection_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": [*inputs, *document_inputs],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(universe.__file__).resolve()), "sha256": _sha256(Path(universe.__file__).resolve())},
            {"path": _relative(Path(official.__file__).resolve()), "sha256": _sha256(Path(official.__file__).resolve())},
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(report)
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_collection(
        assets=args.asset,
        max_documents=args.max_documents,
        sleep_seconds=args.sleep_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        force=args.force,
    )
    keys = (
        "qualification_status",
        "covered_assets",
        "selected_documents",
        "collected_documents",
        "text_extraction_success",
        "failed_or_pending_documents",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
