"""Collect official evidence for known ETF terminal cash events.

This collector resolves identified lifecycle exceptions. It does not claim that
all delisted ETFs have been exhaustively searched for terminal distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import pdfplumber

from . import pit_etf_dividend_announcement_collector as official
from . import pit_etf_sse_share_action_announcement_collector as sse_api


ROOT = Path(__file__).resolve().parents[2]
ETF_MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_terminal_cash_events"
QUERY_PATH = RAW_DIR / "queries" / "511210.json"
DOCUMENT_DIR = RAW_DIR / "documents"
TEXT_DIR = RAW_DIR / "text"
CANDIDATE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "etf_terminal_cash_event_candidates.csv"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_cash_event_collector_latest.json"
)
PROVIDER_DIVIDEND_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_total_return"
    / "511210"
    / "dividend.csv.gz"
)

ASSET = "511210"
KEYWORD_ROLE = {
    "表决结果暨决议生效": "resolution",
    "清算": "liquidation_report",
    "剩余财产": "distribution",
    "终止上市": "delisting",
}
EXPECTED_TITLE_MARKER = {
    "resolution": "表决结果暨决议生效",
    "liquidation_report": "清算报告",
    "distribution": "剩余财产分配公告",
    "delisting": "终止上市的公告",
}
CANDIDATE_COLUMNS = [
    "asset",
    "event_type",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "last_operation_date",
    "liquidation_start_date",
    "liquidation_end_date",
    "liquidation_nav",
    "liquidation_net_assets",
    "liquidation_shares",
    "termination_date",
    "extinguishes_position",
    "available_date",
    "source_urls_json",
    "source_pdf_sha256_set",
    "source_text_sha256_set",
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


def _set_sha256(values: list[str]) -> str:
    material = json.dumps(sorted(set(values)), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _date(text: str, pattern: str, label: str) -> pd.Timestamp:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"terminal event document misses {label}")
    return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))


def _number(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"terminal event document misses {label}")
    return float(match.group(1).replace(",", ""))


def _number_after(text: str, marker: str, label: str, window: int = 160) -> float:
    position = text.find(marker)
    if position < 0:
        raise ValueError(f"terminal event document misses {label}")
    match = re.search(r"([0-9][0-9,.]+)", text[position + len(marker) : position + len(marker) + window])
    if not match:
        raise ValueError(f"terminal event document misses {label}")
    return float(match.group(1).replace(",", ""))


def parse_terminal_event(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Parse the governed 511210 liquidation chain from four official texts."""

    missing = sorted(set(EXPECTED_TITLE_MARKER).difference(documents))
    if missing:
        raise ValueError(f"terminal event evidence roles are missing: {missing}")
    compact = {role: _compact(str(item["text"])) for role, item in documents.items()}
    distribution = compact["distribution"]
    liquidation = compact["liquidation_report"]
    resolution = compact["resolution"]
    delisting = compact["delisting"]

    announcement_date = _date(distribution, r"公告送出日期：(\d{4})年(\d{1,2})月(\d{1,2})日", "announcement date")
    record_date = _date(distribution, r"权益登记日(\d{4})年(\d{1,2})月(\d{1,2})日", "record date")
    ex_date = _date(distribution, r"除息日(\d{4})年(\d{1,2})月(\d{1,2})日", "ex date")
    pay_date = _date(distribution, r"剩余财产发放日(\d{4})年(\d{1,2})月(\d{1,2})日", "pay date")
    cash_per_ten = _number(
        distribution,
        r"每10份基金份额分配([0-9,.]+)元人民币",
        "cash per ten shares",
    )
    liquidation_end = _date(
        distribution,
        r"剩余财产分配基准日(\d{4})年(\d{1,2})月(\d{1,2})日",
        "liquidation end date",
    )
    liquidation_nav = _number_after(distribution, "基准日基金份额净值", "liquidation NAV")
    liquidation_net_assets = _number_after(
        distribution,
        "基准日基金可供分配财产",
        "liquidation net assets",
    )
    last_operation = _date(
        liquidation,
        r"最后运作日定为(\d{4})年(\d{1,2})月(\d{1,2})日",
        "last operation date",
    )
    liquidation_start = _date(
        resolution,
        r"从(\d{4})年(\d{1,2})月(\d{1,2})日起进入清算期",
        "liquidation start date",
    )
    liquidation_shares = _number(
        liquidation,
        r"基金份额总额([0-9,.]+)份",
        "liquidation shares",
    )
    termination_date = _date(
        delisting,
        r"终止上市日：(\d{4})年(\d{1,2})月(\d{1,2})日",
        "termination date",
    )
    source_urls = {role: str(item["source_url"]) for role, item in documents.items()}
    pdf_hashes = [str(item["pdf_sha256"]) for item in documents.values()]
    text_hashes = [str(item["text_sha256"]) for item in documents.values()]
    return {
        "asset": ASSET,
        "event_type": "liquidation_distribution",
        "announcement_date": announcement_date,
        "record_date": record_date,
        "ex_date": ex_date,
        "pay_date": pay_date,
        "cash_per_share": cash_per_ten / 10.0,
        "last_operation_date": last_operation,
        "liquidation_start_date": liquidation_start,
        "liquidation_end_date": liquidation_end,
        "liquidation_nav": liquidation_nav,
        "liquidation_net_assets": liquidation_net_assets,
        "liquidation_shares": liquidation_shares,
        "termination_date": termination_date,
        "extinguishes_position": True,
        "available_date": announcement_date,
        "source_urls_json": json.dumps(source_urls, ensure_ascii=False, sort_keys=True),
        "source_pdf_sha256_set": _set_sha256(pdf_hashes),
        "source_text_sha256_set": _set_sha256(text_hashes),
        "historical_backtest_allowed": False,
    }


def _fetch_query(as_of: pd.Timestamp) -> tuple[dict[str, Any], Any]:
    session = official._session("https://www.sse.com.cn/disclosure/fund/announcement/index.shtml")
    responses = []
    for keyword, role in KEYWORD_ROLE.items():
        response = session.get(
            sse_api.QUERY_URL,
            params=sse_api._query_params(ASSET, keyword, as_of, 1),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("result", [])
        if not isinstance(rows, list):
            raise ValueError(f"SSE terminal query returned invalid rows for {keyword}")
        responses.append({"keyword": keyword, "role": role, "rows": rows})
    artifact = {
        "asset": ASSET,
        "as_of_date": as_of.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "responses": responses,
    }
    return artifact, session


def run(as_of: str | pd.Timestamp) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    query, session = _fetch_query(cutoff)
    _atomic_json(query, QUERY_PATH)

    documents: dict[str, dict[str, Any]] = {}
    for response in query["responses"]:
        role = str(response["role"])
        marker = EXPECTED_TITLE_MARKER[role]
        matches = [row for row in response["rows"] if marker in str(row.get("TITLE", ""))]
        if len(matches) != 1:
            raise ValueError(f"expected one official {role} document, found {len(matches)}")
        row = matches[0]
        source_url = urljoin("https://www.sse.com.cn", str(row["URL"]))
        fetched = session.get(source_url, timeout=60)
        fetched.raise_for_status()
        content = fetched.content
        if not content.startswith(b"%PDF"):
            raise ValueError(f"official terminal document is not a PDF: {source_url}")
        pdf_hash = _sha256_bytes(content)
        pdf_path = DOCUMENT_DIR / f"{pdf_hash}.pdf"
        _atomic_bytes(content, pdf_path)
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        if not text.strip():
            raise ValueError(f"official terminal PDF has no extractable text: {source_url}")
        text_bytes = text.encode("utf-8")
        text_hash = _sha256_bytes(text_bytes)
        text_path = TEXT_DIR / f"{pdf_hash}.txt"
        _atomic_bytes(text_bytes, text_path)
        documents[role] = {
            "role": role,
            "announcement_date": str(row["SSEDATE"]),
            "title": str(row["TITLE"]),
            "source_url": source_url,
            "pdf_path": _relative(pdf_path),
            "pdf_sha256": pdf_hash,
            "text_path": _relative(text_path),
            "text_sha256": text_hash,
            "text": text,
        }

    candidate = parse_terminal_event(documents)
    if max(candidate[column] for column in ("announcement_date", "record_date", "ex_date", "pay_date", "termination_date")) > cutoff:
        raise ValueError("terminal event evidence extends beyond the requested as-of date")
    candidate_frame = pd.DataFrame([candidate], columns=CANDIDATE_COLUMNS)
    _atomic_csv(candidate_frame, CANDIDATE_PATH)

    manifest_documents = [
        {key: value for key, value in document.items() if key != "text"}
        for document in sorted(documents.values(), key=lambda item: item["role"])
    ]
    outputs = [
        {"role": "query", "path": _relative(QUERY_PATH), "sha256": _sha256(QUERY_PATH)},
        {"role": "candidates", "path": _relative(CANDIDATE_PATH), "sha256": _sha256(CANDIDATE_PATH), "rows": 1},
    ]
    for document in manifest_documents:
        outputs.extend(
            [
                {"role": f"official_pdf:{document['role']}", "path": document["pdf_path"], "sha256": document["pdf_sha256"]},
                {"role": f"official_text:{document['role']}", "path": document["text_path"], "sha256": document["text_sha256"]},
            ]
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "asset": ASSET,
        "qualification_status": "KNOWN_TERMINAL_EVENT_OFFICIAL_EVIDENCE_REQUIRES_VALIDATION",
        "scope_complete": False,
        "scope_boundary": "Known 511210 exception only; not an exhaustive terminal-event search over all delisted ETFs.",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "inputs": [
            {"role": "etf_security_master", "path": _relative(ETF_MASTER_PATH), "sha256": _sha256(ETF_MASTER_PATH)},
            {"role": "provider_terminal_marker", "path": _relative(PROVIDER_DIVIDEND_PATH), "sha256": _sha256(PROVIDER_DIVIDEND_PATH)},
        ],
        "documents": manifest_documents,
        "outputs": outputs,
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args.as_of)
    print(json.dumps({key: manifest[key] for key in ("asset", "qualification_status", "scope_complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
