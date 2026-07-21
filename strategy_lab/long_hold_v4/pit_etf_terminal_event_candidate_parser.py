"""Parse observation-only field candidates from official ETF terminal-event texts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import pit_etf_terminal_event_document_merge as document_merge


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_document_merge_latest.json"
)
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
DOCUMENT_PARSE_PATH = OBSERVATION_DIR / "etf_terminal_event_document_parse_candidates.csv"
CASH_EVENT_PATH = OBSERVATION_DIR / "etf_terminal_event_cash_candidates.csv"
LIQUIDATION_PATH = OBSERVATION_DIR / "etf_terminal_event_liquidation_report_candidates.csv"
DELISTING_PATH = OBSERVATION_DIR / "etf_terminal_event_delisting_candidates.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_terminal_event_parse_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_parser" / "parse_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_candidate_parser_latest.json"
)

SCHEMA_VERSION = 2
DATE_TOKEN = r"(?P<year>20\d{2})[年./-](?P<month>\d{1,2})[月./-](?P<day>\d{1,2})日?"
CASH_CONTEXT_MARKERS = (
    "实际发放资金",
    "可获分配",
    "发放资金",
    "分配资金",
    "清算资金",
    "剩余财产",
    "实际分配",
    "分配",
)
DIRECT_CASH_PATTERN = re.compile(
    r"每(?P<denominator>\d+|十|百|千)?份(?P<context>[^。；]{0,140}?)(?:为|:|：)?(?:人民币)?"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)元"
)
DIRECT_CASH_ALTERNATE_PATTERNS = (
    (
        "unit_share_actual_distribution",
        re.compile(
            r"单位基金份额(?P<context>[^。；]{0,100}?(?:实际发放|分配|清算)[^。；]{0,40}?)(?:为|:|：)?"
            r"(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"
        ),
    ),
)
TOTAL_CASH_PATTERNS = (
    (
        "current_distributable_remaining_property",
        0,
        re.compile(r"本次(?:清算)?可供分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"),
    ),
    (
        "current_remaining_property_distribution",
        0,
        re.compile(r"本次(?:应)?分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"),
    ),
    (
        "remaining_property_to_distribute",
        1,
        re.compile(r"应分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"),
    ),
    (
        "remaining_property_total",
        2,
        re.compile(r"剩余财产(?:总额)?(?:为|共计)(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"),
    ),
    (
        "clearing_cash_total",
        2,
        re.compile(r"(?:本次)?(?:清算资金|资金清算)(?:发放)?(?:总额)?(?:为|共计)(?:人民币)?(?P<amount>\d[\d,]*(?:\.\d+)?)元"),
    ),
)
SHARE_PATTERN = re.compile(r"基金份额总额(?:为|:|：)?(?P<shares>\d[\d,]*(?:\.\d+)?)份")
NAV_PATTERN = re.compile(r"基金份额净值(?:为|:|：)?(?:人民币)?(?P<nav>\d[\d,]*(?:\.\d+)?)元?")
SEMANTIC_PATTERNS = {
    "additional_distribution_expected": re.compile(
        r"仍有[^。；]{0,100}(?:未变现|应收款|未收回)|再次进行分配|二次清算|进行二次分配|再次分配"
    ),
    "remaining_property_fully_distributed": re.compile(
        r"(?:本次分配结束后[,，]?)?剩余财产(?:已|将)?全部分配完毕|(?:本次)?剩余财产分配完毕"
    ),
    "exit_registration_announced": re.compile(r"办理[^。；]{0,80}(?:退出登记|基金份额注销)|基金份额(?:将)?注销"),
    "fund_contract_terminated": re.compile(r"基金合同(?:已)?终止|终止基金合同"),
    "cash_clearing_method": re.compile(r"现金清盘|现金红利发放方式|清算资金[^。；]{0,80}发放"),
}
DOCUMENT_PARSE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "published_at",
    "announcement_title",
    "source_url",
    "document_role_candidates_json",
    "text_status",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "direct_cash_candidates_json",
    "total_cash_candidates_json",
    "share_candidates_json",
    "nav_candidates_json",
    "record_date_candidates_json",
    "ex_date_candidates_json",
    "pay_date_candidates_json",
    "last_operation_date_candidates_json",
    "liquidation_start_date_candidates_json",
    "liquidation_end_date_candidates_json",
    "termination_date_candidates_json",
    "semantics_json",
    "parse_warnings_json",
    "available_at",
    "available_trade_date",
    "available_date",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
CASH_EVENT_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "candidate_event_id",
    "distribution_sequence_hint",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "direct_cash_per_share",
    "distribution_total_cash",
    "cash_value_basis",
    "cash_value_parse_status",
    "date_parse_status",
    "additional_distribution_expected",
    "remaining_property_fully_distributed",
    "exit_registration_announced",
    "fund_contract_terminated",
    "economic_extinguishment_candidate",
    "source_url",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "text_status",
    "published_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "candidate_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
LIQUIDATION_COLUMNS = [
    "asset",
    "asset_name",
    "announcement_date",
    "last_operation_date",
    "liquidation_start_date",
    "liquidation_end_date",
    "liquidation_shares",
    "liquidation_nav",
    "remaining_property",
    "share_parse_status",
    "date_parse_status",
    "source_url",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "text_status",
    "candidate_status",
    "historical_backtest_allowed",
]
DELISTING_COLUMNS = [
    "asset",
    "asset_name",
    "announcement_date",
    "termination_date",
    "termination_date_parse_status",
    "source_url",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "candidate_status",
    "historical_backtest_allowed",
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


def compact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("，", ",").replace("．", ".").replace("：", ":")
    return re.sub(r"\s+", "", normalized)


def _number(value: str) -> float:
    return float(str(value).replace(",", ""))


def _snippet(text: str, start: int, end: int, margin: int = 45) -> str:
    return text[max(0, start - margin) : min(len(text), end + margin)]


def _date(year: str, month: str, day: str) -> str | None:
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day)).date().isoformat()
    except ValueError:
        return None


def _distribution_statement_rank(text: str, start: int) -> int:
    """Prefer the current distribution statement over amounts quoted as history."""
    before = text[max(0, start - 260) : start]
    current_position = max(
        (before.rfind(marker) for marker in ("本次分配", "本次发放", "本次清算", "本次")),
        default=-1,
    )
    historical_position = -1
    historical_patterns = (
        r"(?:此前|前次|上次|曾于|已于)[^。；]{0,120}(?:分配|发放)",
        r"于20\d{2}年\d{1,2}月\d{1,2}日[^。；]{0,120}(?:进行了?|完成了?)(?:第[一二三四五六七八九十]+次)?(?:剩余财产)?分配",
        r"第[一二三四五六七八九十]+次(?:剩余财产)?分配",
    )
    for pattern in historical_patterns:
        for match in re.finditer(pattern, before):
            historical_position = max(historical_position, match.start())
    if current_position > historical_position:
        return 0
    if historical_position >= 0:
        return 5
    return 1


def extract_direct_cash_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    denominator_map = {"": 1.0, "十": 10.0, "百": 100.0, "千": 1000.0}
    for match in DIRECT_CASH_PATTERN.finditer(text):
        context = str(match.group("context"))
        if not any(marker in context for marker in CASH_CONTEXT_MARKERS):
            continue
        raw_denominator = str(match.group("denominator") or "")
        denominator = (
            denominator_map[raw_denominator]
            if raw_denominator in denominator_map
            else float(raw_denominator)
        )
        amount = _number(match.group("amount"))
        if denominator <= 0 or amount <= 0:
            continue
        candidates.append(
            {
                "pattern_id": "per_n_shares_distribution",
                "rank": _distribution_statement_rank(text, match.start()),
                "amount": amount,
                "denominator_shares": denominator,
                "cash_per_share": amount / denominator,
                "snippet": _snippet(text, match.start(), match.end()),
            }
        )
    for pattern_id, pattern in DIRECT_CASH_ALTERNATE_PATTERNS:
        for match in pattern.finditer(text):
            amount = _number(match.group("amount"))
            if amount > 0:
                candidates.append(
                    {
                        "pattern_id": pattern_id,
                        "rank": _distribution_statement_rank(text, match.start()),
                        "amount": amount,
                        "denominator_shares": 1.0,
                        "cash_per_share": amount,
                        "snippet": _snippet(text, match.start(), match.end()),
                    }
                )
    return candidates


def extract_total_cash_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pattern_id, rank, pattern in TOTAL_CASH_PATTERNS:
        for match in pattern.finditer(text):
            amount = _number(match.group("amount"))
            if amount > 0:
                candidates.append(
                    {
                        "pattern_id": pattern_id,
                        "rank": rank,
                        "total_cash": amount,
                        "snippet": _snippet(text, match.start(), match.end()),
                    }
                )
    return candidates


def extract_share_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in SHARE_PATTERN.finditer(text):
        shares = _number(match.group("shares"))
        if shares <= 0:
            continue
        context = _snippet(text, match.start(), match.end(), margin=120)
        before = text[max(0, match.start() - 120) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 55)]
        rank = 2
        current_position = max(
            (before.rfind(marker) for marker in ("最后运作日", "清算期", "清算开始日", "期末", "截至")),
            default=-1,
        )
        offering_position = max(
            (before.rfind(marker) for marker in ("认购", "募集", "上市交易", "初始基金份额")),
            default=-1,
        )
        if current_position > offering_position:
            rank = 0
        elif offering_position >= 0 or any(
            marker in after for marker in ("认购", "募集", "上市交易", "初始基金份额")
        ):
            rank = 9
        candidates.append(
            {
                "pattern_id": "fund_share_total",
                "rank": rank,
                "shares": shares,
                "snippet": context,
            }
        )
    return candidates


def extract_nav_candidates(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in NAV_PATTERN.finditer(text):
        value = _number(match.group("nav"))
        if value > 0:
            rows.append(
                {
                    "pattern_id": "fund_unit_nav",
                    "rank": 0 if "清算" in _snippet(text, match.start(), match.end(), 100) else 2,
                    "nav": value,
                    "snippet": _snippet(text, match.start(), match.end()),
                }
            )
    return rows


def _extract_labeled_dates(
    text: str,
    patterns: Iterable[tuple[str, int, re.Pattern[str]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pattern_id, rank, pattern in patterns:
        for match in pattern.finditer(text):
            value = _date(match.group("year"), match.group("month"), match.group("day"))
            if value:
                candidates.append(
                    {
                        "pattern_id": pattern_id,
                        "rank": rank,
                        "date": value,
                        "snippet": _snippet(text, match.start(), match.end()),
                    }
                )
    return candidates


def extract_date_candidates(text: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "record_date": _extract_labeled_dates(
            text,
            (("record_date", 0, re.compile(rf"(?:清盘)?权益登记日(?:为|:)?{DATE_TOKEN}")),),
        ),
        "ex_date": _extract_labeled_dates(
            text,
            (("ex_date", 0, re.compile(rf"除息日(?:为|:)?{DATE_TOKEN}")),),
        ),
        "pay_date": _extract_labeled_dates(
            text,
            (
                (
                    "cash_pay_date",
                    0,
                    re.compile(rf"(?:本次[^。；]{{0,30}})?(?:剩余财产分配资金|清算资金|资金|剩余财产)?(?:发放日|分配日)(?:为|:)?{DATE_TOKEN}"),
                ),
                ("generic_distribution_date", 1, re.compile(rf"发放日(?:为|:)?{DATE_TOKEN}")),
                (
                    "custodian_cash_transfer_date",
                    2,
                    re.compile(rf"(?:上述资金|清算款)[^。；]{{0,35}}?(?:已于|将于|于){DATE_TOKEN}[^。；]{{0,35}}?划出"),
                ),
            ),
        ),
        "last_operation_date": _extract_labeled_dates(
            text,
            (("last_operation_date", 0, re.compile(rf"最后运作日(?:定为|为|:)?{DATE_TOKEN}")),),
        ),
        "liquidation_start_date": _extract_labeled_dates(
            text,
            (
                ("liquidation_period_start", 0, re.compile(rf"清算期(?:为|自)?{DATE_TOKEN}(?:至|起)")),
                ("entered_liquidation", 1, re.compile(rf"自{DATE_TOKEN}起[^。；]{{0,50}}(?:进入|开始)基金?财产?清算")),
            ),
        ),
        "liquidation_end_date": _extract_labeled_dates(
            text,
            (("liquidation_period_end", 0, re.compile(rf"清算期(?:为|自)?20\d{{2}}[年./-]\d{{1,2}}[月./-]\d{{1,2}}日?(?:至|起){DATE_TOKEN}")),),
        ),
        "termination_date": _extract_labeled_dates(
            text,
            (
                ("delisting_date", 0, re.compile(rf"终止上市(?:交易)?日(?:为|:)?{DATE_TOKEN}")),
                ("delisting_from_date", 1, re.compile(rf"自{DATE_TOKEN}起[^。；]{{0,35}}终止上市")),
            ),
        ),
    }


def extract_semantics(text: str) -> dict[str, bool]:
    return {name: bool(pattern.search(text)) for name, pattern in SEMANTIC_PATTERNS.items()}


def _selected_numeric(
    candidates: list[dict[str, Any]],
    field: str,
    *,
    tolerance: float = 1e-10,
) -> tuple[float | None, str]:
    if not candidates:
        return None, "missing"
    best_rank = min(int(item["rank"]) for item in candidates)
    values = [float(item[field]) for item in candidates if int(item["rank"]) == best_rank]
    unique: list[float] = []
    for value in values:
        if not any(np.isclose(value, existing, rtol=0.0, atol=tolerance) for existing in unique):
            unique.append(value)
    return (unique[0], "unique") if len(unique) == 1 else (None, "ambiguous")


def _selected_date(candidates: list[dict[str, Any]]) -> tuple[pd.Timestamp | None, str]:
    if not candidates:
        return None, "missing"
    best_rank = min(int(item["rank"]) for item in candidates)
    values = sorted({str(item["date"]) for item in candidates if int(item["rank"]) == best_rank})
    return (pd.Timestamp(values[0]), "unique") if len(values) == 1 else (None, "ambiguous")


def _sequence_hint(title: str) -> int:
    if re.search(r"三次|第三次", title):
        return 3
    if re.search(r"二次|第二次", title):
        return 2
    return 1


def parse_document(row: Any) -> dict[str, Any]:
    text_path = ROOT / str(row.text_path)
    if not text_path.is_file() or _sha256(text_path) != str(row.text_sha256):
        raise ValueError(f"terminal document text hash mismatch: {row.asset}/{row.source_url}")
    text = compact_text(text_path.read_text(encoding="utf-8-sig"))
    direct_cash = extract_direct_cash_candidates(text)
    total_cash = extract_total_cash_candidates(text)
    shares = extract_share_candidates(text)
    nav = extract_nav_candidates(text)
    dates = extract_date_candidates(text)
    semantics = extract_semantics(text)
    warnings: list[str] = []
    if str(row.text_status) == "ocr_derived_unvalidated":
        warnings.append("ocr_derived_text_requires_original_pdf_review")
    if not text:
        warnings.append("empty_normalized_text")
    return {
        "asset": str(row.asset).zfill(6),
        "asset_name": str(row.asset_name),
        "exchange": str(row.exchange),
        "announcement_date": pd.Timestamp(row.announcement_date).normalize(),
        "published_at": str(row.published_at) if pd.notna(row.published_at) else "",
        "announcement_title": str(row.announcement_title),
        "source_url": str(row.source_url),
        "document_role_candidates_json": str(row.document_role_candidates_json),
        "text_status": str(row.text_status),
        "pdf_path": str(row.pdf_path),
        "pdf_sha256": str(row.pdf_sha256),
        "text_path": str(row.text_path),
        "text_sha256": str(row.text_sha256),
        "direct_cash_candidates_json": json.dumps(direct_cash, ensure_ascii=False, sort_keys=True),
        "total_cash_candidates_json": json.dumps(total_cash, ensure_ascii=False, sort_keys=True),
        "share_candidates_json": json.dumps(shares, ensure_ascii=False, sort_keys=True),
        "nav_candidates_json": json.dumps(nav, ensure_ascii=False, sort_keys=True),
        "record_date_candidates_json": json.dumps(dates["record_date"], ensure_ascii=False, sort_keys=True),
        "ex_date_candidates_json": json.dumps(dates["ex_date"], ensure_ascii=False, sort_keys=True),
        "pay_date_candidates_json": json.dumps(dates["pay_date"], ensure_ascii=False, sort_keys=True),
        "last_operation_date_candidates_json": json.dumps(
            dates["last_operation_date"], ensure_ascii=False, sort_keys=True
        ),
        "liquidation_start_date_candidates_json": json.dumps(
            dates["liquidation_start_date"], ensure_ascii=False, sort_keys=True
        ),
        "liquidation_end_date_candidates_json": json.dumps(
            dates["liquidation_end_date"], ensure_ascii=False, sort_keys=True
        ),
        "termination_date_candidates_json": json.dumps(
            dates["termination_date"], ensure_ascii=False, sort_keys=True
        ),
        "semantics_json": json.dumps(semantics, ensure_ascii=False, sort_keys=True),
        "parse_warnings_json": json.dumps(warnings, ensure_ascii=False),
        "available_at": str(row.available_at),
        "available_trade_date": pd.Timestamp(row.available_trade_date).normalize(),
        "available_date": pd.Timestamp(row.available_date).normalize(),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def build_cash_candidates(parsed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in parsed.itertuples(index=False):
        title = str(row.announcement_title)
        roles = set(json.loads(str(row.document_role_candidates_json)))
        is_cash_document = (
            "提示性" not in title
            and ("cash_distribution" in roles or bool(re.search(r"(?:资金清算|清算资金).*发放", title)))
        )
        if not is_cash_document:
            continue
        direct, direct_status = _selected_numeric(json.loads(row.direct_cash_candidates_json), "cash_per_share")
        total, total_status = _selected_numeric(json.loads(row.total_cash_candidates_json), "total_cash", tolerance=0.005)
        record, record_status = _selected_date(json.loads(row.record_date_candidates_json))
        ex_date, ex_status = _selected_date(json.loads(row.ex_date_candidates_json))
        pay, pay_status = _selected_date(json.loads(row.pay_date_candidates_json))
        semantics = json.loads(row.semantics_json)
        if direct_status == "unique":
            cash_basis, cash_status = "direct_official_per_share", "unique"
        elif total_status == "unique":
            cash_basis, cash_status = "official_total_requires_share_reconciliation", "requires_liquidation_shares"
        elif direct_status == "ambiguous" or total_status == "ambiguous":
            cash_basis, cash_status = "ambiguous_official_amount", "ambiguous"
        else:
            cash_basis, cash_status = "missing_official_amount", "missing"
        date_status = (
            "unique"
            if record_status == "unique" and pay_status == "unique"
            else "ambiguous"
            if "ambiguous" in {record_status, pay_status}
            else "missing"
        )
        available_trade_date = pd.Timestamp(
            getattr(row, "available_trade_date", getattr(row, "available_date", row.announcement_date))
        ).normalize()
        chronology_ok = (
            record is not None
            and pay is not None
            and record <= pay
            and available_trade_date <= pay
        )
        final_candidate = bool(
            semantics["remaining_property_fully_distributed"]
            and semantics["exit_registration_announced"]
            and not semantics["additional_distribution_expected"]
        )
        candidate_status = (
            "field_candidates_complete_requires_independent_validation"
            if cash_status in {"unique", "requires_liquidation_shares"}
            and date_status == "unique"
            and chronology_ok
            else "field_candidates_incomplete_or_ambiguous"
        )
        event_id = hashlib.sha256(f"{row.asset}|{row.source_url}".encode("utf-8")).hexdigest()
        rows.append(
            {
                "asset": str(row.asset),
                "asset_name": str(row.asset_name),
                "exchange": str(row.exchange),
                "candidate_event_id": event_id,
                "distribution_sequence_hint": _sequence_hint(title),
                "announcement_date": pd.Timestamp(row.announcement_date),
                "record_date": record,
                "ex_date": ex_date if ex_status == "unique" else pd.NaT,
                "pay_date": pay,
                "direct_cash_per_share": direct,
                "distribution_total_cash": total,
                "cash_value_basis": cash_basis,
                "cash_value_parse_status": cash_status,
                "date_parse_status": date_status,
                "additional_distribution_expected": bool(semantics["additional_distribution_expected"]),
                "remaining_property_fully_distributed": bool(semantics["remaining_property_fully_distributed"]),
                "exit_registration_announced": bool(semantics["exit_registration_announced"]),
                "fund_contract_terminated": bool(semantics["fund_contract_terminated"]),
                "economic_extinguishment_candidate": final_candidate,
                "source_url": str(row.source_url),
                "pdf_path": str(row.pdf_path),
                "pdf_sha256": str(row.pdf_sha256),
                "text_path": str(row.text_path),
                "text_sha256": str(row.text_sha256),
                "text_status": str(row.text_status),
                "published_at": str(getattr(row, "published_at", "")),
                "available_at": str(getattr(row, "available_at", available_trade_date.isoformat())),
                "available_trade_date": available_trade_date,
                "available_date": pd.Timestamp(getattr(row, "available_date", available_trade_date)),
                "candidate_status": candidate_status,
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=CASH_EVENT_COLUMNS).sort_values(
        ["asset", "pay_date", "announcement_date", "source_url"], na_position="last"
    ).reset_index(drop=True)


def build_liquidation_candidates(parsed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in parsed.itertuples(index=False):
        roles = set(json.loads(str(row.document_role_candidates_json)))
        if "liquidation_report" not in roles or "提示性" in str(row.announcement_title):
            continue
        shares, shares_status = _selected_numeric(json.loads(row.share_candidates_json), "shares", tolerance=0.005)
        nav, _ = _selected_numeric(json.loads(row.nav_candidates_json), "nav", tolerance=5e-5)
        remaining, _ = _selected_numeric(json.loads(row.total_cash_candidates_json), "total_cash", tolerance=0.005)
        last_operation, last_status = _selected_date(json.loads(row.last_operation_date_candidates_json))
        start, start_status = _selected_date(json.loads(row.liquidation_start_date_candidates_json))
        end, end_status = _selected_date(json.loads(row.liquidation_end_date_candidates_json))
        date_statuses = {last_status, start_status, end_status}
        date_status = "unique" if date_statuses == {"unique"} else "ambiguous" if "ambiguous" in date_statuses else "missing"
        candidate_status = (
            "field_candidates_complete_requires_independent_validation"
            if shares_status == "unique" and shares is not None
            else "field_candidates_incomplete_or_ambiguous"
        )
        rows.append(
            {
                "asset": str(row.asset),
                "asset_name": str(row.asset_name),
                "announcement_date": pd.Timestamp(row.announcement_date),
                "last_operation_date": last_operation,
                "liquidation_start_date": start,
                "liquidation_end_date": end,
                "liquidation_shares": shares,
                "liquidation_nav": nav,
                "remaining_property": remaining,
                "share_parse_status": shares_status,
                "date_parse_status": date_status,
                "source_url": str(row.source_url),
                "pdf_path": str(row.pdf_path),
                "pdf_sha256": str(row.pdf_sha256),
                "text_path": str(row.text_path),
                "text_sha256": str(row.text_sha256),
                "text_status": str(row.text_status),
                "candidate_status": candidate_status,
                "historical_backtest_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=LIQUIDATION_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def build_delisting_candidates(parsed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in parsed.itertuples(index=False):
        roles = set(json.loads(str(row.document_role_candidates_json)))
        if "exchange_delisting" not in roles:
            continue
        termination, status = _selected_date(json.loads(row.termination_date_candidates_json))
        rows.append(
            {
                "asset": str(row.asset),
                "asset_name": str(row.asset_name),
                "announcement_date": pd.Timestamp(row.announcement_date),
                "termination_date": termination,
                "termination_date_parse_status": status,
                "source_url": str(row.source_url),
                "pdf_path": str(row.pdf_path),
                "pdf_sha256": str(row.pdf_sha256),
                "text_path": str(row.text_path),
                "text_sha256": str(row.text_sha256),
                "candidate_status": (
                    "field_candidate_complete_requires_independent_validation"
                    if status == "unique"
                    else "field_candidate_incomplete_or_ambiguous"
                ),
                "historical_backtest_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=DELISTING_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def _authenticate_documents() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    manifest = json.loads(DOCUMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        != "FULL_IMMUTABLE_DOCUMENT_TEXT_VIEW_REQUIRES_EVENT_VALIDATION"
        or int(manifest.get("covered_assets", 0)) != 123
        or int(manifest.get("document_rows", 0)) <= 0
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("terminal-event document manifest does not authorize candidate parsing")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal-event document collector code hash mismatch")
    output = next((item for item in manifest.get("outputs", []) if item.get("role") == "merged_document_index"), None)
    if output is None:
        raise ValueError("terminal-event document manifest misses document index")
    path = ROOT / str(output.get("path", ""))
    if not path.is_file() or _sha256(path) != str(output.get("sha256", "")):
        raise ValueError("terminal-event document index hash mismatch")
    frame = pd.read_csv(path, dtype={"asset": str})
    if len(frame) != int(manifest["document_rows"]) or not frame["text_status"].isin(document_merge.TEXT_AVAILABLE_STATUSES).all():
        raise ValueError("terminal-event document index is incomplete or has unavailable text")
    return frame, [
        {"role": "document_manifest", "path": _relative(DOCUMENT_MANIFEST_PATH), "sha256": _sha256(DOCUMENT_MANIFEST_PATH)},
        {"role": "document_index", "path": _relative(path), "sha256": _sha256(path)},
    ]


def run() -> dict[str, Any]:
    index, inputs = _authenticate_documents()
    parsed = pd.DataFrame(
        [parse_document(row) for row in index.itertuples(index=False)],
        columns=DOCUMENT_PARSE_COLUMNS,
    ).sort_values(["asset", "announcement_date", "source_url"]).reset_index(drop=True)
    cash = build_cash_candidates(parsed)
    liquidation = build_liquidation_candidates(parsed)
    delisting = build_delisting_candidates(parsed)
    review = cash[cash["candidate_status"].ne("field_candidates_complete_requires_independent_validation")].copy()
    _atomic_csv(parsed, DOCUMENT_PARSE_PATH)
    _atomic_csv(cash, CASH_EVENT_PATH)
    _atomic_csv(liquidation, LIQUIDATION_PATH)
    _atomic_csv(delisting, DELISTING_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "OBSERVATION_CANDIDATES_PARSED_REQUIRES_INDEPENDENT_VALIDATION",
        "document_rows": int(len(parsed)),
        "cash_candidate_rows": int(len(cash)),
        "cash_candidate_assets": int(cash["asset"].nunique()) if not cash.empty else 0,
        "cash_candidates_complete": int(
            cash["candidate_status"].eq("field_candidates_complete_requires_independent_validation").sum()
        ),
        "cash_candidates_direct_per_share": int(cash["direct_cash_per_share"].notna().sum()),
        "cash_candidates_require_share_reconciliation": int(
            cash["cash_value_parse_status"].eq("requires_liquidation_shares").sum()
        ),
        "liquidation_report_rows": int(len(liquidation)),
        "liquidation_report_assets": int(liquidation["asset"].nunique()) if not liquidation.empty else 0,
        "liquidation_reports_with_unique_shares": int(liquidation["liquidation_shares"].notna().sum()),
        "delisting_candidate_rows": int(len(delisting)),
        "delisting_dates_unique": int(delisting["termination_date"].notna().sum()),
        "ocr_document_rows": int(parsed["text_status"].eq("ocr_derived_unvalidated").sum()),
        "parse_review_queue_rows": int(len(review)),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "Regex outputs are traceable field candidates, not validated event economics or formal PIT rows.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "document_parse_candidates", "path": _relative(DOCUMENT_PARSE_PATH), "sha256": _sha256(DOCUMENT_PARSE_PATH), "rows": int(len(parsed))},
        {"role": "cash_event_candidates", "path": _relative(CASH_EVENT_PATH), "sha256": _sha256(CASH_EVENT_PATH), "rows": int(len(cash))},
        {"role": "liquidation_report_candidates", "path": _relative(LIQUIDATION_PATH), "sha256": _sha256(LIQUIDATION_PATH), "rows": int(len(liquidation))},
        {"role": "delisting_candidates", "path": _relative(DELISTING_PATH), "sha256": _sha256(DELISTING_PATH), "rows": int(len(delisting))},
        {"role": "parse_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review))},
        {"role": "parse_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(document_merge.__file__).resolve()), "sha256": _sha256(Path(document_merge.__file__).resolve())}
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = run()
    keys = (
        "qualification_status",
        "cash_candidate_rows",
        "cash_candidates_complete",
        "cash_candidates_direct_per_share",
        "cash_candidates_require_share_reconciliation",
        "parse_review_queue_rows",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
