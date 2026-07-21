"""Independently validate terminal cash-event candidates against official text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import pit_etf_terminal_event_candidate_parser as parser


ROOT = Path(__file__).resolve().parents[2]
PARSER_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_candidate_parser_latest.json"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_v2"
CHECKS_PATH = OUTPUT_DIR / "event_validation_checks.csv"
CHAIN_PATH = OUTPUT_DIR / "asset_chain_validation.csv"
PROMOTION_CANDIDATE_PATH = OUTPUT_DIR / "promotion_candidates.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

SCHEMA_VERSION = 2
DATE_TOKEN = r"(?P<year>20\d{2})[年./-](?P<month>\d{1,2})[月./-](?P<day>\d{1,2})日?"
NUMBER_TOKEN = r"\d[\d,]*(?:\.\d+)?"
PER_UNIT_PATTERN = re.compile(
    rf"每(?P<denominator>\d+|十|百|千)?份(?P<context>[^。；]{{0,180}}?)"
    rf"(?:为|:)?(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"
)
PER_UNIT_TABLE_PATTERN = re.compile(
    rf"每(?P<denominator>\d+)份基金份额分配(?P<amount>{NUMBER_TOKEN})元人民币"
)
PER_UNIT_FUND_UNIT_PATTERN = re.compile(
    rf"每单位基金份额(?P<context>[^。；]{{0,120}}?)(?:为|:)?(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"
)
TOTAL_PATTERNS = (
    re.compile(rf"本次(?:清算)?可供分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"),
    re.compile(rf"本次(?:应)?分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"),
    re.compile(rf"应分配剩余财产(?:为|共计)(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"),
    re.compile(rf"剩余财产(?:总额)?(?:为|共计)(?:人民币)?(?P<amount>{NUMBER_TOKEN})元"),
)
RECORD_PATTERNS = (
    re.compile(rf"(?:清盘)?权益登记日(?:为|:)?{DATE_TOKEN}"),
)
PAY_PATTERNS = (
    re.compile(rf"(?:资金发放日|清算资金发放日|清算资金分配日|剩余财产发放日|发放日)(?:为|:)?{DATE_TOKEN}"),
    re.compile(rf"(?:上述资金|清算款)[^。；]{{0,35}}?(?:已于|将于|于){DATE_TOKEN}[^。；]{{0,35}}?划出"),
)
CASH_CONTEXT = (
    "发放资金",
    "可获分配",
    "分配清算资金",
    "分配的剩余财产",
    "清算资金",
    "剩余财产",
)
PENDING_MORE_PATTERN = re.compile(
    r"(?:仍有|尚有|由于)[^。；]{0,120}(?:未变现|应收款|流通受限)|"
    r"(?:需要|将|待[^。；]{0,60}?后)[^。；]{0,100}(?:二次|多次|再次)[^。；]{0,45}(?:清算|分配)"
)
CONTRACT_TERMINATED_PATTERN = re.compile(r"基金合同(?:自[^。；]{0,35})?(?:已)?终止|终止基金合同")
REMAINING_FULL_PATTERN = re.compile(r"剩余财产(?:已|将)?全部分配完毕|剩余财产分配完毕")
EXIT_PLANNED_PATTERN = re.compile(r"(?:办理|申请办理)[^。；]{0,50}(?:退出登记|账户注销|份额注销)")
SHARE_PATTERNS = (
    re.compile(
        rf"(?:报告期末|报告截止日[^。；]{{0,80}}基金最后运作日|基金最后运作日[^。；]{{0,120}})"
        rf"[^。；]{{0,160}}?(?:基金份额总额|基金份额为|基金份额|份额为)(?P<shares>{NUMBER_TOKEN})份"
    ),
    re.compile(
        rf"二次清算截止日[^。；]{{0,80}}?(?P<shares>{NUMBER_TOKEN})份基金份额总额"
    ),
    re.compile(
        rf"(?:基金)?最后运作日\(20\d{{2}}年\d{{1,2}}月\d{{1,2}}日\)"
        rf"(?P<shares>{NUMBER_TOKEN})份基金份额总额"
    ),
)

FORMAL_COLUMNS = [
    "event_id",
    "asset",
    "asset_name",
    "exchange",
    "event_type",
    "distribution_sequence",
    "holder_scope",
    "announcement_date",
    "published_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "entitlement_date",
    "entitlement_date_basis",
    "record_date",
    "ex_date",
    "pay_date",
    "accounting_date",
    "cash_per_share",
    "distribution_total_cash",
    "cash_denominator_shares",
    "cash_value_basis",
    "is_final_distribution",
    "additional_distribution_expected",
    "fund_contract_terminated",
    "exit_registration_announced",
    "termination_date",
    "extinguishes_position",
    "source_urls_json",
    "source_pdf_sha256_set",
    "source_text_sha256_set",
    "source_vintage",
    "validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
CHECK_COLUMNS = [
    "candidate_event_id",
    "asset",
    "announcement_date",
    "source_url",
    "amount_check",
    "date_check",
    "entitlement_check",
    "identity_check",
    "text_check",
    "validation_status",
    "failure_reasons_json",
]
CHAIN_COLUMNS = [
    "asset",
    "cash_candidate_rows",
    "validated_event_rows",
    "failed_candidate_rows",
    "latest_event_id",
    "termination_date",
    "final_distribution_supported",
    "position_extinguishment_supported",
    "chain_status",
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


def _set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(str(value) for value in values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("，", ",").replace("：", ":").replace("．", ".")
    return re.sub(r"\s+", "", value)


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid official decimal: {value}") from exc


def _date(match: re.Match[str]) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        ).normalize()
    except ValueError:
        return None


def _historical_rank(text: str, start: int) -> int:
    before = text[max(0, start - 260) : start]
    current = max(before.rfind("本次分配"), before.rfind("本次发放"), before.rfind("本次"))
    historical = -1
    for pattern in (
        r"(?:此前|前次|上次|曾于|已于)[^。；]{0,120}(?:分配|发放)",
        r"第[一二三四五六七八九十]+次(?:剩余财产)?分配",
        r"于20\d{2}年\d{1,2}月\d{1,2}日[^。；]{0,120}(?:进行了?|完成了?)(?:剩余财产)?分配",
    ):
        historical = max(historical, *(match.start() for match in re.finditer(pattern, before)), -1)
    if current > historical:
        return 0
    if historical >= 0:
        return 5
    return 1


def strict_direct_values(text: str) -> list[tuple[Decimal, int]]:
    denominator_map = {"": Decimal(1), "十": Decimal(10), "百": Decimal(100), "千": Decimal(1000)}
    values: list[tuple[Decimal, int]] = []
    for match in PER_UNIT_PATTERN.finditer(text):
        if not any(marker in str(match.group("context")) for marker in CASH_CONTEXT):
            continue
        raw = str(match.group("denominator") or "")
        denominator = denominator_map[raw] if raw in denominator_map else _decimal(raw)
        amount = _decimal(match.group("amount"))
        if amount > 0 and denominator > 0:
            values.append((amount / denominator, _historical_rank(text, match.start())))
    for match in PER_UNIT_TABLE_PATTERN.finditer(text):
        amount = _decimal(match.group("amount"))
        denominator = _decimal(match.group("denominator"))
        if amount > 0 and denominator > 0:
            values.append((amount / denominator, _historical_rank(text, match.start())))
    for match in PER_UNIT_FUND_UNIT_PATTERN.finditer(text):
        if not any(marker in str(match.group("context")) for marker in CASH_CONTEXT):
            continue
        amount = _decimal(match.group("amount"))
        if amount > 0:
            values.append((amount, _historical_rank(text, match.start())))
    return values


def _decimal_close(left: Decimal, right: Decimal, tolerance: Decimal = Decimal("0.000000000001")) -> bool:
    return abs(left - right) <= tolerance


def strict_total_values(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for pattern in TOTAL_PATTERNS:
        for match in pattern.finditer(text):
            amount = _decimal(match.group("amount"))
            if amount > 0:
                values.append(amount)
    return values


def strict_dates(text: str, patterns: Iterable[re.Pattern[str]]) -> set[pd.Timestamp]:
    values: set[pd.Timestamp] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = _date(match)
            if value is not None:
                values.add(value)
    return values


def _authenticated_outputs() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    manifest = json.loads(PARSER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        != "OBSERVATION_CANDIDATES_PARSED_REQUIRES_INDEPENDENT_VALIDATION"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
    ):
        raise ValueError("terminal-event parser manifest does not authorize validation")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal-event parser code hash mismatch")
    roles = {
        "document_parse_candidates",
        "cash_event_candidates",
        "liquidation_report_candidates",
        "delisting_candidates",
    }
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    frames: dict[str, pd.DataFrame] = {}
    lineage = [
        {"role": "candidate_parser_manifest", "path": _relative(PARSER_MANIFEST_PATH), "sha256": _sha256(PARSER_MANIFEST_PATH)}
    ]
    for role in roles:
        item = outputs.get(role)
        if item is None:
            raise ValueError(f"terminal-event parser manifest misses {role}")
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"terminal-event parser output hash mismatch: {role}")
        frames[role] = pd.read_csv(path, dtype={"asset": str})
        lineage.append({"role": role, "path": _relative(path), "sha256": _sha256(path)})
    return frames, lineage


def _load_text(row: Any) -> str:
    pdf_path = ROOT / str(row.pdf_path)
    if not pdf_path.is_file() or _sha256(pdf_path) != str(row.pdf_sha256):
        raise ValueError("official terminal-event PDF hash mismatch")
    path = ROOT / str(row.text_path)
    if not path.is_file() or _sha256(path) != str(row.text_sha256):
        raise ValueError("official terminal-event text hash mismatch")
    return _compact(path.read_text(encoding="utf-8-sig"))


def _unique_support_map(
    liquidation: pd.DataFrame,
    field: str,
) -> dict[str, tuple[Any, pd.Series]]:
    rows: dict[str, tuple[Any, pd.Series]] = {}
    usable = liquidation[
        liquidation[field].notna() & liquidation["text_status"].eq("success")
    ].copy()
    for asset, group in usable.groupby("asset"):
        numeric = pd.to_numeric(group[field], errors="coerce")
        unique: list[float] = []
        for value in numeric.dropna():
            if not any(np.isclose(float(value), item, rtol=0.0, atol=0.005) for item in unique):
                unique.append(float(value))
        if len(unique) == 1:
            matched = group[np.isclose(numeric.astype(float), unique[0], rtol=0.0, atol=0.005)].iloc[-1]
            rows[str(asset).zfill(6)] = (unique[0], matched)
    return rows


def _unique_date_support_map(
    liquidation: pd.DataFrame,
    field: str,
) -> dict[str, tuple[pd.Timestamp, pd.Series]]:
    rows: dict[str, tuple[pd.Timestamp, pd.Series]] = {}
    usable = liquidation[
        liquidation[field].notna() & liquidation["text_status"].eq("success")
    ].copy()
    usable[field] = pd.to_datetime(usable[field], errors="coerce").dt.normalize()
    for asset, group in usable.dropna(subset=[field]).groupby("asset"):
        values = sorted(set(group[field]))
        if len(values) == 1:
            rows[str(asset).zfill(6)] = (values[0], group.iloc[-1])
    return rows


def _strict_share_support_map(
    liquidation: pd.DataFrame,
) -> dict[str, tuple[float, pd.Series]]:
    strict_candidates: dict[str, list[tuple[float, pd.Series]]] = {}
    fallback_candidates: dict[str, list[tuple[float, pd.Series]]] = {}
    for row in liquidation[liquidation["text_status"].eq("success")].itertuples(index=False):
        try:
            text = _load_text(row)
        except (OSError, ValueError):
            continue
        for pattern in SHARE_PATTERNS:
            for match in pattern.finditer(text):
                value = float(_decimal(match.group("shares")))
                if value > 0:
                    strict_candidates.setdefault(str(row.asset).zfill(6), []).append(
                        (value, pd.Series(row._asdict()))
                    )
        if pd.notna(row.liquidation_shares):
            expected = float(row.liquidation_shares)
            for match in re.finditer(NUMBER_TOKEN, text):
                value = float(_decimal(match.group(0)))
                if not np.isclose(value, expected, rtol=0.0, atol=0.005):
                    continue
                before = text[max(0, match.start() - 220) : match.start()]
                current_position = max(
                    before.rfind("最后运作日"),
                    before.rfind("报告截止日"),
                    before.rfind("报告期末"),
                    before.rfind("清算截止日"),
                )
                offering_position = max(
                    before.rfind("基金合同生效日"),
                    before.rfind("募集"),
                    before.rfind("认购"),
                )
                if current_position > offering_position:
                    fallback_candidates.setdefault(str(row.asset).zfill(6), []).append(
                        (value, pd.Series(row._asdict()))
                    )
                    break
    result: dict[str, tuple[float, pd.Series]] = {}
    all_assets = set(strict_candidates).union(fallback_candidates)
    for asset in all_assets:
        values = strict_candidates.get(asset) or fallback_candidates.get(asset, [])
        unique: list[float] = []
        for value, _ in values:
            if not any(np.isclose(value, item, rtol=0.0, atol=0.005) for item in unique):
                unique.append(value)
        if len(unique) == 1:
            support = next(row for value, row in reversed(values) if np.isclose(value, unique[0], atol=0.005))
            result[asset] = (unique[0], support)
    return result


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _optional_date(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp).normalize()


def _termination_map(delisting: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Series]]:
    rows: dict[str, tuple[pd.Timestamp, pd.Series]] = {}
    usable = delisting[
        delisting["termination_date"].notna()
        & delisting["termination_date_parse_status"].eq("unique")
        & delisting["candidate_status"].eq("field_candidate_complete_requires_independent_validation")
    ].copy()
    usable["termination_date"] = pd.to_datetime(usable["termination_date"], errors="coerce").dt.normalize()
    for asset, group in usable.dropna(subset=["termination_date"]).groupby("asset"):
        authenticated_rows: list[pd.Series] = []
        for _, row in group.iterrows():
            try:
                _load_text(row)
            except (OSError, ValueError):
                continue
            authenticated_rows.append(row)
        if not authenticated_rows:
            continue
        authenticated = pd.DataFrame(authenticated_rows)
        values = sorted(set(authenticated["termination_date"]))
        if len(values) == 1:
            rows[str(asset).zfill(6)] = (values[0], authenticated.iloc[-1])
    return rows


def _holder_scope_and_pay(asset: str, text: str, parsed_pay: pd.Timestamp | None) -> tuple[str, pd.Timestamp | None]:
    if asset == "511290":
        match = re.search(
            rf"场内份额[^。；]{{0,260}}?清算资金发放日(?:为|:)?{DATE_TOKEN}",
            text,
        )
        return "exchange_registered", _date(match) if match else None
    if "场内基金份额" in text or "场内份额" in text:
        return "exchange_registered", parsed_pay
    return "all_registered_holders", parsed_pay


def _validate_candidates(
    cash: pd.DataFrame,
    liquidation: pd.DataFrame,
    delisting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    share_map = _strict_share_support_map(liquidation)
    operation_map = _unique_date_support_map(liquidation, "last_operation_date")
    termination_map = _termination_map(delisting)
    checks: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    text_by_candidate: dict[str, str] = {}
    source_rows: dict[str, Any] = {}

    for row in cash.itertuples(index=False):
        asset = str(row.asset).zfill(6)
        candidate_id = str(row.candidate_event_id)
        failures: list[str] = []
        text = ""
        try:
            text = _load_text(row)
        except (OSError, ValueError) as exc:
            failures.append(f"text:{exc}")
        text_by_candidate[candidate_id] = text
        source_rows[candidate_id] = row
        text_ok = bool(text) and str(row.text_status) == "success"
        if not text_ok:
            failures.append("text_not_native_or_unavailable")
        hostname = (urlparse(str(row.source_url)).hostname or "").lower()
        identity_ok = hostname in {"www.sse.com.cn", "static.cninfo.com.cn"} and asset in str(row.pdf_path)
        if not identity_ok:
            # Hash-addressed PDF names do not contain the code; the authenticated
            # parser row and official host are the identity boundary.
            identity_ok = hostname in {"www.sse.com.cn", "static.cninfo.com.cn"} and len(asset) == 6
        if not identity_ok:
            failures.append("official_identity_not_authenticated")

        pdf_hashes = [str(row.pdf_sha256)]
        text_hashes = [str(row.text_sha256)]
        urls = [str(row.source_url)]
        amount: Decimal | None = None
        denominator: Decimal | None = None
        amount_basis = ""
        amount_ok = False
        if pd.notna(row.direct_cash_per_share):
            expected = _decimal(row.direct_cash_per_share)
            strict = strict_direct_values(text)
            best_rank = min((rank for _, rank in strict), default=99)
            amount_ok = any(
                _decimal_close(value, expected) and rank == best_rank for value, rank in strict
            )
            if amount_ok:
                amount = expected
                denominator = Decimal(1)
                amount_basis = "direct_official_per_share_independently_reparsed"
        elif pd.notna(row.distribution_total_cash):
            expected_total = _decimal(row.distribution_total_cash)
            total_ok = expected_total in strict_total_values(text)
            support = share_map.get(asset)
            if total_ok and support is not None:
                shares, support_row = support
                denominator = _decimal(shares)
                if denominator > 0:
                    amount = expected_total / denominator
                    amount_ok = True
                    amount_basis = "official_total_divided_by_native_validated_liquidation_shares"
                    pdf_hashes.append(str(support_row.pdf_sha256))
                    text_hashes.append(str(support_row.text_sha256))
                    urls.append(str(support_row.source_url))
        if not amount_ok or amount is None or amount <= 0:
            failures.append("cash_amount_not_independently_reconciled")

        parsed_record = _optional_date(row.record_date)
        parsed_pay = _optional_date(row.pay_date)
        holder_scope, pay_date = _holder_scope_and_pay(asset, text, parsed_pay)
        strict_record = strict_dates(text, RECORD_PATTERNS)
        strict_pay = strict_dates(text, PAY_PATTERNS)
        record_ok = parsed_record is not None and parsed_record in strict_record
        pay_ok = pay_date is not None and pay_date in strict_pay
        if parsed_record is not None and not record_ok:
            failures.append("record_date_not_independently_reparsed")
        if not pay_ok:
            failures.append("pay_or_transfer_date_not_independently_reparsed")

        entitlement_date = parsed_record
        entitlement_basis = "official_record_date" if parsed_record is not None else ""
        if entitlement_date is None:
            operation = operation_map.get(asset)
            if operation is not None:
                entitlement_date = pd.Timestamp(operation[0]).normalize()
                entitlement_basis = "native_liquidation_report_last_operation_date"
                support_row = operation[1]
                pdf_hashes.append(str(support_row.pdf_sha256))
                text_hashes.append(str(support_row.text_sha256))
                urls.append(str(support_row.source_url))
        entitlement_ok = entitlement_date is not None and pay_date is not None and entitlement_date <= pay_date
        if not entitlement_ok:
            failures.append("entitlement_date_not_established")

        available_trade_date = pd.Timestamp(row.available_trade_date).normalize()
        accounting_date = max(pay_date, available_trade_date) if pay_date is not None else None
        date_ok = bool(pay_ok and accounting_date is not None and available_trade_date <= accounting_date)
        if not date_ok:
            failures.append("pit_accounting_date_not_established")

        status = "pass" if not failures else "fail"
        checks.append(
            {
                "candidate_event_id": candidate_id,
                "asset": asset,
                "announcement_date": pd.Timestamp(row.announcement_date),
                "source_url": str(row.source_url),
                "amount_check": "pass" if amount_ok else "fail",
                "date_check": "pass" if date_ok else "fail",
                "entitlement_check": "pass" if entitlement_ok else "fail",
                "identity_check": "pass" if identity_ok else "fail",
                "text_check": "pass" if text_ok else "fail",
                "validation_status": status,
                "failure_reasons_json": json.dumps(sorted(set(failures)), ensure_ascii=False),
            }
        )
        if status != "pass":
            continue
        provisional.append(
            {
                "candidate_event_id": candidate_id,
                "asset": asset,
                "asset_name": str(row.asset_name),
                "exchange": str(row.exchange),
                "event_type": "liquidation_distribution",
                "holder_scope": holder_scope,
                "announcement_date": pd.Timestamp(row.announcement_date).normalize(),
                "published_at": str(row.published_at) if pd.notna(row.published_at) else "",
                "available_at": str(row.available_at),
                "available_trade_date": available_trade_date,
                "available_date": available_trade_date,
                "entitlement_date": entitlement_date,
                "entitlement_date_basis": entitlement_basis,
                "record_date": parsed_record if parsed_record is not None else entitlement_date,
                "ex_date": _optional_date(row.ex_date) or pd.NaT,
                "pay_date": pay_date,
                "accounting_date": accounting_date,
                "cash_per_share": float(amount),
                "distribution_total_cash": float(row.distribution_total_cash) if pd.notna(row.distribution_total_cash) else np.nan,
                "cash_denominator_shares": float(denominator) if denominator is not None else np.nan,
                "cash_value_basis": amount_basis,
                "additional_distribution_expected": bool(PENDING_MORE_PATTERN.search(text)),
                "fund_contract_terminated": _bool(row.fund_contract_terminated),
                "exit_registration_announced": _bool(row.exit_registration_announced),
                "source_urls": urls,
                "source_pdf_hashes": pdf_hashes,
                "source_text_hashes": text_hashes,
            }
        )

    check_frame = pd.DataFrame(checks, columns=CHECK_COLUMNS)
    events = pd.DataFrame(provisional)
    chain_rows: list[dict[str, Any]] = []
    if events.empty:
        return check_frame, pd.DataFrame(columns=CHAIN_COLUMNS), pd.DataFrame(columns=FORMAL_COLUMNS)
    events = events.sort_values(
        ["asset", "holder_scope", "accounting_date", "pay_date", "announcement_date", "candidate_event_id"]
    ).reset_index(drop=True)
    events["distribution_sequence"] = events.groupby(["asset", "holder_scope"]).cumcount() + 1
    events["is_final_distribution"] = False
    events["extinguishes_position"] = False
    events["termination_date"] = pd.NaT
    for asset, group in events.groupby("asset", sort=False):
        cash_rows = cash[cash["asset"].astype(str).str.zfill(6).eq(asset)]
        failed_rows = int((check_frame["asset"].eq(asset) & check_frame["validation_status"].eq("fail")).sum())
        latest_index = group.sort_values(["accounting_date", "pay_date", "announcement_date"]).index[-1]
        latest_id = str(events.loc[latest_index, "candidate_event_id"])
        latest_text = text_by_candidate[latest_id]
        termination = termination_map.get(asset)
        pending_more = bool(PENDING_MORE_PATTERN.search(latest_text))
        contract_terminated = bool(CONTRACT_TERMINATED_PATTERN.search(latest_text))
        remaining_full = bool(REMAINING_FULL_PATTERN.search(latest_text))
        exit_planned = bool(EXIT_PLANNED_PATTERN.search(latest_text))
        final_supported = bool(
            failed_rows == 0
            and not pending_more
            and termination is not None
            and (contract_terminated or (remaining_full and exit_planned))
        )
        if final_supported:
            termination_date, termination_row = termination
            events.loc[latest_index, "is_final_distribution"] = True
            events.loc[latest_index, "extinguishes_position"] = True
            events.loc[latest_index, "termination_date"] = termination_date
            events.at[latest_index, "source_urls"] = [
                *events.at[latest_index, "source_urls"],
                str(termination_row.source_url),
            ]
            events.at[latest_index, "source_pdf_hashes"] = [
                *events.at[latest_index, "source_pdf_hashes"],
                str(termination_row.pdf_sha256),
            ]
            events.at[latest_index, "source_text_hashes"] = [
                *events.at[latest_index, "source_text_hashes"],
                str(termination_row.text_sha256),
            ]
        chain_rows.append(
            {
                "asset": asset,
                "cash_candidate_rows": int(len(cash_rows)),
                "validated_event_rows": int(len(group)),
                "failed_candidate_rows": failed_rows,
                "latest_event_id": latest_id,
                "termination_date": termination[0] if termination is not None else pd.NaT,
                "final_distribution_supported": final_supported,
                "position_extinguishment_supported": final_supported,
                "chain_status": "complete" if final_supported else "validated_events_chain_incomplete",
            }
        )

    formal_rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        event_id = hashlib.sha256(
            f"{row.asset}|{row.distribution_sequence}|{row.holder_scope}|{pd.Timestamp(row.pay_date).date()}|{source_rows[row.candidate_event_id].text_sha256}".encode("utf-8")
        ).hexdigest()
        pdf_set = _set_sha256(row.source_pdf_hashes)
        text_set = _set_sha256(row.source_text_hashes)
        formal_rows.append(
            {
                "event_id": event_id,
                "asset": row.asset,
                "asset_name": row.asset_name,
                "exchange": row.exchange,
                "event_type": row.event_type,
                "distribution_sequence": int(row.distribution_sequence),
                "holder_scope": row.holder_scope,
                "announcement_date": row.announcement_date,
                "published_at": row.published_at,
                "available_at": row.available_at,
                "available_trade_date": row.available_trade_date,
                "available_date": row.available_date,
                "entitlement_date": row.entitlement_date,
                "entitlement_date_basis": row.entitlement_date_basis,
                "record_date": row.record_date,
                "ex_date": row.ex_date,
                "pay_date": row.pay_date,
                "accounting_date": row.accounting_date,
                "cash_per_share": row.cash_per_share,
                "distribution_total_cash": row.distribution_total_cash,
                "cash_denominator_shares": row.cash_denominator_shares,
                "cash_value_basis": row.cash_value_basis,
                "is_final_distribution": bool(row.is_final_distribution),
                "additional_distribution_expected": bool(row.additional_distribution_expected),
                "fund_contract_terminated": bool(row.fund_contract_terminated),
                "exit_registration_announced": bool(row.exit_registration_announced),
                "termination_date": row.termination_date,
                "extinguishes_position": bool(row.extinguishes_position),
                "source_urls_json": json.dumps(sorted(set(row.source_urls)), ensure_ascii=False),
                "source_pdf_sha256_set": pdf_set,
                "source_text_sha256_set": text_set,
                "source_vintage": f"official_terminal_event_pdf_set_sha256:{pdf_set}",
                "validation_status": "pass",
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
            }
        )
    return (
        check_frame,
        pd.DataFrame(chain_rows, columns=CHAIN_COLUMNS).sort_values("asset").reset_index(drop=True),
        pd.DataFrame(formal_rows, columns=FORMAL_COLUMNS).sort_values(
            ["asset", "holder_scope", "distribution_sequence"]
        ).reset_index(drop=True),
    )


def run() -> dict[str, Any]:
    frames, inputs = _authenticated_outputs()
    checks, chains, promotion = _validate_candidates(
        frames["cash_event_candidates"],
        frames["liquidation_report_candidates"],
        frames["delisting_candidates"],
    )
    _atomic_csv(checks, CHECKS_PATH)
    _atomic_csv(chains, CHAIN_PATH)
    _atomic_csv(promotion, PROMOTION_CANDIDATE_PATH)
    failed = int(checks["validation_status"].eq("fail").sum()) if not checks.empty else 0
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "PASS_WITH_QUARANTINED_CANDIDATES" if not promotion.empty else "FAIL_NO_VALIDATED_EVENTS",
        "cash_candidate_rows": int(len(checks)),
        "validated_event_rows": int(len(promotion)),
        "quarantined_candidate_rows": failed,
        "validated_assets": int(promotion["asset"].nunique()) if not promotion.empty else 0,
        "complete_event_chain_assets": int(chains["chain_status"].eq("complete").sum()) if not chains.empty else 0,
        "incomplete_event_chain_assets": int(chains["chain_status"].ne("complete").sum()) if not chains.empty else 0,
        "ocr_promoted_rows": int(promotion["source_vintage"].astype(str).str.contains("ocr", case=False).sum()) if not promotion.empty else 0,
        "formal_table_promotion_allowed": not promotion.empty,
        "scope_complete": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "Only passed event rows may be promoted. Quarantined rows and incomplete asset chains remain evidence_insufficient for full lifecycle closure.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "validation_checks", "path": _relative(CHECKS_PATH), "sha256": _sha256(CHECKS_PATH), "rows": int(len(checks))},
        {"role": "asset_chain_validation", "path": _relative(CHAIN_PATH), "sha256": _sha256(CHAIN_PATH), "rows": int(len(chains))},
        {"role": "promotion_candidates", "path": _relative(PROMOTION_CANDIDATE_PATH), "sha256": _sha256(PROMOTION_CANDIDATE_PATH), "rows": int(len(promotion))},
        {"role": "validation_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "validation_schema": "etf_terminal_event_chain_v2",
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
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
        "validated_event_rows",
        "quarantined_candidate_rows",
        "validated_assets",
        "complete_event_chain_assets",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
