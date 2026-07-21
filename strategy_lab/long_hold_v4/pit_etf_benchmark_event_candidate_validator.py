"""Independently triage official ETF benchmark-event candidates.

This layer deliberately does not promote benchmark history.  It reads the
authenticated reconciler and parser outputs, verifies every official text
payload again, and separates identity-changing evidence from changes to fees,
methodology, names, codes, or unrelated legal terms.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import pit_etf_benchmark_asset_candidate_reconciler as reconciler
from . import pit_etf_benchmark_document_candidate_parser as parser
from . import pit_etf_benchmark_document_windows_ocr as benchmark_ocr
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
RECONCILER_MANIFEST_PATH = reconciler.MANIFEST_PATH
PARSER_MANIFEST_PATH = parser.MANIFEST_PATH
OCR_MANIFEST_PATH = benchmark_ocr.MANIFEST_PATH
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
VALIDATION_PATH = OBSERVATION_DIR / "etf_benchmark_change_event_validation.csv"
CHAIN_PATH = OBSERVATION_DIR / "etf_benchmark_candidate_chain_validation.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_event_validation_review_queue.csv"
REPORT_PATH = (
    ROOT
    / "outputs"
    / "long_hold_v4"
    / "pit_validation"
    / "etf_benchmark_event_validation"
    / "validation_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_event_candidate_validator_latest.json"

SCHEMA_VERSION = 1
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05

DATE_TOKEN = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"
QUOTED_VALUE = r"[\u201c\"]([^\u201d\"]{2,240})[\u201d\"]"
INDEX_NAME_PAIR_PATTERNS = (
    re.compile(
        r"标的指数(?:名称)?(?:将)?由(?:原)?" + QUOTED_VALUE
        + r"(?:变更|更名|调整|更新)为" + QUOTED_VALUE
    ),
    re.compile(
        r"标的指数(?:名称)?(?:将)?由(?:原)?([^，。；]{2,100}?指数)"
        r"(?:变更|更名|调整|更新)为([^，。；]{2,100}?指数)"
    ),
)
PERFORMANCE_PAIR_PATTERNS = (
    re.compile(
        r"业绩比较基准(?:要素|内容)?(?:将)?由(?:原)?" + QUOTED_VALUE
        + r"(?:变更|调整|更新)为" + QUOTED_VALUE
    ),
    re.compile(
        r"业绩比较基准(?:要素|内容)?(?:将)?由(?:原)?([^。；]{2,220}?)"
        r"(?:变更|调整|更新)为([^。；]{2,220})"
    ),
)
INDEX_CODE_PAIR_PATTERN = re.compile(
    r"(?:标的)?指数代码(?:将)?由(?:原)?[\u201c\"]?([A-Z0-9.]{3,20})[\u201d\"]?"
    r"(?:变更|调整|更新)为[\u201c\"]?([A-Z0-9.]{3,20})[\u201d\"]?",
    re.IGNORECASE,
)
INDEX_CODE_NEW_PATTERN = re.compile(
    r"(?:标的)?指数代码(?:将)?(?:变更|调整|更新)为[\u201c\"]?([A-Z0-9.]{3,20})[\u201d\"]?",
    re.IGNORECASE,
)
INDEX_REPLACEMENT_NEW_PATTERNS = (
    re.compile(r"本基金(?:的)?标的指数将(?:变更|更换)为[\u201c\"]?([^\u201d\"，。；]{2,100}?指数)"),
    re.compile(r"将本基金(?:的)?标的指数(?:变更|更换)为[\u201c\"]?([^\u201d\"，。；]{2,100}?指数)"),
)
PERFORMANCE_NEW_PATTERNS = (
    re.compile(r"将本基金(?:的)?业绩比较基准(?:变更|调整|更新)为" + QUOTED_VALUE),
    re.compile(r"本基金(?:的)?业绩比较基准(?:将)?(?:变更|调整|更新)为" + QUOTED_VALUE),
)
PERFORMANCE_VALUE_CLAUSE_PATTERN = re.compile(
    r"本基金的?业绩比较基准(?:为|是|：|:)[：:]?([^。；]{2,240})"
)
INTERLEAVED_RMB_VALUATION_PAIR_PATTERN = re.compile(
    r"(?P<base>[^\s，。；]{2,80}?指数)[（(]人\s+"
    r"(?P=base)[（(]经估值[\s\S]{0,240}?"
    r"民币[）)]收益率\s+汇率调整[）)]收益率"
)
FUND_TRANSFORMATION_PAIR_PATTERN = re.compile(
    r"(?:基金名称|本基金)?(?:将)?由[\u201c\"]?([^\u201d\"。；]{2,120}?交易型开放式指数证券投资基金)"
    r"[\u201d\"]?(?:正式)?(?:转型|变更|更名)为[\u201c\"]?"
    r"([^\u201d\"。；]{2,120}?交易型开放式指数证券投资基金)"
)

LICENCE_SCOPE_PATTERN = re.compile(r"(?:标的)?指数(?:许可)?使用费|指数使用许可费|许可使用基点费")
METHODOLOGY_SCOPE_PATTERN = re.compile(
    r"标的指数(?:纳入北京证券交易所股票|调整样本|修订指数编制方案)|"
    r"指数编制方案(?:修订|调整)|纳入北京证券交易所股票"
)
SERVICE_METADATA_SCOPE_PATTERN = re.compile(r"基金(?:托管人|管理人)(?:名称)?更名")
INDEX_SHORT_NAME_SCOPE_PATTERN = re.compile(r"标的指数简称.{0,120}?(?:调整|变更|修改)")
PERFORMANCE_METADATA_ENRICHMENT_TITLE_PATTERN = re.compile(
    r"更新业绩比较基准(?:相关)?内容"
)
PERFORMANCE_METADATA_ENRICHMENT_BODY_PATTERN = re.compile(
    r"(?:设定原因|要素基本信息).{0,300}?(?:计算方法|管理投资偏离业绩比较基准)"
)
INTERLEAVED_VALUE_CONTAMINATION_PATTERN = re.compile(
    r"基金管理人|基金份额持有人|持有人大会|本基金的?业绩比较基准为|"
    r"中国证监会|基金合同当事人"
)
SAME_INDEX_IDENTITY_PATTERN = re.compile(
    r"除(?:标的)?指数名称外.{0,80}(?:编制方案|指数编制方案|指数代码).{0,60}(?:不变|保持不变)"
)
STRONG_EVENT_TITLE_PATTERN = re.compile(
    r"(?:变更|更换|调整|更新).{0,60}(?:标的指数|业绩比较基准)|"
    r"(?:标的指数|业绩比较基准).{0,60}(?:变更|更换|调整|更新)|"
    r"基金合同生效暨基金更名"
)
STRONG_EVENT_BODY_PATTERN = re.compile(
    r"决定(?:自20\d{2}年\d{1,2}月\d{1,2}日起)?(?:将)?"
    r"(?:变更|更换|调整|更新)(?:本基金(?:的)?)?(?:标的指数|业绩比较基准)|"
    r"决定(?:自20\d{2}年\d{1,2}月\d{1,2}日起)?(?:将)?"
    r"本基金(?:的)?(?:标的指数|业绩比较基准)(?:变更|更换|调整|更新)"
)
GENERIC_AMENDMENT_PATTERN = re.compile(r"基金合同|托管协议|招募说明书|持有人大会|转型")

VALIDATION_COLUMNS = [
    "asset",
    "asset_name",
    "announcement_date",
    "available_date",
    "announcement_title",
    "source_url",
    "document_key",
    "selection_reasons_json",
    "document_collection_status",
    "parse_status",
    "text_status",
    "event_class",
    "event_types_json",
    "validation_status",
    "old_index_name_candidate",
    "new_index_name_candidate",
    "old_index_code_candidate",
    "new_index_code_candidate",
    "old_performance_benchmark_candidate",
    "new_performance_benchmark_candidate",
    "old_fund_name_candidate",
    "new_fund_name_candidate",
    "effective_date_candidates_json",
    "event_effective_date_candidate",
    "observable_from_date_candidate",
    "evidence_snippets_json",
    "rules_fired_json",
    "document_scope_no_identity_change",
    "independent_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

CHAIN_COLUMNS = [
    "asset",
    "asset_name",
    "initial_reference_type_candidate",
    "initial_index_name_candidate",
    "initial_index_code_candidate",
    "initial_performance_benchmark_candidate",
    "governance_document_count",
    "structured_identity_event_count",
    "structured_performance_event_count",
    "identity_metadata_event_count",
    "fund_transformation_followup_count",
    "validated_non_identity_document_count",
    "ambiguous_document_count",
    "ocr_document_count",
    "candidate_chain_status",
    "independent_validation_status",
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
        raise ValueError(f"ETF benchmark event-validation lineage hash mismatch: {snapshot}")
    return snapshot


def _json_list(values: list[str] | set[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def _compact(value: Any) -> str:
    return (
        re.sub(r"\s+", "", "" if value is None else str(value))
        .replace("（", "(")
        .replace("）", ")")
        .replace("‘", "‘")
        .replace("’", "’")
    )


def _clean_candidate(value: str) -> str:
    cleaned = str(value).strip(" \t\r\n，,。；;：:‘’'\"“”()（）")
    cleaned = re.sub(r"^(?:原|即|本基金的?|跟踪的)", "", cleaned)
    return cleaned.strip(" \t\r\n，,。；;：:‘’'\"“”()（）")


def _clean_index_name_candidate(value: str) -> str:
    cleaned = _clean_candidate(value)
    cleaned = re.sub(r"\((?:简称|指数简称)[:：]?.*$", "", cleaned)
    return _clean_candidate(cleaned)


def _pair_from_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, str, str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            old_value = _clean_candidate(match.group(1))
            new_value = _clean_candidate(match.group(2))
            if old_value and new_value and old_value != new_value:
                return old_value, new_value, match.group(0)
    return "", "", ""


def _new_value_from_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = _clean_candidate(match.group(1))
            if value:
                return value, match.group(0)
    return "", ""


def _side_by_side_performance_pair(
    raw_text: str, compact_text: str
) -> tuple[str, str, str]:
    interleaved = INTERLEAVED_RMB_VALUATION_PAIR_PATTERN.search(raw_text)
    if interleaved:
        base = _clean_candidate(interleaved.group("base"))
        return (
            f"标的指数收益率，即{base}(人民币)收益率",
            f"标的指数收益率，即{base}(经估值汇率调整)收益率",
            interleaved.group(0),
        )
    if "修订前" not in compact_text or "修订后" not in compact_text:
        return "", "", ""
    values: list[tuple[str, str]] = []
    for match in PERFORMANCE_VALUE_CLAUSE_PATTERN.finditer(compact_text):
        value = _clean_candidate(match.group(1))
        if (
            not re.search(r"收益|利率|回报|价格|财富|存款", value)
            or len(value) > 180
            or INTERLEAVED_VALUE_CONTAMINATION_PATTERN.search(value)
            or value.count("(") != value.count(")")
        ):
            continue
        if values and value == values[-1][0]:
            continue
        values.append((value, match.group(0)))
    if len(values) < 2 or values[0][0] == values[1][0]:
        return "", "", ""
    return values[0][0], values[1][0], f"{values[0][1]}；{values[1][1]}"


def _iso_date(groups: tuple[str, str, str]) -> str:
    try:
        return pd.Timestamp(year=int(groups[0]), month=int(groups[1]), day=int(groups[2])).date().isoformat()
    except ValueError:
        return ""


def _contextual_effective_dates(text: str, announcement_date: str) -> tuple[list[str], list[str]]:
    dates: set[str] = set()
    dated_snippets: list[tuple[str, str]] = []
    patterns = (
        re.compile(
            r"(?:自|于)" + DATE_TOKEN
            + r"起.{0,180}?(?:标的指数|业绩比较基准|基金名称|本基金).{0,100}?"
            r"(?:变更|更换|调整|更新|更名|转型|生效)"
        ),
        re.compile(
            r"(?:标的指数|业绩比较基准|基金名称|本基金).{0,100}?"
            r"(?:变更|更换|调整|更新|更名|转型).{0,180}?(?:自|于)" + DATE_TOKEN + r"起"
        ),
        re.compile(r"(?:修订后|修改后|上述调整事项|本次调整事项).{0,100}?(?:自|于)" + DATE_TOKEN + r"起(?:生效|实施)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groups()[-3:]
            value = _iso_date(groups)
            if value:
                dates.add(value)
                dated_snippets.append((value, match.group(0)))
    if re.search(r"(?:上述调整事项|本次调整事项|本次修订).{0,60}自本公告发布之日起(?:生效|实施)", text):
        parsed = pd.to_datetime(announcement_date, errors="coerce")
        if not pd.isna(parsed):
            value = parsed.date().isoformat()
            dates.add(value)
            dated_snippets.append((value, "自本公告发布之日起生效"))
    announced = pd.to_datetime(announcement_date, errors="coerce")
    if not pd.isna(announced):
        lower = announced - pd.Timedelta(days=60)
        upper = announced + pd.Timedelta(days=730)
        dates = {
            value
            for value in dates
            if lower <= pd.Timestamp(value) <= upper
        }
    snippets = [snippet for value, snippet in dated_snippets if value in dates]
    return sorted(dates), snippets[:4]


def _direct_event_effective_dates(text: str, announcement_date: str) -> tuple[list[str], list[str]]:
    event_field = r"(?:标的指数|业绩比较基准|指数代码)"
    event_action = r"(?:变更|更换|调整|更新)"
    strong_patterns = (
        re.compile(
            r"(?:本公司|基金管理人)[^。；]{0,40}?决定自" + DATE_TOKEN
            + r"起[^。；]{0,250}?(?:"
            + event_action + r"[^。；]{0,100}?" + event_field
            + r"|" + event_field + r"[^。；]{0,100}?" + event_action + r")"
        ),
        re.compile(
            r"(?:修改|修订)后(?:的)?[^。；]{0,80}?(?:基金合同|托管协议)"
            r"[^。；]{0,120}?(?:将)?自" + DATE_TOKEN + r"起(?:生效|实施)"
        ),
        re.compile(
            r"自" + DATE_TOKEN
            + r"起[，,]?(?:本基金|[^，。；]{2,120}?交易型开放式指数证券投资基金)"
            r"[^。；]{0,120}?(?:将)?正式(?:转型|转)为"
        ),
        re.compile(
            r"自" + DATE_TOKEN + r"起[，,]?(?:对)?(?:本基金|两只基金|上述基金)"
            r"[^。；]{0,180}?(?:"
            + event_action + r"[^。；]{0,100}?" + event_field
            + r"|" + event_field + r"[^。；]{0,100}?" + event_action + r")"
        ),
    )
    fallback_patterns = (
        re.compile(
            r"自" + DATE_TOKEN + r"起[，,]?本基金(?:的)?[^。；]{0,180}?(?:"
            + event_action + r"[^。；]{0,100}?" + event_field
            + r"|" + event_field + r"[^。；]{0,100}?" + event_action + r")"
        ),
        re.compile(
            event_field + r"[^。；]{0,120}?" + event_action
            + r"[^。；]{0,120}?(?:自|于)" + DATE_TOKEN + r"起"
        ),
    )
    announced = pd.to_datetime(announcement_date, errors="coerce")
    lower = announced - pd.Timedelta(days=60) if not pd.isna(announced) else pd.NaT
    upper = announced + pd.Timedelta(days=730) if not pd.isna(announced) else pd.NaT

    def collect(patterns: tuple[re.Pattern[str], ...]) -> list[tuple[str, str]]:
        matches: list[tuple[str, str]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = _iso_date(match.groups()[-3:])
                if not value:
                    continue
                timestamp = pd.Timestamp(value)
                if not pd.isna(lower) and not (lower <= timestamp <= upper):
                    continue
                matches.append((value, match.group(0)))
        return matches

    candidates = collect(strong_patterns)
    if not candidates:
        candidates = collect(fallback_patterns)
    dates = sorted({value for value, _ in candidates})
    snippets = [snippet for value, snippet in candidates if value in dates]
    return dates, snippets[:4]


def _observable_date(effective_date: str, available_date: str) -> str:
    effective = pd.to_datetime(effective_date, errors="coerce")
    available = pd.to_datetime(available_date, errors="coerce")
    if pd.isna(effective) or pd.isna(available):
        return ""
    return max(effective, available).date().isoformat()


def classify_event_document(
    *,
    text: str,
    title: str,
    announcement_date: str,
    available_date: str,
    text_status: str = "success",
) -> dict[str, Any]:
    """Classify one official document without turning it into a formal history fact."""

    ocr_derived = text_status == "ocr_derived_unvalidated"
    if text_status not in {"success", "ocr_derived_unvalidated"}:
        return {
            "event_class": "ocr_or_document_text_required",
            "event_types_json": _json_list({"text_unavailable"}),
            "validation_status": "blocked_text_unavailable",
            "old_index_name_candidate": "",
            "new_index_name_candidate": "",
            "old_index_code_candidate": "",
            "new_index_code_candidate": "",
            "old_performance_benchmark_candidate": "",
            "new_performance_benchmark_candidate": "",
            "old_fund_name_candidate": "",
            "new_fund_name_candidate": "",
            "effective_date_candidates_json": "[]",
            "event_effective_date_candidate": "",
            "observable_from_date_candidate": "",
            "evidence_snippets_json": "[]",
            "rules_fired_json": _json_list({"text_unavailable_fail_closed"}),
            "document_scope_no_identity_change": False,
            "independent_validation_status": "ocr_or_text_recovery_required",
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }

    compact = _compact(text)
    compact_title = _compact(title)
    evidence: list[str] = []
    rules: set[str] = set()

    old_name, new_name, name_snippet = _pair_from_patterns(compact, INDEX_NAME_PAIR_PATTERNS)
    old_name = _clean_index_name_candidate(old_name)
    new_name = _clean_index_name_candidate(new_name)
    old_performance, new_performance, performance_snippet = _pair_from_patterns(
        compact, PERFORMANCE_PAIR_PATTERNS
    )
    if not old_performance and not new_performance:
        old_performance, new_performance, performance_snippet = _side_by_side_performance_pair(
            text, compact
        )
        if performance_snippet:
            rules.add("side_by_side_performance_benchmark_pair")
    code_match = INDEX_CODE_PAIR_PATTERN.search(compact)
    old_code = _clean_candidate(code_match.group(1)) if code_match else ""
    new_code = _clean_candidate(code_match.group(2)) if code_match else ""
    code_snippet = code_match.group(0) if code_match else ""
    if not new_code:
        code_new_match = INDEX_CODE_NEW_PATTERN.search(compact)
        if code_new_match:
            new_code = _clean_candidate(code_new_match.group(1))
            code_snippet = code_new_match.group(0)
    replacement_new, replacement_snippet = _new_value_from_patterns(
        compact, INDEX_REPLACEMENT_NEW_PATTERNS
    )
    replacement_new = _clean_index_name_candidate(replacement_new)
    if replacement_new and not new_name:
        new_name = replacement_new
    if replacement_snippet:
        rules.add("explicit_index_replacement_statement")
        evidence.append(replacement_snippet)
    performance_new, performance_new_snippet = _new_value_from_patterns(
        compact, PERFORMANCE_NEW_PATTERNS
    )
    if performance_new and not new_performance:
        new_performance = performance_new
    transformation_match = FUND_TRANSFORMATION_PAIR_PATTERN.search(compact)
    old_fund = _clean_candidate(transformation_match.group(1)) if transformation_match else ""
    new_fund = _clean_candidate(transformation_match.group(2)) if transformation_match else ""
    substantive_transformation = bool(
        transformation_match
        and re.search(r"正式(?:转型|转)为|实施转型|本基金将转型为|法律文件失效.{0,100}法律文件生效", compact)
    )

    for snippet, rule in (
        (name_snippet, "explicit_index_name_pair"),
        (performance_snippet, "explicit_performance_benchmark_pair"),
        (performance_new_snippet, "explicit_performance_benchmark_new_value"),
        (code_snippet, "explicit_index_code_change"),
        (
            transformation_match.group(0) if substantive_transformation else "",
            "explicit_fund_transformation_pair",
        ),
        (
            transformation_match.group(0) if transformation_match and not substantive_transformation else "",
            "explicit_fund_name_pair",
        ),
    ):
        if snippet:
            rules.add(rule)
            evidence.append(snippet)

    contextual_dates, contextual_date_snippets = _contextual_effective_dates(compact, announcement_date)
    direct_dates, direct_date_snippets = _direct_event_effective_dates(compact, announcement_date)
    effective_dates = direct_dates or contextual_dates
    date_snippets = direct_date_snippets or contextual_date_snippets
    evidence.extend(date_snippets)
    if effective_dates:
        rules.add("contextual_effective_date")
    if direct_dates:
        rules.add("direct_event_effective_date")
    effective_date = effective_dates[0] if len(effective_dates) == 1 else ""
    observable_date = _observable_date(effective_date, available_date)
    identity_event_text = LICENCE_SCOPE_PATTERN.sub("", compact)
    identity_event_title = LICENCE_SCOPE_PATTERN.sub("", compact_title)
    explicit_event_language = bool(
        STRONG_EVENT_TITLE_PATTERN.search(identity_event_title)
        or STRONG_EVENT_BODY_PATTERN.search(identity_event_text)
    )
    identity_unchanged = bool(SAME_INDEX_IDENTITY_PATTERN.search(compact))
    licence_scope = bool(LICENCE_SCOPE_PATTERN.search(compact_title) and LICENCE_SCOPE_PATTERN.search(compact))
    methodology_scope = bool(
        METHODOLOGY_SCOPE_PATTERN.search(compact_title) and METHODOLOGY_SCOPE_PATTERN.search(compact)
    )
    service_scope = bool(
        SERVICE_METADATA_SCOPE_PATTERN.search(compact_title)
        and SERVICE_METADATA_SCOPE_PATTERN.search(compact)
    )
    short_name_scope = bool(
        INDEX_SHORT_NAME_SCOPE_PATTERN.search(compact_title)
        and INDEX_SHORT_NAME_SCOPE_PATTERN.search(compact)
    )
    performance_metadata_enrichment = bool(
        PERFORMANCE_METADATA_ENRICHMENT_TITLE_PATTERN.search(compact_title)
        and PERFORMANCE_METADATA_ENRICHMENT_BODY_PATTERN.search(compact)
    )

    event_types: set[str] = set()
    if replacement_snippet:
        event_types.add("index_replacement")
    if old_name and new_name:
        event_types.add("index_name_change")
    if old_performance or performance_new:
        event_types.add("performance_benchmark_change")
    if new_code:
        event_types.add("index_code_change")
    if substantive_transformation:
        event_types.add("fund_transformation")

    if replacement_snippet:
        event_class = "index_replacement_candidate"
        complete_fields = bool(old_name and new_name and effective_date)
        validation_status = (
            "structured_event_candidate_chain_review_required"
            if complete_fields
            else "event_field_completion_required"
        )
        independent_status = "independent_text_rule_candidate_complete" if complete_fields else "independent_text_rule_candidate_incomplete"
    elif (old_name and new_name) and (old_performance or performance_new):
        event_class = (
            "index_identity_and_performance_metadata_change_candidate"
            if identity_unchanged
            else "index_name_and_performance_change_candidate"
        )
        complete_fields = bool(old_performance and new_performance and effective_date)
        validation_status = (
            "structured_event_candidate_chain_review_required"
            if complete_fields
            else "event_field_completion_required"
        )
        independent_status = "independent_text_rule_candidate_complete" if complete_fields else "independent_text_rule_candidate_incomplete"
        if identity_unchanged:
            rules.add("official_text_says_index_methodology_unchanged")
    elif old_performance or performance_new:
        event_class = "performance_benchmark_change_candidate"
        complete_fields = bool(old_performance and new_performance and effective_date)
        validation_status = (
            "structured_event_candidate_chain_review_required"
            if complete_fields
            else "event_field_completion_required"
        )
        independent_status = "independent_text_rule_candidate_complete" if complete_fields else "independent_text_rule_candidate_incomplete"
    elif old_name and new_name:
        event_class = "index_identity_metadata_change_candidate" if identity_unchanged else "index_name_change_candidate"
        complete_fields = bool(effective_date)
        validation_status = (
            "structured_metadata_candidate_chain_review_required"
            if complete_fields
            else "event_field_completion_required"
        )
        independent_status = "independent_text_rule_candidate_complete" if complete_fields else "independent_text_rule_candidate_incomplete"
        if identity_unchanged:
            rules.add("official_text_says_index_methodology_unchanged")
    elif new_code:
        event_class = "index_code_metadata_change_candidate"
        complete_fields = bool(old_code and new_code and effective_date)
        validation_status = (
            "structured_metadata_candidate_chain_review_required"
            if complete_fields
            else "event_field_completion_required"
        )
        independent_status = "independent_text_rule_candidate_complete" if complete_fields else "independent_text_rule_candidate_incomplete"
    elif substantive_transformation:
        event_class = "fund_transformation_benchmark_followup_required"
        validation_status = "post_transformation_legal_document_required"
        independent_status = "independent_text_rule_candidate_incomplete"
    elif short_name_scope:
        event_class = "index_short_name_metadata_change"
        event_types.add("document_scope_non_identity")
        validation_status = "document_scope_validated_non_identity_change"
        independent_status = "independent_document_scope_validated"
        rules.add("title_and_body_limit_change_to_index_short_name")
    elif performance_metadata_enrichment:
        event_class = "performance_benchmark_metadata_enrichment"
        event_types.add("document_scope_non_identity")
        validation_status = "document_scope_validated_non_identity_change"
        independent_status = "independent_document_scope_validated"
        rules.add("title_and_body_limit_change_to_performance_metadata_enrichment")
    elif licence_scope:
        event_class = "index_licence_or_fee_change"
        event_types.add("document_scope_non_identity")
        validation_status = "document_scope_validated_non_identity_change"
        independent_status = "independent_document_scope_validated"
        rules.add("title_and_body_limit_change_to_index_licence_or_fee")
    elif methodology_scope:
        event_class = "index_methodology_or_universe_change"
        event_types.add("document_scope_non_identity")
        validation_status = "document_scope_validated_non_identity_change"
        independent_status = "independent_document_scope_validated"
        rules.add("title_and_body_limit_change_to_index_methodology_or_universe")
    elif service_scope:
        event_class = "fund_service_metadata_change"
        event_types.add("document_scope_non_identity")
        validation_status = "document_scope_validated_non_identity_change"
        independent_status = "independent_document_scope_validated"
        rules.add("title_and_body_limit_change_to_fund_service_metadata")
    elif explicit_event_language:
        event_class = "ambiguous_benchmark_event_candidate"
        validation_status = "manual_event_field_review_required"
        independent_status = "independent_text_rule_ambiguous"
        rules.add("explicit_benchmark_event_language_without_structured_pair")
    elif GENERIC_AMENDMENT_PATTERN.search(compact_title):
        event_class = "generic_legal_or_holder_document"
        validation_status = "manual_scope_review_required"
        independent_status = "independent_text_rule_ambiguous"
        rules.add("generic_governance_document_not_safe_for_no_change_claim")
    else:
        event_class = "unresolved_governance_document"
        validation_status = "manual_scope_review_required"
        independent_status = "independent_text_rule_ambiguous"
        rules.add("governance_document_scope_unresolved")

    no_identity_change = validation_status == "document_scope_validated_non_identity_change"
    if ocr_derived:
        event_types.add("ocr_derived_evidence")
        rules.add("ocr_derived_text_cannot_validate_fields_without_original_page_review")
        validation_status = "ocr_derived_scope_candidate_page_review_required"
        independent_status = "ocr_derived_candidate_original_page_review_required"
        no_identity_change = False
    return {
        "event_class": event_class,
        "event_types_json": _json_list(event_types or {event_class}),
        "validation_status": validation_status,
        "old_index_name_candidate": old_name,
        "new_index_name_candidate": new_name,
        "old_index_code_candidate": old_code,
        "new_index_code_candidate": new_code,
        "old_performance_benchmark_candidate": old_performance,
        "new_performance_benchmark_candidate": new_performance,
        "old_fund_name_candidate": old_fund,
        "new_fund_name_candidate": new_fund,
        "effective_date_candidates_json": _json_list(effective_dates),
        "event_effective_date_candidate": effective_date,
        "observable_from_date_candidate": observable_date,
        "evidence_snippets_json": _json_list(evidence[:8]),
        "rules_fired_json": _json_list(rules),
        "document_scope_no_identity_change": no_identity_change,
        "independent_validation_status": independent_status,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def _authenticate_output(
    *, manifest_path: Path, qualification: str, role: str, producer_label: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qualification_status") != qualification or manifest.get("historical_backtest_allowed") is not False:
        raise ValueError(f"ETF benchmark {producer_label} does not authorize independent event validation")
    producer_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        producer_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    item = outputs.get(role, {})
    output_path = ROOT / str(item.get("path", ""))
    if not output_path.is_file() or _sha256(output_path) != str(item.get("sha256", "")):
        raise ValueError(f"ETF benchmark {producer_label} output hash mismatch: {role}")
    manifest_snapshot = _content_snapshot(manifest_path)
    output_snapshot = _content_snapshot(output_path)
    inputs = [
        {
            "role": f"{producer_label}_manifest_snapshot",
            "path": _relative(manifest_snapshot),
            "sha256": _sha256(manifest_snapshot),
        },
        {
            "role": f"{producer_label}_{role}_snapshot",
            "path": _relative(output_snapshot),
            "sha256": _sha256(output_snapshot),
        },
        {
            "role": f"authenticated_{producer_label}_code",
            "path": _relative(authenticated_code),
            "sha256": _sha256(authenticated_code),
        },
    ]
    return pd.read_csv(output_snapshot, dtype={"asset": str}, low_memory=False), inputs


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    changes, change_inputs = _authenticate_output(
        manifest_path=RECONCILER_MANIFEST_PATH,
        qualification="ASSET_CANDIDATES_RECONCILED_INDEPENDENT_VALIDATION_REQUIRED",
        role="benchmark_change_event_candidates",
        producer_label="reconciler_change",
    )
    initial, initial_inputs = _authenticate_output(
        manifest_path=RECONCILER_MANIFEST_PATH,
        qualification="ASSET_CANDIDATES_RECONCILED_INDEPENDENT_VALIDATION_REQUIRED",
        role="benchmark_initial_candidate_reconciliation",
        producer_label="reconciler_initial",
    )
    parsed, parser_inputs = _authenticate_output(
        manifest_path=PARSER_MANIFEST_PATH,
        qualification="HEURISTIC_DOCUMENT_CANDIDATES_INDEPENDENT_VALIDATION_REQUIRED",
        role="benchmark_document_parse_candidates",
        producer_label="parser",
    )
    ocr_sidecars, ocr_inputs = _authenticate_output(
        manifest_path=OCR_MANIFEST_PATH,
        qualification="OCR_SIDECAR_COMPLETE_REQUIRES_FIELD_VALIDATION",
        role="benchmark_document_ocr_index",
        producer_label="ocr",
    )
    if changes["document_key"].duplicated().any() or parsed["document_key"].duplicated().any():
        raise ValueError("ETF benchmark document keys must be unique for event validation")
    parsed_by_key = parsed.set_index("document_key", drop=False)
    missing = set(changes["document_key"]).difference(parsed_by_key.index)
    if missing:
        raise ValueError(f"ETF benchmark event documents are missing parser rows: {len(missing)}")
    if changes["historical_backtest_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("upstream change candidates unexpectedly authorize historical use")
    if ocr_sidecars["document_key"].duplicated().any():
        raise ValueError("ETF benchmark OCR document keys must be unique")
    if not ocr_sidecars["ocr_status"].eq("ocr_derived_unvalidated").all():
        raise ValueError("ETF benchmark OCR sidecars are incomplete")
    ocr_by_key = ocr_sidecars.set_index("document_key", drop=False)
    return (
        changes,
        initial,
        parsed_by_key,
        ocr_by_key,
        change_inputs + initial_inputs + parser_inputs + ocr_inputs,
    )


def _read_authenticated_text(row: Any, ocr_by_key: pd.DataFrame) -> tuple[str, str]:
    text_status = "success" if str(row.parse_status) not in parser.UNPARSED_STATUSES else str(row.parse_status)
    if text_status == "ocr_required":
        key = str(row.document_key)
        if key not in ocr_by_key.index:
            return "", text_status
        ocr_row = ocr_by_key.loc[key]
        if str(ocr_row.pdf_sha256) != str(row.pdf_sha256):
            raise ValueError(f"ETF benchmark OCR source PDF hash mismatch: {key}")
        ocr_path = ROOT / str(ocr_row.ocr_text_path)
        if not ocr_path.is_file() or _sha256(ocr_path) != str(ocr_row.ocr_text_sha256):
            raise ValueError(f"ETF benchmark OCR text hash mismatch: {key}")
        return ocr_path.read_text(encoding="utf-8"), "ocr_derived_unvalidated"
    if text_status != "success":
        return "", text_status
    text_path = ROOT / str(row.text_path)
    if not text_path.is_file() or _sha256(text_path) != str(row.text_sha256):
        raise ValueError(f"ETF benchmark text hash mismatch: {row.document_key}")
    return text_path.read_text(encoding="utf-8"), "success"


def _chain_record(initial_row: Any, rows: pd.DataFrame) -> dict[str, Any]:
    event_types = rows["event_types_json"].astype(str)
    structured_identity = int(event_types.str.contains('"index_replacement"', regex=False).sum())
    structured_performance = int(event_types.str.contains('"performance_benchmark_change"', regex=False).sum())
    identity_metadata = int(
        (
            event_types.str.contains('"index_name_change"', regex=False)
            | event_types.str.contains('"index_code_change"', regex=False)
        ).sum()
    )
    transformations = int(event_types.str.contains('"fund_transformation"', regex=False).sum())
    validated_non_identity = int(rows["document_scope_no_identity_change"].astype(bool).sum())
    ambiguous = int(
        rows["validation_status"].isin(
            {"manual_event_field_review_required", "manual_scope_review_required"}
        ).sum()
    )
    ocr = int(rows["text_status"].ne("success").sum())
    incomplete = int(rows["validation_status"].eq("event_field_completion_required").sum())
    structured = int(
        rows["validation_status"].isin(
            {
                "structured_event_candidate_chain_review_required",
                "structured_metadata_candidate_chain_review_required",
            }
        ).sum()
    )
    if ocr:
        chain_status = "blocked_ocr_or_text_recovery_required"
    elif transformations:
        chain_status = "post_transformation_legal_document_required"
    elif ambiguous:
        chain_status = "manual_governance_scope_review_required"
    elif incomplete:
        chain_status = "structured_event_fields_incomplete"
    elif structured:
        chain_status = "structured_events_chain_review_required"
    else:
        chain_status = "selected_governance_documents_no_identity_event_candidate"
    return {
        "asset": str(initial_row.asset).zfill(6),
        "asset_name": str(initial_row.asset_name),
        "initial_reference_type_candidate": str(initial_row.reference_type_candidate),
        "initial_index_name_candidate": str(initial_row.canonical_index_name_candidate),
        "initial_index_code_candidate": str(initial_row.canonical_index_code_candidate),
        "initial_performance_benchmark_candidate": str(initial_row.canonical_performance_benchmark_candidate),
        "governance_document_count": int(len(rows)),
        "structured_identity_event_count": structured_identity,
        "structured_performance_event_count": structured_performance,
        "identity_metadata_event_count": identity_metadata,
        "fund_transformation_followup_count": transformations,
        "validated_non_identity_document_count": validated_non_identity,
        "ambiguous_document_count": ambiguous,
        "ocr_document_count": ocr,
        "candidate_chain_status": chain_status,
        "independent_validation_status": "candidate_chain_not_promoted",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def run_validation() -> dict[str, Any]:
    changes, initial, parsed_by_key, ocr_by_key, inputs = _load_inputs()
    records: list[dict[str, Any]] = []
    for change in changes.itertuples(index=False):
        parsed_row = parsed_by_key.loc[str(change.document_key)]
        text, text_status = _read_authenticated_text(parsed_row, ocr_by_key)
        result = classify_event_document(
            text=text,
            title=str(change.announcement_title),
            announcement_date=str(change.announcement_date),
            available_date=str(change.available_date),
            text_status=text_status,
        )
        records.append(
            {
                "asset": str(change.asset).zfill(6),
                "asset_name": str(change.asset_name),
                "announcement_date": change.announcement_date,
                "available_date": change.available_date,
                "announcement_title": str(change.announcement_title),
                "source_url": str(change.source_url),
                "document_key": str(change.document_key),
                "selection_reasons_json": str(change.selection_reasons_json),
                "document_collection_status": str(change.document_collection_status),
                "parse_status": str(change.parse_status),
                "text_status": text_status,
                **result,
            }
        )
    validation = pd.DataFrame(records).reindex(columns=VALIDATION_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)

    groups = {asset: rows for asset, rows in validation.groupby("asset", sort=True)}
    empty = validation.iloc[0:0]
    chain = pd.DataFrame(
        [
            _chain_record(row, groups.get(str(row.asset).zfill(6), empty))
            for row in initial.itertuples(index=False)
        ]
    ).reindex(columns=CHAIN_COLUMNS).sort_values("asset").reset_index(drop=True)
    review = validation[validation["validation_status"] != "document_scope_validated_non_identity_change"].copy()

    _atomic_csv(validation, VALIDATION_PATH)
    _atomic_csv(chain, CHAIN_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)

    class_counts = {str(key): int(value) for key, value in validation["event_class"].value_counts().items()}
    status_counts = {
        str(key): int(value) for key, value in validation["validation_status"].value_counts().items()
    }
    chain_counts = {str(key): int(value) for key, value in chain["candidate_chain_status"].value_counts().items()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "INDEPENDENT_EVENT_TRIAGE_COMPLETE_HISTORY_CHAIN_REVIEW_REQUIRED",
        "event_documents": int(len(validation)),
        "native_text_documents": int(validation["text_status"].eq("success").sum()),
        "ocr_derived_documents": int(validation["text_status"].eq("ocr_derived_unvalidated").sum()),
        "unavailable_text_documents": int(
            (~validation["text_status"].isin({"success", "ocr_derived_unvalidated"})).sum()
        ),
        "event_class_counts": class_counts,
        "validation_status_counts": status_counts,
        "candidate_chain_status_counts": chain_counts,
        "document_scope_validated_non_identity_documents": int(
            validation["document_scope_no_identity_change"].astype(bool).sum()
        ),
        "structured_event_candidates": int(
            validation["validation_status"].isin(
                {
                    "structured_event_candidate_chain_review_required",
                    "structured_metadata_candidate_chain_review_required",
                }
            ).sum()
        ),
        "event_field_completion_required_documents": int(
            validation["validation_status"].eq("event_field_completion_required").sum()
        ),
        "manual_review_documents": int(
            validation["validation_status"].isin(
                {
                    "manual_event_field_review_required",
                    "manual_scope_review_required",
                    "ocr_derived_scope_candidate_page_review_required",
                }
            ).sum()
        ),
        "independent_validation_complete": False,
        "formal_history_rows": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Deterministic second-pass triage over hash-authenticated official text. A validated document scope is not "
            "an asset-wide no-change claim. Event candidates require old/new/effective-date continuity and independent "
            "legal-document confirmation before any historical interval is promoted."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {
            "role": "benchmark_change_event_validation",
            "path": _relative(VALIDATION_PATH),
            "sha256": _sha256(VALIDATION_PATH),
            "rows": int(len(validation)),
        },
        {
            "role": "benchmark_candidate_chain_validation",
            "path": _relative(CHAIN_PATH),
            "sha256": _sha256(CHAIN_PATH),
            "rows": int(len(chain)),
        },
        {
            "role": "benchmark_event_validation_review_queue",
            "path": _relative(REVIEW_QUEUE_PATH),
            "sha256": _sha256(REVIEW_QUEUE_PATH),
            "rows": int(len(review)),
        },
        {
            "role": "benchmark_event_validation_report",
            "path": _relative(REPORT_PATH),
            "sha256": _sha256(REPORT_PATH),
        },
    ]
    manifest = {
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {
                "path": _relative(Path(reconciler.__file__).resolve()),
                "sha256": _sha256(Path(reconciler.__file__).resolve()),
            },
            {
                "path": _relative(Path(parser.__file__).resolve()),
                "sha256": _sha256(Path(parser.__file__).resolve()),
            },
            {
                "path": _relative(Path(benchmark_ocr.__file__).resolve()),
                "sha256": _sha256(Path(benchmark_ocr.__file__).resolve()),
            },
            {
                "path": _relative(Path(source_code_archive.__file__).resolve()),
                "sha256": _sha256(Path(source_code_archive.__file__).resolve()),
            },
        ],
        "current_final_snapshot": True,
        "contains_validated_benchmark_history": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def main() -> None:
    result = run_validation()
    keys = (
        "qualification_status",
        "event_documents",
        "native_text_documents",
        "ocr_derived_documents",
        "unavailable_text_documents",
        "event_class_counts",
        "validation_status_counts",
        "structured_event_candidates",
        "manual_review_documents",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
