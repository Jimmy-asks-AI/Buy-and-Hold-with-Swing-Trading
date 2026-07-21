"""Parse review-only benchmark candidates from collected official ETF documents."""

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

from . import pit_etf_benchmark_document_collector as collector
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_MANIFEST_PATH = collector.MANIFEST_PATH
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
CANDIDATE_PATH = OBSERVATION_DIR / "etf_benchmark_document_parse_candidates.csv"
COVERAGE_PATH = OBSERVATION_DIR / "etf_benchmark_document_parse_coverage_registry.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_document_parse_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_benchmark_document_parser" / "parse_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_document_candidate_parser_latest.json"

SCHEMA_VERSION = 1
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
JOINT_INDEX_AND_PERFORMANCE_PATTERN = re.compile(
    r"本基金的?标的指数及业绩比较基准(?:为|是|：|:)([^，。；]{2,100}?指数)"
)
INDEX_NAME_PATTERNS = (
    re.compile(
        r"(?:本基金的?)?标的指数(?:为|是|：|:|，)?(?:指)?[^（(，。；]{2,120}?[（(]"
        r"([^）)，。；]{2,100}?指数)[）)]"
    ),
    JOINT_INDEX_AND_PERFORMANCE_PATTERN,
    re.compile(r"标的指数[“\"]([^”\"]{2,100}?指数)[”\"]"),
    re.compile(r"本基金主要采取指数化投资法投资于([^，。；]{2,100}?指数)(?:成份股|成分股)"),
    re.compile(
        r"本基金的?标的指数(?:为|是|：|:)[^，。；]{2,80}?指数系列中的"
        r"([^，。；]{2,100}?指数)"
    ),
    re.compile(
        r"本基金的?标的指数(?:为|是|：|:)(?:由)?[^，。；]{2,80}?"
        r"(?:编制并发布|发布)的([^，。；]{2,100}?指数)"
    ),
    re.compile(r"本基金的?标的指数(?:为|是|：|:)([^，。；]{2,100}?指数)"),
    re.compile(
        r"标的指数(?:为|是|：|:|，)?指(?:本基金跟踪的基准指数，?(?:是|即))?"
        r"(?:由)?[^，。；]{2,80}?(?:编制并发布|发布)的([^，。；]{2,100}?指数)"
    ),
    re.compile(r"标的指数(?:为|是|：|:|，)?指本基金跟踪的基准指数，?(?:是|即)([^，。；]{2,100}?指数)"),
    re.compile(r"标的指数(?:为|是|：|:|，)?指([^，。；]{2,100}?指数)"),
)
PERFORMANCE_BENCHMARK_PATTERNS = (
    re.compile(r"本基金的?业绩比较基准(?:为|是|：|:)([^。；]{2,240})"),
    re.compile(r"业绩比较基准(?:为|是|：|:)([^。；]{2,240})"),
)
INDEX_CODE_PATTERN = re.compile(
    r"(?<!全收益)(?<!净收益)(?:标的)?指数代码(?:为|是|：|:)?([A-Z0-9.]{3,20}|\d{6})"
)
DATE_PATTERN = re.compile(r"(?:自|于)?(20\d{2})年(\d{1,2})月(\d{1,2})日(?:起|生效|实施)?")
TRACKED_OBJECTIVE_PATTERN = re.compile(
    r"(?:投资目标|基金的投资目标).{0,120}?(?:紧密跟踪|跟踪)(?:其)?标的指数"
)
REPLICATION_OBJECTIVE_PATTERN = re.compile(
    r"(?:投资目标|基金的投资目标).{0,260}?"
    r"(?:指数化投资|对标的指数的有效跟踪|获得与标的指数相似|复制标的指数)"
)
ENHANCED_OBJECTIVE_PATTERN = re.compile(
    r"(?:投资目标|基金的投资目标).{0,320}?"
    r"(?:有效跟踪|跟踪偏离|跟踪误差).{0,220}?(?:超越|超额收益|增强)"
)
ENHANCED_IDENTITY_PATTERN = re.compile(r"(?:增强策略交易型开放式指数证券投资基金|增强型指数基金)")
ENHANCED_RETURN_PATTERN = re.compile(r"(?:超越标的指数|超越业绩比较基准|超额收益)")
TRACKED_DEFINITION_PATTERN = re.compile(
    r"标的指数(?:为|是|：|:|，)?指(?:本基金跟踪|由.{0,80}?(?:编制并发布|发布))"
)
NON_INDEX_TYPE_PATTERN = re.compile(
    r"(?:基金类型|基金类别|基金的类别|基金的类型)(?:与类别)?(?:为|是|：|:)?.{0,40}?"
    r"(?:货币市场基金|货币ETF|主动管理)"
)
COMMODITY_GOLD_IDENTITY_PATTERN = re.compile(r"(?:黄金|上海金)交易型开放式证券投资基金")
COMMODITY_GOLD_OBJECTIVE_PATTERN = re.compile(
    r"(?:投资目标|基金的投资目标).{0,320}?紧密跟踪.{0,160}?"
    r"(?:黄金|上海金|SHAU|业绩比较基准|价格表现)"
)
COMMODITY_REFERENCE_PATTERN = re.compile(r"紧密跟踪([^。；]{2,160}?(?:黄金|上海金|SHAU|价格表现)[^。；]{0,80})")
COMMODITY_BENCHMARK_HEADING_PATTERN = re.compile(
    r"(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)])业绩比较基准"
    r"(?:为|是|：|:)?(?:本基金的?业绩比较基准(?:为|是|：|:))?"
    r"([^。；]{2,240}?(?:基准价格收益率|收益率|收盘价|基准价格(?!收益率)))"
)
INDEX_FUND_IDENTITY_PATTERN = re.compile(r"交易型开放式(?:股票型)?指数证券投资基金")
EFFECTIVE_CONTEXT_PATTERN = re.compile(r"变更|更换|调整|修改|修订|生效|实施|启用")
BENCHMARK_EVENT_CONTEXT_PATTERN = re.compile(r"标的指数|业绩比较基准|指数名称|跟踪基准")
ANCHORS = ("投资目标", "标的指数", "业绩比较基准", "指数代码", "变更", "更换", "生效")
NON_INDEX_NAME_VALUES = {
    "本基金跟踪的基准指数",
    "中证指数",
    "国证指数",
    "投资标的的指数",
    "标的指数",
    "非标的指数",
}
NON_INDEX_NAME_MARKERS = {
    "基金合同",
    "招募说明书",
    "托管协议",
    "交易型开放式",
    "证券投资基金",
    "证券交易所停止向标的指数",
}
INDEX_NAME_PREFIX_PATTERN = re.compile(
    r"^(?:同期|即|因此选取|经估值汇率调整后的|经估值汇率调整的|经估值汇率调整后|"
    r"经人民币汇率调整后的|经人民币汇率调整的|经汇率调整后的|经汇率调整后|"
    r"标的指数收益率[，,]?即|标的指数[，,]?即)+"
)

CANDIDATE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "available_date",
    "announcement_title",
    "source_url",
    "document_key",
    "selection_priority",
    "selection_reasons_json",
    "baseline_selection_state",
    "document_format",
    "raw_document_path",
    "raw_document_sha256",
    "raw_document_bytes",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "document_collection_status",
    "reference_type_candidate",
    "index_name_candidates_json",
    "index_code_candidates_json",
    "performance_benchmark_candidates_json",
    "effective_date_candidates_json",
    "evidence_snippets_json",
    "rules_fired_json",
    "parse_status",
    "independent_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
UNPARSED_STATUSES = {"pending_document_collection", "document_collection_failed", "ocr_required"}
COVERAGE_COLUMNS = [
    "asset",
    "asset_name",
    "selected_document_count",
    "collected_document_count",
    "parsed_document_count",
    "tracked_index_candidate_count",
    "enhanced_index_candidate_count",
    "commodity_spot_reference_candidate_count",
    "non_index_reference_candidate_count",
    "unknown_reference_candidate_count",
    "index_name_candidate_document_count",
    "performance_benchmark_candidate_document_count",
    "pending_or_ocr_document_count",
    "asset_parse_state",
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
        raise ValueError(f"ETF benchmark parser lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _json_list(values: set[str] | list[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def _clean_candidate(value: str) -> str:
    cleaned = re.sub(r"^[：:，,、]+|[：:，,、]+$", "", str(value))
    cleaned = re.split(r"(?:如果|若|当|其中|本基金管理人|基金管理人)", cleaned, maxsplit=1)[0]
    return cleaned.strip()[:240]


def _extract_candidates(compact: str, patterns: tuple[re.Pattern[str], ...]) -> set[str]:
    values: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(compact):
            value = _clean_candidate(match.group(1))
            if 2 <= len(value) <= 240:
                values.add(value)
    return values


def _normalise_index_name_candidate(value: str) -> str | None:
    cleaned = _clean_candidate(value)
    if "即" in cleaned:
        suffix = _clean_candidate(cleaned.rsplit("即", maxsplit=1)[-1])
        if suffix.endswith("指数"):
            cleaned = suffix
    cleaned = _clean_candidate(INDEX_NAME_PREFIX_PATTERN.sub("", cleaned))
    if (
        not 2 <= len(cleaned) <= 80
        or not cleaned.endswith("指数")
        or cleaned in NON_INDEX_NAME_VALUES
        or cleaned.count("（") != cleaned.count("）")
        or cleaned.count("(") != cleaned.count(")")
        or cleaned.startswith(("注", "简称"))
        or any(marker in cleaned for marker in NON_INDEX_NAME_MARKERS)
        or any(marker in cleaned for marker in ("有限公司", "编制并发布", "发布的"))
    ):
        return None
    return cleaned


def _normalise_index_code_candidate(value: str) -> str | None:
    cleaned = _clean_candidate(value).upper()
    exchange_suffix = re.fullmatch(r"([A-Z0-9]{3,16})\.(?:CSI|SH|SZ|SSE|SZSE|SPI)", cleaned)
    if exchange_suffix:
        cleaned = exchange_suffix.group(1)
    if cleaned.isdigit() and len(cleaned) != 6:
        return None
    if re.fullmatch(r"X{2,}[A-Z0-9]*|X\d{1,5}", cleaned):
        return None
    return cleaned or None


def _extract_index_names(compact: str) -> set[str]:
    values: set[str] = set()
    for pattern in INDEX_NAME_PATTERNS:
        for match in pattern.finditer(compact):
            value = _normalise_index_name_candidate(match.group(1))
            trailing = compact[match.end(1) : match.end(1) + 20]
            if (
                value is None
                or value.startswith("由")
                or any(marker in value for marker in ("有限公司", "编制并发布", "发布的"))
                or trailing.startswith("有限公司")
                or trailing.startswith("系列中的")
            ):
                continue
            values.add(value)
    return values


def _extract_performance_benchmarks(compact: str) -> set[str]:
    values = {
        value
        for value in _extract_candidates(compact, PERFORMANCE_BENCHMARK_PATTERNS)
        if re.search(r"收益|利率|回报|净价|价格|财富|存款|组合", value)
    }
    for match in JOINT_INDEX_AND_PERFORMANCE_PATTERN.finditer(compact):
        value = _normalise_index_name_candidate(match.group(1))
        if value is not None:
            values.add(value)
    for match in COMMODITY_REFERENCE_PATTERN.finditer(compact):
        value = _clean_candidate(match.group(1))
        if re.search(r"黄金|上海金|SHAU|价格", value):
            values.add(value)
    for match in COMMODITY_BENCHMARK_HEADING_PATTERN.finditer(compact):
        value = _clean_candidate(match.group(1))
        if re.search(r"黄金|上海金|SHAU|AU99\.99", value, flags=re.IGNORECASE):
            values.add(value)
    return values


def _names_from_performance_benchmarks(values: set[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        for match in re.finditer(r"([^，。；:：]{2,120}?指数)收益率", value):
            candidate = _normalise_index_name_candidate(match.group(1))
            if candidate is not None:
                names.add(candidate)
    return names


def _evidence_snippets(compact: str, *, radius: int = 180, limit: int = 12) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    for anchor in ANCHORS:
        for match in re.finditer(re.escape(anchor), compact):
            snippet = compact[max(0, match.start() - radius) : min(len(compact), match.end() + radius)]
            if snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def _effective_dates(compact: str) -> set[str]:
    dates: set[str] = set()
    for match in DATE_PATTERN.finditer(compact):
        prior_boundaries = [compact.rfind(marker, 0, match.start()) for marker in "。；;"]
        sentence_start = max(prior_boundaries) + 1
        following_boundaries = [
            position
            for marker in "。；;"
            if (position := compact.find(marker, match.end())) >= 0
        ]
        sentence_end = min(following_boundaries) + 1 if following_boundaries else len(compact)
        context = compact[sentence_start:sentence_end]
        if EFFECTIVE_CONTEXT_PATTERN.search(context) and BENCHMARK_EVENT_CONTEXT_PATTERN.search(context):
            try:
                dates.add(pd.Timestamp(int(match.group(1)), int(match.group(2)), int(match.group(3))).date().isoformat())
            except ValueError:
                continue
    return dates


def parse_document_text(text: str, *, include_effective_dates: bool = True) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", str(text))
    code_text = re.sub(r"[^\S\r\n\f]+", "", str(text))
    index_names = _extract_index_names(compact)
    performance_benchmarks = _extract_performance_benchmarks(compact)
    index_names.update(_names_from_performance_benchmarks(performance_benchmarks))
    index_codes = {
        value
        for match in INDEX_CODE_PATTERN.finditer(code_text)
        if (value := _normalise_index_code_candidate(match.group(1))) is not None
    }
    tracked_objective = bool(TRACKED_OBJECTIVE_PATTERN.search(compact))
    replication_objective = bool(REPLICATION_OBJECTIVE_PATTERN.search(compact))
    enhanced_objective = bool(ENHANCED_OBJECTIVE_PATTERN.search(compact))
    enhanced_identity = bool(ENHANCED_IDENTITY_PATTERN.search(compact[:10000]))
    enhanced_return = bool(ENHANCED_RETURN_PATTERN.search(compact[:50000]))
    tracked_definition = bool(TRACKED_DEFINITION_PATTERN.search(compact))
    non_index_type = bool(NON_INDEX_TYPE_PATTERN.search(compact[:30000]))
    index_fund_identity = bool(INDEX_FUND_IDENTITY_PATTERN.search(compact[:10000]))
    commodity_identity = bool(COMMODITY_GOLD_IDENTITY_PATTERN.search(compact[:10000]))
    commodity_objective = bool(COMMODITY_GOLD_OBJECTIVE_PATTERN.search(compact[:50000]))
    commodity_benchmark = bool(
        performance_benchmarks
        and any(
            re.search(r"黄金|上海金|SHAU|AU99\.99", value, flags=re.IGNORECASE)
            for value in performance_benchmarks
        )
    )
    rules: set[str] = set()
    if tracked_objective:
        rules.add("tracked_investment_objective_clause")
    if replication_objective:
        rules.add("index_replication_objective_clause")
    if enhanced_objective:
        rules.add("enhanced_index_objective_clause")
    if enhanced_identity:
        rules.add("enhanced_index_legal_identity")
    if commodity_identity:
        rules.add("gold_etf_legal_identity")
    if commodity_objective:
        rules.add("commodity_spot_tracking_objective_clause")
    if commodity_benchmark:
        rules.add("commodity_spot_performance_benchmark_clause")
    if tracked_definition:
        rules.add("tracked_index_definition_clause")
    if index_names:
        rules.add("index_name_clause_candidate")
    if index_codes:
        rules.add("index_code_clause_candidate")
    if performance_benchmarks:
        rules.add("performance_benchmark_clause_candidate")
    if non_index_type:
        rules.add("non_index_fund_type_clause")
    if index_fund_identity:
        rules.add("index_fund_legal_identity")

    if enhanced_identity and (enhanced_objective or enhanced_return) and bool(index_names):
        reference_type = "enhanced_index"
        parse_status = "enhanced_index_candidate_review_required"
    elif (tracked_objective or replication_objective) and (tracked_definition or bool(index_names)):
        reference_type = "tracked_index"
        parse_status = "tracked_index_candidate_review_required"
    elif non_index_type and performance_benchmarks and not tracked_objective:
        reference_type = "non_index_reference"
        parse_status = "non_index_reference_candidate_review_required"
    elif commodity_identity and (commodity_objective or commodity_benchmark):
        reference_type = "commodity_spot_reference"
        parse_status = "commodity_spot_reference_candidate_review_required"
    else:
        reference_type = "unknown"
        parse_status = (
            "ambiguous_candidate_review_required"
            if index_names or performance_benchmarks or tracked_objective or replication_objective or index_fund_identity
            else "no_candidate_review_required"
        )
    return {
        "reference_type_candidate": reference_type,
        "index_name_candidates_json": _json_list(index_names),
        "index_code_candidates_json": _json_list(index_codes),
        "performance_benchmark_candidates_json": _json_list(performance_benchmarks),
        "effective_date_candidates_json": _json_list(_effective_dates(compact)) if include_effective_dates else "[]",
        "evidence_snippets_json": json.dumps(_evidence_snippets(compact), ensure_ascii=False),
        "rules_fired_json": _json_list(rules),
        "parse_status": parse_status,
    }


def _authenticate_collection() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(COLLECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        not in {
            "PARTIAL_SELECTED_OFFICIAL_DOCUMENT_SET",
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_COMPLETE",
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_OCR_REQUIRED",
        }
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF benchmark document collection is not an authenticated parse input")
    producer_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        producer_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    item = outputs.get("benchmark_document_index", {})
    index_path = ROOT / str(item.get("path", ""))
    if not index_path.is_file() or _sha256(index_path) != str(item.get("sha256", "")):
        raise ValueError("ETF benchmark document index hash mismatch")
    manifest_snapshot = _content_snapshot(COLLECTOR_MANIFEST_PATH)
    index_snapshot = _content_snapshot(index_path)
    inputs = [
        {"role": "collector_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)},
        {"role": "collector_index_snapshot", "path": _relative(index_snapshot), "sha256": _sha256(index_snapshot)},
        {"role": "authenticated_collector_code", "path": _relative(authenticated_code), "sha256": _sha256(authenticated_code)},
    ]
    frame = pd.read_csv(index_snapshot, dtype={"asset": str}, low_memory=False)
    missing = sorted(set(collector.DOCUMENT_INDEX_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"ETF benchmark document index misses columns: {missing}")
    if frame["document_key"].duplicated().any():
        raise ValueError("ETF benchmark document index has duplicate document keys")
    if frame["historical_backtest_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("ETF benchmark document index unexpectedly authorizes historical use")
    return frame, inputs


def run_parser() -> dict[str, Any]:
    document_index, inputs = _authenticate_collection()
    records: list[dict[str, Any]] = []
    text_inputs: list[dict[str, Any]] = []
    for row in document_index.itertuples(index=False):
        base = {
            "asset": str(row.asset).zfill(6),
            "asset_name": str(row.asset_name),
            "exchange": str(row.exchange),
            "announcement_date": row.announcement_date,
            "available_date": row.available_date,
            "announcement_title": str(row.announcement_title),
            "source_url": str(row.source_url),
            "document_key": str(row.document_key),
            "selection_priority": str(row.selection_priority),
            "selection_reasons_json": str(row.selection_reasons_json),
            "baseline_selection_state": str(row.baseline_selection_state),
            "document_format": str(row.document_format),
            "raw_document_path": str(row.raw_document_path),
            "raw_document_sha256": str(row.raw_document_sha256),
            "raw_document_bytes": row.raw_document_bytes,
            "pdf_path": str(row.pdf_path),
            "pdf_sha256": str(row.pdf_sha256),
            "text_path": str(row.text_path),
            "text_sha256": str(row.text_sha256),
            "document_collection_status": str(row.collection_status),
        }
        if str(row.collection_status) == "failed":
            parsed = {
                "reference_type_candidate": "unknown",
                "index_name_candidates_json": "[]",
                "index_code_candidates_json": "[]",
                "performance_benchmark_candidates_json": "[]",
                "effective_date_candidates_json": "[]",
                "evidence_snippets_json": "[]",
                "rules_fired_json": "[]",
                "parse_status": "document_collection_failed",
            }
        elif str(row.collection_status) != "success":
            parsed = {
                "reference_type_candidate": "unknown",
                "index_name_candidates_json": "[]",
                "index_code_candidates_json": "[]",
                "performance_benchmark_candidates_json": "[]",
                "effective_date_candidates_json": "[]",
                "evidence_snippets_json": "[]",
                "rules_fired_json": "[]",
                "parse_status": "pending_document_collection",
            }
        elif str(row.text_status) != "success":
            parsed = {
                "reference_type_candidate": "unknown",
                "index_name_candidates_json": "[]",
                "index_code_candidates_json": "[]",
                "performance_benchmark_candidates_json": "[]",
                "effective_date_candidates_json": "[]",
                "evidence_snippets_json": "[]",
                "rules_fired_json": "[]",
                "parse_status": "ocr_required",
            }
        else:
            text_path = ROOT / str(row.text_path)
            if not text_path.is_file() or _sha256(text_path) != str(row.text_sha256):
                raise ValueError(f"ETF benchmark native text hash mismatch: {row.document_key}")
            text = text_path.read_text(encoding="utf-8")
            selection_reasons = set(json.loads(str(row.selection_reasons_json)))
            include_effective_dates = bool(
                selection_reasons.intersection(
                    {
                        "all_title_routed_benchmark_change_documents",
                        "all_contract_amendments",
                        "all_holder_resolutions",
                    }
                )
            )
            parsed = parse_document_text(text, include_effective_dates=include_effective_dates)
            text_inputs.append(
                {"role": f"parsed_native_text:{row.asset}:{str(row.document_key)[:16]}", "path": _relative(text_path), "sha256": _sha256(text_path)}
            )
        records.append(
            {
                **base,
                **parsed,
                "independent_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    candidates = pd.DataFrame(records).reindex(columns=CANDIDATE_COLUMNS).sort_values(
        ["selection_priority", "asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)

    coverage_records: list[dict[str, Any]] = []
    for asset, rows in candidates.groupby("asset", sort=True):
        parsed = ~rows["parse_status"].isin(UNPARSED_STATUSES)
        pending = ~parsed
        if pending.any():
            state = "document_collection_or_ocr_incomplete"
        elif rows["reference_type_candidate"].eq("unknown").all():
            state = "parsed_no_resolved_reference_candidate"
        else:
            state = "parsed_candidates_independent_validation_required"
        coverage_records.append(
            {
                "asset": asset,
                "asset_name": str(rows.iloc[0]["asset_name"]),
                "selected_document_count": int(len(rows)),
                "collected_document_count": int(
                    (~rows["parse_status"].isin({"pending_document_collection", "document_collection_failed"})).sum()
                ),
                "parsed_document_count": int(parsed.sum()),
                "tracked_index_candidate_count": int(rows["reference_type_candidate"].eq("tracked_index").sum()),
                "enhanced_index_candidate_count": int(rows["reference_type_candidate"].eq("enhanced_index").sum()),
                "commodity_spot_reference_candidate_count": int(
                    rows["reference_type_candidate"].eq("commodity_spot_reference").sum()
                ),
                "non_index_reference_candidate_count": int(rows["reference_type_candidate"].eq("non_index_reference").sum()),
                "unknown_reference_candidate_count": int(rows["reference_type_candidate"].eq("unknown").sum()),
                "index_name_candidate_document_count": int(rows["index_name_candidates_json"].ne("[]").sum()),
                "performance_benchmark_candidate_document_count": int(rows["performance_benchmark_candidates_json"].ne("[]").sum()),
                "pending_or_ocr_document_count": int(pending.sum()),
                "asset_parse_state": state,
                "independent_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    coverage = pd.DataFrame(coverage_records).reindex(columns=COVERAGE_COLUMNS).sort_values("asset").reset_index(drop=True)
    review_queue = candidates.copy()
    _atomic_csv(candidates, CANDIDATE_PATH)
    _atomic_csv(coverage, COVERAGE_PATH)
    _atomic_csv(review_queue, REVIEW_QUEUE_PATH)

    parsed_rows = ~candidates["parse_status"].isin(UNPARSED_STATUSES)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "HEURISTIC_DOCUMENT_CANDIDATES_INDEPENDENT_VALIDATION_REQUIRED",
        "selected_documents": int(len(candidates)),
        "parsed_documents": int(parsed_rows.sum()),
        "pending_collection_documents": int(candidates["parse_status"].eq("pending_document_collection").sum()),
        "failed_collection_documents": int(candidates["parse_status"].eq("document_collection_failed").sum()),
        "ocr_required_documents": int(candidates["parse_status"].eq("ocr_required").sum()),
        "tracked_index_candidate_documents": int(candidates["reference_type_candidate"].eq("tracked_index").sum()),
        "enhanced_index_candidate_documents": int(candidates["reference_type_candidate"].eq("enhanced_index").sum()),
        "commodity_spot_reference_candidate_documents": int(
            candidates["reference_type_candidate"].eq("commodity_spot_reference").sum()
        ),
        "non_index_reference_candidate_documents": int(candidates["reference_type_candidate"].eq("non_index_reference").sum()),
        "unknown_reference_candidate_documents": int(candidates["reference_type_candidate"].eq("unknown").sum()),
        "independent_review_queue_documents": int(len(review_queue)),
        "assets_with_any_resolved_reference_candidate": int(
            candidates[candidates["reference_type_candidate"].ne("unknown")]["asset"].nunique()
        ),
        "independent_validation_complete": False,
        "formal_history_rows": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Heuristic clause candidates only. Boilerplate can contain benchmark terms; every type, name, code, "
            "date, and interval requires independent document-level validation."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "benchmark_document_parse_candidates", "path": _relative(CANDIDATE_PATH), "sha256": _sha256(CANDIDATE_PATH), "rows": int(len(candidates))},
        {"role": "benchmark_document_parse_coverage", "path": _relative(COVERAGE_PATH), "sha256": _sha256(COVERAGE_PATH), "rows": int(len(coverage))},
        {"role": "benchmark_document_parse_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review_queue))},
        {"role": "parse_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        **report,
        "inputs": [*inputs, *text_inputs],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(collector.__file__).resolve()), "sha256": _sha256(Path(collector.__file__).resolve())},
            {"path": _relative(Path(source_code_archive.__file__).resolve()), "sha256": _sha256(Path(source_code_archive.__file__).resolve())},
        ],
        "current_final_snapshot": True,
        "contains_validated_benchmark_facts": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def main() -> None:
    result = run_parser()
    keys = (
        "qualification_status",
        "selected_documents",
        "parsed_documents",
        "pending_collection_documents",
        "failed_collection_documents",
        "tracked_index_candidate_documents",
        "enhanced_index_candidate_documents",
        "commodity_spot_reference_candidate_documents",
        "non_index_reference_candidate_documents",
        "unknown_reference_candidate_documents",
        "assets_with_any_resolved_reference_candidate",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
