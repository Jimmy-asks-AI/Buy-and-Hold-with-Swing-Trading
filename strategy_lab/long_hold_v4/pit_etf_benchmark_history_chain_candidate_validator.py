"""Validate candidate ETF benchmark histories across independent official documents."""

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
from . import pit_etf_benchmark_event_candidate_validator as event_validator
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
EVENT_MANIFEST_PATH = event_validator.MANIFEST_PATH
RECONCILER_MANIFEST_PATH = reconciler.MANIFEST_PATH
PARSER_MANIFEST_PATH = parser.MANIFEST_PATH
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
EVENT_CHAIN_PATH = OBSERVATION_DIR / "etf_benchmark_event_chain_candidate_validation.csv"
ASSET_CHAIN_PATH = OBSERVATION_DIR / "etf_benchmark_asset_history_chain_candidate_validation.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_history_chain_review_queue.csv"
REPORT_PATH = (
    ROOT
    / "outputs"
    / "long_hold_v4"
    / "pit_validation"
    / "etf_benchmark_history_chain"
    / "history_chain_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_history_chain_candidate_validator_latest.json"

SCHEMA_VERSION = 1
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
POST_EVENT_REASONS = {
    "one_first_post_event_fund_contract_document",
    "one_first_post_event_prospectus_document",
}
INITIAL_HISTORY_REASONS = {
    "one_preferred_initial_legal_document",
    "one_listing_only_initial_document",
    "one_fallback_updated_benchmark_document_requires_validation",
    "one_initial_prospectus_supplement_document",
    "one_earliest_updated_prospectus_fallback_document",
    "one_canonical_listing_context_document",
}
NON_FULL_LEGAL_PATTERN = re.compile(r"公告|提示|摘要|产品资料概要|法律意见|托管协议")
GENERIC_TRACKED_PERFORMANCE_PATTERN = re.compile(
    r"^(?:同期)?标的指数(?:同期)?(?:增长率|收益率)?$"
)
FALLBACK_POST_EVENT_DAYS = 180
TARGET_EVENT_TYPES = {
    "index_replacement",
    "index_name_change",
    "performance_benchmark_change",
    "index_code_change",
    "fund_transformation",
}
TARGET_EVENT_CLASSES = {"ambiguous_benchmark_event_candidate"}

EVENT_CHAIN_COLUMNS = [
    "asset",
    "asset_name",
    "announcement_date",
    "available_date",
    "announcement_title",
    "source_url",
    "document_key",
    "event_class",
    "event_types_json",
    "old_index_name_candidate",
    "new_index_name_candidate",
    "old_index_code_candidate",
    "new_index_code_candidate",
    "old_performance_benchmark_candidate",
    "new_performance_benchmark_candidate",
    "effective_date_candidates_json",
    "event_effective_date_candidate",
    "observable_from_date_candidate",
    "prior_index_name_candidate",
    "prior_index_code_candidate",
    "prior_performance_benchmark_candidate",
    "prior_candidate_scope_from",
    "prior_index_name_scope_from",
    "prior_index_code_scope_from",
    "prior_performance_benchmark_scope_from",
    "prior_state_continuity_status",
    "post_event_reference_document_count",
    "post_event_fund_contract_document_count",
    "post_event_prospectus_document_count",
    "post_event_reference_document_keys_json",
    "post_event_index_name_candidates_json",
    "post_event_index_code_candidates_json",
    "post_event_performance_benchmark_candidates_json",
    "post_event_inferred_new_index_name_candidate",
    "post_event_inferred_new_performance_benchmark_candidate",
    "post_event_value_inference_status",
    "new_index_name_confirmation_count",
    "new_index_code_confirmation_count",
    "new_performance_benchmark_confirmation_count",
    "event_chain_status",
    "independent_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

ASSET_CHAIN_COLUMNS = [
    "asset",
    "asset_name",
    "initial_reference_type_candidate",
    "initial_index_name_candidate",
    "initial_index_code_candidate",
    "initial_performance_benchmark_candidate",
    "initial_candidate_scope_from",
    "history_initial_state_source",
    "history_initial_document_keys_json",
    "initial_reconciliation_status",
    "target_event_count",
    "cross_document_closed_event_count",
    "prior_state_inference_event_count",
    "post_legal_value_inference_event_count",
    "unresolved_event_count",
    "latest_index_name_candidate",
    "latest_index_code_candidate",
    "latest_performance_benchmark_candidate",
    "upstream_candidate_chain_status",
    "asset_history_candidate_status",
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
        raise ValueError(f"ETF benchmark history-chain lineage hash mismatch: {snapshot}")
    return snapshot


def _json_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return []
    parsed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, received {type(parsed).__name__}")
    return [str(item) for item in parsed]


def _json_list(values: list[str] | set[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def _value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize(value: str, *, performance: bool = False) -> str:
    compact = re.sub(r"\s+", "", _value(value)).replace("（", "(").replace("）", ")").lower()
    compact = re.sub(r"[，,。；;：:\"'“”‘’]", "", compact)
    if not performance:
        compact = re.sub(r"\((?:简称|英文简称|指数代码)[:：]?[^)]*\)", "", compact)
    return compact


def _matches(target: str, candidate: str, *, performance: bool = False) -> bool:
    left = _normalize(target, performance=performance)
    right = _normalize(candidate, performance=performance)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.6


def _is_generic_tracked_performance(value: str) -> bool:
    return bool(GENERIC_TRACKED_PERFORMANCE_PATTERN.fullmatch(_normalize(value, performance=True)))


def _performance_matches_with_index_context(
    target: str,
    candidate: str,
    *,
    target_index: str,
    candidate_indexes: list[str] | set[str],
) -> bool:
    if _matches(target, candidate, performance=True):
        return True
    if not _is_generic_tracked_performance(candidate) or not target_index:
        return False
    return any(_matches(target_index, value) for value in candidate_indexes)


def _candidate_event_order_dates(events: pd.DataFrame) -> pd.Series:
    effective = events["event_effective_date_candidate"].fillna("").astype(str).str.strip()
    has_effective = effective.ne("") & ~effective.str.lower().isin({"nan", "nat", "none"})
    values = effective.where(has_effective, events["available_date"])
    return pd.to_datetime(values, errors="coerce")


def _field_scope(available_date: str, list_date: str) -> str:
    available = pd.to_datetime(available_date, errors="coerce")
    listed = pd.to_datetime(list_date, errors="coerce")
    if pd.isna(available):
        return ""
    if pd.isna(listed):
        return available.date().isoformat()
    return max(available, listed).date().isoformat()


def _initial_history_state(initial_row: Any, asset_rows: pd.DataFrame) -> dict[str, str]:
    list_date = _value(initial_row.list_date)
    if asset_rows.empty:
        candidates = asset_rows.copy()
    else:
        reasons = asset_rows["selection_reasons_json"].map(lambda value: set(_json_values(value)))
        candidates = asset_rows[
            reasons.map(lambda values: bool(values.intersection(INITIAL_HISTORY_REASONS)))
            & ~asset_rows["parse_status"].isin(parser.UNPARSED_STATUSES)
        ].copy()
        candidates["_available"] = pd.to_datetime(candidates["available_date"], errors="coerce")
        sort_columns = [
            column
            for column in ("_available", "announcement_date", "document_key")
            if column in candidates.columns
        ]
        candidates = candidates[candidates["_available"].notna()].sort_values(sort_columns)

    selected_keys: set[str] = set()

    def earliest(field: str) -> tuple[str, str]:
        for row in candidates.itertuples(index=False):
            values = sorted(set(_json_values(getattr(row, field))))
            if len(values) != 1:
                continue
            selected_keys.add(str(row.document_key))
            return values[0], _field_scope(_value(row.available_date), list_date)
        return "", ""

    name, name_scope = earliest("index_name_candidates_json")
    code, code_scope = earliest("index_code_candidates_json")
    performance, performance_scope = earliest("performance_benchmark_candidates_json")
    source = "earliest_pit_initial_documents" if name or performance else "reconciler_candidate_fallback"
    fallback_scope = _value(initial_row.tradable_scope_from_candidate)
    if not name:
        name = _value(initial_row.canonical_index_name_candidate)
        name_scope = fallback_scope
    if not code:
        code = _value(initial_row.canonical_index_code_candidate)
        code_scope = fallback_scope if code else ""
    if not performance:
        performance = _value(initial_row.canonical_performance_benchmark_candidate)
        performance_scope = fallback_scope if performance else ""
    scopes = [value for value in (name_scope, code_scope, performance_scope) if value]
    scope = max(scopes) if scopes else fallback_scope
    return {
        "index_name": name,
        "index_code": code,
        "performance_benchmark": performance,
        "index_name_scope_from": name_scope,
        "index_code_scope_from": code_scope,
        "performance_benchmark_scope_from": performance_scope,
        "scope_from": scope,
        "source": source,
        "document_keys_json": _json_list(selected_keys),
    }


def _confirmation_count(target: str, rows: pd.DataFrame, field: str, *, performance: bool = False) -> int:
    if not target or rows.empty:
        return 0
    count = 0
    for row in rows.itertuples(index=False):
        if any(_matches(target, value, performance=performance) for value in _json_values(getattr(row, field))):
            count += 1
    return count


def _performance_confirmation_count(
    target: str, target_index: str, rows: pd.DataFrame
) -> int:
    if not target or rows.empty:
        return 0
    count = 0
    for row in rows.itertuples(index=False):
        indexes = _json_values(row.index_name_candidates_json)
        if any(
            _performance_matches_with_index_context(
                target,
                value,
                target_index=target_index,
                candidate_indexes=indexes,
            )
            for value in _json_values(row.performance_benchmark_candidates_json)
        ):
            count += 1
    return count


def _all_candidates(rows: pd.DataFrame, field: str) -> set[str]:
    return {
        value
        for encoded in rows[field].tolist()
        for value in _json_values(encoded)
        if value
    }


def _authenticate_output(
    *, manifest_path: Path, qualification: str, role: str, producer_label: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qualification_status") != qualification or manifest.get("historical_backtest_allowed") is not False:
        raise ValueError(f"ETF benchmark {producer_label} does not authorize history-chain validation")
    code_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        code_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    item = outputs.get(role, {})
    output_path = ROOT / str(item.get("path", ""))
    if not output_path.is_file() or _sha256(output_path) != str(item.get("sha256", "")):
        raise ValueError(f"ETF benchmark {producer_label} output hash mismatch: {role}")
    manifest_snapshot = _content_snapshot(manifest_path)
    output_snapshot = _content_snapshot(output_path)
    header = pd.read_csv(output_snapshot, nrows=0).columns
    string_columns = {
        column: str
        for column in header
        if column == "asset" or "code_candidate" in column
    }
    frame = pd.read_csv(output_snapshot, dtype=string_columns, low_memory=False)
    return frame, [
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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    events, event_inputs = _authenticate_output(
        manifest_path=EVENT_MANIFEST_PATH,
        qualification="INDEPENDENT_EVENT_TRIAGE_COMPLETE_HISTORY_CHAIN_REVIEW_REQUIRED",
        role="benchmark_change_event_validation",
        producer_label="event_validator",
    )
    upstream_chain, chain_inputs = _authenticate_output(
        manifest_path=EVENT_MANIFEST_PATH,
        qualification="INDEPENDENT_EVENT_TRIAGE_COMPLETE_HISTORY_CHAIN_REVIEW_REQUIRED",
        role="benchmark_candidate_chain_validation",
        producer_label="event_chain",
    )
    initial, initial_inputs = _authenticate_output(
        manifest_path=RECONCILER_MANIFEST_PATH,
        qualification="ASSET_CANDIDATES_RECONCILED_INDEPENDENT_VALIDATION_REQUIRED",
        role="benchmark_initial_candidate_reconciliation",
        producer_label="reconciler",
    )
    parsed, parser_inputs = _authenticate_output(
        manifest_path=PARSER_MANIFEST_PATH,
        qualification="HEURISTIC_DOCUMENT_CANDIDATES_INDEPENDENT_VALIDATION_REQUIRED",
        role="benchmark_document_parse_candidates",
        producer_label="parser",
    )
    if set(initial["asset"].astype(str).str.zfill(6)) != set(upstream_chain["asset"].astype(str).str.zfill(6)):
        raise ValueError("ETF benchmark initial and event-chain asset sets differ")
    if events["historical_backtest_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("upstream event validation unexpectedly authorizes historical use")
    return events, upstream_chain, initial, parsed, event_inputs + chain_inputs + initial_inputs + parser_inputs


def _is_target_event(row: Any) -> bool:
    event_types = set(_json_values(row.event_types_json))
    return bool(event_types.intersection(TARGET_EVENT_TYPES) or str(row.event_class) in TARGET_EVENT_CLASSES)


def _full_legal_kind(row: Any) -> str:
    title = str(row.announcement_title)
    if NON_FULL_LEGAL_PATTERN.search(title):
        return ""
    if "基金合同" in title and "招募说明书" not in title and "生效" not in title:
        return "fund_contract"
    if "招募说明书" in title:
        return "prospectus"
    return ""


def _post_event_documents(asset_rows: pd.DataFrame, event_row: Any) -> pd.DataFrame:
    event_available = pd.to_datetime(event_row.available_date, errors="coerce")
    if pd.isna(event_available):
        return asset_rows.iloc[0:0]
    reasons = asset_rows["selection_reasons_json"].map(lambda value: set(_json_values(value)))
    post = asset_rows[reasons.map(lambda values: bool(values.intersection(POST_EVENT_REASONS)))].copy()
    post["_available"] = pd.to_datetime(post["available_date"], errors="coerce")
    post = post[
        post["_available"].ge(event_available)
        & post["_available"].le(event_available + pd.Timedelta(days=730))
        & ~post["parse_status"].isin(parser.UNPARSED_STATUSES)
    ]
    selected: list[pd.DataFrame] = []
    for reason in POST_EVENT_REASONS:
        subset = post[
            post["selection_reasons_json"].map(lambda value: reason in set(_json_values(value)))
        ]
        if not subset.empty:
            earliest = subset["_available"].min()
            selected.append(subset[subset["_available"].eq(earliest)])
    selected_frame = (
        pd.concat(selected, ignore_index=True).drop_duplicates("document_key")
        if selected
        else post.iloc[0:0]
    )
    explicit_kinds = {
        _full_legal_kind(row)
        for row in selected_frame.itertuples(index=False)
        if _full_legal_kind(row)
    }
    fallback = asset_rows.copy()
    fallback["_available"] = pd.to_datetime(fallback["available_date"], errors="coerce")
    fallback = fallback[
        fallback["_available"].ge(event_available)
        & fallback["_available"].le(
            event_available + pd.Timedelta(days=FALLBACK_POST_EVENT_DAYS)
        )
        & ~fallback["parse_status"].isin(parser.UNPARSED_STATUSES)
        & fallback["document_key"].astype(str).ne(str(event_row.document_key))
    ].copy()
    fallback["_legal_kind"] = [
        _full_legal_kind(row) for row in fallback.itertuples(index=False)
    ]
    fallback = fallback[fallback["_legal_kind"].ne("")]
    fallback_parts: list[pd.DataFrame] = []
    for kind in ("fund_contract", "prospectus"):
        if kind in explicit_kinds:
            continue
        kind_rows = fallback[fallback["_legal_kind"].eq(kind)].sort_values(
            ["_available", "announcement_date", "document_key"]
        )
        if not kind_rows.empty:
            fallback_parts.append(kind_rows.head(6))
    frames = [frame for frame in [selected_frame, *fallback_parts] if not frame.empty]
    if not frames:
        return post.iloc[0:0]
    return pd.concat(frames, ignore_index=True).drop_duplicates("document_key")


def _continuity_status(
    *,
    event_types: set[str],
    event_row: Any,
    prior_name: str,
    prior_code: str,
    prior_performance: str,
    prior_scope: str,
    prior_name_scope: str = "",
    prior_code_scope: str = "",
    prior_performance_scope: str = "",
) -> str:
    event_date = pd.to_datetime(
        _value(event_row.observable_from_date_candidate) or _value(event_row.available_date),
        errors="coerce",
    )
    relevant_scopes: list[str] = []
    if event_types.intersection({"index_replacement", "index_name_change", "fund_transformation"}) and prior_name:
        relevant_scopes.append(prior_name_scope or prior_scope)
    if "index_code_change" in event_types and prior_code:
        relevant_scopes.append(prior_code_scope or prior_scope)
    if "performance_benchmark_change" in event_types and prior_performance:
        relevant_scopes.append(prior_performance_scope or prior_scope)
    if any(
        not pd.isna(pd.to_datetime(scope, errors="coerce"))
        and not pd.isna(event_date)
        and pd.to_datetime(scope, errors="coerce") > event_date
        for scope in relevant_scopes
        if scope
    ):
        return "prior_candidate_not_point_in_time_available"
    checks: list[bool] = []
    inference_required = False
    if event_types.intersection({"index_replacement", "index_name_change"}):
        old_name = _value(event_row.old_index_name_candidate)
        if old_name:
            if prior_name:
                checks.append(_matches(old_name, prior_name))
            else:
                inference_required = True
        else:
            inference_required = True
    if "index_code_change" in event_types:
        old_code = _value(event_row.old_index_code_candidate)
        if old_code:
            if prior_code:
                checks.append(_matches(old_code, prior_code))
            else:
                inference_required = True
        else:
            inference_required = True
    if "performance_benchmark_change" in event_types:
        old_performance = _value(event_row.old_performance_benchmark_candidate)
        if old_performance:
            if prior_performance:
                checks.append(
                    _performance_matches_with_index_context(
                        old_performance,
                        prior_performance,
                        target_index=_value(event_row.old_index_name_candidate) or prior_name,
                        candidate_indexes=[prior_name],
                    )
                )
            else:
                inference_required = True
        else:
            inference_required = True
    if checks and not all(checks):
        return "prior_state_conflict"
    if inference_required:
        return "prior_state_inference_required"
    return "prior_state_fields_match"


def evaluate_event_candidate(
    event_row: Any,
    post_rows: pd.DataFrame,
    *,
    prior_name: str,
    prior_code: str,
    prior_performance: str,
    prior_scope: str,
    prior_name_scope: str = "",
    prior_code_scope: str = "",
    prior_performance_scope: str = "",
) -> dict[str, Any]:
    event_types = set(_json_values(event_row.event_types_json)).intersection(TARGET_EVENT_TYPES)
    continuity = _continuity_status(
        event_types=event_types,
        event_row=event_row,
        prior_name=prior_name,
        prior_code=prior_code,
        prior_performance=prior_performance,
        prior_scope=prior_scope,
        prior_name_scope=prior_name_scope,
        prior_code_scope=prior_code_scope,
        prior_performance_scope=prior_performance_scope,
    )
    new_name = _value(event_row.new_index_name_candidate)
    new_code = _value(event_row.new_index_code_candidate)
    new_performance = _value(event_row.new_performance_benchmark_candidate)
    name_confirmations = _confirmation_count(new_name, post_rows, "index_name_candidates_json")
    code_confirmations = _confirmation_count(new_code, post_rows, "index_code_candidates_json")
    performance_confirmations = _performance_confirmation_count(
        new_performance, new_name, post_rows
    )
    reason_sets = (
        post_rows["selection_reasons_json"].map(lambda value: set(_json_values(value)))
        if not post_rows.empty
        else pd.Series(dtype=object)
    )
    contract_document_count = int(
        reason_sets.map(
            lambda values: "one_first_post_event_fund_contract_document" in values
        ).sum()
    ) if not post_rows.empty else 0
    prospectus_document_count = int(
        reason_sets.map(
            lambda values: "one_first_post_event_prospectus_document" in values
        ).sum()
    ) if not post_rows.empty else 0
    post_names = _all_candidates(post_rows, "index_name_candidates_json") if not post_rows.empty else set()
    post_performance = (
        _all_candidates(post_rows, "performance_benchmark_candidates_json")
        if not post_rows.empty
        else set()
    )
    inferred_name = next(iter(post_names)) if len(post_names) == 1 else ""
    inferred_performance = next(iter(post_performance)) if len(post_performance) == 1 else ""
    pure_transformation = "fund_transformation" in event_types and not event_types.intersection(
        {"index_replacement", "index_name_change", "performance_benchmark_change"}
    )
    post_legal_inference_ready = bool(
        pure_transformation
        and len(post_rows) >= 2
        and contract_document_count >= 1
        and prospectus_document_count >= 1
        and prior_name
        and inferred_name
        and not _matches(prior_name, inferred_name)
        and inferred_performance
        and _confirmation_count(inferred_name, post_rows, "index_name_candidates_json") == len(post_rows)
        and _confirmation_count(
            inferred_performance,
            post_rows,
            "performance_benchmark_candidates_json",
            performance=True,
        ) == len(post_rows)
    )
    if not pure_transformation:
        post_legal_inference_status = "not_applicable"
    elif post_legal_inference_ready:
        post_legal_inference_status = "two_legal_document_types_agree_review_required"
    elif contract_document_count < 1 or prospectus_document_count < 1 or len(post_rows) < 2:
        post_legal_inference_status = "both_legal_document_types_required"
    elif not prior_name:
        post_legal_inference_status = "prior_index_name_required"
    elif len(post_names) != 1 or len(post_performance) != 1:
        post_legal_inference_status = "post_legal_candidates_not_unique"
    elif _matches(prior_name, inferred_name):
        post_legal_inference_status = "post_legal_index_does_not_change_prior_state"
    else:
        post_legal_inference_status = "post_legal_documents_do_not_both_confirm_values"
    effective_date = _value(event_row.event_effective_date_candidate)
    date_candidates = _json_values(event_row.effective_date_candidates_json)
    date_complete = bool(effective_date and len(date_candidates) == 1)

    required_fields: list[bool] = []
    if event_types.intersection({"index_replacement", "index_name_change"}):
        required_fields.append(bool(new_name and name_confirmations))
    if "index_code_change" in event_types:
        required_fields.append(bool(new_code and code_confirmations))
    if "performance_benchmark_change" in event_types:
        required_fields.append(bool(new_performance and performance_confirmations))
    if "fund_transformation" in event_types and not event_types.intersection(
        {"index_replacement", "index_name_change", "performance_benchmark_change"}
    ):
        required_fields.append(False)

    if not event_types:
        status = "event_fields_unresolved"
    elif continuity == "prior_candidate_not_point_in_time_available":
        status = "blocked_prior_candidate_not_point_in_time_available"
    elif continuity == "prior_state_conflict":
        status = "blocked_prior_state_conflict"
    elif not date_complete:
        status = "blocked_effective_date_missing_or_conflicting"
    elif post_legal_inference_ready:
        status = "candidate_event_closed_post_legal_value_inference_review_required"
    elif not post_rows.empty and required_fields and all(required_fields):
        status = (
            "candidate_event_closed_prior_state_inference_review_required"
            if continuity == "prior_state_inference_required"
            else "candidate_event_closed_cross_document"
        )
    elif post_rows.empty:
        status = "blocked_post_event_legal_document_missing"
    else:
        status = "blocked_new_value_not_confirmed_by_post_event_legal_document"

    return {
        "prior_state_continuity_status": continuity,
        "post_event_reference_document_count": int(len(post_rows)),
        "post_event_fund_contract_document_count": contract_document_count,
        "post_event_prospectus_document_count": prospectus_document_count,
        "post_event_reference_document_keys_json": _json_list(set(post_rows["document_key"].astype(str))),
        "post_event_index_name_candidates_json": _json_list(post_names),
        "post_event_index_code_candidates_json": _json_list(_all_candidates(post_rows, "index_code_candidates_json")),
        "post_event_performance_benchmark_candidates_json": _json_list(post_performance),
        "post_event_inferred_new_index_name_candidate": inferred_name if pure_transformation else "",
        "post_event_inferred_new_performance_benchmark_candidate": (
            inferred_performance if pure_transformation else ""
        ),
        "post_event_value_inference_status": post_legal_inference_status,
        "new_index_name_confirmation_count": name_confirmations,
        "new_index_code_confirmation_count": code_confirmations,
        "new_performance_benchmark_confirmation_count": performance_confirmations,
        "event_chain_status": status,
        "independent_validation_status": "candidate_chain_review_only",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def run_validation() -> dict[str, Any]:
    events, upstream_chain, initial, parsed, inputs = _load_inputs()
    events = events[events.apply(_is_target_event, axis=1)].copy()
    events["_order_date"] = _candidate_event_order_dates(events)
    events = events.sort_values(["asset", "_order_date", "available_date", "document_key"])
    parsed_groups = {
        str(asset).zfill(6): rows.copy()
        for asset, rows in parsed.groupby(parsed["asset"].astype(str).str.zfill(6), sort=True)
    }
    event_groups = {
        str(asset).zfill(6): rows.copy()
        for asset, rows in events.groupby(events["asset"].astype(str).str.zfill(6), sort=True)
    }
    upstream_lookup = upstream_chain.set_index(upstream_chain["asset"].astype(str).str.zfill(6))

    event_records: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    for initial_row in initial.sort_values("asset").itertuples(index=False):
        asset = str(initial_row.asset).zfill(6)
        asset_events = event_groups.get(asset, events.iloc[0:0])
        asset_parsed = parsed_groups.get(asset, parsed.iloc[0:0])
        initial_state = _initial_history_state(initial_row, asset_parsed)
        prior_name = initial_state["index_name"]
        prior_code = initial_state["index_code"]
        prior_performance = initial_state["performance_benchmark"]
        prior_name_scope = initial_state["index_name_scope_from"]
        prior_code_scope = initial_state["index_code_scope_from"]
        prior_performance_scope = initial_state["performance_benchmark_scope_from"]
        prior_scope = initial_state["scope_from"]
        asset_statuses: list[str] = []

        for event_row in asset_events.itertuples(index=False):
            post_rows = _post_event_documents(asset_parsed, event_row)
            result = evaluate_event_candidate(
                event_row,
                post_rows,
                prior_name=prior_name,
                prior_code=prior_code,
                prior_performance=prior_performance,
                prior_scope=prior_scope,
                prior_name_scope=prior_name_scope,
                prior_code_scope=prior_code_scope,
                prior_performance_scope=prior_performance_scope,
            )
            status = str(result["event_chain_status"])
            asset_statuses.append(status)
            event_records.append(
                {
                    "asset": asset,
                    "asset_name": str(event_row.asset_name),
                    "announcement_date": event_row.announcement_date,
                    "available_date": event_row.available_date,
                    "announcement_title": str(event_row.announcement_title),
                    "source_url": str(event_row.source_url),
                    "document_key": str(event_row.document_key),
                    "event_class": str(event_row.event_class),
                    "event_types_json": str(event_row.event_types_json),
                    "old_index_name_candidate": _value(event_row.old_index_name_candidate),
                    "new_index_name_candidate": _value(event_row.new_index_name_candidate),
                    "old_index_code_candidate": _value(event_row.old_index_code_candidate),
                    "new_index_code_candidate": _value(event_row.new_index_code_candidate),
                    "old_performance_benchmark_candidate": _value(event_row.old_performance_benchmark_candidate),
                    "new_performance_benchmark_candidate": _value(event_row.new_performance_benchmark_candidate),
                    "effective_date_candidates_json": str(event_row.effective_date_candidates_json),
                    "event_effective_date_candidate": _value(event_row.event_effective_date_candidate),
                    "observable_from_date_candidate": _value(event_row.observable_from_date_candidate),
                    "prior_index_name_candidate": prior_name,
                    "prior_index_code_candidate": prior_code,
                    "prior_performance_benchmark_candidate": prior_performance,
                    "prior_candidate_scope_from": prior_scope,
                    "prior_index_name_scope_from": prior_name_scope,
                    "prior_index_code_scope_from": prior_code_scope,
                    "prior_performance_benchmark_scope_from": prior_performance_scope,
                    **result,
                }
            )
            if status in {
                "candidate_event_closed_cross_document",
                "candidate_event_closed_prior_state_inference_review_required",
                "candidate_event_closed_post_legal_value_inference_review_required",
            }:
                new_name = _value(event_row.new_index_name_candidate)
                new_code = _value(event_row.new_index_code_candidate)
                new_performance = _value(event_row.new_performance_benchmark_candidate)
                if status == "candidate_event_closed_post_legal_value_inference_review_required":
                    new_name = _value(result["post_event_inferred_new_index_name_candidate"])
                    new_performance = _value(
                        result["post_event_inferred_new_performance_benchmark_candidate"]
                    )
                new_scope = _value(event_row.observable_from_date_candidate)
                if new_name:
                    prior_name_scope = new_scope or prior_name_scope
                if new_code:
                    prior_code_scope = new_scope or prior_code_scope
                if new_performance:
                    prior_performance_scope = new_scope or prior_performance_scope
                prior_name = new_name or prior_name
                prior_code = new_code or prior_code
                prior_performance = new_performance or prior_performance
                current_scopes = [
                    value
                    for value in (
                        prior_name_scope,
                        prior_code_scope,
                        prior_performance_scope,
                    )
                    if value
                ]
                prior_scope = max(current_scopes) if current_scopes else prior_scope

        closed = sum(status == "candidate_event_closed_cross_document" for status in asset_statuses)
        inferred = sum(
            status == "candidate_event_closed_prior_state_inference_review_required"
            for status in asset_statuses
        )
        post_legal_inferred = sum(
            status == "candidate_event_closed_post_legal_value_inference_review_required"
            for status in asset_statuses
        )
        unresolved = len(asset_statuses) - closed - inferred - post_legal_inferred
        upstream_status = _value(upstream_lookup.loc[asset, "candidate_chain_status"])
        if not asset_statuses:
            history_status = "no_target_event_candidate_history_not_formally_closed"
        elif unresolved:
            history_status = "candidate_history_chain_has_unresolved_events"
        elif post_legal_inferred:
            history_status = "candidate_history_chain_closed_with_post_legal_value_inference"
        elif inferred:
            history_status = "candidate_history_chain_closed_with_prior_state_inference"
        else:
            history_status = "candidate_history_chain_closed_initial_validation_required"
        asset_records.append(
            {
                "asset": asset,
                "asset_name": str(initial_row.asset_name),
                "initial_reference_type_candidate": _value(initial_row.reference_type_candidate),
                "initial_index_name_candidate": initial_state["index_name"],
                "initial_index_code_candidate": initial_state["index_code"],
                "initial_performance_benchmark_candidate": initial_state["performance_benchmark"],
                "initial_candidate_scope_from": initial_state["scope_from"],
                "history_initial_state_source": initial_state["source"],
                "history_initial_document_keys_json": initial_state["document_keys_json"],
                "initial_reconciliation_status": _value(initial_row.initial_reconciliation_status),
                "target_event_count": int(len(asset_statuses)),
                "cross_document_closed_event_count": int(closed),
                "prior_state_inference_event_count": int(inferred),
                "post_legal_value_inference_event_count": int(post_legal_inferred),
                "unresolved_event_count": int(unresolved),
                "latest_index_name_candidate": prior_name,
                "latest_index_code_candidate": prior_code,
                "latest_performance_benchmark_candidate": prior_performance,
                "upstream_candidate_chain_status": upstream_status,
                "asset_history_candidate_status": history_status,
                "independent_validation_status": "initial_mapping_and_event_chain_not_promoted",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )

    event_chain = pd.DataFrame(event_records).reindex(columns=EVENT_CHAIN_COLUMNS).sort_values(
        ["asset", "announcement_date", "document_key"]
    ).reset_index(drop=True)
    asset_chain = pd.DataFrame(asset_records).reindex(columns=ASSET_CHAIN_COLUMNS).sort_values("asset").reset_index(drop=True)
    review = event_chain[
        ~event_chain["event_chain_status"].isin(
            {
                "candidate_event_closed_cross_document",
                "candidate_event_closed_prior_state_inference_review_required",
                "candidate_event_closed_post_legal_value_inference_review_required",
            }
        )
    ].copy()
    _atomic_csv(event_chain, EVENT_CHAIN_PATH)
    _atomic_csv(asset_chain, ASSET_CHAIN_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "CANDIDATE_HISTORY_CHAIN_VALIDATED_FORMAL_PROMOTION_BLOCKED",
        "target_assets": int(len(asset_chain)),
        "target_event_candidates": int(len(event_chain)),
        "event_chain_status_counts": {
            str(key): int(value) for key, value in event_chain["event_chain_status"].value_counts().items()
        },
        "asset_history_candidate_status_counts": {
            str(key): int(value)
            for key, value in asset_chain["asset_history_candidate_status"].value_counts().items()
        },
        "cross_document_closed_event_candidates": int(
            event_chain["event_chain_status"].eq("candidate_event_closed_cross_document").sum()
        ),
        "prior_state_inference_event_candidates": int(
            event_chain["event_chain_status"].eq(
                "candidate_event_closed_prior_state_inference_review_required"
            ).sum()
        ),
        "post_legal_value_inference_event_candidates": int(
            event_chain["event_chain_status"].eq(
                "candidate_event_closed_post_legal_value_inference_review_required"
            ).sum()
        ),
        "unresolved_event_candidates": int(len(review)),
        "formal_history_rows": 0,
        "benchmark_history_complete_assets": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Cross-document event continuity remains candidate evidence because the initial mapping layer is not yet "
            "independently promoted. No candidate interval is written to formal history or consumed by backtests."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {
            "role": "benchmark_event_chain_candidate_validation",
            "path": _relative(EVENT_CHAIN_PATH),
            "sha256": _sha256(EVENT_CHAIN_PATH),
            "rows": int(len(event_chain)),
        },
        {
            "role": "benchmark_asset_history_chain_candidate_validation",
            "path": _relative(ASSET_CHAIN_PATH),
            "sha256": _sha256(ASSET_CHAIN_PATH),
            "rows": int(len(asset_chain)),
        },
        {
            "role": "benchmark_history_chain_review_queue",
            "path": _relative(REVIEW_QUEUE_PATH),
            "sha256": _sha256(REVIEW_QUEUE_PATH),
            "rows": int(len(review)),
        },
        {
            "role": "benchmark_history_chain_report",
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
                "path": _relative(Path(event_validator.__file__).resolve()),
                "sha256": _sha256(Path(event_validator.__file__).resolve()),
            },
            {
                "path": _relative(Path(reconciler.__file__).resolve()),
                "sha256": _sha256(Path(reconciler.__file__).resolve()),
            },
            {
                "path": _relative(Path(parser.__file__).resolve()),
                "sha256": _sha256(Path(parser.__file__).resolve()),
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
        "target_assets",
        "target_event_candidates",
        "event_chain_status_counts",
        "cross_document_closed_event_candidates",
        "prior_state_inference_event_candidates",
        "post_legal_value_inference_event_candidates",
        "unresolved_event_candidates",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
