"""Reconcile ETF benchmark document candidates without promoting historical facts."""

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

from . import pit_etf_benchmark_document_candidate_parser as parser
from . import pit_etf_benchmark_document_selector as selector
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
PARSER_MANIFEST_PATH = parser.MANIFEST_PATH
SELECTOR_MANIFEST_PATH = selector.MANIFEST_PATH
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
INITIAL_PATH = OBSERVATION_DIR / "etf_benchmark_initial_candidate_reconciliation.csv"
CHANGE_PATH = OBSERVATION_DIR / "etf_benchmark_change_event_candidates.csv"
COVERAGE_PATH = OBSERVATION_DIR / "etf_benchmark_asset_candidate_coverage_registry.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_asset_candidate_review_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_benchmark_asset_reconciliation" / "reconciliation_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_asset_candidate_reconciler_latest.json"

SCHEMA_VERSION = 1
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
RESOLVED_REFERENCE_TYPES = {
    "tracked_index",
    "enhanced_index",
    "non_index_reference",
    "commodity_spot_reference",
}
INDEX_REFERENCE_TYPES = {"tracked_index", "enhanced_index"}
BASELINE_STATES = {
    "preferred_initial_legal_candidate",
    "listing_only_initial_candidate",
    "fallback_update_candidate",
}
INITIAL_SUPPLEMENT_REASONS = {
    "one_initial_prospectus_supplement_document",
    "one_earliest_updated_prospectus_fallback_document",
    "one_canonical_listing_context_document",
}
EVENT_REASONS = {
    "all_title_routed_benchmark_change_documents",
    "all_contract_amendments",
    "all_holder_resolutions",
}

INITIAL_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "delist_date",
    "tradable_scope_from_candidate",
    "reference_type_candidate",
    "canonical_index_name_candidate",
    "canonical_index_code_candidate",
    "canonical_performance_benchmark_candidate",
    "name_candidate_score",
    "name_agreement_document_count",
    "code_candidate_count",
    "initial_selected_document_count",
    "initial_parsed_document_count",
    "baseline_parse_status",
    "reference_type_support_source",
    "reference_type_conflicts_json",
    "index_name_candidates_json",
    "index_code_candidates_json",
    "performance_benchmark_candidates_json",
    "supporting_document_keys_json",
    "initial_reconciliation_status",
    "independent_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
CHANGE_COLUMNS = [
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
    "reference_type_candidate",
    "index_name_candidates_json",
    "index_code_candidates_json",
    "performance_benchmark_candidates_json",
    "effective_date_candidates_json",
    "event_candidate_status",
    "independent_validation_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
COVERAGE_COLUMNS = [
    "asset",
    "asset_name",
    "initial_reconciliation_status",
    "reference_type_candidate",
    "change_document_count",
    "parsed_change_document_count",
    "pending_or_failed_change_document_count",
    "asset_candidate_state",
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
        raise ValueError(f"ETF benchmark reconciliation lineage hash mismatch: {snapshot}")
    return snapshot


def _json_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return []
    parsed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, received {type(parsed).__name__}")
    return [str(item) for item in parsed]


def _json_list(values: set[str] | list[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def _normalize(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value)).replace("（", "(").replace("）", ")")
    return re.sub(r"[，,。；;：:]", "", compact).lower()


def _reason_values(row: Any) -> set[str]:
    return set(_json_values(row.selection_reasons_json))


def _document_role(row: Any) -> str:
    if str(row.baseline_selection_state) in BASELINE_STATES:
        return "baseline"
    reasons = _reason_values(row)
    if "one_initial_prospectus_supplement_document" in reasons:
        return "prospectus_supplement"
    if "one_earliest_updated_prospectus_fallback_document" in reasons:
        return "updated_prospectus_fallback"
    if "one_canonical_listing_context_document" in reasons:
        return "listing_context"
    return "other"


def _candidate_quality_score(field: str, value: str) -> int:
    if field != "performance_benchmark_candidates_json":
        return 0
    compact = re.sub(r"\s+", "", str(value))
    score = 1 if 2 <= len(compact) <= 120 else 0
    if re.search(r"(?:收益率|利率|回报率|净价|价格|收盘价)[）)]?$", compact):
        score += 3
    if re.search(r"(?:\d+%|×|\*|\+)", compact):
        score += 1
    return score


def _choose_scored_candidate(rows: pd.DataFrame, field: str) -> tuple[str, int, int]:
    scores: dict[str, int] = {}
    display: dict[str, str] = {}
    support_docs: dict[str, set[str]] = {}
    for row in rows.itertuples(index=False):
        role = _document_role(row)
        role_score = {
            "baseline": 4,
            "prospectus_supplement": 3,
            "updated_prospectus_fallback": 2,
            "listing_context": 1,
        }.get(role, 0)
        values = _json_values(getattr(row, field))
        performance = "|".join(_normalize(value) for value in _json_values(row.performance_benchmark_candidates_json))
        for value in values:
            normalized = _normalize(value)
            if not normalized:
                continue
            score = role_score + _candidate_quality_score(field, value)
            if field == "index_name_candidates_json" and normalized in performance:
                score += 2
            scores[normalized] = scores.get(normalized, 0) + score
            display.setdefault(normalized, value)
            support_docs.setdefault(normalized, set()).add(str(row.document_key))
    if not scores:
        return "", 0, 0
    maximum = max(scores.values())
    winners = sorted(key for key, value in scores.items() if value == maximum)
    if len(winners) != 1:
        return "", maximum, 0
    winner = winners[0]
    return display[winner], maximum, len(support_docs[winner])


def _candidate_scope_date(rows: pd.DataFrame, field: str, candidate: str, list_date: Any) -> str:
    list_timestamp = pd.to_datetime(list_date, errors="coerce")
    evidence_timestamp = pd.NaT
    if candidate and field in rows.columns and "available_date" in rows.columns:
        normalized = _normalize(candidate)
        support_mask = rows[field].map(
            lambda value: any(_normalize(item) == normalized for item in _json_values(value))
        )
        evidence_dates = pd.to_datetime(rows.loc[support_mask, "available_date"], errors="coerce").dropna()
        if not evidence_dates.empty:
            evidence_timestamp = evidence_dates.min()
    timestamps = [value for value in (list_timestamp, evidence_timestamp) if not pd.isna(value)]
    if not timestamps:
        return "" if list_date is None or pd.isna(list_date) else str(list_date)
    return max(timestamps).date().isoformat()


def reconcile_asset(rows: pd.DataFrame, *, list_date: Any, delist_date: Any = None) -> dict[str, Any]:
    if rows.empty:
        raise ValueError("cannot reconcile an empty ETF benchmark asset candidate set")
    baseline = rows[rows["baseline_selection_state"].isin(BASELINE_STATES)]
    if len(baseline) != 1:
        raise ValueError("ETF benchmark reconciliation requires exactly one baseline document")
    initial = rows[
        rows.apply(
            lambda row: str(row["baseline_selection_state"]) in BASELINE_STATES
            or bool(set(_json_values(row["selection_reasons_json"])).intersection(INITIAL_SUPPLEMENT_REASONS)),
            axis=1,
        )
    ].copy()
    parsed = initial[~initial["parse_status"].isin(parser.UNPARSED_STATUSES)].copy()
    baseline_row = baseline.iloc[0]
    baseline_type = str(baseline_row["reference_type_candidate"])
    initial_prospectus_mask = parsed["selection_reasons_json"].map(
        lambda value: "one_initial_prospectus_supplement_document" in set(_json_values(value))
    ).astype(bool)
    updated_prospectus_mask = parsed["selection_reasons_json"].map(
        lambda value: "one_earliest_updated_prospectus_fallback_document" in set(_json_values(value))
    ).astype(bool)
    prospectus_mask = initial_prospectus_mask | updated_prospectus_mask
    prospectus = parsed.loc[prospectus_mask]
    prospectus_types = set(prospectus["reference_type_candidate"]).intersection(RESOLVED_REFERENCE_TYPES)
    initial_prospectus_types = set(
        parsed.loc[initial_prospectus_mask, "reference_type_candidate"]
    ).intersection(RESOLVED_REFERENCE_TYPES)
    conflicts: set[str] = set()
    if baseline_type in RESOLVED_REFERENCE_TYPES:
        reference_type = baseline_type
        support_source = "baseline_document"
        for value in prospectus_types:
            if value != reference_type:
                conflicts.add(value)
    elif len(prospectus_types) == 1:
        reference_type = next(iter(prospectus_types))
        support_source = (
            "prospectus_supplement_recovery"
            if initial_prospectus_types
            else "updated_prospectus_fallback_recovery"
        )
    elif len(prospectus_types) > 1:
        reference_type = "unknown"
        support_source = "conflicting_prospectus_supplements"
        conflicts.update(prospectus_types)
    else:
        reference_type = "unknown"
        support_source = "unresolved"

    canonical_name, name_score, name_agreement = _choose_scored_candidate(
        parsed, "index_name_candidates_json"
    )
    canonical_performance, _, _ = _choose_scored_candidate(
        parsed, "performance_benchmark_candidates_json"
    )
    all_names = {
        value for values in parsed["index_name_candidates_json"].map(_json_values) for value in values
    }
    all_codes = {
        value for values in parsed["index_code_candidates_json"].map(_json_values) for value in values
    }
    all_codes = {
        value
        for value in all_codes
        if not (value.isdigit() and len(value) == 7 and value[:6] in all_codes)
    }
    all_performance = {
        value
        for values in parsed["performance_benchmark_candidates_json"].map(_json_values)
        for value in values
    }
    canonical_code = next(iter(all_codes)) if len(all_codes) == 1 else ""
    scope_field = (
        "index_name_candidates_json"
        if reference_type in INDEX_REFERENCE_TYPES
        else "performance_benchmark_candidates_json"
    )
    scope_candidate = canonical_name if reference_type in INDEX_REFERENCE_TYPES else canonical_performance
    tradable_scope = _candidate_scope_date(parsed, scope_field, scope_candidate, list_date)
    hard_type_conflict = bool(
        conflicts
        and (
            reference_type == "non_index_reference"
            or "non_index_reference" in conflicts
            or any(value not in INDEX_REFERENCE_TYPES for value in conflicts)
        )
    )
    if hard_type_conflict:
        status = "reference_type_conflict_review_required"
    elif reference_type == "unknown":
        if str(baseline_row["parse_status"]) in parser.UNPARSED_STATUSES:
            status = "baseline_document_unavailable_supplement_unresolved"
        else:
            status = "reference_type_unresolved"
    elif reference_type in INDEX_REFERENCE_TYPES and not canonical_name:
        status = "index_name_conflict_or_missing"
    elif reference_type in INDEX_REFERENCE_TYPES and len(all_codes) > 1:
        status = "index_code_conflict_review_required"
    elif reference_type in INDEX_REFERENCE_TYPES and not canonical_code:
        status = "index_name_reconciled_code_missing"
    elif reference_type in INDEX_REFERENCE_TYPES and name_agreement >= 2:
        status = "cross_document_name_code_candidate_review_required"
    elif reference_type in INDEX_REFERENCE_TYPES:
        status = "single_document_name_code_candidate_review_required"
    elif reference_type == "non_index_reference" and canonical_performance:
        status = "non_index_reference_candidate_review_required"
    elif reference_type == "commodity_spot_reference" and canonical_performance:
        status = "commodity_spot_reference_candidate_review_required"
    elif reference_type == "commodity_spot_reference":
        status = "commodity_spot_benchmark_text_missing"
    else:
        status = "non_index_performance_benchmark_missing"

    return {
        "asset": str(rows.iloc[0]["asset"]).zfill(6),
        "asset_name": str(rows.iloc[0]["asset_name"]),
        "exchange": str(rows.iloc[0]["exchange"]),
        "list_date": list_date,
        "delist_date": delist_date,
        "tradable_scope_from_candidate": tradable_scope,
        "reference_type_candidate": reference_type,
        "canonical_index_name_candidate": canonical_name if reference_type in INDEX_REFERENCE_TYPES else "",
        "canonical_index_code_candidate": canonical_code if reference_type in INDEX_REFERENCE_TYPES else "",
        "canonical_performance_benchmark_candidate": canonical_performance,
        "name_candidate_score": int(name_score),
        "name_agreement_document_count": int(name_agreement),
        "code_candidate_count": int(len(all_codes)),
        "initial_selected_document_count": int(len(initial)),
        "initial_parsed_document_count": int(len(parsed)),
        "baseline_parse_status": str(baseline_row["parse_status"]),
        "reference_type_support_source": support_source,
        "reference_type_conflicts_json": _json_list(conflicts),
        "index_name_candidates_json": _json_list(all_names),
        "index_code_candidates_json": _json_list(all_codes),
        "performance_benchmark_candidates_json": _json_list(all_performance),
        "supporting_document_keys_json": _json_list(set(initial["document_key"].astype(str))),
        "initial_reconciliation_status": status,
        "independent_validation_status": "not_started",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def _authenticate_upstream() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    parser_manifest = json.loads(PARSER_MANIFEST_PATH.read_text(encoding="utf-8"))
    selector_manifest = json.loads(SELECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        parser_manifest.get("qualification_status")
        != "HEURISTIC_DOCUMENT_CANDIDATES_INDEPENDENT_VALIDATION_REQUIRED"
        or parser_manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF benchmark parser does not authorize candidate reconciliation")
    if (
        selector_manifest.get("qualification_status") != "FULL_AUTHENTICATED_DOCUMENT_ROUTING_COLLECTION_REQUIRED"
        or selector_manifest.get("selection_policy_version") != selector.SELECTION_POLICY_VERSION
        or int(selector_manifest.get("target_assets", 0)) != 1701
    ):
        raise ValueError("ETF benchmark selector does not match reconciliation policy")
    inputs: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for label, manifest_path, manifest, role in (
        ("parser", PARSER_MANIFEST_PATH, parser_manifest, "benchmark_document_parse_candidates"),
        ("selector", SELECTOR_MANIFEST_PATH, selector_manifest, "benchmark_document_selection_coverage"),
    ):
        producer_path = ROOT / str(manifest.get("code_path", ""))
        authenticated_code = source_code_archive.authenticate_current_or_archive(
            producer_path, str(manifest.get("code_sha256", ""))
        )
        manifest_snapshot = _content_snapshot(manifest_path)
        outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
        item = outputs.get(role, {})
        output_path = ROOT / str(item.get("path", ""))
        if not output_path.is_file() or _sha256(output_path) != str(item.get("sha256", "")):
            raise ValueError(f"ETF benchmark {label} output hash mismatch: {role}")
        output_snapshot = _content_snapshot(output_path)
        paths[label] = output_snapshot
        inputs.extend(
            [
                {"role": f"{label}_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)},
                {"role": f"{label}_output_snapshot", "path": _relative(output_snapshot), "sha256": _sha256(output_snapshot)},
                {"role": f"authenticated_{label}_code", "path": _relative(authenticated_code), "sha256": _sha256(authenticated_code)},
            ]
        )
    candidates = pd.read_csv(paths["parser"], dtype={"asset": str}, low_memory=False)
    selection_coverage = pd.read_csv(paths["selector"], dtype={"asset": str}, low_memory=False)
    if set(candidates["asset"].astype(str).str.zfill(6)) != set(selection_coverage["asset"].astype(str).str.zfill(6)):
        raise ValueError("ETF benchmark parser and selector coverage assets do not match")
    if candidates["historical_backtest_allowed"].astype(str).str.lower().eq("true").any():
        raise ValueError("ETF benchmark parser candidates unexpectedly authorize historical use")
    return candidates, selection_coverage, inputs


def run_reconciliation() -> dict[str, Any]:
    candidates, selection_coverage, inputs = _authenticate_upstream()
    coverage_lookup = selection_coverage.set_index(selection_coverage["asset"].astype(str).str.zfill(6))
    initial_records = [
        reconcile_asset(
            rows,
            list_date=coverage_lookup.loc[str(asset).zfill(6), "list_date"],
            delist_date=coverage_lookup.loc[str(asset).zfill(6), "delist_date"],
        )
        for asset, rows in candidates.groupby(candidates["asset"].astype(str).str.zfill(6), sort=True)
    ]
    initial = pd.DataFrame(initial_records).reindex(columns=INITIAL_COLUMNS).sort_values("asset").reset_index(drop=True)

    event_mask = candidates["selection_reasons_json"].map(
        lambda value: bool(set(_json_values(value)).intersection(EVENT_REASONS))
    )
    event_rows: list[dict[str, Any]] = []
    for row in candidates[event_mask].itertuples(index=False):
        if str(row.parse_status) == "pending_document_collection":
            event_status = "pending_document_collection"
        elif str(row.parse_status) == "document_collection_failed":
            event_status = "document_collection_failed"
        elif str(row.parse_status) == "ocr_required":
            event_status = "ocr_required"
        elif (
            str(row.index_name_candidates_json) != "[]"
            or str(row.index_code_candidates_json) != "[]"
            or str(row.effective_date_candidates_json) != "[]"
        ):
            event_status = "change_fields_candidate_review_required"
        else:
            event_status = "parsed_no_change_fields_review_required"
        event_rows.append(
            {
                "asset": str(row.asset).zfill(6),
                "asset_name": str(row.asset_name),
                "announcement_date": row.announcement_date,
                "available_date": row.available_date,
                "announcement_title": str(row.announcement_title),
                "source_url": str(row.source_url),
                "document_key": str(row.document_key),
                "selection_reasons_json": str(row.selection_reasons_json),
                "document_collection_status": str(row.document_collection_status),
                "parse_status": str(row.parse_status),
                "reference_type_candidate": str(row.reference_type_candidate),
                "index_name_candidates_json": str(row.index_name_candidates_json),
                "index_code_candidates_json": str(row.index_code_candidates_json),
                "performance_benchmark_candidates_json": str(row.performance_benchmark_candidates_json),
                "effective_date_candidates_json": str(row.effective_date_candidates_json),
                "event_candidate_status": event_status,
                "independent_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    changes = pd.DataFrame(event_rows).reindex(columns=CHANGE_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)

    change_groups = {asset: rows for asset, rows in changes.groupby("asset", sort=True)}
    coverage_records: list[dict[str, Any]] = []
    for row in initial.itertuples(index=False):
        asset_changes = change_groups.get(str(row.asset), changes.iloc[0:0])
        parsed_change = ~asset_changes["event_candidate_status"].isin(
            {"pending_document_collection", "document_collection_failed", "ocr_required"}
        )
        pending_change = ~parsed_change
        if str(row.reference_type_candidate) == "unknown":
            state = "initial_reference_unresolved"
        elif pending_change.any():
            state = "initial_candidate_available_change_documents_incomplete"
        else:
            state = "candidate_chain_independent_validation_required"
        coverage_records.append(
            {
                "asset": str(row.asset),
                "asset_name": str(row.asset_name),
                "initial_reconciliation_status": str(row.initial_reconciliation_status),
                "reference_type_candidate": str(row.reference_type_candidate),
                "change_document_count": int(len(asset_changes)),
                "parsed_change_document_count": int(parsed_change.sum()),
                "pending_or_failed_change_document_count": int(pending_change.sum()),
                "asset_candidate_state": state,
                "independent_validation_status": "not_started",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    coverage = pd.DataFrame(coverage_records).reindex(columns=COVERAGE_COLUMNS).sort_values("asset").reset_index(drop=True)
    review_queue = initial.copy()
    _atomic_csv(initial, INITIAL_PATH)
    _atomic_csv(changes, CHANGE_PATH)
    _atomic_csv(coverage, COVERAGE_PATH)
    _atomic_csv(review_queue, REVIEW_QUEUE_PATH)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "ASSET_CANDIDATES_RECONCILED_INDEPENDENT_VALIDATION_REQUIRED",
        "target_assets": int(len(initial)),
        "reference_type_candidate_counts": {
            str(key): int(value) for key, value in initial["reference_type_candidate"].value_counts().items()
        },
        "initial_reconciliation_status_counts": {
            str(key): int(value) for key, value in initial["initial_reconciliation_status"].value_counts().items()
        },
        "change_candidate_documents": int(len(changes)),
        "parsed_change_candidate_documents": int(
            (~changes["event_candidate_status"].isin({"pending_document_collection", "document_collection_failed", "ocr_required"})).sum()
        ),
        "assets_with_name_and_code_candidates": int(
            initial["canonical_index_name_candidate"].ne("").mul(initial["canonical_index_code_candidate"].ne("")).sum()
        ),
        "independent_validation_complete": False,
        "formal_history_rows": 0,
        "official_no_benchmark_change_assets": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Cross-document candidate reconciliation only. Scores resolve review order, not facts; no initial mapping, "
            "change event, no-change claim, or historical interval is promoted here."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "benchmark_initial_candidate_reconciliation", "path": _relative(INITIAL_PATH), "sha256": _sha256(INITIAL_PATH), "rows": int(len(initial))},
        {"role": "benchmark_change_event_candidates", "path": _relative(CHANGE_PATH), "sha256": _sha256(CHANGE_PATH), "rows": int(len(changes))},
        {"role": "benchmark_asset_candidate_coverage", "path": _relative(COVERAGE_PATH), "sha256": _sha256(COVERAGE_PATH), "rows": int(len(coverage))},
        {"role": "benchmark_asset_candidate_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review_queue))},
        {"role": "reconciliation_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(parser.__file__).resolve()), "sha256": _sha256(Path(parser.__file__).resolve())},
            {"path": _relative(Path(selector.__file__).resolve()), "sha256": _sha256(Path(selector.__file__).resolve())},
            {"path": _relative(Path(source_code_archive.__file__).resolve()), "sha256": _sha256(Path(source_code_archive.__file__).resolve())},
        ],
        "current_final_snapshot": True,
        "contains_validated_benchmark_facts": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def main() -> None:
    result = run_reconciliation()
    keys = (
        "qualification_status",
        "target_assets",
        "reference_type_candidate_counts",
        "initial_reconciliation_status_counts",
        "change_candidate_documents",
        "parsed_change_candidate_documents",
        "assets_with_name_and_code_candidates",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
