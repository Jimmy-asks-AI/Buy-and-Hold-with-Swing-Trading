"""Independently validate full-universe official ETF dividend discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_universe_coverage_latest.json"
ANNOUNCEMENT_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_official_announcements.csv"
DOCUMENT_INDEX_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_document_index.csv"
EVENT_INVENTORY_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_event_inventory.csv"
COMPLETE_CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_complete_candidates.csv"
LEGACY_CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_registry_candidates.csv"
LEGACY_VALIDATION_MANIFEST_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_events" / "run_manifest.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_universe_events"
CHECKS_PATH = OUTPUT_DIR / "checks.csv"
EXCEPTIONS_PATH = OUTPUT_DIR / "exceptions.csv"
PROMOTION_CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_validated_candidates.csv"
REPORT_PATH = OUTPUT_DIR / "report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

CHECK_COLUMNS = ["asset", "announcement_date", "check", "status", "detail"]
PROMOTION_COLUMNS = [
    "asset",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "source_document_title",
    "source_url",
    "source_type",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "validation_basis",
    "historical_backtest_allowed",
]
POLICY_TITLE_TOKENS = (
    "调整收益分配",
    "修改基金合同",
    "修订基金合同",
    "收益分配原则",
    "收益分配条款",
    "分红方式变更规则",
    "修改基金份额收益分配",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _safe_path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (ROOT / str(relative)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"ETF dividend universe validation path escapes project root: {relative}")
    return path


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


def _check(
    rows: list[dict[str, str]],
    asset: str,
    announcement_date: Any,
    name: str,
    passed: bool,
    detail: str,
) -> None:
    parsed = pd.to_datetime(announcement_date, errors="coerce")
    rows.append(
        {
            "asset": str(asset),
            "announcement_date": "invalid" if pd.isna(parsed) else str(pd.Timestamp(parsed).date()),
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def _manifest_output_matches(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("outputs", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def _authenticate_code(path_value: Any, hash_value: Any, label: str) -> Path:
    path = _safe_path(str(path_value))
    if not path.is_file() or not str(hash_value) or _sha256(path) != str(hash_value):
        raise ValueError(f"{label} code hash mismatch")
    return path


def _event_key(asset: Any, ex_date: Any, cash_per_share: Any) -> tuple[str, str, float] | None:
    parsed_date = pd.to_datetime(ex_date, errors="coerce")
    parsed_cash = pd.to_numeric(cash_per_share, errors="coerce")
    if pd.isna(parsed_date) or pd.isna(parsed_cash):
        return None
    return str(asset).zfill(6), str(pd.Timestamp(parsed_date).date()), round(float(parsed_cash), 10)


def classify_incomplete_document(row: Any, text: str) -> str:
    status = str(row.parse_status)
    title = str(row.announcement_title)
    if status.startswith("incomplete_cash_") and any(token in title for token in POLICY_TITLE_TOKENS):
        return "non_event_policy_document"
    compact_text = "".join(str(text).split())
    if status == "incomplete_pay_date" and (
        "I类场外份额" in compact_text or "场外I类份额" in compact_text
    ):
        return "off_exchange_share_class_distribution"
    return "unresolved_incomplete_distribution_document"


def document_candidate_event_key(row: Any) -> tuple[str, str, str, str, float] | None:
    def unique_values(column: str, field: str) -> set[Any]:
        try:
            parsed = json.loads(str(getattr(row, column)))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return set()
        if not isinstance(parsed, list):
            return set()
        values: set[Any] = set()
        for item in parsed:
            if not isinstance(item, dict) or field not in item:
                continue
            if field == "date":
                value = pd.to_datetime(item[field], errors="coerce")
                if pd.notna(value):
                    values.add(str(pd.Timestamp(value).date()))
            else:
                value = pd.to_numeric(item[field], errors="coerce")
                if pd.notna(value):
                    values.add(round(float(value), 10))
        return values

    cash = unique_values("cash_candidates_json", "cash_per_share")
    record = unique_values("record_date_candidates_json", "date")
    ex_date = unique_values("ex_date_candidates_json", "date")
    pay = unique_values("pay_date_candidates_json", "date")
    if any(len(values) != 1 for values in (cash, record, ex_date, pay)):
        return None
    return (
        str(row.asset).zfill(6),
        next(iter(record)),
        next(iter(ex_date)),
        next(iter(pay)),
        next(iter(cash)),
    )


def _normalise_dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    result["asset"] = result["asset"].astype(str).str.zfill(6)
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    result["cash_per_share"] = pd.to_numeric(result["cash_per_share"], errors="coerce")
    return result


def build_promotion_candidates(
    legacy: pd.DataFrame,
    complete: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = _normalise_dates(legacy, ("announcement_date", "record_date", "ex_date", "pay_date"))
    complete = _normalise_dates(complete, ("announcement_date", "record_date", "ex_date", "pay_date"))
    if "source_document_title" not in complete.columns and "announcement_title" in complete.columns:
        complete = complete.rename(columns={"announcement_title": "source_document_title"})
    legacy_keys = {
        key
        for row in legacy.itertuples(index=False)
        if (key := _event_key(row.asset, row.ex_date, row.cash_per_share)) is not None
    }
    complete = complete.sort_values(["asset", "ex_date", "cash_per_share", "announcement_date"])
    complete = complete.drop_duplicates(["asset", "ex_date", "cash_per_share"], keep="first")
    extra_mask = [
        _event_key(row.asset, row.ex_date, row.cash_per_share) not in legacy_keys
        for row in complete.itertuples(index=False)
    ]
    extras = complete.loc[extra_mask].copy()

    legacy_output = legacy.reindex(columns=PROMOTION_COLUMNS[:-2]).copy()
    legacy_output["validation_basis"] = "legacy_price_marker_plus_official_document_crosscheck"
    legacy_output["historical_backtest_allowed"] = False
    extra_output = extras.reindex(columns=PROMOTION_COLUMNS[:-2]).copy()
    extra_output["validation_basis"] = "full_universe_official_document_unique_parse"
    extra_output["historical_backtest_allowed"] = False
    combined = pd.concat([legacy_output, extra_output], ignore_index=True).reindex(columns=PROMOTION_COLUMNS)
    combined = combined.sort_values(["asset", "announcement_date", "ex_date"]).reset_index(drop=True)
    return combined, extras.reset_index(drop=True)


def run_validation() -> dict[str, Any]:
    universe = json.loads(UNIVERSE_MANIFEST_PATH.read_text(encoding="utf-8"))
    legacy_authority = json.loads(LEGACY_VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_universe = {
        "qualification_status": "FULL_UNIVERSE_QUERIED_OFFICIAL_EVENTS_REQUIRE_VALIDATION",
        "target_assets": 1701,
        "query_complete_assets": 1701,
        "document_failures": 0,
        "validated_source_documents_missing": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    universe_mismatches = {
        key: (universe.get(key), expected)
        for key, expected in expected_universe.items()
        if universe.get(key) != expected
    }
    if universe_mismatches:
        raise ValueError(f"ETF dividend universe evidence is not complete: {universe_mismatches}")
    universe_code = _authenticate_code(universe.get("code_path"), universe.get("code_sha256"), "universe collector")
    universe_dependencies = [
        _authenticate_code(item.get("path"), item.get("sha256"), "universe collector dependency")
        for item in universe.get("code_dependencies", [])
        if isinstance(item, dict)
    ]
    expected_outputs = (ANNOUNCEMENT_PATH, DOCUMENT_INDEX_PATH, EVENT_INVENTORY_PATH, COMPLETE_CANDIDATE_PATH)
    if not all(_manifest_output_matches(universe, path) for path in expected_outputs):
        raise ValueError("ETF dividend universe manifest does not authenticate all observation outputs")

    expected_legacy = {
        "validation_schema": "etf_dividend_official_cross_evidence_v1",
        "qualification_status": "PASS",
        "failed_check_rows": 0,
        "formal_table_promotion_allowed": True,
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    legacy_mismatches = {
        key: (legacy_authority.get(key), expected)
        for key, expected in expected_legacy.items()
        if legacy_authority.get(key) != expected
    }
    if legacy_mismatches:
        raise ValueError(f"legacy ETF dividend authority is invalid: {legacy_mismatches}")
    legacy_code = _authenticate_code(
        legacy_authority.get("code_path"), legacy_authority.get("code_sha256"), "legacy validator"
    )
    legacy_input = next(
        (
            item
            for item in legacy_authority.get("inputs", [])
            if isinstance(item, dict)
            and str(item.get("path", "")).replace("\\", "/") == _relative(LEGACY_CANDIDATE_PATH)
        ),
        {},
    )
    if str(legacy_input.get("sha256", "")) != _sha256(LEGACY_CANDIDATE_PATH):
        raise ValueError("legacy ETF dividend authority does not authenticate its candidate table")

    announcements = pd.read_csv(ANNOUNCEMENT_PATH, dtype={"asset": str})
    documents = pd.read_csv(DOCUMENT_INDEX_PATH, dtype={"asset": str})
    inventory = pd.read_csv(EVENT_INVENTORY_PATH, dtype={"asset": str})
    complete = pd.read_csv(COMPLETE_CANDIDATE_PATH, dtype={"asset": str})
    legacy = pd.read_csv(LEGACY_CANDIDATE_PATH, dtype={"asset": str})
    combined, extras = build_promotion_candidates(legacy, complete)
    known_event_keys = {
        (
            str(row.asset).zfill(6),
            str(pd.Timestamp(row.record_date).date()),
            str(pd.Timestamp(row.ex_date).date()),
            str(pd.Timestamp(row.pay_date).date()),
            round(float(row.cash_per_share), 10),
        )
        for row in combined.itertuples(index=False)
    }
    documents_by_url = {
        str(row.source_url): row for row in documents.itertuples(index=False)
    }

    checks: list[dict[str, str]] = []
    _check(checks, "ALL", universe.get("as_of_date"), "announcement_source_urls_unique", not announcements["source_url"].duplicated().any(), f"rows={len(announcements)}")
    _check(checks, "ALL", universe.get("as_of_date"), "document_rows_match_announcements", set(documents["source_url"]) == set(announcements["source_url"]), f"documents={len(documents)};announcements={len(announcements)}")
    _check(checks, "ALL", universe.get("as_of_date"), "inventory_rows_match_documents", set(inventory["source_url"]) == set(documents["source_url"]), f"inventory={len(inventory)};documents={len(documents)}")

    for row in documents.itertuples(index=False):
        asset = str(row.asset).zfill(6)
        source_url = str(row.source_url)
        source_type = str(row.source_type)
        source_ok = bool(
            (
                asset.startswith("5")
                and source_type == "exchange_announcement"
                and source_url.startswith("https://www.sse.com.cn/disclosure/fund/announcement/")
            )
            or (
                asset.startswith("1")
                and source_type == "regulatory_filing"
                and source_url.startswith("https://static.cninfo.com.cn/finalpage/")
            )
        )
        _check(checks, asset, row.announcement_date, "official_source_contract", source_ok, source_url)
        for role in ("pdf", "text"):
            relative = str(getattr(row, f"{role}_path"))
            expected_hash = str(getattr(row, f"{role}_sha256"))
            try:
                evidence_path = _safe_path(relative)
                valid_hash = bool(evidence_path.is_file() and expected_hash and _sha256(evidence_path) == expected_hash)
            except (OSError, ValueError):
                valid_hash = False
            _check(checks, asset, row.announcement_date, f"{role}_hash_match", valid_hash, relative)

    legacy_urls = set(legacy["source_url"].astype(str))
    legacy_asset_announcement = {
        (str(row.asset).zfill(6), str(pd.Timestamp(row.announcement_date).date()))
        for row in legacy.itertuples(index=False)
    }
    unresolved_review = 0
    for row in inventory.itertuples(index=False):
        status = str(row.parse_status)
        if status == "complete_unique_official_event":
            continue
        if status == "ambiguous_best_official_event":
            resolved = bool(
                str(row.source_url) in legacy_urls
                or (str(row.asset).zfill(6), str(pd.Timestamp(row.announcement_date).date()))
                in legacy_asset_announcement
            )
            classification = "resolved_by_legacy_event_crosscheck" if resolved else "unresolved_ambiguous_event"
        else:
            text = _safe_path(str(row.text_path)).read_text(encoding="utf-8") if str(row.text_path) else ""
            classification = classify_incomplete_document(row, text)
            if (
                classification == "unresolved_incomplete_distribution_document"
                and "提示性" in str(row.announcement_title)
            ):
                document = documents_by_url.get(str(row.source_url))
                candidate_key = document_candidate_event_key(document) if document is not None else None
                if candidate_key is not None and candidate_key in known_event_keys:
                    classification = "duplicate_prompt_of_validated_event"
            resolved = classification != "unresolved_incomplete_distribution_document"
        unresolved_review += 0 if resolved else 1
        _check(checks, row.asset, row.announcement_date, "non_unique_document_resolution", resolved, classification)

    as_of_date = pd.Timestamp(universe["as_of_date"]).normalize()
    for row in extras.itertuples(index=False):
        dates = [pd.Timestamp(getattr(row, column)) for column in ("announcement_date", "record_date", "ex_date", "pay_date")]
        cash = float(row.cash_per_share)
        chronology = dates[0] <= dates[1] <= dates[2] <= dates[3]
        _check(checks, row.asset, row.announcement_date, "new_event_chronology", chronology, "/".join(str(date.date()) for date in dates))
        _check(checks, row.asset, row.announcement_date, "new_event_known_by_as_of", dates[0] <= as_of_date, f"as_of={as_of_date.date()}")
        _check(checks, row.asset, row.announcement_date, "new_event_cash_positive_finite", np.isfinite(cash) and cash > 0, f"cash={cash}")
        title = str(row.source_document_title)
        _check(
            checks,
            row.asset,
            row.announcement_date,
            "new_event_title_is_distribution",
            "分红公告" in title or "收益分配公告" in title,
            title,
        )

    _check(checks, "ALL", as_of_date, "non_event_review_resolved", unresolved_review == 0, f"unresolved={unresolved_review}")
    _check(checks, "ALL", as_of_date, "promotion_candidate_primary_key_unique", not combined.duplicated(["asset", "announcement_date", "ex_date", "cash_per_share"]).any(), f"rows={len(combined)}")
    _check(checks, "ALL", as_of_date, "promotion_candidate_event_key_unique", not combined.duplicated(["asset", "ex_date", "cash_per_share"]).any(), f"rows={len(combined)}")
    _check(checks, "ALL", as_of_date, "promotion_candidates_stay_disabled", combined["historical_backtest_allowed"].astype(str).str.lower().eq("false").all(), "candidate observation boundary")

    check_frame = pd.DataFrame(checks, columns=CHECK_COLUMNS)
    exceptions = check_frame[check_frame["status"].eq("fail")].copy()
    qualification = "PASS" if exceptions.empty else "FAIL"
    _atomic_csv(check_frame, CHECKS_PATH)
    _atomic_csv(exceptions, EXCEPTIONS_PATH)
    _atomic_csv(combined, PROMOTION_CANDIDATE_PATH)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(as_of_date.date()),
        "qualification_status": qualification,
        "full_universe_query_complete": universe.get("query_complete_assets") == universe.get("target_assets") == 1701,
        "official_document_rows": int(len(documents)),
        "official_document_failures": int(documents["pdf_status"].ne("success").sum()),
        "unresolved_review_documents": unresolved_review,
        "legacy_candidate_rows": int(len(legacy)),
        "new_official_event_rows": int(len(extras)),
        "formal_candidate_rows": int(len(combined)),
        "failed_check_rows": int(len(exceptions)),
        "coverage_start": str(pd.Timestamp(combined["announcement_date"].min()).date()),
        "coverage_end": str(pd.Timestamp(combined["announcement_date"].max()).date()),
        "historical_backtest_allowed": qualification == "PASS",
        "formal_table_promotion_allowed": qualification == "PASS",
        "model_promotion_allowed": False,
        "boundary": "Authority is limited to official ETF distribution events; no total-return reconstruction or model promotion.",
    }
    _atomic_json(report, REPORT_PATH)
    input_paths = [
        UNIVERSE_MANIFEST_PATH,
        ANNOUNCEMENT_PATH,
        DOCUMENT_INDEX_PATH,
        EVENT_INVENTORY_PATH,
        COMPLETE_CANDIDATE_PATH,
        LEGACY_CANDIDATE_PATH,
        LEGACY_VALIDATION_MANIFEST_PATH,
        universe_code,
        *universe_dependencies,
        legacy_code,
    ]
    manifest = {
        "validation_schema": "etf_dividend_full_universe_official_v1",
        **report,
        "inputs": [{"path": _relative(path), "sha256": _sha256(path)} for path in input_paths],
        "outputs": [
            {"role": "checks", "path": _relative(CHECKS_PATH), "sha256": _sha256(CHECKS_PATH)},
            {"role": "exceptions", "path": _relative(EXCEPTIONS_PATH), "sha256": _sha256(EXCEPTIONS_PATH)},
            {"role": "promotion_candidates", "path": _relative(PROMOTION_CANDIDATE_PATH), "sha256": _sha256(PROMOTION_CANDIDATE_PATH)},
            {"role": "report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
        ],
        "exceptions_path": _relative(EXCEPTIONS_PATH),
        "exceptions_sha256": _sha256(EXCEPTIONS_PATH),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    manifest = run_validation()
    keys = (
        "qualification_status",
        "full_universe_query_complete",
        "official_document_rows",
        "unresolved_review_documents",
        "legacy_candidate_rows",
        "new_official_event_rows",
        "formal_candidate_rows",
        "failed_check_rows",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
