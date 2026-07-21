"""Independently validate official ETF cash-distribution candidates."""

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
CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_registry_candidates.csv"
MATCH_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_official_queue_matches.csv"
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_dividend_evidence_queue.csv"
COLLECTOR_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_announcement_collector_latest.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_events"
CHECKS_PATH = OUTPUT_DIR / "candidate_checks.csv"
EXCEPTIONS_PATH = OUTPUT_DIR / "exceptions.csv"
REPORT_PATH = OUTPUT_DIR / "report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
REQUIRED_COLUMNS = {
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
    "source_event_date",
    "cash_relative_error_to_discovery",
    "ex_date_distance_days",
    "review_status",
    "historical_backtest_allowed",
}
CHECK_COLUMNS = ["asset", "ex_date", "check", "status", "detail"]
ALLOWED_REVIEW_STATUSES = {"official_pdf_cash_and_dates_found_review_required"}


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
        raise ValueError(f"ETF dividend evidence path escapes project root: {relative}")
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


def _manifest_output_matches(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def _authenticate_code(entry: dict[str, Any], label: str) -> Path:
    relative = str(entry.get("path", ""))
    expected_hash = str(entry.get("sha256", ""))
    path = _safe_path(relative) if relative else None
    if not path or not path.is_file() or not expected_hash or _sha256(path) != expected_hash:
        raise ValueError(f"{label} code hash mismatch")
    return path


def _check(rows: list[dict[str, str]], asset: str, ex_date: Any, name: str, passed: bool, detail: str) -> None:
    parsed = pd.to_datetime(ex_date, errors="coerce")
    rows.append(
        {
            "asset": asset,
            "ex_date": "invalid" if pd.isna(parsed) else str(pd.Timestamp(parsed).date()),
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def load_candidates(path: Path = CANDIDATE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"ETF dividend candidates are incomplete: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    for column in ("announcement_date", "record_date", "ex_date", "pay_date", "source_event_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    for column in ("cash_per_share", "cash_relative_error_to_discovery", "ex_date_distance_days"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    duplicate_formal = frame.duplicated(["asset", "announcement_date", "ex_date", "cash_per_share"], keep=False)
    duplicate_discovery = frame.duplicated(["asset", "source_event_date"], keep=False)
    for position, row in frame.iterrows():
        asset = str(row["asset"]).zfill(6)
        ex_date = row["ex_date"]
        dates_valid = bool(
            pd.notna(row["announcement_date"])
            and pd.notna(row["record_date"])
            and pd.notna(ex_date)
            and pd.notna(row["pay_date"])
        )
        _check(checks, asset, ex_date, "dates_valid", dates_valid, "announcement/record/ex/pay")
        _check(checks, asset, ex_date, "formal_primary_key_unique", not bool(duplicate_formal.loc[position]), "asset/announcement/ex/cash")
        _check(checks, asset, ex_date, "discovery_event_unique", not bool(duplicate_discovery.loc[position]), "asset/source_event_date")
        cash = row["cash_per_share"]
        cash_valid = bool(pd.notna(cash) and np.isfinite(float(cash)) and float(cash) > 0)
        _check(checks, asset, ex_date, "cash_per_share_positive_finite", cash_valid, f"cash_per_share={cash}")
        chronology_ok = bool(
            dates_valid
            and pd.Timestamp(row["announcement_date"]) <= pd.Timestamp(row["record_date"])
            and pd.Timestamp(row["record_date"]) <= pd.Timestamp(ex_date)
            and pd.Timestamp(ex_date) <= pd.Timestamp(row["pay_date"])
        )
        _check(
            checks,
            asset,
            ex_date,
            "event_chronology",
            chronology_ok,
            f"announcement={row['announcement_date']};record={row['record_date']};ex={ex_date};pay={row['pay_date']}",
        )
        date_distance = row["ex_date_distance_days"]
        discovery_date_ok = bool(
            dates_valid
            and pd.notna(row["source_event_date"])
            and pd.notna(date_distance)
            and int(date_distance) == abs((pd.Timestamp(ex_date) - pd.Timestamp(row["source_event_date"])).days)
            and int(date_distance) <= 3
        )
        _check(checks, asset, ex_date, "discovery_ex_date_crosscheck", discovery_date_ok, f"distance_days={date_distance}")
        cash_error = row["cash_relative_error_to_discovery"]
        cash_crosscheck = bool(pd.notna(cash_error) and np.isfinite(float(cash_error)) and float(cash_error) <= 0.001)
        _check(checks, asset, ex_date, "discovery_cash_crosscheck", cash_crosscheck, f"relative_error={cash_error}")
        source_url = str(row["source_url"])
        source_type = str(row["source_type"])
        source_ok = bool(
            (asset.startswith("5") and source_type == "exchange_announcement" and source_url.startswith("https://www.sse.com.cn/disclosure/fund/announcement/"))
            or (asset.startswith("1") and source_type == "regulatory_filing" and source_url.startswith("https://static.cninfo.com.cn/finalpage/"))
        )
        _check(checks, asset, ex_date, "official_source_contract", source_ok, f"type={source_type};url={source_url}")
        review_status = str(row["review_status"])
        _check(checks, asset, ex_date, "collector_review_status_allowed", review_status in ALLOWED_REVIEW_STATUSES, review_status)
        historical_disabled = row["historical_backtest_allowed"] is False or str(row["historical_backtest_allowed"]).lower() == "false"
        _check(checks, asset, ex_date, "candidate_historical_use_stays_disabled", historical_disabled, repr(row["historical_backtest_allowed"]))
        for role in ("pdf", "text"):
            relative = str(row[f"{role}_path"])
            expected_hash = str(row[f"{role}_sha256"])
            try:
                evidence_path = _safe_path(relative)
                hash_ok = bool(evidence_path.is_file() and expected_hash and _sha256(evidence_path) == expected_hash)
            except (OSError, ValueError):
                hash_ok = False
            _check(checks, asset, ex_date, f"{role}_hash_match", hash_ok, relative)
    return pd.DataFrame(checks, columns=CHECK_COLUMNS)


def run_validation() -> dict[str, Any]:
    collector = json.loads(COLLECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not _manifest_output_matches(collector, CANDIDATE_PATH) or not _manifest_output_matches(collector, MATCH_PATH):
        raise ValueError("ETF dividend collector manifest does not authenticate its candidate outputs")
    collector_code = _authenticate_code(
        {"path": collector.get("code_path"), "sha256": collector.get("code_sha256")},
        "ETF dividend collector",
    )
    collector_dependencies = []
    dependency_roles = set()
    for entry in collector.get("code_dependencies", []):
        if not isinstance(entry, dict):
            raise ValueError("ETF dividend collector dependency entry is invalid")
        dependency_roles.add(str(entry.get("role", "")))
        collector_dependencies.append(_authenticate_code(entry, "ETF dividend collector dependency"))
    if dependency_roles != {"sse_query_contract", "cninfo_query_contract"}:
        raise ValueError(f"ETF dividend collector dependencies are incomplete: {sorted(dependency_roles)}")
    if collector.get("historical_backtest_allowed") is not False or collector.get("current_final_snapshot") is not True:
        raise ValueError("ETF dividend collector discovery boundary is invalid")

    candidates = load_candidates()
    queue = pd.read_csv(QUEUE_PATH, dtype={"asset": str})
    queue["asset"] = queue["asset"].astype(str).str.zfill(6)
    queue["source_event_date"] = pd.to_datetime(queue["source_event_date"], errors="coerce").dt.normalize()
    candidate_keys = set(zip(candidates["asset"], candidates["source_event_date"], strict=True))
    queue_keys = set(zip(queue["asset"], queue["source_event_date"], strict=True))

    checks = evaluate_candidates(candidates)
    global_rows: list[dict[str, str]] = []
    global_date = candidates["ex_date"].max()
    _check(global_rows, "ALL", global_date, "candidate_keys_match_discovery_queue", candidate_keys == queue_keys, f"candidates={len(candidate_keys)};queue={len(queue_keys)}")
    _check(global_rows, "ALL", global_date, "collector_historical_use_stays_disabled", collector.get("historical_backtest_allowed") is False, repr(collector.get("historical_backtest_allowed")))
    checks = pd.concat([checks, pd.DataFrame(global_rows, columns=CHECK_COLUMNS)], ignore_index=True)
    exceptions = checks[checks["status"].eq("fail")].copy()
    qualification = "PASS" if exceptions.empty else "FAIL"
    _atomic_csv(checks, CHECKS_PATH)
    _atomic_csv(exceptions, EXCEPTIONS_PATH)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "candidate_rows": int(len(candidates)),
        "candidate_assets": int(candidates["asset"].nunique()),
        "check_rows": int(len(checks)),
        "failed_check_rows": int(len(exceptions)),
        "coverage_start": pd.Timestamp(candidates["announcement_date"].min()).date().isoformat(),
        "coverage_end": pd.Timestamp(candidates["announcement_date"].max()).date().isoformat(),
        "maximum_cash_relative_error": float(candidates["cash_relative_error_to_discovery"].max()),
        "maximum_ex_date_distance_days": int(candidates["ex_date_distance_days"].max()),
        "boundary": "official ETF cash distributions only; no total-return price or strategy conclusion",
    }
    _atomic_json(report, REPORT_PATH)
    input_paths = [CANDIDATE_PATH, MATCH_PATH, QUEUE_PATH, COLLECTOR_MANIFEST_PATH, collector_code, *collector_dependencies]
    manifest = {
        "validation_schema": "etf_dividend_official_cross_evidence_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "inputs": [{"path": _relative(path), "sha256": _sha256(path)} for path in input_paths],
        "outputs": [
            {"role": "checks", "path": _relative(CHECKS_PATH), "sha256": _sha256(CHECKS_PATH)},
            {"role": "exceptions", "path": _relative(EXCEPTIONS_PATH), "sha256": _sha256(EXCEPTIONS_PATH)},
            {"role": "report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
        ],
        "exceptions_path": _relative(EXCEPTIONS_PATH),
        "exceptions_sha256": _sha256(EXCEPTIONS_PATH),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "candidate_rows": int(len(candidates)),
        "failed_check_rows": int(len(exceptions)),
        "independent_source_count": 2,
        "official_document_sources": ["Shanghai Stock Exchange fund announcements", "CNInfo official fund disclosures"],
        "independent_crosscheck_source": "Sina cumulative distribution markers used only for date and cash crosschecks",
        "formal_table_promotion_allowed": qualification == "PASS",
        "historical_backtest_allowed": qualification == "PASS",
        "model_promotion_allowed": False,
        "method_boundary": (
            "Approval is limited to effective-dated cash-distribution events with availability set to the official announcement date. "
            "It does not qualify total-return prices or authorize model promotion."
        ),
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
        "candidate_rows",
        "failed_check_rows",
        "independent_source_count",
        "formal_table_promotion_allowed",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
