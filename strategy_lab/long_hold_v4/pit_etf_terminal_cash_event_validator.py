"""Independently validate the known 511210 terminal cash-event evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_cash_event_collector_latest.json"
)
ETF_MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
PROVIDER_DIVIDEND_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_total_return"
    / "511210"
    / "dividend.csv.gz"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_cash_event"
CHECKS_PATH = OUTPUT_DIR / "checks.csv"
PROMOTION_CANDIDATE_PATH = OUTPUT_DIR / "promotion_candidates.csv"
REPORT_PATH = OUTPUT_DIR / "report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

EXPECTED = {
    "asset": "511210",
    "event_type": "liquidation_distribution",
    "announcement_date": "2018-01-09",
    "record_date": "2018-01-16",
    "ex_date": "2018-01-17",
    "pay_date": "2018-01-23",
    "cash_per_share": 112.79,
    "last_operation_date": "2017-11-07",
    "liquidation_start_date": "2017-11-08",
    "liquidation_end_date": "2017-11-23",
    "liquidation_nav": 112.6579,
    "liquidation_net_assets": 8_897_157.57,
    "liquidation_shares": 78_975.0,
    "termination_date": "2018-01-26",
}


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


def _check(rows: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def validate() -> dict[str, Any]:
    collector = json.loads(COLLECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    checks: list[dict[str, str]] = []
    _check(checks, "collector_qualification", collector.get("qualification_status") == "KNOWN_TERMINAL_EVENT_OFFICIAL_EVIDENCE_REQUIRES_VALIDATION", str(collector.get("qualification_status")))
    _check(checks, "collector_scope_not_overstated", collector.get("scope_complete") is False, str(collector.get("scope_boundary")))
    code_path = ROOT / str(collector.get("code_path", ""))
    _check(checks, "collector_code_hash", code_path.is_file() and _sha256(code_path) == collector.get("code_sha256"), str(collector.get("code_path")))

    authenticated_outputs: list[dict[str, Any]] = []
    for output in collector.get("outputs", []):
        path = ROOT / str(output.get("path", ""))
        passed = path.is_file() and _sha256(path) == str(output.get("sha256", ""))
        _check(checks, f"output_hash:{output.get('role')}", passed, str(output.get("path")))
        if passed:
            authenticated_outputs.append(output)
    for item in collector.get("inputs", []):
        path = ROOT / str(item.get("path", ""))
        _check(
            checks,
            f"input_hash:{item.get('role')}",
            path.is_file() and _sha256(path) == str(item.get("sha256", "")),
            str(item.get("path")),
        )

    documents = {str(item["role"]): item for item in collector.get("documents", [])}
    _check(checks, "four_official_document_roles", set(documents) == {"resolution", "liquidation_report", "distribution", "delisting"}, ",".join(sorted(documents)))
    query_output = next((item for item in authenticated_outputs if item.get("role") == "query"), None)
    query_path = ROOT / str(query_output.get("path", "")) if query_output else Path("missing")
    query = json.loads(query_path.read_text(encoding="utf-8")) if query_path.is_file() else {"responses": []}
    query_rows_by_role = {str(item.get("role")): item.get("rows", []) for item in query.get("responses", [])}
    for role, document in documents.items():
        source_url = str(document.get("source_url", ""))
        text_path = ROOT / str(document.get("text_path", ""))
        text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
        _check(checks, f"official_source_domain:{role}", source_url.startswith("https://www.sse.com.cn/disclosure/fund/announcement/"), source_url)
        _check(checks, f"fund_identity_in_text:{role}", "上证企债" in text, str(document.get("title")))
        matching_query_rows = [
            row
            for row in query_rows_by_role.get(role, [])
            if str(row.get("SECURITY_CODE", "")) == "511210"
            and str(row.get("TITLE", "")) == str(document.get("title", ""))
            and str(row.get("URL", "")) in source_url
        ]
        _check(checks, f"asset_identity_in_query:{role}", len(matching_query_rows) == 1, f"matches={len(matching_query_rows)}")

    candidate_output = next((item for item in authenticated_outputs if item.get("role") == "candidates"), None)
    candidate_path = ROOT / str(candidate_output.get("path", "")) if candidate_output else Path("missing")
    candidates = pd.read_csv(candidate_path, dtype={"asset": str}) if candidate_path.is_file() else pd.DataFrame()
    _check(checks, "one_candidate_row", len(candidates) == 1, f"rows={len(candidates)}")
    candidate = candidates.iloc[0] if len(candidates) == 1 else pd.Series(dtype=object)
    for field, expected in EXPECTED.items():
        actual = candidate.get(field)
        if isinstance(expected, float):
            passed = pd.notna(actual) and np.isclose(float(actual), expected, rtol=0.0, atol=1e-6)
        else:
            passed = str(actual) == expected
        _check(checks, f"candidate_value:{field}", bool(passed), f"actual={actual};expected={expected}")
    _check(checks, "candidate_disabled_before_promotion", str(candidate.get("historical_backtest_allowed")).lower() == "false", str(candidate.get("historical_backtest_allowed")))
    _check(checks, "available_on_announcement", str(candidate.get("available_date")) == EXPECTED["announcement_date"], str(candidate.get("available_date")))
    _check(checks, "position_extinguished", str(candidate.get("extinguishes_position")).lower() == "true", str(candidate.get("extinguishes_position")))

    chronology_columns = [
        "last_operation_date",
        "liquidation_start_date",
        "liquidation_end_date",
        "announcement_date",
        "record_date",
        "ex_date",
        "pay_date",
        "termination_date",
    ]
    chronology = pd.to_datetime([candidate.get(column) for column in chronology_columns], errors="coerce")
    chronology_pass = not chronology.isna().any() and all(left < right for left, right in zip(chronology[:-1], chronology[1:]))
    _check(checks, "terminal_event_chronology", bool(chronology_pass), ">".join(str(value.date()) for value in chronology if pd.notna(value)))

    if pd.notna(candidate.get("liquidation_net_assets")) and pd.notna(candidate.get("liquidation_shares")):
        derived_nav = float(candidate["liquidation_net_assets"]) / float(candidate["liquidation_shares"])
    else:
        derived_nav = float("nan")
    _check(checks, "liquidation_nav_reconciles", np.isclose(derived_nav, float(candidate.get("liquidation_nav", np.nan)), rtol=0.0, atol=5e-5), f"derived={derived_nav:.8f}")

    marker = pd.read_csv(PROVIDER_DIVIDEND_PATH)
    marker_date = pd.to_datetime(marker.get("date"), errors="coerce")
    marker_cash = pd.to_numeric(marker.get("cumulative_dividend"), errors="coerce")
    marker_pass = len(marker) == 1 and str(marker_date.iloc[0].date()) == EXPECTED["ex_date"] and np.isclose(float(marker_cash.iloc[0]), EXPECTED["cash_per_share"], atol=1e-8)
    _check(checks, "provider_marker_crosscheck", bool(marker_pass), marker.to_json(orient="records", force_ascii=False))

    master = pd.read_csv(ETF_MASTER_PATH, dtype={"asset": str})
    lifecycle = master[master["asset"].eq("511210")]
    listing = lifecycle[lifecycle["event_type"].eq("listing")]
    delisting_rows = lifecycle[lifecycle["event_type"].eq("delisting")]
    master_pass = len(listing) == 1 and len(delisting_rows) == 1
    if master_pass:
        last_listed = pd.Timestamp(delisting_rows.iloc[0]["delist_date"])
        termination = pd.Timestamp(EXPECTED["termination_date"])
        master_pass = last_listed + pd.Timedelta(days=1) == termination
    _check(checks, "master_lifecycle_reconciles", bool(master_pass), lifecycle.to_json(orient="records", force_ascii=False))

    failed = sum(item["status"] == "fail" for item in checks)
    qualification = "PASS" if failed == 0 else "FAIL"
    _atomic_csv(pd.DataFrame(checks), CHECKS_PATH)
    promotion = candidates.copy()
    _atomic_csv(promotion, PROMOTION_CANDIDATE_PATH)
    report = {
        "qualification_status": qualification,
        "failed_check_rows": failed,
        "candidate_rows": int(len(promotion)),
        "formal_table_promotion_allowed": failed == 0,
        "historical_backtest_allowed": failed == 0,
        "model_promotion_allowed": False,
        "scope_complete": False,
        "scope_boundary": "Validated 511210 known exception only; full delisted-ETF terminal-event discovery remains required.",
    }
    _atomic_json(report, REPORT_PATH)
    inputs = [
        {"path": _relative(COLLECTOR_MANIFEST_PATH), "sha256": _sha256(COLLECTOR_MANIFEST_PATH)},
        {"path": _relative(ETF_MASTER_PATH), "sha256": _sha256(ETF_MASTER_PATH)},
        {"path": _relative(PROVIDER_DIVIDEND_PATH), "sha256": _sha256(PROVIDER_DIVIDEND_PATH)},
    ]
    inputs.extend(
        {"path": str(item["path"]), "sha256": str(item["sha256"])}
        for item in authenticated_outputs
    )
    outputs = [
        {"role": "checks", "path": _relative(CHECKS_PATH), "sha256": _sha256(CHECKS_PATH)},
        {"role": "promotion_candidates", "path": _relative(PROMOTION_CANDIDATE_PATH), "sha256": _sha256(PROMOTION_CANDIDATE_PATH)},
        {"role": "report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "validation_schema": "etf_terminal_cash_event_known_exception_v1",
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = validate()
    print(json.dumps({key: result[key] for key in ("qualification_status", "failed_check_rows", "formal_table_promotion_allowed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
