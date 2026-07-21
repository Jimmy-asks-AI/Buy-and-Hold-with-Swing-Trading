"""Promote independently validated ETF terminal cash events to a formal PIT table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MANIFEST_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_cash_event" / "run_manifest.json"
)
OUTPUT_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_terminal_cash_events.csv"
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_terminal_cash_events_builder_latest.json"
)


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


def promote() -> dict[str, Any]:
    validation = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        validation.get("qualification_status") != "PASS"
        or int(validation.get("failed_check_rows", -1)) != 0
        or validation.get("formal_table_promotion_allowed") is not True
        or validation.get("validation_schema") != "etf_terminal_cash_event_known_exception_v1"
    ):
        raise ValueError("terminal cash-event validation does not authorize promotion")
    validation_code = ROOT / str(validation.get("code_path", ""))
    if not validation_code.is_file() or _sha256(validation_code) != validation.get("code_sha256"):
        raise ValueError("terminal cash-event validation code hash mismatch")
    outputs = {str(item.get("role")): item for item in validation.get("outputs", [])}
    for role in ("checks", "promotion_candidates", "report"):
        item = outputs.get(role, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"terminal cash-event validation output hash mismatch: {role}")
    checks = pd.read_csv(ROOT / str(outputs["checks"]["path"]))
    if not checks["status"].eq("pass").all():
        raise ValueError("terminal cash-event validation checks contain failures")
    candidate_path = ROOT / str(outputs["promotion_candidates"]["path"])
    candidates = pd.read_csv(candidate_path, dtype={"asset": str})
    if len(candidates) != 1 or str(candidates.iloc[0]["asset"]).zfill(6) != "511210":
        raise ValueError("terminal cash-event candidate population is unexpected")
    if not candidates["historical_backtest_allowed"].astype(str).str.lower().eq("false").all():
        raise ValueError("terminal cash-event candidate must remain disabled before promotion")
    for column in (
        "announcement_date",
        "record_date",
        "ex_date",
        "pay_date",
        "last_operation_date",
        "liquidation_start_date",
        "liquidation_end_date",
        "termination_date",
        "available_date",
    ):
        candidates[column] = pd.to_datetime(candidates[column], errors="coerce").dt.normalize()
    if candidates.isna().any(axis=None):
        raise ValueError("terminal cash-event formal row has missing fields")
    chronology = candidates.iloc[0][
        [
            "last_operation_date",
            "liquidation_start_date",
            "liquidation_end_date",
            "announcement_date",
            "record_date",
            "ex_date",
            "pay_date",
            "termination_date",
        ]
    ].tolist()
    if not all(left < right for left, right in zip(chronology[:-1], chronology[1:])):
        raise ValueError("terminal cash-event formal chronology is invalid")
    if not candidates["available_date"].eq(candidates["announcement_date"]).all():
        raise ValueError("terminal cash-event available date must equal the official announcement date")
    if not pd.to_numeric(candidates["cash_per_share"], errors="coerce").gt(0).all():
        raise ValueError("terminal cash-event cash amount is invalid")
    candidates["source_vintage"] = (
        "official_terminal_event_pdf_set_sha256:" + candidates["source_pdf_sha256_set"].astype(str)
    )
    candidates["historical_backtest_allowed"] = True
    candidates["model_promotion_allowed"] = False
    _atomic_csv(candidates, OUTPUT_PATH)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "PROMOTED_KNOWN_TERMINAL_CASH_EVENT",
        "scope_complete": False,
        "scope_boundary": "Formal 511210 event only; not evidence of exhaustive terminal-event coverage for all delisted ETFs.",
        "rows": int(len(candidates)),
        "assets": int(candidates["asset"].nunique()),
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "inputs": [
            {"role": "validated_candidates", "path": _relative(candidate_path), "sha256": _sha256(candidate_path)},
            {"role": "validation_authority", "path": _relative(VALIDATION_MANIFEST_PATH), "sha256": _sha256(VALIDATION_MANIFEST_PATH)},
        ],
        "outputs": [
            {"role": "pit_etf_terminal_cash_events", "path": _relative(OUTPUT_PATH), "sha256": _sha256(OUTPUT_PATH), "rows": int(len(candidates))}
        ],
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = promote()
    print(json.dumps({key: result[key] for key in ("qualification_status", "rows", "assets", "scope_complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
