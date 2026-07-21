"""Promote independently validated ETF terminal-event chains to the PIT registry."""

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

from . import pit_etf_terminal_event_validator_v2 as validator


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MANIFEST_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_v2" / "run_manifest.json"
)
OUTPUT_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_terminal_cash_events.csv"
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_terminal_cash_events_builder_latest.json"
)

SCHEMA_VERSION = 2


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


def _bool(series: pd.Series, column: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(values).difference({"true", "false"}))
    if unknown:
        raise ValueError(f"terminal-event promotion has invalid {column}: {unknown}")
    return values.eq("true")


def _authenticated_validation() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "PASS_WITH_QUARANTINED_CANDIDATES"
        or manifest.get("validation_schema") != "etf_terminal_event_chain_v2"
        or manifest.get("formal_table_promotion_allowed") is not True
        or manifest.get("scope_complete") is not False
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
    ):
        raise ValueError("terminal-event V2 validation does not authorize formal promotion")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal-event V2 validator code hash mismatch")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    required = {"validation_checks", "asset_chain_validation", "promotion_candidates", "validation_report"}
    inputs = [
        {"role": "validation_manifest", "path": _relative(VALIDATION_MANIFEST_PATH), "sha256": _sha256(VALIDATION_MANIFEST_PATH)}
    ]
    paths: dict[str, Path] = {}
    for role in required:
        item = outputs.get(role)
        if item is None:
            raise ValueError(f"terminal-event V2 validation misses output: {role}")
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"terminal-event V2 validation output hash mismatch: {role}")
        paths[role] = path
        inputs.append({"role": role, "path": _relative(path), "sha256": _sha256(path)})
    checks = pd.read_csv(paths["validation_checks"], dtype={"asset": str})
    candidates = pd.read_csv(paths["promotion_candidates"], dtype={"asset": str})
    if len(candidates) != int(manifest.get("validated_event_rows", -1)) or candidates.empty:
        raise ValueError("terminal-event V2 promotion population does not match validation manifest")
    if not candidates["validation_status"].eq("pass").all():
        raise ValueError("terminal-event V2 promotion candidates contain failed rows")
    if int(checks["validation_status"].eq("pass").sum()) != len(candidates):
        raise ValueError("terminal-event V2 checks do not reconcile to promotion candidates")
    return candidates, manifest, inputs


def _validate_formal_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(validator.FORMAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"terminal-event V2 formal table misses fields: {missing}")
    formal = frame.reindex(columns=validator.FORMAL_COLUMNS).copy()
    formal["asset"] = formal["asset"].astype(str).str.zfill(6)
    required_dates = (
        "announcement_date",
        "available_trade_date",
        "available_date",
        "entitlement_date",
        "record_date",
        "pay_date",
        "accounting_date",
    )
    optional_dates = ("ex_date", "termination_date")
    for column in (*required_dates, *optional_dates):
        formal[column] = pd.to_datetime(formal[column], errors="coerce").dt.normalize()
    if formal[list(required_dates)].isna().any(axis=None):
        raise ValueError("terminal-event V2 formal table has missing required dates")
    formal["cash_per_share"] = pd.to_numeric(formal["cash_per_share"], errors="coerce")
    if not np.isfinite(formal["cash_per_share"]).all() or formal["cash_per_share"].le(0).any():
        raise ValueError("terminal-event V2 formal table has invalid cash amounts")
    for column in (
        "is_final_distribution",
        "additional_distribution_expected",
        "fund_contract_terminated",
        "exit_registration_announced",
        "extinguishes_position",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    ):
        formal[column] = _bool(formal[column], column)
    if not formal["event_type"].eq("liquidation_distribution").all():
        raise ValueError("terminal-event V2 formal table has unsupported event types")
    if not formal["historical_backtest_allowed"].all() or formal["model_promotion_allowed"].any():
        raise ValueError("terminal-event V2 formal table violates promotion boundaries")
    chronology = (
        formal["entitlement_date"].le(formal["pay_date"])
        & formal["pay_date"].le(formal["accounting_date"])
        & formal["available_trade_date"].le(formal["accounting_date"])
        & formal["available_date"].eq(formal["available_trade_date"])
    )
    if not chronology.all():
        raise ValueError("terminal-event V2 formal table has invalid PIT chronology")
    if formal["event_id"].astype(str).duplicated().any():
        raise ValueError("terminal-event V2 formal table has duplicate event IDs")
    if (
        (formal["is_final_distribution"] & formal["additional_distribution_expected"]).any()
        or (formal["extinguishes_position"] & ~formal["is_final_distribution"]).any()
    ):
        raise ValueError("terminal-event V2 final-distribution semantics are contradictory")
    economic_key = ["asset", "event_type", "holder_scope", "pay_date", "source_pdf_sha256_set"]
    if formal.duplicated(economic_key, keep=False).any():
        raise ValueError("terminal-event V2 formal table has duplicate economic keys")
    for (_, _), group in formal.groupby(["asset", "holder_scope"]):
        ordered = group.sort_values("distribution_sequence")
        expected = list(range(1, len(ordered) + 1))
        if ordered["distribution_sequence"].astype(int).tolist() != expected:
            raise ValueError("terminal-event V2 distribution sequence is not contiguous")
        extinguishing = ordered["extinguishes_position"]
        if int(extinguishing.sum()) > 1 or (extinguishing.any() and not bool(extinguishing.iloc[-1])):
            raise ValueError("terminal-event V2 extinguishment must occur once on the final event")
    if not formal["source_vintage"].astype(str).str.startswith(
        "official_terminal_event_pdf_set_sha256:"
    ).all():
        raise ValueError("terminal-event V2 formal table has invalid source vintage")
    return formal.sort_values(["asset", "holder_scope", "distribution_sequence"]).reset_index(drop=True)


def promote() -> dict[str, Any]:
    candidates, validation, inputs = _authenticated_validation()
    formal = _validate_formal_table(candidates)
    _atomic_csv(formal, OUTPUT_PATH)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2",
        "scope_complete": False,
        "scope_boundary": (
            "Promoted passed cash-event rows only. Quarantined cash candidates, successor conversions, "
            "uncovered delisted-ETF mother-universe rows, and incomplete terminal chains remain outside full lifecycle closure."
        ),
        "rows": int(len(formal)),
        "assets": int(formal["asset"].nunique()),
        "complete_chain_assets": int(validation.get("complete_event_chain_assets", 0)),
        "quarantined_candidate_rows": int(validation.get("quarantined_candidate_rows", 0)),
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "inputs": inputs,
        "outputs": [
            {
                "role": "pit_etf_terminal_cash_events",
                "path": _relative(OUTPUT_PATH),
                "sha256": _sha256(OUTPUT_PATH),
                "rows": int(len(formal)),
            }
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = promote()
    keys = (
        "qualification_status",
        "rows",
        "assets",
        "complete_chain_assets",
        "quarantined_candidate_rows",
        "scope_complete",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
