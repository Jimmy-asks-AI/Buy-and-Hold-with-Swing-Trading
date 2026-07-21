"""Build the governed PIT ETF dividend-event table from validated official records."""

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
CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_registry_candidates.csv"
VALIDATION_MANIFEST_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_events" / "run_manifest.json"
OUTPUT_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_dividend_events.csv"
LINEAGE_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_dividend_events_builder_latest.json"
OUTPUT_COLUMNS = [
    "asset",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "available_date",
    "data_source",
    "source_vintage",
    "source_document_title",
    "source_url",
    "source_type",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_set_sha256(values: set[str]) -> str:
    normalized = sorted(str(value).strip() for value in values if str(value).strip())
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _safe_path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (ROOT / str(relative)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"ETF dividend promotion path escapes project root: {relative}")
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


def _manifest_input_matches(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("inputs", []):
        if isinstance(item, dict) and str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def _authenticate_code(entry: dict[str, Any], label: str) -> None:
    relative = str(entry.get("path", ""))
    expected_hash = str(entry.get("sha256", ""))
    path = _safe_path(relative) if relative else None
    if not path or not path.is_file() or not expected_hash or _sha256(path) != expected_hash:
        raise ValueError(f"{label} code hash mismatch")


def validate_authority(manifest: dict[str, Any]) -> None:
    required = {
        "validation_schema": "etf_dividend_official_cross_evidence_v1",
        "qualification_status": "PASS",
        "failed_check_rows": 0,
        "formal_table_promotion_allowed": True,
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in required.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"ETF dividend validation does not authorize promotion: {mismatches}")
    if not _manifest_input_matches(manifest, CANDIDATE_PATH):
        raise ValueError("ETF dividend validation does not authenticate candidates")
    _authenticate_code({"path": manifest.get("code_path"), "sha256": manifest.get("code_sha256")}, "ETF dividend validator")
    exceptions_path = _safe_path(str(manifest.get("exceptions_path", "")))
    if not exceptions_path.is_file() or _sha256(exceptions_path) != str(manifest.get("exceptions_sha256", "")) or not pd.read_csv(exceptions_path).empty:
        raise ValueError("ETF dividend validation exceptions are missing, changed, or non-empty")


def build_formal_events(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "asset",
        "announcement_date",
        "record_date",
        "ex_date",
        "pay_date",
        "cash_per_share",
        "source_document_title",
        "source_url",
        "source_type",
        "pdf_sha256",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing or candidates.empty:
        raise ValueError(f"ETF dividend candidate table is incomplete: {missing}")
    frame = candidates.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    for column in ("announcement_date", "record_date", "ex_date", "pay_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["cash_per_share"] = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    if frame[["announcement_date", "record_date", "ex_date", "pay_date", "cash_per_share"]].isna().any().any():
        raise ValueError("ETF dividend candidates have missing governed fields")
    frame["available_date"] = frame["announcement_date"]
    frame["data_source"] = frame["source_type"].map(
        {
            "exchange_announcement": "Shanghai Stock Exchange official fund announcement",
            "regulatory_filing": "CNInfo official fund disclosure",
        }
    )
    if frame["data_source"].isna().any():
        raise ValueError("ETF dividend candidates have unsupported source types")
    frame["source_vintage"] = "official_pdf_sha256:" + frame["pdf_sha256"].astype(str)
    output = frame.reindex(columns=OUTPUT_COLUMNS).sort_values(["asset", "announcement_date", "ex_date"]).reset_index(drop=True)
    if output.duplicated(["asset", "announcement_date", "ex_date", "cash_per_share"]).any():
        raise ValueError("formal ETF dividend events have duplicate primary keys")
    return output


def run_promotion() -> dict[str, Any]:
    validation = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_authority(validation)
    candidates = pd.read_csv(CANDIDATE_PATH, dtype={"asset": str})
    output = build_formal_events(candidates)
    _atomic_csv(output, OUTPUT_PATH)
    source_vintages = set(output["source_vintage"].astype(str))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": [
            {"role": "validated_candidates", "path": _relative(CANDIDATE_PATH), "sha256": _sha256(CANDIDATE_PATH)},
            {"role": "validation_authority", "path": _relative(VALIDATION_MANIFEST_PATH), "sha256": _sha256(VALIDATION_MANIFEST_PATH)},
        ],
        "outputs": [
            {"role": "pit_etf_dividend_events", "path": _relative(OUTPUT_PATH), "sha256": _sha256(OUTPUT_PATH), "rows": int(len(output))}
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "source_vintage_count": len(source_vintages),
        "source_vintage_set_sha256": _value_set_sha256(source_vintages),
        "coverage_start": pd.Timestamp(output["announcement_date"].min()).date().isoformat(),
        "coverage_end": pd.Timestamp(output["announcement_date"].max()).date().isoformat(),
        "current_final_snapshot": False,
        "contains_heuristic_corporate_actions": False,
        "qualification_status": "PROMOTED_OFFICIAL_EVENTS",
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "method_boundary": (
            "This table contains official cash-distribution events and uses announcement_date as available_date. "
            "It does not qualify any reconstructed total-return price series or strategy."
        ),
    }
    _atomic_json(manifest, LINEAGE_MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    manifest = run_promotion()
    keys = (
        "qualification_status",
        "rows",
        "assets",
        "coverage_start",
        "coverage_end",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
