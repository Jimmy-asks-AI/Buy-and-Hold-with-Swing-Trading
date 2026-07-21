"""Promote independently validated full-universe ETF dividend candidates."""

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
CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_dividend_universe_validated_candidates.csv"
VALIDATION_MANIFEST_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_universe_events" / "run_manifest.json"
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
        raise ValueError(f"ETF dividend universe promotion path escapes project root: {relative}")
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


def _manifest_output_matches(manifest: dict[str, Any], role: str, path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("outputs", []):
        if not isinstance(item, dict) or str(item.get("role", "")) != role:
            continue
        return bool(
            str(item.get("path", "")).replace("\\", "/") == relative
            and path.is_file()
            and str(item.get("sha256", "")) == _sha256(path)
        )
    return False


def validate_authority(manifest: dict[str, Any]) -> None:
    required = {
        "validation_schema": "etf_dividend_full_universe_official_v1",
        "qualification_status": "PASS",
        "full_universe_query_complete": True,
        "official_document_failures": 0,
        "unresolved_review_documents": 0,
        "new_official_event_rows": 4,
        "formal_candidate_rows": 863,
        "failed_check_rows": 0,
        "formal_table_promotion_allowed": True,
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"ETF dividend universe validation does not authorize promotion: {mismatches}")
    if not _manifest_output_matches(manifest, "promotion_candidates", CANDIDATE_PATH):
        raise ValueError("ETF dividend universe validator does not authenticate promotion candidates")
    code_path = _safe_path(str(manifest.get("code_path", "")))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF dividend universe validator code hash mismatch")
    exceptions_path = _safe_path(str(manifest.get("exceptions_path", "")))
    if (
        not exceptions_path.is_file()
        or _sha256(exceptions_path) != str(manifest.get("exceptions_sha256", ""))
        or not pd.read_csv(exceptions_path).empty
    ):
        raise ValueError("ETF dividend universe validation exceptions are missing, changed, or non-empty")


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
        "historical_backtest_allowed",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing or candidates.empty:
        raise ValueError(f"ETF dividend universe candidate table is incomplete: {missing}")
    frame = candidates.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    for column in ("announcement_date", "record_date", "ex_date", "pay_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    frame["cash_per_share"] = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    if frame[["announcement_date", "record_date", "ex_date", "pay_date", "cash_per_share"]].isna().any().any():
        raise ValueError("ETF dividend universe candidates have missing governed fields")
    if not frame["historical_backtest_allowed"].astype(str).str.lower().eq("false").all():
        raise ValueError("ETF dividend universe candidate rows must remain disabled before promotion")
    chronology = (
        frame["announcement_date"].le(frame["record_date"])
        & frame["record_date"].le(frame["ex_date"])
        & frame["ex_date"].le(frame["pay_date"])
    )
    if not chronology.all() or not frame["cash_per_share"].gt(0).all():
        raise ValueError("ETF dividend universe candidates violate chronology or cash constraints")
    frame["available_date"] = frame["announcement_date"]
    frame["data_source"] = frame["source_type"].map(
        {
            "exchange_announcement": "Shanghai Stock Exchange official fund announcement",
            "regulatory_filing": "CNInfo official fund disclosure",
        }
    )
    if frame["data_source"].isna().any():
        raise ValueError("ETF dividend universe candidates have unsupported source types")
    frame["source_vintage"] = "official_pdf_sha256:" + frame["pdf_sha256"].astype(str)
    output = frame.reindex(columns=OUTPUT_COLUMNS).sort_values(
        ["asset", "announcement_date", "ex_date"]
    ).reset_index(drop=True)
    if output.duplicated(["asset", "announcement_date", "ex_date", "cash_per_share"]).any():
        raise ValueError("formal ETF dividend events have duplicate primary keys")
    if output.duplicated(["asset", "ex_date", "cash_per_share"]).any():
        raise ValueError("formal ETF dividend events have duplicate economic event keys")
    return output


def run_promotion() -> dict[str, Any]:
    validation = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_authority(validation)
    candidates = pd.read_csv(CANDIDATE_PATH, dtype={"asset": str})
    output = build_formal_events(candidates)
    if len(output) != int(validation["formal_candidate_rows"]):
        raise ValueError("ETF dividend formal row count differs from validation authority")
    _atomic_csv(output, OUTPUT_PATH)
    source_vintages = set(output["source_vintage"].astype(str))
    manifest = {
        "schema_version": 2,
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
        "coverage_start": str(pd.Timestamp(output["announcement_date"].min()).date()),
        "coverage_end": str(pd.Timestamp(output["announcement_date"].max()).date()),
        "current_final_snapshot": False,
        "contains_heuristic_corporate_actions": False,
        "qualification_status": "PROMOTED_FULL_UNIVERSE_OFFICIAL_EVENTS",
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "method_boundary": (
            "This table contains independently validated official cash-distribution events and uses announcement_date "
            "as available_date. It does not qualify total-return prices or authorize model promotion."
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
