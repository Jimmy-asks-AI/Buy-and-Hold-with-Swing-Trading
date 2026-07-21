"""Join pure terminal-event discovery with the validated formal event registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import pit_etf_terminal_event_universe_collector as universe


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_universe_collector_latest.json"
)
LINEAGE_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_terminal_event_universe"
    / "coverage_settlement_lineage"
)
OUTPUT_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "etf_terminal_event_settled_coverage_registry.csv"
)
REVIEW_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_terminal_event_settled_review_queue.csv"
REPORT_PATH = (
    ROOT
    / "outputs"
    / "long_hold_v4"
    / "pit_validation"
    / "etf_terminal_event_coverage_settlement"
    / "settlement_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_coverage_settlement_latest.json"
)

SCHEMA_VERSION = 1


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


def _snapshot(path: Path) -> Path:
    digest = _sha256(path)
    output = LINEAGE_DIR / f"{digest}{path.suffix.lower()}"
    if not output.is_file():
        _atomic_bytes(path.read_bytes(), output)
    if _sha256(output) != digest:
        raise ValueError(f"terminal-event settlement snapshot hash mismatch: {path}")
    return output


def _authenticate_discovery() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, list[dict[str, Any]]]:
    manifest = json.loads(DISCOVERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        != "FULL_RELATIVE_TO_AUTHENTICATED_MASTER_OFFICIAL_DISCOVERY_REQUIRES_DOCUMENT_VALIDATION"
        or int(manifest.get("target_assets", 0)) != 123
        or int(manifest.get("query_complete_assets", 0)) != 123
        or int(manifest.get("formal_terminal_events", -1)) != 0
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("terminal-event discovery is not a complete formal-independent input")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal-event discovery code hash mismatch")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    paths: dict[str, Path] = {}
    inputs: list[dict[str, Any]] = []
    manifest_snapshot = _snapshot(DISCOVERY_MANIFEST_PATH)
    inputs.append(
        {"role": "discovery_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)}
    )
    for role in ("official_announcements", "coverage_registry"):
        item = outputs.get(role)
        if item is None:
            raise ValueError(f"terminal-event discovery misses {role}")
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"terminal-event discovery output hash mismatch: {role}")
        snapshot = _snapshot(path)
        paths[role] = snapshot
        inputs.append(
            {"role": f"{role}_snapshot", "path": _relative(snapshot), "sha256": _sha256(snapshot)}
        )
    announcements = pd.read_csv(paths["official_announcements"], dtype={"asset": str})
    coverage = pd.read_csv(paths["coverage_registry"], dtype={"asset": str})
    cutoff = pd.Timestamp(str(manifest["as_of_date"])).normalize()
    return announcements, coverage, cutoff, inputs


def run() -> dict[str, Any]:
    announcements, discovery, cutoff, inputs = _authenticate_discovery()
    formal, formal_inputs = universe._load_formal_events(cutoff)
    formal_snapshots: list[dict[str, Any]] = []
    for item in formal_inputs:
        path = ROOT / str(item["path"])
        snapshot = _snapshot(path)
        formal_snapshots.append(
            {"role": f"{item['role']}_snapshot", "path": _relative(snapshot), "sha256": _sha256(snapshot)}
        )
    targets = discovery[
        ["asset", "asset_name", "exchange", "list_date", "master_delist_date"]
    ].rename(columns={"master_delist_date": "delist_date"})
    query_complete = set(discovery.loc[discovery["query_complete"].astype(str).str.lower().eq("true"), "asset"])
    settled = universe.build_coverage_registry(
        targets,
        announcements,
        query_complete,
        formal,
        cutoff=cutoff,
    )
    review = settled[settled["review_priority"].ne("DONE")].copy()
    _atomic_csv(settled, OUTPUT_PATH)
    _atomic_csv(review, REVIEW_PATH)
    state_counts = settled["final_evidence_state"].value_counts().to_dict()
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": "FORMAL_EVENTS_SETTLED_AGAINST_PURE_DISCOVERY_SCOPE_INCOMPLETE",
        "target_assets": int(len(settled)),
        "formal_event_rows": int(len(formal)),
        "formal_event_assets": int(formal["asset"].nunique()) if not formal.empty else 0,
        "identified_assets": int(settled["final_evidence_state"].eq("terminal_event_identified").sum()),
        "complete_chain_assets": int(settled["formal_event_chain_complete"].sum()),
        "evidence_insufficient_assets": int(settled["final_evidence_state"].eq("evidence_insufficient").sum()),
        "final_evidence_state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "universe_terminal_coverage_complete": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "Settlement is downstream of immutable discovery and validated events. Incomplete chains, failed cash candidates, successor conversions, and mother-universe completeness remain blocked.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "settled_coverage_registry", "path": _relative(OUTPUT_PATH), "sha256": _sha256(OUTPUT_PATH), "rows": int(len(settled))},
        {"role": "settled_review_queue", "path": _relative(REVIEW_PATH), "sha256": _sha256(REVIEW_PATH), "rows": int(len(review))},
        {"role": "settlement_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": [*inputs, *formal_snapshots],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(universe.__file__).resolve()), "sha256": _sha256(Path(universe.__file__).resolve())}
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = run()
    keys = (
        "qualification_status",
        "formal_event_rows",
        "formal_event_assets",
        "complete_chain_assets",
        "evidence_insufficient_assets",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
