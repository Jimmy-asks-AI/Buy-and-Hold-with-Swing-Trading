"""Reconcile the immutable ETF share-action evidence queue to governed records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_evidence_queue.csv"
REGISTRY_PATH = ROOT / "configs" / "long_hold_v4_etf_corporate_actions.json"
LEDGER_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_resolution_ledger.csv"
REMAINING_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_remaining_queue.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_share_action_resolution_ledger_latest.json"


@dataclass(frozen=True)
class ResolutionSource:
    source_id: str
    candidate_path: Path
    validation_manifest_path: Path
    promotion_manifest_path: Path
    validation_schema: str


SOURCES = (
    ResolutionSource(
        source_id="sse_official_fund_announcement",
        candidate_path=ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "sse_etf_share_action_registry_candidates.csv",
        validation_manifest_path=ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "sse_etf_share_actions" / "run_manifest.json",
        promotion_manifest_path=ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_etf_share_action_registry_promotion_latest.json",
        validation_schema="sse_etf_share_action_cross_evidence_v2",
    ),
    ResolutionSource(
        source_id="cninfo_official_fund_disclosure",
        candidate_path=ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "cninfo_etf_share_action_registry_candidates.csv",
        validation_manifest_path=ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "cninfo_etf_share_actions" / "run_manifest.json",
        promotion_manifest_path=ROOT / "data_raw" / "long_hold_v4" / "manifests" / "cninfo_etf_share_action_registry_promotion_latest.json",
        validation_schema="cninfo_etf_share_action_cross_evidence_v1",
    ),
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
        raise ValueError(f"share-action ledger path escapes project root: {relative}")
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


def _manifest_references_path(manifest: dict[str, Any], path: Path) -> bool:
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


def _load_authenticated_candidates(source: ResolutionSource) -> pd.DataFrame:
    validation = json.loads(source.validation_manifest_path.read_text(encoding="utf-8"))
    promotion = json.loads(source.promotion_manifest_path.read_text(encoding="utf-8"))
    expected_validation = {
        "validation_schema": source.validation_schema,
        "qualification_status": "PASS",
        "registry_promotion_allowed": True,
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "failed_check_rows": 0,
    }
    if any(validation.get(key) != value for key, value in expected_validation.items()):
        raise ValueError(f"{source.source_id} validation authority is not current and passing")
    if not _manifest_references_path(validation, source.candidate_path):
        raise ValueError(f"{source.source_id} validation does not authenticate candidates")
    _authenticate_code(
        {"path": validation.get("code_path"), "sha256": validation.get("code_sha256")},
        f"{source.source_id} validator",
    )
    for dependency in validation.get("code_dependencies", []):
        _authenticate_code(dependency, f"{source.source_id} validator dependency")

    expected_promotion = {
        "source_id": "cninfo" if source.source_id.startswith("cninfo") else "sse",
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    if any(promotion.get(key) != value for key, value in expected_promotion.items()):
        raise ValueError(f"{source.source_id} promotion manifest is not authorized")
    if promotion.get("qualification_status") not in {"PROMOTED", "ALREADY_PROMOTED"}:
        raise ValueError(f"{source.source_id} promotion did not complete")
    if not _manifest_references_path(promotion, source.candidate_path):
        raise ValueError(f"{source.source_id} promotion does not authenticate candidates")
    if not _manifest_references_path(promotion, source.validation_manifest_path):
        raise ValueError(f"{source.source_id} promotion does not authenticate validation authority")
    _authenticate_code(
        {"path": promotion.get("code_path"), "sha256": promotion.get("code_sha256")},
        f"{source.source_id} promoter",
    )

    frame = pd.read_csv(source.candidate_path, dtype={"asset": str})
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["price_effective_date"] = pd.to_datetime(frame["price_effective_date"], errors="coerce").dt.normalize()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"], errors="coerce").dt.normalize()
    frame["shares_after_per_share_before"] = pd.to_numeric(frame["shares_after_per_share_before"], errors="coerce")
    if frame.empty or frame[["price_effective_date", "event_date", "announcement_date", "shares_after_per_share_before"]].isna().any().any():
        raise ValueError(f"{source.source_id} candidates contain missing governed fields")
    if frame.duplicated(["asset", "price_effective_date"]).any():
        raise ValueError(f"{source.source_id} candidates contain duplicate event keys")
    frame["resolution_source"] = source.source_id
    frame["validation_schema"] = source.validation_schema
    frame["validation_manifest_sha256"] = _sha256(source.validation_manifest_path)
    frame["promotion_manifest_sha256"] = _sha256(source.promotion_manifest_path)
    return frame


def build_resolution_ledger(
    queue: pd.DataFrame,
    candidates: pd.DataFrame,
    registry_actions: pd.DataFrame,
) -> pd.DataFrame:
    queue = queue.copy()
    queue["asset"] = queue["asset"].astype(str).str.zfill(6)
    queue["price_effective_date"] = pd.to_datetime(queue["price_effective_date"], errors="coerce").dt.normalize()
    queue["inferred_factor"] = pd.to_numeric(queue["inferred_factor"], errors="coerce")
    if queue.empty or queue[["price_effective_date", "inferred_factor"]].isna().any().any():
        raise ValueError("share-action evidence queue is empty or malformed")
    if queue.duplicated(["asset", "price_effective_date"]).any():
        raise ValueError("share-action evidence queue has duplicate event keys")

    candidates = candidates.copy()
    registry = registry_actions.copy()
    for frame in (candidates, registry):
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
        frame["price_effective_date"] = pd.to_datetime(frame["price_effective_date"], errors="coerce").dt.normalize()
    if candidates.duplicated(["asset", "price_effective_date"]).any():
        raise ValueError("combined official candidates contain duplicate event keys")
    if registry.duplicated(["asset", "price_effective_date"]).any():
        raise ValueError("governed registry contains duplicate event keys")

    candidate_lookup = candidates.set_index(["asset", "price_effective_date"], drop=False)
    registry_lookup = registry.set_index(["asset", "price_effective_date"], drop=False)
    rows: list[dict[str, Any]] = []
    for row in queue.to_dict("records"):
        key = (str(row["asset"]).zfill(6), pd.Timestamp(row["price_effective_date"]).normalize())
        output = dict(row)
        output.update(
            {
                "resolution_status": "unresolved_missing_official_candidate",
                "resolution_source": None,
                "official_action_type": None,
                "official_event_date": pd.NaT,
                "official_announcement_date": pd.NaT,
                "official_factor": np.nan,
                "factor_correction_pct": np.nan,
                "official_source_url": None,
                "validation_schema": None,
                "validation_manifest_sha256": None,
                "promotion_manifest_sha256": None,
                "registry_match": False,
            }
        )
        if key in candidate_lookup.index:
            candidate = candidate_lookup.loc[key]
            if isinstance(candidate, pd.DataFrame):
                raise ValueError(f"official candidate key is ambiguous: {key}")
            output.update(
                {
                    "resolution_status": "unresolved_registry_mismatch",
                    "resolution_source": candidate["resolution_source"],
                    "official_action_type": candidate["action_type"],
                    "official_event_date": candidate["event_date"],
                    "official_announcement_date": candidate["announcement_date"],
                    "official_factor": float(candidate["shares_after_per_share_before"]),
                    "factor_correction_pct": (
                        float(candidate["shares_after_per_share_before"]) / float(row["inferred_factor"]) - 1.0
                    )
                    * 100.0,
                    "official_source_url": candidate["source_url"],
                    "validation_schema": candidate["validation_schema"],
                    "validation_manifest_sha256": candidate["validation_manifest_sha256"],
                    "promotion_manifest_sha256": candidate["promotion_manifest_sha256"],
                }
            )
            if key in registry_lookup.index:
                governed = registry_lookup.loc[key]
                if isinstance(governed, pd.DataFrame):
                    raise ValueError(f"governed registry key is ambiguous: {key}")
                registry_match = bool(
                    abs(float(governed["shares_after_per_share_before"]) - float(candidate["shares_after_per_share_before"])) <= 1e-12
                    and pd.Timestamp(governed["event_date"]).normalize() == pd.Timestamp(candidate["event_date"]).normalize()
                    and str(governed["source_url"]) == str(candidate["source_url"])
                    and str(governed.get("review_status", "")) == "verified"
                )
                output["registry_match"] = registry_match
                if registry_match:
                    output["resolution_status"] = "resolved_governed_official"
        rows.append(output)
    return pd.DataFrame(rows).sort_values(["asset", "price_effective_date"]).reset_index(drop=True)


def run_reconciliation() -> dict[str, Any]:
    source_frames = [_load_authenticated_candidates(source) for source in SOURCES]
    candidates = pd.concat(source_frames, ignore_index=True)
    queue = pd.read_csv(QUEUE_PATH, dtype={"asset": str})
    registry_payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = pd.DataFrame(registry_payload.get("actions", []))
    ledger = build_resolution_ledger(queue, candidates, registry)
    remaining = ledger[~ledger["resolution_status"].eq("resolved_governed_official")].copy()
    _atomic_csv(ledger, LEDGER_PATH)
    _atomic_csv(remaining, REMAINING_QUEUE_PATH)

    input_paths = [QUEUE_PATH, REGISTRY_PATH]
    for source in SOURCES:
        input_paths.extend(
            [source.candidate_path, source.validation_manifest_path, source.promotion_manifest_path]
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": [{"path": _relative(path), "sha256": _sha256(path)} for path in input_paths],
        "outputs": [
            {"role": "resolution_ledger", "path": _relative(LEDGER_PATH), "sha256": _sha256(LEDGER_PATH), "rows": int(len(ledger))},
            {"role": "remaining_queue", "path": _relative(REMAINING_QUEUE_PATH), "sha256": _sha256(REMAINING_QUEUE_PATH), "rows": int(len(remaining))},
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "queue_rows": int(len(queue)),
        "resolved_rows": int(ledger["resolution_status"].eq("resolved_governed_official").sum()),
        "remaining_rows": int(len(remaining)),
        "official_correction_rows": int(ledger["factor_correction_pct"].abs().gt(0.1).sum()),
        "qualification_status": "PASS" if remaining.empty else "INCOMPLETE",
        "historical_backtest_allowed": remaining.empty,
        "model_promotion_allowed": False,
        "method_boundary": (
            "Resolution qualifies official ETF share-action normalization only. The original queue remains immutable, "
            "and this ledger does not qualify current-final prices or strategy performance."
        ),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    manifest = run_reconciliation()
    keys = (
        "qualification_status",
        "queue_rows",
        "resolved_rows",
        "remaining_rows",
        "official_correction_rows",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
