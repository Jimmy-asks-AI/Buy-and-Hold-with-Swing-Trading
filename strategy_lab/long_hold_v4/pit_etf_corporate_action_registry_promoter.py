"""Promote independently validated ETF share actions into the governed registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
VALIDATION_ROOT = ROOT / "outputs" / "long_hold_v4" / "pit_validation"
MANIFEST_DIR = ROOT / "data_raw" / "long_hold_v4" / "manifests"
REGISTRY_PATH = ROOT / "configs" / "long_hold_v4_etf_corporate_actions.json"
ARCHIVE_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "registry_snapshots"
ACTION_FIELDS = (
    "asset",
    "action_type",
    "event_date",
    "price_effective_date",
    "shares_after_per_share_before",
    "announcement_date",
    "source_document_title",
    "source_url",
    "source_type",
    "review_status",
)


@dataclass(frozen=True)
class PromotionSpec:
    source_id: str
    candidate_path: Path
    validation_manifest_path: Path
    promotion_manifest_path: Path
    validation_schema: str
    official_source_url_prefix: str
    expected_source_type: str


PROMOTION_SPECS = {
    "sse": PromotionSpec(
        source_id="sse",
        candidate_path=OBSERVATION_DIR / "sse_etf_share_action_registry_candidates.csv",
        validation_manifest_path=VALIDATION_ROOT / "sse_etf_share_actions" / "run_manifest.json",
        promotion_manifest_path=MANIFEST_DIR / "sse_etf_share_action_registry_promotion_latest.json",
        validation_schema="sse_etf_share_action_cross_evidence_v2",
        official_source_url_prefix="https://www.sse.com.cn/disclosure/fund/announcement/",
        expected_source_type="exchange_announcement",
    ),
    "cninfo": PromotionSpec(
        source_id="cninfo",
        candidate_path=OBSERVATION_DIR / "cninfo_etf_share_action_registry_candidates.csv",
        validation_manifest_path=VALIDATION_ROOT / "cninfo_etf_share_actions" / "run_manifest.json",
        promotion_manifest_path=MANIFEST_DIR / "cninfo_etf_share_action_registry_promotion_latest.json",
        validation_schema="cninfo_etf_share_action_cross_evidence_v1",
        official_source_url_prefix="https://static.cninfo.com.cn/finalpage/",
        expected_source_type="regulatory_filing",
    ),
}


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
        raise ValueError(f"registry promotion path escapes project root: {relative}")
    return path


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), path)


def _manifest_input_is_authenticated(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("inputs", []):
        if isinstance(item, dict) and str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def _authenticate_code_entry(entry: dict[str, Any], label: str) -> None:
    relative = str(entry.get("path", ""))
    expected_hash = str(entry.get("sha256", ""))
    path = _safe_path(relative) if relative else None
    if not path or not path.is_file() or not expected_hash or _sha256(path) != expected_hash:
        raise ValueError(f"{label} code hash does not match the promotion authority")


def validate_promotion_authority(manifest: dict[str, Any], spec: PromotionSpec) -> None:
    required = {
        "qualification_status": "PASS",
        "registry_promotion_allowed": True,
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "validation_schema": spec.validation_schema,
        "failed_check_rows": 0,
    }
    mismatches = {
        field: (manifest.get(field), expected)
        for field, expected in required.items()
        if manifest.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"share-action validation manifest does not authorize registry promotion: {mismatches}")
    if not _manifest_input_is_authenticated(manifest, spec.candidate_path):
        raise ValueError("share-action validation manifest does not authenticate the candidate table")
    _authenticate_code_entry(
        {"path": manifest.get("code_path"), "sha256": manifest.get("code_sha256")},
        "share-action validator",
    )
    dependencies = manifest.get("code_dependencies", [])
    if not dependencies:
        raise ValueError("share-action validation manifest omits validator code dependencies")
    for entry in dependencies:
        if not isinstance(entry, dict):
            raise ValueError("share-action validator code dependency entry is invalid")
        _authenticate_code_entry(entry, f"share-action validator dependency {entry.get('role', 'unknown')}")
    exceptions_relative = str(manifest.get("exceptions_path", ""))
    exceptions_path = _safe_path(exceptions_relative) if exceptions_relative else None
    if (
        not exceptions_path
        or not exceptions_path.is_file()
        or _sha256(exceptions_path) != str(manifest.get("exceptions_sha256", ""))
        or not pd.read_csv(exceptions_path).empty
    ):
        raise ValueError("share-action validation exceptions are missing, changed, or non-empty")


def candidate_actions(
    frame: pd.DataFrame,
    *,
    official_source_url_prefix: str,
    expected_source_type: str,
) -> list[dict[str, Any]]:
    required = {
        "asset",
        "action_type",
        "event_date",
        "price_effective_date",
        "shares_after_per_share_before",
        "announcement_date",
        "source_document_title",
        "source_url",
        "source_type",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"share-action candidate table is incomplete: {missing}")
    actions: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        source_url = str(row.source_url)
        source_type = str(row.source_type)
        if not source_url.startswith(official_source_url_prefix):
            raise ValueError(f"candidate has a non-official source URL: {source_url}")
        if source_type != expected_source_type:
            raise ValueError(f"candidate has an unexpected source type: {source_type}")
        actions.append(
            {
                "asset": str(row.asset).zfill(6),
                "action_type": str(row.action_type),
                "event_date": pd.Timestamp(row.event_date).date().isoformat(),
                "price_effective_date": pd.Timestamp(row.price_effective_date).date().isoformat(),
                "shares_after_per_share_before": float(row.shares_after_per_share_before),
                "announcement_date": pd.Timestamp(row.announcement_date).date().isoformat(),
                "source_document_title": str(row.source_document_title),
                "source_url": source_url,
                "source_type": source_type,
                "review_status": "verified",
            }
        )
    return actions


def _action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(action.get(field) for field in ACTION_FIELDS)


def merge_registry_payload(payload: dict[str, Any], additions: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    if payload.get("schema_version") != 1 or payload.get("factor_definition") != "shares_after_per_share_before":
        raise ValueError("unsupported ETF corporate-action registry schema")
    existing = list(payload.get("actions", []))
    by_key = {(str(item["asset"]).zfill(6), str(item["price_effective_date"])): item for item in existing}
    if len(by_key) != len(existing):
        raise ValueError("existing ETF corporate-action registry has duplicate effective actions")
    added = 0
    for action in additions:
        key = (str(action["asset"]).zfill(6), str(action["price_effective_date"]))
        prior = by_key.get(key)
        if prior is not None:
            if _action_signature(prior) != _action_signature(action):
                raise ValueError(f"validated share action conflicts with the governed registry: {key}")
            continue
        by_key[key] = action
        added += 1
    merged = sorted(by_key.values(), key=lambda item: (str(item["asset"]).zfill(6), str(item["price_effective_date"])))
    result = dict(payload)
    result["actions"] = merged
    return result, added


def run_promotion(spec: PromotionSpec) -> dict[str, Any]:
    validation = json.loads(spec.validation_manifest_path.read_text(encoding="utf-8"))
    validate_promotion_authority(validation, spec)
    candidates = pd.read_csv(spec.candidate_path, dtype={"asset": str})
    additions = candidate_actions(
        candidates,
        official_source_url_prefix=spec.official_source_url_prefix,
        expected_source_type=spec.expected_source_type,
    )
    before_payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    before_hash = _sha256(REGISTRY_PATH)
    archive_path = ARCHIVE_DIR / f"long_hold_v4_etf_corporate_actions_{before_hash}.json"
    if not archive_path.exists():
        _atomic_bytes(REGISTRY_PATH.read_bytes(), archive_path)
    merged, added = merge_registry_payload(before_payload, additions)
    if added:
        _atomic_json(merged, REGISTRY_PATH)
    after_hash = _sha256(REGISTRY_PATH)
    manifest = {
        "schema_version": 2,
        "source_id": spec.source_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": [
            {"role": "registry_before", "path": _relative(archive_path), "sha256": before_hash},
            {"role": "validated_candidates", "path": _relative(spec.candidate_path), "sha256": _sha256(spec.candidate_path)},
            {
                "role": "validation_authority",
                "path": _relative(spec.validation_manifest_path),
                "sha256": _sha256(spec.validation_manifest_path),
            },
        ],
        "output_path": _relative(REGISTRY_PATH),
        "output_sha256": after_hash,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "registry_rows_before": len(before_payload.get("actions", [])),
        "validated_candidate_rows": len(additions),
        "added_rows": added,
        "registry_rows_after": len(merged["actions"]),
        "qualification_status": "PROMOTED" if added else "ALREADY_PROMOTED",
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
        "method_boundary": (
            "The promoted records normalize ETF share actions only after their official disclosure. "
            "They do not qualify the underlying current-final price history or authorize a strategy."
        ),
    }
    _atomic_json(manifest, spec.promotion_manifest_path)
    return manifest


def run_selected(source: str) -> list[dict[str, Any]]:
    selected = tuple(PROMOTION_SPECS) if source == "all" else (source,)
    manifests = []
    for source_id in selected:
        manifests.append(run_promotion(PROMOTION_SPECS[source_id]))
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("all", *PROMOTION_SPECS), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = (
        "source_id",
        "qualification_status",
        "registry_rows_before",
        "validated_candidate_rows",
        "added_rows",
        "registry_rows_after",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps([{key: manifest[key] for key in keys} for manifest in run_selected(args.source)], ensure_ascii=False))


if __name__ == "__main__":
    main()
