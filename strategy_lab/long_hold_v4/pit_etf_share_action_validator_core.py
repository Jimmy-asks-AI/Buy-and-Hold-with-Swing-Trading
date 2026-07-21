"""Shared evidence validator for official ETF share-action candidates."""

from __future__ import annotations

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
REQUIRED_COLUMNS = {
    "asset",
    "action_type",
    "event_date",
    "price_effective_date",
    "shares_after_per_share_before",
    "announcement_date",
    "source_document_title",
    "source_url",
    "source_type",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "factor_relative_error_to_inference",
    "factor_relative_error_to_observed_price_ratio",
    "normalized_price_ratio_residual",
    "action_event_date_distance_days",
    "review_status",
    "historical_backtest_allowed",
}
ALLOWED_ACTION_TYPES = {"share_split", "share_merger", "share_conversion"}
ALLOWED_REVIEW_STATUSES = {
    "official_pdf_factor_and_near_date_found_review_required",
    "official_pdf_corrects_heuristic_review_required",
}
CHECK_COLUMNS = ["asset", "price_effective_date", "check", "status", "detail"]


@dataclass(frozen=True)
class ShareActionValidationSpec:
    source_label: str
    asset_prefixes: tuple[str, ...]
    candidate_path: Path
    match_path: Path
    queue_path: Path
    collector_manifest_path: Path
    output_dir: Path
    validation_schema: str
    official_source_url_prefix: str
    expected_source_type: str
    official_document_source: str
    validator_entrypoint_path: Path
    required_collector_dependency_roles: tuple[str, ...] = ()


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
        raise ValueError(f"share-action evidence path escapes project root: {relative}")
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


def _load_candidates(path: Path, source_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{source_label} ETF registry candidates are incomplete: {missing}")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    for column in ("event_date", "price_effective_date", "announcement_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    for column in (
        "shares_after_per_share_before",
        "factor_relative_error_to_inference",
        "factor_relative_error_to_observed_price_ratio",
        "normalized_price_ratio_residual",
        "action_event_date_distance_days",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def _manifest_output_matches(manifest: dict[str, Any], path: Path) -> bool:
    relative = _relative(path)
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("path", "")).replace("\\", "/") == relative:
            return path.is_file() and str(item.get("sha256", "")) == _sha256(path)
    return False


def _date_label(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "invalid" if pd.isna(parsed) else str(pd.Timestamp(parsed).date())


def _check_row(rows: list[dict[str, str]], asset: str, date: Any, name: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "asset": asset,
            "price_effective_date": _date_label(date),
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def evaluate_candidates(
    frame: pd.DataFrame,
    *,
    official_source_url_prefix: str,
    expected_source_type: str,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    duplicate_keys = frame.duplicated(["asset", "price_effective_date"], keep=False)
    for position, row in frame.iterrows():
        asset = str(row["asset"]).zfill(6)
        effective_date = row["price_effective_date"]
        dates_valid = bool(pd.notna(row["event_date"]) and pd.notna(effective_date) and pd.notna(row["announcement_date"]))
        _check_row(checks, asset, effective_date, "dates_valid", dates_valid, "event/effective/announcement")
        _check_row(
            checks,
            asset,
            effective_date,
            "primary_key_unique",
            not bool(duplicate_keys.loc[position]),
            f"asset={asset};price_effective_date={effective_date}",
        )
        factor = row["shares_after_per_share_before"]
        factor_valid = bool(pd.notna(factor) and np.isfinite(float(factor)) and float(factor) > 0)
        _check_row(checks, asset, effective_date, "factor_positive_finite", factor_valid, f"factor={factor}")
        action_type = str(row["action_type"])
        direction_ok = bool(
            factor_valid
            and action_type in ALLOWED_ACTION_TYPES
            and (
                action_type == "share_conversion"
                or (action_type == "share_split" and float(factor) > 1)
                or (action_type == "share_merger" and float(factor) < 1)
            )
        )
        _check_row(checks, asset, effective_date, "action_direction_consistent", direction_ok, f"type={action_type};factor={factor}")
        event_distance = row["action_event_date_distance_days"]
        actual_distance = (
            int((pd.Timestamp(effective_date) - pd.Timestamp(row["event_date"])).days)
            if dates_valid
            else None
        )
        distance_consistent = bool(
            actual_distance is not None and pd.notna(event_distance) and int(event_distance) == actual_distance
        )
        _check_row(
            checks,
            asset,
            effective_date,
            "action_date_distance_consistent",
            distance_consistent,
            f"declared={event_distance};actual={actual_distance}",
        )
        event_date_ok = bool(actual_distance is not None and 0 <= actual_distance <= 5)
        _check_row(
            checks,
            asset,
            effective_date,
            "action_date_precedes_price_effective_date",
            event_date_ok,
            f"distance_days={actual_distance}",
        )
        announcement_ok = bool(dates_valid and pd.Timestamp(row["announcement_date"]) <= pd.Timestamp(effective_date))
        _check_row(
            checks,
            asset,
            effective_date,
            "announcement_not_after_price_effective_date",
            announcement_ok,
            f"announcement_date={row['announcement_date']}",
        )
        price_residual = row["normalized_price_ratio_residual"]
        price_ok = bool(pd.notna(price_residual) and np.isfinite(float(price_residual)) and float(price_residual) <= 0.10)
        _check_row(
            checks,
            asset,
            effective_date,
            "independent_price_ratio_crosscheck",
            price_ok,
            f"abs(observed_ratio/official_factor-1)={price_residual}",
        )
        source_url = str(row["source_url"])
        _check_row(
            checks,
            asset,
            effective_date,
            "official_source_url",
            source_url.startswith(official_source_url_prefix),
            source_url,
        )
        source_type = str(row["source_type"])
        _check_row(
            checks,
            asset,
            effective_date,
            "official_source_type",
            source_type == expected_source_type,
            source_type,
        )
        review_status = str(row["review_status"])
        _check_row(
            checks,
            asset,
            effective_date,
            "collector_review_status_allowed",
            review_status in ALLOWED_REVIEW_STATUSES,
            review_status,
        )
        historical_disabled = row["historical_backtest_allowed"] is False or str(row["historical_backtest_allowed"]).lower() == "false"
        _check_row(
            checks,
            asset,
            effective_date,
            "candidate_historical_use_stays_disabled",
            historical_disabled,
            f"historical_backtest_allowed={row['historical_backtest_allowed']!r}",
        )
        for role in ("pdf", "text"):
            relative = str(row[f"{role}_path"])
            expected_hash = str(row[f"{role}_sha256"])
            try:
                evidence_path = _safe_path(relative)
                hash_ok = bool(evidence_path.is_file() and expected_hash and _sha256(evidence_path) == expected_hash)
            except (OSError, ValueError):
                hash_ok = False
            _check_row(checks, asset, effective_date, f"{role}_hash_match", hash_ok, relative)
    return pd.DataFrame(checks, columns=CHECK_COLUMNS)


def _authenticate_code_entry(entry: dict[str, Any], label: str) -> Path:
    relative = str(entry.get("path", ""))
    expected_hash = str(entry.get("sha256", ""))
    path = _safe_path(relative) if relative else None
    if not path or not path.is_file() or not expected_hash or _sha256(path) != expected_hash:
        raise ValueError(f"{label} code hash does not match its manifest")
    return path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        unique[_relative(path)] = path
    return [unique[key] for key in sorted(unique)]


def run_validation(spec: ShareActionValidationSpec) -> dict[str, Any]:
    collector_manifest = json.loads(spec.collector_manifest_path.read_text(encoding="utf-8"))
    collector_code = _authenticate_code_entry(
        {"path": collector_manifest.get("code_path"), "sha256": collector_manifest.get("code_sha256")},
        "collector",
    )
    collector_dependencies: list[Path] = []
    dependency_roles: set[str] = set()
    for entry in collector_manifest.get("code_dependencies", []):
        if not isinstance(entry, dict):
            raise ValueError("collector code dependency entry is invalid")
        role = str(entry.get("role", ""))
        dependency_roles.add(role)
        collector_dependencies.append(_authenticate_code_entry(entry, f"collector dependency {role or 'unknown'}"))
    missing_dependency_roles = sorted(set(spec.required_collector_dependency_roles).difference(dependency_roles))
    if missing_dependency_roles:
        raise ValueError(f"collector manifest misses required code dependencies: {missing_dependency_roles}")
    if not _manifest_output_matches(collector_manifest, spec.candidate_path):
        raise ValueError("collector manifest does not authenticate the registry candidate output")
    if not _manifest_output_matches(collector_manifest, spec.match_path):
        raise ValueError("collector manifest does not authenticate the queue match output")

    candidates = _load_candidates(spec.candidate_path, spec.source_label)
    queue = pd.read_csv(spec.queue_path, dtype={"asset": str})
    queue["asset"] = queue["asset"].astype(str).str.zfill(6)
    queue["price_effective_date"] = pd.to_datetime(queue["price_effective_date"], errors="coerce").dt.normalize()
    queue = queue[queue["asset"].str.startswith(spec.asset_prefixes)]
    candidate_keys = set(zip(candidates["asset"], candidates["price_effective_date"], strict=True))
    queue_keys = set(zip(queue["asset"], queue["price_effective_date"], strict=True))

    checks = evaluate_candidates(
        candidates,
        official_source_url_prefix=spec.official_source_url_prefix,
        expected_source_type=spec.expected_source_type,
    )
    global_rows: list[dict[str, str]] = []
    global_date = candidates["price_effective_date"].max()
    _check_row(
        global_rows,
        "ALL",
        global_date,
        f"candidate_keys_match_{spec.source_label.lower()}_queue",
        candidate_keys == queue_keys,
        f"candidate_keys={len(candidate_keys)};queue_keys={len(queue_keys)}",
    )
    _check_row(
        global_rows,
        "ALL",
        global_date,
        "collector_historical_use_stays_disabled",
        collector_manifest.get("historical_backtest_allowed") is False,
        f"historical_backtest_allowed={collector_manifest.get('historical_backtest_allowed')!r}",
    )
    checks = pd.concat([checks, pd.DataFrame(global_rows, columns=CHECK_COLUMNS)], ignore_index=True)
    exceptions = checks[checks["status"].eq("fail")].copy()
    qualification = "PASS" if exceptions.empty else "FAIL"

    checks_path = spec.output_dir / "candidate_checks.csv"
    exceptions_path = spec.output_dir / "exceptions.csv"
    report_path = spec.output_dir / "report.json"
    manifest_path = spec.output_dir / "run_manifest.json"
    _atomic_csv(checks, checks_path)
    _atomic_csv(exceptions, exceptions_path)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "candidate_rows": int(len(candidates)),
        "candidate_assets": int(candidates["asset"].nunique()),
        "check_rows": int(len(checks)),
        "failed_check_rows": int(len(exceptions)),
        "official_source_rows": int(candidates["source_url"].astype(str).str.startswith(spec.official_source_url_prefix).sum()),
        "heuristic_correction_rows": int(candidates["review_status"].eq("official_pdf_corrects_heuristic_review_required").sum()),
        "maximum_normalized_price_ratio_residual": float(candidates["normalized_price_ratio_residual"].max()),
        "maximum_action_to_price_effective_days": int(candidates["action_event_date_distance_days"].max()),
        "boundary": "share-action registry evidence only; no strategy, performance, or investment conclusion",
    }
    _atomic_json(report, report_path)

    input_paths = _dedupe_paths(
        [
            spec.candidate_path,
            spec.match_path,
            spec.queue_path,
            spec.collector_manifest_path,
            collector_code,
            *collector_dependencies,
        ]
    )
    core_path = Path(__file__).resolve()
    entrypoint = spec.validator_entrypoint_path.resolve()
    manifest = {
        "validation_schema": spec.validation_schema,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": qualification,
        "inputs": [{"path": _relative(path), "sha256": _sha256(path)} for path in input_paths],
        "outputs": [
            {"role": "checks", "path": _relative(checks_path), "sha256": _sha256(checks_path)},
            {"role": "exceptions", "path": _relative(exceptions_path), "sha256": _sha256(exceptions_path)},
            {"role": "report", "path": _relative(report_path), "sha256": _sha256(report_path)},
        ],
        "exceptions_path": _relative(exceptions_path),
        "exceptions_sha256": _sha256(exceptions_path),
        "code_path": _relative(entrypoint),
        "code_sha256": _sha256(entrypoint),
        "code_dependencies": [
            {"role": "shared_validator_core", "path": _relative(core_path), "sha256": _sha256(core_path)}
        ],
        "candidate_rows": int(len(candidates)),
        "failed_check_rows": int(len(exceptions)),
        "independent_source_count": 2,
        "official_document_source": spec.official_document_source,
        "independent_crosscheck_source": "price-ratio anomalies from the ETF lifecycle observation chain",
        "price_crosscheck_formula": "abs(observed_price_ratio / official_share_factor - 1) <= 0.10",
        "registry_promotion_allowed": qualification == "PASS",
        "historical_backtest_allowed": qualification == "PASS",
        "model_promotion_allowed": False,
        "method_boundary": (
            "Approval is limited to corporate-action normalization after each event becomes public. "
            "It does not qualify the current-final price history or authorize model promotion."
        ),
    }
    _atomic_json(manifest, manifest_path)
    return manifest
