"""Versioned PIT gate and immutable dataset-revision store.

The gate qualifies inputs for a formal walk-forward run. It never promotes a
model and it does not run a backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ContractError


GATE_SCHEMA_VERSION = 1
DATASET_MANIFEST_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
REQUIRED_DATASET_METADATA = (
    "dataset_id",
    "provider",
    "license",
    "coverage_start",
    "coverage_end",
    "observed_at",
    "available_date",
    "revision_id",
    "symbol",
    "frequency",
    "schema_version",
    "file_sha256",
    "manifest_sha256",
    "pit_status",
    "known_limitations",
)
DECISION_COLUMNS = ("dataset_id", "status", "reason_code", "detail")


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if not str(relative).strip():
        raise ContractError("empty project-relative path")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ContractError(f"path escapes project root: {relative}")
    return path


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"path is outside project root: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON object required: {path}")
    return payload


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ContractError("project root is not at a verifiable Git commit")
    return value


def _parse_date(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ContractError(f"invalid {field}: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _parse_timestamp(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ContractError(f"invalid {field}: {value!r}")
    return pd.Timestamp(parsed)


def _validate_revision_identifiers(dataset_id: str, revision_id: str) -> None:
    if not SAFE_ID_RE.fullmatch(dataset_id):
        raise ContractError(f"unsafe dataset_id: {dataset_id!r}")
    if not SAFE_ID_RE.fullmatch(revision_id):
        raise ContractError(f"unsafe revision_id: {revision_id!r}")


def register_dataset_revision(
    project_root: str | Path,
    version_store_root: str | Path,
    source_file: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Copy one dataset revision into a write-once version directory.

    Reusing an existing ``dataset_id/revision_id`` is always rejected, even if
    the bytes are unchanged. A later revision must use a new revision id and,
    once a history exists, bind the previous manifest hash.
    """

    root = Path(project_root).resolve()
    store = _safe_path(root, str(version_store_root))
    source = Path(source_file).resolve()
    if not source.is_file():
        raise ContractError(f"dataset source file is missing: {source}")

    required = set(REQUIRED_DATASET_METADATA).difference(
        {"file_sha256", "manifest_sha256"}
    )
    missing = sorted(
        field for field in required if not str(metadata.get(field, "")).strip()
    )
    if missing:
        raise ContractError(f"dataset revision metadata missing fields: {missing}")

    dataset_id = str(metadata["dataset_id"]).strip()
    revision_id = str(metadata["revision_id"]).strip()
    _validate_revision_identifiers(dataset_id, revision_id)
    version_parent = store / dataset_id
    version_dir = version_parent / revision_id
    if version_dir.exists():
        raise ContractError(
            f"dataset revision overwrite prohibited: {dataset_id}/{revision_id}"
        )

    existing_revisions = (
        sorted(path for path in version_parent.iterdir() if path.is_dir())
        if version_parent.is_dir()
        else []
    )
    previous_revision_id = str(metadata.get("previous_revision_id", "")).strip()
    previous_manifest_sha256 = str(
        metadata.get("previous_manifest_sha256", "")
    ).strip()
    if existing_revisions:
        if not previous_revision_id or not SHA256_RE.fullmatch(
            previous_manifest_sha256
        ):
            raise ContractError(
                "new dataset revision must bind the previous revision manifest"
            )
        _validate_revision_identifiers(dataset_id, previous_revision_id)
        previous_manifest = (
            version_parent / previous_revision_id / "dataset_manifest.json"
        )
        if (
            not previous_manifest.is_file()
            or sha256_file(previous_manifest) != previous_manifest_sha256
        ):
            raise ContractError("previous dataset revision manifest is not preserved")
    elif previous_revision_id or previous_manifest_sha256:
        raise ContractError("first dataset revision cannot declare a previous revision")

    version_parent.mkdir(parents=True, exist_ok=True)
    temporary = version_parent / f".{revision_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    suffixes = "".join(source.suffixes) or ".bin"
    final_data_path = version_dir / f"dataset{suffixes}"
    temporary_data_path = temporary / final_data_path.name
    shutil.copyfile(source, temporary_data_path)
    file_hash = sha256_file(temporary_data_path)

    manifest_payload = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        **{
            field: metadata[field]
            for field in REQUIRED_DATASET_METADATA
            if field not in {"file_sha256", "manifest_sha256"}
        },
        "file_path": _relative_path(root, final_data_path),
        "file_sha256": file_hash,
        "current_snapshot": bool(metadata.get("current_snapshot", False)),
        "contains_current_snapshot_backfill": bool(
            metadata.get("contains_current_snapshot_backfill", False)
        ),
        "contains_current_constituents_backfill": bool(
            metadata.get("contains_current_constituents_backfill", False)
        ),
        "previous_revision_id": previous_revision_id or None,
        "previous_manifest_sha256": previous_manifest_sha256 or None,
        "promotion_allowed": False,
    }
    temporary_manifest = temporary / "dataset_manifest.json"
    temporary_manifest.write_bytes(canonical_json_bytes(manifest_payload))
    temporary.rename(version_dir)

    final_manifest = version_dir / "dataset_manifest.json"
    return {
        **{
            field: manifest_payload[field]
            for field in REQUIRED_DATASET_METADATA
            if field not in {"manifest_sha256"}
        },
        "manifest_path": _relative_path(root, final_manifest),
        "manifest_sha256": sha256_file(final_manifest),
        "file_path": manifest_payload["file_path"],
        "current_snapshot": manifest_payload["current_snapshot"],
        "contains_current_snapshot_backfill": manifest_payload[
            "contains_current_snapshot_backfill"
        ],
        "contains_current_constituents_backfill": manifest_payload[
            "contains_current_constituents_backfill"
        ],
    }


def _decision(
    rows: list[dict[str, str]],
    dataset_id: str,
    reason_code: str,
    passed: bool,
    detail: str,
) -> None:
    rows.append(
        {
            "dataset_id": dataset_id,
            "status": "PASS" if passed else "FAIL",
            "reason_code": reason_code,
            "detail": detail,
        }
    )


def _load_gate_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    required = {
        "schema_version",
        "output_root",
        "version_store_root",
        "required_dataset_ids",
        "allowed_license_values",
        "maximum_manifest_age_days",
        "gate_code_paths",
        "promotion_allowed",
        "independent_review_required",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"PIT gate v2 config missing fields: {missing}")
    if config["schema_version"] != GATE_SCHEMA_VERSION:
        raise ContractError(
            f"PIT gate v2 requires schema_version={GATE_SCHEMA_VERSION}"
        )
    dataset_ids = [str(value) for value in config["required_dataset_ids"]]
    if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
        raise ContractError("required_dataset_ids must be non-empty and unique")
    if config["promotion_allowed"] is not False:
        raise ContractError("PIT gate config must keep promotion_allowed=false")
    if config["independent_review_required"] is not True:
        raise ContractError("PIT gate config must require independent review")
    if int(config["maximum_manifest_age_days"]) < 0:
        raise ContractError("maximum_manifest_age_days must be non-negative")
    if not config["allowed_license_values"]:
        raise ContractError("allowed_license_values must be non-empty")
    if not config["gate_code_paths"]:
        raise ContractError("gate_code_paths must be non-empty")
    return config


def _validate_target_generation(
    root: Path,
    target_manifest: dict[str, Any],
    current_commit: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    generation = target_manifest.get("target_generation")
    if not isinstance(generation, dict):
        _decision(
            rows,
            "__target__",
            "target_generation_present",
            False,
            "target_generation object is missing",
        )
        return {}

    stated_commit = str(generation.get("code_commit", "")).strip().lower()
    _decision(
        rows,
        "__target__",
        "target_code_commit_match",
        stated_commit == current_commit,
        f"stated={stated_commit};current={current_commit}",
    )

    code_files = generation.get("code_files")
    code_files = code_files if isinstance(code_files, list) else []
    _decision(
        rows,
        "__target__",
        "target_code_files_present",
        bool(code_files),
        f"count={len(code_files)}",
    )
    for item in code_files:
        relative = str(item.get("path", "")) if isinstance(item, dict) else ""
        expected = str(item.get("sha256", "")) if isinstance(item, dict) else ""
        try:
            path = _safe_path(root, relative)
        except ContractError as exc:
            _decision(
                rows,
                "__target__",
                "target_code_hash_match",
                False,
                str(exc),
            )
            continue
        passed = (
            path.is_file()
            and SHA256_RE.fullmatch(expected) is not None
            and sha256_file(path) == expected
        )
        _decision(
            rows,
            "__target__",
            "target_code_hash_match",
            passed,
            relative,
        )

    config_binding = generation.get("config")
    config_binding = config_binding if isinstance(config_binding, dict) else {}
    relative = str(config_binding.get("path", ""))
    expected = str(config_binding.get("sha256", ""))
    try:
        path = _safe_path(root, relative)
    except ContractError as exc:
        _decision(
            rows,
            "__target__",
            "target_config_hash_match",
            False,
            str(exc),
        )
    else:
        passed = (
            path.is_file()
            and SHA256_RE.fullmatch(expected) is not None
            and sha256_file(path) == expected
        )
        _decision(
            rows,
            "__target__",
            "target_config_hash_match",
            passed,
            relative,
        )
    return generation


def _audit_dataset_available_dates(
    path: Path, decision_date: pd.Timestamp
) -> tuple[int, int, int]:
    name = path.name.lower()
    if not (name.endswith(".csv") or name.endswith(".csv.gz")):
        raise ContractError(f"unsupported PIT dataset format: {path.name}")
    try:
        chunks = pd.read_csv(
            path, usecols=["available_date"], chunksize=250_000
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot read available_date from {path}: {exc}") from exc
    invalid_count = 0
    row_count = 0
    future_count = 0
    try:
        for chunk in chunks:
            parsed = pd.to_datetime(
                chunk["available_date"], errors="coerce"
            ).dt.normalize()
            invalid_count += int(parsed.isna().sum())
            row_count += len(parsed)
            future_count += int((parsed > decision_date).sum())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot stream available_date from {path}: {exc}") from exc
    return invalid_count, row_count, future_count


def _validate_dataset_entry(
    root: Path,
    config: dict[str, Any],
    entry: dict[str, Any],
    decision_date: pd.Timestamp,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    gate_observed_at: pd.Timestamp,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    dataset_id = str(entry.get("dataset_id", "")).strip() or "__missing_id__"
    missing = [
        field
        for field in REQUIRED_DATASET_METADATA
        if not str(entry.get(field, "")).strip()
    ]
    _decision(
        rows,
        dataset_id,
        "required_metadata_complete",
        not missing,
        f"missing={missing}",
    )

    license_value = str(entry.get("license", "")).strip()
    allowed_licenses = set(str(value) for value in config["allowed_license_values"])
    _decision(
        rows,
        dataset_id,
        "license_approved",
        license_value in allowed_licenses,
        license_value,
    )
    pit_status = str(entry.get("pit_status", "")).strip()
    _decision(
        rows,
        dataset_id,
        "pit_status_qualified",
        pit_status == "QUALIFIED_PIT",
        pit_status,
    )

    snapshot_flags = {
        "current_snapshot": bool(entry.get("current_snapshot", False)),
        "contains_current_snapshot_backfill": bool(
            entry.get("contains_current_snapshot_backfill", False)
        ),
    }
    _decision(
        rows,
        dataset_id,
        "no_current_snapshot_backfill",
        not any(snapshot_flags.values()),
        json.dumps(snapshot_flags, sort_keys=True),
    )
    current_constituents = bool(
        entry.get("contains_current_constituents_backfill", False)
    )
    _decision(
        rows,
        dataset_id,
        "no_current_constituents_backfill",
        not current_constituents,
        f"contains_current_constituents_backfill={current_constituents}",
    )

    try:
        coverage_start = _parse_date(entry.get("coverage_start"), "coverage_start")
        coverage_end = _parse_date(entry.get("coverage_end"), "coverage_end")
    except ContractError as exc:
        _decision(rows, dataset_id, "coverage_in_bounds", False, str(exc))
    else:
        coverage_ok = (
            coverage_start <= required_start
            and coverage_end >= required_end
            and coverage_start <= coverage_end
        )
        _decision(
            rows,
            dataset_id,
            "coverage_in_bounds",
            coverage_ok,
            (
                f"actual={coverage_start.date()}..{coverage_end.date()};"
                f"required={required_start.date()}..{required_end.date()}"
            ),
        )

    try:
        metadata_available = _parse_date(
            entry.get("available_date"), "available_date"
        )
    except ContractError as exc:
        _decision(rows, dataset_id, "available_date_not_future", False, str(exc))
    else:
        _decision(
            rows,
            dataset_id,
            "available_date_not_future",
            metadata_available <= decision_date,
            (
                f"available_date={metadata_available.date()};"
                f"decision_date={decision_date.date()}"
            ),
        )

    try:
        observed_at = _parse_timestamp(entry.get("observed_at"), "observed_at")
    except ContractError as exc:
        _decision(rows, dataset_id, "manifest_fresh", False, str(exc))
    else:
        age_days = (gate_observed_at - observed_at).total_seconds() / 86400.0
        fresh = (
            0 <= age_days <= int(config["maximum_manifest_age_days"])
        )
        _decision(
            rows,
            dataset_id,
            "manifest_fresh",
            fresh,
            f"age_days={age_days:.6f}",
        )

    revision_id = str(entry.get("revision_id", "")).strip()
    try:
        _validate_revision_identifiers(dataset_id, revision_id)
    except ContractError as exc:
        _decision(rows, dataset_id, "revision_id_safe", False, str(exc))
    else:
        _decision(rows, dataset_id, "revision_id_safe", True, revision_id)

    file_relative = str(entry.get("file_path", ""))
    file_hash = str(entry.get("file_sha256", "")).lower()
    manifest_relative = str(entry.get("manifest_path", ""))
    manifest_hash = str(entry.get("manifest_sha256", "")).lower()
    try:
        file_path = _safe_path(root, file_relative)
    except ContractError as exc:
        file_path = None
        _decision(rows, dataset_id, "file_hash_match", False, str(exc))
    else:
        file_ok = (
            file_path.is_file()
            and SHA256_RE.fullmatch(file_hash) is not None
            and sha256_file(file_path) == file_hash
        )
        _decision(rows, dataset_id, "file_hash_match", file_ok, file_relative)

    try:
        manifest_path = _safe_path(root, manifest_relative)
    except ContractError as exc:
        manifest_path = None
        _decision(rows, dataset_id, "dataset_manifest_hash_match", False, str(exc))
    else:
        manifest_ok = (
            manifest_path.is_file()
            and SHA256_RE.fullmatch(manifest_hash) is not None
            and sha256_file(manifest_path) == manifest_hash
        )
        _decision(
            rows,
            dataset_id,
            "dataset_manifest_hash_match",
            manifest_ok,
            manifest_relative,
        )

    expected_version_root = None
    try:
        version_store = _safe_path(root, str(config["version_store_root"]))
        expected_version_root = (version_store / dataset_id / revision_id).resolve()
    except ContractError:
        pass
    layout_ok = bool(
        expected_version_root
        and file_path
        and manifest_path
        and expected_version_root in file_path.resolve().parents
        and manifest_path.resolve()
        == expected_version_root / "dataset_manifest.json"
    )
    _decision(
        rows,
        dataset_id,
        "immutable_revision_layout",
        layout_ok,
        f"expected={expected_version_root}",
    )

    manifest_payload: dict[str, Any] = {}
    if manifest_path and manifest_path.is_file():
        try:
            manifest_payload = _read_json(manifest_path)
        except ContractError as exc:
            _decision(rows, dataset_id, "dataset_manifest_readable", False, str(exc))
        else:
            _decision(
                rows,
                dataset_id,
                "dataset_manifest_readable",
                True,
                manifest_relative,
            )
            comparisons = (
                set(REQUIRED_DATASET_METADATA)
                .difference({"manifest_sha256"})
                .union(
                    {
                        "file_path",
                        "current_snapshot",
                        "contains_current_snapshot_backfill",
                        "contains_current_constituents_backfill",
                    }
                )
            )
            mismatch = sorted(
                field
                for field in comparisons
                if manifest_payload.get(field) != entry.get(field)
            )
            schema_ok = (
                manifest_payload.get("schema_version")
                == DATASET_MANIFEST_SCHEMA_VERSION
            )
            _decision(
                rows,
                dataset_id,
                "dataset_manifest_metadata_match",
                not mismatch and schema_ok,
                f"mismatch={mismatch};schema_ok={schema_ok}",
            )

            previous_revision_id = manifest_payload.get("previous_revision_id")
            previous_hash = manifest_payload.get("previous_manifest_sha256")
            if previous_revision_id:
                previous_path = (
                    expected_version_root.parent
                    / str(previous_revision_id)
                    / "dataset_manifest.json"
                    if expected_version_root
                    else Path("")
                )
                previous_ok = bool(
                    previous_path.is_file()
                    and SHA256_RE.fullmatch(str(previous_hash or "")) is not None
                    and sha256_file(previous_path) == previous_hash
                )
            else:
                previous_ok = previous_hash in {None, ""}
            _decision(
                rows,
                dataset_id,
                "previous_revision_preserved",
                previous_ok,
                f"previous_revision_id={previous_revision_id!r}",
            )

    if file_path and file_path.is_file():
        try:
            bad_dates, row_count, future_count = _audit_dataset_available_dates(
                file_path, decision_date
            )
        except ContractError as exc:
            _decision(
                rows,
                dataset_id,
                "row_available_dates_valid",
                False,
                str(exc),
            )
            _decision(
                rows,
                dataset_id,
                "row_available_dates_not_future",
                False,
                str(exc),
            )
        else:
            _decision(
                rows,
                dataset_id,
                "row_available_dates_valid",
                row_count > 0 and bad_dates == 0,
                f"rows={row_count};invalid={bad_dates}",
            )
            _decision(
                rows,
                dataset_id,
                "row_available_dates_not_future",
                future_count == 0,
                f"future_rows={future_count}",
            )

    return {
        "dataset_id": dataset_id,
        "revision_id": revision_id,
        "file_path": file_relative,
        "file_sha256": file_hash,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_hash,
    }


def run_versioned_pit_gate(
    project_root: str | Path,
    config_path: str | Path,
    target_manifest_path: str | Path,
    *,
    run_id: str,
    gate_observed_at: str,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    if not SAFE_ID_RE.fullmatch(run_id):
        raise ContractError(f"unsafe PIT Gate run_id: {run_id!r}")
    config_file = _safe_path(root, str(config_path))
    target_file = _safe_path(root, str(target_manifest_path))
    config = _load_gate_config(config_file)
    target_manifest = _read_json(target_file)
    if target_manifest.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ContractError(
            f"target manifest requires schema_version={GATE_SCHEMA_VERSION}"
        )

    decision_date = _parse_date(
        target_manifest.get("decision_date"), "decision_date"
    )
    required_start = _parse_date(
        target_manifest.get("required_coverage_start"),
        "required_coverage_start",
    )
    required_end = _parse_date(
        target_manifest.get("required_coverage_end"), "required_coverage_end"
    )
    if required_end > decision_date or required_start > required_end:
        raise ContractError("invalid target coverage interval")
    observed_at = _parse_timestamp(gate_observed_at, "gate_observed_at")
    current_commit = _git_head(root)

    rows: list[dict[str, str]] = []
    target_generation = _validate_target_generation(
        root, target_manifest, current_commit, rows
    )
    datasets = target_manifest.get("datasets")
    datasets = datasets if isinstance(datasets, list) else []
    dataset_map: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        dataset_id = str(item.get("dataset_id", ""))
        if dataset_id in dataset_map:
            duplicates.append(dataset_id)
        dataset_map[dataset_id] = item
    _decision(
        rows,
        "__target__",
        "dataset_ids_unique",
        not duplicates,
        f"duplicates={sorted(set(duplicates))}",
    )

    bindings: list[dict[str, Any]] = []
    for dataset_id in config["required_dataset_ids"]:
        entry = dataset_map.get(str(dataset_id))
        if entry is None:
            _decision(
                rows,
                str(dataset_id),
                "required_dataset_present",
                False,
                "dataset is absent from target manifest",
            )
            continue
        _decision(
            rows,
            str(dataset_id),
            "required_dataset_present",
            True,
            "present",
        )
        bindings.append(
            _validate_dataset_entry(
                root,
                config,
                entry,
                decision_date,
                required_start,
                required_end,
                observed_at,
                rows,
            )
        )

    extra_ids = sorted(
        set(dataset_map).difference(str(value) for value in config["required_dataset_ids"])
    )
    _decision(
        rows,
        "__target__",
        "no_unreviewed_extra_datasets",
        not extra_ids,
        f"extra={extra_ids}",
    )

    gate_code_bindings: list[dict[str, str]] = []
    for relative in config["gate_code_paths"]:
        code_path = _safe_path(root, str(relative))
        if not code_path.is_file():
            raise ContractError(f"gate code path is missing: {relative}")
        gate_code_bindings.append(
            {"path": str(relative), "sha256": sha256_file(code_path)}
        )

    failed = [row for row in rows if row["status"] == "FAIL"]
    gate_passed = not failed
    status = "PASS_PIT_GATE" if gate_passed else "BLOCKED_PIT_GATE"
    summary = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_id": run_id,
        "decision_date": decision_date.date().isoformat(),
        "gate_observed_at": observed_at.isoformat(),
        "status": status,
        "formal_backtest_allowed": gate_passed,
        "promotion_allowed": False,
        "independent_review_required": True,
        "manual_review_signed": False,
        "required_dataset_count": len(config["required_dataset_ids"]),
        "bound_dataset_count": len(bindings),
        "failed_check_count": len(failed),
        "failure_reasons": sorted(
            {
                f"{row['dataset_id']}:{row['reason_code']}"
                for row in failed
            }
        ),
    }

    output_root = _safe_path(root, str(config["output_root"]))
    final_output = output_root / run_id
    temporary_output = output_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    if final_output.exists():
        raise ContractError(f"PIT Gate run_id is immutable and already exists: {run_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_output.mkdir()
    details_path = temporary_output / "decision_details.csv"
    summary_path = temporary_output / "decision.json"
    pd.DataFrame(rows, columns=DECISION_COLUMNS).to_csv(
        details_path, index=False, encoding="utf-8"
    )
    summary_path.write_bytes(canonical_json_bytes(summary))

    gate_manifest = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "decision_date": decision_date.date().isoformat(),
        "gate_observed_at": observed_at.isoformat(),
        "target_manifest": {
            "path": _relative_path(root, target_file),
            "sha256": sha256_file(target_file),
        },
        "input_datasets": bindings,
        "target_generation": target_generation,
        "gate_generation": {
            "code_commit": current_commit,
            "code_files": gate_code_bindings,
            "config": {
                "path": _relative_path(root, config_file),
                "sha256": sha256_file(config_file),
            },
        },
        "outputs": [
            {
                "path": "decision_details.csv",
                "sha256": sha256_file(details_path),
            },
            {"path": "decision.json", "sha256": sha256_file(summary_path)},
        ],
        "formal_backtest_allowed": gate_passed,
        "promotion_allowed": False,
        "independent_review_required": True,
        "manual_review_signed": False,
    }
    manifest_path = temporary_output / "pit_gate_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(gate_manifest))
    manifest_hash = sha256_file(manifest_path)
    seal_path = temporary_output / "pit_gate_manifest_seal.json"
    seal_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": GATE_SCHEMA_VERSION,
                "run_id": run_id,
                "pit_gate_manifest_sha256": manifest_hash,
                "promotion_allowed": False,
            }
        )
    )
    temporary_output.rename(final_output)
    return {
        "run_directory": final_output,
        "decision_details": final_output / details_path.name,
        "decision": final_output / summary_path.name,
        "manifest": final_output / manifest_path.name,
        "seal": final_output / seal_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the versioned Long Hold V4 PIT Gate"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--gate-observed-at",
        default=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    args = parser.parse_args()
    paths = run_versioned_pit_gate(
        args.project_root,
        args.config,
        args.target_manifest,
        run_id=args.run_id,
        gate_observed_at=args.gate_observed_at,
    )
    decision = _read_json(paths["decision"])
    print(
        json.dumps(
            {
                "status": decision["status"],
                "promotion_allowed": False,
                "run_directory": str(paths["run_directory"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
