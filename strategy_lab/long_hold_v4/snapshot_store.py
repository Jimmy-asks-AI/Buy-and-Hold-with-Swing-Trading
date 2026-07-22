"""Locked, hash-bound stock and ETF snapshot publication."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ContractError


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
COMBINED_PATH = RAW_ROOT / "research_snapshot.csv"
COMBINED_MANIFEST_PATH = RAW_ROOT / "combined_snapshot_manifest.json"
PART_PATHS = {
    "stock": RAW_ROOT / "stock_research_snapshot.csv",
    "etf": RAW_ROOT / "etf_research_snapshot.csv",
}
SNAPSHOT_SCHEMA_VERSION = 1
COMBINED_MANIFEST_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"asset": str})
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    return frame


def _write_atomic_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    _write_atomic_bytes(frame.to_csv(index=False).encode("utf-8-sig"), path)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    _write_atomic_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8"), path
    )


def _part_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


@contextmanager
def snapshot_write_lock(lock_path: Path, timeout_seconds: float = 30.0, poll_seconds: float = 0.05):
    """Acquire a shared builder lock using atomic file creation."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ContractError(f"snapshot builder lock timeout: {lock_path}")
            time.sleep(poll_seconds)
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _single_as_of(frame: pd.DataFrame, label: str) -> str:
    if "as_of_date" not in frame.columns:
        raise ContractError(f"{label} snapshot part missing as_of_date")
    values = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.normalize()
    if values.isna().any() or values.nunique() != 1:
        raise ContractError(f"{label} snapshot part must contain one valid as_of_date")
    return str(values.iloc[0].date())


def _code_entries(paths: Iterable[Path], project_root: Path = ROOT) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted({Path(value).resolve() for value in paths}):
        if not path.is_file():
            raise ContractError(f"snapshot builder code file is missing: {path}")
        try:
            relative = path.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
        entries.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return entries


def _validate_part(asset_type: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        raise ContractError(f"{asset_type} snapshot part cannot be empty")
    required = {"asset", "asset_type", "as_of_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"snapshot part missing columns: {missing}")
    if not frame["asset_type"].astype(str).str.lower().eq(asset_type).all():
        raise ContractError(f"snapshot part contains rows other than {asset_type}")
    current = frame.copy()
    current["asset"] = current["asset"].astype(str).str.zfill(6)
    if current["asset"].duplicated().any():
        raise ContractError(f"duplicate assets in {asset_type} snapshot part")
    return current, _single_as_of(current, asset_type)


def _read_part_manifest(asset_type: str, path: Path) -> dict[str, Any]:
    manifest_path = _part_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{asset_type} snapshot part manifest is missing or invalid") from exc
    try:
        snapshot_bytes = int(manifest.get("snapshot_bytes", -1))
    except (TypeError, ValueError):
        snapshot_bytes = -1
    if (
        manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("asset_type") != asset_type
        or manifest.get("snapshot_sha256") != _sha256(path)
        or snapshot_bytes != path.stat().st_size
    ):
        raise ContractError(f"{asset_type} snapshot part manifest integrity failed")
    return manifest


def write_snapshot_part(
    asset_type: str,
    frame: pd.DataFrame,
    *,
    builder_config: Mapping[str, Any] | None = None,
    builder_code_paths: Iterable[Path] | None = None,
    lock_timeout_seconds: float = 30.0,
    project_root: Path = ROOT,
) -> pd.DataFrame:
    """Publish one part and rebuild combined only from authenticated same-date parts."""
    if asset_type not in PART_PATHS:
        raise ValueError(f"unsupported snapshot asset type: {asset_type}")
    current, as_of_date = _validate_part(asset_type, frame)
    code_paths = list(builder_code_paths or [Path(__file__)])
    config_payload = dict(builder_config or {})
    lock_path = COMBINED_PATH.parent / ".snapshot_build.lock"
    with snapshot_write_lock(lock_path, timeout_seconds=lock_timeout_seconds):
        part_path = PART_PATHS[asset_type]
        _write_atomic(current, part_path)
        part_manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "asset_type": asset_type,
            "as_of_date": as_of_date,
            "snapshot_path": part_path.name,
            "snapshot_sha256": _sha256(part_path),
            "snapshot_bytes": part_path.stat().st_size,
            "builder_config_sha256": _canonical_sha256(config_payload),
            "builder_code_files": _code_entries(code_paths, project_root),
        }
        _write_json_atomic(part_manifest, _part_manifest_path(part_path))

        missing = [
            kind
            for kind, path in PART_PATHS.items()
            if not path.is_file() or not _part_manifest_path(path).is_file()
        ]
        if missing:
            return pd.DataFrame()
        manifests = {kind: _read_part_manifest(kind, path) for kind, path in PART_PATHS.items()}
        dates = {kind: str(manifest["as_of_date"]) for kind, manifest in manifests.items()}
        if len(set(dates.values())) != 1:
            raise ContractError(f"snapshot part as_of_date mismatch: {dates}")
        parts = {kind: _read(path) for kind, path in PART_PATHS.items()}
        for kind, part in parts.items():
            if _single_as_of(part, kind) != dates[kind]:
                raise ContractError(f"{kind} snapshot contents disagree with its manifest date")
        combined = pd.concat(list(parts.values()), ignore_index=True, sort=False)
        if combined["asset"].astype(str).duplicated().any():
            duplicates = sorted(
                combined.loc[combined["asset"].astype(str).duplicated(False), "asset"].astype(str).unique()
            )
            raise ContractError(f"duplicate assets across snapshot parts: {duplicates}")
        sort_columns = [column for column in ["asset_type", "sector", "asset"] if column in combined.columns]
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
        _write_atomic(combined, COMBINED_PATH)
        manifest = {
            "schema_version": COMBINED_MANIFEST_SCHEMA_VERSION,
            "as_of_date": next(iter(dates.values())),
            "parts": {
                kind: {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "builder_config_sha256": manifests[kind]["builder_config_sha256"],
                    "builder_code_files": manifests[kind]["builder_code_files"],
                }
                for kind, path in PART_PATHS.items()
            },
            "combined": {
                "path": COMBINED_PATH.relative_to(project_root).as_posix(),
                "sha256": _sha256(COMBINED_PATH),
                "bytes": COMBINED_PATH.stat().st_size,
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
            },
            "input_files": [
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in PART_PATHS.values()
            ],
            "output_files": [
                {
                    "path": COMBINED_PATH.relative_to(project_root).as_posix(),
                    "sha256": _sha256(COMBINED_PATH),
                    "bytes": COMBINED_PATH.stat().st_size,
                }
            ],
            "code_files": _code_entries(
                (
                    path
                for part_manifest in manifests.values()
                    for path in [project_root / entry["path"] for entry in part_manifest["builder_code_files"]]
                ),
                project_root,
            ),
        }
        _write_json_atomic(manifest, COMBINED_MANIFEST_PATH)
        return combined
