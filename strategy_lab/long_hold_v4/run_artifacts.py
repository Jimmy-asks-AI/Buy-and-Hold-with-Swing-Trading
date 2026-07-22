"""Immutable run publication and verification for Long Hold V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .core import ContractError


RUN_MANIFEST_SCHEMA_VERSION = 2
RUN_SEAL_SCHEMA_VERSION = 1
CURRENT_POINTER_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configured_output_root(project_root: Path, configured_path: str | Path) -> Path:
    configured = project_root / configured_path
    return configured.parent if configured.name == "current" else configured


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative_path(value: str) -> PurePosixPath:
    candidate = PurePosixPath(str(value).replace("\\", "/"))
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ContractError(f"unsafe run artifact path: {value}")
    return candidate


def _inside(root: Path, relative: PurePosixPath) -> Path:
    root_resolved = root.resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    if candidate == root_resolved or root_resolved not in candidate.parents:
        raise ContractError(f"run artifact path escapes its root: {relative}")
    return candidate


class RunArtifactPublisher:
    """Stage a run, seal every artifact, then publish it with one directory rename."""

    def __init__(
        self,
        output_root: Path,
        run_id: str,
        *,
        failure_hook: Callable[[str], None] | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", run_id):
            raise ContractError("run_id contains unsafe characters")
        self.output_root = output_root
        self.runs_root = output_root / "runs"
        self.run_id = run_id
        self.stage = self.runs_root / f"{run_id}.tmp"
        self.final = self.runs_root / run_id
        self.failure_hook = failure_hook
        self._schemas: dict[str, int] = {}
        self._started = False

    def _checkpoint(self, name: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(name)

    def begin(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        current_path = self.output_root / "current"
        if current_path.is_dir():
            raise ContractError("legacy current directory requires explicit migration before immutable publication")
        if current_path.exists():
            resolve_current_run(self.output_root)
        if self.final.exists():
            raise ContractError(f"immutable run already exists: {self.final}")
        if self.stage.exists():
            raise ContractError(f"stale temporary run requires explicit recovery: {self.stage}")
        try:
            self.stage.mkdir()
        except FileExistsError as exc:
            raise ContractError(f"concurrent publisher already owns temporary run: {self.stage}") from exc
        self._started = True
        self._checkpoint("stage_created")

    def write_bytes(self, relative_path: str, data: bytes, *, schema_version: int) -> Path:
        if not self._started:
            raise ContractError("run publisher has not been started")
        relative = _safe_relative_path(relative_path)
        path = _inside(self.stage, relative)
        _atomic_write(path, data)
        self._schemas[relative.as_posix()] = int(schema_version)
        self._checkpoint(f"artifact_written:{relative.as_posix()}")
        return path

    def write_json(self, relative_path: str, payload: Any, *, schema_version: int) -> Path:
        return self.write_bytes(relative_path, _json_bytes(payload), schema_version=schema_version)

    def write_csv(self, relative_path: str, frame: Any, *, schema_version: int) -> Path:
        data = frame.to_csv(index=False).encode("utf-8-sig")
        return self.write_bytes(relative_path, data, schema_version=schema_version)

    def _output_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for relative_path, schema_version in sorted(self._schemas.items()):
            path = _inside(self.stage, _safe_relative_path(relative_path))
            entries.append(
                {
                    "path": relative_path,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "schema_version": schema_version,
                }
            )
        return entries

    def finalize(self, manifest_context: Mapping[str, Any]) -> dict[str, Path]:
        if not self._started:
            raise ContractError("run publisher has not been started")
        context = dict(manifest_context)
        if context.get("run_id") != self.run_id:
            raise ContractError("run manifest context does not match publisher run_id")
        binding_sha256 = canonical_sha256(context)
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            **context,
            "order_binding": {
                "algorithm": "sha256-canonical-json",
                "scope": "manifest_context_before_outputs",
                "sha256": binding_sha256,
            },
            "outputs": self._output_entries(),
        }
        manifest_path = self.stage / "run_manifest.json"
        _atomic_write(manifest_path, _json_bytes(manifest))
        manifest_sha256 = sha256_file(manifest_path)
        seal = {
            "schema_version": RUN_SEAL_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_manifest_path": "run_manifest.json",
            "run_manifest_sha256": manifest_sha256,
        }
        _atomic_write(self.stage / "run_manifest_seal.json", _json_bytes(seal))
        self._checkpoint("sealed")
        os.replace(self.stage, self.final)
        self._checkpoint("run_published")
        pointer = {
            "schema_version": CURRENT_POINTER_SCHEMA_VERSION,
            "run_id": self.run_id,
            "relative_path": f"runs/{self.run_id}",
            "run_manifest_sha256": manifest_sha256,
        }
        _atomic_write(self.output_root / "current", _json_bytes(pointer))
        self._checkpoint("current_updated")
        self._started = False
        return {
            "run": self.final,
            "manifest": self.final / "run_manifest.json",
            "seal": self.final / "run_manifest_seal.json",
            "current": self.output_root / "current",
        }


def read_current_pointer(output_root: Path) -> dict[str, Any]:
    pointer_path = output_root / "current"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"current run pointer is unreadable: {pointer_path}") from exc
    required = {"schema_version", "run_id", "relative_path", "run_manifest_sha256"}
    if set(pointer) != required or pointer["schema_version"] != CURRENT_POINTER_SCHEMA_VERSION:
        raise ContractError("current run pointer schema is invalid")
    if not SHA256_RE.fullmatch(str(pointer["run_manifest_sha256"])):
        raise ContractError("current run pointer manifest hash is invalid")
    relative = _safe_relative_path(str(pointer["relative_path"]))
    expected = PurePosixPath("runs") / str(pointer["run_id"])
    if relative != expected or str(pointer["run_id"]).endswith(".tmp"):
        raise ContractError("current run pointer target is invalid")
    return pointer


def resolve_current_run(output_root: Path) -> Path:
    pointer = read_current_pointer(output_root)
    run_path = _inside(output_root, _safe_relative_path(pointer["relative_path"]))
    if not run_path.is_dir():
        raise ContractError(f"current run directory is missing: {run_path}")
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != pointer["run_manifest_sha256"]:
        raise ContractError("current run pointer does not match its manifest")
    return run_path


def _verify_external_entries(project_root: Path, entries: Any, label: str) -> None:
    if not isinstance(entries, list):
        raise ContractError(f"run manifest {label} must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise ContractError(f"run manifest {label} entry schema is invalid")
        relative = _safe_relative_path(str(entry["path"]))
        path = _inside(project_root, relative)
        if not path.is_file():
            raise ContractError(f"run input is missing: {entry['path']}")
        try:
            expected_bytes = int(entry["bytes"])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"run manifest {label} byte count is invalid") from exc
        if path.stat().st_size != expected_bytes or sha256_file(path) != entry["sha256"]:
            raise ContractError(f"run input integrity failed: {entry['path']}")


def verify_run(
    project_root: Path,
    output_root: Path,
    run_id: str | None = None,
    *,
    code_root: Path | None = None,
) -> dict[str, Any]:
    pointer = read_current_pointer(output_root) if run_id is None else None
    selected_run_id = str(pointer["run_id"]) if pointer else str(run_id)
    run_path = output_root / "runs" / selected_run_id
    if run_path.name.endswith(".tmp") or not run_path.is_dir():
        raise ContractError(f"published run is missing: {run_path}")
    manifest_path = run_path / "run_manifest.json"
    seal_path = run_path / "run_manifest_seal.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("run manifest or seal is unreadable") from exc
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION or manifest.get("run_id") != selected_run_id:
        raise ContractError("run manifest identity is invalid")
    expected_seal = {
        "schema_version": RUN_SEAL_SCHEMA_VERSION,
        "run_id": selected_run_id,
        "run_manifest_path": "run_manifest.json",
        "run_manifest_sha256": sha256_file(manifest_path),
    }
    if seal != expected_seal:
        raise ContractError("run manifest seal verification failed")
    if pointer and pointer["run_manifest_sha256"] != expected_seal["run_manifest_sha256"]:
        raise ContractError("current pointer does not match the sealed run manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ContractError("run manifest has no output inventory")
    seen: set[str] = set()
    for entry in outputs:
        required = {"path", "sha256", "bytes", "schema_version"}
        try:
            schema_version = int(entry.get("schema_version")) if isinstance(entry, dict) else 0
        except (TypeError, ValueError):
            schema_version = 0
        if not isinstance(entry, dict) or set(entry) != required or schema_version < 1:
            raise ContractError("run output manifest entry schema is invalid")
        relative = _safe_relative_path(str(entry["path"]))
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise ContractError(f"duplicate run output manifest path: {relative_text}")
        seen.add(relative_text)
        path = _inside(run_path, relative)
        if not path.is_file():
            raise ContractError(f"run output is missing: {relative_text}")
        try:
            expected_bytes = int(entry["bytes"])
        except (TypeError, ValueError) as exc:
            raise ContractError("run output byte count is invalid") from exc
        if path.stat().st_size != expected_bytes or sha256_file(path) != entry["sha256"]:
            raise ContractError(f"run output integrity failed: {relative_text}")
    actual_files = {
        path.relative_to(run_path).as_posix()
        for path in run_path.rglob("*")
        if path.is_file() and path.name not in {"run_manifest.json", "run_manifest_seal.json"}
    }
    if actual_files != seen:
        raise ContractError(
            f"run output inventory mismatch: undeclared={sorted(actual_files - seen)} missing={sorted(seen - actual_files)}"
        )
    context = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema_version", "order_binding", "outputs"}
    }
    binding = manifest.get("order_binding", {})
    if binding != {
        "algorithm": "sha256-canonical-json",
        "scope": "manifest_context_before_outputs",
        "sha256": canonical_sha256(context),
    }:
        raise ContractError("run order-binding context hash is invalid")
    _verify_external_entries(project_root, manifest.get("input_files"), "input_files")
    _verify_external_entries(code_root or project_root, manifest.get("code_files"), "code_files")
    return manifest


def remove_stale_run(output_root: Path, run_id: str) -> Path:
    stage = output_root / "runs" / f"{run_id}.tmp"
    if not stage.is_dir():
        raise ContractError(f"temporary run does not exist: {stage}")
    quarantine = output_root / "quarantine" / f"{run_id}.tmp"
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    if quarantine.exists():
        raise ContractError(f"quarantine target already exists: {quarantine}")
    os.replace(stage, quarantine)
    return quarantine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--run-id")
    recover = subparsers.add_parser("quarantine-temp")
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_run(args.project_root, args.output_root, args.run_id)
        print(json.dumps({"status": "verified", "run_id": manifest["run_id"]}, ensure_ascii=False))
    else:
        path = remove_stale_run(args.output_root, args.run_id)
        print(json.dumps({"status": "quarantined", "path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
