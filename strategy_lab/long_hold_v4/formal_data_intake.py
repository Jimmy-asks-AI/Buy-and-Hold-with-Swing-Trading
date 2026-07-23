"""Register formal PIT data revisions and build a bound target manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .core import ContractError
from .pit_gate_v2 import (
    FORMAL_INPUT_ROLES,
    canonical_json_bytes,
    register_dataset_revision,
    sha256_file,
)


INTAKE_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid intake config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("formal intake config must be a JSON object")
    return payload


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise ContractError(f"formal intake path escapes project root: {value}")
    return path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError(
            f"formal intake file is outside project root: {path}"
        ) from exc


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or re.fullmatch(
        r"[0-9a-f]{40}", commit
    ) is None:
        raise ContractError("formal intake requires a verifiable Git commit")
    return commit


def _require_clean_worktree(root: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ContractError("cannot inspect the Git worktree")
    if completed.stdout.strip():
        raise ContractError(
            "formal intake requires a clean tracked Git worktree"
        )


def _file_binding(root: Path, path_value: str) -> dict[str, str]:
    path = _safe_path(root, path_value)
    if not path.is_file():
        raise ContractError(f"formal intake file is missing: {path_value}")
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
    }


def build_formal_target_manifest(
    project_root: str | Path,
    intake_config_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_path = _safe_path(root, str(intake_config_path))
    config = _read_json(config_path)
    required = {
        "schema_version",
        "target_manifest_id",
        "decision_date",
        "required_coverage_start",
        "required_coverage_end",
        "version_store_root",
        "output_manifest_path",
        "target_generation",
        "point_in_time_usage_path",
        "formal_inputs",
        "datasets",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"formal intake config missing fields: {missing}")
    if config["schema_version"] != INTAKE_SCHEMA_VERSION:
        raise ContractError(
            f"formal intake requires schema_version={INTAKE_SCHEMA_VERSION}"
        )
    _require_clean_worktree(root)
    commit = _git_head(root)

    generation = config["target_generation"]
    if not isinstance(generation, dict):
        raise ContractError("target_generation must be an object")
    code_paths = generation.get("code_paths")
    if not isinstance(code_paths, list) or not code_paths:
        raise ContractError("target_generation code_paths are required")
    code_bindings = [_file_binding(root, str(path)) for path in code_paths]
    generation_config = _file_binding(
        root, str(generation.get("config_path", ""))
    )

    formal_input_map = config["formal_inputs"]
    if not isinstance(formal_input_map, dict):
        raise ContractError("formal_inputs must be a role-to-path object")
    if set(formal_input_map) != FORMAL_INPUT_ROLES:
        raise ContractError(
            "formal_inputs roles must match the formal evaluation contract"
        )
    formal_inputs = [
        {
            "role": role,
            **_file_binding(root, str(formal_input_map[role])),
        }
        for role in sorted(FORMAL_INPUT_ROLES)
    ]
    usage_binding = _file_binding(
        root, str(config["point_in_time_usage_path"])
    )

    output_path = _safe_path(root, str(config["output_manifest_path"]))
    if output_path.exists():
        raise ContractError(
            f"formal target manifest is immutable: {output_path}"
        )

    dataset_configs = config["datasets"]
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ContractError("formal intake datasets must be non-empty")
    entries: list[dict[str, Any]] = []
    temporary: Path | None = None
    try:
        for item in dataset_configs:
            if not isinstance(item, dict):
                raise ContractError(
                    "formal intake dataset item must be an object"
                )
            source_path = _safe_path(
                root, str(item.get("source_path", ""))
            )
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                raise ContractError(
                    "formal intake dataset metadata is required"
                )
            entries.append(
                register_dataset_revision(
                    root,
                    str(config["version_store_root"]),
                    source_path,
                    metadata,
                )
            )

        manifest = {
            "schema_version": 1,
            "target_manifest_id": str(config["target_manifest_id"]),
            "decision_date": str(config["decision_date"]),
            "required_coverage_start": str(
                config["required_coverage_start"]
            ),
            "required_coverage_end": str(config["required_coverage_end"]),
            "target_generation": {
                "code_commit": commit,
                "code_files": code_bindings,
                "config": generation_config,
            },
            "point_in_time_usage": usage_binding,
            "formal_inputs": formal_inputs,
            "datasets": entries,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.name}.tmp-{uuid.uuid4().hex}"
        )
        temporary.write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, output_path)
    except Exception:
        if temporary is not None and temporary.is_file():
            temporary.unlink()
        for entry in reversed(entries):
            manifest_path = _safe_path(
                root, str(entry["manifest_path"])
            )
            version_directory = manifest_path.parent
            if (
                manifest_path.is_file()
                and sha256_file(manifest_path)
                == str(entry["manifest_sha256"])
            ):
                shutil.rmtree(version_directory)
        raise
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register formal PIT data and build a target manifest"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--intake-config", required=True)
    args = parser.parse_args()
    path = build_formal_target_manifest(
        args.project_root, args.intake_config
    )
    print(
        json.dumps(
            {
                "status": "FORMAL_TARGET_MANIFEST_CREATED",
                "path": str(path),
                "sha256": sha256_file(path),
                "promotion_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
