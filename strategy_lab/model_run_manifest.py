from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "model_run_manifest.v1"

REQUIRED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "task_id",
    "run_id",
    "model_version",
    "baseline",
    "status",
    "started_at",
    "finished_at",
    "cwd",
    "output_dir",
    "command",
    "argv",
    "code_refs",
    "config",
    "data_refs",
    "environment",
    "selection",
    "metrics",
    "artifacts",
    "checks",
    "limitations",
    "risk_flags",
    "next_decision",
    "handoff_summary",
]

REQUIRED_ARTIFACT_FIELDS = [
    "name",
    "path",
    "kind",
    "required",
    "exists",
    "sha256",
    "size_bytes",
]

REQUIRED_REF_FIELDS = ["name", "path", "sha256", "exists"]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, args: list[str]) -> str:
    repo = root.resolve().as_posix()
    cmd = ["git", "-c", f"safe.directory={repo}", "-C", str(root), *args]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - depends on host git state
        return f"unknown: {exc}"


def artifact_info(root: Path, path: Path, required: bool = True) -> dict[str, Any]:
    try:
        rel_path = path.relative_to(root).as_posix()
    except ValueError:
        rel_path = path.as_posix()
    info: dict[str, Any] = {
        "name": path.stem,
        "path": rel_path,
        "kind": path.suffix.lstrip(".") or "file",
        "required": required,
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "row_count": None,
        "columns": [],
    }
    if path.exists() and path.suffix.lower() == ".csv":
        try:
            sample = pd.read_csv(path, nrows=5)
            info["columns"] = list(sample.columns)
            with path.open("r", encoding="utf-8-sig") as f:
                info["row_count"] = max(sum(1 for _ in f) - 1, 0)
        except Exception as exc:
            info["read_error"] = str(exc)
    return info


def file_ref(root: Path, name: str, path: Path) -> dict[str, Any]:
    try:
        rel_path = path.relative_to(root).as_posix()
    except ValueError:
        rel_path = path.as_posix()
    return {
        "name": name,
        "path": rel_path,
        "sha256": sha256_file(path),
        "exists": path.exists(),
    }


def environment_info(root: Path) -> dict[str, Any]:
    status = git_value(root, ["status", "--short"])
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timezone": datetime.now().astimezone().tzname(),
        "dependency_versions": {"pandas": pd.__version__},
        "git_commit": git_value(root, ["rev-parse", "HEAD"]),
        "git_dirty_status": status,
        "git_dirty": bool(status.strip()),
    }


def build_model_run_manifest(
    *,
    root: Path,
    task_id: str,
    run_id: str,
    model_version: str,
    baseline: str,
    status: str,
    started_at: str,
    finished_at: str,
    output_dir: Path,
    command: list[str],
    argv: dict[str, Any],
    code_paths: list[Path],
    config_path: Path,
    data_paths: list[tuple[str, Path]],
    artifact_paths: list[Path],
    selection: dict[str, Any],
    metrics: dict[str, Any],
    checks: dict[str, Any],
    limitations: list[str],
    risk_flags: list[str],
    next_decision: str,
    handoff_summary: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "model_version": model_version,
        "baseline": baseline,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "cwd": str(root.resolve()),
        "output_dir": str(output_dir),
        "command": command,
        "argv": argv,
        "code_refs": [file_ref(root, path.stem, path) for path in code_paths],
        "config": file_ref(root, config_path.stem, config_path),
        "data_refs": [file_ref(root, name, path) for name, path in data_paths],
        "environment": environment_info(root),
        "selection": selection,
        "metrics": metrics,
        "artifacts": [artifact_info(root, path, required=True) for path in artifact_paths],
        "checks": checks,
        "limitations": limitations,
        "risk_flags": risk_flags,
        "next_decision": next_decision,
        "handoff_summary": handoff_summary,
    }


def validate_model_run_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in manifest:
            findings.append({"severity": "fail", "field": field, "message": "missing required top-level field"})

    if manifest.get("schema_version") != SCHEMA_VERSION:
        findings.append({"severity": "fail", "field": "schema_version", "message": "unexpected schema version"})

    for group in ["code_refs", "data_refs"]:
        refs = manifest.get(group, [])
        if not isinstance(refs, list) or not refs:
            findings.append({"severity": "fail", "field": group, "message": "must be a non-empty list"})
            continue
        for idx, ref in enumerate(refs):
            for field in REQUIRED_REF_FIELDS:
                if field not in ref:
                    findings.append({"severity": "fail", "field": f"{group}[{idx}].{field}", "message": "missing ref field"})
            if ref.get("exists") is not True or not ref.get("sha256"):
                findings.append({"severity": "fail", "field": f"{group}[{idx}]", "message": "ref file missing or unhashed"})

    config = manifest.get("config", {})
    for field in ["path", "sha256", "exists"]:
        if field not in config:
            findings.append({"severity": "fail", "field": f"config.{field}", "message": "missing config field"})
    if config.get("exists") is not True or not config.get("sha256"):
        findings.append({"severity": "fail", "field": "config", "message": "config missing or unhashed"})

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        findings.append({"severity": "fail", "field": "artifacts", "message": "must be a non-empty list"})
    else:
        for idx, artifact in enumerate(artifacts):
            for field in REQUIRED_ARTIFACT_FIELDS:
                if field not in artifact:
                    findings.append({"severity": "fail", "field": f"artifacts[{idx}].{field}", "message": "missing artifact field"})
            if artifact.get("required") and (artifact.get("exists") is not True or not artifact.get("sha256")):
                findings.append({"severity": "fail", "field": f"artifacts[{idx}]", "message": "required artifact missing or unhashed"})

    checks = manifest.get("checks", {})
    if checks.get("self_check_pass") is not True:
        findings.append({"severity": "fail", "field": "checks.self_check_pass", "message": "self checks did not pass"})
    if checks.get("fail_count", 1) != 0:
        findings.append({"severity": "fail", "field": "checks.fail_count", "message": "fail_count must be zero"})

    for field in ["selection", "metrics", "argv"]:
        if not isinstance(manifest.get(field), dict) or not manifest.get(field):
            findings.append({"severity": "fail", "field": field, "message": "must be a non-empty object"})

    env = manifest.get("environment", {})
    if env.get("git_dirty") is True or str(env.get("git_dirty_status", "")).strip():
        findings.append({"severity": "warn", "field": "environment.git_dirty_status", "message": "workspace was dirty"})
    for field in ["python_version", "platform", "timezone", "dependency_versions", "git_commit", "git_dirty_status"]:
        if field not in env:
            findings.append({"severity": "fail", "field": f"environment.{field}", "message": "missing environment field"})

    for flag in manifest.get("risk_flags", []):
        if flag in {
            "retrospective_manifest",
            "incomplete_original_manifest",
            "hash_mismatch",
            "missing_required_artifact",
            "dirty_code_or_config",
        }:
            findings.append({"severity": "fail", "field": "risk_flags", "message": f"promotion-blocking risk flag: {flag}"})

    return findings


def schema_dict() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_top_level_fields": REQUIRED_TOP_LEVEL_FIELDS,
        "required_ref_fields": REQUIRED_REF_FIELDS,
        "required_artifact_fields": REQUIRED_ARTIFACT_FIELDS,
        "promotion_blocking": [
            "missing required top-level field",
            "missing or unhashed code_refs/data_refs/config",
            "missing or unhashed required artifact",
            "self_check_pass is not true",
            "fail_count is non-zero",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or emit strict model run manifest schema.")
    parser.add_argument("--schema-out")
    parser.add_argument("--check")
    args = parser.parse_args()

    if args.schema_out:
        path = Path(args.schema_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.check:
        manifest = json.loads(Path(args.check).read_text(encoding="utf-8"))
        findings = validate_model_run_manifest(manifest)
        print(json.dumps({"self_check_pass": not any(f["severity"] == "fail" for f in findings), "findings": findings}, ensure_ascii=False, indent=2))
        return 1 if any(f["severity"] == "fail" for f in findings) else 0

    if not args.schema_out:
        print(json.dumps(schema_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
