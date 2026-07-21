"""Archive source-run code by content hash so immutable runs remain verifiable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "data_raw" / "long_hold_v4" / "source_code_archive"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def archive_path_for_hash(digest: str, archive_root: Path = ARCHIVE_ROOT) -> Path:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise ValueError("invalid SHA-256 digest for code archive")
    return archive_root / digest[:2].lower() / f"{digest.lower()}.source"


def authenticate_current_or_archive(
    current_path: Path,
    expected_sha256: str,
    archive_root: Path = ARCHIVE_ROOT,
) -> Path:
    if current_path.is_file() and _sha256(current_path) == expected_sha256:
        return current_path
    archived = archive_path_for_hash(expected_sha256, archive_root)
    if archived.is_file() and _sha256(archived) == expected_sha256:
        return archived
    raise ValueError(f"code hash is unavailable from current path and archive: {expected_sha256}")


def _atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def archive_manifest_code(manifest_path: Path, archive_root: Path = ARCHIVE_ROOT) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("code_files")
    if not isinstance(declared, list) or not declared:
        declared = [{"path": manifest.get("code_path"), "sha256": manifest.get("code_sha256")}]
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in declared:
        source = Path(str(item.get("path", "")))
        source = source if source.is_absolute() else ROOT / source
        expected = str(item.get("sha256", "")).lower()
        key = (_relative(source), expected)
        if key in seen:
            continue
        seen.add(key)
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"declared code does not match source-run manifest: {source}")
        destination = archive_path_for_hash(expected, archive_root)
        if destination.is_file():
            if _sha256(destination) != expected:
                raise ValueError(f"content-addressed code archive is corrupt: {destination}")
        else:
            _atomic_bytes(source.read_bytes(), destination)
        entries.append(
            {
                "original_path": _relative(source),
                "sha256": expected,
                "archive_path": _relative(destination),
                "bytes": destination.stat().st_size,
            }
        )
    manifest_hash = _sha256(manifest_path)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source_run_id": manifest.get("run_id"),
        "source_manifest_path": _relative(manifest_path),
        "source_manifest_sha256": manifest_hash,
        "code_entries": entries,
    }
    index_path = archive_root / "manifests" / f"{manifest_hash}.json"
    _atomic_json(result, index_path)
    return {**result, "archive_index_path": _relative(index_path), "archive_index_sha256": _sha256(index_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    args = parser.parse_args()
    results = [archive_manifest_code(path, args.archive_root) for path in args.manifests]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
