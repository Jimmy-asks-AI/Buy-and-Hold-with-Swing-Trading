from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from strategy_lab.long_hold_v4.pit_source_code_archive import (
    archive_manifest_code,
    archive_path_for_hash,
    authenticate_current_or_archive,
)


def test_code_archive_survives_current_path_change(tmp_path: Path) -> None:
    source = tmp_path / "collector.py"
    source.write_bytes(b"print('v1')\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"run_id": "fixture", "code_path": str(source), "code_sha256": digest}),
        encoding="utf-8",
    )
    archive_root = tmp_path / "archive"
    result = archive_manifest_code(manifest, archive_root)
    archived = archive_path_for_hash(digest, archive_root)
    assert archived.is_file()
    assert result["code_entries"][0]["sha256"] == digest
    source.write_bytes(b"print('v2')\n")
    assert authenticate_current_or_archive(source, digest, archive_root) == archived


def test_code_archive_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "collector.py"
    source.write_bytes(b"content")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"code_path": str(source), "code_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        archive_manifest_code(manifest, tmp_path / "archive")
