"""Recoverable multi-file commits for persistent paper-account artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from .core import ContractError


TRANSACTION_SCHEMA_VERSION = 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_staged(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)


def _normalized_destinations(paths: Sequence[Path]) -> set[Path]:
    destinations = {Path(path).resolve() for path in paths}
    if len(destinations) != len(paths):
        raise ContractError("transaction destinations must be unique")
    return destinations


def recover_pending_write_set(journal_path: Path, expected_destinations: Sequence[Path]) -> bool:
    """Finish an interrupted commit, but only for the caller-declared files."""
    journal = Path(journal_path)
    if not journal.exists():
        return False
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"transaction journal is unreadable: {journal}") from exc
    if payload.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise ContractError("unsupported transaction journal schema version")
    transaction_id = str(payload.get("transaction_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", transaction_id):
        raise ContractError("transaction journal has an invalid transaction id")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ContractError("transaction journal must contain files")
    expected = _normalized_destinations(expected_destinations)
    actual: set[Path] = set()
    normalized_entries: list[tuple[Path, Path, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            raise ContractError("transaction journal file entry must be an object")
        destination = Path(str(item.get("destination", ""))).resolve()
        staged = Path(str(item.get("staged", ""))).resolve()
        digest = str(item.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError("transaction journal contains an invalid file hash")
        expected_name = f".{destination.name}.{transaction_id}.staged"
        if staged.parent != destination.parent or staged.name != expected_name:
            raise ContractError("transaction journal contains an unsafe staged path")
        actual.add(destination)
        normalized_entries.append((destination, staged, digest))
    if actual != expected or len(actual) != len(normalized_entries):
        raise ContractError("transaction journal destinations do not match the requested recovery set")

    for destination, staged, digest in normalized_entries:
        if destination.is_file() and _sha256_file(destination) == digest:
            continue
        if not staged.is_file() or _sha256_file(staged) != digest:
            raise ContractError(f"transaction recovery payload is missing or corrupt: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _replace_staged(staged, destination)
    for destination, staged, digest in normalized_entries:
        if not destination.is_file() or _sha256_file(destination) != digest:
            raise ContractError(f"transaction recovery verification failed: {destination}")
        if staged.exists():
            staged.unlink()
    journal.unlink()
    return True


def commit_write_set(
    payloads: Mapping[Path, bytes],
    journal_path: Path,
) -> None:
    """Commit all bytes as one recoverable write set."""
    if not payloads:
        raise ContractError("transaction write set cannot be empty")
    destinations = _normalized_destinations(list(payloads))
    recover_pending_write_set(journal_path, list(destinations))
    transaction_id = uuid.uuid4().hex
    entries: list[dict[str, str]] = []
    journal_written = False
    journal_temp = Path(journal_path).with_suffix(Path(journal_path).suffix + ".tmp")
    try:
        for raw_destination, content in payloads.items():
            destination = Path(raw_destination).resolve()
            if not isinstance(content, bytes):
                raise ContractError("transaction payloads must be bytes")
            staged = destination.parent / f".{destination.name}.{transaction_id}.staged"
            entry = {
                "destination": str(destination),
                "staged": str(staged.resolve()),
                "sha256": _sha256_bytes(content),
            }
            entries.append(entry)
            _write_fsynced(staged, content)
        journal_payload = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "files": entries,
        }
        journal_bytes = json.dumps(
            journal_payload, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        _write_fsynced(journal_temp, journal_bytes)
        os.replace(journal_temp, journal_path)
        journal_written = True
        for item in entries:
            _replace_staged(Path(item["staged"]), Path(item["destination"]))
        recover_pending_write_set(journal_path, list(destinations))
    except Exception:
        if not journal_written:
            if journal_temp.exists():
                journal_temp.unlink()
            for item in entries:
                staged = Path(item["staged"])
                if staged.exists():
                    staged.unlink()
        raise
