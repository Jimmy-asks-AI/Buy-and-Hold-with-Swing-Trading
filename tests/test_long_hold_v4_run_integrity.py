from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from strategy_lab.long_hold_v4 import snapshot_store
from strategy_lab.long_hold_v4.core import ContractError
from strategy_lab.long_hold_v4.pipeline import verify_source_manifest
from strategy_lab.long_hold_v4.run_artifacts import (
    RunArtifactPublisher,
    read_current_pointer,
    remove_stale_run,
    sha256_file,
    verify_run,
)


def _manifest_context(project_root: Path, run_id: str) -> dict:
    input_path = project_root / "snapshot.csv"
    code_path = project_root / "builder.py"
    if not input_path.exists():
        input_path.write_text("asset,value\n600000,1\n", encoding="utf-8")
    if not code_path.exists():
        code_path.write_text("VALUE = 1\n", encoding="utf-8")
    return {
        "model": {"name": "fixture"},
        "run_id": run_id,
        "run_at": "2026-07-22T09:00:00",
        "as_of_date": "2026-07-21",
        "system_status": "CASH_DATA_BLOCKED",
        "config_sha256": "a" * 64,
        "account_version": 1,
        "account_state_sha256": "b" * 64,
        "order_envelope_schema_version": 1,
        "input_files": [
            {"path": "snapshot.csv", "sha256": sha256_file(input_path), "bytes": input_path.stat().st_size}
        ],
        "code_files": [
            {"path": "builder.py", "sha256": sha256_file(code_path), "bytes": code_path.stat().st_size}
        ],
        "runtime": {"python": "fixture", "pandas": pd.__version__},
    }


def _publish(project_root: Path, output_root: Path, run_id: str) -> dict[str, Path]:
    publisher = RunArtifactPublisher(output_root, run_id)
    publisher.begin()
    for name in [
        "candidate_decisions.csv",
        "target_weights.csv",
        "order_intents.csv",
        "snapshot_copy.csv",
    ]:
        publisher.write_csv(name, pd.DataFrame([{"asset": "600000", "value": 1}]), schema_version=1)
    publisher.write_json("readiness.json", {"system_status": "CASH_DATA_BLOCKED"}, schema_version=1)
    return publisher.finalize(_manifest_context(project_root, run_id))


def test_each_declared_artifact_and_external_input_is_hash_verified(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    published = _publish(tmp_path, output_root, "run-001")
    manifest = verify_run(tmp_path, output_root)
    assert manifest["run_id"] == "run-001"

    for entry in manifest["outputs"]:
        path = published["run"] / entry["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with pytest.raises(ContractError, match="run output integrity failed"):
            verify_run(tmp_path, output_root)
        path.write_bytes(original)

    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("asset,value\n600000,2\n", encoding="utf-8")
    with pytest.raises(ContractError, match="run input integrity failed"):
        verify_run(tmp_path, output_root)


@pytest.mark.parametrize(
    "failure_point",
    ["stage_created", "artifact_written:readiness.json", "sealed", "run_published"],
)
def test_interrupted_run_never_repoints_current(tmp_path: Path, failure_point: str) -> None:
    output_root = tmp_path / "outputs"
    _publish(tmp_path, output_root, "complete")

    def fail(name: str) -> None:
        if name == failure_point:
            raise RuntimeError("injected interruption")

    publisher = RunArtifactPublisher(output_root, f"failed-{failure_point.replace(':', '-')}", failure_hook=fail)
    with pytest.raises(RuntimeError, match="injected interruption"):
        publisher.begin()
        publisher.write_json("readiness.json", {"ok": False}, schema_version=1)
        publisher.finalize(_manifest_context(tmp_path, publisher.run_id))
    assert read_current_pointer(output_root)["run_id"] == "complete"
    assert verify_run(tmp_path, output_root)["run_id"] == "complete"


def test_stale_temp_requires_explicit_quarantine(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    publisher = RunArtifactPublisher(output_root, "stale")
    publisher.begin()
    publisher.write_json("readiness.json", {"ok": False}, schema_version=1)
    with pytest.raises(ContractError, match="explicit recovery"):
        RunArtifactPublisher(output_root, "stale").begin()
    quarantine = remove_stale_run(output_root, "stale")
    assert quarantine.is_dir()
    assert not (output_root / "runs" / "stale.tmp").exists()


def test_wrong_current_pointer_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _publish(tmp_path, output_root, "complete")
    pointer = json.loads((output_root / "current").read_text(encoding="utf-8"))
    pointer["relative_path"] = "../outside"
    (output_root / "current").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ContractError, match="unsafe run artifact path"):
        verify_run(tmp_path, output_root)


def test_undeclared_file_in_published_run_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    published = _publish(tmp_path, output_root, "complete")
    (published["run"] / "undeclared.txt").write_text("not inventoried", encoding="utf-8")
    with pytest.raises(ContractError, match="output inventory mismatch"):
        verify_run(tmp_path, output_root)


def test_manifest_and_seal_tampering_are_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    published = _publish(tmp_path, output_root, "complete")
    manifest_bytes = published["manifest"].read_bytes()
    published["manifest"].write_bytes(manifest_bytes + b" ")
    with pytest.raises(ContractError, match="seal verification failed"):
        verify_run(tmp_path, output_root)
    published["manifest"].write_bytes(manifest_bytes)
    seal = json.loads(published["seal"].read_text(encoding="utf-8"))
    seal["run_manifest_sha256"] = "0" * 64
    published["seal"].write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ContractError, match="seal verification failed"):
        verify_run(tmp_path, output_root)


def test_snapshot_lock_times_out_instead_of_overwriting_owner(tmp_path: Path) -> None:
    lock = tmp_path / ".snapshot_build.lock"
    lock.write_text('{"pid": 1}', encoding="utf-8")
    with pytest.raises(ContractError, match="lock timeout"):
        with snapshot_store.snapshot_write_lock(lock, timeout_seconds=0.01, poll_seconds=0.001):
            pass


def _configure_snapshot_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(snapshot_store, "COMBINED_PATH", root / "research_snapshot.csv")
    monkeypatch.setattr(snapshot_store, "COMBINED_MANIFEST_PATH", root / "combined_snapshot_manifest.json")
    monkeypatch.setattr(
        snapshot_store,
        "PART_PATHS",
        {"stock": root / "stock_research_snapshot.csv", "etf": root / "etf_research_snapshot.csv"},
    )


def _part(asset_type: str, asset: str, as_of: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"as_of_date": as_of, "asset": asset, "asset_type": asset_type, "sector": "fixture"}]
    )


def test_snapshot_parts_require_same_date_and_never_publish_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_snapshot_paths(monkeypatch, tmp_path)
    code = tmp_path / "builder.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    first = snapshot_store.write_snapshot_part(
        "stock", _part("stock", "600000", "2026-07-20"), builder_code_paths=[code], project_root=tmp_path
    )
    assert first.empty
    with pytest.raises(ContractError, match="as_of_date mismatch"):
        snapshot_store.write_snapshot_part(
            "etf", _part("etf", "510880", "2026-07-21"), builder_code_paths=[code], project_root=tmp_path
        )
    assert not snapshot_store.COMBINED_MANIFEST_PATH.exists()


def test_concurrent_snapshot_builders_serialize_and_publish_bound_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_snapshot_paths(monkeypatch, tmp_path)
    code = tmp_path / "builder.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")

    def write(kind: str, asset: str) -> pd.DataFrame:
        return snapshot_store.write_snapshot_part(
            kind,
            _part(kind, asset, "2026-07-21"),
            builder_config={"kind": kind},
            builder_code_paths=[code],
            project_root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda args: write(*args), [("stock", "600000"), ("etf", "510880")]))
    assert any(len(frame) == 2 for frame in results)
    combined = pd.read_csv(snapshot_store.COMBINED_PATH, dtype={"asset": str}, encoding="utf-8-sig")
    assert set(combined["asset"].str.zfill(6)) == {"600000", "510880"}
    manifest = json.loads(snapshot_store.COMBINED_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["as_of_date"] == "2026-07-21"
    assert manifest["combined"]["sha256"] == sha256_file(snapshot_store.COMBINED_PATH)
    assert set(manifest["parts"]) == {"stock", "etf"}
    assert verify_source_manifest(tmp_path, snapshot_store.COMBINED_MANIFEST_PATH, pd.Timestamp("2026-07-21")) == []
    snapshot_store.COMBINED_PATH.write_bytes(snapshot_store.COMBINED_PATH.read_bytes() + b"tampered")
    failures = verify_source_manifest(tmp_path, snapshot_store.COMBINED_MANIFEST_PATH, pd.Timestamp("2026-07-21"))
    assert any("source_manifest_hash_mismatch" in failure for failure in failures)
