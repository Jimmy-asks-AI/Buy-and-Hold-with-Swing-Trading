import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4.core import ContractError
from strategy_lab.long_hold_v4.formal_data_intake import (
    build_formal_target_manifest,
)
from strategy_lab.long_hold_v4.pit_gate_v2 import (
    FORMAL_INPUT_ROLES,
    canonical_json_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_COMMIT = "b" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _intake_fixture(root: Path, *, invalid_second: bool = False) -> Path:
    source = root / "source.csv"
    pd.DataFrame(
        [
            {
                "source_row_key": "row-1",
                "asset": "AAA",
                "available_date": "2020-01-02",
                "value": 1.0,
            }
        ]
    ).to_csv(source, index=False)
    formal = root / "formal.csv"
    pd.DataFrame([{"fixture": 1}]).to_csv(formal, index=False)
    usage = root / "usage.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "dataset_a",
                "source_revision_id": "r1",
                "source_row_key": "row-1",
                "asset": "AAA",
                "decision_date": "2020-01-02",
                "available_date": "2020-01-02",
            }
        ]
    ).to_csv(usage, index=False)
    code = root / "generator.py"
    code.write_text("# fixture\n", encoding="utf-8")
    target_config = root / "target-config.json"
    _write_json(target_config, {"schema_version": 1})

    metadata = {
        "dataset_id": "dataset_a",
        "provider": "fixture",
        "license": "LICENSED_RESEARCH_USE",
        "coverage_start": "2020-01-01",
        "coverage_end": "2020-12-31",
        "observed_at": "2021-01-01T00:00:00+08:00",
        "available_date": "2020-01-02",
        "revision_id": "r1",
        "symbol": "ALL",
        "frequency": "daily",
        "schema_version": 1,
        "pit_status": "QUALIFIED_PIT",
        "known_limitations": "fixture",
        "current_snapshot": False,
        "contains_current_snapshot_backfill": False,
        "contains_current_constituents_backfill": False,
    }
    datasets = [{"source_path": source.name, "metadata": metadata}]
    if invalid_second:
        datasets.append(
            {
                "source_path": source.name,
                "metadata": {
                    **metadata,
                    "dataset_id": "dataset_b",
                    "revision_id": "r1",
                    "provider": "",
                },
            }
        )
    config = {
        "schema_version": 1,
        "target_manifest_id": "formal-intake-fixture",
        "decision_date": "2020-12-31",
        "required_coverage_start": "2020-01-01",
        "required_coverage_end": "2020-12-31",
        "version_store_root": "versions",
        "output_manifest_path": "target-manifest.json",
        "target_generation": {
            "code_paths": [code.name],
            "config_path": target_config.name,
        },
        "point_in_time_usage_path": usage.name,
        "formal_inputs": {
            role: formal.name for role in FORMAL_INPUT_ROLES
        },
        "datasets": datasets,
    }
    config_path = root / "intake.json"
    _write_json(config_path, config)
    return config_path


def test_formal_data_intake_builds_immutable_bound_manifest() -> None:
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        root = Path(tmp)
        config_path = _intake_fixture(root)
        with (
            patch(
                "strategy_lab.long_hold_v4.formal_data_intake._git_head",
                return_value=TEST_COMMIT,
            ),
            patch(
                "strategy_lab.long_hold_v4.formal_data_intake._require_clean_worktree"
            ),
        ):
            output = build_formal_target_manifest(
                root, config_path.name
            )
            assert output.is_file()
            payload = json.loads(output.read_text(encoding="utf-8"))
            assert payload["target_generation"]["code_commit"] == TEST_COMMIT
            assert len(payload["formal_inputs"]) == len(FORMAL_INPUT_ROLES)
            assert (
                root
                / payload["datasets"][0]["manifest_path"]
            ).is_file()
            try:
                build_formal_target_manifest(root, config_path.name)
            except ContractError as exc:
                assert "manifest is immutable" in str(exc)
            else:
                raise AssertionError("formal target manifest was overwritten")


def test_formal_data_intake_rolls_back_new_revisions_on_failure() -> None:
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        root = Path(tmp)
        config_path = _intake_fixture(root, invalid_second=True)
        with (
            patch(
                "strategy_lab.long_hold_v4.formal_data_intake._git_head",
                return_value=TEST_COMMIT,
            ),
            patch(
                "strategy_lab.long_hold_v4.formal_data_intake._require_clean_worktree"
            ),
        ):
            try:
                build_formal_target_manifest(root, config_path.name)
            except ContractError as exc:
                assert "metadata missing fields" in str(exc)
            else:
                raise AssertionError("invalid batch intake unexpectedly passed")
        assert not (root / "versions" / "dataset_a" / "r1").exists()
        assert not (root / "target-manifest.json").exists()
