import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4.core import ContractError
from strategy_lab.long_hold_v4.pit_gate_v2 import (
    canonical_json_bytes,
    register_dataset_revision,
    run_versioned_pit_gate,
    sha256_file,
)
from strategy_lab.long_hold_v4.walk_forward import verify_pit_gate_binding


TEST_COMMIT = "a" * 40


class WorkPackage5PitGateTests(unittest.TestCase):
    def _workspace(self):
        return tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1]
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(payload))

    def _gate_config(self, root: Path, dataset_id: str) -> Path:
        (root / "gate.py").write_text("# gate fixture\n", encoding="utf-8")
        path = root / "gate_config.json"
        self._write_json(
            path,
            {
                "schema_version": 1,
                "output_root": "outputs",
                "version_store_root": "versions",
                "required_dataset_ids": [dataset_id],
                "allowed_license_values": ["LICENSED_RESEARCH_USE"],
                "maximum_manifest_age_days": 7,
                "gate_code_paths": ["gate.py"],
                "promotion_allowed": False,
                "independent_review_required": True,
            },
        )
        return path

    def _registered_entry(
        self,
        root: Path,
        *,
        dataset_id: str = "stock_history",
        revision_id: str = "r1",
        available_date: str = "2026-07-23",
        current_snapshot: bool = False,
        current_constituents: bool = False,
        previous_revision_id: str | None = None,
        previous_manifest_sha256: str | None = None,
    ) -> dict:
        source = root / f"{dataset_id}-{revision_id}.csv"
        pd.DataFrame(
            [
                {
                    "date": "2020-01-02",
                    "symbol": "000001",
                    "source_row_key": f"{dataset_id}-row-1",
                    "asset": "000001",
                    "available_date": available_date,
                    "value": 1.0,
                }
            ]
        ).to_csv(source, index=False)
        metadata = {
            "dataset_id": dataset_id,
            "provider": "fixture_provider",
            "license": "LICENSED_RESEARCH_USE",
            "coverage_start": "2020-01-01",
            "coverage_end": "2026-07-23",
            "observed_at": "2026-07-23T08:00:00+08:00",
            "available_date": available_date,
            "revision_id": revision_id,
            "symbol": "ALL",
            "frequency": "daily",
            "schema_version": 1,
            "pit_status": "QUALIFIED_PIT",
            "known_limitations": "synthetic fixture only",
            "current_snapshot": current_snapshot,
            "contains_current_snapshot_backfill": current_snapshot,
            "contains_current_constituents_backfill": current_constituents,
        }
        if previous_revision_id is not None:
            metadata["previous_revision_id"] = previous_revision_id
            metadata["previous_manifest_sha256"] = previous_manifest_sha256
        return register_dataset_revision(root, "versions", source, metadata)

    def _target_manifest(
        self,
        root: Path,
        entries: list[dict],
        *,
        name: str = "target.json",
        usage_decision_date: str = "2026-07-23",
        usage_available_date: str | None = None,
    ) -> Path:
        target_code = root / "target_generator.py"
        target_config = root / "target_config.json"
        target_code.write_text("# target fixture\n", encoding="utf-8")
        self._write_json(target_config, {"schema_version": 1})
        usage_path = root / f"{name.removesuffix('.json')}-pit-usage.csv"
        pd.DataFrame(
            [
                {
                    "dataset_id": entry["dataset_id"],
                    "source_revision_id": entry["revision_id"],
                    "source_row_key": f"{entry['dataset_id']}-row-1",
                    "asset": "000001",
                    "decision_date": usage_decision_date,
                    "available_date": (
                        usage_available_date
                        if usage_available_date is not None
                        else entry["available_date"]
                    ),
                }
                for entry in entries
            ]
        ).to_csv(usage_path, index=False)
        formal_inputs = []
        for role in (
            "validation_execution_states",
            "validation_target_weights",
            "validation_benchmark_returns",
            "independent_execution_states",
            "independent_target_weights",
            "independent_benchmark_returns",
            "trading_calendar",
            "candidate_registry",
        ):
            formal_path = root / f"{name.removesuffix('.json')}-{role}.csv"
            pd.DataFrame([{"fixture": role}]).to_csv(formal_path, index=False)
            formal_inputs.append(
                {
                    "role": role,
                    "path": formal_path.name,
                    "sha256": sha256_file(formal_path),
                }
            )
        target = {
            "schema_version": 1,
            "target_manifest_id": name.removesuffix(".json"),
            "decision_date": "2026-07-23",
            "required_coverage_start": "2020-01-01",
            "required_coverage_end": "2026-07-23",
            "target_generation": {
                "code_commit": TEST_COMMIT,
                "code_files": [
                    {
                        "path": "target_generator.py",
                        "sha256": sha256_file(target_code),
                    }
                ],
                "config": {
                    "path": "target_config.json",
                    "sha256": sha256_file(target_config),
                },
            },
            "point_in_time_usage": {
                "path": usage_path.name,
                "sha256": sha256_file(usage_path),
            },
            "formal_inputs": formal_inputs,
            "datasets": entries,
        }
        path = root / name
        self._write_json(path, target)
        return path

    def _run(
        self,
        root: Path,
        config: Path,
        target: Path,
        *,
        run_id: str,
    ) -> dict:
        with patch(
            "strategy_lab.long_hold_v4.pit_gate_v2._git_head",
            return_value=TEST_COMMIT,
        ):
            paths = run_versioned_pit_gate(
                root,
                config,
                target,
                run_id=run_id,
                gate_observed_at="2026-07-23T12:00:00+08:00",
            )
        return json.loads(paths["decision"].read_text(encoding="utf-8"))

    def test_fully_bound_versioned_dataset_passes_gate_without_promotion(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root)
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(root, [entry])
            decision = self._run(root, config, target, run_id="pass-fixture")
            self.assertEqual(decision["status"], "PASS_PIT_GATE")
            self.assertTrue(decision["formal_backtest_allowed"])
            self.assertFalse(decision["promotion_allowed"])
            self.assertTrue(decision["independent_review_required"])
            self.assertFalse(decision["manual_review_signed"])

    def test_available_date_later_than_decision_date_fails_gate(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root, available_date="2026-07-24")
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(root, [entry])
            decision = self._run(root, config, target, run_id="future-date")
            self.assertEqual(decision["status"], "BLOCKED_PIT_GATE")
            self.assertIn(
                "stock_history:available_date_not_future",
                decision["failure_reasons"],
            )
            self.assertIn(
                "stock_history:row_available_dates_not_future",
                decision["failure_reasons"],
            )

    def test_usage_ledger_blocks_data_unavailable_at_historical_decision(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root)
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(
                root,
                [entry],
                usage_decision_date="2020-01-02",
            )
            decision = self._run(
                root, config, target, run_id="historical-lookahead"
            )
            self.assertIn(
                "__target__:pit_usage_no_historical_lookahead",
                decision["failure_reasons"],
            )

    def test_usage_ledger_available_date_must_match_versioned_source_row(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(
                root, available_date="2020-01-03"
            )
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(
                root,
                [entry],
                usage_decision_date="2020-01-03",
                usage_available_date="2020-01-02",
            )
            decision = self._run(
                root, config, target, run_id="forged-usage-date"
            )
            self.assertIn(
                "__target__:pit_usage_source_metadata_match",
                decision["failure_reasons"],
            )

    def test_current_snapshot_backfill_fails_gate(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root, current_snapshot=True)
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(root, [entry])
            decision = self._run(root, config, target, run_id="snapshot-backfill")
            self.assertIn(
                "stock_history:no_current_snapshot_backfill",
                decision["failure_reasons"],
            )

    def test_current_constituent_weights_used_for_history_fail_gate(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root, current_constituents=True)
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(root, [entry])
            decision = self._run(
                root, config, target, run_id="constituent-backfill"
            )
            self.assertIn(
                "stock_history:no_current_constituents_backfill",
                decision["failure_reasons"],
            )

    def test_revision_cannot_overwrite_existing_version(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            self._registered_entry(root, revision_id="r1")
            with self.assertRaisesRegex(
                ContractError, "revision overwrite prohibited"
            ):
                self._registered_entry(root, revision_id="r1")

    def test_new_revision_preserves_and_binds_old_manifest(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            first = self._registered_entry(root, revision_id="r1")
            second = self._registered_entry(
                root,
                revision_id="r2",
                previous_revision_id="r1",
                previous_manifest_sha256=first["manifest_sha256"],
            )
            self.assertTrue((root / first["manifest_path"]).is_file())
            self.assertTrue((root / second["manifest_path"]).is_file())
            self.assertNotEqual(first["file_path"], second["file_path"])

    def test_missing_manifest_or_hash_mismatch_fails_gate(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root)
            config = self._gate_config(root, entry["dataset_id"])

            missing_manifest = copy.deepcopy(entry)
            missing_manifest["manifest_path"] = (
                "versions/stock_history/r1/missing.json"
            )
            target = self._target_manifest(
                root, [missing_manifest], name="missing-manifest.json"
            )
            decision = self._run(
                root, config, target, run_id="missing-manifest"
            )
            self.assertIn(
                "stock_history:dataset_manifest_hash_match",
                decision["failure_reasons"],
            )

            bad_hash = copy.deepcopy(entry)
            bad_hash["file_sha256"] = "0" * 64
            target = self._target_manifest(
                root, [bad_hash], name="bad-hash.json"
            )
            decision = self._run(root, config, target, run_id="bad-hash")
            self.assertIn(
                "stock_history:file_hash_match",
                decision["failure_reasons"],
            )

    def test_walk_forward_rechecks_all_gate_bindings_before_use(self):
        with self._workspace() as tmp:
            root = Path(tmp)
            entry = self._registered_entry(root)
            config = self._gate_config(root, entry["dataset_id"])
            target = self._target_manifest(root, [entry])
            with patch(
                "strategy_lab.long_hold_v4.pit_gate_v2._git_head",
                return_value=TEST_COMMIT,
            ):
                paths = run_versioned_pit_gate(
                    root,
                    config,
                    target,
                    run_id="binding-recheck",
                    gate_observed_at="2026-07-23T12:00:00+08:00",
                )
            binding = verify_pit_gate_binding(paths["run_directory"], root)
            self.assertEqual(binding["pit_gate_run_id"], "binding-recheck")
            self.assertFalse(binding["promotion_allowed"])

            data_path = root / entry["file_path"]
            data_path.write_text(
                data_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "dataset file.*hash mismatch"):
                verify_pit_gate_binding(paths["run_directory"], root)


if __name__ == "__main__":
    unittest.main()
