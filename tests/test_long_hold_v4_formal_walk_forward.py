import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4.formal_walk_forward import (
    run_formal_walk_forward,
    verify_formal_run,
)
from strategy_lab.long_hold_v4.core import ContractError
from strategy_lab.long_hold_v4.pit_gate_v2 import (
    canonical_json_bytes,
    register_dataset_revision,
    run_versioned_pit_gate,
    sha256_file,
)
from strategy_lab.long_hold_v4.walk_forward import (
    build_purged_embargoed_plan,
    load_walk_forward_config,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_COMMIT = "a" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def test_formal_runner_enforces_gate_and_publishes_all_windows() -> None:
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        root = Path(tmp)
        strategy_config = json.loads(
            (ROOT / "configs" / "long_hold_v4.json").read_text(
                encoding="utf-8"
            )
        )
        walk_config = json.loads(
            (
                ROOT
                / "configs"
                / "long_hold_v4_work_package_5_walk_forward.json"
            ).read_text(encoding="utf-8")
        )
        walk_config["output_root"] = "wf_runs"
        strategy_path = root / "strategy.json"
        walk_path = root / "walk.json"
        _write_json(strategy_path, strategy_config)
        _write_json(walk_path, walk_config)
        loaded_walk_config = load_walk_forward_config(walk_path)

        calendar = pd.bdate_range("2020-01-01", periods=1300)
        plan = build_purged_embargoed_plan(calendar, loaded_walk_config)
        independent_start = pd.Timestamp(plan["independent_test"]["start"])
        calendar_path = root / "calendar.csv"
        pd.DataFrame({"date": calendar.date.astype(str)}).to_csv(
            calendar_path, index=False
        )

        rows = []
        for number, date in enumerate(calendar):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "asset": "AAA",
                    "asset_type": "stock",
                    "open": 10.0 + number * 0.001,
                    "close": 10.0 + number * 0.0015,
                    "return_basis": "qfq_adjusted",
                    "price_basis": "qfq_adjusted",
                    "available_date": date.date().isoformat(),
                    "list_date": "2010-01-01",
                    "delist_date": None,
                    "has_market_data": True,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_delisted": False,
                }
            )
        all_prices = pd.DataFrame(rows)
        validation_prices = all_prices[
            pd.to_datetime(all_prices["date"]) < independent_start
        ]
        independent_prices = all_prices[
            pd.to_datetime(all_prices["date"]) >= independent_start
        ]
        validation_prices_path = root / "validation-prices.csv"
        independent_prices_path = root / "independent-prices.csv"
        validation_prices.to_csv(validation_prices_path, index=False)
        independent_prices.to_csv(independent_prices_path, index=False)

        def target(window_id: str, signal_date: str) -> dict:
            return {
                "signal_date": signal_date,
                "available_date": signal_date,
                "asset": "AAA",
                "sector": "bank",
                "historical_backtest_allowed": True,
                "target_core_weight": 0.10,
                "target_t_weight": 0.0,
                "target_semantics": "FULL_SNAPSHOT",
                "target_schema_version": 2,
                "snapshot_asset_count": 1,
                "window_id": window_id,
                "candidate_id": "candidate-1",
            }

        validation_targets = pd.DataFrame(
            [
                target(fold["window_id"], fold["validation_start"])
                for fold in plan["validation_windows"]
            ]
        )
        independent_targets = pd.DataFrame(
            [
                target(
                    plan["independent_test"]["window_id"],
                    plan["independent_test"]["start"],
                )
            ]
        )
        validation_targets_path = root / "validation-targets.csv"
        independent_targets_path = root / "independent-targets.csv"
        validation_targets.to_csv(validation_targets_path, index=False)
        independent_targets.to_csv(independent_targets_path, index=False)
        benchmark_rows = pd.DataFrame(
            {
                "date": calendar.date.astype(str),
                "benchmark_id": "000985.CSI",
                "total_return": 0.0,
                "available_date": calendar.date.astype(str),
                "return_basis": "total_return",
                "historical_backtest_allowed": True,
            }
        )
        validation_benchmark_path = root / "validation-benchmark.csv"
        independent_benchmark_path = root / "independent-benchmark.csv"
        benchmark_rows[
            pd.to_datetime(benchmark_rows["date"]) < independent_start
        ].to_csv(validation_benchmark_path, index=False)
        benchmark_rows[
            pd.to_datetime(benchmark_rows["date"]) >= independent_start
        ].to_csv(independent_benchmark_path, index=False)

        candidate_path = root / "candidates.csv"
        pd.DataFrame(
            [
                {
                    "candidate_id": "candidate-1",
                    "parameters_json": '{"lookback": 20}',
                    "train_score": 0.5,
                    "split_roles_used": "train+validation",
                }
            ]
        ).to_csv(candidate_path, index=False)

        source = root / "source.csv"
        pd.DataFrame(
            [
                {
                    "date": calendar[0].date().isoformat(),
                    "asset": "AAA",
                    "source_row_key": "source-row-0",
                    "available_date": calendar[0].date().isoformat(),
                    "value": 1.0,
                }
            ]
        ).to_csv(source, index=False)
        entry = register_dataset_revision(
            root,
            "versions",
            source,
            {
                "dataset_id": "stock_history",
                "provider": "fixture_provider",
                "license": "LICENSED_RESEARCH_USE",
                "coverage_start": calendar[0].date().isoformat(),
                "coverage_end": calendar[-1].date().isoformat(),
                "observed_at": (
                    pd.Timestamp(calendar[-1])
                    .tz_localize("Asia/Shanghai")
                    .isoformat()
                ),
                "available_date": calendar[0].date().isoformat(),
                "revision_id": "r1",
                "symbol": "ALL",
                "frequency": "daily",
                "schema_version": 1,
                "pit_status": "QUALIFIED_PIT",
                "known_limitations": "synthetic fixture",
                "current_snapshot": False,
                "contains_current_snapshot_backfill": False,
                "contains_current_constituents_backfill": False,
            },
        )

        usage_path = root / "pit-usage.csv"
        usage_dates = [
            fold["validation_start"] for fold in plan["validation_windows"]
        ] + [plan["independent_test"]["start"]]
        pd.DataFrame(
            [
                {
                    "dataset_id": "stock_history",
                    "source_revision_id": "r1",
                    "source_row_key": "source-row-0",
                    "asset": "AAA",
                    "decision_date": date,
                    "available_date": calendar[0].date().isoformat(),
                }
                for number, date in enumerate(usage_dates)
            ]
        ).to_csv(usage_path, index=False)

        formal_inputs = [
            ("validation_execution_states", validation_prices_path),
            ("validation_target_weights", validation_targets_path),
            ("validation_benchmark_returns", validation_benchmark_path),
            ("independent_execution_states", independent_prices_path),
            ("independent_target_weights", independent_targets_path),
            (
                "independent_benchmark_returns",
                independent_benchmark_path,
            ),
            ("trading_calendar", calendar_path),
            ("candidate_registry", candidate_path),
        ]
        generator = root / "generator.py"
        gate_code = root / "gate.py"
        runner_code = root / "runner.py"
        generator.write_text("# generator fixture\n", encoding="utf-8")
        gate_code.write_text("# gate fixture\n", encoding="utf-8")
        runner_code.write_text("# runner fixture\n", encoding="utf-8")
        target_manifest_path = root / "target-manifest.json"
        _write_json(
            target_manifest_path,
            {
                "schema_version": 1,
                "target_manifest_id": "formal-fixture",
                "decision_date": calendar[-1].date().isoformat(),
                "required_coverage_start": calendar[0].date().isoformat(),
                "required_coverage_end": calendar[-1].date().isoformat(),
                "target_generation": {
                    "code_commit": TEST_COMMIT,
                    "code_files": [
                        {
                            "path": generator.name,
                            "sha256": sha256_file(generator),
                        }
                    ],
                    "config": {
                        "path": walk_path.name,
                        "sha256": sha256_file(walk_path),
                    },
                },
                "point_in_time_usage": {
                    "path": usage_path.name,
                    "sha256": sha256_file(usage_path),
                },
                "formal_inputs": [
                    {
                        "role": role,
                        "path": path.name,
                        "sha256": sha256_file(path),
                    }
                    for role, path in formal_inputs
                ],
                "datasets": [entry],
            },
        )
        gate_config_path = root / "gate-config.json"
        _write_json(
            gate_config_path,
            {
                "schema_version": 1,
                "output_root": "gate-runs",
                "version_store_root": "versions",
                "required_dataset_ids": ["stock_history"],
                "allowed_license_values": ["LICENSED_RESEARCH_USE"],
                "maximum_manifest_age_days": 7,
                "gate_code_paths": [gate_code.name],
                "promotion_allowed": False,
                "independent_review_required": True,
            },
        )
        observed_at = (
            pd.Timestamp(calendar[-1])
            .tz_localize("Asia/Shanghai")
            .isoformat()
        )
        with patch(
            "strategy_lab.long_hold_v4.pit_gate_v2._git_head",
            return_value=TEST_COMMIT,
        ):
            gate_paths = run_versioned_pit_gate(
                root,
                gate_config_path.name,
                target_manifest_path.name,
                run_id="gate-pass",
                gate_observed_at=observed_at,
            )

        with (
            patch(
                "strategy_lab.long_hold_v4.formal_walk_forward._git_head",
                return_value=TEST_COMMIT,
            ),
            patch(
                "strategy_lab.long_hold_v4.formal_walk_forward.RUNNER_CODE_PATHS",
                (runner_code.name,),
            ),
        ):
            paths = run_formal_walk_forward(
                root,
                pit_gate_run_directory=gate_paths["run_directory"],
                strategy_config_path=strategy_path.name,
                walk_forward_config_path=walk_path.name,
                run_id="formal-run",
                initial_cash=100_000.0,
                consume_independent_test=True,
            )

        manifest = json.loads(
            paths["manifest"].read_text(encoding="utf-8")
        )
        self_status = manifest["status"]
        assert self_status == "FORMAL_EVALUATION_COMPLETE_REVIEW_REQUIRED"
        assert manifest["promotion_allowed"] is False
        assert manifest["live_trading_allowed"] is False
        assert manifest["validation_robustness"]["passed"] is True
        assert (
            manifest["validation_robustness"]["selection_metric"]
            == "active_sharpe_vs_total_return_benchmark"
        )
        assert len(manifest["window_manifests"]) == (
            len(plan["validation_windows"]) + 1
        )
        candidate_metrics = pd.read_csv(
            paths["run_directory"]
            / "candidate_validation_metrics.csv"
        )
        assert set(
            [
                "validation_score",
                "validation_p_value",
                "deflated_sharpe_probability",
            ]
        ).issubset(candidate_metrics.columns)
        candidate_returns = pd.read_csv(
            paths["run_directory"]
            / "candidate_validation_returns.csv"
        )
        assert {
            "recorded_net_return",
            "selection_net_return",
            "benchmark_total_return",
            "selection_active_return",
        }.issubset(candidate_returns.columns)
        performance = pd.read_csv(
            paths["run_directory"] / "window_performance.csv"
        )
        assert set(performance["additional_slippage_bps"]) == {
            0,
            5,
            10,
            20,
        }
        assert set(performance["benchmark_id"]) == {"000985.CSI"}
        assert (
            root
            / "wf_runs"
            / "holdout_consumption"
            / "independent-test.json"
        ).is_file()
        with (
            patch(
                "strategy_lab.long_hold_v4.formal_walk_forward._git_head",
                return_value=TEST_COMMIT,
            ),
            patch(
                "strategy_lab.long_hold_v4.formal_walk_forward.RUNNER_CODE_PATHS",
                (runner_code.name,),
            ),
        ):
            try:
                run_formal_walk_forward(
                    root,
                    pit_gate_run_directory=gate_paths["run_directory"],
                    strategy_config_path=strategy_path.name,
                    walk_forward_config_path=walk_path.name,
                    run_id="formal-run-second-id",
                    initial_cash=100_000.0,
                    consume_independent_test=True,
                )
            except ContractError as exc:
                assert "independent test has already been consumed" in str(exc)
            else:
                raise AssertionError(
                    "same independent test was consumed under a new run_id"
                )

        verified = verify_formal_run(paths["run_directory"], root)
        assert verified["window_count"] == len(plan["validation_windows"]) + 1
        assert verified["live_trading_allowed"] is False

        selected_path = paths["run_directory"] / "selected_candidate.json"
        selected_path.write_text("{}\n", encoding="utf-8")
        try:
            verify_formal_run(paths["run_directory"], root)
        except ContractError as exc:
            assert "formal run output mismatch" in str(exc)
        else:
            raise AssertionError("tampered formal output passed verification")
