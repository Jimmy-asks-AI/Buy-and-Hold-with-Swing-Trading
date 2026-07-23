import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4.core import ContractError, load_config
from strategy_lab.long_hold_v4.walk_forward import (
    audit_split_leakage,
    build_bias_audit,
    build_purged_embargoed_plan,
    build_window_status_registry,
    consume_independent_test_once,
    load_walk_forward_config,
    promotion_decision,
    run_audited_window_backtest,
    select_frozen_candidate,
    sha256_file,
    write_window_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


class WorkPackage5WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.walk_forward_config = load_walk_forward_config(
            ROOT / "configs" / "long_hold_v4_work_package_5_walk_forward.json"
        )
        self.strategy_config = load_config(ROOT / "configs" / "long_hold_v4.json")

    def _calendar(self) -> pd.DatetimeIndex:
        return pd.bdate_range("2020-01-01", periods=1300)

    @staticmethod
    def _targets(core_weight: float = 0.10, t_weight: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "signal_date": "2026-01-01",
                    "available_date": "2026-01-01",
                    "asset": "AAA",
                    "sector": "bank",
                    "historical_backtest_allowed": True,
                    "target_core_weight": core_weight,
                    "target_t_weight": t_weight,
                    "target_semantics": "FULL_SNAPSHOT",
                    "target_schema_version": 2,
                    "snapshot_asset_count": 1,
                }
            ]
        )

    @staticmethod
    def _execution_prices() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-01-02",
                    "asset": "AAA",
                    "asset_type": "stock",
                    "open": 10.0,
                    "close": 10.0,
                    "return_basis": "qfq_adjusted",
                    "price_basis": "qfq_adjusted",
                    "available_date": "2026-01-02",
                    "list_date": "2020-01-01",
                    "delist_date": None,
                    "has_market_data": True,
                    "is_suspended": False,
                    "is_limit_up": True,
                    "is_limit_down": False,
                    "is_delisted": False,
                },
                {
                    "date": "2026-01-05",
                    "asset": "AAA",
                    "asset_type": "stock",
                    "open": 20.0,
                    "close": 20.0,
                    "return_basis": "qfq_adjusted",
                    "price_basis": "qfq_adjusted",
                    "available_date": "2026-01-05",
                    "list_date": "2020-01-01",
                    "delist_date": None,
                    "has_market_data": True,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_delisted": False,
                },
                {
                    "date": "2026-01-06",
                    "asset": "AAA",
                    "asset_type": "stock",
                    "open": 20.0,
                    "close": 22.0,
                    "return_basis": "qfq_adjusted",
                    "price_basis": "qfq_adjusted",
                    "available_date": "2026-01-06",
                    "list_date": "2020-01-01",
                    "delist_date": None,
                    "has_market_data": True,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_delisted": False,
                },
            ]
        )

    def test_purge_and_embargo_regions_have_no_sample_leakage(self):
        calendar = self._calendar()
        plan = build_purged_embargoed_plan(
            calendar, self.walk_forward_config
        )
        audit = audit_split_leakage(calendar, plan)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["failures"])
        self.assertGreaterEqual(len(plan["validation_windows"]), 2)
        label_horizon = plan["label_horizon_sessions"]
        for fold in plan["validation_windows"]:
            self.assertLess(
                fold["_train_end_index"] + label_horizon,
                fold["_validation_start_index"],
            )
        latest_validation_end = max(
            fold["_validation_end_index"]
            for fold in plan["validation_windows"]
        )
        self.assertLess(
            latest_validation_end + label_horizon,
            plan["independent_test"]["_start_index"],
        )
        self.assertFalse(plan["promotion_allowed"])

    def test_independent_test_data_cannot_participate_in_tuning(self):
        registry = pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "parameters_json": '{"lookback": 20}',
                    "train_score": 1.0,
                    "validation_score": 0.5,
                    "validation_p_value": 0.01,
                    "split_roles_used": "train+independent_test",
                }
            ]
        )
        with self.assertRaisesRegex(ContractError, "cannot participate in tuning"):
            select_frozen_candidate(registry, self.walk_forward_config)

    def test_independent_test_can_only_be_consumed_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            ledger = Path(tmp) / "holdout.json"
            payload = consume_independent_test_once(
                ledger,
                run_id="wf-001",
                plan_sha256="a" * 64,
                data_manifest_sha256="b" * 64,
            )
            self.assertTrue(payload["consumed"])
            self.assertFalse(payload["tuning_use_allowed"])
            with self.assertRaisesRegex(
                ContractError, "already been consumed"
            ):
                consume_independent_test_once(
                    ledger,
                    run_id="wf-002",
                    plan_sha256="a" * 64,
                    data_manifest_sha256="b" * 64,
                )

    def test_unfilled_order_attempt_does_not_generate_return(self):
        config = copy.deepcopy(self.strategy_config)
        for key in config["costs"]:
            config["costs"][key] = 0.0
        result = run_audited_window_backtest(
            self._execution_prices(),
            self._targets(),
            config,
            self.walk_forward_config,
            initial_cash=100.0,
        )
        orders = result["orders"]
        nav = result["nav"].set_index("date")
        self.assertEqual(
            orders.loc[orders["status"].eq("UNFILLED"), "reason"].tolist(),
            ["limit_up_conservative_no_trade"],
        )
        self.assertEqual(
            pd.Timestamp(
                orders.loc[orders["status"].eq("FILLED"), "attempt_date"].iloc[0]
            ),
            pd.Timestamp("2026-01-05"),
        )
        self.assertAlmostEqual(nav.loc[pd.Timestamp("2026-01-02"), "nav"], 100.0)
        self.assertAlmostEqual(nav.loc[pd.Timestamp("2026-01-05"), "nav"], 100.0)
        self.assertGreater(nav.loc[pd.Timestamp("2026-01-06"), "nav"], 100.0)

    def test_t_sleeve_gross_cost_and_net_gain_reconcile_independently(self):
        result = run_audited_window_backtest(
            self._execution_prices(),
            self._targets(core_weight=0.10, t_weight=0.02),
            self.strategy_config,
            self.walk_forward_config,
            initial_cash=100000.0,
        )
        final = result["attribution"].iloc[-1]
        self.assertAlmostEqual(
            final["t_net_gain"],
            final["t_gross_pnl"] - final["cumulative_t_trading_cost"],
        )
        self.assertAlmostEqual(
            final["core_plus_t_nav"], result["nav"].iloc[-1]["nav"]
        )
        self.assertAlmostEqual(final["nav_reconciliation_difference"], 0.0)
        self.assertEqual(
            result["cost_scenarios"]["additional_slippage_bps"].tolist(),
            [5, 10, 20],
        )
        self.assertTrue(
            (
                result["cost_scenarios"]["t_net_gain"].diff().dropna() < 0
            ).all()
        )

    def test_bias_audit_covers_survivorship_lookahead_tuning_and_multiple_tests(self):
        passed = build_bias_audit(
            historical_universe_verified=True,
            available_dates_verified=True,
            tuning_split_roles=["train", "validation"],
            independent_test_access_count=1,
            registered_candidate_count=4,
            maximum_candidate_count=24,
            multiple_testing_correction="holm",
            adjusted_p_values_verified=True,
            candidate_registry_frozen_before_holdout=True,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(
            set(passed["checks"]),
            {
                "survivorship_bias",
                "lookahead",
                "repeated_tuning",
                "multiple_testing",
                "multiple_testing_applied",
                "candidate_registry_frozen",
            },
        )
        self.assertFalse(passed["promotion_allowed"])

    def test_window_bundle_saves_required_artifacts_and_stays_unpromoted(self):
        result = run_audited_window_backtest(
            self._execution_prices(),
            self._targets(),
            self.strategy_config,
            self.walk_forward_config,
            initial_cash=100000.0,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            fixture_root = Path(tmp)
            code_path = fixture_root / "generator.py"
            config_path = fixture_root / "config.json"
            formal_path = fixture_root / "formal.csv"
            code_path.write_text("# fixture\n", encoding="utf-8")
            config_path.write_text("{}\n", encoding="utf-8")
            formal_path.write_text("value\n1\n", encoding="utf-8")
            code_binding = {
                "path": code_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(code_path),
            }
            config_binding = {
                "role": "walk_forward_config",
                "path": config_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(config_path),
            }
            formal_binding = {
                "role": "target_weights",
                "path": formal_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(formal_path),
            }
            context = {
                "pit_gate_run_id": "pit-001",
                "pit_gate_manifest_sha256": "a" * 64,
                "target_manifest_sha256": "b" * 64,
                "code_commit": "c" * 40,
                "code_files": [code_binding],
                "config_bindings": [config_binding],
                "formal_input_bindings": [formal_binding],
                "training_parameters": {"lookback": 20},
                "data_manifest": result["input_hashes"],
                "cost_assumptions": {
                    "recorded_costs": True,
                    "additional_slippage_bps": [5, 10, 20],
                },
            }
            with patch(
                "strategy_lab.long_hold_v4.walk_forward.verify_pit_gate_binding",
                return_value={
                    "pit_gate_run_id": "pit-001",
                    "pit_gate_manifest_sha256": "a" * 64,
                    "target_manifest_sha256": "b" * 64,
                },
            ):
                paths = write_window_bundle(
                    fixture_root,
                    project_root=ROOT,
                    pit_gate_run_directory=fixture_root / "pit",
                    run_id="wf-001",
                    window_id="validation-01",
                    split_role="validation",
                    context=context,
                    artifacts=result,
                )
            manifest = json.loads(
                paths["manifest"].read_text(encoding="utf-8")
            )
            output_names = {item["path"] for item in manifest["outputs"]}
            self.assertTrue(
                {
                    "target_weights.csv",
                    "orders.csv",
                    "fills.csv",
                    "pending_targets.csv",
                    "account.csv",
                    "nav.csv",
                    "attribution.csv",
                    "cost_scenarios.csv",
                    "risk_exposures.csv",
                }.issubset(output_names)
            )
            self.assertFalse(manifest["promotion_allowed"])
            self.assertFalse(manifest["manual_review_signed"])

    def test_holm_correction_is_applied_before_candidate_selection(self):
        registry = pd.DataFrame(
            [
                {
                    "candidate_id": "strong",
                    "parameters_json": '{"lookback": 20}',
                    "train_score": 0.5,
                    "validation_score": 0.6,
                    "validation_p_value": 0.01,
                    "split_roles_used": "train+validation",
                },
                {
                    "candidate_id": "weak",
                    "parameters_json": '{"lookback": 60}',
                    "train_score": 1.0,
                    "validation_score": 1.0,
                    "validation_p_value": 0.20,
                    "split_roles_used": "train+validation",
                },
            ]
        )
        selected = select_frozen_candidate(
            registry, self.walk_forward_config
        )
        self.assertEqual(selected["candidate_id"], "strong")
        self.assertAlmostEqual(
            selected["adjusted_validation_p_value"], 0.02
        )
        self.assertEqual(selected["surviving_candidate_count"], 1)

    def test_never_filled_target_remains_in_order_audit(self):
        prices = self._execution_prices().copy()
        prices["is_limit_up"] = True
        result = run_audited_window_backtest(
            prices,
            self._targets(),
            self.strategy_config,
            self.walk_forward_config,
            initial_cash=100.0,
        )
        self.assertTrue(result["fills"].empty)
        self.assertEqual(len(result["pending_targets"]), 1)
        self.assertTrue(
            result["orders"]["status"].eq("PENDING_UNFILLED").all()
        )
        self.assertAlmostEqual(result["nav"].iloc[-1]["nav"], 100.0)

    def test_all_windows_can_be_reported_blocked_when_pit_data_is_missing(self):
        plan = build_purged_embargoed_plan(
            self._calendar(), self.walk_forward_config
        )
        registry = build_window_status_registry(
            plan, blocked_reason="PIT_GATE_BLOCKED"
        )
        self.assertTrue(registry["status"].eq("BLOCKED_NOT_RUN").all())
        self.assertTrue(registry["reason"].eq("PIT_GATE_BLOCKED").all())
        self.assertFalse(registry["promotion_allowed"].any())

    def test_promotion_allowed_defaults_and_remains_false(self):
        default = promotion_decision()
        complete = promotion_decision(
            pit_gate_passed=True,
            all_windows_completed=True,
            independent_test_completed=True,
            cost_adjusted_results_explained=True,
            failed_windows_disclosed=True,
            manual_review_signed=True,
        )
        self.assertFalse(default["promotion_allowed"])
        self.assertFalse(default["automatic_promotion_allowed"])
        self.assertFalse(complete["promotion_allowed"])
        self.assertEqual(
            complete["promotion_blocking_reasons"],
            ["manual_promotion_action_required"],
        )


if __name__ == "__main__":
    unittest.main()
