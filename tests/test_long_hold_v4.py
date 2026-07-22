from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy_lab.a_share_index_data_harvester import ROOT as INDEX_HARVESTER_ROOT
from strategy_lab.a_share_industry_index_harvester import ROOT as INDUSTRY_HARVESTER_ROOT

from strategy_lab.long_hold_v4.backtest import run_weight_backtest, validate_backtest_inputs
from strategy_lab.long_hold_v4.historical_diagnostic import _load_watchlist, _month_ends, _rate_as_of
from strategy_lab.long_hold_v4.core import (
    ContractError,
    allocate_core_targets,
    compute_price_features,
    entry_decision,
    estimate_trade_cost,
    load_config,
    score_universe,
    t_decision,
    validate_config,
)
from strategy_lab.long_hold_v4.stock_snapshot_builder import (
    _annual_financials,
    _dividend_metrics,
    _valuation_metrics,
    current_valuation_metrics_from_observation,
    finalize_snapshot,
)
from strategy_lab.long_hold_v4.pipeline import load_snapshot, plan_orders, run_current
from strategy_lab.long_hold_v4.execution import normalize_account
from strategy_lab.long_hold_v4.order_envelope import normalize_order_state_book, verify_order_frame


ROOT = Path(__file__).resolve().parents[1]


def prepare_runtime_state(root: Path, config: dict) -> None:
    config["data"].setdefault("order_state_path", "order_state.json")
    config["data"].setdefault("trade_calendar_path", "trade_calendar.csv")
    account_path = root / config["data"]["account_path"]
    if not account_path.exists():
        account_path.parent.mkdir(parents=True, exist_ok=True)
        account_path.write_text(
            json.dumps({"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}),
            encoding="utf-8",
        )
    calendar_path = root / config["data"]["trade_calendar_path"]
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.bdate_range("2026-01-01", "2026-07-31")}).to_csv(
        calendar_path, index=False, encoding="utf-8-sig"
    )


def bank_row(**overrides):
    row = {
        "as_of_date": "2026-07-17",
        "available_date": "2026-07-17",
        "asset": "600000",
        "name": "Sample Bank",
        "asset_type": "stock",
        "sector": "bank",
        "is_tradeable": True,
        "is_st": False,
        "history_years": 20,
        "positive_profit_years_5y": 5,
        "dividend_years_5y": 5,
        "dividend_yield": 0.06,
        "dividend_cagr_5y": 0.06,
        "dividend_cut_count_5y": 0,
        "payout_ratio": 0.45,
        "roe_mean_5y": 0.14,
        "roe_std_5y": 0.02,
        "revenue_cagr_5y": 0.05,
        "profit_cagr_5y": 0.07,
        "profit_cv_5y": 0.18,
        "current_pe": 6.0,
        "current_pb": 0.7,
        "pe_percentile_5y": 0.10,
        "pb_percentile_5y": 0.12,
        "sector_pe_percentile": 0.20,
        "sector_pb_percentile": 0.20,
        "yield_spread_cn10y": 0.035,
        "annual_vol_3y": 0.18,
        "max_drawdown_3y": -0.28,
        "npl_ratio": 0.012,
        "provision_coverage": 2.50,
        "core_tier1_ratio": 0.11,
    }
    row.update(overrides)
    return row


def latest_price(**overrides):
    row = {
        "date": pd.Timestamp("2026-07-17"),
        "close": 9.0,
        "ma20": 8.8,
        "ma60": 8.5,
        "drawdown_3y": -0.25,
        "stabilized": True,
        "falling_knife": False,
        "range_regime": True,
        "t_buy_setup": True,
        "t_exit_setup": False,
        "expected_reversion_edge": 0.012,
    }
    row.update(overrides)
    return pd.Series(row)


class LongHoldV4Tests(unittest.TestCase):
    def test_industry_harvester_default_root_is_repository_root(self):
        self.assertEqual(INDUSTRY_HARVESTER_ROOT.resolve(), ROOT.resolve())
        self.assertEqual(INDEX_HARVESTER_ROOT.resolve(), ROOT.resolve())

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "long_hold_v4.json")

    def test_config_rejects_semantically_invalid_risk_and_t_parameters(self):
        mutations = [
            (lambda cfg: cfg["entry"].__setitem__("pe_percentile_5y_max", 1.2), "pe_percentile"),
            (lambda cfg: cfg["entry"].__setitem__("tranche_fractions", [-0.2, 0.6, 1.0]), "tranche_fractions"),
            (lambda cfg: cfg["t_strategy"].__setitem__("minimum_holding_days", 30), "maximum_holding_days"),
            (lambda cfg: cfg["account"].__setitem__("allow_margin", True), "cash-only"),
            (lambda cfg: cfg["portfolio"].__setitem__("max_single_etf_weight", 0.40), "max_sector_weight"),
            (lambda cfg: cfg["execution"].__setitem__("max_price_deviation_bps", -1.0), "max_price_deviation"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                invalid = copy.deepcopy(self.config)
                mutate(invalid)
                with self.assertRaisesRegex(ContractError, message):
                    validate_config(invalid)

    def test_performance_evidence_registry_never_promotes_unvalidated_legacy_results(self):
        registry = pd.read_csv(ROOT / "data_catalog" / "performance_evidence_registry.csv")
        required = {"system_family", "evidence_status", "performance_use_allowed", "model_promotion_allowed"}
        self.assertTrue(required.issubset(registry.columns))
        self.assertFalse(registry["performance_use_allowed"].astype(bool).any())
        self.assertFalse(registry["model_promotion_allowed"].astype(bool).any())
        legacy = registry["evidence_status"].eq("legacy_invalid_performance")
        self.assertGreaterEqual(int(legacy.sum()), 2)

    def test_future_available_date_is_blocked(self):
        scored = score_universe(pd.DataFrame([bank_row(available_date="2026-07-18")]), "2026-07-17", self.config)
        self.assertEqual(scored.loc[0, "data_gate_status"], "blocked")
        self.assertFalse(bool(scored.loc[0, "durable_eligible"]))

    def test_duplicate_latest_rows_are_rejected(self):
        with self.assertRaises(ContractError):
            score_universe(pd.DataFrame([bank_row(), bank_row()]), "2026-07-17", self.config)

    def test_snapshot_loader_preserves_leading_zero_codes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshot.csv"
            pd.DataFrame([{"asset": "002807", "name": "Bank"}]).to_csv(path, index=False, encoding="utf-8-sig")
            self.assertEqual(load_snapshot(path).loc[0, "asset"], "002807")

    def test_value_trap_veto_blocks_weak_bank(self):
        scored = score_universe(pd.DataFrame([bank_row(npl_ratio=0.04)]), "2026-07-17", self.config)
        self.assertTrue(bool(scored.loc[0, "hard_veto"]))
        self.assertIn("npl_ratio_too_high", scored.loc[0, "hard_veto_reasons"])

    def test_nonpositive_current_pe_is_a_value_trap_veto(self):
        scored = score_universe(pd.DataFrame([bank_row(current_pe=-6.0)]), "2026-07-17", self.config)
        self.assertTrue(bool(scored.loc[0, "hard_veto"]))
        self.assertIn("nonpositive_current_pe", scored.loc[0, "hard_veto_reasons"])

    def test_non_numeric_required_value_fails_closed(self):
        scored = score_universe(pd.DataFrame([bank_row(npl_ratio="NOT_A_NUMBER")]), "2026-07-17", self.config)
        self.assertEqual(scored.loc[0, "data_gate_status"], "blocked")
        self.assertIn("invalid_numeric_fields=npl_ratio", scored.loc[0, "data_gate_reasons"])
        self.assertFalse(bool(scored.loc[0, "durable_eligible"]))

    def test_insurance_uses_nbv_instead_of_accounting_revenue_growth(self):
        row = bank_row(
            asset="601318",
            name="Sample Insurer",
            sector="insurance",
            revenue_cagr_5y=-0.03,
            solvency_ratio=1.90,
            new_business_value_cagr_3y=0.08,
        )
        scored = score_universe(pd.DataFrame([row]), "2026-07-17", self.config)
        self.assertNotIn("negative_revenue_growth", scored.loc[0, "hard_veto_reasons"])
        self.assertTrue(bool(scored.loc[0, "durable_eligible"]))

    def test_durable_deep_value_starts_first_tranche(self):
        scored = score_universe(pd.DataFrame([bank_row()]), "2026-07-17", self.config).iloc[0]
        decision = entry_decision(scored, latest_price(), 0.0, "2026-07-17", self.config)
        self.assertEqual(decision["entry_action"], "BUILD_1")
        self.assertAlmostEqual(decision["target_core_fraction"], 0.30)

    def test_falling_knife_keeps_cash(self):
        scored = score_universe(pd.DataFrame([bank_row()]), "2026-07-17", self.config).iloc[0]
        decision = entry_decision(
            scored,
            latest_price(stabilized=False, falling_knife=True),
            0.0,
            "2026-07-17",
            self.config,
        )
        self.assertEqual(decision["entry_action"], "WAIT_STABILIZATION")

    def test_missing_timing_feature_keeps_cash(self):
        scored = score_universe(pd.DataFrame([bank_row()]), "2026-07-17", self.config).iloc[0]
        decision = entry_decision(scored, latest_price(ma20=float("nan")), 0.0, "2026-07-17", self.config)
        self.assertEqual(decision["entry_action"], "KEEP_CASH")
        self.assertIn("missing_price_features=ma20", decision["entry_reasons"])

    def test_existing_core_is_held_when_there_is_no_new_entry_edge(self):
        scored = score_universe(pd.DataFrame([bank_row(pe_percentile_5y=0.80, pb_percentile_5y=0.80)]), "2026-07-17", self.config).iloc[0]
        decision = entry_decision(scored, latest_price(drawdown_3y=-0.05), 0.60, "2026-07-17", self.config)
        self.assertEqual(decision["entry_action"], "HOLD_CORE")
        self.assertAlmostEqual(decision["target_core_fraction"], 0.60)

    def test_existing_core_requires_review_after_hard_veto(self):
        scored = score_universe(pd.DataFrame([bank_row(npl_ratio=0.04)]), "2026-07-17", self.config).iloc[0]
        decision = entry_decision(scored, latest_price(), 0.60, "2026-07-17", self.config)
        self.assertEqual(decision["entry_action"], "REVIEW_CORE")
        self.assertAlmostEqual(decision["target_core_fraction"], 0.60)

    def test_stock_and_etf_costs_match_account_terms(self):
        stock_buy = estimate_trade_cost(100000.0, "buy", "stock", self.config)
        stock_sell = estimate_trade_cost(100000.0, "sell", "stock", self.config)
        etf_buy = estimate_trade_cost(100000.0, "buy", "etf", self.config)
        self.assertAlmostEqual(stock_buy["commission"], 8.0)
        self.assertAlmostEqual(stock_buy["total_cost"], 29.0)
        self.assertAlmostEqual(stock_sell["total_cost"], 79.0)
        self.assertAlmostEqual(etf_buy["total_cost"], 25.0)

    def test_non_positive_price_is_rejected(self):
        prices = pd.DataFrame(
            [("2026-07-17", 10.0, 10.0, 0.0, 10.0)],
            columns=["date", "open", "high", "low", "close"],
        )
        with self.assertRaisesRegex(ContractError, "non-positive"):
            compute_price_features(prices, self.config)

    def test_single_candidate_is_not_renormalized_past_cap(self):
        frame = pd.DataFrame(
            [
                {
                    **bank_row(),
                    "final_score": 80.0,
                    "entry_action": "BUILD_1",
                    "target_core_fraction": 0.30,
                }
            ]
        )
        out = allocate_core_targets(frame, self.config)
        self.assertLessEqual(out.loc[0, "full_target_weight"], 0.12 + 1e-12)
        self.assertAlmostEqual(out.loc[0, "target_core_weight"], 0.03)
        self.assertLessEqual(out.loc[0, "target_core_weight"] + out.loc[0, "target_t_weight_cap"], 0.12 + 1e-12)

    def test_sector_and_portfolio_t_caps_are_enforced_before_orders(self):
        concentrated = pd.DataFrame(
            [
                {
                    "asset": f"B{index}",
                    "asset_type": "stock",
                    "sector": "bank",
                    "final_score": 90.0 - index,
                    "annual_vol_3y": 0.15,
                    "entry_action": "BUILD_3",
                    "target_core_fraction": 1.0,
                }
                for index in range(3)
            ]
        )
        concentrated_out = allocate_core_targets(concentrated, self.config)
        self.assertLessEqual(concentrated_out["full_target_weight"].sum(), 0.30 + 1e-9)
        self.assertLessEqual(
            (concentrated_out["target_core_weight"] + concentrated_out["target_t_weight_cap"]).sum(),
            0.30 + 1e-9,
        )
        self.assertLessEqual(
            (concentrated_out["target_core_weight"] + concentrated_out["target_t_weight_cap"]).max(),
            0.12 + 1e-9,
        )

        diversified = pd.DataFrame(
            [
                {
                    "asset": f"A{index}",
                    "asset_type": "stock",
                    "sector": ["bank", "insurance", "utility"][index % 3],
                    "final_score": 90.0 - index,
                    "annual_vol_3y": 0.15 + index / 1000.0,
                    "entry_action": "BUILD_2",
                    "target_core_fraction": 0.60,
                }
                for index in range(9)
            ]
        )
        out = allocate_core_targets(diversified, self.config)
        self.assertLessEqual(out["target_t_weight_cap"].sum(), 0.10 + 1e-9)
        combined = out.assign(weight=out["target_core_weight"] + out["target_t_weight_cap"])
        self.assertLessEqual(combined.groupby("sector")["weight"].sum().max(), 0.30 + 1e-9)
        self.assertGreaterEqual(float(out["target_cash_weight"].iloc[0]), 0.10 - 1e-9)

    def test_existing_t_sleeve_reduces_sector_room_for_core_additions(self):
        frame = pd.DataFrame(
            [
                {
                    **bank_row(asset="600000"),
                    "final_score": 90.0,
                    "entry_action": "HOLD_CORE",
                    "target_core_fraction": 1.0,
                    "current_core_weight": 0.10,
                    "current_t_weight": 0.10,
                },
                {
                    **bank_row(asset="600001"),
                    "final_score": 85.0,
                    "entry_action": "BUILD_3",
                    "target_core_fraction": 1.0,
                    "current_core_weight": 0.0,
                    "current_t_weight": 0.0,
                },
            ]
        )
        out = allocate_core_targets(frame, self.config)
        combined = out["target_core_weight"] + out["target_t_weight_cap"]
        self.assertLessEqual(float(combined.sum()), 0.30 + 1e-9)
        self.assertLessEqual(float(out.loc[out["asset"].eq("600001"), "target_core_weight"].iloc[0]), 0.10 + 1e-9)

    def test_allocator_preserves_existing_core_when_new_entry_edge_disappears(self):
        frame = pd.DataFrame(
            [
                {
                    **bank_row(),
                    "final_score": 80.0,
                    "entry_action": "HOLD_CORE",
                    "target_core_fraction": 0.60,
                    "current_core_weight": 0.08,
                    "current_t_weight": 0.01,
                }
            ]
        )
        out = allocate_core_targets(frame, self.config)
        self.assertAlmostEqual(out.loc[0, "target_core_weight"], 0.08)
        self.assertGreaterEqual(out.loc[0, "target_t_weight_cap"], 0.01)
        self.assertFalse(bool(out.loc[0, "manual_risk_review_required"]))

    def test_allocator_stops_new_risk_when_current_account_breaches_a_cap(self):
        frame = pd.DataFrame(
            [
                {
                    **bank_row(asset="600000"),
                    "final_score": 85.0,
                    "entry_action": "HOLD_CORE",
                    "target_core_fraction": 1.0,
                    "current_core_weight": 0.13,
                    "current_t_weight": 0.0,
                },
                {
                    **bank_row(asset="600001", sector="utility"),
                    "final_score": 80.0,
                    "entry_action": "BUILD_1",
                    "target_core_fraction": 0.30,
                    "current_core_weight": 0.0,
                    "current_t_weight": 0.0,
                },
            ]
        )
        out = allocate_core_targets(frame, self.config).set_index("asset")
        self.assertTrue(bool(out.loc["600000", "manual_risk_review_required"]))
        self.assertIn("current_single_asset_cap_breach", out.loc["600000", "portfolio_risk_reasons"])
        self.assertAlmostEqual(out.loc["600001", "target_core_weight"], 0.0)

    def test_order_planner_preserves_cash_and_global_t_budget(self):
        nav = 900000.0
        targets = pd.DataFrame(
            [
                {
                    "asset": f"600{index:03d}",
                    "name": f"Stock {index}",
                    "asset_type": "stock",
                    "sector": "bank",
                    "final_score": 90.0 - index,
                    "entry_action": "HOLD_CORE",
                    "target_core_weight": 50000.0 / nav,
                    "target_t_weight_cap": 12500.0 / nav,
                    "full_target_weight": 62500.0 / nav,
                    "t_action": "BUY_T_NEXT_OPEN",
                    "t_reasons": "test",
                    "t_holding_sessions": 0,
                }
                for index in range(8)
            ]
        )
        account = {
            "as_of_date": "2026-07-17",
            "cash_cny": 500000.0,
            "holdings": [
                {
                    "asset": f"600{index:03d}",
                    "name": f"Stock {index}",
                    "asset_type": "stock",
                    "sector": "bank",
                    "core_shares": 5000,
                    "core_average_cost_cny": 10.0,
                    "core_open_date": "2026-01-02",
                    "t_shares": 0,
                    "t_average_cost_cny": 0.0,
                    "t_open_date": None,
                    "full_target_shares_reference": 6250,
                }
                for index in range(8)
            ],
        }
        account = normalize_account(account, self.config)
        prices = {f"600{index:03d}": 10.0 for index in range(8)}
        orders = plan_orders(
            targets,
            account,
            prices,
            self.config,
            "2026-07-17",
            run_id="TEST-RUN",
            run_manifest_sha256="a" * 64,
            trade_calendar_sha256="b" * 64,
            risk_state_at_signal="NORMAL",
        )
        buys = orders[orders["side"] == "buy"]
        t_notional = float(buys.loc[buys["sleeve"] == "t", "notional"].sum())
        projected_cash = 500000.0 - float((buys["notional"] + buys["estimated_cost"]).sum())
        self.assertGreater(t_notional, 0.0)
        self.assertLessEqual(t_notional, nav * 0.10 + 1e-9)
        self.assertGreaterEqual(projected_cash, nav * 0.10 - 1e-6)

    def test_t_requires_established_core(self):
        scored = score_universe(pd.DataFrame([bank_row()]), "2026-07-17", self.config).iloc[0]
        blocked = t_decision(scored, latest_price(), 0.30, 0.0, 0, "2026-07-17", self.config)
        allowed = t_decision(scored, latest_price(), 1.00, 0.0, 0, "2026-07-17", self.config)
        self.assertEqual(blocked["t_action"], "NO_T")
        self.assertEqual(allowed["t_action"], "BUY_T_NEXT_OPEN")

    def test_invalidated_durable_core_exits_existing_t_sleeve(self):
        scored = score_universe(pd.DataFrame([bank_row(npl_ratio=0.04)]), "2026-07-17", self.config).iloc[0]
        decision = t_decision(scored, latest_price(), 0.80, 0.20, 1, "2026-07-17", self.config)
        self.assertEqual(decision["t_action"], "SELL_T_NEXT_OPEN")
        self.assertEqual(decision["t_reasons"], "durable_core_invalidated")

    def test_backtest_executes_close_signal_at_next_open(self):
        config = copy.deepcopy(self.config)
        config["portfolio"]["max_single_stock_weight"] = 0.50
        for key in ["stock_commission_rate", "etf_commission_rate", "stock_sell_stamp_duty_rate", "stock_transfer_fee_rate", "etf_stamp_duty_rate", "slippage_bps_each_side"]:
            config["costs"][key] = 0.0
        prices = pd.DataFrame(
            [
                ("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted"),
                ("2026-01-06", "AAA", "stock", 10.0, 11.0, "qfq_adjusted"),
                ("2026-01-07", "AAA", "stock", 11.0, 11.0, "qfq_adjusted"),
            ],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [("2026-01-05", "AAA", 0.50, 0.0)],
            columns=["signal_date", "asset", "target_core_weight", "target_t_weight"],
        )
        result = run_weight_backtest(prices, targets, config, initial_cash=100.0)
        self.assertAlmostEqual(result["nav"].iloc[0]["nav"], 100.0)
        self.assertAlmostEqual(result["nav"].iloc[1]["nav"], 105.0)
        self.assertEqual(pd.Timestamp(result["trades"].iloc[0]["execution_date"]), pd.Timestamp("2026-01-06"))

    def test_backtest_rejects_single_asset_cap_violation(self):
        prices = pd.DataFrame(
            [("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted")],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [("2026-01-04", "AAA", 0.90, 0.0)],
            columns=["signal_date", "asset", "target_core_weight", "target_t_weight"],
        )
        with self.assertRaisesRegex(ContractError, "single-asset cap"):
            validate_backtest_inputs(prices, targets, self.config)

    def test_backtest_rejects_missing_price_while_asset_is_held(self):
        config = copy.deepcopy(self.config)
        config["portfolio"]["max_single_stock_weight"] = 0.50
        prices = pd.DataFrame(
            [
                ("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted"),
                ("2026-01-06", "AAA", "stock", 10.0, 10.0, "qfq_adjusted"),
                ("2026-01-07", "BBB", "stock", 20.0, 20.0, "qfq_adjusted"),
            ],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [("2026-01-05", "AAA", 0.10, 0.0)],
            columns=["signal_date", "asset", "target_core_weight", "target_t_weight"],
        )
        with self.assertRaisesRegex(ContractError, "held assets missing"):
            run_weight_backtest(prices, targets, config, initial_cash=100.0)

    def test_backtest_input_contract_does_not_self_promote(self):
        prices = pd.DataFrame(
            [
                ("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted"),
                ("2026-01-06", "AAA", "stock", 10.0, 10.1, "qfq_adjusted"),
            ],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [
                {
                    "signal_date": "2026-01-05",
                    "available_date": "2026-01-05",
                    "asset": "AAA",
                    "sector": "bank",
                    "historical_backtest_allowed": True,
                    "target_core_weight": 0.10,
                    "target_t_weight": 0.0,
                }
            ]
        )
        result = run_weight_backtest(prices, targets, self.config, initial_cash=100.0)
        self.assertTrue(result["metrics"]["input_contract_ready"])
        self.assertFalse(result["metrics"]["promotion_allowed"])
        self.assertIn("walk_forward_validation_not_supplied", result["metrics"]["promotion_blocking_reasons"])

    def test_price_proxy_can_run_diagnostics_but_not_promote(self):
        prices = pd.DataFrame(
            [
                ("2026-01-05", "IDX", "etf", 10.0, 10.0, "price_index_proxy"),
                ("2026-01-06", "IDX", "etf", 10.0, 10.2, "price_index_proxy"),
            ],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [("2026-01-05", "IDX", 0.20, 0.0)],
            columns=["signal_date", "asset", "target_core_weight", "target_t_weight"],
        )
        result = run_weight_backtest(prices, targets, self.config, initial_cash=100.0)
        self.assertFalse(result["metrics"]["promotion_allowed"])

    def test_overweight_targets_are_rejected(self):
        prices = pd.DataFrame(
            [("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted")],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        targets = pd.DataFrame(
            [("2026-01-04", "AAA", 0.90, 0.20)],
            columns=["signal_date", "asset", "target_core_weight", "target_t_weight"],
        )
        with self.assertRaises(ContractError):
            validate_backtest_inputs(prices, targets)

    def test_backtest_rejects_combined_asset_cap_and_orphan_t_target(self):
        prices = pd.DataFrame(
            [("2026-01-05", "AAA", "stock", 10.0, 10.0, "qfq_adjusted")],
            columns=["date", "asset", "asset_type", "open", "close", "return_basis"],
        )
        combined_breach = pd.DataFrame(
            [("2026-01-04", "AAA", "bank", 0.11, 0.02)],
            columns=["signal_date", "asset", "sector", "target_core_weight", "target_t_weight"],
        )
        with self.assertRaisesRegex(ContractError, "combined core and T"):
            validate_backtest_inputs(prices, combined_breach, self.config)

        orphan_t = pd.DataFrame(
            [("2026-01-04", "AAA", "bank", 0.0, 0.02)],
            columns=["signal_date", "asset", "sector", "target_core_weight", "target_t_weight"],
        )
        with self.assertRaisesRegex(ContractError, "requires an established core"):
            validate_backtest_inputs(prices, orphan_t, self.config)

    def test_missing_investable_price_blocks_system_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = copy.deepcopy(self.config)
            config["data"] = {
                "snapshot_path": "snapshot.csv",
                "price_directory": "prices",
                "account_path": "account.json",
                "agent_contracts_path": "agents.json",
                "output_directory": "outputs",
                "timing_proxy_paths": {},
            }
            (root / "agents.json").write_text(
                (ROOT / "configs" / "long_hold_v4_agent_contracts.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            pd.DataFrame([bank_row(asset="600000"), bank_row(asset="600001")]).to_csv(
                root / "snapshot.csv", index=False, encoding="utf-8-sig"
            )
            (root / "prices").mkdir()
            pd.DataFrame(
                [("2026-07-17", 10.0, 10.0, 10.0, 10.0, "qfq_adjusted")],
                columns=["date", "open", "high", "low", "close", "return_basis"],
            ).to_csv(root / "prices" / "600000.csv", index=False, encoding="utf-8-sig")
            prepare_runtime_state(root, config)
            paths = run_current(root, config, "2026-07-17")
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            candidate_columns = pd.read_csv(paths["candidates"], encoding="utf-8-sig", nrows=0).columns
            empty_orders = pd.read_csv(paths["orders"], encoding="utf-8-sig")
            self.assertFalse(readiness["fresh_investable_prices"])
            self.assertEqual(readiness["system_status"], "CASH_DATA_BLOCKED")
            self.assertEqual(readiness["price_required_asset_count"], 2)
            self.assertIn("600001", {item["asset"] for item in readiness["price_gate_failures"]})
            self.assertIn("price_drawdown_3y", candidate_columns)
            self.assertIn("price_stabilized", candidate_columns)
            self.assertTrue(empty_orders.empty)
            self.assertIn("estimated_cost", empty_orders.columns)

    def test_stale_snapshot_blocks_data_steward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = copy.deepcopy(self.config)
            config["data"] = {
                "snapshot_path": "snapshot.csv",
                "price_directory": "prices",
                "account_path": "account.json",
                "agent_contracts_path": "agents.json",
                "output_directory": "outputs",
                "timing_proxy_paths": {},
            }
            (root / "agents.json").write_text(
                (ROOT / "configs" / "long_hold_v4_agent_contracts.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            pd.DataFrame([bank_row(as_of_date="2026-07-01", available_date="2026-07-01")]).to_csv(
                root / "snapshot.csv", index=False, encoding="utf-8-sig"
            )
            prepare_runtime_state(root, config)
            paths = run_current(root, config, "2026-07-17")
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            agents = pd.read_csv(paths["agents"], encoding="utf-8-sig").set_index("agent")
            self.assertFalse(readiness["snapshot_fresh"])
            self.assertEqual(readiness["system_status"], "CASH_DATA_BLOCKED")
            self.assertEqual(agents.loc["data_steward", "status"], "blocked")

    def test_end_to_end_existing_core_is_held_without_a_new_entry_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = copy.deepcopy(self.config)
            config["data"] = {
                "snapshot_path": "snapshot.csv",
                "price_directory": "prices",
                "account_path": "account.json",
                "agent_contracts_path": "agents.json",
                "output_directory": "outputs",
                "timing_proxy_paths": {},
            }
            (root / "agents.json").write_text(
                (ROOT / "configs" / "long_hold_v4_agent_contracts.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            pd.DataFrame([bank_row(pe_percentile_5y=0.80, pb_percentile_5y=0.80)]).to_csv(
                root / "snapshot.csv", index=False, encoding="utf-8-sig"
            )
            (root / "account.json").write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-17",
                        "cash_cny": 470000.0,
                        "holdings": [
                            {
                                "asset": "600000",
                                "name": "Sample Bank",
                                "asset_type": "stock",
                                "sector": "bank",
                                "core_shares": 3000,
                                "core_average_cost_cny": 9.0,
                                "core_open_date": "2025-01-02",
                                "t_shares": 0,
                                "t_average_cost_cny": 0.0,
                                "t_open_date": None,
                                "full_target_shares_reference": 5000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "prices").mkdir()
            dates = pd.bdate_range(end="2026-07-17", periods=800)
            close = pd.Series(range(len(dates)), dtype=float) / 4000.0 + 9.80
            pd.DataFrame(
                {
                    "date": dates,
                    "open": close,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "return_basis": "qfq_adjusted",
                }
            ).to_csv(root / "prices" / "600000.csv", index=False, encoding="utf-8-sig")
            prepare_runtime_state(root, config)
            paths = run_current(root, config, "2026-07-17")
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            candidates = pd.read_csv(paths["candidates"], encoding="utf-8-sig", dtype={"asset": str})
            orders = pd.read_csv(paths["orders"], encoding="utf-8-sig")
            self.assertEqual(readiness["system_status"], "HOLDINGS_NO_ACTION")
            self.assertTrue(orders.empty)
            self.assertEqual(candidates.loc[0, "entry_action"], "HOLD_CORE")
            self.assertAlmostEqual(candidates.loc[0, "target_core_weight"], candidates.loc[0, "current_core_weight"])

            (root / "account.json").write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-17",
                        "cash_cny": 450000.0,
                        "holdings": [
                            {
                                "asset": "600000",
                                "name": "Sample Bank",
                                "asset_type": "stock",
                                "sector": "bank",
                                "core_shares": 4000,
                                "core_average_cost_cny": 9.0,
                                "core_open_date": "2025-01-02",
                                "t_shares": 1000,
                                "t_average_cost_cny": 9.0,
                                "t_open_date": "2026-06-01",
                                "full_target_shares_reference": 5000,
                            }
                        ],
                        "nav_history": [
                            {
                                "date": "2026-01-02",
                                "nav_cny": 650000.0,
                                "cash_cny": 650000.0,
                                "market_value_cny": 0.0,
                                "peak_nav_cny": 650000.0,
                                "drawdown": 0.0,
                                "risk_state": "NORMAL",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            prepare_runtime_state(root, config)
            brake_paths = run_current(root, config, "2026-07-17")
            brake_orders = pd.read_csv(brake_paths["orders"], encoding="utf-8-sig")
            brake_account = json.loads(brake_paths["account"].read_text(encoding="utf-8"))
            brake_readiness = json.loads(brake_paths["readiness"].read_text(encoding="utf-8"))
            self.assertEqual(brake_account["portfolio_risk_state"], "BRAKE")
            self.assertIn("portfolio_drawdown_brake", brake_readiness["warnings"])
            self.assertFalse((brake_orders["side"] == "buy").any())
            self.assertEqual(set(brake_orders["side"]), {"review", "sell"})
            t_sell = brake_orders[(brake_orders["sleeve"] == "t") & (brake_orders["side"] == "sell")].iloc[0]
            self.assertTrue(bool(t_sell["risk_override_allowed"]))
            verified_orders = verify_order_frame(brake_orders)
            manifest_sha256 = hashlib.sha256(brake_paths["manifest"].read_bytes()).hexdigest()
            self.assertEqual(set(verified_orders["run_manifest_sha256"]), {manifest_sha256})
            order_state = normalize_order_state_book(
                json.loads(brake_paths["order_state"].read_text(encoding="utf-8"))
            )
            self.assertEqual(order_state["current_run_id"], verified_orders.iloc[0]["run_id"])

    def test_held_asset_missing_from_snapshot_blocks_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = copy.deepcopy(self.config)
            config["data"] = {
                "snapshot_path": "snapshot.csv",
                "price_directory": "prices",
                "account_path": "account.json",
                "agent_contracts_path": "agents.json",
                "output_directory": "outputs",
                "timing_proxy_paths": {},
            }
            (root / "agents.json").write_text(
                (ROOT / "configs" / "long_hold_v4_agent_contracts.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            pd.DataFrame([bank_row(asset="600000")]).to_csv(root / "snapshot.csv", index=False, encoding="utf-8-sig")
            (root / "account.json").write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-17",
                        "cash_cny": 490000.0,
                        "holdings": [
                            {
                                "asset": "600001",
                                "name": "Missing Bank",
                                "asset_type": "stock",
                                "sector": "bank",
                                "core_shares": 1000,
                                "core_average_cost_cny": 10.0,
                                "core_open_date": "2026-01-02",
                                "t_shares": 0,
                                "full_target_shares_reference": 2000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "prices").mkdir()
            pd.DataFrame(
                [("2026-07-17", 10.0, 10.0, 10.0, 10.0, "qfq_adjusted")],
                columns=["date", "open", "high", "low", "close", "return_basis"],
            ).to_csv(root / "prices" / "600000.csv", index=False, encoding="utf-8-sig")
            prepare_runtime_state(root, config)
            paths = run_current(root, config, "2026-07-17")
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            agents = pd.read_csv(paths["agents"], encoding="utf-8-sig").set_index("agent")
            self.assertEqual(readiness["system_status"], "PORTFOLIO_DATA_BLOCKED")
            self.assertFalse(readiness["portfolio_valuation_ready"])
            self.assertIn("600001", {item["asset"] for item in readiness["price_gate_failures"]})
            self.assertEqual(agents.loc["portfolio_risk_engineer", "status"], "blocked")
            self.assertEqual(agents.loc["orchestrator", "status"], "blocked")

    def test_historical_rate_uses_available_date_not_observation_date(self):
        rates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
                "available_date": pd.to_datetime(["2026-01-02", "2026-01-04"]),
                "value": [2.0, 3.0],
            }
        )
        value, available_date = _rate_as_of(rates, pd.Timestamp("2026-01-03"))
        self.assertAlmostEqual(value, 0.02)
        self.assertEqual(available_date, pd.Timestamp("2026-01-02"))

    def test_historical_diagnostic_rejects_promotable_current_watchlist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "watchlist.csv"
            pd.DataFrame(
                [
                    {
                        "asset": "000001",
                        "asset_name": "Bank",
                        "sector": "bank",
                        "historical_backtest_allowed": True,
                    }
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "cannot be historical-backtest enabled"):
                _load_watchlist(path, None)

    def test_historical_diagnostic_generates_calendar_month_ends(self):
        dates = _month_ends("2026-01-01", "2026-03-31")
        self.assertEqual([str(date.date()) for date in dates], ["2026-01-31", "2026-02-28", "2026-03-31"])

    def test_financial_builder_uses_notice_date_not_report_date(self):
        data = pd.DataFrame(
            [
                {"REPORT_DATE": "2024-12-31", "REPORT_DATE_NAME": "2024年报", "NOTICE_DATE": "2025-03-20", "PARENTNETPROFIT": 10},
                {"REPORT_DATE": "2025-12-31", "REPORT_DATE_NAME": "2025年报", "NOTICE_DATE": "2026-03-20", "PARENTNETPROFIT": 12},
            ]
        )
        annual = _annual_financials(data, pd.Timestamp("2025-12-31"))
        self.assertEqual(annual["fiscal_year"].tolist(), [2024])

    def test_financial_revision_is_unavailable_before_update_date(self):
        data = pd.DataFrame(
            [
                {
                    "REPORT_DATE": "2023-12-31",
                    "REPORT_DATE_NAME": "2023年报",
                    "NOTICE_DATE": "2024-03-20",
                    "UPDATE_DATE": "2024-03-20",
                },
                {
                    "REPORT_DATE": "2024-12-31",
                    "REPORT_DATE_NAME": "2024年报",
                    "NOTICE_DATE": "2025-03-20",
                    "UPDATE_DATE": "2026-01-10",
                },
            ]
        )
        annual = _annual_financials(data, pd.Timestamp("2025-12-31"))
        self.assertEqual(annual["fiscal_year"].tolist(), [2023])
        self.assertEqual(pd.Timestamp(annual.iloc[-1]["PIT_AVAILABLE_DATE"]), pd.Timestamp("2024-03-20"))

    def test_dividend_builder_aggregates_interim_and_final_by_fiscal_year(self):
        data = pd.DataFrame(
            [
                {"实施方案公告日期": "2026-01-10", "派息比例": 10.0, "派息日": "2026-01-16", "报告时间": "2025半年报"},
                {"实施方案公告日期": "2026-07-04", "派息比例": 10.0, "派息日": "2026-07-10", "报告时间": "2025年报"},
                {"实施方案公告日期": "2025-07-04", "派息比例": 18.0, "派息日": "2025-07-10", "报告时间": "2024年报"},
            ]
        )
        metrics = _dividend_metrics(data, [2024, 2025], pd.Timestamp("2026-07-17"))
        self.assertEqual(metrics["dividend_years_5y"], 2)
        self.assertAlmostEqual(metrics["latest_fiscal_dps"], 2.0)
        self.assertAlmostEqual(metrics["trailing_12m_dps"], 2.0)

    def test_valuation_metrics_do_not_replace_current_loss_with_stale_positive_pe(self):
        data = pd.DataFrame(
            [
                {"date": "2025-01-02", "value": 8.0},
                {"date": "2026-07-17", "value": -6.0},
            ]
        )
        metrics = _valuation_metrics(data, pd.Timestamp("2026-07-17"), "pe")
        self.assertEqual(metrics["current_pe"], -6.0)
        self.assertTrue(pd.isna(metrics["pe_percentile_5y"]))
        self.assertEqual(pd.Timestamp(metrics["pe_date"]), pd.Timestamp("2026-07-17"))

    def test_current_valuation_observation_uses_real_available_date_and_nonpositive_pe(self):
        observation = pd.DataFrame(
            [
                {
                    "date": "2025-01-02",
                    "asset": "600000",
                    "pe_ttm": 8.0,
                    "pb_mrq": 0.8,
                    "available_date": "2026-07-19",
                    "source_observed_at": "2026-07-19T10:00:00+08:00",
                    "historical_backtest_allowed": False,
                    "pit_actionable": False,
                },
                {
                    "date": "2026-07-17",
                    "asset": "600000",
                    "pe_ttm": -6.0,
                    "pb_mrq": 0.7,
                    "available_date": "2026-07-19",
                    "source_observed_at": "2026-07-19T10:00:00+08:00",
                    "historical_backtest_allowed": False,
                    "pit_actionable": False,
                },
            ]
        )
        metrics = current_valuation_metrics_from_observation(
            observation, pd.Timestamp("2026-07-19")
        ).iloc[0]
        self.assertEqual(metrics["overlay_current_pe"], -6.0)
        self.assertTrue(pd.isna(metrics["overlay_pe_percentile_5y"]))
        self.assertEqual(metrics["valuation_available_date"], pd.Timestamp("2026-07-19"))

    def test_sector_percentiles_are_cross_sectional_not_global(self):
        rows = [
            {"sector": "bank", "asset": "A", "current_pe": 5.0, "current_pb": 0.5},
            {"sector": "bank", "asset": "B", "current_pe": 10.0, "current_pb": 1.0},
            {"sector": "utility", "asset": "C", "current_pe": 20.0, "current_pb": 2.0},
        ]
        base = {
            column: None
            for column in [
                "as_of_date", "available_date", "name", "asset_type", "is_tradeable", "is_st", "history_years",
                "positive_profit_years_5y", "dividend_years_5y", "dividend_yield", "dividend_cagr_5y", "dividend_cut_count_5y",
                "payout_ratio", "roe_mean_5y", "roe_std_5y", "revenue_cagr_5y", "profit_cagr_5y", "profit_cv_5y",
                "pe_percentile_5y", "pb_percentile_5y", "china_10y_yield", "yield_spread_cn10y", "annual_vol_3y",
                "max_drawdown_3y", "npl_ratio", "provision_coverage", "core_tier1_ratio", "solvency_ratio",
                "new_business_value_cagr_3y", "debt_to_assets", "interest_coverage", "fcf_dividend_coverage",
                "current_universe_only", "historical_backtest_allowed", "source_note", "source_errors",
            ]
        }
        snapshot = finalize_snapshot([{**base, **row} for row in rows]).set_index("asset")
        self.assertAlmostEqual(snapshot.loc["A", "sector_pe_percentile"], 0.5)
        self.assertAlmostEqual(snapshot.loc["B", "sector_pe_percentile"], 1.0)
        self.assertAlmostEqual(snapshot.loc["C", "sector_pe_percentile"], 1.0)

    def test_sector_pe_percentile_excludes_nonpositive_earnings(self):
        rows = [
            {"sector": "bank", "asset": "A", "current_pe": -5.0, "current_pb": 0.5},
            {"sector": "bank", "asset": "B", "current_pe": 10.0, "current_pb": 1.0},
        ]
        base = {
            column: None
            for column in [
                "as_of_date", "available_date", "name", "asset_type", "is_tradeable", "is_st", "history_years",
                "positive_profit_years_5y", "dividend_years_5y", "dividend_yield", "dividend_cagr_5y", "dividend_cut_count_5y",
                "payout_ratio", "roe_mean_5y", "roe_std_5y", "revenue_cagr_5y", "profit_cagr_5y", "profit_cv_5y",
                "pe_percentile_5y", "pb_percentile_5y", "china_10y_yield", "yield_spread_cn10y", "annual_vol_3y",
                "max_drawdown_3y", "npl_ratio", "provision_coverage", "core_tier1_ratio", "solvency_ratio",
                "new_business_value_cagr_3y", "debt_to_assets", "interest_coverage", "fcf_dividend_coverage",
                "current_universe_only", "historical_backtest_allowed", "source_note", "source_errors",
            ]
        }
        snapshot = finalize_snapshot([{**base, **row} for row in rows]).set_index("asset")
        self.assertTrue(pd.isna(snapshot.loc["A", "sector_pe_percentile"]))
        self.assertEqual(snapshot.loc["B", "sector_pe_percentile"], 1.0)

    def test_agent_framework_check_supports_package_import(self):
        from strategy_lab import agent_framework_check

        self.assertTrue(agent_framework_check.SCHEMA_VERSION)
        self.assertTrue(callable(agent_framework_check.validate_model_run_manifest))


if __name__ == "__main__":
    unittest.main()
