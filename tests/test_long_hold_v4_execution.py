from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy_lab.long_hold_v4.accounting import apply_account_events, mark_to_market, portfolio_risk_state
from strategy_lab.long_hold_v4.core import ContractError, load_config
from strategy_lab.long_hold_v4.execution import apply_fills, normalize_account, write_ledger_view


ROOT = Path(__file__).resolve().parents[1]


def order(
    order_id: str,
    sleeve: str,
    side: str,
    shares: int,
    *,
    signal_date: str = "2026-07-17",
    reference: float = 3000.0,
    core_fraction: float = 0.0,
    sessions: int = 0,
    risk_override_allowed: bool = False,
) -> dict:
    return {
        "order_id": order_id,
        "signal_date": signal_date,
        "valid_through_date": "2026-07-31",
        "asset": "600000",
        "name": "Sample Bank",
        "asset_type": "stock",
        "sector": "bank",
        "sleeve": sleeve,
        "side": side,
        "shares": shares,
        "status": "RESEARCH_INTENT_REPRICE_NEXT_OPEN",
        "risk_override_allowed": risk_override_allowed,
        "full_target_shares_reference": reference,
        "core_fraction_at_signal": core_fraction,
        "t_holding_sessions": sessions,
    }


def fill(
    fill_id: str,
    order_id: str,
    sleeve: str,
    side: str,
    shares: int,
    price: float,
    fill_date: str = "2026-07-18",
    *,
    manual: bool = False,
    manual_reason: str = "",
    risk_override: bool = False,
) -> dict:
    return {
        "fill_id": fill_id,
        "fill_date": fill_date,
        "order_id": order_id,
        "asset": "600000",
        "name": "Sample Bank",
        "asset_type": "stock",
        "sector": "bank",
        "sleeve": sleeve,
        "side": side,
        "shares": shares,
        "price": price,
        "fee_mode": "model",
        "manual_approval": manual,
        "manual_reason": manual_reason,
        "risk_override": risk_override,
    }


def core_account(*, with_t: bool = False) -> dict:
    return {
        "as_of_date": "2026-07-17",
        "cash_cny": 200000.0,
        "holdings": [
            {
                "asset": "600000",
                "name": "Sample Bank",
                "asset_type": "stock",
                "sector": "bank",
                "core_shares": 5000,
                "core_average_cost_cny": 10.0,
                "core_open_date": "2026-06-01",
                "t_shares": 1200 if with_t else 0,
                "t_average_cost_cny": 9.0 if with_t else 0.0,
                "t_open_date": "2026-07-17" if with_t else None,
                "full_target_shares_reference": 6250,
                "realized_pnl_cny": 0.0,
            }
        ],
    }


class LongHoldV4ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "long_hold_v4.json")

    def test_v1_cash_account_migrates_and_negative_realized_pnl_is_valid(self):
        state = normalize_account(
            {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": [], "realized_pnl_cny": -100.0},
            self.config,
        )
        self.assertEqual(state["schema_version"], 2)
        self.assertEqual(state["base_currency"], "CNY")
        self.assertEqual(state["cash_cny"], 500000.0)
        self.assertEqual(state["realized_pnl_cny"], -100.0)

    def test_core_buy_updates_cash_cost_basis_reference_and_is_idempotent(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        orders = pd.DataFrame([order("O1", "core", "buy", 1000)])
        fills = pd.DataFrame([fill("F1", "O1", "core", "buy", 1000, 10.0)])
        state, result = apply_fills(account, fills, self.config, orders)
        holding = state["holdings"][0]
        self.assertEqual(holding["core_shares"], 1000)
        self.assertEqual(holding["full_target_shares_reference"], 3000.0)
        self.assertGreater(holding["core_average_cost_cny"], 10.0)
        self.assertLess(state["cash_cny"], 490000.0)
        self.assertEqual(result.iloc[0]["status"], "applied")

        replayed, replay = apply_fills(state, fills, self.config, orders)
        self.assertEqual(replayed["cash_cny"], state["cash_cny"])
        self.assertEqual(replay.iloc[0]["status"], "duplicate_ignored")

        changed = fills.copy()
        changed.loc[0, "price"] = 11.0
        with self.assertRaisesRegex(ContractError, "payload changed"):
            apply_fills(state, changed, self.config, orders)

    def test_manual_buy_cash_shortfall_oversell_and_unapproved_core_sell_fail_closed(self):
        manual_buy = pd.DataFrame(
            [fill("F0", "MANUAL", "core", "buy", 100, 10.0, manual=True, manual_reason="test")]
        )
        with self.assertRaisesRegex(ContractError, "cannot bypass a buy"):
            apply_fills({"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}, manual_buy, self.config)

        expensive = pd.DataFrame([fill("F1", "O1", "core", "buy", 1000, 10.0)])
        with self.assertRaisesRegex(ContractError, "insufficient cash"):
            apply_fills(
                {"as_of_date": "2026-07-17", "cash_cny": 100.0, "holdings": []},
                expensive,
                self.config,
                pd.DataFrame([order("O1", "core", "buy", 1000)]),
            )

        oversell = pd.DataFrame(
            [fill("F2", "MANUAL", "core", "sell", 6000, 10.0, manual=True, manual_reason="risk review")]
        )
        with self.assertRaisesRegex(ContractError, "sell more core"):
            apply_fills(core_account(), oversell, self.config)

        sell_order = pd.DataFrame([order("O3", "core", "sell", 1000)])
        unapproved = pd.DataFrame([fill("F3", "O3", "core", "sell", 1000, 10.0)])
        with self.assertRaisesRegex(ContractError, "explicit manual approval"):
            apply_fills(core_account(), unapproved, self.config, sell_order)

    def test_t_buy_requires_core_and_cannot_exceed_reference_cap(self):
        t_order = pd.DataFrame([order("T1", "t", "buy", 1200, reference=6250, core_fraction=0.8)])
        t_fill = pd.DataFrame([fill("TF1", "T1", "t", "buy", 1200, 9.0)])
        with self.assertRaisesRegex(ContractError, "configured core fraction"):
            apply_fills({"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}, t_fill, self.config, t_order)

        too_large_order = pd.DataFrame([order("T2", "t", "buy", 1300, reference=6250, core_fraction=0.8)])
        too_large_fill = pd.DataFrame([fill("TF2", "T2", "t", "buy", 1300, 9.0)])
        with self.assertRaisesRegex(ContractError, "sleeve share cap"):
            apply_fills(core_account(), too_large_fill, self.config, too_large_order)

    def test_t_sell_enforces_settlement_sessions_and_preserves_core(self):
        state = normalize_account(core_account(with_t=True), self.config)
        same_day = pd.DataFrame(
            [
                fill(
                    "TS0",
                    "MANUAL",
                    "t",
                    "sell",
                    1200,
                    9.5,
                    "2026-07-17",
                    manual=True,
                    manual_reason="emergency",
                    risk_override=True,
                )
            ]
        )
        with self.assertRaisesRegex(ContractError, "settlement"):
            apply_fills(state, same_day, self.config)

        early_order = pd.DataFrame([order("TS1", "t", "sell", 1200, signal_date="2026-07-18", sessions=1)])
        early_fill = pd.DataFrame([fill("TSF1", "TS1", "t", "sell", 1200, 9.5, "2026-07-19")])
        with self.assertRaisesRegex(ContractError, "minimum holding sessions"):
            apply_fills(state, early_fill, self.config, early_order)

        valid_order = pd.DataFrame([order("TS2", "t", "sell", 1200, signal_date="2026-07-20", sessions=3)])
        valid_fill = pd.DataFrame([fill("TSF2", "TS2", "t", "sell", 1200, 10.0, "2026-07-21")])
        sold, result = apply_fills(state, valid_fill, self.config, valid_order)
        self.assertEqual(sold["holdings"][0]["core_shares"], 5000)
        self.assertEqual(sold["holdings"][0]["t_shares"], 0)
        self.assertGreater(result.iloc[0]["realized_pnl_cny"], 0.0)

    def test_core_cannot_be_reduced_below_t_support_and_empty_ledger_has_headers(self):
        manual_sell = pd.DataFrame(
            [fill("CS1", "MANUAL", "core", "sell", 100, 10.0, manual=True, manual_reason="risk review")]
        )
        with self.assertRaisesRegex(ContractError, "under-support"):
            apply_fills(core_account(with_t=True), manual_sell, self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.csv"
            write_ledger_view(normalize_account({"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}, self.config), path)
            empty = pd.read_csv(path, encoding="utf-8-sig")
            self.assertTrue(empty.empty)
            self.assertIn("fill_sha256", empty.columns)

    def test_dividend_and_later_tax_are_reconciled_and_idempotent(self):
        events = [
            {
                "event_id": "D1",
                "event_date": "2026-07-18",
                "event_type": "cash_dividend",
                "asset": "600000",
                "name": "Sample Bank",
                "source_ref": "broker-statement-001",
                "eligible_shares": 5000,
                "cash_per_share_cny": 0.2,
                "gross_cash_cny": 1000.0,
                "tax_cny": 0.0,
            },
            {
                "event_id": "D2",
                "event_date": "2026-07-19",
                "event_type": "dividend_tax",
                "asset": "600000",
                "name": "Sample Bank",
                "source_ref": "broker-statement-002",
                "tax_cny": 200.0,
            },
        ]
        state, result = apply_account_events(core_account(), events, self.config)
        self.assertEqual(len(result), 2)
        self.assertEqual(state["cash_cny"], 200800.0)
        self.assertEqual(state["gross_dividend_cny"], 1000.0)
        self.assertEqual(state["dividend_tax_cny"], 200.0)
        self.assertEqual(state["net_dividend_cny"], 800.0)
        self.assertEqual(state["holdings"][0]["cumulative_dividend_net_cny"], 800.0)

        replayed, replay = apply_account_events(state, events, self.config)
        self.assertEqual(replayed["cash_cny"], state["cash_cny"])
        self.assertTrue(all(item["status"] == "duplicate_ignored" for item in replay))

        changed = [dict(events[0], gross_cash_cny=999.0)]
        with self.assertRaisesRegex(ContractError, "does not reconcile|payload changed"):
            apply_account_events(state, changed, self.config)

    def test_share_adjustment_preserves_book_cost_and_sleeve_ratio(self):
        account = core_account(with_t=True)
        event = {
            "event_id": "S1",
            "event_date": "2026-07-18",
            "event_type": "share_adjustment",
            "asset": "600000",
            "name": "Sample Bank",
            "source_ref": "broker-statement-split",
            "core_shares_after": 5500,
            "t_shares_after": 1320,
            "full_target_shares_reference_after": 6875,
        }
        state, _ = apply_account_events(account, [event], self.config)
        holding = state["holdings"][0]
        self.assertAlmostEqual(holding["core_average_cost_cny"] * holding["core_shares"], 50000.0)
        self.assertAlmostEqual(holding["t_average_cost_cny"] * holding["t_shares"], 10800.0)
        self.assertEqual(holding["full_target_shares_reference"], 6875.0)

        malformed = dict(event, event_id="S2", event_date="2026-07-19", t_shares_after=1200)
        with self.assertRaisesRegex(ContractError, "same ratio"):
            apply_account_events(account, [malformed], self.config)

    def test_nav_marks_drive_review_and_brake_states(self):
        account = core_account()
        account["cash_cny"] = 100000.0
        account["holdings"][0]["core_average_cost_cny"] = 70.0
        state, first = mark_to_market(account, {"600000": 80.0}, "2026-07-17", self.config)
        self.assertEqual(first["nav_cny"], 500000.0)
        self.assertEqual(first["risk_state"], "NORMAL")

        _, corrected = mark_to_market(state, {"600000": 78.0}, "2026-07-17", self.config)
        self.assertEqual(corrected["nav_cny"], 490000.0)
        self.assertEqual(corrected["peak_nav_cny"], 490000.0)
        self.assertEqual(corrected["drawdown"], 0.0)

        state, review = mark_to_market(state, {"600000": 66.0}, "2026-07-18", self.config)
        self.assertAlmostEqual(review["drawdown"], -0.14)
        self.assertEqual(review["risk_state"], "REVIEW")
        review_rules = portfolio_risk_state(state, review["nav_cny"], self.config)
        self.assertTrue(review_rules["core_add_allowed"])
        self.assertFalse(review_rules["t_buy_allowed"])

        state, brake = mark_to_market(state, {"600000": 58.0}, "2026-07-19", self.config)
        self.assertAlmostEqual(brake["drawdown"], -0.22)
        self.assertEqual(brake["risk_state"], "BRAKE")
        brake_rules = portfolio_risk_state(state, brake["nav_cny"], self.config)
        self.assertFalse(brake_rules["core_add_allowed"])
        self.assertTrue(brake_rules["force_t_exit"])

        with self.assertRaisesRegex(ContractError, "missing a held asset price"):
            mark_to_market(state, {}, "2026-07-20", self.config)


if __name__ == "__main__":
    unittest.main()
