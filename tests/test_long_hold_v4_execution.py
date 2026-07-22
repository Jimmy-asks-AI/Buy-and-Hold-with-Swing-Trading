from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4 import recoverable_transaction
from strategy_lab.long_hold_v4.accounting import apply_account_events, mark_to_market, portfolio_risk_state
from strategy_lab.long_hold_v4.core import ContractError, load_config
from strategy_lab.long_hold_v4.execution import (
    apply_fills,
    commit_execution_transaction,
    config_sha256,
    initialize_persistent_account,
    normalize_account,
    recover_execution_transaction,
    seal_account_state,
    write_ledger_view,
)
from strategy_lab.long_hold_v4.order_envelope import (
    ORDER_COLUMNS,
    empty_order_state_book,
    normalize_order_state_book,
    order_state_record,
    rebind_order_state_account,
    register_order_envelopes,
    seal_order_envelope,
)
from strategy_lab.long_hold_v4.pipeline import load_account


ROOT = Path(__file__).resolve().parents[1]
RUN_MANIFEST_SHA256 = "a" * 64
TRADE_CALENDAR_SHA256 = "b" * 64
RUN_ID = "LHV4-20260717-TEST"


def trading_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2026-07-16",
                "2026-07-17",
                "2026-07-20",
                "2026-07-21",
                "2026-07-22",
                "2026-07-23",
                "2026-07-24",
                "2026-07-27",
                "2026-07-28",
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
                "2026-08-03",
            ]
        }
    )


def fill(
    fill_id: str,
    order_id: str,
    sleeve: str,
    side: str,
    shares: int,
    price: float,
    fill_date: str = "2026-07-20",
    *,
    asset: str = "600000",
    name: str = "Sample Bank",
    sector: str = "bank",
    manual: bool = False,
    manual_reason: str = "",
    risk_override: bool = False,
) -> dict:
    return {
        "fill_id": fill_id,
        "fill_date": fill_date,
        "order_id": order_id,
        "asset": asset,
        "name": name,
        "asset_type": "stock",
        "sector": sector,
        "sleeve": sleeve,
        "side": side,
        "shares": shares,
        "price": price,
        "fee_mode": "model",
        "manual_approval": manual,
        "manual_reason": manual_reason,
        "risk_override": risk_override,
    }


def core_account(*, with_t: bool = False, cash_cny: float = 500000.0) -> dict:
    return {
        "as_of_date": "2026-07-17",
        "cash_cny": cash_cny,
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


def approved_order(
    account: dict,
    config: dict,
    order_id: str,
    sleeve: str,
    side: str,
    shares: int,
    *,
    run_id: str = RUN_ID,
    signal_date: str = "2026-07-17",
    valid_from_date: str = "2026-07-18",
    valid_through_date: str = "2026-07-31",
    asset: str = "600000",
    name: str = "Sample Bank",
    sector: str = "bank",
    indicative_price: float = 10.0,
    reference: float = 3000.0,
    core_fraction: float = 0.0,
    sessions: int = 0,
    risk_state: str = "NORMAL",
    risk_override_allowed: bool = False,
    manual_approval_required: bool = False,
) -> dict:
    state = normalize_account(account, config)
    return seal_order_envelope(
        {
            "order_schema_version": 1,
            "order_id": order_id,
            "run_id": run_id,
            "run_manifest_sha256": RUN_MANIFEST_SHA256,
            "config_sha256": config_sha256(config),
            "trade_calendar_sha256": TRADE_CALENDAR_SHA256,
            "account_version": state["state_version"],
            "account_state_sha256": state["state_sha256"],
            "signal_date": signal_date,
            "valid_from_date": valid_from_date,
            "valid_through_date": valid_through_date,
            "asset": asset,
            "name": name,
            "asset_type": "stock",
            "sector": sector,
            "sleeve": sleeve,
            "side": side,
            "shares": shares,
            "indicative_price": indicative_price,
            "max_price_deviation_bps": config["execution"]["max_price_deviation_bps"],
            "notional": shares * indicative_price,
            "estimated_cost": 1.0,
            "target_core_weight": 0.10,
            "target_t_weight_cap": 0.02,
            "full_target_weight": 0.12,
            "full_target_shares_reference": reference,
            "core_fraction_at_signal": core_fraction,
            "t_holding_sessions": sessions,
            "risk_state_at_signal": risk_state,
            "risk_override_allowed": risk_override_allowed,
            "manual_approval_required": manual_approval_required,
            "status": "ACTIVE",
            "intent_status": "TEST_APPROVAL",
            "reason": "unit_test",
        }
    )


def state_book(account: dict, config: dict, orders: pd.DataFrame, *, run_id: str = RUN_ID) -> dict:
    state = normalize_account(account, config)
    return register_order_envelopes(
        None,
        orders,
        run_id=run_id,
        account_version=state["state_version"],
        account_state_sha256=state["state_sha256"],
        registered_at="2026-07-17T15:01:00",
    )


def execute(
    account: dict,
    fills: pd.DataFrame,
    config: dict,
    orders: pd.DataFrame,
    *,
    book: dict | None = None,
    prices: dict[str, float] | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    active_book = book or state_book(account, config, orders)
    return apply_fills(
        account,
        fills,
        config,
        approved_orders=orders,
        order_state_book=active_book,
        trading_calendar=trading_calendar(),
        valuation_prices=prices or {"600000": 10.0},
        valuation_as_of_date=pd.Timestamp(fills.iloc[0]["fill_date"]) if not fills.empty else "2026-07-20",
        run_manifest_sha256=RUN_MANIFEST_SHA256,
        expected_run_id=str(active_book["current_run_id"] or RUN_ID),
        expected_config_sha256=config_sha256(config),
        trading_calendar_sha256=TRADE_CALENDAR_SHA256,
    )


class LongHoldV4ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "long_hold_v4.json")

    def test_v1_cash_account_migrates_to_hashed_v3(self):
        state = normalize_account(
            {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": [], "realized_pnl_cny": -100.0},
            self.config,
        )
        self.assertEqual(state["schema_version"], 3)
        self.assertEqual(state["state_version"], 0)
        self.assertRegex(state["state_sha256"], r"^[0-9a-f]{64}$")
        tampered = dict(state, cash_cny=499999.0)
        with self.assertRaisesRegex(ContractError, "state hash mismatch"):
            normalize_account(tampered, self.config)
        example = json.loads(
            (ROOT / "portfolio_lab" / "long_hold_v4" / "account.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(normalize_account(example, self.config)["state_sha256"], example["state_sha256"])
        with self.assertRaisesRegex(ContractError, "net_dividend_cny"):
            normalize_account(dict(example, net_dividend_cny=1.0), self.config)
        with self.assertRaisesRegex(ContractError, "fields mismatch"):
            normalize_account(dict(example, hidden_override=True), self.config)

    def test_order_schema_required_fields_match_runtime_contract(self):
        schema = json.loads(
            (ROOT / "data_catalog" / "schemas" / "long_hold_v4_order_envelope_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(ORDER_COLUMNS))
        self.assertFalse(schema["additionalProperties"])

    def test_core_buy_partial_fill_and_replay_are_stateful_and_idempotent(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        order = approved_order(account, self.config, "O1", "core", "buy", 1000)
        orders = pd.DataFrame([order], columns=ORDER_COLUMNS)
        first = pd.DataFrame([fill("F1", "O1", "core", "buy", 400, 10.0)])
        state, result, book = execute(account, first, self.config, orders)
        self.assertEqual(state["holdings"][0]["core_shares"], 400)
        self.assertEqual(state["state_version"], 1)
        self.assertEqual(order_state_record(book, "O1")["status"], "PARTIALLY_FILLED")
        self.assertEqual(result.iloc[0]["status"], "applied")

        replayed, replay, replay_book = execute(state, first, self.config, orders, book=book)
        self.assertEqual(replayed["state_sha256"], state["state_sha256"])
        self.assertEqual(replay.iloc[0]["status"], "duplicate_ignored")
        self.assertEqual(replay_book["book_sha256"], book["book_sha256"])

        second = pd.DataFrame([fill("F2", "O1", "core", "buy", 600, 10.0, "2026-07-21")])
        finished, _, finished_book = execute(state, second, self.config, orders, book=book)
        self.assertEqual(finished["holdings"][0]["core_shares"], 1000)
        self.assertEqual(order_state_record(finished_book, "O1")["status"], "FILLED")

        changed = first.copy()
        changed.loc[0, "price"] = 10.01
        with self.assertRaisesRegex(ContractError, "payload changed"):
            execute(state, changed, self.config, orders, book=book)

    def test_every_envelope_field_is_hash_authenticated(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        original = approved_order(account, self.config, "AUTH", "core", "buy", 1000)
        original_orders = pd.DataFrame([original], columns=ORDER_COLUMNS)
        book = state_book(account, self.config, original_orders)
        batch = pd.DataFrame([fill("AUTH-F", "AUTH", "core", "buy", 1000, 10.0)])
        for field in [name for name in ORDER_COLUMNS if name not in {"order_id", "order_sha256"}]:
            tampered = dict(original)
            value = tampered[field]
            if isinstance(value, bool):
                tampered[field] = not value
            elif isinstance(value, int):
                tampered[field] = value + 1
            elif isinstance(value, float):
                tampered[field] = value + 0.01
            else:
                tampered[field] = f"{value}X"
            with self.subTest(field=field), self.assertRaises(ContractError):
                execute(
                    account,
                    batch,
                    self.config,
                    pd.DataFrame([tampered], columns=ORDER_COLUMNS),
                    book=book,
                )

        resigned = dict(original, shares=900, notional=9000.0)
        resigned = seal_order_envelope(resigned)
        with self.assertRaisesRegex(ContractError, "order state hash"):
            execute(account, batch, self.config, pd.DataFrame([resigned]), book=book)

    def test_expired_and_superseded_orders_cannot_fill_or_stack_stages(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        duplicate_a = approved_order(account, self.config, "DUP-A", "core", "buy", 500)
        duplicate_b = approved_order(account, self.config, "DUP-B", "core", "buy", 500)
        with self.assertRaisesRegex(ContractError, "multiple active orders"):
            state_book(
                account,
                self.config,
                pd.DataFrame([duplicate_a, duplicate_b], columns=ORDER_COLUMNS),
            )
        old = approved_order(account, self.config, "STAGE-1", "core", "buy", 1000, run_id="RUN-1")
        old_frame = pd.DataFrame([old], columns=ORDER_COLUMNS)
        book = state_book(account, self.config, old_frame, run_id="RUN-1")
        new = approved_order(account, self.config, "STAGE-2", "core", "buy", 1000, run_id="RUN-2")
        new_frame = pd.DataFrame([new], columns=ORDER_COLUMNS)
        state = normalize_account(account, self.config)
        book = register_order_envelopes(
            book,
            new_frame,
            run_id="RUN-2",
            account_version=state["state_version"],
            account_state_sha256=state["state_sha256"],
            registered_at="2026-07-18T15:01:00",
        )
        combined = pd.DataFrame([old, new], columns=ORDER_COLUMNS)
        with self.assertRaisesRegex(ContractError, "SUPERSEDED"):
            execute(
                account,
                pd.DataFrame([fill("OLD-F", "STAGE-1", "core", "buy", 1000, 10.0)]),
                self.config,
                combined,
                book=book,
            )
        state, _, book = execute(
            account,
            pd.DataFrame([fill("NEW-F", "STAGE-2", "core", "buy", 1000, 10.0)]),
            self.config,
            combined,
            book=book,
        )
        self.assertEqual(state["holdings"][0]["core_shares"], 1000)

        expired = approved_order(
            account,
            self.config,
            "EXPIRED",
            "core",
            "buy",
            1000,
            valid_through_date="2026-07-20",
        )
        with self.assertRaisesRegex(ContractError, "EXPIRED"):
            execute(
                account,
                pd.DataFrame([fill("EXP-F", "EXPIRED", "core", "buy", 1000, 10.0, "2026-08-03")]),
                self.config,
                pd.DataFrame([expired], columns=ORDER_COLUMNS),
            )

    def test_manual_fill_cannot_bypass_envelope_and_price_deviation_is_bounded(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        empty = pd.DataFrame(columns=ORDER_COLUMNS)
        with self.assertRaisesRegex(ContractError, "no approved order"):
            execute(
                account,
                pd.DataFrame([fill("M1", "MANUAL", "core", "buy", 100, 10.0, manual=True, manual_reason="test")]),
                self.config,
                empty,
                book=empty_order_state_book(
                    normalize_account(account, self.config)["state_version"],
                    normalize_account(account, self.config)["state_sha256"],
                ),
            )
        order = approved_order(account, self.config, "PRICE", "core", "buy", 1000)
        with self.assertRaisesRegex(ContractError, "deviation"):
            execute(
                account,
                pd.DataFrame([fill("P1", "PRICE", "core", "buy", 1000, 10.11)]),
                self.config,
                pd.DataFrame([order], columns=ORDER_COLUMNS),
            )

    def test_t_rules_use_execution_calendar_not_order_csv_sessions(self):
        account = core_account(with_t=True)
        strict = copy.deepcopy(self.config)
        strict["t_strategy"]["minimum_holding_days"] = 3
        early = approved_order(
            account,
            strict,
            "T-EARLY",
            "t",
            "sell",
            1200,
            reference=6250,
            core_fraction=0.8,
            sessions=999,
        )
        with self.assertRaisesRegex(ContractError, "minimum holding sessions"):
            execute(
                account,
                pd.DataFrame([fill("T-EARLY-F", "T-EARLY", "t", "sell", 1200, 10.0)]),
                strict,
                pd.DataFrame([early], columns=ORDER_COLUMNS),
            )
        valid = approved_order(
            account,
            strict,
            "T-VALID",
            "t",
            "sell",
            1200,
            reference=6250,
            core_fraction=0.8,
            sessions=0,
        )
        sold, result, _ = execute(
            account,
            pd.DataFrame([fill("T-VALID-F", "T-VALID", "t", "sell", 1200, 10.0, "2026-07-21")]),
            strict,
            pd.DataFrame([valid], columns=ORDER_COLUMNS),
        )
        self.assertEqual(sold["holdings"][0]["t_shares"], 0)
        self.assertEqual(result.iloc[0]["t_holding_sessions"], 3)

        same_day = approved_order(
            account,
            self.config,
            "T-SAME",
            "t",
            "sell",
            1200,
            signal_date="2026-07-16",
            valid_from_date="2026-07-17",
            reference=6250,
            core_fraction=0.8,
            risk_override_allowed=True,
            manual_approval_required=True,
        )
        with self.assertRaisesRegex(ContractError, "settlement"):
            execute(
                account,
                pd.DataFrame(
                    [
                        fill(
                            "T-SAME-F",
                            "T-SAME",
                            "t",
                            "sell",
                            1200,
                            10.0,
                            "2026-07-17",
                            manual=True,
                            manual_reason="emergency",
                            risk_override=True,
                        )
                    ]
                ),
                self.config,
                pd.DataFrame([same_day], columns=ORDER_COLUMNS),
            )

    def test_t_buy_and_core_sell_preserve_core_support(self):
        no_core = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        t_order = approved_order(
            no_core, self.config, "T1", "t", "buy", 1200, indicative_price=9.0, reference=6250, core_fraction=0.8
        )
        with self.assertRaisesRegex(ContractError, "configured core fraction"):
            execute(
                no_core,
                pd.DataFrame([fill("TF1", "T1", "t", "buy", 1200, 9.0)]),
                self.config,
                pd.DataFrame([t_order], columns=ORDER_COLUMNS),
                prices={"600000": 9.0},
            )
        account = core_account()
        too_large = approved_order(
            account, self.config, "T2", "t", "buy", 1300, indicative_price=9.0, reference=6250, core_fraction=0.8
        )
        with self.assertRaisesRegex(ContractError, "sleeve share cap"):
            execute(
                account,
                pd.DataFrame([fill("TF2", "T2", "t", "buy", 1300, 9.0)]),
                self.config,
                pd.DataFrame([too_large], columns=ORDER_COLUMNS),
                prices={"600000": 9.0},
            )
        supported = core_account(with_t=True)
        core_sell = approved_order(
            supported,
            self.config,
            "CS1",
            "core",
            "sell",
            100,
            reference=6250,
            core_fraction=0.8,
            manual_approval_required=True,
        )
        with self.assertRaisesRegex(ContractError, "under-support"):
            execute(
                supported,
                pd.DataFrame([fill("CSF1", "CS1", "core", "sell", 100, 10.0, manual=True, manual_reason="review")]),
                self.config,
                pd.DataFrame([core_sell], columns=ORDER_COLUMNS),
            )

    def test_post_fill_checks_cash_single_asset_sector_and_holding_count(self):
        account = {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}
        huge = approved_order(account, self.config, "CASH", "core", "buy", 46000, reference=46000)
        with self.assertRaisesRegex(ContractError, "minimum cash"):
            execute(
                account,
                pd.DataFrame([fill("CASH-F", "CASH", "core", "buy", 46000, 10.0)]),
                self.config,
                pd.DataFrame([huge], columns=ORDER_COLUMNS),
            )
        concentrated = approved_order(account, self.config, "SINGLE", "core", "buy", 7000, reference=7000)
        with self.assertRaisesRegex(ContractError, "single-asset"):
            execute(
                account,
                pd.DataFrame([fill("SINGLE-F", "SINGLE", "core", "buy", 7000, 10.0)]),
                self.config,
                pd.DataFrame([concentrated], columns=ORDER_COLUMNS),
            )

        sector_account = {
            "as_of_date": "2026-07-17",
            "cash_cny": 500000.0,
            "holdings": [
                {
                    "asset": asset,
                    "name": name,
                    "asset_type": "stock",
                    "sector": "bank",
                    "core_shares": 7000,
                    "core_average_cost_cny": 10.0,
                    "core_open_date": "2026-01-02",
                    "t_shares": 0,
                    "full_target_shares_reference": 7000,
                }
                for asset, name in [("600001", "Bank One"), ("600002", "Bank Two")]
            ],
        }
        sector_order = approved_order(
            sector_account,
            self.config,
            "SECTOR",
            "core",
            "buy",
            6000,
            asset="600003",
            name="Bank Three",
            reference=6000,
        )
        with self.assertRaisesRegex(ContractError, "sector cap"):
            execute(
                sector_account,
                pd.DataFrame([fill("SECTOR-F", "SECTOR", "core", "buy", 6000, 10.0, asset="600003", name="Bank Three")]),
                self.config,
                pd.DataFrame([sector_order], columns=ORDER_COLUMNS),
                prices={"600001": 10.0, "600002": 10.0, "600003": 10.0},
            )

        one_asset_config = copy.deepcopy(self.config)
        one_asset_config["universe"]["maximum_assets"] = 1
        existing = core_account()
        second = approved_order(
            existing,
            one_asset_config,
            "COUNT",
            "core",
            "buy",
            1000,
            asset="600001",
            name="Second Bank",
            reference=1000,
        )
        with self.assertRaisesRegex(ContractError, "holding count"):
            execute(
                existing,
                pd.DataFrame([fill("COUNT-F", "COUNT", "core", "buy", 1000, 10.0, asset="600001", name="Second Bank")]),
                one_asset_config,
                pd.DataFrame([second], columns=ORDER_COLUMNS),
                prices={"600000": 10.0, "600001": 10.0},
            )

        core_cap_config = copy.deepcopy(self.config)
        core_cap_config["portfolio"]["target_core_exposure"] = 0.10
        core_cap_order = approved_order(
            core_account(), core_cap_config, "CORE-CAP", "core", "buy", 1000, reference=6250
        )
        with self.assertRaisesRegex(ContractError, "aggregate core cap"):
            execute(
                core_account(),
                pd.DataFrame([fill("CORE-CAP-F", "CORE-CAP", "core", "buy", 1000, 10.0)]),
                core_cap_config,
                pd.DataFrame([core_cap_order], columns=ORDER_COLUMNS),
            )

        t_cap_config = copy.deepcopy(self.config)
        t_cap_config["t_strategy"]["portfolio_t_weight_cap"] = 0.02
        t_cap_order = approved_order(
            core_account(),
            t_cap_config,
            "T-CAP",
            "t",
            "buy",
            1250,
            reference=6250,
            core_fraction=0.8,
        )
        with self.assertRaisesRegex(ContractError, "aggregate T cap"):
            execute(
                core_account(),
                pd.DataFrame([fill("T-CAP-F", "T-CAP", "t", "buy", 1250, 10.0)]),
                t_cap_config,
                pd.DataFrame([t_cap_order], columns=ORDER_COLUMNS),
            )

    def test_missing_account_hard_blocks_and_initialization_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ContractError, "persistent account is missing"):
                load_account(root / "account.json", self.config, pd.Timestamp("2026-07-17"))
            account_path = root / "account.json"
            ledger_path = root / "fill_ledger.csv"
            state_path = root / "order_state.json"
            account = initialize_persistent_account(
                account_path, ledger_path, state_path, self.config, "2026-07-17"
            )
            self.assertTrue(account_path.is_file())
            self.assertTrue(ledger_path.is_file())
            self.assertTrue(state_path.is_file())
            self.assertEqual(account["cash_cny"], self.config["account"]["initial_cash_cny"])
            with self.assertRaisesRegex(ContractError, "refuses to overwrite"):
                initialize_persistent_account(
                    account_path, ledger_path, state_path, self.config, "2026-07-17"
                )

    def test_account_ledger_and_order_state_recover_after_interrupted_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            account_path = root / "account.json"
            ledger_path = root / "fill_ledger.csv"
            state_path = root / "order_state.json"
            account = initialize_persistent_account(
                account_path, ledger_path, state_path, self.config, "2026-07-17"
            )
            book = normalize_order_state_book(json.loads(state_path.read_text(encoding="utf-8")))
            with self.assertRaisesRegex(ContractError, "fields mismatch"):
                normalize_order_state_book(dict(book, hidden_override=True))
            changed = copy.deepcopy(account)
            changed["cash_cny"] -= 100.0
            changed = seal_account_state(changed, self.config, increment_version=True)
            changed_book = rebind_order_state_account(
                book, changed["state_version"], changed["state_sha256"]
            )
            original_replace = recoverable_transaction._replace_staged
            calls = 0

            def fail_second(staged: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated crash")
                original_replace(staged, destination)

            with patch.object(recoverable_transaction, "_replace_staged", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    commit_execution_transaction(
                        changed, changed_book, account_path, ledger_path, state_path
                    )
            self.assertTrue((root / ".execution_transaction.json").is_file())
            self.assertTrue(recover_execution_transaction(account_path, ledger_path, state_path))
            recovered = normalize_account(json.loads(account_path.read_text(encoding="utf-8")), self.config)
            recovered_book = normalize_order_state_book(json.loads(state_path.read_text(encoding="utf-8")))
            self.assertEqual(recovered["cash_cny"], changed["cash_cny"])
            self.assertEqual(recovered_book["account_state_sha256"], recovered["state_sha256"])
            self.assertFalse((root / ".execution_transaction.json").exists())

    def test_empty_ledger_has_stable_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.csv"
            account = normalize_account(
                {"as_of_date": "2026-07-17", "cash_cny": 500000.0, "holdings": []}, self.config
            )
            write_ledger_view(account, path)
            empty = pd.read_csv(path, encoding="utf-8-sig")
            self.assertTrue(empty.empty)
            self.assertIn("order_sha256", empty.columns)

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
        self.assertEqual(state["cash_cny"], 500800.0)
        self.assertEqual(state["net_dividend_cny"], 800.0)
        self.assertEqual(state["state_version"], 1)
        replayed, replay = apply_account_events(state, events, self.config)
        self.assertEqual(replayed["state_sha256"], state["state_sha256"])
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
        malformed = dict(event, event_id="S2", event_date="2026-07-19", t_shares_after=1200)
        with self.assertRaisesRegex(ContractError, "same ratio"):
            apply_account_events(account, [malformed], self.config)

    def test_nav_marks_drive_review_and_brake_states(self):
        account = core_account(cash_cny=100000.0)
        account["holdings"][0]["core_average_cost_cny"] = 70.0
        state, first = mark_to_market(account, {"600000": 80.0}, "2026-07-17", self.config)
        self.assertEqual(first["nav_cny"], 500000.0)
        _, corrected = mark_to_market(state, {"600000": 78.0}, "2026-07-17", self.config)
        self.assertEqual(corrected["peak_nav_cny"], 490000.0)
        state, review = mark_to_market(state, {"600000": 66.0}, "2026-07-18", self.config)
        self.assertEqual(review["risk_state"], "REVIEW")
        self.assertFalse(portfolio_risk_state(state, review["nav_cny"], self.config)["t_buy_allowed"])
        state, brake = mark_to_market(state, {"600000": 58.0}, "2026-07-19", self.config)
        self.assertEqual(brake["risk_state"], "BRAKE")
        self.assertTrue(portfolio_risk_state(state, brake["nav_cny"], self.config)["force_t_exit"])
        with self.assertRaisesRegex(ContractError, "missing a held asset price"):
            mark_to_market(state, {}, "2026-07-20", self.config)


if __name__ == "__main__":
    unittest.main()
