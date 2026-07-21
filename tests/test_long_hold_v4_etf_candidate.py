from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_total_return_candidate_validator as candidate_validator
from strategy_lab.long_hold_v4.pit_etf_total_return_candidate_builder import (
    build_event_usage,
    load_official_dividend_events,
)
from strategy_lab.long_hold_v4.pit_etf_total_return_candidate_validator import (
    align_cash_events,
    total_return_identity,
)


class ETFOfficialEventCandidateTests(unittest.TestCase):
    def test_formal_dividend_registry_preserves_official_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            event_path = directory / "etf_dividend_events.csv"
            manifest_path = directory / "manifest.json"
            producer_path = Path(__file__).resolve()
            pd.DataFrame(
                [
                    {
                        "asset": "512390",
                        "announcement_date": "2024-06-18",
                        "record_date": "2024-06-21",
                        "ex_date": "2024-06-24",
                        "pay_date": "2024-06-28",
                        "cash_per_share": 0.18182,
                        "available_date": "2024-06-18",
                        "data_source": "official_fixture",
                        "source_vintage": "official_fixture:v1",
                    }
                ]
            ).to_csv(event_path, index=False, encoding="utf-8-sig")
            manifest = {
                "qualification_status": "PROMOTED_FULL_UNIVERSE_OFFICIAL_EVENTS",
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
                "current_final_snapshot": False,
                "rows": 1,
                "assets": 1,
                "source_vintage_set_sha256": "a" * 64,
                "code_path": str(producer_path),
                "code_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
                "outputs": [
                    {
                        "role": "pit_etf_dividend_events",
                        "path": str(event_path),
                        "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
                        "rows": 1,
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            events, _ = load_official_dividend_events(
                event_path,
                manifest_path,
                as_of="2026-07-17",
            )
            selected = events[
                events["asset"].eq("512390")
                & events["ex_date"].eq(pd.Timestamp("2024-06-24"))
            ]
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected.iloc[0]["cash_per_share"]), 0.18182, places=8)
            self.assertTrue(events["available_date"].eq(events["announcement_date"]).all())

    def test_event_usage_separates_scheduled_and_terminal_cash(self):
        ordinary = pd.DataFrame(
            [
                {
                    "asset": "510880",
                    "announcement_date": pd.Timestamp("2026-07-15"),
                    "record_date": pd.Timestamp("2026-07-18"),
                    "ex_date": pd.Timestamp("2026-07-20"),
                    "pay_date": pd.Timestamp("2026-07-22"),
                    "cash_per_share": 0.01,
                    "available_date": pd.Timestamp("2026-07-15"),
                    "data_source": "official",
                    "source_vintage": "official:v1",
                }
            ]
        )
        terminal = pd.DataFrame(
            [
                {
                    "asset": "511210",
                    "event_type": "liquidation_distribution",
                    "announcement_date": pd.Timestamp("2018-01-09"),
                    "record_date": pd.Timestamp("2018-01-16"),
                    "ex_date": pd.Timestamp("2018-01-17"),
                    "pay_date": pd.Timestamp("2018-01-23"),
                    "cash_per_share": 112.79,
                    "termination_date": pd.Timestamp("2018-01-26"),
                    "extinguishes_position": True,
                    "available_date": pd.Timestamp("2018-01-09"),
                    "source_vintage": "official_terminal:v1",
                    "historical_backtest_allowed": True,
                    "model_promotion_allowed": False,
                }
            ]
        )
        usage = build_event_usage(ordinary, terminal, pd.Timestamp("2026-07-17"))
        regular = usage[usage["asset"].eq("510880")].iloc[0]
        liquidation = usage[usage["asset"].eq("511210")].iloc[0]
        self.assertFalse(bool(regular["event_effective_by_cutoff"]))
        self.assertFalse(bool(regular["applied_to_price_adjustment"]))
        self.assertTrue(bool(liquidation["event_effective_by_cutoff"]))
        self.assertFalse(bool(liquidation["applied_to_price_adjustment"]))
        self.assertTrue(bool(liquidation["applied_to_cash_ledger"]))
        self.assertTrue(bool(liquidation["extinguishes_position"]))

    def test_cash_event_alignment_uses_next_real_market_date(self):
        prices = pd.DataFrame(
            [
                {"date": "2026-07-17", "asset": "510880", "source_cash_distribution": 0.0},
                {"date": "2026-07-20", "asset": "510880", "source_cash_distribution": 0.01},
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "asset": "510880",
                    "event_type": "cash_distribution",
                    "ex_date": "2026-07-19",
                    "cash_per_share": 0.01,
                    "applied_to_price_adjustment": True,
                },
                {
                    "asset": "511210",
                    "event_type": "liquidation_distribution",
                    "ex_date": "2018-01-17",
                    "cash_per_share": 112.79,
                    "applied_to_price_adjustment": False,
                },
            ]
        )
        checks = align_cash_events(prices, events)
        self.assertEqual(len(checks), 1)
        self.assertEqual(pd.Timestamp(checks.iloc[0]["aligned_price_date"]), pd.Timestamp("2026-07-20"))
        self.assertEqual(int(checks.iloc[0]["maximum_calendar_lag_days"]), 1)
        self.assertTrue(bool(checks.iloc[0]["passed"]))

    def test_total_return_identity_handles_cash_and_share_conversion(self):
        prices = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "asset": "510880",
                    "close": 5.0,
                    "raw_close": 10.0,
                    "cash_distribution": 0.0,
                    "share_adjustment_factor": 0.5,
                },
                {
                    "date": "2026-01-02",
                    "asset": "510880",
                    "close": 5.0,
                    "raw_close": 5.0,
                    "cash_distribution": 0.0,
                    "share_adjustment_factor": 1.0,
                },
                {
                    "date": "2026-01-03",
                    "asset": "510880",
                    "close": 5.0,
                    "raw_close": 4.0,
                    "cash_distribution": 1.0,
                    "share_adjustment_factor": 1.0,
                },
            ]
        )
        summary = total_return_identity(prices)
        self.assertEqual(int(summary.iloc[0]["failed_identity_rows"]), 0)
        self.assertAlmostEqual(float(summary.iloc[0]["maximum_identity_absolute_error"]), 0.0)

    def test_terminal_cash_ledger_validator_accepts_only_final_extinguishment(self):
        base = {
            "asset": "159522",
            "event_type": "liquidation_distribution",
            "holder_scope": "all_registered_holders",
            "available_trade_date": "2024-09-24",
            "accounting_date": "2024-09-25",
            "is_final_distribution": False,
            "additional_distribution_expected": True,
            "extinguishes_position": False,
            "event_effective_by_cutoff": True,
            "applied_to_price_adjustment": False,
            "applied_to_cash_ledger": True,
            "source_pdf_sha256_set": "a" * 64,
        }
        events = pd.DataFrame(
            [
                {**base, "event_id": "interim", "distribution_sequence": 1},
                {
                    **base,
                    "event_id": "final",
                    "distribution_sequence": 2,
                    "available_trade_date": "2024-11-17",
                    "accounting_date": "2024-11-21",
                    "is_final_distribution": True,
                    "additional_distribution_expected": False,
                    "extinguishes_position": True,
                },
            ]
        )
        result = candidate_validator.validate_terminal_cash_ledger(
            events,
            pd.Timestamp("2024-11-21"),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["extinguishing_rows"], 1)

        malformed = events.copy()
        malformed.loc[0, "extinguishes_position"] = True
        failed = candidate_validator.validate_terminal_cash_ledger(
            malformed,
            pd.Timestamp("2024-11-21"),
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(any(item.startswith("extinguishment_") for item in failed["failed_groups"]))


if __name__ == "__main__":
    unittest.main()


def test_lifecycle_authentication_uses_immutable_manifest_and_archived_code(tmp_path, monkeypatch):
    output = tmp_path / "legacy.csv.gz"
    output.write_bytes(b"legacy-output")
    latest = tmp_path / "latest.json"
    immutable = tmp_path / "immutable.json"
    payload = {
        "run_id": "fixture",
        "immutable_manifest_path": immutable.name,
        "historical_backtest_allowed": False,
        "code_files": [{"path": "collector.py", "sha256": "a" * 64}],
        "outputs": [
            {
                "role": "etf_total_return_prices_observation",
                "path": output.name,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ],
    }
    encoded = json.dumps(payload, sort_keys=True)
    latest.write_text(encoded, encoding="utf-8")
    immutable.write_text(encoded, encoding="utf-8")
    calls = []
    monkeypatch.setattr(candidate_validator, "ROOT", tmp_path)
    monkeypatch.setattr(candidate_validator, "LIFECYCLE_MANIFEST_PATH", latest)
    monkeypatch.setattr(
        candidate_validator,
        "authenticate_current_or_archive",
        lambda path, digest: calls.append((path.name, digest)) or path,
    )
    manifest, authenticated_output = candidate_validator._authenticate_lifecycle()
    assert manifest["run_id"] == "fixture"
    assert authenticated_output == output
    assert calls == [("collector.py", "a" * 64)]
