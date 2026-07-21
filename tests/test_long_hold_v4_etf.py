from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab.long_hold_v4.core import load_config, score_universe
from strategy_lab.long_hold_v4.etf_corporate_actions import (
    DEFAULT_REGISTRY_PATH as CORPORATE_ACTION_REGISTRY_PATH,
    conversion_factors_for_asset,
    load_corporate_action_registry,
)
from strategy_lab.long_hold_v4.etf_snapshot_builder import _tracking_error, total_return_adjusted_prices
from strategy_lab.long_hold_v4.etf_index_registry import active_index_map, load_index_registry
from strategy_lab.long_hold_v4.pit_etf_history_observation_builder import (
    QUALIFICATION_STATUS,
    build_dividend_observation,
    build_price_observation,
    process_observation_batch,
)
from strategy_lab.long_hold_v4.pit_etf_index_registry_probe import (
    HistoryInsufficientError,
    ProviderBlockedError,
    archive_latest_probe,
    build_probe_run_id,
    process_entries,
    select_pending_entries,
    validate_history_identity,
)
from strategy_lab.long_hold_v4.pit_etf_total_return_collector import (
    QUALIFICATION_STATUS as LIFECYCLE_QUALIFICATION_STATUS,
    build_lifecycle_observation,
    collapse_lifecycles,
    collect_asset,
    load_terminal_cash_event_registry,
    process_lifecycle_batch,
    provider_circuit_breaker_reason,
    select_lifecycles,
)
from strategy_lab.long_hold_v4 import snapshot_store
from strategy_lab.long_hold_v4.pipeline import verify_source_manifest


ROOT = Path(__file__).resolve().parents[1]


def etf_row(**overrides):
    row = {
        "as_of_date": "2026-07-17",
        "available_date": "2026-07-17",
        "asset": "510880",
        "name": "Dividend ETF",
        "asset_type": "etf",
        "sector": "dividend_index",
        "is_tradeable": True,
        "is_st": False,
        "history_years": 15.0,
        "aum_cny": 2e10,
        "avg_daily_amount_cny": 5e8,
        "expense_ratio": 0.006,
        "tracking_error_1y": 0.01,
        "index_history_years": 15.0,
        "distribution_years_5y": 5,
        "index_dividend_yield": 0.052,
        "index_earnings_cagr_5y": 0.04,
        "pe_percentile_5y": 0.10,
        "yield_spread_cn10y": 0.034,
        "annual_vol_3y": 0.16,
        "max_drawdown_3y": -0.25,
        "total_return_history_ready": True,
        "price_available_date": "2026-07-16",
        "nav_available_date": "2026-07-16",
        "valuation_available_date": "2026-06-23",
        "aum_available_date": "2026-07-17",
        "distribution_available_date": "2026-07-17",
        "expense_available_date": "2026-07-17",
        "index_available_date": "2026-07-17",
        "total_return_available_date": "2026-07-16",
    }
    row.update(overrides)
    return row


class LongHoldV4ETFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "long_hold_v4.json")

    def test_etf_contract_uses_pe_without_inventing_historical_pb(self):
        scored = score_universe(pd.DataFrame([etf_row()]), "2026-07-17", self.config).iloc[0]
        self.assertEqual(scored["data_gate_status"], "pass")
        self.assertTrue(bool(scored["durable_eligible"]))
        self.assertNotIn("pb_percentile_5y", scored["data_gate_reasons"])

    def test_etf_index_registry_activates_only_fully_cached_mappings(self):
        registry = load_index_registry()
        self.assertEqual(len(registry["mappings"]), 17)
        active = active_index_map()
        self.assertEqual(
            set(active),
            {
                "上证红利指数",
                "中证红利指数",
                "中证红利低波动指数",
                "中证中央企业红利指数",
                "沪深300红利低波动指数",
                "中证红利质量指数",
                "上证国有企业红利指数",
                "中证红利价值指数",
            },
        )
        pending = {row["tracking_index_name"]: row for row in registry["mappings"]}
        self.assertEqual(pending["中证中央企业红利指数"]["total_return_code"], "H00825")
        self.assertEqual(pending["中证中央企业红利指数"]["status"], "active")
        self.assertTrue(pending["中证中央企业红利指数"]["local_cache_ready"])
        self.assertIn("中证中央企业红利指数", active)

    def test_etf_index_probe_rejects_a_wrong_return_identity(self):
        dates = pd.bdate_range("2021-01-04", periods=1300)
        frame = pd.DataFrame(
            {
                "date": dates,
                "index_code": "WRONG",
                "index_name": "Wrong Total Return Index",
                "close": np.linspace(1000.0, 1500.0, len(dates)),
            }
        )
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_history_identity(frame, "H00825", "央企红利全收益指数", dates[-1], False)

    def test_etf_index_probe_reports_insufficient_pe_history_with_metrics(self):
        dates = pd.bdate_range("2021-01-04", periods=1300)
        frame = pd.DataFrame(
            {
                "date": dates,
                "index_code": "932039",
                "index_name": "中证国新央企股东回报指数",
                "close": np.linspace(1000.0, 1500.0, len(dates)),
                "pe_ttm": [np.nan] * 400 + [10.0] * 900,
            }
        )
        with self.assertRaises(HistoryInsufficientError) as caught:
            validate_history_identity(frame, "932039", "中证国新央企股东回报指数", dates[-1], True)
        self.assertEqual(caught.exception.metrics["valid_pe_rows"], 900)

        entry = {
            "tracking_index_name": "中证国新央企股东回报指数",
            "price_code": "932039",
            "total_return_code": "932039CNY01",
            "status": "verified_pending_history_probe",
        }

        def insufficient(_item):
            raise caught.exception

        result = process_entries([entry], insufficient)[0]
        self.assertEqual(result["probe_status"], "observation_only_insufficient_history")
        self.assertEqual(result["valid_pe_rows"], 900)

    def test_etf_index_probe_circuit_breaker_defers_remaining_entries(self):
        entries = [
            {
                "tracking_index_name": "First",
                "price_code": "P1",
                "total_return_code": "T1",
                "status": "verified_pending_history_probe",
            },
            {
                "tracking_index_name": "Second",
                "price_code": "P2",
                "total_return_code": "T2",
                "status": "verified_pending_history_probe",
            },
        ]
        calls = []

        def blocked(item):
            calls.append(item["price_code"])
            raise ProviderBlockedError("HTTP 403")

        results = process_entries(entries, blocked)
        self.assertEqual(calls, ["P1"])
        self.assertEqual([row["probe_status"] for row in results], ["provider_blocked", "deferred_provider_blocked"])

    def test_etf_index_probe_can_target_a_later_pending_mapping(self):
        registry = {
            "mappings": [
                {"provider": "csindex", "status": "verified_pending_history_cache", "price_code": "930955"},
                {"provider": "csindex", "status": "verified_pending_history_probe", "price_code": "931468"},
                {"provider": "csindex", "status": "active", "price_code": "000825"},
            ]
        }
        selected = select_pending_entries(registry, limit=1, price_codes=["931468"])
        self.assertEqual([row["price_code"] for row in selected], ["931468"])
        with self.assertRaisesRegex(ValueError, "not pending"):
            select_pending_entries(registry, limit=1, price_codes=["000825"])

    def test_etf_index_probe_archives_immutable_evidence_idempotently(self):
        run_id = build_probe_run_id(
            "2026-07-18T17:26:29+08:00", "2026-07-17", ["932305"], "a" * 64
        )
        self.assertNotIn(":", run_id)
        self.assertIn("932305", run_id)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            output = base / "latest.csv"
            latest_manifest = base / "latest.json"
            run_dir = base / "runs"
            pd.DataFrame(
                [
                    {
                        "price_code": "932305",
                        "probe_status": "observation_only_insufficient_history",
                        "valid_pe_rows": 440,
                    }
                ]
            ).to_csv(output, index=False, encoding="utf-8-sig")
            output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
            latest_manifest.write_text(
                json.dumps(
                    {
                        "created_at": "2026-07-18T17:26:29+08:00",
                        "as_of_date": "2026-07-17",
                        "selected_price_codes": ["932305"],
                        "output_sha256": output_sha,
                    }
                ),
                encoding="utf-8",
            )
            first = archive_latest_probe(output, latest_manifest, run_dir)
            second = archive_latest_probe(output, latest_manifest, run_dir)
            self.assertEqual(first["immutable_run_manifest_path"], second["immutable_run_manifest_path"])
            self.assertTrue((ROOT / first["immutable_result_path"]).exists())
            self.assertTrue((ROOT / first["immutable_run_manifest_path"]).exists())

    def test_csv_round_trip_boolean_numbers_are_valid(self):
        ready = score_universe(
            pd.DataFrame([etf_row(total_return_history_ready=1.0)]), "2026-07-17", self.config
        ).iloc[0]
        self.assertEqual(ready["data_gate_status"], "pass")

        unavailable = score_universe(
            pd.DataFrame([etf_row(total_return_history_ready=0.0)]), "2026-07-17", self.config
        ).iloc[0]
        self.assertEqual(unavailable["data_gate_status"], "pass")
        self.assertIn("missing_total_return_history", unavailable["hard_veto_reasons"])

        invalid = score_universe(
            pd.DataFrame([etf_row(total_return_history_ready=0.5)]), "2026-07-17", self.config
        ).iloc[0]
        self.assertEqual(invalid["data_gate_status"], "blocked")
        self.assertIn("invalid_boolean_fields=total_return_history_ready", invalid["data_gate_reasons"])

    def test_stale_etf_valuation_is_not_hidden_by_a_current_row_date(self):
        scored = score_universe(
            pd.DataFrame([etf_row(valuation_available_date="2026-05-01")]),
            "2026-07-17",
            self.config,
        ).iloc[0]
        self.assertEqual(scored["data_gate_status"], "blocked")
        self.assertIn("stale_valuation_available_date", scored["data_gate_reasons"])

    def test_dividend_adjustment_removes_the_ex_dividend_price_drop(self):
        raw = pd.DataFrame(
            [
                ("2026-01-20", 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0),
                ("2026-01-21", 9.0, 9.0, 9.0, 9.0, 100.0, 900.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        dividends = pd.DataFrame([("2026-01-21", 1.0)], columns=["date", "cumulative_dividend"])
        adjusted = total_return_adjusted_prices(raw, dividends, "510880", pd.Timestamp("2026-01-21"))
        self.assertAlmostEqual(float(adjusted.loc[0, "close"]), 9.0)
        self.assertAlmostEqual(float(adjusted.loc[1, "close"]), 9.0)
        self.assertAlmostEqual(float(adjusted.loc[1, "cash_distribution"]), 1.0)
        self.assertEqual(adjusted.loc[1, "return_basis"], "total_return")

    def test_illiquid_distribution_aligns_to_next_trade_within_one_month(self):
        raw = pd.DataFrame(
            [
                ("2018-01-29", 99.0, 99.0, 99.0, 99.0, 10.0, 990.0),
                ("2018-02-22", 97.7, 97.7, 97.7, 97.7, 10.0, 977.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        dividends = pd.DataFrame([("2018-02-06", 1.3)], columns=["date", "cumulative_dividend"])
        adjusted = total_return_adjusted_prices(raw, dividends, "511230", pd.Timestamp("2018-02-22"))
        self.assertAlmostEqual(float(adjusted.loc[1, "source_cash_distribution"]), 1.3)
        self.assertEqual(adjusted.attrs["distribution_alignments"][0]["calendar_lag_days"], 16)

    def test_distribution_without_trade_within_one_month_fails_closed(self):
        raw = pd.DataFrame(
            [
                ("2018-01-02", 100.0, 100.0, 100.0, 100.0, 10.0, 1000.0),
                ("2018-02-20", 99.0, 99.0, 99.0, 99.0, 10.0, 990.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        dividends = pd.DataFrame([("2018-01-10", 1.0)], columns=["date", "cumulative_dividend"])
        with self.assertRaisesRegex(ValueError, "cannot be aligned"):
            total_return_adjusted_prices(raw, dividends, "511210", pd.Timestamp("2018-02-20"))

    def test_tracking_error_is_zero_for_identical_paths(self):
        dates = pd.bdate_range("2025-01-02", periods=253)
        values = np.cumprod(np.full(len(dates), 1.001))
        nav = pd.DataFrame({"date": dates, "daily_growth_pct": pd.Series(values).pct_change().to_numpy() * 100.0})
        index = pd.DataFrame({"date": dates, "close": values})
        self.assertAlmostEqual(_tracking_error(nav, index, dates[-1]), 0.0, places=12)

    def test_zero_dividend_marker_can_confirm_a_two_for_one_share_conversion(self):
        raw = pd.DataFrame(
            [
                ("2026-01-20", 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0),
                ("2026-01-21", 5.0, 5.0, 5.0, 5.0, 200.0, 1000.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        marker = pd.DataFrame([("2026-01-21", 0.0)], columns=["date", "cumulative_dividend"])
        adjusted = total_return_adjusted_prices(raw, marker, "512890", pd.Timestamp("2026-01-21"))
        self.assertAlmostEqual(float(adjusted.loc[0, "close"]), 5.0)
        self.assertAlmostEqual(float(adjusted.loc[0, "share_adjustment_factor"]), 0.5)
        self.assertEqual(
            adjusted.attrs["applied_share_actions"][0]["evidence_basis"],
            "zero_marker_common_factor_inference",
        )

    def test_zero_dividend_marker_can_confirm_a_five_for_one_share_merger(self):
        raw = pd.DataFrame(
            [
                ("2014-08-28", 0.566, 0.566, 0.566, 0.566, 500.0, 283.0),
                ("2014-09-01", 2.885, 2.885, 2.885, 2.885, 100.0, 288.5),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        markers = pd.DataFrame(
            [("2014-09-01", 0.0)],
            columns=["date", "cumulative_dividend"],
        )
        adjusted = total_return_adjusted_prices(raw, markers, "159901", pd.Timestamp("2014-09-01"))
        self.assertAlmostEqual(float(adjusted.loc[0, "share_adjustment_factor"]), 5.0)
        self.assertLess(abs(float(adjusted["close"].pct_change().iloc[1])), 0.03)

    def test_zero_dividend_marker_can_confirm_an_eight_for_one_share_split(self):
        raw = pd.DataFrame(
            [
                ("2021-06-17", 6.471, 6.471, 6.471, 6.471, 100.0, 647.1),
                ("2021-06-21", 0.798, 0.798, 0.798, 0.798, 800.0, 638.4),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        marker = pd.DataFrame(
            [("2021-06-21", 0.0)],
            columns=["date", "cumulative_dividend"],
        )
        adjusted = total_return_adjusted_prices(raw, marker, "510030", pd.Timestamp("2021-06-21"))
        self.assertAlmostEqual(float(adjusted.loc[0, "share_adjustment_factor"]), 0.125)
        self.assertLess(abs(float(adjusted["close"].pct_change().iloc[1])), 0.03)

    def test_governed_registry_contains_exact_reviewed_conversion_factors(self):
        registry = load_corporate_action_registry(CORPORATE_ACTION_REGISTRY_PATH)
        self.assertEqual(len(registry), 152)
        self.assertEqual(registry["asset"].nunique(), 140)
        expected = {
            "159919": ("2012-12-03", 0.38221954),
            "159943": ("2021-10-22", 11.817461846),
            "159970": ("2025-12-22", 6.0),
            "512100": ("2022-09-05", 0.36555),
            "512200": ("2024-08-12", 0.3580626),
            "516160": ("2024-09-18", 0.30912005),
            "516300": ("2024-10-28", 0.42167573),
            "159845": ("2022-08-02", 0.352275906),
            "560010": ("2022-09-14", 0.35295),
            "563330": ("2024-10-14", 0.33297263),
            "159300": ("2024-06-25", 0.2788902),
            "561380": ("2026-06-25", 2.5),
            "159375": ("2026-06-26", 2.5),
            "159388": ("2026-01-26", 2.5),
            "589100": ("2026-06-24", 2.5),
            "510950": ("2026-04-27", 1.28769271),
            "510100": ("2025-03-10", 0.49719902),
            "510580": ("2024-09-02", 0.24798687),
            "159393": ("2025-03-31", 0.255283787),
            "159558": ("2026-07-09", 3.0),
            "159814": ("2021-02-10", 2.059034965),
            "159845": ("2023-03-01", 0.747644145),
            "159909": ("2021-03-11", 9.795073139),
            "159967": ("2020-11-06", 3.63839094),
            "159991": ("2021-01-27", 2.839985106),
        }
        for asset, (effective_date, expected_factor) in expected.items():
            factors = conversion_factors_for_asset(registry, asset, "2026-07-17")
            self.assertAlmostEqual(factors[pd.Timestamp(effective_date)], expected_factor)

    def test_governed_factor_is_not_available_before_its_evidence_announcement(self):
        registry = load_corporate_action_registry(CORPORATE_ACTION_REGISTRY_PATH)
        self.assertEqual(conversion_factors_for_asset(registry, "159901", "2011-01-01"), {})
        factors = conversion_factors_for_asset(registry, "159901", "2026-07-17")
        self.assertAlmostEqual(factors[pd.Timestamp("2010-11-22")], 5.0)

    def test_governed_factor_can_confirm_a_non_common_share_conversion(self):
        raw = pd.DataFrame(
            [
                ("2012-11-29", 0.813, 0.813, 0.813, 0.813, 100.0, 81.3),
                ("2012-12-03", 2.108, 2.108, 2.108, 2.108, 38.0, 80.1),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        adjusted = total_return_adjusted_prices(
            raw,
            pd.DataFrame(columns=["date", "cumulative_dividend"]),
            "159919",
            pd.Timestamp("2012-12-03"),
            share_conversion_factors={pd.Timestamp("2012-12-03"): 0.38221954},
        )
        self.assertAlmostEqual(
            float(adjusted.loc[0, "share_adjustment_factor"]),
            1.0 / 0.38221954,
        )
        self.assertLess(abs(float(adjusted["close"].pct_change().iloc[1])), 0.03)
        self.assertEqual(
            adjusted.attrs["applied_share_actions"][0]["evidence_basis"],
            "governed_registry",
        )

    def test_governed_factor_must_match_the_observed_price_jump(self):
        raw = pd.DataFrame(
            [
                ("2012-11-29", 0.813, 0.813, 0.813, 0.813, 100.0, 81.3),
                ("2012-12-03", 2.108, 2.108, 2.108, 2.108, 38.0, 80.1),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        with self.assertRaisesRegex(ValueError, "unresolved ETF corporate-action jump"):
            total_return_adjusted_prices(
                raw,
                pd.DataFrame(columns=["date", "cumulative_dividend"]),
                "159919",
                pd.Timestamp("2012-12-03"),
                share_conversion_factors={pd.Timestamp("2012-12-03"): 0.5},
            )

    def test_large_cash_distribution_explains_a_large_raw_price_drop(self):
        raw = pd.DataFrame(
            [
                ("2026-01-20", 1.712, 1.712, 1.712, 1.712, 100.0, 171.2),
                ("2026-01-21", 1.273, 1.273, 1.273, 1.273, 100.0, 127.3),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        dividends = pd.DataFrame([("2026-01-21", 0.437)], columns=["date", "cumulative_dividend"])
        adjusted = total_return_adjusted_prices(raw, dividends, "515100", pd.Timestamp("2026-01-21"))
        expected_return = (1.273 + 0.437) / 1.712 - 1.0
        self.assertAlmostEqual(float(adjusted["close"].pct_change().iloc[1]), expected_return)
        self.assertAlmostEqual(float(adjusted.loc[1, "cash_distribution"]), 0.437)

    def test_share_conversion_restates_prior_cash_distribution(self):
        raw = pd.DataFrame(
            [
                ("2026-01-19", 1.0, 1.0, 1.0, 1.0, 100.0, 100.0),
                ("2026-01-20", 0.95, 0.95, 0.95, 0.95, 100.0, 95.0),
                ("2026-01-21", 0.475, 0.475, 0.475, 0.475, 200.0, 95.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        events = pd.DataFrame(
            [("2026-01-20", 0.05), ("2026-01-21", 0.05)],
            columns=["date", "cumulative_dividend"],
        )
        adjusted = total_return_adjusted_prices(raw, events, "562060", pd.Timestamp("2026-01-21"))
        self.assertAlmostEqual(float(adjusted.loc[1, "source_cash_distribution"]), 0.05)
        self.assertAlmostEqual(float(adjusted.loc[1, "cash_distribution"]), 0.025)
        self.assertAlmostEqual(float(adjusted.loc[1, "share_adjustment_factor"]), 0.5)
        self.assertTrue(np.allclose(adjusted["close"].pct_change().fillna(0.0), 0.0))

    def test_large_unexplained_etf_jump_fails_closed(self):
        raw = pd.DataFrame(
            [
                ("2026-01-20", 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0),
                ("2026-01-21", 5.0, 5.0, 5.0, 5.0, 100.0, 500.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        with self.assertRaisesRegex(ValueError, "unresolved ETF corporate-action jump"):
            total_return_adjusted_prices(raw, pd.DataFrame(), "UNKNOWN", pd.Timestamp("2026-01-21"))

    @staticmethod
    def _observation_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices = pd.DataFrame(
            [
                ("2026-01-20", 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0),
                ("2026-01-21", 9.0, 9.0, 9.0, 9.0, 100.0, 900.0),
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )
        dividends = pd.DataFrame(
            [("2026-01-20", 0.0), ("2026-01-21", 1.0)],
            columns=["date", "cumulative_dividend"],
        )
        nav = pd.DataFrame(
            [("2026-01-20", 1.0, 1.0), ("2026-01-21", 1.0, 1.0)],
            columns=["date", "unit_nav", "cumulative_nav"],
        )
        return prices, dividends, nav

    def test_observation_dividend_events_convert_cumulative_cash(self):
        _, dividends, _ = self._observation_fixture()
        output = build_dividend_observation(
            dividends,
            "510880",
            "Dividend ETF",
            "Dividend Index",
            "2026-01-21",
            "2026-01-22T10:00:00",
            "fixture:v1",
        )
        self.assertEqual(output["cash_distribution"].tolist(), [0.0, 1.0])
        self.assertEqual(output["event_type"].tolist(), ["zero_marker", "cash_distribution"])
        self.assertFalse(output["pit_actionable"].any())
        self.assertEqual(set(output["available_date"].astype(str)), {"2026-01-22"})

    def test_observation_total_return_does_not_double_count_cash(self):
        prices, dividends, nav = self._observation_fixture()
        output, nav_rows = build_price_observation(
            prices,
            dividends,
            nav,
            "510880",
            "Dividend ETF",
            "Dividend Index",
            "SSE",
            "2007-01-18",
            "2026-01-21",
            "2026-01-22T10:00:00",
            "fixture:v1",
        )
        self.assertEqual(nav_rows, 2)
        self.assertAlmostEqual(float(output.loc[1, "period_total_return"]), 0.0)
        self.assertAlmostEqual(float(output.loc[1, "total_return_index"]), 1.0)
        self.assertFalse(output["pit_actionable"].any())
        self.assertEqual(set(output["qualification_status"]), {QUALIFICATION_STATUS})
        self.assertEqual(set(output["available_date"].astype(str)), {"2026-01-22"})

    def test_observation_duplicate_price_dates_fail_closed(self):
        prices, dividends, nav = self._observation_fixture()
        duplicated = pd.concat([prices.iloc[[0]], prices], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            build_price_observation(
                duplicated,
                dividends,
                nav,
                "510880",
                "Dividend ETF",
                "Dividend Index",
                "SSE",
                "2007-01-18",
                "2026-01-21",
                "2026-01-22T10:00:00",
                "fixture:v1",
            )

    def test_observation_batch_quarantines_one_bad_asset(self):
        prices, dividends, nav = self._observation_fixture()
        selected = pd.DataFrame(
            [
                {
                    "asset": "510880",
                    "name": "Good ETF",
                    "tracking_index_name": "Dividend Index",
                    "asset_history_collected": True,
                    "fetched_at": "2026-01-22T10:00:00",
                    "exchange": "SSE",
                    "list_date": "2007-01-18",
                },
                {
                    "asset": "510881",
                    "name": "Bad ETF",
                    "tracking_index_name": "Dividend Index",
                    "asset_history_collected": True,
                    "fetched_at": "2026-01-22T10:00:00",
                    "exchange": "SSE",
                    "list_date": "2007-01-18",
                },
            ]
        )

        def loader(asset):
            asset_prices = prices if asset == "510880" else pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
            return {"price": asset_prices, "dividend": dividends, "nav": nav}

        price_output, _, status = process_observation_batch(selected, loader, "2026-01-21", "fixture:v1")
        self.assertEqual(set(price_output["asset"]), {"510880"})
        self.assertEqual(status.set_index("asset").loc["510881", "status"], "quarantined")
        self.assertFalse(status["historical_backtest_allowed"].any())

    @staticmethod
    def _lifecycle_master_fixture() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "asset_name": "Early Active",
                    "list_date": "2005-02-23",
                    "delist_date": None,
                    "event_type": "listing",
                    "exchange": "SSE",
                    "available_date": "2005-02-23",
                },
                {
                    "asset": "159901",
                    "asset_name": "Later Active",
                    "list_date": "2006-04-24",
                    "delist_date": None,
                    "event_type": "listing",
                    "exchange": "SZSE",
                    "available_date": "2006-04-24",
                },
                {
                    "asset": "159917",
                    "asset_name": "Early Delisted",
                    "list_date": "2012-04-06",
                    "delist_date": None,
                    "event_type": "listing",
                    "exchange": "SZSE",
                    "available_date": "2012-04-06",
                },
                {
                    "asset": "159917",
                    "asset_name": "Early Delisted",
                    "list_date": "2012-04-06",
                    "delist_date": "2015-08-26",
                    "event_type": "delisting",
                    "exchange": "SZSE",
                    "available_date": "2015-08-26",
                },
                {
                    "asset": "510700",
                    "asset_name": "Later Delisted",
                    "list_date": "2013-05-31",
                    "delist_date": None,
                    "event_type": "listing",
                    "exchange": "SSE",
                    "available_date": "2013-05-31",
                },
                {
                    "asset": "510700",
                    "asset_name": "Later Delisted",
                    "list_date": "2013-05-31",
                    "delist_date": "2016-05-12",
                    "event_type": "delisting",
                    "exchange": "SSE",
                    "available_date": "2016-05-12",
                },
            ]
        )

    def test_lifecycle_collapse_does_not_reveal_future_delisting(self):
        master = self._lifecycle_master_fixture()
        before_exit = collapse_lifecycles(master, "2015-01-01").set_index("asset")
        self.assertEqual(before_exit.loc["159917", "lifecycle_status"], "active")
        self.assertTrue(pd.isna(before_exit.loc["159917", "delist_date"]))

        after_exit = collapse_lifecycles(master, "2017-01-01").set_index("asset")
        self.assertEqual(after_exit.loc["159917", "lifecycle_status"], "delisted")
        self.assertEqual(str(after_exit.loc["159917", "delist_date"].date()), "2015-08-26")

    def test_pilot_selection_balances_early_active_and_delisted_assets(self):
        lifecycles = collapse_lifecycles(self._lifecycle_master_fixture(), "2017-01-01")
        selected = select_lifecycles(
            lifecycles,
            mode="pilot",
            earliest_limit=1,
            delisted_limit=1,
        )
        self.assertEqual(selected["asset"].tolist(), ["510050", "159917"])
        self.assertEqual(selected["selection_group"].tolist(), ["active_earliest", "delisted_earliest"])

    @staticmethod
    def _collector_prices(values: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (date, value, value, value, value, 100.0, value * 100.0)
                for date, value in values
            ],
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )

    def test_lifecycle_observation_trims_active_history_to_as_of(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "asset_name": "ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                    "selection_group": "explicit",
                }
            ]
        ).itertuples(index=False).__next__()
        raw = self._collector_prices(
            [("2026-01-20", 10.0), ("2026-01-21", 10.1), ("2026-01-22", 10.2)]
        )
        output, dividends, diagnostics = build_lifecycle_observation(
            raw,
            pd.DataFrame(columns=["date", "cumulative_dividend"]),
            lifecycle,
            "2026-01-21",
            "2026-01-22T10:00:00+08:00",
            "fixture:v1",
        )
        self.assertEqual(output["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-20", "2026-01-21"])
        self.assertTrue(dividends.empty)
        self.assertEqual(diagnostics["listing_start_gap_days"], 0)
        self.assertEqual(set(output["qualification_status"]), {LIFECYCLE_QUALIFICATION_STATUS})
        self.assertFalse(output["historical_backtest_allowed"].any())
        self.assertEqual(set(output["available_date"].astype(str)), {"2026-01-22"})

    def test_lifecycle_observation_rejects_prices_after_delisting(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "159917",
                    "asset_name": "Delisted ETF",
                    "exchange": "SZSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.Timestamp("2026-01-21"),
                    "lifecycle_status": "delisted",
                }
            ]
        ).itertuples(index=False).__next__()
        raw = self._collector_prices(
            [("2026-01-20", 10.0), ("2026-01-21", 10.1), ("2026-01-22", 10.2)]
        )
        with self.assertRaisesRegex(ValueError, "continues after governed delisting"):
            build_lifecycle_observation(
                raw,
                pd.DataFrame(columns=["date", "cumulative_dividend"]),
                lifecycle,
                "2026-01-22",
                "2026-01-22T10:00:00+08:00",
                "fixture:v1",
            )

    def test_lifecycle_observation_removes_exact_weekend_tail_duplicate(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "159927",
                    "asset_name": "Delisted ETF",
                    "exchange": "SZSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.Timestamp("2026-01-23"),
                    "lifecycle_status": "delisted",
                }
            ]
        ).itertuples(index=False).__next__()
        raw = self._collector_prices(
            [("2026-01-20", 10.0), ("2026-01-23", 10.1), ("2026-01-25", 10.1)]
        )
        output, _, diagnostics = build_lifecycle_observation(
            raw,
            pd.DataFrame(columns=["date", "cumulative_dividend"]),
            lifecycle,
            "2026-01-25",
            "2026-01-25T10:00:00+08:00",
            "fixture:v1",
        )
        self.assertEqual(output["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-20", "2026-01-23"])
        self.assertEqual(diagnostics["provider_tail_duplicate_rows_removed"], 1)

    @staticmethod
    def _terminal_cash_event(**overrides) -> dict[str, object]:
        event: dict[str, object] = {
            "asset": "511210",
            "event_type": "liquidation_distribution",
            "announcement_date": "2018-01-09",
            "record_date": "2018-01-16",
            "ex_date": "2018-01-17",
            "pay_date": "2018-01-23",
            "cash_per_share": 112.79,
            "termination_date": "2018-01-26",
            "extinguishes_position": True,
            "available_date": "2018-01-09",
            "source_vintage": "official_terminal_event_pdf_set_sha256:fixture",
            "historical_backtest_allowed": True,
            "model_promotion_allowed": False,
            "validation_status": "pass",
        }
        event.update(overrides)
        return event

    def test_terminal_cash_event_is_separate_from_market_prices(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "511210",
                    "asset_name": "Delisted Bond ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2013-08-16"),
                    "delist_date": pd.Timestamp("2018-01-25"),
                    "lifecycle_status": "delisted",
                    "selection_group": "explicit",
                }
            ]
        ).itertuples(index=False).__next__()
        prices = self._collector_prices(
            [("2017-10-17", 100.0), ("2017-10-18", 100.1)]
        )
        dividends = pd.DataFrame(
            [("2018-01-17", 112.79)],
            columns=["date", "cumulative_dividend"],
        )

        with self.assertRaisesRegex(ValueError, "cannot be aligned to a trading date"):
            build_lifecycle_observation(
                prices,
                dividends,
                lifecycle,
                "2018-01-25",
                "2026-07-18T18:13:36+08:00",
                "fixture:without-terminal-evidence",
            )

        output, event_output, diagnostics = build_lifecycle_observation(
            prices,
            dividends,
            lifecycle,
            "2018-01-25",
            "2026-07-18T18:13:36+08:00",
            "fixture:with-terminal-evidence",
            terminal_cash_event=self._terminal_cash_event(),
        )
        self.assertEqual(output["date"].dt.strftime("%Y-%m-%d").tolist(), ["2017-10-17", "2017-10-18"])
        self.assertEqual(set(output["return_basis"]), {"total_return_pre_terminal_cash_event"})
        self.assertAlmostEqual(float(output["source_cash_distribution"].sum()), 0.0)
        self.assertEqual(len(event_output), 1)
        event = event_output.iloc[0]
        self.assertEqual(event["event_type"], "liquidation_distribution")
        self.assertEqual(pd.Timestamp(event["announcement_date"]), pd.Timestamp("2018-01-09"))
        self.assertEqual(pd.Timestamp(event["pay_date"]), pd.Timestamp("2018-01-23"))
        self.assertEqual(pd.Timestamp(event["available_date"]), pd.Timestamp("2018-01-09"))
        self.assertTrue(bool(event["extinguishes_position"]))
        self.assertTrue(bool(event["pit_actionable"]))
        self.assertTrue(bool(event["historical_backtest_allowed"]))
        self.assertFalse(bool(event["model_promotion_allowed"]))
        self.assertEqual(diagnostics["terminal_cash_events"], 1)
        self.assertEqual(diagnostics["terminal_cash_evidence_status"], "governed_known_exception")

    def test_terminal_cash_registry_is_time_filtered_and_hash_authenticated(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            event_path = directory / "events.csv"
            manifest_path = directory / "manifest.json"
            promoter_path = ROOT / "strategy_lab" / "long_hold_v4" / "pit_etf_terminal_cash_event_promoter.py"
            pd.DataFrame(
                [
                    {
                        "event_id": "fixture:511210:20180123",
                        "asset": "511210",
                        "event_type": "liquidation_distribution",
                        "distribution_sequence": 1,
                        "holder_scope": "record_date_holders",
                        "announcement_date": "2018-01-09",
                        "available_trade_date": "2018-01-10",
                        "available_date": "2018-01-10",
                        "entitlement_date": "2018-01-16",
                        "record_date": "2018-01-16",
                        "pay_date": "2018-01-23",
                        "accounting_date": "2018-01-23",
                        "cash_per_share": 112.79,
                        "is_final_distribution": True,
                        "additional_distribution_expected": False,
                        "extinguishes_position": True,
                        "source_pdf_sha256_set": "a" * 64,
                        "source_vintage": "official_terminal_event_pdf_set_sha256:" + "b" * 64,
                        "historical_backtest_allowed": True,
                        "model_promotion_allowed": False,
                        "validation_status": "pass",
                    }
                ]
            ).to_csv(event_path, index=False, encoding="utf-8-sig")
            event_sha = hashlib.sha256(event_path.read_bytes()).hexdigest()
            manifest = {
                "qualification_status": "PROMOTED_VALIDATED_TERMINAL_EVENT_CHAIN_V2",
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
                "scope_complete": False,
                "rows": 1,
                "assets": 1,
                "complete_chain_assets": 1,
                "quarantined_candidate_rows": 0,
                "code_path": str(promoter_path),
                "code_sha256": hashlib.sha256(promoter_path.read_bytes()).hexdigest(),
                "outputs": [
                    {
                        "role": "pit_etf_terminal_cash_events",
                        "path": str(event_path),
                        "sha256": event_sha,
                        "rows": 1,
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            before_accounting, _ = load_terminal_cash_event_registry(
                event_path,
                manifest_path,
                "2018-01-22",
            )
            on_accounting, metadata = load_terminal_cash_event_registry(
                event_path,
                manifest_path,
                "2018-01-23",
            )
            self.assertTrue(before_accounting["asset"].ne("511210").all())
            selected = on_accounting[on_accounting["asset"].eq("511210")]
            self.assertEqual(len(selected), 1)
            self.assertEqual(pd.Timestamp(selected.iloc[0]["accounting_date"]), pd.Timestamp("2018-01-23"))
            self.assertFalse(metadata["scope_complete"])

            loaded, _ = load_terminal_cash_event_registry(event_path, manifest_path, "2018-01-23")
            self.assertEqual(len(loaded[loaded["asset"].eq("511210")]), 1)
            event_path.write_bytes(event_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "path or hash mismatch"):
                load_terminal_cash_event_registry(event_path, manifest_path, "2018-01-23")

    def test_v2_terminal_event_chain_keeps_interim_and_final_cash_ledger_rows(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "159522",
                    "asset_name": "Delisted ETF",
                    "exchange": "SZSE",
                    "list_date": pd.Timestamp("2023-01-01"),
                    "delist_date": pd.Timestamp("2024-09-20"),
                    "lifecycle_status": "delisted",
                    "selection_group": "explicit",
                }
            ]
        ).itertuples(index=False).__next__()
        prices = self._collector_prices(
            [("2024-09-18", 1.0), ("2024-09-20", 0.99)]
        )
        common = {
            "asset": "159522",
            "event_type": "liquidation_distribution",
            "holder_scope": "all_registered_holders",
            "entitlement_date": "2024-09-20",
            "record_date": "2024-09-20",
            "ex_date": None,
            "termination_date": "2024-09-20",
            "source_pdf_sha256_set": "a" * 64,
            "source_vintage": "official_terminal_event_pdf_set_sha256:fixture",
            "historical_backtest_allowed": True,
            "model_promotion_allowed": False,
            "validation_status": "pass",
        }
        events = pd.DataFrame(
            [
                {
                    **common,
                    "event_id": "interim",
                    "distribution_sequence": 1,
                    "announcement_date": "2024-09-23",
                    "available_trade_date": "2024-09-24",
                    "available_date": "2024-09-24",
                    "pay_date": "2024-09-25",
                    "accounting_date": "2024-09-25",
                    "cash_per_share": 0.95,
                    "is_final_distribution": False,
                    "additional_distribution_expected": True,
                    "extinguishes_position": False,
                },
                {
                    **common,
                    "event_id": "final",
                    "distribution_sequence": 2,
                    "announcement_date": "2024-11-16",
                    "available_trade_date": "2024-11-17",
                    "available_date": "2024-11-17",
                    "pay_date": "2024-11-21",
                    "accounting_date": "2024-11-21",
                    "cash_per_share": 0.02,
                    "is_final_distribution": True,
                    "additional_distribution_expected": False,
                    "extinguishes_position": True,
                },
            ]
        )

        output, cash_ledger, diagnostics = build_lifecycle_observation(
            prices,
            pd.DataFrame(columns=["date", "cumulative_dividend"]),
            lifecycle,
            "2024-11-21",
            "2026-07-19T12:00:00+08:00",
            "fixture:v2-chain",
            terminal_cash_event=events,
        )

        self.assertEqual(len(output), 2)
        self.assertAlmostEqual(float(output["source_cash_distribution"].sum()), 0.0)
        self.assertEqual(len(cash_ledger), 2)
        self.assertEqual(
            cash_ledger["event_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-09-25", "2024-11-21"],
        )
        self.assertEqual(cash_ledger["extinguishes_position"].tolist(), [False, True])
        self.assertEqual(diagnostics["terminal_cash_events"], 2)
        self.assertEqual(diagnostics["terminal_cash_evidence_status"], "formal_validated_event_chain_v2")

    def test_lifecycle_batch_quarantines_unresolved_corporate_action(self):
        selected = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "asset_name": "Good ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                    "selection_group": "explicit",
                },
                {
                    "asset": "510051",
                    "asset_name": "Bad ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                    "selection_group": "explicit",
                },
            ]
        )

        def loader(lifecycle):
            prices = self._collector_prices(
                [("2026-01-20", 10.0), ("2026-01-21", 10.1 if lifecycle.asset == "510050" else 5.0)]
            )
            return {
                "price": prices,
                "dividend": pd.DataFrame(columns=["date", "cumulative_dividend"]),
                "metadata": {
                    "status": "success",
                    "fetched_at": "2026-01-22T10:00:00+08:00",
                    "price_sha256": "price",
                    "dividend_sha256": "dividend",
                    "collection_action": "fixture",
                },
            }

        output, _, statuses = process_lifecycle_batch(selected, loader, "2026-01-21")
        self.assertEqual(set(output["asset"]), {"510050"})
        indexed = statuses.set_index("asset")
        self.assertEqual(indexed.loc["510050", "build_status"], "ready_observation")
        self.assertEqual(indexed.loc["510051", "build_status"], "quarantined")
        self.assertEqual(indexed.loc["510051", "collection_status"], "completed")
        self.assertIn("unresolved ETF corporate-action jump", indexed.loc["510051", "error"])

    def test_lifecycle_status_exposes_inferred_action_evidence(self):
        selected = pd.DataFrame(
            [
                {
                    "asset": "512890",
                    "asset_name": "Inferred split ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                    "selection_group": "explicit",
                }
            ]
        )

        def loader(_lifecycle):
            return {
                "price": self._collector_prices([("2026-01-20", 10.0), ("2026-01-21", 5.0)]),
                "dividend": pd.DataFrame(
                    [("2026-01-21", 0.0)],
                    columns=["date", "cumulative_dividend"],
                ),
                "metadata": {
                    "status": "success",
                    "fetched_at": "2026-01-22T10:00:00+08:00",
                    "price_sha256": "price",
                    "dividend_sha256": "dividend",
                    "collection_action": "fixture",
                },
            }

        _, _, statuses = process_lifecycle_batch(selected, loader, "2026-01-21")
        status = statuses.iloc[0]
        self.assertEqual(status["inferred_corporate_actions"], 1)
        self.assertEqual(status["governed_corporate_actions"], 0)
        self.assertEqual(status["corporate_action_evidence_status"], "heuristic_inference_present")
        detail = json.loads(status["corporate_action_evidence_detail_json"])
        self.assertEqual(detail[0]["price_effective_date"], "2026-01-21")
        self.assertEqual(detail[0]["evidence_basis"], "zero_marker_common_factor_inference")

    def test_lifecycle_collector_reuses_a_hash_validated_cache(self):
        lifecycle = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "asset_name": "ETF",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2026-01-20"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                }
            ]
        ).itertuples(index=False).__next__()
        calls = {"price": 0, "dividend": 0}

        def price_fetcher(_asset):
            calls["price"] += 1
            return self._collector_prices([("2026-01-20", 10.0), ("2026-01-21", 10.1)])

        def dividend_fetcher(_asset):
            calls["dividend"] += 1
            return pd.DataFrame(columns=["date", "cumulative_dividend"])

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            first = collect_asset(
                lifecycle,
                "2026-01-21",
                cache_dir,
                attempts=1,
                price_fetcher=price_fetcher,
                dividend_fetcher=dividend_fetcher,
            )
            second = collect_asset(
                lifecycle,
                "2026-01-21",
                cache_dir,
                attempts=1,
                price_fetcher=price_fetcher,
                dividend_fetcher=dividend_fetcher,
            )
        self.assertEqual(first["collection_action"], "fetched")
        self.assertEqual(second["collection_action"], "cache_reused")
        self.assertEqual(calls, {"price": 1, "dividend": 1})

    def test_etf_collector_only_trips_circuit_breaker_on_provider_failures(self):
        self.assertEqual(provider_circuit_breaker_reason("HTTP 429 Too Many Requests"), "http_429")
        self.assertEqual(provider_circuit_breaker_reason("403 Forbidden"), "http_403")
        self.assertEqual(
            provider_circuit_breaker_reason("RemoteDisconnected: remote disconnected"),
            "connection_rejected",
        )
        self.assertIsNone(provider_circuit_breaker_reason("ETF price response missing fields"))

    def test_etf_snapshot_write_preserves_existing_stock_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_combined = snapshot_store.COMBINED_PATH
            old_parts = snapshot_store.PART_PATHS
            try:
                snapshot_store.COMBINED_PATH = root / "research_snapshot.csv"
                snapshot_store.PART_PATHS = {
                    "stock": root / "stock_research_snapshot.csv",
                    "etf": root / "etf_research_snapshot.csv",
                }
                pd.DataFrame([{"asset": "600000", "asset_type": "stock", "sector": "bank"}]).to_csv(
                    snapshot_store.COMBINED_PATH, index=False, encoding="utf-8-sig"
                )
                combined = snapshot_store.write_snapshot_part(
                    "etf", pd.DataFrame([{"asset": "510880", "asset_type": "etf", "sector": "dividend_index"}])
                )
                self.assertEqual(set(combined["asset"]), {"510880", "600000"})
                self.assertEqual(len(pd.read_csv(snapshot_store.COMBINED_PATH, encoding="utf-8-sig")), 2)
            finally:
                snapshot_store.COMBINED_PATH = old_combined
                snapshot_store.PART_PATHS = old_parts

    def test_source_manifest_detects_upstream_file_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw.csv"
            source.write_text("value\n1\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-17",
                        "input_files": [{"path": "raw.csv", "sha256": digest}],
                        "code_files": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(verify_source_manifest(root, manifest, pd.Timestamp("2026-07-17")), [])
            source.write_text("value\n2\n", encoding="utf-8")
            failures = verify_source_manifest(root, manifest, pd.Timestamp("2026-07-17"))
            self.assertEqual(failures, ["source_manifest_hash_mismatch=raw.csv"])


if __name__ == "__main__":
    unittest.main()
