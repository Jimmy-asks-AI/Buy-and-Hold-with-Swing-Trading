from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_joinquant_validation_collector import (
    normalise_joinquant_nav,
    normalise_joinquant_prices,
    select_window_assets,
)
from strategy_lab.long_hold_v4.pit_etf_price_nav_validator import (
    build_version_inventory,
    compare_nav_panels,
    compare_price_panels,
)


class JoinQuantETFValidationCollectorTests(unittest.TestCase):
    def test_window_selection_respects_listing_and_delisting(self):
        lifecycles = pd.DataFrame(
            [
                {
                    "asset": "510001",
                    "asset_name": "Old",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2020-01-01"),
                    "delist_date": pd.Timestamp("2025-04-30"),
                    "lifecycle_status": "delisted",
                },
                {
                    "asset": "510002",
                    "asset_name": "Inside",
                    "exchange": "SSE",
                    "list_date": pd.Timestamp("2025-05-01"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                },
                {
                    "asset": "159003",
                    "asset_name": "New",
                    "exchange": "SZSE",
                    "list_date": pd.Timestamp("2026-04-02"),
                    "delist_date": pd.NaT,
                    "lifecycle_status": "active",
                },
            ]
        )
        selected = select_window_assets(lifecycles, "2025-05-01", "2026-04-01")
        self.assertEqual(selected["asset"].tolist(), ["510002"])
        self.assertEqual(selected["jq_code"].tolist(), ["510002.XSHG"])

    def test_price_normalisation_preserves_raw_unadjusted_fields(self):
        raw = pd.DataFrame(
            [
                {
                    "time": "2025-05-06",
                    "code": "510880.XSHG",
                    "open": 3.04,
                    "high": 3.05,
                    "low": 3.02,
                    "close": 3.03,
                    "volume": 100.0,
                    "money": 303.0,
                    "paused": 0.0,
                },
                {
                    "time": "2025-05-07",
                    "code": "510880.XSHG",
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": 0.0,
                    "money": 0.0,
                    "paused": 1.0,
                },
            ]
        )
        output = normalise_joinquant_prices(
            raw,
            ["510880.XSHG"],
            "2025-05-01",
            "2026-04-01",
        )
        self.assertEqual(output["asset"].tolist(), ["510880", "510880"])
        self.assertEqual(output["exchange"].tolist(), ["SSE", "SSE"])
        self.assertEqual(output["amount"].tolist(), [303.0, 0.0])
        self.assertEqual(output["paused"].tolist(), [0.0, 1.0])

    def test_price_normalisation_rejects_unrequested_codes(self):
        raw = pd.DataFrame(
            [
                {
                    "time": "2025-05-06",
                    "code": "510881.XSHG",
                    "open": 3.04,
                    "high": 3.05,
                    "low": 3.02,
                    "close": 3.03,
                    "volume": 100.0,
                    "money": 303.0,
                    "paused": 0.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "unrequested codes"):
            normalise_joinquant_prices(raw, ["510880.XSHG"], "2025-05-01", "2026-04-01")

    def test_nav_normalisation_merges_unit_and_cumulative_values(self):
        dates = pd.to_datetime(["2025-05-06", "2025-05-07"])
        unit = pd.DataFrame({"510880.XSHG": [3.0311, 3.0521]}, index=dates)
        cumulative = pd.DataFrame({"510880.XSHG": [2.9252, 2.9390]}, index=dates)
        output = normalise_joinquant_nav(
            unit,
            cumulative,
            {"510880.XSHG": "SSE"},
        )
        self.assertEqual(output["asset"].tolist(), ["510880", "510880"])
        self.assertEqual(output["unit_nav"].tolist(), [3.0311, 3.0521])
        self.assertEqual(output["cumulative_nav"].tolist(), [2.9252, 2.9390])

    def test_price_comparison_uses_explicit_precision_tolerances(self):
        joinquant = pd.DataFrame(
            [
                {
                    "date": "2025-05-06",
                    "asset": "510880",
                    "open": 3.04,
                    "high": 3.05,
                    "low": 3.02,
                    "close": 3.03,
                    "volume": 100.0,
                    "amount": 303.0,
                    "paused": 0,
                }
            ]
        )
        source = joinquant.drop(columns="paused").copy()
        source.loc[0, "close"] += 0.001
        checks = compare_price_panels(joinquant, source)
        self.assertTrue(bool(checks.iloc[0]["ohlc_match"]))
        source.loc[0, "close"] += 0.01
        checks = compare_price_panels(joinquant, source)
        self.assertFalse(bool(checks.iloc[0]["ohlc_match"]))

    def test_price_comparison_excludes_zero_activity_markers(self):
        joinquant = pd.DataFrame(
            [
                {
                    "date": "2025-05-06",
                    "asset": "510880",
                    "open": 3.04,
                    "high": 3.04,
                    "low": 3.04,
                    "close": 3.04,
                    "volume": 0.0,
                    "amount": 0.0,
                    "paused": 0,
                }
            ]
        )
        source = pd.DataFrame(
            columns=["date", "asset", "open", "high", "low", "close", "volume", "amount"]
        )
        self.assertTrue(compare_price_panels(joinquant, source).empty)

    def test_nav_comparison_detects_value_mismatch(self):
        joinquant = pd.DataFrame(
            [
                {
                    "date": "2025-05-06",
                    "asset": "510880",
                    "unit_nav": 3.0311,
                    "cumulative_nav": 2.9252,
                }
            ]
        )
        source = joinquant.copy()
        self.assertTrue(bool(compare_nav_panels(joinquant, source).iloc[0]["nav_match"]))
        source.loc[0, "unit_nav"] += 0.001
        self.assertFalse(bool(compare_nav_panels(joinquant, source).iloc[0]["nav_match"]))

    def test_version_inventory_counts_source_hashes_not_output_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, digest in enumerate(["same", "same", "changed"]):
                path = Path(temporary) / f"run_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "run_id": f"run_{index}",
                            "created_at": f"2026-07-{17 + index:02d}",
                            "inputs": [
                                {
                                    "role": "etf_raw_price",
                                    "asset": "510880",
                                    "sha256": digest,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            inventory = build_version_inventory(paths)
            self.assertEqual(int(inventory.iloc[0]["declared_run_count"]), 3)
            self.assertEqual(int(inventory.iloc[0]["distinct_source_price_versions"]), 2)
            self.assertTrue(bool(inventory.iloc[0]["multiple_source_versions_available"]))


if __name__ == "__main__":
    unittest.main()
