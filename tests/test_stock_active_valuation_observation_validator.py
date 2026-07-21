from __future__ import annotations

import unittest

import pandas as pd

from strategy_lab.long_hold_v4.stock_active_valuation_observation_collector import (
    QUALIFICATION_STATUS,
)
from strategy_lab.long_hold_v4.stock_active_valuation_observation_validator import (
    _source_checks,
    build_asset_metrics,
    validate_observation_integrity,
)


class StockActiveValuationObservationValidatorTests(unittest.TestCase):
    def _observation(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2025-01-02",
                    "asset": "600000",
                    "pe_ttm": 8.0,
                    "pb_mrq": 0.8,
                    "float_market_cap": 100.0,
                    "available_date": "2026-07-19",
                    "data_source": "eastmoney",
                    "current_final_snapshot": True,
                    "pit_actionable": False,
                    "qualification_status": QUALIFICATION_STATUS,
                    "historical_backtest_allowed": False,
                    "model_promotion_allowed": False,
                }
            ]
        )

    def test_integrity_requires_non_pit_flags_and_actual_available_date(self):
        observation = self._observation()
        status = pd.DataFrame(
            [{"asset": "600000", "collection_status": "completed", "error": ""}]
        )
        manifest = {"target_assets": 1, "completed_assets": 1, "observation_rows": 1}
        result = validate_observation_integrity(observation, status, manifest, "2025-01-02")
        self.assertTrue(result["pass"])
        leaked = observation.copy()
        leaked["available_date"] = "2024-12-31"
        self.assertFalse(
            validate_observation_integrity(leaked, status, manifest, "2025-01-02")["pass"]
        )

    def test_joinquant_source_checks_include_asset_coverage(self):
        metrics = {
            "checks": 2_000,
            "pe_median_abs_relative_error": 0.001,
            "pe_p95_abs_relative_error": 0.01,
            "pb_median_abs_relative_error": 0.001,
            "pb_p95_abs_relative_error": 0.05,
            "cap_median_abs_relative_error": 0.001,
            "cap_p95_abs_relative_error": 0.01,
        }
        self.assertTrue(all(_source_checks("joinquant", metrics, 150, 0.95).values()))
        self.assertFalse(_source_checks("joinquant", metrics, 150, 0.90)["minimum_asset_coverage"])

    def test_asset_metrics_keep_tail_disagreement_visible(self):
        checks = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-01-02"),
                    "asset": "600000",
                    "pe_abs_relative_error": 0.01,
                    "pb_abs_relative_error": 0.01,
                    "cap_abs_relative_error": 0.01,
                },
                {
                    "date": pd.Timestamp("2025-01-03"),
                    "asset": "600000",
                    "pe_abs_relative_error": 0.01,
                    "pb_abs_relative_error": 0.50,
                    "cap_abs_relative_error": 0.01,
                },
            ]
        )
        sectors = pd.DataFrame([{"asset": "600000", "sector": "bank"}])
        result = build_asset_metrics(checks, "joinquant", sectors)
        self.assertIn("pb_tail", result.loc[0, "warning_flags"])
        self.assertEqual(result.loc[0, "sector"], "bank")


if __name__ == "__main__":
    unittest.main()
