import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_evidence_queue_builder import build_evidence_queue


class TestEtfDividendEvidenceQueueBuilder(unittest.TestCase):
    def test_only_positive_cash_distributions_enter_discovery_queue(self):
        observation = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "asset_name": "50ETF",
                    "exchange": "SSE",
                    "lifecycle_status": "active",
                    "event_date": "2006-05-19",
                    "cumulative_dividend": 0.024,
                    "cash_distribution": 0.024,
                    "source_observed_at": "2026-07-18T10:00:00+08:00",
                    "source_vintage": "fixture:v1",
                    "event_type": "cash_distribution",
                    "historical_backtest_allowed": False,
                },
                {
                    "asset": "510100",
                    "asset_name": "SZ50ETF",
                    "exchange": "SSE",
                    "lifecycle_status": "active",
                    "event_date": "2025-03-10",
                    "cumulative_dividend": 0.0,
                    "cash_distribution": 0.0,
                    "source_observed_at": "2026-07-18T10:00:00+08:00",
                    "source_vintage": "fixture:v1",
                    "event_type": "zero_marker",
                    "historical_backtest_allowed": False,
                },
            ]
        )
        queue = build_evidence_queue(observation)
        self.assertEqual(queue["asset"].tolist(), ["510050"])
        self.assertAlmostEqual(float(queue.loc[0, "inferred_cash_per_share"]), 0.024)
        self.assertEqual(queue.loc[0, "exchange"], "SSE")
        self.assertFalse(bool(queue.loc[0, "historical_backtest_allowed"]))

    def test_unknown_exchange_code_fails_closed(self):
        observation = pd.DataFrame(
            [
                {
                    "asset": "999999",
                    "asset_name": "Unknown",
                    "lifecycle_status": "active",
                    "event_date": "2026-01-20",
                    "cumulative_dividend": 0.1,
                    "cash_distribution": 0.1,
                    "source_observed_at": "2026-07-18T10:00:00+08:00",
                    "source_vintage": "fixture:v1",
                    "event_type": "cash_distribution",
                    "historical_backtest_allowed": False,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "unknown exchange"):
            build_evidence_queue(observation)


if __name__ == "__main__":
    unittest.main()
