import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_announcement_collector import parse_document_text
from strategy_lab.long_hold_v4.pit_etf_dividend_universe_coverage_collector import (
    recover_delisted_cninfo_identity,
    select_direct_official_event,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, data, timeout):
        self.calls.append((url, data, timeout))
        return FakeResponse(self.payload)


class TestEtfDividendUniverseCoverageCollector(unittest.TestCase):
    def test_selects_complete_event_without_price_derived_target(self):
        parsed = parse_document_text(
            "每10份基金份额派发现金收益0.37元。"
            "权益登记日：2006年11月15日。除息日：2006年11月16日。"
            "红利发放日：2006年11月21日。"
        )
        selected = select_direct_official_event(parsed, pd.Timestamp("2006-11-11"))
        self.assertEqual(selected["parse_status"], "complete_unique_official_event")
        self.assertEqual(selected["ex_date"], pd.Timestamp("2006-11-16"))
        self.assertAlmostEqual(selected["cash_per_share"], 0.037)

    def test_equal_rank_cash_values_stay_ambiguous(self):
        parsed = {
            "cash_candidates": [
                {"cash_per_share": 0.01, "rank": 0},
                {"cash_per_share": 0.02, "rank": 0},
            ],
            "record_date_candidates": [{"date": "2025-01-02", "rank": 0}],
            "ex_date_candidates": [{"date": "2025-01-03", "rank": 0}],
            "pay_date_candidates": [{"date": "2025-01-06", "rank": 0}],
        }
        selected = select_direct_official_event(parsed, "2024-12-30")
        self.assertEqual(selected["parse_status"], "ambiguous_best_official_event")
        self.assertEqual(selected["best_score_tie_count"], 2)

    def test_policy_document_without_effective_dates_is_incomplete(self):
        selected = select_direct_official_event(
            {
                "cash_candidates": [],
                "record_date_candidates": [],
                "ex_date_candidates": [],
                "pay_date_candidates": [],
            },
            "2025-01-01",
        )
        self.assertEqual(
            selected["parse_status"],
            "incomplete_cash_record_date_ex_date_pay_date",
        )

    def test_delisted_identity_fallback_filters_exact_security_code(self):
        session = FakeSession(
            {
                "totalpages": 1,
                "announcements": [
                    {"secCode": "159999", "orgId": "wrong", "announcementTitle": "同名公告"},
                    {
                        "secCode": "159503",
                        "orgId": "jjjl0000089",
                        "announcementTitle": "财富管理ETF清算报告",
                        "adjunctUrl": "finalpage/example.PDF",
                    },
                ],
            }
        )
        identity = recover_delisted_cninfo_identity(
            session,
            asset="159503",
            asset_name="财富管理ETF",
            list_date=pd.Timestamp("2023-07-20"),
            delist_date=pd.Timestamp("2024-07-18"),
            as_of_date=pd.Timestamp("2026-07-17"),
        )
        self.assertEqual(identity["org_id"], "jjjl0000089")
        self.assertEqual(identity["resolution"], "fulltext_name_search_exact_sec_code")
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
