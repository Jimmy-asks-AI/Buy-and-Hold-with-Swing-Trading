import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_validator import evaluate_candidates


def candidate_row(**overrides):
    row = {
        "asset": "510050",
        "announcement_date": pd.Timestamp("2006-11-11"),
        "record_date": pd.Timestamp("2006-11-15"),
        "ex_date": pd.Timestamp("2006-11-16"),
        "pay_date": pd.Timestamp("2006-11-21"),
        "cash_per_share": 0.037,
        "source_document_title": "50ETF收益分配公告",
        "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/example.pdf",
        "source_type": "exchange_announcement",
        "pdf_path": "missing.pdf",
        "pdf_sha256": "a" * 64,
        "text_path": "missing.txt",
        "text_sha256": "b" * 64,
        "source_event_date": pd.Timestamp("2006-11-16"),
        "cash_relative_error_to_discovery": 0.0,
        "ex_date_distance_days": 0,
        "review_status": "official_pdf_cash_and_dates_found_review_required",
        "historical_backtest_allowed": False,
    }
    row.update(overrides)
    return row


class TestEtfDividendValidator(unittest.TestCase):
    def test_business_checks_pass_while_missing_fixture_hashes_fail(self):
        checks = evaluate_candidates(pd.DataFrame([candidate_row()]))
        status = checks.set_index("check")["status"].to_dict()
        for name in (
            "dates_valid",
            "formal_primary_key_unique",
            "discovery_event_unique",
            "cash_per_share_positive_finite",
            "event_chronology",
            "discovery_ex_date_crosscheck",
            "discovery_cash_crosscheck",
            "official_source_contract",
            "collector_review_status_allowed",
            "candidate_historical_use_stays_disabled",
        ):
            self.assertEqual(status[name], "pass")
        self.assertEqual(status["pdf_hash_match"], "fail")
        self.assertEqual(status["text_hash_match"], "fail")

    def test_bad_chronology_source_and_cash_crosscheck_fail(self):
        checks = evaluate_candidates(
            pd.DataFrame(
                [
                    candidate_row(
                        announcement_date=pd.Timestamp("2006-11-17"),
                        source_type="regulatory_filing",
                        cash_relative_error_to_discovery=0.01,
                    )
                ]
            )
        )
        failed = set(checks.loc[checks["status"].eq("fail"), "check"])
        self.assertIn("event_chronology", failed)
        self.assertIn("official_source_contract", failed)
        self.assertIn("discovery_cash_crosscheck", failed)


if __name__ == "__main__":
    unittest.main()
