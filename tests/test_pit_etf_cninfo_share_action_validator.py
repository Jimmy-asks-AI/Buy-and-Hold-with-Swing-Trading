import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_cninfo_share_action_validator import evaluate_candidates


def candidate_row(**overrides):
    row = {
        "asset": "159558",
        "action_type": "share_split",
        "event_date": pd.Timestamp("2026-07-08"),
        "price_effective_date": pd.Timestamp("2026-07-09"),
        "shares_after_per_share_before": 3.0,
        "announcement_date": pd.Timestamp("2026-07-09"),
        "source_document_title": "基金份额拆分结果公告",
        "source_url": "https://static.cninfo.com.cn/finalpage/2026-07-09/example.pdf",
        "source_type": "regulatory_filing",
        "pdf_path": "missing.pdf",
        "pdf_sha256": "a" * 64,
        "text_path": "missing.txt",
        "text_sha256": "b" * 64,
        "factor_relative_error_to_inference": 0.0,
        "factor_relative_error_to_observed_price_ratio": 0.10035587,
        "normalized_price_ratio_residual": 0.09120310,
        "action_event_date_distance_days": 1,
        "review_status": "official_pdf_factor_and_near_date_found_review_required",
        "historical_backtest_allowed": False,
    }
    row.update(overrides)
    return row


class TestCninfoEtfShareActionValidator(unittest.TestCase):
    def test_cninfo_source_and_normalized_price_residual_pass(self):
        checks = evaluate_candidates(pd.DataFrame([candidate_row()]))
        status = checks.set_index("check")["status"].to_dict()
        self.assertEqual(status["official_source_url"], "pass")
        self.assertEqual(status["official_source_type"], "pass")
        self.assertEqual(status["independent_price_ratio_crosscheck"], "pass")
        self.assertEqual(status["action_date_distance_consistent"], "pass")

    def test_wrong_source_type_and_inconsistent_distance_fail(self):
        checks = evaluate_candidates(
            pd.DataFrame(
                [
                    candidate_row(
                        source_type="exchange_announcement",
                        action_event_date_distance_days=2,
                    )
                ]
            )
        )
        failed = set(checks.loc[checks["status"].eq("fail"), "check"])
        self.assertIn("official_source_type", failed)
        self.assertIn("action_date_distance_consistent", failed)


if __name__ == "__main__":
    unittest.main()
