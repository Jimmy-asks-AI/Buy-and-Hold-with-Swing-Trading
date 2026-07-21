import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_sse_share_action_validator import evaluate_candidates


def candidate_row(**overrides):
    row = {
        "asset": "510100",
        "action_type": "share_merger",
        "event_date": pd.Timestamp("2025-03-07"),
        "price_effective_date": pd.Timestamp("2025-03-10"),
        "shares_after_per_share_before": 0.49719902,
        "announcement_date": pd.Timestamp("2025-03-10"),
        "source_document_title": "份额合并结果公告",
        "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/result.pdf",
        "source_type": "exchange_announcement",
        "pdf_path": "missing.pdf",
        "pdf_sha256": "a" * 64,
        "text_path": "missing.txt",
        "text_sha256": "b" * 64,
        "factor_relative_error_to_inference": 0.0056,
        "factor_relative_error_to_observed_price_ratio": 0.0056,
        "normalized_price_ratio_residual": 0.0056,
        "action_event_date_distance_days": 3,
        "review_status": "official_pdf_corrects_heuristic_review_required",
        "historical_backtest_allowed": False,
    }
    row.update(overrides)
    return row


class TestSseEtfShareActionValidator(unittest.TestCase):
    def test_business_checks_pass_while_missing_fixture_hashes_fail(self):
        checks = evaluate_candidates(pd.DataFrame([candidate_row()]))
        status = checks.set_index("check")["status"].to_dict()
        for name in (
            "dates_valid",
            "primary_key_unique",
            "factor_positive_finite",
            "action_direction_consistent",
            "action_date_precedes_price_effective_date",
            "announcement_not_after_price_effective_date",
            "independent_price_ratio_crosscheck",
            "official_source_url",
            "official_source_type",
            "collector_review_status_allowed",
            "candidate_historical_use_stays_disabled",
        ):
            self.assertEqual(status[name], "pass")
        self.assertEqual(status["pdf_hash_match"], "fail")
        self.assertEqual(status["text_hash_match"], "fail")

    def test_wrong_direction_late_event_and_large_price_error_fail(self):
        frame = pd.DataFrame(
            [
                candidate_row(
                    action_type="share_split",
                    event_date=pd.Timestamp("2025-03-11"),
                    shares_after_per_share_before=0.5,
                    normalized_price_ratio_residual=0.25,
                    action_event_date_distance_days=-1,
                )
            ]
        )
        checks = evaluate_candidates(frame)
        failed = set(checks.loc[checks["status"].eq("fail"), "check"])
        self.assertIn("action_direction_consistent", failed)
        self.assertIn("action_date_precedes_price_effective_date", failed)
        self.assertIn("independent_price_ratio_crosscheck", failed)

    def test_price_crosscheck_uses_normalized_return_direction(self):
        checks = evaluate_candidates(
            pd.DataFrame(
                [
                    candidate_row(
                        factor_relative_error_to_observed_price_ratio=0.10035587,
                        normalized_price_ratio_residual=0.09120310,
                    )
                ]
            )
        )
        status = checks.set_index("check")["status"].to_dict()
        self.assertEqual(status["independent_price_ratio_crosscheck"], "pass")

    def test_duplicate_effective_action_fails_both_rows(self):
        frame = pd.DataFrame([candidate_row(), candidate_row()])
        checks = evaluate_candidates(frame)
        duplicates = checks[checks["check"].eq("primary_key_unique")]
        self.assertEqual(duplicates["status"].tolist(), ["fail", "fail"])


if __name__ == "__main__":
    unittest.main()
