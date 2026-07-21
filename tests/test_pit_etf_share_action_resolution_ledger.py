import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_share_action_resolution_ledger import build_resolution_ledger


class TestEtfShareActionResolutionLedger(unittest.TestCase):
    def test_resolved_candidate_must_match_governed_registry(self):
        queue = pd.DataFrame(
            [{"asset": "159558", "price_effective_date": "2026-07-09", "inferred_factor": 3.0}]
        )
        candidate = {
            "asset": "159558",
            "price_effective_date": "2026-07-09",
            "event_date": "2026-07-08",
            "announcement_date": "2026-07-09",
            "shares_after_per_share_before": 3.0,
            "action_type": "share_split",
            "source_url": "https://static.cninfo.com.cn/finalpage/example.pdf",
            "resolution_source": "cninfo_official_fund_disclosure",
            "validation_schema": "cninfo_etf_share_action_cross_evidence_v1",
            "validation_manifest_sha256": "a" * 64,
            "promotion_manifest_sha256": "b" * 64,
        }
        registry = pd.DataFrame([{**candidate, "review_status": "verified"}])
        ledger = build_resolution_ledger(queue, pd.DataFrame([candidate]), registry)
        self.assertEqual(ledger.loc[0, "resolution_status"], "resolved_governed_official")
        self.assertTrue(bool(ledger.loc[0, "registry_match"]))

    def test_candidate_with_different_governed_factor_remains_unresolved(self):
        queue = pd.DataFrame(
            [{"asset": "510100", "price_effective_date": "2025-03-10", "inferred_factor": 0.5}]
        )
        candidate = {
            "asset": "510100",
            "price_effective_date": "2025-03-10",
            "event_date": "2025-03-07",
            "announcement_date": "2025-03-10",
            "shares_after_per_share_before": 0.49719902,
            "action_type": "share_merger",
            "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/example.pdf",
            "resolution_source": "sse_official_fund_announcement",
            "validation_schema": "sse_etf_share_action_cross_evidence_v2",
            "validation_manifest_sha256": "a" * 64,
            "promotion_manifest_sha256": "b" * 64,
        }
        registry = pd.DataFrame(
            [{**candidate, "shares_after_per_share_before": 0.5, "review_status": "verified"}]
        )
        ledger = build_resolution_ledger(queue, pd.DataFrame([candidate]), registry)
        self.assertEqual(ledger.loc[0, "resolution_status"], "unresolved_registry_mismatch")
        self.assertFalse(bool(ledger.loc[0, "registry_match"]))


if __name__ == "__main__":
    unittest.main()
