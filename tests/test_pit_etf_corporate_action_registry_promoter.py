import unittest

from strategy_lab.long_hold_v4.pit_etf_corporate_action_registry_promoter import candidate_actions, merge_registry_payload


def action(asset="510100", factor=0.5):
    return {
        "asset": asset,
        "action_type": "share_merger",
        "event_date": "2025-03-07",
        "price_effective_date": "2025-03-10",
        "shares_after_per_share_before": factor,
        "announcement_date": "2025-03-10",
        "source_document_title": "result",
        "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/result.pdf",
        "source_type": "exchange_announcement",
        "review_status": "verified",
    }


class TestEtfCorporateActionRegistryPromoter(unittest.TestCase):
    def test_merge_adds_new_action_and_is_idempotent(self):
        payload = {"schema_version": 1, "factor_definition": "shares_after_per_share_before", "actions": []}
        merged, added = merge_registry_payload(payload, [action()])
        self.assertEqual(added, 1)
        self.assertEqual(len(merged["actions"]), 1)
        again, added_again = merge_registry_payload(merged, [action()])
        self.assertEqual(added_again, 0)
        self.assertEqual(again, merged)

    def test_merge_rejects_conflicting_effective_action(self):
        payload = {
            "schema_version": 1,
            "factor_definition": "shares_after_per_share_before",
            "actions": [action(factor=0.5)],
        }
        with self.assertRaisesRegex(ValueError, "conflicts"):
            merge_registry_payload(payload, [action(factor=0.49719902)])

    def test_merge_rejects_duplicate_existing_keys(self):
        payload = {
            "schema_version": 1,
            "factor_definition": "shares_after_per_share_before",
            "actions": [action(), action()],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_registry_payload(payload, [])

    def test_candidate_actions_preserve_validated_source_type(self):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    **action(asset="159558", factor=3.0),
                    "source_url": "https://static.cninfo.com.cn/finalpage/2026-07-09/example.pdf",
                    "source_type": "regulatory_filing",
                }
            ]
        )
        actions = candidate_actions(
            frame,
            official_source_url_prefix="https://static.cninfo.com.cn/finalpage/",
            expected_source_type="regulatory_filing",
        )
        self.assertEqual(actions[0]["source_type"], "regulatory_filing")

    def test_candidate_actions_reject_wrong_source_type(self):
        import pandas as pd

        frame = pd.DataFrame([action()])
        with self.assertRaisesRegex(ValueError, "source type"):
            candidate_actions(
                frame,
                official_source_url_prefix="https://www.sse.com.cn/disclosure/fund/announcement/",
                expected_source_type="regulatory_filing",
            )


if __name__ == "__main__":
    unittest.main()
