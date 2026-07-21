import json
import unittest

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_benchmark_document_selector as selector


def _candidate(asset, date, title, tags, roles, suffix):
    return {
        "asset": asset,
        "asset_name": f"ETF-{asset}",
        "exchange": "SZSE" if asset.startswith("1") else "SSE",
        "announcement_date": date,
        "published_at": f"{date}T00:00:00+08:00",
        "announcement_title": title,
        "source_url": f"https://static.cninfo.com.cn/{asset}/{suffix}.PDF",
        "source_type": "official",
        "source_category": "fund",
        "source_observed_at": "2026-07-20T00:00:00+08:00",
        "available_at": f"{date}T00:00:00+08:00",
        "available_trade_date": date,
        "available_date": date,
        "data_source": "official catalogue",
        "source_vintage": "sha256:test",
        "query_path": f"queries/{asset}.json",
        "query_sha256": "a" * 64,
        "title_tags_json": json.dumps(tags),
        "candidate_roles_json": json.dumps(roles),
        "document_validation_status": "not_started",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def _coverage(*assets):
    return pd.DataFrame(
        [
            {
                "asset": asset,
                "asset_name": f"ETF-{asset}",
                "exchange": "SZSE" if asset.startswith("1") else "SSE",
                "list_date": "2020-01-01",
                "delist_date": None,
                "query_complete": True,
            }
            for asset in assets
        ]
    )


class BenchmarkDocumentSelectorTests(unittest.TestCase):
    def test_prefers_initial_legal_document_and_keeps_listing_context(self):
        candidates = pd.DataFrame(
            [
                _candidate("510050", "2004-11-01", "基金合同", ["fund_contract"], ["initial_benchmark_candidate"], "contract"),
                _candidate("510050", "2004-12-01", "初始招募说明书", ["prospectus"], ["initial_benchmark_candidate"], "prospectus"),
                _candidate("510050", "2005-02-01", "上市交易公告书", ["listing_document"], ["initial_benchmark_candidate"], "listing"),
                _candidate("510050", "2020-01-01", "变更标的指数并修订基金合同", ["explicit_benchmark_change", "contract_amendment"], ["benchmark_change_candidate", "contract_content_review_candidate"], "change"),
                _candidate("510050", "2021-01-01", "持有人大会表决结果公告", ["holder_resolution"], ["contract_content_review_candidate"], "holder"),
            ]
        )
        selected, coverage = selector.select_documents(candidates, _coverage("510050"))
        baseline = selected[selected["baseline_selection_state"].ne("not_baseline")]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline.iloc[0]["announcement_title"], "基金合同")
        self.assertEqual(baseline.iloc[0]["selection_priority"], "P0")
        self.assertEqual(len(selected), 5)
        prospectus = selected[selected["source_url"].str.endswith("prospectus.PDF")].iloc[0]
        self.assertEqual(prospectus["selection_priority"], "P1")
        self.assertIn(
            "one_initial_prospectus_supplement_document",
            json.loads(prospectus["selection_reasons_json"]),
        )
        listing = selected[selected["source_url"].str.endswith("listing.PDF")].iloc[0]
        self.assertEqual(listing["selection_priority"], "P1")
        self.assertIn("one_canonical_listing_context_document", json.loads(listing["selection_reasons_json"]))
        change = selected[selected["source_url"].str.endswith("change.PDF")].iloc[0]
        self.assertEqual(change["selection_priority"], "P0")
        self.assertIn("all_contract_amendments", json.loads(change["selection_reasons_json"]))
        self.assertEqual(int(coverage.iloc[0]["benchmark_change_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["holder_resolution_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["legal_baseline_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["listing_context_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["initial_prospectus_supplement_document_count"]), 1)
        self.assertFalse(bool(coverage.iloc[0]["no_change_claim_allowed"]))

    def test_excludes_contract_effective_notice_from_preferred_baseline(self):
        candidates = pd.DataFrame(
            [
                _candidate("159001", "2013-03-01", "基金合同生效公告", ["fund_contract"], ["initial_benchmark_candidate"], "notice"),
                _candidate("159001", "2013-03-02", "基金合同", ["fund_contract"], ["initial_benchmark_candidate"], "contract"),
            ]
        )
        selected, _ = selector.select_documents(candidates, _coverage("159001"))
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected.iloc[0]["source_url"].endswith("contract.PDF"))

    def test_listing_only_asset_uses_listing_as_baseline_and_context(self):
        candidates = pd.DataFrame(
            [
                _candidate(
                    "510180",
                    "2006-05-01",
                    "关于修改招募说明书的公告",
                    ["prospectus"],
                    ["initial_benchmark_candidate"],
                    "misleading-notice",
                ),
                _candidate(
                    "510180",
                    "2006-05-15",
                    "上市交易公告书",
                    ["listing_document"],
                    ["initial_benchmark_candidate"],
                    "listing",
                )
            ]
        )
        selected, coverage = selector.select_documents(candidates, _coverage("510180"))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["baseline_selection_state"], "listing_only_initial_candidate")
        self.assertEqual(selected.iloc[0]["selection_priority"], "P0")
        reasons = json.loads(selected.iloc[0]["selection_reasons_json"])
        self.assertIn("one_listing_only_initial_document", reasons)
        self.assertIn("one_canonical_listing_context_document", reasons)
        self.assertEqual(int(coverage.iloc[0]["listing_only_baseline_document_count"]), 1)
        self.assertFalse(selected["source_url"].str.endswith("misleading-notice.PDF").any())

    def test_listing_only_asset_adds_earliest_full_updated_prospectus_as_dated_fallback(self):
        candidates = pd.DataFrame(
            [
                _candidate(
                    "510180",
                    "2006-05-15",
                    "上市交易公告书",
                    ["listing_document"],
                    ["initial_benchmark_candidate"],
                    "listing",
                ),
                _candidate(
                    "510180",
                    "2006-11-27",
                    "更新的招募说明书",
                    ["prospectus", "prospectus_update"],
                    ["contract_content_review_candidate"],
                    "updated-prospectus",
                ),
                _candidate(
                    "510180",
                    "2006-11-27",
                    "更新的招募说明书摘要",
                    ["prospectus", "prospectus_update"],
                    ["contract_content_review_candidate"],
                    "updated-summary",
                ),
            ]
        )
        selected, coverage = selector.select_documents(candidates, _coverage("510180"))
        fallback = selected[selected["source_url"].str.endswith("updated-prospectus.PDF")].iloc[0]
        self.assertEqual(fallback["selection_priority"], "P1")
        self.assertIn(
            "one_earliest_updated_prospectus_fallback_document",
            json.loads(fallback["selection_reasons_json"]),
        )
        self.assertFalse(selected["source_url"].str.endswith("updated-summary.PDF").any())
        self.assertEqual(int(coverage.iloc[0]["updated_prospectus_fallback_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["initial_prospectus_supplement_document_count"]), 0)

    def test_fallback_update_is_explicitly_insufficient_and_deterministic(self):
        candidates = pd.DataFrame(
            [
                _candidate("159720", "2022-11-23", "基金产品资料概要更新", ["product_summary"], ["contract_content_review_candidate"], "summary"),
                _candidate("159720", "2022-11-23", "更新招募说明书", ["prospectus", "prospectus_update"], ["contract_content_review_candidate"], "prospectus"),
                _candidate("159720", "2024-09-06", "基金合同更新", ["fund_contract", "contract_amendment"], ["contract_content_review_candidate"], "contract"),
            ]
        )
        selected, coverage = selector.select_documents(candidates, _coverage("159720"))
        baseline = selected[selected["baseline_selection_state"].eq("fallback_update_candidate")]
        self.assertEqual(len(baseline), 1)
        self.assertTrue(baseline.iloc[0]["source_url"].endswith("prospectus.PDF"))
        self.assertEqual(baseline.iloc[0]["selection_priority"], "P0")
        self.assertEqual(baseline.iloc[0]["benchmark_evidence_state"], "evidence_insufficient")
        self.assertFalse(bool(baseline.iloc[0]["historical_backtest_allowed"]))
        self.assertEqual(int(coverage.iloc[0]["fallback_baseline_document_count"]), 1)
        self.assertEqual(coverage.iloc[0]["review_priority"], "P0")

    def test_selects_first_full_legal_documents_after_routed_event(self):
        candidates = pd.DataFrame(
            [
                _candidate("159907", "2011-07-01", "基金合同", ["fund_contract"], ["initial_benchmark_candidate"], "initial"),
                _candidate(
                    "159907",
                    "2023-06-12",
                    "关于基金合同生效暨基金更名的公告",
                    ["fund_contract", "index_name_change"],
                    ["benchmark_change_candidate"],
                    "event",
                ),
                _candidate(
                    "159907",
                    "2023-06-12",
                    "广发国证2000交易型开放式指数证券投资基金基金合同",
                    ["fund_contract"],
                    ["contract_content_review_candidate"],
                    "post-contract",
                ),
                _candidate(
                    "159907",
                    "2023-06-12",
                    "广发国证2000交易型开放式指数证券投资基金招募说明书",
                    ["prospectus"],
                    ["contract_content_review_candidate"],
                    "post-prospectus",
                ),
            ]
        )
        selected, coverage = selector.select_documents(candidates, _coverage("159907"))
        contract = selected[selected["source_url"].str.endswith("post-contract.PDF")].iloc[0]
        prospectus = selected[selected["source_url"].str.endswith("post-prospectus.PDF")].iloc[0]
        self.assertIn(
            "one_first_post_event_fund_contract_document",
            json.loads(contract["selection_reasons_json"]),
        )
        self.assertIn(
            "one_first_post_event_prospectus_document",
            json.loads(prospectus["selection_reasons_json"]),
        )
        self.assertEqual(contract["selection_priority"], "P0")
        self.assertEqual(int(coverage.iloc[0]["post_event_fund_contract_document_count"]), 1)
        self.assertEqual(int(coverage.iloc[0]["post_event_prospectus_document_count"]), 1)

    def test_fund_rename_contract_amendment_routes_post_event_legal_documents(self):
        candidates = pd.DataFrame(
            [
                _candidate("159573", "2023-11-30", "基金合同", ["fund_contract"], ["initial_benchmark_candidate"], "initial"),
                _candidate(
                    "159573",
                    "2026-06-15",
                    "关于变更某交易型开放式指数证券投资基金基金名称并修订基金合同的公告",
                    ["contract_amendment", "fund_contract"],
                    ["contract_content_review_candidate"],
                    "rename-event",
                ),
                _candidate(
                    "159573",
                    "2026-06-15",
                    "新名称交易型开放式指数证券投资基金基金合同",
                    ["fund_contract"],
                    ["initial_benchmark_candidate"],
                    "post-contract",
                ),
                _candidate(
                    "159573",
                    "2026-06-16",
                    "新名称交易型开放式指数证券投资基金招募说明书更新",
                    ["prospectus", "prospectus_update"],
                    ["contract_content_review_candidate"],
                    "post-prospectus",
                ),
            ]
        )
        selected, _ = selector.select_documents(candidates, _coverage("159573"))
        contract = selected[selected["source_url"].str.endswith("post-contract.PDF")].iloc[0]
        prospectus = selected[selected["source_url"].str.endswith("post-prospectus.PDF")].iloc[0]
        self.assertIn(
            "one_first_post_event_fund_contract_document",
            json.loads(contract["selection_reasons_json"]),
        )
        self.assertIn(
            "one_first_post_event_prospectus_document",
            json.loads(prospectus["selection_reasons_json"]),
        )
        self.assertEqual(contract["selection_policy_version"], "benchmark_document_routing_v7")

    def test_manager_rename_does_not_route_post_event_legal_documents(self):
        candidates = pd.DataFrame(
            [
                _candidate("510050", "2004-11-01", "基金合同", ["fund_contract"], ["initial_benchmark_candidate"], "initial"),
                _candidate(
                    "510050",
                    "2024-01-01",
                    "关于基金管理人更名并修订基金合同的公告",
                    ["contract_amendment", "fund_contract"],
                    ["contract_content_review_candidate"],
                    "manager-rename",
                ),
                _candidate(
                    "510050",
                    "2024-01-02",
                    "交易型开放式指数证券投资基金基金合同",
                    ["fund_contract"],
                    ["contract_content_review_candidate"],
                    "later-contract",
                ),
            ]
        )
        selected, _ = selector.select_documents(candidates, _coverage("510050"))
        self.assertFalse(selected["source_url"].str.endswith("later-contract.PDF").any())

    def test_requires_one_candidate_set_for_every_covered_asset(self):
        candidates = pd.DataFrame(
            [_candidate("510050", "2005-02-01", "上市交易公告书", ["listing_document"], ["initial_benchmark_candidate"], "listing")]
        )
        with self.assertRaisesRegex(ValueError, "asset mismatch"):
            selector.select_documents(candidates, _coverage("510050", "159901"))


if __name__ == "__main__":
    unittest.main()
