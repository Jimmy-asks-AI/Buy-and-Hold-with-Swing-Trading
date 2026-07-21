import json
import unittest

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_benchmark_asset_candidate_reconciler as reconciler


def _row(
    *,
    key,
    role,
    reference_type,
    names,
    codes,
    performance,
    parse_status="tracked_index_candidate_review_required",
    available_date=None,
):
    reasons = []
    baseline_state = "not_baseline"
    if role == "baseline":
        baseline_state = "preferred_initial_legal_candidate"
        reasons = ["one_preferred_initial_legal_document"]
    elif role == "prospectus":
        reasons = ["one_initial_prospectus_supplement_document"]
    elif role == "listing":
        reasons = ["one_canonical_listing_context_document"]
    elif role == "listing_baseline":
        baseline_state = "listing_only_initial_candidate"
        reasons = ["one_listing_only_initial_document", "one_canonical_listing_context_document"]
    elif role == "updated_prospectus":
        reasons = ["one_earliest_updated_prospectus_fallback_document"]
    row = {
        "asset": "510050",
        "asset_name": "50ETF",
        "exchange": "SSE",
        "document_key": key * 64,
        "selection_reasons_json": json.dumps(reasons),
        "baseline_selection_state": baseline_state,
        "reference_type_candidate": reference_type,
        "index_name_candidates_json": json.dumps(names),
        "index_code_candidates_json": json.dumps(codes),
        "performance_benchmark_candidates_json": json.dumps(performance),
        "parse_status": parse_status,
    }
    if available_date is not None:
        row["available_date"] = available_date
    return row


class BenchmarkAssetCandidateReconcilerTests(unittest.TestCase):
    def test_cross_document_name_and_code_agreement_stays_review_only(self):
        rows = pd.DataFrame(
            [
                _row(key="a", role="baseline", reference_type="tracked_index", names=["上证50指数"], codes=[], performance=["上证50指数收益率"]),
                _row(key="b", role="prospectus", reference_type="tracked_index", names=["上证50指数"], codes=["000016"], performance=["上证50指数收益率"]),
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2005-02-23")
        self.assertEqual(result["canonical_index_name_candidate"], "上证50指数")
        self.assertEqual(result["canonical_index_code_candidate"], "000016")
        self.assertEqual(result["name_agreement_document_count"], 2)
        self.assertEqual(result["initial_reconciliation_status"], "cross_document_name_code_candidate_review_required")
        self.assertFalse(result["historical_backtest_allowed"])

    def test_missing_index_code_remains_insufficient(self):
        rows = pd.DataFrame(
            [_row(key="a", role="baseline", reference_type="tracked_index", names=["上证50指数"], codes=[], performance=["上证50指数收益率"])]
        )
        result = reconciler.reconcile_asset(rows, list_date="2005-02-23")
        self.assertEqual(result["initial_reconciliation_status"], "index_name_reconciled_code_missing")

    def test_non_index_conflict_is_not_scored_away(self):
        rows = pd.DataFrame(
            [
                _row(key="a", role="baseline", reference_type="tracked_index", names=["上证50指数"], codes=["000016"], performance=["上证50指数收益率"]),
                _row(key="b", role="prospectus", reference_type="non_index_reference", names=[], codes=[], performance=["活期存款利率"]),
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2005-02-23")
        self.assertEqual(result["initial_reconciliation_status"], "reference_type_conflict_review_required")

    def test_prospectus_can_recover_unavailable_baseline_as_candidate(self):
        rows = pd.DataFrame(
            [
                _row(key="a", role="baseline", reference_type="unknown", names=[], codes=[], performance=[], parse_status="ocr_required"),
                _row(key="b", role="prospectus", reference_type="tracked_index", names=["中证500指数"], codes=["000905"], performance=["中证500指数收益率"]),
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2013-09-16")
        self.assertEqual(result["reference_type_support_source"], "prospectus_supplement_recovery")
        self.assertEqual(result["canonical_index_code_candidate"], "000905")
        self.assertEqual(result["initial_reconciliation_status"], "single_document_name_code_candidate_review_required")

    def test_unavailable_baseline_without_supplement_remains_unresolved(self):
        rows = pd.DataFrame(
            [
                _row(
                    key="a",
                    role="baseline",
                    reference_type="unknown",
                    names=[],
                    codes=[],
                    performance=[],
                    parse_status="document_collection_failed",
                )
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2021-10-13")
        self.assertEqual(result["reference_type_candidate"], "unknown")
        self.assertEqual(
            result["initial_reconciliation_status"],
            "baseline_document_unavailable_supplement_unresolved",
        )

    def test_page_number_suffix_does_not_create_index_code_conflict(self):
        rows = pd.DataFrame(
            [
                _row(
                    key="a",
                    role="baseline",
                    reference_type="tracked_index",
                    names=["国证石油天然气指数"],
                    codes=["399439", "3994398"],
                    performance=["国证石油天然气指数收益率"],
                )
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2026-01-01")
        self.assertEqual(result["canonical_index_code_candidate"], "399439")
        self.assertNotEqual(result["initial_reconciliation_status"], "index_code_conflict_review_required")

    def test_performance_clause_quality_breaks_definition_text_tie(self):
        rows = pd.DataFrame(
            [
                _row(
                    key="a",
                    role="baseline",
                    reference_type="commodity_spot_reference",
                    names=[],
                    codes=[],
                    performance=[
                        "上海金集中定价合约的午盘基准价格收益率",
                        "黄金价格，使用黄金现货合约或基金合同约定的方式进行申购赎回，并在证券交易所上市交易的开放式基金，与本基金的投资目标",
                    ],
                    parse_status="commodity_spot_reference_candidate_review_required",
                )
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2021-11-30")
        self.assertEqual(
            result["canonical_performance_benchmark_candidate"],
            "上海金集中定价合约的午盘基准价格收益率",
        )
        self.assertEqual(
            result["initial_reconciliation_status"],
            "commodity_spot_reference_candidate_review_required",
        )

    def test_updated_prospectus_recovery_cannot_backdate_candidate_scope(self):
        rows = pd.DataFrame(
            [
                _row(
                    key="a",
                    role="listing_baseline",
                    reference_type="unknown",
                    names=[],
                    codes=[],
                    performance=[],
                    parse_status="ambiguous_candidate_review_required",
                    available_date="2006-05-15",
                ),
                _row(
                    key="b",
                    role="updated_prospectus",
                    reference_type="tracked_index",
                    names=["上证180指数"],
                    codes=["000010"],
                    performance=["上证180指数"],
                    available_date="2006-11-27",
                ),
            ]
        )
        result = reconciler.reconcile_asset(rows, list_date="2006-05-18")
        self.assertEqual(result["reference_type_support_source"], "updated_prospectus_fallback_recovery")
        self.assertEqual(result["tradable_scope_from_candidate"], "2006-11-27")
        self.assertFalse(result["historical_backtest_allowed"])


if __name__ == "__main__":
    unittest.main()
