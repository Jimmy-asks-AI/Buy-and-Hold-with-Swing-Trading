import json
import unittest

from strategy_lab.long_hold_v4 import pit_etf_benchmark_event_candidate_validator as validator


def _classify(
    text,
    title="公告",
    *,
    announcement_date="2024-01-02",
    available_date="2024-01-03",
    text_status="success",
):
    return validator.classify_event_document(
        text=text,
        title=title,
        announcement_date=announcement_date,
        available_date=available_date,
        text_status=text_status,
    )


class BenchmarkEventCandidateValidatorTests(unittest.TestCase):
    def test_explicit_index_replacement_stays_chain_review_candidate(self):
        result = _classify(
            "自2023年6月12日起，本基金标的指数将变更为国证2000指数。",
            "基金合同生效暨基金更名公告",
            announcement_date="2023-06-12",
            available_date="2023-06-13",
        )
        self.assertEqual(result["event_class"], "index_replacement_candidate")
        self.assertEqual(result["new_index_name_candidate"], "国证2000指数")
        self.assertEqual(result["event_effective_date_candidate"], "2023-06-12")
        self.assertEqual(result["observable_from_date_candidate"], "2023-06-13")
        self.assertEqual(result["validation_status"], "event_field_completion_required")
        self.assertFalse(result["historical_backtest_allowed"])

    def test_index_name_pair_and_unchanged_methodology_is_metadata_event(self):
        result = _classify(
            "自2024年10月28日起，标的指数名称由原“中证100指数”变更为“中证A100指数”。"
            "除指数名称外，指数编制方案其余部分不变。",
            "变更基金名称及标的指数名称的公告",
            available_date="2024-10-18",
        )
        self.assertEqual(result["event_class"], "index_identity_metadata_change_candidate")
        self.assertEqual(result["old_index_name_candidate"], "中证100指数")
        self.assertEqual(result["new_index_name_candidate"], "中证A100指数")
        self.assertEqual(result["event_effective_date_candidate"], "2024-10-28")
        self.assertFalse(result["document_scope_no_identity_change"])

    def test_performance_benchmark_pair_requires_chain_review(self):
        result = _classify(
            "自2026年4月23日起，本基金业绩比较基准要素由“富时中国A股自由现金流聚焦指数收益率”"
            "更新为“富时中国A股自由现金流聚焦指数（FTSE China A Free Cash Flow Focus Index）收益率”。",
            "更新业绩比较基准内容并修改基金合同的公告",
            announcement_date="2026-04-23",
            available_date="2026-04-24",
        )
        self.assertEqual(result["event_class"], "performance_benchmark_change_candidate")
        self.assertIn("富时中国", result["old_performance_benchmark_candidate"])
        self.assertIn("FTSE", result["new_performance_benchmark_candidate"])
        self.assertEqual(result["validation_status"], "structured_event_candidate_chain_review_required")

    def test_index_code_new_value_is_metadata_candidate_not_replacement(self):
        result = _classify(
            "自2010年4月12日起，本基金的标的指数代码调整为399005，简称为中小板指。",
            "标的指数代码调整的提示性公告",
            announcement_date="2010-04-09",
            available_date="2010-04-12",
        )
        self.assertEqual(result["event_class"], "index_code_metadata_change_candidate")
        self.assertEqual(result["new_index_code_candidate"], "399005")
        self.assertEqual(result["event_effective_date_candidate"], "2010-04-12")

    def test_direct_fund_event_date_wins_over_provider_context_date(self):
        result = _classify(
            "指数公司自2020年6月15日起将指数名称由旧指数变更为新指数。"
            "本公司决定自2020年6月16日起对两只基金变更标的指数名称及业绩比较基准，"
            "修订后的基金合同自2020年6月16日起生效。",
            "变更基金名称及标的指数名称并修改基金合同的公告",
            announcement_date="2020-06-11",
            available_date="2020-06-12",
        )
        self.assertEqual(
            json.loads(result["effective_date_candidates_json"]),
            ["2020-06-16"],
        )
        self.assertEqual(result["event_effective_date_candidate"], "2020-06-16")

    def test_unbalanced_short_name_fragment_is_removed_from_index_identity(self):
        result = _classify(
            "自2024年10月28日起，标的指数名称由“中证100指数（简称：中证100”"
            "变更为“中证A100指数（简称：中证A100”，业绩比较基准由"
            "“中证100指数收益率”变更为“中证A100指数收益率”。",
            "变更标的指数名称并修改基金合同的公告",
            available_date="2024-10-18",
        )
        self.assertEqual(result["old_index_name_candidate"], "中证100指数")
        self.assertEqual(result["new_index_name_candidate"], "中证A100指数")

    def test_short_name_only_adjustment_is_non_identity_metadata(self):
        result = _classify(
            "本基金标的指数为中小板300成长指数，指数代码399602。"
            "自2014年1月2日起，标的指数简称由“SME300成长”调整为“中小成长”，"
            "指数全称及代码保持不变。",
            "关于标的指数简称调整及相应修改基金合同的公告",
            announcement_date="2014-01-04",
            available_date="2014-01-05",
        )
        self.assertEqual(result["event_class"], "index_short_name_metadata_change")
        self.assertEqual(result["validation_status"], "document_scope_validated_non_identity_change")
        self.assertTrue(result["document_scope_no_identity_change"])

    def test_performance_benchmark_explanation_update_is_not_a_value_change(self):
        result = _classify(
            "本次仅对业绩比较基准的设定原因、要素基本信息、计算方法、"
            "管理投资偏离业绩比较基准的方法进行补充说明。"
            "本基金的业绩比较基准仍为创业板新能源指数收益率。",
            "更新业绩比较基准相关内容并相应更新基金合同等法律文件的公告",
            announcement_date="2026-03-28",
            available_date="2026-03-29",
        )
        self.assertEqual(result["event_class"], "performance_benchmark_metadata_enrichment")
        self.assertEqual(result["validation_status"], "document_scope_validated_non_identity_change")

    def test_side_by_side_contract_revision_extracts_performance_pair(self):
        result = _classify(
            "调整业绩比较基准并修订基金合同。修订前修订后。"
            "本基金的业绩比较基准为标的指数收益率，即中证港股通科技指数（人民币）收益率。"
            "本基金的业绩比较基准为标的指数收益率，即中证港股通科技指数"
            "（经估值汇率调整）收益率。上述调整事项自本公告发布之日起生效。",
            "调整业绩比较基准并修订基金合同的公告",
            announcement_date="2023-10-19",
            available_date="2023-10-20",
        )
        self.assertEqual(
            result["old_performance_benchmark_candidate"],
            "标的指数收益率，即中证港股通科技指数(人民币)收益率",
        )
        self.assertEqual(
            result["new_performance_benchmark_candidate"],
            "标的指数收益率，即中证港股通科技指数(经估值汇率调整)收益率",
        )
        self.assertEqual(result["event_effective_date_candidate"], "2023-10-19")

    def test_interleaved_pdf_table_reconstructs_rmb_valuation_pair(self):
        result = _classify(
            "修订前 修订后\n"
            "中证港股通科技指数（人   中证港股通科技指数（经估值\n"
            "交易型开放式指数证券投资基金 民币）收益率   汇率调整）收益率\n"
            "上述调整事项自本公告发布之日起生效。",
            "调整业绩比较基准并修订基金合同的公告",
            announcement_date="2023-10-19",
            available_date="2023-10-20",
        )
        self.assertEqual(
            result["old_performance_benchmark_candidate"],
            "标的指数收益率，即中证港股通科技指数(人民币)收益率",
        )
        self.assertEqual(
            result["new_performance_benchmark_candidate"],
            "标的指数收益率，即中证港股通科技指数(经估值汇率调整)收益率",
        )

    def test_cross_fund_table_pollution_does_not_create_performance_event(self):
        result = _classify(
            "修订前修订后。"
            "本基金的业绩比较基准为中证200基金管理人应就变更标的指数召开"
            "基金份额持有人大会指数收益率。"
            "本基金的业绩比较基准为中债-本基金的业绩比较基准为中债-1-5年"
            "政策性金融债指数收益率。",
            "旗下基金根据指数基金指引修改基金合同的公告",
            announcement_date="2021-03-31",
            available_date="2021-04-01",
        )
        self.assertEqual(result["old_performance_benchmark_candidate"], "")
        self.assertEqual(result["new_performance_benchmark_candidate"], "")
        self.assertNotEqual(result["event_class"], "performance_benchmark_change_candidate")

    def test_index_licence_fee_scope_can_be_closed_for_that_document_only(self):
        result = _classify(
            "本次仅调整标的指数许可使用费，并相应修改基金合同有关费用条款。",
            "关于调整标的指数许可使用费并修订基金合同的公告",
        )
        self.assertEqual(result["event_class"], "index_licence_or_fee_change")
        self.assertEqual(result["validation_status"], "document_scope_validated_non_identity_change")
        self.assertTrue(result["document_scope_no_identity_change"])

    def test_methodology_scope_is_not_misreported_as_index_replacement(self):
        result = _classify(
            "标的指数纳入北京证券交易所股票，本基金据此修订招募说明书。",
            "关于标的指数纳入北京证券交易所股票并修订招募说明书的公告",
        )
        self.assertEqual(result["event_class"], "index_methodology_or_universe_change")
        self.assertTrue(result["document_scope_no_identity_change"])

    def test_contract_boilerplate_does_not_become_an_event(self):
        result = _classify(
            "未来若标的指数不符合要求，基金管理人可更换标的指数。现修改基金合同信息披露条款。",
            "关于修改基金合同的公告",
        )
        self.assertEqual(result["event_class"], "generic_legal_or_holder_document")
        self.assertEqual(result["validation_status"], "manual_scope_review_required")

    def test_fund_transformation_without_new_index_clause_requires_followup_document(self):
        result = _classify(
            "自2025年9月11日起，本基金由“华宝深证创新100交易型开放式指数证券投资基金”"
            "正式转型为“华宝深证100交易型开放式指数证券投资基金”。",
            "基金合同生效暨基金更名公告",
            announcement_date="2025-09-11",
            available_date="2025-09-12",
        )
        self.assertEqual(result["event_class"], "fund_transformation_benchmark_followup_required")
        self.assertEqual(result["validation_status"], "post_transformation_legal_document_required")

    def test_fund_name_change_alone_is_not_a_transformation(self):
        result = _classify(
            "基金名称由“旧名称交易型开放式指数证券投资基金”更名为“新名称交易型开放式指数证券投资基金”。",
            "基金更名并修订基金合同的公告",
        )
        self.assertNotIn("fund_transformation", result["event_types_json"])
        self.assertNotEqual(result["event_class"], "fund_transformation_benchmark_followup_required")

    def test_ocr_text_fails_closed(self):
        result = _classify("", "持有人大会决议", text_status="ocr_required")
        self.assertEqual(result["validation_status"], "blocked_text_unavailable")
        self.assertFalse(result["model_promotion_allowed"])

    def test_ocr_derived_scope_is_routed_but_not_validated(self):
        result = _classify(
            "本次大会审议并通过终止基金合同的议案，本基金进入清算程序。",
            "基金份额持有人大会决议",
            text_status="ocr_derived_unvalidated",
        )
        self.assertEqual(result["event_class"], "generic_legal_or_holder_document")
        self.assertEqual(result["validation_status"], "ocr_derived_scope_candidate_page_review_required")
        self.assertFalse(result["document_scope_no_identity_change"])
        self.assertFalse(result["historical_backtest_allowed"])


if __name__ == "__main__":
    unittest.main()
