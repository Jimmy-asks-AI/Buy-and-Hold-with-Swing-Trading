import json
import unittest

from strategy_lab.long_hold_v4 import pit_etf_benchmark_document_candidate_parser as parser


class BenchmarkDocumentCandidateParserTests(unittest.TestCase):
    def test_extracts_tracked_index_contract_clause(self):
        result = parser.parse_document_text(
            """
            四、基金的投资目标
            紧密跟踪标的指数，追求跟踪偏离度和跟踪误差最小化。
            八、标的指数
            本基金的标的指数为中证畜牧养殖产业指数。
            标的指数代码：930707。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "tracked_index")
        self.assertIn("中证畜牧养殖产业指数", json.loads(result["index_name_candidates_json"]))
        self.assertIn("930707", json.loads(result["index_code_candidates_json"]))
        self.assertFalse(result["parse_status"].startswith("validated"))

    def test_extracts_non_index_performance_benchmark(self):
        result = parser.parse_document_text(
            """
            二、基金类型：货币ETF。
            四、投资目标
            在保持基金资产低风险和高流动性的前提下，力争超越业绩比较基准。
            七、业绩比较基准
            本基金的业绩比较基准为：人民币活期存款基准利率（税后）。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "non_index_reference")
        benchmarks = json.loads(result["performance_benchmark_candidates_json"])
        self.assertIn("人民币活期存款基准利率（税后）", benchmarks)

    def test_extracts_bond_index_replication_objective(self):
        result = parser.parse_document_text(
            """
            四、基金的投资目标
            本基金通过指数化投资，争取在扣除各项费用之前获得与标的指数相似的总回报。
            八、标的指数
            本基金的标的指数为中证AAA科技创新公司债指数。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "tracked_index")

    def test_extracts_enhanced_index_separately(self):
        result = parser.parse_document_text(
            """
            国泰中证A500增强策略交易型开放式指数证券投资基金基金合同
            四、基金的投资目标
            本基金在对标的指数进行有效跟踪的基础上，通过量化策略力争获取超越业绩比较基准的投资回报。
            八、标的指数
            本基金的标的指数为中证A500指数。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "enhanced_index")

    def test_enhanced_identity_and_excess_return_resolve_without_effective_tracking_phrase(self):
        result = parser.parse_document_text(
            """
            华泰柏瑞中证500增强策略交易型开放式指数证券投资基金基金合同。
            四、基金的投资目标：利用定量模型，力争实现超越标的指数的投资收益。
            八、标的指数：本基金的标的指数为中证500指数。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "enhanced_index")

    def test_extracts_gold_spot_reference_separately(self):
        result = parser.parse_document_text(
            """
            大成上海金交易型开放式证券投资基金基金合同。
            三、基金的投资目标：紧密跟踪上海金集中定价合约SHAU的价格表现。
            本基金的业绩比较基准为上海黄金交易所上海金集中定价合约SHAU午盘基准价格收益率。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "commodity_spot_reference")
        self.assertIn("上海", result["performance_benchmark_candidates_json"])

    def test_gold_legal_identity_and_explicit_benchmark_resolve_without_tight_tracking_phrase(self):
        result = parser.parse_document_text(
            "广发上海金交易型开放式证券投资基金基金合同。"
            "五、业绩比较基准：上海黄金交易所上海金集中定价合约（合约代码：SHAU）的午盘基准价格收益率。"
        )
        self.assertEqual(result["reference_type_candidate"], "commodity_spot_reference")
        self.assertEqual(
            json.loads(result["performance_benchmark_candidates_json"]),
            ["上海黄金交易所上海金集中定价合约（合约代码：SHAU）的午盘基准价格收益率"],
        )
        level_result = parser.parse_document_text(
            "易方达黄金交易型开放式证券投资基金基金合同。"
            "五、业绩比较基准：上海黄金交易所Au99.99现货实盘合约收盘价。"
        )
        self.assertEqual(level_result["reference_type_candidate"], "commodity_spot_reference")
        self.assertEqual(
            json.loads(level_result["performance_benchmark_candidates_json"]),
            ["上海黄金交易所Au99.99现货实盘合约收盘价"],
        )

    def test_derives_index_name_from_performance_benchmark_and_provider_clause(self):
        result = parser.parse_document_text(
            """
            四、投资目标：紧密跟踪标的指数表现。
            本基金的标的指数为中证指数有限公司编制并发布的沪深300指数。
            本基金的业绩比较基准为沪深300指数收益率。
            """
        )
        self.assertEqual(result["reference_type_candidate"], "tracked_index")
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["沪深300指数"])

    def test_provider_name_is_not_an_index_name_candidate(self):
        result = parser.parse_document_text(
            "标的指数：指本基金跟踪的基准指数，是由中证指数有限公司编制并发布的中证畜牧养殖产业指数。"
        )
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["中证畜牧养殖产业指数"])

    def test_extracts_chinese_name_from_bilingual_parenthetical_clause(self):
        result = parser.parse_document_text(
            "投资目标：紧密跟踪标的指数。"
            "本基金标的指数为MSCI China A 50 Connect Index（MSCI中国A50互联互通指数）。"
        )
        self.assertEqual(result["reference_type_candidate"], "tracked_index")
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["MSCI中国A50互联互通指数"])

    def test_extracts_joint_benchmark_clause_and_quoted_target_name(self):
        joint = parser.parse_document_text(
            "投资目标：紧密跟踪标的指数。"
            "本基金的标的指数及业绩比较基准为：中创400指数。"
        )
        self.assertEqual(json.loads(joint["index_name_candidates_json"]), ["中创400指数"])
        self.assertEqual(
            json.loads(joint["performance_benchmark_candidates_json"]),
            ["中创400指数"],
        )
        quoted = parser.parse_document_text("本基金被动跟踪标的指数“中创400指数”。")
        self.assertEqual(json.loads(quoted["index_name_candidates_json"]), ["中创400指数"])

    def test_discards_index_name_notes_and_short_name_labels(self):
        result = parser.parse_document_text(
            "标的指数“注：价格指数”。标的指数“简称上证央企指数”。"
            "本基金的标的指数为上证中央企业50指数。"
        )
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["上证中央企业50指数"])

    def test_extracts_specific_index_from_explicit_indexation_sentence(self):
        result = parser.parse_document_text(
            "本基金主要采取指数化投资法投资于中证海外中国互联网50指数成份股。"
        )
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["中证海外中国互联网50指数"])

    def test_performance_benchmark_discards_generic_and_currency_adjustment_prefixes(self):
        result = parser.parse_document_text(
            "本基金的业绩比较基准为标的指数收益率，即经人民币汇率调整后的国证石油天然气指数收益率。"
        )
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["国证石油天然气指数"])

        generic_result = parser.parse_document_text(
            "本基金的业绩比较基准为经估值汇率调整的标的指数收益率。"
        )
        self.assertEqual(generic_result["index_name_candidates_json"], "[]")

    def test_extracts_specific_member_of_index_series(self):
        result = parser.parse_document_text(
            "本基金的标的指数为上证风格指数系列中的上证180价值指数（指数代码为000029.SH）。"
        )
        self.assertEqual(json.loads(result["index_name_candidates_json"]), ["上证180价值指数"])

    def test_discards_explanatory_prefix_and_unbalanced_short_name_fragment(self):
        result = parser.parse_document_text(
            "本基金的业绩比较基准为中国战略新兴产业成份指数（简称：新兴成指）指数收益率。"
        )
        self.assertEqual(result["index_name_candidates_json"], "[]")

        explanatory_result = parser.parse_document_text(
            "本基金的业绩比较基准为因此选取恒生科技指数收益率。"
        )
        self.assertEqual(json.loads(explanatory_result["index_name_candidates_json"]), ["恒生科技指数"])

    def test_constituent_boilerplate_does_not_create_index_name(self):
        result = parser.parse_document_text(
            "本基金投资于国内依法发行上市的非标的指数成份股，并跟踪与分析标的指数成份股。"
        )
        self.assertEqual(result["index_name_candidates_json"], "[]")

    def test_rejects_page_number_glued_to_six_digit_index_code(self):
        result = parser.parse_document_text("指数代码：9870188。")
        self.assertEqual(result["index_code_candidates_json"], "[]")

    def test_index_code_parser_preserves_table_line_boundaries(self):
        result = parser.parse_document_text(
            "指数名称  指数简称  英文名称  英文简称  指数代码\n"
            "CSI Smart Selected 500 Value Stable Index\n"
            "目标指数代码  931587\n"
        )
        self.assertEqual(json.loads(result["index_code_candidates_json"]), ["931587"])

    def test_index_code_parser_rejects_placeholders_and_total_return_variant(self):
        result = parser.parse_document_text(
            "标的指数代码：XXXXXX。指数代码：X45。指数代码：932160。全收益指数代码：932160T。"
        )
        self.assertEqual(json.loads(result["index_code_candidates_json"]), ["932160"])

    def test_index_code_parser_normalises_known_numeric_exchange_suffix(self):
        result = parser.parse_document_text("标的指数代码：931643.CSI。指数代码：SPNCSCHN.SPI。")
        self.assertEqual(json.loads(result["index_code_candidates_json"]), ["931643", "SPNCSCHN"])

    def test_boilerplate_terms_do_not_resolve_reference_type(self):
        result = parser.parse_document_text(
            "风险提示：指数证券投资基金可能发生跟踪误差。持有人大会可变更基金投资目标。"
        )
        self.assertEqual(result["reference_type_candidate"], "unknown")
        self.assertEqual(result["index_name_candidates_json"], "[]")

    def test_effective_date_requires_change_context(self):
        result = parser.parse_document_text(
            "基金成立于2020年1月1日。基金合同修订于2023年1月1日起生效。标的指数变更事项自2024年9月6日起生效。"
        )
        self.assertEqual(json.loads(result["effective_date_candidates_json"]), ["2024-09-06"])
        baseline_result = parser.parse_document_text(
            "标的指数变更事项自2024年9月6日起生效。",
            include_effective_dates=False,
        )
        self.assertEqual(baseline_result["effective_date_candidates_json"], "[]")


if __name__ == "__main__":
    unittest.main()
