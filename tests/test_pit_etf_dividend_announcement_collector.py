import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_announcement_collector import (
    build_event_document_edges,
    extract_cash_candidates,
    extract_ex_date_candidates,
    extract_pay_date_candidates,
    extract_record_date_candidates,
    is_distribution_title,
    parse_document_text,
)


class TestEtfDividendAnnouncementCollector(unittest.TestCase):
    def test_distribution_title_filter(self):
        self.assertTrue(is_distribution_title("上证50ETF利润分配公告"))
        self.assertTrue(is_distribution_title("某ETF分红公告"))
        self.assertFalse(is_distribution_title("某ETF基金份额拆分结果公告"))

    def test_parses_legacy_distribution_wording(self):
        text = (
            "每10份基金份额派发现金收益0.37元。"
            "权益登记日：2006年11月15日。除息日：2006年11月16日。"
            "红利发放日：2006年11月21日。"
        )
        parsed = parse_document_text(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.037) < 1e-12 for row in parsed["cash_candidates"]))
        self.assertEqual(extract_record_date_candidates(text)[0]["date"], "2006-11-15")
        self.assertEqual(extract_ex_date_candidates(text)[0]["date"], "2006-11-16")
        self.assertEqual(extract_pay_date_candidates(text)[0]["date"], "2006-11-21")

    def test_parses_modern_table_distribution_wording(self):
        text = (
            "本次分红方案（单位：人民币元/10份基金份额）0.85。"
            "权益登记日2025年11月14日（场内）。除息日2025年11月17日。"
            "现金红利发放日2025年11月19日（场内）。"
        )
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.085) < 1e-12 for row in cash))
        self.assertEqual(extract_ex_date_candidates(text)[0]["date"], "2025-11-17")
        self.assertEqual(extract_pay_date_candidates(text)[0]["date"], "2025-11-19")

    def test_parses_pdf_table_with_amount_between_wrapped_unit_lines(self):
        text = "本次分红方案（单位：元/10\n0.11\n份基金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.011) < 1e-12 for row in cash))

    def test_parses_pdf_table_with_amount_after_wrapped_share_unit(self):
        text = "本次分红方案（单位：元/10份\n0.39\n基金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.039) < 1e-12 for row in cash))

    def test_parses_table_amount_after_partially_wrapped_fund_share_label(self):
        text = "本次分红方案（单位：元/10 份基\n0.1750\n金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.0175) < 1e-12 for row in cash))

    def test_parses_table_amount_on_same_line_after_partial_unit(self):
        text = "本次分红方案（单位：元/10份基 0.76\n金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.076) < 1e-12 for row in cash))

    def test_parses_fund_dividend_table_when_amount_precedes_wrapped_denominator(self):
        text = "本次基金分红方案 0.030\n（单位：元/10 份基\n金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.003) < 1e-12 for row in cash))

    def test_parses_table_when_amount_sits_between_yuan_and_wrapped_denominator(self):
        text = "本次基金分红方案（单位：元\n0.030\n/10份基金份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.003) < 1e-12 for row in cash))

    def test_parses_amount_on_following_unit_line(self):
        text = "本次分红方案\n（单位：元/10 份基金 0.085\n份额）"
        cash = extract_cash_candidates(text)
        self.assertTrue(any(abs(row["cash_per_share"] - 0.0085) < 1e-12 for row in cash))

    def test_event_document_window_does_not_match_remote_announcements(self):
        queue = pd.DataFrame([{"asset": "510050", "source_event_date": pd.Timestamp("2025-12-17")}])
        announcements = pd.DataFrame(
            [
                {"asset": "510050", "announcement_date": pd.Timestamp("2025-12-10"), "source_url": "near.pdf"},
                {"asset": "510050", "announcement_date": pd.Timestamp("2024-11-25"), "source_url": "old.pdf"},
            ]
        )
        edges = build_event_document_edges(queue, announcements)
        self.assertEqual(edges["source_url"].tolist(), ["near.pdf"])


if __name__ == "__main__":
    unittest.main()
