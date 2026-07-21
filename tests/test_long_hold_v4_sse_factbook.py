import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_sse_factbook_status_collector import (
    extract_status_reference_events,
    _parse_reference_line,
    _parse_restoration_line,
)


class SseFactbookStatusCollectorTests(unittest.TestCase):
    def parse(self, line: str):
        return _parse_restoration_line(
            line,
            edition=2012,
            event_year=2011,
            page_number=117,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="a" * 64,
        )

    def test_parses_normal_restoration(self):
        row = self.parse("1 600057 *ST夏新 2011-8-23 2011-8-29 象屿股份")
        self.assertIsNotNone(row)
        self.assertEqual(row["asset"], "600057")
        self.assertEqual(row["execution_status"], "normal")
        self.assertEqual(row["restored_name"], "象屿股份")
        self.assertEqual(row["effective_date"], pd.Timestamp("2011-08-29"))

    def test_preserves_st_status_after_restoration(self):
        row = self.parse("2 600681 S*ST万鸿 2011-9-1 2011-9-8 ST万鸿")
        self.assertIsNotNone(row)
        self.assertEqual(row["execution_status"], "risk_warning")
        self.assertTrue(row["is_st"])

    def test_handles_a_and_b_share_columns(self):
        row = self.parse("5 600094 *ST华源 900940 *ST华源B 2011-9-28 2011-10-11 ST华源")
        self.assertIsNotNone(row)
        self.assertEqual(row["asset"], "600094")
        self.assertEqual(row["pre_restoration_name"], "*ST华源")
        self.assertEqual(row["execution_status"], "risk_warning")

    def test_parses_legacy_single_date_table(self):
        row = self.parse("600633 白猫股份 2002-4-18")
        self.assertIsNotNone(row)
        self.assertEqual(row["effective_date"], pd.Timestamp("2002-04-18"))
        self.assertEqual(row["restored_name"], "白猫股份")
        self.assertEqual(row["availability_basis"], "effective_date_proxy_no_announcement_column")

    def test_parses_single_date_with_restored_name_column(self):
        row = self.parse("1 600556 *ST北生 2013-2-8 ST北生")
        self.assertIsNotNone(row)
        self.assertEqual(row["pre_restoration_name"], "*ST北生")
        self.assertEqual(row["restored_name"], "ST北生")
        self.assertEqual(row["execution_status"], "risk_warning")

    def test_rejects_non_data_lines(self):
        self.assertIsNone(self.parse("序号 A股代码 A股简称 恢复上市公告刊登日期"))

    def test_parses_independent_risk_warning_reference(self):
        row = _parse_reference_line(
            "1 600381 贤成矿业 2013-3-6 ST贤成",
            label="实施其他风险警示",
            edition=2014,
            event_year=2013,
            page_number=130,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="b" * 64,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["execution_status"], "risk_warning")
        self.assertFalse(row["used_in_reconciliation"])

    def test_parses_legacy_removal_reference(self):
        row = _parse_reference_line(
            "600167 ST沈新开 沈阳新开 2003-2-12",
            label="取消特别处理",
            edition=2003,
            event_year=2003,
            page_number=62,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="c" * 64,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["execution_status"], "normal")
        self.assertEqual(row["restored_name"], "沈阳新开")

    def test_parses_cancellation_columns_in_implementation_then_announcement_order(self):
        row = _parse_reference_line(
            "600212 *ST江泉 江泉实业 2009-7-3 2009-7-2",
            label="取消特别处理",
            event_subclass="cancel_special_treatment",
            edition=2010,
            event_year=2009,
            page_number=216,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="e" * 64,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["effective_date"], pd.Timestamp("2009-07-03"))
        self.assertEqual(row["announcement_date"], pd.Timestamp("2009-07-02"))
        self.assertEqual(row["execution_status"], "normal")
        self.assertTrue(row["binary_state_change"])

    def test_preserves_st_when_only_delisting_warning_is_removed(self):
        row = _parse_reference_line(
            "2 600234 *ST天龙 2010-06-08 ST天龙 2010-06-09 2010-06-08",
            label="取消特别处理",
            event_subclass="remove_delisting_warning",
            edition=2011,
            event_year=2010,
            page_number=128,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="f" * 64,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["effective_date"], pd.Timestamp("2010-06-09"))
        self.assertEqual(row["announcement_date"], pd.Timestamp("2010-06-08"))
        self.assertEqual(row["restored_name"], "ST天龙")
        self.assertEqual(row["execution_status"], "risk_warning")
        self.assertFalse(row["binary_state_change"])

    def test_tracks_subsection_semantics_across_table_rows(self):
        frame, _ = extract_status_reference_events(
            page_texts=[
                "2011年取消特别处理的公司\n"
                "(二)撤销退市风险警示及实施其他特别处理\n"
                "1 600145 *ST国创 后A股简称 2011-10-28 2011-10-27\n"
            ],
            edition=2012,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="1" * 64,
            as_of="2026-07-17",
        )
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["event_subclass"], "remove_delisting_warning_keep_other_warning")
        self.assertEqual(row["execution_status"], "risk_warning")
        self.assertEqual(row["effective_date"], pd.Timestamp("2011-10-28"))
        self.assertEqual(row["announcement_date"], pd.Timestamp("2011-10-27"))
        self.assertFalse(bool(row["binary_state_change"]))

    def test_preserves_legacy_terminal_announcement_anomaly_without_future_availability(self):
        row = _parse_reference_line(
            "600709 ST生态 2003-5-24 2003-5-23",
            label="终止上市",
            edition=2003,
            event_year=2003,
            page_number=62,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="2" * 64,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["effective_date"], pd.Timestamp("2003-05-23"))
        self.assertEqual(row["announcement_date"], pd.Timestamp("2003-05-24"))
        self.assertEqual(row["available_date"], pd.Timestamp("2003-05-23"))
        self.assertEqual(
            row["availability_basis"],
            "official_terminal_announcement_after_effective_capped_to_effective",
        )

    def test_status_section_stops_before_bond_chapter(self):
        frame, _ = extract_status_reference_events(
            page_texts=[
                "2013年撤销风险警示的公司\n"
                "1 600001 ST样本 样本股份 2013-1-2\n"
                "债券市场\n"
                "2013年公司债券概貌\n"
                "1 600999 债券样本 2013-2-3\n"
            ],
            edition=2014,
            source_url="https://www.sse.com.cn/factbook.pdf",
            source_sha256="d" * 64,
            as_of="2026-07-17",
        )
        self.assertEqual(frame["asset"].tolist(), ["600001"])


if __name__ == "__main__":
    unittest.main()
