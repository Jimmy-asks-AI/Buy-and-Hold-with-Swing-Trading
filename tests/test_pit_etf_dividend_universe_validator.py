import unittest
from types import SimpleNamespace

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_universe_validator import (
    build_promotion_candidates,
    classify_incomplete_document,
    document_candidate_event_key,
)


def event_row(**overrides):
    row = {
        "asset": "510050",
        "announcement_date": "2025-01-01",
        "record_date": "2025-01-02",
        "ex_date": "2025-01-03",
        "pay_date": "2025-01-06",
        "cash_per_share": 0.01,
        "source_document_title": "分红公告",
        "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/example.pdf",
        "source_type": "exchange_announcement",
        "pdf_path": "evidence.pdf",
        "pdf_sha256": "a" * 64,
        "text_path": "evidence.txt",
        "text_sha256": "b" * 64,
    }
    row.update(overrides)
    return row


class TestEtfDividendUniverseValidator(unittest.TestCase):
    def test_policy_and_off_exchange_documents_are_not_formal_events(self):
        policy = SimpleNamespace(
            parse_status="incomplete_cash_record_date_ex_date_pay_date",
            announcement_title="关于调整收益分配原则并修改基金合同的公告",
        )
        self.assertEqual(classify_incomplete_document(policy, ""), "non_event_policy_document")

        off_exchange = SimpleNamespace(
            parse_status="incomplete_pay_date",
            announcement_title="分红公告",
        )
        self.assertEqual(
            classify_incomplete_document(off_exchange, "本次分配只适用于I类场外份额。"),
            "off_exchange_share_class_distribution",
        )

    def test_unexplained_incomplete_document_stays_blocked(self):
        row = SimpleNamespace(parse_status="incomplete_pay_date", announcement_title="ETF分红公告")
        self.assertEqual(
            classify_incomplete_document(row, "场内份额现金分配"),
            "unresolved_incomplete_distribution_document",
        )

    def test_prompt_document_requires_one_exact_cash_and_date_tuple(self):
        row = SimpleNamespace(
            asset="159922",
            cash_candidates_json='[{"cash_per_share":0.1292}]',
            record_date_candidates_json='[{"date":"2024-11-29"}]',
            ex_date_candidates_json='[{"date":"2024-12-02"}]',
            pay_date_candidates_json=(
                '[{"date":"2024-12-03","method":"cash"},'
                '{"date":"2024-12-03","method":"pay"}]'
            ),
        )
        self.assertEqual(
            document_candidate_event_key(row),
            ("159922", "2024-11-29", "2024-12-02", "2024-12-03", 0.1292),
        )
        row.cash_candidates_json = '[{"cash_per_share":0.1292},{"cash_per_share":0.13}]'
        self.assertIsNone(document_candidate_event_key(row))

    def test_promotion_union_keeps_legacy_and_only_new_economic_events(self):
        legacy = pd.DataFrame([event_row()])
        complete = pd.DataFrame(
            [
                event_row(announcement_title="重复提示公告", announcement_date="2025-01-02"),
                event_row(
                    asset="560150",
                    announcement_title="2026年第三次分红公告",
                    announcement_date="2026-07-16",
                    record_date="2026-07-20",
                    ex_date="2026-07-21",
                    pay_date="2026-07-24",
                    cash_per_share=0.0067,
                ),
            ]
        ).drop(columns=["source_document_title"])
        combined, extras = build_promotion_candidates(legacy, complete)
        self.assertEqual(len(combined), 2)
        self.assertEqual(len(extras), 1)
        self.assertEqual(extras.iloc[0]["asset"], "560150")
        self.assertEqual(
            combined.set_index("asset").loc["560150", "source_document_title"],
            "2026年第三次分红公告",
        )
        self.assertTrue(combined["historical_backtest_allowed"].eq(False).all())


if __name__ == "__main__":
    unittest.main()
