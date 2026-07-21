import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_universe_events_promoter import build_formal_events


def candidate(**overrides):
    row = {
        "asset": "560150",
        "announcement_date": "2026-07-16",
        "record_date": "2026-07-20",
        "ex_date": "2026-07-21",
        "pay_date": "2026-07-24",
        "cash_per_share": 0.0067,
        "source_document_title": "2026年第三次分红公告",
        "source_url": "https://www.sse.com.cn/disclosure/fund/announcement/example.pdf",
        "source_type": "exchange_announcement",
        "pdf_sha256": "a" * 64,
        "historical_backtest_allowed": False,
    }
    row.update(overrides)
    return row


class TestEtfDividendUniverseEventsPromoter(unittest.TestCase):
    def test_available_date_is_official_announcement_date(self):
        output = build_formal_events(pd.DataFrame([candidate()]))
        self.assertEqual(output.loc[0, "available_date"], pd.Timestamp("2026-07-16"))
        self.assertEqual(output.loc[0, "ex_date"], pd.Timestamp("2026-07-21"))
        self.assertEqual(output.loc[0, "source_vintage"], "official_pdf_sha256:" + "a" * 64)

    def test_duplicate_economic_event_is_rejected(self):
        frame = pd.DataFrame(
            [
                candidate(),
                candidate(announcement_date="2026-07-17", source_document_title="提示性公告"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate economic event keys"):
            build_formal_events(frame)


if __name__ == "__main__":
    unittest.main()
