import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_dividend_events_promoter import build_formal_events


class TestEtfDividendEventsPromoter(unittest.TestCase):
    def test_available_date_is_the_official_announcement_date(self):
        candidates = pd.DataFrame(
            [
                {
                    "asset": "159901",
                    "announcement_date": "2007-07-06",
                    "record_date": "2007-07-11",
                    "ex_date": "2007-07-11",
                    "pay_date": "2007-07-16",
                    "cash_per_share": 0.12,
                    "source_document_title": "收益分配公告",
                    "source_url": "https://static.cninfo.com.cn/finalpage/example.PDF",
                    "source_type": "regulatory_filing",
                    "pdf_sha256": "a" * 64,
                }
            ]
        )
        output = build_formal_events(candidates)
        self.assertEqual(output.loc[0, "available_date"], pd.Timestamp("2007-07-06"))
        self.assertEqual(output.loc[0, "ex_date"], pd.Timestamp("2007-07-11"))
        self.assertEqual(output.loc[0, "source_vintage"], "official_pdf_sha256:" + "a" * 64)


if __name__ == "__main__":
    unittest.main()
