import unittest

from strategy_lab.long_hold_v4.pit_etf_cninfo_share_action_announcement_collector import (
    clean_announcement_title,
    parse_announcements,
    select_fund_identity,
)


class TestCninfoEtfShareActionAnnouncementCollector(unittest.TestCase):
    def test_selects_unique_fund_identity(self):
        rows = [
            {"code": "159907", "type": "fund", "orgId": "jjjl0000065", "zwjc": "国证2000ETF广发", "category": "ETF"},
            {"code": "159907", "type": "stock", "orgId": "gssh000001"},
        ]
        identity = select_fund_identity("159907", rows)
        self.assertEqual(identity["org_id"], "jjjl0000065")
        self.assertEqual(identity["category"], "ETF")

    def test_rejects_missing_or_ambiguous_identity(self):
        with self.assertRaisesRegex(ValueError, "ambiguous or missing"):
            select_fund_identity("159907", [])
        duplicate = [
            {"code": "159907", "type": "fund", "orgId": "a"},
            {"code": "159907", "type": "fund", "orgId": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous or missing"):
            select_fund_identity("159907", duplicate)

    def test_cleans_highlight_markup_and_parses_announcement(self):
        self.assertEqual(clean_announcement_title("基金<em>份额</em><em>拆分</em>结果公告"), "基金份额拆分结果公告")
        artifact = {
            "asset": "159907",
            "responses": [
                {
                    "rows": [
                        {
                            "announcementTitle": "基金<em>份额</em><em>拆分</em>结果公告",
                            "announcementTime": 1711900800000,
                            "adjunctUrl": "finalpage/2024-04-01/example.PDF",
                            "announcementTypeName": "基金公告",
                        }
                    ]
                }
            ],
        }
        frame = parse_announcements([artifact], {"159907": "国证2000ETF广发"})
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["document_role"], "result")
        self.assertEqual(frame.iloc[0]["action_type"], "share_split")
        self.assertEqual(frame.iloc[0]["source_url"], "https://static.cninfo.com.cn/finalpage/2024-04-01/example.PDF")


if __name__ == "__main__":
    unittest.main()
