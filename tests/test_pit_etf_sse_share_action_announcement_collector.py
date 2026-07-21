import unittest

import pandas as pd

from strategy_lab.long_hold_v4.pit_etf_sse_share_action_announcement_collector import (
    classify_action_title,
    build_registry_candidates,
    extract_action_date_candidates,
    extract_document_dates,
    extract_factor_candidates,
    match_queue_events,
    parse_announcements,
)


class TestSseEtfShareActionAnnouncementCollector(unittest.TestCase):
    def test_classifies_action_and_document_role(self):
        self.assertEqual(classify_action_title("某ETF份额拆分结果的公告"), ("share_split", "result"))
        self.assertEqual(classify_action_title("某ETF实施基金份额合并及相关业务安排"), ("share_merger", "implementation"))
        self.assertEqual(classify_action_title("某ETF基金份额折算日公告"), ("share_conversion", "announcement"))
        self.assertEqual(classify_action_title("基金份额持有人大会公告"), (None, "irrelevant"))

    def test_parses_explicit_before_after_factor_and_dates(self):
        text = "权益登记日为2020年8月13日。每1份基金份额拆分为5份，份额拆分日为2020年8月14日。"
        factors = extract_factor_candidates(text)
        self.assertTrue(any(abs(row["factor"] - 5.0) < 1e-12 for row in factors))
        self.assertEqual(extract_document_dates(text), ["2020-08-13", "2020-08-14"])

    def test_extracts_action_date_from_date_first_and_action_first_wording(self):
        date_first = extract_action_date_candidates(
            "本基金已于2020年8月14日（份额拆分日）进行了基金份额拆分。",
            "share_split",
        )
        action_first = extract_action_date_candidates(
            "份额合并日（权益登记日当日，即2025年3月7日），本基金实施合并。",
            "share_merger",
        )
        fallback = extract_action_date_candidates(
            "权益登记日（拆分当日，即2026年5月21日），登记在册份额按比例拆分。",
            "share_split",
        )
        performed = extract_action_date_candidates(
            "本基金已于2022年3月29日进行了基金份额拆分。",
            "share_split",
        )
        registration_day = extract_action_date_candidates(
            "管理人以2023年2月28日为本基金权益登记日，对该日在登记机构登记在册的基金份额进行了份额折算。",
            "share_conversion",
        )
        self.assertEqual(date_first[0]["date"], "2020-08-14")
        self.assertEqual(action_first[0]["date"], "2025-03-07")
        self.assertEqual(fallback[0]["date"], "2026-05-21")
        self.assertEqual(performed[0]["date"], "2022-03-29")
        self.assertEqual(registration_day[0]["date"], "2023-02-28")
        self.assertEqual(registration_day[0]["method"], "registration_day_declared_before_completed_action")

    def test_parses_ratio_in_both_split_and_merger_wording(self):
        split = extract_factor_candidates("基金份额拆分比例为1:5。")
        merger = extract_factor_candidates("基金份额合并比例为4：1。")
        scalar = extract_factor_candidates("登记结算公司按合并比例 0.25 实施合并，每 4 份基金份额变更为 1 份。")
        spaced = extract_factor_candidates("本 基 金 基 金 份 额 拆 分 比 例 为 1:2。")
        ratio_first = extract_factor_candidates("基金份额将按1：3的拆分比例进行拆分。")
        self.assertTrue(any(abs(row["factor"] - 5.0) < 1e-12 for row in split))
        self.assertTrue(any(abs(row["factor"] - 0.25) < 1e-12 for row in merger))
        self.assertTrue(any(abs(row["factor"] - 0.25) < 1e-12 for row in scalar))
        self.assertTrue(any(abs(row["factor"] - 2.0) < 1e-12 for row in spaced))
        self.assertTrue(any(abs(row["factor"] - 3.0) < 1e-12 for row in ratio_first))

    def test_parse_announcements_deduplicates_urls_and_keeps_primary_fields(self):
        artifact = {
            "asset": "510180",
            "responses": [
                {
                    "rows": [
                        {
                            "TITLE": "180ETF份额拆分结果公告",
                            "SSEDATE": "2009-05-20",
                            "URL": "/disclosure/example.pdf",
                            "BULLETIN_TYPE_DESC": "基金运作(基金)",
                            "ORG_BULLETIN_TYPE_DESC": "其他事项",
                        }
                    ]
                },
                {
                    "rows": [
                        {
                            "TITLE": "180ETF份额拆分结果公告",
                            "SSEDATE": "2009-05-20",
                            "URL": "/disclosure/example.pdf",
                        }
                    ]
                },
            ],
        }
        frame = parse_announcements([artifact], {"510180": "180ETF"})
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["action_type"], "share_split")
        self.assertEqual(frame.iloc[0]["document_role"], "result")
        self.assertEqual(frame.iloc[0]["source_url"], "https://www.sse.com.cn/disclosure/example.pdf")

    def test_match_prefers_result_and_compatible_action_near_event(self):
        queue = pd.DataFrame(
            [
                {
                    "asset": "510180",
                    "asset_name": "180ETF",
                    "price_effective_date": pd.Timestamp("2013-12-23"),
                    "inferred_factor": 0.25,
                    "observed_price_ratio": 0.259,
                }
            ]
        )
        announcements = pd.DataFrame(
            [
                {
                    "asset": "510180",
                    "announcement_date": pd.Timestamp("2013-12-12"),
                    "action_type": "share_merger",
                    "document_role": "implementation",
                    "announcement_title": "实施份额合并公告",
                    "source_url": "https://www.sse.com.cn/implementation.pdf",
                },
                {
                    "asset": "510180",
                    "announcement_date": pd.Timestamp("2013-12-23"),
                    "action_type": "share_merger",
                    "document_role": "result",
                    "announcement_title": "份额合并结果公告",
                    "source_url": "https://www.sse.com.cn/result.pdf",
                },
            ]
        )
        result = match_queue_events(queue, announcements)
        self.assertEqual(result.iloc[0]["source_url"], "https://www.sse.com.cn/result.pdf")
        self.assertEqual(result.iloc[0]["announcement_distance_days"], 0)

    def test_match_reports_no_candidate_outside_review_window(self):
        queue = pd.DataFrame(
            [
                {
                    "asset": "510230",
                    "asset_name": "金融ETF",
                    "price_effective_date": pd.Timestamp("2020-08-17"),
                    "inferred_factor": 5.0,
                    "observed_price_ratio": 4.7,
                }
            ]
        )
        announcements = pd.DataFrame(
            [
                {
                    "asset": "510230",
                    "announcement_date": pd.Timestamp("2019-01-01"),
                    "action_type": "share_split",
                    "document_role": "result",
                    "announcement_title": "旧公告",
                    "source_url": "https://www.sse.com.cn/old.pdf",
                }
            ]
        )
        result = match_queue_events(queue, announcements)
        self.assertEqual(result.iloc[0]["match_status"], "no_candidate_in_120d_window")

    def test_registry_candidates_keep_official_factor_but_remain_unapproved(self):
        matches = pd.DataFrame(
            [
                {
                    "asset": "510100",
                    "candidate_action_type": "share_merger",
                    "best_action_event_date": pd.Timestamp("2025-03-07"),
                    "price_effective_date": pd.Timestamp("2025-03-10"),
                    "best_parsed_factor": 0.49719902,
                    "candidate_announcement_date": pd.Timestamp("2025-03-10"),
                    "candidate_title": "份额合并结果公告",
                    "source_url": "https://www.sse.com.cn/result.pdf",
                    "pdf_status": "success",
                    "pdf_path": "raw/result.pdf",
                    "pdf_sha256": "a" * 64,
                    "text_path": "raw/result.txt",
                    "text_sha256": "b" * 64,
                    "inferred_factor": 0.5,
                    "observed_price_ratio": 0.5,
                    "action_event_date_distance_days": 3,
                    "evidence_review_status": "official_pdf_corrects_heuristic_review_required",
                }
            ]
        )
        result = build_registry_candidates(matches)
        self.assertAlmostEqual(result.iloc[0]["shares_after_per_share_before"], 0.49719902)
        self.assertFalse(result.iloc[0]["historical_backtest_allowed"])


if __name__ == "__main__":
    unittest.main()
