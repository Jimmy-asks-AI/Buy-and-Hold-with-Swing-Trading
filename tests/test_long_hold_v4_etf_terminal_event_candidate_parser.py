import json

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_terminal_event_candidate_parser as parser


def test_direct_cash_parser_handles_per_share_and_per_hundred_units():
    per_hundred = parser.compact_text(
        "本次每百份百强ETF份额实际发放资金为259.840元。"
    )
    candidates = parser.extract_direct_cash_candidates(per_hundred)
    assert len(candidates) == 1
    assert candidates[0]["denominator_shares"] == 100.0
    assert candidates[0]["cash_per_share"] == 2.5984

    per_share = parser.compact_text(
        "本次每份基金份额可获分配清算资金为人民币1.06015元。"
    )
    candidates = parser.extract_direct_cash_candidates(per_share)
    assert len(candidates) == 1
    assert candidates[0]["cash_per_share"] == 1.06015


def test_direct_cash_parser_prefers_current_amount_over_restatement_of_prior_distribution():
    text = parser.compact_text(
        "管理人于2023年8月1日进行了第一次剩余财产分配，"
        "每份基金份额实际分配1.24748元。"
        "本次分配权益登记日为2023年9月1日，"
        "本次每份基金份额实际分配0.00294元。"
    )
    candidates = parser.extract_direct_cash_candidates(text)
    value, status = parser._selected_numeric(candidates, "cash_per_share")
    assert status == "unique"
    assert value == 0.00294
    assert sorted(candidate["rank"] for candidate in candidates) == [0, 5]


def test_total_cash_parser_prefers_current_distributable_amount():
    text = parser.compact_text(
        "应分配剩余财产共计人民币10,043,719.33元，"
        "扣除未变现资产，本次清算可供分配剩余财产为人民币10,019,496.85元。"
    )
    candidates = parser.extract_total_cash_candidates(text)
    value, status = parser._selected_numeric(candidates, "total_cash", tolerance=0.005)
    assert status == "unique"
    assert value == 10_019_496.85


def test_liquidation_share_parser_downranks_initial_offering_shares():
    text = parser.compact_text(
        "截至最后运作日，基金份额总额8,777,361.00份，基金份额净值0.9353元。"
        "本基金募集认购基金份额总额为310,777,361.00份，于上市交易日挂牌。"
    )
    candidates = parser.extract_share_candidates(text)
    value, status = parser._selected_numeric(candidates, "shares", tolerance=0.005)
    assert status == "unique"
    assert value == 8_777_361.0


def test_cash_dates_and_final_distribution_semantics_are_separate():
    text = parser.compact_text(
        "本次剩余财产分配权益登记日为2024年8月19日，"
        "本次剩余财产分配资金发放日为2024年8月20日。"
        "本基金剩余财产分配完毕后将办理退出登记。"
    )
    dates = parser.extract_date_candidates(text)
    record, record_status = parser._selected_date(dates["record_date"])
    pay, pay_status = parser._selected_date(dates["pay_date"])
    semantics = parser.extract_semantics(text)
    assert record_status == pay_status == "unique"
    assert record == pd.Timestamp("2024-08-19")
    assert pay == pd.Timestamp("2024-08-20")
    assert semantics["remaining_property_fully_distributed"]
    assert semantics["exit_registration_announced"]
    assert not semantics["additional_distribution_expected"]

    pending = parser.extract_semantics(
        parser.compact_text("本次分配后仍有未变现资产，将于变现后再次进行分配。")
    )
    assert pending["additional_distribution_expected"]


def test_build_cash_candidate_never_promotes_regex_output():
    parsed = pd.DataFrame(
        [
            {
                "asset": "510700",
                "asset_name": "ETF",
                "exchange": "SSE",
                "announcement_title": "清算资金发放公告",
                "announcement_date": "2015-12-12",
                "document_role_candidates_json": json.dumps(["cash_distribution"]),
                "direct_cash_candidates_json": json.dumps(
                    [{"rank": 0, "cash_per_share": 2.5984}]
                ),
                "total_cash_candidates_json": "[]",
                "record_date_candidates_json": json.dumps(
                    [{"rank": 0, "date": "2015-12-15"}]
                ),
                "ex_date_candidates_json": "[]",
                "pay_date_candidates_json": json.dumps(
                    [{"rank": 0, "date": "2015-12-18"}]
                ),
                "semantics_json": json.dumps(
                    {
                        "additional_distribution_expected": True,
                        "remaining_property_fully_distributed": False,
                        "exit_registration_announced": True,
                        "fund_contract_terminated": True,
                    }
                ),
                "source_url": "https://www.sse.com.cn/example.pdf",
                "pdf_path": "example.pdf",
                "pdf_sha256": "a" * 64,
                "text_path": "example.txt",
                "text_sha256": "b" * 64,
                "text_status": "success",
                "available_date": "2015-12-12",
            }
        ]
    )
    candidate = parser.build_cash_candidates(parsed).iloc[0]
    assert candidate["candidate_status"] == "field_candidates_complete_requires_independent_validation"
    assert not bool(candidate["economic_extinguishment_candidate"])
    assert not bool(candidate["historical_backtest_allowed"])
    assert not bool(candidate["model_promotion_allowed"])
