import json
from types import SimpleNamespace

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_benchmark_history_chain_candidate_validator as validator


def _event(**overrides):
    values = {
        "event_types_json": json.dumps(["index_name_change"]),
        "old_index_name_candidate": "中证100指数",
        "new_index_name_candidate": "中证A100指数",
        "old_index_code_candidate": "",
        "new_index_code_candidate": "",
        "old_performance_benchmark_candidate": "",
        "new_performance_benchmark_candidate": "",
        "event_effective_date_candidate": "2024-10-28",
        "effective_date_candidates_json": json.dumps(["2024-10-28"]),
        "observable_from_date_candidate": "2024-10-28",
        "available_date": "2024-10-18",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _post(*, names=None, codes=None, performance=None):
    return pd.DataFrame(
        [
            {
                "document_key": "a" * 64,
                "selection_reasons_json": json.dumps(["one_first_post_event_fund_contract_document"]),
                "index_name_candidates_json": json.dumps(names or []),
                "index_code_candidates_json": json.dumps(codes or []),
                "performance_benchmark_candidates_json": json.dumps(performance or []),
            }
        ]
    )


def _post_pair(*, names=None, performance=None):
    return pd.DataFrame(
        [
            {
                "document_key": "a" * 64,
                "selection_reasons_json": json.dumps(
                    ["one_first_post_event_fund_contract_document"]
                ),
                "index_name_candidates_json": json.dumps(names or []),
                "index_code_candidates_json": "[]",
                "performance_benchmark_candidates_json": json.dumps(performance or []),
            },
            {
                "document_key": "b" * 64,
                "selection_reasons_json": json.dumps(
                    ["one_first_post_event_prospectus_document"]
                ),
                "index_name_candidates_json": json.dumps(names or []),
                "index_code_candidates_json": "[]",
                "performance_benchmark_candidates_json": json.dumps(performance or []),
            },
        ]
    )


def test_cross_document_name_change_candidate_closes_without_promotion():
    result = validator.evaluate_event_candidate(
        _event(),
        _post(names=["中证A100指数"]),
        prior_name="中证100指数",
        prior_code="399903",
        prior_performance="中证100指数收益率",
        prior_scope="2013-03-05",
    )
    assert result["prior_state_continuity_status"] == "prior_state_fields_match"
    assert result["new_index_name_confirmation_count"] == 1
    assert result["event_chain_status"] == "candidate_event_closed_cross_document"
    assert not result["historical_backtest_allowed"]


def test_prior_state_conflict_blocks_chain_update():
    result = validator.evaluate_event_candidate(
        _event(),
        _post(names=["中证A100指数"]),
        prior_name="沪深300指数",
        prior_code="000300",
        prior_performance="沪深300指数收益率",
        prior_scope="2010-01-01",
    )
    assert result["event_chain_status"] == "blocked_prior_state_conflict"


def test_missing_post_event_legal_document_blocks_candidate():
    empty = _post().iloc[0:0]
    result = validator.evaluate_event_candidate(
        _event(),
        empty,
        prior_name="中证100指数",
        prior_code="399903",
        prior_performance="",
        prior_scope="2013-03-05",
    )
    assert result["event_chain_status"] == "blocked_post_event_legal_document_missing"


def test_candidate_scope_after_event_is_not_backdated():
    result = validator.evaluate_event_candidate(
        _event(),
        _post(names=["中证A100指数"]),
        prior_name="中证100指数",
        prior_code="399903",
        prior_performance="",
        prior_scope="2025-01-01",
    )
    assert result["event_chain_status"] == "blocked_prior_candidate_not_point_in_time_available"


def test_performance_change_requires_post_legal_confirmation():
    event = _event(
        event_types_json=json.dumps(["performance_benchmark_change"]),
        old_index_name_candidate="",
        new_index_name_candidate="",
        old_performance_benchmark_candidate="中证港股通科技指数（人民币）收益率",
        new_performance_benchmark_candidate="中证港股通科技指数收益率（使用估值汇率折算）",
    )
    result = validator.evaluate_event_candidate(
        event,
        _post(performance=["中证港股通科技指数收益率（使用估值汇率折算）"]),
        prior_name="中证港股通科技指数",
        prior_code="931573",
        prior_performance="中证港股通科技指数（人民币）收益率",
        prior_scope="2021-07-02",
    )
    assert result["new_performance_benchmark_confirmation_count"] == 1
    assert result["event_chain_status"] == "candidate_event_closed_cross_document"


def test_transformation_can_only_infer_new_state_from_two_agreeing_legal_documents():
    event = _event(
        event_types_json=json.dumps(["fund_transformation"]),
        old_index_name_candidate="",
        new_index_name_candidate="",
    )
    result = validator.evaluate_event_candidate(
        event,
        _post_pair(
            names=["深证100指数"],
            performance=["标的指数收益率，即深证100指数收益率"],
        ),
        prior_name="深证创新100指数",
        prior_code="399088",
        prior_performance="标的指数收益率，即深证创新100指数收益率",
        prior_scope="2021-05-27",
    )
    assert result["event_chain_status"] == (
        "candidate_event_closed_post_legal_value_inference_review_required"
    )
    assert result["post_event_inferred_new_index_name_candidate"] == "深证100指数"
    assert not result["historical_backtest_allowed"]


def test_transformation_inference_stays_blocked_without_both_legal_document_types():
    event = _event(
        event_types_json=json.dumps(["fund_transformation"]),
        old_index_name_candidate="",
        new_index_name_candidate="",
    )
    result = validator.evaluate_event_candidate(
        event,
        _post(
            names=["深证100指数"],
            performance=["标的指数收益率，即深证100指数收益率"],
        ),
        prior_name="深证创新100指数",
        prior_code="399088",
        prior_performance="标的指数收益率，即深证创新100指数收益率",
        prior_scope="2021-05-27",
    )
    assert result["event_chain_status"] == (
        "blocked_new_value_not_confirmed_by_post_event_legal_document"
    )


def test_missing_effective_date_orders_by_available_date_instead_of_last():
    events = pd.DataFrame(
        [
            {
                "event_effective_date_candidate": "",
                "available_date": "2021-03-13",
            },
            {
                "event_effective_date_candidate": "2023-06-12",
                "available_date": "2023-06-13",
            },
        ]
    )
    ordered = validator._candidate_event_order_dates(events)
    assert ordered.dt.strftime("%Y-%m-%d").tolist() == ["2021-03-13", "2023-06-12"]


def test_history_initial_state_uses_earliest_pit_legal_document():
    initial = SimpleNamespace(
        list_date="2006-09-05",
        canonical_index_name_candidate="中小企业100指数",
        canonical_index_code_candidate="",
        canonical_performance_benchmark_candidate="",
        tradable_scope_from_candidate="2024-01-26",
    )
    rows = pd.DataFrame(
        [
            {
                "available_date": "2006-05-20",
                "document_key": "a" * 64,
                "selection_reasons_json": json.dumps(
                    ["one_initial_prospectus_supplement_document"]
                ),
                "parse_status": "tracked_index_candidate_review_required",
                "index_name_candidates_json": json.dumps(["中小企业板价格指数"]),
                "index_code_candidates_json": json.dumps(["399329"]),
                "performance_benchmark_candidates_json": "[]",
            },
            {
                "available_date": "2024-01-26",
                "document_key": "b" * 64,
                "selection_reasons_json": json.dumps(
                    ["one_preferred_initial_legal_document"]
                ),
                "parse_status": "tracked_index_candidate_review_required",
                "index_name_candidates_json": json.dumps(["中小企业100指数"]),
                "index_code_candidates_json": json.dumps(["399005"]),
                "performance_benchmark_candidates_json": "[]",
            },
        ]
    )
    state = validator._initial_history_state(initial, rows)
    assert state["index_name"] == "中小企业板价格指数"
    assert state["index_code"] == "399329"
    assert state["index_name_scope_from"] == "2006-09-05"
    assert state["source"] == "earliest_pit_initial_documents"


def test_generic_tracked_performance_matches_only_with_same_index_context():
    event = _event(
        event_types_json=json.dumps(["index_name_change", "performance_benchmark_change"]),
        old_performance_benchmark_candidate="中证100指数收益率",
        new_performance_benchmark_candidate="中证A100指数收益率",
    )
    result = validator.evaluate_event_candidate(
        event,
        _post(names=["中证A100指数"], performance=["同期标的指数收益率"]),
        prior_name="中证100指数",
        prior_code="000903",
        prior_performance="同期标的指数收益率",
        prior_scope="2019-07-01",
    )
    assert result["event_chain_status"] == "candidate_event_closed_cross_document"

    mismatch = validator.evaluate_event_candidate(
        event,
        _post(names=["沪深300指数"], performance=["同期标的指数收益率"]),
        prior_name="中证100指数",
        prior_code="000903",
        prior_performance="同期标的指数收益率",
        prior_scope="2019-07-01",
    )
    assert mismatch["event_chain_status"] == (
        "blocked_new_value_not_confirmed_by_post_event_legal_document"
    )


def test_post_event_fallback_uses_full_legal_title_and_excludes_notice():
    event = SimpleNamespace(available_date="2021-03-13", document_key="e" * 64)
    rows = pd.DataFrame(
        [
            {
                "available_date": "2021-03-14",
                "announcement_date": "2021-03-13",
                "announcement_title": "基金合同生效公告",
                "document_key": "n" * 64,
                "selection_reasons_json": json.dumps(["all_contract_amendments"]),
                "parse_status": "ambiguous_candidate_review_required",
            },
            {
                "available_date": "2021-04-03",
                "announcement_date": "2021-04-02",
                "announcement_title": "基金合同（修订后）",
                "document_key": "c" * 64,
                "selection_reasons_json": json.dumps(["all_contract_amendments"]),
                "parse_status": "tracked_index_candidate_review_required",
            },
        ]
    )
    selected = validator._post_event_documents(rows, event)
    assert selected["document_key"].tolist() == ["c" * 64]
