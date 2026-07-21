from __future__ import annotations

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_terminal_event_universe_collector as universe


def test_settlement_keeps_pure_discovery_separate_from_formal_events():
    cutoff = pd.Timestamp("2020-12-31")
    targets = pd.DataFrame(
        [
            {
                "asset": "510001",
                "asset_name": "现金清算样本",
                "exchange": "SSE",
                "list_date": "2015-01-01",
                "delist_date": "2020-06-30",
            },
            {
                "asset": "159001",
                "asset_name": "证据不足样本",
                "exchange": "SZSE",
                "list_date": "2016-01-01",
                "delist_date": "2020-09-30",
            },
        ]
    )
    announcements = pd.DataFrame(
        [
            {
                "asset": "510001",
                "announcement_date": "2020-06-01",
                "candidate_event_types_json": '["cash_liquidation_or_extinguishment"]',
            },
            {
                "asset": "159001",
                "announcement_date": "2020-09-01",
                "candidate_event_types_json": '["exchange_delisting"]',
            },
        ]
    )
    query_complete_assets = {"510001", "159001"}

    discovery = universe.build_coverage_registry(
        targets,
        announcements,
        query_complete_assets,
        pd.DataFrame(),
        cutoff=cutoff,
    )
    assert len(discovery) == 2
    assert discovery["final_evidence_state"].eq("evidence_insufficient").all()
    assert int(discovery["formal_event_count"].sum()) == 0
    assert not discovery["terminal_event_historical_backtest_allowed"].any()

    formal_events = pd.DataFrame(
        [
            {
                "asset": "510001",
                "event_type": "cash_liquidation",
                "pay_date": "2020-07-15",
                "available_date": "2020-07-10",
                "cash_per_share": 1.02,
                "extinguishes_position": True,
                "source_pdf_sha256_set": "synthetic-document-hash",
                "historical_backtest_allowed": True,
            }
        ]
    )
    settled = universe.build_coverage_registry(
        targets,
        announcements,
        query_complete_assets,
        formal_events,
        cutoff=cutoff,
    )

    identified = settled.loc[settled["asset"].eq("510001")].iloc[0]
    unresolved = settled.loc[settled["asset"].eq("159001")].iloc[0]
    assert identified["final_evidence_state"] == "terminal_event_identified"
    assert identified["formal_event_count"] == 1
    assert bool(identified["formal_event_chain_complete"])
    assert bool(identified["terminal_event_historical_backtest_allowed"])
    assert unresolved["final_evidence_state"] == "evidence_insufficient"
    assert unresolved["formal_event_count"] == 0
    assert not bool(unresolved["terminal_event_historical_backtest_allowed"])
    assert not settled["universe_terminal_coverage_complete"].any()
    assert not settled["model_promotion_allowed"].any()
