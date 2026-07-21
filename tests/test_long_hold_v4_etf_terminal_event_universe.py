import hashlib
import json

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_terminal_event_universe_collector as collector
from strategy_lab.long_hold_v4.pit_etf_cninfo_share_action_announcement_collector import (
    parse_announcement_time,
)


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "159917",
                "asset_name": "Converted ETF",
                "exchange": "SZSE",
                "list_date": pd.Timestamp("2012-04-06"),
                "delist_date": pd.Timestamp("2015-08-26"),
            },
            {
                "asset": "510700",
                "asset_name": "Unknown ETF",
                "exchange": "SSE",
                "list_date": pd.Timestamp("2013-05-31"),
                "delist_date": pd.Timestamp("2016-05-12"),
            },
            {
                "asset": "511210",
                "asset_name": "Liquidated ETF",
                "exchange": "SSE",
                "list_date": pd.Timestamp("2013-08-16"),
                "delist_date": pd.Timestamp("2018-01-25"),
            },
        ]
    )


def test_title_classifier_routes_without_claiming_final_economics():
    tags, candidates = collector.classify_title("关于实施转型、份额转换并终止上市的公告")
    assert tags == ["delisting", "successor_share_conversion", "transformation"]
    assert candidates == ["conversion_to_successor", "exchange_delisting"]

    tags, candidates = collector.classify_title("基金财产清算及剩余财产分配公告")
    assert tags == ["liquidation", "terminal_distribution"]
    assert candidates == ["cash_liquidation_or_extinguishment"]

    tags, candidates = collector.classify_title(
        "关于《某交易型开放式指数证券投资基金基金合同》终止的公告"
    )
    assert "fund_contract_termination" in tags
    assert candidates == ["cash_liquidation_or_extinguishment"]


def test_cninfo_epoch_is_converted_to_china_local_date():
    parsed = parse_announcement_time(1545580800000)
    assert parsed == pd.Timestamp("2018-12-24 00:00:00")
    assert parsed.tzinfo is None


def test_zero_query_matches_never_become_a_no_event_conclusion():
    registry = collector.build_coverage_registry(
        _targets().iloc[[1]],
        pd.DataFrame(columns=collector.ANNOUNCEMENT_COLUMNS),
        {"510700"},
        pd.DataFrame(),
        cutoff=pd.Timestamp("2026-07-17"),
    )
    row = registry.iloc[0]
    assert row["discovery_state"] == "official_query_no_match_requires_no_event_proof"
    assert row["final_evidence_state"] == "evidence_insufficient"
    assert bool(row["document_validation_required"])
    assert not bool(row["terminal_event_historical_backtest_allowed"])


def test_only_independently_promoted_event_reaches_identified_state():
    announcements = pd.DataFrame(
        [
            {
                "asset": "159917",
                "candidate_event_types_json": json.dumps(
                    ["conversion_to_successor", "exchange_delisting"]
                ),
                "announcement_date": pd.Timestamp("2015-08-25"),
            }
        ]
    )
    formal = pd.DataFrame(
        [
            {
                "asset": "511210",
                "event_type": "liquidation_distribution",
                "cash_per_share": 112.79,
                "extinguishes_position": True,
                "pay_date": "2018-01-23",
                "available_date": "2018-01-09",
                "source_pdf_sha256_set": "a" * 64,
                "historical_backtest_allowed": True,
            }
        ]
    )
    registry = collector.build_coverage_registry(
        _targets(),
        announcements,
        {"159917", "510700", "511210"},
        formal,
        cutoff=pd.Timestamp("2026-07-17"),
    ).set_index("asset")
    assert registry.loc["159917", "primary_candidate_class"] == "successor_share_candidate"
    assert registry.loc["159917", "final_evidence_state"] == "evidence_insufficient"
    assert registry.loc["511210", "final_evidence_state"] == "terminal_event_identified"
    assert bool(registry.loc["511210", "terminal_value_amount_known"])
    assert bool(registry.loc["511210", "position_extinguishment_known"])
    assert bool(registry.loc["511210", "terminal_event_historical_backtest_allowed"])
    assert not registry["universe_terminal_coverage_complete"].any()


def test_latest_terminal_mechanism_beats_an_old_listing_conversion():
    announcements = pd.DataFrame(
        [
            {
                "asset": "510700",
                "candidate_event_types_json": json.dumps(["conversion_to_successor"]),
                "announcement_date": pd.Timestamp("2013-05-17"),
            },
            {
                "asset": "510700",
                "candidate_event_types_json": json.dumps(["cash_liquidation_or_extinguishment"]),
                "announcement_date": pd.Timestamp("2016-04-26"),
            },
            {
                "asset": "510700",
                "candidate_event_types_json": json.dumps(["exchange_delisting"]),
                "announcement_date": pd.Timestamp("2016-05-12"),
            },
        ]
    )
    registry = collector.build_coverage_registry(
        _targets().iloc[[1]],
        announcements,
        {"510700"},
        pd.DataFrame(),
        cutoff=pd.Timestamp("2026-07-17"),
    )
    assert registry.iloc[0]["primary_candidate_class"] == "cash_or_extinguishment_candidate"


def test_query_cache_is_hash_authenticated(monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "QUERY_DIR", tmp_path)
    cutoff = pd.Timestamp("2026-07-17")
    data_path, meta_path = collector._query_paths("SSE", "510700")
    data_path.parent.mkdir(parents=True)
    artifact = {
        "schema_version": collector.SCHEMA_VERSION,
        "asset": "510700",
        "exchange": "SSE",
        "as_of_date": "2026-07-17",
        "query_keywords": list(collector.KEYWORDS),
        "query_contract_version": collector.QUERY_CONTRACT_VERSION,
        "request_contract_fingerprint": collector._query_contract_fingerprint("SSE"),
        "responses": [
            {
                "keyword": keyword,
                "page_count": 1,
                "requests": [{"page_number": 1}],
                "rows": [],
            }
            for keyword in collector.KEYWORDS
        ],
    }
    data_path.write_text(json.dumps(artifact), encoding="utf-8")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    meta_path.write_text(
        json.dumps(
            {
                "status": "success",
                "sha256": digest,
                "producer_code_sha256": collector._sha256(collector.Path(collector.__file__).resolve()),
                "dependency_hashes": collector._query_dependency_hashes(),
                "request_contract_fingerprint": collector._query_contract_fingerprint("SSE"),
            }
        ),
        encoding="utf-8",
    )
    assert collector._valid_query_cache("SSE", "510700", cutoff)

    data_path.write_text(json.dumps({**artifact, "responses": [{"rows": []}]}), encoding="utf-8")
    assert not collector._valid_query_cache("SSE", "510700", cutoff)


def test_future_formal_event_is_not_visible_and_multiple_events_are_order_independent():
    target = _targets().iloc[[2]]
    rows = [
        {
            "event_id": "interim",
            "asset": "511210",
            "event_type": "liquidation_distribution",
            "pay_date": "2018-01-20",
            "available_date": "2018-01-10",
            "cash_per_share": 100.0,
            "extinguishes_position": False,
            "source_pdf_sha256_set": "a" * 64,
            "historical_backtest_allowed": True,
            "distribution_sequence": 1,
        },
        {
            "event_id": "final",
            "asset": "511210",
            "event_type": "liquidation_distribution",
            "pay_date": "2018-02-20",
            "available_date": "2099-01-01",
            "cash_per_share": 12.79,
            "extinguishes_position": True,
            "source_pdf_sha256_set": "b" * 64,
            "historical_backtest_allowed": True,
            "distribution_sequence": 2,
        },
    ]
    first = collector.build_coverage_registry(
        target,
        pd.DataFrame(columns=collector.ANNOUNCEMENT_COLUMNS),
        {"511210"},
        pd.DataFrame(rows),
        cutoff=pd.Timestamp("2018-12-31"),
    ).iloc[0]
    second = collector.build_coverage_registry(
        target,
        pd.DataFrame(columns=collector.ANNOUNCEMENT_COLUMNS),
        {"511210"},
        pd.DataFrame(list(reversed(rows))),
        cutoff=pd.Timestamp("2018-12-31"),
    ).iloc[0]
    assert first["formal_event_count"] == second["formal_event_count"] == 1
    assert not bool(first["formal_event_chain_complete"])
    assert not bool(second["position_extinguishment_known"])


def test_announcement_keeps_publish_time_and_is_only_trade_available_next_day(monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "QUERY_DIR", tmp_path)
    published = pd.Timestamp("2026-07-16 18:30:00", tz="Asia/Shanghai")
    artifact = {
        "asset": "159917",
        "asset_name": "ETF",
        "exchange": "SZSE",
        "as_of_date": "2026-07-17",
        "fetched_at": "2026-07-19T12:00:00+08:00",
        "responses": [
            {
                "keyword": "清算",
                "rows": [
                    {
                        "secCode": "159917",
                        "announcementTitle": "基金财产清算公告",
                        "adjunctUrl": "finalpage/example.PDF",
                        "announcementTime": int(published.tz_convert("UTC").timestamp() * 1000),
                    }
                ],
            }
        ],
    }
    query_path, _ = collector._query_paths("SZSE", "159917")
    query_path.parent.mkdir(parents=True)
    query_path.write_text(json.dumps(artifact), encoding="utf-8")
    frame = collector.parse_query_artifacts([artifact])
    assert len(frame) == 1
    assert frame.iloc[0]["published_at"] == published.isoformat()
    assert pd.Timestamp(frame.iloc[0]["available_trade_date"]) == pd.Timestamp("2026-07-17")
    assert frame.iloc[0]["available_at"].startswith("2026-07-17T00:00:00+08:00")

    same_day = {**artifact, "as_of_date": "2026-07-16"}
    query_path.write_text(json.dumps(same_day), encoding="utf-8")
    assert collector.parse_query_artifacts([same_day]).empty
