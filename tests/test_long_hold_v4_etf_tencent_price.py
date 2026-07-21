from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from strategy_lab.long_hold_v4.pit_etf_tencent_price_collector import (
    fetch_tencent_history,
    normalise_history,
    parse_tencent_page,
    select_assets,
    tencent_symbol,
)


def _payload(symbol: str, rows: list[list[str]]) -> bytes:
    return json.dumps({"code": 0, "data": {symbol: {"day": rows}}}, separators=(",", ":")).encode()


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.params: list[dict[str, str]] = []

    def get(self, _url: str, *, params: dict[str, str], **_kwargs) -> FakeResponse:
        self.params.append(params)
        return FakeResponse(self.responses.pop(0))


def test_parse_tencent_page_preserves_raw_ohlcv_semantics() -> None:
    symbol = "sh510050"
    frame = parse_tencent_page(
        json.loads(
            _payload(
                symbol,
                [["2020-01-02", "2.100", "2.120", "2.130", "2.090", "1234.000"]],
            )
        ),
        symbol,
    )
    assert frame.to_dict("records") == [
        {
            "date": pd.Timestamp("2020-01-02"),
            "open": 2.1,
            "close": 2.12,
            "high": 2.13,
            "low": 2.09,
            "volume_lots": 1234.0,
            "ohlc_relationship_valid": True,
            "ohlc_rounding_gap": 0.0,
        }
    ]


def test_parse_tencent_page_flags_bounded_high_low_rounding_without_repair() -> None:
    symbol = "sh510050"
    frame = parse_tencent_page(
        json.loads(
            _payload(
                symbol,
                [["2010-02-03", "2.227", "2.282", "2.280", "2.190", "100"]],
            )
        ),
        symbol,
    )
    assert frame.loc[0, "high"] == 2.28
    assert frame.loc[0, "close"] == 2.282
    assert not bool(frame.loc[0, "ohlc_relationship_valid"])
    assert frame.loc[0, "ohlc_rounding_gap"] == pytest.approx(0.002)


@pytest.mark.parametrize(
    "rows",
    [
        [["2020-01-02", "2.1"]],
        [["2020-01-02", "2.10", "2.12", "2.11", "2.09", "100"]],
        [["2020-01-02", "0", "2.12", "2.13", "2.09", "100"]],
    ],
)
def test_parse_tencent_page_rejects_malformed_market_rows(rows: list[list[str]]) -> None:
    symbol = "sh510050"
    with pytest.raises(ValueError):
        parse_tencent_page(json.loads(_payload(symbol, rows)), symbol)


def test_fetch_tencent_history_paginates_backwards_without_overlap() -> None:
    symbol = "sh510050"
    session = FakeSession(
        [
            _payload(
                symbol,
                [
                    ["2020-01-03", "1.0", "1.1", "1.2", "0.9", "10"],
                    ["2020-01-04", "1.1", "1.2", "1.3", "1.0", "11"],
                ],
            ),
            _payload(
                symbol,
                [
                    ["2020-01-01", "0.9", "1.0", "1.1", "0.8", "8"],
                    ["2020-01-02", "1.0", "1.0", "1.1", "0.9", "9"],
                ],
            ),
        ]
    )
    frame, pages = fetch_tencent_history(
        session,
        symbol=symbol,
        list_date="2020-01-01",
        query_end="2020-01-04",
        page_size=2,
        max_pages=3,
    )
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    ]
    assert [page.rows for page in pages] == [2, 2]
    assert session.params[0]["param"].endswith(",2020-01-04,2,")
    assert session.params[1]["param"].endswith(",2020-01-02,2,")


def test_normalise_history_uses_collection_date_not_market_date_as_availability() -> None:
    lifecycle = pd.Series(
        {
            "asset": "159917",
            "asset_name": "中小成长",
            "exchange": "SZSE",
            "list_date": pd.Timestamp("2012-04-06"),
            "delist_date": pd.Timestamp("2015-08-26"),
        }
    )
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2012-04-05", "2012-04-06", "2015-08-26", "2015-08-27"]),
            "open": [1.0, 1.0, 1.7, 1.8],
            "close": [1.0, 1.01, 1.77, 1.8],
            "high": [1.0, 1.02, 1.78, 1.8],
            "low": [1.0, 1.0, 1.69, 1.8],
            "volume_lots": [1.0, 10.0, 20.0, 1.0],
        }
    )
    result = normalise_history(
        raw,
        lifecycle,
        as_of="2026-07-17",
        observed_at="2026-07-19T20:00:00+08:00",
        source_vintage="fixture",
    )
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2012-04-06", "2015-08-26"]
    assert result["volume_shares"].tolist() == [1000.0, 2000.0]
    assert result["available_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-07-19"]
    assert not result["pit_actionable"].any()
    assert not result["historical_backtest_allowed"].any()


def test_selection_prioritises_delisted_assets_and_validates_exchange() -> None:
    lifecycles = pd.DataFrame(
        {
            "asset": ["510050", "159917", "510070"],
            "asset_name": ["50ETF", "中小成长", "民企ETF"],
            "exchange": ["SSE", "SZSE", "SSE"],
            "list_date": pd.to_datetime(["2005-02-23", "2012-04-06", "2010-10-29"]),
            "delist_date": pd.to_datetime([None, "2015-08-26", "2020-08-12"]),
        }
    )
    assert select_assets(lifecycles, selection="all")["asset"].tolist() == ["510070", "159917", "510050"]
    assert tencent_symbol("510050", "SSE") == "sh510050"
    assert tencent_symbol("159917", "SZSE") == "sz159917"
    with pytest.raises(ValueError):
        tencent_symbol("123456", "BSE")
