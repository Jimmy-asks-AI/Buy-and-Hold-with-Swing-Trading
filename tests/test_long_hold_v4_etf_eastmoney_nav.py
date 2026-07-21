from __future__ import annotations

import json

import pandas as pd
import pytest

from strategy_lab.long_hold_v4.pit_etf_eastmoney_nav_collector import (
    EastmoneySourceParseError,
    fetch_eastmoney_nav,
    normalise_nav,
    parse_f10_page,
    parse_pingzhongdata,
)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, params: dict[str, str] | None = None, **_kwargs) -> FakeResponse:
        self.requests.append((url, params))
        return FakeResponse(self.responses.pop(0))


def _js(net_rows: list[dict[str, object]], cumulative_rows: list[list[object]]) -> bytes:
    return (
        "var Data_netWorthTrend = "
        + json.dumps(net_rows, separators=(",", ":"))
        + ";\nvar Data_ACWorthTrend = "
        + json.dumps(cumulative_rows, separators=(",", ":"))
        + ";"
    ).encode()


def _f10(rows: list[dict[str, str]], *, total: int, page_size: int = 20) -> bytes:
    return json.dumps(
        {"ErrCode": 0, "Data": {"LSJZList": rows}, "TotalCount": total, "PageSize": page_size},
        separators=(",", ":"),
    ).encode()


def test_parse_pingzhongdata_preserves_nav_semantics() -> None:
    frame = parse_pingzhongdata(
        _js(
            [
                {"x": 1577923200000, "y": 1.01, "equityReturn": 1.0, "unitMoney": ""},
                {"x": 1578009600000, "y": 1.02, "equityReturn": 0.99, "unitMoney": "每份派现金0.01元"},
            ],
            [[1577923200000, 1.01], [1578009600000, 1.03]],
        )
    )
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03"]
    assert frame["unit_nav"].tolist() == [1.01, 1.02]
    assert frame["cumulative_nav"].tolist() == [1.01, 1.03]
    assert frame.loc[1, "unit_money"] == "每份派现金0.01元"


def test_parse_pingzhongdata_allows_missing_optional_fields() -> None:
    frame = parse_pingzhongdata(
        _js([{"x": 1577923200000, "y": 1.0}], [[1577923200000, 1.0]])
    )
    assert pd.isna(frame.loc[0, "daily_growth_pct"])
    assert frame.loc[0, "unit_money"] == ""


def test_parse_pingzhongdata_discloses_missing_cumulative_nav_without_filling() -> None:
    frame = parse_pingzhongdata(
        _js([{"x": 1577923200000, "y": 1.0}], [[1577923200000, None]])
    )
    assert frame.loc[0, "unit_nav"] == 1.0
    assert pd.isna(frame.loc[0, "cumulative_nav"])


def test_parse_and_normalise_drop_missing_unit_nav_without_filling() -> None:
    raw = parse_pingzhongdata(
        _js(
            [
                {"x": 1577923200000, "y": 1.0},
                {"x": 1578009600000, "y": None},
            ],
            [[1577923200000, 1.0], [1578009600000, None]],
        )
    )
    assert raw["unit_nav"].isna().sum() == 1
    lifecycle = pd.Series(
        {
            "asset": "511120",
            "asset_name": "fixture",
            "exchange": "SSE",
            "list_date": pd.Timestamp("2020-01-01"),
            "delist_date": pd.NaT,
        }
    )
    result = normalise_nav(
        raw,
        lifecycle,
        as_of="2020-01-31",
        observed_at="2026-07-19T20:00:00+08:00",
        source_method="fixture",
        source_vintage="fixture",
    )
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02"]
    assert result["unit_nav"].tolist() == [1.0]


def test_parse_pingzhongdata_returns_empty_when_arrays_are_absent() -> None:
    assert parse_pingzhongdata(b"var unrelated = [];").empty


def test_parse_f10_page_validates_required_fields() -> None:
    frame, total, page_size = parse_f10_page(
        json.loads(
            _f10(
                [{"FSRQ": "2020-01-02", "DWJZ": "1.0100", "LJJZ": "1.0200", "JZZZL": "1.00"}],
                total=1,
            )
        )
    )
    assert total == 1
    assert page_size == 20
    assert frame.loc[0, "unit_nav"] == pytest.approx(1.01)
    with pytest.raises(ValueError, match="incomplete"):
        parse_f10_page({"ErrCode": 0, "Data": {"LSJZList": [{"FSRQ": "2020-01-02"}]}})


def test_fetch_prefers_compact_javascript_source() -> None:
    session = FakeSession(
        [_js([{"x": 1577923200000, "y": 1.0}], [[1577923200000, 1.0]])]
    )
    frame, method, responses = fetch_eastmoney_nav(
        session,
        asset="510050",
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    assert len(frame) == 1
    assert method == "pingzhongdata"
    assert [item.source_kind for item in responses] == ["pingzhongdata"]
    assert len(session.requests) == 1


def test_fetch_falls_back_to_paginated_f10_source() -> None:
    session = FakeSession(
        [
            b"var unrelated = [];",
            _f10(
                [
                    {"FSRQ": "2020-01-03", "DWJZ": "1.0200", "LJJZ": "1.0200"},
                    {"FSRQ": "2020-01-02", "DWJZ": "1.0100", "LJJZ": "1.0100"},
                ],
                total=3,
                page_size=2,
            ),
            _f10(
                [{"FSRQ": "2020-01-01", "DWJZ": "1.0000", "LJJZ": "1.0000"}],
                total=3,
                page_size=2,
            ),
        ]
    )
    frame, method, responses = fetch_eastmoney_nav(
        session,
        asset="159917",
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    assert method == "f10_lsjz"
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert [item.source_kind for item in responses] == ["pingzhongdata", "f10_lsjz", "f10_lsjz"]
    assert session.requests[1][1]["pageIndex"] == "1"
    assert session.requests[2][1]["pageIndex"] == "2"


def test_fetch_preserves_raw_response_when_source_parsing_fails() -> None:
    content = _js([{"x": 1577923200000, "y": 0.0}], [[1577923200000, 1.0]])
    session = FakeSession([content])
    with pytest.raises(EastmoneySourceParseError) as caught:
        fetch_eastmoney_nav(
            session,
            asset="510050",
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
    assert len(caught.value.responses) == 1
    assert caught.value.responses[0].content == content
    assert caught.value.responses[0].rows == 0


def test_normalise_nav_clips_lifecycle_and_keeps_current_final_availability() -> None:
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
            "unit_nav": [1.0, 1.01, 1.7, 1.8],
            "cumulative_nav": [1.0, 1.01, 1.8, 1.9],
            "daily_growth_pct": [0.0, 1.0, 2.0, 3.0],
            "unit_money": ["", "", "", ""],
        }
    )
    result = normalise_nav(
        raw,
        lifecycle,
        as_of="2026-07-17",
        observed_at="2026-07-19T20:00:00+08:00",
        source_method="fixture",
        source_vintage="fixture",
    )
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2012-04-06", "2015-08-26"]
    assert result["available_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-07-19"]
    assert not result["pit_actionable"].any()
    assert not result["historical_backtest_allowed"].any()
    assert not result["model_promotion_allowed"].any()
