from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy_lab.long_hold_v4.pit_history_gate import _sha256
from strategy_lab.long_hold_v4.pit_tushare_trade_state_builder import (
    OUTPUT_COLUMNS,
    _DeterministicGzipCsvWriter,
    _qualification_checks,
    derive_session_trade_state,
    normalise_tushare_session,
)


def active_row(
    asset: str = "600000",
    *,
    list_date: str = "2000-01-04",
    start_index: int = 0,
    is_ipo: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": asset,
                "list_date": pd.Timestamp(list_date),
                "start_index": start_index,
                "is_ipo": is_ipo,
            }
        ]
    )


def traded_row(asset: str = "600000", pre_close: float = 10.0, close: float = 10.5) -> pd.DataFrame:
    return pd.DataFrame([{"asset": asset, "pre_close": pre_close, "close": close}])


def derive(
    active: pd.DataFrame,
    traded: pd.DataFrame,
    *,
    date: str = "2020-01-20",
    session_index: int = 10,
    status: str = "normal",
    previous_paused: set[str] | None = None,
    last_close: dict[str, float] | None = None,
) -> pd.Series:
    asset = str(active.iloc[0]["asset"])
    result = derive_session_trade_state(
        active,
        traded,
        date=pd.Timestamp(date),
        session_index=session_index,
        current_status={asset: status},
        previous_paused=previous_paused or set(),
        last_close=last_close or {},
        source_vintage="test-vintage",
    )
    return result.iloc[0]


class TushareTradeStateTests(unittest.TestCase):
    def test_normalise_session_is_deterministic_and_excludes_non_sh_sz_rows(self):
        raw = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20200102", "close": 16.0, "pre_close": 15.5},
                {"ts_code": "600000.SH", "trade_date": "20200102", "close": 12.0, "pre_close": 11.8},
                {"ts_code": "920001.BJ", "trade_date": "20200102", "close": 8.0, "pre_close": 7.9},
            ]
        )
        result = normalise_tushare_session(raw, pd.Timestamp("2020-01-02"))
        self.assertEqual(result["asset"].tolist(), ["000001", "600000"])
        self.assertEqual(result.columns.tolist(), ["asset", "pre_close", "close"])

    def test_regular_main_board_uses_ten_percent_limits(self):
        row = derive(active_row(), traded_row())
        self.assertFalse(bool(row["is_paused"]))
        self.assertTrue(bool(row["has_price_limit"]))
        self.assertTrue(bool(row["execution_state_known"]))
        self.assertEqual(row["limit_rule"], "regular_main_or_pre_reform_growth_10")
        self.assertAlmostEqual(float(row["limit_up"]), 11.0)
        self.assertAlmostEqual(float(row["limit_down"]), 9.0)

    def test_main_board_st_uses_five_percent_limits(self):
        row = derive(active_row(), traded_row(), status="risk_warning")
        self.assertTrue(bool(row["is_st"]))
        self.assertEqual(row["limit_rule"], "regular_st_5")
        self.assertAlmostEqual(float(row["limit_up"]), 10.5)
        self.assertAlmostEqual(float(row["limit_down"]), 9.5)

    def test_growth_board_uses_twenty_percent_after_reform(self):
        row = derive(
            active_row("300001", list_date="2009-10-30"),
            traded_row("300001"),
            date="2020-08-24",
        )
        self.assertEqual(row["limit_rule"], "regular_growth_20")
        self.assertAlmostEqual(float(row["limit_up"]), 12.0)
        self.assertAlmostEqual(float(row["limit_down"]), 8.0)

    def test_registration_ipo_first_five_sessions_have_no_price_limit(self):
        row = derive(
            active_row("688001", list_date="2020-01-02", start_index=10),
            traded_row("688001"),
            date="2020-01-02",
            session_index=10,
        )
        self.assertEqual(row["limit_rule"], "no_price_limit_listing_window")
        self.assertFalse(bool(row["has_price_limit"]))
        self.assertTrue(bool(row["execution_state_known"]))
        self.assertTrue(pd.isna(row["limit_up"]))

    def test_missing_daily_row_is_a_known_suspension(self):
        row = derive(
            active_row(),
            pd.DataFrame(columns=["asset", "pre_close", "close"]),
            last_close={"600000": 9.8},
        )
        self.assertTrue(bool(row["is_paused"]))
        self.assertEqual(row["limit_rule"], "paused")
        self.assertTrue(bool(row["execution_state_known"]))
        self.assertAlmostEqual(float(row["pre_close"]), 9.8)

    def test_first_trade_after_pause_fails_closed(self):
        row = derive(active_row(), traded_row(), previous_paused={"600000"})
        self.assertEqual(row["limit_rule"], "resumption_limit_unknown")
        self.assertFalse(bool(row["execution_state_known"]))
        self.assertFalse(bool(row["has_price_limit"]))

    def test_legacy_special_transfer_fails_closed(self):
        row = derive(active_row(), traded_row(), status="special_transfer")
        self.assertTrue(bool(row["is_st"]))
        self.assertEqual(row["limit_rule"], "legacy_special_transfer_unknown")
        self.assertFalse(bool(row["execution_state_known"]))

    def test_qualification_fails_when_independent_provider_threshold_misses(self):
        cross = {
            "baostock": {"pause_checks": 800_000, "pause_match_ratio": 0.999},
            "joinquant": {
                "state_checks": 20_000,
                "paused_match_ratio": 0.999,
                "st_checks": 20_000,
                "st_match_ratio": 0.99,
                "limit_checks": 15_000,
                "limit_match_ratio": 0.999,
            },
            "source_errors": [],
        }
        checks = _qualification_checks(
            rows=100,
            expected_rows=100,
            assets=2,
            target_assets=2,
            unknown_ratio=0.01,
            paused_rows=1,
            st_rows=1,
            limited_rows=98,
            cross=cross,
        )
        failed = checks.loc[~checks["passed"], "check_id"].tolist()
        self.assertEqual(failed, ["joinquant_st_match"])

    def test_gzip_writer_is_byte_deterministic_and_abort_after_close_is_safe(self):
        frame = pd.DataFrame(
            [
                {
                    "date": "2020-01-02",
                    "asset": "600000",
                    "is_paused": False,
                    "is_st": False,
                    "pre_close": 10.0,
                    "has_price_limit": True,
                    "limit_up": 11.0,
                    "limit_down": 9.0,
                    "price_limit_rate": 0.1,
                    "limit_rule": "regular_main_or_pre_reform_growth_10",
                    "execution_state_known": True,
                    "available_date": "2020-01-02",
                    "data_source": "test",
                    "source_vintage": "test-vintage",
                }
            ],
            columns=OUTPUT_COLUMNS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "one.csv.gz", Path(tmp) / "two.csv.gz"]
            for path in paths:
                writer = _DeterministicGzipCsvWriter(path, OUTPUT_COLUMNS)
                writer.append(frame)
                writer.close()
                writer.abort()
            self.assertEqual(_sha256(paths[0]), _sha256(paths[1]))


if __name__ == "__main__":
    unittest.main()
