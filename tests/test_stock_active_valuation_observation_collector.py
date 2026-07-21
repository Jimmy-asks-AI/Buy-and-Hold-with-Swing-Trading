from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy_lab.long_hold_v4.stock_active_valuation_observation_collector import (
    collect_observation,
    normalise_observation,
    select_target_assets,
    validate_source_frame,
)


def source_rows(asset: str = "600000") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "数据日期": "2025-01-02",
                "当日收盘价": 10.0,
                "当日涨跌幅": 1.0,
                "总市值": 1000.0,
                "流通市值": 800.0,
                "总股本": 100.0,
                "流通股本": 80.0,
                "PE(TTM)": 8.0,
                "PE(静)": 9.0,
                "市净率": 0.8,
                "PEG值": 1.0,
                "市现率": 5.0,
                "市销率": 2.0,
                "asset": asset,
            },
            {
                "数据日期": "2025-01-03",
                "当日收盘价": 10.2,
                "当日涨跌幅": 2.0,
                "总市值": 1020.0,
                "流通市值": 816.0,
                "总股本": 100.0,
                "流通股本": 80.0,
                "PE(TTM)": 8.2,
                "PE(静)": 9.2,
                "市净率": 0.82,
                "PEG值": 1.1,
                "市现率": 5.1,
                "市销率": 2.1,
                "asset": asset,
            },
        ]
    )


def write_master(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "asset": "600000",
                "asset_name": "浦发银行",
                "list_date": "1999-11-10",
                "delist_date": "",
                "list_status": "listed",
                "event_type": "listing",
                "exchange": "SSE",
            },
            {
                "asset": "000001",
                "asset_name": "平安银行",
                "list_date": "1991-04-03",
                "delist_date": "",
                "list_status": "listed",
                "event_type": "listing",
                "exchange": "SZSE",
            },
            {
                "asset": "000004",
                "asset_name": "退市样本",
                "list_date": "1991-01-14",
                "delist_date": "",
                "list_status": "listed",
                "event_type": "listing",
                "exchange": "SZSE",
            },
            {
                "asset": "000004",
                "asset_name": "退市样本",
                "list_date": "1991-01-14",
                "delist_date": "2025-01-03",
                "list_status": "delisted",
                "event_type": "delisting",
                "exchange": "SZSE",
            },
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")


class StockActiveValuationObservationTests(unittest.TestCase):
    def test_source_validation_rejects_future_rows(self):
        raw = source_rows()
        with self.assertRaisesRegex(ValueError, "future dates"):
            validate_source_frame(raw, "600000", "2025-01-02")

    def test_normalised_rows_use_actual_observation_date_and_are_never_pit(self):
        lifecycle = pd.Series(
            {
                "asset": "600000",
                "asset_name": "浦发银行",
                "exchange": "SSE",
                "list_date": pd.Timestamp("1999-11-10"),
                "delist_date": pd.NaT,
            }
        )
        result = normalise_observation(
            source_rows(), lifecycle, "2025-01-03", "2026-07-19T10:00:00+08:00"
        )
        self.assertEqual(result["available_date"].unique().tolist(), ["2026-07-19"])
        self.assertFalse(result["pit_actionable"].any())
        self.assertFalse(result["historical_backtest_allowed"].any())
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-01-02", "2025-01-03"])

    def test_candidate_selection_excludes_delisted_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "master.csv"
            candidates = root / "candidates.csv"
            write_master(master)
            pd.DataFrame(
                [
                    {"asset": "600000", "sector": "bank"},
                    {"asset": "000004", "sector": "bank"},
                ]
            ).to_csv(candidates, index=False, encoding="utf-8-sig")
            selected = select_target_assets(
                "2025-01-04", "candidates", master_path=master, candidate_path=candidates
            )
            self.assertEqual(selected["asset"].tolist(), ["600000"])

    def test_collection_is_resumable_and_writes_non_promotable_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "master.csv"
            candidates = root / "candidates.csv"
            raw_cache = root / "raw"
            legacy_cache = root / "legacy"
            observation = root / "observation.csv.gz"
            status = root / "status.csv"
            manifest = root / "manifest.json"
            archive = root / "archive"
            write_master(master)
            pd.DataFrame(
                [
                    {"asset": "600000", "sector": "bank"},
                    {"asset": "000001", "sector": "bank"},
                ]
            ).to_csv(candidates, index=False, encoding="utf-8-sig")

            calls: list[str] = []

            def fetcher(asset: str) -> pd.DataFrame:
                calls.append(asset)
                return source_rows(asset)

            first = collect_observation(
                "2025-01-03",
                max_fetch=1,
                sleep_seconds=0,
                retry_attempts=1,
                fetcher=fetcher,
                master_path=master,
                candidate_path=candidates,
                raw_cache_dir=raw_cache,
                legacy_cache_dir=legacy_cache,
                observation_path=observation,
                status_path=status,
                manifest_path=manifest,
                archive_dir=archive,
            )
            self.assertEqual(first["completed_assets"], 1)
            self.assertEqual(first["deferred_assets"], 1)
            self.assertFalse(first["historical_backtest_allowed"])
            self.assertEqual(first["schema_version"], "v2")
            self.assertTrue(first["target_universe_is_current_watchlist"])
            self.assertFalse(first["target_universe_is_current_snapshot"])
            second = collect_observation(
                "2025-01-03",
                max_fetch=1,
                sleep_seconds=0,
                retry_attempts=1,
                fetcher=fetcher,
                master_path=master,
                candidate_path=candidates,
                raw_cache_dir=raw_cache,
                legacy_cache_dir=legacy_cache,
                observation_path=observation,
                status_path=status,
                manifest_path=manifest,
                archive_dir=archive,
            )
            self.assertEqual(second["completed_assets"], 2)
            self.assertEqual(calls, ["000001", "600000"])
            rows = pd.read_csv(observation, compression="gzip", dtype={"asset": str})
            self.assertEqual(rows["asset"].nunique(), 2)
            self.assertFalse(rows["historical_backtest_allowed"].any())


if __name__ == "__main__":
    unittest.main()
