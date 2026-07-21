"""Accepted processed scope for daily-only raw Tushare data.

This adapter keeps the raw CSV files immutable, applies row-level quarantine
rules at read time, and preserves the V3.44 guard that blocks adjusted-return
and valuation misuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from raw_daily_guard import DataCapabilityError, RawDailyOnlyAdapter


IDENTIFIER_COLUMNS = {"ts_code", "trade_date"}


def normalize_trade_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


@dataclass(frozen=True)
class DailyOnlyQuarantine:
    """Row-level exclusions for the accepted processed daily-only scope."""

    rows: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path) -> "DailyOnlyQuarantine":
        if not path.exists():
            raise FileNotFoundError(path)
        rows = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if rows.empty:
            rows = pd.DataFrame(columns=["trade_date", "ts_code", "anomaly_type"])
        required = {"trade_date", "ts_code"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"quarantine file missing columns: {sorted(missing)}")
        cleaned = rows.copy()
        cleaned["trade_date"] = cleaned["trade_date"].map(normalize_trade_date)
        cleaned["ts_code"] = cleaned["ts_code"].astype(str)
        return cls(cleaned)

    def pairs_for_trade_date(self, trade_date: str) -> set[tuple[str, str]]:
        date_key = normalize_trade_date(trade_date)
        data = self.rows.loc[self.rows["trade_date"] == date_key]
        return set(zip(data["trade_date"], data["ts_code"]))

    def filter_trade_date(self, data: pd.DataFrame, trade_date: str) -> tuple[pd.DataFrame, int]:
        if data.empty:
            return data.copy(), 0
        missing = IDENTIFIER_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"cannot apply quarantine without columns: {sorted(missing)}")
        pairs = self.pairs_for_trade_date(trade_date)
        if not pairs:
            return data.copy(), 0
        data_dates = data["trade_date"].map(normalize_trade_date)
        data_codes = data["ts_code"].astype(str)
        mask = pd.Series(
            [(date_key, code) in pairs for date_key, code in zip(data_dates, data_codes)],
            index=data.index,
        )
        return data.loc[~mask].copy(), int(mask.sum())


@dataclass(frozen=True)
class AcceptedDailyOnlyAdapter:
    """Read-only accepted processed view over raw daily-only files."""

    data_root: Path
    quarantine_path: Path

    @property
    def raw_adapter(self) -> RawDailyOnlyAdapter:
        return RawDailyOnlyAdapter(self.data_root)

    @property
    def quarantine(self) -> DailyOnlyQuarantine:
        return DailyOnlyQuarantine.from_csv(self.quarantine_path)

    def trade_date_path(self, trade_date: str) -> Path:
        return self.raw_adapter.trade_date_path(trade_date)

    def require_capability(self, capability: str) -> None:
        self.raw_adapter.require_capability(capability)

    def load_trade_date(
        self,
        trade_date: str,
        columns: Iterable[str] | None = None,
        *,
        apply_quarantine: bool = True,
        add_scope_columns: bool = True,
    ) -> pd.DataFrame:
        requested = list(columns) if columns is not None else None
        load_columns: list[str] | None = None
        if requested is not None:
            load_columns = sorted(set(requested).union(IDENTIFIER_COLUMNS))

        data = self.raw_adapter.load_trade_date(trade_date, columns=load_columns)
        quarantined_rows = 0
        if apply_quarantine:
            data, quarantined_rows = self.quarantine.filter_trade_date(data, trade_date)

        if requested is not None:
            data = data[[col for col in requested if col in data.columns]].copy()

        if add_scope_columns:
            data["data_scope"] = "accepted_processed_daily_only"
            data["quarantine_applied"] = bool(apply_quarantine)
            data["quarantined_rows_for_trade_date"] = quarantined_rows
            data["price_adjustment"] = "none_raw"
            data["adjusted_return_allowed"] = False
        return data

    def get_close(self, trade_date: str, *, price_type: str = "raw_close") -> pd.DataFrame:
        if price_type != "raw_close":
            self.require_capability(price_type)
        data = self.load_trade_date(
            trade_date,
            columns=["ts_code", "trade_date", "close"],
            apply_quarantine=True,
            add_scope_columns=False,
        )
        out = data.rename(columns={"ts_code": "asset", "close": "raw_close"})
        out["data_scope"] = "accepted_processed_daily_only"
        out["quarantine_applied"] = True
        out["price_adjustment"] = "none_raw"
        out["adjusted_return_allowed"] = False
        return out

    def assert_can_run_portfolio_backtest(self, return_basis: str) -> None:
        self.raw_adapter.assert_can_run_portfolio_backtest(return_basis)

    def compute_raw_return_diagnostic(self, panel: pd.DataFrame) -> pd.DataFrame:
        return self.raw_adapter.compute_raw_return_diagnostic(panel)


def has_bad_ohlc_rows(data: pd.DataFrame) -> bool:
    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    bad = ((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).fillna(True)
    return bool(bad.any())


def assert_capability_blocked(adapter: AcceptedDailyOnlyAdapter, capability: str) -> str:
    try:
        adapter.require_capability(capability)
    except DataCapabilityError as exc:
        return str(exc)
    raise AssertionError(f"capability unexpectedly allowed: {capability}")
