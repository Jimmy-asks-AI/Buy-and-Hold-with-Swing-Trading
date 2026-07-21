"""Guardrails for the Tushare daily-only raw price layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class DataCapabilityError(RuntimeError):
    """Raised when a requested research use is unsupported by raw daily data."""


UNSUPPORTED_CAPABILITIES = {
    "adjusted_close",
    "qfq_close",
    "hfq_close",
    "adjusted_return",
    "total_return",
    "dividend_adjusted_return",
    "portfolio_backtest_performance",
    "valuation",
    "market_cap",
    "pe",
    "pb",
    "dividend_yield",
    "pit_index_membership",
}

SUPPORTED_CAPABILITIES = {
    "raw_ohlcv",
    "raw_close",
    "raw_intraday_state_proxy",
    "liquidity_screening",
    "volume_activity_state",
    "coverage_check",
    "raw_pct_chg_diagnostic",
}


@dataclass(frozen=True)
class RawDailyOnlyAdapter:
    """Explicit adapter for daily-only raw OHLCV files.

    The adapter intentionally blocks adjusted-return and valuation use until
    adjustment factors, security master data, and valuation data are available.
    """

    data_root: Path

    def trade_date_path(self, trade_date: str) -> Path:
        return self.data_root / f"trade_date={trade_date}.csv"

    def require_capability(self, capability: str) -> None:
        if capability in UNSUPPORTED_CAPABILITIES:
            raise DataCapabilityError(
                f"Capability '{capability}' is not available from raw Tushare daily data. "
                "Resolve adj_factor/stock_basic/daily_basic before using it."
            )
        if capability not in SUPPORTED_CAPABILITIES:
            raise DataCapabilityError(
                f"Capability '{capability}' is not declared for the raw daily-only layer."
            )

    def load_trade_date(self, trade_date: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
        path = self.trade_date_path(trade_date)
        if not path.exists():
            raise FileNotFoundError(path)
        kwargs = {"encoding": "utf-8-sig", "low_memory": False}
        if columns is not None:
            requested = set(columns)
            kwargs["usecols"] = lambda col: col in requested
        return pd.read_csv(path, **kwargs)

    def get_close(
        self,
        trade_date: str,
        *,
        price_type: str = "raw_close",
    ) -> pd.DataFrame:
        if price_type != "raw_close":
            self.require_capability(price_type)
        data = self.load_trade_date(trade_date, columns=["ts_code", "trade_date", "close"])
        out = data.rename(columns={"ts_code": "asset", "close": "raw_close"})
        out["price_adjustment"] = "none_raw"
        out["adjusted_return_allowed"] = False
        return out

    def assert_can_run_portfolio_backtest(self, return_basis: str) -> None:
        if return_basis not in {"adjusted_return", "total_return"}:
            raise DataCapabilityError(
                "Portfolio backtests must use adjusted or total-return data. "
                f"Requested return_basis='{return_basis}' is not acceptable for model performance."
            )
        self.require_capability(return_basis)

    def compute_raw_return_diagnostic(self, panel: pd.DataFrame) -> pd.DataFrame:
        required = {"asset", "trade_date", "raw_close"}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"missing columns for raw return diagnostic: {sorted(missing)}")
        data = panel.copy()
        data = data.sort_values(["asset", "trade_date"])
        data["raw_close_return"] = data.groupby("asset")["raw_close"].pct_change()
        data["return_basis"] = "raw_close_diagnostic_only"
        data["adjusted_return_allowed"] = False
        return data
