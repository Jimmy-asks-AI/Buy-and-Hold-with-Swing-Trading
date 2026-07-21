#!/usr/bin/env python
"""Technical-indicator signal validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ref(series: pd.Series, n: int) -> pd.Series:
    """N-period lag, equivalent to REF(X, N)."""
    return series.shift(n)


def cross_over(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """True when left crosses above right at the current bar."""
    right_series = right if isinstance(right, pd.Series) else pd.Series(right, index=left.index)
    return (left.shift(1) <= right_series.shift(1)) & (left > right_series)


def cross_under(left: pd.Series, right: pd.Series | float) -> pd.Series:
    """True when left crosses below right at the current bar."""
    right_series = right if isinstance(right, pd.Series) else pd.Series(right, index=left.index)
    return (left.shift(1) >= right_series.shift(1)) & (left < right_series)


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average with highest weight on the latest value."""
    if window <= 0:
        raise ValueError("window must be positive.")
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def kdj_indicator(
    df: pd.DataFrame,
    high_col: str = "high_adj",
    low_col: str = "low_adj",
    close_col: str = "close_adj",
    window: int = 40,
    smooth: int = 3,
    prefix: str = "kdj",
) -> pd.DataFrame:
    """Compute KDJ-like RSV, K and D lines."""
    missing = [col for col in [high_col, low_col, close_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    low_n = out[low_col].rolling(window).min()
    high_n = out[high_col].rolling(window).max()
    denom = (high_n - low_n).replace(0.0, np.nan)
    out[f"{prefix}_low_n"] = low_n
    out[f"{prefix}_high_n"] = high_n
    out[f"{prefix}_rsv"] = (out[close_col] - low_n) / denom * 100.0
    alpha = 1.0 / smooth
    out[f"{prefix}_k"] = out[f"{prefix}_rsv"].ewm(alpha=alpha, adjust=False).mean()
    out[f"{prefix}_d"] = out[f"{prefix}_k"].ewm(alpha=alpha, adjust=False).mean()
    return out


def kdj_cross_signal(
    df: pd.DataFrame,
    k_col: str = "kdj_k",
    d_col: str = "kdj_d",
    output_col: str = "signal",
) -> pd.DataFrame:
    """Generate 1 for K crossing above D and 0 for K crossing below D."""
    missing = [col for col in [k_col, d_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out[output_col] = np.nan
    out.loc[cross_over(out[k_col], out[d_col]), output_col] = 1
    out.loc[cross_under(out[k_col], out[d_col]), output_col] = 0
    return out


def add_forward_return_labels(
    df: pd.DataFrame,
    price_col: str,
    horizons: list[int] | tuple[int, ...] = (1, 5, 10, 20),
    prefix: str = "fwd_return",
) -> pd.DataFrame:
    """Add future return labels for signal evaluation only."""
    if price_col not in df.columns:
        raise ValueError(f"Missing column: {price_col}")
    out = df.copy()
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive.")
        out[f"{prefix}_{horizon}"] = out[price_col].shift(-horizon) / out[price_col] - 1.0
        out[f"{prefix}_{horizon}_positive"] = out[f"{prefix}_{horizon}"] > 0
    return out


def evaluate_binary_signal(
    df: pd.DataFrame,
    signal_col: str,
    forward_return_cols: list[str],
) -> pd.DataFrame:
    """Summarize future returns after signal=1 and signal=0.

    This is signal validation, not a full capital-curve backtest.
    """
    required = [signal_col, *forward_return_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    clean = df.dropna(subset=[signal_col]).copy()
    for signal_value, group in clean.groupby(signal_col):
        for col in forward_return_cols:
            ret = group[col].dropna().astype(float)
            if ret.empty:
                continue
            rows.append(
                {
                    "signal": signal_value,
                    "horizon_col": col,
                    "count": int(ret.shape[0]),
                    "mean": float(ret.mean()),
                    "median": float(ret.median()),
                    "win_rate": float((ret > 0).mean()),
                    "loss_rate": float((ret < 0).mean()),
                    "p10": float(ret.quantile(0.10)),
                    "p90": float(ret.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def formula_operator_reference() -> pd.DataFrame:
    """Return common formula-to-pandas operator mappings."""
    rows = [
        ("REF(X,N)", "series.shift(N)", "N-period lag"),
        ("IF(COND,A,B)", "where/mask or loc assignment", "conditional assignment"),
        ("SUM(X,N)", "series.rolling(N).sum()", "rolling sum"),
        ("CUMSUM(X)", "series.cumsum()", "cumulative sum"),
        ("MAX(X,N)", "series.rolling(N).max()", "rolling max"),
        ("MIN(X,N)", "series.rolling(N).min()", "rolling min"),
        ("MAX(A,B)", "df[[A,B]].max(axis=1)", "row-wise max"),
        ("MIN(A,B)", "df[[A,B]].min(axis=1)", "row-wise min"),
        ("ABS(X)", "series.abs()", "absolute value"),
        ("MA(X,N)", "series.rolling(N).mean()", "simple moving average"),
        ("EMA(X,N)", "series.ewm(span=N, adjust=False).mean()", "exponential moving average"),
        ("WMA(X,N)", "weighted_moving_average(series, N)", "weighted moving average"),
    ]
    return pd.DataFrame(rows, columns=["formula_operator", "pandas_mapping", "meaning"])


def load_technical_indicator_catalog(path: str | Path) -> pd.DataFrame:
    """Load the 125-indicator Excel catalog into a clean table."""
    df = pd.read_excel(path, sheet_name=0, header=1)
    expected = ["指标名称", "计算公式", "指标描述", "买卖信号"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[expected].dropna(subset=["指标名称"]).reset_index(drop=True)
