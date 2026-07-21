#!/usr/bin/env python
"""Futures cross-asset and breakout strategy research helpers."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def normalize_gross_weights(weights: pd.Series | np.ndarray, dollar_neutral: bool = False) -> pd.Series:
    """Normalize weights by gross exposure, optionally removing net exposure."""
    out = pd.Series(weights, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if dollar_neutral:
        out = out - out.mean()
    gross = out.abs().sum()
    if gross <= EPS:
        return out * 0.0
    return out / gross


def intraday_money_flow(
    intraday: pd.DataFrame,
    date_col: str,
    price_col: str,
    amount_col: str,
    output_col: str = "money_flow",
) -> pd.DataFrame:
    """Compute signed intraday money flow normalized by total amount."""
    work = intraday.dropna(subset=[date_col, price_col, amount_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()
    work["_ret"] = work.groupby(date_col)[price_col].pct_change()
    work["_signed_amount"] = np.sign(work["_ret"].fillna(0.0)) * pd.to_numeric(work[amount_col], errors="coerce")
    agg = work.groupby(date_col).agg(signed=("_signed_amount", "sum"), total=(amount_col, "sum")).reset_index()
    agg[output_col] = agg["signed"] / agg["total"].replace(0.0, np.nan)
    return agg[[date_col, output_col]]


def first_canonical_correlation(
    x: pd.DataFrame,
    y: pd.DataFrame,
    regularization: float = 1e-6,
) -> dict[str, object]:
    """Estimate first CCA pair for factor matrix x and asset-return matrix y."""
    x_clean = x.apply(pd.to_numeric, errors="coerce")
    y_clean = y.apply(pd.to_numeric, errors="coerce")
    clean = pd.concat([x_clean.add_prefix("x_"), y_clean.add_prefix("y_")], axis=1).dropna()
    if clean.shape[0] <= max(x.shape[1], y.shape[1]) + 1:
        raise ValueError("Not enough observations for CCA.")
    xmat = clean.iloc[:, : x.shape[1]].to_numpy(dtype=float)
    ymat = clean.iloc[:, x.shape[1] :].to_numpy(dtype=float)
    xmat = xmat - xmat.mean(axis=0, keepdims=True)
    ymat = ymat - ymat.mean(axis=0, keepdims=True)
    n = len(clean) - 1
    sxx = xmat.T @ xmat / n + regularization * np.eye(xmat.shape[1])
    syy = ymat.T @ ymat / n + regularization * np.eye(ymat.shape[1])
    sxy = xmat.T @ ymat / n

    def inv_sqrt(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.maximum(vals, regularization)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    wx = inv_sqrt(sxx)
    wy = inv_sqrt(syy)
    m = wx @ sxy @ wy
    u, singular, vt = np.linalg.svd(m, full_matrices=False)
    x_weights = wx @ u[:, 0]
    y_weights = wy @ vt.T[:, 0]
    x_score = xmat @ x_weights
    y_score = ymat @ y_weights
    corr = np.corrcoef(x_score, y_score)[0, 1]
    return {
        "x_weights": pd.Series(x_weights, index=x.columns),
        "y_weights": pd.Series(y_weights, index=y.columns),
        "canonical_correlation": float(corr),
        "n_obs": int(clean.shape[0]),
    }


def cca_portfolio_weights(
    factor_history: pd.DataFrame,
    return_history: pd.DataFrame,
    dollar_neutral: bool = True,
    regularization: float = 1e-6,
) -> pd.Series:
    """Return normalized asset weights from the first CCA return-side vector."""
    cca = first_canonical_correlation(factor_history, return_history, regularization=regularization)
    return normalize_gross_weights(cca["y_weights"], dollar_neutral=dollar_neutral)


def linear_prediction_position(
    factor_row: pd.Series,
    regression_coef: pd.Series,
    asset_weights: pd.Series,
) -> pd.Series:
    """Sign a fixed asset spread by a linear factor prediction."""
    pred = float(pd.to_numeric(factor_row.reindex(regression_coef.index), errors="coerce").fillna(0.0) @ regression_coef)
    side = 1.0 if pred >= 0 else -1.0
    return normalize_gross_weights(asset_weights * side, dollar_neutral=True)


def intraday_extreme_signals(
    bars: pd.DataFrame,
    session_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    cutoff_bars: int = 15,
    shift_bars: int = 2,
    output_col: str = "intraday_extreme_signal",
) -> pd.DataFrame:
    """Generate trend-following signals from shifted intraday highs/lows."""
    if cutoff_bars < 0 or shift_bars < 0:
        raise ValueError("cutoff_bars and shift_bars must be non-negative.")
    out = bars.copy()
    out["_bar_no"] = out.groupby(session_col).cumcount() + 1
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    close = pd.to_numeric(out[close_col], errors="coerce")
    out["_v_high"] = high.groupby(out[session_col]).cummax().groupby(out[session_col]).shift(shift_bars)
    out["_v_low"] = low.groupby(out[session_col]).cummin().groupby(out[session_col]).shift(shift_bars)
    tradable = out["_bar_no"] >= cutoff_bars + 1
    out[output_col] = 0
    out.loc[tradable & (close >= out["_v_high"]), output_col] = 1
    out.loc[tradable & (close <= out["_v_low"]), output_col] = -1
    return out.drop(columns=["_bar_no"])


def volatility_contraction_breakout(
    bars: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    n_extreme: int = 26,
    n_low_vol: int = 7,
    vol_threshold: float = 5.0,
    vol_window: int | None = None,
    output_col: str = "signal",
) -> pd.DataFrame:
    """Approximate contraction-breakout signals with volatility envelopes."""
    if n_extreme <= 1 or n_low_vol <= 0:
        raise ValueError("n_extreme must be > 1 and n_low_vol must be positive.")
    out = bars.copy()
    vol_window = vol_window or n_extreme
    close = pd.to_numeric(out[close_col], errors="coerce")
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    vol = close.diff().rolling(vol_window).std()
    mid = close.rolling(vol_window).mean()
    upper = mid + vol
    lower = mid - vol
    upper_extreme = upper.rolling(n_extreme).max().shift(1)
    lower_extreme = lower.rolling(n_extreme).min().shift(1)
    low_vol_state = vol.rolling(n_low_vol).min().shift(1) < vol_threshold
    out["volatility"] = vol
    out["upper_envelope_extreme"] = upper_extreme
    out["lower_envelope_extreme"] = lower_extreme
    out[output_col] = 0
    out.loc[low_vol_state & (high > upper_extreme), output_col] = 1
    out.loc[low_vol_state & (low < lower_extreme), output_col] = -1
    return out


def sar_trailing_exit(
    bars: pd.DataFrame,
    high_col: str,
    low_col: str,
    entry_index: int,
    side: int,
    acceleration: float = 0.02,
    step: float = 0.02,
    max_acceleration: float = 0.2,
) -> pd.Series:
    """Compute SAR-like trailing exit path after an entry."""
    if side not in {1, -1}:
        raise ValueError("side must be 1 for long or -1 for short.")
    high = pd.to_numeric(bars[high_col], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(bars[low_col], errors="coerce").reset_index(drop=True)
    sar = pd.Series(np.nan, index=bars.index, dtype="float64")
    if entry_index >= len(bars):
        return sar
    af = acceleration
    if side == 1:
        sar.iloc[entry_index] = low.iloc[entry_index]
        extreme = high.iloc[entry_index]
        for i in range(entry_index + 1, len(bars)):
            sar.iloc[i] = sar.iloc[i - 1] + (extreme - sar.iloc[i - 1]) * af
            if high.iloc[i] > extreme:
                extreme = high.iloc[i]
                af = min(max_acceleration, af + step)
    else:
        sar.iloc[entry_index] = high.iloc[entry_index]
        extreme = low.iloc[entry_index]
        for i in range(entry_index + 1, len(bars)):
            sar.iloc[i] = sar.iloc[i - 1] - (sar.iloc[i - 1] - extreme) * af
            if low.iloc[i] < extreme:
                extreme = low.iloc[i]
                af = min(max_acceleration, af + step)
    return sar


def max_drawdown(equity: pd.Series) -> float:
    curve = pd.to_numeric(equity, errors="coerce").dropna()
    if curve.empty:
        return np.nan
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def streak_lengths(wins: Iterable[bool]) -> tuple[int, int]:
    """Return max winning streak and max losing streak."""
    max_win = max_loss = cur_win = cur_loss = 0
    for win in wins:
        if win:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def trade_metrics(trade_returns: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Summarize trade-level strategy performance."""
    r = pd.Series(trade_returns, dtype="float64").dropna()
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    wins = r > 0
    avg_win = r[wins].mean()
    avg_loss = r[~wins].mean()
    max_win_streak, max_loss_streak = streak_lengths(wins.tolist())
    return {
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "trade_count": int(r.shape[0]),
        "win_count": int(wins.sum()),
        "loss_count": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "average_win": float(avg_win) if not pd.isna(avg_win) else np.nan,
        "average_loss": float(avg_loss) if not pd.isna(avg_loss) else np.nan,
        "profit_loss_ratio": float(abs(avg_win / avg_loss)) if avg_loss and not pd.isna(avg_loss) else np.nan,
        "max_drawdown": max_drawdown(equity),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
    }


def equity_slope_score(trade_returns: pd.Series | np.ndarray) -> float:
    """Slope of cumulative equity regressed on trade number."""
    r = pd.Series(trade_returns, dtype="float64").dropna()
    if r.shape[0] < 2:
        return np.nan
    equity = (1.0 + r).cumprod().to_numpy(dtype=float)
    x = np.arange(1, len(equity) + 1, dtype=float)
    slope = np.polyfit(x, equity, 1)[0]
    return float(slope)


def futures_strategy_checklist() -> pd.DataFrame:
    rows = [
        ("spread", "Prefer relative-return prediction when outright index returns are noisy."),
        ("training", "Separate factor-selection and backtest periods."),
        ("cca", "Normalize CCA asset weights and monitor net exposure."),
        ("execution", "Model close-to-close, next-bar, fees, impact, and margin separately."),
        ("breakout", "Breakout signals need explicit stop and trailing exit rules."),
        ("parameters", "Stress N1, N2, and volatility thresholds across grids."),
        ("objective", "Optimize equity slope or risk-adjusted growth, not terminal return alone."),
        ("streak", "Track max consecutive losses for strategy shutdown rules."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.checklist:
        print(futures_strategy_checklist())


if __name__ == "__main__":
    main()
