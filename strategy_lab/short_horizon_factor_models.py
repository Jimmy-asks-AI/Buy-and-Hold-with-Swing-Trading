#!/usr/bin/env python
"""Short-horizon factor research helpers.

This module collects reusable tools for:

- time-series factor exposure estimation,
- cross-sectional factor-return estimation,
- short-horizon price-volume alpha construction,
- dynamic reversal windows from market swings,
- Level2 trade-share and net-buy factors.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-12


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    if weight is None:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    w = np.sqrt(np.maximum(weight.astype(float), 0.0))
    return np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]


def idiosyncratic_momentum_factor(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
    common_factor_cols: Iterable[str],
    lookback: int = 12,
    min_periods: int = 8,
    output_col: str = "imom",
    risk_adjust: bool = True,
) -> pd.DataFrame:
    """Compute residual momentum after removing common factor returns.

    The report evidence favors residual momentum stripped by common factors
    over raw past return momentum.  The signal is lagged by construction:
    each score only uses returns before the current row.
    """
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    factors = list(common_factor_cols)
    if not factors:
        raise ValueError("common_factor_cols cannot be empty.")
    out = panel.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan
    cols = [return_col, *factors]
    min_obs = max(min_periods, len(factors) + 3)
    for _, group in out.groupby(asset_col, sort=False):
        values = group[cols].apply(pd.to_numeric, errors="coerce")
        scores = pd.Series(np.nan, index=group.index, dtype="float64")
        for pos in range(len(group)):
            hist = values.iloc[max(0, pos - lookback) : pos].dropna()
            if hist.shape[0] < min_obs:
                continue
            x = hist[factors].to_numpy(dtype=float)
            x = np.column_stack([np.ones(hist.shape[0]), x])
            y = hist[return_col].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            resid = y - x @ beta
            if risk_adjust:
                denom = resid.std(ddof=1)
                scores.iloc[pos] = resid.mean() / denom if denom > EPS else np.nan
            else:
                scores.iloc[pos] = resid.mean()
        out.loc[group.index, output_col] = scores.to_numpy(dtype=float)
    return out


def market_state_for_momentum(
    market: pd.DataFrame,
    date_col: str,
    market_return_col: str,
    lookback: int = 12,
    output_col: str = "momentum_state",
) -> pd.DataFrame:
    """Label market states where residual momentum is likely to weaken."""
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    out = market.sort_values(date_col).copy()
    returns = pd.to_numeric(out[market_return_col], errors="coerce")
    trailing = returns.rolling(lookback, min_periods=lookback).mean()
    next_ret = returns.shift(-1)
    prev_label = np.where(trailing >= 0.0, "up", "down")
    next_label = np.where(next_ret >= 0.0, "continue", "reverse")
    out[output_col] = [
        f"{prev}_{nxt}" if not pd.isna(prev_value) and not pd.isna(next_value) else np.nan
        for prev, nxt, prev_value, next_value in zip(prev_label, next_label, trailing, next_ret)
    ]
    return out


def _weighted_r2(y: np.ndarray, y_hat: np.ndarray, weight: np.ndarray | None = None) -> float:
    if weight is None:
        y_bar = y.mean()
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_bar) ** 2)
    else:
        w = np.maximum(weight.astype(float), 0.0)
        if w.sum() <= EPS:
            return np.nan
        y_bar = np.average(y, weights=w)
        ss_res = np.sum(w * (y - y_hat) ** 2)
        ss_tot = np.sum(w * (y - y_bar) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan


def time_series_factor_exposure(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: int = 252,
    min_obs: int | None = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Estimate rolling Fama-French-style time-series factor exposures.

    Returns a long table with one row per asset and window end date.
    Factor returns are inputs; asset exposures are estimated outputs.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    min_obs = min_obs or max(20, window // 2)
    factors = _numeric_frame(factor_returns)
    assets = _numeric_frame(asset_returns)
    common_index = assets.index.intersection(factors.index)
    factors = factors.loc[common_index].sort_index()
    assets = assets.loc[common_index].sort_index()
    factor_cols = list(factors.columns)
    rows: list[dict[str, object]] = []
    for asset in assets.columns:
        data = pd.concat([assets[asset].rename("_asset_return"), factors], axis=1)
        for end in range(window, len(data) + 1):
            sample = data.iloc[end - window : end].dropna()
            if sample.shape[0] < min_obs:
                continue
            x = sample[factor_cols].to_numpy(dtype=float)
            if add_intercept:
                x = np.column_stack([np.ones(sample.shape[0]), x])
            y = sample["_asset_return"].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            row: dict[str, object] = {
                "date": data.index[end - 1],
                "asset": asset,
                "n_obs": int(sample.shape[0]),
                "r2": _weighted_r2(y, x @ beta),
            }
            offset = 0
            if add_intercept:
                row["alpha"] = float(beta[0])
                offset = 1
            for col, value in zip(factor_cols, beta[offset:]):
                row[f"{col}_exposure"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def cross_sectional_factor_returns(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    exposure_cols: Iterable[str],
    weight_col: str | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Estimate Barra-style cross-sectional factor returns by date.

    Factor exposures are known before the return period; factor returns are
    estimated from the cross section of realized returns.
    """
    exposures = list(exposure_cols)
    min_count = min_count or len(exposures) + 3
    need = [date_col, return_col, *exposures]
    if weight_col:
        need.append(weight_col)
    rows: list[dict[str, object]] = []
    for date, group in panel[need].groupby(date_col, sort=True):
        clean = group.dropna(subset=[return_col, *exposures]).copy()
        if clean.shape[0] < min_count:
            continue
        x = clean[exposures].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(clean[return_col], errors="coerce").to_numpy(dtype=float)
        x = np.column_stack([np.ones(clean.shape[0]), x])
        weight = None
        if weight_col:
            weight = pd.to_numeric(clean[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        beta = _weighted_lstsq(x, y, weight=weight)
        row: dict[str, object] = {
            "date": date,
            "n_obs": int(clean.shape[0]),
            "intercept_return": float(beta[0]),
            "r2": _weighted_r2(y, x @ beta, weight=weight),
        }
        for col, value in zip(exposures, beta[1:]):
            row[f"{col}_return"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_corr_by_asset(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    x_col: str,
    y_col: str,
    window: int,
    output_col: str,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute rolling within-asset correlation."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    min_periods = min_periods or window
    out = df.sort_values([asset_col, date_col]).copy()
    pieces = []
    for _, group in out.groupby(asset_col, sort=False):
        x = pd.to_numeric(group[x_col], errors="coerce")
        y = pd.to_numeric(group[y_col], errors="coerce")
        corr = x.rolling(window, min_periods=min_periods).corr(y)
        pieces.append(pd.Series(corr.to_numpy(dtype=float), index=group.index))
    out[output_col] = pd.concat(pieces).sort_index()
    return out


def price_volume_divergence(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    price_col: str = "vwap",
    volume_col: str = "volume",
    window: int = 6,
    output_col: str = "price_volume_divergence",
) -> pd.DataFrame:
    """Negative rolling correlation between VWAP-like price and volume."""
    out = rolling_corr_by_asset(df, asset_col, date_col, price_col, volume_col, window, output_col)
    out[output_col] = -out[output_col]
    return out


def opening_gap(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    open_col: str = "open",
    close_col: str = "close",
    output_col: str = "opening_gap",
) -> pd.DataFrame:
    """Open divided by previous close minus one."""
    out = df.sort_values([asset_col, date_col]).copy()
    prev_close = out.groupby(asset_col)[close_col].shift(1)
    out[output_col] = pd.to_numeric(out[open_col], errors="coerce") / prev_close - 1.0
    return out


def abnormal_volume(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    volume_col: str = "volume",
    window: int = 6,
    lag_reference: bool = True,
    output_col: str = "abnormal_volume",
) -> pd.DataFrame:
    """Negative current volume divided by its rolling historical mean."""
    out = df.sort_values([asset_col, date_col]).copy()
    values = []
    for _, group in out.groupby(asset_col, sort=False):
        volume = pd.to_numeric(group[volume_col], errors="coerce")
        reference = volume.rolling(window, min_periods=window).mean()
        if lag_reference:
            reference = reference.shift(1)
        values.append(pd.Series((-volume / reference.replace(0.0, np.nan)).to_numpy(dtype=float), index=group.index))
    out[output_col] = pd.concat(values).sort_index()
    return out


def volume_amplitude_divergence(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    window: int = 6,
    output_col: str = "volume_amplitude_divergence",
) -> pd.DataFrame:
    """Negative rolling correlation between high/low amplitude and volume."""
    out = df.copy()
    out["_amplitude"] = pd.to_numeric(out[high_col], errors="coerce") / pd.to_numeric(out[low_col], errors="coerce")
    out = rolling_corr_by_asset(out, asset_col, date_col, "_amplitude", volume_col, window, output_col)
    out[output_col] = -out[output_col]
    return out.drop(columns=["_amplitude"])


def volume_price_correlation(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    close_col: str = "close",
    turnover_col: str = "turnover",
    window: int = 10,
    output_col: str = "volume_price_corr",
) -> pd.DataFrame:
    """Pearson correlation between adjusted close and turnover over a short window."""
    return rolling_corr_by_asset(df, asset_col, date_col, close_col, turnover_col, window, output_col)


def volume_price_pattern(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    close_col: str = "close",
    corr_col: str = "volume_price_corr",
    return_window: int = 10,
    corr_threshold: float = 0.0,
    return_threshold: float = 0.0,
    output_col: str = "volume_price_pattern",
) -> pd.DataFrame:
    """Classify short-term volume-price states into four interpretable patterns."""
    out = df.sort_values([asset_col, date_col]).copy()
    if corr_col not in out.columns:
        raise KeyError(f"{corr_col} is required. Run volume_price_correlation first.")
    past_returns = []
    for _, group in out.groupby(asset_col, sort=False):
        close = pd.to_numeric(group[close_col], errors="coerce")
        ret = close / close.shift(return_window) - 1.0
        past_returns.append(pd.Series(ret.to_numpy(dtype=float), index=group.index))
    out["_pattern_return"] = pd.concat(past_returns).sort_index()

    corr = pd.to_numeric(out[corr_col], errors="coerce")
    ret = pd.to_numeric(out["_pattern_return"], errors="coerce")
    out[output_col] = "neutral"
    out.loc[(corr <= -corr_threshold) & (ret <= -return_threshold), output_col] = "volume_down"
    out.loc[(corr <= -corr_threshold) & (ret >= return_threshold), output_col] = "shrinking_up"
    out.loc[(corr >= corr_threshold) & (ret >= return_threshold), output_col] = "volume_up"
    out.loc[(corr >= corr_threshold) & (ret <= -return_threshold), output_col] = "shrinking_down"
    out.loc[corr.isna() | ret.isna(), output_col] = np.nan
    return out.drop(columns=["_pattern_return"])


def price_shape_factors(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    vwap_col: str = "vwap",
    window: int = 10,
    lag: int = 1,
    include_square: bool = True,
    prefix: str = "shape",
) -> pd.DataFrame:
    """Build rolling price-shape factors from OHLCV prices.

    Raw daily components:
    - high/open: intraday run-up after open,
    - close/low: rebound from intraday low,
    - vwap/close: VWAP deviation from close.
    """
    if window <= 0 or lag < 0:
        raise ValueError("window must be positive and lag must be non-negative.")
    out = df.sort_values([asset_col, date_col]).copy()
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    open_ = pd.to_numeric(out[open_col], errors="coerce")
    close = pd.to_numeric(out[close_col], errors="coerce")
    vwap = pd.to_numeric(out[vwap_col], errors="coerce")
    out["_run_up"] = np.log(high / open_.replace(0.0, np.nan))
    out["_low_rebound"] = np.log(close / low.replace(0.0, np.nan))
    out["_vwap_close"] = np.log(vwap / close.replace(0.0, np.nan))
    mapping = {
        "_run_up": f"{prefix}_run_up",
        "_low_rebound": f"{prefix}_low_rebound",
        "_vwap_close": f"{prefix}_vwap_close",
    }
    for raw_col, factor_col in mapping.items():
        pieces = []
        for _, group in out.groupby(asset_col, sort=False):
            value = pd.to_numeric(group[raw_col], errors="coerce").rolling(window, min_periods=window).mean().shift(lag)
            pieces.append(pd.Series(value.to_numpy(dtype=float), index=group.index))
        out[factor_col] = pd.concat(pieces).sort_index()
        if include_square:
            out[f"{factor_col}_square"] = out[factor_col] ** 2
    return out.drop(columns=["_run_up", "_low_rebound", "_vwap_close"])


def intraday_return_moments(
    bars: pd.DataFrame,
    asset_col: str,
    session_col: str,
    price_col: str,
    time_col: str | None = None,
    method: str = "raw",
    min_bars: int = 30,
) -> pd.DataFrame:
    """Compute intraday realized variance, skewness, and kurtosis by asset-session."""
    if method not in {"raw", "centered"}:
        raise ValueError("method must be 'raw' or 'centered'.")
    sort_cols = [asset_col, session_col] + ([time_col] if time_col else [])
    work = bars.sort_values(sort_cols, kind="mergesort").copy()
    work["_log_price"] = np.log(pd.to_numeric(work[price_col], errors="coerce"))
    work["_ret"] = work.groupby([asset_col, session_col])["_log_price"].diff()
    rows: list[dict[str, object]] = []
    for (asset, session), group in work.dropna(subset=["_ret"]).groupby([asset_col, session_col], sort=True):
        r = group["_ret"].to_numpy(dtype=float)
        if r.size < min_bars:
            continue
        if method == "centered":
            r = r - r.mean()
        rv = float(np.sum(r ** 2))
        if rv <= EPS:
            skew = np.nan
            kurt = np.nan
        else:
            n = r.size
            skew = float(n * np.sum(r ** 3) / (rv ** 1.5))
            kurt = float(n * np.sum(r ** 4) / (rv ** 2))
        rows.append(
            {
                asset_col: asset,
                session_col: session,
                "intraday_var": rv,
                "intraday_skew": skew,
                "intraday_kurt": kurt,
                "bar_count": int(r.size),
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def rolling_intraday_moment_factors(
    daily_moments: pd.DataFrame,
    asset_col: str,
    date_col: str,
    moment_cols: Iterable[str] = ("intraday_var", "intraday_skew", "intraday_kurt"),
    window: int = 21,
    lag: int = 1,
    prefix: str = "hf",
) -> pd.DataFrame:
    """Average daily intraday moments into monthly-like factors."""
    out = daily_moments.sort_values([asset_col, date_col]).copy()
    for col in moment_cols:
        pieces = []
        for _, group in out.groupby(asset_col, sort=False):
            value = pd.to_numeric(group[col], errors="coerce").rolling(window, min_periods=window).mean().shift(lag)
            pieces.append(pd.Series(value.to_numpy(dtype=float), index=group.index))
        out[f"{prefix}_{col}"] = pd.concat(pieces).sort_index()
    if f"{prefix}_intraday_skew" in out.columns:
        out[f"{prefix}_skew_alpha"] = -out[f"{prefix}_intraday_skew"]
    return out


def intraday_upside_downside_volatility(
    bars: pd.DataFrame,
    asset_col: str,
    session_col: str,
    price_col: str,
    time_col: str | None = None,
    min_bars: int = 30,
) -> pd.DataFrame:
    """Decompose intraday realized variance into upside and downside parts."""
    sort_cols = [asset_col, session_col] + ([time_col] if time_col else [])
    work = bars.sort_values(sort_cols, kind="mergesort").copy()
    work["_log_price"] = np.log(pd.to_numeric(work[price_col], errors="coerce"))
    work["_ret"] = work.groupby([asset_col, session_col])["_log_price"].diff()
    rows: list[dict[str, object]] = []
    for (asset, session), group in work.dropna(subset=["_ret"]).groupby([asset_col, session_col], sort=True):
        r = group["_ret"].to_numpy(dtype=float)
        if r.size < min_bars:
            continue
        total_var = float(np.sum(r ** 2))
        up_var = float(np.sum((r[r > 0]) ** 2))
        down_var = float(np.sum((r[r < 0]) ** 2))
        rows.append(
            {
                asset_col: asset,
                session_col: session,
                "intraday_total_var": total_var,
                "intraday_up_var": up_var,
                "intraday_down_var": down_var,
                "intraday_up_vol": float(np.sqrt(up_var)),
                "intraday_down_vol": float(np.sqrt(down_var)),
                "intraday_up_var_ratio": up_var / total_var if total_var > EPS else np.nan,
                "intraday_down_var_ratio": down_var / total_var if total_var > EPS else np.nan,
                "bar_count": int(r.size),
            }
        )
    return pd.DataFrame(rows)


def rolling_intraday_volatility_decomposition_factors(
    daily_decomposition: pd.DataFrame,
    asset_col: str,
    date_col: str,
    window: int = 21,
    lag: int = 1,
    prefix: str = "hf",
) -> pd.DataFrame:
    """Roll daily upside/downside volatility decomposition into factor values."""
    cols = ["intraday_total_var", "intraday_up_var", "intraday_down_var", "intraday_up_var_ratio"]
    out = daily_decomposition.sort_values([asset_col, date_col]).copy()
    for col in cols:
        pieces = []
        for _, group in out.groupby(asset_col, sort=False):
            value = pd.to_numeric(group[col], errors="coerce").rolling(window, min_periods=window).mean().shift(lag)
            pieces.append(pd.Series(value.to_numpy(dtype=float), index=group.index))
        out[f"{prefix}_{col}"] = pd.concat(pieces).sort_index()
    out[f"{prefix}_upside_volatility_alpha"] = -out[f"{prefix}_intraday_up_var_ratio"]
    return out


def turnover_coefficient_variation(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    turnover_col: str = "turnover",
    window: int = 20,
    output_col: str = "turnover_cv",
    factor_col: str | None = "turnover_cv_stability",
) -> pd.DataFrame:
    """Rolling coefficient of variation for turnover.

    The raw coefficient of variation is bad in the cited evidence: higher CV
    predicts lower future return. ``factor_col`` stores ``-CV`` so larger is
    better for rank-based alpha models.
    """
    out = df.sort_values([asset_col, date_col]).copy()
    values = []
    for _, group in out.groupby(asset_col, sort=False):
        turnover = pd.to_numeric(group[turnover_col], errors="coerce")
        mean = turnover.rolling(window, min_periods=window).mean()
        std = turnover.rolling(window, min_periods=window).std()
        cv = std / mean.replace(0.0, np.nan)
        values.append(pd.Series(cv.to_numpy(dtype=float), index=group.index))
    out[output_col] = pd.concat(values).sort_index()
    if factor_col:
        out[factor_col] = -out[output_col]
    return out


def lottery_max_return_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    close_col: str = "close",
    window: int = 21,
    top_n: int = 1,
    output_col: str = "lottery_max_return",
    factor_col: str | None = "lottery_avoidance",
) -> pd.DataFrame:
    """Rolling maximum or top-N average daily return.

    High recent extreme upside is a lottery-stock proxy and is bad in the
    cited evidence. ``factor_col`` stores the negative value for ranking.
    """
    if window <= 1 or top_n <= 0:
        raise ValueError("window must be greater than 1 and top_n must be positive.")
    out = df.sort_values([asset_col, date_col]).copy()
    values = []

    def top_n_average(x: np.ndarray) -> float:
        clean = x[~np.isnan(x)]
        if clean.size < min(window, top_n):
            return np.nan
        return float(np.sort(clean)[-top_n:].mean())

    for _, group in out.groupby(asset_col, sort=False):
        close = pd.to_numeric(group[close_col], errors="coerce")
        daily_ret = close.pct_change()
        value = daily_ret.rolling(window, min_periods=window).apply(top_n_average, raw=True)
        values.append(pd.Series(value.to_numpy(dtype=float), index=group.index))
    out[output_col] = pd.concat(values).sort_index()
    if factor_col:
        out[factor_col] = -out[output_col]
    return out


def double_sort_factor_returns(
    df: pd.DataFrame,
    date_col: str,
    target_factor_col: str,
    control_factor_col: str,
    forward_return_col: str,
    control_groups: int = 5,
    target_groups: int = 5,
    min_count: int | None = None,
    low_minus_high: bool = True,
) -> pd.DataFrame:
    """Double-sort by a control factor first and target factor second."""
    min_count = min_count or control_groups * target_groups
    rows: list[dict[str, object]] = []
    need = [date_col, target_factor_col, control_factor_col, forward_return_col]
    for date, group in df[need].dropna().groupby(date_col, sort=True):
        if group.shape[0] < min_count:
            continue
        try:
            control_bucket = pd.qcut(
                group[control_factor_col].rank(method="first"),
                control_groups,
                labels=False,
            )
        except ValueError:
            continue
        bucket_returns: dict[int, list[float]] = {i: [] for i in range(1, target_groups + 1)}
        for _, sub in group.assign(_control=control_bucket).groupby("_control", sort=False):
            if sub.shape[0] < target_groups:
                continue
            try:
                target_bucket = pd.qcut(
                    sub[target_factor_col].rank(method="first"),
                    target_groups,
                    labels=False,
                ) + 1
            except ValueError:
                continue
            temp = sub.assign(_target=target_bucket.astype(int))
            for bucket, value in temp.groupby("_target")[forward_return_col].mean().items():
                bucket_returns[int(bucket)].append(float(value))
        averaged = {bucket: np.mean(values) for bucket, values in bucket_returns.items() if values}
        for bucket, value in averaged.items():
            rows.append({"date": date, "bucket": int(bucket), "return": float(value)})
        if 1 in averaged and target_groups in averaged:
            spread = averaged[1] - averaged[target_groups] if low_minus_high else averaged[target_groups] - averaged[1]
            rows.append({"date": date, "bucket": "long_short", "return": float(spread)})
    return pd.DataFrame(rows)


def regime_conditioned_performance(
    factor_returns: pd.DataFrame,
    date_col: str,
    factor_return_cols: Iterable[str],
    market_return_col: str,
    state_lag: int = 1,
) -> pd.DataFrame:
    """Summarize factor returns after previous market up/down states."""
    out = factor_returns.sort_values(date_col).copy()
    market = pd.to_numeric(out[market_return_col], errors="coerce")
    out["_market_state"] = np.where(market.shift(state_lag) >= 0.0, "up", "down")
    rows = []
    for col in factor_return_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        for state, group in out.assign(_value=values).dropna(subset=["_value"]).groupby("_market_state"):
            v = group["_value"]
            std = v.std(ddof=1)
            rows.append(
                {
                    "factor": col,
                    "state": state,
                    "n_obs": int(v.shape[0]),
                    "mean": float(v.mean()),
                    "std": float(std),
                    "t_stat": float(v.mean() / (std / np.sqrt(v.shape[0]))) if std > EPS else np.nan,
                    "win_rate": float((v > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def factor_zoo_definitions() -> pd.DataFrame:
    """Compact factor-zoo map from the factor lecture report."""
    rows = [
        ("monthly", "Lagged Momentum", "holding return from t-12 to t-7"),
        ("monthly", "Long-Term Reversal", "holding return from t-60 to t-13"),
        ("monthly", "Momentum", "holding return from t-6 to t-1"),
        ("monthly", "Momentum-Reversal", "holding return from t-18 to t-13"),
        ("monthly", "Momentum-Volume", "momentum among top turnover names"),
        ("monthly", "Short-Term Reversal", "last-month holding return"),
        ("monthly", "Seasonality", "historical average return in the same calendar month"),
        ("monthly", "Volume/Market Value of Equity", "monthly trading amount divided by float shares"),
        ("monthly", "Volume Variance", "three-year standard deviation of trading amount"),
        ("monthly", "Share Volume", "daily volume divided by float shares, three-month average"),
        ("monthly", "52-Week High", "close divided by one-year high"),
        ("monthly", "Max", "maximum daily return in the previous month"),
        ("monthly", "Amihud's Measure", "absolute daily return divided by trading amount"),
        ("monthly", "Coskewness", "co-skewness with market return"),
        ("monthly", "Idiosyncratic Risk", "standard deviation of last-month three-factor residual"),
        ("monthly", "Size", "log market capitalization"),
        ("monthly", "Beta", "covariance with market divided by market variance"),
        ("monthly", "Price", "log close price"),
        ("quarterly", "Market Value", "market capitalization"),
        ("quarterly", "Book Equity/Market", "book equity divided by market capitalization"),
        ("quarterly", "Sales/Price", "revenue divided by market price proxy"),
        ("quarterly", "Cash Flow/Market Value of Equity", "net income plus depreciation and amortization over market cap"),
        ("quarterly", "Earnings/Price", "net income divided by market capitalization"),
        ("quarterly", "Gross Profitability", "gross profit divided by lagged total assets"),
        ("quarterly", "Profitability", "net income divided by lagged total assets"),
        ("quarterly", "Percent Total Accrual", "net income minus operating cash flow over absolute net income"),
        ("quarterly", "Return-on-Equity", "net income divided by shareholder equity"),
        ("quarterly", "G-Score", "sum of eight profitability and balance-sheet quality indicators"),
        ("quarterly", "Asset Growth", "total assets growth"),
        ("quarterly", "Change in Profit Margin", "change in net profit margin"),
        ("quarterly", "Earnings Consistency", "five-period earnings consistency score"),
        ("quarterly", "Earnings Surprise", "earnings surprise scaled by historical volatility"),
        ("quarterly", "Growth in Inventory", "inventory growth divided by average assets"),
        ("quarterly", "Growth in LTNOA", "growth in long-term net operating assets minus accruals"),
        ("quarterly", "Revenue Surprises", "revenue surprise scaled like earnings surprise"),
        ("quarterly", "Sales Growth", "ranked weighted-average revenue growth"),
        ("quarterly", "Sustainable Growth", "shareholder equity growth"),
        ("quarterly", "Leverage", "noncurrent liabilities divided by market capitalization"),
        ("quarterly", "Net Operating Assets", "operating assets minus operating liabilities over total assets"),
        ("quarterly", "Net Working Capital Changes", "change in non-cash working capital over total assets"),
        ("quarterly", "Noncurrent Operating Assets Changes", "change in noncurrent operating assets over total assets"),
        ("quarterly", "Asset Turnover", "revenue divided by average net operating assets"),
        ("quarterly", "Change in Asset Turnover", "change in asset turnover"),
        ("quarterly", "Profit Margin", "EBIT divided by revenue"),
        ("quarterly", "Enterprise Component of Book/Price", "enterprise component of book-to-price"),
        ("quarterly", "Enterprise Multiple", "enterprise value divided by operating cash flow"),
        ("quarterly", "Leverage Component of Book/Price", "leverage component of book-to-price"),
        ("quarterly", "Operating Leverage", "operating cost and expenses divided by lagged assets"),
        ("quarterly", "Tax", "income tax divided by net income"),
        ("quarterly", "Accruals", "balance-sheet accruals divided by average total assets"),
        ("quarterly", "R&D/Market Value of Equity", "research and development expense divided by market cap"),
    ]
    return pd.DataFrame(rows, columns=["frequency", "factor", "definition"])


def rolling_factor_return_forecast(
    factor_returns: pd.DataFrame,
    date_col: str,
    value_cols: Iterable[str],
    window: int = 250,
    min_periods: int | None = None,
    shift: int = 1,
    suffix: str = "_forecast",
) -> pd.DataFrame:
    """Use trailing average factor returns as next-period forecasts."""
    if window <= 0:
        raise ValueError("window must be positive.")
    min_periods = min_periods or max(20, window // 2)
    cols = list(value_cols)
    out = factor_returns.sort_values(date_col).copy()
    for col in cols:
        out[f"{col}{suffix}"] = (
            pd.to_numeric(out[col], errors="coerce").rolling(window, min_periods=min_periods).mean().shift(shift)
        )
    return out


def transaction_cost_adjusted_objective(
    expected_alpha: pd.Series,
    candidate_weights: pd.Series,
    current_weights: pd.Series | None = None,
    cost_rate: float = 0.003,
    turnover_divisor: float = 2.0,
) -> dict[str, float]:
    """Evaluate alpha minus transaction-cost penalty for a candidate portfolio."""
    alpha = pd.to_numeric(expected_alpha, errors="coerce").fillna(0.0)
    candidate = pd.to_numeric(candidate_weights.reindex(alpha.index), errors="coerce").fillna(0.0)
    if current_weights is None:
        current = pd.Series(0.0, index=alpha.index)
    else:
        current = pd.to_numeric(current_weights.reindex(alpha.index), errors="coerce").fillna(0.0)
    gross_alpha = float(alpha @ candidate)
    turnover = float((candidate - current).abs().sum() / turnover_divisor)
    cost = float(cost_rate * turnover)
    return {
        "gross_alpha": gross_alpha,
        "turnover": turnover,
        "cost": cost,
        "objective": gross_alpha - cost,
    }


def rolling_swing_threshold(
    index_close: pd.Series,
    lookback: int = 252,
    std_multiple: float = 1.0,
    annualization_horizon: int = 20,
    min_threshold: float = 0.03,
    max_threshold: float = 0.25,
) -> pd.Series:
    """Convert daily index volatility into a swing threshold."""
    close = pd.to_numeric(index_close, errors="coerce").sort_index()
    returns = close.pct_change()
    threshold = returns.rolling(lookback, min_periods=max(20, lookback // 4)).std().shift(1)
    threshold = threshold * np.sqrt(annualization_horizon) * std_multiple
    return threshold.clip(lower=min_threshold, upper=max_threshold).fillna(min_threshold)


def market_swing_windows(
    index_close: pd.Series,
    threshold: float | pd.Series = 0.05,
    min_turn_window: int = 20,
) -> pd.DataFrame:
    """Segment market swings and return days since the latest swing extreme."""
    close = pd.to_numeric(index_close, errors="coerce").dropna().sort_index()
    if close.empty:
        return pd.DataFrame(columns=["date", "reversal_window", "swing_direction", "turning_point"])
    if isinstance(threshold, pd.Series):
        threshold_series = pd.to_numeric(threshold, errors="coerce").reindex(close.index).ffill().fillna(0.05)
    else:
        threshold_series = pd.Series(float(threshold), index=close.index)

    direction = 0
    high_price = low_price = float(close.iloc[0])
    high_pos = low_pos = last_turn_pos = 0
    rows: list[dict[str, object]] = []
    for i, (date, price_raw) in enumerate(close.items()):
        price = float(price_raw)
        th = max(float(threshold_series.iloc[i]), EPS)
        turning_point = False

        if price >= high_price:
            high_price = price
            high_pos = i
        if price <= low_price:
            low_price = price
            low_pos = i

        if direction >= 0 and high_price > EPS:
            drawdown = price / high_price - 1.0
            if drawdown <= -th and i - last_turn_pos >= min_turn_window:
                direction = -1
                last_turn_pos = high_pos
                low_price = price
                low_pos = i
                turning_point = True
        if direction <= 0 and low_price > EPS:
            rebound = price / low_price - 1.0
            if rebound >= th and i - last_turn_pos >= min_turn_window:
                direction = 1
                last_turn_pos = low_pos
                high_price = price
                high_pos = i
                turning_point = True

        rows.append(
            {
                "date": date,
                "reversal_window": max(1, i - last_turn_pos),
                "swing_direction": int(direction),
                "turning_point": bool(turning_point),
            }
        )
    return pd.DataFrame(rows)


def dynamic_reversal_factor(
    prices: pd.DataFrame,
    asset_col: str,
    date_col: str,
    close_col: str,
    window_by_date: pd.Series | pd.DataFrame,
    min_window: int = 20,
    output_col: str = "dynamic_reversal",
) -> pd.DataFrame:
    """Compute negative past return over the market-swing-defined window."""
    out = prices.sort_values([asset_col, date_col]).copy()
    if isinstance(window_by_date, pd.DataFrame):
        if "date" in window_by_date.columns:
            window_map = window_by_date.set_index("date")["reversal_window"]
        else:
            window_map = window_by_date["reversal_window"]
    else:
        window_map = window_by_date
    window_map = pd.to_numeric(window_map, errors="coerce")

    factor_values = []
    for _, group in out.groupby(asset_col, sort=False):
        close = pd.to_numeric(group[close_col], errors="coerce").to_numpy(dtype=float)
        dates = group[date_col].to_numpy()
        values = np.full(len(group), np.nan, dtype=float)
        for i, date in enumerate(dates):
            if date not in window_map.index:
                continue
            window = int(window_map.loc[date])
            if window < min_window or i < window or close[i - window] <= EPS:
                continue
            values[i] = -(close[i] / close[i - window] - 1.0)
        factor_values.append(pd.Series(values, index=group.index))
    out[output_col] = pd.concat(factor_values).sort_index()
    return out


def level2_trade_share_factors(
    df: pd.DataFrame,
    total_amount_col: str,
    amount_cols: Mapping[str, str],
    prefix: str = "l2",
) -> pd.DataFrame:
    """Build Level2 trade-size share factors from amount columns."""
    out = df.copy()
    total = pd.to_numeric(out[total_amount_col], errors="coerce").replace(0.0, np.nan)
    for label, col in amount_cols.items():
        out[f"{prefix}_{label}_share"] = pd.to_numeric(out[col], errors="coerce") / total
    return out


def level2_net_buy_ratio_factors(
    df: pd.DataFrame,
    buy_amount_cols: Mapping[str, str],
    sell_amount_cols: Mapping[str, str],
    prefix: str = "l2",
) -> pd.DataFrame:
    """Build active net-buy ratios by order-size bucket."""
    out = df.copy()
    for label, buy_col in buy_amount_cols.items():
        if label not in sell_amount_cols:
            continue
        buy = pd.to_numeric(out[buy_col], errors="coerce")
        sell = pd.to_numeric(out[sell_amount_cols[label]], errors="coerce")
        out[f"{prefix}_{label}_net_buy_ratio"] = (buy - sell) / (buy + sell).replace(0.0, np.nan)
    return out


def regression_type_reference() -> pd.DataFrame:
    rows = [
        (
            "time_series",
            "Factor returns are observed inputs; asset exposures are estimated.",
            "Return attribution and risk explanation.",
            "Intercept is asset-specific, often written alpha_i.",
        ),
        (
            "cross_section",
            "Asset exposures are known inputs; factor returns are estimated by date.",
            "Factor testing, alpha forecasting, and portfolio construction.",
            "Intercept is date-specific, often written R_z,t.",
        ),
    ]
    return pd.DataFrame(rows, columns=["regression_type", "input_output", "best_use", "formula_clue"])


def short_horizon_factor_checklist() -> pd.DataFrame:
    rows = [
        ("target", "Match factor-test target with portfolio constraints and execution horizon."),
        ("lag", "Use next-period returns and next tradable prices; do not fill unavailable intraday or Level2 data."),
        ("neutralization", "Neutralize industry, size, reversal, and turnover before claiming independent alpha."),
        ("turnover", "Report alpha net of explicit turnover-cost penalty, not only gross RankIC."),
        ("capacity", "Estimate capacity from participation rate and tail weighted trading amount."),
        ("window", "Dynamic reversal windows shorter than 20 trading days often become momentum, not reversal."),
        ("level2", "Treat trade-share factors and net-buy-ratio factors separately; their evidence is not interchangeable."),
        ("regression", "Do not compare Fama-French time-series R2 with Barra cross-sectional R2 as if they answer one question."),
        ("volume_price", "Volume-up winners are often short-leg signals in stock-flow markets; retest by market liquidity regime."),
        ("turnover_cv", "Turnover volatility is a second-moment liquidity signal and must be tested against average turnover."),
        ("lottery", "Extreme positive daily returns proxy lottery demand and must be separated from simple reversal."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    parser.add_argument("--regression-reference", action="store_true")
    parser.add_argument("--factor-zoo", action="store_true")
    args = parser.parse_args()
    if args.regression_reference:
        print(regression_type_reference())
    if args.factor_zoo:
        print(factor_zoo_definitions())
    if args.checklist:
        print(short_horizon_factor_checklist())


if __name__ == "__main__":
    main()
