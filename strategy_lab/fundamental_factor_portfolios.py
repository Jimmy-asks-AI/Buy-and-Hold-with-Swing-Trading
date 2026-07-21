#!/usr/bin/env python
"""Fundamental factor portfolio helpers.

The module focuses on reusable construction logic from fundamental factor
research: fund-overweight signals, deep value, value-with-support portfolios,
and earnings acceleration.  It is intentionally a research toolkit rather than
a full backtester.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _zscore(s: pd.Series) -> pd.Series:
    values = s.astype(float)
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (values - values.mean()) / std


def _rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    return s.astype(float).rank(pct=True, method="first", ascending=ascending)


def composite_score_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    directions: list[float] | None = None,
    weights: list[float] | None = None,
    output_col: str = "composite_score",
    winsor_limits: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """Create a date-wise composite z-score.

    `directions` is 1 for higher-is-better and -1 for lower-is-better.
    `weights` are normalized to sum to one by absolute value.
    """
    _require_columns(df, [date_col, *factor_cols])
    if directions is None:
        directions = [1.0] * len(factor_cols)
    if weights is None:
        weights = [1.0] * len(factor_cols)
    if len(directions) != len(factor_cols) or len(weights) != len(factor_cols):
        raise ValueError("factor_cols, directions, and weights must have the same length")

    weight_array = np.asarray(weights, dtype=float)
    weight_sum = np.abs(weight_array).sum()
    if weight_sum <= EPS:
        raise ValueError("weights cannot all be zero")
    weight_array = weight_array / weight_sum

    out = df.copy()
    parts = []
    for col, direction, weight in zip(factor_cols, directions, weight_array):
        raw = out[col].astype(float)
        if winsor_limits is not None:
            lo_q, hi_q = winsor_limits
            lo = raw.groupby(out[date_col]).transform(lambda s: s.quantile(lo_q))
            hi = raw.groupby(out[date_col]).transform(lambda s: s.quantile(hi_q))
            raw = raw.clip(lower=lo, upper=hi)
        z = raw.groupby(out[date_col], group_keys=False).transform(_zscore)
        parts.append(float(direction) * float(weight) * z)
    out[output_col] = pd.concat(parts, axis=1).sum(axis=1)
    return out


def regression_residualize_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    control_cols: list[str],
    output_col: str | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Residualize a cross-sectional value by controls within each date."""
    _require_columns(df, [date_col, value_col, *control_cols])
    min_count = min_count or len(control_cols) + 5
    target = output_col or f"{value_col}_residual"
    out = df.copy()
    pieces = []
    for _, group in out.groupby(date_col, sort=True):
        work = group[[value_col, *control_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        series = pd.Series(np.nan, index=group.index, name=target)
        if work.shape[0] >= min_count:
            x = np.column_stack([np.ones(work.shape[0]), work[control_cols].to_numpy(dtype=float)])
            y = work[value_col].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            series.loc[work.index] = y - x @ beta
        pieces.append(series)
    out[target] = pd.concat(pieces).sort_index()
    return out


def fundamental_fscore_factor(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    control_cols: list[str] | None = None,
    directions: list[float] | None = None,
    method: str = "zscore",
    output_col: str = "factor_f",
) -> pd.DataFrame:
    """Build a Piotroski-style composite fundamental quality factor."""
    if method not in {"zscore", "rank"}:
        raise ValueError("method must be 'zscore' or 'rank'.")
    _require_columns(df, [date_col, *factor_cols, *(control_cols or [])])
    directions = directions or [1.0] * len(factor_cols)
    if len(directions) != len(factor_cols):
        raise ValueError("directions and factor_cols must have the same length.")
    out = df.copy()
    component_cols = []
    for col, direction in zip(factor_cols, directions):
        component = f"_{col}_component"
        values = pd.to_numeric(out[col], errors="coerce") * float(direction)
        if method == "zscore":
            out[component] = values.groupby(out[date_col]).transform(_zscore)
        else:
            out[component] = values.groupby(out[date_col]).transform(lambda s: s.rank(pct=True, method="average"))
        if control_cols:
            residual_col = f"{component}_residual"
            out = regression_residualize_by_date(out, date_col, component, control_cols, output_col=residual_col)
            component_cols.append(residual_col)
        else:
            component_cols.append(component)
    out[output_col] = out[component_cols].mean(axis=1)
    return out.drop(columns=[col for col in out.columns if col.startswith("_") and col.endswith("_component")])


def factor_persistence_by_asset(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    factor_col: str,
    periods: int = 1,
) -> pd.DataFrame:
    """Estimate persistence of a factor versus its future value."""
    _require_columns(df, [asset_col, date_col, factor_col])
    out = df.sort_values([asset_col, date_col]).copy()
    out["_future_factor"] = out.groupby(asset_col)[factor_col].shift(-periods)
    rows = []
    for date, group in out.dropna(subset=[factor_col, "_future_factor"]).groupby(date_col):
        if group.shape[0] < 20:
            continue
        rows.append(
            {
                "date": date,
                "n_assets": int(group.shape[0]),
                "corr": float(group[factor_col].corr(group["_future_factor"])),
                "rank_corr": float(group[factor_col].rank().corr(group["_future_factor"].rank())),
            }
        )
    return pd.DataFrame(rows)


def fundamental_quality_factor_map() -> pd.DataFrame:
    rows = [
        ("ROE", "return on equity", 1),
        ("dROE", "year-over-year change in ROE", 1),
        ("dCFO", "year-over-year change in operating cash flow per share", 1),
        ("EPS", "earnings per share", 1),
        ("dLEVER", "year-over-year change in equity multiplier", -1),
        ("LIQUID", "current ratio", 1),
        ("dMARGIN", "year-over-year change in single-quarter net margin", 1),
        ("dTURN", "year-over-year change in fixed asset turnover", 1),
    ]
    return pd.DataFrame(rows, columns=["factor", "definition", "default_direction"])


def rolling_profitability_trend(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    profitability_col: str,
    periods: int = 4,
    seasonal_lag: int = 4,
    min_periods: int | None = None,
    output_col: str = "profitability_trend",
) -> pd.DataFrame:
    """Estimate profitability trend as an OLS slope on recent observations.

    `seasonal_lag=4` compares the same fiscal quarter across years and is the
    safer default for quarterly profitability because it reduces seasonality.
    Use `seasonal_lag=1` for adjacent-quarter trends.
    """
    if periods <= 1:
        raise ValueError("periods must be greater than 1.")
    if seasonal_lag <= 0:
        raise ValueError("seasonal_lag must be positive.")
    min_periods = min_periods or max(3, periods // 2 + 1)
    _require_columns(df, [asset_col, date_col, profitability_col])
    out = df.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        values = pd.to_numeric(group[profitability_col], errors="coerce").reset_index(drop=True)
        scores = pd.Series(np.nan, index=group.index, dtype="float64")
        for pos in range(len(group)):
            sample_positions = [pos - seasonal_lag * step for step in range(periods - 1, -1, -1)]
            sample_positions = [idx for idx in sample_positions if idx >= 0]
            sample = values.iloc[sample_positions].dropna()
            if sample.shape[0] < min_periods:
                continue
            x = np.arange(1, sample.shape[0] + 1, dtype=float)
            x = np.column_stack([np.ones(sample.shape[0]), x])
            beta = np.linalg.lstsq(x, sample.to_numpy(dtype=float), rcond=None)[0]
            scores.iloc[pos] = float(beta[1])
        out.loc[group.index, output_col] = scores.to_numpy(dtype=float)
    return out


def select_top_n_by_date(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    n: int,
    output_col: str = "selected",
    ascending: bool = False,
) -> pd.DataFrame:
    """Flag top-N securities by date."""
    _require_columns(df, [date_col, score_col])
    if n <= 0:
        raise ValueError("n must be positive")
    out = df.copy()
    out[output_col] = False
    for _, group in out.groupby(date_col):
        selected_index = group[score_col].astype(float).sort_values(ascending=ascending).head(n).index
        out.loc[selected_index, output_col] = True
    return out


def select_quantile_by_date(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    quantile: float,
    output_col: str = "selected",
    high_is_good: bool = True,
) -> pd.DataFrame:
    """Flag top or bottom cross-sectional quantile by date."""
    _require_columns(df, [date_col, score_col])
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    out = df.copy()
    pct = out.groupby(date_col)[score_col].transform(_rank_pct)
    out[output_col] = pct >= 1.0 - quantile if high_is_good else pct <= quantile
    return out


def deep_value_ncav(
    df: pd.DataFrame,
    cash_col: str,
    receivables_col: str,
    inventory_col: str,
    total_liabilities_col: str,
    price_col: str,
    float_shares_col: str,
    total_shares_col: str,
    current_assets_col: str | None = None,
    preferred_stock_col: str | None = None,
    output_prefix: str = "ncav",
) -> pd.DataFrame:
    """Compute NCAV-style deep value measures.

    NCAV1 = (cash + 0.75 * receivables + 0.5 * inventory - liabilities - preferred stock)
            / float shares
    NCAV2 = (0.75 * current assets - liabilities - preferred stock) / total shares

    A stock is flagged when price is below NCAV, which is the usual deep-value
    definition.  If current assets are not available, cash + receivables +
    inventory is used as a conservative proxy.
    """
    required = [
        cash_col,
        receivables_col,
        inventory_col,
        total_liabilities_col,
        price_col,
        float_shares_col,
        total_shares_col,
    ]
    if current_assets_col:
        required.append(current_assets_col)
    if preferred_stock_col:
        required.append(preferred_stock_col)
    _require_columns(df, required)

    out = df.copy()
    preferred = out[preferred_stock_col].astype(float) if preferred_stock_col else 0.0
    current_assets = (
        out[current_assets_col].astype(float)
        if current_assets_col
        else out[cash_col].astype(float) + out[receivables_col].astype(float) + out[inventory_col].astype(float)
    )
    liabilities = out[total_liabilities_col].astype(float)
    price = out[price_col].astype(float)
    float_shares = out[float_shares_col].astype(float).replace(0, np.nan)
    total_shares = out[total_shares_col].astype(float).replace(0, np.nan)

    out[f"{output_prefix}1"] = (
        out[cash_col].astype(float)
        + 0.75 * out[receivables_col].astype(float)
        + 0.50 * out[inventory_col].astype(float)
        - liabilities
        - preferred
    ) / float_shares
    out[f"{output_prefix}2"] = (0.75 * current_assets - liabilities - preferred) / total_shares
    out[f"{output_prefix}1_deep_value"] = price < out[f"{output_prefix}1"]
    out[f"{output_prefix}2_deep_value"] = price < out[f"{output_prefix}2"]
    return out


def _filter_fund_pool(
    holdings: pd.DataFrame,
    report_date_col: str,
    fund_id_col: str | None,
    fund_performance_col: str | None,
    performance_group: str,
    performance_quantile: float,
) -> pd.DataFrame:
    if performance_group == "all":
        return holdings.copy()
    if fund_id_col is None or fund_performance_col is None:
        raise ValueError("fund_id_col and fund_performance_col are required for performance filtering")
    if not 0 < performance_quantile < 1:
        raise ValueError("performance_quantile must be in (0, 1)")
    if performance_group not in {"top", "bottom"}:
        raise ValueError("performance_group must be one of: all, top, bottom")

    rows = []
    for _, group in holdings.groupby(report_date_col):
        fund_perf = group[[fund_id_col, fund_performance_col]].drop_duplicates(fund_id_col)
        fund_perf = fund_perf.sort_values(fund_performance_col, ascending=False)
        keep_count = max(1, int(np.ceil(fund_perf.shape[0] * performance_quantile)))
        if performance_group == "top":
            keep_funds = fund_perf.head(keep_count)[fund_id_col]
        else:
            keep_funds = fund_perf.tail(keep_count)[fund_id_col]
        rows.append(group[group[fund_id_col].isin(keep_funds)])
    if not rows:
        return holdings.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def fund_overweight_factor(
    holdings: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    report_date_col: str,
    asset_col: str,
    benchmark_weight_col: str,
    holding_weight_col: str | None = None,
    holding_value_col: str | None = None,
    fund_id_col: str | None = None,
    fund_performance_col: str | None = None,
    performance_group: str = "all",
    performance_quantile: float = 0.5,
    disclosure_lag_days: int = 0,
) -> pd.DataFrame:
    """Build a fund-overweight dummy and overweight gap.

    The fund pool weight is compared with benchmark weight.  Use
    `disclosure_lag_days` to create an availability date and avoid treating
    report-date holdings as instantly known.
    """
    if bool(holding_weight_col) == bool(holding_value_col):
        raise ValueError("Provide exactly one of holding_weight_col or holding_value_col")
    required_holdings = [report_date_col, asset_col, holding_weight_col or holding_value_col]
    if fund_id_col:
        required_holdings.append(fund_id_col)
    if fund_performance_col:
        required_holdings.append(fund_performance_col)
    _require_columns(holdings, required_holdings)
    _require_columns(benchmark_weights, [report_date_col, asset_col, benchmark_weight_col])

    filtered = _filter_fund_pool(
        holdings,
        report_date_col=report_date_col,
        fund_id_col=fund_id_col,
        fund_performance_col=fund_performance_col,
        performance_group=performance_group,
        performance_quantile=performance_quantile,
    )
    value_col = holding_value_col or holding_weight_col
    pool = (
        filtered.groupby([report_date_col, asset_col], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "fund_pool_value"})
    )
    total = pool.groupby(report_date_col)["fund_pool_value"].transform("sum").replace(0, np.nan)
    pool["fund_pool_weight"] = pool["fund_pool_value"] / total

    bench = benchmark_weights[[report_date_col, asset_col, benchmark_weight_col]].copy()
    merged = bench.merge(pool[[report_date_col, asset_col, "fund_pool_weight"]], how="outer")
    merged[benchmark_weight_col] = merged[benchmark_weight_col].fillna(0.0)
    merged["fund_pool_weight"] = merged["fund_pool_weight"].fillna(0.0)
    merged["overweight_gap"] = merged["fund_pool_weight"] - merged[benchmark_weight_col]
    merged["fund_overweight_flag"] = (merged["overweight_gap"] > 0).astype(int)
    merged["performance_group"] = performance_group
    merged["available_date"] = pd.to_datetime(merged[report_date_col]) + pd.to_timedelta(disclosure_lag_days, unit="D")
    return merged


def factor_premium_timing_gate(
    premium_df: pd.DataFrame,
    date_col: str,
    premium_col: str,
    window: int = 12,
    min_periods: int = 6,
    allow_negative: bool = False,
    output_col: str = "factor_weight_multiplier",
) -> pd.DataFrame:
    """Gate an unstable factor by trailing premium direction.

    The rolling mean is shifted by one period, so the current premium is never
    used to time itself.  If `allow_negative` is False, negative expected
    premium maps to zero weight.  If True, the output is -1, 0, or 1.
    """
    _require_columns(premium_df, [date_col, premium_col])
    out = premium_df.sort_values(date_col).copy()
    expected = out[premium_col].astype(float).rolling(window=window, min_periods=min_periods).mean().shift(1)
    out["expected_premium"] = expected
    if allow_negative:
        out[output_col] = np.sign(expected).fillna(0.0)
    else:
        out[output_col] = (expected > 0).astype(float).where(expected.notna(), 0.0)
    return out


def premium_direction_persistence(
    premium_df: pd.DataFrame,
    date_col: str,
    premium_col: str,
    window: int = 12,
    min_periods: int = 6,
    significant_col: str | None = None,
) -> pd.DataFrame:
    """Measure how often trailing premium direction matches realized direction."""
    _require_columns(premium_df, [date_col, premium_col])
    if significant_col:
        _require_columns(premium_df, [significant_col])
    gated = factor_premium_timing_gate(
        premium_df,
        date_col=date_col,
        premium_col=premium_col,
        window=window,
        min_periods=min_periods,
        allow_negative=True,
        output_col="predicted_direction",
    )
    sample = gated[gated["predicted_direction"] != 0].copy()
    if significant_col:
        sample = sample[sample[significant_col].astype(bool)]
    if sample.empty:
        ratio = np.nan
    else:
        ratio = float((np.sign(sample[premium_col].astype(float)) == sample["predicted_direction"]).mean())
    return pd.DataFrame(
        [
            {
                "window": window,
                "min_periods": min_periods,
                "sample_count": int(sample.shape[0]),
                "direction_persistence": ratio,
            }
        ]
    )


def earnings_growth_and_acceleration(
    df: pd.DataFrame,
    asset_col: str,
    quarter_col: str,
    eps_col: str,
    method: str = "EAV",
    price_col: str | None = None,
    eps_std_window: int = 8,
    min_abs_denom: float = 1e-6,
    output_prefix: str | None = None,
) -> pd.DataFrame:
    """Compute standardized EPS growth and earnings acceleration.

    Methods:
    - EAA: denominator is absolute EPS from the same quarter last year.
    - EAP: denominator is the previous quarter-end price.
    - EAV: denominator is rolling EPS standard deviation over recent quarters.
    """
    method = method.upper()
    if method not in {"EAA", "EAP", "EAV"}:
        raise ValueError("method must be one of: EAA, EAP, EAV")
    required = [asset_col, quarter_col, eps_col]
    if method == "EAP":
        if price_col is None:
            raise ValueError("price_col is required for EAP")
        required.append(price_col)
    _require_columns(df, required)

    prefix = output_prefix or method.lower()
    out = df.sort_values([asset_col, quarter_col]).copy()
    grouped_eps = out.groupby(asset_col, group_keys=False)[eps_col]
    eps = out[eps_col].astype(float)
    eps_lag4 = grouped_eps.shift(4)
    yoy_change = eps - eps_lag4

    if method == "EAA":
        denominator = eps_lag4.abs()
    elif method == "EAP":
        denominator = out.groupby(asset_col, group_keys=False)[price_col].shift(1).abs()
    else:
        denominator = (
            grouped_eps.rolling(window=eps_std_window, min_periods=eps_std_window)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
            .abs()
        )
    denominator = denominator.mask(denominator < min_abs_denom)
    growth_col = f"{prefix}_growth"
    accel_col = f"{prefix}_acceleration"
    out[f"{prefix}_eps_yoy_change"] = yoy_change
    out[growth_col] = yoy_change / denominator
    out[accel_col] = out[growth_col] - out.groupby(asset_col, group_keys=False)[growth_col].shift(1)
    return out


def add_earnings_acceleration_modes(
    df: pd.DataFrame,
    asset_col: str,
    quarter_col: str,
    growth_col: str,
    acceleration_col: str,
    output_col: str = "earnings_acceleration_mode",
) -> pd.DataFrame:
    """Classify the six earnings acceleration modes from the report."""
    _require_columns(df, [asset_col, quarter_col, growth_col, acceleration_col])
    out = df.sort_values([asset_col, quarter_col]).copy()
    current = out[growth_col].astype(float)
    prior = out.groupby(asset_col, group_keys=False)[growth_col].shift(1).astype(float)
    accel = out[acceleration_col].astype(float)
    mode = pd.Series(pd.NA, index=out.index, dtype="Int64")

    mode[(accel > 0) & (current > 0) & (prior > 0)] = 1
    mode[(accel > 0) & (current > 0) & (prior < 0)] = 2
    mode[(accel > 0) & (current < 0) & (prior < 0)] = 3
    mode[(accel < 0) & (current > 0) & (prior > 0)] = 4
    mode[(accel < 0) & (current < 0) & (prior > 0)] = 5
    mode[(accel < 0) & (current < 0) & (prior < 0)] = 6
    out[output_col] = mode
    return out


def relative_valuation_position(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    valuation_col: str,
    market_valuation_col: str | None = None,
    window: int = 36,
    min_periods: int = 12,
    output_col: str = "relative_valuation_position",
) -> pd.DataFrame:
    """Rolling percentile rank of relative valuation for each asset.

    If market_valuation_col is provided, the rank is based on
    stock valuation / market valuation.  Lower percentiles mean unusually cheap
    versus the asset's own history.
    """
    required = [asset_col, date_col, valuation_col]
    if market_valuation_col:
        required.append(market_valuation_col)
    _require_columns(df, required)
    out = df.sort_values([asset_col, date_col]).copy()
    if market_valuation_col:
        base = out[valuation_col].astype(float) / out[market_valuation_col].astype(float).replace(0, np.nan)
    else:
        base = out[valuation_col].astype(float)
    out["_relative_valuation_base"] = base

    def last_rank_pct(values: np.ndarray) -> float:
        s = pd.Series(values)
        return float(s.rank(pct=True, method="average").iloc[-1])

    out[output_col] = (
        out.groupby(asset_col, group_keys=False)["_relative_valuation_base"]
        .rolling(window=window, min_periods=min_periods)
        .apply(last_rank_pct, raw=True)
        .reset_index(level=0, drop=True)
    )
    return out.drop(columns=["_relative_valuation_base"])


def apply_value_trap_filters(
    df: pd.DataFrame,
    long_position_col: str,
    short_position_col: str,
    min_long_position: float = 0.10,
    max_short_position: float = 0.90,
    output_col: str = "passes_value_position_filter",
) -> pd.DataFrame:
    """Apply relative valuation filters used in value-with-support portfolios."""
    _require_columns(df, [long_position_col, short_position_col])
    out = df.copy()
    out[output_col] = (
        (out[long_position_col].astype(float) >= min_long_position)
        & (out[short_position_col].astype(float) <= max_short_position)
    )
    return out


def value_growth_intersection_candidates(
    df: pd.DataFrame,
    date_col: str,
    value_score_col: str,
    support_score_col: str,
    value_quantile: float = 0.20,
    support_quantile: float = 0.20,
    turnover_col: str | None = None,
    low_turnover_quantile: float | None = None,
    position_filter_col: str | None = None,
    output_col: str = "value_with_support_candidate",
) -> pd.DataFrame:
    """Select the intersection of cheap stocks and fundamental support."""
    required = [date_col, value_score_col, support_score_col]
    if turnover_col:
        required.append(turnover_col)
    if position_filter_col:
        required.append(position_filter_col)
    _require_columns(df, required)

    out = df.copy()
    value_pct = out.groupby(date_col)[value_score_col].transform(lambda s: _rank_pct(s, ascending=True))
    support_pct = out.groupby(date_col)[support_score_col].transform(lambda s: _rank_pct(s, ascending=True))
    flag = (value_pct >= 1.0 - value_quantile) & (support_pct >= 1.0 - support_quantile)
    if turnover_col and low_turnover_quantile is not None:
        turnover_pct = out.groupby(date_col)[turnover_col].transform(lambda s: _rank_pct(s, ascending=True))
        flag = flag & (turnover_pct <= low_turnover_quantile)
    if position_filter_col:
        flag = flag & out[position_filter_col].astype(bool)
    out[output_col] = flag
    return out


def value_composite_score(
    df: pd.DataFrame,
    date_col: str,
    pb_col: str,
    pe_col: str | None = None,
    dividend_yield_col: str | None = None,
    pcf_col: str | None = None,
    output_col: str = "value_score",
) -> pd.DataFrame:
    """Composite value score where higher means cheaper or better value."""
    factors = [pb_col]
    directions = [-1.0]
    if pe_col:
        factors.append(pe_col)
        directions.append(-1.0)
    if dividend_yield_col:
        factors.append(dividend_yield_col)
        directions.append(1.0)
    if pcf_col:
        factors.append(pcf_col)
        directions.append(-1.0)
    return composite_score_by_date(df, date_col, factors, directions=directions, output_col=output_col)


def growth_composite_score(
    df: pd.DataFrame,
    date_col: str,
    earnings_acceleration_col: str,
    sue_col: str,
    net_profit_revision_col: str,
    rd_capital_ratio_col: str | None = None,
    pb_int_col: str | None = None,
    output_col: str = "growth_score",
) -> pd.DataFrame:
    """Composite growth score with optional R&D and adjusted PB balance."""
    factors = [earnings_acceleration_col, sue_col, net_profit_revision_col]
    directions = [1.0, 1.0, 1.0]
    if rd_capital_ratio_col:
        factors.append(rd_capital_ratio_col)
        directions.append(1.0)
    if pb_int_col:
        factors.append(pb_int_col)
        directions.append(-1.0)
    return composite_score_by_date(df, date_col, factors, directions=directions, output_col=output_col)


def capture_ratios(
    returns: pd.DataFrame,
    date_col: str,
    portfolio_return_col: str,
    benchmark_return_col: str,
) -> pd.DataFrame:
    """Compute up-capture and down-capture against a benchmark."""
    _require_columns(returns, [date_col, portfolio_return_col, benchmark_return_col])
    sample = returns.dropna(subset=[portfolio_return_col, benchmark_return_col]).copy()
    up = sample[sample[benchmark_return_col] > 0]
    down = sample[sample[benchmark_return_col] < 0]

    def ratio(group: pd.DataFrame) -> float:
        denom = group[benchmark_return_col].mean()
        if pd.isna(denom) or abs(denom) <= EPS:
            return np.nan
        return float(group[portfolio_return_col].mean() / denom)

    return pd.DataFrame(
        [
            {
                "up_capture": ratio(up),
                "down_capture": ratio(down),
                "up_months": int(up.shape[0]),
                "down_months": int(down.shape[0]),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("composite-score")
    score.add_argument("--csv", required=True)
    score.add_argument("--date-col", required=True)
    score.add_argument("--factors", nargs="+", required=True)
    score.add_argument("--directions", nargs="+", type=float)
    score.add_argument("--output", required=True)
    score.add_argument("--output-col", default="composite_score")

    ea = sub.add_parser("earnings-acceleration")
    ea.add_argument("--csv", required=True)
    ea.add_argument("--asset-col", required=True)
    ea.add_argument("--quarter-col", required=True)
    ea.add_argument("--eps-col", required=True)
    ea.add_argument("--method", choices=["EAA", "EAP", "EAV"], default="EAV")
    ea.add_argument("--price-col")
    ea.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "composite-score":
        df = pd.read_csv(args.csv)
        out = composite_score_by_date(
            df,
            date_col=args.date_col,
            factor_cols=args.factors,
            directions=args.directions,
            output_col=args.output_col,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
    elif args.command == "earnings-acceleration":
        df = pd.read_csv(args.csv)
        out = earnings_growth_and_acceleration(
            df,
            asset_col=args.asset_col,
            quarter_col=args.quarter_col,
            eps_col=args.eps_col,
            method=args.method,
            price_col=args.price_col,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
