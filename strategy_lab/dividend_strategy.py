#!/usr/bin/env python
"""Dividend and dividend-plus strategy helpers."""

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


def _cap_and_normalize(raw: pd.Series, cap: float = 0.10, max_iter: int = 100) -> pd.Series:
    if raw.empty:
        return raw.astype(float)
    if cap <= 0:
        raise ValueError("cap must be positive")
    weights = raw.astype(float).clip(lower=0.0)
    if weights.sum() <= EPS:
        weights = pd.Series(1.0, index=weights.index)
    weights = weights / weights.sum()
    cap_series = pd.Series(cap, index=weights.index)
    if cap_series.sum() < 1.0 - EPS:
        raise ValueError("cap is too small for the number of selected assets")
    fixed = pd.Series(0.0, index=weights.index)
    free = pd.Series(True, index=weights.index)
    for _ in range(max_iter):
        candidate = pd.Series(0.0, index=weights.index)
        base = weights[free]
        remaining = 1.0 - fixed.sum()
        if remaining <= EPS:
            break
        candidate.loc[free] = remaining * base / base.sum() if base.sum() > EPS else remaining / free.sum()
        breach = free & (candidate > cap + EPS)
        if not breach.any():
            fixed.loc[free] = candidate.loc[free]
            break
        fixed.loc[breach] = cap
        free.loc[breach] = False
    return fixed.rename("weight")


def dividend_yield_and_payout(
    df: pd.DataFrame,
    cash_dividend_col: str,
    market_cap_col: str,
    net_profit_col: str | None = None,
    output_prefix: str = "dividend",
) -> pd.DataFrame:
    """Add dividend yield and optional payout ratio."""
    required = [cash_dividend_col, market_cap_col]
    if net_profit_col:
        required.append(net_profit_col)
    _require_columns(df, required)
    out = df.copy()
    market_cap = out[market_cap_col].astype(float).replace(0, np.nan)
    out[f"{output_prefix}_yield"] = out[cash_dividend_col].astype(float) / market_cap
    if net_profit_col:
        profit = out[net_profit_col].astype(float).replace(0, np.nan)
        out[f"{output_prefix}_payout_ratio"] = out[cash_dividend_col].astype(float) / profit
    return out


def continuous_dividend_pool(
    annuals: pd.DataFrame,
    asset_col: str,
    fiscal_year_col: str,
    cash_dividend_col: str,
    net_profit_col: str | None = None,
    years: int = 3,
    payout_min: float = 0.0,
    payout_max: float = 1.0,
    output_col: str = "continuous_dividend_pool",
) -> pd.DataFrame:
    """Flag stocks with consecutive cash dividends and valid payout ratios."""
    if years <= 0:
        raise ValueError("years must be positive")
    required = [asset_col, fiscal_year_col, cash_dividend_col]
    if net_profit_col:
        required.append(net_profit_col)
    _require_columns(annuals, required)

    rows = []
    for asset, group in annuals.sort_values(fiscal_year_col).groupby(asset_col):
        group = group.copy()
        years_index = pd.RangeIndex(int(group[fiscal_year_col].min()), int(group[fiscal_year_col].max()) + 1)
        indexed = group.set_index(fiscal_year_col).reindex(years_index)
        indexed[asset_col] = asset
        indexed[cash_dividend_col] = indexed[cash_dividend_col].fillna(0.0)
        indexed["_positive_cash_dividend"] = indexed[cash_dividend_col].astype(float) > 0
        indexed["_rolling_positive_years"] = (
            indexed["_positive_cash_dividend"].astype(int).rolling(years, min_periods=years).sum()
        )
        if net_profit_col:
            profit = indexed[net_profit_col].astype(float).replace(0, np.nan)
            indexed["_payout_ratio"] = indexed[cash_dividend_col].astype(float) / profit
            indexed["_valid_payout"] = indexed["_payout_ratio"].between(payout_min, payout_max, inclusive="neither")
            indexed["_rolling_valid_payout_years"] = (
                indexed["_valid_payout"].fillna(False).astype(int).rolling(years, min_periods=years).sum()
            )
        else:
            indexed["_payout_ratio"] = np.nan
            indexed["_valid_payout"] = True
            indexed["_rolling_valid_payout_years"] = years
        indexed[output_col] = (indexed["_rolling_positive_years"] >= years) & (
            indexed["_rolling_valid_payout_years"] >= years
        )
        indexed[fiscal_year_col] = indexed.index
        rows.append(indexed.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True) if rows else annuals.iloc[0:0].copy()
    return out


def _prepare_scored_pool(
    df: pd.DataFrame,
    date_col: str,
    eligibility_col: str | None,
    factor_cols: list[str],
    directions: list[float],
    output_col: str,
) -> pd.DataFrame:
    _require_columns(df, [date_col, *factor_cols] + ([eligibility_col] if eligibility_col else []))
    out = df.copy()
    if eligibility_col:
        out = out[out[eligibility_col].astype(bool)].copy()
    parts = []
    for col, direction in zip(factor_cols, directions):
        parts.append(float(direction) * out.groupby(date_col)[col].transform(_zscore))
    out[output_col] = pd.concat(parts, axis=1).mean(axis=1)
    return out


def _selected_weighted_portfolio(
    selected: pd.DataFrame,
    date_col: str,
    asset_col: str,
    weight_base_col: str,
    weight_cap: float,
) -> pd.DataFrame:
    rows = []
    for date, group in selected.groupby(date_col):
        raw = group.set_index(asset_col)[weight_base_col].astype(float).clip(lower=0.0)
        weights = _cap_and_normalize(raw, cap=weight_cap)
        part = group.copy()
        part["weight"] = part[asset_col].map(weights)
        part["portfolio_date"] = date
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else selected.iloc[0:0].copy()


def dividend_plus_growth_portfolio(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    dividend_yield_col: str,
    sue_col: str,
    eligibility_col: str | None = None,
    base_quantile: float = 0.50,
    dividend_top_n: int = 100,
    final_top_n: int = 50,
    weight_cap: float = 0.10,
) -> pd.DataFrame:
    """Build a dividend-plus-growth selection similar to the research recipe."""
    if not 0 < base_quantile < 1:
        raise ValueError("base_quantile must be in (0, 1)")
    scored = _prepare_scored_pool(
        df, date_col, eligibility_col, [dividend_yield_col, sue_col], [1.0, 1.0], "_yield_sue_score"
    )
    rows = []
    for _, group in scored.groupby(date_col):
        base = group[group["_yield_sue_score"] >= group["_yield_sue_score"].quantile(base_quantile)]
        high_yield = base.sort_values(dividend_yield_col, ascending=False).head(dividend_top_n)
        final = high_yield.sort_values(sue_col, ascending=False).head(final_top_n)
        rows.append(final)
    selected = pd.concat(rows, ignore_index=True) if rows else scored.iloc[0:0].copy()
    selected["strategy"] = "dividend_plus_growth"
    return _selected_weighted_portfolio(selected, date_col, asset_col, dividend_yield_col, weight_cap)


def dividend_plus_low_vol_portfolio(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    dividend_yield_col: str,
    fundamental_cols: list[str],
    low_vol_col: str,
    low_turnover_col: str,
    eligibility_col: str | None = None,
    dividend_top_n: int = 75,
    final_top_n: int = 50,
    weight_cap: float = 0.10,
) -> pd.DataFrame:
    """Build a dividend-plus-low-vol selection."""
    scored = _prepare_scored_pool(df, date_col, eligibility_col, fundamental_cols, [1.0] * len(fundamental_cols), "_fund_score")
    rows = []
    for _, group in scored.groupby(date_col):
        base = group[group["_fund_score"] >= group["_fund_score"].median()]
        high_yield = base.sort_values(dividend_yield_col, ascending=False).head(dividend_top_n).copy()
        high_yield["_low_vol_turnover_score"] = (
            high_yield[low_vol_col].astype(float).rank(pct=True, ascending=False)
            + high_yield[low_turnover_col].astype(float).rank(pct=True, ascending=False)
        ) / 2.0
        final = high_yield.sort_values("_low_vol_turnover_score", ascending=False).head(final_top_n)
        rows.append(final)
    selected = pd.concat(rows, ignore_index=True) if rows else scored.iloc[0:0].copy()
    selected["strategy"] = "dividend_plus_low_vol"
    return _selected_weighted_portfolio(selected, date_col, asset_col, dividend_yield_col, weight_cap)


def dividend_potential_portfolio(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    dividend_yield_col: str,
    factor_cols: list[str],
    directions: list[float] | None = None,
    eligibility_col: str = "continuous_dividend_pool",
    final_top_n: int = 50,
    weight_cap: float = 0.10,
) -> pd.DataFrame:
    """Select dividend-potential names from a continuous-dividend pool."""
    if directions is None:
        directions = [1.0] * len(factor_cols)
    scored = _prepare_scored_pool(df, date_col, eligibility_col, factor_cols, directions, "_dividend_potential_score")
    rows = [g.sort_values("_dividend_potential_score", ascending=False).head(final_top_n) for _, g in scored.groupby(date_col)]
    selected = pd.concat(rows, ignore_index=True) if rows else scored.iloc[0:0].copy()
    selected["strategy"] = "dividend_potential"
    return _selected_weighted_portfolio(selected, date_col, asset_col, dividend_yield_col, weight_cap)


def dividend_macro_regime(
    macro: pd.DataFrame,
    date_col: str,
    us_rate_col: str | None = None,
    social_financing_yoy_col: str | None = None,
    market_vol_col: str | None = None,
    rate_lag: int = 12,
    social_lag: int = 6,
    vol_window: int = 6,
) -> pd.DataFrame:
    """Create a dividend style regime score from rates, credit, and volatility.

    Positive score means dividend/value is favored over growth. Social financing
    contraction is treated as a relative-style signal, not an absolute-return
    signal, because it may coincide with weak equity beta.
    """
    required = [date_col]
    for col in [us_rate_col, social_financing_yoy_col, market_vol_col]:
        if col:
            required.append(col)
    _require_columns(macro, required)
    out = macro.sort_values(date_col).copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["dividend_regime_score"] = 0.0
    if us_rate_col:
        out["us_rate_up_yoy"] = out[us_rate_col].astype(float) > out[us_rate_col].astype(float).shift(rate_lag)
        out["dividend_regime_score"] += out["us_rate_up_yoy"].fillna(False).astype(float)
    if social_financing_yoy_col:
        out["social_financing_down"] = out[social_financing_yoy_col].astype(float) < out[
            social_financing_yoy_col
        ].astype(float).shift(social_lag)
        out["dividend_regime_score"] += 0.5 * out["social_financing_down"].fillna(False).astype(float)
    if market_vol_col:
        vol_mean = out[market_vol_col].astype(float).rolling(vol_window, min_periods=max(2, vol_window // 2)).mean()
        out["market_vol_expanding"] = out[market_vol_col].astype(float) > vol_mean
        out["dividend_regime_score"] += out["market_vol_expanding"].fillna(False).astype(float)

    conditions = [
        out["dividend_regime_score"] >= 2.0,
        out["dividend_regime_score"].between(1.0, 2.0, inclusive="left"),
        out["dividend_regime_score"] <= 0.0,
    ]
    labels = ["strong_dividend", "moderate_dividend", "growth_favored"]
    out["dividend_regime"] = np.select(conditions, labels, default="neutral")
    if social_financing_yoy_col:
        out["absolute_return_caution"] = out.get("social_financing_down", False).fillna(False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--dividend-yield-col", required=True)
    parser.add_argument("--sue-col", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = dividend_plus_growth_portfolio(
        df,
        date_col=args.date_col,
        asset_col=args.asset_col,
        dividend_yield_col=args.dividend_yield_col,
        sue_col=args.sue_col,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
