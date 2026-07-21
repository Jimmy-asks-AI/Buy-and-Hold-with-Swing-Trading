#!/usr/bin/env python
"""Behavioral, quasi-alternative-data, and technical timing research helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _zscore(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (values - values.mean()) / std


def margin_financing_growth_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    financing_balance_col: str,
    turnover_amount_col: str,
    lookback: int = 60,
    output_col: str = "margin_financing_growth",
) -> pd.DataFrame:
    """Financing growth over recent trading activity."""
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    _require_columns(df, [asset_col, date_col, financing_balance_col, turnover_amount_col])
    out = df.sort_values([asset_col, date_col]).copy()
    balance = pd.to_numeric(out[financing_balance_col], errors="coerce")
    amount = pd.to_numeric(out[turnover_amount_col], errors="coerce")
    financing_delta = balance.groupby(out[asset_col]).diff()
    avg_delta = financing_delta.groupby(out[asset_col]).transform(lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).mean())
    avg_amount = amount.groupby(out[asset_col]).transform(lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).mean())
    out[output_col] = avg_delta / avg_amount.replace(0.0, np.nan)
    out[f"{output_col}_balance_ratio"] = balance / avg_amount.replace(0.0, np.nan)
    return out


def margin_short_warning_flag(
    df: pd.DataFrame,
    date_col: str,
    short_balance_col: str,
    q: float = 0.8,
    output_col: str = "high_short_balance_warning",
) -> pd.DataFrame:
    """Flag high short-balance names as risk warnings, not alpha signals."""
    _require_columns(df, [date_col, short_balance_col])
    out = df.copy()
    values = pd.to_numeric(out[short_balance_col], errors="coerce")
    threshold = values.groupby(out[date_col]).transform(lambda s: s.quantile(q))
    out[output_col] = values >= threshold
    return out


def parallel_single_factor_extreme_score(
    df: pd.DataFrame,
    date_col: str,
    factor_directions: Mapping[str, int],
    top_n: int | None = 100,
    top_quantile: float | None = None,
    output_col: str = "parallel_single_factor_score",
) -> pd.DataFrame:
    """Parallel single-factor strategy score from each factor's preferred tail."""
    factors = list(factor_directions)
    _require_columns(df, [date_col, *factors])
    out = df.copy()
    out[output_col] = 0.0
    for date, group in out.groupby(date_col, sort=True):
        votes = pd.Series(0.0, index=group.index)
        for factor, direction in factor_directions.items():
            values = pd.to_numeric(group[factor], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                continue
            ascending = direction < 0
            if top_quantile is not None:
                cutoff = max(1, int(np.ceil(valid.shape[0] * top_quantile)))
            elif top_n is not None:
                cutoff = min(top_n, valid.shape[0])
            else:
                raise ValueError("top_n or top_quantile must be provided.")
            selected = valid.sort_values(ascending=ascending).head(cutoff).index
            votes.loc[selected] += 1.0
        out.loc[group.index, output_col] = votes
    out[output_col] = out[output_col] / max(1, len(factors))
    return out


def industry_neutral_extreme_score(
    df: pd.DataFrame,
    date_col: str,
    industry_col: str,
    factor_col: str,
    direction: int,
    top_quantile: float = 0.1,
    output_col: str = "industry_neutral_extreme",
) -> pd.DataFrame:
    """Select a factor's preferred tail inside each industry."""
    _require_columns(df, [date_col, industry_col, factor_col])
    out = df.copy()
    out[output_col] = False
    for _, group in out.groupby([date_col, industry_col], sort=True):
        values = pd.to_numeric(group[factor_col], errors="coerce").dropna()
        if values.empty:
            continue
        n_select = max(1, int(np.ceil(values.shape[0] * top_quantile)))
        selected = values.sort_values(ascending=direction < 0).head(n_select).index
        out.loc[selected, output_col] = True
    return out


def compensation_growth_factors(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    payable_comp_col: str,
    paid_comp_col: str | None = None,
    employee_count_col: str | None = None,
    seasonal_lag: int = 4,
    output_prefix: str = "comp",
) -> pd.DataFrame:
    """Build payable/paid compensation growth and consecutive growth streaks."""
    required = [asset_col, date_col, payable_comp_col]
    if paid_comp_col:
        required.append(paid_comp_col)
    if employee_count_col:
        required.append(employee_count_col)
    _require_columns(df, required)
    out = df.sort_values([asset_col, date_col]).copy()

    def add_growth(col: str, prefix: str) -> None:
        values = pd.to_numeric(out[col], errors="coerce")
        lagged = values.groupby(out[asset_col]).shift(seasonal_lag)
        out[f"{prefix}_growth"] = values / lagged.replace(0.0, np.nan) - 1.0
        growth = out[f"{prefix}_growth"]
        streak = []
        for _, group in out.groupby(asset_col, sort=False):
            count = 0
            for value in growth.loc[group.index]:
                if pd.notna(value) and value > 0:
                    count += 1
                else:
                    count = 0
                streak.append(count)
        out[f"{prefix}_growth_streak"] = streak

    add_growth(payable_comp_col, f"{output_prefix}_payable")
    if paid_comp_col:
        add_growth(paid_comp_col, f"{output_prefix}_paid")
    if employee_count_col:
        employees = pd.to_numeric(out[employee_count_col], errors="coerce").replace(0.0, np.nan)
        out[f"{output_prefix}_payable_per_employee"] = pd.to_numeric(out[payable_comp_col], errors="coerce") / employees
        add_growth(f"{output_prefix}_payable_per_employee", f"{output_prefix}_payable_per_employee")
    return out


def compensation_top_bottom_signal(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    industry_col: str | None = None,
    q: float = 0.1,
    min_names: int = 5,
    output_col: str = "compensation_signal",
) -> pd.DataFrame:
    """Top/bottom compensation-growth signal, optionally within industry."""
    required = [date_col, factor_col]
    if industry_col:
        required.append(industry_col)
    _require_columns(df, required)
    out = df.copy()
    out[output_col] = 0
    group_cols = [date_col, industry_col] if industry_col else [date_col]
    for _, group in out.groupby(group_cols, sort=True):
        values = pd.to_numeric(group[factor_col], errors="coerce").dropna()
        if values.empty:
            continue
        n_select = max(min_names, int(np.ceil(values.shape[0] * q)))
        n_select = min(n_select, max(1, values.shape[0] // 2))
        long_idx = values.sort_values(ascending=False).head(n_select).index
        short_idx = values.sort_values(ascending=True).head(n_select).index
        out.loc[long_idx, output_col] = 1
        out.loc[short_idx, output_col] = -1
    return out


def related_company_momentum(
    returns: pd.DataFrame,
    relations: pd.DataFrame,
    asset_col: str,
    date_col: str,
    return_col: str,
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
    lookback: int = 12,
    output_col: str = "related_company_momentum",
) -> pd.DataFrame:
    """Momentum spillover through patent, geography, customer, or analyst links."""
    _require_columns(returns, [asset_col, date_col, return_col])
    relation_cols = [source_col, target_col]
    if weight_col:
        relation_cols.append(weight_col)
    _require_columns(relations, relation_cols)
    base = returns.sort_values([asset_col, date_col]).copy()
    base["_past_return"] = (
        pd.to_numeric(base[return_col], errors="coerce")
        .groupby(base[asset_col])
        .transform(lambda s: s.rolling(lookback, min_periods=max(3, lookback // 2)).sum())
    )
    relation_data = relations.copy()
    relation_data["_weight"] = pd.to_numeric(relation_data[weight_col], errors="coerce").fillna(1.0) if weight_col else 1.0
    merged = base[[date_col, asset_col]].merge(relation_data, left_on=asset_col, right_on=source_col, how="left")
    target_ret = base[[date_col, asset_col, "_past_return"]].rename(columns={asset_col: target_col, "_past_return": "_target_past_return"})
    merged = merged.merge(target_ret, on=[date_col, target_col], how="left")
    merged["_weighted"] = merged["_weight"] * merged["_target_past_return"]
    agg = (
        merged.groupby([date_col, asset_col], as_index=False)
        .agg(weighted_sum=("_weighted", "sum"), weight_sum=("_weight", "sum"))
    )
    agg[output_col] = agg["weighted_sum"] / agg["weight_sum"].replace(0.0, np.nan)
    return base.drop(columns=["_past_return"]).merge(agg[[date_col, asset_col, output_col]], on=[date_col, asset_col], how="left")


def fund_implied_alpha_signal(
    holdings: pd.DataFrame,
    fund_scores: pd.DataFrame,
    date_col: str,
    fund_col: str,
    asset_col: str,
    weight_col: str,
    fund_alpha_col: str,
    output_col: str = "fund_implied_alpha",
) -> pd.DataFrame:
    """Stock signal implied by skilled funds' holdings."""
    _require_columns(holdings, [date_col, fund_col, asset_col, weight_col])
    _require_columns(fund_scores, [date_col, fund_col, fund_alpha_col])
    merged = holdings.merge(fund_scores[[date_col, fund_col, fund_alpha_col]], on=[date_col, fund_col], how="left")
    merged["_weighted_alpha"] = pd.to_numeric(merged[weight_col], errors="coerce") * pd.to_numeric(merged[fund_alpha_col], errors="coerce")
    out = merged.groupby([date_col, asset_col], as_index=False).agg(weighted_alpha=("_weighted_alpha", "sum"), total_weight=(weight_col, "sum"))
    out[output_col] = out["weighted_alpha"] / pd.to_numeric(out["total_weight"], errors="coerce").replace(0.0, np.nan)
    return out[[date_col, asset_col, output_col]]


def dpo_signal(
    prices: pd.DataFrame,
    date_col: str,
    close_col: str,
    window: int = 20,
    output_col: str = "dpo_signal",
) -> pd.DataFrame:
    """Detrended price oscillator sign signal."""
    _require_columns(prices, [date_col, close_col])
    out = prices.sort_values(date_col).copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    lag = int(window / 2 + 1)
    dpo = close - close.rolling(window, min_periods=max(3, window // 2)).mean().shift(lag)
    out["dpo"] = dpo
    out[output_col] = np.sign(dpo).replace(0.0, np.nan).ffill().fillna(0.0)
    return out


def efficiency_ratio_signal(
    prices: pd.DataFrame,
    date_col: str,
    close_col: str,
    window: int = 20,
    threshold: float = 0.2,
    output_col: str = "er_signal",
) -> pd.DataFrame:
    """Efficiency ratio trend signal."""
    _require_columns(prices, [date_col, close_col])
    out = prices.sort_values(date_col).copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    direction = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window, min_periods=max(3, window // 2)).sum()
    er = direction / volatility.replace(0.0, np.nan)
    trend = np.sign(close.diff(window))
    out["efficiency_ratio"] = er
    out[output_col] = np.where(er >= threshold, trend, 0.0)
    return out


def technical_category_vote(
    signals: pd.DataFrame,
    date_col: str,
    category_to_signal_cols: Mapping[str, Sequence[str]],
    output_col: str = "technical_vote_signal",
) -> pd.DataFrame:
    """Vote within indicator categories, then vote across categories."""
    all_cols = [col for cols in category_to_signal_cols.values() for col in cols]
    _require_columns(signals, [date_col, *all_cols])
    out = signals.copy()
    category_cols = []
    for category, cols in category_to_signal_cols.items():
        col = f"{category}_category_signal"
        category_cols.append(col)
        summed = out[list(cols)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        out[col] = np.sign(summed)
    out[output_col] = np.sign(out[category_cols].sum(axis=1))
    out[f"{output_col}_positive_share"] = (out[category_cols] > 0).sum(axis=1) / max(1, len(category_cols))
    return out


def position_from_category_votes(
    vote_df: pd.DataFrame,
    positive_share_col: str,
    rebalance_step: int = 1,
    output_col: str = "timing_position",
) -> pd.DataFrame:
    """Map share of positive categories to long-only position."""
    _require_columns(vote_df, [positive_share_col])
    out = vote_df.copy()
    position = pd.to_numeric(out[positive_share_col], errors="coerce").clip(0.0, 1.0)
    if rebalance_step > 1:
        position = position.where(np.arange(len(position)) % rebalance_step == 0).ffill()
    out[output_col] = position.fillna(0.0)
    return out


def timing_strategy_returns(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    signal_or_position_col: str,
    transaction_cost: float = 0.0,
    output_col: str = "strategy_return",
) -> pd.DataFrame:
    """Apply a timing signal or long-only position to next-period returns."""
    _require_columns(df, [date_col, return_col, signal_or_position_col])
    out = df.sort_values(date_col).copy()
    exposure = pd.to_numeric(out[signal_or_position_col], errors="coerce").fillna(0.0).shift(1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(abs(exposure))
    out[output_col] = exposure * pd.to_numeric(out[return_col], errors="coerce").fillna(0.0) - transaction_cost * turnover
    out[f"{output_col}_exposure"] = exposure
    out[f"{output_col}_turnover"] = turnover
    return out


def round50_research_checklist() -> pd.DataFrame:
    rows = [
        ("margin_financing", "Focus on financing growth over trading amount; short selling data is supply-constrained."),
        ("parallel_factor", "Parallel single-factor strategies need overlap, capacity, and attribution checks."),
        ("compensation", "Use true report availability, seasonality adjustment, industry controls, and microcap filters."),
        ("quasi_altdata", "Relation data should add information beyond industry, size, and standard momentum."),
        ("technical_timing", "Technical indicators need category voting, OOS validation, turnover, and cost checks."),
    ]
    return pd.DataFrame(rows, columns=["topic", "validation_gate"])
