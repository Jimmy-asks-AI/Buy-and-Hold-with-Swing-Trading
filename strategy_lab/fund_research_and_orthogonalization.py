#!/usr/bin/env python
"""Fund research, fund-factor testing, and factor orthogonalization helpers."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def filter_fund_sample(
    df: pd.DataFrame,
    date_col: str,
    inception_col: str,
    manager_change_col: str | None = None,
    min_history_years: float = 2.0,
    manager_stability_years: float = 2.0,
) -> pd.DataFrame:
    """Filter funds by age and recent manager stability."""
    out = df.copy()
    date = pd.to_datetime(out[date_col])
    inception = pd.to_datetime(out[inception_col])
    keep = (date - inception).dt.days >= int(min_history_years * 365)
    if manager_change_col and manager_change_col in out.columns:
        manager_change = pd.to_datetime(out[manager_change_col], errors="coerce")
        recent_change = manager_change.notna() & ((date - manager_change).dt.days < int(manager_stability_years * 365))
        keep = keep & ~recent_change
    return out.loc[keep].copy()


def mad_winsorize(values: pd.Series, n_mad: float = 3.0) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    median = clean.median()
    mad = (clean - median).abs().median()
    if pd.isna(mad) or mad <= EPS:
        return clean
    return clean.clip(median - n_mad * mad, median + n_mad * mad)


def zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    std = clean.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return clean * 0.0
    return (clean - clean.mean()) / std


def preprocess_fund_factor(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    label_cols: Iterable[str] | None = None,
    output_col: str | None = None,
    n_mad: float = 3.0,
) -> pd.DataFrame:
    """Winsorize and z-score fund factors by date and optional labels."""
    out = df.copy()
    target = output_col or f"{factor_col}_processed"
    group_cols = [date_col, *list(label_cols or [])]

    def transform(s: pd.Series) -> pd.Series:
        return zscore(mad_winsorize(s, n_mad=n_mad))

    out[target] = out.groupby(group_cols, dropna=False)[factor_col].transform(transform)
    return out


def regression_neutralize_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    control_cols: Iterable[str],
    output_col: str | None = None,
    add_polynomial_controls: bool = False,
) -> pd.DataFrame:
    """Cross-sectional regression orthogonalization by date."""
    out = df.copy()
    target = output_col or f"{factor_col}_neutral"
    controls = list(control_cols)
    residuals = []
    for _, group in out.groupby(date_col, sort=False):
        work = group[[factor_col, *controls]].apply(pd.to_numeric, errors="coerce")
        if add_polynomial_controls:
            for col in controls:
                work[f"{col}_squared"] = work[col] ** 2
        clean = work.dropna()
        if clean.empty:
            residuals.append(pd.Series(np.nan, index=group.index, name=target))
            continue
        x_cols = [col for col in clean.columns if col != factor_col]
        x = np.column_stack([np.ones(len(clean)), clean[x_cols].to_numpy(dtype=float)])
        y = clean[factor_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residuals.append(pd.Series(y - x @ beta, index=clean.index, name=target).reindex(group.index))
    out[target] = pd.concat(residuals).sort_index()
    return out


def grouped_neutralize_bucket(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    control_col: str,
    control_buckets: int = 10,
    factor_buckets: int = 10,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Double-sort factor buckets within control buckets."""
    out = df.copy()
    target = output_col or f"{factor_col}_group_bucket"
    pieces = []
    for _, group in out.groupby(date_col, sort=False):
        work = group[[factor_col, control_col]].apply(pd.to_numeric, errors="coerce").dropna()
        bucket = pd.Series(np.nan, index=group.index, name=target)
        if work.shape[0] < control_buckets * factor_buckets:
            pieces.append(bucket)
            continue
        try:
            control_bucket = pd.qcut(work[control_col].rank(method="first"), control_buckets, labels=False)
        except ValueError:
            pieces.append(bucket)
            continue
        temp = work.assign(_control_bucket=control_bucket)
        for _, sub in temp.groupby("_control_bucket"):
            if sub.shape[0] < factor_buckets:
                continue
            try:
                fb = pd.qcut(sub[factor_col].rank(method="first"), factor_buckets, labels=False) + 1
            except ValueError:
                continue
            bucket.loc[sub.index] = fb.astype(float)
        pieces.append(bucket)
    out[target] = pd.concat(pieces).sort_index()
    return out


def rank_ic_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    forward_return_col: str,
    min_count: int = 20,
) -> pd.DataFrame:
    """Compute fund factor RankIC by date."""
    rows = []
    for date, group in df.dropna(subset=[factor_col, forward_return_col]).groupby(date_col):
        if group.shape[0] < min_count:
            continue
        ic = group[factor_col].rank().corr(group[forward_return_col].rank())
        rows.append({"date": date, "rank_ic": float(ic)})
    return pd.DataFrame(rows)


def factor_icir_summary(ic_df: pd.DataFrame, ic_col: str = "rank_ic") -> dict[str, float]:
    """Summarize IC mean, volatility, ICIR, and positive ratio."""
    ic = pd.to_numeric(ic_df[ic_col], errors="coerce").dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "positive_ratio": np.nan}
    std = ic.std(ddof=1)
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(std),
        "icir": float(ic.mean() / std) if std > EPS else np.nan,
        "positive_ratio": float((ic > 0).mean()),
    }


def grouped_factor_returns(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    forward_return_col: str,
    groups: int = 5,
    min_count: int = 20,
) -> pd.DataFrame:
    """Compute equal-weight forward returns by factor group."""
    rows = []
    for date, group in df.dropna(subset=[factor_col, forward_return_col]).groupby(date_col):
        if group.shape[0] < max(min_count, groups):
            continue
        try:
            bucket = pd.qcut(group[factor_col].rank(method="first"), groups, labels=False) + 1
        except ValueError:
            continue
        temp = group.assign(bucket=bucket.astype(int))
        ret = temp.groupby("bucket")[forward_return_col].mean()
        for bucket_id, value in ret.items():
            rows.append({"date": date, "bucket": int(bucket_id), "return": float(value)})
        if 1 in ret.index and groups in ret.index:
            rows.append({"date": date, "bucket": "long_short", "return": float(ret.loc[groups] - ret.loc[1])})
    return pd.DataFrame(rows)


def holding_concentration_hhi(weights: pd.Series | np.ndarray) -> float:
    """Herfindahl-Hirschman concentration index for fund holdings."""
    w = pd.Series(weights, dtype="float64").dropna()
    total = w.abs().sum()
    if total <= EPS:
        return np.nan
    norm = w.abs() / total
    return float((norm ** 2).sum())


def holding_retention_rate(current_holdings: Iterable, previous_holdings: Iterable) -> float:
    """Share of current holdings that also appeared in the previous period."""
    current = set(current_holdings)
    previous = set(previous_holdings)
    if not current:
        return np.nan
    return len(current & previous) / len(current)


def fund_factor_dimension_map() -> pd.DataFrame:
    """Reusable map of fund factor dimensions by fund type."""
    rows = [
        ("active_equity", "return", "positive_return_probability, return_stability, sharpe, sortino, information_ratio"),
        ("active_equity", "skill", "stock_trading_profitability, stock_holding_profitability"),
        ("active_equity", "flow_size", "net_subscription, fund_size_change"),
        ("active_equity", "timing", "equity_timing_return, equity_timing_win_rate"),
        ("active_equity", "holder", "institutional_holder_ratio, manager_holding_ratio"),
        ("active_equity", "holding_style", "sector_hhi, industry_hhi, holding_retention"),
        ("fixed_income_plus", "return", "cumulative_return, annual_volatility, calmar, sortino"),
        ("fixed_income_plus", "skill", "ipo_offline_profit, rate_bull_return, credit_bull_return"),
        ("fixed_income_plus", "flow_size", "company_bond_fund_aum, share_change, size_change"),
        ("fixed_income_plus", "timing", "equity_timing_win_rate, equity_timing_return"),
        ("pure_bond", "return", "cumulative_return, annual_volatility"),
        ("pure_bond", "skill", "rate_bull_return, credit_bull_return, carry_skill"),
        ("pure_bond", "flow_size", "leverage_ratio, share_change"),
        ("pure_bond", "holder", "institutional_holder_ratio, manager_holding_ratio"),
    ]
    return pd.DataFrame(rows, columns=["fund_type", "dimension", "examples"])


def orthogonalization_checklist() -> pd.DataFrame:
    rows = [
        ("scale", "Winsorize and standardize before regression neutralization."),
        ("linearity", "Regression residuals remove linear exposure only."),
        ("nonlinear", "Check residual factor buckets against control-factor averages."),
        ("collinearity", "Control factors can be correlated and change residual interpretation."),
        ("grouping", "Group neutralization handles one control well but scales poorly to many controls."),
        ("sample", "Choose full-universe or sub-universe neutralization deliberately."),
        ("optimization", "Residual factors should remain compatible with portfolio construction."),
        ("funds", "Fund factors must be neutralized within comparable labels or risk buckets."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension-map", action="store_true")
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.dimension_map:
        print(fund_factor_dimension_map())
    if args.checklist:
        print(orthogonalization_checklist())


if __name__ == "__main__":
    main()
