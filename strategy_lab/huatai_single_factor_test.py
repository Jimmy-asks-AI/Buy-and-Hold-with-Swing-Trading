#!/usr/bin/env python
"""Huatai-style single factor testing tools.

This module standardizes the common workflow used in the Huatai multi-factor
series: cross-sectional factor cleaning, industry-controlled WLS regression,
IC/Rank IC, and quantile portfolio diagnostics. It is a research validation
tool, not a production trading engine.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _time_series_t(series: pd.Series) -> float | pd.NA:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] < 2:
        return pd.NA
    std = clean.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.NA
    return float(clean.mean() / std * math.sqrt(clean.shape[0]))


def _normalize_control_cols(control_cols: Iterable[str] | None) -> list[str]:
    return [col for col in (control_cols or []) if col]


def mad_winsorize_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    n_mad: float = 5.0,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Cap factor exposure within each date by median +/- n * median abs dev.

    The Huatai reports use 5 times median absolute deviation. When MAD is zero,
    this implementation leaves the original value unchanged instead of forcing
    all observations to the median, because sparse/tied factors would otherwise
    lose almost all cross-sectional information.
    """
    out = df.copy()
    target = output_col or value_col
    grouped = out.groupby(date_col)[value_col]
    med = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    lower = med - n_mad * mad
    upper = med + n_mad * mad
    value = pd.to_numeric(out[value_col], errors="coerce")
    capped = value.clip(lower=lower, upper=upper)
    out[target] = np.where((mad > 0) & value.notna(), capped, value)
    return out


def zscore_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Standardize factor exposure within each date."""
    out = df.copy()
    target = output_col or value_col

    def zscore(s: pd.Series) -> pd.Series:
        values = pd.to_numeric(s, errors="coerce")
        std = values.std(ddof=1)
        if pd.isna(std) or std == 0:
            return values * 0
        return (values - values.mean()) / std

    out[target] = out.groupby(date_col)[value_col].transform(zscore)
    return out


def preprocess_factor_panel(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    output_col: str = "factor_score",
    n_mad: float = 5.0,
    fill_missing_neutral: bool = True,
) -> pd.DataFrame:
    """Apply Huatai-style winsorization, z-score, and optional neutral fill."""
    winsor_col = f"_{factor_col}_mad_winsor"
    out = mad_winsorize_by_date(df, date_col, factor_col, n_mad=n_mad, output_col=winsor_col)
    out = zscore_by_date(out, date_col, winsor_col, output_col=output_col)
    if fill_missing_neutral:
        out[output_col] = out[output_col].fillna(0.0)
    return out.drop(columns=[winsor_col])


def factor_score_from_direction(
    df: pd.DataFrame,
    factor_col: str,
    expected_alpha_direction: int,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Convert a raw factor into a positive-alpha score.

    expected_alpha_direction=1 means higher raw factor should imply higher
    future return. expected_alpha_direction=-1 means lower raw factor is better,
    such as raw EV/EBITDA or A-share reversal factors.
    """
    if expected_alpha_direction not in {-1, 1}:
        raise ValueError("expected_alpha_direction must be 1 or -1.")
    out = df.copy()
    target = output_col or f"{factor_col}_alpha_score"
    out[target] = pd.to_numeric(out[factor_col], errors="coerce") * expected_alpha_direction
    return out


def _design_matrix(
    group: pd.DataFrame,
    factor_col: str,
    industry_col: str | None = None,
    control_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    controls = _normalize_control_cols(control_cols)
    parts = [pd.Series(1.0, index=group.index, name="intercept")]
    parts.append(pd.to_numeric(group[factor_col], errors="coerce").rename(factor_col))
    for col in controls:
        parts.append(pd.to_numeric(group[col], errors="coerce").rename(col))
    if industry_col:
        dummies = pd.get_dummies(group[industry_col].astype("string"), prefix=industry_col, drop_first=True)
        if not dummies.empty:
            parts.append(dummies.astype(float))
    design = pd.concat(parts, axis=1)
    return design, list(design.columns)


def _weighted_lstsq(
    y: pd.Series,
    x: pd.DataFrame,
    weights: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    x_arr = x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if weights is None:
        w_arr = np.ones_like(y_arr)
    else:
        w_arr = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr).all(axis=1) & np.isfinite(w_arr) & (w_arr > 0)
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]
    w_arr = w_arr[mask]
    if y_arr.size == 0:
        raise ValueError("No valid weighted observations.")
    sqrt_w = np.sqrt(w_arr)
    xw = x_arr * sqrt_w[:, None]
    yw = y_arr * sqrt_w
    beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y_arr - x_arr @ beta
    dof = max(int(y_arr.size - rank), 1)
    sigma2 = float(np.sum(w_arr * resid * resid) / dof)
    xtwx_inv = np.linalg.pinv(xw.T @ xw)
    se = np.sqrt(np.maximum(np.diag(xtwx_inv) * sigma2, 0.0))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    y_bar = float(np.average(y_arr, weights=w_arr))
    total = float(np.sum(w_arr * (y_arr - y_bar) ** 2))
    r2 = float(1.0 - np.sum(w_arr * resid * resid) / total) if total > 0 else np.nan
    return beta, se, t_stat, r2, int(y_arr.size)


def wls_factor_return_by_date(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    factor_col: str,
    industry_col: str | None = None,
    weight_col: str | None = None,
    control_cols: Iterable[str] | None = None,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Estimate cross-sectional factor return and t-stat by date.

    Regression specification:
        next_return_i = intercept + factor_i * factor_return
                        + controls_i + industry_dummies_i + error_i

    If weight_col is supplied, it is used as a positive WLS sample weight. For
    Huatai report replication this is usually sqrt(float market cap).
    """
    rows = []
    required = [date_col, return_col, factor_col]
    if weight_col:
        required.append(weight_col)
    for date, group in df.dropna(subset=required).groupby(date_col):
        design, columns = _design_matrix(group, factor_col, industry_col=industry_col, control_cols=control_cols)
        clean = pd.concat([group[[return_col]], design], axis=1).dropna()
        if clean.shape[0] < max(min_assets, len(columns) + 2):
            continue
        weights = group.loc[clean.index, weight_col] if weight_col else None
        try:
            beta, se, t_stat, r2, n_obs = _weighted_lstsq(clean[return_col], clean[columns], weights)
        except ValueError:
            continue
        factor_idx = columns.index(factor_col)
        rows.append(
            {
                "date": date,
                "factor": factor_col,
                "factor_return": float(beta[factor_idx]),
                "factor_return_se": float(se[factor_idx]),
                "factor_t_stat": float(t_stat[factor_idx]),
                "n_assets": n_obs,
                "n_regressors": len(columns),
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def ic_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Compute Pearson IC and Rank IC by date."""
    rows = []
    clean = df.dropna(subset=[date_col, factor_col, return_col])
    for date, group in clean.groupby(date_col):
        if group.shape[0] < min_assets:
            continue
        factor = pd.to_numeric(group[factor_col], errors="coerce")
        future_return = pd.to_numeric(group[return_col], errors="coerce")
        valid = factor.notna() & future_return.notna()
        factor = factor[valid]
        future_return = future_return[valid]
        if factor.shape[0] < min_assets:
            continue
        rows.append(
            {
                "date": date,
                "factor": factor_col,
                "n_assets": int(factor.shape[0]),
                "ic": float(factor.corr(future_return)),
                "rank_ic": float(factor.rank().corr(future_return.rank())),
            }
        )
    return pd.DataFrame(rows)


def quantile_returns_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    groups: int = 5,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Compute equal-weight future return by factor quantile group."""
    rows = []
    clean = df.dropna(subset=[date_col, factor_col, return_col]).copy()
    for date, group in clean.groupby(date_col):
        if group.shape[0] < max(min_assets, groups):
            continue
        try:
            quantile = pd.qcut(group[factor_col].rank(method="first"), groups, labels=False) + 1
        except ValueError:
            continue
        temp = group.assign(factor_group=quantile.astype(int))
        grouped = temp.groupby("factor_group")[return_col].agg(["mean", "count"]).reset_index()
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "factor_group": int(row["factor_group"]),
                    "return": float(row["mean"]),
                    "count": int(row["count"]),
                }
            )
        pivot = grouped.set_index("factor_group")["mean"]
        if 1 in pivot.index and groups in pivot.index:
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "factor_group": "long_short",
                    "return": float(pivot.loc[groups] - pivot.loc[1]),
                    "count": int(group.shape[0]),
                }
            )
        if grouped.shape[0] >= 3:
            group_rank = grouped["factor_group"].astype(float).rank()
            return_rank = grouped["mean"].astype(float).rank()
            monotonicity = group_rank.corr(return_rank)
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "factor_group": "monotonicity",
                    "return": float(monotonicity),
                    "count": int(group.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def coverage_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    universe_col: str | None = None,
) -> pd.DataFrame:
    """Report factor non-null coverage by date and optional universe bucket."""
    factors = list(factor_cols)
    group_cols = [date_col] + ([universe_col] if universe_col else [])
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {"date": keys[0], "n_assets": int(group.shape[0])}
        if universe_col:
            base[universe_col] = keys[1]
        for factor in factors:
            valid = pd.to_numeric(group[factor], errors="coerce").notna()
            row = dict(base)
            row.update(
                {
                    "factor": factor,
                    "covered_assets": int(valid.sum()),
                    "coverage": float(valid.mean()) if group.shape[0] else pd.NA,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def factor_correlation_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    method: str = "spearman",
    min_assets: int = 30,
) -> pd.DataFrame:
    """Compute pairwise factor correlations by date without SciPy dependency."""
    factors = list(factor_cols)
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    rows = []
    for date, group in df.groupby(date_col):
        data = group[factors].apply(pd.to_numeric, errors="coerce")
        if method == "spearman":
            data = data.rank()
        for i, left in enumerate(factors):
            for right in factors[i + 1 :]:
                pair = data[[left, right]].dropna()
                if pair.shape[0] < min_assets:
                    continue
                rows.append(
                    {
                        "date": date,
                        "factor_left": left,
                        "factor_right": right,
                        "correlation": float(pair[left].corr(pair[right])),
                        "n_assets": int(pair.shape[0]),
                        "method": method,
                    }
                )
    return pd.DataFrame(rows)


def summarize_factor_correlation(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise factor correlations across dates."""
    if correlation_df.empty:
        return pd.DataFrame()
    grouped = correlation_df.groupby(["factor_left", "factor_right"])["correlation"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["mean_abs"] = grouped.apply(lambda s: s.abs().mean()).reset_index(drop=True)
    return out.sort_values("mean_abs", ascending=False)


def rank_within_industry(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    industry_col: str,
    output_col: str | None = None,
    pct: bool = True,
) -> pd.DataFrame:
    """Create within-industry rank factor, useful for consensus factors."""
    out = df.copy()
    target = output_col or f"{factor_col}_industry_rank"
    out[target] = (
        out.groupby([date_col, industry_col])[factor_col]
        .rank(method="average", ascending=True, pct=pct)
        .astype(float)
    )
    return out


def relative_change_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    value_col: str,
    periods: int = 1,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Compute asset-level relative change, e.g. consensus quarterly revision."""
    out = df.sort_values([asset_col, date_col]).copy()
    target = output_col or f"{value_col}_rel_change_{periods}"
    previous = out.groupby(asset_col)[value_col].shift(periods)
    out[target] = pd.to_numeric(out[value_col], errors="coerce") / pd.to_numeric(previous, errors="coerce") - 1.0
    return out


def summarize_single_factor(
    regression_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize regression, IC, and stratified return evidence."""
    factor_values = []
    for frame in [regression_df, ic_df, quantile_df]:
        if not frame.empty and "factor" in frame.columns:
            factor_values.extend(frame["factor"].dropna().astype(str).unique().tolist())
    factor_name = factor_values[0] if factor_values else ""

    reg_return = regression_df["factor_return"] if "factor_return" in regression_df else pd.Series(dtype=float)
    reg_t = regression_df["factor_t_stat"] if "factor_t_stat" in regression_df else pd.Series(dtype=float)
    ic = ic_df["ic"] if "ic" in ic_df else pd.Series(dtype=float)
    rank_ic = ic_df["rank_ic"] if "rank_ic" in ic_df else pd.Series(dtype=float)
    long_short = (
        quantile_df.loc[quantile_df["factor_group"].eq("long_short"), "return"]
        if not quantile_df.empty and "factor_group" in quantile_df
        else pd.Series(dtype=float)
    )
    monotonicity = (
        quantile_df.loc[quantile_df["factor_group"].eq("monotonicity"), "return"]
        if not quantile_df.empty and "factor_group" in quantile_df
        else pd.Series(dtype=float)
    )

    def ir(series: pd.Series) -> float | pd.NA:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.shape[0] < 2:
            return pd.NA
        std = clean.std(ddof=1)
        return float(clean.mean() / std) if std and not pd.isna(std) else pd.NA

    row = {
        "factor": factor_name,
        "n_regression_dates": int(regression_df.shape[0]),
        "factor_return_mean": float(reg_return.mean()) if not reg_return.empty else pd.NA,
        "factor_return_time_t": _time_series_t(reg_return),
        "abs_t_mean": float(reg_t.abs().mean()) if not reg_t.empty else pd.NA,
        "abs_t_gt2_rate": float((reg_t.abs() > 2).mean()) if not reg_t.empty else pd.NA,
        "n_ic_dates": int(ic_df.shape[0]),
        "ic_mean": float(ic.mean()) if not ic.empty else pd.NA,
        "ic_ir": ir(ic),
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else pd.NA,
        "rank_ic_ir": ir(rank_ic),
        "rank_ic_win_rate": float((rank_ic > 0).mean()) if not rank_ic.empty else pd.NA,
        "long_short_mean": float(long_short.mean()) if not long_short.empty else pd.NA,
        "long_short_time_t": _time_series_t(long_short),
        "monotonicity_mean": float(monotonicity.mean()) if not monotonicity.empty else pd.NA,
    }
    return pd.DataFrame([row])


def run_single_factor_test(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    industry_col: str | None = None,
    weight_col: str | None = None,
    control_cols: Iterable[str] | None = None,
    groups: int = 5,
    min_assets: int = 30,
    expected_alpha_direction: int = 1,
    score_col: str = "factor_score",
) -> dict[str, pd.DataFrame]:
    """Run the full single factor validation workflow."""
    scored = factor_score_from_direction(df, factor_col, expected_alpha_direction, output_col=f"_{factor_col}_signed")
    prepared = preprocess_factor_panel(scored, date_col, f"_{factor_col}_signed", output_col=score_col)
    prepared = prepared.drop(columns=[f"_{factor_col}_signed"])
    regression_df = wls_factor_return_by_date(
        prepared,
        date_col=date_col,
        return_col=return_col,
        factor_col=score_col,
        industry_col=industry_col,
        weight_col=weight_col,
        control_cols=control_cols,
        min_assets=min_assets,
    )
    ic_df = ic_by_date(prepared, date_col, score_col, return_col, min_assets=min_assets)
    quantile_df = quantile_returns_by_date(prepared, date_col, score_col, return_col, groups=groups, min_assets=min_assets)
    summary_df = summarize_single_factor(regression_df, ic_df, quantile_df)
    if not summary_df.empty:
        summary_df.loc[:, "raw_factor"] = factor_col
        summary_df.loc[:, "expected_alpha_direction"] = expected_alpha_direction
    return {
        "prepared_data": prepared,
        "regression_by_date": regression_df,
        "ic_by_date": ic_df,
        "quantile_returns": quantile_df,
        "summary": summary_df,
    }


def huatai_factor_definitions() -> pd.DataFrame:
    """Return reusable definitions and raw-factor alpha directions."""
    rows = [
        ("EP", "valuation", "net profit TTM / market cap", 1, "secondary"),
        ("EPcut", "valuation", "recurring net profit TTM / market cap", 1, "secondary"),
        ("BP", "valuation", "net assets / market cap", 1, "core"),
        ("SP", "valuation", "revenue TTM / market cap", 1, "core"),
        ("NCFP", "valuation", "net cash flow TTM / market cap", 1, "weak"),
        ("OCFP", "valuation", "operating cash flow TTM / market cap", 1, "secondary"),
        ("FCFP", "valuation", "free cash flow latest annual / market cap", 1, "weak"),
        ("DP", "valuation", "cash dividend in last 12 months / market cap", 1, "secondary"),
        ("EV2EBITDA", "valuation", "enterprise value excluding cash / EBITDA", -1, "core"),
        ("PEG", "valuation", "PE / net profit TTM YoY growth", -1, "core"),
        ("Sales_G_q", "growth", "quarterly sales YoY growth", 1, "core"),
        ("Sales_G_ttm", "growth", "TTM sales YoY growth", 1, "secondary"),
        ("Sales_G_3y", "growth", "three-year compound sales growth", 1, "weak"),
        ("Profit_G_q", "growth", "quarterly profit YoY growth", 1, "core"),
        ("Profit_G_ttm", "growth", "TTM profit YoY growth", 1, "secondary"),
        ("Profit_G_3y", "growth", "three-year compound profit growth", 1, "weak"),
        ("OCF_G_q", "growth", "quarterly operating cash flow YoY growth", 1, "secondary"),
        ("OCF_G_ttm", "growth", "TTM operating cash flow YoY growth", 1, "weak"),
        ("OCF_G_3y", "growth", "three-year compound operating cash flow growth", 1, "weak"),
        ("ROE_G_q", "growth", "quarterly ROE YoY growth", 1, "core"),
        ("ROE_G_ttm", "growth", "TTM ROE YoY growth", 1, "secondary"),
        ("ROE_G_3y", "growth", "three-year compound ROE growth", 1, "weak"),
        ("HAlpha", "momentum_reversal", "60-month regression intercept vs Shanghai Composite", -1, "weak"),
        ("return_1m", "momentum_reversal", "past 1-month return; lower is better in reversal tests", -1, "core"),
        ("return_3m", "momentum_reversal", "past 3-month return; lower is better in reversal tests", -1, "secondary"),
        ("return_6m", "momentum_reversal", "past 6-month return; lower is better in reversal tests", -1, "secondary"),
        ("return_12m", "momentum_reversal", "past 12-month return; lower is better in reversal tests", -1, "secondary"),
        ("wgt_return_1m", "momentum_reversal", "turnover-weighted past 1-month return", -1, "core"),
        ("wgt_return_3m", "momentum_reversal", "turnover-weighted past 3-month return", -1, "secondary"),
        ("wgt_return_6m", "momentum_reversal", "turnover-weighted past 6-month return", -1, "secondary"),
        ("wgt_return_12m", "momentum_reversal", "turnover-weighted past 12-month return", -1, "secondary"),
        ("exp_wgt_return_1m", "momentum_reversal", "exponential-decay turnover-weighted past 1-month return", -1, "secondary"),
        ("exp_wgt_return_3m", "momentum_reversal", "exponential-decay turnover-weighted past 3-month return", -1, "core"),
        ("exp_wgt_return_6m", "momentum_reversal", "exponential-decay turnover-weighted past 6-month return", -1, "core"),
        ("exp_wgt_return_12m", "momentum_reversal", "exponential-decay turnover-weighted past 12-month return", -1, "secondary"),
        ("turn_1m", "turnover", "average daily turnover in the past 1 month", -1, "core"),
        ("turn_3m", "turnover", "average daily turnover in the past 3 months", -1, "secondary"),
        ("turn_6m", "turnover", "average daily turnover in the past 6 months", -1, "secondary"),
        ("bias_turn_1m", "turnover", "past 1-month average turnover / past 2-year average turnover - 1", -1, "core"),
        ("bias_turn_3m", "turnover", "past 3-month average turnover / past 2-year average turnover - 1", -1, "secondary"),
        ("bias_turn_6m", "turnover", "past 6-month average turnover / past 2-year average turnover - 1", -1, "secondary"),
        ("std_turn_1m", "turnover", "standard deviation of daily turnover in the past 1 month", -1, "core"),
        ("std_turn_3m", "turnover", "standard deviation of daily turnover in the past 3 months", -1, "secondary"),
        ("std_turn_6m", "turnover", "standard deviation of daily turnover in the past 6 months", -1, "secondary"),
        ("bias_std_turn_1m", "turnover", "past 1-month turnover std / past 2-year turnover std - 1", -1, "core"),
        ("bias_std_turn_3m", "turnover", "past 3-month turnover std / past 2-year turnover std - 1", -1, "secondary"),
        ("bias_std_turn_6m", "turnover", "past 6-month turnover std / past 2-year turnover std - 1", -1, "secondary"),
        ("std_4m", "volatility", "standard deviation of daily returns in the past 4 months", -1, "secondary"),
        ("std_1m", "volatility", "standard deviation of daily returns in the past 1 month", -1, "core"),
        ("id1_std_3m", "volatility", "CAPM residual volatility in the past 3 months", -1, "secondary"),
        ("id2_std_3m", "volatility", "Fama-French 3-factor residual volatility in the past 3 months", -1, "core"),
        ("id2_std_up_3m", "volatility", "upside Fama-French residual volatility in the past 3 months", -1, "core"),
        ("id2_std_down_3m", "volatility", "downside Fama-French residual volatility in the past 3 months", -1, "core"),
        ("id2_std_upd_3m", "volatility", "sum of upside and downside Fama-French residual volatility", -1, "core"),
        ("high_r_std_4m", "volatility", "standard deviation of intraday maximum positive return in the past 4 months", -1, "secondary"),
        ("hml_r_std_5m", "volatility", "volatility of intraday max-up minus max-down return in the past 5 months", -1, "core"),
        ("qfa_roe", "financial_quality", "single-quarter ROE", 1, "core"),
        ("qfa_roa", "financial_quality", "single-quarter ROA", 1, "core"),
        ("qfa_grossprofitmargin", "financial_quality", "single-quarter gross profit margin", 1, "secondary"),
        ("nptocostexpense_qfa", "financial_quality", "single-quarter net profit / cost expense", 1, "secondary"),
        ("roic", "financial_quality", "return on invested capital", 1, "core"),
        ("qfa_operateincometoebt", "financial_quality", "single-quarter operating income / EBT", 1, "core"),
        ("qfa_deductedprofittoprofit", "financial_quality", "single-quarter recurring profit / net profit", 1, "secondary"),
        ("qfa_ocftosales", "financial_quality", "single-quarter operating cash flow / revenue", 1, "secondary"),
        ("currentdebttodebt", "financial_quality", "current liabilities / total liabilities", 1, "secondary"),
        ("CON_EP", "consensus_expectation", "consensus expected earnings yield", 1, "core"),
        ("CON_EP_RANK", "consensus_expectation", "within-industry rank of consensus expected earnings yield", 1, "core"),
        ("CON_EP_REL", "consensus_expectation", "quarterly relative change of consensus expected earnings yield", 1, "secondary"),
        ("CON_BP", "consensus_expectation", "consensus expected book-to-price", 1, "core"),
        ("CON_BP_RANK", "consensus_expectation", "within-industry rank of consensus expected book-to-price", 1, "core"),
        ("CON_BP_REL", "consensus_expectation", "quarterly relative change of consensus expected book-to-price", 1, "secondary"),
        ("CON_EPS", "consensus_expectation", "consensus expected EPS", 1, "secondary"),
        ("CON_EPS_RANK", "consensus_expectation", "within-industry rank of consensus expected EPS", 1, "secondary"),
        ("CON_EPS_REL", "consensus_expectation", "quarterly relative change of consensus expected EPS", 1, "core"),
        ("CON_ROE", "consensus_expectation", "consensus expected ROE", 1, "secondary"),
        ("CON_ROE_RANK", "consensus_expectation", "within-industry rank of consensus expected ROE", 1, "secondary"),
        ("CON_ROE_REL", "consensus_expectation", "quarterly relative change of consensus expected ROE", 1, "secondary"),
        ("CON_NP", "consensus_expectation", "consensus expected net profit attributable to parent", 1, "secondary"),
        ("CON_NP_RANK", "consensus_expectation", "within-industry rank of consensus expected net profit", 1, "secondary"),
        ("CON_NP_REL", "consensus_expectation", "quarterly relative change of consensus expected net profit", 1, "secondary"),
        ("BUY_NUMBER", "consensus_expectation", "quarterly count of buy-rating analyst reports", 1, "core"),
    ]
    return pd.DataFrame(
        rows,
        columns=["factor", "category", "definition", "expected_alpha_direction", "huatai_evidence_tier"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--return-col", required=True)
    parser.add_argument("--industry-col")
    parser.add_argument("--weight-col")
    parser.add_argument("--control-cols", default="", help="Comma-separated numeric controls, e.g. log_float_mkt_cap.")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--expected-alpha-direction", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--output-dir", default="huatai_single_factor_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    control_cols = [col.strip() for col in args.control_cols.split(",") if col.strip()]
    results = run_single_factor_test(
        df=df,
        date_col=args.date_col,
        factor_col=args.factor_col,
        return_col=args.return_col,
        industry_col=args.industry_col,
        weight_col=args.weight_col,
        control_cols=control_cols,
        groups=args.groups,
        min_assets=args.min_assets,
        expected_alpha_direction=args.expected_alpha_direction,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    huatai_factor_definitions().to_csv(output_dir / "huatai_factor_definitions.csv", index=False, encoding="utf-8-sig")
    print(results["summary"])
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
