#!/usr/bin/env python
"""Investor sentiment index construction and validation tools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def bull_bear_support_index(
    bullish: pd.Series | np.ndarray,
    bearish: pd.Series | np.ndarray,
    asymmetric_factor: float = 1.0,
    log_strength: bool = True,
) -> pd.Series:
    """Construct online-community support tendency from bullish/bearish counts.

    Positive values indicate bullish support; negative values indicate bearish
    support. ``asymmetric_factor`` lets bad-news sensitivity differ from
    good-news sensitivity.
    """
    bull = pd.Series(bullish, dtype="float64")
    bear = pd.Series(bearish, dtype="float64")
    adj_bear = float(asymmetric_factor) * bear
    denom = bull + adj_bear
    raw = (bull - adj_bear) / denom.replace(0.0, np.nan)
    if log_strength:
        raw = raw * np.log1p(bull + bear)
    return raw.fillna(0.0)


def agreement_index(
    bullish: pd.Series | np.ndarray,
    bearish: pd.Series | np.ndarray,
) -> pd.Series:
    """Antweiler-Frank style agreement index in [0, 1]."""
    bull = pd.Series(bullish, dtype="float64")
    bear = pd.Series(bearish, dtype="float64")
    denom = (bull + bear).replace(0.0, np.nan)
    disagreement = ((bull - bear) / denom) ** 2
    index = 1.0 - np.sqrt(1.0 - disagreement.clip(lower=0.0, upper=1.0))
    return index.fillna(0.0).clip(lower=0.0, upper=1.0)


def sentiment_change(
    df: pd.DataFrame,
    date_col: str,
    sentiment_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Add first difference of a sentiment index."""
    out = df.sort_values(date_col).copy()
    target = output_col or f"delta_{sentiment_col}"
    out[target] = pd.to_numeric(out[sentiment_col], errors="coerce").diff()
    return out


def pca_sentiment_index(
    df: pd.DataFrame,
    date_col: str,
    proxy_cols: Iterable[str],
    output_col: str = "pca_sentiment",
    standardize: bool = True,
    align_col: str | None = None,
) -> pd.DataFrame:
    """Build a first-principal-component sentiment index from proxy variables."""
    proxies = list(proxy_cols)
    out = df.sort_values(date_col).copy()
    data = out[proxies].apply(pd.to_numeric, errors="coerce")
    if standardize:
        data = (data - data.mean()) / data.std(ddof=1).replace(0.0, np.nan)
    filled = data.apply(lambda s: s.fillna(s.mean()), axis=0).fillna(0.0)
    cov = filled.cov().to_numpy(dtype=float)
    eigval, eigvec = np.linalg.eigh(cov)
    loading = eigvec[:, int(np.argmax(eigval))]
    if align_col and align_col in out.columns:
        trial = filled.to_numpy(dtype=float) @ loading
        corr = pd.Series(trial).corr(pd.to_numeric(out[align_col], errors="coerce").reset_index(drop=True))
        if pd.notna(corr) and corr < 0:
            loading = -loading
    elif loading.sum() < 0:
        loading = -loading
    out[output_col] = filled.to_numpy(dtype=float) @ loading
    for col, weight in zip(proxies, loading, strict=True):
        out[f"{output_col}_loading_{col}"] = float(weight)
    return out


def add_distributed_lags(
    df: pd.DataFrame,
    date_col: str,
    cols: Iterable[str],
    lags: int = 5,
) -> pd.DataFrame:
    """Add distributed lag columns for time-series regressions."""
    out = df.sort_values(date_col).copy()
    for col in cols:
        for lag in range(1, lags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def ols_fit(y: pd.Series, x: pd.DataFrame) -> dict[str, object]:
    """Small OLS helper with coefficient t-stats."""
    clean = pd.concat([y.rename("y"), x], axis=1).dropna()
    if clean.shape[0] <= x.shape[1] + 1:
        return {"n_obs": int(clean.shape[0]), "coef": pd.Series(dtype=float), "t_stat": pd.Series(dtype=float), "r2": np.nan}
    yv = clean["y"].to_numpy(dtype=float)
    xv = clean.drop(columns=["y"]).to_numpy(dtype=float)
    xv = np.column_stack([np.ones(xv.shape[0]), xv])
    names = ["intercept", *clean.drop(columns=["y"]).columns.tolist()]
    beta, _, rank, _ = np.linalg.lstsq(xv, yv, rcond=None)
    resid = yv - xv @ beta
    dof = max(int(yv.size - rank), 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(xv.T @ xv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    total = float(((yv - yv.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / total) if total > 0 else np.nan
    return {
        "n_obs": int(yv.size),
        "coef": pd.Series(beta, index=names),
        "t_stat": pd.Series(t, index=names),
        "r2": r2,
    }


def sentiment_predictive_regression(
    df: pd.DataFrame,
    target_col: str,
    sentiment_cols: Iterable[str],
    control_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run a simple predictive regression and return coefficient table."""
    x_cols = [*list(sentiment_cols), *list(control_cols or [])]
    result = ols_fit(df[target_col], df[x_cols])
    rows = []
    for name, coef in result["coef"].items():
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "t_stat": float(result["t_stat"].get(name, np.nan)),
                "n_obs": result["n_obs"],
                "r2": result["r2"],
            }
        )
    return pd.DataFrame(rows)


def sentiment_volatility_design(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    sentiment_change_col: str,
    agreement_col: str | None = None,
    output_prefix: str = "vol",
) -> pd.DataFrame:
    """Create variables for GARCH/GJR-style volatility equations."""
    out = df.sort_values(date_col).copy()
    ret = pd.to_numeric(out[return_col], errors="coerce")
    out[f"{output_prefix}_squared_return"] = ret**2
    out[f"{output_prefix}_negative_return_dummy"] = (ret < 0).astype(int)
    out[f"{output_prefix}_sentiment_change_abs"] = pd.to_numeric(out[sentiment_change_col], errors="coerce").abs()
    out[f"{output_prefix}_sentiment_change_negative"] = (pd.to_numeric(out[sentiment_change_col], errors="coerce") < 0).astype(int)
    if agreement_col:
        out[f"{output_prefix}_agreement"] = pd.to_numeric(out[agreement_col], errors="coerce")
    return out


def herding_consistency_score(
    df: pd.DataFrame,
    date_col: str,
    bullish_col: str,
    bearish_col: str,
    volume_col: str | None = None,
    output_col: str = "herding_consistency",
) -> pd.DataFrame:
    """Score sentiment herding using agreement and optional activity intensity."""
    out = df.sort_values(date_col).copy()
    agreement = agreement_index(out[bullish_col], out[bearish_col])
    if volume_col:
        activity = np.log1p(pd.to_numeric(out[volume_col], errors="coerce").fillna(0.0))
        if activity.std(ddof=1) and not pd.isna(activity.std(ddof=1)):
            activity = (activity - activity.mean()) / activity.std(ddof=1)
        out[output_col] = agreement * (1.0 + activity.clip(lower=0.0))
    else:
        out[output_col] = agreement
    return out


def extreme_state_flags(
    df: pd.DataFrame,
    value_col: str,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Flag low/high extreme sentiment, flow, or liquidity states."""
    if not 0 < lower_q < upper_q < 1:
        raise ValueError("Require 0 < lower_q < upper_q < 1.")
    out = df.copy()
    name = prefix or value_col
    values = pd.to_numeric(out[value_col], errors="coerce")
    lo = values.quantile(lower_q)
    hi = values.quantile(upper_q)
    out[f"{name}_low_state"] = values <= lo
    out[f"{name}_high_state"] = values >= hi
    return out


def panel_fixed_effect_regression(
    df: pd.DataFrame,
    target_col: str,
    x_cols: Iterable[str],
    entity_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """OLS with optional entity/time demeaning as a lightweight FE regression."""
    cols = [target_col, *list(x_cols)]
    clean = df.dropna(subset=cols).copy()
    if clean.empty:
        return pd.DataFrame()
    y = clean[target_col].astype(float)
    x = clean[list(x_cols)].astype(float)
    if entity_col:
        y = y - clean.groupby(entity_col)[target_col].transform("mean")
        for col in x.columns:
            x[col] = x[col] - clean.groupby(entity_col)[col].transform("mean")
    if time_col:
        y = y - pd.Series(y).groupby(clean[time_col]).transform("mean")
        for col in x.columns:
            x[col] = x[col] - x[col].groupby(clean[time_col]).transform("mean")
    result = ols_fit(pd.Series(y, index=clean.index), x)
    rows = []
    for name, coef in result["coef"].items():
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "t_stat": float(result["t_stat"].get(name, np.nan)),
                "n_obs": result["n_obs"],
                "r2": result["r2"],
                "entity_fe": bool(entity_col),
                "time_fe": bool(time_col),
            }
        )
    return pd.DataFrame(rows)


def add_interaction_terms(
    df: pd.DataFrame,
    left_col: str,
    right_cols: Iterable[str],
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add interaction terms for sentiment x risk-aversion/state tests."""
    out = df.copy()
    base = prefix or left_col
    for col in right_cols:
        out[f"{base}_x_{col}"] = pd.to_numeric(out[left_col], errors="coerce") * pd.to_numeric(out[col], errors="coerce")
    return out


def aggregate_fund_holding_beta(
    holdings: pd.DataFrame,
    date_col: str,
    fund_col: str,
    weight_col: str,
    stock_beta_col: str,
    fund_size_col: str | None = None,
) -> pd.DataFrame:
    """Estimate aggregate mutual-fund market beta from holdings."""
    work = holdings.dropna(subset=[date_col, fund_col, weight_col, stock_beta_col]).copy()
    work["_holding_beta"] = work[weight_col].astype(float) * work[stock_beta_col].astype(float)
    fund_beta = work.groupby([date_col, fund_col])["_holding_beta"].sum().reset_index(name="fund_beta")
    if fund_size_col and fund_size_col in holdings.columns:
        fund_size = holdings[[date_col, fund_col, fund_size_col]].drop_duplicates()
        fund_beta = fund_beta.merge(fund_size, on=[date_col, fund_col], how="left")
        total_size = fund_beta.groupby(date_col)[fund_size_col].transform("sum")
        fund_beta["_fund_weight"] = fund_beta[fund_size_col] / total_size.replace(0.0, np.nan)
    else:
        fund_beta["_fund_weight"] = 1.0 / fund_beta.groupby(date_col)[fund_col].transform("count")
    fund_beta["_weighted_beta"] = fund_beta["fund_beta"] * fund_beta["_fund_weight"]
    agg = fund_beta.groupby(date_col)["_weighted_beta"].sum().reset_index(name="aggregate_fund_beta")
    agg["delta_aggregate_fund_beta"] = agg["aggregate_fund_beta"].diff()
    return agg


def conditional_sort_returns(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    return_col: str,
    state_col: str,
    buckets: int = 5,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Sort assets by score and summarize returns conditional on state."""
    rows = []
    clean = df.dropna(subset=[date_col, score_col, return_col, state_col]).copy()
    for (date, state), group in clean.groupby([date_col, state_col]):
        if group.shape[0] < max(min_assets, buckets):
            continue
        try:
            bucket = pd.qcut(group[score_col].rank(method="first"), buckets, labels=False) + 1
        except ValueError:
            continue
        temp = group.assign(bucket=bucket.astype(int))
        grouped = temp.groupby("bucket")[return_col].mean()
        for bucket_id, value in grouped.items():
            rows.append({"date": date, "state": state, "bucket": int(bucket_id), "return": float(value)})
        if 1 in grouped.index and buckets in grouped.index:
            rows.append({"date": date, "state": state, "bucket": "long_short", "return": float(grouped.loc[buckets] - grouped.loc[1])})
    return pd.DataFrame(rows)


def belief_adjustment_index(
    prior_belief: pd.Series | np.ndarray,
    signal: pd.Series | np.ndarray,
    adjustment_speed: float = 0.5,
) -> pd.Series:
    """Simple subjective-belief updating index."""
    if not 0 <= adjustment_speed <= 1:
        raise ValueError("adjustment_speed must be between 0 and 1.")
    prior = pd.Series(prior_belief, dtype="float64")
    news = pd.Series(signal, dtype="float64")
    return (1.0 - adjustment_speed) * prior + adjustment_speed * news


def sentiment_research_checklist() -> pd.DataFrame:
    rows = [
        ("availability", "Timestamp comments, surveys, search, or flow data to the tradable date."),
        ("definition", "Separate mood, bullish/bearish support, attention, disagreement, and fund flow."),
        ("asymmetry", "Test whether bullish and bearish signals have asymmetric effects."),
        ("lag", "Use distributed lags; do not regress future returns on same-day unavailable sentiment."),
        ("volatility", "Test both return prediction and volatility amplification."),
        ("herding", "Measure agreement/dispersion, not just average sentiment."),
        ("controls", "Control market return, volatility, turnover, valuation, and macro state."),
        ("out_of_sample", "Validate by market regime and post-publication period."),
    ]
    return pd.DataFrame(rows, columns=["check", "question"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.checklist:
        result = sentiment_research_checklist()
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            print(result)


if __name__ == "__main__":
    main()
