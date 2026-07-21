#!/usr/bin/env python
"""Industry-specific factor selection and quality-factor helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-12


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return values * 0.0
    return (values - values.mean()) / std


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _cross_sectional_regression_stats(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    factor_cols: list[str],
    min_count: int | None = None,
) -> dict[str, object]:
    min_count = min_count or len(factor_cols) + 8
    betas: dict[str, list[float]] = {col: [] for col in factor_cols}
    r2_values: list[float] = []
    adj_r2_values: list[float] = []
    for _, group in panel[[date_col, return_col, *factor_cols]].dropna().groupby(date_col, sort=True):
        if group.shape[0] < min_count:
            continue
        x = group[factor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[return_col], errors="coerce").to_numpy(dtype=float)
        x = np.column_stack([np.ones(group.shape[0]), x])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        y_hat = x @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot if ss_tot > EPS else np.nan
        n = group.shape[0]
        p = len(factor_cols)
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1) if not pd.isna(r2) else np.nan
        r2_values.append(r2)
        adj_r2_values.append(adj_r2)
        for col, value in zip(factor_cols, beta[1:]):
            betas[col].append(float(value))

    rows = {}
    for col, values in betas.items():
        series = pd.Series(values, dtype="float64").dropna()
        std = series.std(ddof=1)
        rows[col] = {
            "premium_mean": float(series.mean()) if not series.empty else np.nan,
            "premium_t": float(series.mean() / (std / np.sqrt(series.shape[0]))) if series.shape[0] > 1 and std > EPS else np.nan,
            "premium_positive_rate": float((series > 0).mean()) if not series.empty else np.nan,
            "n_periods": int(series.shape[0]),
        }
    return {
        "factor_stats": rows,
        "avg_r2": float(pd.Series(r2_values).mean()),
        "avg_adj_r2": float(pd.Series(adj_r2_values).mean()),
        "n_periods": int(pd.Series(r2_values).dropna().shape[0]),
    }


def industry_stepwise_factor_selection(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    candidate_cols: Iterable[str],
    industry_col: str | None = None,
    target_industry: object | None = None,
    max_factors: int | None = None,
    t_threshold: float = 2.0,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Forward-select industry factors by Fama-MacBeth significance and adjusted R2.

    This implements the pharmaceutical-industry report's logic: add the factor
    that gives the largest adjusted-R2 improvement while keeping all selected
    factor premia statistically meaningful.
    """
    candidates = list(candidate_cols)
    data = panel.copy()
    if industry_col is not None and target_industry is not None:
        data = data[data[industry_col] == target_industry].copy()
    max_factors = max_factors or len(candidates)
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    prev_adj_r2 = -np.inf

    while candidates and len(selected) < max_factors:
        best: dict[str, object] | None = None
        for col in candidates:
            trial = selected + [col]
            stats = _cross_sectional_regression_stats(data, date_col, return_col, trial, min_count=min_count)
            factor_stats = stats["factor_stats"]
            if stats["n_periods"] < 6:
                continue
            t_values = [abs(factor_stats[name]["premium_t"]) for name in trial]
            if any(pd.isna(t) or t < t_threshold for t in t_values):
                continue
            improvement = float(stats["avg_adj_r2"] - prev_adj_r2) if np.isfinite(prev_adj_r2) else float(stats["avg_adj_r2"])
            row = {
                "candidate": col,
                "avg_adj_r2": float(stats["avg_adj_r2"]),
                "avg_r2": float(stats["avg_r2"]),
                "adj_r2_improvement": improvement,
                "new_factor_t": float(factor_stats[col]["premium_t"]),
                "n_periods": int(stats["n_periods"]),
            }
            if best is None or row["avg_adj_r2"] > best["avg_adj_r2"]:
                best = row
        if best is None:
            break
        selected.append(str(best["candidate"]))
        candidates.remove(str(best["candidate"]))
        prev_adj_r2 = float(best["avg_adj_r2"])
        best["step"] = len(selected)
        best["selected_factors"] = ",".join(selected)
        rows.append(best)
    return pd.DataFrame(rows)


def industry_prediction_score(
    df: pd.DataFrame,
    date_col: str,
    factor_directions: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    output_col: str = "industry_expected_return_score",
) -> pd.DataFrame:
    """Create an industry-specific composite score from selected factors."""
    out = df.copy()
    factors = list(factor_directions.keys())
    weights = weights or {col: 1.0 for col in factors}
    total_weight = sum(abs(float(weights.get(col, 0.0))) for col in factors)
    if total_weight <= EPS:
        raise ValueError("weights cannot all be zero.")
    score = pd.Series(0.0, index=out.index)
    for col in factors:
        z = out.groupby(date_col)[col].transform(_zscore)
        score = score + z * float(factor_directions[col]) * float(weights.get(col, 0.0)) / total_weight
    out[output_col] = score
    return out


def net_share_issuance_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    share_count_col: str,
    periods: int = 12,
    output_col: str = "net_share_issuance",
) -> pd.DataFrame:
    """Log change in share count; lower issuance is generally better."""
    out = df.sort_values([asset_col, date_col]).copy()
    shares = pd.to_numeric(out[share_count_col], errors="coerce").clip(lower=0.0)
    out[output_col] = np.log(shares.replace(0.0, np.nan)) - np.log(
        shares.groupby(out[asset_col]).shift(periods).replace(0.0, np.nan)
    )
    return out


def asset_growth_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    total_asset_col: str,
    periods: int = 4,
    output_col: str = "asset_growth",
) -> pd.DataFrame:
    """Total-asset growth; direction should be tested by size bucket."""
    out = df.sort_values([asset_col, date_col]).copy()
    value = pd.to_numeric(out[total_asset_col], errors="coerce")
    out[output_col] = value / value.groupby(out[asset_col]).shift(periods).replace(0.0, np.nan) - 1.0
    return out


def leverage_change_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    debt_col: str,
    asset_total_col: str,
    periods: int = 4,
    output_col: str = "debt_asset_change",
) -> pd.DataFrame:
    """Change in debt/assets; positive direction is conditional on profitability."""
    out = df.sort_values([asset_col, date_col]).copy()
    debt_asset = _safe_divide(out[debt_col], out[asset_total_col])
    out["_debt_asset"] = debt_asset
    out[output_col] = debt_asset - debt_asset.groupby(out[asset_col]).shift(periods)
    return out.drop(columns=["_debt_asset"])


def quality_composite_score(
    df: pd.DataFrame,
    date_col: str,
    component_directions: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    output_col: str = "quality_score",
    size_bucket_col: str | None = None,
    investment_cols: Iterable[str] | None = None,
    small_bucket_values: Iterable[object] = ("small",),
) -> pd.DataFrame:
    """Build a flexible quality score from profitability, growth, safety, and issuance.

    Investment factors can have opposite directions across size buckets, so this
    helper supports flipping specified investment columns for small-cap buckets.
    """
    out = df.copy()
    directions = dict(component_directions)
    if investment_cols and size_bucket_col:
        small_values = set(small_bucket_values)
        for col in investment_cols:
            adjusted_col = f"{col}_size_adjusted"
            sign = np.where(out[size_bucket_col].isin(small_values), 1.0, -1.0)
            out[adjusted_col] = pd.to_numeric(out[col], errors="coerce") * sign
            directions[adjusted_col] = abs(float(directions.get(col, 1.0)))
            directions.pop(col, None)
    return industry_prediction_score(out, date_col, directions, weights=weights, output_col=output_col)


def quality_factor_component_map() -> pd.DataFrame:
    """Common quality components and default A-share caveats."""
    rows = [
        ("profitability", "ROE, ROA, operating profitability, gross margin", 1, "core; use latest available quarterly data"),
        ("growth", "earnings or ROE growth and surprises", 1, "works better when tied to future fundamentals"),
        ("stability", "ROE volatility, earnings volatility", -1, "often weak alone; better as risk control"),
        ("investment", "asset or book-equity growth", "size_conditional", "positive in small caps, negative in large caps"),
        ("net_issuance", "log share-count growth", -1, "monthly useful, quarterly holding can decay"),
        ("leverage_level", "debt/assets", 0, "level is weak in A-shares"),
        ("leverage_change", "change in debt/assets", 1, "stronger among high-profitability firms"),
    ]
    return pd.DataFrame(rows, columns=["component", "definition", "default_direction", "caveat"])


def industry_quality_checklist() -> pd.DataFrame:
    rows = [
        ("industry_sample", "Industry-specific models require enough stocks and stable industry classification."),
        ("redundancy", "Use stepwise or penalized selection; many quality and growth factors are highly correlated."),
        ("direction", "Direction must be set at the industry and size-bucket level, not copied from full-market tests."),
        ("availability", "Use announcement or database availability date for all fundamentals and expectations."),
        ("capacity", "Report turnover, stock-count limits, industry liquidity, and benchmark deviation."),
        ("failure", "Track years when selected factor premia reverse, especially in narrow industry universes."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
