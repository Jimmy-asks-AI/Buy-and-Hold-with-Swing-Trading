#!/usr/bin/env python
"""Network, balance-sheet stability, price-amplitude, and passive-flow factors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _power_centrality(adjacency: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> pd.Series:
    values = adjacency.to_numpy(dtype=float)
    n = values.shape[0]
    if n == 0:
        return pd.Series(dtype="float64")
    vec = np.ones(n, dtype=float) / np.sqrt(n)
    for _ in range(max_iter):
        nxt = values @ vec
        norm = np.linalg.norm(nxt)
        if norm <= EPS:
            return pd.Series(0.0, index=adjacency.index)
        nxt = nxt / norm
        if np.max(np.abs(nxt - vec)) < tol:
            vec = nxt
            break
        vec = nxt
    return pd.Series(vec, index=adjacency.index)


def correlation_network_factor_snapshot(
    return_window: pd.DataFrame,
    spillover_returns: pd.Series | None = None,
    edge_quantile: float = 0.90,
    min_abs_corr: float | None = None,
    output_prefix: str = "corr_net",
) -> pd.DataFrame:
    """Build one correlation-network factor snapshot from wide asset returns.

    Degree and centrality are positive signals in the report. Short-horizon
    neighbor-return spillover is usually reversal-like, so downstream users
    should often use `-neighbor_return` for alpha.
    """
    if not 0.0 < edge_quantile < 1.0:
        raise ValueError("edge_quantile must be between 0 and 1.")
    returns = return_window.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    corr = returns.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_corr = corr.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool)).stack()
    if min_abs_corr is None:
        threshold = float(upper.quantile(edge_quantile)) if not upper.empty else 1.0
    else:
        threshold = float(min_abs_corr)
    adjacency_values = (abs_corr >= threshold).astype(float).to_numpy(copy=True)
    np.fill_diagonal(adjacency_values, 0.0)
    adjacency = pd.DataFrame(adjacency_values, index=corr.index, columns=corr.columns)
    degree = adjacency.sum(axis=1)
    centrality = _power_centrality(adjacency)
    out = pd.DataFrame(
        {
            "asset": adjacency.index,
            f"{output_prefix}_degree": degree.reindex(adjacency.index).to_numpy(dtype=float),
            f"{output_prefix}_centrality": centrality.reindex(adjacency.index).to_numpy(dtype=float),
        }
    )
    if spillover_returns is not None:
        neighbor = adjacency @ pd.to_numeric(spillover_returns, errors="coerce").reindex(adjacency.columns).fillna(0.0)
        out[f"{output_prefix}_neighbor_return"] = _safe_divide(neighbor, degree.replace(0.0, np.nan)).to_numpy(dtype=float)
        out[f"{output_prefix}_short_reversal_alpha"] = -out[f"{output_prefix}_neighbor_return"]
    out[f"{output_prefix}_edge_threshold"] = threshold
    return out


def business_similarity_network_features(
    revenue_rows: pd.DataFrame,
    asset_col: str,
    category_col: str,
    weight_col: str | None = None,
    min_similarity: float = 0.0,
    output_prefix: str = "business_net",
) -> pd.DataFrame:
    """Create product/business-similarity network features from category rows."""
    data = revenue_rows[[asset_col, category_col] + ([weight_col] if weight_col else [])].dropna().copy()
    if weight_col:
        data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce").fillna(0.0)
        matrix = data.pivot_table(index=asset_col, columns=category_col, values=weight_col, aggfunc="sum", fill_value=0.0)
    else:
        data["_presence"] = 1.0
        matrix = data.pivot_table(index=asset_col, columns=category_col, values="_presence", aggfunc="max", fill_value=0.0)
    norm = np.linalg.norm(matrix.to_numpy(dtype=float), axis=1)
    norm = np.where(norm <= EPS, 1.0, norm)
    normalized = matrix.to_numpy(dtype=float) / norm[:, None]
    similarity = normalized @ normalized.T
    adjacency = (similarity > min_similarity).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    adj = pd.DataFrame(adjacency, index=matrix.index, columns=matrix.index)
    degree = adj.sum(axis=1)
    return pd.DataFrame(
        {
            "asset": matrix.index,
            f"{output_prefix}_degree": degree.to_numpy(dtype=float),
            f"{output_prefix}_centrality": _power_centrality(adj).to_numpy(dtype=float),
        }
    )


def asset_growth_volatility_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    asset_value_col: str,
    growth_periods: int = 4,
    volatility_window: int = 4,
    shift: int = 1,
    output_col: str = "asset_growth_volatility",
) -> pd.DataFrame:
    """Rolling volatility of asset growth; lower volatility is the alpha direction."""
    if growth_periods <= 0 or volatility_window <= 1:
        raise ValueError("growth_periods must be positive and volatility_window must be greater than 1.")
    out = df.sort_values([asset_col, date_col]).copy()
    value = pd.to_numeric(out[asset_value_col], errors="coerce")
    growth = value / value.groupby(out[asset_col]).shift(growth_periods).replace(0.0, np.nan) - 1.0
    out[f"{output_col}_growth"] = growth
    vol = growth.groupby(out[asset_col]).transform(
        lambda s: s.rolling(volatility_window, min_periods=max(3, volatility_window // 2 + 1)).std()
    )
    out[output_col] = vol.groupby(out[asset_col]).shift(shift) if shift else vol
    out[f"{output_col}_alpha"] = -out[output_col]
    return out


def industry_asset_stability_factor(
    df: pd.DataFrame,
    date_col: str,
    industry_col: str,
    volatility_col: str,
    output_col: str = "industry_asset_stability",
) -> pd.DataFrame:
    """Industry-level median asset-growth stability factor."""
    data = df[[date_col, industry_col, volatility_col]].copy()
    grouped = (
        data.groupby([date_col, industry_col], dropna=False)[volatility_col]
        .median()
        .rename("industry_asset_growth_volatility")
        .reset_index()
    )
    grouped[output_col] = -pd.to_numeric(grouped["industry_asset_growth_volatility"], errors="coerce")
    return grouped


def capital_structure_change_factors(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    total_asset_col: str,
    equity_col: str,
    debt_col: str,
    long_debt_col: str | None = None,
    fixed_asset_col: str | None = None,
    periods: int = 4,
    prefix: str = "capstruct",
) -> pd.DataFrame:
    """Compute changes in equity ratio, debt ratio, long debt ratio, and equity/fixed assets."""
    out = df.sort_values([asset_col, date_col]).copy()
    total_assets = pd.to_numeric(out[total_asset_col], errors="coerce")
    ratios = {
        f"{prefix}_equity_asset": _safe_divide(out[equity_col], total_assets),
        f"{prefix}_debt_asset": _safe_divide(out[debt_col], total_assets),
    }
    if long_debt_col:
        ratios[f"{prefix}_long_debt_asset"] = _safe_divide(out[long_debt_col], total_assets)
    if fixed_asset_col:
        ratios[f"{prefix}_equity_fixed_asset"] = _safe_divide(out[equity_col], out[fixed_asset_col])
    for name, ratio in ratios.items():
        out[name] = ratio
        out[f"{name}_change"] = ratio - ratio.groupby(out[asset_col]).shift(periods)
    out[f"{prefix}_leverage_change_alpha"] = out[f"{prefix}_debt_asset_change"]
    out[f"{prefix}_equity_ratio_change_alpha"] = -out[f"{prefix}_equity_asset_change"]
    return out


def price_amplitude_factor(
    bars: pd.DataFrame,
    asset_col: str,
    date_col: str,
    high_col: str,
    low_col: str,
    window: int = 42,
    shift: int = 1,
    output_col: str = "price_amplitude",
) -> pd.DataFrame:
    """Rolling high/low price amplitude; positive after price-volume neutralization."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    out = bars.sort_values([asset_col, date_col]).copy()
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    rolling_high = high.groupby(out[asset_col]).transform(lambda s: s.rolling(window, min_periods=window // 2).max())
    rolling_low = low.groupby(out[asset_col]).transform(lambda s: s.rolling(window, min_periods=window // 2).min())
    amplitude = rolling_high / rolling_low.replace(0.0, np.nan) - 1.0
    out[output_col] = amplitude.groupby(out[asset_col]).shift(shift) if shift else amplitude
    return out


def turnover_amplitude_factor(
    bars: pd.DataFrame,
    asset_col: str,
    date_col: str,
    turnover_col: str,
    window: int = 21,
    shift: int = 1,
    output_col: str = "turnover_amplitude",
) -> pd.DataFrame:
    """Rolling max-min turnover amplitude; lower is generally better."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    out = bars.sort_values([asset_col, date_col]).copy()
    turnover = pd.to_numeric(out[turnover_col], errors="coerce")
    rolling_max = turnover.groupby(out[asset_col]).transform(lambda s: s.rolling(window, min_periods=window // 2).max())
    rolling_min = turnover.groupby(out[asset_col]).transform(lambda s: s.rolling(window, min_periods=window // 2).min())
    amplitude = rolling_max - rolling_min
    out[output_col] = amplitude.groupby(out[asset_col]).shift(shift) if shift else amplitude
    out[f"{output_col}_alpha"] = -out[output_col]
    return out


def orthogonalize_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    control_cols: Sequence[str],
    output_col: str | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Cross-sectionally residualize one factor by controls within each date."""
    controls = list(control_cols)
    min_count = min_count or len(controls) + 8
    target = output_col or f"{value_col}_residual"
    out = df.copy()
    residuals = []
    for _, group in out.groupby(date_col, sort=True):
        work = group[[value_col, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
        series = pd.Series(np.nan, index=group.index, dtype="float64")
        if work.shape[0] >= min_count:
            x = np.column_stack([np.ones(work.shape[0]), work[controls].to_numpy(dtype=float)])
            y = work[value_col].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            series.loc[work.index] = y - x @ beta
        residuals.append(series)
    out[target] = pd.concat(residuals).sort_index()
    return out


def passive_expansion_regime(
    passive_series: pd.DataFrame,
    date_col: str,
    passive_share_col: str,
    lookback: int = 2,
    threshold: float | None = None,
    quantile: float = 0.75,
    output_col: str = "passive_regime",
) -> pd.DataFrame:
    """Classify passive-product expansion regimes from passive share changes."""
    out = passive_series.sort_values(date_col).copy()
    share = pd.to_numeric(out[passive_share_col], errors="coerce")
    out["passive_share_change"] = share - share.shift(lookback)
    cutoff = threshold if threshold is not None else float(out["passive_share_change"].dropna().quantile(quantile))
    out[output_col] = np.where(out["passive_share_change"] >= cutoff, "expansion", "stable")
    out["passive_expansion_cutoff"] = cutoff
    return out


def alpha_performance_by_passive_regime(
    alpha_returns: pd.DataFrame,
    passive_regimes: pd.DataFrame,
    date_col: str,
    alpha_cols: Iterable[str],
    regime_col: str = "passive_regime",
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Summarize alpha returns in passive expansion versus stable regimes."""
    cols = list(alpha_cols)
    data = pd.merge(alpha_returns[[date_col, *cols]], passive_regimes[[date_col, regime_col]], on=date_col, how="inner")
    rows: list[dict[str, object]] = []
    for regime, group in data.groupby(regime_col, dropna=False, sort=True):
        for col in cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            vol = values.std(ddof=1)
            rows.append(
                {
                    "regime": regime,
                    "alpha": col,
                    "n_periods": int(values.shape[0]),
                    "annualized_return": float(values.mean() * periods_per_year) if not values.empty else np.nan,
                    "annualized_vol": float(vol * np.sqrt(periods_per_year)) if vol > EPS else np.nan,
                    "information_ratio": float(values.mean() / vol * np.sqrt(periods_per_year)) if vol > EPS else np.nan,
                    "positive_rate": float((values > 0).mean()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def passive_regime_factor_weight_adjustment(
    weights: pd.DataFrame,
    date_col: str,
    factor_col: str,
    weight_col: str,
    passive_regimes: pd.DataFrame,
    reversal_factors: Iterable[str],
    regime_col: str = "passive_regime",
    expansion_multiplier: float = 0.5,
    output_col: str = "adjusted_weight",
) -> pd.DataFrame:
    """Reduce reversal-like factor weights during passive expansion regimes."""
    out = pd.merge(weights, passive_regimes[[date_col, regime_col]], on=date_col, how="left")
    reversal = set(reversal_factors)
    raw = pd.to_numeric(out[weight_col], errors="coerce")
    reduce = out[factor_col].isin(reversal) & (out[regime_col] == "expansion")
    out[output_col] = np.where(reduce, raw * float(expansion_multiplier), raw)
    return out


def round45_factor_checklist() -> pd.DataFrame:
    rows = [
        ("network", "Correlation-network degree and centrality need neutralization against volatility, turnover, size, and industry."),
        ("spillover", "Short-horizon network spillover is reversal-like; long-horizon spillover can become momentum-like."),
        ("asset_stability", "Asset-growth volatility should use recent available quarters and is stronger for current assets."),
        ("capital_structure", "Leverage increase is not universally good; validate by industry and profitability state."),
        ("amplitude", "Price amplitude should be residualized by return, turnover, volatility, liquidity, and size."),
        ("decay", "Price-volume amplitude factors decay quickly; avoid long holding periods without decay tests."),
        ("passive", "Passive expansion is a regime variable, not proof of causality; compare alpha before/after with confounders."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
