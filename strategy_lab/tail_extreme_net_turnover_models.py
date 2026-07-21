#!/usr/bin/env python
"""Tail dependence, extreme-group factor screening, and net-turnover tools."""

from __future__ import annotations

from collections.abc import Sequence

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


def _quantile_group(values: pd.Series, n_groups: int) -> pd.Series:
    ranks = pd.to_numeric(values, errors="coerce").rank(pct=True, method="first")
    groups = np.ceil(ranks * n_groups).clip(1, n_groups)
    return groups.astype("Int64")


def empirical_tail_dependence(
    x: pd.Series,
    y: pd.Series,
    tail: str = "lower",
    q: float = 0.05,
) -> float:
    """Conditional probability that x is extreme when y is extreme."""
    if tail not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'.")
    if not 0 < q < 0.5:
        raise ValueError("q must be between 0 and 0.5.")
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if data.empty:
        return np.nan
    if tail == "lower":
        x_extreme = data["x"] <= data["x"].quantile(q)
        y_extreme = data["y"] <= data["y"].quantile(q)
    else:
        x_extreme = data["x"] >= data["x"].quantile(1.0 - q)
        y_extreme = data["y"] >= data["y"].quantile(1.0 - q)
    denom = int(y_extreme.sum())
    return float((x_extreme & y_extreme).sum() / denom) if denom else np.nan


def hill_tail_index(
    values: pd.Series,
    tail: str = "upper",
    k: int | None = None,
    q: float = 0.05,
) -> float:
    """Hill estimator for positive upper tails or lower loss tails."""
    if tail not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'.")
    series = pd.to_numeric(values, errors="coerce").dropna()
    if tail == "lower":
        series = -series
    series = series[series > 0].sort_values(ascending=False)
    if series.shape[0] < 5:
        return np.nan
    if k is None:
        k = max(3, int(np.ceil(series.shape[0] * q)))
    k = min(max(2, k), series.shape[0] - 1)
    tail_values = series.iloc[:k].to_numpy(dtype=float)
    threshold = float(series.iloc[k])
    if threshold <= EPS:
        return np.nan
    denom = np.log(tail_values / threshold).sum()
    return float(k / denom) if denom > EPS else np.nan


def rolling_tail_dependence_by_asset(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    asset_return_col: str,
    market_return_col: str,
    window: int = 252,
    min_periods: int | None = None,
    q: float = 0.05,
    output_prefix: str = "tail_dep",
) -> pd.DataFrame:
    """Rolling upper/lower tail dependence of each asset versus market."""
    if window <= 5:
        raise ValueError("window must be greater than 5.")
    min_periods = min_periods or max(30, window // 3)
    _require_columns(df, [asset_col, date_col, asset_return_col, market_return_col])
    out = df.sort_values([asset_col, date_col]).copy()
    out[f"{output_prefix}_lower"] = np.nan
    out[f"{output_prefix}_upper"] = np.nan
    out[f"{output_prefix}_hill_lower"] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        asset_ret = pd.to_numeric(group[asset_return_col], errors="coerce").reset_index(drop=True)
        market_ret = pd.to_numeric(group[market_return_col], errors="coerce").reset_index(drop=True)
        lower_scores = []
        upper_scores = []
        hill_scores = []
        for pos in range(len(group)):
            start = max(0, pos - window + 1)
            x = asset_ret.iloc[start : pos + 1]
            y = market_ret.iloc[start : pos + 1]
            if pd.concat([x, y], axis=1).dropna().shape[0] < min_periods:
                lower_scores.append(np.nan)
                upper_scores.append(np.nan)
                hill_scores.append(np.nan)
                continue
            lower_scores.append(empirical_tail_dependence(x, y, "lower", q))
            upper_scores.append(empirical_tail_dependence(x, y, "upper", q))
            hill_scores.append(hill_tail_index(x, "lower", q=q))
        out.loc[group.index, f"{output_prefix}_lower"] = lower_scores
        out.loc[group.index, f"{output_prefix}_upper"] = upper_scores
        out.loc[group.index, f"{output_prefix}_hill_lower"] = hill_scores
    return out


def beta_adjusted_tail_risk(
    df: pd.DataFrame,
    date_col: str,
    lower_tail_col: str,
    beta_col: str,
    output_col: str = "beta_adjusted_tail_risk",
) -> pd.DataFrame:
    """Residual lower-tail risk after removing cross-sectional beta exposure."""
    _require_columns(df, [date_col, lower_tail_col, beta_col])
    out = df.copy()
    out[output_col] = np.nan
    for _, group in out.groupby(date_col, sort=True):
        work = group[[lower_tail_col, beta_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if work.shape[0] < 10:
            continue
        x = np.column_stack([np.ones(work.shape[0]), work[beta_col].to_numpy(dtype=float)])
        y = work[lower_tail_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        out.loc[work.index, output_col] = y - x @ beta
    return out


def factor_group_strength_index(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    forward_return_col: str,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Frequency-based factor strength from grouped future returns."""
    _require_columns(df, [date_col, factor_col, forward_return_col])
    winner_counts = pd.Series(0, index=np.arange(1, n_groups + 1), dtype=float)
    loser_counts = pd.Series(0, index=np.arange(1, n_groups + 1), dtype=float)
    avg_returns = pd.Series(0.0, index=np.arange(1, n_groups + 1), dtype=float)
    periods = 0
    for _, group in df[[date_col, factor_col, forward_return_col]].dropna().groupby(date_col):
        if group.shape[0] < n_groups * 2:
            continue
        labels = _quantile_group(group[factor_col], n_groups)
        means = group.assign(_group=labels).groupby("_group", observed=True)[forward_return_col].mean()
        if means.shape[0] < n_groups:
            continue
        winner_counts.loc[int(means.idxmax())] += 1
        loser_counts.loc[int(means.idxmin())] += 1
        avg_returns = avg_returns.add(means.reindex(avg_returns.index).fillna(0.0), fill_value=0.0)
        periods += 1
    if periods == 0:
        return pd.DataFrame(
            [{"factor": factor_col, "periods": 0, "winner_strength": np.nan, "loser_strength": np.nan, "avg_return_slope": np.nan}]
        )
    index = pd.Series(np.arange(1, n_groups + 1), index=winner_counts.index, dtype=float)
    avg_returns = avg_returns / periods
    rows = {
        "factor": factor_col,
        "periods": periods,
        "winner_strength": float(winner_counts.corr(index)),
        "loser_strength": float(loser_counts.corr(index)),
        "avg_return_slope": float(avg_returns.corr(index)),
        "top_winner_frequency": float(winner_counts.loc[n_groups] / periods),
        "bottom_winner_frequency": float(winner_counts.loc[1] / periods),
    }
    return pd.DataFrame([rows])


def factor_return_tail_probability(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    forward_return_col: str,
    direction: int = 1,
    q: float = 0.1,
) -> float:
    """Probability that the preferred factor tail also lands in the top return tail."""
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1.")
    _require_columns(df, [date_col, factor_col, forward_return_col])
    hits = 0
    total = 0
    for _, group in df[[date_col, factor_col, forward_return_col]].dropna().groupby(date_col):
        if group.shape[0] < 20:
            continue
        factor_rank = group[factor_col].rank(pct=True, method="first")
        return_rank = group[forward_return_col].rank(pct=True, method="first")
        factor_tail = factor_rank >= 1 - q if direction > 0 else factor_rank <= q
        total += int(factor_tail.sum())
        hits += int((factor_tail & (return_rank >= 1 - q)).sum())
    return float(hits / total) if total else np.nan


def lee_ready_trade_sign(
    trades: pd.DataFrame,
    asset_col: str,
    date_col: str,
    time_col: str,
    price_col: str,
    bid_col: str,
    ask_col: str,
    output_col: str = "trade_sign",
) -> pd.DataFrame:
    """Classify trades as active buy/sell using Lee-Ready logic."""
    _require_columns(trades, [asset_col, date_col, time_col, price_col, bid_col, ask_col])
    out = trades.sort_values([asset_col, date_col, time_col]).copy()
    out[output_col] = np.nan
    price = pd.to_numeric(out[price_col], errors="coerce")
    mid = (pd.to_numeric(out[bid_col], errors="coerce") + pd.to_numeric(out[ask_col], errors="coerce")) / 2.0
    quote_sign = pd.Series(np.nan, index=out.index, dtype="float64")
    quote_sign.loc[price > mid] = 1.0
    quote_sign.loc[price < mid] = -1.0
    out["_quote_sign"] = quote_sign
    for _, group in out.groupby([asset_col, date_col], sort=False):
        p = pd.to_numeric(group[price_col], errors="coerce")
        tick = np.sign(p.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
        sign = group["_quote_sign"].fillna(tick)
        out.loc[group.index, output_col] = sign
    return out.drop(columns=["_quote_sign"])


def net_turnover_rate(
    signed_trades: pd.DataFrame,
    asset_col: str,
    date_col: str,
    sign_col: str,
    volume_col: str,
    free_float_col: str,
    output_col: str = "net_turnover",
) -> pd.DataFrame:
    """Daily net turnover: active buy volume minus active sell volume over float."""
    _require_columns(signed_trades, [asset_col, date_col, sign_col, volume_col, free_float_col])
    work = signed_trades.copy()
    work["_signed_volume"] = pd.to_numeric(work[sign_col], errors="coerce") * pd.to_numeric(work[volume_col], errors="coerce")
    grouped = (
        work.groupby([date_col, asset_col], as_index=False)
        .agg(signed_volume=("_signed_volume", "sum"), total_volume=(volume_col, "sum"), free_float=(free_float_col, "last"))
    )
    grouped[output_col] = grouped["signed_volume"] / pd.to_numeric(grouped["free_float"], errors="coerce").replace(0.0, np.nan)
    grouped["turnover"] = pd.to_numeric(grouped["total_volume"], errors="coerce") / pd.to_numeric(grouped["free_float"], errors="coerce").replace(0.0, np.nan)
    return grouped


def rolling_net_turnover_signal(
    daily_net_turnover: pd.DataFrame,
    asset_col: str,
    date_col: str,
    net_turnover_col: str,
    lookback: int = 15,
    output_col: str = "avg_net_turnover",
) -> pd.DataFrame:
    """Average daily net turnover over a recent observation window."""
    _require_columns(daily_net_turnover, [asset_col, date_col, net_turnover_col])
    out = daily_net_turnover.sort_values([asset_col, date_col]).copy()
    out[output_col] = (
        pd.to_numeric(out[net_turnover_col], errors="coerce")
        .groupby(out[asset_col])
        .transform(lambda s: s.rolling(lookback, min_periods=max(2, lookback // 3)).mean())
    )
    return out


def extreme_factor_screen(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    forward_return_col: str,
    lookback_dates: int = 24,
    n_groups: int = 5,
    min_abs_spread: float = 0.008,
    min_win_rate: float = 2 / 3,
    min_extreme_rate: float = 2 / 3,
) -> pd.DataFrame:
    """Rolling screen for factors whose extreme groups drive return extremes."""
    factors = list(factor_cols)
    _require_columns(df, [date_col, forward_return_col, *factors])
    out_rows: list[dict[str, object]] = []
    data = df.sort_values(date_col).copy()
    dates = pd.Index(sorted(data[date_col].dropna().unique()))
    for pos, date in enumerate(dates):
        hist_dates = dates[max(0, pos - lookback_dates) : pos]
        hist = data[data[date_col].isin(hist_dates)]
        if len(hist_dates) < max(6, lookback_dates // 2):
            continue
        for factor in factors:
            spreads = []
            extreme_hits = []
            for _, group in hist[[date_col, factor, forward_return_col]].dropna().groupby(date_col):
                if group.shape[0] < n_groups * 2:
                    continue
                labels = _quantile_group(group[factor], n_groups)
                means = group.assign(_group=labels).groupby("_group", observed=True)[forward_return_col].mean()
                if means.shape[0] < n_groups:
                    continue
                high = float(means.loc[n_groups])
                low = float(means.loc[1])
                spread = high - low
                direction = 1 if spread >= 0 else -1
                preferred_group = n_groups if direction > 0 else 1
                extreme_hits.append(bool(int(means.idxmax()) == preferred_group))
                spreads.append(spread)
            if not spreads:
                continue
            avg_spread = float(np.nanmean(spreads))
            direction = 1 if avg_spread >= 0 else -1
            signed_spreads = np.asarray(spreads, dtype=float) * direction
            win_rate = float(np.nanmean(signed_spreads > 0))
            extreme_rate = float(np.nanmean(extreme_hits))
            out_rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "direction": direction,
                    "avg_spread": avg_spread,
                    "abs_avg_spread": abs(avg_spread),
                    "win_rate": win_rate,
                    "extreme_rate": extreme_rate,
                    "selected": bool(abs(avg_spread) >= min_abs_spread and win_rate >= min_win_rate and extreme_rate >= min_extreme_rate),
                    "n_periods": len(spreads),
                }
            )
    return pd.DataFrame(out_rows)


def extreme_factor_composite_score(
    df: pd.DataFrame,
    screen: pd.DataFrame,
    date_col: str,
    asset_col: str,
    output_col: str = "extreme_factor_score",
) -> pd.DataFrame:
    """Equal-weight composite using factors selected by the extreme-group screen."""
    _require_columns(df, [date_col, asset_col])
    _require_columns(screen, ["date", "factor", "direction", "selected"])
    out = df.copy()
    out[output_col] = np.nan
    for date, selected in screen[screen["selected"]].groupby("date"):
        factors = selected["factor"].tolist()
        dirs = selected.set_index("factor")["direction"].astype(float)
        available = [factor for factor in factors if factor in out.columns]
        if not available:
            continue
        idx = out.index[out[date_col] == date]
        if len(idx) == 0:
            continue
        parts = []
        for factor in available:
            parts.append(out.loc[idx, factor].groupby(out.loc[idx, date_col]).transform(_zscore) * dirs.loc[factor])
        out.loc[idx, output_col] = pd.concat(parts, axis=1).mean(axis=1)
    return out


def selected_return_percentile_distribution(
    df: pd.DataFrame,
    date_col: str,
    selected_col: str,
    forward_return_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Distribution of selected names' future-return percentiles."""
    _require_columns(df, [date_col, selected_col, forward_return_col])
    rows = []
    for date, group in df[[date_col, selected_col, forward_return_col]].dropna().groupby(date_col):
        if group.shape[0] < n_bins * 2:
            continue
        percentile = group[forward_return_col].rank(pct=True, method="first")
        bins = np.ceil(percentile * n_bins).clip(1, n_bins).astype(int)
        selected = group[selected_col].astype(bool)
        for bucket in range(1, n_bins + 1):
            rows.append(
                {
                    "date": date,
                    "bucket": bucket,
                    "selected_count": int(((bins == bucket) & selected).sum()),
                    "all_count": int((bins == bucket).sum()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    summary = result.groupby("bucket", as_index=False)[["selected_count", "all_count"]].sum()
    summary["selected_share"] = summary["selected_count"] / summary["selected_count"].sum()
    summary["baseline_share"] = summary["all_count"] / summary["all_count"].sum()
    return summary


def round49_research_checklist() -> pd.DataFrame:
    rows = [
        ("tail_dependence", "Report lower-tail dependence separately from beta; check tail sample size and q sensitivity."),
        ("factor_strength", "Linear IC can be small while extreme groups work; inspect grouped return shape."),
        ("net_turnover", "Trade-sign rules need limit-up/down correction and exchange-specific quote fields."),
        ("extreme_screen", "Use rolling windows only; require spread, extreme-location, and win-rate gates."),
        ("micro_distribution", "Check whether alpha comes from broad hit rate or a few extreme winners."),
    ]
    return pd.DataFrame(rows, columns=["topic", "validation_gate"])
