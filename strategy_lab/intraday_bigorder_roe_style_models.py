#!/usr/bin/env python
"""Intraday microstructure, large-order, enhanced ROE, and style-cluster tools."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return values * 0.0
    return (values - values.mean()) / std


def intraday_segment_label(
    df: pd.DataFrame,
    time_col: str,
    output_col: str = "intraday_segment",
) -> pd.DataFrame:
    """Label A-share intraday records as open30, close30, or middle."""
    out = df.copy()
    timestamp = pd.to_datetime(out[time_col], errors="coerce")
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    open_end = 10 * 60
    close_start = 14 * 60 + 30
    out[output_col] = np.select(
        [minutes < open_end, minutes >= close_start],
        ["open30", "close30"],
        default="middle",
    )
    out.loc[timestamp.isna(), output_col] = np.nan
    return out


def intraday_microstructure_summary(
    bars: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    amount_col: str,
    return_col: str | None = None,
    spread_col: str | None = None,
    big_order_amount_col: str | None = None,
) -> pd.DataFrame:
    """Summarize U-shaped volume and L-shaped volatility/spread by segment."""
    data = intraday_segment_label(bars, time_col)
    amount = pd.to_numeric(data[amount_col], errors="coerce")
    data["_amount"] = amount
    if big_order_amount_col:
        data["_big_amount"] = pd.to_numeric(data[big_order_amount_col], errors="coerce")
    rows: list[dict[str, object]] = []
    for (date, asset), group in data.groupby([date_col, asset_col], sort=True):
        total_amount = group["_amount"].sum()
        total_big = group["_big_amount"].sum() if big_order_amount_col else np.nan
        for segment, seg in group.groupby("intraday_segment", dropna=False):
            row = {
                "date": date,
                "asset": asset,
                "segment": segment,
                "amount_share": float(seg["_amount"].sum() / total_amount) if total_amount > EPS else np.nan,
            }
            if return_col:
                row["return_volatility"] = float(pd.to_numeric(seg[return_col], errors="coerce").std(ddof=1))
            if spread_col:
                row["avg_spread"] = float(pd.to_numeric(seg[spread_col], errors="coerce").mean())
            if big_order_amount_col:
                row["big_order_amount_share"] = float(seg["_big_amount"].sum() / total_big) if total_big > EPS else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_intraday_factor(
    bars: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    value_col: str,
    segments: Iterable[str] | None = None,
    exclude_segments: Iterable[str] | None = None,
    agg: str = "mean",
    output_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate a high-frequency field over selected intraday segments."""
    data = intraday_segment_label(bars, time_col)
    if segments is not None:
        data = data[data["intraday_segment"].isin(set(segments))]
    if exclude_segments is not None:
        data = data[~data["intraday_segment"].isin(set(exclude_segments))]
    target = output_col or f"{value_col}_{agg}"
    values = pd.to_numeric(data[value_col], errors="coerce")
    data = data.assign(_value=values)
    grouped = data.groupby([date_col, asset_col])["_value"]
    if agg == "mean":
        out = grouped.mean()
    elif agg == "sum":
        out = grouped.sum()
    elif agg == "std":
        out = grouped.std()
    elif agg == "skew":
        out = grouped.skew()
    else:
        raise ValueError("agg must be one of mean, sum, std, skew.")
    return out.rename(target).reset_index().rename(columns={date_col: "date", asset_col: "asset"})


def intraday_factor_window_policy() -> pd.DataFrame:
    rows = [
        ("informed_trading", "open30", "Net order imbalance, buy-intention, and large-order net-buy factors."),
        ("overreaction", "exclude_open30", "High-frequency skewness, downside volatility share, and overreaction factors."),
        ("liquidity_pressure", "close30_or_full_day", "Close-related execution pressure should be tested separately from informed trading."),
    ]
    return pd.DataFrame(rows, columns=["factor_family", "preferred_window", "rationale"])


def _side_masks(side: pd.Series) -> tuple[pd.Series, pd.Series]:
    if pd.api.types.is_numeric_dtype(side):
        numeric = pd.to_numeric(side, errors="coerce")
        return numeric > 0, numeric < 0
    text = side.astype(str).str.upper()
    buy = text.isin({"B", "BUY", "1", "TRUE", "买", "买入", "主动买入"})
    sell = text.isin({"S", "SELL", "-1", "卖", "卖出", "主动卖出"})
    return buy, sell


def refined_large_order_factors(
    orders: pd.DataFrame,
    date_col: str,
    asset_col: str,
    amount_col: str,
    side_col: str,
    time_col: str | None = None,
    open_only: bool = False,
    threshold_lookback: int = 20,
    factor_window: int = 20,
    std_multiplier: float = 1.0,
    absolute_threshold: float | None = None,
    min_history_orders: int = 50,
) -> pd.DataFrame:
    """Build refined large-order net-buy factors from reconstructed order data.

    Thresholds use trailing log single-order amount distributions. The factor
    window then aggregates large buy and sell amounts into ratios and strengths.
    """
    data = orders.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    if time_col and open_only:
        data = intraday_segment_label(data, time_col)
        data = data[data["intraday_segment"] == "open30"].copy()
    data["_amount"] = pd.to_numeric(data[amount_col], errors="coerce").clip(lower=0.0)
    buy_mask, sell_mask = _side_masks(data[side_col])
    data["_buy"] = buy_mask
    data["_sell"] = sell_mask
    data["_large"] = False
    data["_threshold"] = np.nan

    for _, group in data.sort_values([asset_col, date_col]).groupby(asset_col, sort=False):
        dates = pd.Index(sorted(group[date_col].dropna().unique()))
        for pos, date in enumerate(dates):
            hist_dates = set(dates[max(0, pos - threshold_lookback) : pos])
            hist = group[group[date_col].isin(hist_dates)]["_amount"].dropna()
            if hist.shape[0] < min_history_orders:
                continue
            logged = np.log(hist.replace(0.0, np.nan)).dropna()
            if logged.shape[0] < min_history_orders:
                continue
            threshold = float(np.exp(logged.mean() + std_multiplier * logged.std(ddof=1)))
            if absolute_threshold is not None:
                threshold = max(threshold, float(absolute_threshold))
            idx = group.index[group[date_col] == date]
            data.loc[idx, "_threshold"] = threshold
            data.loc[idx, "_large"] = data.loc[idx, "_amount"] >= threshold

    daily = (
        data.assign(
            large_buy_amount=np.where(data["_large"] & data["_buy"], data["_amount"], 0.0),
            large_sell_amount=np.where(data["_large"] & data["_sell"], data["_amount"], 0.0),
        )
        .groupby([asset_col, date_col], sort=True)
        .agg(
            total_amount=("_amount", "sum"),
            large_buy_amount=("large_buy_amount", "sum"),
            large_sell_amount=("large_sell_amount", "sum"),
            threshold=("_threshold", "mean"),
            n_orders=("_amount", "size"),
            n_large=("_large", "sum"),
        )
        .reset_index()
    )
    daily["large_net_buy_amount"] = daily["large_buy_amount"] - daily["large_sell_amount"]
    daily = daily.sort_values([asset_col, date_col])
    for col in ["large_buy_amount", "large_sell_amount", "large_net_buy_amount", "total_amount"]:
        daily[f"{col}_sum"] = daily.groupby(asset_col)[col].transform(
            lambda s: s.rolling(factor_window, min_periods=max(3, factor_window // 3)).sum()
        )
    net = daily["large_net_buy_amount_sum"]
    buy = daily["large_buy_amount_sum"]
    total = daily["total_amount_sum"].replace(0.0, np.nan)
    daily["large_buy_ratio"] = buy / total
    daily["large_net_buy_ratio"] = net / total
    rolling_mean = daily.groupby(asset_col)["large_net_buy_amount"].transform(
        lambda s: s.rolling(factor_window, min_periods=max(3, factor_window // 3)).mean()
    )
    rolling_std = daily.groupby(asset_col)["large_net_buy_amount"].transform(
        lambda s: s.rolling(factor_window, min_periods=max(3, factor_window // 3)).std()
    )
    daily["large_net_buy_strength"] = rolling_mean / rolling_std.replace(0.0, np.nan)
    return daily.rename(columns={asset_col: "asset", date_col: "date"})


def rolling_current_roe_prediction(
    df: pd.DataFrame,
    date_col: str,
    disclosed_roe_col: str,
    consensus_roe_col: str,
    realized_roe_col: str,
    window: int = 12,
    min_obs: int = 50,
    output_col: str = "predicted_current_roe",
) -> pd.DataFrame:
    """Predict current true ROE from latest disclosed ROE and consensus ROE."""
    out = df.sort_values(date_col).copy()
    out[output_col] = pd.to_numeric(out[disclosed_roe_col], errors="coerce")
    dates = pd.Index(sorted(out[date_col].dropna().unique()))
    features = [disclosed_roe_col, consensus_roe_col]
    for pos, date in enumerate(dates):
        hist_dates = set(dates[max(0, pos - window) : pos])
        hist = out[out[date_col].isin(hist_dates)][[realized_roe_col, *features]].apply(pd.to_numeric, errors="coerce").dropna()
        if hist.shape[0] < min_obs:
            continue
        x = np.column_stack([np.ones(hist.shape[0]), hist[features].to_numpy(dtype=float)])
        y = hist[realized_roe_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        current = out[out[date_col] == date][features].apply(pd.to_numeric, errors="coerce")
        valid = current.dropna().index
        if len(valid):
            x_current = np.column_stack([np.ones(len(valid)), current.loc[valid, features].to_numpy(dtype=float)])
            out.loc[valid, output_col] = x_current @ beta
    return out


def roe_volatility_weight(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    roe_col: str,
    window: int = 4,
    shift: int = 1,
    output_col: str = "roe_vol_weight",
) -> pd.DataFrame:
    """Compute inverse-volatility confidence weight for ROE signals."""
    out = df.sort_values([asset_col, date_col]).copy()
    roe = pd.to_numeric(out[roe_col], errors="coerce")
    vol = roe.groupby(out[asset_col]).transform(lambda s: s.rolling(window, min_periods=max(3, window // 2)).std())
    vol = vol.groupby(out[asset_col]).shift(shift) if shift else vol
    inv = 1.0 / vol.replace(0.0, np.nan)
    mean_inv = inv.groupby(out[date_col]).transform("mean")
    out[output_col] = (inv / mean_inv.replace(0.0, np.nan)).clip(lower=0.0, upper=3.0)
    out[f"{output_col}_volatility"] = vol
    return out


def volatility_adjusted_roe_factor(
    df: pd.DataFrame,
    date_col: str,
    roe_col: str,
    weight_col: str,
    output_col: str = "vol_adjusted_roe",
    shift_positive: bool = False,
) -> pd.DataFrame:
    """Shrink noisy high-volatility ROE signals toward the cross-sectional mean."""
    out = df.copy()
    roe = pd.to_numeric(out[roe_col], errors="coerce")
    weight = pd.to_numeric(out[weight_col], errors="coerce").fillna(1.0)
    mean = roe.groupby(out[date_col]).transform("mean")
    adjusted = (roe - mean) * weight
    if shift_positive:
        min_by_date = adjusted.groupby(out[date_col]).transform("min")
        adjusted = adjusted - min_by_date
    out[output_col] = adjusted
    return out


def kmeans_style_classification(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    feature_cols: Sequence[str],
    n_clusters: int = 30,
    initial_group_col: str | None = None,
    max_iter: int = 100,
    output_col: str = "style_cluster",
) -> pd.DataFrame:
    """Classify stocks by style features with dependency-free K-means."""
    features = list(feature_cols)
    out = df.copy()
    out[output_col] = np.nan
    for date, group in out.groupby(date_col, sort=True):
        values = group[features].apply(pd.to_numeric, errors="coerce")
        values = values.apply(_zscore).fillna(0.0)
        k = min(n_clusters, max(1, values.shape[0]))
        if initial_group_col and initial_group_col in group.columns:
            init = values.groupby(group[initial_group_col]).mean()
            init = init.loc[init.index[:k]].to_numpy(dtype=float)
            if init.shape[0] < k:
                fill_idx = np.linspace(0, values.shape[0] - 1, k - init.shape[0]).astype(int)
                init = np.vstack([init, values.iloc[fill_idx].to_numpy(dtype=float)])
        else:
            fill_idx = np.linspace(0, values.shape[0] - 1, k).astype(int)
            init = values.iloc[fill_idx].to_numpy(dtype=float)
        centroids = init[:k].copy()
        labels = np.zeros(values.shape[0], dtype=int)
        x = values.to_numpy(dtype=float)
        for _ in range(max_iter):
            dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                mask = labels == cluster
                if mask.any():
                    centroids[cluster] = x[mask].mean(axis=0)
        out.loc[group.index, output_col] = labels
    return out


def category_neutralize_factor(
    df: pd.DataFrame,
    date_col: str,
    category_col: str,
    value_col: str,
    output_col: str | None = None,
    method: str = "demean",
) -> pd.DataFrame:
    """Neutralize a factor within style categories by date."""
    if method not in {"demean", "zscore"}:
        raise ValueError("method must be 'demean' or 'zscore'.")
    out = df.copy()
    target = output_col or f"{value_col}_category_neutral"
    value = pd.to_numeric(out[value_col], errors="coerce")
    grouped = value.groupby([out[date_col], out[category_col]])
    if method == "demean":
        out[target] = value - grouped.transform("mean")
    else:
        out[target] = grouped.transform(_zscore)
    return out


def style_momentum_spillover(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    category_col: str,
    return_col: str,
    lookback: int = 1,
    output_col: str = "style_momentum_spillover",
) -> pd.DataFrame:
    """Mean trailing return of peers in the same style category, excluding self."""
    out = df.sort_values([asset_col, date_col]).copy()
    ret = pd.to_numeric(out[return_col], errors="coerce")
    trailing = ret.groupby(out[asset_col]).transform(lambda s: s.shift(1).rolling(lookback, min_periods=1).sum())
    out["_trailing_return"] = trailing
    group_sum = out.groupby([date_col, category_col])["_trailing_return"].transform("sum")
    group_count = out.groupby([date_col, category_col])["_trailing_return"].transform("count")
    out[output_col] = (group_sum - out["_trailing_return"]) / (group_count - 1).replace(0.0, np.nan)
    return out.drop(columns=["_trailing_return"])


def round46_model_checklist() -> pd.DataFrame:
    rows = [
        ("intraday_window", "Match factor logic to time window: informed-trading factors favor open30; overreaction factors often exclude open30."),
        ("large_order", "Use log order-size distribution over multiple days; std threshold 0-2 is usually safer than extreme thresholds."),
        ("orthogonalization", "Large-order factors need controls for industry, size, value, turnover, reversal, volatility, and liquidity."),
        ("roe", "Predict current unavailable ROE before using it as a factor; consensus ROE alone has coverage and attention bias."),
        ("roe_volatility", "High ROE volatility means low confidence; shrink high-volatility ROE toward the cross-sectional mean."),
        ("style_cluster", "Style clustering complements industry classification but can create size exposure in top portfolios."),
        ("spillover", "Style-category momentum is related to, but not the same as, industry momentum."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
