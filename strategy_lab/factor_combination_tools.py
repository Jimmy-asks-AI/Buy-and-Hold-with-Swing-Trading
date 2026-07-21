#!/usr/bin/env python
"""Factor combination, time-series rank, and technical alpha helpers.

The functions here convert the Huatai factor-combination, historical percentile,
and price-volume divergence reports into reusable research code. Inputs are
assumed to be research panels, not live trading data feeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def zscore_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    suffix: str = "_z",
) -> pd.DataFrame:
    """Cross-sectionally z-score factor columns by date."""
    out = df.copy()
    for col in factor_cols:
        target = f"{col}{suffix}"

        def zscore(s: pd.Series) -> pd.Series:
            values = pd.to_numeric(s, errors="coerce")
            std = values.std(ddof=1)
            if pd.isna(std) or std == 0:
                return values * 0
            return (values - values.mean()) / std

        out[target] = out.groupby(date_col)[col].transform(zscore)
    return out


def half_life_weights(length: int, half_life: float) -> np.ndarray:
    """Return normalized recency weights where weight halves every H periods."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if half_life <= 0:
        raise ValueError("half_life must be positive.")
    t = np.arange(1, length + 1, dtype=float)
    raw = 2.0 ** ((t - length) / float(half_life))
    return raw / raw.sum()


def normalize_weight_vector(weights: pd.Series, long_only: bool = True) -> pd.Series:
    """Normalize weights to sum to one; optionally clip negative weights."""
    out = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if long_only:
        out = out.clip(lower=0.0)
        total = out.sum()
        if total <= EPS:
            return pd.Series(1.0 / len(out), index=out.index) if len(out) else out
        return out / total
    total = out.abs().sum()
    if total <= EPS:
        return pd.Series(1.0 / len(out), index=out.index) if len(out) else out
    return out / total


def pivot_metric_table(
    metric_df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    metric_col: str,
) -> pd.DataFrame:
    """Convert long metric table to date x factor matrix."""
    out = metric_df.pivot_table(index=date_col, columns=factor_col, values=metric_col, aggfunc="mean")
    return out.sort_index()


def rolling_metric_weights(
    metric_df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    metric_col: str = "rank_ic",
    lookback: int = 12,
    half_life: float | None = None,
    long_only: bool = True,
    min_history: int = 3,
) -> pd.DataFrame:
    """Create next-period factor weights from rolling historical metric means.

    Use this for historical Rank IC or factor-return weighting. Current-period
    metrics are never used for current-period weights.
    """
    matrix = pivot_metric_table(metric_df, date_col, factor_col, metric_col)
    rows: list[dict[str, object]] = []
    for pos, date in enumerate(matrix.index):
        history = matrix.iloc[max(0, pos - lookback) : pos].dropna(how="all")
        if history.shape[0] < min_history:
            raw = pd.Series(1.0, index=matrix.columns)
        elif half_life is None:
            raw = history.mean(skipna=True)
        else:
            w = half_life_weights(history.shape[0], half_life)
            raw = history.mul(w, axis=0).sum(axis=0) / history.notna().mul(w, axis=0).sum(axis=0)
        normalized = normalize_weight_vector(raw, long_only=long_only)
        for factor, weight in normalized.items():
            rows.append({"date": date, "factor": factor, "weight": float(weight), "method": f"rolling_{metric_col}"})
    return pd.DataFrame(rows)


def shrink_covariance_to_identity(cov: pd.DataFrame, shrinkage: float = 0.2) -> pd.DataFrame:
    """Shrink covariance matrix toward scaled identity."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1.")
    values = cov.to_numpy(dtype=float)
    diag_mean = float(np.nanmean(np.diag(values))) if values.size else 1.0
    target = np.eye(values.shape[0]) * diag_mean
    shrunk = (1.0 - shrinkage) * values + shrinkage * target
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def rolling_icir_weights(
    ic_df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    ic_col: str = "rank_ic",
    lookback: int = 12,
    shrinkage: float = 0.2,
    long_only: bool = True,
    min_history: int = 6,
) -> pd.DataFrame:
    """Approximate max-ICIR weights using rolling IC mean and covariance."""
    matrix = pivot_metric_table(ic_df, date_col, factor_col, ic_col)
    rows: list[dict[str, object]] = []
    for pos, date in enumerate(matrix.index):
        history = matrix.iloc[max(0, pos - lookback) : pos].dropna(how="all")
        if history.shape[0] < min_history:
            raw = pd.Series(1.0, index=matrix.columns)
        else:
            filled = history.apply(lambda s: s.fillna(s.mean()), axis=0).fillna(0.0)
            mu = filled.mean(axis=0)
            cov = shrink_covariance_to_identity(filled.cov(), shrinkage=shrinkage)
            raw_values = np.linalg.pinv(cov.to_numpy(dtype=float)) @ mu.to_numpy(dtype=float)
            raw = pd.Series(raw_values, index=matrix.columns)
        normalized = normalize_weight_vector(raw, long_only=long_only)
        for factor, weight in normalized.items():
            rows.append({"date": date, "factor": factor, "weight": float(weight), "method": "rolling_icir"})
    return pd.DataFrame(rows)


def combine_factors_with_weights(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    weights_df: pd.DataFrame | None = None,
    output_col: str = "composite_factor",
    zscore_inputs: bool = True,
) -> pd.DataFrame:
    """Combine factors by date using explicit weights or equal weights."""
    factors = list(factor_cols)
    out = df.copy()
    used_cols = factors
    if zscore_inputs:
        out = zscore_by_date(out, date_col, factors)
        used_cols = [f"{col}_z" for col in factors]

    if weights_df is None:
        out[output_col] = out[used_cols].mean(axis=1)
        return out

    required = {"date", "factor", "weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing columns: {sorted(missing)}")
    out[output_col] = np.nan
    col_map = dict(zip(factors, used_cols, strict=True))
    weight_map = {
        date: group.set_index("factor")["weight"].reindex(factors).fillna(0.0)
        for date, group in weights_df.groupby("date")
    }
    for date, group in out.groupby(date_col):
        weights = normalize_weight_vector(weight_map.get(date, pd.Series(1.0, index=factors)))
        values = group[[col_map[f] for f in factors]].to_numpy(dtype=float)
        out.loc[group.index, output_col] = values @ weights.to_numpy(dtype=float)
    return out


def pca_first_component_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    output_col: str = "pca1_factor",
    zscore_inputs: bool = True,
) -> pd.DataFrame:
    """Create first-principal-component composite by date."""
    factors = list(factor_cols)
    out = df.copy()
    used_cols = factors
    if zscore_inputs:
        out = zscore_by_date(out, date_col, factors)
        used_cols = [f"{col}_z" for col in factors]
    out[output_col] = np.nan
    for _, group in out.groupby(date_col):
        data = group[used_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if data.shape[0] <= len(used_cols):
            continue
        cov = data.cov().to_numpy(dtype=float)
        eigval, eigvec = np.linalg.eigh(cov)
        loading = eigvec[:, int(np.argmax(eigval))]
        if loading.sum() < 0:
            loading = -loading
        scores = out.loc[data.index, used_cols].to_numpy(dtype=float) @ loading
        out.loc[data.index, output_col] = scores
    return out


def ts_rank_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    value_col: str,
    lookback: int,
    output_col: str | None = None,
    pct: bool = True,
) -> pd.DataFrame:
    """Rank current value within an asset's trailing history including today."""
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    out = df.sort_values([asset_col, date_col]).copy()
    target = output_col or f"ts_rank_{value_col}_{lookback}"

    def trailing_rank(values: np.ndarray) -> float:
        s = pd.Series(values)
        return float(s.rank(pct=pct).iloc[-1])

    out[target] = (
        out.groupby(asset_col)[value_col]
        .transform(lambda s: s.rolling(lookback, min_periods=1).apply(trailing_rank, raw=True))
        .astype(float)
    )
    return out


def cross_section_rank(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    output_col: str | None = None,
    pct: bool = True,
) -> pd.DataFrame:
    """Rank values cross-sectionally by date."""
    out = df.copy()
    target = output_col or f"{value_col}_rank"
    out[target] = out.groupby(date_col)[value_col].rank(method="average", pct=pct)
    return out


def rolling_corr_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    left_col: str,
    right_col: str,
    window: int,
    output_col: str,
) -> pd.DataFrame:
    """Asset-level rolling correlation."""
    out = df.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        out.loc[group.index, output_col] = group[left_col].rolling(window).corr(group[right_col])
    return out


def rolling_cov_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    left_col: str,
    right_col: str,
    window: int,
    output_col: str,
) -> pd.DataFrame:
    """Asset-level rolling covariance."""
    out = df.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        out.loc[group.index, output_col] = group[left_col].rolling(window).cov(group[right_col])
    return out


def rolling_op_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    value_col: str,
    window: int,
    op: str,
    output_col: str,
) -> pd.DataFrame:
    """Asset-level rolling min/max/sum."""
    out = df.sort_values([asset_col, date_col]).copy()
    if op == "min":
        values = out.groupby(asset_col)[value_col].transform(lambda s: s.rolling(window).min())
    elif op == "max":
        values = out.groupby(asset_col)[value_col].transform(lambda s: s.rolling(window).max())
    elif op == "sum":
        values = out.groupby(asset_col)[value_col].transform(lambda s: s.rolling(window).sum())
    else:
        raise ValueError("op must be min, max, or sum.")
    out[output_col] = values
    return out


def selected_alpha101_formulas() -> pd.DataFrame:
    """Return the seven Huatai-selected price-volume divergence formulas."""
    rows = [
        ("Alpha3", "(-1 * correlation(rank(OPEN), rank(VOLUME), 10))", 400, "monthly"),
        ("Alpha13", "(-1 * rank(covariance(rank(CLOSE), rank(VOLUME), 5)))", 80, "monthly_or_biweekly"),
        ("Alpha15", "(-1 * sum(rank(correlation(rank(HIGH), rank(VOLUME), 3)), 3))", 80, "monthly"),
        ("Alpha16", "(-1 * rank(covariance(rank(HIGH), rank(VOLUME), 5)))", 40, "monthly"),
        ("Alpha44", "(-1 * correlation(HIGH, rank(VOLUME), 5))", 200, "monthly_or_biweekly"),
        ("Alpha50", "(-1 * ts_max(rank(correlation(rank(VOLUME), rank(VWAP), 5)), 5))", 80, "monthly"),
        ("Alpha55", "(-1 * correlation(rank((CLOSE - ts_min(LOW, 12)) / (ts_max(HIGH, 12) - ts_min(LOW, 12))), rank(VOLUME), 6))", 400, "monthly"),
    ]
    return pd.DataFrame(rows, columns=["alpha", "formula", "suggested_stock_count", "rebalance_hint"])


def compute_selected_alpha101(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    alpha: str,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    output_col: str | None = None,
) -> pd.DataFrame:
    """Compute one of Alpha3/13/15/16/44/50/55.

    These factors are already signed so higher values are better, consistent
    with the Huatai interpretation of price-volume divergence.
    """
    alpha = alpha.upper()
    out = df.sort_values([asset_col, date_col]).copy()
    target = output_col or alpha
    temp_cols: list[str] = []

    def rank(col: str, name: str) -> None:
        nonlocal out
        out = cross_section_rank(out, date_col, col, output_col=name)
        temp_cols.append(name)

    if alpha == "ALPHA3":
        rank(open_col, "_rank_open")
        rank(volume_col, "_rank_volume")
        out = rolling_corr_by_asset(out, date_col, asset_col, "_rank_open", "_rank_volume", 10, target)
        out[target] = -out[target]
    elif alpha == "ALPHA13":
        rank(close_col, "_rank_close")
        rank(volume_col, "_rank_volume")
        out = rolling_cov_by_asset(out, date_col, asset_col, "_rank_close", "_rank_volume", 5, "_cov")
        out = cross_section_rank(out, date_col, "_cov", output_col=target)
        out[target] = -out[target]
        temp_cols.append("_cov")
    elif alpha == "ALPHA15":
        rank(high_col, "_rank_high")
        rank(volume_col, "_rank_volume")
        out = rolling_corr_by_asset(out, date_col, asset_col, "_rank_high", "_rank_volume", 3, "_corr")
        out = cross_section_rank(out, date_col, "_corr", output_col="_corr_rank")
        out = rolling_op_by_asset(out, date_col, asset_col, "_corr_rank", 3, "sum", target)
        out[target] = -out[target]
        temp_cols.extend(["_corr", "_corr_rank"])
    elif alpha == "ALPHA16":
        rank(high_col, "_rank_high")
        rank(volume_col, "_rank_volume")
        out = rolling_cov_by_asset(out, date_col, asset_col, "_rank_high", "_rank_volume", 5, "_cov")
        out = cross_section_rank(out, date_col, "_cov", output_col=target)
        out[target] = -out[target]
        temp_cols.append("_cov")
    elif alpha == "ALPHA44":
        rank(volume_col, "_rank_volume")
        out = rolling_corr_by_asset(out, date_col, asset_col, high_col, "_rank_volume", 5, target)
        out[target] = -out[target]
    elif alpha == "ALPHA50":
        rank(volume_col, "_rank_volume")
        rank(vwap_col, "_rank_vwap")
        out = rolling_corr_by_asset(out, date_col, asset_col, "_rank_volume", "_rank_vwap", 5, "_corr")
        out = cross_section_rank(out, date_col, "_corr", output_col="_corr_rank")
        out = rolling_op_by_asset(out, date_col, asset_col, "_corr_rank", 5, "max", target)
        out[target] = -out[target]
        temp_cols.extend(["_corr", "_corr_rank", "_rank_vwap"])
    elif alpha == "ALPHA55":
        out = rolling_op_by_asset(out, date_col, asset_col, low_col, 12, "min", "_low_min")
        out = rolling_op_by_asset(out, date_col, asset_col, high_col, 12, "max", "_high_max")
        denom = (out["_high_max"] - out["_low_min"]).replace(0.0, np.nan)
        out["_price_position"] = (out[close_col] - out["_low_min"]) / denom
        out = cross_section_rank(out, date_col, "_price_position", output_col="_rank_position")
        rank(volume_col, "_rank_volume")
        out = rolling_corr_by_asset(out, date_col, asset_col, "_rank_position", "_rank_volume", 6, target)
        out[target] = -out[target]
        temp_cols.extend(["_low_min", "_high_max", "_price_position", "_rank_position"])
    else:
        raise ValueError("alpha must be one of Alpha3, Alpha13, Alpha15, Alpha16, Alpha44, Alpha50, Alpha55.")

    return out.drop(columns=[col for col in temp_cols if col in out.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--mode", choices=["ts_rank", "alpha101"], required=True)
    parser.add_argument("--value-col")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--alpha")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    if args.mode == "ts_rank":
        if not args.value_col or not args.lookback:
            raise ValueError("--value-col and --lookback are required for ts_rank mode.")
        result = ts_rank_by_asset(df, args.date_col, args.asset_col, args.value_col, args.lookback)
    else:
        if not args.alpha:
            raise ValueError("--alpha is required for alpha101 mode.")
        result = compute_selected_alpha101(df, args.date_col, args.asset_col, args.alpha)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
