#!/usr/bin/env python
"""Portfolio-level helpers for short-horizon high-frequency signals.

The tools in this module are meant to test how short-lived signals enter a
lower-frequency portfolio. They do not assume that raising rebalance frequency
is free. Typical uses:

- combine several high-frequency factors into one cross-sectional signal;
- delay part of scheduled buys/sells when the short-horizon signal disagrees;
- remove or downweight the short leg after portfolio optimization;
- measure signal decay across holding horizons.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def zscore_columns_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    suffix: str = "_z",
    min_count: int = 3,
) -> pd.DataFrame:
    """Cross-sectionally z-score several columns within each date."""
    out = df.copy()
    for col in factor_cols:
        target = f"{col}{suffix}"

        def zscore(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if valid.shape[0] < min_count:
                return pd.Series(np.nan, index=s.index, dtype="float64")
            std = valid.std(ddof=1)
            if pd.isna(std) or abs(std) <= EPS:
                return pd.Series(0.0, index=s.index, dtype="float64")
            return (s - valid.mean()) / std

        out[target] = out.groupby(date_col)[col].transform(zscore)
    return out


def composite_signal(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    weights: Mapping[str, float] | None = None,
    factor_signs: Mapping[str, float] | None = None,
    output_col: str = "hf_composite_signal",
    min_count: int = 3,
) -> pd.DataFrame:
    """Build a weighted composite from cross-sectional z-scored factors.

    Higher output is assumed to be better. Use ``factor_signs`` to flip factors
    whose natural direction is negative.
    """
    factors = _as_list(factor_cols)
    out = zscore_columns_by_date(df, date_col, factors, min_count=min_count)
    if weights is None:
        weights = {col: 1.0 for col in factors}
    if factor_signs is None:
        factor_signs = {}

    used_cols: list[str] = []
    weighted = []
    for col in factors:
        z_col = f"{col}_z"
        sign = float(factor_signs.get(col, 1.0))
        weight = float(weights.get(col, 0.0))
        if abs(weight) <= EPS:
            continue
        used_cols.append(z_col)
        weighted.append(out[z_col] * sign * weight)
    if not weighted:
        raise ValueError("No non-zero factor weights were provided.")

    denom = sum(abs(float(weights.get(col, 0.0))) for col in factors)
    out[output_col] = sum(weighted) / denom
    out[f"{output_col}_n_factors"] = out[used_cols].notna().sum(axis=1)
    return out


def _select_with_budget(
    candidates: pd.DataFrame,
    trade_col: str,
    budget: float,
    allow_partial: bool,
) -> pd.Series:
    """Select candidate trade amounts up to an absolute trade budget."""
    selected = pd.Series(0.0, index=candidates.index)
    remaining = float(max(budget, 0.0))
    if remaining <= EPS or candidates.empty:
        return selected

    for idx, value in candidates[trade_col].items():
        amount = abs(float(value))
        if amount <= EPS:
            continue
        if amount <= remaining + EPS:
            selected.loc[idx] = float(value)
            remaining -= amount
        elif allow_partial:
            selected.loc[idx] = float(np.sign(value) * remaining)
            remaining = 0.0
        if remaining <= EPS:
            break
    return selected


def build_rebalance_delay_plan(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    current_weight_col: str,
    target_weight_col: str,
    signal_col: str,
    delay_ratio: float = 0.05,
    budget_base: str = "gross_weight",
    allow_partial: bool = True,
) -> pd.DataFrame:
    """Delay part of scheduled buys and sells using a short-horizon signal.

    Rule:

    - planned sells with the highest signal are delayed;
    - planned buys with the lowest signal are delayed;
    - delayed buy and sell notional is balanced so the portfolio remains funded.

    ``delay_ratio`` is interpreted as a fraction of total gross portfolio weight
    by default. Set ``budget_base='one_way_turnover'`` to use scheduled turnover
    as the budget base.
    """
    required = [date_col, asset_col, current_weight_col, target_weight_col, signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not 0 <= delay_ratio <= 1:
        raise ValueError("delay_ratio must be between 0 and 1.")
    if budget_base not in {"gross_weight", "one_way_turnover"}:
        raise ValueError("budget_base must be 'gross_weight' or 'one_way_turnover'.")

    frames: list[pd.DataFrame] = []
    clean = df.copy()
    clean[current_weight_col] = clean[current_weight_col].astype(float).fillna(0.0)
    clean[target_weight_col] = clean[target_weight_col].astype(float).fillna(0.0)
    clean["planned_trade"] = clean[target_weight_col] - clean[current_weight_col]

    for date, group in clean.groupby(date_col, sort=True):
        group = group.copy()
        gross_weight = group[current_weight_col].abs().sum()
        one_way_turnover = group["planned_trade"].abs().sum() / 2.0
        base = gross_weight if budget_base == "gross_weight" else one_way_turnover
        raw_budget = float(delay_ratio * base)

        sells = group[group["planned_trade"] < -EPS].sort_values(signal_col, ascending=False)
        buys = group[group["planned_trade"] > EPS].sort_values(signal_col, ascending=True)
        sell_capacity = sells["planned_trade"].abs().sum()
        buy_capacity = buys["planned_trade"].abs().sum()
        budget = min(raw_budget, float(sell_capacity), float(buy_capacity))

        delayed = pd.Series(0.0, index=group.index)
        delayed.loc[sells.index] = _select_with_budget(sells, "planned_trade", budget, allow_partial)
        delayed.loc[buys.index] = _select_with_budget(buys, "planned_trade", budget, allow_partial)

        group["delayed_trade"] = delayed
        group["immediate_trade"] = group["planned_trade"] - group["delayed_trade"]
        group["immediate_target_weight"] = group[current_weight_col] + group["immediate_trade"]
        group["delay_side"] = np.where(
            group["delayed_trade"] < -EPS,
            "sell_later",
            np.where(group["delayed_trade"] > EPS, "buy_later", "none"),
        )
        group["delay_budget_used"] = group["delayed_trade"].abs().sum() / 2.0
        group["delay_budget_requested"] = raw_budget
        group["delay_budget_base"] = base
        group["delay_budget_date"] = date
        frames.append(group)

    if not frames:
        return pd.DataFrame(columns=[*required, "planned_trade", "delayed_trade", "immediate_trade"])
    return pd.concat(frames, ignore_index=True)


def short_dummy_filter(
    df: pd.DataFrame,
    date_col: str,
    signal_col: str,
    lower_q: float = 0.05,
    high_is_good: bool = True,
    output_col: str = "hf_short_dummy",
    min_assets: int = 20,
) -> pd.DataFrame:
    """Flag the adverse tail of a short-horizon signal within each date."""
    if not 0 < lower_q < 0.5:
        raise ValueError("lower_q must be between 0 and 0.5.")
    out = df.copy()
    out[output_col] = False

    for _, group in out.groupby(date_col, sort=True):
        signal = group[signal_col].dropna()
        if signal.shape[0] < min_assets:
            continue
        q = signal.quantile(lower_q if high_is_good else 1 - lower_q)
        if high_is_good:
            out.loc[group.index, output_col] = group[signal_col] <= q
        else:
            out.loc[group.index, output_col] = group[signal_col] >= q
    out[output_col] = out[output_col].fillna(False).astype(bool)
    return out


def apply_post_trade_short_filter(
    df: pd.DataFrame,
    date_col: str,
    weight_col: str,
    short_flag_col: str,
    output_col: str = "filtered_weight",
    redistribute: bool = True,
) -> pd.DataFrame:
    """Remove flagged names from final weights and optionally redistribute."""
    out = df.copy()
    out[weight_col] = out[weight_col].astype(float).fillna(0.0)
    out[short_flag_col] = out[short_flag_col].fillna(False).astype(bool)
    out[output_col] = out[weight_col]
    out.loc[out[short_flag_col], output_col] = 0.0

    if not redistribute:
        return out

    frames: list[pd.DataFrame] = []
    for _, group in out.groupby(date_col, sort=True):
        group = group.copy()
        original_sum = group[weight_col].sum()
        kept_sum = group[output_col].sum()
        if abs(kept_sum) <= EPS:
            group[output_col] = 0.0
        else:
            group[output_col] = group[output_col] * (original_sum / kept_sum)
        frames.append(group)

    if not frames:
        return out
    return pd.concat(frames, ignore_index=True)


def horizon_rank_ic_table(
    df: pd.DataFrame,
    date_col: str,
    signal_col: str,
    return_cols: Mapping[int, str],
    min_assets: int = 20,
) -> pd.DataFrame:
    """Compute IC and rank IC for several forward-return horizons."""
    rows: list[dict[str, object]] = []
    for horizon, ret_col in return_cols.items():
        clean = df.dropna(subset=[signal_col, ret_col])
        for date, group in clean.groupby(date_col, sort=True):
            if group.shape[0] < min_assets:
                continue
            signal = group[signal_col]
            ret = group[ret_col]
            rows.append(
                {
                    "date": date,
                    "horizon": int(horizon),
                    "n_assets": int(group.shape[0]),
                    "ic": float(signal.corr(ret)),
                    "rank_ic": float(signal.rank().corr(ret.rank())),
                }
            )
    return pd.DataFrame(rows)


def horizon_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize horizon-level IC decay."""
    if ic_df.empty:
        return pd.DataFrame(columns=["horizon", "metric", "mean", "std", "ir", "win_rate", "count"])
    rows: list[dict[str, object]] = []
    for horizon, group in ic_df.groupby("horizon", sort=True):
        for metric in ["ic", "rank_ic"]:
            values = group[metric].dropna()
            if values.empty:
                continue
            std = values.std(ddof=1)
            rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(std) if not pd.isna(std) else np.nan,
                    "ir": float(values.mean() / std) if std and not pd.isna(std) else np.nan,
                    "win_rate": float((values > 0).mean()),
                    "count": int(values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def weighted_corr(x: pd.Series, y: pd.Series, weight: pd.Series) -> float:
    """Weighted Pearson correlation."""
    clean = pd.DataFrame({"x": x, "y": y, "w": weight}).dropna()
    clean = clean[clean["w"] > 0]
    if clean.shape[0] < 3:
        return float("nan")
    w = clean["w"].astype(float)
    w = w / w.sum()
    x_centered = clean["x"] - (w * clean["x"]).sum()
    y_centered = clean["y"] - (w * clean["y"]).sum()
    cov = (w * x_centered * y_centered).sum()
    x_var = (w * x_centered * x_centered).sum()
    y_var = (w * y_centered * y_centered).sum()
    if x_var <= EPS or y_var <= EPS:
        return float("nan")
    return float(cov / np.sqrt(x_var * y_var))


def long_weighted_ic_table(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    groups: int = 5,
    long_group_weight: float = 0.5,
    high_is_good: bool = True,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Compute IC after giving the long group a larger weight.

    This is useful when a factor has strong short-leg returns but weak long-leg
    returns. The aligned factor direction is always "higher is better".
    """
    if groups < 2:
        raise ValueError("groups must be at least 2.")
    if not 0 < long_group_weight < 1:
        raise ValueError("long_group_weight must be between 0 and 1.")
    rows: list[dict[str, object]] = []
    sign = 1.0 if high_is_good else -1.0
    for date, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col, sort=True):
        if group.shape[0] < max(min_assets, groups):
            continue
        work = group.copy()
        work["_aligned_factor"] = work[factor_col].astype(float) * sign
        try:
            work["_factor_group"] = (
                pd.qcut(work["_aligned_factor"].rank(method="first"), groups, labels=False) + 1
            )
        except ValueError:
            continue
        base_group_weight = (1.0 - long_group_weight) / (groups - 1)
        group_weight = pd.Series(base_group_weight, index=work.index)
        group_weight.loc[work["_factor_group"] == groups] = long_group_weight
        group_counts = work.groupby("_factor_group")["_aligned_factor"].transform("count")
        stock_weight = group_weight / group_counts
        rows.append(
            {
                "date": date,
                "n_assets": int(work.shape[0]),
                "ic": float(work["_aligned_factor"].corr(work[return_col])),
                "rank_ic": float(work["_aligned_factor"].rank().corr(work[return_col].rank())),
                "long_weighted_ic": weighted_corr(work["_aligned_factor"], work[return_col], stock_weight),
                "long_group_weight": float(long_group_weight),
                "long_group_return": float(work.loc[work["_factor_group"] == groups, return_col].mean()),
                "short_group_return": float(work.loc[work["_factor_group"] == 1, return_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def group_leg_return_table(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    groups: int = 5,
    high_is_good: bool = True,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Return equal-weight long and short leg returns by date."""
    rows: list[dict[str, object]] = []
    sign = 1.0 if high_is_good else -1.0
    for date, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col, sort=True):
        if group.shape[0] < max(min_assets, groups):
            continue
        work = group.copy()
        work["_aligned_factor"] = work[factor_col].astype(float) * sign
        try:
            work["_factor_group"] = (
                pd.qcut(work["_aligned_factor"].rank(method="first"), groups, labels=False) + 1
            )
        except ValueError:
            continue
        short_ret = work.loc[work["_factor_group"] == 1, return_col].mean()
        long_ret = work.loc[work["_factor_group"] == groups, return_col].mean()
        market_ret = work[return_col].mean()
        rows.append(
            {
                "date": date,
                "n_assets": int(work.shape[0]),
                "short_leg_return": float(short_ret),
                "long_leg_return": float(long_ret),
                "market_return": float(market_ret),
                "long_short_return": float(long_ret - short_ret),
                "long_excess": float(long_ret - market_ret),
                "short_excess": float(market_ret - short_ret),
                "long_contribution_ratio": float((long_ret - market_ret) / (long_ret - short_ret))
                if abs(long_ret - short_ret) > EPS
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def polynomial_factor_expansion(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    powers: Sequence[int] = (2, 4),
    zscore_first: bool = True,
) -> pd.DataFrame:
    """Add polynomial terms for nonlinear factor-return relationships."""
    factors = _as_list(factor_cols)
    out = df.copy()
    source_cols = factors
    if zscore_first:
        out = zscore_columns_by_date(out, date_col, factors)
        source_cols = [f"{col}_z" for col in factors]
    for raw_col, source_col in zip(factors, source_cols, strict=True):
        for power in powers:
            if power < 2:
                raise ValueError("powers must be >= 2.")
            out[f"{raw_col}_pow{power}"] = out[source_col].astype(float) ** int(power)
    return out


def rbf_factor_expansion(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    n_centers: int = 5,
    gamma: float | None = None,
    zscore_first: bool = True,
    output_prefix: str | None = None,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Expand one factor with date-wise radial basis functions.

    Centers are estimated from cross-sectional quantiles each date. This is a
    lightweight alternative to fitting a full RBF network in a research loop.
    """
    if n_centers < 2:
        raise ValueError("n_centers must be at least 2.")
    out = df.copy()
    source_col = factor_col
    if zscore_first:
        out = zscore_columns_by_date(out, date_col, [factor_col])
        source_col = f"{factor_col}_z"
    prefix = output_prefix or f"{factor_col}_rbf"
    for idx in range(n_centers):
        out[f"{prefix}_{idx + 1}"] = np.nan

    quantiles = np.linspace(0.0, 1.0, n_centers)
    for _, group in out.groupby(date_col, sort=True):
        values = group[source_col].dropna().astype(float)
        if values.shape[0] < min_assets:
            continue
        centers = values.quantile(quantiles).to_numpy(dtype=float)
        if gamma is None:
            distances = np.diff(np.unique(centers))
            scale = float(np.median(np.abs(distances))) if distances.size else 1.0
            local_gamma = 1.0 / max(2.0 * scale * scale, EPS)
        else:
            local_gamma = float(gamma)
        aligned_values = out.loc[group.index, source_col].astype(float)
        for idx, center in enumerate(centers):
            out.loc[group.index, f"{prefix}_{idx + 1}"] = np.exp(
                -local_gamma * (aligned_values - center) ** 2
            )
    return out


def multi_factor_short_flags(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    lower_q: float = 0.05,
    factor_signs: Mapping[str, float] | None = None,
    method: str = "zscore_composite",
    weights: Mapping[str, float] | None = None,
    min_memberships: int = 2,
    output_col: str = "multi_factor_short_flag",
    min_assets: int = 20,
) -> pd.DataFrame:
    """Build multi-factor short-leg flags.

    ``method='zscore_composite'`` flags the bottom tail of a composite signal.
    ``method='membership_count'`` flags names that fall into the adverse tail
    for at least ``min_memberships`` single factors.
    """
    factors = _as_list(factor_cols)
    if method not in {"zscore_composite", "membership_count"}:
        raise ValueError("method must be 'zscore_composite' or 'membership_count'.")
    signs = factor_signs or {}

    if method == "zscore_composite":
        out = composite_signal(
            df,
            date_col=date_col,
            factor_cols=factors,
            weights=weights,
            factor_signs=signs,
            output_col="_short_composite_signal",
            min_count=3,
        )
        out = short_dummy_filter(
            out,
            date_col=date_col,
            signal_col="_short_composite_signal",
            lower_q=lower_q,
            high_is_good=True,
            output_col=output_col,
            min_assets=min_assets,
        )
        return out.drop(columns=["_short_composite_signal"], errors="ignore")

    out = df.copy()
    out["_short_membership_count"] = 0
    for col in factors:
        aligned_col = f"_{col}_aligned_short_signal"
        out[aligned_col] = out[col].astype(float) * float(signs.get(col, 1.0))
        for _, group in out.groupby(date_col, sort=True):
            signal = group[aligned_col].dropna()
            if signal.shape[0] < min_assets:
                continue
            threshold = signal.quantile(lower_q)
            out.loc[group.index, "_short_membership_count"] += (
                group[aligned_col] <= threshold
            ).fillna(False).astype(int)
        out = out.drop(columns=[aligned_col])
    out[output_col] = out["_short_membership_count"] >= int(min_memberships)
    return out


def adjust_expected_return_for_short_flags(
    df: pd.DataFrame,
    date_col: str,
    expected_return_col: str,
    short_flag_col: str,
    output_col: str = "adjusted_expected_return",
    mode: str = "date_min",
    penalty: float = 0.0,
) -> pd.DataFrame:
    """Pre-exclusion helper: penalize expected returns for short-leg names."""
    if mode not in {"date_min", "penalty"}:
        raise ValueError("mode must be 'date_min' or 'penalty'.")
    out = df.copy()
    out[output_col] = out[expected_return_col].astype(float)
    flags = out[short_flag_col].fillna(False).astype(bool)
    if mode == "penalty":
        out.loc[flags, output_col] = out.loc[flags, output_col] - float(penalty)
        return out

    for _, group in out.groupby(date_col, sort=True):
        date_min = group[expected_return_col].min()
        out.loc[group.index.intersection(out.index[flags]), output_col] = date_min
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--current-weight-col")
    parser.add_argument("--target-weight-col")
    parser.add_argument("--signal-col")
    parser.add_argument("--factor-cols", nargs="*")
    parser.add_argument("--mode", choices=["composite", "delay_plan", "short_filter"], required=True)
    parser.add_argument("--delay-ratio", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    if args.mode == "composite":
        if not args.factor_cols:
            raise ValueError("--factor-cols is required for composite mode.")
        result = composite_signal(df, args.date_col, args.factor_cols)
    elif args.mode == "delay_plan":
        result = build_rebalance_delay_plan(
            df,
            date_col=args.date_col,
            asset_col=args.asset_col,
            current_weight_col=args.current_weight_col,
            target_weight_col=args.target_weight_col,
            signal_col=args.signal_col,
            delay_ratio=args.delay_ratio,
        )
    else:
        result = short_dummy_filter(df, args.date_col, args.signal_col)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
