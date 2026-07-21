#!/usr/bin/env python
"""Portfolio weighting and implementation constraint helpers.

This module is for the portfolio-construction layer after factor research.
It separates signal ranking from weight expression, turnover, capacity, and
basic constraint checks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def normalize_weights(weights: pd.Series, allow_short: bool = False) -> pd.Series:
    """Normalize weights to sum to one.

    Long-only mode clips negative values to zero. If all weights are zero, an
    equal-weight vector is returned to keep downstream reports explicit.
    """
    out = weights.astype(float).copy()
    if not allow_short:
        out = out.clip(lower=0.0)
    total = out.sum()
    if abs(total) <= EPS:
        if len(out) == 0:
            return out
        return pd.Series(1.0 / len(out), index=out.index)
    return out / total


def equal_weight(assets: pd.Index | list[str]) -> pd.Series:
    index = pd.Index(assets)
    if len(index) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(index), index=index, name="weight")


def market_cap_weight(data: pd.DataFrame, cap_col: str) -> pd.Series:
    if cap_col not in data.columns:
        raise ValueError(f"Missing cap column: {cap_col}")
    return normalize_weights(data[cap_col]).rename("weight")


def inverse_variance_weight(data: pd.DataFrame, vol_col: str, min_vol: float = 1e-6) -> pd.Series:
    if vol_col not in data.columns:
        raise ValueError(f"Missing volatility column: {vol_col}")
    vol = data[vol_col].astype(float).clip(lower=min_vol)
    inv_var = 1.0 / (vol * vol)
    return normalize_weights(inv_var).rename("weight")


def composite_factor_score(
    data: pd.DataFrame,
    factor_cols: list[str],
    directions: list[float] | None = None,
) -> pd.Series:
    """Build an equal-weight composite z-score.

    `directions` should be 1 for higher-is-better and -1 for lower-is-better.
    """
    missing = [col for col in factor_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")
    if directions is None:
        directions = [1.0] * len(factor_cols)
    if len(directions) != len(factor_cols):
        raise ValueError("directions length must match factor_cols length")

    parts = []
    for col, direction in zip(factor_cols, directions):
        parts.append(float(direction) * zscore(data[col].astype(float)))
    return pd.concat(parts, axis=1).mean(axis=1).rename("composite_score")


def _normal_cdf(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))))


def factor_tilt_weight(
    data: pd.DataFrame,
    factor_cols: list[str],
    base_weight: pd.Series | None = None,
    directions: list[float] | None = None,
    strength: float = 0.5,
    method: str = "linear",
) -> pd.Series:
    """Tilt base weights by composite factor score.

    Methods:
    - `linear`: multiplier = max(0, 1 + strength * score)
    - `exp`: multiplier = exp(strength * clipped_score)
    - `normal_cdf`: multiplier = normal CDF(score)
    """
    if base_weight is None:
        base_weight = equal_weight(data.index)
    base = normalize_weights(base_weight.reindex(data.index).fillna(0.0))
    score = composite_factor_score(data, factor_cols, directions=directions)

    method = method.lower()
    if method == "linear":
        multiplier = (1.0 + strength * score).clip(lower=0.0)
    elif method == "exp":
        multiplier = np.exp(strength * score.clip(lower=-10.0, upper=10.0))
    elif method == "normal_cdf":
        multiplier = _normal_cdf(score).clip(lower=EPS)
    else:
        raise ValueError("method must be one of: linear, exp, normal_cdf")

    return normalize_weights(base * multiplier).rename("weight")


def enforce_weight_cap(weights: pd.Series, caps: pd.Series, max_iter: int = 100) -> pd.Series:
    """Apply long-only per-asset caps and redistribute excess weight.

    Raises ValueError when total caps are below 100%, because a fully invested
    long-only portfolio is infeasible under those caps.
    """
    weights = normalize_weights(weights)
    caps = caps.reindex(weights.index).astype(float).fillna(np.inf)
    caps = caps.clip(lower=0.0)
    finite_cap_sum = caps.replace(np.inf, 0.0).sum()
    if np.isfinite(caps).all() and caps.sum() < 1.0 - EPS:
        raise ValueError("Infeasible caps: sum of caps is below 1")
    if finite_cap_sum < 1.0 and np.isinf(caps).any():
        caps = caps.copy()

    capped = pd.Series(0.0, index=weights.index)
    free = pd.Series(True, index=weights.index)
    remaining = 1.0
    base = weights.copy()

    for _ in range(max_iter):
        if remaining <= EPS or not free.any():
            break
        candidate = pd.Series(0.0, index=weights.index)
        free_base = base[free]
        if free_base.sum() <= EPS:
            candidate.loc[free] = remaining / int(free.sum())
        else:
            candidate.loc[free] = remaining * free_base / free_base.sum()
        breach = free & (candidate > caps + EPS)
        if not breach.any():
            capped.loc[free] = candidate.loc[free]
            remaining = 0.0
            break
        capped.loc[breach] = caps.loc[breach]
        remaining = 1.0 - capped.sum()
        free.loc[breach] = False

    if remaining > 1e-8:
        available = caps - capped
        if available.sum() < remaining - 1e-8:
            raise ValueError("Infeasible caps after redistribution")
        add = remaining * available.clip(lower=0.0) / available.clip(lower=0.0).sum()
        capped = capped + add
    return capped.rename("weight")


def capacity_weight_caps(
    amount: pd.Series,
    capacity_base: float,
    participation_rate: float = 0.10,
) -> pd.Series:
    """Convert trading amount into per-asset max weights for a target fund size."""
    if capacity_base <= 0:
        raise ValueError("capacity_base must be positive")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    return (amount.astype(float).clip(lower=0.0) * participation_rate / capacity_base).rename("capacity_cap")


def apply_capacity_weight_cap(
    weights: pd.Series,
    amount: pd.Series,
    capacity_base: float,
    participation_rate: float = 0.10,
) -> pd.Series:
    caps = capacity_weight_caps(amount, capacity_base, participation_rate)
    return enforce_weight_cap(weights, caps)


def portfolio_capacity(
    weights: pd.Series,
    amount: pd.Series,
    participation_rate: float = 0.10,
) -> float:
    """Estimate fund-size capacity from weights and traded amount.

    The binding stock is the minimum of amount_i * participation_rate / weight_i.
    """
    w = weights.astype(float)
    amount = amount.reindex(w.index).astype(float)
    active = w > EPS
    if not active.any():
        return float("nan")
    capacity = amount[active].clip(lower=0.0) * participation_rate / w[active]
    return float(capacity.min())


def one_way_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way turnover for a full rebalance: 0.5 * sum(abs(delta_weight))."""
    index = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(index).fillna(0.0)
    new = new_weights.reindex(index).fillna(0.0)
    return float(0.5 * (new - old).abs().sum())


def industry_exposure(weights: pd.Series, industry: pd.Series) -> pd.Series:
    industry = industry.reindex(weights.index)
    return weights.groupby(industry).sum().sort_values(ascending=False).rename("weight")


def constraint_report(
    weights: pd.Series,
    max_weight: float | None = None,
    industry: pd.Series | None = None,
    industry_bounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return simple long-only constraint violations.

    `industry_bounds` index should be industry name, with optional `lower` and
    `upper` columns.
    """
    rows: list[dict[str, object]] = []
    if max_weight is not None:
        for asset, weight in weights[weights > max_weight + EPS].items():
            rows.append(
                {
                    "constraint": "max_weight",
                    "key": asset,
                    "value": float(weight),
                    "bound": float(max_weight),
                    "violation": float(weight - max_weight),
                }
            )

    if industry is not None and industry_bounds is not None:
        exposure = industry_exposure(weights, industry)
        for ind, bounds in industry_bounds.iterrows():
            value = float(exposure.get(ind, 0.0))
            lower = bounds.get("lower", np.nan)
            upper = bounds.get("upper", np.nan)
            if pd.notna(lower) and value < float(lower) - EPS:
                rows.append(
                    {
                        "constraint": "industry_lower",
                        "key": ind,
                        "value": value,
                        "bound": float(lower),
                        "violation": float(lower - value),
                    }
                )
            if pd.notna(upper) and value > float(upper) + EPS:
                rows.append(
                    {
                        "constraint": "industry_upper",
                        "key": ind,
                        "value": value,
                        "bound": float(upper),
                        "violation": float(value - upper),
                    }
                )
    return pd.DataFrame(rows)


def composite_ir_weights(
    mean_ic: pd.Series,
    ic_cov: pd.DataFrame,
    l2_reg: float = 1e-6,
    normalize: str = "abs_sum",
) -> pd.Series:
    """Estimate factor weights that maximize composite factor IR.

    Under the standard approximation, the unconstrained direction is
    ``inv(IC covariance) @ mean(IC)``. This is a research helper; production
    use still needs turnover, exposure, and robustness constraints.
    """
    if normalize not in {"abs_sum", "sum", "none"}:
        raise ValueError("normalize must be 'abs_sum', 'sum', or 'none'.")
    factors = mean_ic.index.intersection(ic_cov.index).intersection(ic_cov.columns)
    if factors.empty:
        raise ValueError("mean_ic and ic_cov have no common factors.")
    mu = mean_ic.reindex(factors).astype(float).fillna(0.0)
    cov = ic_cov.reindex(index=factors, columns=factors).astype(float).fillna(0.0)
    cov = cov + np.eye(len(factors)) * float(l2_reg)
    raw = pd.Series(np.linalg.pinv(cov.to_numpy()) @ mu.to_numpy(), index=factors, name="factor_weight")
    if normalize == "none":
        return raw
    denom = raw.abs().sum() if normalize == "abs_sum" else raw.sum()
    if abs(denom) <= EPS:
        return pd.Series(1.0 / len(raw), index=raw.index, name="factor_weight")
    return (raw / denom).rename("factor_weight")


def inverse_exclusion_flag(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    lower_q: float = 0.5,
    high_is_good: bool = True,
    output_col: str = "inverse_exclusion_flag",
    min_assets: int = 20,
) -> pd.DataFrame:
    """Flag the adverse side of a factor for inverse exclusion."""
    if not 0 < lower_q < 1:
        raise ValueError("lower_q must be between 0 and 1.")
    out = df.copy()
    out[output_col] = False
    for _, group in out.groupby(date_col, sort=True):
        signal = group[factor_col].dropna().astype(float)
        if signal.shape[0] < min_assets:
            continue
        if high_is_good:
            threshold = signal.quantile(lower_q)
            out.loc[group.index, output_col] = group[factor_col] <= threshold
        else:
            threshold = signal.quantile(1.0 - lower_q)
            out.loc[group.index, output_col] = group[factor_col] >= threshold
    out[output_col] = out[output_col].fillna(False).astype(bool)
    return out


def realized_factor_exposure_table(
    df: pd.DataFrame,
    date_col: str,
    weight_col: str,
    exposure_cols: list[str],
    benchmark_weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute portfolio or active factor exposure by date."""
    missing = [col for col in [date_col, weight_col, *exposure_cols] if col not in df.columns]
    if benchmark_weight_col and benchmark_weight_col not in df.columns:
        missing.append(benchmark_weight_col)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for date, group in df.groupby(date_col, sort=True):
        weights = group[weight_col].astype(float)
        if benchmark_weight_col:
            weights = weights - group[benchmark_weight_col].astype(float).fillna(0.0)
        row: dict[str, object] = {"date": date}
        for col in exposure_cols:
            row[col] = float((weights * group[col].astype(float)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_exposure_caps(
    exposure_df: pd.DataFrame,
    date_col: str,
    exposure_cols: list[str],
    window: int = 12,
    method: str = "mean",
    min_periods: int | None = None,
    cap_floor: float | None = None,
    cap_ceiling: float | None = None,
    suffix: str = "_cap",
) -> pd.DataFrame:
    """Set next-period exposure caps from prior realized exposures.

    The cap is based on shifted rolling absolute realized exposure, so it does
    not look ahead.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'.")
    if window < 1:
        raise ValueError("window must be positive.")
    min_periods = window if min_periods is None else min_periods
    out = exposure_df.sort_values(date_col).copy()
    for col in exposure_cols:
        past_abs = out[col].astype(float).abs().shift(1)
        rolling = past_abs.rolling(window=window, min_periods=min_periods)
        cap = rolling.mean() if method == "mean" else rolling.median()
        if cap_floor is not None:
            cap = cap.clip(lower=float(cap_floor))
        if cap_ceiling is not None:
            cap = cap.clip(upper=float(cap_ceiling))
        out[f"{col}{suffix}"] = cap
    return out


def build_weight_table(
    df: pd.DataFrame,
    method: str,
    asset_col: str,
    cap_col: str | None = None,
    vol_col: str | None = None,
    factor_cols: list[str] | None = None,
    directions: list[float] | None = None,
    base_weight_col: str | None = None,
    tilt_method: str = "linear",
    tilt_strength: float = 0.5,
    max_weight: float | None = None,
    amount_col: str | None = None,
    capacity_base: float | None = None,
    participation_rate: float = 0.10,
) -> pd.DataFrame:
    out = df.copy().set_index(asset_col, drop=False)
    method = method.lower()
    if method == "equal":
        weights = equal_weight(out.index)
    elif method == "market_cap":
        if cap_col is None:
            raise ValueError("market_cap method requires cap_col")
        weights = market_cap_weight(out, cap_col)
    elif method == "inverse_variance":
        if vol_col is None:
            raise ValueError("inverse_variance method requires vol_col")
        weights = inverse_variance_weight(out, vol_col)
    elif method == "factor_tilt":
        if not factor_cols:
            raise ValueError("factor_tilt method requires factor_cols")
        base_weight = out[base_weight_col] if base_weight_col else None
        weights = factor_tilt_weight(
            out,
            factor_cols=factor_cols,
            base_weight=base_weight,
            directions=directions,
            strength=tilt_strength,
            method=tilt_method,
        )
    else:
        raise ValueError("method must be one of: equal, market_cap, inverse_variance, factor_tilt")

    if max_weight is not None:
        weights = enforce_weight_cap(weights, pd.Series(max_weight, index=weights.index))
    if amount_col and capacity_base:
        weights = apply_capacity_weight_cap(
            weights,
            amount=out[amount_col],
            capacity_base=capacity_base,
            participation_rate=participation_rate,
        )

    out["weight"] = weights
    if amount_col:
        out["capacity_contribution"] = np.where(
            out["weight"] > EPS,
            out[amount_col].astype(float) * participation_rate / out["weight"],
            np.nan,
        )
    return out.reset_index(drop=True)


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--method", required=True, choices=["equal", "market_cap", "inverse_variance", "factor_tilt"])
    parser.add_argument("--cap-col")
    parser.add_argument("--vol-col")
    parser.add_argument("--factor-cols", help="Comma-separated factor columns.")
    parser.add_argument("--directions", help="Comma-separated directions, 1 or -1.")
    parser.add_argument("--base-weight-col")
    parser.add_argument("--tilt-method", default="linear", choices=["linear", "exp", "normal_cdf"])
    parser.add_argument("--tilt-strength", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float)
    parser.add_argument("--amount-col")
    parser.add_argument("--capacity-base", type=float)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--old-weight-csv")
    parser.add_argument("--old-weight-col", default="weight")
    parser.add_argument("--output-dir", default="portfolio_weighting_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    factor_cols = [col.strip() for col in args.factor_cols.split(",")] if args.factor_cols else None
    directions = parse_float_list(args.directions)
    result = build_weight_table(
        df,
        method=args.method,
        asset_col=args.asset_col,
        cap_col=args.cap_col,
        vol_col=args.vol_col,
        factor_cols=factor_cols,
        directions=directions,
        base_weight_col=args.base_weight_col,
        tilt_method=args.tilt_method,
        tilt_strength=args.tilt_strength,
        max_weight=args.max_weight,
        amount_col=args.amount_col,
        capacity_base=args.capacity_base,
        participation_rate=args.participation_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "weights.csv", index=False, encoding="utf-8-sig")

    summary = {
        "asset_count": int(result.shape[0]),
        "active_weight_count": int((result["weight"] > EPS).sum()),
        "max_weight": float(result["weight"].max()) if not result.empty else float("nan"),
        "effective_n": float(1.0 / (result["weight"] ** 2).sum()) if (result["weight"] ** 2).sum() > EPS else float("nan"),
    }
    if args.amount_col:
        summary["portfolio_capacity"] = portfolio_capacity(
            result.set_index(args.asset_col)["weight"],
            result.set_index(args.asset_col)[args.amount_col],
            participation_rate=args.participation_rate,
        )

    if args.old_weight_csv:
        old = pd.read_csv(args.old_weight_csv, encoding=args.encoding).set_index(args.asset_col)[args.old_weight_col]
        new = result.set_index(args.asset_col)["weight"]
        summary["one_way_turnover"] = one_way_turnover(old, new)

    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([summary]))
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
