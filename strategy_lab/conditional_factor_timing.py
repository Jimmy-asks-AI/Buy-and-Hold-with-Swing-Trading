#!/usr/bin/env python
"""Conditional-expectation factor timing.

This module implements a reusable version of the conditional expectation
framework: use market-state variables to adjust expected factor returns and
factor-return covariance, then translate the adjusted moments into dynamic
factor weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _regularized_inverse(matrix: np.ndarray, ridge: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    reg = ridge * np.eye(matrix.shape[0])
    return np.linalg.pinv(matrix + reg)


def conditional_moments(
    history: pd.DataFrame,
    factor_cols: list[str],
    condition_cols: list[str],
    current_conditions: pd.Series | dict[str, float],
    ridge: float = 1e-6,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate E[f | v*] and Cov[f | v*] under a joint-normal assumption."""
    _require_columns(history, [*factor_cols, *condition_cols])
    clean = history[[*factor_cols, *condition_cols]].dropna()
    if clean.shape[0] <= max(3, len(factor_cols) + len(condition_cols)):
        raise ValueError("Not enough history to estimate conditional moments")
    factors = clean[factor_cols].astype(float)
    if not condition_cols:
        return factors.mean(), factors.cov()

    conditions = clean[condition_cols].astype(float)
    mean_f = factors.mean().to_numpy(dtype=float)
    mean_v = conditions.mean().to_numpy(dtype=float)
    cov = clean.astype(float).cov()
    cov_ff = cov.loc[factor_cols, factor_cols].to_numpy(dtype=float)
    cov_fv = cov.loc[factor_cols, condition_cols].to_numpy(dtype=float)
    cov_vv = cov.loc[condition_cols, condition_cols].to_numpy(dtype=float)
    inv_vv = _regularized_inverse(cov_vv, ridge=ridge)

    current = pd.Series(current_conditions, dtype=float).reindex(condition_cols).to_numpy(dtype=float)
    delta_mean = cov_fv @ inv_vv @ (current - mean_v)
    cond_mean = mean_f + delta_mean
    cond_cov = cov_ff - cov_fv @ inv_vv @ cov_fv.T
    cond_cov = (cond_cov + cond_cov.T) / 2.0
    cond_cov = cond_cov + ridge * np.eye(cond_cov.shape[0])
    return pd.Series(cond_mean, index=factor_cols, name="conditional_mean"), pd.DataFrame(
        cond_cov, index=factor_cols, columns=factor_cols
    )


def max_ic_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    long_only: bool = True,
    normalize: str = "sum",
    ridge: float = 1e-6,
) -> pd.Series:
    """Convert expected factor returns and covariance into factor weights."""
    factors = list(expected_returns.index)
    cov = covariance.reindex(index=factors, columns=factors).fillna(0.0).to_numpy(dtype=float)
    mu = expected_returns.astype(float).to_numpy()
    raw = _regularized_inverse(cov, ridge=ridge) @ mu
    weights = pd.Series(raw, index=factors, dtype="float64")
    if long_only:
        weights = weights.clip(lower=0.0)
    if normalize == "sum":
        total = weights.sum()
        if abs(total) <= EPS:
            weights = pd.Series(1.0 / len(weights), index=factors)
        else:
            weights = weights / total
    elif normalize == "gross":
        total = weights.abs().sum()
        if total <= EPS:
            weights = pd.Series(1.0 / len(weights), index=factors)
        else:
            weights = weights / total
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: sum, gross, none")
    return weights.rename("factor_weight")


def aic_for_conditions(
    history: pd.DataFrame,
    factor_cols: list[str],
    condition_cols: list[str],
    ridge: float = 1e-6,
) -> float:
    """Compute the report-style AIC for a selected condition set."""
    _require_columns(history, [*factor_cols, *condition_cols])
    clean = history[[*factor_cols, *condition_cols]].dropna()
    if clean.shape[0] <= max(3, len(factor_cols) + len(condition_cols)):
        return float("inf")
    factors = clean[factor_cols].astype(float)
    if condition_cols:
        current = clean[condition_cols].astype(float).iloc[-1]
        _, cond_cov = conditional_moments(clean, factor_cols, condition_cols, current, ridge=ridge)
        cov = cond_cov.to_numpy(dtype=float)
    else:
        cov = factors.cov().to_numpy(dtype=float) + ridge * np.eye(len(factor_cols))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    t = clean.shape[0]
    n_factors = len(factor_cols)
    k_conditions = len(condition_cols)
    return float(t * logdet + 2 * n_factors * k_conditions)


def forward_select_conditions_aic(
    history: pd.DataFrame,
    factor_cols: list[str],
    candidate_condition_cols: list[str],
    max_conditions: int | None = None,
    ridge: float = 1e-6,
) -> list[str]:
    """Forward-select condition variables by minimizing AIC."""
    _require_columns(history, [*factor_cols, *candidate_condition_cols])
    selected: list[str] = []
    remaining = list(candidate_condition_cols)
    current_aic = aic_for_conditions(history, factor_cols, selected, ridge=ridge)
    while remaining and (max_conditions is None or len(selected) < max_conditions):
        trial_scores = []
        for candidate in remaining:
            cols = selected + [candidate]
            trial_scores.append((aic_for_conditions(history, factor_cols, cols, ridge=ridge), candidate))
        best_aic, best_candidate = min(trial_scores, key=lambda x: x[0])
        if best_aic >= current_aic:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_aic = best_aic
    return selected


def rolling_conditional_factor_weights(
    panel: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    condition_cols: list[str],
    lookback: int = 24,
    min_periods: int | None = None,
    use_aic: bool = True,
    max_conditions: int | None = None,
    long_only: bool = True,
    ridge: float = 1e-6,
) -> pd.DataFrame:
    """Generate live-safe rolling conditional factor weights.

    Row i uses rows before i as estimation history and row i's condition values
    as the current observable market state.
    """
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    if min_periods is None:
        min_periods = max(12, lookback // 2)
    _require_columns(panel, [date_col, *factor_cols, *condition_cols])
    data = panel.sort_values(date_col).reset_index(drop=True).copy()
    rows = []
    for idx, row in data.iterrows():
        start = max(0, idx - lookback)
        history = data.iloc[start:idx]
        if history.shape[0] < min_periods:
            continue
        selected = (
            forward_select_conditions_aic(history, factor_cols, condition_cols, max_conditions=max_conditions, ridge=ridge)
            if use_aic
            else list(condition_cols)
        )
        try:
            mean, cov = conditional_moments(history, factor_cols, selected, row[selected], ridge=ridge)
            weights = max_ic_weights(mean, cov, long_only=long_only, normalize="sum", ridge=ridge)
        except ValueError:
            continue
        for factor, weight in weights.items():
            rows.append(
                {
                    "date": row[date_col],
                    "factor": factor,
                    "factor_weight": float(weight),
                    "selected_conditions": ",".join(selected),
                    "n_conditions": len(selected),
                    "lookback_obs": int(history.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def add_rolling_market_conditions(
    market: pd.DataFrame,
    date_col: str,
    price_cols: list[str] | None = None,
    turnover_cols: list[str] | None = None,
    valuation_cols: list[str] | None = None,
    rate_cols: list[str] | None = None,
    windows: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """Build common condition variables from market time series."""
    cols = [date_col]
    for group in [price_cols, turnover_cols, valuation_cols, rate_cols]:
        if group:
            cols.extend(group)
    _require_columns(market, cols)
    out = market.sort_values(date_col).copy()
    if price_cols:
        for col in price_cols:
            ret = out[col].astype(float).pct_change()
            for window in windows:
                out[f"{col}_ret_{window}"] = out[col].astype(float).pct_change(window)
                out[f"{col}_vol_{window}"] = ret.rolling(window, min_periods=max(3, window // 2)).std()
    if turnover_cols:
        for col in turnover_cols:
            for window in windows:
                out[f"{col}_turnover_mean_{window}"] = out[col].astype(float).rolling(
                    window, min_periods=max(3, window // 2)
                ).mean()
    if valuation_cols:
        for col in valuation_cols:
            out[f"{col}_level"] = out[col].astype(float)
    if rate_cols:
        for col in rate_cols:
            out[f"{col}_level"] = out[col].astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--factors", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--no-aic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = rolling_conditional_factor_weights(
        df,
        date_col=args.date_col,
        factor_cols=args.factors,
        condition_cols=args.conditions,
        lookback=args.lookback,
        use_aic=not args.no_aic,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
