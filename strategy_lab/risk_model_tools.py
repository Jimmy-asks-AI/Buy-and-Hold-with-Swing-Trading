#!/usr/bin/env python
"""Risk-model helpers for factor covariance and portfolio risk checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def exp_decay_weights(length: int, half_life: float) -> np.ndarray:
    """Return oldest-to-newest exponential decay weights."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if half_life <= 0:
        raise ValueError("half_life must be positive.")
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = 0.5 ** (ages / float(half_life))
    return weights / weights.sum()


def ewma_covariance(
    returns: pd.DataFrame,
    half_life: float = 90.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Exponentially weighted covariance matrix."""
    clean = returns.dropna(how="all").astype(float)
    min_periods = clean.shape[1] + 1 if min_periods is None else min_periods
    clean = clean.dropna(axis=0)
    if clean.shape[0] < min_periods:
        raise ValueError("Not enough observations to estimate covariance.")
    weights = exp_decay_weights(clean.shape[0], half_life)
    mean = np.average(clean.to_numpy(), axis=0, weights=weights)
    centered = clean.to_numpy() - mean
    cov = (centered * weights[:, None]).T @ centered
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def newey_west_covariance(
    returns: pd.DataFrame,
    horizon: int = 1,
    lags: int = 2,
    half_life: float = 90.0,
) -> pd.DataFrame:
    """Newey-West adjusted covariance for a target return horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if lags < 0:
        raise ValueError("lags must be non-negative.")
    clean = returns.dropna(axis=0).astype(float)
    if clean.shape[0] <= lags + 1:
        raise ValueError("Not enough observations for Newey-West adjustment.")
    base = ewma_covariance(clean, half_life=half_life)
    values = clean.to_numpy()
    weights = exp_decay_weights(clean.shape[0], half_life)
    adjustment = base.to_numpy().copy()
    for lag in range(1, lags + 1):
        x = values[:-lag]
        y = values[lag:]
        lag_weights = weights[lag:]
        lag_weights = lag_weights / lag_weights.sum()
        x_center = x - np.average(x, axis=0, weights=lag_weights)
        y_center = y - np.average(y, axis=0, weights=lag_weights)
        gamma = (x_center * lag_weights[:, None]).T @ y_center
        bartlett = 1.0 - lag / (lags + 1.0)
        adjustment += bartlett * (gamma + gamma.T)
    adjustment = horizon * adjustment
    return pd.DataFrame(adjustment, index=clean.columns, columns=clean.columns)


def nearest_psd(matrix: pd.DataFrame, min_eigenvalue: float = 1e-8) -> pd.DataFrame:
    """Clip eigenvalues to make a symmetric matrix positive semi-definite."""
    values = matrix.to_numpy(dtype=float)
    sym = (values + values.T) / 2.0
    eigval, eigvec = np.linalg.eigh(sym)
    eigval = np.clip(eigval, min_eigenvalue, None)
    psd = eigvec @ np.diag(eigval) @ eigvec.T
    return pd.DataFrame(psd, index=matrix.index, columns=matrix.columns)


def factor_model_covariance(
    exposure: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
) -> pd.DataFrame:
    """Build stock covariance matrix from exposures, factor cov, and specific variance."""
    factors = exposure.columns.intersection(factor_cov.index).intersection(factor_cov.columns)
    if factors.empty:
        raise ValueError("No common factor columns.")
    x = exposure[factors].astype(float)
    f = factor_cov.reindex(index=factors, columns=factors).astype(float)
    common = x.to_numpy() @ f.to_numpy() @ x.to_numpy().T
    spec = specific_var.reindex(x.index).astype(float).fillna(0.0).clip(lower=0.0)
    cov = common + np.diag(spec.to_numpy())
    return pd.DataFrame(cov, index=x.index, columns=x.index)


def portfolio_predicted_vol(
    weights: pd.Series,
    covariance: pd.DataFrame,
    annualization: float | None = None,
) -> float:
    """Predicted portfolio volatility from covariance matrix."""
    idx = weights.index.intersection(covariance.index).intersection(covariance.columns)
    if idx.empty:
        return float("nan")
    w = weights.reindex(idx).astype(float).fillna(0.0).to_numpy()
    cov = covariance.reindex(index=idx, columns=idx).astype(float).to_numpy()
    var = float(w.T @ cov @ w)
    vol = np.sqrt(max(var, 0.0))
    if annualization:
        vol *= np.sqrt(float(annualization))
    return float(vol)


def bias_statistics(
    realized_return: pd.Series,
    predicted_vol: pd.Series,
    window_name: str = "sample",
) -> dict[str, float | str]:
    """Compute risk-model bias statistic from standardized returns."""
    clean = pd.DataFrame({"ret": realized_return, "vol": predicted_vol}).dropna()
    clean = clean[clean["vol"].abs() > EPS]
    if clean.shape[0] < 3:
        return {"window": window_name, "count": int(clean.shape[0]), "bias": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    standardized = clean["ret"] / clean["vol"]
    bias = standardized.std(ddof=1)
    n = standardized.shape[0]
    ci_half_width = np.sqrt(2.0 / n)
    return {
        "window": window_name,
        "count": int(n),
        "bias": float(bias),
        "ci_low": float(1.0 - ci_half_width),
        "ci_high": float(1.0 + ci_half_width),
        "risk_underestimated": bool(bias > 1.0 + ci_half_width),
        "risk_overestimated": bool(bias < 1.0 - ci_half_width),
    }


def factor_return_attribution(
    holdings: pd.DataFrame,
    factor_returns: pd.DataFrame,
    date_col: str,
    asset_col: str,
    weight_col: str,
    exposure_cols: Sequence[str],
    specific_return_col: str | None = None,
) -> pd.DataFrame:
    """Attribute portfolio return to factor exposure and optional specific return."""
    exposures = list(exposure_cols)
    rows: list[dict[str, object]] = []
    factor_returns = factor_returns.set_index(date_col)
    for date, group in holdings.groupby(date_col, sort=True):
        if date not in factor_returns.index:
            continue
        weights = group[weight_col].astype(float)
        row: dict[str, object] = {"date": date}
        total_factor = 0.0
        for col in exposures:
            exposure = float((weights * group[col].astype(float)).sum())
            factor_ret = float(factor_returns.loc[date, col])
            contrib = exposure * factor_ret
            row[f"{col}_exposure"] = exposure
            row[f"{col}_return"] = factor_ret
            row[f"{col}_contribution"] = contrib
            total_factor += contrib
        row["factor_contribution"] = total_factor
        if specific_return_col:
            row["specific_contribution"] = float((weights * group[specific_return_col].astype(float)).sum())
            row["total_attributed_return"] = row["factor_contribution"] + row["specific_contribution"]
        rows.append(row)
    return pd.DataFrame(rows)


def tracking_error_objective(
    active_weight: pd.Series,
    alpha: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> float:
    """Risk-adjusted active return objective: alpha'x - lambda*x'Cov*x."""
    idx = active_weight.index.intersection(alpha.index).intersection(covariance.index).intersection(covariance.columns)
    if idx.empty:
        return float("nan")
    x = active_weight.reindex(idx).astype(float).fillna(0.0).to_numpy()
    a = alpha.reindex(idx).astype(float).fillna(0.0).to_numpy()
    cov = covariance.reindex(index=idx, columns=idx).astype(float).to_numpy()
    return float(a @ x - float(risk_aversion) * (x.T @ cov @ x))


def rolling_factor_sign_return_stats(
    factor_returns: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    window: int = 24,
    min_periods: int = 12,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute trailing average positive and negative factor returns."""
    out = factor_returns.sort_values(date_col).copy()
    for col in factor_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        pos = values.where(values > 0)
        neg = values.where(values < 0)
        out[f"{col}_avg_pos"] = pos.rolling(window, min_periods=min_periods).mean().shift(shift)
        out[f"{col}_avg_neg"] = neg.rolling(window, min_periods=min_periods).mean().shift(shift)
    return out


def dynamic_factor_exposure_bounds(
    forecasts: pd.DataFrame,
    date_col: str,
    factor_cols: Sequence[str],
    risk_aversion: float = 6.0,
    max_abs_bound: float | None = 1.0,
    suffix: str = "_forecast",
) -> pd.DataFrame:
    """Map factor-return forecasts to dynamic active exposure bounds.

    For a positive forecast, the lower bound is zero and the upper bound is
    forecast divided by average loss and risk aversion. For a negative forecast,
    the upper bound is zero and the lower bound is negative.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    rows = []
    for _, row in forecasts.sort_values(date_col).iterrows():
        date = row[date_col]
        for factor in factor_cols:
            forecast = row.get(f"{factor}{suffix}", np.nan)
            avg_pos = row.get(f"{factor}_avg_pos", np.nan)
            avg_neg = row.get(f"{factor}_avg_neg", np.nan)
            if pd.isna(forecast) or pd.isna(avg_pos) or pd.isna(avg_neg):
                continue
            forecast = float(forecast)
            avg_pos = float(avg_pos)
            avg_neg = float(avg_neg)
            lower = 0.0
            upper = 0.0
            max_loss = abs(forecast) / risk_aversion
            if forecast > 0 and avg_neg < -EPS:
                upper = -max_loss / avg_neg
            elif forecast < 0 and avg_pos > EPS:
                lower = -max_loss / avg_pos
            if max_abs_bound is not None:
                upper = min(float(max_abs_bound), max(0.0, upper))
                lower = max(-float(max_abs_bound), min(0.0, lower))
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "forecast": forecast,
                    "avg_positive_return": avg_pos,
                    "avg_negative_return": avg_neg,
                    "risk_aversion": float(risk_aversion),
                    "lower": float(lower),
                    "upper": float(upper),
                    "max_loss": float(max_loss),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-return-csv")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--output")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lags", type=int, default=2)
    args = parser.parse_args()

    if not args.factor_return_csv:
        raise ValueError("--factor-return-csv is required.")
    returns = pd.read_csv(args.factor_return_csv, encoding=args.encoding)
    cov = newey_west_covariance(returns, horizon=args.horizon, lags=args.lags)
    cov = nearest_psd(cov)
    output = Path(args.output or "risk_model_covariance.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(output, encoding="utf-8-sig")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
