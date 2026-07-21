#!/usr/bin/env python
"""Adaptive factor-premium and quantile-regression research helpers."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def ewma_weights(length: int, decay: float) -> np.ndarray:
    """Weights for EWMA where newest observation weight is proportional to decay."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1].")
    ages = np.arange(length - 1, -1, -1, dtype=float)
    raw = decay * ((1.0 - decay) ** ages)
    return raw / raw.sum()


def ewma_factor_premium_forecast(
    premia: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    window: int = 24,
    decay: float = 0.01,
    shift: int = 1,
    suffix: str = "_forecast",
) -> pd.DataFrame:
    """Forecast factor premia with fixed-decay EWMA."""
    cols = list(factor_cols)
    out = premia.sort_values(date_col).copy()
    weights = ewma_weights(window, decay)
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce")
        forecast = values.rolling(window, min_periods=window).apply(lambda x: float(np.dot(weights, x)), raw=True)
        out[f"{col}{suffix}"] = forecast.shift(shift)
    return out


def adaptive_decay_from_r2(
    r2: pd.Series,
    window: int = 24,
    decay_min: float = 0.01,
    decay_max: float = 0.50,
) -> pd.Series:
    """Map low recent model fit to faster EWMA decay."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    values = pd.to_numeric(r2, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    for i in range(window - 1, len(values)):
        sample = values.iloc[i - window + 1 : i + 1].dropna()
        if sample.shape[0] < window:
            continue
        current = values.iloc[i]
        rank_desc = int((sample > current).sum() + 1)
        out.iloc[i] = decay_min + (rank_desc - 1) / (window - 1) * (decay_max - decay_min)
    return out


def adaptive_ewma_factor_premium_forecast(
    premia: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    fit_r2_col: str,
    window: int = 24,
    decay_min: float = 0.01,
    decay_max: float = 0.50,
    shift: int = 1,
    suffix: str = "_adaptive_forecast",
) -> pd.DataFrame:
    """Forecast factor premia with R2-conditioned EWMA decay."""
    cols = list(factor_cols)
    out = premia.sort_values(date_col).copy()
    out["adaptive_decay"] = adaptive_decay_from_r2(out[fit_r2_col], window, decay_min, decay_max)
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        forecasts = np.full(len(out), np.nan, dtype=float)
        for i in range(window - 1, len(out)):
            decay = out["adaptive_decay"].iloc[i]
            if pd.isna(decay):
                continue
            sample = values[i - window + 1 : i + 1]
            if np.isnan(sample).any():
                continue
            forecasts[i] = float(np.dot(ewma_weights(window, float(decay)), sample))
        out[f"{col}{suffix}"] = pd.Series(forecasts, index=out.index).shift(shift)
    return out


def volatility_adjusted_premium_forecast(
    forecast_df: pd.DataFrame,
    factor_cols: Iterable[str],
    volatility_cols: Iterable[str],
    suffix: str = "_vol_adj",
) -> pd.DataFrame:
    """Divide factor-premium forecasts by recent premium volatility."""
    out = forecast_df.copy()
    for factor, vol_col in zip(factor_cols, volatility_cols):
        out[f"{factor}{suffix}"] = pd.to_numeric(out[factor], errors="coerce") / pd.to_numeric(
            out[vol_col], errors="coerce"
        ).replace(0.0, np.nan)
    return out


def rolling_premium_volatility(
    premia: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    window: int = 21,
    suffix: str = "_vol",
) -> pd.DataFrame:
    """Rolling volatility of factor premium series."""
    out = premia.sort_values(date_col).copy()
    for col in factor_cols:
        out[f"{col}{suffix}"] = pd.to_numeric(out[col], errors="coerce").rolling(window, min_periods=window).std()
    return out


def fit_quantile_regression(
    x: pd.DataFrame,
    y: pd.Series,
    quantile: float = 0.1,
    learning_rate: float = 0.05,
    max_iter: int = 4000,
    tol: float = 1e-7,
    l2: float = 1e-6,
) -> pd.Series:
    """Fit linear quantile regression with simple subgradient descent."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1).")
    clean = pd.concat([y.rename("_y"), x], axis=1).dropna()
    if clean.shape[0] <= x.shape[1] + 2:
        raise ValueError("Not enough observations for quantile regression.")
    yv = clean["_y"].to_numpy(dtype=float)
    xv_raw = clean[x.columns].to_numpy(dtype=float)
    mean = xv_raw.mean(axis=0)
    std = xv_raw.std(axis=0, ddof=1)
    std = np.where(std <= EPS, 1.0, std)
    xv = (xv_raw - mean) / std
    design = np.column_stack([np.ones(xv.shape[0]), xv])
    beta = np.zeros(design.shape[1], dtype=float)
    last_loss = np.inf
    for step in range(max_iter):
        residual = yv - design @ beta
        grad = -(quantile - (residual < 0).astype(float)) @ design / len(yv)
        grad[1:] += l2 * beta[1:]
        rate = learning_rate / np.sqrt(step + 1.0)
        beta = beta - rate * grad
        if step % 50 == 0:
            residual = yv - design @ beta
            loss = np.mean(np.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual))
            if abs(last_loss - loss) < tol:
                break
            last_loss = loss
    coef = beta[1:] / std
    intercept = beta[0] - np.sum(beta[1:] * mean / std)
    return pd.Series([intercept, *coef], index=["intercept", *x.columns])


def cross_sectional_quantile_coefficients(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    factor_cols: Iterable[str],
    quantile: float = 0.1,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Fit quantile regression separately for each cross section."""
    factors = list(factor_cols)
    min_count = min_count or len(factors) + 20
    rows = []
    for date, group in panel[[date_col, return_col, *factors]].dropna().groupby(date_col, sort=True):
        if group.shape[0] < min_count:
            continue
        try:
            coef = fit_quantile_regression(group[factors], group[return_col], quantile=quantile)
        except ValueError:
            continue
        row = {"date": date, "quantile": quantile, "n_obs": int(group.shape[0])}
        row.update({f"coef_{k}": float(v) for k, v in coef.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_average_coefficients(
    coef_df: pd.DataFrame,
    date_col: str = "date",
    window: int = 6,
    shift: int = 1,
) -> pd.DataFrame:
    """Rolling-average cross-sectional regression coefficients."""
    out = coef_df.sort_values(date_col).copy()
    coef_cols = [col for col in out.columns if col.startswith("coef_")]
    for col in coef_cols:
        out[f"{col}_forecast"] = pd.to_numeric(out[col], errors="coerce").rolling(window, min_periods=window).mean().shift(shift)
    return out


def predict_from_coefficient_table(
    panel: pd.DataFrame,
    coef_df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    output_col: str = "quantile_score",
) -> pd.DataFrame:
    """Apply date-specific coefficient forecasts to panel factors."""
    factors = list(factor_cols)
    coef = coef_df.set_index(date_col)
    out = panel.copy()
    out[output_col] = np.nan
    for date, group in out.groupby(date_col):
        if date not in coef.index:
            continue
        row = coef.loc[date]
        intercept = row.get("coef_intercept_forecast", np.nan)
        if pd.isna(intercept):
            continue
        values = np.full(group.shape[0], float(intercept))
        ok = True
        for factor in factors:
            c = row.get(f"coef_{factor}_forecast", np.nan)
            if pd.isna(c):
                ok = False
                break
            values = values + float(c) * pd.to_numeric(group[factor], errors="coerce").to_numpy(dtype=float)
        if ok:
            out.loc[group.index, output_col] = values
    return out


def factor_premium_model_checklist() -> pd.DataFrame:
    rows = [
        ("window", "Short windows are noisy; long windows adapt slowly to regime changes."),
        ("ewma", "EWMA decay should be chosen from observable state variables, not full-sample optimization."),
        ("r2", "Low recent cross-sectional R2 implies high heterogeneity and faster decay."),
        ("volatility", "Premium forecasts should be scaled by recent premium volatility when risk is high."),
        ("rebound", "Adaptive forecasts can lag during violent factor mean reversion."),
        ("quantile", "Choose quantile by slope significance and ranking separation, not by in-sample return only."),
        ("risk", "Low-quantile regression can amplify factor beta and needs factor-risk monitoring."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.checklist:
        print(factor_premium_model_checklist())


if __name__ == "__main__":
    main()
