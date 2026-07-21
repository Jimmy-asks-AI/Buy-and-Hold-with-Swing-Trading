#!/usr/bin/env python
"""Macro sensitivity and uncertainty-beta factor helpers."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def macro_change_features(
    macro: pd.DataFrame,
    date_col: str,
    macro_cols: Iterable[str],
    method: str = "diff",
    suffix: str | None = None,
) -> pd.DataFrame:
    """Create stationary macro features with diff or percent-change transforms."""
    if method not in {"diff", "pct_change"}:
        raise ValueError("method must be 'diff' or 'pct_change'.")
    out = macro.sort_values(date_col).copy()
    suffix = suffix or f"_{method}"
    for col in macro_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}{suffix}"] = values.diff() if method == "diff" else values.pct_change()
    return out


def rolling_macro_sensitivity(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
    macro_col: str,
    control_cols: Iterable[str] | None = None,
    window: int = 36,
    min_periods: int | None = None,
    shift: int = 1,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Estimate rolling stock MacroBeta from returns on macro variables.

    The output beta is shifted so row t uses information available before t.
    """
    controls = list(control_cols or [])
    min_periods = min_periods or max(18, len(controls) + 8)
    target = output_col or f"{macro_col}_beta"
    out = panel.sort_values([asset_col, date_col]).copy()
    out[target] = np.nan
    cols = [return_col, macro_col, *controls]
    for _, group in out.groupby(asset_col, sort=False):
        values = group[cols].apply(pd.to_numeric, errors="coerce")
        betas = pd.Series(np.nan, index=group.index, dtype="float64")
        for pos in range(len(group)):
            hist = values.iloc[max(0, pos - window) : pos].dropna()
            if hist.shape[0] < min_periods:
                continue
            x = hist[[macro_col, *controls]].to_numpy(dtype=float)
            x = np.column_stack([np.ones(hist.shape[0]), x])
            y = hist[return_col].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            betas.iloc[pos] = beta[1]
        out.loc[group.index, target] = betas.shift(shift).to_numpy(dtype=float)
    return out


def rolling_macro_sensitivity_stats(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
    macro_col: str,
    control_cols: Iterable[str] | None = None,
    window: int = 36,
    min_periods: int | None = None,
    shift: int = 1,
    beta_col: str | None = None,
    t_col: str | None = None,
) -> pd.DataFrame:
    """Estimate rolling MacroBeta and its t-statistic by asset."""
    controls = list(control_cols or [])
    min_periods = min_periods or max(18, len(controls) + 8)
    beta_target = beta_col or f"{macro_col}_beta"
    t_target = t_col or f"{macro_col}_t"
    out = panel.sort_values([asset_col, date_col]).copy()
    out[beta_target] = np.nan
    out[t_target] = np.nan
    cols = [return_col, macro_col, *controls]
    for _, group in out.groupby(asset_col, sort=False):
        values = group[cols].apply(pd.to_numeric, errors="coerce")
        betas = pd.Series(np.nan, index=group.index, dtype="float64")
        t_stats = pd.Series(np.nan, index=group.index, dtype="float64")
        for pos in range(len(group)):
            hist = values.iloc[max(0, pos - window) : pos].dropna()
            k = len(controls) + 2
            if hist.shape[0] < max(min_periods, k + 2):
                continue
            x = hist[[macro_col, *controls]].to_numpy(dtype=float)
            x = np.column_stack([np.ones(hist.shape[0]), x])
            y = hist[return_col].to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            resid = y - x @ beta
            dof = hist.shape[0] - x.shape[1]
            sigma2 = float(resid @ resid / dof) if dof > 0 else np.nan
            cov_beta = sigma2 * np.linalg.pinv(x.T @ x)
            se = np.sqrt(np.diag(cov_beta))
            betas.iloc[pos] = beta[1]
            t_stats.iloc[pos] = beta[1] / se[1] if se[1] > EPS else np.nan
        out.loc[group.index, beta_target] = betas.shift(shift).to_numpy(dtype=float)
        out.loc[group.index, t_target] = t_stats.shift(shift).to_numpy(dtype=float)
    return out


def macro_tvalue_signal(
    df: pd.DataFrame,
    t_col: str,
    macro_forecast_col: str,
    threshold: float = 1.96,
    output_col: str = "macro_t_signal",
) -> pd.DataFrame:
    """Select significant positive/negative macro-sensitive stocks by t-value."""
    out = df.copy()
    t_value = pd.to_numeric(out[t_col], errors="coerce")
    direction = np.sign(pd.to_numeric(out[macro_forecast_col], errors="coerce"))
    signal = pd.Series(0.0, index=out.index)
    signal[(t_value >= threshold) & (direction > 0)] = 1.0
    signal[(t_value <= -threshold) & (direction < 0)] = 1.0
    signal[(t_value >= threshold) & (direction < 0)] = -1.0
    signal[(t_value <= -threshold) & (direction > 0)] = -1.0
    out[output_col] = signal
    return out


def macro_score_factor(
    df: pd.DataFrame,
    beta_col: str,
    macro_forecast_col: str,
    output_col: str = "macro_score",
) -> pd.DataFrame:
    """Compute macro score as macro sensitivity times expected macro change."""
    out = df.copy()
    out[output_col] = pd.to_numeric(out[beta_col], errors="coerce") * pd.to_numeric(
        out[macro_forecast_col], errors="coerce"
    )
    return out


def macro_beta_stability(
    beta_panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    beta_col: str,
    lag: int = 1,
    window: int = 24,
    output_col: str = "macro_beta_stability",
) -> pd.DataFrame:
    """Measure rolling stability of macro beta by asset."""
    out = beta_panel.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        beta = pd.to_numeric(group[beta_col], errors="coerce")
        stability = beta.rolling(window, min_periods=max(6, window // 3)).corr(beta.shift(lag))
        out.loc[group.index, output_col] = stability.to_numpy(dtype=float)
    return out


def uncertainty_beta_factor(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
    epu_col: str,
    market_return_col: str | None = None,
    smb_col: str | None = None,
    hml_col: str | None = None,
    window: int = 36,
    output_col: str = "epu_beta",
) -> pd.DataFrame:
    """Estimate EPU/uncertainty beta with optional market, SMB, and HML controls."""
    controls = [col for col in [market_return_col, smb_col, hml_col] if col]
    return rolling_macro_sensitivity(
        panel=panel,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        macro_col=epu_col,
        control_cols=controls,
        window=window,
        output_col=output_col,
    )


def cross_sectional_factor_premium(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    factor_cols: Iterable[str],
    min_count: int | None = None,
) -> pd.DataFrame:
    """Run date-by-date cross-sectional return regressions for factor premia."""
    factors = list(factor_cols)
    min_count = min_count or len(factors) + 20
    rows = []
    for date, group in panel[[date_col, return_col, *factors]].dropna().groupby(date_col, sort=True):
        if group.shape[0] < min_count:
            continue
        x = group[factors].to_numpy(dtype=float)
        x = np.column_stack([np.ones(group.shape[0]), x])
        y = group[return_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        row = {"date": date, "n_obs": int(group.shape[0]), "intercept": float(beta[0])}
        row.update({f"{factor}_premium": float(value) for factor, value in zip(factors, beta[1:])})
        rows.append(row)
    return pd.DataFrame(rows)


def macro_factor_checklist() -> pd.DataFrame:
    rows = [
        ("availability", "Use initial release dates and vintage data when possible; macro revisions create look-ahead risk."),
        ("stationarity", "Use differences, surprises, or asset-mimicking portfolios before time-series regression."),
        ("style", "MacroBeta is often size-contaminated; neutralize common styles before claiming alpha."),
        ("forecast", "MacroBeta alone is not a buy signal; multiply by expected macro direction to form macro score."),
        ("t_value", "Select macro-sensitive stocks by beta t-statistics, not by top/bottom raw beta alone."),
        ("stability", "Require beta stability before using a macro score in stock selection."),
        ("epu", "EPU beta is an uncertainty exposure; validate separately by stock universe such as HS300."),
        ("controls", "For EPU beta, control market, SMB, HML, turnover, valuation, profitability, and idiosyncratic vol."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def macro_indicator_catalog() -> pd.DataFrame:
    rows = [
        ("activity", "GDP YoY", "quarterly; publication lag is large"),
        ("activity", "industrial value added YoY", "directly linked to corporate operations"),
        ("activity", "manufacturing PMI", "monthly diffusion index"),
        ("investment", "fixed asset investment YoY", "cumulative data needs careful transform"),
        ("consumption", "retail sales YoY", "consumer demand channel"),
        ("inflation", "CPI YoY", "consumer price pressure"),
        ("inflation", "PPI YoY", "producer price and industrial profit channel"),
        ("money", "M0/M1/M2 YoY", "liquidity and money supply"),
        ("credit", "new RMB loans", "credit impulse proxy"),
        ("fx", "FX reserves", "external liquidity and capital-flow proxy"),
        ("rate", "treasury yield and term spread", "discount-rate channel"),
        ("credit", "credit spread and TED spread", "risk-premium channel"),
        ("commodity", "gold and Brent oil", "global risk and cost channels"),
        ("uncertainty", "China EPU index", "macro policy uncertainty proxy"),
    ]
    return pd.DataFrame(rows, columns=["category", "indicator", "note"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.catalog:
        print(macro_indicator_catalog())
    if args.checklist:
        print(macro_factor_checklist())


if __name__ == "__main__":
    main()
