#!/usr/bin/env python
"""Behavioral finance helpers for quant research.

The module connects classic behavioral-finance ideas to testable factor
research: expected utility, prospect-theory value, extrapolation-driven value
premia, and system feedback/crowding risk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def normalize_probabilities(probabilities: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(probabilities), dtype=float)
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative.")
    total = p.sum()
    if total <= EPS:
        raise ValueError("probabilities must sum to a positive value.")
    return p / total


def bernoulli_log_utility(wealth: float | np.ndarray) -> np.ndarray:
    """Bernoulli-style concave utility of wealth."""
    value = np.asarray(wealth, dtype=float)
    if np.any(value <= 0):
        raise ValueError("wealth must be positive for log utility.")
    return np.log(value)


def expected_log_utility(
    initial_wealth: float,
    outcomes: Iterable[float],
    probabilities: Iterable[float],
) -> float:
    """Expected log utility for risky wealth changes."""
    p = normalize_probabilities(probabilities)
    final_wealth = initial_wealth + np.asarray(list(outcomes), dtype=float)
    return float(np.sum(p * bernoulli_log_utility(final_wealth)))


def certainty_equivalent_log(
    initial_wealth: float,
    outcomes: Iterable[float],
    probabilities: Iterable[float],
) -> float:
    """Certainty-equivalent gain under log utility."""
    expected_u = expected_log_utility(initial_wealth, outcomes, probabilities)
    return float(np.exp(expected_u) - initial_wealth)


def prospect_value_function(
    outcomes: Iterable[float],
    alpha: float = 0.88,
    beta: float = 0.88,
    loss_aversion: float = 2.25,
) -> np.ndarray:
    """Prospect-theory value for gains/losses relative to a reference point."""
    x = np.asarray(list(outcomes), dtype=float)
    values = np.zeros_like(x, dtype=float)
    gain_mask = x >= 0
    loss_mask = ~gain_mask
    values[gain_mask] = x[gain_mask] ** alpha
    values[loss_mask] = -loss_aversion * ((-x[loss_mask]) ** beta)
    return values


def tk_probability_weight(probabilities: Iterable[float], gamma: float = 0.61) -> np.ndarray:
    """Tversky-Kahneman style inverse-S probability weighting."""
    p = np.clip(np.asarray(list(probabilities), dtype=float), EPS, 1.0 - EPS)
    numerator = p**gamma
    denominator = (p**gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)
    return numerator / denominator


def prospect_score(
    outcomes: Iterable[float],
    probabilities: Iterable[float],
    alpha: float = 0.88,
    beta: float = 0.88,
    loss_aversion: float = 2.25,
    gamma_gain: float = 0.61,
    gamma_loss: float = 0.69,
) -> float:
    """Approximate prospect value using separate gain/loss decision weights.

    This is a simple discrete-prospect research score, not a full cumulative
    prospect-theory implementation.
    """
    x = np.asarray(list(outcomes), dtype=float)
    p = normalize_probabilities(probabilities)
    values = prospect_value_function(x, alpha=alpha, beta=beta, loss_aversion=loss_aversion)
    weights = np.where(x >= 0, tk_probability_weight(p, gamma_gain), tk_probability_weight(p, gamma_loss))
    return float(np.sum(values * weights))


def _zscore_by_date(df: pd.DataFrame, date_col: str, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        target = f"{col}_z"

        def zscore(s: pd.Series) -> pd.Series:
            values = pd.to_numeric(s, errors="coerce")
            std = values.std(ddof=1)
            if pd.isna(std) or std == 0:
                return values * 0
            return (values - values.mean()) / std

        out[target] = out.groupby(date_col)[col].transform(zscore)
    return out


def lsv_value_glamour_score(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Iterable[str],
    past_growth_cols: Iterable[str],
    output_col: str = "lsv_value_score",
) -> pd.DataFrame:
    """Score stocks using the LSV value-vs-glamour interpretation.

    Higher value metrics and lower extrapolated past growth imply a higher
    contrarian value score.
    """
    value_cols = list(value_cols)
    growth_cols = list(past_growth_cols)
    out = _zscore_by_date(df, date_col, [*value_cols, *growth_cols])
    value_score = out[[f"{col}_z" for col in value_cols]].mean(axis=1)
    growth_score = out[[f"{col}_z" for col in growth_cols]].mean(axis=1)
    out[output_col] = value_score - growth_score
    out[f"{output_col}_value_component"] = value_score
    out[f"{output_col}_extrapolation_component"] = -growth_score
    return out


def assign_quantile_bucket(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    buckets: int = 5,
    output_col: str = "bucket",
) -> pd.DataFrame:
    """Assign date-wise quantile bucket by score."""
    out = df.copy()
    out[output_col] = np.nan
    for _, group in out.dropna(subset=[score_col]).groupby(date_col):
        if group.shape[0] < buckets:
            continue
        try:
            labels = pd.qcut(group[score_col].rank(method="first"), buckets, labels=False) + 1
        except ValueError:
            continue
        out.loc[group.index, output_col] = labels.astype(int)
    return out


def extrapolation_error_summary(
    df: pd.DataFrame,
    bucket_col: str,
    past_growth_col: str,
    expected_growth_col: str,
    future_growth_col: str,
) -> pd.DataFrame:
    """Compare past, expected, and realized future growth by value/glamour bucket."""
    required = [bucket_col, past_growth_col, expected_growth_col, future_growth_col]
    clean = df.dropna(subset=required).copy()
    if clean.empty:
        return pd.DataFrame()
    out = (
        clean.groupby(bucket_col)[[past_growth_col, expected_growth_col, future_growth_col]]
        .mean()
        .rename(
            columns={
                past_growth_col: "past_growth_mean",
                expected_growth_col: "expected_growth_mean",
                future_growth_col: "future_growth_mean",
            }
        )
        .reset_index()
    )
    out["expectation_error"] = out["expected_growth_mean"] - out["future_growth_mean"]
    return out


def downside_state_performance(
    returns_df: pd.DataFrame,
    date_col: str,
    strategy_cols: Iterable[str],
    market_col: str,
    tail_count: int = 25,
) -> pd.DataFrame:
    """Evaluate strategy returns in worst/best market states."""
    strategies = list(strategy_cols)
    clean = returns_df.dropna(subset=[market_col, *strategies]).sort_values(market_col).copy()
    if clean.empty:
        return pd.DataFrame()
    n = min(tail_count, max(1, clean.shape[0] // 4))
    labels = pd.Series("middle", index=clean.index)
    labels.loc[clean.index[:n]] = "worst"
    labels.loc[clean.index[-n:]] = "best"
    clean["_state"] = labels
    rows = []
    for state, group in clean.groupby("_state"):
        row = {"state": state, "n_periods": int(group.shape[0]), "market_return": float(group[market_col].mean())}
        for col in strategies:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_win_rate"] = float((group[col] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("state")


def synchronization_risk_score(
    df: pd.DataFrame,
    date_col: str,
    leverage_col: str | None = None,
    crowding_col: str | None = None,
    correlation_col: str | None = None,
    liquidity_stress_col: str | None = None,
    volatility_col: str | None = None,
    output_col: str = "system_feedback_risk",
) -> pd.DataFrame:
    """System feedback risk inspired by liquidity/crowding amplification."""
    out = df.sort_values(date_col).copy()
    cols = [col for col in [leverage_col, crowding_col, correlation_col, liquidity_stress_col, volatility_col] if col]
    if not cols:
        raise ValueError("At least one risk column is required.")
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(ddof=1)
        out[f"_{col}_z"] = (values - values.mean()) / std if std and not pd.isna(std) else 0.0
    out[output_col] = out[[f"_{col}_z" for col in cols]].mean(axis=1)
    return out.drop(columns=[f"_{col}_z" for col in cols])


def behavioral_factor_checklist() -> pd.DataFrame:
    """Return a checklist for converting behavioral hypotheses into factors."""
    rows = [
        ("reference_point", "What is the investor's reference level: cost, high watermark, index, or peer?"),
        ("loss_aversion", "Does selling pressure or holding behavior change after losses?"),
        ("extrapolation", "Are investors overpricing recent winners or extrapolated growth?"),
        ("attention", "Is the signal driven by media, analyst coverage, search, or trading attention?"),
        ("limits_to_arbitrage", "Can the mispricing persist because of short-sale, funding, career, or tracking-error limits?"),
        ("crowding", "Could many agents reacting to the same price/liquidity signal amplify shocks?"),
        ("implementation", "Does the long leg work after costs, or is the effect mostly from an unavailable short leg?"),
        ("falsification", "Does the signal survive neutralization, downside-state tests, and sample splits?"),
    ]
    return pd.DataFrame(rows, columns=["check", "question"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.checklist:
        result = behavioral_factor_checklist()
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            print(result)


if __name__ == "__main__":
    main()
