#!/usr/bin/env python
"""Capacity, liquidity, and implementation-cost diagnostics."""

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


def tradeable_by_participation(
    df: pd.DataFrame,
    amount_col: str,
    fund_size: float,
    target_trade_weight: float = 0.001,
    participation_rate: float = 0.05,
    output_prefix: str = "capacity",
) -> pd.DataFrame:
    """Flag tradeable stocks by order amount versus session turnover.

    Research examples use `5% * session amount >= 0.1% * fund size`.
    For weekly strategies the `amount_col` should usually be opening-half-hour
    amount; for monthly strategies it can be full-day amount.
    """
    if fund_size <= 0:
        raise ValueError("fund_size must be positive")
    if not 0 < target_trade_weight <= 1:
        raise ValueError("target_trade_weight must be in (0, 1]")
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    _require_columns(df, [amount_col])
    out = df.copy()
    out[f"{output_prefix}_order_amount"] = fund_size * target_trade_weight
    out[f"{output_prefix}_tradable_amount"] = out[amount_col].astype(float).clip(lower=0.0) * participation_rate
    out[f"{output_prefix}_tradeable"] = out[f"{output_prefix}_tradable_amount"] >= out[f"{output_prefix}_order_amount"]
    out[f"{output_prefix}_participation"] = out[f"{output_prefix}_order_amount"] / out[amount_col].replace(0, np.nan)
    return out


def trade_amount_from_weights(
    weights: pd.DataFrame,
    asset_col: str,
    old_weight_col: str,
    new_weight_col: str,
    fund_size: float,
    output_col: str = "trade_amount",
) -> pd.DataFrame:
    """Convert portfolio weight changes into currency trade amounts."""
    if fund_size <= 0:
        raise ValueError("fund_size must be positive")
    _require_columns(weights, [asset_col, old_weight_col, new_weight_col])
    out = weights.copy()
    out[output_col] = (out[new_weight_col].astype(float) - out[old_weight_col].astype(float)).abs() * fund_size
    return out


def total_cost_filter(
    df: pd.DataFrame,
    cost_cols: list[str],
    threshold: float = 0.01,
    output_col: str = "cost_tradeable",
) -> pd.DataFrame:
    """Flag stocks whose estimated implementation cost is below threshold."""
    _require_columns(df, cost_cols)
    out = df.copy()
    out["estimated_total_cost"] = out[cost_cols].astype(float).abs().sum(axis=1)
    out[output_col] = out["estimated_total_cost"] <= threshold
    return out


def order_book_liquidity_cost_flag(
    df: pd.DataFrame,
    order_amount_col: str,
    best_level_depth_col: str,
    output_col: str = "order_book_liquidity_cost",
) -> pd.DataFrame:
    """Flag orders large enough to consume the best bid or ask level."""
    _require_columns(df, [order_amount_col, best_level_depth_col])
    out = df.copy()
    out[output_col] = out[order_amount_col].astype(float) > out[best_level_depth_col].astype(float)
    return out


def estimate_linear_impact_coefficients(
    df: pd.DataFrame,
    date_col: str,
    session_return_col: str,
    big_net_buy_ratio_col: str,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Estimate cross-sectional big-order impact coefficient by date."""
    _require_columns(df, [date_col, session_return_col, big_net_buy_ratio_col])
    rows = []
    for date, group in df.groupby(date_col):
        clean = group[[session_return_col, big_net_buy_ratio_col]].dropna()
        if clean.shape[0] < min_obs:
            rows.append({"date": date, "impact_beta": np.nan, "obs": int(clean.shape[0])})
            continue
        x = clean[big_net_buy_ratio_col].astype(float)
        y = clean[session_return_col].astype(float)
        var_x = x.var(ddof=1)
        beta = np.nan if pd.isna(var_x) or var_x <= EPS else float(x.cov(y) / var_x)
        rows.append({"date": date, "impact_beta": beta, "obs": int(clean.shape[0])})
    return pd.DataFrame(rows)


def attach_trailing_impact_beta(
    beta_df: pd.DataFrame,
    date_col: str = "date",
    beta_col: str = "impact_beta",
    lookback: int = 21,
    min_periods: int = 10,
    output_col: str = "trailing_impact_beta",
) -> pd.DataFrame:
    """Use trailing mean beta shifted by one date for live-safe prediction."""
    _require_columns(beta_df, [date_col, beta_col])
    out = beta_df.sort_values(date_col).copy()
    out[output_col] = out[beta_col].astype(float).rolling(lookback, min_periods=min_periods).mean().shift(1)
    return out


def large_order_impact_cost(
    orders: pd.DataFrame,
    signed_order_amount_col: str,
    session_amount_col: str,
    impact_beta_col: str,
    output_col: str = "large_order_impact_cost",
) -> pd.DataFrame:
    """Estimate adverse cost from induced change in big-net-buy ratio."""
    _require_columns(orders, [signed_order_amount_col, session_amount_col, impact_beta_col])
    out = orders.copy()
    delta_ratio = out[signed_order_amount_col].astype(float) / out[session_amount_col].replace(0, np.nan)
    out["induced_big_net_buy_ratio"] = delta_ratio
    out[output_col] = (out[impact_beta_col].astype(float) * delta_ratio).abs()
    return out


def capacity_rank_ic(
    factor_data: pd.DataFrame,
    factor_col: str,
    forward_return_col: str,
    amount_col: str,
    fund_sizes: list[float],
    date_col: str | None = None,
    target_trade_weight: float = 0.001,
    participation_rate: float = 0.05,
) -> pd.DataFrame:
    """Compute rank IC after applying tradeability filters at each fund size."""
    required = [factor_col, forward_return_col, amount_col]
    if date_col:
        required.append(date_col)
    _require_columns(factor_data, required)
    rows = []
    for fund_size in fund_sizes:
        screened = tradeable_by_participation(
            factor_data,
            amount_col=amount_col,
            fund_size=fund_size,
            target_trade_weight=target_trade_weight,
            participation_rate=participation_rate,
        )
        screened = screened[screened["capacity_tradeable"]].copy()
        groups = screened.groupby(date_col) if date_col else [(None, screened)]
        ics = []
        obs = 0
        for date, group in groups:
            clean = group[[factor_col, forward_return_col]].dropna()
            if clean.shape[0] < 3:
                continue
            ic = clean[factor_col].rank().corr(clean[forward_return_col].rank())
            ics.append({"date": date, "rank_ic": float(ic), "obs": int(clean.shape[0])})
            obs += int(clean.shape[0])
        ic_df = pd.DataFrame(ics)
        rows.append(
            {
                "fund_size": fund_size,
                "mean_rank_ic": float(ic_df["rank_ic"].mean()) if not ic_df.empty else np.nan,
                "ic_count": int(ic_df.shape[0]),
                "obs": int(obs),
                "tradeable_ratio": float(screened.shape[0] / factor_data.shape[0]) if factor_data.shape[0] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--return-col", required=True)
    parser.add_argument("--amount-col", required=True)
    parser.add_argument("--fund-sizes", nargs="+", type=float, required=True)
    parser.add_argument("--date-col")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = capacity_rank_ic(
        df,
        factor_col=args.factor_col,
        forward_return_col=args.return_col,
        amount_col=args.amount_col,
        fund_sizes=args.fund_sizes,
        date_col=args.date_col,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
