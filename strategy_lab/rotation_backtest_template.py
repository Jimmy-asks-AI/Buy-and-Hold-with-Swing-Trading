#!/usr/bin/env python
"""Minimal long-only rotation backtest template.

Input: a daily panel with date, asset, adjusted price, and factor score.
Rule: rebalance monthly, buy top-N assets by factor, equal-weight, apply cost.

This is still a research template. Before using it for conclusions, audit data
availability, ETF listing dates, liquidity, benchmark, and execution assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    drawdown = nav / peak - 1
    return float(drawdown.min())


def performance_metrics(nav_df: pd.DataFrame, turnover_df: pd.DataFrame | None = None) -> pd.DataFrame:
    returns = nav_df["portfolio_return"].dropna()
    if returns.empty:
        return pd.DataFrame()
    total_return = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    years = max((nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days / 365.25, 1 / TRADING_DAYS)
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = returns.std(ddof=1) * (TRADING_DAYS**0.5)
    sharpe = annual_return / annual_vol if annual_vol else pd.NA
    avg_turnover = turnover_df["turnover"].mean() if turnover_df is not None and not turnover_df.empty else pd.NA
    return pd.DataFrame(
        [
            {
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "annual_vol": float(annual_vol) if not pd.isna(annual_vol) else pd.NA,
                "sharpe_no_rf": float(sharpe) if not pd.isna(sharpe) else pd.NA,
                "max_drawdown": max_drawdown(nav_df["nav"]),
                "avg_turnover": avg_turnover,
            }
        ]
    )


def month_end_rebalance_dates(dates: pd.Series) -> list[pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    return list(unique_dates.groupby(unique_dates.dt.to_period("M")).max())


def equal_weights(assets: list[str]) -> dict[str, float]:
    if not assets:
        return {}
    weight = 1.0 / len(assets)
    return {asset: weight for asset in assets}


def compute_turnover(old: dict[str, float], new: dict[str, float]) -> float:
    assets = set(old) | set(new)
    return sum(abs(new.get(asset, 0.0) - old.get(asset, 0.0)) for asset in assets)


def run_rotation_backtest(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    price_col: str,
    factor_col: str,
    top_n: int = 3,
    cost_bps: float = 10.0,
    min_factor_count: int = 3,
) -> dict[str, pd.DataFrame]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([asset_col, date_col])
    data["asset_return"] = data.groupby(asset_col)[price_col].pct_change()

    all_dates = pd.Series(data[date_col].drop_duplicates()).sort_values().reset_index(drop=True)
    rebalance_dates = month_end_rebalance_dates(all_dates)
    holdings_rows = []
    nav_rows = []
    turnover_rows = []
    prev_weights: dict[str, float] = {}
    nav = 1.0

    for idx, rebalance_date in enumerate(rebalance_dates):
        signal_panel = data[data[date_col] == rebalance_date].dropna(subset=[factor_col])
        if signal_panel.shape[0] < max(top_n, min_factor_count):
            continue
        selected = (
            signal_panel.sort_values(factor_col, ascending=False)
            .head(top_n)[asset_col]
            .astype(str)
            .tolist()
        )
        new_weights = equal_weights(selected)
        turnover = compute_turnover(prev_weights, new_weights)

        future_dates = all_dates[all_dates > rebalance_date]
        if future_dates.empty:
            break
        start_date = future_dates.iloc[0]
        if idx + 1 < len(rebalance_dates):
            end_date = rebalance_dates[idx + 1]
            hold_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
        else:
            hold_dates = all_dates[all_dates >= start_date]

        turnover_rows.append({"date": start_date, "turnover": turnover, "cost_bps": cost_bps})
        for asset, weight in new_weights.items():
            holdings_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "effective_date": start_date,
                    "asset": asset,
                    "weight": weight,
                    "factor": float(signal_panel.loc[signal_panel[asset_col].astype(str) == asset, factor_col].iloc[0]),
                }
            )

        first_day = True
        for date in hold_dates:
            day_panel = data[(data[date_col] == date) & (data[asset_col].astype(str).isin(selected))]
            weighted_returns = []
            for asset, weight in new_weights.items():
                asset_ret = day_panel.loc[day_panel[asset_col].astype(str) == asset, "asset_return"]
                if asset_ret.empty or pd.isna(asset_ret.iloc[0]):
                    continue
                weighted_returns.append(weight * float(asset_ret.iloc[0]))
            portfolio_return = sum(weighted_returns)
            if first_day and turnover:
                portfolio_return -= turnover * cost_bps / 10000.0
                first_day = False
            nav *= 1 + portfolio_return
            nav_rows.append(
                {
                    "date": date,
                    "portfolio_return": portfolio_return,
                    "nav": nav,
                    "n_selected": len(selected),
                    "n_return_available": len(weighted_returns),
                }
            )

        prev_weights = new_weights

    nav_df = pd.DataFrame(nav_rows)
    holdings_df = pd.DataFrame(holdings_rows)
    turnover_df = pd.DataFrame(turnover_rows)
    metrics_df = performance_metrics(nav_df, turnover_df) if not nav_df.empty else pd.DataFrame()
    return {
        "nav": nav_df,
        "holdings": holdings_df,
        "turnover": turnover_df,
        "metrics": metrics_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--price-col", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="rotation_backtest_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    results = run_rotation_backtest(
        df,
        date_col=args.date_col,
        asset_col=args.asset_col,
        price_col=args.price_col,
        factor_col=args.factor_col,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(results["metrics"])
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
