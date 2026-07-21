#!/usr/bin/env python
"""Reusable performance metrics for quant research backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def annual_return(nav: pd.Series, dates: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / TRADING_DAYS)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    return float((1 + total) ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std(ddof=1) * (periods**0.5))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    rf_period = (1 + risk_free_annual) ** (1 / periods) - 1
    excess = clean - rf_period
    vol = clean.std(ddof=1) * (periods**0.5)
    if vol == 0 or pd.isna(vol):
        return float("nan")
    return float(excess.mean() * periods / vol)


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    rf_period = (1 + risk_free_annual) ** (1 / periods) - 1
    excess = clean - rf_period
    downside = excess[excess < 0].std(ddof=1) * (periods**0.5)
    if downside == 0 or pd.isna(downside):
        return float("nan")
    return float(excess.mean() * periods / downside)


def summarize_nav(
    df: pd.DataFrame,
    date_col: str = "date",
    nav_col: str = "nav",
    return_col: str | None = None,
    benchmark_nav_col: str | None = None,
    risk_free_annual: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)
    nav = out[nav_col].astype(float)
    returns = out[return_col].astype(float) if return_col else nav.pct_change()
    ann_ret = annual_return(nav, out[date_col])
    ann_vol = annual_volatility(returns)
    mdd = max_drawdown(nav)
    row = {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "max_drawdown": mdd,
        "sharpe": sharpe_ratio(returns, risk_free_annual=risk_free_annual),
        "sortino": sortino_ratio(returns, risk_free_annual=risk_free_annual),
        "calmar": float(ann_ret / abs(mdd)) if mdd else float("nan"),
        "win_rate": float((returns.dropna() > 0).mean()),
        "n_periods": int(returns.dropna().shape[0]),
    }
    if benchmark_nav_col:
        bench = out[benchmark_nav_col].astype(float)
        bench_returns = bench.pct_change()
        excess = returns - bench_returns
        row["benchmark_total_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)
        row["excess_annual_return"] = annual_return((1 + excess.fillna(0)).cumprod(), out[date_col])
        row["tracking_error"] = annual_volatility(excess)
        te = row["tracking_error"]
        row["information_ratio"] = float(row["excess_annual_return"] / te) if te else float("nan")
    return pd.DataFrame([row])


def yearly_returns(df: pd.DataFrame, date_col: str = "date", nav_col: str = "nav") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)
    out["year"] = out[date_col].dt.year
    rows = []
    for year, group in out.groupby("year"):
        rows.append({"year": int(year), "return": float(group[nav_col].iloc[-1] / group[nav_col].iloc[0] - 1)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--nav-col", default="nav")
    parser.add_argument("--return-col")
    parser.add_argument("--benchmark-nav-col")
    parser.add_argument("--output-dir", default="performance_metrics_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    summary = summarize_nav(
        df,
        date_col=args.date_col,
        nav_col=args.nav_col,
        return_col=args.return_col,
        benchmark_nav_col=args.benchmark_nav_col,
    )
    yearly = yearly_returns(df, date_col=args.date_col, nav_col=args.nav_col)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    print(summary)
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
