#!/usr/bin/env python
"""Build first-wave low-cost A-share factors from market and financial panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import csv_io


EPS = 1e-12


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float).replace(0.0, np.nan)


def add_forward_return(
    df: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "adj_close",
    horizon: int = 20,
    output_col: str = "fwd_return",
) -> pd.DataFrame:
    """Add future return label. This label must not be used in factor formulas."""
    out = df.sort_values([asset_col, date_col]).copy()
    price = _num(out, price_col)
    out[output_col] = out.groupby(asset_col, group_keys=False)[price_col].shift(-horizon) / price - 1.0
    return out


def trailing_return(
    df: pd.DataFrame,
    asset_col: str,
    price_col: str,
    window: int,
    skip: int = 0,
) -> pd.Series:
    """Trailing return ending skip periods before current date."""
    price = _num(df, price_col)
    by_asset = df.assign(_price=price).groupby(asset_col)["_price"]
    end_price = by_asset.shift(skip)
    start_price = by_asset.shift(window + skip)
    return end_price / start_price - 1.0


def rolling_std_return(
    df: pd.DataFrame,
    asset_col: str,
    return_col: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = min_periods or max(5, window // 2)
    return (
        df.sort_values([asset_col, "date"])
        .groupby(asset_col)[return_col]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=min_periods).std())
    )


def rolling_mean(
    df: pd.DataFrame,
    asset_col: str,
    value_col: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = min_periods or max(5, window // 2)
    return (
        df.sort_values([asset_col, "date"])
        .groupby(asset_col)[value_col]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=min_periods).mean())
    )


def efficiency_ratio(
    df: pd.DataFrame,
    asset_col: str,
    price_col: str,
    window: int = 20,
) -> pd.Series:
    data = df.sort_values([asset_col, "date"]).copy()
    price = _num(data, price_col)
    change = price.groupby(data[asset_col]).diff(window).abs()
    path = price.groupby(data[asset_col]).diff().abs().groupby(data[asset_col]).transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 2)).sum()
    )
    return change / path.replace(0.0, np.nan)


def detrended_price_oscillator(
    df: pd.DataFrame,
    asset_col: str,
    price_col: str,
    window: int = 20,
) -> pd.Series:
    data = df.sort_values([asset_col, "date"]).copy()
    price = _num(data, price_col)
    ma = price.groupby(data[asset_col]).transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).mean())
    return price / ma.replace(0.0, np.nan) - 1.0


def build_low_cost_factors(
    panel: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Build low-cost candidate factors aligned with a_share_factor_registry_v0.csv."""
    out = panel.sort_values([asset_col, date_col]).copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if date_col != "date":
        out["date"] = out[date_col]

    market_cap = _num(out, "market_cap")
    enterprise_value = _num(out, "enterprise_value")
    total_equity = _num(out, "total_equity")
    total_assets = _num(out, "total_assets")
    total_liabilities = _num(out, "total_liabilities")
    current_assets = _num(out, "current_assets")
    current_liabilities = _num(out, "current_liabilities")
    net_profit = _num(out, "net_profit_ttm")
    revenue = _num(out, "revenue_ttm")
    gross_profit = _num(out, "gross_profit_ttm")
    operating_profit = _num(out, "operating_profit_ttm")
    operating_cf = _num(out, "operating_cashflow_ttm")
    free_cf = _num(out, "free_cashflow_ttm")
    ebitda = _num(out, "ebitda_ttm")
    interest_expense = _num(out, "interest_expense_ttm").abs()
    dividend = _num(out, "cash_dividend_ttm")
    capex = _num(out, "capex_ttm").abs()

    out["log_mkt_cap"] = np.log(market_cap.replace(0.0, np.nan))
    out["value_ep_ttm_raw"] = safe_divide(net_profit, market_cap)
    out["value_bp_raw"] = safe_divide(total_equity, market_cap)
    out["value_sp_ttm_raw"] = safe_divide(revenue, market_cap)
    out["value_ocfp_ttm_raw"] = safe_divide(operating_cf, market_cap)
    out["value_fcfp_ttm_raw"] = safe_divide(free_cf, market_cap)
    out["value_ebitda_ev_raw"] = safe_divide(ebitda, enterprise_value)
    out["dividend_yield_ttm_raw"] = safe_divide(dividend, market_cap)

    out["quality_roe_ttm_raw"] = safe_divide(net_profit, total_equity)
    out["quality_roa_ttm_raw"] = safe_divide(net_profit, total_assets)
    out["quality_gross_margin_ttm_raw"] = safe_divide(gross_profit, revenue)
    out["quality_operating_margin_ttm_raw"] = safe_divide(operating_profit, revenue)
    out["quality_cash_earnings_raw"] = safe_divide(operating_cf, net_profit.abs())
    out["quality_accruals_raw"] = safe_divide(net_profit - operating_cf, total_assets)

    out["growth_revenue_yoy_raw"] = safe_divide(revenue, out.groupby(asset_col)["revenue_ttm"].shift(12)) - 1.0
    out["growth_profit_yoy_raw"] = safe_divide(net_profit, out.groupby(asset_col)["net_profit_ttm"].shift(12)) - 1.0
    out["growth_sales_acceleration_raw"] = out["growth_revenue_yoy_raw"] - out.groupby(asset_col)["growth_revenue_yoy_raw"].shift(12)

    out["investment_asset_growth_raw"] = safe_divide(total_assets, out.groupby(asset_col)["total_assets"].shift(12)) - 1.0
    out["investment_capex_growth_raw"] = safe_divide(capex, out.groupby(asset_col)["capex_ttm"].shift(12).abs()) - 1.0
    out["investment_net_issuance_raw"] = _num(out, "net_equity_issuance_ttm") / market_cap.replace(0.0, np.nan)

    out["leverage_debt_to_assets_raw"] = safe_divide(total_liabilities, total_assets)
    out["leverage_interest_coverage_raw"] = safe_divide(operating_profit, interest_expense)
    out["leverage_current_ratio_raw"] = safe_divide(current_assets, current_liabilities)

    price = _num(out, price_col)
    out["_daily_return"] = price.groupby(out[asset_col]).pct_change()
    out["momentum_6m_skip1m_raw"] = trailing_return(out, asset_col, price_col, window=126, skip=21)
    out["momentum_12m_skip1m_raw"] = trailing_return(out, asset_col, price_col, window=252, skip=21)
    out["reversal_1m_raw"] = -trailing_return(out, asset_col, price_col, window=21, skip=0)
    out["reversal_5d_raw"] = -trailing_return(out, asset_col, price_col, window=5, skip=0)

    out["liquidity_turnover_raw"] = rolling_mean(out, asset_col, "turnover", 21)
    out["liquidity_turnover_volatility_raw"] = (
        out.sort_values([asset_col, date_col])
        .groupby(asset_col)["turnover"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(21, min_periods=10).std())
    )
    out["liquidity_amount_size_residual_raw"] = np.log1p(_num(out, "amount")) - np.log(market_cap.replace(0.0, np.nan))

    out["volatility_total_raw"] = rolling_std_return(out, asset_col, "_daily_return", 21)
    downside = out["_daily_return"].where(out["_daily_return"] < 0, 0.0)
    out["_downside_return"] = downside
    out["volatility_downside_raw"] = rolling_std_return(out, asset_col, "_downside_return", 21)

    out["technical_efficiency_ratio_raw"] = efficiency_ratio(out, asset_col, price_col, 20)
    out["technical_dpo_raw"] = detrended_price_oscillator(out, asset_col, price_col, 20)

    out.drop(columns=[c for c in ["_daily_return", "_downside_return"] if c in out.columns], inplace=True)
    return out


def factor_coverage_report(
    panel: pd.DataFrame,
    factor_cols: Iterable[str],
    date_col: str = "date",
) -> pd.DataFrame:
    rows = []
    for col in factor_cols:
        if col not in panel.columns:
            rows.append({"factor_col": col, "exists": False, "coverage": 0.0, "date_coverage_min": np.nan})
            continue
        coverage_by_date = panel.groupby(date_col)[col].apply(lambda s: pd.to_numeric(s, errors="coerce").notna().mean())
        rows.append(
            {
                "factor_col": col,
                "exists": True,
                "coverage": float(pd.to_numeric(panel[col], errors="coerce").notna().mean()),
                "date_coverage_min": float(coverage_by_date.min()) if not coverage_by_date.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_synthetic_low_cost_panel(n_dates: int = 320, n_assets: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i:04d}" for i in range(n_assets)]
    rows = []
    for asset_idx, asset in enumerate(assets):
        price = 20.0 + rng.normal()
        equity = 5e9 * np.exp(rng.normal(0, 0.4))
        assets_total = equity * rng.uniform(1.2, 3.0)
        for date in dates:
            price *= float(np.exp(rng.normal(0.0002, 0.02)))
            market_cap = price * rng.uniform(1e8, 3e8)
            revenue = market_cap * rng.uniform(0.25, 1.4)
            profit_margin = rng.uniform(0.03, 0.18)
            net_profit = revenue * profit_margin
            gross_profit = revenue * rng.uniform(0.15, 0.55)
            operating_profit = revenue * rng.uniform(0.05, 0.25)
            operating_cf = net_profit * rng.uniform(0.5, 1.5)
            free_cf = operating_cf - abs(rng.normal(0.0, 0.15 * max(operating_cf, 1.0)))
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "industry": f"industry_{asset_idx % 8}",
                    "adj_close": price,
                    "market_cap": market_cap,
                    "enterprise_value": market_cap + assets_total - equity,
                    "total_equity": equity,
                    "total_assets": assets_total,
                    "total_liabilities": assets_total - equity,
                    "current_assets": assets_total * rng.uniform(0.2, 0.7),
                    "current_liabilities": (assets_total - equity) * rng.uniform(0.15, 0.7),
                    "net_profit_ttm": net_profit,
                    "revenue_ttm": revenue,
                    "gross_profit_ttm": gross_profit,
                    "operating_profit_ttm": operating_profit,
                    "operating_cashflow_ttm": operating_cf,
                    "free_cashflow_ttm": free_cf,
                    "ebitda_ttm": operating_profit + rng.uniform(0.02, 0.08) * revenue,
                    "interest_expense_ttm": max(1.0, (assets_total - equity) * rng.uniform(0.005, 0.03)),
                    "cash_dividend_ttm": max(0.0, net_profit * rng.uniform(0.0, 0.5)),
                    "capex_ttm": revenue * rng.uniform(0.01, 0.15),
                    "net_equity_issuance_ttm": market_cap * rng.normal(0.0, 0.02),
                    "turnover": abs(rng.normal(0.02, 0.01)),
                    "amount": market_cap * abs(rng.normal(0.015, 0.01)),
                    "financial_available_at": date,
                    "dividend_available_at": date,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-output")
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args()
    if args.synthetic_demo:
        panel = make_synthetic_low_cost_panel()
    else:
        if not args.input:
            raise ValueError("Provide --input or use --synthetic-demo.")
        panel = csv_io.read_csv_robust(args.input)
    out = build_low_cost_factors(panel)
    out = add_forward_return(out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    factor_cols = [col for col in out.columns if col.endswith("_raw")]
    coverage = factor_coverage_report(out, factor_cols)
    if args.coverage_output:
        Path(args.coverage_output).parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.coverage_output, index=False, encoding="utf-8-sig")
    print({"rows": int(out.shape[0]), "factor_cols": len(factor_cols), "coverage_mean": float(coverage["coverage"].mean())})


if __name__ == "__main__":
    main()
