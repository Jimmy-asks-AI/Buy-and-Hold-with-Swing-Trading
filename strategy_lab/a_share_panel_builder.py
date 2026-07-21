#!/usr/bin/env python
"""Build point-in-time A-share research panels from market and fundamental tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import a_share_low_cost_factor_builder as lowcost
import csv_io
import real_data_adapter


def _asof_merge_by_asset(
    left: pd.DataFrame,
    right: pd.DataFrame,
    asset_col: str,
    left_date_col: str,
    right_date_col: str,
) -> pd.DataFrame:
    if right.empty:
        return left
    l = left.copy()
    r = right.copy()
    if asset_col not in l.columns or asset_col not in r.columns:
        raise ValueError(f"Both tables must include {asset_col} for point-in-time as-of merge.")
    l[left_date_col] = pd.to_datetime(l[left_date_col], errors="coerce")
    r[right_date_col] = pd.to_datetime(r[right_date_col], errors="coerce")
    if l[left_date_col].isna().any():
        raise ValueError(f"left table has invalid {left_date_col} values.")
    if r[right_date_col].isna().any():
        raise ValueError(f"right table has invalid {right_date_col} values.")
    frames = []
    right_groups = {asset: group.sort_values(right_date_col) for asset, group in r.groupby(asset_col, sort=False)}
    for asset, left_group in l.groupby(asset_col, sort=False):
        right_group = right_groups.get(asset)
        if right_group is None or right_group.empty:
            frames.append(left_group.copy())
            continue
        frames.append(
            pd.merge_asof(
                left_group.sort_values(left_date_col),
                right_group.sort_values(right_date_col),
                left_on=left_date_col,
                right_on=right_date_col,
                direction="backward",
                allow_exact_matches=True,
            )
        )
    merged = pd.concat(frames, ignore_index=True) if frames else l
    if f"{asset_col}_x" in merged.columns:
        merged[asset_col] = merged[f"{asset_col}_x"]
        drop_cols = [c for c in [f"{asset_col}_x", f"{asset_col}_y"] if c in merged.columns]
        merged = merged.drop(columns=drop_cols)
    return merged


def build_point_in_time_panel(
    market: pd.DataFrame,
    financial: pd.DataFrame | None = None,
    industry: pd.DataFrame | None = None,
    date_col: str = "date",
    asset_col: str = "asset",
    financial_available_col: str = "financial_available_at",
    industry_available_col: str = "industry_available_at",
    price_col: str = "adj_close",
    forward_horizon: int = 20,
) -> pd.DataFrame:
    """Build a point-in-time panel using backward as-of joins."""
    required = {date_col, asset_col, price_col}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market table missing columns: {sorted(missing)}")
    panel = market.copy()
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    if panel[date_col].isna().any():
        raise ValueError(f"market table has invalid {date_col} values.")
    duplicate_market = int(panel.duplicated([date_col, asset_col]).sum())
    if duplicate_market:
        raise ValueError(f"market table has duplicated date-asset rows: {duplicate_market}")
    if financial is not None and not financial.empty:
        if financial_available_col not in financial.columns:
            raise ValueError(f"financial table missing {financial_available_col}")
        panel = _asof_merge_by_asset(panel, financial, asset_col, date_col, financial_available_col)
    if industry is not None and not industry.empty:
        if industry_available_col not in industry.columns:
            raise ValueError(f"industry table missing {industry_available_col}")
        panel = _asof_merge_by_asset(panel, industry, asset_col, date_col, industry_available_col)
    if "list_date" in panel.columns:
        panel["listed_days"] = (panel[date_col] - pd.to_datetime(panel["list_date"], errors="coerce")).dt.days
    if "market_cap" in panel.columns and "log_mkt_cap" not in panel.columns:
        panel["log_mkt_cap"] = np.log(pd.to_numeric(panel["market_cap"], errors="coerce").replace(0.0, pd.NA))
    panel = derive_tradeable_status(panel)
    panel = lowcost.add_forward_return(panel, date_col=date_col, asset_col=asset_col, price_col=price_col, horizon=forward_horizon)
    return panel


def derive_tradeable_status(panel: pd.DataFrame) -> pd.DataFrame:
    """Derive a conservative is_tradeable flag when source state columns exist."""
    out = panel.copy()
    if "is_tradeable" in out.columns:
        out["is_tradeable"] = csv_io.coerce_bool_series(out["is_tradeable"], default=True).fillna(True).astype(bool)
        return out
    state_cols = [col for col in ["is_suspended", "is_limit_up", "is_limit_down", "is_st"] if col in out.columns]
    if not state_cols:
        return out
    tradeable = pd.Series(True, index=out.index)
    for col in state_cols:
        flag = csv_io.coerce_bool_series(out[col], default=False).fillna(False).astype(bool)
        tradeable = tradeable & ~flag
    out["is_tradeable"] = tradeable
    return out


def split_synthetic_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_cols = [
        "date",
        "asset",
        "adj_close",
        "market_cap",
        "enterprise_value",
        "turnover",
        "amount",
    ]
    financial_cols = [
        "asset",
        "financial_available_at",
        "total_equity",
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "net_profit_ttm",
        "revenue_ttm",
        "gross_profit_ttm",
        "operating_profit_ttm",
        "operating_cashflow_ttm",
        "free_cashflow_ttm",
        "ebitda_ttm",
        "interest_expense_ttm",
        "cash_dividend_ttm",
        "capex_ttm",
        "net_equity_issuance_ttm",
        "dividend_available_at",
    ]
    industry_cols = ["asset", "industry_available_at", "industry"]
    synthetic = panel.copy()
    synthetic["industry_available_at"] = synthetic["date"]
    return synthetic[market_cols], synthetic[financial_cols].drop_duplicates(), synthetic[industry_cols].drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-csv")
    parser.add_argument("--financial-csv")
    parser.add_argument("--industry-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-demo", action="store_true")
    parser.add_argument("--mapping-csv", default=str(real_data_adapter.DEFAULT_MAPPING))
    parser.add_argument("--no-auto-map", action="store_true")
    args = parser.parse_args()
    if args.synthetic_demo:
        raw = lowcost.make_synthetic_low_cost_panel()
        market, financial, industry = split_synthetic_tables(raw)
    else:
        if not args.market_csv:
            raise ValueError("Provide --market-csv or use --synthetic-demo.")
        market = csv_io.read_csv_robust(args.market_csv)
        financial = csv_io.read_csv_robust(args.financial_csv) if args.financial_csv else None
        industry = csv_io.read_csv_robust(args.industry_csv) if args.industry_csv else None
        if not args.no_auto_map:
            mapping = real_data_adapter.load_mapping(args.mapping_csv)
            market, _ = real_data_adapter.canonicalize_table(market, mapping, "market")
            if financial is not None:
                financial, _ = real_data_adapter.canonicalize_table(financial, mapping, "financial")
            if industry is not None:
                industry, _ = real_data_adapter.canonicalize_table(industry, mapping, "industry", strict_required=False)
    panel = build_point_in_time_panel(market, financial=financial, industry=industry)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False, encoding="utf-8-sig")
    print({"rows": int(panel.shape[0]), "assets": int(panel["asset"].nunique()), "dates": int(pd.to_datetime(panel["date"]).nunique())})


if __name__ == "__main__":
    main()
