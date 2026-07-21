#!/usr/bin/env python
"""Reusable factor evaluation template.

This is a factor validation tool, not a full strategy backtest. It checks
whether a factor has cross-sectional information about future returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_forward_returns(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    price_col: str,
    horizon: int,
    output_col: str = "fwd_return",
) -> pd.DataFrame:
    """Add forward returns per asset.

    Forward return is a label for evaluation. Do not use it in factor formulas.
    """
    out = df.sort_values([asset_col, date_col]).copy()
    future_price = out.groupby(asset_col, group_keys=False)[price_col].shift(-horizon)
    out[output_col] = future_price / out[price_col] - 1
    return out


def winsorize_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Winsorize a factor within each date."""
    out = df.copy()
    target = output_col or value_col

    def clip_one(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        return s.clip(lower=lo, upper=hi)

    out[target] = out.groupby(date_col)[value_col].transform(clip_one)
    return out


def zscore_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Z-score a factor within each date."""
    out = df.copy()
    target = output_col or value_col

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=1)
        if pd.isna(std) or std == 0:
            return s * 0
        return (s - s.mean()) / std

    out[target] = out.groupby(date_col)[value_col].transform(zscore)
    return out


def date_ic_table(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str = "fwd_return",
    min_assets: int = 10,
) -> pd.DataFrame:
    """Compute Pearson IC and Rank IC by date."""
    rows = []
    for date, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col):
        if group.shape[0] < min_assets:
            continue
        factor = group[factor_col]
        fwd_ret = group[return_col]
        rows.append(
            {
                "date": date,
                "n_assets": int(group.shape[0]),
                "ic": float(factor.corr(fwd_ret)),
                "rank_ic": float(factor.rank().corr(fwd_ret.rank())),
            }
        )
    return pd.DataFrame(rows)


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC stability."""
    rows = []
    for col in ["ic", "rank_ic"]:
        series = ic_df[col].dropna()
        if series.empty:
            rows.append({"metric": col, "mean": pd.NA, "std": pd.NA, "ir": pd.NA, "win_rate": pd.NA})
            continue
        std = series.std(ddof=1)
        rows.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "std": float(std) if not pd.isna(std) else pd.NA,
                "ir": float(series.mean() / std) if std and not pd.isna(std) else pd.NA,
                "win_rate": float((series > 0).mean()),
                "count": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def quantile_group_returns(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str = "fwd_return",
    groups: int = 5,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Compute equal-weight forward returns by factor quantile group."""
    rows = []
    clean = df.dropna(subset=[factor_col, return_col]).copy()
    for date, group in clean.groupby(date_col):
        if group.shape[0] < max(min_assets, groups):
            continue
        try:
            group = group.assign(
                factor_group=pd.qcut(group[factor_col].rank(method="first"), groups, labels=False) + 1
            )
        except ValueError:
            continue
        grouped = group.groupby("factor_group")[return_col].mean()
        for factor_group, value in grouped.items():
            rows.append({"date": date, "factor_group": int(factor_group), "return": float(value)})
    return pd.DataFrame(rows)


def group_summary(group_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize average return by factor group."""
    if group_df.empty:
        return pd.DataFrame()
    summary = group_df.groupby("factor_group")["return"].agg(["mean", "std", "count"]).reset_index()
    summary["mean_rank"] = summary["mean"].rank(method="dense")
    return summary


def evaluate_factor(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    factor_col: str,
    price_col: str | None,
    return_col: str | None,
    horizon: int,
    groups: int,
    min_assets: int,
    winsorize: bool,
    zscore: bool,
) -> dict[str, pd.DataFrame]:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    factor_used = factor_col
    if winsorize:
        factor_used = f"{factor_used}_winsor"
        out = winsorize_by_date(out, date_col, factor_col, output_col=factor_used)
    if zscore:
        z_col = f"{factor_used}_z"
        out = zscore_by_date(out, date_col, factor_used, output_col=z_col)
        factor_used = z_col

    if return_col:
        out["fwd_return"] = out[return_col]
    elif price_col:
        out = add_forward_returns(out, date_col, asset_col, price_col, horizon)
    else:
        raise ValueError("Provide either --return-col or --price-col.")

    ic_df = date_ic_table(out, date_col, factor_used, min_assets=min_assets)
    group_df = quantile_group_returns(out, date_col, factor_used, groups=groups, min_assets=min_assets)
    return {
        "prepared_data": out,
        "ic_by_date": ic_df,
        "ic_summary": ic_summary(ic_df),
        "group_returns": group_df,
        "group_summary": group_summary(group_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--price-col")
    parser.add_argument("--return-col")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--winsorize", action="store_true")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--output-dir", default="factor_evaluation_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    results = evaluate_factor(
        df=df,
        date_col=args.date_col,
        asset_col=args.asset_col,
        factor_col=args.factor_col,
        price_col=args.price_col,
        return_col=args.return_col,
        horizon=args.horizon,
        groups=args.groups,
        min_assets=args.min_assets,
        winsorize=args.winsorize,
        zscore=args.zscore,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(results["ic_summary"])
    print(results["group_summary"])
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
