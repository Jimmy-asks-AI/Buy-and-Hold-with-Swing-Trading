#!/usr/bin/env python
"""Rank IC / Spearman factor effectiveness analysis.

Input must be a cross-sectional panel with date, asset, future return, and one
or more factor columns. The script outputs Rank IC by date, rolling statistics,
and simple factor-selection signals.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional
    stats = None


def normal_two_sided_pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def t_two_sided_pvalue(t_stat: float, df: int) -> float:
    if stats is not None and df > 0:
        return float(2 * stats.t.sf(abs(t_stat), df))
    return normal_two_sided_pvalue(t_stat)


def spearman_one(group: pd.DataFrame, factor_col: str, return_col: str) -> tuple[float, float | None, int]:
    clean = group[[factor_col, return_col]].dropna()
    n = clean.shape[0]
    if n < 3:
        return float("nan"), None, n
    if stats is not None:
        result = stats.spearmanr(clean[factor_col], clean[return_col])
        return float(result.correlation), float(result.pvalue), n
    corr = clean[factor_col].rank().corr(clean[return_col].rank())
    return float(corr), None, n


def rank_ic_by_date(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    factor_cols: list[str],
    min_assets: int = 30,
) -> pd.DataFrame:
    rows = []
    for date, group in df.groupby(date_col):
        for factor_col in factor_cols:
            rank_ic, p_value, n = spearman_one(group, factor_col, return_col)
            if n < min_assets:
                continue
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "rank_ic": rank_ic,
                    "p_value": p_value,
                    "n_assets": n,
                }
            )
    return pd.DataFrame(rows)


def add_rolling_stats(ic_df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    if ic_df.empty:
        return ic_df.copy()
    out = ic_df.sort_values(["factor", "date"]).copy()
    grouped = out.groupby("factor", group_keys=False)["rank_ic"]
    out["rolling_mean"] = grouped.transform(lambda s: s.rolling(window, min_periods=max(3, window // 2)).mean())
    out["rolling_std"] = grouped.transform(lambda s: s.rolling(window, min_periods=max(3, window // 2)).std(ddof=1))
    out["rolling_count"] = grouped.transform(lambda s: s.rolling(window, min_periods=max(3, window // 2)).count())
    out["rolling_icir"] = out["rolling_mean"] / out["rolling_std"]
    out["rolling_t"] = out["rolling_mean"] / out["rolling_std"] * out["rolling_count"].pow(0.5)
    out["rolling_p"] = [
        t_two_sided_pvalue(t, int(n) - 1) if pd.notna(t) and pd.notna(n) and n > 1 else pd.NA
        for t, n in zip(out["rolling_t"], out["rolling_count"])
    ]
    return out


def factor_selection(
    rolling_df: pd.DataFrame,
    min_abs_mean: float = 0.02,
    max_p_value: float = 0.05,
) -> pd.DataFrame:
    if rolling_df.empty:
        return rolling_df.copy()
    out = rolling_df.copy()
    out["selected_by_mean"] = out["rolling_mean"].abs() >= min_abs_mean
    out["selected_by_p"] = out["rolling_p"].astype("float64") <= max_p_value
    out["selected"] = out["selected_by_mean"] & out["selected_by_p"]
    out["score_direction"] = out["rolling_mean"].apply(lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) and x < 0 else 0))
    return out


def summary_table(ic_df: pd.DataFrame) -> pd.DataFrame:
    if ic_df.empty:
        return pd.DataFrame()
    rows = []
    for factor, group in ic_df.groupby("factor"):
        series = group["rank_ic"].dropna()
        if series.empty:
            continue
        std = series.std(ddof=1)
        rows.append(
            {
                "factor": factor,
                "mean_rank_ic": float(series.mean()),
                "std_rank_ic": float(std),
                "icir": float(series.mean() / std) if std else pd.NA,
                "win_rate": float((series > 0).mean()),
                "count": int(series.shape[0]),
                "mean_abs_rank_ic": float(series.abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_rank_ic", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--return-col", required=True)
    parser.add_argument("--factor-cols", required=True, help="Comma-separated factor columns.")
    parser.add_argument("--min-assets", type=int, default=30)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--min-abs-mean", type=float, default=0.02)
    parser.add_argument("--max-p-value", type=float, default=0.05)
    parser.add_argument("--output-dir", default="rank_ic_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    factor_cols = [col.strip() for col in args.factor_cols.split(",") if col.strip()]
    missing = [col for col in [args.date_col, args.asset_col, args.return_col, *factor_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ic_df = rank_ic_by_date(df, args.date_col, args.return_col, factor_cols, min_assets=args.min_assets)
    rolling_df = add_rolling_stats(ic_df, window=args.window)
    selected_df = factor_selection(rolling_df, min_abs_mean=args.min_abs_mean, max_p_value=args.max_p_value)
    summary_df = summary_table(ic_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(output_dir / "rank_ic_by_date.csv", index=False, encoding="utf-8-sig")
    rolling_df.to_csv(output_dir / "rolling_rank_ic.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "factor_selection.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df)
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
