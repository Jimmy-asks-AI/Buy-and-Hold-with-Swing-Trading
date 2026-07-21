#!/usr/bin/env python
"""Calendar seasonality tools for factor timing and exposure schedules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional
    stats = None


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _t_pvalue(t_stat: float, df: int) -> float:
    if stats is not None and df > 0:
        return float(2.0 * stats.t.sf(abs(t_stat), df))
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _series_summary(values: pd.Series, periods_per_year: int) -> dict[str, float | int]:
    clean = values.astype(float).dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "annualized_mean": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "win_rate": np.nan,
        }
    std = clean.std(ddof=1)
    t_stat = clean.mean() / std * math.sqrt(clean.shape[0]) if std and not pd.isna(std) else np.nan
    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "annualized_mean": float(clean.mean() * periods_per_year),
        "std": float(std) if not pd.isna(std) else np.nan,
        "t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
        "p_value": _t_pvalue(float(t_stat), int(clean.shape[0] - 1)) if not pd.isna(t_stat) else np.nan,
        "win_rate": float((clean > 0).mean()),
    }


def month_premium_summary(
    premium_df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    premium_col: str,
    direction_col: str | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Summarize factor premium by calendar month.

    If direction_col is provided, premium is multiplied by that sign before
    summarizing, so positive values always mean the intended factor direction
    worked.
    """
    required = [date_col, factor_col, premium_col]
    if direction_col:
        required.append(direction_col)
    _require_columns(premium_df, required)
    out = premium_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["month"] = out[date_col].dt.month
    out["_oriented_premium"] = out[premium_col].astype(float)
    if direction_col:
        out["_oriented_premium"] = out["_oriented_premium"] * out[direction_col].astype(float)

    rows = []
    for (factor, month), group in out.groupby([factor_col, "month"]):
        row = {"factor": factor, "month": int(month)}
        row.update(_series_summary(group["_oriented_premium"], periods_per_year=periods_per_year))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["factor", "month"])


def month_bucket_regression(
    premium_df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    premium_col: str,
    month_buckets: dict[str, list[int]],
    direction_col: str | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Estimate dummy-month effects versus all other months."""
    required = [date_col, factor_col, premium_col]
    if direction_col:
        required.append(direction_col)
    _require_columns(premium_df, required)
    out = premium_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["_oriented_premium"] = out[premium_col].astype(float)
    if direction_col:
        out["_oriented_premium"] = out["_oriented_premium"] * out[direction_col].astype(float)

    rows = []
    for factor, group in out.groupby(factor_col):
        y = group["_oriented_premium"].astype(float)
        for bucket_name, months in month_buckets.items():
            dummy = group[date_col].dt.month.isin(months)
            in_bucket = y[dummy]
            out_bucket = y[~dummy]
            diff = in_bucket.mean() - out_bucket.mean()
            pooled_var = in_bucket.var(ddof=1) / max(in_bucket.shape[0], 1) + out_bucket.var(ddof=1) / max(out_bucket.shape[0], 1)
            t_stat = diff / math.sqrt(pooled_var) if pooled_var and not pd.isna(pooled_var) else np.nan
            rows.append(
                {
                    "factor": factor,
                    "bucket": bucket_name,
                    "months": ",".join(str(m) for m in months),
                    "bucket_annualized_mean": float(in_bucket.mean() * periods_per_year) if not in_bucket.empty else np.nan,
                    "other_annualized_mean": float(out_bucket.mean() * periods_per_year) if not out_bucket.empty else np.nan,
                    "annualized_diff": float(diff * periods_per_year) if not pd.isna(diff) else np.nan,
                    "t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
                    "p_value": _t_pvalue(float(t_stat), int(in_bucket.shape[0] + out_bucket.shape[0] - 2))
                    if not pd.isna(t_stat)
                    else np.nan,
                    "bucket_count": int(in_bucket.shape[0]),
                    "other_count": int(out_bucket.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def holiday_window_summary(
    premium_df: pd.DataFrame,
    holidays: pd.DataFrame,
    date_col: str,
    factor_col: str,
    premium_col: str,
    holiday_col: str,
    pre_days: int = 5,
    post_days: int = 5,
) -> pd.DataFrame:
    """Summarize daily factor premium in pre/post holiday trading windows."""
    _require_columns(premium_df, [date_col, factor_col, premium_col])
    _require_columns(holidays, [holiday_col])
    daily = premium_df.copy()
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.normalize()
    holiday_dates = pd.to_datetime(holidays[holiday_col]).dt.normalize().sort_values().drop_duplicates()
    trading_dates = pd.Index(sorted(daily[date_col].drop_duplicates()))

    rows = []
    for holiday_date in holiday_dates:
        if holiday_date not in trading_dates:
            before = trading_dates[trading_dates < holiday_date][-pre_days:]
            after = trading_dates[trading_dates > holiday_date][:post_days]
        else:
            loc = trading_dates.get_loc(holiday_date)
            before = trading_dates[max(0, loc - pre_days) : loc]
            after = trading_dates[loc + 1 : loc + 1 + post_days]
        for label, dates in [("pre", before), ("post", after)]:
            sample = daily[daily[date_col].isin(dates)]
            for factor, group in sample.groupby(factor_col):
                rows.append(
                    {
                        "holiday": holiday_date,
                        "window": label,
                        "factor": factor,
                        "mean_premium": float(group[premium_col].astype(float).mean()),
                        "sum_premium": float(group[premium_col].astype(float).sum()),
                        "count": int(group.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def seasonal_exposure_schedule(
    dates: pd.Series | pd.Index,
    rules: list[dict[str, object]],
) -> pd.DataFrame:
    """Create dated factor exposure overrides from calendar rules.

    Each rule is a dict with `factor`, `months`, `lower`, and `upper`.
    """
    date_index = pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values()
    rows = []
    for date in date_index:
        month = int(date.month)
        for rule in rules:
            months = {int(m) for m in rule.get("months", [])}
            if month in months:
                rows.append(
                    {
                        "date": date,
                        "factor": str(rule["factor"]),
                        "lower": float(rule["lower"]),
                        "upper": float(rule["upper"]),
                        "rule_name": str(rule.get("name", "")),
                    }
                )
    return pd.DataFrame(rows)


def apply_exposure_schedule(
    base_bounds: pd.DataFrame,
    schedule: pd.DataFrame,
    date_col: str = "date",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """Expand base exposure bounds and apply date-specific overrides."""
    _require_columns(base_bounds, [factor_col, "lower", "upper"])
    _require_columns(schedule, [date_col, factor_col, "lower", "upper"])
    rows = []
    for date, day_schedule in schedule.groupby(date_col):
        day = base_bounds.copy()
        day[date_col] = pd.to_datetime(date)
        for _, override in day_schedule.iterrows():
            mask = day[factor_col] == override[factor_col]
            if mask.any():
                day.loc[mask, "lower"] = float(override["lower"])
                day.loc[mask, "upper"] = float(override["upper"])
        rows.append(day[[date_col, factor_col, "lower", "upper"]])
    if not rows:
        return pd.DataFrame(columns=[date_col, factor_col, "lower", "upper"])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--premium-col", required=True)
    parser.add_argument("--output", default="month_premium_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    summary = month_premium_summary(df, args.date_col, args.factor_col, args.premium_col)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
