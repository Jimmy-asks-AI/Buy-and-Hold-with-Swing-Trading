#!/usr/bin/env python
"""Analyst rating factor helpers.

The functions here turn raw analyst reports into monthly buy-rating signals,
with the safeguards highlighted by sell-side research: report-type filtering,
new-vs-continuous separation, fundamental support, and analyst-view blending.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
DEFAULT_BUY_RATINGS = ("buy", "strong buy", "outperform", "买入", "强烈推荐", "增持", "推荐")
DEFAULT_REPORT_TYPES = ("comment", "deep", "点评", "公司点评", "深度", "深度报告", "公司深度")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _zscore(s: pd.Series) -> pd.Series:
    values = s.astype(float)
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (values - values.mean()) / std


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def monthly_buy_rating_features(
    reports: pd.DataFrame,
    report_date_col: str,
    asset_col: str,
    rating_col: str | None = None,
    report_type_col: str | None = None,
    analyst_col: str | None = None,
    allowed_ratings: tuple[str, ...] = DEFAULT_BUY_RATINGS,
    allowed_report_types: tuple[str, ...] | None = DEFAULT_REPORT_TYPES,
    period_freq: str = "M",
    fill_missing_periods: bool = True,
) -> pd.DataFrame:
    """Create buy-rating counts, new-rating flags, and continuity flags.

    The default report-type filter keeps only comment and deep reports, because
    the broad all-report buy-rating dummy showed alpha decay in recent samples.
    Set `allowed_report_types=None` to keep every report type.
    """
    required = [report_date_col, asset_col]
    if rating_col:
        required.append(rating_col)
    if report_type_col:
        required.append(report_type_col)
    if analyst_col:
        required.append(analyst_col)
    _require_columns(reports, required)

    out = reports.copy()
    out[report_date_col] = pd.to_datetime(out[report_date_col])
    if rating_col:
        allowed_rating_set = {str(x).strip().lower() for x in allowed_ratings}
        out = out[_normalize_text(out[rating_col]).isin(allowed_rating_set)]
    if report_type_col and allowed_report_types is not None:
        allowed_type_set = {str(x).strip().lower() for x in allowed_report_types}
        out = out[_normalize_text(out[report_type_col]).isin(allowed_type_set)]
    if out.empty:
        return pd.DataFrame(columns=["period", asset_col, "buy_report_count", "analyst_count"])

    out["period"] = out[report_date_col].dt.to_period(period_freq)
    agg_dict = {report_date_col: "count"}
    if analyst_col:
        agg_dict[analyst_col] = pd.Series.nunique
    grouped = out.groupby(["period", asset_col], as_index=False).agg(agg_dict)
    grouped = grouped.rename(columns={report_date_col: "buy_report_count"})
    if analyst_col:
        grouped = grouped.rename(columns={analyst_col: "analyst_count"})
    else:
        grouped["analyst_count"] = np.nan

    if fill_missing_periods:
        periods = pd.period_range(grouped["period"].min(), grouped["period"].max(), freq=period_freq)
        assets = pd.Index(grouped[asset_col].drop_duplicates())
        full_index = pd.MultiIndex.from_product([periods, assets], names=["period", asset_col])
        grouped = grouped.set_index(["period", asset_col]).reindex(full_index).reset_index()
        grouped["buy_report_count"] = grouped["buy_report_count"].fillna(0).astype(int)
        grouped["analyst_count"] = grouped["analyst_count"].fillna(0)

    grouped = grouped.sort_values([asset_col, "period"]).copy()
    grouped["date"] = grouped["period"].dt.to_timestamp(how="end").dt.normalize()
    grouped["buy_rating"] = grouped["buy_report_count"] > 0
    prev = grouped.groupby(asset_col)["buy_rating"].shift(1).fillna(False)
    grouped["new_buy_rating"] = grouped["buy_rating"] & ~prev
    grouped["continuous_buy_rating"] = grouped["buy_rating"] & prev
    grouped["buy_rating_strength"] = np.log1p(grouped["buy_report_count"].astype(float))
    grouped["analyst_coverage_strength"] = np.log1p(grouped["analyst_count"].astype(float))
    return grouped


def add_fundamental_support(
    rating_features: pd.DataFrame,
    fundamentals: pd.DataFrame,
    date_col: str,
    asset_col: str,
    sue_col: str,
    bottom_quantile: float = 1.0 / 3.0,
    output_col: str = "fundamental_supported",
) -> pd.DataFrame:
    """Flag rating signals with acceptable earnings surprise support.

    The default support rule follows the report logic: exclude the bottom third
    of SUE names rather than demanding top-tercile fundamentals.
    """
    if not 0 < bottom_quantile < 1:
        raise ValueError("bottom_quantile must be in (0, 1)")
    _require_columns(rating_features, [date_col, asset_col])
    _require_columns(fundamentals, [date_col, asset_col, sue_col])
    left = rating_features.copy()
    right = fundamentals[[date_col, asset_col, sue_col]].copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[date_col] = pd.to_datetime(right[date_col])
    out = left.merge(right, on=[date_col, asset_col], how="left")
    sue_rank = out.groupby(date_col)[sue_col].transform(lambda s: s.astype(float).rank(pct=True, method="first"))
    out[output_col] = sue_rank > bottom_quantile
    out["supported_buy_rating"] = out.get("buy_rating", False).astype(bool) & out[output_col].fillna(False)
    out["supported_new_buy_rating"] = out.get("new_buy_rating", False).astype(bool) & out[output_col].fillna(False)
    out["unsupported_buy_rating"] = out.get("buy_rating", False).astype(bool) & ~out[output_col].fillna(False)
    return out


def analyst_view_score(
    rating_features: pd.DataFrame,
    date_col: str,
    coverage_col: str = "analyst_coverage_strength",
    improved_rating_col: str = "supported_new_buy_rating",
    output_col: str = "analyst_view_score",
    coverage_weight: float = 0.5,
    rating_weight: float = 0.5,
) -> pd.DataFrame:
    """Blend analyst coverage and improved buy-rating signal by date."""
    _require_columns(rating_features, [date_col, coverage_col, improved_rating_col])
    if abs(coverage_weight) + abs(rating_weight) <= EPS:
        raise ValueError("At least one weight must be non-zero")
    out = rating_features.copy()
    out["_coverage_z"] = out.groupby(date_col)[coverage_col].transform(_zscore)
    improved = out[improved_rating_col].astype(float)
    out["_rating_z"] = improved.groupby(out[date_col]).transform(_zscore)
    denom = abs(coverage_weight) + abs(rating_weight)
    out[output_col] = (coverage_weight * out["_coverage_z"] + rating_weight * out["_rating_z"]) / denom
    return out.drop(columns=["_coverage_z", "_rating_z"])


def rating_signal_summary(
    rating_features: pd.DataFrame,
    date_col: str,
    signal_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize cross-sectional coverage of rating dummies by period."""
    if signal_cols is None:
        signal_cols = ["buy_rating", "new_buy_rating", "continuous_buy_rating", "supported_new_buy_rating"]
    _require_columns(rating_features, [date_col, *[c for c in signal_cols if c in rating_features.columns]])
    rows = []
    for date, group in rating_features.groupby(date_col):
        row: dict[str, object] = {"date": date, "asset_count": int(group.shape[0])}
        for col in signal_cols:
            if col in group.columns:
                row[f"{col}_coverage"] = float(group[col].astype(bool).mean())
                row[f"{col}_count"] = int(group[col].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--rating-col")
    parser.add_argument("--report-type-col")
    parser.add_argument("--analyst-col")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = pd.read_csv(args.reports)
    out = monthly_buy_rating_features(
        reports,
        report_date_col=args.date_col,
        asset_col=args.asset_col,
        rating_col=args.rating_col,
        report_type_col=args.report_type_col,
        analyst_col=args.analyst_col,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
