#!/usr/bin/env python
"""Feature processing and selection helpers for ML alpha research."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def signed_log1p(s: pd.Series) -> pd.Series:
    """Sign-preserving log transform for skewed features."""
    values = s.astype(float)
    return np.sign(values) * np.log1p(values.abs())


def skew_adjust_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str | None = None,
    skew_threshold: float = 1.0,
    output_suffix: str = "_skewadj",
) -> pd.DataFrame:
    """Apply signed log adjustment to features with high skewness."""
    _require_columns(df, feature_cols + ([date_col] if date_col else []))
    out = df.copy()
    for col in feature_cols:
        if date_col:
            skew = out.groupby(date_col)[col].transform(lambda s: s.astype(float).skew())
            adjusted = signed_log1p(out[col])
            out[f"{col}{output_suffix}"] = np.where(skew.abs() >= skew_threshold, adjusted, out[col].astype(float))
        else:
            if abs(out[col].astype(float).skew()) >= skew_threshold:
                out[f"{col}{output_suffix}"] = signed_log1p(out[col])
            else:
                out[f"{col}{output_suffix}"] = out[col].astype(float)
    return out


def winsorize_features_by_date(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    n_std: float = 3.0,
    output_suffix: str = "_win",
) -> pd.DataFrame:
    """Winsorize features within each date by mean +/- n * std."""
    _require_columns(df, [date_col, *feature_cols])
    out = df.copy()
    for col in feature_cols:
        mean = out.groupby(date_col)[col].transform("mean")
        std = out.groupby(date_col)[col].transform("std").fillna(0.0)
        out[f"{col}{output_suffix}"] = out[col].astype(float).clip(lower=mean - n_std * std, upper=mean + n_std * std)
    return out


def standardize_features(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    mode: str = "cross_section",
    rolling_window: int = 20,
    min_periods: int | None = None,
    output_suffix: str = "_z",
) -> pd.DataFrame:
    """Standardize features by cross section or rolling multi-cross-section panel."""
    _require_columns(df, [date_col, *feature_cols])
    mode = mode.lower()
    if mode not in {"cross_section", "rolling_panel"}:
        raise ValueError("mode must be one of: cross_section, rolling_panel")
    out = df.sort_values(date_col).copy()
    if min_periods is None:
        min_periods = max(3, rolling_window // 2)

    if mode == "cross_section":
        for col in feature_cols:
            mean = out.groupby(date_col)[col].transform("mean")
            std = out.groupby(date_col)[col].transform("std").replace(0, np.nan)
            out[f"{col}{output_suffix}"] = (out[col].astype(float) - mean) / std
        return out

    dates = pd.Index(pd.to_datetime(out[date_col]).drop_duplicates().sort_values())
    for col in feature_cols:
        values = pd.Series(np.nan, index=out.index, dtype="float64")
        for idx, date in enumerate(dates):
            window_dates = dates[max(0, idx - rolling_window + 1) : idx + 1]
            if len(window_dates) < min_periods:
                continue
            mask_window = pd.to_datetime(out[date_col]).isin(window_dates)
            mask_date = pd.to_datetime(out[date_col]) == date
            panel = out.loc[mask_window, col].astype(float)
            std = panel.std(ddof=1)
            if pd.isna(std) or std <= EPS:
                values.loc[mask_date] = 0.0
            else:
                values.loc[mask_date] = (out.loc[mask_date, col].astype(float) - panel.mean()) / std
        out[f"{col}{output_suffix}"] = values
    return out


def feature_contribution_summary(
    attribution_df: pd.DataFrame,
    feature_col: str,
    contribution_col: str,
    date_col: str | None = None,
    use_abs: bool = True,
) -> pd.DataFrame:
    """Summarize feature contribution values such as integrated gradients."""
    required = [feature_col, contribution_col]
    if date_col:
        required.append(date_col)
    _require_columns(attribution_df, required)
    out = attribution_df.copy()
    out["_contribution_for_rank"] = out[contribution_col].astype(float).abs() if use_abs else out[contribution_col].astype(float)
    group_cols = [feature_col] if date_col is None else [date_col, feature_col]
    summary = (
        out.groupby(group_cols)["_contribution_for_rank"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_contribution", "median": "median_contribution", "std": "std_contribution"})
    )
    rank_group = date_col if date_col else None
    if rank_group:
        summary["contribution_rank"] = summary.groupby(rank_group)["mean_contribution"].rank(ascending=False, method="first")
    else:
        summary["contribution_rank"] = summary["mean_contribution"].rank(ascending=False, method="first")
    return summary.sort_values(["contribution_rank"])


def select_top_features_static(
    contribution_summary: pd.DataFrame,
    feature_col: str,
    contribution_col: str = "mean_contribution",
    top_n: int = 64,
) -> list[str]:
    """Select the global top-N features by contribution."""
    _require_columns(contribution_summary, [feature_col, contribution_col])
    return (
        contribution_summary.groupby(feature_col)[contribution_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.astype(str)
        .tolist()
    )


def select_top_features_dynamic(
    attribution_df: pd.DataFrame,
    date_col: str,
    feature_col: str,
    contribution_col: str,
    top_n: int = 64,
    lookback: int = 12,
    min_periods: int = 6,
) -> pd.DataFrame:
    """Select top-N features from trailing contribution history.

    The current date's contribution is shifted out of the rolling average, so
    the dynamic selection plan is usable in live simulation.
    """
    _require_columns(attribution_df, [date_col, feature_col, contribution_col])
    daily = (
        attribution_df.assign(_abs_contribution=attribution_df[contribution_col].astype(float).abs())
        .groupby([date_col, feature_col], as_index=False)["_abs_contribution"]
        .mean()
        .sort_values([feature_col, date_col])
    )
    daily["trailing_contribution"] = daily.groupby(feature_col)["_abs_contribution"].transform(
        lambda s: s.rolling(lookback, min_periods=min_periods).mean().shift(1)
    )
    rows = []
    for date, group in daily.dropna(subset=["trailing_contribution"]).groupby(date_col):
        selected = group.sort_values("trailing_contribution", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            rows.append(
                {
                    "date": date,
                    "feature": row[feature_col],
                    "rank": rank,
                    "trailing_contribution": float(row["trailing_contribution"]),
                    "selected": True,
                }
            )
    return pd.DataFrame(rows)


def ensemble_predictions(
    predictions: pd.DataFrame,
    pred_cols: list[str],
    output_col: str = "ensemble_prediction",
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Combine single-granularity model predictions."""
    _require_columns(predictions, pred_cols)
    out = predictions.copy()
    if weights is None:
        weights = [1.0] * len(pred_cols)
    if len(weights) != len(pred_cols):
        raise ValueError("weights length must match pred_cols length")
    w = np.asarray(weights, dtype=float)
    if np.abs(w).sum() <= EPS:
        raise ValueError("weights cannot all be zero")
    w = w / np.abs(w).sum()
    out[output_col] = out[pred_cols].astype(float).mul(w, axis=1).sum(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["cross_section", "rolling_panel"], default="cross_section")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = standardize_features(df, args.date_col, args.features, mode=args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
