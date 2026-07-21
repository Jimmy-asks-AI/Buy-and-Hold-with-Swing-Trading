#!/usr/bin/env python
"""Deep-learning model-zoo ensemble diagnostics."""

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


def _zscore(s: pd.Series) -> pd.Series:
    values = s.astype(float)
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (values - values.mean()) / std


def model_zoo_reference() -> pd.DataFrame:
    """Return the report's model-zoo result table for quick benchmarking."""
    rows = [
        ("BiAGRU", "rnn", 0.135, 0.125, 32.5, 24.7, 38.5, 28.2, 0.75, 40),
        ("TCN", "convolution", 0.133, 0.121, 32.6, 23.0, 37.3, 25.1, 0.68, 50),
        ("BiATCN", "convolution", 0.137, 0.126, 33.8, 24.5, 40.6, 28.6, 0.70, 48),
        ("TimesNet", "convolution", 0.124, 0.115, 28.7, 21.1, 33.5, 23.8, 0.77, 40),
        ("Transformer", "transformer", 0.129, 0.119, 30.6, 22.9, 35.8, 25.9, 0.75, 40),
        ("Informer", "transformer", 0.125, 0.115, 27.9, 19.5, 33.6, 22.7, 0.72, 45),
        ("PatchTST", "transformer", 0.111, 0.105, 23.7, 18.1, 26.7, 19.3, 0.80, 30),
        ("DBiAGRU", "rnn_decomposition", 0.134, 0.124, 33.0, 25.2, 41.0, 30.4, 0.76, 40),
        ("RLinear", "linear", 0.111, 0.105, 22.5, 16.5, 25.7, 17.8, 0.80, 33),
        ("RMLP", "linear", 0.117, 0.110, 24.2, 17.0, 27.6, 18.3, 0.76, 40),
        ("DLinear", "linear", 0.119, 0.111, 24.6, 17.7, 26.5, 17.8, 0.77, 37),
        ("TSMixer", "linear", 0.122, 0.114, 26.8, 19.6, 32.6, 22.9, 0.76, 39),
        ("BiAGRU+BiATCN", "ensemble", 0.142, 0.131, 35.2, 26.7, 41.2, 30.2, 0.74, 43),
        ("BiAGRU+BiATCN+Transformer", "ensemble", 0.141, 0.129, 34.2, 26.2, 39.9, 29.5, 0.76, 40),
        ("DBiAGRU+BiATCN+Transformer", "ensemble", 0.142, 0.130, 34.8, 27.0, 40.8, 30.4, 0.76, 40),
        ("DBiAGRU+BiATCN+Transformer+TSMixer", "ensemble", 0.140, 0.129, 33.8, 26.2, 39.6, 29.6, 0.77, 38),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "family",
            "rank_ic_t0_close",
            "rank_ic_t1_vwap",
            "top10_excess_pre_cost",
            "top10_excess_after_cost",
            "top100_excess_pre_cost",
            "top100_excess_after_cost",
            "factor_autocorr",
            "top10_turnover_x",
        ],
    )


def zscore_model_predictions(
    predictions: pd.DataFrame,
    date_col: str,
    model_cols: list[str],
    output_suffix: str = "_z",
) -> pd.DataFrame:
    """Cross-sectionally standardize model predictions by date."""
    _require_columns(predictions, [date_col, *model_cols])
    out = predictions.copy()
    for col in model_cols:
        out[f"{col}{output_suffix}"] = out.groupby(date_col)[col].transform(_zscore)
    return out


def equal_weight_model_ensemble(
    predictions: pd.DataFrame,
    date_col: str,
    model_cols: list[str],
    output_col: str = "model_ensemble_score",
    standardize: bool = True,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Build an equal or user-weighted model ensemble score."""
    _require_columns(predictions, [date_col, *model_cols])
    out = predictions.copy()
    cols = model_cols
    if standardize:
        out = zscore_model_predictions(out, date_col, model_cols)
        cols = [f"{c}_z" for c in model_cols]
    if weights is None:
        weights = [1.0] * len(cols)
    if len(weights) != len(cols):
        raise ValueError("weights length must match model_cols length")
    w = np.asarray(weights, dtype=float)
    if np.abs(w).sum() <= EPS:
        raise ValueError("weights cannot all be zero")
    w = w / np.abs(w).sum()
    out[output_col] = out[cols].astype(float).mul(w, axis=1).sum(axis=1)
    return out


def mean_cross_sectional_correlation(
    predictions: pd.DataFrame,
    date_col: str,
    model_cols: list[str],
) -> pd.DataFrame:
    """Average pairwise model-score correlations across dates."""
    _require_columns(predictions, [date_col, *model_cols])
    corr_sum = pd.DataFrame(0.0, index=model_cols, columns=model_cols)
    count = 0
    for _, group in predictions.groupby(date_col):
        corr = group[model_cols].astype(float).corr()
        if corr.isna().all().all():
            continue
        corr_sum = corr_sum.add(corr.fillna(0.0), fill_value=0.0)
        count += 1
    if count == 0:
        return corr_sum.replace(0.0, np.nan)
    return corr_sum / count


def model_uniqueness_weights(corr: pd.DataFrame, floor: float = 0.0) -> pd.Series:
    """Turn a model correlation matrix into simple inverse-crowding weights."""
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    avg_abs_corr = corr.abs().where(~np.eye(corr.shape[0], dtype=bool)).mean(axis=1)
    uniqueness = (1.0 - avg_abs_corr).clip(lower=floor)
    if uniqueness.sum() <= EPS:
        return pd.Series(1.0 / corr.shape[0], index=corr.index, name="model_weight")
    return (uniqueness / uniqueness.sum()).rename("model_weight")


def factor_exposure_shift_report(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
    date_col: str,
    asset_col: str,
    model_cols: list[str],
    factor_cols: list[str],
) -> pd.DataFrame:
    """Measure model-score correlation with low-frequency factor exposures."""
    _require_columns(predictions, [date_col, asset_col, *model_cols])
    _require_columns(factors, [date_col, asset_col, *factor_cols])
    left = predictions.copy()
    right = factors.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[date_col] = pd.to_datetime(right[date_col])
    merged = left.merge(right[[date_col, asset_col, *factor_cols]], on=[date_col, asset_col], how="inner")
    rows = []
    for date, group in merged.groupby(date_col):
        for model in model_cols:
            for factor in factor_cols:
                clean = group[[model, factor]].dropna()
                corr = clean[model].astype(float).corr(clean[factor].astype(float)) if clean.shape[0] >= 3 else np.nan
                rows.append({"date": date, "model": model, "factor": factor, "corr": corr, "obs": int(clean.shape[0])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--csv")
    parser.add_argument("--date-col")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.reference:
        out = model_zoo_reference()
    else:
        if not args.csv or not args.date_col or not args.models:
            raise SystemExit("--csv, --date-col and --models are required unless --reference is used")
        df = pd.read_csv(args.csv)
        out = equal_weight_model_ensemble(df, args.date_col, args.models)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
