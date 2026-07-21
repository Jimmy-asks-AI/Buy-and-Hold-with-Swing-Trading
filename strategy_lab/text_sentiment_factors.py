#!/usr/bin/env python
"""Text sentiment factor helpers for news and analyst reports."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def linear_decay_weights(length: int) -> np.ndarray:
    """Return oldest-to-newest linear decay weights summing to one."""
    if length <= 0:
        raise ValueError("length must be positive.")
    raw = np.arange(1, length + 1, dtype=float)
    return raw / raw.sum()


def bert_probability_to_score(
    prob_positive: pd.Series | np.ndarray,
    negative_multiplier: float = 1.0,
) -> pd.Series:
    """Convert positive-class probability to centered sentiment score."""
    score = pd.Series(prob_positive, dtype="float64") - 0.5
    if negative_multiplier != 1.0:
        score = score.where(score >= 0, score * negative_multiplier)
    return score


def daily_news_sentiment_score(
    news: pd.DataFrame,
    date_col: str,
    asset_col: str,
    label_col: str,
    positive_value=1,
    negative_value=0,
    output_col: str = "news_score",
) -> pd.DataFrame:
    """Aggregate tagged news into positive-count minus negative-count score."""
    work = news.dropna(subset=[date_col, asset_col, label_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()
    label = work[label_col]
    work["_signed_label"] = np.select([label == positive_value, label == negative_value], [1.0, -1.0], default=np.nan)
    out = work.dropna(subset=["_signed_label"]).groupby([asset_col, date_col])["_signed_label"].sum().reset_index(name=output_col)
    out["news_count"] = work.groupby([asset_col, date_col]).size().to_numpy()
    return out


def average_event_score_by_day(
    events: pd.DataFrame,
    date_col: str,
    asset_col: str,
    score_col: str,
    output_col: str = "event_score",
) -> pd.DataFrame:
    """Average multiple sentence/report scores to asset-day level."""
    work = events.dropna(subset=[date_col, asset_col, score_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()
    return work.groupby([asset_col, date_col])[score_col].mean().reset_index(name=output_col)


def build_decay_sentiment_factor(
    daily_scores: pd.DataFrame,
    trade_dates: Iterable,
    date_col: str,
    asset_col: str,
    score_col: str,
    window_days: int = 30,
    output_col: str = "sentiment_factor",
    include_empty_assets: Iterable | None = None,
) -> pd.DataFrame:
    """Build natural-day rolling linear-decay sentiment factor."""
    scores = daily_scores.dropna(subset=[date_col, asset_col, score_col]).copy()
    scores[date_col] = pd.to_datetime(scores[date_col]).dt.normalize()
    trade = pd.Series(pd.to_datetime(list(trade_dates))).dt.normalize().drop_duplicates().sort_values()
    assets = pd.Index(include_empty_assets if include_empty_assets is not None else scores[asset_col].dropna().unique())
    rows = []
    weights = linear_decay_weights(window_days)
    by_asset = {asset: g.set_index(date_col)[score_col].sort_index() for asset, g in scores.groupby(asset_col)}
    for t in trade:
        calendar = pd.date_range(t - pd.Timedelta(days=window_days - 1), t, freq="D")
        for asset in assets:
            series = by_asset.get(asset)
            if series is None:
                continue
            window = series.reindex(calendar).fillna(0.0)
            if window.abs().sum() <= EPS:
                continue
            rows.append({asset_col: asset, date_col: t, output_col: float(np.dot(weights, window.to_numpy(dtype=float)))})
    return pd.DataFrame(rows)


def factor_coverage_by_date(
    factor_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    factor_col: str,
) -> pd.DataFrame:
    """Compute factor coverage relative to a date-asset universe."""
    universe_count = universe_df.groupby(date_col)[asset_col].nunique()
    covered = factor_df.dropna(subset=[factor_col]).groupby(date_col)[asset_col].nunique()
    out = pd.DataFrame({"universe": universe_count, "covered": covered}).fillna(0.0)
    out["coverage"] = out["covered"] / out["universe"].replace(0.0, np.nan)
    return out.reset_index()


def residualize_factor_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    control_cols: Iterable[str],
    output_col: str | None = None,
) -> pd.DataFrame:
    """Cross-sectionally residualize a factor against control factors."""
    target = output_col or f"{factor_col}_res"
    out = df.copy()
    control_cols = list(control_cols)
    residuals = []
    for _, group in out.groupby(date_col, sort=False):
        clean = group[[factor_col, *control_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        if clean.empty:
            residuals.append(pd.Series(np.nan, index=group.index, name=target))
            continue
        x = np.column_stack([np.ones(len(clean)), clean[control_cols].to_numpy(dtype=float)])
        y = clean[factor_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x @ beta
        residuals.append(pd.Series(y - fitted, index=clean.index, name=target).reindex(group.index))
    out[target] = pd.concat(residuals).sort_index()
    return out


def sentiment_factor_checklist() -> pd.DataFrame:
    rows = [
        ("timestamp", "Use database entry time, not article creation time, when modeling availability."),
        ("deduplication", "Remove repeated news and syndicated copies before counting sentiment."),
        ("filtering", "Filter market-tape headlines that mechanically contain up/down price words."),
        ("aggregation", "Aggregate sentence/report/news scores before rolling decay."),
        ("decay", "Use natural-day rolling windows, then map to tradable dates."),
        ("coverage", "Report coverage by index, industry, and time."),
        ("bias", "Check source, large-cap, industry, and analyst-coverage bias."),
        ("negative_weight", "Stress test higher weights for scarce negative analyst language."),
        ("orthogonal", "Residualize analyst sentiment against report score and report count."),
        ("validation", "Test post-training and post-publication samples separately."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.checklist:
        print(sentiment_factor_checklist())


if __name__ == "__main__":
    main()
