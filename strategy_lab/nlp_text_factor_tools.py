#!/usr/bin/env python
"""NLP/text-factor utilities with explicit timestamp alignment."""

from __future__ import annotations

import re
from datetime import time

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def clean_text(text: str) -> str:
    """Normalize whitespace for Chinese/English mixed text."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mixed_tokenize(text: str, min_token_len: int = 1) -> list[str]:
    """Simple tokenizer fallback when jieba/HanLP is unavailable.

    Chinese characters are emitted as single-character tokens. For production
    Chinese NLP, replace this with a domain tokenizer and keep this function as
    a deterministic fallback.
    """
    tokens = TOKEN_RE.findall(clean_text(text).lower())
    return [token for token in tokens if len(token) >= min_token_len]


def character_ngrams(text: str, n: int = 2) -> list[str]:
    """Build Chinese character n-grams for dictionary-light text factors."""
    chars = [token for token in mixed_tokenize(text) if re.fullmatch(r"[\u4e00-\u9fff]", token)]
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(chars) < n:
        return []
    return ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]


def dictionary_sentiment_score(
    text: str,
    positive_words: set[str],
    negative_words: set[str],
    use_bigrams: bool = True,
) -> float:
    """Compute a transparent dictionary sentiment score."""
    tokens = mixed_tokenize(text)
    terms = list(tokens)
    if use_bigrams:
        terms.extend(character_ngrams(text, n=2))
    pos = sum(1 for token in terms if token in positive_words)
    neg = sum(1 for token in terms if token in negative_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def dictionary_entity_match(text: str, entity_dict: dict[str, str]) -> pd.DataFrame:
    """Match known entities in text.

    entity_dict maps display name or alias to entity id, such as stock code.
    """
    clean = clean_text(text)
    rows = []
    for alias, entity_id in entity_dict.items():
        if alias and alias in clean:
            rows.append({"alias": alias, "entity_id": entity_id})
    return pd.DataFrame(rows)


def align_text_events_to_trade_date(
    events: pd.DataFrame,
    calendar: pd.Series | list,
    publish_time_col: str = "publish_time",
    cutoff: time = time(15, 0),
    output_col: str = "available_trade_date",
) -> pd.DataFrame:
    """Map text events to the first trade date when they are usable.

    Events published after cutoff are assigned to the next trade date. This is
    the key anti-leakage step for news, announcements, social posts and reports.
    """
    out = events.copy()
    publish_time = pd.to_datetime(out[publish_time_col])
    trade_dates = pd.Index(pd.to_datetime(pd.Series(calendar)).dt.normalize().drop_duplicates().sort_values())
    event_dates = publish_time.dt.normalize()
    after_cutoff = publish_time.dt.time > cutoff
    candidate_dates = event_dates + pd.to_timedelta(after_cutoff.astype(int), unit="D")

    positions = trade_dates.searchsorted(candidate_dates, side="left")
    aligned = [trade_dates[pos] if pos < len(trade_dates) else pd.NaT for pos in positions]
    out[output_col] = aligned
    out["after_cutoff"] = after_cutoff
    return out


def aggregate_daily_text_features(
    events: pd.DataFrame,
    date_col: str,
    asset_col: str,
    sentiment_col: str = "sentiment",
    count_col: str = "text_count",
) -> pd.DataFrame:
    """Aggregate event-level text sentiment into asset-date features."""
    required = [date_col, asset_col, sentiment_col]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    clean = events.dropna(subset=[date_col, asset_col]).copy()
    clean[sentiment_col] = clean[sentiment_col].astype(float)
    grouped = clean.groupby([date_col, asset_col])[sentiment_col]
    out = grouped.agg(["count", "mean", "sum", "std"]).reset_index()
    out = out.rename(
        columns={
            "count": count_col,
            "mean": f"{sentiment_col}_mean",
            "sum": f"{sentiment_col}_sum",
            "std": f"{sentiment_col}_std",
        }
    )
    out[f"{sentiment_col}_positive_ratio"] = (
        clean.assign(is_positive=clean[sentiment_col] > 0)
        .groupby([date_col, asset_col])["is_positive"]
        .mean()
        .to_numpy()
    )
    out[f"{sentiment_col}_negative_ratio"] = (
        clean.assign(is_negative=clean[sentiment_col] < 0)
        .groupby([date_col, asset_col])["is_negative"]
        .mean()
        .to_numpy()
    )
    return out


def rolling_text_features(
    daily_features: pd.DataFrame,
    date_col: str,
    asset_col: str,
    feature_cols: list[str],
    windows: tuple[int, ...] = (5, 20),
) -> pd.DataFrame:
    """Create live-safe trailing text features by asset."""
    out = daily_features.sort_values([asset_col, date_col]).copy()
    for col in feature_cols:
        for window in windows:
            shifted = out.groupby(asset_col)[col].shift(1)
            out[f"{col}_roll_mean_{window}"] = shifted.groupby(out[asset_col]).rolling(
                window, min_periods=max(2, window // 2)
            ).mean().reset_index(level=0, drop=True)
            out[f"{col}_roll_sum_{window}"] = shifted.groupby(out[asset_col]).rolling(
                window, min_periods=max(2, window // 2)
            ).sum().reset_index(level=0, drop=True)
    return out


def text_factor_coverage_report(
    events: pd.DataFrame,
    date_col: str,
    asset_col: str,
    sentiment_col: str = "sentiment",
) -> pd.Series:
    """Report basic coverage and missingness before using text factors."""
    return pd.Series(
        {
            "n_events": int(events.shape[0]),
            "n_dates": int(events[date_col].nunique()) if date_col in events else 0,
            "n_assets": int(events[asset_col].nunique()) if asset_col in events else 0,
            "sentiment_missing_rate": float(events[sentiment_col].isna().mean()) if sentiment_col in events else np.nan,
            "events_per_asset_median": float(events.groupby(asset_col).size().median()) if asset_col in events else np.nan,
        }
    )
