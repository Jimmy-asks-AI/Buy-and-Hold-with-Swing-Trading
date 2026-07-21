"""Feature builders allowed by the accepted daily-only raw OHLCV scope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return 0.0
    return float(numerator) / float(denominator)


def quantile_value(values: pd.Series, q: float) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


@dataclass(frozen=True)
class DailyOnlyFeatureConfig:
    low_amount_raw_threshold: float = 5000.0
    low_volume_raw_threshold: float = 100.0
    high_activity_amount_quantile: float = 0.8
    top_amount_count: int = 10


def compute_daily_market_features(
    data: pd.DataFrame,
    trade_date: str,
    config: DailyOnlyFeatureConfig | None = None,
) -> dict[str, float | int | str]:
    """Compute market-level diagnostics from accepted raw daily OHLCV data.

    These features are cross-sectional diagnostics only. They do not create
    adjusted returns, valuation factors, dividend metrics, or backtest labels.
    """

    if config is None:
        config = DailyOnlyFeatureConfig()
    asset_count = int(data.shape[0])
    if asset_count == 0:
        return {
            "trade_date": trade_date,
            "asset_count": 0,
            "active_asset_count": 0,
            "active_asset_ratio": 0.0,
            "zero_volume_count": 0,
            "zero_amount_count": 0,
            "low_amount_count": 0,
            "low_volume_count": 0,
            "sum_volume_raw": 0.0,
            "sum_amount_raw": 0.0,
            "median_amount_raw": float("nan"),
            "p20_amount_raw": float("nan"),
            "p80_amount_raw": float("nan"),
            "top_amount_share": 0.0,
            "advancer_count": 0,
            "decliner_count": 0,
            "flat_count": 0,
            "advance_decline_balance": 0.0,
            "advance_ratio": 0.0,
            "decline_ratio": 0.0,
            "median_pct_chg_raw": float("nan"),
            "mean_pct_chg_raw": float("nan"),
            "limit_like_up_count": 0,
            "limit_like_down_count": 0,
            "median_intraday_strength_raw": float("nan"),
            "median_range_ratio_raw": float("nan"),
            "bad_ohlc_after_adapter": 0,
            "data_scope": "accepted_processed_daily_only",
            "price_adjustment": "none_raw",
            "feature_scope": "daily_only_market_diagnostic",
        }

    open_ = numeric_series(data, "open")
    high = numeric_series(data, "high")
    low = numeric_series(data, "low")
    close = numeric_series(data, "close")
    pct_chg = numeric_series(data, "pct_chg")
    volume = numeric_series(data, "vol").fillna(0.0)
    amount = numeric_series(data, "amount").fillna(0.0)

    active = (volume > 0) & (amount > 0)
    zero_volume = volume <= 0
    zero_amount = amount <= 0
    low_amount = (amount > 0) & (amount <= config.low_amount_raw_threshold)
    low_volume = (volume > 0) & (volume <= config.low_volume_raw_threshold)
    total_amount = float(amount.sum())
    total_volume = float(volume.sum())
    top_amount = float(amount.nlargest(min(config.top_amount_count, asset_count)).sum()) if asset_count else 0.0

    advancers = pct_chg > 0
    decliners = pct_chg < 0
    flat = pct_chg == 0
    valid_pct = int(pct_chg.notna().sum())
    intraday_strength = (close - open_) / open_.replace(0, np.nan)
    range_ratio = (high - low) / close.replace(0, np.nan)
    bad_ohlc = ((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).fillna(True)

    return {
        "trade_date": trade_date,
        "asset_count": asset_count,
        "active_asset_count": int(active.sum()),
        "active_asset_ratio": safe_ratio(float(active.sum()), asset_count),
        "zero_volume_count": int(zero_volume.sum()),
        "zero_amount_count": int(zero_amount.sum()),
        "low_amount_count": int(low_amount.sum()),
        "low_volume_count": int(low_volume.sum()),
        "sum_volume_raw": total_volume,
        "sum_amount_raw": total_amount,
        "median_amount_raw": float(amount.replace(0, np.nan).median()),
        "p20_amount_raw": quantile_value(amount.replace(0, np.nan), 0.2),
        "p80_amount_raw": quantile_value(amount.replace(0, np.nan), config.high_activity_amount_quantile),
        "top_amount_share": safe_ratio(top_amount, total_amount),
        "advancer_count": int(advancers.sum()),
        "decliner_count": int(decliners.sum()),
        "flat_count": int(flat.sum()),
        "advance_decline_balance": safe_ratio(float(advancers.sum() - decliners.sum()), max(valid_pct, 1)),
        "advance_ratio": safe_ratio(float(advancers.sum()), max(valid_pct, 1)),
        "decline_ratio": safe_ratio(float(decliners.sum()), max(valid_pct, 1)),
        "median_pct_chg_raw": float(pct_chg.median()),
        "mean_pct_chg_raw": float(pct_chg.mean()),
        "limit_like_up_count": int((pct_chg >= 9.8).sum()),
        "limit_like_down_count": int((pct_chg <= -9.8).sum()),
        "median_intraday_strength_raw": float(intraday_strength.median()),
        "median_range_ratio_raw": float(range_ratio.median()),
        "bad_ohlc_after_adapter": int(bad_ohlc.sum()),
        "data_scope": "accepted_processed_daily_only",
        "price_adjustment": "none_raw",
        "feature_scope": "daily_only_market_diagnostic",
    }


def compute_asset_activity_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build per-asset raw activity diagnostics for selected dates."""

    required = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol", "amount"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"missing columns for asset activity features: {missing}")
    out = data[required].copy()
    out["asset"] = out["ts_code"].astype(str)
    open_ = numeric_series(out, "open")
    high = numeric_series(out, "high")
    low = numeric_series(out, "low")
    close = numeric_series(out, "close")
    out["raw_amount"] = numeric_series(out, "amount")
    out["raw_volume"] = numeric_series(out, "vol")
    out["raw_pct_chg_diagnostic"] = numeric_series(out, "pct_chg")
    out["intraday_strength_raw"] = (close - open_) / open_.replace(0, np.nan)
    out["range_ratio_raw"] = (high - low) / close.replace(0, np.nan)
    out["active_asset"] = (out["raw_volume"] > 0) & (out["raw_amount"] > 0)
    out["data_scope"] = "accepted_processed_daily_only"
    out["price_adjustment"] = "none_raw"
    out["feature_scope"] = "daily_only_asset_activity_sample"
    return out[
        [
            "trade_date",
            "asset",
            "raw_amount",
            "raw_volume",
            "raw_pct_chg_diagnostic",
            "intraday_strength_raw",
            "range_ratio_raw",
            "active_asset",
            "data_scope",
            "price_adjustment",
            "feature_scope",
        ]
    ]


FEATURE_DICTIONARY_ROWS = [
    {
        "feature": "asset_count",
        "level": "market_daily",
        "definition": "Number of accepted stock rows for a trade date after V3.45.1 quarantine.",
        "allowed_use": "coverage and breadth diagnostics",
        "forbidden_use": "not a tradable return signal by itself",
    },
    {
        "feature": "active_asset_ratio",
        "level": "market_daily",
        "definition": "Share of accepted rows with both raw volume and raw amount above zero.",
        "allowed_use": "market activity and suspension/liquidity regime diagnostics",
        "forbidden_use": "not adjusted for float, listing status, or survivorship",
    },
    {
        "feature": "sum_amount_raw",
        "level": "market_daily",
        "definition": "Cross-sectional sum of raw Tushare daily amount in provider units.",
        "allowed_use": "market activity regime diagnostics",
        "forbidden_use": "do not compare as precise currency without provider-unit normalization",
    },
    {
        "feature": "top_amount_share",
        "level": "market_daily",
        "definition": "Share of total raw amount contributed by the top-N amount rows.",
        "allowed_use": "turnover concentration diagnostics",
        "forbidden_use": "not a capacity estimate",
    },
    {
        "feature": "advance_decline_balance",
        "level": "market_daily",
        "definition": "(Advancers minus decliners) divided by valid pct_chg count.",
        "allowed_use": "raw market breadth diagnostic",
        "forbidden_use": "not an adjusted-return market index",
    },
    {
        "feature": "median_intraday_strength_raw",
        "level": "market_daily",
        "definition": "Median cross-sectional (close - open) / open using raw daily prices.",
        "allowed_use": "same-day intraday pressure diagnostic",
        "forbidden_use": "not a long-horizon return or backtest label",
    },
    {
        "feature": "median_range_ratio_raw",
        "level": "market_daily",
        "definition": "Median cross-sectional (high - low) / close using raw daily prices.",
        "allowed_use": "raw volatility/activity diagnostic",
        "forbidden_use": "not split/dividend adjusted",
    },
]
