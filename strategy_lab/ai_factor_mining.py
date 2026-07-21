#!/usr/bin/env python
"""AI factor-mining helpers inspired by Huatai GP and AlphaNet reports.

This module intentionally keeps the core utilities dependency-light. It does not
train gplearn or TensorFlow models; it provides the reusable research plumbing
needed before those heavy models are introduced: preprocessing, fitness
functions, nonlinear transforms, validation diagnostics, AlphaNet version
specifications, and rolling train/validation windows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class GeneticProgrammingConfig:
    """Default GP search settings from the Huatai research flow."""

    generations: int = 3
    population_size: int = 1000
    init_depth_min: int = 1
    init_depth_max: int = 4
    tournament_size: int = 20
    p_crossover: float = 0.40
    p_subtree_mutation: float = 0.01
    p_hoist_mutation: float = 0.00
    p_point_mutation: float = 0.01
    p_point_replace: float = 0.40
    parsimony_coefficient: float = 0.001
    target_horizon: int = 20
    validation_ratio: float = 0.20


def safe_divide(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> pd.Series:
    """Elementwise division with zero and infinity protection."""
    left = pd.Series(x, dtype="float64")
    right = pd.Series(y, dtype="float64")
    out = left / right.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def signed_power(x: pd.Series | np.ndarray, power: float) -> pd.Series:
    """sign(x) * abs(x) ** power."""
    values = pd.Series(x, dtype="float64")
    return np.sign(values) * np.abs(values).pow(power)


def cross_section_rank(values: pd.Series) -> pd.Series:
    """Percentile rank within one cross-section."""
    return pd.to_numeric(values, errors="coerce").rank(pct=True, method="average")


def mad_winsorize(values: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """Median absolute deviation winsorization."""
    clean = pd.to_numeric(values, errors="coerce")
    median = clean.median()
    mad = (clean - median).abs().median()
    if pd.isna(mad) or mad <= EPS:
        return clean
    return clean.clip(lower=median - n_mad * mad, upper=median + n_mad * mad)


def zscore(values: pd.Series) -> pd.Series:
    """Standardize a vector; returns zeros when variance is unavailable."""
    clean = pd.to_numeric(values, errors="coerce")
    std = clean.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return clean * 0.0
    return (clean - clean.mean()) / std


def neutralize_cross_section(
    values: pd.Series,
    controls: pd.DataFrame,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Neutralize values against controls by OLS/WLS and return residuals."""
    y = pd.to_numeric(values, errors="coerce")
    x = controls.apply(pd.to_numeric, errors="coerce")
    clean = pd.concat([y.rename("_y"), x], axis=1).dropna()
    if clean.empty:
        return pd.Series(np.nan, index=values.index)
    xmat = np.column_stack([np.ones(len(clean)), clean[x.columns].to_numpy(dtype=float)])
    yvec = clean["_y"].to_numpy(dtype=float)
    if weights is not None:
        w = pd.to_numeric(weights.reindex(clean.index), errors="coerce").fillna(0.0).clip(lower=0.0)
        scale = np.sqrt(w.to_numpy(dtype=float))
        xmat = xmat * scale[:, None]
        yvec = yvec * scale
    beta = np.linalg.lstsq(xmat, yvec, rcond=None)[0]
    fitted = np.column_stack([np.ones(len(clean)), clean[x.columns].to_numpy(dtype=float)]) @ beta
    residual = pd.Series(yvec if weights is None else clean["_y"].to_numpy(dtype=float), index=clean.index) - fitted
    return residual.reindex(values.index)


def preprocess_factor_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    industry_col: str | None = None,
    neutralize_cols: Iterable[str] | None = None,
    weight_col: str | None = None,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Winsorize, neutralize, and standardize a factor cross-sectionally."""
    out = df.copy()
    target = output_col or f"{factor_col}_processed"
    parts = []
    neutral_cols = list(neutralize_cols or [])
    for _, group in out.groupby(date_col, sort=False):
        values = mad_winsorize(group[factor_col])
        controls = pd.DataFrame(index=group.index)
        if industry_col:
            controls = pd.concat(
                [controls, pd.get_dummies(group[industry_col], prefix=industry_col, dtype=float)],
                axis=1,
            )
        for col in neutral_cols:
            controls[col] = pd.to_numeric(group[col], errors="coerce")
        if controls.shape[1]:
            weights = None
            if weight_col and weight_col in group.columns:
                weights = pd.to_numeric(group[weight_col], errors="coerce")
            values = neutralize_cross_section(values, controls, weights=weights)
        parts.append(zscore(values).rename(target))
    out[target] = pd.concat(parts).sort_index()
    return out


def rank_ic_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Compute Spearman RankIC by date."""
    rows = []
    for date, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col):
        if group.shape[0] < min_assets:
            continue
        ic = group[factor_col].rank().corr(group[return_col].rank())
        rows.append({"date": date, "rank_ic": float(ic)})
    return pd.DataFrame(rows)


def rank_ic_fitness(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    min_assets: int = 20,
) -> float:
    """Mean RankIC fitness."""
    ic = rank_ic_by_date(df, date_col, factor_col, return_col, min_assets=min_assets)
    return float(ic["rank_ic"].mean()) if not ic.empty else np.nan


def discretized_mutual_information(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    bins: int = 10,
) -> float:
    """Estimate mutual information by quantile discretization."""
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if clean.shape[0] < bins * 2:
        return np.nan
    try:
        xb = pd.qcut(clean["x"].rank(method="first"), bins, labels=False)
        yb = pd.qcut(clean["y"].rank(method="first"), bins, labels=False)
    except ValueError:
        return np.nan
    joint = pd.crosstab(xb, yb).to_numpy(dtype=float)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / np.maximum(expected[mask], EPS))))


def mutual_information_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    bins: int = 10,
    min_assets: int = 50,
) -> pd.DataFrame:
    """Compute quantile mutual information by date."""
    rows = []
    for date, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col):
        if group.shape[0] < min_assets:
            continue
        mi = discretized_mutual_information(group[factor_col], group[return_col], bins=bins)
        if not pd.isna(mi):
            rows.append({"date": date, "mutual_information": mi})
    return pd.DataFrame(rows)


def mutual_information_fitness(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    bins: int = 10,
    min_assets: int = 50,
) -> float:
    """Mean mutual-information fitness."""
    mi = mutual_information_by_date(df, date_col, factor_col, return_col, bins=bins, min_assets=min_assets)
    return float(mi["mutual_information"].mean()) if not mi.empty else np.nan


def long_excess_fitness(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    return_col: str,
    buckets: int = 5,
    min_assets: int = 50,
) -> float:
    """Fitness based on the better side's mean excess return."""
    spreads = []
    for _, group in df.dropna(subset=[factor_col, return_col]).groupby(date_col):
        if group.shape[0] < max(min_assets, buckets):
            continue
        ranks = pd.qcut(group[factor_col].rank(method="first"), buckets, labels=False) + 1
        temp = group.assign(_bucket=ranks.astype(int))
        base = temp[return_col].mean()
        top = temp.loc[temp["_bucket"] == buckets, return_col].mean() - base
        bottom = temp.loc[temp["_bucket"] == 1, return_col].mean() - base
        spreads.append(max(top, bottom))
    return float(np.nanmean(spreads)) if spreads else np.nan


def cubic_residual_transform(values: pd.Series) -> pd.Series:
    """BARRA-style nonlinear transform: residual of x^3 on x."""
    x = zscore(pd.to_numeric(values, errors="coerce"))
    clean = pd.DataFrame({"x": x, "x3": x.pow(3)}).dropna()
    if clean.empty:
        return pd.Series(np.nan, index=values.index)
    beta = np.linalg.lstsq(clean[["x"]].to_numpy(dtype=float), clean["x3"].to_numpy(dtype=float), rcond=None)[0]
    residual = clean["x3"] - clean["x"] * beta[0]
    return zscore(residual).reindex(values.index)


def polynomial_factor_transform(
    values: pd.Series,
    coefficients: Sequence[float],
) -> pd.Series:
    """Apply a polynomial transform with coefficients ordered high-to-low."""
    x = pd.to_numeric(values, errors="coerce")
    transformed = pd.Series(np.polyval(list(coefficients), x), index=values.index)
    return zscore(transformed)


def fit_polynomial_transform(
    factor: pd.Series,
    future_return: pd.Series,
    degree: int = 3,
) -> np.ndarray:
    """Fit return on polynomial factor terms and return high-to-low coefficients."""
    clean = pd.DataFrame({"factor": factor, "return": future_return}).dropna()
    if clean.shape[0] <= degree + 1:
        return np.full(degree + 1, np.nan)
    return np.polyfit(clean["factor"].to_numpy(dtype=float), clean["return"].to_numpy(dtype=float), degree)


def validation_convergence(
    generation_stats: pd.DataFrame,
    generation_col: str = "generation",
    validation_col: str = "validation_fitness",
    patience: int = 2,
    min_delta: float = 0.0,
) -> dict[str, float | int | bool]:
    """Detect whether validation fitness has stopped improving."""
    clean = generation_stats.dropna(subset=[generation_col, validation_col]).sort_values(generation_col)
    if clean.empty:
        return {"best_generation": -1, "best_fitness": np.nan, "should_stop": False}
    values = clean[validation_col].to_numpy(dtype=float)
    best_idx = int(np.argmax(values))
    best_generation = int(clean.iloc[best_idx][generation_col])
    best_fitness = float(values[best_idx])
    tail = values[best_idx + 1 :]
    no_improve = len(tail) >= patience and np.all(tail[:patience] <= best_fitness + min_delta)
    return {"best_generation": best_generation, "best_fitness": best_fitness, "should_stop": bool(no_improve)}


def default_gp_function_set() -> pd.DataFrame:
    """Document the GP operator set used for price-volume factor mining."""
    rows = [
        ("add", "x + y", "basic"),
        ("sub", "x - y", "basic"),
        ("mul", "x * y", "basic"),
        ("div", "safe x / y", "basic"),
        ("abs", "abs(x)", "basic"),
        ("sqrt", "sqrt(abs(x))", "basic"),
        ("log", "log(abs(x))", "basic"),
        ("inv", "safe 1 / x", "basic"),
        ("rank", "cross-sectional percentile rank", "cross_section"),
        ("delay", "lagged value", "time_series"),
        ("ts_corr", "rolling correlation", "time_series"),
        ("ts_cov", "rolling covariance", "time_series"),
        ("delta", "x - delay(x, d)", "time_series"),
        ("decay_linear", "linearly decayed rolling mean", "time_series"),
        ("ts_min", "rolling minimum", "time_series"),
        ("ts_max", "rolling maximum", "time_series"),
        ("ts_rank", "rolling percentile rank", "time_series"),
        ("ts_sum", "rolling sum", "time_series"),
        ("ts_stddev", "rolling standard deviation", "time_series"),
        ("ts_zscore", "rolling mean / rolling standard deviation", "time_series"),
        ("rank_sub", "rank(x) - rank(y)", "cross_section"),
        ("rank_div", "safe rank(x) / rank(y)", "cross_section"),
        ("sigmoid", "1 / (1 + exp(-x))", "nonlinear"),
    ]
    return pd.DataFrame(rows, columns=["operator", "definition", "family"])


def alphanet_feature_list(version: str = "v2") -> list[str]:
    """Return raw AlphaNet feature names for v1/v2/v3."""
    base = ["return1", "open", "close", "high", "low", "volume", "vwap", "turn", "free_turn"]
    ratios = ["close/free_turn", "open/turn", "volume/low", "vwap/high", "low/high", "vwap/close"]
    if version.lower() == "v1":
        return base
    if version.lower() in {"v2", "v3"}:
        return base + ratios
    raise ValueError("version must be v1, v2, or v3.")


def alphanet_architecture(version: str = "v2") -> pd.DataFrame:
    """Return the documented AlphaNet architecture components."""
    version = version.lower()
    if version == "v1":
        rows = [
            ("feature_layer", "ts_corr/ts_cov/ts_stddev/ts_zscore/ts_return/ts_decaylinear/ts_mean + BN", "lookback=10, stride=10"),
            ("pooling_layer", "ts_mean/ts_max/ts_min + BN", "lookback=3, stride=3"),
            ("dense_hidden", "Dense(30), ReLU, Dropout(0.5)", "truncated_normal"),
            ("output", "Dense(1), linear", "MSE, RMSProp lr=0.0001, batch_size=1000"),
            ("split", "time ordered train/validation", "1:1"),
        ]
    elif version == "v2":
        rows = [
            ("feature_layer", "ts_corr/ts_cov/ts_stddev/ts_zscore/ts_return/ts_decaylinear + BN", "lookback=10, stride=10"),
            ("sequence_layer", "LSTM(30) + BN", "time_step=3"),
            ("output", "Dense(1), linear", "MSE, Adam lr=0.0001"),
            ("features", "v1 raw features + 6 ratio features", "15 x 30 image"),
            ("split", "time ordered train/validation", "4:1"),
        ]
    elif version == "v3":
        rows = [
            ("feature_layer_1", "v2 feature functions + BN", "lookback=10, stride=10"),
            ("feature_layer_2", "v2 feature functions + BN", "lookback=5, stride=5"),
            ("sequence_layer_1", "GRU(30) + BN", "time_step=3"),
            ("sequence_layer_2", "GRU(30) + BN", "time_step=6"),
            ("output", "Dense(1), linear", "MSE, Adam lr=0.0001"),
            ("split", "time ordered train/validation", "4:1"),
        ]
    else:
        raise ValueError("version must be v1, v2, or v3.")
    return pd.DataFrame(rows, columns=["component", "contents", "settings"])


def rolling_train_validation_windows(
    dates: Sequence[pd.Timestamp] | pd.Series,
    train_length: int = 1500,
    validation_ratio: float = 0.2,
    retrain_step: int = 126,
) -> pd.DataFrame:
    """Build time-ordered rolling train/validation/prediction windows."""
    unique_dates = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values().reset_index(drop=True)
    rows = []
    validation_length = max(1, int(round(train_length * validation_ratio)))
    fit_length = train_length - validation_length
    for end in range(train_length, len(unique_dates), retrain_step):
        train_start = end - train_length
        train_end = train_start + fit_length - 1
        valid_start = train_end + 1
        valid_end = end - 1
        pred_start = end
        pred_end = min(end + retrain_step - 1, len(unique_dates) - 1)
        rows.append(
            {
                "train_start": unique_dates.iloc[train_start],
                "train_end": unique_dates.iloc[train_end],
                "validation_start": unique_dates.iloc[valid_start],
                "validation_end": unique_dates.iloc[valid_end],
                "prediction_start": unique_dates.iloc[pred_start],
                "prediction_end": unique_dates.iloc[pred_end],
            }
        )
    return pd.DataFrame(rows)


def make_panel_image(
    asset_df: pd.DataFrame,
    date_col: str,
    feature_cols: Sequence[str],
    end_date,
    lookback: int = 30,
) -> np.ndarray:
    """Create a feature x time matrix for one asset ending at end_date."""
    work = asset_df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    end = pd.to_datetime(end_date)
    window = work.loc[work[date_col] <= end].sort_values(date_col).tail(lookback)
    if window.shape[0] < lookback:
        raise ValueError("Not enough history to build panel image.")
    return window[list(feature_cols)].to_numpy(dtype=float).T


def ai_factor_research_checklist() -> pd.DataFrame:
    """Quality gates for GP/AlphaNet factor-mining research."""
    rows = [
        ("timestamp", "Use only data available at the prediction date."),
        ("preprocess", "Winsorize, neutralize, and standardize before fitness tests."),
        ("fitness", "Compare RankIC, mutual information, and long-excess fitness."),
        ("complexity", "Penalize formula length/depth and monitor parsimony."),
        ("validation", "Use time-ordered validation, not random cross-section splits."),
        ("nonlinear", "Inspect middle-bucket winners and transform or route them to ML models."),
        ("seeds", "Average or stress test neural predictions across random seeds."),
        ("incremental", "Report industry/size/reversal/turnover/volatility neutralized results."),
        ("cost", "Include VWAP execution, fees, turnover, limit-up/down, and suspension filters."),
        ("decay", "Test IC decay and model degradation after publication or regime change."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp-config", action="store_true")
    parser.add_argument("--function-set", action="store_true")
    parser.add_argument("--alphanet", choices=["v1", "v2", "v3"])
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()

    if args.gp_config:
        print(pd.Series(asdict(GeneticProgrammingConfig())))
    if args.function_set:
        print(default_gp_function_set())
    if args.alphanet:
        print(alphanet_architecture(args.alphanet))
    if args.checklist:
        print(ai_factor_research_checklist())


if __name__ == "__main__":
    main()
