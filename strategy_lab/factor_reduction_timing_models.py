#!/usr/bin/env python
"""Factor reduction and factor-timing research helpers.

This module converts the Haitong reports on bottom-level factor reduction,
factor-timing indicator libraries, and lasso/elastic-net timing models into
dependency-light pandas code.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def half_life_weights(length: int, half_life: float) -> np.ndarray:
    """Return normalized recency weights where weight halves every half_life periods."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if half_life <= 0:
        raise ValueError("half_life must be positive.")
    age = np.arange(length - 1, -1, -1, dtype=float)
    raw = 0.5 ** (age / float(half_life))
    return raw / raw.sum()


def zscore_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    suffix: str = "_z",
) -> pd.DataFrame:
    """Cross-sectionally z-score factor columns by date."""
    out = df.copy()
    for col in factor_cols:
        target = f"{col}{suffix}"

        def zscore(values: pd.Series) -> pd.Series:
            values = pd.to_numeric(values, errors="coerce")
            std = values.std(ddof=1)
            if pd.isna(std) or std <= EPS:
                return values * 0.0
            return (values - values.mean()) / std

        out[target] = out.groupby(date_col)[col].transform(zscore)
    return out


def rolling_category_ic_weights(
    ic_df: pd.DataFrame,
    date_col: str,
    category_col: str,
    factor_col: str,
    ic_col: str = "rank_ic",
    lookback: int = 24,
    method: str = "best",
    min_history: int = 6,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Create category reduction weights from trailing factor IC.

    method='best' selects the factor with the largest absolute trailing IC mean.
    method='weighted' assigns signed weights proportional to trailing IC means.
    Current-period IC is excluded from current-period weights.
    """
    if method not in {"best", "weighted"}:
        raise ValueError("method must be 'best' or 'weighted'.")
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    data = ic_df[[date_col, category_col, factor_col, ic_col]].dropna().copy()
    data = data.sort_values(date_col)
    all_dates = pd.Index(sorted(data[date_col].dropna().unique()))
    rows: list[dict[str, object]] = []

    for category, group in data.groupby(category_col, sort=True):
        matrix = group.pivot_table(index=date_col, columns=factor_col, values=ic_col, aggfunc="mean").reindex(all_dates)
        for pos, date in enumerate(matrix.index):
            hist = matrix.iloc[max(0, pos - lookback) : pos].dropna(how="all")
            if hist.shape[0] < min_history:
                weights = pd.Series(1.0 / len(matrix.columns), index=matrix.columns)
            else:
                if half_life is None:
                    score = hist.mean(skipna=True)
                else:
                    w = half_life_weights(hist.shape[0], half_life)
                    score = hist.mul(w, axis=0).sum(axis=0) / hist.notna().mul(w, axis=0).sum(axis=0)
                score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if method == "best":
                    weights = pd.Series(0.0, index=matrix.columns)
                    chosen = score.abs().idxmax()
                    sign = np.sign(score.loc[chosen])
                    weights.loc[chosen] = float(sign if sign != 0 else 1.0)
                else:
                    denom = score.abs().sum()
                    weights = score / denom if denom > EPS else pd.Series(1.0 / len(score), index=score.index)
            for factor, weight in weights.items():
                rows.append(
                    {
                        "date": date,
                        "category": category,
                        "factor": factor,
                        "weight": float(weight),
                        "method": method,
                    }
                )
    return pd.DataFrame(rows)


def rolling_category_pca_ic_weights(
    ic_df: pd.DataFrame,
    date_col: str,
    category_col: str,
    factor_col: str,
    ic_col: str = "rank_ic",
    lookback: int = 24,
    min_history: int = 6,
) -> pd.DataFrame:
    """Create category weights from the first principal component of IC series.

    The Haitong evidence suggests this is usually weaker than best-IC or
    IC-weighted reduction, so this helper is mainly for replication comparison.
    """
    data = ic_df[[date_col, category_col, factor_col, ic_col]].dropna().copy()
    data = data.sort_values(date_col)
    all_dates = pd.Index(sorted(data[date_col].dropna().unique()))
    rows: list[dict[str, object]] = []
    for category, group in data.groupby(category_col, sort=True):
        matrix = group.pivot_table(index=date_col, columns=factor_col, values=ic_col, aggfunc="mean").reindex(all_dates)
        for pos, date in enumerate(matrix.index):
            hist = matrix.iloc[max(0, pos - lookback) : pos].dropna(how="all")
            if hist.shape[0] < min_history or hist.shape[1] == 1:
                weights = pd.Series(1.0 / len(matrix.columns), index=matrix.columns)
            else:
                filled = hist.apply(lambda s: s.fillna(s.mean()), axis=0).fillna(0.0)
                cov = filled.cov().to_numpy(dtype=float)
                eigval, eigvec = np.linalg.eigh(cov)
                loading = eigvec[:, int(np.argmax(eigval))]
                if np.nansum(loading) < 0:
                    loading = -loading
                denom = np.sum(np.abs(loading))
                weights = pd.Series(loading / denom if denom > EPS else 1.0 / len(loading), index=filled.columns)
                weights = weights.reindex(matrix.columns).fillna(0.0)
            for factor, weight in weights.items():
                rows.append(
                    {
                        "date": date,
                        "category": category,
                        "factor": factor,
                        "weight": float(weight),
                        "method": "pca_ic",
                    }
                )
    return pd.DataFrame(rows)


def apply_category_reduction(
    df: pd.DataFrame,
    date_col: str,
    factor_cols_by_category: Mapping[str, Iterable[str]],
    weights_df: pd.DataFrame,
    zscore_inputs: bool = True,
    output_prefix: str = "cat",
) -> pd.DataFrame:
    """Apply category reduction weights to a stock-factor panel."""
    out = df.copy()
    factor_cols = [col for cols in factor_cols_by_category.values() for col in cols]
    used_cols = {col: col for col in factor_cols}
    if zscore_inputs:
        out = zscore_by_date(out, date_col, factor_cols)
        used_cols = {col: f"{col}_z" for col in factor_cols}

    required = {"date", "category", "factor", "weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing columns: {sorted(missing)}")
    weight_lookup = {
        (date, category): group.set_index("factor")["weight"]
        for (date, category), group in weights_df.groupby(["date", "category"])
    }

    for category, cols_iter in factor_cols_by_category.items():
        cols = list(cols_iter)
        target = f"{output_prefix}_{category}"
        out[target] = np.nan
        for date, group in out.groupby(date_col, sort=True):
            raw_weights = weight_lookup.get((date, category), pd.Series(1.0 / len(cols), index=cols))
            weights = raw_weights.reindex(cols).fillna(0.0)
            denom = weights.abs().sum()
            if denom <= EPS:
                weights = pd.Series(1.0 / len(cols), index=cols)
            else:
                weights = weights / denom
            values = group[[used_cols[col] for col in cols]].to_numpy(dtype=float)
            out.loc[group.index, target] = values @ weights.to_numpy(dtype=float)
    return out


def lagged_cross_correlation(
    timing_df: pd.DataFrame,
    date_col: str,
    feature_col: str,
    factor_return_col: str,
    forward: int = 1,
) -> float:
    """Correlation between a timing indicator at t and factor return at t+forward."""
    data = timing_df.sort_values(date_col)[[feature_col, factor_return_col]].copy()
    data["_future_return"] = pd.to_numeric(data[factor_return_col], errors="coerce").shift(-forward)
    return float(pd.to_numeric(data[feature_col], errors="coerce").corr(data["_future_return"]))


def rolling_lagged_correlation(
    timing_df: pd.DataFrame,
    date_col: str,
    feature_col: str,
    factor_return_col: str,
    window: int = 36,
    forward: int = 1,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Rolling version of lagged_cross_correlation."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    out = timing_df.sort_values(date_col).copy()
    feature = pd.to_numeric(out[feature_col], errors="coerce")
    future = pd.to_numeric(out[factor_return_col], errors="coerce").shift(-forward)
    target = output_col or f"corr_{feature_col}_{factor_return_col}"
    out[target] = feature.rolling(window, min_periods=window).corr(future)
    return out


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_regularized_linear(
    x: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
    l1_ratio: float = 1.0,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> pd.Series:
    """Fit lasso/elastic-net linear regression with coordinate descent."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be in [0, 1].")
    clean = pd.concat([y.rename("_y"), x], axis=1).dropna()
    if clean.shape[0] <= x.shape[1] + 2:
        raise ValueError("Not enough observations for regularized regression.")
    yv = clean["_y"].to_numpy(dtype=float)
    xv = clean[x.columns].to_numpy(dtype=float)
    if sample_weight is None:
        weights = np.ones(len(yv), dtype=float) / len(yv)
    else:
        weights = np.asarray(sample_weight, dtype=float)[-len(yv) :]
        weights = weights / weights.sum()

    x_mean = np.average(xv, axis=0, weights=weights)
    y_mean = float(np.average(yv, weights=weights))
    x_centered = xv - x_mean
    x_std = np.sqrt(np.average(x_centered**2, axis=0, weights=weights))
    x_std = np.where(x_std <= EPS, 1.0, x_std)
    xs = x_centered / x_std
    yc = yv - y_mean

    beta = np.zeros(xs.shape[1], dtype=float)
    col_norm = np.sum(weights[:, None] * xs * xs, axis=0)
    for _ in range(max_iter):
        old = beta.copy()
        for j in range(xs.shape[1]):
            residual = yc - xs @ beta + xs[:, j] * beta[j]
            rho = float(np.sum(weights * xs[:, j] * residual))
            beta[j] = _soft_threshold(rho, alpha * l1_ratio) / (col_norm[j] + alpha * (1.0 - l1_ratio) + EPS)
        if np.max(np.abs(beta - old)) < tol:
            break

    coef = beta / x_std
    intercept = y_mean - float(np.sum(coef * x_mean))
    return pd.Series([intercept, *coef], index=["intercept", *x.columns])


def rolling_regularized_factor_timing(
    factor_returns: pd.DataFrame,
    timing_features: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    feature_cols: Iterable[str],
    window: int = 24,
    alpha: float = 0.05,
    l1_ratio: float = 1.0,
    half_life: float | None = None,
    min_history: int | None = None,
) -> pd.DataFrame:
    """Predict factor returns from timing indicators with rolling lasso/elastic net.

    factor_returns should already be aligned so row t is the target return to be
    predicted using row t timing features. The function excludes the current row
    from training and predicts at the current row.
    """
    factors = list(factor_cols)
    features = list(feature_cols)
    min_history = min_history or max(12, len(features) + 6)
    data = pd.merge(
        factor_returns[[date_col, *factors]],
        timing_features[[date_col, *features]],
        on=date_col,
        how="inner",
    ).sort_values(date_col)

    rows: list[dict[str, object]] = []
    for pos, row in data.iterrows():
        loc = data.index.get_loc(pos)
        hist = data.iloc[max(0, loc - window) : loc].dropna(subset=features, how="any")
        if hist.shape[0] < min_history:
            continue
        weights = half_life_weights(hist.shape[0], half_life) if half_life else None
        current_x = row[features].apply(pd.to_numeric, errors="coerce")
        if current_x.isna().any():
            continue
        for factor in factors:
            train = hist[[factor, *features]].dropna()
            if train.shape[0] < min_history:
                continue
            local_weights = weights[-train.shape[0] :] if weights is not None else None
            try:
                coef = fit_regularized_linear(
                    train[features],
                    train[factor],
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    sample_weight=local_weights,
                )
            except ValueError:
                continue
            pred = float(coef["intercept"] + np.dot(current_x.to_numpy(dtype=float), coef[features].to_numpy(dtype=float)))
            rows.append(
                {
                    "date": row[date_col],
                    "factor": factor,
                    "prediction": pred,
                    "nonzero_features": int((coef[features].abs() > EPS).sum()),
                    "method": "lasso" if l1_ratio == 1.0 else "elastic_net",
                }
            )
    return pd.DataFrame(rows)


def fit_logistic_regression(
    x: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.01,
    learning_rate: float = 0.1,
    max_iter: int = 3000,
    tol: float = 1e-7,
) -> pd.Series:
    """Fit a small logistic model for style probability research."""
    clean = pd.concat([y.rename("_y"), x], axis=1).dropna()
    if clean.shape[0] <= x.shape[1] + 2:
        raise ValueError("Not enough observations for logistic regression.")
    yv = clean["_y"].astype(float).clip(0.0, 1.0).to_numpy(dtype=float)
    xv_raw = clean[x.columns].to_numpy(dtype=float)
    mean = xv_raw.mean(axis=0)
    std = xv_raw.std(axis=0, ddof=1)
    std = np.where(std <= EPS, 1.0, std)
    xv = (xv_raw - mean) / std
    design = np.column_stack([np.ones(xv.shape[0]), xv])
    beta = np.zeros(design.shape[1], dtype=float)
    last_loss = np.inf
    for step in range(max_iter):
        logits = np.clip(design @ beta, -40, 40)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = design.T @ (prob - yv) / len(yv)
        grad[1:] += alpha * beta[1:]
        beta -= learning_rate / np.sqrt(step + 1.0) * grad
        if step % 50 == 0:
            logits = np.clip(design @ beta, -40, 40)
            prob = 1.0 / (1.0 + np.exp(-logits))
            loss = -np.mean(yv * np.log(prob + EPS) + (1.0 - yv) * np.log(1.0 - prob + EPS))
            if abs(last_loss - loss) < tol:
                break
            last_loss = loss
    coef = beta[1:] / std
    intercept = beta[0] - np.sum(beta[1:] * mean / std)
    return pd.Series([intercept, *coef], index=["intercept", *x.columns])


def rolling_style_probability(
    data: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: Iterable[str],
    window: int = 36,
    alpha: float = 0.01,
    min_history: int | None = None,
    output_col: str = "style_probability",
) -> pd.DataFrame:
    """Rolling logistic probability that next style return/IC is positive."""
    features = list(feature_cols)
    min_history = min_history or max(18, len(features) + 8)
    ordered = data.sort_values(date_col).copy()
    ordered[output_col] = np.nan
    for pos in range(len(ordered)):
        hist = ordered.iloc[max(0, pos - window) : pos][[target_col, *features]].dropna()
        if hist.shape[0] < min_history:
            continue
        current = ordered.iloc[pos][features].apply(pd.to_numeric, errors="coerce")
        if current.isna().any():
            continue
        try:
            coef = fit_logistic_regression(hist[features], (hist[target_col] > 0).astype(float), alpha=alpha)
        except ValueError:
            continue
        logit = float(coef["intercept"] + np.dot(current.to_numpy(dtype=float), coef[features].to_numpy(dtype=float)))
        ordered.iloc[pos, ordered.columns.get_loc(output_col)] = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
    return ordered


def factor_timing_indicator_catalog() -> pd.DataFrame:
    """Report-derived factor timing indicator map."""
    rows = [
        ("size", "PPI YoY", "+", "Higher PPI tends to favor large-cap exposure.", "macro_inflation"),
        ("size", "short policy-bank yield change", "+", "Rising short rates tend to favor large caps.", "bond_rate"),
        ("size", "broad-index volatility", "-", "Higher volatility tends to favor small caps in this sample.", "equity_vol"),
        ("size", "return dispersion", "-", "High dispersion tends to favor small caps.", "equity_cross_section"),
        ("midcap", "HS300 minus All-A return spread", "-", "High large-cap relative return tends to favor mid caps.", "equity_return"),
        ("midcap", "HS300 minus All-A PB spread", "-", "High large-cap valuation spread tends to favor mid caps.", "equity_valuation"),
        ("liquidity", "All-A one-month return", "-", "Weak market return tends to favor high-turnover stocks.", "equity_return"),
        ("liquidity", "volatility spread change", "-", "Falling volatility spread tends to favor high-turnover stocks.", "equity_vol"),
        ("reversal", "credit spread change", "-", "Widening credit spread tends to favor reversal.", "bond_credit"),
        ("reversal", "CSI500 volatility", "-", "High volatility tends to favor reversal over momentum.", "equity_vol"),
        ("volatility", "three-month broad-index volatility", "+", "High market volatility tends to favor high systematic-volatility share.", "equity_vol"),
        ("value", "trade growth change", "+", "Falling trade growth is interpreted as growth preference in the report.", "macro_trade"),
        ("value", "TED spread", "+", "High TED spread tends to favor growth over value in the report definition.", "bond_credit"),
        ("value", "CSI500 three-month return", "-", "Strong recent market tends to favor value.", "equity_return"),
        ("profitability", "TED spread", "+", "High risk premium tends to favor high profitability.", "bond_credit"),
        ("profitability", "policy-bank yield change", "+", "Rising rates tend to favor high profitability.", "bond_rate"),
        ("profitability", "return dispersion change", "-", "Narrowing dispersion tends to favor profitability.", "equity_cross_section"),
        ("profit_growth", "CPI YoY", "+", "Higher CPI tends to favor high profit growth.", "macro_inflation"),
        ("profit_growth", "CSI500 PB", "+", "Higher CSI500 valuation tends to favor profit growth.", "equity_valuation"),
    ]
    return pd.DataFrame(rows, columns=["factor", "indicator", "direction", "interpretation", "source_layer"])


def factor_reduction_timing_checklist() -> pd.DataFrame:
    rows = [
        ("category", "Group factors by economic logic before reduction; PCA on mixed factors is hard to interpret."),
        ("lookahead", "Use only trailing IC/returns for reduction and timing weights."),
        ("orthogonal_ic", "Prefer orthogonalized IC when comparing factors across already selected categories."),
        ("pca", "Treat IC-series PCA as a benchmark, not a default production method."),
        ("indicator_library", "Separate macro, bond, equity-market, and factor-history indicators."),
        ("selection", "Use lasso/elastic net for dynamic indicator selection, but report selected variables and stability."),
        ("insurance", "Factor timing may sacrifice full-sample return to reduce bad-regime losses."),
        ("probability", "Style probability models need threshold and turnover validation before rotation use."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.catalog:
        print(factor_timing_indicator_catalog())
    if args.checklist:
        print(factor_reduction_timing_checklist())


if __name__ == "__main__":
    main()
