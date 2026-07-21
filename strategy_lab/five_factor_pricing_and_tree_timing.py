#!/usr/bin/env python
"""A-share factor-pricing and regression-tree timing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd


EPS = 1e-12


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(values.mean())
    weights = pd.to_numeric(weights, errors="coerce").reindex(values.index).fillna(0.0).clip(lower=0.0)
    mask = values.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def two_by_three_factor_returns(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    size_col: str,
    factor_specs: dict[str, dict[str, object]],
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Construct 2x3 size-by-characteristic factor returns.

    `factor_specs` maps output factor name to:
    - `column`: raw characteristic column.
    - `direction`: 1 for high-minus-low, -1 for low-minus-high.
    """
    rows: list[dict[str, object]] = []
    needed = [date_col, return_col, size_col, *[str(spec["column"]) for spec in factor_specs.values()]]
    if weight_col:
        needed.append(weight_col)
    data = panel[needed].copy()
    for date, group in data.groupby(date_col, sort=True):
        row: dict[str, object] = {"date": date}
        size = pd.to_numeric(group[size_col], errors="coerce")
        size_cut = size.median()
        small = group[size <= size_cut]
        big = group[size > size_cut]
        smb_parts = []
        for factor_name, spec in factor_specs.items():
            col = str(spec["column"])
            direction = float(spec.get("direction", 1.0))
            char = pd.to_numeric(group[col], errors="coerce")
            low_cut = char.quantile(0.30)
            high_cut = char.quantile(0.70)
            factor_returns = {}
            for size_name, size_group in {"S": small, "B": big}.items():
                low = size_group[pd.to_numeric(size_group[col], errors="coerce") <= low_cut]
                high = size_group[pd.to_numeric(size_group[col], errors="coerce") >= high_cut]
                weight_low = low[weight_col] if weight_col else None
                weight_high = high[weight_col] if weight_col else None
                factor_returns[(size_name, "L")] = _weighted_mean(low[return_col], weight_low)
                factor_returns[(size_name, "H")] = _weighted_mean(high[return_col], weight_high)
                smb_parts.append(_weighted_mean(size_group[return_col], size_group[weight_col] if weight_col else None))
            high_leg = np.nanmean([factor_returns[("S", "H")], factor_returns[("B", "H")]])
            low_leg = np.nanmean([factor_returns[("S", "L")], factor_returns[("B", "L")]])
            row[factor_name] = direction * (high_leg - low_leg)
        row["SMB"] = _weighted_mean(small[return_col], small[weight_col] if weight_col else None) - _weighted_mean(
            big[return_col], big[weight_col] if weight_col else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def a_share_five_factor_spec(
    pe_col: str = "pe",
    sue_col: str = "sue",
    turnover_col: str = "turnover",
) -> dict[str, dict[str, object]]:
    """Default A-share five-factor characteristic specs from the report."""
    return {
        "FPE": {"column": pe_col, "direction": -1.0},
        "FSUE": {"column": sue_col, "direction": 1.0},
        "FTurn": {"column": turnover_col, "direction": -1.0},
    }


def factor_model_regression(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.Series:
    """Regress a strategy, event, anomaly, or fund return on factor returns."""
    factors = list(factor_cols)
    clean = pd.concat([returns.rename("_ret"), factor_returns[factors]], axis=1).dropna()
    if clean.shape[0] <= len(factors) + 2:
        raise ValueError("Not enough observations for factor regression.")
    y = clean["_ret"].to_numpy(dtype=float)
    x = clean[factors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(clean.shape[0]), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    dof = max(clean.shape[0] - x.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > EPS else np.nan
    out = {"alpha": float(beta[0]), "alpha_t": float(beta[0] / se[0]) if se[0] > EPS else np.nan, "r2": r2}
    for col, value, stderr in zip(factors, beta[1:], se[1:]):
        out[f"{col}_beta"] = float(value)
        out[f"{col}_t"] = float(value / stderr) if stderr > EPS else np.nan
    return pd.Series(out)


def factor_return_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    """Decompose strategy return into fitted factor contribution and residual."""
    stats = factor_model_regression(strategy_returns, factor_returns, factor_cols)
    data = pd.concat([strategy_returns.rename("strategy_return"), factor_returns[list(factor_cols)]], axis=1).dropna()
    out = data[["strategy_return"]].copy()
    fitted = pd.Series(stats["alpha"], index=out.index, dtype="float64")
    out["alpha_component"] = stats["alpha"]
    for col in factor_cols:
        contrib = data[col] * stats[f"{col}_beta"]
        out[f"{col}_contribution"] = contrib
        fitted = fitted + contrib
    out["fitted_return"] = fitted
    out["residual_return"] = out["strategy_return"] - out["fitted_return"]
    return out


def bic_model_posterior(
    factor_returns: pd.DataFrame,
    candidate_factors: Sequence[str],
    required_factors: Sequence[str] = ("Mkt",),
    max_optional_factors: int | None = None,
) -> pd.DataFrame:
    """Approximate model posterior from BIC for parsimonious factor comparison.

    For each model, omitted candidate factors are treated as test assets and
    regressed on included factors. Lower residual BIC receives higher posterior
    weight. This is not a full Barillas-Shanken implementation, but it preserves
    the report's practical rule: penalize redundant factors.
    """
    required = list(required_factors)
    optional = [col for col in candidate_factors if col not in required]
    max_optional_factors = len(optional) if max_optional_factors is None else max_optional_factors
    rows: list[dict[str, object]] = []
    for size in range(0, max_optional_factors + 1):
        for subset in combinations(optional, size):
            included = [*required, *subset]
            omitted = [col for col in candidate_factors if col not in included]
            data = factor_returns[[*included, *omitted]].dropna()
            if data.shape[0] <= len(included) + 2:
                continue
            bic = 0.0
            if omitted:
                for target in omitted:
                    y = data[target].to_numpy(dtype=float)
                    x = data[included].to_numpy(dtype=float)
                    x = np.column_stack([np.ones(data.shape[0]), x])
                    beta = np.linalg.lstsq(x, y, rcond=None)[0]
                    resid = y - x @ beta
                    sigma2 = max(float(resid @ resid / data.shape[0]), EPS)
                    bic += data.shape[0] * np.log(sigma2) + x.shape[1] * np.log(data.shape[0])
            else:
                bic = len(included) * np.log(data.shape[0])
            rows.append({"model": ",".join(included), "n_factors": len(included), "bic": float(bic)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    score = np.exp(-0.5 * (out["bic"] - out["bic"].min()))
    out["posterior"] = score / score.sum()
    return out.sort_values("posterior", ascending=False).reset_index(drop=True)


def _tree_leaf(value: float, n_obs: int) -> dict[str, object]:
    return {"type": "leaf", "value": float(value), "n_obs": int(n_obs)}


def _weighted_sse(y: np.ndarray, weights: np.ndarray) -> float:
    if y.size == 0 or weights.sum() <= EPS:
        return np.inf
    mean = np.average(y, weights=weights)
    return float(np.sum(weights * (y - mean) ** 2))


def _build_tree(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    depth: int,
    max_depth: int,
    min_leaf: int,
    weights: np.ndarray,
) -> dict[str, object]:
    y = pd.to_numeric(data[target_col], errors="coerce").to_numpy(dtype=float)
    value = float(np.average(y, weights=weights)) if weights.sum() > EPS else float(np.nanmean(y))
    if depth >= max_depth or data.shape[0] < min_leaf * 2:
        return _tree_leaf(value, data.shape[0])

    best: dict[str, object] | None = None
    parent_sse = _weighted_sse(y, weights)
    for col in feature_cols:
        x = pd.to_numeric(data[col], errors="coerce")
        thresholds = np.unique(np.nanquantile(x.dropna(), [0.2, 0.35, 0.5, 0.65, 0.8])) if x.notna().sum() else []
        for threshold in thresholds:
            left_mask = x <= threshold
            right_mask = x > threshold
            if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                continue
            left_sse = _weighted_sse(y[left_mask.to_numpy()], weights[left_mask.to_numpy()])
            right_sse = _weighted_sse(y[right_mask.to_numpy()], weights[right_mask.to_numpy()])
            gain = parent_sse - left_sse - right_sse
            if best is None or gain > best["gain"]:
                best = {"feature": col, "threshold": float(threshold), "gain": float(gain), "left": left_mask, "right": right_mask}
    if best is None or best["gain"] <= EPS:
        return _tree_leaf(value, data.shape[0])
    left_data = data.loc[best["left"]].copy()
    right_data = data.loc[best["right"]].copy()
    return {
        "type": "node",
        "feature": best["feature"],
        "threshold": best["threshold"],
        "gain": best["gain"],
        "n_obs": int(data.shape[0]),
        "value": value,
        "left": _build_tree(left_data, target_col, feature_cols, depth + 1, max_depth, min_leaf, weights[best["left"].to_numpy()]),
        "right": _build_tree(right_data, target_col, feature_cols, depth + 1, max_depth, min_leaf, weights[best["right"].to_numpy()]),
    }


def fit_regression_tree(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    max_depth: int = 2,
    min_leaf: int = 12,
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit a small dependency-free CART-style regression tree."""
    features = list(feature_cols)
    clean = df[[target_col, *features]].dropna().copy()
    if clean.shape[0] < min_leaf * 2:
        raise ValueError("Not enough observations for regression tree.")
    if sample_weight is None:
        weights = np.ones(clean.shape[0], dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)[-clean.shape[0] :]
    weights = weights / max(weights.sum(), EPS)
    return _build_tree(clean, target_col, features, 0, max_depth, min_leaf, weights)


def predict_regression_tree(tree: dict[str, object], row: pd.Series) -> float:
    """Predict one row from a tree returned by fit_regression_tree."""
    node = tree
    while node.get("type") == "node":
        feature = str(node["feature"])
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(node["value"])
        node = node["left"] if float(value) <= float(node["threshold"]) else node["right"]
    return float(node["value"])


def half_life_weights(length: int, half_life: float) -> np.ndarray:
    """Oldest-to-newest half-life weights."""
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = 0.5 ** (ages / float(half_life))
    return weights / weights.sum()


def rolling_tree_factor_timing(
    factor_returns: pd.DataFrame,
    timing_features: pd.DataFrame,
    date_col: str,
    factor_col: str,
    feature_cols: Iterable[str],
    window: int = 60,
    min_history: int = 36,
    max_depth: int = 2,
    min_leaf: int = 8,
    half_life: float | None = None,
    output_col: str = "tree_prediction",
) -> pd.DataFrame:
    """Rolling regression-tree prediction for next factor return."""
    features = list(feature_cols)
    data = pd.merge(
        factor_returns[[date_col, factor_col]],
        timing_features[[date_col, *features]],
        on=date_col,
        how="inner",
    ).sort_values(date_col)
    rows: list[dict[str, object]] = []
    for pos in range(len(data)):
        hist = data.iloc[max(0, pos - window) : pos].dropna(subset=[factor_col, *features])
        if hist.shape[0] < min_history:
            continue
        weights = half_life_weights(hist.shape[0], half_life) if half_life else None
        try:
            tree = fit_regression_tree(hist, factor_col, features, max_depth=max_depth, min_leaf=min_leaf, sample_weight=weights)
        except ValueError:
            continue
        current = data.iloc[pos]
        if current[features].isna().any():
            continue
        rows.append(
            {
                "date": current[date_col],
                "factor": factor_col,
                output_col: predict_regression_tree(tree, current),
                "tree_depth": max_depth,
                "training_obs": int(hist.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def factor_return_momentum_prediction(
    factor_returns: pd.DataFrame,
    date_col: str,
    factor_col: str,
    lookback: int = 12,
    output_col: str = "momentum_prediction",
) -> pd.DataFrame:
    """Trailing factor-return mean used as a simple factor momentum forecast."""
    out = factor_returns.sort_values(date_col).copy()
    out[output_col] = pd.to_numeric(out[factor_col], errors="coerce").shift(1).rolling(lookback, min_periods=lookback // 2).mean()
    return out[[date_col, output_col]]


def defensive_timing_signal(
    predictions: pd.DataFrame,
    momentum_col: str = "momentum_prediction",
    tree_col: str = "tree_prediction",
    output_col: str = "defensive_prediction",
) -> pd.DataFrame:
    """Set factor forecast to zero unless momentum and tree directions agree."""
    out = predictions.copy()
    mom = pd.to_numeric(out[momentum_col], errors="coerce")
    tree = pd.to_numeric(out[tree_col], errors="coerce")
    agree = np.sign(mom) == np.sign(tree)
    out[output_col] = np.where(agree & mom.notna() & tree.notna(), tree, 0.0)
    out[f"{output_col}_active"] = (out[output_col].abs() > EPS).astype(float)
    return out


def pricing_and_timing_checklist() -> pd.DataFrame:
    rows = [
        ("parsimony", "Factor-pricing models should penalize redundant factors, not maximize raw R2."),
        ("factor_form", "A-share evidence favors PE over PB and SUE over ROE in the studied sample."),
        ("attribution", "Event, anomaly, and fund alpha must be tested after factor exposure attribution."),
        ("tree_timing", "Regression trees are scenario tools; require rolling out-of-sample validation."),
        ("defensive", "Use tree timing defensively when it agrees with factor-return momentum; otherwise close exposure."),
        ("ensemble", "Single trees are unstable; ensemble methods are a natural robustness extension."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
