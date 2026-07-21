#!/usr/bin/env python
"""Research governance helpers for quant experiments.

The goal is to turn an idea into a reproducible, auditable experiment before
spending effort on a full backtest or model search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManifestGate:
    section: str
    field: str
    severity: str
    rationale: str


MANIFEST_GATES = [
    ManifestGate("idea", "hypothesis", "fail", "Research hypothesis must be explicit and falsifiable."),
    ManifestGate("object", "asset_pool", "fail", "Asset pool defines what result can be generalized to."),
    ManifestGate("object", "benchmark", "fail", "Benchmark must match asset pool and style exposure."),
    ManifestGate("object", "exclusion_rules", "warn", "ST, suspension, new-listing and delisting rules affect A-share tests."),
    ManifestGate("data", "data_sources", "fail", "Data source and field definitions must be reproducible."),
    ManifestGate("data", "timestamp_policy", "fail", "Information availability time prevents look-ahead bias."),
    ManifestGate("data", "survivorship_policy", "warn", "Historical constituents and delisted names change results."),
    ManifestGate("signal", "signal_definition", "fail", "Indicator/factor/signal formula must be calculable without labels."),
    ManifestGate("signal", "label_definition", "warn", "Future return labels are for evaluation only, not signal construction."),
    ManifestGate("portfolio", "rebalance_rule", "fail", "A strategy needs an explicit when-to-trade rule."),
    ManifestGate("portfolio", "position_rule", "fail", "A signal is not a strategy until translated into weights."),
    ManifestGate("execution", "execution_price", "fail", "Execution price determines whether the backtest is tradable."),
    ManifestGate("execution", "cost_model", "fail", "Costs, slippage and capacity can reverse paper alpha."),
    ManifestGate("validation", "out_of_sample_plan", "fail", "In-sample validation alone is not evidence."),
    ManifestGate("validation", "parameter_sensitivity", "warn", "Fragile parameter peaks indicate data mining risk."),
    ManifestGate("validation", "multiple_testing_control", "warn", "Many tried variants require PBO/FDR/holdout discipline."),
    ManifestGate("risk", "risk_controls", "warn", "Size, industry, beta, liquidity and crowding exposure need attribution."),
    ManifestGate("risk", "failure_modes", "warn", "A usable model states when it should not be trusted."),
    ManifestGate("operations", "live_monitoring", "warn", "Live deployment needs drift, turnover and cost monitors."),
    ManifestGate("operations", "human_review", "warn", "LLM/agent-assisted research needs human signoff and audit logs."),
]


def build_experiment_manifest(
    hypothesis: str,
    asset_pool: str,
    benchmark: str,
    signal_definition: str,
    data_sources: list[str] | str,
    rebalance_rule: str,
    position_rule: str,
    execution_price: str,
    cost_model: str,
    out_of_sample_plan: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a simple flat manifest for one quant experiment."""
    manifest: dict[str, Any] = {
        "hypothesis": hypothesis,
        "asset_pool": asset_pool,
        "benchmark": benchmark,
        "signal_definition": signal_definition,
        "data_sources": data_sources,
        "rebalance_rule": rebalance_rule,
        "position_rule": position_rule,
        "execution_price": execution_price,
        "cost_model": cost_model,
        "out_of_sample_plan": out_of_sample_plan,
    }
    manifest.update(kwargs)
    return manifest


def research_quality_gates() -> pd.DataFrame:
    """Return the reusable research audit checklist."""
    return pd.DataFrame([gate.__dict__ for gate in MANIFEST_GATES])


def audit_experiment_manifest(manifest: dict[str, Any]) -> pd.DataFrame:
    """Check whether a manifest passes the minimum research governance gates."""
    rows = []
    for gate in MANIFEST_GATES:
        value = manifest.get(gate.field)
        present = value is not None and not (isinstance(value, str) and not value.strip())
        status = "pass" if present else gate.severity
        rows.append(
            {
                "section": gate.section,
                "field": gate.field,
                "status": status,
                "present": bool(present),
                "rationale": gate.rationale,
            }
        )
    return pd.DataFrame(rows)


def chronological_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_fraction: float = 0.7,
    embargo_periods: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, never by random shuffle."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be non-negative.")
    data = df.sort_values(date_col).copy()
    dates = pd.Index(pd.to_datetime(data[date_col]).drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("Need at least three unique dates for a train/test split.")
    cut = int(len(dates) * train_fraction)
    train_dates = dates[:cut]
    test_dates = dates[min(cut + embargo_periods, len(dates)) :]
    train = data[pd.to_datetime(data[date_col]).isin(train_dates)].copy()
    test = data[pd.to_datetime(data[date_col]).isin(test_dates)].copy()
    return train, test


def walk_forward_splits(
    dates: pd.Series | pd.Index | list,
    train_window: int,
    test_window: int,
    step: int | None = None,
    expanding: bool = False,
    embargo: int = 0,
) -> pd.DataFrame:
    """Generate walk-forward split boundaries on unique sorted dates."""
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive.")
    if embargo < 0:
        raise ValueError("embargo must be non-negative.")
    step = step or test_window
    unique_dates = pd.Index(pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values())
    rows = []
    start = 0
    split_id = 0
    while True:
        train_start = 0 if expanding else start
        train_end = start + train_window
        test_start = train_end + embargo
        test_end = test_start + test_window
        if test_end > len(unique_dates):
            break
        rows.append(
            {
                "split_id": split_id,
                "train_start": unique_dates[train_start],
                "train_end": unique_dates[train_end - 1],
                "test_start": unique_dates[test_start],
                "test_end": unique_dates[test_end - 1],
                "train_obs": int(train_end - train_start),
                "test_obs": int(test_window),
                "embargo_obs": int(embargo),
            }
        )
        split_id += 1
        start += step
    return pd.DataFrame(rows)


def fit_standardizer(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Fit mean/std parameters on the training sample only."""
    params = train[feature_cols].astype(float).agg(["mean", "std"]).T.reset_index()
    params.columns = ["feature", "mean", "std"]
    params["std"] = params["std"].replace(0.0, np.nan)
    return params


def apply_standardizer(df: pd.DataFrame, params: pd.DataFrame, suffix: str = "_z") -> pd.DataFrame:
    """Apply pre-fitted standardization parameters."""
    out = df.copy()
    for row in params.itertuples(index=False):
        feature = str(row.feature)
        std = float(row.std) if pd.notna(row.std) else np.nan
        out[f"{feature}{suffix}"] = np.nan if pd.isna(std) or std == 0 else (out[feature].astype(float) - row.mean) / std
    return out


def standardize_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    suffix: str = "_z",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize train/test data without leaking test distribution."""
    params = fit_standardizer(train, feature_cols)
    return apply_standardizer(train, params, suffix=suffix), apply_standardizer(test, params, suffix=suffix), params


def rank_auc(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """AUC computed from ranks, equivalent to Mann-Whitney U."""
    y = pd.Series(y_true).astype(int)
    score = pd.Series(y_score).astype(float)
    clean = pd.DataFrame({"y": y, "score": score}).dropna()
    positives = clean["y"] == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = clean["score"].rank(method="average")
    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> pd.Series:
    """Evaluate a binary classifier used in up/down or top/bottom labels."""
    clean = pd.DataFrame({"y": pd.Series(y_true).astype(int), "score": pd.Series(y_score).astype(float)}).dropna()
    if clean.empty:
        return pd.Series(dtype="float64")
    pred = clean["score"] >= threshold
    truth = clean["y"] == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    tn = int((~pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and pd.notna(precision + recall) else np.nan
    return pd.Series(
        {
            "accuracy": (tp + tn) / clean.shape[0],
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auc": rank_auc(clean["y"], clean["score"]),
            "n": int(clean.shape[0]),
            "positive_rate": float(truth.mean()),
        }
    )
