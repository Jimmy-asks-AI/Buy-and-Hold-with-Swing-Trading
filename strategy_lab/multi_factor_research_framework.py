#!/usr/bin/env python
"""Research framework for hundreds-factor quantitative models.

This module connects ideas learned from local reports and public quant projects:
factor registry, cross-sectional cleaning, neutralization, factor diagnostics,
redundancy control, family-level aggregation, walk-forward weighting, portfolio
construction, and performance review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_corr(x: pd.Series, y: pd.Series, rank: bool = False) -> float:
    """Correlation that returns NaN for constant or insufficient samples."""
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if clean.shape[0] < 3:
        return float("nan")
    x_values = clean["x"].rank() if rank else clean["x"]
    y_values = clean["y"].rank() if rank else clean["y"]
    if x_values.std(ddof=1) <= EPS or y_values.std(ddof=1) <= EPS:
        return float("nan")
    return float(x_values.corr(y_values))


@dataclass(frozen=True)
class FactorSpec:
    """Metadata for one factor in a large factor library."""

    factor_id: str
    column: str
    family: str
    direction: float = 1.0
    horizon: int = 20
    data_type: str = "unknown"
    availability_col: str | None = None
    cost_tier: str = "medium"
    description: str = ""


@dataclass(frozen=True)
class ValidationConfig:
    min_assets: int = 30
    groups: int = 5
    corr_threshold: float = 0.85
    min_ic_periods: int = 12
    min_abs_rank_ic: float = 0.01
    min_icir: float = 0.10
    top_quantile: float = 0.20
    max_weight: float = 0.05


def registry_from_records(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Build a factor registry table from dictionaries or FactorSpec-like records."""
    rows = []
    required = {"factor_id", "column", "family"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"factor record missing fields: {sorted(missing)}")
        spec = FactorSpec(**{**asdict(FactorSpec("", "", "")), **dict(record)})
        rows.append(asdict(spec))
    table = pd.DataFrame(rows)
    if table["factor_id"].duplicated().any():
        dupes = table.loc[table["factor_id"].duplicated(), "factor_id"].tolist()
        raise ValueError(f"duplicated factor_id values: {dupes}")
    return table


def apply_factor_directions(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    suffix: str = "_alpha",
) -> tuple[pd.DataFrame, list[str]]:
    """Convert raw factors into alpha-score columns where larger is better."""
    out = panel.copy()
    alpha_cols: list[str] = []
    for _, row in registry.iterrows():
        col = str(row["column"])
        if col not in out.columns:
            continue
        factor_id = str(row["factor_id"])
        direction = float(row.get("direction", 1.0))
        alpha_col = f"{factor_id}{suffix}"
        out[alpha_col] = pd.to_numeric(out[col], errors="coerce") * direction
        alpha_cols.append(alpha_col)
    return out, alpha_cols


def check_availability_dates(
    panel: pd.DataFrame,
    date_col: str,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Report potential point-in-time violations from factor availability dates."""
    rows = []
    data = panel.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    for _, row in registry.dropna(subset=["availability_col"]).iterrows():
        availability_col = row["availability_col"]
        if not availability_col or availability_col not in data.columns:
            continue
        available = pd.to_datetime(data[availability_col], errors="coerce")
        valid = available.notna()
        violations = valid & (available > data[date_col])
        rows.append(
            {
                "factor_id": row["factor_id"],
                "availability_col": availability_col,
                "checked_rows": int(valid.sum()),
                "violations": int(violations.sum()),
                "violation_rate": float(violations.mean()) if len(data) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def winsorize_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    suffix: str = "_win",
) -> tuple[pd.DataFrame, list[str]]:
    """Winsorize factor columns within each date."""
    out = df.copy()
    created = []
    for col in factor_cols:
        target = f"{col}{suffix}"

        def clip_one(s: pd.Series) -> pd.Series:
            values = pd.to_numeric(s, errors="coerce")
            lo = values.quantile(lower_q)
            hi = values.quantile(upper_q)
            return values.clip(lo, hi)

        out[target] = out.groupby(date_col)[col].transform(clip_one)
        created.append(target)
    return out, created


def zscore_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    suffix: str = "_z",
) -> tuple[pd.DataFrame, list[str]]:
    """Cross-sectionally z-score factor columns by date."""
    out = df.copy()
    created = []
    for col in factor_cols:
        target = f"{col}{suffix}"

        def zscore(s: pd.Series) -> pd.Series:
            values = pd.to_numeric(s, errors="coerce")
            std = values.std(ddof=1)
            if pd.isna(std) or abs(std) <= EPS:
                return values * 0.0
            return (values - values.mean()) / std

        out[target] = out.groupby(date_col)[col].transform(zscore)
        created.append(target)
    return out, created


def _design_matrix(
    group: pd.DataFrame,
    control_cols: Sequence[str],
    industry_col: str | None = None,
) -> pd.DataFrame:
    pieces = [pd.Series(1.0, index=group.index, name="intercept")]
    if control_cols:
        controls = group[list(control_cols)].apply(pd.to_numeric, errors="coerce")
        pieces.append(controls)
    if industry_col and industry_col in group.columns:
        dummies = pd.get_dummies(group[industry_col].astype(str), prefix=industry_col, drop_first=True)
        pieces.append(dummies.astype(float))
    return pd.concat(pieces, axis=1)


def neutralize_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    control_cols: Sequence[str] = (),
    industry_col: str | None = None,
    weight_col: str | None = None,
    suffix: str = "_neu",
    min_obs: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Cross-sectional residualization against controls and optional industry dummies."""
    out = df.copy()
    factors = list(factor_cols)
    created = [f"{col}{suffix}" for col in factors]
    for target in created:
        out[target] = np.nan
    for _, group in out.groupby(date_col, sort=True):
        x_all = _design_matrix(group, control_cols, industry_col=industry_col)
        for col, target in zip(factors, created, strict=True):
            work = pd.concat([group[col].rename("_y"), x_all], axis=1).dropna()
            min_required = min_obs or max(10, x_all.shape[1] + 5)
            if work.shape[0] < min_required:
                continue
            y = work["_y"].to_numpy(dtype=float)
            x = work.drop(columns=["_y"]).to_numpy(dtype=float)
            if weight_col and weight_col in group.columns:
                w = pd.to_numeric(group.loc[work.index, weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                root_w = np.sqrt(np.clip(w, EPS, None))
                beta = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
            else:
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
            out.loc[work.index, target] = y - x @ beta
    return out, created


def factor_ic_long(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    forward_return_col: str,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Compute Pearson IC and Rank IC for many factors by date."""
    rows = []
    for factor in factor_cols:
        for date, group in df[[date_col, factor, forward_return_col]].dropna().groupby(date_col):
            if group.shape[0] < min_assets:
                continue
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "n_assets": int(group.shape[0]),
                    "ic": _safe_corr(group[factor], group[forward_return_col]),
                    "rank_ic": _safe_corr(group[factor], group[forward_return_col], rank=True),
                }
            )
    return pd.DataFrame(rows)


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC stability by factor."""
    rows = []
    for factor, group in ic_df.groupby("factor"):
        rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
        std = rank_ic.std(ddof=1)
        rows.append(
            {
                "factor": factor,
                "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_std": float(std) if not pd.isna(std) else np.nan,
                "rank_icir": float(rank_ic.mean() / std) if std and not pd.isna(std) else np.nan,
                "rank_ic_win_rate": float((rank_ic > 0).mean()) if not rank_ic.empty else np.nan,
                "ic_mean": float(ic.mean()) if not ic.empty else np.nan,
                "periods": int(rank_ic.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def quantile_spread_long(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    forward_return_col: str,
    groups: int = 5,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Compute factor quantile average returns and top-minus-bottom spread."""
    rows = []
    for factor in factor_cols:
        clean = df[[date_col, factor, forward_return_col]].dropna()
        for date, group in clean.groupby(date_col):
            if group.shape[0] < max(min_assets, groups):
                continue
            try:
                bucket = pd.qcut(group[factor].rank(method="first"), groups, labels=False) + 1
            except ValueError:
                continue
            tmp = group.assign(bucket=bucket)
            means = tmp.groupby("bucket")[forward_return_col].mean()
            for bucket_id, value in means.items():
                rows.append({"date": date, "factor": factor, "bucket": int(bucket_id), "return": float(value)})
            if 1 in means.index and groups in means.index:
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "bucket": "top_minus_bottom",
                        "return": float(means.loc[groups] - means.loc[1]),
                    }
                )
    return pd.DataFrame(rows)


def top_quantile_turnover(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    factor_cols: Iterable[str],
    top_quantile: float = 0.20,
    min_assets: int = 30,
) -> pd.DataFrame:
    """Estimate selected-set turnover for each factor's top quantile."""
    rows = []
    data = df.sort_values([date_col, asset_col]).copy()
    for factor in factor_cols:
        previous: set[object] | None = None
        for date, group in data[[date_col, asset_col, factor]].dropna().groupby(date_col):
            if group.shape[0] < min_assets:
                continue
            top_n = max(1, int(np.ceil(group.shape[0] * top_quantile)))
            selected = set(group.nlargest(top_n, factor)[asset_col])
            if previous is not None:
                overlap = len(selected & previous) / max(len(previous), 1)
                rows.append({"date": date, "factor": factor, "turnover": float(1.0 - overlap), "selected": len(selected)})
            previous = selected
    return pd.DataFrame(rows)


def average_factor_correlation(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    min_assets: int = 30,
) -> pd.DataFrame:
    """Average cross-sectional factor correlation matrix across dates."""
    factors = list(factor_cols)
    matrices = []
    for _, group in df[[date_col, *factors]].dropna(how="all").groupby(date_col):
        clean = group[factors].apply(pd.to_numeric, errors="coerce")
        if clean.dropna(how="all").shape[0] >= min_assets:
            usable = clean.loc[:, clean.std(skipna=True) > EPS]
            if not usable.empty:
                matrices.append(usable.corr().reindex(index=factors, columns=factors))
    if not matrices:
        return pd.DataFrame(np.eye(len(factors)), index=factors, columns=factors)
    total = sum(m.reindex(index=factors, columns=factors).fillna(0.0) for m in matrices)
    corr = total / len(matrices)
    values = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=factors, columns=factors)


def correlation_clusters(corr: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Cluster factors by absolute correlation using connected components."""
    factors = list(corr.index)
    seen: set[str] = set()
    rows = []
    cluster_id = 0
    for factor in factors:
        if factor in seen:
            continue
        stack = [factor]
        component = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            neighbors = corr.columns[corr.loc[current].abs() >= threshold].tolist()
            stack.extend([n for n in neighbors if n not in seen])
        for member in component:
            rows.append({"factor": member, "cluster": cluster_id, "cluster_size": len(component)})
        cluster_id += 1
    return pd.DataFrame(rows)


def factor_quality_table(
    ic_stats: pd.DataFrame,
    turnover_df: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a practical quality score for large-library factor selection."""
    out = ic_stats.copy()
    if turnover_df is not None and not turnover_df.empty:
        turnover = turnover_df.groupby("factor")["turnover"].mean().rename("avg_turnover")
        out = out.merge(turnover, on="factor", how="left")
    else:
        out["avg_turnover"] = np.nan
    out["avg_turnover"] = out["avg_turnover"].fillna(out["avg_turnover"].median()).fillna(0.5)
    out["quality_score"] = (
        out["rank_ic_mean"].fillna(0.0)
        * np.sqrt(out["periods"].clip(lower=1))
        * (out["rank_ic_win_rate"].fillna(0.5) + 0.5)
        / (1.0 + out["avg_turnover"].clip(lower=0.0))
    )
    if registry is not None and not registry.empty:
        mapping = registry.set_index("factor_id")["family"].to_dict()
        out["factor_id"] = out["factor"].str.replace("_alpha_win_z_neu", "", regex=False)
        out["family"] = out["factor_id"].map(mapping).fillna("unknown")
    return out.sort_values("quality_score", ascending=False)


def select_factors(
    quality: pd.DataFrame,
    clusters: pd.DataFrame,
    config: ValidationConfig = ValidationConfig(),
    max_per_family: int = 30,
) -> pd.DataFrame:
    """Select non-redundant factors that pass stability thresholds."""
    q = quality.copy()
    q = q.merge(clusters, on="factor", how="left")
    q = q[
        (q["periods"] >= config.min_ic_periods)
        & (q["rank_ic_mean"].abs() >= config.min_abs_rank_ic)
        & (q["rank_icir"].abs() >= config.min_icir)
    ].copy()
    if q.empty:
        return q
    q = q.sort_values("quality_score", ascending=False)
    q = q.groupby("cluster", as_index=False, group_keys=False).head(1)
    if "family" in q.columns:
        q = q.groupby("family", as_index=False, group_keys=False).head(max_per_family)
    return q.sort_values("quality_score", ascending=False)


def combine_by_family(
    df: pd.DataFrame,
    date_col: str,
    selected: pd.DataFrame,
    output_prefix: str = "family_score",
) -> tuple[pd.DataFrame, list[str]]:
    """Combine selected factors into family-level scores."""
    if selected.empty:
        raise ValueError("selected factor table is empty.")
    out = df.copy()
    family_cols = []
    for family, group in selected.groupby("family"):
        cols = [c for c in group["factor"].tolist() if c in out.columns]
        if not cols:
            continue
        target = f"{output_prefix}_{family}"
        weights = group.set_index("factor")["quality_score"].reindex(cols).fillna(0.0)
        if weights.abs().sum() <= EPS:
            weights = pd.Series(1.0, index=cols)
        weights = weights / weights.abs().sum()
        out[target] = out[cols].to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
        out, z_cols = zscore_by_date(out, date_col, [target], suffix="_z")
        family_cols.extend(z_cols)
    return out, family_cols


def combine_families(
    df: pd.DataFrame,
    date_col: str,
    family_score_cols: Iterable[str],
    family_weights: Mapping[str, float] | None = None,
    output_col: str = "multi_factor_score",
) -> pd.DataFrame:
    """Combine family-level scores into one model score."""
    out = df.copy()
    cols = list(family_score_cols)
    if not cols:
        raise ValueError("family_score_cols is empty.")
    if family_weights is None:
        weights = pd.Series(1.0 / len(cols), index=cols)
    else:
        weights = pd.Series({col: float(family_weights.get(col, 0.0)) for col in cols})
        if weights.abs().sum() <= EPS:
            weights = pd.Series(1.0 / len(cols), index=cols)
        else:
            weights = weights / weights.abs().sum()
    out[output_col] = out[cols].to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    out, _ = zscore_by_date(out, date_col, [output_col], suffix="_z")
    return out


def walk_forward_splits(
    dates: Iterable[object],
    train_periods: int,
    test_periods: int,
    step: int | None = None,
) -> pd.DataFrame:
    """Create chronological train/test windows."""
    unique_dates = pd.Index(sorted(pd.Series(list(dates)).dropna().unique()))
    step = step or test_periods
    rows = []
    end_train = train_periods
    split = 0
    while end_train + test_periods <= len(unique_dates):
        train = unique_dates[end_train - train_periods : end_train]
        test = unique_dates[end_train : end_train + test_periods]
        rows.append(
            {
                "split": split,
                "train_start": train[0],
                "train_end": train[-1],
                "test_start": test[0],
                "test_end": test[-1],
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        split += 1
        end_train += step
    return pd.DataFrame(rows)


def portfolio_weights_from_scores(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    score_col: str,
    long_only: bool = True,
    top_quantile: float = 0.20,
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Convert model scores to simple research portfolio weights."""
    rows = []
    for date, group in df[[date_col, asset_col, score_col]].dropna().groupby(date_col):
        if group.empty:
            continue
        n = max(1, int(np.ceil(group.shape[0] * top_quantile)))
        if long_only:
            selected = group.nlargest(n, score_col).copy()
            raw = selected[score_col].clip(lower=0.0)
            if raw.sum() <= EPS:
                raw = pd.Series(1.0, index=selected.index)
            weights = raw / raw.sum()
        else:
            long_leg = group.nlargest(n, score_col).copy()
            short_leg = group.nsmallest(n, score_col).copy()
            weights = pd.concat(
                [
                    pd.Series(0.5 / len(long_leg), index=long_leg.index),
                    pd.Series(-0.5 / len(short_leg), index=short_leg.index),
                ]
            )
            selected = pd.concat([long_leg, short_leg])
        weights = weights.clip(lower=-max_weight, upper=max_weight)
        denom = weights.sum() if long_only else weights.abs().sum()
        weights = weights / denom if abs(denom) > EPS else weights
        for idx, weight in weights.items():
            rows.append({"date": date, "asset": selected.loc[idx, asset_col], "weight": float(weight)})
    return pd.DataFrame(rows)


def portfolio_forward_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
) -> pd.Series:
    """Compute portfolio forward returns from date-asset weights."""
    merged = weights.merge(
        returns[[date_col, asset_col, return_col]],
        left_on=["date", "asset"],
        right_on=[date_col, asset_col],
        how="left",
    )
    merged["_weighted_return"] = merged["weight"] * pd.to_numeric(merged[return_col], errors="coerce")
    return merged.groupby("date")["_weighted_return"].sum().sort_index()


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Return common performance metrics for a return series."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    downside = r[r < 0]
    ann_return = equity.iloc[-1] ** (periods_per_year / max(len(r), 1)) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    return pd.Series(
        {
            "annual_return": float(ann_return),
            "annual_volatility": float(ann_vol) if not pd.isna(ann_vol) else np.nan,
            "sharpe": float(ann_return / ann_vol) if ann_vol and not pd.isna(ann_vol) else np.nan,
            "sortino": float(ann_return / downside_vol) if downside_vol and not pd.isna(downside_vol) else np.nan,
            "max_drawdown": float(drawdown.min()),
            "calmar": float(ann_return / abs(drawdown.min())) if drawdown.min() < 0 else np.nan,
            "win_rate": float((r > 0).mean()),
            "periods": int(len(r)),
        }
    )


def github_quant_project_lessons() -> pd.DataFrame:
    """Condensed lessons from public AI-Quant and Quant GitHub projects."""
    rows = [
        ("microsoft/qlib", "full_pipeline", "Keep data, model training, backtest, risk model, portfolio optimization, and execution connected but loosely coupled."),
        ("quantopian/alphalens", "factor_diagnostics", "Every factor needs IC, return spread, turnover, and grouped analysis before entering a model."),
        ("polakowo/vectorbt", "scale", "Vectorized arrays and broadcasting are useful for large parameter sweeps, but final validation still needs realistic execution assumptions."),
        ("QuantConnect/Lean", "event_driven_engine", "Production-grade backtesting benefits from event-driven execution, modular plug-ins, alternative data, and live-trading consistency."),
        ("mementum/backtrader", "execution_details", "Orders, slippage, commission, data feeds, and analyzers must be explicit research objects."),
        ("AI4Finance/FinRL", "rl_environment", "RL research needs market environments, agents, applications, and baselines rather than isolated reward optimization."),
        ("AI4Finance/FinRL-Trading", "weight_interface", "A target-weight vector is a clean contract between strategy logic and downstream execution."),
        ("AI4Finance/FinGPT", "llm_data_pipeline", "LLM finance signals require timely data curation, task benchmarks, and retrieval/finetuning controls."),
        ("AI4Finance/FinRobot", "agent_workflow", "LLM agents are useful for report generation and tool orchestration, but outputs must be grounded in structured data."),
        ("microsoft/RD-Agent", "auto_rd", "AI quant factories need hypothesis, implementation, execution feedback, and correction loops for factor-model co-optimization."),
        ("ranaroussi/quantstats", "performance_report", "Performance reporting should separate stats, plots, reports, tail risk, and Monte Carlo diagnostics."),
        ("robcarver17/pysystemtrade", "system_design", "Systematic trading infrastructure should encode position sizing, diversification, risk targeting, and production feedback."),
    ]
    return pd.DataFrame(rows, columns=["project", "module", "lesson"])


def hundred_factor_model_checklist() -> pd.DataFrame:
    """Quality gates before claiming a hundreds-factor model is investable."""
    rows = [
        ("registry", "Every factor has id, column, family, direction, horizon, data source, and availability rule."),
        ("point_in_time", "Financial, analyst, text, and alternative data must use announcement or ingestion dates."),
        ("cleaning", "Winsorization, standardization, missing coverage, and universe filters are logged per date."),
        ("neutralization", "Report raw, industry/size-neutral, and known-factor-neutral versions separately."),
        ("diagnostics", "IC, RankIC, quantile spread, turnover, coverage, monotonicity, and failure years are required."),
        ("redundancy", "Highly correlated factors are clustered; only one or a small weighted composite survives each cluster."),
        ("family_layer", "Combine factors inside economic families before mixing unrelated signals."),
        ("model_layer", "Use walk-forward training; current-period returns must never set current-period weights."),
        ("portfolio", "Weights obey liquidity, max weight, industry, turnover, and cost constraints."),
        ("validation", "Evaluate sample-out, regime splits, cost sensitivity, capacity, PBO, and live paper-trading drift."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
