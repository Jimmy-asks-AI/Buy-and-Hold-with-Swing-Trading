#!/usr/bin/env python
"""Equity duration, intangible value, earnings quality, and mixed-frequency tools."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _zscore(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    std = values.std(ddof=1)
    if pd.isna(std) or std <= EPS:
        return pd.Series(0.0, index=s.index)
    return (values - values.mean()) / std


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def implied_equity_duration(
    df: pd.DataFrame,
    market_value_col: str,
    book_value_col: str,
    roe_col: str,
    growth_col: str,
    expected_return_col: str,
    horizon: int = 20,
    output_col: str = "implied_equity_duration",
) -> pd.DataFrame:
    """Estimate cash-flow-implied equity duration.

    Cash flow to equity is approximated as `(ROE - book growth) * book`.
    Terminal value is inferred as market value minus finite-horizon discounted
    cash flows, matching the report's "implied" duration idea.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    _require_columns(df, [market_value_col, book_value_col, roe_col, growth_col, expected_return_col])
    out = df.copy()
    market = pd.to_numeric(out[market_value_col], errors="coerce")
    book = pd.to_numeric(out[book_value_col], errors="coerce")
    roe = pd.to_numeric(out[roe_col], errors="coerce")
    growth = pd.to_numeric(out[growth_col], errors="coerce")
    discount = pd.to_numeric(out[expected_return_col], errors="coerce").clip(lower=0.005)

    pv_sum = pd.Series(0.0, index=out.index, dtype="float64")
    weighted_pv_sum = pd.Series(0.0, index=out.index, dtype="float64")
    current_book = book.copy()
    for step in range(1, horizon + 1):
        cash_flow = (roe - growth) * current_book
        pv = cash_flow / (1.0 + discount) ** step
        pv_sum = pv_sum + pv
        weighted_pv_sum = weighted_pv_sum + step * pv
        current_book = current_book * (1.0 + growth)

    terminal_pv = market - pv_sum
    terminal_duration = horizon + (1.0 + discount) / discount
    out[output_col] = (weighted_pv_sum + terminal_pv * terminal_duration) / market.replace(0.0, np.nan)
    return out


def bond_similarity_from_returns(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    stock_return_col: str,
    bond_return_col: str,
    market_return_col: str,
    riskfree_col: str | None = None,
    window: int = 24,
    min_periods: int | None = None,
    output_col: str = "bond_similarity",
) -> pd.DataFrame:
    """Rolling bond beta after controlling for equity market beta."""
    if window <= 2:
        raise ValueError("window must be greater than 2.")
    min_periods = min_periods or max(12, window // 2)
    cols = [asset_col, date_col, stock_return_col, bond_return_col, market_return_col]
    if riskfree_col:
        cols.append(riskfree_col)
    _require_columns(df, cols)
    out = df.sort_values([asset_col, date_col]).copy()
    out[output_col] = np.nan

    for _, group in out.groupby(asset_col, sort=False):
        work = group[[stock_return_col, bond_return_col, market_return_col, *( [riskfree_col] if riskfree_col else [] )]].apply(
            pd.to_numeric, errors="coerce"
        )
        if riskfree_col:
            y_all = work[stock_return_col] - work[riskfree_col]
            bond_all = work[bond_return_col] - work[riskfree_col]
            market_all = work[market_return_col] - work[riskfree_col]
        else:
            y_all = work[stock_return_col]
            bond_all = work[bond_return_col]
            market_all = work[market_return_col]
        scores = pd.Series(np.nan, index=group.index, dtype="float64")
        for pos in range(len(group)):
            start = max(0, pos - window + 1)
            sample = pd.DataFrame(
                {
                    "y": y_all.iloc[start : pos + 1],
                    "bond": bond_all.iloc[start : pos + 1],
                    "market": market_all.iloc[start : pos + 1],
                }
            ).dropna()
            if sample.shape[0] < min_periods:
                continue
            x = np.column_stack([np.ones(sample.shape[0]), sample[["bond", "market"]].to_numpy(dtype=float)])
            beta = np.linalg.lstsq(x, sample["y"].to_numpy(dtype=float), rcond=None)[0]
            scores.iloc[pos] = beta[1]
        out.loc[group.index, output_col] = scores
    return out


def duration_rate_regime_weight(
    rates: pd.DataFrame,
    date_col: str,
    rate_col: str,
    lookback: int = 12,
    output_col: str = "duration_factor_weight",
) -> pd.DataFrame:
    """Map rate changes to a short-duration factor weight."""
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    _require_columns(rates, [date_col, rate_col])
    out = rates.sort_values(date_col).copy()
    rate = pd.to_numeric(out[rate_col], errors="coerce")
    delta = rate.diff()
    rolling_mean = delta.rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    rolling_std = delta.rolling(lookback, min_periods=max(3, lookback // 2)).std(ddof=1).replace(0.0, np.nan)
    z = (rolling_mean / rolling_std).clip(-2.0, 2.0)
    out[output_col] = (1.0 + 0.25 * z).clip(0.5, 1.5)
    out[f"{output_col}_rate_delta_z"] = z
    return out


def rolling_roe_prediction_with_guidance(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    realized_roe_col: str,
    predictor_cols: Sequence[str],
    lookback_dates: int = 20,
    min_observations: int | None = None,
    output_col: str = "predicted_current_roe",
) -> pd.DataFrame:
    """Predict current true ROE from latest disclosure, guidance, and consensus."""
    predictors = list(predictor_cols)
    _require_columns(df, [asset_col, date_col, realized_roe_col, *predictors])
    min_observations = min_observations or max(20, len(predictors) * 5)
    out = df.sort_values([date_col, asset_col]).copy()
    out[output_col] = out[predictors].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    dates = pd.Index(sorted(out[date_col].dropna().unique()))
    for pos, date in enumerate(dates):
        hist_dates = dates[max(0, pos - lookback_dates) : pos]
        hist = out[out[date_col].isin(hist_dates)][[realized_roe_col, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
        current = out[out[date_col] == date][predictors].apply(pd.to_numeric, errors="coerce")
        if hist.shape[0] < min_observations or current.empty:
            continue
        x = np.column_stack([np.ones(hist.shape[0]), hist[predictors].to_numpy(dtype=float)])
        y = hist[realized_roe_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        valid = current.dropna()
        if valid.empty:
            continue
        pred = np.column_stack([np.ones(valid.shape[0]), valid[predictors].to_numpy(dtype=float)]) @ beta
        out.loc[valid.index, output_col] = pred
    return out


def industry_effectiveness_weights(
    df: pd.DataFrame,
    date_col: str,
    industry_col: str,
    factor_col: str,
    forward_return_col: str,
    lookback_dates: int = 20,
    improvement_threshold: float = 0.0005,
    weak_weight: float = 0.5,
    output_col: str = "industry_effectiveness_weight",
) -> pd.DataFrame:
    """Down-weight industries where removing the industry improves trailing IC."""
    _require_columns(df, [date_col, industry_col, factor_col, forward_return_col])
    out = df.sort_values(date_col).copy()
    out[output_col] = 1.0
    dates = pd.Index(sorted(out[date_col].dropna().unique()))
    for pos, date in enumerate(dates):
        hist_dates = dates[max(0, pos - lookback_dates) : pos]
        hist = out[out[date_col].isin(hist_dates)]
        if len(hist_dates) < max(4, lookback_dates // 4):
            continue
        daily_ics = []
        for _, group in hist[[date_col, factor_col, forward_return_col]].dropna().groupby(date_col):
            if group.shape[0] >= 10:
                daily_ics.append(group[factor_col].rank().corr(group[forward_return_col].rank()))
        baseline = float(np.nanmean(daily_ics)) if daily_ics else np.nan
        if pd.isna(baseline):
            continue
        penalties: dict[object, float] = {}
        for industry in hist[industry_col].dropna().unique():
            reduced = hist[hist[industry_col] != industry]
            reduced_ics = []
            for _, group in reduced[[date_col, factor_col, forward_return_col]].dropna().groupby(date_col):
                if group.shape[0] >= 10:
                    reduced_ics.append(group[factor_col].rank().corr(group[forward_return_col].rank()))
            reduced_ic = float(np.nanmean(reduced_ics)) if reduced_ics else np.nan
            if pd.notna(reduced_ic) and reduced_ic - baseline > improvement_threshold:
                penalties[industry] = weak_weight
        idx = out.index[out[date_col] == date]
        out.loc[idx, output_col] = out.loc[idx, industry_col].map(penalties).fillna(1.0)
    return out


def sue_from_net_profit(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    net_profit_col: str,
    seasonal_lag: int = 4,
    volatility_window: int = 4,
    output_col: str = "sue",
) -> pd.DataFrame:
    """Standardized unexpected earnings from year-over-year profit changes."""
    if seasonal_lag <= 0 or volatility_window <= 1:
        raise ValueError("seasonal_lag must be positive and volatility_window must exceed 1.")
    _require_columns(df, [asset_col, date_col, net_profit_col])
    out = df.sort_values([asset_col, date_col]).copy()
    profit = pd.to_numeric(out[net_profit_col], errors="coerce")
    yoy_change = profit - profit.groupby(out[asset_col]).shift(seasonal_lag)
    scale = yoy_change.groupby(out[asset_col]).transform(lambda s: s.rolling(volatility_window, min_periods=2).std(ddof=1))
    out[output_col] = yoy_change / scale.replace(0.0, np.nan)
    out[f"{output_col}_yoy_change"] = yoy_change
    return out


def intangible_adjusted_book_value(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    book_value_col: str,
    goodwill_col: str,
    rd_expense_col: str,
    sga_expense_col: str,
    market_value_col: str | None = None,
    rd_decay: float = 0.30,
    organization_decay: float = 0.20,
    organization_share: float = 0.30,
    initial_growth: float = 0.20,
    output_prefix: str = "intangible",
) -> pd.DataFrame:
    """Capitalize R&D and organization capital, then adjust book value."""
    _require_columns(df, [asset_col, date_col, book_value_col, goodwill_col, rd_expense_col, sga_expense_col])
    if market_value_col:
        _require_columns(df, [market_value_col])
    out = df.sort_values([asset_col, date_col]).copy()
    kc_col = f"{output_prefix}_knowledge_capital"
    oc_col = f"{output_prefix}_organization_capital"
    out[kc_col] = np.nan
    out[oc_col] = np.nan
    for _, group in out.groupby(asset_col, sort=False):
        rd = pd.to_numeric(group[rd_expense_col], errors="coerce").fillna(0.0)
        org_invest = pd.to_numeric(group[sga_expense_col], errors="coerce").fillna(0.0) * organization_share
        kc_values = []
        oc_values = []
        kc_prev = np.nan
        oc_prev = np.nan
        for rd_value, org_value in zip(rd, org_invest, strict=False):
            if np.isnan(kc_prev):
                kc_prev = rd_value / (initial_growth + rd_decay) if initial_growth + rd_decay > EPS else rd_value
            else:
                kc_prev = (1.0 - rd_decay) * kc_prev + rd_value
            if np.isnan(oc_prev):
                oc_prev = org_value / (initial_growth + organization_decay) if initial_growth + organization_decay > EPS else org_value
            else:
                oc_prev = (1.0 - organization_decay) * oc_prev + org_value
            kc_values.append(kc_prev)
            oc_values.append(oc_prev)
        out.loc[group.index, kc_col] = kc_values
        out.loc[group.index, oc_col] = oc_values

    book = pd.to_numeric(out[book_value_col], errors="coerce")
    goodwill = pd.to_numeric(out[goodwill_col], errors="coerce").fillna(0.0)
    intangible = out[kc_col] + out[oc_col]
    adjusted_book = book - goodwill + intangible
    out[f"{output_prefix}_capital"] = intangible
    out[f"{output_prefix}_adjusted_book"] = adjusted_book
    out[f"{output_prefix}_ratio"] = intangible / adjusted_book.replace(0.0, np.nan)
    if market_value_col:
        out[f"pb_{output_prefix}"] = pd.to_numeric(out[market_value_col], errors="coerce") / adjusted_book.replace(0.0, np.nan)
    return out


def pb_int_roe_value_score(
    df: pd.DataFrame,
    date_col: str,
    pb_int_col: str,
    roe_col: str,
    output_col: str = "pb_int_roe_value_score",
) -> pd.DataFrame:
    """Composite value score: lower intangible-adjusted PB and higher ROE."""
    _require_columns(df, [date_col, pb_int_col, roe_col])
    out = df.copy()
    value = -pd.to_numeric(out[pb_int_col], errors="coerce")
    quality = pd.to_numeric(out[roe_col], errors="coerce")
    out[output_col] = value.groupby(out[date_col]).transform(_zscore) + quality.groupby(out[date_col]).transform(_zscore)
    return out


def make_mixed_frequency_sequence_dataset(
    daily_panel: pd.DataFrame,
    intraday_panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    intraday_order_col: str,
    daily_feature_cols: Sequence[str],
    intraday_feature_cols: Sequence[str],
    target_col: str,
    daily_lookback: int = 60,
    intraday_lookback_days: int = 60,
) -> dict[str, object]:
    """Create aligned daily and intraday tensors for mixed-frequency models."""
    daily_features = list(daily_feature_cols)
    intraday_features = list(intraday_feature_cols)
    _require_columns(daily_panel, [date_col, asset_col, target_col, *daily_features])
    _require_columns(intraday_panel, [date_col, asset_col, intraday_order_col, *intraday_features])
    daily = daily_panel.sort_values([asset_col, date_col]).copy()
    intraday = intraday_panel.sort_values([asset_col, date_col, intraday_order_col]).copy()
    intraday_groups = {(asset, date): group for (asset, date), group in intraday.groupby([asset_col, date_col], sort=False)}

    daily_x: list[np.ndarray] = []
    intraday_x: list[np.ndarray] = []
    y: list[float] = []
    index_rows: list[dict[str, object]] = []
    for asset, group in daily.groupby(asset_col, sort=False):
        group = group.sort_values(date_col)
        dates = list(group[date_col])
        daily_values = group[daily_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        targets = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)
        for pos in range(daily_lookback, len(group)):
            daily_window = daily_values[pos - daily_lookback : pos]
            hf_dates = dates[max(0, pos - intraday_lookback_days) : pos]
            hf_parts = []
            for d in hf_dates:
                part = intraday_groups.get((asset, d))
                if part is not None:
                    hf_parts.append(part[intraday_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
            if not hf_parts:
                continue
            hf_window = np.vstack(hf_parts)
            if np.isnan(daily_window).any() or np.isnan(hf_window).any() or np.isnan(targets[pos]):
                continue
            daily_x.append(daily_window)
            intraday_x.append(hf_window)
            y.append(float(targets[pos]))
            index_rows.append({"asset": asset, "date": dates[pos]})
    return {
        "daily_x": np.asarray(daily_x, dtype=object),
        "intraday_x": np.asarray(intraday_x, dtype=object),
        "y": np.asarray(y, dtype=float),
        "index": pd.DataFrame(index_rows),
        "daily_features": daily_features,
        "intraday_features": intraday_features,
    }


def orthogonal_factor_head(
    df: pd.DataFrame,
    date_col: str,
    prediction_cols: Sequence[str],
    control_cols: Sequence[str] | None = None,
    output_prefix: str = "orth_factor",
) -> pd.DataFrame:
    """Generate mutually orthogonal cross-sectional factor heads by date."""
    preds = list(prediction_cols)
    controls = list(control_cols or [])
    _require_columns(df, [date_col, *preds, *controls])
    out = df.copy()
    for idx in range(len(preds)):
        out[f"{output_prefix}_{idx}"] = np.nan
    for _, group in out.groupby(date_col, sort=True):
        work = group[preds + controls].apply(pd.to_numeric, errors="coerce").dropna()
        if work.shape[0] < max(5, len(preds) + len(controls) + 2):
            continue
        residuals = []
        control_matrix = None
        if controls:
            control_matrix = np.column_stack([np.ones(work.shape[0]), work[controls].to_numpy(dtype=float)])
        for col in preds:
            y = work[col].to_numpy(dtype=float)
            if control_matrix is not None:
                beta = np.linalg.lstsq(control_matrix, y, rcond=None)[0]
                y = y - control_matrix @ beta
            for previous in residuals:
                denom = float(previous @ previous)
                if denom > EPS:
                    y = y - previous * float(previous @ y) / denom
            y = y - np.nanmean(y)
            std = np.nanstd(y, ddof=1)
            y = y / std if std > EPS else np.zeros_like(y)
            residuals.append(y)
        for idx, values in enumerate(residuals):
            out.loc[work.index, f"{output_prefix}_{idx}"] = values
    return out


def round48_research_checklist() -> pd.DataFrame:
    """Reusable validation checklist for this research batch."""
    rows = [
        ("equity_duration", "Check PB/PE/value correlations and rate-up/rate-down regime dependence."),
        ("bond_similarity", "Treat as statistical similarity, not cash-flow duration; control market beta."),
        ("roe_sue", "Use only available guidance/consensus snapshots and leave-one-industry IC weights."),
        ("intangible_pb", "Capitalize R&D and organization capital, remove goodwill, test parameter stability."),
        ("mixed_frequency_dl", "Evaluate turnover/cost, path dependence, universe dependence, and factor novelty."),
        ("orthogonal_heads", "Orthogonality can weaken alpha; report IC before and after controls."),
    ]
    return pd.DataFrame(rows, columns=["topic", "validation_gate"])
