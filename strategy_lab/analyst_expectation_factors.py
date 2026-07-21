#!/usr/bin/env python
"""Analyst expectation and analyst coverage factor helpers."""

from __future__ import annotations

import argparse
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)


def add_consensus_valuation_factors(
    df: pd.DataFrame,
    price_col: str,
    con_eps_col: str | None = None,
    con_book_value_col: str | None = None,
    con_growth_col: str | None = None,
    con_target_price_col: str | None = None,
    con_net_profit_col: str | None = None,
    con_net_asset_col: str | None = None,
    prefix: str = "con",
) -> pd.DataFrame:
    """Add common consensus valuation and target-return factors."""
    out = df.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")
    if con_eps_col:
        out[f"{prefix}_pe"] = _safe_divide(price, out[con_eps_col])
    if con_book_value_col:
        out[f"{prefix}_pb"] = _safe_divide(price, out[con_book_value_col])
    if con_growth_col and f"{prefix}_pe" in out.columns:
        out[f"{prefix}_peg"] = _safe_divide(out[f"{prefix}_pe"], out[con_growth_col])
    if con_target_price_col:
        out[f"{prefix}_rr"] = _safe_divide(out[con_target_price_col], price) - 1.0
    if con_net_profit_col and con_net_asset_col:
        out[f"{prefix}_roe"] = _safe_divide(out[con_net_profit_col], out[con_net_asset_col])
    return out


def add_relative_change_factors(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    factor_cols: Iterable[str],
    periods: int = 1,
    suffix: str = "_rel",
) -> pd.DataFrame:
    """Add month-over-month relative changes for consensus factors."""
    out = df.sort_values([asset_col, date_col]).copy()
    for col in factor_cols:
        pieces = []
        for _, group in out.groupby(asset_col, sort=False):
            value = pd.to_numeric(group[col], errors="coerce")
            rel = value / value.shift(periods).replace(0.0, np.nan) - 1.0
            pieces.append(pd.Series(rel.to_numpy(dtype=float), index=group.index))
        out[f"{col}{suffix}"] = pd.concat(pieces).sort_index()
    return out


def idiosyncratic_analyst_coverage(
    df: pd.DataFrame,
    date_col: str,
    report_count_col: str,
    control_cols: Iterable[str],
    output_col: str = "atot",
    log_transform: bool = True,
    min_count: int | None = None,
) -> pd.DataFrame:
    """Residualize analyst report count against firm characteristics by date."""
    controls = list(control_cols)
    min_count = min_count or len(controls) + 5
    out = df.copy()
    residuals = []
    for _, group in out.groupby(date_col, sort=True):
        y = pd.to_numeric(group[report_count_col], errors="coerce")
        if log_transform:
            y = np.log1p(y.clip(lower=0.0))
        x = group[controls].apply(pd.to_numeric, errors="coerce")
        clean = pd.concat([y.rename("_y"), x], axis=1).dropna()
        series = pd.Series(np.nan, index=group.index, name=output_col)
        if clean.shape[0] >= min_count:
            xmat = np.column_stack([np.ones(clean.shape[0]), clean[controls].to_numpy(dtype=float)])
            beta = np.linalg.lstsq(xmat, clean["_y"].to_numpy(dtype=float), rcond=None)[0]
            series.loc[clean.index] = clean["_y"].to_numpy(dtype=float) - xmat @ beta
        residuals.append(series)
    out[output_col] = pd.concat(residuals).sort_index()
    return out


def analyst_fscore(
    df: pd.DataFrame,
    indicators: Mapping[str, str],
    output_col: str = "fscore",
) -> pd.DataFrame:
    """Sum future fundamental indicator variables into an analyst-coverage F-score."""
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for _, col in indicators.items():
        score = score + (pd.to_numeric(out[col], errors="coerce") > 0).astype(float)
    out[output_col] = score
    return out


def lagged_factor_values(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    factor_col: str,
    lags: Iterable[int] = (1, 2, 3, 4, 5, 6),
) -> pd.DataFrame:
    """Add lagged factor values for persistence tests."""
    out = df.sort_values([asset_col, date_col]).copy()
    for lag in lags:
        out[f"{factor_col}_lag{lag}"] = out.groupby(asset_col)[factor_col].shift(lag)
    return out


def factor_coverage_by_group(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: Iterable[str],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Measure factor availability by date and optional group."""
    group_cols = [date_col] + ([group_col] if group_col else [])
    rows = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {"date": key_tuple[0], "n_total": int(group.shape[0])}
        if group_col:
            row[group_col] = key_tuple[1]
        for col in factor_cols:
            row[f"{col}_coverage"] = float(group[col].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_error_panel(
    df: pd.DataFrame,
    forecast_actual_pairs: Mapping[str, str],
    group_cols: Iterable[str] | None = None,
    extreme_threshold: float = 3.0,
) -> pd.DataFrame:
    """Compute absolute percentage forecast errors and extreme-error flags."""
    out = df.copy()
    group_cols = list(group_cols or [])
    for forecast_col, actual_col in forecast_actual_pairs.items():
        forecast = pd.to_numeric(out[forecast_col], errors="coerce")
        actual = pd.to_numeric(out[actual_col], errors="coerce")
        error = (forecast - actual).abs() / actual.abs().replace(0.0, np.nan)
        base = forecast_col.replace("forecast_", "").replace("con_", "")
        out[f"{base}_abs_pct_error"] = error
        out[f"{base}_extreme_error"] = error > extreme_threshold
    if not group_cols:
        return out
    error_cols = [col for col in out.columns if col.endswith("_abs_pct_error")]
    flag_cols = [col for col in out.columns if col.endswith("_extreme_error")]
    rows = []
    for key, group in out.groupby(group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_tuple)}
        row["n_obs"] = int(group.shape[0])
        for col in error_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        for col in flag_cols:
            row[f"{col}_rate"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def clean_extreme_forecasts(
    df: pd.DataFrame,
    forecast_actual_pairs: Mapping[str, str],
    extreme_threshold: float = 3.0,
) -> pd.DataFrame:
    """Set vendor forecasts with extreme realized errors to NaN for quality testing."""
    out = df.copy()
    for forecast_col, actual_col in forecast_actual_pairs.items():
        forecast = pd.to_numeric(out[forecast_col], errors="coerce")
        actual = pd.to_numeric(out[actual_col], errors="coerce")
        error = (forecast - actual).abs() / actual.abs().replace(0.0, np.nan)
        out.loc[error > extreme_threshold, forecast_col] = np.nan
    return out


def earnings_surprise_factor(
    df: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
    scale_col: str | None = None,
    output_col: str = "earnings_surprise",
    winsor_limit: float | None = 3.0,
) -> pd.DataFrame:
    """Compute realized earnings surprise versus the latest available forecast."""
    out = df.copy()
    actual = pd.to_numeric(out[actual_col], errors="coerce")
    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    if scale_col:
        scale = pd.to_numeric(out[scale_col], errors="coerce").abs().replace(0.0, np.nan)
    else:
        scale = forecast.abs().replace(0.0, np.nan)
    surprise = (actual - forecast) / scale
    if winsor_limit is not None:
        surprise = surprise.clip(lower=-winsor_limit, upper=winsor_limit)
    out[output_col] = surprise
    return out


def expectation_adjustment_factor(
    df: pd.DataFrame,
    asset_col: str,
    date_col: str,
    expectation_col: str,
    lookback: int = 12,
    volatility_lookback: int | None = None,
    prediction_type_col: str | None = None,
    reliable_types: Iterable[int] = (1, 2),
    fill_unreliable_zero: bool = True,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Build time-series standardized consensus adjustment factors.

    The raw adjustment is current expectation minus trailing mean expectation.
    The standardized version divides that adjustment by trailing expectation
    volatility, matching the report's stability-adjusted construction.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    volatility_lookback = volatility_lookback or lookback
    base = prefix or f"d_{expectation_col}"
    out = df.sort_values([asset_col, date_col]).copy()
    exp = pd.to_numeric(out[expectation_col], errors="coerce")
    trailing_mean = (
        exp.groupby(out[asset_col])
        .transform(lambda s: s.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).mean())
    )
    trailing_std = (
        exp.groupby(out[asset_col])
        .transform(lambda s: s.shift(1).rolling(volatility_lookback, min_periods=max(2, volatility_lookback // 3)).std())
    )
    out[f"{base}_raw"] = exp - trailing_mean
    out[f"{base}_stability"] = trailing_std
    out[f"{base}_std"] = out[f"{base}_raw"] / trailing_std.replace(0.0, np.nan)
    if prediction_type_col:
        reliable = out[prediction_type_col].isin(set(reliable_types))
        out[f"{base}_reliable"] = reliable
        if fill_unreliable_zero:
            out.loc[~reliable, f"{base}_std"] = 0.0
    return out


def select_consensus_fiscal_year_series(
    df: pd.DataFrame,
    date_col: str,
    fiscal_year_col: str,
    value_col: str,
    current_year_col: str,
    asset_col: str | None = None,
    method: str = "locked_fiscal_year",
    target_fiscal_year_col: str | None = None,
    output_col: str = "consensus_base",
) -> pd.DataFrame:
    """Select bottom consensus fiscal-year data before standardization.

    Supported methods:
    - `current_year`: nearest current fiscal-year forecast.
    - `locked_fiscal_year`: fixed target fiscal year, preferably provided by
      `target_fiscal_year_col`; otherwise it falls back to current year.
    - `smooth_next_year`: blend current and next fiscal-year forecasts as
      quarterly reports move the market's focus forward.
    """
    methods = {"current_year", "locked_fiscal_year", "smooth_next_year"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}.")
    keys = [asset_col, date_col] if asset_col else [date_col]
    needed = [date_col, fiscal_year_col, value_col, current_year_col]
    if asset_col:
        needed.append(asset_col)
    if target_fiscal_year_col:
        needed.append(target_fiscal_year_col)
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df[needed].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[fiscal_year_col] = pd.to_numeric(work[fiscal_year_col], errors="coerce")
    work[current_year_col] = pd.to_numeric(work[current_year_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if target_fiscal_year_col:
        work[target_fiscal_year_col] = pd.to_numeric(work[target_fiscal_year_col], errors="coerce")

    base_cols = [*keys, current_year_col]
    if method == "locked_fiscal_year" and target_fiscal_year_col:
        base_cols.append(target_fiscal_year_col)
    base = work[base_cols].drop_duplicates().copy()
    value_table = work[[*keys, fiscal_year_col, value_col]].drop_duplicates()

    if method == "smooth_next_year":
        current = base.copy()
        current["_target_fy"] = current[current_year_col]
        current = current.merge(
            value_table,
            left_on=[*keys, "_target_fy"],
            right_on=[*keys, fiscal_year_col],
            how="left",
        ).rename(columns={value_col: "_current_value"})
        nxt = base.copy()
        nxt["_target_fy"] = nxt[current_year_col] + 1.0
        nxt = nxt.merge(
            value_table,
            left_on=[*keys, "_target_fy"],
            right_on=[*keys, fiscal_year_col],
            how="left",
        ).rename(columns={value_col: "_next_value"})
        merged = current[[*keys, "_current_value"]].merge(nxt[[*keys, "_next_value"]], on=keys, how="left")
        month = pd.to_datetime(merged[date_col]).dt.month
        next_weight = np.select([month <= 4, month <= 8, month <= 10], [0.0, 0.25, 0.50], 0.75)
        merged[output_col] = (
            (1.0 - next_weight) * merged["_current_value"] + next_weight * merged["_next_value"]
        )
        missing_next = merged["_next_value"].isna()
        merged.loc[missing_next, output_col] = merged.loc[missing_next, "_current_value"]
        return merged[[*keys, output_col]]

    selected = base.copy()
    if method == "current_year":
        selected["_target_fy"] = selected[current_year_col]
    elif target_fiscal_year_col:
        selected["_target_fy"] = selected[target_fiscal_year_col]
    else:
        selected["_target_fy"] = selected[current_year_col]
    selected = selected.merge(
        value_table,
        left_on=[*keys, "_target_fy"],
        right_on=[*keys, fiscal_year_col],
        how="left",
    )
    selected[output_col] = selected[value_col]
    return selected[[*keys, output_col]]


def vendor_quality_summary(
    df: pd.DataFrame,
    vendor_col: str,
    date_col: str,
    coverage_col: str,
    error_cols: Iterable[str],
) -> pd.DataFrame:
    """Summarize consensus vendor coverage, forecast errors, and stability by date."""
    rows = []
    errors = list(error_cols)
    for (date, vendor), group in df.groupby([date_col, vendor_col], dropna=False, sort=True):
        row = {"date": date, "vendor": vendor, "n_obs": int(group.shape[0])}
        row["coverage"] = float(pd.to_numeric(group[coverage_col], errors="coerce").mean())
        for col in errors:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_extreme_rate"] = float((values > 3.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fama_macbeth_stepwise_selection(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    candidate_cols: Iterable[str],
    min_abs_ic: float = 0.02,
    t_threshold: float = 2.0,
    max_factors: int | None = None,
) -> pd.DataFrame:
    """Simple Fama-MacBeth forward selection using average R2 and coefficient t-stats."""
    candidates = list(candidate_cols)
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    max_factors = max_factors or len(candidates)

    def rank_ic(col: str) -> float:
        values = []
        for _, group in panel[[date_col, return_col, col]].dropna().groupby(date_col):
            if group.shape[0] >= 20:
                values.append(group[col].rank().corr(group[return_col].rank()))
        return float(np.nanmean(values)) if values else np.nan

    candidates = [col for col in candidates if abs(rank_ic(col)) >= min_abs_ic]
    while candidates and len(selected) < max_factors:
        best = None
        for col in candidates:
            cols = selected + [col]
            betas = []
            r2s = []
            for _, group in panel[[date_col, return_col, *cols]].dropna().groupby(date_col):
                if group.shape[0] < len(cols) + 5:
                    continue
                x = group[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(group[return_col], errors="coerce").to_numpy(dtype=float)
                x = np.column_stack([np.ones(group.shape[0]), x])
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                y_hat = x @ beta
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1.0 - np.sum((y - y_hat) ** 2) / ss_tot if ss_tot > EPS else np.nan
                betas.append(beta[-1])
                r2s.append(r2)
            beta_series = pd.Series(betas, dtype="float64").dropna()
            if beta_series.shape[0] < 6:
                continue
            std = beta_series.std(ddof=1)
            t_stat = beta_series.mean() / (std / np.sqrt(beta_series.shape[0])) if std > EPS else np.nan
            avg_r2 = float(pd.Series(r2s, dtype="float64").mean())
            if abs(t_stat) >= t_threshold:
                candidate = {"factor": col, "t_stat": float(t_stat), "avg_r2": avg_r2, "n_periods": int(beta_series.shape[0])}
                if best is None or candidate["avg_r2"] > best["avg_r2"]:
                    best = candidate
        if best is None:
            break
        selected.append(best["factor"])
        candidates.remove(best["factor"])
        best["step"] = len(selected)
        best["selected_factors"] = ",".join(selected)
        rows.append(best)
    return pd.DataFrame(rows)


def consensus_factor_map() -> pd.DataFrame:
    rows = [
        ("con_eps", "consensus EPS", "higher raw forecast can be good after neutralization"),
        ("con_net_profit", "consensus net profit", "strongly size-contaminated before neutralization"),
        ("con_target_return", "target price / current price - 1", "often strong but low coverage"),
        ("con_score", "consensus rating score", "watch vendor scoring direction"),
        ("con_profit_growth", "consensus net profit growth", "growth expectation"),
        ("con_pe", "price / consensus EPS", "lower is better"),
        ("con_pb", "price / consensus book value", "lower is better"),
        ("con_peg", "consensus PE divided by growth", "lower is better"),
        ("con_roe", "consensus net profit / consensus net asset", "higher can be better after neutralization"),
        ("atot", "residual log analyst report count", "higher idiosyncratic coverage is better"),
    ]
    return pd.DataFrame(rows, columns=["factor", "definition", "direction_note"])


def analyst_factor_checklist() -> pd.DataFrame:
    rows = [
        ("timestamp", "Use forecast database availability date, not report period end."),
        ("coverage", "Report factor coverage overall and by industry before testing returns."),
        ("neutralization", "Neutralize size, turnover, prior return, volatility, valuation, and industry."),
        ("direction", "Vendor rating and expectation fields need explicit sign normalization."),
        ("target_price", "Target-return factors can be strong but may have low coverage and selection bias."),
        ("industry", "Analyst factors can be industry-specific; bank and broker factors differ from industrials."),
        ("persistence", "Test lagged ATOT and forecast factors to separate information from temporary pressure."),
        ("fundamental_link", "Validate that coverage/expectation factors predict future fundamentals, not only returns."),
        ("vendor_quality", "Compare vendor coverage, extreme forecast rate, and cleaned forecast accuracy before factor use."),
        ("surprise", "Earnings-surprise factors should be built only after the report is public and held for explicit horizons."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--checklist", action="store_true")
    args = parser.parse_args()
    if args.map:
        print(consensus_factor_map())
    if args.checklist:
        print(analyst_factor_checklist())


if __name__ == "__main__":
    main()
