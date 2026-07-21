#!/usr/bin/env python
"""Fund classification, pool construction and label-scoring tools."""

from __future__ import annotations

import numpy as np
import pandas as pd


EQUITY_MANAGEMENT_TYPES = ["passive_index", "enhanced_index", "active"]
BENCHMARK_SCOPES = ["broad_base", "style_strategy", "sector_theme"]


def equity_fund_3x3_category(management_type: str, benchmark_scope: str) -> str:
    """Classify equity funds by management type and benchmark/stock-picking scope."""
    if management_type not in EQUITY_MANAGEMENT_TYPES:
        raise ValueError(f"management_type must be one of {EQUITY_MANAGEMENT_TYPES}.")
    if benchmark_scope not in BENCHMARK_SCOPES:
        raise ValueError(f"benchmark_scope must be one of {BENCHMARK_SCOPES}.")
    return f"{management_type}__{benchmark_scope}"


def active_equity_base_pool(
    fund_quarterly: pd.DataFrame,
    fund_col: str = "fund_id",
    report_date_col: str = "report_date",
    size_col: str = "fund_size_cny",
    manager_tenure_col: str = "manager_tenure_years",
    stock_weight_col: str = "stock_weight",
    theme_tag_col: str | None = "theme_tag",
    min_size_cny: float = 1e8,
    min_manager_tenure_years: float = 1.0,
    avg_stock_weight_min: float = 0.60,
    current_stock_weight_min: float = 0.50,
    lookback_quarters: int = 4,
    all_market_theme_values: set[str] | None = None,
) -> pd.DataFrame:
    """Build an active-equity broad-base fund pool.

    Defaults follow the Zhejiang report: size > 100m CNY, current first manager
    tenure > 1y, recent average stock weight > 60%, minimum stock weight > 50%,
    and all-market stock-picking theme.
    """
    required = [fund_col, report_date_col, size_col, manager_tenure_col, stock_weight_col]
    if theme_tag_col:
        required.append(theme_tag_col)
    missing = [col for col in required if col not in fund_quarterly.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    all_market_theme_values = all_market_theme_values or {"all_market", "broad_base", "全市场选股"}
    data = fund_quarterly.sort_values([fund_col, report_date_col]).copy()
    recent = data.groupby(fund_col).tail(lookback_quarters)
    agg = recent.groupby(fund_col).agg(
        latest_report_date=(report_date_col, "max"),
        latest_size=(size_col, "last"),
        latest_manager_tenure=(manager_tenure_col, "last"),
        avg_stock_weight=(stock_weight_col, "mean"),
        min_stock_weight=(stock_weight_col, "min"),
    )
    if theme_tag_col:
        agg["latest_theme_tag"] = recent.groupby(fund_col)[theme_tag_col].last()
    mask = (
        (agg["latest_size"] > min_size_cny)
        & (agg["latest_manager_tenure"] > min_manager_tenure_years)
        & (agg["avg_stock_weight"] > avg_stock_weight_min)
        & (agg["min_stock_weight"] > current_stock_weight_min)
    )
    if theme_tag_col:
        mask &= agg["latest_theme_tag"].isin(all_market_theme_values)
    return agg.assign(in_base_pool=mask).reset_index()


def detect_theme_by_keywords(
    contract_text: str,
    benchmark_name: str,
    holdings_theme_weight: float | None = None,
    theme_keywords: set[str] | None = None,
    benchmark_keywords: set[str] | None = None,
    holdings_threshold: float = 0.50,
) -> dict[str, bool]:
    """Detect a sector/theme fund from contract, benchmark and holdings evidence."""
    theme_keywords = theme_keywords or set()
    benchmark_keywords = benchmark_keywords or set()
    contract_text = "" if contract_text is None or pd.isna(contract_text) else str(contract_text)
    benchmark_name = "" if benchmark_name is None or pd.isna(benchmark_name) else str(benchmark_name)
    contract_hit = any(keyword in contract_text for keyword in theme_keywords)
    benchmark_hit = any(keyword in benchmark_name for keyword in benchmark_keywords)
    holdings_hit = holdings_theme_weight is not None and holdings_theme_weight >= holdings_threshold
    return {
        "contract_hit": bool(contract_hit),
        "benchmark_hit": bool(benchmark_hit),
        "holdings_hit": bool(holdings_hit),
        "is_theme_fund": bool(contract_hit or benchmark_hit or holdings_hit),
    }


def peer_rank_score(
    df: pd.DataFrame,
    group_col: str,
    metric_specs: dict[str, bool],
    output_prefix: str = "score",
) -> pd.DataFrame:
    """Rank metrics within peer groups and combine them.

    metric_specs maps metric column -> higher_is_better. Lower drawdown or
    volatility metrics should be set to False.
    """
    required = [group_col, *metric_specs.keys()]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    score_cols = []
    for metric, higher_is_better in metric_specs.items():
        rank_col = f"{output_prefix}_{metric}"
        out[rank_col] = out.groupby(group_col)[metric].rank(pct=True, ascending=not higher_is_better)
        score_cols.append(rank_col)
    out[f"{output_prefix}_mean"] = out[score_cols].mean(axis=1)
    return out


def weighted_label_score(
    df: pd.DataFrame,
    score_cols: dict[str, float],
    output_col: str = "weighted_score",
) -> pd.DataFrame:
    """Combine pre-computed score columns with explicit weights."""
    missing = [col for col in score_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    total_weight = sum(score_cols.values())
    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")
    out = df.copy()
    score = sum(out[col].astype(float) * weight for col, weight in score_cols.items()) / total_weight
    out[output_col] = score
    return out


def zhejiang_manager_portrait_score(
    metrics: pd.DataFrame,
    output_col: str = "manager_portrait_score",
) -> pd.DataFrame:
    """Combine active equity manager labels using the Zhejiang report structure."""
    weights = {
        "performance_score": 40.0,
        "stock_selection_score": 25.0,
        "industry_allocation_score": 5.0,
        "style_allocation_score": 5.0,
        "trading_score": 10.0,
        "timing_score": 5.0,
        "platform_score": 10.0,
    }
    available = {col: weight for col, weight in weights.items() if col in metrics.columns}
    if not available:
        raise ValueError("No expected score columns found.")
    return weighted_label_score(metrics, available, output_col=output_col)


def pool_transition(
    current: pd.DataFrame,
    fund_col: str = "fund_id",
    current_pool_col: str = "pool",
    preferred_score_col: str = "preferred_score",
    has_deep_report_col: str = "has_deep_report",
    underperform_quarters_col: str = "underperform_quarters",
    major_change_col: str = "major_change",
    preferred_cutoff: float = 0.90,
) -> pd.DataFrame:
    """Apply simple base/preferred/watch/investment pool transition rules."""
    required = [
        fund_col,
        current_pool_col,
        preferred_score_col,
        has_deep_report_col,
        underperform_quarters_col,
        major_change_col,
    ]
    missing = [col for col in required if col not in current.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = current.copy()

    def next_pool(row: pd.Series) -> str:
        if bool(row[major_change_col]):
            return "watch"
        if row[current_pool_col] == "investment" and row[underperform_quarters_col] >= 2:
            return "preferred"
        if bool(row[has_deep_report_col]) and row[current_pool_col] == "preferred":
            return "investment"
        if row[preferred_score_col] >= preferred_cutoff:
            return "preferred"
        return row[current_pool_col]

    out["next_pool"] = out.apply(next_pool, axis=1)
    return out


def fund_label_schema() -> pd.DataFrame:
    """Return a broad active-fund label schema."""
    rows = [
        ("basic_info", "fund size, benchmark, stock limit, HK access, fees, subscription limit"),
        ("performance", "return, excess win rate, drawdown, volatility, Sharpe, Calmar, scenario behavior"),
        ("portfolio_management", "position, region, industry, stock selection, concentration, style, valuation, turnover"),
        ("manager", "experience, capacity circle, philosophy, framework, trading preference, team support, negative events"),
        ("operation_review", "timing, industry allocation, stock selection, consistency, framework evolution, capacity"),
        ("tracking", "daily NAV, weekly heavy holdings, monthly market/fund changes, quarterly holdings and interviews"),
    ]
    return pd.DataFrame(rows, columns=["label_group", "description"])
