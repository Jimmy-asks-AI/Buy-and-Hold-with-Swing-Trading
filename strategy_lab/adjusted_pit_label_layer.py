"""Adjusted/PIT forward-return label layer contracts and gates.

V3.51 defines what is required before V3.49/V3.50 signals can be validated.
It deliberately refuses to create labels from unadjusted raw close data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_LABEL_COLUMNS = {
    "signal_date",
    "asset",
    "horizon",
    "forward_adjusted_return",
    "return_basis",
    "label_available_date",
    "price_adjustment_source",
}

ALLOWED_RETURN_BASIS = {"adjusted_return", "total_return"}


LABEL_SOURCE_REQUIREMENTS = [
    {
        "label_scope": "market",
        "dataset_id": "market_total_return_index",
        "priority": 1,
        "required_for": "MARKET-level forward total-return labels for V3.50 market-state signals.",
        "provider_options": "CSI total-return index vendor; JoinQuant index price with fq support; self-built adjusted broad-market index",
        "required_fields": "date,asset_or_index,total_return_index_or_adjusted_close,available_date,data_source,source_vintage",
        "current_status": "missing",
        "blocker": "No verified PIT total-return market index is available in the current local dataset.",
        "manual_action": "obtain_market_total_return_index_or_authorize_provider_api",
    },
    {
        "label_scope": "stock",
        "dataset_id": "stock_adj_factor",
        "priority": 1,
        "required_for": "Stock-level adjusted close and adjusted forward returns.",
        "provider_options": "tushare.adj_factor; JoinQuant get_price fq=pre/post; other PIT adjustment-factor vendor",
        "required_fields": "date,asset,adj_factor,available_date,data_source,source_vintage",
        "current_status": "missing_permission",
        "blocker": "Tushare adj_factor was recorded as unavailable under the current permission profile.",
        "manual_action": "open_tushare_adj_factor_or_obtain_adjustment_factor_history",
    },
    {
        "label_scope": "stock",
        "dataset_id": "stock_universe_all_status",
        "priority": 1,
        "required_for": "Lifecycle-aware labels and survivorship-bias control.",
        "provider_options": "tushare.stock_basic; JoinQuant security master; exchange security master",
        "required_fields": "asset,name,exchange,list_status,list_date,delist_date,available_date,data_source",
        "current_status": "missing_permission",
        "blocker": "Tushare stock_basic was recorded as unavailable under the current permission profile.",
        "manual_action": "open_tushare_stock_basic_or_obtain_all_status_security_master",
    },
    {
        "label_scope": "stock",
        "dataset_id": "stock_tradeability_flags",
        "priority": 2,
        "required_for": "Suspend/ST/limit-aware label usability and execution screens.",
        "provider_options": "tushare.suspend_d + stk_limit + namechange; JoinQuant paused/is_st fields",
        "required_fields": "date,asset,is_paused,is_st,up_limit,down_limit,available_date,data_source",
        "current_status": "missing_permission",
        "blocker": "Tradeability flags are not available in the current daily-only pipeline.",
        "manual_action": "obtain_tradeability_flags_from_tushare_or_joinquant",
    },
    {
        "label_scope": "stock",
        "dataset_id": "stock_daily_raw_accepted",
        "priority": 2,
        "required_for": "Raw base bars used only as inputs to derived adjusted views.",
        "provider_options": "V3.46 accepted daily-only adapter",
        "required_fields": "date,asset,open,high,low,close,volume,amount,available_date,data_source,price_adjustment",
        "current_status": "available_raw_only",
        "blocker": "Available but explicitly marked none_raw; cannot by itself produce adjusted labels.",
        "manual_action": "combine_with_adj_factor_before_any_return_label",
    },
    {
        "label_scope": "calendar",
        "dataset_id": "trade_calendar",
        "priority": 2,
        "required_for": "Trading-day horizon alignment and label_available_date calculation.",
        "provider_options": "exchange calendar; Tushare trade_cal; existing business-date proxy with validation",
        "required_fields": "calendar_date,is_trading_day,exchange,available_date,data_source",
        "current_status": "partial_proxy_available",
        "blocker": "Current business-date proxy is enough for file coverage but not a formal exchange calendar label source.",
        "manual_action": "obtain_or_validate_exchange_trade_calendar",
    },
]


LABEL_BUILD_STEPS = [
    {
        "step": 1,
        "name": "ingest_pit_sources",
        "description": "Ingest adjustment factors, security master, tradeability flags, and market total-return index with available_date and source_vintage.",
        "gate": "all priority-1 datasets present and point-in-time valid",
    },
    {
        "step": 2,
        "name": "derive_adjusted_prices",
        "description": "Join accepted raw close with same-date adjustment factor to build derived adjusted close without mutating raw files.",
        "gate": "price_adjustment_source is explicit and no raw close is labelled as adjusted",
    },
    {
        "step": 3,
        "name": "build_forward_labels",
        "description": "For each signal_date, asset, and horizon, compute future adjusted or total return using trading-day horizons.",
        "gate": "label_available_date is strictly after signal_date and horizon end date",
    },
    {
        "step": 4,
        "name": "join_signal_contract",
        "description": "Join labels to V3.50 signal_panel by signal_date, asset, and horizon.",
        "gate": "no label exists for unavailable future dates or missing adjusted source",
    },
    {
        "step": 5,
        "name": "state_stratified_validation",
        "description": "Only after labels pass, run V3.49 IC/ICIR/hit-rate/effect-size/negative-control validation.",
        "gate": "no portfolio backtest or model promotion before validation evidence",
    },
]


@dataclass(frozen=True)
class LabelLayerConfig:
    horizons: tuple[int, ...]
    signal_asset_values: tuple[str, ...]
    min_required_signal_rows: int


def dataframe_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def validate_label_frame(labels: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    missing = sorted(REQUIRED_LABEL_COLUMNS.difference(labels.columns))
    if missing:
        issues.append(f"missing_required_label_columns={missing}")
        return issues
    basis = set(labels["return_basis"].astype(str).unique())
    if not basis.issubset(ALLOWED_RETURN_BASIS):
        issues.append(f"invalid_return_basis={sorted(basis.difference(ALLOWED_RETURN_BASIS))}")
    raw_sources = labels["price_adjustment_source"].astype(str).str.lower()
    if raw_sources.str.contains("none_raw|raw_close|unadjusted", regex=True).any():
        issues.append("raw_or_unadjusted_source_used_for_adjusted_label")
    signal_dates = pd.to_datetime(labels["signal_date"].astype(str), errors="coerce")
    available_dates = pd.to_datetime(labels["label_available_date"].astype(str), errors="coerce")
    if (available_dates <= signal_dates).any():
        issues.append("label_available_date_not_after_signal_date")
    return issues


def label_path_status(label_path: Path | None) -> tuple[bool, str, pd.DataFrame]:
    if label_path is None:
        return False, "missing_adjusted_pit_label_path", pd.DataFrame()
    if not label_path.exists():
        return False, f"label_path_not_found={label_path}", pd.DataFrame()
    labels = pd.read_csv(label_path, encoding="utf-8-sig", low_memory=False)
    issues = validate_label_frame(labels)
    if issues:
        return False, ";".join(issues), labels
    return True, "adjusted_pit_label_frame_ready", labels


def infer_required_label_scopes(signal_panel: pd.DataFrame) -> pd.DataFrame:
    if "asset" not in signal_panel.columns:
        raise ValueError("signal panel missing asset column")
    assets = signal_panel["asset"].astype(str)
    rows = []
    if (assets == "MARKET").any():
        rows.append(
            {
                "label_scope": "market",
                "asset": "MARKET",
                "signal_rows": int((assets == "MARKET").sum()),
                "required_dataset": "market_total_return_index",
                "status": "blocked_missing_market_total_return_label_source",
            }
        )
    stock_rows = int((assets != "MARKET").sum())
    if stock_rows:
        rows.append(
            {
                "label_scope": "stock",
                "asset": "*",
                "signal_rows": stock_rows,
                "required_dataset": "stock_adj_factor + stock_universe_all_status",
                "status": "blocked_missing_stock_adjustment_and_lifecycle_sources",
            }
        )
    return pd.DataFrame(rows)


def build_label_schema_template(horizons: tuple[int, ...], assets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for asset in assets:
        for horizon in horizons:
            rows.append(
                {
                    "signal_date": "YYYYMMDD",
                    "asset": asset,
                    "horizon": horizon,
                    "forward_adjusted_return": "",
                    "return_basis": "adjusted_return",
                    "label_available_date": "YYYYMMDD_after_horizon_end",
                    "price_adjustment_source": "explicit_pit_adjustment_or_total_return_source",
                }
            )
    return pd.DataFrame(rows)


def build_manual_request_queue(requirements: pd.DataFrame) -> str:
    lines = [
        "# Manual Data Request Queue V3.51",
        "",
        "These datasets are required before V3.49/V3.50 can run real state-stratified performance validation.",
        "Do not replace them with unadjusted raw close returns.",
        "",
        "| Priority | Scope | Dataset | Current Status | Manual Action |",
        "|---:|---|---|---|---|",
    ]
    for row in requirements.sort_values(["priority", "dataset_id"]).itertuples(index=False):
        lines.append(
            f"| {int(row.priority)} | `{row.label_scope}` | `{row.dataset_id}` | `{row.current_status}` | `{row.manual_action}` |"
        )
    lines.extend(
        [
            "",
            "## Minimum To Unblock Current V3.50 MARKET Signals",
            "",
            "1. A point-in-time market total-return or adjusted market index series for `MARKET` labels.",
            "2. A verified trade calendar for horizon alignment.",
            "3. If future signals become stock-level, `adj_factor` plus all-status security master become mandatory.",
            "",
        ]
    )
    return "\n".join(lines)


def build_readiness_checks(
    signal_panel: pd.DataFrame,
    requirements: pd.DataFrame,
    label_ready: tuple[bool, str, pd.DataFrame],
    config: LabelLayerConfig,
) -> pd.DataFrame:
    market_needed = bool((signal_panel["asset"].astype(str) == "MARKET").any())
    stock_needed = bool((signal_panel["asset"].astype(str) != "MARKET").any())
    priority1 = requirements.loc[requirements["priority"] == 1]
    available_priority1 = priority1["current_status"].astype(str).isin({"available", "ready"}).all()
    rows = [
        {
            "check": "signal_panel_has_required_rows",
            "status": "pass" if int(signal_panel.shape[0]) >= config.min_required_signal_rows else "fail",
            "detail": str(int(signal_panel.shape[0])),
        },
        {
            "check": "market_label_scope_required",
            "status": "pass" if market_needed else "not_applicable",
            "detail": "MARKET" if market_needed else "",
        },
        {
            "check": "stock_label_scope_required",
            "status": "pass" if stock_needed else "not_applicable",
            "detail": str(int((signal_panel["asset"].astype(str) != "MARKET").sum())),
        },
        {
            "check": "priority1_label_sources_available",
            "status": "pass" if available_priority1 else "blocked",
            "detail": ",".join(priority1.loc[~priority1["current_status"].astype(str).isin({"available", "ready"}), "dataset_id"].astype(str)),
        },
        {
            "check": "adjusted_pit_label_path_ready",
            "status": "pass" if label_ready[0] else "blocked",
            "detail": label_ready[1],
        },
        {
            "check": "raw_close_label_substitution_forbidden",
            "status": "pass",
            "detail": "none_raw/raw_close/unadjusted sources are rejected by validate_label_frame",
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "pass" if label_ready[0] and available_priority1 else "blocked",
            "detail": "requires adjusted/PIT labels and priority-1 sources",
        },
    ]
    return pd.DataFrame(rows)


def build_no_label_guard(label_ready: tuple[bool, str, pd.DataFrame]) -> pd.DataFrame:
    produced = bool(label_ready[0])
    return pd.DataFrame(
        [
            {
                "result_type": "adjusted_pit_forward_return_labels",
                "produced": produced,
                "blocked": not produced,
                "reason": "ready" if produced else label_ready[1],
            },
            {
                "result_type": "state_stratified_performance_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.51 defines/validates label readiness only; V3.49 performs validation after labels exist.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No portfolio backtest may use unadjusted daily-only raw data.",
            },
        ]
    )
