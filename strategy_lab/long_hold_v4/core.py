"""Deterministic research rules for durable dividend assets.

The module deliberately keeps three decisions separate:

1. Is the asset durable enough to own for years?
2. Is the current price deeply undervalued and no longer falling freely?
3. Is a temporary T sleeve justified after costs while the core stays intact?

No function places broker orders. All outputs are research intents subject to
the point-in-time and freshness gates in this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


class ContractError(ValueError):
    """Raised when a trust-boundary data or configuration contract fails."""


STOCK_REQUIRED = {
    "positive_profit_years_5y",
    "dividend_years_5y",
    "dividend_yield",
    "dividend_cagr_5y",
    "dividend_cut_count_5y",
    "payout_ratio",
    "roe_mean_5y",
    "roe_std_5y",
    "revenue_cagr_5y",
    "profit_cagr_5y",
    "profit_cv_5y",
    "current_pe",
    "current_pb",
    "pe_percentile_5y",
    "pb_percentile_5y",
    "sector_pe_percentile",
    "sector_pb_percentile",
    "yield_spread_cn10y",
    "annual_vol_3y",
    "max_drawdown_3y",
}

SECTOR_REQUIRED = {
    "bank": {"npl_ratio", "provision_coverage", "core_tier1_ratio"},
    "insurance": {"solvency_ratio", "new_business_value_cagr_3y"},
    "utility": {"debt_to_assets", "interest_coverage", "fcf_dividend_coverage"},
}

GENERAL_STOCK_REQUIRED = {"debt_to_assets", "interest_coverage", "fcf_dividend_coverage"}

ETF_REQUIRED = {
    "aum_cny",
    "avg_daily_amount_cny",
    "expense_ratio",
    "tracking_error_1y",
    "index_history_years",
    "distribution_years_5y",
    "index_dividend_yield",
    "index_earnings_cagr_5y",
    "pe_percentile_5y",
    "yield_spread_cn10y",
    "annual_vol_3y",
    "max_drawdown_3y",
    "total_return_history_ready",
    "price_available_date",
    "nav_available_date",
    "valuation_available_date",
    "aum_available_date",
    "distribution_available_date",
    "expense_available_date",
    "index_available_date",
    "total_return_available_date",
}

BASE_REQUIRED = {
    "as_of_date",
    "available_date",
    "asset",
    "name",
    "asset_type",
    "sector",
    "is_tradeable",
    "is_st",
    "history_years",
}

BASE_NON_NUMERIC = {"as_of_date", "available_date", "asset", "name", "asset_type", "sector", "is_tradeable", "is_st"}
BOOLEAN_REQUIRED = {"is_tradeable", "is_st", "total_return_history_ready"}
ETF_DATE_REQUIRED = {
    "price_available_date",
    "nav_available_date",
    "valuation_available_date",
    "aum_available_date",
    "distribution_available_date",
    "expense_available_date",
    "index_available_date",
    "total_return_available_date",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _require_config_keys(section: Any, required: set[str], path: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ContractError(f"{path} must be an object")
    missing = sorted(required.difference(section))
    if missing:
        raise ContractError(f"{path} missing keys: {missing}")
    return section


def _config_float(
    section: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
    maximum_open: bool = False,
) -> float:
    value = section.get(key)
    if isinstance(value, bool):
        raise ContractError(f"{path}.{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{path}.{key} must be a finite number") from exc
    if not math.isfinite(number):
        raise ContractError(f"{path}.{key} must be a finite number")
    if minimum is not None and (number <= minimum if minimum_open else number < minimum):
        boundary = ">" if minimum_open else ">="
        raise ContractError(f"{path}.{key} must be {boundary} {minimum}")
    if maximum is not None and (number >= maximum if maximum_open else number > maximum):
        boundary = "<" if maximum_open else "<="
        raise ContractError(f"{path}.{key} must be {boundary} {maximum}")
    return number


def _config_int(
    section: dict[str, Any], key: str, path: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{path}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{path}.{key} must be <= {maximum}")
    return value


def _config_bool(section: dict[str, Any], key: str, path: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{path}.{key} must be boolean")
    return value


def _config_path(section: dict[str, Any], key: str, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path}.{key} must be a non-empty path string")
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "account",
        "costs",
        "universe",
        "quality_gates",
        "score_weights",
        "entry",
        "t_strategy",
        "execution",
        "portfolio",
        "validation",
        "data",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"config missing sections: {missing}")

    model = _require_config_keys(
        config["model"], {"name", "version", "purpose", "max_snapshot_age_days", "max_price_age_days"}, "model"
    )
    for key in ("name", "version", "purpose"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise ContractError(f"model.{key} must be a non-empty string")
    _config_int(model, "max_snapshot_age_days", "model", minimum=0)
    _config_int(model, "max_price_age_days", "model", minimum=0)

    account = _require_config_keys(
        config["account"], {"initial_cash_cny", "base_currency", "allow_margin", "allow_short"}, "account"
    )
    _config_float(account, "initial_cash_cny", "account", minimum=0.0, minimum_open=True)
    if not isinstance(account["base_currency"], str) or not account["base_currency"].strip():
        raise ContractError("account.base_currency must be a non-empty string")
    if _config_bool(account, "allow_margin", "account") or _config_bool(account, "allow_short", "account"):
        raise ContractError("Long Hold V4 is cash-only and cannot enable margin or short selling")

    costs = _require_config_keys(
        config["costs"],
        {
            "stock_commission_rate",
            "etf_commission_rate",
            "minimum_commission_cny",
            "stock_sell_stamp_duty_rate",
            "stock_transfer_fee_rate",
            "etf_stamp_duty_rate",
            "slippage_bps_each_side",
        },
        "costs",
    )
    for key in (
        "stock_commission_rate",
        "etf_commission_rate",
        "stock_sell_stamp_duty_rate",
        "stock_transfer_fee_rate",
        "etf_stamp_duty_rate",
    ):
        _config_float(costs, key, "costs", minimum=0.0, maximum=1.0, maximum_open=True)
    _config_float(costs, "minimum_commission_cny", "costs", minimum=0.0)
    _config_float(costs, "slippage_bps_each_side", "costs", minimum=0.0, maximum=10_000.0)

    universe = _require_config_keys(
        config["universe"],
        {"allowed_asset_types", "allowed_sectors", "minimum_history_years", "maximum_assets"},
        "universe",
    )
    supported_asset_types = {"stock", "etf"}
    supported_sectors = {"bank", "insurance", "utility", "transport", "telecom", "consumer_staples", "dividend_index"}
    for key, supported in (("allowed_asset_types", supported_asset_types), ("allowed_sectors", supported_sectors)):
        values = universe[key]
        if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
            raise ContractError(f"universe.{key} must be a non-empty string list")
        normalized = [value.strip().lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ContractError(f"universe.{key} cannot contain duplicates")
        unsupported = sorted(set(normalized).difference(supported))
        if unsupported:
            raise ContractError(f"universe.{key} contains unsupported values: {unsupported}")
    _config_float(universe, "minimum_history_years", "universe", minimum=0.0)
    _config_int(universe, "maximum_assets", "universe", minimum=1)

    quality = _require_config_keys(
        config["quality_gates"], {"common_stock", "bank", "insurance", "utility", "general_stock", "etf"}, "quality_gates"
    )
    common = _require_config_keys(
        quality["common_stock"],
        {
            "positive_profit_years_5y_min",
            "dividend_years_5y_min",
            "roe_mean_5y_min",
            "revenue_cagr_5y_min",
            "profit_cagr_5y_min",
            "profit_cv_5y_max",
            "payout_ratio_min",
            "payout_ratio_max",
            "dividend_cut_count_5y_max",
        },
        "quality_gates.common_stock",
    )
    _config_int(common, "positive_profit_years_5y_min", "quality_gates.common_stock", minimum=0, maximum=5)
    _config_int(common, "dividend_years_5y_min", "quality_gates.common_stock", minimum=0, maximum=5)
    _config_float(common, "roe_mean_5y_min", "quality_gates.common_stock", minimum=-1.0, maximum=1.0)
    _config_float(common, "revenue_cagr_5y_min", "quality_gates.common_stock", minimum=-1.0)
    _config_float(common, "profit_cagr_5y_min", "quality_gates.common_stock", minimum=-1.0)
    _config_float(common, "profit_cv_5y_max", "quality_gates.common_stock", minimum=0.0)
    payout_min = _config_float(common, "payout_ratio_min", "quality_gates.common_stock", minimum=0.0, maximum=2.0)
    payout_max = _config_float(common, "payout_ratio_max", "quality_gates.common_stock", minimum=0.0, maximum=2.0)
    if payout_min > payout_max:
        raise ContractError("quality_gates.common_stock payout_ratio_min cannot exceed payout_ratio_max")
    _config_int(common, "dividend_cut_count_5y_max", "quality_gates.common_stock", minimum=0, maximum=5)

    bank = _require_config_keys(
        quality["bank"], {"npl_ratio_max", "provision_coverage_min", "core_tier1_ratio_min"}, "quality_gates.bank"
    )
    _config_float(bank, "npl_ratio_max", "quality_gates.bank", minimum=0.0, maximum=1.0)
    _config_float(bank, "provision_coverage_min", "quality_gates.bank", minimum=0.0)
    _config_float(bank, "core_tier1_ratio_min", "quality_gates.bank", minimum=0.0, maximum=1.0)

    insurance = _require_config_keys(
        quality["insurance"], {"solvency_ratio_min", "new_business_value_cagr_3y_min"}, "quality_gates.insurance"
    )
    _config_float(insurance, "solvency_ratio_min", "quality_gates.insurance", minimum=0.0)
    _config_float(insurance, "new_business_value_cagr_3y_min", "quality_gates.insurance", minimum=-1.0)
    for gate_name in ("utility", "general_stock"):
        gate = _require_config_keys(
            quality[gate_name],
            {"debt_to_assets_max", "interest_coverage_min", "fcf_dividend_coverage_min"},
            f"quality_gates.{gate_name}",
        )
        _config_float(gate, "debt_to_assets_max", f"quality_gates.{gate_name}", minimum=0.0, maximum=1.0)
        _config_float(gate, "interest_coverage_min", f"quality_gates.{gate_name}")
        _config_float(gate, "fcf_dividend_coverage_min", f"quality_gates.{gate_name}")

    etf_gate = _require_config_keys(
        quality["etf"],
        {
            "aum_cny_min",
            "avg_daily_amount_cny_min",
            "expense_ratio_max",
            "tracking_error_1y_max",
            "index_history_years_min",
            "distribution_years_5y_min",
            "total_return_history_required",
            "nav_max_age_days",
            "valuation_max_age_days",
            "aum_max_age_days",
            "distribution_max_age_days",
            "expense_max_age_days",
            "index_max_age_days",
        },
        "quality_gates.etf",
    )
    _config_float(etf_gate, "aum_cny_min", "quality_gates.etf", minimum=0.0)
    _config_float(etf_gate, "avg_daily_amount_cny_min", "quality_gates.etf", minimum=0.0)
    _config_float(etf_gate, "expense_ratio_max", "quality_gates.etf", minimum=0.0, maximum=1.0)
    _config_float(etf_gate, "tracking_error_1y_max", "quality_gates.etf", minimum=0.0, maximum=1.0)
    _config_float(etf_gate, "index_history_years_min", "quality_gates.etf", minimum=0.0)
    _config_int(etf_gate, "distribution_years_5y_min", "quality_gates.etf", minimum=0, maximum=5)
    _config_bool(etf_gate, "total_return_history_required", "quality_gates.etf")
    for key in (
        "nav_max_age_days",
        "valuation_max_age_days",
        "aum_max_age_days",
        "distribution_max_age_days",
        "expense_max_age_days",
        "index_max_age_days",
    ):
        _config_int(etf_gate, key, "quality_gates.etf", minimum=0)

    score_weights = _require_config_keys(
        config["score_weights"], {"quality", "dividend", "growth", "stability", "valuation"}, "score_weights"
    )
    if set(score_weights) != {"quality", "dividend", "growth", "stability", "valuation"}:
        raise ContractError("score_weights contains unsupported keys")
    weight_values = [
        _config_float(score_weights, key, "score_weights", minimum=0.0, maximum=1.0) for key in score_weights
    ]
    if not math.isclose(sum(weight_values), 1.0, abs_tol=1e-9):
        raise ContractError("score_weights must sum to one")

    entry = _require_config_keys(
        config["entry"],
        {
            "minimum_final_score",
            "pe_percentile_5y_max",
            "pb_percentile_5y_max",
            "sector_valuation_percentile_max",
            "yield_spread_cn10y_min",
            "stock_drawdown_3y_max",
            "etf_drawdown_3y_max",
            "stabilization_distance_from_60d_low_min",
            "stabilization_ma20_slope_10d_min",
            "stabilization_vol20_to_vol60_max",
            "range_efficiency_ratio_20d_max",
            "tranche_fractions",
        },
        "entry",
    )
    _config_float(entry, "minimum_final_score", "entry", minimum=0.0, maximum=100.0)
    for key in ("pe_percentile_5y_max", "pb_percentile_5y_max", "sector_valuation_percentile_max"):
        _config_float(entry, key, "entry", minimum=0.0, maximum=1.0)
    _config_float(entry, "yield_spread_cn10y_min", "entry", minimum=-1.0, maximum=1.0)
    for key in ("stock_drawdown_3y_max", "etf_drawdown_3y_max"):
        _config_float(entry, key, "entry", minimum=-1.0, maximum=0.0, maximum_open=True)
    _config_float(entry, "stabilization_distance_from_60d_low_min", "entry", minimum=0.0)
    _config_float(entry, "stabilization_ma20_slope_10d_min", "entry", minimum=-1.0, maximum=1.0)
    _config_float(entry, "stabilization_vol20_to_vol60_max", "entry", minimum=0.0, minimum_open=True)
    _config_float(entry, "range_efficiency_ratio_20d_max", "entry", minimum=0.0, maximum=1.0)
    raw_tranches = entry["tranche_fractions"]
    if not isinstance(raw_tranches, list) or not raw_tranches:
        raise ContractError("entry.tranche_fractions must be a non-empty list")
    if any(isinstance(value, bool) for value in raw_tranches):
        raise ContractError("entry.tranche_fractions must contain finite numbers")
    try:
        tranches = [float(value) for value in raw_tranches]
    except (TypeError, ValueError) as exc:
        raise ContractError("entry.tranche_fractions must contain finite numbers") from exc
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in tranches):
        raise ContractError("entry.tranche_fractions must be within (0, 1]")
    if any(current <= previous for previous, current in zip(tranches, tranches[1:])) or tranches[-1] != 1.0:
        raise ContractError("entry.tranche_fractions must be strictly ascending and end at 1.0")

    t_strategy = _require_config_keys(
        config["t_strategy"],
        {
            "enabled",
            "core_fraction_required",
            "t_sleeve_fraction_of_full_position",
            "portfolio_t_weight_cap",
            "entry_zscore_20d_max",
            "exit_zscore_20d_min",
            "minimum_holding_days",
            "maximum_holding_days",
            "minimum_edge_after_cost",
            "same_day_t_allowed",
            "stock_settlement",
            "etf_settlement",
        },
        "t_strategy",
    )
    _config_bool(t_strategy, "enabled", "t_strategy")
    _config_float(t_strategy, "core_fraction_required", "t_strategy", minimum=0.0, maximum=1.0)
    _config_float(t_strategy, "t_sleeve_fraction_of_full_position", "t_strategy", minimum=0.0, maximum=1.0)
    _config_float(t_strategy, "portfolio_t_weight_cap", "t_strategy", minimum=0.0, maximum=1.0)
    entry_z = _config_float(t_strategy, "entry_zscore_20d_max", "t_strategy")
    exit_z = _config_float(t_strategy, "exit_zscore_20d_min", "t_strategy")
    if exit_z <= entry_z:
        raise ContractError("t_strategy.exit_zscore_20d_min must exceed entry_zscore_20d_max")
    minimum_holding = _config_int(t_strategy, "minimum_holding_days", "t_strategy", minimum=0)
    maximum_holding = _config_int(t_strategy, "maximum_holding_days", "t_strategy", minimum=1)
    if maximum_holding < minimum_holding:
        raise ContractError("t_strategy.maximum_holding_days cannot be less than minimum_holding_days")
    _config_float(t_strategy, "minimum_edge_after_cost", "t_strategy", minimum=0.0, maximum=1.0)
    _config_bool(t_strategy, "same_day_t_allowed", "t_strategy")
    for asset_type in ("stock", "etf"):
        if str(t_strategy.get(f"{asset_type}_settlement", "")).upper() not in {"T+0", "T+1"}:
            raise ContractError(f"unsupported {asset_type} settlement rule")

    execution = _require_config_keys(
        config["execution"], {"order_valid_calendar_days", "max_price_deviation_bps"}, "execution"
    )
    _config_int(execution, "order_valid_calendar_days", "execution", minimum=1, maximum=31)
    _config_float(execution, "max_price_deviation_bps", "execution", minimum=0.0, maximum=10_000.0)

    portfolio = _require_config_keys(
        config["portfolio"],
        {
            "target_core_exposure",
            "minimum_cash_weight",
            "max_single_stock_weight",
            "max_single_etf_weight",
            "max_sector_weight",
            "stock_lot_size",
            "etf_lot_size",
            "drawdown_review_trigger",
            "drawdown_risk_reduction_trigger",
        },
        "portfolio",
    )
    target_core = _config_float(portfolio, "target_core_exposure", "portfolio", minimum=0.0, maximum=1.0)
    minimum_cash = _config_float(portfolio, "minimum_cash_weight", "portfolio", minimum=0.0, maximum=1.0)
    max_stock = _config_float(portfolio, "max_single_stock_weight", "portfolio", minimum=0.0, maximum=1.0, minimum_open=True)
    max_etf = _config_float(portfolio, "max_single_etf_weight", "portfolio", minimum=0.0, maximum=1.0, minimum_open=True)
    max_sector = _config_float(portfolio, "max_sector_weight", "portfolio", minimum=0.0, maximum=1.0, minimum_open=True)
    if max_stock > max_sector or max_etf > max_sector:
        raise ContractError("single-asset caps cannot exceed max_sector_weight")
    _config_int(portfolio, "stock_lot_size", "portfolio", minimum=1)
    _config_int(portfolio, "etf_lot_size", "portfolio", minimum=1)
    committed_weight = (
        target_core
        + float(t_strategy["portfolio_t_weight_cap"])
        + minimum_cash
    )
    if committed_weight > 1.0 + EPS:
        raise ContractError("core exposure plus T cap and minimum cash cannot exceed 100%")
    review_trigger = _config_float(portfolio, "drawdown_review_trigger", "portfolio")
    brake_trigger = _config_float(portfolio, "drawdown_risk_reduction_trigger", "portfolio")
    if not -1.0 < brake_trigger < review_trigger < 0.0:
        raise ContractError("drawdown triggers must satisfy -100% < brake < review < 0%")

    validation = _require_config_keys(
        config["validation"],
        {
            "require_available_date",
            "require_historical_universe",
            "require_total_return_or_adjusted_prices",
            "signal_at_close_execute_next_open",
            "cost_scenarios_bps",
            "price_index_proxy_promotion_allowed",
        },
        "validation",
    )
    for key in (
        "require_available_date",
        "require_historical_universe",
        "require_total_return_or_adjusted_prices",
        "signal_at_close_execute_next_open",
        "price_index_proxy_promotion_allowed",
    ):
        _config_bool(validation, key, "validation")
    scenarios = validation["cost_scenarios_bps"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ContractError("validation.cost_scenarios_bps must be a non-empty list")
    if any(isinstance(value, bool) for value in scenarios):
        raise ContractError("validation.cost_scenarios_bps must contain finite non-negative numbers")
    try:
        scenario_values = [float(value) for value in scenarios]
    except (TypeError, ValueError) as exc:
        raise ContractError("validation.cost_scenarios_bps must contain finite non-negative numbers") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in scenario_values):
        raise ContractError("validation.cost_scenarios_bps must contain finite non-negative numbers")
    if scenario_values != sorted(set(scenario_values)):
        raise ContractError("validation.cost_scenarios_bps must be sorted and unique")

    data = _require_config_keys(
        config["data"],
        {
            "snapshot_path",
            "price_directory",
            "account_path",
            "order_state_path",
            "trade_calendar_path",
            "agent_contracts_path",
            "output_directory",
            "source_manifest_paths",
            "timing_proxy_paths",
        },
        "data",
    )
    for key in (
        "snapshot_path",
        "price_directory",
        "account_path",
        "order_state_path",
        "trade_calendar_path",
        "agent_contracts_path",
        "output_directory",
    ):
        _config_path(data, key, "data")
    manifests = data["source_manifest_paths"]
    if not isinstance(manifests, list) or not manifests or any(not isinstance(value, str) or not value.strip() for value in manifests):
        raise ContractError("data.source_manifest_paths must be a non-empty path list")
    proxies = data["timing_proxy_paths"]
    if not isinstance(proxies, dict) or not proxies or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
        for key, value in proxies.items()
    ):
        raise ContractError("data.timing_proxy_paths must be a path mapping")


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return bool(float(value) == 1.0)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _valid_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value) in {0.0, 1.0}
    return str(value).strip().lower() in {
        "0",
        "1",
        "false",
        "true",
        "no",
        "yes",
        "n",
        "y",
        "fail",
        "pass",
        "blocked",
        "ready",
    }


def _number(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(_number(value))
    except (TypeError, ValueError):
        return False


def _higher(value: Any, bad: float, good: float) -> float:
    if not _finite(value):
        return 0.0
    if good <= bad:
        raise ValueError("good must exceed bad")
    return float(np.clip((_number(value) - bad) / (good - bad), 0.0, 1.0))


def _lower(value: Any, good: float, bad: float) -> float:
    if not _finite(value):
        return 0.0
    return 1.0 - _higher(value, good, bad)


def _centered(value: Any, low: float, target: float, high: float) -> float:
    if not _finite(value):
        return 0.0
    number = _number(value)
    if number <= target:
        return _higher(number, low, target)
    return _lower(number, target, high)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _required_for_row(row: pd.Series) -> set[str]:
    asset_type = str(row.get("asset_type", "")).lower()
    sector = str(row.get("sector", "")).lower()
    if asset_type == "etf":
        return BASE_REQUIRED | ETF_REQUIRED
    if asset_type == "stock":
        return BASE_REQUIRED | STOCK_REQUIRED | SECTOR_REQUIRED.get(sector, GENERAL_STOCK_REQUIRED)
    return BASE_REQUIRED


def audit_snapshot(snapshot: pd.DataFrame, as_of: str | pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
    """Audit each latest research row without silently accepting missing fields."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    rows: list[dict[str, Any]] = []
    allowed_types = set(config["universe"]["allowed_asset_types"])
    allowed_sectors = set(config["universe"]["allowed_sectors"])
    max_age = int(config["model"]["max_snapshot_age_days"])

    if snapshot.empty:
        return pd.DataFrame(columns=["asset", "data_gate_status", "data_gate_reasons"])
    missing_base = sorted(BASE_REQUIRED.difference(snapshot.columns))
    if missing_base:
        raise ContractError(f"snapshot missing base columns: {missing_base}")
    if snapshot["asset"].astype(str).duplicated().any():
        raise ContractError("snapshot must contain exactly one latest PIT row per asset")

    for _, row in snapshot.iterrows():
        reasons: list[str] = []
        required = _required_for_row(row)
        missing = sorted(
            col
            for col in required
            if col not in snapshot.columns
            or pd.isna(row.get(col))
            or (isinstance(row.get(col), str) and not row.get(col).strip())
        )
        if missing:
            reasons.append("missing_fields=" + ",".join(missing))
        numeric_required = required.difference(BASE_NON_NUMERIC | BOOLEAN_REQUIRED | ETF_DATE_REQUIRED)
        invalid_numeric = sorted(
            col for col in numeric_required if col not in missing and not _finite(row.get(col))
        )
        if invalid_numeric:
            reasons.append("invalid_numeric_fields=" + ",".join(invalid_numeric))
        invalid_boolean = sorted(
            col for col in required.intersection(BOOLEAN_REQUIRED) if col not in missing and not _valid_boolean(row.get(col))
        )
        if invalid_boolean:
            reasons.append("invalid_boolean_fields=" + ",".join(invalid_boolean))

        asset_type = str(row["asset_type"]).lower()
        sector = str(row["sector"]).lower()
        if asset_type not in allowed_types:
            reasons.append("asset_type_not_allowed")
        if sector not in allowed_sectors:
            reasons.append("sector_not_allowed")
        if (
            asset_type == "stock"
            and "valuation_current_cross_source_status" in snapshot.columns
            and str(row.get("valuation_current_cross_source_status", "")).lower() == "block"
        ):
            reasons.append("current_valuation_cross_source_blocked")
        if not _truth(row["is_tradeable"]):
            reasons.append("not_tradeable_or_unknown")
        if _truth(row["is_st"]):
            reasons.append("st_or_special_treatment")
        if _finite(row["history_years"]) and _number(row["history_years"]) < float(
            config["universe"]["minimum_history_years"]
        ):
            reasons.append("insufficient_history")

        available = pd.to_datetime(row["available_date"], errors="coerce")
        snapshot_date = pd.to_datetime(row["as_of_date"], errors="coerce")
        if pd.isna(available):
            reasons.append("invalid_available_date")
        elif available.normalize() > as_of_ts:
            reasons.append("future_available_date")
        if pd.isna(snapshot_date):
            reasons.append("invalid_as_of_date")
        elif snapshot_date.normalize() > as_of_ts:
            reasons.append("future_snapshot_date")
        elif (as_of_ts - snapshot_date.normalize()).days > max_age:
            reasons.append("stale_snapshot")

        if asset_type == "etf":
            etf_gate = config["quality_gates"]["etf"]
            date_limits = {
                "price_available_date": int(config["model"]["max_price_age_days"]),
                "nav_available_date": int(etf_gate["nav_max_age_days"]),
                "valuation_available_date": int(etf_gate["valuation_max_age_days"]),
                "aum_available_date": int(etf_gate["aum_max_age_days"]),
                "distribution_available_date": int(etf_gate["distribution_max_age_days"]),
                "expense_available_date": int(etf_gate["expense_max_age_days"]),
                "index_available_date": int(etf_gate["index_max_age_days"]),
                "total_return_available_date": int(config["model"]["max_price_age_days"]),
            }
            for field, age_limit in date_limits.items():
                if field in missing:
                    continue
                field_date = pd.to_datetime(row.get(field), errors="coerce")
                if pd.isna(field_date):
                    reasons.append(f"invalid_{field}")
                elif field_date.normalize() > as_of_ts:
                    reasons.append(f"future_{field}")
                elif (as_of_ts - field_date.normalize()).days > age_limit:
                    reasons.append(f"stale_{field}")

        rows.append(
            {
                "asset": str(row["asset"]),
                "data_gate_status": "pass" if not reasons else "blocked",
                "data_gate_reasons": ";".join(reasons),
                "missing_field_count": len(missing),
            }
        )
    return pd.DataFrame(rows)


def _stock_hard_gates(row: pd.Series, config: dict[str, Any]) -> list[str]:
    common = config["quality_gates"]["common_stock"]
    sector = str(row["sector"]).lower()
    checks = [
        ("nonpositive_current_pe", _number(row["current_pe"]) <= 0),
        ("nonpositive_current_pb", _number(row["current_pb"]) <= 0),
        ("insufficient_positive_profit_years", _number(row["positive_profit_years_5y"]) < common["positive_profit_years_5y_min"]),
        ("insufficient_dividend_years", _number(row["dividend_years_5y"]) < common["dividend_years_5y_min"]),
        ("roe_below_floor", _number(row["roe_mean_5y"]) < common["roe_mean_5y_min"]),
        ("negative_profit_growth", _number(row["profit_cagr_5y"]) < common["profit_cagr_5y_min"]),
        ("unstable_profit", _number(row["profit_cv_5y"]) > common["profit_cv_5y_max"]),
        ("payout_too_low", _number(row["payout_ratio"]) < common["payout_ratio_min"]),
        ("payout_too_high", _number(row["payout_ratio"]) > common["payout_ratio_max"]),
        ("repeated_dividend_cuts", _number(row["dividend_cut_count_5y"]) > common["dividend_cut_count_5y_max"]),
    ]
    if sector != "insurance":
        checks.append(("negative_revenue_growth", _number(row["revenue_cagr_5y"]) < common["revenue_cagr_5y_min"]))
    if sector == "bank":
        gate = config["quality_gates"]["bank"]
        checks.extend(
            [
                ("npl_ratio_too_high", _number(row["npl_ratio"]) > gate["npl_ratio_max"]),
                ("provision_coverage_too_low", _number(row["provision_coverage"]) < gate["provision_coverage_min"]),
                ("core_tier1_too_low", _number(row["core_tier1_ratio"]) < gate["core_tier1_ratio_min"]),
            ]
        )
    elif sector == "insurance":
        gate = config["quality_gates"]["insurance"]
        checks.extend(
            [
                ("solvency_ratio_too_low", _number(row["solvency_ratio"]) < gate["solvency_ratio_min"]),
                ("new_business_value_shrinking", _number(row["new_business_value_cagr_3y"]) < gate["new_business_value_cagr_3y_min"]),
            ]
        )
    else:
        gate_name = "utility" if sector == "utility" else "general_stock"
        gate = config["quality_gates"][gate_name]
        checks.extend(
            [
                ("leverage_too_high", _number(row["debt_to_assets"]) > gate["debt_to_assets_max"]),
                ("interest_coverage_too_low", _number(row["interest_coverage"]) < gate["interest_coverage_min"]),
                ("dividend_not_covered_by_fcf", _number(row["fcf_dividend_coverage"]) < gate["fcf_dividend_coverage_min"]),
            ]
        )
    return [name for name, failed in checks if failed]


def _etf_hard_gates(row: pd.Series, config: dict[str, Any]) -> list[str]:
    gate = config["quality_gates"]["etf"]
    checks = [
        ("aum_too_small", _number(row["aum_cny"]) < gate["aum_cny_min"]),
        ("liquidity_too_low", _number(row["avg_daily_amount_cny"]) < gate["avg_daily_amount_cny_min"]),
        ("expense_ratio_too_high", _number(row["expense_ratio"]) > gate["expense_ratio_max"]),
        ("tracking_error_too_high", _number(row["tracking_error_1y"]) > gate["tracking_error_1y_max"]),
        ("index_history_too_short", _number(row["index_history_years"]) < gate["index_history_years_min"]),
        ("distribution_history_too_short", _number(row["distribution_years_5y"]) < gate["distribution_years_5y_min"]),
        ("missing_total_return_history", gate["total_return_history_required"] and not _truth(row["total_return_history_ready"])),
    ]
    return [name for name, failed in checks if failed]


def _stock_scores(row: pd.Series) -> dict[str, float]:
    sector = str(row["sector"]).lower()
    quality_parts = [_higher(row["roe_mean_5y"], 0.06, 0.16), _lower(row["roe_std_5y"], 0.01, 0.08)]
    if sector == "bank":
        quality_parts.extend(
            [
                _lower(row["npl_ratio"], 0.008, 0.025),
                _higher(row["provision_coverage"], 1.5, 3.0),
                _higher(row["core_tier1_ratio"], 0.085, 0.13),
            ]
        )
    elif sector == "insurance":
        quality_parts.extend(
            [
                _higher(row["solvency_ratio"], 1.5, 2.5),
                _higher(row["new_business_value_cagr_3y"], 0.0, 0.10),
            ]
        )
    else:
        quality_parts.extend(
            [
                _lower(row["debt_to_assets"], 0.35, 0.75),
                _higher(row["interest_coverage"], 2.0, 8.0),
                _higher(row["fcf_dividend_coverage"], 1.0, 2.0),
            ]
        )

    dividend = _mean(
        [
            _higher(row["dividend_yield"], 0.025, 0.07),
            _higher(row["dividend_years_5y"], 4.0, 5.0),
            _higher(row["dividend_cagr_5y"], -0.02, 0.08),
            _centered(row["payout_ratio"], 0.18, 0.50, 0.85),
        ]
    )
    growth_inputs = [
        _higher(row["profit_cagr_5y"], 0.0, 0.10),
        _higher(row["new_business_value_cagr_3y"], 0.0, 0.10)
        if sector == "insurance"
        else _higher(row["revenue_cagr_5y"], 0.0, 0.08),
    ]
    growth = _mean(growth_inputs)
    stability = _mean(
        [
            _lower(row["roe_std_5y"], 0.01, 0.08),
            _lower(row["profit_cv_5y"], 0.10, 0.70),
            _lower(row["annual_vol_3y"], 0.12, 0.35),
            _lower(abs(_number(row["max_drawdown_3y"])), 0.18, 0.55),
        ]
    )
    valuation = _mean(
        [
            _lower(row["pe_percentile_5y"], 0.10, 0.80),
            _lower(row["pb_percentile_5y"], 0.10, 0.80),
            _lower(row["sector_pe_percentile"], 0.10, 0.80),
            _lower(row["sector_pb_percentile"], 0.10, 0.80),
            _higher(row["yield_spread_cn10y"], 0.005, 0.04),
        ]
    )
    penalty = min(40.0, 6.0 * _number(row["dividend_cut_count_5y"]))
    if _number(row["payout_ratio"]) > 0.70:
        penalty += 5.0
    if _number(row["profit_cagr_5y"]) <= 0.01:
        penalty += 5.0
    return {
        "quality_score": 100.0 * _mean(quality_parts),
        "dividend_score": 100.0 * dividend,
        "growth_score": 100.0 * growth,
        "stability_score": 100.0 * stability,
        "valuation_score": 100.0 * valuation,
        "value_trap_penalty": min(40.0, penalty),
    }


def _etf_scores(row: pd.Series) -> dict[str, float]:
    quality = _mean(
        [
            _higher(row["aum_cny"], 1e9, 1e10),
            _higher(row["avg_daily_amount_cny"], 5e7, 5e8),
            _lower(row["expense_ratio"], 0.002, 0.008),
            _lower(row["tracking_error_1y"], 0.003, 0.02),
        ]
    )
    dividend = _mean(
        [
            _higher(row["index_dividend_yield"], 0.025, 0.065),
            _higher(row["distribution_years_5y"], 3.0, 5.0),
        ]
    )
    growth = _higher(row["index_earnings_cagr_5y"], 0.0, 0.08)
    stability = _mean(
        [
            _lower(row["annual_vol_3y"], 0.10, 0.30),
            _lower(abs(_number(row["max_drawdown_3y"])), 0.15, 0.45),
            _higher(row["index_history_years"], 5.0, 15.0),
        ]
    )
    valuation = _mean(
        [
            _lower(row["pe_percentile_5y"], 0.10, 0.80),
            _higher(row["yield_spread_cn10y"], 0.005, 0.04),
        ]
    )
    return {
        "quality_score": 100.0 * quality,
        "dividend_score": 100.0 * dividend,
        "growth_score": 100.0 * growth,
        "stability_score": 100.0 * stability,
        "valuation_score": 100.0 * valuation,
        "value_trap_penalty": 0.0,
    }


def score_universe(snapshot: pd.DataFrame, as_of: str | pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
    """Apply PIT data gates, durable-income screens, and value-trap vetoes."""
    audit = audit_snapshot(snapshot, as_of, config)
    if snapshot.empty:
        return snapshot.copy()
    latest = snapshot.copy()
    latest["asset"] = latest["asset"].astype(str)
    out = latest.merge(audit, on="asset", how="left", validate="one_to_one")
    score_rows: list[dict[str, Any]] = []
    weights = config["score_weights"]

    for _, row in out.iterrows():
        hard_veto: list[str] = []
        scores = {
            "quality_score": np.nan,
            "dividend_score": np.nan,
            "growth_score": np.nan,
            "stability_score": np.nan,
            "valuation_score": np.nan,
            "value_trap_penalty": np.nan,
        }
        if row["data_gate_status"] == "pass":
            if str(row["asset_type"]).lower() == "etf":
                hard_veto = _etf_hard_gates(row, config)
                scores = _etf_scores(row)
            else:
                hard_veto = _stock_hard_gates(row, config)
                scores = _stock_scores(row)
        gross = sum(float(weights[key.replace("_score", "")]) * float(value) for key, value in scores.items() if key.endswith("_score") and pd.notna(value))
        final_score = float(np.clip(gross - float(scores["value_trap_penalty"]), 0.0, 100.0)) if pd.notna(scores["value_trap_penalty"]) else np.nan
        eligible = row["data_gate_status"] == "pass" and not hard_veto
        score_rows.append(
            {
                **scores,
                "gross_score": gross if pd.notna(final_score) else np.nan,
                "final_score": final_score,
                "hard_veto": bool(hard_veto),
                "hard_veto_reasons": ";".join(hard_veto),
                "durable_eligible": bool(eligible),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(score_rows)], axis=1)


def compute_price_features(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute past-only drawdown, stabilization, and range-regime features."""
    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ContractError(f"prices missing columns: {missing}")
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ContractError("prices contain invalid dates")
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["open", "high", "low", "close"]].isna().any().any():
        raise ContractError("prices contain non-numeric OHLC values")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("prices contain non-positive OHLC values")
    if ((out["high"] < out[["open", "close", "low"]].max(axis=1)) | (out["low"] > out[["open", "close", "high"]].min(axis=1))).any():
        raise ContractError("prices contain invalid OHLC ordering")

    ret = out["close"].pct_change()
    out["ret_1d"] = ret
    out["rolling_high_3y"] = out["close"].rolling(756, min_periods=252).max()
    out["drawdown_3y"] = out["close"] / out["rolling_high_3y"] - 1.0
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma60"] = out["close"].rolling(60, min_periods=60).mean()
    out["ma200"] = out["close"].rolling(200, min_periods=200).mean()
    out["ma20_slope_10d"] = out["ma20"] / out["ma20"].shift(10) - 1.0
    out["vol20"] = ret.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(252.0)
    out["vol60"] = ret.rolling(60, min_periods=60).std(ddof=1) * math.sqrt(252.0)
    out["vol_ratio_20_60"] = out["vol20"] / out["vol60"].replace(0.0, np.nan)
    std20 = out["close"].rolling(20, min_periods=20).std(ddof=1).replace(0.0, np.nan)
    out["zscore20"] = (out["close"] - out["ma20"]) / std20
    path = out["close"].diff().abs().rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    out["efficiency_ratio20"] = out["close"].diff(20).abs() / path
    out["low60"] = out["low"].rolling(60, min_periods=60).min()
    out["distance_from_low60"] = out["close"] / out["low60"].replace(0.0, np.nan) - 1.0

    entry = config["entry"]
    out["stabilized"] = (
        (out["distance_from_low60"] >= float(entry["stabilization_distance_from_60d_low_min"]))
        & (out["ma20_slope_10d"] >= float(entry["stabilization_ma20_slope_10d_min"]))
        & (out["vol_ratio_20_60"] <= float(entry["stabilization_vol20_to_vol60_max"]))
        & (out["efficiency_ratio20"] <= float(entry["range_efficiency_ratio_20d_max"]))
    )
    out["falling_knife"] = (
        (out["distance_from_low60"] < 0.01)
        | (out["ma20_slope_10d"] < -0.02)
        | (out["vol_ratio_20_60"] > 1.50)
    )
    out["range_regime"] = (
        (out["efficiency_ratio20"] <= float(entry["range_efficiency_ratio_20d_max"]))
        & (out["vol_ratio_20_60"] <= 1.20)
        & ((out["ma60"] / out["ma60"].shift(20) - 1.0).abs() <= 0.04)
    )
    out["expected_reversion_edge"] = ((out["ma20"] - out["close"]) / out["close"]).clip(lower=0.0)
    out["t_buy_setup"] = out["range_regime"] & (
        out["zscore20"] <= float(config["t_strategy"]["entry_zscore_20d_max"])
    )
    out["t_exit_setup"] = (out["zscore20"] >= float(config["t_strategy"]["exit_zscore_20d_min"])) | (
        out["close"] >= out["ma20"]
    )
    return out


def _price_is_fresh(price_date: Any, as_of: Any, config: dict[str, Any]) -> bool:
    price_ts = pd.Timestamp(price_date).normalize()
    as_of_ts = pd.Timestamp(as_of).normalize()
    age = (as_of_ts - price_ts).days
    return 0 <= age <= int(config["model"]["max_price_age_days"])


def entry_decision(
    scored_asset: pd.Series,
    latest_price: pd.Series,
    current_core_fraction: float,
    as_of: str | pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a staged core-entry decision; cash is the fail-closed default."""
    target_fraction = float(np.clip(current_core_fraction, 0.0, 1.0))
    has_core = target_fraction > EPS
    blocking_reasons: list[str] = []
    if scored_asset.get("data_gate_status") != "pass":
        blocking_reasons.append("snapshot_data_gate_blocked")
    if not _price_is_fresh(latest_price["date"], as_of, config):
        blocking_reasons.append("stale_price")
    missing_price_features = [
        field for field in ["close", "ma20", "ma60", "drawdown_3y"] if not _finite(latest_price.get(field))
    ]
    if missing_price_features:
        blocking_reasons.append("missing_price_features=" + ",".join(missing_price_features))
    if not _truth(scored_asset.get("durable_eligible")):
        blocking_reasons.append("durability_or_value_trap_gate_failed")
    if blocking_reasons:
        return {
            "entry_action": "REVIEW_CORE" if has_core else "KEEP_CASH",
            "target_core_fraction": target_fraction,
            "t_enabled": False,
            "entry_reasons": ";".join(blocking_reasons),
        }

    if not _finite(scored_asset.get("final_score")) or _number(scored_asset.get("final_score")) < float(
        config["entry"]["minimum_final_score"]
    ):
        return {
            "entry_action": "HOLD_CORE" if has_core else "KEEP_CASH",
            "target_core_fraction": target_fraction,
            "t_enabled": False,
            "entry_reasons": "held_without_addition" if has_core else "final_score_below_threshold",
        }

    entry = config["entry"]
    if str(scored_asset["asset_type"]).lower() == "etf":
        absolute_value = _number(scored_asset["pe_percentile_5y"]) <= float(entry["pe_percentile_5y_max"])
        relative_value = True
    else:
        absolute_value = (
            _number(scored_asset["pe_percentile_5y"]) <= float(entry["pe_percentile_5y_max"])
            or _number(scored_asset["pb_percentile_5y"]) <= float(entry["pb_percentile_5y_max"])
        )
        relative_value = min(
            _number(scored_asset["sector_pe_percentile"]), _number(scored_asset["sector_pb_percentile"])
        ) <= float(entry["sector_valuation_percentile_max"])
    income_spread = _number(scored_asset["yield_spread_cn10y"]) >= float(entry["yield_spread_cn10y_min"])
    if not (absolute_value and relative_value and income_spread):
        return {
            "entry_action": "HOLD_CORE" if has_core else "WAIT_DEEP_VALUE",
            "target_core_fraction": target_fraction,
            "t_enabled": bool(
                has_core
                and target_fraction >= float(config["t_strategy"]["core_fraction_required"])
                and _truth(latest_price["range_regime"])
            ),
            "entry_reasons": "held_without_addition" if has_core else "valuation_or_yield_spread_not_extreme",
        }

    drawdown_limit = float(
        entry["etf_drawdown_3y_max"] if str(scored_asset["asset_type"]).lower() == "etf" else entry["stock_drawdown_3y_max"]
    )
    if _number(latest_price["drawdown_3y"]) > drawdown_limit:
        return {
            "entry_action": "HOLD_CORE" if has_core else "WAIT_DEEP_DRAWDOWN",
            "target_core_fraction": target_fraction,
            "t_enabled": bool(
                has_core
                and target_fraction >= float(config["t_strategy"]["core_fraction_required"])
                and _truth(latest_price["range_regime"])
            ),
            "entry_reasons": "held_without_addition" if has_core else "drawdown_not_deep_enough",
        }
    if _truth(latest_price["falling_knife"]) or not _truth(latest_price["stabilized"]):
        return {
            "entry_action": "HOLD_CORE" if has_core else "WAIT_STABILIZATION",
            "target_core_fraction": target_fraction,
            "t_enabled": False,
            "entry_reasons": "held_without_addition_falling_price" if has_core else "deep_value_but_price_not_stabilized",
        }

    tranches = [float(v) for v in entry["tranche_fractions"]]
    if current_core_fraction < tranches[0] - 0.05:
        action, target_fraction = "BUILD_1", tranches[0]
    elif current_core_fraction < tranches[1] - 0.05 and _number(latest_price["close"]) >= _number(latest_price["ma20"]):
        action, target_fraction = "BUILD_2", tranches[1]
    elif current_core_fraction < tranches[2] - 0.05 and _number(latest_price["close"]) >= _number(latest_price["ma60"]):
        action, target_fraction = "BUILD_3", tranches[2]
    else:
        action, target_fraction = "HOLD_CORE", max(current_core_fraction, tranches[0])
    t_enabled = target_fraction >= float(config["t_strategy"]["core_fraction_required"]) and _truth(latest_price["range_regime"])
    return {
        "entry_action": action,
        "target_core_fraction": min(1.0, target_fraction),
        "t_enabled": bool(t_enabled),
        "entry_reasons": "durable+deep_value+deep_drawdown+stabilized",
    }


def estimate_trade_cost(notional: float, side: str, asset_type: str, config: dict[str, Any]) -> dict[str, float]:
    if notional < 0:
        raise ValueError("notional cannot be negative")
    side = side.lower()
    asset_type = asset_type.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if asset_type not in {"stock", "etf"}:
        raise ValueError("asset_type must be stock or etf")
    costs = config["costs"]
    commission_rate = float(costs[f"{asset_type}_commission_rate"])
    commission = max(float(costs["minimum_commission_cny"]), notional * commission_rate) if notional > 0 else 0.0
    stamp_rate = float(costs["stock_sell_stamp_duty_rate"]) if asset_type == "stock" and side == "sell" else 0.0
    if asset_type == "etf" and side == "sell":
        stamp_rate = float(costs["etf_stamp_duty_rate"])
    transfer_rate = float(costs["stock_transfer_fee_rate"]) if asset_type == "stock" else 0.0
    slippage_rate = float(costs["slippage_bps_each_side"]) / 10000.0
    stamp = notional * stamp_rate
    transfer = notional * transfer_rate
    slippage = notional * slippage_rate
    total = commission + stamp + transfer + slippage
    return {
        "commission": commission,
        "stamp_duty": stamp,
        "transfer_fee": transfer,
        "slippage": slippage,
        "total_cost": total,
        "effective_rate": total / notional if notional > 0 else 0.0,
    }


def round_trip_cost_rate(asset_type: str, config: dict[str, Any]) -> float:
    unit = 1_000_000.0
    return (
        estimate_trade_cost(unit, "buy", asset_type, config)["total_cost"]
        + estimate_trade_cost(unit, "sell", asset_type, config)["total_cost"]
    ) / unit


def t_decision(
    scored_asset: pd.Series,
    latest_price: pd.Series,
    core_fraction: float,
    current_t_fraction: float,
    t_holding_days: int,
    as_of: str | pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    t_cfg = config["t_strategy"]
    if not t_cfg["enabled"]:
        if current_t_fraction > EPS:
            return {"t_action": "SELL_T_NEXT_OPEN", "target_t_fraction": 0.0, "t_reasons": "t_strategy_disabled"}
        return {"t_action": "DISABLED", "target_t_fraction": 0.0, "t_reasons": "config_disabled"}
    if not _price_is_fresh(latest_price["date"], as_of, config):
        return {"t_action": "HOLD_T" if current_t_fraction > EPS else "NO_T", "target_t_fraction": current_t_fraction, "t_reasons": "stale_price"}
    if not _truth(scored_asset.get("durable_eligible")) or core_fraction < float(t_cfg["core_fraction_required"]):
        if current_t_fraction > EPS:
            return {
                "t_action": "SELL_T_NEXT_OPEN",
                "target_t_fraction": 0.0,
                "t_reasons": "durable_core_invalidated",
            }
        return {"t_action": "NO_T", "target_t_fraction": 0.0, "t_reasons": "durable_core_not_established"}

    asset_type = str(scored_asset["asset_type"]).lower()
    hurdle = round_trip_cost_rate(asset_type, config) + float(t_cfg["minimum_edge_after_cost"])
    if current_t_fraction <= EPS:
        if _truth(latest_price["t_buy_setup"]) and _number(latest_price["expected_reversion_edge"]) >= hurdle:
            return {
                "t_action": "BUY_T_NEXT_OPEN",
                "target_t_fraction": float(t_cfg["t_sleeve_fraction_of_full_position"]),
                "t_reasons": f"range_oversold_edge_above_{hurdle:.6f}",
            }
        return {"t_action": "NO_T", "target_t_fraction": 0.0, "t_reasons": "no_cost_adjusted_oversold_setup"}

    if t_holding_days >= int(t_cfg["maximum_holding_days"]):
        return {"t_action": "SELL_T_NEXT_OPEN", "target_t_fraction": 0.0, "t_reasons": "maximum_holding_days"}
    if t_holding_days >= int(t_cfg["minimum_holding_days"]) and _truth(latest_price["t_exit_setup"]):
        return {"t_action": "SELL_T_NEXT_OPEN", "target_t_fraction": 0.0, "t_reasons": "mean_reversion_exit"}
    return {"t_action": "HOLD_T", "target_t_fraction": current_t_fraction, "t_reasons": "waiting_for_exit"}


def _allocate_capped_budget(
    capacity: pd.Series,
    priority: pd.Series,
    groups: pd.Series,
    budget: float,
    group_caps: dict[str, float],
) -> pd.Series:
    """Proportionally allocate a budget without breaching asset or group caps."""
    capacity = pd.to_numeric(capacity, errors="coerce").astype(float)
    priority = pd.to_numeric(priority.reindex(capacity.index), errors="coerce").astype(float)
    groups = groups.reindex(capacity.index).astype(str)
    if capacity.isna().any() or priority.isna().any() or (capacity < -EPS).any() or not math.isfinite(float(budget)):
        raise ContractError("allocation inputs must be finite and non-negative")
    if any(not math.isfinite(float(value)) or float(value) < -EPS for value in group_caps.values()):
        raise ContractError("allocation group caps must be finite and non-negative")

    allocation = pd.Series(0.0, index=capacity.index)
    remaining = max(0.0, float(budget))
    for _ in range(200):
        if remaining <= 1e-10:
            break
        asset_room = (capacity - allocation).clip(lower=0.0)
        group_used = allocation.groupby(groups).sum()
        group_room = groups.map(
            lambda group: max(0.0, float(group_caps.get(group, 0.0)) - float(group_used.get(group, 0.0)))
        )
        active = (asset_room > 1e-10) & (group_room > 1e-10)
        if not active.any():
            break

        active_priority = priority[active].clip(lower=0.0)
        if active_priority.sum() <= EPS:
            active_priority = pd.Series(1.0, index=active_priority.index)
        addition = (remaining * active_priority / active_priority.sum()).clip(upper=asset_room[active])
        for group, indices in groups[active].groupby(groups[active]).groups.items():
            allowed = max(0.0, float(group_caps[group]) - float(group_used.get(group, 0.0)))
            proposed = float(addition.loc[indices].sum())
            if proposed > allowed + EPS:
                addition.loc[indices] *= allowed / proposed
        added = float(addition.sum())
        if added <= 1e-10:
            break
        allocation.loc[addition.index] += addition
        remaining -= added
    return allocation


def _full_target_weights(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float, name="full_target_weight")
    annual_vol = pd.to_numeric(candidates["annual_vol_3y"], errors="coerce")
    final_score = pd.to_numeric(candidates["final_score"], errors="coerce")
    if annual_vol.isna().any() or final_score.isna().any() or (annual_vol <= 0).any():
        raise ContractError("allocation candidates require finite positive volatility and finite scores")
    priority = final_score.clip(lower=0.0) / annual_vol.clip(lower=0.08)
    hard_asset_caps = candidates["asset_type"].astype(str).str.lower().map(
        {
            "stock": float(config["portfolio"]["max_single_stock_weight"]),
            "etf": float(config["portfolio"]["max_single_etf_weight"]),
        }
    )
    t_reserve = float(config["t_strategy"]["t_sleeve_fraction_of_full_position"]) if config["t_strategy"]["enabled"] else 0.0
    asset_caps = hard_asset_caps / (1.0 + t_reserve)
    sectors = candidates["sector"].astype(str)
    sector_cap = float(config["portfolio"]["max_sector_weight"])
    group_caps = {sector: sector_cap for sector in sectors.unique()}
    return _allocate_capped_budget(
        asset_caps,
        priority,
        sectors,
        float(config["portfolio"]["target_core_exposure"]),
        group_caps,
    ).rename("full_target_weight")


def allocate_core_targets(scored_and_decided: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Allocate capped full positions, then apply the staged-entry fraction.

    Residual capital remains cash. Weights are never renormalized after caps.
    """
    required = {"asset", "asset_type", "sector", "final_score", "annual_vol_3y", "entry_action", "target_core_fraction"}
    missing = sorted(required.difference(scored_and_decided.columns))
    if missing:
        raise ContractError(f"allocation input missing columns: {missing}")
    allowed_actions = {"BUILD_1", "BUILD_2", "BUILD_3", "HOLD_CORE"}
    candidates = scored_and_decided[scored_and_decided["entry_action"].isin(allowed_actions)].copy()
    candidates = candidates.sort_values(["final_score", "asset"], ascending=[False, True]).head(
        int(config["universe"]["maximum_assets"])
    )
    out = scored_and_decided.copy()
    out["full_target_weight"] = 0.0
    full = _full_target_weights(candidates, config)
    out.loc[full.index, "full_target_weight"] = full
    current_core = pd.to_numeric(
        out.get("current_core_weight", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    current_t = pd.to_numeric(
        out.get("current_t_weight", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    if (current_core < -EPS).any() or (current_t < -EPS).any():
        raise ContractError("current portfolio weights cannot be negative")
    out["current_core_weight"] = current_core
    out["current_t_weight"] = current_t

    portfolio = config["portfolio"]
    core_cap = float(portfolio["target_core_exposure"])
    t_cap = float(config["t_strategy"]["portfolio_t_weight_cap"])
    sector_cap = float(portfolio["max_sector_weight"])
    minimum_cash = float(portfolio["minimum_cash_weight"])
    hard_asset_caps = out["asset_type"].astype(str).str.lower().map(
        {"stock": float(portfolio["max_single_stock_weight"]), "etf": float(portfolio["max_single_etf_weight"])}
    )
    current_combined_sector = out.assign(_combined=current_core + current_t).groupby("sector")["_combined"].sum()
    risk_reasons: list[str] = []
    if hard_asset_caps.isna().any() or (current_core + current_t > hard_asset_caps + 1e-9).any():
        risk_reasons.append("current_single_asset_cap_breach")
    if current_core.sum() > core_cap + 1e-9:
        risk_reasons.append("current_core_portfolio_cap_breach")
    if current_t.sum() > t_cap + 1e-9:
        risk_reasons.append("current_t_portfolio_cap_breach")
    if not current_combined_sector.empty and current_combined_sector.max() > sector_cap + 1e-9:
        risk_reasons.append("current_sector_cap_breach")
    if current_core.sum() + current_t.sum() > 1.0 - minimum_cash + 1e-9:
        risk_reasons.append("current_minimum_cash_breach")

    proposed_core = out["full_target_weight"] * out["target_core_fraction"].astype(float).clip(0.0, 1.0)
    out["target_core_weight"] = current_core
    build_mask = out["entry_action"].astype(str).str.startswith("BUILD")
    core_capacity = (proposed_core - current_core).clip(lower=0.0).where(build_mask, 0.0)
    core_add_candidates = core_capacity > EPS
    if core_add_candidates.any() and not risk_reasons:
        current_combined_sector = (current_core + current_t).groupby(out["sector"]).sum()
        sector_rooms = {
            sector: max(0.0, sector_cap - float(current_combined_sector.get(sector, 0.0)))
            for sector in out.loc[core_add_candidates, "sector"].astype(str).unique()
        }
        priority = out.loc[core_add_candidates, "final_score"].astype(float) / out.loc[
            core_add_candidates, "annual_vol_3y"
        ].astype(float).clip(lower=0.08)
        out.loc[core_add_candidates, "target_core_weight"] += _allocate_capped_budget(
            core_capacity[core_add_candidates],
            priority,
            out.loc[core_add_candidates, "sector"],
            max(0.0, core_cap - float(current_core.sum())),
            sector_rooms,
        )

    t_capacity = (
        out["full_target_weight"] * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"]) - current_t
    ).clip(lower=0.0)
    out["target_t_weight_cap"] = current_t
    t_candidates = t_capacity > EPS
    if t_candidates.any() and not risk_reasons:
        committed_sector = out.assign(_combined=out["target_core_weight"] + current_t).groupby("sector")["_combined"].sum()
        t_sector_rooms = {
            sector: max(0.0, sector_cap - float(committed_sector.get(sector, 0.0)))
            for sector in out.loc[t_candidates, "sector"].astype(str).unique()
        }
        portfolio_room = max(
            0.0,
            1.0 - minimum_cash - float(out["target_core_weight"].sum()) - float(current_t.sum()),
        )
        t_priority = out.loc[t_candidates, "final_score"].astype(float) / out.loc[
            t_candidates, "annual_vol_3y"
        ].astype(float).clip(lower=0.08)
        out.loc[t_candidates, "target_t_weight_cap"] += _allocate_capped_budget(
            t_capacity[t_candidates],
            t_priority,
            out.loc[t_candidates, "sector"],
            min(max(0.0, t_cap - float(current_t.sum())), portfolio_room),
            t_sector_rooms,
        )

    out["manual_risk_review_required"] = bool(risk_reasons)
    out["portfolio_risk_reasons"] = ";".join(risk_reasons)
    combined_sector = out.assign(_combined=out["target_core_weight"] + out["target_t_weight_cap"]).groupby(
        "sector"
    )["_combined"].sum()
    deployed_cap = float(out["target_core_weight"].sum() + out["target_t_weight_cap"].sum())
    if not risk_reasons:
        if ((out["target_core_weight"] + out["target_t_weight_cap"]) > hard_asset_caps + 1e-9).any():
            raise AssertionError("combined single-asset cap violated")
        if out["target_core_weight"].sum() > core_cap + 1e-9:
            raise AssertionError("core exposure cap violated")
        if out["target_t_weight_cap"].sum() > t_cap + 1e-9:
            raise AssertionError("portfolio T sleeve cap violated")
        if not combined_sector.empty and combined_sector.max() > sector_cap + 1e-9:
            raise AssertionError("combined core and T sector cap violated")
        if deployed_cap > 1.0 - minimum_cash + 1e-9:
            raise AssertionError("minimum cash reserve violated")
    out["target_cash_weight"] = max(0.0, 1.0 - deployed_cap)
    return out
