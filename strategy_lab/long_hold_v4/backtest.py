"""Event-driven weight backtest for the V4 core and T sleeves.

Signals are observed after a close and can only execute at the next available
open. The engine compounds actual daily holdings; it never compounds
overlapping forward-return labels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .core import ContractError, estimate_trade_cost


INVESTABLE_RETURN_BASES = {"qfq_adjusted", "hfq_adjusted", "total_return"}
DIAGNOSTIC_RETURN_BASES = INVESTABLE_RETURN_BASES | {"price_index_proxy"}


def _is_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _validate_target_risk(prices: pd.DataFrame, targets: pd.DataFrame, config: dict[str, Any]) -> None:
    asset_types = prices.drop_duplicates("asset").set_index("asset")["asset_type"].astype(str).str.lower()
    typed = targets.copy()
    typed["_asset_type"] = typed["asset"].map(asset_types)
    if typed["_asset_type"].isna().any():
        raise ContractError("target asset type is unavailable")
    asset_caps = typed["_asset_type"].map(
        {
            "stock": float(config["portfolio"]["max_single_stock_weight"]),
            "etf": float(config["portfolio"]["max_single_etf_weight"]),
        }
    )
    if asset_caps.isna().any():
        raise ContractError("unsupported target asset type")
    if (typed["target_core_weight"] > asset_caps + 1e-9).any():
        raise ContractError("target core weight exceeds single-asset cap")
    if ((typed["target_core_weight"] + typed["target_t_weight"]) > asset_caps + 1e-9).any():
        raise ContractError("combined core and T weight exceeds single-asset cap")
    t_asset_caps = asset_caps * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"])
    if (typed["target_t_weight"] > t_asset_caps + 1e-9).any():
        raise ContractError("target T weight exceeds single-asset sleeve cap")
    active_t = typed["target_t_weight"] > 1e-12
    support_ratio = float(config["t_strategy"]["t_sleeve_fraction_of_full_position"]) / float(
        config["t_strategy"]["core_fraction_required"]
    )
    if (active_t & (typed["target_core_weight"] <= 1e-12)).any():
        raise ContractError("target T weight requires an established core target")
    if (
        active_t
        & (typed["target_t_weight"] > typed["target_core_weight"] * support_ratio + 1e-9)
    ).any():
        raise ContractError("target T weight exceeds supported core ratio")

    grouped = typed.groupby("signal_date")[["target_core_weight", "target_t_weight"]].sum()
    if (grouped["target_core_weight"] > float(config["portfolio"]["target_core_exposure"]) + 1e-9).any():
        raise ContractError("target core weights exceed portfolio core cap")
    if (grouped["target_t_weight"] > float(config["t_strategy"]["portfolio_t_weight_cap"]) + 1e-9).any():
        raise ContractError("target T weights exceed portfolio T cap")
    investable_cap = 1.0 - float(config["portfolio"]["minimum_cash_weight"])
    if (grouped.sum(axis=1) > investable_cap + 1e-9).any():
        raise ContractError("target weights violate minimum cash reserve")

    nonzero_assets = typed.assign(_active=(typed["target_core_weight"] + typed["target_t_weight"]) > 1e-12).groupby(
        "signal_date"
    )["_active"].sum()
    if (nonzero_assets > int(config["universe"]["maximum_assets"])).any():
        raise ContractError("target asset count exceeds configured maximum")
    if "sector" in typed.columns and typed["sector"].notna().all() and typed["sector"].astype(str).str.strip().ne("").all():
        sector_weight = typed.assign(
            _sector_weight=typed["target_core_weight"] + typed["target_t_weight"]
        ).groupby(["signal_date", "sector"])["_sector_weight"].sum()
        if (sector_weight > float(config["portfolio"]["max_sector_weight"]) + 1e-9).any():
            raise ContractError("target weights exceed sector cap")


def _promotion_blockers(prices: pd.DataFrame, targets: pd.DataFrame, config: dict[str, Any] | None) -> list[str]:
    blockers: list[str] = []
    bases = set(prices["return_basis"].astype(str).str.lower())
    if not bases.issubset(INVESTABLE_RETURN_BASES):
        blockers.append("non_investable_return_basis")
    if config is None:
        blockers.append("missing_validation_config")
        return blockers
    if bool(config["validation"].get("require_available_date", True)) and "available_date" not in targets.columns:
        blockers.append("missing_target_available_date")
    if bool(config["validation"].get("require_historical_universe", True)):
        if "historical_backtest_allowed" not in targets.columns:
            blockers.append("missing_historical_universe_evidence")
        elif not targets["historical_backtest_allowed"].map(_is_true).all():
            blockers.append("historical_universe_evidence_failed")
    if "sector" not in targets.columns or targets["sector"].isna().any() or targets["sector"].astype(str).str.strip().eq("").any():
        blockers.append("missing_sector_metadata")
    return blockers


def validate_backtest_inputs(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    price_required = {"date", "asset", "asset_type", "open", "close", "return_basis"}
    target_required = {"signal_date", "asset", "target_core_weight", "target_t_weight"}
    price_missing = sorted(price_required.difference(prices.columns))
    target_missing = sorted(target_required.difference(targets.columns))
    if price_missing:
        raise ContractError(f"backtest prices missing columns: {price_missing}")
    if target_missing:
        raise ContractError(f"backtest targets missing columns: {target_missing}")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["asset"] = p["asset"].astype(str)
    p["open"] = pd.to_numeric(p["open"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    if p[["date", "open", "close"]].isna().any().any() or (p[["open", "close"]] <= 0).any().any():
        raise ContractError("backtest prices contain invalid dates or prices")
    if p.duplicated(["date", "asset"]).any():
        raise ContractError("backtest prices contain duplicate date-asset rows")
    bases = set(p["return_basis"].astype(str).str.lower())
    invalid = sorted(bases.difference(DIAGNOSTIC_RETURN_BASES))
    if invalid:
        raise ContractError(f"unsupported return_basis values: {invalid}")
    inconsistent_types = p.groupby("asset")["asset_type"].nunique()
    if (inconsistent_types > 1).any():
        raise ContractError("asset_type changes within price history")

    t = targets.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"], errors="coerce")
    t["asset"] = t["asset"].astype(str)
    for col in ["target_core_weight", "target_t_weight"]:
        t[col] = pd.to_numeric(t[col], errors="coerce")
    if t[["signal_date", "target_core_weight", "target_t_weight"]].isna().any().any():
        raise ContractError("backtest targets contain invalid values")
    if t.duplicated(["signal_date", "asset"]).any():
        raise ContractError("backtest targets contain duplicate signal-date asset rows")
    if (t[["target_core_weight", "target_t_weight"]] < -1e-12).any().any():
        raise ContractError("negative target weights are not allowed")
    if "available_date" in t.columns:
        t["available_date"] = pd.to_datetime(t["available_date"], errors="coerce")
        if t["available_date"].isna().any():
            raise ContractError("backtest targets contain invalid available_date values")
        if (t["available_date"].dt.normalize() > t["signal_date"].dt.normalize()).any():
            raise ContractError("backtest targets contain look-ahead available_date values")
    missing_assets = sorted(set(t["asset"]).difference(p["asset"]))
    if missing_assets:
        raise ContractError(f"targets reference assets without prices: {missing_assets}")
    grouped = t.groupby("signal_date")[["target_core_weight", "target_t_weight"]].sum().sum(axis=1)
    if (grouped > 1.0 + 1e-9).any():
        raise ContractError("target weights exceed 100%")
    if config is not None:
        _validate_target_risk(p, t, config)
    input_contract_ready = not _promotion_blockers(p, t, config)
    return p.sort_values(["date", "asset"]), t.sort_values(["signal_date", "asset"]), input_contract_ready


def _next_execution_map(signal_dates: pd.Series, trading_dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    values = trading_dates.to_numpy()
    for signal_date in pd.DatetimeIndex(signal_dates.dropna().unique()).sort_values():
        location = int(np.searchsorted(values, np.datetime64(signal_date), side="right"))
        if location < len(trading_dates):
            mapping[pd.Timestamp(signal_date)] = pd.Timestamp(trading_dates[location])
    return mapping


def run_weight_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    initial_cash: float | None = None,
) -> dict[str, Any]:
    """Run a multi-asset core/T backtest with next-open execution."""
    p, t, input_contract_ready = validate_backtest_inputs(prices, targets, config)
    if initial_cash is None:
        initial_cash = float(config["account"]["initial_cash_cny"])
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    open_wide = p.pivot(index="date", columns="asset", values="open").reindex(dates)
    close_wide = p.pivot(index="date", columns="asset", values="close").reindex(dates)
    asset_type = p.drop_duplicates("asset").set_index("asset")["asset_type"].astype(str).str.lower().to_dict()
    execution_map = _next_execution_map(t["signal_date"], dates)
    scheduled: dict[pd.Timestamp, pd.DataFrame] = {}
    for signal_date, execution_date in execution_map.items():
        if execution_date in scheduled:
            raise ContractError(f"multiple signal dates map to one execution date: {execution_date.date()}")
        scheduled[execution_date] = t[t["signal_date"] == signal_date].copy()

    assets = sorted(set(p["asset"]) | set(t["asset"]))
    core = pd.Series(0.0, index=assets)
    sleeve_t = pd.Series(0.0, index=assets)
    cash = float(initial_cash)
    previous_close: pd.Series | None = None
    nav_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for date in dates:
        open_px = open_wide.loc[date].reindex(assets)
        close_px = close_wide.loc[date].reindex(assets)
        if previous_close is not None:
            held_overnight = (core + sleeve_t) > 1e-10
            missing_held = held_overnight & (previous_close.isna() | open_px.isna())
            if missing_held.any():
                raise ContractError(
                    f"held assets missing previous close or current open: {sorted(missing_held[missing_held].index)}"
                )
            overnight = (open_px / previous_close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            core *= 1.0 + overnight
            sleeve_t *= 1.0 + overnight

        day_cost = 0.0
        day_notional = 0.0
        if date in scheduled:
            plan = scheduled[date].set_index("asset")
            if open_px.reindex(plan.index).isna().any():
                missing_open = sorted(open_px.reindex(plan.index)[open_px.reindex(plan.index).isna()].index)
                raise ContractError(f"cannot execute without an open price: {missing_open}")
            nav_open = cash + float(core.sum()) + float(sleeve_t.sum())
            desired_core = plan["target_core_weight"].reindex(assets).fillna(0.0) * nav_open
            desired_t = plan["target_t_weight"].reindex(assets).fillna(0.0) * nav_open
            core_delta = desired_core - core
            t_delta = desired_t - sleeve_t

            # Sells fund buys; the strategy always reserves cash, so cost does not
            # require leverage or an ex-post weight renormalization.
            for sleeve_name, delta in (("core", core_delta), ("t", t_delta)):
                for asset, amount in delta[delta < -1e-10].items():
                    notional = abs(float(amount))
                    cost = estimate_trade_cost(notional, "sell", asset_type[asset], config)["total_cost"]
                    cash += notional - cost
                    day_cost += cost
                    day_notional += notional
                    trade_rows.append(
                        {
                            "execution_date": date,
                            "asset": asset,
                            "sleeve": sleeve_name,
                            "side": "sell",
                            "notional": notional,
                            "cost": cost,
                        }
                    )
            core = core.add(core_delta.clip(upper=0.0), fill_value=0.0).clip(lower=0.0)
            sleeve_t = sleeve_t.add(t_delta.clip(upper=0.0), fill_value=0.0).clip(lower=0.0)

            total_buy_need = 0.0
            buy_specs: list[tuple[str, str, float, float]] = []
            for sleeve_name, delta in (("core", core_delta), ("t", t_delta)):
                for asset, amount in delta[delta > 1e-10].items():
                    notional = float(amount)
                    cost = estimate_trade_cost(notional, "buy", asset_type[asset], config)["total_cost"]
                    total_buy_need += notional + cost
                    buy_specs.append((sleeve_name, asset, notional, cost))
            scale = min(1.0, cash / total_buy_need) if total_buy_need > 0 else 1.0
            for sleeve_name, asset, raw_notional, raw_cost in buy_specs:
                notional = raw_notional * scale
                cost = estimate_trade_cost(notional, "buy", asset_type[asset], config)["total_cost"]
                cash -= notional + cost
                if sleeve_name == "core":
                    core.loc[asset] += notional
                else:
                    sleeve_t.loc[asset] += notional
                day_cost += cost
                day_notional += notional
                trade_rows.append(
                    {
                        "execution_date": date,
                        "asset": asset,
                        "sleeve": sleeve_name,
                        "side": "buy",
                        "notional": notional,
                        "cost": cost,
                    }
                )

        held_intraday = (core + sleeve_t) > 1e-10
        missing_held = held_intraday & (open_px.isna() | close_px.isna())
        if missing_held.any():
            raise ContractError(f"held assets missing current open or close: {sorted(missing_held[missing_held].index)}")
        intraday = (close_px / open_px - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        core *= 1.0 + intraday
        sleeve_t *= 1.0 + intraday
        nav = cash + float(core.sum()) + float(sleeve_t.sum())
        if nav <= 0 or cash < -1e-6:
            raise AssertionError("backtest became unfunded")
        nav_rows.append(
            {
                "date": date,
                "nav": nav,
                "cash": cash,
                "core_value": float(core.sum()),
                "t_value": float(sleeve_t.sum()),
                "core_weight": float(core.sum()) / nav,
                "t_weight": float(sleeve_t.sum()) / nav,
                "trade_notional": day_notional,
                "trading_cost": day_cost,
                "one_way_turnover": 0.5 * day_notional / nav,
            }
        )
        previous_close = close_px

    nav = pd.DataFrame(nav_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = summarize_backtest(nav, float(initial_cash))
    promotion_blockers = _promotion_blockers(p, t, config)
    if input_contract_ready:
        promotion_blockers.append("walk_forward_validation_not_supplied")
    metrics["input_contract_ready"] = bool(input_contract_ready)
    metrics["promotion_allowed"] = False
    metrics["promotion_blocking_reasons"] = promotion_blockers
    metrics["promotion_scope"] = "full strategy promotion requires separate walk-forward evidence"
    metrics["return_basis"] = ",".join(sorted(set(p["return_basis"].astype(str).str.lower())))
    metrics["signal_execution"] = "close_signal_next_open_execution"
    metrics["overlapping_forward_returns_compounded"] = False
    return {"nav": nav, "trades": trades, "metrics": metrics}


def summarize_backtest(nav: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    if nav.empty:
        raise ValueError("nav is empty")
    values = nav["nav"].astype(float)
    returns = pd.concat([pd.Series([values.iloc[0] / initial_cash - 1.0]), values.pct_change().iloc[1:]], ignore_index=True)
    periods = int(len(returns))
    total_return = float(values.iloc[-1] / initial_cash - 1.0)
    annual_return = float((values.iloc[-1] / initial_cash) ** (252.0 / periods) - 1.0) if periods > 0 else np.nan
    annual_vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if periods > 1 else np.nan
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)) if periods > 1 and returns.std(ddof=1) > 0 else np.nan
    wealth = pd.concat([pd.Series([initial_cash]), values], ignore_index=True)
    max_drawdown = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "periods": periods,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe_zero_rf": sharpe,
        "max_drawdown": max_drawdown,
        "total_trading_cost": float(nav["trading_cost"].sum()),
        "average_one_way_turnover": float(nav["one_way_turnover"].mean()),
        "average_cash_weight": float((nav["cash"] / nav["nav"]).mean()),
    }
