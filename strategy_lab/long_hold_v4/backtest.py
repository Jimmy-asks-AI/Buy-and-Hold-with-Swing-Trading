"""Event-driven weight backtest for the V4 core and T sleeves.

Signals are observed after a close. Each asset executes at its own next
tradable open; suspended assets retain their last valuation until trading
resumes. Formal runs require explicit full-snapshot targets.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .core import ContractError, estimate_trade_cost


INVESTABLE_RETURN_BASES = {"qfq_adjusted", "hfq_adjusted", "total_return"}
DIAGNOSTIC_RETURN_BASES = INVESTABLE_RETURN_BASES | {"price_index_proxy"}
TARGET_SEMANTICS = {"FULL_SNAPSHOT", "DELTA"}
TARGET_SCHEMA_VERSION = 2


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
    if (active_t & (typed["target_t_weight"] > typed["target_core_weight"] * support_ratio + 1e-9)).any():
        raise ContractError("target T weight exceeds supported core ratio")

    full = typed[typed["target_semantics"].eq("FULL_SNAPSHOT")]
    if full.empty:
        return
    grouped = full.groupby("signal_date")[["target_core_weight", "target_t_weight"]].sum()
    if (grouped["target_core_weight"] > float(config["portfolio"]["target_core_exposure"]) + 1e-9).any():
        raise ContractError("target core weights exceed portfolio core cap")
    if (grouped["target_t_weight"] > float(config["t_strategy"]["portfolio_t_weight_cap"]) + 1e-9).any():
        raise ContractError("target T weights exceed portfolio T cap")
    investable_cap = 1.0 - float(config["portfolio"]["minimum_cash_weight"])
    if (grouped.sum(axis=1) > investable_cap + 1e-9).any():
        raise ContractError("target weights violate minimum cash reserve")
    nonzero_assets = full.assign(
        _active=(full["target_core_weight"] + full["target_t_weight"]) > 1e-12
    ).groupby("signal_date")["_active"].sum()
    if (nonzero_assets > int(config["universe"]["maximum_assets"])).any():
        raise ContractError("target asset count exceeds configured maximum")
    if "sector" in full.columns and full["sector"].notna().all() and full["sector"].astype(str).str.strip().ne("").all():
        sector_weight = full.assign(
            _sector_weight=full["target_core_weight"] + full["target_t_weight"]
        ).groupby(["signal_date", "sector"])["_sector_weight"].sum()
        if (sector_weight > float(config["portfolio"]["max_sector_weight"]) + 1e-9).any():
            raise ContractError("target weights exceed sector cap")


def _promotion_blockers(prices: pd.DataFrame, targets: pd.DataFrame, config: dict[str, Any] | None) -> list[str]:
    blockers: list[str] = []
    bases = set(prices["return_basis"].astype(str).str.lower())
    if not bases.issubset(INVESTABLE_RETURN_BASES):
        blockers.append("non_investable_return_basis")
    if not targets["target_semantics"].eq("FULL_SNAPSHOT").all():
        blockers.append("non_full_snapshot_targets")
    if not {"is_tradable", "list_date", "delist_date"}.issubset(prices.columns):
        blockers.append("missing_asset_lifecycle_fields")
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


def _normalize_lifecycle(prices: pd.DataFrame, targets: pd.DataFrame, *, required: bool) -> pd.DataFrame:
    lifecycle_fields = {"is_tradable", "list_date", "delist_date"}
    if not lifecycle_fields.issubset(prices.columns):
        if required:
            raise ContractError(f"formal backtest prices missing lifecycle columns: {sorted(lifecycle_fields.difference(prices.columns))}")
        prices = prices.copy()
        prices["is_tradable"] = True
        prices["list_date"] = prices.groupby("asset")["date"].transform("min")
        prices["delist_date"] = pd.NaT
    prices["is_tradable"] = prices["is_tradable"].map(_is_true)
    prices["list_date"] = pd.to_datetime(prices["list_date"], errors="coerce").dt.normalize()
    prices["delist_date"] = pd.to_datetime(prices["delist_date"], errors="coerce").dt.normalize()
    if prices["list_date"].isna().any():
        raise ContractError("backtest prices contain invalid list_date values")
    for asset, history in prices.groupby("asset"):
        if history["list_date"].nunique() != 1 or history["delist_date"].dropna().nunique() > 1:
            raise ContractError(f"asset lifecycle changes within price history: {asset}")
        listed = pd.Timestamp(history["list_date"].iloc[0])
        delisted_values = history["delist_date"].dropna()
        delisted = pd.Timestamp(delisted_values.iloc[0]) if not delisted_values.empty else None
        if delisted is not None and delisted < listed:
            raise ContractError(f"asset delist_date predates list_date: {asset}")
        if (history["date"] < listed).any() or (delisted is not None and (history["date"] > delisted).any()):
            raise ContractError(f"price rows fall outside asset lifecycle: {asset}")
        asset_targets = targets[targets["asset"].eq(asset)]
        if (asset_targets["signal_date"] < listed).any():
            raise ContractError(f"target signal predates asset listing: {asset}")
        if delisted is not None and (asset_targets["signal_date"] >= delisted).any():
            raise ContractError(f"target signal is on or after asset delisting: {asset}")
    return prices


def _validate_full_snapshot_contract(targets: pd.DataFrame) -> None:
    full = targets[targets["target_semantics"].eq("FULL_SNAPSHOT")]
    if full.empty:
        return
    required = {"snapshot_asset_count"}
    missing = sorted(required.difference(full.columns))
    if missing:
        raise ContractError(f"FULL_SNAPSHOT targets missing snapshot contract columns: {missing}")
    declared_counts = pd.to_numeric(full["snapshot_asset_count"], errors="coerce")
    actual_counts = full.groupby("signal_date")["asset"].transform("count")
    if declared_counts.isna().any() or not declared_counts.eq(actual_counts).all():
        raise ContractError("FULL_SNAPSHOT target row count does not match snapshot_asset_count")
    if full.groupby("signal_date")["snapshot_asset_count"].nunique().gt(1).any():
        raise ContractError("FULL_SNAPSHOT must declare one snapshot_asset_count per signal date")

    active: set[str] = set()
    for _, rows in targets.groupby("signal_date", sort=True):
        semantics = str(rows["target_semantics"].iloc[0])
        present = set(rows["asset"].astype(str))
        if semantics == "FULL_SNAPSHOT":
            omitted = sorted(active.difference(present))
            if omitted:
                raise ContractError(
                    f"FULL_SNAPSHOT omits previously held assets; explicit zero targets are required: {omitted}"
                )
            active = set()
        for row in rows.itertuples(index=False):
            if float(row.target_core_weight) + float(row.target_t_weight) > 1e-12:
                active.add(str(row.asset))
            else:
                active.discard(str(row.asset))


def validate_backtest_inputs(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    formal_backtest: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    price_required = {"date", "asset", "asset_type", "open", "close", "return_basis", "price_basis"}
    target_required = {
        "signal_date",
        "asset",
        "target_core_weight",
        "target_t_weight",
        "target_semantics",
        "target_schema_version",
    }
    price_missing = sorted(price_required.difference(prices.columns))
    target_missing = sorted(target_required.difference(targets.columns))
    if price_missing:
        raise ContractError(f"backtest prices missing columns: {price_missing}")
    if target_missing:
        raise ContractError(f"backtest targets missing columns: {target_missing}")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["asset"] = p["asset"].astype(str)
    p["open"] = pd.to_numeric(p["open"], errors="coerce")
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    if p[["date", "open", "close"]].isna().any().any() or (p[["open", "close"]] <= 0).any().any():
        raise ContractError("backtest prices contain invalid dates or prices")
    if p.duplicated(["date", "asset"]).any():
        raise ContractError("backtest prices contain duplicate date-asset rows")
    bases = set(p["return_basis"].astype(str).str.lower())
    price_bases = set(p["price_basis"].astype(str).str.strip().str.lower())
    invalid = sorted(bases.difference(DIAGNOSTIC_RETURN_BASES))
    if invalid:
        raise ContractError(f"unsupported return_basis values: {invalid}")
    if formal_backtest and not bases.issubset(INVESTABLE_RETURN_BASES):
        raise ContractError("formal backtest prices require adjusted or total-return basis")
    invalid_price_bases = sorted(price_bases.difference(DIAGNOSTIC_RETURN_BASES))
    if invalid_price_bases:
        raise ContractError(f"unsupported price_basis values: {invalid_price_bases}")
    if bases != price_bases:
        raise ContractError("backtest return_basis and price_basis must agree")
    inconsistent_types = p.groupby("asset")["asset_type"].nunique()
    if (inconsistent_types > 1).any():
        raise ContractError("asset_type changes within price history")

    t = targets.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"], errors="coerce").dt.normalize()
    t["asset"] = t["asset"].astype(str)
    t["target_semantics"] = t["target_semantics"].astype(str).str.strip().str.upper()
    for col in ["target_core_weight", "target_t_weight"]:
        t[col] = pd.to_numeric(t[col], errors="coerce")
    if t[["signal_date", "target_core_weight", "target_t_weight"]].isna().any().any():
        raise ContractError("backtest targets contain invalid values")
    invalid_semantics = sorted(set(t["target_semantics"]).difference(TARGET_SEMANTICS))
    if invalid_semantics:
        raise ContractError(f"unsupported target semantics: {invalid_semantics}")
    if t.groupby("signal_date")["target_semantics"].nunique().gt(1).any():
        raise ContractError("one signal date cannot mix target semantics")
    schema_versions = pd.to_numeric(t["target_schema_version"], errors="coerce")
    if schema_versions.isna().any() or not schema_versions.eq(TARGET_SCHEMA_VERSION).all():
        raise ContractError(f"backtest targets require schema version {TARGET_SCHEMA_VERSION}")
    if formal_backtest and not t["target_semantics"].eq("FULL_SNAPSHOT").all():
        raise ContractError("formal backtest requires FULL_SNAPSHOT targets")
    if t.duplicated(["signal_date", "asset"]).any():
        raise ContractError("backtest targets contain duplicate signal-date asset rows")
    if (t[["target_core_weight", "target_t_weight"]] < -1e-12).any().any():
        raise ContractError("negative target weights are not allowed")
    if "available_date" in t.columns:
        t["available_date"] = pd.to_datetime(t["available_date"], errors="coerce").dt.normalize()
        if t["available_date"].isna().any():
            raise ContractError("backtest targets contain invalid available_date values")
        if (t["available_date"] > t["signal_date"]).any():
            raise ContractError("backtest targets contain look-ahead available_date values")
    missing_assets = sorted(set(t["asset"]).difference(p["asset"]))
    if missing_assets:
        raise ContractError(f"targets reference assets without prices: {missing_assets}")
    _validate_full_snapshot_contract(t)
    full = t[t["target_semantics"].eq("FULL_SNAPSHOT")]
    grouped = full.groupby("signal_date")[["target_core_weight", "target_t_weight"]].sum().sum(axis=1)
    if (grouped > 1.0 + 1e-9).any():
        raise ContractError("target weights exceed 100%")
    if config is not None:
        _validate_target_risk(p, t, config)
    p = _normalize_lifecycle(p, t, required=formal_backtest)
    input_contract_ready = not _promotion_blockers(p, t, config)
    return p.sort_values(["date", "asset"]), t.sort_values(["signal_date", "asset"]), input_contract_ready


def _asset_execution_schedule(
    prices: pd.DataFrame, targets: pd.DataFrame
) -> tuple[dict[pd.Timestamp, pd.DataFrame], int, pd.DataFrame]:
    tradable_dates = {
        asset: pd.DatetimeIndex(group.loc[group["is_tradable"], "date"].sort_values().unique())
        for asset, group in prices.groupby("asset")
    }
    rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        dates = tradable_dates[str(target["asset"])]
        location = int(np.searchsorted(dates.to_numpy(), np.datetime64(target["signal_date"]), side="right"))
        if location >= len(dates):
            pending_rows.append(
                {
                    **target.to_dict(),
                    "pending_reason": "no_asset_level_tradable_session_before_window_end",
                }
            )
            continue
        rows.append({**target.to_dict(), "execution_date": pd.Timestamp(dates[location])})
    scheduled_rows = pd.DataFrame(rows)
    if scheduled_rows.empty:
        return {}, 0, pd.DataFrame(pending_rows)
    scheduled_rows = scheduled_rows.sort_values(["execution_date", "asset", "signal_date"])
    superseded = int(scheduled_rows.duplicated(["execution_date", "asset"], keep="last").sum())
    scheduled_rows = scheduled_rows.drop_duplicates(["execution_date", "asset"], keep="last")
    return (
        {
            pd.Timestamp(date): frame.drop(columns="execution_date").reset_index(drop=True)
            for date, frame in scheduled_rows.groupby("execution_date", sort=True)
        },
        superseded,
        pd.DataFrame(pending_rows),
    )


def _buy_scale(
    cash: float,
    holdings_value: float,
    buy_specs: list[tuple[str, str, float]],
    asset_type: dict[str, str],
    config: dict[str, Any],
) -> float:
    if not buy_specs:
        return 1.0
    minimum_cash_weight = float(config["portfolio"]["minimum_cash_weight"])
    nav_before_buys = cash + holdings_value

    def feasible(scale: float) -> bool:
        notional = sum(raw_notional * scale for _, _, raw_notional in buy_specs)
        cost = sum(
            estimate_trade_cost(raw_notional * scale, "buy", asset_type[asset], config)["total_cost"]
            for _, asset, raw_notional in buy_specs
            if raw_notional * scale > 1e-12
        )
        post_cash = cash - notional - cost
        post_nav = nav_before_buys - cost
        return post_cash >= minimum_cash_weight * post_nav - 1e-9

    if feasible(1.0):
        return 1.0
    if not feasible(0.0):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def run_weight_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    initial_cash: float | None = None,
    *,
    mode: str = "formal",
    allow_delta_targets: bool = False,
) -> dict[str, Any]:
    """Run a multi-asset core/T backtest with asset-level next-open execution."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"formal", "diagnostic"}:
        raise ValueError("backtest mode must be formal or diagnostic")
    p, t, input_contract_ready = validate_backtest_inputs(
        prices, targets, config, formal_backtest=normalized_mode == "formal"
    )
    if t["target_semantics"].eq("DELTA").any() and not (
        normalized_mode == "diagnostic" and allow_delta_targets
    ):
        raise ContractError("DELTA targets require diagnostic mode with allow_delta_targets=True")
    if initial_cash is None:
        initial_cash = float(config["account"]["initial_cash_cny"])
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    open_wide = p.pivot(index="date", columns="asset", values="open").reindex(dates)
    close_wide = p.pivot(index="date", columns="asset", values="close").reindex(dates)
    tradable_wide = p.pivot(index="date", columns="asset", values="is_tradable").reindex(dates).fillna(False)
    asset_type = p.drop_duplicates("asset").set_index("asset")["asset_type"].astype(str).str.lower().to_dict()
    lifecycle = p.groupby("asset").agg(list_date=("list_date", "first"), delist_date=("delist_date", "first"))
    scheduled, superseded_signals, pending_targets = _asset_execution_schedule(p, t)

    assets = sorted(set(p["asset"]) | set(t["asset"]))
    core = pd.Series(0.0, index=assets)
    sleeve_t = pd.Series(0.0, index=assets)
    cash = float(initial_cash)
    previous_close = pd.Series(np.nan, index=assets, dtype=float)
    nav_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for date in dates:
        tradable_today = tradable_wide.loc[date].reindex(assets).fillna(False).astype(bool)
        raw_open = open_wide.loc[date].reindex(assets).where(tradable_today)
        raw_close = close_wide.loc[date].reindex(assets).where(tradable_today)
        open_px = raw_open.combine_first(previous_close)
        held_overnight = (core + sleeve_t) > 1e-10
        missing_held = held_overnight & previous_close.isna()
        if missing_held.any():
            raise ContractError(f"held assets have no prior valuation: {sorted(missing_held[missing_held].index)}")
        overnight = (open_px / previous_close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        core *= 1.0 + overnight
        sleeve_t *= 1.0 + overnight

        day_cost = 0.0
        day_notional = 0.0
        post_trade_cash_weight = float("nan")
        if date in scheduled:
            plan = scheduled[date].set_index("asset")
            executable_open = raw_open.reindex(plan.index)
            if executable_open.isna().any():
                missing_open = sorted(executable_open[executable_open.isna()].index)
                raise ContractError(f"asset-level schedule has no executable open price: {missing_open}")
            nav_open = cash + float(core.sum()) + float(sleeve_t.sum())
            desired_core = plan["target_core_weight"] * nav_open
            desired_t = plan["target_t_weight"] * nav_open
            core_delta = desired_core - core.reindex(plan.index)
            t_delta = desired_t - sleeve_t.reindex(plan.index)

            for sleeve_name, delta in (("core", core_delta), ("t", t_delta)):
                for asset, amount in delta[delta < -1e-10].items():
                    notional = abs(float(amount))
                    cost = estimate_trade_cost(notional, "sell", asset_type[asset], config)["total_cost"]
                    cash += notional - cost
                    if sleeve_name == "core":
                        core.loc[asset] -= notional
                    else:
                        sleeve_t.loc[asset] -= notional
                    day_cost += cost
                    day_notional += notional
                    trade_rows.append(
                        {
                            "signal_date": plan.loc[asset, "signal_date"],
                            "execution_date": date,
                            "asset": asset,
                            "sleeve": sleeve_name,
                            "side": "sell",
                            "notional": notional,
                            "cost": cost,
                            "target_semantics": plan.loc[asset, "target_semantics"],
                        }
                    )
            core = core.clip(lower=0.0)
            sleeve_t = sleeve_t.clip(lower=0.0)

            buy_specs: list[tuple[str, str, float]] = []
            for sleeve_name, delta in (("core", core_delta), ("t", t_delta)):
                for asset, amount in delta[delta > 1e-10].items():
                    buy_specs.append((sleeve_name, asset, float(amount)))
            scale = _buy_scale(cash, float(core.sum() + sleeve_t.sum()), buy_specs, asset_type, config)
            for sleeve_name, asset, raw_notional in buy_specs:
                notional = raw_notional * scale
                if notional <= 1e-10:
                    continue
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
                        "signal_date": plan.loc[asset, "signal_date"],
                        "execution_date": date,
                        "asset": asset,
                        "sleeve": sleeve_name,
                        "side": "buy",
                        "notional": notional,
                        "cost": cost,
                        "target_semantics": plan.loc[asset, "target_semantics"],
                    }
                )
            post_trade_nav = cash + float(core.sum()) + float(sleeve_t.sum())
            post_trade_cash_weight = cash / post_trade_nav
            if post_trade_cash_weight + 1e-9 < float(config["portfolio"]["minimum_cash_weight"]):
                raise AssertionError("backtest trade breached the minimum cash reserve after costs")
            active_count = int(((core + sleeve_t) > 1e-10).sum())
            if active_count > int(config["universe"]["maximum_assets"]):
                raise ContractError("post-trade holding count exceeds configured maximum including existing holdings")

        close_px = raw_close.combine_first(open_px)
        held_intraday = (core + sleeve_t) > 1e-10
        missing_held = held_intraday & close_px.isna()
        if missing_held.any():
            raise ContractError(f"held assets have no current or carried valuation: {sorted(missing_held[missing_held].index)}")
        intraday = (close_px / open_px - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        core *= 1.0 + intraday
        sleeve_t *= 1.0 + intraday
        for asset in assets:
            delist_date = lifecycle.loc[asset, "delist_date"]
            if pd.notna(delist_date) and date >= pd.Timestamp(delist_date) and core[asset] + sleeve_t[asset] > 1e-10:
                raise ContractError(f"held asset reaches delisting without an executable exit: {asset}")
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
                "cash_weight": cash / nav,
                "post_trade_cash_weight": post_trade_cash_weight,
                "holding_count": int(((core + sleeve_t) > 1e-10).sum()),
                "trade_notional": day_notional,
                "trading_cost": day_cost,
                "one_way_turnover": 0.5 * day_notional / nav,
            }
        )
        previous_close = raw_close.combine_first(previous_close)

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
    metrics["target_semantics"] = ",".join(sorted(set(t["target_semantics"])))
    metrics["backtest_mode"] = normalized_mode
    metrics["signal_execution"] = "close_signal_asset_level_next_tradable_open"
    metrics["superseded_pending_signals"] = superseded_signals
    metrics["pending_unexecuted_signals"] = len(pending_targets)
    metrics["overlapping_forward_returns_compounded"] = False
    return {
        "nav": nav,
        "trades": trades,
        "pending_targets": pending_targets,
        "metrics": metrics,
    }


def summarize_backtest(nav: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    if nav.empty:
        raise ValueError("nav is empty")
    values = nav["nav"].astype(float)
    returns = pd.concat(
        [pd.Series([values.iloc[0] / initial_cash - 1.0]), values.pct_change().iloc[1:]], ignore_index=True
    )
    periods = int(len(returns))
    total_return = float(values.iloc[-1] / initial_cash - 1.0)
    annual_return = float((values.iloc[-1] / initial_cash) ** (252.0 / periods) - 1.0) if periods > 0 else np.nan
    annual_vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if periods > 1 else np.nan
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
        if periods > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
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
