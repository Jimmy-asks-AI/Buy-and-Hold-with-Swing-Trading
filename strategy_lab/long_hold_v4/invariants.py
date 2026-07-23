"""Reusable cash-only portfolio invariants for replay and randomized tests."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .core import ContractError
from .execution import normalize_account
from .order_envelope import normalize_order_state_book


def validate_paper_account_invariants(
    account: dict[str, Any],
    valuation_prices: Mapping[str, float],
    config: dict[str, Any],
    *,
    order_state_book: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Fail closed when a paper account violates a Long Hold V4 hard limit."""
    if bool(config["account"].get("allow_short")) or bool(config["account"].get("allow_margin")):
        raise ContractError("cash-only invariant rejects short selling or margin")
    state = normalize_account(account, config)
    cash = float(state["cash_cny"])
    if cash < -1e-9:
        raise ContractError("cash invariant failed: cash is negative")

    market_value = 0.0
    core_value = 0.0
    t_value = 0.0
    sector_values: dict[str, float] = {}
    active_holdings = 0
    tolerance = 1e-8
    for holding in state["holdings"]:
        asset = str(holding["asset"])
        if asset not in valuation_prices:
            raise ContractError(f"invariant price is missing: {asset}")
        price = float(valuation_prices[asset])
        if not math.isfinite(price) or price <= 0:
            raise ContractError(f"invariant price must be finite and positive: {asset}")
        core_shares = holding["core_shares"]
        t_shares = holding["t_shares"]
        if (
            not isinstance(core_shares, int)
            or isinstance(core_shares, bool)
            or not isinstance(t_shares, int)
            or isinstance(t_shares, bool)
            or core_shares < 0
            or t_shares < 0
        ):
            raise ContractError("share invariant requires non-negative integers")
        if t_shares > 0:
            reference = float(holding["full_target_shares_reference"])
            if core_shares <= 0 or reference <= 0:
                raise ContractError("T sleeve invariant requires an existing core holding")
            if core_shares / reference + tolerance < float(config["t_strategy"]["core_fraction_required"]):
                raise ContractError("T sleeve invariant is under-supported by core shares")
            if t_shares > reference * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"]) + tolerance:
                raise ContractError("per-asset T sleeve invariant exceeds its cap")
        value = (core_shares + t_shares) * price
        if value > 0:
            active_holdings += 1
        market_value += value
        core_value += core_shares * price
        t_value += t_shares * price
        sector = str(holding["sector"])
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    nav = cash + market_value
    if not math.isfinite(nav) or nav <= 0:
        raise ContractError("account NAV invariant must be finite and positive")
    if cash / nav + tolerance < float(config["portfolio"]["minimum_cash_weight"]):
        raise ContractError("minimum cash invariant failed")
    if active_holdings > int(config["universe"]["maximum_assets"]):
        raise ContractError("holding count invariant exceeds maximum assets")
    if core_value / nav > float(config["portfolio"]["target_core_exposure"]) + tolerance:
        raise ContractError("aggregate core invariant exceeds its cap")
    if t_value / nav > float(config["t_strategy"]["portfolio_t_weight_cap"]) + tolerance:
        raise ContractError("aggregate T invariant exceeds its cap")
    if any(value / nav > float(config["portfolio"]["max_sector_weight"]) + tolerance for value in sector_values.values()):
        raise ContractError("sector invariant exceeds its cap")
    for holding in state["holdings"]:
        value = (int(holding["core_shares"]) + int(holding["t_shares"])) * float(
            valuation_prices[str(holding["asset"])]
        )
        cap_key = f"max_single_{holding['asset_type']}_weight"
        if value / nav > float(config["portfolio"][cap_key]) + tolerance:
            raise ContractError("single-asset invariant exceeds its cap")

    history = state["fill_history"]
    processed = {str(item["fill_id"]) for item in state["processed_fills"]}
    history_ids = [str(item["fill_id"]) for item in history]
    if len(history_ids) != len(set(history_ids)) or set(history_ids) != processed:
        raise ContractError("ledger invariant does not match processed fills")
    for index, row in enumerate(history):
        before = float(row["cash_before_cny"])
        after = float(row["cash_after_cny"])
        if index and not math.isclose(before, float(history[index - 1]["cash_after_cny"]), abs_tol=1e-8):
            raise ContractError("ledger cash chain invariant failed")
        direction = -1.0 if str(row["side"]) == "buy" else 1.0
        expected_after = before + direction * float(row["notional"])
        expected_after -= float(row["total_cost_cny"])
        if not math.isclose(after, expected_after, abs_tol=1e-8):
            raise ContractError("ledger cash arithmetic invariant failed")
    if history and not state["event_history"] and not math.isclose(
        cash, float(history[-1]["cash_after_cny"]), abs_tol=1e-8
    ):
        raise ContractError("account cash does not match the terminal ledger balance")

    if order_state_book is not None:
        book = normalize_order_state_book(order_state_book)
        if int(book["account_version"]) != int(state["state_version"]) or str(
            book["account_state_sha256"]
        ) != str(state["state_sha256"]):
            raise ContractError("order state invariant is not bound to the account")

    return {
        "nav_cny": nav,
        "cash_cny": cash,
        "market_value_cny": market_value,
        "core_value_cny": core_value,
        "t_value_cny": t_value,
        "holding_count": active_holdings,
    }
