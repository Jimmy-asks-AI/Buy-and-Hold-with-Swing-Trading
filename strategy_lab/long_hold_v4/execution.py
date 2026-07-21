"""Auditable paper/broker fill accounting for Long Hold V4 core and T sleeves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ContractError, estimate_trade_cost, load_config


ROOT = Path(__file__).resolve().parents[2]
ACCOUNT_SCHEMA_VERSION = 2
FILL_REQUIRED = {
    "fill_id",
    "fill_date",
    "order_id",
    "asset",
    "name",
    "asset_type",
    "sector",
    "sleeve",
    "side",
    "shares",
    "price",
    "fee_mode",
    "manual_approval",
    "manual_reason",
    "risk_override",
}
ACTUAL_FEE_COLUMNS = ["commission_cny", "stamp_duty_cny", "transfer_fee_cny", "other_fees_cny"]
LEDGER_COLUMNS = [
    "fill_id",
    "fill_sha256",
    "order_id",
    "fill_date",
    "asset",
    "name",
    "asset_type",
    "sector",
    "sleeve",
    "side",
    "shares",
    "price",
    "notional",
    "commission_cny",
    "stamp_duty_cny",
    "transfer_fee_cny",
    "slippage_cny",
    "other_fees_cny",
    "total_cost_cny",
    "realized_pnl_cny",
    "cash_before_cny",
    "cash_after_cny",
    "shares_before",
    "shares_after",
    "fee_mode",
    "manual_approval",
    "manual_reason",
    "status",
]
EXECUTABLE_ORDER_STATUS = "RESEARCH_INTENT_REPRICE_NEXT_OPEN"


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "approved"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ContractError(f"{field} must be a boolean")


def _finite_number(value: Any, field: str) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        raise ContractError(f"{field} must be finite")
    return float(number)


def _nonnegative_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number < 0:
        raise ContractError(f"{field} must be non-negative")
    return number


def _whole_nonnegative(value: Any, field: str) -> int:
    number = _nonnegative_number(value, field)
    if not number.is_integer():
        raise ContractError(f"{field} must be a whole number")
    return int(number)


def _shares(value: Any, field: str = "shares") -> int:
    number = _whole_nonnegative(value, field)
    if number <= 0:
        raise ContractError(f"{field} must be a positive whole number")
    return number


def _date_or_none(value: Any, field: str) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ContractError(f"{field} must be a valid date")
    return str(pd.Timestamp(timestamp).date())


def _asset_code(value: Any) -> str:
    asset = str(value).strip().zfill(6)
    if not re.fullmatch(r"\d{6}", asset):
        raise ContractError("asset must be a six-digit A-share or ETF code")
    return asset


def _normalized_holding(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("each account holding must be an object")
    core_shares = _whole_nonnegative(raw.get("core_shares", 0), "core_shares")
    t_shares = _whole_nonnegative(raw.get("t_shares", 0), "t_shares")
    asset_type = str(raw.get("asset_type", "")).lower()
    if core_shares + t_shares > 0 and asset_type not in {"stock", "etf"}:
        raise ContractError("non-empty holding requires stock or etf asset_type")
    name = str(raw.get("name", "")).strip()
    sector = str(raw.get("sector", "")).strip().lower()
    if core_shares + t_shares > 0 and (not name or not sector):
        raise ContractError("non-empty holding requires name and sector")
    if t_shares > 0 and core_shares <= 0:
        raise ContractError("T holding requires an existing core holding")
    core_cost = _nonnegative_number(raw.get("core_average_cost_cny", 0), "core_average_cost_cny")
    t_cost = _nonnegative_number(raw.get("t_average_cost_cny", 0), "t_average_cost_cny")
    core_open_date = _date_or_none(raw.get("core_open_date"), "core_open_date")
    t_open_date = _date_or_none(raw.get("t_open_date"), "t_open_date")
    if core_shares > 0 and (core_cost <= 0 or core_open_date is None):
        raise ContractError("non-empty core holding requires positive cost and open date")
    if t_shares > 0 and (t_cost <= 0 or t_open_date is None):
        raise ContractError("non-empty T holding requires positive cost and open date")
    reference = _nonnegative_number(raw.get("full_target_shares_reference", 0), "full_target_shares_reference")
    if core_shares > 0 and reference <= 0:
        legacy_fraction = _nonnegative_number(raw.get("core_fraction_of_full_target", 0), "core_fraction_of_full_target")
        if legacy_fraction <= 0:
            raise ContractError("held core requires full_target_shares_reference")
        reference = core_shares / legacy_fraction
    if reference + 1e-9 < core_shares:
        raise ContractError("full_target_shares_reference cannot be below core shares")
    return {
        "asset": _asset_code(raw.get("asset", "")),
        "name": name,
        "asset_type": asset_type,
        "sector": sector,
        "core_shares": core_shares,
        "core_average_cost_cny": core_cost if core_shares else 0.0,
        "core_open_date": core_open_date if core_shares else None,
        "t_shares": t_shares,
        "t_average_cost_cny": t_cost if t_shares else 0.0,
        "t_open_date": t_open_date if t_shares else None,
        "full_target_shares_reference": float(reference),
        "realized_pnl_cny": _finite_number(raw.get("realized_pnl_cny", 0), "realized_pnl_cny"),
        "cumulative_dividend_net_cny": _finite_number(
            raw.get("cumulative_dividend_net_cny", 0), "cumulative_dividend_net_cny"
        ),
    }


def normalize_account(account: dict[str, Any], config: dict[str, Any], as_of: str | pd.Timestamp | None = None) -> dict[str, Any]:
    if not isinstance(account, dict):
        raise ContractError("account must be a JSON object")
    cash = _nonnegative_number(account.get("cash_cny", -1), "cash_cny")
    holdings_raw = account.get("holdings", [])
    if not isinstance(holdings_raw, list):
        raise ContractError("account holdings must be a list")
    holdings = [_normalized_holding(item) for item in holdings_raw]
    assets = [item["asset"] for item in holdings]
    if len(assets) != len(set(assets)):
        raise ContractError("account holdings must be unique by asset")
    processed = account.get("processed_fills", [])
    history = account.get("fill_history", [])
    processed_events = account.get("processed_events", [])
    event_history = account.get("event_history", [])
    nav_history = account.get("nav_history", [])
    if not all(isinstance(value, list) for value in [processed, history, processed_events, event_history, nav_history]):
        raise ContractError("account fill, event, and NAV histories must be lists")
    as_of_value = as_of if as_of is not None else account.get("as_of_date")
    as_of_date = _date_or_none(as_of_value, "as_of_date")
    if as_of_date is None:
        raise ContractError("account requires as_of_date")
    base_currency = str(account.get("base_currency", config["account"]["base_currency"])).upper()
    if base_currency != str(config["account"]["base_currency"]).upper():
        raise ContractError("account base currency mismatches config")
    normalized = {
        "schema_version": ACCOUNT_SCHEMA_VERSION,
        "account_id": str(account.get("account_id", "long_hold_v4_primary")),
        "base_currency": base_currency,
        "as_of_date": as_of_date,
        "cash_cny": cash,
        "holdings": holdings,
        "realized_pnl_cny": _finite_number(account.get("realized_pnl_cny", 0.0), "realized_pnl_cny"),
        "gross_dividend_cny": _nonnegative_number(account.get("gross_dividend_cny", 0.0), "gross_dividend_cny"),
        "dividend_tax_cny": _nonnegative_number(account.get("dividend_tax_cny", 0.0), "dividend_tax_cny"),
        "processed_fills": copy.deepcopy(processed),
        "fill_history": copy.deepcopy(history),
        "processed_events": copy.deepcopy(processed_events),
        "event_history": copy.deepcopy(event_history),
        "nav_history": copy.deepcopy(nav_history),
    }
    normalized["net_dividend_cny"] = normalized["gross_dividend_cny"] - normalized["dividend_tax_cny"]
    if any(
        pd.Timestamp(item[date_field]) > pd.Timestamp(as_of_date)
        for item in holdings
        for date_field in ("core_open_date", "t_open_date")
        if item[date_field] is not None
    ):
        raise ContractError("holding open date cannot be after account as_of_date")
    if any(not isinstance(item, dict) for item in normalized["processed_fills"]):
        raise ContractError("processed fill entries must be objects")
    fill_ids = [str(item.get("fill_id", "")).strip() for item in normalized["processed_fills"]]
    if any(not fill_id for fill_id in fill_ids) or len(fill_ids) != len(set(fill_ids)):
        raise ContractError("processed fill ids must be non-empty and unique")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) for item in normalized["processed_fills"]):
        raise ContractError("processed fill hashes must be lowercase SHA-256 values")
    if any(not isinstance(item, dict) for item in normalized["fill_history"]):
        raise ContractError("fill history entries must be objects")
    history_ids = [str(item.get("fill_id", "")).strip() for item in normalized["fill_history"]]
    if len(history_ids) != len(set(history_ids)) or not set(history_ids).issubset(fill_ids):
        raise ContractError("fill history ids must be unique and present in processed_fills")
    fill_hashes = {str(item["fill_id"]): str(item["sha256"]) for item in normalized["processed_fills"]}
    if any(str(item.get("fill_sha256", "")) != fill_hashes.get(str(item.get("fill_id", ""))) for item in normalized["fill_history"]):
        raise ContractError("fill history hash mismatches processed_fills")
    if any(not isinstance(item, dict) for item in normalized["processed_events"]):
        raise ContractError("processed event entries must be objects")
    event_ids = [str(item.get("event_id", "")).strip() for item in normalized["processed_events"]]
    if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ContractError("processed event ids must be non-empty and unique")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) for item in normalized["processed_events"]):
        raise ContractError("processed event hashes must be lowercase SHA-256 values")
    if any(not isinstance(item, dict) for item in normalized["event_history"]):
        raise ContractError("event history entries must be objects")
    ledger_event_ids = [str(item.get("event_id", "")).strip() for item in normalized["event_history"]]
    if len(ledger_event_ids) != len(set(ledger_event_ids)) or not set(ledger_event_ids).issubset(event_ids):
        raise ContractError("event history ids must be unique and present in processed_events")
    event_hashes = {str(item["event_id"]): str(item["sha256"]) for item in normalized["processed_events"]}
    if any(
        str(item.get("event_sha256", "")) != event_hashes.get(str(item.get("event_id", "")))
        for item in normalized["event_history"]
    ):
        raise ContractError("event history hash mismatches processed_events")
    normalized_nav: list[dict[str, Any]] = []
    for item in normalized["nav_history"]:
        if not isinstance(item, dict):
            raise ContractError("NAV history entries must be objects")
        date = _date_or_none(item.get("date"), "nav_history.date")
        if date is None or pd.Timestamp(date) > pd.Timestamp(as_of_date):
            raise ContractError("NAV history date must not exceed account as_of_date")
        nav = _nonnegative_number(item.get("nav_cny", -1), "nav_history.nav_cny")
        cash_value = _nonnegative_number(item.get("cash_cny", -1), "nav_history.cash_cny")
        market_value = _nonnegative_number(item.get("market_value_cny", -1), "nav_history.market_value_cny")
        peak = _nonnegative_number(item.get("peak_nav_cny", -1), "nav_history.peak_nav_cny")
        drawdown = _finite_number(item.get("drawdown", 0), "nav_history.drawdown")
        if nav <= 0 or peak + 1e-6 < nav or not math.isclose(nav, cash_value + market_value, abs_tol=0.01):
            raise ContractError("NAV history values are internally inconsistent")
        if drawdown > 1e-9 or not math.isclose(drawdown, nav / peak - 1.0, abs_tol=1e-8):
            raise ContractError("NAV history drawdown is inconsistent")
        normalized_nav.append(
            {
                "date": date,
                "nav_cny": nav,
                "cash_cny": cash_value,
                "market_value_cny": market_value,
                "peak_nav_cny": peak,
                "drawdown": drawdown,
                "risk_state": str(item.get("risk_state", "")),
            }
        )
    nav_dates = [item["date"] for item in normalized_nav]
    if len(nav_dates) != len(set(nav_dates)):
        raise ContractError("NAV history must contain at most one row per date")
    normalized["nav_history"] = sorted(normalized_nav, key=lambda item: item["date"])
    running_peak = 0.0
    review_trigger = float(config["portfolio"]["drawdown_review_trigger"])
    brake_trigger = float(config["portfolio"]["drawdown_risk_reduction_trigger"])
    for item in normalized["nav_history"]:
        running_peak = max(running_peak, item["nav_cny"])
        expected_state = "BRAKE" if item["drawdown"] <= brake_trigger else "REVIEW" if item["drawdown"] <= review_trigger else "NORMAL"
        if not math.isclose(item["peak_nav_cny"], running_peak, abs_tol=0.01):
            raise ContractError("NAV history peak is not monotonic")
        if item["risk_state"] != expected_state:
            raise ContractError("NAV history risk_state is inconsistent")
    return normalized


def _fill_hash(row: pd.Series) -> str:
    payload = {
        key: (None if pd.isna(value) else str(value))
        for key, value in sorted(row.to_dict().items())
        if key not in {"status", "calculated_total_cost_cny"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prepare_fills(fills: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(FILL_REQUIRED.difference(fills.columns))
    if missing:
        raise ContractError(f"fills missing columns: {missing}")
    out = fills.copy()
    for col in ["fill_id", "order_id", "asset", "name", "asset_type", "sector", "sleeve", "side", "fee_mode"]:
        out[col] = out[col].astype(str).str.strip()
    out["asset"] = out["asset"].map(_asset_code)
    out["fill_date"] = pd.to_datetime(out["fill_date"], errors="coerce")
    if out["fill_date"].isna().any() or out["fill_id"].eq("").any() or out["order_id"].eq("").any():
        raise ContractError("fills contain invalid dates or empty fill/order ids")
    if out["fill_id"].duplicated().any():
        raise ContractError("fill batch contains duplicate fill ids")
    for col in ["asset_type", "sector", "sleeve", "side", "fee_mode"]:
        out[col] = out[col].str.lower()
    if out["name"].eq("").any() or out["sector"].eq("").any():
        raise ContractError("fill name and sector cannot be empty")
    if not out["asset_type"].isin({"stock", "etf"}).all():
        raise ContractError("fill asset_type must be stock or etf")
    if not out["sleeve"].isin({"core", "t"}).all() or not out["side"].isin({"buy", "sell"}).all():
        raise ContractError("fill sleeve or side is invalid")
    if not out["fee_mode"].isin({"model", "actual"}).all():
        raise ContractError("fee_mode must be model or actual")
    out["manual_approval"] = out["manual_approval"].map(lambda value: _boolean(value, "manual_approval"))
    out["risk_override"] = out["risk_override"].map(lambda value: _boolean(value, "risk_override"))
    out["manual_reason"] = out["manual_reason"].fillna("").astype(str).str.strip()
    out["shares"] = out["shares"].map(_shares)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if out["price"].isna().any() or (~out["price"].map(math.isfinite)).any() or (out["price"] <= 0).any():
        raise ContractError("fill price must be finite and positive")
    return out.sort_values(["fill_date", "fill_id"]).reset_index(drop=True)


def _approved_order(row: pd.Series, approved_orders: pd.DataFrame | None) -> pd.Series | None:
    if bool(row["manual_approval"]):
        if str(row["side"]) != "sell":
            raise ContractError("manual approval cannot bypass a buy order")
        if not str(row["manual_reason"]).strip():
            raise ContractError("manual sell requires manual_reason")
        if str(row["sleeve"]) == "t" and not bool(row["risk_override"]):
            raise ContractError("manual T sell requires risk_override")
        return None
    if approved_orders is None or approved_orders.empty:
        raise ContractError(f"fill {row['fill_id']} has no approved order source")
    required = {
        "order_id",
        "signal_date",
        "valid_through_date",
        "asset",
        "asset_type",
        "sector",
        "sleeve",
        "side",
        "shares",
        "status",
        "risk_override_allowed",
    }
    missing = sorted(required.difference(approved_orders.columns))
    if missing:
        raise ContractError(f"approved orders missing columns: {missing}")
    matches = approved_orders[approved_orders["order_id"].astype(str) == str(row["order_id"])]
    if len(matches) != 1:
        raise ContractError(f"fill {row['fill_id']} must match exactly one approved order")
    order = matches.iloc[0]
    if str(order["status"]) != EXECUTABLE_ORDER_STATUS:
        raise ContractError(f"fill {row['fill_id']} matched a non-executable order")
    if bool(row["risk_override"]) and not _boolean(order["risk_override_allowed"], "risk_override_allowed"):
        raise ContractError("fill risk_override is not authorized by the approved order")
    for field in ["asset", "asset_type", "sector", "sleeve", "side"]:
        order_value = _asset_code(order[field]) if field == "asset" else str(order[field]).strip().lower()
        if order_value != str(row[field]).strip().lower():
            raise ContractError(f"fill {row['fill_id']} mismatches approved order field {field}")
    if _shares(order["shares"], "approved order shares") < int(row["shares"]):
        raise ContractError(f"fill {row['fill_id']} exceeds approved shares")
    signal_date = pd.to_datetime(order["signal_date"], errors="coerce")
    valid_through = pd.to_datetime(order["valid_through_date"], errors="coerce")
    if pd.isna(signal_date) or pd.isna(valid_through) or pd.Timestamp(valid_through) <= pd.Timestamp(signal_date):
        raise ContractError("approved order has invalid signal or validity dates")
    signal_date = pd.Timestamp(signal_date).normalize()
    valid_through = pd.Timestamp(valid_through).normalize()
    if pd.Timestamp(row["fill_date"]).normalize() <= signal_date:
        raise ContractError("close signal fill must occur on a later date")
    if pd.Timestamp(row["fill_date"]).normalize() > valid_through:
        raise ContractError("fill occurs after approved order expiry")
    return order


def _costs(row: pd.Series, config: dict[str, Any]) -> dict[str, float]:
    notional = float(row["shares"] * row["price"])
    if row["fee_mode"] == "model":
        return estimate_trade_cost(notional, row["side"], row["asset_type"], config)
    missing = [column for column in ACTUAL_FEE_COLUMNS if column not in row.index or pd.isna(row[column])]
    if missing:
        raise ContractError(f"actual fee fill missing fields: {missing}")
    commission, stamp, transfer, other = (
        _nonnegative_number(row[column], column) for column in ACTUAL_FEE_COLUMNS
    )
    total = commission + stamp + transfer + other
    return {
        "commission": commission,
        "stamp_duty": stamp,
        "transfer_fee": transfer,
        "slippage": 0.0,
        "other_fees": other,
        "total_cost": total,
        "effective_rate": total / notional,
    }


def apply_fills(
    account: dict[str, Any],
    fills: pd.DataFrame,
    config: dict[str, Any],
    approved_orders: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    state = normalize_account(account, config)
    prepared = _prepare_fills(fills)
    processed = {str(item["fill_id"]): str(item["sha256"]) for item in state["processed_fills"]}
    batch_order_fills: dict[str, int] = {}
    result_rows: list[dict[str, Any]] = []

    for _, row in prepared.iterrows():
        fill_id = str(row["fill_id"])
        digest = _fill_hash(row)
        if fill_id in processed:
            if processed[fill_id] != digest:
                raise ContractError(f"fill_id payload changed after processing: {fill_id}")
            result_rows.append({"fill_id": fill_id, "status": "duplicate_ignored"})
            continue
        order = _approved_order(row, approved_orders)
        if pd.Timestamp(row["fill_date"]).normalize() < pd.Timestamp(state["as_of_date"]):
            raise ContractError(f"fill {fill_id} predates the account state")
        if order is not None:
            order_id = str(row["order_id"])
            already_filled = sum(
                int(item["shares"])
                for item in state["fill_history"]
                if str(item.get("order_id", "")) == order_id and item.get("status") == "applied"
            )
            batch_filled = batch_order_fills.get(order_id, 0)
            if already_filled + batch_filled + int(row["shares"]) > int(float(order["shares"])):
                raise ContractError(f"aggregate fills exceed approved order shares: {order_id}")
            batch_order_fills[order_id] = batch_filled + int(row["shares"])

        asset = str(row["asset"])
        holding = next((item for item in state["holdings"] if item["asset"] == asset), None)
        if holding is None:
            holding = _normalized_holding(
                {
                    "asset": asset,
                    "name": "" if pd.isna(row.get("name", "")) else str(row.get("name", "")).strip(),
                    "asset_type": row["asset_type"],
                    "sector": row["sector"],
                }
            )
            state["holdings"].append(holding)
        if holding["asset_type"] != row["asset_type"] or holding["sector"] != str(row["sector"]):
            raise ContractError("fill metadata mismatches existing holding")

        sleeve = str(row["sleeve"])
        side = str(row["side"])
        shares = int(row["shares"])
        price = float(row["price"])
        share_key = f"{sleeve}_shares"
        cost_key = f"{sleeve}_average_cost_cny"
        before_shares = int(holding[share_key])
        before_cash = float(state["cash_cny"])
        costs = _costs(row, config)
        notional = shares * price
        realized = 0.0

        if sleeve == "t" and side == "buy":
            reference = float(holding["full_target_shares_reference"])
            core_fraction = holding["core_shares"] / reference if reference > 0 else 0.0
            required_core = float(config["t_strategy"]["core_fraction_required"])
            if holding["core_shares"] <= 0 or core_fraction + 1e-9 < required_core:
                raise ContractError("T buy requires the configured core fraction at fill time")
            if order is not None and _nonnegative_number(
                order.get("core_fraction_at_signal", float("nan")), "core_fraction_at_signal"
            ) + 1e-9 < required_core:
                raise ContractError("T buy approved order lacks required core fraction")
            t_limit = reference * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"])
            if before_shares + shares > t_limit + 1e-9:
                raise ContractError("T buy exceeds the sleeve share cap")
        if side == "buy":
            cash_need = notional + float(costs["total_cost"])
            if cash_need > state["cash_cny"] + 1e-9:
                raise ContractError(f"insufficient cash for fill {fill_id}")
            old_cost = before_shares * float(holding[cost_key])
            holding[share_key] = before_shares + shares
            holding[cost_key] = (old_cost + cash_need) / holding[share_key]
            state["cash_cny"] -= cash_need
            if before_shares == 0:
                holding[f"{sleeve}_open_date"] = str(pd.Timestamp(row["fill_date"]).date())
            if sleeve == "core":
                reference = order.get("full_target_shares_reference") if order is not None else row.get(
                    "full_target_shares_reference"
                )
                reference = _nonnegative_number(reference, "full_target_shares_reference")
                if reference + 1e-9 < holding["core_shares"]:
                    raise ContractError("core buy full target reference is below resulting core shares")
                holding["full_target_shares_reference"] = float(reference)
        else:
            if shares > before_shares:
                raise ContractError(f"fill {fill_id} attempts to sell more {sleeve} shares than held")
            if sleeve == "core" and not bool(row["manual_approval"]):
                raise ContractError("core sell requires explicit manual approval")
            if sleeve == "core" and holding["t_shares"] > 0:
                remaining_fraction = (before_shares - shares) / float(holding["full_target_shares_reference"])
                if remaining_fraction + 1e-9 < float(config["t_strategy"]["core_fraction_required"]):
                    raise ContractError("core sell would orphan or under-support the T sleeve")
            if sleeve == "t":
                open_date = pd.Timestamp(holding["t_open_date"]).normalize() if holding["t_open_date"] else None
                settlement = str(config["t_strategy"].get(f"{row['asset_type']}_settlement", "T+1")).upper()
                if (
                    open_date is None
                    or (
                        settlement == "T+1"
                        and not bool(config["t_strategy"].get("same_day_t_allowed", False))
                        and pd.Timestamp(row["fill_date"]).normalize() <= open_date
                    )
                ):
                    raise ContractError(f"{row['asset_type']} T sell violates settlement rules")
                session_value = order.get("t_holding_sessions", 0) if order is not None else row.get(
                    "t_holding_sessions", 0
                )
                sessions = _whole_nonnegative(session_value, "t_holding_sessions")
                if sessions < int(config["t_strategy"]["minimum_holding_days"]) and not bool(row["risk_override"]):
                    raise ContractError("T sell violates minimum holding sessions")
            proceeds = notional - float(costs["total_cost"])
            realized = proceeds - shares * float(holding[cost_key])
            holding[share_key] = before_shares - shares
            state["cash_cny"] += proceeds
            if holding[share_key] == 0:
                holding[cost_key] = 0.0
                holding[f"{sleeve}_open_date"] = None
                if sleeve == "core":
                    holding["full_target_shares_reference"] = 0.0

        holding["realized_pnl_cny"] = float(holding["realized_pnl_cny"] + realized)
        state["realized_pnl_cny"] = float(state["realized_pnl_cny"] + realized)
        state["as_of_date"] = str(max(pd.Timestamp(state["as_of_date"]), pd.Timestamp(row["fill_date"])).date())
        ledger = {
            "fill_id": fill_id,
            "fill_sha256": digest,
            "order_id": str(row["order_id"]),
            "fill_date": str(pd.Timestamp(row["fill_date"]).date()),
            "asset": asset,
            "name": str(row.get("name", holding["name"])),
            "asset_type": str(row["asset_type"]),
            "sector": str(row["sector"]),
            "sleeve": sleeve,
            "side": side,
            "shares": shares,
            "price": price,
            "notional": notional,
            "commission_cny": float(costs["commission"]),
            "stamp_duty_cny": float(costs["stamp_duty"]),
            "transfer_fee_cny": float(costs["transfer_fee"]),
            "slippage_cny": float(costs["slippage"]),
            "other_fees_cny": float(costs.get("other_fees", 0.0)),
            "total_cost_cny": float(costs["total_cost"]),
            "realized_pnl_cny": realized,
            "cash_before_cny": before_cash,
            "cash_after_cny": float(state["cash_cny"]),
            "shares_before": before_shares,
            "shares_after": int(holding[share_key]),
            "fee_mode": str(row["fee_mode"]),
            "manual_approval": bool(row["manual_approval"]),
            "manual_reason": str(row["manual_reason"]),
            "status": "applied",
        }
        state["fill_history"].append(ledger)
        state["processed_fills"].append({"fill_id": fill_id, "sha256": digest})
        processed[fill_id] = digest
        result_rows.append(ledger)

    state["holdings"] = [
        item for item in state["holdings"] if int(item["core_shares"]) > 0 or int(item["t_shares"]) > 0
    ]
    return state, pd.DataFrame(result_rows)


def write_account_atomic(account: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def write_ledger_view(account: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(account.get("fill_history", []), columns=LEDGER_COLUMNS).to_csv(
        temp_path, index=False, encoding="utf-8-sig"
    )
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "long_hold_v4.json")
    parser.add_argument("--account", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "account.json")
    parser.add_argument("--fills", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "pending_fills.csv")
    parser.add_argument("--orders", type=Path, default=ROOT / "outputs" / "long_hold_v4" / "current" / "order_intents.csv")
    parser.add_argument("--ledger", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "fill_ledger.csv")
    parser.add_argument("--apply", action="store_true", help="Persist account changes; default is dry-run")
    args = parser.parse_args()
    config = load_config(args.config)
    account = json.loads(args.account.read_text(encoding="utf-8"))
    fills = pd.read_csv(args.fills, encoding="utf-8-sig", dtype={"fill_id": str, "order_id": str, "asset": str})
    orders = pd.read_csv(args.orders, encoding="utf-8-sig", dtype={"order_id": str, "asset": str})
    state = normalize_account(account, config)
    results = pd.DataFrame()
    if not fills.empty:
        state, results = apply_fills(state, fills, config, orders)
    if args.apply:
        write_account_atomic(state, args.account)
        write_ledger_view(state, args.ledger)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "fills_in_batch": len(fills),
                "result_rows": len(results),
                "cash_cny": state["cash_cny"],
                "holding_count": len(state["holdings"]),
                "account_path": str(args.account),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
