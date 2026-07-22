"""Auditable paper/broker fill accounting for Long Hold V4 core and T sleeves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ContractError, estimate_trade_cost, load_config
from .order_envelope import (
    FILLABLE_ORDER_STATES,
    ORDER_COLUMNS,
    apply_order_fill,
    assert_order_state_account_binding,
    empty_order_state_book,
    normalize_order_state_book,
    order_state_record,
    rebind_order_state_account,
    refresh_expired_orders,
    verify_order_frame,
)
from .recoverable_transaction import commit_write_set, recover_pending_write_set


ROOT = Path(__file__).resolve().parents[2]
ACCOUNT_SCHEMA_VERSION = 3
ACCOUNT_FIELDS = {
    "schema_version",
    "state_version",
    "state_sha256",
    "account_id",
    "base_currency",
    "as_of_date",
    "cash_cny",
    "holdings",
    "realized_pnl_cny",
    "gross_dividend_cny",
    "dividend_tax_cny",
    "net_dividend_cny",
    "processed_fills",
    "fill_history",
    "processed_events",
    "event_history",
    "nav_history",
}
HOLDING_FIELDS = {
    "asset",
    "name",
    "asset_type",
    "sector",
    "core_shares",
    "core_average_cost_cny",
    "core_open_date",
    "t_shares",
    "t_average_cost_cny",
    "t_open_date",
    "full_target_shares_reference",
    "realized_pnl_cny",
    "cumulative_dividend_net_cny",
}
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
    "order_sha256",
    "run_id",
    "account_version_before",
    "account_state_sha256_before",
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
    "t_holding_sessions",
    "status",
]


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


def config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _account_state_sha256(account: dict[str, Any]) -> str:
    payload = copy.deepcopy(account)
    payload.pop("state_sha256", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def normalize_account(
    account: dict[str, Any],
    config: dict[str, Any],
    as_of: str | pd.Timestamp | None = None,
    *,
    _verify_state_hash: bool = True,
) -> dict[str, Any]:
    if not isinstance(account, dict):
        raise ContractError("account must be a JSON object")
    source_schema_version = _whole_nonnegative(account.get("schema_version", 1), "schema_version")
    if source_schema_version > ACCOUNT_SCHEMA_VERSION:
        raise ContractError("account schema version is newer than this runtime")
    if source_schema_version == ACCOUNT_SCHEMA_VERSION and _verify_state_hash:
        missing = sorted(ACCOUNT_FIELDS.difference(account))
        unknown = sorted(set(account).difference(ACCOUNT_FIELDS))
        if missing or unknown:
            raise ContractError(f"account v3 fields mismatch: missing={missing} unknown={unknown}")
        for raw_holding in account.get("holdings", []):
            if isinstance(raw_holding, dict) and set(raw_holding) != HOLDING_FIELDS:
                raise ContractError("account v3 holding fields mismatch")
    state_version = _whole_nonnegative(account.get("state_version", 0), "state_version")
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
        "state_version": state_version,
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
    if source_schema_version == ACCOUNT_SCHEMA_VERSION and _verify_state_hash:
        supplied_net_dividend = _finite_number(account.get("net_dividend_cny"), "net_dividend_cny")
        if not math.isclose(supplied_net_dividend, normalized["net_dividend_cny"], abs_tol=1e-9):
            raise ContractError("account net_dividend_cny is inconsistent")
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
    calculated_state_sha256 = _account_state_sha256(normalized)
    if source_schema_version >= ACCOUNT_SCHEMA_VERSION and _verify_state_hash:
        supplied_state_sha256 = str(account.get("state_sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", supplied_state_sha256):
            raise ContractError("account state_sha256 must be a lowercase SHA-256 value")
        if supplied_state_sha256 != calculated_state_sha256:
            raise ContractError("account state hash mismatch")
    normalized["state_sha256"] = calculated_state_sha256
    return normalized


def seal_account_state(
    account: dict[str, Any], config: dict[str, Any], *, increment_version: bool
) -> dict[str, Any]:
    candidate = copy.deepcopy(account)
    current_version = _whole_nonnegative(candidate.get("state_version", 0), "state_version")
    candidate["schema_version"] = ACCOUNT_SCHEMA_VERSION
    candidate["state_version"] = current_version + (1 if increment_version else 0)
    return normalize_account(candidate, config, _verify_state_hash=False)


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
    if out["fill_date"].dt.normalize().nunique() > 1:
        raise ContractError("one execution batch cannot span multiple fill dates")
    return out.sort_values(["fill_date", "fill_id"]).reset_index(drop=True)


def _approved_order(
    row: pd.Series,
    approved_orders: pd.DataFrame,
    order_state_book: dict[str, Any],
    account: dict[str, Any],
    *,
    expected_run_manifest_sha256: str,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_trade_calendar_sha256: str,
) -> pd.Series:
    if approved_orders.empty:
        raise ContractError(f"fill {row['fill_id']} has no approved order source")
    matches = approved_orders[approved_orders["order_id"].astype(str) == str(row["order_id"])]
    if len(matches) != 1:
        raise ContractError(f"fill {row['fill_id']} must match exactly one approved order")
    order = matches.iloc[0]
    record = order_state_record(order_state_book, str(order["order_id"]))
    if record["order_sha256"] != str(order["order_sha256"]):
        raise ContractError("order state hash does not match the approved envelope")
    if record["status"] not in FILLABLE_ORDER_STATES:
        raise ContractError(f"order is not fillable: {record['order_id']}/{record['status']}")
    if record["run_id"] != str(order["run_id"]):
        raise ContractError("order run binding mismatches its lifecycle state")
    if str(order["run_id"]) != expected_run_id or order_state_book["current_run_id"] != expected_run_id:
        raise ContractError("order run id does not match the active run manifest")
    if str(order["run_manifest_sha256"]) != expected_run_manifest_sha256:
        raise ContractError("order run manifest binding mismatch")
    if str(order["config_sha256"]) != expected_config_sha256:
        raise ContractError("order config binding mismatch")
    if str(order["trade_calendar_sha256"]) != expected_trade_calendar_sha256:
        raise ContractError("order trading-calendar binding mismatch")
    if record["filled_shares"] == 0 and (
        int(order["account_version"]) != int(account["state_version"])
        or str(order["account_state_sha256"]) != str(account["state_sha256"])
    ):
        raise ContractError("unfilled order is stale relative to the account state")
    manual_approval = bool(row["manual_approval"])
    manual_required = bool(order["manual_approval_required"])
    if manual_required and not manual_approval:
        raise ContractError("approved order requires explicit manual approval")
    if manual_approval and not str(row["manual_reason"]).strip():
        raise ContractError("manual approval requires manual_reason")
    if str(row["sleeve"]) == "core" and str(row["side"]) == "sell" and not (
        manual_required and manual_approval
    ):
        raise ContractError("core sell requires an envelope-authorized manual approval")
    if bool(row["risk_override"]) and not _boolean(order["risk_override_allowed"], "risk_override_allowed"):
        raise ContractError("fill risk_override is not authorized by the approved order")
    if bool(row["risk_override"]) and not manual_approval:
        raise ContractError("risk override requires explicit manual approval")
    if str(row["side"]) == "buy":
        risk_state = str(order["risk_state_at_signal"])
        if str(row["sleeve"]) == "core" and risk_state == "BRAKE":
            raise ContractError("core buy is blocked by the envelope risk state")
        if str(row["sleeve"]) == "t" and risk_state != "NORMAL":
            raise ContractError("T buy requires a NORMAL envelope risk state")
    for field in ["asset", "name", "asset_type", "sector", "sleeve", "side"]:
        order_value = _asset_code(order[field]) if field == "asset" else str(order[field]).strip().lower()
        if order_value != str(row[field]).strip().lower():
            raise ContractError(f"fill {row['fill_id']} mismatches approved order field {field}")
    if int(row["shares"]) > int(record["remaining_shares"]):
        raise ContractError(f"fill {row['fill_id']} exceeds remaining approved shares")
    signal_date = pd.to_datetime(order["signal_date"], errors="coerce")
    valid_from = pd.to_datetime(order["valid_from_date"], errors="coerce")
    valid_through = pd.to_datetime(order["valid_through_date"], errors="coerce")
    if pd.isna(signal_date) or pd.isna(valid_from) or pd.isna(valid_through):
        raise ContractError("approved order has invalid signal or validity dates")
    signal_date = pd.Timestamp(signal_date).normalize()
    valid_from = pd.Timestamp(valid_from).normalize()
    valid_through = pd.Timestamp(valid_through).normalize()
    fill_date = pd.Timestamp(row["fill_date"]).normalize()
    if fill_date < valid_from or fill_date > valid_through or valid_from <= signal_date:
        raise ContractError("fill is outside the approved order validity window")
    deviation_bps = abs(float(row["price"]) / float(order["indicative_price"]) - 1.0) * 10_000.0
    if deviation_bps > float(order.get("max_price_deviation_bps", float("nan"))):
        raise ContractError("fill price exceeds the envelope deviation limit")
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


def _trading_dates(trading_calendar: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(trading_calendar, pd.DataFrame) or trading_calendar.empty:
        raise ContractError("execution requires a non-empty trading calendar")
    date_column = next((name for name in ("date", "trade_date", "calendar_date") if name in trading_calendar), None)
    if date_column is None:
        raise ContractError("trading calendar requires date, trade_date, or calendar_date")
    rows = trading_calendar.copy()
    if "is_trading_day" in rows:
        rows = rows[rows["is_trading_day"].map(lambda value: _boolean(value, "is_trading_day"))]
    elif "is_open" in rows:
        rows = rows[rows["is_open"].map(lambda value: _boolean(value, "is_open"))]
    dates = pd.to_datetime(rows[date_column], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        raise ContractError("trading calendar contains no valid open dates")
    return pd.DatetimeIndex(dates)


def _holding_trade_sessions(trading_dates: pd.DatetimeIndex, open_date: Any, fill_date: Any) -> int:
    opened = pd.to_datetime(open_date, errors="coerce")
    filled = pd.to_datetime(fill_date, errors="coerce")
    if pd.isna(opened) or pd.isna(filled):
        raise ContractError("T holding dates must be valid")
    opened = pd.Timestamp(opened).normalize()
    filled = pd.Timestamp(filled).normalize()
    return int(((trading_dates >= opened) & (trading_dates <= filled)).sum())


def _post_fill_portfolio_check(
    account: dict[str, Any], valuation_prices: dict[str, float], config: dict[str, Any]
) -> None:
    holdings = [
        item
        for item in account["holdings"]
        if int(item["core_shares"]) > 0 or int(item["t_shares"]) > 0
    ]
    if len(holdings) > int(config["universe"]["maximum_assets"]):
        raise ContractError("post-fill portfolio exceeds maximum holding count")
    market_values: dict[str, float] = {}
    for holding in holdings:
        asset = str(holding["asset"])
        if asset not in valuation_prices:
            raise ContractError(f"post-fill risk check is missing a held asset price: {asset}")
        price = _nonnegative_number(valuation_prices[asset], f"valuation price[{asset}]")
        if price <= 0:
            raise ContractError(f"post-fill valuation price must be positive: {asset}")
        market_values[asset] = (int(holding["core_shares"]) + int(holding["t_shares"])) * price
    nav = float(account["cash_cny"] + sum(market_values.values()))
    if nav <= 0 or not math.isfinite(nav):
        raise ContractError("post-fill account NAV must be finite and positive")
    tolerance = 1e-9
    if float(account["cash_cny"]) / nav + tolerance < float(config["portfolio"]["minimum_cash_weight"]):
        raise ContractError("post-fill portfolio breaches minimum cash weight")
    sector_values: dict[str, float] = {}
    core_value = 0.0
    t_value = 0.0
    for holding in holdings:
        asset = str(holding["asset"])
        price = float(valuation_prices[asset])
        asset_value = market_values[asset]
        cap = float(config["portfolio"][f"max_single_{holding['asset_type']}_weight"])
        if asset_value / nav > cap + tolerance:
            raise ContractError(f"post-fill portfolio breaches single-asset cap: {asset}")
        sector = str(holding["sector"])
        sector_values[sector] = sector_values.get(sector, 0.0) + asset_value
        core_value += int(holding["core_shares"]) * price
        t_value += int(holding["t_shares"]) * price
        if int(holding["t_shares"]) > 0:
            reference = float(holding["full_target_shares_reference"])
            if int(holding["core_shares"]) <= 0 or reference <= 0:
                raise ContractError("post-fill T sleeve lacks a valid core holding")
            core_fraction = int(holding["core_shares"]) / reference
            if core_fraction + tolerance < float(config["t_strategy"]["core_fraction_required"]):
                raise ContractError("post-fill T sleeve is under-supported by the core holding")
            t_share_cap = reference * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"])
            if int(holding["t_shares"]) > t_share_cap + tolerance:
                raise ContractError("post-fill T sleeve exceeds its per-asset share cap")
    if any(
        value / nav > float(config["portfolio"]["max_sector_weight"]) + tolerance
        for value in sector_values.values()
    ):
        raise ContractError("post-fill portfolio breaches sector cap")
    if core_value / nav > float(config["portfolio"]["target_core_exposure"]) + tolerance:
        raise ContractError("post-fill portfolio breaches aggregate core cap")
    if t_value / nav > float(config["t_strategy"]["portfolio_t_weight_cap"]) + tolerance:
        raise ContractError("post-fill portfolio breaches aggregate T cap")


def apply_fills(
    account: dict[str, Any],
    fills: pd.DataFrame,
    config: dict[str, Any],
    *,
    approved_orders: pd.DataFrame,
    order_state_book: dict[str, Any],
    trading_calendar: pd.DataFrame,
    valuation_prices: dict[str, float],
    valuation_as_of_date: str | pd.Timestamp,
    run_manifest_sha256: str,
    expected_run_id: str,
    expected_config_sha256: str,
    trading_calendar_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    state = normalize_account(account, config)
    runtime_config_sha256 = config_sha256(config)
    if expected_config_sha256 != runtime_config_sha256:
        raise ContractError("execution config hash does not match the approved run")
    if not re.fullmatch(r"[0-9a-f]{64}", str(run_manifest_sha256)):
        raise ContractError("execution requires a valid run manifest SHA-256")
    if not str(expected_run_id).strip():
        raise ContractError("execution requires a non-empty run id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(trading_calendar_sha256)):
        raise ContractError("execution requires a valid trading calendar SHA-256")
    orders = verify_order_frame(approved_orders)
    book = assert_order_state_account_binding(
        order_state_book, state["state_version"], state["state_sha256"]
    )
    prepared = _prepare_fills(fills)
    trading_dates = _trading_dates(trading_calendar)
    if not prepared.empty:
        batch_date = pd.Timestamp(prepared.iloc[0]["fill_date"]).normalize()
        if batch_date not in trading_dates:
            raise ContractError("fill date is not an open session in the trading calendar")
        valuation_date = pd.to_datetime(valuation_as_of_date, errors="coerce")
        if pd.isna(valuation_date):
            raise ContractError("execution valuation snapshot requires a valid as_of date")
        valuation_age = int((batch_date - pd.Timestamp(valuation_date).normalize()).days)
        if valuation_age < 0 or valuation_age > int(config["model"]["max_price_age_days"]):
            raise ContractError("execution valuation snapshot is stale or later than the fill date")
        book = refresh_expired_orders(book, batch_date, updated_at=batch_date.isoformat())
    prices: dict[str, float] = {}
    for asset, value in valuation_prices.items():
        code = _asset_code(asset)
        price = _nonnegative_number(value, f"valuation price[{code}]")
        if price <= 0:
            raise ContractError(f"valuation price must be positive: {code}")
        prices[code] = price
    processed = {str(item["fill_id"]): str(item["sha256"]) for item in state["processed_fills"]}
    result_rows: list[dict[str, Any]] = []
    applied_order_ids: set[str] = set()
    account_version_before = int(state["state_version"])
    account_hash_before = str(state["state_sha256"])

    for _, row in prepared.iterrows():
        fill_id = str(row["fill_id"])
        digest = _fill_hash(row)
        if fill_id in processed:
            if processed[fill_id] != digest:
                raise ContractError(f"fill_id payload changed after processing: {fill_id}")
            result_rows.append({"fill_id": fill_id, "status": "duplicate_ignored"})
            continue
        order = _approved_order(
            row,
            orders,
            book,
            state,
            expected_run_manifest_sha256=str(run_manifest_sha256),
            expected_run_id=str(expected_run_id),
            expected_config_sha256=runtime_config_sha256,
            expected_trade_calendar_sha256=str(trading_calendar_sha256),
        )
        if pd.Timestamp(row["fill_date"]).normalize() < pd.Timestamp(state["as_of_date"]):
            raise ContractError(f"fill {fill_id} predates the account state")

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
        t_holding_sessions_at_fill: int | None = None

        if sleeve == "t" and side == "buy":
            reference = float(holding["full_target_shares_reference"])
            core_fraction = holding["core_shares"] / reference if reference > 0 else 0.0
            required_core = float(config["t_strategy"]["core_fraction_required"])
            if holding["core_shares"] <= 0 or core_fraction + 1e-9 < required_core:
                raise ContractError("T buy requires the configured core fraction at fill time")
            if _nonnegative_number(
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
                reference = order.get("full_target_shares_reference")
                reference = _nonnegative_number(reference, "full_target_shares_reference")
                if reference + 1e-9 < holding["core_shares"]:
                    raise ContractError("core buy full target reference is below resulting core shares")
                holding["full_target_shares_reference"] = float(reference)
        else:
            if shares > before_shares:
                raise ContractError(f"fill {fill_id} attempts to sell more {sleeve} shares than held")
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
                t_holding_sessions_at_fill = _holding_trade_sessions(trading_dates, open_date, row["fill_date"])
                if t_holding_sessions_at_fill < int(config["t_strategy"]["minimum_holding_days"]) and not bool(row["risk_override"]):
                    raise ContractError("T sell violates minimum holding sessions")
            proceeds = notional - float(costs["total_cost"])
            if proceeds < -1e-9:
                raise ContractError("sell transaction costs cannot exceed proceeds")
            realized = proceeds - shares * float(holding[cost_key])
            holding[share_key] = before_shares - shares
            state["cash_cny"] += proceeds
            if holding[share_key] == 0:
                holding[cost_key] = 0.0
                holding[f"{sleeve}_open_date"] = None
                if sleeve == "core":
                    holding["full_target_shares_reference"] = 0.0

        prices[asset] = price
        _post_fill_portfolio_check(state, prices, config)
        holding["realized_pnl_cny"] = float(holding["realized_pnl_cny"] + realized)
        state["realized_pnl_cny"] = float(state["realized_pnl_cny"] + realized)
        state["as_of_date"] = str(max(pd.Timestamp(state["as_of_date"]), pd.Timestamp(row["fill_date"])).date())
        ledger = {
            "fill_id": fill_id,
            "fill_sha256": digest,
            "order_id": str(row["order_id"]),
            "order_sha256": str(order["order_sha256"]),
            "run_id": str(order["run_id"]),
            "account_version_before": account_version_before,
            "account_state_sha256_before": account_hash_before,
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
            "t_holding_sessions": t_holding_sessions_at_fill,
            "status": "applied",
        }
        state["fill_history"].append(ledger)
        state["processed_fills"].append({"fill_id": fill_id, "sha256": digest})
        processed[fill_id] = digest
        book = apply_order_fill(book, str(order["order_id"]), shares, updated_at=pd.Timestamp(row["fill_date"]).isoformat())
        applied_order_ids.add(str(order["order_id"]))
        result_rows.append(ledger)

    state["holdings"] = [
        item for item in state["holdings"] if int(item["core_shares"]) > 0 or int(item["t_shares"]) > 0
    ]
    if applied_order_ids:
        state = seal_account_state(state, config, increment_version=True)
        retained_partial_ids = {
            order_id
            for order_id in applied_order_ids
            if order_state_record(book, order_id)["status"] == "PARTIALLY_FILLED"
        }
        book = rebind_order_state_account(
            book,
            state["state_version"],
            state["state_sha256"],
            updated_at=pd.Timestamp(state["as_of_date"]).isoformat(),
            retain_partially_filled_order_ids=retained_partial_ids,
            invalidate_reason="account_state_changed_after_fill",
        )
    return state, pd.DataFrame(result_rows), book


def write_ledger_view(account: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(account.get("fill_history", []), columns=LEDGER_COLUMNS).to_csv(
        temp_path, index=False, encoding="utf-8-sig"
    )
    os.replace(temp_path, path)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _ledger_bytes(account: dict[str, Any]) -> bytes:
    buffer = io.StringIO()
    pd.DataFrame(account.get("fill_history", []), columns=LEDGER_COLUMNS).to_csv(buffer, index=False)
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


def execution_transaction_path(account_path: Path) -> Path:
    return account_path.parent / ".execution_transaction.json"


def recover_execution_transaction(
    account_path: Path, ledger_path: Path, order_state_path: Path
) -> bool:
    return recover_pending_write_set(
        execution_transaction_path(account_path), [account_path, ledger_path, order_state_path]
    )


def commit_execution_transaction(
    account: dict[str, Any],
    order_state_book: dict[str, Any],
    account_path: Path,
    ledger_path: Path,
    order_state_path: Path,
) -> None:
    if str(account.get("state_sha256", "")) != _account_state_sha256(account):
        raise ContractError("account state hash is invalid at transaction commit")
    order_state_book = normalize_order_state_book(order_state_book)
    if int(order_state_book.get("account_version", -1)) != int(account.get("state_version", -2)) or str(
        order_state_book.get("account_state_sha256", "")
    ) != str(account.get("state_sha256", "")):
        raise ContractError("account and order state book are not transactionally bound")
    commit_write_set(
        {
            account_path: _json_bytes(account),
            ledger_path: _ledger_bytes(account),
            order_state_path: _json_bytes(order_state_book),
        },
        execution_transaction_path(account_path),
    )


def initialize_persistent_account(
    account_path: Path,
    ledger_path: Path,
    order_state_path: Path,
    config: dict[str, Any],
    as_of: str | pd.Timestamp,
    *,
    account_id: str = "long_hold_v4_primary",
) -> dict[str, Any]:
    recover_execution_transaction(account_path, ledger_path, order_state_path)
    if account_path.exists() or order_state_path.exists():
        raise ContractError("persistent account initialization refuses to overwrite existing state")
    date = _date_or_none(as_of, "initial account as_of")
    if date is None:
        raise ContractError("persistent account initialization requires an as_of date")
    account = seal_account_state(
        {
            "schema_version": ACCOUNT_SCHEMA_VERSION,
            "state_version": 0,
            "account_id": account_id,
            "base_currency": config["account"]["base_currency"],
            "as_of_date": date,
            "cash_cny": float(config["account"]["initial_cash_cny"]),
            "holdings": [],
            "realized_pnl_cny": 0.0,
            "gross_dividend_cny": 0.0,
            "dividend_tax_cny": 0.0,
            "processed_fills": [],
            "fill_history": [],
            "processed_events": [],
            "event_history": [],
            "nav_history": [
                {
                    "date": date,
                    "nav_cny": float(config["account"]["initial_cash_cny"]),
                    "cash_cny": float(config["account"]["initial_cash_cny"]),
                    "market_value_cny": 0.0,
                    "peak_nav_cny": float(config["account"]["initial_cash_cny"]),
                    "drawdown": 0.0,
                    "risk_state": "NORMAL",
                }
            ],
        },
        config,
        increment_version=False,
    )
    book = empty_order_state_book(account["state_version"], account["state_sha256"])
    commit_execution_transaction(account, book, account_path, ledger_path, order_state_path)
    return account


def _load_execution_prices(
    path: Path, fill_date: pd.Timestamp, config: dict[str, Any]
) -> tuple[dict[str, float], pd.Timestamp]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
    required = {"asset", "price", "as_of_date"}
    if not required.issubset(frame.columns):
        raise ContractError(f"execution valuation snapshot missing columns: {sorted(required.difference(frame.columns))}")
    dates = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.nunique() != 1:
        raise ContractError("execution valuation snapshot must have one valid as_of date")
    price_date = pd.Timestamp(dates.iloc[0]).normalize()
    age = int((fill_date.normalize() - price_date).days)
    if age < 0 or age > int(config["model"]["max_price_age_days"]):
        raise ContractError("execution valuation snapshot is stale or later than the fill date")
    if frame["asset"].astype(str).str.zfill(6).duplicated().any():
        raise ContractError("execution valuation snapshot contains duplicate assets")
    return (
        {
            _asset_code(row["asset"]): _nonnegative_number(row["price"], f"valuation price[{row['asset']}]")
            for _, row in frame.iterrows()
        },
        price_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "long_hold_v4.json")
    parser.add_argument("--account", type=Path)
    parser.add_argument("--fills", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "pending_fills.csv")
    parser.add_argument("--orders", type=Path)
    parser.add_argument("--order-state", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--valuation-prices", type=Path)
    parser.add_argument("--ledger", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "fill_ledger.csv")
    parser.add_argument("--initialize-account", action="store_true")
    parser.add_argument("--migrate-account-state", action="store_true")
    parser.add_argument("--initial-as-of")
    parser.add_argument("--apply", action="store_true", help="Persist account changes; default is dry-run")
    args = parser.parse_args()
    config = load_config(args.config)
    account_path = args.account or ROOT / config["data"]["account_path"]
    order_state_path = args.order_state or ROOT / config["data"]["order_state_path"]
    orders_path = args.orders or ROOT / config["data"]["output_directory"] / "order_intents.csv"
    manifest_path = args.manifest or ROOT / config["data"]["output_directory"] / "run_manifest.json"
    calendar_path = args.calendar or ROOT / config["data"]["trade_calendar_path"]
    valuation_path = args.valuation_prices or ROOT / config["data"]["output_directory"] / "execution_valuation_prices.csv"
    recover_execution_transaction(account_path, args.ledger, order_state_path)
    if args.initialize_account:
        if not args.initial_as_of:
            raise ContractError("--initialize-account requires --initial-as-of")
        account = initialize_persistent_account(
            account_path, args.ledger, order_state_path, config, args.initial_as_of
        )
        print(
            json.dumps(
                {
                    "mode": "initialized",
                    "account_path": str(account_path),
                    "state_version": account["state_version"],
                    "state_sha256": account["state_sha256"],
                },
                ensure_ascii=False,
            )
        )
        return
    if not account_path.is_file():
        raise ContractError(
            "persistent account is missing; use --initialize-account --initial-as-of YYYY-MM-DD"
        )
    raw_account = json.loads(account_path.read_text(encoding="utf-8"))
    if args.migrate_account_state:
        if order_state_path.exists():
            raise ContractError("account migration refuses to overwrite an existing order state book")
        account = seal_account_state(
            normalize_account(raw_account, config), config, increment_version=False
        )
        book = empty_order_state_book(account["state_version"], account["state_sha256"])
        commit_execution_transaction(account, book, account_path, args.ledger, order_state_path)
        print(
            json.dumps(
                {
                    "mode": "migrated",
                    "account_path": str(account_path),
                    "schema_version": account["schema_version"],
                    "state_version": account["state_version"],
                    "state_sha256": account["state_sha256"],
                },
                ensure_ascii=False,
            )
        )
        return
    if not order_state_path.is_file():
        raise ContractError("persistent order state book is missing; run explicit account migration")
    for required_path, label in [
        (orders_path, "approved order source"),
        (manifest_path, "run manifest"),
        (calendar_path, "trading calendar"),
    ]:
        if not required_path.is_file():
            raise ContractError(f"{label} is missing: {required_path}")
    account = normalize_account(raw_account, config)
    fills = pd.read_csv(args.fills, encoding="utf-8-sig", dtype={"fill_id": str, "order_id": str, "asset": str})
    orders = pd.read_csv(orders_path, encoding="utf-8-sig", dtype={"order_id": str, "asset": str})
    order_state = json.loads(order_state_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    runtime_config_sha256 = config_sha256(config)
    if manifest.get("config_sha256") != runtime_config_sha256 or not str(manifest.get("run_id", "")).strip():
        raise ContractError("run manifest is not bound to the active config and run id")
    calendar = pd.read_csv(calendar_path, encoding="utf-8-sig", low_memory=False)
    state = account
    results = pd.DataFrame()
    if not fills.empty:
        prepared_dates = pd.to_datetime(fills["fill_date"], errors="coerce").dropna().dt.normalize()
        if len(prepared_dates) != len(fills) or prepared_dates.nunique() != 1:
            raise ContractError("fills require one valid execution date")
        if not valuation_path.is_file():
            raise ContractError(f"execution valuation snapshot is missing: {valuation_path}")
        valuation_prices, valuation_as_of_date = _load_execution_prices(
            valuation_path, prepared_dates.iloc[0], config
        )
        state, results, order_state = apply_fills(
            state,
            fills,
            config,
            approved_orders=orders,
            order_state_book=order_state,
            trading_calendar=calendar,
            valuation_prices=valuation_prices,
            valuation_as_of_date=valuation_as_of_date,
            run_manifest_sha256=manifest_sha256,
            expected_run_id=str(manifest["run_id"]),
            expected_config_sha256=runtime_config_sha256,
            trading_calendar_sha256=hashlib.sha256(calendar_path.read_bytes()).hexdigest(),
        )
    if args.apply:
        commit_execution_transaction(state, order_state, account_path, args.ledger, order_state_path)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "fills_in_batch": len(fills),
                "result_rows": len(results),
                "cash_cny": state["cash_cny"],
                "holding_count": len(state["holdings"]),
                "account_path": str(account_path),
                "state_version": state["state_version"],
                "state_sha256": state["state_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
