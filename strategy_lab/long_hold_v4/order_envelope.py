"""Versioned, hash-authenticated order envelopes and lifecycle state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .core import ContractError


ORDER_ENVELOPE_SCHEMA_VERSION = 1
ORDER_STATE_BOOK_SCHEMA_VERSION = 1
ORDER_LIFECYCLE_STATES = {
    "ACTIVE",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELLED",
    "EXPIRED",
    "SUPERSEDED",
}
FILLABLE_ORDER_STATES = {"ACTIVE", "PARTIALLY_FILLED"}
TERMINAL_ORDER_STATES = ORDER_LIFECYCLE_STATES.difference(FILLABLE_ORDER_STATES)
ALLOWED_TRANSITIONS = {
    "ACTIVE": {"PARTIALLY_FILLED", "FILLED", "CANCELLED", "EXPIRED", "SUPERSEDED"},
    "PARTIALLY_FILLED": {"FILLED", "CANCELLED", "EXPIRED", "SUPERSEDED"},
}
ORDER_STATE_BOOK_FIELDS = {
    "schema_version",
    "account_version",
    "account_state_sha256",
    "current_run_id",
    "orders",
    "book_sha256",
}
ORDER_STATE_RECORD_FIELDS = {
    "order_id",
    "order_sha256",
    "run_id",
    "asset",
    "sleeve",
    "side",
    "ordered_shares",
    "filled_shares",
    "remaining_shares",
    "status",
    "valid_through_date",
    "superseded_by_order_id",
    "status_reason",
    "created_at",
    "updated_at",
}

ORDER_COLUMNS = [
    "order_schema_version",
    "order_id",
    "order_sha256",
    "run_id",
    "run_manifest_sha256",
    "config_sha256",
    "trade_calendar_sha256",
    "account_version",
    "account_state_sha256",
    "signal_date",
    "valid_from_date",
    "valid_through_date",
    "asset",
    "name",
    "asset_type",
    "sector",
    "sleeve",
    "side",
    "shares",
    "indicative_price",
    "max_price_deviation_bps",
    "notional",
    "estimated_cost",
    "target_core_weight",
    "target_t_weight_cap",
    "full_target_weight",
    "full_target_shares_reference",
    "core_fraction_at_signal",
    "t_holding_sessions",
    "risk_state_at_signal",
    "risk_override_allowed",
    "manual_approval_required",
    "status",
    "intent_status",
    "reason",
]

_FLOAT_PRECISION = {
    "indicative_price": 8,
    "max_price_deviation_bps": 4,
    "notional": 8,
    "estimated_cost": 8,
    "target_core_weight": 12,
    "target_t_weight_cap": 12,
    "full_target_weight": 12,
    "full_target_shares_reference": 8,
    "core_fraction_at_signal": 12,
}


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ContractError(f"{field} must be a boolean")


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an integer")
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)) or not float(number).is_integer():
        raise ContractError(f"{field} must be an integer")
    result = int(number)
    if result < minimum:
        raise ContractError(f"{field} must be at least {minimum}")
    return result


def _number(value: Any, field: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be finite")
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        raise ContractError(f"{field} must be finite")
    result = float(number)
    if result < minimum or (maximum is not None and result > maximum):
        qualifier = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ContractError(f"{field} must be{qualifier}")
    return result


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = ""
    else:
        text = str(value).strip()
    if not allow_empty and not text:
        raise ContractError(f"{field} cannot be empty")
    return text


def _date(value: Any, field: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ContractError(f"{field} must be a valid date")
    return str(pd.Timestamp(timestamp).date())


def _sha256(value: Any, field: str) -> str:
    digest = _text(value, field).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContractError(f"{field} must be a lowercase SHA-256 value")
    return digest


def _asset(value: Any) -> str:
    asset = _text(value, "asset").zfill(6)
    if not re.fullmatch(r"\d{6}", asset):
        raise ContractError("asset must be a six-digit A-share or ETF code")
    return asset


def _decimal_text(value: float, places: int) -> str:
    text = f"{float(value):.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _raw_mapping(raw: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    return dict(raw.items())


def _normalize_envelope_fields(raw: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    item = _raw_mapping(raw)
    missing = sorted(set(ORDER_COLUMNS).difference(item).difference({"order_sha256"}))
    unknown = sorted(set(item).difference(ORDER_COLUMNS))
    if missing or unknown:
        raise ContractError(f"order envelope fields mismatch: missing={missing} unknown={unknown}")
    normalized: dict[str, Any] = {
        "order_schema_version": _integer(item.get("order_schema_version"), "order_schema_version", minimum=1),
        "order_id": _text(item.get("order_id"), "order_id"),
        "run_id": _text(item.get("run_id"), "run_id"),
        "run_manifest_sha256": _sha256(item.get("run_manifest_sha256"), "run_manifest_sha256"),
        "config_sha256": _sha256(item.get("config_sha256"), "config_sha256"),
        "trade_calendar_sha256": _sha256(
            item.get("trade_calendar_sha256"), "trade_calendar_sha256"
        ),
        "account_version": _integer(item.get("account_version"), "account_version"),
        "account_state_sha256": _sha256(item.get("account_state_sha256"), "account_state_sha256"),
        "signal_date": _date(item.get("signal_date"), "signal_date"),
        "valid_from_date": _date(item.get("valid_from_date"), "valid_from_date"),
        "valid_through_date": _date(item.get("valid_through_date"), "valid_through_date"),
        "asset": _asset(item.get("asset")),
        "name": _text(item.get("name"), "name"),
        "asset_type": _text(item.get("asset_type"), "asset_type").lower(),
        "sector": _text(item.get("sector"), "sector").lower(),
        "sleeve": _text(item.get("sleeve"), "sleeve").lower(),
        "side": _text(item.get("side"), "side").lower(),
        "shares": _integer(item.get("shares"), "shares"),
        "indicative_price": _number(item.get("indicative_price"), "indicative_price", minimum=0.0),
        "max_price_deviation_bps": _number(
            item.get("max_price_deviation_bps"), "max_price_deviation_bps", minimum=0.0, maximum=10_000.0
        ),
        "notional": _number(item.get("notional"), "notional", minimum=0.0),
        "estimated_cost": _number(item.get("estimated_cost"), "estimated_cost", minimum=0.0),
        "target_core_weight": _number(item.get("target_core_weight"), "target_core_weight", maximum=1.0),
        "target_t_weight_cap": _number(item.get("target_t_weight_cap"), "target_t_weight_cap", maximum=1.0),
        "full_target_weight": _number(item.get("full_target_weight"), "full_target_weight", maximum=1.0),
        "full_target_shares_reference": _number(
            item.get("full_target_shares_reference"), "full_target_shares_reference", minimum=0.0
        ),
        "core_fraction_at_signal": _number(
            item.get("core_fraction_at_signal"), "core_fraction_at_signal", maximum=1.0
        ),
        "t_holding_sessions": _integer(item.get("t_holding_sessions"), "t_holding_sessions"),
        "risk_state_at_signal": _text(item.get("risk_state_at_signal"), "risk_state_at_signal").upper(),
        "risk_override_allowed": _boolean(item.get("risk_override_allowed"), "risk_override_allowed"),
        "manual_approval_required": _boolean(
            item.get("manual_approval_required"), "manual_approval_required"
        ),
        "status": _text(item.get("status"), "status").upper(),
        "intent_status": _text(item.get("intent_status"), "intent_status").upper(),
        "reason": _text(item.get("reason"), "reason", allow_empty=True),
    }
    if normalized["order_schema_version"] != ORDER_ENVELOPE_SCHEMA_VERSION:
        raise ContractError("unsupported order envelope schema version")
    if normalized["asset_type"] not in {"stock", "etf"}:
        raise ContractError("order asset_type must be stock or etf")
    if normalized["sleeve"] not in {"core", "t"}:
        raise ContractError("order sleeve must be core or t")
    if normalized["side"] not in {"buy", "sell", "review"}:
        raise ContractError("order side must be buy, sell, or review")
    if normalized["risk_state_at_signal"] not in {"NORMAL", "REVIEW", "BRAKE", "UNKNOWN"}:
        raise ContractError("order risk_state_at_signal is invalid")
    if normalized["status"] not in ORDER_LIFECYCLE_STATES:
        raise ContractError("order status is invalid")
    signal = pd.Timestamp(normalized["signal_date"])
    valid_from = pd.Timestamp(normalized["valid_from_date"])
    valid_through = pd.Timestamp(normalized["valid_through_date"])
    if valid_from <= signal or valid_through < valid_from:
        raise ContractError("order validity window must start after the close signal")
    executable = normalized["side"] in {"buy", "sell"}
    if executable:
        if normalized["status"] != "ACTIVE":
            raise ContractError("an executable envelope must start ACTIVE")
        if normalized["shares"] <= 0 or normalized["indicative_price"] <= 0 or normalized["notional"] <= 0:
            raise ContractError("an executable envelope requires positive shares, price, and notional")
        expected_notional = normalized["shares"] * normalized["indicative_price"]
        if not math.isclose(normalized["notional"], expected_notional, rel_tol=1e-9, abs_tol=0.01):
            raise ContractError("order notional does not reconcile to shares and indicative price")
    elif normalized["shares"] != 0 or normalized["notional"] != 0 or normalized["status"] != "CANCELLED":
        raise ContractError("a review notice must be non-executable and start CANCELLED")
    if normalized["risk_override_allowed"] and not (
        normalized["sleeve"] == "t" and normalized["side"] == "sell"
    ):
        raise ContractError("risk override can only authorize a T-sleeve sell")
    return normalized


def _canonical_envelope_payload(normalized: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in ORDER_COLUMNS:
        if field == "order_sha256":
            continue
        value = normalized[field]
        payload[field] = _decimal_text(value, _FLOAT_PRECISION[field]) if field in _FLOAT_PRECISION else value
    return payload


def order_envelope_sha256(raw: Mapping[str, Any] | pd.Series) -> str:
    normalized = _normalize_envelope_fields(raw)
    encoded = json.dumps(
        _canonical_envelope_payload(normalized), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_order_envelope(raw: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    normalized = _normalize_envelope_fields(raw)
    normalized["order_sha256"] = order_envelope_sha256(normalized)
    return {field: normalized[field] for field in ORDER_COLUMNS}


def verify_order_envelope(raw: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    item = _raw_mapping(raw)
    supplied = _sha256(item.get("order_sha256"), "order_sha256")
    normalized = _normalize_envelope_fields(item)
    calculated = order_envelope_sha256(normalized)
    if supplied != calculated:
        raise ContractError(f"order envelope hash mismatch: {normalized['order_id']}")
    normalized["order_sha256"] = supplied
    return {field: normalized[field] for field in ORDER_COLUMNS}


def verify_order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ORDER_COLUMNS).difference(frame.columns))
    unknown = sorted(set(frame.columns).difference(ORDER_COLUMNS))
    if missing or unknown:
        raise ContractError(f"approved order envelope columns mismatch: missing={missing} unknown={unknown}")
    normalized = [verify_order_envelope(row) for _, row in frame.iterrows()]
    out = pd.DataFrame(normalized, columns=ORDER_COLUMNS)
    if out["order_id"].duplicated().any():
        raise ContractError("approved order source contains duplicate order ids")
    return out


def _timestamp(value: Any, field: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ContractError(f"{field} must be a valid timestamp")
    return pd.Timestamp(timestamp).isoformat()


def _normalize_state_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != ORDER_STATE_RECORD_FIELDS:
        raise ContractError("order state record fields mismatch")
    record = {
        "order_id": _text(raw.get("order_id"), "order state order_id"),
        "order_sha256": _sha256(raw.get("order_sha256"), "order state order_sha256"),
        "run_id": _text(raw.get("run_id"), "order state run_id"),
        "asset": _asset(raw.get("asset")),
        "sleeve": _text(raw.get("sleeve"), "order state sleeve").lower(),
        "side": _text(raw.get("side"), "order state side").lower(),
        "ordered_shares": _integer(raw.get("ordered_shares"), "ordered_shares"),
        "filled_shares": _integer(raw.get("filled_shares"), "filled_shares"),
        "remaining_shares": _integer(raw.get("remaining_shares"), "remaining_shares"),
        "status": _text(raw.get("status"), "order state status").upper(),
        "valid_through_date": _date(raw.get("valid_through_date"), "order state valid_through_date"),
        "superseded_by_order_id": (
            None
            if raw.get("superseded_by_order_id") in {None, ""}
            else _text(raw.get("superseded_by_order_id"), "superseded_by_order_id")
        ),
        "status_reason": _text(raw.get("status_reason", ""), "status_reason", allow_empty=True),
        "created_at": _timestamp(raw.get("created_at"), "order state created_at"),
        "updated_at": _timestamp(raw.get("updated_at"), "order state updated_at"),
    }
    if record["sleeve"] not in {"core", "t"} or record["side"] not in {"buy", "sell", "review"}:
        raise ContractError("order state contains an invalid sleeve or side")
    if record["status"] not in ORDER_LIFECYCLE_STATES:
        raise ContractError("order state contains an invalid lifecycle status")
    if record["filled_shares"] + record["remaining_shares"] != record["ordered_shares"]:
        raise ContractError("order state shares do not reconcile")
    expected_status = (
        "FILLED"
        if record["ordered_shares"] > 0 and record["remaining_shares"] == 0
        else "PARTIALLY_FILLED"
        if record["filled_shares"] > 0 and record["remaining_shares"] > 0
        else None
    )
    if expected_status and record["status"] in FILLABLE_ORDER_STATES.union({"FILLED"}) and record["status"] != expected_status:
        raise ContractError("order lifecycle status does not match filled shares")
    if record["status"] == "SUPERSEDED" and not record["status_reason"]:
        raise ContractError("superseded order state requires a reason")
    return record


def _canonical_book_payload(book: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(book["schema_version"]),
        "account_version": int(book["account_version"]),
        "account_state_sha256": str(book["account_state_sha256"]),
        "current_run_id": str(book["current_run_id"]),
        "orders": sorted(copy.deepcopy(list(book["orders"])), key=lambda item: item["order_id"]),
    }


def _book_sha256(book: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_book_payload(book), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_book(book: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_book_payload(book)
    payload["book_sha256"] = _book_sha256(payload)
    return payload


def normalize_order_state_book(book: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(book, Mapping):
        raise ContractError("order state book must be an object")
    if set(book) != ORDER_STATE_BOOK_FIELDS:
        raise ContractError("order state book fields mismatch")
    schema_version = _integer(book.get("schema_version"), "order state schema_version", minimum=1)
    if schema_version != ORDER_STATE_BOOK_SCHEMA_VERSION:
        raise ContractError("unsupported order state book schema version")
    orders_raw = book.get("orders")
    if not isinstance(orders_raw, list):
        raise ContractError("order state book orders must be a list")
    normalized = {
        "schema_version": schema_version,
        "account_version": _integer(book.get("account_version"), "order state account_version"),
        "account_state_sha256": _sha256(book.get("account_state_sha256"), "order state account_state_sha256"),
        "current_run_id": _text(book.get("current_run_id", ""), "current_run_id", allow_empty=True),
        "orders": [_normalize_state_record(item) for item in orders_raw],
    }
    ids = [item["order_id"] for item in normalized["orders"]]
    if len(ids) != len(set(ids)):
        raise ContractError("order state book contains duplicate order ids")
    supplied = _sha256(book.get("book_sha256"), "book_sha256")
    calculated = _book_sha256(normalized)
    if supplied != calculated:
        raise ContractError("order state book hash mismatch")
    normalized["orders"].sort(key=lambda item: item["order_id"])
    normalized["book_sha256"] = supplied
    return normalized


def empty_order_state_book(account_version: int, account_state_sha256: str) -> dict[str, Any]:
    return _seal_book(
        {
            "schema_version": ORDER_STATE_BOOK_SCHEMA_VERSION,
            "account_version": _integer(account_version, "account_version"),
            "account_state_sha256": _sha256(account_state_sha256, "account_state_sha256"),
            "current_run_id": "",
            "orders": [],
        }
    )


def assert_order_state_account_binding(
    book: Mapping[str, Any], account_version: int, account_state_sha256: str
) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    if normalized["account_version"] != int(account_version) or normalized["account_state_sha256"] != str(
        account_state_sha256
    ):
        raise ContractError("order state book is stale relative to the account state")
    return normalized


def register_order_envelopes(
    existing_book: Mapping[str, Any] | None,
    orders: pd.DataFrame,
    *,
    run_id: str,
    account_version: int,
    account_state_sha256: str,
    registered_at: Any,
) -> dict[str, Any]:
    run = _text(run_id, "run_id")
    timestamp = _timestamp(registered_at, "registered_at")
    verified = verify_order_frame(orders) if not orders.empty else pd.DataFrame(columns=ORDER_COLUMNS)
    if not verified.empty:
        if set(verified["run_id"]) != {run}:
            raise ContractError("new order envelopes must belong to one declared run")
        if set(verified["account_version"]) != {int(account_version)} or set(
            verified["account_state_sha256"]
        ) != {str(account_state_sha256)}:
            raise ContractError("new order envelopes are not bound to the current account state")
        active_keys = verified.loc[
            verified["status"].eq("ACTIVE"), ["asset", "sleeve"]
        ]
        if active_keys.duplicated().any():
            raise ContractError("one run cannot register multiple active orders for the same asset and sleeve")
    # A research rerun is the authorized recovery path after an external account
    # event: all prior open orders are invalidated before the book is rebound.
    book = (
        normalize_order_state_book(existing_book)
        if existing_book is not None
        else empty_order_state_book(account_version, account_state_sha256)
    )
    if book["current_run_id"] == run and (
        int(book["account_version"]) != int(account_version)
        or str(book["account_state_sha256"]) != str(account_state_sha256)
    ):
        raise ContractError("a run id cannot be reused against a different account state")
    records = {item["order_id"]: copy.deepcopy(item) for item in book["orders"]}
    replacements = {
        (str(row["asset"]), str(row["sleeve"])): str(row["order_id"])
        for _, row in verified.iterrows()
        if str(row["status"]) == "ACTIVE"
    }
    for record in records.values():
        if record["run_id"] != run and record["status"] in FILLABLE_ORDER_STATES:
            replacement = replacements.get((record["asset"], record["sleeve"]))
            record.update(
                {
                    "status": "SUPERSEDED",
                    "superseded_by_order_id": replacement,
                    "status_reason": (
                        "new_run_replacement" if replacement else "new_run_without_replacement"
                    ),
                    "updated_at": timestamp,
                }
            )
    for _, order in verified.iterrows():
        order_id = str(order["order_id"])
        if order_id in records:
            if records[order_id]["order_sha256"] != str(order["order_sha256"]):
                raise ContractError(f"order id was reused with a different envelope: {order_id}")
            continue
        ordered_shares = int(order["shares"])
        records[order_id] = {
            "order_id": order_id,
            "order_sha256": str(order["order_sha256"]),
            "run_id": str(order["run_id"]),
            "asset": str(order["asset"]),
            "sleeve": str(order["sleeve"]),
            "side": str(order["side"]),
            "ordered_shares": ordered_shares,
            "filled_shares": 0,
            "remaining_shares": ordered_shares,
            "status": str(order["status"]),
            "valid_through_date": str(order["valid_through_date"]),
            "superseded_by_order_id": None,
            "status_reason": "manual_review_notice" if str(order["status"]) == "CANCELLED" else "",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    return _seal_book(
        {
            "schema_version": ORDER_STATE_BOOK_SCHEMA_VERSION,
            "account_version": int(account_version),
            "account_state_sha256": str(account_state_sha256),
            "current_run_id": run,
            "orders": list(records.values()),
        }
    )


def _transition(record: dict[str, Any], new_status: str, timestamp: str, reason: str = "") -> None:
    current = str(record["status"])
    if new_status == current:
        record["status_reason"] = reason
        record["updated_at"] = timestamp
        return
    if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ContractError(f"invalid order state transition: {current}->{new_status}")
    record["status"] = new_status
    record["status_reason"] = reason
    record["updated_at"] = timestamp


def refresh_expired_orders(book: Mapping[str, Any], as_of: Any, *, updated_at: Any) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    date = pd.Timestamp(_date(as_of, "order expiry as_of"))
    timestamp = _timestamp(updated_at, "updated_at")
    for record in normalized["orders"]:
        if record["status"] in FILLABLE_ORDER_STATES and date > pd.Timestamp(record["valid_through_date"]):
            _transition(record, "EXPIRED", timestamp, "validity_window_elapsed")
    return _seal_book(normalized)


def cancel_order(book: Mapping[str, Any], order_id: str, *, updated_at: Any, reason: str) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    target = next((item for item in normalized["orders"] if item["order_id"] == str(order_id)), None)
    if target is None:
        raise ContractError(f"order state is missing: {order_id}")
    _transition(target, "CANCELLED", _timestamp(updated_at, "updated_at"), _text(reason, "cancel reason"))
    return _seal_book(normalized)


def order_state_record(book: Mapping[str, Any], order_id: str) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    matches = [item for item in normalized["orders"] if item["order_id"] == str(order_id)]
    if len(matches) != 1:
        raise ContractError(f"order state must contain exactly one record: {order_id}")
    return copy.deepcopy(matches[0])


def apply_order_fill(book: Mapping[str, Any], order_id: str, shares: int, *, updated_at: Any) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    target = next((item for item in normalized["orders"] if item["order_id"] == str(order_id)), None)
    if target is None:
        raise ContractError(f"order state is missing: {order_id}")
    if target["status"] not in FILLABLE_ORDER_STATES:
        raise ContractError(f"order is not fillable: {order_id}/{target['status']}")
    fill_shares = _integer(shares, "fill shares", minimum=1)
    if fill_shares > int(target["remaining_shares"]):
        raise ContractError(f"aggregate fills exceed approved order shares: {order_id}")
    target["filled_shares"] += fill_shares
    target["remaining_shares"] -= fill_shares
    new_status = "FILLED" if target["remaining_shares"] == 0 else "PARTIALLY_FILLED"
    _transition(target, new_status, _timestamp(updated_at, "updated_at"), "fill_applied")
    return _seal_book(normalized)


def rebind_order_state_account(
    book: Mapping[str, Any],
    account_version: int,
    account_state_sha256: str,
    *,
    updated_at: Any | None = None,
    retain_partially_filled_order_ids: set[str] | None = None,
    invalidate_reason: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_order_state_book(book)
    if invalidate_reason is not None:
        timestamp = _timestamp(updated_at, "updated_at")
        retained = {str(value) for value in (retain_partially_filled_order_ids or set())}
        for record in normalized["orders"]:
            if record["status"] not in FILLABLE_ORDER_STATES:
                continue
            if record["status"] == "PARTIALLY_FILLED" and record["order_id"] in retained:
                continue
            _transition(record, "SUPERSEDED", timestamp, _text(invalidate_reason, "invalidate reason"))
    normalized["account_version"] = _integer(account_version, "account_version")
    normalized["account_state_sha256"] = _sha256(account_state_sha256, "account_state_sha256")
    return _seal_book(normalized)
