"""Long-term account events, NAV marking, and portfolio drawdown state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import INVESTABLE_RETURN_BASES
from .core import ContractError, load_config
from .execution import normalize_account, write_account_atomic


ROOT = Path(__file__).resolve().parents[2]
EVENT_TYPES = {"cash_dividend", "dividend_tax", "share_adjustment"}
EVENT_LEDGER_COLUMNS = [
    "event_id",
    "event_sha256",
    "event_date",
    "event_type",
    "asset",
    "name",
    "source_ref",
    "cash_before_cny",
    "cash_after_cny",
    "gross_cash_cny",
    "tax_cny",
    "net_cash_cny",
    "eligible_shares",
    "cash_per_share_cny",
    "core_shares_before",
    "core_shares_after",
    "t_shares_before",
    "t_shares_after",
    "full_target_shares_reference_before",
    "full_target_shares_reference_after",
    "status",
]
NAV_LEDGER_COLUMNS = ["date", "nav_cny", "cash_cny", "market_value_cny", "peak_nav_cny", "drawdown", "risk_state"]


def _finite(value: Any, field: str) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        raise ContractError(f"{field} must be finite")
    return float(number)


def _nonnegative(value: Any, field: str) -> float:
    number = _finite(value, field)
    if number < 0:
        raise ContractError(f"{field} must be non-negative")
    return number


def _whole(value: Any, field: str, *, positive: bool = False) -> int:
    number = _nonnegative(value, field)
    if not number.is_integer() or (positive and number <= 0):
        qualifier = "positive " if positive else ""
        raise ContractError(f"{field} must be a {qualifier}whole number")
    return int(number)


def _date(value: Any, field: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ContractError(f"{field} must be a valid date")
    return str(pd.Timestamp(timestamp).date())


def _text(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ContractError(f"{field} cannot be empty")
    return text


def _asset(value: Any) -> str:
    asset = str(value).strip().zfill(6)
    if not re.fullmatch(r"\d{6}", asset):
        raise ContractError("event asset must be a six-digit code")
    return asset


def _normalized_event(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("each account event must be an object")
    event = {
        "event_id": _text(raw.get("event_id", ""), "event_id"),
        "event_date": _date(raw.get("event_date"), "event_date"),
        "event_type": _text(raw.get("event_type", ""), "event_type").lower(),
        "asset": _asset(raw.get("asset", "")),
        "name": _text(raw.get("name", ""), "name"),
        "source_ref": _text(raw.get("source_ref", ""), "source_ref"),
    }
    if event["event_type"] not in EVENT_TYPES:
        raise ContractError(f"unsupported account event type: {event['event_type']}")
    if event["event_type"] == "cash_dividend":
        event.update(
            {
                "eligible_shares": _whole(raw.get("eligible_shares"), "eligible_shares", positive=True),
                "cash_per_share_cny": _nonnegative(raw.get("cash_per_share_cny"), "cash_per_share_cny"),
                "gross_cash_cny": _nonnegative(raw.get("gross_cash_cny"), "gross_cash_cny"),
                "tax_cny": _nonnegative(raw.get("tax_cny", 0), "tax_cny"),
            }
        )
        if event["gross_cash_cny"] <= 0 or event["tax_cny"] > event["gross_cash_cny"] + 1e-9:
            raise ContractError("cash dividend gross and tax values are invalid")
        expected_gross = event["eligible_shares"] * event["cash_per_share_cny"]
        if not math.isclose(event["gross_cash_cny"], expected_gross, abs_tol=0.05):
            raise ContractError("cash dividend does not reconcile to eligible shares and cash per share")
    elif event["event_type"] == "dividend_tax":
        event["tax_cny"] = _nonnegative(raw.get("tax_cny"), "tax_cny")
        if event["tax_cny"] <= 0:
            raise ContractError("dividend tax must be positive")
    else:
        event.update(
            {
                "core_shares_after": _whole(raw.get("core_shares_after"), "core_shares_after", positive=True),
                "t_shares_after": _whole(raw.get("t_shares_after", 0), "t_shares_after"),
                "full_target_shares_reference_after": _nonnegative(
                    raw.get("full_target_shares_reference_after"), "full_target_shares_reference_after"
                ),
            }
        )
        if event["full_target_shares_reference_after"] + 1e-9 < event["core_shares_after"]:
            raise ContractError("post-action full target reference cannot be below core shares")
    return event


def _event_hash(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_account_events(
    account: dict[str, Any], events: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_account(account, config)
    prepared = [_normalized_event(event) for event in events]
    ids = [event["event_id"] for event in prepared]
    if len(ids) != len(set(ids)):
        raise ContractError("account event batch contains duplicate event ids")
    prepared.sort(key=lambda event: (event["event_date"], event["event_id"]))
    processed = {str(item["event_id"]): str(item["sha256"]) for item in state["processed_events"]}
    results: list[dict[str, Any]] = []

    for event in prepared:
        event_id = event["event_id"]
        digest = _event_hash(event)
        if event_id in processed:
            if processed[event_id] != digest:
                raise ContractError(f"event_id payload changed after processing: {event_id}")
            results.append({"event_id": event_id, "status": "duplicate_ignored"})
            continue
        if pd.Timestamp(event["event_date"]) < pd.Timestamp(state["as_of_date"]):
            raise ContractError(f"event {event_id} predates the account state")

        holding = next((item for item in state["holdings"] if item["asset"] == event["asset"]), None)
        if holding is not None and holding["name"] != event["name"]:
            raise ContractError("account event name mismatches the holding")
        cash_before = float(state["cash_cny"])
        ledger = {column: None for column in EVENT_LEDGER_COLUMNS}
        ledger.update(
            {
                "event_id": event_id,
                "event_sha256": digest,
                "event_date": event["event_date"],
                "event_type": event["event_type"],
                "asset": event["asset"],
                "name": event["name"],
                "source_ref": event["source_ref"],
                "cash_before_cny": cash_before,
                "status": "applied",
            }
        )

        if event["event_type"] == "cash_dividend":
            gross = float(event["gross_cash_cny"])
            tax = float(event["tax_cny"])
            net = gross - tax
            state["cash_cny"] += net
            state["gross_dividend_cny"] += gross
            state["dividend_tax_cny"] += tax
            if holding is not None:
                holding["cumulative_dividend_net_cny"] += net
            ledger.update(
                {
                    "gross_cash_cny": gross,
                    "tax_cny": tax,
                    "net_cash_cny": net,
                    "eligible_shares": event["eligible_shares"],
                    "cash_per_share_cny": event["cash_per_share_cny"],
                }
            )
        elif event["event_type"] == "dividend_tax":
            tax = float(event["tax_cny"])
            if tax > state["cash_cny"] + 1e-9:
                raise ContractError(f"insufficient cash for dividend tax event {event_id}")
            state["cash_cny"] -= tax
            state["dividend_tax_cny"] += tax
            if holding is not None:
                holding["cumulative_dividend_net_cny"] -= tax
            ledger.update({"tax_cny": tax, "net_cash_cny": -tax})
        else:
            if holding is None or int(holding["core_shares"]) <= 0:
                raise ContractError("share adjustment requires an existing core holding")
            core_before = int(holding["core_shares"])
            t_before = int(holding["t_shares"])
            reference_before = float(holding["full_target_shares_reference"])
            core_after = int(event["core_shares_after"])
            t_after = int(event["t_shares_after"])
            reference_after = float(event["full_target_shares_reference_after"])
            ratio = core_after / core_before
            if (t_before == 0 and t_after != 0) or (t_before > 0 and abs(t_after - t_before * ratio) > 1.0 + 1e-9):
                raise ContractError("share adjustment must apply the same ratio to core and T shares")
            if abs(reference_after - reference_before * ratio) > 1.0 + 1e-9:
                raise ContractError("share adjustment reference does not match the core share ratio")
            core_book_cost = core_before * float(holding["core_average_cost_cny"])
            t_book_cost = t_before * float(holding["t_average_cost_cny"])
            holding["core_shares"] = core_after
            holding["core_average_cost_cny"] = core_book_cost / core_after
            holding["t_shares"] = t_after
            holding["t_average_cost_cny"] = t_book_cost / t_after if t_after else 0.0
            holding["t_open_date"] = holding["t_open_date"] if t_after else None
            holding["full_target_shares_reference"] = reference_after
            ledger.update(
                {
                    "core_shares_before": core_before,
                    "core_shares_after": core_after,
                    "t_shares_before": t_before,
                    "t_shares_after": t_after,
                    "full_target_shares_reference_before": reference_before,
                    "full_target_shares_reference_after": reference_after,
                }
            )

        state["net_dividend_cny"] = state["gross_dividend_cny"] - state["dividend_tax_cny"]
        state["as_of_date"] = str(max(pd.Timestamp(state["as_of_date"]), pd.Timestamp(event["event_date"])).date())
        ledger["cash_after_cny"] = float(state["cash_cny"])
        state["event_history"].append(ledger)
        state["processed_events"].append({"event_id": event_id, "sha256": digest})
        processed[event_id] = digest
        results.append(ledger)

    return normalize_account(state, config), results


def portfolio_risk_state(account: dict[str, Any], current_nav_cny: float, config: dict[str, Any]) -> dict[str, Any]:
    current_nav = _nonnegative(current_nav_cny, "current_nav_cny")
    if current_nav <= 0:
        raise ContractError("current NAV must be positive")
    historical_nav = [_nonnegative(item["nav_cny"], "nav_history.nav_cny") for item in account.get("nav_history", [])]
    peak = max([current_nav, *historical_nav])
    drawdown = current_nav / peak - 1.0
    review = float(config["portfolio"]["drawdown_review_trigger"])
    brake = float(config["portfolio"]["drawdown_risk_reduction_trigger"])
    if drawdown <= brake:
        state = "BRAKE"
    elif drawdown <= review:
        state = "REVIEW"
    else:
        state = "NORMAL"
    return {
        "risk_state": state,
        "current_nav_cny": current_nav,
        "peak_nav_cny": peak,
        "drawdown": drawdown,
        "history_ready": bool(account.get("nav_history")),
        "core_add_allowed": state != "BRAKE",
        "t_buy_allowed": state == "NORMAL",
        "force_t_exit": state == "BRAKE",
        "manual_core_review": state == "BRAKE",
    }


def mark_to_market(
    account: dict[str, Any], latest_prices: dict[str, float], as_of: str | pd.Timestamp, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = normalize_account(account, config)
    as_of_date = _date(as_of, "mark as_of")
    if pd.Timestamp(as_of_date) < pd.Timestamp(state["as_of_date"]):
        raise ContractError("mark date predates the account state")
    market_value = 0.0
    for holding in state["holdings"]:
        asset = holding["asset"]
        if asset not in latest_prices:
            raise ContractError(f"mark is missing a held asset price: {asset}")
        price = _nonnegative(latest_prices[asset], f"price[{asset}]")
        if price <= 0:
            raise ContractError(f"mark price must be positive: {asset}")
        market_value += (holding["core_shares"] + holding["t_shares"]) * price
    nav = float(state["cash_cny"] + market_value)
    history_without_same_date = [item for item in state["nav_history"] if item["date"] != as_of_date]
    risk_account = {**state, "nav_history": history_without_same_date}
    risk = portfolio_risk_state(risk_account, nav, config)
    mark = {
        "date": as_of_date,
        "nav_cny": nav,
        "cash_cny": float(state["cash_cny"]),
        "market_value_cny": float(market_value),
        "peak_nav_cny": float(risk["peak_nav_cny"]),
        "drawdown": float(risk["drawdown"]),
        "risk_state": risk["risk_state"],
    }
    state["nav_history"] = history_without_same_date + [mark]
    state["nav_history"].sort(key=lambda item: item["date"])
    state["as_of_date"] = str(max(pd.Timestamp(state["as_of_date"]), pd.Timestamp(as_of_date)).date())
    return normalize_account(state, config), mark


def _latest_prices(root: Path, config: dict[str, Any], account: dict[str, Any], as_of: pd.Timestamp) -> dict[str, float]:
    price_dir = root / config["data"]["price_directory"]
    latest: dict[str, float] = {}
    for holding in account["holdings"]:
        asset = holding["asset"]
        path = price_dir / f"{asset}.csv"
        if not path.exists():
            raise ContractError(f"held asset price file not found: {asset}")
        prices = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if "close" not in prices.columns:
            for candidate in ["close_adj", "adj_close"]:
                if candidate in prices.columns:
                    prices = prices.rename(columns={candidate: "close"})
                    break
        required = {"date", "close", "return_basis"}
        if not required.issubset(prices.columns):
            raise ContractError(f"mark price file has an invalid schema: {asset}")
        bases = set(prices["return_basis"].astype(str).str.lower())
        if not bases or not bases.issubset(INVESTABLE_RETURN_BASES):
            raise ContractError(f"mark price file has a non-investable return basis: {asset}")
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices[(prices["date"] <= as_of) & prices["close"].notna()].sort_values("date")
        if prices.empty:
            raise ContractError(f"held asset has no price on or before mark date: {asset}")
        latest[asset] = float(prices.iloc[-1]["close"])
    return latest


def _write_csv_atomic(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "long_hold_v4.json")
    parser.add_argument("--account", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "account.json")
    parser.add_argument(
        "--events", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "pending_account_events.json"
    )
    parser.add_argument(
        "--event-ledger", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "account_event_ledger.csv"
    )
    parser.add_argument("--nav-ledger", type=Path, default=ROOT / "portfolio_lab" / "long_hold_v4" / "nav_ledger.csv")
    parser.add_argument("--as-of", help="NAV mark date; defaults to the account date")
    parser.add_argument("--mark", action="store_true", help="Append or replace the end-of-day NAV mark")
    parser.add_argument("--apply", action="store_true", help="Persist changes; default is dry-run")
    args = parser.parse_args()

    config = load_config(args.config)
    account = normalize_account(json.loads(args.account.read_text(encoding="utf-8")), config)
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ContractError("pending account events must be a JSON list")
    state, event_results = apply_account_events(account, events, config)
    mark = None
    if args.mark:
        mark_date = pd.Timestamp(args.as_of or state["as_of_date"]).normalize()
        state, mark = mark_to_market(state, _latest_prices(ROOT, config, state, mark_date), mark_date, config)
    if args.apply:
        write_account_atomic(state, args.account)
        _write_csv_atomic(state["event_history"], EVENT_LEDGER_COLUMNS, args.event_ledger)
        _write_csv_atomic(state["nav_history"], NAV_LEDGER_COLUMNS, args.nav_ledger)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "events_in_batch": len(events),
                "event_results": len(event_results),
                "marked": mark is not None,
                "cash_cny": state["cash_cny"],
                "nav_cny": mark["nav_cny"] if mark else None,
                "risk_state": mark["risk_state"] if mark else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
