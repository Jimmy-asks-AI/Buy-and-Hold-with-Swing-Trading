"""Current-state research pipeline and fail-closed order-intent planner."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .core import (
    ContractError,
    allocate_core_targets,
    compute_price_features,
    entry_decision,
    estimate_trade_cost,
    score_universe,
    t_decision,
)
from .backtest import INVESTABLE_RETURN_BASES
from .accounting import portfolio_risk_state
from .execution import config_sha256, normalize_account
from .order_envelope import (
    ORDER_COLUMNS,
    normalize_order_state_book,
    register_order_envelopes,
    seal_order_envelope,
)


EXPECTED_AGENTS = {
    "data_steward",
    "durable_income_analyst",
    "valuation_analyst",
    "value_trap_auditor",
    "timing_analyst",
    "t_execution_analyst",
    "portfolio_risk_engineer",
    "validation_auditor",
    "orchestrator",
}

PRICE_AUDIT_COLUMNS = [
    "asset",
    "latest_price_date",
    "latest_close",
    "price_drawdown_3y",
    "price_ma20",
    "price_ma60",
    "price_stabilized",
    "price_falling_knife",
    "price_range_regime",
    "price_zscore20",
    "expected_reversion_edge",
    "price_data_age_days",
    "price_fresh",
]

PROXY_COLUMNS = [
    "asset",
    "date",
    "close",
    "drawdown_3y",
    "stabilized",
    "falling_knife",
    "range_regime",
    "zscore20",
    "data_age_days",
    "fresh",
    "return_basis",
    "promotion_allowed",
]

SETUP_COLUMNS = ["asset", "date", "close", "drawdown_3y", "stabilized", "range_regime", "zscore20", "promotion_allowed"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest(root: Path, path: Path, as_of: pd.Timestamp) -> list[str]:
    """Verify an upstream builder manifest without trusting its paths."""
    if not path.exists():
        return [f"source_manifest_missing={path}"]
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"source_manifest_invalid={path}:{exc}"]
    failures: list[str] = []
    manifest_date = pd.to_datetime(payload.get("as_of_date"), errors="coerce")
    if pd.isna(manifest_date) or pd.Timestamp(manifest_date).normalize() != as_of.normalize():
        failures.append(f"source_manifest_as_of_mismatch={path}")
    root_resolved = root.resolve()
    entries = payload.get("input_files", []) + payload.get("code_files", [])
    if not entries:
        failures.append(f"source_manifest_empty={path}")
        return failures
    for item in entries:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            failures.append(f"source_manifest_entry_invalid={path}")
            continue
        source = (root / str(item["path"])).resolve()
        if source != root_resolved and root_resolved not in source.parents:
            failures.append(f"source_manifest_path_escape={item['path']}")
        elif not source.is_file():
            failures.append(f"source_manifest_file_missing={item['path']}")
        elif file_sha256(source) != str(item["sha256"]):
            failures.append(f"source_manifest_hash_mismatch={item['path']}")
    return failures


def validate_agent_contracts(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"agent contracts not found: {path}")
    contracts = read_json(path)
    agents = contracts.get("agents")
    if not isinstance(agents, list):
        raise ContractError("agent contracts must contain an agents list")
    names = [str(agent.get("name", "")) for agent in agents]
    if len(names) != len(set(names)):
        raise ContractError("agent names must be unique")
    if set(names) != EXPECTED_AGENTS:
        raise ContractError(f"agent roster mismatch: expected={sorted(EXPECTED_AGENTS)} actual={sorted(names)}")
    required = {"fixed_inputs", "outputs", "forbidden", "acceptance"}
    for agent in agents:
        missing = sorted(required.difference(agent))
        if missing or any(not isinstance(agent[key], list) or not agent[key] for key in required):
            raise ContractError(f"incomplete agent contract: {agent.get('name')} missing={missing}")
    return contracts


def load_account(path: Path, config: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(
            f"persistent account is missing: {path}; initialize it explicitly before running research"
        )
    account = normalize_account(read_json(path), config)
    if pd.Timestamp(account["as_of_date"]) > pd.Timestamp(as_of).normalize():
        raise ContractError("account state cannot be later than research as_of date")
    return account


def load_snapshot(path: Path) -> pd.DataFrame:
    snapshot = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"asset": str})
    if "asset" in snapshot.columns:
        snapshot["asset"] = snapshot["asset"].astype(str).str.zfill(6)
    return snapshot


def _holding(account: dict[str, Any], asset: str) -> dict[str, Any]:
    return next((item for item in account.get("holdings", []) if str(item.get("asset")) == str(asset)), {})


def _sleeve_fraction(holding: dict[str, Any], sleeve: str) -> float:
    reference = float(holding.get("full_target_shares_reference", 0.0))
    return float(holding.get(f"{sleeve}_shares", 0.0)) / reference if reference > 0 else 0.0


def _holding_sessions(prices: pd.DataFrame, open_date: Any, as_of: pd.Timestamp) -> int:
    opened = pd.to_datetime(open_date, errors="coerce")
    if pd.isna(opened):
        return 0
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna().dt.normalize().drop_duplicates()
    return int(((dates >= pd.Timestamp(opened).normalize()) & (dates <= as_of)).sum())


def _load_investable_prices(path: Path, asset: str) -> pd.DataFrame:
    prices = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    rename = {}
    for base in ["open", "high", "low", "close"]:
        for candidate in [f"{base}_adj", f"adj_{base}"]:
            if base not in prices.columns and candidate in prices.columns:
                rename[candidate] = base
                break
    prices = prices.rename(columns=rename)
    if "return_basis" not in prices.columns:
        raise ContractError(f"investable price file must declare return_basis: {path}")
    bases = set(prices["return_basis"].astype(str).str.lower())
    if not bases or not bases.issubset(INVESTABLE_RETURN_BASES):
        raise ContractError(f"non-investable return_basis for {asset}: {sorted(bases)}")
    prices["asset"] = str(asset)
    return prices


def _load_timing_proxy(path: Path, asset: str) -> pd.DataFrame:
    prices = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = ["date", "open", "high", "low", "close"]
    prices = prices.dropna(subset=[col for col in required if col in prices.columns]).copy()
    prices["asset"] = str(asset)
    prices["return_basis"] = "price_index_proxy"
    return prices


def account_value(account: dict[str, Any], latest_prices: dict[str, float]) -> float:
    value = float(account.get("cash_cny", 0.0))
    for holding in account.get("holdings", []):
        asset = str(holding["asset"])
        price = latest_prices.get(asset)
        if price is None:
            raise ContractError(f"missing current price for held asset: {asset}")
        if not math.isfinite(float(price)) or float(price) <= 0:
            raise ContractError(f"invalid current price for held asset: {asset}")
        shares = float(holding.get("core_shares", 0.0)) + float(holding.get("t_shares", 0.0))
        value += shares * price
    if not math.isfinite(value) or value <= 0:
        raise ContractError("account NAV must be finite and positive")
    return value


def plan_orders(
    targets: pd.DataFrame,
    account: dict[str, Any],
    latest_prices: dict[str, float],
    config: dict[str, Any],
    signal_date: str | pd.Timestamp | None = None,
    *,
    run_id: str,
    run_manifest_sha256: str,
    trade_calendar_sha256: str,
    risk_state_at_signal: str,
) -> pd.DataFrame:
    """Create indicative next-open intents; core sales are never automatic."""
    if targets.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    signal_value = signal_date if signal_date is not None else account.get("as_of_date")
    signal_ts = pd.to_datetime(signal_value, errors="coerce")
    if pd.isna(signal_ts):
        raise ContractError("order planning requires a valid signal_date")
    signal_ts = pd.Timestamp(signal_ts).normalize()
    valid_from = signal_ts + pd.Timedelta(days=1)
    valid_through = signal_ts + pd.Timedelta(days=int(config["execution"]["order_valid_calendar_days"]))
    active_config_sha256 = config_sha256(config)
    nav = account_value(account, latest_prices)
    minimum_cash_cny = nav * float(config["portfolio"]["minimum_cash_weight"])
    cash_available = max(0.0, float(account.get("cash_cny", 0.0)) - minimum_cash_cny)
    current_t_value = sum(
        float(holding.get("t_shares", 0.0)) * latest_prices[str(holding["asset"])]
        for holding in account.get("holdings", [])
    )
    remaining_t_budget = max(
        0.0,
        nav * float(config["t_strategy"]["portfolio_t_weight_cap"]) - current_t_value,
    )
    rows: list[dict[str, Any]] = []

    for _, row in targets.sort_values(["final_score", "asset"], ascending=[False, True]).iterrows():
        asset = str(row["asset"])
        if asset not in latest_prices:
            continue
        price = float(latest_prices[asset])
        holding = _holding(account, asset)
        core_shares = int(holding.get("core_shares", 0))
        t_shares = int(holding.get("t_shares", 0))
        lot = int(config["portfolio"][f"{str(row['asset_type']).lower()}_lot_size"])
        full_weight = float(row.get("full_target_weight", row.get("target_core_weight", 0.0)))
        computed_reference = math.floor(nav * max(0.0, full_weight) / (price * lot)) * lot
        existing_reference = float(holding.get("full_target_shares_reference", 0.0))
        base_reference = max(float(core_shares), existing_reference, float(computed_reference))
        core_fraction = core_shares / base_reference if base_reference > 0 else 0.0
        t_holding_sessions = int(row.get("t_holding_sessions", 0) or 0)

        def add_intent(
            sleeve: str,
            side: str,
            shares: int,
            notional: float,
            estimated_cost: float,
            status: str,
            reason: str,
            *,
            manual_approval_required: bool = False,
            risk_override_allowed: bool = False,
        ) -> None:
            reference = (
                max(base_reference, float(core_shares + shares))
                if sleeve == "core" and side == "buy"
                else base_reference
            )
            identity = f"{run_id}|{asset}|{sleeve}|{side}|{len(rows)}"
            identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            lifecycle_status = "ACTIVE" if side in {"buy", "sell"} else "CANCELLED"
            rows.append(
                seal_order_envelope(
                    {
                        "order_schema_version": 1,
                        "order_id": f"LHV4-{signal_ts:%Y%m%d}-{asset}-{sleeve}-{side}-{identity_digest}",
                        "run_id": run_id,
                        "run_manifest_sha256": run_manifest_sha256,
                        "config_sha256": active_config_sha256,
                        "trade_calendar_sha256": trade_calendar_sha256,
                        "account_version": int(account["state_version"]),
                        "account_state_sha256": str(account["state_sha256"]),
                        "signal_date": str(signal_ts.date()),
                        "valid_from_date": str(valid_from.date()),
                        "valid_through_date": str(valid_through.date()),
                        "asset": asset,
                        "name": row.get("name", ""),
                        "asset_type": str(row["asset_type"]).lower(),
                        "sector": str(row.get("sector", "")).lower(),
                        "sleeve": sleeve,
                        "side": side,
                        "shares": int(shares),
                        "indicative_price": price,
                        "max_price_deviation_bps": float(config["execution"]["max_price_deviation_bps"]),
                        "notional": float(notional),
                        "estimated_cost": float(estimated_cost),
                        "target_core_weight": float(row.get("target_core_weight", 0.0)),
                        "target_t_weight_cap": float(row.get("target_t_weight_cap", 0.0)),
                        "full_target_weight": full_weight,
                        "full_target_shares_reference": float(reference),
                        "core_fraction_at_signal": float(core_fraction),
                        "t_holding_sessions": t_holding_sessions,
                        "risk_state_at_signal": str(risk_state_at_signal).upper(),
                        "risk_override_allowed": bool(risk_override_allowed),
                        "manual_approval_required": bool(manual_approval_required),
                        "status": lifecycle_status,
                        "intent_status": status,
                        "reason": str(reason),
                    }
                )
            )

        review_reasons: list[str] = []
        if str(row.get("entry_action", "")) == "REVIEW_CORE" and core_shares > 0:
            review_reasons.append(str(row.get("hard_veto_reasons", "core_quality_or_data_veto")))
        if bool(row.get("manual_risk_review_required", False)) and core_shares + t_shares > 0:
            review_reasons.append(str(row.get("portfolio_risk_reasons", "portfolio_limit_breach")))
        if review_reasons:
            add_intent(
                "core",
                "review",
                0,
                0.0,
                0.0,
                "MANUAL_RISK_REVIEW_NO_AUTOMATIC_CORE_SELL",
                ";".join(dict.fromkeys(reason for reason in review_reasons if reason)),
                manual_approval_required=True,
            )

        desired_core = nav * float(row["target_core_weight"])
        current_core = core_shares * price
        core_delta = desired_core - current_core
        if core_delta >= price * lot - 1e-9 and str(row["entry_action"]).startswith("BUILD"):
            per_lot_notional = price * lot
            per_lot_cost = estimate_trade_cost(per_lot_notional, "buy", row["asset_type"], config)["total_cost"]
            lots = math.floor(min(core_delta, cash_available) / (per_lot_notional + per_lot_cost))
            shares = int(max(0, lots) * lot)
            if shares:
                notional = shares * price
                cost = estimate_trade_cost(notional, "buy", row["asset_type"], config)["total_cost"]
                cash_available -= notional + cost
                add_intent(
                    "core",
                    "buy",
                    shares,
                    notional,
                    cost,
                    "RESEARCH_INTENT_REPRICE_NEXT_OPEN",
                    str(row["entry_action"]),
                )
        elif core_delta < -price * lot and not review_reasons:
            add_intent(
                "core",
                "review",
                0,
                0.0,
                0.0,
                "MANUAL_RISK_REVIEW_NO_AUTOMATIC_CORE_SELL",
                str(row.get("hard_veto_reasons", "target_below_current_core")),
                manual_approval_required=True,
            )

        if (
            row.get("t_action") == "BUY_T_NEXT_OPEN"
            and core_fraction + 1e-9 >= float(config["t_strategy"]["core_fraction_required"])
        ):
            desired_t = nav * min(float(row["target_t_weight_cap"]), float(config["t_strategy"]["portfolio_t_weight_cap"]))
            t_share_limit = math.floor(
                base_reference * float(config["t_strategy"]["t_sleeve_fraction_of_full_position"]) / lot
            ) * lot
            reference_room = max(0.0, (t_share_limit - t_shares) * price)
            delta = min(max(0.0, desired_t - t_shares * price), remaining_t_budget, reference_room)
            per_lot_notional = price * lot
            per_lot_cost = estimate_trade_cost(per_lot_notional, "buy", row["asset_type"], config)["total_cost"]
            lots = math.floor(min(delta, cash_available) / (per_lot_notional + per_lot_cost))
            shares = int(max(0, lots) * lot)
            if shares:
                notional = shares * price
                cost = estimate_trade_cost(notional, "buy", row["asset_type"], config)["total_cost"]
                cash_available -= notional + cost
                remaining_t_budget -= notional
                add_intent(
                    "t",
                    "buy",
                    shares,
                    notional,
                    cost,
                    "RESEARCH_INTENT_REPRICE_NEXT_OPEN",
                    str(row.get("t_reasons", "")),
                )
        elif row.get("t_action") == "SELL_T_NEXT_OPEN" and t_shares > 0:
            notional = t_shares * price
            cost = estimate_trade_cost(notional, "sell", row["asset_type"], config)["total_cost"]
            t_reason = str(row.get("t_reasons", ""))
            add_intent(
                "t",
                "sell",
                t_shares,
                notional,
                cost,
                "RESEARCH_INTENT_REPRICE_NEXT_OPEN",
                t_reason,
                risk_override_allowed=t_reason
                in {"t_strategy_disabled", "durable_core_invalidated", "portfolio_drawdown_brake"},
            )
    if cash_available < -1e-6:
        raise AssertionError("order plan violates minimum cash reserve")
    if current_t_value + sum(row["notional"] for row in rows if row["sleeve"] == "t" and row["side"] == "buy") > (
        nav * float(config["t_strategy"]["portfolio_t_weight_cap"]) + 1e-6
    ):
        raise AssertionError("order plan violates portfolio T sleeve cap")
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def _agent_log(
    data_ready: bool,
    scored: pd.DataFrame,
    orders: pd.DataFrame,
    account: dict[str, Any],
    portfolio_risk: dict[str, Any] | None,
) -> pd.DataFrame:
    eligible = int(scored.get("durable_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not scored.empty else 0
    t_order_count = int((orders.get("sleeve", pd.Series(dtype=str)) == "t").sum())
    has_holdings = bool(account.get("holdings"))
    has_t_holdings = any(int(item.get("t_shares", 0)) > 0 for item in account.get("holdings", []))
    risk_review = bool(
        not scored.empty
        and scored.get("manual_risk_review_required", pd.Series(False, index=scored.index)).fillna(False).astype(bool).any()
    )
    drawdown_state = str((portfolio_risk or {}).get("risk_state", "UNKNOWN"))
    portfolio_status = (
        "blocked"
        if not data_ready and has_holdings
        else "brake"
        if drawdown_state == "BRAKE"
        else "review"
        if risk_review or drawdown_state == "REVIEW"
        else "pass"
    )
    orchestrator_status = (
        "blocked"
        if not data_ready
        else "research_intents"
        if not orders.empty
        else "hold"
        if has_holdings
        else "cash"
    )
    decisions = [
        ("data_steward", "pass" if data_ready else "blocked", "PIT snapshot and investable-price freshness gate"),
        ("durable_income_analyst", "pass" if eligible else "blocked", f"eligible_assets={eligible}"),
        ("valuation_analyst", "pass" if eligible else "blocked", "own-history and sector-relative valuation required"),
        ("value_trap_auditor", "pass" if eligible else "blocked", "hard veto cannot be overridden"),
        ("timing_analyst", "pass" if data_ready else "blocked", "deep drawdown plus stabilization on each asset"),
        (
            "t_execution_analyst",
            "pass" if t_order_count else "holding" if has_t_holdings else "idle",
            "T requires established core, lifecycle, settlement, and cost hurdle",
        ),
        ("portfolio_risk_engineer", portfolio_status, "cash, combined asset, sector, and sleeve caps enforced"),
        ("validation_auditor", "blocked" if not data_ready else "observation", "promotion requires PIT total-return walk-forward evidence"),
        ("orchestrator", orchestrator_status, "cannot override upstream vetoes"),
    ]
    return pd.DataFrame(decisions, columns=["agent", "status", "decision"])


def run_current(root: Path, config: dict[str, Any], as_of: str | pd.Timestamp) -> dict[str, Path]:
    as_of_ts = pd.Timestamp(as_of).normalize()
    run_at = datetime.now().isoformat(timespec="microseconds")
    run_id = f"LHV4-{as_of_ts:%Y%m%d}-{datetime.now():%Y%m%dT%H%M%S%f}"
    active_config_sha256 = config_sha256(config)
    data_cfg = config["data"]
    output = root / data_cfg["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / data_cfg["snapshot_path"]
    account_path = root / data_cfg["account_path"]
    order_state_path = root / data_cfg["order_state_path"]
    trade_calendar_path = root / data_cfg["trade_calendar_path"]
    agent_contracts_path = root / data_cfg["agent_contracts_path"]
    price_dir = root / data_cfg["price_directory"]
    validate_agent_contracts(agent_contracts_path)
    account = load_account(account_path, config, as_of_ts)

    snapshot_ready = snapshot_path.exists()
    scored = pd.DataFrame()
    latest_prices: dict[str, float] = {}
    feature_by_asset: dict[str, pd.Series] = {}
    feature_history_by_asset: dict[str, pd.DataFrame] = {}
    required_price_status: dict[str, bool] = {}
    price_gate_failures: list[dict[str, str]] = []
    held_assets = {str(item["asset"]) for item in account.get("holdings", [])}
    portfolio_risk: dict[str, Any] | None = None
    snapshot_fresh = False
    input_paths: list[Path] = [agent_contracts_path]
    trade_calendar_ready = trade_calendar_path.is_file()
    trade_calendar_sha256 = file_sha256(trade_calendar_path) if trade_calendar_ready else "0" * 64
    if trade_calendar_ready:
        input_paths.append(trade_calendar_path)
    source_manifest_failures: list[str] = []
    for relative_path in data_cfg.get("source_manifest_paths", []):
        source_manifest_path = root / relative_path
        source_manifest_failures.extend(verify_source_manifest(root, source_manifest_path, as_of_ts))
        if source_manifest_path.exists():
            input_paths.append(source_manifest_path)
    if account_path.exists():
        input_paths.append(account_path)
    if snapshot_ready:
        snapshot = load_snapshot(snapshot_path)
        input_paths.append(snapshot_path)
        scored = score_universe(snapshot, as_of_ts, config)
        snapshot_assets = set(scored["asset"].astype(str))
        required_price_status = {
            str(row["asset"]): True for _, row in scored.iterrows() if row.get("data_gate_status") == "pass"
        }
        for held_asset in held_assets:
            required_price_status.setdefault(held_asset, True)
            if held_asset not in snapshot_assets:
                required_price_status[held_asset] = False
                price_gate_failures.append({"asset": held_asset, "reason": "held_asset_missing_from_snapshot"})
        snapshot_reasons = scored.get("data_gate_reasons", pd.Series(dtype=str)).fillna("").astype(str)
        snapshot_fresh = not snapshot_reasons.str.contains(
            r"(?:invalid_as_of_date|future_snapshot_date|stale_snapshot)", regex=True
        ).any()
        decision_rows: list[dict[str, Any]] = []
        price_audit_rows: list[dict[str, Any]] = []
        for _, row in scored.iterrows():
            asset = str(row["asset"])
            price_required = asset in required_price_status
            holding = _holding(account, asset)
            current_core_fraction = _sleeve_fraction(holding, "core")
            if holding and (
                str(holding.get("asset_type", "")).lower() != str(row.get("asset_type", "")).lower()
                or str(holding.get("sector", "")).lower() != str(row.get("sector", "")).lower()
            ):
                required_price_status[asset] = False
                price_gate_failures.append({"asset": asset, "reason": "account_snapshot_metadata_mismatch"})
            price_path = price_dir / f"{asset}.csv"
            if not price_path.exists():
                if price_required:
                    required_price_status[asset] = False
                    price_gate_failures.append({"asset": asset, "reason": "missing_investable_price_file"})
                decision_rows.append(
                    {
                        "asset": asset,
                        "entry_action": "REVIEW_CORE" if current_core_fraction > 0 else "KEEP_CASH",
                        "target_core_fraction": current_core_fraction,
                        "t_enabled": False,
                        "entry_reasons": "missing_investable_price_file",
                    }
                )
                continue
            try:
                raw_prices = _load_investable_prices(price_path, asset)
                features = compute_price_features(raw_prices, config)
            except (ContractError, OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                if price_required:
                    required_price_status[asset] = False
                    price_gate_failures.append({"asset": asset, "reason": f"price_contract_blocked:{exc}"})
                decision_rows.append(
                    {
                        "asset": asset,
                        "entry_action": "REVIEW_CORE" if current_core_fraction > 0 else "KEEP_CASH",
                        "target_core_fraction": current_core_fraction,
                        "t_enabled": False,
                        "entry_reasons": f"price_contract_blocked:{exc}",
                    }
                )
                continue
            features = features[features["date"] <= as_of_ts]
            if features.empty:
                if price_required:
                    required_price_status[asset] = False
                    price_gate_failures.append({"asset": asset, "reason": "no_price_rows_on_or_before_as_of"})
                decision_rows.append(
                    {
                        "asset": asset,
                        "entry_action": "REVIEW_CORE" if current_core_fraction > 0 else "KEEP_CASH",
                        "target_core_fraction": current_core_fraction,
                        "t_enabled": False,
                        "entry_reasons": "no_price_rows_on_or_before_as_of",
                    }
                )
                continue
            latest = features.iloc[-1]
            latest_prices[asset] = float(latest["close"])
            feature_by_asset[asset] = latest
            feature_history_by_asset[asset] = features
            fresh = 0 <= (as_of_ts - pd.Timestamp(latest["date"]).normalize()).days <= int(
                config["model"]["max_price_age_days"]
            )
            price_audit_rows.append(
                {
                    "asset": asset,
                    "latest_price_date": latest["date"],
                    "latest_close": latest["close"],
                    "price_drawdown_3y": latest["drawdown_3y"],
                    "price_ma20": latest["ma20"],
                    "price_ma60": latest["ma60"],
                    "price_stabilized": bool(latest["stabilized"]),
                    "price_falling_knife": bool(latest["falling_knife"]),
                    "price_range_regime": bool(latest["range_regime"]),
                    "price_zscore20": latest["zscore20"],
                    "expected_reversion_edge": latest["expected_reversion_edge"],
                    "price_data_age_days": int((as_of_ts - pd.Timestamp(latest["date"]).normalize()).days),
                    "price_fresh": fresh,
                }
            )
            if price_required:
                required_price_status[asset] = bool(required_price_status.get(asset, True) and fresh)
                if not fresh:
                    price_gate_failures.append({"asset": asset, "reason": "stale_investable_price"})
            input_paths.append(price_path)
            decision_rows.append(
                {"asset": asset, **entry_decision(row, latest, current_core_fraction, as_of_ts, config)}
            )
        scored = scored.merge(pd.DataFrame(decision_rows), on="asset", how="left")
        scored = scored.merge(pd.DataFrame(price_audit_rows, columns=PRICE_AUDIT_COLUMNS), on="asset", how="left")
        scored["entry_action"] = scored["entry_action"].fillna("KEEP_CASH")
        scored["target_core_fraction"] = scored["target_core_fraction"].fillna(0.0)
        portfolio_valuation_ready = held_assets.issubset(latest_prices)
        portfolio_nav = account_value(account, latest_prices) if portfolio_valuation_ready else None
        if portfolio_nav is not None:
            portfolio_risk = portfolio_risk_state(account, portfolio_nav, config)
            scored["current_core_weight"] = scored["asset"].map(
                lambda asset: float(_holding(account, str(asset)).get("core_shares", 0)) * latest_prices.get(str(asset), 0.0) / portfolio_nav
            )
            scored["current_t_weight"] = scored["asset"].map(
                lambda asset: float(_holding(account, str(asset)).get("t_shares", 0)) * latest_prices.get(str(asset), 0.0) / portfolio_nav
            )
            scored["entry_action_before_portfolio_risk"] = scored["entry_action"]
            if portfolio_risk["risk_state"] == "BRAKE":
                build_mask = scored["entry_action"].astype(str).str.startswith("BUILD")
                for index in scored.index[build_mask]:
                    holding = _holding(account, str(scored.at[index, "asset"]))
                    current_fraction = _sleeve_fraction(holding, "core")
                    scored.at[index, "entry_action"] = "HOLD_CORE" if current_fraction > 0 else "KEEP_CASH"
                    scored.at[index, "target_core_fraction"] = current_fraction
                    prior_value = scored.at[index, "entry_reasons"]
                    prior = "" if pd.isna(prior_value) else str(prior_value)
                    scored.at[index, "entry_reasons"] = ";".join(
                        reason for reason in [prior, "portfolio_drawdown_brake"] if reason
                    )
            scored = allocate_core_targets(scored, config)
            if portfolio_risk["risk_state"] == "BRAKE":
                held_mask = (scored["current_core_weight"] + scored["current_t_weight"]) > 1e-12
                scored.loc[held_mask, "manual_risk_review_required"] = True
                scored.loc[held_mask, "portfolio_risk_reasons"] = scored.loc[
                    held_mask, "portfolio_risk_reasons"
                ].map(lambda value: ";".join(reason for reason in [str(value), "portfolio_drawdown_brake"] if reason))
        else:
            for column in [
                "current_core_weight",
                "current_t_weight",
                "full_target_weight",
                "target_core_weight",
                "target_t_weight_cap",
                "target_cash_weight",
            ]:
                scored[column] = float("nan")
            scored["manual_risk_review_required"] = True
            scored["portfolio_risk_reasons"] = "portfolio_valuation_unavailable"

        t_rows: list[dict[str, Any]] = []
        for _, row in scored.iterrows():
            asset = str(row["asset"])
            if asset not in feature_by_asset:
                t_rows.append({"asset": asset, "t_action": "NO_T", "target_t_fraction": 0.0, "t_reasons": "missing_price"})
                continue
            holding = _holding(account, asset)
            core_fraction = _sleeve_fraction(holding, "core")
            current_t_fraction = _sleeve_fraction(holding, "t")
            sessions = _holding_sessions(feature_history_by_asset[asset], holding.get("t_open_date"), as_of_ts)
            t_result = t_decision(
                row,
                feature_by_asset[asset],
                core_fraction,
                current_t_fraction,
                sessions,
                as_of_ts,
                config,
            )
            if portfolio_risk is not None and portfolio_risk["risk_state"] == "REVIEW" and t_result["t_action"] == "BUY_T_NEXT_OPEN":
                t_result = {
                    "t_action": "NO_T",
                    "target_t_fraction": current_t_fraction,
                    "t_reasons": "portfolio_drawdown_review",
                }
            elif portfolio_risk is not None and portfolio_risk["risk_state"] == "BRAKE":
                t_result = {
                    "t_action": "SELL_T_NEXT_OPEN" if current_t_fraction > 0 else "NO_T",
                    "target_t_fraction": 0.0,
                    "t_reasons": "portfolio_drawdown_brake",
                }
            t_rows.append(
                {
                    "asset": asset,
                    "t_holding_sessions": sessions,
                    **t_result,
                }
            )
        scored = scored.merge(pd.DataFrame(t_rows), on="asset", how="left")

    price_required_count = len(required_price_status)
    investable_prices_fresh = bool(required_price_status) and all(required_price_status.values())
    portfolio_valuation_ready = held_assets.issubset(latest_prices)
    data_ready = (
        snapshot_ready
        and snapshot_fresh
        and investable_prices_fresh
        and portfolio_valuation_ready
        and trade_calendar_ready
        and not source_manifest_failures
        and not scored.empty
    )
    if portfolio_risk is None and portfolio_valuation_ready:
        portfolio_risk = portfolio_risk_state(account, account_value(account, latest_prices), config)
    provisional_manifest_sha256 = "0" * 64
    risk_state_at_signal = str((portfolio_risk or {}).get("risk_state", "UNKNOWN"))
    orders = (
        plan_orders(
            scored,
            account,
            latest_prices,
            config,
            as_of_ts,
            run_id=run_id,
            run_manifest_sha256=provisional_manifest_sha256,
            trade_calendar_sha256=trade_calendar_sha256,
            risk_state_at_signal=risk_state_at_signal,
        )
        if data_ready
        else pd.DataFrame(columns=ORDER_COLUMNS)
    )

    proxy_rows: list[dict[str, Any]] = []
    setup_rows: list[pd.DataFrame] = []
    proxy_fresh_flags: list[bool] = []
    for asset, rel_path in data_cfg.get("timing_proxy_paths", {}).items():
        path = root / rel_path
        if not path.exists():
            continue
        input_paths.append(path)
        features = compute_price_features(_load_timing_proxy(path, asset), config)
        features = features[features["date"] <= as_of_ts]
        if features.empty:
            continue
        latest = features.iloc[-1]
        age = int((as_of_ts - pd.Timestamp(latest["date"]).normalize()).days)
        fresh = 0 <= age <= int(config["model"]["max_price_age_days"])
        proxy_fresh_flags.append(fresh)
        proxy_rows.append(
            {
                "asset": asset,
                "date": latest["date"],
                "close": latest["close"],
                "drawdown_3y": latest["drawdown_3y"],
                "stabilized": bool(latest["stabilized"]),
                "falling_knife": bool(latest["falling_knife"]),
                "range_regime": bool(latest["range_regime"]),
                "zscore20": latest["zscore20"],
                "data_age_days": age,
                "fresh": fresh,
                "return_basis": "price_index_proxy",
                "promotion_allowed": False,
            }
        )
        setup = features[(features["drawdown_3y"] <= -0.15) & features["stabilized"]].tail(25).copy()
        if not setup.empty:
            setup["asset"] = asset
            setup["promotion_allowed"] = False
            setup_rows.append(setup[["asset", "date", "close", "drawdown_3y", "stabilized", "range_regime", "zscore20", "promotion_allowed"]])

    proxies = pd.DataFrame(proxy_rows, columns=PROXY_COLUMNS)
    setups = pd.concat(setup_rows, ignore_index=True) if setup_rows else pd.DataFrame(columns=SETUP_COLUMNS)
    proxies_fresh = bool(proxy_fresh_flags) and all(proxy_fresh_flags)
    eligible_count = int(scored.get("durable_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not scored.empty else 0
    blocked_asset_count = int((scored.get("data_gate_status", pd.Series(dtype=str)) != "pass").sum()) if not scored.empty else 0
    known_account_value = account_value(account, latest_prices) if portfolio_valuation_ready else None
    if portfolio_risk is None and known_account_value is not None:
        portfolio_risk = portfolio_risk_state(account, known_account_value, config)
    agents = _agent_log(data_ready, scored, orders, account, portfolio_risk)
    if not data_ready:
        system_status = "PORTFOLIO_DATA_BLOCKED" if account.get("holdings") else "CASH_DATA_BLOCKED"
    elif not orders.empty:
        system_status = "RESEARCH_INTENTS_AVAILABLE_NOT_BROKER_ORDERS"
    elif account.get("holdings"):
        system_status = "HOLDINGS_NO_ACTION"
    elif eligible_count == 0:
        system_status = "CASH_NO_DURABLE_CANDIDATE"
    else:
        system_status = "CASH_NO_ENTRY_SIGNAL"

    readiness_warnings: list[str] = []
    if not proxies_fresh:
        readiness_warnings.append("timing_proxy_data_stale_or_missing")
    if portfolio_risk is not None and portfolio_risk["risk_state"] != "NORMAL":
        readiness_warnings.append(f"portfolio_drawdown_{portfolio_risk['risk_state'].lower()}")
    if account.get("holdings") and portfolio_risk is not None and not portfolio_risk["history_ready"]:
        readiness_warnings.append("portfolio_nav_history_not_initialized")
    account_report = {
        "as_of_date": str(as_of_ts.date()),
        "cash_cny": float(account.get("cash_cny", 0.0)),
        "holding_count": len(account.get("holdings", [])),
        "known_account_value_cny": known_account_value,
        "gross_dividend_cny": float(account.get("gross_dividend_cny", 0.0)),
        "dividend_tax_cny": float(account.get("dividend_tax_cny", 0.0)),
        "net_dividend_cny": float(account.get("net_dividend_cny", 0.0)),
        "portfolio_peak_nav_cny": (portfolio_risk or {}).get("peak_nav_cny"),
        "portfolio_drawdown": (portfolio_risk or {}).get("drawdown"),
        "portfolio_risk_state": (portfolio_risk or {}).get("risk_state", "UNKNOWN"),
        "portfolio_nav_history_ready": (portfolio_risk or {}).get("history_ready", False),
        "system_status": system_status,
        "orders_generated": int(len(orders)),
        "new_risk_is_blocked_when_data_blocked": True,
    }
    readiness = {
        "system_status": system_status,
        "snapshot_path": str(snapshot_path),
        "snapshot_exists": snapshot_ready,
        "snapshot_fresh": snapshot_fresh,
        "fresh_investable_prices": investable_prices_fresh,
        "portfolio_valuation_ready": portfolio_valuation_ready,
        "portfolio_risk_state": (portfolio_risk or {}).get("risk_state", "UNKNOWN"),
        "portfolio_drawdown": (portfolio_risk or {}).get("drawdown"),
        "portfolio_nav_history_ready": (portfolio_risk or {}).get("history_ready", False),
        "price_required_asset_count": price_required_count,
        "fresh_price_asset_count": int(sum(required_price_status.values())),
        "price_gate_failures": price_gate_failures,
        "fresh_timing_proxies": proxies_fresh,
        "trading_calendar_ready": trade_calendar_ready,
        "source_manifests_ready": not source_manifest_failures,
        "source_manifest_failures": source_manifest_failures,
        "eligible_asset_count": eligible_count,
        "blocked_asset_count": blocked_asset_count,
        "order_intent_count": int(len(orders)),
        "blocking_reasons": [
            reason
            for condition, reason in [
                (not snapshot_ready, "missing_pit_research_snapshot"),
                (snapshot_ready and not snapshot_fresh, "pit_research_snapshot_stale_or_invalid"),
                (snapshot_ready and not scored.empty and price_required_count == 0, "no_data_gate_pass_assets"),
                (snapshot_ready and not investable_prices_fresh, "investable_price_data_stale_or_missing"),
                (not portfolio_valuation_ready, "held_asset_valuation_unavailable"),
                (not trade_calendar_ready, "trading_calendar_missing"),
                (bool(source_manifest_failures), "source_manifest_integrity_failed"),
                (scored.empty, "no_scored_assets"),
            ]
            if condition
        ],
        "warnings": readiness_warnings,
        "investment_boundary": "research_only; no broker connection; manual review required",
    }

    paths = {
        "readiness": output / "readiness.json",
        "account": output / "account_summary.json",
        "candidates": output / "candidate_decisions.csv",
        "orders": output / "order_intents.csv",
        "proxies": output / "timing_proxy_latest.csv",
        "setups": output / "timing_proxy_historical_setups.csv",
        "agents": output / "agent_decision_log.csv",
        "manifest": output / "run_manifest.json",
        "order_state": order_state_path,
    }
    write_json(readiness, paths["readiness"])
    write_json(account_report, paths["account"])
    scored.to_csv(paths["candidates"], index=False, encoding="utf-8-sig")
    proxies.to_csv(paths["proxies"], index=False, encoding="utf-8-sig")
    setups.to_csv(paths["setups"], index=False, encoding="utf-8-sig")
    agents.to_csv(paths["agents"], index=False, encoding="utf-8-sig")
    manifest = {
        "model": config["model"],
        "run_id": run_id,
        "run_at": run_at,
        "as_of_date": str(as_of_ts.date()),
        "system_status": system_status,
        "config_sha256": active_config_sha256,
        "account_version": int(account["state_version"]),
        "account_state_sha256": str(account["state_sha256"]),
        "order_envelope_schema_version": 1,
        "input_files": [
            {"path": str(path.relative_to(root)), "sha256": file_sha256(path)} for path in sorted(set(input_paths))
        ],
        "code_files": [
            {"path": str(path.relative_to(Path(__file__).resolve().parents[2])), "sha256": file_sha256(path)}
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        ],
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
        "outputs": {
            key: str(path.relative_to(root))
            for key, path in paths.items()
            if key not in {"manifest", "order_state"}
        },
    }
    write_json(manifest, paths["manifest"])
    manifest_sha256 = file_sha256(paths["manifest"])
    final_orders = (
        plan_orders(
            scored,
            account,
            latest_prices,
            config,
            as_of_ts,
            run_id=run_id,
            run_manifest_sha256=manifest_sha256,
            trade_calendar_sha256=trade_calendar_sha256,
            risk_state_at_signal=risk_state_at_signal,
        )
        if data_ready
        else pd.DataFrame(columns=ORDER_COLUMNS)
    )
    if len(final_orders) != len(orders):
        raise AssertionError("sealing the run manifest changed the order plan")
    final_orders.to_csv(paths["orders"], index=False, encoding="utf-8-sig")
    existing_order_state = read_json(order_state_path) if order_state_path.exists() else None
    if existing_order_state is not None:
        normalize_order_state_book(existing_order_state)
    next_order_state = register_order_envelopes(
        existing_order_state,
        final_orders,
        run_id=run_id,
        account_version=int(account["state_version"]),
        account_state_sha256=str(account["state_sha256"]),
        registered_at=run_at,
    )
    write_json(next_order_state, order_state_path)
    return paths
