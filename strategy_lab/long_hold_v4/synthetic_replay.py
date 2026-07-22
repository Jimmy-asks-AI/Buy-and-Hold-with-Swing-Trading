"""Deterministic, offline paper-trading replay built only from synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .accounting import mark_to_market
from .core import ContractError, load_config
from .execution import apply_fills, config_sha256, seal_account_state
from .invariants import validate_paper_account_invariants
from .order_envelope import rebind_order_state_account, register_order_envelopes
from .pipeline import plan_orders


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "synthetic-replay-v1"
SIGNAL_DATE = "2026-01-05"
FILL_DATE = "2026-01-06"
SYNTHETIC_NOTICE = "SYNTHETIC_ONLY_NOT_REAL_SECURITY_OR_INVESTMENT_ADVICE"
CHAIN_FILES = [
    "snapshot.csv",
    "asset_market_states.csv",
    "trading_calendar.csv",
    "candidate_decisions.csv",
    "target_weights.csv",
    "orders.csv",
    "fills.csv",
    "account.json",
    "ledger.csv",
    "order_state.json",
    "nav.csv",
]


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = pd.DataFrame(
        [
            {
                "as_of_date": SIGNAL_DATE,
                "available_date": SIGNAL_DATE,
                "asset": "000000",
                "name": "合成股票甲（非真实证券）",
                "asset_type": "stock",
                "sector": "bank",
                "is_tradeable": True,
                "trade_status": "OPEN",
                "current_pe": 6.0,
                "current_pb": 0.7,
                "roe_mean_5y": 0.12,
                "dividend_yield": 0.05,
            },
            {
                "as_of_date": SIGNAL_DATE,
                "available_date": SIGNAL_DATE,
                "asset": "999999",
                "name": "合成ETF甲（非真实证券）",
                "asset_type": "etf",
                "sector": "dividend_index",
                "is_tradeable": True,
                "trade_status": "OPEN",
                "current_pe": 8.0,
                "current_pb": 0.9,
                "roe_mean_5y": 0.10,
                "dividend_yield": 0.04,
            },
        ]
    )
    states = pd.DataFrame(
        [
            {"date": "2026-01-02", "asset": "000000", "market_state": "SUSPENDED"},
            {"date": "2026-01-02", "asset": "999999", "market_state": "OPEN"},
            {"date": SIGNAL_DATE, "asset": "000000", "market_state": "OPEN"},
            {"date": SIGNAL_DATE, "asset": "999999", "market_state": "OPEN"},
            {"date": FILL_DATE, "asset": "000000", "market_state": "OPEN"},
            {"date": FILL_DATE, "asset": "999999", "market_state": "OPEN"},
        ]
    )
    calendar = pd.DataFrame(
        [
            {"date": "2026-01-02", "is_open": True},
            {"date": SIGNAL_DATE, "is_open": True},
            {"date": FILL_DATE, "is_open": True},
        ]
    )
    return snapshot, states, calendar


def _validate_market_states(snapshot: pd.DataFrame, states: pd.DataFrame, fill_date: str) -> None:
    required = {"date", "asset", "market_state"}
    if not required.issubset(states.columns):
        raise ContractError("synthetic asset market states are incomplete")
    if states.duplicated(["date", "asset"]).any():
        raise ContractError("synthetic asset market states contain duplicates")
    fill_states = states[states["date"].astype(str).eq(fill_date)]
    expected_assets = set(snapshot["asset"].astype(str).str.zfill(6))
    if set(fill_states["asset"].astype(str).str.zfill(6)) != expected_assets:
        raise ContractError("synthetic fill date is missing an asset market state")
    if not fill_states["market_state"].astype(str).eq("OPEN").all():
        raise ContractError("synthetic fill attempts to trade a suspended or delisted asset")


def _candidate_and_targets(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = snapshot[
        ["as_of_date", "available_date", "asset", "name", "asset_type", "sector", "is_tradeable"]
    ].copy()
    candidate["candidate_status"] = "SYNTHETIC_PAPER_ELIGIBLE"
    candidate["final_score"] = [72.0, 70.0]
    candidate["entry_action"] = "BUILD_FIRST_TRANCHE"
    candidate["research_notice"] = SYNTHETIC_NOTICE
    targets = candidate[["asset", "name", "asset_type", "sector", "final_score", "entry_action"]].copy()
    targets.insert(0, "signal_date", SIGNAL_DATE)
    targets["available_date"] = SIGNAL_DATE
    targets["target_semantics"] = "FULL_SNAPSHOT"
    targets["target_schema_version"] = 2
    targets["snapshot_asset_count"] = len(targets)
    targets["full_target_weight"] = [0.12, 0.25]
    targets["target_core_weight"] = [0.06, 0.10]
    targets["target_t_weight_cap"] = 0.0
    targets["target_t_weight"] = 0.0
    targets["t_action"] = "HOLD_T"
    targets["t_reasons"] = "synthetic_core_build_only"
    targets["hard_veto_reasons"] = ""
    targets["manual_risk_review_required"] = False
    targets["portfolio_risk_reasons"] = ""
    return candidate, targets


def _initial_account(config: dict[str, Any]) -> dict[str, Any]:
    return seal_account_state(
        {
            "schema_version": 1,
            "state_version": 0,
            "account_id": "synthetic-paper-account",
            "base_currency": "CNY",
            "as_of_date": SIGNAL_DATE,
            "cash_cny": float(config["account"]["initial_cash_cny"]),
            "holdings": [],
            "realized_pnl_cny": 0.0,
            "gross_dividend_cny": 0.0,
            "dividend_tax_cny": 0.0,
            "processed_fills": [],
            "fill_history": [],
            "processed_events": [],
            "event_history": [],
            "nav_history": [],
        },
        config,
        increment_version=False,
    )


def _manifest(output: Path, *, config_digest: str, binding_digest: str) -> dict[str, Any]:
    files = []
    for relative in CHAIN_FILES:
        payload = (output / relative).read_bytes()
        files.append({"path": relative, "bytes": len(payload), "sha256": _sha256(payload)})
    bundle_payload = "\n".join(f"{item['path']}:{item['sha256']}" for item in files).encode("utf-8")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "notice": SYNTHETIC_NOTICE,
        "config_sha256": config_digest,
        "order_binding_sha256": binding_digest,
        "chain": "snapshot -> candidate -> target -> order -> fill -> account -> NAV",
        "files": files,
        "bundle_sha256": _sha256(bundle_payload),
    }


def verify_replay_bundle(output: Path, expected_manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError("synthetic replay manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("files", []):
        path = output / str(entry["path"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or _sha256(path.read_bytes()) != entry["sha256"]:
            raise ContractError(f"synthetic replay manifest integrity failed: {entry['path']}")
    recalculated = _manifest(
        output,
        config_digest=str(manifest.get("config_sha256", "")),
        binding_digest=str(manifest.get("order_binding_sha256", "")),
    )
    if recalculated != manifest:
        raise ContractError("synthetic replay manifest was modified or is inconsistent")
    if expected_manifest_path is not None:
        if not expected_manifest_path.is_file():
            raise ContractError("expected synthetic replay manifest is missing")
        expected = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
        if expected != manifest:
            raise ContractError(
                f"synthetic replay hash mismatch: expected={expected.get('bundle_sha256')} actual={manifest.get('bundle_sha256')}"
            )
    return manifest


def run_synthetic_replay(
    bundle: Path,
    *,
    output: Path | None = None,
    verify_expected: bool = True,
) -> dict[str, Any]:
    bundle = bundle.resolve()
    output = (output or bundle / "output").resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    snapshot, states, calendar = _synthetic_inputs()
    _validate_market_states(snapshot, states, FILL_DATE)
    candidate, targets = _candidate_and_targets(snapshot)
    input_payloads = {
        "snapshot.csv": _csv_bytes(snapshot),
        "asset_market_states.csv": _csv_bytes(states),
        "trading_calendar.csv": _csv_bytes(calendar),
        "candidate_decisions.csv": _csv_bytes(candidate),
        "target_weights.csv": _csv_bytes(targets),
    }
    for relative, payload in input_payloads.items():
        _write(output / relative, payload)

    calendar_digest = _sha256(input_payloads["trading_calendar.csv"])
    binding_context = {
        "run_id": RUN_ID,
        "snapshot_sha256": _sha256(input_payloads["snapshot.csv"]),
        "target_sha256": _sha256(input_payloads["target_weights.csv"]),
        "target_semantics": "FULL_SNAPSHOT",
    }
    binding_digest = _sha256(json.dumps(binding_context, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    account = _initial_account(config)
    signal_prices = {"000000": 10.0, "999999": 5.0}
    orders = plan_orders(
        targets,
        account,
        signal_prices,
        config,
        SIGNAL_DATE,
        run_id=RUN_ID,
        run_manifest_sha256=binding_digest,
        trade_calendar_sha256=calendar_digest,
        risk_state_at_signal="NORMAL",
    )
    if len(orders) != 2:
        raise ContractError("synthetic replay must produce one stock and one ETF order")
    order_state = register_order_envelopes(
        None,
        orders,
        run_id=RUN_ID,
        account_version=int(account["state_version"]),
        account_state_sha256=str(account["state_sha256"]),
        registered_at=f"{SIGNAL_DATE}T16:00:00",
    )
    fills = orders[["order_id", "asset", "name", "asset_type", "sector", "sleeve", "side", "shares"]].copy()
    fills.insert(0, "fill_id", [f"SYN-FILL-{index + 1:02d}" for index in range(len(fills))])
    fills.insert(1, "fill_date", FILL_DATE)
    fills["price"] = fills["asset"].map(signal_prices)
    fills["fee_mode"] = "model"
    fills["manual_approval"] = False
    fills["manual_reason"] = ""
    fills["risk_override"] = False
    account, ledger, order_state = apply_fills(
        account,
        fills,
        config,
        approved_orders=orders,
        order_state_book=order_state,
        trading_calendar=calendar,
        valuation_prices=signal_prices,
        valuation_as_of_date=FILL_DATE,
        run_manifest_sha256=binding_digest,
        expected_run_id=RUN_ID,
        expected_config_sha256=config_sha256(config),
        trading_calendar_sha256=calendar_digest,
    )
    mark_prices = {"000000": 10.1, "999999": 5.05}
    account, mark = mark_to_market(account, mark_prices, FILL_DATE, config, price_basis="unadjusted_executable")
    order_state = rebind_order_state_account(
        order_state,
        int(account["state_version"]),
        str(account["state_sha256"]),
    )
    validate_paper_account_invariants(account, mark_prices, config, order_state_book=order_state)

    outputs = {
        "orders.csv": _csv_bytes(orders),
        "fills.csv": _csv_bytes(fills),
        "account.json": _json_bytes(account),
        "ledger.csv": _csv_bytes(ledger),
        "order_state.json": _json_bytes(order_state),
        "nav.csv": _csv_bytes(pd.DataFrame([mark])),
    }
    for relative, payload in outputs.items():
        _write(output / relative, payload)
    manifest = _manifest(output, config_digest=config_sha256(config), binding_digest=binding_digest)
    _write(output / "manifest.json", _json_bytes(manifest))
    return verify_replay_bundle(
        output,
        bundle / "expected_manifest.json" if verify_expected else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "examples" / "synthetic_run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-expected", action="store_true", help="maintainer-only: refresh the committed expected manifest")
    args = parser.parse_args()
    manifest = run_synthetic_replay(args.bundle, output=args.output, verify_expected=not args.write_expected)
    if args.write_expected:
        _write(args.bundle / "expected_manifest.json", _json_bytes(manifest))
    print(f"synthetic replay verified: {manifest['bundle_sha256']}")


if __name__ == "__main__":
    main()
