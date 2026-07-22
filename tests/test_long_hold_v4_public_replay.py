from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from strategy_lab.long_hold_v4.backtest import validate_backtest_inputs
from strategy_lab.long_hold_v4.core import ContractError, load_config
from strategy_lab.long_hold_v4.execution import config_sha256
from strategy_lab.long_hold_v4.order_envelope import (
    order_state_record,
    refresh_expired_orders,
    register_order_envelopes,
    verify_order_frame,
)
from strategy_lab.long_hold_v4.pipeline import plan_orders
from strategy_lab.long_hold_v4.recoverable_transaction import recover_pending_write_set
from strategy_lab.long_hold_v4.synthetic_replay import (
    ROOT,
    RUN_ID,
    SIGNAL_DATE,
    _initial_account,
    _validate_market_states,
    run_synthetic_replay,
    verify_replay_bundle,
)


def _run(tmp_path: Path) -> Path:
    output = tmp_path / "replay"
    run_synthetic_replay(ROOT / "examples" / "synthetic_run", output=output)
    return output


def test_replay_is_deterministic_and_covers_complete_chain(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert first_manifest["chain"] == "snapshot -> candidate -> target -> order -> fill -> account -> NAV"
    assert "SYNTHETIC_ONLY" in first_manifest["notice"]
    assert {item["path"] for item in first_manifest["files"]} >= {
        "snapshot.csv", "candidate_decisions.csv", "target_weights.csv", "orders.csv",
        "fills.csv", "account.json", "ledger.csv", "nav.csv",
    }
    for entry in first_manifest["files"]:
        assert (first / entry["path"]).read_bytes() == (second / entry["path"]).read_bytes()


def test_order_tampering_and_duplicate_order_ids_are_rejected(tmp_path: Path) -> None:
    orders = pd.read_csv(_run(tmp_path) / "orders.csv", dtype={"asset": str})
    tampered = orders.copy()
    tampered.loc[0, "name"] = "被篡改的名称"
    with pytest.raises(ContractError, match="hash mismatch"):
        verify_order_frame(tampered)
    with pytest.raises(ContractError, match="duplicate order ids"):
        verify_order_frame(pd.concat([orders, orders.iloc[[0]]], ignore_index=True))


def test_expired_and_superseded_orders_cannot_remain_active(tmp_path: Path) -> None:
    output = _run(tmp_path)
    orders = pd.read_csv(output / "orders.csv", dtype={"asset": str})
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    account = _initial_account(config)
    old_book = register_order_envelopes(
        None, orders, run_id=RUN_ID, account_version=account["state_version"],
        account_state_sha256=account["state_sha256"], registered_at=f"{SIGNAL_DATE}T16:00:00",
    )
    expired = refresh_expired_orders(old_book, "2026-02-01", updated_at="2026-02-01T09:00:00")
    assert all(item["status"] == "EXPIRED" for item in expired["orders"])

    targets = pd.read_csv(output / "target_weights.csv", dtype={"asset": str})
    new_orders = plan_orders(
        targets, account, {"000000": 10.0, "999999": 5.0}, config, SIGNAL_DATE,
        run_id="synthetic-replay-v2", run_manifest_sha256="e" * 64,
        trade_calendar_sha256=str(orders.iloc[0]["trade_calendar_sha256"]), risk_state_at_signal="NORMAL",
    )
    superseded = register_order_envelopes(
        old_book, new_orders, run_id="synthetic-replay-v2", account_version=account["state_version"],
        account_state_sha256=account["state_sha256"], registered_at="2026-01-05T17:00:00",
    )
    for order_id in orders["order_id"]:
        assert order_state_record(superseded, str(order_id))["status"] == "SUPERSEDED"


def test_interrupted_write_set_is_recovered_exactly(tmp_path: Path) -> None:
    destination = (tmp_path / "account.json").resolve()
    transaction_id = "a" * 32
    payload = b'{"synthetic":true}\n'
    staged = destination.parent / f".{destination.name}.{transaction_id}.staged"
    staged.write_bytes(payload)
    journal = tmp_path / ".transaction.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": transaction_id,
                "files": [{"destination": str(destination), "staged": str(staged), "sha256": hashlib.sha256(payload).hexdigest()}],
            }
        ),
        encoding="utf-8",
    )
    assert recover_pending_write_set(journal, [destination])
    assert destination.read_bytes() == payload
    assert not journal.exists()


@pytest.mark.parametrize("state", ["SUSPENDED", "DELISTED"])
def test_fill_date_suspension_or_delisting_is_rejected(tmp_path: Path, state: str) -> None:
    output = _run(tmp_path)
    snapshot = pd.read_csv(output / "snapshot.csv", dtype={"asset": str})
    states = pd.read_csv(output / "asset_market_states.csv", dtype={"asset": str})
    states.loc[(states["date"] == "2026-01-06") & (states["asset"] == "000000"), "market_state"] = state
    with pytest.raises(ContractError, match="suspended or delisted"):
        _validate_market_states(snapshot, states, "2026-01-06")


def test_missing_account_and_modified_manifest_fail_verification(tmp_path: Path) -> None:
    output = _run(tmp_path)
    (output / "account.json").unlink()
    with pytest.raises(ContractError, match="account.json"):
        verify_replay_bundle(output)
    output = _run(tmp_path / "again")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["notice"] = "changed"
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="modified or is inconsistent"):
        verify_replay_bundle(output)


def test_full_snapshot_and_delta_misuse_are_rejected() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2026-01-05", "asset": "000000", "asset_type": "stock", "open": 10, "close": 10, "return_basis": "qfq_adjusted", "price_basis": "qfq_adjusted"},
            {"date": "2026-01-05", "asset": "999999", "asset_type": "etf", "open": 5, "close": 5, "return_basis": "qfq_adjusted", "price_basis": "qfq_adjusted"},
        ]
    )
    delta = pd.DataFrame(
        [{"signal_date": "2026-01-05", "asset": "000000", "target_core_weight": 0.1, "target_t_weight": 0.0, "target_semantics": "DELTA", "target_schema_version": 2}]
    )
    with pytest.raises(ContractError, match="requires FULL_SNAPSHOT"):
        validate_backtest_inputs(prices, delta, formal_backtest=True)

    omitted = pd.DataFrame(
        [
            {"signal_date": "2026-01-04", "asset": "000000", "target_core_weight": 0.1, "target_t_weight": 0.0, "target_semantics": "FULL_SNAPSHOT", "target_schema_version": 2, "snapshot_asset_count": 1},
            {"signal_date": "2026-01-05", "asset": "999999", "target_core_weight": 0.1, "target_t_weight": 0.0, "target_semantics": "FULL_SNAPSHOT", "target_schema_version": 2, "snapshot_asset_count": 1},
        ]
    )
    with pytest.raises(ContractError, match="omits previously held assets"):
        validate_backtest_inputs(prices, omitted)
