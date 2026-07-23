from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from strategy_lab.long_hold_v4.core import ContractError, load_config
from strategy_lab.long_hold_v4.execution import seal_account_state
from strategy_lab.long_hold_v4.invariants import validate_paper_account_invariants
from strategy_lab.long_hold_v4.synthetic_replay import ROOT, run_synthetic_replay


PROPERTY_SEED = 20260723
PROPERTY_CASES = 250
SECTORS = ["bank", "insurance", "utility", "transport", "telecom", "consumer_staples", "dividend_index"]


def _random_account(rng: random.Random, config: dict, case_index: int) -> tuple[dict, dict[str, float]]:
    holdings = []
    prices: dict[str, float] = {}
    count = rng.randint(0, min(7, int(config["universe"]["maximum_assets"])))
    for index in range(count):
        asset = f"{100000 + case_index * 10 + index:06d}"[-6:]
        price = round(rng.uniform(4.0, 18.0), 2)
        reference = 1000
        core_shares = rng.choice([800, 900, 1000])
        t_shares = rng.choice([0, 100, 200])
        holdings.append(
            {
                "asset": asset,
                "name": f"合成资产{case_index}-{index}",
                "asset_type": "etf" if index == 6 else "stock",
                "sector": SECTORS[index],
                "core_shares": core_shares,
                "core_average_cost_cny": price,
                "core_open_date": "2026-01-02",
                "t_shares": t_shares,
                "t_average_cost_cny": price if t_shares else 0.0,
                "t_open_date": "2026-01-03" if t_shares else None,
                "full_target_shares_reference": reference,
                "realized_pnl_cny": 0.0,
                "cumulative_dividend_net_cny": 0.0,
            }
        )
        prices[asset] = price
    account = seal_account_state(
        {
            "schema_version": 1,
            "state_version": 0,
            "account_id": f"synthetic-property-{case_index}",
            "base_currency": "CNY",
            "as_of_date": "2026-01-05",
            "cash_cny": 500000.0,
            "holdings": holdings,
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
    return account, prices


def test_fixed_seed_randomized_cash_only_invariants_are_replayable() -> None:
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    rng = random.Random(PROPERTY_SEED)
    for case_index in range(PROPERTY_CASES):
        account, prices = _random_account(rng, config, case_index)
        sample = {
            "seed": PROPERTY_SEED,
            "case": case_index,
            "account": account,
            "prices": prices,
        }
        try:
            state = validate_paper_account_invariants(account, prices, config)
            assert state["cash_cny"] >= 0
            assert state["holding_count"] <= config["universe"]["maximum_assets"]
        except Exception as exc:  # pragma: no cover - message is the replay contract
            pytest.fail(
                "randomized invariant failure; replay sample="
                + json.dumps(sample, ensure_ascii=False, sort_keys=True)
                + f"; error={exc}"
            )


@pytest.mark.parametrize("flag", ["allow_short", "allow_margin"])
def test_cash_only_invariant_rejects_short_or_margin(flag: str) -> None:
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    account, prices = _random_account(random.Random(PROPERTY_SEED), config, 0)
    unsafe = copy.deepcopy(config)
    unsafe["account"][flag] = True
    with pytest.raises(ContractError, match="cash-only"):
        validate_paper_account_invariants(account, prices, unsafe)


def test_ledger_cash_and_account_balance_are_one_reconciled_chain(tmp_path: Path) -> None:
    bundle = ROOT / "examples" / "synthetic_run"
    output = tmp_path / "replay"
    run_synthetic_replay(bundle, output=output)
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    account = json.loads((output / "account.json").read_text(encoding="utf-8"))
    order_state = json.loads((output / "order_state.json").read_text(encoding="utf-8"))
    prices = {"000000": 10.1, "999999": 5.05}
    validate_paper_account_invariants(account, prices, config, order_state_book=order_state)

    changed = copy.deepcopy(account)
    changed["fill_history"][-1]["cash_after_cny"] += 1.0
    changed = seal_account_state(changed, config, increment_version=False)
    with pytest.raises(ContractError, match="ledger cash arithmetic|terminal ledger"):
        validate_paper_account_invariants(changed, prices, config)


def test_portfolio_limit_and_t_support_mutations_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "long_hold_v4.json")
    account, prices = _random_account(random.Random(PROPERTY_SEED), config, 1)
    if not account["holdings"]:
        pytest.fail("fixed seed unexpectedly produced an empty replay sample")
    holding = account["holdings"][0]
    holding["core_shares"] = 100000
    holding["full_target_shares_reference"] = 100000.0
    holding["core_average_cost_cny"] = prices[holding["asset"]]
    account = seal_account_state(account, config, increment_version=False)
    with pytest.raises(ContractError, match="cash|core|sector|single-asset"):
        validate_paper_account_invariants(account, prices, config)

    supported, supported_prices = _random_account(random.Random(PROPERTY_SEED + 1), config, 2)
    if not supported["holdings"]:
        pytest.fail("fixed seed unexpectedly produced an empty T-support sample")
    supported["holdings"][0]["core_shares"] = 700
    supported["holdings"][0]["t_shares"] = 100
    supported["holdings"][0]["t_average_cost_cny"] = supported_prices[supported["holdings"][0]["asset"]]
    supported["holdings"][0]["t_open_date"] = "2026-01-03"
    supported = seal_account_state(supported, config, increment_version=False)
    with pytest.raises(ContractError, match="under-supported"):
        validate_paper_account_invariants(supported, supported_prices, config)
