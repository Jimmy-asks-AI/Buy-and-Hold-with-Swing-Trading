#!/usr/bin/env python
"""Smoke tests for the governed quant model system."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import a_share_low_cost_factor_builder as lowcost
import a_share_panel_builder as panel_builder
import csv_io
import data_quality_report as dq
import factor_factory_walk_forward as wf
import paper_trading_monitor as paper
import quant_model_system as qms
import real_data_adapter


ROOT = Path("Introduction-to-Quantitative-Finance")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_execution_constraints_respect_tradeability_and_capacity() -> None:
    weights = pd.DataFrame(
        {
            "date": ["2020-01-31"] * 4,
            "asset": ["A", "B", "C", "D"],
            "weight": [0.4, 0.3, 0.2, 0.1],
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4),
            "asset": ["A", "B", "C", "D"],
            "is_tradeable": ["TRUE", "False", "1", "yes"],
            "amount": [1_000_000.0, 1_000_000.0, 100_000.0, 50_000.0],
        }
    )
    config = wf.WalkForwardConfig(fund_size=10_000_000.0, max_participation_rate=0.05)
    constrained = wf.apply_execution_constraints(weights, panel, config)
    caps = panel.set_index("asset")["amount"] * config.max_participation_rate / config.fund_size
    by_asset = constrained.set_index("asset")["weight"]
    _assert(abs(float(by_asset.loc["B"])) < 1e-12, "untradeable asset kept non-zero weight")
    for asset in ["A", "C", "D"]:
        _assert(float(by_asset.loc[asset]) <= float(caps.loc[asset]) + 1e-12, f"{asset} exceeds capacity cap")
    _assert(float(by_asset.sum()) <= float(caps[["A", "C", "D"]].sum()) + 1e-12, "weights exceed total capacity")


def test_real_data_adapter_maps_aliases_and_bool_strings() -> None:
    raw = pd.DataFrame(
        {
            "trade_date": ["2020-01-02"],
            "ts_code": ["000001.SZ"],
            "close_adj": [10.5],
            "turnover_value": [1000000.0],
            "total_mv": [100000000.0],
            "tradeable": ["False"],
        }
    )
    mapping = real_data_adapter.load_mapping(ROOT / "data_catalog" / "a_share_real_data_field_mapping_template.csv")
    mapped, report = real_data_adapter.canonicalize_table(raw, mapping, "market")
    _assert({"date", "asset", "adj_close", "amount", "market_cap", "is_tradeable"}.issubset(mapped.columns), "canonical columns missing")
    _assert(str(mapped["asset"].iloc[0]) == "000001.SZ", "asset alias mapping failed")
    _assert(bool(mapped["is_tradeable"].iloc[0]) is False, "bool string mapping failed")
    _assert(not (report["status"] == "missing_required").any(), "required alias mapping failed")


def test_point_in_time_panel_builder() -> None:
    raw = lowcost.make_synthetic_low_cost_panel(n_dates=160, n_assets=40)
    market, financial, industry = panel_builder.split_synthetic_tables(raw)
    panel = panel_builder.build_point_in_time_panel(market, financial=financial, industry=industry)
    _assert(panel.shape[0] == market.shape[0], "panel row count changed unexpectedly")
    _assert(panel["asset"].nunique() == market["asset"].nunique(), "asset count changed unexpectedly")
    _assert(panel.duplicated(["date", "asset"]).sum() == 0, "duplicate date-asset rows")
    available = pd.to_datetime(panel["financial_available_at"], errors="coerce")
    dates = pd.to_datetime(panel["date"], errors="coerce")
    _assert(bool((available.dropna() <= dates.loc[available.dropna().index]).all()), "future financial data leaked")
    _assert(float(panel["fwd_return"].notna().mean()) > 0.5, "forward-return coverage too low")


def test_data_quality_report_gates() -> None:
    panel = lowcost.add_forward_return(lowcost.build_low_cost_factors(lowcost.make_synthetic_low_cost_panel(n_dates=180, n_assets=50)))
    registry = pd.read_csv(ROOT / "data_catalog" / "a_share_factor_registry_v0.csv", encoding="utf-8-sig")
    config = wf.load_config(ROOT / "configs" / "factor_factory_smoke.json")
    report = dq.build_quality_report(panel, registry, config)
    _assert({"summary", "column_coverage", "factor_coverage", "date_health", "gates"}.issubset(report.keys()), "missing data-quality tables")
    summary = report["summary"].set_index("metric")
    _assert(summary.loc["duplicate_date_asset", "status"] == "pass", "duplicate gate failed")
    _assert(float(summary.loc["available_registered_factors", "value"]) > 0, "no available registered factors")
    _assert(not (report["gates"]["status"] == "fail").any(), "data-quality report has failed gates")


def test_data_quality_report_blocks_future_availability() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "asset": ["A", "A"],
            "fwd_return": [0.01, 0.02],
            "industry": ["bank", "bank"],
            "log_mkt_cap": [10.0, 10.0],
            "amount": [1000.0, 1000.0],
            "market_cap": [10000.0, 10000.0],
            "is_tradeable": ["true", "true"],
            "raw_factor": [1.0, 2.0],
            "raw_factor_available_at": ["2020-01-01", "2020-01-10"],
        }
    )
    registry = pd.DataFrame(
        [
            {
                "factor_id": "future_test",
                "column": "raw_factor",
                "family": "test",
                "direction": 1,
                "horizon": 20,
                "data_type": "test",
                "availability_col": "raw_factor_available_at",
            }
        ]
    )
    config = wf.WalkForwardConfig(min_assets=1, train_periods=1, test_periods=1)
    report = dq.build_quality_report(panel, registry, config)
    gates = report["gates"].set_index("gate")
    _assert(gates.loc["future_availability_leakage", "status"] == "fail", "future availability leakage not blocked")


def test_paper_drift_report() -> None:
    target = pd.DataFrame({"asset": ["A", "B", "C"], "weight": [0.50, 0.30, 0.20]})
    current = pd.DataFrame({"asset": ["A", "B", "D"], "current_weight": [0.45, 0.35, 0.20]})
    drift, summary = paper.paper_drift_report(target, current)
    _assert({"target_weight", "current_weight", "trade_weight", "abs_drift"}.issubset(drift.columns), "drift columns missing")
    row_c = drift.loc[drift["asset"] == "C"].iloc[0]
    row_d = drift.loc[drift["asset"] == "D"].iloc[0]
    _assert(float(row_c["trade_weight"]) > 0, "new target position should require buy")
    _assert(float(row_d["trade_weight"]) < 0, "exit position should require sell")
    _assert(int(summary["new_positions"].iloc[0]) == 1, "new position count mismatch")
    _assert(int(summary["exit_positions"].iloc[0]) == 1, "exit position count mismatch")


def test_preflight_outputs_without_backtest() -> None:
    out_dir = ROOT / "outputs" / "quant_model_system_preflight_smoke"
    panel = lowcost.add_forward_return(lowcost.build_low_cost_factors(lowcost.make_synthetic_low_cost_panel(n_dates=180, n_assets=50)))
    panel_path = out_dir / "preflight_panel.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    result_dir = qms.run_preflight_from_files(
        str(panel_path),
        str(ROOT / "data_catalog" / "a_share_factor_registry_v0.csv"),
        str(ROOT / "configs" / "factor_factory_smoke.json"),
        str(out_dir),
    )
    required = [
        result_dir / "panel_validation.csv",
        result_dir / "data_quality" / "gates.csv",
        result_dir / "registry_summary.csv",
        result_dir / "SYSTEM_RUN_SUMMARY.md",
    ]
    for path in required:
        _assert(path.exists(), f"preflight missing output: {path}")


def test_one_command_system_demo_outputs() -> None:
    output = qms.run_demo(ROOT, "quant_model_system_smoke_test")
    required = [
        output / "panel_validation.csv",
        output / "ledger_row.csv",
        output / "MODEL_RUN_REPORT.md",
        output / "SYSTEM_RUN_SUMMARY.md",
        output / "walk_forward" / "walk_forward_performance.csv",
        output / "walk_forward" / "weights.csv",
        output / "paper_tracking" / "paper_state.json",
    ]
    for path in required:
        _assert(path.exists(), f"missing output: {path}")
    validation = pd.read_csv(output / "panel_validation.csv", encoding="utf-8-sig")
    _assert(not (validation["status"] == "fail").any(), "panel validation has failed gates")
    performance = pd.read_csv(output / "walk_forward" / "walk_forward_performance.csv", encoding="utf-8-sig")
    _assert({"metric", "gross", "net"}.issubset(performance.columns), "performance table missing gross/net columns")
    state = json.loads((output / "paper_tracking" / "paper_state.json").read_text(encoding="utf-8"))
    _assert(state["status"] == "paper_tracking", "paper state status mismatch")
    _assert(state["live_trading_allowed"] is False, "paper state incorrectly allows live trading")


def main() -> None:
    tests = [
        test_execution_constraints_respect_tradeability_and_capacity,
        test_real_data_adapter_maps_aliases_and_bool_strings,
        test_point_in_time_panel_builder,
        test_data_quality_report_gates,
        test_data_quality_report_blocks_future_availability,
        test_paper_drift_report,
        test_preflight_outputs_without_backtest,
        test_one_command_system_demo_outputs,
    ]
    rows = []
    for test in tests:
        test()
        rows.append({"test": test.__name__, "status": "pass"})
    out = ROOT / "outputs" / "quant_model_system_smoke_test" / "smoke_test_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
