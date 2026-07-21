from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4.pit_history_gate import (
    ROOT,
    ContractError,
    _sha256,
    _value_set_sha256,
    load_gate_config,
    run_gate,
    validate_dataset,
)
from strategy_lab.long_hold_v4.pit_macro_adapter import build_macro_rate_history
from strategy_lab.long_hold_v4 import pit_macro_adapter
from strategy_lab.long_hold_v4.pit_etf_benchmark_probe import build_etf_benchmark_observation
from strategy_lab.long_hold_v4.pit_etf_master_builder import build_etf_security_master
from strategy_lab.long_hold_v4.pit_stock_adjustment_builder import (
    dividend_alignment_passes,
    normalise_factor_events,
    parse_sina_hfq_response,
)
from strategy_lab.long_hold_v4.pit_stock_adjustment_validator import validate_factor_transitions
from strategy_lab.long_hold_v4.pit_stock_market_history_builder import (
    REQUIRED_HISTORY_COLUMNS,
    _safe_baostock_send_msg,
    build_monthly_valuation,
    build_trade_state,
    effective_market_date,
    filter_lifecycles_by_asset_file,
    normalise_baostock_history,
    relevant_lifecycles,
    select_collection_shard,
)
from strategy_lab.long_hold_v4 import pit_stock_market_history_builder
from strategy_lab.long_hold_v4 import pit_tushare_daily_refresher
from strategy_lab.long_hold_v4.pit_stock_market_history_orchestrator import (
    _read_fresh_shard_report,
    run_orchestration,
    worker_command,
)
from strategy_lab.long_hold_v4.pit_stock_market_history_validator import (
    TRADE_QUALIFICATION_CHECKS,
    VALUATION_QUALIFICATION_CHECKS,
    _available_builder_assets,
    _read_asset_subset,
    _scope_checks,
    compare_eastmoney_valuation,
    compare_joinquant_state,
    compare_joinquant_valuation,
    compare_tushare_prices,
    build_adjustment_ratios,
    partition_eastmoney_validation_assets,
    select_validation_assets,
    split_eastmoney_eligibility,
)
from strategy_lab.long_hold_v4.pit_stock_master_builder import build_stock_security_master
from strategy_lab.long_hold_v4.pit_stock_name_history_collector import (
    classify_security_name,
    normalise_name_events,
    parse_sina_name_history,
    parse_sina_undated_name_summary,
)
from strategy_lab.long_hold_v4.pit_stock_status_event_builder import (
    build_status_events,
    official_status_events_as_name_events,
)
from strategy_lab.long_hold_v4.pit_stock_status_event_validator import (
    _daily_metrics,
    apply_status_events,
    build_baostock_era_metrics,
    build_event_index,
    compare_factbook_holdout_events,
    compare_sse_holdout_events,
    compare_transition_dates,
    expected_binary_transitions,
    select_factbook_holdout_assets,
    select_sse_holdout_assets,
)
from strategy_lab.long_hold_v4.pit_stock_status_event_reconciler import (
    reconcile_asset_events,
    select_factbook_reference_candidates,
)
from strategy_lab.long_hold_v4.pit_sse_status_announcement_collector import (
    classify_announcement_title,
    market_session_on_or_after,
    next_market_session,
    parse_asset_events,
)
from strategy_lab.long_hold_v4.pit_sse_execution_status_history_collector import (
    select_shard as select_sse_status_shard,
    target_sse_assets,
)
from strategy_lab.long_hold_v4.pit_stock_industry_history_builder import build_stock_industry_history
from strategy_lab.long_hold_v4.pit_stock_dividend_builder import build_stock_dividend_events, report_periods
from strategy_lab.long_hold_v4.pit_stock_fundamentals_builder import build_stock_fundamentals


def basic_rule(dataset_id: str, sleeve: str, path: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "sleeve": sleeve,
        "priority": "P0",
        "path": path,
        "required_columns": ["date", "asset", "value", "available_date", "data_source", "source_vintage"],
        "primary_key": ["date", "asset"],
        "date_columns": ["date"],
        "available_date_column": "available_date",
        "availability_floor_columns": ["date"],
        "coverage_column": "date",
        "minimum_start_date": "2020-01-02",
        "minimum_end_date": "2020-01-02",
        "minimum_rows": 1,
        "minimum_assets": 1,
        "finite_numeric_columns": ["value"],
        "provider_options": ["test fixture"],
    }


def valid_row(**overrides) -> dict:
    row = {
        "date": "2020-01-02",
        "asset": "000001",
        "value": 1.0,
        "available_date": "2020-01-02",
        "data_source": "test.source",
        "source_vintage": "test.v1",
    }
    row.update(overrides)
    return row


class PitHistoryGateTests(unittest.TestCase):
    def _workspace(self):
        return tempfile.TemporaryDirectory(dir=ROOT)

    @staticmethod
    def _relative(path: Path) -> str:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()

    @staticmethod
    def _write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def test_etf_continuous_history_gate_starts_at_first_etf_listing(self):
        config = load_gate_config(ROOT / "configs" / "long_hold_v4_pit_gate.json")
        rules = {rule["dataset_id"]: rule for rule in config["datasets"]}
        for dataset_id in (
            "etf_benchmark_history",
            "etf_total_return_prices",
            "etf_aum_liquidity_history",
            "etf_fee_tracking_history",
        ):
            self.assertEqual(rules[dataset_id]["minimum_start_date"], "2005-02-23")
        self.assertEqual(rules["index_total_return_valuation"]["minimum_start_date"], "2005-01-01")
        dividend_rule = rules["etf_dividend_events"]
        self.assertEqual(dividend_rule["minimum_start_date"], "2006-05-13")
        self.assertEqual(dividend_rule["minimum_rows"], 800)
        self.assertIn("complete 1701-ETF", dividend_rule["coverage_threshold_basis"])
        self.assertEqual(
            dividend_rule["validation_schema"],
            "etf_dividend_full_universe_official_v1",
        )

    def test_six_etf_history_rules_require_lineage_validation_and_availability_lag(self):
        config = load_gate_config(ROOT / "configs" / "long_hold_v4_pit_gate.json")
        rules = {rule["dataset_id"]: rule for rule in config["datasets"]}
        dataset_ids = (
            "etf_benchmark_history",
            "etf_total_return_prices",
            "etf_aum_liquidity_history",
            "etf_fee_tracking_history",
            "etf_dividend_events",
            "index_total_return_valuation",
        )
        for dataset_id in dataset_ids:
            with self.subTest(dataset_id=dataset_id):
                rule = rules[dataset_id]
                self.assertTrue(rule["require_external_evidence"])
                self.assertTrue(rule["lineage_manifest"])
                self.assertTrue(rule["validation_manifest"])
                self.assertIn("current_final_snapshot", rule["lineage_required_values"])
                self.assertFalse(rule["lineage_required_values"]["current_final_snapshot"])
                self.assertGreaterEqual(rule["maximum_availability_lag_days"], 0)

    def test_external_evidence_contract_rejects_missing_manifest_paths(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            rule = basic_rule("etf_history", "etf", self._relative(base / "history.csv"))
            rule["require_external_evidence"] = True
            config_path = base / "config.json"
            config_path.write_text(
                json.dumps({"model": "test", "output_directory": self._relative(base / "out"), "datasets": [rule]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "requires external evidence"):
                load_gate_config(config_path)

    def test_missing_required_dataset_blocks_without_promoting(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            missing = base / "missing.csv"
            out = base / "out"
            rule = basic_rule("stock_missing", "stock", self._relative(missing))
            config = {"model": "test", "output_directory": self._relative(out), "datasets": [rule]}
            paths = run_gate(ROOT, config, "2026-07-17")
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            queue = pd.read_csv(paths["missing_data_queue"], encoding="utf-8-sig")
            self.assertEqual(readiness["system_status"], "BLOCKED_MISSING_OR_INVALID_PIT_DATA")
            self.assertFalse(readiness["historical_inputs_ready"])
            self.assertFalse(readiness["promotion_allowed"])
            self.assertEqual(queue.loc[0, "status"], "missing")

    def test_streaming_gate_accepts_strictly_sorted_primary_keys_across_chunks(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            source = base / "streaming.csv.gz"
            rows = [
                valid_row(date="2020-01-02", asset="000001"),
                valid_row(date="2020-01-02", asset="000002"),
                valid_row(date="2020-01-03", asset="000001", available_date="2020-01-03"),
            ]
            pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")
            rule = basic_rule("stock_streaming", "stock", self._relative(source))
            rule.update(
                {
                    "streaming_csv": True,
                    "streaming_chunksize": 1,
                    "minimum_end_date": "2020-01-03",
                }
            )
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "pass")
            status = checks.set_index("check")["status"].to_dict()
            self.assertEqual(status["streaming_primary_key_order"], "pass")
            self.assertEqual(status["primary_key_unique"], "pass")

    def test_streaming_gate_blocks_unsorted_cross_chunk_keys(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            source = base / "streaming.csv.gz"
            rows = [
                valid_row(date="2020-01-03", asset="000001", available_date="2020-01-03"),
                valid_row(date="2020-01-02", asset="000001"),
            ]
            pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")
            rule = basic_rule("stock_streaming", "stock", self._relative(source))
            rule.update({"streaming_csv": True, "streaming_chunksize": 1})
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
            self.assertIn("streaming_primary_key_order", failed)
            self.assertIn("primary_key_unique", failed)

    def test_valid_stock_etf_macro_inputs_only_qualify_for_walk_forward(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            rules = []
            for dataset_id, sleeve in (("stock_ok", "stock"), ("etf_ok", "etf"), ("macro_ok", "macro")):
                path = base / f"{dataset_id}.csv"
                self._write(path, [valid_row(asset=dataset_id)])
                rules.append(basic_rule(dataset_id, sleeve, self._relative(path)))
            config_path = base / "config.json"
            config = {"model": "test", "output_directory": self._relative(base / "out"), "datasets": rules}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            paths = run_gate(ROOT, config, "2026-07-17", config_path)
            readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
            self.assertTrue(readiness["stock_history_ready"])
            self.assertTrue(readiness["etf_history_ready"])
            self.assertTrue(readiness["macro_history_ready"])
            self.assertTrue(readiness["historical_inputs_ready"])
            self.assertEqual(readiness["system_status"], "PIT_INPUTS_READY_FOR_WALK_FORWARD")
            self.assertFalse(readiness["walk_forward_completed"])
            self.assertFalse(readiness["promotion_allowed"])
            empty_queue = pd.read_csv(paths["missing_data_queue"], encoding="utf-8-sig")
            self.assertTrue(empty_queue.empty)
            self.assertIn("dataset_id", empty_queue.columns)

    def test_unadjusted_return_basis_is_rejected(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "prices.csv"
            self._write(path, [valid_row(return_basis="none_raw")])
            rule = basic_rule("etf_prices", "etf", self._relative(path))
            rule["required_columns"].append("return_basis")
            rule["allowed_values"] = {"return_basis": ["total_return", "qfq_adjusted", "hfq_adjusted"]}
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("allowed_values:return_basis"), "status"].iloc[0], "fail")

    def test_future_available_date_is_rejected(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "future.csv"
            self._write(path, [valid_row(date="2026-07-16", available_date="2026-07-18")])
            rule = basic_rule("future", "stock", self._relative(path))
            rule["minimum_start_date"] = "2026-07-16"
            rule["minimum_end_date"] = "2026-07-16"
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("available_date_not_future"), "status"].iloc[0], "fail")

    def test_current_final_snapshot_availability_lag_is_rejected_in_memory_and_streaming_modes(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "current_final.csv"
            self._write(path, [valid_row(date="2020-01-02", available_date="2026-07-17")])
            for streaming in (False, True):
                with self.subTest(streaming=streaming):
                    rule = basic_rule("etf_prices", "etf", self._relative(path))
                    rule["maximum_availability_lag_days"] = 7
                    rule["availability_lag_reference_column"] = "date"
                    if streaming:
                        rule.update({"streaming_csv": True, "streaming_chunksize": 1})
                    checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
                    self.assertEqual(summary["status"], "blocked")
                    lag_check = checks.loc[checks["check"].eq("availability_lag_within_limit"), "status"]
                    self.assertEqual(lag_check.iloc[0], "fail")

    def test_current_only_master_without_delisted_assets_is_rejected(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "master.csv"
            row = valid_row(list_status="listed")
            self._write(path, [row])
            rule = basic_rule("master", "stock", self._relative(path))
            rule["required_columns"].append("list_status")
            rule["require_any_values"] = [{"column": "list_status", "values": ["delisted"]}]
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("required_population:list_status"), "status"].iloc[0], "fail")

    def test_overlapping_effective_intervals_are_rejected(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "industry.csv"
            rows = [
                valid_row(date="2020-01-01", effective_from="2020-01-01", effective_to="2021-01-01"),
                valid_row(date="2020-06-01", effective_from="2020-06-01", effective_to="2022-01-01"),
            ]
            self._write(path, rows)
            rule = basic_rule("industry", "stock", self._relative(path))
            rule["primary_key"] = ["asset", "effective_from"]
            rule["required_columns"].extend(["effective_from", "effective_to"])
            rule["date_columns"].extend(["effective_from", "effective_to"])
            rule["interval"] = {
                "group_columns": ["asset"],
                "start_column": "effective_from",
                "end_column": "effective_to",
            }
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("effective_intervals_non_overlapping"), "status"].iloc[0], "fail")

    def test_conditional_numeric_gate_rejects_fabricated_or_reversed_price_limits(self):
        with self._workspace() as tmp:
            path = Path(tmp) / "trade_state.csv"
            rows = [
                {
                    **valid_row(),
                    "has_price_limit": True,
                    "limit_up": 11.0,
                    "limit_down": 9.0,
                    "execution_state_known": True,
                },
                {
                    **valid_row(date="2020-01-03", available_date="2020-01-03"),
                    "has_price_limit": False,
                    "limit_up": None,
                    "limit_down": None,
                    "execution_state_known": False,
                },
            ]
            self._write(path, rows)
            rule = basic_rule("trade_state", "stock", self._relative(path))
            rule["required_columns"].extend(
                ["has_price_limit", "limit_up", "limit_down", "execution_state_known"]
            )
            rule["minimum_start_date"] = "2020-01-02"
            rule["minimum_end_date"] = "2020-01-03"
            rule["conditional_finite_numeric_columns"] = [
                {
                    "when_column": "has_price_limit",
                    "when_values": ["true"],
                    "columns": ["limit_up", "limit_down"],
                }
            ]
            rule["conditional_null_columns"] = [
                {
                    "when_column": "has_price_limit",
                    "when_values": ["false"],
                    "columns": ["limit_up", "limit_down"],
                }
            ]
            rule["conditional_numeric_relations"] = [
                {
                    "when_column": "has_price_limit",
                    "when_values": ["true"],
                    "left_column": "limit_up",
                    "operator": ">",
                    "right_column": "limit_down",
                }
            ]
            rule["value_ratio_limits"] = [
                {"column": "execution_state_known", "values": ["false"], "maximum": 0.50}
            ]
            _, good, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(good["status"], "pass")

            rows[0]["limit_up"] = 8.0
            rows[1]["limit_down"] = 0.0
            self._write(path, rows)
            checks, blocked, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(
                checks.loc[checks["check"].eq("numeric_relation:limit_up>limit_down"), "status"].iloc[0],
                "fail",
            )
            self.assertEqual(checks.loc[checks["check"].eq("conditional_null:limit_down"), "status"].iloc[0], "fail")

    def test_macro_adapter_maps_three_rate_series_and_keeps_vintage(self):
        source = pd.DataFrame(
            [
                {"date": "2020-01-02", "available_date": "2020-01-03", "series_id": old, "value": value, "source": "test"}
                for old, value in (
                    ("cn_10y_gov_bond_yield", 3.1),
                    ("us_10y_treasury_yield", 1.9),
                    ("cn_us_10y_rate_spread", 1.2),
                )
            ]
        )
        result = build_macro_rate_history(source, "fixture:v1", "2020-01-03")
        self.assertEqual(set(result["series_id"]), {"CN10Y", "US10Y", "CN_US_10Y_SPREAD"})
        self.assertEqual(set(result["source_vintage"]), {"fixture:v1"})
        self.assertTrue((result["available_date"] >= result["observation_date"]).all())

    def test_lineage_manifest_detects_tampered_dataset(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            path = base / "macro.csv"
            input_path = base / "raw.csv"
            manifest_path = base / "lineage.json"
            input_path.write_text("raw-source\n", encoding="utf-8")
            row = valid_row(source_vintage="fixture:v1")
            self._write(path, [row])
            rule = basic_rule("macro", "macro", self._relative(path))
            rule["lineage_manifest"] = self._relative(manifest_path)
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [{"path": self._relative(input_path), "sha256": _sha256(input_path)}],
                "source_vintage": "fixture:v1",
                "output_path": rule["path"],
                "output_sha256": _sha256(path),
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "historical_backtest_allowed": True,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, good_summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(good_summary["status"], "pass")

            input_path.write_text("tampered\n", encoding="utf-8")
            checks, input_summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(input_summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("lineage_inputs_hash_match"), "status"].iloc[0], "fail")
            input_path.write_text("raw-source\n", encoding="utf-8")

            self._write(path, [{**row, "value": 2.0}])
            checks, bad_summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(bad_summary["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("lineage_output_hash_match"), "status"].iloc[0], "fail")

    def test_lineage_required_values_reject_current_final_snapshot(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            path = base / "etf_prices.csv"
            input_path = base / "raw.csv"
            manifest_path = base / "lineage.json"
            input_path.write_text("raw-source\n", encoding="utf-8")
            self._write(path, [valid_row(source_vintage="fixture:v1")])
            rule = basic_rule("etf_prices", "etf", self._relative(path))
            rule["lineage_manifest"] = self._relative(manifest_path)
            rule["lineage_required_values"] = {"current_final_snapshot": False}
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [{"path": self._relative(input_path), "sha256": _sha256(input_path)}],
                "source_vintage": "fixture:v1",
                "output_path": rule["path"],
                "output_sha256": _sha256(path),
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "historical_backtest_allowed": True,
                "current_final_snapshot": True,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            current_check = checks.loc[
                checks["check"].eq("lineage_required_value:current_final_snapshot"), "status"
            ]
            self.assertEqual(current_check.iloc[0], "fail")

    def test_lineage_role_output_and_source_vintage_set_are_authenticated(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            path = base / "events.csv"
            input_path = base / "candidates.csv"
            manifest_path = base / "lineage.json"
            input_path.write_text("validated candidates\n", encoding="utf-8")
            vintages = {"official_pdf_sha256:" + "a" * 64, "official_pdf_sha256:" + "b" * 64}
            self._write(
                path,
                [
                    valid_row(asset="000001", source_vintage=sorted(vintages)[0]),
                    valid_row(asset="000002", source_vintage=sorted(vintages)[1]),
                ],
            )
            rule = basic_rule("events", "etf", self._relative(path))
            rule.update(
                {
                    "minimum_rows": 2,
                    "minimum_assets": 2,
                    "lineage_manifest": self._relative(manifest_path),
                    "lineage_required_output_role": "pit_events",
                    "lineage_source_vintage_mode": "set_sha256",
                }
            )
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [{"path": self._relative(input_path), "sha256": _sha256(input_path)}],
                "outputs": [
                    {
                        "role": "pit_events",
                        "path": rule["path"],
                        "sha256": _sha256(path),
                    }
                ],
                "source_vintage_count": 2,
                "source_vintage_set_sha256": _value_set_sha256(vintages),
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "historical_backtest_allowed": True,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                checks.loc[checks["check"].eq("lineage_source_vintage_match"), "status"].iloc[0],
                "pass",
            )

            manifest["source_vintage_count"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(
                checks.loc[checks["check"].eq("lineage_source_vintage_match"), "status"].iloc[0],
                "fail",
            )

    def test_secondary_validation_manifest_requires_hashed_inputs_and_zero_hard_exceptions(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            dataset = base / "factor.csv"
            raw = base / "raw.csv"
            checks_path = base / "checks.csv"
            exceptions_path = base / "exceptions.csv"
            review_path = base / "review.csv"
            report_path = base / "report.json"
            manifest_path = base / "validation.json"
            self._write(dataset, [valid_row()])
            raw.write_text("raw\n", encoding="utf-8")
            pd.DataFrame([{"ok": True}]).to_csv(checks_path, index=False)
            pd.DataFrame(columns=["asset"]).to_csv(exceptions_path, index=False)
            pd.DataFrame([{"asset": "000001"}]).to_csv(review_path, index=False)
            report_path.write_text("{}", encoding="utf-8")
            rule = basic_rule("factor", "stock", self._relative(dataset))
            rule["validation_manifest"] = self._relative(manifest_path)
            rule["validation_minimum_input_count"] = 2
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [
                    {"path": self._relative(dataset), "sha256": _sha256(dataset)},
                    {"path": self._relative(raw), "sha256": _sha256(raw)},
                ],
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "output_path": self._relative(checks_path),
                "output_sha256": _sha256(checks_path),
                "exceptions_path": self._relative(exceptions_path),
                "exceptions_sha256": _sha256(exceptions_path),
                "long_gap_review_path": self._relative(review_path),
                "long_gap_review_sha256": _sha256(review_path),
                "report_path": self._relative(report_path),
                "report_sha256": _sha256(report_path),
                "qualification_status": "PASS",
                "review_status": "LONG_SUSPENSION_REVIEW_REQUIRED",
                "continuous_trading_large_jump_count": 0,
                "long_gap_large_jump_count": 1,
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, good, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(good["status"], "pass")

            pd.DataFrame([{"asset": "000001"}]).to_csv(exceptions_path, index=False)
            manifest["exceptions_sha256"] = _sha256(exceptions_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, blocked, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(checks.loc[checks["check"].eq("validation_exception_file_empty"), "status"].iloc[0], "fail")

    def test_generic_validation_manifest_uses_declared_schema_and_output_roles(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            dataset = base / "market.csv"
            raw = base / "raw.csv"
            checks_path = base / "checks.csv"
            exceptions_path = base / "exceptions.csv"
            report_path = base / "report.json"
            manifest_path = base / "validation.json"
            self._write(dataset, [valid_row()])
            raw.write_text("raw\n", encoding="utf-8")
            pd.DataFrame([{"ok": True}]).to_csv(checks_path, index=False)
            pd.DataFrame(columns=["check", "status", "detail"]).to_csv(exceptions_path, index=False)
            report_path.write_text("{}", encoding="utf-8")
            rule = basic_rule("market", "stock", self._relative(dataset))
            rule.update(
                {
                    "validation_manifest": self._relative(manifest_path),
                    "validation_schema": "cross_provider_v1",
                    "validation_minimum_input_count": 2,
                    "validation_required_zero_counts": [],
                    "validation_allowed_values": {},
                    "validation_required_output_roles": ["checks", "exceptions", "report"],
                }
            )
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [
                    {"path": self._relative(dataset), "sha256": _sha256(dataset)},
                    {"path": self._relative(raw), "sha256": _sha256(raw)},
                ],
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "validation_schema": "cross_provider_v1",
                "outputs": [
                    {"role": "checks", "path": self._relative(checks_path), "sha256": _sha256(checks_path)},
                    {
                        "role": "exceptions",
                        "path": self._relative(exceptions_path),
                        "sha256": _sha256(exceptions_path),
                    },
                    {"role": "report", "path": self._relative(report_path), "sha256": _sha256(report_path)},
                ],
                "exceptions_path": self._relative(exceptions_path),
                "exceptions_sha256": _sha256(exceptions_path),
                "qualification_status": "PASS",
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "pass")

            manifest["outputs"] = manifest["outputs"][:-1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(
                checks.loc[checks["check"].eq("validation_output_hash_match:report"), "status"].iloc[0], "fail"
            )

    def test_post_validation_promotion_contract_does_not_require_downstream_input_cycle(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            dataset = base / "promoted.csv"
            candidates = base / "candidates.csv"
            checks_path = base / "checks.csv"
            exceptions_path = base / "exceptions.csv"
            report_path = base / "report.json"
            manifest_path = base / "validation.json"
            self._write(dataset, [valid_row()])
            candidates.write_text("candidate\n", encoding="utf-8")
            pd.DataFrame([{"ok": True}]).to_csv(checks_path, index=False)
            pd.DataFrame(columns=["check", "status", "detail"]).to_csv(exceptions_path, index=False)
            report_path.write_text("{}", encoding="utf-8")
            rule = basic_rule("promoted", "etf", self._relative(dataset))
            rule.update(
                {
                    "validation_manifest": self._relative(manifest_path),
                    "validation_schema": "candidate_first_v1",
                    "validation_required_zero_counts": ["failed_check_rows"],
                    "validation_allowed_values": {},
                    "validation_require_dataset_input_match": False,
                    "validation_required_output_roles": ["checks", "exceptions", "report"],
                }
            )
            code_path = Path(pit_macro_adapter.__file__).resolve()
            manifest = {
                "inputs": [{"path": self._relative(candidates), "sha256": _sha256(candidates)}],
                "code_path": self._relative(code_path),
                "code_sha256": _sha256(code_path),
                "validation_schema": "candidate_first_v1",
                "outputs": [
                    {"role": "checks", "path": self._relative(checks_path), "sha256": _sha256(checks_path)},
                    {"role": "exceptions", "path": self._relative(exceptions_path), "sha256": _sha256(exceptions_path)},
                    {"role": "report", "path": self._relative(report_path), "sha256": _sha256(report_path)},
                ],
                "exceptions_path": self._relative(exceptions_path),
                "exceptions_sha256": _sha256(exceptions_path),
                "qualification_status": "PASS",
                "failed_check_rows": 0,
                "historical_backtest_allowed": True,
                "model_promotion_allowed": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checks, summary, _ = validate_dataset(ROOT, rule, pd.Timestamp("2026-07-17"))
            self.assertEqual(summary["status"], "pass")
            match = checks.loc[checks["check"].eq("validation_dataset_input_match")].iloc[0]
            self.assertEqual(match["status"], "pass")
            self.assertIn("post_validation_promotion", match["detail"])

    @staticmethod
    def _stock_master_sources(delist_date: str = "2009-12-29") -> dict[str, pd.DataFrame]:
        return {
            "sse_main": pd.DataFrame([{"证券代码": "600000", "证券简称": "浦发银行", "上市日期": "1999-11-10"}]),
            "sse_star": pd.DataFrame(
                [
                    {"证券代码": "688001", "证券简称": "华兴源创", "上市日期": "2019-07-22"},
                    {"证券代码": "689009", "证券简称": "九号公司", "上市日期": "2020-10-29"},
                ]
            ),
            "sse_delisted": pd.DataFrame(
                [{"A_STOCK_CODE": "600001", "COMPANY_ABBR": "邯郸钢铁", "LIST_DATE": "1998-01-22", "DELIST_DATE": delist_date}]
            ),
            "szse_a": pd.DataFrame(
                [
                    {"A股代码": "000001", "A股简称": "平安银行", "A股上市日期": "1991-04-03"},
                    {"A股代码": "302132", "A股简称": "中航成飞", "A股上市日期": "2010-08-27"},
                ]
            ),
            "szse_delisted": pd.DataFrame(
                [
                    {"证券代码": "000003", "证券简称": "PT金田A", "上市日期": "1991-01-14", "终止上市日期": "2002-06-14"},
                    {"证券代码": "200003", "证券简称": "PT金田B", "上市日期": "1993-06-29", "终止上市日期": "2002-06-14"},
                ]
            ),
        }

    def test_stock_master_is_event_sourced_without_future_delist_leakage(self):
        result = build_stock_security_master(self._stock_master_sources(), "fixture:v1", "2020-01-01")
        self.assertEqual(result["asset"].nunique(), 6)
        self.assertIn("302132", set(result["asset"]))
        self.assertNotIn("200003", set(result["asset"]))
        self.assertNotIn("689009", set(result["asset"]))
        self.assertEqual(result["list_status"].value_counts().to_dict(), {"listed": 6, "delisted": 2})
        self.assertTrue(result.loc[result["list_status"].eq("listed"), "delist_date"].isna().all())
        exits = result[result["list_status"].eq("delisted")]
        self.assertTrue((exits["available_date"] == exits["delist_date"]).all())

        before_exit = build_stock_security_master(
            self._stock_master_sources(delist_date="2025-01-02"), "fixture:v2", "2020-01-01"
        )
        self.assertEqual(before_exit.loc[before_exit["asset"].eq("600001"), "list_status"].tolist(), ["listed"])

    def test_stock_master_rejects_conflicting_lifecycle_identity(self):
        sources = self._stock_master_sources()
        sources["sse_main"] = pd.concat(
            [
                sources["sse_main"],
                pd.DataFrame([{"证券代码": "600001", "证券简称": "冲突记录", "上市日期": "1999-01-01"}]),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "conflicting lifecycle identity"):
            build_stock_security_master(sources, "fixture:v1", "2020-01-01")

    def test_stock_master_reconciles_code_migration_without_backdating_new_code(self):
        sources = self._stock_master_sources()
        sources["joinquant_stock"] = pd.DataFrame(
            [
                {
                    "code": "300114.XSHE",
                    "display_name": "中航电测",
                    "start_date": "2010-08-27",
                    "end_date": "2025-02-14",
                },
                {
                    "code": "302132.XSHE",
                    "display_name": "中航成飞",
                    "start_date": "2025-02-17",
                    "end_date": "2200-01-01",
                },
                {
                    "code": "689009.XSHG",
                    "display_name": "九号公司",
                    "start_date": "2020-10-29",
                    "end_date": "2200-01-01",
                },
            ]
        )
        result = build_stock_security_master(sources, "fixture:v1", "2026-01-01")
        listed = result[result["list_status"].eq("listed")].set_index("asset")
        self.assertEqual(str(listed.loc["302132", "list_date"].date()), "2025-02-17")
        self.assertEqual(listed.loc["302132", "lifecycle_resolution"], "joinquant_late_start_guard")
        self.assertEqual(str(listed.loc["300114", "list_date"].date()), "2010-08-27")
        self.assertNotIn("689009", listed.index)
        predecessor_exit = result[(result["asset"].eq("300114")) & result["list_status"].eq("delisted")]
        self.assertEqual(str(predecessor_exit.iloc[0]["delist_date"].date()), "2025-02-14")

    def test_stock_master_applies_official_code_lineage_without_overlapping_identifiers(self):
        sources = self._stock_master_sources()
        sources["sse_main"] = pd.concat(
            [
                sources["sse_main"],
                pd.DataFrame([{"\u8bc1\u5238\u4ee3\u7801": "601607", "\u8bc1\u5238\u7b80\u79f0": "\u4e0a\u6d77\u533b\u836f", "\u4e0a\u5e02\u65e5\u671f": "1994-03-24"}]),
            ],
            ignore_index=True,
        )
        sources["joinquant_stock"] = pd.DataFrame(
            [
                {
                    "code": "600849.XSHG",
                    "display_name": "\u4e0a\u836f\u8f6c\u6362",
                    "start_date": "1994-03-24",
                    "end_date": "2010-03-08",
                },
                {
                    "code": "601607.XSHG",
                    "display_name": "\u4e0a\u6d77\u533b\u836f",
                    "start_date": "1994-03-24",
                    "end_date": "2200-01-01",
                },
            ]
        )
        lineage = {
            "code_migrations": [
                {
                    "predecessor": "600849",
                    "successor": "601607",
                    "effective_date": "2010-03-05",
                    "predecessor_end_date": "2010-03-08",
                    "successor_start_date": "2010-03-09",
                    "identity_continuity": True,
                }
            ]
        }
        result = build_stock_security_master(sources, "fixture:v1", "2020-01-01", lineage)
        listed = result[result["event_type"].eq("listing")].set_index("asset")
        self.assertEqual(str(listed.loc["601607", "list_date"].date()), "2010-03-09")
        self.assertEqual(listed.loc["601607", "predecessor_asset"], "600849")
        predecessor_exit = result[(result["asset"].eq("600849")) & result["event_type"].eq("delisting")].iloc[0]
        self.assertEqual(str(predecessor_exit["delist_date"].date()), "2010-03-08")
        self.assertEqual(predecessor_exit["successor_asset"], "601607")

    def test_stock_industry_snapshot_delays_availability_and_marks_retroactive_rows(self):
        source = pd.DataFrame(
            [
                {
                    "\u80a1\u7968\u4ee3\u7801": "000001",
                    "\u8ba1\u5165\u65e5\u671f": "2014-02-21",
                    "\u884c\u4e1a\u4ee3\u7801": "480101",
                    "\u66f4\u65b0\u65e5\u671f": "2024-09-27 09:08:00",
                },
                {
                    "\u80a1\u7968\u4ee3\u7801": "000001",
                    "\u8ba1\u5165\u65e5\u671f": "2021-07-30",
                    "\u884c\u4e1a\u4ee3\u7801": "480301",
                    "\u66f4\u65b0\u65e5\u671f": "2025-12-15 16:33:00",
                },
                {
                    "\u80a1\u7968\u4ee3\u7801": "999999",
                    "\u8ba1\u5165\u65e5\u671f": "2020-01-01",
                    "\u884c\u4e1a\u4ee3\u7801": "110101",
                    "\u66f4\u65b0\u65e5\u671f": "2020-01-02 10:00:00",
                },
            ]
        )
        master = pd.DataFrame(
            [{"asset": "000001", "event_type": "listing", "list_date": "1991-04-03", "delist_date": None}]
        )
        result = build_stock_industry_history(source, master, "fixture:v1", "2026-07-17")
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["asset"]), {"000001"})
        first = result.iloc[0]
        self.assertEqual(str(first["available_date"].date()), "2024-09-28")
        self.assertEqual(str(first["effective_to"].date()), "2021-07-30")
        self.assertFalse(bool(first["pit_actionable"]))
        self.assertFalse(bool(result["retrieval_provenance_verified"].any()))

    def test_stock_industry_snapshot_filters_rows_unavailable_by_as_of(self):
        source = pd.DataFrame(
            [
                {
                    "\u80a1\u7968\u4ee3\u7801": "600000",
                    "\u8ba1\u5165\u65e5\u671f": "2020-01-01",
                    "\u884c\u4e1a\u4ee3\u7801": "480301",
                    "\u66f4\u65b0\u65e5\u671f": "2020-01-02 10:00:00",
                },
                {
                    "\u80a1\u7968\u4ee3\u7801": "600000",
                    "\u8ba1\u5165\u65e5\u671f": "2021-01-01",
                    "\u884c\u4e1a\u4ee3\u7801": "480401",
                    "\u66f4\u65b0\u65e5\u671f": "2022-01-01 10:00:00",
                },
            ]
        )
        result = build_stock_industry_history(
            source,
            pd.DataFrame(
                [{"asset": "600000", "event_type": "listing", "list_date": "1999-11-10", "delist_date": None}]
            ),
            "fixture:v1",
            "2021-12-31",
        )
        self.assertEqual(result["industry_code"].tolist(), ["480301"])
        self.assertTrue(pd.isna(result.loc[0, "effective_to"]))

    def test_stock_industry_history_splits_a_current_code_across_official_lineage(self):
        source = pd.DataFrame(
            [
                {
                    "\u80a1\u7968\u4ee3\u7801": "601607",
                    "\u8ba1\u5165\u65e5\u671f": "1994-03-24",
                    "\u884c\u4e1a\u4ee3\u7801": "370401",
                    "\u66f4\u65b0\u65e5\u671f": "2015-10-27 15:29:00",
                }
            ]
        )
        master = pd.DataFrame(
            [
                {"asset": "600849", "event_type": "listing", "list_date": "1994-03-24", "delist_date": None},
                {"asset": "600849", "event_type": "delisting", "list_date": "1994-03-24", "delist_date": "2010-03-08"},
                {"asset": "601607", "event_type": "listing", "list_date": "2010-03-09", "delist_date": None},
            ]
        )
        lineage = {
            "code_migrations": [
                {
                    "predecessor": "600849",
                    "successor": "601607",
                    "successor_start_date": "2010-03-09",
                }
            ]
        }
        result = build_stock_industry_history(source, master, "fixture:v1", "2026-07-17", lineage)
        self.assertEqual(set(result["asset"]), {"600849", "601607"})
        predecessor = result[result["asset"].eq("600849")].iloc[0]
        successor = result[result["asset"].eq("601607")].iloc[0]
        self.assertEqual(str(predecessor["effective_to"].date()), "2010-03-09")
        self.assertEqual(str(successor["effective_from"].date()), "2010-03-09")
        self.assertEqual(successor["security_code_resolution"], "carried_forward_successor")
        self.assertFalse(bool(predecessor["pit_actionable"]))

    def test_hfq_factor_events_are_bounded_by_security_lifecycle(self):
        raw = pd.DataFrame(
            [
                {"date": "1900-01-01", "hfq_factor": "1.0"},
                {"date": "2010-08-27", "hfq_factor": "1.0"},
                {"date": "2023-06-06", "hfq_factor": "5.97"},
                {"date": "2025-06-17", "hfq_factor": "5.98"},
                {"date": "2026-06-16", "hfq_factor": "6.06"},
            ]
        )
        predecessor = normalise_factor_events(raw, "300114", "2010-08-27", "2025-02-14", "2026-07-17")
        successor = normalise_factor_events(raw, "302132", "2025-02-17", None, "2026-07-17")
        self.assertEqual(str(predecessor["effective_date"].max().date()), "2023-06-06")
        self.assertEqual(str(successor["effective_date"].min().date()), "2025-02-17")
        self.assertEqual(successor.iloc[0]["adj_factor"], 5.97)
        self.assertNotIn(pd.Timestamp("2010-08-27"), set(successor["effective_date"]))

    def test_sina_factor_parser_rejects_non_literal_payload(self):
        good = parse_sina_hfq_response("var x={'data':[{'d':'2020-01-01','f':'1.0'}]}\n")
        self.assertEqual(good.loc[0, "hfq_factor"], "1.0")
        with self.assertRaisesRegex(ValueError, "invalid Sina"):
            parse_sina_hfq_response("var x=__import__('os').system('echo unsafe')\n")

    def test_adjustment_factor_alignment_threshold_cannot_silently_relax(self):
        self.assertTrue(dividend_alignment_passes({"match_ratio": 0.995}))
        self.assertFalse(dividend_alignment_passes({"match_ratio": 0.994999}))
        self.assertFalse(dividend_alignment_passes({"match_ratio": None}))

    def test_adjustment_validator_uses_adjusted_close_jump_not_unreliable_pre_close(self):
        factors = pd.DataFrame(
            [
                {"asset": asset, "effective_date": pd.Timestamp(date), "adj_factor": factor}
                for asset, date, factor in (
                    ("000001", "2020-01-01", 1.0),
                    ("000001", "2020-01-03", 2.0),
                    ("000002", "2020-01-01", 1.0),
                    ("000002", "2020-01-03", 3.0),
                )
            ]
        ).sort_values(["effective_date", "asset"])
        with tempfile.TemporaryDirectory() as temp_dir:
            daily = Path(temp_dir)
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20200102", "close": 10.0, "pre_close": 10.0},
                    {"ts_code": "000002.SZ", "trade_date": "20200102", "close": 10.0, "pre_close": 10.0},
                ]
            ).to_csv(daily / "trade_date=20200102.csv", index=False)
            pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20200103", "close": 5.2, "pre_close": 10.0},
                    {"ts_code": "000002.SZ", "trade_date": "20200103", "close": 5.0, "pre_close": 10.0},
                ]
            ).to_csv(daily / "trade_date=20200103.csv", index=False)
            checks, summary, lineage = validate_factor_transitions(factors, daily, set())
        by_asset = checks.set_index("asset")
        self.assertAlmostEqual(float(by_asset.loc["000001", "adjusted_close_jump"]), 0.04)
        self.assertGreater(float(by_asset.loc["000001", "pre_close_continuity_error_diagnostic"]), 0.9)
        self.assertFalse(bool(by_asset.loc["000001", "large_adjusted_jump"]))
        self.assertTrue(bool(by_asset.loc["000002", "large_adjusted_jump"]))
        self.assertTrue(bool(by_asset.loc["000002", "continuous_trading_large_jump"]))
        self.assertEqual(summary["large_adjusted_jump_count"], 1)
        self.assertEqual(summary["continuous_trading_large_jump_count"], 1)
        self.assertEqual(len(lineage), 2)

    @staticmethod
    def _baostock_rows(asset: str, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "code": f"sh.{asset}",
            "open": "10",
            "high": "10",
            "low": "10",
            "close": "10",
            "preclose": "10",
            "volume": "1000000",
            "amount": "10000000",
            "adjustflag": "3",
            "turn": "1",
            "tradestatus": "1",
            "pctChg": "0",
            "peTTM": "8",
            "pbMRQ": "1",
            "psTTM": "1",
            "pcfNcfTTM": "5",
            "isST": "0",
        }
        return pd.DataFrame([{**defaults, **row} for row in rows], columns=sorted(REQUIRED_HISTORY_COLUMNS))

    def test_baostock_history_is_lifecycle_bounded_and_code_strict(self):
        raw = self._baostock_rows(
            "600000",
            [
                {"date": "2019-12-31"},
                {"date": "2020-01-02"},
                {"date": "2020-01-03"},
                {"date": "2020-01-06"},
            ],
        )
        result = normalise_baostock_history(raw, "600000", "2020-01-02", "2020-01-03", "2020-12-31")
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2020-01-02", "2020-01-03"])
        self.assertEqual(set(result["asset"]), {"600000"})
        bad = raw.copy()
        bad.loc[1, "code"] = "sh.600001"
        with self.assertRaisesRegex(ValueError, "different security code"):
            normalise_baostock_history(bad, "600000", "2020-01-02", "2020-01-03", "2020-12-31")

    def test_market_history_collection_scope_excludes_pre_backtest_delistings(self):
        lifecycles = pd.DataFrame(
            [
                {"asset": "000001", "list_date": "1991-01-01", "delist_date": None},
                {"asset": "000002", "list_date": "1991-01-01", "delist_date": "2004-12-31"},
                {"asset": "000003", "list_date": "1991-01-01", "delist_date": "2005-01-01"},
            ]
        )
        lifecycles["list_date"] = pd.to_datetime(lifecycles["list_date"])
        lifecycles["delist_date"] = pd.to_datetime(lifecycles["delist_date"])
        self.assertEqual(relevant_lifecycles(lifecycles)["asset"].tolist(), ["000001", "000003"])

    def test_market_history_global_collection_limit_is_partitioned_without_overlap(self):
        lifecycles = pd.DataFrame({"asset": [f"{value:06d}" for value in range(10)]})
        first = select_collection_shard(lifecycles, shard_count=2, shard_index=0, collect_limit=5)
        second = select_collection_shard(lifecycles, shard_count=2, shard_index=1, collect_limit=5)
        self.assertEqual(first["asset"].tolist(), ["000000", "000002", "000004"])
        self.assertEqual(second["asset"].tolist(), ["000001", "000003"])
        self.assertEqual(set(first["asset"]).intersection(second["asset"]), set())
        self.assertEqual(set(first["asset"]).union(second["asset"]), set(lifecycles.head(5)["asset"]))

    def test_market_history_asset_file_preserves_requested_order(self):
        lifecycles = pd.DataFrame(
            {
                "asset": ["600001", "600002", "600003"],
                "exchange": ["SSE", "SSE", "SSE"],
                "list_date": pd.to_datetime(["2000-01-01"] * 3),
                "delist_date": [pd.NaT] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "assets.csv"
            pd.DataFrame({"asset": ["600003", "600001"]}).to_csv(path, index=False)
            result = filter_lifecycles_by_asset_file(lifecycles, path)
        self.assertEqual(result["asset"].tolist(), ["600003", "600001"])

    def test_market_history_collection_limit_counts_only_pending_assets(self):
        lifecycles = pd.DataFrame(
            [
                {
                    "asset": asset,
                    "exchange": "SZSE",
                    "list_date": pd.Timestamp("2020-01-01"),
                    "delist_date": pd.NaT,
                }
                for asset in ("000001", "000002", "000003")
            ]
        )

        class FakeSession:
            def __init__(self, server_ip):
                self.server_ip = server_ip

            def close(self):
                pass

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(pit_stock_market_history_builder, "RAW_DIR", Path(tmp)),
            patch.object(
                pit_stock_market_history_builder,
                "_valid_cache",
                side_effect=lambda asset, start, end: asset == "000001",
            ),
            patch.object(pit_stock_market_history_builder, "BaoStockSession", FakeSession),
            patch.object(
                pit_stock_market_history_builder,
                "_fetch_one",
                side_effect=lambda session, row, as_of, server_ip: {
                    "status": "failed",
                    "asset": row.asset,
                    "error": "BaoStock login failed: 10001011 blacklist",
                },
            ) as fetch,
        ):
            result = pit_stock_market_history_builder.collect_raw_history(
                lifecycles,
                pd.Timestamp("2026-07-17"),
                "127.0.0.1",
                collect_limit=1,
                sleep_seconds=0,
            )

        self.assertEqual(result["target_assets"], 3)
        self.assertEqual(result["already_cached_assets"], 1)
        self.assertEqual(result["pending_assets_before_limit"], 2)
        self.assertEqual(result["selected_assets"], 1)
        self.assertEqual(fetch.call_args.args[1].asset, "000002")
        self.assertEqual(result["circuit_breaker"], "provider_blacklist")
        self.assertEqual(result["deferred_assets"], [])

    def test_market_history_orchestrator_worker_is_collection_only(self):
        command = worker_command(
            as_of="2026-07-17",
            server_ip="114.94.20.42",
            workers=1,
            shard_index=0,
            collect_limit=64,
            sleep_seconds=0.05,
        )
        self.assertIn("--collect-only", command)
        self.assertEqual(command[command.index("--shard-count") + 1], "1")
        self.assertEqual(command[command.index("--shard-index") + 1], "0")
        self.assertEqual(command[command.index("--collect-limit") + 1], "64")

    def test_tushare_daily_refresher_reuses_valid_dates_and_fetches_only_missing_dates(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            daily_dir = base / "daily"
            calendar_path = base / "trade_calendar.csv"
            manifest_dir = base / "manifests"
            status_path = base / "status.csv"
            calendar_path.write_text("date\n2026-07-16\n2026-07-17\n", encoding="utf-8")

            def raw_for(trade_date: str) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ts_code": "000001.SZ",
                            "trade_date": trade_date,
                            "open": 10.0,
                            "high": 10.5,
                            "low": 9.8,
                            "close": 10.2,
                            "pre_close": 10.0,
                            "change": 0.2,
                            "pct_chg": 2.0,
                            "vol": 100.0,
                            "amount": 1000.0,
                        }
                    ]
                )

            daily_dir.mkdir()
            existing = pit_tushare_daily_refresher.normalise_daily(
                raw_for("20260716"), "20260716", "2026-07-16T16:00:00+08:00"
            )
            existing.to_csv(
                daily_dir / "trade_date=20260716.csv",
                index=False,
                encoding="utf-8-sig",
            )

            class FakePro:
                def __init__(self):
                    self.calls = []

                def daily(self, trade_date):
                    self.calls.append(trade_date)
                    return raw_for(trade_date)

            pro = FakePro()
            paths = {
                "DAILY_DIR": daily_dir,
                "CALENDAR_PATH": calendar_path,
                "MANIFEST_DIR": manifest_dir,
                "LATEST_MANIFEST": manifest_dir / "latest.json",
                "RUN_DIR": manifest_dir / "runs",
                "STATUS_PATH": status_path,
            }
            with patch.multiple(pit_tushare_daily_refresher, **paths):
                result = pit_tushare_daily_refresher.refresh_daily(
                    "2026-07-16",
                    "2026-07-17",
                    max_calls=2,
                    sleep_seconds=0,
                    pro=pro,
                )

            self.assertEqual(pro.calls, ["20260717"])
            self.assertEqual(result["qualification_status"], "REFRESH_COMPLETE_RAW_DAILY_ONLY")
            self.assertEqual(result["reused_trade_dates"], 1)
            self.assertEqual(result["fetched_trade_dates"], 1)
            self.assertTrue((daily_dir / "trade_date=20260717.csv").is_file())
            self.assertTrue((manifest_dir / "latest.json").is_file())

    def test_market_history_orchestrator_rejects_provider_unsafe_concurrency(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 1"):
            run_orchestration(
                as_of="2026-07-17",
                workers=2,
                collect_limit=10,
                server_ip="127.0.0.1",
            )

    def test_baostock_transport_fails_on_peer_eof_instead_of_spinning(self):
        import baostock.common.context as context

        class ClosedSocket:
            def sendall(self, payload):
                self.payload = payload

            def recv(self, size):
                return b""

        had_socket = hasattr(context, "default_socket")
        prior_socket = getattr(context, "default_socket", None)
        context.default_socket = ClosedSocket()
        try:
            with self.assertRaisesRegex(ConnectionError, "peer closed"):
                _safe_baostock_send_msg("fixture")
        finally:
            if had_socket:
                context.default_socket = prior_socket
            else:
                delattr(context, "default_socket")

    def test_market_history_orchestrator_compares_report_freshness_in_local_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "shard.json"
            report_path.write_text(
                json.dumps({"created_at": "2026-07-18T14:20:43+08:00"}), encoding="utf-8"
            )
            report, status = _read_fresh_shard_report(
                report_path, pd.Timestamp("2026-07-18T14:20:40+08:00")
            )
        self.assertEqual(status, "ok")
        self.assertEqual(report["created_at"], "2026-07-18T14:20:43+08:00")

    def test_market_history_cache_inventory_separates_missing_from_failed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pit_stock_market_history_builder, "RAW_DIR", Path(tmp)
        ):
            start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
            self.assertEqual(
                pit_stock_market_history_builder._cache_artifact_state("600000", start, end),
                ("missing", "not_collected"),
            )
            metadata = {"status": "failed", "error": "provider_timeout"}
            (Path(tmp) / "600000.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(
                pit_stock_market_history_builder._cache_artifact_state("600000", start, end),
                ("failed", "provider_timeout"),
            )

    def test_market_history_uses_latest_trading_day_for_weekend_as_of(self):
        with tempfile.TemporaryDirectory() as tmp:
            calendar_path = Path(tmp) / "trade_calendar.csv"
            calendar_path.write_text(
                "date\n2026-07-16\n2026-07-17\n2026-07-20\n",
                encoding="utf-8",
            )
            result = effective_market_date("2026-07-18", calendar_path)
        self.assertEqual(result, pd.Timestamp("2026-07-17"))

    def test_market_history_output_build_is_asset_streamed_without_global_concat(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            raw_dir = base / "raw"
            raw_dir.mkdir()
            master_path = base / "stock_security_master.csv"
            master_path.write_text("fixture\n", encoding="utf-8")
            lifecycles = pd.DataFrame(
                [
                    {
                        "asset": "000001",
                        "exchange": "SZSE",
                        "list_date": pd.Timestamp("2019-01-01"),
                        "delist_date": pd.NaT,
                    },
                    {
                        "asset": "000002",
                        "exchange": "SZSE",
                        "list_date": pd.Timestamp("2019-01-01"),
                        "delist_date": pd.NaT,
                    },
                ]
            )
            for asset in lifecycles["asset"]:
                (raw_dir / f"{asset}.csv.gz").write_bytes(b"fixture")
                (raw_dir / f"{asset}.json").write_text(
                    json.dumps({"sha256": asset * 10 + asset[:4]}), encoding="utf-8"
                )

            def cache_paths(asset: str):
                return raw_dir / f"{asset}.csv.gz", raw_dir / f"{asset}.json"

            def load_history(row, as_of, validate_cache=True):
                asset = str(row.asset).zfill(6)
                return pd.DataFrame({"asset": [asset]}), {"sha256": asset * 10 + asset[:4]}

            def trade_state(history, list_date):
                asset = str(history.iloc[0]["asset"])
                return pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2020-01-02"),
                            "asset": asset,
                            "is_paused": asset == "000001",
                            "is_st": asset == "000002",
                            "pre_close": 10.0,
                            "has_price_limit": asset == "000002",
                            "limit_up": 11.0 if asset == "000002" else None,
                            "limit_down": 9.0 if asset == "000002" else None,
                            "price_limit_rate": 0.1 if asset == "000002" else None,
                            "limit_rule": "regular_st_5" if asset == "000002" else "paused",
                            "execution_state_known": True,
                            "available_date": pd.Timestamp("2020-01-02"),
                        }
                    ]
                )

            def valuation(history, dividends=None):
                asset = str(history.iloc[0]["asset"])
                return pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2020-01-31"),
                            "asset": asset,
                            "pe_ttm": 8.0,
                            "pb": 1.0,
                            "dividend_yield": 0.05,
                            "market_cap": 1_000_000_000.0,
                            "market_cap_basis": "fixture",
                            "dividend_yield_basis": "fixture",
                            "available_date": pd.Timestamp("2020-01-31"),
                        }
                    ]
                )

            paths = {
                "MASTER_PATH": master_path,
                "DIVIDEND_PATH": base / "missing_dividends.csv",
                "RAW_DIR": raw_dir,
                "COLLECTION_INVENTORY": raw_dir / "inventory.csv",
                "ASSET_SUMMARY": raw_dir / "summary.csv",
                "OBSERVATION_DIR": base / "observations",
                "TRADE_OUTPUT": base / "stock_trade_state.csv",
                "VALUATION_OUTPUT": base / "stock_valuation_history.csv",
                "TRADE_OBSERVATION": base / "observations" / "trade.csv",
                "VALUATION_OBSERVATION": base / "observations" / "valuation.csv",
                "COMBINED_MANIFEST": base / "manifests" / "combined.json",
                "TRADE_MANIFEST": base / "manifests" / "trade.json",
                "VALUATION_MANIFEST": base / "manifests" / "valuation.json",
            }
            with (
                patch.multiple(pit_stock_market_history_builder, **paths),
                patch.object(pit_stock_market_history_builder, "load_lifecycles", return_value=lifecycles),
                patch.object(pit_stock_market_history_builder, "relevant_lifecycles", side_effect=lambda x: x),
                patch.object(
                    pit_stock_market_history_builder,
                    "_cache_artifact_state",
                    return_value=("completed", "validated_cache"),
                ),
                patch.object(pit_stock_market_history_builder, "_cache_paths", side_effect=cache_paths),
                patch.object(pit_stock_market_history_builder, "_load_cached_history", side_effect=load_history),
                patch.object(pit_stock_market_history_builder, "build_trade_state", side_effect=trade_state),
                patch.object(pit_stock_market_history_builder, "build_monthly_valuation", side_effect=valuation),
                patch.object(pit_stock_market_history_builder.pd, "concat", side_effect=AssertionError("global concat")),
            ):
                result = pit_stock_market_history_builder.build_outputs("2026-07-17")

            self.assertEqual(result["build_mode"], "two_pass_asset_streaming_v1")
            self.assertEqual(result["completed_assets"], 2)
            self.assertTrue(result["historical_backtest_allowed"])
            trade = pd.read_csv(base / "stock_trade_state.csv", dtype={"asset": str})
            summary = pd.read_csv(raw_dir / "summary.csv", dtype={"asset": str})
            self.assertEqual(trade["asset"].str.zfill(6).tolist(), ["000001", "000002"])
            self.assertEqual(summary["status"].tolist(), ["completed", "completed"])

    def test_market_history_promotes_trade_state_independently_from_valuation(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            raw_dir = base / "raw"
            raw_dir.mkdir()
            master_path = base / "stock_security_master.csv"
            master_path.write_text("fixture\n", encoding="utf-8")
            lifecycles = pd.DataFrame(
                [
                    {
                        "asset": "000001",
                        "exchange": "SZSE",
                        "list_date": pd.Timestamp("2019-01-01"),
                        "delist_date": pd.NaT,
                    }
                ]
            )
            (raw_dir / "000001.csv.gz").write_bytes(b"fixture")
            (raw_dir / "000001.json").write_text(
                json.dumps({"sha256": "1" * 64}), encoding="utf-8"
            )

            def cache_paths(asset: str):
                return raw_dir / f"{asset}.csv.gz", raw_dir / f"{asset}.json"

            def trade_state(history, list_date):
                return pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2020-01-02"),
                            "asset": "000001",
                            "is_paused": True,
                            "is_st": False,
                            "pre_close": 10.0,
                            "has_price_limit": False,
                            "limit_up": None,
                            "limit_down": None,
                            "price_limit_rate": None,
                            "limit_rule": "paused",
                            "execution_state_known": True,
                            "available_date": pd.Timestamp("2020-01-02"),
                        },
                        {
                            "date": pd.Timestamp("2020-01-03"),
                            "asset": "000001",
                            "is_paused": False,
                            "is_st": True,
                            "pre_close": 10.0,
                            "has_price_limit": True,
                            "limit_up": 10.5,
                            "limit_down": 9.5,
                            "price_limit_rate": 0.05,
                            "limit_rule": "regular_st_5",
                            "execution_state_known": True,
                            "available_date": pd.Timestamp("2020-01-03"),
                        },
                    ]
                )

            def incomplete_valuation(history, dividends=None):
                return pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2020-01-31"),
                            "asset": "000001",
                            "pe_ttm": None,
                            "pb": 1.0,
                            "dividend_yield": 0.05,
                            "market_cap": 1_000_000_000.0,
                            "market_cap_basis": "fixture",
                            "dividend_yield_basis": "fixture",
                            "available_date": pd.Timestamp("2020-01-31"),
                        }
                    ]
                )

            paths = {
                "MASTER_PATH": master_path,
                "DIVIDEND_PATH": base / "missing_dividends.csv",
                "RAW_DIR": raw_dir,
                "COLLECTION_INVENTORY": raw_dir / "inventory.csv",
                "ASSET_SUMMARY": raw_dir / "summary.csv",
                "OBSERVATION_DIR": base / "observations",
                "TRADE_OUTPUT": base / "stock_trade_state.csv",
                "VALUATION_OUTPUT": base / "stock_valuation_history.csv",
                "TRADE_OBSERVATION": base / "observations" / "trade.csv",
                "VALUATION_OBSERVATION": base / "observations" / "valuation.csv",
                "COMBINED_MANIFEST": base / "manifests" / "combined.json",
                "TRADE_MANIFEST": base / "manifests" / "trade.json",
                "VALUATION_MANIFEST": base / "manifests" / "valuation.json",
            }
            with (
                patch.multiple(pit_stock_market_history_builder, **paths),
                patch.object(pit_stock_market_history_builder, "load_lifecycles", return_value=lifecycles),
                patch.object(pit_stock_market_history_builder, "relevant_lifecycles", side_effect=lambda x: x),
                patch.object(
                    pit_stock_market_history_builder,
                    "_cache_artifact_state",
                    return_value=("completed", "validated_cache"),
                ),
                patch.object(pit_stock_market_history_builder, "_cache_paths", side_effect=cache_paths),
                patch.object(
                    pit_stock_market_history_builder,
                    "_load_cached_history",
                    return_value=(pd.DataFrame({"asset": ["000001"]}), {"sha256": "1" * 64}),
                ),
                patch.object(pit_stock_market_history_builder, "build_trade_state", side_effect=trade_state),
                patch.object(
                    pit_stock_market_history_builder,
                    "build_monthly_valuation",
                    side_effect=incomplete_valuation,
                ),
            ):
                result = pit_stock_market_history_builder.build_outputs("2026-07-17")

            self.assertTrue(result["trade_state"]["historical_backtest_allowed"])
            self.assertFalse(result["valuation"]["historical_backtest_allowed"])
            self.assertFalse(result["historical_backtest_allowed"])
            self.assertTrue((base / "stock_trade_state.csv").is_file())
            self.assertFalse((base / "stock_valuation_history.csv").is_file())
            self.assertTrue((base / "observations" / "valuation.csv").is_file())
            summary = pd.read_csv(raw_dir / "summary.csv", dtype={"asset": str})
            self.assertEqual(summary.loc[0, "status"], "partial")
            self.assertEqual(summary.loc[0, "trade_status"], "completed")
            self.assertEqual(summary.loc[0, "valuation_status"], "failed")

    def test_trade_state_rules_keep_special_days_explicit_and_fail_closed(self):
        raw = self._baostock_rows(
            "600000",
            [
                {"date": "2020-01-02", "preclose": "10"},
                {"date": "2020-01-03", "preclose": "10"},
                {"date": "2020-01-06", "tradestatus": "0", "preclose": "10"},
                {"date": "2020-01-07", "preclose": "10"},
                {"date": "2020-01-08", "preclose": "10", "isST": "1"},
            ],
        )
        history = normalise_baostock_history(raw, "600000", "2020-01-02", None, "2020-01-08")
        result = build_trade_state(history, "2020-01-02").set_index("date")
        self.assertEqual(result.loc[pd.Timestamp("2020-01-02"), "limit_rule"], "first_session_special_unknown")
        self.assertFalse(bool(result.loc[pd.Timestamp("2020-01-02"), "execution_state_known"]))
        self.assertTrue(bool(result.loc[pd.Timestamp("2020-01-06"), "is_paused"]))
        self.assertEqual(result.loc[pd.Timestamp("2020-01-07"), "limit_rule"], "resumption_limit_unknown")
        self.assertFalse(bool(result.loc[pd.Timestamp("2020-01-07"), "execution_state_known"]))
        self.assertEqual(result.loc[pd.Timestamp("2020-01-08"), "limit_rule"], "regular_st_5")
        self.assertAlmostEqual(float(result.loc[pd.Timestamp("2020-01-08"), "limit_up"]), 10.50)
        self.assertAlmostEqual(float(result.loc[pd.Timestamp("2020-01-08"), "limit_down"]), 9.50)

        star_raw = self._baostock_rows(
            "688001",
            [{"date": date, "preclose": "10"} for date in pd.bdate_range("2019-07-22", periods=6).strftime("%Y-%m-%d")],
        )
        star = normalise_baostock_history(star_raw, "688001", "2019-07-22", None, "2019-07-29")
        star_state = build_trade_state(star, "2019-07-22")
        self.assertEqual(star_state.iloc[:5]["limit_rule"].unique().tolist(), ["no_price_limit_listing_window"])
        self.assertEqual(star_state.iloc[5]["limit_rule"], "regular_growth_20")
        self.assertAlmostEqual(float(star_state.iloc[5]["limit_up"]), 12.00)

    def test_trade_state_rules_respect_reform_boundaries_and_code_migrations(self):
        pre_reform_raw = self._baostock_rows(
            "300001",
            [
                {"date": "2020-08-21", "preclose": "10", "isST": "1"},
                {"date": "2020-08-24", "preclose": "10", "isST": "1"},
            ],
        )
        pre_reform = normalise_baostock_history(
            pre_reform_raw,
            "300001",
            "2009-10-30",
            None,
            "2020-08-24",
        )
        reform_state = build_trade_state(pre_reform, "2009-10-30")
        self.assertEqual(reform_state.iloc[0]["limit_rule"], "regular_st_5")
        self.assertAlmostEqual(float(reform_state.iloc[0]["price_limit_rate"]), 0.05)
        self.assertEqual(reform_state.iloc[1]["limit_rule"], "regular_growth_20")
        self.assertAlmostEqual(float(reform_state.iloc[1]["price_limit_rate"]), 0.20)

        migrated_raw = self._baostock_rows(
            "302132",
            [{"date": date, "preclose": "10"} for date in pd.bdate_range("2025-02-17", periods=6).strftime("%Y-%m-%d")],
        )
        migrated = normalise_baostock_history(
            migrated_raw,
            "302132",
            "2025-02-17",
            None,
            "2025-02-24",
        )
        migrated_state = build_trade_state(migrated, "2025-02-17", is_ipo=False)
        self.assertEqual(migrated_state.iloc[0]["limit_rule"], "first_session_special_unknown")
        self.assertEqual(migrated_state.iloc[1]["limit_rule"], "regular_growth_20")
        self.assertAlmostEqual(float(migrated_state.iloc[1]["price_limit_rate"]), 0.20)

    def test_no_limit_window_counts_market_sessions_including_paused_days(self):
        raw = self._baostock_rows(
            "688001",
            [
                {"date": "2019-07-22", "preclose": "10"},
                {"date": "2019-07-23", "tradestatus": "0", "preclose": "10"},
                {"date": "2019-07-24", "preclose": "10"},
                {"date": "2019-07-25", "preclose": "10"},
                {"date": "2019-07-26", "preclose": "10"},
                {"date": "2019-07-29", "preclose": "10"},
            ],
        )
        history = normalise_baostock_history(raw, "688001", "2019-07-22", None, "2019-07-29")
        state = build_trade_state(history, "2019-07-22")
        self.assertEqual(state.iloc[0]["limit_rule"], "no_price_limit_listing_window")
        self.assertEqual(state.iloc[1]["limit_rule"], "paused")
        self.assertEqual(state.iloc[4]["limit_rule"], "no_price_limit_listing_window")
        self.assertEqual(state.iloc[5]["limit_rule"], "regular_growth_20")

    def test_name_history_parser_and_status_classification(self):
        html = """
        <html><body><div>公司资料</div><table>
        <tr><td>更名日期</td><td>更名前</td><td>更名后</td></tr>
        <tr><td>2020-01-02</td><td>公司A</td><td>ST公司A</td></tr>
        <tr><td>2021-01-04</td><td>ST公司A</td><td>公司A</td></tr>
        </table></body></html>
        """.encode("gb18030")
        parsed = parse_sina_name_history(html)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed.iloc[0]["new_name"], "ST公司A")
        self.assertEqual(classify_security_name("S*ST公司"), "risk_warning")
        self.assertEqual(classify_security_name("公司退"), "delisting")
        self.assertEqual(classify_security_name("PT公司"), "special_transfer")
        self.assertEqual(classify_security_name("C公司"), "listing_marker")

        sparse_html = """
        <html><body><div>公司资料</div>
        <table><tr><td></td><td></td><td></td></tr></table>
        <table>
        <tr><td>更名日期</td><td>更名前</td><td>更名后</td></tr>
        <tr><td>2020-01-02</td><td></td><td>公司A</td></tr>
        </table></body></html>
        """.encode("utf-8")
        sparse = parse_sina_name_history(sparse_html)
        self.assertEqual(len(sparse), 1)
        self.assertEqual(sparse.iloc[0]["old_name"], "")

        fallback = """
        <table><tr><td class="ct">证券简称更名历史：</td>
        <td colspan="3" class="ccl">公司A 公司B </td></tr></table>
        """.encode("gb18030")
        self.assertEqual(parse_sina_undated_name_summary(fallback), ["公司A", "公司B"])

    def test_name_history_normalisation_is_lifecycle_and_asof_bounded(self):
        lifecycle = type(
            "Lifecycle",
            (),
            {
                "asset": "000001",
                "exchange": "SZSE",
                "list_date": pd.Timestamp("2020-01-01"),
                "delist_date": pd.NaT,
            },
        )()
        raw = pd.DataFrame(
            [
                {"变更日期": "2019-12-31", "证券代码": "1", "变更前简称": "旧", "变更后简称": "早"},
                {"变更日期": "2020-01-02", "证券代码": "1", "变更前简称": "公司", "变更后简称": "ST公司"},
                {"变更日期": "2022-01-02", "证券代码": "1", "变更前简称": "ST公司", "变更后简称": "公司"},
            ]
        )
        result = normalise_name_events(
            raw,
            lifecycle,
            "2021-12-31",
            source_tier="official_exchange",
            data_source="fixture",
            source_url="https://example.invalid",
            source_hash="a" * 64,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["asset"], "000001")
        self.assertEqual(result.iloc[0]["new_status"], "risk_warning")
        self.assertEqual(result.iloc[0]["available_date"], pd.Timestamp("2020-01-02"))

    def test_status_event_builder_collapses_non_execution_renames(self):
        lifecycles = pd.DataFrame(
            [
                {
                    "asset": "600001",
                    "asset_name": "公司C",
                    "exchange": "SSE",
                    "list_date": "2019-01-02",
                    "delist_date": None,
                },
                {
                    "asset": "600002",
                    "asset_name": "公司D",
                    "exchange": "SSE",
                    "list_date": "2019-01-02",
                    "delist_date": None,
                },
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "asset": "600001",
                    "effective_date": "2020-01-02",
                    "old_status": "normal",
                    "new_status": "normal",
                },
                {
                    "asset": "600001",
                    "effective_date": "2021-01-04",
                    "old_status": "normal",
                    "new_status": "risk_warning",
                },
                {
                    "asset": "600001",
                    "effective_date": "2022-01-04",
                    "old_status": "risk_warning",
                    "new_status": "normal",
                },
            ]
        )
        result, unresolved = build_status_events(
            lifecycles,
            events,
            {"600001": "dated_history", "600002": "no_name_changes"},
            "2022-12-31",
        )
        self.assertFalse(unresolved)
        first = result[result["asset"].eq("600001")]
        self.assertEqual(first["execution_status"].tolist(), ["normal", "risk_warning", "normal"])
        self.assertEqual(first["is_st"].tolist(), [False, True, False])
        second = result[result["asset"].eq("600002")]
        self.assertEqual(second["execution_status"].tolist(), ["normal"])

    def test_status_event_builder_fails_closed_on_unresolved_initial_status(self):
        lifecycles = pd.DataFrame(
            [
                {
                    "asset": "600001",
                    "asset_name": "公司A",
                    "exchange": "SSE",
                    "list_date": "2019-01-02",
                    "delist_date": None,
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "asset": "600001",
                    "effective_date": "2020-01-02",
                    "old_status": "unknown",
                    "new_status": "risk_warning",
                }
            ]
        )
        result, unresolved = build_status_events(
            lifecycles,
            events,
            {"600001": "dated_history"},
            "2022-12-31",
        )
        self.assertTrue(result.empty)
        self.assertIn("600001", unresolved)

    def test_official_status_supplement_converts_to_state_transitions(self):
        official = pd.DataFrame(
            [
                {"asset": "601106", "effective_date": "2017-04-21", "execution_status": "risk_warning"},
                {"asset": "601106", "effective_date": "2018-05-09", "execution_status": "normal"},
                {"asset": "601106", "effective_date": "2018-06-01", "execution_status": "normal"},
            ]
        )
        result = official_status_events_as_name_events(official)
        self.assertEqual(result["old_status"].tolist(), ["normal", "risk_warning"])
        self.assertEqual(result["new_status"].tolist(), ["risk_warning", "normal"])

    def test_sse_announcement_classifier_excludes_non_effective_notices(self):
        self.assertIsNone(classify_announcement_title("关于公司股票可能被实施退市风险警示的提示性公告"))
        self.assertIsNone(classify_announcement_title("关于申请撤销公司股票退市风险警示的公告"))
        self.assertEqual(classify_announcement_title("关于公司股票实施退市风险警示的公告"), "risk_warning")
        self.assertEqual(classify_announcement_title("股票交易实行退市风险警示的公告"), "risk_warning")
        self.assertEqual(classify_announcement_title("股票将实行退市风险警示特别处理的公告"), "risk_warning")
        self.assertEqual(classify_announcement_title("股票交易实行特别处理的公告"), "risk_warning")
        self.assertEqual(classify_announcement_title("关于公司股票撤销退市风险警示的公告"), "normal")
        self.assertEqual(classify_announcement_title("关于撤消股票其他特别处理的公告"), "normal")
        self.assertEqual(classify_announcement_title("股票解除退市风险警示公告"), "normal")
        self.assertEqual(
            classify_announcement_title("撤销退市风险警示并实施其他特别处理的公告"),
            "risk_warning",
        )
        self.assertEqual(
            classify_announcement_title("关于撤销退市风险警示并继续实施其他风险警示的公告"),
            "risk_warning",
        )
        self.assertEqual(
            classify_announcement_title("关于公司股票撤销退市风险警示及实施其他风险警示的公告"),
            "risk_warning",
        )
        self.assertEqual(
            classify_announcement_title("关于撤销退市风险警示及将被实施其他风险警示的公告"),
            "risk_warning",
        )
        self.assertEqual(
            classify_announcement_title("关于撤销因重整而被实施退市风险警示暨继续被实施退市风险警示的公告"),
            "risk_warning",
        )
        self.assertEqual(
            classify_announcement_title("关于股票交易实施其他风险警示暨公司股票停牌的提示性公告"),
            "risk_warning",
        )
        self.assertIsNone(classify_announcement_title("关于暂不提交撤销股票退市风险警示的提示性公告"))
        self.assertIsNone(classify_announcement_title("关于公司股票不能撤销其他风险警示的公告"))
        self.assertIsNone(classify_announcement_title("关于撤回撤销退市风险警示申请的公告"))
        self.assertIsNone(classify_announcement_title("关于申请公司股票撤销退市风险警示的公告"))
        self.assertIsNone(classify_announcement_title("关于提交撤销其他特别处理申请的公告"))
        self.assertIsNone(classify_announcement_title("关于股票存在被实施退市风险警示可能的提示性公告"))
        self.assertIsNone(classify_announcement_title("股票存在被实施退市风险警示及暂停上市风险的提示公告"))
        self.assertIsNone(classify_announcement_title("关于公司股票可能继续被实施退市风险警示的公告"))
        self.assertEqual(
            classify_announcement_title("股票于退市整理期交易的第一次风险提示公告"),
            "delisting_date_requires_document",
        )

    def test_sse_announcement_holdout_can_preserve_isolated_removal_event(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(["2006-01-10", "2006-01-11", "2006-01-12"]))
        artifact = {
            "asset": "600613",
            "responses": [
                {
                    "rows": [
                        {
                            "SSEDATE": "2006-01-10",
                            "TITLE": "*ST永生股票获准撤销退市风险警示及其他特别处理的公告",
                            "URL": "/notice/normal.pdf",
                        }
                    ]
                }
            ],
        }
        collapsed, _ = parse_asset_events(artifact, calendar)
        preserved, _ = parse_asset_events(artifact, calendar, collapse_state_changes=False)
        self.assertTrue(collapsed.empty)
        self.assertEqual(preserved["execution_status"].tolist(), ["normal"])
        self.assertEqual(preserved["effective_date"].tolist(), [pd.Timestamp("2006-01-11")])

    def test_marketwide_sse_status_targets_only_listed_sse_lifecycles(self):
        master = pd.DataFrame(
            [
                ["600001", "listing", "SSE", "2000-01-01"],
                ["600001", "delisting", "SSE", "2000-01-01"],
                ["000001", "listing", "SZSE", "1991-01-01"],
                ["600002", "listing", "SSE", "2027-01-01"],
            ],
            columns=["asset", "event_type", "exchange", "list_date"],
        )
        self.assertEqual(target_sse_assets(master, "2026-07-17"), ["600001"])

    def test_marketwide_sse_status_shards_are_disjoint_and_complete(self):
        assets = [f"60000{value}" for value in range(6)]
        shards = [select_sse_status_shard(assets, 3, index) for index in range(3)]
        self.assertEqual(sorted(asset for shard in shards for asset in shard), assets)
        self.assertFalse(set(shards[0]).intersection(shards[1]))

    def test_status_reconciler_adds_official_later_cycles_and_collapses_repeats(self):
        columns = [
            "asset",
            "effective_date",
            "execution_status",
            "is_st",
            "available_date",
            "source_coverage_mode",
            "data_source",
            "source_vintage",
            "evidence_priority",
            "evidence_role",
        ]
        base = pd.DataFrame(
            [
                ["600385", "2001-07-23", "normal", False, "2001-07-23", "dated_history", "base", "v1", 0, "base_governed_name_status"],
                ["600385", "2003-04-09", "risk_warning", True, "2003-04-09", "dated_history", "base", "v1", 0, "base_governed_name_status"],
            ],
            columns=columns,
        )
        official = pd.DataFrame(
            [
                ["600385", "2004-04-22", "risk_warning", True, "2004-04-21", "official", "official", "v2", 1, "official_sse_company_announcement"],
                ["600385", "2005-05-12", "normal", False, "2005-05-11", "official", "official", "v2", 1, "official_sse_company_announcement"],
                ["600385", "2007-02-15", "risk_warning", True, "2007-02-14", "official", "official", "v2", 1, "official_sse_company_announcement"],
            ],
            columns=columns,
        )
        for frame in (base, official):
            frame["effective_date"] = pd.to_datetime(frame["effective_date"])
            frame["available_date"] = pd.to_datetime(frame["available_date"])
        result, metrics = reconcile_asset_events(base, official, exchange="SSE")
        self.assertEqual(
            result["execution_status"].tolist(),
            ["normal", "risk_warning", "normal", "risk_warning"],
        )
        self.assertNotIn(pd.Timestamp("2004-04-22"), result["effective_date"].tolist())
        self.assertEqual(metrics["official_rows_added_after_collapse"], 2)

    def test_status_reconciler_prefers_official_evidence_on_same_date(self):
        columns = [
            "asset",
            "effective_date",
            "execution_status",
            "is_st",
            "available_date",
            "source_coverage_mode",
            "data_source",
            "source_vintage",
            "evidence_priority",
            "evidence_role",
        ]
        base = pd.DataFrame(
            [
                ["600001", "2000-01-01", "normal", False, "2000-01-01", "base", "base", "v1", 0, "base_governed_name_status"],
                ["600001", "2005-01-04", "risk_warning", True, "2005-01-04", "base", "base", "v1", 0, "base_governed_name_status"],
            ],
            columns=columns,
        )
        official = pd.DataFrame(
            [
                ["600001", "2005-01-04", "risk_warning", True, "2005-01-03", "official", "official", "v2", 1, "official_sse_company_announcement"],
            ],
            columns=columns,
        )
        for frame in (base, official):
            frame["effective_date"] = pd.to_datetime(frame["effective_date"])
            frame["available_date"] = pd.to_datetime(frame["available_date"])
        result, metrics = reconcile_asset_events(base, official, exchange="SSE")
        event = result[result["effective_date"].eq(pd.Timestamp("2005-01-04"))].iloc[0]
        self.assertEqual(event["evidence_role"], "official_sse_company_announcement")
        self.assertEqual(event["available_date"], pd.Timestamp("2005-01-03"))
        self.assertEqual(metrics["same_day_conflicts_resolved"], 0)

    def test_status_reconciler_uses_earlier_official_factbook_transition_over_secondary_name_page(self):
        columns = [
            "asset",
            "effective_date",
            "execution_status",
            "is_st",
            "available_date",
            "source_coverage_mode",
            "data_source",
            "source_vintage",
            "evidence_priority",
            "evidence_role",
        ]
        base = pd.DataFrame(
            [
                ["600603", "1992-01-13", "normal", False, "1992-01-13", "base", "base", "v1", 0, "base_governed_name_status"],
                ["600603", "2002-03-26", "risk_warning", True, "2002-03-26", "base", "base", "v1", 0, "base_governed_name_status"],
            ],
            columns=columns,
        )
        factbook = pd.DataFrame(
            [
                ["600603", "2002-03-22", "risk_warning", True, "2002-03-22", "factbook", "factbook", "v2", 2, "official_sse_factbook_status_reference"],
            ],
            columns=columns,
        )
        official = pd.DataFrame(columns=columns)
        for frame in (base, factbook):
            frame["effective_date"] = pd.to_datetime(frame["effective_date"])
            frame["available_date"] = pd.to_datetime(frame["available_date"])
        result, metrics = reconcile_asset_events(base, official, factbook, exchange="SSE")
        risk_dates = result.loc[result["execution_status"].eq("risk_warning"), "effective_date"].tolist()
        self.assertEqual(risk_dates, [pd.Timestamp("2002-03-22")])
        self.assertEqual(metrics["factbook_reference_rows_added_after_collapse"], 1)

    def test_factbook_terminal_lifecycle_rows_do_not_become_tradable_delisting_states(self):
        frame = pd.DataFrame(
            [
                ["600001", "delisting", False, False],
                ["600002", "listing_suspended", False, False],
                ["600003", "risk_warning", True, False],
                ["600004", "normal", True, True],
            ],
            columns=["asset", "execution_status", "binary_state_change", "used_in_reconciliation"],
        )
        selected = select_factbook_reference_candidates(frame)
        self.assertEqual(selected["asset"].tolist(), ["600002", "600003"])

    def test_sse_announcement_events_use_next_exchange_session(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(["2017-04-20", "2017-04-21", "2017-04-24", "2018-05-08", "2018-05-09"]))
        self.assertEqual(next_market_session("2017-04-20", calendar), pd.Timestamp("2017-04-21"))
        artifact = {
            "asset": "601106",
            "responses": [
                {
                    "rows": [
                        {
                            "SSEDATE": "2017-04-20",
                            "TITLE": "中国一重关于公司股票实施退市风险警示的公告",
                            "URL": "/start.pdf",
                        },
                        {
                            "SSEDATE": "2018-05-08",
                            "TITLE": "*ST一重关于公司股票撤销退市风险警示的公告",
                            "URL": "/end.pdf",
                        },
                    ]
                }
            ],
        }
        events, unresolved = parse_asset_events(artifact, calendar)
        self.assertFalse(unresolved)
        self.assertEqual(events["effective_date"].tolist(), [pd.Timestamp("2017-04-21"), pd.Timestamp("2018-05-09")])
        self.assertEqual(events["execution_status"].tolist(), ["risk_warning", "normal"])

    def test_sse_delisting_event_uses_official_first_notice_and_same_or_next_session(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"]))
        self.assertEqual(market_session_on_or_after("2026-06-06", calendar), pd.Timestamp("2026-06-08"))
        artifact = {
            "asset": "605081",
            "responses": [
                {
                    "rows": [
                        {
                            "SSEDATE": "2026-05-30",
                            "TITLE": "关于公司股票进入退市整理期交易的公告",
                            "URL": "/general.pdf",
                        },
                        {
                            "SSEDATE": "2026-06-06",
                            "TITLE": "关于公司股票进入退市整理期交易的第一次风险提示公告",
                            "URL": "/first.pdf",
                        },
                    ]
                }
            ],
        }
        events, unresolved = parse_asset_events(artifact, calendar)
        self.assertFalse(unresolved)
        self.assertEqual(events["execution_status"].tolist(), ["delisting"])
        self.assertEqual(events["effective_date"].tolist(), [pd.Timestamp("2026-06-08")])

    def test_monthly_valuation_uses_only_past_share_and_dividend_information(self):
        raw = self._baostock_rows(
            "600000",
            [
                {"date": "2020-01-02", "close": "10", "volume": "1000000", "turn": "1"},
                {"date": "2020-01-31", "close": "10", "volume": "1000000", "turn": "1"},
                {"date": "2020-02-28", "close": "20", "volume": "1000000", "turn": "1"},
            ],
        )
        history = normalise_baostock_history(raw, "600000", "2020-01-02", None, "2020-02-28")
        dividends = pd.DataFrame(
            [
                {
                    "asset": "600000",
                    "ex_date": "2020-01-15",
                    "available_date": "2020-01-10",
                    "cash_per_share": 0.5,
                },
                {
                    "asset": "600000",
                    "ex_date": "2020-03-15",
                    "available_date": "2020-03-10",
                    "cash_per_share": 9.0,
                },
            ]
        )
        result = build_monthly_valuation(history, dividends).set_index("date")
        self.assertAlmostEqual(float(result.loc[pd.Timestamp("2020-01-31"), "market_cap"]), 1_000_000_000.0)
        self.assertAlmostEqual(float(result.loc[pd.Timestamp("2020-01-31"), "dividend_yield"]), 0.05)
        self.assertAlmostEqual(float(result.loc[pd.Timestamp("2020-02-28"), "dividend_yield"]), 0.025)

    def test_market_history_validation_sample_is_deterministic_and_forced_first(self):
        assets = {"000001", "000002", "000004", "600000", "600001", "600002"}
        first = select_validation_assets(assets, 5)
        second = select_validation_assets(set(reversed(sorted(assets))), 5)
        self.assertEqual(first, second)
        self.assertEqual(first[:4], ["000001", "000002", "000004", "600000"])

    def test_market_history_validation_scopes_do_not_cross_block(self):
        checks = {
            name: True
            for name in set(TRADE_QUALIFICATION_CHECKS).union(VALUATION_QUALIFICATION_CHECKS)
        }
        checks["delisted_valuation_cross_source_population"] = False
        trade = _scope_checks(checks, TRADE_QUALIFICATION_CHECKS)
        valuation = _scope_checks(checks, VALUATION_QUALIFICATION_CHECKS)
        self.assertTrue(all(trade.values()))
        self.assertFalse(all(valuation.values()))

    def test_market_history_validator_reads_only_selected_assets_in_chunks(self):
        with self._workspace() as tmp:
            base = Path(tmp)
            output_path = base / "streamed.csv"
            summary_path = base / "summary.csv"
            pd.DataFrame(
                [
                    {"date": "2020-01-02", "asset": f"{asset:06d}", "value": asset}
                    for asset in range(8)
                ]
            ).to_csv(output_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "asset": f"{asset:06d}",
                        "status": "completed" if asset < 7 else "failed",
                        "trade_rows": 1,
                        "valuation_valid_rows": 1 if asset != 6 else 0,
                    }
                    for asset in range(8)
                ]
            ).to_csv(summary_path, index=False, encoding="utf-8-sig")
            subset = _read_asset_subset(output_path, {"000001", "000006"}, chunksize=2)
            eligible = _available_builder_assets(
                {
                    "asset_summary_path": self._relative(summary_path),
                    "asset_summary_sha256": _sha256(summary_path),
                }
            )
            with self.assertRaisesRegex(ValueError, "hash validation"):
                _available_builder_assets(
                    {
                        "asset_summary_path": self._relative(summary_path),
                        "asset_summary_sha256": "0" * 64,
                    }
                )
        self.assertEqual(subset["asset"].str.zfill(6).tolist(), ["000001", "000006"])
        self.assertIn("000001", eligible)
        self.assertNotIn("000006", eligible)
        self.assertNotIn("000007", eligible)

    def test_tushare_crosscheck_reports_missing_rows_and_price_mismatches(self):
        baostock = pd.DataFrame(
            [
                {"date": pd.Timestamp("2020-01-02"), "asset": "600000", "tradestatus": 1, "close": 10, "preclose": 9.9, "pctChg": 1.0},
                {"date": pd.Timestamp("2020-01-03"), "asset": "600000", "tradestatus": 1, "close": 11, "preclose": 10, "pctChg": 10.0},
            ]
        )
        tushare = pd.DataFrame(
            [
                {"date": pd.Timestamp("2020-01-02"), "asset": "600000", "ts_close": 10, "ts_pre_close": 9.9, "ts_pct_chg": 1.0},
                {"date": pd.Timestamp("2020-01-03"), "asset": "600000", "ts_close": 10.5, "ts_pre_close": 10, "ts_pct_chg": 5.0},
            ]
        )
        checks, metrics = compare_tushare_prices(baostock, tushare)
        self.assertEqual(metrics["matched_rows"], 2)
        self.assertEqual(metrics["close_mismatch_rows"], 1)
        self.assertEqual(metrics["close_match_ratio"], 0.5)
        self.assertEqual(metrics["maximum_close_absolute_error"], 0.5)
        self.assertEqual(metrics["large_close_mismatch_rows"], 1)
        self.assertEqual(metrics["large_relevant_close_mismatch_rows"], 1)
        self.assertEqual(len(checks), 2)

    def test_tushare_crosscheck_excludes_rows_before_formal_backtest_scope(self):
        baostock = pd.DataFrame(
            [
                {"date": pd.Timestamp("2000-04-21"), "asset": "000008", "tradestatus": 1, "close": 70.6, "preclose": 74.87, "pctChg": -5.7},
                {"date": pd.Timestamp("2005-01-04"), "asset": "000008", "tradestatus": 1, "close": 5.0, "preclose": 5.0, "pctChg": 0.0},
            ]
        )
        tushare = pd.DataFrame(
            [
                {"date": pd.Timestamp("2000-04-21"), "asset": "000008", "ts_close": 73.39, "ts_pre_close": 74.87, "ts_pct_chg": -1.98},
                {"date": pd.Timestamp("2005-01-04"), "asset": "000008", "ts_close": 5.0, "ts_pre_close": 5.0, "ts_pct_chg": 0.0},
            ]
        )
        checks, metrics = compare_tushare_prices(baostock, tushare)
        self.assertEqual(len(checks), 1)
        self.assertEqual(metrics["validation_start"], "2005-01-01")
        self.assertEqual(metrics["large_close_mismatch_rows"], 0)

    def test_tushare_crosscheck_hard_price_scope_is_limited_to_valuation_snapshots(self):
        baostock = pd.DataFrame(
            [
                {"date": pd.Timestamp("2013-06-21"), "asset": "000018", "tradestatus": 1, "close": 9.92, "preclose": 9.92, "pctChg": 0.0},
                {"date": pd.Timestamp("2013-06-28"), "asset": "000018", "tradestatus": 1, "close": 10.0, "preclose": 10.0, "pctChg": 0.0},
            ]
        )
        tushare = pd.DataFrame(
            [
                {"date": pd.Timestamp("2013-06-21"), "asset": "000018", "ts_close": 10.03, "ts_pre_close": 9.92, "ts_pct_chg": 1.11},
                {"date": pd.Timestamp("2013-06-28"), "asset": "000018", "ts_close": 10.0, "ts_pre_close": 10.0, "ts_pct_chg": 0.0},
            ]
        )
        keys = {(pd.Timestamp("2013-06-28"), "000018")}
        _, metrics = compare_tushare_prices(baostock, tushare, relevant_price_keys=keys)
        self.assertEqual(metrics["large_close_mismatch_rows"], 1)
        self.assertEqual(metrics["large_relevant_close_mismatch_rows"], 0)
        self.assertEqual(metrics["relevant_close_match_ratio"], 1.0)
        self.assertEqual(metrics["pre_close_match_ratio"], 1.0)

    def test_tushare_pre_close_is_aligned_on_corporate_action_dates(self):
        baostock = pd.DataFrame(
            [{"date": pd.Timestamp("2013-07-19"), "asset": "000010", "tradestatus": 1, "close": 7.0, "preclose": 5.968, "pctChg": 17.29}]
        )
        tushare = pd.DataFrame(
            [{"date": pd.Timestamp("2013-07-19"), "asset": "000010", "ts_close": 7.0, "ts_pre_close": 23.87, "ts_pct_chg": -70.67}]
        )
        factors = pd.DataFrame(
            [
                {"asset": "000010", "effective_date": "1999-05-31", "adj_factor": 3.3234872110855},
                {"asset": "000010", "effective_date": "2013-07-19", "adj_factor": 13.293948844342},
            ]
        )
        ratios = build_adjustment_ratios(factors)
        _, metrics = compare_tushare_prices(baostock, tushare, adjustment_ratios=ratios)
        self.assertAlmostEqual(metrics["maximum_raw_pre_close_absolute_error"], 17.902)
        self.assertLess(metrics["maximum_pre_close_absolute_error"], 0.001)
        self.assertEqual(metrics["corporate_action_ratio_rows"], 1)
        self.assertEqual(metrics["pre_close_factor_aligned_rows"], 1)
        self.assertEqual(metrics["pre_close_match_ratio"], 1.0)

    def test_tushare_pre_close_hard_gate_excludes_no_limit_or_unknown_execution_dates(self):
        baostock = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2005-11-11"),
                    "asset": "000065",
                    "tradestatus": 1,
                    "close": 5.21,
                    "preclose": 4.93,
                    "pctChg": 5.68,
                },
                {
                    "date": pd.Timestamp("2005-11-14"),
                    "asset": "000065",
                    "tradestatus": 1,
                    "close": 5.73,
                    "preclose": 5.21,
                    "pctChg": 9.98,
                },
            ]
        )
        tushare = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2005-11-11"),
                    "asset": "000065",
                    "ts_close": 5.21,
                    "ts_pre_close": 6.99,
                    "ts_pct_chg": -25.47,
                },
                {
                    "date": pd.Timestamp("2005-11-14"),
                    "asset": "000065",
                    "ts_close": 5.73,
                    "ts_pre_close": 5.21,
                    "ts_pct_chg": 9.98,
                },
            ]
        )
        relevant = {(pd.Timestamp("2005-11-14"), "000065")}
        checks, metrics = compare_tushare_prices(
            baostock,
            tushare,
            relevant_pre_close_keys=relevant,
        )
        self.assertEqual(metrics["pre_close_checks"], 1)
        self.assertEqual(metrics["pre_close_excluded_rows"], 1)
        self.assertEqual(metrics["excluded_pre_close_mismatch_rows"], 1)
        self.assertEqual(metrics["pre_close_match_ratio"], 1.0)
        self.assertEqual(metrics["maximum_pre_close_absolute_error"], 0.0)
        self.assertGreater(metrics["maximum_diagnostic_pre_close_absolute_error"], 2.0)
        self.assertFalse(bool(checks.iloc[0]["pre_close_gate_relevant"]))

    def test_joinquant_crosscheck_normalises_string_booleans_and_limits(self):
        trade = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-05-06"),
                    "asset": "600000",
                    "is_paused": "false",
                    "is_st": "0",
                    "has_price_limit": "true",
                    "limit_up": 11.0,
                    "limit_down": 9.0,
                    "execution_state_known": "1",
                    "limit_rule": "regular_main_or_pre_reform_growth_10",
                },
                {
                    "date": pd.Timestamp("2025-05-07"),
                    "asset": "600000",
                    "is_paused": "true",
                    "is_st": "false",
                    "has_price_limit": "false",
                    "limit_up": None,
                    "limit_down": None,
                    "execution_state_known": "true",
                    "limit_rule": "paused",
                },
            ]
        )
        jq = pd.DataFrame(
            [
                {"date": pd.Timestamp("2025-05-06"), "asset": "600000", "paused": 0, "is_st": False, "high_limit": 11.0, "low_limit": 9.0},
                {"date": pd.Timestamp("2025-05-07"), "asset": "600000", "paused": 1, "is_st": False, "high_limit": 11.0, "low_limit": 9.0},
            ]
        )
        _, metrics = compare_joinquant_state(trade, jq)
        self.assertEqual(metrics["paused_match_ratio"], 1.0)
        self.assertEqual(metrics["st_checks"], 2)
        self.assertEqual(metrics["st_match_ratio"], 1.0)
        self.assertEqual(metrics["limit_checks"], 1)
        self.assertEqual(metrics["limit_match_ratio"], 1.0)

    def test_joinquant_st_comparison_excludes_unknown_delisting_semantics(self):
        trade = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2025-06-06"),
                    "asset": "002750",
                    "is_paused": False,
                    "is_st": False,
                    "has_price_limit": False,
                    "limit_up": pd.NA,
                    "limit_down": pd.NA,
                    "execution_state_known": False,
                    "limit_rule": "delisting_limit_unknown",
                },
                {
                    "date": pd.Timestamp("2025-06-06"),
                    "asset": "600000",
                    "is_paused": False,
                    "is_st": False,
                    "has_price_limit": True,
                    "limit_up": 11.0,
                    "limit_down": 9.0,
                    "execution_state_known": True,
                    "limit_rule": "regular_main_or_pre_reform_growth_10",
                },
            ]
        )
        jq = pd.DataFrame(
            [
                {"date": pd.Timestamp("2025-06-06"), "asset": "002750", "paused": 0, "is_st": True, "high_limit": 10.5, "low_limit": 9.5},
                {"date": pd.Timestamp("2025-06-06"), "asset": "600000", "paused": 0, "is_st": False, "high_limit": 11.0, "low_limit": 9.0},
            ]
        )
        checks, metrics = compare_joinquant_state(trade, jq)
        self.assertEqual(metrics["state_checks"], 2)
        self.assertEqual(metrics["st_checks"], 1)
        self.assertEqual(metrics["st_match_ratio"], 1.0)
        self.assertFalse(bool(checks.loc[checks["asset"].eq("002750"), "st_checked"].iloc[0]))

    def test_eastmoney_crosscheck_uses_symmetric_relative_error(self):
        valuation = pd.DataFrame(
            [{"date": pd.Timestamp("2025-01-02"), "asset": "600000", "pe_ttm": -0.01, "pb": 1.0, "market_cap": 100.0}]
        )
        eastmoney = pd.DataFrame(
            [{"date": pd.Timestamp("2025-01-02"), "asset": "600000", "em_pe_ttm": 0.01, "em_pb": 1.0, "em_float_market_cap": 100.0}]
        )
        checks, metrics = compare_eastmoney_valuation(valuation, eastmoney)
        self.assertAlmostEqual(float(checks.loc[0, "pe_abs_relative_error"]), 0.02)
        self.assertEqual(metrics["pb_median_abs_relative_error"], 0.0)
        self.assertEqual(metrics["cap_p95_abs_relative_error"], 0.0)

    def test_eastmoney_validation_coverage_denominator_excludes_delisted_codes(self):
        lifecycles = pd.DataFrame(
            [
                {"asset": "000001", "delist_date": pd.NaT},
                {"asset": "000024", "delist_date": pd.Timestamp("2015-12-30")},
            ]
        )
        active, delisted = split_eastmoney_eligibility(["000001", "000024"], lifecycles)
        self.assertEqual(active, ["000001"])
        self.assertEqual(delisted, ["000024"])

    def test_eastmoney_validation_reuses_cached_delisted_evidence_without_live_fetch(self):
        lifecycles = pd.DataFrame(
            [
                {"asset": "000001", "delist_date": pd.NaT},
                {"asset": "000004", "delist_date": pd.Timestamp("2026-07-15")},
                {"asset": "000024", "delist_date": pd.Timestamp("2015-12-30")},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            (cache_root / "000004.csv.gz").touch()
            with patch(
                "strategy_lab.long_hold_v4.pit_stock_market_history_validator._em_cache_path",
                side_effect=lambda asset: cache_root / f"{asset}.csv.gz",
            ):
                active, cached, unavailable, collection = partition_eastmoney_validation_assets(
                    ["000001", "000004", "000024"], lifecycles
                )
        self.assertEqual(active, ["000001"])
        self.assertEqual(cached, ["000004"])
        self.assertEqual(unavailable, ["000024"])
        self.assertEqual(collection, ["000001", "000004"])

    def test_joinquant_valuation_crosscheck_converts_market_cap_units(self):
        valuation = pd.DataFrame(
            [{"date": pd.Timestamp("2025-06-30"), "asset": "000004", "pe_ttm": -8.8439, "pb": 22.3797, "market_cap": 1_117_650_000.0}]
        )
        joinquant = pd.DataFrame(
            [{"date": pd.Timestamp("2025-06-30"), "asset": "000004", "pe_ratio": -8.8439, "pb_ratio": 22.3797, "circulating_market_cap": 11.1765}]
        )
        checks, metrics = compare_joinquant_valuation(valuation, joinquant)
        self.assertEqual(len(checks), 1)
        self.assertEqual(metrics["pe_p95_abs_relative_error"], 0.0)
        self.assertEqual(metrics["pb_p95_abs_relative_error"], 0.0)
        self.assertEqual(metrics["cap_p95_abs_relative_error"], 0.0)

    def test_etf_master_is_event_sourced_and_filters_future_listings(self):
        source = pd.DataFrame(
            [
                {
                    "code": "510050.XSHG",
                    "display_name": "50ETF",
                    "start_date": "2005-02-23",
                    "end_date": "2200-01-01",
                    "type": "etf",
                },
                {
                    "code": "510090.XSHG",
                    "display_name": "责任ETF",
                    "start_date": "2010-08-09",
                    "end_date": "2015-08-26",
                    "type": "etf",
                },
                {
                    "code": "159999.XSHE",
                    "display_name": "未来ETF",
                    "start_date": "2026-07-22",
                    "end_date": "2200-01-01",
                    "type": "etf",
                },
            ]
        )
        result = build_etf_security_master(source, "fixture:v1", "2026-07-17")
        self.assertEqual(set(result["asset"]), {"510050", "510090"})
        self.assertEqual(result["list_status"].value_counts().to_dict(), {"listed": 2, "delisted": 1})
        self.assertTrue(result.loc[result["event_type"].eq("listing"), "delist_date"].isna().all())
        exit_row = result[result["event_type"].eq("delisting")].iloc[0]
        self.assertEqual(exit_row["available_date"], exit_row["delist_date"])

    def test_etf_master_rejects_duplicate_provider_codes(self):
        source = pd.DataFrame(
            [
                {
                    "code": "510050.XSHG",
                    "display_name": name,
                    "start_date": "2005-02-23",
                    "end_date": "2200-01-01",
                    "type": "etf",
                }
                for name in ("50ETF", "重复50ETF")
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate ETF codes"):
            build_etf_security_master(source, "fixture:v1", "2026-07-17")

    def test_etf_benchmark_probe_keeps_announcement_availability_and_filters_future_rows(self):
        source = pd.DataFrame(
            [
                {
                    "code": "510880.XSHG",
                    "pub_date": "2025-04-08",
                    "start_date": "2025-04-14",
                    "end_date": None,
                    "traced_index_name": "上证红利(000015)",
                    "traced_index_code": None,
                },
                {
                    "code": "159999.XSHE",
                    "pub_date": "2026-07-18",
                    "start_date": "2026-07-22",
                    "end_date": None,
                    "traced_index_name": "未来指数(999999)",
                    "traced_index_code": "999999.CSI",
                },
            ]
        )
        result = build_etf_benchmark_observation(source, "fixture:v1", "2026-07-17")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "asset"], "510880")
        self.assertEqual(result.loc[0, "index_code"], "000015")
        self.assertEqual(result.loc[0, "available_date"], result.loc[0, "announcement_date"])

    def test_dividend_builder_uses_final_notice_and_deduplicates_cash_flow(self):
        rows = [
            {
                "SECURITY_CODE": "600000",
                "SECURITY_NAME_ABBR": "浦发银行",
                "REPORT_DATE": report_date,
                "PLAN_NOTICE_DATE": "2020-01-10",
                "NOTICE_DATE": "2020-02-01",
                "EX_DIVIDEND_DATE": "2020-02-10",
                "PRETAX_BONUS_RMB": 10.0,
                "ASSIGN_PROGRESS": "实施分配",
            }
            for report_date in ("2019-12-31", "2020-03-31")
        ]
        rows.append(
            {
                **rows[0],
                "SECURITY_CODE": "000001",
                "SECURITY_NAME_ABBR": "平安银行",
                "NOTICE_DATE": "2020-03-01",
                "EX_DIVIDEND_DATE": "2020-03-10",
            }
        )
        result = build_stock_dividend_events(pd.DataFrame(rows), "fixture:v1", "2020-02-15")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "asset"], "600000")
        self.assertEqual(result.loc[0, "cash_per_share"], 1.0)
        self.assertEqual(result.loc[0, "available_date"], result.loc[0, "announcement_date"])
        self.assertTrue(pd.isna(result.loc[0, "pay_date"]))

    def test_dividend_builder_reassigns_predecessor_and_drops_prelisting_events(self):
        raw = pd.DataFrame(
            [
                {
                    "SECURITY_CODE": asset,
                    "SECURITY_NAME_ABBR": name,
                    "REPORT_DATE": report_date,
                    "PLAN_NOTICE_DATE": notice,
                    "NOTICE_DATE": notice,
                    "EX_DIVIDEND_DATE": ex_date,
                    "PRETAX_BONUS_RMB": 1.0,
                    "ASSIGN_PROGRESS": "实施分配",
                }
                for asset, name, report_date, notice, ex_date in (
                    ("302132", "中航成飞", "2012-12-31", "2013-06-13", "2013-06-21"),
                    ("302132", "中航成飞", "2024-12-31", "2025-06-10", "2025-06-17"),
                    ("600018", "上港集团", "2004-12-31", "2005-06-07", "2005-06-13"),
                )
            ]
        )
        master = pd.DataFrame(
            [
                {"asset": "300114", "asset_name": "中航电测", "event_type": "listing", "list_date": "2010-08-27", "delist_date": None},
                {"asset": "300114", "asset_name": "中航电测", "event_type": "delisting", "list_date": "2010-08-27", "delist_date": "2025-02-14"},
                {"asset": "302132", "asset_name": "中航成飞", "event_type": "listing", "list_date": "2025-02-17", "delist_date": None},
                {"asset": "600018", "asset_name": "上港集团", "event_type": "listing", "list_date": "2006-10-26", "delist_date": None},
            ]
        )
        lineage = {"code_migrations": [{"predecessor": "300114", "successor": "302132"}]}
        result = build_stock_dividend_events(raw, "fixture:v1", "2026-07-17", master, lineage)
        self.assertEqual(set(result["asset"]), {"300114", "302132"})
        predecessor = result[result["asset"].eq("300114")].iloc[0]
        self.assertEqual(predecessor["asset_name"], "中航电测")
        self.assertEqual(predecessor["security_code_resolution"], "reassigned_predecessor")
        self.assertEqual(result.attrs["lifecycle_reconciliation"]["dropped_outside_lifecycle"], 1)

    def test_dividend_report_periods_never_query_beyond_as_of(self):
        periods = report_periods(2025, "2026-07-17")
        self.assertEqual(periods[-1], "2026-06-30")
        self.assertNotIn("2026-09-30", periods)

    @staticmethod
    def _fundamental_sources(update_date: str = "2020-04-01") -> dict[str, pd.DataFrame]:
        return {
            "performance": pd.DataFrame(
                [
                    {
                        "SECURITY_CODE": "600000",
                        "SECURITY_NAME_ABBR": "浦发银行",
                        "REPORTDATE": "2019-12-31",
                        "NOTICE_DATE": "2020-03-01",
                        "UPDATE_DATE": update_date,
                        "TOTAL_OPERATE_INCOME": 100.0,
                        "PARENT_NETPROFIT": 20.0,
                        "WEIGHTAVG_ROE": 10.0,
                    }
                ]
            ),
            "balance": pd.DataFrame(
                [
                    {
                        "SECURITY_CODE": "600000",
                        "REPORT_DATE": "2019-12-31",
                        "NOTICE_DATE": "1900-01-01",
                        "TOTAL_ASSETS": 1000.0,
                        "TOTAL_LIABILITIES": 800.0,
                    }
                ]
            ),
            "cashflow": pd.DataFrame(
                [
                    {
                        "SECURITY_CODE": "600000",
                        "REPORT_DATE": "2019-12-31",
                        "NOTICE_DATE": "2020-03-20",
                        "NETCASH_OPERATE": 30.0,
                    }
                ]
            ),
        }

    def test_fundamental_builder_delays_final_values_to_latest_source_date(self):
        result = build_stock_fundamentals(self._fundamental_sources(), "fixture:v1", "2020-04-02")
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result.loc[0, "ann_date"].date()), "2020-03-01")
        self.assertEqual(str(result.loc[0, "available_date"].date()), "2020-04-01")
        self.assertEqual(result.loc[0, "operating_cash_flow"], 30.0)

        with self.assertRaisesRegex(ValueError, "no unique PIT records"):
            build_stock_fundamentals(
                self._fundamental_sources(update_date="2020-04-03"), "fixture:v2", "2020-04-02"
            )

    def test_fundamental_builder_maps_reports_across_tradable_code_lineage(self):
        sources = self._fundamental_sources(update_date="2011-03-01")
        sources["performance"].loc[:, ["REPORTDATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-15"]
        sources["balance"].loc[:, ["REPORT_DATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-20"]
        sources["cashflow"].loc[:, ["REPORT_DATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-25"]
        for frame in sources.values():
            frame["SECURITY_CODE"] = "601607"
        sources["performance"]["SECURITY_NAME_ABBR"] = "\u4e0a\u6d77\u533b\u836f"
        master = pd.DataFrame(
            [
                {"asset": "600849", "asset_name": "\u4e0a\u836f\u8f6c\u6362", "event_type": "listing", "list_date": "1994-03-24", "delist_date": None},
                {"asset": "600849", "asset_name": "\u4e0a\u836f\u8f6c\u6362", "event_type": "delisting", "list_date": "1994-03-24", "delist_date": "2010-03-08"},
                {"asset": "601607", "asset_name": "\u4e0a\u6d77\u533b\u836f", "event_type": "listing", "list_date": "2010-03-09", "delist_date": None},
            ]
        )
        lineage = {
            "code_migrations": [
                {
                    "predecessor": "600849",
                    "successor": "601607",
                    "predecessor_end_date": "2010-03-08",
                    "successor_start_date": "2010-03-09",
                }
            ]
        }
        result = build_stock_fundamentals(
            sources, "fixture:v1", "2012-01-01", master=master, lineage=lineage
        )
        self.assertEqual(result["asset"].tolist(), ["601607"])
        self.assertEqual(result.loc[0, "security_code_resolution"], "direct")

        sources = self._fundamental_sources(update_date="2010-03-01")
        sources["performance"].loc[:, ["REPORTDATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-15"]
        sources["balance"].loc[:, ["REPORT_DATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-20"]
        sources["cashflow"].loc[:, ["REPORT_DATE", "NOTICE_DATE"]] = ["2009-12-31", "2010-02-25"]
        for frame in sources.values():
            frame["SECURITY_CODE"] = "601607"
        sources["performance"]["SECURITY_NAME_ABBR"] = "\u4e0a\u6d77\u533b\u836f"
        result = build_stock_fundamentals(
            sources, "fixture:v2", "2012-01-01", master=master, lineage=lineage
        )
        self.assertEqual(set(result["asset"]), {"600849", "601607"})
        predecessor = result[result["asset"].eq("600849")].iloc[0]
        successor = result[result["asset"].eq("601607")].iloc[0]
        self.assertEqual(predecessor["security_code_resolution"], "reassigned_predecessor")
        self.assertEqual(successor["security_code_resolution"], "carried_forward_successor")
        self.assertEqual(str(successor["available_date"].date()), "2010-03-09")

    def test_status_validator_applies_last_effective_event(self):
        events = pd.DataFrame(
            [
                {
                    "asset": "600000",
                    "effective_date": "2020-01-01",
                    "is_st": False,
                    "execution_status": "normal",
                },
                {
                    "asset": "600000",
                    "effective_date": "2020-01-03",
                    "is_st": True,
                    "execution_status": "risk_warning",
                },
            ]
        )
        events["effective_date"] = pd.to_datetime(events["effective_date"])
        rows = pd.DataFrame(
            {
                "asset": ["600000"] * 4,
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
            }
        )
        result = apply_status_events(rows, build_event_index(events))
        self.assertEqual(result["expected_is_st"].astype(bool).tolist(), [False, False, True, True])
        self.assertEqual(result["expected_status"].tolist(), ["normal", "normal", "risk_warning", "risk_warning"])
        self.assertTrue(result["expected_state_known"].all())

    def test_status_validator_extracts_only_binary_transitions(self):
        events = pd.DataFrame(
            [
                ["600000", "2020-01-01", False, "normal", "dated_history"],
                ["600000", "2020-01-02", False, "delisting", "dated_history"],
                ["600000", "2020-01-03", True, "risk_warning", "dated_history"],
                ["600000", "2020-01-06", False, "normal", "dated_history"],
            ],
            columns=["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"],
        )
        events["effective_date"] = pd.to_datetime(events["effective_date"])
        result = expected_binary_transitions(events)
        self.assertEqual(result["effective_date"].dt.strftime("%Y-%m-%d").tolist(), ["2020-01-03", "2020-01-06"])

    def test_status_validator_does_not_treat_delisting_as_st_removal(self):
        events = pd.DataFrame(
            [
                ["600000", "2020-01-01", False, "normal", "dated_history"],
                ["600000", "2020-01-03", True, "risk_warning", "dated_history"],
                ["600000", "2020-01-06", False, "delisting", "dated_history"],
            ],
            columns=["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"],
        )
        events["effective_date"] = pd.to_datetime(events["effective_date"])
        result = expected_binary_transitions(events)
        self.assertEqual(result["effective_date"].dt.strftime("%Y-%m-%d").tolist(), ["2020-01-03"])

    def test_status_validator_does_not_compare_pt_rows_to_provider_is_st_field(self):
        checks = pd.DataFrame(
            {
                "actual_is_st": [False],
                "expected_is_st": [True],
                "expected_state_known": [True],
                "expected_status": ["special_transfer"],
            }
        )
        metrics = _daily_metrics(checks, "actual_is_st")
        self.assertEqual(metrics["daily_checks"], 0)

        events = pd.DataFrame(
            [
                ["600000", "2020-01-01", False, "normal", "dated_history"],
                ["600000", "2020-01-02", True, "special_transfer", "dated_history"],
                ["600000", "2020-01-03", True, "risk_warning", "dated_history"],
                ["600000", "2020-01-06", False, "normal", "dated_history"],
            ],
            columns=["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"],
        )
        events["effective_date"] = pd.to_datetime(events["effective_date"])
        transitions = expected_binary_transitions(events)
        self.assertEqual(
            transitions["effective_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2020-01-03", "2020-01-06"],
        )

    def test_status_validator_reports_non_overlapping_baostock_eras(self):
        daily = pd.DataFrame(
            {
                "asset": ["600001", "600001", "600001"],
                "date": pd.to_datetime(["2005-01-04", "2015-01-05", "2025-01-06"]),
                "actual_is_st": [False, True, False],
                "expected_is_st": [False, True, False],
                "expected_state_known": [True, True, True],
                "expected_status": ["normal", "risk_warning", "normal"],
            }
        )
        transitions = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(["2005-01-04", "2015-01-05", "2025-01-06"]),
                "exact_match": [True, True, True],
                "within_one_session": [True, True, True],
            }
        )
        result = build_baostock_era_metrics(daily, transitions)
        self.assertEqual(result["era"].tolist(), ["2000_2009", "2010_2019", "2020_present"])
        self.assertEqual(result["daily_checks"].tolist(), [1, 1, 1])
        self.assertEqual(result["transition_checks"].tolist(), [1, 1, 1])

    def test_status_validator_measures_market_session_distance(self):
        expected = pd.DataFrame(
            [
                {
                    "asset": "600000",
                    "effective_date": pd.Timestamp("2020-01-03"),
                    "is_st": True,
                    "source_coverage_mode": "dated_history",
                }
            ]
        )
        actual = pd.DataFrame(
            [
                {"asset": "600000", "actual_date": pd.Timestamp("2020-01-01"), "actual_is_st": False},
                {"asset": "600000", "actual_date": pd.Timestamp("2020-01-06"), "actual_is_st": True},
            ]
        )
        ranges = pd.DataFrame(
            [{"asset": "600000", "min_date": pd.Timestamp("2020-01-01"), "max_date": pd.Timestamp("2020-01-07")}]
        )
        calendar = pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]))
        result = compare_transition_dates(expected, actual, ranges, calendar)
        self.assertEqual(result.loc[0, "session_distance"], 1.0)
        self.assertFalse(result.loc[0, "exact_match"])
        self.assertTrue(result.loc[0, "within_one_session"])

    def test_status_validator_holdout_selection_excludes_non_sse_and_official_sources(self):
        events = pd.DataFrame(
            [
                ["600001", "2010-01-01", False, "normal", "dated_history"],
                ["600001", "2011-01-04", True, "risk_warning", "dated_history"],
                ["600002", "2010-01-01", False, "normal", "official_sse_announcement_status"],
                ["600002", "2012-01-04", True, "risk_warning", "official_sse_announcement_status"],
                ["000001", "2010-01-01", False, "normal", "official_dated_history"],
                ["000001", "2013-01-04", True, "risk_warning", "official_dated_history"],
            ],
            columns=["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"],
        )
        events["effective_date"] = pd.to_datetime(events["effective_date"])
        master = pd.DataFrame(
            [
                ["600001", "listing", "SSE"],
                ["600002", "listing", "SSE"],
                ["000001", "listing", "SZSE"],
            ],
            columns=["asset", "event_type", "exchange"],
        )
        result = select_sse_holdout_assets(events, master, 10)
        self.assertEqual(result["asset"].tolist(), ["600001"])

    def test_status_validator_compares_official_holdout_dates(self):
        expected = pd.DataFrame(
            [
                ["600001", "2020-01-03", True, "risk_warning", "dated_history"],
                ["600001", "2020-01-07", False, "normal", "dated_history"],
            ],
            columns=["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"],
        )
        expected["effective_date"] = pd.to_datetime(expected["effective_date"])
        official = pd.DataFrame(
            [
                ["600001", "2020-01-03", True, "risk_warning", "2020-01-02", "risk", "u1"],
                ["600001", "2020-01-08", False, "normal", "2020-01-07", "normal", "u2"],
            ],
            columns=[
                "asset",
                "effective_date",
                "is_st",
                "execution_status",
                "announcement_date",
                "announcement_title",
                "source_url",
            ],
        )
        calendar = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]))
        result = compare_sse_holdout_events(expected, official, {"600001"}, calendar)
        self.assertEqual(result["session_distance"].tolist(), [0.0, 1.0])
        self.assertEqual(result["exact_match"].tolist(), [True, False])
        self.assertTrue(result["within_one_session"].all())

    def test_factbook_holdout_excludes_build_evidence_and_is_deterministic(self):
        reference = pd.DataFrame(
            [
                ["600001", "2003-01-02", "risk_warning", False],
                ["600002", "2008-01-02", "normal", False],
                ["600003", "2011-01-04", "normal", True],
            ],
            columns=["asset", "effective_date", "execution_status", "used_in_reconciliation"],
        )
        reference["effective_date"] = pd.to_datetime(reference["effective_date"])
        independent = reference[~reference["used_in_reconciliation"]]
        first = select_factbook_holdout_assets(independent, 2)
        second = select_factbook_holdout_assets(independent, 2)
        self.assertEqual(first["asset"].tolist(), second["asset"].tolist())
        self.assertNotIn("600003", first["asset"].tolist())

    def test_factbook_holdout_matches_status_family_and_timing(self):
        reference = pd.DataFrame(
            [
                ["600001", "2003-01-03", "risk_warning", 2004, "特别处理", "u1"],
                ["600001", "2003-01-07", "normal", 2004, "取消特别处理", "u2"],
            ],
            columns=[
                "asset",
                "effective_date",
                "execution_status",
                "factbook_edition",
                "event_class",
                "source_url",
            ],
        )
        candidate = pd.DataFrame(
            [
                ["600001", "2003-01-03", "special_transfer"],
                ["600001", "2003-01-08", "normal"],
            ],
            columns=["asset", "effective_date", "execution_status"],
        )
        for frame in (reference, candidate):
            frame["effective_date"] = pd.to_datetime(frame["effective_date"])
        calendar = pd.DatetimeIndex(pd.to_datetime(["2003-01-02", "2003-01-03", "2003-01-06", "2003-01-07", "2003-01-08"]))
        result = compare_factbook_holdout_events(reference, candidate, {"600001"}, calendar)
        self.assertEqual(result["session_distance"].tolist(), [0.0, 1.0])
        self.assertTrue(result["within_one_session"].all())

    def test_factbook_holdout_excludes_state_preserving_star_st_to_st_rows(self):
        reference = pd.DataFrame(
            [
                ["600001", "2010-01-04", "risk_warning", False, 2011, "取消特别处理", "u1"],
                ["600002", "2010-01-05", "risk_warning", True, 2011, "实施退市风险警示", "u2"],
            ],
            columns=[
                "asset",
                "effective_date",
                "execution_status",
                "binary_state_change",
                "factbook_edition",
                "event_class",
                "source_url",
            ],
        )
        reference["effective_date"] = pd.to_datetime(reference["effective_date"])
        selected = select_factbook_holdout_assets(reference, 2)
        self.assertEqual(selected["asset"].tolist(), ["600002"])
        candidate = pd.DataFrame(
            [["600001", "2010-01-04", "risk_warning"], ["600002", "2010-01-05", "risk_warning"]],
            columns=["asset", "effective_date", "execution_status"],
        )
        candidate["effective_date"] = pd.to_datetime(candidate["effective_date"])
        calendar = pd.DatetimeIndex(pd.to_datetime(["2010-01-04", "2010-01-05"]))
        result = compare_factbook_holdout_events(reference, candidate, {"600001", "600002"}, calendar)
        self.assertEqual(result["asset"].tolist(), ["600002"])


if __name__ == "__main__":
    unittest.main()
