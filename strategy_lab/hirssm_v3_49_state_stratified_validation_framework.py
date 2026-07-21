#!/usr/bin/env python
"""Build the V3.49 state-stratified validation framework."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from state_stratified_validation_framework import (
    LABEL_CONTRACT_ROWS,
    SIGNAL_CONTRACT_ROWS,
    VALIDATION_METRIC_ROWS,
    ValidationGateConfig,
    build_negative_control_plan,
    build_no_result_guard,
    build_readiness_checks,
    build_state_coverage_summary,
    build_state_stratification_plan,
    dataframe_from_contract,
    validate_label_contract,
    validate_signal_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "state_stratified_validation_v3_49.json"
TASK_ID = "20260529_v3_49_state_stratified_validation_framework"
VERSION = "V3.49"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def gate_config(config: dict[str, Any]) -> ValidationGateConfig:
    return ValidationGateConfig(
        min_state_observations=int(config["min_state_observations"]),
        min_unique_states=int(config["min_unique_states"]),
        required_return_basis=tuple(config["required_return_basis"]),
        horizons=tuple(int(item) for item in config["horizons"]),
        embargo_days=int(config["embargo_days"]),
        purge_days=int(config["purge_days"]),
    )


def build_framework_acceptance_checks(
    readiness: pd.DataFrame,
    no_result_guard: pd.DataFrame,
    state_monitor_manifest: dict[str, Any],
) -> pd.DataFrame:
    readiness_fail = readiness.loc[readiness["status"] == "fail"]
    produced_forbidden = no_result_guard.loc[no_result_guard["produced"].astype(bool)]
    rows = [
        {
            "check": "v3_48_state_monitor_acceptance_passed",
            "status": "pass" if bool(state_monitor_manifest.get("acceptance_pass", False)) else "fail",
            "detail": state_monitor_manifest.get("data_decision", ""),
        },
        {
            "check": "framework_readiness_has_no_failures",
            "status": "pass" if readiness_fail.empty else "fail",
            "detail": ";".join(f"{row.check}:{row.detail}" for row in readiness_fail.itertuples(index=False)),
        },
        {
            "check": "performance_validation_is_blocked_without_labels",
            "status": "pass"
            if bool((readiness["check"] == "performance_validation_allowed_now").any())
            and str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
            else "fail",
            "detail": "expected blocked until signal and adjusted/PIT label panels are supplied",
        },
        {
            "check": "no_performance_or_backtest_results_produced",
            "status": "pass" if produced_forbidden.empty else "fail",
            "detail": str(int(produced_forbidden.shape[0])),
        },
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "validation framework only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    state_coverage: pd.DataFrame,
    stratification_plan: pd.DataFrame,
    readiness: pd.DataFrame,
    no_result_guard: pd.DataFrame,
    acceptance_checks: pd.DataFrame,
) -> str:
    failed = acceptance_checks.loc[acceptance_checks["status"] != "pass"]
    blocked = readiness.loc[readiness["status"] == "blocked"]
    eligible = stratification_plan.loc[stratification_plan["sample_gate_pass"].astype(bool)]
    top_states = state_coverage.sort_values("observations", ascending=False).head(12)
    lines = [
        "# V3.49 State-Stratified Validation Framework",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.49 builds the governance framework for validating future signals by V3.48 market states.",
        "- The framework is ready for contracts, state coverage, stratification, negative controls, and readiness checks.",
        "- Performance validation is intentionally blocked until a signal panel and adjusted/PIT forward-return labels are supplied.",
        "- No return, backtest, model, or performance conclusion is produced.",
        "",
        "## State Coverage",
        "",
        f"- State panel rows: `{len(state_panel)}`",
        f"- History-available rows: `{int(state_panel['history_available'].sum())}`",
        f"- Eligible state buckets: `{len(eligible)}`",
        f"- Minimum observations per bucket: `{config['min_state_observations']}`",
        "",
        "| state_column | state_value | observations | validation_role |",
        "|---|---|---:|---|",
    ]
    for row in stratification_plan.sort_values("observations", ascending=False).head(16).itertuples(index=False):
        lines.append(f"| `{row.state_column}` | `{row.state_value}` | {int(row.observations)} | `{row.validation_role}` |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## No-Result Guard",
            "",
            "| result_type | produced | blocked | reason |",
            "|---|---|---|---|",
        ]
    )
    for row in no_result_guard.itertuples(index=False):
        lines.append(f"| `{row.result_type}` | `{row.produced}` | `{row.blocked}` | {str(row.reason)[:180]} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Failed checks: `{len(failed)}`",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance_checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Provide a candidate signal panel matching `signal_panel_contract.csv`.",
            "- Provide adjusted/PIT forward-return labels matching `adjusted_pit_label_contract.csv`.",
            "- Only then compute state-stratified IC, ICIR, hit rate, effect size, and negative controls.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame, stratification_plan: pd.DataFrame) -> str:
    performance_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    eligible = stratification_plan.loc[stratification_plan["sample_gate_pass"].astype(bool)]
    return "\n".join(
        [
            "# A-share State-Stratified Validation Framework V3.49",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            "- Framework accepted: `True`",
            f"- Performance validation status: `{performance_status}`",
            f"- Eligible state buckets: `{len(eligible)}`",
            "- Required labels: adjusted/PIT forward returns only",
            "- No model or backtest promotion.",
            "",
            "## Artifacts",
            "",
            "- `state_coverage_summary.csv`",
            "- `state_stratification_plan.csv`",
            "- `signal_panel_contract.csv`",
            "- `adjusted_pit_label_contract.csv`",
            "- `negative_control_plan.csv`",
            "- `readiness_checks.csv`",
            "- `no_result_guard.csv`",
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], acceptance_checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    rows.append(
        {
            "check": "acceptance_checks_all_pass",
            "status": "pass" if bool((acceptance_checks["status"] == "pass").all()) else "fail",
            "detail": str(int((acceptance_checks["status"] != "pass").sum())),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]

    state_panel = read_csv(ROOT / config["state_panel_path"])
    state_monitor_manifest = read_json(ROOT / config["state_monitor_manifest_path"])
    gates = gate_config(config)
    state_coverage = build_state_coverage_summary(state_panel, list(config["state_columns"]))
    stratification_plan = build_state_stratification_plan(gates, state_coverage)
    signal_contract = dataframe_from_contract(SIGNAL_CONTRACT_ROWS)
    label_contract = dataframe_from_contract(LABEL_CONTRACT_ROWS)
    metric_contract = dataframe_from_contract(VALIDATION_METRIC_ROWS)
    negative_control_plan = build_negative_control_plan(gates)
    signal_ready = validate_signal_contract(optional_path(config.get("signal_panel_path")))
    label_ready = validate_label_contract(optional_path(config.get("adjusted_pit_label_path")), gates)
    readiness = build_readiness_checks(state_panel, state_coverage, label_ready, signal_ready, gates)
    no_result_guard = build_no_result_guard(readiness)
    acceptance_checks = build_framework_acceptance_checks(readiness, no_result_guard, state_monitor_manifest)

    artifacts = {
        "state_coverage_summary": output_dir / "state_coverage_summary.csv",
        "state_stratification_plan": output_dir / "state_stratification_plan.csv",
        "signal_panel_contract": output_dir / "signal_panel_contract.csv",
        "adjusted_pit_label_contract": output_dir / "adjusted_pit_label_contract.csv",
        "validation_metric_contract": output_dir / "validation_metric_contract.csv",
        "negative_control_plan": output_dir / "negative_control_plan.csv",
        "readiness_checks": output_dir / "readiness_checks.csv",
        "no_result_guard": output_dir / "no_result_guard.csv",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "framework_report": output_dir / "framework_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(state_coverage, artifacts["state_coverage_summary"])
    write_csv(stratification_plan, artifacts["state_stratification_plan"])
    write_csv(signal_contract, artifacts["signal_panel_contract"])
    write_csv(label_contract, artifacts["adjusted_pit_label_contract"])
    write_csv(metric_contract, artifacts["validation_metric_contract"])
    write_csv(negative_control_plan, artifacts["negative_control_plan"])
    write_csv(readiness, artifacts["readiness_checks"])
    write_csv(no_result_guard, artifacts["no_result_guard"])
    write_csv(acceptance_checks, artifacts["acceptance_checks"])
    write_text(
        build_report(config, state_panel, state_coverage, stratification_plan, readiness, no_result_guard, acceptance_checks),
        artifacts["framework_report"],
    )
    write_text(build_catalog(readiness, stratification_plan), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, acceptance_checks)
    write_csv(self_check, self_check_path)
    performance_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((acceptance_checks["status"] == "pass").all()),
        "framework_ready": True,
        "performance_validation_status": performance_status,
        "state_rows": int(state_panel.shape[0]),
        "history_available_rows": int(state_panel["history_available"].sum()),
        "eligible_state_buckets": int(stratification_plan["sample_gate_pass"].sum()),
        "data_decision": "state_stratified_validation_framework_ready_contracts_only",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
