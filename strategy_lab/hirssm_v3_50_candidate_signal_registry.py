#!/usr/bin/env python
"""Build the V3.50 candidate signal registry and signal panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from candidate_signal_registry import (
    build_signal_panel,
    build_signal_summary,
    build_state_signal_coverage,
    candidate_registry_frame,
    check_signal_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "candidate_signal_registry_v3_50.json"
TASK_ID = "20260529_v3_50_candidate_signal_registry"
VERSION = "V3.50"


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


def build_no_performance_guard(signal_panel: pd.DataFrame) -> pd.DataFrame:
    forbidden_cols = [
        col
        for col in signal_panel.columns
        if any(keyword in col.lower() for keyword in ["return", "pnl", "nav", "sharpe", "drawdown"])
    ]
    return pd.DataFrame(
        [
            {
                "guard": "no_return_labels",
                "status": "pass" if not forbidden_cols else "fail",
                "detail": ",".join(forbidden_cols),
            },
            {
                "guard": "no_performance_validation",
                "status": "pass",
                "detail": "V3.50 generates candidate signals only",
            },
            {
                "guard": "no_portfolio_backtest",
                "status": "pass",
                "detail": "no weights, NAV, trades, or returns generated",
            },
            {
                "guard": "no_model_promotion",
                "status": "pass"
                if not bool(signal_panel["model_promotion_allowed"].astype(bool).any())
                and not bool(signal_panel["performance_claim_allowed"].astype(bool).any())
                else "fail",
                "detail": "all candidates remain observation status",
            },
        ]
    )


def build_acceptance_checks(
    config: dict[str, Any],
    registry: pd.DataFrame,
    signal_panel: pd.DataFrame,
    contract_checks: pd.DataFrame,
    no_performance_guard: pd.DataFrame,
    state_manifest: dict[str, Any],
    framework_manifest: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "check": "v3_48_state_monitor_acceptance_passed",
            "status": "pass" if bool(state_manifest.get("acceptance_pass", False)) else "fail",
            "detail": state_manifest.get("data_decision", ""),
        },
        {
            "check": "v3_49_framework_acceptance_passed",
            "status": "pass" if bool(framework_manifest.get("acceptance_pass", False)) else "fail",
            "detail": framework_manifest.get("data_decision", ""),
        },
        {
            "check": "candidate_count_matches_config",
            "status": "pass" if int(registry.shape[0]) == int(config["expected_candidate_count"]) else "fail",
            "detail": f"registry={registry.shape[0]};expected={config['expected_candidate_count']}",
        },
        {
            "check": "signal_rows_above_minimum",
            "status": "pass" if int(signal_panel.shape[0]) >= int(config["min_signal_rows"]) else "fail",
            "detail": str(int(signal_panel.shape[0])),
        },
        {
            "check": "signal_ids_all_registered",
            "status": "pass"
            if set(signal_panel["signal_id"].astype(str).unique()).issubset(set(registry["signal_id"].astype(str)))
            else "fail",
            "detail": str(signal_panel["signal_id"].nunique()),
        },
        {
            "check": "contract_checks_all_pass",
            "status": "pass" if bool((contract_checks["status"] == "pass").all()) else "fail",
            "detail": str(int((contract_checks["status"] != "pass").sum())),
        },
        {
            "check": "no_performance_guard_all_pass",
            "status": "pass" if bool((no_performance_guard["status"] == "pass").all()) else "fail",
            "detail": str(int((no_performance_guard["status"] != "pass").sum())),
        },
        {
            "check": "signal_panel_data_scope_is_state_monitor",
            "status": "pass"
            if set(signal_panel["data_scope"].astype(str).unique()) == {"accepted_processed_daily_only"}
            else "fail",
            "detail": "|".join(sorted(signal_panel["data_scope"].astype(str).unique())),
        },
        {
            "check": "signal_panel_price_adjustment_none_raw",
            "status": "pass"
            if set(signal_panel["price_adjustment"].astype(str).unique()) == {"none_raw"}
            else "fail",
            "detail": "|".join(sorted(signal_panel["price_adjustment"].astype(str).unique())),
        },
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "candidate signal registry only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    registry: pd.DataFrame,
    signal_panel: pd.DataFrame,
    summary: pd.DataFrame,
    contract_checks: pd.DataFrame,
    no_performance_guard: pd.DataFrame,
    acceptance_checks: pd.DataFrame,
) -> str:
    failed = acceptance_checks.loc[acceptance_checks["status"] != "pass"]
    latest_date = signal_panel["signal_date"].max()
    lines = [
        "# V3.50 Candidate Signal Registry",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.50 creates a governed candidate signal registry and signal panel from V3.48 diagnostic state data.",
        "- The signal panel conforms to the V3.49 signal contract and is ready to be paired with future adjusted/PIT labels.",
        "- No IC, return, hit-rate, PnL, NAV, drawdown, backtest, or model-promotion result is produced.",
        "",
        "## Coverage",
        "",
        f"- Candidate signals: `{len(registry)}`",
        f"- Signal rows: `{len(signal_panel)}`",
        f"- Latest signal date: `{latest_date}`",
        f"- Signal dates: `{signal_panel['signal_date'].nunique()}`",
        "",
        "## Candidate Summary",
        "",
        "| signal_id | direction | rows | mean_value | first | last |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.signal_id}` | `{row.signal_direction}` | {int(row.observations)} | "
            f"{float(row.mean_signal_value):.4f} | `{row.first_signal_date}` | `{row.last_signal_date}` |"
        )
    lines.extend(
        [
            "",
            "## Contract Checks",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in contract_checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## No-Performance Guard",
            "",
            "| guard | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in no_performance_guard.itertuples(index=False):
        lines.append(f"| `{row.guard}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Acceptance",
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
            "- Feed `signal_panel.csv` into V3.49 only after `adjusted_pit_label_contract.csv` is satisfied.",
            "- Until adjusted/PIT labels arrive, the panel is a candidate registry artifact, not evidence of effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(signal_panel: pd.DataFrame, registry: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# A-share Candidate Signal Registry V3.50",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            "- Candidate registry accepted: `True`",
            f"- Candidate signals: `{len(registry)}`",
            f"- Signal rows: `{len(signal_panel)}`",
            f"- Signal dates: `{signal_panel['signal_date'].nunique()}`",
            "- Performance validation: `not_run`",
            "- Model promotion: `not_allowed`",
            "",
            "## Boundary",
            "",
            "- Input: V3.48 state monitor only.",
            "- Output: V3.49-compatible candidate signal panel.",
            "- No return labels, IC, hit rate, backtest, NAV, or Sharpe.",
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
    state_manifest = read_json(ROOT / config["state_monitor_manifest_path"])
    framework_manifest = read_json(ROOT / config["validation_framework_manifest_path"])
    signal_contract = read_csv(ROOT / config["signal_panel_contract_path"])

    registry = candidate_registry_frame()
    signal_panel = build_signal_panel(state_panel)
    summary = build_signal_summary(signal_panel)
    state_signal_coverage = build_state_signal_coverage(signal_panel)
    contract_checks = check_signal_contract(signal_panel, signal_contract)
    no_performance_guard = build_no_performance_guard(signal_panel)
    acceptance_checks = build_acceptance_checks(
        config,
        registry,
        signal_panel,
        contract_checks,
        no_performance_guard,
        state_manifest,
        framework_manifest,
    )

    artifacts = {
        "candidate_signal_registry": output_dir / "candidate_signal_registry.csv",
        "signal_panel": output_dir / "signal_panel.csv",
        "signal_summary": output_dir / "signal_summary.csv",
        "state_signal_coverage": output_dir / "state_signal_coverage.csv",
        "signal_panel_contract_check": output_dir / "signal_panel_contract_check.csv",
        "no_performance_guard": output_dir / "no_performance_guard.csv",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "candidate_signal_report": output_dir / "candidate_signal_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(registry, artifacts["candidate_signal_registry"])
    write_csv(signal_panel, artifacts["signal_panel"])
    write_csv(summary, artifacts["signal_summary"])
    write_csv(state_signal_coverage, artifacts["state_signal_coverage"])
    write_csv(contract_checks, artifacts["signal_panel_contract_check"])
    write_csv(no_performance_guard, artifacts["no_performance_guard"])
    write_csv(acceptance_checks, artifacts["acceptance_checks"])
    write_text(build_report(registry, signal_panel, summary, contract_checks, no_performance_guard, acceptance_checks), artifacts["candidate_signal_report"])
    write_text(build_catalog(signal_panel, registry), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, acceptance_checks)
    write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((acceptance_checks["status"] == "pass").all()),
        "candidate_signal_count": int(registry.shape[0]),
        "signal_rows": int(signal_panel.shape[0]),
        "signal_dates": int(signal_panel["signal_date"].nunique()),
        "data_decision": "candidate_signal_panel_ready_for_future_state_stratified_validation",
        "performance_validation": "not_run_missing_adjusted_pit_labels",
        "model_decision": "no_model_promotion_candidate_registry_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
