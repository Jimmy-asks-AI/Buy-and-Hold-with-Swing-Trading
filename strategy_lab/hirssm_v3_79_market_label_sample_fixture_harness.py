#!/usr/bin/env python
"""Run HIRSSM V3.79 MARKET label sample fixture harness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_sample_fixture_harness import (
    FixtureHarnessConfig,
    build_catalog,
    build_expected_vs_actual,
    build_harness_acceptance,
    build_regression_summary,
    build_report,
    build_sample_config,
    write_fixtures,
)
from market_label_sample_intake_validator import (
    build_action_queue,
    build_no_execution_guard,
    discover_sample_files,
    validate_samples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_sample_fixture_harness_v3_79.json"
TASK_ID = "20260601_v3_79_market_label_sample_fixture_harness"
VERSION = "V3.79"
AGENT = "code_quality_engineer"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> FixtureHarnessConfig:
    return FixtureHarnessConfig(
        v3_78_config_path=resolve_path(raw["v3_78_config_path"]),
        v3_78_manifest_path=resolve_path(raw["v3_78_manifest_path"]),
        fixture_sample_dir=resolve_path(raw["fixture_sample_dir"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        expected_cases=tuple(dict(item) for item in raw["expected_cases"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_sample_fixture_harness.py",
        ROOT / "strategy_lab" / "hirssm_v3_79_market_label_sample_fixture_harness.py",
        ROOT / "configs" / "market_label_sample_fixture_harness_v3_79.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_79_market_label_sample_fixture_harness.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir
    base_raw = read_json(config.v3_78_config_path)
    v3_78_manifest = read_json(config.v3_78_manifest_path)

    fixture_manifest = write_fixtures(config)
    sample_config = build_sample_config(base_raw, config, ROOT)
    inventory = discover_sample_files(sample_config)
    validation_checks, sample_decisions = validate_samples(sample_config)
    action_queue = build_action_queue(sample_decisions, sample_config)
    guard = build_no_execution_guard()
    expected_vs_actual = build_expected_vs_actual(validation_checks=validation_checks, sample_decisions=sample_decisions, expected_cases=config.expected_cases)
    summary = build_regression_summary(expected_vs_actual, sample_decisions)
    acceptance = build_harness_acceptance(fixture_manifest, expected_vs_actual, sample_decisions, guard, config)
    report = build_report(fixture_manifest, sample_decisions, expected_vs_actual, summary, acceptance, config)
    catalog = build_catalog(summary, config)

    output_paths = {
        "fixture_manifest": output_dir / "fixture_manifest.csv",
        "inventory": output_dir / "fixture_sample_inventory.csv",
        "validation": output_dir / "fixture_validation_checks.csv",
        "decisions": output_dir / "fixture_candidate_file_decision.csv",
        "action_queue": output_dir / "fixture_action_queue.csv",
        "expected_vs_actual": output_dir / "fixture_expected_vs_actual.csv",
        "summary": output_dir / "fixture_regression_summary.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_sample_fixture_harness_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(fixture_manifest, output_paths["fixture_manifest"])
    write_csv(inventory, output_paths["inventory"])
    write_csv(validation_checks, output_paths["validation"])
    write_csv(sample_decisions, output_paths["decisions"])
    write_csv(action_queue, output_paths["action_queue"])
    write_csv(expected_vs_actual, output_paths["expected_vs_actual"])
    write_csv(summary, output_paths["summary"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    forbidden_terms = {
        "nav",
        "sharpe",
        "annualized_return",
        "portfolio_return",
        "max_drawdown",
        "official_total_return_label",
        "default_enabled",
    }
    forbidden_columns = sorted(
        term
        for term in forbidden_terms
        if term in " ".join(output_columns([fixture_manifest, inventory, validation_checks, sample_decisions, action_queue, expected_vs_actual, summary, guard, acceptance])).lower()
    )
    target_was_written = config.target_source_path.exists() and not bool(v3_78_manifest.get("metrics", {}).get("target_source_exists", False))
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass_or_warn_only",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_78_manifest_passed",
                "status": "pass" if bool(v3_78_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_78_manifest.get('self_check_pass')}",
            },
            {
                "check": "all_expected_regressions_pass",
                "status": "pass" if expected_vs_actual["regression_pass"].astype(bool).all() else "fail",
                "detail": ",".join(expected_vs_actual.loc[~expected_vs_actual["regression_pass"].astype(bool), "case_id"].astype(str)),
            },
            {
                "check": "target_csv_not_written_by_v3_79",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={config.target_source_path.exists()}",
            },
            {
                "check": "v3_53_not_executable",
                "status": "pass" if not action_queue["may_execute_now"].astype(bool).any() else "fail",
                "detail": "fixture validation only",
            },
            {
                "check": "forbidden_performance_columns_absent",
                "status": "pass" if not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["fixture_manifest"],
        output_paths["inventory"],
        output_paths["validation"],
        output_paths["decisions"],
        output_paths["action_queue"],
        output_paths["expected_vs_actual"],
        output_paths["summary"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    fixture_paths = sorted(path for path in config.fixture_sample_dir.glob("*.csv") if path.is_file())
    write_text(build_changed_files(outputs_for_changed + fixture_paths), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    regression_pass_cases = int(expected_vs_actual["regression_pass"].astype(bool).sum()) if not expected_vs_actual.empty else 0
    passing_samples = int(sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if not sample_decisions.empty else 0
    rejected_samples = int(sample_decisions["decision"].astype(str).eq("rejected_or_needs_repair").sum()) if not sample_decisions.empty else 0
    warning_count = int(sample_decisions["warning_count"].sum()) if "warning_count" in sample_decisions.columns else 0
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "fixture_case_count": int(len(fixture_manifest)),
        "expected_case_count": int(len(expected_vs_actual)),
        "regression_pass_cases": regression_pass_cases,
        "passing_samples": passing_samples,
        "rejected_samples": rejected_samples,
        "warning_count": warning_count,
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]] + fixture_paths
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.78 sample intake validator",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_79_market_label_sample_fixture_harness.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_sample_fixture_harness.py",
            "strategy_lab/hirssm_v3_79_market_label_sample_fixture_harness.py",
            "strategy_lab/market_label_sample_intake_validator.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in all_outputs],
        "changed_files": build_changed_files(all_outputs).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warning_count,
        "limitations": [
            "Fixtures are synthetic and only validate the intake validator logic.",
            "A fixture pass is not evidence that any real provider file is licensed or correct.",
            "No label generation, portfolio validation, or model promotion is allowed.",
        ],
        "risk_flags": [
            "synthetic_fixture_only",
            "external_license_still_required",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Use the fixture harness as a regression guard; then add controlled real-sample review once provider samples arrive.",
        "handoff_summary": "V3.79 generated synthetic sample fixtures and proved the V3.78 validator catches the declared pass/fail/warn cases.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
