#!/usr/bin/env python
"""Run HIRSSM V3.81 raw registry fixture harness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_raw_registry_fixture_harness import (
    RawRegistryFixtureHarnessConfig,
    build_catalog,
    build_duplicate_hash_report,
    build_expected_vs_actual,
    build_first_seen_stability_report,
    build_harness_acceptance,
    build_license_gate_report,
    build_registry_config,
    build_regression_summary,
    build_report,
    build_source_gate_report,
    build_tamper_report,
    write_fixtures,
)
from market_label_raw_sample_registry import (
    build_controlled_handoff,
    build_license_review_queue,
    build_no_execution_guard,
    build_raw_sample_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_raw_registry_fixture_harness_v3_81.json"
TASK_ID = "20260601_v3_81_market_label_raw_registry_fixture_harness"
VERSION = "V3.81"
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


def load_config(raw: dict[str, Any]) -> RawRegistryFixtureHarnessConfig:
    return RawRegistryFixtureHarnessConfig(
        v3_80_config_path=resolve_path(raw["v3_80_config_path"]),
        v3_80_manifest_path=resolve_path(raw["v3_80_manifest_path"]),
        fixture_sample_dir=resolve_path(raw["fixture_sample_dir"]),
        license_status_path=resolve_path(raw["license_status_path"]),
        license_evidence_dir=resolve_path(raw["license_evidence_dir"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        expected_cases=tuple(dict(item) for item in raw["expected_cases"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_raw_sample_registry.py",
        ROOT / "strategy_lab" / "hirssm_v3_80_market_label_raw_sample_registry.py",
        ROOT / "strategy_lab" / "market_label_raw_registry_fixture_harness.py",
        ROOT / "strategy_lab" / "hirssm_v3_81_market_label_raw_registry_fixture_harness.py",
        ROOT / "configs" / "market_label_raw_sample_registry_v3_80.json",
        ROOT / "configs" / "market_label_raw_registry_fixture_harness_v3_81.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_81_market_label_raw_registry_fixture_harness.json",
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
    base_raw = read_json(config.v3_80_config_path)
    v3_80_manifest = read_json(config.v3_80_manifest_path)

    fixture_manifest, license_status = write_fixtures(config)
    first_config = build_registry_config(base_raw, config, ROOT, output_dir / "nonexistent_previous_registry.csv")
    first_registry = build_raw_sample_registry(first_config)
    first_registry_path = output_dir / "first_raw_sample_registry.csv"
    write_csv(first_registry, first_registry_path)
    second_config = build_registry_config(base_raw, config, ROOT, first_registry_path)
    second_registry = build_raw_sample_registry(second_config)
    license_queue = build_license_review_queue(second_registry)
    handoff = build_controlled_handoff(second_registry, second_config)
    guard = build_no_execution_guard()

    duplicate_report = build_duplicate_hash_report(second_registry)
    tamper_report = build_tamper_report(second_registry)
    first_seen_report = build_first_seen_stability_report(first_registry, second_registry)
    source_gate_report = build_source_gate_report(second_registry)
    license_gate_report = build_license_gate_report(second_registry)
    expected_vs_actual = build_expected_vs_actual(
        duplicate_report,
        tamper_report,
        first_seen_report,
        source_gate_report,
        license_gate_report,
        config.expected_cases,
    )
    summary = build_regression_summary(expected_vs_actual, second_registry)
    acceptance = build_harness_acceptance(expected_vs_actual, second_registry, guard, second_config.target_source_path)
    report = build_report(fixture_manifest, second_registry, expected_vs_actual, summary, acceptance, config)
    catalog = build_catalog(summary, config)

    output_paths = {
        "fixture_manifest": output_dir / "fixture_manifest.csv",
        "license_status": output_dir / "fixture_license_status.csv",
        "first_registry": output_dir / "first_raw_sample_registry.csv",
        "second_registry": output_dir / "second_raw_sample_registry.csv",
        "license_queue": output_dir / "license_review_queue.csv",
        "handoff": output_dir / "controlled_review_handoff.csv",
        "duplicate_report": output_dir / "duplicate_hash_report.csv",
        "tamper_report": output_dir / "tamper_detection_report.csv",
        "first_seen": output_dir / "first_seen_stability_report.csv",
        "source_gate": output_dir / "source_token_gate_report.csv",
        "license_gate": output_dir / "license_gate_report.csv",
        "expected_vs_actual": output_dir / "fixture_expected_vs_actual.csv",
        "summary": output_dir / "fixture_regression_summary.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_raw_registry_fixture_harness_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(fixture_manifest, output_paths["fixture_manifest"])
    write_csv(license_status, output_paths["license_status"])
    write_csv(second_registry, output_paths["second_registry"])
    write_csv(license_queue, output_paths["license_queue"])
    write_csv(handoff, output_paths["handoff"])
    write_csv(duplicate_report, output_paths["duplicate_report"])
    write_csv(tamper_report, output_paths["tamper_report"])
    write_csv(first_seen_report, output_paths["first_seen"])
    write_csv(source_gate_report, output_paths["source_gate"])
    write_csv(license_gate_report, output_paths["license_gate"])
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
        if term
        in " ".join(
            output_columns(
                [
                    fixture_manifest,
                    license_status,
                    first_registry,
                    second_registry,
                    license_queue,
                    handoff,
                    duplicate_report,
                    tamper_report,
                    first_seen_report,
                    source_gate_report,
                    license_gate_report,
                    expected_vs_actual,
                    summary,
                    guard,
                    acceptance,
                ]
            )
        ).lower()
    )
    target_was_written = second_config.target_source_path.exists() and not bool(v3_80_manifest.get("metrics", {}).get("target_source_exists", False))
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass_or_warn_only",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_80_manifest_passed",
                "status": "pass" if bool(v3_80_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_80_manifest.get('self_check_pass')}",
            },
            {
                "check": "all_expected_regressions_pass",
                "status": "pass" if expected_vs_actual["regression_pass"].astype(bool).all() else "fail",
                "detail": ",".join(expected_vs_actual.loc[~expected_vs_actual["regression_pass"].astype(bool), "case_id"].astype(str)),
            },
            {
                "check": "target_csv_not_written_by_v3_81",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={second_config.target_source_path.exists()}",
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not handoff["may_execute_now"].astype(bool).any() else "fail",
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

    fixture_paths = sorted(path for path in config.fixture_sample_dir.glob("*.csv") if path.is_file())
    evidence_paths = sorted(path for path in config.license_evidence_dir.glob("*") if path.is_file())
    outputs_for_changed = [
        output_paths["fixture_manifest"],
        output_paths["license_status"],
        output_paths["first_registry"],
        output_paths["second_registry"],
        output_paths["license_queue"],
        output_paths["handoff"],
        output_paths["duplicate_report"],
        output_paths["tamper_report"],
        output_paths["first_seen"],
        output_paths["source_gate"],
        output_paths["license_gate"],
        output_paths["expected_vs_actual"],
        output_paths["summary"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed + fixture_paths + evidence_paths + [config.license_status_path]), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    regression_pass_cases = int(expected_vs_actual["regression_pass"].astype(bool).sum()) if not expected_vs_actual.empty else 0
    registered_count = int(second_registry["sha256"].astype(str).ne("").sum()) if "sha256" in second_registry.columns else 0
    allowed_count = int(second_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in second_registry.columns else 0
    metrics = {
        "target_source_exists": second_config.target_source_path.exists(),
        "fixture_case_count": int(len(fixture_manifest)),
        "expected_case_count": int(len(expected_vs_actual)),
        "regression_pass_cases": regression_pass_cases,
        "registered_raw_sample_count": registered_count,
        "duplicate_hash_rows": int(second_registry["sha256"].duplicated(keep=False).sum()),
        "v3_78_review_allowed_count": allowed_count,
        "may_execute_v3_78_now": False,
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]] + fixture_paths + evidence_paths + [config.license_status_path]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.80 raw sample registry",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_81_market_label_raw_registry_fixture_harness.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_raw_sample_registry.py",
            "strategy_lab/market_label_raw_registry_fixture_harness.py",
            "strategy_lab/hirssm_v3_81_market_label_raw_registry_fixture_harness.py",
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
        "warn_count": int(acceptance["status"].eq("warn").sum()),
        "limitations": [
            "V3.81 uses synthetic fixtures and validates registry logic only.",
            "Fixture license evidence is synthetic and cannot approve real provider data.",
            "No sample validation, label generation, portfolio validation, or model promotion is allowed.",
        ],
        "risk_flags": [
            "synthetic_fixture_only",
            "external_license_still_required_for_real_samples",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Use V3.81 as the registry regression guard, then proceed to real-sample license evidence intake when a provider file arrives.",
        "handoff_summary": "V3.81 generated raw registry fixtures and proved V3.80 hash, duplicate, tamper, source-token, license, and first-seen gates.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
