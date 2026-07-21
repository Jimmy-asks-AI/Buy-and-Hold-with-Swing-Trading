#!/usr/bin/env python
"""Run HIRSSM V3.78 MARKET label sample intake validator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_sample_intake_validator import (
    SampleIntakeConfig,
    build_acceptance_checks,
    build_action_queue,
    build_catalog,
    build_dropzone_readme,
    build_expected_contract,
    build_no_execution_guard,
    build_report,
    discover_sample_files,
    validate_samples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_sample_intake_validator_v3_78.json"
TASK_ID = "20260601_v3_78_market_label_sample_intake_validator"
VERSION = "V3.78"
AGENT = "data_steward"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def load_config(raw: dict[str, Any]) -> SampleIntakeConfig:
    return SampleIntakeConfig(
        v3_77_manifest_path=resolve_path(raw["v3_77_manifest_path"]),
        v3_77_source_candidates_path=resolve_path(raw["v3_77_source_candidates_path"]),
        v3_75_source_contract_path=resolve_path(raw["v3_75_source_contract_path"]),
        incoming_sample_dir=resolve_path(raw["incoming_sample_dir"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        min_sample_rows=int(raw["min_sample_rows"]),
        max_available_lag_days=int(raw["max_available_lag_days"]),
        max_abs_daily_change=float(raw["max_abs_daily_change"]),
        allowed_asset_values=tuple(str(x) for x in raw["allowed_asset_values"]),
        approved_source_decisions=tuple(str(x) for x in raw["approved_source_decisions"]),
        approved_source_tokens=tuple(str(x) for x in raw["approved_source_tokens"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_sample_intake_validator.py",
        ROOT / "strategy_lab" / "hirssm_v3_78_market_label_sample_intake_validator.py",
        ROOT / "configs" / "market_label_sample_intake_validator_v3_78.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_78_market_label_sample_intake_validator.json",
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

    v3_77_manifest = read_json(config.v3_77_manifest_path)
    v3_77_sources = read_csv(config.v3_77_source_candidates_path)
    v3_75_contract = read_csv(config.v3_75_source_contract_path)

    inventory = discover_sample_files(config)
    expected_contract = build_expected_contract(v3_75_contract)
    validation_checks, sample_decisions = validate_samples(config)
    action_queue = build_action_queue(sample_decisions, config)
    dropzone_readme = build_dropzone_readme(config, expected_contract)
    guard = build_no_execution_guard()
    acceptance = build_acceptance_checks(inventory, validation_checks, sample_decisions, action_queue, guard, config)
    report = build_report(inventory, validation_checks, sample_decisions, action_queue, acceptance, config)
    catalog = build_catalog(sample_decisions, config)

    output_paths = {
        "inventory": output_dir / "sample_inventory.csv",
        "contract": output_dir / "expected_sample_contract.csv",
        "validation": output_dir / "sample_validation_checks.csv",
        "decisions": output_dir / "candidate_file_decision.csv",
        "action_queue": output_dir / "intake_action_queue.csv",
        "dropzone_readme": output_dir / "incoming_sample_dropzone_readme.md",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_sample_intake_validator_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(inventory, output_paths["inventory"])
    write_csv(expected_contract, output_paths["contract"])
    write_csv(validation_checks, output_paths["validation"])
    write_csv(sample_decisions, output_paths["decisions"])
    write_csv(action_queue, output_paths["action_queue"])
    write_text(dropzone_readme, output_paths["dropzone_readme"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    allowed_sources = set(config.approved_source_decisions)
    approved_route_count = int(v3_77_sources["decision"].astype(str).isin(allowed_sources).sum()) if "decision" in v3_77_sources.columns else 0
    target_was_written = config.target_source_path.exists() and not bool(v3_77_manifest.get("metrics", {}).get("target_source_exists", False))
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
        if term in " ".join(output_columns([inventory, expected_contract, validation_checks, sample_decisions, action_queue, guard, acceptance])).lower()
    )
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass_or_warn_only",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_77_manifest_passed",
                "status": "pass" if bool(v3_77_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_77_manifest.get('self_check_pass')}",
            },
            {
                "check": "approved_procurement_routes_available",
                "status": "pass" if approved_route_count >= 1 else "fail",
                "detail": f"approved_route_count={approved_route_count}",
            },
            {
                "check": "target_csv_not_written_by_v3_78",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={config.target_source_path.exists()}",
            },
            {
                "check": "v3_53_not_executable",
                "status": "pass" if not action_queue["may_execute_now"].astype(bool).any() else "fail",
                "detail": "all downstream actions remain manual-controlled",
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
        output_paths["inventory"],
        output_paths["contract"],
        output_paths["validation"],
        output_paths["decisions"],
        output_paths["action_queue"],
        output_paths["dropzone_readme"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    warn_count = int(acceptance["status"].eq("warn").sum()) + int(sample_decisions["warning_count"].sum()) if "warning_count" in sample_decisions.columns else 0
    passing_samples = int(sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if not sample_decisions.empty else 0
    waiting_samples = int(sample_decisions["decision"].astype(str).eq("waiting_for_sample").sum()) if not sample_decisions.empty else 0
    found_samples = int(inventory["status"].astype(str).eq("found").sum()) if not inventory.empty else 0
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "incoming_sample_dir_exists": config.incoming_sample_dir.exists(),
        "sample_files_found": found_samples,
        "passing_samples": passing_samples,
        "waiting_sample_rows": waiting_samples,
        "approved_procurement_route_count": approved_route_count,
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.77 source discovery route decision",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_78_market_label_sample_intake_validator.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_sample_intake_validator.py",
            "strategy_lab/hirssm_v3_78_market_label_sample_intake_validator.py",
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
        "warn_count": warn_count,
        "limitations": [
            "V3.78 validates incoming samples but does not fetch, license, or write official source data.",
            "A passing sample still requires V3.75 contract review and V3.76 routing before V3.53.",
            "No portfolio validation or model promotion is allowed from sample intake evidence.",
        ],
        "risk_flags": [
            "no_sample_file_found" if found_samples == 0 else "sample_file_needs_contract_review",
            "external_license_required",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Place a licensed provider sample under the incoming sample directory and rerun V3.78; if it passes, rerun V3.75 on a controlled copy.",
        "handoff_summary": "V3.78 created a reusable sample intake validator, dropzone instructions, candidate-file decisions, and action queue for provider samples.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
