#!/usr/bin/env python
"""Run HIRSSM V3.82 registered-sample intake orchestrator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_registered_sample_intake_orchestrator import (
    RegisteredSampleIntakeConfig,
    build_acceptance_checks,
    build_catalog,
    build_controlled_v3_78_configs,
    build_eligible_samples,
    build_execution_plan,
    build_no_execution_guard,
    build_registry_snapshot,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_registered_sample_intake_orchestrator_v3_82.json"
TASK_ID = "20260601_v3_82_market_label_registered_sample_intake_orchestrator"
VERSION = "V3.82"
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


def load_config(raw: dict[str, Any]) -> RegisteredSampleIntakeConfig:
    return RegisteredSampleIntakeConfig(
        v3_80_manifest_path=resolve_path(raw["v3_80_manifest_path"]),
        v3_80_registry_path=resolve_path(raw["v3_80_registry_path"]),
        v3_81_manifest_path=resolve_path(raw["v3_81_manifest_path"]),
        v3_78_config_path=resolve_path(raw["v3_78_config_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        controlled_config_dir=resolve_path(raw["controlled_config_dir"]),
        execute_v3_78=bool(raw.get("execute_v3_78", False)),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_registered_sample_intake_orchestrator.py",
        ROOT / "strategy_lab" / "hirssm_v3_82_market_label_registered_sample_intake_orchestrator.py",
        ROOT / "configs" / "market_label_registered_sample_intake_orchestrator_v3_82.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_82_market_label_registered_sample_intake_orchestrator.json",
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

    v3_80_manifest = read_json(config.v3_80_manifest_path)
    v3_81_manifest = read_json(config.v3_81_manifest_path)
    base_v3_78_config = read_json(config.v3_78_config_path)
    registry = read_csv(config.v3_80_registry_path)

    snapshot = build_registry_snapshot(registry)
    eligible = build_eligible_samples(snapshot)
    controlled_plan, controlled_payloads = build_controlled_v3_78_configs(eligible, base_v3_78_config, config)
    config.controlled_config_dir.mkdir(parents=True, exist_ok=True)
    written_controlled_configs: list[Path] = []
    for path_text, payload in controlled_payloads.items():
        path = ROOT / path_text
        write_json(payload, path)
        written_controlled_configs.append(path)
    execution_plan = build_execution_plan(eligible, controlled_plan, config)
    guard = build_no_execution_guard(config)
    acceptance = build_acceptance_checks(snapshot, eligible, execution_plan, guard, config)
    report = build_report(snapshot, eligible, controlled_plan, execution_plan, acceptance, config)
    catalog = build_catalog(snapshot, eligible, controlled_plan, config)

    output_paths = {
        "snapshot": output_dir / "registry_route_snapshot.csv",
        "eligible": output_dir / "eligible_registered_samples.csv",
        "controlled_plan": output_dir / "controlled_v3_78_plan.csv",
        "execution_plan": output_dir / "v3_78_execution_plan.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "registered_sample_intake_orchestrator_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(snapshot, output_paths["snapshot"])
    write_csv(eligible, output_paths["eligible"])
    write_csv(controlled_plan, output_paths["controlled_plan"])
    write_csv(execution_plan, output_paths["execution_plan"])
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
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([snapshot, eligible, controlled_plan, execution_plan, guard, acceptance])).lower())
    target_was_written = config.target_source_path.exists() and not bool(v3_80_manifest.get("metrics", {}).get("target_source_exists", False))
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
                "check": "v3_81_regression_passed",
                "status": "pass" if int(v3_81_manifest.get("metrics", {}).get("regression_pass_cases", 0)) == int(v3_81_manifest.get("metrics", {}).get("expected_case_count", -1)) else "fail",
                "detail": f"regression={v3_81_manifest.get('metrics', {}).get('regression_pass_cases')}/{v3_81_manifest.get('metrics', {}).get('expected_case_count')}",
            },
            {
                "check": "target_csv_not_written_by_v3_82",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={config.target_source_path.exists()}",
            },
            {
                "check": "v3_78_not_executed_by_default",
                "status": "pass" if not config.execute_v3_78 else "fail",
                "detail": f"execute_v3_78={config.execute_v3_78}",
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
        output_paths["snapshot"],
        output_paths["eligible"],
        output_paths["controlled_plan"],
        output_paths["execution_plan"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed + written_controlled_configs), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "registry_rows": int(len(snapshot)),
        "eligible_registered_sample_count": int(len(eligible)),
        "controlled_v3_78_config_count": int(len(written_controlled_configs)),
        "execute_v3_78": config.execute_v3_78,
        "may_execute_v3_78_now": bool(execution_plan["may_execute_now"].astype(bool).any()),
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]] + written_controlled_configs
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.80 registry and V3.81 registry regression",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_82_market_label_registered_sample_intake_orchestrator.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_registered_sample_intake_orchestrator.py",
            "strategy_lab/hirssm_v3_82_market_label_registered_sample_intake_orchestrator.py",
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
            "V3.82 is a route planner and does not execute V3.78 by default.",
            "If no V3.80 registry row is eligible, the route remains blocked.",
            "No raw sample copy, target source write, label generation, portfolio validation, or model promotion is allowed.",
        ],
        "risk_flags": [
            "no_eligible_registered_sample" if len(eligible) == 0 else "registered_sample_plan_created",
            "v3_78_not_executed",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "When V3.80 has an eligible real sample, rerun V3.82 and then execute the generated V3.78 controlled plan manually.",
        "handoff_summary": "V3.82 connected V3.80 registry approval to a guarded V3.78 route plan without executing downstream validation.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
