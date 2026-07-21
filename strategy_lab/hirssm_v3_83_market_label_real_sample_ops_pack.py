#!/usr/bin/env python
"""Run HIRSSM V3.83 real-sample operations pack."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_real_sample_ops_pack import (
    RealSampleOpsPackConfig,
    build_acceptance_checks,
    build_catalog,
    build_command_runbook,
    build_human_input_checklist,
    build_license_evidence_template,
    build_license_review_status_template,
    build_no_execution_guard,
    build_ops_manual,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_real_sample_ops_pack_v3_83.json"
TASK_ID = "20260601_v3_83_market_label_real_sample_ops_pack"
VERSION = "V3.83"
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> RealSampleOpsPackConfig:
    return RealSampleOpsPackConfig(
        v3_82_manifest_path=resolve_path(raw["v3_82_manifest_path"]),
        v3_80_registry_template_path=resolve_path(raw["v3_80_registry_template_path"]),
        v3_78_config_path=resolve_path(raw["v3_78_config_path"]),
        v3_75_vendor_template_path=resolve_path(raw["v3_75_vendor_template_path"]),
        incoming_sample_dir=resolve_path(raw["incoming_sample_dir"]),
        license_evidence_dir=resolve_path(raw["license_evidence_dir"]),
        license_status_path=resolve_path(raw["license_status_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_real_sample_ops_pack.py",
        ROOT / "strategy_lab" / "hirssm_v3_83_market_label_real_sample_ops_pack.py",
        ROOT / "configs" / "market_label_real_sample_ops_pack_v3_83.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_83_market_label_real_sample_ops_pack.json",
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

    v3_82_manifest = read_json(config.v3_82_manifest_path)

    command_runbook = build_command_runbook(config)
    checklist = build_human_input_checklist(config)
    license_status_template = build_license_review_status_template(config)
    guard = build_no_execution_guard()
    acceptance = build_acceptance_checks(command_runbook, checklist, guard, config, v3_82_manifest)
    ops_manual = build_ops_manual(config, command_runbook, checklist)
    license_evidence_template = build_license_evidence_template(config)
    report = build_report(command_runbook, checklist, acceptance, guard, config)
    catalog = build_catalog(config)

    output_paths = {
        "ops_manual": output_dir / "real_sample_ops_manual.md",
        "license_evidence_template": output_dir / "license_evidence.template.md",
        "license_status_template": output_dir / "license_review_status.template.csv",
        "command_runbook": output_dir / "command_runbook.csv",
        "human_checklist": output_dir / "human_input_checklist.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "real_sample_ops_pack_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_text(ops_manual, output_paths["ops_manual"])
    write_text(license_evidence_template, output_paths["license_evidence_template"])
    write_csv(license_status_template, output_paths["license_status_template"])
    write_csv(command_runbook, output_paths["command_runbook"])
    write_csv(checklist, output_paths["human_checklist"])
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
                    license_status_template,
                    command_runbook,
                    checklist,
                    guard,
                    acceptance,
                ]
            )
        ).lower()
    )
    output_files_exist = all(path.exists() for key, path in output_paths.items() if key not in {"self_check", "changed_files", "manifest"})
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_82_manifest_passed",
                "status": "pass" if bool(v3_82_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_82_manifest.get('self_check_pass')}",
            },
            {
                "check": "required_templates_written",
                "status": "pass" if output_files_exist else "fail",
                "detail": f"outputs_written={output_files_exist}",
            },
            {
                "check": "command_runbook_blocks_execution",
                "status": "pass" if not command_runbook["may_execute_now"].astype(bool).any() else "fail",
                "detail": "all may_execute_now values are false",
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard["produced"].astype(bool).iloc[1:].any() else "fail",
                "detail": "only ops_pack produced",
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
        output_paths["ops_manual"],
        output_paths["license_evidence_template"],
        output_paths["license_status_template"],
        output_paths["command_runbook"],
        output_paths["human_checklist"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "ops_manual_written": output_paths["ops_manual"].exists(),
        "license_template_written": output_paths["license_evidence_template"].exists(),
        "license_status_template_written": output_paths["license_status_template"].exists(),
        "command_runbook_rows": int(len(command_runbook)),
        "human_checklist_rows": int(len(checklist)),
        "target_source_exists": config.target_source_path.exists(),
        "incoming_sample_dir_exists": config.incoming_sample_dir.exists(),
        "may_execute_v3_78_now": False,
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
        "baseline": "V3.82 registered-sample route planner",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_83_market_label_real_sample_ops_pack.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_real_sample_ops_pack.py",
            "strategy_lab/hirssm_v3_83_market_label_real_sample_ops_pack.py",
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
            "V3.83 is an operations pack, not a data ingestion run.",
            "It does not create the real incoming sample directory or write the license status file in place.",
            "No sample validation, label generation, portfolio validation, or model promotion is allowed.",
        ],
        "risk_flags": [
            "manual_sample_required",
            "manual_license_review_required",
            "v3_78_not_executed",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "When a real provider sample is available, follow the V3.83 runbook through V3.80, V3.82, and controlled V3.78 validation.",
        "handoff_summary": "V3.83 converted the blocked real-sample intake gap into a repeatable operator runbook with license and execution gates.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
