#!/usr/bin/env python
"""Run HIRSSM V3.85 real-sample intake dry-run harness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_real_sample_dry_run_harness import (
    RealSampleDryRunConfig,
    build_acceptance_checks,
    build_catalog,
    build_dry_run_sample,
    build_license_evidence,
    build_license_status,
    build_no_execution_guard,
    build_pipeline_trace,
    build_report,
    build_v3_78_execution_plan,
    make_raw_registry_config,
    make_registered_sample_config,
    read_csv_or_empty,
    rewrite_controlled_payloads,
    run_v3_78_command,
)
from market_label_raw_sample_registry import build_raw_sample_registry
from market_label_registered_sample_intake_orchestrator import (
    build_controlled_v3_78_configs,
    build_eligible_samples,
    build_execution_plan,
    build_registry_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_real_sample_dry_run_harness_v3_85.json"
TASK_ID = "20260602_v3_85_market_label_real_sample_dry_run_harness"
VERSION = "V3.85"
AGENT = "backtest_validation_auditor"


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


def load_config(raw: dict[str, Any]) -> RealSampleDryRunConfig:
    return RealSampleDryRunConfig(
        v3_83_manifest_path=resolve_path(raw["v3_83_manifest_path"]),
        v3_84_manifest_path=resolve_path(raw["v3_84_manifest_path"]),
        v3_78_base_config_path=resolve_path(raw["v3_78_base_config_path"]),
        v3_80_base_config_path=resolve_path(raw["v3_80_base_config_path"]),
        v3_82_base_config_path=resolve_path(raw["v3_82_base_config_path"]),
        real_incoming_sample_dir=resolve_path(raw["real_incoming_sample_dir"]),
        real_license_status_path=resolve_path(raw["real_license_status_path"]),
        real_target_source_path=resolve_path(raw["real_target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        sandbox_root=resolve_path(raw["sandbox_root"]),
        sandbox_incoming_sample_dir=resolve_path(raw["sandbox_incoming_sample_dir"]),
        sandbox_license_status_path=resolve_path(raw["sandbox_license_status_path"]),
        sandbox_target_source_path=resolve_path(raw["sandbox_target_source_path"]),
        sandbox_registry_path=resolve_path(raw["sandbox_registry_path"]),
        controlled_config_dir=resolve_path(raw["controlled_config_dir"]),
        dry_run_sample_rows=int(raw.get("dry_run_sample_rows", 30)),
    )


def build_changed_files(outputs: list[Path], generated_configs: list[Path], v3_78_outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_raw_sample_registry.py",
        ROOT / "strategy_lab" / "market_label_sample_intake_validator.py",
        ROOT / "strategy_lab" / "market_label_real_sample_dry_run_harness.py",
        ROOT / "strategy_lab" / "hirssm_v3_85_market_label_real_sample_dry_run_harness.py",
        ROOT / "configs" / "market_label_real_sample_dry_run_harness_v3_85.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260602_v3_85_market_label_real_sample_dry_run_harness.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + generated_configs + v3_78_outputs + outputs)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    config.sandbox_incoming_sample_dir.mkdir(parents=True, exist_ok=True)
    config.controlled_config_dir.mkdir(parents=True, exist_ok=True)

    v3_83_manifest = read_json(config.v3_83_manifest_path)
    v3_84_manifest = read_json(config.v3_84_manifest_path)
    v3_78_base = read_json(config.v3_78_base_config_path)
    v3_80_base = read_json(config.v3_80_base_config_path)
    v3_82_base = read_json(config.v3_82_base_config_path)

    real_incoming_before = config.real_incoming_sample_dir.exists()
    real_target_before = config.real_target_source_path.exists()

    dry_run_sample_path = config.sandbox_incoming_sample_dir / "csindex_official_total_return_service_market_sample.csv"
    dry_run_sample = build_dry_run_sample(config.dry_run_sample_rows)
    write_csv(dry_run_sample, dry_run_sample_path)

    missing_license_path = output_dir / "missing_license_status_DO_NOT_CREATE.csv"
    stage1_registry_config = make_raw_registry_config(v3_80_base, config, missing_license_path, output_dir / "v3_80_stage1_without_license")
    stage1_registry = build_raw_sample_registry(stage1_registry_config)
    if "sample_file" in stage1_registry.columns and "sha256" in stage1_registry.columns:
        stage1_registry.loc[stage1_registry["sha256"].astype(str).ne(""), "sample_file"] = rel(dry_run_sample_path)
    write_csv(stage1_registry, output_dir / "v3_80_stage1_registry_without_license.csv")
    write_csv(stage1_registry, config.sandbox_registry_path)

    sample_sha = str(stage1_registry.loc[stage1_registry["sha256"].astype(str).ne(""), "sha256"].iloc[0])
    license_evidence_path = output_dir / "license_evidence" / "dry_run_license_evidence.md"
    write_text(build_license_evidence(dry_run_sample_path, sample_sha), license_evidence_path)
    license_status = build_license_status(dry_run_sample_path, sample_sha, license_evidence_path)
    license_status.loc[:, "sample_file"] = rel(dry_run_sample_path)
    write_csv(license_status, config.sandbox_license_status_path)
    write_csv(license_status, output_dir / "sandbox_license_review_status.csv")

    stage2_registry_config = make_raw_registry_config(v3_80_base, config, config.sandbox_license_status_path, output_dir / "v3_80_stage2_with_license")
    stage2_registry = build_raw_sample_registry(stage2_registry_config)
    if "sample_file" in stage2_registry.columns and "sha256" in stage2_registry.columns:
        stage2_registry.loc[stage2_registry["sha256"].astype(str).ne(""), "sample_file"] = rel(dry_run_sample_path)
    write_csv(stage2_registry, output_dir / "v3_80_stage2_registry_with_license.csv")
    write_csv(stage2_registry, config.sandbox_registry_path)

    route_config = make_registered_sample_config(config, v3_82_base)
    route_snapshot = build_registry_snapshot(stage2_registry)
    eligible = build_eligible_samples(route_snapshot)
    controlled_plan, controlled_payloads = build_controlled_v3_78_configs(eligible, v3_78_base, route_config)
    rewritten_payloads = rewrite_controlled_payloads(controlled_payloads, config)
    generated_configs: list[Path] = []
    for path_text, payload in rewritten_payloads.items():
        path = ROOT / path_text
        write_json(payload, path)
        generated_configs.append(path)
    execution_plan = build_execution_plan(eligible, controlled_plan, route_config)
    v3_78_execution_plan = build_v3_78_execution_plan(controlled_plan, rewritten_payloads)

    execution_rows: list[dict[str, Any]] = []
    v3_78_decision_frames: list[pd.DataFrame] = []
    v3_78_outputs: list[Path] = []
    for _, row in v3_78_execution_plan.iterrows():
        returncode, stdout, stderr = run_v3_78_command(ROOT, str(row["command"]))
        v3_78_output_dir = ROOT / str(row["v3_78_output_dir"])
        decision_path = v3_78_output_dir / "candidate_file_decision.csv"
        manifest_path = v3_78_output_dir / "agent_run_manifest.json"
        decisions = read_csv_or_empty(decision_path)
        if not decisions.empty:
            decisions["controlled_config_path"] = row["controlled_config_path"]
            v3_78_decision_frames.append(decisions)
        for path in [decision_path, manifest_path, v3_78_output_dir / "self_check.csv", v3_78_output_dir / "acceptance_checks.csv"]:
            if path.exists():
                v3_78_outputs.append(path)
        execution_rows.append(
            {
                "controlled_config_path": row["controlled_config_path"],
                "command": row["command"],
                "returncode": int(returncode),
                "v3_78_output_dir": row["v3_78_output_dir"],
                "v3_78_manifest_path": rel(manifest_path) if manifest_path.exists() else "",
                "v3_78_decision_path": rel(decision_path) if decision_path.exists() else "",
                "stdout_tail": stdout.strip()[-500:],
                "stderr_tail": stderr.strip()[-500:],
            }
        )
    execution_results = pd.DataFrame(execution_rows)
    v3_78_decisions = pd.concat(v3_78_decision_frames, ignore_index=True) if v3_78_decision_frames else pd.DataFrame()

    pipeline_trace = build_pipeline_trace(stage1_registry, stage2_registry, eligible, controlled_plan, execution_results, v3_78_decisions)
    real_incoming_after = config.real_incoming_sample_dir.exists()
    real_target_after = config.real_target_source_path.exists()
    guard = build_no_execution_guard(real_incoming_before, real_incoming_after, real_target_before, real_target_after)
    acceptance = build_acceptance_checks(
        pipeline_trace,
        stage1_registry,
        stage2_registry,
        eligible,
        controlled_plan,
        execution_results,
        v3_78_decisions,
        guard,
        real_incoming_before,
        real_incoming_after,
        real_target_before,
        real_target_after,
    )
    report = build_report(pipeline_trace, acceptance, stage2_registry, controlled_plan, v3_78_decisions, config)
    catalog = build_catalog(config, pipeline_trace)

    output_paths = {
        "dry_run_sample": dry_run_sample_path,
        "stage1_registry": output_dir / "v3_80_stage1_registry_without_license.csv",
        "license_evidence": license_evidence_path,
        "license_status": output_dir / "sandbox_license_review_status.csv",
        "stage2_registry": output_dir / "v3_80_stage2_registry_with_license.csv",
        "route_snapshot": output_dir / "v3_82_registry_route_snapshot.csv",
        "eligible": output_dir / "v3_82_eligible_registered_samples.csv",
        "controlled_plan": output_dir / "v3_82_controlled_v3_78_plan.csv",
        "route_execution_plan": output_dir / "v3_82_execution_plan.csv",
        "v3_78_execution_plan": output_dir / "v3_78_execution_plan.csv",
        "v3_78_execution_results": output_dir / "v3_78_execution_results.csv",
        "v3_78_decisions": output_dir / "v3_78_candidate_file_decision.csv",
        "pipeline_trace": output_dir / "dry_run_pipeline_trace.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "real_sample_intake_dry_run_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(route_snapshot, output_paths["route_snapshot"])
    write_csv(eligible, output_paths["eligible"])
    write_csv(controlled_plan, output_paths["controlled_plan"])
    write_csv(execution_plan, output_paths["route_execution_plan"])
    write_csv(v3_78_execution_plan, output_paths["v3_78_execution_plan"])
    write_csv(execution_results, output_paths["v3_78_execution_results"])
    write_csv(v3_78_decisions, output_paths["v3_78_decisions"])
    write_csv(pipeline_trace, output_paths["pipeline_trace"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    v3_78_manifest_pass = False
    for manifest_path_text in execution_results.get("v3_78_manifest_path", pd.Series(dtype=str)).dropna().astype(str):
        if manifest_path_text:
            manifest = read_json(ROOT / manifest_path_text)
            v3_78_manifest_pass = v3_78_manifest_pass or bool(manifest.get("self_check_pass", False))

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
                    stage1_registry,
                    license_status,
                    stage2_registry,
                    route_snapshot,
                    eligible,
                    controlled_plan,
                    execution_plan,
                    v3_78_execution_plan,
                    execution_results,
                    v3_78_decisions,
                    pipeline_trace,
                    guard,
                    acceptance,
                ]
            )
        ).lower()
    )
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_83_manifest_passed",
                "status": "pass" if bool(v3_83_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_83_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_84_manifest_passed",
                "status": "pass" if bool(v3_84_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_84_manifest.get('self_check_pass')}",
            },
            {
                "check": "controlled_v3_78_manifest_passed",
                "status": "pass" if v3_78_manifest_pass else "fail",
                "detail": f"self_check={v3_78_manifest_pass}",
            },
            {
                "check": "real_paths_unchanged",
                "status": "pass" if real_incoming_before == real_incoming_after and real_target_before == real_target_after else "fail",
                "detail": f"incoming={real_incoming_before}->{real_incoming_after};target={real_target_before}->{real_target_after}",
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
        output_paths["dry_run_sample"],
        output_paths["stage1_registry"],
        output_paths["license_evidence"],
        output_paths["license_status"],
        output_paths["stage2_registry"],
        output_paths["route_snapshot"],
        output_paths["eligible"],
        output_paths["controlled_plan"],
        output_paths["route_execution_plan"],
        output_paths["v3_78_execution_plan"],
        output_paths["v3_78_execution_results"],
        output_paths["v3_78_decisions"],
        output_paths["pipeline_trace"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed, generated_configs, v3_78_outputs), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    stage1_allowed = int(stage1_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage1_registry.columns else 0
    stage2_allowed = int(stage2_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage2_registry.columns else 0
    v3_78_pass_count = int(v3_78_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if "decision" in v3_78_decisions.columns else 0
    metrics = {
        "dry_run_sample_rows": int(len(dry_run_sample)),
        "v3_80_stage1_review_allowed_count": stage1_allowed,
        "v3_80_stage2_review_allowed_count": stage2_allowed,
        "v3_82_eligible_sample_count": int(len(eligible)),
        "controlled_v3_78_config_count": int(len(generated_configs)),
        "controlled_v3_78_returncode_zero": bool(not execution_results.empty and execution_results["returncode"].eq(0).all()),
        "controlled_v3_78_pass_count": v3_78_pass_count,
        "real_incoming_dir_state_unchanged": bool(real_incoming_before == real_incoming_after),
        "real_target_source_state_unchanged": bool(real_target_before == real_target_after),
        "sandbox_target_source_exists": config.sandbox_target_source_path.exists(),
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + generated_configs + v3_78_outputs + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.83 real-sample operations pack and V3.84 global governance gate",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_85_market_label_real_sample_dry_run_harness.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_real_sample_dry_run_harness.py",
            "strategy_lab/hirssm_v3_85_market_label_real_sample_dry_run_harness.py",
            "strategy_lab/market_label_raw_sample_registry.py",
            "strategy_lab/market_label_registered_sample_intake_orchestrator.py",
            "strategy_lab/market_label_sample_intake_validator.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in all_outputs],
        "changed_files": build_changed_files(all_outputs, [], []).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": 0,
        "limitations": [
            "V3.85 uses synthetic sandbox data; it is pipeline evidence, not market evidence.",
            "A real provider sample still requires manual license evidence and V3.75/V3.76 gates.",
            "No official target source, labels, portfolio validation, or model promotion is produced.",
        ],
        "risk_flags": [
            "synthetic_dry_run_only",
            "real_provider_sample_still_required",
            "official_target_write_blocked",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Use V3.83 operator runbook with a real licensed provider sample, then rerun V3.80, V3.82, and controlled V3.78 before V3.75/V3.76.",
        "handoff_summary": "V3.85 proved the governed sample-intake plumbing in a sandbox from registry through controlled V3.78 validation.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
