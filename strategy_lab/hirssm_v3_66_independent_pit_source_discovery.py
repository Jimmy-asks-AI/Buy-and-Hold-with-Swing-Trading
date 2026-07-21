#!/usr/bin/env python
"""Run HIRSSM V3.66 independent PIT source discovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from independent_pit_source_discovery import (
    IndependentPitSourceConfig,
    build_acceptance_checks,
    build_blocked_data_requests,
    build_catalog,
    build_no_promotion_guard,
    build_report,
    build_signal_blueprints,
    build_source_inventory,
    build_v3_65_branch_policy,
    build_validation_queue,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "independent_pit_source_discovery_v3_66.json"
TASK_ID = "20260529_v3_66_independent_pit_source_discovery"
VERSION = "V3.66"
AGENT = "chief_quant_orchestrator"


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


def load_config(raw: dict[str, Any]) -> IndependentPitSourceConfig:
    return IndependentPitSourceConfig(
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        v3_65_manifest_path=resolve_path(raw["v3_65_manifest_path"]),
        v3_65_decision_path=resolve_path(raw["v3_65_decision_path"]),
        min_ready_priority_score=float(raw["queue_thresholds"]["min_ready_priority_score"]),
        max_feature_layer_queue_rows=int(raw["queue_thresholds"]["max_feature_layer_queue_rows"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "independent_pit_source_discovery.py",
        ROOT / "strategy_lab" / "hirssm_v3_66_independent_pit_source_discovery.py",
        ROOT / "configs" / "independent_pit_source_discovery_v3_66.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_66_independent_pit_source_discovery.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    v3_65_manifest = read_json(config.v3_65_manifest_path)
    v3_65_decisions = pd.read_csv(config.v3_65_decision_path, encoding="utf-8-sig", low_memory=False)

    input_checks = validate_inputs(v3_65_manifest, v3_65_decisions)
    inventory = build_source_inventory(ROOT)
    blueprints = build_signal_blueprints(inventory)
    branch_policy = build_v3_65_branch_policy(v3_65_decisions)
    queue = build_validation_queue(blueprints, inventory, config)
    blocked_requests = build_blocked_data_requests(inventory)
    guard = build_no_promotion_guard()
    output_frames = [input_checks, inventory, blueprints, branch_policy, queue, blocked_requests, guard]
    acceptance = build_acceptance_checks(input_checks, inventory, blueprints, queue, branch_policy, guard, output_columns(output_frames))
    report = build_report(inventory, blueprints, queue, blocked_requests, branch_policy, input_checks, acceptance)
    catalog = build_catalog(inventory, blueprints, queue)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "inventory": output_dir / "source_inventory.csv",
        "blueprints": output_dir / "signal_blueprints.csv",
        "branch_policy": output_dir / "v3_65_branch_policy.csv",
        "queue": output_dir / "next_validation_queue.csv",
        "blocked_requests": output_dir / "blocked_data_requests.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "independent_pit_source_discovery_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(inventory, output_paths["inventory"])
    write_csv(blueprints, output_paths["blueprints"])
    write_csv(branch_policy, output_paths["branch_policy"])
    write_csv(queue, output_paths["queue"])
    write_csv(blocked_requests, output_paths["blocked_requests"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "outputs_present",
                "status": "pass" if not inventory.empty and not blueprints.empty and not queue.empty else "fail",
                "detail": f"inventory={len(inventory)};blueprints={len(blueprints)};queue={len(queue)}",
            },
            {
                "check": "promotion_guard_blocks_model_change",
                "status": "pass"
                if not guard.loc[guard["result_type"].isin(["portfolio_harness", "model_promotion"]), "produced"].astype(bool).any()
                else "fail",
                "detail": "source discovery only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["inventory"],
        output_paths["blueprints"],
        output_paths["branch_policy"],
        output_paths["queue"],
        output_paths["blocked_requests"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    feature_ready_stages = {"ready_for_feature_layer", "ready_for_nonreturn_breadth_feature_layer"}
    feature_ready_sources = int(inventory["allowed_stage"].isin(feature_ready_stages).sum()) if not inventory.empty else 0
    eligible_blueprints = int(blueprints["allowed_for_v3_67_feature_layer"].astype(bool).sum()) if not blueprints.empty else 0
    blocked_sources = int(inventory["allowed_stage"].astype(str).str.startswith("blocked").sum()) if not inventory.empty else 0
    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "source_inventory_rows": int(len(inventory)),
        "feature_ready_source_rows": feature_ready_sources,
        "blocked_source_rows": blocked_sources,
        "signal_blueprint_rows": int(len(blueprints)),
        "eligible_feature_layer_blueprint_rows": eligible_blueprints,
        "next_validation_queue_rows": int(len(queue)),
        "blocked_data_request_rows": int(len(blocked_requests)),
        "portfolio_harness_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.65 walk-forward failure attribution",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_66_independent_pit_source_discovery.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_65_manifest_path),
            rel(config.v3_65_decision_path),
            "data_raw/index/akshare_sw_industry/daily_sw",
            "data_raw/index/akshare_csindex/daily_csindex",
            "data_raw/macro/macro_pit_panel.csv",
            "data_raw/tushare_daily_only/v3_38/daily",
        ],
        "code_refs": [
            "strategy_lab/independent_pit_source_discovery.py",
            "strategy_lab/hirssm_v3_66_independent_pit_source_discovery.py",
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
        "warn_count": 0,
        "limitations": [
            "V3.66 is source and signal discovery only.",
            "No portfolio harness or default model change is produced.",
            "Tushare raw daily stock data is allowed for breadth features only until adjustment factors are available.",
        ],
        "risk_flags": [
            "source_discovery_only",
            "raw_stock_adjustment_guard_required",
            "current_snapshot_sources_blocked",
            "model_promotion_blocked",
        ],
        "next_decision": "Start V3.67 with market participation breadth feature-layer construction under a raw-adjustment guard.",
        "handoff_summary": "V3.66 pivoted from same-branch proxy tuning to independent PIT source discovery and queued feature-layer tasks.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
