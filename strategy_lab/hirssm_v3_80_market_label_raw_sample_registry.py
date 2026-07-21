#!/usr/bin/env python
"""Run HIRSSM V3.80 immutable raw sample registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_raw_sample_registry import (
    RawSampleRegistryConfig,
    build_acceptance_checks,
    build_catalog,
    build_controlled_handoff,
    build_immutability_policy,
    build_license_review_queue,
    build_no_execution_guard,
    build_raw_sample_registry,
    build_registry_template,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_raw_sample_registry_v3_80.json"
TASK_ID = "20260601_v3_80_market_label_raw_sample_registry"
VERSION = "V3.80"
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


def load_config(raw: dict[str, Any]) -> RawSampleRegistryConfig:
    return RawSampleRegistryConfig(
        v3_78_manifest_path=resolve_path(raw["v3_78_manifest_path"]),
        v3_79_manifest_path=resolve_path(raw["v3_79_manifest_path"]),
        incoming_sample_dir=resolve_path(raw["incoming_sample_dir"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        previous_registry_path=resolve_path(raw["previous_registry_path"]),
        license_status_path=resolve_path(raw.get("license_status_path", "data_raw/market_labels/incoming_samples/license_review_status.csv")),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        approved_source_tokens=tuple(str(x) for x in raw["approved_source_tokens"]),
        allowed_extensions=tuple(str(x) for x in raw["allowed_extensions"]),
        license_approved_values=tuple(str(x) for x in raw["license_approved_values"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_raw_sample_registry.py",
        ROOT / "strategy_lab" / "hirssm_v3_80_market_label_raw_sample_registry.py",
        ROOT / "configs" / "market_label_raw_sample_registry_v3_80.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_80_market_label_raw_sample_registry.json",
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

    v3_78_manifest = read_json(config.v3_78_manifest_path)
    v3_79_manifest = read_json(config.v3_79_manifest_path)

    registry = build_raw_sample_registry(config)
    template = build_registry_template()
    license_queue = build_license_review_queue(registry)
    handoff = build_controlled_handoff(registry, config)
    policy = build_immutability_policy(config)
    guard = build_no_execution_guard()
    acceptance = build_acceptance_checks(registry, license_queue, handoff, guard, config)
    report = build_report(registry, license_queue, handoff, acceptance, config)
    catalog = build_catalog(registry, config)

    output_paths = {
        "registry": output_dir / "raw_sample_registry.csv",
        "template": output_dir / "raw_sample_registry_template.csv",
        "license_queue": output_dir / "license_review_queue.csv",
        "handoff": output_dir / "controlled_review_handoff.csv",
        "policy": output_dir / "immutability_policy.md",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_raw_sample_registry_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(registry, output_paths["registry"])
    write_csv(template, output_paths["template"])
    write_csv(license_queue, output_paths["license_queue"])
    write_csv(handoff, output_paths["handoff"])
    write_text(policy, output_paths["policy"])
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
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([registry, template, license_queue, handoff, guard, acceptance])).lower())
    target_was_written = config.target_source_path.exists() and not bool(v3_78_manifest.get("metrics", {}).get("target_source_exists", False))
    registered_count = int(registry["sha256"].astype(str).ne("").sum()) if "sha256" in registry.columns else 0
    allowed_count = int(registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in registry.columns else 0
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
                "check": "v3_79_regression_passed",
                "status": "pass" if int(v3_79_manifest.get("metrics", {}).get("regression_pass_cases", 0)) == int(v3_79_manifest.get("metrics", {}).get("expected_case_count", -1)) else "fail",
                "detail": f"regression={v3_79_manifest.get('metrics', {}).get('regression_pass_cases')}/{v3_79_manifest.get('metrics', {}).get('expected_case_count')}",
            },
            {
                "check": "target_csv_not_written_by_v3_80",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={config.target_source_path.exists()}",
            },
            {
                "check": "v3_78_not_executed_by_registry",
                "status": "pass" if not handoff["may_execute_now"].astype(bool).any() else "fail",
                "detail": "registry does not execute downstream steps",
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
        output_paths["registry"],
        output_paths["template"],
        output_paths["license_queue"],
        output_paths["handoff"],
        output_paths["policy"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "incoming_sample_dir_exists": config.incoming_sample_dir.exists(),
        "registered_raw_sample_count": registered_count,
        "v3_78_review_allowed_count": allowed_count,
        "license_blocked_rows": int(license_queue["review_status"].astype(str).eq("blocked").sum()) if "review_status" in license_queue.columns else 0,
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
        "baseline": "V3.78 sample validator and V3.79 regression harness",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_80_market_label_raw_sample_registry.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_raw_sample_registry.py",
            "strategy_lab/hirssm_v3_80_market_label_raw_sample_registry.py",
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
            "V3.80 records provenance and license status only.",
            "License fields are manual governance inputs and remain missing until evidence is attached.",
            "No sample validation, label generation, portfolio validation, or model promotion is allowed.",
        ],
        "risk_flags": [
            "no_registered_real_sample" if registered_count == 0 else "registered_sample_waiting_license",
            "external_license_required",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Attach license evidence for real samples, rerun V3.80, then rerun V3.78 only for registry-approved samples.",
        "handoff_summary": "V3.80 created an immutable raw-sample registry, license review queue, controlled handoff, and provenance policy.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
