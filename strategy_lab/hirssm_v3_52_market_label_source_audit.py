#!/usr/bin/env python
"""Run HIRSSM V3.52 market-label source audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_source_audit import (
    MARKET_LABEL_REQUIREMENTS,
    AuditConfig,
    audit_local_sources,
    build_acceptance_checks,
    build_acquisition_plan,
    build_catalog,
    build_no_label_guard,
    build_readiness_checks,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_source_audit_v3_52.json"
TASK_ID = "20260529_v3_52_market_label_source_audit"
VERSION = "V3.52"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> AuditConfig:
    return AuditConfig(
        glob_patterns=tuple(str(item) for item in raw["candidate_file_globs"]),
        market_proxy_codes=tuple(str(item) for item in raw["market_proxy_codes"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_source_audit.py",
        ROOT / "strategy_lab" / "hirssm_v3_52_market_label_source_audit.py",
        ROOT / "configs" / "market_label_source_audit_v3_52.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_52_market_label_source_audit.json",
    ]
    all_files = static_files + outputs
    return "\n".join(rel(path) for path in all_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    raw = read_json(config_path)
    config = load_config(raw)
    output_dir = config.output_dir

    inventory, assessments = audit_local_sources(ROOT, config)
    readiness = build_readiness_checks(assessments)
    guard = build_no_label_guard(assessments)
    requirements = pd.DataFrame(MARKET_LABEL_REQUIREMENTS)
    acquisition_plan = build_acquisition_plan()
    acceptance = build_acceptance_checks(assessments, readiness, guard)
    report = build_report(inventory, assessments, readiness, acquisition_plan, acceptance)
    catalog = build_catalog(readiness, assessments)

    output_paths = {
        "requirements": output_dir / "market_label_source_requirements.csv",
        "inventory": output_dir / "local_market_source_inventory.csv",
        "assessments": output_dir / "market_label_candidate_assessment.csv",
        "acquisition_plan": output_dir / "market_label_acquisition_plan.csv",
        "readiness": output_dir / "market_label_readiness_checks.csv",
        "guard": output_dir / "no_label_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_source_report.md",
        "catalog": config.catalog_path,
        "changed_files": output_dir / "changed_files.txt",
        "self_check": output_dir / "self_check.csv",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(requirements, output_paths["requirements"])
    write_csv(inventory, output_paths["inventory"])
    write_csv(assessments, output_paths["assessments"])
    write_csv(acquisition_plan, output_paths["acquisition_plan"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    accepted_sources = int(assessments["accepted_for_market_label"].sum()) if not assessments.empty else 0
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "labels_not_produced",
                "status": "pass" if not guard["produced"].any() else "fail",
                "detail": "source audit only",
            },
            {
                "check": "blocked_when_no_accepted_source",
                "status": "pass"
                if accepted_sources > 0
                or str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
                else "fail",
                "detail": f"accepted_sources={accepted_sources}",
            },
            {
                "check": "raw_or_price_only_not_accepted",
                "status": "pass"
                if not assessments.loc[
                    assessments["decision"].isin(["rejected_price_only", "rejected_backtest_derived"]),
                    "accepted_for_market_label",
                ].any()
                else "fail",
                "detail": "rejected candidates have accepted_for_market_label=false",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["requirements"],
        output_paths["inventory"],
        output_paths["assessments"],
        output_paths["acquisition_plan"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "generated_at": now_text(),
        "audit_ready": True,
        "acceptance_pass": bool(acceptance["status"].eq("pass").all()),
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "candidate_files_audited": int(len(assessments)),
        "accepted_market_label_source_count": accepted_sources,
        "market_label_source_ready": bool(accepted_sources > 0),
        "labels_produced": False,
        "performance_validation_status": str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]),
        "data_decision": "market_label_source_ready" if accepted_sources else "market_label_source_blocked_no_accepted_total_return_source",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"]]],
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["acceptance_pass"] and manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
