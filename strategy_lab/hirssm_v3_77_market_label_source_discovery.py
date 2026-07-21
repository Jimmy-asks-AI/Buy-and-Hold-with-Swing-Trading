#!/usr/bin/env python
"""Run HIRSSM V3.77 independent MARKET label-source discovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_source_discovery import (
    SourceDiscoveryConfig,
    build_acceptance_checks,
    build_acquisition_route_decision,
    build_catalog,
    build_due_diligence_matrix,
    build_evidence_sources,
    build_no_execution_guard,
    build_provider_questionnaire,
    build_report,
    build_source_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_source_discovery_v3_77.json"
TASK_ID = "20260529_v3_77_market_label_source_discovery"
VERSION = "V3.77"
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


def load_config(raw: dict[str, Any]) -> SourceDiscoveryConfig:
    return SourceDiscoveryConfig(
        v3_76_manifest_path=resolve_path(raw["v3_76_manifest_path"]),
        v3_76_next_commands_path=resolve_path(raw["v3_76_next_commands_path"]),
        v3_75_requirements_path=resolve_path(raw["v3_75_requirements_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        required_start_date=str(raw["required_start_date"]),
        required_end_date=str(raw["required_end_date"]),
        source_candidates=tuple(dict(item) for item in raw["source_candidates"]),
        score_weights=dict(raw["score_weights"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_source_discovery.py",
        ROOT / "strategy_lab" / "hirssm_v3_77_market_label_source_discovery.py",
        ROOT / "configs" / "market_label_source_discovery_v3_77.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_77_market_label_source_discovery.json",
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

    v3_76_manifest = read_json(config.v3_76_manifest_path)
    v3_76_next_commands = read_csv(config.v3_76_next_commands_path)
    v3_75_requirements = read_csv(config.v3_75_requirements_path)

    candidates = build_source_candidates(config)
    due_diligence = build_due_diligence_matrix(candidates, config)
    route_decision = build_acquisition_route_decision(candidates, config)
    questionnaire = build_provider_questionnaire(candidates, config)
    evidence_sources = build_evidence_sources(candidates)
    guard = build_no_execution_guard(config)
    acceptance = build_acceptance_checks(candidates, due_diligence, route_decision, guard)
    report = build_report(candidates, due_diligence, route_decision, acceptance, config)
    catalog = build_catalog(candidates, route_decision, config)

    output_paths = {
        "candidates": output_dir / "source_candidates.csv",
        "due_diligence": output_dir / "source_due_diligence_matrix.csv",
        "route_decision": output_dir / "acquisition_route_decision.csv",
        "questionnaire": output_dir / "provider_questionnaire.md",
        "evidence": output_dir / "source_evidence_register.md",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_source_discovery_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(candidates, output_paths["candidates"])
    write_csv(due_diligence, output_paths["due_diligence"])
    write_csv(route_decision, output_paths["route_decision"])
    write_text(questionnaire, output_paths["questionnaire"])
    write_text(evidence_sources, output_paths["evidence"])
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
        if term in " ".join(output_columns([candidates, due_diligence, route_decision, guard, acceptance])).lower()
    )
    target_was_written = config.target_source_path.exists() and not bool(v3_76_manifest.get("metrics", {}).get("target_source_exists", False))
    active_procurement = int(route_decision["status"].astype(str).eq("active").sum())
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "v3_76_manifest_passed",
                "status": "pass" if bool(v3_76_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_76_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_76_is_blocked_input_state",
                "status": "pass" if not bool(v3_76_manifest.get("metrics", {}).get("may_execute_v3_53_now", True)) else "fail",
                "detail": f"may_execute_v3_53_now={v3_76_manifest.get('metrics', {}).get('may_execute_v3_53_now')}",
            },
            {
                "check": "target_csv_not_written_by_v3_77",
                "status": "pass" if not target_was_written else "fail",
                "detail": f"target_exists_now={config.target_source_path.exists()}",
            },
            {
                "check": "at_least_one_active_procurement_step",
                "status": "pass" if active_procurement >= 1 else "fail",
                "detail": f"active_steps={active_procurement}",
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
        output_paths["candidates"],
        output_paths["due_diligence"],
        output_paths["route_decision"],
        output_paths["questionnaire"],
        output_paths["evidence"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    top_source = candidates.iloc[0].to_dict() if not candidates.empty else {}
    procurement_ready = int(candidates["decision"].isin(["primary_procurement_route", "secondary_procurement_route"]).sum())
    price_only_rejected = int(candidates["decision"].astype(str).eq("reject_as_final_label_price_only").sum())
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "candidate_source_count": int(len(candidates)),
        "procurement_ready_source_count": procurement_ready,
        "price_only_rejected_count": price_only_rejected,
        "top_source_id": top_source.get("source_id", ""),
        "top_source_decision": top_source.get("decision", ""),
        "active_route_steps": active_procurement,
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
        "baseline": "V3.76 intake orchestrator blocked state",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_77_market_label_source_discovery.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_source_discovery.py",
            "strategy_lab/hirssm_v3_77_market_label_source_discovery.py",
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
        "warn_count": int(route_decision["status"].astype(str).isin(["blocked", "pending"]).sum()),
        "limitations": [
            "V3.77 ranks acquisition routes but does not fetch or license data.",
            "Official source readiness still depends on external delivery and V3.75 contract validation.",
            "Public web and price-only sources are not accepted as final labels.",
            f"V3.76 input had {len(v3_76_next_commands)} next-command rows and V3.75 had {len(v3_75_requirements)} procurement requirement rows.",
        ],
        "risk_flags": [
            "target_source_missing" if not config.target_source_path.exists() else "target_source_present_needs_revalidation",
            "external_license_required",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Contact primary and secondary procurement routes, collect a licensed sample CSV, then rerun V3.75 and V3.76 before V3.53.",
        "handoff_summary": "V3.77 created an independent data-source discovery table, due-diligence matrix, provider questionnaire, and guarded acquisition route decision.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
