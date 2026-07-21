#!/usr/bin/env python
"""Run HIRSSM V3.73 higher-quality label-source review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from higher_quality_label_source_review import (
    HigherQualityLabelSourceReviewConfig,
    build_acceptance_checks,
    build_catalog,
    build_label_source_inventory,
    build_no_label_guard,
    build_provider_route_review,
    build_report,
    build_required_source_contract,
    build_survivor_label_review_queue,
    inspect_target_source,
    select_strict_survivors,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "higher_quality_label_source_review_v3_73.json"
TASK_ID = "20260529_v3_73_higher_quality_label_source_review"
VERSION = "V3.73"
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


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> HigherQualityLabelSourceReviewConfig:
    return HigherQualityLabelSourceReviewConfig(
        v3_72_manifest_path=resolve_path(raw["v3_72_manifest_path"]),
        v3_72_decision_path=resolve_path(raw["v3_72_decision_path"]),
        v3_52_manifest_path=resolve_path(raw["v3_52_manifest_path"]),
        v3_52_candidate_assessment_path=resolve_path(raw["v3_52_candidate_assessment_path"]),
        v3_52_readiness_path=resolve_path(raw["v3_52_readiness_path"]),
        v3_53_manifest_path=resolve_path(raw["v3_53_manifest_path"]),
        v3_53_source_contract_path=resolve_path(raw["v3_53_source_contract_path"]),
        v3_53_readiness_path=resolve_path(raw["v3_53_readiness_path"]),
        v3_54_manifest_path=resolve_path(raw["v3_54_manifest_path"]),
        v3_54_acquisition_routes_path=resolve_path(raw["v3_54_acquisition_routes_path"]),
        v3_54_provider_readiness_path=resolve_path(raw["v3_54_provider_readiness_path"]),
        v3_54_readiness_path=resolve_path(raw["v3_54_readiness_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        template_source_path=resolve_path(raw["template_source_path"]),
        price_proxy_label_path=resolve_path(raw["price_proxy_label_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        strict_survivor_status=str(raw.get("strict_survivor_status", "strict_proxy_survivor_for_label_review")),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "higher_quality_label_source_review.py",
        ROOT / "strategy_lab" / "hirssm_v3_73_higher_quality_label_source_review.py",
        ROOT / "configs" / "higher_quality_label_source_review_v3_73.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_73_higher_quality_label_source_review.json",
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

    v3_72_manifest = read_json(config.v3_72_manifest_path)
    decisions = read_csv(config.v3_72_decision_path)
    v3_52_manifest = read_json(config.v3_52_manifest_path)
    v3_52_assessment = read_optional_csv(config.v3_52_candidate_assessment_path)
    v3_52_readiness = read_optional_csv(config.v3_52_readiness_path)
    v3_53_manifest = read_json(config.v3_53_manifest_path)
    contract_raw = read_optional_csv(config.v3_53_source_contract_path)
    v3_53_readiness = read_optional_csv(config.v3_53_readiness_path)
    v3_54_manifest = read_json(config.v3_54_manifest_path)
    routes_raw = read_optional_csv(config.v3_54_acquisition_routes_path)
    provider_readiness = read_optional_csv(config.v3_54_provider_readiness_path)
    v3_54_readiness = read_optional_csv(config.v3_54_readiness_path)

    input_checks = validate_inputs(
        v3_72_manifest,
        decisions,
        v3_52_manifest,
        v3_52_readiness,
        v3_53_manifest,
        v3_53_readiness,
        v3_54_manifest,
        routes_raw,
        provider_readiness,
        contract_raw,
        config,
    )
    survivors = select_strict_survivors(decisions, config)
    inventory = build_label_source_inventory(config, v3_52_manifest, v3_52_assessment, v3_53_manifest, v3_54_manifest, ROOT)
    provider_routes = build_provider_route_review(routes_raw, provider_readiness)
    source_contract = build_required_source_contract(contract_raw)
    target_state = inspect_target_source(config.target_source_path, v3_53_manifest)
    source_accepted = bool(target_state["higher_quality_label_source_accepted"])
    source_ready_for_v3_53 = bool(target_state["validation_ready_for_v3_53"])
    labels_produced = bool(target_state["labels_produced_by_v3_53"])
    queue = build_survivor_label_review_queue(survivors, source_accepted, source_ready_for_v3_53)
    guard = build_no_label_guard(source_accepted, labels_produced)
    acceptance = build_acceptance_checks(input_checks, inventory, provider_routes, source_contract, queue, guard, source_accepted)
    report = build_report(inventory, provider_routes, source_contract, queue, guard, acceptance, input_checks, config)
    catalog = build_catalog(inventory, queue, provider_routes, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "inventory": output_dir / "higher_quality_label_source_inventory.csv",
        "provider_routes": output_dir / "provider_route_review.csv",
        "source_contract": output_dir / "required_total_return_source_contract.csv",
        "queue": output_dir / "strict_survivor_label_review_queue.csv",
        "guard": output_dir / "no_label_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "higher_quality_label_source_review_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(inventory, output_paths["inventory"])
    write_csv(provider_routes, output_paths["provider_routes"])
    write_csv(source_contract, output_paths["source_contract"])
    write_csv(queue, output_paths["queue"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    forbidden_terms = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([input_checks, inventory, provider_routes, source_contract, queue, guard, acceptance])).lower())
    blocked_queue_rows = int(queue["higher_quality_label_review_status"].astype(str).str.startswith("blocked").sum()) if not queue.empty else 0
    ready_routes = int(provider_routes.get("route_ready", pd.Series(dtype=bool)).astype(bool).sum()) if not provider_routes.empty else 0
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "strict_survivors_loaded",
                "status": "pass" if len(queue) > 0 else "fail",
                "detail": f"strict_survivor_rows={len(queue)}",
            },
            {
                "check": "missing_source_blocks_queue",
                "status": "pass" if source_accepted or blocked_queue_rows == len(queue) else "fail",
                "detail": f"source_accepted={source_accepted};blocked_rows={blocked_queue_rows}",
            },
            {
                "check": "no_labels_or_portfolio_outputs",
                "status": "pass"
                if not bool(guard.loc[guard["result_type"].isin(["portfolio_backtest", "model_promotion"]), "produced"].any())
                else "fail",
                "detail": "governance review only",
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
        output_paths["input_checks"],
        output_paths["inventory"],
        output_paths["provider_routes"],
        output_paths["source_contract"],
        output_paths["queue"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    warn_count = int(input_checks["status"].eq("blocked").sum())
    metrics = {
        "strict_survivor_rows": int(len(queue)),
        "accepted_higher_quality_label_sources": int(inventory["higher_quality_label_source_accepted"].astype(bool).sum()) if not inventory.empty else 0,
        "target_source_exists": bool(config.target_source_path.exists()),
        "forward_total_return_labels_produced": labels_produced,
        "blocked_survivor_rows": blocked_queue_rows,
        "provider_ready_routes": ready_routes,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
        "next_required_file": rel(config.target_source_path),
    }

    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.72 strict proxy survivors plus V3.52/V3.53/V3.54 label-source governance",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_73_higher_quality_label_source_review.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/higher_quality_label_source_review.py",
            "strategy_lab/hirssm_v3_73_higher_quality_label_source_review.py",
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
            "No compliant source was acquired by this run.",
            "Strict survivors remain research candidates until V3.53 validates a higher-quality label source.",
            "Provider readiness comes from prior V3.54 outputs and is not refreshed live in V3.73.",
        ],
        "risk_flags": [
            "official_or_adjusted_market_label_source_missing",
            "proxy_label_survivors_blocked",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Provide data_raw/market_labels/market_total_return_index.csv, rerun V3.53, then run V3.74 official/higher-quality label validation for strict survivors.",
        "handoff_summary": "V3.73 converted V3.72 strict survivors into a blocked higher-quality label review queue and preserved the total-return source contract/acquisition route evidence.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
