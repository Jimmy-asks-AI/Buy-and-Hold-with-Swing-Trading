#!/usr/bin/env python
"""Run HIRSSM V3.75 manual label-source procurement package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from label_source_procurement_pack import (
    LabelSourceProcurementConfig,
    build_acceptance_checks,
    build_catalog,
    build_import_decision,
    build_no_promotion_guard,
    build_procurement_requirements,
    build_report,
    build_short_window_smoke_test,
    build_signal_coverage,
    build_source_contract,
    build_source_template,
    build_vendor_request_markdown,
    read_optional_csv,
    validate_source_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "label_source_procurement_pack_v3_75.json"
TASK_ID = "20260529_v3_75_label_source_procurement_pack"
VERSION = "V3.75"
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


def load_config(raw: dict[str, Any]) -> LabelSourceProcurementConfig:
    return LabelSourceProcurementConfig(
        v3_74_manifest_path=resolve_path(raw["v3_74_manifest_path"]),
        v3_74_manual_interfaces_path=resolve_path(raw["v3_74_manual_interfaces_path"]),
        v3_74_candidate_path=resolve_path(raw["v3_74_candidate_path"]),
        source_contract_path=resolve_path(raw["source_contract_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        required_start_date=str(raw["required_start_date"]),
        required_end_date=str(raw["required_end_date"]),
        min_source_rows=int(raw["min_source_rows"]),
        min_signal_coverage_ratio=float(raw["min_signal_coverage_ratio"]),
        horizons=tuple(int(x) for x in raw["horizons"]),
        accepted_source_bases=tuple(str(x) for x in raw["accepted_source_bases"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "label_source_procurement_pack.py",
        ROOT / "strategy_lab" / "hirssm_v3_75_label_source_procurement_pack.py",
        ROOT / "configs" / "label_source_procurement_pack_v3_75.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_75_label_source_procurement_pack.json",
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

    v3_74_manifest = read_json(config.v3_74_manifest_path)
    v3_74_manual = read_csv(config.v3_74_manual_interfaces_path)
    v3_74_candidate = read_optional_csv(config.v3_74_candidate_path)
    contract_raw = read_csv(config.source_contract_path)
    signal_panel = read_csv(config.signal_panel_path)
    target_source = read_optional_csv(config.target_source_path)
    target_exists = config.target_source_path.exists()

    requirements = build_procurement_requirements(config, v3_74_manifest)
    source_contract = build_source_contract(contract_raw)
    source_template = build_source_template(config)
    vendor_request = build_vendor_request_markdown(requirements, config)
    candidate_validation = validate_source_candidate(v3_74_candidate, config, "v3_74_joinquant_short_window_candidate")
    candidate_coverage = build_signal_coverage(v3_74_candidate, signal_panel, config, "v3_74_joinquant_short_window_candidate")
    target_validation = validate_source_candidate(target_source, config, "target_market_total_return_index_csv")
    target_coverage = build_signal_coverage(target_source, signal_panel, config, "target_market_total_return_index_csv")
    smoke = build_short_window_smoke_test(v3_74_candidate, candidate_validation, candidate_coverage)
    import_decision = build_import_decision(target_exists, target_validation, target_coverage, smoke, config)
    guard = build_no_promotion_guard(import_decision)
    acceptance = build_acceptance_checks(requirements, source_contract, candidate_validation, candidate_coverage, smoke, import_decision, guard)
    report = build_report(requirements, source_contract, candidate_validation, candidate_coverage, smoke, import_decision, acceptance, config)
    catalog = build_catalog(import_decision, smoke, config)

    output_paths = {
        "requirements": output_dir / "procurement_requirements.csv",
        "v3_74_manual": output_dir / "v3_74_manual_required_data_interfaces_snapshot.csv",
        "contract": output_dir / "import_source_contract.csv",
        "template": output_dir / "market_total_return_index.vendor_template.csv",
        "vendor_request": output_dir / "provider_request_template.md",
        "candidate_validation": output_dir / "v3_74_candidate_import_validation.csv",
        "candidate_coverage": output_dir / "v3_74_candidate_signal_coverage.csv",
        "target_validation": output_dir / "target_source_import_validation.csv",
        "target_coverage": output_dir / "target_source_signal_coverage.csv",
        "smoke": output_dir / "short_window_smoke_test.csv",
        "import_decision": output_dir / "import_decision.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "label_source_procurement_pack_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(requirements, output_paths["requirements"])
    write_csv(v3_74_manual, output_paths["v3_74_manual"])
    write_csv(source_contract, output_paths["contract"])
    write_csv(source_template, output_paths["template"])
    write_text(vendor_request, output_paths["vendor_request"])
    write_csv(candidate_validation, output_paths["candidate_validation"])
    write_csv(candidate_coverage, output_paths["candidate_coverage"])
    write_csv(target_validation, output_paths["target_validation"])
    write_csv(target_coverage, output_paths["target_coverage"])
    write_csv(smoke, output_paths["smoke"])
    write_csv(import_decision, output_paths["import_decision"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    forbidden_terms = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([requirements, source_contract, candidate_validation, candidate_coverage, target_validation, target_coverage, smoke, import_decision, guard, acceptance])).lower())
    ready_for_v3_53 = bool(import_decision["ready_for_v3_53"].iloc[0]) if not import_decision.empty else False
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "v3_74_manifest_passed",
                "status": "pass" if bool(v3_74_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_74_manifest.get('self_check_pass')};candidate_rows={v3_74_manifest.get('metrics', {}).get('joinquant_candidate_rows')}",
            },
            {
                "check": "provider_request_written",
                "status": "pass" if output_paths["vendor_request"].exists() else "fail",
                "detail": rel(output_paths["vendor_request"]),
            },
            {
                "check": "source_template_written",
                "status": "pass" if output_paths["template"].exists() else "fail",
                "detail": rel(output_paths["template"]),
            },
            {
                "check": "ready_for_v3_53_false_until_target_passes",
                "status": "pass" if not ready_for_v3_53 else "fail",
                "detail": str(import_decision["decision"].iloc[0]) if not import_decision.empty else "",
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
        output_paths["requirements"],
        output_paths["v3_74_manual"],
        output_paths["contract"],
        output_paths["template"],
        output_paths["vendor_request"],
        output_paths["candidate_validation"],
        output_paths["candidate_coverage"],
        output_paths["target_validation"],
        output_paths["target_coverage"],
        output_paths["smoke"],
        output_paths["import_decision"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    min_candidate_coverage = float(candidate_coverage["coverage_ratio"].min()) if not candidate_coverage.empty else 0.0
    metrics = {
        "v3_74_candidate_rows": int(len(v3_74_candidate)),
        "v3_74_candidate_min_coverage": min_candidate_coverage,
        "target_source_exists": target_exists,
        "ready_for_v3_53": ready_for_v3_53,
        "procurement_requirement_rows": int(len(requirements)),
        "provider_request_written": output_paths["vendor_request"].exists(),
        "source_template_written": output_paths["template"].exists(),
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.74 live label-source acquisition attempt",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_75_label_source_procurement_pack.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/label_source_procurement_pack.py",
            "strategy_lab/hirssm_v3_75_label_source_procurement_pack.py",
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
        "warn_count": int(candidate_validation["status"].eq("blocked").sum() + candidate_coverage["coverage_status"].eq("blocked").sum()),
        "limitations": [
            "V3.75 does not acquire new data; it creates procurement instructions and validators.",
            "V3.74 short-window data is a parser smoke test only and remains blocked for coverage/source-basis.",
            "V3.53 should be rerun only after target_source_exists and ready_for_v3_53 are true.",
        ],
        "risk_flags": [
            "official_source_missing",
            "short_window_smoke_not_evidence",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Deliver a compliant vendor/provider CSV at data_raw/market_labels/market_total_return_index.csv, rerun V3.75, then V3.53.",
        "handoff_summary": "V3.75 created the manual procurement request, source template, import validator outputs, and smoke-test evidence for the MARKET label source gap.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
