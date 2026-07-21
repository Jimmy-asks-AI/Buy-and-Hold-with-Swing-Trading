#!/usr/bin/env python
"""Run HIRSSM V3.53 MARKET total-return label importer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_total_return_label_importer import (
    MarketLabelImportConfig,
    build_acceptance_checks,
    build_catalog,
    build_forward_labels,
    build_import_readiness,
    build_no_label_guard,
    build_report,
    contract_frame,
    empty_labels,
    label_schema_frame,
    read_optional_csv,
    select_source_asset,
    source_template,
    source_validation_passed,
    validate_market_source,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_total_return_label_importer_v3_53.json"
TASK_ID = "20260529_v3_53_market_total_return_label_importer"
VERSION = "V3.53"


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


def load_config(raw: dict[str, Any]) -> MarketLabelImportConfig:
    return MarketLabelImportConfig(
        source_path=resolve_path(raw["source_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        market_proxy_codes=tuple(str(item) for item in raw["market_proxy_codes"]),
        source_asset_priority=tuple(str(item) for item in raw["source_asset_priority"]),
        min_source_rows=int(raw["min_source_rows"]),
        min_signal_coverage_ratio=float(raw["min_signal_coverage_ratio"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_total_return_label_importer.py",
        ROOT / "strategy_lab" / "hirssm_v3_53_market_total_return_label_importer.py",
        ROOT / "configs" / "market_total_return_label_importer_v3_53.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_53_market_total_return_label_importer.json",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    raw = read_json(config_path)
    config = load_config(raw)
    output_dir = config.output_dir
    signal_panel = pd.read_csv(config.signal_panel_path, encoding="utf-8-sig", low_memory=False)
    source_exists, source, source_error = read_optional_csv(config.source_path)

    selected_asset = ""
    selected_source = pd.DataFrame()
    selected_reason = "source_missing"
    if source_exists and source_error == "":
        validation_checks = validate_market_source(source, config)
        if source_validation_passed(validation_checks):
            selected_asset, selected_source, selected_reason = select_source_asset(source, config)
            if selected_source.empty:
                validation_checks = pd.concat(
                    [
                        validation_checks,
                        pd.DataFrame(
                            [
                                {
                                    "check": "source_asset_selected",
                                    "status": "fail",
                                    "detail": selected_reason,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        else:
            selected_reason = "source_validation_failed"
    else:
        validation_checks = pd.DataFrame(
            [
                {
                    "check": "source_file_available_for_validation",
                    "status": "blocked",
                    "detail": source_error,
                }
            ]
        )

    if source_validation_passed(validation_checks) and not selected_source.empty:
        labels, coverage = build_forward_labels(signal_panel, selected_source, config)
    else:
        labels, coverage = empty_labels(), pd.DataFrame()

    readiness = build_import_readiness(source_exists, source_error, validation_checks, labels, coverage, config)
    guard = build_no_label_guard(labels, readiness)
    acceptance = build_acceptance_checks(source_exists, validation_checks, labels, readiness, guard)
    report = build_report(config, source_exists, selected_asset, selected_reason, validation_checks, coverage, readiness, acceptance, labels)
    catalog = build_catalog(readiness, labels)

    output_paths = {
        "contract": output_dir / "market_total_return_source_contract.csv",
        "template": output_dir / "market_total_return_index.template.csv",
        "data_raw_template": ROOT / "data_raw" / "market_labels" / "market_total_return_index.template.csv",
        "label_schema": output_dir / "market_forward_label_schema.csv",
        "source_validation": output_dir / "source_validation_checks.csv",
        "coverage": output_dir / "market_label_coverage.csv",
        "readiness": output_dir / "market_label_import_readiness_checks.csv",
        "guard": output_dir / "no_label_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "labels": output_dir / "market_forward_labels.csv",
        "report": output_dir / "market_total_return_import_report.md",
        "catalog": config.catalog_path,
        "changed_files": output_dir / "changed_files.txt",
        "self_check": output_dir / "self_check.csv",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(contract_frame(), output_paths["contract"])
    write_csv(source_template(), output_paths["template"])
    write_csv(source_template(), output_paths["data_raw_template"])
    write_csv(label_schema_frame(), output_paths["label_schema"])
    write_csv(validation_checks, output_paths["source_validation"])
    write_csv(coverage, output_paths["coverage"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    if len(labels) > 0:
        write_csv(labels, output_paths["labels"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "missing_source_does_not_produce_labels",
                "status": "pass" if source_exists or len(labels) == 0 else "fail",
                "detail": f"source_exists={source_exists};label_rows={len(labels)}",
            },
            {
                "check": "performance_validation_status_consistent",
                "status": "pass" if perf_status == "pass" or len(labels) == 0 else "fail",
                "detail": f"performance_status={perf_status};label_rows={len(labels)}",
            },
            {
                "check": "required_artifact_contracts_written",
                "status": "pass"
                if output_paths["contract"].exists() and output_paths["template"].exists() and output_paths["label_schema"].exists()
                else "fail",
                "detail": "contract/template/label_schema",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["contract"],
        output_paths["template"],
        output_paths["data_raw_template"],
        output_paths["label_schema"],
        output_paths["source_validation"],
        output_paths["coverage"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    if len(labels) > 0:
        outputs_for_changed.append(output_paths["labels"])
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "generated_at": now_text(),
        "importer_ready": True,
        "source_path": rel(config.source_path),
        "source_exists": source_exists,
        "source_validation_passed": bool(source_validation_passed(validation_checks)),
        "selected_asset": selected_asset,
        "labels_produced": bool(len(labels) > 0),
        "label_rows": int(len(labels)),
        "performance_validation_status": perf_status,
        "acceptance_pass": bool(acceptance["status"].eq("pass").all()),
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "data_decision": "market_forward_labels_ready" if len(labels) > 0 else "market_total_return_source_missing_or_blocked",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"]]],
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["acceptance_pass"] and manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
