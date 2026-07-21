#!/usr/bin/env python
"""Build V3.51 adjusted/PIT label-layer readiness artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adjusted_pit_label_layer import (
    LABEL_BUILD_STEPS,
    LABEL_SOURCE_REQUIREMENTS,
    LabelLayerConfig,
    build_label_schema_template,
    build_manual_request_queue,
    build_no_label_guard,
    build_readiness_checks,
    dataframe_from_rows,
    infer_required_label_scopes,
    label_path_status,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "adjusted_pit_label_layer_v3_51.json"
TASK_ID = "20260529_v3_51_adjusted_pit_label_layer"
VERSION = "V3.51"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


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


def optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def layer_config(config: dict[str, Any]) -> LabelLayerConfig:
    return LabelLayerConfig(
        horizons=tuple(int(item) for item in config["horizons"]),
        signal_asset_values=tuple(str(item) for item in config["signal_asset_values"]),
        min_required_signal_rows=int(config["min_required_signal_rows"]),
    )


def build_acceptance_checks(
    candidate_manifest: dict[str, Any],
    readiness: pd.DataFrame,
    no_label_guard: pd.DataFrame,
) -> pd.DataFrame:
    readiness_fail = readiness.loc[readiness["status"] == "fail"]
    labels_produced = bool(no_label_guard.loc[no_label_guard["result_type"] == "adjusted_pit_forward_return_labels", "produced"].iloc[0])
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    rows = [
        {
            "check": "v3_50_candidate_signal_panel_acceptance_passed",
            "status": "pass" if bool(candidate_manifest.get("acceptance_pass", False)) else "fail",
            "detail": candidate_manifest.get("data_decision", ""),
        },
        {
            "check": "label_readiness_has_no_failures",
            "status": "pass" if readiness_fail.empty else "fail",
            "detail": ";".join(f"{row.check}:{row.detail}" for row in readiness_fail.itertuples(index=False)),
        },
        {
            "check": "adjusted_labels_not_produced_without_sources",
            "status": "pass" if not labels_produced else "fail",
            "detail": "expected blocked until adjusted/PIT label path is supplied",
        },
        {
            "check": "performance_validation_blocked_without_labels",
            "status": "pass" if perf_status == "blocked" else "fail",
            "detail": perf_status,
        },
        {
            "check": "no_model_or_backtest_promotion",
            "status": "pass",
            "detail": "label readiness only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    signal_panel: pd.DataFrame,
    requirements: pd.DataFrame,
    required_scopes: pd.DataFrame,
    readiness: pd.DataFrame,
    no_label_guard: pd.DataFrame,
    acceptance_checks: pd.DataFrame,
) -> str:
    failed = acceptance_checks.loc[acceptance_checks["status"] != "pass"]
    blocked = readiness.loc[readiness["status"] == "blocked"]
    lines = [
        "# V3.51 Adjusted/PIT Label Layer",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.51 defines the adjusted/PIT forward-return label layer required before V3.49/V3.50 can perform real signal validation.",
        "- Current label production is blocked because no adjusted/PIT return label source has been supplied.",
        "- Raw `none_raw` daily close is explicitly rejected as a substitute for adjusted or total-return labels.",
        "- No IC, hit rate, return, NAV, drawdown, portfolio backtest, or model promotion is produced.",
        "",
        "## Signal Scope",
        "",
        f"- Signal rows needing labels: `{len(signal_panel)}`",
        f"- Signal assets: `{', '.join(sorted(signal_panel['asset'].astype(str).unique()))}`",
        f"- Signal date range: `{signal_panel['signal_date'].min()}` to `{signal_panel['signal_date'].max()}`",
        "",
        "## Required Label Scopes",
        "",
        "| scope | asset | signal_rows | required_dataset | status |",
        "|---|---|---:|---|---|",
    ]
    for row in required_scopes.itertuples(index=False):
        lines.append(f"| `{row.label_scope}` | `{row.asset}` | {int(row.signal_rows)} | `{row.required_dataset}` | `{row.status}` |")
    lines.extend(
        [
            "",
            "## Priority Sources",
            "",
            "| priority | scope | dataset | status | blocker |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in requirements.sort_values(["priority", "dataset_id"]).itertuples(index=False):
        lines.append(f"| {int(row.priority)} | `{row.label_scope}` | `{row.dataset_id}` | `{row.current_status}` | {str(row.blocker)[:160]} |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## No-Label Guard",
            "",
            "| result_type | produced | blocked | reason |",
            "|---|---|---|---|",
        ]
    )
    for row in no_label_guard.itertuples(index=False):
        lines.append(f"| `{row.result_type}` | `{row.produced}` | `{row.blocked}` | {str(row.reason)[:180]} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- Failed checks: `{len(failed)}`",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance_checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Supply a PIT market total-return label source for current `MARKET` signals, or point `existing_adjusted_pit_label_path` to a file satisfying the label contract.",
            "- For future stock-level signals, add `adj_factor`, all-status security master, and tradeability flags first.",
            "- After labels pass V3.51, rerun V3.49 with the label path and V3.50 signal panel.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame) -> str:
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return "\n".join(
        [
            "# A-share Adjusted/PIT Label Layer V3.51",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            "- Label layer contract accepted: `True`",
            f"- Performance validation status: `{perf_status}`",
            "- Current labels produced: `False`",
            "- Raw close substitution: `forbidden`",
            "",
            "## Boundary",
            "",
            "- Required before validation: PIT adjusted or total-return labels.",
            "- Current daily-only raw data remains unsuitable for label creation.",
            "- No model, return, or backtest output was produced.",
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], acceptance_checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    rows.append(
        {
            "check": "acceptance_checks_all_pass",
            "status": "pass" if bool((acceptance_checks["status"] == "pass").all()) else "fail",
            "detail": str(int((acceptance_checks["status"] != "pass").sum())),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]
    lcfg = layer_config(config)

    signal_panel = read_csv(ROOT / config["signal_panel_path"])
    candidate_manifest = read_json(ROOT / config["candidate_signal_manifest_path"])
    label_contract = read_csv(ROOT / config["label_contract_path"])
    requirements = dataframe_from_rows(LABEL_SOURCE_REQUIREMENTS)
    build_plan = dataframe_from_rows(LABEL_BUILD_STEPS)
    required_scopes = infer_required_label_scopes(signal_panel)
    label_template = build_label_schema_template(lcfg.horizons, lcfg.signal_asset_values)
    label_ready = label_path_status(optional_path(config.get("existing_adjusted_pit_label_path")))
    readiness = build_readiness_checks(signal_panel, requirements, label_ready, lcfg)
    no_label_guard = build_no_label_guard(label_ready)
    acceptance_checks = build_acceptance_checks(candidate_manifest, readiness, no_label_guard)
    manual_queue = build_manual_request_queue(requirements)

    artifacts = {
        "label_source_requirements": output_dir / "label_source_requirements.csv",
        "label_build_plan": output_dir / "label_build_plan.csv",
        "required_label_scopes": output_dir / "required_label_scopes.csv",
        "label_contract": output_dir / "adjusted_pit_label_contract.csv",
        "label_schema_template": output_dir / "label_schema_template.csv",
        "label_readiness_checks": output_dir / "label_readiness_checks.csv",
        "no_label_guard": output_dir / "no_label_guard.csv",
        "manual_data_request_queue": output_dir / "manual_data_request_queue.md",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "label_layer_report": output_dir / "label_layer_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(requirements, artifacts["label_source_requirements"])
    write_csv(build_plan, artifacts["label_build_plan"])
    write_csv(required_scopes, artifacts["required_label_scopes"])
    write_csv(label_contract, artifacts["label_contract"])
    write_csv(label_template, artifacts["label_schema_template"])
    write_csv(readiness, artifacts["label_readiness_checks"])
    write_csv(no_label_guard, artifacts["no_label_guard"])
    write_text(manual_queue, artifacts["manual_data_request_queue"])
    write_csv(acceptance_checks, artifacts["acceptance_checks"])
    write_text(build_report(signal_panel, requirements, required_scopes, readiness, no_label_guard, acceptance_checks), artifacts["label_layer_report"])
    write_text(build_catalog(readiness), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, acceptance_checks)
    write_csv(self_check, self_check_path)
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((acceptance_checks["status"] == "pass").all()),
        "label_layer_ready": True,
        "labels_produced": False,
        "performance_validation_status": perf_status,
        "signal_rows_needing_labels": int(signal_panel.shape[0]),
        "required_label_scope_count": int(required_scopes.shape[0]),
        "data_decision": "adjusted_pit_label_layer_contract_ready_labels_blocked",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
