#!/usr/bin/env python
"""Run HIRSSM V3.60 guarded state-stratified proxy validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from state_stratified_proxy_validation import (
    ProxyValidationConfig,
    build_acceptance_checks,
    build_candidate_gate_decision,
    build_catalog,
    build_joined_panel,
    build_negative_control_summary,
    build_no_promotion_guard,
    build_readiness_checks,
    build_report,
    build_signal_validation_summary,
    build_state_coverage,
    build_state_stratified_summary,
    validate_proxy_labels,
    validate_signal_panel,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "state_stratified_proxy_validation_v3_60.json"
TASK_ID = "20260529_v3_60_state_stratified_proxy_validation"
VERSION = "V3.60"
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


def load_config(raw: dict[str, Any]) -> ProxyValidationConfig:
    return ProxyValidationConfig(
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        label_path=resolve_path(raw["label_path"]),
        v3_59_manifest_path=resolve_path(raw["v3_59_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        state_columns=tuple(str(item) for item in raw["state_columns"]),
        min_joined_rows=int(raw["minimums"]["joined_rows"]),
        min_signal_observations=int(raw["minimums"]["signal_observations"]),
        min_state_observations=int(raw["minimums"]["state_observations"]),
        top_quantile=float(raw["validation"]["top_quantile"]),
        bottom_quantile=float(raw["validation"]["bottom_quantile"]),
        min_abs_proxy_spearman=float(raw["validation"]["min_abs_proxy_spearman"]),
        min_proxy_qspread=float(raw["validation"]["min_proxy_qspread"]),
        min_top_directional_alignment=float(raw["validation"]["min_top_directional_alignment"]),
        negative_control_shift=int(raw["validation"]["negative_control_shift"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "state_stratified_proxy_validation.py",
        ROOT / "strategy_lab" / "hirssm_v3_60_state_stratified_proxy_validation.py",
        ROOT / "configs" / "state_stratified_proxy_validation_v3_60.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_60_state_stratified_proxy_validation.json",
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

    signals = pd.read_csv(config.signal_panel_path, encoding="utf-8-sig", low_memory=False)
    labels = pd.read_csv(config.label_path, encoding="utf-8-sig", low_memory=False)
    v3_59_manifest = read_json(config.v3_59_manifest_path)

    signal_checks = validate_signal_panel(signals, config)
    label_checks = validate_proxy_labels(labels, config)
    joined = build_joined_panel(signals, labels, config)
    signal_summary = build_signal_validation_summary(joined, config)
    state_coverage = build_state_coverage(joined, config)
    state_summary = build_state_stratified_summary(joined, config)
    controls = build_negative_control_summary(joined, config)
    gates = build_candidate_gate_decision(signal_summary)
    readiness = build_readiness_checks(signal_checks, label_checks, joined, signal_summary, v3_59_manifest, config)
    guard = build_no_promotion_guard(signal_summary)
    official_source = ROOT / "data_raw" / "market_labels" / "market_total_return_index.csv"
    output_frames = [joined, signal_summary, state_coverage, state_summary, controls, gates, readiness, guard]
    acceptance = build_acceptance_checks(readiness, signal_summary, gates, guard, official_source.exists(), output_columns(output_frames))
    report = build_report(joined, signal_summary, state_coverage, state_summary, controls, readiness, acceptance)
    catalog = build_catalog(config.label_path, signal_summary)

    output_paths = {
        "signal_checks": output_dir / "signal_contract_checks.csv",
        "label_checks": output_dir / "price_proxy_label_checks.csv",
        "joined": output_dir / "joined_proxy_validation_panel.csv",
        "signal_summary": output_dir / "signal_proxy_validation_summary.csv",
        "state_coverage": output_dir / "state_coverage.csv",
        "state_summary": output_dir / "state_stratified_proxy_validation.csv",
        "controls": output_dir / "negative_control_summary.csv",
        "gates": output_dir / "candidate_gate_decision.csv",
        "readiness": output_dir / "proxy_validation_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "state_stratified_proxy_validation_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(signal_checks, output_paths["signal_checks"])
    write_csv(label_checks, output_paths["label_checks"])
    write_csv(joined, output_paths["joined"])
    write_csv(signal_summary, output_paths["signal_summary"])
    write_csv(state_coverage, output_paths["state_coverage"])
    write_csv(state_summary, output_paths["state_summary"])
    write_csv(controls, output_paths["controls"])
    write_csv(gates, output_paths["gates"])
    write_csv(readiness, output_paths["readiness"])
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
                "check": "candidate_gates_do_not_promote",
                "status": "pass" if not gates["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"gate_rows={len(gates)}",
            },
            {
                "check": "official_total_return_source_not_created",
                "status": "pass" if not official_source.exists() else "fail",
                "detail": rel(official_source),
            },
            {
                "check": "portfolio_or_model_promotion_blocked",
                "status": "pass"
                if readiness.loc[
                    readiness["check"].isin(["official_total_return_validation_allowed_now", "portfolio_or_model_promotion_allowed_now"]),
                    "status",
                ].eq("blocked").all()
                else "fail",
                "detail": "proxy diagnostics only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["signal_checks"],
        output_paths["label_checks"],
        output_paths["joined"],
        output_paths["signal_summary"],
        output_paths["state_coverage"],
        output_paths["state_summary"],
        output_paths["controls"],
        output_paths["gates"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    positive_count = int((signal_summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not signal_summary.empty else 0
    metrics = {
        "joined_rows": int(len(joined)),
        "signal_summary_rows": int(len(signal_summary)),
        "state_summary_rows": int(len(state_summary)),
        "proxy_positive_observation_rows": positive_count,
        "default_model_allowed_rows": int(gates["default_model_allowed"].astype(bool).sum()) if not gates.empty else 0,
        "official_total_return_evidence_rows": int(signal_summary["official_total_return_evidence"].astype(bool).sum()) if not signal_summary.empty else 0,
        "return_basis": "price_index_return",
        "validation_scope": "non_official_price_proxy",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.59 MARKET price-proxy label importer",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -X utf8 strategy_lab/hirssm_v3_60_state_stratified_proxy_validation.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [rel(config.signal_panel_path), rel(config.label_path), rel(config.v3_59_manifest_path)],
        "code_refs": [
            "strategy_lab/state_stratified_proxy_validation.py",
            "strategy_lab/hirssm_v3_60_state_stratified_proxy_validation.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]],
        "changed_files": build_changed_files(outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "fail_count": int((self_check["status"] != "pass").sum()),
        "warn_count": 0,
        "limitations": [
            "Validation uses price-index proxy labels and excludes dividends.",
            "Results are not official total-return evidence.",
            "No portfolio backtest, NAV, model promotion, or default enablement is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_validation_not_total_return",
            "multiple_testing_risk",
            "state_definition_overfit_risk",
            "model_promotion_blocked",
        ],
        "next_decision": "V3.61 should review proxy-positive observations for economic plausibility and artifact risk before any portfolio harness is considered.",
        "handoff_summary": "State-stratified proxy validation completed with no official total-return claim and no model promotion.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
