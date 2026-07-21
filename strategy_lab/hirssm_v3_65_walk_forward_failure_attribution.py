#!/usr/bin/env python
"""Run HIRSSM V3.65 walk-forward failure attribution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from walk_forward_failure_attribution import (
    WalkForwardFailureAttributionConfig,
    build_acceptance_checks,
    build_catalog,
    build_no_promotion_guard,
    build_report,
    build_retire_repair_decisions,
    build_signal_horizon_failure_attribution,
    build_state_failure_attribution,
    build_train_oos_drift,
    build_year_regime_attribution,
    build_yearly_failure_attribution,
    normalize_inputs,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "walk_forward_failure_attribution_v3_65.json"
TASK_ID = "20260529_v3_65_walk_forward_failure_attribution"
VERSION = "V3.65"
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


def load_config(raw: dict[str, Any]) -> WalkForwardFailureAttributionConfig:
    return WalkForwardFailureAttributionConfig(
        window_results_path=resolve_path(raw["window_results_path"]),
        survivor_panel_path=resolve_path(raw["survivor_panel_path"]),
        oos_summary_path=resolve_path(raw["oos_summary_path"]),
        v3_64_manifest_path=resolve_path(raw["v3_64_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        broad_failure_pass_rate=float(raw["classification_thresholds"]["broad_failure_pass_rate"]),
        strong_proxy_pass_rate=float(raw["classification_thresholds"]["strong_proxy_pass_rate"]),
        long_horizons=tuple(int(x) for x in raw["classification_thresholds"]["long_horizons"]),
        retire_long_horizon_median_spearman=float(raw["classification_thresholds"]["retire_long_horizon_median_spearman"]),
        min_state_rows=int(raw["state_attribution"]["min_state_rows"]),
        top_quantile=float(raw["bucket_attribution"]["top_quantile"]),
        bottom_quantile=float(raw["bucket_attribution"]["bottom_quantile"]),
        state_min_spearman=float(raw["state_attribution"]["state_min_spearman"]),
        state_min_qspread=float(raw["state_attribution"]["state_min_qspread"]),
        state_min_top_alignment=float(raw["state_attribution"]["state_min_top_alignment"]),
        max_broad_failure_row_share_for_state_retest=float(raw["state_attribution"]["max_broad_failure_row_share_for_state_retest"]),
        min_strong_proxy_row_share_for_state_retest=float(raw["state_attribution"]["min_strong_proxy_row_share_for_state_retest"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    for frame in frames:
        columns.extend(str(col) for col in frame.columns)
    return columns


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "walk_forward_failure_attribution.py",
        ROOT / "strategy_lab" / "hirssm_v3_65_walk_forward_failure_attribution.py",
        ROOT / "configs" / "walk_forward_failure_attribution_v3_65.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_65_walk_forward_failure_attribution.json",
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

    windows = pd.read_csv(config.window_results_path, encoding="utf-8-sig", low_memory=False)
    panel = pd.read_csv(config.survivor_panel_path, encoding="utf-8-sig", low_memory=False)
    summary = pd.read_csv(config.oos_summary_path, encoding="utf-8-sig", low_memory=False)
    v3_64_manifest = read_json(config.v3_64_manifest_path)
    windows, panel, summary = normalize_inputs(windows, panel, summary)

    input_checks = validate_inputs(windows, panel, summary, v3_64_manifest)
    yearly = build_yearly_failure_attribution(windows, config)
    signal_attr = build_signal_horizon_failure_attribution(windows, summary, config)
    drift = build_train_oos_drift(windows)
    year_regime = build_year_regime_attribution(panel, yearly, config)
    state_attr = build_state_failure_attribution(panel, yearly, config)
    decisions = build_retire_repair_decisions(signal_attr, state_attr, yearly)
    guard = build_no_promotion_guard()
    output_frames = [input_checks, yearly, signal_attr, drift, year_regime, state_attr, decisions, guard]
    acceptance = build_acceptance_checks(input_checks, yearly, signal_attr, state_attr, decisions, guard, output_columns(output_frames))
    report = build_report(yearly, signal_attr, drift, state_attr, decisions, input_checks, acceptance)
    catalog = build_catalog(yearly, signal_attr, decisions, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "yearly": output_dir / "yearly_failure_attribution.csv",
        "signal_attr": output_dir / "signal_horizon_failure_attribution.csv",
        "drift": output_dir / "train_oos_drift.csv",
        "year_regime": output_dir / "year_regime_attribution.csv",
        "state_attr": output_dir / "state_failure_attribution.csv",
        "decisions": output_dir / "retire_repair_decisions.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "walk_forward_failure_attribution_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(yearly, output_paths["yearly"])
    write_csv(signal_attr, output_paths["signal_attr"])
    write_csv(drift, output_paths["drift"])
    write_csv(year_regime, output_paths["year_regime"])
    write_csv(state_attr, output_paths["state_attr"])
    write_csv(decisions, output_paths["decisions"])
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
                "check": "failure_attribution_outputs_present",
                "status": "pass" if not yearly.empty and not signal_attr.empty and not decisions.empty else "fail",
                "detail": f"yearly={len(yearly)};signal_attr={len(signal_attr)};decisions={len(decisions)}",
            },
            {
                "check": "promotion_guard_blocks_model_change",
                "status": "pass"
                if not guard.loc[guard["result_type"].isin(["investable_model_change", "portfolio_backtest"]), "produced"].astype(bool).any()
                else "fail",
                "detail": "diagnostic only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["yearly"],
        output_paths["signal_attr"],
        output_paths["drift"],
        output_paths["year_regime"],
        output_paths["state_attr"],
        output_paths["decisions"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    broad_years = int(yearly["year_failure_class"].eq("broad_failure_year").sum()) if not yearly.empty else 0
    partial_years = int(yearly["year_failure_class"].eq("partial_success_year").sum()) if not yearly.empty else 0
    strong_years = int(yearly["year_failure_class"].eq("strong_proxy_year").sum()) if not yearly.empty else 0
    retire_rows = int(decisions["final_action"].astype(str).str.startswith("retire").sum()) if not decisions.empty else 0
    state_retests = int(decisions["final_action"].eq("repair_with_predeclared_state_filter_only").sum()) if not decisions.empty else 0
    warn_count = int(input_checks["status"].eq("warn").sum())
    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "year_rows": int(len(yearly)),
        "signal_horizon_rows": int(len(signal_attr)),
        "state_attribution_rows": int(len(state_attr)),
        "year_regime_rows": int(len(year_regime)),
        "broad_failure_years": broad_years,
        "partial_success_years": partial_years,
        "strong_proxy_years": strong_years,
        "retire_signal_horizon_rows": retire_rows,
        "state_condition_retest_rows": state_retests,
        "proxy_oos_gate_pass_rows_from_v3_64": int(windows["train_gate_and_oos_pass"].sum()) if "train_gate_and_oos_pass" in windows else 0,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.64 narrow walk-forward proxy review",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_65_walk_forward_failure_attribution.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.window_results_path),
            rel(config.survivor_panel_path),
            rel(config.oos_summary_path),
            rel(config.v3_64_manifest_path),
        ],
        "code_refs": [
            "strategy_lab/walk_forward_failure_attribution.py",
            "strategy_lab/hirssm_v3_65_walk_forward_failure_attribution.py",
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
            "Attribution inherits V3.64 price-index proxy labels and excludes dividends.",
            "State-condition candidates are retest queues, not default-model evidence.",
            "No portfolio harness, performance series, or model change is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_only",
            "same_branch_repair_overfit_risk",
            "model_promotion_blocked",
        ],
        "next_decision": "V3.66 should avoid broad retuning; either run predeclared state-filter repair only or pivot to independent PIT source discovery.",
        "handoff_summary": "V3.65 decomposed V3.64 OOS proxy failures by year, signal horizon, state, and train-to-OOS drift.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
