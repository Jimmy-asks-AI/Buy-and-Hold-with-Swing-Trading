#!/usr/bin/env python
"""Run HIRSSM V3.64 narrow walk-forward proxy review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from narrow_walk_forward_proxy_review import (
    NarrowWalkForwardProxyConfig,
    build_acceptance_checks,
    build_candidate_decision,
    build_catalog,
    build_no_promotion_guard,
    build_oos_summary,
    build_report,
    build_survivor_panel,
    build_walk_forward_windows,
    select_survivors,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "narrow_walk_forward_proxy_review_v3_64.json"
TASK_ID = "20260529_v3_64_narrow_walk_forward_proxy_review"
VERSION = "V3.64"
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


def load_config(raw: dict[str, Any]) -> NarrowWalkForwardProxyConfig:
    return NarrowWalkForwardProxyConfig(
        joined_panel_path=resolve_path(raw["joined_panel_path"]),
        candidate_decision_path=resolve_path(raw["candidate_decision_path"]),
        v3_63_manifest_path=resolve_path(raw["v3_63_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        train_years=int(raw["walk_forward"]["train_years"]),
        test_years=int(raw["walk_forward"]["test_years"]),
        min_train_observations=int(raw["walk_forward"]["min_train_observations"]),
        min_test_observations=int(raw["walk_forward"]["min_test_observations"]),
        min_proxy_spearman=float(raw["gate_thresholds"]["min_proxy_spearman"]),
        min_proxy_qspread=float(raw["gate_thresholds"]["min_proxy_qspread"]),
        min_top_directional_alignment=float(raw["gate_thresholds"]["min_top_directional_alignment"]),
        min_gated_windows=int(raw["decision_thresholds"]["min_gated_windows"]),
        min_oos_pass_rate=float(raw["decision_thresholds"]["min_oos_pass_rate"]),
        min_oos_median_spearman=float(raw["decision_thresholds"]["min_oos_median_spearman"]),
        min_oos_positive_qspread_share=float(raw["decision_thresholds"]["min_oos_positive_qspread_share"]),
        top_quantile=float(raw["bucket_validation"]["top_quantile"]),
        bottom_quantile=float(raw["bucket_validation"]["bottom_quantile"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "narrow_walk_forward_proxy_review.py",
        ROOT / "strategy_lab" / "hirssm_v3_64_narrow_walk_forward_proxy_review.py",
        ROOT / "configs" / "narrow_walk_forward_proxy_review_v3_64.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_64_narrow_walk_forward_proxy_review.json",
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

    joined = pd.read_csv(config.joined_panel_path, encoding="utf-8-sig", low_memory=False)
    decisions_v3_63 = pd.read_csv(config.candidate_decision_path, encoding="utf-8-sig", low_memory=False)
    v3_63_manifest = read_json(config.v3_63_manifest_path)

    survivors = select_survivors(decisions_v3_63)
    survivor_panel = build_survivor_panel(joined, survivors)
    input_checks = validate_inputs(survivor_panel, survivors, v3_63_manifest, config)
    windows = build_walk_forward_windows(survivor_panel, config)
    oos_summary = build_oos_summary(windows, config)
    candidate_decision = build_candidate_decision(oos_summary)
    guard = build_no_promotion_guard(windows, candidate_decision)
    output_frames = [survivors, survivor_panel, input_checks, windows, oos_summary, candidate_decision, guard]
    acceptance = build_acceptance_checks(input_checks, windows, oos_summary, candidate_decision, guard, output_columns(output_frames))
    report = build_report(survivors, windows, oos_summary, candidate_decision, input_checks, acceptance)
    catalog = build_catalog(oos_summary, config)

    output_paths = {
        "survivors": output_dir / "v3_63_survivor_rows.csv",
        "survivor_panel": output_dir / "survivor_joined_proxy_panel.csv",
        "input_checks": output_dir / "walk_forward_input_checks.csv",
        "windows": output_dir / "walk_forward_window_results.csv",
        "summary": output_dir / "walk_forward_oos_summary.csv",
        "decisions": output_dir / "walk_forward_candidate_decision.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "narrow_walk_forward_proxy_review_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(survivors, output_paths["survivors"])
    write_csv(survivor_panel, output_paths["survivor_panel"])
    write_csv(input_checks, output_paths["input_checks"])
    write_csv(windows, output_paths["windows"])
    write_csv(oos_summary, output_paths["summary"])
    write_csv(candidate_decision, output_paths["decisions"])
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
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not candidate_decision["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"decision_rows={len(candidate_decision)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass"
                if not survivor_panel["official_total_return_evidence"].astype(bool).any()
                and not candidate_decision["official_total_return_evidence"].astype(bool).any()
                else "fail",
                "detail": "proxy walk-forward only",
            },
            {
                "check": "portfolio_or_model_promotion_blocked",
                "status": "pass"
                if not survivor_panel["portfolio_backtest_allowed"].astype(bool).any()
                and not candidate_decision["portfolio_backtest_allowed"].astype(bool).any()
                else "fail",
                "detail": "no NAV or model promotion",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["survivors"],
        output_paths["survivor_panel"],
        output_paths["input_checks"],
        output_paths["windows"],
        output_paths["summary"],
        output_paths["decisions"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    passed_count = int(oos_summary["walk_forward_proxy_review_status"].astype(str).eq("passes_narrow_proxy_walk_forward").sum()) if not oos_summary.empty else 0
    watch_count = int(oos_summary["walk_forward_proxy_review_status"].astype(str).eq("watchlist_or_repair").sum()) if not oos_summary.empty else 0
    train_gated = int(windows["train_gate_pass"].astype(bool).sum()) if not windows.empty else 0
    train_oos_pass = int(windows["train_gate_and_oos_pass"].astype(bool).sum()) if not windows.empty else 0
    metrics = {
        "v3_63_survivor_rows": int(len(survivors)),
        "survivor_panel_rows": int(len(survivor_panel)),
        "walk_forward_window_rows": int(len(windows)),
        "train_gated_window_rows": train_gated,
        "train_gate_and_oos_pass_rows": train_oos_pass,
        "signal_horizon_passes_narrow_proxy_walk_forward_rows": passed_count,
        "signal_horizon_watchlist_or_repair_rows": watch_count,
        "train_years": config.train_years,
        "test_years": config.test_years,
        "return_basis": "price_index_return",
        "validation_scope": "non_official_narrow_walk_forward_proxy_review",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    next_decision = (
        "V3.65 should run stricter label-source review for rows passing V3.64; still no portfolio promotion."
        if passed_count
        else "V3.65 should repair or retire the survivor rows because none passed narrow walk-forward proxy review."
    )

    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.63 lag-safe proxy validation",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_64_narrow_walk_forward_proxy_review.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.joined_panel_path),
            rel(config.candidate_decision_path),
            rel(config.v3_63_manifest_path),
        ],
        "code_refs": [
            "strategy_lab/narrow_walk_forward_proxy_review.py",
            "strategy_lab/hirssm_v3_64_narrow_walk_forward_proxy_review.py",
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
        "warn_count": watch_count,
        "limitations": [
            "Walk-forward review uses price-index proxy labels and excludes dividends.",
            "Passing rows are only candidates for stricter label-source review, not investable evidence.",
            "No portfolio backtest, NAV, model promotion, or default enablement is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_validation_not_total_return",
            "walk_forward_proxy_diagnostics_only",
            "limited_label_source_quality",
            "model_promotion_blocked",
        ],
        "next_decision": next_decision,
        "handoff_summary": "V3.63 survivors were reviewed with rolling five-year train and one-year OOS price-proxy diagnostics.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
