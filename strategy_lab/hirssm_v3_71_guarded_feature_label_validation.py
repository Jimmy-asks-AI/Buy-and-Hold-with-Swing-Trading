#!/usr/bin/env python
"""Run HIRSSM V3.71 guarded feature-label validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from guarded_feature_label_validation import (
    GuardedFeatureLabelValidationConfig,
    build_acceptance_checks,
    build_candidate_decision,
    build_catalog,
    build_feature_horizon_summary,
    build_full_sample_feature_stats,
    build_multiple_testing_report,
    build_no_promotion_guard,
    build_report,
    build_source_family_summary,
    build_validation_universe,
    build_walk_forward_feature_results,
    select_validation_features,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "guarded_feature_label_validation_v3_71.json"
TASK_ID = "20260529_v3_71_guarded_feature_label_validation"
VERSION = "V3.71"
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


def load_config(raw: dict[str, Any]) -> GuardedFeatureLabelValidationConfig:
    wf = raw["walk_forward"]
    gates = raw["gate_thresholds"]
    return GuardedFeatureLabelValidationConfig(
        combined_manifest_path=resolve_path(raw["combined_manifest_path"]),
        combined_panel_path=resolve_path(raw["combined_panel_path"]),
        feature_registry_path=resolve_path(raw["feature_registry_path"]),
        label_path=resolve_path(raw["label_path"]),
        v3_59_manifest_path=resolve_path(raw["v3_59_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(x) for x in raw["horizons"]),
        train_years=int(wf["train_years"]),
        test_years=int(wf["test_years"]),
        min_full_sample_observations=int(gates["min_full_sample_observations"]),
        min_train_observations=int(gates["min_train_observations"]),
        min_test_observations=int(gates["min_test_observations"]),
        min_abs_train_spearman=float(gates["min_abs_train_spearman"]),
        min_abs_train_qspread=float(gates["min_abs_train_qspread"]),
        min_signed_oos_spearman=float(gates["min_signed_oos_spearman"]),
        min_signed_oos_qspread=float(gates["min_signed_oos_qspread"]),
        min_gated_windows=int(gates["min_gated_windows"]),
        min_oos_pass_rate=float(gates["min_oos_pass_rate"]),
        min_oos_median_signed_spearman=float(gates["min_oos_median_signed_spearman"]),
        min_oos_positive_qspread_share=float(gates["min_oos_positive_qspread_share"]),
        top_quantile=float(raw["bucket_validation"]["top_quantile"]),
        bottom_quantile=float(raw["bucket_validation"]["bottom_quantile"]),
        negative_control_shift=int(raw["negative_control"]["shift_rows"]),
        fdr_alpha=float(raw["multiple_testing"]["fdr_alpha"]),
        exclude_stale_macro_rows=bool(raw["stale_macro_policy"]["exclude_stale_macro_rows"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "guarded_feature_label_validation.py",
        ROOT / "strategy_lab" / "hirssm_v3_71_guarded_feature_label_validation.py",
        ROOT / "configs" / "guarded_feature_label_validation_v3_71.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_71_guarded_feature_label_validation.json",
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

    combined_manifest = read_json(config.combined_manifest_path)
    v3_59_manifest = read_json(config.v3_59_manifest_path)
    combined = read_csv(config.combined_panel_path)
    registry = read_csv(config.feature_registry_path)
    labels = read_csv(config.label_path)

    input_checks = validate_inputs(combined_manifest, v3_59_manifest, combined, registry, labels, config)
    features = select_validation_features(registry, combined)
    universe = build_validation_universe(combined, labels, config)
    full_stats = build_full_sample_feature_stats(universe, features, config)
    windows = build_walk_forward_feature_results(universe, features, config)
    summary = build_feature_horizon_summary(full_stats, windows, config)
    source_summary = build_source_family_summary(summary)
    candidate_decision = build_candidate_decision(summary)
    multiple_testing = build_multiple_testing_report(summary, config)
    guard = build_no_promotion_guard(summary)
    output_frames = [input_checks, features, universe, full_stats, windows, summary, source_summary, candidate_decision, multiple_testing, guard]
    acceptance = build_acceptance_checks(input_checks, universe, features, windows, summary, guard, output_columns(output_frames), config)
    report = build_report(universe, features, full_stats, windows, summary, source_summary, multiple_testing, input_checks, acceptance, guard)
    catalog = build_catalog(universe, features, summary, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "validation_features": output_dir / "validation_feature_registry.csv",
        "validation_universe": output_dir / "validation_universe.csv",
        "full_sample_stats": output_dir / "full_sample_feature_stats.csv",
        "walk_forward": output_dir / "walk_forward_feature_results.csv",
        "summary": output_dir / "feature_horizon_summary.csv",
        "source_summary": output_dir / "source_family_summary.csv",
        "candidate_decision": output_dir / "candidate_decision.csv",
        "multiple_testing": output_dir / "multiple_testing_report.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "guarded_feature_label_validation_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(features, output_paths["validation_features"])
    write_csv(
        universe.loc[
            :,
            [
                "signal_date",
                "horizon",
                "asset",
                "forward_price_index_return",
                "return_basis",
                "label_available_date",
                "combined_feature_validation_ready",
                "macro_any_stale_warning",
                "validation_scope",
                "official_total_return_evidence",
                "portfolio_backtest_allowed",
                "default_model_allowed",
            ],
        ],
        output_paths["validation_universe"],
    )
    write_csv(full_stats, output_paths["full_sample_stats"])
    write_csv(windows, output_paths["walk_forward"])
    write_csv(summary, output_paths["summary"])
    write_csv(source_summary, output_paths["source_summary"])
    write_csv(candidate_decision, output_paths["candidate_decision"])
    write_csv(multiple_testing, output_paths["multiple_testing"])
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
                "status": "pass" if candidate_decision.empty or not candidate_decision["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"decision_rows={len(candidate_decision)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass"
                if (summary.empty or not summary["official_total_return_evidence"].astype(bool).any())
                and not guard.loc[guard["result_type"].eq("official_total_return_validation"), "produced"].astype(bool).any()
                else "fail",
                "detail": "price-index proxy validation only",
            },
            {
                "check": "portfolio_or_model_promotion_blocked",
                "status": "pass"
                if not guard.loc[guard["result_type"].isin(["portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any()
                else "fail",
                "detail": "no portfolio harness or model promotion",
            },
            {
                "check": "macro_stale_policy_applied",
                "status": "pass" if config.exclude_stale_macro_rows else "fail",
                "detail": f"exclude_stale_macro_rows={config.exclude_stale_macro_rows}",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["validation_features"],
        output_paths["validation_universe"],
        output_paths["full_sample_stats"],
        output_paths["walk_forward"],
        output_paths["summary"],
        output_paths["source_summary"],
        output_paths["candidate_decision"],
        output_paths["multiple_testing"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    pass_count = int(summary["proxy_validation_status"].astype(str).eq("passes_proxy_walk_forward_for_stricter_review").sum()) if not summary.empty else 0
    gate_windows = int(windows["train_gate_pass"].astype(bool).sum()) if not windows.empty else 0
    oos_pass_windows = int(windows["train_gate_and_oos_pass"].astype(bool).sum()) if not windows.empty else 0
    fail_count = int(self_check["status"].eq("fail").sum())
    warn_count = 0
    metrics = {
        "validation_universe_rows": int(len(universe)),
        "validation_feature_rows": int(len(features)),
        "feature_horizon_summary_rows": int(len(summary)),
        "walk_forward_window_rows": int(len(windows)),
        "train_gated_window_rows": gate_windows,
        "train_gate_and_oos_pass_rows": oos_pass_windows,
        "proxy_pass_for_stricter_review_rows": pass_count,
        "source_family_count": int(source_summary["source_family"].nunique()) if not source_summary.empty else 0,
        "horizons": list(config.horizons),
        "return_basis": "price_index_return",
        "validation_scope": "non_official_price_proxy_guarded_feature_validation",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }

    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.70 combined feature registry plus V3.59 price-index proxy labels",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_71_guarded_feature_label_validation.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.combined_manifest_path),
            rel(config.combined_panel_path),
            rel(config.feature_registry_path),
            rel(config.label_path),
            rel(config.v3_59_manifest_path),
        ],
        "code_refs": [
            "strategy_lab/guarded_feature_label_validation.py",
            "strategy_lab/hirssm_v3_71_guarded_feature_label_validation.py",
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
            "V3.71 uses non-official price-index proxy labels, not official total-return labels.",
            "Proxy-positive feature rows are only eligible for stricter label-source review, not model promotion.",
            "No portfolio backtest, NAV, Sharpe, drawdown, annualized return, or trading cost estimate is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_label_only",
            "dividend_exclusion_bias",
            "multiple_testing_risk_controlled_by_bh_report_only",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Proceed to V3.72 stricter label-source review or feature clustering for V3.71 proxy-positive rows; do not move directly to portfolio harness.",
        "handoff_summary": "V3.71 ran guarded feature-label validation for V3.70 registered numeric features against V3.59 price-index proxy labels with walk-forward, negative-control, and no-promotion guards.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
