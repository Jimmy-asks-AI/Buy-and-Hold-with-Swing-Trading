#!/usr/bin/env python
"""Run HIRSSM V3.63 lag-safe proxy validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lag_safe_proxy_validation import (
    LagSafeProxyValidationConfig,
    build_acceptance_checks,
    build_candidate_decision,
    build_catalog,
    build_joined_panel,
    build_market_trend_audit,
    build_negative_control_summary,
    build_no_promotion_guard,
    build_readiness_checks,
    build_report,
    build_signal_validation_summary,
    build_state_dependence_audit,
    build_state_stratified_summary,
    build_temporal_artifact_audit,
    load_market_context,
    validate_proxy_labels,
    validate_repaired_signal_panel,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lag_safe_proxy_validation_v3_63.json"
TASK_ID = "20260529_v3_63_lag_safe_proxy_validation"
VERSION = "V3.63"
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


def load_config(raw: dict[str, Any]) -> LagSafeProxyValidationConfig:
    return LagSafeProxyValidationConfig(
        repaired_signal_panel_path=resolve_path(raw["repaired_signal_panel_path"]),
        price_proxy_label_path=resolve_path(raw["price_proxy_label_path"]),
        market_proxy_source_path=resolve_path(raw["market_proxy_source_path"]),
        v3_62_manifest_path=resolve_path(raw["v3_62_manifest_path"]),
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
        negative_control_shift=int(raw["artifact_screen"]["negative_control_shift"]),
        min_control_degradation=float(raw["artifact_screen"]["min_control_degradation"]),
        max_abs_same_day_corr=float(raw["artifact_screen"]["max_abs_same_day_corr"]),
        max_same_day_corr_ratio=float(raw["artifact_screen"]["max_same_day_corr_ratio"]),
        max_future_lead_corr_excess=float(raw["artifact_screen"]["max_future_lead_corr_excess"]),
        min_primary_state_support=int(raw["artifact_screen"]["min_primary_state_support"]),
        max_bull_state_share=float(raw["artifact_screen"]["max_bull_state_share"]),
        trend_window=int(raw["market_regime"]["trend_window"]),
        bull_return_threshold=float(raw["market_regime"]["bull_return_threshold"]),
        bear_return_threshold=float(raw["market_regime"]["bear_return_threshold"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "lag_safe_proxy_validation.py",
        ROOT / "strategy_lab" / "hirssm_v3_63_lag_safe_proxy_validation.py",
        ROOT / "configs" / "lag_safe_proxy_validation_v3_63.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_63_lag_safe_proxy_validation.json",
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

    repaired_signals = pd.read_csv(config.repaired_signal_panel_path, encoding="utf-8-sig", low_memory=False)
    labels = pd.read_csv(config.price_proxy_label_path, encoding="utf-8-sig", low_memory=False)
    market_source = pd.read_csv(config.market_proxy_source_path, encoding="utf-8-sig", low_memory=False)
    v3_62_manifest = read_json(config.v3_62_manifest_path)
    v3_59_manifest = read_json(config.v3_59_manifest_path)

    signal_checks = validate_repaired_signal_panel(repaired_signals, config)
    label_checks = validate_proxy_labels(labels, config)
    market_context = load_market_context(market_source, config)
    joined = build_joined_panel(repaired_signals, labels, market_context, config)
    signal_summary = build_signal_validation_summary(joined, config)
    state_summary = build_state_stratified_summary(joined, config)
    negative_control = build_negative_control_summary(joined, config)
    temporal = build_temporal_artifact_audit(joined, config)
    state_audit = build_state_dependence_audit(state_summary, config)
    trend_audit = build_market_trend_audit(joined, config)
    decisions = build_candidate_decision(signal_summary, temporal, negative_control, state_audit)
    readiness = build_readiness_checks(signal_checks, label_checks, joined, signal_summary, v3_62_manifest, v3_59_manifest, config)
    guard = build_no_promotion_guard(signal_summary, decisions)
    output_frames = [joined, signal_summary, state_summary, negative_control, temporal, state_audit, trend_audit, decisions, readiness, guard]
    acceptance = build_acceptance_checks(readiness, signal_summary, decisions, guard, output_columns(output_frames))
    report = build_report(joined, signal_summary, temporal, negative_control, state_audit, trend_audit, decisions, readiness, acceptance)
    catalog = build_catalog(signal_summary, decisions, config)

    output_paths = {
        "signal_checks": output_dir / "repaired_signal_contract_checks.csv",
        "label_checks": output_dir / "price_proxy_label_checks.csv",
        "joined": output_dir / "joined_lag_safe_proxy_panel.csv",
        "summary": output_dir / "lag_safe_proxy_validation_summary.csv",
        "state_summary": output_dir / "lag_safe_state_stratified_proxy_validation.csv",
        "negative_control": output_dir / "lag_safe_negative_control_summary.csv",
        "temporal": output_dir / "lag_safe_temporal_artifact_audit.csv",
        "state_audit": output_dir / "lag_safe_state_dependence_audit.csv",
        "trend_audit": output_dir / "lag_safe_market_trend_audit.csv",
        "decisions": output_dir / "candidate_validation_decision.csv",
        "readiness": output_dir / "lag_safe_proxy_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "lag_safe_proxy_validation_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(signal_checks, output_paths["signal_checks"])
    write_csv(label_checks, output_paths["label_checks"])
    write_csv(joined, output_paths["joined"])
    write_csv(signal_summary, output_paths["summary"])
    write_csv(state_summary, output_paths["state_summary"])
    write_csv(negative_control, output_paths["negative_control"])
    write_csv(temporal, output_paths["temporal"])
    write_csv(state_audit, output_paths["state_audit"])
    write_csv(trend_audit, output_paths["trend_audit"])
    write_csv(decisions, output_paths["decisions"])
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
                "check": "source_dates_strictly_lagged",
                "status": "pass"
                if not joined.empty
                and (pd.to_datetime(joined["source_trade_date"], format="%Y%m%d") < pd.to_datetime(joined["signal_date"], format="%Y%m%d")).all()
                else "fail",
                "detail": f"joined_rows={len(joined)}",
            },
            {
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not decisions["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"decision_rows={len(decisions)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass"
                if not joined["official_total_return_evidence"].astype(bool).any()
                and not decisions["official_total_return_evidence"].astype(bool).any()
                else "fail",
                "detail": "price proxy diagnostics only",
            },
            {
                "check": "portfolio_or_model_promotion_blocked",
                "status": "pass"
                if readiness.loc[readiness["check"].eq("portfolio_or_model_promotion_allowed_now"), "status"].eq("blocked").all()
                else "fail",
                "detail": "no NAV or model promotion",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["signal_checks"],
        output_paths["label_checks"],
        output_paths["joined"],
        output_paths["summary"],
        output_paths["state_summary"],
        output_paths["negative_control"],
        output_paths["temporal"],
        output_paths["state_audit"],
        output_paths["trend_audit"],
        output_paths["decisions"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    positive_count = int((signal_summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not signal_summary.empty else 0
    survivor_count = int(decisions["artifact_review_status"].astype(str).eq("survives_for_walk_forward_proxy_review").sum()) if not decisions.empty else 0
    blocked_count = int(decisions["artifact_review_status"].astype(str).eq("artifact_risk_blocks_escalation").sum()) if not decisions.empty else 0
    no_edge_count = int(decisions["artifact_review_status"].astype(str).eq("no_proxy_edge_observed").sum()) if not decisions.empty else 0
    metrics = {
        "joined_rows": int(len(joined)),
        "signal_horizon_rows": int(len(signal_summary)),
        "state_summary_rows": int(len(state_summary)),
        "proxy_positive_observation_rows": positive_count,
        "walk_forward_proxy_review_candidate_rows": survivor_count,
        "artifact_risk_blocks_escalation_rows": blocked_count,
        "no_proxy_edge_observed_rows": no_edge_count,
        "same_day_artifact_flag_rows": int(temporal["same_day_artifact_flag"].astype(bool).sum()) if not temporal.empty else 0,
        "future_signal_artifact_flag_rows": int(temporal["future_signal_artifact_flag"].astype(bool).sum()) if not temporal.empty else 0,
        "negative_control_artifact_flag_rows": int(negative_control["negative_control_artifact_flag"].astype(bool).sum()) if not negative_control.empty else 0,
        "bull_state_proxy_flag_rows": int(state_audit["bull_state_proxy_flag"].astype(bool).sum()) if not state_audit.empty else 0,
        "return_basis": "price_index_return",
        "validation_scope": "non_official_lag_safe_price_proxy_validation",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    if survivor_count > 0:
        next_decision = "V3.64 should run a narrow walk-forward proxy review for surviving rows only, still without portfolio promotion."
    else:
        next_decision = "V3.64 should redesign or retire weak lag-safe signals before any walk-forward review."

    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.62 lag-safe non-price signal repair",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_63_lag_safe_proxy_validation.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.repaired_signal_panel_path),
            rel(config.price_proxy_label_path),
            rel(config.market_proxy_source_path),
            rel(config.v3_62_manifest_path),
            rel(config.v3_59_manifest_path),
        ],
        "code_refs": [
            "strategy_lab/lag_safe_proxy_validation.py",
            "strategy_lab/hirssm_v3_63_lag_safe_proxy_validation.py",
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
        "warn_count": blocked_count + no_edge_count,
        "limitations": [
            "Validation uses price-index proxy labels and excludes dividends.",
            "Surviving rows, if any, are a proxy-review queue only, not investable evidence.",
            "No portfolio backtest, NAV, model promotion, or default enablement is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_validation_not_total_return",
            "lag_safe_proxy_diagnostics_only",
            "multiple_testing_risk",
            "model_promotion_blocked",
        ],
        "next_decision": next_decision,
        "handoff_summary": "Repaired lag-safe non-price signals were paired with price-proxy labels and screened for temporal, negative-control, state, and trend-regime artifacts.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
