#!/usr/bin/env python
"""Run HIRSSM V3.61 proxy artifact-risk audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from proxy_artifact_audit import (
    ProxyArtifactAuditConfig,
    build_acceptance_checks,
    build_audit_panel,
    build_candidate_artifact_decision,
    build_catalog,
    build_negative_control_audit,
    build_no_promotion_guard,
    build_readiness_checks,
    build_regime_proxy_audit,
    build_report,
    build_state_dependence_audit,
    build_temporal_artifact_audit,
    load_market_context,
    select_proxy_positive_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "proxy_artifact_audit_v3_61.json"
TASK_ID = "20260529_v3_61_proxy_artifact_audit"
VERSION = "V3.61"
AGENT = "research_reporter"


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


def load_config(raw: dict[str, Any]) -> ProxyArtifactAuditConfig:
    return ProxyArtifactAuditConfig(
        joined_panel_path=resolve_path(raw["joined_panel_path"]),
        signal_summary_path=resolve_path(raw["signal_summary_path"]),
        state_summary_path=resolve_path(raw["state_summary_path"]),
        negative_control_path=resolve_path(raw["negative_control_path"]),
        market_proxy_source_path=resolve_path(raw["market_proxy_source_path"]),
        v3_60_manifest_path=resolve_path(raw["v3_60_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        positive_status=str(raw["candidate_selection"]["positive_status"]),
        min_abs_proxy_spearman=float(raw["audit_thresholds"]["min_abs_proxy_spearman"]),
        min_control_degradation=float(raw["audit_thresholds"]["min_control_degradation"]),
        max_same_day_corr_ratio=float(raw["audit_thresholds"]["max_same_day_corr_ratio"]),
        max_future_lead_corr_excess=float(raw["audit_thresholds"]["max_future_lead_corr_excess"]),
        min_primary_state_support=int(raw["audit_thresholds"]["min_primary_state_support"]),
        max_bull_state_share=float(raw["audit_thresholds"]["max_bull_state_share"]),
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
        ROOT / "strategy_lab" / "proxy_artifact_audit.py",
        ROOT / "strategy_lab" / "hirssm_v3_61_proxy_artifact_audit.py",
        ROOT / "configs" / "proxy_artifact_audit_v3_61.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_61_proxy_artifact_audit.json",
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
    signal_summary = pd.read_csv(config.signal_summary_path, encoding="utf-8-sig", low_memory=False)
    state_summary = pd.read_csv(config.state_summary_path, encoding="utf-8-sig", low_memory=False)
    negative_control = pd.read_csv(config.negative_control_path, encoding="utf-8-sig", low_memory=False)
    market_proxy_source = pd.read_csv(config.market_proxy_source_path, encoding="utf-8-sig", low_memory=False)
    v3_60_manifest = read_json(config.v3_60_manifest_path)

    candidates = select_proxy_positive_candidates(signal_summary, config)
    market_context = load_market_context(market_proxy_source, config)
    audit_panel = build_audit_panel(joined, market_context, candidates)
    readiness = build_readiness_checks(joined, signal_summary, negative_control, market_context, v3_60_manifest, candidates)
    temporal = build_temporal_artifact_audit(audit_panel, config)
    controls = build_negative_control_audit(negative_control, candidates, config)
    state_audit = build_state_dependence_audit(state_summary, candidates, config)
    regime_audit = build_regime_proxy_audit(audit_panel, candidates, config)
    decisions = build_candidate_artifact_decision(temporal, controls, state_audit, config)
    guard = build_no_promotion_guard(decisions)
    output_frames = [audit_panel, readiness, temporal, controls, state_audit, regime_audit, decisions, guard]
    acceptance = build_acceptance_checks(readiness, temporal, decisions, guard, output_columns(output_frames))
    report = build_report(candidates, temporal, controls, state_audit, regime_audit, decisions, readiness, acceptance)
    catalog = build_catalog(decisions, config)

    output_paths = {
        "candidates": output_dir / "proxy_positive_candidates.csv",
        "audit_panel": output_dir / "artifact_audit_panel.csv",
        "readiness": output_dir / "proxy_artifact_readiness_checks.csv",
        "temporal": output_dir / "temporal_artifact_audit.csv",
        "controls": output_dir / "negative_control_artifact_audit.csv",
        "state": output_dir / "state_dependence_artifact_audit.csv",
        "regime": output_dir / "market_trend_regime_audit.csv",
        "decisions": output_dir / "candidate_artifact_decision.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "proxy_artifact_audit_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(candidates, output_paths["candidates"])
    write_csv(audit_panel, output_paths["audit_panel"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(temporal, output_paths["temporal"])
    write_csv(controls, output_paths["controls"])
    write_csv(state_audit, output_paths["state"])
    write_csv(regime_audit, output_paths["regime"])
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
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not decisions["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"decision_rows={len(decisions)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass" if not decisions["official_total_return_evidence"].astype(bool).any() else "fail",
                "detail": "proxy artifact audit only",
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
        output_paths["candidates"],
        output_paths["audit_panel"],
        output_paths["readiness"],
        output_paths["temporal"],
        output_paths["controls"],
        output_paths["state"],
        output_paths["regime"],
        output_paths["decisions"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    plausible_count = int(decisions["artifact_review_status"].astype(str).eq("plausible_for_stricter_forward_validation").sum()) if not decisions.empty else 0
    blocked_count = int(decisions["artifact_review_status"].astype(str).eq("artifact_risk_blocks_escalation").sum()) if not decisions.empty else 0
    mixed_count = int(decisions["artifact_review_status"].astype(str).eq("mixed_evidence_requires_manual_review").sum()) if not decisions.empty else 0
    metrics = {
        "proxy_positive_candidates": int(len(candidates)),
        "artifact_decision_rows": int(len(decisions)),
        "plausible_for_stricter_forward_validation_rows": plausible_count,
        "artifact_risk_blocks_escalation_rows": blocked_count,
        "mixed_evidence_requires_manual_review_rows": mixed_count,
        "same_day_artifact_flag_rows": int(temporal["same_day_artifact_flag"].astype(bool).sum()) if not temporal.empty else 0,
        "future_signal_artifact_flag_rows": int(temporal["future_signal_artifact_flag"].astype(bool).sum()) if not temporal.empty else 0,
        "negative_control_artifact_flag_rows": int(controls["negative_control_artifact_flag"].astype(bool).sum()) if not controls.empty else 0,
        "bull_state_proxy_flag_rows": int(state_audit["bull_state_proxy_flag"].astype(bool).sum()) if not state_audit.empty else 0,
        "return_basis": "price_index_return",
        "validation_scope": "non_official_price_proxy_artifact_audit",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    if plausible_count > 0:
        next_decision = "V3.62 should carry only plausible rows into a narrow walk-forward proxy review; blocked rows need signal redesign or official label data."
    else:
        next_decision = "V3.62 should repair signal definitions with stricter lag discipline and new non-price proxy inputs; no row survived for walk-forward escalation."

    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.60 guarded state-stratified proxy validation",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_61_proxy_artifact_audit.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.joined_panel_path),
            rel(config.signal_summary_path),
            rel(config.state_summary_path),
            rel(config.negative_control_path),
            rel(config.market_proxy_source_path),
            rel(config.v3_60_manifest_path),
        ],
        "code_refs": [
            "strategy_lab/proxy_artifact_audit.py",
            "strategy_lab/hirssm_v3_61_proxy_artifact_audit.py",
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
        "warn_count": blocked_count + mixed_count,
        "limitations": [
            "Audit uses price-index proxy labels and excludes dividends.",
            "Artifact flags are diagnostics, not proof of factor invalidity.",
            "No portfolio backtest, NAV, model promotion, or default enablement is produced.",
        ],
        "risk_flags": [
            "price_index_proxy_validation_not_total_return",
            "artifact_risk_diagnostics_only",
            "negative_control_sensitivity",
            "model_promotion_blocked",
        ],
        "next_decision": next_decision,
        "handoff_summary": "Proxy-positive V3.60 observations were audited for same-day, future-signal, negative-control, and state-dependence artifact risk.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
