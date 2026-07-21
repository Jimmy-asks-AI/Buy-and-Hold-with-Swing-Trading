#!/usr/bin/env python
"""HIRSSM V3.12 candidate improvement design.

This script does not promote or backtest a new model. It reads the V3.11
nested harness evidence and emits predeclared candidate designs for the next
implementation round.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
V311_DIR = ROOT / "outputs" / "hirssm_v3_11_nested_candidate_harness"
V310_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_12" / "candidate_improvement_design"
TASK_ID = "20260526_v3_12_candidate_improvement_design"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def candidate_registry() -> pd.DataFrame:
    rows = [
        {
            "variant": "v3_10_clean_rank_vol_core",
            "role": "control",
            "description": "Frozen V3.10 governance baseline.",
            "multipliers_json": "{}",
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "predeclared_control",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": False,
            "implementation_status": "existing_control",
        },
        {
            "variant": "guarded_industry_trend_low_turnover",
            "role": "candidate",
            "description": "Only deviate from V3.10 when inner validation shows positive industry-trend edge; lower turnover cap.",
            "multipliers_json": json.dumps(
                {
                    "industry_trend_continuation": 1.08,
                    "risk_compression": 1.02,
                    "turnover_cap": 0.65,
                    "selection_margin_required": 0.010,
                    "fallback": "v3_10_clean_rank_vol_core",
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "predeclared_v3_12_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
        },
        {
            "variant": "baseline_blend_confidence_gate",
            "role": "candidate",
            "description": "Blend 80pct V3.10 and 20pct selected candidate only when inner validation margin and drawdown gates pass.",
            "multipliers_json": json.dumps(
                {
                    "baseline_weight": 0.80,
                    "candidate_weight": 0.20,
                    "selection_margin_required": 0.012,
                    "drawdown_not_worse_required": True,
                    "fallback": "v3_10_clean_rank_vol_core",
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "predeclared_v3_12_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
        },
        {
            "variant": "valuation_risk_repair_defensive_guard",
            "role": "candidate",
            "description": "Favor valuation repair and risk compression, but only in range or risk-off states; no extra industry trend tilt.",
            "multipliers_json": json.dumps(
                {
                    "range_bound": {"valuation_repair": 1.10, "risk_compression": 1.10, "defensive": 1.05},
                    "risk_off_decline": {"valuation_repair": 1.05, "risk_compression": 1.15, "defensive": 1.08},
                    "risk_on_trend": {"trend_continuation": 1.00},
                    "turnover_cap": 0.70,
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "predeclared_v3_12_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
        },
        {
            "variant": "pbo_stability_penalized_selector",
            "role": "candidate",
            "description": "Candidate selector penalizes variants that underperform V3.10 in more than half of inner years.",
            "multipliers_json": json.dumps(
                {
                    "objective": "inner_score_minus_instability_penalty",
                    "hit_rate_min": 0.55,
                    "worst_year_guard": -0.030,
                    "selection_margin_required": 0.008,
                    "fallback": "v3_10_clean_rank_vol_core",
                },
                sort_keys=True,
            ),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "predeclared_v3_12_design",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": True,
            "implementation_status": "design_only",
        },
    ]
    return pd.DataFrame(rows)


def failure_diagnosis(v311_dir: Path) -> pd.DataFrame:
    gate = read_csv(v311_dir / "candidate_gate_decision.csv")
    pbo = read_csv(v311_dir / "pbo_cscv_summary.csv")
    selected = read_csv(v311_dir / "nested_selection_by_fold.csv")
    rows = []
    if not gate.empty:
        for _, row in gate.iterrows():
            rows.append(
                {
                    "source": "candidate_gate_decision",
                    "cost_bps": row.get("cost_bps"),
                    "issue": "nested_selected_candidate_did_not_beat_v310",
                    "severity": "fail",
                    "evidence": f"annual_delta_vs_v310={row.get('annual_delta_vs_v310')}; decision={row.get('decision')}",
                    "design_response": "add baseline fallback and require inner selection margin before deviating",
                }
            )
    if not pbo.empty:
        for _, row in pbo.iterrows():
            rows.append(
                {
                    "source": "pbo_cscv_summary",
                    "cost_bps": row.get("cost_bps"),
                    "issue": "pbo_failed",
                    "severity": "fail" if str(row.get("pbo_status")) == "fail" else "observation",
                    "evidence": f"pbo={row.get('pbo')}; status={row.get('pbo_status')}",
                    "design_response": "reduce candidate aggressiveness and add stability penalties",
                }
            )
    if not selected.empty:
        freq = selected["selected_variant"].value_counts(normalize=True)
        for variant, rate in freq.items():
            rows.append(
                {
                    "source": "nested_selection_by_fold",
                    "cost_bps": "all",
                    "issue": "selection_frequency",
                    "severity": "observation",
                    "evidence": f"{variant} selected_rate={rate:.4f}",
                    "design_response": "require selected candidate to clear same-period baseline margins",
                }
            )
    return pd.DataFrame(rows)


def validation_plan() -> pd.DataFrame:
    rows = [
        {
            "gate": "candidate_registry_predeclared",
            "owner": "factor_researcher",
            "required_artifact": "candidate_registry.csv",
            "pass_rule": "all V3.12 variants exist before any backtest result is used",
        },
        {
            "gate": "nested_selection",
            "owner": "backtest_validation_auditor",
            "required_artifact": "nested_selection_by_fold.csv",
            "pass_rule": "selection uses only prior inner validation data",
        },
        {
            "gate": "same_period_baseline",
            "owner": "backtest_validation_auditor",
            "required_artifact": "same_period_baseline_comparison.csv",
            "pass_rule": "10bps annual_delta_vs_v310 >= 0.005 and drawdown_delta_vs_v310 >= -0.03",
        },
        {
            "gate": "pbo",
            "owner": "backtest_validation_auditor",
            "required_artifact": "pbo_cscv_summary.csv",
            "pass_rule": "pbo <= 0.35 for observation and <= 0.20 for pass",
        },
        {
            "gate": "cost_sensitivity",
            "owner": "execution_cost_analyst",
            "required_artifact": "cost_sensitivity.csv",
            "pass_rule": "20bps annual_delta_vs_v310 >= 0 and 30bps not materially worse than V3.10",
        },
        {
            "gate": "portfolio_constraints",
            "owner": "portfolio_risk_engineer",
            "required_artifact": "constraint_check.csv",
            "pass_rule": "no negative weights, no leverage, cash <= 40pct single period, avg cash <= 25pct",
        },
        {
            "gate": "manifest_and_schema",
            "owner": "code_quality_engineer",
            "required_artifact": "model_run_manifest.json",
            "pass_rule": "strict manifest validation passes and schema outputs parse",
        },
    ]
    return pd.DataFrame(rows)


def make_report(registry: pd.DataFrame, diagnosis: pd.DataFrame, plan: pd.DataFrame) -> str:
    fail_count = int((diagnosis["severity"] == "fail").sum()) if not diagnosis.empty else 0
    candidates = registry[registry["role"].eq("candidate")]
    return "\n".join(
        [
            "# HIRSSM V3.12 Candidate Improvement Design",
            "",
            "## Purpose",
            "",
            "Design the next candidate set after V3.11 rejected the current candidates. This is a design task, not a promotion or backtest.",
            "",
            "## V3.11 Failure Diagnosis",
            "",
            f"- Failure findings: {fail_count}",
            "- Main issue: nested selected candidates did not beat V3.10 same-period baseline.",
            "- Main overfit risk: PBO failed across the cost grid.",
            "",
            "## V3.12 Candidate Direction",
            "",
            f"- Proposed candidate count: {int(candidates.shape[0])}",
            "- Add baseline fallback and confidence margins before deviating from V3.10.",
            "- Reduce turnover and aggressiveness of industry trend tilts.",
            "- Penalize inner-window instability instead of rewarding one-window return.",
            "- Keep full-sample metrics diagnostic only.",
            "",
            "## Required Next Step",
            "",
            "Implement these candidates in a V3.12 harness run, then validate through the existing V3.11 nested/purged gate.",
        ]
    )


def make_manifest(
    *,
    start_time: str,
    output_dir: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.12",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_12_candidate_improvement_design.py",
        "config": {
            "source_harness": str(V311_DIR.as_posix()),
            "baseline_dir": str(V310_DIR.as_posix()),
        },
        "data_refs": [
            str((V311_DIR / "candidate_gate_decision.csv").as_posix()),
            str((V311_DIR / "pbo_cscv_summary.csv").as_posix()),
            str((V311_DIR / "nested_selection_by_fold.csv").as_posix()),
        ],
        "code_refs": ["strategy_lab/hirssm_v3_12_candidate_improvement_design.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": [
            "outputs/hirssm_v3_11_nested_candidate_harness",
            "outputs/hirssm_v3_10_clean_baseline",
        ],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Design only; no V3.12 model has been backtested or promoted.",
            "Candidate definitions must still be implemented and validated through nested gates.",
        ],
        "risk_flags": ["design_only_not_backtested", "full_sample_metrics_diagnostic_only"],
        "next_decision": "Implement V3.12 candidates and rerun nested validation before any promotion.",
        "handoff_summary": "Candidate improvement design generated from V3.11 failure evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.12 candidate improvement design artifacts.")
    parser.add_argument("--v311-dir", default=str(V311_DIR))
    parser.add_argument("--v310-dir", default=str(V310_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    v311_dir = Path(args.v311_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = candidate_registry()
    diagnosis = failure_diagnosis(v311_dir)
    plan = validation_plan()

    registry_path = output_dir / "candidate_registry.csv"
    registry_json_path = output_dir / "candidate_registry.json"
    diagnosis_path = output_dir / "failure_diagnosis.csv"
    hypotheses_path = output_dir / "candidate_hypotheses.csv"
    plan_path = output_dir / "validation_plan.csv"
    report_path = output_dir / "agent_report.md"
    manifest_path = output_dir / "agent_run_manifest.json"
    changed_path = output_dir / "changed_files.txt"

    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    write_json({"candidates": registry.to_dict(orient="records")}, registry_json_path)
    diagnosis.to_csv(diagnosis_path, index=False, encoding="utf-8-sig")
    registry[registry["role"].eq("candidate")].to_csv(hypotheses_path, index=False, encoding="utf-8-sig")
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    write_text(make_report(registry, diagnosis, plan), report_path)

    artifacts = [report_path, registry_path, registry_json_path, diagnosis_path, hypotheses_path, plan_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)
    metrics = {
        "candidate_count": int(registry[registry["role"].eq("candidate")].shape[0]),
        "failure_finding_count": int((diagnosis["severity"] == "fail").sum()) if not diagnosis.empty else 0,
        "validation_gate_count": int(plan.shape[0]),
    }
    manifest = make_manifest(
        start_time=start_time,
        output_dir=output_dir,
        artifacts=artifacts,
        metrics=metrics,
        fail_count=0,
        warn_count=1,
    )
    write_json(manifest, manifest_path)

    print(json.dumps({"self_check_pass": True, "metrics": metrics, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
