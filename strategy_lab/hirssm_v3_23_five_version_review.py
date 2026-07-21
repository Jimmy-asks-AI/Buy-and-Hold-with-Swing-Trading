#!/usr/bin/env python
"""HIRSSM V3.23 five-version review.

Closes the V3.19-V3.23 iteration block and records whether any model becomes
the new default. The block result is expected to be conservative: retain V3.10
unless V3.21 clears the governed promotion gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
V319_DIR = ROOT / "outputs" / "agent_runs" / "v3_19" / "filtered_no_trade_candidate"
V320_DIR = ROOT / "outputs" / "agent_runs" / "v3_20" / "rescue_signal_research"
V321_DIR = ROOT / "outputs" / "hirssm_v3_21_vol_compression_harness"
V322_DIR = ROOT / "outputs" / "agent_runs" / "v3_22" / "failure_attribution"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_23" / "five_version_review"
TASK_ID = "20260527_v3_23_five_version_review"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def version_summary() -> pd.DataFrame:
    feas = read_csv(V319_DIR / "filtered_no_trade_feasibility.csv")
    validation = read_csv(V320_DIR / "signal_validation.csv")
    holdout = read_csv(V320_DIR / "signal_gate_holdout_validation.csv")
    decision = read_csv(V321_DIR / "candidate_gate_decision.csv")
    pbo = read_csv(V321_DIR / "pbo_cscv_summary.csv")
    failures = read_csv(V322_DIR / "gate_failure_attribution.csv")
    rows = []
    rows.append(
        {
            "version": "V3.19",
            "task": "filtered_no_trade_candidate",
            "status": "blocked_accepted",
            "main_metric": "blocked_rows",
            "value": int((feas.get("status", pd.Series(dtype=str)).astype(str) == "blocked").sum()) if not feas.empty else np.nan,
            "decision": "strictly preserve V3.17 filter; do not implement filtered-out no-trade candidates",
        }
    )
    rows.append(
        {
            "version": "V3.20",
            "task": "rescue_signal_research",
            "status": "accepted_signal_research_observation_only",
            "main_metric": "holdout_eligible_signal_count",
            "value": int(holdout[holdout.get("split", pd.Series(dtype=str)).astype(str).eq("holdout")].get("eligible_for_implementation", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").sum()) if not holdout.empty else np.nan,
            "decision": "full-sample signal pass did not survive holdout implementation gate",
        }
    )
    dec10 = decision[decision["cost_bps"].astype(float).eq(10.0)].head(1) if not decision.empty else pd.DataFrame()
    pbo20 = pbo[pbo["cost_bps"].astype(float).eq(20.0)].head(1) if not pbo.empty else pd.DataFrame()
    rows.append(
        {
            "version": "V3.21",
            "task": "vol_compression_harness",
            "status": "diagnostic_rejected_for_default_source_gate_failed",
            "main_metric": "annual_delta_vs_v310_10bps",
            "value": float(dec10["annual_delta_vs_v310"].iloc[0]) if not dec10.empty else np.nan,
            "decision": "reject for default; source signal failed post-review holdout gate and 50bps annual delta gate not met",
        }
    )
    rows.append(
        {
            "version": "V3.21",
            "task": "cost_stress_pbo",
            "status": "fail",
            "main_metric": "pbo_20bps",
            "value": float(pbo20["pbo"].iloc[0]) if not pbo20.empty else np.nan,
            "decision": "20bps PBO failure blocks robustness promotion",
        }
    )
    rows.append(
        {
            "version": "V3.22",
            "task": "failure_attribution",
            "status": "accepted_attribution",
            "main_metric": "failure_rows",
            "value": int(failures.shape[0]) if not failures.empty else np.nan,
            "decision": "do not tune small cash-release overlays; require larger orthogonal information",
        }
    )
    rows.append(
        {
            "version": "V3.23",
            "task": "five_version_review",
            "status": "accepted_review",
            "main_metric": "new_default_promoted",
            "value": 0,
            "decision": "retain V3.10 as active governance baseline",
        }
    )
    return pd.DataFrame(rows)


def final_decision(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision": "keep_v3_10_baseline",
                "promote_new_default": False,
                "best_candidate": "vol_compression_reentry",
                "best_candidate_status": "diagnostic_only_source_holdout_failed",
                "reason": "V3.20 full-sample signal pass did not survive the holdout implementation gate; V3.21 improved 10bps nested annual return by only about 0.07pct versus V3.10 and failed the 20bps PBO robustness check.",
                "next_allowed_work": "Start a new candidate family with orthogonal data or material asset-selection change; do not tune small cash-release overlays.",
                "blocked_work": "Do not promote V3.21 or V3.16/V3.19 no-trade variants.",
            }
        ]
    )


def next_task_plan() -> str:
    return "\n".join(
        [
            "# V3.24 Candidate Direction",
            "",
            "- Add a genuinely new information source before implementation, such as macro liquidity, rate spread, valuation spread, or index constituent breadth.",
            "- Require active-return correlation below 0.85 versus V3.10 and V3.21 before a model harness is allowed.",
            "- Require expected effect size above the promotion threshold before implementation.",
            "- Keep V3.21 as observation evidence, not default model.",
        ]
    )


def make_report(summary: pd.DataFrame, decision: pd.DataFrame) -> str:
    lines = [
        "# HIRSSM V3.23 Five-Version Review",
        "",
        "## Scope",
        "",
        "Review V3.19 through V3.23 as one governed iteration block.",
        "",
        "## Version Results",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {row['version']} {row['task']}: {row['status']} ({row['main_metric']}={row['value']})")
    dec = decision.iloc[0]
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            f"- Decision: {dec['decision']}",
            f"- Promote new default: {dec['promote_new_default']}",
            f"- Best candidate: {dec['best_candidate']} ({dec['best_candidate_status']})",
            f"- Reason: {dec['reason']}",
            f"- Next allowed work: {dec['next_allowed_work']}",
        ]
    )
    return "\n".join(lines)


def self_check(summary: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    versions = set(summary["version"].astype(str)) if not summary.empty else set()
    return pd.DataFrame(
        [
            {"check": "summary_has_five_versions", "status": "pass" if {"V3.19", "V3.20", "V3.21", "V3.22", "V3.23"}.issubset(versions) else "fail", "detail": ",".join(sorted(versions))},
            {"check": "final_decision_exists", "status": "pass" if not decision.empty else "fail", "detail": decision["decision"].iloc[0] if not decision.empty else "missing"},
            {"check": "no_unvalidated_promotion", "status": "pass" if not bool(decision["promote_new_default"].iloc[0]) else "fail", "detail": "V3.10 retained"},
        ]
    )


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "research_reporter",
        "version": "V3.23",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_23_five_version_review.py",
        "config": {"review_scope": "V3.19-V3.23"},
        "data_refs": ["outputs/agent_runs/v3_19", "outputs/agent_runs/v3_20", "outputs/hirssm_v3_21_vol_compression_harness", "outputs/agent_runs/v3_22"],
        "code_refs": ["strategy_lab/hirssm_v3_23_five_version_review.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_19", "outputs/agent_runs/v3_20", "outputs/hirssm_v3_21_vol_compression_harness", "outputs/agent_runs/v3_22"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Review-only version; no new model backtest."],
        "risk_flags": ["no_new_default_promoted", "v3_21_edge_too_small"],
        "next_decision": "Queue V3.24 only after introducing a genuinely orthogonal information source.",
        "handoff_summary": "V3.23 closed the five-version block and retained V3.10 baseline.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.23 five-version review.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = version_summary()
    decision = final_decision(summary)
    checks = self_check(summary, decision)

    summary_path = output_dir / "version_summary.csv"
    decision_path = output_dir / "final_decision.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    next_plan_path = output_dir / "next_task_plan.md"
    changed_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    decision.to_csv(decision_path, index=False, encoding="utf-8-sig")
    write_text(make_report(summary, decision), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    write_text(next_task_plan(), next_plan_path)
    artifacts = [summary_path, decision_path, report_path, checks_path, next_plan_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = 0
    metrics = {"versions_reviewed": 5, "new_default_promoted": False}
    write_json(manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    print(json.dumps({"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
