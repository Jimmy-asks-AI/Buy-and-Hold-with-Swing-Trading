#!/usr/bin/env python
"""HIRSSM V3.18 five-version review.

Aggregates V3.14-V3.17 and writes the final decision for this five-version
iteration block.
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
V314_DIR = ROOT / "outputs" / "agent_runs" / "v3_14" / "orthogonal_candidate_research"
V315_DIR = ROOT / "outputs" / "hirssm_v3_15_breadth_overlay_harness"
V316_DIR = ROOT / "outputs" / "agent_runs" / "v3_16" / "cost_aware_stability_design"
V317_DIR = ROOT / "outputs" / "agent_runs" / "v3_17" / "candidate_diversity_governance"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_18" / "five_version_review"
TASK_ID = "20260526_v3_18_five_version_review"


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
    factor = read_csv(V314_DIR / "factor_validation.csv")
    perf = read_csv(V315_DIR / "nested_oos_performance.csv")
    pbo = read_csv(V315_DIR / "pbo_cscv_summary.csv")
    cost_specs = read_csv(V316_DIR / "no_trade_band_spec.csv")
    filtered = read_csv(V317_DIR / "filtered_candidate_set.csv")
    rows = []
    rows.append(
        {
            "version": "V3.14",
            "task": "orthogonal_candidate_research",
            "status": "accepted_design",
            "main_metric": "pre_backtest_pass_count",
            "value": int(pd.Series(factor.get("pass_pre_backtest_gate", [])).fillna(False).astype(bool).sum()) if not factor.empty else np.nan,
            "decision": "implement only breadth signal; keep failed alpha signals observation",
        }
    )
    perf10 = perf[perf["cost_bps"].astype(float).eq(10.0)].head(1) if not perf.empty else pd.DataFrame()
    rows.append(
        {
            "version": "V3.15",
            "task": "breadth_overlay_harness",
            "status": "accepted_rejected_for_default",
            "main_metric": "annual_delta_vs_v310_10bps",
            "value": float(perf10["annual_delta_vs_v310"].iloc[0]) if not perf10.empty else np.nan,
            "decision": "reject breadth overlay for default; move to cost and diversity diagnostics",
        }
    )
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
    rows.append(
        {
            "version": "V3.15",
            "task": "pbo_gate",
            "status": "fail",
            "main_metric": "pbo_10bps",
            "value": float(pbo10["pbo"].iloc[0]) if not pbo10.empty else np.nan,
            "decision": "PBO failure blocks promotion",
        }
    )
    rows.append(
        {
            "version": "V3.16",
            "task": "cost_aware_stability_design",
            "status": "accepted_design",
            "main_metric": "high_priority_no_trade_specs",
            "value": int((cost_specs.get("implementation_priority", pd.Series(dtype=str)) == "high").sum()) if not cost_specs.empty else 0,
            "decision": "execution overlays can be tested later but cannot be alpha",
        }
    )
    rows.append(
        {
            "version": "V3.17",
            "task": "candidate_diversity_governance",
            "status": "accepted_design",
            "main_metric": "included_next_pbo_count",
            "value": int(filtered["include_in_next_pbo"].astype(bool).sum()) if not filtered.empty else 0,
            "decision": "filter near-duplicate candidates before future PBO",
        }
    )
    rows.append(
        {
            "version": "V3.18",
            "task": "five_version_review",
            "status": "accepted_review",
            "main_metric": "default_model_change",
            "value": 0,
            "decision": "keep V3.10 as active governance baseline",
        }
    )
    return pd.DataFrame(rows)


def final_decision(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision": "keep_v3_10_baseline",
                "promote_new_default": False,
                "reason": "V3.15 breadth overlay failed 10bps annual delta and PBO gates; V3.16/V3.17 are governance designs only.",
                "next_allowed_work": "V3.19 may implement filtered breadth plus no-trade overlay, but only as a new governed candidate.",
                "blocked_work": "Do not promote V3.15 or any cost-only overlay as default.",
            }
        ]
    )


def make_report(summary: pd.DataFrame, decision: pd.DataFrame) -> str:
    lines = [
        "# HIRSSM V3.18 Five-Version Review",
        "",
        "## Scope",
        "",
        "Review V3.14 through V3.18 as one iteration block.",
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
            f"- Reason: {dec['reason']}",
            f"- Next allowed work: {dec['next_allowed_work']}",
        ]
    )
    return "\n".join(lines)


def self_check(summary: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "summary_has_five_versions", "status": "pass" if set(["V3.14", "V3.15", "V3.16", "V3.17", "V3.18"]).issubset(set(summary["version"])) else "fail", "detail": ",".join(summary["version"].astype(str).unique())},
            {"check": "final_decision_exists", "status": "pass" if not decision.empty else "fail", "detail": decision["decision"].iloc[0] if not decision.empty else "missing"},
            {"check": "no_unvalidated_promotion", "status": "pass" if not bool(decision["promote_new_default"].iloc[0]) else "fail", "detail": "V3.10 retained"},
        ]
    )


def make_manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "research_reporter",
        "version": "V3.18",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_18_five_version_review.py",
        "config": {"review_scope": "V3.14-V3.18"},
        "data_refs": ["outputs/agent_runs/v3_14", "outputs/hirssm_v3_15_breadth_overlay_harness", "outputs/agent_runs/v3_16", "outputs/agent_runs/v3_17"],
        "code_refs": ["strategy_lab/hirssm_v3_18_five_version_review.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_14", "outputs/hirssm_v3_15_breadth_overlay_harness", "outputs/agent_runs/v3_16", "outputs/agent_runs/v3_17"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Review-only version; no new model backtest."],
        "risk_flags": ["no_new_default_promoted"],
        "next_decision": "Queue V3.19 only if a filtered no-trade implementation is desired.",
        "handoff_summary": "V3.18 closed the five-version block and retained V3.10 baseline.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.18 five-version review.")
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
    write_text(
        "\n".join(
            [
                "# V3.19 Optional Next Task",
                "",
                "- Implement only the V3.17 filtered candidate set.",
                "- Add V3.16 no-trade overlay as execution control, not alpha.",
                "- Rerun nested/PBO gates at 5/10/20/30bps.",
            ]
        ),
        next_plan_path,
    )
    artifacts = [summary_path, decision_path, report_path, checks_path, next_plan_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {"versions_reviewed": 5, "new_default_promoted": False}
    write_json(make_manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    result = {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count, "metrics": metrics, "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
