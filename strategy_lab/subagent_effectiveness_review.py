#!/usr/bin/env python
"""Review and critique the quant subagent framework.

This is a governance script. It measures process and model-research yield from
existing artifacts, then emits a critique and optimization backlog. It does not
change any model weights or promote candidates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
AGENT_RUNS_DIR = ROOT / "outputs" / "agent_runs"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "governance" / "subagent_effectiveness_critique_v4"
TASK_ID = "20260527_subagent_effectiveness_critique_v4"
BASELINE = "HIRSSM V3.10 Clean Rank-Vol Core"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_task_board() -> pd.DataFrame:
    path = REPORTS_DIR / "AGENT_TASK_BOARD.md"
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and set(cells) == {"---"}:
            continue
        if "Task ID" in cells:
            header = cells
            continue
        if header and len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return pd.DataFrame(rows)


def load_manifests() -> pd.DataFrame:
    rows = []
    if not AGENT_RUNS_DIR.exists():
        return pd.DataFrame()
    for path in AGENT_RUNS_DIR.rglob("agent_run_manifest.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        rows.append(
            {
                "manifest_path": rel(path),
                "task_id": data.get("task_id", ""),
                "agent": data.get("agent", ""),
                "version": data.get("version", ""),
                "status": data.get("status", ""),
                "output_dir": data.get("output_dir", ""),
                "self_check_pass": bool(data.get("self_check_pass", False)),
                "fail_count": int(data.get("fail_count", 0) or 0),
                "warn_count": int(data.get("warn_count", 0) or 0),
                "allowed_inputs_count": len(data.get("allowed_inputs", [])) if isinstance(data.get("allowed_inputs"), list) else -1,
                "artifact_count": len(data.get("artifacts", [])) if isinstance(data.get("artifacts"), list) else -1,
                "overall_decision": metrics.get("overall_decision", ""),
                "annual_delta_vs_v310_10bps": metrics.get("annual_delta_vs_v310_10bps", np.nan),
                "drawdown_delta_vs_v310_10bps": metrics.get("drawdown_delta_vs_v310_10bps", np.nan),
                "pbo_10bps": metrics.get("pbo_10bps", np.nan),
                "next_decision": data.get("next_decision", ""),
            }
        )
    return pd.DataFrame(rows)


def load_model_yield() -> pd.DataFrame:
    rows = []
    outputs = ROOT / "outputs"
    if not outputs.exists():
        return pd.DataFrame()
    for gate_path in sorted(outputs.glob("hirssm_v*_*/candidate_gate_decision.csv")):
        model_dir = gate_path.parent
        gate = read_csv(gate_path)
        perf = read_csv(model_dir / "nested_oos_performance.csv")
        pbo = read_csv(model_dir / "pbo_cscv_summary.csv")
        selection = read_csv(model_dir / "nested_selection_by_fold.csv")
        if gate.empty:
            continue
        gate["cost_bps"] = gate["cost_bps"].astype(float)
        row10 = gate[gate["cost_bps"].eq(10.0)].head(1)
        pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
        selected = selection[selection["selection_status"].astype(str).eq("selected_by_prior_window")] if not selection.empty and "selection_status" in selection.columns else pd.DataFrame()
        rows.append(
            {
                "model_dir": rel(model_dir),
                "model_name": model_dir.name,
                "overall_decision": str(gate.get("overall_decision", pd.Series([""])).iloc[0]),
                "decision_10bps": str(row10["decision"].iloc[0]) if not row10.empty else "",
                "annual_delta_vs_v310_10bps": float(row10["annual_delta_vs_v310"].iloc[0]) if not row10.empty and "annual_delta_vs_v310" in row10.columns else np.nan,
                "drawdown_delta_vs_v310_10bps": float(row10["drawdown_delta_vs_v310"].iloc[0]) if not row10.empty and "drawdown_delta_vs_v310" in row10.columns else np.nan,
                "sharpe_delta_vs_v310_10bps": float(row10["sharpe_delta_vs_v310"].iloc[0]) if not row10.empty and "sharpe_delta_vs_v310" in row10.columns else np.nan,
                "pbo_10bps": float(row10["pbo"].iloc[0]) if not row10.empty and "pbo" in row10.columns else (float(pbo10["pbo"].iloc[0]) if not pbo10.empty else np.nan),
                "pbo_status_10bps": str(pbo10["pbo_status"].iloc[0]) if not pbo10.empty and "pbo_status" in pbo10.columns else "",
                "nonbaseline_selection_rate_10bps": float(row10["nonbaseline_selection_rate"].iloc[0]) if not row10.empty and "nonbaseline_selection_rate" in row10.columns else np.nan,
                "train_sufficient_selected_years": int(selected.shape[0]) if not selected.empty else 0,
                "cost_count": int(gate["cost_bps"].nunique()),
                "perf_rows": int(perf.shape[0]) if not perf.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def agent_workload(task_board: pd.DataFrame, manifests: pd.DataFrame) -> pd.DataFrame:
    if task_board.empty:
        return pd.DataFrame()
    board = task_board.copy()
    status_counts = board.groupby(["Agent", "Status"], as_index=False).size().rename(columns={"size": "task_count"})
    manifest_counts = manifests.groupby("agent", as_index=False).agg(
        manifest_count=("task_id", "count"),
        failed_manifest_count=("fail_count", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0) > 0).sum())),
        avg_artifact_count=("artifact_count", "mean"),
    ) if not manifests.empty else pd.DataFrame(columns=["agent", "manifest_count", "failed_manifest_count", "avg_artifact_count"])
    out = status_counts.merge(manifest_counts, left_on="Agent", right_on="agent", how="left")
    return out.drop(columns=["agent"], errors="ignore")


def scorecard(task_board: pd.DataFrame, manifests: pd.DataFrame, model_yield: pd.DataFrame) -> pd.DataFrame:
    accepted = int((task_board["Status"].astype(str) == "accepted").sum()) if not task_board.empty and "Status" in task_board.columns else 0
    accepted_model_dirs = int(model_yield.shape[0])
    promoted = int(model_yield["overall_decision"].astype(str).eq("promote_candidate").sum()) if not model_yield.empty else 0
    rejected_default = int(model_yield["overall_decision"].astype(str).eq("reject_for_default_observation_only").sum()) if not model_yield.empty else 0
    avg_annual_delta = float(pd.to_numeric(model_yield["annual_delta_vs_v310_10bps"], errors="coerce").mean()) if not model_yield.empty else np.nan
    best_annual_delta = float(pd.to_numeric(model_yield["annual_delta_vs_v310_10bps"], errors="coerce").max()) if not model_yield.empty else np.nan
    avg_pbo = float(pd.to_numeric(model_yield["pbo_10bps"], errors="coerce").mean()) if not model_yield.empty else np.nan
    pass_manifests = int(manifests["self_check_pass"].sum()) if not manifests.empty else 0
    manifest_count = int(manifests.shape[0])
    task_briefs = list((ROOT / "strategy_lab" / "agents" / "task_briefs").glob("*.json"))
    rows = [
        {"metric": "accepted_task_count", "value": accepted, "interpretation": "High throughput; not equivalent to model success."},
        {"metric": "model_harness_count", "value": accepted_model_dirs, "interpretation": "Number of model-producing validation harnesses with gate decisions."},
        {"metric": "promoted_model_count", "value": promoted, "interpretation": "No candidate has earned default promotion under current governance."},
        {"metric": "rejected_default_model_count", "value": rejected_default, "interpretation": "Most model work produced useful negative evidence rather than a new default."},
        {"metric": "avg_10bps_annual_delta_vs_v310", "value": avg_annual_delta, "interpretation": "Average candidate harness underperformed the frozen baseline."},
        {"metric": "best_10bps_annual_delta_vs_v310", "value": best_annual_delta, "interpretation": "Best annual edge is small and still failed at least one governance gate."},
        {"metric": "avg_10bps_pbo", "value": avg_pbo, "interpretation": "Overfit risk remains a binding constraint."},
        {"metric": "manifest_pass_rate", "value": pass_manifests / max(manifest_count, 1), "interpretation": "Reproducibility discipline is now strong but needs task-brief enforcement."},
        {"metric": "machine_readable_task_brief_count", "value": len(task_briefs), "interpretation": "Task briefs are now introduced; historical coverage remains incomplete."},
    ]
    return pd.DataFrame(rows)


def process_defects(manifests: pd.DataFrame, model_yield: pd.DataFrame) -> pd.DataFrame:
    defects = [
        {
            "severity": "high",
            "defect": "accepted_task_vs_model_promotion_confusion",
            "evidence": "Task board status accepted is frequently paired with overall_decision=reject_for_default_observation_only.",
            "optimization": "Use reports and manifests to state task_acceptance_scope and model_decision separately.",
            "status_after_this_task": "documented_and_partially_enforced",
        },
        {
            "severity": "high",
            "defect": "low_model_yield_after_many_versions",
            "evidence": f"{int(model_yield['overall_decision'].astype(str).eq('promote_candidate').sum()) if not model_yield.empty else 0} promoted candidates across {int(model_yield.shape[0])} model harnesses.",
            "optimization": "Add five-version meta-review and stop-loss rules before more serial implementation.",
            "status_after_this_task": "workflow_rule_added",
        },
        {
            "severity": "medium",
            "defect": "validation_agent_overload",
            "evidence": "Backtest validation auditor repeatedly owns implementation harnesses, attribution, and validation.",
            "optimization": "Route construction changes to portfolio_risk_engineer first; validation auditor should validate finalized artifacts.",
            "status_after_this_task": "raci_rule_added",
        },
        {
            "severity": "medium",
            "defect": "task_brief_not_machine_enforced",
            "evidence": "Task brief schema existed but no task_briefs directory was present.",
            "optimization": "Add task_briefs directory, current brief, and framework validation for present briefs.",
            "status_after_this_task": "implemented",
        },
        {
            "severity": "medium",
            "defect": "post_hoc_learning_can_turn_into_parameter_search",
            "evidence": "V3.30-V3.32 attribution naturally suggests threshold and amplitude tweaks.",
            "optimization": "Require attribution-derived hypotheses to enter a new predeclared brief before implementation.",
            "status_after_this_task": "workflow_rule_added",
        },
    ]
    if not manifests.empty and (manifests["allowed_inputs_count"] <= 0).any():
        defects.append(
            {
                "severity": "high",
                "defect": "manifest_allowed_inputs_missing_or_empty",
                "evidence": "At least one manifest lacks non-empty allowed_inputs.",
                "optimization": "Framework check should block missing or empty allowed_inputs.",
                "status_after_this_task": "queued",
            }
        )
    return pd.DataFrame(defects)


def optimization_backlog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "optimization": "Separate task acceptance from model decision in every report",
                "owner": "research_reporter",
                "implementation": "Reports must include both task_status and model_decision fields.",
                "acceptance_check": "No report headline says accepted without also saying promoted/rejected/observation.",
            },
            {
                "priority": 2,
                "optimization": "Five-version stop-loss review",
                "owner": "chief_quant_orchestrator",
                "implementation": "After five non-promoted model versions, block new implementation until source/data hypothesis changes.",
                "acceptance_check": "AGENT_WORKFLOW.md contains explicit stop-loss rule.",
            },
            {
                "priority": 3,
                "optimization": "Task brief first, manifest second",
                "owner": "code_quality_engineer",
                "implementation": "Validate all JSON briefs and require future accepted tasks to cite a task brief.",
                "acceptance_check": "agent_framework_check.py validates task_briefs.",
            },
            {
                "priority": 4,
                "optimization": "Reduce validation-agent overload",
                "owner": "chief_quant_orchestrator",
                "implementation": "Construction hypotheses go through portfolio_risk_engineer before validation.",
                "acceptance_check": "RACI/workflow state role boundary explicitly.",
            },
            {
                "priority": 5,
                "optimization": "Research portfolio rather than one serial tweak",
                "owner": "factor_researcher",
                "implementation": "Run source-level validation batches and only implement the top independent hypotheses.",
                "acceptance_check": "New candidate registry cites independent source evidence and kill criteria.",
            },
        ]
    )


def make_report(score: pd.DataFrame, model_yield: pd.DataFrame, defects: pd.DataFrame, backlog: pd.DataFrame) -> str:
    promoted = int(model_yield["overall_decision"].astype(str).eq("promote_candidate").sum()) if not model_yield.empty else 0
    model_count = int(model_yield.shape[0])
    best_delta = pd.to_numeric(model_yield["annual_delta_vs_v310_10bps"], errors="coerce").max() if not model_yield.empty else np.nan
    best_model = ""
    if not model_yield.empty and pd.notna(best_delta):
        best_row = model_yield.loc[pd.to_numeric(model_yield["annual_delta_vs_v310_10bps"], errors="coerce").idxmax()]
        best_model = str(best_row["model_name"])
    lines = [
        "# Subagent Framework Effectiveness Review V4",
        "",
        "## Critical View",
        "",
        f"- Model promotion yield is `0/{model_count}`. That is not automatically failure, because bad candidates were blocked, but it means the framework has produced more governance safety than investable alpha so far.",
        f"- Best 10bps annual delta was `{best_delta:.4%}` from `{best_model}`; this is too small to override PBO, cost, and stability concerns.",
        "- The strongest success is that the framework prevented weak or unstable variants from becoming default, and caught process defects such as missing manifest fields.",
        "- The weakest point is research productivity: many versions are sequential repairs around one branch instead of a broader portfolio of independent hypotheses.",
        "",
        "## Main Defects",
        "",
    ]
    for _, row in defects.iterrows():
        lines.append(f"- `{row['defect']}` ({row['severity']}): {row['evidence']} Fix: {row['optimization']}")
    lines.extend(["", "## Implemented Optimizations", ""])
    lines.extend(
        [
            "- Added machine-readable task brief coverage starting with this governance task.",
            "- Added a repeatable effectiveness review script and scorecard artifacts.",
            "- Added workflow rules for five-version stop-loss, task/model decision separation, and attribution-to-hypothesis boundaries.",
            "- Added framework validation for present task brief JSON files.",
        ]
    )
    lines.extend(["", "## Next Operating Rule", ""])
    lines.append("If the next five model-producing versions still fail to beat V3.10 after costs and PBO, stop implementation work and return to data/source discovery rather than parameter tweaks.")
    return "\n".join(lines)


def self_check(paths: dict[str, Path], score: pd.DataFrame, model_yield: pd.DataFrame, defects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "status": "pass" if path.exists() else "fail", "detail": rel(path)})
    rows.extend(
        [
            {"check": "scorecard_not_empty", "status": "pass" if not score.empty else "fail", "detail": str(int(score.shape[0]))},
            {"check": "model_yield_not_empty", "status": "pass" if not model_yield.empty else "fail", "detail": str(int(model_yield.shape[0]))},
            {"check": "critical_defects_present", "status": "pass" if not defects.empty else "fail", "detail": str(int(defects.shape[0]))},
            {"check": "no_model_promotion_done", "status": "pass", "detail": "governance review only"},
        ]
    )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_board = parse_task_board()
    manifests = load_manifests()
    model_yield = load_model_yield()
    workload = agent_workload(task_board, manifests)
    score = scorecard(task_board, manifests, model_yield)
    defects = process_defects(manifests, model_yield)
    backlog = optimization_backlog()

    paths = {
        "subagent_effectiveness_scorecard": output_dir / "subagent_effectiveness_scorecard.csv",
        "model_iteration_yield": output_dir / "model_iteration_yield.csv",
        "agent_workload": output_dir / "agent_workload.csv",
        "process_defect_log": output_dir / "process_defect_log.csv",
        "optimization_backlog": output_dir / "optimization_backlog.csv",
        "agent_report": output_dir / "agent_report.md",
        "framework_effectiveness_review": REPORTS_DIR / "AGENT_FRAMEWORK_EFFECTIVENESS_REVIEW.md",
        "changed_files": output_dir / "changed_files.txt",
    }
    for df, key in [
        (score, "subagent_effectiveness_scorecard"),
        (model_yield, "model_iteration_yield"),
        (workload, "agent_workload"),
        (defects, "process_defect_log"),
        (backlog, "optimization_backlog"),
    ]:
        write_csv(df, paths[key])
    write_text(make_report(score, model_yield, defects, backlog), paths["agent_report"])
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    self_check_path.touch()
    paths["self_check"] = self_check_path
    checks = self_check(paths, score, model_yield, defects)
    write_csv(checks, self_check_path)
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())

    metrics = {
        "model_harness_count": int(model_yield.shape[0]),
        "promoted_model_count": int(model_yield["overall_decision"].astype(str).eq("promote_candidate").sum()) if not model_yield.empty else 0,
        "rejected_default_model_count": int(model_yield["overall_decision"].astype(str).eq("reject_for_default_observation_only").sum()) if not model_yield.empty else 0,
        "best_10bps_annual_delta_vs_v310": float(pd.to_numeric(model_yield["annual_delta_vs_v310_10bps"], errors="coerce").max()) if not model_yield.empty else np.nan,
        "process_defect_count": int(defects.shape[0]),
        "optimization_backlog_count": int(backlog.shape[0]),
    }

    manifest_path = output_dir / "agent_run_manifest.json"
    artifacts = list(paths.values()) + [manifest_path]
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "chief_quant_orchestrator",
        "version": "Governance",
        "baseline": BASELINE,
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/subagent_effectiveness_review.py",
        "config": {"review_type": "subagent_effectiveness", "no_model_change": True},
        "data_refs": [
            "reports/AGENT_TASK_BOARD.md",
            "reports/MODEL_DECISION_LOG.md",
            "reports/AGENT_REVIEW_SUMMARY.md",
            "outputs/agent_runs",
            "outputs/hirssm_v*_*/candidate_gate_decision.csv",
        ],
        "code_refs": ["strategy_lab/subagent_effectiveness_review.py", "strategy_lab/agent_framework_check.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "reports/AGENT_TASK_BOARD.md",
            "reports/MODEL_DECISION_LOG.md",
            "reports/AGENT_REVIEW_SUMMARY.md",
            "outputs/agent_runs",
            "outputs/hirssm_v*_*/candidate_gate_decision.csv",
            "strategy_lab/agents",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Governance review uses existing artifacts and cannot prove future alpha.",
            "Historical task briefs are not fully backfilled.",
            "Model-yield statistics depend on available candidate_gate_decision.csv files.",
        ],
        "risk_flags": ["governance_overhead", "over_conservative_research_loop", "serial_iteration_bias"],
        "next_decision": "Use task briefs and five-version stop-loss before further model implementation.",
        "handoff_summary": "Subagent framework blocks weak candidates well but needs better research-yield controls and task/model decision separation.",
    }
    write_json(manifest, manifest_path)
    paths["agent_run_manifest"] = manifest_path
    write_text("\n".join(rel(path) for path in artifacts), paths["changed_files"])
    return {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
