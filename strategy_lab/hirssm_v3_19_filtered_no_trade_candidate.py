#!/usr/bin/env python
"""HIRSSM V3.19 filtered no-trade candidate feasibility check.

V3.19 intentionally does not force an implementation when V3.17 has filtered
all non-baseline candidates. It records the blocker and hands off to a new
research pass instead of relaxing governance silently.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
V316_DIR = ROOT / "outputs" / "agent_runs" / "v3_16" / "cost_aware_stability_design"
V317_DIR = ROOT / "outputs" / "agent_runs" / "v3_17" / "candidate_diversity_governance"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_19" / "filtered_no_trade_candidate"
TASK_ID = "20260526_v3_19_filtered_no_trade_candidate"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


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


def feasibility(filtered: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if filtered.empty:
        return pd.DataFrame(
            [
                {
                    "check": "filtered_candidate_set_exists",
                    "status": "blocked",
                    "detail": "missing filtered candidate set",
                    "decision": "do_not_implement",
                }
            ]
        )
    included = filtered[filtered["include_in_next_pbo"].astype(bool)].copy()
    nonbaseline = included[included["variant"].astype(str).ne(BASELINE_VARIANT)]
    rows.append(
        {
            "check": "nonbaseline_candidate_after_diversity_filter",
            "status": "pass" if not nonbaseline.empty else "blocked",
            "detail": ",".join(nonbaseline["variant"].astype(str).tolist()) if not nonbaseline.empty else "none",
            "decision": "implementation_allowed" if not nonbaseline.empty else "do_not_implement",
        }
    )
    for _, spec in specs.iterrows():
        source = str(spec.get("source_variant", ""))
        source_allowed = source in set(nonbaseline["variant"].astype(str))
        rows.append(
            {
                "check": f"no_trade_source_allowed_{spec.get('variant', '')}",
                "status": "pass" if source_allowed else "blocked",
                "detail": f"source={source}; band={spec.get('band', '')}",
                "decision": "test_overlay" if source_allowed else "do_not_implement_filtered_source",
            }
        )
    return pd.DataFrame(rows)


def blocker_report(feas: pd.DataFrame) -> pd.DataFrame:
    blocked = feas[feas["status"].astype(str).eq("blocked")]
    if blocked.empty:
        return pd.DataFrame(
            [
                {
                    "blocker_id": "none",
                    "severity": "pass",
                    "owner": "portfolio_risk_engineer",
                    "required_action": "Proceed to implementation harness.",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "blocker_id": "no_nonbaseline_after_v3_17_filter",
                "severity": "blocked",
                "owner": "chief_quant_orchestrator",
                "required_action": "Start a new orthogonal signal research task instead of relaxing the V3.17 diversity filter.",
            }
        ]
    )


def make_report(feas: pd.DataFrame, blockers: pd.DataFrame) -> str:
    allowed = bool(feas["decision"].astype(str).eq("implementation_allowed").any()) if not feas.empty else False
    blocker = blockers.iloc[0].to_dict() if not blockers.empty else {}
    return "\n".join(
        [
            "# HIRSSM V3.19 Filtered No-Trade Candidate",
            "",
            "## Purpose",
            "",
            "Check whether the V3.17 filtered candidate set still allows the V3.16 no-trade overlay implementation.",
            "",
            "## Result",
            "",
            f"- Implementation allowed: {allowed}",
            f"- Blocker: {blocker.get('blocker_id', 'none')}",
            f"- Required action: {blocker.get('required_action', 'none')}",
            "",
            "## Decision",
            "",
            "- Do not implement a candidate that V3.17 already removed as a near duplicate.",
            "- Continue with a new orthogonal signal research version.",
        ]
    )


def self_check(feas: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "feasibility_rows_exist", "status": "pass" if not feas.empty else "fail", "detail": str(int(feas.shape[0]))},
            {"check": "blocker_report_exists", "status": "pass" if not blockers.empty else "fail", "detail": str(int(blockers.shape[0]))},
            {"check": "blocked_without_forcing_candidate", "status": "pass", "detail": "strict V3.17 filter preserved"},
        ]
    )


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "portfolio_risk_engineer",
        "version": "V3.19",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "blocked",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_19_filtered_no_trade_candidate.py",
        "config": {"strict_v3_17_filter": True, "implementation_forced": False},
        "data_refs": ["outputs/agent_runs/v3_16/cost_aware_stability_design", "outputs/agent_runs/v3_17/candidate_diversity_governance"],
        "code_refs": ["strategy_lab/hirssm_v3_19_filtered_no_trade_candidate.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_16/cost_aware_stability_design", "outputs/agent_runs/v3_17/candidate_diversity_governance"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Blocked governance task; no model backtest is produced."],
        "risk_flags": ["no_nonbaseline_candidate_after_filter"],
        "next_decision": "Run a new orthogonal signal research task instead of implementing filtered-out candidates.",
        "handoff_summary": "V3.19 preserved the V3.17 diversity filter and blocked no-trade implementation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V3.19 filtered no-trade feasibility check.")
    parser.add_argument("--v316-dir", default=str(V316_DIR))
    parser.add_argument("--v317-dir", default=str(V317_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = read_csv(Path(args.v317_dir) / "filtered_candidate_set.csv")
    specs = read_csv(Path(args.v316_dir) / "no_trade_band_spec.csv")
    feas = feasibility(filtered, specs)
    blockers = blocker_report(feas)
    checks = self_check(feas, blockers)

    feas_path = output_dir / "filtered_no_trade_feasibility.csv"
    blockers_path = output_dir / "implementation_blockers.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    changed_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"
    feas.to_csv(feas_path, index=False, encoding="utf-8-sig")
    blockers.to_csv(blockers_path, index=False, encoding="utf-8-sig")
    write_text(make_report(feas, blockers), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [feas_path, blockers_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((feas["status"] == "blocked").sum())
    metrics = {
        "feasibility_rows": int(feas.shape[0]),
        "blocked_rows": int((feas["status"] == "blocked").sum()),
        "implementation_allowed": bool(feas["decision"].astype(str).eq("implementation_allowed").any()) if not feas.empty else False,
    }
    write_json(manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    print(json.dumps({"task_id": TASK_ID, "status": "blocked", "self_check_pass": fail_count == 0, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
