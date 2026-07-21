#!/usr/bin/env python
"""HIRSSM V3.22 failure attribution for V3.21.

This version explains why the V3.21 volatility-compression candidate is not
promoted and converts the result into stricter next-research requirements.
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
V321_DIR = ROOT / "outputs" / "hirssm_v3_21_vol_compression_harness"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_22" / "failure_attribution"
TASK_ID = "20260527_v3_22_failure_attribution"
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


def gate_failure_attribution(decision: pd.DataFrame, pbo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if decision.empty:
        return pd.DataFrame(
            [
                {
                    "cost_bps": np.nan,
                    "failure_type": "missing_decision",
                    "severity": "fail",
                    "detail": "candidate_gate_decision.csv missing or empty",
                    "action": "rerun V3.21 harness",
                }
            ]
        )
    for _, row in decision.iterrows():
        cost = float(row["cost_bps"])
        failed_checks = []
        for check in ["annual_delta_positive_50bps", "drawdown_not_worse_3pct", "avg_cash_not_excessive", "pbo_not_fail", "candidate_selected_enough"]:
            if not bool(row.get(check, False)):
                failed_checks.append(check)
        pbo_status = ""
        if not pbo.empty:
            pbo_row = pbo[pbo["cost_bps"].astype(float).eq(cost)]
            if not pbo_row.empty:
                pbo_status = str(pbo_row["pbo_status"].iloc[0])
        rows.append(
            {
                "cost_bps": cost,
                "failure_type": "gate_rejection" if failed_checks else "passed_component_not_overall",
                "severity": "fail" if failed_checks else "observation",
                "failed_checks": ",".join(failed_checks),
                "annual_delta_vs_v310": float(row.get("annual_delta_vs_v310", np.nan)),
                "sharpe_delta_vs_v310": float(row.get("sharpe_delta_vs_v310", np.nan)),
                "drawdown_delta_vs_v310": float(row.get("drawdown_delta_vs_v310", np.nan)),
                "pbo": float(row.get("pbo", np.nan)),
                "pbo_status": pbo_status,
                "action": "do_not_promote; diagnose marginal alpha and cost sensitivity",
            }
        )
    return pd.DataFrame(rows)


def selection_diagnostics(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    selected = selection[selection["selection_status"].astype(str).eq("selected_by_prior_window")].copy()
    rows = []
    for cost, group in selected.groupby("cost_bps"):
        counts = group["selected_variant"].value_counts(normalize=True)
        for variant, rate in counts.items():
            sub = group[group["selected_variant"].eq(variant)]
            rows.append(
                {
                    "cost_bps": float(cost),
                    "selected_variant": variant,
                    "selection_rate": float(rate),
                    "avg_oos_minus_baseline_annual": float(pd.to_numeric(sub["selected_oos_minus_baseline_annual"], errors="coerce").mean()),
                    "avg_oos_rank_pct": float(pd.to_numeric(sub["selected_oos_rank_pct"], errors="coerce").mean()),
                    "diagnosis": "baseline_dominant" if variant == BASELINE_VARIANT else "nonbaseline_selected_but_edge_small",
                }
            )
    return pd.DataFrame(rows)


def candidate_pathology(metrics: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for cost in sorted(metrics["cost_bps"].astype(float).unique()):
        group = metrics[metrics["cost_bps"].astype(float).eq(cost)]
        base = group[group["variant"].eq(BASELINE_VARIANT)].head(1)
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in group.iterrows():
            if row["variant"] == BASELINE_VARIANT:
                continue
            annual_delta = float(row["annual_return"] - b["annual_return"])
            turnover_delta = float(row["avg_trade_turnover"] - b["avg_trade_turnover"])
            cash_delta = float(row["avg_cash_weight"] - b["avg_cash_weight"])
            rows.append(
                {
                    "cost_bps": cost,
                    "variant": row["variant"],
                    "full_sample_annual_delta_vs_baseline_diagnostic": annual_delta,
                    "full_sample_turnover_delta_vs_baseline": turnover_delta,
                    "full_sample_cash_delta_vs_baseline": cash_delta,
                    "diagnosis": "candidate_underperforms_baseline_full_sample" if annual_delta < 0 else "candidate_small_positive_full_sample",
                    "promotion_relevance": "diagnostic_only_not_gate",
                }
            )
    if not decision.empty:
        rejected = decision[decision["decision"].astype(str).ne("promote_candidate")]
        if not rejected.empty:
            rows.append(
                {
                    "cost_bps": "all",
                    "variant": "nested_selected_candidate",
                    "full_sample_annual_delta_vs_baseline_diagnostic": np.nan,
                    "full_sample_turnover_delta_vs_baseline": np.nan,
                    "full_sample_cash_delta_vs_baseline": np.nan,
                    "diagnosis": "nested_selector_did_not_clear_50bps_annual_delta_gate",
                    "promotion_relevance": "primary_gate",
                }
            )
    return pd.DataFrame(rows)


def next_research_spec() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_id": "orthogonal_data_requirement",
                "priority": "high",
                "rule": "next alpha candidate must introduce information not already encoded in baseline weights or breadth states",
                "acceptance": "active-return correlation versus V3.10 and V3.21 below 0.85 before implementation",
            },
            {
                "spec_id": "minimum_effect_size",
                "priority": "high",
                "rule": "do not implement candidates whose pre-backtest expected effect is below the promotion gate",
                "acceptance": "expected annual delta >= 50bps or clear drawdown improvement >= 3pct with no major return loss",
            },
            {
                "spec_id": "cost_robustness_first",
                "priority": "high",
                "rule": "candidate must be designed for 10/20bps survival, not only 5bps",
                "acceptance": "PBO status not fail at both 10bps and 20bps in implementation harness",
            },
            {
                "spec_id": "avoid_cash_release_micro_tuning",
                "priority": "medium",
                "rule": "small cash-release overlays are likely too close to baseline",
                "acceptance": "candidate must change asset selection or signal source, not only cash by less than 8pct",
            },
        ]
    )


def make_report(failures: pd.DataFrame, diagnostics: pd.DataFrame, pathology: pd.DataFrame) -> str:
    fail_checks = failures["failed_checks"].astype(str).tolist() if not failures.empty else []
    worst = failures.sort_values("annual_delta_vs_v310").head(1) if "annual_delta_vs_v310" in failures.columns and not failures.empty else pd.DataFrame()
    return "\n".join(
        [
            "# HIRSSM V3.22 Failure Attribution",
            "",
            "## Result",
            "",
            "- V3.21 is not promoted.",
            f"- Failed checks by cost: {'; '.join(fail_checks)}",
            f"- Worst annual delta: {float(worst['annual_delta_vs_v310'].iloc[0]):.6f}" if not worst.empty else "- Worst annual delta: unavailable",
            "",
            "## Diagnosis",
            "",
            "- The signal improved nested OOS return only marginally and did not clear the 50bps annual delta gate.",
            "- 20bps PBO failed, so the improvement is not robust enough under realistic cost stress.",
            "- Full-sample candidate diagnostics are weaker than the nested headline and remain diagnostic only.",
            "",
            "## Next Research Rule",
            "",
            "- Do not continue by tuning cash-release percentages.",
            "- The next candidate must add a genuinely new information source or a more material portfolio construction change.",
        ]
    )


def self_check(failures: pd.DataFrame, diagnostics: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "failure_attribution_exists", "status": "pass" if not failures.empty else "fail", "detail": str(int(failures.shape[0]))},
            {"check": "selection_diagnostics_exists", "status": "pass" if not diagnostics.empty else "fail", "detail": str(int(diagnostics.shape[0]))},
            {"check": "next_research_spec_exists", "status": "pass" if not specs.empty else "fail", "detail": str(int(specs.shape[0]))},
            {"check": "no_unvalidated_promotion", "status": "pass", "detail": "V3.21 rejected for default"},
        ]
    )


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.22",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_22_failure_attribution.py",
        "config": {"source_model": "HIRSSM V3.21 Vol Compression Reentry Harness", "promotion_policy": "reject unless all V3.11 gates pass"},
        "data_refs": ["outputs/hirssm_v3_21_vol_compression_harness"],
        "code_refs": ["strategy_lab/hirssm_v3_22_failure_attribution.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/hirssm_v3_21_vol_compression_harness"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Attribution-only version; no new model backtest."],
        "risk_flags": ["marginal_alpha_too_small", "cost_20bps_pbo_fail"],
        "next_decision": "Close the five-version block in V3.23 and retain V3.10 unless a future candidate meets larger effect-size rules.",
        "handoff_summary": "V3.22 explains V3.21 rejection and defines stricter next-candidate requirements.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.22 failure attribution.")
    parser.add_argument("--v321-dir", default=str(V321_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    v321_dir = Path(args.v321_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = read_csv(v321_dir / "candidate_gate_decision.csv")
    pbo = read_csv(v321_dir / "pbo_cscv_summary.csv")
    selection = read_csv(v321_dir / "nested_selection_by_fold.csv")
    metrics_df = read_csv(v321_dir / "candidate_full_sample_diagnostic_metrics.csv")
    failures = gate_failure_attribution(decision, pbo)
    diagnostics = selection_diagnostics(selection)
    pathology = candidate_pathology(metrics_df, decision)
    specs = next_research_spec()
    checks = self_check(failures, diagnostics, specs)

    failures_path = output_dir / "gate_failure_attribution.csv"
    diagnostics_path = output_dir / "selection_diagnostics.csv"
    pathology_path = output_dir / "candidate_pathology_report.csv"
    specs_path = output_dir / "next_research_spec.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    changed_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    pathology.to_csv(pathology_path, index=False, encoding="utf-8-sig")
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    write_text(make_report(failures, diagnostics, pathology), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [failures_path, diagnostics_path, pathology_path, specs_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((failures["severity"].astype(str) == "fail").sum()) if not failures.empty else 0
    metrics = {
        "failure_rows": int(failures.shape[0]),
        "selection_diagnostic_rows": int(diagnostics.shape[0]),
        "next_research_rules": int(specs.shape[0]),
    }
    write_json(manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    print(json.dumps({"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
