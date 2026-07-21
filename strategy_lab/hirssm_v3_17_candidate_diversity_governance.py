#!/usr/bin/env python
"""HIRSSM V3.17 candidate diversity governance.

V3.17 addresses the V3.14/V3.15 finding that candidate active returns are too
similar. It filters near-duplicate candidates before future PBO runs.
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
V315_DIR = ROOT / "outputs" / "hirssm_v3_15_breadth_overlay_harness"
V316_DIR = ROOT / "outputs" / "agent_runs" / "v3_16" / "cost_aware_stability_design"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_17" / "candidate_diversity_governance"
TASK_ID = "20260526_v3_17_candidate_diversity_governance"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


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


def active_return_wide(v315_dir: Path, cost_bps: int = 10) -> pd.DataFrame:
    frames = []
    for path in sorted(v315_dir.glob(f"nav_*_{cost_bps}bps.csv")):
        name = path.stem.removeprefix("nav_").removesuffix(f"_{cost_bps}bps")
        if name.startswith("nested_selected"):
            continue
        nav = read_csv(path)
        if {"date", "portfolio_return", "benchmark_return"}.issubset(nav.columns):
            item = nav[["date", "portfolio_return", "benchmark_return"]].copy()
            item["date"] = pd.to_datetime(item["date"])
            item[name] = pd.to_numeric(item["portfolio_return"], errors="coerce") - pd.to_numeric(item["benchmark_return"], errors="coerce")
            frames.append(item[["date", name]])
    if not frames:
        return pd.DataFrame()
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="date", how="inner")
    return wide


def similarity_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    corr = wide.drop(columns=["date"]).corr()
    corr.index.name = "variant"
    return corr.reset_index()


def candidate_scores(v315_dir: Path) -> pd.DataFrame:
    metrics = read_csv(v315_dir / "candidate_full_sample_diagnostic_metrics.csv")
    if metrics.empty:
        return pd.DataFrame()
    m10 = metrics[metrics["cost_bps"].astype(float).eq(10.0)].copy()
    return m10[["variant", "annual_return", "sharpe_no_rf", "max_drawdown", "avg_trade_turnover", "diagnostic_full_sample_only"]]


def duplicate_clusters(sim: pd.DataFrame, scores: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    if sim.empty:
        return pd.DataFrame()
    corr = sim.set_index("variant")
    score_map = scores.set_index("variant")["annual_return"].to_dict() if not scores.empty else {}
    rows = []
    variants = list(corr.index)
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= threshold:
                left_score = float(score_map.get(left, np.nan))
                right_score = float(score_map.get(right, np.nan))
                keep = left if (pd.notna(left_score) and (pd.isna(right_score) or left_score >= right_score)) else right
                drop = right if keep == left else left
                if left == BASELINE_VARIANT:
                    keep, drop = left, right
                elif right == BASELINE_VARIANT:
                    keep, drop = right, left
                rows.append(
                    {
                        "left_variant": left,
                        "right_variant": right,
                        "active_return_corr": value,
                        "threshold": threshold,
                        "keep_variant": keep,
                        "drop_variant": drop,
                        "reason": "near_duplicate_active_return",
                    }
                )
    return pd.DataFrame(rows)


def filtered_candidate_set(scores: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    drops = set(clusters["drop_variant"].astype(str)) if not clusters.empty else set()
    rows = []
    for _, row in scores.iterrows():
        variant = str(row["variant"])
        rows.append(
            {
                "variant": variant,
                "include_in_next_pbo": variant not in drops,
                "role": "control" if variant == BASELINE_VARIANT else "candidate",
                "drop_reason": "near_duplicate_active_return" if variant in drops else "",
                "annual_return_10bps_diagnostic": row["annual_return"],
                "sharpe_10bps_diagnostic": row["sharpe_no_rf"],
                "diagnostic_full_sample_only": True,
            }
        )
    return pd.DataFrame(rows)


def selector_guardrail_spec(filtered: pd.DataFrame, v316_dir: Path) -> pd.DataFrame:
    no_trade = read_csv(v316_dir / "no_trade_band_spec.csv")
    included = filtered[filtered["include_in_next_pbo"].astype(bool)]["variant"].tolist() if not filtered.empty else []
    rows = [
        {
            "guardrail": "candidate_diversity_filter",
            "rule": "exclude candidate if abs(active_return_corr) >= 0.90 versus a kept candidate",
            "applies_to": "candidate universe before PBO",
            "required_artifact": "candidate_similarity_matrix.csv",
        },
        {
            "guardrail": "minimum_candidate_set",
            "rule": "baseline plus at least one nonbaseline candidate required; otherwise block implementation",
            "applies_to": ",".join(included),
            "required_artifact": "filtered_candidate_set.csv",
        },
        {
            "guardrail": "execution_overlay_separate_from_alpha",
            "rule": "no-trade overlays from V3.16 must be evaluated against their source candidate and cannot be promoted as alpha alone",
            "applies_to": ",".join(no_trade["variant"].astype(str).tolist()) if not no_trade.empty else "none",
            "required_artifact": "no_trade_band_spec.csv",
        },
    ]
    return pd.DataFrame(rows)


def make_report(sim: pd.DataFrame, clusters: pd.DataFrame, filtered: pd.DataFrame) -> str:
    max_corr = np.nan
    if not sim.empty:
        values = sim.set_index("variant").to_numpy(dtype=float)
        if values.shape[0] > 1:
            max_corr = float(np.nanmax(np.abs(values[~np.eye(values.shape[0], dtype=bool)])))
    included = filtered[filtered["include_in_next_pbo"].astype(bool)]["variant"].tolist() if not filtered.empty else []
    return "\n".join(
        [
            "# HIRSSM V3.17 Candidate Diversity Governance",
            "",
            "## Purpose",
            "",
            "Filter near-duplicate candidates before another PBO run.",
            "",
            "## Findings",
            "",
            f"- Max absolute active-return correlation: {max_corr:.6f}" if pd.notna(max_corr) else "- Similarity unavailable.",
            f"- Duplicate pairs: {int(clusters.shape[0])}",
            f"- Included next-PBO candidates: {', '.join(included)}",
            "",
            "## Decision",
            "",
            "- V3.17 is accepted as selector governance only.",
            "- Future implementation must run PBO after duplicate filtering.",
        ]
    )


def self_check(sim: pd.DataFrame, clusters: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "similarity_matrix_exists", "status": "pass" if not sim.empty else "fail", "detail": str(int(sim.shape[0]))},
            {"check": "filtered_candidate_set_exists", "status": "pass" if not filtered.empty else "fail", "detail": str(int(filtered.shape[0]))},
            {"check": "baseline_retained", "status": "pass" if BASELINE_VARIANT in set(filtered.loc[filtered["include_in_next_pbo"].astype(bool), "variant"]) else "fail", "detail": BASELINE_VARIANT},
            {"check": "diversity_governance_only", "status": "pass", "detail": "no model promotion"},
        ]
    )


def make_manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "chief_quant_orchestrator",
        "version": "V3.17",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_17_candidate_diversity_governance.py",
        "config": {"correlation_threshold": 0.90, "design_only": True},
        "data_refs": ["outputs/hirssm_v3_15_breadth_overlay_harness", "outputs/agent_runs/v3_16/cost_aware_stability_design"],
        "code_refs": ["strategy_lab/hirssm_v3_17_candidate_diversity_governance.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/hirssm_v3_15_breadth_overlay_harness", "outputs/agent_runs/v3_16/cost_aware_stability_design"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Design-only; filtered set has not been rebacktested.", "Correlation measured on historical active returns."],
        "risk_flags": ["candidate_similarity_high", "selector_governance_not_alpha"],
        "next_decision": "Use filtered candidate set in a future implementation harness.",
        "handoff_summary": "V3.17 filtered near-duplicate candidates and documented selector guardrails.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.17 candidate diversity governance artifacts.")
    parser.add_argument("--v315-dir", default=str(V315_DIR))
    parser.add_argument("--v316-dir", default=str(V316_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wide = active_return_wide(Path(args.v315_dir), cost_bps=10)
    sim = similarity_matrix(wide)
    scores = candidate_scores(Path(args.v315_dir))
    clusters = duplicate_clusters(sim, scores, threshold=0.90)
    filtered = filtered_candidate_set(scores, clusters)
    guardrails = selector_guardrail_spec(filtered, Path(args.v316_dir))
    checks = self_check(sim, clusters, filtered)

    sim_path = output_dir / "candidate_similarity_matrix.csv"
    clusters_path = output_dir / "duplicate_cluster_report.csv"
    filtered_path = output_dir / "filtered_candidate_set.csv"
    guardrails_path = output_dir / "selector_guardrail_spec.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    manifest_path = output_dir / "agent_run_manifest.json"
    changed_path = output_dir / "changed_files.txt"
    sim.to_csv(sim_path, index=False, encoding="utf-8-sig")
    clusters.to_csv(clusters_path, index=False, encoding="utf-8-sig")
    filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    guardrails.to_csv(guardrails_path, index=False, encoding="utf-8-sig")
    write_text(make_report(sim, clusters, filtered), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [sim_path, clusters_path, filtered_path, guardrails_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "candidate_count": int(filtered.shape[0]),
        "included_count": int(filtered["include_in_next_pbo"].astype(bool).sum()) if not filtered.empty else 0,
        "duplicate_pair_count": int(clusters.shape[0]),
    }
    write_json(make_manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    result = {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count, "metrics": metrics, "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
