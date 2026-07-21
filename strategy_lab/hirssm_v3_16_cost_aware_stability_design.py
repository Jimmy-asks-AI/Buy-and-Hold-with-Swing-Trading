#!/usr/bin/env python
"""HIRSSM V3.16 cost-aware stability design.

This version does not promote a model. It audits V3.15 turnover/cost behavior
and defines no-trade-band candidates for a future implementation round.
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
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_16" / "cost_aware_stability_design"
TASK_ID = "20260526_v3_16_cost_aware_stability_design"
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


def turnover_cost_diagnostics(v315_dir: Path) -> pd.DataFrame:
    metrics = read_csv(v315_dir / "candidate_full_sample_diagnostic_metrics.csv")
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for cost, group in metrics.groupby("cost_bps"):
        base = group[group["variant"].eq(BASELINE_VARIANT)].head(1)
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in group.iterrows():
            rows.append(
                {
                    "cost_bps": float(cost),
                    "variant": row["variant"],
                    "annual_return": float(row["annual_return"]),
                    "annual_delta_vs_baseline_full_sample_diagnostic": float(row["annual_return"] - b["annual_return"]),
                    "avg_trade_turnover": float(row["avg_trade_turnover"]),
                    "turnover_reduction_vs_baseline": float(b["avg_trade_turnover"] - row["avg_trade_turnover"]),
                    "total_cost": float(row["total_cost"]),
                    "cost_saving_vs_baseline": float(b["total_cost"] - row["total_cost"]),
                    "avg_cash_weight": float(row["avg_cash_weight"]),
                    "diagnostic_full_sample_only": True,
                }
            )
    return pd.DataFrame(rows)


def no_trade_band_capacity(v315_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(v315_dir.glob("target_weights_*.csv")):
        variant = path.stem.removeprefix("target_weights_")
        targets = read_csv(path)
        if targets.empty:
            continue
        targets["signal_date"] = pd.to_datetime(targets["signal_date"])
        prev: dict[str, float] = {}
        for signal_date, group in targets.groupby("signal_date", sort=True):
            current = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
            assets = sorted(set(prev) | set(current))
            if not prev:
                prev = current
                continue
            changes = pd.Series({asset: abs(current.get(asset, 0.0) - prev.get(asset, 0.0)) for asset in assets})
            total_turnover = float(changes.sum())
            for band in [0.02, 0.03, 0.05, 0.08]:
                small = float(changes[changes < band].sum())
                rows.append(
                    {
                        "variant": variant,
                        "signal_date": signal_date,
                        "band": band,
                        "total_turnover": total_turnover,
                        "skippable_turnover": small,
                        "skippable_turnover_share": small / total_turnover if total_turnover > 0 else 0.0,
                    }
                )
            prev = current
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return raw.groupby(["variant", "band"]).agg(
        observations=("signal_date", "size"),
        avg_total_turnover=("total_turnover", "mean"),
        avg_skippable_turnover=("skippable_turnover", "mean"),
        avg_skippable_turnover_share=("skippable_turnover_share", "mean"),
    ).reset_index()


def no_trade_band_spec(capacity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in sorted(capacity["variant"].unique()) if not capacity.empty else []:
        if variant == BASELINE_VARIANT:
            continue
        for band in [0.03, 0.05]:
            sub = capacity[(capacity["variant"].eq(variant)) & (capacity["band"].astype(float).eq(band))]
            if sub.empty:
                continue
            share = float(sub["avg_skippable_turnover_share"].iloc[0])
            rows.append(
                {
                    "variant": f"{variant}_no_trade_{int(band * 100)}pct",
                    "source_variant": variant,
                    "role": "execution_overlay_candidate",
                    "band": band,
                    "expected_turnover_reduction_share": share,
                    "implementation_priority": "high" if share >= 0.08 else "observation",
                    "forbidden": "do not claim alpha; evaluate only after-cost robustness and turnover reduction",
                    "acceptance_standard": "turnover falls at least 10pct and 20/30bps annual delta versus source does not deteriorate materially",
                }
            )
    return pd.DataFrame(rows)


def make_report(diag: pd.DataFrame, capacity: pd.DataFrame, specs: pd.DataFrame) -> str:
    best_cost = diag[(diag["cost_bps"].astype(float).eq(10.0)) & (~diag["variant"].eq(BASELINE_VARIANT))].sort_values("turnover_reduction_vs_baseline", ascending=False).head(1)
    lines = [
        "# HIRSSM V3.16 Cost-Aware Stability Design",
        "",
        "## Purpose",
        "",
        "Audit V3.15 cost behavior and define no-trade-band overlays before any new portfolio promotion.",
        "",
        "## Findings",
        "",
    ]
    if not best_cost.empty:
        row = best_cost.iloc[0]
        lines.extend(
            [
                f"- Best 10bps turnover reduction variant: {row['variant']}",
                f"- Turnover reduction vs baseline: {float(row['turnover_reduction_vs_baseline']):.6f}",
                f"- Annual delta vs baseline, full-sample diagnostic only: {float(row['annual_delta_vs_baseline_full_sample_diagnostic']):.6f}",
            ]
        )
    lines.extend(
        [
            f"- No-trade overlay specs: {int(specs.shape[0])}",
            "",
            "## Decision",
            "",
            "- V3.16 is accepted as cost design only.",
            "- Cost overlays must be tested as execution/risk controls, not alpha.",
        ]
    )
    return "\n".join(lines)


def self_check(diag: pd.DataFrame, capacity: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "turnover_cost_diagnostics_exists", "status": "pass" if not diag.empty else "fail", "detail": str(int(diag.shape[0]))},
            {"check": "no_trade_band_capacity_exists", "status": "pass" if not capacity.empty else "fail", "detail": str(int(capacity.shape[0]))},
            {"check": "no_trade_specs_exist", "status": "pass" if not specs.empty else "fail", "detail": str(int(specs.shape[0]))},
            {"check": "design_only_no_promotion", "status": "pass", "detail": "cost overlay design only"},
        ]
    )


def make_manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "execution_cost_analyst",
        "version": "V3.16",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_16_cost_aware_stability_design.py",
        "config": {"source_model": "HIRSSM V3.15 Breadth Overlay Harness", "design_only": True},
        "data_refs": ["outputs/hirssm_v3_15_breadth_overlay_harness"],
        "code_refs": ["strategy_lab/hirssm_v3_16_cost_aware_stability_design.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/hirssm_v3_15_breadth_overlay_harness"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Design-only; no after-cost overlay backtest yet.", "Turnover reduction is estimated from target deltas."],
        "risk_flags": ["cost_overlay_not_alpha", "full_sample_diagnostic_inputs"],
        "next_decision": "Use V3.16 no-trade specs only after candidate diversity filtering in V3.17.",
        "handoff_summary": "V3.16 audited cost behavior and produced no-trade-band overlay specs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.16 cost-aware stability design.")
    parser.add_argument("--v315-dir", default=str(V315_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    start_time = now_text()
    v315_dir = Path(args.v315_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diag = turnover_cost_diagnostics(v315_dir)
    capacity = no_trade_band_capacity(v315_dir)
    specs = no_trade_band_spec(capacity)
    checks = self_check(diag, capacity, specs)

    diag_path = output_dir / "turnover_cost_diagnostics.csv"
    capacity_path = output_dir / "no_trade_band_capacity.csv"
    specs_path = output_dir / "no_trade_band_spec.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    manifest_path = output_dir / "agent_run_manifest.json"
    changed_path = output_dir / "changed_files.txt"
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")
    capacity.to_csv(capacity_path, index=False, encoding="utf-8-sig")
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    write_text(make_report(diag, capacity, specs), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [diag_path, capacity_path, specs_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "diagnostic_rows": int(diag.shape[0]),
        "no_trade_spec_count": int(specs.shape[0]),
        "high_priority_specs": int((specs.get("implementation_priority", pd.Series(dtype=str)) == "high").sum()) if not specs.empty else 0,
    }
    write_json(make_manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    result = {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count, "metrics": metrics, "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
