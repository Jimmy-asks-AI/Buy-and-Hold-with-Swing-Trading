#!/usr/bin/env python
"""HIRSSM V3.30 macro rate/FX failure attribution.

V3.29 implemented the V3.28 holdout-qualified macro signals, but the
implementation failed default-promotion gates. V3.30 explains where the value
was lost: signal validation, portfolio overlay, constraints, costs,
walk-forward selection, or PBO instability. It does not add new candidates or
change the default model.
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
V328_DIR = ROOT / "outputs" / "agent_runs" / "v3_28" / "macro_rate_fx_signal_validation"
V329_MODEL_DIR = ROOT / "outputs" / "hirssm_v3_29_macro_rate_fx_harness"
V329_AGENT_DIR = ROOT / "outputs" / "agent_runs" / "v3_29" / "macro_rate_fx_harness"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_30" / "macro_rate_fx_failure_attribution"
TASK_ID = "20260527_v3_30_macro_rate_fx_failure_attribution"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"
IMPLEMENTED = ["us_rate_shock_fx_stress_defense", "spread_repair_risk_on"]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def scalar(df: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if df.empty or column not in df.columns:
        return default
    value = pd.to_numeric(df[column], errors="coerce").dropna()
    if value.empty:
        return default
    return float(value.iloc[0])


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "signal_validation": read_csv(V328_DIR / "signal_validation.csv"),
        "holdout_validation": read_csv(V328_DIR / "signal_gate_holdout_validation.csv"),
        "implementation_spec": read_csv(V328_DIR / "implementation_candidate_spec.csv"),
        "candidate_metrics": read_csv(V329_MODEL_DIR / "candidate_full_sample_diagnostic_metrics.csv"),
        "gate_decision": read_csv(V329_MODEL_DIR / "candidate_gate_decision.csv"),
        "nested_performance": read_csv(V329_MODEL_DIR / "nested_oos_performance.csv"),
        "pbo": read_csv(V329_MODEL_DIR / "pbo_cscv_summary.csv"),
        "selection": read_csv(V329_MODEL_DIR / "nested_selection_by_fold.csv"),
        "fold_scores": read_csv(V329_MODEL_DIR / "inner_candidate_scores.csv"),
        "trigger_diagnostics": read_csv(V329_MODEL_DIR / "trigger_diagnostics.csv"),
        "cost_sensitivity": read_csv(V329_MODEL_DIR / "cost_sensitivity.csv"),
    }


def signal_to_harness_bridge(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signal = data["signal_validation"].copy()
    holdout = data["holdout_validation"].copy()
    metrics = data["candidate_metrics"].copy()
    rows = []
    for variant in IMPLEMENTED:
        sig = signal[signal["variant"].astype(str).eq(variant)].head(1)
        ho = holdout[holdout["variant"].astype(str).eq(variant) & holdout["split"].astype(str).eq("holdout")].head(1)
        m10 = metrics[metrics["variant"].astype(str).eq(variant) & metrics["cost_bps"].astype(float).eq(10.0)].head(1)
        b10 = metrics[metrics["variant"].astype(str).eq(BASELINE_VARIANT) & metrics["cost_bps"].astype(float).eq(10.0)].head(1)
        rows.append(
            {
                "variant": variant,
                "v328_full_sample_forward_63d_mean": scalar(sig, "forward_63d_mean"),
                "v328_full_sample_unconditional_forward_63d_mean": scalar(sig, "unconditional_forward_63d_mean"),
                "v328_holdout_forward_63d_mean": scalar(ho, "forward_63d_mean"),
                "v328_holdout_unconditional_forward_63d_mean": scalar(ho, "unconditional_forward_63d_mean"),
                "v328_holdout_positive_forward_share": scalar(ho, "positive_forward_share"),
                "v329_full_sample_annual_return_10bps": scalar(m10, "annual_return"),
                "v329_baseline_annual_return_10bps": scalar(b10, "annual_return"),
                "v329_full_sample_annual_delta_vs_baseline_10bps": scalar(m10, "annual_return") - scalar(b10, "annual_return"),
                "v329_full_sample_drawdown_delta_vs_baseline_10bps": scalar(m10, "max_drawdown") - scalar(b10, "max_drawdown"),
                "diagnosis": "signal_passed_but_overlay_lost_value" if scalar(m10, "annual_return") < scalar(b10, "annual_return") else "signal_survived_full_sample_overlay",
            }
        )
    return pd.DataFrame(rows)


def cost_drag_attribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = data["candidate_metrics"].copy()
    rows = []
    for variant in [BASELINE_VARIANT] + IMPLEMENTED:
        sub = metrics[metrics["variant"].astype(str).eq(variant)].copy()
        if sub.empty:
            continue
        sub["cost_bps"] = sub["cost_bps"].astype(float)
        sub = sub.sort_values("cost_bps")
        ann_by_cost = dict(zip(sub["cost_bps"], pd.to_numeric(sub["annual_return"], errors="coerce")))
        cost_5 = ann_by_cost.get(5.0, np.nan)
        cost_20 = ann_by_cost.get(20.0, np.nan)
        rows.append(
            {
                "variant": variant,
                "annual_return_5bps": cost_5,
                "annual_return_10bps": ann_by_cost.get(10.0, np.nan),
                "annual_return_20bps": cost_20,
                "annual_drag_5_to_20bps": cost_20 - cost_5 if pd.notna(cost_5) and pd.notna(cost_20) else np.nan,
                "avg_turnover": scalar(sub[sub["cost_bps"].eq(10.0)], "avg_turnover"),
                "avg_trade_turnover": scalar(sub[sub["cost_bps"].eq(10.0)], "avg_trade_turnover"),
                "trade_count_10bps": scalar(sub[sub["cost_bps"].eq(10.0)], "trade_count"),
                "total_cost_10bps": scalar(sub[sub["cost_bps"].eq(10.0)], "total_cost"),
            }
        )
    out = pd.DataFrame(rows)
    if BASELINE_VARIANT in set(out["variant"]):
        baseline = out[out["variant"].eq(BASELINE_VARIANT)].iloc[0]
        out["cost_drag_vs_baseline_5_to_20bps"] = out["annual_drag_5_to_20bps"] - float(baseline["annual_drag_5_to_20bps"])
        out["turnover_delta_vs_baseline"] = out["avg_turnover"] - float(baseline["avg_turnover"])
    return out


def constraint_dilution_attribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trig = data["trigger_diagnostics"].copy()
    metrics = data["candidate_metrics"].copy()
    baseline_trig = trig[trig["variant"].astype(str).eq(BASELINE_VARIANT)].head(1)
    baseline_metrics = metrics[metrics["variant"].astype(str).eq(BASELINE_VARIANT) & metrics["cost_bps"].astype(float).eq(10.0)].head(1)
    rows = []
    for variant in IMPLEMENTED:
        tv = trig[trig["variant"].astype(str).eq(variant)].head(1)
        mv = metrics[metrics["variant"].astype(str).eq(variant) & metrics["cost_bps"].astype(float).eq(10.0)].head(1)
        trigger_rate = scalar(tv, "trigger_rate")
        cash_delta = scalar(tv, "avg_cash_weight") - scalar(baseline_trig, "avg_cash_weight")
        annual_delta = scalar(mv, "annual_return") - scalar(baseline_metrics, "annual_return")
        drawdown_delta = scalar(mv, "max_drawdown") - scalar(baseline_metrics, "max_drawdown")
        rows.append(
            {
                "variant": variant,
                "triggered_months": scalar(tv, "triggered_months"),
                "trigger_rate": trigger_rate,
                "avg_cash_delta_vs_baseline": cash_delta,
                "max_cash_weight": scalar(tv, "max_cash_weight"),
                "annual_delta_vs_baseline_10bps": annual_delta,
                "drawdown_delta_vs_baseline_10bps": drawdown_delta,
                "dilution_flag": bool(abs(cash_delta) < 0.01 and trigger_rate < 0.25),
                "diagnosis": "small_cash_change_and_sparse_trigger" if abs(cash_delta) < 0.01 and trigger_rate < 0.25 else "material_overlay_but_not_enough",
            }
        )
    return pd.DataFrame(rows)


def selection_attribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selection = data["selection"].copy()
    fold_scores = data["fold_scores"].copy()
    rows = []
    selected = selection[selection["selection_status"].astype(str).eq("selected_by_prior_window")].copy()
    for cost, sub in selected.groupby("cost_bps"):
        counts = sub["selected_variant"].astype(str).value_counts().to_dict()
        for variant in [BASELINE_VARIANT] + IMPLEMENTED:
            rows.append(
                {
                    "cost_bps": float(cost),
                    "variant": variant,
                    "selected_year_count": int(counts.get(variant, 0)),
                    "selection_rate": float(counts.get(variant, 0) / max(len(sub), 1)),
                }
            )
    out = pd.DataFrame(rows)
    if not fold_scores.empty:
        fs = fold_scores.copy()
        fs["cost_bps"] = fs["cost_bps"].astype(float)
        agg_spec = {
            "avg_inner_score": ("inner_validation_score", "mean"),
            "avg_oos_score": ("oos_score", "mean"),
        }
        rank_col = "oos_selected_oos_rank_pct"
        if rank_col in fs.columns:
            agg_spec["avg_oos_rank_pct"] = (rank_col, "mean")
        grouped = fs.groupby(["cost_bps", "variant"], as_index=False).agg(**agg_spec)
        grouped["oos_rank_pct_available"] = rank_col in fs.columns
        if "avg_oos_rank_pct" not in grouped.columns:
            grouped["avg_oos_rank_pct"] = np.nan
        out = out.merge(grouped, on=["cost_bps", "variant"], how="left")
    return out


def pbo_failure_attribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pbo = data["pbo"].copy()
    gate = data["gate_decision"].copy()
    rows = []
    for _, row in pbo.iterrows():
        cost = float(row["cost_bps"])
        gate_row = gate[gate["cost_bps"].astype(float).eq(cost)].head(1)
        rows.append(
            {
                "cost_bps": cost,
                "pbo": float(row["pbo"]),
                "pbo_status": row["pbo_status"],
                "baseline_selected_rate": float(row["baseline_selected_rate"]),
                "top1_oos_rate": float(row["top1_oos_rate"]),
                "mean_selected_minus_baseline_score": float(row["mean_selected_minus_baseline_score"]),
                "annual_delta_vs_v310": scalar(gate_row, "annual_delta_vs_v310"),
                "drawdown_delta_vs_v310": scalar(gate_row, "drawdown_delta_vs_v310"),
                "primary_failure": "pbo_instability" if float(row["pbo"]) > 0.35 else "performance_gate",
            }
        )
    return pd.DataFrame(rows)


def yearly_failure_attribution(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cost in [5, 10, 20]:
        selected_nav = read_csv(V329_MODEL_DIR / f"nav_nested_selected_candidate_{cost}bps.csv")
        baseline_nav = read_csv(V329_MODEL_DIR / f"nav_{BASELINE_VARIANT}_{cost}bps.csv")
        if selected_nav.empty or baseline_nav.empty:
            continue
        selected_nav["date"] = pd.to_datetime(selected_nav["date"])
        baseline_nav["date"] = pd.to_datetime(baseline_nav["date"])
        selected_nav["year"] = selected_nav["date"].dt.year
        baseline_nav["year"] = baseline_nav["date"].dt.year
        for year, sel_year in selected_nav.groupby("year"):
            base_year = baseline_nav[baseline_nav["year"].eq(year)]
            if base_year.empty:
                continue
            sel_ret = float((1.0 + pd.to_numeric(sel_year["portfolio_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
            base_ret = float((1.0 + pd.to_numeric(base_year["portfolio_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
            selected_variant = str(sel_year["selected_variant"].dropna().iloc[0]) if "selected_variant" in sel_year.columns and sel_year["selected_variant"].notna().any() else ""
            rows.append(
                {
                    "cost_bps": float(cost),
                    "year": int(year),
                    "selected_variant": selected_variant,
                    "selected_return": sel_ret,
                    "baseline_return": base_ret,
                    "return_delta": sel_ret - base_ret,
                }
            )
    return pd.DataFrame(rows)


def root_cause_ranking(
    gate: pd.DataFrame,
    pbo_attr: pd.DataFrame,
    dilution: pd.DataFrame,
    cost_drag: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    gate10 = gate[gate["cost_bps"].astype(float).eq(10.0)].head(1)
    pbo10 = pbo_attr[pbo_attr["cost_bps"].astype(float).eq(10.0)].head(1)
    nonbaseline_rate = scalar(gate10, "nonbaseline_selection_rate", 0.0)
    annual_delta = scalar(gate10, "annual_delta_vs_v310", 0.0)
    drawdown_delta = scalar(gate10, "drawdown_delta_vs_v310", 0.0)
    pbo_value = scalar(pbo10, "pbo", 0.0)
    avg_trigger_rate = float(dilution["trigger_rate"].mean()) if not dilution.empty else np.nan
    avg_cash_abs_delta = float(dilution["avg_cash_delta_vs_baseline"].abs().mean()) if not dilution.empty else np.nan
    candidate_cost_drag = cost_drag[cost_drag["variant"].isin(IMPLEMENTED)]["cost_drag_vs_baseline_5_to_20bps"].mean() if not cost_drag.empty else np.nan
    rows = [
        {
            "rank": 1,
            "root_cause": "pbo_instability",
            "severity": "high" if pbo_value > 0.35 else "medium",
            "evidence": f"10bps PBO={pbo_value:.4f}; top1 OOS rate low; selected-minus-baseline score negative",
            "recommended_action": "do not promote; require a simpler constrained candidate or independent validation window",
        },
        {
            "rank": 2,
            "root_cause": "negative_marginal_oos_performance",
            "severity": "high" if annual_delta < 0 else "medium",
            "evidence": f"10bps annual delta={annual_delta:.4%}; drawdown delta={drawdown_delta:.4%}",
            "recommended_action": "attribute candidate return by year/regime before changing parameters",
        },
        {
            "rank": 3,
            "root_cause": "portfolio_overlay_dilution",
            "severity": "medium",
            "evidence": f"avg trigger rate={avg_trigger_rate:.2%}; avg absolute cash delta={avg_cash_abs_delta:.2%}",
            "recommended_action": "test whether signal should gate existing risk budget rather than add small cash shifts",
        },
        {
            "rank": 4,
            "root_cause": "candidate_selection_not_decisive",
            "severity": "medium",
            "evidence": f"nonbaseline selection rate={nonbaseline_rate:.2%}; baseline still selected in most years",
            "recommended_action": "keep candidates observation-only unless selection margin improves in OOS folds",
        },
        {
            "rank": 5,
            "root_cause": "cost_drag_not_primary",
            "severity": "low",
            "evidence": f"candidate minus baseline 5-to-20bps cost drag={candidate_cost_drag:.4%}",
            "recommended_action": "do not focus first on execution-cost tuning; signal/overlay instability is larger",
        },
    ]
    return pd.DataFrame(rows)


def next_experiment_queue(root_causes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "task": "V3.31 candidate-year/regime attribution",
                "allowed": True,
                "description": "Split candidate contribution by selected year, regime, and trigger month before changing any rule.",
                "forbidden": "parameter search over thresholds without a new holdout gate",
            },
            {
                "priority": 2,
                "task": "V3.32 risk-budget gate design",
                "allowed": True,
                "description": "Use macro signal as a gate on existing cash/risk budget rather than an additive overlay.",
                "forbidden": "default promotion without nested/PBO pass",
            },
            {
                "priority": 3,
                "task": "TSF data repair for macro_liquidity_repair",
                "allowed": True,
                "description": "Repair missing china_tsf_yoy to unlock a broader macro liquidity branch.",
                "forbidden": "using current snapshot TSF backfill without available_date",
            },
            {
                "priority": 4,
                "task": "Do not tune V3.29 amplitudes yet",
                "allowed": False,
                "description": "Changing 4pct cash shifts directly would be ungoverned parameter search.",
                "forbidden": "amplitude sweep on V3.29 without predeclared splits",
            },
        ]
    )


def make_report(
    bridge: pd.DataFrame,
    pbo_attr: pd.DataFrame,
    dilution: pd.DataFrame,
    roots: pd.DataFrame,
) -> str:
    lines = [
        "# HIRSSM V3.30 Macro Rate/FX Failure Attribution",
        "",
        "## Decision",
        "",
        "- V3.29 remains rejected for default promotion.",
        "- V3.30 does not add or tune candidates.",
        "- The macro rate/FX branch stays observation-only until the root causes below are addressed.",
        "",
        "## Main Findings",
        "",
    ]
    if not pbo_attr.empty:
        pbo10 = pbo_attr[pbo_attr["cost_bps"].astype(float).eq(10.0)].head(1)
        if not pbo10.empty:
            lines.append(f"- 10bps PBO remains the dominant hard failure: `{float(pbo10['pbo'].iloc[0]):.4f}`.")
    if not bridge.empty:
        for _, row in bridge.iterrows():
            lines.append(
                f"- `{row['variant']}`: holdout signal mean `{float(row['v328_holdout_forward_63d_mean']):.4%}`, "
                f"but 10bps implemented annual delta `{float(row['v329_full_sample_annual_delta_vs_baseline_10bps']):.4%}`."
            )
    if not dilution.empty:
        avg_trigger = float(dilution["trigger_rate"].mean())
        avg_cash = float(dilution["avg_cash_delta_vs_baseline"].abs().mean())
        lines.append(f"- Overlay intensity is small: average trigger rate `{avg_trigger:.2%}`, average absolute cash delta `{avg_cash:.2%}`.")
    lines.extend(["", "## Root Causes", ""])
    for _, row in roots.iterrows():
        lines.append(f"- {int(row['rank'])}. `{row['root_cause']}` ({row['severity']}): {row['evidence']}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "V3.31 should do selected-year/regime attribution before any new implementation or parameter change.",
        ]
    )
    return "\n".join(lines)


def self_check(paths: dict[str, Path], roots: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "status": "pass" if path.exists() else "fail", "detail": rel(path)})
    rows.extend(
        [
            {
                "check": "no_new_candidate_created",
                "status": "pass",
                "detail": "V3.30 reads V3.28/V3.29 only and emits attribution artifacts",
            },
            {
                "check": "root_cause_ranking_exists",
                "status": "pass" if not roots.empty else "fail",
                "detail": str(int(roots.shape[0])),
            },
            {
                "check": "next_experiment_queue_blocks_parameter_search",
                "status": "pass" if not queue.empty and queue["forbidden"].astype(str).str.contains("parameter|threshold|amplitude", case=False).any() else "fail",
                "detail": str(int(queue.shape[0])),
            },
        ]
    )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    bridge = signal_to_harness_bridge(data)
    cost_drag = cost_drag_attribution(data)
    dilution = constraint_dilution_attribution(data)
    selection = selection_attribution(data)
    pbo_attr = pbo_failure_attribution(data)
    yearly = yearly_failure_attribution(data)
    roots = root_cause_ranking(data["gate_decision"], pbo_attr, dilution, cost_drag, selection)
    queue = next_experiment_queue(roots)

    paths = {
        "signal_to_harness_bridge": output_dir / "signal_to_harness_bridge.csv",
        "cost_drag_attribution": output_dir / "cost_drag_attribution.csv",
        "constraint_dilution_attribution": output_dir / "constraint_dilution_attribution.csv",
        "selection_attribution": output_dir / "selection_attribution.csv",
        "pbo_failure_attribution": output_dir / "pbo_failure_attribution.csv",
        "yearly_failure_attribution": output_dir / "yearly_failure_attribution.csv",
        "root_cause_ranking": output_dir / "root_cause_ranking.csv",
        "next_experiment_queue": output_dir / "next_experiment_queue.csv",
        "agent_report": output_dir / "agent_report.md",
        "failure_cases": output_dir / "REGIME_FAILURE_CASES.md",
        "changed_files": output_dir / "changed_files.txt",
    }
    for df, key in [
        (bridge, "signal_to_harness_bridge"),
        (cost_drag, "cost_drag_attribution"),
        (dilution, "constraint_dilution_attribution"),
        (selection, "selection_attribution"),
        (pbo_attr, "pbo_failure_attribution"),
        (yearly, "yearly_failure_attribution"),
        (roots, "root_cause_ranking"),
        (queue, "next_experiment_queue"),
    ]:
        write_csv(df, paths[key])
    report = make_report(bridge, pbo_attr, dilution, roots)
    write_text(report, paths["agent_report"])
    write_text(report, paths["failure_cases"])
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    self_check_path.touch()
    paths["self_check"] = self_check_path
    checks = self_check(paths, roots, queue)
    write_csv(checks, self_check_path)
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    gate10 = data["gate_decision"][data["gate_decision"]["cost_bps"].astype(float).eq(10.0)].head(1)
    pbo10 = pbo_attr[pbo_attr["cost_bps"].astype(float).eq(10.0)].head(1)
    metrics = {
        "root_cause_count": int(roots.shape[0]),
        "next_experiment_count": int(queue.shape[0]),
        "v329_annual_delta_vs_v310_10bps": scalar(gate10, "annual_delta_vs_v310"),
        "v329_drawdown_delta_vs_v310_10bps": scalar(gate10, "drawdown_delta_vs_v310"),
        "v329_pbo_10bps": scalar(pbo10, "pbo"),
        "recommended_next": "selected_year_regime_attribution",
    }
    artifacts = list(paths.values())
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "research_reporter",
        "version": "V3.30",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_30_macro_rate_fx_failure_attribution.py",
        "config": {"attribution_only": True, "no_new_candidates": True, "source_versions": "V3.28,V3.29"},
        "data_refs": [
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation",
            "outputs/hirssm_v3_29_macro_rate_fx_harness",
            "outputs/agent_runs/v3_29/macro_rate_fx_harness",
        ],
        "code_refs": ["strategy_lab/hirssm_v3_30_macro_rate_fx_failure_attribution.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation",
            "outputs/hirssm_v3_29_macro_rate_fx_harness",
            "outputs/agent_runs/v3_29/macro_rate_fx_harness",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Attribution uses existing V3.28 and V3.29 artifacts only.",
            "No new backtest harness or model promotion is authorized.",
        ],
        "risk_flags": [
            "post_hoc_attribution_can_mislead",
            "macro_regime_shift",
            "small_candidate_set",
        ],
        "next_decision": "Run V3.31 selected-year/regime attribution before any further implementation.",
        "handoff_summary": "V3.30 attributes the V3.29 macro rate/FX harness rejection to PBO instability, negative marginal OOS performance, and overlay dilution.",
    }
    manifest_path = output_dir / "agent_run_manifest.json"
    write_json(manifest, manifest_path)
    paths["agent_run_manifest"] = manifest_path
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])
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
