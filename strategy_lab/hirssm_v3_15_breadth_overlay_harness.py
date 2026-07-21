#!/usr/bin/env python
"""HIRSSM V3.15 breadth overlay implementation harness.

V3.15 implements only the V3.14 signal that passed pre-backtest validation:
orthogonal breadth regime overlay. Residual industry momentum and style
barbell remain observation-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hirssm_v2_model as model
import hirssm_v2_walk_forward as wf
import hirssm_v3_10_clean_baseline as v310
import hirssm_v3_11_nested_candidate_harness as v311
from model_run_manifest import build_model_run_manifest, validate_model_run_manifest


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
V314_DIR = ROOT / "outputs" / "agent_runs" / "v3_14" / "orthogonal_candidate_research"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_15_breadth_overlay_harness"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_15" / "breadth_overlay_harness"
TASK_ID = "20260526_v3_15_breadth_overlay_harness"
MODEL_VERSION = "HIRSSM V3.15 Breadth Overlay Harness"
BASELINE_VARIANT = v311.BASELINE_VARIANT


CANDIDATES = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol baseline.",
        "mode": "baseline",
        "weak_scale": 1.0,
        "repair_cash_release": 0.0,
    },
    {
        "variant": "orthogonal_breadth_regime_overlay",
        "role": "candidate",
        "description": "Scale gross exposure down when industry breadth is weak; release cash only when breadth repair is strong.",
        "mode": "breadth_overlay",
        "weak_scale": 0.85,
        "repair_cash_release": 0.05,
    },
    {
        "variant": "conservative_breadth_cash_guard",
        "role": "candidate",
        "description": "Lower-amplitude breadth risk budget overlay for robustness comparison.",
        "mode": "breadth_overlay",
        "weak_scale": 0.92,
        "repair_cash_release": 0.03,
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_costs(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def candidate_registry(config: dict) -> pd.DataFrame:
    disabled = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    rows = []
    for item in CANDIDATES:
        rows.append(
            {
                "variant": item["variant"],
                "role": item["role"],
                "description": item["description"],
                "multipliers_json": json.dumps(
                    {"mode": item["mode"], "weak_scale": item["weak_scale"], "repair_cash_release": item["repair_cash_release"]},
                    sort_keys=True,
                ),
                "disabled_experts": disabled,
                "selection_source": "v3_14_pre_backtest_signal_gate",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": item["role"] == "candidate",
            }
        )
    return pd.DataFrame(rows)


def breadth_by_date(regimes: pd.DataFrame) -> pd.DataFrame:
    reg = regimes.copy()
    reg["date"] = pd.to_datetime(reg["date"])
    reg["breadth_repair_signal"] = (
        pd.to_numeric(reg["industry_above_ma60_ratio"], errors="coerce")
        + pd.to_numeric(reg["industry_positive_ret20_ratio"], errors="coerce")
    ) / 2.0
    reg["breadth_state"] = np.select(
        [reg["breadth_repair_signal"] < 0.35, reg["breadth_repair_signal"] > 0.60],
        ["weak", "repair"],
        default="neutral",
    )
    return reg[["date", "breadth_repair_signal", "breadth_state"]]


def normalize_date_group(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    total = float(pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).sum())
    if total > 0:
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0) / total
    g["target_weight"] = g["weight"]
    return g


def apply_breadth_overlay(base_targets: pd.DataFrame, panel: dict[str, pd.DataFrame], spec: dict[str, Any], config: dict) -> pd.DataFrame:
    if spec["mode"] == "baseline":
        out = base_targets.copy()
        out["variant"] = spec["variant"]
        return out
    breadth = breadth_by_date(panel["regimes"]).rename(columns={"date": "signal_date"})
    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    targets = targets.merge(breadth, on="signal_date", how="left")
    rows = []
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = group.copy()
        state = str(g["breadth_state"].dropna().iloc[0]) if g["breadth_state"].notna().any() else "neutral"
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if not g["asset"].eq("CASH").any():
            template = {col: "" for col in g.columns}
            template.update({"signal_date": signal_date, "asset": "CASH", "asset_type": "cash", "weight": 0.0, "target_weight": 0.0})
            g = pd.concat([g, pd.DataFrame([template])], ignore_index=True)
        cash_idx = g.index[g["asset"].eq("CASH")][0]
        noncash_idx = g.index[~g["asset"].eq("CASH")]
        if state == "weak":
            scale = float(spec["weak_scale"])
            released = float(g.loc[noncash_idx, "weight"].sum() * (1.0 - scale))
            g.loc[noncash_idx, "weight"] = g.loc[noncash_idx, "weight"] * scale
            g.loc[cash_idx, "weight"] = float(g.loc[cash_idx, "weight"]) + released
        elif state == "repair":
            release = min(float(spec["repair_cash_release"]), float(g.loc[cash_idx, "weight"]))
            noncash_sum = float(g.loc[noncash_idx, "weight"].sum())
            if release > 0 and noncash_sum > 0:
                g.loc[noncash_idx, "weight"] = g.loc[noncash_idx, "weight"] + release * g.loc[noncash_idx, "weight"] / noncash_sum
                g.loc[cash_idx, "weight"] = float(g.loc[cash_idx, "weight"]) - release
        g["breadth_overlay_state"] = state
        g["breadth_overlay_applied"] = state in {"weak", "repair"}
        rows.append(normalize_date_group(g))
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = v310.enforce_cash_cap(out, config)
    out = v310.enforce_turnover_cap(out, max_turnover=0.60)
    out["variant"] = spec["variant"]
    out["candidate_role"] = spec["role"]
    return out


def run_candidates(panel: dict[str, pd.DataFrame], config: dict, costs: list[float], output_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], dict[str, pd.DataFrame]]:
    base_targets = v310.build_targets(panel, config)
    rows = []
    navs: dict[tuple[str, float], pd.DataFrame] = {}
    targets_by_variant: dict[str, pd.DataFrame] = {}
    for spec in CANDIDATES:
        targets = apply_breadth_overlay(base_targets, panel, spec, config)
        targets_by_variant[spec["variant"]] = targets
        model.write_csv(targets, output_dir / f"target_weights_{spec['variant']}.csv")
        for cost in costs:
            bt = model.run_backtest(panel["returns"], targets, cost, panel["broad_code"])
            nav = bt["nav"].copy()
            nav["variant"] = spec["variant"]
            nav["cost_bps"] = float(cost)
            summary = model.summarize_nav(nav)
            if not summary.empty:
                summary.insert(0, "variant", spec["variant"])
                summary.insert(1, "role", spec["role"])
                summary.insert(2, "cost_bps", float(cost))
                summary["diagnostic_full_sample_only"] = True
                summary["annual_excess_vs_benchmark"] = summary["annual_return"] - summary["benchmark_annual_return"]
                rows.append(summary)
            navs[(spec["variant"], float(cost))] = nav
            model.write_csv(nav, output_dir / f"nav_{spec['variant']}_{int(cost)}bps.csv")
            model.write_csv(bt["trades"], output_dir / f"trades_{spec['variant']}_{int(cost)}bps.csv")
    metrics = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return metrics, navs, targets_by_variant


def target_integrity(targets_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, targets in targets_by_variant.items():
        weights = pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0)
        sums = targets.assign(_w=weights).groupby("signal_date")["_w"].sum()
        rows.append(
            {
                "check": f"target_integrity_{variant}",
                "status": "pass" if not targets.empty and float(weights.min()) >= -1e-10 and float((sums - 1.0).abs().max()) <= 1e-6 else "fail",
                "detail": f"rows={targets.shape[0]}; max_sum_error={float((sums - 1.0).abs().max()) if not sums.empty else np.nan:.8f}",
            }
        )
    return pd.DataFrame(rows)


def make_report(performance: pd.DataFrame, pbo: pd.DataFrame, decision: pd.DataFrame) -> str:
    perf10 = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1)
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1)
    lines = [
        "# HIRSSM V3.15 Breadth Overlay Harness",
        "",
        "## Purpose",
        "",
        "Implement the V3.14 passing breadth regime signal as a governed candidate overlay.",
        "",
        "## 10bps Nested OOS",
        "",
    ]
    if not perf10.empty:
        item = perf10.iloc[0]
        lines.extend(
            [
                f"- Annual return: {float(item['annual_return']):.6f}",
                f"- Sharpe no RF: {float(item['sharpe_no_rf']):.6f}",
                f"- Max drawdown: {float(item['max_drawdown']):.6f}",
                f"- Annual delta vs V3.10: {float(item.get('annual_delta_vs_v310', np.nan)):.6f}",
            ]
        )
    if not pbo10.empty:
        lines.extend(["", "## PBO", "", f"- 10bps PBO: {float(pbo10['pbo'].iloc[0]):.6f}", f"- 10bps status: {pbo10['pbo_status'].iloc[0]}"])
    decision_text = decision["overall_decision"].iloc[0] if not decision.empty else "blocked"
    lines.extend(["", "## Decision", "", f"- Overall decision: {decision_text}", "- Full-sample candidate metrics remain diagnostic only."])
    return "\n".join(lines)


def make_agent_manifest(start_time: str, agent_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.15",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_15_breadth_overlay_harness.py",
        "config": {"config_path": str(CONFIG.as_posix()), "baseline_variant": BASELINE_VARIANT, "candidate_count": len(CANDIDATES)},
        "data_refs": ["outputs/agent_runs/v3_14/orthogonal_candidate_research", "data_raw/index/akshare_csindex", "data_raw/index/akshare_sw_industry"],
        "code_refs": ["strategy_lab/hirssm_v3_15_breadth_overlay_harness.py", "strategy_lab/hirssm_v3_10_clean_baseline.py", "strategy_lab/hirssm_v3_11_nested_candidate_harness.py"],
        "output_dir": str(agent_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_14/orthogonal_candidate_research", "configs/hirssm_v2_default.json", "data_raw/index"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Only breadth signal passed V3.14; other proposed alpha signals are observation-only.", "PBO applies to a small candidate set."],
        "risk_flags": ["breadth_overlay_may_reduce_upside", "full_sample_metrics_diagnostic_only"],
        "next_decision": "If not promoted, pass V3.15 targets to V3.16 cost-aware stabilization.",
        "handoff_summary": "V3.15 implemented breadth risk-budget candidates and emitted nested/PBO gate artifacts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.15 breadth overlay harness.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--agent-output-dir", default=str(AGENT_OUTPUT_DIR))
    parser.add_argument("--costs", default="5,10,20,30")
    args = parser.parse_args()

    start_time = now_text()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    agent_dir = Path(args.agent_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    costs = parse_costs(args.costs)

    panel = wf.build_panel(model, root, config, None, None)
    registry = candidate_registry(config)
    candidate_metrics, navs, targets_by_variant = run_candidates(panel, config, costs, output_dir)
    selection, fold_scores, selected_navs, nested_performance = v311.nested_walk_forward(
        navs=navs,
        costs=costs,
        lookback_years=5,
        inner_validation_years=1,
        min_train_days=756,
        embargo_days=21,
    )
    pbo_folds, pbo_summary = v311.purged_block_pbo(navs=navs, costs=costs, n_blocks=10, train_blocks=5, purge_days=63, embargo_days=21)
    decision = v311.promotion_decision(nested_performance, pbo_summary, selection)
    split_manifest = v311.split_manifest_from_selection(selection)
    embargo_audit = v311.embargo_purge_audit(split_manifest)
    outer_oos = v311.outer_fold_oos_results(selection)
    same_period = v311.same_period_baseline_comparison(nested_performance)
    cost_sensitivity = v311.cost_sensitivity_table(nested_performance, decision)
    checks = pd.concat(
        [
            v311.build_constraint_checks(registry=registry, selection=selection, performance=nested_performance, pbo_summary=pbo_summary, decision=decision),
            target_integrity(targets_by_variant),
        ],
        ignore_index=True,
        sort=False,
    )
    findings = v311.validation_findings(checks, decision, pbo_summary)
    leakage = v311.leakage_checklist(split_manifest, embargo_audit)
    robustness = v311.robustness_summary(decision, pbo_summary, cost_sensitivity)

    paths = {
        "candidate_registry": output_dir / "candidate_registry.csv",
        "candidate_registry_json": output_dir / "candidate_registry.json",
        "split_manifest": output_dir / "split_manifest.csv",
        "split_manifest_json": output_dir / "split_manifest.json",
        "embargo_purge_audit": output_dir / "embargo_purge_audit.csv",
        "inner_candidate_scores": output_dir / "inner_candidate_scores.csv",
        "nested_selection_by_fold": output_dir / "nested_selection_by_fold.csv",
        "outer_fold_oos_results": output_dir / "outer_fold_oos_results.csv",
        "same_period_baseline_comparison": output_dir / "same_period_baseline_comparison.csv",
        "cost_sensitivity": output_dir / "cost_sensitivity.csv",
        "pbo_cscv_summary": output_dir / "pbo_cscv_summary.csv",
        "pbo_cscv_splits": output_dir / "pbo_cscv_splits.csv",
        "candidate_gate_decision": output_dir / "candidate_gate_decision.csv",
        "validation_findings": output_dir / "validation_findings.csv",
        "leakage_checklist": output_dir / "leakage_checklist.csv",
        "robustness_summary": output_dir / "robustness_summary.csv",
        "candidate_metrics": output_dir / "candidate_full_sample_diagnostic_metrics.csv",
        "pbo_report": output_dir / "pbo_report.csv",
        "nested_oos_performance": output_dir / "nested_oos_performance.csv",
        "promotion_decision": output_dir / "promotion_decision.csv",
        "constraint_check": output_dir / "constraint_check.csv",
        "report": output_dir / "WALK_FORWARD_REPORT.md",
        "self_check": output_dir / "SELF_CHECK_REPORT.md",
        "model_manifest": output_dir / "model_run_manifest.json",
        "model_manifest_check": output_dir / "model_run_manifest_check.csv",
    }
    model.write_csv(registry, paths["candidate_registry"])
    write_json({"candidates": registry.to_dict(orient="records")}, paths["candidate_registry_json"])
    model.write_csv(split_manifest, paths["split_manifest"])
    write_json({"splits": split_manifest.astype(str).to_dict(orient="records")}, paths["split_manifest_json"])
    for df, key in [
        (embargo_audit, "embargo_purge_audit"),
        (fold_scores, "inner_candidate_scores"),
        (selection, "nested_selection_by_fold"),
        (outer_oos, "outer_fold_oos_results"),
        (same_period, "same_period_baseline_comparison"),
        (cost_sensitivity, "cost_sensitivity"),
        (pbo_summary, "pbo_cscv_summary"),
        (pbo_folds, "pbo_cscv_splits"),
        (decision, "candidate_gate_decision"),
        (findings, "validation_findings"),
        (leakage, "leakage_checklist"),
        (robustness, "robustness_summary"),
        (candidate_metrics, "candidate_metrics"),
        (pbo_summary, "pbo_report"),
        (nested_performance, "nested_oos_performance"),
        (decision, "promotion_decision"),
        (checks, "constraint_check"),
    ]:
        model.write_csv(df, paths[key])
    write_text(make_report(nested_performance, pbo_summary, decision), paths["report"])
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    write_text("# HIRSSM V3.15 Self Check\n\n" + "\n".join(f"- {r.check}: {r.status} ({r.detail})" for r in checks.itertuples()), paths["self_check"])

    selected_artifacts = []
    for cost, nav in selected_navs.items():
        nav_path = output_dir / f"nav_nested_selected_candidate_{int(cost)}bps.csv"
        model.write_csv(nav, nav_path)
        selected_artifacts.append(nav_path)

    artifact_paths = [path for key, path in paths.items() if key not in {"model_manifest", "model_manifest_check"}] + selected_artifacts
    for variant in [item["variant"] for item in CANDIDATES]:
        artifact_paths.append(output_dir / f"target_weights_{variant}.csv")
        for cost in costs:
            artifact_paths.append(output_dir / f"nav_{variant}_{int(cost)}bps.csv")
            artifact_paths.append(output_dir / f"trades_{variant}_{int(cost)}bps.csv")

    perf10 = nested_performance[nested_performance["cost_bps"].astype(float).eq(10.0)].head(1)
    metrics = {
        "candidate_count": int(registry.shape[0]),
        "overall_decision": str(decision["overall_decision"].iloc[0]) if not decision.empty else "blocked",
        "pbo_10bps": float(pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)]["pbo"].iloc[0]) if not pbo_summary.empty else np.nan,
    }
    if not perf10.empty:
        row = perf10.iloc[0]
        metrics.update({"annual_return_10bps": float(row["annual_return"]), "sharpe_10bps": float(row["sharpe_no_rf"]), "annual_delta_vs_v310_10bps": float(row.get("annual_delta_vs_v310", np.nan))})

    manifest = build_model_run_manifest(
        root=root,
        task_id=TASK_ID,
        run_id=f"{TASK_ID}_run_001",
        model_version=MODEL_VERSION,
        baseline="HIRSSM V3.10 Clean Rank-Vol Core",
        status="success" if fail_count == 0 else "fail",
        started_at=start_time,
        finished_at=now_text(),
        output_dir=output_dir,
        command=["python", "-X", "utf8", "strategy_lab/hirssm_v3_15_breadth_overlay_harness.py"],
        argv={"costs": costs},
        code_paths=[root / "strategy_lab" / "hirssm_v3_15_breadth_overlay_harness.py", root / "strategy_lab" / "hirssm_v3_10_clean_baseline.py", root / "strategy_lab" / "hirssm_v3_11_nested_candidate_harness.py", root / "strategy_lab" / "hirssm_v2_model.py", root / "strategy_lab" / "hirssm_v2_walk_forward.py"],
        config_path=Path(args.config),
        data_paths=v311.collect_data_refs(root, config),
        artifact_paths=artifact_paths,
        selection={"baseline_variant": BASELINE_VARIANT, "candidate_count": int(registry.shape[0]), "selection_method": "v3_11_nested_prior_window", "full_sample_metrics_diagnostic_only": True},
        metrics=metrics,
        checks={"self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count},
        limitations=["Only V3.14-passing breadth signal implemented.", "Full-sample diagnostics are not promotion evidence."],
        risk_flags=["breadth_overlay_candidate"],
        next_decision="Apply cost-aware no-trade stabilization in V3.16 if V3.15 is not promoted.",
        handoff_summary="V3.15 implemented breadth overlays and generated nested/PBO validation artifacts.",
    )
    write_json(manifest, paths["model_manifest"])
    manifest_findings = validate_model_run_manifest(manifest)
    manifest_check = pd.DataFrame(manifest_findings) if manifest_findings else pd.DataFrame([{"severity": "pass", "field": "model_run_manifest", "message": "no failures"}])
    model.write_csv(manifest_check, paths["model_manifest_check"])
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")

    agent_report = agent_dir / "agent_report.md"
    agent_manifest_path = agent_dir / "agent_run_manifest.json"
    write_text(make_report(nested_performance, pbo_summary, decision), agent_report)
    for name, df in [("candidate_registry.csv", registry), ("promotion_decision.csv", decision), ("constraint_check.csv", checks), ("validation_findings.csv", findings), ("leakage_checklist.csv", leakage), ("robustness_summary.csv", robustness)]:
        model.write_csv(df, agent_dir / name)
    agent_artifacts = [agent_report, agent_dir / "candidate_registry.csv", agent_dir / "promotion_decision.csv", agent_dir / "constraint_check.csv", agent_dir / "validation_findings.csv", agent_dir / "leakage_checklist.csv", agent_dir / "robustness_summary.csv", agent_manifest_path]
    write_json(make_agent_manifest(start_time, agent_dir, agent_artifacts, metrics, fail_count + manifest_fail_count, warn_count), agent_manifest_path)

    result = {"model_version": MODEL_VERSION, "self_check_pass": fail_count == 0 and manifest_fail_count == 0, "fail_count": fail_count, "manifest_fail_count": manifest_fail_count, "metrics": metrics, "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 and manifest_fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
