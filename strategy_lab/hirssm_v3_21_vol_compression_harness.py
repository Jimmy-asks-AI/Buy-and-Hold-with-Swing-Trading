#!/usr/bin/env python
"""HIRSSM V3.21 volatility-compression reentry harness.

Implements only the V3.20 signal that passed pre-backtest validation:
``vol_compression_reentry``. Full-sample metrics are diagnostic; promotion is
controlled by V3.11-style nested/PBO gates.
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
V320_DIR = ROOT / "outputs" / "agent_runs" / "v3_20" / "rescue_signal_research"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_21_vol_compression_harness"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_21" / "vol_compression_harness"
TASK_ID = "20260527_v3_21_vol_compression_harness"
MODEL_VERSION = "HIRSSM V3.21 Vol Compression Reentry Harness"
BASELINE_VARIANT = v311.BASELINE_VARIANT


CANDIDATES = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol baseline.",
        "mode": "baseline",
        "cash_release": 0.0,
        "no_trade_band": 0.0,
    },
    {
        "variant": "vol_compression_reentry",
        "role": "candidate",
        "description": "Release cash when volatility compresses after stress and benchmark momentum is positive.",
        "mode": "vol_compression_reentry",
        "cash_release": 0.05,
        "no_trade_band": 0.0,
    },
    {
        "variant": "vol_compression_reentry_no_trade_3pct",
        "role": "candidate",
        "description": "Same reentry signal with 3pct no-trade band to reduce execution churn.",
        "mode": "vol_compression_reentry",
        "cash_release": 0.05,
        "no_trade_band": 0.03,
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


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
                    {"mode": item["mode"], "cash_release": item["cash_release"], "no_trade_band": item["no_trade_band"]},
                    sort_keys=True,
                ),
                "disabled_experts": disabled,
                "selection_source": "v3_20_pre_backtest_signal_gate",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": item["role"] == "candidate",
            }
        )
    return pd.DataFrame(rows)


def feature_panel(v320_dir: Path) -> pd.DataFrame:
    features = read_csv(v320_dir / "signal_feature_panel.csv")
    if features.empty:
        return features
    features["signal_date"] = pd.to_datetime(features["signal_date"])
    for col in ["vol_60", "vol_120", "drawdown_252", "benchmark_ret_20"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features["vol_compression_reentry_trigger"] = (
        (features["vol_60"] <= 0.85 * features["vol_120"])
        & (features["drawdown_252"] <= -0.10)
        & (features["benchmark_ret_20"] > 0)
    )
    return features[["signal_date", "vol_60", "vol_120", "drawdown_252", "benchmark_ret_20", "vol_compression_reentry_trigger"]]


def normalize_group(g: pd.DataFrame) -> pd.DataFrame:
    total = float(pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
    if total > 0:
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0) / total
    g["target_weight"] = g["weight"]
    return g


def ensure_cash_row(group: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    if group["asset"].eq("CASH").any():
        return group
    template = {col: "" for col in group.columns}
    template.update({"signal_date": signal_date, "asset": "CASH", "asset_type": "cash", "weight": 0.0, "target_weight": 0.0})
    return pd.concat([group, pd.DataFrame([template])], ignore_index=True)


def apply_reentry(base_targets: pd.DataFrame, features: pd.DataFrame, spec: dict[str, Any], config: dict) -> pd.DataFrame:
    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    if spec["mode"] == "baseline":
        targets["variant"] = spec["variant"]
        targets["vol_compression_reentry_trigger"] = False
        targets["no_trade_band"] = 0.0
        return targets
    targets = targets.merge(features, on="signal_date", how="left")
    rows = []
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = ensure_cash_row(group.copy(), signal_date)
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        triggered = bool(g["vol_compression_reentry_trigger"].fillna(False).any())
        if triggered:
            cash_idx = g.index[g["asset"].eq("CASH")][0]
            noncash_idx = g.index[~g["asset"].eq("CASH")]
            release = min(float(spec["cash_release"]), float(g.loc[cash_idx, "weight"]))
            noncash_sum = float(g.loc[noncash_idx, "weight"].sum())
            if release > 0 and noncash_sum > 0:
                g.loc[noncash_idx, "weight"] = g.loc[noncash_idx, "weight"] + release * g.loc[noncash_idx, "weight"] / noncash_sum
                g.loc[cash_idx, "weight"] = float(g.loc[cash_idx, "weight"]) - release
        g["reentry_cash_release"] = float(spec["cash_release"]) if triggered else 0.0
        g["no_trade_band"] = float(spec["no_trade_band"])
        rows.append(normalize_group(g))
    out = pd.concat(rows, ignore_index=True, sort=False)
    if float(spec["no_trade_band"]) > 0:
        out = apply_no_trade_band(out, float(spec["no_trade_band"]))
    out = v310.enforce_cash_cap(out, config)
    out = v310.enforce_turnover_cap(out, max_turnover=0.60)
    out["variant"] = spec["variant"]
    out["candidate_role"] = spec["role"]
    return out


def apply_no_trade_band(targets: pd.DataFrame, band: float) -> pd.DataFrame:
    rows = []
    prev: dict[str, float] = {}
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = group.copy()
        if prev:
            for idx, row in g.iterrows():
                asset = str(row["asset"])
                current = float(row["weight"])
                previous = float(prev.get(asset, 0.0))
                if abs(current - previous) < band:
                    g.loc[idx, "weight"] = previous
        g = normalize_group(g)
        prev = {str(row["asset"]): float(row["weight"]) for _, row in g.iterrows()}
        rows.append(g)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["no_trade_band_applied"] = band
    return out


def run_candidates(panel: dict[str, pd.DataFrame], config: dict, costs: list[float], output_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], dict[str, pd.DataFrame]]:
    base_targets = v310.build_targets(panel, config)
    features = feature_panel(V320_DIR)
    rows = []
    navs: dict[tuple[str, float], pd.DataFrame] = {}
    targets_by_variant: dict[str, pd.DataFrame] = {}
    for spec in CANDIDATES:
        targets = apply_reentry(base_targets, features, spec, config)
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
    perf10 = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1) if not performance.empty else pd.DataFrame()
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
    lines = [
        "# HIRSSM V3.21 Vol Compression Reentry Harness",
        "",
        "## Purpose",
        "",
        "Implement only the V3.20-passing volatility-compression reentry signal.",
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
        "version": "V3.21",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_21_vol_compression_harness.py",
        "config": {"config_path": str(CONFIG.as_posix()), "baseline_variant": BASELINE_VARIANT, "candidate_count": len(CANDIDATES)},
        "data_refs": ["outputs/agent_runs/v3_20/rescue_signal_research", "data_raw/index/akshare_csindex", "data_raw/index/akshare_sw_industry"],
        "code_refs": ["strategy_lab/hirssm_v3_21_vol_compression_harness.py", "strategy_lab/hirssm_v3_10_clean_baseline.py", "strategy_lab/hirssm_v3_11_nested_candidate_harness.py"],
        "output_dir": str(agent_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_20/rescue_signal_research", "configs/hirssm_v2_default.json", "data_raw/index"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Only V3.20-passing signal implemented.", "PBO applies to a small candidate set."],
        "risk_flags": ["vol_compression_signal_may_be_regime_specific", "full_sample_metrics_diagnostic_only"],
        "next_decision": "If rejected, send to V3.22 failure attribution rather than promotion.",
        "handoff_summary": "V3.21 implemented volatility-compression reentry candidates and emitted nested/PBO gate artifacts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.21 volatility-compression reentry harness.")
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
    write_text("# HIRSSM V3.21 Self Check\n\n" + "\n".join(f"- {r.check}: {r.status} ({r.detail})" for r in checks.itertuples()), paths["self_check"])

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
        command=["python", "-X", "utf8", "strategy_lab/hirssm_v3_21_vol_compression_harness.py"],
        argv={"costs": costs},
        code_paths=[root / "strategy_lab" / "hirssm_v3_21_vol_compression_harness.py", root / "strategy_lab" / "hirssm_v3_10_clean_baseline.py", root / "strategy_lab" / "hirssm_v3_11_nested_candidate_harness.py", root / "strategy_lab" / "hirssm_v2_model.py", root / "strategy_lab" / "hirssm_v2_walk_forward.py"],
        config_path=Path(args.config),
        data_paths=v311.collect_data_refs(root, config),
        artifact_paths=artifact_paths,
        selection={"baseline_variant": BASELINE_VARIANT, "candidate_count": int(registry.shape[0]), "selection_method": "v3_11_nested_prior_window", "full_sample_metrics_diagnostic_only": True},
        metrics=metrics,
        checks={"self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count},
        limitations=["Only V3.20-passing volatility compression signal implemented.", "Full-sample diagnostics are not promotion evidence."],
        risk_flags=["vol_compression_reentry_candidate"],
        next_decision="Run V3.22 failure attribution if not promoted.",
        handoff_summary="V3.21 implemented vol-compression reentry and generated nested/PBO validation artifacts.",
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
