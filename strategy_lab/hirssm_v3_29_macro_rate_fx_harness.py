#!/usr/bin/env python
"""HIRSSM V3.29 macro rate/FX constrained implementation harness.

V3.29 implements only the two V3.28 holdout-qualified macro signals:

- us_rate_shock_fx_stress_defense
- spread_repair_risk_on

The harness compares candidates with the frozen V3.10 baseline under
5/10/20bps costs, nested prior-window selection, and purged block PBO
diagnostics. Full-sample candidate metrics are diagnostic only.
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


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
BASELINE_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
V328_DIR = ROOT / "outputs" / "agent_runs" / "v3_28" / "macro_rate_fx_signal_validation"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_29_macro_rate_fx_harness"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_29" / "macro_rate_fx_harness"
TASK_ID = "20260527_v3_29_macro_rate_fx_harness"
MODEL_VERSION = "HIRSSM V3.29 Macro Rate/FX Harness"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"
HIGH_BETA_ASSETS = {"000985", "000300", "000905", "000852"}
STYLE_ASSETS = {"000985", "000922", "000016", "000905", "000852", "000300"}


CANDIDATES = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol baseline.",
        "mode": "baseline",
        "cash_shift": 0.0,
        "trigger_column": "",
        "cash_cap": 0.40,
    },
    {
        "variant": "us_rate_shock_fx_stress_defense",
        "role": "candidate",
        "description": "Move 4pct from high-beta/style sleeves to cash when US 10Y shocks higher and RMB weakens.",
        "mode": "stress_defense",
        "cash_shift": 0.04,
        "trigger_column": "us_rate_shock_fx_stress_defense_trigger",
        "cash_cap": 0.44,
    },
    {
        "variant": "spread_repair_risk_on",
        "role": "candidate",
        "description": "Release 4pct cash to style sleeves when China-US spread repairs and RMB stabilizes.",
        "mode": "risk_on_release",
        "cash_shift": 0.04,
        "trigger_column": "spread_repair_risk_on_trigger",
        "cash_cap": 0.40,
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def parse_costs(text: str) -> list[float]:
    costs = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not costs:
        raise ValueError("cost list is empty")
    return costs


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
                    {
                        "mode": item["mode"],
                        "cash_shift": item["cash_shift"],
                        "trigger_column": item["trigger_column"],
                        "cash_cap": item["cash_cap"],
                    },
                    sort_keys=True,
                ),
                "disabled_experts": disabled,
                "selection_source": "v3_28_holdout_qualified_macro_signal" if item["role"] == "candidate" else "v3_10_clean_baseline",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": False,
            }
        )
    return pd.DataFrame(rows)


def load_macro_triggers() -> pd.DataFrame:
    features = read_csv(V328_DIR / "macro_rate_fx_signal_feature_panel.csv")
    if features.empty:
        raise FileNotFoundError(V328_DIR / "macro_rate_fx_signal_feature_panel.csv")
    features["signal_date"] = pd.to_datetime(features["signal_date"])
    features["us_rate_shock_fx_stress_defense_trigger"] = (
        (pd.to_numeric(features["us_10y_chg60_lag1"], errors="coerce") >= 0.25)
        & (pd.to_numeric(features["usdcny_ret60_lag1"], errors="coerce") >= 0.005)
    )
    features["spread_repair_risk_on_trigger"] = (
        (pd.to_numeric(features["cn_us_rate_spread_chg60_lag1"], errors="coerce") >= 0.20)
        & (pd.to_numeric(features["usdcny_ret60_lag1"], errors="coerce") <= 0.0)
    )
    cols = [
        "signal_date",
        "us_rate_shock_fx_stress_defense_trigger",
        "spread_repair_risk_on_trigger",
        "us_10y_chg60_lag1",
        "usdcny_ret60_lag1",
        "cn_us_rate_spread_chg60_lag1",
    ]
    return features[cols].copy()


def load_baseline_targets() -> pd.DataFrame:
    targets = read_csv(BASELINE_DIR / "target_weights.csv")
    if targets.empty:
        raise FileNotFoundError(BASELINE_DIR / "target_weights.csv")
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    targets["weight"] = pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0)
    if "target_weight" not in targets.columns:
        targets["target_weight"] = targets["weight"]
    return targets


def normalize_group(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(g["weight"].sum())
    if total > 0:
        g["weight"] = g["weight"] / total
    g["target_weight"] = g["weight"]
    return g


def ensure_cash_row(group: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    if group["asset"].astype(str).eq("CASH").any():
        return group
    template = {col: "" for col in group.columns}
    template.update({"signal_date": signal_date, "asset": "CASH", "asset_type": "cash", "weight": 0.0, "target_weight": 0.0})
    return pd.concat([group, pd.DataFrame([template])], ignore_index=True)


def cap_cash(group: pd.DataFrame, max_cash: float) -> pd.DataFrame:
    g = group.copy()
    if not g["asset"].astype(str).eq("CASH").any():
        return normalize_group(g)
    cash_idx = g.index[g["asset"].astype(str).eq("CASH")][0]
    cash_weight = float(g.loc[cash_idx, "weight"])
    if cash_weight <= max_cash + 1e-12:
        return normalize_group(g)
    excess = cash_weight - max_cash
    noncash = g.index[~g["asset"].astype(str).eq("CASH")]
    noncash_sum = float(g.loc[noncash, "weight"].sum())
    if noncash_sum > 0:
        g.loc[noncash, "weight"] = g.loc[noncash, "weight"] + excess * g.loc[noncash, "weight"] / noncash_sum
        g.loc[cash_idx, "weight"] = max_cash
    return normalize_group(g)


def move_to_cash(group: pd.DataFrame, amount: float, high_beta_assets: set[str], max_cash: float) -> pd.DataFrame:
    g = group.copy()
    cash_idx = g.index[g["asset"].astype(str).eq("CASH")][0]
    current_cash = float(g.loc[cash_idx, "weight"])
    room = max(max_cash - current_cash, 0.0)
    shift = min(float(amount), room)
    if shift <= 0:
        return normalize_group(g)
    high_beta = g.index[g["asset"].astype(str).isin(high_beta_assets)]
    source_sum = float(g.loc[high_beta, "weight"].sum())
    if source_sum < shift - 1e-12:
        fallback = g.index[(~g["asset"].astype(str).eq("CASH")) & (~g.index.isin(high_beta))]
        fallback_sum = float(g.loc[fallback, "weight"].sum())
    else:
        fallback = pd.Index([])
        fallback_sum = 0.0

    take_from_high = min(shift, source_sum)
    if take_from_high > 0 and source_sum > 0:
        g.loc[high_beta, "weight"] = g.loc[high_beta, "weight"] - take_from_high * g.loc[high_beta, "weight"] / source_sum
    remaining = shift - take_from_high
    if remaining > 0 and fallback_sum > 0:
        g.loc[fallback, "weight"] = g.loc[fallback, "weight"] - remaining * g.loc[fallback, "weight"] / fallback_sum
    actual_shift = shift - max(remaining - fallback_sum, 0.0)
    g.loc[cash_idx, "weight"] = current_cash + actual_shift
    return cap_cash(g, max_cash)


def release_cash(group: pd.DataFrame, amount: float, target_assets: set[str]) -> pd.DataFrame:
    g = group.copy()
    cash_idx = g.index[g["asset"].astype(str).eq("CASH")][0]
    current_cash = float(g.loc[cash_idx, "weight"])
    release = min(float(amount), current_cash)
    if release <= 0:
        return normalize_group(g)
    target_idx = g.index[g["asset"].astype(str).isin(target_assets)]
    target_sum = float(g.loc[target_idx, "weight"].sum())
    if target_sum <= 0:
        target_idx = g.index[~g["asset"].astype(str).eq("CASH")]
        target_sum = float(g.loc[target_idx, "weight"].sum())
    if target_sum > 0:
        g.loc[target_idx, "weight"] = g.loc[target_idx, "weight"] + release * g.loc[target_idx, "weight"] / target_sum
        g.loc[cash_idx, "weight"] = current_cash - release
    return normalize_group(g)


def apply_macro_overlay(base_targets: pd.DataFrame, triggers: pd.DataFrame, spec: dict[str, Any], config: dict) -> pd.DataFrame:
    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    if spec["mode"] == "baseline":
        out = targets.copy()
        out["variant"] = spec["variant"]
        out["candidate_role"] = spec["role"]
        out["macro_trigger_active"] = False
        return out

    targets = targets.merge(triggers, on="signal_date", how="left")
    rows = []
    trigger_col = str(spec["trigger_column"])
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = ensure_cash_row(group.copy(), signal_date)
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        triggered = bool(g[trigger_col].fillna(False).any()) if trigger_col in g.columns else False
        if triggered and spec["mode"] == "stress_defense":
            g = move_to_cash(g, float(spec["cash_shift"]), HIGH_BETA_ASSETS, float(spec["cash_cap"]))
        elif triggered and spec["mode"] == "risk_on_release":
            g = release_cash(g, float(spec["cash_shift"]), STYLE_ASSETS)
            g = cap_cash(g, float(spec["cash_cap"]))
        else:
            g = normalize_group(g)
        g["macro_trigger_active"] = triggered
        g["macro_overlay_mode"] = spec["mode"]
        rows.append(g)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = v310.enforce_turnover_cap(out, max_turnover=0.60)
    out["variant"] = spec["variant"]
    out["candidate_role"] = spec["role"]
    out["macro_cash_shift"] = float(spec["cash_shift"])
    out["macro_cash_cap"] = float(spec["cash_cap"])
    return out


def run_candidates(
    panel: dict[str, pd.DataFrame],
    config: dict,
    costs: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], dict[str, pd.DataFrame], dict[tuple[str, float], pd.DataFrame]]:
    base_targets = load_baseline_targets()
    triggers = load_macro_triggers()
    rows = []
    navs: dict[tuple[str, float], pd.DataFrame] = {}
    targets_by_variant: dict[str, pd.DataFrame] = {}
    trades: dict[tuple[str, float], pd.DataFrame] = {}
    for spec in CANDIDATES:
        targets = apply_macro_overlay(base_targets, triggers, spec, config)
        targets_by_variant[spec["variant"]] = targets
        model.write_csv(targets, output_dir / f"target_weights_{spec['variant']}.csv")
        for cost in costs:
            bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
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
            trades[(spec["variant"], float(cost))] = bt["trades"]
            model.write_csv(nav, output_dir / f"nav_{spec['variant']}_{int(cost)}bps.csv")
            model.write_csv(bt["trades"], output_dir / f"trades_{spec['variant']}_{int(cost)}bps.csv")
    metrics = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return metrics, navs, targets_by_variant, trades


def target_integrity(targets_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, targets in targets_by_variant.items():
        weights = pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0)
        sums = targets.assign(_w=weights).groupby("signal_date")["_w"].sum()
        min_weight = float(weights.min()) if not weights.empty else np.nan
        max_sum_error = float((sums - 1.0).abs().max()) if not sums.empty else np.nan
        rows.append(
            {
                "check": f"target_integrity_{variant}",
                "status": "pass" if not targets.empty and min_weight >= -1e-10 and max_sum_error <= 1e-6 else "fail",
                "detail": f"rows={targets.shape[0]}; min_weight={min_weight:.8f}; max_sum_error={max_sum_error:.8f}",
            }
        )
    return pd.DataFrame(rows)


def trigger_diagnostics(targets_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in CANDIDATES:
        targets = targets_by_variant.get(spec["variant"], pd.DataFrame())
        if targets.empty:
            continue
        by_date = targets.drop_duplicates("signal_date")
        active = by_date["macro_trigger_active"].astype(str).str.lower().eq("true") if "macro_trigger_active" in by_date.columns else pd.Series(False, index=by_date.index)
        cash = targets[targets["asset"].astype(str).eq("CASH")].copy()
        rows.append(
            {
                "variant": spec["variant"],
                "triggered_months": int(active.sum()),
                "total_months": int(by_date.shape[0]),
                "trigger_rate": float(active.mean()) if not active.empty else np.nan,
                "avg_cash_weight": float(pd.to_numeric(cash["weight"], errors="coerce").mean()) if not cash.empty else np.nan,
                "max_cash_weight": float(pd.to_numeric(cash["weight"], errors="coerce").max()) if not cash.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_report(performance: pd.DataFrame, pbo: pd.DataFrame, decision: pd.DataFrame, trigger_diag: pd.DataFrame) -> str:
    perf10 = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1) if not performance.empty else pd.DataFrame()
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
    decision_text = decision["overall_decision"].iloc[0] if not decision.empty else "blocked"
    lines = [
        "# HIRSSM V3.29 Macro Rate/FX Harness",
        "",
        "## Purpose",
        "",
        "Implement only V3.28 holdout-qualified macro rate/FX candidates in a constrained harness.",
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
                f"- Drawdown delta vs V3.10: {float(item.get('drawdown_delta_vs_v310', np.nan)):.6f}",
            ]
        )
    if not pbo10.empty:
        lines.extend(["", "## PBO", "", f"- 10bps PBO: {float(pbo10['pbo'].iloc[0]):.6f}", f"- 10bps PBO status: {pbo10['pbo_status'].iloc[0]}"])
    if not trigger_diag.empty:
        lines.extend(["", "## Trigger Diagnostics", ""])
        for _, row in trigger_diag.iterrows():
            lines.append(f"- {row['variant']}: triggered_months={int(row['triggered_months'])}, avg_cash={float(row['avg_cash_weight']):.4f}")
    lines.extend(["", "## Decision", "", f"- Overall decision: {decision_text}", "- Full-sample candidate metrics remain diagnostic only."])
    return "\n".join(lines)


def make_failure_cases(decision: pd.DataFrame, pbo: pd.DataFrame) -> str:
    lines = [
        "# V3.29 Failure Cases",
        "",
        "- Macro overlays may add value in signal validation but fail after turnover and portfolio constraints.",
        "- If selection picks baseline most of the time, the candidate lacks investable marginal contribution.",
        "- A low PBO does not remove macro regime-shift risk.",
        "",
    ]
    if not decision.empty:
        lines.append("## Gate Decisions")
        lines.append("")
        for _, row in decision.iterrows():
            lines.append(
                f"- {int(float(row['cost_bps']))}bps: {row['decision']}; annual_delta={float(row['annual_delta_vs_v310']):.6f}; "
                f"drawdown_delta={float(row['drawdown_delta_vs_v310']):.6f}; pbo={float(row['pbo']):.6f}"
            )
    if not pbo.empty:
        lines.extend(["", "## PBO Summary", ""])
        for _, row in pbo.iterrows():
            lines.append(f"- {int(float(row['cost_bps']))}bps: pbo={float(row['pbo']):.6f}, status={row['pbo_status']}")
    return "\n".join(lines)


def make_agent_manifest(start_time: str, agent_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.29",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_29_macro_rate_fx_harness.py",
        "config": {
            "baseline_variant": BASELINE_VARIANT,
            "candidate_count": len(CANDIDATES),
            "costs": "5,10,20",
            "default_promotion_authorized": False,
        },
        "data_refs": [
            "outputs/hirssm_v3_10_clean_baseline/target_weights.csv",
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation",
            "data_raw/index",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_29_macro_rate_fx_harness.py",
            "strategy_lab/hirssm_v3_10_clean_baseline.py",
            "strategy_lab/hirssm_v3_11_nested_candidate_harness.py",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
        ],
        "output_dir": rel(agent_dir),
        "allowed_inputs": [
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation",
            "configs/hirssm_v2_default.json",
            "data_raw/index",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Only V3.28 holdout-qualified macro signals are implemented.",
            "Default promotion is still blocked pending review.",
            "Full-sample metrics remain diagnostic only.",
        ],
        "risk_flags": [
            "macro_regime_shift",
            "portfolio_constraint_dilution",
            "small_candidate_set_pbo",
        ],
        "next_decision": "Use V3.30 for failure attribution or constrained promotion review; do not promote automatically.",
        "handoff_summary": "V3.29 implemented two macro rate/FX candidates and generated nested/PBO validation artifacts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.29 macro rate/FX constrained harness.")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--agent-output-dir", default=str(AGENT_OUTPUT_DIR))
    parser.add_argument("--costs", default="5,10,20")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--inner-validation-years", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--embargo-days", type=int, default=21)
    parser.add_argument("--pbo-blocks", type=int, default=10)
    parser.add_argument("--pbo-train-blocks", type=int, default=5)
    parser.add_argument("--pbo-purge-days", type=int, default=63)
    args = parser.parse_args()

    start_time = now_text()
    config = model.read_json(Path(args.config))
    costs = parse_costs(args.costs)
    output_dir = Path(args.output_dir)
    agent_dir = Path(args.agent_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    panel = wf.build_panel(model, ROOT, config, None, None)
    registry = candidate_registry(config)
    candidate_metrics, navs, targets_by_variant, trades = run_candidates(panel, config, costs, output_dir)
    selection, fold_scores, selected_navs, nested_performance = v311.nested_walk_forward(
        navs=navs,
        costs=costs,
        lookback_years=args.lookback_years,
        inner_validation_years=args.inner_validation_years,
        min_train_days=args.min_train_days,
        embargo_days=args.embargo_days,
    )
    pbo_folds, pbo_summary = v311.purged_block_pbo(
        navs=navs,
        costs=costs,
        n_blocks=args.pbo_blocks,
        train_blocks=args.pbo_train_blocks,
        purge_days=args.pbo_purge_days,
        embargo_days=args.embargo_days,
    )
    decision = v311.promotion_decision(nested_performance, pbo_summary, selection)
    split_manifest = v311.split_manifest_from_selection(selection)
    embargo_audit = v311.embargo_purge_audit(split_manifest)
    outer_oos = v311.outer_fold_oos_results(selection)
    same_period = v311.same_period_baseline_comparison(nested_performance)
    cost_sensitivity = v311.cost_sensitivity_table(nested_performance, decision)
    trigger_diag = trigger_diagnostics(targets_by_variant)
    checks = pd.concat(
        [
            v311.build_constraint_checks(registry=registry, selection=selection, performance=nested_performance, pbo_summary=pbo_summary, decision=decision),
            target_integrity(targets_by_variant),
            pd.DataFrame(
                [
                    {
                        "check": "only_v3_28_holdout_candidates_implemented",
                        "status": "pass" if set(registry.loc[registry["role"].eq("candidate"), "variant"]) == {"us_rate_shock_fx_stress_defense", "spread_repair_risk_on"} else "fail",
                        "detail": ",".join(registry.loc[registry["role"].eq("candidate"), "variant"].astype(str)),
                    },
                    {
                        "check": "default_promotion_not_auto_authorized",
                        "status": "pass",
                        "detail": "manual review required after V3.29",
                    },
                ]
            ),
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
        "nested_oos_performance": output_dir / "nested_oos_performance.csv",
        "promotion_decision": output_dir / "promotion_decision.csv",
        "constraint_check": output_dir / "constraint_check.csv",
        "trigger_diagnostics": output_dir / "trigger_diagnostics.csv",
        "pbo_report": output_dir / "pbo_report.csv",
        "report": output_dir / "WALK_FORWARD_REPORT.md",
        "failure_cases": output_dir / "REGIME_FAILURE_CASES.md",
        "self_check": output_dir / "SELF_CHECK_REPORT.md",
        "changed_files": output_dir / "changed_files.txt",
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
        (nested_performance, "nested_oos_performance"),
        (decision, "promotion_decision"),
        (checks, "constraint_check"),
        (trigger_diag, "trigger_diagnostics"),
        (pbo_summary, "pbo_report"),
    ]:
        model.write_csv(df, paths[key])

    write_text(make_report(nested_performance, pbo_summary, decision, trigger_diag), paths["report"])
    write_text(make_failure_cases(decision, pbo_summary), paths["failure_cases"])
    write_text("# HIRSSM V3.29 Self Check\n\n" + "\n".join(f"- {r.check}: {r.status} ({r.detail})" for r in checks.itertuples()), paths["self_check"])

    selected_artifacts: list[Path] = []
    for cost, nav in selected_navs.items():
        nav_path = output_dir / f"nav_nested_selected_candidate_{int(cost)}bps.csv"
        model.write_csv(nav, nav_path)
        selected_artifacts.append(nav_path)
        if not nav.empty:
            yearly_path = output_dir / f"yearly_returns_nested_selected_candidate_{int(cost)}bps.csv"
            regime_path = output_dir / f"regime_returns_nested_selected_candidate_{int(cost)}bps.csv"
            model.write_csv(model.yearly_returns(nav), yearly_path)
            model.write_csv(model.regime_returns(nav, panel["regimes"]), regime_path)
            selected_artifacts.extend([yearly_path, regime_path])

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    perf10 = nested_performance[nested_performance["cost_bps"].astype(float).eq(10.0)].head(1)
    decision10 = decision[decision["cost_bps"].astype(float).eq(10.0)].head(1)
    metrics = {
        "candidate_count": int(registry.shape[0]),
        "implemented_signal_count": int((registry["role"] == "candidate").sum()),
        "walk_forward_selected_year_count": int(selection["selection_status"].eq("selected_by_prior_window").sum()) if not selection.empty else 0,
        "overall_decision": str(decision["overall_decision"].iloc[0]) if not decision.empty else "blocked",
        "pbo_10bps": float(pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)]["pbo"].iloc[0]) if not pbo_summary.empty and pbo_summary["cost_bps"].astype(float).eq(10.0).any() else np.nan,
    }
    if not perf10.empty:
        row = perf10.iloc[0]
        metrics.update(
            {
                "annual_return_10bps": float(row["annual_return"]),
                "sharpe_10bps": float(row["sharpe_no_rf"]),
                "max_drawdown_10bps": float(row["max_drawdown"]),
                "annual_delta_vs_v310_10bps": float(row.get("annual_delta_vs_v310", np.nan)),
                "drawdown_delta_vs_v310_10bps": float(row.get("drawdown_delta_vs_v310", np.nan)),
            }
        )
    if not decision10.empty:
        metrics["gate_decision_10bps"] = str(decision10["decision"].iloc[0])

    artifact_paths = list(paths.values()) + selected_artifacts
    for variant in [item["variant"] for item in CANDIDATES]:
        artifact_paths.append(output_dir / f"target_weights_{variant}.csv")
        for cost in costs:
            artifact_paths.append(output_dir / f"nav_{variant}_{int(cost)}bps.csv")
            artifact_paths.append(output_dir / f"trades_{variant}_{int(cost)}bps.csv")
    artifact_paths = list(dict.fromkeys(artifact_paths))
    write_text("\n".join(rel(path) for path in artifact_paths), paths["changed_files"])

    agent_report = agent_dir / "agent_report.md"
    agent_manifest_path = agent_dir / "agent_run_manifest.json"
    write_text(make_report(nested_performance, pbo_summary, decision, trigger_diag), agent_report)
    for name, df in [
        ("candidate_registry.csv", registry),
        ("promotion_decision.csv", decision),
        ("constraint_check.csv", checks),
        ("validation_findings.csv", findings),
        ("leakage_checklist.csv", leakage),
        ("robustness_summary.csv", robustness),
        ("trigger_diagnostics.csv", trigger_diag),
    ]:
        model.write_csv(df, agent_dir / name)
    agent_artifacts = [
        agent_report,
        agent_dir / "candidate_registry.csv",
        agent_dir / "promotion_decision.csv",
        agent_dir / "constraint_check.csv",
        agent_dir / "validation_findings.csv",
        agent_dir / "leakage_checklist.csv",
        agent_dir / "robustness_summary.csv",
        agent_dir / "trigger_diagnostics.csv",
        agent_manifest_path,
    ]
    write_json(make_agent_manifest(start_time, agent_dir, agent_artifacts, metrics, fail_count, warn_count), agent_manifest_path)

    result = {
        "model_version": MODEL_VERSION,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "metrics": metrics,
        "output_dir": rel(output_dir),
        "agent_output_dir": rel(agent_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
