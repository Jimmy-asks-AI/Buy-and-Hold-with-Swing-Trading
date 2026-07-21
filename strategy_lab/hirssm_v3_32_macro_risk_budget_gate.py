#!/usr/bin/env python
"""HIRSSM V3.32 macro risk-budget gate harness.

V3.31 showed that the V3.29 macro overlay was too small and unstable. V3.32
tests a predeclared alternative: macro signals gate the existing risky budget
instead of adding or subtracting a fixed 4pct cash sleeve.

This is still a validation harness. It does not promote any candidate unless
nested OOS, cost, and PBO gates pass.
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
import hirssm_v3_29_macro_rate_fx_harness as v329


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
BASELINE_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
V328_DIR = ROOT / "outputs" / "agent_runs" / "v3_28" / "macro_rate_fx_signal_validation"
V331_DIR = ROOT / "outputs" / "agent_runs" / "v3_31" / "selected_year_regime_attribution"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_32_macro_risk_budget_gate"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_32" / "macro_risk_budget_gate"
TASK_ID = "20260527_v3_32_macro_risk_budget_gate"
MODEL_VERSION = "HIRSSM V3.32 Macro Risk-Budget Gate"
BASELINE_VARIANT = v311.BASELINE_VARIANT


CANDIDATES = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol baseline.",
        "mode": "baseline",
    },
    {
        "variant": "stress_budget_gate",
        "role": "candidate",
        "description": "When US 10Y shocks higher and RMB weakens, cut 25pct of current risky budget and move it to cash.",
        "mode": "stress_only",
        "stress_risky_budget_cut": 0.25,
        "stress_cash_cap": 0.55,
        "repair_cash_release_fraction": 0.0,
        "repair_cash_floor": 0.10,
        "repair_allowed_states": "",
    },
    {
        "variant": "state_confirmed_dual_budget_gate",
        "role": "candidate",
        "description": "Stress gate plus conservative repair release only when state is range-bound or risk-on-trend.",
        "mode": "stress_plus_state_confirmed_repair",
        "stress_risky_budget_cut": 0.25,
        "stress_cash_cap": 0.55,
        "repair_cash_release_fraction": 0.25,
        "repair_cash_floor": 0.10,
        "repair_allowed_states": "range_bound,risk_on_trend",
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


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def candidate_registry(config: dict) -> pd.DataFrame:
    disabled = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    rows = []
    for item in CANDIDATES:
        params = {
            key: item.get(key)
            for key in [
                "mode",
                "stress_risky_budget_cut",
                "stress_cash_cap",
                "repair_cash_release_fraction",
                "repair_cash_floor",
                "repair_allowed_states",
            ]
            if key in item
        }
        rows.append(
            {
                "variant": item["variant"],
                "role": item["role"],
                "description": item["description"],
                "multipliers_json": json.dumps(params, sort_keys=True),
                "disabled_experts": disabled,
                "selection_source": "v3_31_predeclared_macro_risk_budget_hypothesis" if item["role"] == "candidate" else "v3_10_clean_baseline",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": False,
            }
        )
    return pd.DataFrame(rows)


def load_baseline_targets() -> pd.DataFrame:
    targets = read_csv(BASELINE_DIR / "target_weights.csv")
    if targets.empty:
        raise FileNotFoundError(BASELINE_DIR / "target_weights.csv")
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    targets["weight"] = pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0)
    if "target_weight" not in targets.columns:
        targets["target_weight"] = targets["weight"]
    return targets


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


def scale_to_cash_target(group: pd.DataFrame, target_cash: float) -> pd.DataFrame:
    g = group.copy()
    g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cash_idx = g.index[g["asset"].astype(str).eq("CASH")][0]
    noncash = g.index[~g["asset"].astype(str).eq("CASH")]
    target_cash = float(np.clip(target_cash, 0.0, 0.95))
    noncash_sum = float(g.loc[noncash, "weight"].sum())
    g.loc[cash_idx, "weight"] = target_cash
    if noncash_sum > 0:
        g.loc[noncash, "weight"] = g.loc[noncash, "weight"] * ((1.0 - target_cash) / noncash_sum)
    return normalize_group(g)


def apply_risk_budget_gate(base_targets: pd.DataFrame, triggers: pd.DataFrame, spec: dict[str, Any], config: dict) -> pd.DataFrame:
    targets = base_targets.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    if spec["mode"] == "baseline":
        out = targets.copy()
        out["variant"] = spec["variant"]
        out["candidate_role"] = spec["role"]
        out["macro_trigger_active"] = False
        out["stress_gate_active"] = False
        out["repair_gate_active"] = False
        out["macro_budget_mode"] = spec["mode"]
        return out

    targets = targets.merge(triggers, on="signal_date", how="left")
    rows = []
    allowed_states = {
        item.strip()
        for item in str(spec.get("repair_allowed_states", "")).split(",")
        if item.strip()
    }
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = ensure_cash_row(group.copy(), signal_date)
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        cash_idx = g.index[g["asset"].astype(str).eq("CASH")][0]
        current_cash = float(g.loc[cash_idx, "weight"])
        risky_budget = max(1.0 - current_cash, 0.0)
        state = str(g["state"].dropna().astype(str).iloc[0]) if "state" in g.columns and g["state"].dropna().size else ""
        stress_trigger = bool(g["us_rate_shock_fx_stress_defense_trigger"].fillna(False).any()) if "us_rate_shock_fx_stress_defense_trigger" in g.columns else False
        repair_trigger = bool(g["spread_repair_risk_on_trigger"].fillna(False).any()) if "spread_repair_risk_on_trigger" in g.columns else False
        stress_gate_active = False
        repair_gate_active = False

        if stress_trigger:
            cut = float(spec.get("stress_risky_budget_cut", 0.0)) * risky_budget
            target_cash = min(current_cash + cut, float(spec.get("stress_cash_cap", 0.55)))
            g = scale_to_cash_target(g, target_cash)
            stress_gate_active = True
        elif spec["mode"] == "stress_plus_state_confirmed_repair" and repair_trigger and state in allowed_states:
            release = float(spec.get("repair_cash_release_fraction", 0.0)) * current_cash
            target_cash = max(current_cash - release, float(spec.get("repair_cash_floor", 0.10)))
            g = scale_to_cash_target(g, target_cash)
            repair_gate_active = True
        else:
            g = normalize_group(g)

        g["macro_trigger_active"] = stress_gate_active or repair_gate_active
        g["stress_gate_active"] = stress_gate_active
        g["repair_gate_active"] = repair_gate_active
        g["macro_budget_mode"] = spec["mode"]
        rows.append(g)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = v310.enforce_turnover_cap(out, max_turnover=0.60)
    out["variant"] = spec["variant"]
    out["candidate_role"] = spec["role"]
    out["stress_risky_budget_cut"] = float(spec.get("stress_risky_budget_cut", 0.0))
    out["stress_cash_cap"] = float(spec.get("stress_cash_cap", 0.0))
    out["repair_cash_release_fraction"] = float(spec.get("repair_cash_release_fraction", 0.0))
    out["repair_cash_floor"] = float(spec.get("repair_cash_floor", 0.0))
    out["repair_allowed_states"] = str(spec.get("repair_allowed_states", ""))
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
        targets = apply_risk_budget_gate(base_targets, triggers, spec, config)
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


def gate_exposure_diagnostics(targets_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    base = targets_by_variant.get(BASELINE_VARIANT, pd.DataFrame()).copy()
    base_cash = (
        base[base["asset"].astype(str).eq("CASH")]
        .assign(signal_date=lambda x: pd.to_datetime(x["signal_date"]))
        .loc[:, ["signal_date", "weight"]]
        .rename(columns={"weight": "baseline_cash_weight"})
    )
    for spec in CANDIDATES:
        targets = targets_by_variant.get(spec["variant"], pd.DataFrame())
        if targets.empty:
            continue
        by_date = targets.drop_duplicates("signal_date").copy()
        by_date["signal_date"] = pd.to_datetime(by_date["signal_date"])
        cash = (
            targets[targets["asset"].astype(str).eq("CASH")]
            .assign(signal_date=lambda x: pd.to_datetime(x["signal_date"]))
            .loc[:, ["signal_date", "weight"]]
            .rename(columns={"weight": "cash_weight"})
        )
        diag = by_date.merge(cash, on="signal_date", how="left").merge(base_cash, on="signal_date", how="left")
        for col in ["macro_trigger_active", "stress_gate_active", "repair_gate_active"]:
            if col not in diag.columns:
                diag[col] = False
            diag[col] = diag[col].astype(str).str.lower().eq("true")
        diag["cash_delta_vs_baseline"] = pd.to_numeric(diag["cash_weight"], errors="coerce") - pd.to_numeric(diag["baseline_cash_weight"], errors="coerce")
        rows.append(
            {
                "variant": spec["variant"],
                "mode": spec["mode"],
                "active_months": int(diag["macro_trigger_active"].sum()),
                "stress_active_months": int(diag["stress_gate_active"].sum()),
                "repair_active_months": int(diag["repair_gate_active"].sum()),
                "total_months": int(diag.shape[0]),
                "active_rate": float(diag["macro_trigger_active"].mean()) if not diag.empty else np.nan,
                "avg_cash_weight": float(pd.to_numeric(diag["cash_weight"], errors="coerce").mean()),
                "max_cash_weight": float(pd.to_numeric(diag["cash_weight"], errors="coerce").max()),
                "avg_cash_delta_vs_baseline": float(pd.to_numeric(diag["cash_delta_vs_baseline"], errors="coerce").mean()),
                "avg_abs_cash_delta_vs_baseline_when_active": float(
                    pd.to_numeric(diag.loc[diag["macro_trigger_active"], "cash_delta_vs_baseline"], errors="coerce").abs().mean()
                )
                if diag["macro_trigger_active"].any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_report(performance: pd.DataFrame, pbo: pd.DataFrame, decision: pd.DataFrame, gate_diag: pd.DataFrame) -> str:
    lines = [
        "# HIRSSM V3.32 Macro Risk-Budget Gate",
        "",
        "## Decision",
        "",
    ]
    overall = str(decision["overall_decision"].iloc[0]) if not decision.empty and "overall_decision" in decision.columns else "blocked"
    lines.append(f"- Overall decision: `{overall}`.")
    lines.append("- V3.32 is a validation harness; default promotion still requires all nested/PBO gates.")
    lines.extend(["", "## 10bps Result", ""])
    perf10 = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1) if not performance.empty else pd.DataFrame()
    if not perf10.empty:
        row = perf10.iloc[0]
        lines.append(f"- Annual return: `{float(row['annual_return']):.4%}`.")
        lines.append(f"- Sharpe: `{float(row['sharpe_no_rf']):.4f}`.")
        lines.append(f"- Max drawdown: `{float(row['max_drawdown']):.4%}`.")
        lines.append(f"- Annual delta vs V3.10: `{float(row.get('annual_delta_vs_v310', np.nan)):.4%}`.")
        lines.append(f"- Drawdown delta vs V3.10: `{float(row.get('drawdown_delta_vs_v310', np.nan)):.4%}`.")
    pbo10 = pbo[pbo["cost_bps"].astype(float).eq(10.0)].head(1) if not pbo.empty else pd.DataFrame()
    if not pbo10.empty:
        lines.extend(["", "## PBO", "", f"- 10bps PBO: `{float(pbo10['pbo'].iloc[0]):.6f}`.", f"- 10bps PBO status: `{pbo10['pbo_status'].iloc[0]}`."])
    if not gate_diag.empty:
        lines.extend(["", "## Gate Exposure", ""])
        for _, row in gate_diag.iterrows():
            lines.append(
                f"- `{row['variant']}`: active months `{int(row['active_months'])}`, "
                f"stress `{int(row['stress_active_months'])}`, repair `{int(row['repair_active_months'])}`, "
                f"active cash delta `{float(row['avg_abs_cash_delta_vs_baseline_when_active']):.2%}`."
            )
    lines.extend(["", "## Next Step", "", "Use V3.33 for either failure attribution or a tightly constrained promotion review; do not promote automatically."])
    return "\n".join(lines)


def make_failure_cases(decision: pd.DataFrame, pbo: pd.DataFrame, gate_diag: pd.DataFrame) -> str:
    lines = ["# HIRSSM V3.32 Failure Cases", ""]
    if not decision.empty:
        for _, row in decision.iterrows():
            if str(row.get("decision", "")) != "promote_candidate":
                lines.append(
                    f"- {int(float(row['cost_bps']))}bps rejected: annual_delta={float(row['annual_delta_vs_v310']):.6f}; "
                    f"drawdown_delta={float(row['drawdown_delta_vs_v310']):.6f}; pbo={float(row['pbo']):.6f}"
                )
    if not pbo.empty:
        lines.extend(["", "## PBO Details"])
        for _, row in pbo.iterrows():
            lines.append(f"- {int(float(row['cost_bps']))}bps: pbo={float(row['pbo']):.6f}, status={row['pbo_status']}")
    if not gate_diag.empty:
        lines.extend(["", "## Gate Exposure Risk"])
        for _, row in gate_diag.iterrows():
            lines.append(
                f"- {row['variant']}: active_rate={float(row['active_rate']):.2%}; "
                f"avg_abs_cash_delta_when_active={float(row['avg_abs_cash_delta_vs_baseline_when_active']):.2%}"
            )
    return "\n".join(lines)


def self_check_rows(registry: pd.DataFrame, checks: pd.DataFrame, decision: pd.DataFrame, pbo: pd.DataFrame) -> pd.DataFrame:
    candidate_set = set(registry.loc[registry["role"].eq("candidate"), "variant"].astype(str))
    expected = {"stress_budget_gate", "state_confirmed_dual_budget_gate"}
    rows = [
        {
            "check": "predeclared_candidates_only",
            "status": "pass" if candidate_set == expected else "fail",
            "detail": ",".join(sorted(candidate_set)),
        },
        {
            "check": "no_default_promotion_without_nested_pbo",
            "status": "pass" if not decision.empty and "overall_decision" in decision.columns else "fail",
            "detail": str(decision["overall_decision"].iloc[0]) if not decision.empty and "overall_decision" in decision.columns else "missing",
        },
        {
            "check": "pbo_all_costs_available",
            "status": "pass" if set(pbo["cost_bps"].astype(float).round(1)) == {5.0, 10.0, 20.0} else "fail",
            "detail": ",".join(str(x) for x in sorted(pbo["cost_bps"].astype(float).unique())) if not pbo.empty else "missing",
        },
    ]
    return pd.concat([checks, pd.DataFrame(rows)], ignore_index=True, sort=False)


def make_agent_manifest(start_time: str, agent_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.32",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_32_macro_risk_budget_gate.py",
        "config": {
            "candidate_count": len(CANDIDATES),
            "costs": [5, 10, 20],
            "risk_budget_gate": True,
            "no_parameter_search": True,
        },
        "data_refs": [
            "outputs/hirssm_v3_10_clean_baseline/target_weights.csv",
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation/macro_rate_fx_signal_feature_panel.csv",
            "outputs/agent_runs/v3_31/selected_year_regime_attribution",
        ],
        "allowed_inputs": [
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_28/macro_rate_fx_signal_validation",
            "outputs/agent_runs/v3_31/selected_year_regime_attribution",
            "configs/hirssm_v2_default.json",
            "data_raw/index",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_32_macro_risk_budget_gate.py",
            "strategy_lab/hirssm_v3_11_nested_candidate_harness.py",
        ],
        "output_dir": rel(agent_dir),
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "The stress and repair budget fractions are predeclared hypotheses, not optimized parameters.",
            "The dual gate uses state confirmation and therefore must remain under PBO governance.",
            "Default promotion requires explicit review even if gates pass.",
        ],
        "risk_flags": [
            "macro_regime_shift",
            "small_candidate_set_pbo",
            "state_confirmed_rule_can_overfit",
        ],
        "next_decision": "Use V3.33 for failure attribution or constrained review based on V3.32 nested/PBO results.",
        "handoff_summary": "V3.32 tested macro risk-budget gates against V3.10 under 5/10/20bps, nested selection, and PBO.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.32 macro risk-budget gate harness.")
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
    gate_diag = gate_exposure_diagnostics(targets_by_variant)
    checks = pd.concat(
        [
            v311.build_constraint_checks(registry=registry, selection=selection, performance=nested_performance, pbo_summary=pbo_summary, decision=decision),
            target_integrity(targets_by_variant),
        ],
        ignore_index=True,
        sort=False,
    )
    checks = self_check_rows(registry, checks, decision, pbo_summary)
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
        "gate_exposure_diagnostics": output_dir / "gate_exposure_diagnostics.csv",
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
        (gate_diag, "gate_exposure_diagnostics"),
        (pbo_summary, "pbo_report"),
    ]:
        model.write_csv(df, paths[key])

    write_text(make_report(nested_performance, pbo_summary, decision, gate_diag), paths["report"])
    write_text(make_failure_cases(decision, pbo_summary, gate_diag), paths["failure_cases"])
    write_text("# HIRSSM V3.32 Self Check\n\n" + "\n".join(f"- {r.check}: {r.status} ({r.detail})" for r in checks.itertuples()), paths["self_check"])

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
    write_text(make_report(nested_performance, pbo_summary, decision, gate_diag), agent_report)
    for name, df in [
        ("candidate_registry.csv", registry),
        ("promotion_decision.csv", decision),
        ("constraint_check.csv", checks),
        ("validation_findings.csv", findings),
        ("leakage_checklist.csv", leakage),
        ("robustness_summary.csv", robustness),
        ("gate_exposure_diagnostics.csv", gate_diag),
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
        agent_dir / "gate_exposure_diagnostics.csv",
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
