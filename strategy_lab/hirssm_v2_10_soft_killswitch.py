#!/usr/bin/env python
"""HIRSSM V2.10 soft expert governance with strong-negative kill switch.

V2.10 demotes V2.1 hard expert gates from portfolio construction to audit.
Core experts stay active by default, receive only continuous multiplier
adjustments, and are hard-disabled only after consecutive strong negative
state-specific evidence. Observation experts still require positive evidence
before they can affect the portfolio.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
MODEL_PATH = ROOT / "strategy_lab" / "hirssm_v2_model.py"
WF_PATH = ROOT / "strategy_lab" / "hirssm_v2_walk_forward.py"
V21_PATH = ROOT / "strategy_lab" / "hirssm_v2_1_walk_forward.py"
V22_PATH = ROOT / "strategy_lab" / "hirssm_v2_2_walk_forward.py"
V23_PATH = ROOT / "strategy_lab" / "hirssm_v2_3_nested_walk_forward.py"
V24_PATH = ROOT / "strategy_lab" / "hirssm_v2_4_stable_nested_selection.py"
V25_PATH = ROOT / "strategy_lab" / "hirssm_v2_5_portfolio_risk_overlay.py"
V26_PATH = ROOT / "strategy_lab" / "hirssm_v2_6_to_v2_9_risk_iteration.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v2_10_1_soft_killswitch"


V210_SOFT_GOVERNANCE = {
    "version": "HIRSSM_V2_10_1",
    "train_years": 5,
    "horizon_days": 21,
    "min_observations": 8,
    "base_multiplier": 1.0,
    "min_multiplier": 0.90,
    "max_multiplier": 1.06,
    "shrink_strength": 0.08,
    "rank_ic_scale": 0.08,
    "icir_scale": 0.35,
    "positive_rate_scale": 0.15,
    "weak_negative_multiplier": 0.85,
    "negative_rank_ic_mean_max": -0.02,
    "negative_positive_ic_rate_max": 0.48,
    "negative_icir_max": -0.05,
    "strong_negative_rank_ic_mean_max": -0.06,
    "strong_negative_positive_ic_rate_max": 0.42,
    "strong_negative_icir_max": -0.18,
    "strong_negative_consecutive_windows": 3,
    "kill_switch_multiplier": 0.0,
    "kill_switch_eligible_experts": [
        "industry_trend_continuation",
        "industry_liquidity_overlay",
    ],
    "observation_entry_multiplier": 0.25,
    "observation_max_multiplier": 0.65,
    "observation_positive_ic_rate_min": 0.52,
    "observation_rank_ic_mean_min": 0.0,
    "observation_icir_min": 0.0,
    "observation_consecutive_pass_windows": 2,
    "consecutive_pass_windows_by_expert": {
        "range_reversal": 1,
    },
    "state_whitelist_by_expert": {
        "range_reversal": ["risk_off_decline", "crash_rebound"],
    },
    "candidate_experts": [
        "trend_continuation",
        "style_trend_continuation",
        "industry_trend_continuation",
        "valuation_repair",
        "risk_compression",
        "range_reversal",
        "defensive",
        "liquidity_overlay",
        "style_liquidity_overlay",
        "industry_liquidity_overlay",
    ],
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def active_specs(wf, soft_cfg: dict) -> dict:
    candidates = soft_cfg.get("candidate_experts", wf.EXPERT_SPECS.keys())
    return {name: wf.EXPERT_SPECS[name] for name in candidates if name in wf.EXPERT_SPECS}


def state_allowed(soft_cfg: dict, expert: str, state: str) -> bool:
    allowed = soft_cfg.get("state_whitelist_by_expert", {}).get(expert)
    if not allowed:
        return True
    return state in set(str(item) for item in allowed)


def observation_requirement(config: dict, soft_cfg: dict, expert: str) -> int:
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    default_disabled = set(str(item) for item in config.get("disabled_experts_by_default", []))
    if expert not in observation_experts and expert not in default_disabled:
        return 1
    by_expert = soft_cfg.get("consecutive_pass_windows_by_expert", {})
    return int(by_expert.get(expert, soft_cfg.get("observation_consecutive_pass_windows", 2)))


def evidence_score(stats: dict, soft_cfg: dict) -> float:
    pieces = []
    if pd.notna(stats.get("rank_ic_mean")):
        pieces.append(clip(float(stats["rank_ic_mean"]) / float(soft_cfg.get("rank_ic_scale", 0.08)), -1.0, 1.0))
    if pd.notna(stats.get("positive_ic_rate")):
        pieces.append(clip((float(stats["positive_ic_rate"]) - 0.5) / float(soft_cfg.get("positive_rate_scale", 0.15)), -1.0, 1.0))
    if pd.notna(stats.get("rank_icir")):
        pieces.append(clip(float(stats["rank_icir"]) / float(soft_cfg.get("icir_scale", 0.35)), -1.0, 1.0))
    return float(np.mean(pieces)) if pieces else 0.0


def positive_observation_pass(stats: dict, soft_cfg: dict) -> bool:
    return (
        stats["observations"] >= int(soft_cfg.get("min_observations", 8))
        and pd.notna(stats["rank_ic_mean"])
        and float(stats["rank_ic_mean"]) > float(soft_cfg.get("observation_rank_ic_mean_min", 0.0))
        and pd.notna(stats["positive_ic_rate"])
        and float(stats["positive_ic_rate"]) > float(soft_cfg.get("observation_positive_ic_rate_min", 0.52))
        and pd.notna(stats["rank_icir"])
        and float(stats["rank_icir"]) > float(soft_cfg.get("observation_icir_min", 0.0))
    )


def weak_negative_hit(stats: dict, soft_cfg: dict) -> bool:
    return (
        stats["observations"] >= int(soft_cfg.get("min_observations", 8))
        and pd.notna(stats["rank_ic_mean"])
        and float(stats["rank_ic_mean"]) < float(soft_cfg.get("negative_rank_ic_mean_max", -0.02))
        and pd.notna(stats["positive_ic_rate"])
        and float(stats["positive_ic_rate"]) < float(soft_cfg.get("negative_positive_ic_rate_max", 0.48))
        and pd.notna(stats["rank_icir"])
        and float(stats["rank_icir"]) < float(soft_cfg.get("negative_icir_max", -0.05))
    )


def strong_negative_hit(stats: dict, soft_cfg: dict) -> bool:
    return (
        stats["observations"] >= int(soft_cfg.get("min_observations", 8))
        and pd.notna(stats["rank_ic_mean"])
        and float(stats["rank_ic_mean"]) < float(soft_cfg.get("strong_negative_rank_ic_mean_max", -0.06))
        and pd.notna(stats["positive_ic_rate"])
        and float(stats["positive_ic_rate"]) < float(soft_cfg.get("strong_negative_positive_ic_rate_max", 0.42))
        and pd.notna(stats["rank_icir"])
        and float(stats["rank_icir"]) < float(soft_cfg.get("strong_negative_icir_max", -0.18))
    )


def multiplier_reason(row: dict) -> str:
    if not row["structurally_allowed"]:
        return "blocked by predeclared structural state whitelist"
    if row["hard_kill_applied"]:
        return "hard kill: consecutive strong negative state evidence"
    if row["requires_positive_gate"] and row["multiplier"] <= 0:
        return "observation expert inactive: positive evidence gate not met"
    if row["requires_positive_gate"]:
        return "observation expert enabled by positive state evidence"
    if row["observations"] < row["min_observations"]:
        return "core expert kept at baseline: insufficient evidence"
    if row["weak_negative_hit"]:
        return "core expert softly cut by weak negative evidence"
    if row["multiplier"] > 1:
        return "core expert softly tilted up by positive evidence"
    if row["multiplier"] < 1:
        return "core expert softly shrunk by weak evidence"
    return "core expert baseline multiplier"


def build_v210_multiplier_history(monthly_ic: pd.DataFrame, config: dict, wf, specs: dict, soft_cfg: dict) -> pd.DataFrame:
    if monthly_ic.empty:
        return pd.DataFrame()
    clean_ic = monthly_ic.copy()
    clean_ic["state"] = clean_ic["state"].replace("", np.nan).fillna("unknown").astype(str)
    train_years = int(soft_cfg.get("train_years", 5))
    states = [str(item) for item in config.get("regime_model", {}).get("states", [])]
    candidates = [item for item in soft_cfg.get("candidate_experts", specs.keys()) if item in specs]
    observation_experts = set(str(item) for item in config.get("expert_gates", {}).get("observation_experts", []))
    default_disabled = set(str(item) for item in config.get("disabled_experts_by_default", []))
    years = sorted(int(y) for y in clean_ic["year"].dropna().unique())
    first_year = min(years) + train_years
    consecutive_pass = {(expert, state): 0 for expert in candidates for state in states}
    consecutive_strong_negative = {(expert, state): 0 for expert in candidates for state in states}
    rows = []

    for test_year in [year for year in years if year >= first_year]:
        train_start = pd.Timestamp(year=test_year - train_years, month=1, day=1)
        train_end = pd.Timestamp(year=test_year, month=1, day=1)
        for expert in candidates:
            spec = specs[expert]
            applies_to_portfolio = bool(spec.get("applies_to_portfolio", True))
            requires_positive = expert in observation_experts or expert in default_disabled
            for state in states:
                train_ic = clean_ic[
                    clean_ic["expert"].eq(expert)
                    & clean_ic["state"].eq(state)
                    & (clean_ic["date"] >= train_start)
                    & (clean_ic["date"] < train_end)
                ]
                stats = wf.summarize_ic(train_ic)
                structural = state_allowed(soft_cfg, expert, state)
                raw_pass = bool(structural and positive_observation_pass(stats, soft_cfg))
                weak_hit = bool(structural and weak_negative_hit(stats, soft_cfg))
                strong_hit = bool(structural and strong_negative_hit(stats, soft_cfg))
                key = (expert, state)
                consecutive_pass[key] = consecutive_pass[key] + 1 if raw_pass else 0
                consecutive_strong_negative[key] = consecutive_strong_negative[key] + 1 if strong_hit else 0
                consecutive_required = observation_requirement(config, soft_cfg, expert)
                strong_required = int(soft_cfg.get("strong_negative_consecutive_windows", 2))
                evidence = evidence_score(stats, soft_cfg)
                kill_eligible = expert in set(str(item) for item in soft_cfg.get("kill_switch_eligible_experts", []))
                hard_kill = bool(kill_eligible and (not requires_positive) and structural and consecutive_strong_negative[key] >= strong_required)

                if not structural:
                    multiplier = 0.0
                elif hard_kill:
                    multiplier = float(soft_cfg.get("kill_switch_multiplier", 0.0))
                elif requires_positive:
                    if raw_pass and consecutive_pass[key] >= consecutive_required:
                        entry = float(soft_cfg.get("observation_entry_multiplier", 0.25))
                        obs_max = float(soft_cfg.get("observation_max_multiplier", 0.65))
                        multiplier = clip(entry + max(evidence, 0.0) * (obs_max - entry), entry, obs_max)
                    else:
                        multiplier = 0.0
                elif stats["observations"] < int(soft_cfg.get("min_observations", 8)):
                    multiplier = float(soft_cfg.get("base_multiplier", 1.0))
                else:
                    base = float(soft_cfg.get("base_multiplier", 1.0))
                    strength = float(soft_cfg.get("shrink_strength", 0.12))
                    multiplier = base + strength * evidence
                    multiplier = clip(multiplier, float(soft_cfg.get("min_multiplier", 0.85)), float(soft_cfg.get("max_multiplier", 1.08)))
                    if weak_hit:
                        multiplier = min(multiplier, float(soft_cfg.get("weak_negative_multiplier", 0.75)))

                row = {
                    "variant": "soft_killswitch_v2_10_1",
                    "test_year": int(test_year),
                    "state": state,
                    "train_start": train_start.date().isoformat(),
                    "train_end": (train_end - pd.Timedelta(days=1)).date().isoformat(),
                    "expert": expert,
                    "asset_type_scope": spec["asset_type"] or "all",
                    "score_col": spec["score_col"],
                    "observations": stats["observations"],
                    "min_observations": int(soft_cfg.get("min_observations", 8)),
                    "rank_ic_mean": stats["rank_ic_mean"],
                    "rank_ic_std": stats["rank_ic_std"],
                    "rank_icir": stats["rank_icir"],
                    "positive_ic_rate": stats["positive_ic_rate"],
                    "evidence_score": evidence,
                    "weak_negative_hit": weak_hit,
                    "strong_negative_hit": strong_hit,
                    "consecutive_strong_negative_windows": consecutive_strong_negative[key],
                    "strong_negative_required": strong_required,
                    "hard_kill_applied": hard_kill,
                    "kill_switch_eligible": kill_eligible,
                    "requires_positive_gate": requires_positive,
                    "raw_observation_pass": raw_pass,
                    "consecutive_pass_windows": consecutive_pass[key],
                    "consecutive_required": consecutive_required,
                    "structurally_allowed": structural,
                    "applies_to_portfolio": applies_to_portfolio,
                    "multiplier": float(multiplier),
                    "economic_logic": spec["economic_logic"],
                    "failure_scenario": spec["failure_scenario"],
                }
                row["reason"] = multiplier_reason(row)
                rows.append(row)

    history = pd.DataFrame(rows)
    if history.empty:
        return history
    test_stats = (
        clean_ic.groupby(["year", "state", "expert"], dropna=False)
        .agg(
            test_rank_ic_mean=("rank_ic", "mean"),
            test_positive_ic_rate=("rank_ic", lambda s: float((s > 0).mean())),
            test_observations=("rank_ic", "count"),
        )
        .reset_index()
        .rename(columns={"year": "test_year"})
    )
    return history.merge(test_stats, on=["test_year", "state", "expert"], how="left")


def build_targets_from_history(model, panel: dict, config: dict, v22, history: pd.DataFrame) -> pd.DataFrame:
    multiplier_map = v22.build_multiplier_map(history)
    first_test_year = int(history["test_year"].min()) if not history.empty else int(panel["eligible"]["date"].dt.year.min())
    start_date = pd.Timestamp(year=first_test_year, month=1, day=1)
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=set(),
        expert_multipliers_by_year_state=multiplier_map,
    )
    targets = v22.add_multiplier_columns(targets, history)
    if not targets.empty and "test_year" in targets.columns and not history.empty:
        targets = targets[targets["test_year"].isin(history["test_year"].unique())]
    return targets


def run_static_costs(model, panel: dict, targets: pd.DataFrame, costs: list[float], output_dir: Path, prefix: str, variant: str) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    rows = []
    nav_by_cost = {}
    for cost in costs:
        bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
        summary = model.summarize_nav(bt["nav"])
        if not summary.empty:
            summary.insert(0, "variant", variant)
            summary.insert(1, "cost_bps", float(cost))
            summary["target_rows"] = int(targets.shape[0])
            summary["oos_start"] = bt["nav"]["date"].min() if not bt["nav"].empty else pd.NaT
            summary["oos_end"] = bt["nav"]["date"].max() if not bt["nav"].empty else pd.NaT
            summary["oos_years"] = (
                (pd.to_datetime(bt["nav"]["date"].max()) - pd.to_datetime(bt["nav"]["date"].min())).days / 365.25
                if not bt["nav"].empty
                else np.nan
            )
            rows.append(summary)
        suffix = f"{prefix}_{int(cost)}bps"
        model.write_csv(bt["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(bt["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(bt["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        nav_by_cost[float(cost)] = bt["nav"]
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(), nav_by_cost


def compare_against(summary: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    if summary.empty or reference.empty:
        return pd.DataFrame()
    out = summary.merge(reference.add_prefix(f"{label}_"), left_on="cost_bps", right_on=f"{label}_cost_bps", how="left")
    for col in ["annual_return", "sharpe_no_rf", "max_drawdown", "avg_cash_weight", "avg_trade_turnover", "annual_vol", "total_return"]:
        ref_col = f"{label}_{col}"
        if col in out.columns and ref_col in out.columns:
            out[f"delta_{col}_vs_{label}"] = out[col] - out[ref_col]
    return out


def governance_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    applied = history[history["applies_to_portfolio"].astype(bool)]
    if applied.empty:
        return pd.DataFrame()
    return (
        applied.groupby("expert", dropna=False)
        .agg(
            rows=("expert", "size"),
            avg_multiplier=("multiplier", "mean"),
            min_multiplier=("multiplier", "min"),
            max_multiplier=("multiplier", "max"),
            hard_kill_rate=("hard_kill_applied", "mean"),
            weak_negative_rate=("weak_negative_hit", "mean"),
            avg_train_rank_ic=("rank_ic_mean", "mean"),
            avg_test_rank_ic=("test_rank_ic_mean", "mean"),
            avg_test_positive_ic_rate=("test_positive_ic_rate", "mean"),
        )
        .reset_index()
        .sort_values(["hard_kill_rate", "avg_multiplier"], ascending=[False, True])
    )


def make_self_check(smoke: pd.DataFrame, history: pd.DataFrame, summary: pd.DataFrame, comparison_v24: pd.DataFrame, comparison_v27: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    applied = history[history["applies_to_portfolio"].astype(bool)] if not history.empty else pd.DataFrame()
    observation = {"range_reversal", "style_trend_continuation", "style_liquidity_overlay"}
    core = applied[~applied["expert"].isin(observation)] if not applied.empty else pd.DataFrame()
    non_kill_zero = core[(core["multiplier"].astype(float) <= 0) & (~core["hard_kill_applied"].astype(bool))] if not core.empty else pd.DataFrame()
    row10 = summary[summary["cost_bps"].astype(float).eq(10.0)].head(1) if not summary.empty else pd.DataFrame()
    v24_10 = comparison_v24[comparison_v24["cost_bps"].astype(float).eq(10.0)].head(1) if not comparison_v24.empty else pd.DataFrame()
    v27_10 = comparison_v27[comparison_v27["cost_bps"].astype(float).eq(10.0)].head(1) if not comparison_v27.empty else pd.DataFrame()
    rows = [
        {"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""},
        {"check": "required_cost_rows", "pass": bool(set([5.0, 10.0, 20.0, 30.0]).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""},
        {"check": "no_core_hard_zero_without_strong_kill", "pass": bool(non_kill_zero.empty), "detail": str(non_kill_zero.shape[0]) if not non_kill_zero.empty else "0"},
        {"check": "governance_history_non_empty", "pass": bool(not history.empty), "detail": str(history.shape[0]) if not history.empty else ""},
        {"check": "annual_return_positive_10bps", "pass": bool(not row10.empty and float(row10["annual_return"].iloc[0]) > 0), "detail": f"{float(row10['annual_return'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "drawdown_not_worse_than_v2_4", "pass": bool(not v24_10.empty and float(v24_10["delta_max_drawdown_vs_v24"].iloc[0]) >= 0), "detail": f"{float(v24_10['delta_max_drawdown_vs_v24'].iloc[0]):.6f}" if not v24_10.empty else ""},
        {"check": "sharpe_not_worse_than_v2_7_by_0_03", "pass": bool(not v27_10.empty and float(v27_10["delta_sharpe_no_rf_vs_v27"].iloc[0]) >= -0.03), "detail": f"{float(v27_10['delta_sharpe_no_rf_vs_v27'].iloc[0]):.6f}" if not v27_10.empty else ""},
    ]
    for name in ["WALK_FORWARD_REPORT.md", "FACTOR_GATE_REPORT.md", "MODEL_CHANGELOG.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def make_reports(
    output_dir: Path,
    soft_cfg: dict,
    raw_summary: pd.DataFrame,
    summary: pd.DataFrame,
    v24_summary: pd.DataFrame,
    v27_summary: pd.DataFrame,
    comparison_v24: pd.DataFrame,
    comparison_v27: pd.DataFrame,
    history: pd.DataFrame,
    decisions: pd.DataFrame,
    smoke: pd.DataFrame,
    self_check: pd.DataFrame | None,
) -> None:
    gate_summary = governance_summary(history)
    key_params = pd.DataFrame(
        [
            {"parameter": key, "value": str(value)}
            for key, value in soft_cfg.items()
            if key
            in {
                "min_multiplier",
                "max_multiplier",
                "shrink_strength",
                "weak_negative_multiplier",
                "strong_negative_rank_ic_mean_max",
                "strong_negative_positive_ic_rate_max",
                "strong_negative_icir_max",
                "strong_negative_consecutive_windows",
                "kill_switch_multiplier",
                "observation_entry_multiplier",
                "observation_max_multiplier",
            }
        ]
    )
    lines = [
        "# HIRSSM V2.10.1 Walk-Forward Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Intent",
        "",
        "- Remove V2.1 hard expert gates from direct portfolio construction.",
        "- Keep audit records for every expert x state window.",
        "- Apply continuous soft multipliers to core experts.",
        "- Apply hard kill only after consecutive strong negative evidence for explicitly eligible industry experts.",
        "- Keep observation experts inactive until positive evidence appears.",
        "- Reuse V2.7 local risk overlay; no new alpha factor is introduced.",
        "",
        "## Final V2.10.1 Performance",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No final summary.",
        "",
        "## Raw Soft-Governance Performance",
        "",
        raw_summary.to_markdown(index=False) if not raw_summary.empty else "No raw summary.",
        "",
        "## V2.4 Reference",
        "",
        v24_summary.to_markdown(index=False) if not v24_summary.empty else "No V2.4 reference.",
        "",
        "## V2.7 Reference",
        "",
        v27_summary.to_markdown(index=False) if not v27_summary.empty else "No V2.7 reference.",
        "",
        "## Comparison vs V2.4",
        "",
        comparison_v24.to_markdown(index=False) if not comparison_v24.empty else "No comparison.",
        "",
        "## Comparison vs V2.7",
        "",
        comparison_v27.to_markdown(index=False) if not comparison_v27.empty else "No comparison.",
        "",
        "## Governance Parameters",
        "",
        key_params.to_markdown(index=False),
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    factor_lines = [
        "# HIRSSM V2.10.1 Factor Gate Report",
        "",
        "V2.10.1 treats V2.1-style hard gating as audit evidence. Core expert-state pairs are not hard-disabled; hard kill is limited to explicitly eligible industry expert-state pairs after consecutive strong-negative evidence.",
        "",
        "## Expert Multiplier Summary",
        "",
        gate_summary.to_markdown(index=False) if not gate_summary.empty else "No governance summary.",
        "",
        "## Latest Decisions",
        "",
        decisions.sort_values(["test_year", "state", "expert"]).tail(80).to_markdown(index=False) if not decisions.empty else "No decisions.",
    ]
    (output_dir / "FACTOR_GATE_REPORT.md").write_text("\n".join(factor_lines), encoding="utf-8")

    changelog = [
        "# HIRSSM V2.10.1 Model Changelog",
        "",
        "## Changed",
        "",
        "- Replaced direct hard expert gating with continuous soft multipliers.",
        "- Limited strong-negative consecutive-window kill switch to industry trend and industry liquidity experts.",
        "- Removed the V2.1 `industry_trend_continuation` hard state whitelist from core expert governance.",
        "- Preserved observation treatment for `range_reversal`, `style_trend_continuation`, and `style_liquidity_overlay`.",
        "- Reused V2.7 local risk overlay for final portfolio construction.",
        "",
        "## Why",
        "",
        "V2.10 still hard-killed too many core experts. V2.10.1 keeps governance auditability while avoiding excessive 0/1 portfolio jumps in trend, valuation, risk-compression, and defensive priors.",
        "",
        "## Default Decision",
        "",
        "V2.10.1 can be evaluated as a candidate only if its self-check and comparison against V2.7 are acceptable. It should not replace V2.7 solely because it is newer.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")

    self_lines = [
        "# HIRSSM V2.10.1 Self Check Report",
        "",
        self_check.to_markdown(index=False) if self_check is not None and not self_check.empty else "Self check is written after report generation.",
        "",
        "## Smoke Test",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke test.",
    ]
    (output_dir / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v21 = load_module("hirssm_v2_1_walk_forward", V21_PATH)
    v22 = load_module("hirssm_v2_2_walk_forward", V22_PATH)
    v23 = load_module("hirssm_v2_3_nested_walk_forward", V23_PATH)
    v24 = load_module("hirssm_v2_4_stable_nested_selection", V24_PATH)
    v25 = load_module("hirssm_v2_5_portfolio_risk_overlay", V25_PATH)
    v26 = load_module("hirssm_v2_6_to_v2_9_risk_iteration", V26_PATH)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    soft_cfg = copy.deepcopy(V210_SOFT_GOVERNANCE)
    specs = active_specs(wf, soft_cfg)

    context = v25.build_v24_context(model, wf, v21, v22, v23, v24, root, config, args.start_date, args.end_date)
    panel = context["panel"]
    monthly_ic = context["monthly_ic"]
    base_targets = context["base_targets"]
    costs = [5.0, 10.0, 20.0, 30.0]

    history = build_v210_multiplier_history(monthly_ic, config, wf, specs, soft_cfg)
    raw_targets = build_targets_from_history(model, panel, config, v22, history)

    v27_cfg = v26.resolved_version_config(config, "portfolio_risk_overlay_v2_7")
    v210_risk_cfg = deep_merge(
        v27_cfg,
        {
            "version": "HIRSSM_V2_10_1",
            "description": "V2.10.1 soft expert governance plus V2.7 local risk overlay.",
            "cost_bps_scenarios": costs,
            "self_check_thresholds": {
                "max_annual_return_slippage_vs_v2_4_10bps": 0.02,
                "min_drawdown_improvement_vs_v2_4_10bps": 0.08,
                "min_delta_sharpe_vs_v2_4_10bps": -0.02,
                "max_cash_weight": 0.34,
                "min_20bps_annual_return": 0.04,
            },
        },
    )
    final_targets, overlay_decisions = v26.overlay_targets_local(raw_targets, panel, v210_risk_cfg, "HIRSSM_V2_10_1")
    v27_targets, _ = v26.overlay_targets_local(base_targets, panel, v27_cfg, "HIRSSM_V2_7_REFERENCE")

    raw_summary, _ = run_static_costs(model, panel, raw_targets, costs, output_dir, "v2_10_1_raw_soft", "v2_10_1_raw_soft_killswitch")
    v24_summary = v26.run_base_costs(model, panel, base_targets, costs, output_dir / "reference_v2_4")
    v27_summary, _, _ = v26.run_costs(model, panel, v27_targets, v27_cfg, "HIRSSM_V2_7_REFERENCE", output_dir / "reference_v2_7")
    summary, _, brake_by_cost = v26.run_costs(model, panel, final_targets, v210_risk_cfg, "HIRSSM_V2_10_1", output_dir)

    comparison_v24 = compare_against(summary, v24_summary, "v24")
    comparison_v27 = compare_against(summary, v27_summary, "v27")
    smoke = wf.smoke_test_targets(final_targets)

    model.write_csv(history, output_dir / "expert_soft_killswitch_history.csv")
    model.write_csv(history, output_dir / "EXPERT_DECISION_LOG.csv")
    model.write_csv(governance_summary(history), output_dir / "expert_multiplier_summary.csv")
    model.write_csv(raw_targets, output_dir / "walk_forward_target_weights_raw_soft.csv")
    model.write_csv(final_targets, output_dir / "walk_forward_target_weights.csv")
    model.write_csv(raw_summary, output_dir / "raw_soft_oos_performance.csv")
    model.write_csv(summary, output_dir / "oos_performance.csv")
    model.write_csv(v24_summary, output_dir / "reference_v2_4_oos_performance.csv")
    model.write_csv(v27_summary, output_dir / "reference_v2_7_oos_performance.csv")
    model.write_csv(comparison_v24, output_dir / "v2_10_1_vs_v2_4_comparison.csv")
    model.write_csv(comparison_v27, output_dir / "v2_10_1_vs_v2_7_comparison.csv")
    model.write_csv(overlay_decisions, output_dir / "risk_overlay_decisions.csv")
    model.write_csv(smoke, output_dir / "smoke_test_results.csv")

    make_reports(
        output_dir,
        soft_cfg,
        raw_summary,
        summary,
        v24_summary,
        v27_summary,
        comparison_v24,
        comparison_v27,
        history,
        history,
        smoke,
        None,
    )
    self_check = make_self_check(smoke, history, summary, comparison_v24, comparison_v27, output_dir)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    make_reports(
        output_dir,
        soft_cfg,
        raw_summary,
        summary,
        v24_summary,
        v27_summary,
        comparison_v24,
        comparison_v27,
        history,
        history,
        smoke,
        self_check,
    )

    manifest = {
        "version": "HIRSSM_V2_10_1",
        "generated_at": now_text(),
        "output_dir": str(output_dir),
        "self_check_pass": bool(self_check["pass"].all()) if not self_check.empty else False,
        "costs": costs,
        "target_rows": int(final_targets.shape[0]),
        "history_rows": int(history.shape[0]),
    }
    (output_dir / "run_manifest.json").write_text(pd.Series(manifest).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(pd.Series(manifest).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
