#!/usr/bin/env python
"""HIRSSM V3.0/V3.1 benchmark-relative and core-satellite iteration.

V3.0 changes the research objective from absolute stable performance to
benchmark-relative performance against CSI All Share (000985).

V3.1 builds on the V3.0 selected active sleeve and adds a state-conditioned
000985 core position plus a satellite alpha sleeve. This is a structural
iteration, not a small expert-gate parameter tweak.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
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
V210_PATH = ROOT / "strategy_lab" / "hirssm_v2_10_soft_killswitch.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core"


COSTS = [5.0, 10.0, 20.0, 30.0]
BENCHMARK_ASSET = "000985"


V31_VARIANTS = {
    "v3_1_balanced_core": {
        "description": "Balanced benchmark core with moderate satellite and explicit risk-off cash.",
        "core_weight_by_state": {
            "risk_on_trend": 0.55,
            "risk_on_overheat": 0.50,
            "range_bound": 0.45,
            "risk_off_decline": 0.20,
            "crash_rebound": 0.50,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.00,
            "risk_on_overheat": 0.06,
            "range_bound": 0.08,
            "risk_off_decline": 0.30,
            "crash_rebound": 0.05,
        },
    },
    "v3_1_offensive_core": {
        "description": "Higher beta participation in risk-on states with lower cash floors.",
        "core_weight_by_state": {
            "risk_on_trend": 0.65,
            "risk_on_overheat": 0.55,
            "range_bound": 0.45,
            "risk_off_decline": 0.15,
            "crash_rebound": 0.60,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.00,
            "risk_on_overheat": 0.04,
            "range_bound": 0.05,
            "risk_off_decline": 0.25,
            "crash_rebound": 0.03,
        },
    },
    "v3_1_defensive_core": {
        "description": "Lower benchmark core and higher cash in non-trending or risk-off states.",
        "core_weight_by_state": {
            "risk_on_trend": 0.45,
            "risk_on_overheat": 0.40,
            "range_bound": 0.35,
            "risk_off_decline": 0.15,
            "crash_rebound": 0.45,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.00,
            "risk_on_overheat": 0.08,
            "range_bound": 0.12,
            "risk_off_decline": 0.35,
            "crash_rebound": 0.08,
        },
    },
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


def run_static_costs(model, panel: dict, targets: pd.DataFrame, output_dir: Path, prefix: str, variant: str) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    rows = []
    nav_by_cost = {}
    for cost in COSTS:
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


def add_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["annual_excess_vs_benchmark"] = out["annual_return"] - out["benchmark_annual_return"]
    out["total_excess_vs_benchmark"] = out["total_return"] - out["benchmark_total_return"]
    out["drawdown_improvement_vs_benchmark"] = out["max_drawdown"] - out["benchmark_max_drawdown"]
    out["vol_reduction_vs_benchmark"] = out["benchmark_annual_vol"] - out["annual_vol"]
    out["cash_penalty"] = np.maximum(out["avg_cash_weight"] - 0.25, 0.0)
    out["turnover_penalty"] = out["avg_trade_turnover"].fillna(0.0)
    return out


def benchmark_relative_score(summary: pd.DataFrame, variant: str, source: str) -> pd.DataFrame:
    rel = add_relative_metrics(summary)
    rows = []
    weights = {10.0: 0.50, 20.0: 0.30, 30.0: 0.20}
    for cost, weight in weights.items():
        row = rel[rel["cost_bps"].astype(float).eq(cost)].head(1)
        if row.empty:
            continue
        item = row.iloc[0]
        score = (
            2.20 * float(item["annual_excess_vs_benchmark"])
            + 0.35 * float(item["drawdown_improvement_vs_benchmark"])
            + 0.20 * float(item["vol_reduction_vs_benchmark"])
            + 0.12 * float(item["information_ratio"])
            + 0.20 * float(item["sharpe_no_rf"])
            - 0.25 * float(item["cash_penalty"])
            - 0.015 * float(item["turnover_penalty"])
        )
        rows.append(
            {
                "variant": variant,
                "source": source,
                "cost_bps": cost,
                "weight": weight,
                "score_component": score,
                "weighted_score_component": score * weight,
                "annual_return": float(item["annual_return"]),
                "annual_excess_vs_benchmark": float(item["annual_excess_vs_benchmark"]),
                "sharpe_no_rf": float(item["sharpe_no_rf"]),
                "information_ratio": float(item["information_ratio"]),
                "max_drawdown": float(item["max_drawdown"]),
                "drawdown_improvement_vs_benchmark": float(item["drawdown_improvement_vs_benchmark"]),
                "avg_cash_weight": float(item["avg_cash_weight"]),
                "avg_trade_turnover": float(item["avg_trade_turnover"]),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    total = detail.groupby(["variant", "source"], as_index=False).agg(
        benchmark_relative_score=("weighted_score_component", "sum"),
        avg_annual_excess=("annual_excess_vs_benchmark", "mean"),
        avg_drawdown_improvement=("drawdown_improvement_vs_benchmark", "mean"),
        avg_information_ratio=("information_ratio", "mean"),
        mean_cash_weight=("avg_cash_weight", "mean"),
    )
    return detail.merge(total, on=["variant", "source"], how="left")


def build_stable_variant_targets(model, wf, v22, panel: dict, config: dict, monthly_ic: pd.DataFrame, variant: str) -> pd.DataFrame:
    stable_grid = config.get("expert_state_stable_selection", {}).get("stable_variant_grid", [])
    override = next((item for item in stable_grid if item.get("variant") == variant), None)
    if override is None:
        raise ValueError(f"missing stable variant: {variant}")
    specs = v22.active_specs(wf, config)
    run = v22.run_shrinkage_variant(model, wf, panel, config, specs, monthly_ic, override, variant, cost_bps=10.0)
    return run["targets"]


def build_v2101_targets(model, wf, v22, v26, v210, panel: dict, config: dict, monthly_ic: pd.DataFrame) -> pd.DataFrame:
    soft_cfg = copy.deepcopy(v210.V210_SOFT_GOVERNANCE)
    specs = v210.active_specs(wf, soft_cfg)
    history = v210.build_v210_multiplier_history(monthly_ic, config, wf, specs, soft_cfg)
    raw_targets = v210.build_targets_from_history(model, panel, config, v22, history)
    v27_cfg = v26.resolved_version_config(config, "portfolio_risk_overlay_v2_7")
    v210_cfg = deep_merge(
        v27_cfg,
        {
            "version": "HIRSSM_V3_0_V2101_GOVERNED_CANDIDATE",
            "cost_bps_scenarios": COSTS,
        },
    )
    targets, _ = v26.overlay_targets_local(raw_targets, panel, v210_cfg, "HIRSSM_V3_0_V2101_GOVERNED_CANDIDATE")
    return targets


def normalize_targets(group: pd.DataFrame) -> pd.Series:
    noncash = group[~group["asset"].astype(str).eq("CASH")].copy()
    weights = pd.to_numeric(noncash["weight"], errors="coerce").clip(lower=0)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return weights / total


def build_core_satellite_targets(alpha_targets: pd.DataFrame, variant: str, cfg: dict, benchmark_asset: str = BENCHMARK_ASSET) -> pd.DataFrame:
    rows = []
    alpha = alpha_targets.copy()
    alpha["signal_date"] = pd.to_datetime(alpha["signal_date"])
    alpha["weight"] = pd.to_numeric(alpha["weight"], errors="coerce").fillna(0.0)
    for signal_date, group in alpha.sort_values(["signal_date", "asset"]).groupby("signal_date"):
        state = str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else "range_bound"
        core_weight = float(cfg["core_weight_by_state"].get(state, cfg["core_weight_by_state"].get("range_bound", 0.45)))
        cash_floor = float(cfg["cash_floor_by_state"].get(state, cfg["cash_floor_by_state"].get("range_bound", 0.08)))
        core_weight = min(max(core_weight, 0.0), 1.0)
        cash_floor = min(max(cash_floor, 0.0), 1.0 - core_weight)
        satellite_budget = max(0.0, 1.0 - core_weight - cash_floor)
        noncash = group[~group["asset"].astype(str).eq("CASH")].copy()
        norm = normalize_targets(group)
        weights: dict[str, float] = {}
        if not norm.empty:
            for idx, ratio in norm.items():
                asset = str(noncash.loc[idx, "asset"])
                weights[asset] = weights.get(asset, 0.0) + satellite_budget * float(ratio)
        weights[benchmark_asset] = weights.get(benchmark_asset, 0.0) + core_weight
        noncash_sum = sum(max(v, 0.0) for v in weights.values())
        if noncash_sum > 1.0:
            weights = {asset: weight / noncash_sum for asset, weight in weights.items()}
            noncash_sum = 1.0
        cash = max(0.0, 1.0 - noncash_sum)
        prev = None
        for asset, weight in sorted(weights.items()):
            ref = noncash[noncash["asset"].astype(str).eq(asset)].head(1)
            asset_type = "style" if asset == benchmark_asset else (str(ref["asset_type"].iloc[0]) if not ref.empty else "")
            score = float(ref["score"].iloc[0]) if "score" in ref.columns and not ref.empty else 0.0
            risk_adjusted_alpha = float(ref["risk_adjusted_alpha"].iloc[0]) if "risk_adjusted_alpha" in ref.columns and not ref.empty else 0.0
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": asset,
                    "weight": float(weight),
                    "state": state,
                    "asset_type": asset_type,
                    "score": score,
                    "risk_adjusted_alpha": risk_adjusted_alpha,
                    "turnover": np.nan,
                    "v3_1_variant": variant,
                    "core_weight": core_weight,
                    "satellite_budget": satellite_budget,
                    "cash_floor": cash_floor,
                }
            )
            prev = asset
        if cash > 0:
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": "CASH",
                    "weight": float(cash),
                    "state": state,
                    "asset_type": "cash",
                    "score": 0.0,
                    "risk_adjusted_alpha": 0.0,
                    "turnover": np.nan,
                    "v3_1_variant": variant,
                    "core_weight": core_weight,
                    "satellite_budget": satellite_budget,
                    "cash_floor": cash_floor,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["signal_date", "asset"]).reset_index(drop=True)
    prev_weights: dict[str, float] = {}
    turnovers = {}
    for signal_date, group in out.groupby("signal_date", sort=True):
        current = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
        turnover = sum(abs(current.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in set(current) | set(prev_weights))
        turnovers[pd.Timestamp(signal_date)] = turnover
        prev_weights = current
    out["turnover"] = out["signal_date"].map(turnovers)
    return out


def make_candidate_table(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    cols = [
        "variant",
        "source",
        "benchmark_relative_score",
        "avg_annual_excess",
        "avg_drawdown_improvement",
        "avg_information_ratio",
        "mean_cash_weight",
    ]
    return scores[cols].drop_duplicates().sort_values("benchmark_relative_score", ascending=False)


def make_self_check(smoke: pd.DataFrame, summary: pd.DataFrame, score_table: pd.DataFrame, output_dir: Path, version: str) -> pd.DataFrame:
    row10 = add_relative_metrics(summary)
    row10 = row10[row10["cost_bps"].astype(float).eq(10.0)].head(1) if not row10.empty else pd.DataFrame()
    rows = [
        {"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""},
        {"check": "required_cost_rows", "pass": bool(set(COSTS).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""},
        {"check": "positive_annual_excess_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "investment_gate_annual_excess_above_3pct_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) >= 0.03), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "drawdown_better_than_benchmark_10bps", "pass": bool(not row10.empty and float(row10["drawdown_improvement_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['drawdown_improvement_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "avg_cash_not_too_high_10bps", "pass": bool(not row10.empty and float(row10["avg_cash_weight"].iloc[0]) <= 0.35), "detail": f"{float(row10['avg_cash_weight'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "score_table_non_empty", "pass": bool(not score_table.empty), "detail": str(score_table.shape[0]) if not score_table.empty else ""},
    ]
    for name in ["MODEL_CHANGELOG.md", "WALK_FORWARD_REPORT.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{version}_{name}", "pass": bool((output_dir / version / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def write_reports(
    output_dir: Path,
    version_dir: str,
    title: str,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    selected_variant: str,
    smoke: pd.DataFrame,
    self_check: pd.DataFrame | None,
    notes: list[str],
) -> None:
    path = output_dir / version_dir
    path.mkdir(parents=True, exist_ok=True)
    report = [
        f"# {title}",
        "",
        f"Run time: {now_text()}",
        "",
        "## Selected Variant",
        "",
        f"- `{selected_variant}`",
        "",
        "## Notes",
        "",
        *[f"- {item}" for item in notes],
        "",
        "## Performance",
        "",
        add_relative_metrics(summary).to_markdown(index=False) if not summary.empty else "No summary.",
        "",
        "## Benchmark-Relative Score Table",
        "",
        make_candidate_table(scores).to_markdown(index=False) if not scores.empty else "No score table.",
        "",
        "## Smoke Test",
        "",
        smoke.to_markdown(index=False) if not smoke.empty else "No smoke test.",
    ]
    (path / "WALK_FORWARD_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    changelog = [
        f"# {title} Model Changelog",
        "",
        "## Changed",
        "",
        *[f"- {item}" for item in notes],
        "",
        "## Governance",
        "",
        "- Selection uses a predeclared benchmark-relative objective.",
        "- 10/20/30bps costs enter the objective; 5bps is reported but not selected on.",
        "- The strategy is compared against CSI All Share through annual excess, drawdown improvement, tracking error and information ratio.",
    ]
    (path / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")

    self_lines = [
        f"# {title} Self Check Report",
        "",
        self_check.to_markdown(index=False) if self_check is not None and not self_check.empty else "Self check pending.",
    ]
    (path / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


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
    v210 = load_module("hirssm_v2_10_soft_killswitch", V210_PATH)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))

    context = v25.build_v24_context(model, wf, v21, v22, v23, v24, root, config, args.start_date, args.end_date)
    panel = context["panel"]
    monthly_ic = context["monthly_ic"]
    base_targets = context["base_targets"]

    candidates: dict[str, dict] = {}
    candidates["v2_4_stable_selected"] = {"targets": base_targets, "mode": "static", "source": "V2.4 selected stable nested sleeve"}
    for stable_variant in ["stable_balanced", "stable_conservative"]:
        targets = build_stable_variant_targets(model, wf, v22, panel, config, monthly_ic, stable_variant)
        candidates[f"v3_0_{stable_variant}"] = {"targets": targets, "mode": "static", "source": f"V3.0 candidate from {stable_variant}"}

    v27_cfg = v26.resolved_version_config(config, "portfolio_risk_overlay_v2_7")
    v27_targets, _ = v26.overlay_targets_local(base_targets, panel, v27_cfg, "HIRSSM_V3_0_V2_7_CANDIDATE")
    candidates["v3_0_v2_7_risk_overlay"] = {"targets": v27_targets, "mode": "dynamic", "cfg": v27_cfg, "source": "V2.7 local risk overlay candidate"}

    v2101_targets = build_v2101_targets(model, wf, v22, v26, v210, panel, config, monthly_ic)
    candidates["v3_0_v2_10_1_governed"] = {"targets": v2101_targets, "mode": "dynamic", "cfg": v27_cfg, "source": "V2.10.1 governed risk-overlay candidate"}

    v30_dir = output_dir / "v3_0"
    v30_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_scores = []
    for name, item in candidates.items():
        candidate_dir = v30_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if item["mode"] == "dynamic":
            summary, _, _ = v26.run_costs(model, panel, item["targets"], item["cfg"], name, candidate_dir)
        else:
            summary, _ = run_static_costs(model, panel, item["targets"], candidate_dir, name, name)
        rel_summary = add_relative_metrics(summary)
        rel_summary["candidate"] = name
        all_summaries.append(rel_summary)
        all_scores.append(benchmark_relative_score(summary, name, item["source"]))
        model.write_csv(item["targets"], candidate_dir / "target_weights.csv")

    v30_summary_all = pd.concat(all_summaries, ignore_index=True, sort=False)
    v30_scores = pd.concat(all_scores, ignore_index=True, sort=False)
    v30_score_table = make_candidate_table(v30_scores)
    selected_v30 = str(v30_score_table.iloc[0]["variant"])
    selected_v30_targets = candidates[selected_v30]["targets"]
    selected_v30_summary = v30_summary_all[v30_summary_all["candidate"].eq(selected_v30)].drop(columns=["candidate"])
    selected_v30_smoke = wf.smoke_test_targets(selected_v30_targets)
    model.write_csv(v30_summary_all, v30_dir / "all_candidate_oos_performance.csv")
    model.write_csv(v30_scores, v30_dir / "benchmark_relative_score_detail.csv")
    model.write_csv(v30_score_table, v30_dir / "benchmark_relative_score_table.csv")
    model.write_csv(selected_v30_targets, v30_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected_v30_summary, v30_dir / "oos_performance.csv")
    model.write_csv(selected_v30_smoke, v30_dir / "smoke_test_results.csv")
    write_reports(
        output_dir,
        "v3_0",
        "HIRSSM V3.0 Benchmark-Relative Selection",
        selected_v30_summary,
        v30_scores,
        selected_v30,
        selected_v30_smoke,
        None,
        [
            "Changed selection objective to benchmark-relative performance against CSI All Share.",
            "Selected from predeclared active sleeves instead of optimizing full-sample return.",
            "No new alpha factor was added in V3.0.",
        ],
    )
    v30_self = make_self_check(selected_v30_smoke, selected_v30_summary, v30_score_table, output_dir, "v3_0")
    model.write_csv(v30_self, v30_dir / "self_check_results.csv")
    write_reports(
        output_dir,
        "v3_0",
        "HIRSSM V3.0 Benchmark-Relative Selection",
        selected_v30_summary,
        v30_scores,
        selected_v30,
        selected_v30_smoke,
        v30_self,
        [
            "Changed selection objective to benchmark-relative performance against CSI All Share.",
            "Selected from predeclared active sleeves instead of optimizing full-sample return.",
            "No new alpha factor was added in V3.0.",
        ],
    )

    v31_dir = output_dir / "v3_1"
    v31_dir.mkdir(parents=True, exist_ok=True)
    v31_summaries = []
    v31_scores_list = []
    v31_targets = {}
    for name, cfg in V31_VARIANTS.items():
        targets = build_core_satellite_targets(selected_v30_targets, name, cfg, BENCHMARK_ASSET)
        v31_targets[name] = targets
        candidate_dir = v31_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        summary, _ = run_static_costs(model, panel, targets, candidate_dir, name, name)
        rel_summary = add_relative_metrics(summary)
        rel_summary["candidate"] = name
        v31_summaries.append(rel_summary)
        v31_scores_list.append(benchmark_relative_score(summary, name, cfg["description"]))
        model.write_csv(targets, candidate_dir / "target_weights.csv")

    v31_summary_all = pd.concat(v31_summaries, ignore_index=True, sort=False)
    v31_scores = pd.concat(v31_scores_list, ignore_index=True, sort=False)
    v31_score_table = make_candidate_table(v31_scores)
    selected_v31 = str(v31_score_table.iloc[0]["variant"])
    selected_v31_targets = v31_targets[selected_v31]
    selected_v31_summary = v31_summary_all[v31_summary_all["candidate"].eq(selected_v31)].drop(columns=["candidate"])
    selected_v31_smoke = wf.smoke_test_targets(selected_v31_targets)
    model.write_csv(v31_summary_all, v31_dir / "all_candidate_oos_performance.csv")
    model.write_csv(v31_scores, v31_dir / "benchmark_relative_score_detail.csv")
    model.write_csv(v31_score_table, v31_dir / "benchmark_relative_score_table.csv")
    model.write_csv(selected_v31_targets, v31_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected_v31_summary, v31_dir / "oos_performance.csv")
    model.write_csv(selected_v31_smoke, v31_dir / "smoke_test_results.csv")
    write_reports(
        output_dir,
        "v3_1",
        "HIRSSM V3.1 Core-Satellite Selection",
        selected_v31_summary,
        v31_scores,
        selected_v31,
        selected_v31_smoke,
        None,
        [
            f"Built state-conditioned 000985 core-satellite variants on top of V3.0 selected sleeve `{selected_v30}`.",
            "Reallocated satellite non-cash holdings into explicit satellite budgets instead of inheriting all alpha-sleeve cash.",
            "Selected the best predeclared core-satellite schedule by the same benchmark-relative objective.",
        ],
    )
    v31_self = make_self_check(selected_v31_smoke, selected_v31_summary, v31_score_table, output_dir, "v3_1")
    model.write_csv(v31_self, v31_dir / "self_check_results.csv")
    write_reports(
        output_dir,
        "v3_1",
        "HIRSSM V3.1 Core-Satellite Selection",
        selected_v31_summary,
        v31_scores,
        selected_v31,
        selected_v31_smoke,
        v31_self,
        [
            f"Built state-conditioned 000985 core-satellite variants on top of V3.0 selected sleeve `{selected_v30}`.",
            "Reallocated satellite non-cash holdings into explicit satellite budgets instead of inheriting all alpha-sleeve cash.",
            "Selected the best predeclared core-satellite schedule by the same benchmark-relative objective.",
        ],
    )

    combined = pd.concat(
        [
            selected_v30_summary.assign(model_version="V3.0", selected_variant=selected_v30),
            selected_v31_summary.assign(model_version="V3.1", selected_variant=selected_v31),
        ],
        ignore_index=True,
        sort=False,
    )
    model.write_csv(combined, output_dir / "v3_0_v3_1_selected_performance.csv")
    manifest = {
        "generated_at": now_text(),
        "output_dir": str(output_dir),
        "v3_0_selected": selected_v30,
        "v3_0_self_check_pass": bool(v30_self["pass"].all()) if not v30_self.empty else False,
        "v3_1_selected": selected_v31,
        "v3_1_self_check_pass": bool(v31_self["pass"].all()) if not v31_self.empty else False,
        "costs": COSTS,
        "benchmark": BENCHMARK_ASSET,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
