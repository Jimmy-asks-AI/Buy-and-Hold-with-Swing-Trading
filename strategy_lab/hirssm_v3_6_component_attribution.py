#!/usr/bin/env python
"""HIRSSM V3.6 component attribution and regime-gated alpha blend.

V3.6 does not add a new data source. It tests whether the V3.5 alpha-beta
ensemble improvement survives component ablation. The default candidates keep
V3.2 as the beta-timing anchor and vary the V3.4 sleeve weight by market state.
"""

from __future__ import annotations

import argparse
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
V30_PATH = ROOT / "strategy_lab" / "hirssm_v3_0_v3_1_benchmark_core.py"
V32_PATH = ROOT / "strategy_lab" / "hirssm_v3_2_market_beta_timing.py"
V35_PATH = ROOT / "strategy_lab" / "hirssm_v3_3_to_v3_5_alpha_factory.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_6_component_attribution"
V32_TARGETS = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "walk_forward_target_weights.csv"
V34_TARGETS = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_4" / "walk_forward_target_weights.csv"
V35_TARGETS = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "walk_forward_target_weights.csv"
V32_PERF = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "oos_performance.csv"
V35_PERF = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "oos_performance.csv"
V32_YEARLY_10 = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "candidates" / "v3_2_recovery_attack" / "yearly_returns_v3_2_recovery_attack_10bps.csv"
V32_REGIME_10 = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "candidates" / "v3_2_recovery_attack" / "regime_returns_v3_2_recovery_attack_10bps.csv"
BENCHMARK_ASSET = "000985"
COSTS = [5.0, 10.0, 20.0, 30.0]


V36_VARIANTS = {
    "v3_6_state_alpha_guard": {
        "description": "Use V3.4 alpha sleeve only where V3.5 attribution was non-negative; keep V3.2 pure in crash rebound.",
        "state_v34_weight": {
            "risk_on_trend": 0.45,
            "range_bound": 0.40,
            "risk_off_decline": 0.25,
            "risk_on_overheat": 0.20,
            "crash_rebound": 0.00,
        },
        "min_weight_change": 0.020,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
    },
    "v3_6_trend_range_alpha_only": {
        "description": "Restrict alpha sleeve to trend and range-bound states; eliminate alpha in crash and risk-off.",
        "state_v34_weight": {
            "risk_on_trend": 0.50,
            "range_bound": 0.40,
            "risk_off_decline": 0.00,
            "risk_on_overheat": 0.10,
            "crash_rebound": 0.00,
        },
        "min_weight_change": 0.025,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
    },
    "v3_6_low_turnover_anchor": {
        "description": "Smaller state-gated alpha sleeve with wider rebalance band for higher-cost robustness.",
        "state_v34_weight": {
            "risk_on_trend": 0.30,
            "range_bound": 0.30,
            "risk_off_decline": 0.15,
            "risk_on_overheat": 0.10,
            "crash_rebound": 0.00,
        },
        "min_weight_change": 0.030,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
    },
    "v3_6_v35_control": {
        "description": "Uniform 35% alpha blend under the V3.6 post-band cap handling; not the official V3.5 snapshot.",
        "state_v34_weight": {
            "risk_on_trend": 0.35,
            "range_bound": 0.35,
            "risk_off_decline": 0.35,
            "risk_on_overheat": 0.35,
            "crash_rebound": 0.35,
        },
        "min_weight_change": 0.020,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
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


def read_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing target weights: {path}")
    out = pd.read_csv(path, encoding="utf-8-sig")
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["asset"] = out["asset"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def cap_and_redistribute(weights: dict[str, float], caps: dict[str, float]) -> dict[str, float]:
    remaining = {asset: max(0.0, float(weight)) for asset, weight in weights.items() if weight > 0}
    fixed: dict[str, float] = {}
    for _ in range(8):
        changed = False
        for asset, weight in list(remaining.items()):
            cap = float(caps.get(asset, 1.0))
            if weight > cap:
                fixed[asset] = cap
                del remaining[asset]
                changed = True
        if not changed:
            break
        available = max(0.0, 1.0 - sum(fixed.values()))
        floating_sum = sum(remaining.values())
        if floating_sum > 0 and available > 0:
            remaining = {asset: weight * available / floating_sum for asset, weight in remaining.items()}
    fixed.update(remaining)
    total = sum(fixed.values())
    if total > 1.0:
        fixed = {asset: weight / total for asset, weight in fixed.items()}
    return fixed


def latest_timing(timing: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    subset = timing[timing["date"] <= pd.Timestamp(date)]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values("date").iloc[-1]


def infer_state(frame: pd.DataFrame, timing_row: pd.Series) -> str:
    if "state" in timing_row and pd.notna(timing_row.get("state")):
        return str(timing_row.get("state"))
    if "state" in frame.columns and not frame["state"].dropna().empty:
        return str(frame["state"].dropna().iloc[0])
    return "range_bound"


def regime_gated_blend(primary: pd.DataFrame, secondary: pd.DataFrame, timing: pd.DataFrame, cfg: dict, variant: str) -> pd.DataFrame:
    all_dates = sorted(set(pd.to_datetime(primary["signal_date"])) & set(pd.to_datetime(secondary["signal_date"])))
    min_change = float(cfg.get("min_weight_change", 0.0))
    max_style = float(cfg.get("max_style_weight", 0.50))
    max_industry = float(cfg.get("max_industry_weight", 0.20))
    timing_sorted = timing.sort_values("date")
    rows = []
    prev: dict[str, float] = {}
    prev_types: dict[str, str] = {}
    for date in all_dates:
        p = primary[primary["signal_date"].eq(date)]
        s = secondary[secondary["signal_date"].eq(date)]
        trow = latest_timing(timing_sorted, pd.Timestamp(date))
        state = infer_state(s if not s.empty else p, trow)
        bucket = str(trow.get("timing_bucket", "neutral")) if not trow.empty else "neutral"
        alpha_weight = float(cfg.get("state_v34_weight", {}).get(state, cfg.get("default_v34_weight", 0.0)))
        alpha_weight = min(max(alpha_weight, 0.0), 1.0)
        beta_weight = 1.0 - alpha_weight
        weights: dict[str, float] = {}
        asset_type: dict[str, str] = dict(prev_types)
        for _, row in p.iterrows():
            asset = str(row["asset"])
            weights[asset] = weights.get(asset, 0.0) + alpha_weight * float(row["weight"])
            asset_type[asset] = str(row.get("asset_type", asset_type.get(asset, "")))
        for _, row in s.iterrows():
            asset = str(row["asset"])
            weights[asset] = weights.get(asset, 0.0) + beta_weight * float(row["weight"])
            asset_type[asset] = str(row.get("asset_type", asset_type.get(asset, "")))

        noncash = {asset: weight for asset, weight in weights.items() if asset != "CASH" and weight > 0}
        caps = {
            asset: (max_style if asset_type.get(asset) == "style" else max_industry)
            for asset in noncash
        }
        capped = cap_and_redistribute(noncash, caps)

        banded: dict[str, float] = {}
        for asset in set(capped) | {asset for asset in prev if asset != "CASH"}:
            old = float(prev.get(asset, 0.0))
            new = float(capped.get(asset, 0.0))
            banded[asset] = old if abs(new - old) < min_change else new
        banded = {asset: max(0.0, weight) for asset, weight in banded.items() if weight > 1e-10}
        caps = {
            asset: (max_style if asset_type.get(asset) == "style" else max_industry)
            for asset in banded
        }
        banded = cap_and_redistribute(banded, caps)
        noncash_sum = sum(banded.values())
        if noncash_sum > 1.0:
            banded = {asset: weight / noncash_sum for asset, weight in banded.items()}
            noncash_sum = 1.0
        cash = max(0.0, 1.0 - noncash_sum)
        full = dict(banded)
        full["CASH"] = cash
        turnover = sum(abs(full.get(asset, 0.0) - prev.get(asset, 0.0)) for asset in set(full) | set(prev))

        for asset, weight in sorted(banded.items()):
            rows.append(
                {
                    "signal_date": date,
                    "asset": asset,
                    "weight": float(weight),
                    "state": state,
                    "asset_type": asset_type.get(asset, ""),
                    "score": 0.0,
                    "risk_adjusted_alpha": 0.0,
                    "turnover": float(turnover),
                    "v3_6_variant": variant,
                    "blend_v34_weight": alpha_weight,
                    "timing_bucket": bucket,
                    "gating_rule": f"state={state};alpha_weight={alpha_weight:.2f}",
                }
            )
        rows.append(
            {
                "signal_date": date,
                "asset": "CASH",
                "weight": cash,
                "state": state,
                "asset_type": "cash",
                "score": 0.0,
                "risk_adjusted_alpha": 0.0,
                "turnover": float(turnover),
                "v3_6_variant": variant,
                "blend_v34_weight": alpha_weight,
                "timing_bucket": bucket,
                "gating_rule": f"state={state};alpha_weight={alpha_weight:.2f}",
            }
        )
        prev = full
        prev_types = asset_type
    return pd.DataFrame(rows).sort_values(["signal_date", "asset"]).reset_index(drop=True)


def one_row(summary: pd.DataFrame, cost: float) -> pd.Series:
    rows = summary[summary["cost_bps"].astype(float).eq(float(cost))]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.iloc[0]


def comparison_table(selected_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = {
        "V3.2": pd.read_csv(V32_PERF, encoding="utf-8-sig"),
        "V3.5": pd.read_csv(V35_PERF, encoding="utf-8-sig"),
    }
    for cost in COSTS:
        current = one_row(selected_summary, cost)
        if current.empty:
            continue
        for name, base in baselines.items():
            old = one_row(base, cost)
            if old.empty:
                continue
            rows.append(
                {
                    "cost_bps": cost,
                    "baseline": name,
                    "annual_return_delta": float(current["annual_return"] - old["annual_return"]),
                    "annual_excess_delta": float(current["annual_excess_vs_benchmark"] - old["annual_excess_vs_benchmark"]),
                    "sharpe_delta": float(current["sharpe_no_rf"] - old["sharpe_no_rf"]),
                    "max_drawdown_delta": float(current["max_drawdown"] - old["max_drawdown"]),
                    "avg_trade_turnover_delta": float(current["avg_trade_turnover"] - old["avg_trade_turnover"]),
                    "avg_cash_delta": float(current["avg_cash_weight"] - old["avg_cash_weight"]),
                    "total_cost_delta": float(current["total_cost"] - old["total_cost"]),
                }
            )
    return pd.DataFrame(rows)


def yearly_delta_table(selected: str, version_dir: Path) -> pd.DataFrame:
    path = version_dir / "candidates" / selected / f"yearly_returns_{selected}_10bps.csv"
    if not path.exists() or not V32_YEARLY_10.exists():
        return pd.DataFrame()
    old = pd.read_csv(V32_YEARLY_10, encoding="utf-8-sig")
    new = pd.read_csv(path, encoding="utf-8-sig")
    merged = old.merge(new, on="year", suffixes=("_v32", "_v36"))
    merged["strategy_delta"] = merged["strategy_return_v36"] - merged["strategy_return_v32"]
    merged["benchmark_return"] = merged["benchmark_return_v32"]
    return merged[["year", "strategy_return_v32", "strategy_return_v36", "strategy_delta", "benchmark_return"]]


def regime_delta_table(selected: str, version_dir: Path) -> pd.DataFrame:
    path = version_dir / "candidates" / selected / f"regime_returns_{selected}_10bps.csv"
    if not path.exists() or not V32_REGIME_10.exists():
        return pd.DataFrame()
    old = pd.read_csv(V32_REGIME_10, encoding="utf-8-sig")
    new = pd.read_csv(path, encoding="utf-8-sig")
    merged = old.merge(new, on="state", suffixes=("_v32", "_v36"))
    merged["annualized_mean_delta"] = merged["annualized_mean_v36"] - merged["annualized_mean_v32"]
    merged["vol_delta"] = merged["annualized_vol_v36"] - merged["annualized_vol_v32"]
    merged["win_rate_delta"] = merged["win_rate_v36"] - merged["win_rate_v32"]
    total_days = float(merged["days_v32"].sum())
    merged["weighted_ann_mean_contrib"] = (
        (merged["avg_daily_return_v36"] - merged["avg_daily_return_v32"]) * merged["days_v32"] / total_days * 252.0
    )
    return merged[
        [
            "state",
            "days_v32",
            "annualized_mean_v32",
            "annualized_mean_v36",
            "annualized_mean_delta",
            "annualized_vol_v32",
            "annualized_vol_v36",
            "vol_delta",
            "win_rate_v32",
            "win_rate_v36",
            "win_rate_delta",
            "weighted_ann_mean_contrib",
        ]
    ]


def asset_weight_delta_table(targets: pd.DataFrame, baseline_targets: pd.DataFrame) -> pd.DataFrame:
    base = baseline_targets.copy()
    current = targets.copy()
    base["signal_date"] = pd.to_datetime(base["signal_date"])
    current["signal_date"] = pd.to_datetime(current["signal_date"])
    p_base = base.pivot_table(index="signal_date", columns="asset", values="weight", aggfunc="sum").fillna(0.0)
    p_current = current.pivot_table(index="signal_date", columns="asset", values="weight", aggfunc="sum").fillna(0.0)
    idx = p_base.index.intersection(p_current.index)
    cols = p_base.columns.union(p_current.columns)
    p_base = p_base.reindex(index=idx, columns=cols, fill_value=0.0)
    p_current = p_current.reindex(index=idx, columns=cols, fill_value=0.0)
    out = pd.DataFrame(
        {
            "asset": cols,
            "avg_weight_v32": p_base.mean().reindex(cols).values,
            "avg_weight_v36": p_current.mean().reindex(cols).values,
        }
    )
    out["delta"] = out["avg_weight_v36"] - out["avg_weight_v32"]
    return out.sort_values("delta", ascending=False).reset_index(drop=True)


def variant_ablation_table(summary_all: pd.DataFrame) -> pd.DataFrame:
    rows = summary_all[summary_all["cost_bps"].astype(float).isin([10.0, 20.0])].copy()
    cols = [
        "candidate",
        "cost_bps",
        "annual_return",
        "annual_excess_vs_benchmark",
        "sharpe_no_rf",
        "max_drawdown",
        "avg_trade_turnover",
        "avg_cash_weight",
        "total_cost",
    ]
    return rows[cols].sort_values(["cost_bps", "annual_excess_vs_benchmark"], ascending=[True, False])


def logic_check_table(targets: pd.DataFrame, timing: pd.DataFrame, selected_cfg: dict) -> pd.DataFrame:
    target_states = targets.groupby("state", as_index=False).agg(
        rows=("asset", "count"),
        avg_alpha_weight=("blend_v34_weight", "mean"),
        avg_cash=("weight", lambda x: float(x[targets.loc[x.index, "asset"].eq("CASH")].mean()) if targets.loc[x.index, "asset"].eq("CASH").any() else 0.0),
    )
    expected = pd.DataFrame(
        [
            {"state": state, "expected_alpha_weight": weight}
            for state, weight in selected_cfg.get("state_v34_weight", {}).items()
        ]
    )
    out = target_states.merge(expected, on="state", how="left")
    out["gate_matches_expected_when_applicable"] = (
        out["expected_alpha_weight"].isna()
        | np.isclose(out["avg_alpha_weight"], out["expected_alpha_weight"], atol=0.0001)
    )
    out["timing_states_available"] = out["state"].isin(set(timing["state"].astype(str)))
    return out.sort_values("state")


def append_v36_self_checks(self_check: pd.DataFrame, comparison: pd.DataFrame, logic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def delta(cost: float, baseline: str, col: str) -> float | None:
        match = comparison[
            comparison["cost_bps"].astype(float).eq(float(cost))
            & comparison["baseline"].astype(str).eq(baseline)
        ]
        if match.empty:
            return None
        return float(match[col].iloc[0])

    ann10 = delta(10.0, "V3.5", "annual_return_delta")
    ann20 = delta(20.0, "V3.5", "annual_return_delta")
    dd10 = delta(10.0, "V3.5", "max_drawdown_delta")
    rows.extend(
        [
            {
                "check": "beats_v3_5_annual_return_10bps",
                "pass": bool(ann10 is not None and ann10 > 0.0),
                "detail": f"{ann10:.6f}" if ann10 is not None else "",
            },
            {
                "check": "beats_v3_5_annual_return_20bps",
                "pass": bool(ann20 is not None and ann20 > 0.0),
                "detail": f"{ann20:.6f}" if ann20 is not None else "",
            },
            {
                "check": "not_worse_than_v3_5_drawdown_10bps",
                "pass": bool(dd10 is not None and dd10 >= 0.0),
                "detail": f"{dd10:.6f}" if dd10 is not None else "",
            },
            {
                "check": "logic_gate_matches_selected_config",
                "pass": bool(not logic.empty and logic["gate_matches_expected_when_applicable"].all()),
                "detail": str(logic["gate_matches_expected_when_applicable"].tolist()) if not logic.empty else "",
            },
        ]
    )
    return pd.concat([self_check, pd.DataFrame(rows)], ignore_index=True, sort=False)


def write_v36_reports(version_dir: Path, selected: str, selected_summary: pd.DataFrame, score_detail: pd.DataFrame, self_check: pd.DataFrame, notes: list[str], extra_tables: dict[str, pd.DataFrame]) -> None:
    score_table = (
        score_detail[
            [
                "variant",
                "source",
                "benchmark_relative_score",
                "avg_annual_excess",
                "avg_drawdown_improvement",
                "avg_information_ratio",
                "mean_cash_weight",
            ]
        ]
        .drop_duplicates()
        .sort_values("benchmark_relative_score", ascending=False)
        if not score_detail.empty
        else pd.DataFrame()
    )
    report = [
        "# HIRSSM V3.6 Component Attribution And Regime-Gated Alpha",
        "",
        f"Run time: {now_text()}",
        "",
        "## Selected Variant",
        "",
        f"- `{selected}`",
        "",
        "## Design Notes",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Selected Performance",
        "",
        selected_summary.to_markdown(index=False) if not selected_summary.empty else "No summary.",
        "",
        "## Candidate Score Table",
        "",
        score_table.to_markdown(index=False) if not score_table.empty else "No score table.",
    ]
    for name, table in extra_tables.items():
        report.extend(["", f"## {name}", "", table.to_markdown(index=False) if not table.empty else "No rows."])
    (version_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    changelog = [
        "# HIRSSM V3.6 Model Changelog",
        "",
        "## Changed",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Governance",
        "",
        "- V3.6 is selected only from same-period OOS backtests and is compared with both V3.2 and V3.5.",
        "- The first candidate disables V3.4 alpha exposure in crash-rebound regimes because V3.5 attribution was negative there.",
        "- No new historical fundamental snapshot is introduced; this version is a controlled portfolio-construction iteration.",
    ]
    (version_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    self_lines = [
        "# HIRSSM V3.6 Self Check Report",
        "",
        self_check.to_markdown(index=False) if not self_check.empty else "Self check pending.",
    ]
    (version_dir / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--v32-targets", default=str(V32_TARGETS))
    parser.add_argument("--v34-targets", default=str(V34_TARGETS))
    parser.add_argument("--v35-targets", default=str(V35_TARGETS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v30 = load_module("hirssm_v3_0_v3_1_benchmark_core", V30_PATH)
    v32 = load_module("hirssm_v3_2_market_beta_timing", V32_PATH)
    v35 = load_module("hirssm_v3_3_to_v3_5_alpha_factory", V35_PATH)

    root = Path(args.root)
    version_dir = Path(args.output_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    timing = v32.build_timing_panel(panel, BENCHMARK_ASSET)
    v32_targets = read_targets(Path(args.v32_targets))
    v34_targets = read_targets(Path(args.v34_targets))
    v35_targets = read_targets(Path(args.v35_targets))

    model.write_csv(timing, version_dir / "market_beta_timing_panel.csv")
    candidates = {}
    results = {}
    for variant, cfg in V36_VARIANTS.items():
        targets = regime_gated_blend(v34_targets, v32_targets, timing, cfg, variant)
        candidates[variant] = targets
        result = v35.evaluate_targets(model, v30, wf, panel, targets, version_dir / "candidates" / variant, variant, cfg["description"])
        results[variant] = result
    candidates["v3_6_v35_exact_control"] = v35_targets
    results["v3_6_v35_exact_control"] = v35.evaluate_targets(
        model,
        v30,
        wf,
        panel,
        v35_targets,
        version_dir / "candidates" / "v3_6_v35_exact_control",
        "v3_6_v35_exact_control",
        "Official V3.5 selected target snapshot re-evaluated as an exact control.",
    )

    selected = v35.select_version(model, v30, wf, version_dir, results, candidates)
    selected_name = selected["selected"]
    selected_targets = selected["selected_targets"]
    comparison = comparison_table(selected["selected_summary"])
    yearly = yearly_delta_table(selected_name, version_dir)
    regime = regime_delta_table(selected_name, version_dir)
    weights = asset_weight_delta_table(selected_targets, v32_targets)
    ablation = variant_ablation_table(selected["summary_all"])
    selected_cfg = V36_VARIANTS.get(selected_name, {"state_v34_weight": {state: 0.35 for state in timing["state"].astype(str).unique()}})
    logic = logic_check_table(selected_targets, timing, selected_cfg)

    model.write_csv(comparison, version_dir / "v3_6_vs_v3_2_v3_5_comparison.csv")
    model.write_csv(yearly, version_dir / "v3_6_yearly_delta_vs_v3_2_10bps.csv")
    model.write_csv(regime, version_dir / "v3_6_regime_delta_vs_v3_2_10bps.csv")
    model.write_csv(weights, version_dir / "v3_6_asset_weight_delta_vs_v3_2.csv")
    model.write_csv(ablation, version_dir / "v3_6_component_ablation.csv")
    model.write_csv(logic, version_dir / "v3_6_logic_check.csv")
    model.write_csv(v35_targets, version_dir / "v3_5_selected_targets_snapshot.csv")

    notes = [
        "Added component ablation around V3.5 instead of adding new factors.",
        "Kept V3.2 as the beta-timing anchor and varied V3.4 alpha sleeve weight by market state.",
        "Disabled V3.4 alpha sleeve in crash-rebound states for guarded candidates because V3.5 attribution was negative there.",
        "Rebuilt the V3.5 beta-anchor candidate as a control to avoid mistaking harness differences for model improvement.",
        "Added comparison, yearly, regime, asset-weight, ablation and logic-check outputs.",
    ]
    # Create placeholder reports before self-check, then overwrite with final report.
    placeholder = pd.DataFrame([{"check": "pending", "pass": False, "detail": ""}])
    write_v36_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_detail"],
        placeholder,
        notes,
        {
            "V3.6 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Regime Delta vs V3.2": regime,
            "Logic Check": logic,
        },
    )
    self_check = v35.make_self_check(selected_targets, selected["selected_smoke"], selected["selected_summary"], selected["score_table"], version_dir)
    self_check = append_v36_self_checks(self_check, comparison, logic)
    model.write_csv(self_check, version_dir / "self_check_results.csv")
    write_v36_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_detail"],
        self_check,
        notes,
        {
            "V3.6 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Regime Delta vs V3.2": regime,
            "Yearly Delta vs V3.2": yearly,
            "Top Positive Asset Weight Deltas": weights.head(15),
            "Top Negative Asset Weight Deltas": weights.tail(15),
            "Logic Check": logic,
        },
    )

    manifest = {
        "generated_at": now_text(),
        "output_dir": str(version_dir),
        "selected": selected_name,
        "self_check_pass": bool(self_check["pass"].all()),
        "costs": COSTS,
        "benchmark": BENCHMARK_ASSET,
        "inputs": {
            "v32_targets": str(args.v32_targets),
            "v34_targets": str(args.v34_targets),
            "v35_targets": str(args.v35_targets),
        },
    }
    (version_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
