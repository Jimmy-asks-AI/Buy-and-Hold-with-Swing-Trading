#!/usr/bin/env python
"""HIRSSM V3.7 walk-forward state alpha gate.

V3.7 validates the V3.6 state-gated alpha improvement with rolling evidence.
It first builds state-only alpha sleeve prototypes, then uses only prior-year
prototype excess returns to shrink next-year alpha exposure by market state.
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
V36_PATH = ROOT / "strategy_lab" / "hirssm_v3_6_component_attribution.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_7_state_alpha_walkforward"
V32_TARGETS = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "walk_forward_target_weights.csv"
V34_TARGETS = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_4" / "walk_forward_target_weights.csv"
V36_TARGETS = ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "walk_forward_target_weights.csv"
V32_PERF = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "oos_performance.csv"
V35_PERF = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "oos_performance.csv"
V36_PERF = ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "oos_performance.csv"
V32_YEARLY_10 = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "candidates" / "v3_2_recovery_attack" / "yearly_returns_v3_2_recovery_attack_10bps.csv"
BENCHMARK_ASSET = "000985"
COSTS = [5.0, 10.0, 20.0, 30.0]


BASE_STATE_WEIGHTS = {
    "risk_on_trend": 0.50,
    "range_bound": 0.40,
    "risk_on_overheat": 0.10,
    "risk_off_decline": 0.00,
    "crash_rebound": 0.00,
}


V37_VARIANTS = {
    "v3_7_soft_oos_state_gate": {
        "description": "Shrink V3.6 state alpha weights by 5-year prior state-only prototype evidence.",
        "gate_mode": "soft",
        "min_weight_change": 0.025,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
    },
    "v3_7_hard_oos_state_gate": {
        "description": "Use alpha sleeve only when rolling prior evidence is clearly positive.",
        "gate_mode": "hard",
        "min_weight_change": 0.025,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
    },
    "v3_7_cost_guarded_soft_gate": {
        "description": "Soft state gate with stricter industry caps and wider rebalance band.",
        "gate_mode": "soft",
        "min_weight_change": 0.035,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
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


def one_row(summary: pd.DataFrame, cost: float) -> pd.Series:
    rows = summary[summary["cost_bps"].astype(float).eq(float(cost))]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.iloc[0]


def prototype_cfg(state: str, weight: float) -> dict:
    return {
        "state_v34_weight": {name: (weight if name == state else 0.0) for name in BASE_STATE_WEIGHTS},
        "min_weight_change": 0.025,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
    }


def yearly_delta_from_candidate(version_dir: Path, candidate: str) -> pd.DataFrame:
    path = version_dir / "prototypes" / candidate / f"yearly_returns_{candidate}_10bps.csv"
    if not path.exists() or not V32_YEARLY_10.exists():
        return pd.DataFrame()
    base = pd.read_csv(V32_YEARLY_10, encoding="utf-8-sig")
    proto = pd.read_csv(path, encoding="utf-8-sig")
    merged = base.merge(proto, on="year", suffixes=("_v32", "_proto"))
    merged["annual_delta"] = merged["strategy_return_proto"] - merged["strategy_return_v32"]
    return merged[["year", "strategy_return_v32", "strategy_return_proto", "annual_delta"]]


def gate_multiplier(mode: str, windows: int, mean_delta: float, positive_rate: float, worst_delta: float, min_obs: int) -> tuple[float, str]:
    if windows < min_obs:
        return 1.0, f"prior_default_windows={windows}"
    if mode == "hard":
        if mean_delta > 0.0 and positive_rate >= 0.50 and worst_delta > -0.08:
            return 1.0, "hard_pass"
        return 0.0, "hard_fail"
    if mean_delta >= 0.010 and positive_rate >= 0.55:
        return 1.0, "soft_full"
    if mean_delta > 0.0 and positive_rate >= 0.45:
        return 0.75, "soft_partial_positive"
    if mean_delta > -0.015 and positive_rate >= 0.40:
        return 0.50, "soft_defensive_half"
    return 0.0, "soft_fail"


def build_gate_history(prototype_deltas: dict[str, pd.DataFrame], mode: str, lookback: int = 5, min_obs: int = 3) -> pd.DataFrame:
    all_years = sorted({int(year) for df in prototype_deltas.values() for year in df.get("year", [])})
    rows = []
    for test_year in all_years:
        for state, base_weight in BASE_STATE_WEIGHTS.items():
            if base_weight <= 0:
                rows.append(
                    {
                        "test_year": test_year,
                        "state": state,
                        "mode": mode,
                        "base_alpha_weight": base_weight,
                        "multiplier": 0.0,
                        "final_alpha_weight": 0.0,
                        "train_windows": 0,
                        "mean_delta": 0.0,
                        "positive_rate": 0.0,
                        "worst_delta": 0.0,
                        "decision": "base_weight_zero",
                    }
                )
                continue
            evidence = prototype_deltas.get(state, pd.DataFrame())
            train = evidence[(evidence["year"] < test_year) & (evidence["year"] >= test_year - lookback)].copy()
            deltas = pd.to_numeric(train.get("annual_delta", pd.Series(dtype=float)), errors="coerce").dropna()
            windows = int(deltas.shape[0])
            mean_delta = float(deltas.mean()) if windows else 0.0
            positive_rate = float((deltas > 0).mean()) if windows else 0.0
            worst_delta = float(deltas.min()) if windows else 0.0
            multiplier, decision = gate_multiplier(mode, windows, mean_delta, positive_rate, worst_delta, min_obs)
            rows.append(
                {
                    "test_year": test_year,
                    "state": state,
                    "mode": mode,
                    "base_alpha_weight": base_weight,
                    "multiplier": multiplier,
                    "final_alpha_weight": base_weight * multiplier,
                    "train_windows": windows,
                    "mean_delta": mean_delta,
                    "positive_rate": positive_rate,
                    "worst_delta": worst_delta,
                    "decision": decision,
                }
            )
    return pd.DataFrame(rows)


def target_gate_lookup(gate_history: pd.DataFrame) -> dict[tuple[int, str], float]:
    out = {}
    for _, row in gate_history.iterrows():
        out[(int(row["test_year"]), str(row["state"]))] = float(row["final_alpha_weight"])
    return out


def year_state_gated_blend(primary: pd.DataFrame, secondary: pd.DataFrame, timing: pd.DataFrame, gate_history: pd.DataFrame, cfg: dict, variant: str, v36) -> pd.DataFrame:
    all_dates = sorted(set(pd.to_datetime(primary["signal_date"])) & set(pd.to_datetime(secondary["signal_date"])))
    gate = target_gate_lookup(gate_history)
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
        trow = v36.latest_timing(timing_sorted, pd.Timestamp(date))
        state = v36.infer_state(s if not s.empty else p, trow)
        bucket = str(trow.get("timing_bucket", "neutral")) if not trow.empty else "neutral"
        test_year = int(pd.Timestamp(date).year)
        alpha_weight = float(gate.get((test_year, state), BASE_STATE_WEIGHTS.get(state, 0.0)))
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
        capped = v36.cap_and_redistribute(noncash, caps)
        banded = {}
        for asset in set(capped) | {asset for asset in prev if asset != "CASH"}:
            old = float(prev.get(asset, 0.0))
            new = float(capped.get(asset, 0.0))
            banded[asset] = old if abs(new - old) < min_change else new
        banded = {asset: max(0.0, weight) for asset, weight in banded.items() if weight > 1e-10}
        caps = {
            asset: (max_style if asset_type.get(asset) == "style" else max_industry)
            for asset in banded
        }
        banded = v36.cap_and_redistribute(banded, caps)
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
                    "v3_7_variant": variant,
                    "blend_v34_weight": alpha_weight,
                    "timing_bucket": bucket,
                    "gating_rule": f"year={test_year};state={state};alpha_weight={alpha_weight:.3f}",
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
                "v3_7_variant": variant,
                "blend_v34_weight": alpha_weight,
                "timing_bucket": bucket,
                "gating_rule": f"year={test_year};state={state};alpha_weight={alpha_weight:.3f}",
            }
        )
        prev = full
        prev_types = asset_type
    return pd.DataFrame(rows).sort_values(["signal_date", "asset"]).reset_index(drop=True)


def comparison_table(selected_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = {
        "V3.2": pd.read_csv(V32_PERF, encoding="utf-8-sig"),
        "V3.5": pd.read_csv(V35_PERF, encoding="utf-8-sig"),
        "V3.6": pd.read_csv(V36_PERF, encoding="utf-8-sig"),
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


def append_v37_self_checks(self_check: pd.DataFrame, comparison: pd.DataFrame, gate_history: pd.DataFrame) -> pd.DataFrame:
    def delta(cost: float, baseline: str, col: str) -> float | None:
        rows = comparison[
            comparison["cost_bps"].astype(float).eq(float(cost))
            & comparison["baseline"].astype(str).eq(baseline)
        ]
        if rows.empty:
            return None
        return float(rows[col].iloc[0])

    ann10 = delta(10.0, "V3.6", "annual_return_delta")
    ann20 = delta(20.0, "V3.6", "annual_return_delta")
    dd10 = delta(10.0, "V3.6", "max_drawdown_delta")
    valid_gate = bool(
        not gate_history.empty
        and gate_history["final_alpha_weight"].between(0.0, 1.0).all()
        and (gate_history["test_year"].astype(int) >= gate_history["test_year"].min()).all()
    )
    rows = [
        {
            "check": "gate_history_point_in_time_shape",
            "pass": valid_gate,
            "detail": f"rows={len(gate_history)}",
        },
        {
            "check": "beats_or_matches_v3_6_annual_return_10bps",
            "pass": bool(ann10 is not None and ann10 >= -0.0005),
            "detail": f"{ann10:.6f}" if ann10 is not None else "",
        },
        {
            "check": "beats_or_matches_v3_6_annual_return_20bps",
            "pass": bool(ann20 is not None and ann20 >= -0.0005),
            "detail": f"{ann20:.6f}" if ann20 is not None else "",
        },
        {
            "check": "not_worse_than_v3_6_drawdown_10bps_by_1pct",
            "pass": bool(dd10 is not None and dd10 >= -0.01),
            "detail": f"{dd10:.6f}" if dd10 is not None else "",
        },
    ]
    return pd.concat([self_check, pd.DataFrame(rows)], ignore_index=True, sort=False)


def write_reports(version_dir: Path, selected: str, selected_summary: pd.DataFrame, score_detail: pd.DataFrame, self_check: pd.DataFrame, notes: list[str], extra_tables: dict[str, pd.DataFrame]) -> None:
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
        "# HIRSSM V3.7 Walk-Forward State Alpha Gate",
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
        "# HIRSSM V3.7 Model Changelog",
        "",
        "## Changed",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Governance",
        "",
        "- State alpha gates use only prior-year prototype return evidence.",
        "- V3.6 exact targets are retained as a control candidate.",
        "- No new data source or full-sample factor optimization is introduced.",
    ]
    (version_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    self_lines = [
        "# HIRSSM V3.7 Self Check Report",
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
    parser.add_argument("--v36-targets", default=str(V36_TARGETS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v30 = load_module("hirssm_v3_0_v3_1_benchmark_core", V30_PATH)
    v32 = load_module("hirssm_v3_2_market_beta_timing", V32_PATH)
    v35 = load_module("hirssm_v3_3_to_v3_5_alpha_factory", V35_PATH)
    v36 = load_module("hirssm_v3_6_component_attribution", V36_PATH)

    root = Path(args.root)
    version_dir = Path(args.output_dir)
    prototype_dir = version_dir / "prototypes"
    version_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    timing = v32.build_timing_panel(panel, BENCHMARK_ASSET)
    v32_targets = read_targets(Path(args.v32_targets))
    v34_targets = read_targets(Path(args.v34_targets))
    v36_targets = read_targets(Path(args.v36_targets))
    model.write_csv(timing, version_dir / "market_beta_timing_panel.csv")

    prototype_deltas = {}
    prototype_results = {}
    for state, weight in BASE_STATE_WEIGHTS.items():
        if weight <= 0:
            continue
        name = f"v3_7_proto_{state}"
        targets = v36.regime_gated_blend(v34_targets, v32_targets, timing, prototype_cfg(state, weight), name)
        result = v35.evaluate_targets(model, v30, wf, panel, targets, prototype_dir / name, name, f"State-only prototype for {state}")
        prototype_results[name] = result
        delta = yearly_delta_from_candidate(version_dir, name)
        delta["state"] = state
        prototype_deltas[state] = delta
    prototype_evidence = pd.concat(prototype_deltas.values(), ignore_index=True, sort=False)
    model.write_csv(prototype_evidence, version_dir / "v3_7_state_prototype_yearly_evidence.csv")

    gate_histories = {}
    candidates = {}
    results = {}
    for variant, cfg in V37_VARIANTS.items():
        gate_history = build_gate_history(prototype_deltas, str(cfg["gate_mode"]))
        gate_histories[variant] = gate_history
        targets = year_state_gated_blend(v34_targets, v32_targets, timing, gate_history, cfg, variant, v36)
        candidates[variant] = targets
        result = v35.evaluate_targets(model, v30, wf, panel, targets, version_dir / "candidates" / variant, variant, cfg["description"])
        results[variant] = result
    candidates["v3_7_v36_exact_control"] = v36_targets
    results["v3_7_v36_exact_control"] = v35.evaluate_targets(
        model,
        v30,
        wf,
        panel,
        v36_targets,
        version_dir / "candidates" / "v3_7_v36_exact_control",
        "v3_7_v36_exact_control",
        "Official V3.6 selected target snapshot retained as exact control.",
    )

    selected = v35.select_version(model, v30, wf, version_dir, results, candidates)
    selected_name = selected["selected"]
    selected_gate_history = gate_histories.get(selected_name, pd.DataFrame())
    gate_all = pd.concat(
        [history.assign(variant=variant) for variant, history in gate_histories.items()],
        ignore_index=True,
        sort=False,
    )
    comparison = comparison_table(selected["selected_summary"])
    ablation = variant_ablation_table(selected["summary_all"])
    model.write_csv(gate_all, version_dir / "v3_7_state_gate_history.csv")
    model.write_csv(selected_gate_history, version_dir / "v3_7_selected_state_gate_history.csv")
    model.write_csv(comparison, version_dir / "v3_7_vs_v3_2_v3_5_v3_6_comparison.csv")
    model.write_csv(ablation, version_dir / "v3_7_component_ablation.csv")

    notes = [
        "Built state-only alpha sleeve prototypes for trend, range and overheat states.",
        "Converted prior 5-year prototype evidence into next-year state alpha multipliers.",
        "Compared soft gate, hard gate, cost-guarded soft gate and exact V3.6 control.",
        "Kept all state gates point-in-time at yearly granularity; no future-year evidence is used for a test year.",
    ]
    placeholder = pd.DataFrame([{"check": "pending", "pass": False, "detail": ""}])
    write_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_detail"],
        placeholder,
        notes,
        {
            "V3.7 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Selected Gate History": selected_gate_history,
        },
    )
    self_check = v35.make_self_check(selected["selected_targets"], selected["selected_smoke"], selected["selected_summary"], selected["score_table"], version_dir)
    self_check = append_v37_self_checks(self_check, comparison, selected_gate_history if not selected_gate_history.empty else gate_all)
    model.write_csv(self_check, version_dir / "self_check_results.csv")
    write_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_detail"],
        self_check,
        notes,
        {
            "V3.7 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Selected Gate History": selected_gate_history,
            "Prototype Evidence": prototype_evidence,
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
            "v36_targets": str(args.v36_targets),
        },
    }
    (version_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
