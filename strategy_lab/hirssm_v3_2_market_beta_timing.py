#!/usr/bin/env python
"""HIRSSM V3.2 market beta timing overlay.

V3.2 keeps the V3.0 selected alpha sleeve and adds an independent market beta
timing layer. The overlay changes total non-cash exposure by market trend,
breadth, drawdown and recovery evidence. It does not use leverage and never
allows negative cash.
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
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing"
BASE_TARGETS = ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_0" / "walk_forward_target_weights.csv"
BENCHMARK_ASSET = "000985"
COSTS = [5.0, 10.0, 20.0, 30.0]


V32_VARIANTS = {
    "v3_2_balanced_beta": {
        "description": "Balanced beta timing: deploy cash only when trend and breadth are constructive.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 0.95,
            "neutral": 0.90,
            "cautious": 0.72,
            "risk_off": 0.48,
            "panic": 0.32,
        },
        "min_gross_by_state": {
            "risk_on_trend": 0.92,
            "crash_rebound": 0.72,
        },
        "max_gross_by_state": {
            "risk_off_decline": 0.68,
            "risk_on_overheat": 0.94,
        },
    },
    "v3_2_recovery_attack": {
        "description": "Aggressive recovery participation after deep drawdown stabilization.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 1.00,
            "neutral": 0.96,
            "cautious": 0.80,
            "risk_off": 0.52,
            "panic": 0.30,
        },
        "min_gross_by_state": {
            "risk_on_trend": 0.96,
            "crash_rebound": 0.86,
        },
        "max_gross_by_state": {
            "risk_off_decline": 0.72,
            "risk_on_overheat": 0.98,
        },
    },
    "v3_2_1_recovery_attack_smooth": {
        "description": "V3.2.1 recovery attack with turnover-aware gross exposure smoothing.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 1.00,
            "neutral": 0.96,
            "cautious": 0.80,
            "risk_off": 0.52,
            "panic": 0.30,
        },
        "min_gross_by_state": {
            "risk_on_trend": 0.96,
            "crash_rebound": 0.86,
        },
        "max_gross_by_state": {
            "risk_off_decline": 0.72,
            "risk_on_overheat": 0.98,
        },
        "gross_change_buffer": 0.04,
        "max_gross_step_up": 0.16,
        "max_gross_step_down": 0.22,
    },
    "v3_2_drawdown_guard": {
        "description": "Defensive beta timing: preserve V3.0 upside but cut harder when breadth breaks.",
        "target_gross_by_bucket": {
            "risk_on": 0.98,
            "recovery": 0.90,
            "neutral": 0.84,
            "cautious": 0.64,
            "risk_off": 0.40,
            "panic": 0.24,
        },
        "min_gross_by_state": {
            "risk_on_trend": 0.88,
            "crash_rebound": 0.64,
        },
        "max_gross_by_state": {
            "risk_off_decline": 0.56,
            "risk_on_overheat": 0.90,
        },
    },
    "v3_2_trend_follow_beta": {
        "description": "Trend-following beta overlay: nearly full invested in broad uptrends, otherwise cautious.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 0.92,
            "neutral": 0.88,
            "cautious": 0.62,
            "risk_off": 0.36,
            "panic": 0.20,
        },
        "min_gross_by_state": {
            "risk_on_trend": 0.98,
        },
        "max_gross_by_state": {
            "risk_off_decline": 0.50,
            "crash_rebound": 0.92,
            "risk_on_overheat": 0.92,
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


def clip(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, lo), hi))


def smooth_clip(value: object, scale: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x) or scale == 0:
        return 0.0
    return clip(x / scale, lo, hi)


def read_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing base targets: {path}")
    out = pd.read_csv(path, encoding="utf-8-sig")
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def build_timing_panel(panel: dict, benchmark_asset: str = BENCHMARK_ASSET) -> pd.DataFrame:
    eligible = panel["eligible"].copy()
    regimes = panel["regimes"].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    regimes["date"] = pd.to_datetime(regimes["date"])
    market = eligible[eligible["asset"].astype(str).eq(benchmark_asset)].copy()
    if market.empty:
        raise ValueError(f"benchmark asset {benchmark_asset} not found in eligible panel")
    market = market.merge(regimes, on="date", how="left", suffixes=("", "_regime"))
    market = market.sort_values("date")
    market["vol60_median_252"] = market["vol_60"].rolling(252, min_periods=80).median()
    rows = []
    for _, row in market.iterrows():
        state = str(row.get("state", "range_bound"))
        trend_score = (
            0.30 * smooth_clip(row.get("ma_gap_200"), 0.08)
            + 0.22 * smooth_clip(row.get("ma_gap_60"), 0.06)
            + 0.22 * smooth_clip(row.get("ret_60"), 0.12)
            + 0.18 * smooth_clip(row.get("ret_120"), 0.20)
            + 0.08 * smooth_clip(row.get("ma_slope_60"), 0.08)
        )
        breadth_ma = row.get("industry_above_ma60_ratio")
        breadth_ret = row.get("industry_positive_ret20_ratio")
        breadth_score = (
            0.65 * clip((float(breadth_ma) - 0.50) / 0.30, -1.0, 1.0) if pd.notna(breadth_ma) else 0.0
        ) + (
            0.35 * clip((float(breadth_ret) - 0.50) / 0.30, -1.0, 1.0) if pd.notna(breadth_ret) else 0.0
        )
        vol_ratio = float(row["vol_60"] / row["vol60_median_252"]) if pd.notna(row.get("vol_60")) and pd.notna(row.get("vol60_median_252")) and row.get("vol60_median_252") else 1.0
        drawdown = float(row.get("max_drawdown_120")) if pd.notna(row.get("max_drawdown_120")) else 0.0
        risk_score = -0.55 * clip((vol_ratio - 1.0) / 0.55, -1.0, 1.0) + 0.45 * clip((drawdown + 0.12) / 0.16, -1.0, 1.0)
        recovery_score = 0.0
        if drawdown <= -0.16 and pd.notna(row.get("ret_20")) and float(row["ret_20"]) > 0.02:
            recovery_score += 0.45
        if drawdown <= -0.20 and pd.notna(row.get("ma_gap_60")) and float(row["ma_gap_60"]) > -0.06:
            recovery_score += 0.35
        if pd.notna(breadth_ret) and float(breadth_ret) > 0.55 and pd.notna(row.get("ret_20")) and float(row["ret_20"]) > 0:
            recovery_score += 0.20
        total_score = 0.45 * trend_score + 0.30 * breadth_score + 0.15 * risk_score + 0.10 * recovery_score

        if drawdown <= -0.28 and total_score < -0.35:
            bucket = "panic"
        elif recovery_score >= 0.55 and total_score > -0.25:
            bucket = "recovery"
        elif total_score >= 0.35 and state in {"risk_on_trend", "crash_rebound", "range_bound"}:
            bucket = "risk_on"
        elif total_score >= 0.05:
            bucket = "neutral"
        elif total_score <= -0.55 or state == "risk_off_decline":
            bucket = "risk_off"
        else:
            bucket = "cautious"

        rows.append(
            {
                "date": row["date"],
                "state": state,
                "timing_score": float(total_score),
                "trend_score": float(trend_score),
                "breadth_score": float(breadth_score),
                "risk_score": float(risk_score),
                "recovery_score": float(recovery_score),
                "timing_bucket": bucket,
                "market_ret_20": float(row.get("ret_20")) if pd.notna(row.get("ret_20")) else np.nan,
                "market_ret_60": float(row.get("ret_60")) if pd.notna(row.get("ret_60")) else np.nan,
                "market_ma_gap_200": float(row.get("ma_gap_200")) if pd.notna(row.get("ma_gap_200")) else np.nan,
                "market_drawdown_120": drawdown,
                "breadth_ma60": float(breadth_ma) if pd.notna(breadth_ma) else np.nan,
                "breadth_ret20": float(breadth_ret) if pd.notna(breadth_ret) else np.nan,
                "vol_ratio": vol_ratio,
            }
        )
    return pd.DataFrame(rows)


def latest_timing_before(timing: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    subset = timing[timing["date"] <= pd.Timestamp(date)]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values("date").iloc[-1]


def target_gross_for(row: pd.Series, cfg: dict) -> tuple[float, str]:
    bucket = str(row.get("timing_bucket", "neutral"))
    state = str(row.get("state", "range_bound"))
    gross = float(cfg["target_gross_by_bucket"].get(bucket, cfg["target_gross_by_bucket"].get("neutral", 0.85)))
    min_state = cfg.get("min_gross_by_state", {}).get(state)
    max_state = cfg.get("max_gross_by_state", {}).get(state)
    reason = f"bucket={bucket}"
    if min_state is not None and gross < float(min_state):
        gross = float(min_state)
        reason += f"; raised_to_state_min_{state}"
    if max_state is not None and gross > float(max_state):
        gross = float(max_state)
        reason += f"; capped_by_state_max_{state}"
    return clip(gross, 0.0, 1.0), reason


def overlay_beta_targets(base_targets: pd.DataFrame, timing: pd.DataFrame, cfg: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    decisions = []
    base = base_targets.copy()
    base["signal_date"] = pd.to_datetime(base["signal_date"])
    prev_target_gross: float | None = None
    for signal_date, group in base.sort_values(["signal_date", "asset"]).groupby("signal_date"):
        timing_row = latest_timing_before(timing, signal_date)
        if timing_row.empty:
            state = str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else "range_bound"
            timing_row = pd.Series({"state": state, "timing_bucket": "neutral", "timing_score": 0.0})
        raw_target_gross, reason = target_gross_for(timing_row, cfg)
        target_gross = raw_target_gross
        if prev_target_gross is not None and any(key in cfg for key in ["gross_change_buffer", "max_gross_step_up", "max_gross_step_down"]):
            diff = raw_target_gross - prev_target_gross
            buffer = float(cfg.get("gross_change_buffer", 0.0))
            if abs(diff) < buffer:
                target_gross = prev_target_gross
                reason += f"; smoothing_hold_prev_gross_{prev_target_gross:.2f}"
            else:
                step_up = float(cfg.get("max_gross_step_up", 1.0))
                step_down = float(cfg.get("max_gross_step_down", 1.0))
                clipped = min(diff, step_up) if diff > 0 else max(diff, -step_down)
                target_gross = prev_target_gross + clipped
                if abs(clipped - diff) > 1e-12:
                    reason += f"; smoothing_step_limited_from_{raw_target_gross:.2f}_to_{target_gross:.2f}"
        target_gross = clip(target_gross, 0.0, 1.0)
        prev_target_gross = target_gross
        noncash = group[~group["asset"].astype(str).eq("CASH")].copy()
        noncash["weight"] = pd.to_numeric(noncash["weight"], errors="coerce").fillna(0.0).clip(lower=0)
        current_gross = float(noncash["weight"].sum())
        scale = target_gross / current_gross if current_gross > 0 else 0.0
        scaled = noncash.copy()
        scaled["weight"] = scaled["weight"] * scale
        noncash_sum = float(scaled["weight"].sum())
        if noncash_sum > 1.0:
            scaled["weight"] = scaled["weight"] / noncash_sum
            noncash_sum = 1.0
        cash = max(0.0, 1.0 - noncash_sum)
        for _, item in scaled.iterrows():
            if float(item["weight"]) <= 0:
                continue
            row = item.to_dict()
            row["weight"] = float(item["weight"])
            row["v3_2_variant"] = variant
            row["base_gross"] = current_gross
            row["target_gross"] = target_gross
            row["beta_scale"] = scale
            row["timing_bucket"] = str(timing_row.get("timing_bucket", "neutral"))
            row["timing_score"] = float(timing_row.get("timing_score", 0.0))
            row["timing_reason"] = reason
            rows.append(row)
        cash_row = group[group["asset"].astype(str).eq("CASH")].head(1)
        template = cash_row.iloc[0].to_dict() if not cash_row.empty else group.iloc[0].to_dict()
        template.update(
            {
                "asset": "CASH",
                "weight": cash,
                "asset_type": "cash",
                "score": 0.0,
                "risk_adjusted_alpha": 0.0,
                "v3_2_variant": variant,
                "base_gross": current_gross,
                "target_gross": target_gross,
                "beta_scale": scale,
                "timing_bucket": str(timing_row.get("timing_bucket", "neutral")),
                "timing_score": float(timing_row.get("timing_score", 0.0)),
                "timing_reason": reason,
            }
        )
        rows.append(template)
        decisions.append(
            {
                "signal_date": signal_date,
                "state": str(timing_row.get("state", "")),
                "timing_bucket": str(timing_row.get("timing_bucket", "")),
                "timing_score": float(timing_row.get("timing_score", 0.0)),
                "base_gross": current_gross,
                "target_gross": target_gross,
                "raw_target_gross": raw_target_gross,
                "cash": cash,
                "beta_scale": scale,
                "reason": reason,
                "variant": variant,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame(decisions)
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out = out.sort_values(["signal_date", "asset"]).reset_index(drop=True)
    prev_weights: dict[str, float] = {}
    turnovers = {}
    for signal_date, group in out.groupby("signal_date", sort=True):
        current = {str(row["asset"]): float(row["weight"]) for _, row in group.iterrows()}
        turnover = sum(abs(current.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in set(current) | set(prev_weights))
        turnovers[pd.Timestamp(signal_date)] = turnover
        prev_weights = current
    out["turnover"] = out["signal_date"].map(turnovers)
    return out, pd.DataFrame(decisions)


def run_costs(model, v30, panel: dict, targets: pd.DataFrame, output_dir: Path, prefix: str, variant: str) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    return v30.run_static_costs(model, panel, targets, output_dir, prefix, variant)


def make_self_check(model, wf, targets: pd.DataFrame, summary: pd.DataFrame, score_table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    smoke = wf.smoke_test_targets(targets)
    rel = summary.copy()
    row10 = rel[rel["cost_bps"].astype(float).eq(10.0)].head(1) if not rel.empty else pd.DataFrame()
    weight_sums = targets.groupby("signal_date")["weight"].sum() if not targets.empty else pd.Series(dtype=float)
    rows = [
        {"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""},
        {"check": "required_cost_rows", "pass": bool(set(COSTS).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""},
        {"check": "no_negative_weights", "pass": bool((pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0) >= -1e-9).all()) if not targets.empty else False, "detail": ""},
        {"check": "no_leverage_weight_sum_lte_1", "pass": bool((weight_sums <= 1.000001).all()) if not weight_sums.empty else False, "detail": f"max={float(weight_sums.max()):.6f}" if not weight_sums.empty else ""},
        {"check": "positive_annual_excess_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "investment_gate_annual_excess_above_3pct_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) >= 0.03), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "drawdown_better_than_benchmark_10bps", "pass": bool(not row10.empty and float(row10["drawdown_improvement_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['drawdown_improvement_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "avg_cash_not_too_high_10bps", "pass": bool(not row10.empty and float(row10["avg_cash_weight"].iloc[0]) <= 0.35), "detail": f"{float(row10['avg_cash_weight'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "score_table_non_empty", "pass": bool(not score_table.empty), "detail": str(score_table.shape[0]) if not score_table.empty else ""},
    ]
    for name in ["MODEL_CHANGELOG.md", "WALK_FORWARD_REPORT.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def write_reports(output_dir: Path, selected: str, summary: pd.DataFrame, score_detail: pd.DataFrame, decisions: pd.DataFrame, self_check: pd.DataFrame | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_table = score_detail[["variant", "source", "benchmark_relative_score", "avg_annual_excess", "avg_drawdown_improvement", "avg_information_ratio", "mean_cash_weight"]].drop_duplicates().sort_values("benchmark_relative_score", ascending=False) if not score_detail.empty else pd.DataFrame()
    bucket_stats = (
        decisions.groupby(["variant", "timing_bucket"], dropna=False)
        .agg(periods=("signal_date", "count"), avg_target_gross=("target_gross", "mean"), avg_base_gross=("base_gross", "mean"), avg_score=("timing_score", "mean"))
        .reset_index()
        if not decisions.empty
        else pd.DataFrame()
    )
    report = [
        "# HIRSSM V3.2 Market Beta Timing Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Selected Variant",
        "",
        f"- `{selected}`",
        "",
        "## Design",
        "",
        "- V3.2 overlays market beta timing on V3.0 selected alpha sleeve.",
        "- No leverage and no negative cash are allowed.",
        "- Timing evidence uses broad-market trend, breadth, volatility/drawdown risk, and drawdown recovery.",
        "- The overlay changes total non-cash gross exposure but does not introduce a fixed 000985 core position.",
        "",
        "## Performance",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No summary.",
        "",
        "## Candidate Score Table",
        "",
        score_table.to_markdown(index=False) if not score_table.empty else "No score table.",
        "",
        "## Timing Bucket Diagnostics",
        "",
        bucket_stats.to_markdown(index=False) if not bucket_stats.empty else "No timing diagnostics.",
    ]
    (output_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    changelog = [
        "# HIRSSM V3.2 Model Changelog",
        "",
        "## Changed",
        "",
        "- Added independent market beta timing overlay.",
        "- Target gross exposure is selected by timing bucket and bounded by market state.",
        "- Reused V3.0 selected alpha sleeve; no new cross-sectional alpha expert was added.",
        "- Explicit no-leverage and non-negative cash checks were added.",
        "",
        "## Why",
        "",
        "V3.1 showed that fixed 000985 core exposure worsened return and drawdown. V3.2 tests conditional beta exposure instead.",
    ]
    (output_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    self_lines = [
        "# HIRSSM V3.2 Self Check Report",
        "",
        self_check.to_markdown(index=False) if self_check is not None and not self_check.empty else "Self check pending.",
    ]
    (output_dir / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--base-targets", default=str(BASE_TARGETS))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v30 = load_module("hirssm_v3_0_v3_1_benchmark_core", V30_PATH)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    base_targets = read_targets(Path(args.base_targets))
    timing = build_timing_panel(panel, BENCHMARK_ASSET)
    model.write_csv(timing, output_dir / "market_beta_timing_panel.csv")

    all_summaries = []
    all_scores = []
    all_decisions = []
    targets_by_variant = {}
    for variant, cfg in V32_VARIANTS.items():
        targets, decisions = overlay_beta_targets(base_targets, timing, cfg, variant)
        targets_by_variant[variant] = targets
        candidate_dir = output_dir / "candidates" / variant
        candidate_dir.mkdir(parents=True, exist_ok=True)
        summary, _ = run_costs(model, v30, panel, targets, candidate_dir, variant, variant)
        rel_summary = v30.add_relative_metrics(summary)
        rel_summary["candidate"] = variant
        all_summaries.append(rel_summary)
        all_scores.append(v30.benchmark_relative_score(summary, variant, cfg["description"]))
        decisions["description"] = cfg["description"]
        all_decisions.append(decisions)
        model.write_csv(targets, candidate_dir / "target_weights.csv")
        model.write_csv(decisions, candidate_dir / "beta_timing_decisions.csv")

    summary_all = pd.concat(all_summaries, ignore_index=True, sort=False)
    score_detail = pd.concat(all_scores, ignore_index=True, sort=False)
    score_table = v30.make_candidate_table(score_detail)
    decisions_all = pd.concat(all_decisions, ignore_index=True, sort=False)
    selected = str(score_table.iloc[0]["variant"])
    selected_targets = targets_by_variant[selected]
    selected_summary = summary_all[summary_all["candidate"].eq(selected)].drop(columns=["candidate"])
    selected_decisions = decisions_all[decisions_all["variant"].eq(selected)].copy()

    model.write_csv(summary_all, output_dir / "all_candidate_oos_performance.csv")
    model.write_csv(score_detail, output_dir / "benchmark_relative_score_detail.csv")
    model.write_csv(score_table, output_dir / "benchmark_relative_score_table.csv")
    model.write_csv(decisions_all, output_dir / "beta_timing_decisions_all.csv")
    model.write_csv(selected_targets, output_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected_summary, output_dir / "oos_performance.csv")
    model.write_csv(selected_decisions, output_dir / "beta_timing_decisions.csv")
    write_reports(output_dir, selected, selected_summary, score_detail, selected_decisions, None)
    self_check = make_self_check(model, wf, selected_targets, selected_summary, score_table, output_dir)
    model.write_csv(self_check, output_dir / "self_check_results.csv")
    write_reports(output_dir, selected, selected_summary, score_detail, selected_decisions, self_check)

    manifest = {
        "generated_at": now_text(),
        "output_dir": str(output_dir),
        "selected": selected,
        "self_check_pass": bool(self_check["pass"].all()) if not self_check.empty else False,
        "benchmark": BENCHMARK_ASSET,
        "costs": COSTS,
        "base_targets": str(args.base_targets),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
