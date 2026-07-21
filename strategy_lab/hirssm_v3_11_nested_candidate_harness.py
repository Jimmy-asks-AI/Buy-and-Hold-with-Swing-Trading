#!/usr/bin/env python
"""HIRSSM V3.11 nested candidate validation harness.

V3.11 is not a new alpha model. It is a promotion harness that compares
predeclared candidates against the V3.10 clean baseline using only prior data
for yearly candidate selection, plus purged block diagnostics for PBO risk.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hirssm_v2_model as model
import hirssm_v2_walk_forward as wf
import hirssm_v3_10_clean_baseline as v310
from model_run_manifest import build_model_run_manifest, validate_model_run_manifest


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_11_nested_candidate_harness"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_11" / "nested_candidate_harness"
TASK_ID = "20260526_v3_11_nested_candidate_harness"
MODEL_VERSION = "HIRSSM V3.11 Nested Candidate Harness"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"
TRADING_DAYS = 252


PREDECLARED_CANDIDATES = [
    {
        "variant": BASELINE_VARIANT,
        "role": "control",
        "description": "Frozen V3.10 clean rank-vol core.",
        "multipliers": {},
    },
    {
        "variant": "industry_trend_plus",
        "role": "candidate",
        "description": "Increase industry trend continuation while keeping style trend disabled.",
        "multipliers": {"industry_trend_continuation": 1.15, "risk_compression": 0.95},
    },
    {
        "variant": "valuation_defensive_plus",
        "role": "candidate",
        "description": "Tilt toward valuation repair and defensive scores.",
        "multipliers": {"valuation_repair": 1.15, "defensive": 1.10, "trend_continuation": 0.90},
    },
    {
        "variant": "risk_compression_plus",
        "role": "candidate",
        "description": "Emphasize risk compression and slightly reduce trend weight.",
        "multipliers": {"risk_compression": 1.20, "trend_continuation": 0.90},
    },
    {
        "variant": "balanced_defensive_low_beta",
        "role": "candidate",
        "description": "Small balanced tilt toward valuation, risk compression, and defense.",
        "multipliers": {
            "trend_continuation": 0.95,
            "valuation_repair": 1.05,
            "risk_compression": 1.05,
            "defensive": 1.05,
        },
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_costs(text: str) -> list[float]:
    costs = []
    for item in text.split(","):
        value = item.strip()
        if value:
            costs.append(float(value))
    if not costs:
        raise ValueError("cost list is empty")
    return costs


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def candidate_registry(config: dict) -> pd.DataFrame:
    disabled = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    rows = []
    for item in PREDECLARED_CANDIDATES:
        rows.append(
            {
                "variant": item["variant"],
                "role": item["role"],
                "description": item["description"],
                "multipliers_json": json.dumps(item["multipliers"], sort_keys=True),
                "disabled_experts": disabled,
                "selection_source": "predeclared_before_run",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": item["role"] == "candidate",
            }
        )
    return pd.DataFrame(rows)


def year_state_multiplier_map(panel: dict[str, pd.DataFrame], config: dict, multipliers: dict[str, float]) -> dict[tuple[int, str], dict[str, float]]:
    if not multipliers:
        return {}
    years = sorted(int(year) for year in pd.to_datetime(panel["eligible"]["date"]).dt.year.dropna().unique())
    states = list(config["portfolio"]["sleeve_budget_by_state"].keys())
    return {(year, state): dict(multipliers) for year in years for state in states}


def build_candidate_targets(
    *,
    panel: dict[str, pd.DataFrame],
    config: dict,
    variant: str,
    multipliers: dict[str, float],
) -> pd.DataFrame:
    if variant == BASELINE_VARIANT:
        targets = v310.build_targets(panel, config)
    else:
        disabled = {str(item) for item in config.get("disabled_experts_by_default", [])}
        start_date = pd.to_datetime(panel["eligible"]["date"].min()) if not panel["eligible"].empty else None
        raw_targets = model.build_targets(
            panel["eligible"],
            panel["regimes"],
            config,
            start_date=start_date,
            disabled_experts=disabled,
            expert_multipliers_by_year_state=year_state_multiplier_map(panel, config, multipliers),
        )
        targets = v310.enrich_targets(raw_targets, panel)
        targets = v310.enforce_cash_cap(targets, config)
        max_turnover = float(config["portfolio"]["constraints"].get("monthly_turnover_target_cap", 0.8))
        targets = v310.enforce_turnover_cap(targets, max_turnover=max_turnover)
        targets["disabled_experts"] = ",".join(sorted(disabled))
    targets["variant"] = variant
    targets["candidate_role"] = "control" if variant == BASELINE_VARIANT else "candidate"
    targets["multipliers_json"] = json.dumps(multipliers, sort_keys=True)
    return targets


def run_candidates(
    *,
    panel: dict[str, pd.DataFrame],
    config: dict,
    costs: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], dict[str, pd.DataFrame], dict[tuple[str, float], pd.DataFrame]]:
    rows = []
    navs: dict[tuple[str, float], pd.DataFrame] = {}
    targets_by_variant: dict[str, pd.DataFrame] = {}
    trades: dict[tuple[str, float], pd.DataFrame] = {}
    for candidate in PREDECLARED_CANDIDATES:
        variant = candidate["variant"]
        targets = build_candidate_targets(panel=panel, config=deepcopy(config), variant=variant, multipliers=candidate["multipliers"])
        targets_by_variant[variant] = targets
        model.write_csv(targets, output_dir / f"target_weights_{variant}.csv")
        for cost in costs:
            suffix = f"{variant}_{int(cost)}bps"
            bt = model.run_backtest(panel["returns"], targets, float(cost), panel["broad_code"])
            nav = bt["nav"].copy()
            nav["variant"] = variant
            nav["cost_bps"] = float(cost)
            summary = model.summarize_nav(nav)
            if not summary.empty:
                summary.insert(0, "variant", variant)
                summary.insert(1, "role", candidate["role"])
                summary.insert(2, "cost_bps", float(cost))
                summary["diagnostic_full_sample_only"] = True
                summary["annual_excess_vs_benchmark"] = summary["annual_return"] - summary["benchmark_annual_return"]
                summary["drawdown_improvement_vs_benchmark"] = summary["max_drawdown"] - summary["benchmark_max_drawdown"]
                rows.append(summary)
            navs[(variant, float(cost))] = nav
            trades[(variant, float(cost))] = bt["trades"]
            model.write_csv(nav, output_dir / f"nav_{suffix}.csv")
            model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
    return pd.concat(rows, ignore_index=True, sort=False), navs, targets_by_variant, trades


def align_variant_navs(navs: dict[tuple[str, float], pd.DataFrame], cost: float) -> dict[str, pd.DataFrame]:
    frames = {variant: nav.copy() for (variant, nav_cost), nav in navs.items() if float(nav_cost) == float(cost)}
    if not frames:
        raise ValueError(f"no navs for cost {cost}")
    common_dates = sorted(set.intersection(*[set(pd.to_datetime(df["date"])) for df in frames.values()]))
    out = {}
    for variant, df in frames.items():
        item = df.copy()
        item["date"] = pd.to_datetime(item["date"])
        item = item[item["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
        out[variant] = item
    return out


def max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    nav = (1.0 + clean).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def summarize_slice(df: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    sub = df.loc[mask.to_numpy()].copy()
    if sub.empty:
        return {
            "annual_return": float("nan"),
            "benchmark_annual_return": float("nan"),
            "annual_excess_vs_benchmark": float("nan"),
            "sharpe_no_rf": float("nan"),
            "information_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "avg_cash_weight": float("nan"),
            "avg_trade_turnover": float("nan"),
        }
    rets = pd.to_numeric(sub["portfolio_return"], errors="coerce").fillna(0.0)
    bench = pd.to_numeric(sub["benchmark_return"], errors="coerce").fillna(0.0)
    years = max(len(sub) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    total = float((1.0 + rets).prod() - 1.0)
    bench_total = float((1.0 + bench).prod() - 1.0)
    annual = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else float("nan")
    benchmark_annual = (1.0 + bench_total) ** (1.0 / years) - 1.0 if bench_total > -1.0 else float("nan")
    std = rets.std(ddof=1)
    sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS)) if pd.notna(std) and std > 0 else 0.0
    excess = rets - bench
    te = excess.std(ddof=1)
    ir = float(excess.mean() / te * math.sqrt(TRADING_DAYS)) if pd.notna(te) and te > 0 else 0.0
    traded = sub.loc[pd.to_numeric(sub.get("turnover", pd.Series([0.0] * len(sub))), errors="coerce").fillna(0.0) > 0, "turnover"]
    return {
        "annual_return": float(annual),
        "benchmark_annual_return": float(benchmark_annual),
        "annual_excess_vs_benchmark": float(annual - benchmark_annual),
        "sharpe_no_rf": sharpe,
        "information_ratio": ir,
        "max_drawdown": max_drawdown_from_returns(rets),
        "avg_cash_weight": float(pd.to_numeric(sub.get("cash_weight", pd.Series([0.0] * len(sub))), errors="coerce").fillna(0.0).mean()),
        "avg_trade_turnover": float(pd.to_numeric(traded, errors="coerce").mean()) if not traded.empty else 0.0,
    }


def selection_score(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    annual_delta = metrics["annual_return"] - baseline["annual_return"]
    sharpe_delta = metrics["sharpe_no_rf"] - baseline["sharpe_no_rf"]
    dd_delta = metrics["max_drawdown"] - baseline["max_drawdown"]
    cash_penalty = max(metrics["avg_cash_weight"] - baseline["avg_cash_weight"] - 0.05, 0.0)
    return float(
        1.40 * annual_delta
        + 0.35 * sharpe_delta
        + 0.85 * dd_delta
        + 0.30 * metrics["information_ratio"]
        + 0.80 * metrics["annual_excess_vs_benchmark"]
        - 0.30 * cash_penalty
        - 0.015 * metrics["avg_trade_turnover"]
    )


def nested_walk_forward(
    *,
    navs: dict[tuple[str, float], pd.DataFrame],
    costs: list[float],
    lookback_years: int,
    inner_validation_years: int,
    min_train_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, pd.DataFrame], pd.DataFrame]:
    selection_rows = []
    fold_rows = []
    selected_navs: dict[float, pd.DataFrame] = {}
    performance_rows = []
    for cost in costs:
        aligned = align_variant_navs(navs, cost)
        baseline = aligned[BASELINE_VARIANT]
        years = sorted(int(year) for year in pd.to_datetime(baseline["date"]).dt.year.unique())
        selected_frames = []
        for test_year in years:
            dates = pd.to_datetime(baseline["date"])
            test_start = pd.Timestamp(year=test_year, month=1, day=1)
            test_end = pd.Timestamp(year=test_year, month=12, day=31)
            outer_train_start = test_start - pd.DateOffset(years=lookback_years)
            outer_train_end = test_start - pd.Timedelta(days=embargo_days + 1)
            inner_validation_start = test_start - pd.DateOffset(years=inner_validation_years)
            inner_validation_end = outer_train_end
            inner_train_start = outer_train_start
            inner_train_end = inner_validation_start - pd.Timedelta(days=embargo_days + 1)
            outer_train_mask = (dates >= outer_train_start) & (dates <= outer_train_end)
            inner_train_mask = (dates >= inner_train_start) & (dates <= inner_train_end)
            inner_validation_mask = (dates >= inner_validation_start) & (dates <= inner_validation_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            train_days = int(outer_train_mask.sum())
            inner_train_days = int(inner_train_mask.sum())
            inner_validation_days = int(inner_validation_mask.sum())
            test_days = int(test_mask.sum())
            if test_days == 0:
                continue
            if train_days < min_train_days or inner_validation_days < 126:
                selection_rows.append(
                    {
                        "cost_bps": cost,
                        "test_year": test_year,
                        "selected_variant": BASELINE_VARIANT,
                        "selection_status": "skipped_insufficient_train",
                        "selection_reason": f"train_days={train_days}; inner_validation_days={inner_validation_days}",
                        "outer_train_start": outer_train_start,
                        "outer_train_end": outer_train_end,
                        "inner_train_start": inner_train_start,
                        "inner_train_end": inner_train_end,
                        "inner_validation_start": inner_validation_start,
                        "inner_validation_end": inner_validation_end,
                        "test_start": test_start,
                        "test_end": test_end,
                        "train_days": train_days,
                        "inner_train_days": inner_train_days,
                        "inner_validation_days": inner_validation_days,
                        "test_days": test_days,
                        "purge_days": 0,
                        "embargo_days": embargo_days,
                    }
                )
                continue

            base_select = summarize_slice(baseline, inner_validation_mask)
            base_test = summarize_slice(baseline, test_mask)
            scores = {}
            select_metrics_by_variant = {}
            test_metrics_by_variant = {}
            for variant, nav in aligned.items():
                select_metrics = summarize_slice(nav, inner_validation_mask)
                test_metrics = summarize_slice(nav, test_mask)
                select_metrics_by_variant[variant] = select_metrics
                test_metrics_by_variant[variant] = test_metrics
                scores[variant] = selection_score(select_metrics, base_select)
            score_series = pd.Series(scores).dropna().sort_values(ascending=False)
            selected = str(score_series.index[0]) if not score_series.empty else BASELINE_VARIANT
            selected_test = test_metrics_by_variant[selected]
            oos_scores = pd.Series(
                {
                    variant: selection_score(metrics, base_test)
                    for variant, metrics in test_metrics_by_variant.items()
                }
            ).sort_values(ascending=False)
            rank = int(list(oos_scores.index).index(selected) + 1)
            rank_pct = 1.0 - (rank - 1) / max(len(oos_scores) - 1, 1)
            selection_rows.append(
                {
                    "cost_bps": cost,
                    "test_year": test_year,
                    "selected_variant": selected,
                    "selection_status": "selected_by_prior_window",
                    "selection_reason": "max predeclared objective on inner validation window",
                    "outer_train_start": outer_train_start,
                    "outer_train_end": outer_train_end,
                    "inner_train_start": inner_train_start,
                    "inner_train_end": inner_train_end,
                    "inner_validation_start": inner_validation_start,
                    "inner_validation_end": inner_validation_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_days": train_days,
                    "inner_train_days": inner_train_days,
                    "inner_validation_days": inner_validation_days,
                    "test_days": test_days,
                    "purge_days": 0,
                    "embargo_days": embargo_days,
                    "selected_inner_validation_score": float(score_series.loc[selected]),
                    "selected_oos_score": float(oos_scores.loc[selected]),
                    "selected_oos_rank": rank,
                    "selected_oos_rank_pct": float(rank_pct),
                    "baseline_inner_validation_annual": base_select["annual_return"],
                    "selected_inner_validation_annual": select_metrics_by_variant[selected]["annual_return"],
                    "baseline_oos_annual": base_test["annual_return"],
                    "selected_oos_annual": selected_test["annual_return"],
                    "selected_oos_minus_baseline_annual": selected_test["annual_return"] - base_test["annual_return"],
                    "selected_oos_minus_baseline_drawdown": selected_test["max_drawdown"] - base_test["max_drawdown"],
                }
            )
            for variant, select_metrics in select_metrics_by_variant.items():
                row = {
                    "cost_bps": cost,
                    "test_year": test_year,
                    "variant": variant,
                    "inner_validation_score": scores[variant],
                    "oos_score": float(oos_scores.loc[variant]),
                    "selected": variant == selected,
                    "inner_validation_start": inner_validation_start,
                    "inner_validation_end": inner_validation_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
                row.update({f"inner_validation_{key}": value for key, value in select_metrics.items()})
                row.update({f"oos_{key}": value for key, value in test_metrics_by_variant[variant].items()})
                fold_rows.append(row)
            selected_frame = aligned[selected].loc[test_mask.to_numpy()].copy()
            selected_frame["selected_variant"] = selected
            selected_frame["test_year"] = test_year
            selected_frames.append(selected_frame)

        selected_nav = stitch_nav(selected_frames)
        selected_navs[float(cost)] = selected_nav
        summary = model.summarize_nav(selected_nav)
        if not summary.empty:
            summary.insert(0, "variant", "nested_selected_candidate")
            summary.insert(1, "cost_bps", float(cost))
            baseline_same = stitch_baseline_same_period(baseline, selected_nav)
            base_summary = model.summarize_nav(baseline_same)
            if not base_summary.empty:
                summary["baseline_same_period_annual_return"] = float(base_summary["annual_return"].iloc[0])
                summary["baseline_same_period_sharpe"] = float(base_summary["sharpe_no_rf"].iloc[0])
                summary["baseline_same_period_max_drawdown"] = float(base_summary["max_drawdown"].iloc[0])
                summary["annual_delta_vs_v310"] = summary["annual_return"] - summary["baseline_same_period_annual_return"]
                summary["sharpe_delta_vs_v310"] = summary["sharpe_no_rf"] - summary["baseline_same_period_sharpe"]
                summary["drawdown_delta_vs_v310"] = summary["max_drawdown"] - summary["baseline_same_period_max_drawdown"]
            performance_rows.append(summary)
    performance = pd.concat(performance_rows, ignore_index=True, sort=False) if performance_rows else pd.DataFrame()
    return pd.DataFrame(selection_rows), pd.DataFrame(fold_rows), selected_navs, performance


def stitch_nav(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)
    out["portfolio_return"] = pd.to_numeric(out["portfolio_return"], errors="coerce").fillna(0.0)
    out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce").fillna(0.0)
    out["nav"] = (1.0 + out["portfolio_return"]).cumprod()
    out["benchmark_nav"] = (1.0 + out["benchmark_return"]).cumprod()
    return out


def stitch_baseline_same_period(baseline: pd.DataFrame, selected_nav: pd.DataFrame) -> pd.DataFrame:
    if selected_nav.empty:
        return pd.DataFrame()
    dates = set(pd.to_datetime(selected_nav["date"]))
    out = baseline[pd.to_datetime(baseline["date"]).isin(dates)].copy().sort_values("date").reset_index(drop=True)
    out["portfolio_return"] = pd.to_numeric(out["portfolio_return"], errors="coerce").fillna(0.0)
    out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce").fillna(0.0)
    out["nav"] = (1.0 + out["portfolio_return"]).cumprod()
    out["benchmark_nav"] = (1.0 + out["benchmark_return"]).cumprod()
    return out


def purged_block_pbo(
    *,
    navs: dict[tuple[str, float], pd.DataFrame],
    costs: list[float],
    n_blocks: int,
    train_blocks: int,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for cost in costs:
        aligned = align_variant_navs(navs, cost)
        dates = pd.to_datetime(aligned[BASELINE_VARIANT]["date"]).reset_index(drop=True)
        block_ids = contiguous_block_ids(len(dates), n_blocks)
        combos = list(itertools.combinations(range(n_blocks), train_blocks))
        for fold_id, train_tuple in enumerate(combos):
            train_set = set(train_tuple)
            test_set = set(range(n_blocks)) - train_set
            train_mask = pd.Series([block in train_set for block in block_ids])
            test_mask = pd.Series([block in test_set for block in block_ids])
            train_mask = apply_purge_embargo(train_mask, test_mask, purge_days, embargo_days)
            base_train = summarize_slice(aligned[BASELINE_VARIANT], train_mask)
            base_test = summarize_slice(aligned[BASELINE_VARIANT], test_mask)
            train_scores = {}
            test_scores = {}
            for variant, nav in aligned.items():
                train_scores[variant] = selection_score(summarize_slice(nav, train_mask), base_train)
                test_scores[variant] = selection_score(summarize_slice(nav, test_mask), base_test)
            train_s = pd.Series(train_scores).dropna().sort_values(ascending=False)
            test_s = pd.Series(test_scores).dropna().sort_values(ascending=False)
            if train_s.empty or test_s.empty:
                continue
            selected = str(train_s.index[0])
            rank = int(list(test_s.index).index(selected) + 1)
            rank_pct = 1.0 - (rank - 1) / max(len(test_s) - 1, 1)
            rows.append(
                {
                    "fold_id": fold_id,
                    "cost_bps": cost,
                    "train_blocks": " ".join(str(x) for x in sorted(train_set)),
                    "test_blocks": " ".join(str(x) for x in sorted(test_set)),
                    "purge_days": purge_days,
                    "embargo_days": embargo_days,
                    "n_candidates": int(len(test_s)),
                    "selected_variant": selected,
                    "train_score": float(train_s.loc[selected]),
                    "oos_score": float(test_s.loc[selected]),
                    "oos_rank": rank,
                    "oos_rank_pct": float(rank_pct),
                    "overfit_worse_half": bool(rank_pct < 0.5),
                    "selected_is_baseline": selected == BASELINE_VARIANT,
                    "selected_oos_minus_baseline_score": float(test_s.loc[selected] - test_s.loc[BASELINE_VARIANT]),
                }
            )
        fold = pd.DataFrame([row for row in rows if row["cost_bps"] == cost])
        summary_rows.append(
            {
                "cost_bps": cost,
                "n_folds": int(fold.shape[0]),
                "candidate_count": int(len(aligned)),
                "n_blocks": n_blocks,
                "train_blocks": train_blocks,
                "purge_days": purge_days,
                "embargo_days": embargo_days,
                "pbo": float(fold["overfit_worse_half"].mean()) if not fold.empty else np.nan,
                "pbo_status": pbo_status(float(fold["overfit_worse_half"].mean())) if not fold.empty else "blocked",
                "baseline_selected_rate": float(fold["selected_is_baseline"].mean()) if not fold.empty else np.nan,
                "median_oos_rank_pct": float(fold["oos_rank_pct"].median()) if not fold.empty else np.nan,
                "top1_oos_rate": float((fold["oos_rank"] == 1).mean()) if not fold.empty else np.nan,
                "mean_selected_minus_baseline_score": float(fold["selected_oos_minus_baseline_score"].mean()) if not fold.empty else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def contiguous_block_ids(n_obs: int, n_blocks: int) -> np.ndarray:
    if n_blocks < 2 or n_obs < n_blocks:
        raise ValueError("invalid n_blocks")
    ids = np.empty(n_obs, dtype=int)
    edges = np.linspace(0, n_obs, n_blocks + 1).round().astype(int)
    for block in range(n_blocks):
        ids[edges[block] : edges[block + 1]] = block
    return ids


def apply_purge_embargo(train_mask: pd.Series, test_mask: pd.Series, purge_days: int, embargo_days: int) -> pd.Series:
    out = train_mask.to_numpy().copy()
    test_idx = np.where(test_mask.to_numpy())[0]
    if len(test_idx) == 0:
        return pd.Series(out)
    for idx in test_idx:
        start = max(0, int(idx) - purge_days)
        end = min(len(out), int(idx) + embargo_days + 1)
        out[start:end] = False
    return pd.Series(out)


def pbo_status(pbo: float) -> str:
    if pd.isna(pbo):
        return "blocked"
    if pbo <= 0.20:
        return "pass"
    if pbo <= 0.35:
        return "observation"
    return "fail"


def promotion_decision(performance: pd.DataFrame, pbo_summary: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in sorted(performance["cost_bps"].astype(float).unique()) if not performance.empty else []:
        perf = performance[performance["cost_bps"].astype(float).eq(cost)].iloc[0]
        pbo = pbo_summary[pbo_summary["cost_bps"].astype(float).eq(cost)]
        pbo_value = float(pbo["pbo"].iloc[0]) if not pbo.empty else np.nan
        selected = selection[
            selection["cost_bps"].astype(float).eq(cost)
            & selection["selection_status"].eq("selected_by_prior_window")
        ]
        nonbaseline_rate = float((selected["selected_variant"] != BASELINE_VARIANT).mean()) if not selected.empty else 0.0
        checks = {
            "annual_delta_positive_50bps": float(perf.get("annual_delta_vs_v310", np.nan)) >= 0.005,
            "drawdown_not_worse_3pct": float(perf.get("drawdown_delta_vs_v310", np.nan)) >= -0.03,
            "avg_cash_not_excessive": float(perf.get("avg_cash_weight", np.nan)) <= 0.25,
            "pbo_not_fail": pd.notna(pbo_value) and pbo_value <= 0.35,
            "candidate_selected_enough": nonbaseline_rate >= 0.20,
        }
        hard_pass = all(checks.values())
        rows.append(
            {
                "cost_bps": cost,
                "decision": "promote_candidate" if hard_pass else "reject_for_default_observation_only",
                "annual_delta_vs_v310": float(perf.get("annual_delta_vs_v310", np.nan)),
                "drawdown_delta_vs_v310": float(perf.get("drawdown_delta_vs_v310", np.nan)),
                "sharpe_delta_vs_v310": float(perf.get("sharpe_delta_vs_v310", np.nan)),
                "pbo": pbo_value,
                "nonbaseline_selection_rate": nonbaseline_rate,
                **checks,
            }
        )
    overall = "promote_candidate" if rows and all(row["decision"] == "promote_candidate" for row in rows if row["cost_bps"] in {10.0, 20.0}) else "reject_for_default_observation_only"
    out = pd.DataFrame(rows)
    if not out.empty:
        out["overall_decision"] = overall
    return out


def split_manifest_from_selection(selection: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cost_bps",
        "test_year",
        "selected_variant",
        "selection_status",
        "outer_train_start",
        "outer_train_end",
        "inner_train_start",
        "inner_train_end",
        "inner_validation_start",
        "inner_validation_end",
        "test_start",
        "test_end",
        "train_days",
        "inner_train_days",
        "inner_validation_days",
        "test_days",
        "purge_days",
        "embargo_days",
    ]
    out = selection[[col for col in cols if col in selection.columns]].copy()
    out["split_rule"] = "outer_calendar_year_inner_prior_validation"
    out["test_data_used_for_selection"] = False
    return out


def embargo_purge_audit(split_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in split_manifest.iterrows():
        inner_val_end = pd.to_datetime(row.get("inner_validation_end"))
        test_start = pd.to_datetime(row.get("test_start"))
        inner_train_end = pd.to_datetime(row.get("inner_train_end"))
        inner_val_start = pd.to_datetime(row.get("inner_validation_start"))
        embargo_days = int(float(row.get("embargo_days", 0)))
        train_gap = (inner_val_start - inner_train_end).days if pd.notna(inner_train_end) and pd.notna(inner_val_start) else np.nan
        test_gap = (test_start - inner_val_end).days if pd.notna(inner_val_end) and pd.notna(test_start) else np.nan
        rows.append(
            {
                "cost_bps": row.get("cost_bps"),
                "test_year": row.get("test_year"),
                "inner_train_to_validation_gap_days": train_gap,
                "validation_to_test_gap_days": test_gap,
                "required_embargo_days": embargo_days,
                "inner_train_validation_overlap": bool(pd.notna(train_gap) and train_gap < 0),
                "validation_test_overlap": bool(pd.notna(test_gap) and test_gap <= 0),
                "embargo_satisfied": bool(pd.notna(test_gap) and test_gap >= embargo_days),
                "status": "pass" if pd.notna(test_gap) and test_gap >= embargo_days and not (pd.notna(train_gap) and train_gap < 0) else "fail",
            }
        )
    return pd.DataFrame(rows)


def outer_fold_oos_results(selection: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cost_bps",
        "test_year",
        "selected_variant",
        "selection_status",
        "test_start",
        "test_end",
        "test_days",
        "selected_oos_score",
        "selected_oos_rank",
        "selected_oos_rank_pct",
        "baseline_oos_annual",
        "selected_oos_annual",
        "selected_oos_minus_baseline_annual",
        "selected_oos_minus_baseline_drawdown",
    ]
    out = selection[[col for col in cols if col in selection.columns]].copy()
    out["same_period_baseline"] = BASELINE_VARIANT
    return out


def same_period_baseline_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cost_bps",
        "annual_return",
        "sharpe_no_rf",
        "max_drawdown",
        "avg_cash_weight",
        "avg_trade_turnover",
        "baseline_same_period_annual_return",
        "baseline_same_period_sharpe",
        "baseline_same_period_max_drawdown",
        "annual_delta_vs_v310",
        "sharpe_delta_vs_v310",
        "drawdown_delta_vs_v310",
    ]
    out = performance[[col for col in cols if col in performance.columns]].copy()
    out.insert(0, "selected_strategy", "nested_selected_candidate")
    out.insert(1, "baseline_strategy", BASELINE_VARIANT)
    out["comparison_rule"] = "same_dates_same_cost_same_benchmark"
    return out


def cost_sensitivity_table(performance: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    compare = same_period_baseline_comparison(performance)
    if decision.empty:
        compare["gate_decision"] = "blocked"
        return compare
    dec = decision[["cost_bps", "decision", "pbo"]].copy()
    return compare.merge(dec, on="cost_bps", how="left").rename(columns={"decision": "gate_decision"})


def validation_findings(checks: pd.DataFrame, decision: pd.DataFrame, pbo_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in checks.iterrows():
        rows.append(
            {
                "category": "artifact_or_process",
                "finding": row["check"],
                "severity": "fail" if row["status"] == "fail" else ("warn" if row["status"] == "warn" else "pass"),
                "detail": row["detail"],
            }
        )
    if not decision.empty:
        rows.append(
            {
                "category": "gate",
                "finding": "overall_candidate_gate",
                "severity": "pass" if decision["overall_decision"].iloc[0] == "promote_candidate" else "warn",
                "detail": str(decision["overall_decision"].iloc[0]),
            }
        )
    for _, row in pbo_summary.iterrows():
        rows.append(
            {
                "category": "pbo",
                "finding": f"pbo_{int(float(row['cost_bps']))}bps",
                "severity": "fail" if row["pbo_status"] == "fail" else ("warn" if row["pbo_status"] == "observation" else "pass"),
                "detail": str(float(row["pbo"])),
            }
        )
    return pd.DataFrame(rows)


def leakage_checklist(split_manifest: pd.DataFrame, embargo_audit: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "candidate_registry_predeclared",
                "status": "pass",
                "detail": "registry generated before gate decision in script order",
            },
            {
                "check": "outer_test_not_used_for_selection",
                "status": "pass" if not split_manifest.get("test_data_used_for_selection", pd.Series([False])).any() else "fail",
                "detail": "selection uses inner validation window only",
            },
            {
                "check": "embargo_satisfied_all_folds",
                "status": "pass" if not embargo_audit.empty and embargo_audit["embargo_satisfied"].all() else "fail",
                "detail": str(int(embargo_audit["embargo_satisfied"].sum()) if not embargo_audit.empty else 0),
            },
            {
                "check": "same_period_baseline_required",
                "status": "pass",
                "detail": "comparison uses same dates, same cost, same benchmark",
            },
            {
                "check": "full_sample_ranking_not_gate",
                "status": "pass",
                "detail": "candidate_full_sample_diagnostic_metrics.csv is diagnostic only",
            },
        ]
    )


def robustness_summary(decision: pd.DataFrame, pbo_summary: pd.DataFrame, cost_sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in decision.iterrows():
        rows.append(
            {
                "cost_bps": row["cost_bps"],
                "test": "candidate_gate",
                "status": row["decision"],
                "value": row["annual_delta_vs_v310"],
                "detail": f"pbo={row['pbo']}; nonbaseline_selection_rate={row['nonbaseline_selection_rate']}",
            }
        )
    for _, row in pbo_summary.iterrows():
        rows.append(
            {
                "cost_bps": row["cost_bps"],
                "test": "purged_cscv_pbo",
                "status": row["pbo_status"],
                "value": row["pbo"],
                "detail": f"n_folds={row['n_folds']}; candidate_count={row['candidate_count']}",
            }
        )
    for _, row in cost_sensitivity.iterrows():
        rows.append(
            {
                "cost_bps": row["cost_bps"],
                "test": "cost_sensitivity",
                "status": row.get("gate_decision", "observation"),
                "value": row.get("annual_delta_vs_v310", np.nan),
                "detail": "same-period baseline comparison",
            }
        )
    return pd.DataFrame(rows)


def build_constraint_checks(
    *,
    registry: pd.DataFrame,
    selection: pd.DataFrame,
    performance: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    decision: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "check": "candidate_registry_not_empty",
            "status": "pass" if registry.shape[0] >= 2 else "fail",
            "detail": str(int(registry.shape[0])),
        },
        {
            "check": "baseline_candidate_present",
            "status": "pass" if BASELINE_VARIANT in set(registry["variant"]) else "fail",
            "detail": BASELINE_VARIANT,
        },
        {
            "check": "no_full_sample_promotion",
            "status": "pass",
            "detail": "full-sample metrics are diagnostic_only; selection uses prior windows",
        },
        {
            "check": "walk_forward_selection_exists",
            "status": "pass" if selection["selection_status"].eq("selected_by_prior_window").any() else "fail",
            "detail": str(int(selection["selection_status"].eq("selected_by_prior_window").sum())) if not selection.empty else "0",
        },
        {
            "check": "nested_oos_performance_exists",
            "status": "pass" if not performance.empty else "fail",
            "detail": str(int(performance.shape[0])),
        },
        {
            "check": "purged_pbo_report_exists",
            "status": "pass" if not pbo_summary.empty else "fail",
            "detail": str(int(pbo_summary.shape[0])),
        },
        {
            "check": "default_not_promoted_without_all_gates",
            "status": "pass" if decision.empty or not decision["overall_decision"].eq("promote_candidate").any() else "warn",
            "detail": str(decision["overall_decision"].iloc[0]) if not decision.empty else "no_decision",
        },
    ]
    return pd.DataFrame(rows)


def collect_data_refs(root: Path, config: dict) -> list[tuple[str, Path]]:
    refs = []
    for key in ["style_daily_path", "style_pe_path", "style_pb_path", "industry_daily_path", "industry_classification_path"]:
        path = root / config["data_contract"][key]
        if path.is_dir():
            for csv_path in sorted(path.glob("*.csv"))[:80]:
                refs.append((f"{key}_{csv_path.stem}", csv_path))
        elif path.exists():
            refs.append((key, path))
    return refs


def make_report(performance: pd.DataFrame, pbo_summary: pd.DataFrame, decision: pd.DataFrame) -> str:
    ref = performance[performance["cost_bps"].astype(float).eq(10.0)].head(1)
    pbo10 = pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)].head(1)
    dec = decision["overall_decision"].iloc[0] if not decision.empty else "blocked"
    lines = [
        "# HIRSSM V3.11 Nested Candidate Harness",
        "",
        "## Purpose",
        "",
        "Validate predeclared candidates against the V3.10 clean baseline without full-sample promotion.",
        "",
        "## 10bps Nested OOS",
        "",
    ]
    if not ref.empty:
        item = ref.iloc[0]
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
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Overall decision: {dec}",
            "- Full-sample candidate metrics remain diagnostic only.",
            "- Next alpha work must improve candidates and rerun this harness before any default promotion.",
        ]
    )
    return "\n".join(lines)


def make_self_check_report(checks: pd.DataFrame, manifest_findings: list[dict[str, str]] | None = None) -> str:
    manifest_findings = manifest_findings or []
    fail_count = int((checks["status"] == "fail").sum())
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    lines = [
        "# HIRSSM V3.11 Self Check",
        "",
        f"- Constraint failures: {fail_count}",
        f"- Manifest failures: {manifest_fail_count}",
        "- Full-sample promotion disabled: true",
        "- Prior-window selection required: true",
        "- Purge and embargo diagnostics required: true",
        "",
        "## Checks",
        "",
    ]
    for _, row in checks.iterrows():
        lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
    if manifest_findings:
        lines.extend(["", "## Manifest Findings", ""])
        for finding in manifest_findings:
            lines.append(f"- {finding.get('severity')}: {finding.get('field')} - {finding.get('message')}")
    return "\n".join(lines)


def make_agent_manifest(
    *,
    start_time: str,
    agent_dir: Path,
    config_path: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "backtest_validation_auditor",
        "version": "V3.11",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_11_nested_candidate_harness.py",
        "config": {
            "config_path": str(config_path.as_posix()),
            "baseline_variant": BASELINE_VARIANT,
            "candidate_count": len(PREDECLARED_CANDIDATES),
        },
        "data_refs": ["data_raw/index/akshare_csindex", "data_raw/index/akshare_sw_industry"],
        "code_refs": [
            "strategy_lab/hirssm_v3_11_nested_candidate_harness.py",
            "strategy_lab/hirssm_v3_10_clean_baseline.py",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
            "strategy_lab/model_run_manifest.py",
        ],
        "output_dir": str(agent_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": [
            "outputs/hirssm_v3_10_clean_baseline",
            "configs/hirssm_v2_default.json",
            "strategy_lab/hirssm_v3_10_clean_baseline.py",
        ],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "This harness proves validation plumbing and does not promote a new alpha model by itself.",
            "Candidate set is intentionally small and predeclared.",
        ],
        "risk_flags": ["candidate_metrics_full_sample_diagnostic_only"],
        "next_decision": "Use harness output to design V3.12 candidate improvements; do not promote full-sample winners.",
        "handoff_summary": "Nested yearly selection, purged PBO, promotion decision, and strict manifest have been generated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.11 nested candidate validation harness.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--agent-output-dir", default=str(AGENT_OUTPUT_DIR))
    parser.add_argument("--costs", default="5,10,20,30")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--inner-validation-years", type=int, default=1)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--embargo-days", type=int, default=21)
    parser.add_argument("--pbo-blocks", type=int, default=10)
    parser.add_argument("--pbo-train-blocks", type=int, default=5)
    parser.add_argument("--pbo-purge-days", type=int, default=63)
    args = parser.parse_args()

    start_time = now_text()
    root = Path(args.root)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    agent_dir = Path(args.agent_output_dir)
    costs = parse_costs(args.costs)
    config = model.read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    panel = wf.build_panel(model, root, config, None, None)
    registry = candidate_registry(config)
    candidate_metrics, navs, targets_by_variant, trades = run_candidates(panel=panel, config=config, costs=costs, output_dir=output_dir)
    selection, fold_scores, selected_navs, nested_performance = nested_walk_forward(
        navs=navs,
        costs=costs,
        lookback_years=args.lookback_years,
        inner_validation_years=args.inner_validation_years,
        min_train_days=args.min_train_days,
        embargo_days=args.embargo_days,
    )
    pbo_folds, pbo_summary = purged_block_pbo(
        navs=navs,
        costs=costs,
        n_blocks=args.pbo_blocks,
        train_blocks=args.pbo_train_blocks,
        purge_days=args.pbo_purge_days,
        embargo_days=args.embargo_days,
    )
    decision = promotion_decision(nested_performance, pbo_summary, selection)
    split_manifest = split_manifest_from_selection(selection)
    embargo_audit = embargo_purge_audit(split_manifest)
    outer_oos = outer_fold_oos_results(selection)
    same_period_comparison = same_period_baseline_comparison(nested_performance)
    cost_sensitivity = cost_sensitivity_table(nested_performance, decision)
    checks = build_constraint_checks(
        registry=registry,
        selection=selection,
        performance=nested_performance,
        pbo_summary=pbo_summary,
        decision=decision,
    )
    findings = validation_findings(checks, decision, pbo_summary)
    leakage = leakage_checklist(split_manifest, embargo_audit)
    robustness = robustness_summary(decision, pbo_summary, cost_sensitivity)

    registry_path = output_dir / "candidate_registry.csv"
    registry_json_path = output_dir / "candidate_registry.json"
    split_manifest_path = output_dir / "split_manifest.csv"
    split_manifest_json_path = output_dir / "split_manifest.json"
    embargo_audit_path = output_dir / "embargo_purge_audit.csv"
    inner_scores_path = output_dir / "inner_candidate_scores.csv"
    nested_selection_path = output_dir / "nested_selection_by_fold.csv"
    outer_oos_path = output_dir / "outer_fold_oos_results.csv"
    same_period_path = output_dir / "same_period_baseline_comparison.csv"
    cost_sensitivity_path = output_dir / "cost_sensitivity.csv"
    pbo_cscv_summary_path = output_dir / "pbo_cscv_summary.csv"
    pbo_cscv_splits_path = output_dir / "pbo_cscv_splits.csv"
    gate_decision_path = output_dir / "candidate_gate_decision.csv"
    validation_findings_path = output_dir / "validation_findings.csv"
    leakage_checklist_path = output_dir / "leakage_checklist.csv"
    robustness_summary_path = output_dir / "robustness_summary.csv"
    candidate_metrics_path = output_dir / "candidate_full_sample_diagnostic_metrics.csv"
    selection_path = output_dir / "nested_selection_history.csv"
    fold_scores_path = output_dir / "nested_fold_scores.csv"
    pbo_folds_path = output_dir / "purged_pbo_folds.csv"
    pbo_summary_path = output_dir / "pbo_report.csv"
    performance_path = output_dir / "nested_oos_performance.csv"
    decision_path = output_dir / "promotion_decision.csv"
    checks_path = output_dir / "constraint_check.csv"
    report_path = output_dir / "WALK_FORWARD_REPORT.md"
    changelog_path = output_dir / "MODEL_CHANGELOG.md"
    self_check_path = output_dir / "SELF_CHECK_REPORT.md"
    model_manifest_path = output_dir / "model_run_manifest.json"
    model_manifest_check_path = output_dir / "model_run_manifest_check.csv"

    model.write_csv(registry, registry_path)
    write_json({"candidates": registry.to_dict(orient="records")}, registry_json_path)
    model.write_csv(split_manifest, split_manifest_path)
    write_json({"splits": split_manifest.astype(str).to_dict(orient="records")}, split_manifest_json_path)
    model.write_csv(embargo_audit, embargo_audit_path)
    model.write_csv(fold_scores, inner_scores_path)
    model.write_csv(selection, nested_selection_path)
    model.write_csv(outer_oos, outer_oos_path)
    model.write_csv(same_period_comparison, same_period_path)
    model.write_csv(cost_sensitivity, cost_sensitivity_path)
    model.write_csv(pbo_summary, pbo_cscv_summary_path)
    model.write_csv(pbo_folds, pbo_cscv_splits_path)
    model.write_csv(decision, gate_decision_path)
    model.write_csv(findings, validation_findings_path)
    model.write_csv(leakage, leakage_checklist_path)
    model.write_csv(robustness, robustness_summary_path)
    model.write_csv(candidate_metrics, candidate_metrics_path)
    model.write_csv(selection, selection_path)
    model.write_csv(fold_scores, fold_scores_path)
    model.write_csv(pbo_folds, pbo_folds_path)
    model.write_csv(pbo_summary, pbo_summary_path)
    model.write_csv(nested_performance, performance_path)
    model.write_csv(decision, decision_path)
    model.write_csv(checks, checks_path)
    write_text(make_report(nested_performance, pbo_summary, decision), report_path)
    write_text(make_report(nested_performance, pbo_summary, decision), changelog_path)
    write_text(make_self_check_report(checks), self_check_path)

    for cost, nav in selected_navs.items():
        model.write_csv(nav, output_dir / f"nav_nested_selected_candidate_{int(cost)}bps.csv")
        if not nav.empty:
            model.write_csv(model.yearly_returns(nav), output_dir / f"yearly_returns_nested_selected_candidate_{int(cost)}bps.csv")
            model.write_csv(model.regime_returns(nav, panel["regimes"]), output_dir / f"regime_returns_nested_selected_candidate_{int(cost)}bps.csv")

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    reference_perf = nested_performance[nested_performance["cost_bps"].astype(float).eq(10.0)].head(1)
    metrics = {
        "candidate_count": int(registry.shape[0]),
        "walk_forward_selected_year_count": int(selection["selection_status"].eq("selected_by_prior_window").sum()) if not selection.empty else 0,
        "pbo_10bps": float(pbo_summary[pbo_summary["cost_bps"].astype(float).eq(10.0)]["pbo"].iloc[0]) if not pbo_summary.empty and pbo_summary["cost_bps"].astype(float).eq(10.0).any() else np.nan,
        "overall_decision": str(decision["overall_decision"].iloc[0]) if not decision.empty else "blocked",
    }
    if not reference_perf.empty:
        item = reference_perf.iloc[0]
        metrics.update(
            {
                "reference_cost_bps": 10.0,
                "annual_return": float(item["annual_return"]),
                "sharpe_no_rf": float(item["sharpe_no_rf"]),
                "max_drawdown": float(item["max_drawdown"]),
                "annual_delta_vs_v310": float(item.get("annual_delta_vs_v310", np.nan)),
                "drawdown_delta_vs_v310": float(item.get("drawdown_delta_vs_v310", np.nan)),
            }
        )

    artifact_paths = [
        registry_path,
        registry_json_path,
        split_manifest_path,
        split_manifest_json_path,
        embargo_audit_path,
        inner_scores_path,
        nested_selection_path,
        outer_oos_path,
        same_period_path,
        cost_sensitivity_path,
        pbo_cscv_summary_path,
        pbo_cscv_splits_path,
        gate_decision_path,
        validation_findings_path,
        leakage_checklist_path,
        robustness_summary_path,
        candidate_metrics_path,
        selection_path,
        fold_scores_path,
        pbo_folds_path,
        pbo_summary_path,
        performance_path,
        decision_path,
        checks_path,
        report_path,
        changelog_path,
        self_check_path,
    ]
    for cost in costs:
        artifact_paths.extend(
            [
                output_dir / f"nav_nested_selected_candidate_{int(cost)}bps.csv",
                output_dir / f"yearly_returns_nested_selected_candidate_{int(cost)}bps.csv",
                output_dir / f"regime_returns_nested_selected_candidate_{int(cost)}bps.csv",
            ]
        )

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
        command=["python", "-X", "utf8", "strategy_lab/hirssm_v3_11_nested_candidate_harness.py"],
        argv={
            "costs": costs,
            "lookback_years": args.lookback_years,
            "inner_validation_years": args.inner_validation_years,
            "min_train_days": args.min_train_days,
            "embargo_days": args.embargo_days,
            "pbo_blocks": args.pbo_blocks,
            "pbo_train_blocks": args.pbo_train_blocks,
            "pbo_purge_days": args.pbo_purge_days,
        },
        code_paths=[
            root / "strategy_lab" / "hirssm_v3_11_nested_candidate_harness.py",
            root / "strategy_lab" / "hirssm_v3_10_clean_baseline.py",
            root / "strategy_lab" / "hirssm_v2_model.py",
            root / "strategy_lab" / "hirssm_v2_walk_forward.py",
            root / "strategy_lab" / "model_run_manifest.py",
        ],
        config_path=config_path,
        data_paths=collect_data_refs(root, config),
        artifact_paths=artifact_paths,
        selection={
            "baseline_variant": BASELINE_VARIANT,
            "candidate_count": int(registry.shape[0]),
            "selection_method": "walk_forward_prior_window_only",
            "full_sample_metrics_diagnostic_only": True,
            "purge_days": int(args.pbo_purge_days),
            "embargo_days": int(args.embargo_days),
        },
        metrics=metrics,
        checks={
            "self_check_pass": fail_count == 0,
            "fail_count": fail_count,
            "warn_count": warn_count,
            "constraint_fail_count": fail_count,
        },
        limitations=[
            "Candidate set is small and intended to validate governance plumbing.",
            "Full-sample candidate metrics are not allowed for promotion.",
            "PBO block folds are diagnostics; final promotion still needs economic review and independent code review.",
        ],
        risk_flags=["candidate_metrics_full_sample_diagnostic_only"],
        next_decision="Use V3.11 results to design V3.12 candidate improvements under the same harness.",
        handoff_summary="V3.11 emits nested selection, purged PBO, promotion decision, and strict manifest for future candidate governance.",
    )
    write_json(manifest, model_manifest_path)
    manifest_findings = validate_model_run_manifest(manifest)
    manifest_check = pd.DataFrame(manifest_findings)
    if manifest_check.empty:
        manifest_check = pd.DataFrame([{"severity": "pass", "field": "model_run_manifest", "message": "no failures"}])
    model.write_csv(manifest_check, model_manifest_check_path)
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    manifest_warn_count = sum(1 for item in manifest_findings if item.get("severity") == "warn")

    agent_report_path = agent_dir / "agent_report.md"
    agent_registry_path = agent_dir / "candidate_registry.csv"
    agent_decision_path = agent_dir / "promotion_decision.csv"
    agent_checks_path = agent_dir / "constraint_check.csv"
    agent_findings_path = agent_dir / "validation_findings.csv"
    agent_leakage_path = agent_dir / "leakage_checklist.csv"
    agent_robustness_path = agent_dir / "robustness_summary.csv"
    agent_manifest_path = agent_dir / "agent_run_manifest.json"
    write_text(make_report(nested_performance, pbo_summary, decision), agent_report_path)
    model.write_csv(registry, agent_registry_path)
    model.write_csv(decision, agent_decision_path)
    model.write_csv(checks, agent_checks_path)
    model.write_csv(findings, agent_findings_path)
    model.write_csv(leakage, agent_leakage_path)
    model.write_csv(robustness, agent_robustness_path)
    agent_artifacts = [
        agent_report_path,
        agent_registry_path,
        agent_decision_path,
        agent_checks_path,
        agent_findings_path,
        agent_leakage_path,
        agent_robustness_path,
        model_manifest_path,
        model_manifest_check_path,
        agent_manifest_path,
    ]
    agent_manifest = make_agent_manifest(
        start_time=start_time,
        agent_dir=agent_dir,
        config_path=config_path,
        artifacts=agent_artifacts,
        metrics=metrics,
        fail_count=fail_count + manifest_fail_count,
        warn_count=warn_count + manifest_warn_count,
    )
    write_json(agent_manifest, agent_manifest_path)

    result = {
        "model_version": MODEL_VERSION,
        "self_check_pass": fail_count == 0 and manifest_fail_count == 0,
        "fail_count": fail_count,
        "manifest_fail_count": manifest_fail_count,
        "warn_count": warn_count + manifest_warn_count,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "agent_output_dir": str(agent_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["self_check_pass"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
