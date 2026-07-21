#!/usr/bin/env python
"""HIRSSM V3.8 risk-budget overlay.

V3.8 keeps V3.6 as the alpha/beta signal source and only changes portfolio
construction. It tests volatility budgeting, cluster concentration control and
drawdown-contribution guards, with the official V3.6 targets retained as an
exact control.
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
V35_PATH = ROOT / "strategy_lab" / "hirssm_v3_3_to_v3_5_alpha_factory.py"
V36_PATH = ROOT / "strategy_lab" / "hirssm_v3_6_component_attribution.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay"
V36_TARGETS = ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "walk_forward_target_weights.csv"
V32_PERF = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "oos_performance.csv"
V35_PERF = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "oos_performance.csv"
V36_PERF = ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "oos_performance.csv"
BENCHMARK_ASSET = "000985"
COSTS = [5.0, 10.0, 20.0, 30.0]
TRADING_DAYS = 252


V38_VARIANTS = {
    "v3_8_vol_budget_overlay": {
        "description": "Inverse-volatility risk budget while preserving V3.6 gross exposure.",
        "vol_power": 0.45,
        "min_mult": 0.65,
        "max_mult": 1.25,
        "preserve_gross": True,
        "min_weight_change": 0.020,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
    },
    "v3_8_correlation_cluster_guard": {
        "description": "Reduce concentration by broad style, growth beta, financial and cyclical clusters.",
        "vol_power": 0.20,
        "min_mult": 0.75,
        "max_mult": 1.18,
        "preserve_gross": False,
        "min_weight_change": 0.020,
        "max_style_weight": 0.48,
        "max_industry_weight": 0.18,
        "cluster_caps": {
            "broad_large_style": 0.38,
            "mid_small_style": 0.34,
            "dividend_defensive": 0.24,
            "growth_beta": 0.24,
            "financial_value": 0.28,
            "cyclical_value": 0.22,
            "defensive_consumption": 0.26,
            "other_industry": 0.24,
        },
        "min_gross_exposure": 0.782,
    },
    "v3_8_drawdown_contribution_guard": {
        "description": "Haircut assets with high recent drawdown contribution and redistribute to remaining sleeve.",
        "vol_power": 0.20,
        "min_mult": 0.70,
        "max_mult": 1.15,
        "drawdown_threshold": -0.16,
        "drawdown_full": -0.34,
        "max_drawdown_haircut": 0.40,
        "preserve_gross": True,
        "min_weight_change": 0.025,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
    },
    "v3_8_turnover_aware_risk_budget": {
        "description": "Moderate volatility and drawdown guard with wider rebalance band.",
        "vol_power": 0.35,
        "min_mult": 0.75,
        "max_mult": 1.18,
        "drawdown_threshold": -0.18,
        "drawdown_full": -0.36,
        "max_drawdown_haircut": 0.28,
        "preserve_gross": True,
        "min_weight_change": 0.040,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.18,
    },
    "v3_8_defensive_cash_brake": {
        "description": "Raise small cash buffer only when portfolio-level estimated risk is elevated.",
        "vol_power": 0.25,
        "min_mult": 0.75,
        "max_mult": 1.15,
        "preserve_gross": True,
        "min_weight_change": 0.030,
        "max_style_weight": 0.48,
        "max_industry_weight": 0.18,
        "portfolio_vol_cash_threshold": 0.235,
        "max_extra_cash": 0.08,
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


def clip(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return lo
    return float(min(max(value, lo), hi))


def asset_cluster(asset: str, asset_type: str) -> str:
    asset = str(asset)
    asset_type = str(asset_type)
    if asset == "CASH":
        return "cash"
    if asset_type == "style":
        if asset in {"000016", "000300", "000985"}:
            return "broad_large_style"
        if asset in {"000905", "000852"}:
            return "mid_small_style"
        if asset in {"000922", "000015"}:
            return "dividend_defensive"
        return "other_style"
    if asset in {"801780", "801790", "801193", "801194", "801191"}:
        return "financial_value"
    if asset in {"801080", "801081", "801083", "801085", "801730", "801770", "801102", "801223"}:
        return "growth_beta"
    if asset in {"801040", "801050", "801055", "801030", "801710", "801720", "801890"}:
        return "cyclical_value"
    if asset in {"801010", "801110", "801120", "801130", "801150", "801160", "801170"}:
        return "defensive_consumption"
    return "other_industry"


def build_risk_panel(returns: pd.DataFrame) -> pd.DataFrame:
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["asset"] = ret["asset"].astype(str)
    wide = ret.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    vol60 = wide.rolling(60, min_periods=30).std() * np.sqrt(TRADING_DAYS)
    downside60 = wide.clip(upper=0.0).rolling(60, min_periods=30).std() * np.sqrt(TRADING_DAYS)
    cum = (1.0 + wide).cumprod()
    dd120 = cum / cum.rolling(120, min_periods=40).max() - 1.0
    mom20 = wide.rolling(20, min_periods=10).sum()
    broad = wide[BENCHMARK_ASSET] if BENCHMARK_ASSET in wide.columns else wide.mean(axis=1)
    corr120 = wide.rolling(120, min_periods=60).corr(broad)
    rows = []
    for date in wide.index:
        for asset in wide.columns:
            rows.append(
                {
                    "date": date,
                    "asset": str(asset),
                    "vol60": float(vol60.at[date, asset]) if pd.notna(vol60.at[date, asset]) else np.nan,
                    "downside_vol60": float(downside60.at[date, asset]) if pd.notna(downside60.at[date, asset]) else np.nan,
                    "drawdown120": float(dd120.at[date, asset]) if pd.notna(dd120.at[date, asset]) else np.nan,
                    "momentum20": float(mom20.at[date, asset]) if pd.notna(mom20.at[date, asset]) else np.nan,
                    "corr_to_broad120": float(corr120.at[date, asset]) if date in corr120.index and asset in corr120.columns and pd.notna(corr120.at[date, asset]) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["date", "asset"]).reset_index(drop=True)


def latest_metrics(risk_panel: pd.DataFrame, date: pd.Timestamp, assets: list[str]) -> pd.DataFrame:
    subset = risk_panel[(risk_panel["date"] <= pd.Timestamp(date)) & (risk_panel["asset"].isin(assets))]
    if subset.empty:
        return pd.DataFrame({"asset": assets})
    latest = subset.sort_values("date").groupby("asset", as_index=False).tail(1)
    missing = sorted(set(assets) - set(latest["asset"].astype(str)))
    if missing:
        latest = pd.concat([latest, pd.DataFrame({"asset": missing})], ignore_index=True, sort=False)
    return latest


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


def apply_cluster_caps(weights: dict[str, float], asset_type: dict[str, str], cluster_caps: dict[str, float]) -> tuple[dict[str, float], float]:
    if not cluster_caps:
        return weights, 0.0
    out = dict(weights)
    freed = 0.0
    for cluster, cap in cluster_caps.items():
        assets = [asset for asset in out if asset_cluster(asset, asset_type.get(asset, "")) == cluster]
        total = sum(out.get(asset, 0.0) for asset in assets)
        if total > cap and total > 0:
            scale = cap / total
            for asset in assets:
                old = out[asset]
                out[asset] = old * scale
                freed += old - out[asset]
    return out, freed


def raise_to_min_gross(weights: dict[str, float], asset_type: dict[str, str], caps: dict[str, float], cluster_caps: dict[str, float], min_gross: float) -> dict[str, float]:
    out = dict(weights)
    target_add = float(min_gross) - sum(out.values())
    if target_add <= 1e-10 or not out:
        return out
    for _ in range(8):
        if target_add <= 1e-10:
            break
        cluster_totals: dict[str, float] = {}
        for asset, weight in out.items():
            cluster = asset_cluster(asset, asset_type.get(asset, ""))
            cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + weight
        capacities = {}
        for asset, weight in out.items():
            cluster = asset_cluster(asset, asset_type.get(asset, ""))
            asset_capacity = max(0.0, float(caps.get(asset, 1.0)) - weight)
            cluster_capacity = max(0.0, float(cluster_caps.get(cluster, 1.0)) - cluster_totals.get(cluster, 0.0))
            capacities[asset] = min(asset_capacity, cluster_capacity)
        total_capacity = sum(capacities.values())
        if total_capacity <= 1e-10:
            break
        add_now = min(target_add, total_capacity)
        for asset, capacity in capacities.items():
            if capacity <= 0:
                continue
            out[asset] = out.get(asset, 0.0) + add_now * capacity / total_capacity
        target_add = float(min_gross) - sum(out.values())
    return out


def overlay_targets(base_targets: pd.DataFrame, risk_panel: pd.DataFrame, cfg: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    prev: dict[str, float] = {}
    dates = sorted(pd.to_datetime(base_targets["signal_date"]).unique())
    for date in dates:
        group = base_targets[base_targets["signal_date"].eq(date)].copy()
        state = str(group["state"].dropna().iloc[0]) if "state" in group.columns and group["state"].notna().any() else ""
        bucket = str(group["timing_bucket"].dropna().iloc[0]) if "timing_bucket" in group.columns and group["timing_bucket"].notna().any() else ""
        noncash = group[~group["asset"].astype(str).eq("CASH")].copy()
        cash_weight = float(group.loc[group["asset"].astype(str).eq("CASH"), "weight"].sum())
        gross = float(noncash["weight"].sum())
        asset_type = {str(row["asset"]): str(row.get("asset_type", "")) for _, row in group.iterrows()}
        assets = [str(asset) for asset in noncash["asset"]]
        metrics = latest_metrics(risk_panel, pd.Timestamp(date), assets).set_index("asset")
        vols = pd.to_numeric(metrics.get("vol60", pd.Series(dtype=float)), errors="coerce")
        median_vol = float(vols.replace([np.inf, -np.inf], np.nan).median(skipna=True))
        if not np.isfinite(median_vol) or median_vol <= 1e-9:
            median_vol = 0.20
        adjusted = {}
        for _, row in noncash.iterrows():
            asset = str(row["asset"])
            base_weight = float(row["weight"])
            vol = float(metrics.at[asset, "vol60"]) if asset in metrics.index and pd.notna(metrics.at[asset, "vol60"]) else median_vol
            vol = max(vol, 0.03)
            vol_mult = clip((median_vol / vol) ** float(cfg.get("vol_power", 0.0)), float(cfg.get("min_mult", 0.0)), float(cfg.get("max_mult", 10.0)))
            dd = float(metrics.at[asset, "drawdown120"]) if asset in metrics.index and pd.notna(metrics.at[asset, "drawdown120"]) else 0.0
            dd_mult = 1.0
            if "drawdown_threshold" in cfg and dd < float(cfg["drawdown_threshold"]):
                threshold = float(cfg["drawdown_threshold"])
                full = float(cfg.get("drawdown_full", -0.35))
                severity = clip((threshold - dd) / max(threshold - full, 1e-9), 0.0, 1.0)
                dd_mult = 1.0 - float(cfg.get("max_drawdown_haircut", 0.0)) * severity
            adjusted[asset] = base_weight * vol_mult * dd_mult
            audit_rows.append(
                {
                    "signal_date": date,
                    "variant": variant,
                    "asset": asset,
                    "asset_type": asset_type.get(asset, ""),
                    "cluster": asset_cluster(asset, asset_type.get(asset, "")),
                    "state": state,
                    "base_weight": base_weight,
                    "vol60": vol,
                    "drawdown120": dd,
                    "vol_multiplier": vol_mult,
                    "drawdown_multiplier": dd_mult,
                    "pre_cap_weight": adjusted[asset],
                    "risk_budget_proxy": base_weight * vol,
                }
            )

        caps = {
            asset: float(cfg.get("max_style_weight", 0.50)) if asset_type.get(asset) == "style" else float(cfg.get("max_industry_weight", 0.18))
            for asset in adjusted
        }
        adjusted = cap_and_redistribute(adjusted, caps)
        adjusted, freed_by_cluster = apply_cluster_caps(adjusted, asset_type, cfg.get("cluster_caps", {}))
        if "min_gross_exposure" in cfg:
            adjusted = raise_to_min_gross(adjusted, asset_type, caps, cfg.get("cluster_caps", {}), float(cfg["min_gross_exposure"]))
        adjusted_sum = sum(adjusted.values())
        if bool(cfg.get("preserve_gross", True)) and adjusted_sum > 0:
            adjusted = {asset: weight * gross / adjusted_sum for asset, weight in adjusted.items()}
            adjusted = cap_and_redistribute(adjusted, caps)
        adjusted_sum = sum(adjusted.values())
        if "portfolio_vol_cash_threshold" in cfg and adjusted_sum > 0:
            port_vol_proxy = sum(
                adjusted.get(asset, 0.0) * (float(metrics.at[asset, "vol60"]) if asset in metrics.index and pd.notna(metrics.at[asset, "vol60"]) else median_vol)
                for asset in adjusted
            )
            threshold = float(cfg["portfolio_vol_cash_threshold"])
            if port_vol_proxy > threshold:
                extra_cash = min(float(cfg.get("max_extra_cash", 0.0)), (port_vol_proxy - threshold) / max(threshold, 1e-9) * float(cfg.get("max_extra_cash", 0.0)))
                scale = max(0.0, 1.0 - extra_cash)
                adjusted = {asset: weight * scale for asset, weight in adjusted.items()}
        adjusted_sum = sum(adjusted.values())
        if adjusted_sum > 1.0:
            adjusted = {asset: weight / adjusted_sum for asset, weight in adjusted.items()}

        min_change = float(cfg.get("min_weight_change", 0.0))
        banded = {}
        for asset in set(adjusted) | {asset for asset in prev if asset != "CASH"}:
            old = float(prev.get(asset, 0.0))
            new = float(adjusted.get(asset, 0.0))
            banded[asset] = old if abs(new - old) < min_change else new
        banded = {asset: max(0.0, weight) for asset, weight in banded.items() if weight > 1e-10}
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
                    "v3_8_variant": variant,
                    "timing_bucket": bucket,
                    "cluster": asset_cluster(asset, asset_type.get(asset, "")),
                    "risk_overlay_reason": f"vol_power={cfg.get('vol_power', 0.0)};cluster_freed={freed_by_cluster:.4f}",
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
                "v3_8_variant": variant,
                "timing_bucket": bucket,
                "cluster": "cash",
                "risk_overlay_reason": f"cash_from_overlay={cash - cash_weight:.4f}",
            }
        )
        prev = full
    return pd.DataFrame(rows).sort_values(["signal_date", "asset"]).reset_index(drop=True), pd.DataFrame(audit_rows)


def evaluate_targets(model, v30, wf, panel: dict, targets: pd.DataFrame, output_dir: Path, variant: str, source: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, _ = v30.run_static_costs(model, panel, targets, output_dir, variant, variant)
    rel_summary = v30.add_relative_metrics(summary)
    score_detail = v30.benchmark_relative_score(summary, variant, source)
    smoke = wf.smoke_test_targets(targets)
    model.write_csv(targets, output_dir / "target_weights.csv")
    return {"summary": rel_summary, "score_detail": score_detail, "smoke": smoke}


def score_v38(summary_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(V36_PERF, encoding="utf-8-sig")
    rows = []
    weights = {10.0: 0.50, 20.0: 0.30, 30.0: 0.20}
    for variant, group in summary_all.groupby("candidate"):
        for cost, cost_weight in weights.items():
            row = one_row(group, cost)
            base = one_row(baseline, cost)
            if row.empty or base.empty:
                continue
            ann_delta = float(row["annual_return"] - base["annual_return"])
            dd_delta = float(row["max_drawdown"] - base["max_drawdown"])
            sharpe_delta = float(row["sharpe_no_rf"] - base["sharpe_no_rf"])
            cash_excess = max(float(row["avg_cash_weight"]) - 0.22, 0.0)
            annual_guard_penalty = max(-ann_delta - 0.005, 0.0)
            score = (
                1.80 * float(row["annual_excess_vs_benchmark"])
                + 0.85 * float(row["drawdown_improvement_vs_benchmark"])
                + 0.30 * float(row["sharpe_no_rf"])
                + 0.20 * float(row["information_ratio"])
                + 1.20 * dd_delta
                + 0.50 * ann_delta
                + 0.30 * sharpe_delta
                - 1.25 * annual_guard_penalty
                - 0.40 * cash_excess
                - 0.018 * float(row["avg_trade_turnover"])
            )
            rows.append(
                {
                    "variant": variant,
                    "source": str(row.get("variant", variant)),
                    "cost_bps": cost,
                    "weight": cost_weight,
                    "score_component": score,
                    "weighted_score_component": score * cost_weight,
                    "annual_return": float(row["annual_return"]),
                    "annual_excess_vs_benchmark": float(row["annual_excess_vs_benchmark"]),
                    "sharpe_no_rf": float(row["sharpe_no_rf"]),
                    "information_ratio": float(row["information_ratio"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "drawdown_improvement_vs_benchmark": float(row["drawdown_improvement_vs_benchmark"]),
                    "avg_cash_weight": float(row["avg_cash_weight"]),
                    "avg_trade_turnover": float(row["avg_trade_turnover"]),
                    "annual_return_delta_vs_v36": ann_delta,
                    "max_drawdown_delta_vs_v36": dd_delta,
                    "sharpe_delta_vs_v36": sharpe_delta,
                    "annual_guard_penalty": annual_guard_penalty,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    total = detail.groupby(["variant", "source"], as_index=False).agg(
        v38_risk_budget_score=("weighted_score_component", "sum"),
        avg_annual_excess=("annual_excess_vs_benchmark", "mean"),
        avg_drawdown_improvement=("drawdown_improvement_vs_benchmark", "mean"),
        avg_information_ratio=("information_ratio", "mean"),
        mean_cash_weight=("avg_cash_weight", "mean"),
        avg_annual_delta_vs_v36=("annual_return_delta_vs_v36", "mean"),
        avg_drawdown_delta_vs_v36=("max_drawdown_delta_vs_v36", "mean"),
    )
    detail = detail.merge(total, on=["variant", "source"], how="left")
    table = total.sort_values("v38_risk_budget_score", ascending=False).reset_index(drop=True)
    return detail, table


def select_version(model, version_dir: Path, candidate_results: dict[str, dict], targets_by_variant: dict[str, pd.DataFrame]) -> dict:
    summary_all = []
    score_detail_all = []
    for variant, result in candidate_results.items():
        item = result["summary"].copy()
        item["candidate"] = variant
        summary_all.append(item)
        score_detail_all.append(result["score_detail"])
    summary = pd.concat(summary_all, ignore_index=True, sort=False) if summary_all else pd.DataFrame()
    benchmark_score_detail = pd.concat(score_detail_all, ignore_index=True, sort=False) if score_detail_all else pd.DataFrame()
    v38_score_detail, v38_score_table = score_v38(summary)
    selected = str(v38_score_table.iloc[0]["variant"]) if not v38_score_table.empty else str(summary["candidate"].iloc[0])
    selected_summary = summary[summary["candidate"].eq(selected)].drop(columns=["candidate"])
    selected_targets = targets_by_variant[selected]
    selected_smoke = candidate_results[selected]["smoke"]
    model.write_csv(summary, version_dir / "all_candidate_oos_performance.csv")
    model.write_csv(benchmark_score_detail, version_dir / "benchmark_relative_score_detail.csv")
    model.write_csv(v38_score_detail, version_dir / "v3_8_risk_budget_score_detail.csv")
    model.write_csv(v38_score_table, version_dir / "v3_8_risk_budget_score_table.csv")
    model.write_csv(selected_summary, version_dir / "oos_performance.csv")
    model.write_csv(selected_targets, version_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected_smoke, version_dir / "smoke_test_results.csv")
    candidate_dir = version_dir / "candidates" / selected
    for cost in COSTS:
        src = candidate_dir / f"nav_{selected}_{int(cost)}bps.csv"
        if src.exists():
            model.write_csv(pd.read_csv(src, encoding="utf-8-sig"), version_dir / f"nav_selected_{int(cost)}bps.csv")
    return {
        "selected": selected,
        "summary_all": summary,
        "benchmark_score_detail": benchmark_score_detail,
        "score_detail": v38_score_detail,
        "score_table": v38_score_table,
        "selected_summary": selected_summary,
        "selected_targets": selected_targets,
        "selected_smoke": selected_smoke,
    }


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


def risk_contribution_report(targets: pd.DataFrame, risk_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cluster_rows = []
    for date, group in targets.groupby("signal_date"):
        noncash = group[~group["asset"].astype(str).eq("CASH")].copy()
        assets = [str(asset) for asset in noncash["asset"]]
        metrics = latest_metrics(risk_panel, pd.Timestamp(date), assets).set_index("asset")
        total_proxy = 0.0
        temp = []
        for _, row in noncash.iterrows():
            asset = str(row["asset"])
            weight = float(row["weight"])
            vol = float(metrics.at[asset, "vol60"]) if asset in metrics.index and pd.notna(metrics.at[asset, "vol60"]) else np.nan
            if not np.isfinite(vol):
                vol = 0.20
            cluster = str(row.get("cluster", asset_cluster(asset, str(row.get("asset_type", "")))))
            proxy = weight * vol
            total_proxy += proxy
            temp.append((asset, row.get("asset_type", ""), cluster, weight, vol, proxy))
        for asset, asset_type, cluster, weight, vol, proxy in temp:
            rows.append(
                {
                    "signal_date": date,
                    "asset": asset,
                    "asset_type": asset_type,
                    "cluster": cluster,
                    "weight": weight,
                    "vol60": vol,
                    "risk_budget_proxy": proxy,
                    "risk_contribution_pct": proxy / total_proxy if total_proxy > 0 else 0.0,
                }
            )
        cluster_weights = noncash.groupby("cluster", as_index=False)["weight"].sum() if "cluster" in noncash.columns else pd.DataFrame()
        for _, crow in cluster_weights.iterrows():
            cluster_rows.append({"signal_date": date, "cluster": crow["cluster"], "weight": float(crow["weight"])})
    asset_report = pd.DataFrame(rows)
    cluster_report = pd.DataFrame(cluster_rows)
    if not asset_report.empty:
        asset_report = asset_report.groupby(["asset", "asset_type", "cluster"], as_index=False).agg(
            avg_weight=("weight", "mean"),
            avg_vol60=("vol60", "mean"),
            avg_risk_contribution_pct=("risk_contribution_pct", "mean"),
            max_risk_contribution_pct=("risk_contribution_pct", "max"),
        ).sort_values("avg_risk_contribution_pct", ascending=False)
    if not cluster_report.empty:
        cluster_report = cluster_report.groupby("cluster", as_index=False).agg(
            avg_weight=("weight", "mean"),
            max_weight=("weight", "max"),
        ).sort_values("avg_weight", ascending=False)
    return asset_report, cluster_report


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


def append_self_checks(self_check: pd.DataFrame, comparison: pd.DataFrame, selected_summary: pd.DataFrame, risk_asset: pd.DataFrame, risk_cluster: pd.DataFrame) -> pd.DataFrame:
    def delta(cost: float, baseline: str, col: str) -> float | None:
        rows = comparison[
            comparison["cost_bps"].astype(float).eq(float(cost))
            & comparison["baseline"].astype(str).eq(baseline)
        ]
        if rows.empty:
            return None
        return float(rows[col].iloc[0])

    row10 = one_row(selected_summary, 10.0)
    row20 = one_row(selected_summary, 20.0)
    ann10 = delta(10.0, "V3.6", "annual_return_delta")
    dd10 = delta(10.0, "V3.6", "max_drawdown_delta")
    rows = [
        {
            "check": "v3_8_ann_return_not_down_more_than_50bp_unless_dd_improves",
            "pass": bool(ann10 is not None and (ann10 >= -0.005 or (dd10 is not None and dd10 >= 0.03))),
            "detail": f"ann_delta={ann10:.6f};dd_delta={dd10:.6f}" if ann10 is not None and dd10 is not None else "",
        },
        {
            "check": "v3_8_avg_cash_lte_22pct_10bps",
            "pass": bool(not row10.empty and float(row10["avg_cash_weight"]) <= 0.22),
            "detail": f"{float(row10['avg_cash_weight']):.6f}" if not row10.empty else "",
        },
        {
            "check": "v3_8_positive_excess_20bps",
            "pass": bool(not row20.empty and float(row20["annual_excess_vs_benchmark"]) > 0),
            "detail": f"{float(row20['annual_excess_vs_benchmark']):.6f}" if not row20.empty else "",
        },
        {
            "check": "risk_contribution_report_non_empty",
            "pass": bool(not risk_asset.empty and not risk_cluster.empty),
            "detail": f"asset_rows={len(risk_asset)};cluster_rows={len(risk_cluster)}",
        },
    ]
    return pd.concat([self_check, pd.DataFrame(rows)], ignore_index=True, sort=False)


def write_reports(version_dir: Path, selected: str, selected_summary: pd.DataFrame, score_table: pd.DataFrame, self_check: pd.DataFrame, notes: list[str], extra_tables: dict[str, pd.DataFrame]) -> None:
    report = [
        "# HIRSSM V3.8 Risk Budget Overlay",
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
        "## V3.8 Risk Budget Score Table",
        "",
        score_table.to_markdown(index=False) if not score_table.empty else "No score table.",
    ]
    for name, table in extra_tables.items():
        report.extend(["", f"## {name}", "", table.to_markdown(index=False) if not table.empty else "No rows."])
    (version_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    changelog = [
        "# HIRSSM V3.8 Model Changelog",
        "",
        "## Changed",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Governance",
        "",
        "- V3.8 only changes portfolio construction on top of V3.6 targets.",
        "- V3.6 exact targets are retained as a control candidate.",
        "- The selection score penalizes annual-return loss larger than 50bp unless drawdown improves materially.",
    ]
    (version_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    self_lines = [
        "# HIRSSM V3.8 Self Check Report",
        "",
        self_check.to_markdown(index=False) if not self_check.empty else "Self check pending.",
    ]
    (version_dir / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--v36-targets", default=str(V36_TARGETS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v30 = load_module("hirssm_v3_0_v3_1_benchmark_core", V30_PATH)
    v35 = load_module("hirssm_v3_3_to_v3_5_alpha_factory", V35_PATH)

    root = Path(args.root)
    version_dir = Path(args.output_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    v36_targets = read_targets(Path(args.v36_targets))
    risk_panel = build_risk_panel(panel["returns"])
    model.write_csv(risk_panel, version_dir / "v3_8_asset_risk_panel.csv")

    candidates = {}
    results = {}
    audits = []
    for variant, cfg in V38_VARIANTS.items():
        targets, audit = overlay_targets(v36_targets, risk_panel, cfg, variant)
        candidates[variant] = targets
        audit["variant"] = variant
        audits.append(audit)
        results[variant] = evaluate_targets(model, v30, wf, panel, targets, version_dir / "candidates" / variant, variant, cfg["description"])
        model.write_csv(audit, version_dir / "candidates" / variant / "risk_overlay_audit.csv")
    candidates["v3_8_v36_exact_control"] = v36_targets
    results["v3_8_v36_exact_control"] = evaluate_targets(
        model,
        v30,
        wf,
        panel,
        v36_targets,
        version_dir / "candidates" / "v3_8_v36_exact_control",
        "v3_8_v36_exact_control",
        "Official V3.6 selected target snapshot retained as exact control.",
    )

    selected = select_version(model, version_dir, results, candidates)
    selected_name = selected["selected"]
    audit_all = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    comparison = comparison_table(selected["selected_summary"])
    ablation = variant_ablation_table(selected["summary_all"])
    risk_asset, risk_cluster = risk_contribution_report(selected["selected_targets"], risk_panel)
    model.write_csv(audit_all, version_dir / "v3_8_risk_overlay_audit.csv")
    model.write_csv(comparison, version_dir / "v3_8_vs_v3_2_v3_5_v3_6_comparison.csv")
    model.write_csv(ablation, version_dir / "v3_8_component_ablation.csv")
    model.write_csv(risk_asset, version_dir / "v3_8_selected_asset_risk_contribution.csv")
    model.write_csv(risk_cluster, version_dir / "v3_8_selected_cluster_exposure.csv")

    notes = [
        "Added volatility-budget, correlation-cluster, drawdown-contribution and turnover-aware risk overlay candidates.",
        "Kept the V3.6 selected target snapshot as an exact control candidate.",
        "Selection uses a V3.8 risk-budget objective with explicit penalty for losing more than 50bp annual return unless drawdown improves materially.",
        "Generated risk contribution and cluster exposure reports for the selected candidate.",
    ]
    placeholder = pd.DataFrame([{"check": "pending", "pass": False, "detail": ""}])
    write_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_table"],
        placeholder,
        notes,
        {
            "V3.8 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Selected Asset Risk Contribution": risk_asset.head(30),
            "Selected Cluster Exposure": risk_cluster,
        },
    )
    self_check = v35.make_self_check(selected["selected_targets"], selected["selected_smoke"], selected["selected_summary"], selected["score_table"], version_dir)
    self_check = append_self_checks(self_check, comparison, selected["selected_summary"], risk_asset, risk_cluster)
    model.write_csv(self_check, version_dir / "self_check_results.csv")
    write_reports(
        version_dir,
        selected_name,
        selected["selected_summary"],
        selected["score_table"],
        self_check,
        notes,
        {
            "V3.8 vs Baselines": comparison,
            "Component Ablation": ablation,
            "Selected Asset Risk Contribution": risk_asset.head(30),
            "Selected Cluster Exposure": risk_cluster,
            "Risk Budget Score Detail": selected["score_detail"],
        },
    )

    manifest = {
        "generated_at": now_text(),
        "output_dir": str(version_dir),
        "selected": selected_name,
        "self_check_pass": bool(self_check["pass"].all()),
        "costs": COSTS,
        "benchmark": BENCHMARK_ASSET,
        "inputs": {"v36_targets": str(args.v36_targets)},
    }
    (version_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
