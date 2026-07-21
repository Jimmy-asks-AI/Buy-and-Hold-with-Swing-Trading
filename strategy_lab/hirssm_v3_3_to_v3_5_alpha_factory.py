#!/usr/bin/env python
"""HIRSSM V3.3-V3.5 alpha factory iterations.

V3.3 adds a governed cross-sectional alpha factory on the existing index and
industry universe. V3.4 combines that alpha sleeve with the V3.2 market beta
timing overlay. V3.5 blends V3.2 and V3.4 into a steadier ensemble with caps
and rebalance bands.
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
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory"
V32_TARGETS = ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "walk_forward_target_weights.csv"
BENCHMARK_ASSET = "000985"
COSTS = [5.0, 10.0, 20.0, 30.0]
FACTOR_HORIZON = 21


FACTOR_COLS = [
    "relative_momentum_alpha",
    "industry_prosperity_alpha",
    "style_value_repair_alpha",
    "quality_low_risk_alpha",
    "crowding_relief_alpha",
    "recovery_reversal_alpha",
    "liquidity_confirmation_alpha",
]


V33_VARIANTS = {
    "v3_3_balanced_alpha_factory": {
        "description": "Balanced factor factory: prosperity, value repair, risk quality and crowding relief.",
        "factor_weights": {
            "relative_momentum_alpha": 0.22,
            "industry_prosperity_alpha": 0.22,
            "style_value_repair_alpha": 0.18,
            "quality_low_risk_alpha": 0.16,
            "crowding_relief_alpha": 0.10,
            "recovery_reversal_alpha": 0.06,
            "liquidity_confirmation_alpha": 0.06,
        },
        "min_gate_multiplier": 0.35,
        "style_budget_by_state": {
            "risk_on_trend": 0.42,
            "risk_on_overheat": 0.54,
            "range_bound": 0.56,
            "risk_off_decline": 0.58,
            "crash_rebound": 0.50,
        },
        "industry_budget_by_state": {
            "risk_on_trend": 0.54,
            "risk_on_overheat": 0.28,
            "range_bound": 0.34,
            "risk_off_decline": 0.14,
            "crash_rebound": 0.42,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.02,
            "risk_on_overheat": 0.12,
            "range_bound": 0.08,
            "risk_off_decline": 0.28,
            "crash_rebound": 0.06,
        },
    },
    "v3_3_prosperity_momentum": {
        "description": "Industry prosperity led alpha: emphasize relative momentum, breadth proxy and liquidity confirmation.",
        "factor_weights": {
            "relative_momentum_alpha": 0.30,
            "industry_prosperity_alpha": 0.32,
            "style_value_repair_alpha": 0.08,
            "quality_low_risk_alpha": 0.10,
            "crowding_relief_alpha": 0.06,
            "recovery_reversal_alpha": 0.04,
            "liquidity_confirmation_alpha": 0.10,
        },
        "min_gate_multiplier": 0.30,
        "style_budget_by_state": {
            "risk_on_trend": 0.36,
            "risk_on_overheat": 0.46,
            "range_bound": 0.48,
            "risk_off_decline": 0.56,
            "crash_rebound": 0.42,
        },
        "industry_budget_by_state": {
            "risk_on_trend": 0.62,
            "risk_on_overheat": 0.36,
            "range_bound": 0.42,
            "risk_off_decline": 0.14,
            "crash_rebound": 0.52,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.02,
            "risk_on_overheat": 0.14,
            "range_bound": 0.08,
            "risk_off_decline": 0.30,
            "crash_rebound": 0.04,
        },
    },
    "v3_3_value_quality_repair": {
        "description": "Value and quality repair alpha: favor low-risk value repair and defensive style exposure.",
        "factor_weights": {
            "relative_momentum_alpha": 0.12,
            "industry_prosperity_alpha": 0.12,
            "style_value_repair_alpha": 0.28,
            "quality_low_risk_alpha": 0.24,
            "crowding_relief_alpha": 0.12,
            "recovery_reversal_alpha": 0.08,
            "liquidity_confirmation_alpha": 0.04,
        },
        "min_gate_multiplier": 0.40,
        "style_budget_by_state": {
            "risk_on_trend": 0.52,
            "risk_on_overheat": 0.62,
            "range_bound": 0.62,
            "risk_off_decline": 0.68,
            "crash_rebound": 0.58,
        },
        "industry_budget_by_state": {
            "risk_on_trend": 0.42,
            "risk_on_overheat": 0.22,
            "range_bound": 0.28,
            "risk_off_decline": 0.08,
            "crash_rebound": 0.34,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.04,
            "risk_on_overheat": 0.16,
            "range_bound": 0.10,
            "risk_off_decline": 0.24,
            "crash_rebound": 0.08,
        },
    },
    "v3_3_recovery_repair": {
        "description": "Crash repair alpha: boost valuation repair and rebound signals after deep drawdowns.",
        "factor_weights": {
            "relative_momentum_alpha": 0.14,
            "industry_prosperity_alpha": 0.18,
            "style_value_repair_alpha": 0.22,
            "quality_low_risk_alpha": 0.14,
            "crowding_relief_alpha": 0.10,
            "recovery_reversal_alpha": 0.16,
            "liquidity_confirmation_alpha": 0.06,
        },
        "min_gate_multiplier": 0.30,
        "style_budget_by_state": {
            "risk_on_trend": 0.44,
            "risk_on_overheat": 0.54,
            "range_bound": 0.56,
            "risk_off_decline": 0.56,
            "crash_rebound": 0.48,
        },
        "industry_budget_by_state": {
            "risk_on_trend": 0.52,
            "risk_on_overheat": 0.28,
            "range_bound": 0.34,
            "risk_off_decline": 0.14,
            "crash_rebound": 0.48,
        },
        "cash_floor_by_state": {
            "risk_on_trend": 0.04,
            "risk_on_overheat": 0.14,
            "range_bound": 0.10,
            "risk_off_decline": 0.30,
            "crash_rebound": 0.04,
        },
    },
}


V34_BETA_VARIANTS = {
    "v3_4_balanced_alpha_beta": {
        "description": "V3.3 alpha factory with balanced V3.2-style market beta timing.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 0.96,
            "neutral": 0.92,
            "cautious": 0.76,
            "risk_off": 0.50,
            "panic": 0.30,
        },
        "min_gross_by_state": {"risk_on_trend": 0.94, "crash_rebound": 0.78},
        "max_gross_by_state": {"risk_off_decline": 0.66, "risk_on_overheat": 0.94},
    },
    "v3_4_recovery_alpha_beta": {
        "description": "V3.3 alpha factory with stronger recovery participation and risk-off caps.",
        "target_gross_by_bucket": {
            "risk_on": 1.00,
            "recovery": 1.00,
            "neutral": 0.96,
            "cautious": 0.82,
            "risk_off": 0.52,
            "panic": 0.28,
        },
        "min_gross_by_state": {"risk_on_trend": 0.96, "crash_rebound": 0.88},
        "max_gross_by_state": {"risk_off_decline": 0.70, "risk_on_overheat": 0.96},
    },
    "v3_4_defensive_alpha_beta": {
        "description": "V3.3 alpha factory with defensive beta caps and higher cash in weak breadth.",
        "target_gross_by_bucket": {
            "risk_on": 0.96,
            "recovery": 0.90,
            "neutral": 0.86,
            "cautious": 0.68,
            "risk_off": 0.42,
            "panic": 0.22,
        },
        "min_gross_by_state": {"risk_on_trend": 0.86, "crash_rebound": 0.66},
        "max_gross_by_state": {"risk_off_decline": 0.56, "risk_on_overheat": 0.88},
    },
}


V35_BLEND_VARIANTS = {
    "v3_5_balanced_ensemble": {
        "description": "Equal blend of V3.2 beta-timed sleeve and V3.4 alpha-beta sleeve.",
        "v34_weight": 0.50,
        "min_weight_change": 0.015,
        "max_style_weight": 0.48,
        "max_industry_weight": 0.22,
    },
    "v3_5_alpha_dominant": {
        "description": "Higher allocation to V3.4 alpha-beta sleeve with moderate turnover bands.",
        "v34_weight": 0.70,
        "min_weight_change": 0.015,
        "max_style_weight": 0.48,
        "max_industry_weight": 0.22,
    },
    "v3_5_beta_anchor": {
        "description": "Higher allocation to V3.2 beta-timed sleeve for lower model risk.",
        "v34_weight": 0.35,
        "min_weight_change": 0.020,
        "max_style_weight": 0.50,
        "max_industry_weight": 0.20,
    },
    "v3_5_cash_guarded": {
        "description": "Balanced ensemble with stricter caps and cash preservation after weak timing signals.",
        "v34_weight": 0.55,
        "min_weight_change": 0.020,
        "max_style_weight": 0.44,
        "max_industry_weight": 0.18,
        "cash_add_by_bucket": {"risk_off": 0.08, "panic": 0.12, "cautious": 0.04},
    },
}


STATE_FACTOR_ADJ = {
    "risk_on_trend": {
        "relative_momentum_alpha": 1.20,
        "industry_prosperity_alpha": 1.20,
        "style_value_repair_alpha": 0.80,
        "quality_low_risk_alpha": 0.85,
        "crowding_relief_alpha": 0.85,
        "recovery_reversal_alpha": 0.70,
        "liquidity_confirmation_alpha": 1.10,
    },
    "risk_on_overheat": {
        "relative_momentum_alpha": 0.75,
        "industry_prosperity_alpha": 0.75,
        "style_value_repair_alpha": 1.10,
        "quality_low_risk_alpha": 1.25,
        "crowding_relief_alpha": 1.30,
        "recovery_reversal_alpha": 0.60,
        "liquidity_confirmation_alpha": 0.85,
    },
    "range_bound": {
        "relative_momentum_alpha": 0.95,
        "industry_prosperity_alpha": 0.95,
        "style_value_repair_alpha": 1.15,
        "quality_low_risk_alpha": 1.10,
        "crowding_relief_alpha": 1.10,
        "recovery_reversal_alpha": 0.95,
        "liquidity_confirmation_alpha": 1.00,
    },
    "risk_off_decline": {
        "relative_momentum_alpha": 0.55,
        "industry_prosperity_alpha": 0.45,
        "style_value_repair_alpha": 1.20,
        "quality_low_risk_alpha": 1.35,
        "crowding_relief_alpha": 1.25,
        "recovery_reversal_alpha": 0.70,
        "liquidity_confirmation_alpha": 0.75,
    },
    "crash_rebound": {
        "relative_momentum_alpha": 0.90,
        "industry_prosperity_alpha": 1.10,
        "style_value_repair_alpha": 1.25,
        "quality_low_risk_alpha": 0.95,
        "crowding_relief_alpha": 1.05,
        "recovery_reversal_alpha": 1.35,
        "liquidity_confirmation_alpha": 1.15,
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


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def cs_z(series: pd.Series, dates: pd.Series) -> pd.Series:
    temp = pd.DataFrame({"date": pd.to_datetime(dates), "x": pd.to_numeric(series, errors="coerce")})

    def transform(x: pd.Series) -> pd.Series:
        x = x.replace([np.inf, -np.inf], np.nan)
        med = x.median(skipna=True)
        mad = (x - med).abs().median(skipna=True)
        if pd.isna(mad) or mad <= 1e-12:
            std = x.std(skipna=True, ddof=0)
            if pd.isna(std) or std <= 1e-12:
                return pd.Series(0.0, index=x.index)
            return ((x - x.mean(skipna=True)) / std).clip(-3, 3).fillna(0.0)
        return ((x - med) / (1.4826 * mad)).clip(-3, 3).fillna(0.0)

    return temp.groupby("date")["x"].transform(transform).fillna(0.0)


def add_alpha_factors(eligible: pd.DataFrame, horizon: int = FACTOR_HORIZON) -> pd.DataFrame:
    out = eligible.sort_values(["asset", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset"] = out["asset"].astype(str)
    out["asset_type"] = out["asset_type"].astype(str)
    out["future_ret_21"] = out.groupby("asset")["close"].shift(-horizon) / out["close"] - 1.0

    trend = safe_numeric(out, "trend_expert_score")
    rel = safe_numeric(out, "relative_strength_score")
    valuation = safe_numeric(out, "valuation_repair_score")
    risk = safe_numeric(out, "risk_compression_score")
    liquidity = safe_numeric(out, "liquidity_score")
    reversal = safe_numeric(out, "range_reversal_score")
    defensive = safe_numeric(out, "defensive_score")
    crowding_z = cs_z(safe_numeric(out, "crowding_score"), out["date"])
    low_vol_z = -cs_z(safe_numeric(out, "vol_60"), out["date"])
    drawdown_z = -cs_z(safe_numeric(out, "max_drawdown_120"), out["date"])
    price_confirm = safe_numeric(out, "price_volume_confirmation_z")
    amount_z = safe_numeric(out, "amount_zscore_60_z")

    is_industry = out["asset_type"].eq("industry")
    is_style = out["asset_type"].eq("style")

    out["relative_momentum_alpha"] = cs_z(0.55 * rel + 0.30 * trend + 0.15 * price_confirm, out["date"])
    industry_raw = 0.38 * rel + 0.24 * trend + 0.18 * liquidity + 0.12 * price_confirm - 0.08 * crowding_z
    out["industry_prosperity_alpha"] = cs_z(industry_raw.where(is_industry, 0.35 * rel + 0.20 * trend), out["date"])
    style_value_raw = 0.50 * valuation + 0.22 * risk + 0.18 * defensive + 0.10 * rel
    out["style_value_repair_alpha"] = cs_z(style_value_raw.where(is_style, 0.0), out["date"])
    out["quality_low_risk_alpha"] = cs_z(0.50 * risk + 0.25 * low_vol_z + 0.15 * defensive + 0.10 * drawdown_z, out["date"])
    out["crowding_relief_alpha"] = cs_z(-0.65 * crowding_z + 0.20 * risk + 0.15 * low_vol_z, out["date"])
    out["recovery_reversal_alpha"] = cs_z(0.45 * reversal + 0.25 * valuation + 0.20 * risk + 0.10 * price_confirm, out["date"])
    out["liquidity_confirmation_alpha"] = cs_z(0.55 * liquidity + 0.25 * price_confirm + 0.20 * amount_z, out["date"])
    return out


def monthly_factor_panel(model, alpha_panel: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    rebalance_dates = set(pd.to_datetime(model.month_end_dates(alpha_panel["date"])))
    monthly = alpha_panel[alpha_panel["date"].isin(rebalance_dates)].copy()
    states = regimes[["date", "state"]].copy()
    states["date"] = pd.to_datetime(states["date"])
    monthly = monthly.merge(states, on="date", how="left")
    return monthly.sort_values(["date", "asset"]).reset_index(drop=True)


def rank_ic(group: pd.DataFrame, factor: str, ret_col: str = "future_ret_21") -> float:
    valid = group[[factor, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.shape[0] < 5:
        return np.nan
    x = valid[factor].rank()
    y = valid[ret_col].rank()
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan
    return float(x.corr(y))


def compute_factor_gate_history(monthly: pd.DataFrame, factors: list[str], train_years: int = 5) -> pd.DataFrame:
    rows = []
    years = sorted(int(y) for y in monthly["date"].dt.year.dropna().unique())
    for year in years:
        train_start = pd.Timestamp(year=year - train_years, month=1, day=1)
        train_end = pd.Timestamp(year=year, month=1, day=1)
        train = monthly[(monthly["date"] >= train_start) & (monthly["date"] < train_end)].copy()
        for factor in factors:
            ics = train.groupby("date").apply(lambda g, f=factor: rank_ic(g, f)).dropna()
            obs = int(ics.shape[0])
            mean_ic = float(ics.mean()) if obs else np.nan
            std_ic = float(ics.std(ddof=0)) if obs else np.nan
            icir = float(mean_ic / std_ic) if obs and std_ic > 1e-12 else 0.0
            pos_rate = float((ics > 0).mean()) if obs else np.nan
            if obs < 24:
                multiplier = 0.50
                decision = "limited_history_observe"
            elif mean_ic > 0.010 and pos_rate >= 0.54 and icir > 0.05:
                multiplier = min(1.35, 0.85 + 5.0 * mean_ic + 0.25 * min(icir, 1.0))
                decision = "promote"
            elif mean_ic > 0.0 and pos_rate >= 0.50:
                multiplier = 0.85
                decision = "soft_pass"
            elif mean_ic < -0.015 and pos_rate <= 0.46:
                multiplier = 0.20
                decision = "strong_negative_gate"
            else:
                multiplier = 0.55
                decision = "observe"
            rows.append(
                {
                    "test_year": year,
                    "factor": factor,
                    "observations": obs,
                    "rank_ic_mean": mean_ic,
                    "rank_ic_std": std_ic,
                    "icir": icir,
                    "positive_ic_rate": pos_rate,
                    "multiplier": float(multiplier),
                    "decision": decision,
                    "train_start": train_start,
                    "train_end": train_end - pd.Timedelta(days=1),
                }
            )
    return pd.DataFrame(rows)


def gate_for_year(gate_history: pd.DataFrame, year: int, min_multiplier: float) -> dict[str, float]:
    rows = gate_history[gate_history["test_year"].eq(int(year))]
    out = {factor: 0.50 for factor in FACTOR_COLS}
    if rows.empty:
        return out
    for _, row in rows.iterrows():
        out[str(row["factor"])] = max(min_multiplier, float(row["multiplier"]))
    return out


def weighted_alpha(day: pd.DataFrame, cfg: dict, gates: dict[str, float], state: str) -> pd.Series:
    weights = cfg["factor_weights"]
    adj = STATE_FACTOR_ADJ.get(state, STATE_FACTOR_ADJ["range_bound"])
    score = pd.Series(0.0, index=day.index, dtype=float)
    for factor, base_weight in weights.items():
        if factor not in day.columns:
            continue
        score = score + float(base_weight) * float(gates.get(factor, 0.5)) * float(adj.get(factor, 1.0)) * safe_numeric(day, factor)
    return cs_z(score, day["date"])


def select_candidates(day: pd.DataFrame, asset_type: str, score_col: str, max_n: int, min_n: int) -> pd.DataFrame:
    pool = day[day["asset_type"].astype(str).eq(asset_type)].copy()
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
    if pool.empty:
        return pool
    pool = pool.sort_values(score_col, ascending=False)
    positive = pool[pool[score_col] > 0]
    selected = positive.head(max_n)
    if selected.shape[0] < min_n:
        selected = pool.head(min(max_n, max(min_n, pool.shape[0])))
    return selected


def allocate_budget(selected: pd.DataFrame, budget: float, score_col: str) -> dict[str, float]:
    if selected.empty or budget <= 0:
        return {}
    raw_score = pd.to_numeric(selected[score_col], errors="coerce").fillna(0.0)
    shifted = raw_score - raw_score.min() + 0.05
    if float(shifted.sum()) <= 0:
        shifted = raw_score.rank(method="first")
    vol = pd.to_numeric(selected.get("vol_60", pd.Series(0.16, index=selected.index)), errors="coerce").replace(0, np.nan)
    vol = vol.fillna(vol.median()).clip(lower=0.03)
    raw = shifted.clip(lower=0.01) / np.sqrt(vol)
    total = float(raw.sum())
    if total <= 0:
        return {}
    return {str(asset): float(weight) for asset, weight in zip(selected["asset"], budget * raw / total)}


def cap_and_redistribute(weights: dict[str, float], caps: dict[str, float]) -> dict[str, float]:
    out = {asset: max(0.0, float(weight)) for asset, weight in weights.items() if float(weight) > 0}
    for _ in range(8):
        excess = 0.0
        capacity = {}
        changed = False
        for asset, weight in list(out.items()):
            cap = caps.get(asset, 1.0)
            if weight > cap:
                excess += weight - cap
                out[asset] = cap
                changed = True
            else:
                capacity[asset] = max(0.0, cap - weight)
        if not changed or excess <= 1e-12:
            break
        cap_sum = sum(capacity.values())
        if cap_sum <= 1e-12:
            break
        for asset, cap_left in capacity.items():
            out[asset] += excess * cap_left / cap_sum
    return out


def build_alpha_targets(model, monthly: pd.DataFrame, gate_history: pd.DataFrame, cfg: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    decisions = []
    prev_weights: dict[str, float] = {}
    min_gate = float(cfg.get("min_gate_multiplier", 0.30))
    for signal_date, group in monthly.groupby("date", sort=True):
        day = group.copy()
        state = str(day["state"].dropna().iloc[0]) if day["state"].notna().any() else "range_bound"
        gates = gate_for_year(gate_history, int(pd.Timestamp(signal_date).year), min_gate)
        day["alpha_factory_score"] = weighted_alpha(day, cfg, gates, state)
        style_budget = float(cfg["style_budget_by_state"].get(state, cfg["style_budget_by_state"]["range_bound"]))
        industry_budget = float(cfg["industry_budget_by_state"].get(state, cfg["industry_budget_by_state"]["range_bound"]))
        cash_floor = float(cfg["cash_floor_by_state"].get(state, cfg["cash_floor_by_state"]["range_bound"]))
        budget_total = min(1.0, max(0.0, style_budget) + max(0.0, industry_budget) + max(0.0, cash_floor))
        if budget_total > 1.0:
            style_budget = style_budget / budget_total
            industry_budget = industry_budget / budget_total
            cash_floor = cash_floor / budget_total
        style_selected = select_candidates(day, "style", "alpha_factory_score", max_n=3, min_n=1)
        industry_max = 5 if state != "risk_off_decline" else 2
        industry_min = 2 if state not in {"risk_off_decline", "risk_on_overheat"} else 1
        industry_selected = select_candidates(day, "industry", "alpha_factory_score", max_n=industry_max, min_n=industry_min)
        weights = {}
        weights.update(allocate_budget(style_selected, style_budget, "alpha_factory_score"))
        industry_weights = allocate_budget(industry_selected, industry_budget, "alpha_factory_score")
        for asset, weight in industry_weights.items():
            weights[asset] = weights.get(asset, 0.0) + weight
        asset_types = {str(row["asset"]): str(row["asset_type"]) for _, row in day.iterrows()}
        caps = {
            asset: (0.42 if asset_types.get(asset) == "style" else 0.20)
            for asset in weights
        }
        weights = cap_and_redistribute(weights, caps)
        noncash_sum = sum(weights.values())
        if noncash_sum > 1.0 - cash_floor:
            scale = (1.0 - cash_floor) / noncash_sum if noncash_sum > 0 else 0.0
            weights = {asset: weight * scale for asset, weight in weights.items()}
            noncash_sum = sum(weights.values())
        cash = max(0.0, 1.0 - noncash_sum)
        current = dict(weights)
        current["CASH"] = cash
        turnover = sum(abs(current.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in set(current) | set(prev_weights))
        for asset, weight in sorted(weights.items()):
            ref = day[day["asset"].astype(str).eq(asset)].head(1)
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": asset,
                    "weight": float(weight),
                    "state": state,
                    "asset_type": asset_types.get(asset, ""),
                    "score": float(ref["alpha_factory_score"].iloc[0]) if not ref.empty else 0.0,
                    "risk_adjusted_alpha": float(ref["alpha_factory_score"].iloc[0]) if not ref.empty else 0.0,
                    "turnover": turnover,
                    "v3_3_variant": variant,
                    "gate_summary": ",".join(f"{k}={v:.2f}" for k, v in sorted(gates.items())),
                }
            )
        rows.append(
            {
                "signal_date": signal_date,
                "asset": "CASH",
                "weight": float(cash),
                "state": state,
                "asset_type": "cash",
                "score": 0.0,
                "risk_adjusted_alpha": 0.0,
                "turnover": turnover,
                "v3_3_variant": variant,
                "gate_summary": ",".join(f"{k}={v:.2f}" for k, v in sorted(gates.items())),
            }
        )
        decisions.append(
            {
                "signal_date": signal_date,
                "variant": variant,
                "state": state,
                "style_budget": style_budget,
                "industry_budget": industry_budget,
                "cash": cash,
                "turnover": turnover,
                "selected_style": ",".join(style_selected["asset"].astype(str).tolist()),
                "selected_industry": ",".join(industry_selected["asset"].astype(str).tolist()),
                **{f"gate_{k}": v for k, v in gates.items()},
            }
        )
        prev_weights = current
    targets = pd.DataFrame(rows).sort_values(["signal_date", "asset"]).reset_index(drop=True)
    decisions_df = pd.DataFrame(decisions)
    return targets, decisions_df


def read_targets(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path, encoding="utf-8-sig")
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["asset"] = out["asset"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def evaluate_targets(model, v30, wf, panel: dict, targets: pd.DataFrame, output_dir: Path, variant: str, source: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, _ = v30.run_static_costs(model, panel, targets, output_dir, variant, variant)
    rel_summary = v30.add_relative_metrics(summary)
    score_detail = v30.benchmark_relative_score(summary, variant, source)
    smoke = wf.smoke_test_targets(targets)
    model.write_csv(targets, output_dir / "target_weights.csv")
    return {"summary": rel_summary, "score_detail": score_detail, "smoke": smoke}


def select_version(model, v30, wf, version_dir: Path, candidate_results: dict[str, dict], targets_by_variant: dict[str, pd.DataFrame]) -> dict:
    all_summary = []
    all_scores = []
    for variant, result in candidate_results.items():
        item = result["summary"].copy()
        item["candidate"] = variant
        all_summary.append(item)
        all_scores.append(result["score_detail"])
    summary_all = pd.concat(all_summary, ignore_index=True, sort=False) if all_summary else pd.DataFrame()
    score_detail = pd.concat(all_scores, ignore_index=True, sort=False) if all_scores else pd.DataFrame()
    score_table = v30.make_candidate_table(score_detail)
    selected = str(score_table.iloc[0]["variant"])
    selected_summary = summary_all[summary_all["candidate"].eq(selected)].drop(columns=["candidate"])
    selected_targets = targets_by_variant[selected]
    selected_smoke = candidate_results[selected]["smoke"]
    model.write_csv(summary_all, version_dir / "all_candidate_oos_performance.csv")
    model.write_csv(score_detail, version_dir / "benchmark_relative_score_detail.csv")
    model.write_csv(score_table, version_dir / "benchmark_relative_score_table.csv")
    model.write_csv(selected_summary, version_dir / "oos_performance.csv")
    model.write_csv(selected_targets, version_dir / "walk_forward_target_weights.csv")
    model.write_csv(selected_smoke, version_dir / "smoke_test_results.csv")
    candidate_dir = version_dir / "candidates" / selected
    for cost in COSTS:
        src = candidate_dir / f"nav_{selected}_{int(cost)}bps.csv"
        if src.exists():
            nav = pd.read_csv(src, encoding="utf-8-sig")
            model.write_csv(nav, version_dir / f"nav_selected_{int(cost)}bps.csv")
    return {
        "selected": selected,
        "summary_all": summary_all,
        "score_detail": score_detail,
        "score_table": score_table,
        "selected_summary": selected_summary,
        "selected_targets": selected_targets,
        "selected_smoke": selected_smoke,
    }


def make_self_check(targets: pd.DataFrame, smoke: pd.DataFrame, summary: pd.DataFrame, score_table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    row10 = summary[summary["cost_bps"].astype(float).eq(10.0)].head(1) if not summary.empty else pd.DataFrame()
    row20 = summary[summary["cost_bps"].astype(float).eq(20.0)].head(1) if not summary.empty else pd.DataFrame()
    weight_sums = targets.groupby("signal_date")["weight"].sum() if not targets.empty else pd.Series(dtype=float)
    rows = [
        {"check": "smoke_all_pass", "pass": bool(smoke["pass"].all()) if not smoke.empty else False, "detail": ""},
        {"check": "required_cost_rows", "pass": bool(set(COSTS).issubset(set(summary["cost_bps"].astype(float)))) if not summary.empty else False, "detail": str(sorted(summary["cost_bps"].astype(float).tolist())) if not summary.empty else ""},
        {"check": "no_negative_weights", "pass": bool((pd.to_numeric(targets["weight"], errors="coerce").fillna(0.0) >= -1e-9).all()) if not targets.empty else False, "detail": ""},
        {"check": "no_leverage_weight_sum_lte_1", "pass": bool((weight_sums <= 1.000001).all()) if not weight_sums.empty else False, "detail": f"max={float(weight_sums.max()):.6f}" if not weight_sums.empty else ""},
        {"check": "positive_annual_excess_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "investment_gate_annual_excess_above_3pct_10bps", "pass": bool(not row10.empty and float(row10["annual_excess_vs_benchmark"].iloc[0]) >= 0.03), "detail": f"{float(row10['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "positive_annual_excess_20bps", "pass": bool(not row20.empty and float(row20["annual_excess_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row20['annual_excess_vs_benchmark'].iloc[0]):.6f}" if not row20.empty else ""},
        {"check": "drawdown_better_than_benchmark_10bps", "pass": bool(not row10.empty and float(row10["drawdown_improvement_vs_benchmark"].iloc[0]) > 0.0), "detail": f"{float(row10['drawdown_improvement_vs_benchmark'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "avg_cash_not_too_high_10bps", "pass": bool(not row10.empty and float(row10["avg_cash_weight"].iloc[0]) <= 0.35), "detail": f"{float(row10['avg_cash_weight'].iloc[0]):.6f}" if not row10.empty else ""},
        {"check": "score_table_non_empty", "pass": bool(not score_table.empty), "detail": str(score_table.shape[0]) if not score_table.empty else ""},
    ]
    for name in ["MODEL_CHANGELOG.md", "WALK_FORWARD_REPORT.md", "SELF_CHECK_REPORT.md"]:
        rows.append({"check": f"exists_{name}", "pass": bool((output_dir / name).exists()), "detail": name})
    return pd.DataFrame(rows)


def write_version_reports(version_dir: Path, title: str, selected: str, summary: pd.DataFrame, score_detail: pd.DataFrame, self_check: pd.DataFrame | None, notes: list[str], extra_tables: dict[str, pd.DataFrame] | None = None) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    score_table = score_detail[["variant", "source", "benchmark_relative_score", "avg_annual_excess", "avg_drawdown_improvement", "avg_information_ratio", "mean_cash_weight"]].drop_duplicates().sort_values("benchmark_relative_score", ascending=False) if not score_detail.empty else pd.DataFrame()
    report = [
        f"# {title}",
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
        summary.to_markdown(index=False) if not summary.empty else "No summary.",
        "",
        "## Candidate Score Table",
        "",
        score_table.to_markdown(index=False) if not score_table.empty else "No score table.",
    ]
    for name, table in (extra_tables or {}).items():
        report.extend(["", f"## {name}", "", table.to_markdown(index=False) if not table.empty else "No rows."])
    (version_dir / "WALK_FORWARD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    changelog = [
        f"# {title} Model Changelog",
        "",
        "## Changed",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Governance",
        "",
        "- Candidate selection uses the same benchmark-relative objective as V3.0.",
        "- 10/20/30bps costs affect selection; 5bps is reported as an optimistic scenario.",
        "- Point-in-time fundamental data is not fabricated. Current V3.3 factors are price/volume/valuation-history proxies available in the existing index panel.",
    ]
    (version_dir / "MODEL_CHANGELOG.md").write_text("\n".join(changelog), encoding="utf-8")
    self_lines = [
        f"# {title} Self Check Report",
        "",
        self_check.to_markdown(index=False) if self_check is not None and not self_check.empty else "Self check pending.",
    ]
    (version_dir / "SELF_CHECK_REPORT.md").write_text("\n".join(self_lines), encoding="utf-8")


def blend_targets(primary: pd.DataFrame, secondary: pd.DataFrame, cfg: dict, timing: pd.DataFrame, variant: str) -> pd.DataFrame:
    alpha_weight = float(cfg["v34_weight"])
    beta_weight = 1.0 - alpha_weight
    min_change = float(cfg.get("min_weight_change", 0.0))
    all_dates = sorted(set(pd.to_datetime(primary["signal_date"])) & set(pd.to_datetime(secondary["signal_date"])))
    rows = []
    prev: dict[str, float] = {}
    timing_lookup = timing.sort_values("date")
    for date in all_dates:
        p = primary[primary["signal_date"].eq(date)]
        s = secondary[secondary["signal_date"].eq(date)]
        weights: dict[str, float] = {}
        asset_type: dict[str, str] = {}
        state = ""
        for _, row in p.iterrows():
            asset = str(row["asset"])
            weights[asset] = weights.get(asset, 0.0) + alpha_weight * float(row["weight"])
            asset_type[asset] = str(row.get("asset_type", ""))
            state = str(row.get("state", state))
        for _, row in s.iterrows():
            asset = str(row["asset"])
            weights[asset] = weights.get(asset, 0.0) + beta_weight * float(row["weight"])
            asset_type[asset] = str(row.get("asset_type", asset_type.get(asset, "")))
            state = str(row.get("state", state))
        latest = timing_lookup[timing_lookup["date"] <= date].tail(1)
        bucket = str(latest["timing_bucket"].iloc[0]) if not latest.empty else "neutral"
        cash_add = float(cfg.get("cash_add_by_bucket", {}).get(bucket, 0.0))
        noncash = {asset: max(0.0, weight) for asset, weight in weights.items() if asset != "CASH" and weight > 0}
        caps = {
            asset: (float(cfg.get("max_style_weight", 0.50)) if asset_type.get(asset) == "style" else float(cfg.get("max_industry_weight", 0.22)))
            for asset in noncash
        }
        noncash = cap_and_redistribute(noncash, caps)
        noncash_sum = sum(noncash.values())
        if cash_add > 0 and noncash_sum > 0:
            scale = max(0.0, 1.0 - cash_add) / noncash_sum
            if scale < 1.0:
                noncash = {asset: weight * scale for asset, weight in noncash.items()}
                noncash_sum = sum(noncash.values())
        current = dict(noncash)
        for asset in set(current) | set(prev):
            old = prev.get(asset, 0.0)
            new = current.get(asset, 0.0)
            if abs(new - old) < min_change:
                current[asset] = old
        noncash_sum = sum(max(0.0, v) for v in current.values())
        if noncash_sum > 1.0:
            current = {asset: value / noncash_sum for asset, value in current.items()}
            noncash_sum = 1.0
        cash = max(0.0, 1.0 - noncash_sum)
        full = dict(current)
        full["CASH"] = cash
        turnover = sum(abs(full.get(asset, 0.0) - prev.get(asset, 0.0)) for asset in set(full) | set(prev))
        for asset, weight in sorted(current.items()):
            if weight <= 0:
                continue
            rows.append(
                {
                    "signal_date": date,
                    "asset": asset,
                    "weight": float(weight),
                    "state": state,
                    "asset_type": asset_type.get(asset, ""),
                    "score": 0.0,
                    "risk_adjusted_alpha": 0.0,
                    "turnover": turnover,
                    "v3_5_variant": variant,
                    "blend_v34_weight": alpha_weight,
                    "timing_bucket": bucket,
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
                "turnover": turnover,
                "v3_5_variant": variant,
                "blend_v34_weight": alpha_weight,
                "timing_bucket": bucket,
            }
        )
        prev = full
    return pd.DataFrame(rows).sort_values(["signal_date", "asset"]).reset_index(drop=True)


def run_version_self_check(model, targets: pd.DataFrame, smoke: pd.DataFrame, summary: pd.DataFrame, score_table: pd.DataFrame, version_dir: Path, title: str, selected: str, score_detail: pd.DataFrame, notes: list[str], extra_tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    write_version_reports(version_dir, title, selected, summary, score_detail, None, notes, extra_tables=extra_tables)
    self_check = make_self_check(targets, smoke, summary, score_table, version_dir)
    model.write_csv(self_check, version_dir / "self_check_results.csv")
    write_version_reports(version_dir, title, selected, summary, score_detail, self_check, notes, extra_tables=extra_tables)
    return self_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--v32-targets", default=str(V32_TARGETS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    model = load_module("hirssm_v2_model", MODEL_PATH)
    wf = load_module("hirssm_v2_walk_forward", WF_PATH)
    v30 = load_module("hirssm_v3_0_v3_1_benchmark_core", V30_PATH)
    v32 = load_module("hirssm_v3_2_market_beta_timing", V32_PATH)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    alpha_panel = add_alpha_factors(panel["eligible"], FACTOR_HORIZON)
    monthly = monthly_factor_panel(model, alpha_panel, panel["regimes"])
    gate_history = compute_factor_gate_history(monthly, FACTOR_COLS)
    timing = v32.build_timing_panel(panel, BENCHMARK_ASSET)
    model.write_csv(alpha_panel, output_dir / "alpha_factory_daily_panel.csv")
    model.write_csv(monthly, output_dir / "alpha_factory_monthly_panel.csv")
    model.write_csv(gate_history, output_dir / "factor_gate_history.csv")
    model.write_csv(timing, output_dir / "market_beta_timing_panel.csv")

    # V3.3: governed alpha factory.
    v33_dir = output_dir / "v3_3"
    v33_dir.mkdir(parents=True, exist_ok=True)
    v33_targets = {}
    v33_results = {}
    v33_decisions = []
    for variant, cfg in V33_VARIANTS.items():
        targets, decisions = build_alpha_targets(model, monthly, gate_history, cfg, variant)
        v33_targets[variant] = targets
        v33_decisions.append(decisions)
        result = evaluate_targets(model, v30, wf, panel, targets, v33_dir / "candidates" / variant, variant, cfg["description"])
        v33_results[variant] = result
        model.write_csv(decisions, v33_dir / "candidates" / variant / "alpha_decision_log.csv")
    v33_selected = select_version(model, v30, wf, v33_dir, v33_results, v33_targets)
    v33_decisions_all = pd.concat(v33_decisions, ignore_index=True, sort=False)
    model.write_csv(v33_decisions_all, v33_dir / "alpha_decision_log.csv")
    gate_summary = (
        gate_history.groupby(["factor", "decision"], as_index=False)
        .agg(windows=("test_year", "count"), avg_multiplier=("multiplier", "mean"), avg_rank_ic=("rank_ic_mean", "mean"))
        .sort_values(["factor", "decision"])
    )
    v33_self = run_version_self_check(
        model,
        v33_selected["selected_targets"],
        v33_selected["selected_smoke"],
        v33_selected["selected_summary"],
        v33_selected["score_table"],
        v33_dir,
        "HIRSSM V3.3 Governed Alpha Factory",
        v33_selected["selected"],
        v33_selected["score_detail"],
        [
            "Added cross-sectional alpha factors from existing point-in-time index price, volume and historical style valuation features.",
            "Added rolling 5-year factor IC gates; gates only affect future test-year weights.",
            "No unavailable earnings-revision or current industry valuation snapshot is used in historical backtest.",
        ],
        extra_tables={"Factor Gate Summary": gate_summary},
    )

    # V3.4: alpha factory plus beta timing.
    v34_dir = output_dir / "v3_4"
    v34_dir.mkdir(parents=True, exist_ok=True)
    v34_targets = {}
    v34_results = {}
    v34_decisions = []
    for variant, cfg in V34_BETA_VARIANTS.items():
        targets, decisions = v32.overlay_beta_targets(v33_selected["selected_targets"], timing, cfg, variant)
        v34_targets[variant] = targets
        v34_decisions.append(decisions)
        result = evaluate_targets(model, v30, wf, panel, targets, v34_dir / "candidates" / variant, variant, cfg["description"])
        v34_results[variant] = result
        model.write_csv(decisions, v34_dir / "candidates" / variant / "beta_timing_decisions.csv")
    v34_selected = select_version(model, v30, wf, v34_dir, v34_results, v34_targets)
    v34_decisions_all = pd.concat(v34_decisions, ignore_index=True, sort=False)
    model.write_csv(v34_decisions_all, v34_dir / "beta_timing_decisions.csv")
    bucket_summary = (
        v34_decisions_all.groupby(["variant", "timing_bucket"], as_index=False)
        .agg(periods=("signal_date", "count"), avg_target_gross=("target_gross", "mean"), avg_cash=("cash", "mean"))
        .sort_values(["variant", "timing_bucket"])
    )
    v34_self = run_version_self_check(
        model,
        v34_selected["selected_targets"],
        v34_selected["selected_smoke"],
        v34_selected["selected_summary"],
        v34_selected["score_table"],
        v34_dir,
        "HIRSSM V3.4 Alpha Factory Plus Beta Timing",
        v34_selected["selected"],
        v34_selected["score_detail"],
        [
            f"Applied V3.2-style market beta timing to V3.3 selected alpha sleeve `{v33_selected['selected']}`.",
            "Target gross exposure is bounded by timing bucket and market state.",
            "The version tests whether alpha ranking and beta participation improve together after costs.",
        ],
        extra_tables={"Timing Bucket Summary": bucket_summary},
    )

    # V3.5: ensemble of V3.2 and V3.4 with caps and rebalance bands.
    v35_dir = output_dir / "v3_5"
    v35_dir.mkdir(parents=True, exist_ok=True)
    v32_targets = read_targets(Path(args.v32_targets))
    v35_targets = {}
    v35_results = {}
    for variant, cfg in V35_BLEND_VARIANTS.items():
        targets = blend_targets(v34_selected["selected_targets"], v32_targets, cfg, timing, variant)
        v35_targets[variant] = targets
        result = evaluate_targets(model, v30, wf, panel, targets, v35_dir / "candidates" / variant, variant, cfg["description"])
        v35_results[variant] = result
    v35_selected = select_version(model, v30, wf, v35_dir, v35_results, v35_targets)
    v35_self = run_version_self_check(
        model,
        v35_selected["selected_targets"],
        v35_selected["selected_smoke"],
        v35_selected["selected_summary"],
        v35_selected["score_table"],
        v35_dir,
        "HIRSSM V3.5 Robust Alpha-Beta Ensemble",
        v35_selected["selected"],
        v35_selected["score_detail"],
        [
            f"Blended V3.4 selected sleeve `{v34_selected['selected']}` with the existing V3.2 selected sleeve.",
            "Added single-asset caps and rebalance bands to reduce model and turnover risk.",
            "Kept no-leverage and non-negative cash constraints.",
        ],
    )

    selected_perf = pd.concat(
        [
            v33_selected["selected_summary"].assign(model_version="V3.3", selected_variant=v33_selected["selected"], self_check_pass=bool(v33_self["pass"].all())),
            v34_selected["selected_summary"].assign(model_version="V3.4", selected_variant=v34_selected["selected"], self_check_pass=bool(v34_self["pass"].all())),
            v35_selected["selected_summary"].assign(model_version="V3.5", selected_variant=v35_selected["selected"], self_check_pass=bool(v35_self["pass"].all())),
        ],
        ignore_index=True,
        sort=False,
    )
    model.write_csv(selected_perf, output_dir / "v3_3_to_v3_5_selected_performance.csv")
    manifest = {
        "generated_at": now_text(),
        "output_dir": str(output_dir),
        "v3_3_selected": v33_selected["selected"],
        "v3_3_self_check_pass": bool(v33_self["pass"].all()),
        "v3_4_selected": v34_selected["selected"],
        "v3_4_self_check_pass": bool(v34_self["pass"].all()),
        "v3_5_selected": v35_selected["selected"],
        "v3_5_self_check_pass": bool(v35_self["pass"].all()),
        "costs": COSTS,
        "benchmark": BENCHMARK_ASSET,
        "factor_horizon_days": FACTOR_HORIZON,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
