#!/usr/bin/env python
"""HIRSSM V2.0 industry rotation and size-style switching model.

This implementation intentionally enables only modules supported by the
current local data: index prices, turnover proxies, breadth, risk, and style
PE/PB spreads. Industry valuation snapshots and current components are not
used in historical backtests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
TRADING_DAYS = 252


STYLE_NAMES = {
    "000016": "SSE50",
    "000300": "CSI300",
    "000852": "CSI1000",
    "000905": "CSI500",
    "000922": "CSI_DIVIDEND",
    "000985": "CSI_ALL",
}


@dataclass
class RunPaths:
    root: Path
    output_dir: Path
    config_path: Path


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_code(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        std = x.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return ((x - x.mean(skipna=True)) / std).clip(-5, 5)
    return ((x - med) / (1.4826 * mad)).clip(-5, 5)


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str], group_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"{col}_z"] = out.groupby(group_col)[col].transform(robust_zscore)
    return out


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def mdd(values: np.ndarray) -> float:
        arr = pd.Series(values)
        peak = arr.cummax()
        dd = arr / peak - 1.0
        return float(dd.min())

    return close.rolling(window, min_periods=max(20, window // 3)).apply(mdd, raw=True)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-diff.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_one_daily(path: Path, asset_type: str, index_name: str | None = None) -> pd.DataFrame:
    code = normalize_code(path.stem)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"index_code": str})
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path} missing date/close")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "asset": code,
            "asset_type": asset_type,
            "index_name": index_name or str(df.get("index_name", pd.Series([code])).dropna().iloc[0] if "index_name" in df.columns and df["index_name"].notna().any() else code),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "open": pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else np.nan,
            "high": pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else np.nan,
            "low": pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else np.nan,
            "volume": pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else np.nan,
            "amount": pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns else np.nan,
        }
    )
    if "is_full_ohlc_bar" in df.columns:
        out["is_full_ohlc_bar"] = df["is_full_ohlc_bar"].astype(str).str.lower().isin({"true", "1"})
    else:
        out["is_full_ohlc_bar"] = out[["open", "high", "low", "close"]].notna().all(axis=1)
    return out.dropna(subset=["date", "close"]).sort_values(["asset", "date"])


def load_classification(root: Path, config: dict) -> pd.DataFrame:
    base = root / config["data_contract"]["industry_classification_path"]
    frames = []
    for file_name in ["sw_level1_info.csv", "sw_level2_info.csv"]:
        path = base / file_name
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig", dtype={"index_code": str})
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["index_code", "index_name", "sw_level", "parent_industry"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["index_code"] = out["index_code"].map(normalize_code)
    return out


def load_style_daily(root: Path, config: dict) -> pd.DataFrame:
    style_map = config["asset_universe"]["style"]
    wanted = set(normalize_code(v) for v in style_map.values())
    base = root / config["data_contract"]["style_daily_path"]
    frames = []
    for code in sorted(wanted):
        path = base / f"{code}.csv"
        if path.exists():
            frames.append(load_one_daily(path, "style", STYLE_NAMES.get(code, code)))
    if not frames:
        raise FileNotFoundError(f"no style daily files under {base}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_industry_daily(root: Path, config: dict) -> pd.DataFrame:
    base = root / config["data_contract"]["industry_daily_path"]
    classification = load_classification(root, config)
    level1 = classification[classification.get("sw_level").eq("first")]["index_code"].dropna().astype(str).tolist()
    enhanced = [normalize_code(code) for code in config["asset_universe"]["industry"].get("enhanced_level2", [])]
    wanted = list(dict.fromkeys(level1 + enhanced))
    meta = classification.set_index("index_code").to_dict("index")
    frames = []
    for code in wanted:
        path = base / f"{code}.csv"
        if not path.exists():
            continue
        name = meta.get(code, {}).get("index_name", code)
        df = load_one_daily(path, "industry", str(name))
        df["sw_level"] = meta.get(code, {}).get("sw_level", "")
        df["parent_industry"] = meta.get(code, {}).get("parent_industry", "")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no industry daily files under {base}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_style_valuation(root: Path, config: dict, style_codes: Iterable[str]) -> pd.DataFrame:
    pe_base = root / config["data_contract"]["style_pe_path"]
    pb_base = root / config["data_contract"]["style_pb_path"]
    frames = []
    for raw_code in style_codes:
        code = normalize_code(raw_code)
        pe_path = pe_base / f"{code}.csv"
        pb_path = pb_base / f"{code}.csv"
        parts = []
        if pe_path.exists():
            pe = pd.read_csv(pe_path, encoding="utf-8-sig", dtype={"index_code": str})
            pe = pe[["date", "pe_ttm"]].copy() if "pe_ttm" in pe.columns else pe[["date"]].copy()
            parts.append(pe)
        if pb_path.exists():
            pb = pd.read_csv(pb_path, encoding="utf-8-sig", dtype={"index_code": str})
            pb = pb[["date", "pb"]].copy() if "pb" in pb.columns else pb[["date"]].copy()
            parts.append(pb)
        if not parts:
            continue
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on="date", how="outer")
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged["asset"] = code
        frames.append(merged.sort_values("date"))
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "pe_ttm", "pb"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["pe_ttm", "pb"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_pctile"] = out.groupby("asset")[col].transform(expanding_percentile)
    return out


def expanding_percentile(series: pd.Series, min_periods: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = []
    seen: list[float] = []
    for value in values:
        if pd.isna(value):
            out.append(np.nan)
            continue
        seen.append(float(value))
        if len(seen) < min_periods:
            out.append(np.nan)
            continue
        arr = np.asarray(seen, dtype=float)
        out.append(float((arr <= value).mean()))
    return pd.Series(out, index=series.index)


def add_features(panel: pd.DataFrame, market_returns: pd.Series | None = None) -> pd.DataFrame:
    out = panel.sort_values(["asset", "date"]).copy()
    g = out.groupby("asset", group_keys=False)
    out["ret_1d"] = g["close"].pct_change()
    for window in [5, 10, 20, 60, 120, 250]:
        out[f"ret_{window}"] = g["close"].pct_change(window)
    for window in [20, 60, 120, 200]:
        ma = g["close"].transform(lambda s, w=window: s.rolling(w, min_periods=max(10, w // 3)).mean())
        out[f"ma_{window}"] = ma
        out[f"ma_gap_{window}"] = out["close"] / ma - 1
    out["ma_slope_60"] = g["ma_60"].pct_change(20)
    out["ma_slope_120"] = g["ma_120"].pct_change(20)
    rolling_high_120 = g["close"].transform(lambda s: s.rolling(120, min_periods=60).max())
    rolling_low_60 = g["close"].transform(lambda s: s.rolling(60, min_periods=30).min())
    out["breakout_120"] = out["close"] / rolling_high_120 - 1
    out["drawup_from_60d_low"] = out["close"] / rolling_low_60 - 1
    out["distance_to_ma20"] = out["ma_gap_20"]
    out["distance_to_ma60"] = out["ma_gap_60"]
    out["rsi_14"] = g["close"].transform(lambda s: rsi(s, 14))
    out["vol_20"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=12).std(ddof=1) * np.sqrt(TRADING_DAYS))
    out["vol_60"] = g["ret_1d"].transform(lambda s: s.rolling(60, min_periods=30).std(ddof=1) * np.sqrt(TRADING_DAYS))
    out["downside_vol_60"] = g["ret_1d"].transform(lambda s: s.where(s < 0).rolling(60, min_periods=20).std(ddof=1) * np.sqrt(TRADING_DAYS))
    out["max_drawdown_120"] = g["close"].transform(lambda s: rolling_max_drawdown(s, 120))
    out["amount_zscore_60"] = g["amount"].transform(lambda s: (s - s.rolling(60, min_periods=30).mean()) / s.rolling(60, min_periods=30).std(ddof=0))
    out["volume_zscore_60"] = g["volume"].transform(lambda s: (s - s.rolling(60, min_periods=30).mean()) / s.rolling(60, min_periods=30).std(ddof=0))
    out["price_volume_confirmation"] = np.sign(out["ret_60"].fillna(0)) * out["amount_zscore_60"].clip(-3, 3)
    out["crowding_score"] = (out["ret_60"].rank(pct=True) + out["amount_zscore_60"].rank(pct=True)) / 2
    if market_returns is not None:
        out = out.merge(market_returns.rename("market_ret_1d"), left_on="date", right_index=True, how="left")
        out["excess_ret_20"] = out["ret_20"] - out.groupby("asset")["market_ret_1d"].transform(lambda s: (1 + s).rolling(20, min_periods=12).apply(np.prod, raw=True) - 1)
        out["excess_ret_60"] = out["ret_60"] - out.groupby("asset")["market_ret_1d"].transform(lambda s: (1 + s).rolling(60, min_periods=30).apply(np.prod, raw=True) - 1)
        out["excess_ret_120"] = out["ret_120"] - out.groupby("asset")["market_ret_1d"].transform(lambda s: (1 + s).rolling(120, min_periods=60).apply(np.prod, raw=True) - 1)
        out["corr_to_market_120"] = out.groupby("asset", group_keys=False).apply(lambda x: x["ret_1d"].rolling(120, min_periods=60).corr(x["market_ret_1d"]))
        out["residual_momentum_60"] = out["excess_ret_60"]
        out["residual_momentum_120"] = out["excess_ret_120"]
    return out


def add_valuation_scores(style: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    if valuation.empty:
        style["valuation_score"] = 0.0
        return style
    out = style.merge(valuation[["date", "asset", "pe_ttm_pctile", "pb_pctile"]], on=["date", "asset"], how="left")
    out["valuation_score"] = -out[["pe_ttm_pctile", "pb_pctile"]].mean(axis=1)
    confirm = ((out["ma_gap_60"] > -0.03) | (out["vol_60"] < out.groupby("asset")["vol_60"].transform(lambda s: s.rolling(252, min_periods=80).median())))
    out.loc[~confirm, "valuation_score"] = 0.0
    return out


def score_assets(panel: pd.DataFrame, is_style: bool) -> pd.DataFrame:
    out = panel.copy()
    trend_cols = ["ret_60", "ret_120", "ma_gap_120", "ma_slope_60", "breakout_120"]
    rel_cols = ["excess_ret_60", "excess_ret_120", "residual_momentum_60"]
    risk_cols = ["vol_60", "downside_vol_60", "max_drawdown_120", "corr_to_market_120"]
    reversal_cols = ["ret_20", "distance_to_ma20", "distance_to_ma60", "rsi_14"]
    liquidity_cols = ["price_volume_confirmation", "amount_zscore_60"]

    for col in trend_cols + rel_cols + risk_cols + reversal_cols + liquidity_cols + ["valuation_score"]:
        if col not in out.columns:
            out[col] = np.nan

    out = cross_sectional_zscore(out, trend_cols + rel_cols + risk_cols + reversal_cols + liquidity_cols + ["valuation_score"])
    out["trend_expert_score"] = out[[f"{c}_z" for c in trend_cols if f"{c}_z" in out.columns]].mean(axis=1)
    out["relative_strength_score"] = out[[f"{c}_z" for c in rel_cols if f"{c}_z" in out.columns]].mean(axis=1)
    out["valuation_repair_score"] = out["valuation_score_z"].fillna(0.0) if "valuation_score_z" in out.columns else 0.0
    risk_z_cols = [f"{c}_z" for c in risk_cols if f"{c}_z" in out.columns]
    out["risk_compression_score"] = -out[risk_z_cols].mean(axis=1)
    rev_z_cols = [f"{c}_z" for c in reversal_cols if f"{c}_z" in out.columns]
    liq_z_cols = [f"{c}_z" for c in liquidity_cols if f"{c}_z" in out.columns]
    out["liquidity_score"] = out[liq_z_cols].mean(axis=1)
    raw_reversal = -out[rev_z_cols].mean(axis=1)
    stabilization_gate = (
        ((out["ma_gap_60"] > -0.08) & (out["ret_20"] > -0.10))
        | ((out["max_drawdown_120"] < -0.15) & (out["ret_5"] > 0))
        | (out["risk_compression_score"] > 0.5)
    )
    out["range_reversal_score"] = raw_reversal.where(stabilization_gate.fillna(False), raw_reversal * 0.25)
    out["defensive_score"] = out["risk_compression_score"].fillna(0.0)
    if is_style:
        defensive_assets = {"000016", "000922"}
        out.loc[out["asset"].isin(defensive_assets), "defensive_score"] = out.loc[out["asset"].isin(defensive_assets), "defensive_score"] + 0.5
    else:
        defensive_industries = {"801780", "801160", "801120", "801110"}
        out.loc[out["asset"].isin(defensive_industries), "defensive_score"] = out.loc[out["asset"].isin(defensive_industries), "defensive_score"] + 0.4
    return out


def compute_breadth(industry: pd.DataFrame) -> pd.DataFrame:
    primary = industry[industry.get("sw_level", "").eq("first")].copy() if "sw_level" in industry.columns else industry.copy()
    daily = primary.groupby("date").agg(
        industry_above_ma60_ratio=("ma_gap_60", lambda s: float((s > 0).mean())),
        industry_positive_ret20_ratio=("ret_20", lambda s: float((s > 0).mean())),
        industry_return_dispersion=("ret_20", lambda s: float(pd.to_numeric(s, errors="coerce").std(skipna=True))),
    )
    return daily.reset_index()


def assign_regime(style: pd.DataFrame, industry: pd.DataFrame, config: dict) -> pd.DataFrame:
    broad_code = normalize_code(config["asset_universe"]["style"].get("broad_market", "000985"))
    market = style[style["asset"].eq(broad_code)].copy()
    if market.empty:
        market = style[style["asset"].eq(normalize_code(config["asset_universe"]["style"].get("large_cap", "000300")))].copy()
    breadth = compute_breadth(industry)
    market = market.merge(breadth, on="date", how="left")
    market["vol60_median_252"] = market["vol_60"].rolling(252, min_periods=80).median()
    raw_state = []
    for row in market.to_dict("records"):
        ret20 = row.get("ret_20", np.nan)
        ret60 = row.get("ret_60", np.nan)
        ret120 = row.get("ret_120", np.nan)
        ma200 = row.get("ma_gap_200", np.nan)
        mdd = row.get("max_drawdown_120", np.nan)
        breadth_ma = row.get("industry_above_ma60_ratio", np.nan)
        vol = row.get("vol_60", np.nan)
        vol_med = row.get("vol60_median_252", np.nan)
        if pd.notna(ma200) and pd.notna(ret60) and ma200 < 0 and ret60 < 0 and pd.notna(breadth_ma) and breadth_ma < 0.35:
            state = "risk_off_decline"
        elif pd.notna(mdd) and mdd < -0.15 and pd.notna(ret20) and ret20 > 0:
            state = "crash_rebound"
        elif pd.notna(ret120) and pd.notna(ma200) and ret120 > 0 and ma200 > 0 and pd.notna(breadth_ma) and breadth_ma > 0.6:
            state = "risk_on_overheat" if pd.notna(ret20) and ret20 > 0.12 and pd.notna(vol) and pd.notna(vol_med) and vol > vol_med else "risk_on_trend"
        else:
            state = "range_bound"
        raw_state.append(state)
    market["raw_state"] = raw_state
    smooth_days = int(config.get("regime_model", {}).get("state_smoothing_days", 5))
    states = []
    current = None
    candidate = None
    candidate_count = 0
    for state in market["raw_state"]:
        if current is None:
            current = state
        elif state != current:
            if state == candidate:
                candidate_count += 1
            else:
                candidate = state
                candidate_count = 1
            if candidate_count >= smooth_days:
                current = candidate
                candidate = None
                candidate_count = 0
        else:
            candidate = None
            candidate_count = 0
        states.append(current)
    market["state"] = states
    return market[["date", "state", "raw_state", "industry_above_ma60_ratio", "industry_positive_ret20_ratio", "industry_return_dispersion"]]


def month_end_dates(dates: pd.Series) -> list[pd.Timestamp]:
    clean = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    return list(clean.groupby(clean.dt.to_period("M")).max())


def apply_caps(weights: dict[str, float], caps: dict[str, float]) -> dict[str, float]:
    return {asset: min(weight, caps.get(asset, 1.0)) for asset, weight in weights.items() if weight > 0}


def add_weights(base: dict[str, float], addition: dict[str, float]) -> None:
    for asset, weight in addition.items():
        if weight > 0:
            base[asset] = base.get(asset, 0.0) + float(weight)


def cap_and_redistribute(weights: dict[str, float], caps: dict[str, float]) -> dict[str, float]:
    """Apply asset caps while preserving the intended non-cash budget when possible."""
    raw = {asset: max(0.0, float(weight)) for asset, weight in weights.items() if weight > 0}
    if not raw:
        return {}
    target_total = sum(raw.values())
    total_capacity = sum(max(0.0, float(caps.get(asset, 1.0))) for asset in raw)
    target_total = min(target_total, total_capacity)
    remaining_assets = dict(raw)
    remaining_budget = target_total
    capped: dict[str, float] = {}

    for _ in range(len(raw) + 1):
        if not remaining_assets or remaining_budget <= 0:
            break
        raw_sum = sum(remaining_assets.values())
        if raw_sum <= 0:
            equal = remaining_budget / len(remaining_assets)
            proposed = {asset: equal for asset in remaining_assets}
        else:
            proposed = {asset: remaining_budget * weight / raw_sum for asset, weight in remaining_assets.items()}
        over_cap = {
            asset: min(float(caps.get(asset, 1.0)), remaining_budget)
            for asset, weight in proposed.items()
            if weight > float(caps.get(asset, 1.0)) + 1e-12
        }
        if not over_cap:
            capped.update(proposed)
            break
        capped.update(over_cap)
        used = sum(over_cap.values())
        remaining_budget = max(0.0, remaining_budget - used)
        remaining_assets = {asset: weight for asset, weight in remaining_assets.items() if asset not in over_cap}

    return {asset: weight for asset, weight in capped.items() if weight > 1e-10}


def normalize_to_budget(raw: pd.Series, budget: float, vol: pd.Series | None = None) -> dict[str, float]:
    scores = pd.to_numeric(raw, errors="coerce").dropna()
    scores = scores[scores > scores.quantile(0.2)] if scores.shape[0] > 3 else scores
    if scores.empty or budget <= 0:
        return {}
    ranks = scores.rank(method="first")
    base = ranks / ranks.sum()
    if vol is not None:
        adj_vol = pd.to_numeric(vol.reindex(base.index), errors="coerce").replace(0, np.nan)
        inv_vol = 1.0 / adj_vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
        base = base * inv_vol
    base = base / base.sum()
    return {asset: float(weight * budget) for asset, weight in base.items()}


def select_top(scores: pd.DataFrame, score_col: str, max_n: int, min_n: int) -> pd.DataFrame:
    available = scores.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    positive = available[available[score_col] > 0]
    selected = positive.head(max_n)
    if selected.shape[0] < min_n:
        selected = available.head(min(max_n, max(min_n, available.shape[0])))
    return selected


def build_targets(
    scored: pd.DataFrame,
    regimes: pd.DataFrame,
    config: dict,
    start_date: pd.Timestamp | None = None,
    disabled_experts: set[str] | None = None,
    disabled_experts_by_year: dict[int, set[str]] | None = None,
    disabled_experts_by_year_state: dict[tuple[int, str], set[str]] | None = None,
    expert_multipliers_by_year_state: dict[tuple[int, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    panel = scored.merge(regimes[["date", "state"]], on="date", how="left")
    rebalance_dates = month_end_dates(panel["date"])
    if start_date is not None:
        rebalance_dates = [d for d in rebalance_dates if d >= start_date]
    budgets_by_state = config["portfolio"]["sleeve_budget_by_state"]
    priors = config["expert_state_priors"]
    constraints = config["portfolio"]["constraints"]
    cash_substitution = config["portfolio"].get("cash_substitution", {})
    rows = []
    prev_weights: dict[str, float] = {}
    base_disabled = disabled_experts or set()
    year_disabled_map = disabled_experts_by_year or {}
    year_state_disabled_map = disabled_experts_by_year_state or {}
    year_state_multiplier_map = expert_multipliers_by_year_state or {}

    for signal_date in rebalance_dates:
        day = panel[panel["date"].eq(signal_date)].copy()
        if day.empty:
            continue
        state = str(day["state"].dropna().iloc[0]) if day["state"].notna().any() else "range_bound"
        budgets = budgets_by_state.get(state, budgets_by_state["range_bound"])
        prior = priors.get(state, priors["range_bound"])
        day["trend_state_score"] = day[["trend_expert_score", "relative_strength_score"]].mean(axis=1)
        asset_type = day["asset_type"].astype(str)
        signal_year = int(signal_date.year)
        disabled = (
            set(base_disabled)
            | set(year_disabled_map.get(signal_year, set()))
            | set(year_state_disabled_map.get((signal_year, state), set()))
        )
        multipliers = year_state_multiplier_map.get((signal_year, state), {})

        def expert_weight_series(expert: str, default_weight: float) -> pd.Series:
            if expert in disabled:
                return pd.Series(0.0, index=day.index)
            return asset_type.map(
                lambda item: 0.0
                if f"{item}_{expert}" in disabled
                else float(default_weight) * max(0.0, float(multipliers.get(f"{item}_{expert}", multipliers.get(expert, 1.0))))
            )

        trend_weight = expert_weight_series("trend_continuation", prior.get("trend_continuation", 0))
        valuation_weight = expert_weight_series("valuation_repair", prior.get("valuation_repair", 0))
        risk_weight = expert_weight_series("risk_compression", prior.get("risk_compression", 0))
        reversal_weight = expert_weight_series("range_reversal", prior.get("range_reversal", 0))
        defensive_weight = expert_weight_series("defensive", prior.get("defensive", 0))
        liquidity_weight = expert_weight_series("liquidity_overlay", 0.05)
        day["final_score"] = (
            trend_weight * day["trend_state_score"].fillna(0)
            + valuation_weight * day["valuation_repair_score"].fillna(0)
            + risk_weight * day["risk_compression_score"].fillna(0)
            + reversal_weight * day["range_reversal_score"].fillna(0)
            + defensive_weight * day["defensive_score"].fillna(0)
            + liquidity_weight * day["liquidity_score"].fillna(0)
        )
        day["confidence"] = day[["trend_expert_score", "risk_compression_score", "range_reversal_score"]].apply(lambda x: 1.0 / (1.0 + x.std(skipna=True)), axis=1)
        day["risk_adjusted_alpha"] = day["final_score"].fillna(0) * day["confidence"].fillna(0.5) / day["vol_60"].replace(0, np.nan)

        target: dict[str, float] = {}
        style_day = day[day["asset_type"].eq("style")].set_index("asset")
        industry_day = day[day["asset_type"].eq("industry")].copy()

        style_candidates = select_top(style_day.reset_index(), "risk_adjusted_alpha", max_n=2, min_n=1).set_index("asset")
        add_weights(target, normalize_to_budget(style_candidates["risk_adjusted_alpha"], budgets.get("style", 0), style_candidates["vol_60"]))

        industry_candidates = select_top(industry_day, "risk_adjusted_alpha", constraints["max_industry_holdings"], constraints["min_industry_holdings"])
        if state == "risk_off_decline":
            industry_candidates = industry_candidates.head(max(1, min(2, industry_candidates.shape[0])))
        add_weights(target, normalize_to_budget(industry_candidates.set_index("asset")["risk_adjusted_alpha"], budgets.get("industry", 0), industry_candidates.set_index("asset")["vol_60"]))

        defensive_assets = ["000016", "000922"]
        defensive_available = style_day.loc[style_day.index.intersection(defensive_assets)]
        if not defensive_available.empty and cash_substitution.get("enabled", False):
            min_final_score = float(cash_substitution.get("defensive_min_final_score", 0.0))
            min_risk_score = float(cash_substitution.get("defensive_min_risk_compression_score", 0.0))
            defensive_available = defensive_available[
                (defensive_available["final_score"].fillna(-np.inf) >= min_final_score)
                | (defensive_available["risk_compression_score"].fillna(-np.inf) >= min_risk_score)
            ]
        if not defensive_available.empty:
            add_weights(target, normalize_to_budget(defensive_available["defensive_score"], budgets.get("defensive", 0), defensive_available["vol_60"]))

        caps = {}
        for asset in target:
            if asset in style_day.index:
                caps[asset] = float(constraints["max_single_style_weight"])
            elif asset in set(industry_day.loc[industry_day.get("sw_level", "").eq("second"), "asset"]):
                caps[asset] = float(constraints["max_single_level2_industry_weight"])
            else:
                caps[asset] = float(constraints["max_single_level1_industry_weight"])
        target = cap_and_redistribute(target, caps)

        min_change = float(config["rebalance"].get("min_weight_change_to_trade", 0.0))
        adjusted = {}
        for asset in (set(prev_weights) | set(target)) - {"CASH"}:
            old = prev_weights.get(asset, 0.0)
            new = target.get(asset, 0.0)
            adjusted[asset] = old if abs(new - old) < min_change else new
        noncash_sum = sum(max(v, 0.0) for v in adjusted.values())
        if noncash_sum > 1:
            adjusted = {asset: value / noncash_sum for asset, value in adjusted.items()}
            noncash_sum = 1.0
        adjusted["CASH"] = max(0.0, 1.0 - noncash_sum)
        turnover = sum(abs(adjusted.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in set(adjusted) | set(prev_weights))
        for asset, weight in adjusted.items():
            if weight <= 0:
                continue
            asset_row = day[day["asset"].eq(asset)].head(1)
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": asset,
                    "weight": weight,
                    "state": state,
                    "asset_type": "cash" if asset == "CASH" else (asset_row["asset_type"].iloc[0] if not asset_row.empty else ""),
                    "score": float(asset_row["final_score"].iloc[0]) if not asset_row.empty else 0.0,
                    "risk_adjusted_alpha": float(asset_row["risk_adjusted_alpha"].iloc[0]) if not asset_row.empty else 0.0,
                    "turnover": turnover,
                }
            )
        prev_weights = adjusted
    return pd.DataFrame(rows)


def build_trade_schedule(targets: pd.DataFrame, all_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(all_dates).dropna().unique()).sort_values().reset_index(drop=True)
    rows = []
    for signal_date, group in targets.groupby("signal_date"):
        future = dates[dates > pd.to_datetime(signal_date)]
        if future.empty:
            continue
        trade_date = future.iloc[0]
        for _, item in group.iterrows():
            rows.append({**item.to_dict(), "trade_date": trade_date})
    return pd.DataFrame(rows)


def run_backtest(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    cost_bps: float,
    benchmark_asset: str,
) -> dict[str, pd.DataFrame]:
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret_wide = ret.pivot(index="date", columns="asset", values="ret_1d").sort_index().fillna(0.0)
    schedule = build_trade_schedule(targets, ret_wide.index.to_series())
    if not schedule.empty:
        first_trade_date = pd.to_datetime(schedule["trade_date"]).min()
        ret_wide = ret_wide[ret_wide.index >= first_trade_date]
    schedule_map = {date: group for date, group in schedule.groupby("trade_date")} if not schedule.empty else {}
    current_weights: dict[str, float] = {"CASH": 1.0}
    nav = 1.0
    rows = []
    trades = []
    for date, day_returns in ret_wide.iterrows():
        gross_ret = 0.0
        for asset, weight in current_weights.items():
            if asset == "CASH":
                continue
            gross_ret += weight * float(day_returns.get(asset, 0.0))
        cost = 0.0
        turnover = 0.0
        if date in schedule_map:
            new_weights = {str(row["asset"]): float(row["weight"]) for _, row in schedule_map[date].iterrows()}
            turnover = sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in set(new_weights) | set(current_weights))
            cost = turnover * cost_bps / 10000.0
            for asset, weight in new_weights.items():
                trades.append({"date": date, "asset": asset, "weight": weight, "cost_bps": cost_bps, "turnover": turnover})
            current_weights = new_weights
        net_ret = gross_ret - cost
        nav *= 1 + net_ret
        rows.append(
            {
                "date": date,
                "gross_return": gross_ret,
                "cost": cost,
                "portfolio_return": net_ret,
                "nav": nav,
                "turnover": turnover,
                "cash_weight": current_weights.get("CASH", 0.0),
                "benchmark_return": float(day_returns.get(benchmark_asset, 0.0)),
            }
        )
    nav_df = pd.DataFrame(rows)
    if not nav_df.empty:
        nav_df["benchmark_nav"] = (1 + nav_df["benchmark_return"]).cumprod()
    return {"nav": nav_df, "trades": pd.DataFrame(trades), "schedule": schedule}


def summarize_nav(nav: pd.DataFrame) -> pd.DataFrame:
    df = nav.dropna(subset=["portfolio_return"]).copy()
    if df.empty:
        return pd.DataFrame()
    years = max((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25, 1 / TRADING_DAYS)
    total = df["nav"].iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1
    vol = df["portfolio_return"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    bench_total = df["benchmark_nav"].iloc[-1] - 1
    excess = df["portfolio_return"] - df["benchmark_return"]
    te = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    nav_path = pd.concat([pd.Series([1.0]), df["nav"].reset_index(drop=True)], ignore_index=True)
    benchmark_path = pd.concat([pd.Series([1.0]), df["benchmark_nav"].reset_index(drop=True)], ignore_index=True)
    mdd = float((nav_path / nav_path.cummax() - 1).min())
    bench_mdd = float((benchmark_path / benchmark_path.cummax() - 1).min())
    bench_ann = (1 + bench_total) ** (1 / years) - 1
    bench_vol = df["benchmark_return"].std(ddof=1) * np.sqrt(TRADING_DAYS)
    traded_turnover = df.loc[df["turnover"] > 0, "turnover"]
    return pd.DataFrame(
        [
            {
                "total_return": total,
                "annual_return": ann,
                "annual_vol": vol,
                "sharpe_no_rf": ann / vol if vol else np.nan,
                "max_drawdown": mdd,
                "calmar": ann / abs(mdd) if mdd else np.nan,
                "win_rate": float((df["portfolio_return"] > 0).mean()),
                "avg_turnover": float(df["turnover"].mean()),
                "avg_trade_turnover": float(traded_turnover.mean()) if not traded_turnover.empty else 0.0,
                "trade_count": int(traded_turnover.shape[0]),
                "total_cost": float(df["cost"].sum()),
                "avg_cash_weight": float(df["cash_weight"].mean()),
                "avg_gross_exposure": float((1.0 - df["cash_weight"]).mean()),
                "benchmark_total_return": bench_total,
                "benchmark_annual_return": bench_ann,
                "benchmark_annual_vol": bench_vol,
                "benchmark_sharpe_no_rf": bench_ann / bench_vol if bench_vol else np.nan,
                "benchmark_max_drawdown": bench_mdd,
                "excess_annual_mean": float(excess.mean() * TRADING_DAYS),
                "tracking_error": te,
                "information_ratio": float(excess.mean() * TRADING_DAYS / te) if te else np.nan,
            }
        ]
    )


def yearly_returns(nav: pd.DataFrame) -> pd.DataFrame:
    df = nav.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    rows = []
    for year, group in df.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float((1 + group["portfolio_return"]).prod() - 1),
                "benchmark_return": float((1 + group["benchmark_return"]).prod() - 1),
            }
        )
    return pd.DataFrame(rows)


def regime_returns(nav: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    df = nav.merge(regimes[["date", "state"]], on="date", how="left")
    rows = []
    for state, group in df.groupby("state", dropna=False):
        if group.empty:
            continue
        rows.append(
            {
                "state": state,
                "days": int(group.shape[0]),
                "avg_daily_return": float(group["portfolio_return"].mean()),
                "annualized_mean": float(group["portfolio_return"].mean() * TRADING_DAYS),
                "annualized_vol": float(group["portfolio_return"].std(ddof=1) * np.sqrt(TRADING_DAYS)),
                "win_rate": float((group["portfolio_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def target_exposure_summary(targets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if targets.empty:
        empty = pd.DataFrame()
        return {"monthly": empty, "by_state": empty}
    monthly = targets.pivot_table(
        index=["signal_date", "state"],
        columns="asset_type",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for col in ["style", "industry", "cash"]:
        if col not in monthly.columns:
            monthly[col] = 0.0
    monthly["gross_exposure"] = 1.0 - monthly["cash"]
    by_state = monthly.groupby("state", dropna=False).agg(
        observations=("signal_date", "count"),
        avg_style_weight=("style", "mean"),
        avg_industry_weight=("industry", "mean"),
        avg_cash_weight=("cash", "mean"),
        avg_gross_exposure=("gross_exposure", "mean"),
    ).reset_index()
    return {"monthly": monthly, "by_state": by_state}


def spearman_ic(group: pd.DataFrame, score_col: str, return_col: str) -> float:
    x = pd.to_numeric(group[score_col], errors="coerce")
    y = pd.to_numeric(group[return_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return np.nan
    xr = x[valid].rank()
    yr = y[valid].rank()
    if xr.nunique(dropna=True) <= 1 or yr.nunique(dropna=True) <= 1:
        return np.nan
    return float(xr.corr(yr))


def expert_rank_ic_report(scored: pd.DataFrame, regimes: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    score_cols = [
        "trend_expert_score",
        "relative_strength_score",
        "valuation_repair_score",
        "risk_compression_score",
        "range_reversal_score",
        "defensive_score",
        "liquidity_score",
    ]
    available_cols = [col for col in score_cols if col in scored.columns]
    if not available_cols:
        return pd.DataFrame()
    panel = scored.sort_values(["asset", "date"]).copy()
    panel[f"fwd_ret_{horizon}"] = panel.groupby("asset")["close"].shift(-horizon) / panel["close"] - 1
    signal_dates = month_end_dates(panel["date"])
    panel = panel[panel["date"].isin(signal_dates)].merge(regimes[["date", "state"]], on="date", how="left")
    rows = []
    return_col = f"fwd_ret_{horizon}"
    for asset_type, type_group in panel.groupby("asset_type"):
        for score_col in available_cols:
            daily_ics = []
            for signal_date, day in type_group.groupby("date"):
                ic = spearman_ic(day, score_col, return_col)
                if pd.notna(ic):
                    daily_ics.append({"date": signal_date, "ic": ic})
            ic_df = pd.DataFrame(daily_ics)
            if ic_df.empty:
                continue
            rows.append(
                {
                    "asset_type": asset_type,
                    "expert": score_col,
                    "horizon_days": horizon,
                    "observations": int(ic_df.shape[0]),
                    "rank_ic_mean": float(ic_df["ic"].mean()),
                    "rank_ic_std": float(ic_df["ic"].std(ddof=1)),
                    "rank_icir": float(ic_df["ic"].mean() / ic_df["ic"].std(ddof=1)) if ic_df["ic"].std(ddof=1) else np.nan,
                    "positive_ic_rate": float((ic_df["ic"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def ablation_report(
    scored: pd.DataFrame,
    regimes: pd.DataFrame,
    config: dict,
    returns: pd.DataFrame,
    benchmark_asset: str,
    start_date: pd.Timestamp,
    cost_bps: float = 10.0,
    base_disabled_experts: set[str] | None = None,
) -> pd.DataFrame:
    base_disabled = base_disabled_experts or set()
    candidates = [
        "trend_continuation",
        "valuation_repair",
        "risk_compression",
        "range_reversal",
        "defensive",
        "liquidity_overlay",
    ]
    candidates = [expert for expert in candidates if expert not in base_disabled]
    base_targets = build_targets(scored, regimes, config, start_date=start_date, disabled_experts=base_disabled)
    base_nav = run_backtest(returns, base_targets, cost_bps, benchmark_asset)["nav"]
    base_summary = summarize_nav(base_nav)
    if base_summary.empty:
        return pd.DataFrame()
    base = base_summary.iloc[0]
    rows = [
        {
            "variant": "base",
            "disabled_expert": ",".join(sorted(base_disabled)),
            "target_rows": int(base_targets.shape[0]),
            "annual_return": float(base["annual_return"]),
            "annual_vol": float(base["annual_vol"]),
            "sharpe_no_rf": float(base["sharpe_no_rf"]),
            "max_drawdown": float(base["max_drawdown"]),
            "avg_cash_weight": float(base["avg_cash_weight"]),
            "delta_annual_return_vs_base": 0.0,
            "delta_sharpe_vs_base": 0.0,
        }
    ]
    for expert in candidates:
        variant_disabled = set(base_disabled) | {expert}
        variant_targets = build_targets(scored, regimes, config, start_date=start_date, disabled_experts=variant_disabled)
        variant_nav = run_backtest(returns, variant_targets, cost_bps, benchmark_asset)["nav"]
        summary = summarize_nav(variant_nav)
        if summary.empty:
            continue
        item = summary.iloc[0]
        rows.append(
            {
                "variant": f"without_{expert}",
                "disabled_expert": ",".join(sorted(variant_disabled)),
                "target_rows": int(variant_targets.shape[0]),
                "annual_return": float(item["annual_return"]),
                "annual_vol": float(item["annual_vol"]),
                "sharpe_no_rf": float(item["sharpe_no_rf"]),
                "max_drawdown": float(item["max_drawdown"]),
                "avg_cash_weight": float(item["avg_cash_weight"]),
                "delta_annual_return_vs_base": float(item["annual_return"] - base["annual_return"]),
                "delta_sharpe_vs_base": float(item["sharpe_no_rf"] - base["sharpe_no_rf"]),
            }
        )
    return pd.DataFrame(rows)


def make_report(
    output_dir: Path,
    summaries: pd.DataFrame,
    latest_targets: pd.DataFrame,
    config: dict,
    expert_ic: pd.DataFrame,
    ablation: pd.DataFrame,
    exposure_by_state: pd.DataFrame,
    disabled_experts: set[str],
) -> None:
    best = summaries.sort_values("cost_bps").head(1)
    lines = [
        "# HIRSSM V2.0 Model Run Report",
        "",
        f"Run time: {now_text()}",
        "",
        "## Scope",
        "",
        "- Enabled: price/volume features, rule regimes, expert scores, style PE/PB spread, hierarchical budgets, monthly rebalance.",
        "- Disabled: industry historical valuation, component aggregation, stock-level size factor, macro beta, ML ranker.",
        f"- Disabled experts by governance: `{', '.join(sorted(disabled_experts)) if disabled_experts else 'none'}`.",
        "- Data safety: current industry snapshots and current components are not used in historical backtests.",
        "",
        "## Cost Sensitivity",
        "",
        summaries.to_markdown(index=False),
        "",
        "## Latest Target Weights",
        "",
        latest_targets.sort_values("weight", ascending=False).head(20).to_markdown(index=False) if not latest_targets.empty else "No latest target.",
        "",
        "## Exposure By State",
        "",
        exposure_by_state.to_markdown(index=False) if not exposure_by_state.empty else "No exposure summary.",
        "",
        "## Expert RankIC",
        "",
        expert_ic.sort_values(["asset_type", "rank_ic_mean"], ascending=[True, False]).to_markdown(index=False) if not expert_ic.empty else "No RankIC report.",
        "",
        "## Expert Ablation",
        "",
        ablation.sort_values("delta_annual_return_vs_base", ascending=False).to_markdown(index=False) if not ablation.empty else "No ablation report.",
        "",
        "## Output Files",
        "",
        "- `target_weights_monthly.csv`: monthly target portfolio weights.",
        "- `latest_target_weights.csv`: latest model target weights.",
        "- `cost_sensitivity_summary.csv`: cost scenario performance.",
        "- `expert_rank_ic.csv`: cross-sectional expert RankIC validation.",
        "- `expert_ablation_summary.csv`: 10bps cost ablation backtest.",
        "- `monthly_target_exposure.csv` and `target_exposure_by_state.csv`: sleeve exposure diagnostics.",
        "",
        "## Governance",
        "",
        f"- Config: `{CONFIG}`",
        f"- Rebalance: `{config['rebalance']['main_frequency']}`",
        f"- Construction: `{config['portfolio']['construction_stage']}`",
        "- Research status: backtest prototype, not live trading authorization.",
    ]
    (output_dir / "HIRSSM_V2_MODEL_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_model(
    paths: RunPaths,
    start_date: str | None,
    end_date: str | None,
    disabled_experts: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    config = read_json(paths.config_path)
    configured_disabled = {str(item) for item in config.get("disabled_experts_by_default", [])}
    active_disabled = configured_disabled | (disabled_experts or set())
    style = load_style_daily(paths.root, config)
    industry = load_industry_daily(paths.root, config)
    if start_date:
        start = pd.to_datetime(start_date)
        style = style[style["date"] >= start]
        industry = industry[industry["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        style = style[style["date"] <= end]
        industry = industry[industry["date"] <= end]
    broad_code = normalize_code(config["asset_universe"]["style"].get("broad_market", "000985"))
    style_raw = style.sort_values(["asset", "date"]).copy()
    style_raw["ret_1d"] = style_raw.groupby("asset")["close"].pct_change()
    market_returns = style_raw[style_raw["asset"].eq(broad_code)].set_index("date")["ret_1d"]
    style_features = add_features(style, market_returns=market_returns)
    industry_features = add_features(industry, market_returns=market_returns)
    valuation = load_style_valuation(paths.root, config, style_features["asset"].drop_duplicates())
    style_features = add_valuation_scores(style_features, valuation)
    industry_features["valuation_score"] = 0.0
    style_scores = score_assets(style_features, is_style=True)
    industry_scores = score_assets(industry_features, is_style=False)
    scored = pd.concat([style_scores, industry_scores], ignore_index=True, sort=False)
    regimes = assign_regime(style_scores, industry_scores, config)
    min_history_days = int(config["feature_pipeline"].get("min_history_days", 504))
    eligible = scored.copy()
    eligible["history_count"] = eligible.groupby("asset").cumcount() + 1
    eligible = eligible[eligible["history_count"] >= min_history_days]
    targets = build_targets(eligible, regimes, config, start_date=eligible["date"].min(), disabled_experts=active_disabled)
    returns = scored[["date", "asset", "ret_1d"]].dropna().copy()
    exposure = target_exposure_summary(targets)
    expert_ic = expert_rank_ic_report(scored, regimes)
    ablation = ablation_report(
        eligible,
        regimes,
        config,
        returns,
        broad_code,
        eligible["date"].min(),
        cost_bps=10.0,
        base_disabled_experts=active_disabled,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scored, paths.output_dir / "feature_score_panel.csv")
    write_csv(regimes, paths.output_dir / "regime_states.csv")
    write_csv(targets, paths.output_dir / "target_weights_monthly.csv")
    write_csv(exposure["monthly"], paths.output_dir / "monthly_target_exposure.csv")
    write_csv(exposure["by_state"], paths.output_dir / "target_exposure_by_state.csv")
    write_csv(expert_ic, paths.output_dir / "expert_rank_ic.csv")
    write_csv(ablation, paths.output_dir / "expert_ablation_summary.csv")
    write_csv(pd.DataFrame({"expert": sorted(active_disabled)}), paths.output_dir / "disabled_experts.csv")
    summaries = []
    latest_targets = targets[targets["signal_date"].eq(targets["signal_date"].max())].copy() if not targets.empty else pd.DataFrame()
    results = {"features": scored, "regimes": regimes, "targets": targets, "expert_ic": expert_ic, "ablation": ablation}
    for cost_bps in config["validation"]["cost_bps_scenarios"]:
        bt = run_backtest(returns, targets, float(cost_bps), broad_code)
        nav = bt["nav"]
        summary = summarize_nav(nav)
        if not summary.empty:
            summary.insert(0, "cost_bps", cost_bps)
            summaries.append(summary)
        suffix = f"{int(cost_bps)}bps"
        write_csv(nav, paths.output_dir / f"nav_{suffix}.csv")
        write_csv(bt["trades"], paths.output_dir / f"trades_{suffix}.csv")
        write_csv(yearly_returns(nav), paths.output_dir / f"yearly_returns_{suffix}.csv")
        write_csv(regime_returns(nav, regimes), paths.output_dir / f"regime_returns_{suffix}.csv")
        results[f"nav_{suffix}"] = nav
    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    write_csv(summary_df, paths.output_dir / "cost_sensitivity_summary.csv")
    if not latest_targets.empty:
        write_csv(latest_targets, paths.output_dir / "latest_target_weights.csv")
    make_report(paths.output_dir, summary_df, latest_targets, config, expert_ic, ablation, exposure["by_state"], active_disabled)
    return results | {"summary": summary_df, "disabled_experts": pd.DataFrame({"expert": sorted(active_disabled)})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "hirssm_v2_0"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--disabled-experts", nargs="*", default=[], help="Expert names to disable, such as range_reversal.")
    args = parser.parse_args()
    paths = RunPaths(root=Path(args.root), output_dir=Path(args.output_dir), config_path=Path(args.config))
    disabled_experts = {str(item) for item in args.disabled_experts}
    results = run_model(paths, args.start_date, args.end_date, disabled_experts=disabled_experts)
    active_disabled = results.get("disabled_experts", pd.DataFrame(columns=["expert"]))["expert"].tolist()
    print(
        json.dumps(
            {
                "output_dir": str(paths.output_dir.resolve()),
                "targets": int(results["targets"].shape[0]),
                "summary_rows": int(results["summary"].shape[0]),
                "disabled_experts": active_disabled,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
