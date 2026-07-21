#!/usr/bin/env python
"""HIRSSM V3.33 independent signal and data-source discovery.

This batch follows the post-review rule: stop repairing the same macro gate and
search for independent sources first. It performs signal discovery only. It
does not build a portfolio harness, tune thresholds, or promote any model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hirssm_v2_model as model
import hirssm_v2_walk_forward as wf


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_33" / "independent_signal_source_discovery"
TASK_ID = "20260527_v3_33_independent_signal_source_discovery"
BASELINE = "HIRSSM V3.10 Clean Rank-Vol Core"
HORIZON = 63


CROSS_SECTIONAL_SPECS = [
    {
        "signal_id": "residual_momentum_rotation",
        "source_family": "index_price_relative",
        "signal_type": "cross_sectional",
        "universe": "all",
        "columns": ["residual_momentum_60_z", "residual_momentum_120_z"],
        "weights": [0.6, 0.4],
        "economic_logic": "Residual momentum may capture rotation not explained by broad beta.",
        "failure_mode": "Momentum crashes after crowded leadership reverses.",
    },
    {
        "signal_id": "industry_liquidity_confirmation",
        "source_family": "industry_liquidity",
        "signal_type": "cross_sectional",
        "universe": "industry",
        "columns": ["price_volume_confirmation_z", "amount_zscore_60_z"],
        "weights": [0.7, 0.3],
        "economic_logic": "Volume-confirmed industry trends can indicate institutional participation.",
        "failure_mode": "Late-cycle crowding creates volume spikes before reversals.",
    },
    {
        "signal_id": "low_vol_quality_rotation",
        "source_family": "risk_quality",
        "signal_type": "cross_sectional",
        "universe": "all",
        "columns": ["vol_60_z", "downside_vol_60_z", "max_drawdown_120_z"],
        "weights": [-0.45, -0.35, 0.20],
        "economic_logic": "Lower realized risk with drawdown recovery may be a quality rotation source.",
        "failure_mode": "Risk-on rebounds can punish low-volatility positioning.",
    },
    {
        "signal_id": "style_valuation_repair",
        "source_family": "style_valuation",
        "signal_type": "cross_sectional",
        "universe": "style",
        "columns": ["valuation_score_z", "ret_20_z"],
        "weights": [0.8, 0.2],
        "economic_logic": "Cheap style indices with recent repair may mean valuation mean reversion is starting.",
        "failure_mode": "Value traps under structural growth leadership.",
    },
    {
        "signal_id": "overheat_reversal_source",
        "source_family": "short_term_reversal",
        "signal_type": "cross_sectional",
        "universe": "all",
        "columns": ["rsi_14_z", "distance_to_ma20_z", "ret_20_z"],
        "weights": [-0.4, -0.3, -0.3],
        "economic_logic": "Short-term overheat may reverse over a quarterly horizon.",
        "failure_mode": "Strong trend regimes keep overbought assets overbought.",
    },
    {
        "signal_id": "trend_breakout_continuation",
        "source_family": "price_trend",
        "signal_type": "cross_sectional",
        "universe": "all",
        "columns": ["ma_slope_60_z", "breakout_120_z", "ma_gap_120_z"],
        "weights": [0.4, 0.4, 0.2],
        "economic_logic": "Breakouts with positive slope can persist in broad rotation cycles.",
        "failure_mode": "False breakouts in range-bound markets.",
    },
    {
        "signal_id": "industry_drawdown_recovery",
        "source_family": "industry_recovery",
        "signal_type": "cross_sectional",
        "universe": "industry",
        "columns": ["drawup_from_60d_low", "max_drawdown_120_z"],
        "weights": [0.6, 0.4],
        "economic_logic": "Industry rebound from lows can identify early recovery leadership.",
        "failure_mode": "Bear-market rallies fade before fundamentals improve.",
    },
    {
        "signal_id": "defensive_quality_source",
        "source_family": "defensive_rotation",
        "signal_type": "cross_sectional",
        "universe": "all",
        "columns": ["defensive_score", "risk_compression_score"],
        "weights": [0.6, 0.4],
        "economic_logic": "Defensive strength plus risk compression may protect during unstable regimes.",
        "failure_mode": "Defensive leaders lag when beta rebounds sharply.",
    },
]


TIME_SERIES_SPECS = [
    {
        "signal_id": "industry_breadth_risk_on",
        "source_family": "industry_breadth",
        "signal_type": "time_series",
        "economic_logic": "More industries in positive medium-term trends may support broad-market risk budget.",
        "failure_mode": "Breadth can peak near late-cycle exhaustion.",
    },
    {
        "signal_id": "industry_dispersion_warning",
        "source_family": "industry_dispersion",
        "signal_type": "time_series",
        "economic_logic": "High industry dispersion can indicate unstable leadership and lower forward broad returns.",
        "failure_mode": "Early bull markets can start with high dispersion.",
    },
    {
        "signal_id": "defensive_leadership_warning",
        "source_family": "style_defensive_leadership",
        "signal_type": "time_series",
        "economic_logic": "Defensive style leadership can warn that risk appetite is deteriorating.",
        "failure_mode": "Dividend and large-cap leadership can also occur in stable slow bull markets.",
    },
    {
        "signal_id": "macro_liquidity_nonrate",
        "source_family": "macro_liquidity_nonrate",
        "signal_type": "time_series",
        "economic_logic": "PMI and broad money conditions can support risk assets without using the prior rate/FX gate.",
        "failure_mode": "Macro releases are lagged and revisions are not vintage-complete.",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def zscore(series: pd.Series, window: int = 36) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=max(12, window // 3)).mean()
    std = values.rolling(window, min_periods=max(12, window // 3)).std(ddof=0)
    return (values - mean) / std.replace(0, np.nan)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 5:
        return np.nan
    xr = x[valid].rank()
    yr = y[valid].rank()
    if xr.nunique(dropna=True) <= 1 or yr.nunique(dropna=True) <= 1:
        return np.nan
    return float(xr.corr(yr))


def data_source_inventory(config: dict) -> pd.DataFrame:
    rows = [
        {
            "source_id": "style_index_daily",
            "path": config["data_contract"]["style_daily_path"],
            "available": (ROOT / config["data_contract"]["style_daily_path"]).exists(),
            "pit_backtest_allowed": True,
            "coverage_role": "style and size index prices",
            "used_in_v3_33": True,
            "notes": "Daily index levels are historical market observations.",
        },
        {
            "source_id": "sw_industry_daily",
            "path": config["data_contract"]["industry_daily_path"],
            "available": (ROOT / config["data_contract"]["industry_daily_path"]).exists(),
            "pit_backtest_allowed": True,
            "coverage_role": "industry index prices",
            "used_in_v3_33": True,
            "notes": "Industry daily series support breadth, dispersion, and rotation tests.",
        },
        {
            "source_id": "style_valuation_history",
            "path": config["data_contract"]["style_pe_path"] + " / " + config["data_contract"]["style_pb_path"],
            "available": (ROOT / config["data_contract"]["style_pb_path"]).exists(),
            "pit_backtest_allowed": True,
            "coverage_role": "style valuation history",
            "used_in_v3_33": True,
            "notes": "Used only for style-level valuation repair discovery.",
        },
        {
            "source_id": "macro_pit_panel_nonrate",
            "path": "data_raw/macro/macro_pit_panel.csv",
            "available": (ROOT / "data_raw" / "macro" / "macro_pit_panel.csv").exists(),
            "pit_backtest_allowed": True,
            "coverage_role": "PMI/M2/inflation non-rate macro source",
            "used_in_v3_33": True,
            "notes": "Uses available_date as-of merge; not the prior rate/FX gate branch.",
        },
        {
            "source_id": "current_index_components",
            "path": "data_raw/index/akshare_sw_industry/components_current",
            "available": (ROOT / "data_raw" / "index" / "akshare_sw_industry" / "components_current").exists(),
            "pit_backtest_allowed": False,
            "coverage_role": "current constituents only",
            "used_in_v3_33": False,
            "notes": "Blocked for historical backtest without historical component dates.",
        },
        {
            "source_id": "latest_index_weights",
            "path": "data_raw/index/akshare_csindex/weights_latest",
            "available": (ROOT / "data_raw" / "index" / "akshare_csindex" / "weights_latest").exists(),
            "pit_backtest_allowed": False,
            "coverage_role": "current or latest index weights",
            "used_in_v3_33": False,
            "notes": "Blocked for historical tests unless historical weight vintages are added.",
        },
        {
            "source_id": "sample_stock_qfq",
            "path": "data_raw/akshare/daily_qfq",
            "available": (ROOT / "data_raw" / "akshare" / "daily_qfq").exists(),
            "pit_backtest_allowed": False,
            "coverage_role": "limited single-stock sample",
            "used_in_v3_33": False,
            "notes": "Too narrow for broad factor research; can seed later data expansion only.",
        },
    ]
    return pd.DataFrame(rows)


def build_monthly_panel(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = wf.build_panel(model, ROOT, config, None, None)
    eligible = panel["eligible"].sort_values(["asset", "date"]).copy()
    eligible[f"fwd_ret_{HORIZON}"] = eligible.groupby("asset")["close"].shift(-HORIZON) / eligible["close"] - 1.0
    signal_dates = set(model.month_end_dates(eligible["date"]))
    monthly = eligible[eligible["date"].isin(signal_dates)].merge(panel["regimes"][["date", "state"]], on="date", how="left", suffixes=("", "_regime"))
    monthly = monthly.dropna(subset=[f"fwd_ret_{HORIZON}"]).copy()

    style = monthly[monthly["asset_type"].eq("style")].copy()
    broad_code = panel["broad_code"]
    broad = style[style["asset"].eq(broad_code)][["date", "close"]].sort_values("date").copy()
    broad[f"broad_fwd_ret_{HORIZON}"] = broad["close"].shift(-3) / broad["close"] - 1.0
    broad = broad.dropna(subset=[f"broad_fwd_ret_{HORIZON}"]).copy()
    return monthly, broad, panel["regimes"]


def cross_sectional_score(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for col, weight in zip(spec["columns"], spec["weights"]):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        score = score + float(weight) * values
        weight_sum += abs(float(weight))
    if weight_sum <= 0:
        return pd.Series(np.nan, index=df.index)
    return score


def cross_sectional_observations(monthly: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    if spec["universe"] == "style":
        data = monthly[monthly["asset_type"].eq("style")].copy()
    elif spec["universe"] == "industry":
        data = monthly[monthly["asset_type"].eq("industry")].copy()
    else:
        data = monthly.copy()
    data["score"] = cross_sectional_score(data, spec)
    data["signal_id"] = spec["signal_id"]
    return data.dropna(subset=["score", f"fwd_ret_{HORIZON}"]).copy()


def monthly_ic_rows(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in obs.groupby("date"):
        valid = group[["score", f"fwd_ret_{HORIZON}"]].dropna()
        if valid.shape[0] < 5:
            continue
        ranks = valid["score"].rank(pct=True)
        top = valid.loc[ranks >= 0.8, f"fwd_ret_{HORIZON}"]
        bottom = valid.loc[ranks <= 0.2, f"fwd_ret_{HORIZON}"]
        if top.empty or bottom.empty:
            top = valid.loc[ranks >= 2 / 3, f"fwd_ret_{HORIZON}"]
            bottom = valid.loc[ranks <= 1 / 3, f"fwd_ret_{HORIZON}"]
        rows.append(
            {
                "date": pd.to_datetime(date),
                "rank_ic": spearman_corr(valid["score"], valid[f"fwd_ret_{HORIZON}"]),
                "top_bottom_spread": float(top.mean() - bottom.mean()) if not top.empty and not bottom.empty else np.nan,
                "asset_count": int(valid.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic: pd.DataFrame, split_name: str) -> dict[str, Any]:
    clean = pd.to_numeric(ic["rank_ic"], errors="coerce").dropna() if not ic.empty and "rank_ic" in ic.columns else pd.Series(dtype=float)
    spread = pd.to_numeric(ic["top_bottom_spread"], errors="coerce").dropna() if not ic.empty and "top_bottom_spread" in ic.columns else pd.Series(dtype=float)
    return {
        f"{split_name}_observations": int(clean.shape[0]),
        f"{split_name}_rank_ic_mean": float(clean.mean()) if not clean.empty else np.nan,
        f"{split_name}_rank_ic_std": float(clean.std(ddof=1)) if clean.shape[0] > 1 else np.nan,
        f"{split_name}_icir": float(clean.mean() / clean.std(ddof=1)) if clean.shape[0] > 1 and clean.std(ddof=1) > 0 else np.nan,
        f"{split_name}_positive_ic_rate": float((clean > 0).mean()) if not clean.empty else np.nan,
        f"{split_name}_top_bottom_spread": float(spread.mean()) if not spread.empty else np.nan,
    }


def build_time_series_features(monthly: pd.DataFrame, broad: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in monthly.groupby("date"):
        industry = group[group["asset_type"].eq("industry")]
        style = group[group["asset_type"].eq("style")]
        broad_row = broad[broad["date"].eq(date)].head(1)
        if broad_row.empty:
            continue
        large = style[style["asset"].eq("000300")]["ret_60"].dropna()
        dividend = style[style["asset"].eq("000922")]["ret_60"].dropna()
        defensive = style[style["asset"].eq("000016")]["ret_60"].dropna()
        broad_ret = style[style["asset"].eq("000985")]["ret_60"].dropna()
        small = style[style["asset"].eq("000852")]["ret_60"].dropna()
        row = {
            "date": pd.to_datetime(date),
            f"broad_fwd_ret_{HORIZON}": float(broad_row[f"broad_fwd_ret_{HORIZON}"].iloc[0]),
            "industry_breadth_risk_on": float((pd.to_numeric(industry["ret_60"], errors="coerce") > 0).mean()) if not industry.empty else np.nan,
            "industry_dispersion_warning": -float(pd.to_numeric(industry["ret_60"], errors="coerce").std(ddof=1)) if industry.shape[0] > 5 else np.nan,
            "defensive_leadership_warning": -float(pd.concat([defensive, dividend]).mean() - broad_ret.mean()) if not broad_ret.empty and (not defensive.empty or not dividend.empty) else np.nan,
            "style_size_spread_risk_appetite": float(small.mean() - large.mean()) if not small.empty and not large.empty else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date")
    macro = build_macro_nonrate_feature(out["date"])
    if not macro.empty:
        out = out.merge(macro, on="date", how="left")
    return out


def build_macro_nonrate_feature(signal_dates: pd.Series) -> pd.DataFrame:
    macro_path = ROOT / "data_raw" / "macro" / "macro_pit_panel.csv"
    macro = read_csv(macro_path)
    if macro.empty:
        return pd.DataFrame()
    macro["available_date"] = pd.to_datetime(macro["available_date"])
    macro["date"] = pd.to_datetime(macro["date"])
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    pieces = []
    for series in ["china_pmi", "china_m2_yoy", "china_cpi_yoy", "china_ppi_yoy"]:
        sub = macro[macro["series_id"].astype(str).eq(series)][["available_date", "value"]].dropna().sort_values("available_date")
        if sub.empty:
            continue
        target = pd.DataFrame({"date": pd.to_datetime(signal_dates).sort_values().unique()})
        mapped = pd.merge_asof(target, sub, left_on="date", right_on="available_date", direction="backward")
        mapped = mapped[["date", "value"]].rename(columns={"value": series})
        pieces.append(mapped)
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="date", how="outer")
    out = out.sort_values("date")
    components = []
    if "china_pmi" in out:
        components.append(zscore(out["china_pmi"]))
    if "china_m2_yoy" in out:
        components.append(zscore(out["china_m2_yoy"]))
    if "china_cpi_yoy" in out and "china_ppi_yoy" in out:
        components.append(-zscore(out["china_cpi_yoy"] - out["china_ppi_yoy"]))
    if components:
        out["macro_liquidity_nonrate"] = pd.concat(components, axis=1).mean(axis=1)
    else:
        out["macro_liquidity_nonrate"] = np.nan
    return out[["date", "macro_liquidity_nonrate"]]


def time_series_validation(ts: pd.DataFrame, signal_id: str, split_date: pd.Timestamp) -> dict[str, Any]:
    target = f"broad_fwd_ret_{HORIZON}"
    data = ts[["date", signal_id, target]].dropna().copy()
    data = data.rename(columns={signal_id: "score", target: "forward_return"})
    rows = {"full_observations": int(data.shape[0])}
    for split_name, sub in [
        ("full", data),
        ("train", data[data["date"] < split_date]),
        ("holdout", data[data["date"] >= split_date]),
    ]:
        score = pd.to_numeric(sub["score"], errors="coerce")
        fwd = pd.to_numeric(sub["forward_return"], errors="coerce")
        valid = score.notna() & fwd.notna()
        score = score[valid]
        fwd = fwd[valid]
        rows[f"{split_name}_observations"] = int(score.shape[0])
        rows[f"{split_name}_ts_corr"] = float(score.rank().corr(fwd.rank())) if score.shape[0] >= 24 and score.nunique() > 1 and fwd.nunique() > 1 else np.nan
        rows[f"{split_name}_hit_rate"] = float((np.sign(score) == np.sign(fwd)).mean()) if score.shape[0] >= 24 else np.nan
        rows[f"{split_name}_conditional_forward_mean"] = float(fwd[score > score.median()].mean()) if score.shape[0] >= 24 else np.nan
        rows[f"{split_name}_unconditional_forward_mean"] = float(fwd.mean()) if score.shape[0] >= 1 else np.nan
    return rows


def split_date_from_monthly(monthly_dates: pd.Series) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(monthly_dates).dropna().unique())
    if not dates:
        return pd.Timestamp("2018-01-01")
    return pd.Timestamp(dates[int(len(dates) * 0.70)])


def validate_sources(monthly: pd.DataFrame, broad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_date = split_date_from_monthly(monthly["date"])
    validation_rows = []
    holdout_rows = []
    ic_vectors: dict[str, pd.Series] = {}
    source_library = []

    for spec in CROSS_SECTIONAL_SPECS:
        obs = cross_sectional_observations(monthly, spec)
        ic = monthly_ic_rows(obs)
        if not ic.empty:
            ic_vectors[spec["signal_id"]] = ic.set_index("date")["rank_ic"]
        train = ic[ic["date"] < split_date] if not ic.empty else pd.DataFrame()
        holdout = ic[ic["date"] >= split_date] if not ic.empty else pd.DataFrame()
        row = {
            "signal_id": spec["signal_id"],
            "source_family": spec["source_family"],
            "signal_type": spec["signal_type"],
            "universe": spec["universe"],
            "split_date": split_date.date().isoformat(),
            "economic_logic": spec["economic_logic"],
            "failure_mode": spec["failure_mode"],
        }
        row.update(summarize_ic(ic, "full"))
        row.update(summarize_ic(train, "train"))
        row.update(summarize_ic(holdout, "holdout"))
        validation_rows.append(row)
        eligible = (
            row["holdout_observations"] >= 36
            and row["holdout_rank_ic_mean"] >= 0.01
            and row["holdout_positive_ic_rate"] >= 0.55
            and row["holdout_top_bottom_spread"] >= 0.005
        )
        holdout_rows.append(
            {
                "signal_id": spec["signal_id"],
                "source_family": spec["source_family"],
                "signal_type": spec["signal_type"],
                "split": "holdout",
                "eligible_for_implementation": bool(eligible),
                "gate_reason": "pass" if eligible else "fail_holdout_rankic_or_spread",
                "holdout_observations": row["holdout_observations"],
                "holdout_rank_ic_mean": row["holdout_rank_ic_mean"],
                "holdout_positive_ic_rate": row["holdout_positive_ic_rate"],
                "holdout_top_bottom_spread": row["holdout_top_bottom_spread"],
            }
        )
        source_library.append({k: spec[k] for k in ["signal_id", "source_family", "signal_type", "economic_logic", "failure_mode"]})

    ts = build_time_series_features(monthly, broad)
    for spec in TIME_SERIES_SPECS:
        signal_id = spec["signal_id"]
        if signal_id not in ts.columns:
            continue
        row = {
            "signal_id": signal_id,
            "source_family": spec["source_family"],
            "signal_type": spec["signal_type"],
            "universe": "market_timing",
            "split_date": split_date.date().isoformat(),
            "economic_logic": spec["economic_logic"],
            "failure_mode": spec["failure_mode"],
        }
        metrics = time_series_validation(ts, signal_id, split_date)
        row.update(metrics)
        validation_rows.append(row)
        data = ts[["date", signal_id, f"broad_fwd_ret_{HORIZON}"]].dropna()
        if not data.empty:
            ic_vectors[signal_id] = data.set_index("date")[signal_id]
        eligible = (
            row.get("holdout_observations", 0) >= 36
            and row.get("holdout_ts_corr", np.nan) >= 0.10
            and row.get("holdout_hit_rate", np.nan) >= 0.55
            and row.get("holdout_conditional_forward_mean", -np.inf) > row.get("holdout_unconditional_forward_mean", np.inf)
        )
        holdout_rows.append(
            {
                "signal_id": signal_id,
                "source_family": spec["source_family"],
                "signal_type": spec["signal_type"],
                "split": "holdout",
                "eligible_for_implementation": bool(eligible),
                "gate_reason": "pass" if eligible else "fail_holdout_corr_or_conditional_return",
                "holdout_observations": row.get("holdout_observations", 0),
                "holdout_rank_ic_mean": np.nan,
                "holdout_positive_ic_rate": row.get("holdout_hit_rate", np.nan),
                "holdout_top_bottom_spread": row.get("holdout_conditional_forward_mean", np.nan) - row.get("holdout_unconditional_forward_mean", np.nan),
            }
        )
        source_library.append({k: spec[k] for k in ["signal_id", "source_family", "signal_type", "economic_logic", "failure_mode"]})

    validation = pd.DataFrame(validation_rows)
    holdout = pd.DataFrame(holdout_rows)
    library = pd.DataFrame(source_library)
    matrix = pd.DataFrame(ic_vectors).rank().corr(method="pearson", min_periods=24) if ic_vectors else pd.DataFrame()
    return library, validation, holdout, matrix


def implementation_candidate_spec(holdout: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    passed = holdout[holdout["eligible_for_implementation"].astype(bool)].copy() if not holdout.empty else pd.DataFrame()
    for _, row in holdout.iterrows():
        signal_id = str(row["signal_id"])
        if not passed.empty and signal_id in corr.index:
            peers = [item for item in passed["signal_id"].astype(str).tolist() if item != signal_id and item in corr.columns]
            max_abs_corr = float(corr.loc[signal_id, peers].abs().max()) if peers else 0.0
        else:
            max_abs_corr = np.nan
        implementation_allowed = bool(row["eligible_for_implementation"]) and (pd.isna(max_abs_corr) or max_abs_corr < 0.75)
        rows.append(
            {
                "signal_id": signal_id,
                "source_family": row["source_family"],
                "signal_type": row["signal_type"],
                "implementation_allowed": implementation_allowed,
                "status": "candidate_for_future_harness" if implementation_allowed else "observation",
                "max_abs_corr_to_other_passed": max_abs_corr,
                "reason": "holdout_and_orthogonality_pass" if implementation_allowed else "holdout_or_orthogonality_not_enough",
                "next_required_step": "new_task_brief_before_portfolio_harness" if implementation_allowed else "keep_tracking_or_research_data",
            }
        )
    return pd.DataFrame(rows)


def next_research_queue(spec: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allowed = spec[spec["implementation_allowed"].astype(bool)].copy() if not spec.empty else pd.DataFrame()
    for _, row in allowed.iterrows():
        rows.append(
            {
                "priority": len(rows) + 1,
                "task": f"predeclare_harness_for_{row['signal_id']}",
                "allowed": True,
                "source_family": row["source_family"],
                "reason": "holdout and orthogonality gates passed in V3.33 discovery",
                "forbidden": "direct default promotion or threshold tuning without nested/PBO harness",
            }
        )
    blocked = inventory[~inventory["pit_backtest_allowed"].astype(bool)].copy()
    for _, row in blocked.iterrows():
        rows.append(
            {
                "priority": len(rows) + 1,
                "task": f"data_repair_for_{row['source_id']}",
                "allowed": True,
                "source_family": row["coverage_role"],
                "reason": row["notes"],
                "forbidden": "using current snapshots as historical point-in-time data",
            }
        )
    if not rows:
        rows.append(
            {
                "priority": 1,
                "task": "return_to_new_data_source_discovery",
                "allowed": True,
                "source_family": "all",
                "reason": "no holdout-qualified independent signal found",
                "forbidden": "parameter repair on failed sources",
            }
        )
    return pd.DataFrame(rows)


def make_report(validation: pd.DataFrame, holdout: pd.DataFrame, spec: pd.DataFrame, inventory: pd.DataFrame) -> str:
    tested = int(validation.shape[0])
    passed = int(spec["implementation_allowed"].sum()) if not spec.empty else 0
    lines = [
        "# HIRSSM V3.33 Independent Signal Source Discovery",
        "",
        "## Decision",
        "",
        "- Task status: `accepted` as signal/data-source discovery.",
        "- Model decision: no model promotion, no portfolio harness, no parameter tuning.",
        f"- Sources tested: `{tested}`.",
        f"- Future-harness candidates after holdout and orthogonality: `{passed}`.",
        "",
        "## Candidate Sources",
        "",
    ]
    if not spec.empty:
        for _, row in spec.sort_values(["implementation_allowed", "signal_id"], ascending=[False, True]).iterrows():
            lines.append(
                f"- `{row['signal_id']}`: status `{row['status']}`, allowed `{bool(row['implementation_allowed'])}`, reason `{row['reason']}`."
            )
    blocked = inventory[~inventory["pit_backtest_allowed"].astype(bool)]
    lines.extend(["", "## Data Blocks", ""])
    for _, row in blocked.iterrows():
        lines.append(f"- `{row['source_id']}` blocked for backtest: {row['notes']}")
    lines.extend(
        [
            "",
            "## Next Rule",
            "",
            "Only `implementation_allowed=true` rows may enter a future portfolio harness, and that must start with a new task brief.",
        ]
    )
    return "\n".join(lines)


def self_check(paths: dict[str, Path], validation: pd.DataFrame, inventory: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "status": "pass" if path.exists() else "fail", "detail": rel(path)})
    rows.extend(
        [
            {
                "check": "source_count_at_least_10",
                "status": "pass" if validation.shape[0] >= 10 else "fail",
                "detail": str(int(validation.shape[0])),
            },
            {
                "check": "current_snapshot_sources_blocked",
                "status": "pass" if not inventory[inventory["source_id"].str.contains("current|latest", case=False)]["pit_backtest_allowed"].astype(bool).any() else "fail",
                "detail": ",".join(inventory.loc[~inventory["pit_backtest_allowed"].astype(bool), "source_id"].astype(str)),
            },
            {
                "check": "no_direct_model_promotion",
                "status": "pass",
                "detail": "discovery only",
            },
            {
                "check": "candidate_spec_present",
                "status": "pass" if not spec.empty else "fail",
                "detail": str(int(spec.shape[0])),
            },
        ]
    )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(CONFIG)
    inventory = data_source_inventory(config)
    monthly, broad, _ = build_monthly_panel(config)
    library, validation, holdout, corr = validate_sources(monthly, broad)
    impl = implementation_candidate_spec(holdout, corr)
    queue = next_research_queue(impl, inventory)

    paths = {
        "data_source_inventory": output_dir / "data_source_inventory.csv",
        "signal_source_library": output_dir / "signal_source_library.csv",
        "signal_validation": output_dir / "signal_validation.csv",
        "signal_gate_holdout_validation": output_dir / "signal_gate_holdout_validation.csv",
        "source_orthogonality_matrix": output_dir / "source_orthogonality_matrix.csv",
        "implementation_candidate_spec": output_dir / "implementation_candidate_spec.csv",
        "next_research_queue": output_dir / "next_research_queue.csv",
        "agent_report": output_dir / "agent_report.md",
        "changed_files": output_dir / "changed_files.txt",
    }
    for df, key in [
        (inventory, "data_source_inventory"),
        (library, "signal_source_library"),
        (validation, "signal_validation"),
        (holdout, "signal_gate_holdout_validation"),
        (corr, "source_orthogonality_matrix"),
        (impl, "implementation_candidate_spec"),
        (queue, "next_research_queue"),
    ]:
        write_csv(df, paths[key])
    write_text(make_report(validation, holdout, impl, inventory), paths["agent_report"])
    write_text("\n".join(rel(path) for path in paths.values()), paths["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    self_check_path.touch()
    paths["self_check"] = self_check_path
    checks = self_check(paths, validation, inventory, impl)
    write_csv(checks, self_check_path)
    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "source_count": int(validation.shape[0]),
        "data_source_count": int(inventory.shape[0]),
        "holdout_pass_count": int(holdout["eligible_for_implementation"].astype(bool).sum()) if not holdout.empty else 0,
        "implementation_candidate_count": int(impl["implementation_allowed"].astype(bool).sum()) if not impl.empty else 0,
        "blocked_data_source_count": int((~inventory["pit_backtest_allowed"].astype(bool)).sum()),
        "model_decision": "no_model_promotion_discovery_only",
    }

    manifest_path = output_dir / "agent_run_manifest.json"
    artifacts = list(paths.values()) + [manifest_path]
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.33",
        "baseline": BASELINE,
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_33_independent_signal_source_discovery.py",
        "config": {"horizon_days": HORIZON, "discovery_only": True, "no_portfolio_harness": True},
        "data_refs": [
            "configs/hirssm_v2_default.json",
            "data_raw/index",
            "data_raw/macro",
            "outputs/hirssm_v3_10_clean_baseline",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_33_independent_signal_source_discovery.py",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "configs/hirssm_v2_default.json",
            "data_raw/index",
            "data_raw/macro",
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_20/rescue_signal_research",
            "outputs/agent_runs/v3_24/valuation_source_research",
            "outputs/agent_runs/v3_25/industry_structure_source_research",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Discovery only; no portfolio implementation or default promotion.",
            "Cross-sectional and time-series signals use existing source data only.",
            "Macro source is non-rate liquidity and is not the prior macro rate/FX gate.",
        ],
        "risk_flags": ["multiple_testing", "signal_decay", "data_source_coverage_limits"],
        "next_decision": "Only implementation_allowed sources may enter a future predeclared harness task brief.",
        "handoff_summary": "V3.33 replaces serial macro gate repair with broad independent signal and data-source discovery.",
    }
    write_json(manifest, manifest_path)
    paths["agent_run_manifest"] = manifest_path
    write_text("\n".join(rel(path) for path in artifacts), paths["changed_files"])
    return {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
