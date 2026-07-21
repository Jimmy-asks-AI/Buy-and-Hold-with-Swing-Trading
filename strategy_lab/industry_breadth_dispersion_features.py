"""Industry breadth and dispersion feature layer for HIRSSM V3.68.

This module builds point-in-time daily features from SW industry index price
and volume history. It is a feature layer only: no portfolio harness, NAV,
Sharpe, drawdown, model promotion, or official total-return evidence is
produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class IndustryBreadthDispersionConfig:
    v3_66_manifest_path: Path
    v3_66_queue_path: Path
    v3_66_inventory_path: Path
    industry_daily_dir: Path
    classification_path: Path
    output_dir: Path
    catalog_path: Path
    industry_level: str
    min_history: int
    trailing_window: int
    short_window: int
    medium_window: int
    long_window: int
    slow_window: int
    min_feature_rows: int
    min_industry_count: int


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)


def _rolling_zscore(values: pd.Series, window: int, min_history: int) -> pd.Series:
    data = pd.to_numeric(values, errors="coerce")
    mean = data.rolling(window, min_periods=min_history).mean()
    std = data.rolling(window, min_periods=min_history).std(ddof=0).replace(0, np.nan)
    return (data - mean) / std


def _trailing_percentile(values: pd.Series, window: int, min_history: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    out = np.full(len(arr), np.nan)
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        start = max(0, idx - window)
        history = arr[start:idx]
        history = history[np.isfinite(history)]
        if len(history) < min_history:
            continue
        out[idx] = float((history <= value).mean())
    return pd.Series(out, index=values.index)


def _clip_score(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(-3, 3) / 3.0


def _score_component(values: pd.Series) -> pd.Series:
    return _clip_score(values).fillna(0.0)


def _row_top_mean(frame: pd.DataFrame, n: int, largest: bool) -> pd.Series:
    values = frame.to_numpy(dtype="float64")
    out = np.full(values.shape[0], np.nan)
    for idx, row in enumerate(values):
        row = row[np.isfinite(row)]
        if len(row) < n:
            continue
        row = np.sort(row)
        selected = row[-n:] if largest else row[:n]
        out[idx] = float(np.mean(selected))
    return pd.Series(out, index=frame.index)


def _row_rank_autocorr(rank_frame: pd.DataFrame, lag: int) -> pd.Series:
    current = rank_frame.to_numpy(dtype="float64")
    previous = rank_frame.shift(lag).to_numpy(dtype="float64")
    out = np.full(len(rank_frame), np.nan)
    for idx in range(lag, len(rank_frame)):
        x = current[idx]
        y = previous[idx]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        out[idx] = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="pearson"))
    return pd.Series(out, index=rank_frame.index)


def normalize_trade_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    out = parsed.dt.strftime("%Y%m%d")
    fallback = (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str[:8]
    )
    return out.fillna(fallback)


def validate_inputs(
    config: IndustryBreadthDispersionConfig,
    v3_66_manifest: dict[str, Any],
    v3_66_queue: pd.DataFrame,
    source_inventory: pd.DataFrame,
) -> pd.DataFrame:
    queue_ok = False
    queue_rows = 0
    if not v3_66_queue.empty and {"version_hint", "source_id"}.issubset(v3_66_queue.columns):
        selected = v3_66_queue[
            v3_66_queue["version_hint"].astype(str).eq("V3.68")
            & v3_66_queue["source_id"].astype(str).eq("sw_industry_daily_history")
        ]
        queue_rows = int(selected.shape[0])
        queue_ok = queue_rows >= 3

    source_ready = False
    if not source_inventory.empty and {"source_id", "allowed_stage"}.issubset(source_inventory.columns):
        source_ready = bool(
            source_inventory.loc[
                source_inventory["source_id"].astype(str).eq("sw_industry_daily_history")
                & source_inventory["allowed_stage"].astype(str).eq("ready_for_feature_layer")
            ].shape[0]
        )

    forbidden_input_roots = ["constituent", "component", "weights_latest", "current"]
    input_text = " ".join(
        [
            str(config.industry_daily_dir).lower(),
            str(config.classification_path).lower(),
        ]
    )
    forbidden_hits = sorted(term for term in forbidden_input_roots if term in input_text and term != "current")
    return pd.DataFrame(
        [
            {
                "check": "v3_66_manifest_self_check_passed",
                "status": _status(bool(v3_66_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_66_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_66_queue_contains_three_v3_68_industry_tasks",
                "status": _status(queue_ok),
                "detail": f"queue_rows={queue_rows}",
            },
            {
                "check": "v3_66_inventory_marks_sw_industry_ready",
                "status": _status(source_ready),
                "detail": "sw_industry_daily_history",
            },
            {
                "check": "industry_daily_directory_exists",
                "status": _status(config.industry_daily_dir.exists()),
                "detail": str(config.industry_daily_dir),
            },
            {
                "check": "classification_file_exists",
                "status": _status(config.classification_path.exists()),
                "detail": str(config.classification_path),
            },
            {
                "check": "current_component_snapshot_excluded",
                "status": _status(not forbidden_hits),
                "detail": ",".join(forbidden_hits),
            },
        ]
    )


def load_industry_info(config: IndustryBreadthDispersionConfig) -> pd.DataFrame:
    info = pd.read_csv(config.classification_path, encoding="utf-8-sig", low_memory=False)
    required = {"index_code", "index_name", "sw_level"}
    missing = sorted(required.difference(info.columns))
    if missing:
        raise ValueError(f"classification missing columns: {missing}")
    info["index_code"] = info["index_code"].astype(str).str.strip()
    info["index_name"] = info["index_name"].astype(str).str.strip()
    info["sw_level"] = info["sw_level"].astype(str).str.strip()
    info = info[info["sw_level"].eq(config.industry_level)].copy()
    if info.empty:
        raise ValueError(f"no industry rows for level={config.industry_level}")
    return info.drop_duplicates("index_code").sort_values("index_code").reset_index(drop=True)


def load_industry_daily(config: IndustryBreadthDispersionConfig, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    names = dict(zip(info["index_code"].astype(str), info["index_name"].astype(str)))
    for code in info["index_code"].astype(str).tolist():
        path = config.industry_daily_dir / f"{code}.csv"
        if not path.exists():
            quality_rows.append({"index_code": code, "index_name": names.get(code, ""), "rows": 0, "status": "missing"})
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"index_code", "date", "close", "high", "low", "volume", "amount"}
        missing = sorted(required.difference(df.columns))
        if missing:
            quality_rows.append({"index_code": code, "index_name": names.get(code, ""), "rows": int(df.shape[0]), "status": "missing_columns", "missing": ",".join(missing)})
            continue
        df = df.copy()
        df["index_code"] = code
        df["index_name"] = names.get(code, code)
        df["trade_date"] = normalize_trade_date(df["date"])
        df = df.sort_values("trade_date").drop_duplicates("trade_date")
        for col in ["close", "open", "high", "low", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        quality_rows.append(
            {
                "index_code": code,
                "index_name": names.get(code, ""),
                "rows": int(df.shape[0]),
                "start_trade_date": str(df["trade_date"].min()),
                "end_trade_date": str(df["trade_date"].max()),
                "close_nonnull": int(df["close"].notna().sum()),
                "amount_nonnull": int(df["amount"].notna().sum()),
                "status": "pass" if int(df["close"].notna().sum()) >= config.min_feature_rows else "short_history",
            }
        )
        frames.append(df[["trade_date", "index_code", "index_name", "close", "high", "low", "volume", "amount"]])
    if not frames:
        raise RuntimeError("no industry daily frames loaded")
    long = pd.concat(frames, ignore_index=True)
    quality = pd.DataFrame(quality_rows)
    return long, quality


def _pivot(long: pd.DataFrame, value_col: str, codes: list[str]) -> pd.DataFrame:
    wide = long.pivot(index="trade_date", columns="index_code", values=value_col).sort_index()
    for code in codes:
        if code not in wide.columns:
            wide[code] = np.nan
    return wide.loc[:, codes]


def build_industry_feature_panel(long: pd.DataFrame, info: pd.DataFrame, config: IndustryBreadthDispersionConfig) -> pd.DataFrame:
    codes = info["index_code"].astype(str).tolist()
    close = _pivot(long, "close", codes).ffill()
    high = _pivot(long, "high", codes).ffill()
    low = _pivot(long, "low", codes).ffill()
    volume = _pivot(long, "volume", codes).fillna(0.0)
    amount = _pivot(long, "amount", codes).fillna(0.0)

    valid_close = close.notna() & close.gt(0)
    industry_count = valid_close.sum(axis=1)
    close = close.where(valid_close)
    high = high.where(valid_close)
    low = low.where(valid_close)

    ret1 = close / close.shift(1) - 1.0
    ret5 = close / close.shift(config.short_window) - 1.0
    ret20 = close / close.shift(config.medium_window) - 1.0
    ret60 = close / close.shift(config.long_window) - 1.0
    ret120 = close / close.shift(config.slow_window) - 1.0
    ma20 = close.rolling(config.medium_window, min_periods=max(5, config.medium_window // 2)).mean()
    ma60 = close.rolling(config.long_window, min_periods=max(10, config.long_window // 2)).mean()
    ma120 = close.rolling(config.slow_window, min_periods=max(20, config.slow_window // 2)).mean()
    range_ratio = (high - low) / close.replace(0, np.nan)

    amount_total = amount.sum(axis=1)
    amount_share = amount.div(amount_total.replace(0, np.nan), axis=0)
    amount_hhi = (amount_share**2).sum(axis=1)
    amount_top3_share = amount_share.apply(lambda row: row.nlargest(min(3, row.notna().sum())).sum(), axis=1)
    amount_ma60 = amount.rolling(config.long_window, min_periods=max(10, config.long_window // 2)).mean()
    amount_expansion = amount > amount_ma60
    zero_amount_ratio = amount.le(0).sum(axis=1) / industry_count.replace(0, np.nan)

    ret20_rank = ret20.rank(axis=1, pct=True)
    ret60_rank = ret60.rank(axis=1, pct=True)
    rank_persistence_20 = _row_rank_autocorr(ret20_rank, config.medium_window)
    rank_persistence_60 = _row_rank_autocorr(ret60_rank, config.long_window)

    out = pd.DataFrame(index=close.index)
    out.index.name = "trade_date"
    out["industry_count"] = industry_count
    out["active_industry_count"] = amount.gt(0).sum(axis=1)
    out["active_industry_ratio"] = out["active_industry_count"] / out["industry_count"].replace(0, np.nan)
    out["up_industry_ratio_1d"] = ret1.gt(0).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["down_industry_ratio_1d"] = ret1.lt(0).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["positive_momentum_5d_ratio"] = ret5.gt(0).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["positive_momentum_20d_ratio"] = ret20.gt(0).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["positive_momentum_60d_ratio"] = ret60.gt(0).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["above_ma20_ratio"] = close.gt(ma20).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["above_ma60_ratio"] = close.gt(ma60).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["above_ma120_ratio"] = close.gt(ma120).sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["industry_equal_momentum_5d"] = ret5.mean(axis=1)
    out["industry_equal_momentum_20d"] = ret20.mean(axis=1)
    out["industry_equal_momentum_60d"] = ret60.mean(axis=1)
    out["industry_equal_momentum_120d"] = ret120.mean(axis=1)
    out["industry_ret20_dispersion"] = ret20.std(axis=1)
    out["industry_ret60_dispersion"] = ret60.std(axis=1)
    out["industry_ret20_iqr"] = ret20.quantile(0.75, axis=1) - ret20.quantile(0.25, axis=1)
    out["industry_ret60_iqr"] = ret60.quantile(0.75, axis=1) - ret60.quantile(0.25, axis=1)
    out["top3_momentum_20d_mean"] = _row_top_mean(ret20, 3, largest=True)
    out["bottom3_momentum_20d_mean"] = _row_top_mean(ret20, 3, largest=False)
    out["leader_laggard_spread_20d"] = out["top3_momentum_20d_mean"] - out["bottom3_momentum_20d_mean"]
    out["top3_momentum_60d_mean"] = _row_top_mean(ret60, 3, largest=True)
    out["bottom3_momentum_60d_mean"] = _row_top_mean(ret60, 3, largest=False)
    out["leader_laggard_spread_60d"] = out["top3_momentum_60d_mean"] - out["bottom3_momentum_60d_mean"]
    out["amount_total_raw"] = amount_total
    out["amount_per_industry_raw"] = amount_total / out["industry_count"].replace(0, np.nan)
    out["amount_per_industry_log"] = np.log1p(out["amount_per_industry_raw"].clip(lower=0))
    out["volume_total_raw"] = volume.sum(axis=1)
    out["amount_hhi"] = amount_hhi
    out["top3_amount_share"] = amount_top3_share
    out["amount_expansion_60d_ratio"] = amount_expansion.sum(axis=1) / out["industry_count"].replace(0, np.nan)
    out["zero_amount_ratio"] = zero_amount_ratio
    out["median_range_ratio"] = range_ratio.median(axis=1)
    out["rank_persistence_20d"] = rank_persistence_20
    out["rank_persistence_60d"] = rank_persistence_60
    out["breadth_change_5d"] = out["above_ma60_ratio"] - out["above_ma60_ratio"].shift(config.short_window)
    out["breadth_change_20d"] = out["above_ma60_ratio"] - out["above_ma60_ratio"].shift(config.medium_window)
    out["positive_momentum_change_20d"] = out["positive_momentum_20d_ratio"] - out["positive_momentum_20d_ratio"].shift(config.medium_window)

    z_cols = [
        "above_ma60_ratio",
        "positive_momentum_20d_ratio",
        "breadth_change_20d",
        "industry_equal_momentum_20d",
        "industry_ret20_dispersion",
        "industry_ret60_dispersion",
        "leader_laggard_spread_20d",
        "amount_hhi",
        "top3_amount_share",
        "amount_per_industry_log",
        "amount_expansion_60d_ratio",
        "zero_amount_ratio",
        "median_range_ratio",
        "rank_persistence_20d",
        "rank_persistence_60d",
    ]
    for col in z_cols:
        out[f"{col}_prior_pctile_{config.trailing_window}d"] = _trailing_percentile(out[col], config.trailing_window, config.min_history)
        out[f"{col}_z_{config.trailing_window}d"] = _rolling_zscore(out[col], config.trailing_window, config.min_history)

    out["industry_breadth_thrust_score"] = (
        _score_component(out[f"above_ma60_ratio_z_{config.trailing_window}d"])
        + _score_component(out[f"positive_momentum_20d_ratio_z_{config.trailing_window}d"])
        + _score_component(out[f"breadth_change_20d_z_{config.trailing_window}d"])
        + _score_component(out[f"industry_equal_momentum_20d_z_{config.trailing_window}d"])
    ) / 4.0
    out["industry_dispersion_risk_score"] = (
        _score_component(out[f"industry_ret20_dispersion_z_{config.trailing_window}d"])
        + _score_component(out[f"industry_ret60_dispersion_z_{config.trailing_window}d"])
        + _score_component(out[f"leader_laggard_spread_20d_z_{config.trailing_window}d"])
        + _score_component(out[f"amount_hhi_z_{config.trailing_window}d"])
        + _score_component(out[f"top3_amount_share_z_{config.trailing_window}d"])
    ) / 5.0
    out["industry_liquidity_breadth_score"] = (
        _score_component(out[f"amount_per_industry_log_z_{config.trailing_window}d"])
        + _score_component(out[f"amount_expansion_60d_ratio_z_{config.trailing_window}d"])
        - _score_component(out[f"zero_amount_ratio_z_{config.trailing_window}d"])
    ) / 3.0
    out["industry_rotation_persistence_score"] = (
        _score_component(out[f"rank_persistence_20d_z_{config.trailing_window}d"])
        + _score_component(out[f"rank_persistence_60d_z_{config.trailing_window}d"])
        + _score_component(out[f"positive_momentum_20d_ratio_z_{config.trailing_window}d"])
        - _score_component(out[f"industry_ret20_dispersion_z_{config.trailing_window}d"])
    ) / 4.0
    out["industry_breadth_dispersion_score"] = (
        0.40 * out["industry_breadth_thrust_score"]
        + 0.25 * out["industry_liquidity_breadth_score"]
        + 0.20 * out["industry_rotation_persistence_score"]
        - 0.15 * out["industry_dispersion_risk_score"]
    )

    out = out.reset_index()
    out["source_trade_date"] = out["trade_date"]
    out["available_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    next_trade_date = out["trade_date"].shift(-1)
    out["signal_use_date"] = pd.to_datetime(next_trade_date, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    out["signal_timing"] = "after_close_for_next_trade_date"
    out["breadth_state"] = np.select(
        [
            (out["above_ma60_ratio"] >= 0.65) & (out["positive_momentum_20d_ratio"] >= 0.55),
            (out["above_ma60_ratio"] <= 0.35),
            (out["breadth_change_20d"] > 0.10) & (out["positive_momentum_20d_ratio"] >= 0.45),
        ],
        ["broad_participation", "narrow_participation", "breadth_repair"],
        default="neutral_breadth",
    )
    out["dispersion_state"] = np.select(
        [
            out[f"industry_ret20_dispersion_prior_pctile_{config.trailing_window}d"] >= 0.80,
            out[f"industry_ret20_dispersion_prior_pctile_{config.trailing_window}d"] <= 0.20,
        ],
        ["high_dispersion", "low_dispersion"],
        default="normal_dispersion",
    )
    out["leadership_state"] = np.select(
        [
            (out[f"top3_amount_share_prior_pctile_{config.trailing_window}d"] >= 0.80)
            | (out[f"amount_hhi_prior_pctile_{config.trailing_window}d"] >= 0.80),
            (out[f"top3_amount_share_prior_pctile_{config.trailing_window}d"] <= 0.30)
            & (out[f"amount_hhi_prior_pctile_{config.trailing_window}d"] <= 0.30),
        ],
        ["crowded_leadership", "distributed_leadership"],
        default="normal_leadership",
    )
    out["rotation_state"] = np.select(
        [
            (out["rank_persistence_20d"] >= 0.35) & (out["rank_persistence_60d"] >= 0.15),
            (out["rank_persistence_20d"] <= 0.0) & (out["rank_persistence_60d"] <= 0.0),
        ],
        ["persistent_rotation", "unstable_rotation"],
        default="mixed_rotation",
    )
    out["composite_industry_state"] = np.select(
        [
            (out["breadth_state"] == "narrow_participation") & (out["leadership_state"] == "crowded_leadership"),
            (out["breadth_state"] == "breadth_repair") & (out["industry_breadth_thrust_score"] > 0),
            (out["breadth_state"] == "broad_participation") & (out["rotation_state"] == "persistent_rotation"),
            (out["dispersion_state"] == "high_dispersion") & (out["breadth_state"] != "narrow_participation"),
        ],
        ["fragile_narrow_leadership", "breadth_repair_thrust", "broad_persistent_participation", "rotation_opportunity"],
        default="neutral_industry_structure",
    )
    out["history_sufficient"] = out[f"above_ma60_ratio_prior_pctile_{config.trailing_window}d"].notna()
    out["source_scope"] = f"sw_{config.industry_level}_industry_index_price_volume"
    out["feature_family"] = "industry_breadth_dispersion"
    out["component_snapshot_used"] = False
    out["portfolio_harness_allowed"] = False
    out["model_promotion_allowed"] = False
    out["official_total_return_evidence"] = False
    out["price_adjustment_boundary"] = "index_provider_price_series_only"

    preferred = [
        "trade_date",
        "source_trade_date",
        "available_date",
        "signal_use_date",
        "signal_timing",
        "industry_count",
        "active_industry_count",
        "active_industry_ratio",
        "up_industry_ratio_1d",
        "down_industry_ratio_1d",
        "positive_momentum_5d_ratio",
        "positive_momentum_20d_ratio",
        "positive_momentum_60d_ratio",
        "above_ma20_ratio",
        "above_ma60_ratio",
        "above_ma120_ratio",
        "breadth_change_5d",
        "breadth_change_20d",
        "positive_momentum_change_20d",
        "industry_equal_momentum_5d",
        "industry_equal_momentum_20d",
        "industry_equal_momentum_60d",
        "industry_equal_momentum_120d",
        "industry_ret20_dispersion",
        "industry_ret60_dispersion",
        "industry_ret20_iqr",
        "industry_ret60_iqr",
        "leader_laggard_spread_20d",
        "leader_laggard_spread_60d",
        "amount_total_raw",
        "amount_per_industry_raw",
        "amount_per_industry_log",
        "volume_total_raw",
        "amount_hhi",
        "top3_amount_share",
        "amount_expansion_60d_ratio",
        "zero_amount_ratio",
        "median_range_ratio",
        "rank_persistence_20d",
        "rank_persistence_60d",
        "industry_breadth_thrust_score",
        "industry_dispersion_risk_score",
        "industry_liquidity_breadth_score",
        "industry_rotation_persistence_score",
        "industry_breadth_dispersion_score",
        "history_sufficient",
        "breadth_state",
        "dispersion_state",
        "leadership_state",
        "rotation_state",
        "composite_industry_state",
        "source_scope",
        "feature_family",
        "component_snapshot_used",
        "portfolio_harness_allowed",
        "model_promotion_allowed",
        "official_total_return_evidence",
        "price_adjustment_boundary",
    ]
    extras = [col for col in out.columns if col.endswith(f"_prior_pctile_{config.trailing_window}d") or col.endswith(f"_z_{config.trailing_window}d")]
    return out[[col for col in preferred + extras if col in out.columns]].reset_index(drop=True)


def build_feature_dictionary(config: IndustryBreadthDispersionConfig) -> pd.DataFrame:
    rows = [
        {
            "feature": "above_ma60_ratio",
            "definition": "Share of selected SW industry indices whose close is above its own 60-day moving average.",
            "allowed_use": "industry participation breadth diagnostic",
            "forbidden_use": "same-day executable rule or model promotion without validation",
        },
        {
            "feature": "positive_momentum_20d_ratio",
            "definition": "Share of selected SW industry indices with positive trailing 20-day price momentum.",
            "allowed_use": "industry breadth thrust feature",
            "forbidden_use": "official broad-market return label",
        },
        {
            "feature": "industry_ret20_dispersion",
            "definition": "Cross-industry dispersion of trailing 20-day industry price momentum.",
            "allowed_use": "rotation opportunity or concentration-risk diagnostic",
            "forbidden_use": "portfolio risk estimate without later harness",
        },
        {
            "feature": "top3_amount_share",
            "definition": "Share of SW industry raw amount concentrated in the three largest amount industries.",
            "allowed_use": "leadership crowding diagnostic",
            "forbidden_use": "capacity estimate without provider unit review",
        },
        {
            "feature": "rank_persistence_20d",
            "definition": "Cross-sectional rank persistence of 20-day industry momentum versus 20 trading days earlier.",
            "allowed_use": "rotation persistence feature",
            "forbidden_use": "claim that leaders are investable winners without validation",
        },
        {
            "feature": "industry_breadth_dispersion_score",
            "definition": "Composite score using breadth thrust, liquidity breadth, rotation persistence, and dispersion risk.",
            "allowed_use": "candidate feature for later guarded validation only",
            "forbidden_use": "default model, portfolio harness, or performance claim in V3.68",
        },
    ]
    for row in rows:
        row["source_scope"] = f"sw_{config.industry_level}_industry_index_price_volume"
        row["timing"] = "after_close_for_next_trade_date"
        row["trailing_window"] = config.trailing_window
    return pd.DataFrame(rows)


def build_signal_validation_plan(config: IndustryBreadthDispersionConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "v3_66_industry_breadth_thrust_v1",
                "feature_columns": "industry_breadth_thrust_score,above_ma60_ratio,positive_momentum_20d_ratio,breadth_change_20d",
                "hypothesis": "Broad industry participation after weak breadth is more durable than narrow index rebound.",
                "next_validation": "Join to permitted market or industry index label source with after-close lag and purged walk-forward splits.",
                "blocked_now": "portfolio_harness;model_promotion;same_branch_threshold_tuning",
            },
            {
                "signal_id": "v3_66_industry_dispersion_risk_v1",
                "feature_columns": "industry_dispersion_risk_score,industry_ret20_dispersion,leader_laggard_spread_20d,top3_amount_share",
                "hypothesis": "High dispersion and concentrated leadership can signal fragility or rotation opportunity depending on breadth state.",
                "next_validation": "Validate state-conditioned outcomes rather than unconditional thresholds.",
                "blocked_now": "portfolio_harness;model_promotion;official_total_return_claim",
            },
            {
                "signal_id": "v3_66_industry_rotation_persistence_v1",
                "feature_columns": "industry_rotation_persistence_score,rank_persistence_20d,rank_persistence_60d,amount_expansion_60d_ratio",
                "hypothesis": "Persistent leadership confirmed by activity breadth may be more stable than one-day industry rallies.",
                "next_validation": "Test with independent holdout windows and turnover-aware implementation constraints.",
                "blocked_now": "portfolio_harness;model_promotion;current_component_snapshot_use",
            },
        ]
    )


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "industry_breadth_dispersion_feature_layer",
                "produced": True,
                "blocked": False,
                "reason": "V3.68 creates point-in-time industry index features only",
            },
            {
                "result_type": "portfolio_harness",
                "produced": False,
                "blocked": True,
                "reason": "feature layer has no positions, trades, or performance series",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "requires a later guarded validation task",
            },
            {
                "result_type": "current_component_snapshot",
                "produced": False,
                "blocked": True,
                "reason": "V3.68 uses industry index history and classification metadata only",
            },
        ]
    )


def build_source_quality_checks(quality: pd.DataFrame, panel: pd.DataFrame, config: IndustryBreadthDispersionConfig) -> pd.DataFrame:
    available_rows = int((pd.to_numeric(quality.get("close_nonnull", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if not quality.empty else 0
    long_history_rows = int((pd.to_numeric(quality.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0) >= config.min_feature_rows).sum()) if not quality.empty else 0
    min_core_long_history = min(15, config.min_industry_count)
    latest = panel.sort_values("trade_date").tail(1)
    return pd.DataFrame(
        [
            {
                "check": "minimum_industry_files_available",
                "status": _status(available_rows >= config.min_industry_count),
                "detail": f"available_rows={available_rows};min={config.min_industry_count}",
            },
            {
                "check": "long_history_core_industries_present",
                "status": _status(long_history_rows >= min_core_long_history),
                "detail": f"long_history_rows={long_history_rows};min_core={min_core_long_history}",
            },
            {
                "check": "latest_industry_count_large_enough",
                "status": _status(not latest.empty and int(latest["industry_count"].iloc[0]) >= config.min_industry_count),
                "detail": f"latest_industry_count={int(latest['industry_count'].iloc[0]) if not latest.empty else 0}",
            },
            {
                "check": "no_industry_source_missing_or_empty",
                "status": _status(available_rows == int(quality.shape[0])),
                "detail": f"min_rows={int(quality['rows'].astype(int).min()) if not quality.empty else 0}",
            },
        ]
    )


def build_feature_quality_checks(
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    source_checks: pd.DataFrame,
    config: IndustryBreadthDispersionConfig,
) -> pd.DataFrame:
    forbidden_output_terms = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(panel.columns).lower()})
    score_cols = [
        "industry_breadth_thrust_score",
        "industry_dispersion_risk_score",
        "industry_liquidity_breadth_score",
        "industry_rotation_persistence_score",
        "industry_breadth_dispersion_score",
    ]
    finite_scores = panel.loc[panel["history_sufficient"].astype(bool), score_cols].replace([np.inf, -np.inf], np.nan)
    latest = panel.sort_values("trade_date").tail(1)
    snapshot_any = _bool_series(panel["component_snapshot_used"]).any() if "component_snapshot_used" in panel else True
    model_any = _bool_series(panel["model_promotion_allowed"]).any() if "model_promotion_allowed" in panel else True
    portfolio_any = _bool_series(panel["portfolio_harness_allowed"]).any() if "portfolio_harness_allowed" in panel else True
    official_any = _bool_series(panel["official_total_return_evidence"]).any() if "official_total_return_evidence" in panel else True
    return pd.DataFrame(
        [
            {
                "check": "minimum_feature_rows_present",
                "status": _status(len(panel) >= config.min_feature_rows),
                "detail": f"rows={len(panel)};min={config.min_feature_rows}",
            },
            {
                "check": "source_quality_checks_passed",
                "status": _status(source_checks["status"].eq("pass").all()),
                "detail": ";".join(source_checks.loc[source_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "latest_industry_count_large_enough",
                "status": _status(not latest.empty and int(latest["industry_count"].iloc[0]) >= config.min_industry_count),
                "detail": f"latest_industry_count={int(latest['industry_count'].iloc[0]) if not latest.empty else 0}",
            },
            {
                "check": "history_sufficient_after_min_history",
                "status": _status(int(panel["history_sufficient"].sum()) >= len(panel) - config.min_history - config.slow_window - 3),
                "detail": f"history_sufficient={int(panel['history_sufficient'].sum())}",
            },
            {
                "check": "feature_timing_has_next_trade_date",
                "status": _status(panel["signal_use_date"].astype(str).str.len().gt(0).sum() >= len(panel) - 1),
                "detail": f"missing={int((~panel['signal_use_date'].astype(str).str.len().gt(0)).sum())}",
            },
            {
                "check": "score_columns_finite_when_history_sufficient",
                "status": _status(not finite_scores.empty and finite_scores.notna().all().all()),
                "detail": ",".join(score_cols),
            },
            {
                "check": "component_snapshots_not_used",
                "status": _status(not snapshot_any),
                "detail": f"component_snapshot_any={bool(snapshot_any)}",
            },
            {
                "check": "promotion_and_official_evidence_blocked",
                "status": _status(not model_any and not portfolio_any and not official_any),
                "detail": f"model={bool(model_any)};portfolio={bool(portfolio_any)};official={bool(official_any)}",
            },
            {
                "check": "forbidden_promotion_columns_absent",
                "status": _status(not forbidden_output_terms),
                "detail": ",".join(forbidden_output_terms),
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    source_checks: pd.DataFrame,
    quality_checks: pd.DataFrame,
    no_promotion: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    column_text = " ".join(output_column_names).lower()
    hits = sorted(term for term in forbidden if term in column_text)
    blocked_result_types = {"portfolio_harness", "model_promotion", "current_component_snapshot"}
    blocked_correctly = not no_promotion.loc[no_promotion["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if input_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "source_checks_passed",
                "status": "pass" if source_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(source_checks.loc[source_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "feature_quality_checks_passed",
                "status": "pass" if quality_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(quality_checks.loc[quality_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "promotion_outputs_blocked",
                "status": "pass" if blocked_correctly and not hits else "fail",
                "detail": ",".join(hits),
            },
        ]
    )


def build_feature_contract(config: IndustryBreadthDispersionConfig) -> str:
    return "\n".join(
        [
            "# V3.68 Industry Breadth And Dispersion Feature Contract",
            "",
            "## Scope",
            "",
            "- Source: SW industry index daily price and amount history.",
            "- Universe: configured SW industry level only; default is level-1 industries.",
            "- Timing: features are available after the source trade-date close and may only be used from the next trade date.",
            "",
            "## Allowed",
            "",
            "- Industry-index trailing momentum, moving-average breadth, amount breadth, dispersion, and rank persistence.",
            "- Prior-observation rolling z-scores and trailing percentiles.",
            "- State tags for later validation design.",
            "",
            "## Forbidden",
            "",
            "- Current component snapshots, latest weights, or current constituent mappings.",
            "- Portfolio harness, NAV, Sharpe, drawdown, annualized performance, or default model promotion.",
            "- Official total-return claims from provider price-index history.",
            "- Same-branch threshold tuning against V3.64/V3.65 proxy outcomes.",
            "",
            "## Required Next Gate",
            "",
            "A later validation task must join V3.68 features to a permitted label source with explicit after-close lag, purged walk-forward splits, and state-conditioned diagnostics.",
            f"",
            f"Configured trailing window: {config.trailing_window} trading days.",
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 24) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    validation_plan: pd.DataFrame,
    input_checks: pd.DataFrame,
    source_checks: pd.DataFrame,
    quality_checks: pd.DataFrame,
    no_promotion: pd.DataFrame,
) -> str:
    latest = panel.sort_values("trade_date").tail(1).iloc[0]
    valid = panel.loc[panel["history_sufficient"].astype(bool)]
    lines = [
        "# V3.68 Industry Breadth And Dispersion Feature Layer",
        "",
        "## Decision",
        "",
        "- V3.68 builds a point-in-time feature layer from SW industry index history.",
        "- It covers industry breadth thrust, dispersion risk, liquidity breadth, and rotation persistence.",
        "- It does not produce portfolio outputs, official total-return evidence, or model promotion evidence.",
        "",
        "## Coverage",
        "",
        f"- Feature rows: `{len(panel)}`",
        f"- History-sufficient rows: `{len(valid)}`",
        f"- Date range: `{panel['trade_date'].min()}` to `{panel['trade_date'].max()}`",
        f"- Latest industry count: `{int(latest.industry_count)}`",
        "",
        "## Latest Snapshot",
        "",
        f"- Latest trade date: `{latest.trade_date}`",
        f"- Signal use date: `{latest.signal_use_date}`",
        f"- Above MA60 ratio: `{float(latest.above_ma60_ratio):.4f}`",
        f"- Positive 20d momentum ratio: `{float(latest.positive_momentum_20d_ratio):.4f}`",
        f"- Industry 20d dispersion: `{float(latest.industry_ret20_dispersion):.4f}`",
        f"- Industry breadth dispersion score: `{float(latest.industry_breadth_dispersion_score):.4f}`",
        f"- Composite state: `{latest.composite_industry_state}`",
        "",
        "## Feature Sample",
        "",
    ]
    lines.extend(
        markdown_table(
            panel.sort_values("trade_date").tail(12),
            [
                "trade_date",
                "signal_use_date",
                "above_ma60_ratio",
                "positive_momentum_20d_ratio",
                "industry_ret20_dispersion",
                "top3_amount_share",
                "rank_persistence_20d",
                "industry_breadth_thrust_score",
                "industry_dispersion_risk_score",
                "industry_breadth_dispersion_score",
                "composite_industry_state",
            ],
            max_rows=12,
        )
    )
    lines.extend(["", "## Feature Dictionary", ""])
    lines.extend(markdown_table(feature_dictionary, ["feature", "definition", "allowed_use", "forbidden_use"], max_rows=12))
    lines.extend(["", "## Signal Validation Plan", ""])
    lines.extend(markdown_table(validation_plan, ["signal_id", "feature_columns", "next_validation", "blocked_now"], max_rows=12))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Source Checks", ""])
    lines.extend(markdown_table(source_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Quality Checks", ""])
    lines.extend(markdown_table(quality_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## No Promotion Guard", ""])
    lines.extend(markdown_table(no_promotion, ["result_type", "produced", "blocked", "reason"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.69 can build the macro growth-liquidity feature layer from the macro PIT panel.",
            "- A later guarded validation task can test V3.67, V3.68, and V3.69 together against a permitted label source.",
            "- Do not use this feature layer as a trading strategy by itself.",
        ]
    )
    return "\n".join(lines)


def build_catalog(panel: pd.DataFrame, config: IndustryBreadthDispersionConfig) -> str:
    return "\n".join(
        [
            "# A-share Industry Breadth And Dispersion Feature Layer V3.68",
            "",
            "## Dataset Role",
            "",
            "V3.68 converts SW industry index daily history into industry breadth, dispersion, liquidity, and rotation-persistence features.",
            "",
            "## Governance",
            "",
            "- Uses industry index price and amount history only.",
            "- Current component snapshots and latest weights are blocked.",
            "- Portfolio outputs and default model promotion are blocked.",
            "- Feature timing is after-close for next trade-date research.",
            "",
            "## Produced Shape",
            "",
            f"- Feature rows: `{len(panel)}`",
            f"- Date range: `{panel['trade_date'].min()}` to `{panel['trade_date'].max()}`",
            f"- History-sufficient rows: `{int(panel['history_sufficient'].sum())}`",
            f"- Industry level: `{config.industry_level}`",
            f"- Trailing window: `{config.trailing_window}`",
        ]
    )
