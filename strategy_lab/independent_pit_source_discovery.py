"""Independent PIT source discovery for HIRSSM V3.66.

This module inventories local point-in-time candidate sources after V3.65
showed that repairing the same MARKET proxy branch has high overfit risk.
It produces research queues only: no portfolio harness, no performance claim,
and no model promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


@dataclass(frozen=True)
class IndependentPitSourceConfig:
    output_dir: Path
    catalog_path: Path
    v3_65_manifest_path: Path
    v3_65_decision_path: Path
    min_ready_priority_score: float
    max_feature_layer_queue_rows: int


SOURCE_SPECS: list[dict[str, Any]] = [
    {
        "source_id": "sw_industry_daily_history",
        "source_family": "industry_index_price_volume",
        "path": "data_raw/index/akshare_sw_industry/daily_sw",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["date"],
        "required_columns": ["date", "close", "volume", "amount", "index_code"],
        "source_role": "industry breadth, dispersion, leadership, and rotation feature layer",
        "pit_status": "pit_observed_market_history",
        "current_snapshot_risk": "none",
        "raw_adjustment_risk": "low_for_index_series",
        "independent_from_v3_65": True,
        "base_independence_score": 30,
    },
    {
        "source_id": "csindex_daily_history",
        "source_family": "broad_style_index_price_volume",
        "path": "data_raw/index/akshare_csindex/daily_csindex",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["date"],
        "required_columns": ["date", "close", "volume", "amount", "index_code"],
        "source_role": "style and broad-market regime feature layer",
        "pit_status": "pit_observed_market_history",
        "current_snapshot_risk": "none",
        "raw_adjustment_risk": "low_for_index_series",
        "independent_from_v3_65": False,
        "base_independence_score": 16,
    },
    {
        "source_id": "csindex_valuation_pe_history",
        "source_family": "style_valuation_history",
        "path": "data_raw/index/akshare_csindex/valuation_pe_lg",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["date"],
        "required_columns": ["date", "pe_ttm", "pe_ttm_median", "index_code"],
        "source_role": "valuation spread and valuation repair feature layer",
        "pit_status": "historical_series_provider_no_vintage",
        "current_snapshot_risk": "medium_provider_reconstructed_history",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 26,
    },
    {
        "source_id": "csindex_valuation_pb_history",
        "source_family": "style_valuation_history",
        "path": "data_raw/index/akshare_csindex/valuation_pb_lg",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["date"],
        "required_columns": ["date", "pb", "pb_median", "index_code"],
        "source_role": "valuation spread and balance-sheet valuation feature layer",
        "pit_status": "historical_series_provider_no_vintage",
        "current_snapshot_risk": "medium_provider_reconstructed_history",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 26,
    },
    {
        "source_id": "macro_pit_panel",
        "source_family": "macro_growth_liquidity_inflation",
        "path": "data_raw/macro/macro_pit_panel.csv",
        "path_kind": "file",
        "date_mode": "column",
        "date_columns": ["date", "available_date"],
        "required_columns": ["date", "available_date", "series_id", "value", "pit_quality"],
        "source_role": "macro growth, liquidity, inflation, commodity, and rate context",
        "pit_status": "pit_asof_release_or_daily_observation",
        "current_snapshot_risk": "low_with_available_date",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 28,
    },
    {
        "source_id": "tushare_daily_raw_market_partitions",
        "source_family": "stock_market_participation_breadth",
        "path": "data_raw/tushare_daily_only/v3_38/daily",
        "path_kind": "directory",
        "date_mode": "filename_trade_date",
        "date_columns": ["trade_date", "available_date"],
        "required_columns": ["ts_code", "trade_date", "pct_chg", "vol", "amount", "price_adjustment"],
        "source_role": "whole-market participation, liquidity breadth, limit crowding, and turnover breadth",
        "pit_status": "pit_observed_daily_partition",
        "current_snapshot_risk": "none",
        "raw_adjustment_risk": "high_for_stock_return_labels_low_for_breadth_counts",
        "independent_from_v3_65": True,
        "base_independence_score": 32,
    },
    {
        "source_id": "akshare_stock_qfq_sample",
        "source_family": "stock_sample_adjusted_price",
        "path": "data_raw/akshare/daily_qfq",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["date"],
        "required_columns": ["date", "adj_close", "turnover_pct", "asset"],
        "source_role": "sample stock adjusted price smoke source only",
        "pit_status": "sample_only_not_broad_universe",
        "current_snapshot_risk": "sample_scope_too_small",
        "raw_adjustment_risk": "qfq_adjusted_but_sample_only",
        "independent_from_v3_65": True,
        "base_independence_score": 12,
    },
    {
        "source_id": "akshare_financial_indicator_sample",
        "source_family": "stock_sample_financial_indicator",
        "path": "data_raw/akshare/financial_indicator",
        "path_kind": "directory",
        "date_mode": "column",
        "date_columns": ["报告期"],
        "required_columns": ["报告期", "净利润", "每股净资产", "asset"],
        "source_role": "sample financial statement smoke source only",
        "pit_status": "sample_only_no_announcement_available_date",
        "current_snapshot_risk": "sample_scope_and_release_date_missing",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 10,
    },
    {
        "source_id": "csindex_current_constituents",
        "source_family": "current_constituent_snapshot",
        "path": "data_raw/index/akshare_csindex/constituents_current",
        "path_kind": "directory",
        "date_mode": "none",
        "date_columns": [],
        "required_columns": [],
        "source_role": "current constituent lookup only",
        "pit_status": "blocked_current_snapshot",
        "current_snapshot_risk": "high_current_snapshot",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 4,
    },
    {
        "source_id": "csindex_latest_weights",
        "source_family": "latest_weight_snapshot",
        "path": "data_raw/index/akshare_csindex/weights_latest",
        "path_kind": "directory",
        "date_mode": "none",
        "date_columns": [],
        "required_columns": [],
        "source_role": "current or latest index weight lookup only",
        "pit_status": "blocked_latest_snapshot",
        "current_snapshot_risk": "high_latest_snapshot",
        "raw_adjustment_risk": "not_price_adjustment_source",
        "independent_from_v3_65": True,
        "base_independence_score": 4,
    },
]


SIGNAL_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "signal_id": "v3_66_industry_breadth_thrust_v1",
        "source_id": "sw_industry_daily_history",
        "signal_scope": "time_series_market_risk_budget",
        "horizon_bucket": "20d_60d",
        "formula_sketch": "share of SW industries above MA60, 20d breadth thrust, and new-high participation",
        "economic_logic": "Broad industry participation is less dependent on one market index and can separate durable rallies from narrow rebounds.",
        "failure_mode": "Breadth can peak near exhaustion and can lag after sharp policy turns.",
        "pit_requirements": "use daily industry observations with signal available after close; no component snapshots",
        "allowed_next_stage": "feature_layer_build",
        "priority_bias": 10,
    },
    {
        "signal_id": "v3_66_industry_dispersion_risk_v1",
        "source_id": "sw_industry_daily_history",
        "signal_scope": "time_series_market_risk_budget",
        "horizon_bucket": "20d_60d",
        "formula_sketch": "cross-industry dispersion of 20d and 60d returns plus leadership concentration",
        "economic_logic": "High dispersion and concentrated leadership often signal unstable rotation and weaker broad-market reward-to-risk.",
        "failure_mode": "Early bull markets can begin with high dispersion before breadth catches up.",
        "pit_requirements": "daily industry observations only; no current constituents",
        "allowed_next_stage": "feature_layer_build",
        "priority_bias": 8,
    },
    {
        "signal_id": "v3_66_industry_rotation_persistence_v1",
        "source_id": "sw_industry_daily_history",
        "signal_scope": "cross_sectional_industry_rotation",
        "horizon_bucket": "20d_60d",
        "formula_sketch": "risk-adjusted 60d industry momentum confirmed by amount trend and downside-vol control",
        "economic_logic": "Persistent industry leadership with liquidity confirmation may survive better than single-index MARKET proxies.",
        "failure_mode": "Crowded industry leaders reverse quickly in policy or earnings shocks.",
        "pit_requirements": "industry index price and amount only; no current component weights",
        "allowed_next_stage": "feature_layer_build",
        "priority_bias": 8,
    },
    {
        "signal_id": "v3_66_market_participation_breadth_v1",
        "source_id": "tushare_daily_raw_market_partitions",
        "signal_scope": "time_series_market_risk_budget",
        "horizon_bucket": "1d_5d_20d",
        "formula_sketch": "daily share of stocks rising, amount breadth, volume breadth, upper/lower tail crowding, and zero-amount share",
        "economic_logic": "Whole-market participation uses the broad daily-permission dataset and avoids relying on one index price proxy.",
        "failure_mode": "Raw unadjusted stock prices cannot be used as long-horizon return labels; breadth also needs IPO/suspension filters.",
        "pit_requirements": "use same-day close as next-session signal; no adjusted stock-return label generation",
        "allowed_next_stage": "feature_layer_build_with_adjustment_guard",
        "priority_bias": 12,
    },
    {
        "signal_id": "v3_66_style_valuation_spread_repair_v1",
        "source_id": "csindex_valuation_pe_history",
        "signal_scope": "cross_sectional_style_rotation",
        "horizon_bucket": "60d_120d",
        "formula_sketch": "PE/PB rolling percentile spread between 000300, 000905, 000852, 000016 and repair momentum",
        "economic_logic": "Relative valuation spread can create slower, more interpretable style-allocation signals than short MARKET proxy timing.",
        "failure_mode": "Provider reconstructed valuation history and structural value traps can mislead.",
        "pit_requirements": "treat historical valuation as research-grade until vintage availability is documented",
        "allowed_next_stage": "data_quality_review_then_feature_layer",
        "priority_bias": 4,
    },
    {
        "signal_id": "v3_66_macro_growth_liquidity_mix_v1",
        "source_id": "macro_pit_panel",
        "signal_scope": "time_series_market_regime",
        "horizon_bucket": "60d_120d",
        "formula_sketch": "PMI trend, M2/social-financing impulse, CPI/PPI pressure, and commodity stress using available_date as-of merge",
        "economic_logic": "Growth-liquidity-inflation context is slower and less mechanically tied to prior MARKET price-proxy failures.",
        "failure_mode": "Macro data revisions and release-lag approximation can weaken timing precision.",
        "pit_requirements": "strict available_date merge; no same-month unreleased values",
        "allowed_next_stage": "feature_layer_build",
        "priority_bias": 6,
    },
    {
        "signal_id": "v3_66_financial_quality_breadth_v1",
        "source_id": "akshare_financial_indicator_sample",
        "signal_scope": "cross_sectional_stock_quality",
        "horizon_bucket": "quarterly",
        "formula_sketch": "ROE, profit growth, cash-flow quality, leverage, and industry aggregation after broad universe and announcement dates exist",
        "economic_logic": "Fundamental quality breadth could diversify away from pure price and macro signals.",
        "failure_mode": "Current sample has only a few stocks and lacks announcement available dates.",
        "pit_requirements": "requires broad coverage plus announcement_date or conservative delay rule",
        "allowed_next_stage": "blocked_until_data_expansion",
        "priority_bias": -8,
    },
]


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _to_date_text(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.strftime("%Y-%m-%d")


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _read_sample(path: Path, nrows: int = 5) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _csv_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return []


def _date_from_filename(path: Path) -> str:
    text = path.stem
    if "trade_date=" in text:
        raw = text.split("trade_date=", 1)[1][:8]
        if raw.isdigit():
            return _to_date_text(raw)
    return ""


def _date_range_from_files(files: list[Path], spec: dict[str, Any]) -> tuple[str, str]:
    dates: list[str] = []
    if spec["date_mode"] == "filename_trade_date":
        dates = [_date_from_filename(path) for path in files]
        dates = [date for date in dates if date]
    elif spec["date_mode"] == "column":
        for path in files:
            cols = list(spec.get("date_columns", []))
            if not cols:
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in cols, low_memory=False)
            except Exception:
                continue
            for col in cols:
                if col in frame.columns:
                    clean = pd.to_datetime(frame[col], errors="coerce").dropna()
                    if not clean.empty:
                        dates.append(clean.min().strftime("%Y-%m-%d"))
                        dates.append(clean.max().strftime("%Y-%m-%d"))
    if not dates:
        return "", ""
    clean_dates = sorted(dates)
    return clean_dates[0], clean_dates[-1]


def _required_column_status(files: list[Path], required: list[str]) -> tuple[bool, str]:
    if not required:
        return True, ""
    if not files:
        return False, ",".join(required)
    sample = _read_sample(files[0], nrows=1)
    missing = sorted(set(required).difference(sample.columns))
    return not missing, ",".join(missing)


def _score_source(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    score = 0
    if row["exists"]:
        score += 15
    if row["file_count"] > 0:
        score += 10
    if row["row_count_estimate"] >= 1000:
        score += 15
    if row["row_count_estimate"] >= 100000:
        score += 8
    if row["start_date"] and row["start_date"] <= "2005-12-31":
        score += 10
    if row["end_date"] and row["end_date"] >= "2025-01-01":
        score += 10
    if row["required_columns_ok"]:
        score += 12
    pit_status = spec["pit_status"]
    if pit_status in {"pit_observed_market_history", "pit_asof_release_or_daily_observation", "pit_observed_daily_partition"}:
        score += 15
    elif pit_status == "historical_series_provider_no_vintage":
        score += 8
    elif "blocked" in pit_status:
        score -= 30
    elif "sample_only" in pit_status:
        score -= 12
    if "high" in spec["current_snapshot_risk"]:
        score -= 25
    if "sample" in spec["current_snapshot_risk"]:
        score -= 10
    if "high_for_stock_return_labels" in spec["raw_adjustment_risk"]:
        score -= 5
    data_quality_score = max(0, min(100, score))
    independence_score = int(spec["base_independence_score"])
    priority_score = max(0, min(100, data_quality_score + independence_score))
    if "blocked" in pit_status:
        allowed_stage = "blocked_historical_use"
        next_action = "acquire_historical_vintages_before_factor_use"
    elif "sample_only" in pit_status:
        allowed_stage = "sample_only_repair_first"
        next_action = "expand_universe_and_add_available_date_before_validation"
    elif spec["source_id"] == "tushare_daily_raw_market_partitions":
        allowed_stage = "ready_for_nonreturn_breadth_feature_layer"
        next_action = "build market participation breadth features with raw-adjustment guard"
    elif pit_status == "historical_series_provider_no_vintage":
        allowed_stage = "research_ready_requires_vintage_caveat"
        next_action = "document provider history and keep observation before validation"
    else:
        allowed_stage = "ready_for_feature_layer"
        next_action = "build feature layer then run guarded signal validation"
    return {
        "data_quality_score": data_quality_score,
        "independence_score": independence_score,
        "priority_score": priority_score,
        "allowed_stage": allowed_stage,
        "next_action": next_action,
    }


def build_source_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec["path"]
        files = _csv_files(path)
        nonempty = [file for file in files if _line_count(file) > 0]
        row_count = sum(_line_count(file) for file in files)
        start_date, end_date = _date_range_from_files(files, spec)
        columns_ok, missing_columns = _required_column_status(nonempty or files, list(spec.get("required_columns", [])))
        row: dict[str, Any] = {
            "source_id": spec["source_id"],
            "source_family": spec["source_family"],
            "path": spec["path"],
            "exists": path.exists(),
            "file_count": len(files),
            "nonempty_file_count": len(nonempty),
            "row_count_estimate": int(row_count),
            "start_date": start_date,
            "end_date": end_date,
            "required_columns_ok": columns_ok,
            "missing_required_columns": missing_columns,
            "source_role": spec["source_role"],
            "pit_status": spec["pit_status"],
            "current_snapshot_risk": spec["current_snapshot_risk"],
            "raw_adjustment_risk": spec["raw_adjustment_risk"],
            "independent_from_v3_65": bool(spec["independent_from_v3_65"]),
        }
        row.update(_score_source(row, spec))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["priority_score", "source_id"], ascending=[False, True]).reset_index(drop=True)


def build_signal_blueprints(inventory: pd.DataFrame) -> pd.DataFrame:
    source_map = inventory.set_index("source_id").to_dict("index") if not inventory.empty else {}
    rows: list[dict[str, Any]] = []
    for spec in SIGNAL_BLUEPRINTS:
        source = source_map.get(spec["source_id"], {})
        source_priority = float(source.get("priority_score", 0))
        source_stage = str(source.get("allowed_stage", "missing_source"))
        if spec["allowed_next_stage"].startswith("blocked") or source_stage.startswith("blocked") or source_stage.startswith("sample"):
            research_status = "blocked_or_repair_first"
            allowed_for_v3_67 = False
            blocked_reason = source.get("next_action", "source_not_ready")
        elif "vintage_caveat" in source_stage:
            research_status = "observation_until_vintage_review"
            allowed_for_v3_67 = False
            blocked_reason = "valuation history needs stricter vintage documentation"
        else:
            research_status = "candidate_for_feature_layer"
            allowed_for_v3_67 = True
            blocked_reason = ""
        priority_score = max(0, min(100, source_priority + float(spec["priority_bias"])))
        rows.append(
            {
                "signal_id": spec["signal_id"],
                "source_id": spec["source_id"],
                "source_family": source.get("source_family", ""),
                "signal_scope": spec["signal_scope"],
                "horizon_bucket": spec["horizon_bucket"],
                "formula_sketch": spec["formula_sketch"],
                "economic_logic": spec["economic_logic"],
                "failure_mode": spec["failure_mode"],
                "pit_requirements": spec["pit_requirements"],
                "source_allowed_stage": source_stage,
                "research_status": research_status,
                "allowed_for_v3_67_feature_layer": bool(allowed_for_v3_67),
                "blocked_reason": blocked_reason,
                "priority_score": priority_score,
                "model_promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["allowed_for_v3_67_feature_layer", "priority_score"], ascending=[False, False]).reset_index(drop=True)


def build_v3_65_branch_policy(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(
            [
                {
                    "policy_item": "v3_65_decisions_missing",
                    "affected_rows": 0,
                    "policy_decision": "block_same_branch_repair_until_decisions_exist",
                    "allowed_next_stage": "blocked",
                    "reason": "V3.65 retire/repair queue was not readable",
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for action, group in decisions.groupby("final_action", dropna=False):
        if str(action) == "repair_with_predeclared_state_filter_only":
            policy_decision = "limited_retest_backlog_not_main_v3_66_path"
            allowed_next_stage = "separate_predeclared_state_filter_task_only"
        elif str(action).startswith("retire"):
            policy_decision = "retire_from_proxy_branch"
            allowed_next_stage = "blocked_until_new_label_or_source"
        else:
            policy_decision = "observation_only"
            allowed_next_stage = "do_not_retest_in_v3_66"
        rows.append(
            {
                "policy_item": str(action),
                "affected_rows": int(len(group)),
                "policy_decision": policy_decision,
                "allowed_next_stage": allowed_next_stage,
                "reason": "V3.66 prioritizes independent PIT source discovery over same-branch tuning",
            }
        )
    return pd.DataFrame(rows).sort_values("policy_item").reset_index(drop=True)


def build_validation_queue(blueprints: pd.DataFrame, inventory: pd.DataFrame, config: IndependentPitSourceConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = blueprints.loc[
        blueprints["allowed_for_v3_67_feature_layer"].astype(bool)
        & (pd.to_numeric(blueprints["priority_score"], errors="coerce") >= config.min_ready_priority_score)
    ].copy()
    source_order = {
        "tushare_daily_raw_market_partitions": 0,
        "sw_industry_daily_history": 1,
        "macro_pit_panel": 2,
    }
    eligible["queue_order"] = eligible["source_id"].map(source_order).fillna(9).astype(int)
    eligible = eligible.sort_values(["queue_order", "priority_score", "signal_id"], ascending=[True, False, True]).head(config.max_feature_layer_queue_rows)
    for i, row in enumerate(eligible.itertuples(index=False), start=1):
        if row.source_id == "tushare_daily_raw_market_partitions":
            version_hint = "V3.67"
            assigned_agent = "data_steward"
            required_outputs = "market_participation_breadth_feature_panel.csv;feature_contract.md"
            success_gate = "features use only current-or-prior daily partitions and block stock return labels"
        elif row.source_id == "sw_industry_daily_history":
            version_hint = "V3.68"
            assigned_agent = "factor_researcher"
            required_outputs = "industry_breadth_dispersion_feature_panel.csv;signal_validation_plan.csv"
            success_gate = "features are industry-index based and exclude current component snapshots"
        elif row.source_id == "macro_pit_panel":
            version_hint = "V3.69"
            assigned_agent = "factor_researcher"
            required_outputs = "macro_asof_feature_panel.csv;asof_join_checks.csv"
            success_gate = "all macro values are joined by available_date only"
        else:
            version_hint = "V3.70"
            assigned_agent = "factor_researcher"
            required_outputs = "feature_panel.csv;source_quality_checks.csv"
            success_gate = "source-specific PIT checks pass"
        version_slug = version_hint.lower().replace(".", "_")
        rows.append(
            {
                "priority": i,
                "task_id": f"20260529_{version_slug}_{row.signal_id}",
                "version_hint": version_hint,
                "assigned_agent": assigned_agent,
                "signal_id": row.signal_id,
                "source_id": row.source_id,
                "objective": f"Build a PIT feature layer for {row.signal_id}; no portfolio harness.",
                "allowed_inputs": str(inventory.loc[inventory["source_id"].eq(row.source_id), "path"].iloc[0]) if not inventory.empty else "",
                "required_outputs": required_outputs,
                "success_gate": success_gate,
                "forbidden": "no portfolio harness; no model promotion; no official total-return claim; no same-branch threshold tuning",
            }
        )
    return pd.DataFrame(rows)


def build_blocked_data_requests(inventory: pd.DataFrame) -> pd.DataFrame:
    requests = [
        {
            "request_id": "historical_index_constituents_vintage",
            "provider_hint": "JoinQuant or paid Tushare/index vendor",
            "needed_for": "historical industry/stock look-through and constituent-aware factors",
            "current_blocker": "current constituents cannot be used as historical point-in-time membership",
            "minimum_fields": "index_code, constituent_code, in_date, out_date, weight_available_date",
            "priority": "high",
        },
        {
            "request_id": "historical_index_weight_vintage",
            "provider_hint": "CSI/JoinQuant/Tushare premium if available",
            "needed_for": "historical index exposure, crowding, and look-through valuation",
            "current_blocker": "latest weights are snapshot-only",
            "minimum_fields": "index_code, constituent_code, weight, trade_date, available_date",
            "priority": "high",
        },
        {
            "request_id": "stock_adjustment_factor_full_universe",
            "provider_hint": "Tushare adj_factor if permission is later available; JoinQuant price panel if available",
            "needed_for": "stock-level returns beyond breadth counts",
            "current_blocker": "daily permission currently gives raw prices only",
            "minimum_fields": "ts_code, trade_date, adj_factor, available_date",
            "priority": "high",
        },
        {
            "request_id": "financial_statement_release_dates_full_universe",
            "provider_hint": "JoinQuant finance tables or Tushare income/balancesheet/cashflow with announcement date",
            "needed_for": "quality, profitability, accrual, and dividend-factor research",
            "current_blocker": "sample financial indicators lack broad universe and release dates",
            "minimum_fields": "ts_code, report_period, announcement_date, statement_fields",
            "priority": "medium",
        },
        {
            "request_id": "official_total_return_index_labels",
            "provider_hint": "CSI total return index or dividend-inclusive vendor source",
            "needed_for": "investable validation of dividend-sensitive index strategies",
            "current_blocker": "current labels are price-index proxy only",
            "minimum_fields": "index_code, trade_date, total_return_close, source_timestamp",
            "priority": "medium",
        },
    ]
    blocked_sources = set(inventory.loc[inventory["allowed_stage"].astype(str).str.startswith("blocked"), "source_id"]) if not inventory.empty else set()
    for item in requests:
        item["linked_blocked_sources"] = ",".join(sorted(blocked_sources))
    return pd.DataFrame(requests)


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "independent_source_discovery",
                "produced": True,
                "blocked": False,
                "reason": "V3.66 inventories sources and queues feature-layer tasks only",
            },
            {
                "result_type": "portfolio_harness",
                "produced": False,
                "blocked": True,
                "reason": "feature discovery does not create positions or trades",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "candidate sources require later PIT feature build and walk-forward validation",
            },
            {
                "result_type": "official_total_return_review",
                "produced": False,
                "blocked": True,
                "reason": "no new official label source is introduced in V3.66",
            },
        ]
    )


def validate_inputs(v3_65_manifest: dict[str, Any], v3_65_decisions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "v3_65_manifest_self_check_passed",
                "status": _status(bool(v3_65_manifest.get("self_check_pass"))),
                "detail": f"self_check={v3_65_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_65_decisions_present",
                "status": _status(not v3_65_decisions.empty),
                "detail": f"rows={len(v3_65_decisions)}",
            },
            {
                "check": "v3_65_model_promotion_blocked",
                "status": _status(str(v3_65_manifest.get("metrics", {}).get("model_promotion_status")) == "blocked"),
                "detail": str(v3_65_manifest.get("metrics", {}).get("model_promotion_status")),
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    inventory: pd.DataFrame,
    blueprints: pd.DataFrame,
    queue: pd.DataFrame,
    branch_policy: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    blocked_result_types = {"portfolio_harness", "model_promotion", "official_total_return_review"}
    blocked_correctly = not guard.loc[guard["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    current_snapshot_ready = inventory.loc[
        inventory["pit_status"].astype(str).str.contains("snapshot", case=False, na=False)
        & inventory["allowed_stage"].astype(str).str.contains("ready", case=False, na=False)
    ]
    same_branch_main = branch_policy.loc[branch_policy["allowed_next_stage"].astype(str).eq("do_not_retest_in_v3_66")]
    ready_candidates = int(blueprints["allowed_for_v3_67_feature_layer"].astype(bool).sum()) if not blueprints.empty else 0
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if input_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "source_inventory_produced",
                "status": _status(len(inventory) >= 8),
                "detail": f"rows={len(inventory)}",
            },
            {
                "check": "feature_layer_candidates_found",
                "status": _status(ready_candidates >= 3),
                "detail": f"ready_candidates={ready_candidates}",
            },
            {
                "check": "validation_queue_produced",
                "status": _status(not queue.empty),
                "detail": f"rows={len(queue)}",
            },
            {
                "check": "current_snapshot_sources_blocked",
                "status": _status(current_snapshot_ready.empty),
                "detail": f"bad_rows={len(current_snapshot_ready)}",
            },
            {
                "check": "same_branch_v3_65_not_main_path",
                "status": _status(not same_branch_main.empty),
                "detail": f"observation_only_rows={int(same_branch_main['affected_rows'].sum()) if not same_branch_main.empty else 0}",
            },
            {
                "check": "promotion_outputs_blocked",
                "status": _status(blocked_correctly and not forbidden_columns),
                "detail": ",".join(forbidden_columns),
            },
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
    inventory: pd.DataFrame,
    blueprints: pd.DataFrame,
    queue: pd.DataFrame,
    blocked_requests: pd.DataFrame,
    branch_policy: pd.DataFrame,
    input_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    feature_ready_stages = {"ready_for_feature_layer", "ready_for_nonreturn_breadth_feature_layer"}
    ready_sources = int(inventory["allowed_stage"].isin(feature_ready_stages).sum()) if not inventory.empty else 0
    ready_blueprints = int(blueprints["allowed_for_v3_67_feature_layer"].astype(bool).sum()) if not blueprints.empty else 0
    lines = [
        "# V3.66 Independent PIT Source Discovery",
        "",
        "## Decision",
        "",
        "- V3.66 pivots away from broad retuning of the V3.64/V3.65 MARKET proxy branch.",
        "- It inventories independent point-in-time sources and queues feature-layer work only.",
        "- It does not run a portfolio harness, produce performance claims, or promote a model.",
        "",
        "## Headline",
        "",
        f"- Sources inventoried: `{len(inventory)}`",
        f"- Sources usable for feature-layer work: `{ready_sources}`",
        f"- Signal blueprints reviewed: `{len(blueprints)}`",
        f"- Blueprints eligible for next feature-layer task: `{ready_blueprints}`",
        f"- Queued next tasks: `{len(queue)}`",
        "",
        "## Source Inventory",
        "",
    ]
    lines.extend(
        markdown_table(
            inventory,
            [
                "source_id",
                "source_family",
                "file_count",
                "row_count_estimate",
                "start_date",
                "end_date",
                "pit_status",
                "allowed_stage",
                "priority_score",
                "next_action",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Signal Blueprints", ""])
    lines.extend(
        markdown_table(
            blueprints,
            [
                "signal_id",
                "source_id",
                "signal_scope",
                "horizon_bucket",
                "research_status",
                "allowed_for_v3_67_feature_layer",
                "priority_score",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Next Validation Queue", ""])
    lines.extend(markdown_table(queue, ["priority", "version_hint", "assigned_agent", "signal_id", "source_id", "success_gate"], max_rows=12))
    lines.extend(["", "## V3.65 Branch Policy", ""])
    lines.extend(markdown_table(branch_policy, ["policy_item", "affected_rows", "policy_decision", "allowed_next_stage"], max_rows=12))
    lines.extend(["", "## Blocked Data Requests", ""])
    lines.extend(markdown_table(blocked_requests, ["request_id", "provider_hint", "needed_for", "current_blocker", "priority"], max_rows=12))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=12))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=12))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Start V3.67 with the market participation breadth feature layer because it uses the currently available Tushare daily-only permission and is not another MARKET proxy threshold tune.",
            "- Keep stock-level return labels blocked until adjustment-factor or adjusted-price coverage exists.",
            "- Keep current constituents and latest weights out of historical validation until vintage data is acquired.",
        ]
    )
    return "\n".join(lines)


def build_catalog(inventory: pd.DataFrame, blueprints: pd.DataFrame, queue: pd.DataFrame) -> str:
    feature_ready_stages = {"ready_for_feature_layer", "ready_for_nonreturn_breadth_feature_layer"}
    ready_sources = int(inventory["allowed_stage"].isin(feature_ready_stages).sum()) if not inventory.empty else 0
    ready_blueprints = int(blueprints["allowed_for_v3_67_feature_layer"].astype(bool).sum()) if not blueprints.empty else 0
    return "\n".join(
        [
            "# A-share Independent PIT Source Discovery V3.66",
            "",
            "## Dataset Role",
            "",
            "V3.66 records independent local PIT source readiness after V3.65 diagnosed same-branch proxy instability.",
            "",
            "## Governance",
            "",
            "- Portfolio harness: not produced.",
            "- Model promotion: not allowed.",
            "- Same-branch MARKET proxy retuning: not allowed.",
            "- Current snapshot sources: blocked for historical validation.",
            "",
            "## Produced Shape",
            "",
            f"- Source inventory rows: `{len(inventory)}`",
            f"- Feature-layer ready sources: `{ready_sources}`",
            f"- Signal blueprint rows: `{len(blueprints)}`",
            f"- Feature-layer eligible blueprints: `{ready_blueprints}`",
            f"- Queued validation tasks: `{len(queue)}`",
        ]
    )
