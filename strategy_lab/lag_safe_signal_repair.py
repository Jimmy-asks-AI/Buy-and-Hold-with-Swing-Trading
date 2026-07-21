"""Lag-safe non-price signal repair for HIRSSM V3.62.

V3.62 rebuilds a small candidate signal panel after V3.61 found artifact risk
in the V3.60 proxy-positive observations. The repaired panel uses only
non-price daily activity and liquidity inputs from the prior trading day and is
not a validation, backtest, or model-promotion artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS, _corr, normalize_date


ASSET = "MARKET"
DATA_SOURCE = "lag_safe_nonprice_signal_repair_v3_62"
STATE_COLUMNS = [
    "composite_state",
    "liquidity_state",
    "breadth_state",
    "activity_state",
    "concentration_state",
    "limit_crowding_state",
]


@dataclass(frozen=True)
class LagSafeRepairConfig:
    state_panel_path: Path
    v3_48_manifest_path: Path
    v3_61_decision_path: Path
    v3_61_manifest_path: Path
    market_proxy_source_path: Path
    output_dir: Path
    catalog_path: Path
    min_signal_rows: int
    min_repaired_candidates: int
    max_abs_same_day_market_move_corr_warn: float


@dataclass(frozen=True)
class RepairedSignalSpec:
    signal_id: str
    family: str
    signal_direction: str
    formula_name: str
    source_columns: tuple[str, ...]
    description: str
    economic_hypothesis: str
    failure_modes: str
    repair_source: str
    source_component_type: str = "non_price_only"
    status: str = "observation_repaired"
    allowed_use: str = "future lag-safe proxy validation only"
    forbidden_use: str = "no trading decision, portfolio backtest, or performance claim"


def _clip01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(lower=0.0, upper=1.0)


def _inv01(values: pd.Series) -> pd.Series:
    return 1.0 - _clip01(values)


def _liquidity_expansion_np(panel: pd.DataFrame) -> pd.Series:
    return (
        0.55 * _clip01(panel["amount_per_asset_raw_trailing_pctile"])
        + 0.35 * _clip01(panel["volume_per_asset_raw_trailing_pctile"])
        + 0.10 * _clip01(panel["active_asset_ratio_trailing_pctile"])
    ).clip(lower=0.0, upper=1.0)


def _liquidity_quality_np(panel: pd.DataFrame) -> pd.Series:
    return (
        0.35 * _clip01(panel["amount_per_asset_raw_trailing_pctile"])
        + 0.25 * _clip01(panel["volume_per_asset_raw_trailing_pctile"])
        + 0.20 * _clip01(panel["active_asset_ratio_trailing_pctile"])
        + 0.10 * _inv01(panel["low_amount_share_trailing_pctile"])
        + 0.10 * _inv01(panel["top_amount_share_trailing_pctile"])
    ).clip(lower=0.0, upper=1.0)


def _liquidity_dryup_np(panel: pd.DataFrame) -> pd.Series:
    return (
        0.35 * _inv01(panel["amount_per_asset_raw_trailing_pctile"])
        + 0.25 * _inv01(panel["volume_per_asset_raw_trailing_pctile"])
        + 0.25 * _clip01(panel["low_amount_share_trailing_pctile"])
        + 0.15 * _inv01(panel["active_asset_ratio_trailing_pctile"])
    ).clip(lower=0.0, upper=1.0)


def _turnover_concentration_np(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["top_amount_share_trailing_pctile"])


def _turnover_diffusion_np(panel: pd.DataFrame) -> pd.Series:
    return (
        0.50 * _inv01(panel["top_amount_share_trailing_pctile"])
        + 0.25 * _clip01(panel["active_asset_ratio_trailing_pctile"])
        + 0.25 * _clip01(panel["amount_per_asset_raw_trailing_pctile"])
    ).clip(lower=0.0, upper=1.0)


def _participation_quality_np(panel: pd.DataFrame) -> pd.Series:
    return (
        0.35 * _clip01(panel["active_asset_ratio_trailing_pctile"])
        + 0.30 * _inv01(panel["low_amount_share_trailing_pctile"])
        + 0.20 * _clip01(panel["amount_per_asset_raw_trailing_pctile"])
        + 0.15 * _clip01(panel["volume_per_asset_raw_trailing_pctile"])
    ).clip(lower=0.0, upper=1.0)


SIGNAL_FORMULAS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "liquidity_expansion_np": _liquidity_expansion_np,
    "liquidity_quality_np": _liquidity_quality_np,
    "liquidity_dryup_np": _liquidity_dryup_np,
    "turnover_concentration_np": _turnover_concentration_np,
    "turnover_diffusion_np": _turnover_diffusion_np,
    "participation_quality_np": _participation_quality_np,
}


REPAIRED_SIGNAL_SPECS = [
    RepairedSignalSpec(
        signal_id="lagged_liquidity_expansion_np_v1",
        family="nonprice_liquidity_activity",
        signal_direction="positive",
        formula_name="liquidity_expansion_np",
        source_columns=(
            "amount_per_asset_raw_trailing_pctile",
            "volume_per_asset_raw_trailing_pctile",
            "active_asset_ratio_trailing_pctile",
        ),
        description="Prior-day non-price amount, volume, and active-asset participation expansion.",
        economic_hypothesis="Broad liquidity expansion can indicate improving participation without relying on same-day price signs.",
        failure_modes="Can still fail during speculative churn, structural turnover changes, or after high-volume reversal days.",
        repair_source="liquidity_expansion_score_v1",
    ),
    RepairedSignalSpec(
        signal_id="lagged_liquidity_quality_np_v1",
        family="nonprice_liquidity_quality",
        signal_direction="positive",
        formula_name="liquidity_quality_np",
        source_columns=(
            "amount_per_asset_raw_trailing_pctile",
            "volume_per_asset_raw_trailing_pctile",
            "active_asset_ratio_trailing_pctile",
            "low_amount_share_trailing_pctile",
            "top_amount_share_trailing_pctile",
        ),
        description="Prior-day liquidity expansion penalized by thin-tail activity and turnover concentration.",
        economic_hypothesis="Higher quality participation should be more durable than raw turnover spikes.",
        failure_modes="Can underweight narrow but persistent leader-led markets.",
        repair_source="liquidity_expansion_score_v1",
    ),
    RepairedSignalSpec(
        signal_id="lagged_liquidity_dryup_np_v1",
        family="nonprice_liquidity_activity",
        signal_direction="negative",
        formula_name="liquidity_dryup_np",
        source_columns=(
            "amount_per_asset_raw_trailing_pctile",
            "volume_per_asset_raw_trailing_pctile",
            "low_amount_share_trailing_pctile",
            "active_asset_ratio_trailing_pctile",
        ),
        description="Prior-day non-price low liquidity and thin participation risk.",
        economic_hypothesis="Thin trading and low participation can make market conditions fragile.",
        failure_modes="Can flag quiet accumulation or holiday effects as risk.",
        repair_source="liquidity_dry_up_risk_v1",
    ),
    RepairedSignalSpec(
        signal_id="lagged_turnover_concentration_np_v1",
        family="nonprice_concentration",
        signal_direction="negative",
        formula_name="turnover_concentration_np",
        source_columns=("top_amount_share_trailing_pctile",),
        description="Prior-day non-price turnover concentration risk.",
        economic_hypothesis="Concentrated trading can indicate narrow participation and crowded positioning.",
        failure_modes="Can fail in durable leader-led trends.",
        repair_source="turnover_concentration_risk_v1",
    ),
    RepairedSignalSpec(
        signal_id="lagged_turnover_diffusion_np_v1",
        family="nonprice_concentration",
        signal_direction="positive",
        formula_name="turnover_diffusion_np",
        source_columns=(
            "top_amount_share_trailing_pctile",
            "active_asset_ratio_trailing_pctile",
            "amount_per_asset_raw_trailing_pctile",
        ),
        description="Prior-day diffuse turnover plus active participation.",
        economic_hypothesis="Diffused activity can indicate broader market participation than concentrated heat.",
        failure_modes="May lag fast rotations and can be diluted by broad but low-quality churn.",
        repair_source="new_nonprice_proxy_input",
    ),
    RepairedSignalSpec(
        signal_id="lagged_participation_quality_np_v1",
        family="nonprice_participation",
        signal_direction="positive",
        formula_name="participation_quality_np",
        source_columns=(
            "active_asset_ratio_trailing_pctile",
            "low_amount_share_trailing_pctile",
            "amount_per_asset_raw_trailing_pctile",
            "volume_per_asset_raw_trailing_pctile",
        ),
        description="Prior-day active-asset participation with low thin-tail activity.",
        economic_hypothesis="Healthy participation should have active breadth without many near-dormant names.",
        failure_modes="Can be distorted by listing-count changes or temporary market-wide activity shifts.",
        repair_source="new_nonprice_proxy_input",
    ),
]


def repaired_registry_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in REPAIRED_SIGNAL_SPECS:
        rows.append(
            {
                "signal_id": spec.signal_id,
                "family": spec.family,
                "signal_direction": spec.signal_direction,
                "formula_name": spec.formula_name,
                "source_columns": ",".join(spec.source_columns),
                "source_component_type": spec.source_component_type,
                "description": spec.description,
                "economic_hypothesis": spec.economic_hypothesis,
                "failure_modes": spec.failure_modes,
                "repair_source": spec.repair_source,
                "status": spec.status,
                "allowed_use": spec.allowed_use,
                "forbidden_use": spec.forbidden_use,
                "data_source": DATA_SOURCE,
                "asset": ASSET,
                "model_promotion_allowed": False,
                "performance_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _valid_state_rows(state_panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "history_available",
        "data_scope",
        "price_adjustment",
        "data_quality_state",
        "model_usage_allowed",
        "backtest_usage_allowed",
        *STATE_COLUMNS,
    }
    for spec in REPAIRED_SIGNAL_SPECS:
        required.update(spec.source_columns)
    missing = sorted(required.difference(state_panel.columns))
    if missing:
        raise ValueError(f"state panel missing columns: {missing}")
    valid = state_panel.copy()
    valid["trade_date"] = normalize_date(valid["trade_date"])
    valid = valid.sort_values("trade_date").reset_index(drop=True)
    valid = valid.loc[valid["history_available"].astype(bool)]
    valid = valid.loc[valid["data_scope"].astype(str) == "accepted_processed_daily_only"]
    valid = valid.loc[valid["price_adjustment"].astype(str) == "none_raw"]
    valid = valid.loc[valid["data_quality_state"].astype(str) == "quality_ok"]
    valid = valid.loc[~valid["model_usage_allowed"].astype(bool)]
    valid = valid.loc[~valid["backtest_usage_allowed"].astype(bool)]
    valid = valid.sort_values("trade_date").reset_index(drop=True)
    valid["signal_date"] = valid["trade_date"].shift(-1)
    valid["available_date"] = valid["signal_date"]
    valid["trade_lag_steps"] = 1
    valid = valid.dropna(subset=["signal_date"]).copy()
    return valid


def build_repaired_signal_panel(state_panel: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_state_rows(state_panel)
    rows: list[dict[str, Any]] = []
    for spec in REPAIRED_SIGNAL_SPECS:
        values = SIGNAL_FORMULAS[spec.formula_name](valid)
        for idx, value in values.items():
            if not np.isfinite(value):
                continue
            row = valid.loc[idx]
            out: dict[str, Any] = {
                "signal_id": spec.signal_id,
                "signal_date": str(row["signal_date"]),
                "source_trade_date": str(row["trade_date"]),
                "asset": ASSET,
                "signal_value": float(value),
                "signal_direction": spec.signal_direction,
                "available_date": str(row["available_date"]),
                "trade_lag_steps": int(row["trade_lag_steps"]),
                "signal_lag_rule": "prior_trading_day_close_to_next_trading_day",
                "source_component_type": spec.source_component_type,
                "source_columns": ",".join(spec.source_columns),
                "data_source": DATA_SOURCE,
                "candidate_status": spec.status,
                "signal_family": spec.family,
                "formula_name": spec.formula_name,
                "repair_source": spec.repair_source,
                "data_quality_state": row["data_quality_state"],
                "data_scope": row["data_scope"],
                "price_adjustment": row["price_adjustment"],
                "state_reference_date": str(row["trade_date"]),
                "stratification_state_source": "lagged_source_state",
                "model_promotion_allowed": False,
                "performance_claim_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
            for col in STATE_COLUMNS:
                out[col] = row[col]
            rows.append(out)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    return panel.sort_values(["signal_id", "signal_date"]).reset_index(drop=True)


def build_signal_summary(signal_panel: pd.DataFrame) -> pd.DataFrame:
    if signal_panel.empty:
        return pd.DataFrame()
    data = signal_panel.copy()
    grouped = data.groupby(["signal_id", "signal_direction", "signal_family", "source_component_type"], as_index=False)
    return grouped.agg(
        observations=("signal_value", "count"),
        first_signal_date=("signal_date", "min"),
        last_signal_date=("signal_date", "max"),
        mean_signal_value=("signal_value", "mean"),
        median_signal_value=("signal_value", "median"),
        source_first_date=("source_trade_date", "min"),
        source_last_date=("source_trade_date", "max"),
    )


def build_source_column_audit() -> pd.DataFrame:
    non_price_columns = {
        "active_asset_ratio_trailing_pctile",
        "amount_per_asset_raw_trailing_pctile",
        "volume_per_asset_raw_trailing_pctile",
        "low_amount_share_trailing_pctile",
        "top_amount_share_trailing_pctile",
    }
    rows: list[dict[str, Any]] = []
    for spec in REPAIRED_SIGNAL_SPECS:
        for col in spec.source_columns:
            rows.append(
                {
                    "signal_id": spec.signal_id,
                    "source_column": col,
                    "source_component_type": "non_price_only" if col in non_price_columns else "review_required",
                    "same_day_price_component": False,
                    "allowed_for_repair": col in non_price_columns,
                    "reason": "daily activity/liquidity/concentration input; not derived from pct_chg, limit state, or index return",
                }
            )
    return pd.DataFrame(rows)


def build_retired_signal_decision(v3_61_decisions: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "liquidity_expansion_score_v1": "mapped_to_lagged_nonprice_liquidity_repairs",
        "liquidity_dry_up_risk_v1": "mapped_to_lagged_nonprice_dryup_repair",
        "risk_on_composite_event_v1": "retired_composite_state_price_mixed_proxy",
        "up_limit_heat_score_v1": "retired_limit_price_derived_proxy",
    }
    replacement = {
        "liquidity_expansion_score_v1": "lagged_liquidity_expansion_np_v1;lagged_liquidity_quality_np_v1",
        "liquidity_dry_up_risk_v1": "lagged_liquidity_dryup_np_v1",
        "risk_on_composite_event_v1": "",
        "up_limit_heat_score_v1": "",
    }
    data = v3_61_decisions.copy()
    data["repair_decision"] = data["signal_id"].map(mapping).fillna("not_in_v3_62_repair_scope")
    data["replacement_signal_id"] = data["signal_id"].map(replacement).fillna("")
    data["default_model_allowed"] = False
    data["portfolio_backtest_allowed"] = False
    data["official_total_return_evidence"] = False
    return data.sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_lag_discipline_checks(
    signal_panel: pd.DataFrame,
    registry: pd.DataFrame,
    source_audit: pd.DataFrame,
    config: LagSafeRepairConfig,
    v3_48_manifest: dict[str, Any],
    v3_61_manifest: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "check": "v3_48_manifest_accepted",
            "status": "pass" if bool(v3_48_manifest.get("acceptance_pass", v3_48_manifest.get("self_check_pass", False))) else "fail",
            "detail": f"acceptance={v3_48_manifest.get('acceptance_pass')};self_check={v3_48_manifest.get('self_check_pass')}",
        },
        {
            "check": "v3_61_manifest_accepted",
            "status": "pass" if bool(v3_61_manifest.get("self_check_pass")) else "fail",
            "detail": f"self_check={v3_61_manifest.get('self_check_pass')}",
        },
        {
            "check": "repaired_candidate_count_minimum",
            "status": "pass" if len(registry) >= config.min_repaired_candidates else "fail",
            "detail": f"registry={len(registry)};min={config.min_repaired_candidates}",
        },
        {
            "check": "signal_rows_above_minimum",
            "status": "pass" if len(signal_panel) >= config.min_signal_rows else "fail",
            "detail": f"rows={len(signal_panel)};min={config.min_signal_rows}",
        },
    ]
    if signal_panel.empty:
        rows.append({"check": "signal_panel_non_empty", "status": "fail", "detail": "no rows"})
        return pd.DataFrame(rows)

    source_dates = pd.to_datetime(signal_panel["source_trade_date"], format="%Y%m%d", errors="coerce")
    signal_dates = pd.to_datetime(signal_panel["signal_date"], format="%Y%m%d", errors="coerce")
    available_dates = pd.to_datetime(signal_panel["available_date"], format="%Y%m%d", errors="coerce")
    rows.extend(
        [
            {
                "check": "source_date_before_signal_date",
                "status": "pass" if bool((source_dates < signal_dates).all()) else "fail",
                "detail": f"bad_rows={int((source_dates >= signal_dates).sum())}",
            },
            {
                "check": "available_date_equals_signal_date",
                "status": "pass" if bool((available_dates == signal_dates).all()) else "fail",
                "detail": f"bad_rows={int((available_dates != signal_dates).sum())}",
            },
            {
                "check": "trade_lag_steps_is_one",
                "status": "pass" if set(signal_panel["trade_lag_steps"].astype(int).unique()) == {1} else "fail",
                "detail": ",".join(map(str, sorted(signal_panel["trade_lag_steps"].astype(int).unique()))),
            },
            {
                "check": "source_columns_non_price_only",
                "status": "pass" if bool(source_audit["allowed_for_repair"].astype(bool).all()) else "fail",
                "detail": f"bad_rows={int((~source_audit['allowed_for_repair'].astype(bool)).sum())}",
            },
            {
                "check": "signal_scope_market_only",
                "status": "pass" if set(signal_panel["asset"].astype(str).unique()) == {ASSET} else "fail",
                "detail": ",".join(sorted(signal_panel["asset"].astype(str).unique())),
            },
            {
                "check": "no_model_or_performance_promotion",
                "status": "pass"
                if not bool(signal_panel["model_promotion_allowed"].astype(bool).any())
                and not bool(signal_panel["performance_claim_allowed"].astype(bool).any())
                and not bool(signal_panel["portfolio_backtest_allowed"].astype(bool).any())
                else "fail",
                "detail": "observation only",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_market_move_artifact_screen(signal_panel: pd.DataFrame, market_proxy_source: pd.DataFrame, config: LagSafeRepairConfig) -> pd.DataFrame:
    if signal_panel.empty:
        return pd.DataFrame()
    market = market_proxy_source.copy()
    market["signal_date"] = normalize_date(market["date"])
    market["market_same_day_pct_move"] = pd.to_numeric(market["pct_chg"], errors="coerce") / 100.0
    joined = signal_panel.merge(market.loc[:, ["signal_date", "market_same_day_pct_move"]], on="signal_date", how="left")
    direction = joined["signal_direction"].astype(str).str.lower().map({"positive": 1.0, "negative": -1.0}).fillna(0.0)
    joined["signed_market_same_day_pct_move"] = joined["market_same_day_pct_move"] * direction
    rows: list[dict[str, Any]] = []
    for (signal_id, direction_name), group in joined.groupby(["signal_id", "signal_direction"], dropna=False):
        corr = _corr(group["signal_value"], group["signed_market_same_day_pct_move"], "spearman")
        rows.append(
            {
                "signal_id": signal_id,
                "signal_direction": direction_name,
                "observations": int(group["signed_market_same_day_pct_move"].notna().sum()),
                "same_day_market_move_spearman": corr,
                "artifact_warning": bool(pd.notna(corr) and abs(corr) > config.max_abs_same_day_market_move_corr_warn),
                "warning_threshold": config.max_abs_same_day_market_move_corr_warn,
                "screen_role": "diagnostic_only_not_validation",
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact_warning", "signal_id"], ascending=[False, True]).reset_index(drop=True)


def build_no_promotion_guard(signal_panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "lag_safe_repaired_signal_panel",
                "produced": not signal_panel.empty,
                "blocked": signal_panel.empty,
                "reason": "candidate signal definitions only",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.62 does not use total-return labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.62 does not create weights, trades, NAV, Sharpe, or drawdown",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "repaired signal definitions require later validation",
            },
        ]
    )


def build_acceptance_checks(
    lag_checks: pd.DataFrame,
    signal_panel: pd.DataFrame,
    registry: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(["official_total_return_validation", "portfolio_backtest", "model_promotion"]),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "lag_discipline_checks_passed",
                "status": "pass" if bool(lag_checks["status"].eq("pass").all()) else "fail",
                "detail": ";".join(lag_checks.loc[lag_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "registry_and_panel_non_empty",
                "status": "pass" if not registry.empty and not signal_panel.empty else "fail",
                "detail": f"registry={len(registry)};panel={len(signal_panel)}",
            },
            {
                "check": "candidate_definitions_do_not_promote",
                "status": "pass"
                if not signal_panel.empty and not signal_panel["model_promotion_allowed"].astype(bool).any()
                else "fail",
                "detail": f"rows={len(signal_panel)}",
            },
            {
                "check": "portfolio_or_promotion_outputs_not_produced",
                "status": "pass" if not forbidden_produced and not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
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
    registry: pd.DataFrame,
    signal_panel: pd.DataFrame,
    summary: pd.DataFrame,
    source_audit: pd.DataFrame,
    retired: pd.DataFrame,
    market_screen: pd.DataFrame,
    lag_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    warning_rows = int(market_screen["artifact_warning"].astype(bool).sum()) if not market_screen.empty else 0
    lines = [
        "# V3.62 Lag-Safe Non-Price Signal Repair",
        "",
        "## Decision",
        "",
        "- V3.62 repairs V3.50/V3.60 artifact-risk signals by enforcing prior-trading-day source data.",
        "- Repaired formulas use non-price activity, liquidity, and turnover concentration inputs only.",
        "- It does not run a portfolio backtest, write NAV, or promote any default model.",
        "",
        "## Scope",
        "",
        f"- Repaired candidates: `{len(registry)}`",
        f"- Repaired signal rows: `{len(signal_panel)}`",
        f"- Signal dates: `{signal_panel['signal_date'].nunique() if not signal_panel.empty else 0}`",
        f"- Same-day market-move warning rows: `{warning_rows}`",
        "",
        "## Repaired Candidate Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            summary,
            [
                "signal_id",
                "signal_direction",
                "signal_family",
                "source_component_type",
                "observations",
                "first_signal_date",
                "last_signal_date",
                "mean_signal_value",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Source Column Audit", ""])
    lines.extend(markdown_table(source_audit, ["signal_id", "source_column", "source_component_type", "same_day_price_component", "allowed_for_repair"], max_rows=30))
    lines.extend(["", "## V3.61 Candidate Disposition", ""])
    lines.extend(markdown_table(retired, ["signal_id", "horizon", "artifact_review_status", "repair_decision", "replacement_signal_id"], max_rows=24))
    lines.extend(["", "## Same-Day Market Move Screen", ""])
    lines.extend(markdown_table(market_screen, ["signal_id", "same_day_market_move_spearman", "artifact_warning", "screen_role"], max_rows=20))
    lines.extend(["", "## Lag Discipline Checks", ""])
    lines.extend(markdown_table(lag_checks, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=20))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.62 is a repaired candidate registry, not effectiveness evidence.",
            "- V3.63 should pair this repaired panel with existing price-proxy labels and rerun artifact screening before any walk-forward validation.",
            "- Price-derived V3.50 candidates remain retired until redesigned with independent inputs.",
        ]
    )
    return "\n".join(lines)


def build_catalog(signal_panel: pd.DataFrame, registry: pd.DataFrame, source_audit: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# A-share Lag-Safe Signal Repair V3.62",
            "",
            "## Dataset Role",
            "",
            "V3.62 creates repaired MARKET candidate signals after V3.61 artifact audit.",
            "",
            "## Governance",
            "",
            "- Source timing: prior trading day close to next trading day.",
            "- Formula source type: non-price activity and liquidity inputs only.",
            "- Official total-return evidence: false.",
            "- Portfolio backtest: not produced.",
            "- Default model promotion: not allowed.",
            "",
            "## Produced Shape",
            "",
            f"- Repaired candidates: `{len(registry)}`",
            f"- Signal rows: `{len(signal_panel)}`",
            f"- Source audit rows: `{len(source_audit)}`",
        ]
    )
