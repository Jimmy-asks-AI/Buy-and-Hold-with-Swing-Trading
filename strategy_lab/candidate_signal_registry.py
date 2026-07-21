"""Candidate signal registry and signal-panel builder for V3.50.

The registry turns V3.48 daily-only diagnostic states into standardized
candidate signal rows. It intentionally does not validate performance, compute
forward returns, run a backtest, or promote any signal into a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


DATA_SOURCE = "accepted_daily_only_state_monitor_v3_48"
ASSET = "MARKET"


@dataclass(frozen=True)
class CandidateSignalSpec:
    signal_id: str
    family: str
    signal_direction: str
    formula_name: str
    source_columns: tuple[str, ...]
    description: str
    economic_hypothesis: str
    failure_modes: str
    status: str = "observation"
    allowed_use: str = "future state-stratified validation after adjusted/PIT labels arrive"
    forbidden_use: str = "no trading decision, portfolio backtest, or performance claim"


def _clip01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(lower=0.0, upper=1.0)


def _event(values: pd.Series, expected: set[str]) -> pd.Series:
    return values.astype(str).isin(expected).astype(float)


def _breadth_recovery_score(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["advance_decline_balance_trailing_pctile"])


def _breadth_stress_risk(panel: pd.DataFrame) -> pd.Series:
    return 1.0 - _clip01(panel["advance_decline_balance_trailing_pctile"])


def _liquidity_expansion_score(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["amount_per_asset_raw_trailing_pctile"])


def _liquidity_dry_up_risk(panel: pd.DataFrame) -> pd.Series:
    base = 1.0 - _clip01(panel["amount_per_asset_raw_trailing_pctile"])
    low_amount = _clip01(panel["low_amount_share_trailing_pctile"])
    return (0.7 * base + 0.3 * low_amount).clip(lower=0.0, upper=1.0)


def _range_activity_risk(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["median_range_ratio_raw_trailing_pctile"])


def _turnover_concentration_risk(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["top_amount_share_trailing_pctile"])


def _up_limit_heat_score(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["limit_up_share_trailing_pctile"])


def _down_limit_stress_risk(panel: pd.DataFrame) -> pd.Series:
    return _clip01(panel["limit_down_share_trailing_pctile"])


def _risk_on_composite_event(panel: pd.DataFrame) -> pd.Series:
    return _event(panel["composite_state"], {"risk_on_breadth_liquidity", "breadth_recovery_only"})


def _stress_composite_event(panel: pd.DataFrame) -> pd.Series:
    return _event(
        panel["composite_state"],
        {"stress_liquidity_breadth", "stress_limit_breadth", "breadth_stress_only", "liquidity_soft_patch"},
    )


def _crowded_activity_event(panel: pd.DataFrame) -> pd.Series:
    return _event(panel["composite_state"], {"crowded_high_activity"})


SIGNAL_FORMULAS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "breadth_recovery_score": _breadth_recovery_score,
    "breadth_stress_risk": _breadth_stress_risk,
    "liquidity_expansion_score": _liquidity_expansion_score,
    "liquidity_dry_up_risk": _liquidity_dry_up_risk,
    "range_activity_risk": _range_activity_risk,
    "turnover_concentration_risk": _turnover_concentration_risk,
    "up_limit_heat_score": _up_limit_heat_score,
    "down_limit_stress_risk": _down_limit_stress_risk,
    "risk_on_composite_event": _risk_on_composite_event,
    "stress_composite_event": _stress_composite_event,
    "crowded_activity_event": _crowded_activity_event,
}


CANDIDATE_SIGNAL_SPECS = [
    CandidateSignalSpec(
        signal_id="breadth_recovery_score_v1",
        family="market_breadth",
        signal_direction="positive",
        formula_name="breadth_recovery_score",
        source_columns=("advance_decline_balance_trailing_pctile", "breadth_state"),
        description="Higher trailing percentile of advance-decline balance.",
        economic_hypothesis="Broad participation recovery can mark improving risk appetite.",
        failure_modes="Can fail in short-covering rebounds, bear-market rallies, and narrow index-led markets.",
    ),
    CandidateSignalSpec(
        signal_id="breadth_stress_risk_v1",
        family="market_breadth",
        signal_direction="negative",
        formula_name="breadth_stress_risk",
        source_columns=("advance_decline_balance_trailing_pctile", "breadth_state"),
        description="Low trailing percentile of advance-decline balance expressed as stress risk.",
        economic_hypothesis="Broad selling pressure can indicate fragile forward conditions.",
        failure_modes="Can fail near capitulation lows or after policy intervention.",
    ),
    CandidateSignalSpec(
        signal_id="liquidity_expansion_score_v1",
        family="liquidity_activity",
        signal_direction="positive",
        formula_name="liquidity_expansion_score",
        source_columns=("amount_per_asset_raw_trailing_pctile", "liquidity_state"),
        description="Higher trailing percentile of raw amount per listed asset.",
        economic_hypothesis="Expanding participation and turnover can support market follow-through.",
        failure_modes="Can fail in speculative blow-offs or when turnover is concentrated in low-quality themes.",
    ),
    CandidateSignalSpec(
        signal_id="liquidity_dry_up_risk_v1",
        family="liquidity_activity",
        signal_direction="negative",
        formula_name="liquidity_dry_up_risk",
        source_columns=("amount_per_asset_raw_trailing_pctile", "low_amount_share_trailing_pctile", "liquidity_state"),
        description="Low raw amount percentile plus high low-amount share as dry-up risk.",
        economic_hypothesis="Weak turnover and thin participation can amplify downside and reduce signal reliability.",
        failure_modes="Can fail during calm accumulation or holiday/structural turnover shifts.",
    ),
    CandidateSignalSpec(
        signal_id="range_activity_risk_v1",
        family="intraday_raw_diagnostics",
        signal_direction="negative",
        formula_name="range_activity_risk",
        source_columns=("median_range_ratio_raw_trailing_pctile", "activity_state"),
        description="High percentile of median raw intraday range ratio.",
        economic_hypothesis="High same-day range can flag unstable trading conditions.",
        failure_modes="Can also accompany healthy trend expansion; requires later return validation.",
    ),
    CandidateSignalSpec(
        signal_id="turnover_concentration_risk_v1",
        family="liquidity_activity",
        signal_direction="negative",
        formula_name="turnover_concentration_risk",
        source_columns=("top_amount_share_trailing_pctile", "concentration_state"),
        description="High percentile of top-name turnover concentration.",
        economic_hypothesis="Concentrated turnover can mean narrow leadership and crowded positioning.",
        failure_modes="Can fail in durable leader-led bull phases.",
    ),
    CandidateSignalSpec(
        signal_id="up_limit_heat_score_v1",
        family="market_breadth",
        signal_direction="positive",
        formula_name="up_limit_heat_score",
        source_columns=("limit_up_share_trailing_pctile", "limit_crowding_state"),
        description="High percentile of limit-like upside participation.",
        economic_hypothesis="Upside limit crowding may capture speculative risk appetite.",
        failure_modes="Can reverse sharply after euphoria; later validation must check horizon sensitivity.",
    ),
    CandidateSignalSpec(
        signal_id="down_limit_stress_risk_v1",
        family="market_breadth",
        signal_direction="negative",
        formula_name="down_limit_stress_risk",
        source_columns=("limit_down_share_trailing_pctile", "limit_crowding_state"),
        description="High percentile of limit-like downside participation.",
        economic_hypothesis="Downside limit crowding can indicate forced selling and weak liquidity.",
        failure_modes="Can fail at panic lows and policy floors.",
    ),
    CandidateSignalSpec(
        signal_id="risk_on_composite_event_v1",
        family="composite_state",
        signal_direction="positive",
        formula_name="risk_on_composite_event",
        source_columns=("composite_state",),
        description="Binary event for risk-on or breadth-recovery composite states.",
        economic_hypothesis="Joint breadth/liquidity improvement can indicate better market condition.",
        failure_modes="Composite state can lag sudden reversals and can overfit state definitions.",
    ),
    CandidateSignalSpec(
        signal_id="stress_composite_event_v1",
        family="composite_state",
        signal_direction="negative",
        formula_name="stress_composite_event",
        source_columns=("composite_state",),
        description="Binary event for stress or soft-patch composite states.",
        economic_hypothesis="Stress states can mark periods when candidate alpha is less reliable.",
        failure_modes="Can fail at tradable bottoms; not a short signal without validation.",
    ),
    CandidateSignalSpec(
        signal_id="crowded_activity_event_v1",
        family="composite_state",
        signal_direction="negative",
        formula_name="crowded_activity_event",
        source_columns=("composite_state",),
        description="Binary event for crowded high-activity composite state.",
        economic_hypothesis="High activity plus concentration can be unstable or crowded.",
        failure_modes="Can fail in strong momentum tapes where crowding persists.",
    ),
]


def candidate_registry_frame() -> pd.DataFrame:
    rows = []
    for spec in CANDIDATE_SIGNAL_SPECS:
        rows.append(
            {
                "signal_id": spec.signal_id,
                "family": spec.family,
                "signal_direction": spec.signal_direction,
                "formula_name": spec.formula_name,
                "source_columns": ",".join(spec.source_columns),
                "description": spec.description,
                "economic_hypothesis": spec.economic_hypothesis,
                "failure_modes": spec.failure_modes,
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
        "composite_state",
        "model_usage_allowed",
        "backtest_usage_allowed",
    }
    missing = sorted(required.difference(state_panel.columns))
    if missing:
        raise ValueError(f"state panel missing columns: {missing}")
    valid = state_panel.copy()
    valid = valid.loc[valid["history_available"].astype(bool)]
    valid = valid.loc[valid["data_scope"].astype(str) == "accepted_processed_daily_only"]
    valid = valid.loc[valid["price_adjustment"].astype(str) == "none_raw"]
    valid = valid.loc[valid["data_quality_state"].astype(str) == "quality_ok"]
    valid = valid.loc[~valid["model_usage_allowed"].astype(bool)]
    valid = valid.loc[~valid["backtest_usage_allowed"].astype(bool)]
    return valid.copy()


def build_signal_panel(state_panel: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_state_rows(state_panel)
    rows = []
    for spec in CANDIDATE_SIGNAL_SPECS:
        missing = sorted(set(spec.source_columns).difference(valid.columns))
        if missing:
            raise ValueError(f"{spec.signal_id} missing source columns: {missing}")
        values = SIGNAL_FORMULAS[spec.formula_name](valid)
        for idx, value in values.items():
            if not np.isfinite(value):
                continue
            row = valid.loc[idx]
            rows.append(
                {
                    "signal_id": spec.signal_id,
                    "signal_date": str(row["trade_date"]),
                    "asset": ASSET,
                    "signal_value": float(value),
                    "signal_direction": spec.signal_direction,
                    "available_date": str(row["trade_date"]),
                    "data_source": DATA_SOURCE,
                    "candidate_status": spec.status,
                    "signal_family": spec.family,
                    "formula_name": spec.formula_name,
                    "composite_state": row["composite_state"],
                    "liquidity_state": row.get("liquidity_state", ""),
                    "breadth_state": row.get("breadth_state", ""),
                    "activity_state": row.get("activity_state", ""),
                    "concentration_state": row.get("concentration_state", ""),
                    "limit_crowding_state": row.get("limit_crowding_state", ""),
                    "data_quality_state": row["data_quality_state"],
                    "data_scope": row["data_scope"],
                    "price_adjustment": row["price_adjustment"],
                    "model_promotion_allowed": False,
                    "performance_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def build_signal_summary(signal_panel: pd.DataFrame) -> pd.DataFrame:
    if signal_panel.empty:
        return pd.DataFrame()
    data = signal_panel.copy()
    grouped = data.groupby(["signal_id", "signal_direction", "signal_family"], as_index=False)
    return grouped.agg(
        observations=("signal_value", "count"),
        first_signal_date=("signal_date", "min"),
        last_signal_date=("signal_date", "max"),
        mean_signal_value=("signal_value", "mean"),
        median_signal_value=("signal_value", "median"),
        positive_event_count=("signal_value", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
    )


def check_signal_contract(signal_panel: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required_cols = contract.loc[contract["required"].astype(str).str.lower() == "true", "column"].astype(str).tolist()
    missing = sorted(set(required_cols).difference(signal_panel.columns))
    rows.append(
        {
            "check": "required_contract_columns_present",
            "status": "pass" if not missing else "fail",
            "detail": ",".join(missing),
        }
    )
    for col in required_cols:
        if col not in signal_panel.columns:
            continue
        null_count = int(signal_panel[col].isna().sum())
        blank_count = int((signal_panel[col].astype(str).str.len() == 0).sum())
        rows.append(
            {
                "check": f"required_column_non_null_{col}",
                "status": "pass" if null_count + blank_count == 0 else "fail",
                "detail": str(null_count + blank_count),
            }
        )
    directions = set(signal_panel["signal_direction"].astype(str).unique()) if "signal_direction" in signal_panel.columns else set()
    rows.append(
        {
            "check": "signal_direction_values_allowed",
            "status": "pass" if directions.issubset({"positive", "negative"}) else "fail",
            "detail": ",".join(sorted(directions)),
        }
    )
    date_ok = bool((signal_panel["signal_date"].astype(str) == signal_panel["available_date"].astype(str)).all())
    rows.append(
        {
            "check": "available_date_equals_signal_date_for_same_day_state_data",
            "status": "pass" if date_ok else "fail",
            "detail": "V3.48 states are same-day diagnostics",
        }
    )
    rows.append(
        {
            "check": "no_model_or_performance_promotion",
            "status": "pass"
            if not bool(signal_panel["model_promotion_allowed"].astype(bool).any())
            and not bool(signal_panel["performance_claim_allowed"].astype(bool).any())
            else "fail",
            "detail": "candidate registry only",
        }
    )
    forbidden_cols = [
        col
        for col in signal_panel.columns
        if any(keyword in col.lower() for keyword in ["forward_return", "adjusted_return", "total_return", "pnl", "nav", "sharpe"])
    ]
    rows.append(
        {
            "check": "no_return_or_performance_columns",
            "status": "pass" if not forbidden_cols else "fail",
            "detail": ",".join(forbidden_cols),
        }
    )
    return pd.DataFrame(rows)


def build_state_signal_coverage(signal_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "composite_state",
        "liquidity_state",
        "breadth_state",
        "activity_state",
        "concentration_state",
        "limit_crowding_state",
    ]:
        counts = signal_panel.groupby(["signal_id", col], as_index=False).size().rename(columns={"size": "observations"})
        counts = counts.rename(columns={col: "state_value"})
        counts["state_column"] = col
        rows.append(counts[["signal_id", "state_column", "state_value", "observations"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
