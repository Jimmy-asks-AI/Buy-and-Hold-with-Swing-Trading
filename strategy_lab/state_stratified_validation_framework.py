"""State-stratified validation contracts for future signal research.

The framework is deliberately separate from performance validation. It can
prepare state coverage, schemas, and gates using V3.48 state tags, but it must
not produce signal performance conclusions without adjusted/PIT return labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_STATE_COLUMNS = {
    "trade_date",
    "history_available",
    "data_scope",
    "price_adjustment",
    "data_quality_state",
    "composite_state",
    "liquidity_state",
    "breadth_state",
    "activity_state",
    "concentration_state",
    "limit_crowding_state",
    "model_usage_allowed",
    "backtest_usage_allowed",
}

SIGNAL_CONTRACT_ROWS = [
    {
        "column": "signal_id",
        "required": True,
        "description": "Stable identifier for a candidate signal or expert.",
        "example": "breadth_recovery_candidate_v1",
    },
    {
        "column": "signal_date",
        "required": True,
        "description": "Date when the signal is known and actionable.",
        "example": "20260528",
    },
    {
        "column": "asset",
        "required": False,
        "description": "Asset identifier. Use MARKET for market-level signals.",
        "example": "MARKET",
    },
    {
        "column": "signal_value",
        "required": True,
        "description": "Numeric raw signal value available at signal_date.",
        "example": "0.73",
    },
    {
        "column": "signal_direction",
        "required": True,
        "description": "Expected relation with future adjusted returns: positive or negative.",
        "example": "positive",
    },
    {
        "column": "available_date",
        "required": True,
        "description": "Point-in-time availability date. Must be no later than live-use date and no earlier than source release.",
        "example": "20260528",
    },
    {
        "column": "data_source",
        "required": True,
        "description": "Source lineage for signal inputs.",
        "example": "accepted_daily_only_state_monitor_v3_48",
    },
]

LABEL_CONTRACT_ROWS = [
    {
        "column": "signal_date",
        "required": True,
        "description": "Signal date to join against signal panel.",
        "example": "20260528",
    },
    {
        "column": "asset",
        "required": True,
        "description": "Asset identifier matching the signal asset, or MARKET.",
        "example": "MARKET",
    },
    {
        "column": "horizon",
        "required": True,
        "description": "Forward horizon in trading days.",
        "example": "20",
    },
    {
        "column": "forward_adjusted_return",
        "required": True,
        "description": "Forward adjusted or total return label; cannot be derived from unadjusted raw close.",
        "example": "0.0342",
    },
    {
        "column": "return_basis",
        "required": True,
        "description": "Must be adjusted_return or total_return.",
        "example": "adjusted_return",
    },
    {
        "column": "label_available_date",
        "required": True,
        "description": "Date when the label becomes observable; used for leakage checks.",
        "example": "20260626",
    },
    {
        "column": "price_adjustment_source",
        "required": True,
        "description": "Adjustment factor or total-return source lineage.",
        "example": "tushare.adj_factor_or_equivalent_pit_source",
    },
]

VALIDATION_METRIC_ROWS = [
    {
        "metric": "rank_ic_by_state",
        "level": "cross_sectional",
        "minimum_input": "signal_value + forward_adjusted_return",
        "purpose": "Test monotonic relation between signal and future adjusted returns within each state.",
    },
    {
        "metric": "icir_by_state",
        "level": "time_series",
        "minimum_input": "daily rank_ic_by_state history",
        "purpose": "Evaluate stability of signal effect by state.",
    },
    {
        "metric": "hit_rate_by_state",
        "level": "signal_bucket",
        "minimum_input": "signal direction + forward_adjusted_return",
        "purpose": "Check whether directional success survives state stratification.",
    },
    {
        "metric": "effect_size_vs_unconditional",
        "level": "state",
        "minimum_input": "state labels + forward_adjusted_return",
        "purpose": "Compare state-conditional outcomes with the unconditional baseline.",
    },
    {
        "metric": "purged_walk_forward_state_stability",
        "level": "validation_protocol",
        "minimum_input": "signal panel + labels + state panel",
        "purpose": "Prevent temporal leakage and detect state-specific overfitting.",
    },
    {
        "metric": "negative_control_check",
        "level": "validation_protocol",
        "minimum_input": "randomized or lag-broken signal variants",
        "purpose": "Detect accidental leakage or data-mining artifacts.",
    },
]


@dataclass(frozen=True)
class ValidationGateConfig:
    min_state_observations: int
    min_unique_states: int
    required_return_basis: tuple[str, ...]
    horizons: tuple[int, ...]
    embargo_days: int
    purge_days: int


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def validate_state_panel_schema(state_panel: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED_STATE_COLUMNS.difference(state_panel.columns))
    issues = [f"missing_state_columns={missing}"] if missing else []
    if "data_scope" in state_panel.columns:
        values = set(state_panel["data_scope"].astype(str).unique())
        if values != {"accepted_processed_daily_only"}:
            issues.append(f"unexpected_data_scope={sorted(values)}")
    if "price_adjustment" in state_panel.columns:
        values = set(state_panel["price_adjustment"].astype(str).unique())
        if values != {"none_raw"}:
            issues.append(f"unexpected_price_adjustment={sorted(values)}")
    if "model_usage_allowed" in state_panel.columns and bool(state_panel["model_usage_allowed"].any()):
        issues.append("state_panel_model_usage_allowed_true")
    if "backtest_usage_allowed" in state_panel.columns and bool(state_panel["backtest_usage_allowed"].any()):
        issues.append("state_panel_backtest_usage_allowed_true")
    return issues


def build_state_coverage_summary(state_panel: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    valid = state_panel.loc[state_panel["history_available"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for state_column in state_columns:
        counts = valid[state_column].astype(str).value_counts(dropna=False)
        for state_value, count in counts.items():
            rows.append(
                {
                    "state_column": state_column,
                    "state_value": state_value,
                    "observations": int(count),
                    "share": float(count / max(len(valid), 1)),
                    "usable_for_stratification": bool(count > 0),
                }
            )
    return pd.DataFrame(rows)


def build_state_stratification_plan(config: ValidationGateConfig, state_coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in state_coverage.itertuples(index=False):
        enough = int(row.observations) >= config.min_state_observations
        rows.append(
            {
                "state_column": row.state_column,
                "state_value": row.state_value,
                "observations": int(row.observations),
                "minimum_observations": config.min_state_observations,
                "validation_role": "primary_stratum" if enough else "monitor_only_low_sample",
                "sample_gate_pass": enough,
            }
        )
    return pd.DataFrame(rows)


def dataframe_from_contract(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def validate_label_contract(label_path: Path | None, config: ValidationGateConfig) -> tuple[bool, str]:
    if label_path is None:
        return False, "missing_adjusted_pit_return_label_path"
    if not label_path.exists():
        return False, f"label_path_not_found={label_path}"
    labels = pd.read_csv(label_path, encoding="utf-8-sig", low_memory=False)
    required = {row["column"] for row in LABEL_CONTRACT_ROWS if row["required"]}
    missing = sorted(required.difference(labels.columns))
    if missing:
        return False, f"label_missing_columns={missing}"
    basis = set(labels["return_basis"].astype(str).unique())
    allowed = set(config.required_return_basis)
    if not basis.issubset(allowed):
        return False, f"return_basis_not_allowed={sorted(basis.difference(allowed))}"
    return True, "label_contract_ready"


def validate_signal_contract(signal_path: Path | None) -> tuple[bool, str]:
    if signal_path is None:
        return False, "missing_signal_panel_path"
    if not signal_path.exists():
        return False, f"signal_path_not_found={signal_path}"
    signals = pd.read_csv(signal_path, encoding="utf-8-sig", low_memory=False)
    required = {row["column"] for row in SIGNAL_CONTRACT_ROWS if row["required"]}
    missing = sorted(required.difference(signals.columns))
    if missing:
        return False, f"signal_missing_columns={missing}"
    return True, "signal_contract_ready"


def build_readiness_checks(
    state_panel: pd.DataFrame,
    state_coverage: pd.DataFrame,
    label_ready: tuple[bool, str],
    signal_ready: tuple[bool, str],
    config: ValidationGateConfig,
) -> pd.DataFrame:
    schema_issues = validate_state_panel_schema(state_panel)
    enough_states = state_coverage.loc[state_coverage["observations"] >= config.min_state_observations]
    rows = [
        {
            "check": "state_panel_schema_ready",
            "status": "pass" if not schema_issues else "fail",
            "detail": ";".join(schema_issues),
        },
        {
            "check": "state_history_available",
            "status": "pass" if int(state_panel["history_available"].sum()) > 0 else "fail",
            "detail": str(int(state_panel["history_available"].sum())),
        },
        {
            "check": "sufficient_state_buckets_for_future_validation",
            "status": "pass" if int(enough_states.shape[0]) >= config.min_unique_states else "fail",
            "detail": f"eligible_buckets={int(enough_states.shape[0])};required={config.min_unique_states}",
        },
        {
            "check": "signal_contract_ready",
            "status": "pass" if signal_ready[0] else "blocked",
            "detail": signal_ready[1],
        },
        {
            "check": "adjusted_pit_label_contract_ready",
            "status": "pass" if label_ready[0] else "blocked",
            "detail": label_ready[1],
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "pass" if signal_ready[0] and label_ready[0] else "blocked",
            "detail": "requires signal panel and adjusted/PIT return labels",
        },
        {
            "check": "framework_ready_without_performance_claims",
            "status": "pass",
            "detail": "state-stratified framework artifacts can be used after labels arrive",
        },
    ]
    return pd.DataFrame(rows)


def build_no_result_guard(readiness_checks: pd.DataFrame) -> pd.DataFrame:
    blocked = readiness_checks.loc[readiness_checks["status"] == "blocked"]
    return pd.DataFrame(
        [
            {
                "result_type": "state_stratified_performance_validation",
                "produced": False,
                "blocked": bool(not blocked.empty),
                "reason": ";".join(f"{row.check}:{row.detail}" for row in blocked.itertuples(index=False)),
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.49 is a validation framework only; portfolio backtests require adjusted/PIT returns and a separate approved harness.",
            },
        ]
    )


def build_negative_control_plan(config: ValidationGateConfig) -> pd.DataFrame:
    rows = [
        {
            "control": "date_shuffle_within_year",
            "purpose": "Break signal-date alignment while preserving broad calendar conditions.",
            "pass_condition": "State-stratified performance should disappear or weaken materially.",
        },
        {
            "control": "asset_shuffle_within_date",
            "purpose": "Break cross-sectional signal-to-asset mapping.",
            "pass_condition": "Cross-sectional IC should collapse toward zero.",
        },
        {
            "control": "lag_broken_signal",
            "purpose": f"Shift signal by at least purge_days={config.purge_days} plus embargo_days={config.embargo_days}.",
            "pass_condition": "Signal effect should not survive if it came from timing leakage.",
        },
        {
            "control": "state_label_shuffle_after_history_gate",
            "purpose": "Check whether state segmentation itself is overfit.",
            "pass_condition": "State-specific edge should not concentrate in shuffled states.",
        },
    ]
    return pd.DataFrame(rows)
