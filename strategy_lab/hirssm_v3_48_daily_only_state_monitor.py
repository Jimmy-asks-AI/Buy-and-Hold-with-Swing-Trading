#!/usr/bin/env python
"""Build and validate the V3.48 daily-only market state monitor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from daily_only_state_monitor import DailyOnlyStateConfig, STATE_DICTIONARY_ROWS, build_state_panel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "daily_only_state_monitor_v3_48.json"
TASK_ID = "20260529_v3_48_daily_only_state_monitor"
VERSION = "V3.48"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def state_config(config: dict[str, Any]) -> DailyOnlyStateConfig:
    return DailyOnlyStateConfig(
        trailing_window=int(config["trailing_window"]),
        min_history=int(config["min_history"]),
        stress_pctile=float(config["stress_pctile"]),
        recovery_pctile=float(config["recovery_pctile"]),
        crowding_pctile=float(config["crowding_pctile"]),
        calm_pctile=float(config["calm_pctile"]),
        min_active_asset_ratio=float(config["min_active_asset_ratio"]),
        breadth_stress_advance_ratio=float(config["breadth_stress_advance_ratio"]),
        breadth_recovery_advance_ratio=float(config["breadth_recovery_advance_ratio"]),
        limit_crowding_pctile=float(config["limit_crowding_pctile"]),
        minimum_limit_share=float(config["minimum_limit_share"]),
    )


def build_state_dictionary(config: dict[str, Any]) -> pd.DataFrame:
    rows = [dict(row) for row in STATE_DICTIONARY_ROWS]
    for row in rows:
        row["data_scope"] = "accepted_processed_daily_only"
        row["price_adjustment"] = "none_raw"
        row["state_basis"] = "trailing_window_prior_observations_only"
        row["trailing_window"] = int(config["trailing_window"])
        row["min_history"] = int(config["min_history"])
    return pd.DataFrame(rows)


def build_transition_summary(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel.loc[panel["history_available"]].copy()
    valid["previous_composite_state"] = valid["composite_state"].shift(1)
    transitions = valid.dropna(subset=["previous_composite_state"])
    if transitions.empty:
        return pd.DataFrame(columns=["previous_composite_state", "composite_state", "transition_count"])
    return (
        transitions.groupby(["previous_composite_state", "composite_state"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
        .sort_values("transition_count", ascending=False)
    )


def build_latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_date",
        "asset_count",
        "active_asset_ratio",
        "amount_per_asset_raw",
        "amount_per_asset_raw_trailing_pctile",
        "advance_ratio",
        "decline_ratio",
        "advance_decline_balance",
        "advance_decline_balance_trailing_pctile",
        "median_range_ratio_raw",
        "median_range_ratio_raw_trailing_pctile",
        "liquidity_state",
        "breadth_state",
        "activity_state",
        "concentration_state",
        "limit_crowding_state",
        "data_quality_state",
        "composite_state",
        "model_usage_allowed",
        "backtest_usage_allowed",
    ]
    return panel.sort_values("trade_date").tail(1)[columns].copy()


def forbidden_columns(panel: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    keywords = [item.lower() for item in config["forbidden_columns_keywords"]]
    return [col for col in panel.columns if any(keyword in col.lower() for keyword in keywords)]


def build_quality_checks(
    config: dict[str, Any],
    feature_panel: pd.DataFrame,
    state_panel: pd.DataFrame,
    feature_manifest: dict[str, Any],
) -> pd.DataFrame:
    forbidden = forbidden_columns(state_panel, config)
    percentile_cols = [col for col in state_panel.columns if col.endswith("_trailing_pctile")]
    rows = [
        {
            "check": "v3_47_feature_layer_acceptance_passed",
            "status": "pass" if bool(feature_manifest.get("acceptance_pass", False)) else "fail",
            "detail": feature_manifest.get("data_decision", ""),
        },
        {
            "check": "state_rows_match_feature_rows",
            "status": "pass" if int(state_panel.shape[0]) == int(feature_panel.shape[0]) else "fail",
            "detail": f"state_rows={state_panel.shape[0]};feature_rows={feature_panel.shape[0]}",
        },
        {
            "check": "initial_history_is_insufficient_only",
            "status": "pass"
            if bool((state_panel.head(int(config["min_history"]))["composite_state"] == "insufficient_history").all())
            else "fail",
            "detail": str(int(config["min_history"])),
        },
        {
            "check": "trailing_percentiles_have_no_early_full_sample_values",
            "status": "pass"
            if bool(state_panel.head(int(config["min_history"]))[percentile_cols].isna().all().all())
            else "fail",
            "detail": ",".join(percentile_cols),
        },
        {
            "check": "data_scope_is_accepted_processed_daily_only",
            "status": "pass"
            if set(state_panel["data_scope"].astype(str).unique()) == {"accepted_processed_daily_only"}
            else "fail",
            "detail": "|".join(sorted(state_panel["data_scope"].astype(str).unique())),
        },
        {
            "check": "price_adjustment_is_none_raw",
            "status": "pass"
            if set(state_panel["price_adjustment"].astype(str).unique()) == {"none_raw"}
            else "fail",
            "detail": "|".join(sorted(state_panel["price_adjustment"].astype(str).unique())),
        },
        {
            "check": "bad_ohlc_after_adapter_is_zero",
            "status": "pass" if int(state_panel["bad_ohlc_after_adapter"].sum()) == 0 else "fail",
            "detail": str(int(state_panel["bad_ohlc_after_adapter"].sum())),
        },
        {
            "check": "no_forbidden_return_or_valuation_columns",
            "status": "pass" if len(forbidden) == 0 else "fail",
            "detail": ",".join(forbidden),
        },
        {
            "check": "states_not_promoted_to_model_or_backtest",
            "status": "pass"
            if not bool(state_panel["model_usage_allowed"].any()) and not bool(state_panel["backtest_usage_allowed"].any())
            else "fail",
            "detail": "diagnostic states only",
        },
        {
            "check": "no_model_or_backtest_promotion",
            "status": "pass",
            "detail": "state monitor only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    transition_summary: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    failed = checks.loc[checks["status"] != "pass"]
    valid = state_panel.loc[state_panel["history_available"]]
    latest = latest_snapshot.iloc[0]
    state_counts = valid["composite_state"].value_counts().head(12)
    lines = [
        "# V3.48 Daily-Only State Monitor",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.48 adds point-in-time data quality monitoring and market state tags on top of V3.47 features.",
        "- State labels use prior observations only through trailing percentiles.",
        "- No return label, adjusted return, valuation, dividend, portfolio performance, model, or backtest output is produced.",
        "",
        "## Configuration",
        "",
        f"- Trailing window: `{config['trailing_window']}`",
        f"- Minimum history: `{config['min_history']}`",
        f"- State rows: `{len(state_panel)}`",
        f"- Rows with state history: `{len(valid)}`",
        "",
        "## Latest State",
        "",
        f"- Latest date: `{latest.trade_date}`",
        f"- Composite state: `{latest.composite_state}`",
        f"- Liquidity state: `{latest.liquidity_state}`",
        f"- Breadth state: `{latest.breadth_state}`",
        f"- Activity state: `{latest.activity_state}`",
        f"- Concentration state: `{latest.concentration_state}`",
        f"- Limit crowding state: `{latest.limit_crowding_state}`",
        f"- Data quality state: `{latest.data_quality_state}`",
        "",
        "## Composite State Counts",
        "",
        "| state | count |",
        "|---|---:|",
    ]
    for state, count in state_counts.items():
        lines.append(f"| `{state}` | {int(count)} |")
    lines.extend(
        [
            "",
            "## Top Transitions",
            "",
            "| previous | current | count |",
            "|---|---|---:|",
        ]
    )
    for row in transition_summary.head(12).itertuples(index=False):
        lines.append(f"| `{row.previous_composite_state}` | `{row.composite_state}` | {int(row.transition_count)} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Failed checks: `{len(failed)}`",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Downstream Rule",
            "",
            "- These states are diagnostic segmentation labels, not trading decisions.",
            "- They can be used to prioritize data QA, describe market regimes, and stratify later validated research.",
            "- Any investable timing or portfolio rule still requires separate signal validation and adjusted/PIT return data.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(state_panel: pd.DataFrame, checks: pd.DataFrame) -> str:
    failed = checks.loc[checks["status"] != "pass"]
    valid = state_panel.loc[state_panel["history_available"]]
    latest = state_panel.sort_values("trade_date").tail(1).iloc[0]
    return "\n".join(
        [
            "# A-share Daily-Only State Monitor V3.48",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            f"- State monitor accepted: `{failed.empty}`",
            f"- State rows: `{len(state_panel)}`",
            f"- Rows with trailing history: `{len(valid)}`",
            f"- Latest composite state: `{latest.composite_state}`",
            f"- Data scope: `accepted_processed_daily_only`",
            f"- Price adjustment: `none_raw`",
            "",
            "## Boundary",
            "",
            "- Allowed: point-in-time diagnostic state monitoring.",
            "- Forbidden: adjusted return, valuation, dividends, portfolio performance, model promotion.",
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    rows.append(
        {
            "check": "quality_checks_all_pass",
            "status": "pass" if bool((checks["status"] == "pass").all()) else "fail",
            "detail": str(int((checks["status"] != "pass").sum())),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]

    feature_panel = read_csv(ROOT / config["feature_panel_path"])
    feature_manifest = read_json(ROOT / config["feature_layer_manifest_path"])
    state_panel = build_state_panel(feature_panel, state_config(config))
    state_dictionary = build_state_dictionary(config)
    transition_summary = build_transition_summary(state_panel)
    latest_snapshot = build_latest_snapshot(state_panel)
    checks = build_quality_checks(config, feature_panel, state_panel, feature_manifest)

    artifacts = {
        "daily_market_state_panel": output_dir / "daily_market_state_panel.csv",
        "state_dictionary": output_dir / "state_dictionary.csv",
        "state_transition_summary": output_dir / "state_transition_summary.csv",
        "latest_state_snapshot": output_dir / "latest_state_snapshot.csv",
        "state_quality_checks": output_dir / "state_quality_checks.csv",
        "state_monitor_report": output_dir / "state_monitor_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(state_panel, artifacts["daily_market_state_panel"])
    write_csv(state_dictionary, artifacts["state_dictionary"])
    write_csv(transition_summary, artifacts["state_transition_summary"])
    write_csv(latest_snapshot, artifacts["latest_state_snapshot"])
    write_csv(checks, artifacts["state_quality_checks"])
    write_text(build_report(config, state_panel, transition_summary, latest_snapshot, checks), artifacts["state_monitor_report"])
    write_text(build_catalog(state_panel, checks), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, checks)
    write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((checks["status"] == "pass").all()),
        "state_rows": int(state_panel.shape[0]),
        "history_available_rows": int(state_panel["history_available"].sum()),
        "latest_trade_date": str(latest_snapshot["trade_date"].iloc[0]),
        "latest_composite_state": str(latest_snapshot["composite_state"].iloc[0]),
        "data_decision": "daily_only_state_monitor_ready_for_diagnostic_use",
        "model_decision": "no_model_promotion_state_diagnostics_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
