#!/usr/bin/env python
"""Run HIRSSM V3.69 macro growth-liquidity feature layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from macro_growth_liquidity_features import (
    MacroGrowthLiquidityConfig,
    add_macro_features,
    build_acceptance_checks,
    build_asof_join_checks,
    build_catalog,
    build_feature_contract,
    build_feature_dictionary,
    build_feature_quality_checks,
    build_macro_asof_panel,
    build_no_promotion_guard,
    build_report,
    build_series_coverage_report,
    build_signal_validation_plan,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "macro_growth_liquidity_v3_69.json"
TASK_ID = "20260529_v3_69_macro_growth_liquidity_features"
VERSION = "V3.69"
AGENT = "factor_researcher"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def load_config(raw: dict[str, Any]) -> MacroGrowthLiquidityConfig:
    return MacroGrowthLiquidityConfig(
        v3_66_manifest_path=resolve_path(raw["v3_66_manifest_path"]),
        v3_66_queue_path=resolve_path(raw["v3_66_queue_path"]),
        v3_66_inventory_path=resolve_path(raw["v3_66_inventory_path"]),
        macro_panel_path=resolve_path(raw["macro_panel_path"]),
        trade_calendar_path=resolve_path(raw["trade_calendar_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        start_date=str(raw["feature_parameters"]["start_date"]),
        min_history=int(raw["feature_parameters"]["min_history"]),
        trailing_window=int(raw["feature_parameters"]["trailing_window"]),
        short_window=int(raw["feature_parameters"]["short_window"]),
        medium_window=int(raw["feature_parameters"]["medium_window"]),
        long_window=int(raw["feature_parameters"]["long_window"]),
        min_feature_rows=int(raw["quality_thresholds"]["min_feature_rows"]),
        min_history_sufficient_rows=int(raw["quality_thresholds"]["min_history_sufficient_rows"]),
        min_required_series=int(raw["quality_thresholds"]["min_required_series"]),
        stale_monthly_warn_days=int(raw["quality_thresholds"]["stale_monthly_warn_days"]),
        stale_daily_warn_days=int(raw["quality_thresholds"]["stale_daily_warn_days"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "macro_growth_liquidity_features.py",
        ROOT / "strategy_lab" / "hirssm_v3_69_macro_growth_liquidity_features.py",
        ROOT / "configs" / "macro_growth_liquidity_v3_69.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_69_macro_growth_liquidity_features.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    v3_66_manifest = read_json(config.v3_66_manifest_path)
    v3_66_queue = read_csv(config.v3_66_queue_path)
    v3_66_inventory = read_csv(config.v3_66_inventory_path)
    macro_panel = read_csv(config.macro_panel_path)
    trade_calendar = read_csv(config.trade_calendar_path)

    input_checks = validate_inputs(config, v3_66_manifest, v3_66_queue, v3_66_inventory, macro_panel, trade_calendar)
    series_coverage = build_series_coverage_report(macro_panel)
    asof_base = build_macro_asof_panel(macro_panel, trade_calendar, config)
    feature_panel = add_macro_features(asof_base, config)
    asof_checks = build_asof_join_checks(feature_panel, config)
    feature_dictionary = build_feature_dictionary(config)
    validation_plan = build_signal_validation_plan(config)
    no_promotion = build_no_promotion_guard()
    quality_checks = build_feature_quality_checks(feature_panel, asof_checks, input_checks, config)
    output_frames = [input_checks, series_coverage, asof_checks, feature_panel, feature_dictionary, validation_plan, quality_checks, no_promotion]
    acceptance = build_acceptance_checks(input_checks, asof_checks, quality_checks, no_promotion, output_columns(output_frames))
    latest_snapshot = feature_panel.sort_values("trade_date").tail(1).copy()
    feature_contract = build_feature_contract(config)
    report = build_report(feature_panel, series_coverage, asof_checks, feature_dictionary, validation_plan, input_checks, quality_checks, no_promotion)
    catalog = build_catalog(feature_panel, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "series_coverage": output_dir / "series_coverage_report.csv",
        "asof_checks": output_dir / "asof_join_checks.csv",
        "feature_panel": output_dir / "macro_asof_feature_panel.csv",
        "feature_dictionary": output_dir / "feature_dictionary.csv",
        "validation_plan": output_dir / "signal_validation_plan.csv",
        "feature_contract": output_dir / "feature_contract.md",
        "quality": output_dir / "feature_quality_checks.csv",
        "no_promotion": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "latest": output_dir / "latest_feature_snapshot.csv",
        "report": output_dir / "macro_growth_liquidity_feature_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(series_coverage, output_paths["series_coverage"])
    write_csv(asof_checks, output_paths["asof_checks"])
    write_csv(feature_panel, output_paths["feature_panel"])
    write_csv(feature_dictionary, output_paths["feature_dictionary"])
    write_csv(validation_plan, output_paths["validation_plan"])
    write_text(feature_contract, output_paths["feature_contract"])
    write_csv(quality_checks, output_paths["quality"])
    write_csv(no_promotion, output_paths["no_promotion"])
    write_csv(acceptance, output_paths["acceptance"])
    write_csv(latest_snapshot, output_paths["latest"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "feature_panel_present",
                "status": "pass" if len(feature_panel) >= config.min_feature_rows else "fail",
                "detail": f"rows={len(feature_panel)}",
            },
            {
                "check": "required_v3_69_outputs_present",
                "status": "pass"
                if output_paths["feature_panel"].exists()
                and output_paths["asof_checks"].exists()
                and output_paths["feature_contract"].exists()
                else "fail",
                "detail": "macro_asof_feature_panel;asof_join_checks;feature_contract",
            },
            {
                "check": "no_promotion_outputs",
                "status": "pass"
                if not feature_panel["model_promotion_allowed"].astype(bool).any()
                and not feature_panel["portfolio_harness_allowed"].astype(bool).any()
                else "fail",
                "detail": "feature layer only",
            },
            {
                "check": "asof_join_no_future_dates",
                "status": "pass" if not asof_checks["status"].eq("fail").any() else "fail",
                "detail": ";".join(asof_checks.loc[asof_checks["status"].eq("fail"), "series_id"].astype(str)),
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["series_coverage"],
        output_paths["asof_checks"],
        output_paths["feature_panel"],
        output_paths["feature_dictionary"],
        output_paths["validation_plan"],
        output_paths["feature_contract"],
        output_paths["quality"],
        output_paths["no_promotion"],
        output_paths["acceptance"],
        output_paths["latest"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    warn_count = int(quality_checks["status"].eq("warn").sum() + asof_checks["status"].eq("warn").sum())
    latest = latest_snapshot.iloc[0] if not latest_snapshot.empty else None
    metrics = {
        "feature_rows": int(len(feature_panel)),
        "history_sufficient_rows": int(feature_panel["history_sufficient"].sum()),
        "start_trade_date": str(feature_panel["trade_date"].min()),
        "end_trade_date": str(feature_panel["trade_date"].max()),
        "latest_required_series_count": int(latest["macro_required_series_available_count"]) if latest is not None else 0,
        "latest_macro_growth_liquidity_mix_score": float(latest["macro_growth_liquidity_mix_score"]) if latest is not None else None,
        "latest_macro_risk_pressure_score": float(latest["macro_risk_pressure_score"]) if latest is not None else None,
        "latest_composite_macro_state": str(latest["composite_macro_state"]) if latest is not None else "",
        "asof_join_fail_count": int(asof_checks["status"].eq("fail").sum()),
        "asof_stale_warn_count": int(asof_checks["status"].eq("warn").sum()),
        "portfolio_harness_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.66 independent PIT source discovery plus V3.27 macro PIT panel",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_69_macro_growth_liquidity_features.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_66_manifest_path),
            rel(config.v3_66_queue_path),
            rel(config.v3_66_inventory_path),
            rel(config.macro_panel_path),
            rel(config.trade_calendar_path),
        ],
        "code_refs": [
            "strategy_lab/macro_growth_liquidity_features.py",
            "strategy_lab/hirssm_v3_69_macro_growth_liquidity_features.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in all_outputs],
        "changed_files": build_changed_files(all_outputs).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "V3.69 is a feature layer, not a strategy or signal validation result.",
            "Some macro series are vendor snapshots or release-event histories, not complete vintage databases.",
            "Stale monthly macro series are flagged and must be considered in later validation.",
        ],
        "risk_flags": [
            "macro_vintage_limitation",
            "monthly_macro_staleness_warning",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Proceed to V3.70 combined feature registry joining V3.67/V3.68/V3.69 by signal_use_date, then run guarded validation against a permitted label source.",
        "handoff_summary": "V3.69 built available-date as-of macro growth-liquidity features with explicit stale-data and no-promotion guards.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
