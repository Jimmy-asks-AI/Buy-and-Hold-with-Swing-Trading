#!/usr/bin/env python
"""Run HIRSSM V3.67 market participation breadth feature layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_participation_breadth_features import (
    MarketParticipationBreadthConfig,
    build_catalog,
    build_feature_contract,
    build_feature_dictionary,
    build_feature_quality_checks,
    build_no_promotion_guard,
    build_participation_feature_panel,
    build_raw_adjustment_guard_checks,
    build_report,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_participation_breadth_v3_67.json"
TASK_ID = "20260529_v3_67_market_participation_breadth_features"
VERSION = "V3.67"
AGENT = "data_steward"


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


def load_config(raw: dict[str, Any]) -> MarketParticipationBreadthConfig:
    return MarketParticipationBreadthConfig(
        v3_66_manifest_path=resolve_path(raw["v3_66_manifest_path"]),
        v3_66_queue_path=resolve_path(raw["v3_66_queue_path"]),
        v3_66_inventory_path=resolve_path(raw["v3_66_inventory_path"]),
        v3_44_guard_manifest_path=resolve_path(raw["v3_44_guard_manifest_path"]),
        v3_44_capability_matrix_path=resolve_path(raw["v3_44_capability_matrix_path"]),
        v3_47_manifest_path=resolve_path(raw["v3_47_manifest_path"]),
        v3_47_feature_panel_path=resolve_path(raw["v3_47_feature_panel_path"]),
        v3_48_manifest_path=resolve_path(raw["v3_48_manifest_path"]),
        v3_48_state_panel_path=resolve_path(raw["v3_48_state_panel_path"]),
        raw_partition_dir=resolve_path(raw["raw_partition_dir"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        min_history=int(raw["feature_parameters"]["min_history"]),
        trailing_window=int(raw["feature_parameters"]["trailing_window"]),
        breadth_short_window=int(raw["feature_parameters"]["breadth_short_window"]),
        breadth_medium_window=int(raw["feature_parameters"]["breadth_medium_window"]),
        breadth_long_window=int(raw["feature_parameters"]["breadth_long_window"]),
        min_feature_rows=int(raw["quality_thresholds"]["min_feature_rows"]),
        min_latest_asset_count=int(raw["quality_thresholds"]["min_latest_asset_count"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_participation_breadth_features.py",
        ROOT / "strategy_lab" / "hirssm_v3_67_market_participation_breadth_features.py",
        ROOT / "configs" / "market_participation_breadth_v3_67.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_67_market_participation_breadth_features.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    raw_guard: pd.DataFrame,
    quality: pd.DataFrame,
    no_promotion: pd.DataFrame,
    panel: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    column_text = " ".join(output_column_names).lower()
    hits = sorted(term for term in forbidden if term in column_text)
    blocked_result_types = {"stock_return_label", "portfolio_harness", "model_promotion"}
    blocked_correctly = not no_promotion.loc[no_promotion["result_type"].isin(blocked_result_types), "produced"].astype(bool).any()
    stock_label_any = panel["stock_return_label_generated"].astype(bool).any() if "stock_return_label_generated" in panel else True
    adjusted_any = panel["adjustment_based_label_generated"].astype(bool).any() if "adjustment_based_label_generated" in panel else True
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if input_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "raw_adjustment_guard_passed",
                "status": "pass" if raw_guard["status"].eq("pass").all() else "fail",
                "detail": ";".join(raw_guard.loc[raw_guard["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "feature_quality_checks_passed",
                "status": "pass" if quality["status"].eq("pass").all() else "fail",
                "detail": ";".join(quality.loc[quality["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "no_stock_or_adjusted_return_labels",
                "status": "pass" if not stock_label_any and not adjusted_any else "fail",
                "detail": f"stock={bool(stock_label_any)};adjustment_based={bool(adjusted_any)}",
            },
            {
                "check": "promotion_outputs_blocked",
                "status": "pass" if blocked_correctly and not hits else "fail",
                "detail": ",".join(hits),
            },
        ]
    )


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
    v3_44_manifest = read_json(config.v3_44_guard_manifest_path)
    v3_47_manifest = read_json(config.v3_47_manifest_path)
    v3_48_manifest = read_json(config.v3_48_manifest_path)
    v3_66_queue = read_csv(config.v3_66_queue_path)
    v3_66_inventory = read_csv(config.v3_66_inventory_path)
    capability_matrix = read_csv(config.v3_44_capability_matrix_path)
    daily_features = read_csv(config.v3_47_feature_panel_path)
    state_panel = read_csv(config.v3_48_state_panel_path)

    input_checks = validate_inputs(
        config,
        v3_66_manifest,
        v3_44_manifest,
        v3_47_manifest,
        v3_48_manifest,
        v3_66_queue,
        capability_matrix,
        v3_66_inventory,
    )
    feature_panel = build_participation_feature_panel(daily_features, state_panel, config)
    raw_guard = build_raw_adjustment_guard_checks(feature_panel, capability_matrix)
    feature_dictionary = build_feature_dictionary(config)
    feature_contract = build_feature_contract(config)
    no_promotion = build_no_promotion_guard()
    quality = build_feature_quality_checks(feature_panel, daily_features, state_panel, raw_guard, config)
    output_frames = [input_checks, feature_panel, raw_guard, feature_dictionary, quality, no_promotion]
    acceptance = build_acceptance_checks(input_checks, raw_guard, quality, no_promotion, feature_panel, output_columns(output_frames))
    latest_snapshot = feature_panel.sort_values("trade_date").tail(1).copy()
    report = build_report(feature_panel, feature_dictionary, raw_guard, quality, input_checks, no_promotion)
    catalog = build_catalog(feature_panel, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "feature_panel": output_dir / "market_participation_breadth_feature_panel.csv",
        "raw_guard": output_dir / "raw_adjustment_guard_checks.csv",
        "feature_dictionary": output_dir / "feature_dictionary.csv",
        "feature_contract": output_dir / "feature_contract.md",
        "quality": output_dir / "feature_quality_checks.csv",
        "no_promotion": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "latest": output_dir / "latest_feature_snapshot.csv",
        "report": output_dir / "market_participation_breadth_feature_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(feature_panel, output_paths["feature_panel"])
    write_csv(raw_guard, output_paths["raw_guard"])
    write_csv(feature_dictionary, output_paths["feature_dictionary"])
    write_text(feature_contract, output_paths["feature_contract"])
    write_csv(quality, output_paths["quality"])
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
                "check": "no_promotion_or_label_outputs",
                "status": "pass"
                if not feature_panel["model_promotion_allowed"].astype(bool).any()
                and not feature_panel["portfolio_harness_allowed"].astype(bool).any()
                and not feature_panel["stock_return_label_generated"].astype(bool).any()
                else "fail",
                "detail": "non-return feature layer only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["feature_panel"],
        output_paths["raw_guard"],
        output_paths["feature_dictionary"],
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
    metrics = {
        "feature_rows": int(len(feature_panel)),
        "history_sufficient_rows": int(feature_panel["history_sufficient"].sum()),
        "start_trade_date": str(feature_panel["trade_date"].min()),
        "end_trade_date": str(feature_panel["trade_date"].max()),
        "latest_asset_count": int(latest_snapshot["asset_count"].iloc[0]) if not latest_snapshot.empty else 0,
        "latest_market_participation_breadth_score": float(latest_snapshot["market_participation_breadth_score"].iloc[0]) if not latest_snapshot.empty else None,
        "raw_guard_pass": bool(raw_guard["status"].eq("pass").all()),
        "stock_return_label_status": "blocked",
        "portfolio_harness_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.66 independent PIT source discovery plus V3.47/V3.48 accepted daily-only lineage",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_67_market_participation_breadth_features.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_66_manifest_path),
            rel(config.v3_66_queue_path),
            rel(config.v3_44_guard_manifest_path),
            rel(config.v3_47_feature_panel_path),
            rel(config.v3_48_state_panel_path),
            rel(config.raw_partition_dir),
        ],
        "code_refs": [
            "strategy_lab/market_participation_breadth_features.py",
            "strategy_lab/hirssm_v3_67_market_participation_breadth_features.py",
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
        "warn_count": 0,
        "limitations": [
            "V3.67 is a feature layer, not a strategy or signal validation result.",
            "Raw Tushare daily prices are used only for cross-sectional breadth and participation diagnostics.",
            "Stock-level return labels remain blocked until adjusted-price or adjustment-factor coverage is available.",
        ],
        "risk_flags": [
            "raw_daily_only_nonreturn_use",
            "stock_return_label_blocked",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Proceed to V3.68 industry breadth and dispersion feature layer, then validate combined features in a later guarded walk-forward task.",
        "handoff_summary": "V3.67 built market participation breadth features from accepted daily-only raw data with an explicit raw-adjustment guard.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
