#!/usr/bin/env python
"""Run HIRSSM V3.68 industry breadth and dispersion feature layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from industry_breadth_dispersion_features import (
    IndustryBreadthDispersionConfig,
    build_acceptance_checks,
    build_catalog,
    build_feature_contract,
    build_feature_dictionary,
    build_feature_quality_checks,
    build_industry_feature_panel,
    build_no_promotion_guard,
    build_report,
    build_signal_validation_plan,
    build_source_quality_checks,
    load_industry_daily,
    load_industry_info,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "industry_breadth_dispersion_v3_68.json"
TASK_ID = "20260529_v3_68_industry_breadth_dispersion_features"
VERSION = "V3.68"
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


def load_config(raw: dict[str, Any]) -> IndustryBreadthDispersionConfig:
    return IndustryBreadthDispersionConfig(
        v3_66_manifest_path=resolve_path(raw["v3_66_manifest_path"]),
        v3_66_queue_path=resolve_path(raw["v3_66_queue_path"]),
        v3_66_inventory_path=resolve_path(raw["v3_66_inventory_path"]),
        industry_daily_dir=resolve_path(raw["industry_daily_dir"]),
        classification_path=resolve_path(raw["classification_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        industry_level=str(raw["feature_parameters"]["industry_level"]),
        min_history=int(raw["feature_parameters"]["min_history"]),
        trailing_window=int(raw["feature_parameters"]["trailing_window"]),
        short_window=int(raw["feature_parameters"]["short_window"]),
        medium_window=int(raw["feature_parameters"]["medium_window"]),
        long_window=int(raw["feature_parameters"]["long_window"]),
        slow_window=int(raw["feature_parameters"]["slow_window"]),
        min_feature_rows=int(raw["quality_thresholds"]["min_feature_rows"]),
        min_industry_count=int(raw["quality_thresholds"]["min_industry_count"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "industry_breadth_dispersion_features.py",
        ROOT / "strategy_lab" / "hirssm_v3_68_industry_breadth_dispersion_features.py",
        ROOT / "configs" / "industry_breadth_dispersion_v3_68.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_68_industry_breadth_dispersion_features.json",
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
    input_checks = validate_inputs(config, v3_66_manifest, v3_66_queue, v3_66_inventory)
    industry_info = load_industry_info(config)
    industry_long, source_quality = load_industry_daily(config, industry_info)
    feature_panel = build_industry_feature_panel(industry_long, industry_info, config)
    source_checks = build_source_quality_checks(source_quality, feature_panel, config)
    feature_dictionary = build_feature_dictionary(config)
    validation_plan = build_signal_validation_plan(config)
    no_promotion = build_no_promotion_guard()
    quality_checks = build_feature_quality_checks(feature_panel, source_quality, source_checks, config)
    output_frames = [input_checks, source_quality, source_checks, feature_panel, feature_dictionary, validation_plan, quality_checks, no_promotion]
    acceptance = build_acceptance_checks(input_checks, source_checks, quality_checks, no_promotion, output_columns(output_frames))
    latest_snapshot = feature_panel.sort_values("trade_date").tail(1).copy()
    feature_contract = build_feature_contract(config)
    report = build_report(feature_panel, feature_dictionary, validation_plan, input_checks, source_checks, quality_checks, no_promotion)
    catalog = build_catalog(feature_panel, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "source_quality": output_dir / "source_quality_report.csv",
        "source_checks": output_dir / "source_quality_checks.csv",
        "feature_panel": output_dir / "industry_breadth_dispersion_feature_panel.csv",
        "feature_dictionary": output_dir / "feature_dictionary.csv",
        "validation_plan": output_dir / "signal_validation_plan.csv",
        "feature_contract": output_dir / "feature_contract.md",
        "quality": output_dir / "feature_quality_checks.csv",
        "no_promotion": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "latest": output_dir / "latest_feature_snapshot.csv",
        "report": output_dir / "industry_breadth_dispersion_feature_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(source_quality, output_paths["source_quality"])
    write_csv(source_checks, output_paths["source_checks"])
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
                "check": "required_v3_68_outputs_present",
                "status": "pass"
                if output_paths["feature_panel"].exists()
                and output_paths["validation_plan"].exists()
                and output_paths["feature_contract"].exists()
                else "fail",
                "detail": "feature_panel;signal_validation_plan;feature_contract",
            },
            {
                "check": "no_promotion_outputs",
                "status": "pass"
                if not feature_panel["model_promotion_allowed"].astype(bool).any()
                and not feature_panel["portfolio_harness_allowed"].astype(bool).any()
                and not feature_panel["component_snapshot_used"].astype(bool).any()
                else "fail",
                "detail": "feature layer only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["source_quality"],
        output_paths["source_checks"],
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
    latest = latest_snapshot.iloc[0] if not latest_snapshot.empty else None
    metrics = {
        "feature_rows": int(len(feature_panel)),
        "history_sufficient_rows": int(feature_panel["history_sufficient"].sum()),
        "start_trade_date": str(feature_panel["trade_date"].min()),
        "end_trade_date": str(feature_panel["trade_date"].max()),
        "latest_industry_count": int(latest["industry_count"]) if latest is not None else 0,
        "latest_above_ma60_ratio": float(latest["above_ma60_ratio"]) if latest is not None else None,
        "latest_industry_ret20_dispersion": float(latest["industry_ret20_dispersion"]) if latest is not None else None,
        "latest_industry_breadth_dispersion_score": float(latest["industry_breadth_dispersion_score"]) if latest is not None else None,
        "latest_composite_industry_state": str(latest["composite_industry_state"]) if latest is not None else "",
        "portfolio_harness_status": "not_run",
        "model_promotion_status": "blocked",
        "component_snapshot_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.66 independent PIT source discovery plus SW industry daily history",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_68_industry_breadth_dispersion_features.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_66_manifest_path),
            rel(config.v3_66_queue_path),
            rel(config.v3_66_inventory_path),
            rel(config.industry_daily_dir),
            rel(config.classification_path),
        ],
        "code_refs": [
            "strategy_lab/industry_breadth_dispersion_features.py",
            "strategy_lab/hirssm_v3_68_industry_breadth_dispersion_features.py",
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
            "V3.68 is a feature layer, not a strategy or signal validation result.",
            "SW industry price-index history is not official total-return evidence.",
            "Current industry components and latest weights are not used.",
        ],
        "risk_flags": [
            "industry_index_price_feature_only",
            "component_snapshot_blocked",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Proceed to V3.69 macro growth-liquidity feature layer, then run a later guarded validation task combining V3.67/V3.68/V3.69 features.",
        "handoff_summary": "V3.68 built SW industry breadth, dispersion, liquidity breadth, and rotation-persistence features with no portfolio or model promotion.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
