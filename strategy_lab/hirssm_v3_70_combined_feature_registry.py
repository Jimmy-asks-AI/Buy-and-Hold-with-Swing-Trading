#!/usr/bin/env python
"""Run HIRSSM V3.70 combined feature registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from combined_feature_registry import (
    CombinedFeatureRegistryConfig,
    build_acceptance_checks,
    build_catalog,
    build_combined_feature_panel,
    build_feature_contract,
    build_feature_coverage_report,
    build_feature_registry,
    build_no_promotion_guard,
    build_quality_checks,
    build_readiness_by_date,
    build_report,
    build_source_alignment_checks,
    validate_inputs,
)
from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "combined_feature_registry_v3_70.json"
TASK_ID = "20260529_v3_70_combined_feature_registry"
VERSION = "V3.70"
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


def load_config(raw: dict[str, Any]) -> CombinedFeatureRegistryConfig:
    thresholds = raw["quality_thresholds"]
    return CombinedFeatureRegistryConfig(
        v3_67_manifest_path=resolve_path(raw["v3_67_manifest_path"]),
        market_panel_path=resolve_path(raw["market_panel_path"]),
        v3_68_manifest_path=resolve_path(raw["v3_68_manifest_path"]),
        industry_panel_path=resolve_path(raw["industry_panel_path"]),
        v3_69_manifest_path=resolve_path(raw["v3_69_manifest_path"]),
        macro_panel_path=resolve_path(raw["macro_panel_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        min_combined_rows=int(thresholds["min_combined_rows"]),
        min_full_source_rows=int(thresholds["min_full_source_rows"]),
        min_validation_ready_rows=int(thresholds["min_validation_ready_rows"]),
        min_feature_registry_rows=int(thresholds["min_feature_registry_rows"]),
        stale_monthly_warn_days=int(thresholds["stale_monthly_warn_days"]),
        stale_daily_warn_days=int(thresholds["stale_daily_warn_days"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "combined_feature_registry.py",
        ROOT / "strategy_lab" / "hirssm_v3_70_combined_feature_registry.py",
        ROOT / "configs" / "combined_feature_registry_v3_70.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_70_combined_feature_registry.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def latest_float(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def latest_int(row: pd.Series | None, column: str) -> int | None:
    if row is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    manifests = {
        "market": read_json(config.v3_67_manifest_path),
        "industry": read_json(config.v3_68_manifest_path),
        "macro": read_json(config.v3_69_manifest_path),
    }
    panels = {
        "market": read_csv(config.market_panel_path),
        "industry": read_csv(config.industry_panel_path),
        "macro": read_csv(config.macro_panel_path),
    }

    input_checks = validate_inputs(manifests, panels)
    combined = build_combined_feature_panel(panels, config)
    registry = build_feature_registry(panels, combined)
    alignment = build_source_alignment_checks(manifests, panels, combined, config)
    coverage = build_feature_coverage_report(registry)
    readiness = build_readiness_by_date(combined)
    no_promotion = build_no_promotion_guard()
    quality = build_quality_checks(combined, registry, alignment, no_promotion, config)
    acceptance = build_acceptance_checks(input_checks, alignment, quality, no_promotion)
    latest_snapshot = combined.sort_values("signal_date").tail(1).copy()
    contract = build_feature_contract(config)
    report = build_report(combined, registry, coverage, readiness, input_checks, alignment, quality, no_promotion)
    catalog = build_catalog(combined, registry, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "combined_panel": output_dir / "combined_feature_panel.csv",
        "registry": output_dir / "feature_registry.csv",
        "alignment": output_dir / "source_alignment_checks.csv",
        "coverage": output_dir / "feature_coverage_report.csv",
        "readiness": output_dir / "readiness_by_year.csv",
        "contract": output_dir / "feature_contract.md",
        "quality": output_dir / "feature_quality_checks.csv",
        "no_promotion": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "latest": output_dir / "latest_feature_snapshot.csv",
        "report": output_dir / "combined_feature_registry_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(combined, output_paths["combined_panel"])
    write_csv(registry, output_paths["registry"])
    write_csv(alignment, output_paths["alignment"])
    write_csv(coverage, output_paths["coverage"])
    write_csv(readiness, output_paths["readiness"])
    write_text(contract, output_paths["contract"])
    write_csv(quality, output_paths["quality"])
    write_csv(no_promotion, output_paths["no_promotion"])
    write_csv(acceptance, output_paths["acceptance"])
    write_csv(latest_snapshot, output_paths["latest"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    output_frames = [input_checks, combined, registry, alignment, coverage, readiness, quality, no_promotion, acceptance, latest_snapshot]
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_columns(output_frames)).lower()})
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "combined_feature_panel_present",
                "status": "pass" if len(combined) >= config.min_combined_rows else "fail",
                "detail": f"rows={len(combined)}",
            },
            {
                "check": "feature_registry_present",
                "status": "pass" if len(registry) >= config.min_feature_registry_rows else "fail",
                "detail": f"rows={len(registry)}",
            },
            {
                "check": "required_v3_70_outputs_present",
                "status": "pass"
                if output_paths["combined_panel"].exists()
                and output_paths["registry"].exists()
                and output_paths["alignment"].exists()
                and output_paths["contract"].exists()
                else "fail",
                "detail": "combined_feature_panel;feature_registry;source_alignment_checks;feature_contract",
            },
            {
                "check": "no_promotion_outputs",
                "status": "pass"
                if not combined["combined_model_promotion_allowed"].astype(bool).any()
                and not combined["combined_portfolio_harness_allowed"].astype(bool).any()
                and not no_promotion.loc[no_promotion["result_type"].isin(["label_layer", "portfolio_harness", "model_promotion"]), "produced"].astype(bool).any()
                else "fail",
                "detail": "feature registry only",
            },
            {
                "check": "forbidden_performance_columns_absent",
                "status": "pass" if not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["combined_panel"],
        output_paths["registry"],
        output_paths["alignment"],
        output_paths["coverage"],
        output_paths["readiness"],
        output_paths["contract"],
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
    warn_count = int(alignment["status"].eq("warn").sum() + quality["status"].eq("warn").sum() + input_checks["status"].eq("warn").sum())
    latest = latest_snapshot.iloc[0] if not latest_snapshot.empty else None
    metrics = {
        "combined_rows": int(len(combined)),
        "full_source_rows": int(combined["all_core_sources_available"].sum()),
        "validation_ready_rows": int(combined["combined_feature_validation_ready"].sum()),
        "feature_registry_rows": int(len(registry)),
        "start_signal_date": str(combined["signal_date"].min()),
        "end_signal_date": str(combined["signal_date"].max()),
        "latest_signal_date": str(latest["signal_date"]) if latest is not None else "",
        "latest_source_count_available": latest_int(latest, "source_count_available"),
        "latest_combined_row_status": str(latest["combined_row_status"]) if latest is not None else "",
        "latest_market_participation_breadth_score": latest_float(latest, "market_market_participation_breadth_score"),
        "latest_industry_breadth_dispersion_score": latest_float(latest, "industry_industry_breadth_dispersion_score"),
        "latest_macro_growth_liquidity_mix_score": latest_float(latest, "macro_macro_growth_liquidity_mix_score"),
        "macro_stale_warning_rows": int(combined["macro_any_stale_warning"].sum()),
        "portfolio_harness_status": "not_run",
        "model_promotion_status": "blocked",
    }

    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.67/V3.68/V3.69 governed feature layers",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_70_combined_feature_registry.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_67_manifest_path),
            rel(config.market_panel_path),
            rel(config.v3_68_manifest_path),
            rel(config.industry_panel_path),
            rel(config.v3_69_manifest_path),
            rel(config.macro_panel_path),
        ],
        "code_refs": [
            "strategy_lab/combined_feature_registry.py",
            "strategy_lab/hirssm_v3_70_combined_feature_registry.py",
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
            "V3.70 is a feature registry, not a signal validation result or trading strategy.",
            "The latest combined rows can be partial-source because V3.68 industry source currently ends earlier than V3.67 and V3.69.",
            "Macro stale warnings are carried forward and must be handled by any later validation task.",
        ],
        "risk_flags": [
            "portfolio_harness_not_run",
            "model_promotion_blocked",
            "partial_latest_source_coverage",
            "macro_staleness_warning",
        ],
        "next_decision": "Proceed to V3.71 guarded feature-label validation using this registry and an explicitly permitted label source.",
        "handoff_summary": "V3.70 joined V3.67 market, V3.68 industry, and V3.69 macro feature panels by signal_use_date and produced governed feature registry outputs only.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
