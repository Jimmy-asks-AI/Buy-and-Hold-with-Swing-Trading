#!/usr/bin/env python
"""Run HIRSSM V3.62 lag-safe non-price signal repair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lag_safe_signal_repair import (
    LagSafeRepairConfig,
    build_acceptance_checks,
    build_catalog,
    build_lag_discipline_checks,
    build_market_move_artifact_screen,
    build_no_promotion_guard,
    build_repaired_signal_panel,
    build_report,
    build_retired_signal_decision,
    build_signal_summary,
    build_source_column_audit,
    repaired_registry_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lag_safe_signal_repair_v3_62.json"
TASK_ID = "20260529_v3_62_lag_safe_signal_repair"
VERSION = "V3.62"
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> LagSafeRepairConfig:
    return LagSafeRepairConfig(
        state_panel_path=resolve_path(raw["state_panel_path"]),
        v3_48_manifest_path=resolve_path(raw["v3_48_manifest_path"]),
        v3_61_decision_path=resolve_path(raw["v3_61_decision_path"]),
        v3_61_manifest_path=resolve_path(raw["v3_61_manifest_path"]),
        market_proxy_source_path=resolve_path(raw["market_proxy_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        min_signal_rows=int(raw["minimums"]["signal_rows"]),
        min_repaired_candidates=int(raw["minimums"]["repaired_candidates"]),
        max_abs_same_day_market_move_corr_warn=float(raw["artifact_screen"]["max_abs_same_day_market_move_corr_warn"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "lag_safe_signal_repair.py",
        ROOT / "strategy_lab" / "hirssm_v3_62_lag_safe_signal_repair.py",
        ROOT / "configs" / "lag_safe_signal_repair_v3_62.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_62_lag_safe_signal_repair.json",
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

    state_panel = pd.read_csv(config.state_panel_path, encoding="utf-8-sig", low_memory=False)
    v3_48_manifest = read_json(config.v3_48_manifest_path)
    v3_61_decisions = pd.read_csv(config.v3_61_decision_path, encoding="utf-8-sig", low_memory=False)
    v3_61_manifest = read_json(config.v3_61_manifest_path)
    market_proxy_source = pd.read_csv(config.market_proxy_source_path, encoding="utf-8-sig", low_memory=False)

    registry = repaired_registry_frame()
    signal_panel = build_repaired_signal_panel(state_panel)
    summary = build_signal_summary(signal_panel)
    source_audit = build_source_column_audit()
    retired = build_retired_signal_decision(v3_61_decisions)
    market_screen = build_market_move_artifact_screen(signal_panel, market_proxy_source, config)
    lag_checks = build_lag_discipline_checks(signal_panel, registry, source_audit, config, v3_48_manifest, v3_61_manifest)
    guard = build_no_promotion_guard(signal_panel)
    output_frames = [registry, signal_panel, summary, source_audit, retired, market_screen, lag_checks, guard]
    acceptance = build_acceptance_checks(lag_checks, signal_panel, registry, guard, output_columns(output_frames))
    report = build_report(registry, signal_panel, summary, source_audit, retired, market_screen, lag_checks, acceptance)
    catalog = build_catalog(signal_panel, registry, source_audit)

    output_paths = {
        "registry": output_dir / "repaired_signal_registry.csv",
        "signal_panel": output_dir / "repaired_signal_panel.csv",
        "summary": output_dir / "repaired_signal_summary.csv",
        "source_audit": output_dir / "source_column_audit.csv",
        "retired": output_dir / "retired_signal_decision.csv",
        "market_screen": output_dir / "same_day_market_move_artifact_screen.csv",
        "lag_checks": output_dir / "lag_discipline_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "lag_safe_signal_repair_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(registry, output_paths["registry"])
    write_csv(signal_panel, output_paths["signal_panel"])
    write_csv(summary, output_paths["summary"])
    write_csv(source_audit, output_paths["source_audit"])
    write_csv(retired, output_paths["retired"])
    write_csv(market_screen, output_paths["market_screen"])
    write_csv(lag_checks, output_paths["lag_checks"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
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
                "check": "source_dates_strictly_lagged",
                "status": "pass"
                if not signal_panel.empty and (pd.to_datetime(signal_panel["source_trade_date"]) < pd.to_datetime(signal_panel["signal_date"])).all()
                else "fail",
                "detail": f"rows={len(signal_panel)}",
            },
            {
                "check": "source_columns_are_non_price",
                "status": "pass" if source_audit["allowed_for_repair"].astype(bool).all() else "fail",
                "detail": f"source_rows={len(source_audit)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass" if not signal_panel["official_total_return_evidence"].astype(bool).any() else "fail",
                "detail": "signal definitions only",
            },
            {
                "check": "portfolio_or_model_promotion_blocked",
                "status": "pass"
                if not signal_panel["portfolio_backtest_allowed"].astype(bool).any()
                and not signal_panel["model_promotion_allowed"].astype(bool).any()
                else "fail",
                "detail": "no NAV or model promotion",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["registry"],
        output_paths["signal_panel"],
        output_paths["summary"],
        output_paths["source_audit"],
        output_paths["retired"],
        output_paths["market_screen"],
        output_paths["lag_checks"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    warning_rows = int(market_screen["artifact_warning"].astype(bool).sum()) if not market_screen.empty else 0
    retired_rows = int(retired["repair_decision"].astype(str).str.startswith("retired").sum()) if not retired.empty else 0
    mapped_rows = int(retired["repair_decision"].astype(str).str.startswith("mapped").sum()) if not retired.empty else 0
    metrics = {
        "repaired_candidate_count": int(len(registry)),
        "repaired_signal_rows": int(len(signal_panel)),
        "repaired_signal_dates": int(signal_panel["signal_date"].nunique()) if not signal_panel.empty else 0,
        "non_price_source_audit_rows": int(len(source_audit)),
        "source_columns_allowed_rows": int(source_audit["allowed_for_repair"].astype(bool).sum()) if not source_audit.empty else 0,
        "retired_v3_61_rows": retired_rows,
        "mapped_v3_61_rows": mapped_rows,
        "same_day_market_move_warning_rows": warning_rows,
        "return_basis": "not_applicable_signal_definition_only",
        "validation_scope": "lag_safe_signal_repair_no_label_validation",
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }

    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.61 proxy artifact audit",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_62_lag_safe_signal_repair.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.state_panel_path),
            rel(config.v3_48_manifest_path),
            rel(config.v3_61_decision_path),
            rel(config.v3_61_manifest_path),
            rel(config.market_proxy_source_path),
        ],
        "code_refs": [
            "strategy_lab/lag_safe_signal_repair.py",
            "strategy_lab/hirssm_v3_62_lag_safe_signal_repair.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]],
        "changed_files": build_changed_files(outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "fail_count": int((self_check["status"] != "pass").sum()),
        "warn_count": warning_rows + retired_rows,
        "limitations": [
            "V3.62 creates signal definitions only and does not test effectiveness.",
            "Non-price inputs can still proxy market regimes and require later artifact screening.",
            "No portfolio backtest, NAV, model promotion, or default enablement is produced.",
        ],
        "risk_flags": [
            "signal_repair_not_validation",
            "nonprice_inputs_can_proxy_regime",
            "price_derived_candidates_retired",
            "model_promotion_blocked",
        ],
        "next_decision": "V3.63 should validate the repaired lag-safe signal panel against price-proxy labels and rerun V3.61-style artifact screening before walk-forward escalation.",
        "handoff_summary": "Built a lag-safe non-price repaired signal panel with prior-day source dates, retired price-derived proxy-positive candidates, and kept all outputs observation-only.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
