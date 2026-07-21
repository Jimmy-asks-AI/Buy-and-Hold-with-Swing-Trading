#!/usr/bin/env python
"""Run HIRSSM V3.57 limited proxy state diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from limited_proxy_state_diagnostic import (
    LimitedStateDiagnosticConfig,
    build_acceptance_checks,
    build_catalog,
    build_joined_contract_coverage,
    build_joined_contract_panel,
    build_label_contract_diagnostics,
    build_no_promotion_guard,
    build_proxy_state_panel,
    build_readiness_checks,
    build_report,
    build_state_coverage,
    build_state_transition_summary,
    validate_labels,
    validate_signal_panel,
    validate_source,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "limited_proxy_state_diagnostic_v3_57.json"
TASK_ID = "20260529_v3_57_limited_proxy_state_diagnostic"
VERSION = "V3.57"
AGENT = "backtest_validation_auditor"


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


def load_config(raw: dict[str, Any]) -> LimitedStateDiagnosticConfig:
    return LimitedStateDiagnosticConfig(
        source_path=resolve_path(raw["source_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        label_path=resolve_path(raw["label_path"]),
        v3_56_manifest_path=resolve_path(raw["v3_56_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        trend_window=int(raw["state_windows"]["trend"]),
        volatility_window=int(raw["state_windows"]["volatility"]),
        drawdown_window=int(raw["state_windows"]["drawdown"]),
        trend_up_threshold=float(raw["thresholds"]["trend_up"]),
        trend_down_threshold=float(raw["thresholds"]["trend_down"]),
        low_volatility_threshold=float(raw["thresholds"]["low_volatility"]),
        high_volatility_threshold=float(raw["thresholds"]["high_volatility"]),
        moderate_drawdown_threshold=float(raw["thresholds"]["moderate_drawdown"]),
        deep_drawdown_threshold=float(raw["thresholds"]["deep_drawdown"]),
        label_abs_outlier_threshold=float(raw["thresholds"]["label_abs_outlier"]),
        min_source_rows=int(raw["minimums"]["source_rows"]),
        min_signal_rows=int(raw["minimums"]["signal_rows"]),
        min_label_rows=int(raw["minimums"]["label_rows"]),
        min_joined_rows=int(raw["minimums"]["joined_rows"]),
        min_state_date_coverage_ratio=float(raw["minimums"]["state_date_coverage_ratio"]),
        min_history_available_signal_date_ratio=float(raw["minimums"]["history_available_signal_date_ratio"]),
        min_state_bucket_signal_dates=int(raw["minimums"]["state_bucket_signal_dates"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "limited_proxy_state_diagnostic.py",
        ROOT / "strategy_lab" / "hirssm_v3_57_limited_proxy_state_diagnostic.py",
        ROOT / "configs" / "limited_proxy_state_diagnostic_v3_57.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_57_limited_proxy_state_diagnostic.json",
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

    source = pd.read_csv(config.source_path, encoding="utf-8-sig", low_memory=False)
    signals = pd.read_csv(config.signal_panel_path, encoding="utf-8-sig", low_memory=False)
    labels = pd.read_csv(config.label_path, encoding="utf-8-sig", low_memory=False)
    v3_56_manifest = read_json(config.v3_56_manifest_path)

    source_checks = validate_source(source, config)
    signal_checks = validate_signal_panel(signals, config)
    label_checks = validate_labels(labels, config)
    state_panel = build_proxy_state_panel(source, config)
    state_coverage = build_state_coverage(state_panel, signals, config)
    state_transitions = build_state_transition_summary(state_panel)
    label_diagnostics = build_label_contract_diagnostics(labels, config)
    joined = build_joined_contract_panel(signals, labels, state_panel)
    joined_coverage = build_joined_contract_coverage(joined, config)
    readiness = build_readiness_checks(source_checks, signal_checks, label_checks, state_panel, signals, joined, v3_56_manifest, config)
    guard = build_no_promotion_guard()
    official_source = ROOT / "data_raw" / "market_labels" / "market_total_return_index.csv"
    acceptance = build_acceptance_checks(
        readiness,
        guard,
        state_panel,
        joined,
        official_source.exists(),
        output_columns([state_panel, state_coverage, label_diagnostics, joined_coverage, readiness, acceptance if False else pd.DataFrame()]),
    )
    report = build_report(source, signals, labels, state_panel, state_coverage, label_diagnostics, joined, joined_coverage, readiness, acceptance)
    catalog = build_catalog(state_panel, joined)

    output_paths = {
        "source_checks": output_dir / "source_contract_checks.csv",
        "signal_checks": output_dir / "signal_contract_checks.csv",
        "label_checks": output_dir / "label_contract_checks.csv",
        "state_panel": output_dir / "limited_proxy_state_panel.csv",
        "state_coverage": output_dir / "state_coverage.csv",
        "state_transitions": output_dir / "state_transition_summary.csv",
        "label_diagnostics": output_dir / "label_contract_diagnostics.csv",
        "joined": output_dir / "limited_joined_contract_panel.csv",
        "joined_coverage": output_dir / "joined_contract_coverage.csv",
        "readiness": output_dir / "state_diagnostic_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "limited_proxy_state_diagnostic_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(source_checks, output_paths["source_checks"])
    write_csv(signal_checks, output_paths["signal_checks"])
    write_csv(label_checks, output_paths["label_checks"])
    write_csv(state_panel, output_paths["state_panel"])
    write_csv(state_coverage, output_paths["state_coverage"])
    write_csv(state_transitions, output_paths["state_transitions"])
    write_csv(label_diagnostics, output_paths["label_diagnostics"])
    write_csv(joined, output_paths["joined"])
    write_csv(joined_coverage, output_paths["joined_coverage"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    fail_count = int((acceptance["status"] != "pass").sum())
    warn_count = 0
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if fail_count == 0 else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "performance_validation_blocked",
                "status": "pass"
                if str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
                else "fail",
                "detail": "limited smoke diagnostic only",
            },
            {
                "check": "official_source_not_created",
                "status": "pass" if not official_source.exists() else "fail",
                "detail": rel(official_source),
            },
            {
                "check": "joined_contract_panel_nonempty",
                "status": "pass" if len(joined) > 0 else "fail",
                "detail": f"joined_rows={len(joined)}",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["source_checks"],
        output_paths["signal_checks"],
        output_paths["label_checks"],
        output_paths["state_panel"],
        output_paths["state_coverage"],
        output_paths["state_transitions"],
        output_paths["label_diagnostics"],
        output_paths["joined"],
        output_paths["joined_coverage"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    metrics = {
        "source_rows": int(len(source)),
        "signal_rows": int(len(signals)),
        "label_rows": int(len(labels)),
        "state_rows": int(len(state_panel)),
        "joined_rows": int(len(joined)),
        "state_date_coverage_ratio": float(
            readiness.loc[readiness["check"] == "state_date_coverage_passed", "detail"].iloc[0].split(";")[0].split("=")[1]
        ),
        "history_available_signal_date_ratio": float(
            readiness.loc[readiness["check"] == "history_available_signal_date_ratio_passed", "detail"].iloc[0].split(";")[0].split("=")[1]
        ),
        "performance_validation_status": "blocked",
    }
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.56 limited proxy label-chain test",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -X utf8 strategy_lab/hirssm_v3_57_limited_proxy_state_diagnostic.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [rel(config.source_path), rel(config.signal_panel_path), rel(config.label_path), rel(config.v3_56_manifest_path)],
        "code_refs": [
            "strategy_lab/limited_proxy_state_diagnostic.py",
            "strategy_lab/hirssm_v3_57_limited_proxy_state_diagnostic.py",
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
        "warn_count": warn_count,
        "limitations": [
            "JoinQuant trial window is short and cannot support full-history validation.",
            "The proxy is an adjusted-price proxy, not an official total-return index.",
            "V3.57 emits no IC, hit-rate, backtest, NAV, Sharpe, or model promotion evidence.",
        ],
        "risk_flags": [
            "limited_window",
            "adjusted_proxy_not_official_total_return",
            "performance_validation_blocked",
        ],
        "next_decision": "Use V3.57 only as smoke evidence; acquire full-history governed label source before real validation.",
        "handoff_summary": "Limited proxy states, label sanity checks, and signal-label-state joins are mechanically ready under short-window restrictions.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
