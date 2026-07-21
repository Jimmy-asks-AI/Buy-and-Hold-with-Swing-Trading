#!/usr/bin/env python
"""Run HIRSSM V3.74 live label-source acquisition attempt."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from live_label_source_acquisition_attempt import (
    LiveLabelSourceAcquisitionConfig,
    build_acceptance_checks,
    build_catalog,
    build_no_promotion_guard,
    build_probe_plan,
    build_report,
    build_signal_coverage,
    build_source_quality_assessment,
    build_write_decision,
    credential_pair,
    execute_joinquant_attempt,
    manual_data_requirements,
    maybe_write_target,
    provider_readiness,
    redact,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "live_label_source_acquisition_attempt_v3_74.json"
TASK_ID = "20260529_v3_74_live_label_source_acquisition_attempt"
VERSION = "V3.74"
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


def load_config(raw: dict[str, Any]) -> LiveLabelSourceAcquisitionConfig:
    return LiveLabelSourceAcquisitionConfig(
        v3_73_manifest_path=resolve_path(raw["v3_73_manifest_path"]),
        v3_73_queue_path=resolve_path(raw["v3_73_queue_path"]),
        v3_73_source_contract_path=resolve_path(raw["v3_73_source_contract_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        target_security=str(raw["target_security"]),
        target_asset_or_index=str(raw["target_asset_or_index"]),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        fields=tuple(str(x) for x in raw["fields"]),
        fq=str(raw["fq"]),
        execute_probe=bool(raw["execute_probe"]),
        allow_target_write=bool(raw["allow_target_write"]),
        approved_source_basis=str(raw["approved_source_basis"]),
        min_source_rows=int(raw["min_source_rows"]),
        min_signal_coverage_ratio=float(raw["min_signal_coverage_ratio"]),
        horizons=tuple(int(x) for x in raw["horizons"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "live_label_source_acquisition_attempt.py",
        ROOT / "strategy_lab" / "hirssm_v3_74_live_label_source_acquisition_attempt.py",
        ROOT / "configs" / "live_label_source_acquisition_attempt_v3_74.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_74_live_label_source_acquisition_attempt.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    for frame in frames:
        columns.extend(str(col) for col in frame.columns)
    return columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    v3_73_manifest = read_json(config.v3_73_manifest_path)
    v3_73_queue = read_csv(config.v3_73_queue_path)
    contract = read_csv(config.v3_73_source_contract_path)
    signal_panel = read_csv(config.signal_panel_path)

    readiness = provider_readiness(ROOT, config)
    probe_plan = build_probe_plan(config, readiness)
    try:
        attempt, candidate = execute_joinquant_attempt(ROOT, config, readiness)
    except Exception as exc:  # noqa: BLE001
        detail = redact(f"{type(exc).__name__}: {exc}", ROOT)
        permission_window = re.search(r"(\d{4}-\d{2}-\d{2})\D+(\d{4}-\d{2}-\d{2})", detail)
        if permission_window:
            fallback_config = replace(config, start_date=permission_window.group(1), end_date=permission_window.group(2))
            try:
                attempt, candidate = execute_joinquant_attempt(ROOT, fallback_config, readiness)
                attempt["attempt_id"] = "joinquant_permission_window_market_proxy"
                attempt["status"] = attempt["status"].astype(str).replace({"ok": "ok_limited_permission_window"})
                attempt["detail"] = attempt["detail"].astype(str) + ";fallback_from_full_history_error=" + detail
            except Exception as fallback_exc:  # noqa: BLE001
                attempt = pd.DataFrame(
                    [
                        {
                            "attempt_id": "joinquant_permission_window_market_proxy",
                            "status": "error",
                            "rows": 0,
                            "first_date": "",
                            "last_date": "",
                            "detail": detail + ";fallback_error=" + redact(f"{type(fallback_exc).__name__}: {fallback_exc}", ROOT),
                        }
                    ]
                )
                candidate = pd.DataFrame()
        else:
            attempt = pd.DataFrame(
                [
                    {
                        "attempt_id": "joinquant_full_history_market_proxy",
                        "status": "error",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "detail": detail,
                    }
                ]
            )
            candidate = pd.DataFrame()
    quality = build_source_quality_assessment(candidate, config)
    coverage = build_signal_coverage(candidate, signal_panel, config)
    write_decision = build_write_decision(candidate, quality, coverage, config)
    write_result = maybe_write_target(candidate, write_decision, config)
    manual_requirements = manual_data_requirements(config, coverage)
    guard = build_no_promotion_guard(write_result)
    acceptance = build_acceptance_checks(readiness, attempt, quality, coverage, write_decision, write_result, guard)
    report = build_report(readiness, attempt, quality, coverage, write_decision, manual_requirements, acceptance, config)
    catalog = build_catalog(attempt, quality, coverage, write_decision)

    output_paths = {
        "readiness": output_dir / "provider_readiness.csv",
        "probe_plan": output_dir / "probe_plan.csv",
        "attempt": output_dir / "live_acquisition_attempt.csv",
        "candidate": output_dir / "joinquant_adjusted_proxy_candidate.csv",
        "candidate_preview": output_dir / "joinquant_adjusted_proxy_candidate_preview.csv",
        "quality": output_dir / "source_quality_assessment.csv",
        "coverage": output_dir / "signal_coverage_assessment.csv",
        "write_decision": output_dir / "target_write_decision.csv",
        "write_result": output_dir / "target_write_result.csv",
        "manual": output_dir / "manual_required_data_interfaces.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "live_label_source_acquisition_attempt_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(readiness, output_paths["readiness"])
    write_csv(probe_plan, output_paths["probe_plan"])
    write_csv(attempt, output_paths["attempt"])
    write_csv(candidate, output_paths["candidate"])
    write_csv(candidate.head(20), output_paths["candidate_preview"])
    write_csv(quality, output_paths["quality"])
    write_csv(coverage, output_paths["coverage"])
    write_csv(write_decision, output_paths["write_decision"])
    write_csv(write_result, output_paths["write_result"])
    write_csv(manual_requirements, output_paths["manual"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    forbidden_terms = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([readiness, probe_plan, attempt, quality, coverage, write_decision, write_result, guard, acceptance])).lower())
    target_written = bool(write_result["status"].eq("written").any()) if not write_result.empty else False
    v3_73_blocked_rows = int(v3_73_queue["higher_quality_label_review_status"].astype(str).str.startswith("blocked").sum()) if "higher_quality_label_review_status" in v3_73_queue else 0
    secret_text = "\n".join(
        [
            readiness.to_csv(index=False),
            attempt.to_csv(index=False),
            write_decision.to_csv(index=False),
            report,
            catalog,
        ]
    )
    username, password = credential_pair(ROOT)
    leaked_secret = any(secret and secret in secret_text for secret in [username, password])
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_have_no_fail",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "v3_73_queue_loaded",
                "status": "pass" if len(v3_73_queue) > 0 and bool(v3_73_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"queue_rows={len(v3_73_queue)};blocked_rows={v3_73_blocked_rows}",
            },
            {
                "check": "contract_loaded",
                "status": "pass" if {"column", "v3_73_enforcement"}.issubset(contract.columns) else "fail",
                "detail": f"contract_rows={len(contract)}",
            },
            {
                "check": "target_not_written_without_write_allowed",
                "status": "pass" if (not target_written or bool(write_decision['write_allowed'].iloc[0])) else "fail",
                "detail": f"target_written={target_written}",
            },
            {
                "check": "forbidden_performance_columns_absent",
                "status": "pass" if not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
            {
                "check": "credentials_not_written",
                "status": "pass" if not leaked_secret else "fail",
                "detail": "outputs do not include configured secrets",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["readiness"],
        output_paths["probe_plan"],
        output_paths["attempt"],
        output_paths["candidate"],
        output_paths["candidate_preview"],
        output_paths["quality"],
        output_paths["coverage"],
        output_paths["write_decision"],
        output_paths["write_result"],
        output_paths["manual"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    candidate_rows = int(len(candidate))
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    quality_pass = bool(quality["status"].eq("pass").all()) if not quality.empty else False
    coverage_pass = bool(coverage["coverage_status"].eq("pass").all()) if not coverage.empty else False
    metrics = {
        "v3_73_strict_survivor_rows": int(len(v3_73_queue)),
        "v3_73_blocked_survivor_rows": v3_73_blocked_rows,
        "joinquant_candidate_rows": candidate_rows,
        "joinquant_candidate_first_date": str(candidate["date"].min()) if candidate_rows else "",
        "joinquant_candidate_last_date": str(candidate["date"].max()) if candidate_rows else "",
        "source_quality_gate_pass": quality_pass,
        "signal_coverage_gate_pass": coverage_pass,
        "minimum_signal_coverage_ratio": min_coverage,
        "target_source_written": target_written,
        "target_source_path": rel(config.target_source_path),
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.73 blocked strict survivor label-source queue",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_74_live_label_source_acquisition_attempt.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/live_label_source_acquisition_attempt.py",
            "strategy_lab/hirssm_v3_74_live_label_source_acquisition_attempt.py",
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
        "warn_count": int(quality["status"].eq("blocked").sum() + coverage["coverage_status"].eq("blocked").sum()),
        "limitations": [
            "JoinQuant get_price candidate is not treated as certified total-return evidence.",
            "The target source file is not written unless quality, coverage, and explicit approval gates pass.",
            "If provider permissions return short history, V3.53 label generation remains blocked.",
        ],
        "risk_flags": [
            "candidate_adjusted_proxy_not_certified_total_return",
            "target_source_write_blocked",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Acquire a long-history official total-return or explicitly approved adjusted MARKET source, then rerun V3.53 and V3.73.",
        "handoff_summary": "V3.74 attempted live JoinQuant MARKET source acquisition and converted the result into source-quality, coverage, and write-gate evidence.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
