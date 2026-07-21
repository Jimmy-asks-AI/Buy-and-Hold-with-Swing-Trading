#!/usr/bin/env python
"""Run HIRSSM V3.76 MARKET label-source intake orchestrator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_source_intake_orchestrator import (
    LabelSourceIntakeConfig,
    build_acceptance_checks,
    build_catalog,
    build_intake_status,
    build_next_commands,
    build_no_execution_guard,
    build_report,
    build_watchlist,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_label_source_intake_orchestrator_v3_76.json"
TASK_ID = "20260529_v3_76_market_label_source_intake_orchestrator"
VERSION = "V3.76"
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


def load_config(raw: dict[str, Any]) -> LabelSourceIntakeConfig:
    return LabelSourceIntakeConfig(
        v3_75_manifest_path=resolve_path(raw["v3_75_manifest_path"]),
        v3_75_import_decision_path=resolve_path(raw["v3_75_import_decision_path"]),
        v3_75_procurement_requirements_path=resolve_path(raw["v3_75_procurement_requirements_path"]),
        v3_75_vendor_template_path=resolve_path(raw["v3_75_vendor_template_path"]),
        v3_53_config_path=resolve_path(raw["v3_53_config_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        execute_downstream=bool(raw["execute_downstream"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_label_source_intake_orchestrator.py",
        ROOT / "strategy_lab" / "hirssm_v3_76_market_label_source_intake_orchestrator.py",
        ROOT / "configs" / "market_label_source_intake_orchestrator_v3_76.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_76_market_label_source_intake_orchestrator.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    v3_75_manifest = read_json(config.v3_75_manifest_path)
    import_decision = read_csv(config.v3_75_import_decision_path)
    requirements = read_csv(config.v3_75_procurement_requirements_path)

    status = build_intake_status(v3_75_manifest, import_decision, config)
    next_commands = build_next_commands(import_decision, config)
    watchlist = build_watchlist(requirements, config)
    guard = build_no_execution_guard(config, next_commands)
    acceptance = build_acceptance_checks(status, next_commands, watchlist, guard)
    report = build_report(status, next_commands, watchlist, acceptance, config)
    catalog = build_catalog(status, next_commands, config)

    output_paths = {
        "status": output_dir / "intake_status.csv",
        "next_commands": output_dir / "next_commands.csv",
        "watchlist": output_dir / "source_watchlist.csv",
        "guard": output_dir / "no_execution_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_label_source_intake_orchestrator_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(status, output_paths["status"])
    write_csv(next_commands, output_paths["next_commands"])
    write_csv(watchlist, output_paths["watchlist"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    forbidden_terms = {"nav", "sharpe", "annualized_return", "portfolio_return", "max_drawdown", "official_total_return_label", "default_enabled"}
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([status, next_commands, watchlist, guard, acceptance])).lower())
    may_execute_v3_53 = bool(next_commands.loc[next_commands["condition"].eq("v3_75_ready_for_v3_53_true"), "may_execute_now"].any()) if not next_commands.empty else False
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "v3_75_manifest_passed",
                "status": "pass" if bool(v3_75_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_75_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_53_not_executed",
                "status": "pass" if not config.execute_downstream else "fail",
                "detail": f"execute_downstream={config.execute_downstream};may_execute_v3_53={may_execute_v3_53}",
            },
            {
                "check": "target_missing_keeps_blocked",
                "status": "pass" if (config.target_source_path.exists() or status["status"].astype(str).eq("blocked").any()) else "fail",
                "detail": f"target_exists={config.target_source_path.exists()}",
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
        output_paths["status"],
        output_paths["next_commands"],
        output_paths["watchlist"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "target_source_exists": config.target_source_path.exists(),
        "v3_75_ready_for_v3_53": bool(import_decision.get("ready_for_v3_53", pd.Series([False])).astype(str).str.lower().isin(["true", "1", "yes"]).any()),
        "may_execute_v3_53_now": may_execute_v3_53,
        "active_next_command_rows": int(next_commands["status"].astype(str).eq("active").sum()),
        "blocked_status_rows": int(status["status"].astype(str).eq("blocked").sum()),
        "execute_downstream": config.execute_downstream,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.75 label-source procurement package",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_76_market_label_source_intake_orchestrator.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/market_label_source_intake_orchestrator.py",
            "strategy_lab/hirssm_v3_76_market_label_source_intake_orchestrator.py",
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
        "warn_count": int(status["status"].astype(str).eq("blocked").sum() + next_commands["status"].astype(str).eq("blocked").sum()),
        "limitations": [
            "V3.76 is an orchestrator and does not execute V3.53 by default.",
            "If target CSV is missing, the only active route is manual/vendor delivery.",
            "Portfolio validation remains blocked until V3.53 labels and a later label-validation review pass.",
        ],
        "risk_flags": [
            "target_source_missing" if not config.target_source_path.exists() else "target_source_present_needs_v3_75_recheck",
            "label_generation_not_run",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Place the compliant target CSV, rerun V3.75, then rerun V3.76 to unlock V3.53 command routing.",
        "handoff_summary": "V3.76 created the post-procurement intake status, source watchlist, and next-command router for the MARKET label-source pipeline.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
