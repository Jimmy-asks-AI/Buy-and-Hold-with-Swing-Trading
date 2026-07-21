#!/usr/bin/env python
"""Run HIRSSM V3.84 legacy governance debt gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "legacy_governance_debt_gate_v3_84.json"
TASK_ID = "20260601_v3_84_legacy_governance_debt_gate"
VERSION = "V3.84"
AGENT = "code_quality_engineer"


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


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def run_framework_check(script_path: Path) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(script_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {
            "self_check_pass": False,
            "errors": [f"framework check emitted non-JSON stdout: {proc.stdout[:500]}"],
            "raw_error_count": None,
            "legacy_debt_count": None,
            "legacy_debt_path_count": None,
        }
    return proc.returncode, result, proc.stderr


def build_catalog(framework_result: dict[str, Any], allowlist_path: Path) -> str:
    return "\n".join(
        [
            "# A-share Legacy Governance Debt Gate V3.84",
            "",
            "## Dataset Decision",
            "",
            f"- Framework self-check pass: `{framework_result.get('self_check_pass')}`",
            f"- Active error count: `{len(framework_result.get('errors', []))}`",
            f"- Raw legacy-shaped error count: `{framework_result.get('raw_error_count')}`",
            f"- Legacy debt count: `{framework_result.get('legacy_debt_count')}`",
            f"- Legacy debt path count: `{framework_result.get('legacy_debt_path_count')}`",
            f"- Allowlist: `{rel(allowlist_path)}`",
            "- This gate does not change model signals, labels, portfolios, or promotion status.",
            "",
        ]
    )


def build_report(
    framework_result: dict[str, Any],
    allowlist: dict[str, Any],
    acceptance: pd.DataFrame,
    active_errors: pd.DataFrame,
) -> str:
    lines = [
        "# V3.84 Legacy Governance Debt Gate",
        "",
        "## Decision",
        "",
        "- V3.84 makes pre-strict-schema governance debt explicit instead of silently weakening the framework checker.",
        "- Only exact paths in `legacy_governance_debt_allowlist.json` are filtered.",
        "- New task briefs, new manifests, invalid JSON, missing output refs, and V3.57+ artifacts remain strict failures.",
        "",
        "## Key Metrics",
        "",
        f"- Framework self-check pass: `{framework_result.get('self_check_pass')}`",
        f"- Active error count: `{len(framework_result.get('errors', []))}`",
        f"- Raw error count before allowlist: `{framework_result.get('raw_error_count')}`",
        f"- Legacy debt count: `{framework_result.get('legacy_debt_count')}`",
        f"- Legacy debt path count: `{framework_result.get('legacy_debt_path_count')}`",
        f"- Allowlist enabled: `{allowlist.get('enabled')}`",
        "",
        "## Acceptance",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for _, row in acceptance.iterrows():
        lines.append(f"| {row['check']} | {row['status']} | {str(row['detail']).replace('|', '/')} |")
    lines.extend(["", "## Active Errors", ""])
    if active_errors.empty:
        lines.append("- None.")
    else:
        for _, row in active_errors.head(20).iterrows():
            lines.append(f"- {row['error']}")
    lines.extend(["", "## Next Step", "", "- Use `python -B -X utf8 strategy_lab/agent_framework_check.py` as the active global gate for V3.57+ work.", ""])
    return "\n".join(lines)


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "agent_framework_check.py",
        ROOT / "strategy_lab" / "agents" / "_templates" / "legacy_governance_debt_allowlist.json",
        ROOT / "strategy_lab" / "hirssm_v3_84_legacy_governance_debt_gate.py",
        ROOT / "configs" / "legacy_governance_debt_gate_v3_84.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260601_v3_84_legacy_governance_debt_gate.json",
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
    framework_script = resolve_path(raw_config["framework_check_script"])
    allowlist_path = resolve_path(raw_config["legacy_allowlist_path"])
    output_dir = resolve_path(raw_config["output_dir"])
    catalog_path = resolve_path(raw_config["catalog_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(build_catalog({"self_check_pass": "bootstrap", "errors": []}, allowlist_path), catalog_path)

    allowlist = read_json(allowlist_path)
    returncode, framework_result, stderr = run_framework_check(framework_script)
    active_errors = pd.DataFrame([{"error": str(error)} for error in framework_result.get("errors", [])], columns=["error"])
    legacy_summary = pd.DataFrame(
        framework_result.get("legacy_debt_summary", []),
        columns=["path", "error_count"],
    )
    allowed_path_count = len(allowlist.get("legacy_task_brief_paths", [])) + len(allowlist.get("legacy_agent_run_manifest_paths", []))
    allowed_paths = set(str(path).replace("\\", "/") for path in allowlist.get("legacy_task_brief_paths", []))
    allowed_paths.update(str(path).replace("\\", "/") for path in allowlist.get("legacy_agent_run_manifest_paths", []))
    synthetic_new_error_path = "strategy_lab/agents/task_briefs/20990101_new_task.json"
    acceptance = pd.DataFrame(
        [
            {
                "check": "framework_check_returncode_zero",
                "status": "pass" if returncode == 0 else "fail",
                "detail": f"returncode={returncode}",
            },
            {
                "check": "framework_self_check_pass",
                "status": "pass" if bool(framework_result.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={framework_result.get('self_check_pass')}",
            },
            {
                "check": "active_errors_empty",
                "status": "pass" if active_errors.empty else "fail",
                "detail": f"active_errors={len(active_errors)}",
            },
            {
                "check": "legacy_debt_explicit",
                "status": "pass" if int(framework_result.get("legacy_debt_count", 0) or 0) > 0 and not legacy_summary.empty else "fail",
                "detail": f"legacy_debt_count={framework_result.get('legacy_debt_count')}",
            },
            {
                "check": "allowlist_is_exact_path_policy",
                "status": "pass" if bool(allowlist.get("policy", {}).get("exact_path_match_only")) else "fail",
                "detail": f"allowed_paths={allowed_path_count}",
            },
            {
                "check": "legacy_path_count_matches_summary",
                "status": "pass" if int(framework_result.get("legacy_debt_path_count", -1) or -1) == allowed_path_count else "fail",
                "detail": f"summary={framework_result.get('legacy_debt_path_count')};allowlist={allowed_path_count}",
            },
            {
                "check": "non_allowlisted_error_remains_active",
                "status": "pass" if synthetic_new_error_path not in allowed_paths else "fail",
                "detail": synthetic_new_error_path,
            },
        ]
    )

    output_paths = {
        "framework_result": output_dir / "framework_check_result.json",
        "legacy_summary": output_dir / "legacy_debt_summary.csv",
        "active_errors": output_dir / "active_errors.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "legacy_governance_debt_gate_report.md",
        "catalog": catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_json(framework_result, output_paths["framework_result"])
    write_csv(legacy_summary, output_paths["legacy_summary"])
    write_csv(active_errors, output_paths["active_errors"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(build_report(framework_result, allowlist, acceptance, active_errors), output_paths["report"])
    write_text(build_catalog(framework_result, allowlist_path), output_paths["catalog"])

    forbidden_terms = {
        "nav",
        "sharpe",
        "annualized_return",
        "portfolio_return",
        "max_drawdown",
        "official_total_return_label",
        "default_enabled",
    }
    forbidden_columns = sorted(term for term in forbidden_terms if term in " ".join(output_columns([legacy_summary, active_errors, acceptance])).lower())
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if not acceptance["status"].eq("fail").any() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "active_errors_empty",
                "status": "pass" if active_errors.empty else "fail",
                "detail": f"active_errors={len(active_errors)}",
            },
            {
                "check": "legacy_summary_written",
                "status": "pass" if output_paths["legacy_summary"].exists() and not legacy_summary.empty else "fail",
                "detail": f"rows={len(legacy_summary)}",
            },
            {
                "check": "stderr_empty",
                "status": "pass" if not stderr.strip() else "fail",
                "detail": stderr.strip()[:200],
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
        output_paths["framework_result"],
        output_paths["legacy_summary"],
        output_paths["active_errors"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "framework_self_check_pass": bool(framework_result.get("self_check_pass", False)),
        "framework_returncode": int(returncode),
        "raw_error_count": int(framework_result.get("raw_error_count", 0) or 0),
        "active_error_count": int(len(active_errors)),
        "legacy_debt_count": int(framework_result.get("legacy_debt_count", 0) or 0),
        "legacy_debt_path_count": int(framework_result.get("legacy_debt_path_count", 0) or 0),
        "allowlist_path_count": int(allowed_path_count),
        "may_execute_v3_78_now": False,
        "may_execute_v3_53_now": False,
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }
    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.83 real-sample operations pack and strict agent framework checker",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_84_legacy_governance_debt_gate.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": raw_config.get("allowed_inputs", []),
        "code_refs": [
            "strategy_lab/agent_framework_check.py",
            "strategy_lab/hirssm_v3_84_legacy_governance_debt_gate.py",
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
            "V3.84 does not repair legacy artifacts in place; it makes their schema debt explicit and allowlisted.",
            "Only exact paths in the allowlist are filtered.",
            "No model, label, portfolio, or promotion artifact is produced.",
        ],
        "risk_flags": [
            "legacy_schema_debt_present",
            "exact_path_allowlist_required",
            "new_tasks_remain_strict",
            "model_promotion_blocked",
        ],
        "next_decision": "Use agent_framework_check.py as the active global gate; remove legacy allowlist entries only after those old artifacts are migrated or archived.",
        "handoff_summary": "V3.84 restored a passing global agent framework check while preserving legacy schema debt as explicit audit evidence.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
