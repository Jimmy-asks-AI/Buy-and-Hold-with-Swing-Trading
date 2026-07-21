#!/usr/bin/env python
"""HIRSSM V3.37 credential bootstrap.

This data-steward task prepares safe local credential handling for the V3.36
pilot. It checks readiness without printing secrets and does not run data
acquisition, factor validation, or portfolio backtests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260527_v3_37_credential_bootstrap"
VERSION = "V3.37"
BASELINE = "V3.36 PIT Data Pilot Readiness"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_37" / "credential_bootstrap"
CATALOG_PATH = ROOT / "data_catalog" / "a_share_credential_bootstrap_v3_37.md"
LOCAL_CREDENTIALS = ROOT / "configs" / "data_credentials.json"
LOCAL_TEMPLATE = ROOT / "configs" / "data_credentials.local.template.json"
EXECUTE_EXAMPLE = ROOT / "configs" / "pit_data_pilot_v3_36_execute.example.json"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def gitignore_lines() -> list[str]:
    path = ROOT / ".gitignore"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def gitignore_covers(pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    lines = [line.replace("\\", "/") for line in gitignore_lines()]
    return normalized in lines


def mask_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    placeholders = ["PASTE_", "TOKEN_HERE", "USERNAME_HERE", "PASSWORD_HERE"]
    return not any(item in text.upper() for item in placeholders)


def local_config_section(provider: str) -> dict[str, Any]:
    config = read_json_if_exists(LOCAL_CREDENTIALS)
    section = config.get(provider, {}) if isinstance(config, dict) else {}
    return section if isinstance(section, dict) else {}


def credential_value(provider: str, field: str) -> str | None:
    if provider == "tushare" and field == "token":
        return (
            os.getenv("TUSHARE_TOKEN")
            or os.getenv("TUSHARE_PRO_TOKEN")
            or local_config_section("tushare").get("token")
        )
    if provider == "joinquant" and field == "username":
        return (
            os.getenv("JQDATA_USERNAME")
            or os.getenv("JOINQUANT_USERNAME")
            or os.getenv("JQ_USER")
            or local_config_section("joinquant").get("username")
        )
    if provider == "joinquant" and field == "password":
        return (
            os.getenv("JQDATA_PASSWORD")
            or os.getenv("JOINQUANT_PASSWORD")
            or os.getenv("JQ_PASSWORD")
            or local_config_section("joinquant").get("password")
        )
    return None


def credential_readiness() -> pd.DataFrame:
    rows = [
        {
            "provider": "tushare",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("tushare") is not None,
            "detail": "python package tushare",
        },
        {
            "provider": "tushare",
            "check": "token_present",
            "ready": mask_bool(credential_value("tushare", "token")),
            "detail": "env var or configs/data_credentials.json token",
        },
        {
            "provider": "joinquant",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("jqdatasdk") is not None,
            "detail": "python package jqdatasdk",
        },
        {
            "provider": "joinquant",
            "check": "username_present",
            "ready": mask_bool(credential_value("joinquant", "username")),
            "detail": "env var or configs/data_credentials.json username",
        },
        {
            "provider": "joinquant",
            "check": "password_present",
            "ready": mask_bool(credential_value("joinquant", "password")),
            "detail": "env var or configs/data_credentials.json password",
        },
    ]
    return pd.DataFrame(rows)


def provider_ready(readiness: pd.DataFrame, provider: str) -> bool:
    rows = readiness.loc[readiness["provider"] == provider]
    if provider == "tushare":
        return bool(rows.loc[rows["check"] == "sdk_installed", "ready"].any() and rows.loc[rows["check"] == "token_present", "ready"].any())
    if provider == "joinquant":
        return bool(
            rows.loc[rows["check"] == "sdk_installed", "ready"].any()
            and rows.loc[rows["check"] == "username_present", "ready"].any()
            and rows.loc[rows["check"] == "password_present", "ready"].any()
        )
    return False


def template_has_placeholders(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "PASTE_TUSHARE_TOKEN_HERE" in text and "PASTE_JOINQUANT_USERNAME_HERE" in text and "PASTE_JOINQUANT_PASSWORD_HERE" in text


def suspicious_secret_in_report_text(text: str) -> bool:
    patterns = [
        r"token['\"]?\s*[:=]\s*['\"](?!PASTE_)[A-Za-z0-9_\-]{12,}",
        r"password['\"]?\s*[:=]\s*['\"](?!<|PASTE_).{6,}",
        r"TUSHARE_TOKEN\s*=\s*['\"]?(?!<|PASTE_)[A-Za-z0-9_\-]{12,}",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def credential_policy_check() -> pd.DataFrame:
    rows = [
        {
            "check": "data_credentials_json_gitignored",
            "status": "pass" if gitignore_covers("configs/data_credentials.json") else "fail",
            "detail": "configs/data_credentials.json",
        },
        {
            "check": "local_json_pattern_gitignored",
            "status": "pass" if gitignore_covers("configs/*.local.json") else "fail",
            "detail": "configs/*.local.json",
        },
        {
            "check": "env_local_gitignored",
            "status": "pass" if gitignore_covers("*.env.local") else "fail",
            "detail": "*.env.local",
        },
        {
            "check": "local_template_exists",
            "status": "pass" if LOCAL_TEMPLATE.exists() else "fail",
            "detail": rel(LOCAL_TEMPLATE),
        },
        {
            "check": "local_template_placeholders_only",
            "status": "pass" if template_has_placeholders(LOCAL_TEMPLATE) else "fail",
            "detail": rel(LOCAL_TEMPLATE),
        },
        {
            "check": "execute_example_exists",
            "status": "pass" if EXECUTE_EXAMPLE.exists() else "fail",
            "detail": rel(EXECUTE_EXAMPLE),
        },
        {
            "check": "real_credential_file_exists",
            "status": "observation" if LOCAL_CREDENTIALS.exists() else "blocked",
            "detail": "exists" if LOCAL_CREDENTIALS.exists() else "missing",
        },
    ]
    return pd.DataFrame(rows)


def local_secret_paths() -> pd.DataFrame:
    rows = [
        {
            "path": "configs/data_credentials.json",
            "purpose": "local API credentials",
            "exists": LOCAL_CREDENTIALS.exists(),
            "gitignored": gitignore_covers("configs/data_credentials.json"),
            "safe_to_commit": False,
        },
        {
            "path": "configs/data_credentials.local.template.json",
            "purpose": "placeholder-only template",
            "exists": LOCAL_TEMPLATE.exists(),
            "gitignored": gitignore_covers("configs/*.local.json"),
            "safe_to_commit": True,
        },
        {
            "path": "configs/pit_data_pilot_v3_36_execute.example.json",
            "purpose": "execution config without secrets",
            "exists": EXECUTE_EXAMPLE.exists(),
            "gitignored": False,
            "safe_to_commit": True,
        },
    ]
    return pd.DataFrame(rows)


def pilot_unblock_checklist(readiness: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "step": 1,
            "item": "Tushare SDK installed",
            "status": "pass" if readiness.loc[(readiness["provider"] == "tushare") & (readiness["check"] == "sdk_installed"), "ready"].any() else "blocked",
            "how_to_fix": "python -m pip install tushare",
        },
        {
            "step": 2,
            "item": "Tushare token available",
            "status": "pass" if readiness.loc[(readiness["provider"] == "tushare") & (readiness["check"] == "token_present"), "ready"].any() else "blocked",
            "how_to_fix": "set TUSHARE_TOKEN or fill configs/data_credentials.json locally",
        },
        {
            "step": 3,
            "item": "JoinQuant SDK installed",
            "status": "pass" if readiness.loc[(readiness["provider"] == "joinquant") & (readiness["check"] == "sdk_installed"), "ready"].any() else "blocked",
            "how_to_fix": "python -m pip install jqdatasdk",
        },
        {
            "step": 4,
            "item": "JoinQuant username available",
            "status": "pass" if readiness.loc[(readiness["provider"] == "joinquant") & (readiness["check"] == "username_present"), "ready"].any() else "blocked",
            "how_to_fix": "set JQDATA_USERNAME or fill configs/data_credentials.json locally",
        },
        {
            "step": 5,
            "item": "JoinQuant password available",
            "status": "pass" if readiness.loc[(readiness["provider"] == "joinquant") & (readiness["check"] == "password_present"), "ready"].any() else "blocked",
            "how_to_fix": "set JQDATA_PASSWORD or fill configs/data_credentials.json locally",
        },
        {
            "step": 6,
            "item": "Run V3.36 pilot with explicit execution config",
            "status": "ready" if provider_ready(readiness, "tushare") or provider_ready(readiness, "joinquant") else "blocked",
            "how_to_fix": "python -X utf8 strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py --config configs/pit_data_pilot_v3_36_execute.example.json --execute",
        },
    ]
    return pd.DataFrame(rows)


def blocked_reason(readiness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for provider in ["tushare", "joinquant"]:
        ready = provider_ready(readiness, provider)
        missing = readiness.loc[(readiness["provider"] == provider) & (~readiness["ready"]), "check"].tolist()
        rows.append(
            {
                "provider": provider,
                "blocked": not ready,
                "missing_checks": ",".join(missing),
                "next_action": "ready_for_v3_36_pilot" if ready else "provide local credentials without committing secrets",
            }
        )
    return pd.DataFrame(rows)


def execution_next_steps(readiness: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# V3.37 Credential Bootstrap Next Steps",
            "",
            "## Current Readiness",
            "",
            f"- Tushare ready: `{provider_ready(readiness, 'tushare')}`",
            f"- JoinQuant ready: `{provider_ready(readiness, 'joinquant')}`",
            "",
            "## Local Credential Option",
            "",
            "Use the placeholder template at `configs/data_credentials.local.template.json` as the shape for `configs/data_credentials.json`.",
            "`configs/data_credentials.json` is ignored by git and must remain local.",
            "",
            "## Environment Variable Option",
            "",
            "PowerShell session variables accepted by the pipeline:",
            "",
            "```powershell",
            "$env:TUSHARE_TOKEN = '<your_tushare_token>'",
            "$env:JQDATA_USERNAME = '<your_joinquant_username>'",
            "$env:JQDATA_PASSWORD = '<your_joinquant_password>'",
            "```",
            "",
            "## Pilot Command After Credentials Are Ready",
            "",
            "```powershell",
            "python -X utf8 strategy_lab\\hirssm_v3_36_pit_data_pilot_readiness.py --config configs\\pit_data_pilot_v3_36_execute.example.json --execute",
            "```",
            "",
            "The pilot remains small: all-status stock universe, one index-weight pilot, three stock symbols, and one JoinQuant membership cross-check.",
        ]
    ) + "\n"


def build_agent_report(readiness: pd.DataFrame, policy: pd.DataFrame, checklist: pd.DataFrame) -> str:
    blocked_count = int((checklist["status"] == "blocked").sum())
    policy_fail = int((policy["status"] == "fail").sum())
    return "\n".join(
        [
            "# V3.37 Credential Bootstrap",
            "",
            f"Generated at: `{now_text()}`",
            "",
            "## Decision",
            "",
            "- Task accepted as credential bootstrap and safety guard.",
            "- No API secrets were written or printed.",
            "- No live data acquisition was run.",
            "- No factor, portfolio, or model harness was run.",
            "",
            "## Readiness",
            "",
            f"- Tushare ready: `{provider_ready(readiness, 'tushare')}`",
            f"- JoinQuant ready: `{provider_ready(readiness, 'joinquant')}`",
            f"- Checklist blocked items: `{blocked_count}`",
            f"- Policy failures: `{policy_fail}`",
            "",
            "## Next Gate",
            "",
            "V3.36 pilot remains blocked until credentials are provided locally. After credentials are present, rerun the explicit pilot command in `execution_next_steps.md`.",
        ]
    )


def build_catalog(readiness: pd.DataFrame, policy: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# A-share Credential Bootstrap V3.37",
            "",
            f"Updated: `{now_text()}`",
            "",
            "This catalog entry records credential hygiene for PIT data acquisition.",
            "",
            "## Provider Readiness",
            "",
            f"- Tushare ready: `{provider_ready(readiness, 'tushare')}`",
            f"- JoinQuant ready: `{provider_ready(readiness, 'joinquant')}`",
            "",
            "## Secret Policy",
            "",
            "- Real credentials belong only in environment variables or local `configs/data_credentials.json`.",
            "- `configs/data_credentials.json` is gitignored.",
            "- Output artifacts contain boolean readiness only, never secret values.",
            "",
            "## Policy Checks",
            "",
            "| Check | Status |",
            "|---|---:|",
            *[f"| `{row.check}` | `{row.status}` |" for row in policy.itertuples(index=False)],
        ]
    ) + "\n"


def build_self_check(paths: dict[str, Path], readiness: pd.DataFrame, policy: pd.DataFrame, next_steps: str) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    policy_failures = policy.loc[policy["status"] == "fail", "check"].tolist()
    rows.append(
        {
            "check": "credential_paths_gitignored",
            "status": "pass" if not policy_failures else "fail",
            "detail": ",".join(policy_failures),
        }
    )
    rows.append(
        {
            "check": "no_secret_in_next_steps",
            "status": "pass" if not suspicious_secret_in_report_text(next_steps) else "fail",
            "detail": "placeholder-only commands",
        }
    )
    rows.append(
        {
            "check": "readiness_reports_boolean_only",
            "status": "pass" if set(readiness.columns) == {"provider", "check", "ready", "detail"} else "fail",
            "detail": ",".join(readiness.columns),
        }
    )
    rows.append(
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "credential bootstrap only",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness = credential_readiness()
    policy = credential_policy_check()
    paths_table = local_secret_paths()
    checklist = pilot_unblock_checklist(readiness)
    blocked = blocked_reason(readiness)
    next_steps = execution_next_steps(readiness)

    artifacts = {
        "credential_readiness": OUTPUT_DIR / "credential_readiness.csv",
        "credential_policy_check": OUTPUT_DIR / "credential_policy_check.csv",
        "local_secret_paths": OUTPUT_DIR / "local_secret_paths.csv",
        "pilot_unblock_checklist": OUTPUT_DIR / "pilot_unblock_checklist.csv",
        "execution_next_steps": OUTPUT_DIR / "execution_next_steps.md",
        "blocked_reason": OUTPUT_DIR / "blocked_reason.csv",
        "agent_report": OUTPUT_DIR / "agent_report.md",
        "catalog_update": CATALOG_PATH,
        "changed_files": OUTPUT_DIR / "changed_files.txt",
    }
    write_csv(readiness, artifacts["credential_readiness"])
    write_csv(policy, artifacts["credential_policy_check"])
    write_csv(paths_table, artifacts["local_secret_paths"])
    write_csv(checklist, artifacts["pilot_unblock_checklist"])
    write_text(next_steps, artifacts["execution_next_steps"])
    write_csv(blocked, artifacts["blocked_reason"])
    write_text(build_agent_report(readiness, policy, checklist), artifacts["agent_report"])
    write_text(build_catalog(readiness, policy), artifacts["catalog_update"])
    changed_files = [rel(path) for path in artifacts.values()]
    write_text("\n".join(changed_files) + "\n", artifacts["changed_files"])

    self_check_path = OUTPUT_DIR / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, readiness, policy, next_steps)
    write_csv(self_check, self_check_path)
    fail_count = int((self_check["status"] == "fail").sum())
    blocked_count = int((checklist["status"] == "blocked").sum())
    metrics = {
        "tushare_ready": provider_ready(readiness, "tushare"),
        "joinquant_ready": provider_ready(readiness, "joinquant"),
        "blocked_checklist_count": blocked_count,
        "policy_fail_count": int((policy["status"] == "fail").sum()),
        "local_credentials_file_exists": LOCAL_CREDENTIALS.exists(),
        "model_decision": "no_model_promotion_credential_bootstrap_only",
    }
    manifest_path = OUTPUT_DIR / "agent_run_manifest.json"
    manifest = {
        "run_id": "20260527_v3_37_credential_bootstrap_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": VERSION,
        "baseline": BASELINE,
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_37_credential_bootstrap.py",
        "config": {
            "no_live_download": True,
            "no_secret_output": True,
            "no_model_promotion": True,
        },
        "data_refs": [
            ".gitignore",
            "configs/data_credentials.example.json",
            "configs/pit_data_pilot_v3_36.json",
            "outputs/agent_runs/v3_36/pit_data_pilot_readiness/credential_readiness.csv",
            "outputs/agent_runs/v3_36/pit_data_pilot_readiness/blocked_reason.csv",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_37_credential_bootstrap.py",
            "strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py",
            "strategy_lab/agents/data_steward/AGENT.md",
        ],
        "output_dir": rel(OUTPUT_DIR),
        "allowed_inputs": [
            ".gitignore",
            "configs/data_credentials.example.json",
            "configs/pit_data_pilot_v3_36.json",
            "outputs/agent_runs/v3_36/pit_data_pilot_readiness/",
            "strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py",
            "data_catalog/",
        ],
        "artifacts": [*changed_files, rel(self_check_path)],
        "outputs": [*changed_files, rel(self_check_path)],
        "changed_files": [
            ".gitignore",
            "configs/data_credentials.local.template.json",
            "configs/pit_data_pilot_v3_36_execute.example.json",
            *changed_files,
            rel(self_check_path),
            rel(manifest_path),
        ],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": blocked_count,
        "limitations": [
            "No real credential values were available in this run.",
            "No pilot acquisition was executed.",
            "The next run still requires user-provided local credentials.",
        ],
        "risk_flags": [
            "credentials_missing" if blocked_count else "credentials_ready",
            "pilot_not_acquired",
        ],
        "next_decision": "Provide local credentials, then rerun V3.36 with explicit pilot execution enabled.",
        "handoff_summary": "V3.37 adds safe credential bootstrap artifacts and keeps acquisition blocked until secrets are supplied locally.",
    }
    write_json(manifest, manifest_path)

    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "self_check_pass": fail_count == 0,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
