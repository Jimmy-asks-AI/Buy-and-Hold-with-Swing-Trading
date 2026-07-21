"""Controlled registered-sample intake orchestrator for HIRSSM V3.82."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegisteredSampleIntakeConfig:
    v3_80_manifest_path: Path
    v3_80_registry_path: Path
    v3_81_manifest_path: Path
    v3_78_config_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    controlled_config_dir: Path
    execute_v3_78: bool


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value or "").strip()


def build_registry_snapshot(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame(
            [
                {
                    "registry_status": "missing_registry_rows",
                    "sample_file": "",
                    "file_exists": False,
                    "sha256": "",
                    "source_token_approved": False,
                    "license_status": "",
                    "license_evidence_path": "",
                    "v3_78_review_allowed": False,
                    "route_status": "blocked",
                    "route_reason": "V3.80 registry has no rows",
                }
            ]
        )
    rows = []
    for row in registry.itertuples(index=False):
        review_allowed = _bool(getattr(row, "v3_78_review_allowed", False))
        file_exists = _bool(getattr(row, "file_exists", False))
        sha256 = _text(getattr(row, "sha256", ""))
        sample_file = _text(getattr(row, "sample_file", ""))
        route_status = "eligible_for_v3_78_plan" if review_allowed and file_exists and sha256 else "blocked"
        if route_status == "eligible_for_v3_78_plan":
            reason = "registry source and license gates passed"
        elif not sample_file or "incoming_samples" in sample_file and not file_exists:
            reason = "no real incoming sample registered"
        elif not _bool(getattr(row, "source_token_approved", False)):
            reason = "source token is not approved"
        elif not str(getattr(row, "license_evidence_path", "") or ""):
            reason = "license evidence is missing"
        else:
            reason = "registry row is not eligible"
        rows.append(
            {
                "registry_status": getattr(row, "registry_status", ""),
                "sample_file": sample_file,
                "file_exists": file_exists,
                "sha256": sha256,
                "source_token_approved": _bool(getattr(row, "source_token_approved", False)),
                "license_status": _text(getattr(row, "license_status", "")),
                "license_evidence_path": _text(getattr(row, "license_evidence_path", "")),
                "v3_78_review_allowed": review_allowed,
                "route_status": route_status,
                "route_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_eligible_samples(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=list(snapshot.columns) + ["controlled_config_path", "v3_78_command"])
    eligible = snapshot.loc[snapshot["route_status"].astype(str).eq("eligible_for_v3_78_plan")].copy()
    if eligible.empty:
        return pd.DataFrame(columns=list(snapshot.columns) + ["controlled_config_path", "v3_78_command"])
    eligible["controlled_config_path"] = ""
    eligible["v3_78_command"] = ""
    return eligible.reset_index(drop=True)


def build_controlled_v3_78_configs(
    eligible: pd.DataFrame,
    base_v3_78_config: dict[str, Any],
    config: RegisteredSampleIntakeConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    columns = [
        "plan_id",
        "sample_file",
        "sha256",
        "license_status",
        "controlled_config_path",
        "v3_78_command",
        "may_execute_now",
        "execution_status",
        "reason",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=columns), {}
    plan_rows = []
    config_payloads: dict[str, dict[str, Any]] = {}
    for idx, row in eligible.reset_index(drop=True).iterrows():
        sample_path = str(row["sample_file"])
        sample_stem = Path(sample_path).stem or f"sample_{idx + 1}"
        config_name = f"v3_82_registered_sample_{idx + 1}_{sample_stem}.json"
        controlled_path = config.controlled_config_dir / config_name
        sample_input_dir = str(Path(sample_path).parent).replace("\\", "/")
        raw = dict(base_v3_78_config)
        raw["task_id"] = f"20260601_v3_82_registered_sample_v3_78_{idx + 1}"
        raw["incoming_sample_dir"] = sample_input_dir
        raw["output_dir"] = f"outputs/agent_runs/v3_82/registered_sample_intake_orchestrator/v3_78_candidate_{idx + 1}_{sample_stem}"
        raw["catalog_path"] = f"data_catalog/a_share_v3_82_registered_sample_v3_78_candidate_{idx + 1}.md"
        raw["registry_sha256"] = row["sha256"]
        raw["registry_sample_file"] = sample_path
        raw["registry_license_evidence_path"] = row["license_evidence_path"]
        raw["execute_context"] = "v3_82_controlled_registered_sample_plan"
        config_payloads[_workspace_suffix(controlled_path)] = raw
        command = f"python -B -X utf8 strategy_lab/hirssm_v3_78_market_label_sample_intake_validator.py --config {_workspace_suffix(controlled_path)}"
        plan_rows.append(
            {
                "plan_id": f"registered_sample_{idx + 1}",
                "sample_file": sample_path,
                "sha256": row["sha256"],
                "license_status": row["license_status"],
                "controlled_config_path": _workspace_suffix(controlled_path),
                "v3_78_command": command,
                "may_execute_now": False,
                "execution_status": "planned_only",
                "reason": "V3.82 writes a controlled plan but does not execute V3.78 by default",
            }
        )
    return pd.DataFrame(plan_rows, columns=columns), config_payloads


def build_execution_plan(eligible: pd.DataFrame, controlled_plan: pd.DataFrame, config: RegisteredSampleIntakeConfig) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame(
            [
                {
                    "step_order": 1,
                    "action": "wait for V3.80 registry row with v3_78_review_allowed=true",
                    "status": "active",
                    "may_execute_now": False,
                    "reason": "no registered and licensed real sample is available",
                },
                {
                    "step_order": 2,
                    "action": "run V3.78 using V3.82 controlled config",
                    "status": "blocked",
                    "may_execute_now": False,
                    "reason": "blocked until eligible registered sample exists",
                },
                {
                    "step_order": 3,
                    "action": "rerun V3.75 on controlled copy only after V3.78 pass",
                    "status": "blocked",
                    "may_execute_now": False,
                    "reason": "V3.75 contract gate remains required",
                },
                {
                    "step_order": 4,
                    "action": "write final target source",
                    "status": "blocked",
                    "may_execute_now": False,
                    "reason": f"target remains protected at {_workspace_suffix(config.target_source_path)}",
                },
            ]
        )
    rows = []
    for idx, row in controlled_plan.iterrows():
        rows.append(
            {
                "step_order": idx + 1,
                "action": row["v3_78_command"],
                "status": "active" if config.execute_v3_78 else "planned",
                "may_execute_now": bool(config.execute_v3_78),
                "reason": "operator explicitly enabled execute_v3_78" if config.execute_v3_78 else "planned only; execute_v3_78=false",
            }
        )
    rows.append(
        {
            "step_order": len(rows) + 1,
            "action": "rerun V3.75 after V3.78 pass",
            "status": "blocked",
            "may_execute_now": False,
            "reason": "V3.78 result must pass first",
        }
    )
    return pd.DataFrame(rows)


def build_no_execution_guard(config: RegisteredSampleIntakeConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "registered_sample_route_plan",
                "produced": True,
                "blocked": False,
                "reason": "V3.82 creates route evidence and optional controlled configs.",
            },
            {
                "result_type": "v3_78_execution",
                "produced": False,
                "blocked": not config.execute_v3_78,
                "reason": "V3.82 does not execute V3.78 unless explicitly enabled.",
            },
            {
                "result_type": "target_csv_write",
                "produced": False,
                "blocked": True,
                "reason": "Orchestrator must not write the official target CSV.",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": True,
                "reason": "V3.53 remains blocked until V3.75 and V3.76 pass.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No labels are produced here.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Route planning is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(
    snapshot: pd.DataFrame,
    eligible: pd.DataFrame,
    execution_plan: pd.DataFrame,
    guard: pd.DataFrame,
    config: RegisteredSampleIntakeConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "registry_snapshot_written",
                "status": "pass" if not snapshot.empty else "fail",
                "detail": f"rows={len(snapshot)}",
            },
            {
                "check": "eligible_state_is_explicit",
                "status": "pass" if not eligible.empty or snapshot["route_status"].astype(str).eq("blocked").any() else "fail",
                "detail": f"eligible_rows={len(eligible)}",
            },
            {
                "check": "execution_plan_written",
                "status": "pass" if not execution_plan.empty else "fail",
                "detail": f"rows={len(execution_plan)}",
            },
            {
                "check": "v3_78_not_executed_by_default",
                "status": "pass" if not config.execute_v3_78 else "fail",
                "detail": f"execute_v3_78={config.execute_v3_78}",
            },
            {
                "check": "target_source_not_written",
                "status": "pass" if not config.target_source_path.exists() else "warn",
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["v3_78_execution", "target_csv_write", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "route planning only",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    actual = [col for col in columns if col in frame.columns]
    for _, row in frame.loc[:, actual].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_report(
    snapshot: pd.DataFrame,
    eligible: pd.DataFrame,
    controlled_plan: pd.DataFrame,
    execution_plan: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: RegisteredSampleIntakeConfig,
) -> str:
    lines = [
        "# V3.82 Registered Sample Intake Orchestrator",
        "",
        "## Decision",
        "",
        "- V3.82 routes only V3.80-approved real samples toward V3.78.",
        "- It does not copy raw samples, write the target source, run V3.78 by default, generate labels, run portfolios, or promote a model.",
        "- If no registry row has `v3_78_review_allowed=true`, the route remains blocked.",
        "",
        "## Key Metrics",
        "",
        f"- Registry rows: `{len(snapshot)}`",
        f"- Eligible V3.78 samples: `{len(eligible)}`",
        f"- Controlled V3.78 configs planned: `{len(controlled_plan)}`",
        f"- execute_v3_78: `{config.execute_v3_78}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        "",
        "## Registry Snapshot",
        "",
    ]
    lines.extend(markdown_table(snapshot, ["sample_file", "file_exists", "sha256", "license_status", "v3_78_review_allowed", "route_status", "route_reason"], 20))
    lines.extend(["", "## Eligible Samples", ""])
    lines.extend(markdown_table(eligible, ["sample_file", "sha256", "license_status", "license_evidence_path"], 20))
    lines.extend(["", "## Controlled V3.78 Plan", ""])
    lines.extend(markdown_table(controlled_plan, ["plan_id", "sample_file", "controlled_config_path", "may_execute_now", "execution_status"], 20))
    lines.extend(["", "## Execution Plan", ""])
    lines.extend(markdown_table(execution_plan, ["step_order", "action", "status", "may_execute_now", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Add a real provider sample plus license evidence, rerun V3.80, then rerun V3.82.", ""])
    return "\n".join(lines)


def build_catalog(snapshot: pd.DataFrame, eligible: pd.DataFrame, controlled_plan: pd.DataFrame, config: RegisteredSampleIntakeConfig) -> str:
    return "\n".join(
        [
            "# A-share Registered Sample Intake Orchestrator V3.82",
            "",
            "## Dataset Decision",
            "",
            f"- Registry rows: `{len(snapshot)}`",
            f"- Eligible V3.78 samples: `{len(eligible)}`",
            f"- Controlled V3.78 configs planned: `{len(controlled_plan)}`",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
