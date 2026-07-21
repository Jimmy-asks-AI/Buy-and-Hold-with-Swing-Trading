"""Real-sample intake dry-run harness for HIRSSM V3.85."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_raw_sample_registry import (
    RawSampleRegistryConfig,
    build_raw_sample_registry,
)
from market_label_registered_sample_intake_orchestrator import (
    RegisteredSampleIntakeConfig,
    build_controlled_v3_78_configs,
    build_eligible_samples,
    build_execution_plan,
    build_registry_snapshot,
)


@dataclass(frozen=True)
class RealSampleDryRunConfig:
    v3_83_manifest_path: Path
    v3_84_manifest_path: Path
    v3_78_base_config_path: Path
    v3_80_base_config_path: Path
    v3_82_base_config_path: Path
    real_incoming_sample_dir: Path
    real_license_status_path: Path
    real_target_source_path: Path
    output_dir: Path
    catalog_path: Path
    sandbox_root: Path
    sandbox_incoming_sample_dir: Path
    sandbox_license_status_path: Path
    sandbox_target_source_path: Path
    sandbox_registry_path: Path
    controlled_config_dir: Path
    dry_run_sample_rows: int


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def build_dry_run_sample(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    levels = [1000.0]
    for idx in range(1, rows):
        levels.append(round(levels[-1] * (1.001 + (idx % 5) * 0.0002), 6))
    return pd.DataFrame(
        {
            "date": [int(date.strftime("%Y%m%d")) for date in dates],
            "asset_or_index": ["MARKET"] * rows,
            "total_return_index_or_adjusted_close": levels,
            "available_date": [int((date + pd.Timedelta(days=1)).strftime("%Y%m%d")) for date in dates],
            "data_source": ["csindex_official_total_return_service"] * rows,
            "source_vintage": ["dry_run_v3_85_20260602"] * rows,
        }
    )


def make_raw_registry_config(
    base: dict[str, Any],
    config: RealSampleDryRunConfig,
    license_status_path: Path,
    output_dir: Path,
) -> RawSampleRegistryConfig:
    return RawSampleRegistryConfig(
        v3_78_manifest_path=Path(base["v3_78_manifest_path"]),
        v3_79_manifest_path=Path(base["v3_79_manifest_path"]),
        incoming_sample_dir=config.sandbox_incoming_sample_dir,
        target_source_path=config.sandbox_target_source_path,
        previous_registry_path=config.sandbox_registry_path,
        license_status_path=license_status_path,
        output_dir=output_dir,
        catalog_path=config.catalog_path,
        approved_source_tokens=tuple(str(x) for x in base["approved_source_tokens"]),
        allowed_extensions=tuple(str(x) for x in base["allowed_extensions"]),
        license_approved_values=tuple(str(x) for x in base["license_approved_values"]),
    )


def build_license_evidence(sample_path: Path, sha256: str) -> str:
    return "\n".join(
        [
            "# V3.85 Dry-run License Evidence",
            "",
            "This is synthetic dry-run evidence. It is not a real provider license.",
            "",
            f"- sample_file: `{_workspace_suffix(sample_path)}`",
            f"- sha256: `{sha256}`",
            "- license_status: `approved_research_and_derived_labels`",
            "- permitted_scope: local dry-run validation only",
            "- reviewer: dry_run_harness",
            "- review_date: 2026-06-02",
            "",
        ]
    )


def build_license_status(sample_path: Path, sha256: str, evidence_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sha256": sha256,
                "sample_file": _workspace_suffix(sample_path),
                "license_status": "approved_research_and_derived_labels",
                "license_evidence_path": _workspace_suffix(evidence_path),
                "reviewer": "dry_run_harness",
                "review_date": "2026-06-02",
                "review_note": "synthetic dry-run only; no real provider data",
            }
        ]
    )


def make_registered_sample_config(config: RealSampleDryRunConfig, v3_82_base: dict[str, Any]) -> RegisteredSampleIntakeConfig:
    return RegisteredSampleIntakeConfig(
        v3_80_manifest_path=Path(v3_82_base["v3_80_manifest_path"]),
        v3_80_registry_path=config.sandbox_registry_path,
        v3_81_manifest_path=Path(v3_82_base["v3_81_manifest_path"]),
        v3_78_config_path=config.v3_78_base_config_path,
        target_source_path=config.sandbox_target_source_path,
        output_dir=config.output_dir / "route_plan",
        catalog_path=config.catalog_path,
        controlled_config_dir=config.controlled_config_dir,
        execute_v3_78=False,
    )


def rewrite_controlled_payloads(
    payloads: dict[str, dict[str, Any]],
    config: RealSampleDryRunConfig,
) -> dict[str, dict[str, Any]]:
    rewritten: dict[str, dict[str, Any]] = {}
    for idx, (path_text, payload) in enumerate(payloads.items(), start=1):
        raw = dict(payload)
        raw["task_id"] = f"20260602_v3_85_controlled_v3_78_dry_run_{idx}"
        raw["target_source_path"] = _workspace_suffix(config.sandbox_target_source_path)
        raw["output_dir"] = _workspace_suffix(config.output_dir / f"controlled_v3_78_run_{idx}")
        raw["catalog_path"] = f"data_catalog/a_share_v3_85_controlled_v3_78_dry_run_{idx}.md"
        raw["execute_context"] = "v3_85_real_sample_intake_dry_run"
        raw["allowed_inputs"] = [
            raw.get("v3_77_manifest_path", ""),
            raw.get("v3_77_source_candidates_path", ""),
            raw.get("v3_75_source_contract_path", ""),
            f"{raw.get('incoming_sample_dir', '')}/*.csv",
            raw["target_source_path"],
        ]
        rewritten[path_text] = raw
    return rewritten


def build_v3_78_execution_plan(controlled_plan: pd.DataFrame, rewritten_payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, (path_text, payload) in enumerate(rewritten_payloads.items(), start=1):
        rows.append(
            {
                "step_order": idx,
                "controlled_config_path": path_text,
                "command": f"python -B -X utf8 strategy_lab/hirssm_v3_78_market_label_sample_intake_validator.py --config {path_text}",
                "v3_78_output_dir": payload["output_dir"],
                "planned_by_v3_82": bool(not controlled_plan.empty),
                "may_execute_now": True,
                "execution_scope": "sandbox_dry_run_only",
            }
        )
    return pd.DataFrame(rows)


def run_v3_78_command(root: Path, command: str) -> tuple[int, str, str]:
    parts = command.split()
    proc = subprocess.run(parts, cwd=root, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_pipeline_trace(
    stage1_registry: pd.DataFrame,
    stage2_registry: pd.DataFrame,
    eligible: pd.DataFrame,
    controlled_plan: pd.DataFrame,
    execution_results: pd.DataFrame,
    v3_78_decisions: pd.DataFrame,
) -> pd.DataFrame:
    stage1_allowed = int(stage1_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage1_registry.columns else 0
    stage2_allowed = int(stage2_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage2_registry.columns else 0
    v3_78_pass = int(v3_78_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if "decision" in v3_78_decisions.columns else 0
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "stage": "synthetic_sample_created",
                "status": "pass" if not stage1_registry.empty else "fail",
                "detail": f"registry_rows={len(stage1_registry)}",
            },
            {
                "step_order": 2,
                "stage": "v3_80_without_license",
                "status": "pass" if stage1_allowed == 0 else "fail",
                "detail": f"review_allowed={stage1_allowed}",
            },
            {
                "step_order": 3,
                "stage": "sandbox_license_status_written",
                "status": "pass",
                "detail": "license status exists only in V3.85 sandbox",
            },
            {
                "step_order": 4,
                "stage": "v3_80_with_license",
                "status": "pass" if stage2_allowed >= 1 else "fail",
                "detail": f"review_allowed={stage2_allowed}",
            },
            {
                "step_order": 5,
                "stage": "v3_82_route_plan",
                "status": "pass" if len(eligible) >= 1 and len(controlled_plan) >= 1 else "fail",
                "detail": f"eligible={len(eligible)};controlled_plan={len(controlled_plan)}",
            },
            {
                "step_order": 6,
                "stage": "controlled_v3_78_execution",
                "status": "pass" if not execution_results.empty and execution_results["returncode"].eq(0).all() and v3_78_pass >= 1 else "fail",
                "detail": f"returncodes={','.join(execution_results['returncode'].astype(str)) if not execution_results.empty else ''};passes={v3_78_pass}",
            },
            {
                "step_order": 7,
                "stage": "official_target_still_blocked",
                "status": "pass",
                "detail": "dry-run does not write the official target source",
            },
        ]
    )


def build_no_execution_guard(real_incoming_before: bool, real_incoming_after: bool, real_target_before: bool, real_target_after: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "dry_run_synthetic_sample",
                "produced": True,
                "blocked": False,
                "reason": "Synthetic sample is written only under the V3.85 sandbox.",
            },
            {
                "result_type": "sandbox_license_status",
                "produced": True,
                "blocked": False,
                "reason": "License status is written only under the V3.85 sandbox.",
            },
            {
                "result_type": "v3_80_registry_dry_run",
                "produced": True,
                "blocked": False,
                "reason": "Registry logic is exercised on the sandbox sample.",
            },
            {
                "result_type": "v3_82_route_plan_dry_run",
                "produced": True,
                "blocked": False,
                "reason": "Route logic is exercised on the sandbox registry.",
            },
            {
                "result_type": "controlled_v3_78_validation_dry_run",
                "produced": True,
                "blocked": False,
                "reason": "Controlled V3.78 validation is allowed in the sandbox.",
            },
            {
                "result_type": "real_incoming_sample_dir_write",
                "produced": real_incoming_before != real_incoming_after,
                "blocked": True,
                "reason": "Real incoming sample path must not be created or modified by V3.85.",
            },
            {
                "result_type": "real_license_status_write",
                "produced": False,
                "blocked": True,
                "reason": "Real license status remains manual operator input.",
            },
            {
                "result_type": "official_target_csv_write",
                "produced": real_target_before != real_target_after,
                "blocked": True,
                "reason": "Official target source remains protected.",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": True,
                "reason": "Label generation remains blocked after dry-run validation.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No governed label file is created by this dry-run.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Dry-run plumbing evidence is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(
    pipeline_trace: pd.DataFrame,
    stage1_registry: pd.DataFrame,
    stage2_registry: pd.DataFrame,
    eligible: pd.DataFrame,
    controlled_plan: pd.DataFrame,
    execution_results: pd.DataFrame,
    v3_78_decisions: pd.DataFrame,
    guard: pd.DataFrame,
    real_incoming_before: bool,
    real_incoming_after: bool,
    real_target_before: bool,
    real_target_after: bool,
) -> pd.DataFrame:
    stage1_allowed = int(stage1_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage1_registry.columns else 0
    stage2_allowed = int(stage2_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage2_registry.columns else 0
    v3_78_pass = int(v3_78_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if "decision" in v3_78_decisions.columns else 0
    v3_78_rejected = int(v3_78_decisions["decision"].astype(str).eq("rejected_or_needs_repair").sum()) if "decision" in v3_78_decisions.columns else 0
    forbidden_produced = guard.loc[
        guard["result_type"].isin(
            [
                "real_incoming_sample_dir_write",
                "real_license_status_write",
                "official_target_csv_write",
                "v3_53_label_generation",
                "portfolio_backtest",
                "model_promotion",
            ]
        ),
        "produced",
    ].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "pipeline_trace_passed",
                "status": "pass" if not pipeline_trace["status"].eq("fail").any() else "fail",
                "detail": ";".join(pipeline_trace.loc[pipeline_trace["status"].eq("fail"), "stage"].astype(str)),
            },
            {
                "check": "v3_80_without_license_blocks_review",
                "status": "pass" if stage1_allowed == 0 else "fail",
                "detail": f"review_allowed={stage1_allowed}",
            },
            {
                "check": "v3_80_with_license_allows_review",
                "status": "pass" if stage2_allowed >= 1 else "fail",
                "detail": f"review_allowed={stage2_allowed}",
            },
            {
                "check": "v3_82_plans_controlled_v3_78",
                "status": "pass" if len(eligible) >= 1 and len(controlled_plan) >= 1 else "fail",
                "detail": f"eligible={len(eligible)};controlled_plan={len(controlled_plan)}",
            },
            {
                "check": "controlled_v3_78_executes_in_sandbox",
                "status": "pass" if not execution_results.empty and execution_results["returncode"].eq(0).all() and v3_78_pass >= 1 else "fail",
                "detail": f"returncodes={','.join(execution_results['returncode'].astype(str)) if not execution_results.empty else ''};passes={v3_78_pass}",
            },
            {
                "check": "controlled_v3_78_has_no_rejected_samples",
                "status": "pass" if v3_78_rejected == 0 else "fail",
                "detail": f"rejected={v3_78_rejected}",
            },
            {
                "check": "real_incoming_dir_state_unchanged",
                "status": "pass" if real_incoming_before == real_incoming_after else "fail",
                "detail": f"before={real_incoming_before};after={real_incoming_after}",
            },
            {
                "check": "real_target_source_state_unchanged",
                "status": "pass" if real_target_before == real_target_after else "fail",
                "detail": f"before={real_target_before};after={real_target_after}",
            },
            {
                "check": "forbidden_outputs_not_produced",
                "status": "pass" if not forbidden_produced else "fail",
                "detail": "real paths and downstream model work stay blocked",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    actual = [col for col in columns if col in frame.columns]
    for _, row in frame.loc[:, actual].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_report(
    pipeline_trace: pd.DataFrame,
    acceptance: pd.DataFrame,
    stage2_registry: pd.DataFrame,
    controlled_plan: pd.DataFrame,
    v3_78_decisions: pd.DataFrame,
    config: RealSampleDryRunConfig,
) -> str:
    allowed = int(stage2_registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in stage2_registry.columns else 0
    v3_78_pass = int(v3_78_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if "decision" in v3_78_decisions.columns else 0
    lines = [
        "# V3.85 Real Sample Intake Dry-run Harness",
        "",
        "## Decision",
        "",
        "- V3.85 proves the governed real-sample intake path using a synthetic sandbox sample.",
        "- It exercises V3.80 registry logic, V3.82 route planning, and controlled V3.78 validation.",
        "- It does not create the real incoming directory, write real license status, write the official target source, generate labels, run portfolios, or promote a model.",
        "",
        "## Key Metrics",
        "",
        f"- Dry-run sample rows: `{config.dry_run_sample_rows}`",
        f"- V3.80 review-allowed rows after license: `{allowed}`",
        f"- Controlled V3.78 configs planned: `{len(controlled_plan)}`",
        f"- V3.78 passing samples: `{v3_78_pass}`",
        f"- Sandbox root: `{_workspace_suffix(config.sandbox_root)}`",
        "",
        "## Pipeline Trace",
        "",
    ]
    lines.extend(markdown_table(pipeline_trace, ["step_order", "stage", "status", "detail"], 20))
    lines.extend(["", "## V3.82 Controlled Plan", ""])
    lines.extend(markdown_table(controlled_plan, ["plan_id", "sample_file", "controlled_config_path", "execution_status", "reason"], 20))
    lines.extend(["", "## V3.78 Decision", ""])
    lines.extend(markdown_table(v3_78_decisions, ["sample_file", "row_count", "blocking_fail_count", "warning_count", "decision", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Use the same flow for a real licensed provider sample, but keep the official target write blocked until V3.75 and V3.76 pass.", ""])
    return "\n".join(lines)


def build_catalog(config: RealSampleDryRunConfig, pipeline_trace: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# A-share Real Sample Intake Dry-run Harness V3.85",
            "",
            "## Dataset Decision",
            "",
            f"- Sandbox root: `{_workspace_suffix(config.sandbox_root)}`",
            f"- Pipeline stages: `{len(pipeline_trace)}`",
            f"- Failed stages: `{int(pipeline_trace['status'].eq('fail').sum()) if not pipeline_trace.empty else 0}`",
            f"- Real incoming sample dir: `{_workspace_suffix(config.real_incoming_sample_dir)}`",
            f"- Real target source path: `{_workspace_suffix(config.real_target_source_path)}`",
            "- Only synthetic dry-run data is generated.",
            "- No official target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
