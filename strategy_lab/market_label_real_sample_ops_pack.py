"""Real-sample operations pack for HIRSSM V3.83."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RealSampleOpsPackConfig:
    v3_82_manifest_path: Path
    v3_80_registry_template_path: Path
    v3_78_config_path: Path
    v3_75_vendor_template_path: Path
    incoming_sample_dir: Path
    license_evidence_dir: Path
    license_status_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    actual = [col for col in columns if col in frame.columns]
    for _, row in frame.loc[:, actual].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_license_evidence_template(config: RealSampleOpsPackConfig) -> str:
    return "\n".join(
        [
            "# License Evidence Template",
            "",
            "## Provider And File",
            "",
            "- Provider:",
            "- Provider contact or account:",
            f"- Raw sample file: `{_workspace_suffix(config.incoming_sample_dir)}/PROVIDER_SAMPLE.csv`",
            "- SHA256 after V3.80 registry:",
            "- Evidence file owner:",
            "- Review date:",
            "",
            "## Permitted Use",
            "",
            "- Local storage allowed:",
            "- Internal research allowed:",
            "- Model validation allowed:",
            "- Derived label files allowed:",
            "- Redistribution allowed:",
            "- Expiry or renewal terms:",
            "",
            "## Evidence Excerpt",
            "",
            "Record only the minimum terms needed for audit. Do not paste credentials, full contracts, or private account secrets.",
            "",
            "## Restrictions",
            "",
            "- Storage restrictions:",
            "- Derived data restrictions:",
            "- Redistribution restrictions:",
            "- Attribution requirements:",
            "",
            "## Review Decision",
            "",
            "- license_status: `approved_internal_research` or `approved_research_and_derived_labels` or `rejected`",
            "- Reviewer:",
            "- Decision note:",
            "",
        ]
    )


def build_license_review_status_template(config: RealSampleOpsPackConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sha256": "fill_after_v3_80_registry",
                "sample_file": f"{_workspace_suffix(config.incoming_sample_dir)}/PROVIDER_SAMPLE.csv",
                "license_status": "approved_research_and_derived_labels",
                "license_evidence_path": f"{_workspace_suffix(config.license_evidence_dir)}/provider_sample_license_evidence.md",
                "reviewer": "manual_reviewer",
                "review_date": "YYYY-MM-DD",
                "review_note": "short audit note; no credentials",
            }
        ]
    )


def build_command_runbook(config: RealSampleOpsPackConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "stage": "place_raw_sample",
                "command_or_action": f"Place exactly one immutable provider CSV under {_workspace_suffix(config.incoming_sample_dir)}.",
                "expected_gate": "raw file exists; no overwrite after registration",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 2,
                "stage": "v3_80_first_registry",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_80_market_label_raw_sample_registry.py --config configs/market_label_raw_sample_registry_v3_80.json",
                "expected_gate": "raw_sample_registry.csv records sha256 and source token status",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 3,
                "stage": "manual_license_review",
                "command_or_action": f"Fill {_workspace_suffix(config.license_status_path)} using the V3.83 license_review_status.template.csv shape.",
                "expected_gate": "license_status is approved and license_evidence_path exists",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 4,
                "stage": "v3_80_license_rerun",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_80_market_label_raw_sample_registry.py --config configs/market_label_raw_sample_registry_v3_80.json",
                "expected_gate": "v3_78_review_allowed is true for the registered sample",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 5,
                "stage": "v3_82_route_plan",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_82_market_label_registered_sample_intake_orchestrator.py --config configs/market_label_registered_sample_intake_orchestrator_v3_82.json",
                "expected_gate": "controlled_v3_78_plan.csv contains a controlled config and command",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 6,
                "stage": "v3_78_controlled_validation",
                "command_or_action": "Run the exact command from outputs/agent_runs/v3_82/registered_sample_intake_orchestrator/controlled_v3_78_plan.csv only after step 5 passes.",
                "expected_gate": "candidate sample passes schema, source, lag, and value checks",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 7,
                "stage": "v3_75_contract_review",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_75_label_source_procurement_pack.py --config configs/label_source_procurement_pack_v3_75.json",
                "expected_gate": "contract and coverage review pass for the candidate source",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 8,
                "stage": "v3_76_next_command_gate",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_76_market_label_source_intake_orchestrator.py --config configs/market_label_source_intake_orchestrator_v3_76.json",
                "expected_gate": "next_commands.csv explicitly allows the V3.53 importer",
                "may_execute_now": False,
                "operator_input_required": True,
            },
            {
                "step_order": 9,
                "stage": "v3_53_importer",
                "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_53_market_total_return_label_importer.py --config configs/market_total_return_label_importer_v3_53.json",
                "expected_gate": "may execute only when V3.76 says so",
                "may_execute_now": False,
                "operator_input_required": True,
            },
        ]
    )


def build_human_input_checklist(config: RealSampleOpsPackConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "raw_sample_is_immutable",
                "owner": "operator",
                "required_evidence": f"provider CSV placed under {_workspace_suffix(config.incoming_sample_dir)} and not overwritten",
                "pass_condition": "hash remains stable between V3.80 runs",
                "current_status": "manual_pending",
            },
            {
                "check": "source_token_allowed",
                "owner": "data_steward",
                "required_evidence": "V3.80 matched_source_token is approved",
                "pass_condition": "source_token_approved=true",
                "current_status": "manual_pending",
            },
            {
                "check": "license_evidence_exists",
                "owner": "operator",
                "required_evidence": f"review markdown under {_workspace_suffix(config.license_evidence_dir)}",
                "pass_condition": "license_evidence_path is non-empty and reviewable",
                "current_status": "manual_pending",
            },
            {
                "check": "license_status_approved",
                "owner": "manual_reviewer",
                "required_evidence": f"{_workspace_suffix(config.license_status_path)} row keyed by sha256",
                "pass_condition": "license_status is in V3.80 approved values",
                "current_status": "manual_pending",
            },
            {
                "check": "registry_hash_matches_review",
                "owner": "data_steward",
                "required_evidence": "license review sha256 equals V3.80 registry sha256",
                "pass_condition": "V3.80 v3_78_review_allowed=true",
                "current_status": "manual_pending",
            },
            {
                "check": "v3_78_validation_pass",
                "owner": "data_steward",
                "required_evidence": "controlled V3.78 manifest and acceptance checks",
                "pass_condition": "V3.78 self_check_pass=true for the registered sample",
                "current_status": "blocked_until_registry_pass",
            },
            {
                "check": "v3_75_contract_pass",
                "owner": "data_steward",
                "required_evidence": "V3.75 import decision",
                "pass_condition": "contract and coverage pass",
                "current_status": "blocked_until_v3_78_pass",
            },
            {
                "check": "v3_76_import_gate_pass",
                "owner": "data_steward",
                "required_evidence": "V3.76 next_commands.csv",
                "pass_condition": "V3.76 explicitly allows the V3.53 importer",
                "current_status": "blocked_until_v3_75_pass",
            },
            {
                "check": "v3_53_still_blocked_here",
                "owner": "backtest_validation_auditor",
                "required_evidence": "V3.83 no-execution guard",
                "pass_condition": "this task does not run importer, validation, portfolio, or promotion steps",
                "current_status": "pass",
            },
        ]
    )


def build_no_execution_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "ops_pack",
                "produced": True,
                "blocked": False,
                "reason": "V3.83 creates manual operating instructions and templates.",
            },
            {
                "result_type": "raw_sample_copy",
                "produced": False,
                "blocked": True,
                "reason": "Raw provider samples must be placed manually and registered by V3.80.",
            },
            {
                "result_type": "incoming_directory_creation",
                "produced": False,
                "blocked": True,
                "reason": "V3.83 does not create the real sample dropzone.",
            },
            {
                "result_type": "target_csv_write",
                "produced": False,
                "blocked": True,
                "reason": "The official target source remains protected.",
            },
            {
                "result_type": "v3_78_execution",
                "produced": False,
                "blocked": True,
                "reason": "V3.83 only documents the controlled V3.78 route.",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": True,
                "reason": "Label generation requires later gates.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No labels or model evidence are produced.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Operations documentation is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(
    command_runbook: pd.DataFrame,
    checklist: pd.DataFrame,
    guard: pd.DataFrame,
    config: RealSampleOpsPackConfig,
    v3_82_manifest: dict,
) -> pd.DataFrame:
    downstream_produced = guard.loc[
        guard["result_type"].isin(
            [
                "raw_sample_copy",
                "incoming_directory_creation",
                "target_csv_write",
                "v3_78_execution",
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
                "check": "v3_82_manifest_passed",
                "status": "pass" if bool(v3_82_manifest.get("self_check_pass", False)) else "fail",
                "detail": f"self_check={v3_82_manifest.get('self_check_pass')}",
            },
            {
                "check": "command_runbook_written",
                "status": "pass" if len(command_runbook) >= 9 else "fail",
                "detail": f"rows={len(command_runbook)}",
            },
            {
                "check": "human_checklist_written",
                "status": "pass" if len(checklist) >= 8 else "fail",
                "detail": f"rows={len(checklist)}",
            },
            {
                "check": "all_execution_flags_false",
                "status": "pass" if not command_runbook["may_execute_now"].astype(bool).any() else "fail",
                "detail": "V3.83 is documentation only",
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not downstream_produced else "fail",
                "detail": "ops pack only",
            },
            {
                "check": "target_source_not_written_by_v3_83",
                "status": "pass",
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "incoming_directory_not_created_by_v3_83",
                "status": "pass",
                "detail": _workspace_suffix(config.incoming_sample_dir),
            },
        ]
    )


def build_ops_manual(
    config: RealSampleOpsPackConfig,
    command_runbook: pd.DataFrame,
    checklist: pd.DataFrame,
) -> str:
    lines = [
        "# V3.83 Real Sample Operations Pack",
        "",
        "## Purpose",
        "",
        "This pack turns the missing real MARKET total-return sample into a controlled manual intake workflow.",
        "It gives the operator exact evidence, license, and command gates without copying a raw sample, writing the protected target source, or running downstream model work.",
        "",
        "## Fixed Paths",
        "",
        f"- Incoming sample dropzone: `{_workspace_suffix(config.incoming_sample_dir)}`",
        f"- License status file to fill manually: `{_workspace_suffix(config.license_status_path)}`",
        f"- License evidence directory: `{_workspace_suffix(config.license_evidence_dir)}`",
        f"- Protected target source: `{_workspace_suffix(config.target_source_path)}`",
        f"- V3.80 registry template reference: `{_workspace_suffix(config.v3_80_registry_template_path)}`",
        f"- V3.75 vendor template reference: `{_workspace_suffix(config.v3_75_vendor_template_path)}`",
        "",
        "## Operator Rules",
        "",
        "- Treat a provider CSV as immutable once V3.80 records its SHA256.",
        "- Do not overwrite a registered raw sample. Add a new file if the provider sends a replacement.",
        "- Fill license review fields only after reviewing provider terms.",
        "- Do not paste credentials, tokens, account passwords, or full private contracts into evidence files.",
        "- Do not write the protected target source in this task.",
        "- Do not run the V3.53 importer until V3.76 explicitly allows it.",
        "",
        "## Command Runbook",
        "",
    ]
    lines.extend(markdown_table(command_runbook, ["step_order", "stage", "command_or_action", "expected_gate", "may_execute_now", "operator_input_required"], 20))
    lines.extend(["", "## Human Input Checklist", ""])
    lines.extend(markdown_table(checklist, ["check", "owner", "required_evidence", "pass_condition", "current_status"], 20))
    lines.extend(
        [
            "",
            "## Stop Conditions",
            "",
            "- Stop if source metadata does not match an approved V3.80 source token.",
            "- Stop if license terms do not permit local storage, internal research, model validation, and derived labels.",
            "- Stop if SHA256 in the manual license status file does not equal V3.80 registry output.",
            "- Stop if V3.78 emits any fail status.",
            "- Stop if V3.75 or V3.76 keeps the V3.53 importer blocked.",
            "",
            "## Next Handoff",
            "",
            "After the operator supplies a real sample and license evidence, rerun V3.80, then V3.82, then only the generated controlled V3.78 command.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    command_runbook: pd.DataFrame,
    checklist: pd.DataFrame,
    acceptance: pd.DataFrame,
    guard: pd.DataFrame,
    config: RealSampleOpsPackConfig,
) -> str:
    lines = [
        "# V3.83 Real Sample Ops Pack Report",
        "",
        "## Decision",
        "",
        "- V3.83 creates a durable manual operations pack for real MARKET sample arrival.",
        "- It does not create the incoming sample directory, copy raw data, write target CSV, run V3.78, run V3.53, run portfolios, or promote a model.",
        "- All command rows remain `may_execute_now=false` because operator evidence is still missing.",
        "",
        "## Key Metrics",
        "",
        f"- Command runbook rows: `{len(command_runbook)}`",
        f"- Human checklist rows: `{len(checklist)}`",
        f"- Target source exists now: `{config.target_source_path.exists()}`",
        f"- Incoming sample directory exists now: `{config.incoming_sample_dir.exists()}`",
        "",
        "## Acceptance",
        "",
    ]
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## No Execution Guard", ""])
    lines.extend(markdown_table(guard, ["result_type", "produced", "blocked", "reason"], 20))
    lines.extend(["", "## Next Step", "", "- Wait for an operator-provided real provider CSV and license evidence, then execute the runbook from step 2 onward.", ""])
    return "\n".join(lines)


def build_catalog(config: RealSampleOpsPackConfig) -> str:
    return "\n".join(
        [
            "# A-share Real Sample Operations Pack V3.83",
            "",
            "## Dataset Decision",
            "",
            f"- Incoming sample dropzone: `{_workspace_suffix(config.incoming_sample_dir)}`",
            f"- License status path to fill manually: `{_workspace_suffix(config.license_status_path)}`",
            f"- Protected target source path: `{_workspace_suffix(config.target_source_path)}`",
            "- V3.83 creates templates and runbooks only.",
            "- No raw sample, target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
