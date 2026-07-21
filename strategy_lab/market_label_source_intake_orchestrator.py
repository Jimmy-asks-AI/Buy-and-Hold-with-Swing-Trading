"""MARKET label-source intake orchestrator for HIRSSM V3.76.

V3.76 turns the post-procurement handoff into a deterministic router. It checks
whether the official/procured target CSV exists and whether V3.75 declared it
ready for V3.53. It only emits commands and guardrail evidence; it does not run
label generation, validation, portfolio backtests, or model promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LabelSourceIntakeConfig:
    v3_75_manifest_path: Path
    v3_75_import_decision_path: Path
    v3_75_procurement_requirements_path: Path
    v3_75_vendor_template_path: Path
    v3_53_config_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    execute_downstream: bool


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
    except Exception:
        return []


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig", usecols=[0], low_memory=False)))
    except Exception:
        try:
            return int(len(pd.read_csv(path, encoding="utf-8-sig", low_memory=False)))
        except Exception:
            return 0


def build_intake_status(
    v3_75_manifest: dict[str, Any],
    import_decision: pd.DataFrame,
    config: LabelSourceIntakeConfig,
) -> pd.DataFrame:
    target_exists = config.target_source_path.exists()
    ready_from_v3_75 = False
    decision_text = ""
    if not import_decision.empty and "ready_for_v3_53" in import_decision.columns:
        ready_from_v3_75 = _bool_text(import_decision["ready_for_v3_53"].iloc[0])
        decision_text = str(import_decision.get("decision", pd.Series([""])).iloc[0])
    manifest_ready = bool(v3_75_manifest.get("self_check_pass", False))
    rows = [
        {
            "check": "v3_75_manifest_passed",
            "status": _status(manifest_ready),
            "detail": f"self_check={v3_75_manifest.get('self_check_pass')};status={v3_75_manifest.get('status')}",
        },
        {
            "check": "target_source_exists",
            "status": _status(target_exists, "blocked"),
            "detail": _workspace_suffix(config.target_source_path),
        },
        {
            "check": "target_source_header_readable",
            "status": _status(bool(_read_header(config.target_source_path)), "blocked"),
            "detail": "|".join(_read_header(config.target_source_path)),
        },
        {
            "check": "target_source_row_count",
            "status": _status(_row_count(config.target_source_path) > 0, "blocked"),
            "detail": f"rows={_row_count(config.target_source_path)}",
        },
        {
            "check": "v3_75_ready_for_v3_53",
            "status": _status(ready_from_v3_75, "blocked"),
            "detail": decision_text,
        },
        {
            "check": "execute_downstream_enabled",
            "status": "pass" if config.execute_downstream else "not_applicable",
            "detail": f"execute_downstream={config.execute_downstream}",
        },
    ]
    return pd.DataFrame(rows)


def build_next_commands(import_decision: pd.DataFrame, config: LabelSourceIntakeConfig) -> pd.DataFrame:
    ready = False
    target_exists = config.target_source_path.exists()
    if not import_decision.empty and "ready_for_v3_53" in import_decision.columns:
        ready = _bool_text(import_decision["ready_for_v3_53"].iloc[0])
    commands = [
        {
            "step_order": 1,
            "condition": "target_source_missing",
            "status": "active" if not target_exists else "done",
            "command_or_action": "place compliant CSV at data_raw/market_labels/market_total_return_index.csv",
            "purpose": "provide official/procured MARKET label source",
            "may_execute_now": False,
        },
        {
            "step_order": 2,
            "condition": "target_source_present",
            "status": "active" if target_exists and not ready else "pending" if not target_exists else "done",
            "command_or_action": "python -B -X utf8 strategy_lab/hirssm_v3_75_label_source_procurement_pack.py --config configs/label_source_procurement_pack_v3_75.json",
            "purpose": "validate target source contract and signal coverage",
            "may_execute_now": target_exists,
        },
        {
            "step_order": 3,
            "condition": "v3_75_ready_for_v3_53_true",
            "status": "active" if ready else "blocked",
            "command_or_action": f"python -B -X utf8 strategy_lab/hirssm_v3_53_market_total_return_label_importer.py --config {_workspace_suffix(config.v3_53_config_path)}",
            "purpose": "generate governed forward total-return labels",
            "may_execute_now": ready and config.execute_downstream,
        },
        {
            "step_order": 4,
            "condition": "v3_53_labels_pass",
            "status": "blocked",
            "command_or_action": "run future V3.77 label validation review before any portfolio harness",
            "purpose": "validate strict survivor labels after official source is available",
            "may_execute_now": False,
        },
    ]
    return pd.DataFrame(commands)


def build_watchlist(requirements: pd.DataFrame, config: LabelSourceIntakeConfig) -> pd.DataFrame:
    rows = [
        {
            "watch_item": "official_target_source_file",
            "path": _workspace_suffix(config.target_source_path),
            "exists": config.target_source_path.exists(),
            "row_count": _row_count(config.target_source_path),
            "required_before": "V3.53",
            "action_if_missing": "obtain vendor/provider CSV using V3.75 provider_request_template.md",
        },
        {
            "watch_item": "v3_75_vendor_template",
            "path": _workspace_suffix(config.v3_75_vendor_template_path),
            "exists": config.v3_75_vendor_template_path.exists(),
            "row_count": _row_count(config.v3_75_vendor_template_path),
            "required_before": "manual delivery",
            "action_if_missing": "rerun V3.75",
        },
    ]
    for row in requirements.head(5).itertuples(index=False):
        rows.append(
            {
                "watch_item": str(getattr(row, "requirement_id", "")),
                "path": "",
                "exists": False,
                "row_count": 0,
                "required_before": "source delivery",
                "action_if_missing": str(getattr(row, "next_action", "")),
            }
        )
    return pd.DataFrame(rows)


def build_no_execution_guard(config: LabelSourceIntakeConfig, next_commands: pd.DataFrame) -> pd.DataFrame:
    downstream_allowed = bool(next_commands.get("may_execute_now", pd.Series(dtype=bool)).astype(bool).any())
    return pd.DataFrame(
        [
            {
                "result_type": "intake_orchestration",
                "produced": True,
                "blocked": False,
                "reason": "V3.76 routing evidence produced",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": not downstream_allowed or not config.execute_downstream,
                "reason": "V3.76 does not execute downstream generation by default",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "labels must pass validation before portfolio harness",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "no validated official labels in this run",
            },
        ]
    )


def build_acceptance_checks(status: pd.DataFrame, next_commands: pd.DataFrame, watchlist: pd.DataFrame, guard: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "intake_status_written",
                "status": "pass" if not status.empty else "fail",
                "detail": f"rows={len(status)}",
            },
            {
                "check": "next_commands_written",
                "status": "pass" if len(next_commands) >= 3 else "fail",
                "detail": f"rows={len(next_commands)}",
            },
            {
                "check": "watchlist_written",
                "status": "pass" if not watchlist.empty else "fail",
                "detail": f"rows={len(watchlist)}",
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "orchestration only",
            },
            {
                "check": "blocked_state_is_explicit",
                "status": "pass" if status["status"].astype(str).isin(["blocked"]).any() or next_commands["status"].astype(str).isin(["active"]).any() else "fail",
                "detail": "target missing or actionable command present",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    for _, row in frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_report(status: pd.DataFrame, next_commands: pd.DataFrame, watchlist: pd.DataFrame, acceptance: pd.DataFrame, config: LabelSourceIntakeConfig) -> str:
    ready = bool(next_commands.loc[next_commands["condition"].eq("v3_75_ready_for_v3_53_true"), "may_execute_now"].any()) if not next_commands.empty else False
    lines = [
        "# V3.76 MARKET Label Source Intake Orchestrator",
        "",
        "## Decision",
        "",
        "- V3.76 checks the post-procurement handoff state and emits the next executable command.",
        "- It does not run V3.53, build labels, run portfolios, or promote any model.",
        "- Current route remains blocked unless the official target CSV is present and V3.75 says it is ready.",
        "",
        "## Key Metrics",
        "",
        f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        f"- V3.53 may execute now: `{ready}`",
        "",
        "## Intake Status",
        "",
    ]
    lines.extend(markdown_table(status, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Commands", ""])
    lines.extend(markdown_table(next_commands, ["step_order", "condition", "status", "command_or_action", "may_execute_now"], 20))
    lines.extend(["", "## Watchlist", ""])
    lines.extend(markdown_table(watchlist, ["watch_item", "path", "exists", "row_count", "required_before", "action_if_missing"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Deliver the target CSV or keep the pipeline blocked. Once delivered, rerun V3.75 and then V3.76.", ""])
    return "\n".join(lines)


def build_catalog(status: pd.DataFrame, next_commands: pd.DataFrame, config: LabelSourceIntakeConfig) -> str:
    blocked_checks = int(status["status"].astype(str).eq("blocked").sum()) if not status.empty else 0
    active_steps = int(next_commands["status"].astype(str).eq("active").sum()) if not next_commands.empty else 0
    return "\n".join(
        [
            "# A-share MARKET Label Source Intake Orchestrator V3.76",
            "",
            "## Dataset Decision",
            "",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            f"- Target source exists: `{config.target_source_path.exists()}`",
            f"- Blocked checks: `{blocked_checks}`",
            f"- Active next steps: `{active_steps}`",
            "- No labels, portfolios, or model promotion are produced.",
            "",
        ]
    )
