"""Independent MARKET label-source discovery for HIRSSM V3.77.

The module ranks external routes for acquiring a governed daily MARKET
total-return index source. It only produces procurement evidence and routing
decisions; it must not write labels, run validation, or produce portfolio
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceDiscoveryConfig:
    v3_76_manifest_path: Path
    v3_76_next_commands_path: Path
    v3_75_requirements_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    required_start_date: str
    required_end_date: str
    source_candidates: tuple[dict[str, Any], ...]
    score_weights: dict[str, float]


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_row(row: dict[str, Any], weights: dict[str, float]) -> float:
    bool_keys = [
        "official_calculation",
        "target_index_match",
        "daily_history",
        "total_return_basis",
        "point_in_time_auditability",
        "machine_readable",
        "license_clear_for_research",
        "local_access_ready",
        "cross_source_validation_value",
    ]
    score = 0.0
    for key in bool_keys:
        score += weights.get(key, 0.0) * float(_bool(row.get(key, False)))
    score += weights.get("history_depth_score", 0.0) * max(0.0, min(1.0, _float(row.get("history_depth_score"))))
    score += weights.get("access_practicality_score", 0.0) * max(0.0, min(1.0, _float(row.get("access_practicality_score"))))
    score -= weights.get("legal_or_terms_penalty", 0.0) * max(0.0, min(1.0, _float(row.get("legal_or_terms_penalty"))))
    score -= weights.get("proxy_penalty", 0.0) * max(0.0, min(1.0, _float(row.get("proxy_penalty"))))
    return round(max(0.0, min(100.0, score)), 2)


def _decision(row: pd.Series) -> str:
    if not _bool(row.get("total_return_basis")):
        return "reject_as_final_label_price_only"
    if not _bool(row.get("target_index_match")):
        return "reject_as_final_label_wrong_index"
    if not _bool(row.get("daily_history")):
        return "observe_or_secondary_not_daily"
    if row.get("route_type") == "official_index_provider":
        return "primary_procurement_route"
    if row.get("route_type") == "licensed_terminal_vendor":
        return "secondary_procurement_route"
    if row.get("route_type") == "reconstruction":
        return "research_reconstruction_route"
    if row.get("route_type") == "public_web":
        return "manual_secondary_validation_only"
    return "manual_review"


def build_source_candidates(config: SourceDiscoveryConfig) -> pd.DataFrame:
    rows = []
    for raw in config.source_candidates:
        row = dict(raw)
        row["target_source_exists_now"] = config.target_source_path.exists()
        row["score"] = _score_row(row, config.score_weights)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["decision"] = frame.apply(_decision, axis=1)
    frame["rank"] = frame["score"].rank(method="first", ascending=False).astype(int)
    frame = frame.sort_values(["rank", "source_id"]).reset_index(drop=True)
    preferred = frame["decision"].isin(["primary_procurement_route", "secondary_procurement_route"])
    frame["procurement_priority"] = "not_for_procurement"
    frame.loc[preferred, "procurement_priority"] = frame.loc[preferred, "rank"].map(lambda x: f"P{int(x)}")
    return frame


def build_due_diligence_matrix(candidates: pd.DataFrame, config: SourceDiscoveryConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        source_id = str(row.get("source_id"))
        rows.extend(
            [
                {
                    "source_id": source_id,
                    "check": "daily_total_return_series_available",
                    "required_answer": "yes, daily close level from at least required_start_date",
                    "current_status": "unknown" if row.get("decision") in {"primary_procurement_route", "secondary_procurement_route"} else "not_sufficient",
                    "evidence_needed": f"sample CSV covering {config.required_start_date} to {config.required_end_date}",
                },
                {
                    "source_id": source_id,
                    "check": "methodology_and_code_match",
                    "required_answer": "000985 or documented MARKET total-return equivalent",
                    "current_status": "candidate_claim" if _bool(row.get("target_index_match")) else "fail",
                    "evidence_needed": "provider metadata with index code, name, currency, and return basis",
                },
                {
                    "source_id": source_id,
                    "check": "point_in_time_use_allowed",
                    "required_answer": "historical values are final daily publications or versioned corrections",
                    "current_status": "candidate_claim" if _bool(row.get("point_in_time_auditability")) else "needs_contract_review",
                    "evidence_needed": "data dictionary with publish time and correction policy",
                },
                {
                    "source_id": source_id,
                    "check": "research_license",
                    "required_answer": "permitted for internal research and model validation",
                    "current_status": "candidate_claim" if _bool(row.get("license_clear_for_research")) else "needs_legal_review",
                    "evidence_needed": "license or terms text allowing local storage and derived research",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_acquisition_route_decision(candidates: pd.DataFrame, config: SourceDiscoveryConfig) -> pd.DataFrame:
    target_missing = not config.target_source_path.exists()
    rows = []
    primary = candidates.loc[candidates["decision"].eq("primary_procurement_route")].head(1)
    secondary = candidates.loc[candidates["decision"].eq("secondary_procurement_route")].head(2)
    reconstruction = candidates.loc[candidates["decision"].eq("research_reconstruction_route")].head(1)
    public_web = candidates.loc[candidates["decision"].eq("manual_secondary_validation_only")].head(2)

    def add(step: int, source: pd.Series | None, action: str, status: str, may_execute: bool, reason: str) -> None:
        rows.append(
            {
                "step_order": step,
                "source_id": "" if source is None else str(source.get("source_id")),
                "action": action,
                "status": status,
                "may_execute_now": may_execute,
                "reason": reason,
            }
        )

    add(
        1,
        primary.iloc[0] if not primary.empty else None,
        "request official/provider daily total-return CSV matching V3.75 vendor template",
        "active" if target_missing and not primary.empty else "done" if not target_missing else "blocked",
        False,
        "best route for final labels but requires external delivery",
    )
    for offset, (_, row) in enumerate(secondary.iterrows(), start=2):
        add(
            offset,
            row,
            "ask licensed terminal/vendor for the same field and sample file",
            "pending" if target_missing else "done",
            False,
            "backup route if official provider delivery is slow or unavailable",
        )
    add(
        4,
        reconstruction.iloc[0] if not reconstruction.empty else None,
        "design reconstruction feasibility test using historical constituents, weights, dividends, and corporate actions",
        "pending",
        False,
        "research fallback only; not accepted as official label without reconciliation",
    )
    if not public_web.empty:
        add(
            5,
            public_web.iloc[0],
            "use public web source only as manual cross-check after license review",
            "blocked",
            False,
            "terms and reproducibility are insufficient for default label generation",
        )
    add(
        6,
        None,
        "place approved CSV at data_raw/market_labels/market_total_return_index.csv and rerun V3.75 then V3.76",
        "blocked" if target_missing else "active",
        False,
        "V3.53 remains blocked until the target file passes the contract",
    )
    return pd.DataFrame(rows)


def build_provider_questionnaire(candidates: pd.DataFrame, config: SourceDiscoveryConfig) -> str:
    priority = candidates.loc[candidates["decision"].isin(["primary_procurement_route", "secondary_procurement_route"])]
    source_lines = "\n".join(f"- {row.source_id}: {row.provider_name}" for row in priority.itertuples())
    if not source_lines:
        source_lines = "- No procurement-ready source identified."
    lines = [
        "# V3.77 Provider Questionnaire",
        "",
        "## Target File",
        "",
        f"- Path after delivery: `{_workspace_suffix(config.target_source_path)}`",
        "- Required columns: `trade_date,index_code,index_name,close,total_return_close,return_basis,source,available_date`",
        f"- Required history: `{config.required_start_date}` to `{config.required_end_date}` or later",
        "",
        "## Priority Sources",
        "",
        source_lines,
        "",
        "## Questions",
        "",
        "1. Can you provide daily close levels for the 000985 MARKET total-return equivalent, not price-only levels?",
        "2. What is the exact index code, return basis, currency, base date, and base point?",
        "3. Are historical values final as of each trade date, or can they be restated later?",
        "4. What is the publication time and timezone for each daily value?",
        "5. Does the license permit local storage, internal research, model validation, and derived feature/label files?",
        "6. Can you provide a 20-row sample CSV and a data dictionary before full delivery?",
        "7. If 000985 total-return is unavailable, which official broad A-share total-return alternatives are available?",
        "",
    ]
    return "\n".join(lines)


def build_evidence_sources(candidates: pd.DataFrame) -> str:
    lines = [
        "# V3.77 Source Evidence Register",
        "",
        "This file stores URLs and manual verification notes used by the discovery run.",
        "",
        "| source_id | provider | url | evidence_note |",
        "|---|---|---|---|",
    ]
    for row in candidates.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(getattr(row, "source_id", "")),
                    str(getattr(row, "provider_name", "")).replace("|", "/"),
                    str(getattr(row, "source_url", "")).replace("|", "/"),
                    str(getattr(row, "evidence_note", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_no_execution_guard(config: SourceDiscoveryConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "source_discovery",
                "produced": True,
                "blocked": False,
                "reason": "V3.77 creates source discovery and route evidence.",
            },
            {
                "result_type": "target_csv_write",
                "produced": False,
                "blocked": True,
                "reason": "V3.77 must not write the official target CSV.",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": True,
                "reason": "Labels remain blocked until an approved source file passes V3.75 and V3.76.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No validated label file exists in this run.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Source discovery is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(
    candidates: pd.DataFrame,
    due_diligence: pd.DataFrame,
    route_decision: pd.DataFrame,
    guard: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "source_candidates_written",
                "status": "pass" if len(candidates) >= 8 else "fail",
                "detail": f"rows={len(candidates)}",
            },
            {
                "check": "primary_or_secondary_route_exists",
                "status": "pass" if candidates["decision"].isin(["primary_procurement_route", "secondary_procurement_route"]).any() else "fail",
                "detail": ",".join(candidates.loc[candidates["decision"].isin(["primary_procurement_route", "secondary_procurement_route"]), "source_id"].astype(str).head(5)),
            },
            {
                "check": "price_only_sources_rejected",
                "status": "pass" if candidates.loc[~candidates["total_return_basis"].astype(bool), "decision"].astype(str).str.contains("reject").all() else "fail",
                "detail": f"price_only_rows={int((~candidates['total_return_basis'].astype(bool)).sum())}",
            },
            {
                "check": "due_diligence_matrix_written",
                "status": "pass" if len(due_diligence) >= len(candidates) * 3 else "fail",
                "detail": f"rows={len(due_diligence)}",
            },
            {
                "check": "route_decision_blocks_v3_53",
                "status": "pass" if not route_decision["may_execute_now"].astype(bool).any() else "fail",
                "detail": "no acquisition command is executable by the script",
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["target_csv_write", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "discovery only",
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
    candidates: pd.DataFrame,
    due_diligence: pd.DataFrame,
    route_decision: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: SourceDiscoveryConfig,
) -> str:
    best = candidates.iloc[0] if not candidates.empty else pd.Series(dtype=object)
    lines = [
        "# V3.77 Independent MARKET Label Source Discovery",
        "",
        "## Decision",
        "",
        "- V3.77 does not acquire or write the target label file.",
        "- It ranks independent routes for a daily MARKET total-return source and keeps V3.53 blocked.",
        "- Price-only index history can support diagnostics, but it is rejected as a final MARKET label source.",
        "",
        "## Key Metrics",
        "",
        f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        f"- Candidate sources reviewed: `{len(candidates)}`",
        f"- Top source: `{best.get('source_id', '')}`",
        f"- Top source decision: `{best.get('decision', '')}`",
        "",
        "## Ranked Sources",
        "",
    ]
    lines.extend(markdown_table(candidates, ["rank", "source_id", "provider_name", "route_type", "score", "decision", "procurement_priority"], 20))
    lines.extend(["", "## Route Decision", ""])
    lines.extend(markdown_table(route_decision, ["step_order", "source_id", "action", "status", "may_execute_now", "reason"], 20))
    lines.extend(["", "## Due Diligence Sample", ""])
    lines.extend(markdown_table(due_diligence, ["source_id", "check", "current_status", "evidence_needed"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Send the provider questionnaire to the primary and secondary procurement routes.",
            "- When a licensed sample arrives, place it in a temporary review location first, then rerun V3.75 before writing the final target source.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(candidates: pd.DataFrame, route_decision: pd.DataFrame, config: SourceDiscoveryConfig) -> str:
    primary_count = int(candidates["decision"].isin(["primary_procurement_route", "secondary_procurement_route"]).sum()) if not candidates.empty else 0
    blocked_steps = int(route_decision["status"].astype(str).eq("blocked").sum()) if not route_decision.empty else 0
    return "\n".join(
        [
            "# A-share MARKET Label Source Discovery V3.77",
            "",
            "## Dataset Decision",
            "",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            f"- Target source exists: `{config.target_source_path.exists()}`",
            f"- Candidate sources reviewed: `{len(candidates)}`",
            f"- Procurement-ready routes: `{primary_count}`",
            f"- Blocked route steps: `{blocked_steps}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
