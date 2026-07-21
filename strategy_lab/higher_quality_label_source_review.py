"""Higher-quality MARKET label-source review for HIRSSM V3.73.

V3.73 is a governance layer between proxy-positive feature review and any
portfolio work. It checks whether V3.72 strict survivors have a compliant
total-return or explicitly adjusted MARKET label source. If the source is
missing or unvalidated, every survivor remains blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MARKET_SOURCE_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]

STRICT_SURVIVOR_STATUS = "strict_proxy_survivor_for_label_review"


@dataclass(frozen=True)
class HigherQualityLabelSourceReviewConfig:
    v3_72_manifest_path: Path
    v3_72_decision_path: Path
    v3_52_manifest_path: Path
    v3_52_candidate_assessment_path: Path
    v3_52_readiness_path: Path
    v3_53_manifest_path: Path
    v3_53_source_contract_path: Path
    v3_53_readiness_path: Path
    v3_54_manifest_path: Path
    v3_54_acquisition_routes_path: Path
    v3_54_provider_readiness_path: Path
    v3_54_readiness_path: Path
    target_source_path: Path
    template_source_path: Path
    price_proxy_label_path: Path
    output_dir: Path
    catalog_path: Path
    strict_survivor_status: str = STRICT_SURVIVOR_STATUS


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig", usecols=[0], low_memory=False)))
    except Exception:
        try:
            return int(len(pd.read_csv(path, encoding="utf-8-sig", low_memory=False)))
        except Exception:
            return 0


def _read_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
    except Exception:
        return []


def _manifest_passed(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status", "pass")).lower()
    self_check = bool(manifest.get("self_check_pass", manifest.get("acceptance_pass", False)))
    return self_check and status != "fail"


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _safe_text(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    if frame.empty:
        return ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|", "| " + " | ".join([""] * len(columns)) + " |"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).iterrows():
        values = [_safe_text(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def validate_inputs(
    v3_72_manifest: dict[str, Any],
    decisions: pd.DataFrame,
    v3_52_manifest: dict[str, Any],
    v3_52_readiness: pd.DataFrame,
    v3_53_manifest: dict[str, Any],
    v3_53_readiness: pd.DataFrame,
    v3_54_manifest: dict[str, Any],
    routes: pd.DataFrame,
    provider_readiness: pd.DataFrame,
    contract: pd.DataFrame,
    config: HigherQualityLabelSourceReviewConfig,
) -> pd.DataFrame:
    survivor_rows = select_strict_survivors(decisions, config).shape[0] if "strict_review_status" in decisions.columns else 0
    accepted_v3_52 = int(v3_52_manifest.get("accepted_market_label_source_count", 0) or 0)
    source_exists_v3_53 = bool(v3_53_manifest.get("source_exists", False))
    labels_produced_v3_53 = bool(v3_53_manifest.get("labels_produced", False))
    return pd.DataFrame(
        [
            {
                "check": "v3_72_manifest_passed",
                "status": _status(_manifest_passed(v3_72_manifest)),
                "detail": f"self_check={v3_72_manifest.get('self_check_pass')};status={v3_72_manifest.get('status')}",
            },
            {
                "check": "v3_72_strict_survivors_present",
                "status": _status(survivor_rows > 0),
                "detail": f"strict_survivor_rows={survivor_rows}",
            },
            {
                "check": "v3_52_prior_label_audit_loaded",
                "status": _status(_manifest_passed(v3_52_manifest)),
                "detail": f"accepted_sources={accepted_v3_52};self_check={v3_52_manifest.get('self_check_pass')}",
            },
            {
                "check": "v3_52_readiness_preserves_source_block",
                "status": _status(
                    bool(
                        v3_52_readiness.get("check", pd.Series(dtype=str))
                        .astype(str)
                        .eq("accepted_market_total_return_source_available")
                        .any()
                    )
                ),
                "detail": "prior audit readiness included accepted-source gate",
            },
            {
                "check": "v3_53_importer_loaded",
                "status": _status(_manifest_passed(v3_53_manifest)),
                "detail": f"source_exists={source_exists_v3_53};labels_produced={labels_produced_v3_53}",
            },
            {
                "check": "v3_53_contract_loaded",
                "status": _status(set(REQUIRED_MARKET_SOURCE_COLUMNS).issubset(set(contract.get("column", pd.Series(dtype=str)).astype(str)))),
                "detail": ",".join(REQUIRED_MARKET_SOURCE_COLUMNS),
            },
            {
                "check": "v3_54_acquirer_loaded",
                "status": _status(bool(v3_54_manifest.get("self_check_pass", False))),
                "detail": f"ready_routes={','.join(str(x) for x in v3_54_manifest.get('ready_routes', []))}",
            },
            {
                "check": "v3_54_routes_loaded",
                "status": _status(not routes.empty),
                "detail": f"routes={len(routes)}",
            },
            {
                "check": "v3_54_provider_readiness_loaded",
                "status": _status(not provider_readiness.empty),
                "detail": f"provider_checks={len(provider_readiness)}",
            },
            {
                "check": "target_source_file_exists",
                "status": _status(config.target_source_path.exists(), "blocked"),
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "template_is_not_treated_as_source",
                "status": _status(config.template_source_path.exists()),
                "detail": "template exists only as contract example",
            },
        ]
    )


def select_strict_survivors(decisions: pd.DataFrame, config: HigherQualityLabelSourceReviewConfig) -> pd.DataFrame:
    if decisions.empty or "strict_review_status" not in decisions.columns:
        return decisions.iloc[0:0].copy()
    survivors = decisions.loc[decisions["strict_review_status"].astype(str).eq(config.strict_survivor_status)].copy()
    return survivors.sort_values(["horizon", "source_family", "feature_id"]).reset_index(drop=True)


def inspect_target_source(path: Path, v3_53_manifest: dict[str, Any]) -> dict[str, Any]:
    exists = path.exists()
    header = _read_header(path)
    missing = sorted(set(REQUIRED_MARKET_SOURCE_COLUMNS).difference(header))
    row_count = _csv_row_count(path)
    source_validation_passed = bool(v3_53_manifest.get("source_validation_passed", False))
    labels_produced = bool(v3_53_manifest.get("labels_produced", False))
    validation_ready = bool(exists and not missing)
    accepted = bool(source_validation_passed and labels_produced)
    if not exists:
        decision = "missing_required_source"
        reason = "target CSV is absent"
    elif missing:
        decision = "blocked_contract_mismatch"
        reason = "missing required columns: " + ",".join(missing)
    elif not source_validation_passed:
        decision = "candidate_needs_v3_53_revalidation"
        reason = "contract columns present, but V3.53 has not validated this source"
    else:
        decision = "validated_by_v3_53"
        reason = "source validation passed in V3.53"
    return {
        "exists": exists,
        "row_count": row_count,
        "columns": "|".join(header),
        "missing_columns": ",".join(missing),
        "validation_ready_for_v3_53": validation_ready,
        "higher_quality_label_source_accepted": accepted,
        "source_validation_passed_by_v3_53": source_validation_passed,
        "labels_produced_by_v3_53": labels_produced,
        "decision": decision,
        "reason": reason,
    }


def build_label_source_inventory(
    config: HigherQualityLabelSourceReviewConfig,
    v3_52_manifest: dict[str, Any],
    v3_52_assessment: pd.DataFrame,
    v3_53_manifest: dict[str, Any],
    v3_54_manifest: dict[str, Any],
    root: Path,
) -> pd.DataFrame:
    target = inspect_target_source(config.target_source_path, v3_53_manifest)
    template_cols = _read_header(config.template_source_path)
    price_proxy_cols = _read_header(config.price_proxy_label_path)
    price_proxy_rows = _csv_row_count(config.price_proxy_label_path)
    accepted_sources = int(v3_52_manifest.get("accepted_market_label_source_count", 0) or 0)
    rejected_price_only = 0
    audited_files = int(v3_52_manifest.get("candidate_files_audited", 0) or 0)
    if not v3_52_assessment.empty and "decision" in v3_52_assessment.columns:
        rejected_price_only = int(v3_52_assessment["decision"].astype(str).eq("rejected_price_only").sum())
    rows = [
        {
            "source_id": "target_market_total_return_index_csv",
            "path_or_origin": _rel(config.target_source_path, root),
            "source_family": "manual_or_vendor_total_return_source",
            "exists_or_rows": int(target["row_count"]),
            "status": "ready_for_v3_53" if target["validation_ready_for_v3_53"] else "blocked",
            "decision": target["decision"],
            "higher_quality_label_source_accepted": bool(target["higher_quality_label_source_accepted"]),
            "validation_ready_for_v3_53": bool(target["validation_ready_for_v3_53"]),
            "reason": target["reason"],
            "next_action": "rerun_v3_53_importer" if target["validation_ready_for_v3_53"] else "provide_compliant_total_return_source_csv",
        },
        {
            "source_id": "market_total_return_index_template_csv",
            "path_or_origin": _rel(config.template_source_path, root),
            "source_family": "contract_template_only",
            "exists_or_rows": int(_csv_row_count(config.template_source_path)),
            "status": "not_a_source",
            "decision": "template_only",
            "higher_quality_label_source_accepted": False,
            "validation_ready_for_v3_53": False,
            "reason": "sample rows are documentation, not investable label data",
            "next_action": "replace_with_real_vendor_or_certified_source_file",
        },
        {
            "source_id": "v3_59_market_price_proxy_forward_labels",
            "path_or_origin": _rel(config.price_proxy_label_path, root),
            "source_family": "price_index_proxy_labels",
            "exists_or_rows": price_proxy_rows,
            "status": "research_only",
            "decision": "rejected_as_higher_quality_label_source",
            "higher_quality_label_source_accepted": False,
            "validation_ready_for_v3_53": False,
            "reason": "price-index proxy excludes dividend/total-return basis",
            "next_action": "do_not_use_for_investable_performance_or_default_model",
        },
        {
            "source_id": "v3_52_prior_market_source_audit",
            "path_or_origin": _rel(config.v3_52_candidate_assessment_path, root),
            "source_family": "prior_local_source_audit",
            "exists_or_rows": audited_files,
            "status": "blocked",
            "decision": "accepted_sources_zero" if accepted_sources == 0 else "accepted_sources_present",
            "higher_quality_label_source_accepted": accepted_sources > 0,
            "validation_ready_for_v3_53": False,
            "reason": f"accepted_sources={accepted_sources};rejected_price_only={rejected_price_only}",
            "next_action": "use_audited_accepted_source_or_acquire_new_source",
        },
        {
            "source_id": "v3_54_acquisition_router",
            "path_or_origin": _rel(config.v3_54_acquisition_routes_path, root),
            "source_family": "provider_route_inventory",
            "exists_or_rows": len(v3_54_manifest.get("ready_routes", [])),
            "status": "ready" if v3_54_manifest.get("ready_routes") else "blocked",
            "decision": str(v3_54_manifest.get("data_decision", "")),
            "higher_quality_label_source_accepted": False,
            "validation_ready_for_v3_53": bool(v3_54_manifest.get("target_exists_after", False)),
            "reason": f"ready_routes={','.join(str(x) for x in v3_54_manifest.get('ready_routes', []))}",
            "next_action": "execute_ready_route_or_manual_vendor_file",
        },
    ]
    if target["exists"]:
        rows[0]["reason"] = f"{rows[0]['reason']};rows={target['row_count']};missing={target['missing_columns']}"
    else:
        rows[0]["reason"] = f"{rows[0]['reason']};required_columns={','.join(REQUIRED_MARKET_SOURCE_COLUMNS)}"
    if template_cols:
        rows[1]["reason"] = f"{rows[1]['reason']};columns={'/'.join(template_cols)}"
    if price_proxy_cols:
        rows[2]["reason"] = f"{rows[2]['reason']};columns={'/'.join(price_proxy_cols[:12])}"
    return pd.DataFrame(rows)


def build_provider_route_review(routes: pd.DataFrame, provider_readiness: pd.DataFrame) -> pd.DataFrame:
    if routes.empty:
        return pd.DataFrame(
            columns=[
                "route_id",
                "provider",
                "route_type",
                "route_ready",
                "will_execute",
                "route_status",
                "blocker",
                "provider_checks_ready",
                "current_decision",
            ]
        )
    provider_ok: dict[str, bool] = {}
    if not provider_readiness.empty and {"provider", "ready"}.issubset(provider_readiness.columns):
        ready = provider_readiness.copy()
        ready["ready"] = _bool_series(ready["ready"])
        provider_ok = ready.groupby("provider")["ready"].all().to_dict()
    out = routes.copy()
    out["route_ready"] = _bool_series(out.get("ready", pd.Series(False, index=out.index)))
    out["will_execute"] = _bool_series(out.get("will_execute", pd.Series(False, index=out.index)))
    out["route_status"] = out.get("status", pd.Series("", index=out.index)).astype(str)
    out["provider_checks_ready"] = out.get("provider", pd.Series("", index=out.index)).map(provider_ok).fillna(False).astype(bool)
    out["current_decision"] = out.apply(
        lambda row: "available_route" if bool(row["route_ready"]) else f"blocked:{row['route_status']}",
        axis=1,
    )
    columns = [
        "priority",
        "route_id",
        "provider",
        "route_type",
        "route_ready",
        "will_execute",
        "route_status",
        "blocker",
        "acceptance_note",
        "provider_checks_ready",
        "current_decision",
    ]
    return out.loc[:, [col for col in columns if col in out.columns]].sort_values([col for col in ["priority", "route_id"] if col in out.columns]).reset_index(drop=True)


def build_required_source_contract(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(
            [
                {
                    "column": column,
                    "required": True,
                    "type": "",
                    "rule": "required by V3.73 because V3.53 contract was not loaded",
                    "example": "",
                    "v3_73_enforcement": "blocking",
                }
                for column in REQUIRED_MARKET_SOURCE_COLUMNS
            ]
        )
    out = contract.copy()
    out["v3_73_enforcement"] = out["column"].astype(str).apply(lambda x: "blocking" if x in REQUIRED_MARKET_SOURCE_COLUMNS else "optional")
    out["v3_73_note"] = out["column"].astype(str).apply(
        lambda x: "must be present before any official label validation" if x in REQUIRED_MARKET_SOURCE_COLUMNS else "not required by V3.73"
    )
    return out


def build_survivor_label_review_queue(survivors: pd.DataFrame, source_accepted: bool, source_ready_for_v3_53: bool) -> pd.DataFrame:
    queue = survivors.copy()
    if queue.empty:
        return queue
    if source_accepted:
        status = "ready_for_total_return_label_validation"
        action = "run_next_validation_against_v3_53_labels"
        reason = "V3.53 accepted source and produced labels"
    elif source_ready_for_v3_53:
        status = "blocked_pending_v3_53_revalidation"
        action = "rerun_v3_53_importer_then_repeat_v3_73"
        reason = "source file exists with required columns but has not produced validated labels"
    else:
        status = "blocked_missing_total_return_source"
        action = "provide_compliant_total_return_source_then_rerun_v3_53"
        reason = "no compliant official/adjusted MARKET label source exists"
    queue["source_review_version"] = "V3.73"
    queue["higher_quality_label_review_status"] = status
    queue["required_next_action"] = action
    queue["source_review_reason"] = reason
    queue["official_total_return_evidence"] = bool(source_accepted)
    queue["portfolio_backtest_allowed"] = False
    queue["default_model_allowed"] = False
    return queue


def build_no_label_guard(source_accepted: bool, labels_produced: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "higher_quality_label_source_review",
                "produced": True,
                "blocked": False,
                "reason": "V3.73 governance review produced",
            },
            {
                "result_type": "accepted_total_return_source",
                "produced": source_accepted,
                "blocked": not source_accepted,
                "reason": "requires V3.53 source validation and label production",
            },
            {
                "result_type": "forward_total_return_labels",
                "produced": labels_produced,
                "blocked": not labels_produced,
                "reason": "labels must be built by V3.53 from accepted source",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.73 cannot run a portfolio harness",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "strict survivors have no accepted investable label evidence yet",
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    inventory: pd.DataFrame,
    provider_routes: pd.DataFrame,
    contract: pd.DataFrame,
    queue: pd.DataFrame,
    guard: pd.DataFrame,
    source_accepted: bool,
) -> pd.DataFrame:
    blocked_rows = int(queue.get("higher_quality_label_review_status", pd.Series(dtype=str)).astype(str).str.startswith("blocked").sum()) if not queue.empty else 0
    promoted_rows = 0
    if not queue.empty:
        promoted_rows = int(_bool_series(queue.get("portfolio_backtest_allowed", pd.Series(False, index=queue.index))).sum())
        promoted_rows += int(_bool_series(queue.get("default_model_allowed", pd.Series(False, index=queue.index))).sum())
    required_contract_loaded = set(REQUIRED_MARKET_SOURCE_COLUMNS).issubset(set(contract.get("column", pd.Series(dtype=str)).astype(str)))
    source_inventory_has_target = bool(inventory.get("source_id", pd.Series(dtype=str)).astype(str).eq("target_market_total_return_index_csv").any())
    blocked_outputs = guard.loc[guard["result_type"].isin(["portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any()
    return pd.DataFrame(
        [
            {
                "check": "input_manifests_and_contract_loaded",
                "status": _status(not input_checks.loc[input_checks["status"].eq("fail")].shape[0]),
                "detail": ";".join(input_checks.loc[input_checks["status"].eq("fail"), "check"].astype(str)),
            },
            {
                "check": "strict_survivor_queue_built",
                "status": _status(len(queue) > 0),
                "detail": f"queue_rows={len(queue)}",
            },
            {
                "check": "target_source_inventory_recorded",
                "status": _status(source_inventory_has_target),
                "detail": "target_market_total_return_index_csv",
            },
            {
                "check": "required_source_contract_recorded",
                "status": _status(required_contract_loaded),
                "detail": ",".join(REQUIRED_MARKET_SOURCE_COLUMNS),
            },
            {
                "check": "survivors_blocked_when_source_unaccepted",
                "status": _status(source_accepted or blocked_rows == len(queue)),
                "detail": f"blocked_rows={blocked_rows};queue_rows={len(queue)};source_accepted={source_accepted}",
            },
            {
                "check": "no_survivor_promoted",
                "status": _status(promoted_rows == 0),
                "detail": f"promoted_permission_rows={promoted_rows}",
            },
            {
                "check": "provider_routes_reviewed",
                "status": _status(not provider_routes.empty),
                "detail": f"routes={len(provider_routes)}",
            },
            {
                "check": "no_portfolio_or_model_outputs",
                "status": _status(not blocked_outputs),
                "detail": "V3.73 writes only governance artifacts",
            },
        ]
    )


def build_report(
    inventory: pd.DataFrame,
    provider_routes: pd.DataFrame,
    contract: pd.DataFrame,
    queue: pd.DataFrame,
    guard: pd.DataFrame,
    acceptance: pd.DataFrame,
    input_checks: pd.DataFrame,
    config: HigherQualityLabelSourceReviewConfig,
) -> str:
    source_accepted = bool(inventory["higher_quality_label_source_accepted"].astype(bool).any()) if not inventory.empty else False
    blocked_queue = int(queue.get("higher_quality_label_review_status", pd.Series(dtype=str)).astype(str).str.startswith("blocked").sum()) if not queue.empty else 0
    ready_routes = int(provider_routes.get("route_ready", pd.Series(dtype=bool)).astype(bool).sum()) if not provider_routes.empty else 0
    lines = [
        "# V3.73 Higher-Quality Label Source Review",
        "",
        "## Decision",
        "",
        "- V3.73 reviews whether V3.72 strict proxy survivors may move beyond proxy-label research.",
        "- Current decision: all strict survivors stay blocked until a compliant MARKET total-return or explicitly adjusted source is validated by V3.53.",
        "- This run does not create labels, portfolio results, or model promotion evidence.",
        "",
        "## Key Metrics",
        "",
        f"- Strict survivor rows reviewed: `{len(queue)}`",
        f"- Blocked survivor rows: `{blocked_queue}`",
        f"- Accepted higher-quality label sources: `{int(source_accepted)}`",
        f"- Ready provider/acquisition routes: `{ready_routes}`",
        f"- Required target source path: `{_workspace_suffix(config.target_source_path)}`",
        "",
        "## Label Source Inventory",
        "",
    ]
    lines.extend(markdown_table(inventory, ["source_id", "status", "decision", "higher_quality_label_source_accepted", "reason", "next_action"], 20))
    lines.extend(["", "## Provider Routes", ""])
    lines.extend(markdown_table(provider_routes, ["priority", "route_id", "provider", "route_ready", "route_status", "blocker", "current_decision"], 20))
    lines.extend(["", "## Required Source Contract", ""])
    lines.extend(markdown_table(contract, ["column", "required", "type", "rule", "v3_73_enforcement"], 20))
    lines.extend(["", "## Strict Survivor Queue", ""])
    preview_cols = [
        "feature_id",
        "source_family",
        "horizon",
        "review_score",
        "higher_quality_label_review_status",
        "required_next_action",
    ]
    lines.extend(markdown_table(queue, preview_cols, 30))
    lines.extend(["", "## No-Label Guard", ""])
    lines.extend(markdown_table(guard, ["result_type", "produced", "blocked", "reason"], 20))
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], 30))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "1. Place a compliant file at `data_raw/market_labels/market_total_return_index.csv` with the required contract columns.",
            "2. Rerun V3.53 to validate the source and generate forward total-return labels.",
            "3. Rerun this V3.73 gate. If accepted, run V3.74 label validation for the 20 strict survivors only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(inventory: pd.DataFrame, queue: pd.DataFrame, provider_routes: pd.DataFrame, config: HigherQualityLabelSourceReviewConfig) -> str:
    accepted_sources = int(inventory["higher_quality_label_source_accepted"].astype(bool).sum()) if not inventory.empty else 0
    blocked_rows = int(queue.get("higher_quality_label_review_status", pd.Series(dtype=str)).astype(str).str.startswith("blocked").sum()) if not queue.empty else 0
    ready_routes = int(provider_routes.get("route_ready", pd.Series(dtype=bool)).astype(bool).sum()) if not provider_routes.empty else 0
    return "\n".join(
        [
            "# A-share Higher-Quality Label Source Review V3.73",
            "",
            "## Dataset Decision",
            "",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            f"- Accepted higher-quality label sources: `{accepted_sources}`",
            f"- Strict survivor rows reviewed: `{len(queue)}`",
            f"- Strict survivor rows blocked: `{blocked_rows}`",
            f"- Ready provider/acquisition routes: `{ready_routes}`",
            "- Portfolio validation remains blocked until V3.53 validates a compliant source.",
            "",
        ]
    )
