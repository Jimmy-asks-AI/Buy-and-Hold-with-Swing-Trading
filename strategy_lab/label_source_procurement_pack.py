"""Manual/procured MARKET label-source package for HIRSSM V3.75.

V3.75 converts the V3.74 live-acquisition blockage into a concrete procurement
and import-validation package. It can validate a candidate CSV against the
V3.53 source contract, but it does not write the official target source, build
labels, run validation, or promote a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]

FORBIDDEN_LINEAGE_TERMS = {
    "backtest",
    "benchmark",
    "strategy_nav",
    "portfolio_nav",
    "raw_close",
    "none_raw",
    "unadjusted",
    "price_only",
}


@dataclass(frozen=True)
class LabelSourceProcurementConfig:
    v3_74_manifest_path: Path
    v3_74_manual_interfaces_path: Path
    v3_74_candidate_path: Path
    source_contract_path: Path
    signal_panel_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    required_start_date: str
    required_end_date: str
    min_source_rows: int
    min_signal_coverage_ratio: float
    horizons: tuple[int, ...]
    accepted_source_bases: tuple[str, ...]


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    text = text.str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def format_date(values: pd.Series) -> pd.Series:
    return normalize_date(values).dt.strftime("%Y%m%d")


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
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    for _, row in frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).iterrows():
        lines.append("| " + " | ".join(_safe_text(row.get(col, "")) for col in columns) + " |")
    return lines


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_procurement_requirements(config: LabelSourceProcurementConfig, v3_74_manifest: dict[str, Any]) -> pd.DataFrame:
    current_rows = int(v3_74_manifest.get("metrics", {}).get("joinquant_candidate_rows", 0) or 0)
    current_min_coverage = float(v3_74_manifest.get("metrics", {}).get("minimum_signal_coverage_ratio", 0.0) or 0.0)
    required_fields = ",".join(REQUIRED_SOURCE_COLUMNS)
    return pd.DataFrame(
        [
            {
                "requirement_id": "certified_broad_market_total_return_index",
                "priority": 1,
                "acceptable_basis": "certified_total_return",
                "preferred_asset_or_index": "MARKET or 000985/000300 equivalent",
                "required_history": f"{config.required_start_date} to {config.required_end_date}",
                "required_fields": required_fields,
                "minimum_rows": config.min_source_rows,
                "minimum_signal_coverage_ratio": config.min_signal_coverage_ratio,
                "pit_requirement": "available_date must be the publication/known date and not before date",
                "rejected_substitutes": "strategy_nav;benchmark_nav;raw_close;price_only_index_close;current_snapshot",
                "current_gap": f"no accepted source; JoinQuant candidate rows={current_rows};min_coverage={current_min_coverage:.4f}",
                "next_action": "request/export vendor CSV and run V3.75 validator, then V3.53",
            },
            {
                "requirement_id": "explicit_long_history_adjusted_market_proxy",
                "priority": 2,
                "acceptable_basis": "explicit_adjusted_market_proxy",
                "preferred_asset_or_index": "MARKET or approved broad index",
                "required_history": f"{config.required_start_date} to {config.required_end_date}",
                "required_fields": required_fields,
                "minimum_rows": config.min_source_rows,
                "minimum_signal_coverage_ratio": config.min_signal_coverage_ratio,
                "pit_requirement": "adjustment method and available_date must be documented",
                "rejected_substitutes": "unadjusted close;short trial window;unknown adjustment rule",
                "current_gap": "can be used only after explicit approval as adjusted proxy, not as certified total return",
                "next_action": "upgrade provider permission or vendor export; keep basis explicit",
            },
            {
                "requirement_id": "provider_api_or_export_lineage",
                "priority": 3,
                "acceptable_basis": "auditable_provider_file_or_endpoint",
                "preferred_asset_or_index": "MARKET",
                "required_history": f"{config.required_start_date} to {config.required_end_date}",
                "required_fields": required_fields,
                "minimum_rows": config.min_source_rows,
                "minimum_signal_coverage_ratio": config.min_signal_coverage_ratio,
                "pit_requirement": "data_source and source_vintage must identify provider and export batch",
                "rejected_substitutes": "hand-edited series without lineage",
                "current_gap": "known Tushare/JQ trial permissions are insufficient for full-history source",
                "next_action": "record provider endpoint, permission tier, and export vintage",
            },
        ]
    )


def build_source_contract(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        out = pd.DataFrame(
            [
                {
                    "column": column,
                    "required": True,
                    "type": "",
                    "rule": "required by V3.53/V3.75",
                    "example": "",
                }
                for column in REQUIRED_SOURCE_COLUMNS
            ]
        )
    else:
        out = contract.copy()
    out["v3_75_enforcement"] = out["column"].astype(str).apply(lambda x: "blocking" if x in REQUIRED_SOURCE_COLUMNS else "optional")
    out["import_stage"] = out["column"].astype(str).apply(lambda x: "source_csv_precheck" if x in REQUIRED_SOURCE_COLUMNS else "metadata")
    return out


def build_vendor_request_markdown(requirements: pd.DataFrame, config: LabelSourceProcurementConfig) -> str:
    lines = [
        "# MARKET Total-Return / Adjusted Source Vendor Request",
        "",
        "Please provide a daily A-share broad-market total-return index series or an explicitly adjusted broad-market proxy for quantitative research validation.",
        "",
        "## Required CSV Path After Delivery",
        "",
        "`data_raw/market_labels/market_total_return_index.csv`",
        "",
        "## Required Columns",
        "",
        "`date, asset_or_index, total_return_index_or_adjusted_close, available_date, data_source, source_vintage`",
        "",
        "## Required History",
        "",
        f"- Start: `{config.required_start_date}`",
        f"- End: `{config.required_end_date}`",
        f"- Minimum rows: `{config.min_source_rows}`",
        "",
        "## Accepted Basis",
        "",
        "- `certified_total_return`: official full-return/total-return index level.",
        "- `explicit_adjusted_market_proxy`: adjusted broad-market close only if the adjustment method is documented and approved.",
        "",
        "## Rejected Substitutes",
        "",
        "- Strategy NAV or benchmark NAV.",
        "- Price-only index close or raw/unadjusted close.",
        "- Current snapshot valuation data.",
        "- A short trial window that cannot cover the signal panel.",
        "",
        "## Machine-Readable Requirements",
        "",
    ]
    lines.extend(markdown_table(requirements, ["requirement_id", "priority", "acceptable_basis", "required_history", "minimum_rows", "current_gap"], 20))
    lines.extend(["", "After delivery, run V3.75 then V3.53. Do not run portfolio validation until V3.53 labels pass coverage gates.", ""])
    return "\n".join(lines)


def build_source_template(config: LabelSourceProcurementConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "20000104",
                "asset_or_index": "MARKET",
                "total_return_index_or_adjusted_close": 1000.0,
                "available_date": "20000105",
                "data_source": "vendor.total_return_index_or_explicit_adjusted_proxy",
                "source_vintage": "vendor_export_YYYYMMDD",
            },
            {
                "date": "20000105",
                "asset_or_index": "MARKET",
                "total_return_index_or_adjusted_close": 1001.5,
                "available_date": "20000106",
                "data_source": "vendor.total_return_index_or_explicit_adjusted_proxy",
                "source_vintage": "vendor_export_YYYYMMDD",
            },
        ]
    )


def validate_source_candidate(candidate: pd.DataFrame, config: LabelSourceProcurementConfig, source_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(candidate.columns))
    rows.append(
        {
            "source_name": source_name,
            "check": "required_columns_present",
            "status": _status(not missing, "blocked"),
            "detail": ",".join(missing),
        }
    )
    if missing or candidate.empty:
        if candidate.empty:
            rows.append(
                {
                    "source_name": source_name,
                    "check": "source_rows_present",
                    "status": "blocked",
                    "detail": "candidate source is empty or missing",
                }
            )
        return pd.DataFrame(rows)

    work = candidate.copy()
    dates = normalize_date(work["date"])
    available = normalize_date(work["available_date"])
    values = pd.to_numeric(work["total_return_index_or_adjusted_close"], errors="coerce")
    duplicate_count = int(work.duplicated(["date", "asset_or_index"]).sum())
    lineage = (
        work["data_source"].astype(str)
        + " "
        + work["source_vintage"].astype(str)
        + " "
        + work["asset_or_index"].astype(str)
    ).str.lower()
    forbidden = lineage.apply(lambda text: any(term in text for term in FORBIDDEN_LINEAGE_TERMS))
    candidate_role = set(work.get("candidate_source_role", pd.Series("", index=work.index)).astype(str).unique())
    not_certified_candidate = "candidate_not_certified_total_return" in candidate_role
    required_start = pd.to_datetime(config.required_start_date, errors="coerce")
    required_end = pd.to_datetime(config.required_end_date, errors="coerce")
    first_date = dates.min()
    last_date = dates.max()
    rows.extend(
        [
            {
                "source_name": source_name,
                "check": "minimum_history_rows",
                "status": _status(len(work) >= config.min_source_rows, "blocked"),
                "detail": f"rows={len(work)};min={config.min_source_rows}",
            },
            {
                "source_name": source_name,
                "check": "date_parseable",
                "status": _status(dates.notna().all(), "blocked"),
                "detail": f"bad_rows={int(dates.isna().sum())}",
            },
            {
                "source_name": source_name,
                "check": "available_date_parseable",
                "status": _status(available.notna().all(), "blocked"),
                "detail": f"bad_rows={int(available.isna().sum())}",
            },
            {
                "source_name": source_name,
                "check": "available_date_not_before_date",
                "status": _status((available >= dates).all(), "blocked"),
                "detail": f"bad_rows={int((available < dates).sum())}",
            },
            {
                "source_name": source_name,
                "check": "positive_numeric_level",
                "status": _status(np.isfinite(values).all() and (values > 0).all(), "blocked"),
                "detail": f"bad_rows={int((~np.isfinite(values) | (values <= 0)).sum())}",
            },
            {
                "source_name": source_name,
                "check": "no_duplicate_date_asset",
                "status": _status(duplicate_count == 0, "blocked"),
                "detail": f"duplicates={duplicate_count}",
            },
            {
                "source_name": source_name,
                "check": "required_date_range_covered",
                "status": _status(pd.notna(first_date) and pd.notna(last_date) and first_date <= required_start and last_date >= required_end, "blocked"),
                "detail": f"first_date={first_date.strftime('%Y%m%d') if pd.notna(first_date) else ''};last_date={last_date.strftime('%Y%m%d') if pd.notna(last_date) else ''}",
            },
            {
                "source_name": source_name,
                "check": "forbidden_lineage_terms_absent",
                "status": _status(not forbidden.any(), "blocked"),
                "detail": f"bad_rows={int(forbidden.sum())}",
            },
            {
                "source_name": source_name,
                "check": "not_short_window_candidate_proxy",
                "status": _status(not not_certified_candidate, "blocked"),
                "detail": "candidate_not_certified_total_return present" if not_certified_candidate else "",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_signal_coverage(candidate: pd.DataFrame, signal_panel: pd.DataFrame, config: LabelSourceProcurementConfig, source_name: str) -> pd.DataFrame:
    if candidate.empty or signal_panel.empty or "date" not in candidate.columns:
        return pd.DataFrame(
            [
                {
                    "source_name": source_name,
                    "horizon": int(h),
                    "unique_signal_dates": int(signal_panel.get("signal_date", pd.Series(dtype=str)).astype(str).nunique()) if not signal_panel.empty else 0,
                    "matched_source_dates": 0,
                    "enough_future_dates": 0,
                    "coverage_ratio": 0.0,
                    "coverage_status": "blocked",
                }
                for h in config.horizons
            ]
        )
    signals = signal_panel.loc[signal_panel["asset"].astype(str).eq("MARKET")].copy()
    signal_dates = sorted(signals["signal_date"].astype(str).str.replace("-", "", regex=False).unique())
    source = candidate.copy()
    source["_date"] = normalize_date(source["date"])
    source = source.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    source_dates = source["_date"].dt.strftime("%Y%m%d").tolist()
    pos_by_date = {date: idx for idx, date in enumerate(source_dates)}
    rows = []
    for horizon in config.horizons:
        matched = 0
        enough_future = 0
        for signal_date in signal_dates:
            pos = pos_by_date.get(signal_date)
            if pos is None:
                continue
            matched += 1
            if pos + int(horizon) < len(source):
                enough_future += 1
        coverage = enough_future / len(signal_dates) if signal_dates else 0.0
        rows.append(
            {
                "source_name": source_name,
                "horizon": int(horizon),
                "unique_signal_dates": int(len(signal_dates)),
                "matched_source_dates": int(matched),
                "enough_future_dates": int(enough_future),
                "coverage_ratio": float(coverage),
                "coverage_status": "pass" if coverage >= config.min_signal_coverage_ratio else "blocked",
            }
        )
    return pd.DataFrame(rows)


def build_short_window_smoke_test(candidate: pd.DataFrame, validation: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    parser_contract_ok = bool(
        not validation.empty
        and validation.loc[validation["check"].isin(["required_columns_present", "date_parseable", "available_date_parseable", "positive_numeric_level"]), "status"].eq("pass").all()
    )
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    return pd.DataFrame(
        [
            {
                "smoke_test": "v3_74_candidate_parser_and_coverage",
                "candidate_rows": int(len(candidate)),
                "parser_contract_ok": parser_contract_ok,
                "minimum_signal_coverage_ratio": min_coverage,
                "eligible_as_official_source": False,
                "label_generation_allowed": False,
                "portfolio_backtest_allowed": False,
                "status": "parser_smoke_pass_but_coverage_blocked" if parser_contract_ok and min_coverage < 0.8 else "blocked",
                "reason": "short-window candidate can test parsing only; not official or sufficient coverage evidence",
            }
        ]
    )


def build_import_decision(
    target_exists: bool,
    target_validation: pd.DataFrame,
    target_coverage: pd.DataFrame,
    smoke: pd.DataFrame,
    config: LabelSourceProcurementConfig,
) -> pd.DataFrame:
    target_checks_pass = bool(not target_validation.empty and target_validation["status"].eq("pass").all())
    coverage_pass = bool(not target_coverage.empty and target_coverage["coverage_status"].eq("pass").all())
    ready_for_v3_53 = bool(target_exists and target_checks_pass and coverage_pass)
    return pd.DataFrame(
        [
            {
                "decision_id": "official_market_source_import_decision",
                "target_source_path": _workspace_suffix(config.target_source_path),
                "target_exists": target_exists,
                "target_validation_pass": target_checks_pass,
                "target_coverage_pass": coverage_pass,
                "ready_for_v3_53": ready_for_v3_53,
                "smoke_test_status": str(smoke["status"].iloc[0]) if not smoke.empty else "",
                "decision": "rerun_v3_53" if ready_for_v3_53 else "wait_for_manual_or_vendor_source",
                "reason": "" if ready_for_v3_53 else "official/procured target source missing or failed validation/coverage gates",
            }
        ]
    )


def build_no_promotion_guard(import_decision: pd.DataFrame) -> pd.DataFrame:
    ready = bool(import_decision["ready_for_v3_53"].iloc[0]) if not import_decision.empty else False
    return pd.DataFrame(
        [
            {
                "result_type": "procurement_pack",
                "produced": True,
                "blocked": False,
                "reason": "V3.75 package produced",
            },
            {
                "result_type": "official_source_ready_for_v3_53",
                "produced": ready,
                "blocked": not ready,
                "reason": "requires target source validation and coverage pass",
            },
            {
                "result_type": "forward_total_return_labels",
                "produced": False,
                "blocked": True,
                "reason": "V3.75 does not build labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.75 does not run portfolio validation",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "no accepted label evidence yet",
            },
        ]
    )


def build_acceptance_checks(
    requirements: pd.DataFrame,
    contract: pd.DataFrame,
    candidate_validation: pd.DataFrame,
    candidate_coverage: pd.DataFrame,
    smoke: pd.DataFrame,
    import_decision: pd.DataFrame,
    guard: pd.DataFrame,
) -> pd.DataFrame:
    blocked_outputs = bool(guard.loc[guard["result_type"].isin(["portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any())
    return pd.DataFrame(
        [
            {
                "check": "procurement_requirements_written",
                "status": "pass" if len(requirements) >= 3 else "fail",
                "detail": f"rows={len(requirements)}",
            },
            {
                "check": "source_contract_written",
                "status": "pass" if set(REQUIRED_SOURCE_COLUMNS).issubset(set(contract["column"].astype(str))) else "fail",
                "detail": ",".join(REQUIRED_SOURCE_COLUMNS),
            },
            {
                "check": "candidate_validator_ran",
                "status": "pass" if not candidate_validation.empty else "fail",
                "detail": f"rows={len(candidate_validation)}",
            },
            {
                "check": "candidate_coverage_ran",
                "status": "pass" if not candidate_coverage.empty else "fail",
                "detail": f"rows={len(candidate_coverage)}",
            },
            {
                "check": "short_window_smoke_is_not_evidence",
                "status": "pass" if not smoke.empty and not _bool_series(smoke["eligible_as_official_source"]).any() else "fail",
                "detail": str(smoke["status"].iloc[0]) if not smoke.empty else "",
            },
            {
                "check": "import_decision_blocks_without_target_source",
                "status": "pass" if not import_decision.empty and not bool(import_decision["ready_for_v3_53"].iloc[0]) else "fail",
                "detail": str(import_decision["decision"].iloc[0]) if not import_decision.empty else "",
            },
            {
                "check": "no_portfolio_or_model_outputs",
                "status": "pass" if not blocked_outputs else "fail",
                "detail": "procurement and validation package only",
            },
        ]
    )


def build_report(
    requirements: pd.DataFrame,
    contract: pd.DataFrame,
    candidate_validation: pd.DataFrame,
    candidate_coverage: pd.DataFrame,
    smoke: pd.DataFrame,
    import_decision: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: LabelSourceProcurementConfig,
) -> str:
    ready = bool(import_decision["ready_for_v3_53"].iloc[0]) if not import_decision.empty else False
    min_coverage = float(candidate_coverage["coverage_ratio"].min()) if not candidate_coverage.empty else 0.0
    lines = [
        "# V3.75 Manual Label Source Procurement Pack",
        "",
        "## Decision",
        "",
        "- V3.75 creates the procurement and import-validation package for the missing MARKET label source.",
        "- V3.74 short-window JoinQuant data is used only as a parser/coverage smoke test.",
        "- Official V3.53 label generation remains blocked until a compliant target CSV is delivered.",
        "",
        "## Key Metrics",
        "",
        f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
        f"- Ready for V3.53: `{ready}`",
        f"- Short-window smoke minimum coverage: `{min_coverage:.4f}`",
        "",
        "## Procurement Requirements",
        "",
    ]
    lines.extend(markdown_table(requirements, ["requirement_id", "priority", "acceptable_basis", "required_history", "minimum_rows", "current_gap"], 20))
    lines.extend(["", "## Source Contract", ""])
    lines.extend(markdown_table(contract, ["column", "required", "type", "rule", "v3_75_enforcement"], 20))
    lines.extend(["", "## V3.74 Candidate Validation", ""])
    lines.extend(markdown_table(candidate_validation, ["source_name", "check", "status", "detail"], 30))
    lines.extend(["", "## V3.74 Candidate Coverage", ""])
    lines.extend(markdown_table(candidate_coverage, ["source_name", "horizon", "unique_signal_dates", "matched_source_dates", "coverage_ratio", "coverage_status"], 20))
    lines.extend(["", "## Smoke Test", ""])
    lines.extend(markdown_table(smoke, ["smoke_test", "candidate_rows", "parser_contract_ok", "minimum_signal_coverage_ratio", "eligible_as_official_source", "label_generation_allowed", "status"], 20))
    lines.extend(["", "## Import Decision", ""])
    lines.extend(markdown_table(import_decision, ["decision_id", "target_exists", "target_validation_pass", "target_coverage_pass", "ready_for_v3_53", "decision", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "1. Obtain or export a compliant long-history source using `provider_request_template.md`.",
            "2. Place it at `data_raw/market_labels/market_total_return_index.csv`.",
            "3. Rerun V3.75. If `ready_for_v3_53=true`, rerun V3.53 to generate labels.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(import_decision: pd.DataFrame, smoke: pd.DataFrame, config: LabelSourceProcurementConfig) -> str:
    ready = bool(import_decision["ready_for_v3_53"].iloc[0]) if not import_decision.empty else False
    smoke_status = str(smoke["status"].iloc[0]) if not smoke.empty else ""
    return "\n".join(
        [
            "# A-share Manual Label Source Procurement Pack V3.75",
            "",
            "## Dataset Decision",
            "",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            f"- Ready for V3.53: `{ready}`",
            f"- Smoke status: `{smoke_status}`",
            "- No labels, portfolio validation, or model promotion are produced by this package.",
            "",
        ]
    )
