"""Market-level total-return label source audit for HIRSSM V3.52.

This module searches local files for candidates that could support MARKET-level
forward total-return labels. It is intentionally conservative: strategy NAVs,
benchmark outputs, and price-only index histories are inventoried but rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_LABEL_REQUIREMENTS = [
    {
        "requirement": "date",
        "required": True,
        "description": "Observation date for the market total-return series.",
    },
    {
        "requirement": "asset_or_index",
        "required": True,
        "description": "MARKET or a verified broad-market index identifier.",
    },
    {
        "requirement": "total_return_index_or_adjusted_close",
        "required": True,
        "description": "Total-return index level or explicitly adjusted market proxy.",
    },
    {
        "requirement": "available_date",
        "required": True,
        "description": "Date when this observation became usable in research.",
    },
    {
        "requirement": "data_source",
        "required": True,
        "description": "Provider and endpoint/file lineage.",
    },
    {
        "requirement": "source_vintage",
        "required": True,
        "description": "Provider file vintage or ingestion timestamp.",
    },
]

VALUE_COLUMNS = {
    "total_return_index",
    "total_return_index_or_adjusted_close",
    "adjusted_close",
    "adjusted_index_close",
    "adjusted_nav",
    "total_return_nav",
}
PRICE_ONLY_COLUMNS = {"close", "open", "high", "low", "pct_chg", "change"}
DATE_COLUMNS = {"date", "trade_date", "calendar_date"}
ASSET_COLUMNS = {"asset", "asset_or_index", "index_code", "symbol", "ts_code"}
SOURCE_COLUMNS = {"data_source", "source", "provider", "source_vintage", "fetched_at"}
FORBIDDEN_SOURCE_TERMS = {
    "backtest",
    "benchmark_nav",
    "strategy_nav",
    "nav_",
    "none_raw",
    "raw_close",
    "unadjusted",
    "price_only",
}


@dataclass(frozen=True)
class AuditConfig:
    glob_patterns: tuple[str, ...]
    market_proxy_codes: tuple[str, ...]
    output_dir: Path
    catalog_path: Path


def _norm_columns(columns: pd.Index) -> list[str]:
    return [str(col).strip() for col in columns]


def _safe_read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False), ""
    except Exception as exc:  # pragma: no cover - defensive file audit path.
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _find_column(columns: set[str], choices: set[str]) -> str:
    hits = sorted(columns.intersection(choices))
    return hits[0] if hits else ""


def _date_range(df: pd.DataFrame, date_col: str) -> tuple[str, str]:
    if not date_col or date_col not in df.columns or df.empty:
        return "", ""
    values = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    if values.notna().sum() == 0:
        return "", ""
    return values.min().strftime("%Y%m%d"), values.max().strftime("%Y%m%d")


def _source_zone(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root).as_posix()
    if rel.startswith("outputs/"):
        return "outputs"
    if rel.startswith("data_raw/"):
        return "data_raw"
    if rel.startswith("data_catalog/"):
        return "data_catalog"
    if rel.startswith("configs/"):
        return "configs"
    return "other"


def _lineage_type(path: Path, columns: set[str], root: Path) -> str:
    rel = path.resolve().relative_to(root).as_posix().lower()
    name = path.name.lower()
    if rel.startswith("outputs/") and ("nav" in name or "return" in name or "benchmark" in name):
        return "backtest_or_strategy_derived"
    if "daily_csindex" in rel:
        return "external_csindex_price_only"
    if "daily_sw" in rel:
        return "external_industry_price_only"
    if "daily_qfq" in rel:
        return "stock_adjusted_price_not_market_label"
    if "valuation" in rel:
        return "valuation_not_return_label"
    if "weight" in rel or "constituent" in rel:
        return "constituent_or_weight_not_return_label"
    if "macro" in rel or "commodity" in rel:
        return "macro_not_market_return_label"
    if VALUE_COLUMNS.intersection(columns):
        return "possible_adjusted_or_total_return_file"
    return "metadata_or_contract"


def _has_forbidden_lineage(path: Path, columns: set[str], root: Path) -> bool:
    rel = path.resolve().relative_to(root).as_posix().lower()
    joined = " ".join(sorted(columns)).lower()
    return any(term in rel or term in joined for term in FORBIDDEN_SOURCE_TERMS) or rel.startswith("outputs/")


def expand_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for item in root.glob(pattern):
            if not item.is_file():
                continue
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(item)
    return sorted(paths, key=lambda p: p.as_posix())


def inventory_file(path: Path, root: Path, market_proxy_codes: tuple[str, ...]) -> dict[str, Any]:
    df, error = _safe_read_csv(path)
    columns = set(_norm_columns(df.columns)) if not df.empty or not error else set()
    date_col = _find_column(columns, DATE_COLUMNS)
    asset_col = _find_column(columns, ASSET_COLUMNS)
    value_col = _find_column(columns, VALUE_COLUMNS)
    source_col = _find_column(columns, SOURCE_COLUMNS)
    date_min, date_max = _date_range(df, date_col)
    zone = _source_zone(path, root)
    lineage = _lineage_type(path, columns, root) if not error else "unreadable"
    broad_market_code_present = False
    if asset_col and asset_col in df.columns:
        broad_market_code_present = bool(df[asset_col].astype(str).isin(market_proxy_codes).any())
    if "index_code" in df.columns:
        broad_market_code_present = broad_market_code_present or bool(df["index_code"].astype(str).isin(market_proxy_codes).any())
    has_available_date = "available_date" in columns
    has_source_vintage = "source_vintage" in columns or "fetched_at" in columns
    has_price_only = bool(PRICE_ONLY_COLUMNS.intersection(columns)) and not value_col
    row = {
        "path": path.resolve().relative_to(root).as_posix(),
        "file_name": path.name,
        "source_zone": zone,
        "lineage_type": lineage,
        "read_error": error,
        "rows": int(len(df)) if not error else 0,
        "columns": "|".join(_norm_columns(df.columns)) if not error else "",
        "date_column": date_col,
        "date_min": date_min,
        "date_max": date_max,
        "asset_column": asset_col,
        "value_column": value_col,
        "source_column": source_col,
        "has_available_date": has_available_date,
        "has_source_vintage": has_source_vintage,
        "has_price_only_columns": has_price_only,
        "broad_market_code_present": broad_market_code_present,
        "file_size_bytes": int(path.stat().st_size),
        "modified_time": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
    }
    return row


def assess_candidate(row: dict[str, Any], root: Path) -> dict[str, Any]:
    path = root / str(row["path"])
    columns = set(str(row["columns"]).split("|")) if row.get("columns") else set()
    accepted = False
    decision = "rejected"
    reasons: list[str] = []

    if row["read_error"]:
        reasons.append("file_unreadable")
    if row["source_zone"] == "outputs":
        reasons.append("derived_output_or_backtest_artifact")
    if row["lineage_type"] == "external_csindex_price_only":
        reasons.append("csindex_price_index_not_total_return")
    if row["lineage_type"] == "external_industry_price_only":
        reasons.append("industry_price_index_not_market_total_return")
    if row["lineage_type"] == "stock_adjusted_price_not_market_label":
        reasons.append("stock_adjusted_price_not_market_scope")
    if row["lineage_type"] in {"valuation_not_return_label", "constituent_or_weight_not_return_label", "macro_not_market_return_label"}:
        reasons.append(row["lineage_type"])
    if not row["date_column"]:
        reasons.append("missing_date_column")
    if not row["asset_column"] and not row["broad_market_code_present"]:
        reasons.append("missing_market_asset_or_index_identifier")
    if not row["value_column"]:
        reasons.append("missing_total_return_or_adjusted_value_column")
    if not row["has_available_date"]:
        reasons.append("missing_available_date")
    if not row["has_source_vintage"]:
        reasons.append("missing_source_vintage")
    if _has_forbidden_lineage(path, columns, root):
        reasons.append("forbidden_or_non_pit_lineage")

    minimum_fields_ok = (
        bool(row["date_column"])
        and (bool(row["asset_column"]) or bool(row["broad_market_code_present"]))
        and bool(row["value_column"])
        and bool(row["has_available_date"])
        and bool(row["has_source_vintage"])
    )
    if minimum_fields_ok and not reasons:
        accepted = True
        decision = "accepted_market_total_return_candidate"
        reasons.append("all_market_label_source_requirements_met")
    elif row["value_column"] and not row["source_zone"] == "outputs":
        decision = "rejected_candidate_needs_pit_lineage_review"
    elif row["has_price_only_columns"]:
        decision = "rejected_price_only"
    elif row["source_zone"] == "outputs":
        decision = "rejected_backtest_derived"

    return {
        **row,
        "accepted_for_market_label": accepted,
        "decision": decision,
        "reason": ";".join(dict.fromkeys(reasons)),
    }


def audit_local_sources(root: Path, config: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = expand_globs(root, config.glob_patterns)
    inventory_rows = [inventory_file(path, root, config.market_proxy_codes) for path in paths]
    inventory = pd.DataFrame(inventory_rows)
    assessments = pd.DataFrame([assess_candidate(row, root) for row in inventory_rows])
    return inventory, assessments


def build_readiness_checks(assessments: pd.DataFrame) -> pd.DataFrame:
    accepted = int(assessments["accepted_for_market_label"].sum()) if not assessments.empty else 0
    price_only_rejected = bool((assessments["decision"] == "rejected_price_only").any()) if not assessments.empty else False
    backtest_rejected = bool((assessments["decision"] == "rejected_backtest_derived").any()) if not assessments.empty else False
    return pd.DataFrame(
        [
            {
                "check": "local_market_label_source_audit_completed",
                "status": "pass",
                "detail": f"candidate_files={len(assessments)}",
            },
            {
                "check": "accepted_market_total_return_source_available",
                "status": "pass" if accepted else "blocked",
                "detail": f"accepted_sources={accepted}",
            },
            {
                "check": "price_only_index_sources_rejected",
                "status": "pass" if price_only_rejected else "not_applicable",
                "detail": "CSIndex/SW price-only daily files are not accepted as total-return labels.",
            },
            {
                "check": "backtest_or_nav_outputs_rejected",
                "status": "pass" if backtest_rejected else "not_applicable",
                "detail": "Backtest NAV and benchmark outputs are rejected as label sources.",
            },
            {
                "check": "performance_validation_allowed_now",
                "status": "pass" if accepted else "blocked",
                "detail": "requires accepted PIT market total-return source and verified calendar",
            },
        ]
    )


def build_no_label_guard(assessments: pd.DataFrame) -> pd.DataFrame:
    accepted = int(assessments["accepted_for_market_label"].sum()) if not assessments.empty else 0
    return pd.DataFrame(
        [
            {
                "result_type": "market_forward_total_return_labels",
                "produced": False,
                "blocked": True,
                "reason": "V3.52 audits sources only; label generation waits for an accepted source.",
            },
            {
                "result_type": "state_stratified_signal_validation",
                "produced": False,
                "blocked": True,
                "reason": "accepted_market_total_return_sources=" + str(accepted),
            },
            {
                "result_type": "portfolio_backtest_or_performance_claim",
                "produced": False,
                "blocked": True,
                "reason": "No backtest may be run from rejected price-only or derived NAV files.",
            },
        ]
    )


def build_acquisition_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "route": "trusted_vendor_market_total_return_index",
                "target_dataset": "market_total_return_index",
                "minimum_fields": "date,asset_or_index,total_return_index_or_adjusted_close,available_date,data_source,source_vintage",
                "preferred_coverage": "20000101 to latest trading day",
                "notes": "Use CSI total-return index, JoinQuant adjusted index proxy, Wind, or another provider with clear total-return methodology.",
            },
            {
                "priority": 2,
                "route": "manual_csv_import_contract",
                "target_dataset": "data_raw/market_labels/market_total_return_index.csv",
                "minimum_fields": "date,asset_or_index,total_return_index_or_adjusted_close,available_date,data_source,source_vintage",
                "preferred_coverage": "same as V3.50 signal panel",
                "notes": "A later importer can validate this file and produce V3.49/V3.51 labels.",
            },
            {
                "priority": 3,
                "route": "verified_trade_calendar",
                "target_dataset": "trade_calendar",
                "minimum_fields": "calendar_date,is_trading_day,exchange,available_date,data_source",
                "preferred_coverage": "20000101 to latest trading day plus label horizon buffer",
                "notes": "Needed to align 1/5/20/60 trading-day horizons and label_available_date.",
            },
            {
                "priority": 4,
                "route": "stock_level_adjustment_sources",
                "target_dataset": "stock_adj_factor + stock_universe_all_status",
                "minimum_fields": "date,asset,adj_factor,available_date,data_source,source_vintage",
                "preferred_coverage": "full A-share lifecycle",
                "notes": "Not required for current V3.50 MARKET signals, but mandatory before stock-level signals.",
            },
        ]
    )


def build_acceptance_checks(assessments: pd.DataFrame, readiness: pd.DataFrame, guard: pd.DataFrame) -> pd.DataFrame:
    accepted = int(assessments["accepted_for_market_label"].sum()) if not assessments.empty else 0
    labels_produced = bool(guard["produced"].any()) if not guard.empty else False
    performance_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return pd.DataFrame(
        [
            {
                "check": "audit_inventory_nonempty",
                "status": "pass" if len(assessments) > 0 else "fail",
                "detail": f"candidate_files={len(assessments)}",
            },
            {
                "check": "no_label_production_in_audit_step",
                "status": "pass" if not labels_produced else "fail",
                "detail": "V3.52 is source audit only.",
            },
            {
                "check": "no_performance_validation_without_accepted_source",
                "status": "pass" if (accepted > 0 or performance_status == "blocked") else "fail",
                "detail": f"accepted_sources={accepted};performance_status={performance_status}",
            },
            {
                "check": "price_only_or_derived_sources_not_promoted",
                "status": "pass"
                if not assessments.loc[
                    assessments["decision"].isin(["rejected_price_only", "rejected_backtest_derived"]),
                    "accepted_for_market_label",
                ].any()
                else "fail",
                "detail": "rejected files remain rejected.",
            },
            {
                "check": "next_manual_data_request_defined",
                "status": "pass",
                "detail": "market_total_return_index plus verified trade_calendar.",
            },
        ]
    )


def build_report(
    inventory: pd.DataFrame,
    assessments: pd.DataFrame,
    readiness: pd.DataFrame,
    acquisition_plan: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    accepted = assessments.loc[assessments["accepted_for_market_label"]] if not assessments.empty else pd.DataFrame()
    by_decision = assessments["decision"].value_counts().to_dict() if not assessments.empty else {}
    lines = [
        "# V3.52 Market Label Source Audit",
        "",
        "## Decision",
        "",
        "- V3.52 audits local candidates for the MARKET-level PIT total-return label source required by V3.51.",
        "- Backtest NAV, benchmark outputs, and price-only index histories are inventoried but not accepted as labels.",
        "- No forward labels, IC, hit rate, NAV, drawdown, portfolio backtest, or model promotion is produced.",
        "",
        "## Summary",
        "",
        f"- Candidate files audited: `{len(inventory)}`",
        f"- Accepted MARKET label sources: `{len(accepted)}`",
        f"- Decision counts: `{by_decision}`",
        "",
        "## Accepted Sources",
        "",
    ]
    if accepted.empty:
        lines.append("No local file passed the MARKET total-return/PIT label-source gate.")
    else:
        lines.extend(["| path | date_range | value_column | source_column |", "|---|---|---|---|"])
        for row in accepted.itertuples(index=False):
            lines.append(f"| `{row.path}` | `{row.date_min}` to `{row.date_max}` | `{row.value_column}` | `{row.source_column}` |")
    lines.extend(
        [
            "",
            "## Main Rejections",
            "",
            "| path | decision | reason |",
            "|---|---|---|",
        ]
    )
    for row in assessments.head(30).itertuples(index=False):
        lines.append(f"| `{row.path}` | `{row.decision}` | {str(row.reason)[:180]} |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Acquisition Plan",
            "",
            "| priority | route | target_dataset | notes |",
            "|---:|---|---|---|",
        ]
    )
    for row in acquisition_plan.itertuples(index=False):
        lines.append(f"| {int(row.priority)} | `{row.route}` | `{row.target_dataset}` | {row.notes} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- If an accepted source is later added, run a strict importer that validates the V3.52 contract before generating V3.51 labels.",
            "- Until then, V3.49/V3.50 remain validation-blocked for realized performance.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame, assessments: pd.DataFrame) -> str:
    accepted = int(assessments["accepted_for_market_label"].sum()) if not assessments.empty else 0
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return "\n".join(
        [
            "# A-share Market Label Source Audit V3.52",
            "",
            "## Dataset Decision",
            "",
            f"- Audit ready: `true`",
            f"- Accepted MARKET total-return label sources: `{accepted}`",
            f"- Performance validation status: `{perf_status}`",
            "- Rejected source classes include old backtest NAV/benchmark outputs and price-only index histories.",
            "",
            "## Required Source Contract",
            "",
            "`date, asset_or_index, total_return_index_or_adjusted_close, available_date, data_source, source_vintage`",
            "",
        ]
    )
