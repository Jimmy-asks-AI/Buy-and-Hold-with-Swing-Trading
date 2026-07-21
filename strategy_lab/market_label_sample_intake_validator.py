"""Incoming MARKET label sample validator for HIRSSM V3.78.

The validator inspects vendor/manual sample CSV files before any file is allowed
to become the official MARKET label source. It writes only intake evidence and
does not create labels, portfolios, or model-promotion artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SampleIntakeConfig:
    v3_77_manifest_path: Path
    v3_77_source_candidates_path: Path
    v3_75_source_contract_path: Path
    incoming_sample_dir: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    min_sample_rows: int
    max_available_lag_days: int
    max_abs_daily_change: float
    allowed_asset_values: tuple[str, ...]
    approved_source_decisions: tuple[str, ...]
    approved_source_tokens: tuple[str, ...]


CANONICAL_ALIASES = {
    "date": ("date", "trade_date", "tradedate"),
    "asset_or_index": ("asset_or_index", "index_code", "index_name", "asset", "symbol"),
    "total_return_index_or_adjusted_close": (
        "total_return_index_or_adjusted_close",
        "total_return_close",
        "total_return_index",
        "adjusted_close",
        "adj_close",
        "close",
    ),
    "available_date": ("available_date", "publish_date", "release_date", "asof_date"),
    "data_source": ("data_source", "source", "provider", "endpoint"),
    "source_vintage": ("source_vintage", "vintage", "file_vintage", "batch_id", "version"),
}

RESERVED_NON_SAMPLE_CSVS = {
    "license_review_status.csv",
}


def _workspace_suffix(path: Path) -> str:
    anchors = ("outputs", "data_raw", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _norm_col(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    norm_map = {_norm_col(col): col for col in columns}
    for alias in aliases:
        found = norm_map.get(_norm_col(alias))
        if found is not None:
            return found
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    eight_digit = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text, errors="coerce")
    if eight_digit.any():
        parsed.loc[eight_digit] = pd.to_datetime(text.loc[eight_digit], format="%Y%m%d", errors="coerce")
    return parsed


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _sample_csv_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.csv")
        if path.is_file() and path.name.lower() not in RESERVED_NON_SAMPLE_CSVS
    )


def discover_sample_files(config: SampleIntakeConfig) -> pd.DataFrame:
    if not config.incoming_sample_dir.exists():
        return pd.DataFrame(
            [
                {
                    "sample_file": _workspace_suffix(config.incoming_sample_dir),
                    "exists": False,
                    "size_bytes": 0,
                    "status": "no_sample_directory",
                    "detail": "create this directory and place vendor/manual sample CSV files here",
                }
            ]
        )
    files = _sample_csv_files(config.incoming_sample_dir)
    if not files:
        return pd.DataFrame(
            [
                {
                    "sample_file": _workspace_suffix(config.incoming_sample_dir),
                    "exists": True,
                    "size_bytes": 0,
                    "status": "no_sample_files",
                    "detail": "place at least one CSV sample here",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "sample_file": _workspace_suffix(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "status": "found",
                "detail": "",
            }
            for path in files
        ]
    )


def build_expected_contract(source_contract: pd.DataFrame) -> pd.DataFrame:
    required = source_contract.copy()
    required["accepted_aliases"] = required["column"].map(lambda col: ",".join(CANONICAL_ALIASES.get(str(col), (str(col),))))
    required["sample_gate"] = required["v3_75_enforcement"].map(lambda value: "blocking" if str(value).lower() == "blocking" else "review")
    return required


def _source_is_approved(file_name: str, values: pd.Series, config: SampleIntakeConfig) -> bool:
    joined = (file_name + " " + " ".join(values.dropna().astype(str).head(20))).lower()
    return any(token.lower() in joined for token in config.approved_source_tokens)


def _return_basis_is_acceptable(frame: pd.DataFrame, value_col: str | None) -> bool:
    if value_col is None:
        return False
    value_name = _norm_col(value_col)
    if "total_return" in value_name or "adjusted" in value_name or "adj_" in value_name:
        return True
    basis_col = _find_column(list(frame.columns), ("return_basis", "basis", "收益口径"))
    if basis_col is None:
        return False
    basis = " ".join(frame[basis_col].dropna().astype(str).head(20)).lower()
    return "total" in basis or "adjusted" in basis


def _validate_one_sample(path: Path, config: SampleIntakeConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_name = _workspace_suffix(path)
    checks: list[dict[str, Any]] = []
    decision = {
        "sample_file": sample_name,
        "row_count": 0,
        "blocking_fail_count": 0,
        "warning_count": 0,
        "decision": "rejected",
        "reason": "",
    }

    def add(check: str, status: str, severity: str, detail: str) -> None:
        checks.append(
            {
                "sample_file": sample_name,
                "check": check,
                "status": status,
                "severity": severity,
                "detail": detail,
            }
        )

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive branch for bad vendor files.
        add("csv_readable", "fail", "blocking", type(exc).__name__)
        decision["blocking_fail_count"] = 1
        decision["reason"] = "CSV is not readable"
        return checks, decision

    decision["row_count"] = int(len(frame))
    add("csv_readable", "pass", "blocking", f"columns={len(frame.columns)}")
    add("min_sample_rows", "pass" if len(frame) >= config.min_sample_rows else "fail", "blocking", f"rows={len(frame)};min={config.min_sample_rows}")

    col_map = {key: _find_column(list(frame.columns), aliases) for key, aliases in CANONICAL_ALIASES.items()}
    missing = sorted(key for key, value in col_map.items() if value is None)
    add("required_columns_or_aliases", "pass" if not missing else "fail", "blocking", ",".join(missing))
    if missing:
        decision["blocking_fail_count"] = sum(1 for row in checks if row["status"] == "fail" and row["severity"] == "blocking")
        decision["warning_count"] = sum(1 for row in checks if row["status"] == "warn")
        decision["decision"] = "rejected_or_needs_repair"
        decision["reason"] = "missing required columns"
        return checks, decision

    date_values = _parse_dates(frame[col_map["date"]])
    available_values = _parse_dates(frame[col_map["available_date"]])
    level_values = pd.to_numeric(frame[col_map["total_return_index_or_adjusted_close"]], errors="coerce")
    assets = frame[col_map["asset_or_index"]].astype(str).str.strip()

    add("date_parse", "pass" if date_values.notna().all() else "fail", "blocking", f"bad_rows={int(date_values.isna().sum())}")
    add("available_date_parse", "pass" if available_values.notna().all() else "fail", "blocking", f"bad_rows={int(available_values.isna().sum())}")
    add("available_date_not_before_date", "pass" if (available_values >= date_values).fillna(False).all() else "fail", "blocking", "available_date must be >= date")

    lag_days = (available_values - date_values).dt.days
    lag_ok = lag_days.dropna().le(config.max_available_lag_days).all() if not lag_days.dropna().empty else False
    add("available_lag_reasonable", "pass" if lag_ok else "warn", "warning", f"max_lag_days={lag_days.max() if not lag_days.dropna().empty else ''}")

    add("level_positive_numeric", "pass" if (level_values > 0).fillna(False).all() else "fail", "blocking", f"bad_rows={int((~(level_values > 0).fillna(False)).sum())}")
    add("accepted_return_basis", "pass" if _return_basis_is_acceptable(frame, col_map["total_return_index_or_adjusted_close"]) else "fail", "blocking", col_map["total_return_index_or_adjusted_close"] or "")

    accepted_assets = {value.upper() for value in config.allowed_asset_values}
    asset_ok = assets.str.upper().isin(accepted_assets).all()
    add("approved_asset_or_index", "pass" if asset_ok else "fail", "blocking", ",".join(sorted(assets.dropna().astype(str).unique())[:10]))

    duplicates = frame.assign(_date=date_values, _asset=assets).duplicated(["_date", "_asset"]).sum()
    add("no_duplicate_date_asset", "pass" if duplicates == 0 else "fail", "blocking", f"duplicates={int(duplicates)}")

    if date_values.notna().all() and len(frame) > 1:
        sorted_levels = pd.Series(level_values.values, index=date_values).sort_index()
        change = sorted_levels.pct_change().abs()
        outliers = int(change.gt(config.max_abs_daily_change).sum())
        add("daily_change_outlier_screen", "pass" if outliers == 0 else "warn", "warning", f"outliers={outliers};threshold={config.max_abs_daily_change}")
        add("date_order_unique", "pass" if date_values.is_unique else "fail", "blocking", f"unique_dates={date_values.nunique()}")
    else:
        add("daily_change_outlier_screen", "warn", "warning", "insufficient valid rows")
        add("date_order_unique", "fail", "blocking", "date parse failed or insufficient rows")

    source_ok = _source_is_approved(path.name, frame[col_map["data_source"]], config)
    add("source_matches_v3_77_procurement_route", "pass" if source_ok else "fail", "blocking", "source must reference an approved V3.77 route token")

    vintage_ok = frame[col_map["source_vintage"]].astype(str).str.strip().ne("").all()
    add("source_vintage_nonempty", "pass" if vintage_ok else "fail", "blocking", "source_vintage required")

    blocking_fail_count = sum(1 for row in checks if row["status"] == "fail" and row["severity"] == "blocking")
    warning_count = sum(1 for row in checks if row["status"] == "warn")
    decision["blocking_fail_count"] = blocking_fail_count
    decision["warning_count"] = warning_count
    if blocking_fail_count == 0:
        decision["decision"] = "candidate_pass_to_v3_75_review"
        decision["reason"] = "sample passed intake; rerun V3.75 on a controlled copy before target write"
    else:
        decision["decision"] = "rejected_or_needs_repair"
        decision["reason"] = "blocking checks failed"
    return checks, decision


def validate_samples(config: SampleIntakeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.incoming_sample_dir.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "sample_file": _workspace_suffix(config.incoming_sample_dir),
                        "check": "sample_directory_exists",
                        "status": "blocked",
                        "severity": "blocking",
                        "detail": "incoming sample directory is missing",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "sample_file": _workspace_suffix(config.incoming_sample_dir),
                        "row_count": 0,
                        "blocking_fail_count": 0,
                        "warning_count": 0,
                        "decision": "waiting_for_sample",
                        "reason": "create directory and place provider sample CSV here",
                    }
                ]
            ),
        )
    files = _sample_csv_files(config.incoming_sample_dir)
    if not files:
        return (
            pd.DataFrame(
                [
                    {
                        "sample_file": _workspace_suffix(config.incoming_sample_dir),
                        "check": "sample_file_exists",
                        "status": "blocked",
                        "severity": "blocking",
                        "detail": "no CSV sample found",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "sample_file": _workspace_suffix(config.incoming_sample_dir),
                        "row_count": 0,
                        "blocking_fail_count": 0,
                        "warning_count": 0,
                        "decision": "waiting_for_sample",
                        "reason": "place provider sample CSV here",
                    }
                ]
            ),
        )
    all_checks: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    for path in files:
        checks, decision = _validate_one_sample(path, config)
        all_checks.extend(checks)
        all_decisions.append(decision)
    return pd.DataFrame(all_checks), pd.DataFrame(all_decisions)


def build_action_queue(sample_decisions: pd.DataFrame, config: SampleIntakeConfig) -> pd.DataFrame:
    any_pass = sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").any() if not sample_decisions.empty else False
    waiting = sample_decisions["decision"].astype(str).eq("waiting_for_sample").any() if not sample_decisions.empty else True
    rows = [
        {
            "step_order": 1,
            "action": f"place provider sample CSV under {_workspace_suffix(config.incoming_sample_dir)}",
            "status": "active" if waiting else "done",
            "may_execute_now": False,
            "reason": "validator needs a sample file but does not fetch data",
        },
        {
            "step_order": 2,
            "action": "rerun V3.78 sample intake validator",
            "status": "active" if waiting else "done",
            "may_execute_now": False,
            "reason": "repeat after each new sample delivery",
        },
        {
            "step_order": 3,
            "action": "copy a passing sample to a controlled review path, then rerun V3.75",
            "status": "active" if any_pass else "blocked",
            "may_execute_now": False,
            "reason": "V3.75 remains the contract gate before final target write",
        },
        {
            "step_order": 4,
            "action": "write final approved source to data_raw/market_labels/market_total_return_index.csv",
            "status": "blocked",
            "may_execute_now": False,
            "reason": "manual-controlled action only after V3.75/V3.76 pass",
        },
    ]
    return pd.DataFrame(rows)


def build_dropzone_readme(config: SampleIntakeConfig, expected_contract: pd.DataFrame) -> str:
    lines = [
        "# V3.78 Incoming Sample Dropzone",
        "",
        f"- Place provider/manual sample CSV files under `{_workspace_suffix(config.incoming_sample_dir)}`.",
        "- The validator reads CSV files only; it does not write the official target source.",
        f"- Minimum sample rows: `{config.min_sample_rows}`.",
        f"- Approved asset/index values: `{', '.join(config.allowed_asset_values)}`.",
        "",
        "## Required Contract",
        "",
    ]
    lines.extend(markdown_table(expected_contract, ["column", "required", "type", "rule", "accepted_aliases"], 20))
    lines.extend(
        [
            "",
            "## Passing Sample Handoff",
            "",
            "1. Keep the original provider sample immutable.",
            "2. Rerun V3.78 and inspect `candidate_file_decision.csv`.",
            "3. If a file passes, rerun V3.75 on a controlled copy before writing any final target source.",
            "",
        ]
    )
    return "\n".join(lines)


def build_no_execution_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "sample_intake_validation",
                "produced": True,
                "blocked": False,
                "reason": "V3.78 creates intake evidence for sample files.",
            },
            {
                "result_type": "target_csv_write",
                "produced": False,
                "blocked": True,
                "reason": "V3.78 must not write the official target CSV.",
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
                "reason": "No validated label file is generated here.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Sample intake is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(
    inventory: pd.DataFrame,
    validation_checks: pd.DataFrame,
    sample_decisions: pd.DataFrame,
    action_queue: pd.DataFrame,
    guard: pd.DataFrame,
    config: SampleIntakeConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "inventory_written",
                "status": "pass" if not inventory.empty else "fail",
                "detail": f"rows={len(inventory)}",
            },
            {
                "check": "validation_checks_written",
                "status": "pass" if not validation_checks.empty else "fail",
                "detail": f"rows={len(validation_checks)}",
            },
            {
                "check": "sample_decision_written",
                "status": "pass" if not sample_decisions.empty else "fail",
                "detail": f"rows={len(sample_decisions)}",
            },
            {
                "check": "waiting_state_allowed_without_sample",
                "status": "pass" if sample_decisions["decision"].astype(str).isin(["waiting_for_sample", "candidate_pass_to_v3_75_review", "rejected_or_needs_repair"]).all() else "fail",
                "detail": ",".join(sample_decisions["decision"].astype(str).unique()),
            },
            {
                "check": "action_queue_blocks_target_write",
                "status": "pass" if not action_queue.loc[action_queue["action"].astype(str).str.contains("write final approved source", case=False, na=False), "may_execute_now"].astype(bool).any() else "fail",
                "detail": "target write is manual-controlled",
            },
            {
                "check": "target_source_not_written",
                "status": "pass" if not config.target_source_path.exists() else "warn",
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["target_csv_write", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "intake only",
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
    inventory: pd.DataFrame,
    validation_checks: pd.DataFrame,
    sample_decisions: pd.DataFrame,
    action_queue: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: SampleIntakeConfig,
) -> str:
    passed = int(sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if not sample_decisions.empty else 0
    waiting = int(sample_decisions["decision"].astype(str).eq("waiting_for_sample").sum()) if not sample_decisions.empty else 0
    lines = [
        "# V3.78 MARKET Label Sample Intake Validator",
        "",
        "## Decision",
        "",
        "- V3.78 validates incoming sample CSV files before any source can be promoted to V3.75 review.",
        "- It does not write the official target CSV, generate labels, run portfolios, or promote a model.",
        "- If no sample exists, the run is still valid and records the waiting state.",
        "",
        "## Key Metrics",
        "",
        f"- Incoming sample directory: `{_workspace_suffix(config.incoming_sample_dir)}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        f"- Passing samples: `{passed}`",
        f"- Waiting sample rows: `{waiting}`",
        "",
        "## Sample Inventory",
        "",
    ]
    lines.extend(markdown_table(inventory, ["sample_file", "exists", "size_bytes", "status", "detail"], 20))
    lines.extend(["", "## Candidate File Decision", ""])
    lines.extend(markdown_table(sample_decisions, ["sample_file", "row_count", "blocking_fail_count", "warning_count", "decision", "reason"], 20))
    lines.extend(["", "## Validation Checks", ""])
    lines.extend(markdown_table(validation_checks, ["sample_file", "check", "status", "severity", "detail"], 40))
    lines.extend(["", "## Action Queue", ""])
    lines.extend(markdown_table(action_queue, ["step_order", "action", "status", "may_execute_now", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Put a licensed provider sample in the incoming sample directory and rerun V3.78.", ""])
    return "\n".join(lines)


def build_catalog(sample_decisions: pd.DataFrame, config: SampleIntakeConfig) -> str:
    passed = int(sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if not sample_decisions.empty else 0
    waiting = int(sample_decisions["decision"].astype(str).eq("waiting_for_sample").sum()) if not sample_decisions.empty else 0
    return "\n".join(
        [
            "# A-share MARKET Label Sample Intake Validator V3.78",
            "",
            "## Dataset Decision",
            "",
            f"- Incoming sample directory: `{_workspace_suffix(config.incoming_sample_dir)}`",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            f"- Passing samples: `{passed}`",
            f"- Waiting sample rows: `{waiting}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
