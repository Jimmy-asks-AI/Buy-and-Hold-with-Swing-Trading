"""Regression fixtures for the V3.78 MARKET label sample validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_sample_intake_validator import (
    SampleIntakeConfig,
    build_acceptance_checks as build_v3_78_acceptance_checks,
    build_action_queue,
    build_no_execution_guard,
    discover_sample_files,
    validate_samples,
)


@dataclass(frozen=True)
class FixtureHarnessConfig:
    v3_78_config_path: Path
    v3_78_manifest_path: Path
    fixture_sample_dir: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    expected_cases: tuple[dict[str, Any], ...]


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _base_rows(n: int = 22) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    levels = [1000.0]
    for idx in range(1, n):
        levels.append(round(levels[-1] * (1.0 + 0.001 + 0.0001 * (idx % 3)), 4))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "asset_or_index": "MARKET",
            "total_return_index_or_adjusted_close": levels,
            "available_date": (dates + pd.Timedelta(days=1)).strftime("%Y%m%d"),
            "data_source": "csindex_official_total_return_service",
            "source_vintage": "fixture_v3_79_20260601",
        }
    )


def _fixture_frames() -> dict[str, pd.DataFrame]:
    valid = _base_rows()

    missing_column = valid.drop(columns=["available_date"]).copy()

    price_only = valid.rename(columns={"total_return_index_or_adjusted_close": "close"}).copy()
    price_only["return_basis"] = "price_index"

    available_violation = valid.copy()
    date_values = pd.to_datetime(available_violation["date"], format="%Y%m%d")
    available_violation["available_date"] = (date_values - pd.Timedelta(days=1)).dt.strftime("%Y%m%d")

    outlier_warning = valid.copy()
    outlier_warning.loc[10, "total_return_index_or_adjusted_close"] = round(
        float(outlier_warning.loc[9, "total_return_index_or_adjusted_close"]) * 1.25,
        4,
    )

    unapproved_source = valid.copy()
    unapproved_source["data_source"] = "tushare_price_index_daily"

    short_sample = valid.head(5).copy()

    return {
        "valid_csindex_sample.csv": valid,
        "missing_available_date_sample.csv": missing_column,
        "price_only_sample.csv": price_only,
        "available_date_violation_sample.csv": available_violation,
        "outlier_warning_sample.csv": outlier_warning,
        "unapproved_source_sample.csv": unapproved_source,
        "short_sample.csv": short_sample,
    }


def write_fixtures(config: FixtureHarnessConfig) -> pd.DataFrame:
    config.fixture_sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, frame in _fixture_frames().items():
        path = config.fixture_sample_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "fixture_file": _workspace_suffix(path),
                "case_id": name.replace(".csv", ""),
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def build_sample_config(base_raw: dict[str, Any], harness: FixtureHarnessConfig, root: Path) -> SampleIntakeConfig:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    return SampleIntakeConfig(
        v3_77_manifest_path=resolve(base_raw["v3_77_manifest_path"]),
        v3_77_source_candidates_path=resolve(base_raw["v3_77_source_candidates_path"]),
        v3_75_source_contract_path=resolve(base_raw["v3_75_source_contract_path"]),
        incoming_sample_dir=harness.fixture_sample_dir,
        target_source_path=resolve(base_raw["target_source_path"]),
        output_dir=harness.output_dir,
        catalog_path=harness.catalog_path,
        min_sample_rows=int(base_raw["min_sample_rows"]),
        max_available_lag_days=int(base_raw["max_available_lag_days"]),
        max_abs_daily_change=float(base_raw["max_abs_daily_change"]),
        allowed_asset_values=tuple(str(x) for x in base_raw["allowed_asset_values"]),
        approved_source_decisions=tuple(str(x) for x in base_raw["approved_source_decisions"]),
        approved_source_tokens=tuple(str(x) for x in base_raw["approved_source_tokens"]),
    )


def build_expected_vs_actual(
    sample_decisions: pd.DataFrame,
    validation_checks: pd.DataFrame,
    expected_cases: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    rows = []
    decisions = sample_decisions.copy()
    decisions["case_id"] = decisions["sample_file"].map(lambda value: Path(str(value)).stem)
    checks = validation_checks.copy()
    checks["case_id"] = checks["sample_file"].map(lambda value: Path(str(value)).stem)
    for expected in expected_cases:
        case_id = str(expected["case_id"])
        decision_rows = decisions.loc[decisions["case_id"].eq(case_id)]
        actual_decision = str(decision_rows["decision"].iloc[0]) if not decision_rows.empty else "missing"
        expected_decision = str(expected["expected_decision"])
        expected_check = str(expected.get("expected_check", ""))
        expected_status = str(expected.get("expected_check_status", ""))
        check_rows = checks.loc[checks["case_id"].eq(case_id) & checks["check"].astype(str).eq(expected_check)]
        actual_status = str(check_rows["status"].iloc[0]) if not check_rows.empty else ""
        decision_pass = actual_decision == expected_decision
        check_pass = True if not expected_check else actual_status == expected_status
        rows.append(
            {
                "case_id": case_id,
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "decision_pass": decision_pass,
                "expected_check": expected_check,
                "expected_check_status": expected_status,
                "actual_check_status": actual_status,
                "check_pass": check_pass,
                "regression_pass": bool(decision_pass and check_pass),
            }
        )
    return pd.DataFrame(rows)


def build_regression_summary(expected_vs_actual: pd.DataFrame, sample_decisions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "fixture_cases",
                "value": int(len(expected_vs_actual)),
            },
            {
                "metric": "regression_pass_cases",
                "value": int(expected_vs_actual["regression_pass"].astype(bool).sum()) if not expected_vs_actual.empty else 0,
            },
            {
                "metric": "passing_sample_cases",
                "value": int(sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum()) if not sample_decisions.empty else 0,
            },
            {
                "metric": "rejected_sample_cases",
                "value": int(sample_decisions["decision"].astype(str).eq("rejected_or_needs_repair").sum()) if not sample_decisions.empty else 0,
            },
            {
                "metric": "warning_count",
                "value": int(sample_decisions["warning_count"].sum()) if "warning_count" in sample_decisions.columns else 0,
            },
        ]
    )


def build_harness_acceptance(
    fixture_manifest: pd.DataFrame,
    expected_vs_actual: pd.DataFrame,
    sample_decisions: pd.DataFrame,
    guard: pd.DataFrame,
    config: FixtureHarnessConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "fixtures_written",
                "status": "pass" if len(fixture_manifest) >= 6 else "fail",
                "detail": f"rows={len(fixture_manifest)}",
            },
            {
                "check": "expected_vs_actual_all_pass",
                "status": "pass" if expected_vs_actual["regression_pass"].astype(bool).all() else "fail",
                "detail": ",".join(expected_vs_actual.loc[~expected_vs_actual["regression_pass"].astype(bool), "case_id"].astype(str)),
            },
            {
                "check": "positive_cases_present",
                "status": "pass" if sample_decisions["decision"].astype(str).eq("candidate_pass_to_v3_75_review").sum() >= 2 else "fail",
                "detail": "valid and warning-only samples should pass to V3.75 review",
            },
            {
                "check": "negative_cases_present",
                "status": "pass" if sample_decisions["decision"].astype(str).eq("rejected_or_needs_repair").sum() >= 4 else "fail",
                "detail": "known bad samples should be rejected",
            },
            {
                "check": "target_source_not_written",
                "status": "pass" if not config.target_source_path.exists() else "warn",
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["target_csv_write", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "fixture harness only",
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
    fixture_manifest: pd.DataFrame,
    sample_decisions: pd.DataFrame,
    expected_vs_actual: pd.DataFrame,
    summary: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: FixtureHarnessConfig,
) -> str:
    passed = int(expected_vs_actual["regression_pass"].astype(bool).sum()) if not expected_vs_actual.empty else 0
    lines = [
        "# V3.79 MARKET Label Sample Fixture Harness",
        "",
        "## Decision",
        "",
        "- V3.79 creates synthetic CSV fixtures and validates V3.78 behavior end to end.",
        "- Fixtures live under the V3.79 output directory, not under the official raw label path.",
        "- The harness does not write the official target CSV, generate labels, run portfolios, or promote a model.",
        "",
        "## Key Metrics",
        "",
        f"- Fixture directory: `{_workspace_suffix(config.fixture_sample_dir)}`",
        f"- Fixture cases: `{len(expected_vs_actual)}`",
        f"- Regression pass cases: `{passed}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        "",
        "## Fixture Manifest",
        "",
    ]
    lines.extend(markdown_table(fixture_manifest, ["case_id", "fixture_file", "row_count", "column_count", "size_bytes"], 20))
    lines.extend(["", "## Candidate File Decision", ""])
    lines.extend(markdown_table(sample_decisions, ["sample_file", "row_count", "blocking_fail_count", "warning_count", "decision", "reason"], 20))
    lines.extend(["", "## Expected Vs Actual", ""])
    lines.extend(markdown_table(expected_vs_actual, ["case_id", "expected_decision", "actual_decision", "expected_check", "expected_check_status", "actual_check_status", "regression_pass"], 20))
    lines.extend(["", "## Regression Summary", ""])
    lines.extend(markdown_table(summary, ["metric", "value"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Use this harness as a regression test before accepting real provider samples.", ""])
    return "\n".join(lines)


def build_catalog(summary: pd.DataFrame, config: FixtureHarnessConfig) -> str:
    values = {str(row.metric): row.value for row in summary.itertuples(index=False)}
    return "\n".join(
        [
            "# A-share MARKET Label Sample Fixture Harness V3.79",
            "",
            "## Dataset Decision",
            "",
            f"- Fixture directory: `{_workspace_suffix(config.fixture_sample_dir)}`",
            f"- Fixture cases: `{values.get('fixture_cases', 0)}`",
            f"- Regression pass cases: `{values.get('regression_pass_cases', 0)}`",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
