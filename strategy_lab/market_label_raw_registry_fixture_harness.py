"""Regression harness for the V3.80 raw-sample registry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_label_raw_sample_registry import (
    RawSampleRegistryConfig,
    build_acceptance_checks as build_v3_80_acceptance_checks,
    build_controlled_handoff,
    build_license_review_queue,
    build_no_execution_guard,
    build_raw_sample_registry,
)


@dataclass(frozen=True)
class RawRegistryFixtureHarnessConfig:
    v3_80_config_path: Path
    v3_80_manifest_path: Path
    fixture_sample_dir: Path
    license_status_path: Path
    license_evidence_dir: Path
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_sample() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=22)
    levels = [1000.0]
    for idx in range(1, len(dates)):
        levels.append(round(levels[-1] * (1.0 + 0.001 + 0.0001 * (idx % 3)), 4))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "asset_or_index": "MARKET",
            "total_return_index_or_adjusted_close": levels,
            "available_date": (dates + pd.Timedelta(days=1)).strftime("%Y%m%d"),
            "data_source": "csindex_official_total_return_service",
            "source_vintage": "fixture_v3_81_20260601",
        }
    )


def write_fixtures(config: RawRegistryFixtureHarnessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.fixture_sample_dir.mkdir(parents=True, exist_ok=True)
    config.license_evidence_dir.mkdir(parents=True, exist_ok=True)
    base = _base_sample()
    frames = {
        "valid_csindex_raw.csv": base,
        "valid_csindex_duplicate.csv": base.copy(),
        "valid_csindex_tampered.csv": base.assign(
            total_return_index_or_adjusted_close=lambda df: df["total_return_index_or_adjusted_close"].where(
                df.index != 10,
                df.loc[10, "total_return_index_or_adjusted_close"] + 8.0,
            )
        ),
        "unapproved_tushare_raw.csv": base.assign(data_source="tushare_price_index_daily"),
    }
    rows = []
    for name, frame in frames.items():
        path = config.fixture_sample_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "case_id": name.replace(".csv", ""),
                "fixture_file": _workspace_suffix(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    fixture_manifest = pd.DataFrame(rows)

    valid_hash = str(fixture_manifest.loc[fixture_manifest["case_id"].eq("valid_csindex_raw"), "sha256"].iloc[0])
    evidence_path = config.license_evidence_dir / "valid_csindex_license_note.md"
    evidence_path.write_text(
        "\n".join(
            [
                "# Fixture License Evidence",
                "",
                "Synthetic V3.81 evidence file for registry regression only.",
                "Status: approved_research_and_derived_labels",
                "",
            ]
        ),
        encoding="utf-8",
    )
    license_status = pd.DataFrame(
        [
            {
                "sha256": valid_hash,
                "sample_file": "",
                "license_status": "approved_research_and_derived_labels",
                "license_evidence_path": _workspace_suffix(evidence_path),
                "reviewer": "v3_81_fixture",
                "review_note": "synthetic approval for registry gate test only",
            }
        ]
    )
    config.license_status_path.parent.mkdir(parents=True, exist_ok=True)
    license_status.to_csv(config.license_status_path, index=False, encoding="utf-8-sig")
    return fixture_manifest, license_status


def build_registry_config(
    base_raw: dict[str, Any],
    harness: RawRegistryFixtureHarnessConfig,
    root: Path,
    previous_registry_path: Path,
) -> RawSampleRegistryConfig:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    return RawSampleRegistryConfig(
        v3_78_manifest_path=resolve(base_raw["v3_78_manifest_path"]),
        v3_79_manifest_path=resolve(base_raw["v3_79_manifest_path"]),
        incoming_sample_dir=harness.fixture_sample_dir,
        target_source_path=resolve(base_raw["target_source_path"]),
        previous_registry_path=previous_registry_path,
        license_status_path=harness.license_status_path,
        output_dir=harness.output_dir,
        catalog_path=harness.catalog_path,
        approved_source_tokens=tuple(str(x) for x in base_raw["approved_source_tokens"]),
        allowed_extensions=tuple(str(x) for x in base_raw["allowed_extensions"]),
        license_approved_values=tuple(str(x) for x in base_raw["license_approved_values"]),
    )


def _row_by_case(registry: pd.DataFrame, case_id: str) -> pd.Series:
    rows = registry.loc[registry["sample_file"].astype(str).map(lambda value: Path(value).stem).eq(case_id)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def build_duplicate_hash_report(second_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = second_registry.loc[second_registry["sha256"].astype(str).ne("")].groupby("sha256")
    for digest, group in grouped:
        if len(group) > 1:
            rows.append(
                {
                    "sha256": digest,
                    "duplicate_count": int(len(group)),
                    "sample_files": "|".join(group["sample_file"].astype(str).tolist()),
                    "duplicate_detected": True,
                }
            )
    if not rows:
        rows.append({"sha256": "", "duplicate_count": 0, "sample_files": "", "duplicate_detected": False})
    return pd.DataFrame(rows)


def build_tamper_report(second_registry: pd.DataFrame) -> pd.DataFrame:
    valid = _row_by_case(second_registry, "valid_csindex_raw")
    tampered = _row_by_case(second_registry, "valid_csindex_tampered")
    valid_hash = str(valid.get("sha256", ""))
    tampered_hash = str(tampered.get("sha256", ""))
    return pd.DataFrame(
        [
            {
                "case_id": "tamper_hash_differs",
                "valid_sha256": valid_hash,
                "tampered_sha256": tampered_hash,
                "tamper_detected": bool(valid_hash and tampered_hash and valid_hash != tampered_hash),
            }
        ]
    )


def build_first_seen_stability_report(first_registry: pd.DataFrame, second_registry: pd.DataFrame) -> pd.DataFrame:
    merged = first_registry.loc[first_registry["sha256"].astype(str).ne(""), ["sha256", "first_seen_utc"]].drop_duplicates().merge(
        second_registry.loc[second_registry["sha256"].astype(str).ne(""), ["sha256", "first_seen_utc"]].drop_duplicates(),
        on="sha256",
        suffixes=("_first", "_second"),
    )
    if merged.empty:
        return pd.DataFrame([{"sha256": "", "first_seen_utc_first": "", "first_seen_utc_second": "", "stable": False}])
    merged["stable"] = merged["first_seen_utc_first"].astype(str).eq(merged["first_seen_utc_second"].astype(str))
    return merged


def build_source_gate_report(second_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id in ["valid_csindex_raw", "unapproved_tushare_raw"]:
        row = _row_by_case(second_registry, case_id)
        rows.append(
            {
                "case_id": case_id,
                "matched_source_token": str(row.get("matched_source_token", "")),
                "source_token_approved": bool(row.get("source_token_approved", False)),
                "v3_78_review_allowed": bool(row.get("v3_78_review_allowed", False)),
            }
        )
    return pd.DataFrame(rows)


def build_license_gate_report(second_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id in ["valid_csindex_raw", "valid_csindex_duplicate", "valid_csindex_tampered"]:
        row = _row_by_case(second_registry, case_id)
        rows.append(
            {
                "case_id": case_id,
                "license_status": str(row.get("license_status", "")),
                "license_evidence_path": str(row.get("license_evidence_path", "")),
                "v3_78_review_allowed": bool(row.get("v3_78_review_allowed", False)),
            }
        )
    return pd.DataFrame(rows)


def build_expected_vs_actual(
    duplicate_report: pd.DataFrame,
    tamper_report: pd.DataFrame,
    first_seen_report: pd.DataFrame,
    source_gate_report: pd.DataFrame,
    license_gate_report: pd.DataFrame,
    expected_cases: tuple[dict[str, Any], ...],
) -> pd.DataFrame:
    observed = {
        "duplicate_same_hash": bool(duplicate_report["duplicate_detected"].astype(bool).any()),
        "tamper_hash_differs": bool(tamper_report["tamper_detected"].astype(bool).all()),
        "first_seen_stable": bool(first_seen_report["stable"].astype(bool).all()),
        "source_token_allows_valid": bool(source_gate_report.loc[source_gate_report["case_id"].eq("valid_csindex_raw"), "source_token_approved"].astype(bool).all()),
        "source_token_blocks_unapproved": not bool(source_gate_report.loc[source_gate_report["case_id"].eq("unapproved_tushare_raw"), "source_token_approved"].astype(bool).any()),
        "license_allows_approved_hash": bool(license_gate_report.loc[license_gate_report["case_id"].eq("valid_csindex_raw"), "v3_78_review_allowed"].astype(bool).all()),
        "license_blocks_unknown_tamper": not bool(license_gate_report.loc[license_gate_report["case_id"].eq("valid_csindex_tampered"), "v3_78_review_allowed"].astype(bool).any()),
    }
    rows = []
    for expected in expected_cases:
        case_id = str(expected["case_id"])
        actual = bool(observed.get(case_id, False))
        expected_value = bool(expected["expected_pass"])
        rows.append(
            {
                "case_id": case_id,
                "expected_pass": expected_value,
                "actual_pass": actual,
                "regression_pass": actual == expected_value,
                "detail": str(expected.get("detail", "")),
            }
        )
    return pd.DataFrame(rows)


def build_regression_summary(expected_vs_actual: pd.DataFrame, second_registry: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "expected_cases", "value": int(len(expected_vs_actual))},
            {"metric": "regression_pass_cases", "value": int(expected_vs_actual["regression_pass"].astype(bool).sum())},
            {"metric": "registered_files", "value": int(second_registry["sha256"].astype(str).ne("").sum())},
            {"metric": "v3_78_review_allowed_count", "value": int(second_registry["v3_78_review_allowed"].astype(bool).sum())},
            {"metric": "source_token_blocked_count", "value": int((~second_registry["source_token_approved"].astype(bool)).sum())},
        ]
    )


def build_harness_acceptance(expected_vs_actual: pd.DataFrame, second_registry: pd.DataFrame, guard: pd.DataFrame, target_source_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "expected_vs_actual_all_pass",
                "status": "pass" if expected_vs_actual["regression_pass"].astype(bool).all() else "fail",
                "detail": ",".join(expected_vs_actual.loc[~expected_vs_actual["regression_pass"].astype(bool), "case_id"].astype(str)),
            },
            {
                "check": "duplicate_case_present",
                "status": "pass" if second_registry["sha256"].duplicated(keep=False).any() else "fail",
                "detail": "duplicate hash should be present",
            },
            {
                "check": "approved_and_blocked_cases_present",
                "status": "pass" if second_registry["v3_78_review_allowed"].astype(bool).any() and (~second_registry["v3_78_review_allowed"].astype(bool)).any() else "fail",
                "detail": "both allowed and blocked registry rows should exist",
            },
            {
                "check": "target_source_not_written",
                "status": "pass" if not target_source_path.exists() else "warn",
                "detail": _workspace_suffix(target_source_path),
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["target_csv_write", "v3_78_sample_validation", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "registry fixture only",
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
    second_registry: pd.DataFrame,
    expected_vs_actual: pd.DataFrame,
    summary: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: RawRegistryFixtureHarnessConfig,
) -> str:
    passed = int(expected_vs_actual["regression_pass"].astype(bool).sum()) if not expected_vs_actual.empty else 0
    lines = [
        "# V3.81 MARKET Raw Registry Fixture Harness",
        "",
        "## Decision",
        "",
        "- V3.81 validates V3.80 registry behavior using synthetic raw samples.",
        "- It tests hash stability, duplicate identity, tamper detection, source-token gating, license gating, and first-seen persistence.",
        "- Synthetic files remain under the V3.81 output directory and are not official source data.",
        "",
        "## Key Metrics",
        "",
        f"- Fixture directory: `{_workspace_suffix(config.fixture_sample_dir)}`",
        f"- Expected cases: `{len(expected_vs_actual)}`",
        f"- Regression pass cases: `{passed}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        "",
        "## Fixture Manifest",
        "",
    ]
    lines.extend(markdown_table(fixture_manifest, ["case_id", "fixture_file", "size_bytes", "sha256"], 20))
    lines.extend(["", "## Second Registry", ""])
    lines.extend(markdown_table(second_registry, ["sample_file", "sha256", "matched_source_token", "source_token_approved", "license_status", "v3_78_review_allowed"], 20))
    lines.extend(["", "## Expected Vs Actual", ""])
    lines.extend(markdown_table(expected_vs_actual, ["case_id", "expected_pass", "actual_pass", "regression_pass", "detail"], 20))
    lines.extend(["", "## Regression Summary", ""])
    lines.extend(markdown_table(summary, ["metric", "value"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Keep V3.81 as the registry regression guard before real sample intake.", ""])
    return "\n".join(lines)


def build_catalog(summary: pd.DataFrame, config: RawRegistryFixtureHarnessConfig) -> str:
    values = {str(row.metric): row.value for row in summary.itertuples(index=False)}
    return "\n".join(
        [
            "# A-share MARKET Raw Registry Fixture Harness V3.81",
            "",
            "## Dataset Decision",
            "",
            f"- Fixture directory: `{_workspace_suffix(config.fixture_sample_dir)}`",
            f"- Expected cases: `{values.get('expected_cases', 0)}`",
            f"- Regression pass cases: `{values.get('regression_pass_cases', 0)}`",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
