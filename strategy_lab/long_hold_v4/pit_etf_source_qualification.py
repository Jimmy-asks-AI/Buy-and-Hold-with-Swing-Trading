"""Aggregate ETF price/NAV evidence without converting current-final data into PIT history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .pit_source_code_archive import authenticate_current_or_archive


ROOT = Path(__file__).resolve().parents[2]
RECENT_AUDIT_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_price_nav" / "run_manifest.json"
TENCENT_AUDIT_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_tencent_price" / "run_manifest.json"
EASTMONEY_AUDIT_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_eastmoney_nav" / "run_manifest.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_source_qualification"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(path)


def load_authenticated_audit(path: Path, *, required_outputs: set[str]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    code_path = ROOT / str(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError:
        raise ValueError(f"ETF source audit code hash mismatch: {path}")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    if not required_outputs.issubset(outputs):
        raise ValueError(f"ETF source audit omits required outputs: {path}")
    for role, item in outputs.items():
        output_path = ROOT / str(item.get("path", ""))
        if not output_path.is_file() or _sha256(output_path) != str(item.get("sha256", "")):
            raise ValueError(f"ETF source audit output hash mismatch: {role}")
    if manifest.get("historical_backtest_allowed") is not False:
        raise ValueError(f"ETF source audit violates observation-only boundary: {path}")
    return manifest


def build_qualification_checks(
    recent: dict[str, Any],
    tencent: dict[str, Any],
    eastmoney: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "check": "recent_joinquant_cross_source_content",
            "passed": recent.get("recent_cross_source_passed") is True,
            "observed": recent.get("qualification_status"),
            "required": True,
            "evidence_class": "current_final_content",
        },
        {
            "check": "delisted_price_independent_coverage",
            "passed": bool(
                tencent.get("cross_source_content_passed") is True
                and int(tencent.get("selected_delisted_assets", 0)) == 123
                and int(tencent.get("ready_assets", 0)) >= 123
            ),
            "observed": f"{tencent.get('ready_assets', 0)}/{tencent.get('selected_delisted_assets', 0)}",
            "required": "123/123",
            "evidence_class": "current_final_content",
        },
        {
            "check": "full_market_price_independent_coverage",
            "passed": tencent.get("full_market_current_final_source_passed") is True,
            "observed": f"{tencent.get('ready_assets', 0)}/{tencent.get('selected_assets', 0)}",
            "required": "full master current-final content pass",
            "evidence_class": "current_final_content",
        },
        {
            "check": "full_lifecycle_nav_independent_coverage",
            "passed": eastmoney.get("full_market_current_final_source_passed") is True,
            "observed": f"{eastmoney.get('ready_assets', 0)}/{eastmoney.get('selected_assets', 0)}",
            "required": "full master current-final content pass",
            "evidence_class": "current_final_content",
        },
        {
            "check": "terminal_event_nav_cash_separation",
            "passed": bool((eastmoney.get("terminal_event_boundary") or {}).get("boundary_passed") is True),
            "observed": (eastmoney.get("terminal_event_boundary") or {}).get("last_nav_date"),
            "required": "regular NAV ends before liquidation cash event",
            "evidence_class": "current_final_content",
        },
        {
            "check": "price_version_monitoring_depth",
            "passed": tencent.get("version_monitoring_ready") is True,
            "observed": tencent.get("version_depth_coverage", 0.0),
            "required": 1.0,
            "evidence_class": "revision_monitoring",
        },
        {
            "check": "nav_version_monitoring_depth",
            "passed": float(eastmoney.get("version_depth_coverage", 0.0)) == 1.0,
            "observed": eastmoney.get("version_depth_coverage", 0.0),
            "required": 1.0,
            "evidence_class": "revision_monitoring",
        },
        {
            "check": "historical_available_date_evidence",
            "passed": False,
            "observed": "collection-date availability only",
            "required": "source evidence available on or before every historical trade date",
            "evidence_class": "pit_history",
        },
    ]
    return pd.DataFrame(rows)


def qualify(
    recent_path: Path = RECENT_AUDIT_PATH,
    tencent_path: Path = TENCENT_AUDIT_PATH,
    eastmoney_path: Path = EASTMONEY_AUDIT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    recent = load_authenticated_audit(
        recent_path,
        required_outputs={"price_checks", "nav_checks", "summary", "qualification_checks"},
    )
    tencent = load_authenticated_audit(
        tencent_path,
        required_outputs={
            "asset_summary",
            "material_close_mismatches",
            "summary",
            "qualification_checks",
            "source_version_inventory",
        },
    )
    eastmoney = load_authenticated_audit(
        eastmoney_path,
        required_outputs={"nav_asset_summary", "summary", "qualification_checks", "source_version_inventory"},
    )
    checks = build_qualification_checks(recent, tencent, eastmoney)
    content_checks = checks[checks["evidence_class"].eq("current_final_content")]
    current_final_content_passed = bool(content_checks["passed"].all())
    full_price_pending = not bool(
        checks.loc[checks["check"].eq("full_market_price_independent_coverage"), "passed"].iloc[0]
    )
    if current_final_content_passed:
        qualification = "PASS_CURRENT_FINAL_PRICE_NAV_CONTENT_PIT_BLOCKED"
    elif bool(
        checks.loc[
            checks["check"].isin(
                {
                    "recent_joinquant_cross_source_content",
                    "delisted_price_independent_coverage",
                    "full_lifecycle_nav_independent_coverage",
                    "terminal_event_nav_cash_separation",
                }
            ),
            "passed",
        ].all()
    ):
        qualification = "PASS_EXPANDED_CURRENT_FINAL_CONTENT_FULL_PRICE_PENDING"
    else:
        qualification = "BLOCKED_CURRENT_FINAL_SOURCE_CONTENT_OR_COVERAGE"
    summary = {
        "as_of_date": eastmoney.get("as_of_date"),
        "qualification_status": qualification,
        "current_final_content_passed": current_final_content_passed,
        "full_market_price_pending": full_price_pending,
        "recent_cross_source_passed": recent.get("recent_cross_source_passed") is True,
        "tencent_selected_assets": int(tencent.get("selected_assets", 0)),
        "tencent_ready_assets": int(tencent.get("ready_assets", 0)),
        "tencent_delisted_assets": int(tencent.get("selected_delisted_assets", 0)),
        "tencent_price_rows": int(tencent.get("tencent_rows", 0)),
        "tencent_price_coverage_start": tencent.get("coverage_start"),
        "tencent_price_coverage_end": tencent.get("coverage_end"),
        "tencent_material_close_mismatch_rows": int(tencent.get("material_close_mismatch_rows", 0)),
        "tencent_assets_below_95pct_row_coverage": int(tencent.get("assets_below_95pct_row_coverage", 0)),
        "tencent_assets_below_95pct_row_coverage_codes": tencent.get(
            "assets_below_95pct_row_coverage_codes", []
        ),
        "eastmoney_selected_assets": int(eastmoney.get("selected_assets", 0)),
        "eastmoney_ready_assets": int(eastmoney.get("ready_assets", 0)),
        "eastmoney_nav_rows": int(eastmoney.get("nav_rows", 0)),
        "eastmoney_nav_coverage_start": eastmoney.get("coverage_start"),
        "eastmoney_nav_coverage_end": eastmoney.get("coverage_end"),
        "price_version_depth_coverage": float(tencent.get("version_depth_coverage", 0.0)),
        "nav_version_depth_coverage": float(eastmoney.get("version_depth_coverage", 0.0)),
        "historical_available_date_evidence_passed": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "formal_table_promotion_allowed": False,
        "boundary": "content agreement and lifecycle coverage are separate from historical PIT availability",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checks_path = output_dir / "qualification_checks.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "ETF_SOURCE_QUALIFICATION.md"
    _atomic_csv(checks, checks_path)
    _atomic_json(summary, summary_path)
    report = "\n".join(
        [
            "# ETF Source Qualification",
            "",
            f"Qualification: `{qualification}`",
            "",
            f"- Recent JoinQuant content check: `{str(summary['recent_cross_source_passed']).lower()}`",
            f"- Tencent independent prices: {summary['tencent_ready_assets']:,}/{summary['tencent_selected_assets']:,} assets",
            f"- Eastmoney independent NAV: {summary['eastmoney_ready_assets']:,}/{summary['eastmoney_selected_assets']:,} assets",
            f"- Current-final price coverage: {summary['tencent_price_coverage_start']} to {summary['tencent_price_coverage_end']}",
            f"- Current-final NAV coverage: {summary['eastmoney_nav_coverage_start']} to {summary['eastmoney_nav_coverage_end']}",
            f"- Disclosed material price mismatches / low-coverage assets: {summary['tencent_material_close_mismatch_rows']} / {summary['tencent_assets_below_95pct_row_coverage']}",
            f"- Price/NAV version monitoring depth: {summary['price_version_depth_coverage']:.2%} / {summary['nav_version_depth_coverage']:.2%}",
            "",
            "## Boundary",
            "",
            "A current-final series may validate values and lifecycle coverage. It cannot prove that the same values were available on historical trade dates. This report never authorizes a formal PIT table, historical backtest, or model promotion.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    code_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        **summary,
        "inputs": [
            {"role": "recent_price_nav_audit", "path": _relative(recent_path), "sha256": _sha256(recent_path)},
            {"role": "tencent_price_audit", "path": _relative(tencent_path), "sha256": _sha256(tencent_path)},
            {"role": "eastmoney_nav_audit", "path": _relative(eastmoney_path), "sha256": _sha256(eastmoney_path)},
        ],
        "outputs": [
            {"role": "qualification_checks", "path": _relative(checks_path), "sha256": _sha256(checks_path), "rows": len(checks)},
            {"role": "summary", "path": _relative(summary_path), "sha256": _sha256(summary_path), "rows": 1},
            {"role": "report", "path": _relative(report_path), "sha256": _sha256(report_path), "rows": len(report.splitlines())},
        ],
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
    }
    _atomic_json(manifest, output_dir / "run_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent-audit", type=Path, default=RECENT_AUDIT_PATH)
    parser.add_argument("--tencent-audit", type=Path, default=TENCENT_AUDIT_PATH)
    parser.add_argument("--eastmoney-audit", type=Path, default=EASTMONEY_AUDIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            qualify(args.recent_audit, args.tencent_audit, args.eastmoney_audit, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
