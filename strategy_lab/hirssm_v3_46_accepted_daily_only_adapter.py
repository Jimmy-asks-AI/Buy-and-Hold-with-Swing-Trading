#!/usr/bin/env python
"""Validate the accepted daily-only adapter introduced in V3.46."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from accepted_daily_only_adapter import (
    AcceptedDailyOnlyAdapter,
    assert_capability_blocked,
    has_bad_ohlc_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "accepted_daily_only_adapter_v3_46.json"
TASK_ID = "20260529_v3_46_accepted_daily_only_adapter"
VERSION = "V3.46"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_adapter(config: dict[str, Any]) -> AcceptedDailyOnlyAdapter:
    return AcceptedDailyOnlyAdapter(
        data_root=ROOT / config["data_root"],
        quarantine_path=ROOT / config["quarantine_path"],
    )


def validate_sample_dates(adapter: AcceptedDailyOnlyAdapter, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date in config["sample_dates"]:
        raw = adapter.load_trade_date(trade_date, apply_quarantine=False, add_scope_columns=False)
        accepted = adapter.load_trade_date(trade_date, apply_quarantine=True, add_scope_columns=True)
        raw_bad = has_bad_ohlc_rows(raw)
        accepted_bad = has_bad_ohlc_rows(accepted)
        quarantined = int(accepted["quarantined_rows_for_trade_date"].iloc[0]) if not accepted.empty else 0
        close_panel = adapter.get_close(trade_date)
        rows.append(
            {
                "trade_date": trade_date,
                "raw_rows": int(raw.shape[0]),
                "accepted_rows": int(accepted.shape[0]),
                "rows_removed_by_quarantine": int(raw.shape[0] - accepted.shape[0]),
                "adapter_quarantined_rows": quarantined,
                "raw_has_bad_ohlc": raw_bad,
                "accepted_has_bad_ohlc": accepted_bad,
                "close_panel_rows": int(close_panel.shape[0]),
                "scope_values": "|".join(sorted(accepted["data_scope"].astype(str).unique())),
                "price_adjustment_values": "|".join(sorted(accepted["price_adjustment"].astype(str).unique())),
                "adjusted_return_allowed_values": "|".join(sorted(accepted["adjusted_return_allowed"].astype(str).unique())),
                "status": "pass" if not accepted_bad and int(close_panel.shape[0]) == int(accepted.shape[0]) else "fail",
            }
        )
    return pd.DataFrame(rows)


def validate_capabilities(adapter: AcceptedDailyOnlyAdapter, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for capability in config["allowed_capabilities_to_test"]:
        try:
            adapter.require_capability(capability)
            status = "pass"
            detail = "allowed"
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            detail = repr(exc)
        rows.append({"capability": capability, "expected": "allowed", "status": status, "detail": detail})

    for capability in config["blocked_capabilities_to_test"]:
        try:
            detail = assert_capability_blocked(adapter, capability)
            status = "pass"
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            detail = repr(exc)
        rows.append({"capability": capability, "expected": "blocked", "status": status, "detail": detail})
    for return_basis in config["blocked_portfolio_return_basis_to_test"]:
        capability = f"portfolio_backtest_return_basis:{return_basis}"
        try:
            adapter.assert_can_run_portfolio_backtest(return_basis)
            status = "fail"
            detail = "portfolio backtest unexpectedly allowed"
        except Exception as exc:  # noqa: BLE001
            status = "pass"
            detail = repr(exc)
        rows.append({"capability": capability, "expected": "blocked", "status": status, "detail": detail})
    return pd.DataFrame(rows)


def build_acceptance_checks(
    config: dict[str, Any],
    validation: pd.DataFrame,
    capability_checks: pd.DataFrame,
    processed_summary: pd.DataFrame,
    guard_manifest: dict[str, Any],
    acceptance_manifest: dict[str, Any],
) -> pd.DataFrame:
    summary = dict(zip(processed_summary["metric"], processed_summary["value"]))
    quarantine_removed = int(validation["rows_removed_by_quarantine"].sum())
    expected_quarantine = int(summary["quarantined_ohlc_rows"])
    rows = [
        {
            "check": "v3_45_1_acceptance_passed",
            "status": "pass" if bool(acceptance_manifest.get("acceptance_pass", False)) else "fail",
            "detail": acceptance_manifest.get("data_decision", ""),
        },
        {
            "check": "v3_44_guard_still_passed",
            "status": "pass" if bool(guard_manifest.get("self_check_pass", False)) else "fail",
            "detail": f"hard_blocked_uses={guard_manifest.get('hard_blocked_uses')}",
        },
        {
            "check": "sample_dates_have_no_bad_ohlc_after_adapter",
            "status": "pass" if not bool(validation["accepted_has_bad_ohlc"].any()) else "fail",
            "detail": str(int(validation["accepted_has_bad_ohlc"].sum())),
        },
        {
            "check": "sample_quarantine_count_matches_summary",
            "status": "pass" if quarantine_removed == expected_quarantine else "fail",
            "detail": f"sample_removed={quarantine_removed};summary={expected_quarantine}",
        },
        {
            "check": "normal_date_not_unnecessarily_reduced",
            "status": "pass"
            if int(validation.loc[validation["trade_date"] == config["sample_dates"][-1], "rows_removed_by_quarantine"].iloc[0]) == 0
            else "fail",
            "detail": config["sample_dates"][-1],
        },
        {
            "check": "capability_gate_all_pass",
            "status": "pass" if bool((capability_checks["status"] == "pass").all()) else "fail",
            "detail": str(int((capability_checks["status"] != "pass").sum())),
        },
        {
            "check": "accepted_scope_is_none_raw",
            "status": "pass"
            if bool((validation["price_adjustment_values"] == "none_raw").all())
            else "fail",
            "detail": "|".join(sorted(validation["price_adjustment_values"].unique())),
        },
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "adapter validation only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    validation: pd.DataFrame,
    capability_checks: pd.DataFrame,
    checks: pd.DataFrame,
    processed_summary: pd.DataFrame,
) -> str:
    summary = dict(zip(processed_summary["metric"], processed_summary["value"]))
    failed = checks.loc[checks["status"] != "pass"]
    lines = [
        "# V3.46 Accepted Daily-Only Adapter",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.46 adds a reusable accepted daily-only adapter.",
        "- The adapter applies the V3.45.1 row-level quarantine by default.",
        "- It keeps V3.44 capability blocks active, so adjusted-return, valuation, dividend-yield, and portfolio-performance use remain unavailable.",
        "- No model, factor, or backtest promotion occurred.",
        "",
        "## Processed Scope",
        "",
        f"- Raw rows: `{int(summary['raw_rows'])}`",
        f"- Quarantined OHLC rows: `{int(summary['quarantined_ohlc_rows'])}`",
        f"- Accepted processed rows: `{int(summary['accepted_processed_rows'])}`",
        "",
        "## Sample Validation",
        "",
        "| trade_date | raw_rows | accepted_rows | removed | raw_bad_ohlc | accepted_bad_ohlc | status |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in validation.itertuples(index=False):
        lines.append(
            f"| `{row.trade_date}` | {row.raw_rows} | {row.accepted_rows} | {row.rows_removed_by_quarantine} | "
            f"`{row.raw_has_bad_ohlc}` | `{row.accepted_has_bad_ohlc}` | `{row.status}` |"
        )
    lines.extend(
        [
            "",
            "## Capability Gate",
            "",
            "| capability | expected | status |",
            "|---|---|---|",
        ]
    )
    for row in capability_checks.itertuples(index=False):
        lines.append(f"| `{row.capability}` | `{row.expected}` | `{row.status}` |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Failed checks: `{len(failed)}`",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Downstream Rule",
            "",
            "- Downstream scripts must import `AcceptedDailyOnlyAdapter` instead of reading `data_raw/tushare_daily_only/v3_38/daily` directly.",
            "- This adapter only approves raw OHLCV diagnostics and liquidity/activity/coverage use.",
            "- A future adjusted-return research layer still requires `adj_factor`, security master, and valuation interfaces.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(processed_summary: pd.DataFrame, checks: pd.DataFrame) -> str:
    summary = dict(zip(processed_summary["metric"], processed_summary["value"]))
    failed = checks.loc[checks["status"] != "pass"]
    return "\n".join(
        [
            "# A-share Accepted Daily-Only Adapter V3.46",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            f"- Adapter accepted: `{failed.empty}`",
            f"- Raw rows: `{int(summary['raw_rows'])}`",
            f"- Quarantined rows: `{int(summary['quarantined_ohlc_rows'])}`",
            f"- Accepted processed rows: `{int(summary['accepted_processed_rows'])}`",
            "",
            "## Contract",
            "",
            "- Use `strategy_lab/accepted_daily_only_adapter.py` for downstream reads.",
            "- Quarantine is applied by default.",
            "- Output scope is `accepted_processed_daily_only`.",
            "- Price adjustment remains `none_raw`; adjusted-return and valuation research remain blocked.",
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    rows.append(
        {
            "check": "acceptance_checks_all_pass",
            "status": "pass" if bool((checks["status"] == "pass").all()) else "fail",
            "detail": str(int((checks["status"] != "pass").sum())),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    output_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]

    adapter = build_adapter(config)
    processed_summary = read_csv(ROOT / config["processed_scope_summary_path"])
    guard_manifest = read_json(ROOT / config["guard_manifest_path"])
    acceptance_manifest = read_json(ROOT / config["acceptance_manifest_path"])
    validation = validate_sample_dates(adapter, config)
    capability_checks = validate_capabilities(adapter, config)
    checks = build_acceptance_checks(config, validation, capability_checks, processed_summary, guard_manifest, acceptance_manifest)

    artifacts = {
        "adapter_validation": output_dir / "adapter_validation.csv",
        "capability_gate_check": output_dir / "capability_gate_check.csv",
        "acceptance_checks": output_dir / "acceptance_checks.csv",
        "adapter_report": output_dir / "adapter_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(validation, artifacts["adapter_validation"])
    write_csv(capability_checks, artifacts["capability_gate_check"])
    write_csv(checks, artifacts["acceptance_checks"])
    write_text(build_report(config, validation, capability_checks, checks, processed_summary), artifacts["adapter_report"])
    write_text(build_catalog(processed_summary, checks), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, checks)
    write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "acceptance_pass": bool((checks["status"] == "pass").all()),
        "raw_rows": int(processed_summary.loc[processed_summary["metric"] == "raw_rows", "value"].iloc[0]),
        "quarantined_rows": int(processed_summary.loc[processed_summary["metric"] == "quarantined_ohlc_rows", "value"].iloc[0]),
        "accepted_processed_rows": int(processed_summary.loc[processed_summary["metric"] == "accepted_processed_rows", "value"].iloc[0]),
        "data_decision": "accepted_daily_only_adapter_ready_for_limited_downstream_use",
        "model_decision": "no_model_promotion_adapter_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
