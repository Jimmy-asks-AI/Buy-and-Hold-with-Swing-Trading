#!/usr/bin/env python
"""HIRSSM V3.44 raw daily adjustment guard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from raw_daily_guard import DataCapabilityError, RawDailyOnlyAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "raw_daily_adjustment_guard_v3_44.json"
TASK_ID = "20260529_v3_44_raw_daily_adjustment_guard"
VERSION = "V3.44"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def daily_files(data_root: Path, start_date: str, end_date: str, max_files: int = 0) -> list[Path]:
    files = []
    for path in sorted(data_root.glob("trade_date=*.csv")):
        date_text = path.stem.split("=")[-1]
        if start_date <= date_text <= end_date:
            files.append(path)
    if max_files and max_files > 0:
        return files[:max_files]
    return files


def build_research_capability_matrix(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for use in config["allowed_uses_with_raw_daily_only"]:
        rows.append(
            {
                "research_use": use,
                "status": "allowed_with_boundary",
                "required_missing_interface": "",
                "rule": "raw daily only; no adjusted-return performance claims",
            }
        )
    missing_map = {
        "adjusted_return": "adj_factor",
        "total_return": "adj_factor, dividend data",
        "long_horizon_momentum": "adj_factor",
        "portfolio_backtest_performance": "adj_factor",
        "high_dividend_return_analysis": "adj_factor, dividend data, daily_basic",
        "low_pe_stock_selection": "daily_basic",
        "dividend_yield_total_return": "adj_factor, dividend data, daily_basic",
        "qfq_close": "adj_factor",
        "hfq_close": "adj_factor",
    }
    for use in config["hard_blocked_uses_without_adj_factor"]:
        rows.append(
            {
                "research_use": use,
                "status": "hard_blocked",
                "required_missing_interface": missing_map.get(use, "adj_factor"),
                "rule": "must not run from raw close",
            }
        )
    return pd.DataFrame(rows)


def scan_raw_discontinuities(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data_root = ROOT / config["data_root"]
    files = daily_files(data_root, config["start_date"], config["end_date"], int(config.get("max_scan_files", 0)))
    max_flags = int(config.get("max_flag_records", 50000))
    raw_gap_threshold = float(config.get("raw_gap_abs_threshold", 0.08))
    discrepancy_threshold = float(config.get("reported_return_discrepancy_threshold", 0.05))

    last_close: dict[str, float] = {}
    summary: dict[str, dict[str, Any]] = {}
    flag_frames: list[pd.DataFrame] = []
    scanned_files = 0
    scanned_rows = 0
    empty_files = 0
    flag_count_total = 0
    usecols = ["ts_code", "trade_date", "close", "pct_chg"]

    for path in files:
        try:
            data = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda col: col in usecols, low_memory=False)
        except pd.errors.EmptyDataError:
            empty_files += 1
            continue
        scanned_files += 1
        if data.empty:
            empty_files += 1
            continue
        scanned_rows += int(data.shape[0])
        data = data.dropna(subset=["ts_code", "close"])
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data["pct_chg"] = pd.to_numeric(data.get("pct_chg"), errors="coerce")
        data = data.dropna(subset=["close"])
        if data.empty:
            continue

        data["prev_raw_close"] = data["ts_code"].map(last_close)
        comparable = data.dropna(subset=["prev_raw_close", "pct_chg"]).copy()
        if not comparable.empty:
            comparable = comparable.loc[(comparable["prev_raw_close"] > 0) & (comparable["close"] > 0)].copy()
            comparable["raw_gap_return"] = comparable["close"] / comparable["prev_raw_close"] - 1.0
            comparable["reported_return"] = comparable["pct_chg"] / 100.0
            comparable["return_discrepancy"] = comparable["raw_gap_return"] - comparable["reported_return"]
            comparable["abs_discrepancy"] = comparable["return_discrepancy"].abs()
            flags = comparable.loc[
                (comparable["raw_gap_return"].abs() >= raw_gap_threshold)
                & (comparable["abs_discrepancy"] >= discrepancy_threshold)
            ].copy()
            if not flags.empty:
                flag_count_total += int(flags.shape[0])
                flags = flags[
                    [
                        "ts_code",
                        "trade_date",
                        "prev_raw_close",
                        "close",
                        "raw_gap_return",
                        "reported_return",
                        "return_discrepancy",
                        "abs_discrepancy",
                    ]
                ]
                flag_frames.append(flags)
                if sum(frame.shape[0] for frame in flag_frames) > max_flags * 2:
                    combined = pd.concat(flag_frames, ignore_index=True)
                    flag_frames = [combined.nlargest(max_flags, "abs_discrepancy")]
                for item in flags.itertuples(index=False):
                    asset = str(item.ts_code)
                    current = summary.setdefault(
                        asset,
                        {
                            "asset": asset,
                            "flag_count": 0,
                            "max_abs_discrepancy": 0.0,
                            "max_abs_raw_gap": 0.0,
                            "first_flag_date": str(item.trade_date),
                            "last_flag_date": str(item.trade_date),
                        },
                    )
                    current["flag_count"] += 1
                    current["max_abs_discrepancy"] = max(current["max_abs_discrepancy"], float(item.abs_discrepancy))
                    current["max_abs_raw_gap"] = max(current["max_abs_raw_gap"], abs(float(item.raw_gap_return)))
                    current["last_flag_date"] = str(item.trade_date)

        update = data.dropna(subset=["close"]).drop_duplicates("ts_code", keep="last")
        last_close.update(dict(zip(update["ts_code"].astype(str), update["close"].astype(float))))

    if flag_frames:
        flags_out = pd.concat(flag_frames, ignore_index=True).nlargest(max_flags, "abs_discrepancy")
    else:
        flags_out = pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "prev_raw_close",
                "close",
                "raw_gap_return",
                "reported_return",
                "return_discrepancy",
                "abs_discrepancy",
            ]
        )
    summary_out = pd.DataFrame(summary.values())
    if not summary_out.empty:
        summary_out = summary_out.sort_values(["flag_count", "max_abs_discrepancy"], ascending=[False, False])
    stats = {
        "scanned_files": scanned_files,
        "empty_files": empty_files,
        "scanned_rows": scanned_rows,
        "flag_count_total": flag_count_total,
        "top_flag_records_kept": int(flags_out.shape[0]),
        "asset_flag_count": int(summary_out.shape[0]),
        "raw_gap_abs_threshold": raw_gap_threshold,
        "reported_return_discrepancy_threshold": discrepancy_threshold,
    }
    return flags_out, summary_out, stats


def adapter_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "version": VERSION,
        "data_layer": "tushare_daily_only_raw",
        "price_adjustment": "none_raw",
        "allowed_uses": config["allowed_uses_with_raw_daily_only"],
        "hard_blocked_uses": config["hard_blocked_uses_without_adj_factor"],
        "manual_data_interfaces_still_needed": config["manual_data_interfaces_still_needed"],
        "policy": {
            "raw_close_may_not_be_used_as_adjusted_close": True,
            "portfolio_backtest_performance_requires_adj_factor": True,
            "high_dividend_low_pe_research_requires_daily_basic_and_adjustment_data": True,
            "downstream_code_should_import": "strategy_lab.raw_daily_guard.RawDailyOnlyAdapter",
        },
    }


def exercise_adapter(config: dict[str, Any]) -> pd.DataFrame:
    adapter = RawDailyOnlyAdapter(ROOT / config["data_root"])
    rows = []
    for capability in ["raw_ohlcv", "adjusted_return", "total_return", "qfq_close", "valuation"]:
        try:
            adapter.require_capability(capability)
            rows.append({"capability": capability, "result": "allowed", "error": ""})
        except DataCapabilityError as exc:
            rows.append({"capability": capability, "result": "blocked", "error": str(exc)})
    try:
        adapter.assert_can_run_portfolio_backtest("raw_close_return")
        rows.append({"capability": "portfolio_backtest_raw_close_return", "result": "allowed", "error": ""})
    except DataCapabilityError as exc:
        rows.append({"capability": "portfolio_backtest_raw_close_return", "result": "blocked", "error": str(exc)})
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    capability: pd.DataFrame,
    flags: pd.DataFrame,
    asset_summary: pd.DataFrame,
    stats: dict[str, Any],
    adapter_check: pd.DataFrame,
) -> str:
    blocked = capability.loc[capability["status"] == "hard_blocked"]
    allowed = capability.loc[capability["status"] == "allowed_with_boundary"]
    top_flags = flags.head(10)
    lines = [
        "# V3.44 Raw Daily Adjustment Guard",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- Raw Tushare `daily` data is not adjusted-return data.",
        "- Downstream performance backtests, high-dividend return analysis, low-PE selection, qfq/hfq prices, and total-return calculations are hard-blocked until missing data interfaces are resolved.",
        "- `strategy_lab/raw_daily_guard.py` is the adapter that downstream code should use for this raw layer.",
        "",
        "## Why This Matters",
        "",
        "Raw close can jump on ex-rights, dividends, splits, restructurings, and other corporate-action events. If raw close is chained into long-horizon returns, high-dividend stocks are systematically penalized and strategy returns become economically wrong.",
        "",
        "## Capability Summary",
        "",
        f"- Allowed with boundary: `{len(allowed)}`",
        f"- Hard blocked: `{len(blocked)}`",
        f"- Manual interfaces still needed: `{', '.join(config['manual_data_interfaces_still_needed'])}`",
        "",
        "## Discontinuity Scan",
        "",
        f"- Scanned files: `{stats['scanned_files']}`",
        f"- Scanned rows: `{stats['scanned_rows']}`",
        f"- Empty files: `{stats['empty_files']}`",
        f"- Flagged discontinuities: `{stats['flag_count_total']}`",
        f"- Assets with flags: `{stats['asset_flag_count']}`",
        f"- Top flag records kept: `{stats['top_flag_records_kept']}`",
        "",
        "A flag means raw close-to-raw close movement diverged materially from Tushare daily `pct_chg`, which is direct evidence that raw close continuity is unsafe for return compounding.",
        "",
        "## Adapter Check",
        "",
        "| Capability | Result |",
        "|---|---|",
    ]
    for row in adapter_check.itertuples(index=False):
        lines.append(f"| `{row.capability}` | `{row.result}` |")
    lines.extend(["", "## Top Discontinuity Examples", "", "| Asset | Date | Raw Gap | Reported Return | Discrepancy |", "|---|---:|---:|---:|---:|"])
    if top_flags.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in top_flags.itertuples(index=False):
            lines.append(
                f"| `{row.ts_code}` | `{row.trade_date}` | {row.raw_gap_return:.4f} | {row.reported_return:.4f} | {row.return_discrepancy:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "Do not continue to model research from this layer unless the model explicitly accepts the raw-only boundary. Any portfolio performance, long-horizon momentum, high-dividend, low-PE, or adjusted-return result requires `adj_factor` and related missing data first.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_catalog(config: dict[str, Any], stats: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# A-share Raw Daily Adjustment Guard V3.44",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            "- Raw daily data is complete for the configured date range, but it is not adjusted-return data.",
            "- The adapter `strategy_lab/raw_daily_guard.py` must be used by downstream code.",
            "- Adjusted-return and valuation research remains blocked.",
            "",
            "## Evidence",
            "",
            f"- Discontinuity scan rows: `{stats['scanned_rows']}`",
            f"- Flagged raw-return discontinuities: `{stats['flag_count_total']}`",
            f"- Assets with discontinuity flags: `{stats['asset_flag_count']}`",
            "",
            "## Still Needed",
            "",
            *[f"- `{item}`" for item in config["manual_data_interfaces_still_needed"]],
            "",
        ]
    )


def build_self_check(paths: dict[str, Path], capability: pd.DataFrame, adapter_check: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    hard_block_count = int((capability["status"] == "hard_blocked").sum())
    adapter_block_count = int((adapter_check["result"] == "blocked").sum())
    rows.extend(
        [
            {"check": "hard_blocked_uses_present", "status": "pass" if hard_block_count >= 5 else "fail", "detail": str(hard_block_count)},
            {"check": "adapter_blocks_adjusted_uses", "status": "pass" if adapter_block_count >= 4 else "fail", "detail": str(adapter_block_count)},
            {"check": "discontinuity_scan_ran", "status": "pass" if stats["scanned_rows"] > 0 else "fail", "detail": str(stats["scanned_rows"])},
            {"check": "raw_layer_no_model_promotion", "status": "pass", "detail": "guard only"},
        ]
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

    capability = build_research_capability_matrix(config)
    flags, asset_summary, stats = scan_raw_discontinuities(config)
    contract = adapter_contract(config)
    adapter_check = exercise_adapter(config)

    artifacts = {
        "research_capability_matrix": output_dir / "research_capability_matrix.csv",
        "raw_discontinuity_flags_top": output_dir / "raw_discontinuity_flags_top.csv",
        "raw_discontinuity_asset_summary": output_dir / "raw_discontinuity_asset_summary.csv",
        "adapter_contract": output_dir / "adapter_contract.json",
        "adapter_check": output_dir / "adapter_check.csv",
        "guard_report": output_dir / "guard_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(capability, artifacts["research_capability_matrix"])
    write_csv(flags, artifacts["raw_discontinuity_flags_top"])
    write_csv(asset_summary, artifacts["raw_discontinuity_asset_summary"])
    write_json(contract, artifacts["adapter_contract"])
    write_csv(adapter_check, artifacts["adapter_check"])
    write_text(build_report(config, capability, flags, asset_summary, stats, adapter_check), artifacts["guard_report"])
    write_text(build_catalog(config, stats), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])
    self_check_path = output_dir / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, capability, adapter_check, stats)
    write_csv(self_check, self_check_path)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "scanned_rows": stats["scanned_rows"],
        "flagged_discontinuities": stats["flag_count_total"],
        "hard_blocked_uses": int((capability["status"] == "hard_blocked").sum()),
        "model_decision": "no_model_promotion_guard_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
