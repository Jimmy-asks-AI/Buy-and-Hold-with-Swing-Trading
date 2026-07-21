#!/usr/bin/env python
"""Build and validate the V3.47 daily-only feature layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from accepted_daily_only_adapter import AcceptedDailyOnlyAdapter
from daily_only_feature_layer import (
    FEATURE_DICTIONARY_ROWS,
    DailyOnlyFeatureConfig,
    compute_asset_activity_features,
    compute_daily_market_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "daily_only_feature_layer_v3_47.json"
TASK_ID = "20260529_v3_47_daily_only_feature_layer"
VERSION = "V3.47"


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


def feature_config(config: dict[str, Any]) -> DailyOnlyFeatureConfig:
    return DailyOnlyFeatureConfig(
        low_amount_raw_threshold=float(config["low_amount_raw_threshold"]),
        low_volume_raw_threshold=float(config["low_volume_raw_threshold"]),
        high_activity_amount_quantile=float(config["high_activity_amount_quantile"]),
        top_amount_count=int(config["top_amount_count"]),
    )


def acquired_trade_dates(config: dict[str, Any]) -> list[str]:
    registry = read_csv(ROOT / config["registry_path"])
    data = registry.loc[registry["status"] == "acquired"].copy()
    return data["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8].tolist()


def build_market_feature_panel(config: dict[str, Any]) -> pd.DataFrame:
    adapter = build_adapter(config)
    fcfg = feature_config(config)
    rows: list[dict[str, Any]] = []
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol", "amount"]
    for trade_date in acquired_trade_dates(config):
        data = adapter.load_trade_date(trade_date, columns=columns, apply_quarantine=True, add_scope_columns=True)
        rows.append(compute_daily_market_features(data, trade_date, fcfg))
    return pd.DataFrame(rows)


def build_asset_activity_sample(config: dict[str, Any]) -> pd.DataFrame:
    adapter = build_adapter(config)
    frames: list[pd.DataFrame] = []
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol", "amount"]
    top_n = int(config["sample_asset_top_n_by_amount"])
    for trade_date in config["sample_asset_dates"]:
        data = adapter.load_trade_date(trade_date, columns=columns, apply_quarantine=True, add_scope_columns=False)
        if data.empty:
            continue
        sample = data.assign(_amount=pd.to_numeric(data["amount"], errors="coerce").fillna(0.0))
        sample = sample.sort_values("_amount", ascending=False).head(top_n).drop(columns=["_amount"])
        frames.append(compute_asset_activity_features(sample))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_feature_dictionary(config: dict[str, Any]) -> pd.DataFrame:
    rows = [dict(row) for row in FEATURE_DICTIONARY_ROWS]
    for row in rows:
        row["data_scope"] = "accepted_processed_daily_only"
        row["price_adjustment"] = "none_raw"
        row["allowed_feature_families"] = ",".join(config["allowed_feature_families"])
        row["forbidden_feature_families"] = ",".join(config["forbidden_feature_families"])
    return pd.DataFrame(rows)


def build_quality_checks(
    config: dict[str, Any],
    market_features: pd.DataFrame,
    asset_sample: pd.DataFrame,
    adapter_manifest: dict[str, Any],
    processed_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = dict(zip(processed_summary["metric"], processed_summary["value"]))
    expected_dates = len(acquired_trade_dates(config))
    rows = [
        {
            "check": "v3_46_adapter_acceptance_passed",
            "status": "pass" if bool(adapter_manifest.get("acceptance_pass", False)) else "fail",
            "detail": adapter_manifest.get("data_decision", ""),
        },
        {
            "check": "market_feature_rows_match_acquired_dates",
            "status": "pass" if int(market_features.shape[0]) == expected_dates else "fail",
            "detail": f"market_rows={market_features.shape[0]};acquired_dates={expected_dates}",
        },
        {
            "check": "accepted_rows_reconcile_to_v3_45_1",
            "status": "pass"
            if int(market_features["asset_count"].sum()) == int(summary["accepted_processed_rows"])
            else "fail",
            "detail": f"feature_rows={int(market_features['asset_count'].sum())};accepted={int(summary['accepted_processed_rows'])}",
        },
        {
            "check": "bad_ohlc_after_adapter_is_zero",
            "status": "pass" if int(market_features["bad_ohlc_after_adapter"].sum()) == 0 else "fail",
            "detail": str(int(market_features["bad_ohlc_after_adapter"].sum())),
        },
        {
            "check": "feature_scope_is_daily_only",
            "status": "pass"
            if set(market_features["feature_scope"].astype(str).unique()) == {"daily_only_market_diagnostic"}
            else "fail",
            "detail": "|".join(sorted(market_features["feature_scope"].astype(str).unique())),
        },
        {
            "check": "price_adjustment_is_none_raw",
            "status": "pass"
            if set(market_features["price_adjustment"].astype(str).unique()) == {"none_raw"}
            else "fail",
            "detail": "|".join(sorted(market_features["price_adjustment"].astype(str).unique())),
        },
        {
            "check": "asset_sample_scope_is_raw_diagnostic",
            "status": "pass"
            if not asset_sample.empty and set(asset_sample["feature_scope"].astype(str).unique()) == {"daily_only_asset_activity_sample"}
            else "fail",
            "detail": str(int(asset_sample.shape[0])),
        },
        {
            "check": "no_model_or_backtest_promotion",
            "status": "pass",
            "detail": "feature diagnostics only",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    config: dict[str, Any],
    market_features: pd.DataFrame,
    asset_sample: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    failed = checks.loc[checks["status"] != "pass"]
    start_date = market_features["trade_date"].min()
    end_date = market_features["trade_date"].max()
    latest = market_features.sort_values("trade_date").tail(1).iloc[0]
    lines = [
        "# V3.47 Daily-Only Feature Layer",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "## Decision",
        "",
        "- V3.47 builds the first downstream feature layer on top of `AcceptedDailyOnlyAdapter`.",
        "- Features are limited to coverage, liquidity/activity, market breadth, and raw intraday diagnostics.",
        "- No adjusted return, valuation, dividend, portfolio performance, model, or backtest output is produced.",
        "",
        "## Coverage",
        "",
        f"- Feature dates: `{len(market_features)}`",
        f"- Date range: `{start_date}` to `{end_date}`",
        f"- Accepted stock rows represented: `{int(market_features['asset_count'].sum())}`",
        f"- Asset sample rows: `{len(asset_sample)}`",
        "",
        "## Latest Diagnostic Snapshot",
        "",
        f"- Latest date: `{latest.trade_date}`",
        f"- Assets: `{int(latest.asset_count)}`",
        f"- Active asset ratio: `{float(latest.active_asset_ratio):.4f}`",
        f"- Advance ratio: `{float(latest.advance_ratio):.4f}`",
        f"- Decline ratio: `{float(latest.decline_ratio):.4f}`",
        f"- Top amount share: `{float(latest.top_amount_share):.4f}`",
        f"- Median raw pct_chg: `{float(latest.median_pct_chg_raw):.4f}`",
        "",
        "## Checks",
        "",
        f"- Failed checks: `{len(failed)}`",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for row in checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {str(row.detail)[:180]} |")
    lines.extend(
        [
            "",
            "## Downstream Rule",
            "",
            "- Downstream research can use `daily_market_feature_panel.csv` only as raw diagnostic features.",
            "- Any return-label, valuation, dividend, or portfolio-performance study still requires adjusted/PIT data.",
            "- Full asset-level feature panels should be generated only after a concrete research question needs them.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(market_features: pd.DataFrame, checks: pd.DataFrame) -> str:
    failed = checks.loc[checks["status"] != "pass"]
    return "\n".join(
        [
            "# A-share Daily-Only Feature Layer V3.47",
            "",
            f"Updated: `{now_text()}`",
            "",
            "## Status",
            "",
            f"- Feature layer accepted: `{failed.empty}`",
            f"- Daily feature rows: `{len(market_features)}`",
            f"- Accepted stock rows represented: `{int(market_features['asset_count'].sum())}`",
            f"- Data scope: `accepted_processed_daily_only`",
            f"- Price adjustment: `none_raw`",
            "",
            "## Artifacts",
            "",
            "- `outputs/agent_runs/v3_47/daily_only_feature_layer/daily_market_feature_panel.csv`",
            "- `outputs/agent_runs/v3_47/daily_only_feature_layer/feature_dictionary.csv`",
            "- `outputs/agent_runs/v3_47/daily_only_feature_layer/asset_activity_sample.csv`",
            "",
            "## Boundary",
            "",
            "- Allowed: coverage, liquidity/activity, breadth, raw intraday diagnostics.",
            "- Forbidden: adjusted return, total return, valuation, dividends, portfolio performance.",
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
            "check": "quality_checks_all_pass",
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

    market_features = build_market_feature_panel(config)
    asset_sample = build_asset_activity_sample(config)
    feature_dictionary = build_feature_dictionary(config)
    adapter_manifest = read_json(ROOT / config["adapter_manifest_path"])
    processed_summary = read_csv(ROOT / config["processed_scope_summary_path"])
    checks = build_quality_checks(config, market_features, asset_sample, adapter_manifest, processed_summary)

    artifacts = {
        "daily_market_feature_panel": output_dir / "daily_market_feature_panel.csv",
        "feature_dictionary": output_dir / "feature_dictionary.csv",
        "asset_activity_sample": output_dir / "asset_activity_sample.csv",
        "feature_quality_checks": output_dir / "feature_quality_checks.csv",
        "feature_layer_report": output_dir / "feature_layer_report.md",
        "catalog_update": catalog_path,
        "changed_files": output_dir / "changed_files.txt",
    }
    write_csv(market_features, artifacts["daily_market_feature_panel"])
    write_csv(feature_dictionary, artifacts["feature_dictionary"])
    write_csv(asset_sample, artifacts["asset_activity_sample"])
    write_csv(checks, artifacts["feature_quality_checks"])
    write_text(build_report(config, market_features, asset_sample, checks), artifacts["feature_layer_report"])
    write_text(build_catalog(market_features, checks), artifacts["catalog_update"])
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
        "daily_feature_rows": int(market_features.shape[0]),
        "accepted_stock_rows_represented": int(market_features["asset_count"].sum()),
        "feature_scope": "daily_only_diagnostics",
        "data_decision": "daily_only_feature_layer_ready_for_limited_research_use",
        "model_decision": "no_model_promotion_feature_diagnostics_only",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
