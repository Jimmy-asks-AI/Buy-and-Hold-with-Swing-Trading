"""Cross-provider validation for stock trade-state and valuation history.

The validator uses three independent observations:

* accepted Tushare raw daily data for traded close coverage and equality;
* JoinQuant's visible trial window for paused/ST/high-limit/low-limit checks;
* Eastmoney valuation history for PE/PB/free-float-market-cap checks.

Partial source collection remains observation-only. A PASS requires complete
builder coverage, minimum validation populations and all fixed thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_stock_market_history_builder import (
    BACKTEST_START,
    DIVIDEND_PATH,
    MASTER_PATH,
    RAW_DIR,
    ROOT,
    _cache_paths,
    _sha256,
    load_lifecycles,
    normalise_baostock_history,
)


BUILDER_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_market_history_builder_latest.json"
ADJUSTMENT_FACTOR_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_adjustment_factor.csv"
ADJUSTMENT_FACTOR_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_adjustment_builder_latest.json"
TUSHARE_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
TUSHARE_ACCEPTANCE = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine" / "agent_run_manifest.json"
)
TUSHARE_QUARANTINE = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine" / "ohlc_quarantine_rows.csv"
)
TUSHARE_FILE_ACCEPTANCE = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine" / "file_acceptance_report.csv"
)
TUSHARE_REFRESH_MANIFEST = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "tushare_daily_refresh_latest.json"
)
CREDENTIALS_PATH = ROOT / "configs" / "data_credentials.json"
CURRENT_SNAPSHOT = ROOT / "data_raw" / "long_hold_v4" / "research_snapshot.csv"
VALIDATION_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "validation_sources"
JQ_RAW_DIR = VALIDATION_RAW_DIR / "joinquant_trade_state"
JQ_VALUATION_RAW_DIR = VALIDATION_RAW_DIR / "joinquant_valuation"
EM_RAW_DIR = VALIDATION_RAW_DIR / "eastmoney_valuation"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "stock_market_history_validation"
TUSHARE_CHECKS = OUTPUT_DIR / "tushare_price_checks.csv"
TUSHARE_VALUATION_CHECKS = OUTPUT_DIR / "tushare_valuation_price_checks.csv"
JQ_CHECKS = OUTPUT_DIR / "joinquant_trade_state_checks.csv"
JQ_VALUATION_CHECKS = OUTPUT_DIR / "joinquant_valuation_checks.csv"
EM_CHECKS = OUTPUT_DIR / "eastmoney_valuation_checks.csv"
EM_DELISTED_CHECKS = OUTPUT_DIR / "eastmoney_delisted_valuation_checks.csv"
YIELD_CHECKS = OUTPUT_DIR / "current_dividend_yield_checks.csv"
SOURCE_OBSERVATIONS = OUTPUT_DIR / "source_observations.csv"
WARNINGS = OUTPUT_DIR / "validation_warnings.csv"
EXCEPTIONS = OUTPUT_DIR / "validation_exceptions.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
TRADE_EXCEPTIONS = OUTPUT_DIR / "trade_state_validation_exceptions.csv"
VALUATION_EXCEPTIONS = OUTPUT_DIR / "valuation_validation_exceptions.csv"
TRADE_REPORT_PATH = OUTPUT_DIR / "trade_state_validation_report.json"
VALUATION_REPORT_PATH = OUTPUT_DIR / "valuation_validation_report.json"
TRADE_MANIFEST_PATH = OUTPUT_DIR / "trade_state_run_manifest.json"
VALUATION_MANIFEST_PATH = OUTPUT_DIR / "valuation_run_manifest.json"

FORCED_SAMPLE = ("000001", "000002", "000004", "000005", "600000", "600355", "300750", "688981")
JQ_START = pd.Timestamp("2025-05-01")
JQ_END = pd.Timestamp("2026-04-01")

THRESHOLDS = {
    "tushare_min_checks": 100_000,
    "tushare_coverage_min": 0.995,
    "tushare_close_match_min": 0.999,
    "tushare_pre_close_match_min": 0.999,
    "pre_close_scope_holdout_min_assets": 32,
    "tushare_relevant_close_match_min": 0.999,
    "tushare_max_close_absolute_error": 0.05,
    "joinquant_min_state_checks": 5_000,
    "joinquant_source_coverage_min": 0.95,
    "joinquant_paused_match_min": 0.995,
    "joinquant_st_match_min": 0.995,
    "joinquant_limit_match_min": 0.995,
    "joinquant_min_valuation_checks": 500,
    "joinquant_valuation_pe_median_abs_relative_max": 0.02,
    "joinquant_valuation_pe_p95_abs_relative_max": 0.10,
    "joinquant_valuation_pb_median_abs_relative_max": 0.02,
    "joinquant_valuation_pb_p95_abs_relative_max": 0.10,
    "joinquant_valuation_cap_median_abs_relative_max": 0.01,
    "joinquant_valuation_cap_p95_abs_relative_max": 0.03,
    "eastmoney_source_coverage_min": 0.90,
    "eastmoney_min_checks": 2_000,
    "delisted_valuation_min_cross_source_checks": 100,
    "delisted_valuation_min_cross_source_assets": 5,
    "eastmoney_pe_median_abs_relative_max": 0.02,
    "eastmoney_pe_p95_abs_relative_max": 0.10,
    "eastmoney_pb_median_abs_relative_max": 0.05,
    "eastmoney_pb_p95_abs_relative_max": 0.15,
    "eastmoney_cap_median_abs_relative_max": 0.01,
    "eastmoney_cap_p95_abs_relative_max": 0.03,
    "yield_min_checks": 10,
    "yield_median_absolute_error_max": 0.005,
    "yield_p95_absolute_error_max": 0.02,
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_validation_assets(available_assets: set[str], max_assets: int) -> list[str]:
    available = sorted(str(asset).zfill(6) for asset in available_assets)
    forced = [asset for asset in FORCED_SAMPLE if asset in available]
    remainder = [asset for asset in available if asset not in forced]
    remainder.sort(key=lambda asset: hashlib.sha256(f"long-hold-v4:{asset}".encode()).hexdigest())
    return (forced + remainder)[: max(0, max_assets)]


def _load_builder_manifest() -> dict[str, Any]:
    if not BUILDER_MANIFEST.is_file():
        raise FileNotFoundError(BUILDER_MANIFEST)
    return json.loads(BUILDER_MANIFEST.read_text(encoding="utf-8"))


def _read_asset_subset(path: Path, assets: set[str], chunksize: int = 250_000) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    normalised_assets = {str(asset).zfill(6) for asset in assets}
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, dtype={"asset": str}, low_memory=False, chunksize=chunksize):
        chunk["asset"] = chunk["asset"].astype(str).str.zfill(6)
        selected = chunk[chunk["asset"].isin(normalised_assets)].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _available_builder_assets(manifest: dict[str, Any], scope: str = "both") -> set[str]:
    if scope not in {"trade_state", "valuation", "both"}:
        raise ValueError(f"unsupported builder asset scope: {scope}")
    summary_path_value = manifest.get("asset_summary_path")
    if not summary_path_value:
        raise ValueError("builder manifest has no asset-level derived summary")
    summary_path = ROOT / str(summary_path_value)
    expected_hash = str(manifest.get("asset_summary_sha256", ""))
    if not expected_hash or _sha256(summary_path) != expected_hash:
        raise ValueError("builder asset summary failed manifest hash validation")
    summary = pd.read_csv(summary_path, dtype={"asset": str})
    required = {"asset", "status", "trade_rows", "valuation_valid_rows"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"builder asset summary missing columns: {missing}")
    summary["asset"] = summary["asset"].astype(str).str.zfill(6)
    if summary["asset"].duplicated().any():
        raise ValueError("builder asset summary contains duplicate assets")
    trade_status = summary.get("trade_status", summary["status"]).astype(str)
    valuation_status = summary.get("valuation_status", summary["status"]).astype(str)
    trade_eligible = trade_status.eq("completed") & pd.to_numeric(
        summary["trade_rows"], errors="coerce"
    ).gt(0)
    valuation_eligible = valuation_status.eq("completed") & pd.to_numeric(
        summary["valuation_valid_rows"], errors="coerce"
    ).gt(0)
    eligible = {
        "trade_state": trade_eligible,
        "valuation": valuation_eligible,
        "both": trade_eligible & valuation_eligible,
    }[scope]
    return set(summary.loc[eligible, "asset"].astype(str).str.zfill(6))


def _load_builder_output(
    manifest: dict[str, Any],
    dataset_key: str,
    assets: set[str],
) -> pd.DataFrame:
    if dataset_key not in {"trade_state", "valuation"}:
        raise ValueError(f"unsupported builder output: {dataset_key}")
    descriptor = manifest[dataset_key]
    path = ROOT / descriptor["output_path"]
    if _sha256(path) != descriptor.get("output_sha256"):
        raise ValueError(f"builder {dataset_key} output failed manifest hash validation")
    frame = _read_asset_subset(path, assets)
    if frame.empty:
        raise ValueError(f"builder {dataset_key} output subset is empty for selected validation assets")
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _load_builder_outputs(
    manifest: dict[str, Any],
    assets: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _load_builder_output(manifest, "trade_state", assets),
        _load_builder_output(manifest, "valuation", assets),
    )


def _load_sample_histories(assets: list[str], as_of: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    lifecycles = load_lifecycles(as_of=as_of).set_index("asset")
    parts: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    for asset in assets:
        if asset not in lifecycles.index:
            continue
        row = lifecycles.loc[asset]
        data_path, meta_path = _cache_paths(asset)
        if not data_path.is_file() or not meta_path.is_file():
            continue
        raw = pd.read_csv(data_path, compression="gzip", low_memory=False)
        parts.append(normalise_baostock_history(raw, asset, row.list_date, row.delist_date, as_of))
        inputs.append({"path": _relative(data_path), "sha256": _sha256(data_path)})
        inputs.append({"path": _relative(meta_path), "sha256": _sha256(meta_path)})
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), inputs


def _load_quarantine() -> set[tuple[str, str]]:
    frame = pd.read_csv(TUSHARE_QUARANTINE, dtype={"ts_code": str, "trade_date": str})
    dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    return set(zip(dates, frame["ts_code"].astype(str)))


def _accepted_empty_tushare_dates() -> set[str]:
    frame = pd.read_csv(TUSHARE_FILE_ACCEPTANCE, dtype={"trade_date": str})
    required = {"trade_date", "file_exists", "readable", "rows", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Tushare file acceptance report missing columns: {missing}")
    empty = frame[
        frame["status"].astype(str).eq("empty")
        & pd.to_numeric(frame["rows"], errors="coerce").eq(0)
        & frame["file_exists"].astype(str).str.lower().eq("true")
        & frame["readable"].astype(str).str.lower().eq("true")
    ]
    return set(empty["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8])


def _validate_tushare_refresh_manifest(as_of: pd.Timestamp) -> dict[str, str]:
    if not TUSHARE_REFRESH_MANIFEST.is_file():
        raise FileNotFoundError(TUSHARE_REFRESH_MANIFEST)
    payload = json.loads(TUSHARE_REFRESH_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("qualification_status") != "REFRESH_COMPLETE_RAW_DAILY_ONLY":
        raise ValueError("Tushare daily refresh is not complete")
    if pd.Timestamp(payload.get("as_of_date")).normalize() != pd.Timestamp(as_of).normalize():
        raise ValueError("Tushare daily refresh as-of date does not match validation date")
    failures = []
    for item in payload.get("outputs", []):
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            failures.append(str(item.get("path", "")))
    if failures or len(payload.get("outputs", [])) != int(payload.get("target_trade_dates", -1)):
        raise ValueError(f"Tushare daily refresh output validation failed: {failures[:10]}")
    code_path = ROOT / str(payload.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(payload.get("code_sha256", "")):
        raise ValueError("Tushare daily refresh code hash does not match")
    return {"path": _relative(TUSHARE_REFRESH_MANIFEST), "sha256": _sha256(TUSHARE_REFRESH_MANIFEST)}


def collect_tushare_sample(assets: list[str]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    asset_set = set(assets)
    quarantine = _load_quarantine()
    accepted_empty_dates = _accepted_empty_tushare_dates()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for path in sorted(TUSHARE_DIR.glob("trade_date=*.csv")):
        inputs.append({"path": _relative(path), "sha256": _sha256(path)})
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            required = {"ts_code", "trade_date", "close", "pre_close", "pct_chg"}
            missing = sorted(required.difference(reader.fieldnames))
            if missing:
                file_date = path.stem.removeprefix("trade_date=").replace("-", "")[:8]
                if file_date in accepted_empty_dates and next(reader, None) is None:
                    continue
                raise ValueError(f"{path.name} missing Tushare columns: {missing}")
            for row in reader:
                asset = str(row["ts_code"])[:6]
                date_key = str(row["trade_date"]).replace("-", "")[:8]
                if asset not in asset_set or (date_key, str(row["ts_code"])) in quarantine:
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(date_key),
                        "asset": asset,
                        "ts_close": pd.to_numeric(row["close"], errors="coerce"),
                        "ts_pre_close": pd.to_numeric(row["pre_close"], errors="coerce"),
                        "ts_pct_chg": pd.to_numeric(row["pct_chg"], errors="coerce"),
                    }
                )
    return pd.DataFrame(rows), inputs


def compare_tushare_prices(
    baostock: pd.DataFrame,
    tushare: pd.DataFrame,
    validation_start: pd.Timestamp = BACKTEST_START,
    relevant_price_keys: set[tuple[pd.Timestamp, str]] | None = None,
    relevant_pre_close_keys: set[tuple[pd.Timestamp, str]] | None = None,
    adjustment_ratios: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bs = baostock.copy()
    bs = bs[bs["tradestatus"].eq(1)].copy()
    bs = bs.rename(columns={"close": "bs_close", "preclose": "bs_pre_close", "pctChg": "bs_pct_chg"})
    for column in ("bs_close", "bs_pre_close", "bs_pct_chg"):
        bs[column] = pd.to_numeric(bs[column], errors="coerce")
    if tushare.empty:
        return pd.DataFrame(), {
            "validation_start": pd.Timestamp(validation_start).date().isoformat(),
            "expected_rows": int(bs["date"].ge(validation_start).sum()),
            "matched_rows": 0,
            "coverage_ratio": 0.0,
            "close_match_ratio": 0.0,
            "pre_close_match_ratio": 0.0,
            "pre_close_checks": 0,
            "pre_close_excluded_rows": 0,
            "excluded_pre_close_mismatch_rows": 0,
            "relevant_close_checks": 0,
            "relevant_close_match_ratio": 0.0,
            "maximum_close_absolute_error": None,
            "large_close_mismatch_rows": 0,
            "maximum_relevant_close_absolute_error": None,
            "large_relevant_close_mismatch_rows": 0,
            "maximum_pre_close_absolute_error": None,
            "maximum_diagnostic_pre_close_absolute_error": None,
        }
    start = max(pd.Timestamp(validation_start).normalize(), pd.Timestamp(tushare["date"].min()).normalize())
    end = tushare["date"].max()
    expected = bs[bs["date"].between(start, end)].copy()
    checks = expected.merge(tushare, on=["date", "asset"], how="left", validate="one_to_one")
    if adjustment_ratios is not None and not adjustment_ratios.empty:
        ratios = adjustment_ratios[["date", "asset", "adjustment_ratio"]].copy()
        checks = checks.merge(ratios, on=["date", "asset"], how="left", validate="one_to_one")
    else:
        checks["adjustment_ratio"] = 1.0
    checks["adjustment_ratio"] = pd.to_numeric(checks["adjustment_ratio"], errors="coerce").fillna(1.0)
    if checks["adjustment_ratio"].le(0).any():
        raise ValueError("adjustment ratios must be positive")
    checks["source_row_matched"] = checks["ts_close"].notna()
    checks["close_absolute_error"] = (checks["bs_close"] - checks["ts_close"]).abs()
    checks["close_match"] = checks["source_row_matched"] & checks["close_absolute_error"].le(0.011)
    checks["raw_pre_close_absolute_error"] = (checks["bs_pre_close"] - checks["ts_pre_close"]).abs()
    checks["factor_aligned_pre_close"] = checks["ts_pre_close"] / checks["adjustment_ratio"]
    checks["factor_aligned_pre_close_absolute_error"] = (
        checks["bs_pre_close"] - checks["factor_aligned_pre_close"]
    ).abs()
    checks["factor_alignment_selected"] = checks["adjustment_ratio"].ne(1.0) & checks[
        "factor_aligned_pre_close_absolute_error"
    ].lt(checks["raw_pre_close_absolute_error"])
    checks["ts_pre_close_aligned"] = checks["ts_pre_close"].where(
        ~checks["factor_alignment_selected"], checks["factor_aligned_pre_close"]
    )
    checks["pre_close_absolute_error"] = (checks["bs_pre_close"] - checks["ts_pre_close_aligned"]).abs()
    checks["pre_close_match"] = checks["ts_pre_close"].notna() & checks["pre_close_absolute_error"].le(0.011)
    if relevant_pre_close_keys is None:
        checks["pre_close_gate_relevant"] = True
    else:
        normalised_pre_close_keys = {
            (pd.Timestamp(date).normalize(), str(asset).zfill(6)) for date, asset in relevant_pre_close_keys
        }
        checks["pre_close_gate_relevant"] = [
            (pd.Timestamp(date).normalize(), str(asset).zfill(6)) in normalised_pre_close_keys
            for date, asset in zip(checks["date"], checks["asset"])
        ]
    if relevant_price_keys is None:
        checks["relevant_price_check"] = True
    else:
        normalised_keys = {(pd.Timestamp(date).normalize(), str(asset).zfill(6)) for date, asset in relevant_price_keys}
        checks["relevant_price_check"] = [
            (pd.Timestamp(date).normalize(), str(asset).zfill(6)) in normalised_keys
            for date, asset in zip(checks["date"], checks["asset"])
        ]
    relevant = checks["source_row_matched"] & checks["relevant_price_check"]
    pre_close_relevant = checks["source_row_matched"] & checks["pre_close_gate_relevant"]
    pre_close_excluded = checks["source_row_matched"] & ~checks["pre_close_gate_relevant"]
    matched = int(checks["source_row_matched"].sum())
    relevant_checks = int(relevant.sum())
    pre_close_checks = int(pre_close_relevant.sum())
    return checks, {
        "validation_start": start.date().isoformat(),
        "expected_rows": int(len(checks)),
        "matched_rows": matched,
        "coverage_ratio": round(matched / len(checks), 8) if len(checks) else 0.0,
        "close_match_ratio": round(float(checks.loc[checks["source_row_matched"], "close_match"].mean()), 8)
        if matched
        else 0.0,
        "close_mismatch_rows": int((checks["source_row_matched"] & ~checks["close_match"]).sum()),
        "pre_close_match_ratio": round(float(checks.loc[pre_close_relevant, "pre_close_match"].mean()), 8)
        if pre_close_checks
        else 0.0,
        "pre_close_checks": pre_close_checks,
        "pre_close_excluded_rows": int(pre_close_excluded.sum()),
        "excluded_pre_close_mismatch_rows": int((pre_close_excluded & ~checks["pre_close_match"]).sum()),
        "relevant_close_checks": relevant_checks,
        "relevant_close_match_ratio": round(float(checks.loc[relevant, "close_match"].mean()), 8)
        if relevant_checks
        else 0.0,
        "maximum_close_absolute_error": round(
            float(checks.loc[checks["source_row_matched"], "close_absolute_error"].max()), 8
        )
        if matched
        else None,
        "large_close_mismatch_rows": int(
            (
                checks["source_row_matched"]
                & checks["close_absolute_error"].gt(THRESHOLDS["tushare_max_close_absolute_error"])
            ).sum()
        ),
        "maximum_relevant_close_absolute_error": round(float(checks.loc[relevant, "close_absolute_error"].max()), 8)
        if relevant_checks
        else None,
        "large_relevant_close_mismatch_rows": int(
            (relevant & checks["close_absolute_error"].gt(THRESHOLDS["tushare_max_close_absolute_error"])).sum()
        ),
        "maximum_pre_close_absolute_error": round(
            float(checks.loc[pre_close_relevant, "pre_close_absolute_error"].max()), 8
        )
        if pre_close_checks
        else None,
        "maximum_diagnostic_pre_close_absolute_error": round(
            float(checks.loc[checks["source_row_matched"], "pre_close_absolute_error"].max()), 8
        )
        if matched
        else None,
        "maximum_raw_pre_close_absolute_error": round(
            float(checks.loc[checks["source_row_matched"], "raw_pre_close_absolute_error"].max()), 8
        )
        if matched
        else None,
        "corporate_action_ratio_rows": int(checks["adjustment_ratio"].ne(1.0).sum()),
        "pre_close_factor_aligned_rows": int(checks["factor_alignment_selected"].sum()),
    }


def build_adjustment_ratios(factors: pd.DataFrame) -> pd.DataFrame:
    required = {"asset", "effective_date", "adj_factor"}
    missing = sorted(required.difference(factors.columns))
    if missing:
        raise ValueError(f"adjustment factor input missing columns: {missing}")
    frame = factors[["asset", "effective_date", "adj_factor"]].copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    frame = frame.dropna(subset=["asset", "date", "adj_factor"]).sort_values(["asset", "date"])
    if frame.duplicated(["date", "asset"]).any() or frame["adj_factor"].le(0).any():
        raise ValueError("adjustment factors contain duplicate keys or non-positive values")
    previous = frame.groupby("asset")["adj_factor"].shift(1)
    frame["adjustment_ratio"] = (frame["adj_factor"] / previous).fillna(1.0)
    return frame[["date", "asset", "adjustment_ratio"]]


def _jq_code(asset: str, exchange: str) -> str:
    suffix = {"SSE": "XSHG", "SZSE": "XSHE"}.get(exchange)
    if suffix is None:
        raise ValueError(f"unsupported JoinQuant exchange: {exchange}")
    return f"{asset}.{suffix}"


def _jq_cache_path(asset: str) -> Path:
    return JQ_RAW_DIR / f"{asset}_{JQ_START.date()}_{JQ_END.date()}.csv.gz"


def _as_nullable_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    return mapped.astype("boolean")


def _provider_error(source: str, stage: str, exc: Exception, asset: str = "") -> dict[str, str]:
    return {
        "source": source,
        "stage": stage,
        "asset": asset,
        "error_type": type(exc).__name__,
    }


def collect_joinquant_sample(
    assets: list[str], as_of: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    lifecycles = load_lifecycles(as_of=as_of).set_index("asset")
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    try:
        credentials = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8-sig"))["joinquant"]
        from jqdatasdk import auth, get_extras, get_price

        auth(credentials["username"], credentials["password"])
    except Exception as exc:  # Provider/auth failures are evidence, not a reason to lose the whole report.
        errors.append(_provider_error("joinquant", "authenticate", exc))
        return pd.DataFrame(), inputs, errors

    JQ_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        if asset not in lifecycles.index:
            continue
        lifecycle = lifecycles.loc[asset]
        if pd.notna(lifecycle.delist_date) and pd.Timestamp(lifecycle.delist_date) < JQ_START:
            continue
        if pd.Timestamp(lifecycle.list_date) > JQ_END:
            continue
        path = _jq_cache_path(asset)
        try:
            if not path.is_file():
                code = _jq_code(asset, lifecycle.exchange)
                price = get_price(
                    code,
                    start_date=JQ_START.date().isoformat(),
                    end_date=JQ_END.date().isoformat(),
                    frequency="daily",
                    fields=["close", "pre_close", "high_limit", "low_limit", "paused"],
                    skip_paused=False,
                    fq=None,
                    panel=False,
                )
                if price is None or price.empty:
                    raise ValueError("empty price response")
                st = get_extras(
                    "is_st",
                    [code],
                    start_date=JQ_START.date().isoformat(),
                    end_date=JQ_END.date().isoformat(),
                    df=True,
                )
                raw = price.reset_index()
                date_column = price.index.name or raw.columns[0]
                raw = raw.rename(columns={date_column: "date"})
                if "date" not in raw.columns:
                    raise ValueError("price response has no date index")
                raw["asset"] = asset
                raw["is_st"] = st.iloc[:, 0].reindex(price.index).to_numpy() if not st.empty else math.nan
                raw["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                raw["data_source"] = "jqdatasdk.get_price+get_extras"
                raw.to_csv(path, index=False, compression="gzip", encoding="utf-8-sig", date_format="%Y-%m-%d")
            frame = pd.read_csv(path, compression="gzip", dtype={"asset": str})
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frames.append(frame)
            inputs.append({"path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append(_provider_error("joinquant", "fetch_or_read", exc, asset))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs, errors


def _jq_valuation_cache_path(date: pd.Timestamp) -> Path:
    return JQ_VALUATION_RAW_DIR / f"date={date:%Y%m%d}.csv.gz"


def collect_joinquant_valuation_sample(
    assets: list[str], dates: list[pd.Timestamp]
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    try:
        credentials = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8-sig"))["joinquant"]
        from jqdatasdk import auth, get_fundamentals, query, valuation as jq_valuation

        auth(credentials["username"], credentials["password"])
    except Exception as exc:
        errors.append(_provider_error("joinquant_valuation", "authenticate", exc))
        return pd.DataFrame(), inputs, errors

    asset_set = set(assets)
    JQ_VALUATION_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for date in sorted(pd.Timestamp(value).normalize() for value in set(dates)):
        path = _jq_valuation_cache_path(date)
        try:
            if not path.is_file():
                request = query(
                    jq_valuation.code,
                    jq_valuation.pe_ratio,
                    jq_valuation.pb_ratio,
                    jq_valuation.circulating_market_cap,
                )
                raw = get_fundamentals(request, date=date.date().isoformat())
                if raw is None or raw.empty:
                    raise ValueError("empty valuation response")
                raw["date"] = date
                raw["asset"] = raw["code"].astype(str).str[:6]
                raw["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                raw["data_source"] = "jqdatasdk.get_fundamentals/valuation"
                raw.to_csv(path, index=False, compression="gzip", encoding="utf-8-sig", date_format="%Y-%m-%d")
            frame = pd.read_csv(path, compression="gzip", dtype={"asset": str, "code": str})
            required = ["date", "asset", "pe_ratio", "pb_ratio", "circulating_market_cap"]
            missing = sorted(set(required).difference(frame.columns))
            if missing:
                raise ValueError(f"cached JoinQuant valuation missing columns: {missing}")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["asset"] = frame["asset"].astype(str).str.zfill(6)
            frames.append(frame.loc[frame["asset"].isin(asset_set), required])
            inputs.append({"path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append(_provider_error("joinquant_valuation", "fetch_or_read", exc, date.date().isoformat()))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs, errors


def compare_joinquant_state(trade: pd.DataFrame, jq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if jq.empty:
        return pd.DataFrame(), {
            "state_checks": 0,
            "paused_match_ratio": 0.0,
            "st_checks": 0,
            "st_match_ratio": 0.0,
            "limit_checks": 0,
            "limit_match_ratio": 0.0,
        }
    columns = [
        "date",
        "asset",
        "is_paused",
        "is_st",
        "has_price_limit",
        "limit_up",
        "limit_down",
        "execution_state_known",
        "limit_rule",
    ]
    checks = trade[columns].merge(jq, on=["date", "asset"], how="inner", validate="one_to_one")
    bool_columns = ["is_paused", "is_st_x", "has_price_limit", "execution_state_known", "paused", "is_st_y"]
    for column in bool_columns:
        checks[column] = _as_nullable_bool(checks[column])
    checks = checks[checks["paused"].notna() & checks["is_st_y"].notna()].copy()
    checks["paused_match"] = checks["is_paused"].eq(checks["paused"])
    checks["st_match"] = checks["is_st_x"].eq(checks["is_st_y"])
    checks["st_checked"] = ~checks["limit_rule"].isin(
        {
            "status_unknown",
            "status_trade_conflict",
            "legacy_special_transfer_unknown",
            "delisting_limit_unknown",
        }
    )
    known_limit = checks["has_price_limit"].fillna(False) & checks["execution_state_known"].fillna(False)
    checks["limit_checked"] = known_limit & checks["high_limit"].notna() & checks["low_limit"].notna()
    checks["limit_match"] = (
        (pd.to_numeric(checks["limit_up"], errors="coerce") - pd.to_numeric(checks["high_limit"], errors="coerce"))
        .abs()
        .le(0.011)
        & (pd.to_numeric(checks["limit_down"], errors="coerce") - pd.to_numeric(checks["low_limit"], errors="coerce"))
        .abs()
        .le(0.011)
    )
    limit_checks = int(checks["limit_checked"].sum())
    st_checks = int(checks["st_checked"].sum())
    return checks, {
        "state_checks": int(len(checks)),
        "paused_match_ratio": round(float(checks["paused_match"].mean()), 8) if len(checks) else 0.0,
        "st_checks": st_checks,
        "st_match_ratio": round(float(checks.loc[checks["st_checked"], "st_match"].mean()), 8)
        if st_checks
        else 0.0,
        "limit_checks": limit_checks,
        "limit_match_ratio": round(float(checks.loc[checks["limit_checked"], "limit_match"].mean()), 8)
        if limit_checks
        else 0.0,
        "unknown_state_rows_visible_in_joinquant": int((~checks["execution_state_known"].fillna(False)).sum()),
    }


def _em_cache_path(asset: str) -> Path:
    return EM_RAW_DIR / f"{asset}.csv.gz"


def collect_eastmoney_sample(
    assets: list[str],
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    try:
        import akshare as ak
    except Exception as exc:
        errors.append(_provider_error("eastmoney", "import_akshare", exc))
        return pd.DataFrame(), inputs, errors

    EM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        path = _em_cache_path(asset)
        try:
            if not path.is_file():
                raw = ak.stock_value_em(symbol=asset)
                if raw is None or raw.empty or "数据日期" not in raw.columns:
                    raise ValueError("empty or malformed valuation response")
                raw["asset"] = asset
                raw["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                raw["data_source"] = "akshare.stock_value_em/eastmoney"
                raw.to_csv(path, index=False, compression="gzip", encoding="utf-8-sig", date_format="%Y-%m-%d")
            frame = pd.read_csv(path, compression="gzip", dtype={"asset": str})
            frame = frame.rename(
                columns={
                    "数据日期": "date",
                    "流通市值": "em_float_market_cap",
                    "PE(TTM)": "em_pe_ttm",
                    "市净率": "em_pb",
                }
            )
            required = ["date", "asset", "em_float_market_cap", "em_pe_ttm", "em_pb"]
            missing = sorted(set(required).difference(frame.columns))
            if missing:
                raise ValueError(f"cached valuation response missing columns: {missing}")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frames.append(frame[required])
            inputs.append({"path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append(_provider_error("eastmoney", "fetch_or_read", exc, asset))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs, errors


def split_eastmoney_eligibility(
    assets: list[str], lifecycles: pd.DataFrame
) -> tuple[list[str], list[str]]:
    lifecycle_index = lifecycles.set_index("asset")
    active = [
        asset
        for asset in assets
        if asset in lifecycle_index.index and pd.isna(lifecycle_index.loc[asset].delist_date)
    ]
    return active, [asset for asset in assets if asset not in active]


def partition_eastmoney_validation_assets(
    assets: list[str], lifecycles: pd.DataFrame
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Select live-fetchable assets while retaining cached delisted evidence."""

    active, delisted = split_eastmoney_eligibility(assets, lifecycles)
    cached_delisted = [asset for asset in delisted if _em_cache_path(asset).is_file()]
    unavailable_delisted = [asset for asset in delisted if asset not in cached_delisted]
    collection_assets = [*active, *cached_delisted]
    return active, cached_delisted, unavailable_delisted, collection_assets


def _relative_error(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).clip(lower=1.0)
    return (left - right).abs() / denominator


def _valuation_agreement_metrics(checks: pd.DataFrame) -> dict[str, Any]:
    def metric(column: str, quantile: float) -> float | None:
        return round(float(checks[column].quantile(quantile)), 8) if len(checks) else None

    return {
        "checks": int(len(checks)),
        "pe_median_abs_relative_error": metric("pe_abs_relative_error", 0.5),
        "pe_p95_abs_relative_error": metric("pe_abs_relative_error", 0.95),
        "pb_median_abs_relative_error": metric("pb_abs_relative_error", 0.5),
        "pb_p95_abs_relative_error": metric("pb_abs_relative_error", 0.95),
        "cap_median_abs_relative_error": metric("cap_abs_relative_error", 0.5),
        "cap_p95_abs_relative_error": metric("cap_abs_relative_error", 0.95),
    }


def compare_joinquant_valuation(
    valuation: pd.DataFrame, joinquant: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if joinquant.empty:
        return pd.DataFrame(), {"checks": 0}
    jq = joinquant.rename(
        columns={
            "pe_ratio": "jq_pe_ttm",
            "pb_ratio": "jq_pb",
            "circulating_market_cap": "jq_float_market_cap_100m",
        }
    ).copy()
    jq["jq_float_market_cap"] = pd.to_numeric(jq["jq_float_market_cap_100m"], errors="coerce") * 100_000_000
    checks = valuation.merge(jq, on=["date", "asset"], how="inner", validate="one_to_one")
    numeric = ["pe_ttm", "pb", "market_cap", "jq_pe_ttm", "jq_pb", "jq_float_market_cap"]
    for column in numeric:
        checks[column] = pd.to_numeric(checks[column], errors="coerce")
    checks = checks.dropna(subset=numeric).copy()
    checks["pe_abs_relative_error"] = _relative_error(checks["pe_ttm"], checks["jq_pe_ttm"])
    checks["pb_abs_relative_error"] = _relative_error(checks["pb"], checks["jq_pb"])
    checks["cap_abs_relative_error"] = _relative_error(checks["market_cap"], checks["jq_float_market_cap"])
    return checks, _valuation_agreement_metrics(checks)


def compare_eastmoney_valuation(valuation: pd.DataFrame, eastmoney: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if eastmoney.empty:
        return pd.DataFrame(), {"checks": 0}
    checks = valuation.merge(eastmoney, on=["date", "asset"], how="inner", validate="one_to_one")
    numeric = ["pe_ttm", "pb", "market_cap", "em_pe_ttm", "em_pb", "em_float_market_cap"]
    for column in numeric:
        checks[column] = pd.to_numeric(checks[column], errors="coerce")
    checks = checks.dropna(subset=numeric).copy()
    checks["pe_abs_relative_error"] = _relative_error(checks["pe_ttm"], checks["em_pe_ttm"])
    checks["pb_abs_relative_error"] = _relative_error(checks["pb"], checks["em_pb"])
    checks["cap_abs_relative_error"] = _relative_error(checks["market_cap"], checks["em_float_market_cap"])

    return checks, _valuation_agreement_metrics(checks)


def compare_current_dividend_yield(valuation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not CURRENT_SNAPSHOT.is_file() or valuation.empty:
        return pd.DataFrame(), {"checks": 0, "median_absolute_error": None, "p95_absolute_error": None}
    current = pd.read_csv(CURRENT_SNAPSHOT, dtype={"asset": str})
    current["asset"] = current["asset"].astype(str).str.zfill(6)
    current = current[current["asset_type"].eq("stock")][["asset", "dividend_yield"]].dropna()
    latest = valuation.sort_values("date").groupby("asset", as_index=False).tail(1)
    checks = latest[["date", "asset", "dividend_yield"]].merge(
        current.rename(columns={"dividend_yield": "current_snapshot_dividend_yield"}), on="asset", how="inner"
    )
    checks["dividend_yield"] = pd.to_numeric(checks["dividend_yield"], errors="coerce")
    checks["current_snapshot_dividend_yield"] = pd.to_numeric(
        checks["current_snapshot_dividend_yield"], errors="coerce"
    )
    checks = checks.dropna().copy()
    checks["absolute_error"] = (
        checks["dividend_yield"] - checks["current_snapshot_dividend_yield"]
    ).abs()
    return checks, {
        "checks": int(len(checks)),
        "median_absolute_error": round(float(checks["absolute_error"].median()), 8) if len(checks) else None,
        "p95_absolute_error": round(float(checks["absolute_error"].quantile(0.95)), 8) if len(checks) else None,
    }


def _passes(report: dict[str, Any]) -> dict[str, bool]:
    ts = report["tushare"]
    tsv = report["tushare_valuation"]
    jq = report["joinquant"]
    jqv = report["joinquant_valuation"]
    em = report["eastmoney"]
    dy = report["dividend_yield"]
    return {
        "builder_complete": bool(report["builder_complete"]),
        "trade_builder_complete": bool(report["trade_builder_complete"]),
        "valuation_builder_complete": bool(report["valuation_builder_complete"]),
        "joinquant_source_coverage": report["source_coverage"]["joinquant"]
        >= THRESHOLDS["joinquant_source_coverage_min"],
        "joinquant_trade_state_source_coverage": report["source_coverage"]["joinquant_trade_state"]
        >= THRESHOLDS["joinquant_source_coverage_min"],
        "joinquant_valuation_source_coverage": report["source_coverage"]["joinquant_valuation"]
        >= THRESHOLDS["joinquant_source_coverage_min"],
        "eastmoney_source_coverage": report["source_coverage"]["eastmoney"]
        >= THRESHOLDS["eastmoney_source_coverage_min"],
        "tushare_population": ts["matched_rows"] >= THRESHOLDS["tushare_min_checks"],
        "tushare_coverage": ts["coverage_ratio"] >= THRESHOLDS["tushare_coverage_min"],
        "tushare_close": ts["close_match_ratio"] >= THRESHOLDS["tushare_close_match_min"],
        "tushare_pre_close": ts["pre_close_match_ratio"] >= THRESHOLDS["tushare_pre_close_match_min"]
        and ts.get("maximum_pre_close_absolute_error") is not None
        and ts["maximum_pre_close_absolute_error"] <= THRESHOLDS["tushare_max_close_absolute_error"],
        "pre_close_scope_holdout_population": len(
            report["threshold_lock"]["pre_close_scope_revision"]["independent_holdout_assets"]
        )
        >= THRESHOLDS["pre_close_scope_holdout_min_assets"],
        "tushare_relevant_close": tsv.get("relevant_close_checks", 0) > 0
        and tsv["relevant_close_match_ratio"] >= THRESHOLDS["tushare_relevant_close_match_min"]
        and tsv.get("maximum_relevant_close_absolute_error") is not None
        and tsv["maximum_relevant_close_absolute_error"] <= THRESHOLDS["tushare_max_close_absolute_error"]
        and tsv.get("large_relevant_close_mismatch_rows", 0) == 0,
        "joinquant_population": jq["state_checks"] >= THRESHOLDS["joinquant_min_state_checks"],
        "joinquant_paused": jq["paused_match_ratio"] >= THRESHOLDS["joinquant_paused_match_min"],
        "joinquant_st": jq["st_match_ratio"] >= THRESHOLDS["joinquant_st_match_min"],
        "joinquant_limits": jq["limit_checks"] > 0 and jq["limit_match_ratio"] >= THRESHOLDS["joinquant_limit_match_min"],
        "joinquant_valuation_population": jqv.get("checks", 0) >= THRESHOLDS["joinquant_min_valuation_checks"],
        "joinquant_valuation_pe": jqv.get("pe_median_abs_relative_error") is not None
        and jqv["pe_median_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_pe_median_abs_relative_max"]
        and jqv["pe_p95_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_pe_p95_abs_relative_max"],
        "joinquant_valuation_pb": jqv.get("pb_median_abs_relative_error") is not None
        and jqv["pb_median_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_pb_median_abs_relative_max"]
        and jqv["pb_p95_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_pb_p95_abs_relative_max"],
        "joinquant_valuation_cap": jqv.get("cap_median_abs_relative_error") is not None
        and jqv["cap_median_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_cap_median_abs_relative_max"]
        and jqv["cap_p95_abs_relative_error"] <= THRESHOLDS["joinquant_valuation_cap_p95_abs_relative_max"],
        "eastmoney_population": em.get("checks", 0) >= THRESHOLDS["eastmoney_min_checks"],
        "delisted_valuation_cross_source_population": report["delisted_valuation_validation"]["checks"]
        >= THRESHOLDS["delisted_valuation_min_cross_source_checks"],
        "delisted_valuation_cross_source_asset_population": report["delisted_valuation_validation"][
            "validated_asset_count"
        ]
        >= THRESHOLDS["delisted_valuation_min_cross_source_assets"],
        "eastmoney_pe": em.get("pe_median_abs_relative_error") is not None
        and em["pe_median_abs_relative_error"] <= THRESHOLDS["eastmoney_pe_median_abs_relative_max"]
        and em["pe_p95_abs_relative_error"] <= THRESHOLDS["eastmoney_pe_p95_abs_relative_max"],
        "eastmoney_pb_central": em.get("pb_median_abs_relative_error") is not None
        and em["pb_median_abs_relative_error"] <= THRESHOLDS["eastmoney_pb_median_abs_relative_max"],
        "eastmoney_cap": em.get("cap_median_abs_relative_error") is not None
        and em["cap_median_abs_relative_error"] <= THRESHOLDS["eastmoney_cap_median_abs_relative_max"]
        and em["cap_p95_abs_relative_error"] <= THRESHOLDS["eastmoney_cap_p95_abs_relative_max"],
        "yield_population": dy["checks"] >= THRESHOLDS["yield_min_checks"],
        "yield_match": dy.get("median_absolute_error") is not None
        and dy["median_absolute_error"] <= THRESHOLDS["yield_median_absolute_error_max"]
        and dy["p95_absolute_error"] <= THRESHOLDS["yield_p95_absolute_error_max"],
    }


TRADE_QUALIFICATION_CHECKS = (
    "trade_builder_complete",
    "joinquant_trade_state_source_coverage",
    "tushare_population",
    "tushare_coverage",
    "tushare_close",
    "tushare_pre_close",
    "pre_close_scope_holdout_population",
    "joinquant_population",
    "joinquant_paused",
    "joinquant_st",
    "joinquant_limits",
)

VALUATION_QUALIFICATION_CHECKS = (
    "valuation_builder_complete",
    "joinquant_valuation_source_coverage",
    "eastmoney_source_coverage",
    "tushare_relevant_close",
    "joinquant_valuation_population",
    "joinquant_valuation_pe",
    "joinquant_valuation_pb",
    "joinquant_valuation_cap",
    "eastmoney_population",
    "delisted_valuation_cross_source_population",
    "delisted_valuation_cross_source_asset_population",
    "eastmoney_pe",
    "eastmoney_pb_central",
    "eastmoney_cap",
    "yield_population",
    "yield_match",
)


def _scope_checks(checks: dict[str, bool], names: tuple[str, ...]) -> dict[str, bool]:
    missing = [name for name in names if name not in checks]
    if missing:
        raise ValueError(f"qualification scope references missing checks: {missing}")
    return {name: bool(checks[name]) for name in names}


def run_validation(as_of: str, sample_assets: int = 96, eastmoney_assets: int = 30) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    tushare_refresh_input = _validate_tushare_refresh_manifest(as_of_date)
    builder = _load_builder_manifest()
    trade_available_assets = _available_builder_assets(builder, "trade_state")
    valuation_available_assets = _available_builder_assets(builder, "valuation")
    trade_assets = select_validation_assets(trade_available_assets, sample_assets)
    valuation_assets = select_validation_assets(valuation_available_assets, sample_assets)
    if not trade_assets or not valuation_assets:
        raise ValueError("trade-state and valuation validation populations must both be non-empty")
    trade = _load_builder_output(builder, "trade_state", set(trade_assets))
    valuation = _load_builder_output(builder, "valuation", set(valuation_assets))
    source_assets = sorted(set(trade_assets).union(valuation_assets))
    baostock, baostock_inputs = _load_sample_histories(source_assets, as_of_date)
    tushare, tushare_inputs = collect_tushare_sample(source_assets)
    adjustment_factors = pd.read_csv(ADJUSTMENT_FACTOR_PATH, dtype={"asset": str})
    adjustment_ratios = build_adjustment_ratios(adjustment_factors)
    sample_valuation = valuation[valuation["asset"].isin(valuation_assets)].copy()
    relevant_price_keys = set(zip(sample_valuation["date"], sample_valuation["asset"]))
    sample_trade = trade[trade["asset"].isin(trade_assets)].copy()
    sample_trade["has_price_limit"] = _as_nullable_bool(sample_trade["has_price_limit"])
    sample_trade["execution_state_known"] = _as_nullable_bool(sample_trade["execution_state_known"])
    pre_close_scope = sample_trade[
        sample_trade["has_price_limit"].fillna(False)
        & sample_trade["execution_state_known"].fillna(False)
    ]
    relevant_pre_close_keys = set(zip(pre_close_scope["date"], pre_close_scope["asset"]))
    ts_checks, ts_metrics = compare_tushare_prices(
        baostock[baostock["asset"].isin(trade_assets)],
        tushare[tushare["asset"].isin(trade_assets)],
        relevant_pre_close_keys=relevant_pre_close_keys,
        adjustment_ratios=adjustment_ratios[adjustment_ratios["asset"].isin(trade_assets)],
    )
    ts_valuation_checks, ts_valuation_metrics = compare_tushare_prices(
        baostock[baostock["asset"].isin(valuation_assets)],
        tushare[tushare["asset"].isin(valuation_assets)],
        relevant_price_keys=relevant_price_keys,
        relevant_pre_close_keys=set(),
        adjustment_ratios=adjustment_ratios[adjustment_ratios["asset"].isin(valuation_assets)],
    )
    jq_raw, jq_inputs, jq_errors = collect_joinquant_sample(trade_assets, as_of_date)
    jq_checks, jq_metrics = compare_joinquant_state(trade, jq_raw)
    jq_valuation_scope = sample_valuation[
        sample_valuation["date"].between(JQ_START, JQ_END)
    ].copy()
    jq_valuation_dates = jq_valuation_scope["date"].dropna().drop_duplicates().tolist()
    jq_valuation_raw, jq_valuation_inputs, jq_valuation_errors = collect_joinquant_valuation_sample(
        valuation_assets, jq_valuation_dates
    )
    jq_valuation_checks, jq_valuation_metrics = compare_joinquant_valuation(
        jq_valuation_scope, jq_valuation_raw
    )
    em_assets = valuation_assets[: max(0, eastmoney_assets)]
    (
        em_active_assets,
        em_cached_delisted_assets,
        em_unavailable_delisted_assets,
        em_collection_assets,
    ) = partition_eastmoney_validation_assets(
        em_assets, load_lifecycles(as_of=as_of_date)
    )
    em_delisted_assets = [*em_cached_delisted_assets, *em_unavailable_delisted_assets]
    em_raw, em_inputs, em_errors = collect_eastmoney_sample(em_collection_assets)
    em_checks, em_metrics = compare_eastmoney_valuation(
        valuation[valuation["asset"].isin(em_assets)], em_raw
    )
    em_delisted_checks = em_checks[
        em_checks["asset"].isin(em_cached_delisted_assets)
    ].copy()
    em_delisted_metrics = _valuation_agreement_metrics(em_delisted_checks)
    em_delisted_validated_assets = sorted(
        em_delisted_checks["asset"].astype(str).str.zfill(6).unique().tolist()
    )
    yield_checks, yield_metrics = compare_current_dividend_yield(sample_valuation)
    jq_trade_eligible_assets = set(
        trade.loc[trade["date"].between(JQ_START, JQ_END), "asset"]
    )
    jq_valuation_eligible_assets = set(
        sample_valuation.loc[sample_valuation["date"].between(JQ_START, JQ_END), "asset"]
    )
    jq_state_assets = set(jq_raw["asset"].astype(str).str.zfill(6)) if not jq_raw.empty else set()
    jq_valuation_assets = (
        set(jq_valuation_raw["asset"].astype(str).str.zfill(6)) if not jq_valuation_raw.empty else set()
    )
    jq_state_coverage = (
        len(jq_state_assets.intersection(jq_trade_eligible_assets)) / len(jq_trade_eligible_assets)
        if jq_trade_eligible_assets
        else 0.0
    )
    jq_valuation_coverage = (
        len(jq_valuation_assets.intersection(jq_valuation_eligible_assets)) / len(jq_valuation_eligible_assets)
        if jq_valuation_eligible_assets
        else 0.0
    )
    em_success_assets = set(em_raw["asset"].astype(str).str.zfill(6)) if not em_raw.empty else set()
    em_coverage = (
        len(em_success_assets.intersection(em_active_assets)) / len(em_active_assets) if em_active_assets else 0.0
    )
    source_errors = {
        "joinquant_trade_state": jq_errors,
        "joinquant_valuation": jq_valuation_errors,
        "eastmoney": em_errors,
    }
    trade_builder_complete = bool(
        builder.get("trade_state", {}).get("qualification_status")
        == "READY_FOR_CROSS_SOURCE_VALIDATION"
        and builder.get("trade_state", {}).get("historical_backtest_allowed") is True
    )
    valuation_builder_complete = bool(
        builder.get("valuation", {}).get("qualification_status")
        == "READY_FOR_CROSS_SOURCE_VALIDATION"
        and builder.get("valuation", {}).get("historical_backtest_allowed") is True
    )
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "sample_assets": source_assets,
        "trade_sample_assets": trade_assets,
        "valuation_sample_assets": valuation_assets,
        "builder_complete": trade_builder_complete and valuation_builder_complete,
        "trade_builder_complete": trade_builder_complete,
        "valuation_builder_complete": valuation_builder_complete,
        "tushare": ts_metrics,
        "tushare_valuation": ts_valuation_metrics,
        "joinquant": jq_metrics,
        "joinquant_valuation": jq_valuation_metrics,
        "eastmoney": em_metrics,
        "eastmoney_eligible_assets": {
            "active_assets": em_active_assets,
            "cached_delisted_assets": em_cached_delisted_assets,
            "unavailable_delisted_assets": em_unavailable_delisted_assets,
            "collection_assets": em_collection_assets,
        },
        "delisted_valuation_validation": {
            "assets": em_delisted_assets,
            "cached_assets": em_cached_delisted_assets,
            "unavailable_assets": em_unavailable_delisted_assets,
            "validated_assets": em_delisted_validated_assets,
            "validated_asset_count": len(em_delisted_validated_assets),
            **em_delisted_metrics,
            "status": (
                "PASS_MINIMUM_CROSS_SOURCE_SAMPLE"
                if em_delisted_metrics["checks"]
                >= THRESHOLDS["delisted_valuation_min_cross_source_checks"]
                and len(em_delisted_validated_assets)
                >= THRESHOLDS["delisted_valuation_min_cross_source_assets"]
                else "BLOCKED_INSUFFICIENT_INDEPENDENT_HISTORICAL_VALUATION_COVERAGE"
            ),
            "required_source": "licensed PIT valuation history covering delisted A-shares",
        },
        "dividend_yield": yield_metrics,
        "source_coverage": {
            "joinquant": round(min(jq_state_coverage, jq_valuation_coverage), 8),
            "joinquant_trade_state": round(jq_state_coverage, 8),
            "joinquant_valuation": round(jq_valuation_coverage, 8),
            "eastmoney": round(em_coverage, 8),
        },
        "source_errors": source_errors,
        "thresholds": THRESHOLDS,
        "threshold_lock": {
            "status": "LOCKED_AFTER_24_ASSET_PILOT_RULE_REVISION",
            "design_sample_assets": [
                "000001", "000002", "000004", "000005", "000006", "000007", "000008", "000009",
                "000010", "000011", "000012", "000014", "000016", "000017", "000018", "000019",
                "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027"
            ],
            "locked_at": "2026-07-18",
            "interpretation": "the twenty-four-asset pilot cannot validate thresholds selected or revised after inspecting it",
            "pre_close_scope_revision": {
                "trigger_sample_boundary": 64,
                "reason": "a 2005 share-reform resumption exposed that exchange rules did not define an ex-right reference price",
                "rule": "hard pre-close agreement is limited to execution-known sessions with an active price limit",
                "official_rule_source": "https://www.szse.cn/marketServices/deal/reform/t20050620_519267.html",
                "independent_holdout_assets": trade_assets[64:],
                "holdout_status": "PASS_MINIMUM_32" if len(trade_assets[64:]) >= 32 else "PENDING_MINIMUM_32",
            },
        },
        "pre_close_gate_scope": {
            "hard_gate": "execution_state_known=true and has_price_limit=true",
            "diagnostic_only": [
                "share-reform resumption",
                "other resumption sessions with unknown price-limit treatment",
                "listing sessions without a price limit",
            ],
            "excluded_rows_remain_in_tushare_checks": True,
        },
        "method_boundary": (
            "Tushare validates raw traded closes and pre-close only where execution state and price limits are known; "
            "no-limit and unknown-resumption pre-close rows remain diagnostic. JoinQuant validates trade state and valuation in the visible "
            "trial window; Eastmoney validates longer active-security valuation consistency and any independently "
            "cached delisted-security rows, while its active-source coverage denominator remains active-only; broad "
            "delisted-security valuation remains blocked without a second historical source; no validation result is "
            "model-performance evidence"
        ),
        "model_promotion_allowed": False,
    }
    checks = _passes(report)
    trade_checks = _scope_checks(checks, TRADE_QUALIFICATION_CHECKS)
    valuation_checks = _scope_checks(checks, VALUATION_QUALIFICATION_CHECKS)
    report["qualification_checks"] = checks
    report["trade_state_qualification_checks"] = trade_checks
    report["valuation_qualification_checks"] = valuation_checks
    report["trade_state_qualification_status"] = (
        "PASS" if all(trade_checks.values()) else "OBSERVATION_SAMPLE_INCOMPLETE"
    )
    report["valuation_qualification_status"] = (
        "PASS" if all(valuation_checks.values()) else "OBSERVATION_SAMPLE_INCOMPLETE"
    )
    report["qualification_status"] = "PASS" if all(checks.values()) else "OBSERVATION_SAMPLE_INCOMPLETE"
    report["historical_backtest_allowed"] = report["qualification_status"] == "PASS"

    _write_csv(ts_checks, TUSHARE_CHECKS)
    _write_csv(ts_valuation_checks, TUSHARE_VALUATION_CHECKS)
    _write_csv(jq_checks, JQ_CHECKS)
    _write_csv(jq_valuation_checks, JQ_VALUATION_CHECKS)
    _write_csv(em_checks, EM_CHECKS)
    _write_csv(em_delisted_checks, EM_DELISTED_CHECKS)
    _write_csv(yield_checks, YIELD_CHECKS)
    source_observation_rows = [error for errors in source_errors.values() for error in errors]
    _write_csv(
        pd.DataFrame(source_observation_rows, columns=["source", "stage", "asset", "error_type"]),
        SOURCE_OBSERVATIONS,
    )
    warning_rows: list[dict[str, Any]] = []
    if (
        em_metrics.get("pb_p95_abs_relative_error") is not None
        and em_metrics["pb_p95_abs_relative_error"] > THRESHOLDS["eastmoney_pb_p95_abs_relative_max"]
    ):
        warning_rows.append(
            {
                "warning": "eastmoney_pb_tail_disagreement",
                "value": em_metrics["pb_p95_abs_relative_error"],
                "threshold": THRESHOLDS["eastmoney_pb_p95_abs_relative_max"],
                "detail": "non-qualifying diagnostic; JoinQuant is the visible-window adjudicator",
            }
        )
    if ts_metrics.get("close_mismatch_rows", 0) > 0:
        warning_rows.append(
            {
                "warning": "tushare_close_mismatch_rows",
                "value": ts_metrics["close_mismatch_rows"],
                "threshold": THRESHOLDS["tushare_close_match_min"],
                "detail": "all-day close is diagnostic; pre-close and valuation-snapshot close have hard gates",
            }
        )
    if ts_metrics.get("excluded_pre_close_mismatch_rows", 0) > 0:
        warning_rows.append(
            {
                "warning": "tushare_pre_close_mismatch_outside_execution_gate",
                "value": ts_metrics["excluded_pre_close_mismatch_rows"],
                "threshold": 0,
                "detail": "diagnostic only; these rows have no active price-limit execution contract",
            }
        )
    _write_csv(pd.DataFrame(warning_rows, columns=["warning", "value", "threshold", "detail"]), WARNINGS)
    exception_rows = [
        {"check": name, "status": "fail", "detail": "fixed validation threshold not satisfied"}
        for name, passed in checks.items()
        if not passed
    ]
    _write_csv(pd.DataFrame(exception_rows, columns=["check", "status", "detail"]), EXCEPTIONS)
    trade_exception_rows = [
        {"check": name, "status": "fail", "detail": "fixed trade-state validation threshold not satisfied"}
        for name, passed in trade_checks.items()
        if not passed
    ]
    valuation_exception_rows = [
        {"check": name, "status": "fail", "detail": "fixed valuation validation threshold not satisfied"}
        for name, passed in valuation_checks.items()
        if not passed
    ]
    _write_csv(
        pd.DataFrame(trade_exception_rows, columns=["check", "status", "detail"]),
        TRADE_EXCEPTIONS,
    )
    _write_csv(
        pd.DataFrame(valuation_exception_rows, columns=["check", "status", "detail"]),
        VALUATION_EXCEPTIONS,
    )
    _write_json(report, REPORT_PATH)
    trade_report = {
        "created_at": report["created_at"],
        "as_of_date": report["as_of_date"],
        "dataset_id": "stock_trade_state",
        "sample_assets": trade_assets,
        "qualification_checks": trade_checks,
        "qualification_status": report["trade_state_qualification_status"],
        "historical_backtest_allowed": report["trade_state_qualification_status"] == "PASS",
        "tushare": ts_metrics,
        "joinquant": jq_metrics,
        "source_coverage": {
            "joinquant_trade_state": report["source_coverage"]["joinquant_trade_state"]
        },
        "threshold_lock": report["threshold_lock"],
        "method_boundary": report["method_boundary"],
        "model_promotion_allowed": False,
    }
    valuation_report = {
        "created_at": report["created_at"],
        "as_of_date": report["as_of_date"],
        "dataset_id": "stock_valuation_history",
        "sample_assets": valuation_assets,
        "qualification_checks": valuation_checks,
        "qualification_status": report["valuation_qualification_status"],
        "historical_backtest_allowed": report["valuation_qualification_status"] == "PASS",
        "tushare_valuation": ts_valuation_metrics,
        "joinquant_valuation": jq_valuation_metrics,
        "eastmoney": em_metrics,
        "delisted_valuation_validation": report["delisted_valuation_validation"],
        "dividend_yield": yield_metrics,
        "source_coverage": {
            "joinquant_valuation": report["source_coverage"]["joinquant_valuation"],
            "eastmoney": report["source_coverage"]["eastmoney"],
        },
        "threshold_lock": report["threshold_lock"],
        "method_boundary": report["method_boundary"],
        "model_promotion_allowed": False,
    }
    _write_json(trade_report, TRADE_REPORT_PATH)
    _write_json(valuation_report, VALUATION_REPORT_PATH)

    input_paths = [
        {"path": _relative(BUILDER_MANIFEST), "sha256": _sha256(BUILDER_MANIFEST)},
        {
            "path": builder["asset_summary_path"],
            "sha256": builder["asset_summary_sha256"],
        },
        {
            "path": builder["trade_state"]["output_path"],
            "sha256": builder["trade_state"]["output_sha256"],
        },
        {
            "path": builder["valuation"]["output_path"],
            "sha256": builder["valuation"]["output_sha256"],
        },
        {"path": _relative(TUSHARE_ACCEPTANCE), "sha256": _sha256(TUSHARE_ACCEPTANCE)},
        {"path": _relative(TUSHARE_QUARANTINE), "sha256": _sha256(TUSHARE_QUARANTINE)},
        {"path": _relative(TUSHARE_FILE_ACCEPTANCE), "sha256": _sha256(TUSHARE_FILE_ACCEPTANCE)},
        tushare_refresh_input,
        {"path": _relative(ADJUSTMENT_FACTOR_PATH), "sha256": _sha256(ADJUSTMENT_FACTOR_PATH)},
        {"path": _relative(ADJUSTMENT_FACTOR_MANIFEST), "sha256": _sha256(ADJUSTMENT_FACTOR_MANIFEST)},
        *baostock_inputs,
        *tushare_inputs,
        *jq_inputs,
        *jq_valuation_inputs,
        *em_inputs,
    ]
    code_path = Path(__file__).resolve()
    manifest = {
        "created_at": report["created_at"],
        "validation_schema": "cross_provider_stock_market_history_v4",
        "inputs": input_paths,
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "output_path": _relative(TUSHARE_CHECKS),
        "output_sha256": _sha256(TUSHARE_CHECKS),
        "tushare_valuation_checks_path": _relative(TUSHARE_VALUATION_CHECKS),
        "tushare_valuation_checks_sha256": _sha256(TUSHARE_VALUATION_CHECKS),
        "joinquant_checks_path": _relative(JQ_CHECKS),
        "joinquant_checks_sha256": _sha256(JQ_CHECKS),
        "joinquant_valuation_checks_path": _relative(JQ_VALUATION_CHECKS),
        "joinquant_valuation_checks_sha256": _sha256(JQ_VALUATION_CHECKS),
        "eastmoney_checks_path": _relative(EM_CHECKS),
        "eastmoney_checks_sha256": _sha256(EM_CHECKS),
        "eastmoney_delisted_checks_path": _relative(EM_DELISTED_CHECKS),
        "eastmoney_delisted_checks_sha256": _sha256(EM_DELISTED_CHECKS),
        "yield_checks_path": _relative(YIELD_CHECKS),
        "yield_checks_sha256": _sha256(YIELD_CHECKS),
        "exceptions_path": _relative(EXCEPTIONS),
        "exceptions_sha256": _sha256(EXCEPTIONS),
        "report_path": _relative(REPORT_PATH),
        "report_sha256": _sha256(REPORT_PATH),
        "source_observations_path": _relative(SOURCE_OBSERVATIONS),
        "source_observations_sha256": _sha256(SOURCE_OBSERVATIONS),
        "warnings_path": _relative(WARNINGS),
        "warnings_sha256": _sha256(WARNINGS),
        "outputs": [
            {"role": "tushare_checks", "path": _relative(TUSHARE_CHECKS), "sha256": _sha256(TUSHARE_CHECKS)},
            {"role": "tushare_valuation_checks", "path": _relative(TUSHARE_VALUATION_CHECKS), "sha256": _sha256(TUSHARE_VALUATION_CHECKS)},
            {"role": "joinquant_checks", "path": _relative(JQ_CHECKS), "sha256": _sha256(JQ_CHECKS)},
            {"role": "joinquant_valuation_checks", "path": _relative(JQ_VALUATION_CHECKS), "sha256": _sha256(JQ_VALUATION_CHECKS)},
            {"role": "eastmoney_checks", "path": _relative(EM_CHECKS), "sha256": _sha256(EM_CHECKS)},
            {"role": "eastmoney_delisted_checks", "path": _relative(EM_DELISTED_CHECKS), "sha256": _sha256(EM_DELISTED_CHECKS)},
            {"role": "dividend_yield_checks", "path": _relative(YIELD_CHECKS), "sha256": _sha256(YIELD_CHECKS)},
            {"role": "source_observations", "path": _relative(SOURCE_OBSERVATIONS), "sha256": _sha256(SOURCE_OBSERVATIONS)},
            {"role": "warnings", "path": _relative(WARNINGS), "sha256": _sha256(WARNINGS)},
            {"role": "exceptions", "path": _relative(EXCEPTIONS), "sha256": _sha256(EXCEPTIONS)},
            {"role": "report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
        ],
        "threshold_lock": report["threshold_lock"],
        "qualification_status": report["qualification_status"],
        "qualification_checks": checks,
        "trade_state_qualification_status": report["trade_state_qualification_status"],
        "valuation_qualification_status": report["valuation_qualification_status"],
        "historical_backtest_allowed": report["historical_backtest_allowed"],
        "model_promotion_allowed": False,
    }
    _write_json(manifest, MANIFEST_PATH)
    trade_outputs = [
        {"role": "tushare_checks", "path": _relative(TUSHARE_CHECKS), "sha256": _sha256(TUSHARE_CHECKS)},
        {"role": "joinquant_checks", "path": _relative(JQ_CHECKS), "sha256": _sha256(JQ_CHECKS)},
        {"role": "source_observations", "path": _relative(SOURCE_OBSERVATIONS), "sha256": _sha256(SOURCE_OBSERVATIONS)},
        {"role": "warnings", "path": _relative(WARNINGS), "sha256": _sha256(WARNINGS)},
        {"role": "exceptions", "path": _relative(TRADE_EXCEPTIONS), "sha256": _sha256(TRADE_EXCEPTIONS)},
        {"role": "report", "path": _relative(TRADE_REPORT_PATH), "sha256": _sha256(TRADE_REPORT_PATH)},
    ]
    valuation_outputs = [
        {"role": "tushare_valuation_checks", "path": _relative(TUSHARE_VALUATION_CHECKS), "sha256": _sha256(TUSHARE_VALUATION_CHECKS)},
        {"role": "joinquant_valuation_checks", "path": _relative(JQ_VALUATION_CHECKS), "sha256": _sha256(JQ_VALUATION_CHECKS)},
        {"role": "eastmoney_checks", "path": _relative(EM_CHECKS), "sha256": _sha256(EM_CHECKS)},
        {"role": "eastmoney_delisted_checks", "path": _relative(EM_DELISTED_CHECKS), "sha256": _sha256(EM_DELISTED_CHECKS)},
        {"role": "dividend_yield_checks", "path": _relative(YIELD_CHECKS), "sha256": _sha256(YIELD_CHECKS)},
        {"role": "source_observations", "path": _relative(SOURCE_OBSERVATIONS), "sha256": _sha256(SOURCE_OBSERVATIONS)},
        {"role": "warnings", "path": _relative(WARNINGS), "sha256": _sha256(WARNINGS)},
        {"role": "exceptions", "path": _relative(VALUATION_EXCEPTIONS), "sha256": _sha256(VALUATION_EXCEPTIONS)},
        {"role": "report", "path": _relative(VALUATION_REPORT_PATH), "sha256": _sha256(VALUATION_REPORT_PATH)},
    ]
    scoped_common = {
        "created_at": report["created_at"],
        "inputs": input_paths,
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "threshold_lock": report["threshold_lock"],
        "model_promotion_allowed": False,
    }
    trade_manifest = {
        **scoped_common,
        "validation_schema": "cross_provider_stock_trade_state_v1",
        "dataset_id": "stock_trade_state",
        "outputs": trade_outputs,
        "qualification_status": trade_report["qualification_status"],
        "qualification_checks": trade_checks,
        "historical_backtest_allowed": trade_report["historical_backtest_allowed"],
    }
    valuation_manifest = {
        **scoped_common,
        "validation_schema": "cross_provider_stock_valuation_v2",
        "dataset_id": "stock_valuation_history",
        "outputs": valuation_outputs,
        "qualification_status": valuation_report["qualification_status"],
        "qualification_checks": valuation_checks,
        "historical_backtest_allowed": valuation_report["historical_backtest_allowed"],
    }
    _write_json(trade_manifest, TRADE_MANIFEST_PATH)
    _write_json(valuation_manifest, VALUATION_MANIFEST_PATH)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--sample-assets", type=int, default=96)
    parser.add_argument("--eastmoney-assets", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_validation(args.as_of, args.sample_assets, args.eastmoney_assets)
    print(
        json.dumps(
            {
                "qualification_status": report["qualification_status"],
                "sample_asset_count": len(report["sample_assets"]),
                "trade_state_qualification_status": report["trade_state_qualification_status"],
                "valuation_qualification_status": report["valuation_qualification_status"],
                "tushare": report["tushare"],
                "joinquant": report["joinquant"],
                "joinquant_valuation": report["joinquant_valuation"],
                "eastmoney": report["eastmoney"],
                "dividend_yield": report["dividend_yield"],
                "historical_backtest_allowed": report["historical_backtest_allowed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
