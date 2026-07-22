"""Build non-promotable monthly PIT diagnostics for today's durable-income watchlist.

This module intentionally does not create a historical stock-selection backtest.
The available universe is the current SW constituent snapshot and the cached QFQ
prices do not carry point-in-time adjustment vintages. Outputs therefore answer
only: "How would today's watchlist have looked under data available then?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .core import allocate_core_targets, compute_price_features, entry_decision, load_config, score_universe
from .stock_snapshot_builder import _normalize_price, build_stock_row, finalize_snapshot


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4" / "raw"
WATCHLIST_PATH = ROOT / "data_catalog" / "long_hold_v4_watchlist.csv"
CHINA_10Y_PATH = ROOT / "data_raw" / "macro" / "china_10y_yield.csv"
RAW_DATASETS = {
    "financial": "financial",
    "dividend": "dividend",
    "pe": "valuation_pe",
    "pb": "valuation_pb",
    "price": "price_qfq",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_watchlist(path: Path, max_assets: int | None) -> pd.DataFrame:
    watchlist = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
    required = {"asset", "asset_name", "sector", "historical_backtest_allowed"}
    missing = sorted(required.difference(watchlist.columns))
    if missing:
        raise ValueError(f"watchlist missing columns: {missing}")
    watchlist["asset"] = watchlist["asset"].astype(str).str.zfill(6)
    if watchlist["asset"].duplicated().any():
        raise ValueError("watchlist asset codes must be unique")
    if watchlist["historical_backtest_allowed"].astype(str).str.lower().isin({"true", "1", "yes"}).any():
        raise ValueError("current constituent watchlist cannot be historical-backtest enabled")
    return watchlist.head(max_assets).copy() if max_assets else watchlist


def _load_asset_raw(asset: str, root: Path) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    datasets: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []
    for name, directory in RAW_DATASETS.items():
        path = root / directory / f"{asset}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        datasets[name] = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        paths.append(path)
    return datasets, paths


def _load_china_10y(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"date", "available_date", "value"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"China 10Y data missing columns: {missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["available_date"] = pd.to_datetime(data["available_date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "available_date", "value"]).sort_values(["available_date", "date"])
    if data.empty or (data["available_date"] < data["date"]).any():
        raise ValueError("China 10Y data has no valid PIT rows")
    return data


def _rate_as_of(data: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, pd.Timestamp]:
    available = data[data["available_date"] <= as_of]
    if available.empty:
        raise ValueError(f"no China 10Y value available by {as_of.date()}")
    latest = available.iloc[-1]
    return float(latest["value"] / 100.0), pd.Timestamp(latest["available_date"])


def _month_ends(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("start must not exceed end")
    dates = pd.date_range(start_ts, end_ts, freq="ME")
    if dates.empty:
        raise ValueError("date range contains no month end")
    return dates


def run_historical_diagnostic(
    root: Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    max_assets: int | None = None,
    output_directory: Path | None = None,
) -> dict[str, Path]:
    config = load_config(root / "configs" / "long_hold_v4.json")
    output = output_directory or Path("outputs/long_hold_v4/historical_diagnostic")
    output = output if output.is_absolute() else root / output
    output.mkdir(parents=True, exist_ok=True)
    watchlist_path = root / WATCHLIST_PATH.relative_to(ROOT)
    china_10y_path = root / CHINA_10Y_PATH.relative_to(ROOT)
    watchlist = _load_watchlist(watchlist_path, max_assets)
    rates = _load_china_10y(china_10y_path)

    raw_cache: dict[str, dict[str, pd.DataFrame]] = {}
    price_feature_cache: dict[str, pd.DataFrame] = {}
    first_price_dates: dict[str, pd.Timestamp] = {}
    input_paths: list[Path] = [watchlist_path, china_10y_path, root / "configs" / "long_hold_v4.json"]
    load_errors: list[dict[str, str]] = []
    for _, watch in watchlist.iterrows():
        asset = str(watch["asset"])
        try:
            raw_cache[asset], paths = _load_asset_raw(asset, root / RAW_ROOT.relative_to(ROOT))
            price = raw_cache[asset]["price"]
            date_column = "date" if "date" in price.columns else "日期" if "日期" in price.columns else None
            price_dates = pd.to_datetime(price[date_column], errors="coerce") if date_column else pd.Series(dtype="datetime64[ns]")
            if price_dates.dropna().empty:
                raise ValueError("price source has no valid dates")
            first_price_dates[asset] = pd.Timestamp(price_dates.min()).normalize()
            normalized, _ = _normalize_price(raw_cache[asset]["price"], asset, pd.Timestamp(end), write_output=False)
            price_feature_cache[asset] = compute_price_features(normalized, config)
            input_paths.extend(paths)
        except (FileNotFoundError, OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raw_cache.pop(asset, None)
            load_errors.append({"asset": asset, "as_of_date": "", "error": f"raw_load_failed:{exc}"})

    panel_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    build_errors = list(load_errors)
    exclusions: list[dict[str, str]] = []
    for as_of in _month_ends(start, end):
        china_10y, china_10y_date = _rate_as_of(rates, as_of)
        rows: list[dict[str, Any]] = []
        for _, watch in watchlist.iterrows():
            asset = str(watch["asset"])
            if asset not in raw_cache:
                continue
            if as_of < first_price_dates[asset]:
                exclusions.append(
                    {
                        "asset": asset,
                        "as_of_date": str(as_of.date()),
                        "reason": "before_first_available_price",
                        "first_price_date": str(first_price_dates[asset].date()),
                    }
                )
                continue
            try:
                rows.append(
                    build_stock_row(
                        watch,
                        raw_cache[asset],
                        china_10y,
                        china_10y_date,
                        as_of,
                        write_price_output=False,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                build_errors.append({"asset": asset, "as_of_date": str(as_of.date()), "error": repr(exc)})
        snapshot = finalize_snapshot(rows)
        scored = score_universe(snapshot, as_of, config) if not snapshot.empty else snapshot
        if not scored.empty:
            decision_rows: list[dict[str, Any]] = []
            for _, scored_row in scored.iterrows():
                asset = str(scored_row["asset"])
                features = price_feature_cache[asset]
                available_features = features[features["date"] <= as_of]
                if available_features.empty:
                    raise AssertionError(f"no timing feature on or before {as_of.date()} for {asset}")
                latest = available_features.iloc[-1]
                if pd.Timestamp(latest["date"]) > as_of:
                    raise AssertionError("timing feature look-ahead detected")
                decision_rows.append(
                    {
                        "asset": asset,
                        **entry_decision(scored_row, latest, 0.0, as_of, config),
                        "latest_price_date": latest["date"],
                        "latest_close": latest["close"],
                        "price_drawdown_3y": latest["drawdown_3y"],
                        "price_ma20": latest["ma20"],
                        "price_ma60": latest["ma60"],
                        "price_stabilized": bool(latest["stabilized"]),
                        "price_falling_knife": bool(latest["falling_knife"]),
                        "price_range_regime": bool(latest["range_regime"]),
                        "price_zscore20": latest["zscore20"],
                        "ma20_reversion_distance": latest["ma20_reversion_distance"],
                        "technical_t_buy_setup": bool(latest["t_buy_setup"]),
                        "technical_t_exit_setup": bool(latest["t_exit_setup"]),
                        "t_buy_setup_if_core_established": bool(scored_row["durable_eligible"])
                        and bool(latest["t_buy_setup"]),
                    }
                )
            scored = scored.merge(pd.DataFrame(decision_rows), on="asset", how="left", validate="one_to_one")
            scored = allocate_core_targets(scored, config)
            scored["diagnostic_scope"] = "current_2026_constituents_conditioned_history"
            scored["universe_history_status"] = "current_constituents_only_survivorship_bias"
            scored["price_history_status"] = "qfq_latest_vintage_not_pit_adjustment_history"
            scored["financial_availability_rule"] = "max_notice_date_update_date"
            scored["position_state_assumption"] = "flat_start_each_month_entry_screen"
            scored["timing_feature_rule"] = "past_rows_only_on_latest_qfq_vintage"
            scored["promotion_allowed"] = False
            available = pd.to_datetime(scored["available_date"], errors="coerce")
            if (available.dropna() > as_of).any() or scored["historical_backtest_allowed"].astype(bool).any():
                raise AssertionError("historical diagnostic violated PIT or promotion boundary")
            panel_parts.append(scored)
        coverage_rows.append(
            {
                "as_of_date": str(as_of.date()),
                "watchlist_assets": len(watchlist),
                "snapshot_rows": len(scored),
                "data_gate_pass": int((scored.get("data_gate_status", pd.Series(dtype=str)) == "pass").sum()),
                "durable_eligible": int(scored.get("durable_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "build_1_signals": int((scored.get("entry_action", pd.Series(dtype=str)) == "BUILD_1").sum()),
                "diagnostic_target_core_weight": float(scored.get("target_core_weight", pd.Series(dtype=float)).sum()),
                "range_regime_assets": int(scored.get("price_range_regime", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "t_buy_setups_if_core_established": int(
                    scored.get("t_buy_setup_if_core_established", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
                ),
                "promotion_allowed": False,
            }
        )

    panel = pd.concat(panel_parts, ignore_index=True) if panel_parts else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    errors = pd.DataFrame(build_errors, columns=["asset", "as_of_date", "error"])
    exclusions_frame = pd.DataFrame(
        exclusions,
        columns=["asset", "as_of_date", "reason", "first_price_date"],
    )
    build_signals = panel[panel["entry_action"] == "BUILD_1"].copy() if "entry_action" in panel.columns else panel.head(0)
    t_setups = (
        panel[panel["t_buy_setup_if_core_established"].fillna(False).astype(bool)].copy()
        if "t_buy_setup_if_core_established" in panel.columns
        else panel.head(0)
    )
    paths = {
        "panel": output / "historical_diagnostic_panel.csv",
        "coverage": output / "monthly_coverage.csv",
        "build_signals": output / "build_1_signals.csv",
        "t_setups": output / "conditional_t_buy_setups.csv",
        "errors": output / "build_errors.csv",
        "exclusions": output / "expected_exclusions.csv",
        "summary": output / "summary.json",
        "manifest": output / "run_manifest.json",
    }
    panel.to_csv(paths["panel"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    build_signals.to_csv(paths["build_signals"], index=False, encoding="utf-8-sig")
    t_setups.to_csv(paths["t_setups"], index=False, encoding="utf-8-sig")
    errors.to_csv(paths["errors"], index=False, encoding="utf-8-sig")
    exclusions_frame.to_csv(paths["exclusions"], index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "months": len(coverage),
        "watchlist_assets": len(watchlist),
        "panel_rows": len(panel),
        "build_1_signal_rows": len(build_signals),
        "months_with_build_1": int((coverage.get("build_1_signals", pd.Series(dtype=int)) > 0).sum()),
        "conditional_t_buy_setup_rows": len(t_setups),
        "build_error_rows": len(errors),
        "expected_exclusion_rows": len(exclusions_frame),
        "diagnostic_only": True,
        "historical_backtest_allowed": False,
        "promotion_allowed": False,
        "blocking_reasons": [
            "current_constituents_only_survivorship_bias",
            "qfq_adjustment_history_has_no_pit_vintages",
            "no_all_status_security_master_or_historical_industry_membership",
        ],
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "generated_at": summary["generated_at"],
        "diagnostic_only": True,
        "input_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)} for path in sorted(set(input_paths))
        ],
        "outputs": {key: str(path.relative_to(root)) for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-31")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    paths = run_historical_diagnostic(ROOT, args.start, args.end, args.max_assets, args.output_directory)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(json.dumps({"summary": summary, "outputs": {key: str(path) for key, path in paths.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
