"""Cross-check sparse stock adjustment events against independent raw closes.

Tushare ``pre_close`` is retained as a diagnostic only: older corporate-action
rows often keep the unadjusted prior close. The review gate therefore uses the
realized adjusted-close jump across a factor transition and quarantines only
large unresolved jumps for corporate-action review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FACTOR_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_adjustment_factor.csv"
DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
ACCEPTANCE_MANIFEST = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine" / "agent_run_manifest.json"
)
QUARANTINE_PATH = (
    ROOT / "outputs" / "agent_runs" / "v3_45_1" / "daily_only_data_acceptance_quarantine" / "ohlc_quarantine_rows.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "stock_adjustment_validation"
CHECKS_PATH = OUTPUT_DIR / "factor_transition_checks.csv"
EXCEPTIONS_PATH = OUTPUT_DIR / "unresolved_continuous_trading_jumps.csv"
LONG_GAP_REVIEW_PATH = OUTPUT_DIR / "long_suspension_large_jumps.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
LARGE_ADJUSTED_JUMP = 0.30
VERY_LARGE_ADJUSTED_JUMP = 1.00
CONTINUOUS_TRADING_MAX_CALENDAR_GAP_DAYS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _date_key(value: Any) -> str:
    return str(value).replace("-", "")[:8]


def _load_quarantine(path: Path = QUARANTINE_PATH) -> set[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    required = {"ts_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OHLC quarantine missing columns: {missing}")
    return set(zip(frame["trade_date"].map(_date_key), frame["ts_code"].astype(str)))


def _load_factor_events(path: Path = FACTOR_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    required = {"asset", "effective_date", "adj_factor"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"adjustment factors missing columns: {missing}")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    if frame[list(required)].isna().any(axis=None) or (frame["adj_factor"] <= 0).any():
        raise ValueError("adjustment factors contain invalid keys or values")
    if frame.duplicated(["asset", "effective_date"]).any():
        raise ValueError("adjustment factors contain duplicate asset-date events")
    return frame.sort_values(["effective_date", "asset"]).reset_index(drop=True)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_factor_transitions(
    factors: pd.DataFrame,
    daily_dir: Path = DAILY_DIR,
    quarantine: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, str]]]:
    quarantine = quarantine or set()
    event_rows = list(factors[["effective_date", "asset", "adj_factor"]].itertuples(index=False, name=None))
    factor_assets = set(factors["asset"])
    current: dict[str, tuple[float, pd.Timestamp, int]] = {}
    previous_trade: dict[str, tuple[str, float, float, int]] = {}
    checks: list[dict[str, Any]] = []
    pointer = 0
    files_read = 0
    empty_files_skipped = 0
    rows_read = 0
    rows_quarantined = 0
    first_trade_date: str | None = None
    last_trade_date: str | None = None
    source_inputs: list[dict[str, str]] = []

    for path in sorted(daily_dir.glob("trade_date=*.csv")):
        try:
            source_path = _relative(path)
        except ValueError:
            source_path = str(path.resolve())
        source_inputs.append({"path": source_path, "sha256": _sha256(path)})
        trade_date = path.stem.split("=", 1)[-1]
        trade_ts = pd.Timestamp(trade_date)
        while pointer < len(event_rows) and event_rows[pointer][0] <= trade_ts:
            effective_date, asset, factor = event_rows[pointer]
            prior_serial = current.get(asset, (math.nan, effective_date, 0))[2]
            current[asset] = (float(factor), effective_date, prior_serial + 1)
            pointer += 1

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"ts_code", "trade_date", "close", "pre_close"}
                if reader.fieldnames is None:
                    empty_files_skipped += 1
                    continue
                first_row = next(reader, None)
                if first_row is None:
                    empty_files_skipped += 1
                    continue
                missing = required.difference(reader.fieldnames)
                if missing:
                    raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
                files_read += 1
                for row in itertools.chain([first_row], reader):
                    rows_read += 1
                    ts_code = str(row["ts_code"])
                    asset = ts_code[:6]
                    if asset not in factor_assets:
                        continue
                    row_date = _date_key(row["trade_date"])
                    if (row_date, ts_code) in quarantine:
                        rows_quarantined += 1
                        continue
                    state = current.get(asset)
                    close = _safe_float(row["close"])
                    pre_close = _safe_float(row["pre_close"])
                    if state is None or close is None or close <= 0:
                        continue
                    factor, latest_effective_date, serial = state
                    prior = previous_trade.get(asset)
                    if prior is not None:
                        prior_date, prior_close, prior_factor, prior_serial = prior
                        if serial != prior_serial and not math.isclose(factor, prior_factor, abs_tol=1e-12, rel_tol=0.0):
                            adjusted_jump = close * factor / (prior_close * prior_factor) - 1.0
                            calendar_gap_days = (pd.Timestamp(row_date) - pd.Timestamp(prior_date)).days
                            large_jump = abs(adjusted_jump) > LARGE_ADJUSTED_JUMP
                            pre_close_error = (
                                pre_close * factor / (prior_close * prior_factor) - 1.0
                                if pre_close is not None and pre_close > 0
                                else math.nan
                            )
                            checks.append(
                                {
                                    "asset": asset,
                                    "previous_trade_date": prior_date,
                                    "trade_date": row_date,
                                    "calendar_gap_days": calendar_gap_days,
                                    "latest_factor_effective_date": latest_effective_date.date().isoformat(),
                                    "events_since_previous_trade": serial - prior_serial,
                                    "old_factor": prior_factor,
                                    "new_factor": factor,
                                    "factor_ratio": factor / prior_factor,
                                    "previous_close": prior_close,
                                    "pre_close": pre_close,
                                    "close": close,
                                    "adjusted_close_jump": adjusted_jump,
                                    "pre_close_continuity_error_diagnostic": pre_close_error,
                                    "large_adjusted_jump": large_jump,
                                    "very_large_adjusted_jump": abs(adjusted_jump) > VERY_LARGE_ADJUSTED_JUMP,
                                    "continuous_trading_large_jump": bool(
                                        large_jump and calendar_gap_days <= CONTINUOUS_TRADING_MAX_CALENDAR_GAP_DAYS
                                    ),
                                    "long_gap_large_jump": bool(
                                        large_jump and calendar_gap_days > CONTINUOUS_TRADING_MAX_CALENDAR_GAP_DAYS
                                    ),
                                }
                            )
                    previous_trade[asset] = (row_date, close, factor, serial)
                    first_trade_date = row_date if first_trade_date is None else min(first_trade_date, row_date)
                    last_trade_date = row_date if last_trade_date is None else max(last_trade_date, row_date)
        except UnicodeDecodeError as exc:
            raise ValueError(f"cannot decode raw daily file: {path}") from exc

    columns = [
        "asset",
        "previous_trade_date",
        "trade_date",
        "calendar_gap_days",
        "latest_factor_effective_date",
        "events_since_previous_trade",
        "old_factor",
        "new_factor",
        "factor_ratio",
        "previous_close",
        "pre_close",
        "close",
        "adjusted_close_jump",
        "pre_close_continuity_error_diagnostic",
        "large_adjusted_jump",
        "very_large_adjusted_jump",
        "continuous_trading_large_jump",
        "long_gap_large_jump",
    ]
    output = pd.DataFrame(checks, columns=columns)
    absolute = output["adjusted_close_jump"].abs() if not output.empty else pd.Series(dtype=float)
    summary = {
        "daily_files_read": files_read,
        "empty_daily_files_skipped": empty_files_skipped,
        "daily_source_files_hashed": len(source_inputs),
        "raw_rows_read": rows_read,
        "quarantined_rows_skipped": rows_quarantined,
        "daily_coverage_start": first_trade_date,
        "daily_coverage_end": last_trade_date,
        "transition_checks": int(len(output)),
        "transition_assets": int(output["asset"].nunique()) if not output.empty else 0,
        "large_adjusted_jump_count": int((absolute > LARGE_ADJUSTED_JUMP).sum()),
        "very_large_adjusted_jump_count": int((absolute > VERY_LARGE_ADJUSTED_JUMP).sum()),
        "continuous_trading_large_jump_count": int(output["continuous_trading_large_jump"].sum()) if not output.empty else 0,
        "long_gap_large_jump_count": int(output["long_gap_large_jump"].sum()) if not output.empty else 0,
        "absolute_adjusted_jump_quantiles": {
            str(quantile): round(float(absolute.quantile(quantile)), 6) if not absolute.empty else None
            for quantile in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
        },
        "method_boundary": (
            "large adjusted-close jumps after long suspensions require corporate-action review; only large jumps "
            "within the continuous-trading calendar-gap threshold fail qualification; Tushare pre_close remains "
            "diagnostic because it is not consistently ex-right adjusted"
        ),
    }
    return output, summary, source_inputs


def run_validation() -> dict[str, Any]:
    if not FACTOR_PATH.is_file() or not ACCEPTANCE_MANIFEST.is_file() or not QUARANTINE_PATH.is_file():
        raise FileNotFoundError("factor file and accepted daily-only lineage are required")
    acceptance = json.loads(ACCEPTANCE_MANIFEST.read_text(encoding="utf-8"))
    if not acceptance.get("acceptance_pass") or acceptance.get("data_decision") != "accepted_after_row_level_ohlc_quarantine":
        raise ValueError("daily-only source is not accepted with the required OHLC quarantine")
    factors = _load_factor_events()
    checks, summary, daily_inputs = validate_factor_transitions(factors, DAILY_DIR, _load_quarantine())
    exceptions = checks[checks["continuous_trading_large_jump"]].copy()
    long_gap_review = checks[checks["long_gap_large_jump"]].copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks.to_csv(CHECKS_PATH, index=False, encoding="utf-8-sig")
    exceptions.to_csv(EXCEPTIONS_PATH, index=False, encoding="utf-8-sig")
    long_gap_review.to_csv(LONG_GAP_REVIEW_PATH, index=False, encoding="utf-8-sig")
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "CONTINUOUS_TRADING_DISCONTINUITY" if not exceptions.empty else "PASS",
        "review_status": "LONG_SUSPENSION_REVIEW_REQUIRED" if not long_gap_review.empty else "CLEAR",
        "historical_backtest_allowed": bool(exceptions.empty),
        "model_promotion_allowed": False,
        "thresholds": {
            "large_adjusted_jump_absolute": LARGE_ADJUSTED_JUMP,
            "very_large_adjusted_jump_absolute": VERY_LARGE_ADJUSTED_JUMP,
            "continuous_trading_max_calendar_gap_days": CONTINUOUS_TRADING_MAX_CALENDAR_GAP_DAYS,
        },
        **summary,
        "outputs": {
            "checks": _relative(CHECKS_PATH),
            "exceptions": _relative(EXCEPTIONS_PATH),
            "long_gap_review": _relative(LONG_GAP_REVIEW_PATH),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    code_path = Path(__file__).resolve()
    manifest = {
        "created_at": report["created_at"],
        "inputs": [
            {"path": _relative(FACTOR_PATH), "sha256": _sha256(FACTOR_PATH)},
            {"path": _relative(ACCEPTANCE_MANIFEST), "sha256": _sha256(ACCEPTANCE_MANIFEST)},
            {"path": _relative(QUARANTINE_PATH), "sha256": _sha256(QUARANTINE_PATH)},
            *daily_inputs,
        ],
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "output_path": _relative(CHECKS_PATH),
        "output_sha256": _sha256(CHECKS_PATH),
        "exceptions_path": _relative(EXCEPTIONS_PATH),
        "exceptions_sha256": _sha256(EXCEPTIONS_PATH),
        "long_gap_review_path": _relative(LONG_GAP_REVIEW_PATH),
        "long_gap_review_sha256": _sha256(LONG_GAP_REVIEW_PATH),
        "report_path": _relative(REPORT_PATH),
        "report_sha256": _sha256(REPORT_PATH),
        "daily_source_directory": _relative(DAILY_DIR),
        "daily_source_file_count": int(len(list(DAILY_DIR.glob("trade_date=*.csv")))),
        "qualification_status": report["qualification_status"],
        "review_status": report["review_status"],
        "continuous_trading_large_jump_count": report["continuous_trading_large_jump_count"],
        "long_gap_large_jump_count": report["long_gap_large_jump_count"],
        "historical_backtest_allowed": report["historical_backtest_allowed"],
        "model_promotion_allowed": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run_validation(), ensure_ascii=False))


if __name__ == "__main__":
    main()
