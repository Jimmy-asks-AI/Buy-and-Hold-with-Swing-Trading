"""Cross-source validation for historical stock execution-status events.

The governed status-event dataset remains in the observation layer until all
independent validation scopes pass. BaoStock provides a long daily panel with
era-specific gates, JoinQuant provides a recent independent daily panel, SSE
factbooks are audited as governed inputs, and Tushare traded rows verify
tradable-delisting and restoration dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .pit_sse_status_announcement_collector import (
    KEYWORDS,
    QUERY_URL,
    _session as sse_session,
    parse_asset_events,
)
from .pit_stock_market_history_builder import ROOT, _sha256
from .pit_stock_status_event_reconciler import select_factbook_reference_candidates


STATUS_EVENTS_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "stock_execution_status_events_reconciled.csv"
)
STATUS_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_status_event_reconciler_latest.json"
FACTBOOK_RESTORATION_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "sse_factbook_restoration_events.csv"
)
FACTBOOK_REFERENCE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "sse_factbook_status_reference_events.csv"
)
FACTBOOK_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_factbook_status_latest.json"
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
CALENDAR_PATH = ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv"
BAOSTOCK_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_baostock_stock_daily"
BAOSTOCK_SUMMARY_PATH = BAOSTOCK_DIR / "derived_asset_summary.csv"
JQ_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "joinquant_trade_state"
)
SSE_HOLDOUT_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "validation_sources"
    / "sse_status_holdout"
)
TUSHARE_DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "stock_status_event_validation"
BAOSTOCK_SUMMARY_OUTPUT = OUTPUT_DIR / "baostock_daily_summary.csv"
BAOSTOCK_MISMATCHES_OUTPUT = OUTPUT_DIR / "baostock_status_mismatches.csv.gz"
BAOSTOCK_TRANSITIONS_OUTPUT = OUTPUT_DIR / "baostock_transition_checks.csv"
JQ_CHECKS_OUTPUT = OUTPUT_DIR / "joinquant_daily_checks.csv.gz"
JQ_TRANSITIONS_OUTPUT = OUTPUT_DIR / "joinquant_transition_checks.csv"
BAOSTOCK_ERAS_OUTPUT = OUTPUT_DIR / "baostock_era_metrics.csv"
FACTBOOK_INGESTION_OUTPUT = OUTPUT_DIR / "factbook_ingestion_checks.csv"
DELISTING_CHECKS_OUTPUT = OUTPUT_DIR / "delisting_trade_day_checks.csv"
RESTORATION_CHECKS_OUTPUT = OUTPUT_DIR / "factbook_restoration_trade_day_checks.csv"
QUALIFICATION_CHECKS_OUTPUT = OUTPUT_DIR / "qualification_checks.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

HOLDOUT_VERSION = "sse_factbook_status_holdout_v1"
DEFAULT_HOLDOUT_ASSETS = 36
BACKTEST_START = pd.Timestamp("2000-01-01")
# BaoStock/JoinQuant ``is_st`` is a literal ST flag. Legacy PT special-transfer
# rows are execution-restricted but are not encoded as ST by those providers,
# so they require separate execution-rule evidence and are outside this field
# comparison.
BINARY_VALIDATION_STATUSES = {"normal", "risk_warning"}
VALIDATION_ERAS = {
    "2000_2009": (pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
    "2010_2019": (pd.Timestamp("2010-01-01"), pd.Timestamp("2019-12-31")),
    "2020_present": (pd.Timestamp("2020-01-01"), pd.Timestamp.max.normalize()),
}

THRESHOLDS: dict[str, float | int] = {
    "baostock_min_assets": 180,
    "baostock_min_daily_checks": 750_000,
    "baostock_min_actual_st_rows": 50_000,
    "baostock_daily_match_min": 0.995,
    "baostock_st_precision_min": 0.985,
    "baostock_st_recall_min": 0.985,
    "baostock_min_transition_checks": 150,
    "baostock_transition_within_one_session_min": 0.95,
    "baostock_era_min_assets": 150,
    "baostock_era_min_daily_checks": 100_000,
    "baostock_era_min_actual_st_rows": 1_000,
    "baostock_era_daily_match_min": 0.995,
    "baostock_era_st_precision_min": 0.985,
    "baostock_era_st_recall_min": 0.985,
    "baostock_era_min_transition_checks": 30,
    "baostock_era_transition_within_one_session_min": 0.95,
    "joinquant_min_assets": 80,
    "joinquant_min_daily_checks": 15_000,
    "joinquant_min_actual_st_rows": 50,
    "joinquant_daily_match_min": 0.995,
    "joinquant_st_precision_min": 0.98,
    "joinquant_st_recall_min": 0.98,
    "joinquant_min_transition_checks": 5,
    "joinquant_transition_within_one_session_min": 0.95,
    "factbook_min_reference_candidates": 200,
    "delisting_min_trade_day_checks": 50,
    "delisting_trade_day_match_min": 0.95,
    "restoration_min_trade_day_checks": 30,
    "restoration_trade_day_match_min": 0.95,
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.csv.gz")
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
            lineterminator="\n",
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
    else:
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
    temporary.replace(path)


def _as_nullable_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(pd.NA, index=values.index, dtype="boolean")
    output.loc[numeric.eq(1)] = True
    output.loc[numeric.eq(0)] = False
    unresolved = output.isna() & values.notna()
    if unresolved.any():
        text = values.astype(str).str.strip().str.lower()
        output.loc[unresolved & text.isin({"true", "t", "yes"})] = True
        output.loc[unresolved & text.isin({"false", "f", "no"})] = False
    return output


def load_status_events(as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(STATUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("as_of_date") != as_of.date().isoformat():
        raise ValueError("status-event manifest as-of date mismatch")
    if manifest.get("output_path") != _relative(STATUS_EVENTS_PATH):
        raise ValueError("status-event manifest output path mismatch")
    if manifest.get("output_sha256") != _sha256(STATUS_EVENTS_PATH):
        raise ValueError("status-event output hash mismatch")
    if manifest.get("code_sha256") != _sha256(ROOT / str(manifest["code_path"])):
        raise ValueError("status-event builder code hash mismatch")
    events = pd.read_csv(STATUS_EVENTS_PATH, dtype={"asset": str})
    required = {
        "asset",
        "effective_date",
        "execution_status",
        "is_st",
        "available_date",
        "source_coverage_mode",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"status-event dataset missing columns: {missing}")
    events["asset"] = events["asset"].astype(str).str.zfill(6)
    events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
    events["available_date"] = pd.to_datetime(events["available_date"], errors="coerce").dt.normalize()
    events["is_st"] = _as_nullable_bool(events["is_st"])
    if events[["asset", "effective_date", "execution_status", "is_st", "available_date"]].isna().any(axis=None):
        raise ValueError("status-event dataset contains incomplete keys")
    if events.duplicated(["asset", "effective_date"]).any():
        raise ValueError("status-event dataset contains duplicate asset dates")
    if (events["available_date"] > events["effective_date"]).any():
        raise ValueError("status-event dataset contains post-effective availability dates")
    return events.sort_values(["asset", "effective_date"]).reset_index(drop=True), manifest


def load_factbook_validation_events(as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(FACTBOOK_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("as_of_date") != as_of.date().isoformat():
        raise ValueError("factbook manifest as-of date mismatch")
    if manifest.get("qualification_status") != "READY_FOR_RECONCILIATION":
        raise ValueError("factbook restoration evidence is incomplete")
    if manifest.get("reference_qualification_status") != "READY_FOR_HOLDOUT":
        raise ValueError("factbook holdout evidence is incomplete")
    expected_paths = {
        "output_path": FACTBOOK_RESTORATION_PATH,
        "reference_output_path": FACTBOOK_REFERENCE_PATH,
    }
    for manifest_key, path in expected_paths.items():
        if manifest.get(manifest_key) != _relative(path):
            raise ValueError(f"factbook manifest path mismatch: {manifest_key}")
        hash_key = manifest_key.replace("path", "sha256")
        if manifest.get(hash_key) != _sha256(path):
            raise ValueError(f"factbook output hash mismatch: {manifest_key}")
    if manifest.get("code_sha256") != _sha256(ROOT / str(manifest["code_path"])):
        raise ValueError("factbook collector code hash mismatch")
    restoration = pd.read_csv(FACTBOOK_RESTORATION_PATH, dtype={"asset": str})
    reference = pd.read_csv(FACTBOOK_REFERENCE_PATH, dtype={"asset": str})
    for frame in (restoration, reference):
        frame["asset"] = frame["asset"].astype(str).str.zfill(6)
        frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.normalize()
        frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce").dt.normalize()
        frame["is_st"] = _as_nullable_bool(frame["is_st"])
    reference["used_in_reconciliation"] = _as_nullable_bool(reference["used_in_reconciliation"])
    if "binary_state_change" not in reference.columns:
        raise ValueError("factbook reference evidence is missing binary_state_change semantics")
    reference["binary_state_change"] = _as_nullable_bool(reference["binary_state_change"])
    if restoration[["asset", "effective_date", "execution_status"]].isna().any(axis=None):
        raise ValueError("factbook restoration evidence contains incomplete keys")
    if reference[
        ["asset", "effective_date", "execution_status", "used_in_reconciliation", "binary_state_change"]
    ].isna().any(axis=None):
        raise ValueError("factbook reference evidence contains incomplete keys")
    status_reference = reference[~reference["used_in_reconciliation"].astype(bool)].copy()
    return restoration, status_reference, manifest


def build_event_index(events: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for asset, group in events.groupby("asset", sort=False):
        ordered = group.sort_values("effective_date")
        index[str(asset)] = (
            ordered["effective_date"].to_numpy(dtype="datetime64[ns]"),
            ordered["is_st"].astype(bool).to_numpy(),
            ordered["execution_status"].astype(str).to_numpy(),
        )
    return index


def apply_status_events(frame: pd.DataFrame, event_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    expected_st = np.zeros(len(output), dtype=bool)
    expected_status = np.full(len(output), "unknown", dtype=object)
    expected_known = np.zeros(len(output), dtype=bool)
    for asset, positions in output.groupby("asset", sort=False).indices.items():
        indexed = event_index.get(str(asset))
        if indexed is None:
            continue
        dates, is_st, statuses = indexed
        row_dates = output.loc[positions, "date"].to_numpy(dtype="datetime64[ns]")
        event_positions = np.searchsorted(dates, row_dates, side="right") - 1
        known = event_positions >= 0
        expected_known[positions] = known
        if known.any():
            target = np.asarray(positions)[known]
            expected_st[target] = is_st[event_positions[known]]
            expected_status[target] = statuses[event_positions[known]]
    output["expected_status"] = expected_status
    output["expected_is_st"] = pd.Series(expected_st, dtype="boolean")
    output["expected_state_known"] = expected_known
    return output


def _daily_metrics(checks: pd.DataFrame, actual_column: str) -> dict[str, Any]:
    actual = _as_nullable_bool(checks[actual_column])
    expected = _as_nullable_bool(checks["expected_is_st"])
    status_scope = checks["expected_status"].isin(BINARY_VALIDATION_STATUSES)
    valid = (
        checks["expected_state_known"].fillna(False).astype(bool)
        & status_scope
        & actual.notna()
        & expected.notna()
    )
    actual = actual.loc[valid].astype(bool)
    expected = expected.loc[valid].astype(bool)
    true_positive = int((actual & expected).sum())
    actual_positive = int(actual.sum())
    expected_positive = int(expected.sum())
    matches = int(actual.eq(expected).sum())
    return {
        "daily_checks": int(valid.sum()),
        "daily_matches": matches,
        "daily_match_ratio": round(matches / int(valid.sum()), 8) if valid.any() else 0.0,
        "actual_st_rows": actual_positive,
        "expected_st_rows": expected_positive,
        "true_positive_rows": true_positive,
        "st_precision": round(true_positive / expected_positive, 8) if expected_positive else 0.0,
        "st_recall": round(true_positive / actual_positive, 8) if actual_positive else 0.0,
        "false_positive_rows": int((~actual & expected).sum()),
        "false_negative_rows": int((actual & ~expected).sum()),
    }


def actual_transition_rows(frame: pd.DataFrame, actual_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["asset", "actual_date", "actual_is_st"])
    source = frame[["asset", "date", actual_column]].copy()
    source[actual_column] = _as_nullable_bool(source[actual_column])
    source = source.dropna(subset=["date", actual_column]).sort_values(["asset", "date"])
    changed = source.groupby("asset", sort=False)[actual_column].transform(lambda values: values.ne(values.shift()))
    transitions = source.loc[changed].rename(columns={"date": "actual_date", actual_column: "actual_is_st"})
    return transitions.reset_index(drop=True)


def expected_binary_transitions(events: pd.DataFrame) -> pd.DataFrame:
    source = events[["asset", "effective_date", "is_st", "execution_status", "source_coverage_mode"]].copy()
    source = source[source["execution_status"].isin(BINARY_VALIDATION_STATUSES)].copy()
    source["is_st"] = _as_nullable_bool(source["is_st"])
    prior = source.groupby("asset", sort=False)["is_st"].shift()
    output = source[prior.notna() & source["is_st"].ne(prior)].copy()
    return output.reset_index(drop=True)


def compare_transition_dates(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    source_ranges: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "asset",
        "effective_date",
        "expected_is_st",
        "actual_date",
        "session_distance",
        "exact_match",
        "within_one_session",
        "source_coverage_mode",
    ]
    if expected.empty or actual.empty or source_ranges.empty:
        return pd.DataFrame(columns=columns)
    calendar_positions = {pd.Timestamp(date).normalize(): position for position, date in enumerate(calendar)}
    actual_groups = {
        asset: group.sort_values("actual_date").reset_index(drop=True)
        for asset, group in actual.groupby("asset", sort=False)
    }
    ranges = source_ranges.set_index("asset")
    rows: list[dict[str, Any]] = []
    for event in expected.itertuples(index=False):
        asset = str(event.asset)
        if asset not in ranges.index or asset not in actual_groups:
            continue
        source_range = ranges.loc[asset]
        effective = pd.Timestamp(event.effective_date).normalize()
        if effective < pd.Timestamp(source_range.min_date) or effective > pd.Timestamp(source_range.max_date):
            continue
        target = bool(event.is_st)
        candidates = actual_groups[asset]
        candidates = candidates[candidates["actual_is_st"].astype(bool).eq(target)]
        candidates = candidates[candidates["actual_date"].gt(pd.Timestamp(source_range.min_date))]
        actual_date: pd.Timestamp | pd.NaT = pd.NaT
        distance: float = np.nan
        if not candidates.empty and effective in calendar_positions:
            candidate_distances = candidates["actual_date"].map(
                lambda date: abs(calendar_positions.get(pd.Timestamp(date).normalize(), 10**9) - calendar_positions[effective])
            )
            nearest = candidate_distances.idxmin()
            actual_date = pd.Timestamp(candidates.loc[nearest, "actual_date"]).normalize()
            distance = float(candidate_distances.loc[nearest])
        rows.append(
            {
                "asset": asset,
                "effective_date": effective,
                "expected_is_st": target,
                "actual_date": actual_date,
                "session_distance": distance,
                "exact_match": bool(distance == 0),
                "within_one_session": bool(distance <= 1),
                "source_coverage_mode": event.source_coverage_mode,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _source_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("asset", as_index=False)["date"]
        .agg(min_date="min", max_date="max")
        .sort_values("asset")
        .reset_index(drop=True)
    )


def build_baostock_era_metrics(daily: pd.DataFrame, transition_checks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for era, (start, end) in VALIDATION_ERAS.items():
        era_daily = daily[daily["date"].between(start, end)].copy() if not daily.empty else daily.copy()
        metrics = _daily_metrics(era_daily, "actual_is_st")
        era_transitions = (
            transition_checks[transition_checks["effective_date"].between(start, end)].copy()
            if not transition_checks.empty
            else transition_checks.copy()
        )
        rows.append(
            {
                "era": era,
                "start_date": start.date().isoformat(),
                "end_date": (
                    pd.Timestamp(era_daily["date"].max()).date().isoformat()
                    if not era_daily.empty
                    else start.date().isoformat()
                ),
                "assets": int(era_daily["asset"].nunique()) if not era_daily.empty else 0,
                **metrics,
                "transition_checks": int(len(era_transitions)),
                "transition_exact_ratio": _ratio(era_transitions, "exact_match"),
                "transition_within_one_session_ratio": _ratio(era_transitions, "within_one_session"),
            }
        )
    return pd.DataFrame(rows)


def build_factbook_ingestion_metrics(
    reference: pd.DataFrame,
    status_manifest: dict[str, Any],
    factbook_manifest: dict[str, Any],
) -> dict[str, Any]:
    eligible = select_factbook_reference_candidates(reference)
    lineage_items = [
        item
        for item in status_manifest.get("inputs", [])
        if isinstance(item, dict) and item.get("role") == "sse_factbook_status_tables"
    ]
    lineage_match = bool(
        len(lineage_items) == 1
        and lineage_items[0].get("path") == _relative(FACTBOOK_REFERENCE_PATH)
        and lineage_items[0].get("sha256") == _sha256(FACTBOOK_REFERENCE_PATH)
    )
    manifest_candidates = int(status_manifest.get("factbook_reference_candidate_rows", -1))
    future_available = int(reference["available_date"].gt(reference["effective_date"]).sum())
    return {
        "collector_version": factbook_manifest.get("collector_version"),
        "reference_rows": int(len(reference)),
        "eligible_reference_candidates": int(len(eligible)),
        "reconciler_manifest_candidates": manifest_candidates,
        "candidate_count_match": manifest_candidates == len(eligible),
        "retained_after_state_collapse": int(
            status_manifest.get("factbook_reference_rows_retained_after_state_collapse", 0)
        ),
        "lineage_hash_match": lineage_match,
        "future_available_rows": future_available,
        "circular_alignment_excluded_from_qualification": True,
    }


def load_baostock_checks(
    event_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    summary = pd.read_csv(BAOSTOCK_SUMMARY_PATH, dtype={"asset": str})
    completed = set(summary.loc[summary["trade_status"].eq("completed"), "asset"].astype(str).str.zfill(6))
    checks: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(BAOSTOCK_DIR.glob("[0-9][0-9][0-9][0-9][0-9][0-9].csv.gz")):
        asset = path.name[:6]
        if asset not in completed:
            continue
        try:
            frame = pd.read_csv(path, compression="gzip", usecols=["date", "isST"], dtype={"isST": str})
            frame.insert(0, "asset", asset)
            frame = frame.rename(columns={"isST": "actual_is_st"})
            frame = apply_status_events(frame, event_index)
            frame["actual_is_st"] = _as_nullable_bool(frame["actual_is_st"])
            checks.append(frame)
            inputs.append({"role": "baostock_status_history", "path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append({"asset": asset, "source": "baostock", "error": f"{type(exc).__name__}: {exc}"})
    if not checks:
        return pd.DataFrame(), pd.DataFrame(), inputs, errors
    daily = pd.concat(checks, ignore_index=True)
    transitions = actual_transition_rows(daily, "actual_is_st")
    return daily, transitions, inputs, errors


def load_joinquant_checks(
    event_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(JQ_DIR.glob("*.csv.gz")):
        asset = path.name[:6]
        try:
            frame = pd.read_csv(path, compression="gzip", usecols=["date", "asset", "is_st"], dtype={"asset": str})
            frame["asset"] = frame["asset"].astype(str).str.zfill(6)
            if not frame["asset"].eq(asset).all():
                raise ValueError("cached JoinQuant asset does not match filename")
            frame = frame.rename(columns={"is_st": "actual_is_st"})
            frame = apply_status_events(frame, event_index)
            frame["actual_is_st"] = _as_nullable_bool(frame["actual_is_st"])
            frames.append(frame)
            inputs.append({"role": "joinquant_status_history", "path": _relative(path), "sha256": _sha256(path)})
        except Exception as exc:
            errors.append({"asset": asset, "source": "joinquant", "error": f"{type(exc).__name__}: {exc}"})
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), inputs, errors
    daily = pd.concat(frames, ignore_index=True)
    transitions = actual_transition_rows(daily, "actual_is_st")
    return daily, transitions, inputs, errors


def select_sse_holdout_assets(events: pd.DataFrame, master: pd.DataFrame, count: int) -> pd.DataFrame:
    exchange = master.loc[master["event_type"].eq("listing"), ["asset", "exchange"]].drop_duplicates("asset")
    exchange["asset"] = exchange["asset"].astype(str).str.zfill(6)
    transitions = expected_binary_transitions(events).merge(exchange, on="asset", how="left", validate="many_to_one")
    candidates = transitions[
        transitions["exchange"].eq("SSE")
        & transitions["source_coverage_mode"].eq("dated_history")
        & transitions["effective_date"].ge(BACKTEST_START)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["asset", "first_transition_date", "transition_count", "selection_stratum"])
    summary = (
        candidates.groupby("asset", as_index=False)
        .agg(first_transition_date=("effective_date", "min"), transition_count=("effective_date", "size"))
    )
    summary["selection_stratum"] = (summary["first_transition_date"].dt.year // 5 * 5).astype(str)
    summary["selection_hash"] = summary["asset"].map(
        lambda asset: hashlib.sha256(f"{HOLDOUT_VERSION}|{asset}".encode()).hexdigest()
    )
    groups = {
        stratum: group.sort_values("selection_hash").to_dict("records")
        for stratum, group in summary.groupby("selection_stratum", sort=True)
    }
    selected: list[dict[str, Any]] = []
    strata = sorted(groups)
    while len(selected) < min(count, len(summary)):
        added = False
        for stratum in strata:
            if groups[stratum]:
                selected.append(groups[stratum].pop(0))
                added = True
                if len(selected) >= min(count, len(summary)):
                    break
        if not added:
            break
    return pd.DataFrame(selected).sort_values("asset").reset_index(drop=True)


def select_factbook_holdout_assets(reference: pd.DataFrame, count: int) -> pd.DataFrame:
    binary = (
        _as_nullable_bool(reference["binary_state_change"]).fillna(False)
        if "binary_state_change" in reference.columns
        else pd.Series(True, index=reference.index, dtype=bool)
    )
    source = reference[
        binary.astype(bool)
        & reference["effective_date"].ge(BACKTEST_START)
        & reference["execution_status"].isin({"normal", "risk_warning", "special_transfer"})
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=["asset", "first_reference_date", "reference_count", "selection_stratum"])
    source["selection_stratum"] = (
        (source["effective_date"].dt.year // 5 * 5).astype(str)
        + "_"
        + source["execution_status"].replace({"special_transfer": "risk_warning"}).astype(str)
    )
    summary = (
        source.sort_values("effective_date")
        .groupby("asset", as_index=False)
        .agg(
            first_reference_date=("effective_date", "min"),
            reference_count=("effective_date", "size"),
            selection_stratum=("selection_stratum", "first"),
        )
    )
    summary["selection_hash"] = summary["asset"].map(
        lambda asset: hashlib.sha256(f"{HOLDOUT_VERSION}|{asset}".encode()).hexdigest()
    )
    groups = {
        stratum: group.sort_values("selection_hash").to_dict("records")
        for stratum, group in summary.groupby("selection_stratum", sort=True)
    }
    selected: list[dict[str, Any]] = []
    strata = sorted(groups)
    while len(selected) < min(count, len(summary)):
        added = False
        for stratum in strata:
            if groups[stratum]:
                selected.append(groups[stratum].pop(0))
                added = True
                if len(selected) >= min(count, len(summary)):
                    break
        if not added:
            break
    return pd.DataFrame(selected).sort_values("asset").reset_index(drop=True)


def _status_family(status: Any) -> str:
    value = str(status)
    return "st" if value in {"risk_warning", "special_transfer"} else value


def compare_factbook_holdout_events(
    reference: pd.DataFrame,
    candidate_events: pd.DataFrame,
    selected_assets: set[str],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "asset",
        "reference_date",
        "reference_status",
        "candidate_date",
        "candidate_status",
        "session_distance",
        "exact_match",
        "within_one_session",
        "factbook_edition",
        "event_class",
        "source_url",
    ]
    binary = (
        _as_nullable_bool(reference["binary_state_change"]).fillna(False)
        if "binary_state_change" in reference.columns
        else pd.Series(True, index=reference.index, dtype=bool)
    )
    source = reference[
        binary.astype(bool)
        & reference["asset"].isin(selected_assets)
        & reference["effective_date"].ge(BACKTEST_START)
        & reference["execution_status"].isin({"normal", "risk_warning", "special_transfer"})
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=columns)
    candidates = candidate_events.copy()
    candidates["status_family"] = candidates["execution_status"].map(_status_family)
    candidate_groups = {
        asset: group.sort_values("effective_date").reset_index(drop=True)
        for asset, group in candidates.groupby("asset", sort=False)
    }
    calendar_positions = {pd.Timestamp(date).normalize(): position for position, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for event in source.itertuples(index=False):
        reference_date = pd.Timestamp(event.effective_date).normalize()
        family = _status_family(event.execution_status)
        matching = candidate_groups.get(str(event.asset), pd.DataFrame())
        if not matching.empty:
            matching = matching[matching["status_family"].eq(family)]
        matched: pd.Series | None = None
        distance = np.nan
        if not matching.empty and reference_date in calendar_positions:
            distances = matching["effective_date"].map(
                lambda date: abs(
                    calendar_positions.get(pd.Timestamp(date).normalize(), 10**9)
                    - calendar_positions[reference_date]
                )
            )
            nearest = distances.idxmin()
            matched = matching.loc[nearest]
            distance = float(distances.loc[nearest])
        rows.append(
            {
                "asset": str(event.asset),
                "reference_date": reference_date,
                "reference_status": str(event.execution_status),
                "candidate_date": matched["effective_date"] if matched is not None else pd.NaT,
                "candidate_status": matched["execution_status"] if matched is not None else "",
                "session_distance": distance,
                "exact_match": bool(distance == 0),
                "within_one_session": bool(distance <= 1),
                "factbook_edition": event.factbook_edition,
                "event_class": event.event_class,
                "source_url": event.source_url,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _holdout_cache_paths(asset: str) -> tuple[Path, Path]:
    return SSE_HOLDOUT_DIR / f"{asset}.json", SSE_HOLDOUT_DIR / f"{asset}.meta.json"


def _valid_holdout_cache(asset: str, as_of: pd.Timestamp) -> bool:
    data_path, meta_path = _holdout_cache_paths(asset)
    if not data_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        meta.get("status") == "success"
        and artifact.get("as_of_date") == as_of.date().isoformat()
        and meta.get("sha256") == _sha256(data_path)
        and meta.get("holdout_version") == HOLDOUT_VERSION
        and artifact.get("query_keywords") == list(KEYWORDS)
    )


def query_sse_holdout_asset(session: requests.Session, asset: str, as_of: pd.Timestamp) -> Path:
    responses: list[dict[str, Any]] = []
    for keyword in KEYWORDS:
        params = {
            "isPagination": "true",
            "productId": asset,
            "keyWord": keyword,
            "securityType": "0101,120100,020100,020200,120200",
            "reportType": "ALL",
            "beginDate": "1990-01-01",
            "endDate": as_of.date().isoformat(),
            "pageHelp.pageSize": "200",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.endPage": "5",
        }
        response = session.get(QUERY_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("pageHelp", {}).get("data", [])
        if not isinstance(rows, list):
            raise ValueError(f"SSE announcement response has invalid rows for {asset}")
        responses.append({"keyword": keyword, "request_url": response.url, "rows": rows})
    artifact = {
        "asset": asset,
        "as_of_date": as_of.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "holdout_version": HOLDOUT_VERSION,
        "query_keywords": list(KEYWORDS),
        "responses": responses,
    }
    data_path, meta_path = _holdout_cache_paths(asset)
    _atomic_json(artifact, data_path)
    _atomic_json(
        {
            "status": "success",
            "asset": asset,
            "holdout_version": HOLDOUT_VERSION,
            "path": _relative(data_path),
            "sha256": _sha256(data_path),
            "rows": int(sum(len(item["rows"]) for item in responses)),
            "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        meta_path,
    )
    return data_path


def collect_sse_holdout(
    selected: pd.DataFrame,
    as_of: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    session = sse_session()
    SSE_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in selected["asset"].astype(str):
        try:
            if not _valid_holdout_cache(asset, as_of):
                query_sse_holdout_asset(session, asset, as_of)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            data_path, meta_path = _holdout_cache_paths(asset)
            artifact = json.loads(data_path.read_text(encoding="utf-8"))
            frame, unresolved = parse_asset_events(artifact, calendar, collapse_state_changes=False)
            if unresolved:
                errors.extend(
                    {"asset": asset, "source": "sse_holdout", "error": row["reason"]}
                    for row in unresolved
                )
            if not frame.empty:
                frames.append(frame)
            inputs.extend(
                [
                    {"role": "sse_holdout_artifact", "path": _relative(data_path), "sha256": _sha256(data_path)},
                    {"role": "sse_holdout_metadata", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
                ]
            )
        except Exception as exc:
            errors.append({"asset": asset, "source": "sse_holdout", "error": f"{type(exc).__name__}: {exc}"})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs, errors


def compare_sse_holdout_events(
    expected: pd.DataFrame,
    official: pd.DataFrame,
    selected_assets: set[str],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "asset",
        "expected_date",
        "expected_is_st",
        "official_date",
        "official_status",
        "session_distance",
        "exact_match",
        "within_one_session",
        "announcement_date",
        "announcement_title",
        "source_url",
    ]
    source = expected[expected["asset"].isin(selected_assets)].copy()
    source = source[source["execution_status"].isin({"normal", "risk_warning", "special_transfer"})]
    if source.empty:
        return pd.DataFrame(columns=columns)
    official = official.copy()
    if not official.empty:
        official["asset"] = official["asset"].astype(str).str.zfill(6)
        official["effective_date"] = pd.to_datetime(official["effective_date"], errors="coerce").dt.normalize()
        official = official[official["execution_status"].isin({"normal", "risk_warning"})]
    official_groups = {
        asset: group.sort_values("effective_date").reset_index(drop=True)
        for asset, group in official.groupby("asset", sort=False)
    }
    calendar_positions = {pd.Timestamp(date).normalize(): position for position, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for event in source.itertuples(index=False):
        expected_date = pd.Timestamp(event.effective_date).normalize()
        expected_is_st = bool(event.is_st)
        candidates = official_groups.get(str(event.asset), pd.DataFrame())
        if not candidates.empty:
            candidates = candidates[candidates["is_st"].astype(bool).eq(expected_is_st)]
        matched: pd.Series | None = None
        distance: float = np.nan
        if not candidates.empty and expected_date in calendar_positions:
            distances = candidates["effective_date"].map(
                lambda date: abs(calendar_positions.get(pd.Timestamp(date).normalize(), 10**9) - calendar_positions[expected_date])
            )
            nearest = distances.idxmin()
            matched = candidates.loc[nearest]
            distance = float(distances.loc[nearest])
        rows.append(
            {
                "asset": str(event.asset),
                "expected_date": expected_date,
                "expected_is_st": expected_is_st,
                "official_date": matched["effective_date"] if matched is not None else pd.NaT,
                "official_status": matched["execution_status"] if matched is not None else "",
                "session_distance": distance,
                "exact_match": bool(distance == 0),
                "within_one_session": bool(distance <= 1),
                "announcement_date": matched["announcement_date"] if matched is not None else pd.NaT,
                "announcement_title": matched["announcement_title"] if matched is not None else "",
                "source_url": matched["source_url"] if matched is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def validate_delisting_trade_days(events: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    delist = events[events["execution_status"].eq("delisting")][["asset", "effective_date"]].copy()
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    exits["asset"] = exits["asset"].astype(str).str.zfill(6)
    exits["delist_date"] = pd.to_datetime(exits["delist_date"], errors="coerce").dt.normalize()
    delist = delist.merge(exits, on="asset", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    date_cache: dict[pd.Timestamp, set[str]] = {}
    for event in delist.itertuples(index=False):
        effective = pd.Timestamp(event.effective_date).normalize()
        path = TUSHARE_DAILY_DIR / f"trade_date={effective:%Y%m%d}.csv"
        traded: bool | pd.NA = pd.NA
        if effective >= BACKTEST_START and path.is_file():
            if effective not in date_cache:
                daily = pd.read_csv(path, usecols=["ts_code"], dtype={"ts_code": str})
                date_cache[effective] = set(daily["ts_code"].astype(str).str[:6])
                inputs.append({"role": "tushare_delisting_trade_date", "path": _relative(path), "sha256": _sha256(path)})
            traded = str(event.asset) in date_cache[effective]
        rows.append(
            {
                "asset": str(event.asset),
                "effective_date": effective,
                "master_delist_date": event.delist_date,
                "within_lifecycle": bool(pd.isna(event.delist_date) or effective <= pd.Timestamp(event.delist_date)),
                "tushare_file_available": path.is_file(),
                "traded_on_effective_date": traded,
                "tushare_path": _relative(path) if path.is_file() else "",
            }
        )
    return pd.DataFrame(rows), inputs


def validate_restoration_trade_days(
    restoration_events: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    listings = master[master["event_type"].eq("listing")][["asset", "list_date"]].drop_duplicates("asset")
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].drop_duplicates("asset")
    lifecycle = listings.merge(exits, on="asset", how="left", validate="one_to_one")
    lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"], errors="coerce").dt.normalize()
    lifecycle["delist_date"] = pd.to_datetime(lifecycle["delist_date"], errors="coerce").dt.normalize()
    source = restoration_events[["asset", "effective_date", "factbook_edition", "restored_name"]].copy()
    source = source.merge(lifecycle, on="asset", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    date_cache: dict[pd.Timestamp, set[str]] = {}
    for event in source.itertuples(index=False):
        effective = pd.Timestamp(event.effective_date).normalize()
        path = TUSHARE_DAILY_DIR / f"trade_date={effective:%Y%m%d}.csv"
        traded: bool | pd.NA = pd.NA
        if effective >= BACKTEST_START and path.is_file():
            if effective not in date_cache:
                daily = pd.read_csv(path, usecols=["ts_code"], dtype={"ts_code": str})
                date_cache[effective] = set(daily["ts_code"].astype(str).str[:6])
                inputs.append(
                    {"role": "tushare_restoration_trade_date", "path": _relative(path), "sha256": _sha256(path)}
                )
            traded = str(event.asset) in date_cache[effective]
        within_lifecycle = bool(
            pd.notna(event.list_date)
            and effective >= pd.Timestamp(event.list_date)
            and (pd.isna(event.delist_date) or effective <= pd.Timestamp(event.delist_date))
        )
        rows.append(
            {
                "asset": str(event.asset),
                "effective_date": effective,
                "factbook_edition": event.factbook_edition,
                "restored_name": event.restored_name,
                "within_lifecycle": within_lifecycle,
                "tushare_file_available": path.is_file(),
                "traded_on_effective_date": traded,
                "tushare_path": _relative(path) if path.is_file() else "",
            }
        )
    return pd.DataFrame(rows), inputs


def _ratio(frame: pd.DataFrame, column: str) -> float:
    return round(float(frame[column].mean()), 8) if not frame.empty else 0.0


def _check(check_id: str, observed: float | int, operator: str, threshold: float | int, detail: str) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= threshold
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unsupported qualification operator: {operator}")
    return {
        "check_id": check_id,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
        "detail": detail,
    }


def run_validation(as_of: str, holdout_assets: int = DEFAULT_HOLDOUT_ASSETS, sleep_seconds: float = 0.15) -> dict[str, Any]:
    _ = (holdout_assets, sleep_seconds)  # Kept for CLI compatibility with earlier validation runs.
    as_of_date = pd.Timestamp(as_of).normalize()
    events, status_manifest = load_status_events(as_of_date)
    restoration_events, factbook_reference, factbook_manifest = load_factbook_validation_events(as_of_date)
    event_index = build_event_index(events)
    calendar_frame = pd.read_csv(CALENDAR_PATH)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_frame["date"], errors="coerce").dropna().sort_values().unique())
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    master["asset"] = master["asset"].astype(str).str.zfill(6)
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce").dt.normalize()

    baostock_daily, baostock_actual_transitions, baostock_inputs, baostock_errors = load_baostock_checks(event_index)
    baostock_metrics = _daily_metrics(baostock_daily, "actual_is_st") if not baostock_daily.empty else _daily_metrics(
        pd.DataFrame(columns=["actual_is_st", "expected_is_st", "expected_state_known", "expected_status"]),
        "actual_is_st",
    )
    baostock_metrics["assets"] = int(baostock_daily["asset"].nunique()) if not baostock_daily.empty else 0
    expected_transitions = expected_binary_transitions(events)
    baostock_transition_checks = compare_transition_dates(
        expected_transitions,
        baostock_actual_transitions,
        _source_ranges(baostock_daily) if not baostock_daily.empty else pd.DataFrame(),
        calendar,
    )
    baostock_metrics["transition_checks"] = int(len(baostock_transition_checks))
    baostock_metrics["transition_exact_ratio"] = _ratio(baostock_transition_checks, "exact_match")
    baostock_metrics["transition_within_one_session_ratio"] = _ratio(
        baostock_transition_checks, "within_one_session"
    )
    baostock_era_metrics = build_baostock_era_metrics(baostock_daily, baostock_transition_checks)

    if not baostock_daily.empty:
        valid = (
            baostock_daily["expected_state_known"]
            & baostock_daily["expected_status"].isin(BINARY_VALIDATION_STATUSES)
            & baostock_daily["actual_is_st"].notna()
        )
        baostock_daily["match"] = _as_nullable_bool(baostock_daily["actual_is_st"]).eq(
            _as_nullable_bool(baostock_daily["expected_is_st"])
        )
        baostock_mismatches = baostock_daily.loc[
            valid & ~baostock_daily["match"].fillna(False),
            ["asset", "date", "actual_is_st", "expected_is_st", "expected_status"],
        ].copy()
        coverage_mode = events.drop_duplicates("asset").set_index("asset")["source_coverage_mode"]
        baostock_daily["source_coverage_mode"] = baostock_daily["asset"].map(coverage_mode)
        baostock_summary = (
            baostock_daily.loc[valid]
            .groupby("source_coverage_mode", dropna=False)
            .agg(rows=("asset", "size"), assets=("asset", "nunique"), match_ratio=("match", "mean"))
            .reset_index()
        )
    else:
        baostock_mismatches = pd.DataFrame(
            columns=["asset", "date", "actual_is_st", "expected_is_st", "expected_status"]
        )
        baostock_summary = pd.DataFrame(columns=["source_coverage_mode", "rows", "assets", "match_ratio"])

    jq_daily, jq_actual_transitions, jq_inputs, jq_errors = load_joinquant_checks(event_index)
    jq_metrics = _daily_metrics(jq_daily, "actual_is_st") if not jq_daily.empty else _daily_metrics(
        pd.DataFrame(columns=["actual_is_st", "expected_is_st", "expected_state_known", "expected_status"]),
        "actual_is_st",
    )
    jq_metrics["assets"] = int(jq_daily["asset"].nunique()) if not jq_daily.empty else 0
    jq_transition_checks = compare_transition_dates(
        expected_transitions,
        jq_actual_transitions,
        _source_ranges(jq_daily) if not jq_daily.empty else pd.DataFrame(),
        calendar,
    )
    jq_metrics["transition_checks"] = int(len(jq_transition_checks))
    jq_metrics["transition_exact_ratio"] = _ratio(jq_transition_checks, "exact_match")
    jq_metrics["transition_within_one_session_ratio"] = _ratio(jq_transition_checks, "within_one_session")
    if not jq_daily.empty:
        jq_daily["match"] = _as_nullable_bool(jq_daily["actual_is_st"]).eq(_as_nullable_bool(jq_daily["expected_is_st"]))

    factbook_ingestion = build_factbook_ingestion_metrics(
        factbook_reference,
        status_manifest,
        factbook_manifest,
    )

    delisting_checks, delisting_inputs = validate_delisting_trade_days(events, master)
    delisting_scope = delisting_checks[delisting_checks["traded_on_effective_date"].notna()].copy()
    delisting_metrics = {
        "events": int(len(delisting_checks)),
        "trade_day_checks": int(len(delisting_scope)),
        "trade_day_match_ratio": _ratio(delisting_scope, "traded_on_effective_date"),
        "lifecycle_match_ratio": _ratio(delisting_checks, "within_lifecycle"),
    }

    restoration_checks, restoration_inputs = validate_restoration_trade_days(restoration_events, master)
    restoration_scope = restoration_checks[restoration_checks["traded_on_effective_date"].notna()].copy()
    restoration_metrics = {
        "events": int(len(restoration_checks)),
        "trade_day_checks": int(len(restoration_scope)),
        "trade_day_match_ratio": _ratio(restoration_scope, "traded_on_effective_date"),
        "lifecycle_match_ratio": _ratio(restoration_checks, "within_lifecycle"),
    }

    qualification = [
        _check("baostock_assets", baostock_metrics["assets"], ">=", THRESHOLDS["baostock_min_assets"], "independent cached assets"),
        _check("baostock_daily_checks", baostock_metrics["daily_checks"], ">=", THRESHOLDS["baostock_min_daily_checks"], "daily ST observations"),
        _check("baostock_actual_st_rows", baostock_metrics["actual_st_rows"], ">=", THRESHOLDS["baostock_min_actual_st_rows"], "positive-class observations"),
        _check("baostock_daily_match", baostock_metrics["daily_match_ratio"], ">=", THRESHOLDS["baostock_daily_match_min"], "all daily observations"),
        _check("baostock_st_precision", baostock_metrics["st_precision"], ">=", THRESHOLDS["baostock_st_precision_min"], "candidate ST precision"),
        _check("baostock_st_recall", baostock_metrics["st_recall"], ">=", THRESHOLDS["baostock_st_recall_min"], "candidate ST recall"),
        _check("baostock_transition_checks", baostock_metrics["transition_checks"], ">=", THRESHOLDS["baostock_min_transition_checks"], "binary status transitions in source range"),
        _check("baostock_transition_timing", baostock_metrics["transition_within_one_session_ratio"], ">=", THRESHOLDS["baostock_transition_within_one_session_min"], "event timing within one market session"),
        _check("joinquant_assets", jq_metrics["assets"], ">=", THRESHOLDS["joinquant_min_assets"], "independent recent-window assets"),
        _check("joinquant_daily_checks", jq_metrics["daily_checks"], ">=", THRESHOLDS["joinquant_min_daily_checks"], "recent daily ST observations"),
        _check("joinquant_actual_st_rows", jq_metrics["actual_st_rows"], ">=", THRESHOLDS["joinquant_min_actual_st_rows"], "recent positive-class observations"),
        _check("joinquant_daily_match", jq_metrics["daily_match_ratio"], ">=", THRESHOLDS["joinquant_daily_match_min"], "recent daily observations"),
        _check("joinquant_st_precision", jq_metrics["st_precision"], ">=", THRESHOLDS["joinquant_st_precision_min"], "recent candidate ST precision"),
        _check("joinquant_st_recall", jq_metrics["st_recall"], ">=", THRESHOLDS["joinquant_st_recall_min"], "recent candidate ST recall"),
        _check("joinquant_transition_checks", jq_metrics["transition_checks"], ">=", THRESHOLDS["joinquant_min_transition_checks"], "recent binary transitions"),
        _check("joinquant_transition_timing", jq_metrics["transition_within_one_session_ratio"], ">=", THRESHOLDS["joinquant_transition_within_one_session_min"], "recent event timing within one market session"),
        _check("factbook_reference_candidates", factbook_ingestion["eligible_reference_candidates"], ">=", THRESHOLDS["factbook_min_reference_candidates"], "official status-table candidates ingested"),
        _check("factbook_candidate_count_match", int(factbook_ingestion["candidate_count_match"]), "==", 1, "reconciler candidate count matches governed table"),
        _check("factbook_lineage_hash", int(factbook_ingestion["lineage_hash_match"]), "==", 1, "reconciler input hash matches official table output"),
        _check("factbook_future_availability", factbook_ingestion["future_available_rows"], "==", 0, "execution status is available no later than effective date"),
        _check("delisting_trade_day_checks", delisting_metrics["trade_day_checks"], ">=", THRESHOLDS["delisting_min_trade_day_checks"], "delisting events inside available Tushare history"),
        _check("delisting_trade_day_match", delisting_metrics["trade_day_match_ratio"], ">=", THRESHOLDS["delisting_trade_day_match_min"], "effective date has a real traded row"),
        _check("delisting_lifecycle", delisting_metrics["lifecycle_match_ratio"], "==", 1.0, "event does not exceed legal lifecycle"),
        _check("restoration_trade_day_checks", restoration_metrics["trade_day_checks"], ">=", THRESHOLDS["restoration_min_trade_day_checks"], "official restoration events inside available Tushare history"),
        _check("restoration_trade_day_match", restoration_metrics["trade_day_match_ratio"], ">=", THRESHOLDS["restoration_trade_day_match_min"], "restoration effective date has a real traded row"),
        _check("restoration_lifecycle", restoration_metrics["lifecycle_match_ratio"], "==", 1.0, "restoration event stays inside legal lifecycle"),
        _check("baostock_read_errors", len(baostock_errors), "==", 0, "cached BaoStock files parse cleanly"),
        _check("joinquant_read_errors", len(jq_errors), "==", 0, "cached JoinQuant files parse cleanly"),
    ]
    for era in baostock_era_metrics.itertuples(index=False):
        prefix = f"baostock_{era.era}"
        qualification.extend(
            [
                _check(f"{prefix}_assets", int(era.assets), ">=", THRESHOLDS["baostock_era_min_assets"], "independent assets in era"),
                _check(f"{prefix}_daily_checks", int(era.daily_checks), ">=", THRESHOLDS["baostock_era_min_daily_checks"], "daily ST observations in era"),
                _check(f"{prefix}_actual_st_rows", int(era.actual_st_rows), ">=", THRESHOLDS["baostock_era_min_actual_st_rows"], "positive-class rows in era"),
                _check(f"{prefix}_daily_match", float(era.daily_match_ratio), ">=", THRESHOLDS["baostock_era_daily_match_min"], "daily agreement in era"),
                _check(f"{prefix}_st_precision", float(era.st_precision), ">=", THRESHOLDS["baostock_era_st_precision_min"], "candidate ST precision in era"),
                _check(f"{prefix}_st_recall", float(era.st_recall), ">=", THRESHOLDS["baostock_era_st_recall_min"], "candidate ST recall in era"),
                _check(f"{prefix}_transition_checks", int(era.transition_checks), ">=", THRESHOLDS["baostock_era_min_transition_checks"], "binary transitions in era"),
                _check(f"{prefix}_transition_timing", float(era.transition_within_one_session_ratio), ">=", THRESHOLDS["baostock_era_transition_within_one_session_min"], "transition timing in era"),
            ]
        )
    qualification_frame = pd.DataFrame(qualification)
    passed = bool(qualification_frame["passed"].all())

    _atomic_csv(baostock_summary, BAOSTOCK_SUMMARY_OUTPUT)
    _atomic_csv(baostock_mismatches, BAOSTOCK_MISMATCHES_OUTPUT)
    _atomic_csv(baostock_transition_checks, BAOSTOCK_TRANSITIONS_OUTPUT)
    _atomic_csv(baostock_era_metrics, BAOSTOCK_ERAS_OUTPUT)
    _atomic_csv(jq_daily, JQ_CHECKS_OUTPUT)
    _atomic_csv(jq_transition_checks, JQ_TRANSITIONS_OUTPUT)
    _atomic_csv(pd.DataFrame([factbook_ingestion]), FACTBOOK_INGESTION_OUTPUT)
    _atomic_csv(delisting_checks, DELISTING_CHECKS_OUTPUT)
    _atomic_csv(restoration_checks, RESTORATION_CHECKS_OUTPUT)
    _atomic_csv(qualification_frame, QUALIFICATION_CHECKS_OUTPUT)

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "qualification_status": "PASS" if passed else "FAIL",
        "historical_backtest_allowed": passed,
        "model_promotion_allowed": False,
        "status_event_source_qualification_before_validation": status_manifest.get("qualification_status"),
        "thresholds": THRESHOLDS,
        "baostock": baostock_metrics,
        "baostock_eras": baostock_era_metrics.set_index("era").to_dict("index"),
        "joinquant": jq_metrics,
        "factbook_ingestion": factbook_ingestion,
        "delisting": delisting_metrics,
        "restoration": restoration_metrics,
        "failed_checks": qualification_frame.loc[~qualification_frame["passed"], "check_id"].tolist(),
        "source_errors": [*baostock_errors, *jq_errors],
        "evidence_boundary": {
            "baostock": "230-asset cached historical panel; not full-market coverage",
            "baostock_eras": "independent long-history evidence is gated separately for three non-overlapping eras",
            "joinquant": "trial-window independent daily panel only",
            "factbook_ingestion": "official factbook rows are model inputs; only lineage and availability are qualified, never circular alignment",
            "restoration": "factbook restoration rows used by reconciliation are checked against actual Tushare trade dates",
            "promotion": "PASS validates execution-status events only; it does not qualify prices, valuation, fundamentals, or the model",
        },
    }
    _atomic_json(report, REPORT_PATH)

    output_paths = [
        BAOSTOCK_SUMMARY_OUTPUT,
        BAOSTOCK_MISMATCHES_OUTPUT,
        BAOSTOCK_TRANSITIONS_OUTPUT,
        BAOSTOCK_ERAS_OUTPUT,
        JQ_CHECKS_OUTPUT,
        JQ_TRANSITIONS_OUTPUT,
        FACTBOOK_INGESTION_OUTPUT,
        DELISTING_CHECKS_OUTPUT,
        RESTORATION_CHECKS_OUTPUT,
        QUALIFICATION_CHECKS_OUTPUT,
        REPORT_PATH,
    ]
    fixed_inputs = [
        {"role": "status_events", "path": _relative(STATUS_EVENTS_PATH), "sha256": _sha256(STATUS_EVENTS_PATH)},
        {"role": "status_event_manifest", "path": _relative(STATUS_MANIFEST_PATH), "sha256": _sha256(STATUS_MANIFEST_PATH)},
        {"role": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"role": "trade_calendar", "path": _relative(CALENDAR_PATH), "sha256": _sha256(CALENDAR_PATH)},
        {"role": "baostock_asset_summary", "path": _relative(BAOSTOCK_SUMMARY_PATH), "sha256": _sha256(BAOSTOCK_SUMMARY_PATH)},
        {"role": "sse_factbook_manifest", "path": _relative(FACTBOOK_MANIFEST_PATH), "sha256": _sha256(FACTBOOK_MANIFEST_PATH)},
        {"role": "sse_factbook_reference_events", "path": _relative(FACTBOOK_REFERENCE_PATH), "sha256": _sha256(FACTBOOK_REFERENCE_PATH)},
        {"role": "sse_factbook_restoration_events", "path": _relative(FACTBOOK_RESTORATION_PATH), "sha256": _sha256(FACTBOOK_RESTORATION_PATH)},
    ]
    manifest = {
        "created_at": report["created_at"],
        "as_of_date": report["as_of_date"],
        "qualification_status": report["qualification_status"],
        "historical_backtest_allowed": passed,
        "model_promotion_allowed": False,
        "validation_design_version": "status_event_cross_source_v2",
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "inputs": [*fixed_inputs, *baostock_inputs, *jq_inputs, *delisting_inputs, *restoration_inputs],
        "outputs": [
            {"path": _relative(path), "sha256": _sha256(path)}
            for path in output_paths
        ],
        "failed_checks": report["failed_checks"],
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--holdout-assets", type=int, default=DEFAULT_HOLDOUT_ASSETS)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_validation(args.as_of, args.holdout_assets, args.sleep_seconds)
    print(
        json.dumps(
            {
                "qualification_status": report["qualification_status"],
                "failed_checks": report["failed_checks"],
                "baostock": report["baostock"],
                "baostock_eras": report["baostock_eras"],
                "joinquant": report["joinquant"],
                "factbook_ingestion": report["factbook_ingestion"],
                "delisting": report["delisting"],
                "restoration": report["restoration"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
