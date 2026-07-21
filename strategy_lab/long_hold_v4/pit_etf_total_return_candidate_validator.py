"""Validate the official-event ETF total-return candidate without promoting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pit_source_code_archive import authenticate_current_or_archive


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_official_event_candidate_latest.json"
)
SOURCE_AUDIT_MANIFEST_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_price_nav" / "run_manifest.json"
)
SOURCE_QUALIFICATION_MANIFEST_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_source_qualification" / "run_manifest.json"
)
LIFECYCLE_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
)
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_total_return_candidate"
EVENT_ALIGNMENT_PATH = OUTPUT_DIR / "event_alignment_checks.csv"
RETURN_IDENTITY_PATH = OUTPUT_DIR / "total_return_identity_by_asset.csv"
INDEPENDENT_PRICE_PATH = OUTPUT_DIR / "independent_raw_price_summary.csv"
LEGACY_DIFFERENCE_PATH = OUTPUT_DIR / "legacy_difference_summary.json"
CHECKS_PATH = OUTPUT_DIR / "qualification_checks.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "ETF_TOTAL_RETURN_CANDIDATE_VALIDATION.md"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
QUALIFICATION_STATUS = "PASS_DETERMINISTIC_CANDIDATE_FULL_HISTORY_SOURCE_BLOCKED"


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


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def _authenticate_item(item: dict[str, Any], role: str) -> Path:
    path = _resolve(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError(f"ETF total-return candidate validation hash mismatch: {role}")
    return path


def _authenticate_code_item(item: dict[str, Any], role: str) -> Path:
    path = _resolve(item.get("path", ""))
    try:
        return authenticate_current_or_archive(path, str(item.get("sha256", "")))
    except ValueError as exc:
        raise ValueError(f"ETF total-return candidate validation code hash mismatch: {role}") from exc


def _authenticate_candidate() -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = json.loads(CANDIDATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "CANDIDATE_OFFICIAL_EVENTS_CURRENT_FINAL_PRICE"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
        or manifest.get("source_quality_gate_passed") is not False
        or manifest.get("contains_synthetic_market_rows") is not False
    ):
        raise ValueError("ETF total-return candidate violates its non-promotable contract")
    code_files = manifest.get("code_files")
    if not isinstance(code_files, list) or not code_files:
        raise ValueError("ETF total-return candidate code bundle is absent")
    for item in code_files:
        _authenticate_code_item(item, "candidate_code")
    for item in manifest.get("inputs", []):
        _authenticate_item(item, str(item.get("role", "candidate_input")))
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    required = ("etf_total_return_price_candidate", "official_event_usage", "candidate_asset_status")
    if any(role not in outputs for role in required):
        raise ValueError("ETF total-return candidate manifest is missing required outputs")
    return manifest, {role: _authenticate_item(outputs[role], role) for role in required}


def _authenticate_source_audit() -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = json.loads(SOURCE_AUDIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("recent_cross_source_passed") is not True
        or manifest.get("formal_table_promotion_allowed") is not False
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF price/NAV source audit has an unexpected qualification boundary")
    code_path = _resolve(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError:
        raise ValueError("ETF price/NAV source-audit code hash mismatch")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    required = ("price_checks", "summary", "qualification_checks")
    if any(role not in outputs for role in required):
        raise ValueError("ETF price/NAV source audit is missing required outputs")
    return manifest, {role: _authenticate_item(outputs[role], f"source_audit:{role}") for role in required}


def _authenticate_source_qualification() -> dict[str, Any]:
    manifest = json.loads(SOURCE_QUALIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
        or manifest.get("formal_table_promotion_allowed") is not False
        or manifest.get("historical_available_date_evidence_passed") is not False
    ):
        raise ValueError("ETF source qualification violates its current-final evidence boundary")
    code_path = _resolve(manifest.get("code_path", ""))
    try:
        authenticate_current_or_archive(code_path, str(manifest.get("code_sha256", "")))
    except ValueError:
        raise ValueError("ETF source-qualification code hash mismatch")
    for item in manifest.get("inputs", []):
        _authenticate_item(item, f"source_qualification_input:{item.get('role', '')}")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    required = {"qualification_checks", "summary", "report"}
    if not required.issubset(outputs):
        raise ValueError("ETF source qualification is missing required outputs")
    for role in required:
        _authenticate_item(outputs[role], f"source_qualification:{role}")
    return manifest


def _authenticate_lifecycle() -> tuple[dict[str, Any], Path]:
    manifest = json.loads(LIFECYCLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    immutable_value = str(manifest.get("immutable_manifest_path", ""))
    immutable_path = _resolve(immutable_value)
    if not immutable_value or not immutable_path.is_file():
        raise ValueError("ETF lifecycle immutable manifest is absent")
    if _sha256(immutable_path) != _sha256(LIFECYCLE_MANIFEST_PATH):
        raise ValueError("ETF lifecycle latest and immutable manifests differ")
    code_files = manifest.get("code_files")
    if not isinstance(code_files, list) or not code_files:
        raise ValueError("ETF lifecycle code bundle is absent")
    for item in code_files:
        _authenticate_code_item(item, "lifecycle_code")
    if manifest.get("historical_backtest_allowed") is not False:
        raise ValueError("ETF lifecycle observation violates its non-promotable contract")
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    legacy = outputs.get("etf_total_return_prices_observation")
    if legacy is None:
        raise ValueError("ETF lifecycle observation omits the total-return output")
    return manifest, _authenticate_item(legacy, "legacy_total_return_observation")


def align_cash_events(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    market = prices[["date", "asset", "source_cash_distribution"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["asset"] = market["asset"].astype(str).str.zfill(6)
    market["source_cash_distribution"] = pd.to_numeric(
        market["source_cash_distribution"], errors="coerce"
    )
    effective = events[
        events["event_type"].eq("cash_distribution")
        & events["applied_to_price_adjustment"].astype(str).str.lower().eq("true")
    ].copy()
    effective["ex_date"] = pd.to_datetime(effective["ex_date"], errors="coerce").dt.normalize()
    effective["cash_per_share"] = pd.to_numeric(effective["cash_per_share"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for asset, grouped in effective.groupby("asset"):
        asset = str(asset).zfill(6)
        dates = market.loc[market["asset"].eq(asset), "date"].sort_values().reset_index(drop=True)
        if dates.empty:
            raise ValueError(f"ETF candidate has no market dates for official cash events: {asset}")
        date_values = dates.to_numpy(dtype="datetime64[ns]")
        for event in grouped.itertuples(index=False):
            ex_date = pd.Timestamp(event.ex_date).normalize()
            position = int(np.searchsorted(date_values, np.datetime64(ex_date), side="left"))
            aligned = dates.iloc[position] if position < len(dates) else pd.NaT
            rows.append(
                {
                    "asset": asset,
                    "ex_date": ex_date,
                    "aligned_price_date": aligned,
                    "calendar_lag_days": int((aligned - ex_date).days) if pd.notna(aligned) else None,
                    "cash_per_share": float(event.cash_per_share),
                }
            )
    alignment = pd.DataFrame(rows)
    expected = alignment.groupby(["asset", "aligned_price_date"], as_index=False).agg(
        official_event_count=("ex_date", "size"),
        expected_source_cash_distribution=("cash_per_share", "sum"),
        maximum_calendar_lag_days=("calendar_lag_days", "max"),
    )
    actual = market[market["source_cash_distribution"].gt(0)][
        ["asset", "date", "source_cash_distribution"]
    ].rename(columns={"date": "aligned_price_date", "source_cash_distribution": "observed_source_cash_distribution"})
    checks = expected.merge(actual, on=["asset", "aligned_price_date"], how="outer", validate="one_to_one")
    checks["amount_absolute_error"] = (
        checks["expected_source_cash_distribution"] - checks["observed_source_cash_distribution"]
    ).abs()
    checks["amount_match"] = checks["amount_absolute_error"].le(1e-10)
    checks["alignment_lag_allowed"] = checks["maximum_calendar_lag_days"].between(0, 31)
    checks["passed"] = checks["amount_match"] & checks["alignment_lag_allowed"]
    return checks.sort_values(["asset", "aligned_price_date"]).reset_index(drop=True)


def total_return_identity(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices[
        ["date", "asset", "close", "raw_close", "cash_distribution", "share_adjustment_factor"]
    ].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["asset"] = data["asset"].astype(str).str.zfill(6)
    numeric = ["close", "raw_close", "cash_distribution", "share_adjustment_factor"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)
    data["share_adjusted_raw_close"] = data["raw_close"] * data["share_adjustment_factor"]
    previous_internal = data.groupby("asset")["share_adjusted_raw_close"].shift(1)
    previous_candidate = data.groupby("asset")["close"].shift(1)
    expected = (data["share_adjusted_raw_close"] + data["cash_distribution"]) / previous_internal
    observed = data["close"] / previous_candidate
    data["identity_absolute_error"] = (expected - observed).abs()
    comparable = previous_internal.notna()
    data["identity_passed"] = ~comparable | data["identity_absolute_error"].le(1e-10)
    return data.groupby("asset", as_index=False).agg(
        price_rows=("date", "size"),
        comparable_return_rows=("identity_absolute_error", "count"),
        maximum_identity_absolute_error=("identity_absolute_error", "max"),
        failed_identity_rows=("identity_passed", lambda values: int((~values).sum())),
    )


def validate_terminal_cash_ledger(events: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    terminal = events[events["event_type"].eq("liquidation_distribution")].copy()
    required = {
        "event_id",
        "asset",
        "distribution_sequence",
        "holder_scope",
        "available_trade_date",
        "accounting_date",
        "is_final_distribution",
        "additional_distribution_expected",
        "extinguishes_position",
        "event_effective_by_cutoff",
        "applied_to_price_adjustment",
        "applied_to_cash_ledger",
        "source_pdf_sha256_set",
    }
    missing = sorted(required.difference(terminal.columns))
    if missing:
        return {
            "passed": False,
            "rows": int(len(terminal)),
            "assets": int(terminal["asset"].nunique()) if "asset" in terminal else 0,
            "extinguishing_rows": 0,
            "failed_groups": [f"missing_columns:{','.join(missing)}"],
        }

    terminal["available_trade_date"] = pd.to_datetime(
        terminal["available_trade_date"], errors="coerce"
    ).dt.normalize()
    terminal["accounting_date"] = pd.to_datetime(terminal["accounting_date"], errors="coerce").dt.normalize()
    terminal["distribution_sequence"] = pd.to_numeric(
        terminal["distribution_sequence"], errors="coerce"
    )
    boolean_columns = (
        "is_final_distribution",
        "additional_distribution_expected",
        "extinguishes_position",
        "event_effective_by_cutoff",
        "applied_to_price_adjustment",
        "applied_to_cash_ledger",
    )
    booleans = {
        column: terminal[column].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        for column in boolean_columns
    }
    failed_groups: list[str] = []
    if any(values.isna().any() for values in booleans.values()):
        failed_groups.append("invalid_boolean")
    chronology = (
        terminal["available_trade_date"].notna()
        & terminal["accounting_date"].notna()
        & terminal["available_trade_date"].le(terminal["accounting_date"])
        & terminal["accounting_date"].le(cutoff)
    )
    if not chronology.all():
        failed_groups.append("invalid_pit_chronology")
    if terminal["event_id"].astype(str).eq("").any() or terminal["event_id"].astype(str).duplicated().any():
        failed_groups.append("duplicate_or_blank_event_id")
    if terminal["source_pdf_sha256_set"].astype(str).str.strip().isin({"", "nan"}).any():
        failed_groups.append("missing_official_pdf_hash")
    if not booleans["event_effective_by_cutoff"].fillna(False).all():
        failed_groups.append("ineffective_terminal_cash")
    if booleans["applied_to_price_adjustment"].fillna(True).any():
        failed_groups.append("terminal_cash_applied_to_price")
    if not booleans["applied_to_cash_ledger"].fillna(False).all():
        failed_groups.append("terminal_cash_missing_from_ledger")

    for (asset, scope), group in terminal.groupby(["asset", "holder_scope"], dropna=False):
        ordered = group.sort_values(["distribution_sequence", "accounting_date", "event_id"], kind="mergesort")
        sequence = ordered["distribution_sequence"]
        expected = list(range(1, len(ordered) + 1))
        if sequence.isna().any() or not np.equal(sequence, np.floor(sequence)).all() or sequence.astype(int).tolist() != expected:
            failed_groups.append(f"sequence:{asset}:{scope}")
            continue
        extinguishing = booleans["extinguishes_position"].loc[ordered.index].fillna(False)
        final_flags = booleans["is_final_distribution"].loc[ordered.index].fillna(False)
        more_expected = booleans["additional_distribution_expected"].loc[ordered.index].fillna(True)
        if int(extinguishing.sum()) > 1 or (extinguishing.any() and not bool(extinguishing.iloc[-1])):
            failed_groups.append(f"extinguishment_position:{asset}:{scope}")
        if (extinguishing & ~final_flags).any() or (extinguishing & more_expected).any():
            failed_groups.append(f"extinguishment_semantics:{asset}:{scope}")

    return {
        "passed": not failed_groups,
        "rows": int(len(terminal)),
        "assets": int(terminal["asset"].nunique()),
        "extinguishing_rows": int(booleans["extinguishes_position"].fillna(False).sum()),
        "failed_groups": sorted(set(failed_groups)),
    }


def validate(as_of: str | pd.Timestamp) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    candidate_manifest, candidate_paths = _authenticate_candidate()
    source_audit, source_paths = _authenticate_source_audit()
    source_qualification = _authenticate_source_qualification()
    lifecycle, legacy_path = _authenticate_lifecycle()
    price_columns = [
        "date",
        "asset",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "raw_close",
        "source_cash_distribution",
        "cash_distribution",
        "share_adjustment_factor",
        "adjustment_factor",
        "return_basis",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    ]
    prices = pd.read_csv(
        candidate_paths["etf_total_return_price_candidate"],
        usecols=price_columns,
        dtype={"asset": str},
        low_memory=False,
    )
    events = pd.read_csv(candidate_paths["official_event_usage"], dtype={"asset": str}, low_memory=False)
    statuses = pd.read_csv(candidate_paths["candidate_asset_status"], dtype={"asset": str}, low_memory=False)
    prices["asset"] = prices["asset"].astype(str).str.zfill(6)
    events["asset"] = events["asset"].astype(str).str.zfill(6)
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "raw_close",
        "source_cash_distribution",
        "cash_distribution",
        "share_adjustment_factor",
        "adjustment_factor",
    ]
    numeric_values = prices[numeric].to_numpy(dtype=float)
    duplicate_price_keys = int(prices.duplicated(["date", "asset"]).sum())
    nonfinite_values = int((~np.isfinite(numeric_values)).sum())
    nonpositive_required = int(
        prices[["open", "high", "low", "close", "raw_close", "share_adjustment_factor", "adjustment_factor"]]
        .le(0)
        .sum()
        .sum()
    )
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    ordered = prices.sort_values(["asset", "date"])
    maximum_absolute_return = float(ordered.groupby("asset")["close"].pct_change().abs().max())

    event_alignment = align_cash_events(prices, events)
    return_identity = total_return_identity(prices)
    terminal_validation = validate_terminal_cash_ledger(events, cutoff)

    independent_checks = pd.read_csv(source_paths["price_checks"], dtype={"asset": str}, low_memory=False)
    independent_checks["asset"] = independent_checks["asset"].astype(str).str.zfill(6)
    independent_checks["date"] = pd.to_datetime(independent_checks["date"], errors="coerce").dt.normalize()
    independent = independent_checks[["date", "asset", "close_joinquant"]].merge(
        prices[["date", "asset", "raw_close"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    independent["absolute_error"] = (independent["raw_close"] - independent["close_joinquant"]).abs()
    independent["match"] = independent["raw_close"].notna() & independent["absolute_error"].le(0.0011)
    independent_summary = independent.groupby("asset", as_index=False).agg(
        compared_rows=("date", "size"),
        matched_rows=("match", "sum"),
        maximum_absolute_error=("absolute_error", "max"),
    )

    legacy = pd.read_csv(
        legacy_path,
        usecols=["date", "asset", "close", "raw_close", "source_cash_distribution"],
        dtype={"asset": str},
        low_memory=False,
    )
    legacy["asset"] = legacy["asset"].astype(str).str.zfill(6)
    legacy["date"] = pd.to_datetime(legacy["date"], errors="coerce").dt.normalize()
    legacy_comparison = prices[["date", "asset", "close", "raw_close", "source_cash_distribution"]].merge(
        legacy,
        on=["date", "asset"],
        suffixes=("_candidate", "_legacy"),
        validate="one_to_one",
    )
    raw_difference = (legacy_comparison["raw_close_candidate"] - legacy_comparison["raw_close_legacy"]).abs()
    adjusted_difference = (legacy_comparison["close_candidate"] - legacy_comparison["close_legacy"]).abs()
    cash_difference = (
        legacy_comparison["source_cash_distribution_candidate"]
        - legacy_comparison["source_cash_distribution_legacy"]
    ).abs()
    adjusted_difference_assets = sorted(
        legacy_comparison.loc[adjusted_difference.gt(1e-12), "asset"].unique().tolist()
    )
    cash_difference_assets = sorted(legacy_comparison.loc[cash_difference.gt(1e-12), "asset"].unique().tolist())
    legacy_difference = {
        "candidate_rows": int(len(prices)),
        "legacy_rows": int(len(legacy)),
        "merged_rows": int(len(legacy_comparison)),
        "maximum_raw_close_absolute_error": float(raw_difference.max()),
        "adjusted_close_difference_rows": int(adjusted_difference.gt(1e-12).sum()),
        "adjusted_close_difference_assets": adjusted_difference_assets,
        "maximum_adjusted_close_absolute_error": float(adjusted_difference.max()),
        "source_cash_difference_rows": int(cash_difference.gt(1e-12).sum()),
        "source_cash_difference_assets": cash_difference_assets,
        "maximum_source_cash_absolute_error": float(cash_difference.max()),
        "expected_official_precision_difference": bool(
            raw_difference.max() <= 1e-12
            and adjusted_difference_assets == cash_difference_assets
        ),
    }
    market_key_population_match = bool(
        len(prices) == len(legacy) == len(legacy_comparison)
        and raw_difference.max() <= 1e-12
    )

    event_counts = events["event_type"].value_counts().to_dict()
    effective_events = int(events["event_effective_by_cutoff"].astype(str).str.lower().eq("true").sum())
    price_applied_events = int(events["applied_to_price_adjustment"].astype(str).str.lower().eq("true").sum())
    cash_applied_events = int(events["applied_to_cash_ledger"].astype(str).str.lower().eq("true").sum())
    expected_assets = int(candidate_manifest["selected_assets"])
    expected_price_rows = int(candidate_manifest["price_rows"])
    expected_ordinary_events = int(candidate_manifest["official_dividend_events_available"])
    expected_effective_ordinary = int(candidate_manifest["official_dividend_events_effective"])
    expected_terminal_events = int(candidate_manifest["terminal_cash_event_rows"])
    expected_event_rows = int(candidate_manifest["event_usage_rows"])
    terminal_no_synthetic_row = bool(
        terminal_validation["passed"]
        and terminal_validation["rows"] == expected_terminal_events
        and market_key_population_match
    )
    deterministic_checks = [
        ("candidate_population", len(prices) == expected_price_rows and prices["asset"].nunique() == expected_assets),
        ("candidate_asset_status", len(statuses) == expected_assets and statuses["build_status"].eq("ready_candidate").all()),
        ("candidate_unique_keys", duplicate_price_keys == 0),
        ("candidate_finite_numeric", nonfinite_values == 0 and nonpositive_required == 0),
        ("candidate_return_jump", maximum_absolute_return <= 0.21),
        ("candidate_disabled_flags", not prices["historical_backtest_allowed"].astype(str).str.lower().eq("true").any() and not prices["model_promotion_allowed"].astype(str).str.lower().eq("true").any()),
        ("official_event_population", len(events) == expected_event_rows and event_counts.get("cash_distribution", 0) == expected_ordinary_events and event_counts.get("liquidation_distribution", 0) == expected_terminal_events),
        ("official_event_application", effective_events == expected_effective_ordinary + expected_terminal_events and price_applied_events == expected_effective_ordinary and cash_applied_events == expected_effective_ordinary + expected_terminal_events),
        ("official_event_alignment", int(event_alignment["official_event_count"].sum()) == expected_effective_ordinary and event_alignment["passed"].all()),
        ("total_return_identity", int(return_identity["failed_identity_rows"].sum()) == 0),
        ("terminal_cash_ledger", terminal_no_synthetic_row),
        ("independent_raw_price", independent["match"].all() and len(independent) == int(source_audit["price_matched_rows"])),
        ("legacy_precision_difference", legacy_difference["expected_official_precision_difference"]),
    ]
    deterministic_passed = bool(all(passed for _, passed in deterministic_checks))
    source_full_history_qualified = bool(source_qualification["formal_table_promotion_allowed"])
    formal_promotion_allowed = bool(deterministic_passed and source_full_history_qualified)
    checks = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "gate": "deterministic_candidate"}
            for name, passed in deterministic_checks
        ]
        + [
            {
                "check": "full_history_source_qualified",
                "passed": source_full_history_qualified,
                "gate": "formal_promotion",
            }
        ]
    )
    summary = {
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": QUALIFICATION_STATUS if deterministic_passed else "FAIL_DETERMINISTIC_CANDIDATE",
        "deterministic_candidate_passed": deterministic_passed,
        "recent_independent_source_passed": bool(source_audit["recent_cross_source_passed"]),
        "current_final_source_content_passed": bool(source_qualification["current_final_content_passed"]),
        "full_market_independent_price_pending": bool(source_qualification["full_market_price_pending"]),
        "full_history_source_qualified": source_full_history_qualified,
        "formal_table_promotion_allowed": formal_promotion_allowed,
        "price_rows": int(len(prices)),
        "price_assets": int(prices["asset"].nunique()),
        "duplicate_price_keys": duplicate_price_keys,
        "nonfinite_numeric_values": nonfinite_values,
        "nonpositive_required_values": nonpositive_required,
        "maximum_absolute_total_return": maximum_absolute_return,
        "event_rows": int(len(events)),
        "effective_event_rows": effective_events,
        "price_applied_event_rows": price_applied_events,
        "cash_ledger_event_rows": cash_applied_events,
        "event_alignment_rows": int(len(event_alignment)),
        "event_alignment_failures": int((~event_alignment["passed"]).sum()),
        "maximum_event_alignment_lag_days": int(event_alignment["maximum_calendar_lag_days"].max()),
        "maximum_event_cash_absolute_error": float(event_alignment["amount_absolute_error"].max()),
        "return_identity_failed_rows": int(return_identity["failed_identity_rows"].sum()),
        "maximum_return_identity_absolute_error": float(return_identity["maximum_identity_absolute_error"].max()),
        "terminal_cash_no_synthetic_price_row": terminal_no_synthetic_row,
        "terminal_cash_event_rows": int(terminal_validation["rows"]),
        "terminal_cash_event_assets": int(terminal_validation["assets"]),
        "terminal_cash_extinguishing_rows": int(terminal_validation["extinguishing_rows"]),
        "terminal_cash_failed_groups": terminal_validation["failed_groups"],
        "market_key_population_match": market_key_population_match,
        "independent_raw_price_rows": int(len(independent)),
        "independent_raw_price_match_ratio": float(independent["match"].mean()),
        "independent_raw_price_maximum_absolute_error": float(independent["absolute_error"].max()),
        "legacy_difference": legacy_difference,
        "source_version_depth_coverage": float(source_audit["source_version_depth_coverage"]),
        "independent_price_version_depth_coverage": float(source_qualification["price_version_depth_coverage"]),
        "independent_nav_version_depth_coverage": float(source_qualification["nav_version_depth_coverage"]),
        "current_final_independent_price_rows": int(source_qualification["tencent_price_rows"]),
        "current_final_independent_price_assets": int(source_qualification["tencent_ready_assets"]),
        "current_final_independent_price_coverage_start": source_qualification.get("tencent_price_coverage_start"),
        "current_final_independent_price_coverage_end": source_qualification.get("tencent_price_coverage_end"),
        "current_final_material_price_mismatch_rows": int(
            source_qualification.get("tencent_material_close_mismatch_rows", 0)
        ),
        "current_final_low_coverage_price_assets": int(
            source_qualification.get("tencent_assets_below_95pct_row_coverage", 0)
        ),
        "recent_joinquant_coverage_start": str(source_audit["independent_coverage_start"]),
        "recent_delisted_assets_with_rows": int(source_audit["recent_delisted_assets_with_rows"]),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "candidate reconstruction passed; full-history source qualification remains blocked",
    }
    _atomic_csv(event_alignment, EVENT_ALIGNMENT_PATH)
    _atomic_csv(return_identity, RETURN_IDENTITY_PATH)
    _atomic_csv(independent_summary, INDEPENDENT_PRICE_PATH)
    _atomic_json(legacy_difference, LEGACY_DIFFERENCE_PATH)
    _atomic_csv(checks, CHECKS_PATH)
    _atomic_json(summary, SUMMARY_PATH)
    report = "\n".join(
        [
            "# ETF Total-Return Candidate Validation",
            "",
            f"As of: {summary['as_of_date']}",
            f"Qualification: `{summary['qualification_status']}`",
            f"Formal promotion allowed: `{str(formal_promotion_allowed).lower()}`",
            "",
            "## Deterministic reconstruction",
            "",
            f"- Price assets/rows: {summary['price_assets']}/{summary['price_rows']}",
            f"- Official events: {summary['event_rows']} total, {price_applied_events} price-applied, {cash_applied_events} cash-ledger applied",
            f"- Event alignment failures: {summary['event_alignment_failures']}",
            f"- Return identity failures: {summary['return_identity_failed_rows']}",
            f"- Independent raw-price match: {summary['independent_raw_price_match_ratio']:.8%}",
            f"- Terminal cash events/assets: {summary['terminal_cash_event_rows']}/{summary['terminal_cash_event_assets']}",
            f"- Terminal cash chain failures: {summary['terminal_cash_failed_groups']}",
            f"- Terminal event chain creates synthetic market rows: `{str(not terminal_no_synthetic_row).lower()}`",
            "",
            "## Expected correction",
            "",
            f"- Adjusted-price differences versus the provider-event observation affect {legacy_difference['adjusted_close_difference_rows']} rows.",
            f"- Difference assets: {legacy_difference['adjusted_close_difference_assets']}.",
            f"- Maximum source-cash correction: {legacy_difference['maximum_source_cash_absolute_error']:.8f}.",
            "",
            "## Promotion blockers",
            "",
            f"- Full-market current-final independent prices cover {summary['current_final_independent_price_assets']:,} assets and {summary['current_final_independent_price_rows']:,} rows from {summary['current_final_independent_price_coverage_start']} to {summary['current_final_independent_price_coverage_end']}.",
            f"- The full-market audit discloses {summary['current_final_material_price_mismatch_rows']:,} material price disagreements and {summary['current_final_low_coverage_price_assets']:,} asset below 95% row coverage; no source row is rewritten.",
            f"- The separate JoinQuant precision window begins {summary['recent_joinquant_coverage_start']} and contains {summary['recent_delisted_assets_with_rows']}/123 delisted ETFs.",
            f"- Full current-final price/NAV content pass: `{str(summary['current_final_source_content_passed']).lower()}`.",
            f"- Independent price/NAV monitoring depth is {summary['independent_price_version_depth_coverage']:.2%}/{summary['independent_nav_version_depth_coverage']:.2%}.",
            "- Current-final agreement cannot establish historical available_date evidence.",
            "- The candidate remains observation-only and is not the formal etf_total_return_prices table.",
            "",
        ]
    )
    _atomic_text(report, REPORT_PATH)
    output_paths = [
        ("event_alignment", EVENT_ALIGNMENT_PATH),
        ("return_identity", RETURN_IDENTITY_PATH),
        ("independent_price_summary", INDEPENDENT_PRICE_PATH),
        ("legacy_difference", LEGACY_DIFFERENCE_PATH),
        ("qualification_checks", CHECKS_PATH),
        ("summary", SUMMARY_PATH),
        ("report", REPORT_PATH),
    ]
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **summary,
        "inputs": [
            {"role": "candidate_manifest", "path": _relative(CANDIDATE_MANIFEST_PATH), "sha256": _sha256(CANDIDATE_MANIFEST_PATH)},
            {"role": "source_audit_manifest", "path": _relative(SOURCE_AUDIT_MANIFEST_PATH), "sha256": _sha256(SOURCE_AUDIT_MANIFEST_PATH)},
            {"role": "source_qualification_manifest", "path": _relative(SOURCE_QUALIFICATION_MANIFEST_PATH), "sha256": _sha256(SOURCE_QUALIFICATION_MANIFEST_PATH)},
            {"role": "lifecycle_manifest", "path": _relative(LIFECYCLE_MANIFEST_PATH), "sha256": _sha256(LIFECYCLE_MANIFEST_PATH)},
        ],
        "outputs": [
            {"role": role, "path": _relative(path), "sha256": _sha256(path)} for role, path in output_paths
        ],
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.as_of)
    keys = (
        "qualification_status",
        "deterministic_candidate_passed",
        "recent_independent_source_passed",
        "full_history_source_qualified",
        "formal_table_promotion_allowed",
        "price_assets",
        "price_rows",
        "event_alignment_failures",
        "return_identity_failed_rows",
        "independent_raw_price_match_ratio",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
