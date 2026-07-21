"""Live higher-quality MARKET label-source acquisition attempt for V3.74.

This module attempts to acquire a compliant MARKET label source, but keeps a
hard separation between candidate provider data and the official source file.
It never writes ``data_raw/market_labels/market_total_return_index.csv`` unless
all governance gates explicitly allow that action.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]


@dataclass(frozen=True)
class LiveLabelSourceAcquisitionConfig:
    v3_73_manifest_path: Path
    v3_73_queue_path: Path
    v3_73_source_contract_path: Path
    signal_panel_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    target_security: str
    target_asset_or_index: str
    start_date: str
    end_date: str
    fields: tuple[str, ...]
    fq: str
    execute_probe: bool
    allow_target_write: bool
    approved_source_basis: str
    min_source_rows: int
    min_signal_coverage_ratio: float
    horizons: tuple[int, ...]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_credentials(root: Path) -> dict[str, Any]:
    path = root / "configs" / "data_credentials.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _truthy_secret(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    upper = text.upper()
    if "PASTE_" in upper or upper in {"NONE", "NULL", "TODO"}:
        return False
    return True


def credential_pair(root: Path) -> tuple[str | None, str | None]:
    credentials = read_credentials(root)
    jq_cfg = credentials.get("joinquant", {}) if isinstance(credentials, dict) else {}
    username = os.getenv("JQDATA_USERNAME") or os.getenv("JOINQUANT_USERNAME") or os.getenv("JQ_USER") or jq_cfg.get("username")
    password = os.getenv("JQDATA_PASSWORD") or os.getenv("JOINQUANT_PASSWORD") or os.getenv("JQ_PASSWORD") or jq_cfg.get("password")
    return username if _truthy_secret(username) else None, password if _truthy_secret(password) else None


def redact(text: Any, root: Path) -> str:
    out = str(text)
    username, password = credential_pair(root)
    for secret in [username, password]:
        if secret:
            out = out.replace(secret, "[REDACTED]")
    return out


def normalize_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    text = text.str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def format_date(values: pd.Series) -> pd.Series:
    return normalize_date(values).dt.strftime("%Y%m%d")


def provider_readiness(root: Path, config: LiveLabelSourceAcquisitionConfig) -> pd.DataFrame:
    username, password = credential_pair(root)
    sdk_ready = importlib.util.find_spec("jqdatasdk") is not None
    target_exists = bool(config.target_security)
    return pd.DataFrame(
        [
            {
                "provider": "joinquant",
                "check": "jqdatasdk_installed",
                "ready": sdk_ready,
                "status": "pass" if sdk_ready else "blocked",
                "detail": "python package jqdatasdk",
            },
            {
                "provider": "joinquant",
                "check": "username_present",
                "ready": username is not None,
                "status": "pass" if username is not None else "blocked",
                "detail": "env var or configs/data_credentials.json username",
            },
            {
                "provider": "joinquant",
                "check": "password_present",
                "ready": password is not None,
                "status": "pass" if password is not None else "blocked",
                "detail": "env var or configs/data_credentials.json password",
            },
            {
                "provider": "joinquant",
                "check": "target_security_configured",
                "ready": target_exists,
                "status": "pass" if target_exists else "blocked",
                "detail": config.target_security,
            },
            {
                "provider": "local",
                "check": "target_source_absent_before_attempt",
                "ready": not config.target_source_path.exists(),
                "status": "pass" if not config.target_source_path.exists() else "blocked",
                "detail": config.target_source_path.as_posix(),
            },
        ]
    )


def readiness_ok(readiness: pd.DataFrame) -> bool:
    required = {"jqdatasdk_installed", "username_present", "password_present", "target_security_configured"}
    rows = readiness.loc[readiness["check"].isin(required)].copy()
    return bool(len(rows) == len(required) and rows["ready"].astype(bool).all())


def build_probe_plan(config: LiveLabelSourceAcquisitionConfig, readiness: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attempt_id": "joinquant_full_history_market_proxy",
                "provider": "joinquant",
                "endpoint": "jqdatasdk.get_price",
                "target_security": config.target_security,
                "target_asset_or_index": config.target_asset_or_index,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "fields": ",".join(config.fields),
                "fq": config.fq,
                "execute_probe": config.execute_probe,
                "ready_to_execute": bool(config.execute_probe and readiness_ok(readiness)),
                "acceptance_boundary": "candidate only unless coverage, contract, and approval gates pass",
            },
            {
                "attempt_id": "official_target_write_gate",
                "provider": "local",
                "endpoint": "write_market_total_return_index_csv",
                "target_security": config.target_security,
                "target_asset_or_index": config.target_asset_or_index,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "fields": ",".join(config.fields),
                "fq": config.fq,
                "execute_probe": config.execute_probe,
                "ready_to_execute": False,
                "acceptance_boundary": "write requires explicit allow_target_write and accepted source basis",
            },
        ]
    )


def normalize_joinquant_result(raw: pd.DataFrame, config: LiveLabelSourceAcquisitionConfig) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_SOURCE_COLUMNS + ["candidate_source_role"])
    work = raw.copy()
    if "time" in work.columns:
        date_values = pd.to_datetime(work["time"], errors="coerce")
    else:
        date_values = pd.to_datetime(work.index, errors="coerce")
    if "close" not in work.columns:
        raise ValueError("JoinQuant get_price result missing close column")
    close = pd.to_numeric(work["close"], errors="coerce")
    dates = date_values.strftime("%Y%m%d")
    out = pd.DataFrame(
        {
            "date": dates,
            "asset_or_index": config.target_asset_or_index,
            "total_return_index_or_adjusted_close": close,
            "available_date": dates,
            "data_source": f"jqdatasdk.get_price.{config.target_security}.fq_{config.fq}.candidate_adjusted_proxy",
            "source_vintage": f"fetched_at_{now_text()}",
            "candidate_source_role": "candidate_not_certified_total_return",
        }
    )
    return out.dropna(subset=["date", "total_return_index_or_adjusted_close"]).sort_values("date").reset_index(drop=True)


def execute_joinquant_attempt(root: Path, config: LiveLabelSourceAcquisitionConfig, readiness: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.execute_probe:
        return (
            pd.DataFrame(
                [
                    {
                        "attempt_id": "joinquant_full_history_market_proxy",
                        "status": "not_executed",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "detail": "execute_probe=false",
                    }
                ]
            ),
            pd.DataFrame(columns=REQUIRED_SOURCE_COLUMNS + ["candidate_source_role"]),
        )
    if not readiness_ok(readiness):
        blocked = ",".join(readiness.loc[~readiness["ready"].astype(bool), "check"].astype(str))
        return (
            pd.DataFrame(
                [
                    {
                        "attempt_id": "joinquant_full_history_market_proxy",
                        "status": "blocked",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "detail": blocked,
                    }
                ]
            ),
            pd.DataFrame(columns=REQUIRED_SOURCE_COLUMNS + ["candidate_source_role"]),
        )

    username, password = credential_pair(root)
    import jqdatasdk as jq  # type: ignore

    jq.auth(username, password)
    raw = jq.get_price(
        config.target_security,
        start_date=config.start_date,
        end_date=config.end_date,
        frequency="daily",
        fields=list(config.fields),
        fq=config.fq,
    )
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"unexpected get_price result type: {type(raw).__name__}")
    normalized = normalize_joinquant_result(raw, config)
    return (
        pd.DataFrame(
            [
                {
                    "attempt_id": "joinquant_full_history_market_proxy",
                    "status": "ok",
                    "rows": int(len(normalized)),
                    "first_date": str(normalized["date"].min()) if not normalized.empty else "",
                    "last_date": str(normalized["date"].max()) if not normalized.empty else "",
                    "detail": f"raw_rows={len(raw)};normalized_rows={len(normalized)}",
                }
            ]
        ),
        normalized,
    )


def build_source_quality_assessment(candidate: pd.DataFrame, config: LiveLabelSourceAcquisitionConfig) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(candidate.columns))
    values = pd.to_numeric(candidate.get("total_return_index_or_adjusted_close", pd.Series(dtype=float)), errors="coerce")
    dates = normalize_date(candidate.get("date", pd.Series(dtype=str))) if not candidate.empty else pd.Series(dtype="datetime64[ns]")
    available = normalize_date(candidate.get("available_date", pd.Series(dtype=str))) if not candidate.empty else pd.Series(dtype="datetime64[ns]")
    row_count = int(len(candidate))
    date_span_ok = bool(row_count >= config.min_source_rows)
    basis_accepted = config.approved_source_basis in {"certified_total_return", "explicit_adjusted_market_proxy"}
    rows = [
        {
            "check": "required_columns_present",
            "status": _status(not missing, "blocked"),
            "detail": ",".join(missing),
        },
        {
            "check": "minimum_history_rows",
            "status": _status(row_count >= config.min_source_rows, "blocked"),
            "detail": f"rows={row_count};min={config.min_source_rows}",
        },
        {
            "check": "positive_numeric_values",
            "status": _status(row_count > 0 and values.notna().all() and (values > 0).all(), "blocked"),
            "detail": f"bad_rows={int((values.isna() | (values <= 0)).sum()) if row_count else 0}",
        },
        {
            "check": "dates_parseable",
            "status": _status(row_count > 0 and dates.notna().all() and available.notna().all(), "blocked"),
            "detail": f"bad_date_rows={int(dates.isna().sum()) if row_count else 0};bad_available_rows={int(available.isna().sum()) if row_count else 0}",
        },
        {
            "check": "available_date_not_before_date",
            "status": _status(row_count > 0 and (available >= dates).all(), "blocked"),
            "detail": f"bad_rows={int((available < dates).sum()) if row_count else 0}",
        },
        {
            "check": "accepted_source_basis",
            "status": _status(basis_accepted, "blocked"),
            "detail": f"approved_source_basis={config.approved_source_basis}",
        },
        {
            "check": "live_candidate_is_not_auto_certified_total_return",
            "status": "pass",
            "detail": "JoinQuant get_price probe remains candidate evidence unless source basis is explicitly approved",
        },
        {
            "check": "long_history_gate",
            "status": _status(date_span_ok, "blocked"),
            "detail": f"first_date={str(candidate['date'].min()) if row_count else ''};last_date={str(candidate['date'].max()) if row_count else ''}",
        },
    ]
    return pd.DataFrame(rows)


def build_signal_coverage(candidate: pd.DataFrame, signal_panel: pd.DataFrame, config: LiveLabelSourceAcquisitionConfig) -> pd.DataFrame:
    if candidate.empty or signal_panel.empty:
        return pd.DataFrame(
            [
                {
                    "horizon": int(h),
                    "unique_signal_dates": int(signal_panel.get("signal_date", pd.Series(dtype=str)).astype(str).nunique()) if not signal_panel.empty else 0,
                    "matched_source_dates": 0,
                    "enough_future_dates": 0,
                    "coverage_ratio": 0.0,
                    "coverage_status": "blocked",
                }
                for h in config.horizons
            ]
        )
    signals = signal_panel.loc[signal_panel["asset"].astype(str).eq("MARKET")].copy()
    signal_dates = sorted(signals["signal_date"].astype(str).str.replace("-", "", regex=False).unique())
    source = candidate.copy().sort_values("date").reset_index(drop=True)
    source_dates = source["date"].astype(str).tolist()
    pos_by_date = {date: idx for idx, date in enumerate(source_dates)}
    rows = []
    for horizon in config.horizons:
        matched = 0
        enough_future = 0
        for signal_date in signal_dates:
            pos = pos_by_date.get(signal_date)
            if pos is None:
                continue
            matched += 1
            if pos + int(horizon) < len(source):
                enough_future += 1
        coverage = enough_future / len(signal_dates) if signal_dates else 0.0
        rows.append(
            {
                "horizon": int(horizon),
                "unique_signal_dates": int(len(signal_dates)),
                "matched_source_dates": int(matched),
                "enough_future_dates": int(enough_future),
                "coverage_ratio": float(coverage),
                "coverage_status": "pass" if coverage >= config.min_signal_coverage_ratio else "blocked",
            }
        )
    return pd.DataFrame(rows)


def build_write_decision(
    candidate: pd.DataFrame,
    quality: pd.DataFrame,
    coverage: pd.DataFrame,
    config: LiveLabelSourceAcquisitionConfig,
) -> pd.DataFrame:
    quality_ok = bool(not quality.empty and quality["status"].eq("pass").all())
    coverage_ok = bool(not coverage.empty and coverage["coverage_status"].eq("pass").all())
    target_absent = not config.target_source_path.exists()
    allow = bool(config.allow_target_write and quality_ok and coverage_ok and target_absent)
    reasons = []
    if not config.allow_target_write:
        reasons.append("allow_target_write=false")
    if not quality_ok:
        reasons.append("source_quality_gate_not_passed")
    if not coverage_ok:
        reasons.append("signal_coverage_gate_not_passed")
    if not target_absent:
        reasons.append("target_source_already_exists")
    return pd.DataFrame(
        [
            {
                "target_path": config.target_source_path.as_posix(),
                "candidate_rows": int(len(candidate)),
                "quality_gate_pass": quality_ok,
                "signal_coverage_gate_pass": coverage_ok,
                "allow_target_write": config.allow_target_write,
                "write_allowed": allow,
                "decision": "write_target_source" if allow else "do_not_write_target_source",
                "reason": ";".join(reasons),
            }
        ]
    )


def maybe_write_target(candidate: pd.DataFrame, write_decision: pd.DataFrame, config: LiveLabelSourceAcquisitionConfig) -> pd.DataFrame:
    write_allowed = bool(write_decision["write_allowed"].iloc[0]) if not write_decision.empty else False
    if not write_allowed:
        return pd.DataFrame(
            [
                {
                    "action": "write_market_total_return_index_csv",
                    "status": "blocked",
                    "rows": 0,
                    "detail": "write decision blocked",
                }
            ]
        )
    out = candidate.loc[:, REQUIRED_SOURCE_COLUMNS].copy()
    config.target_source_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.target_source_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(
        [
            {
                "action": "write_market_total_return_index_csv",
                "status": "written",
                "rows": int(len(out)),
                "detail": config.target_source_path.as_posix(),
            }
        ]
    )


def manual_data_requirements(config: LiveLabelSourceAcquisitionConfig, coverage: pd.DataFrame) -> pd.DataFrame:
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    return pd.DataFrame(
        [
            {
                "requirement_id": "official_total_return_market_index",
                "priority": 1,
                "provider_or_route": "CSIndex/Wind/Choice/vendor file",
                "required_fields": ",".join(REQUIRED_SOURCE_COLUMNS),
                "minimum_history": f"{config.start_date} to {config.end_date}",
                "current_status": "required",
                "reason": "price-only broad index and short adjusted proxy cannot support investable label validation",
            },
            {
                "requirement_id": "joinquant_extended_history_adjusted_proxy",
                "priority": 2,
                "provider_or_route": "JoinQuant paid or upgraded data permission",
                "required_fields": ",".join(REQUIRED_SOURCE_COLUMNS),
                "minimum_history": f"{config.min_source_rows}+ daily rows and {config.min_signal_coverage_ratio:.0%}+ signal coverage",
                "current_status": "optional_if_explicitly_approved",
                "reason": f"current candidate min signal coverage={min_coverage:.4f}",
            },
            {
                "requirement_id": "tushare_or_vendor_total_return_endpoint",
                "priority": 3,
                "provider_or_route": "Tushare/Wind/Choice custom endpoint or exported file",
                "required_fields": ",".join(REQUIRED_SOURCE_COLUMNS),
                "minimum_history": f"{config.start_date} to {config.end_date}",
                "current_status": "manual_api_permission_needed",
                "reason": "current known Tushare permission does not expose a governed total-return index source",
            },
        ]
    )


def build_no_promotion_guard(write_result: pd.DataFrame) -> pd.DataFrame:
    target_written = bool(write_result["status"].eq("written").any()) if not write_result.empty else False
    return pd.DataFrame(
        [
            {
                "result_type": "candidate_source_probe",
                "produced": True,
                "blocked": False,
                "reason": "source acquisition attempt evidence produced",
            },
            {
                "result_type": "official_market_total_return_source",
                "produced": target_written,
                "blocked": not target_written,
                "reason": "target source write requires all V3.74 gates",
            },
            {
                "result_type": "forward_total_return_labels",
                "produced": False,
                "blocked": True,
                "reason": "V3.53 must validate target source before labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.74 does not run portfolio validation",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "no accepted label evidence yet",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    attempt: pd.DataFrame,
    quality: pd.DataFrame,
    coverage: pd.DataFrame,
    write_decision: pd.DataFrame,
    write_result: pd.DataFrame,
    guard: pd.DataFrame,
) -> pd.DataFrame:
    readiness_fail = readiness["status"].eq("fail").any() if not readiness.empty else True
    attempt_recorded = not attempt.empty
    quality_recorded = not quality.empty
    coverage_recorded = not coverage.empty
    target_written = bool(write_result["status"].eq("written").any()) if not write_result.empty else False
    blocked_outputs_produced = bool(guard.loc[guard["result_type"].isin(["portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any())
    return pd.DataFrame(
        [
            {
                "check": "provider_readiness_recorded",
                "status": "pass" if not readiness_fail else "fail",
                "detail": ",".join(readiness.loc[readiness["status"].eq("fail"), "check"].astype(str)) if not readiness.empty else "missing_readiness",
            },
            {
                "check": "live_attempt_recorded",
                "status": "pass" if attempt_recorded else "fail",
                "detail": f"attempt_rows={len(attempt)}",
            },
            {
                "check": "source_quality_assessment_recorded",
                "status": "pass" if quality_recorded else "fail",
                "detail": f"quality_rows={len(quality)}",
            },
            {
                "check": "signal_coverage_assessment_recorded",
                "status": "pass" if coverage_recorded else "fail",
                "detail": f"coverage_rows={len(coverage)}",
            },
            {
                "check": "target_write_decision_recorded",
                "status": "pass" if not write_decision.empty else "fail",
                "detail": str(write_decision["decision"].iloc[0]) if not write_decision.empty else "",
            },
            {
                "check": "target_not_written_unless_allowed",
                "status": "pass" if (not target_written or bool(write_decision["write_allowed"].iloc[0])) else "fail",
                "detail": f"target_written={target_written}",
            },
            {
                "check": "no_portfolio_or_model_outputs",
                "status": "pass" if not blocked_outputs_produced else "fail",
                "detail": "source acquisition only",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    for _, row in frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_report(
    readiness: pd.DataFrame,
    attempt: pd.DataFrame,
    quality: pd.DataFrame,
    coverage: pd.DataFrame,
    write_decision: pd.DataFrame,
    manual_requirements: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: LiveLabelSourceAcquisitionConfig,
) -> str:
    rows = int(attempt["rows"].iloc[0]) if not attempt.empty and "rows" in attempt.columns else 0
    first_date = str(attempt["first_date"].iloc[0]) if not attempt.empty and "first_date" in attempt.columns else ""
    last_date = str(attempt["last_date"].iloc[0]) if not attempt.empty and "last_date" in attempt.columns else ""
    write_allowed = bool(write_decision["write_allowed"].iloc[0]) if not write_decision.empty else False
    lines = [
        "# V3.74 Live Label Source Acquisition Attempt",
        "",
        "## Decision",
        "",
        "- V3.74 performs a live provider acquisition attempt for the missing MARKET label source.",
        "- Provider candidate data is not automatically treated as certified total-return evidence.",
        "- Target source write remains blocked unless all quality, coverage, and approval gates pass.",
        "",
        "## Key Metrics",
        "",
        f"- Candidate rows: `{rows}`",
        f"- Candidate date range: `{first_date}` to `{last_date}`",
        f"- Target write allowed: `{write_allowed}`",
        f"- Target source path: `{config.target_source_path.as_posix()}`",
        "",
        "## Provider Readiness",
        "",
    ]
    lines.extend(markdown_table(readiness, ["provider", "check", "ready", "status", "detail"], 20))
    lines.extend(["", "## Attempt Result", ""])
    lines.extend(markdown_table(attempt, ["attempt_id", "status", "rows", "first_date", "last_date", "detail"], 20))
    lines.extend(["", "## Source Quality", ""])
    lines.extend(markdown_table(quality, ["check", "status", "detail"], 20))
    lines.extend(["", "## Signal Coverage", ""])
    lines.extend(markdown_table(coverage, ["horizon", "unique_signal_dates", "matched_source_dates", "enough_future_dates", "coverage_ratio", "coverage_status"], 20))
    lines.extend(["", "## Write Decision", ""])
    lines.extend(markdown_table(write_decision, ["candidate_rows", "quality_gate_pass", "signal_coverage_gate_pass", "allow_target_write", "write_allowed", "decision", "reason"], 20))
    lines.extend(["", "## Manual Data Requirements", ""])
    lines.extend(markdown_table(manual_requirements, ["requirement_id", "priority", "provider_or_route", "current_status", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- If a vendor or upgraded provider can supply a compliant long-history source, place it at `data_raw/market_labels/market_total_return_index.csv` and rerun V3.53.",
            "- If an adjusted proxy is explicitly approved later, keep it separate from certified total-return claims and rerun V3.74 with `allow_target_write=true` only after coverage passes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(attempt: pd.DataFrame, quality: pd.DataFrame, coverage: pd.DataFrame, write_decision: pd.DataFrame) -> str:
    rows = int(attempt["rows"].iloc[0]) if not attempt.empty and "rows" in attempt.columns else 0
    quality_pass = bool(quality["status"].eq("pass").all()) if not quality.empty else False
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    write_allowed = bool(write_decision["write_allowed"].iloc[0]) if not write_decision.empty else False
    return "\n".join(
        [
            "# A-share Live Label Source Acquisition Attempt V3.74",
            "",
            "## Dataset Decision",
            "",
            f"- Candidate rows: `{rows}`",
            f"- Quality gate pass: `{quality_pass}`",
            f"- Minimum signal coverage: `{min_coverage:.4f}`",
            f"- Target write allowed: `{write_allowed}`",
            "- Official label generation remains delegated to V3.53 after a compliant source exists.",
            "",
        ]
    )
