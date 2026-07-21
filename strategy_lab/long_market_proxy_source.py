"""Governed long-history MARKET proxy source builder for HIRSSM V3.58.

V3.58 converts local CSIndex daily index levels into an explicitly labelled
long-history price-index proxy. It must not present the proxy as an official
total-return index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateIndexSource:
    index_code: str
    asset_or_index: str
    role: str
    source_path: Path


@dataclass(frozen=True)
class LongMarketProxyConfig:
    candidate_sources: tuple[CandidateIndexSource, ...]
    primary_index_code: str
    output_dir: Path
    catalog_path: Path
    raw_proxy_path: Path
    official_total_return_path: Path
    as_of_date: str
    min_primary_rows: int
    min_regular_rows: int
    min_start_date: str
    min_end_date: str
    max_staleness_days: int
    max_return_diff_bps: float


REQUIRED_DAILY_COLUMNS = {
    "date",
    "index_code",
    "index_name",
    "close",
    "pct_chg",
    "data_source",
    "fetched_at",
}


def normalize_yyyymmdd(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values.astype(str), errors="coerce")
    return parsed.dt.strftime("%Y%m%d")


def _status(ok: bool, fail_status: str = "blocked") -> str:
    return "pass" if ok else fail_status


def _read_daily(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _regular_trading_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if "is_full_ohlc_bar" in frame.columns:
        mask &= frame["is_full_ohlc_bar"].fillna(False).astype(bool)
    if "is_zero_volume_amount" in frame.columns:
        mask &= ~frame["is_zero_volume_amount"].fillna(False).astype(bool)
    close = pd.to_numeric(frame.get("close", pd.Series(np.nan, index=frame.index)), errors="coerce")
    mask &= close.notna() & (close > 0)
    return mask


def _pct_consistency_bps(frame: pd.DataFrame) -> tuple[float, int]:
    ordered = frame.sort_values("date").copy()
    close = pd.to_numeric(ordered["close"], errors="coerce")
    reported = pd.to_numeric(ordered["pct_chg"], errors="coerce")
    computed = close.pct_change() * 100.0
    diff_bps = (computed - reported).abs() * 100.0
    valid = diff_bps.dropna()
    if valid.empty:
        return np.nan, 0
    return float(valid.quantile(0.99)), int((valid > 5.0).sum())


def audit_candidate_sources(config: LongMarketProxyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in config.candidate_sources:
        path = candidate.source_path
        if not path.exists():
            rows.append(
                {
                    "index_code": candidate.index_code,
                    "asset_or_index": candidate.asset_or_index,
                    "role": candidate.role,
                    "source_path": str(path),
                    "status": "blocked_missing_file",
                    "rows_all": 0,
                    "regular_rows": 0,
                    "date_min": "",
                    "date_max": "",
                    "quality_note": "source file missing",
                }
            )
            continue
        frame = _read_daily(path)
        missing = sorted(REQUIRED_DAILY_COLUMNS.difference(frame.columns))
        if missing:
            rows.append(
                {
                    "index_code": candidate.index_code,
                    "asset_or_index": candidate.asset_or_index,
                    "role": candidate.role,
                    "source_path": str(path),
                    "status": "blocked_missing_columns",
                    "rows_all": int(len(frame)),
                    "regular_rows": 0,
                    "date_min": "",
                    "date_max": "",
                    "quality_note": ",".join(missing),
                }
            )
            continue
        regular = frame.loc[_regular_trading_mask(frame)].copy()
        regular["date"] = normalize_yyyymmdd(regular["date"])
        duplicate_dates = int(regular["date"].duplicated().sum())
        nonpositive_close = int((pd.to_numeric(regular["close"], errors="coerce") <= 0).sum())
        p99_diff_bps, large_diff_count = _pct_consistency_bps(regular)
        date_min = str(regular["date"].min()) if not regular.empty else ""
        date_max = str(regular["date"].max()) if not regular.empty else ""
        quality_ok = (
            len(regular) >= config.min_regular_rows
            and duplicate_dates == 0
            and nonpositive_close == 0
            and date_min <= config.min_start_date
            and date_max >= config.min_end_date
        )
        rows.append(
            {
                "index_code": candidate.index_code,
                "asset_or_index": candidate.asset_or_index,
                "role": candidate.role,
                "source_path": str(path),
                "status": "ok" if quality_ok else "monitor_or_blocked",
                "rows_all": int(len(frame)),
                "regular_rows": int(len(regular)),
                "date_min": date_min,
                "date_max": date_max,
                "duplicate_dates": duplicate_dates,
                "nonpositive_close": nonpositive_close,
                "p99_pct_chg_diff_bps": p99_diff_bps,
                "large_pct_chg_diff_rows": large_diff_count,
                "close_only_rows": int(frame.get("is_close_only_bar", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
                "zero_volume_amount_rows": int(frame.get("is_zero_volume_amount", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
                "data_source": ",".join(sorted(frame["data_source"].astype(str).unique())[:5]),
                "fetched_at": ",".join(sorted(frame["fetched_at"].astype(str).unique())[:3]),
                "quality_note": "long_price_index_proxy_candidate_not_total_return",
            }
        )
    return pd.DataFrame(rows)


def build_primary_proxy_source(config: LongMarketProxyConfig) -> pd.DataFrame:
    candidates = {candidate.index_code: candidate for candidate in config.candidate_sources}
    primary = candidates[config.primary_index_code]
    frame = _read_daily(primary.source_path)
    out = frame.loc[_regular_trading_mask(frame)].copy()
    out["date"] = normalize_yyyymmdd(out["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(out["close"], errors="coerce")
    out["asset_or_index"] = primary.asset_or_index
    out["index_code"] = primary.index_code
    out["source_symbol"] = f"{primary.index_code}.CSI"
    out["market_level"] = close
    out["available_date"] = out["date"]
    out["data_source"] = out["data_source"].astype(str) + "|long_price_index_proxy_v3_58"
    out["source_vintage"] = out["fetched_at"].astype(str)
    out["source_type"] = "csindex_price_index_proxy"
    out["return_basis_candidate"] = "price_index_return"
    out["price_adjustment_status"] = "index_price_level_not_total_return"
    out["official_total_return"] = False
    out["point_in_time_note"] = "same_day_after_close_index_level"
    out["proxy_label_generation_allowed"] = True
    out["official_label_generation_allowed"] = False
    out["model_promotion_allowed"] = False
    out["performance_claim_allowed"] = False
    columns = [
        "date",
        "asset_or_index",
        "index_code",
        "source_symbol",
        "index_name",
        "index_short_name",
        "market_level",
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "volume",
        "amount",
        "available_date",
        "data_source",
        "source_vintage",
        "source_type",
        "return_basis_candidate",
        "price_adjustment_status",
        "official_total_return",
        "point_in_time_note",
        "proxy_label_generation_allowed",
        "official_label_generation_allowed",
        "model_promotion_allowed",
        "performance_claim_allowed",
    ]
    return out[[col for col in columns if col in out.columns]].reset_index(drop=True)


def build_source_contract_checks(
    audit: pd.DataFrame,
    proxy: pd.DataFrame,
    config: LongMarketProxyConfig,
) -> pd.DataFrame:
    primary = audit.loc[audit["index_code"].astype(str) == config.primary_index_code]
    primary_row = primary.iloc[0].to_dict() if not primary.empty else {}
    p99_diff_bps = pd.to_numeric(pd.Series([primary_row.get("p99_pct_chg_diff_bps")]), errors="coerce").iloc[0]
    dates = proxy["date"].astype(str) if not proxy.empty else pd.Series(dtype=str)
    available = proxy["available_date"].astype(str) if not proxy.empty else pd.Series(dtype=str)
    as_of = pd.to_datetime(config.as_of_date, format="%Y%m%d", errors="coerce")
    end_date = pd.to_datetime(str(dates.max()), format="%Y%m%d", errors="coerce") if not dates.empty else pd.NaT
    staleness_days = int((as_of - end_date).days) if pd.notna(as_of) and pd.notna(end_date) else 999999
    official_flags = proxy.get("official_total_return", pd.Series(True, index=proxy.index)).astype(bool)
    rows = [
        {
            "check": "primary_candidate_file_audited",
            "status": _status(not primary.empty and str(primary_row.get("status")) == "ok"),
            "detail": f"status={primary_row.get('status', '')}",
        },
        {
            "check": "primary_proxy_minimum_rows",
            "status": _status(len(proxy) >= config.min_primary_rows),
            "detail": f"rows={len(proxy)};min={config.min_primary_rows}",
        },
        {
            "check": "primary_proxy_date_range",
            "status": _status((not dates.empty) and dates.min() <= config.min_start_date and dates.max() >= config.min_end_date),
            "detail": f"date_min={dates.min() if not dates.empty else ''};date_max={dates.max() if not dates.empty else ''}",
        },
        {
            "check": "primary_proxy_staleness",
            "status": _status(staleness_days <= config.max_staleness_days, "warn"),
            "detail": f"staleness_days={staleness_days};max={config.max_staleness_days};as_of={config.as_of_date}",
        },
        {
            "check": "primary_proxy_no_duplicate_dates",
            "status": _status(int(dates.duplicated().sum()) == 0),
            "detail": f"duplicates={int(dates.duplicated().sum()) if not dates.empty else 0}",
        },
        {
            "check": "primary_pct_chg_consistency",
            "status": _status(pd.notna(p99_diff_bps) and float(p99_diff_bps) <= config.max_return_diff_bps),
            "detail": f"p99_diff_bps={float(p99_diff_bps) if pd.notna(p99_diff_bps) else np.nan:.4f};max={config.max_return_diff_bps:.4f}",
        },
        {
            "check": "primary_proxy_positive_level",
            "status": _status(pd.to_numeric(proxy.get("market_level", pd.Series(dtype=float)), errors="coerce").gt(0).all()),
            "detail": f"bad_rows={int((pd.to_numeric(proxy.get('market_level', pd.Series(dtype=float)), errors='coerce') <= 0).sum())}",
        },
        {
            "check": "available_date_not_before_observation_date",
            "status": _status((available >= dates).all() if not proxy.empty else False),
            "detail": f"bad_rows={int((available < dates).sum()) if not proxy.empty else 0}",
        },
        {
            "check": "proxy_marked_not_official_total_return",
            "status": _status(not official_flags.any()),
            "detail": "official_total_return must remain false",
        },
        {
            "check": "source_symbol_preserves_padded_index_code",
            "status": _status("source_symbol" in proxy.columns and proxy["source_symbol"].astype(str).eq(f"{config.primary_index_code}.CSI").all()),
            "detail": f"expected={config.primary_index_code}.CSI",
        },
        {
            "check": "official_total_return_source_absent_or_untouched",
            "status": _status(not config.official_total_return_path.exists()),
            "detail": str(config.official_total_return_path),
        },
    ]
    return pd.DataFrame(rows)


def build_write_policy(config: LongMarketProxyConfig, checks: pd.DataFrame) -> pd.DataFrame:
    critical = checks.loc[~checks["check"].isin(["primary_proxy_staleness"])]
    proxy_ready = bool(critical["status"].eq("pass").all())
    return pd.DataFrame(
        [
            {
                "target": str(config.raw_proxy_path),
                "write_allowed": proxy_ready,
                "written_by_v3_58": proxy_ready,
                "target_role": "long_history_market_price_index_proxy",
                "must_not_be_used_as": "official_total_return_index",
            },
            {
                "target": str(config.official_total_return_path),
                "write_allowed": False,
                "written_by_v3_58": False,
                "target_role": "official_market_total_return_index",
                "must_not_be_used_as": "price_proxy_substitution",
            },
        ]
    )


def build_label_contract_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contract_item": "V3.53 official label importer",
                "current_requirement": "return_basis must be total_return or adjusted_return from a governed source",
                "v3_58_source_status": "price_index_return proxy only",
                "decision": "do_not_feed_into_official_total_return_importer",
            },
            {
                "contract_item": "future V3.59 proxy label importer",
                "current_requirement": "separate schema and explicit price_proxy basis",
                "v3_58_source_status": "ready for governed proxy-label smoke or proxy validation only",
                "decision": "allowed_after_new_guarded_importer",
            },
        ]
    )


def build_no_promotion_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "long_market_price_proxy_source",
                "produced": True,
                "blocked": False,
                "reason": "source construction only",
            },
            {
                "result_type": "official_market_total_return_source",
                "produced": False,
                "blocked": True,
                "reason": "CSIndex daily close is a price-index proxy, not an official total-return source",
            },
            {
                "result_type": "state_stratified_signal_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.58 does not create labels or validate signals",
            },
            {
                "result_type": "portfolio_backtest_or_model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "source acquisition cannot promote a model",
            },
        ]
    )


def build_acceptance_checks(
    checks: pd.DataFrame,
    write_policy: pd.DataFrame,
    guard: pd.DataFrame,
    config: LongMarketProxyConfig,
) -> pd.DataFrame:
    critical = checks.loc[~checks["check"].isin(["primary_proxy_staleness"])]
    proxy_policy = write_policy.loc[write_policy["target_role"] == "long_history_market_price_index_proxy"].iloc[0]
    official_policy = write_policy.loc[write_policy["target_role"] == "official_market_total_return_index"].iloc[0]
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(
                ["official_market_total_return_source", "state_stratified_signal_validation", "portfolio_backtest_or_model_promotion"]
            ),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "critical_source_checks_passed",
                "status": "pass" if critical["status"].eq("pass").all() else "fail",
                "detail": ";".join(critical.loc[critical["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "long_price_proxy_written",
                "status": "pass" if bool(proxy_policy.write_allowed) and config.raw_proxy_path.exists() else "fail",
                "detail": str(config.raw_proxy_path),
            },
            {
                "check": "official_total_return_not_written",
                "status": "pass" if not bool(official_policy.written_by_v3_58) and not config.official_total_return_path.exists() else "fail",
                "detail": str(config.official_total_return_path),
            },
            {
                "check": "performance_outputs_not_produced",
                "status": "pass" if not forbidden_produced else "fail",
                "detail": "V3.58 source acquisition only",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 16) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def build_report(
    audit: pd.DataFrame,
    proxy: pd.DataFrame,
    checks: pd.DataFrame,
    write_policy: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: LongMarketProxyConfig,
) -> str:
    dates = proxy["date"].astype(str) if not proxy.empty else pd.Series(dtype=str)
    lines = [
        "# V3.58 Long MARKET Price Proxy Source",
        "",
        "## Decision",
        "",
        "- V3.58 normalizes local CSIndex `000985` daily index levels into a governed long-history MARKET price proxy.",
        "- It writes a separate proxy file and does not write `data_raw/market_labels/market_total_return_index.csv`.",
        "- It does not create labels, run signal validation, run a portfolio backtest, or promote any model.",
        "",
        "## Primary Proxy",
        "",
        f"- Index code: `{config.primary_index_code}`",
        f"- Rows: `{len(proxy)}`",
        f"- Date range: `{dates.min() if not dates.empty else ''}` to `{dates.max() if not dates.empty else ''}`",
        f"- Raw proxy path: `{config.raw_proxy_path}`",
        f"- Official total-return path untouched: `{config.official_total_return_path}`",
        "",
        "## Candidate Audit",
        "",
    ]
    lines.extend(markdown_table(audit, ["index_code", "asset_or_index", "role", "status", "regular_rows", "date_min", "date_max", "quality_note"]))
    lines.extend(["", "## Source Checks", ""])
    lines.extend(markdown_table(checks, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## Write Policy", ""])
    lines.extend(markdown_table(write_policy, ["target_role", "write_allowed", "written_by_v3_58", "must_not_be_used_as"], max_rows=10))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=10))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.59 can build a separate proxy-label importer using `return_basis=price_index_return`.",
            "- Official total-return validation remains blocked until a real total-return source is acquired.",
            "- Any result based on this proxy must be labelled price-index proxy evidence, not investable total-return evidence.",
        ]
    )
    return "\n".join(lines)


def build_catalog(config: LongMarketProxyConfig, proxy: pd.DataFrame) -> str:
    dates = proxy["date"].astype(str) if not proxy.empty else pd.Series(dtype=str)
    return "\n".join(
        [
            "# A-share Long MARKET Price Proxy Source V3.58",
            "",
            "## Dataset Role",
            "",
            "This dataset is a governed long-history price-index proxy for MARKET-level diagnostics.",
            "",
            "## Source",
            "",
            f"- Primary index: `{config.primary_index_code}`",
            f"- Rows: `{len(proxy)}`",
            f"- Date range: `{dates.min() if not dates.empty else ''}` to `{dates.max() if not dates.empty else ''}`",
            f"- Proxy file: `{config.raw_proxy_path}`",
            "",
            "## Governance",
            "",
            "- Allowed use: price-index proxy labels, state diagnostics, and non-official validation experiments after an explicit guarded importer exists.",
            "- Forbidden use: official total-return labels, dividend-inclusive performance claims, direct model promotion, or portfolio backtest promotion.",
            "- `data_raw/market_labels/market_total_return_index.csv` is intentionally not written by V3.58.",
        ]
    )
