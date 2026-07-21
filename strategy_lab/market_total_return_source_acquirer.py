"""Governed acquisition router for MARKET total-return sources, V3.54."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MARKET_SOURCE_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]

TOTAL_RETURN_KEYWORDS = ("全收益", "总收益", "total return", "tr")
FORBIDDEN_ACCEPTANCE_NOTE = "price-only index close must not be promoted as total-return label source"


@dataclass(frozen=True)
class SourceAcquirerConfig:
    index_list_path: Path
    target_source_path: Path
    output_dir: Path
    catalog_path: Path
    start_date: str
    end_date: str
    preferred_market_asset: str
    execute: bool
    allow_adjusted_proxy: bool
    allow_overwrite_target: bool
    route_priority: tuple[str, ...]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def provider_readiness(root: Path) -> pd.DataFrame:
    credentials = read_credentials(root)
    jq_cfg = credentials.get("joinquant", {}) if isinstance(credentials, dict) else {}
    ts_cfg = credentials.get("tushare", {}) if isinstance(credentials, dict) else {}
    jq_user = os.getenv("JQDATA_USERNAME") or os.getenv("JOINQUANT_USERNAME") or os.getenv("JQ_USER") or jq_cfg.get("username")
    jq_password = os.getenv("JQDATA_PASSWORD") or os.getenv("JOINQUANT_PASSWORD") or os.getenv("JQ_PASSWORD") or jq_cfg.get("password")
    ts_token = os.getenv("TUSHARE_TOKEN") or ts_cfg.get("token")
    rows = [
        {
            "provider": "akshare",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("akshare") is not None,
            "detail": "python package akshare",
        },
        {
            "provider": "joinquant",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("jqdatasdk") is not None,
            "detail": "python package jqdatasdk",
        },
        {
            "provider": "joinquant",
            "check": "username_present",
            "ready": _truthy_secret(jq_user),
            "detail": "env var or configs/data_credentials.json username",
        },
        {
            "provider": "joinquant",
            "check": "password_present",
            "ready": _truthy_secret(jq_password),
            "detail": "env var or configs/data_credentials.json password",
        },
        {
            "provider": "tushare",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("tushare") is not None,
            "detail": "python package tushare",
        },
        {
            "provider": "tushare",
            "check": "token_present",
            "ready": _truthy_secret(ts_token),
            "detail": "env var or configs/data_credentials.json token",
        },
    ]
    return pd.DataFrame(rows)


def provider_ready(readiness: pd.DataFrame, provider: str) -> bool:
    rows = readiness.loc[readiness["provider"] == provider]
    return bool(not rows.empty and rows["ready"].astype(bool).all())


def normalize_index_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return ""
    if "." in text:
        text = text.split(".")[0]
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(6) if len(digits) <= 6 else digits


def load_index_list(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), f"missing_index_list={path}"
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna(""), ""
    except Exception as exc:  # pragma: no cover - defensive file handling.
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def discover_total_return_index_candidates(index_list: pd.DataFrame) -> pd.DataFrame:
    if index_list.empty:
        return pd.DataFrame(
            columns=[
                "index_code",
                "index_short_name",
                "index_full_name",
                "asset_class",
                "index_category",
                "publish_date",
                "candidate_family",
                "acceptance_status",
                "acceptance_reason",
            ]
        )
    short_col = "指数简称"
    full_col = "指数全称"
    asset_col = "资产类别"
    category_col = "指数类别"
    publish_col = "发布时间"
    rows = []
    for _, row in index_list.iterrows():
        short_name = str(row.get(short_col, ""))
        full_name = str(row.get(full_col, ""))
        asset_class = str(row.get(asset_col, ""))
        combined = f"{short_name} {full_name}".lower()
        keyword_hit = any(keyword.lower() in combined for keyword in TOTAL_RETURN_KEYWORDS)
        if not keyword_hit:
            continue
        code = normalize_index_code(row.get("index_code", ""))
        equity = asset_class == "股票"
        rows.append(
            {
                "index_code": code,
                "index_short_name": short_name,
                "index_full_name": full_name,
                "asset_class": asset_class,
                "index_category": str(row.get(category_col, "")),
                "publish_date": str(row.get(publish_col, "")),
                "candidate_family": "csindex_total_return_keyword",
                "acceptance_status": "candidate" if equity and code else "rejected",
                "acceptance_reason": "equity_total_return_keyword_match" if equity and code else "not_equity_or_missing_code",
            }
        )
    return pd.DataFrame(rows)


def route_table(readiness: pd.DataFrame, candidates: pd.DataFrame, config: SourceAcquirerConfig, target_exists: bool) -> pd.DataFrame:
    equity_candidates = candidates.loc[candidates["acceptance_status"] == "candidate"] if not candidates.empty else pd.DataFrame()
    routes = [
        {
            "route_id": "existing_contract_csv",
            "provider": "local",
            "route_type": "manual_or_vendor_csv",
            "ready": target_exists,
            "will_execute": False,
            "target": config.target_source_path.as_posix(),
            "status": "ready_for_v3_53" if target_exists else "blocked_missing_file",
            "blocker": "" if target_exists else "place compliant CSV at target path",
            "acceptance_note": "validated by V3.53 importer, not by this router",
        },
        {
            "route_id": "csindex_total_return_code",
            "provider": "akshare_csindex",
            "route_type": "auto_fetch_if_equity_total_return_code_discovered",
            "ready": provider_ready(readiness, "akshare") and not equity_candidates.empty,
            "will_execute": bool(config.execute and provider_ready(readiness, "akshare") and not equity_candidates.empty),
            "target": config.target_source_path.as_posix(),
            "status": "ready" if provider_ready(readiness, "akshare") and not equity_candidates.empty else "blocked_no_equity_total_return_code",
            "blocker": "" if not equity_candidates.empty else "local CSIndex list has no equity full-return/total-return code",
            "acceptance_note": "accepted only when index name explicitly contains full-return/total-return keyword",
        },
        {
            "route_id": "joinquant_adjusted_market_proxy",
            "provider": "joinquant",
            "route_type": "jqdatasdk_get_price_adjusted_proxy",
            "ready": provider_ready(readiness, "joinquant") and config.allow_adjusted_proxy,
            "will_execute": bool(config.execute and provider_ready(readiness, "joinquant") and config.allow_adjusted_proxy),
            "target": config.target_source_path.as_posix(),
            "status": "ready" if provider_ready(readiness, "joinquant") and config.allow_adjusted_proxy else "blocked",
            "blocker": "" if provider_ready(readiness, "joinquant") else "JoinQuant SDK credentials not ready",
            "acceptance_note": "requires explicit adjusted-proxy approval; not treated as certified total-return by default",
        },
        {
            "route_id": "tushare_index_total_return",
            "provider": "tushare",
            "route_type": "future_endpoint_or_manual_vendor_file",
            "ready": False,
            "will_execute": False,
            "target": config.target_source_path.as_posix(),
            "status": "blocked_no_known_permissioned_endpoint",
            "blocker": "current permission profile has not shown a total-return index endpoint",
            "acceptance_note": "do not substitute index_daily price-only close",
        },
    ]
    order = {route_id: pos for pos, route_id in enumerate(config.route_priority)}
    out = pd.DataFrame(routes)
    out["priority"] = out["route_id"].map(order).fillna(999).astype(int)
    return out.sort_values(["priority", "route_id"]).reset_index(drop=True)


def _date_text(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


def normalize_csindex_total_return_daily(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "指数代码": "index_code",
        "指数简称": "index_short_name",
        "收盘": "close",
        "收盘点位": "close",
        "close": "close",
        "date": "date",
    }
    out = raw.rename(columns=rename).copy()
    if "date" not in out.columns:
        raise ValueError("CSIndex total-return candidate missing date column")
    if "close" not in out.columns:
        raise ValueError("CSIndex total-return candidate missing close column")
    close = pd.to_numeric(out["close"], errors="coerce")
    dates = _date_text(out["date"])
    clean = pd.DataFrame(
        {
            "date": dates,
            "asset_or_index": code,
            "total_return_index_or_adjusted_close": close,
            "available_date": dates,
            "data_source": f"akshare.stock_zh_index_hist_csindex.total_return_candidate.{code}",
            "source_vintage": f"fetched_at_{now_text()}",
        }
    )
    clean = clean.dropna(subset=["date", "total_return_index_or_adjusted_close"]).sort_values("date")
    return clean


def execute_csindex_total_return_fetch(candidates: pd.DataFrame, config: SourceAcquirerConfig) -> tuple[bool, pd.DataFrame, str]:
    equity_candidates = candidates.loc[candidates["acceptance_status"] == "candidate"] if not candidates.empty else pd.DataFrame()
    if equity_candidates.empty:
        return False, pd.DataFrame(), "no_equity_total_return_candidate"
    code = str(equity_candidates.iloc[0]["index_code"])
    if config.target_source_path.exists() and not config.allow_overwrite_target:
        return False, pd.DataFrame(), "target_exists_and_overwrite_disabled"
    import akshare as ak  # type: ignore

    raw = ak.stock_zh_index_hist_csindex(symbol=code, start_date=config.start_date, end_date=config.end_date)
    clean = normalize_csindex_total_return_daily(raw, code)
    config.target_source_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.target_source_path, index=False, encoding="utf-8-sig")
    return True, clean, f"fetched_csindex_total_return_candidate={code}"


def build_source_readiness(routes: pd.DataFrame, acquisition_log: pd.DataFrame, target_exists_after: bool) -> pd.DataFrame:
    executable_ready = bool(routes["ready"].astype(bool).any())
    wrote_target = bool((acquisition_log["status"] == "acquired").any()) if not acquisition_log.empty else False
    return pd.DataFrame(
        [
            {
                "check": "acquisition_route_inventory_built",
                "status": "pass",
                "detail": f"routes={len(routes)}",
            },
            {
                "check": "at_least_one_route_ready",
                "status": "pass" if executable_ready else "blocked",
                "detail": ",".join(routes.loc[routes["ready"].astype(bool), "route_id"].astype(str)),
            },
            {
                "check": "target_market_source_exists_after_run",
                "status": "pass" if target_exists_after else "blocked",
                "detail": "target file ready for V3.53" if target_exists_after else "target source file not present",
            },
            {
                "check": "source_written_by_v3_54",
                "status": "pass" if wrote_target else "not_applicable",
                "detail": "acquired" if wrote_target else "no route executed or route blocked",
            },
            {
                "check": "performance_validation_allowed_now",
                "status": "blocked",
                "detail": "V3.54 only acquires source; V3.53 must validate and V3.49 must run signal validation.",
            },
        ]
    )


def manual_instructions(config: SourceAcquirerConfig) -> str:
    return "\n".join(
        [
            "# Manual MARKET Total-Return Source Instructions V3.54",
            "",
            "If automated routes are blocked, provide a CSV at:",
            "",
            f"`{config.target_source_path.as_posix()}`",
            "",
            "Required columns:",
            "",
            "`date, asset_or_index, total_return_index_or_adjusted_close, available_date, data_source, source_vintage`",
            "",
            "Rules:",
            "",
            "- Use a certified total-return index or explicitly approved adjusted market proxy.",
            "- Do not use strategy NAV, benchmark NAV, price-only index close, raw close, or unadjusted close.",
            "- `available_date` must be the date the observation is known to the research process.",
            "- After placing the file, rerun V3.53 to validate and create forward labels.",
            "",
        ]
    )


def build_no_source_guard(target_exists_after: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "market_total_return_source_csv",
                "produced": target_exists_after,
                "blocked": not target_exists_after,
                "reason": "target source ready" if target_exists_after else "no certified source route acquired",
            },
            {
                "result_type": "market_forward_labels",
                "produced": False,
                "blocked": True,
                "reason": "V3.53 importer must validate source and create labels.",
            },
            {
                "result_type": "state_stratified_validation_or_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.54 does not run signal validation or portfolio backtest.",
            },
        ]
    )


def build_acceptance_checks(
    routes: pd.DataFrame,
    candidates: pd.DataFrame,
    readiness: pd.DataFrame,
    guard: pd.DataFrame,
    config: SourceAcquirerConfig,
) -> pd.DataFrame:
    price_only_promoted = False
    if config.target_source_path.exists():
        try:
            cols = set(pd.read_csv(config.target_source_path, encoding="utf-8-sig", nrows=1).columns)
            price_only_promoted = "close" in cols and "total_return_index_or_adjusted_close" not in cols
        except Exception:
            price_only_promoted = False
    return pd.DataFrame(
        [
            {
                "check": "routes_have_explicit_acceptance_notes",
                "status": "pass" if routes["acceptance_note"].astype(str).str.len().gt(0).all() else "fail",
                "detail": f"routes={len(routes)}",
            },
            {
                "check": "csindex_equity_total_return_discovery_recorded",
                "status": "pass",
                "detail": f"candidate_rows={len(candidates)}",
            },
            {
                "check": "price_only_source_not_promoted",
                "status": "pass" if not price_only_promoted else "fail",
                "detail": FORBIDDEN_ACCEPTANCE_NOTE,
            },
            {
                "check": "labels_and_backtest_not_run_here",
                "status": "pass"
                if not bool(guard.loc[guard["result_type"].isin(["market_forward_labels", "state_stratified_validation_or_backtest"]), "produced"].any())
                else "fail",
                "detail": "source acquisition only",
            },
            {
                "check": "readiness_report_blocks_performance_validation",
                "status": "pass"
                if str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
                else "fail",
                "detail": "validation remains delegated to V3.53/V3.49",
            },
        ]
    )


def build_report(
    providers: pd.DataFrame,
    candidates: pd.DataFrame,
    routes: pd.DataFrame,
    acquisition_log: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: SourceAcquirerConfig,
) -> str:
    lines = [
        "# V3.54 Market Total-Return Source Acquirer",
        "",
        "## Decision",
        "",
        "- V3.54 creates the governed acquisition router for the MARKET total-return source required by V3.53.",
        "- It may write `market_total_return_index.csv` only from a certified total-return route.",
        "- It does not run labels, IC, returns, NAV, drawdown, Sharpe, portfolio backtest, or model promotion.",
        "",
        "## Current Result",
        "",
        f"- Execute mode: `{config.execute}`",
        f"- Target source path: `{config.target_source_path.as_posix()}`",
        f"- Target exists after run: `{config.target_source_path.exists()}`",
        "",
        "## Provider Readiness",
        "",
        "| provider | check | ready | detail |",
        "|---|---|---|---|",
    ]
    for row in providers.itertuples(index=False):
        lines.append(f"| `{row.provider}` | `{row.check}` | `{row.ready}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Total-Return Candidate Discovery",
            "",
            f"- Candidate rows: `{len(candidates)}`",
            "",
            "| index_code | short_name | full_name | asset_class | status | reason |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in candidates.head(20).itertuples(index=False):
        lines.append(
            f"| `{row.index_code}` | `{row.index_short_name}` | `{row.index_full_name}` | `{row.asset_class}` | `{row.acceptance_status}` | {row.acceptance_reason} |"
        )
    if candidates.empty:
        lines.append("|  |  |  |  |  | no total-return keyword candidates found in local index list |")
    lines.extend(
        [
            "",
            "## Acquisition Routes",
            "",
            "| priority | route_id | provider | ready | will_execute | status | blocker |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in routes.itertuples(index=False):
        lines.append(f"| {int(row.priority)} | `{row.route_id}` | `{row.provider}` | `{row.ready}` | `{row.will_execute}` | `{row.status}` | {row.blocker} |")
    lines.extend(
        [
            "",
            "## Acquisition Log",
            "",
            "| route_id | status | rows | detail |",
            "|---|---|---:|---|",
        ]
    )
    for row in acquisition_log.itertuples(index=False):
        lines.append(f"| `{row.route_id}` | `{row.status}` | {int(row.rows)} | {row.detail} |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- If the target source exists, rerun V3.53 to validate and build labels.",
            "- If automated routes are blocked, follow `manual_total_return_source_instructions.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame, routes: pd.DataFrame, target_path: Path) -> str:
    ready_routes = ",".join(routes.loc[routes["ready"].astype(bool), "route_id"].astype(str))
    return "\n".join(
        [
            "# A-share Market Total-Return Source Acquirer V3.54",
            "",
            "## Dataset Decision",
            "",
            "- Acquisition router ready: `true`",
            f"- Ready routes: `{ready_routes}`",
            f"- Target source exists: `{target_path.exists()}`",
            "- Performance validation remains blocked here; use V3.53 then V3.49.",
            "",
        ]
    )
