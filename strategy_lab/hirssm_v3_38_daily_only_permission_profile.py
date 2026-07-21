#!/usr/bin/env python
"""HIRSSM V3.38 daily-only permission profile.

This data-steward task continues acquisition under the current Tushare
permission profile where only the raw stock daily endpoint is reliable. It does
not call known blocked endpoints and does not run factor validation or portfolio
backtests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260528_v3_38_daily_only_permission_profile"
VERSION = "V3.38"
BASELINE = "V3.37 Credential Bootstrap"
DEFAULT_CONFIG = ROOT / "configs" / "daily_only_permission_v3_38.json"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_38" / "daily_only_permission_profile"
CATALOG_PATH = ROOT / "data_catalog" / "a_share_daily_only_permission_v3_38.md"
LOCAL_CREDENTIALS = ROOT / "configs" / "data_credentials.json"
REQUIRED_DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", dtype=str, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def mask_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    placeholders = ["PASTE_", "TOKEN_HERE", "USERNAME_HERE", "PASSWORD_HERE"]
    return not any(item in text.upper() for item in placeholders)


def load_credentials_config() -> dict[str, Any]:
    if not LOCAL_CREDENTIALS.exists():
        return {}
    return read_json(LOCAL_CREDENTIALS)


def credential_value(provider: str, field: str) -> str | None:
    config = load_credentials_config()
    section = config.get(provider, {}) if isinstance(config, dict) else {}
    if not isinstance(section, dict):
        section = {}
    if provider == "tushare" and field == "token":
        return os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN") or section.get("token")
    return None


def credential_readiness() -> pd.DataFrame:
    rows = [
        {
            "provider": "tushare",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("tushare") is not None,
            "detail": "python package tushare",
        },
        {
            "provider": "tushare",
            "check": "token_present",
            "ready": mask_bool(credential_value("tushare", "token")),
            "detail": "env var or configs/data_credentials.json token",
        },
    ]
    return pd.DataFrame(rows)


def provider_ready(readiness: pd.DataFrame) -> bool:
    return bool(readiness["ready"].all()) if not readiness.empty else False


def build_permission_profile(config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "provider": config.get("provider", "tushare"),
            "permission_profile": config.get("permission_profile", "daily_only"),
            "endpoint": config.get("daily_endpoint", "daily"),
            "status": "allowed",
            "research_use": "raw_price_volume_only",
            "limitation": "non-adjusted OHLCV; no valuation, no adjusted return, no historical universe lifecycle",
        }
    ]
    for item in config.get("blocked_endpoints", []):
        rows.append(
            {
                "provider": config.get("provider", "tushare"),
                "permission_profile": config.get("permission_profile", "daily_only"),
                "endpoint": item.get("endpoint", ""),
                "status": item.get("current_status", "permission_blocked"),
                "research_use": item.get("needed_for", ""),
                "limitation": item.get("fallback", ""),
            }
        )
    return pd.DataFrame(rows)


def build_blocked_endpoint_registry(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for priority, item in enumerate(config.get("blocked_endpoints", []), start=1):
        endpoint = item.get("endpoint", "")
        min_action = "open_tushare_permission_or_find_alternative_source"
        if endpoint == "adj_factor":
            min_action = "open_tushare_adj_factor_or_obtain_adjustment_factor_history"
        elif endpoint == "daily_basic":
            min_action = "open_tushare_daily_basic_or_obtain_historical_valuation_market_cap"
        elif endpoint == "index_weight":
            min_action = "open_tushare_index_weight_or_obtain_historical_index_constituents_weights"
        elif endpoint == "stock_basic":
            min_action = "open_tushare_stock_basic_or_obtain_all_status_security_master"
        elif endpoint == "index_dailybasic":
            min_action = "open_tushare_index_dailybasic_or_obtain_historical_index_valuation"
        rows.append(
            {
                "priority": priority,
                "provider": config.get("provider", "tushare"),
                "endpoint": endpoint,
                "status": item.get("current_status", "permission_blocked"),
                "needed_for": item.get("needed_for", ""),
                "fallback_now": item.get("fallback", ""),
                "manual_action_needed": min_action,
            }
        )
    return pd.DataFrame(rows)


def normalize_trade_date(date_text: str) -> str:
    return str(date_text).replace("-", "")[:8]


def business_day_candidates(start_date: str, end_date: str) -> list[str]:
    start = pd.to_datetime(normalize_trade_date(start_date), format="%Y%m%d")
    end = pd.to_datetime(normalize_trade_date(end_date), format="%Y%m%d")
    dates = pd.bdate_range(start=start, end=end)
    return [item.strftime("%Y%m%d") for item in dates]


def output_path_for_trade_date(config: dict[str, Any], trade_date: str) -> Path:
    return ROOT / str(config["output_root"]) / "daily" / f"trade_date={trade_date}.csv"


def build_daily_harvest_plan(config: dict[str, Any], readiness: pd.DataFrame) -> pd.DataFrame:
    execute = bool(config.get("execute", False))
    allow_network = bool(config.get("allow_network", False))
    ready = provider_ready(readiness)
    max_new = int(config.get("max_new_trade_dates", 30))
    reuse_existing = bool(config.get("reuse_existing_outputs", True))
    candidates = business_day_candidates(config["start_date"], config["end_date"])
    rows = []
    new_count = 0
    for trade_date in candidates:
        path = output_path_for_trade_date(config, trade_date)
        exists = path.exists()
        if exists and reuse_existing:
            planned_status = "cached"
            can_execute = False
            blocked_reason = ""
        elif new_count < max_new:
            can_execute = bool(execute and allow_network and ready)
            planned_status = "ready_to_execute" if can_execute else "blocked"
            if not execute:
                blocked_reason = "execute_false"
            elif not allow_network:
                blocked_reason = "allow_network_false"
            elif not ready:
                blocked_reason = "tushare_sdk_or_token_not_ready"
            else:
                blocked_reason = ""
            new_count += 1
        else:
            planned_status = "queued_after_max_new_trade_dates"
            can_execute = False
            blocked_reason = f"max_new_trade_dates={max_new}"
        rows.append(
            {
                "provider": config.get("provider", "tushare"),
                "endpoint": config.get("daily_endpoint", "daily"),
                "trade_date": trade_date,
                "planned_output": rel(path),
                "output_exists": exists,
                "execute_requested": execute,
                "allow_network": allow_network,
                "provider_ready": ready,
                "can_execute": can_execute,
                "planned_status": planned_status,
                "blocked_reason": blocked_reason,
            }
        )
    return pd.DataFrame(rows)


def fetch_tushare_daily_for_trade_date(trade_date: str) -> pd.DataFrame:
    token = credential_value("tushare", "token")
    import tushare as ts  # type: ignore

    pro = ts.pro_api(token)
    data = pro.daily(trade_date=trade_date)
    out = data.copy()
    out["asset"] = out["ts_code"] if "ts_code" in out.columns else ""
    out["date"] = trade_date
    out["available_date"] = pd.to_datetime(trade_date, format="%Y%m%d").strftime("%Y-%m-%d")
    out["data_source"] = "tushare.daily"
    out["fetched_at"] = now_text()
    out["price_adjustment"] = "none_raw"
    return out


def execute_daily_harvest(config: dict[str, Any], plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_rows = []
    registry_rows = []
    sleep_seconds = float(config.get("sleep_seconds_between_calls", 1.35))
    for row in plan.itertuples(index=False):
        output_path = ROOT / str(row.planned_output)
        started = now_text()
        if row.planned_status == "cached":
            existing = read_csv_if_exists(output_path)
            row_count = int(existing.shape[0])
            status = "cached"
            error = ""
        elif not bool(row.can_execute):
            row_count = 0
            status = row.planned_status
            error = row.blocked_reason
        else:
            try:
                data = fetch_tushare_daily_for_trade_date(str(row.trade_date))
                write_csv(data, output_path)
                row_count = int(data.shape[0])
                status = "acquired" if row_count > 0 else "empty"
                error = ""
                time.sleep(sleep_seconds)
            except Exception as exc:  # noqa: BLE001
                row_count = 0
                status = "error"
                error = repr(exc)
        log_rows.append(
            {
                "trade_date": row.trade_date,
                "endpoint": row.endpoint,
                "status": status,
                "rows": row_count,
                "output_path": row.planned_output,
                "started_at": started,
                "ended_at": now_text(),
                "error": error,
            }
        )
        if status in {"acquired", "cached", "empty"}:
            registry_status = "acquired" if status in {"acquired", "cached"} and row_count > 0 else "empty"
            research_use = "raw_price_volume_research_only" if registry_status == "acquired" else "non_trading_or_empty"
        else:
            registry_status = "not_acquired"
            research_use = "blocked"
        registry_rows.append(
            {
                "dataset_id": "stock_daily_raw_by_trade_date",
                "provider": row.provider,
                "endpoint": row.endpoint,
                "trade_date": row.trade_date,
                "status": registry_status,
                "path": row.planned_output,
                "rows": row_count,
                "research_use": research_use,
                "reason": "raw daily only; no adjusted-return or valuation use" if registry_status == "acquired" else error,
            }
        )
    return pd.DataFrame(log_rows), pd.DataFrame(registry_rows)


def quality_for_daily_file(path: Path, trade_date: str, status: str) -> dict[str, Any]:
    if not path.exists() or status != "acquired":
        return {
            "trade_date": trade_date,
            "status": status,
            "rows": 0,
            "missing_required_columns": ",".join(REQUIRED_DAILY_COLUMNS),
            "duplicate_asset_date_count": 0,
            "null_close_rate": 1.0,
            "null_volume_rate": 1.0,
            "min_close": "",
            "max_close": "",
            "quality_flag": "not_acquired",
        }
    data = read_csv_if_exists(path)
    missing = [col for col in REQUIRED_DAILY_COLUMNS if col not in data.columns]
    key_cols = [col for col in ["ts_code", "trade_date"] if col in data.columns]
    duplicate_count = int(data.duplicated(key_cols).sum()) if key_cols else 0
    close = pd.to_numeric(data["close"], errors="coerce") if "close" in data.columns else pd.Series(dtype=float)
    volume_col = "vol" if "vol" in data.columns else "volume" if "volume" in data.columns else ""
    volume = pd.to_numeric(data[volume_col], errors="coerce") if volume_col else pd.Series(dtype=float)
    null_close_rate = float(close.isna().mean()) if len(data) else 1.0
    null_volume_rate = float(volume.isna().mean()) if len(data) and volume_col else 1.0
    quality_flag = "pass" if not missing and duplicate_count == 0 and null_close_rate == 0.0 else "review"
    return {
        "trade_date": trade_date,
        "status": status,
        "rows": int(data.shape[0]),
        "missing_required_columns": ",".join(missing),
        "duplicate_asset_date_count": duplicate_count,
        "null_close_rate": null_close_rate,
        "null_volume_rate": null_volume_rate,
        "min_close": float(close.min()) if len(close) and close.notna().any() else "",
        "max_close": float(close.max()) if len(close) and close.notna().any() else "",
        "quality_flag": quality_flag,
    }


def build_quality_report(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples(index=False):
        rows.append(quality_for_daily_file(ROOT / str(item.path), str(item.trade_date), str(item.status)))
    return pd.DataFrame(rows)


def build_model_use_constraints() -> pd.DataFrame:
    rows = [
        {
            "research_area": "price_volume_factor",
            "allowed_now": True,
            "constraint": "raw close only; use forward raw close returns with explicit warning",
        },
        {
            "research_area": "adjusted_return_factor",
            "allowed_now": False,
            "constraint": "requires adj_factor or trusted adjusted price history",
        },
        {
            "research_area": "valuation_factor",
            "allowed_now": False,
            "constraint": "requires daily_basic or historical valuation/market-cap source",
        },
        {
            "research_area": "dividend_total_return",
            "allowed_now": False,
            "constraint": "requires dividends and adjustment history",
        },
        {
            "research_area": "index_constituent_pit",
            "allowed_now": False,
            "constraint": "requires historical index constituents and weights",
        },
        {
            "research_area": "industry_rotation_pit",
            "allowed_now": False,
            "constraint": "requires historical industry classification or industry index series",
        },
    ]
    return pd.DataFrame(rows)


def build_agent_report(
    config: dict[str, Any],
    readiness: pd.DataFrame,
    profile: pd.DataFrame,
    log: pd.DataFrame,
    registry: pd.DataFrame,
    blocked: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    acquired_dates = int((registry["status"] == "acquired").sum()) if not registry.empty else 0
    error_dates = int((log["status"] == "error").sum()) if not log.empty else 0
    queued_dates = int((log["status"] == "queued_after_max_new_trade_dates").sum()) if not log.empty else 0
    quality_pass = int((quality["quality_flag"] == "pass").sum()) if not quality.empty else 0
    return "\n".join(
        [
            "# V3.38 Daily-Only Permission Profile",
            "",
            f"Generated at: `{now_text()}`",
            "",
            "## Decision",
            "",
            "- Current acquisition mode is `daily-only`.",
            "- Only `tushare.daily` is allowed to be called by this task.",
            "- Known blocked endpoints are recorded for manual permission/data-source follow-up.",
            "- No factor validation, portfolio backtest, or model promotion was run.",
            "",
            "## Current Permission Result",
            "",
            f"- Tushare SDK/token ready: `{provider_ready(readiness)}`",
            f"- Allowed endpoint count: `{int((profile['status'] == 'allowed').sum())}`",
            f"- Permission-blocked endpoint count: `{int((profile['status'] == 'permission_blocked').sum())}`",
            "",
            "## Daily Harvest",
            "",
            f"- Execute flag: `{bool(config.get('execute', False))}`",
            f"- Network allowed flag: `{bool(config.get('allow_network', False))}`",
            f"- Max new trade-date calls this run: `{int(config.get('max_new_trade_dates', 30))}`",
            f"- Acquired/cached dates with rows: `{acquired_dates}`",
            f"- Error dates: `{error_dates}`",
            f"- Queued after cap: `{queued_dates}`",
            f"- Quality pass rows: `{quality_pass}`",
            "",
            "## Manual Follow-Up Interfaces",
            "",
            "| Priority | Endpoint | Needed For | Manual Action |",
            "|---:|---|---|---|",
            *[
                f"| {row.priority} | `{row.endpoint}` | {row.needed_for} | {row.manual_action_needed} |"
                for row in blocked.itertuples(index=False)
            ],
            "",
            "## Research Boundary",
            "",
            "This dataset is raw-price only. It can support data plumbing and limited price/volume research, but it is not sufficient for adjusted-return backtests, valuation factors, dividend total return, PIT index membership, or industry rotation with historical constituents.",
        ]
    ) + "\n"


def build_manual_data_interface_queue_md(blocked: pd.DataFrame) -> str:
    lines = [
        "# Manual Data Interface Queue V3.38",
        "",
        f"Generated at: `{now_text()}`",
        "",
        "These interfaces are not called by the daily-only pipeline. Open Tushare permission or provide an alternative point-in-time data source before using the related research area.",
        "",
        "| Priority | Provider | Endpoint | Needed For | Current Fallback | Manual Action |",
        "|---:|---|---|---|---|---|",
    ]
    for row in blocked.itertuples(index=False):
        lines.append(
            f"| {row.priority} | `{row.provider}` | `{row.endpoint}` | {row.needed_for} | {row.fallback_now} | {row.manual_action_needed} |"
        )
    lines.extend(
        [
            "",
            "## Minimum Priority",
            "",
            "1. `adj_factor`: needed before any adjusted-return or realistic stock backtest.",
            "2. `stock_basic`: needed for all-status security master and lifecycle filtering.",
            "3. `daily_basic`: needed for valuation, turnover, market cap, PE/PB, and size style factors.",
            "4. `index_weight`: needed for PIT index constituent and benchmark replication research.",
            "5. `index_dailybasic`: needed for index valuation repair and index valuation timing.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_catalog(
    config: dict[str, Any],
    profile: pd.DataFrame,
    registry: pd.DataFrame,
    constraints: pd.DataFrame,
) -> str:
    acquired_dates = int((registry["status"] == "acquired").sum()) if not registry.empty else 0
    rows = [
        "# A-share Daily-Only Permission Data V3.38",
        "",
        f"Updated: `{now_text()}`",
        "",
        "## Scope",
        "",
        "- Provider: `tushare`",
        "- Allowed endpoint: `daily`",
        "- Data type: stock non-adjusted daily OHLCV",
        f"- Configured date range: `{config['start_date']}` to `{config['end_date']}`",
        f"- Acquired/cached dates with rows: `{acquired_dates}`",
        "",
        "## Permission Profile",
        "",
        "| Endpoint | Status | Research Use | Limitation |",
        "|---|---|---|---|",
    ]
    for row in profile.itertuples(index=False):
        rows.append(f"| `{row.endpoint}` | `{row.status}` | {row.research_use} | {row.limitation} |")
    rows.extend(
        [
            "",
            "## Model Use Constraints",
            "",
            "| Research Area | Allowed Now | Constraint |",
            "|---|---:|---|",
        ]
    )
    for row in constraints.itertuples(index=False):
        rows.append(f"| `{row.research_area}` | `{row.allowed_now}` | {row.constraint} |")
    rows.extend(
        [
            "",
            "## Governance",
            "",
            "- Do not use this dataset as adjusted-return data.",
            "- Do not infer valuation, market cap, dividend yield, or index membership from this dataset.",
            "- Before full backtests, upgrade data layer or explicitly constrain the model to raw-price/volume research.",
        ]
    )
    return "\n".join(rows) + "\n"


def build_self_check(
    artifacts: dict[str, Path],
    config: dict[str, Any],
    plan: pd.DataFrame,
    log: pd.DataFrame,
    profile: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, path in artifacts.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    called_endpoints = set(log["endpoint"].dropna().astype(str).unique()) if not log.empty else set()
    rows.append(
        {
            "check": "only_daily_endpoint_called",
            "status": "pass" if called_endpoints.issubset({"daily"}) else "fail",
            "detail": ",".join(sorted(called_endpoints)),
        }
    )
    rows.append(
        {
            "check": "blocked_endpoints_recorded",
            "status": "pass" if int((profile["status"] == "permission_blocked").sum()) >= 1 else "fail",
            "detail": str(int((profile["status"] == "permission_blocked").sum())),
        }
    )
    rows.append(
        {
            "check": "execute_requires_network_flag",
            "status": "pass" if (not bool(config.get("execute", False)) or bool(config.get("allow_network", False))) else "fail",
            "detail": f"execute={config.get('execute', False)}, allow_network={config.get('allow_network', False)}",
        }
    )
    ready_rows = int((plan["planned_status"] == "ready_to_execute").sum()) if not plan.empty else 0
    executed_rows = int(log["status"].isin(["acquired", "empty", "error"]).sum()) if not log.empty else 0
    rows.append(
        {
            "check": "max_new_trade_dates_respected",
            "status": "pass" if executed_rows <= int(config.get("max_new_trade_dates", 30)) else "fail",
            "detail": f"ready={ready_rows}, executed={executed_rows}",
        }
    )
    rows.append(
        {
            "check": "raw_only_quality_boundary",
            "status": "pass",
            "detail": "no adjusted-return, valuation, or index-constituent approval",
        }
    )
    rows.append(
        {
            "check": "quality_report_generated",
            "status": "pass" if not quality.empty else "fail",
            "detail": str(len(quality)),
        }
    )
    rows.append(
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "data acquisition only",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="Enable daily-only acquisition when config allow_network is also true.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    if args.execute:
        config["execute"] = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness = credential_readiness()
    profile = build_permission_profile(config)
    blocked = build_blocked_endpoint_registry(config)
    plan = build_daily_harvest_plan(config, readiness)
    log, registry = execute_daily_harvest(config, plan)
    quality = build_quality_report(registry)
    constraints = build_model_use_constraints()

    artifacts = {
        "credential_readiness": OUTPUT_DIR / "credential_readiness.csv",
        "permission_profile": OUTPUT_DIR / "permission_profile.csv",
        "blocked_endpoint_registry": OUTPUT_DIR / "blocked_endpoint_registry.csv",
        "manual_data_interface_queue": OUTPUT_DIR / "manual_data_interface_queue.csv",
        "manual_data_interface_queue_md": OUTPUT_DIR / "manual_data_interface_queue.md",
        "daily_harvest_plan": OUTPUT_DIR / "daily_harvest_plan.csv",
        "daily_harvest_log": OUTPUT_DIR / "daily_harvest_log.csv",
        "daily_dataset_registry": OUTPUT_DIR / "daily_dataset_registry.csv",
        "daily_quality_report": OUTPUT_DIR / "daily_quality_report.csv",
        "model_use_constraints": OUTPUT_DIR / "model_use_constraints.csv",
        "agent_report": OUTPUT_DIR / "agent_report.md",
        "catalog_update": CATALOG_PATH,
        "changed_files": OUTPUT_DIR / "changed_files.txt",
    }
    write_csv(readiness, artifacts["credential_readiness"])
    write_csv(profile, artifacts["permission_profile"])
    write_csv(blocked, artifacts["blocked_endpoint_registry"])
    write_csv(blocked, artifacts["manual_data_interface_queue"])
    write_text(build_manual_data_interface_queue_md(blocked), artifacts["manual_data_interface_queue_md"])
    write_csv(plan, artifacts["daily_harvest_plan"])
    write_csv(log, artifacts["daily_harvest_log"])
    write_csv(registry, artifacts["daily_dataset_registry"])
    write_csv(quality, artifacts["daily_quality_report"])
    write_csv(constraints, artifacts["model_use_constraints"])
    write_text(build_agent_report(config, readiness, profile, log, registry, blocked, quality), artifacts["agent_report"])
    write_text(build_catalog(config, profile, registry, constraints), artifacts["catalog_update"])
    write_text("\n".join(rel(path) for path in artifacts.values()) + "\n", artifacts["changed_files"])

    self_check_path = OUTPUT_DIR / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, config, plan, log, profile, quality)
    write_csv(self_check, self_check_path)

    manifest = {
        "run_id": "20260528_v3_38_daily_only_permission_profile_run_001",
        "task_id": TASK_ID,
        "version": VERSION,
        "baseline": BASELINE,
        "agent": "data_steward",
        "command": "python -X utf8 strategy_lab/hirssm_v3_38_daily_only_permission_profile.py",
        "execute": bool(config.get("execute", False)),
        "allow_network": bool(config.get("allow_network", False)),
        "allowed_endpoint": "daily",
        "blocked_endpoint_count": int((profile["status"] == "permission_blocked").sum()),
        "acquired_or_cached_dates": int((registry["status"] == "acquired").sum()) if not registry.empty else 0,
        "error_dates": int((log["status"] == "error").sum()) if not log.empty else 0,
        "self_check_pass": bool((self_check["status"] == "pass").all()),
        "model_decision": "no_model_promotion_daily_only_data_layer",
        "outputs": [rel(path) for path in artifacts.values()],
    }
    manifest_path = OUTPUT_DIR / "agent_run_manifest.json"
    write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "self_check_pass": manifest["self_check_pass"],
                "metrics": {
                    "execute": manifest["execute"],
                    "allow_network": manifest["allow_network"],
                    "allowed_endpoint": manifest["allowed_endpoint"],
                    "blocked_endpoint_count": manifest["blocked_endpoint_count"],
                    "acquired_or_cached_dates": manifest["acquired_or_cached_dates"],
                    "error_dates": manifest["error_dates"],
                    "model_decision": manifest["model_decision"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
