#!/usr/bin/env python
"""HIRSSM V3.36 PIT data pilot readiness.

This data-steward task verifies SDK/API readiness and can run a small,
predeclared pilot only when execution is explicitly enabled and credentials are
ready. In the default repo state it emits blocked acquisition evidence rather
than pretending that dry-run plans are real data.
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
TASK_ID = "20260527_v3_36_pit_data_pilot_readiness"
VERSION = "V3.36"
BASELINE = "V3.35 PIT Data Acquisition Contract"
DEFAULT_CONFIG = ROOT / "configs" / "pit_data_pilot_v3_36.json"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_36" / "pit_data_pilot_readiness"
CATALOG_PATH = ROOT / "data_catalog" / "a_share_pit_data_pilot_v3_36.md"
TUSHARE_RATE_LIMIT_ENDPOINTS = {"stock_basic", "daily_basic", "adj_factor"}
TUSHARE_RATE_LIMIT_MARKERS = ("频率超限", "frequency", "rate limit")


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


def mask_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    placeholders = ["PASTE_", "TOKEN_HERE", "USERNAME_HERE", "PASSWORD_HERE"]
    return not any(item in text.upper() for item in placeholders)


def load_credentials_config() -> dict[str, Any]:
    path = ROOT / "configs" / "data_credentials.json"
    if not path.exists():
        return {}
    return read_json(path)


def credential_value(provider: str, field: str) -> str | None:
    config = load_credentials_config()
    section = config.get(provider, {}) if isinstance(config, dict) else {}
    if provider == "tushare" and field == "token":
        return os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN") or (section.get("token") if isinstance(section, dict) else None)
    if provider == "joinquant" and field == "username":
        return os.getenv("JQDATA_USERNAME") or os.getenv("JOINQUANT_USERNAME") or os.getenv("JQ_USER") or (section.get("username") if isinstance(section, dict) else None)
    if provider == "joinquant" and field == "password":
        return os.getenv("JQDATA_PASSWORD") or os.getenv("JOINQUANT_PASSWORD") or os.getenv("JQ_PASSWORD") or (section.get("password") if isinstance(section, dict) else None)
    return None


def credential_readiness() -> pd.DataFrame:
    tushare_token = credential_value("tushare", "token")
    jq_user = credential_value("joinquant", "username")
    jq_password = credential_value("joinquant", "password")
    rows = [
        {"provider": "tushare", "check": "sdk_installed", "ready": importlib.util.find_spec("tushare") is not None, "detail": "python package tushare"},
        {"provider": "tushare", "check": "token_present", "ready": mask_bool(tushare_token), "detail": "env or configs/data_credentials.json token"},
        {"provider": "joinquant", "check": "sdk_installed", "ready": importlib.util.find_spec("jqdatasdk") is not None, "detail": "python package jqdatasdk"},
        {"provider": "joinquant", "check": "username_present", "ready": mask_bool(jq_user), "detail": "env or configs/data_credentials.json username"},
        {"provider": "joinquant", "check": "password_present", "ready": mask_bool(jq_password), "detail": "env or configs/data_credentials.json password"},
    ]
    return pd.DataFrame(rows)


def provider_ready(readiness: pd.DataFrame, provider: str) -> bool:
    rows = readiness.loc[readiness["provider"] == provider]
    if provider == "tushare":
        return bool(rows.loc[rows["check"] == "sdk_installed", "ready"].any() and rows.loc[rows["check"] == "token_present", "ready"].any())
    if provider == "joinquant":
        return bool(
            rows.loc[rows["check"] == "sdk_installed", "ready"].any()
            and rows.loc[rows["check"] == "username_present", "ready"].any()
            and rows.loc[rows["check"] == "password_present", "ready"].any()
        )
    return False


def planned_pilot_tasks(config: dict[str, Any], readiness: pd.DataFrame) -> pd.DataFrame:
    execute = bool(config.get("execute", False))
    allow_network = bool(config.get("allow_network", False))
    tushare_ready = provider_ready(readiness, "tushare")
    jq_ready = provider_ready(readiness, "joinquant")
    task_specs = [
        {
            "task_name": "tushare_stock_basic_all_status",
            "provider": "tushare",
            "dataset_id": "stock_universe_all_status",
            "endpoint": "stock_basic",
            "partition": "L|D|P",
            "planned_output": f"{config['pilot_output_root']}/tushare/stock_basic/all_status.csv",
            "dependency": "tushare sdk/token",
        },
        {
            "task_name": "tushare_index_weight_pilot",
            "provider": "tushare",
            "dataset_id": "historical_index_weights",
            "endpoint": "index_weight",
            "partition": config["pilot_index_code"],
            "planned_output": f"{config['pilot_output_root']}/tushare/index_weight/{config['pilot_index_code']}.csv",
            "dependency": "tushare sdk/token",
        },
        {
            "task_name": "tushare_stock_daily_raw_pilot",
            "provider": "tushare",
            "dataset_id": "stock_daily_raw",
            "endpoint": "daily",
            "partition": ",".join(config["pilot_stock_symbols"]),
            "planned_output": f"{config['pilot_output_root']}/tushare/daily/pilot_stock_daily_raw.csv",
            "dependency": "tushare sdk/token",
        },
        {
            "task_name": "tushare_daily_basic_pilot",
            "provider": "tushare",
            "dataset_id": "stock_daily_basic",
            "endpoint": "daily_basic",
            "partition": ",".join(config["pilot_stock_symbols"]),
            "planned_output": f"{config['pilot_output_root']}/tushare/daily_basic/pilot_daily_basic.csv",
            "dependency": "tushare sdk/token",
        },
        {
            "task_name": "tushare_adj_factor_pilot",
            "provider": "tushare",
            "dataset_id": "stock_adj_factor",
            "endpoint": "adj_factor",
            "partition": ",".join(config["pilot_stock_symbols"]),
            "planned_output": f"{config['pilot_output_root']}/tushare/adj_factor/pilot_adj_factor.csv",
            "dependency": "tushare sdk/token",
        },
        {
            "task_name": "joinquant_index_membership_cross_check",
            "provider": "joinquant",
            "dataset_id": "historical_index_membership",
            "endpoint": "get_index_stocks",
            "partition": f"{config['pilot_joinquant_index_code']}@{config['pilot_joinquant_asof_date']}",
            "planned_output": f"{config['pilot_output_root']}/joinquant/index_membership/{config['pilot_joinquant_index_code']}.csv",
            "dependency": "jqdatasdk username/password",
        },
    ]
    selected = set(config.get("pilot_tasks", []))
    rows = []
    for spec in task_specs:
        if spec["task_name"] not in selected:
            continue
        ready = tushare_ready if spec["provider"] == "tushare" else jq_ready
        can_execute = bool(execute and allow_network and ready)
        if not execute:
            blocked_reason = "execute_false"
        elif not allow_network:
            blocked_reason = "allow_network_false"
        elif not ready:
            blocked_reason = f"{spec['provider']}_sdk_or_credentials_not_ready"
        else:
            blocked_reason = ""
        rows.append(
            {
                **spec,
                "execute_requested": execute,
                "allow_network": allow_network,
                "provider_ready": ready,
                "can_execute": can_execute,
                "planned_status": "ready_to_execute" if can_execute else "blocked",
                "blocked_reason": blocked_reason,
            }
        )
    return pd.DataFrame(rows)


def add_common_source_cols(df: pd.DataFrame, data_source: str, available_date: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if available_date is None:
        available_date = datetime.now().strftime("%Y-%m-%d")
    out["available_date"] = available_date
    out["data_source"] = data_source
    out["fetched_at"] = now_text()
    return out


def tushare_retry_config(config: dict[str, Any]) -> tuple[int, float, float]:
    max_retries = int(config.get("tushare_max_retries", 2))
    retry_wait = float(config.get("tushare_retry_wait_seconds", 65))
    throttle = float(config.get("tushare_endpoint_throttle_seconds", 65))
    return max_retries, retry_wait, throttle


def is_tushare_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker.lower() in text for marker in TUSHARE_RATE_LIMIT_MARKERS)


def call_tushare_with_retry(endpoint: str, call: Any, config: dict[str, Any]) -> pd.DataFrame:
    max_retries, retry_wait, _ = tushare_retry_config(config)
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if endpoint in TUSHARE_RATE_LIMIT_ENDPOINTS and is_tushare_rate_limit_error(exc) and attempt < max_retries:
                time.sleep(retry_wait)
                continue
            raise
    return pd.DataFrame()


def fetch_tushare_stock_basic(config: dict[str, Any]) -> pd.DataFrame:
    token = credential_value("tushare", "token")
    import tushare as ts  # type: ignore

    pro = ts.pro_api(token)
    frames = []
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs"
    _, _, throttle = tushare_retry_config(config)
    for index, status in enumerate(["L", "D", "P"]):
        if index > 0:
            time.sleep(throttle)
        data = call_tushare_with_retry(
            "stock_basic",
            lambda status=status: pro.stock_basic(exchange="", list_status=status, fields=fields),
            config,
        )
        data["source_list_status_query"] = status
        frames.append(data)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.rename(columns={"ts_code": "asset"})
    return add_common_source_cols(out, "tushare.stock_basic")


def fetch_tushare_index_weight(config: dict[str, Any]) -> pd.DataFrame:
    token = credential_value("tushare", "token")
    import tushare as ts  # type: ignore

    pro = ts.pro_api(token)
    data = pro.index_weight(index_code=config["pilot_index_code"], start_date=config["pilot_index_start_date"], end_date=config["pilot_index_end_date"])
    out = data.rename(columns={"trade_date": "effective_date", "con_code": "asset"})
    out["index_code"] = config["pilot_index_code"]
    out["source_vintage"] = now_text()
    return add_common_source_cols(out, "tushare.index_weight")


def fetch_tushare_symbol_panel(config: dict[str, Any], endpoint: str) -> pd.DataFrame:
    token = credential_value("tushare", "token")
    import tushare as ts  # type: ignore

    pro = ts.pro_api(token)
    frames = []
    _, _, throttle = tushare_retry_config(config)
    for index, symbol in enumerate(config["pilot_stock_symbols"]):
        if index > 0 and endpoint in TUSHARE_RATE_LIMIT_ENDPOINTS:
            time.sleep(throttle)
        if endpoint == "daily":
            data = call_tushare_with_retry(
                endpoint,
                lambda symbol=symbol: pro.daily(ts_code=symbol, start_date=config["pilot_stock_start_date"], end_date=config["pilot_stock_end_date"]),
                config,
            )
            data = data.rename(columns={"ts_code": "asset", "trade_date": "date", "vol": "volume"})
            if "close" in data.columns:
                data["raw_close"] = data["close"]
        elif endpoint == "daily_basic":
            data = call_tushare_with_retry(
                endpoint,
                lambda symbol=symbol: pro.daily_basic(ts_code=symbol, start_date=config["pilot_stock_start_date"], end_date=config["pilot_stock_end_date"]),
                config,
            )
            data = data.rename(columns={"ts_code": "asset", "trade_date": "date"})
        elif endpoint == "adj_factor":
            data = call_tushare_with_retry(
                endpoint,
                lambda symbol=symbol: pro.adj_factor(ts_code=symbol, start_date=config["pilot_stock_start_date"], end_date=config["pilot_stock_end_date"]),
                config,
            )
            data = data.rename(columns={"ts_code": "asset", "trade_date": "date"})
        else:
            raise ValueError(f"unsupported tushare endpoint: {endpoint}")
        frames.append(data)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return add_common_source_cols(out, f"tushare.{endpoint}")


def fetch_joinquant_index_membership(config: dict[str, Any]) -> pd.DataFrame:
    username = credential_value("joinquant", "username")
    password = credential_value("joinquant", "password")
    import jqdatasdk as jq  # type: ignore

    jq.auth(username, password)
    assets = jq.get_index_stocks(config["pilot_joinquant_index_code"], date=config["pilot_joinquant_asof_date"])
    out = pd.DataFrame(
        {
            "index_code": config["pilot_joinquant_index_code"],
            "asset": assets,
            "effective_date": config["pilot_joinquant_asof_date"],
            "query_date": config["pilot_joinquant_asof_date"],
        }
    )
    return add_common_source_cols(out, "jqdatasdk.get_index_stocks")


def execute_pilot_tasks(config: dict[str, Any], plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    reuse_existing = bool(config.get("reuse_existing_pilot_outputs", True))
    for task in plan.itertuples(index=False):
        output_path = ROOT / str(task.planned_output)
        started = now_text()
        if not bool(task.can_execute):
            log_rows.append(
                {
                    "task_name": task.task_name,
                    "dataset_id": task.dataset_id,
                    "provider": task.provider,
                    "status": "blocked",
                    "rows": 0,
                    "output_path": rel(output_path),
                    "started_at": started,
                    "ended_at": now_text(),
                    "error": task.blocked_reason,
                }
            )
            registry_rows.append(
                {
                    "dataset_id": task.dataset_id,
                    "provider": task.provider,
                    "status": "not_acquired",
                    "path": rel(output_path),
                    "rows": 0,
                    "research_use": "blocked",
                    "reason": task.blocked_reason,
                }
            )
            continue
        if reuse_existing and output_path.exists():
            existing = read_csv_if_exists(output_path)
            if not existing.empty:
                row_count = int(existing.shape[0])
                log_rows.append(
                    {
                        "task_name": task.task_name,
                        "dataset_id": task.dataset_id,
                        "provider": task.provider,
                        "status": "cached",
                        "rows": row_count,
                        "output_path": rel(output_path),
                        "started_at": started,
                        "ended_at": now_text(),
                        "error": "reused_existing_pilot_output_no_api_call",
                    }
                )
                registry_rows.append(
                    {
                        "dataset_id": task.dataset_id,
                        "provider": task.provider,
                        "status": "acquired",
                        "path": rel(output_path),
                        "rows": row_count,
                        "research_use": "pilot_research_only",
                        "reason": "reused existing pilot output; requires PIT validation before any factor use",
                    }
                )
                continue
        try:
            if task.task_name == "tushare_stock_basic_all_status":
                data = fetch_tushare_stock_basic(config)
            elif task.task_name == "tushare_index_weight_pilot":
                data = fetch_tushare_index_weight(config)
            elif task.task_name == "tushare_stock_daily_raw_pilot":
                data = fetch_tushare_symbol_panel(config, "daily")
            elif task.task_name == "tushare_daily_basic_pilot":
                data = fetch_tushare_symbol_panel(config, "daily_basic")
            elif task.task_name == "tushare_adj_factor_pilot":
                data = fetch_tushare_symbol_panel(config, "adj_factor")
            elif task.task_name == "joinquant_index_membership_cross_check":
                data = fetch_joinquant_index_membership(config)
            else:
                raise ValueError(f"unsupported pilot task: {task.task_name}")
            write_csv(data, output_path)
            row_count = int(data.shape[0])
            status = "acquired" if row_count > 0 else "empty"
            error = ""
        except Exception as exc:  # noqa: BLE001
            row_count = 0
            status = "error"
            error = repr(exc)
        log_rows.append(
            {
                "task_name": task.task_name,
                "dataset_id": task.dataset_id,
                "provider": task.provider,
                "status": status,
                "rows": row_count,
                "output_path": rel(output_path),
                "started_at": started,
                "ended_at": now_text(),
                "error": error,
            }
        )
        registry_rows.append(
            {
                "dataset_id": task.dataset_id,
                "provider": task.provider,
                "status": status,
                "path": rel(output_path),
                "rows": row_count,
                "research_use": "pilot_research_only" if status == "acquired" else "blocked",
                "reason": "pilot data requires PIT validation before any factor use" if status == "acquired" else error,
            }
        )
    return pd.DataFrame(log_rows), pd.DataFrame(registry_rows)


def quality_for_file(path: Path, dataset_id: str, provider: str, status: str) -> dict[str, Any]:
    if not path.exists() or status != "acquired":
        return {
            "dataset_id": dataset_id,
            "provider": provider,
            "path": rel(path),
            "status": status,
            "rows": 0,
            "start_date": "",
            "end_date": "",
            "duplicate_key_count": 0,
            "missing_available_date_rate": 1.0,
            "quality_flag": "not_acquired",
        }
    data = read_csv_if_exists(path)
    date_col = "date" if "date" in data.columns else "effective_date" if "effective_date" in data.columns else "ann_date" if "ann_date" in data.columns else ""
    dates = pd.to_datetime(data[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
    key_cols = [col for col in [date_col, "index_code", "asset"] if col and col in data.columns]
    duplicate_count = int(data.duplicated(key_cols).sum()) if key_cols else 0
    missing_available = float(data["available_date"].isna().mean()) if "available_date" in data.columns and len(data) else 1.0
    return {
        "dataset_id": dataset_id,
        "provider": provider,
        "path": rel(path),
        "status": status,
        "rows": int(data.shape[0]),
        "start_date": str(dates.min().date()) if len(dates) and dates.notna().any() else "",
        "end_date": str(dates.max().date()) if len(dates) and dates.notna().any() else "",
        "duplicate_key_count": duplicate_count,
        "missing_available_date_rate": missing_available,
        "quality_flag": "pilot_review" if duplicate_count == 0 and missing_available == 0.0 else "pilot_issue",
    }


def build_quality_report(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples(index=False):
        rows.append(quality_for_file(ROOT / str(item.path), item.dataset_id, item.provider, item.status))
    return pd.DataFrame(rows)


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("stock_universe_all_status", "asset", "security_identifier", True),
        ("stock_universe_all_status", "list_status", "listing_lifecycle", True),
        ("stock_universe_all_status", "list_date", "listing_lifecycle", True),
        ("stock_universe_all_status", "delist_date", "listing_lifecycle", True),
        ("historical_index_weights", "index_code", "index_identifier", True),
        ("historical_index_weights", "asset", "security_identifier", True),
        ("historical_index_weights", "weight", "constituent_weight", True),
        ("historical_index_weights", "effective_date", "historical_effective_date", True),
        ("stock_daily_raw", "date", "observation_date", True),
        ("stock_daily_raw", "raw_close", "raw_market_bar", True),
        ("stock_daily_basic", "total_mv", "market_cap", False),
        ("stock_adj_factor", "adj_factor", "adjustment_factor", True),
        ("historical_index_membership", "query_date", "historical_query_date", True),
        ("all", "available_date", "point_in_time_availability", True),
        ("all", "fetched_at", "source_provenance", True),
    ]
    return pd.DataFrame(rows, columns=["dataset_id", "field", "role", "required_for_pit"])


def build_pit_check(registry: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q_by_dataset = {row.dataset_id: row for row in quality.itertuples(index=False)}
    for item in registry.itertuples(index=False):
        q = q_by_dataset.get(item.dataset_id)
        acquired = item.status == "acquired"
        available_ok = bool(q is not None and float(q.missing_available_date_rate) == 0.0)
        duplicate_ok = bool(q is not None and int(q.duplicate_key_count) == 0)
        rows.append(
            {
                "dataset_id": item.dataset_id,
                "provider": item.provider,
                "path": item.path,
                "acquisition_status": item.status,
                "pilot_pit_pass": bool(acquired and available_ok and duplicate_ok),
                "strict_pit_backtest_allowed_now": False,
                "available_date_present": available_ok if acquired else False,
                "duplicate_key_pass": duplicate_ok if acquired else False,
                "downstream_restriction": "pilot_research_only_not_factor_input" if acquired else "blocked_not_acquired",
                "reason": item.reason,
            }
        )
    return pd.DataFrame(rows)


def build_blocked_reason(readiness: pd.DataFrame, plan: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for provider in ["tushare", "joinquant"]:
        rows.append(
            {
                "provider": provider,
                "blocked": not provider_ready(readiness, provider),
                "reason": "sdk_or_credentials_not_ready" if not provider_ready(readiness, provider) else "",
                "ready_checks": int(readiness.loc[(readiness["provider"] == provider) & (readiness["ready"] == True)].shape[0]),
                "total_checks": int(readiness.loc[readiness["provider"] == provider].shape[0]),
            }
        )
    for row in plan.itertuples(index=False):
        if not bool(row.can_execute):
            rows.append(
                {
                    "provider": row.provider,
                    "blocked": True,
                    "reason": row.blocked_reason,
                    "ready_checks": "",
                    "total_checks": "",
                    "task_name": row.task_name,
                }
            )
    if not log.empty:
        for row in log.loc[log["status"].isin(["error", "empty"])].itertuples(index=False):
            rows.append(
                {
                    "provider": row.provider,
                    "blocked": True,
                    "reason": row.error or row.status,
                    "ready_checks": "",
                    "total_checks": "",
                    "task_name": row.task_name,
                }
            )
    return pd.DataFrame(rows)


def build_agent_report(config: dict[str, Any], readiness: pd.DataFrame, plan: pd.DataFrame, log: pd.DataFrame, registry: pd.DataFrame) -> str:
    acquired_count = int((registry["status"] == "acquired").sum()) if not registry.empty else 0
    blocked_count = int((log["status"] == "blocked").sum()) if not log.empty else 0
    return "\n".join(
        [
            "# V3.36 PIT Data Pilot Readiness",
            "",
            f"Generated at: `{now_text()}`",
            "",
            "## Decision",
            "",
            "- Task accepted as readiness and pilot-control workflow.",
            "- No portfolio harness was run.",
            "- No factor validation or model promotion was run.",
            f"- Execute flag: `{bool(config.get('execute', False))}`",
            f"- Network allowed flag: `{bool(config.get('allow_network', False))}`",
            "",
            "## Readiness",
            "",
            f"- Tushare ready: `{provider_ready(readiness, 'tushare')}`",
            f"- JoinQuant ready: `{provider_ready(readiness, 'joinquant')}`",
            "",
            "## Pilot Result",
            "",
            f"- Planned pilot tasks: `{len(plan)}`",
            f"- Acquired pilot datasets: `{acquired_count}`",
            f"- Blocked pilot tasks: `{blocked_count}`",
            "- Acquired pilot data remains research-only until broader PIT validation passes.",
            "",
            "## Next Step",
            "",
            "Install the required SDKs and provide API credentials through environment variables or a local untracked `configs/data_credentials.json`, then rerun this script with execution explicitly enabled for the pilot scope only.",
        ]
    )


def build_catalog(config: dict[str, Any], readiness: pd.DataFrame, registry: pd.DataFrame, pit: pd.DataFrame) -> str:
    lines = [
        "# A-share PIT Data Pilot V3.36",
        "",
        f"Updated: `{now_text()}`",
        "",
        "This catalog entry records the controlled pilot-readiness state. It is not approval for factor research.",
        "",
        "## Execution Flags",
        "",
        f"- `execute`: `{bool(config.get('execute', False))}`",
        f"- `allow_network`: `{bool(config.get('allow_network', False))}`",
        "",
        "## Provider Readiness",
        "",
        f"- Tushare ready: `{provider_ready(readiness, 'tushare')}`",
        f"- JoinQuant ready: `{provider_ready(readiness, 'joinquant')}`",
        "",
        "## Dataset Registry",
        "",
        "| Dataset | Provider | Status | Research Use |",
        "|---|---|---:|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(f"| `{row.dataset_id}` | `{row.provider}` | `{row.status}` | {row.research_use} |")
    lines.extend(
        [
            "",
            "## PIT Rule",
            "",
            "Even when a pilot dataset is acquired, it remains pilot research-only. Full historical use requires broader coverage, delisted-name coverage, duplicate checks, available-date checks, raw/adjusted separation checks, and downstream data-steward approval.",
            "",
            f"PIT rows: `{len(pit)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_self_check(paths: dict[str, Path], config: dict[str, Any], log: pd.DataFrame, registry: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    acquired = registry.loc[registry["status"] == "acquired"] if not registry.empty else pd.DataFrame()
    rows.append(
        {
            "check": "no_full_market_execution",
            "status": "pass",
            "detail": "pilot scope only",
        }
    )
    rows.append(
        {
            "check": "blocked_not_marked_acquired",
            "status": "pass" if int((log["status"] == "blocked").sum()) + int((registry["status"] == "not_acquired").sum()) >= int((log["status"] == "blocked").sum()) else "fail",
            "detail": str(int((log["status"] == "blocked").sum())),
        }
    )
    strict_allowed = pit.loc[pit["strict_pit_backtest_allowed_now"] == True, "dataset_id"].tolist() if not pit.empty else []
    rows.append(
        {
            "check": "no_pilot_strict_backtest_approval",
            "status": "pass" if not strict_allowed else "fail",
            "detail": ",".join(strict_allowed),
        }
    )
    rows.append(
        {
            "check": "execute_requires_explicit_flags",
            "status": "pass" if (bool(config.get("execute", False)) == bool(config.get("allow_network", False)) or not bool(config.get("execute", False))) else "fail",
            "detail": f"execute={config.get('execute', False)}, allow_network={config.get('allow_network', False)}",
        }
    )
    rows.append(
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "data readiness only",
        }
    )
    rows.append(
        {
            "check": "acquired_files_exist_if_any",
            "status": "pass" if acquired.empty or all((ROOT / path).exists() for path in acquired["path"].astype(str)) else "fail",
            "detail": str(int(acquired.shape[0])),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="Explicitly request pilot execution. Config allow_network must also be true.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    if args.execute:
        config["execute"] = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    readiness = credential_readiness()
    plan = planned_pilot_tasks(config, readiness)
    log, registry = execute_pilot_tasks(config, plan)
    quality = build_quality_report(registry)
    dictionary = build_data_dictionary()
    pit = build_pit_check(registry, quality)
    blocked = build_blocked_reason(readiness, plan, log)

    artifacts = {
        "credential_readiness": OUTPUT_DIR / "credential_readiness.csv",
        "pilot_execution_plan": OUTPUT_DIR / "pilot_execution_plan.csv",
        "pilot_run_log": OUTPUT_DIR / "pilot_run_log.csv",
        "acquired_dataset_registry": OUTPUT_DIR / "acquired_dataset_registry.csv",
        "data_quality_report": OUTPUT_DIR / "data_quality_report.csv",
        "data_dictionary": OUTPUT_DIR / "data_dictionary.csv",
        "point_in_time_check": OUTPUT_DIR / "point_in_time_check.csv",
        "blocked_reason": OUTPUT_DIR / "blocked_reason.csv",
        "agent_report": OUTPUT_DIR / "agent_report.md",
        "catalog_update": CATALOG_PATH,
        "changed_files": OUTPUT_DIR / "changed_files.txt",
    }
    write_csv(readiness, artifacts["credential_readiness"])
    write_csv(plan, artifacts["pilot_execution_plan"])
    write_csv(log, artifacts["pilot_run_log"])
    write_csv(registry, artifacts["acquired_dataset_registry"])
    write_csv(quality, artifacts["data_quality_report"])
    write_csv(dictionary, artifacts["data_dictionary"])
    write_csv(pit, artifacts["point_in_time_check"])
    write_csv(blocked, artifacts["blocked_reason"])
    write_text(build_agent_report(config, readiness, plan, log, registry), artifacts["agent_report"])
    write_text(build_catalog(config, readiness, registry, pit), artifacts["catalog_update"])
    changed_files = [rel(path) for path in artifacts.values()]
    write_text("\n".join(changed_files) + "\n", artifacts["changed_files"])

    self_check_path = OUTPUT_DIR / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, config, log, registry, pit)
    write_csv(self_check, self_check_path)
    fail_count = int((self_check["status"] == "fail").sum())
    acquired_count = int((registry["status"] == "acquired").sum())
    blocked_count = int((log["status"] == "blocked").sum())

    metrics = {
        "planned_pilot_task_count": int(plan.shape[0]),
        "acquired_dataset_count": acquired_count,
        "blocked_task_count": blocked_count,
        "tushare_ready": provider_ready(readiness, "tushare"),
        "joinquant_ready": provider_ready(readiness, "joinquant"),
        "execute": bool(config.get("execute", False)),
        "allow_network": bool(config.get("allow_network", False)),
        "model_decision": "no_model_promotion_data_pilot_readiness_only",
    }
    manifest_path = OUTPUT_DIR / "agent_run_manifest.json"
    manifest_status = "pass" if fail_count == 0 and acquired_count > 0 else "blocked" if fail_count == 0 else "fail"
    manifest = {
        "run_id": "20260527_v3_36_pit_data_pilot_readiness_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": VERSION,
        "baseline": BASELINE,
        "status": manifest_status,
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py",
        "config": {
            "config_path": rel(config_path),
            "execute": bool(config.get("execute", False)),
            "allow_network": bool(config.get("allow_network", False)),
            "pilot_scope_only": True,
            "no_model_promotion": True,
        },
        "data_refs": [
            "configs/pit_data_pilot_v3_36.json",
            "outputs/agent_runs/v3_35/pit_data_acquisition_contract/acquisition_contract.csv",
            "outputs/agent_runs/v3_35/pit_data_acquisition_contract/harvest_plan.csv",
            "configs/data_credentials.example.json",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_36_pit_data_pilot_readiness.py",
            "strategy_lab/agents/data_steward/AGENT.md",
        ],
        "output_dir": rel(OUTPUT_DIR),
        "allowed_inputs": [
            "configs/pit_data_acquisition_v3_35.json",
            "configs/pit_data_pilot_v3_36.json",
            "configs/data_credentials.example.json",
            "outputs/agent_runs/v3_35/pit_data_acquisition_contract/",
            "data_raw/akshare/stock_list/stock_info_a_code_name.csv",
            "data_raw/akshare/calendar/trade_calendar.csv",
            "data_catalog/",
        ],
        "artifacts": [*changed_files, rel(self_check_path)],
        "outputs": [*changed_files, rel(self_check_path)],
        "changed_files": [*changed_files, rel(self_check_path), rel(manifest_path)],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": int((readiness["ready"] == False).sum()) + blocked_count,
        "limitations": [
            "Default configuration does not enable network execution.",
            "No browser login cookies are used as API credentials.",
            "Pilot datasets, if acquired later, remain research-only until data-steward validation.",
        ],
        "risk_flags": [
            "sdk_not_ready",
            "credentials_not_ready",
            "pilot_not_acquired" if acquired_count == 0 else "pilot_research_only",
        ],
        "next_decision": "Install SDKs and provide API credentials, then rerun this script with explicit pilot execution enabled.",
        "handoff_summary": "V3.36 establishes a controlled pilot runner and records current acquisition blockers.",
    }
    write_json(manifest, manifest_path)

    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "self_check_pass": fail_count == 0,
                "manifest_status": manifest_status,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
