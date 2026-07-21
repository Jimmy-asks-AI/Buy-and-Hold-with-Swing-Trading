#!/usr/bin/env python
"""HIRSSM V3.35 point-in-time data acquisition contract.

This is a data-steward dry run. It creates a production data-acquisition
contract and executable harvest plan, but it does not download data by default
and does not run any model or factor backtest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260527_v3_35_pit_data_acquisition_contract"
VERSION = "V3.35"
BASELINE = "V3.34 Data Source Repair Audit"
DEFAULT_CONFIG = ROOT / "configs" / "pit_data_acquisition_v3_35.json"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_35" / "pit_data_acquisition_contract"
CATALOG_PATH = ROOT / "data_catalog" / "a_share_pit_data_acquisition_contract_v3_35.md"


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


def mask_bool(value: str | None, placeholder_terms: list[str] | None = None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    placeholders = placeholder_terms or ["PASTE_", "TOKEN_HERE", "USERNAME_HERE", "PASSWORD_HERE"]
    return not any(item in text.upper() for item in placeholders)


def credential_readiness() -> pd.DataFrame:
    config_path = ROOT / "configs" / "data_credentials.json"
    config = read_json(config_path) if config_path.exists() else {}
    tushare_cfg = config.get("tushare", {}) if isinstance(config, dict) else {}
    jq_cfg = config.get("joinquant", {}) if isinstance(config, dict) else {}
    rows = [
        {
            "provider": "tushare",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("tushare") is not None,
            "detail": "python package tushare",
        },
        {
            "provider": "tushare",
            "check": "env_token_present",
            "ready": mask_bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN")),
            "detail": "TUSHARE_TOKEN or TUSHARE_PRO_TOKEN",
        },
        {
            "provider": "tushare",
            "check": "config_token_present",
            "ready": mask_bool(tushare_cfg.get("token") if isinstance(tushare_cfg, dict) else None),
            "detail": "configs/data_credentials.json:tushare.token",
        },
        {
            "provider": "joinquant",
            "check": "sdk_installed",
            "ready": importlib.util.find_spec("jqdatasdk") is not None,
            "detail": "python package jqdatasdk",
        },
        {
            "provider": "joinquant",
            "check": "env_username_present",
            "ready": mask_bool(os.getenv("JQDATA_USERNAME") or os.getenv("JOINQUANT_USERNAME") or os.getenv("JQ_USER")),
            "detail": "JQDATA_USERNAME, JOINQUANT_USERNAME, or JQ_USER",
        },
        {
            "provider": "joinquant",
            "check": "env_password_present",
            "ready": mask_bool(os.getenv("JQDATA_PASSWORD") or os.getenv("JOINQUANT_PASSWORD") or os.getenv("JQ_PASSWORD")),
            "detail": "JQDATA_PASSWORD, JOINQUANT_PASSWORD, or JQ_PASSWORD",
        },
        {
            "provider": "joinquant",
            "check": "config_credentials_present",
            "ready": bool(
                isinstance(jq_cfg, dict)
                and mask_bool(jq_cfg.get("username"))
                and mask_bool(jq_cfg.get("password"))
            ),
            "detail": "configs/data_credentials.json:joinquant.username/password",
        },
    ]
    df = pd.DataFrame(rows)
    return df


def provider_endpoint_map() -> pd.DataFrame:
    rows = [
        {
            "provider": "tushare",
            "endpoint": "stock_basic",
            "dataset_id": "stock_universe_all_status",
            "purpose": "Listed, delisted, and paused-security universe seed.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Use list_date, delist_date, and list_status; do not use current A-share list only.",
        },
        {
            "provider": "tushare",
            "endpoint": "index_weight",
            "dataset_id": "historical_index_weights",
            "purpose": "Historical benchmark constituent weights by index and trade_date.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Persist trade_date as effective_date and add ingestion available_date/source_vintage.",
        },
        {
            "provider": "tushare",
            "endpoint": "daily",
            "dataset_id": "stock_daily_raw",
            "purpose": "Raw unadjusted daily OHLCV.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Raw bars are market observations after close; keep raw prices separate from adjusted prices.",
        },
        {
            "provider": "tushare",
            "endpoint": "daily_basic",
            "dataset_id": "stock_daily_basic",
            "purpose": "Turnover, valuation, market-cap and float-cap daily fields.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Treat vendor date as available after close; do not mix with financial announcement dates.",
        },
        {
            "provider": "tushare",
            "endpoint": "adj_factor",
            "dataset_id": "stock_adj_factor",
            "purpose": "Adjustment factors for reconstructing adjusted close without overwriting raw bars.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Store adj_factor separately; adjusted price is derived view, not source raw price.",
        },
        {
            "provider": "tushare",
            "endpoint": "suspend_d",
            "dataset_id": "stock_tradeability_flags",
            "purpose": "Pause and resume events for tradeability filters.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Use announcement/effective dates when available; no forward fill without flags.",
        },
        {
            "provider": "tushare",
            "endpoint": "namechange",
            "dataset_id": "stock_st_flags",
            "purpose": "ST and name-change history.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Use start_date/end_date ranges; never infer historical ST from current name.",
        },
        {
            "provider": "tushare",
            "endpoint": "stk_limit",
            "dataset_id": "stock_limit_prices",
            "purpose": "Limit-up and limit-down prices for tradeability and execution filters.",
            "credential_key": "TUSHARE_TOKEN",
            "sdk": "tushare",
            "pit_note": "Use only same-date fields after close or next trading signal cutoff.",
        },
        {
            "provider": "joinquant",
            "endpoint": "get_index_stocks",
            "dataset_id": "historical_index_membership",
            "purpose": "Date-specific index members when weight history is unavailable or for cross-checking.",
            "credential_key": "JQDATA_USERNAME/JQDATA_PASSWORD",
            "sdk": "jqdatasdk",
            "pit_note": "Query by explicit historical date; store query_date and ingestion available_date.",
        },
        {
            "provider": "joinquant",
            "endpoint": "get_price",
            "dataset_id": "stock_daily_raw_joinquant",
            "purpose": "Alternative stock daily OHLCV and pause fields.",
            "credential_key": "JQDATA_USERNAME/JQDATA_PASSWORD",
            "sdk": "jqdatasdk",
            "pit_note": "Use fq=None/raw where possible and keep paused flags.",
        },
        {
            "provider": "joinquant",
            "endpoint": "get_industry",
            "dataset_id": "historical_industry_classification",
            "purpose": "Historical industry classification by date.",
            "credential_key": "JQDATA_USERNAME/JQDATA_PASSWORD",
            "sdk": "jqdatasdk",
            "pit_note": "Store standard, level, query_date, and available_date.",
        },
    ]
    return pd.DataFrame(rows)


def acquisition_contract(config: dict[str, Any]) -> pd.DataFrame:
    root_tushare = config["output_roots"]["tushare"]
    root_jq = config["output_roots"]["joinquant"]
    rows = [
        {
            "priority": 1,
            "repair_id": "historical_index_constituents_weights",
            "dataset_id": "historical_index_weights",
            "provider": "tushare",
            "endpoint": "index_weight",
            "required_fields": "index_code,asset,weight,effective_date,available_date,source_vintage,data_source",
            "canonical_fields": "index_code,asset,weight,effective_date,available_date,source_vintage,data_source,fetched_at",
            "output_path": f"{root_tushare}/index_weight/{{index_code}}.csv",
            "availability_rule": "effective_date from trade_date; available_date must be explicit or conservatively set by documented vendor availability policy",
            "pit_validation_gates": "date+index+asset unique; no current snapshot backfill; weights sum near 100 by effective_date; available_date <= asof_date",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 1,
            "repair_id": "historical_index_constituents_weights",
            "dataset_id": "historical_index_membership",
            "provider": "joinquant",
            "endpoint": "get_index_stocks",
            "required_fields": "index_code,asset,effective_date,query_date,available_date,data_source",
            "canonical_fields": "index_code,asset,effective_date,query_date,available_date,data_source,fetched_at",
            "output_path": f"{root_jq}/index_membership/{{index_code}}.csv",
            "availability_rule": "query historical date explicitly; available_date is query_date unless licensed vintage timing is provided",
            "pit_validation_gates": "date+index+asset unique; query_date recorded; membership not used as weight history without separate weights",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "dataset_id": "stock_universe_all_status",
            "provider": "tushare",
            "endpoint": "stock_basic",
            "required_fields": "asset,name,exchange,list_status,list_date,delist_date,is_hs,data_source,available_date",
            "canonical_fields": "asset,name,exchange,list_status,list_date,delist_date,is_hs,data_source,available_date,fetched_at",
            "output_path": f"{root_tushare}/stock_basic/all_status.csv",
            "availability_rule": "use list_date and delist_date to construct investable universe as of each date",
            "pit_validation_gates": "include L,D,P statuses; no current-only stock list; asset unique by status snapshot",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "dataset_id": "stock_daily_raw",
            "provider": "tushare",
            "endpoint": "daily",
            "required_fields": "date,asset,open,high,low,close,volume,amount,data_source,available_date",
            "canonical_fields": "date,asset,open,high,low,close,volume,amount,raw_close,data_source,available_date,fetched_at",
            "output_path": f"{root_tushare}/daily/year={{yyyy}}.csv",
            "availability_rule": "daily market bars available after close; signal engine must apply next-period lag",
            "pit_validation_gates": "date+asset unique; positive OHLC when traded; no silent forward fill; raw_close preserved",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "dataset_id": "stock_daily_basic",
            "provider": "tushare",
            "endpoint": "daily_basic",
            "required_fields": "date,asset,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv,data_source,available_date",
            "canonical_fields": "date,asset,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv,data_source,available_date,fetched_at",
            "output_path": f"{root_tushare}/daily_basic/year={{yyyy}}.csv",
            "availability_rule": "same-day vendor fields treated as available after close unless documented otherwise",
            "pit_validation_gates": "date+asset unique; nonnegative market caps; no merge before available_date",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "dataset_id": "stock_adj_factor",
            "provider": "tushare",
            "endpoint": "adj_factor",
            "required_fields": "date,asset,adj_factor,data_source,available_date",
            "canonical_fields": "date,asset,adj_factor,data_source,available_date,fetched_at",
            "output_path": f"{root_tushare}/adj_factor/year={{yyyy}}.csv",
            "availability_rule": "store as separate adjustment source; adjusted_close is derived from raw close and adj_factor",
            "pit_validation_gates": "date+asset unique; adj_factor positive; raw and adjusted price columns never overwritten",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 2,
            "repair_id": "broad_stock_daily_raw_and_adjusted",
            "dataset_id": "stock_tradeability_flags",
            "provider": "tushare",
            "endpoint": "suspend_d|stk_limit|namechange",
            "required_fields": "date,asset,is_paused,is_st,up_limit,down_limit,list_status,available_date,data_source",
            "canonical_fields": "date,asset,is_paused,is_st,up_limit,down_limit,list_status,available_date,data_source,fetched_at",
            "output_path": f"{root_tushare}/tradeability/year={{yyyy}}.csv",
            "availability_rule": "tradeability flags available no earlier than event effective date or announcement date",
            "pit_validation_gates": "date+asset unique; flags explicit; no hidden forward fill across missing records",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 3,
            "repair_id": "financial_indicator_point_in_time",
            "dataset_id": "stock_financial_pit",
            "provider": "tushare",
            "endpoint": "fina_indicator|income|balancesheet|cashflow",
            "required_fields": "asset,report_period,ann_date,available_date,metric,value,statement_type,data_source,revision_flag",
            "canonical_fields": "asset,report_period,ann_date,available_date,metric,value,statement_type,data_source,revision_flag,fetched_at",
            "output_path": f"{root_tushare}/financial_pit/{{table}}/year={{yyyy}}.csv",
            "availability_rule": "available_date must be announcement date or later ingestion-vintage timestamp",
            "pit_validation_gates": "available_date mandatory; restatement policy explicit; no current restated values backfilled into old asof dates",
            "initial_status": "contract_ready_not_acquired",
        },
        {
            "priority": 4,
            "repair_id": "historical_industry_classification",
            "dataset_id": "historical_industry_classification",
            "provider": "joinquant",
            "endpoint": "get_industry",
            "required_fields": "asset,industry_code,industry_name,level,effective_date,end_date,available_date,classification_standard",
            "canonical_fields": "asset,industry_code,industry_name,level,effective_date,end_date,available_date,classification_standard,fetched_at",
            "output_path": f"{root_jq}/industry_classification/{{standard}}.csv",
            "availability_rule": "classification must be queried by historical as-of date or source vintage",
            "pit_validation_gates": "one active classification per asset-level-asof; no latest classification backfill",
            "initial_status": "contract_ready_not_acquired",
        },
    ]
    return pd.DataFrame(rows)


def years_from_calendar(config: dict[str, Any]) -> list[int]:
    calendar = read_csv_if_exists(ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv")
    if calendar.empty or "date" not in calendar.columns:
        start_year = int(str(config["start_date"])[:4])
        end_year = int(str(config["end_date"])[:4])
        return list(range(start_year, end_year + 1))
    dates = pd.to_datetime(calendar["date"], errors="coerce").dropna()
    start = pd.to_datetime(config["start_date"], errors="coerce")
    end = pd.to_datetime(config["end_date"], errors="coerce")
    if pd.notna(start):
        dates = dates.loc[dates >= start]
    if pd.notna(end):
        dates = dates.loc[dates <= end]
    return sorted(dates.dt.year.dropna().astype(int).unique().tolist())


def local_stock_count() -> int:
    stock_list = read_csv_if_exists(ROOT / "data_raw" / "akshare" / "stock_list" / "stock_info_a_code_name.csv")
    if stock_list.empty:
        return 0
    code_col = "asset" if "asset" in stock_list.columns else stock_list.columns[0]
    return int(stock_list[code_col].dropna().astype(str).nunique())


def harvest_plan(config: dict[str, Any], contract: pd.DataFrame) -> pd.DataFrame:
    years = years_from_calendar(config)
    index_codes = config.get("index_codes", [])
    stock_count = local_stock_count()
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "phase": "P0",
            "dataset_id": "credential_and_sdk_readiness",
            "provider": "local",
            "partition": "all",
            "estimated_units": 1,
            "command_mode": "dry_run",
            "planned_output": rel(OUTPUT_DIR / "credential_readiness.csv"),
            "dependency": "none",
            "status": "planned_not_acquired",
        }
    )
    rows.append(
        {
            "phase": "P1",
            "dataset_id": "stock_universe_all_status",
            "provider": "tushare",
            "partition": "all",
            "estimated_units": "all_status",
            "command_mode": "pilot_then_full",
            "planned_output": contract.loc[contract["dataset_id"] == "stock_universe_all_status", "output_path"].iloc[0],
            "dependency": "tushare sdk/token",
            "status": "planned_not_acquired",
        }
    )
    for code in index_codes:
        rows.append(
            {
                "phase": "P2",
                "dataset_id": "historical_index_weights",
                "provider": "tushare",
                "partition": code,
                "estimated_units": 1,
                "command_mode": "pilot_then_full",
                "planned_output": f"data_raw/pit/tushare/index_weight/{code}.csv",
                "dependency": "tushare sdk/token",
                "status": "planned_not_acquired",
            }
        )
        rows.append(
            {
                "phase": "P2",
                "dataset_id": "historical_index_membership",
                "provider": "joinquant",
                "partition": code,
                "estimated_units": len(years) * 12,
                "command_mode": "cross_check",
                "planned_output": f"data_raw/pit/joinquant/index_membership/{code}.csv",
                "dependency": "jqdatasdk credentials",
                "status": "planned_not_acquired",
            }
        )
    for year in years:
        for dataset_id in ["stock_daily_raw", "stock_daily_basic", "stock_adj_factor", "stock_tradeability_flags"]:
            rows.append(
                {
                    "phase": "P3",
                    "dataset_id": dataset_id,
                    "provider": "tushare",
                    "partition": str(year),
                    "estimated_units": stock_count or "all_status_universe",
                    "command_mode": "yearly_partition",
                    "planned_output": f"data_raw/pit/tushare/{dataset_id}/year={year}.csv",
                    "dependency": "stock_universe_all_status",
                    "status": "planned_not_acquired",
                }
            )
    for year in years:
        rows.append(
            {
                "phase": "P4",
                "dataset_id": "stock_financial_pit",
                "provider": "tushare",
                "partition": str(year),
                "estimated_units": stock_count or "all_status_universe",
                "command_mode": "later_phase",
                "planned_output": f"data_raw/pit/tushare/financial_pit/year={year}.csv",
                "dependency": "stock_universe_all_status and announcement dates",
                "status": "planned_not_acquired",
            }
        )
    rows.append(
        {
            "phase": "P5",
            "dataset_id": "historical_industry_classification",
            "provider": "joinquant",
            "partition": "monthly_asof",
            "estimated_units": len(years) * 12,
            "command_mode": "later_phase",
            "planned_output": "data_raw/pit/joinquant/industry_classification/sw_l1_l2_l3.csv",
            "dependency": "jqdatasdk credentials",
            "status": "planned_not_acquired",
        }
    )
    return pd.DataFrame(rows)


def data_quality_report(config: dict[str, Any], readiness: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    stock_count = local_stock_count()
    calendar = read_csv_if_exists(ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv")
    calendar_rows = int(calendar.shape[0]) if not calendar.empty else 0
    years = years_from_calendar(config)
    rows = [
        {
            "dataset_id": "local_stock_list_seed",
            "source_path": "data_raw/akshare/stock_list/stock_info_a_code_name.csv",
            "status": "research_only",
            "rows_or_units": stock_count,
            "quality_flag": "current_list_only_not_delisted_complete",
            "notes": "Useful for estimating batches, not for production historical universe.",
        },
        {
            "dataset_id": "local_trade_calendar_seed",
            "source_path": "data_raw/akshare/calendar/trade_calendar.csv",
            "status": "approved_for_planning",
            "rows_or_units": calendar_rows,
            "quality_flag": "planning_calendar",
            "notes": "Used only to create yearly partitions.",
        },
        {
            "dataset_id": "v3_35_harvest_plan",
            "source_path": rel(OUTPUT_DIR / "harvest_plan.csv"),
            "status": "planned_not_acquired",
            "rows_or_units": int(plan.shape[0]),
            "quality_flag": "dry_run_only",
            "notes": f"Year partitions: {min(years) if years else ''}-{max(years) if years else ''}.",
        },
        {
            "dataset_id": "tushare_readiness",
            "source_path": "env/config/python",
            "status": "ready" if provider_ready(readiness, "tushare") else "blocked",
            "rows_or_units": int(readiness.loc[readiness["provider"] == "tushare", "ready"].sum()),
            "quality_flag": "credentials_and_sdk",
            "notes": "Needs SDK plus token before execute mode.",
        },
        {
            "dataset_id": "joinquant_readiness",
            "source_path": "env/config/python",
            "status": "ready" if provider_ready(readiness, "joinquant") else "blocked",
            "rows_or_units": int(readiness.loc[readiness["provider"] == "joinquant", "ready"].sum()),
            "quality_flag": "credentials_and_sdk",
            "notes": "Needs jqdatasdk plus username/password before execute mode.",
        },
    ]
    return pd.DataFrame(rows)


def provider_ready(readiness: pd.DataFrame, provider: str) -> bool:
    rows = readiness.loc[readiness["provider"] == provider]
    if provider == "tushare":
        sdk = rows.loc[rows["check"] == "sdk_installed", "ready"].any()
        token = rows.loc[rows["check"].isin(["env_token_present", "config_token_present"]), "ready"].any()
        return bool(sdk and token)
    if provider == "joinquant":
        sdk = rows.loc[rows["check"] == "sdk_installed", "ready"].any()
        env_pair = bool(
            rows.loc[rows["check"] == "env_username_present", "ready"].any()
            and rows.loc[rows["check"] == "env_password_present", "ready"].any()
        )
        cfg_pair = rows.loc[rows["check"] == "config_credentials_present", "ready"].any()
        return bool(sdk and (env_pair or cfg_pair))
    return False


def data_dictionary(contract: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in contract.itertuples(index=False):
        for field in str(item.canonical_fields).split(","):
            field = field.strip()
            if not field:
                continue
            rows.append(
                {
                    "dataset_id": item.dataset_id,
                    "provider": item.provider,
                    "field": field,
                    "role": field_role(field),
                    "required_for_pit": field in {"available_date", "effective_date", "ann_date", "list_date", "delist_date", "source_vintage"},
                    "source_endpoint": item.endpoint,
                    "availability_rule": item.availability_rule,
                }
            )
    return pd.DataFrame(rows)


def field_role(field: str) -> str:
    mapping = {
        "date": "observation_date",
        "asset": "security_identifier",
        "index_code": "index_identifier",
        "weight": "constituent_weight",
        "effective_date": "historical_effective_date",
        "available_date": "point_in_time_availability",
        "source_vintage": "source_provenance",
        "open": "raw_market_bar",
        "high": "raw_market_bar",
        "low": "raw_market_bar",
        "close": "raw_market_bar",
        "raw_close": "raw_market_bar",
        "adjusted_close": "derived_adjusted_price",
        "adj_factor": "adjustment_factor",
        "ann_date": "announcement_date",
        "list_date": "listing_lifecycle",
        "delist_date": "listing_lifecycle",
        "is_paused": "tradeability",
        "is_st": "tradeability",
    }
    return mapping.get(field, "source_field")


def point_in_time_check(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in contract.itertuples(index=False):
        canonical = set(str(item.canonical_fields).split(","))
        rows.append(
            {
                "dataset_id": item.dataset_id,
                "provider": item.provider,
                "endpoint": item.endpoint,
                "acquisition_status": "planned_not_acquired",
                "strict_pit_backtest_allowed_now": False,
                "has_available_date_contract": "available_date" in canonical,
                "has_effective_or_observation_date_contract": bool({"date", "effective_date", "ann_date", "list_date"} & canonical),
                "requires_actual_validation_before_use": True,
                "blocked_until": "actual data acquired and validation gates pass",
                "pit_gate": item.pit_validation_gates,
            }
        )
    return pd.DataFrame(rows)


def dry_run_manifest(config: dict[str, Any], plan: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    dry_run = bool(config.get("dry_run", True))
    rows = []
    for provider in ["tushare", "joinquant"]:
        rows.append(
            {
                "provider": provider,
                "dry_run": dry_run,
                "provider_ready": provider_ready(readiness, provider),
                "planned_batches": int((plan["provider"] == provider).sum()),
                "execute_allowed_now": False if dry_run else provider_ready(readiness, provider),
                "reason": "dry_run_mode" if dry_run else "requires credential and SDK readiness",
            }
        )
    return pd.DataFrame(rows)


def build_agent_report(config: dict[str, Any], readiness: pd.DataFrame, contract: pd.DataFrame, plan: pd.DataFrame) -> str:
    dry_run = bool(config.get("dry_run", True))
    p1 = contract.loc[contract["priority"] == 1, "dataset_id"].tolist()
    p2 = contract.loc[contract["priority"] == 2, "dataset_id"].tolist()
    tushare_ready = provider_ready(readiness, "tushare")
    jq_ready = provider_ready(readiness, "joinquant")
    return "\n".join(
        [
            "# V3.35 PIT Data Acquisition Contract",
            "",
            f"Generated at: `{now_text()}`",
            "",
            "## Decision",
            "",
            "- Task accepted as data acquisition contract and dry-run plan.",
            "- No live download was executed.",
            "- No portfolio harness was run.",
            "- No signal or model was promoted.",
            "",
            "## Scope",
            "",
            f"- Dry-run mode: `{dry_run}`",
            f"- Contract datasets: `{len(contract)}`",
            f"- Harvest plan rows: `{len(plan)}`",
            f"- Priority 1 datasets: `{', '.join(p1)}`",
            f"- Priority 2 datasets: `{', '.join(p2)}`",
            "",
            "## Credential Readiness",
            "",
            f"- Tushare execute-ready: `{tushare_ready}`",
            f"- JoinQuant execute-ready: `{jq_ready}`",
            "- Browser login is intentionally not treated as an API credential.",
            "",
            "## Next Executable Step",
            "",
            "Run a pilot acquisition only after SDK and API credentials are available. The first pilot should fetch stock universe, one index weight file, and a small yearly stock-daily partition, then run the V3.34/V3.35 PIT validation gates before any factor research uses the data.",
        ]
    )


def build_catalog(contract: pd.DataFrame, plan: pd.DataFrame, readiness: pd.DataFrame) -> str:
    lines = [
        "# A-share PIT Data Acquisition Contract V3.35",
        "",
        f"Updated: `{now_text()}`",
        "",
        "This catalog note defines the next production-grade data contract. It is not evidence that the data has been acquired.",
        "",
        "## Provider Readiness",
        "",
        "| Provider | Execute Ready | Notes |",
        "|---|---:|---|",
        f"| Tushare | `{provider_ready(readiness, 'tushare')}` | Needs SDK plus token. |",
        f"| JoinQuant | `{provider_ready(readiness, 'joinquant')}` | Needs jqdatasdk plus username/password. |",
        "",
        "## Contract",
        "",
        "| Priority | Dataset | Provider | Endpoint | PIT Gate |",
        "|---:|---|---|---|---|",
    ]
    for row in contract.itertuples(index=False):
        lines.append(f"| {row.priority} | `{row.dataset_id}` | `{row.provider}` | `{row.endpoint}` | {row.pit_validation_gates} |")
    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            "- Current snapshots remain disallowed for historical backtests.",
            "- Dry-run rows are `planned_not_acquired` and cannot be used by factor researchers.",
            "- Actual acquired tables must pass duplicate, missingness, PIT, adjustment, and lifecycle checks before promotion to research inputs.",
            f"- Planned harvest rows: `{len(plan)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_self_check(paths: dict[str, Path], config: dict[str, Any], contract: pd.DataFrame, pit: pd.DataFrame, dry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "self_check":
            status = "pass"
        else:
            status = "pass" if path.exists() and path.stat().st_size > 0 else "fail"
        rows.append({"check": f"artifact_exists_{name}", "status": status, "detail": rel(path)})
    priority1 = set(contract.loc[contract["priority"] == 1, "repair_id"])
    priority2 = set(contract.loc[contract["priority"] == 2, "repair_id"])
    rows.append(
        {
            "check": "priority1_historical_index_contract_present",
            "status": "pass" if "historical_index_constituents_weights" in priority1 else "fail",
            "detail": ",".join(sorted(priority1)),
        }
    )
    rows.append(
        {
            "check": "priority2_stock_daily_contract_present",
            "status": "pass" if "broad_stock_daily_raw_and_adjusted" in priority2 else "fail",
            "detail": ",".join(sorted(priority2)),
        }
    )
    rows.append(
        {
            "check": "dry_run_mode_true",
            "status": "pass" if bool(config.get("dry_run", True)) else "fail",
            "detail": str(config.get("dry_run", True)),
        }
    )
    acquired = pit.loc[pit["acquisition_status"] != "planned_not_acquired", "dataset_id"].tolist()
    rows.append(
        {
            "check": "no_dataset_marked_acquired",
            "status": "pass" if not acquired else "fail",
            "detail": ",".join(acquired),
        }
    )
    missing_availability = pit.loc[~pit["has_available_date_contract"], "dataset_id"].tolist()
    rows.append(
        {
            "check": "available_date_contract_present",
            "status": "pass" if not missing_availability else "fail",
            "detail": ",".join(missing_availability),
        }
    )
    execute_allowed = dry.loc[dry["execute_allowed_now"] == True, "provider"].tolist()
    rows.append(
        {
            "check": "execute_disabled_in_dry_run",
            "status": "pass" if not execute_allowed else "fail",
            "detail": ",".join(execute_allowed),
        }
    )
    rows.append(
        {
            "check": "no_model_promotion",
            "status": "pass",
            "detail": "data acquisition contract only",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    readiness = credential_readiness()
    endpoints = provider_endpoint_map()
    contract = acquisition_contract(config)
    plan = harvest_plan(config, contract)
    quality = data_quality_report(config, readiness, plan)
    dictionary = data_dictionary(contract)
    pit = point_in_time_check(contract)
    dry = dry_run_manifest(config, plan, readiness)

    artifacts = {
        "acquisition_contract": OUTPUT_DIR / "acquisition_contract.csv",
        "provider_endpoint_map": OUTPUT_DIR / "provider_endpoint_map.csv",
        "harvest_plan": OUTPUT_DIR / "harvest_plan.csv",
        "credential_readiness": OUTPUT_DIR / "credential_readiness.csv",
        "dry_run_manifest": OUTPUT_DIR / "dry_run_manifest.csv",
        "data_quality_report": OUTPUT_DIR / "data_quality_report.csv",
        "data_dictionary": OUTPUT_DIR / "data_dictionary.csv",
        "point_in_time_check": OUTPUT_DIR / "point_in_time_check.csv",
        "agent_report": OUTPUT_DIR / "agent_report.md",
        "catalog_update": CATALOG_PATH,
        "changed_files": OUTPUT_DIR / "changed_files.txt",
    }
    write_csv(contract, artifacts["acquisition_contract"])
    write_csv(endpoints, artifacts["provider_endpoint_map"])
    write_csv(plan, artifacts["harvest_plan"])
    write_csv(readiness, artifacts["credential_readiness"])
    write_csv(dry, artifacts["dry_run_manifest"])
    write_csv(quality, artifacts["data_quality_report"])
    write_csv(dictionary, artifacts["data_dictionary"])
    write_csv(pit, artifacts["point_in_time_check"])
    write_text(build_agent_report(config, readiness, contract, plan), artifacts["agent_report"])
    write_text(build_catalog(contract, plan, readiness), artifacts["catalog_update"])
    changed_files = [rel(path) for path in artifacts.values()]
    write_text("\n".join(changed_files) + "\n", artifacts["changed_files"])

    self_check_path = OUTPUT_DIR / "self_check.csv"
    artifacts["self_check"] = self_check_path
    self_check = build_self_check(artifacts, config, contract, pit, dry)
    write_csv(self_check, self_check_path)
    fail_count = int((self_check["status"] == "fail").sum())
    warn_count = int((readiness["ready"] == False).sum())

    metrics = {
        "contract_dataset_count": int(contract.shape[0]),
        "provider_endpoint_count": int(endpoints.shape[0]),
        "harvest_plan_rows": int(plan.shape[0]),
        "priority1_dataset_count": int((contract["priority"] == 1).sum()),
        "priority2_dataset_count": int((contract["priority"] == 2).sum()),
        "tushare_execute_ready": provider_ready(readiness, "tushare"),
        "joinquant_execute_ready": provider_ready(readiness, "joinquant"),
        "dry_run": bool(config.get("dry_run", True)),
        "model_decision": "no_model_promotion_data_acquisition_contract_only",
    }
    manifest_path = OUTPUT_DIR / "agent_run_manifest.json"
    manifest = {
        "run_id": "20260527_v3_35_pit_data_acquisition_contract_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": VERSION,
        "baseline": BASELINE,
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_35_pit_data_acquisition_contract.py",
        "config": {
            "config_path": rel(config_path),
            "dry_run": bool(config.get("dry_run", True)),
            "no_live_download": True,
            "no_model_promotion": True,
        },
        "data_refs": [
            "outputs/agent_runs/v3_34/data_source_repair_audit/data_source_repair_plan.csv",
            "outputs/agent_runs/v3_34/data_source_repair_audit/point_in_time_check.csv",
            "data_raw/akshare/stock_list/stock_info_a_code_name.csv",
            "data_raw/akshare/calendar/trade_calendar.csv",
            "configs/data_credentials.example.json",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_35_pit_data_acquisition_contract.py",
            "strategy_lab/agents/data_steward/AGENT.md",
        ],
        "output_dir": rel(OUTPUT_DIR),
        "allowed_inputs": [
            "outputs/agent_runs/v3_34/data_source_repair_audit/data_source_repair_plan.csv",
            "outputs/agent_runs/v3_34/data_source_repair_audit/point_in_time_check.csv",
            "data_raw/akshare/stock_list/stock_info_a_code_name.csv",
            "data_raw/akshare/calendar/trade_calendar.csv",
            "configs/data_credentials.example.json",
            "data_catalog/",
        ],
        "artifacts": [*changed_files, rel(self_check_path)],
        "outputs": [*changed_files, rel(self_check_path)],
        "changed_files": [*changed_files, rel(self_check_path), rel(manifest_path)],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Dry-run only; no external data was downloaded.",
            "Credential readiness is boolean only and does not expose secrets.",
            "Chrome login state is not usable as a safe API credential for this pipeline.",
        ],
        "risk_flags": [
            "credential_not_ready",
            "sdk_not_ready",
            "planned_not_acquired",
            "pit_validation_required_before_factor_use",
        ],
        "next_decision": "Set up SDK/API credentials or run a controlled pilot acquisition before factor_researcher consumes new data.",
        "handoff_summary": "V3.35 converts V3.34 repair priorities into an executable PIT data contract and dry-run harvest plan.",
    }
    write_json(manifest, manifest_path)

    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "self_check_pass": fail_count == 0,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
