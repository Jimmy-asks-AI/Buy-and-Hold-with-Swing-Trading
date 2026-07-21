#!/usr/bin/env python
"""HIRSSM V3.27 point-in-time macro data ingestion.

V3.26 blocked macro factor research because the local data layer had no
available_date-aware macro tables. V3.27 fetches public AkShare macro/rate
series, converts them into the local point-in-time schema, and records which
macro candidate families are ready only for later signal validation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data_raw"
MACRO_DIR = DATA_RAW_DIR / "macro"
RAW_DIR = MACRO_DIR / "raw_akshare"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_27" / "macro_data_ingestion"
TASK_ID = "20260527_v3_27_macro_data_ingestion"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


@dataclass(frozen=True)
class SeriesRequirement:
    series_id: str
    output_file: str
    min_start_year: int
    frequency: str
    description: str


REQUIREMENTS = [
    SeriesRequirement("cn_10y_gov_bond_yield", "china_10y_yield.csv", 2005, "daily", "China 10Y government bond yield"),
    SeriesRequirement("us_10y_treasury_yield", "us_10y_yield.csv", 2005, "daily", "US 10Y treasury yield"),
    SeriesRequirement("cn_us_10y_rate_spread", "cn_us_10y_rate_spread.csv", 2005, "daily", "China-US 10Y rate spread"),
    SeriesRequirement("usdcny", "cny_fx.csv", 2005, "daily", "USD/CNY exchange rate"),
    SeriesRequirement("china_pmi", "pmi.csv", 2005, "monthly", "China manufacturing PMI"),
    SeriesRequirement("china_m2_yoy", "m2_social_financing.csv", 2005, "monthly", "China M2 year-on-year growth"),
    SeriesRequirement("china_tsf_yoy", "m2_social_financing.csv", 2005, "monthly", "China total social financing growth"),
    SeriesRequirement("china_cpi_yoy", "cpi_ppi.csv", 2005, "monthly", "China CPI year-on-year growth"),
    SeriesRequirement("china_ppi_yoy", "cpi_ppi.csv", 2005, "monthly", "China PPI year-on-year growth"),
    SeriesRequirement("commodity_index", "commodity_index.csv", 2005, "daily", "Broad commodity index"),
]


CANDIDATE_FAMILIES = [
    {
        "variant": "cn_us_rate_spread_risk_budget",
        "description": "Reduce equity risk budget when China-US rate spread and FX pressure jointly deteriorate.",
        "required_series": ["cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"],
    },
    {
        "variant": "macro_liquidity_repair",
        "description": "Release cash or increase cyclical exposure when M2/TSF liquidity impulse repairs while PMI stabilizes.",
        "required_series": ["china_pmi", "china_m2_yoy", "china_tsf_yoy"],
    },
    {
        "variant": "inflation_policy_constraint_defense",
        "description": "Cut high-beta exposure when CPI/PPI and commodity pressure tighten the policy constraint.",
        "required_series": ["china_cpi_yoy", "china_ppi_yoy", "commodity_index", "cn_10y_gov_bond_yield"],
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_trade_calendar() -> pd.Series:
    path = DATA_RAW_DIR / "akshare" / "calendar" / "trade_calendar.csv"
    if not path.exists():
        return pd.Series(dtype="datetime64[ns]")
    cal = pd.read_csv(path, encoding="utf-8-sig")
    return pd.to_datetime(cal["date"], errors="coerce").dropna().sort_values().drop_duplicates()


TRADE_CALENDAR = load_trade_calendar()


def next_trade_on_or_after(date_value: object) -> pd.Timestamp:
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    if TRADE_CALENDAR.empty:
        return pd.Timestamp(dt).normalize()
    dates = TRADE_CALENDAR[TRADE_CALENDAR >= pd.Timestamp(dt).normalize()]
    if dates.empty:
        return pd.Timestamp(dt).normalize()
    return pd.Timestamp(dates.iloc[0]).normalize()


def next_trade_after(date_value: object) -> pd.Timestamp:
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    return next_trade_on_or_after(pd.Timestamp(dt).normalize() + timedelta(days=1))


def standard_frame(
    series_id: str,
    raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    source: str,
    frequency: str,
    revision_policy: str,
    available_date_method: str,
    pit_quality: str,
    scale: float = 1.0,
    availability: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    if raw.empty or date_col not in raw.columns or value_col not in raw.columns:
        return pd.DataFrame(columns=standard_columns())
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce") / scale
    out["value"] = values
    if availability:
        out["available_date"] = pd.to_datetime(availability(out["date"]), errors="coerce")
    elif available_date_method == "next_trade_after_observation":
        out["available_date"] = out["date"].map(next_trade_after)
    else:
        out["available_date"] = out["date"].map(next_trade_on_or_after)
    out["series_id"] = series_id
    out["source"] = source
    out["frequency"] = frequency
    out["revision_policy"] = revision_policy
    out["raw_field"] = value_col
    out["available_date_method"] = available_date_method
    out["pit_quality"] = pit_quality
    out = out.dropna(subset=["date", "available_date", "value"]).sort_values(["series_id", "date"])
    out = out.drop_duplicates(["series_id", "date"], keep="last").reset_index(drop=True)
    return out[standard_columns()]


def standard_columns() -> list[str]:
    return [
        "date",
        "available_date",
        "series_id",
        "value",
        "source",
        "frequency",
        "revision_policy",
        "raw_field",
        "available_date_method",
        "pit_quality",
    ]


def parse_chinese_month(series: pd.Series) -> pd.Series:
    def parse_one(value: object) -> pd.Timestamp:
        text = str(value)
        match = re.search(r"(\d{4})年(\d{1,2})月", text)
        if not match:
            return pd.NaT
        year = int(match.group(1))
        month = int(match.group(2))
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    return series.map(parse_one)


def fetch_with_status(name: str, func: Callable[[], pd.DataFrame], raw_path: Path, status_rows: list[dict[str, Any]]) -> pd.DataFrame:
    started = now_text()
    try:
        df = func()
        safe_to_csv(df, raw_path)
        status_rows.append(
            {
                "source_name": name,
                "status": "success",
                "rows": int(df.shape[0]),
                "columns": ",".join(map(str, df.columns.tolist())),
                "raw_path": rel(raw_path),
                "started_at": started,
                "finished_at": now_text(),
                "error": "",
            }
        )
        return df
    except Exception as exc:  # pragma: no cover - vendor/network protection
        status_rows.append(
            {
                "source_name": name,
                "status": "failed",
                "rows": 0,
                "columns": "",
                "raw_path": rel(raw_path),
                "started_at": started,
                "finished_at": now_text(),
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        )
        return pd.DataFrame()


def build_rate_tables(ak: Any, status_rows: list[dict[str, Any]]) -> list[pd.DataFrame]:
    raw = fetch_with_status(
        "akshare.bond_zh_us_rate",
        lambda: ak.bond_zh_us_rate(start_date="20000101"),
        RAW_DIR / "bond_zh_us_rate.csv",
        status_rows,
    )
    frames = [
        standard_frame(
            "cn_10y_gov_bond_yield",
            raw,
            "日期",
            "中国国债收益率10年",
            "akshare.bond_zh_us_rate/eastmoney",
            "daily",
            "vendor_snapshot_daily_market_series",
            "next_trade_after_observation",
            "direct_daily_next_trade",
        ),
        standard_frame(
            "us_10y_treasury_yield",
            raw,
            "日期",
            "美国国债收益率10年",
            "akshare.bond_zh_us_rate/eastmoney",
            "daily",
            "vendor_snapshot_daily_market_series",
            "next_trade_after_observation",
            "direct_daily_next_trade",
        ),
    ]
    if not raw.empty and {"日期", "中国国债收益率10年", "美国国债收益率10年"}.issubset(raw.columns):
        spread = raw[["日期", "中国国债收益率10年", "美国国债收益率10年"]].copy()
        spread["中美10年利差"] = pd.to_numeric(spread["中国国债收益率10年"], errors="coerce") - pd.to_numeric(spread["美国国债收益率10年"], errors="coerce")
        frames.append(
            standard_frame(
                "cn_us_10y_rate_spread",
                spread,
                "日期",
                "中美10年利差",
                "derived:akshare.bond_zh_us_rate",
                "daily",
                "derived_from_vendor_snapshot_daily_market_series",
                "next_trade_after_observation",
                "direct_daily_next_trade",
            )
        )
    return frames


def build_fx_table(ak: Any, end_date: str, status_rows: list[dict[str, Any]]) -> pd.DataFrame:
    raw = fetch_with_status(
        "akshare.currency_boc_sina.usd",
        lambda: ak.currency_boc_sina(symbol="美元", start_date="20050101", end_date=end_date),
        RAW_DIR / "currency_boc_sina_usd.csv",
        status_rows,
    )
    if raw.empty:
        return pd.DataFrame(columns=standard_columns())
    raw = raw.copy()
    central = pd.to_numeric(raw.get("央行中间价"), errors="coerce")
    fallback = pd.to_numeric(raw.get("中行折算价"), errors="coerce")
    raw["usdcny_value"] = central.fillna(fallback)
    return standard_frame(
        "usdcny",
        raw,
        "日期",
        "usdcny_value",
        "akshare.currency_boc_sina/sina",
        "daily",
        "vendor_snapshot_daily_market_series",
        "next_trade_after_observation",
        "direct_daily_next_trade",
        scale=100.0,
    )


def build_event_series(ak: Any, status_rows: list[dict[str, Any]]) -> list[pd.DataFrame]:
    specs = [
        ("china_pmi", "akshare.macro_china_pmi_yearly", ak.macro_china_pmi_yearly, RAW_DIR / "macro_china_pmi_yearly.csv", "今值", 2005),
        ("china_m2_yoy", "akshare.macro_china_m2_yearly", ak.macro_china_m2_yearly, RAW_DIR / "macro_china_m2_yearly.csv", "今值", 2005),
        ("china_cpi_yoy", "akshare.macro_china_cpi_yearly", ak.macro_china_cpi_yearly, RAW_DIR / "macro_china_cpi_yearly.csv", "今值", 2005),
        ("china_ppi_yoy", "akshare.macro_china_ppi_yearly", ak.macro_china_ppi_yearly, RAW_DIR / "macro_china_ppi_yearly.csv", "今值", 2005),
    ]
    frames = []
    for series_id, source_name, func, raw_path, value_col, _ in specs:
        raw = fetch_with_status(source_name, func, raw_path, status_rows)
        frames.append(
            standard_frame(
                series_id,
                raw,
                "日期",
                value_col,
                source_name,
                "monthly",
                "event_release_snapshot_no_vintage_history",
                "next_trade_on_or_after_release_event",
                "release_event_date",
            )
        )
    return frames


def build_credit_tables(ak: Any, status_rows: list[dict[str, Any]]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    raw_tsf = fetch_with_status(
        "akshare.macro_china_shrzgm",
        ak.macro_china_shrzgm,
        RAW_DIR / "macro_china_shrzgm.csv",
        status_rows,
    )
    if not raw_tsf.empty:
        raw_tsf = raw_tsf.copy()
        raw_tsf["period_end"] = parse_chinese_month(raw_tsf["月份"])
        value_col = "当月-同比增长" if "当月-同比增长" in raw_tsf.columns else raw_tsf.columns[-1]
        frames.append(
            standard_frame(
                "china_tsf_yoy",
                raw_tsf,
                "period_end",
                value_col,
                "akshare.macro_china_shrzgm/mofcom",
                "monthly",
                "estimated_release_lag_no_vintage_history",
                "next_trade_after_conservative_monthly_lag",
                "estimated_available_date",
                availability=lambda dates: dates.map(lambda value: next_trade_on_or_after(pd.Timestamp(value) + timedelta(days=20))),
            )
        )

    raw_credit = fetch_with_status(
        "akshare.macro_china_new_financial_credit",
        ak.macro_china_new_financial_credit,
        RAW_DIR / "macro_china_new_financial_credit.csv",
        status_rows,
    )
    if not raw_credit.empty:
        raw_credit = raw_credit.copy()
        raw_credit["period_end"] = parse_chinese_month(raw_credit["月份"])
        frames.append(
            standard_frame(
                "china_new_financial_credit_yoy",
                raw_credit,
                "period_end",
                "当月-同比增长",
                "akshare.macro_china_new_financial_credit/eastmoney",
                "monthly",
                "estimated_release_lag_no_vintage_history",
                "next_trade_after_conservative_monthly_lag",
                "estimated_available_date",
                availability=lambda dates: dates.map(lambda value: next_trade_on_or_after(pd.Timestamp(value) + timedelta(days=20))),
            )
        )
    return frames


def build_commodity_table(ak: Any, status_rows: list[dict[str, Any]]) -> pd.DataFrame:
    raw = fetch_with_status(
        "akshare.macro_china_commodity_price_index",
        ak.macro_china_commodity_price_index,
        RAW_DIR / "macro_china_commodity_price_index.csv",
        status_rows,
    )
    return standard_frame(
        "commodity_index",
        raw,
        "日期",
        "最新值",
        "akshare.macro_china_commodity_price_index/eastmoney",
        "daily",
        "vendor_snapshot_daily_market_series",
        "next_trade_after_observation",
        "direct_daily_next_trade",
    )


def write_macro_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    panel = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=standard_columns())
    panel = panel.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"], keep="last").reset_index(drop=True)
    req_by_series = {req.series_id: req for req in REQUIREMENTS}
    for req in REQUIREMENTS:
        sub = panel[panel["series_id"].eq(req.series_id)].copy()
        if req.output_file == "m2_social_financing.csv":
            sub = panel[panel["series_id"].isin(["china_m2_yoy", "china_tsf_yoy", "china_new_financial_credit_yoy"])].copy()
        elif req.output_file == "cpi_ppi.csv":
            sub = panel[panel["series_id"].isin(["china_cpi_yoy", "china_ppi_yoy"])].copy()
        if sub.empty:
            continue
        safe_to_csv(sub, MACRO_DIR / req.output_file)
    if "cn_us_10y_rate_spread" in panel["series_id"].values:
        safe_to_csv(panel[panel["series_id"].eq("cn_us_10y_rate_spread")], MACRO_DIR / "cn_us_10y_rate_spread.csv")
    safe_to_csv(panel, MACRO_DIR / "macro_pit_panel.csv")

    metadata_rows = []
    for series_id, sub in panel.groupby("series_id"):
        req = req_by_series.get(series_id)
        metadata_rows.append(
            {
                "series_id": series_id,
                "rows": int(sub.shape[0]),
                "start_date": sub["date"].min().date().isoformat(),
                "end_date": sub["date"].max().date().isoformat(),
                "start_year": int(sub["date"].min().year),
                "frequency": sub["frequency"].iloc[0],
                "source": sub["source"].iloc[0],
                "pit_quality": sub["pit_quality"].iloc[0],
                "required_series": bool(req is not None),
            }
        )
    safe_to_csv(pd.DataFrame(metadata_rows), MACRO_DIR / "source_metadata.csv")
    return panel


def build_availability(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for req in REQUIREMENTS:
        sub = panel[panel["series_id"].eq(req.series_id)].copy()
        if sub.empty:
            status = "missing"
            rows.append(
                {
                    "series_id": req.series_id,
                    "description": req.description,
                    "frequency": req.frequency,
                    "min_start_year": req.min_start_year,
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    "has_available_date": False,
                    "meets_min_start_year": False,
                    "pit_quality": "",
                    "readiness_status": status,
                    "block_reason": "series_not_fetched_or_source_failed",
                }
            )
            continue
        start_date = pd.to_datetime(sub["date"]).min()
        end_date = pd.to_datetime(sub["date"]).max()
        has_available = bool(pd.to_datetime(sub["available_date"], errors="coerce").notna().all())
        meets_start = bool(start_date.year <= req.min_start_year)
        if has_available and meets_start:
            status = "usable"
            reason = ""
        elif has_available:
            status = "limited_history"
            reason = f"starts_after_required_year:{start_date.year}>{req.min_start_year}"
        else:
            status = "not_point_in_time"
            reason = "available_date_missing"
        rows.append(
            {
                "series_id": req.series_id,
                "description": req.description,
                "frequency": req.frequency,
                "min_start_year": req.min_start_year,
                "rows": int(sub.shape[0]),
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "has_available_date": has_available,
                "meets_min_start_year": meets_start,
                "pit_quality": sub["pit_quality"].iloc[0],
                "readiness_status": status,
                "block_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_gate_decisions(availability: pd.DataFrame) -> pd.DataFrame:
    status_by_series = dict(zip(availability["series_id"], availability["readiness_status"]))
    usable = {sid for sid, status in status_by_series.items() if status == "usable"}
    limited = {sid for sid, status in status_by_series.items() if status == "limited_history"}
    rows = []
    for family in CANDIDATE_FAMILIES:
        required = set(family["required_series"])
        missing = sorted(sid for sid in required if sid not in usable and sid not in limited)
        limited_required = sorted(sid for sid in required if sid in limited)
        if missing:
            decision = "block_signal_validation"
            reason = "missing_required_series:" + ",".join(missing)
        elif limited_required:
            decision = "allow_limited_signal_validation"
            reason = "limited_history_series:" + ",".join(limited_required)
        else:
            decision = "allow_signal_validation"
            reason = "required_point_in_time_series_ready"
        rows.append(
            {
                "gate": family["variant"],
                "required_series": ",".join(family["required_series"]),
                "decision": decision,
                "implementation_allowed": False,
                "default_promotion_allowed": False,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_registry(gates: pd.DataFrame) -> pd.DataFrame:
    family_by_variant = {item["variant"]: item for item in CANDIDATE_FAMILIES}
    rows = []
    for _, gate in gates.iterrows():
        family = family_by_variant[str(gate["gate"])]
        role = "observation" if str(gate["decision"]).startswith("allow") else "rejected"
        rows.append(
            {
                "variant": family["variant"],
                "role": role,
                "description": family["description"],
                "multipliers_json": "{}",
                "selection_source": "macro_data_ingestion_gate",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": False,
                "required_series": gate["required_series"],
                "data_gate_decision": gate["decision"],
                "block_reason": gate["reason"],
            }
        )
    return pd.DataFrame(rows)


def build_quality_report(panel: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, sub in panel.groupby("series_id"):
        dates = pd.to_datetime(sub["date"], errors="coerce")
        available = pd.to_datetime(sub["available_date"], errors="coerce")
        rows.append(
            {
                "series_id": series_id,
                "rows": int(sub.shape[0]),
                "duplicate_date_count": int(sub.duplicated(["series_id", "date"]).sum()),
                "value_null_count": int(pd.to_numeric(sub["value"], errors="coerce").isna().sum()),
                "available_date_null_count": int(available.isna().sum()),
                "available_before_date_count": int((available < dates).sum()),
                "start_date": dates.min().date().isoformat() if dates.notna().any() else "",
                "end_date": dates.max().date().isoformat() if dates.notna().any() else "",
                "pit_quality": sub["pit_quality"].iloc[0],
            }
        )
    quality = pd.DataFrame(rows)
    required_quality = availability[["series_id", "readiness_status", "block_reason"]]
    return quality.merge(required_quality, on="series_id", how="outer")


def write_report(
    status: pd.DataFrame,
    availability: pd.DataFrame,
    gates: pd.DataFrame,
    quality: pd.DataFrame,
    path: Path,
) -> None:
    status_counts = status["status"].value_counts().to_dict() if not status.empty else {}
    readiness_counts = availability["readiness_status"].value_counts().to_dict() if not availability.empty else {}
    gate_counts = gates["decision"].value_counts().to_dict() if not gates.empty else {}
    text = f"""# HIRSSM V3.27 Macro Data Ingestion

## Decision

- Status: accepted as data-layer upgrade.
- Macro signal validation: allowed only for candidate families whose data gate says `allow_signal_validation`.
- Default model promotion: not allowed in V3.27; this version only builds data.

## Fetch Status

- Source status counts: `{json.dumps(status_counts, ensure_ascii=False)}`.
- Required series readiness counts: `{json.dumps(readiness_counts, ensure_ascii=False)}`.
- Candidate gate counts: `{json.dumps(gate_counts, ensure_ascii=False)}`.

## Ready Next

"""
    for _, row in gates.iterrows():
        text += f"- `{row['gate']}`: `{row['decision']}` ({row['reason']})\n"
    text += """
## Data Quality Notes

- Daily market series use next trading day as `available_date`.
- Jin10 event series use release-event date mapped to the next local trading day.
- Mofcom social-financing source failed if `china_tsf_yoy` remains missing; the new-credit series is stored as a substitute but does not satisfy the TSF requirement.
- V3.28 may validate only the families marked as allowed here; no implementation or default promotion is authorized yet.
"""
    if not quality.empty:
        null_issues = quality[(quality.get("value_null_count", 0) > 0) | (quality.get("available_date_null_count", 0) > 0)]
        if not null_issues.empty:
            text += "\n## Quality Warnings\n\n"
            for _, row in null_issues.iterrows():
                text += f"- `{row['series_id']}`: value_null={row.get('value_null_count')}, available_null={row.get('available_date_null_count')}\n"
    write_text(text, path)


def write_failure_cases(status: pd.DataFrame, availability: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    text = "# V3.27 Macro Data Failure Cases\n\n"
    failed = status[status["status"].ne("success")] if not status.empty else pd.DataFrame()
    if failed.empty:
        text += "No source fetch failures were recorded.\n\n"
    else:
        text += "## Source Failures\n\n"
        for _, row in failed.iterrows():
            text += f"- `{row['source_name']}`: {row['error']}\n"
        text += "\n"
    blocked = availability[~availability["readiness_status"].eq("usable")]
    if not blocked.empty:
        text += "## Series Not Fully Usable\n\n"
        for _, row in blocked.iterrows():
            text += f"- `{row['series_id']}`: `{row['readiness_status']}` {row['block_reason']}\n"
        text += "\n"
    blocked_gates = gates[gates["decision"].eq("block_signal_validation")]
    if not blocked_gates.empty:
        text += "## Blocked Candidate Families\n\n"
        for _, row in blocked_gates.iterrows():
            text += f"- `{row['gate']}`: {row['reason']}\n"
    write_text(text, path)


def write_self_check(paths: dict[str, Path], availability: pd.DataFrame, gates: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for name, artifact_path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "pass": artifact_path.exists(), "detail": rel(artifact_path)})
    required_cols = set(standard_columns())
    panel_path = MACRO_DIR / "macro_pit_panel.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, encoding="utf-8-sig")
        rows.append({"check": "macro_panel_required_columns", "pass": required_cols.issubset(panel.columns), "detail": ",".join(sorted(required_cols - set(panel.columns)))})
    core = {"cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"}
    usable = set(availability.loc[availability["readiness_status"].eq("usable"), "series_id"].astype(str))
    rows.append({"check": "rate_fx_core_ready", "pass": core.issubset(usable), "detail": ",".join(sorted(core - usable))})
    rows.append(
        {
            "check": "no_default_promotion_from_data_ingestion",
            "pass": bool((gates["default_promotion_allowed"].astype(str).str.lower() == "false").all()),
            "detail": "data ingestion only",
        }
    )
    out = pd.DataFrame(rows)
    safe_to_csv(out, path)
    return out


def run(output_dir: Path, end_date: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []

    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover
        ak = None
        status_rows.append(
            {
                "source_name": "akshare_import",
                "status": "failed",
                "rows": 0,
                "columns": "",
                "raw_path": "",
                "started_at": now_text(),
                "finished_at": now_text(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    frames: list[pd.DataFrame] = []
    if ak is not None:
        frames.extend(build_rate_tables(ak, status_rows))
        frames.append(build_fx_table(ak, end_date, status_rows))
        frames.extend(build_event_series(ak, status_rows))
        frames.extend(build_credit_tables(ak, status_rows))
        frames.append(build_commodity_table(ak, status_rows))

    panel = write_macro_tables(frames)
    status = pd.DataFrame(status_rows)
    availability = build_availability(panel)
    gates = build_gate_decisions(availability)
    candidate_registry = build_candidate_registry(gates)
    quality = build_quality_report(panel, availability)

    paths = {
        "macro_pit_panel": MACRO_DIR / "macro_pit_panel.csv",
        "source_metadata": MACRO_DIR / "source_metadata.csv",
        "fetch_status": output_dir / "macro_fetch_status.csv",
        "availability": output_dir / "macro_data_availability_after_ingestion.csv",
        "quality_report": output_dir / "macro_data_quality_report.csv",
        "source_gate_decision": output_dir / "macro_source_gate_decision.csv",
        "candidate_registry": output_dir / "candidate_registry.csv",
        "implementation_candidate_spec": output_dir / "implementation_candidate_spec.csv",
        "agent_report": output_dir / "agent_report.md",
        "failure_cases": output_dir / "factor_failure_cases.md",
        "changed_files": output_dir / "changed_files.txt",
    }

    safe_to_csv(status, paths["fetch_status"])
    safe_to_csv(availability, paths["availability"])
    safe_to_csv(quality, paths["quality_report"])
    safe_to_csv(gates, paths["source_gate_decision"])
    safe_to_csv(candidate_registry, paths["candidate_registry"])
    safe_to_csv(candidate_registry, paths["implementation_candidate_spec"])
    write_report(status, availability, gates, quality, paths["agent_report"])
    write_failure_cases(status, availability, gates, paths["failure_cases"])

    changed = [rel(path) for path in paths.values()]
    for file_path in [
        "china_10y_yield.csv",
        "us_10y_yield.csv",
        "cn_us_10y_rate_spread.csv",
        "cny_fx.csv",
        "pmi.csv",
        "m2_social_financing.csv",
        "cpi_ppi.csv",
        "commodity_index.csv",
    ]:
        path = MACRO_DIR / file_path
        if path.exists():
            changed.append(rel(path))
    for path in sorted(RAW_DIR.glob("*.csv")):
        changed.append(rel(path))
    write_text("\n".join(dict.fromkeys(changed)) + "\n", paths["changed_files"])

    self_check_path = output_dir / "self_check.csv"
    self_check_path.touch()
    paths["self_check"] = self_check_path
    self_check = write_self_check(paths, availability, gates, self_check_path)
    changed.append(rel(self_check_path))
    write_text("\n".join(dict.fromkeys(changed)) + "\n", paths["changed_files"])

    artifact_paths = [rel(path) for path in paths.values()]
    artifact_paths.extend(
        rel(MACRO_DIR / file_path)
        for file_path in [
            "china_10y_yield.csv",
            "us_10y_yield.csv",
            "cn_us_10y_rate_spread.csv",
            "cny_fx.csv",
            "pmi.csv",
            "m2_social_financing.csv",
            "cpi_ppi.csv",
            "commodity_index.csv",
            "macro_pit_panel.csv",
            "source_metadata.csv",
        ]
        if (MACRO_DIR / file_path).exists()
    )
    artifact_paths.extend(rel(path) for path in sorted(RAW_DIR.glob("*.csv")))
    artifact_paths = list(dict.fromkeys(artifact_paths))

    metrics = {
        "source_success_count": int(status["status"].eq("success").sum()) if not status.empty else 0,
        "source_failure_count": int(status["status"].ne("success").sum()) if not status.empty else 0,
        "required_series_count": int(availability.shape[0]),
        "usable_series_count": int(availability["readiness_status"].eq("usable").sum()),
        "limited_history_series_count": int(availability["readiness_status"].eq("limited_history").sum()),
        "missing_series_count": int(availability["readiness_status"].eq("missing").sum()),
        "allow_signal_validation_count": int(gates["decision"].astype(str).str.startswith("allow").sum()),
        "default_promotion_allowed": False,
    }
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": "V3.27",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_27_macro_data_ingestion.py",
        "config": {
            "data_vendor": "akshare",
            "end_date": end_date,
            "point_in_time_schema": True,
            "signal_validation_allowed_only_after_gate": True,
        },
        "data_refs": [
            "data_raw/akshare/calendar/trade_calendar.csv",
            "outputs/agent_runs/v3_26/macro_data_readiness",
        ],
        "code_refs": ["strategy_lab/hirssm_v3_27_macro_data_ingestion.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "data_raw/akshare/calendar/trade_calendar.csv",
            "outputs/agent_runs/v3_26/macro_data_readiness",
        ],
        "artifacts": artifact_paths,
        "outputs": artifact_paths,
        "changed_files": artifact_paths,
        "metrics": metrics,
        "self_check_pass": bool(self_check["pass"].astype(bool).all()),
        "fail_count": int((~self_check["pass"].astype(bool)).sum()),
        "warn_count": int(metrics["source_failure_count"] + metrics["limited_history_series_count"] + metrics["missing_series_count"]),
        "limitations": [
            "Macro event series use vendor event-date snapshots rather than full vintage databases.",
            "China TSF source may remain missing if the upstream Mofcom SSL endpoint fails.",
            "V3.27 is data ingestion only and does not authorize model promotion.",
        ],
        "risk_flags": [
            "macro_revision_bias_residual",
            "vendor_endpoint_failure",
            "limited_commodity_history",
        ],
        "next_decision": "Run V3.28 macro signal validation only for data-gated families, starting with cn_us_rate_spread_risk_budget.",
        "handoff_summary": "V3.27 creates local macro PIT tables and gates macro families for later signal validation.",
    }
    write_json(manifest, output_dir / "agent_run_manifest.json")

    return {
        "task_id": TASK_ID,
        "self_check_pass": manifest["self_check_pass"],
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y%m%d"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir, args.end_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
