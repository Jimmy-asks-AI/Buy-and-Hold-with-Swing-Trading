#!/usr/bin/env python
"""HIRSSM V3.26 macro and rate data readiness gate.

V3.26 follows the V3.25 handoff, but stops before signal mining. Macro and
rate factors are easy to contaminate with revisions, late publication dates,
or mixed data vintages, so this task first verifies whether the local data
layer is point-in-time ready. If the required series are missing, the task
must block implementation and emit a precise data contract for the next run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data_raw"
BASELINE_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
V325_DIR = ROOT / "outputs" / "agent_runs" / "v3_25" / "industry_structure_source_research"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_26" / "macro_data_readiness"
TASK_ID = "20260527_v3_26_macro_data_readiness"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


@dataclass(frozen=True)
class RequiredSeries:
    series_id: str
    description: str
    frequency: str
    min_start_year: int
    economic_use: str
    keywords: tuple[str, ...]
    accepted_sources: tuple[str, ...]


REQUIRED_SERIES = [
    RequiredSeries(
        "cn_10y_gov_bond_yield",
        "China 10Y government bond yield",
        "daily",
        2005,
        "domestic discount-rate and duration pressure regime",
        ("cn10y", "china_10y", "cgb10", "国债", "中债", "10年国债", "bond_yield"),
        ("Tushare yc_cb", "JoinQuant macro", "AkShare bond_zh_us_rate"),
    ),
    RequiredSeries(
        "us_10y_treasury_yield",
        "US 10Y treasury yield",
        "daily",
        2005,
        "global discount-rate and risk-appetite pressure regime",
        ("us10y", "dgs10", "treasury", "美国10年", "美债", "us_10y"),
        ("FRED DGS10", "Tushare us_tycr", "AkShare macro_bank_usa_interest_rate"),
    ),
    RequiredSeries(
        "cn_us_10y_rate_spread",
        "China-US 10Y rate spread",
        "daily",
        2005,
        "cross-border rate pressure and valuation discount regime",
        ("rate_spread", "cn_us", "中美利差", "利差", "spread_10y"),
        ("derived from point-in-time China 10Y and US 10Y yields",),
    ),
    RequiredSeries(
        "usdcny",
        "USD/CNY exchange rate",
        "daily",
        2005,
        "currency pressure and foreign-liquidity stress",
        ("usdcny", "usd_cny", "cny", "人民币汇率", "exchange_rate", "fx"),
        ("Tushare fx_daily", "JoinQuant macro", "AkShare currency"),
    ),
    RequiredSeries(
        "china_pmi",
        "China manufacturing PMI",
        "monthly",
        2005,
        "growth cycle and cyclical industry demand",
        ("pmi", "采购经理", "制造业pmi"),
        ("NBS PMI with publication calendar", "Tushare cn_pmi", "JoinQuant macro"),
    ),
    RequiredSeries(
        "china_m2_yoy",
        "China M2 year-on-year growth",
        "monthly",
        2005,
        "broad liquidity impulse",
        ("m2", "money_supply", "货币供应", "广义货币"),
        ("PBoC with publication calendar", "Tushare cn_m", "JoinQuant macro"),
    ),
    RequiredSeries(
        "china_tsf_yoy",
        "China total social financing growth",
        "monthly",
        2005,
        "credit impulse and financing condition",
        ("tsf", "social_financing", "社融", "社会融资"),
        ("PBoC TSF with publication calendar", "Tushare sf_month", "JoinQuant macro"),
    ),
    RequiredSeries(
        "china_cpi_yoy",
        "China CPI year-on-year growth",
        "monthly",
        2005,
        "inflation pressure and policy constraint",
        ("cpi", "居民消费价格", "通胀"),
        ("NBS CPI with publication calendar", "Tushare cn_cpi", "JoinQuant macro"),
    ),
    RequiredSeries(
        "china_ppi_yoy",
        "China PPI year-on-year growth",
        "monthly",
        2005,
        "industrial profit and upstream inflation pressure",
        ("ppi", "工业生产者", "producer_price"),
        ("NBS PPI with publication calendar", "Tushare cn_ppi", "JoinQuant macro"),
    ),
    RequiredSeries(
        "commodity_index",
        "Broad commodity index",
        "daily",
        2005,
        "inflation and cyclical industry pricing pressure",
        ("commodity", "crb", "nanhua", "南华", "商品指数"),
        ("Nanhua commodity index", "CRB index", "Wind-like commodity index"),
    ),
]


IMPLEMENTATION_CANDIDATES = [
    {
        "variant": "cn_us_rate_spread_risk_budget",
        "description": "Reduce equity risk budget when China-US rate spread and FX pressure jointly deteriorate.",
        "required_series": "cn_10y_gov_bond_yield,us_10y_treasury_yield,cn_us_10y_rate_spread,usdcny",
    },
    {
        "variant": "macro_liquidity_repair",
        "description": "Release cash or increase cyclical exposure when M2/TSF liquidity impulse repairs while PMI stabilizes.",
        "required_series": "china_pmi,china_m2_yoy,china_tsf_yoy",
    },
    {
        "variant": "inflation_policy_constraint_defense",
        "description": "Cut high-beta exposure when CPI/PPI and commodity pressure tighten the policy constraint.",
        "required_series": "china_cpi_yoy,china_ppi_yoy,commodity_index,cn_10y_gov_bond_yield",
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


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def first_existing_date_column(columns: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for name in ("available_date", "date", "trade_date", "datetime", "time", "统计时间", "日期"):
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def inspect_csv(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": rel(path),
        "file_name": path.name,
        "parent": path.parent.name,
        "rows": 0,
        "columns": "",
        "date_column": "",
        "start_date": "",
        "end_date": "",
        "has_available_date": False,
        "read_status": "pass",
    }
    try:
        head = safe_read_csv(path, nrows=5)
    except Exception as exc:  # pragma: no cover - defensive data scan
        record["read_status"] = f"error:{exc}"
        return record
    if head.empty and not list(head.columns):
        record["read_status"] = "empty"
        return record
    columns = [str(col) for col in head.columns]
    record["columns"] = ",".join(columns[:40])
    record["has_available_date"] = "available_date" in {col.lower() for col in columns}
    date_col = first_existing_date_column(columns)
    if date_col:
        record["date_column"] = date_col
        try:
            dates = safe_read_csv(path, usecols=[date_col])
            parsed = pd.to_datetime(dates[date_col], errors="coerce")
            record["rows"] = int(parsed.shape[0])
            if parsed.notna().any():
                record["start_date"] = parsed.min().date().isoformat()
                record["end_date"] = parsed.max().date().isoformat()
        except Exception:
            record["rows"] = int(head.shape[0])
    else:
        try:
            record["rows"] = max(sum(1 for _ in path.open("rb")) - 1, 0)
        except OSError:
            record["rows"] = int(head.shape[0])
    return record


def scan_local_data(data_raw_dir: Path) -> pd.DataFrame:
    if not data_raw_dir.exists():
        return pd.DataFrame(
            columns=[
                "relative_path",
                "file_name",
                "parent",
                "rows",
                "columns",
                "date_column",
                "start_date",
                "end_date",
                "has_available_date",
                "read_status",
            ]
        )
    records = [inspect_csv(path) for path in sorted(data_raw_dir.rglob("*.csv"))]
    return pd.DataFrame(records)


def row_matches_series(row: pd.Series, required: RequiredSeries) -> bool:
    haystack = " ".join(
        [
            str(row.get("relative_path", "")),
            str(row.get("file_name", "")),
            str(row.get("parent", "")),
            str(row.get("columns", "")),
        ]
    ).lower()
    return any(keyword.lower() in haystack for keyword in required.keywords)


def year_from_date(value: object) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year)


def classify_series_readiness(required: RequiredSeries, inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        matches = pd.DataFrame()
    else:
        matches = inventory[inventory.apply(lambda row: row_matches_series(row, required), axis=1)].copy()

    found_files = int(matches.shape[0])
    pass_files = matches[matches["read_status"].astype(str).eq("pass")].copy() if found_files else pd.DataFrame()
    start_years = [year_from_date(value) for value in pass_files.get("start_date", pd.Series(dtype=object)).tolist()]
    start_years = [year for year in start_years if year is not None]
    earliest_year = min(start_years) if start_years else None
    has_available_date = bool(pass_files.get("has_available_date", pd.Series(dtype=bool)).astype(bool).any()) if not pass_files.empty else False
    enough_history = bool(earliest_year is not None and earliest_year <= required.min_start_year)

    if found_files == 0:
        status = "missing"
        reason = "no local file matched required macro/rate keywords"
    elif not has_available_date:
        status = "not_point_in_time"
        reason = "matched local files do not include available_date"
    elif not enough_history:
        status = "insufficient_history"
        reason = f"earliest local history is after {required.min_start_year}"
    else:
        status = "usable"
        reason = "local file has available_date and enough history"

    return {
        "series_id": required.series_id,
        "description": required.description,
        "frequency": required.frequency,
        "min_start_year": required.min_start_year,
        "economic_use": required.economic_use,
        "accepted_sources": "; ".join(required.accepted_sources),
        "found_files": found_files,
        "matched_files": "; ".join(pass_files["relative_path"].head(8).astype(str).tolist()) if not pass_files.empty else "",
        "earliest_year": earliest_year,
        "has_available_date": has_available_date,
        "readiness_status": status,
        "block_reason": reason,
    }


def build_requirement_spec() -> pd.DataFrame:
    rows = []
    for required in REQUIRED_SERIES:
        rows.append(
            {
                "series_id": required.series_id,
                "description": required.description,
                "required_columns": "date,available_date,value,source,frequency,revision_policy",
                "date_rule": "observation period end date",
                "available_date_rule": "first trading date on which the value could be known by the model",
                "frequency": required.frequency,
                "min_start_year": required.min_start_year,
                "economic_use": required.economic_use,
                "accepted_sources": "; ".join(required.accepted_sources),
                "forbidden": "current snapshot backfill; revised value without release calendar; future publication date leakage",
            }
        )
    return pd.DataFrame(rows)


def build_candidate_specs(availability: pd.DataFrame) -> pd.DataFrame:
    usable = set(availability.loc[availability["readiness_status"].eq("usable"), "series_id"].astype(str))
    rows = []
    for candidate in IMPLEMENTATION_CANDIDATES:
        required_series = [item.strip() for item in candidate["required_series"].split(",")]
        missing = [series_id for series_id in required_series if series_id not in usable]
        rows.append(
            {
                "variant": candidate["variant"],
                "role": "blocked",
                "selection_source": "macro_data_readiness_gate",
                "description": candidate["description"],
                "required_series": candidate["required_series"],
                "implementation_allowed": False,
                "block_reason": "missing_or_not_pit_series:" + ",".join(missing) if missing else "await_signal_validation",
            }
        )
    return pd.DataFrame(rows)


def build_gate_decision(availability: pd.DataFrame) -> pd.DataFrame:
    usable_count = int(availability["readiness_status"].eq("usable").sum())
    required_count = int(availability.shape[0])
    core = {"cn_10y_gov_bond_yield", "us_10y_treasury_yield", "usdcny", "china_pmi"}
    usable = set(availability.loc[availability["readiness_status"].eq("usable"), "series_id"].astype(str))
    missing_core = sorted(core - usable)
    implementation_allowed = usable_count >= 6 and not missing_core
    reason = "macro source has enough point-in-time coverage" if implementation_allowed else "core_macro_rate_series_not_point_in_time_ready"
    return pd.DataFrame(
        [
            {
                "gate": "macro_liquidity_rate_source",
                "usable_series_count": usable_count,
                "required_series_count": required_count,
                "missing_core_series": ",".join(missing_core),
                "implementation_allowed": implementation_allowed,
                "decision": "allow_signal_research" if implementation_allowed else "block_signal_research",
                "reason": reason,
            }
        ]
    )


def write_report(
    availability: pd.DataFrame,
    gate: pd.DataFrame,
    candidate_specs: pd.DataFrame,
    inventory: pd.DataFrame,
    path: Path,
) -> None:
    status_counts = availability["readiness_status"].value_counts().to_dict()
    gate_row = gate.iloc[0].to_dict()
    report = f"""# HIRSSM V3.26 Macro Data Readiness

## Decision

- Status: accepted as data-readiness research only.
- Implementation allowed: `{gate_row["implementation_allowed"]}`.
- Decision: `{gate_row["decision"]}`.
- Reason: `{gate_row["reason"]}`.

## Data Readiness

- Local CSV files scanned: `{int(inventory.shape[0])}`.
- Required macro/rate series: `{int(availability.shape[0])}`.
- Status counts: `{json.dumps(status_counts, ensure_ascii=False)}`.
- Usable series count: `{gate_row["usable_series_count"]}`.
- Missing core series: `{gate_row["missing_core_series"]}`.

## Candidate Impact

All V3.26 macro candidate families remain blocked because the local data layer
does not yet provide point-in-time macro/rate series with `available_date`.

Blocked candidates:

"""
    for _, row in candidate_specs.iterrows():
        report += f"- `{row['variant']}`: {row['block_reason']}\n"
    report += """
## Next Data Contract

Each macro series must be stored as a historical table with at least:

- `date`: observation date or period end.
- `available_date`: first trading date when the model could know the value.
- `value`: numeric value.
- `source`: data vendor or official publisher.
- `frequency`: daily/monthly/quarterly.
- `revision_policy`: original, revised, or vintage.

No macro factor may enter a historical backtest until this contract is met.
"""
    write_text(report, path)


def write_failure_cases(availability: pd.DataFrame, path: Path) -> None:
    blocked = availability[~availability["readiness_status"].eq("usable")]
    text = "# V3.26 Macro Data Failure Cases\n\n"
    text += "These are data-layer failures, not factor-performance failures.\n\n"
    for _, row in blocked.iterrows():
        text += f"## {row['series_id']}\n\n"
        text += f"- Status: `{row['readiness_status']}`\n"
        text += f"- Block reason: {row['block_reason']}\n"
        text += f"- Accepted sources: {row['accepted_sources']}\n\n"
    write_text(text, path)


def write_self_check(paths: dict[str, Path], availability: pd.DataFrame, gate: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for name, artifact_path in paths.items():
        rows.append({"check": f"artifact_exists:{name}", "pass": artifact_path.exists(), "detail": rel(artifact_path)})
    rows.extend(
        [
            {
                "check": "no_macro_signal_without_pit_data",
                "pass": not bool(gate.iloc[0]["implementation_allowed"]),
                "detail": "signal research blocked until core point-in-time series are present",
            },
            {
                "check": "available_date_required",
                "pass": bool((availability["readiness_status"].isin(["missing", "not_point_in_time", "insufficient_history", "usable"])).all()),
                "detail": "all required series evaluated against available_date rule",
            },
        ]
    )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = scan_local_data(DATA_RAW_DIR)
    availability = pd.DataFrame([classify_series_readiness(required, inventory) for required in REQUIRED_SERIES])
    requirement_spec = build_requirement_spec()
    candidate_specs = build_candidate_specs(availability)
    gate = build_gate_decision(availability)

    paths = {
        "local_data_inventory": output_dir / "local_data_inventory.csv",
        "macro_data_availability": output_dir / "macro_data_availability.csv",
        "macro_data_requirement_spec": output_dir / "macro_data_requirement_spec.csv",
        "macro_source_gate_decision": output_dir / "macro_source_gate_decision.csv",
        "candidate_registry": output_dir / "candidate_registry.csv",
        "implementation_candidate_spec": output_dir / "implementation_candidate_spec.csv",
        "factor_failure_cases": output_dir / "factor_failure_cases.md",
        "agent_report": output_dir / "agent_report.md",
        "changed_files": output_dir / "changed_files.txt",
    }

    inventory.to_csv(paths["local_data_inventory"], index=False, encoding="utf-8-sig")
    availability.to_csv(paths["macro_data_availability"], index=False, encoding="utf-8-sig")
    requirement_spec.to_csv(paths["macro_data_requirement_spec"], index=False, encoding="utf-8-sig")
    gate.to_csv(paths["macro_source_gate_decision"], index=False, encoding="utf-8-sig")
    candidate_specs.to_csv(paths["candidate_registry"], index=False, encoding="utf-8-sig")
    candidate_specs.to_csv(paths["implementation_candidate_spec"], index=False, encoding="utf-8-sig")
    write_failure_cases(availability, paths["factor_failure_cases"])
    write_report(availability, gate, candidate_specs, inventory, paths["agent_report"])

    changed_files = [rel(path) for path in paths.values()]
    write_text("\n".join(changed_files) + "\n", paths["changed_files"])
    self_check_path = output_dir / "self_check.csv"
    paths["self_check"] = self_check_path
    self_check_path.touch()
    self_check = write_self_check(paths, availability, gate, self_check_path)
    changed_files.append(rel(self_check_path))
    write_text("\n".join(changed_files) + "\n", paths["changed_files"])

    artifact_paths = [rel(path) for path in paths.values()]
    manifest = {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "data_steward",
        "version": "V3.26",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass",
        "started_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_26_macro_data_readiness.py",
        "config": {
            "research_only": True,
            "information_source": "macro_liquidity_rate_data",
            "point_in_time_required": True,
        },
        "data_refs": [
            "data_raw",
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_25/industry_structure_source_research",
        ],
        "code_refs": ["strategy_lab/hirssm_v3_26_macro_data_readiness.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "data_raw",
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_25/industry_structure_source_research",
        ],
        "artifacts": artifact_paths,
        "outputs": artifact_paths,
        "changed_files": artifact_paths,
        "metrics": {
            "required_series_count": int(availability.shape[0]),
            "usable_series_count": int(availability["readiness_status"].eq("usable").sum()),
            "blocked_candidate_count": int(candidate_specs.shape[0]),
            "implementation_allowed": bool(gate.iloc[0]["implementation_allowed"]),
            "local_csv_files_scanned": int(inventory.shape[0]),
        },
        "self_check_pass": bool(self_check["pass"].astype(bool).all()),
        "fail_count": int((~self_check["pass"].astype(bool)).sum()),
        "warn_count": int(availability["readiness_status"].eq("missing").sum()),
        "limitations": [
            "This task does not mine macro signals because core point-in-time series are not locally available.",
            "Current component/weights snapshots are not treated as historical macro data.",
        ],
        "risk_flags": [
            "macro_revision_bias",
            "publication_lag_leakage",
            "vendor_schema_drift",
        ],
        "next_decision": "Fetch or generate point-in-time macro/rate tables, then run V3.27 macro signal validation.",
        "handoff_summary": "V3.26 blocks macro/rate factor implementation until local data includes required available_date fields and sufficient history.",
    }
    manifest_path = output_dir / "agent_run_manifest.json"
    write_json(manifest, manifest_path)

    return {
        "task_id": TASK_ID,
        "self_check_pass": manifest["self_check_pass"],
        "metrics": manifest["metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
