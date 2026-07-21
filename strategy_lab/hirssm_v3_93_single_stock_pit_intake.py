from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "quant_research_assistant_v3_93_single_stock_pit_intake.json"
SCRIPT_PATH = ROOT / "strategy_lab" / "hirssm_v3_93_single_stock_pit_intake.py"
TASK_BRIEF_PATH = ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260603_v3_93_single_stock_pit_intake.json"
TASK_BOARD_PATH = ROOT / "reports" / "AGENT_TASK_BOARD.md"


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stock_suffix(code: str) -> str:
    if code.startswith("6"):
        return "SH"
    if code.startswith(("4", "8")):
        return "BJ"
    return "SZ"


def ts_code(code: str) -> str:
    return f"{code}.{stock_suffix(code)}"


def load_stock_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
    if "asset" not in df.columns or "name" not in df.columns:
        return {}
    df["asset"] = df["asset"].astype(str).str.zfill(6)
    return dict(zip(df["asset"], df["name"].astype(str)))


def safe_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.date().isoformat()


def safe_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def qfq_price_coverage(code: str, path: Path) -> dict[str, Any]:
    row = {
        "object_id": code,
        "source": "akshare_daily_qfq",
        "source_path": rel_path(path),
        "file_exists": path.exists(),
        "row_count": 0,
        "start_date": "",
        "end_date": "",
        "available_date_min": "",
        "available_date_max": "",
        "has_available_date": False,
        "adjustment": "qfq",
        "duplicate_date_count": 0,
        "null_close_count": 0,
        "latest_research_allowed": False,
        "historical_price_state_allowed": False,
        "historical_total_return_allowed": False,
        "pit_status": "missing",
        "downstream_restriction": "missing file",
    }
    if not path.exists():
        return row
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
    row["row_count"] = int(len(df))
    if "date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["date"], errors="coerce")
        row["start_date"] = safe_date(dates.min())
        row["end_date"] = safe_date(dates.max())
        row["duplicate_date_count"] = int(df["date"].duplicated().sum())
    if "close" in df.columns:
        row["null_close_count"] = int(pd.to_numeric(df["close"], errors="coerce").isna().sum())
    if "adjust" in df.columns and not df.empty:
        adjust = df["adjust"].dropna().astype(str).unique().tolist()
        row["adjustment"] = ",".join(sorted(adjust)) if adjust else "qfq"
    row["latest_research_allowed"] = True
    row["pit_status"] = "research_only_no_row_available_date"
    row["downstream_restriction"] = "usable for latest technical profile; blocked as historical adjusted-return label until row-level available_date is controlled"
    return row


def init_raw_state(codes: list[str]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for code in codes:
        state[code] = {
            "row_count": 0,
            "trade_dates": set(),
            "available_dates": set(),
            "price_adjustments": set(),
            "data_sources": set(),
            "files_with_row": 0,
            "duplicate_trade_date_count": 0,
        }
    return state


def update_raw_state(state: dict[str, dict[str, Any]], code: str, row: dict[str, str]) -> None:
    bucket = state[code]
    trade_date = str(row.get("trade_date") or row.get("date") or "").strip()
    available_date = str(row.get("available_date") or "").strip()
    if trade_date in bucket["trade_dates"]:
        bucket["duplicate_trade_date_count"] += 1
    bucket["trade_dates"].add(trade_date)
    if available_date:
        bucket["available_dates"].add(available_date)
    adjustment = str(row.get("price_adjustment") or "").strip()
    if adjustment:
        bucket["price_adjustments"].add(adjustment)
    data_source = str(row.get("data_source") or "").strip()
    if data_source:
        bucket["data_sources"].add(data_source)
    bucket["row_count"] += 1
    bucket["files_with_row"] += 1


def parse_csv_line(header: list[str], line: str) -> dict[str, str]:
    values = next(csv.reader([line]))
    return dict(zip(header, values))


def scan_tushare_raw_daily(
    codes: list[str],
    daily_dir: Path,
    scan_mode: str,
    max_prefix_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = init_raw_state(codes)
    targets = set(codes)
    target_ts_codes = {ts_code(code): code for code in codes}
    max_code = max(targets)
    files = sorted(path for path in daily_dir.glob("trade_date=*.csv") if path.is_file())
    prefix_limit_hit = 0
    parse_errors = 0
    matched_rows = 0
    lines_scanned = 0
    for path in files:
        rows_read = 0
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header_line = handle.readline()
                if not header_line:
                    continue
                header = next(csv.reader([header_line]))
                for line in handle:
                    rows_read += 1
                    lines_scanned += 1
                    if scan_mode == "full_line_filter":
                        code = target_ts_codes.get(line[:9])
                        if not code:
                            continue
                        row = parse_csv_line(header, line)
                        update_raw_state(state, code, row)
                        matched_rows += 1
                        continue
                    row = parse_csv_line(header, line)
                    raw_code = str(row.get("ts_code") or row.get("asset") or "").strip()
                    code = raw_code[:6]
                    if code in targets:
                        update_raw_state(state, code, row)
                        matched_rows += 1
                    if scan_mode == "prefix_until_code_gt_max":
                        if code.isdigit() and code > max_code:
                            break
                        if rows_read >= max_prefix_rows:
                            prefix_limit_hit += 1
                            break
        except Exception:
            parse_errors += 1
    coverage: list[dict[str, Any]] = []
    for code in codes:
        bucket = state[code]
        trade_dates = sorted(value for value in bucket["trade_dates"] if value)
        available_dates = sorted(value for value in bucket["available_dates"] if value)
        row_count = int(bucket["row_count"])
        coverage.append(
            {
                "object_id": code,
                "source": "tushare_raw_daily",
                "source_path": rel_path(daily_dir),
                "file_exists": daily_dir.exists(),
                "row_count": row_count,
                "start_date": safe_date(trade_dates[0]) if trade_dates else "",
                "end_date": safe_date(trade_dates[-1]) if trade_dates else "",
                "available_date_min": safe_date(available_dates[0]) if available_dates else "",
                "available_date_max": safe_date(available_dates[-1]) if available_dates else "",
                "has_available_date": bool(available_dates) and len(available_dates) == row_count,
                "adjustment": ",".join(sorted(bucket["price_adjustments"])) if bucket["price_adjustments"] else "none_raw",
                "duplicate_date_count": int(bucket["duplicate_trade_date_count"]),
                "null_close_count": "",
                "latest_research_allowed": row_count > 0,
                "historical_price_state_allowed": row_count > 0 and bool(available_dates),
                "historical_total_return_allowed": False,
                "pit_status": "pit_raw_price_state_only" if row_count > 0 and bool(available_dates) else "missing_or_unverified",
                "downstream_restriction": "PIT raw daily OHLCV only; unadjusted, so not a total-return label and not suitable for long-horizon adjusted-return tests",
            }
        )
    meta = {
        "files_scanned": len(files),
        "scan_mode": scan_mode,
        "max_prefix_rows_per_file": max_prefix_rows,
        "prefix_limit_hit": prefix_limit_hit,
        "parse_errors": parse_errors,
        "matched_rows": matched_rows,
        "lines_scanned": lines_scanned,
    }
    return coverage, meta


def latest_financial_snapshot(code: str, path: Path) -> dict[str, Any]:
    base = {
        "object_id": code,
        "source": "akshare_financial_indicator",
        "source_path": rel_path(path),
        "file_exists": path.exists(),
        "report_period": "",
        "available_date": "",
        "fetched_at": "",
        "net_profit_raw": "",
        "revenue_raw": "",
        "eps_raw": "",
        "bvps_raw": "",
        "roe_raw": "",
        "debt_to_asset_raw": "",
        "pit_status": "missing",
        "latest_research_allowed": False,
        "historical_backtest_allowed": False,
        "confidence_cap_reason": "missing file",
    }
    if not path.exists():
        return base
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
    if df.empty or "报告期" not in df.columns:
        base["pit_status"] = "invalid_no_report_period"
        base["confidence_cap_reason"] = "missing report_period"
        return base
    df["_report_dt"] = pd.to_datetime(df["报告期"], errors="coerce")
    latest = df.sort_values("_report_dt").iloc[-1]
    base.update(
        {
            "report_period": safe_date(latest.get("报告期")),
            "fetched_at": str(latest.get("fetched_at") or ""),
            "net_profit_raw": str(latest.get("净利润") or ""),
            "revenue_raw": str(latest.get("营业总收入") or ""),
            "eps_raw": str(latest.get("基本每股收益") or ""),
            "bvps_raw": str(latest.get("每股净资产") or ""),
            "roe_raw": str(latest.get("净资产收益率") or ""),
            "debt_to_asset_raw": str(latest.get("资产负债率") or ""),
            "pit_status": "current_snapshot_no_available_date",
            "latest_research_allowed": True,
            "historical_backtest_allowed": False,
            "confidence_cap_reason": "missing available_date; fetched_at is collection timestamp, not market availability timestamp",
        }
    )
    return base


def build_universe(
    codes: list[str],
    names: dict[str, str],
    qfq_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    financial_rows: list[dict[str, Any]],
    asof_date: str,
) -> list[dict[str, Any]]:
    qfq_by_code = {row["object_id"]: row for row in qfq_rows}
    raw_by_code = {row["object_id"]: row for row in raw_rows}
    financial_by_code = {row["object_id"]: row for row in financial_rows}
    rows: list[dict[str, Any]] = []
    for code in codes:
        qfq = qfq_by_code[code]
        raw = raw_by_code[code]
        financial = financial_by_code[code]
        name = names.get(code, code)
        rows.append(
            {
                "object_id": code,
                "object_name": name,
                "object_type": "stock",
                "exchange_suffix": stock_suffix(code),
                "ts_code": ts_code(code),
                "asof_date": asof_date,
                "latest_trade_date": qfq["end_date"] or raw["end_date"],
                "qfq_price_path": qfq["source_path"],
                "raw_daily_path": raw["source_path"],
                "financial_indicator_path": financial["source_path"],
                "qfq_price_status": qfq["pit_status"],
                "raw_daily_status": raw["pit_status"],
                "financial_status": financial["pit_status"],
                "latest_technical_research_allowed": bool(qfq["latest_research_allowed"]),
                "pit_raw_price_state_allowed": bool(raw["historical_price_state_allowed"]),
                "latest_fundamental_research_allowed": bool(financial["latest_research_allowed"]),
                "historical_fundamental_backtest_allowed": bool(financial["historical_backtest_allowed"]),
                "research_use_status": "research_ready_with_boundaries" if qfq["latest_research_allowed"] or financial["latest_research_allowed"] else "missing_core_inputs",
                "special_risk_flag": "special_treatment_name" if "ST" in name.upper() else "",
            }
        )
    return rows


def build_pit_readiness(
    qfq_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    financial_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_rows in [qfq_rows, raw_rows]:
        for row in source_rows:
            if row["source"] == "akshare_daily_qfq":
                decision = "research_only"
            elif row["row_count"] and row["has_available_date"]:
                decision = "approved_limited_raw_price_state"
            else:
                decision = "blocked"
            rows.append(
                {
                    "object_id": row["object_id"],
                    "dataset": row["source"],
                    "status": decision,
                    "row_count": row["row_count"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "has_available_date": row["has_available_date"],
                    "adjustment": row["adjustment"],
                    "historical_price_state_allowed": row["historical_price_state_allowed"],
                    "historical_total_return_allowed": row["historical_total_return_allowed"],
                    "historical_fundamental_backtest_allowed": False,
                    "restriction": row["downstream_restriction"],
                }
            )
    for row in financial_rows:
        rows.append(
            {
                "object_id": row["object_id"],
                "dataset": row["source"],
                "status": "research_only",
                "row_count": 1 if row["file_exists"] else 0,
                "start_date": row["report_period"],
                "end_date": row["report_period"],
                "has_available_date": False,
                "adjustment": "not_price_data",
                "historical_price_state_allowed": False,
                "historical_total_return_allowed": False,
                "historical_fundamental_backtest_allowed": False,
                "restriction": row["confidence_cap_reason"],
            }
        )
    return rows


def build_gap_register(universe_rows: list[dict[str, Any]], readiness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in readiness_rows:
        if row["dataset"] == "akshare_financial_indicator" and not row["has_available_date"]:
            rows.append(
                {
                    "object_id": row["object_id"],
                    "dataset": row["dataset"],
                    "gap_type": "missing_available_date",
                    "severity": "high",
                    "owner_agent": "data_steward",
                    "impact": "blocks historical fundamental factor backtests and factor timing joins",
                    "repair_action": "ingest PIT financial announcement dates or vendor-provided financial statement vintage table",
                }
            )
        if row["dataset"] == "akshare_daily_qfq" and not row["has_available_date"]:
            rows.append(
                {
                    "object_id": row["object_id"],
                    "dataset": row["dataset"],
                    "gap_type": "missing_row_available_date",
                    "severity": "medium",
                    "owner_agent": "data_steward",
                    "impact": "blocks adjusted-return label use in historical tests unless data vintage is controlled",
                    "repair_action": "attach controlled collection date or replace with vendor adjusted daily data with PIT availability",
                }
            )
        if row["dataset"] == "tushare_raw_daily" and str(row["adjustment"]).lower() == "none_raw":
            rows.append(
                {
                    "object_id": row["object_id"],
                    "dataset": row["dataset"],
                    "gap_type": "unadjusted_price_series",
                    "severity": "medium",
                    "owner_agent": "data_steward",
                    "impact": "raw prices can distort long-horizon returns around splits/dividends/rights issues",
                    "repair_action": "add adj_factor/dividend/split data before total-return labels or long-horizon return factors",
                }
            )
    for row in universe_rows:
        if row["special_risk_flag"]:
            rows.append(
                {
                    "object_id": row["object_id"],
                    "dataset": "stock_master",
                    "gap_type": row["special_risk_flag"],
                    "severity": "medium",
                    "owner_agent": "fundamental_equity_analyst",
                    "impact": "special-treatment names require separate listing status and corporate event checks",
                    "repair_action": "ingest listing status, ST history, suspension and delisting event table",
                }
            )
    return rows


def build_boundary_audit(readiness_rows: list[dict[str, Any]], financial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in readiness_rows:
        failed: list[str] = []
        if row["dataset"] == "akshare_financial_indicator" and row["historical_fundamental_backtest_allowed"]:
            failed.append("financial_history_allowed_without_available_date")
        if row["dataset"] == "akshare_daily_qfq" and row["historical_total_return_allowed"]:
            failed.append("qfq_total_return_allowed_without_available_date")
        if row["dataset"] == "tushare_raw_daily" and row["historical_total_return_allowed"]:
            failed.append("raw_total_return_allowed_without_adjustment")
        if row["dataset"] == "tushare_raw_daily" and row["row_count"] and not row["has_available_date"]:
            failed.append("raw_daily_missing_available_date")
        rows.append(
            {
                "object_id": row["object_id"],
                "dataset": row["dataset"],
                "status": "pass" if not failed else "fail",
                "failed_checks": ";".join(failed),
                "research_decision": row["status"],
                "alpha_claim_allowed": False,
                "order_instruction_allowed": False,
                "portfolio_weight_allowed": False,
                "historical_backtest_allowed_scope": "raw_price_state_only" if row["historical_price_state_allowed"] else "blocked_or_latest_research_only",
            }
        )
    rows.append(
        {
            "object_id": "ALL",
            "dataset": "v3_93_intake_outputs",
            "status": "pass",
            "failed_checks": "",
            "research_decision": "data_readiness_only",
            "alpha_claim_allowed": False,
            "order_instruction_allowed": False,
            "portfolio_weight_allowed": False,
            "historical_backtest_allowed_scope": "no strategy backtest in V3.93",
        }
    )
    return rows


def build_data_dictionary() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "akshare_daily_qfq",
            "field": "date/open/high/low/close/volume/amount/adjust/fetched_at",
            "meaning": "front-adjusted daily OHLCV collected from AkShare",
            "pit_status": "no row-level available_date",
            "downstream_restriction": "latest technical profile only until availability is controlled",
        },
        {
            "dataset": "tushare_raw_daily",
            "field": "trade_date/open/high/low/close/pre_close/pct_chg/vol/amount/available_date/price_adjustment",
            "meaning": "Tushare daily raw unadjusted OHLCV with collection-side available_date",
            "pit_status": "PIT for raw daily price-state features when available_date is present",
            "downstream_restriction": "not total-return labels; not long-horizon adjusted returns without corporate-action adjustment",
        },
        {
            "dataset": "akshare_financial_indicator",
            "field": "报告期/净利润/营业总收入/基本每股收益/每股净资产/净资产收益率/资产负债率/fetched_at",
            "meaning": "current collected financial indicator history snapshot",
            "pit_status": "current snapshot; no available_date or announcement vintage",
            "downstream_restriction": "latest fundamental profile only; historical fundamental backtests blocked",
        },
    ]


def acceptance_rows(checks: list[tuple[str, bool, str]]) -> list[dict[str, Any]]:
    return [{"check": name, "status": "pass" if ok else "fail", "detail": detail} for name, ok, detail in checks]


def self_check_rows(paths: list[Path], checks: list[tuple[str, bool, str]]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "check": f"exists:{rel_path(path)}",
                "status": "pass" if path.exists() else "fail",
                "detail": "file exists" if path.exists() else "missing file",
            }
        )
    rows.extend(acceptance_rows(checks))
    return rows


def render_agent_report(
    universe_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    raw_scan_meta: dict[str, Any],
) -> str:
    status_counts = pd.DataFrame(readiness_rows)["status"].value_counts().to_dict() if readiness_rows else {}
    lines = [
        "# V3.93 Single-Stock PIT Intake Report",
        "",
        "Scope: data readiness only. This run does not evaluate alpha, portfolio weights, orders, or strategy returns.",
        "",
        f"- Stock count: {len(universe_rows)}",
        f"- Readiness status counts: {json.dumps(status_counts, ensure_ascii=False)}",
        f"- Data gap count: {len(gap_rows)}",
        f"- Tushare raw daily files scanned: {raw_scan_meta.get('files_scanned', 0)}",
        f"- Tushare scan mode: {raw_scan_meta.get('scan_mode', '')}",
        "",
        "Decision: single-stock research intake is accepted with boundaries. Historical fundamental backtests remain blocked until financial statement available_date or announcement vintage data is ingested. Adjusted return labels remain blocked until PIT-adjusted price data is controlled.",
        "",
    ]
    return "\n".join(lines)


def render_catalog(
    output_dir: Path,
    universe_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    raw_scan_meta: dict[str, Any],
) -> str:
    approved_limited = sum(row["status"] == "approved_limited_raw_price_state" for row in readiness_rows)
    research_only = sum(row["status"] == "research_only" for row in readiness_rows)
    blocked = sum(row["status"] == "blocked" for row in readiness_rows)
    lines = [
        "# A-share Single-Stock PIT Intake V3.93",
        "",
        "Purpose: register the first single-stock data readiness pilot for the quant research assistant framework.",
        "",
        "Strict boundary: this catalog describes data usability only. It is not a stock-picking model, backtest, trade signal, or portfolio recommendation.",
        "",
        "## Universe",
        "",
        f"- Stocks: {', '.join(row['object_id'] for row in universe_rows)}",
        f"- Output directory: `{rel_path(output_dir)}`",
        "",
        "## Source Decisions",
        "",
        f"- `tushare_raw_daily`: {approved_limited} stock-source rows approved only for PIT raw price-state features.",
        f"- `akshare_daily_qfq` and `akshare_financial_indicator`: {research_only} stock-source rows remain research-only.",
        f"- Blocked source rows: {blocked}",
        "",
        "## Key Data Restrictions",
        "",
        "- AkShare `financial_indicator` lacks `available_date`; it cannot enter historical fundamental factor backtests.",
        "- AkShare `daily_qfq` is adjusted but lacks row-level availability control; it is latest technical research data, not a governed historical adjusted-return label.",
        "- Tushare daily data has `available_date`, but is `none_raw`; it is isolated to raw price-state features until corporate-action adjustment data is added.",
        "",
        "## Scan Metadata",
        "",
        f"- Files scanned: {raw_scan_meta.get('files_scanned', 0)}",
        f"- Scan mode: {raw_scan_meta.get('scan_mode', '')}",
        f"- Prefix limit hits: {raw_scan_meta.get('prefix_limit_hit', 0)}",
        f"- Parse errors: {raw_scan_meta.get('parse_errors', 0)}",
        "",
        "## Artifacts",
        "",
        "- `single_stock_research_universe.csv`",
        "- `single_stock_price_coverage.csv`",
        "- `single_stock_fundamental_snapshot.csv`",
        "- `single_stock_pit_readiness.csv`",
        "- `single_stock_data_gap_register.csv`",
        "- `single_stock_boundary_audit.csv`",
        "- `single_stock_intake_decision.md`",
        "- `data_dictionary.csv`",
        "- `agent_run_manifest.json`",
        "",
        f"Data gap rows recorded: {len(gap_rows)}",
        "",
    ]
    return "\n".join(lines)


def write_manifest(
    out_dir: Path,
    config_path: Path,
    catalog_path: Path,
    artifacts: list[Path],
    outputs: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> Path:
    path = out_dir / "agent_run_manifest.json"
    manifest = {
        "run_id": "20260603_v3_93_single_stock_pit_intake_run",
        "task_id": "20260603_v3_93_single_stock_pit_intake",
        "agent": "data_steward",
        "version": "V3.93",
        "baseline": "V3.92 sample research run; V3.93 adds single-stock data readiness only, not alpha validation",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"python {rel_path(SCRIPT_PATH)} --config {rel_path(config_path)}",
        "config": {"path": rel_path(config_path)},
        "data_refs": [
            "data_raw/akshare/stock_list/",
            "data_raw/akshare/daily_qfq/",
            "data_raw/akshare/financial_indicator/",
            "data_raw/tushare_daily_only/v3_38/daily/",
        ],
        "code_refs": [rel_path(SCRIPT_PATH)],
        "output_dir": rel_path(out_dir),
        "allowed_inputs": [
            rel_path(config_path),
            "data_raw/akshare/stock_list/",
            "data_raw/akshare/daily_qfq/",
            "data_raw/akshare/financial_indicator/",
            "data_raw/tushare_daily_only/v3_38/daily/",
            "outputs/agent_runs/v3_92/sample_research_run/",
        ],
        "artifacts": [rel_path(path_item) for path_item in artifacts],
        "outputs": [rel_path(path_item) for path_item in outputs],
        "changed_files": [
            rel_path(SCRIPT_PATH),
            rel_path(config_path),
            rel_path(TASK_BRIEF_PATH),
            rel_path(TASK_BOARD_PATH),
            rel_path(catalog_path),
        ],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": int(fail_count),
        "warn_count": int(warn_count),
        "limitations": [
            "AkShare financial_indicator lacks available_date and is latest research only.",
            "AkShare qfq price is adjusted but lacks row-level available_date.",
            "Tushare daily-only raw price has available_date but is unadjusted.",
            "V3.93 performs data readiness only and does not validate alpha or portfolio performance.",
        ],
        "risk_flags": [
            "fundamental_history_not_pit_safe",
            "qfq_adjusted_label_not_pit_controlled",
            "raw_daily_not_total_return",
            "single_stock_special_treatment_history_missing",
        ],
        "next_decision": "handoff_to_fundamental_equity_analyst_after PIT-safe financial vintage data is acquired; otherwise use latest profile only",
        "handoff_summary": "single-stock data intake accepted with strict PIT and adjustment restrictions",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(config_path: Path) -> int:
    config = read_json(config_path)
    codes = [str(code).zfill(6) for code in config["stock_universe"]]
    akshare_root = ROOT / config["akshare_root"]
    out_dir = ROOT / config["output_dir"]
    catalog_path = ROOT / config["catalog_path"]
    ensure_dir(out_dir)

    names = load_stock_names(akshare_root / "stock_list" / "stock_info_a_code_name.csv")
    qfq_rows = [qfq_price_coverage(code, akshare_root / "daily_qfq" / f"{code}.csv") for code in codes]
    raw_rows, raw_scan_meta = scan_tushare_raw_daily(
        codes,
        ROOT / config["tushare_daily_dir"],
        str(config.get("tushare_daily_scan_mode", "full_line_filter")),
        int(config.get("tushare_max_prefix_rows_per_file", 80)),
    )
    financial_rows = [latest_financial_snapshot(code, akshare_root / "financial_indicator" / f"{code}.csv") for code in codes]
    universe_rows = build_universe(codes, names, qfq_rows, raw_rows, financial_rows, str(config.get("asof_date", "")))
    price_rows = [*qfq_rows, *raw_rows]
    readiness_rows = build_pit_readiness(qfq_rows, raw_rows, financial_rows)
    gap_rows = build_gap_register(universe_rows, readiness_rows)
    boundary_rows = build_boundary_audit(readiness_rows, financial_rows)
    dictionary_rows = build_data_dictionary()

    paths = {
        "universe": out_dir / "single_stock_research_universe.csv",
        "price_coverage": out_dir / "single_stock_price_coverage.csv",
        "fundamental_snapshot": out_dir / "single_stock_fundamental_snapshot.csv",
        "pit_readiness": out_dir / "single_stock_pit_readiness.csv",
        "gap_register": out_dir / "single_stock_data_gap_register.csv",
        "boundary_audit": out_dir / "single_stock_boundary_audit.csv",
        "decision": out_dir / "single_stock_intake_decision.md",
        "agent_report": out_dir / "agent_report.md",
        "data_dictionary": out_dir / "data_dictionary.csv",
        "acceptance": out_dir / "acceptance_checks.csv",
        "self_check": out_dir / "self_check.csv",
    }
    write_csv(paths["universe"], universe_rows)
    write_csv(paths["price_coverage"], price_rows)
    write_csv(paths["fundamental_snapshot"], financial_rows)
    write_csv(paths["pit_readiness"], readiness_rows)
    write_csv(paths["gap_register"], gap_rows)
    write_csv(paths["boundary_audit"], boundary_rows)
    write_csv(paths["data_dictionary"], dictionary_rows)
    write_text(paths["agent_report"], render_agent_report(universe_rows, readiness_rows, gap_rows, raw_scan_meta))
    write_text(
        paths["decision"],
        "\n".join(
            [
                "# V3.93 Single-Stock Intake Decision",
                "",
                "Decision: accepted as a data-readiness pilot only.",
                "",
                "Allowed now:",
                "- Latest stock technical profile from AkShare qfq prices, with no historical adjusted-return claim.",
                "- PIT raw price-state research from Tushare daily-only rows when `available_date` is present.",
                "- Latest fundamental profile from AkShare financial snapshots.",
                "",
                "Blocked:",
                "- Historical fundamental factor backtests from AkShare financial snapshots.",
                "- Adjusted-return labels from AkShare qfq prices without row-level availability control.",
                "- Total-return or long-horizon return tests from Tushare raw unadjusted prices.",
                "- Any alpha, order, portfolio, or strategy-performance claim in V3.93.",
                "",
            ]
        ),
    )
    write_text(catalog_path, render_catalog(out_dir, universe_rows, readiness_rows, gap_rows, raw_scan_meta))

    boundary_fail_count = sum(str(row["status"]).lower() == "fail" for row in boundary_rows)
    checks = [
        ("universe_count_matches_config", len(universe_rows) == len(codes), f"{len(universe_rows)} rows for {len(codes)} configured stocks"),
        ("qfq_price_files_exist", all(row["file_exists"] for row in qfq_rows), "all qfq files present"),
        ("financial_snapshot_files_exist", all(row["file_exists"] for row in financial_rows), "all financial snapshot files present"),
        (
            "financials_blocked_from_historical_backtest",
            all(not row["historical_backtest_allowed"] for row in financial_rows),
            "financial snapshots remain latest-profile only",
        ),
        (
            "qfq_blocked_from_historical_total_return_without_available_date",
            all(not row["historical_total_return_allowed"] for row in qfq_rows),
            "qfq adjusted prices are not promoted to PIT total-return labels",
        ),
        (
            "raw_daily_available_date_present_when_rows_exist",
            all((not row["row_count"]) or row["has_available_date"] for row in raw_rows),
            "Tushare raw rows with coverage have available_date",
        ),
        ("data_gaps_recorded", len(gap_rows) > 0, f"{len(gap_rows)} gap rows"),
        ("boundary_audit_pass", boundary_fail_count == 0, f"{boundary_fail_count} boundary failures"),
        ("catalog_written", catalog_path.exists(), rel_path(catalog_path)),
    ]
    write_csv(paths["acceptance"], acceptance_rows(checks))
    artifacts_without_manifest = [*paths.values(), catalog_path]
    write_csv(paths["self_check"], self_check_rows(artifacts_without_manifest, checks))

    acceptance_fail_count = sum(not ok for _, ok, _ in checks)
    fail_count = int(acceptance_fail_count + boundary_fail_count)
    warn_count = len(gap_rows)
    metrics = {
        "stock_count": len(codes),
        "qfq_source_rows": len(qfq_rows),
        "raw_source_rows": len(raw_rows),
        "financial_source_rows": len(financial_rows),
        "readiness_row_count": len(readiness_rows),
        "gap_count": len(gap_rows),
        "boundary_fail_count": boundary_fail_count,
        "acceptance_fail_count": acceptance_fail_count,
        "raw_daily_files_scanned": raw_scan_meta.get("files_scanned", 0),
        "raw_daily_parse_errors": raw_scan_meta.get("parse_errors", 0),
    }
    manifest_path = write_manifest(
        out_dir,
        config_path,
        catalog_path,
        [*artifacts_without_manifest, out_dir / "agent_run_manifest.json"],
        [*artifacts_without_manifest, out_dir / "agent_run_manifest.json"],
        metrics,
        fail_count,
        warn_count,
    )
    write_csv(paths["self_check"], self_check_rows([*artifacts_without_manifest, manifest_path], checks))
    print(
        json.dumps(
            {
                "version": config["version"],
                "task_id": config["task_id"],
                "output_dir": rel_path(out_dir),
                "fail_count": fail_count,
                "warn_count": warn_count,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()
    return run(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
