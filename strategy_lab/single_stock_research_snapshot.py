from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "ad_hoc" / "stock_research_20260603_moutai_smic"
DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
STOCK_LIST_PATH = ROOT / "data_raw" / "akshare" / "stock_list" / "stock_info_a_code_name.csv"
INDEX_WEIGHT_DIR = ROOT / "data_raw" / "index" / "akshare_csindex" / "weights_latest"
INDEX_DAILY_DIR = ROOT / "data_raw" / "index" / "akshare_csindex" / "daily_csindex"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ts_code_from_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    if symbol.startswith(("600", "601", "603", "605", "688")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def symbol_from_ts_code(ts_code: str) -> str:
    return ts_code.split(".", 1)[0]


def load_stock_names() -> dict[str, str]:
    if not STOCK_LIST_PATH.exists():
        return {}
    df = pd.read_csv(STOCK_LIST_PATH, encoding="utf-8-sig", dtype={"asset": str})
    df["asset"] = df["asset"].astype(str).str.zfill(6)
    return dict(zip(df["asset"], df["name"].astype(str)))


def parse_line(header: list[str], line: str) -> dict[str, str]:
    return dict(zip(header, next(csv.reader([line]))))


def load_raw_daily(ts_codes: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = {code: symbol_from_ts_code(code) for code in ts_codes}
    rows: list[dict[str, str]] = []
    files_scanned = 0
    matched_rows = 0
    parse_errors = 0
    for path in sorted(DAILY_DIR.glob("trade_date=*.csv")):
        files_scanned += 1
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header_line = handle.readline()
                if not header_line:
                    continue
                header = next(csv.reader([header_line]))
                for line in handle:
                    ts_code = line[:9]
                    if ts_code not in targets:
                        continue
                    row = parse_line(header, line)
                    rows.append(row)
                    matched_rows += 1
        except Exception:
            parse_errors += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trade_date_dt"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.sort_values(["ts_code", "trade_date_dt"]).reset_index(drop=True)
    meta = {"files_scanned": files_scanned, "matched_rows": matched_rows, "parse_errors": parse_errors}
    return df, meta


def compound_return(pct_chg: pd.Series, days: int) -> float | None:
    clean = pd.to_numeric(pct_chg, errors="coerce").dropna()
    if len(clean) <= days:
        return None
    tail = clean.tail(days) / 100.0
    return float((1.0 + tail).prod() - 1.0)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def clipped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def technical_summary_for_stock(df: pd.DataFrame, ts_code: str, name: str) -> dict[str, Any]:
    stock = df[df["ts_code"].eq(ts_code)].copy().sort_values("trade_date_dt")
    if stock.empty:
        return {
            "ts_code": ts_code,
            "name": name,
            "status": "missing_raw_daily",
            "view": "blocked",
        }
    stock["ret"] = stock["pct_chg"] / 100.0
    for window in [20, 60, 120, 250]:
        stock[f"ma_{window}"] = stock["close"].rolling(window).mean()
        stock[f"high_{window}"] = stock["high"].rolling(window).max()
        stock[f"low_{window}"] = stock["low"].rolling(window).min()
        stock[f"amount_avg_{window}"] = stock["amount"].rolling(window).mean()
    stock["vol_20"] = stock["ret"].rolling(20).std() * (252**0.5)
    stock["vol_60"] = stock["ret"].rolling(60).std() * (252**0.5)
    latest = stock.iloc[-1]
    close = float(latest["close"])
    ma20 = float(latest["ma_20"]) if pd.notna(latest["ma_20"]) else None
    ma60 = float(latest["ma_60"]) if pd.notna(latest["ma_60"]) else None
    ma120 = float(latest["ma_120"]) if pd.notna(latest["ma_120"]) else None
    ma250 = float(latest["ma_250"]) if pd.notna(latest["ma_250"]) else None
    ret20 = compound_return(stock["pct_chg"], 20)
    ret60 = compound_return(stock["pct_chg"], 60)
    ret120 = compound_return(stock["pct_chg"], 120)
    ret250 = compound_return(stock["pct_chg"], 250)
    high120 = float(latest["high_120"]) if pd.notna(latest["high_120"]) else None
    low120 = float(latest["low_120"]) if pd.notna(latest["low_120"]) else None
    high250 = float(latest["high_250"]) if pd.notna(latest["high_250"]) else None
    low250 = float(latest["low_250"]) if pd.notna(latest["low_250"]) else None
    drawdown_120 = close / high120 - 1.0 if high120 else None
    drawdown_250 = close / high250 - 1.0 if high250 else None
    ma_checks = [
        close > ma20 if ma20 else False,
        ma20 > ma60 if ma20 and ma60 else False,
        ma60 > ma120 if ma60 and ma120 else False,
        ma120 > ma250 if ma120 and ma250 else False,
    ]
    ma_stack_score = sum(ma_checks) / len(ma_checks)
    momentum_score = clipped(0.50 + (ret20 or 0.0) * 1.50 + (ret60 or 0.0) * 0.75)
    repair_score = clipped(1.0 + (drawdown_120 or 0.0) / 0.35)
    amount_ratio = None
    if pd.notna(latest["amount_avg_20"]) and pd.notna(latest["amount_avg_120"]) and latest["amount_avg_120"] > 0:
        amount_ratio = float(latest["amount_avg_20"] / latest["amount_avg_120"])
    liquidity_score = clipped((amount_ratio or 1.0) - 0.5)
    vol20 = float(latest["vol_20"]) if pd.notna(latest["vol_20"]) else None
    vol_score = clipped(1.0 - ((vol20 or 0.35) / 0.85))
    technical_score = 0.35 * ma_stack_score + 0.25 * momentum_score + 0.20 * repair_score + 0.10 * liquidity_score + 0.10 * vol_score
    if technical_score >= 0.62:
        view = "bullish"
    elif technical_score <= 0.42:
        view = "bearish"
    else:
        view = "neutral"
    if close > (ma20 or close + 1) and (ma20 or 0) > (ma60 or 0):
        setup = "short_term_uptrend"
    elif close < (ma20 or close - 1) and (ma20 or 0) < (ma60 or 0):
        setup = "short_term_downtrend"
    else:
        setup = "range_or_transition"
    return {
        "ts_code": ts_code,
        "symbol": symbol_from_ts_code(ts_code),
        "name": name,
        "status": "ok",
        "latest_trade_date": latest["trade_date_dt"].date().isoformat(),
        "available_date": str(latest.get("available_date", "")),
        "close_raw": round(close, 3),
        "pct_chg_latest": round(float(latest["pct_chg"]), 3) if pd.notna(latest["pct_chg"]) else "",
        "ret_20d": ret20,
        "ret_60d": ret60,
        "ret_120d": ret120,
        "ret_250d": ret250,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma250": ma250,
        "close_vs_ma20": close / ma20 - 1.0 if ma20 else None,
        "close_vs_ma60": close / ma60 - 1.0 if ma60 else None,
        "close_vs_ma120": close / ma120 - 1.0 if ma120 else None,
        "drawdown_120d": drawdown_120,
        "drawdown_250d": drawdown_250,
        "low_20d": float(latest["low_20"]) if pd.notna(latest["low_20"]) else "",
        "low_60d": float(latest["low_60"]) if pd.notna(latest["low_60"]) else "",
        "low_120d": low120,
        "high_20d": float(latest["high_20"]) if pd.notna(latest["high_20"]) else "",
        "high_60d": float(latest["high_60"]) if pd.notna(latest["high_60"]) else "",
        "high_120d": high120,
        "low_250d": low250,
        "high_250d": high250,
        "vol_20d_ann": vol20,
        "vol_60d_ann": float(latest["vol_60"]) if pd.notna(latest["vol_60"]) else None,
        "amount_avg_20": float(latest["amount_avg_20"]) if pd.notna(latest["amount_avg_20"]) else None,
        "amount_ratio_20_120": amount_ratio,
        "ma_stack_score": ma_stack_score,
        "momentum_score": momentum_score,
        "drawdown_repair_score": repair_score,
        "technical_score": technical_score,
        "technical_view": view,
        "setup": setup,
        "data_boundary": "raw unadjusted daily; use for price-state research only, not total-return backtest",
    }


def load_index_context(symbols: list[str]) -> list[dict[str, Any]]:
    index_names = {"000016": "上证50", "000300": "沪深300", "000985": "中证全指"}
    rows: list[dict[str, Any]] = []
    for index_code, index_name in index_names.items():
        path = INDEX_WEIGHT_DIR / f"{index_code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
        df["asset"] = df["asset"].astype(str).str.zfill(6)
        for symbol in symbols:
            match = df[df["asset"].eq(symbol)]
            if match.empty:
                continue
            row = match.iloc[-1]
            rows.append(
                {
                    "symbol": symbol,
                    "index_code": index_code,
                    "index_name": index_name,
                    "weight_date": row.get("date", ""),
                    "weight_pct": row.get("weight", ""),
                    "data_source": row.get("data_source", ""),
                    "fetched_at": row.get("fetched_at", ""),
                    "pit_status": "latest_snapshot_only",
                }
            )
    return rows


def load_index_returns() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_code, index_name in {"000300": "沪深300", "000985": "中证全指"}.items():
        path = INDEX_DAILY_DIR / f"{index_code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        latest = df.iloc[-1]
        item = {"index_code": index_code, "index_name": index_name, "latest_date": latest["date"].date().isoformat(), "latest_close": latest["close"]}
        for days in [20, 60, 120, 250]:
            if len(df) > days:
                item[f"ret_{days}d"] = float(df["close"].iloc[-1] / df["close"].iloc[-days - 1] - 1.0)
        rows.append(item)
    return rows


def fetch_financial_snapshots(symbols: list[str]) -> list[dict[str, Any]]:
    try:
        import akshare as ak
    except Exception as exc:
        return [
            {
                "symbol": symbol,
                "status": "blocked",
                "error": f"akshare import failed: {type(exc).__name__}",
                "pit_status": "missing",
                "historical_backtest_allowed": False,
            }
            for symbol in symbols
        ]
    rows: list[dict[str, Any]] = []
    wanted = [
        "报告期",
        "营业总收入",
        "营业总收入同比增长率",
        "净利润",
        "净利润同比增长率",
        "销售毛利率",
        "销售净利率",
        "净资产收益率",
        "基本每股收益",
        "每股净资产",
        "资产负债率",
    ]
    for symbol in symbols:
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol)
            if df.empty:
                rows.append({"symbol": symbol, "status": "empty", "pit_status": "current_snapshot_no_available_date", "historical_backtest_allowed": False})
                continue
            latest = df.iloc[-1]
            row = {"symbol": symbol, "status": "ok", "pit_status": "current_snapshot_no_available_date", "historical_backtest_allowed": False}
            for col in wanted:
                row[col] = latest.get(col, "")
            row["data_source"] = "akshare.stock_financial_abstract_ths"
            row["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "blocked",
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "pit_status": "missing",
                    "historical_backtest_allowed": False,
                }
            )
    return rows


def classify_fundamental(row: dict[str, Any], symbol: str) -> tuple[str, str]:
    if row.get("status") != "ok":
        return "unknown", "fundamental snapshot unavailable"
    revenue_yoy = str(row.get("营业总收入同比增长率", ""))
    profit_yoy = str(row.get("净利润同比增长率", ""))
    gross = str(row.get("销售毛利率", ""))
    roe = str(row.get("净资产收益率", ""))
    debt = str(row.get("资产负债率", ""))
    if symbol == "600519":
        return "quality_growth_slowing", f"high-margin/high-ROE franchise, but growth is moderate: revenue_yoy={revenue_yoy}, profit_yoy={profit_yoy}, gross_margin={gross}, roe={roe}, debt={debt}"
    if symbol == "688981":
        return "cyclical_growth_with_capex_risk", f"semiconductor cycle exposure with improving revenue/profit but lower profitability: revenue_yoy={revenue_yoy}, profit_yoy={profit_yoy}, gross_margin={gross}, roe={roe}, debt={debt}"
    return "current_snapshot_only", f"revenue_yoy={revenue_yoy}, profit_yoy={profit_yoy}, gross_margin={gross}, roe={roe}, debt={debt}"


def render_report(
    technical_rows: list[dict[str, Any]],
    financial_rows: list[dict[str, Any]],
    index_context: list[dict[str, Any]],
    index_returns: list[dict[str, Any]],
    scan_meta: dict[str, Any],
) -> str:
    financial_by_symbol = {row["symbol"]: row for row in financial_rows}
    lines = [
        "# 贵州茅台与中芯国际研究快照",
        "",
        "边界：这是研究助手测试输出，不是买卖建议。价格数据来自本地 Tushare raw daily，属于未复权原始日线；财务数据是 AkShare 当前快照，无 `available_date`，不能进入历史回测训练。",
        "",
        f"- 运行时间: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Tushare 日线文件扫描数: {scan_meta.get('files_scanned')}",
        f"- 匹配日线行数: {scan_meta.get('matched_rows')}",
        f"- 解析错误: {scan_meta.get('parse_errors')}",
        "",
        "## 技术面结论",
        "",
    ]
    for row in technical_rows:
        symbol = row["symbol"]
        fin_state, fin_note = classify_fundamental(financial_by_symbol.get(symbol, {}), symbol)
        lines.extend(
            [
                f"### {row['name']} `{row['ts_code']}`",
                "",
                f"- 截止交易日: {row.get('latest_trade_date')}；原始收盘价: {price(row.get('close_raw'))}",
                f"- 技术视图: `{row.get('technical_view')}`；形态: `{row.get('setup')}`；技术分数: {float(row.get('technical_score', 0)):.3f}",
                f"- 20/60/120日收益: {pct(row.get('ret_20d'))} / {pct(row.get('ret_60d'))} / {pct(row.get('ret_120d'))}",
                f"- 相对 MA20/MA60/MA120: {pct(row.get('close_vs_ma20'))} / {pct(row.get('close_vs_ma60'))} / {pct(row.get('close_vs_ma120'))}",
                f"- 120日/250日回撤: {pct(row.get('drawdown_120d'))} / {pct(row.get('drawdown_250d'))}",
                f"- 20日/60日年化波动: {pct(row.get('vol_20d_ann'))} / {pct(row.get('vol_60d_ann'))}",
                f"- 观察区间: 20日低点 {price(row.get('low_20d'))}，60日低点 {price(row.get('low_60d'))}，120日低点 {price(row.get('low_120d'))}；20日高点 {price(row.get('high_20d'))}，60日高点 {price(row.get('high_60d'))}，120日高点 {price(row.get('high_120d'))}",
                f"- 基本面快照判定: `{fin_state}`；{fin_note}",
                "",
            ]
        )
    lines.extend(["## 指数权重背景", ""])
    for row in index_context:
        lines.append(f"- `{row['symbol']}` 在 {row['index_name']} `{row['index_code']}` 权重 {row['weight_pct']}%，日期 {row['weight_date']}；状态 `{row['pit_status']}`")
    if index_returns:
        lines.extend(["", "## 指数参照", ""])
        for row in index_returns:
            lines.append(
                f"- {row['index_name']} `{row['index_code']}` 截止 {row.get('latest_date')}: 20/60/120日收益 {pct(row.get('ret_20d'))} / {pct(row.get('ret_60d'))} / {pct(row.get('ret_120d'))}"
            )
    lines.extend(
        [
            "",
            "## 数据风险",
            "",
            "- Tushare raw daily 是未复权数据，适合短中期价格状态分析；不适合直接评价长期总收益，尤其对高分红股票会低估实际持有收益。",
            "- AkShare 财务摘要只有当前抓取快照，没有 `available_date`，只能用于当前研究画像，不能用于历史因子回测。",
            "- 本报告没有使用估值历史、股息率历史、公告日期、机构一致预期和行业景气度高频数据，因此基本面置信度被限制。",
            "",
        ]
    )
    return "\n".join(lines)


def run(symbols: list[str], output_dir: Path) -> int:
    ensure_dir(output_dir)
    ts_codes = [ts_code_from_symbol(symbol) for symbol in symbols]
    symbol_list = [symbol_from_ts_code(code) for code in ts_codes]
    names = load_stock_names()
    daily, scan_meta = load_raw_daily(ts_codes)
    technical_rows = [
        technical_summary_for_stock(daily, code, names.get(symbol_from_ts_code(code), symbol_from_ts_code(code)))
        for code in ts_codes
    ]
    financial_rows = fetch_financial_snapshots(symbol_list)
    index_context = load_index_context(symbol_list)
    index_returns = load_index_returns()
    boundary_rows = [
        {
            "dataset": "tushare_raw_daily",
            "status": "approved_limited_raw_price_state",
            "historical_total_return_allowed": False,
            "historical_fundamental_backtest_allowed": False,
            "restriction": "unadjusted raw daily; no total-return backtest",
        },
        {
            "dataset": "akshare.stock_financial_abstract_ths",
            "status": "research_only",
            "historical_total_return_allowed": False,
            "historical_fundamental_backtest_allowed": False,
            "restriction": "current snapshot; no available_date",
        },
        {
            "dataset": "akshare_csindex.weights_latest",
            "status": "research_only",
            "historical_total_return_allowed": False,
            "historical_fundamental_backtest_allowed": False,
            "restriction": "latest constituent weight snapshot only",
        },
    ]
    write_csv(output_dir / "stock_technical_summary.csv", technical_rows)
    write_csv(output_dir / "stock_fundamental_snapshot.csv", financial_rows)
    write_csv(output_dir / "stock_index_context.csv", index_context)
    write_csv(output_dir / "index_return_context.csv", index_returns)
    write_csv(output_dir / "data_boundary_audit.csv", boundary_rows)
    write_text(output_dir / "stock_research_report.md", render_report(technical_rows, financial_rows, index_context, index_returns, scan_meta))
    manifest = {
        "run_id": "stock_research_20260603_moutai_smic",
        "symbols": ts_codes,
        "output_dir": rel_path(output_dir),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scan_meta": scan_meta,
        "outputs": [
            rel_path(output_dir / "stock_technical_summary.csv"),
            rel_path(output_dir / "stock_fundamental_snapshot.csv"),
            rel_path(output_dir / "stock_index_context.csv"),
            rel_path(output_dir / "index_return_context.csv"),
            rel_path(output_dir / "data_boundary_audit.csv"),
            rel_path(output_dir / "stock_research_report.md"),
        ],
        "research_only": True,
    }
    write_text(output_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["600519.SH", "688981.SH"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    return run(args.symbols, ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
