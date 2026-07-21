#!/usr/bin/env python
"""Harvest broad A-share raw data into a governed local data lake."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("akshare is not installed.") from exc
    return ak


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _normalize_code(code: str) -> str:
    text = str(code).strip()
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def fetch_akshare_stock_list(output_root: Path) -> pd.DataFrame:
    ak = _akshare()
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "asset", "name": "name"})
    df["asset"] = df["asset"].astype(str).str.zfill(6)
    df["fetched_at"] = _now()
    _write_csv(df, output_root / "stock_list" / "stock_info_a_code_name.csv")
    return df


def fetch_akshare_trade_calendar(output_root: Path) -> pd.DataFrame:
    ak = _akshare()
    df = ak.tool_trade_date_hist_sina()
    df = df.rename(columns={"trade_date": "date"})
    if "date" not in df.columns and df.shape[1] >= 1:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["fetched_at"] = _now()
    _write_csv(df, output_root / "calendar" / "trade_calendar.csv")
    return df


def _akshare_prefixed_symbol(code: str) -> str:
    code = _normalize_code(code)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def normalize_akshare_daily(df: pd.DataFrame, code: str, adjust: str, data_source: str) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "股票代码": "asset",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude_pct",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover_pct",
    }
    out = df.rename(columns=rename).copy()
    if "asset" not in out.columns:
        out["asset"] = code
    out["asset"] = out["asset"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(code)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude_pct", "pct_chg", "change", "turnover_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "turnover" in out.columns and "turnover_pct" not in out.columns:
        out["turnover_pct"] = pd.to_numeric(out["turnover"], errors="coerce") * 100.0
    if "close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["close"], errors="coerce")
    if adjust == "" and {"close", "outstanding_share"}.issubset(out.columns):
        out["market_cap"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(out["outstanding_share"], errors="coerce")
    out["adjust"] = adjust
    out["data_source"] = data_source
    out["fetched_at"] = _now()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def fetch_one_akshare_daily(ak, code: str, start_date: str, end_date: str, adjust: str, backend: str) -> pd.DataFrame:
    errors: list[str] = []
    if backend in {"auto", "eastmoney"}:
        try:
            raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
            return normalize_akshare_daily(raw, code, adjust, "akshare.stock_zh_a_hist")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"eastmoney={repr(exc)}")
            if backend == "eastmoney":
                raise
    if backend in {"auto", "sina"}:
        try:
            raw = ak.stock_zh_a_daily(symbol=_akshare_prefixed_symbol(code), start_date=start_date, end_date=end_date, adjust=adjust)
            return normalize_akshare_daily(raw, code, adjust, "akshare.stock_zh_a_daily")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sina={repr(exc)}")
            if backend == "sina":
                raise
    raise RuntimeError("all daily backends failed: " + " | ".join(errors))


def fetch_akshare_daily(
    output_root: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    adjust: str,
    backend: str,
    sleep_seconds: float,
    resume: bool,
) -> pd.DataFrame:
    ak = _akshare()
    out_dir = output_root / f"daily_{adjust or 'raw'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, code in enumerate(symbols, start=1):
        code = _normalize_code(code)
        path = out_dir / f"{code}.csv"
        if resume and path.exists() and path.stat().st_size > 0:
            rows.append({"asset": code, "dataset": f"daily_{adjust or 'raw'}", "status": "skipped_existing", "rows": "", "error": "", "fetched_at": _now()})
            continue
        started = time.time()
        try:
            clean = fetch_one_akshare_daily(ak, code, start_date, end_date, adjust, backend)
            _write_csv(clean, path)
            rows.append({"asset": code, "dataset": f"daily_{adjust or 'raw'}", "status": "ok", "rows": int(clean.shape[0]), "error": "", "seconds": round(time.time() - started, 3), "fetched_at": _now()})
        except Exception as exc:  # noqa: BLE001
            rows.append({"asset": code, "dataset": f"daily_{adjust or 'raw'}", "status": "error", "rows": 0, "error": repr(exc), "seconds": round(time.time() - started, 3), "fetched_at": _now()})
        if sleep_seconds > 0 and idx < len(symbols):
            time.sleep(sleep_seconds)
    manifest = pd.DataFrame(rows)
    _write_csv(manifest, output_root / "manifests" / f"manifest_daily_{adjust or 'raw'}_{_now().replace(':', '-')}.csv")
    return manifest


def normalize_akshare_financial_indicator(df: pd.DataFrame, code: str, data_source: str) -> pd.DataFrame:
    out = df.copy()
    out["asset"] = code
    out["data_source"] = data_source
    out["fetched_at"] = _now()
    return out


def fetch_one_akshare_financial_indicator(ak, code: str, start_year: str, backend: str) -> pd.DataFrame:
    errors: list[str] = []
    if backend in {"auto", "analysis_indicator"}:
        try:
            raw = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
            return normalize_akshare_financial_indicator(raw, code, "akshare.stock_financial_analysis_indicator")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"analysis_indicator={repr(exc)}")
            if backend == "analysis_indicator":
                raise
    if backend in {"auto", "abstract_ths"}:
        try:
            raw = ak.stock_financial_abstract_ths(symbol=code)
            return normalize_akshare_financial_indicator(raw, code, "akshare.stock_financial_abstract_ths")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"abstract_ths={repr(exc)}")
            if backend == "abstract_ths":
                raise
    raise RuntimeError("all financial indicator backends failed: " + " | ".join(errors))


def fetch_akshare_financial_indicator(
    output_root: Path,
    symbols: list[str],
    start_year: str,
    backend: str,
    sleep_seconds: float,
    resume: bool,
) -> pd.DataFrame:
    ak = _akshare()
    out_dir = output_root / "financial_indicator"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, code in enumerate(symbols, start=1):
        code = _normalize_code(code)
        path = out_dir / f"{code}.csv"
        if resume and path.exists() and path.stat().st_size > 0:
            rows.append({"asset": code, "dataset": "financial_indicator", "status": "skipped_existing", "rows": "", "error": "", "fetched_at": _now()})
            continue
        started = time.time()
        try:
            clean = fetch_one_akshare_financial_indicator(ak, code, start_year, backend)
            _write_csv(clean, path)
            rows.append({"asset": code, "dataset": "financial_indicator", "status": "ok", "rows": int(clean.shape[0]), "error": "", "seconds": round(time.time() - started, 3), "fetched_at": _now()})
        except Exception as exc:  # noqa: BLE001
            rows.append({"asset": code, "dataset": "financial_indicator", "status": "error", "rows": 0, "error": repr(exc), "seconds": round(time.time() - started, 3), "fetched_at": _now()})
        if sleep_seconds > 0 and idx < len(symbols):
            time.sleep(sleep_seconds)
    manifest = pd.DataFrame(rows)
    _write_csv(manifest, output_root / "manifests" / f"manifest_financial_indicator_{_now().replace(':', '-')}.csv")
    return manifest


def select_symbols(stock_list: pd.DataFrame, symbols: str | None, max_symbols: int | None) -> list[str]:
    if symbols:
        selected = [_normalize_code(item) for item in symbols.split(",") if item.strip()]
    else:
        selected = stock_list["asset"].dropna().astype(str).map(_normalize_code).drop_duplicates().tolist()
    if max_symbols:
        selected = selected[:max_symbols]
    return selected


def write_dataset_plan(output_root: Path, source: str, datasets: list[str], selected_symbols: list[str], args: argparse.Namespace) -> None:
    plan = {
        "source": source,
        "datasets": datasets,
        "selected_symbols": len(selected_symbols),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "adjust": args.adjust,
        "daily_backend": args.daily_backend,
        "financial_backend": args.financial_backend,
        "created_at": _now(),
        "quality_note": "AkShare is broad public-web data. Use Tushare/JQData with point-in-time dates for production-grade precision.",
    }
    (output_root / "DATASET_PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--source", default="akshare", choices=["akshare"])
    parser.add_argument("--datasets", default="stock_list,trade_calendar,daily", help="Comma-separated: stock_list,trade_calendar,daily,financial_indicator")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--start-year", default="2000")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--daily-backend", default="auto", choices=["auto", "eastmoney", "sina"])
    parser.add_argument("--financial-backend", default="auto", choices=["auto", "analysis_indicator", "abstract_ths"])
    parser.add_argument("--symbols")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / "data_raw" / args.source
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    stock_list = fetch_akshare_stock_list(output_root) if "stock_list" in datasets or "daily" in datasets or "financial_indicator" in datasets else pd.DataFrame()
    selected = select_symbols(stock_list, args.symbols, args.max_symbols) if not stock_list.empty or args.symbols else []
    write_dataset_plan(output_root, args.source, datasets, selected, args)
    outputs = {}
    if "trade_calendar" in datasets:
        outputs["trade_calendar_rows"] = int(fetch_akshare_trade_calendar(output_root).shape[0])
    if "daily" in datasets:
        outputs["daily_manifest_rows"] = int(fetch_akshare_daily(output_root, selected, args.start_date, args.end_date, args.adjust, args.daily_backend, args.sleep_seconds, args.resume).shape[0])
    if "financial_indicator" in datasets:
        outputs["financial_indicator_manifest_rows"] = int(fetch_akshare_financial_indicator(output_root, selected, args.start_year, args.financial_backend, args.sleep_seconds, args.resume).shape[0])
    print(json.dumps({"output_root": str(output_root.resolve()), "symbols": len(selected), **outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
