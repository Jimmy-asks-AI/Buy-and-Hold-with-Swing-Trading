#!/usr/bin/env python
"""Harvest governed A-share index data for quant research.

The first production target is broad, high-quality index data:
daily OHLCV, current constituents, latest weights, latest CSIndex
valuation snapshots, and historical PE/PB from Legulegu where available.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "akshare_csindex"

INDEX_UNIVERSE: dict[str, dict[str, str]] = {
    "000016": {"name": "SSE 50", "lg_name": "\u4e0a\u8bc150"},
    "000300": {"name": "CSI 300", "lg_name": "\u6caa\u6df1300"},
    "000905": {"name": "CSI 500", "lg_name": "\u4e2d\u8bc1500"},
    "000852": {"name": "CSI 1000", "lg_name": "\u4e2d\u8bc11000"},
    "000906": {"name": "CSI 800", "lg_name": "\u4e2d\u8bc1800"},
    "000985": {"name": "CSI All Share", "lg_name": ""},
    "000922": {"name": "CSI Dividend", "lg_name": ""},
    "000015": {"name": "SSE Dividend", "lg_name": ""},
}

ZH = {
    "date": "\u65e5\u671f",
    "index_code": "\u6307\u6570\u4ee3\u7801",
    "index_name": "\u6307\u6570\u4e2d\u6587\u5168\u79f0",
    "index_short_name": "\u6307\u6570\u4e2d\u6587\u7b80\u79f0",
    "index_english_name": "\u6307\u6570\u82f1\u6587\u5168\u79f0",
    "index_english_short_name": "\u6307\u6570\u82f1\u6587\u7b80\u79f0",
    "open": "\u5f00\u76d8",
    "high": "\u6700\u9ad8",
    "low": "\u6700\u4f4e",
    "close": "\u6536\u76d8",
    "change": "\u6da8\u8dcc",
    "pct_chg": "\u6da8\u8dcc\u5e45",
    "volume": "\u6210\u4ea4\u91cf",
    "amount": "\u6210\u4ea4\u91d1\u989d",
    "constituent_count": "\u6837\u672c\u6570\u91cf",
    "pe_ttm": "\u6eda\u52a8\u5e02\u76c8\u7387",
    "simple_index_name": "\u6307\u6570\u540d\u79f0",
    "level": "\u6307\u6570",
    "asset": "\u6210\u5206\u5238\u4ee3\u7801",
    "asset_name": "\u6210\u5206\u5238\u540d\u79f0",
    "exchange": "\u4ea4\u6613\u6240",
    "weight": "\u6743\u91cd",
    "pe_static_equal_weight": "\u7b49\u6743\u9759\u6001\u5e02\u76c8\u7387",
    "pe_static": "\u9759\u6001\u5e02\u76c8\u7387",
    "pe_static_median": "\u9759\u6001\u5e02\u76c8\u7387\u4e2d\u4f4d\u6570",
    "pe_ttm_equal_weight": "\u7b49\u6743\u6eda\u52a8\u5e02\u76c8\u7387",
    "pe_ttm_median": "\u6eda\u52a8\u5e02\u76c8\u7387\u4e2d\u4f4d\u6570",
    "pb": "\u5e02\u51c0\u7387",
    "pb_equal_weight": "\u7b49\u6743\u5e02\u51c0\u7387",
    "pb_median": "\u5e02\u51c0\u7387\u4e2d\u4f4d\u6570",
    "dividend_yield": "\u80a1\u606f\u7387",
}

DAILY_RENAME = {
    ZH["date"]: "date",
    ZH["index_code"]: "index_code",
    ZH["index_name"]: "index_name",
    ZH["index_short_name"]: "index_short_name",
    ZH["index_english_name"]: "index_english_name",
    ZH["index_english_short_name"]: "index_english_short_name",
    ZH["open"]: "open",
    ZH["high"]: "high",
    ZH["low"]: "low",
    ZH["close"]: "close",
    ZH["change"]: "change",
    ZH["pct_chg"]: "pct_chg",
    ZH["volume"]: "volume",
    ZH["amount"]: "amount",
    ZH["constituent_count"]: "constituent_count",
    ZH["pe_ttm"]: "pe_ttm",
}

COMPONENT_RENAME = {
    ZH["date"]: "date",
    ZH["index_code"]: "index_code",
    ZH["simple_index_name"]: "index_name",
    ZH["asset"]: "asset",
    ZH["asset_name"]: "asset_name",
    ZH["exchange"]: "exchange",
    ZH["weight"]: "weight",
}

VALUATION_RENAME = {
    ZH["date"]: "date",
    ZH["index_code"]: "index_code",
    ZH["simple_index_name"]: "index_name",
    ZH["level"]: "close",
    ZH["pe_ttm"]: "pe_ttm",
    ZH["pe_static_equal_weight"]: "pe_static_equal_weight",
    ZH["pe_static"]: "pe_static",
    ZH["pe_static_median"]: "pe_static_median",
    ZH["pe_ttm_equal_weight"]: "pe_ttm_equal_weight",
    ZH["pe_ttm_median"]: "pe_ttm_median",
    ZH["pb"]: "pb",
    ZH["pb_equal_weight"]: "pb_equal_weight",
    ZH["pb_median"]: "pb_median",
    ZH["dividend_yield"]: "dividend_yield",
}

NUMERIC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_chg",
    "volume",
    "amount",
    "constituent_count",
    "pe_ttm",
    "pe_static_equal_weight",
    "pe_static",
    "pe_static_median",
    "pe_ttm_equal_weight",
    "pe_ttm_median",
    "pb",
    "pb_equal_weight",
    "pb_median",
    "dividend_yield",
    "weight",
}


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


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_code(code: str) -> str:
    text = str(code).strip().upper()
    if "." in text:
        text = text.split(".")[0]
    text = text.removeprefix("SH").removeprefix("SZ")
    return text.zfill(6)


def _sina_index_symbol(code: str) -> str:
    code = _normalize_code(code)
    if code.startswith("399"):
        return f"sz{code}"
    return f"sh{code}"


def _date_text(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d")


def _coerce_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "date" in out.columns:
        out["date"] = _date_text(out["date"])
        out = out.dropna(subset=["date"]).sort_values("date")
    if "index_code" in out.columns:
        out["index_code"] = out["index_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    if "asset" in out.columns:
        out["asset"] = out["asset"].astype(str).str.extract(r"(\d{6})", expand=False)
    return out


def _add_daily_bar_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ohlc = [col for col in ["open", "high", "low", "close"] if col in out.columns]
    if ohlc:
        out["is_full_ohlc_bar"] = out[ohlc].notna().all(axis=1)
        out["is_close_only_bar"] = out["close"].notna() & ~out["is_full_ohlc_bar"] if "close" in out.columns else False
    if {"volume", "amount"}.issubset(out.columns):
        volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
        amount = pd.to_numeric(out["amount"], errors="coerce").fillna(0)
        out["is_zero_volume_amount"] = (volume == 0) & (amount == 0)
    return out


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    dates = pd.to_datetime(df["date"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if pd.notna(start):
        mask &= dates >= start
    if pd.notna(end):
        mask &= dates <= end
    return df.loc[mask].copy()


def _with_meta(df: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    out = df.copy()
    if "index_code" not in out.columns:
        out["index_code"] = code
    out["index_code"] = out["index_code"].fillna(code).astype(str).map(_normalize_code)
    out["data_source"] = source
    out["fetched_at"] = _now()
    return out


def normalize_daily_csindex(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.rename(columns=DAILY_RENAME).copy()
    out = _coerce_common(out)
    out = _add_daily_bar_flags(out)
    return _with_meta(out, code, "akshare.stock_zh_index_hist_csindex")


def normalize_daily_sina(df: pd.DataFrame, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}).copy()
    out = _coerce_common(out)
    out = _filter_date_range(out, start_date, end_date)
    out = _add_daily_bar_flags(out)
    return _with_meta(out, code, "akshare.stock_zh_index_daily")


def normalize_valuation(df: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    out = df.rename(columns=VALUATION_RENAME).copy()
    for col in list(out.columns):
        if col.startswith(ZH["dividend_yield"]):
            suffix = col.removeprefix(ZH["dividend_yield"]).strip()
            out = out.rename(columns={col: f"dividend_yield{suffix}" if suffix else "dividend_yield"})
    out = _coerce_common(out)
    return _with_meta(out, code, source)


def normalize_components(df: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    out = df.rename(columns=COMPONENT_RENAME).copy()
    out = _coerce_common(out)
    return _with_meta(out, code, source)


def normalize_index_list(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=DAILY_RENAME | COMPONENT_RENAME).copy()
    out = _coerce_common(out)
    out["data_source"] = "akshare.index_csindex_all"
    out["fetched_at"] = _now()
    return out


def fetch_index_list(output_root: Path) -> pd.DataFrame:
    ak = _akshare()
    raw = ak.index_csindex_all()
    clean = normalize_index_list(raw)
    _write_csv(clean, output_root / "index_list" / "index_csindex_all.csv")
    return clean


def fetch_daily(ak, code: str, start_date: str, end_date: str, backend: str) -> pd.DataFrame:
    errors: list[str] = []
    if backend in {"auto", "csindex"}:
        try:
            raw = ak.stock_zh_index_hist_csindex(symbol=code, start_date=start_date, end_date=end_date)
            return normalize_daily_csindex(raw, code)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"csindex={repr(exc)}")
            if backend == "csindex":
                raise
    if backend in {"auto", "sina"}:
        try:
            raw = ak.stock_zh_index_daily(symbol=_sina_index_symbol(code))
            return normalize_daily_sina(raw, code, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sina={repr(exc)}")
            if backend == "sina":
                raise
    raise RuntimeError("all daily index backends failed: " + " | ".join(errors))


def fetch_latest_valuation(ak, code: str) -> pd.DataFrame:
    raw = ak.stock_zh_index_value_csindex(symbol=code)
    return normalize_valuation(raw, code, "akshare.stock_zh_index_value_csindex")


def fetch_pe_lg(ak, code: str, lg_name: str) -> pd.DataFrame:
    if not lg_name:
        raise RuntimeError("no Legulegu symbol configured")
    raw = ak.stock_index_pe_lg(symbol=lg_name)
    return normalize_valuation(raw, code, "akshare.stock_index_pe_lg")


def fetch_pb_lg(ak, code: str, lg_name: str) -> pd.DataFrame:
    if not lg_name:
        raise RuntimeError("no Legulegu symbol configured")
    raw = ak.stock_index_pb_lg(symbol=lg_name)
    return normalize_valuation(raw, code, "akshare.stock_index_pb_lg")


def fetch_constituents(ak, code: str) -> pd.DataFrame:
    raw = ak.index_stock_cons_csindex(symbol=code)
    return normalize_components(raw, code, "akshare.index_stock_cons_csindex")


def fetch_weights(ak, code: str) -> pd.DataFrame:
    raw = ak.index_stock_cons_weight_csindex(symbol=code)
    return normalize_components(raw, code, "akshare.index_stock_cons_weight_csindex")


def _csv_row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)
    except UnicodeDecodeError:
        return max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)


def _manifest_row(code: str, dataset: str, path: Path, status: str, rows: int | str, started: float, error: str = "") -> dict:
    return {
        "index_code": code,
        "dataset": dataset,
        "status": status,
        "rows": rows,
        "path": str(path),
        "error": error,
        "seconds": round(time.time() - started, 3),
        "fetched_at": _now(),
    }


def run_symbol_dataset(
    output_root: Path,
    dataset: str,
    symbols: list[str],
    fetcher: Callable[[str], pd.DataFrame],
    sleep_seconds: float,
    resume: bool,
) -> pd.DataFrame:
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, code in enumerate(symbols, start=1):
        code = _normalize_code(code)
        path = out_dir / f"{code}.csv"
        started = time.time()
        if resume and path.exists() and path.stat().st_size > 0:
            rows.append(_manifest_row(code, dataset, path, "skipped_existing", _csv_row_count(path), started))
            continue
        try:
            clean = fetcher(code)
            _write_csv(clean, path)
            rows.append(_manifest_row(code, dataset, path, "ok", int(clean.shape[0]), started))
        except Exception as exc:  # noqa: BLE001
            rows.append(_manifest_row(code, dataset, path, "error", 0, started, repr(exc)))
        if sleep_seconds > 0 and idx < len(symbols):
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def build_quality_summary(output_root: Path, manifests: list[pd.DataFrame]) -> pd.DataFrame:
    manifest = pd.concat([m for m in manifests if not m.empty], ignore_index=True) if manifests else pd.DataFrame()
    rows: list[dict] = []
    for _, item in manifest.iterrows():
        path = Path(str(item.get("path", "")))
        status = str(item.get("status", ""))
        dataset = str(item.get("dataset", ""))
        raw_code = str(item.get("index_code", ""))
        code = raw_code if raw_code == "ALL" else _normalize_code(raw_code)
        if status not in {"ok", "skipped_existing"} or not path.exists():
            rows.append({"index_code": code, "dataset": dataset, "status": status, "rows": 0, "quality_flag": "missing_or_error"})
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        quality_flag = "ok"
        duplicate_dates = 0
        null_close = 0
        nonpositive_close = 0
        full_ohlc_rows = ""
        close_only_rows = ""
        zero_volume_amount_rows = ""
        start_date = ""
        end_date = ""
        if "date" in df.columns:
            duplicate_key = ["date", "asset"] if "asset" in df.columns else ["date"]
            duplicate_dates = int(df.duplicated(subset=duplicate_key).sum())
            dates = pd.to_datetime(df["date"], errors="coerce")
            if dates.notna().any():
                start_date = str(dates.min().date())
                end_date = str(dates.max().date())
        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            null_close = int(close.isna().sum())
            nonpositive_close = int((close <= 0).sum())
        if "is_full_ohlc_bar" in df.columns:
            full_ohlc_rows = int(df["is_full_ohlc_bar"].astype(str).str.lower().isin({"true", "1"}).sum())
        if "is_close_only_bar" in df.columns:
            close_only_rows = int(df["is_close_only_bar"].astype(str).str.lower().isin({"true", "1"}).sum())
        if "is_zero_volume_amount" in df.columns:
            zero_volume_amount_rows = int(df["is_zero_volume_amount"].astype(str).str.lower().isin({"true", "1"}).sum())
        if duplicate_dates or nonpositive_close:
            quality_flag = "review"
        rows.append(
            {
                "index_code": code,
                "dataset": dataset,
                "status": status,
                "rows": int(df.shape[0]),
                "start_date": start_date,
                "end_date": end_date,
                "duplicate_dates": duplicate_dates,
                "null_close": null_close,
                "nonpositive_close": nonpositive_close,
                "full_ohlc_rows": full_ohlc_rows,
                "close_only_rows": close_only_rows,
                "zero_volume_amount_rows": zero_volume_amount_rows,
                "quality_flag": quality_flag,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        _write_csv(out, output_root / "manifests" / f"quality_summary_{_now().replace(':', '-')}.csv")
    return out


def selected_symbols(symbols_arg: str | None, max_symbols: int | None) -> list[str]:
    if symbols_arg:
        symbols = [_normalize_code(item) for item in symbols_arg.split(",") if item.strip()]
    else:
        symbols = list(INDEX_UNIVERSE)
    if max_symbols:
        symbols = symbols[:max_symbols]
    return symbols


def write_dataset_plan(output_root: Path, datasets: list[str], symbols: list[str], args: argparse.Namespace) -> None:
    plan = {
        "source": DEFAULT_SOURCE,
        "created_at": _now(),
        "selected_symbols": symbols,
        "datasets": datasets,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "daily_backend": args.daily_backend,
        "quality_note": (
            "CSIndex/AkShare index daily, current constituents, weights, and valuation snapshots are suitable "
            "for research prototyping. Production-grade point-in-time backtests still need licensed historical "
            "constituent and weight history from Tushare Pro, JoinQuant, Wind, or CSIndex files."
        ),
    }
    _write_json(plan, output_root / "DATASET_PLAN.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--symbols", default="000016,000300,000905")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--datasets", default="index_list,daily,valuation_latest,valuation_pe_lg,valuation_pb_lg,constituents,weights")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--daily-backend", default="auto", choices=["auto", "csindex", "sina"])
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / "data_raw" / "index" / DEFAULT_SOURCE
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    symbols = selected_symbols(args.symbols, args.max_symbols)
    write_dataset_plan(output_root, datasets, symbols, args)

    manifests: list[pd.DataFrame] = []
    ak = _akshare()
    if "index_list" in datasets:
        started = time.time()
        path = output_root / "index_list" / "index_csindex_all.csv"
        try:
            clean = fetch_index_list(output_root)
            manifests.append(pd.DataFrame([_manifest_row("ALL", "index_list", path, "ok", int(clean.shape[0]), started)]))
        except Exception as exc:  # noqa: BLE001
            manifests.append(pd.DataFrame([_manifest_row("ALL", "index_list", path, "error", 0, started, repr(exc))]))

    if "daily" in datasets:
        manifests.append(
            run_symbol_dataset(
                output_root,
                "daily_csindex",
                symbols,
                lambda code: fetch_daily(ak, code, args.start_date, args.end_date, args.daily_backend),
                args.sleep_seconds,
                args.resume,
            )
        )
    if "valuation_latest" in datasets:
        manifests.append(
            run_symbol_dataset(output_root, "valuation_latest_csindex", symbols, lambda code: fetch_latest_valuation(ak, code), args.sleep_seconds, args.resume)
        )
    if "valuation_pe_lg" in datasets:
        manifests.append(
            run_symbol_dataset(
                output_root,
                "valuation_pe_lg",
                symbols,
                lambda code: fetch_pe_lg(ak, code, INDEX_UNIVERSE.get(code, {}).get("lg_name", "")),
                args.sleep_seconds,
                args.resume,
            )
        )
    if "valuation_pb_lg" in datasets:
        manifests.append(
            run_symbol_dataset(
                output_root,
                "valuation_pb_lg",
                symbols,
                lambda code: fetch_pb_lg(ak, code, INDEX_UNIVERSE.get(code, {}).get("lg_name", "")),
                args.sleep_seconds,
                args.resume,
            )
        )
    if "constituents" in datasets:
        manifests.append(
            run_symbol_dataset(output_root, "constituents_current", symbols, lambda code: fetch_constituents(ak, code), args.sleep_seconds, args.resume)
        )
    if "weights" in datasets:
        manifests.append(run_symbol_dataset(output_root, "weights_latest", symbols, lambda code: fetch_weights(ak, code), args.sleep_seconds, args.resume))

    manifest = pd.concat([m for m in manifests if not m.empty], ignore_index=True) if manifests else pd.DataFrame()
    if not manifest.empty:
        manifest_path = output_root / "manifests" / f"manifest_all_{_now().replace(':', '-')}.csv"
        _write_csv(manifest, manifest_path)
    quality = build_quality_summary(output_root, manifests)

    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "symbols": symbols,
                "manifest_rows": int(manifest.shape[0]),
                "quality_rows": int(quality.shape[0]),
                "ok_rows": int((manifest["status"] == "ok").sum()) if not manifest.empty else 0,
                "error_rows": int((manifest["status"] == "error").sum()) if not manifest.empty else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
