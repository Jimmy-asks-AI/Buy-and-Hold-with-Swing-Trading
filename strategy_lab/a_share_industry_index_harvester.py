#!/usr/bin/env python
"""Harvest governed A-share industry index data.

Primary source: SWS Research industry index series via AkShare.
This is preferred over generic web board data because it has stable index
codes, explicit hierarchy, long daily history, and current constituent weights.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "akshare_sw_industry"

PRIORITY_SW_CODES = {
    "801080": {"alias": "electronics_broad", "note": "电子元器件广义：申万一级电子"},
    "801083": {"alias": "electronic_components", "note": "电子元器件/元件：申万二级元件"},
    "801081": {"alias": "semiconductor", "note": "电子元器件相关：半导体"},
    "801085": {"alias": "consumer_electronics", "note": "电子元器件相关：消费电子"},
    "801770": {"alias": "communication_broad", "note": "通信广义：申万一级通信"},
    "801102": {"alias": "communication_equipment", "note": "通信设备：申万二级通信设备"},
    "801223": {"alias": "communication_service", "note": "通信服务：申万二级通信服务"},
    "801050": {"alias": "nonferrous_metals", "note": "有色金属：基本金属上级行业"},
    "801055": {"alias": "industrial_metals", "note": "基本金属近似口径：申万二级工业金属"},
    "801780": {"alias": "bank", "note": "银行：申万一级银行"},
    "801790": {"alias": "non_bank_finance", "note": "金融/非银金融：申万一级非银金融"},
    "801193": {"alias": "brokerage", "note": "证券：申万二级证券"},
    "801194": {"alias": "insurance", "note": "保险：申万二级保险"},
    "801191": {"alias": "diversified_finance", "note": "多元金融：申万二级多元金融"},
}

INFO_RENAME = {
    "行业代码": "index_code",
    "行业名称": "index_name",
    "上级行业": "parent_industry",
    "成份个数": "constituent_count",
    "静态市盈率": "pe_static",
    "TTM(滚动)市盈率": "pe_ttm",
    "市净率": "pb",
    "静态股息率": "dividend_yield_static",
}

DAILY_RENAME = {
    "代码": "index_code",
    "日期": "date",
    "收盘": "close",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

COMPONENT_RENAME = {
    "序号": "rank",
    "证券代码": "asset",
    "证券名称": "asset_name",
    "最新权重": "weight",
    "计入日期": "inclusion_date",
}

ANALYSIS_RENAME = {
    "指数代码": "index_code",
    "指数名称": "index_name",
    "发布日期": "date",
    "收盘指数": "close",
    "成交量": "volume",
    "涨跌幅": "pct_chg",
    "换手率": "turnover_pct",
    "市盈率": "pe",
    "市净率": "pb",
    "均价": "average_price",
    "成交额占比": "amount_share_pct",
    "流通市值": "float_market_cap",
    "平均流通市值": "average_float_market_cap",
    "股息率": "dividend_yield",
}

NUMERIC_COLUMNS = {
    "rank",
    "constituent_count",
    "pe_static",
    "pe_ttm",
    "pe",
    "pb",
    "dividend_yield_static",
    "dividend_yield",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "pct_chg",
    "turnover_pct",
    "average_price",
    "amount_share_pct",
    "float_market_cap",
    "average_float_market_cap",
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


def _normalize_code(code: object) -> str:
    text = str(code).strip().upper()
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def _date_text(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d")


def _coerce_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "index_code" in out.columns:
        out["index_code"] = out["index_code"].map(_normalize_code)
    if "asset" in out.columns:
        out["asset"] = out["asset"].astype(str).str.extract(r"(\d{6})", expand=False)
    for date_col in ["date", "inclusion_date"]:
        if date_col in out.columns:
            out[date_col] = _date_text(out[date_col])
    for col in out.columns:
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "date" in out.columns:
        out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _add_daily_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ohlc = [col for col in ["open", "high", "low", "close"] if col in out.columns]
    if ohlc:
        out["is_full_ohlc_bar"] = out[ohlc].notna().all(axis=1)
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


def _add_index_meta(df: pd.DataFrame, info_map: dict[str, dict], source: str) -> pd.DataFrame:
    out = df.copy()
    if "index_code" in out.columns:
        out["index_code"] = out["index_code"].map(_normalize_code)
        out["index_name"] = out.get("index_name", out["index_code"]).where(out.get("index_name", pd.Series(index=out.index)).notna(), out["index_code"])
        out["sw_level"] = out["index_code"].map(lambda code: info_map.get(code, {}).get("sw_level", ""))
        out["parent_industry"] = out["index_code"].map(lambda code: info_map.get(code, {}).get("parent_industry", ""))
        out["priority_alias"] = out["index_code"].map(lambda code: PRIORITY_SW_CODES.get(code, {}).get("alias", ""))
        out["priority_note"] = out["index_code"].map(lambda code: PRIORITY_SW_CODES.get(code, {}).get("note", ""))
    out["data_source"] = source
    out["fetched_at"] = _now()
    return out


def normalize_info(df: pd.DataFrame, level: str) -> pd.DataFrame:
    out = df.rename(columns=INFO_RENAME).copy()
    out = _coerce_common(out)
    out["sw_level"] = level
    out["data_source"] = f"akshare.sw_index_{level}_info"
    out["fetched_at"] = _now()
    return out


def normalize_daily(df: pd.DataFrame, info_map: dict[str, dict], start_date: str, end_date: str) -> pd.DataFrame:
    out = df.rename(columns=DAILY_RENAME).copy()
    out = _coerce_common(out)
    out = _filter_date_range(out, start_date, end_date)
    out = _add_daily_flags(out)
    return _add_index_meta(out, info_map, "akshare.index_hist_sw")


def normalize_components(df: pd.DataFrame, code: str, info_map: dict[str, dict]) -> pd.DataFrame:
    out = df.rename(columns=COMPONENT_RENAME).copy()
    out["index_code"] = code
    out = _coerce_common(out)
    return _add_index_meta(out, info_map, "akshare.index_component_sw")


def normalize_analysis(df: pd.DataFrame, info_map: dict[str, dict], source: str) -> pd.DataFrame:
    out = df.rename(columns=ANALYSIS_RENAME).copy()
    out = _coerce_common(out)
    return _add_index_meta(out, info_map, source)


def fetch_classification(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ak = _akshare()
    first = normalize_info(ak.sw_index_first_info(), "first")
    second = normalize_info(ak.sw_index_second_info(), "second")
    third = normalize_info(ak.sw_index_third_info(), "third")
    _write_csv(first, output_root / "classification" / "sw_level1_info.csv")
    _write_csv(second, output_root / "classification" / "sw_level2_info.csv")
    _write_csv(third, output_root / "classification" / "sw_level3_info.csv")
    return first, second, third


def build_info_map(first: pd.DataFrame, second: pd.DataFrame, third: pd.DataFrame) -> dict[str, dict]:
    info = pd.concat([first, second, third], ignore_index=True, sort=False)
    info["index_code"] = info["index_code"].map(_normalize_code)
    records = {}
    for row in info.to_dict("records"):
        records[row["index_code"]] = row
    return records


def select_symbols(universe: str, symbols_arg: str | None, first: pd.DataFrame, second: pd.DataFrame) -> list[str]:
    if symbols_arg:
        return [_normalize_code(item) for item in symbols_arg.split(",") if item.strip()]
    first_codes = first["index_code"].map(_normalize_code).dropna().drop_duplicates().tolist()
    second_codes = second["index_code"].map(_normalize_code).dropna().drop_duplicates().tolist()
    priority = [code for code in PRIORITY_SW_CODES if code in set(first_codes + second_codes)]
    if universe == "priority":
        return priority
    if universe == "all_first":
        return first_codes
    if universe == "all_second":
        return second_codes
    if universe == "priority_plus_first":
        return list(dict.fromkeys(first_codes + priority))
    raise ValueError(f"unknown universe: {universe}")


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
    rows = []
    for idx, code in enumerate(symbols, start=1):
        path = out_dir / f"{code}.csv"
        started = time.time()
        if resume and path.exists() and path.stat().st_size > 0:
            existing = pd.read_csv(path, encoding="utf-8-sig")
            rows.append(_manifest_row(code, dataset, path, "skipped_existing", int(existing.shape[0]), started))
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


def fetch_daily_dataset(ak, code: str, info_map: dict[str, dict], start_date: str, end_date: str) -> pd.DataFrame:
    raw = ak.index_hist_sw(symbol=code, period="day")
    return normalize_daily(raw, info_map, start_date, end_date)


def fetch_component_dataset(ak, code: str, info_map: dict[str, dict]) -> pd.DataFrame:
    raw = ak.index_component_sw(symbol=code)
    return normalize_components(raw, code, info_map)


def fetch_analysis_with_fallback(output_root: Path, info_map: dict[str, dict], levels: list[str], analysis_date: str) -> pd.DataFrame:
    ak = _akshare()
    symbol_map = {"first": "一级行业", "second": "二级行业"}
    frames = []
    for level in levels:
        symbol = symbol_map[level]
        selected_date = pd.to_datetime(analysis_date, errors="coerce")
        last_error = ""
        for offset in range(0, 12):
            trial = selected_date - timedelta(days=offset)
            date_text = trial.strftime("%Y%m%d")
            try:
                raw = ak.index_analysis_daily_sw(symbol=symbol, start_date=date_text, end_date=date_text)
                clean = normalize_analysis(raw, info_map, "akshare.index_analysis_daily_sw")
                if not clean.empty:
                    clean["analysis_level"] = level
                    clean["requested_analysis_date"] = analysis_date
                    clean["resolved_analysis_date"] = date_text
                    frames.append(clean)
                    _write_csv(clean, output_root / "analysis_latest_sw" / f"{level}_{date_text}.csv")
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
        else:
            frames.append(pd.DataFrame([{"analysis_level": level, "status": "error", "error": last_error, "requested_analysis_date": analysis_date}]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_quality_summary(output_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in manifest.iterrows():
        path = Path(str(item.get("path", "")))
        status = str(item.get("status", ""))
        dataset = str(item.get("dataset", ""))
        code = str(item.get("index_code", ""))
        if status not in {"ok", "skipped_existing"} or not path.exists():
            rows.append({"index_code": code, "dataset": dataset, "status": status, "rows": 0, "quality_flag": "missing_or_error"})
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"index_code": str, "asset": str})
        quality_flag = "ok"
        start_date = ""
        end_date = ""
        duplicate_keys = 0
        null_close = 0
        nonpositive_close = 0
        full_ohlc_rows = ""
        zero_volume_amount_rows = ""
        component_count = ""
        weight_sum = ""
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            if dates.notna().any():
                start_date = str(dates.min().date())
                end_date = str(dates.max().date())
            if "index_code" in df.columns:
                duplicate_keys = int(df.duplicated(subset=["date", "index_code"]).sum())
            else:
                duplicate_keys = int(df.duplicated(subset=["date"]).sum())
        elif "inclusion_date" in df.columns:
            dates = pd.to_datetime(df["inclusion_date"], errors="coerce")
            if dates.notna().any():
                start_date = str(dates.min().date())
                end_date = str(dates.max().date())
        if "asset" in df.columns:
            duplicate_keys = int(df.duplicated(subset=["asset"]).sum())
            component_count = int(df["asset"].nunique())
        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            null_close = int(close.isna().sum())
            nonpositive_close = int((close <= 0).sum())
        if "is_full_ohlc_bar" in df.columns:
            full_ohlc_rows = int(df["is_full_ohlc_bar"].astype(str).str.lower().isin({"true", "1"}).sum())
        if "is_zero_volume_amount" in df.columns:
            zero_volume_amount_rows = int(df["is_zero_volume_amount"].astype(str).str.lower().isin({"true", "1"}).sum())
        if "weight" in df.columns:
            weight_sum = round(float(pd.to_numeric(df["weight"], errors="coerce").sum()), 4)
        if duplicate_keys or nonpositive_close:
            quality_flag = "review"
        rows.append(
            {
                "index_code": code,
                "dataset": dataset,
                "status": status,
                "rows": int(df.shape[0]),
                "start_date": start_date,
                "end_date": end_date,
                "duplicate_keys": duplicate_keys,
                "null_close": null_close,
                "nonpositive_close": nonpositive_close,
                "full_ohlc_rows": full_ohlc_rows,
                "zero_volume_amount_rows": zero_volume_amount_rows,
                "component_count": component_count,
                "weight_sum": weight_sum,
                "quality_flag": quality_flag,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        _write_csv(out, output_root / "manifests" / f"quality_summary_{_now().replace(':', '-')}.csv")
    return out


def write_dataset_plan(output_root: Path, datasets: list[str], symbols: list[str], args: argparse.Namespace) -> None:
    _write_json(
        {
            "source": DEFAULT_SOURCE,
            "created_at": _now(),
            "universe": args.universe,
            "selected_symbols": symbols,
            "datasets": datasets,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "analysis_date": args.analysis_date,
            "quality_note": (
                "SWS industry indices provide stable hierarchy, long daily history, and current constituents. "
                "Current component weights are not a full historical point-in-time constituent-weight database."
            ),
        },
        output_root / "DATASET_PLAN.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--datasets", default="classification,analysis_latest,daily,components")
    parser.add_argument("--universe", default="priority_plus_first", choices=["priority", "priority_plus_first", "all_first", "all_second"])
    parser.add_argument("--symbols")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--analysis-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = root / "data_raw" / "index" / DEFAULT_SOURCE
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    first, second, third = fetch_classification(output_root)
    info_map = build_info_map(first, second, third)
    symbols = select_symbols(args.universe, args.symbols, first, second)
    write_dataset_plan(output_root, datasets, symbols, args)

    manifest_frames = []
    if "classification" in datasets:
        for level, df in [("level1", first), ("level2", second), ("level3", third)]:
            path = output_root / "classification" / f"sw_{level}_info.csv"
            manifest_frames.append(pd.DataFrame([_manifest_row("ALL", f"classification_{level}", path, "ok", int(df.shape[0]), time.time())]))

    if "analysis_latest" in datasets:
        started = time.time()
        analysis = fetch_analysis_with_fallback(output_root, info_map, ["first", "second"], args.analysis_date)
        path = output_root / "analysis_latest_sw" / "analysis_latest_combined.csv"
        if not analysis.empty:
            _write_csv(analysis, path)
        manifest_frames.append(pd.DataFrame([_manifest_row("ALL", "analysis_latest", path, "ok" if not analysis.empty else "error", int(analysis.shape[0]), started)]))

    ak = _akshare()
    if "daily" in datasets:
        manifest_frames.append(
            run_symbol_dataset(
                output_root,
                "daily_sw",
                symbols,
                lambda code: fetch_daily_dataset(ak, code, info_map, args.start_date, args.end_date),
                args.sleep_seconds,
                args.resume,
            )
        )
    if "components" in datasets:
        manifest_frames.append(
            run_symbol_dataset(output_root, "components_current", symbols, lambda code: fetch_component_dataset(ak, code, info_map), args.sleep_seconds, args.resume)
        )

    manifest = pd.concat(manifest_frames, ignore_index=True, sort=False) if manifest_frames else pd.DataFrame()
    if not manifest.empty:
        _write_csv(manifest, output_root / "manifests" / f"manifest_all_{_now().replace(':', '-')}.csv")
    quality = build_quality_summary(output_root, manifest)
    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "symbols": len(symbols),
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
