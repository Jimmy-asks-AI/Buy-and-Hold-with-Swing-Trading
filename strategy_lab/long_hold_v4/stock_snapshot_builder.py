"""Build the current stock research snapshot from cached public data.

The current SW constituent lists are used only to seed today's watchlist. They
are explicitly not valid historical universes. Financial rows retain their
notice dates, and all network responses are cached for audit and resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .snapshot_store import snapshot_write_lock, write_snapshot_part


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
WATCHLIST_PATH = ROOT / "data_catalog" / "long_hold_v4_watchlist.csv"
SNAPSHOT_PATH = RAW_ROOT / "research_snapshot.csv"
STOCK_SNAPSHOT_PATH = RAW_ROOT / "stock_research_snapshot.csv"
ACTIVE_VALUATION_OBSERVATION_PATH = (
    RAW_ROOT / "pit_history" / "observations" / "stock_active_valuation_history_eastmoney.csv.gz"
)
ACTIVE_VALUATION_VALIDATION_DIR = (
    ROOT / "outputs" / "long_hold_v4" / "stock_active_valuation_observation_validation"
)
ACTIVE_VALUATION_VALIDATION_REPORT_PATH = ACTIVE_VALUATION_VALIDATION_DIR / "validation_report.json"
ACTIVE_VALUATION_VALIDATION_MANIFEST_PATH = ACTIVE_VALUATION_VALIDATION_DIR / "run_manifest.json"
ACTIVE_VALUATION_OUTLIERS_PATH = ACTIVE_VALUATION_VALIDATION_DIR / "cross_source_outliers.csv"
CURRENT_VALUATION_WARNING_RELATIVE_ERROR = 0.10
CURRENT_VALUATION_BLOCK_RELATIVE_ERROR = 0.20

UNIVERSE_SOURCES = {
    "bank": ROOT / "data_raw" / "index" / "akshare_sw_industry" / "components_current" / "801780.csv",
    "insurance": ROOT / "data_raw" / "index" / "akshare_sw_industry" / "components_current" / "801194.csv",
    "utility": ROOT / "data_raw" / "index" / "akshare_sw_industry" / "components_current" / "801160.csv",
}


def _akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("akshare is required by the stock snapshot builder") from exc
    return ak


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def _market_symbol(asset: str) -> str:
    code = str(asset).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _sina_symbol(asset: str) -> str:
    code = str(asset).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def build_watchlist(max_assets_per_sector: int | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sector, path in UNIVERSE_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_csv(path, encoding="utf-8-sig", dtype={"asset": str})
        required = {"asset", "asset_name", "weight", "fetched_at"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"watchlist source missing fields: {path} {missing}")
        part = data.copy()
        part["asset"] = part["asset"].astype(str).str.zfill(6)
        part["sector"] = sector
        part["current_component_source"] = str(path.relative_to(ROOT))
        part["historical_backtest_allowed"] = False
        part = part.sort_values(["weight", "asset"], ascending=[False, True])
        if max_assets_per_sector:
            part = part.head(max_assets_per_sector)
        rows.append(part[["asset", "asset_name", "sector", "weight", "fetched_at", "current_component_source", "historical_backtest_allowed"]])
    watchlist = pd.concat(rows, ignore_index=True).drop_duplicates("asset", keep="first")
    watchlist = watchlist.sort_values(["sector", "weight", "asset"], ascending=[True, False, True]).reset_index(drop=True)
    _write_csv(watchlist, WATCHLIST_PATH)
    return watchlist


def _fetch_cached(
    path: Path,
    fetcher: Callable[[], pd.DataFrame],
    source: str,
    refresh: bool,
    retries: int = 3,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    error: Exception | None = None
    for attempt in range(retries):
        try:
            data = fetcher()
            if not isinstance(data, pd.DataFrame) or (data.empty and not allow_empty):
                raise ValueError(f"empty response from {source}")
            data = data.copy()
            data["data_source"] = source
            data["fetched_at"] = _now()
            _write_csv(data, path)
            return data
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{source} failed after {retries} attempts: {error!r}")


def fetch_china_10y(as_of: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    ak = _akshare()
    path = RAW_ROOT / "macro" / "china_10y.csv"
    start = (as_of - pd.DateOffset(months=11)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    return _fetch_cached(
        path,
        lambda: ak.bond_china_yield(start_date=start, end_date=end),
        "akshare.bond_china_yield",
        refresh,
    )


def fetch_asset_raw(asset: str, as_of: pd.Timestamp, refresh: bool) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    ak = _akshare()
    symbol = _market_symbol(asset)
    start = (as_of - pd.DateOffset(years=11)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    raw = RAW_ROOT / "raw"

    def fetch_qfq_price() -> pd.DataFrame:
        try:
            return ak.stock_zh_a_hist(symbol=asset, period="daily", start_date=start, end_date=end, adjust="qfq")
        except Exception:  # noqa: BLE001
            return ak.stock_zh_a_daily(symbol=_sina_symbol(asset), start_date=start, end_date=end, adjust="qfq")

    def fetch_raw_price() -> pd.DataFrame:
        try:
            return ak.stock_zh_a_hist(symbol=asset, period="daily", start_date=start, end_date=end, adjust="")
        except Exception:  # noqa: BLE001
            return ak.stock_zh_a_daily(symbol=_sina_symbol(asset), start_date=start, end_date=end, adjust="")

    fetchers: dict[str, tuple[Path, Callable[[], pd.DataFrame], str]] = {
        "financial": (
            raw / "financial" / f"{asset}.csv",
            lambda: ak.stock_financial_analysis_indicator_em(symbol=symbol, indicator="按报告期"),
            "akshare.stock_financial_analysis_indicator_em",
        ),
        "dividend": (
            raw / "dividend" / f"{asset}.csv",
            lambda: ak.stock_dividend_cninfo(symbol=asset),
            "akshare.stock_dividend_cninfo",
        ),
        "pe": (
            raw / "valuation_pe" / f"{asset}.csv",
            lambda: ak.stock_zh_valuation_baidu(symbol=asset, indicator="市盈率(TTM)", period="近十年"),
            "akshare.stock_zh_valuation_baidu.pe_ttm",
        ),
        "pb": (
            raw / "valuation_pb" / f"{asset}.csv",
            lambda: ak.stock_zh_valuation_baidu(symbol=asset, indicator="市净率", period="近十年"),
            "akshare.stock_zh_valuation_baidu.pb",
        ),
        "price": (
            raw / "price_qfq" / f"{asset}.csv",
            fetch_qfq_price,
            "akshare.qfq.eastmoney_or_sina",
        ),
        "price_raw": (
            raw / "price_unadjusted" / f"{asset}.csv",
            fetch_raw_price,
            "akshare.unadjusted.eastmoney_or_sina",
        ),
    }
    datasets: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for name, (path, fetcher, source) in fetchers.items():
        try:
            datasets[name] = _fetch_cached(path, fetcher, source, refresh)
        except Exception as exc:  # noqa: BLE001
            datasets[name] = pd.DataFrame()
            errors[name] = repr(exc)
    return datasets, errors


def _num(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _cagr(values: pd.Series) -> float:
    clean = _num(values).dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0 or clean.iloc[-1] <= 0:
        return float("nan")
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1.0 / (len(clean) - 1)) - 1.0)


def _consecutive_fiscal_years(years: list[int], endpoint_count: int) -> bool:
    selected = [int(value) for value in years[-endpoint_count:]]
    return len(selected) == endpoint_count and all(
        current == previous + 1 for previous, current in zip(selected, selected[1:])
    )


def _window_cagr(values: pd.Series, dates: pd.Series, endpoint_count: int) -> dict[str, Any]:
    numeric = _num(values).tail(endpoint_count)
    timestamps = pd.to_datetime(dates, errors="coerce").tail(endpoint_count)
    default = {
        "value": np.nan,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "start_year": np.nan,
        "end_year": np.nan,
        "span_years": np.nan,
    }
    if len(numeric) != endpoint_count or len(timestamps) != endpoint_count or numeric.isna().any() or timestamps.isna().any():
        return default
    fiscal_years = timestamps.dt.year.astype(int).tolist()
    if not _consecutive_fiscal_years(fiscal_years, endpoint_count):
        return default
    start, end = float(numeric.iloc[0]), float(numeric.iloc[-1])
    start_date, end_date = pd.Timestamp(timestamps.iloc[0]), pd.Timestamp(timestamps.iloc[-1])
    span_years = float((end_date - start_date).days / 365.25)
    if start <= 0 or end <= 0 or span_years <= 0:
        return {
            **default,
            "start_date": start_date,
            "end_date": end_date,
            "start_year": int(start_date.year),
            "end_year": int(end_date.year),
            "span_years": span_years,
        }
    return {
        "value": float((end / start) ** (1.0 / span_years) - 1.0),
        "start_date": start_date,
        "end_date": end_date,
        "start_year": int(start_date.year),
        "end_year": int(end_date.year),
        "span_years": span_years,
    }


def _cv(values: pd.Series) -> float:
    clean = _num(values).dropna()
    if len(clean) < 2 or abs(float(clean.mean())) <= 1e-12:
        return float("nan")
    return float(clean.std(ddof=1) / abs(clean.mean()))


def _annual_financials(data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    required = {"REPORT_DATE", "REPORT_DATE_NAME", "NOTICE_DATE"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    out = data.copy()
    out["REPORT_DATE"] = pd.to_datetime(out["REPORT_DATE"], errors="coerce")
    out["NOTICE_DATE"] = pd.to_datetime(out["NOTICE_DATE"], errors="coerce")
    update_date = pd.to_datetime(out["UPDATE_DATE"], errors="coerce") if "UPDATE_DATE" in out.columns else out["NOTICE_DATE"]
    out["PIT_AVAILABLE_DATE"] = pd.concat([out["NOTICE_DATE"], update_date], axis=1).max(axis=1)
    out = out[
        (out["PIT_AVAILABLE_DATE"] <= as_of)
        & out["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)
    ].copy()
    out["fiscal_year"] = out["REPORT_DATE"].dt.year
    duplicate_years = sorted(
        out.loc[out["fiscal_year"].duplicated(keep=False), "fiscal_year"].dropna().astype(int).unique().tolist()
    )
    out = out.sort_values(["fiscal_year", "PIT_AVAILABLE_DATE"]).drop_duplicates("fiscal_year", keep="last")
    result = out.sort_values("fiscal_year").tail(6)
    result.attrs["duplicate_fiscal_years"] = duplicate_years
    return result


def _financial_window_ready(annual: pd.DataFrame, endpoint_count: int) -> bool:
    if annual.empty:
        return False
    years = pd.to_numeric(annual.get("fiscal_year"), errors="coerce")
    if years.isna().any():
        return False
    selected_years = years.astype(int).tolist()[-endpoint_count:]
    duplicate_years = {int(value) for value in annual.attrs.get("duplicate_fiscal_years", [])}
    if duplicate_years.intersection(selected_years):
        return False
    return _consecutive_fiscal_years(selected_years, endpoint_count)


def _dividend_metrics(data: pd.DataFrame, annual_years: list[int], as_of: pd.Timestamp) -> dict[str, Any]:
    defaults = {
        "dividend_years_5y": np.nan,
        "dividend_cagr_5y": np.nan,
        "dividend_cut_count_5y": np.nan,
        "latest_fiscal_dps": np.nan,
        "trailing_12m_dps": np.nan,
        "dividend_available_date": pd.NaT,
        "dividend_cagr_5y_start_date": pd.NaT,
        "dividend_cagr_5y_end_date": pd.NaT,
        "dividend_cagr_5y_start_year": np.nan,
        "dividend_cagr_5y_end_year": np.nan,
        "dividend_cagr_5y_span_years": np.nan,
        "dividend_history_consecutive": False,
    }
    required = {"实施方案公告日期", "派息比例", "派息日", "报告时间"}
    if not required.issubset(data.columns) or not _consecutive_fiscal_years(annual_years, 6):
        return defaults
    out = data.copy()
    out["announcement_date"] = pd.to_datetime(out["实施方案公告日期"], errors="coerce")
    out["pay_date"] = pd.to_datetime(out["派息日"], errors="coerce")
    out["cash_per_10"] = _num(out["派息比例"])
    out["fiscal_year"] = _num(out["报告时间"].astype(str).str.extract(r"(\d{4})", expand=False))
    out = out[(out["announcement_date"] <= as_of) & (out["cash_per_10"] > 0)].copy()
    if out.empty:
        return defaults
    selected_years = annual_years[-6:]
    dps_by_year = (out.groupby("fiscal_year")["cash_per_10"].sum() / 10.0).reindex(selected_years, fill_value=0.0)
    dividend_cagr = _window_cagr(
        dps_by_year.reset_index(drop=True),
        pd.Series([pd.Timestamp(year=int(year), month=12, day=31) for year in selected_years]),
        6,
    )
    cuts = int(((dps_by_year / dps_by_year.shift(1) - 1.0) < -0.10).fillna(False).sum())
    trailing = out[(out["pay_date"] > as_of - pd.Timedelta(days=365)) & (out["pay_date"] <= as_of)]["cash_per_10"].sum() / 10.0
    return {
        "dividend_years_5y": int((dps_by_year.tail(5) > 0).sum()),
        "dividend_cagr_5y": dividend_cagr["value"],
        "dividend_cut_count_5y": cuts,
        "latest_fiscal_dps": float(dps_by_year.iloc[-1]),
        "trailing_12m_dps": float(trailing),
        "dividend_available_date": out["announcement_date"].max(),
        "dividend_cagr_5y_start_date": dividend_cagr["start_date"],
        "dividend_cagr_5y_end_date": dividend_cagr["end_date"],
        "dividend_cagr_5y_start_year": dividend_cagr["start_year"],
        "dividend_cagr_5y_end_year": dividend_cagr["end_year"],
        "dividend_cagr_5y_span_years": dividend_cagr["span_years"],
        "dividend_history_consecutive": True,
    }


def _valuation_metrics(data: pd.DataFrame, as_of: pd.Timestamp, prefix: str) -> dict[str, Any]:
    defaults = {f"current_{prefix}": np.nan, f"{prefix}_percentile_5y": np.nan, f"{prefix}_date": pd.NaT}
    if not {"date", "value"}.issubset(data.columns):
        return defaults
    out = data.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = _num(out["value"])
    out = out[(out["date"] <= as_of) & out["value"].notna()].sort_values("date")
    if out.empty:
        return defaults
    latest = out.iloc[-1]
    current = float(latest["value"])
    lookback = out[
        (out["date"] >= pd.Timestamp(latest["date"]) - pd.DateOffset(years=5))
        & (out["value"] > 0)
    ]
    percentile = (
        float((lookback["value"] <= current).mean())
        if current > 0 and not lookback.empty
        else np.nan
    )
    return {
        f"current_{prefix}": current,
        f"{prefix}_percentile_5y": percentile,
        f"{prefix}_date": latest["date"],
    }


def current_valuation_metrics_from_observation(
    observation: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    required = {
        "date",
        "asset",
        "pe_ttm",
        "pb_mrq",
        "available_date",
        "source_observed_at",
        "historical_backtest_allowed",
        "pit_actionable",
    }
    missing = sorted(required.difference(observation.columns))
    if missing:
        raise ValueError(f"active valuation observation missing fields: {missing}")
    frame = observation.copy()
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce").dt.normalize()
    frame["pe_ttm"] = _num(frame["pe_ttm"])
    frame["pb_mrq"] = _num(frame["pb_mrq"])
    historical_allowed = frame["historical_backtest_allowed"].astype(str).str.lower().eq("true")
    pit_actionable = frame["pit_actionable"].astype(str).str.lower().eq("true")
    if historical_allowed.any() or pit_actionable.any():
        raise ValueError("current-final valuation observation has an invalid PIT promotion flag")
    frame = frame[
        frame["date"].notna()
        & frame["available_date"].notna()
        & frame["date"].le(as_of)
        & frame["available_date"].le(as_of)
    ].copy()
    if frame.empty:
        raise ValueError("no active valuation observations are available by the decision date")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("active valuation observation contains duplicate keys")

    rows: list[dict[str, Any]] = []
    for asset, group in frame.groupby("asset", sort=True):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        start = pd.Timestamp(latest["date"]) - pd.DateOffset(years=5)
        lookback = group[group["date"].ge(start)]
        pe = float(latest["pe_ttm"]) if pd.notna(latest["pe_ttm"]) else np.nan
        pb = float(latest["pb_mrq"]) if pd.notna(latest["pb_mrq"]) else np.nan
        positive_pe = lookback.loc[lookback["pe_ttm"].gt(0), "pe_ttm"]
        positive_pb = lookback.loc[lookback["pb_mrq"].gt(0), "pb_mrq"]
        rows.append(
            {
                "asset": asset,
                "overlay_current_pe": pe,
                "overlay_current_pb": pb,
                "overlay_pe_percentile_5y": (
                    float(positive_pe.le(pe).mean()) if pe > 0 and not positive_pe.empty else np.nan
                ),
                "overlay_pb_percentile_5y": (
                    float(positive_pb.le(pb).mean()) if pb > 0 and not positive_pb.empty else np.nan
                ),
                "valuation_market_date": pd.Timestamp(latest["date"]),
                "valuation_available_date": pd.Timestamp(latest["available_date"]),
                "valuation_source_observed_at": str(latest["source_observed_at"]),
                "valuation_history_rows_5y": int(len(lookback)),
            }
        )
    return pd.DataFrame(rows)


def _symmetric_relative_error(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = pd.concat(
        [left.abs(), right.abs(), pd.Series(1.0, index=left.index)], axis=1
    ).max(axis=1)
    return (left - right).abs() / denominator


def apply_current_valuation_overlay(
    snapshot: pd.DataFrame,
    as_of: pd.Timestamp,
    observation_path: Path = ACTIVE_VALUATION_OBSERVATION_PATH,
    validation_report_path: Path = ACTIVE_VALUATION_VALIDATION_REPORT_PATH,
    validation_manifest_path: Path = ACTIVE_VALUATION_VALIDATION_MANIFEST_PATH,
    outliers_path: Path = ACTIVE_VALUATION_OUTLIERS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if snapshot.empty:
        raise ValueError("cannot apply current valuation overlay to an empty stock snapshot")
    for path in [observation_path, validation_report_path, validation_manifest_path, outliers_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    if report.get("current_historical_percentile_diagnostic_allowed") is not True:
        raise ValueError("active valuation validation did not permit current percentile diagnostics")
    if report.get("historical_backtest_allowed") is not False:
        raise ValueError("active valuation validation has an invalid historical backtest flag")
    manifest_inputs = {item["path"]: item["sha256"] for item in manifest.get("inputs", [])}
    manifest_outputs = {item["path"]: item["sha256"] for item in manifest.get("outputs", [])}
    observation_key = str(observation_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    report_key = str(validation_report_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    if manifest_inputs.get(observation_key) != _sha256(observation_path):
        raise ValueError("active valuation observation hash is absent or mismatched in validation manifest")
    if manifest_outputs.get(report_key) != _sha256(validation_report_path):
        raise ValueError("active valuation report hash is absent or mismatched in validation manifest")

    observation = pd.read_csv(
        observation_path,
        compression="gzip",
        dtype={"asset": str},
        low_memory=False,
    )
    overlay = current_valuation_metrics_from_observation(observation, as_of)
    result = snapshot.copy()
    result["asset"] = result["asset"].astype(str).str.zfill(6)
    result = result.merge(overlay, on="asset", how="left", validate="one_to_one")
    missing_assets = result.loc[result["overlay_current_pb"].isna(), "asset"].tolist()
    if missing_assets:
        raise ValueError(f"current valuation overlay is incomplete for assets: {missing_assets[:10]}")

    result["baidu_current_pe"] = pd.to_numeric(result["current_pe"], errors="coerce")
    result["baidu_current_pb"] = pd.to_numeric(result["current_pb"], errors="coerce")
    result["baidu_pe_percentile_5y"] = pd.to_numeric(result["pe_percentile_5y"], errors="coerce")
    result["baidu_pb_percentile_5y"] = pd.to_numeric(result["pb_percentile_5y"], errors="coerce")
    result["valuation_current_pe_relative_error"] = _symmetric_relative_error(
        result["overlay_current_pe"], result["baidu_current_pe"]
    )
    result["valuation_current_pb_relative_error"] = _symmetric_relative_error(
        result["overlay_current_pb"], result["baidu_current_pb"]
    )
    sign_mismatch = (
        result["overlay_current_pe"].notna()
        & result["baidu_current_pe"].notna()
        & result["overlay_current_pe"].ne(0)
        & result["baidu_current_pe"].ne(0)
        & result["overlay_current_pe"].gt(0).ne(result["baidu_current_pe"].gt(0))
    )
    maximum_error = result[
        ["valuation_current_pe_relative_error", "valuation_current_pb_relative_error"]
    ].max(axis=1)
    block = sign_mismatch | maximum_error.gt(CURRENT_VALUATION_BLOCK_RELATIVE_ERROR)
    warning = ~block & maximum_error.gt(CURRENT_VALUATION_WARNING_RELATIVE_ERROR)
    result["valuation_current_cross_source_status"] = np.select(
        [block, warning], ["block", "warning"], default="pass"
    )
    result["valuation_current_cross_source_reasons"] = np.select(
        [sign_mismatch, block, warning],
        ["pe_sign_mismatch", "relative_error_above_20pct", "relative_error_above_10pct"],
        default="",
    )

    outliers = pd.read_csv(outliers_path, dtype={"asset": str}, low_memory=False)
    if outliers.empty:
        historical_warnings = pd.Series(dtype=str)
    else:
        outliers["asset"] = outliers["asset"].astype(str).str.zfill(6)
        outliers["detail"] = outliers["source"].astype(str) + ":" + outliers["warning_flags"].astype(str)
        historical_warnings = outliers.groupby("asset")["detail"].agg("|".join)
    result["valuation_historical_cross_source_warnings"] = (
        result["asset"].map(historical_warnings).fillna("")
    )
    result["current_pe"] = result["overlay_current_pe"]
    result["current_pb"] = result["overlay_current_pb"]
    result["pe_percentile_5y"] = result["overlay_pe_percentile_5y"]
    result["pb_percentile_5y"] = result["overlay_pb_percentile_5y"]
    result["sector_pe_percentile"] = result["current_pe"].where(result["current_pe"].gt(0)).groupby(
        result["sector"]
    ).rank(pct=True, method="average")
    result["sector_pb_percentile"] = result["current_pb"].where(result["current_pb"].gt(0)).groupby(
        result["sector"]
    ).rank(pct=True, method="average")
    existing_available = pd.to_datetime(result["available_date"], errors="coerce")
    result["available_date"] = pd.concat(
        [existing_available, result["valuation_available_date"]], axis=1
    ).max(axis=1).dt.strftime("%Y-%m-%d")
    result["valuation_market_date"] = pd.to_datetime(result["valuation_market_date"]).dt.strftime("%Y-%m-%d")
    result["valuation_available_date"] = pd.to_datetime(result["valuation_available_date"]).dt.strftime("%Y-%m-%d")
    result["valuation_source"] = "eastmoney_current_final_snapshot_cross_source_diagnostic"
    result["valuation_history_pit_actionable"] = False
    result["source_note"] = result["source_note"].astype(str) + " + validated Eastmoney current valuation overlay"
    result = result.drop(
        columns=[
            "overlay_current_pe",
            "overlay_current_pb",
            "overlay_pe_percentile_5y",
            "overlay_pb_percentile_5y",
        ]
    )
    summary = {
        "assets": int(len(result)),
        "pass_assets": int(result["valuation_current_cross_source_status"].eq("pass").sum()),
        "warning_assets": int(result["valuation_current_cross_source_status"].eq("warning").sum()),
        "blocked_assets": int(result["valuation_current_cross_source_status"].eq("block").sum()),
        "nonpositive_pe_assets": int(pd.to_numeric(result["current_pe"], errors="coerce").le(0).sum()),
        "historical_warning_assets": int(
            result["valuation_historical_cross_source_warnings"].ne("").sum()
        ),
        "observation_path": str(observation_path.relative_to(ROOT)),
        "validation_report_path": str(validation_report_path.relative_to(ROOT)),
        "historical_backtest_allowed": False,
    }
    return result, summary


def _normalize_price(
    data: pd.DataFrame,
    asset: str,
    as_of: pd.Timestamp,
    write_output: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    out = data.rename(columns=rename).copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(out.columns):
        raise ValueError(f"price fields missing for {asset}: {sorted(required.difference(out.columns))}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = _num(out[col])
    out = out[(out["date"] <= as_of) & out["close"].notna()].sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise ValueError(f"no price rows for {asset}")
    recent = out.tail(756)
    ret = recent["close"].pct_change()
    drawdown = recent["close"] / recent["close"].cummax() - 1.0
    latest = out.iloc[-1]
    metrics = {
        "current_close": float(latest["close"]),
        "latest_price_date": latest["date"],
        "latest_volume": float(latest["volume"]),
        "latest_amount_cny": float(latest["amount"]),
        "history_years": float((out["date"].iloc[-1] - out["date"].iloc[0]).days / 365.25),
        "annual_vol_3y": float(ret.std(ddof=1) * math.sqrt(252.0)),
        "max_drawdown_3y": float(drawdown.min()),
    }
    normalized = out[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    normalized["asset"] = asset
    normalized["asset_type"] = "stock"
    normalized["return_basis"] = "qfq_adjusted"
    normalized["price_basis"] = "qfq_adjusted"
    normalized["available_date"] = normalized["date"]
    normalized["data_source"] = "akshare.stock_zh_a_hist.qfq"
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
    if write_output:
        _write_csv(normalized, RAW_ROOT / "prices" / f"{asset}.csv")
    return normalized, metrics


def _normalize_execution_price(data: pd.DataFrame, asset: str, as_of: pd.Timestamp, write_output: bool = True) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    out = data.rename(columns=rename).copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"unadjusted execution price fields missing for {asset}: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for column in required.difference({"date"}):
        out[column] = _num(out[column])
    out = out[(out["date"] <= as_of) & out[list(required.difference({"date"}))].notna().all(axis=1)].copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    if out.empty or (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"invalid unadjusted execution prices for {asset}")
    normalized = out[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    normalized["asset"] = asset
    normalized["asset_type"] = "stock"
    normalized["return_basis"] = "unadjusted_executable"
    normalized["price_basis"] = "unadjusted_executable"
    normalized["available_date"] = normalized["date"]
    normalized["data_source"] = "akshare.stock_zh_a_hist.unadjusted"
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
    if write_output:
        _write_csv(normalized, RAW_ROOT / "execution_prices" / f"{asset}.csv")
    return normalized


def _latest_number(row: pd.Series, columns: list[str], scale: float = 1.0) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value) * scale
    return float("nan")


def build_stock_row(
    watch: pd.Series,
    raw: dict[str, pd.DataFrame],
    china_10y: float,
    china_10y_date: pd.Timestamp,
    as_of: pd.Timestamp,
    write_price_output: bool = True,
) -> dict[str, Any]:
    asset = str(watch["asset"]).zfill(6)
    sector = str(watch["sector"])
    _, price = _normalize_price(raw["price"], asset, as_of, write_output=write_price_output)
    _normalize_execution_price(raw["price_raw"], asset, as_of, write_output=write_price_output)
    annual = _annual_financials(raw["financial"], as_of)
    years = annual["fiscal_year"].astype(int).tolist() if not annual.empty else []
    five_year_financials_ready = _financial_window_ready(annual, 6)
    three_year_financials_ready = _financial_window_ready(annual, 4)
    dividend = _dividend_metrics(raw["dividend"], years if five_year_financials_ready else [], as_of)
    pe = _valuation_metrics(raw["pe"], as_of, "pe")
    pb = _valuation_metrics(raw["pb"], as_of, "pb")

    result: dict[str, Any] = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "available_date": pd.NaT,
        "asset": asset,
        "name": str(watch["asset_name"]),
        "asset_type": "stock",
        "sector": sector,
        "is_tradeable": (as_of - pd.Timestamp(price["latest_price_date"])).days <= 5 and price["latest_volume"] > 0,
        "is_st": "ST" in str(watch["asset_name"]).upper(),
        "history_years": price["history_years"],
        "positive_profit_years_5y": np.nan,
        "dividend_years_5y": dividend["dividend_years_5y"],
        "dividend_yield": dividend["trailing_12m_dps"] / price["current_close"] if price["current_close"] > 0 else np.nan,
        "dividend_cagr_5y": dividend["dividend_cagr_5y"],
        "dividend_cagr_5y_start_date": dividend["dividend_cagr_5y_start_date"],
        "dividend_cagr_5y_end_date": dividend["dividend_cagr_5y_end_date"],
        "dividend_cagr_5y_start_year": dividend["dividend_cagr_5y_start_year"],
        "dividend_cagr_5y_end_year": dividend["dividend_cagr_5y_end_year"],
        "dividend_cagr_5y_span_years": dividend["dividend_cagr_5y_span_years"],
        "dividend_history_consecutive": dividend["dividend_history_consecutive"],
        "dividend_cut_count_5y": dividend["dividend_cut_count_5y"],
        "payout_ratio": np.nan,
        "roe_mean_5y": np.nan,
        "roe_std_5y": np.nan,
        "revenue_cagr_5y": np.nan,
        "revenue_cagr_5y_start_date": pd.NaT,
        "revenue_cagr_5y_end_date": pd.NaT,
        "revenue_cagr_5y_start_year": np.nan,
        "revenue_cagr_5y_end_year": np.nan,
        "revenue_cagr_5y_span_years": np.nan,
        "profit_cagr_5y": np.nan,
        "profit_cagr_5y_start_date": pd.NaT,
        "profit_cagr_5y_end_date": pd.NaT,
        "profit_cagr_5y_start_year": np.nan,
        "profit_cagr_5y_end_year": np.nan,
        "profit_cagr_5y_span_years": np.nan,
        "profit_cv_5y": np.nan,
        "financial_years_consecutive": False,
        "financial_metric_status": "insufficient_or_nonconsecutive_fiscal_years",
        "current_pe": pe["current_pe"],
        "current_pb": pb["current_pb"],
        "pe_percentile_5y": pe["pe_percentile_5y"],
        "pb_percentile_5y": pb["pb_percentile_5y"],
        "sector_pe_percentile": np.nan,
        "sector_pb_percentile": np.nan,
        "china_10y_yield": china_10y,
        "yield_spread_cn10y": np.nan,
        "annual_vol_3y": price["annual_vol_3y"],
        "max_drawdown_3y": price["max_drawdown_3y"],
        "npl_ratio": np.nan,
        "provision_coverage": np.nan,
        "core_tier1_ratio": np.nan,
        "solvency_ratio": np.nan,
        "new_business_value_cagr_3y": np.nan,
        "new_business_value_cagr_3y_start_date": pd.NaT,
        "new_business_value_cagr_3y_end_date": pd.NaT,
        "new_business_value_cagr_3y_start_year": np.nan,
        "new_business_value_cagr_3y_end_year": np.nan,
        "new_business_value_cagr_3y_span_years": np.nan,
        "debt_to_assets": np.nan,
        "interest_coverage": np.nan,
        "fcf_dividend_coverage": np.nan,
        "current_universe_only": True,
        "historical_backtest_allowed": False,
        "source_note": "Eastmoney financial notice dates + CNInfo dividends + Baidu valuation + AkShare separated qfq research and unadjusted execution prices",
        "source_errors": "",
    }
    if not annual.empty:
        profit = _num(annual.get("PARENTNETPROFIT"))
        revenue = _num(annual.get("TOTALOPERATEREVE"))
        roe = _num(annual.get("ROEJQ")) / 100.0
        latest = annual.iloc[-1]
        revenue_cagr = _window_cagr(revenue, annual["REPORT_DATE"], 6) if five_year_financials_ready else {}
        profit_cagr = _window_cagr(profit, annual["REPORT_DATE"], 6) if five_year_financials_ready else {}
        result.update(
            {
                "positive_profit_years_5y": int((profit.tail(5) > 0).sum()) if five_year_financials_ready else np.nan,
                "roe_mean_5y": float(roe.tail(5).mean()) if five_year_financials_ready else np.nan,
                "roe_std_5y": float(roe.tail(5).std(ddof=1)) if five_year_financials_ready else np.nan,
                "revenue_cagr_5y": revenue_cagr.get("value", np.nan),
                "revenue_cagr_5y_start_date": revenue_cagr.get("start_date", pd.NaT),
                "revenue_cagr_5y_end_date": revenue_cagr.get("end_date", pd.NaT),
                "revenue_cagr_5y_start_year": revenue_cagr.get("start_year", np.nan),
                "revenue_cagr_5y_end_year": revenue_cagr.get("end_year", np.nan),
                "revenue_cagr_5y_span_years": revenue_cagr.get("span_years", np.nan),
                "profit_cagr_5y": profit_cagr.get("value", np.nan),
                "profit_cagr_5y_start_date": profit_cagr.get("start_date", pd.NaT),
                "profit_cagr_5y_end_date": profit_cagr.get("end_date", pd.NaT),
                "profit_cagr_5y_start_year": profit_cagr.get("start_year", np.nan),
                "profit_cagr_5y_end_year": profit_cagr.get("end_year", np.nan),
                "profit_cagr_5y_span_years": profit_cagr.get("span_years", np.nan),
                "profit_cv_5y": _cv(profit.tail(5)) if five_year_financials_ready else np.nan,
                "financial_years_consecutive": five_year_financials_ready,
                "financial_metric_status": "ready" if five_year_financials_ready else "nonconsecutive_fiscal_years_blocked",
            }
        )
        eps = _latest_number(latest, ["EPSJB"])
        if eps > 0 and pd.notna(dividend["latest_fiscal_dps"]):
            result["payout_ratio"] = float(dividend["latest_fiscal_dps"] / eps)
        if sector == "bank":
            result["npl_ratio"] = _latest_number(latest, ["NONPERLOAN", "NON_PERFORMING_LOAN"], 0.01)
            result["provision_coverage"] = _latest_number(latest, ["BLDKBBL"], 0.01)
            result["core_tier1_ratio"] = _latest_number(latest, ["HXYJBCZL"], 0.01)
        elif sector == "insurance":
            result["solvency_ratio"] = _latest_number(latest, ["SOLVENCY_AR"], 0.01)
            if three_year_financials_ready:
                nbv_cagr = _window_cagr(_num(annual.get("NBV_LIFE")), annual["REPORT_DATE"], 4)
                result["new_business_value_cagr_3y"] = nbv_cagr["value"]
                result["new_business_value_cagr_3y_start_date"] = nbv_cagr["start_date"]
                result["new_business_value_cagr_3y_end_date"] = nbv_cagr["end_date"]
                result["new_business_value_cagr_3y_start_year"] = nbv_cagr["start_year"]
                result["new_business_value_cagr_3y_end_year"] = nbv_cagr["end_year"]
                result["new_business_value_cagr_3y_span_years"] = nbv_cagr["span_years"]
        else:
            result["debt_to_assets"] = _latest_number(latest, ["ZCFZL"], 0.01)
            result["interest_coverage"] = _latest_number(latest, ["INTEREST_COVERAGE_RATIO"])
            fcf = _latest_number(latest, ["FCFF_FORWARD"])
            net_profit = _latest_number(latest, ["PARENTNETPROFIT"])
            if fcf > 0 and eps > 0 and net_profit > 0 and dividend["latest_fiscal_dps"] > 0:
                estimated_shares = net_profit / eps
                result["fcf_dividend_coverage"] = fcf / (estimated_shares * dividend["latest_fiscal_dps"])

        available_dates = [
            annual["PIT_AVAILABLE_DATE"].max(),
            dividend["dividend_available_date"],
            pe["pe_date"],
            pb["pb_date"],
            price["latest_price_date"],
            china_10y_date,
        ]
        valid_dates = [pd.Timestamp(value) for value in available_dates if pd.notna(value)]
        result["available_date"] = max(valid_dates).strftime("%Y-%m-%d") if valid_dates else pd.NaT
    if not five_year_financials_ready:
        prior_errors = str(result.get("source_errors", "")).strip()
        result["source_errors"] = ";".join(filter(None, [prior_errors, "nonconsecutive_fiscal_years_blocked"]))
    if pd.notna(result["dividend_yield"]):
        result["yield_spread_cn10y"] = float(result["dividend_yield"] - china_10y)
    return result


def china_10y_latest(data: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, pd.Timestamp]:
    required = {"曲线名称", "日期", "10年"}
    if not required.issubset(data.columns):
        raise ValueError(f"China yield response missing fields: {sorted(required.difference(data.columns))}")
    out = data.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out["10年"] = _num(out["10年"])
    out = out[(out["日期"] <= as_of) & out["曲线名称"].astype(str).eq("中债国债收益率曲线") & out["10年"].notna()].sort_values("日期")
    if out.empty:
        raise ValueError("no China government 10Y yield on or before as_of")
    latest = out.iloc[-1]
    return float(latest["10年"] / 100.0), pd.Timestamp(latest["日期"])


def finalize_snapshot(rows: list[dict[str, Any]]) -> pd.DataFrame:
    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot
    snapshot["sector_pe_percentile"] = snapshot["current_pe"].where(snapshot["current_pe"].gt(0)).groupby(
        snapshot["sector"]
    ).rank(pct=True, method="average")
    snapshot["sector_pb_percentile"] = snapshot["current_pb"].where(snapshot["current_pb"].gt(0)).groupby(
        snapshot["sector"]
    ).rank(pct=True, method="average")
    columns = [
        "as_of_date", "available_date", "asset", "name", "asset_type", "sector", "is_tradeable", "is_st", "history_years",
        "positive_profit_years_5y", "dividend_years_5y", "dividend_yield", "dividend_cagr_5y", "dividend_cagr_5y_start_date",
        "dividend_cagr_5y_end_date", "dividend_cagr_5y_start_year", "dividend_cagr_5y_end_year", "dividend_cagr_5y_span_years", "dividend_history_consecutive", "dividend_cut_count_5y", "payout_ratio",
        "roe_mean_5y", "roe_std_5y", "revenue_cagr_5y", "revenue_cagr_5y_start_date", "revenue_cagr_5y_end_date",
        "revenue_cagr_5y_start_year", "revenue_cagr_5y_end_year", "revenue_cagr_5y_span_years", "profit_cagr_5y", "profit_cagr_5y_start_date", "profit_cagr_5y_end_date",
        "profit_cagr_5y_start_year", "profit_cagr_5y_end_year", "profit_cagr_5y_span_years", "profit_cv_5y", "financial_years_consecutive", "financial_metric_status", "current_pe", "current_pb",
        "pe_percentile_5y", "pb_percentile_5y", "sector_pe_percentile", "sector_pb_percentile", "china_10y_yield", "yield_spread_cn10y",
        "annual_vol_3y", "max_drawdown_3y", "npl_ratio", "provision_coverage", "core_tier1_ratio", "solvency_ratio",
        "new_business_value_cagr_3y", "new_business_value_cagr_3y_start_date", "new_business_value_cagr_3y_end_date",
        "new_business_value_cagr_3y_start_year", "new_business_value_cagr_3y_end_year", "new_business_value_cagr_3y_span_years", "debt_to_assets", "interest_coverage", "fcf_dividend_coverage", "current_universe_only",
        "historical_backtest_allowed", "source_note", "source_errors",
    ]
    return snapshot.reindex(columns=columns).sort_values(["sector", "asset"]).reset_index(drop=True)


def _run_builder(
    as_of: pd.Timestamp,
    max_assets_per_sector: int | None,
    refresh: bool,
    sleep_seconds: float,
) -> dict[str, Any]:
    watchlist = build_watchlist(max_assets_per_sector=max_assets_per_sector)
    yield_data = fetch_china_10y(as_of, refresh=refresh)
    china_10y, china_10y_date = china_10y_latest(yield_data, as_of)
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, watch in watchlist.iterrows():
        asset = str(watch["asset"]).zfill(6)
        started = time.time()
        try:
            raw, dataset_errors = fetch_asset_raw(asset, as_of, refresh=refresh)
            row = build_stock_row(watch, raw, china_10y, china_10y_date, as_of)
            metric_errors = str(row.get("source_errors", "")).strip()
            dataset_error_text = json.dumps(dataset_errors, ensure_ascii=False, sort_keys=True) if dataset_errors else ""
            row["source_errors"] = ";".join(filter(None, [metric_errors, dataset_error_text]))
            rows.append(row)
            status, error = ("partial", row["source_errors"]) if row["source_errors"] else ("ok", "")
        except Exception as exc:  # noqa: BLE001
            status, error = "error", repr(exc)
        manifest.append(
            {
                "asset": asset,
                "sector": watch["sector"],
                "status": status,
                "error": error,
                "seconds": round(time.time() - started, 3),
                "completed": index + 1,
                "total": len(watchlist),
                "fetched_at": _now(),
            }
        )
        print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)
        if sleep_seconds > 0 and index + 1 < len(watchlist):
            time.sleep(sleep_seconds)

    snapshot = finalize_snapshot(rows)
    snapshot, valuation_overlay = apply_current_valuation_overlay(snapshot, as_of)
    combined = write_snapshot_part(
        "stock",
        snapshot,
        builder_config={
            "as_of_date": str(as_of.date()),
            "max_assets_per_sector": max_assets_per_sector,
            "refresh": refresh,
        },
        builder_code_paths=[Path(__file__), Path(__file__).with_name("snapshot_store.py")],
    )
    manifest_frame = pd.DataFrame(manifest)
    manifest_path = RAW_ROOT / "manifests" / f"stock_snapshot_{as_of.strftime('%Y%m%d')}.csv"
    _write_csv(manifest_frame, manifest_path)
    price_paths = [RAW_ROOT / "prices" / f"{asset}.csv" for asset in snapshot["asset"].astype(str)]
    execution_price_paths = [RAW_ROOT / "execution_prices" / f"{asset}.csv" for asset in snapshot["asset"].astype(str)]
    price_paths.extend(execution_price_paths)
    missing_prices = [path for path in price_paths if not path.is_file()]
    if missing_prices:
        raise RuntimeError(f"stock source manifest missing normalized prices: {missing_prices[:5]}")
    source_paths = sorted(
        set((RAW_ROOT / "raw").rglob("*.csv"))
        | set((RAW_ROOT / "macro").rglob("*.csv"))
        | set(UNIVERSE_SOURCES.values())
        | {
            WATCHLIST_PATH,
            STOCK_SNAPSHOT_PATH,
            manifest_path,
            ACTIVE_VALUATION_OBSERVATION_PATH,
            ACTIVE_VALUATION_VALIDATION_REPORT_PATH,
            ACTIVE_VALUATION_VALIDATION_MANIFEST_PATH,
            ACTIVE_VALUATION_OUTLIERS_PATH,
            *price_paths,
        }
    )
    source_manifest = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "generated_at": _now(),
        "input_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "code_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in sorted([Path(__file__), Path(__file__).with_name("snapshot_store.py")])
        ],
        "current_universe_only": True,
        "historical_backtest_allowed": False,
        "current_valuation_overlay": valuation_overlay,
    }
    source_manifest_path = RAW_ROOT / "manifests" / f"stock_source_manifest_{as_of.strftime('%Y%m%d')}.json"
    source_manifest_latest_path = RAW_ROOT / "manifests" / "stock_source_manifest_latest.json"
    payload = json.dumps(source_manifest, ensure_ascii=False, indent=2)
    source_manifest_path.write_text(payload, encoding="utf-8")
    source_manifest_latest_path.write_text(payload, encoding="utf-8")
    result = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "watchlist_rows": len(watchlist),
        "snapshot_rows": len(snapshot),
        "combined_snapshot_rows": len(combined),
        "success": int(manifest_frame["status"].isin(["ok", "partial"]).sum()),
        "partial": int((manifest_frame["status"] == "partial").sum()),
        "errors": int((manifest_frame["status"] == "error").sum()),
        "china_10y": china_10y,
        "china_10y_date": str(china_10y_date.date()),
        "snapshot_path": str(SNAPSHOT_PATH),
        "manifest_path": str(manifest_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_latest_path": str(source_manifest_latest_path),
        "source_manifest_input_count": len(source_manifest["input_files"]),
        "current_universe_only": True,
        "historical_backtest_allowed": False,
        "current_valuation_overlay": valuation_overlay,
    }
    (RAW_ROOT / "stock_snapshot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_builder(
    as_of: pd.Timestamp,
    max_assets_per_sector: int | None,
    refresh: bool,
    sleep_seconds: float,
) -> dict[str, Any]:
    with snapshot_write_lock(RAW_ROOT / ".stock_snapshot_builder.lock"):
        return _run_builder(as_of, max_assets_per_sector, refresh, sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--max-assets-per-sector", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    args = parser.parse_args()
    result = run_builder(pd.Timestamp(args.as_of).normalize(), args.max_assets_per_sector, args.refresh, args.sleep_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
