"""Build a current, non-promotable snapshot for domestic dividend ETFs.

The builder uses exchange/fund/index sources, reconstructs a dividend-adjusted
ETF price series, and records field-level dates. It deliberately does not turn
today's ETF list or full-history downloads into a historical PIT universe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from .etf_index_registry import INDEX_REGISTRY_PATH, active_index_map
from .snapshot_store import write_snapshot_part
from .stock_snapshot_builder import _fetch_cached, _now, _write_csv, china_10y_latest, fetch_china_10y


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
ETF_SNAPSHOT_PATH = RAW_ROOT / "etf_research_snapshot.csv"
WATCHLIST_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_watchlist.csv"
SPOT_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
ETF_KEYWORDS = re.compile(r"(?:红利|股息|高股息)")
CROSS_BORDER = re.compile(r"(?:港股|港股通|恒生|香港|H股|中概|纳斯达克|标普500|日经|德国|沙特|QDII|海外)", re.I)
MAX_DISTRIBUTION_ALIGNMENT_CALENDAR_DAYS = 31

# Only registry rows with complete identity evidence and ready local caches are
# activated. Factsheet-only or guessed mappings remain unavailable.
INDEX_MAP = active_index_map()


def _akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("akshare is required by the ETF snapshot builder") from exc
    return ak


def _without_proxy(call: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    proxy_environment = {key: value for key, value in os.environ.items() if "proxy" in key.lower()}
    for key in proxy_environment:
        os.environ.pop(key, None)
    try:
        return call()
    finally:
        os.environ.update(proxy_environment)


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def _number(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def _percent(value: Any) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)%", str(value))
    return float(match.group(1)) / 100.0 if match else float("nan")


def _cn_date(value: Any) -> pd.Timestamp:
    match = re.search(r"(\d{4})年(\d{2})月(\d{2})日", str(value))
    return pd.Timestamp("-".join(match.groups())) if match else pd.NaT


def _cn_amount(value: Any) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(万|亿)?元", str(value))
    if not match:
        return float("nan")
    scale = {None: 1.0, "万": 1e4, "亿": 1e8}[match.group(2)]
    return float(match.group(1)) * scale


def _distribution_count(value: Any) -> int:
    match = re.search(r"[（(](\d+)次[）)]", str(value))
    return int(match.group(1)) if match else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_spot() -> pd.DataFrame:
    params = {
        "pz": "100",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
        "fields": "f2,f3,f5,f6,f12,f13,f14,f20,f21,f38,f124,f297",
    }

    def page(number: int) -> tuple[int, list[dict[str, Any]]]:
        response = _session().get(SPOT_URL, params={**params, "pn": str(number)}, timeout=20)
        response.raise_for_status()
        payload = response.json().get("data") or {}
        return int(payload.get("total") or 0), list(payload.get("diff") or [])

    total, first = page(1)
    page_count = math.ceil(total / 100)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        remaining = list(executor.map(page, range(2, page_count + 1)))
    rows = first + [row for _, batch in remaining for row in batch]
    out = pd.DataFrame(rows).rename(
        columns={
            "f12": "asset",
            "f14": "name",
            "f2": "price",
            "f5": "volume",
            "f6": "amount_cny",
            "f20": "market_cap_cny",
            "f21": "float_market_cap_cny",
            "f38": "latest_shares",
            "f124": "updated_epoch",
            "f297": "data_date",
        }
    )
    required = {"asset", "name", "price", "amount_cny", "market_cap_cny", "latest_shares", "data_date"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"ETF spot response missing fields: {missing}")
    out["asset"] = out["asset"].astype(str).str.zfill(6)
    for column in ["price", "volume", "amount_cny", "market_cap_cny", "float_market_cap_cny", "latest_shares"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["data_date"] = pd.to_datetime(out["data_date"].astype(str), format="%Y%m%d", errors="coerce")
    return out.drop_duplicates("asset", keep="last")


def build_watchlist(spot: pd.DataFrame, max_assets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = spot[
        spot["name"].astype(str).str.contains(ETF_KEYWORDS, na=False)
        & ~spot["name"].astype(str).str.contains(CROSS_BORDER, na=False)
    ].copy()
    candidates = candidates.sort_values(["market_cap_cny", "amount_cny", "asset"], ascending=[False, False, True])
    candidates["candidate_rank"] = range(1, len(candidates) + 1)
    candidates["selected_for_enrichment"] = candidates["candidate_rank"] <= max_assets
    candidates["current_universe_only"] = True
    candidates["historical_backtest_allowed"] = False
    return candidates[candidates["selected_for_enrichment"]].copy(), candidates


def fetch_profile(asset: str) -> pd.DataFrame:
    ak = _akshare()
    raw = _without_proxy(lambda: ak.fund_overview_em(symbol=asset))
    if raw.empty:
        return raw
    row = raw.iloc[0]
    return pd.DataFrame(
        [
            {
                "fund_name": row.get("基金简称"),
                "inception_raw": row.get("成立日期/规模"),
                "profile_aum_raw": row.get("净资产规模"),
                "management_fee_raw": row.get("管理费率"),
                "custody_fee_raw": row.get("托管费率"),
                "sales_service_fee_raw": row.get("销售服务费率"),
                "benchmark": row.get("业绩比较基准"),
                "tracking_index_name": row.get("跟踪标的"),
                "distributions_since_inception_raw": row.get("成立来分红"),
            }
        ]
    )


def fetch_nav(asset: str, as_of: pd.Timestamp) -> pd.DataFrame:
    start_date = (as_of - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    base_params = {
            "fundCode": asset,
            "pageSize": "20",
            "startDate": start_date,
            "endDate": as_of.strftime("%Y-%m-%d"),
    }
    headers = {"Referer": f"https://fundf10.eastmoney.com/jjjz_{asset}.html"}

    def page(number: int) -> tuple[int, list[dict[str, Any]]]:
        response = _session().get(
            NAV_URL,
            params={**base_params, "pageIndex": str(number)},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        rows = ((payload.get("Data") or {}).get("LSJZList") or [])
        return int(payload.get("TotalCount") or 0), list(rows)

    total, first = page(1)
    page_count = math.ceil(total / 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        remaining = list(executor.map(page, range(2, page_count + 1)))
    rows = first + [row for _, batch in remaining for row in batch]
    out = pd.DataFrame(rows).rename(
        columns={"FSRQ": "date", "DWJZ": "unit_nav", "LJJZ": "cumulative_nav", "JZZZL": "daily_growth_pct"}
    )
    required = {"date", "unit_nav", "cumulative_nav"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"ETF NAV response missing fields: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ["unit_nav", "cumulative_nav", "daily_growth_pct"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def fetch_dividends(asset: str) -> pd.DataFrame:
    ak = _akshare()
    market = "sh" if asset.startswith(("5", "6")) else "sz"
    raw = _without_proxy(lambda: ak.fund_etf_dividend_sina(symbol=market + asset))
    if raw.empty:
        return pd.DataFrame(columns=["date", "cumulative_dividend"])
    out = raw.rename(columns={"日期": "date", "累计分红": "cumulative_dividend"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["cumulative_dividend"] = pd.to_numeric(out["cumulative_dividend"], errors="coerce")
    return out[["date", "cumulative_dividend"]].dropna().sort_values("date").reset_index(drop=True)


def fetch_price(asset: str) -> pd.DataFrame:
    ak = _akshare()
    market = "sh" if asset.startswith(("5", "6")) else "sz"
    raw = _without_proxy(lambda: ak.fund_etf_hist_sina(symbol=market + asset))
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF price response missing fields: {missing}")
    columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    out = raw[columns].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in required.difference({"date"}):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def fetch_index_history(code: str, as_of: pd.Timestamp) -> pd.DataFrame:
    ak = _akshare()
    raw = _without_proxy(
        lambda: ak.stock_zh_index_hist_csindex(symbol=code, start_date="20000101", end_date=as_of.strftime("%Y%m%d"))
    )
    out = raw.rename(
        columns={
            "日期": "date",
            "指数代码": "index_code",
            "指数中文全称": "index_name",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交金额": "amount",
            "滚动市盈率": "pe_ttm",
        }
    ).copy()
    required = {"date", "index_code", "index_name", "close", "pe_ttm"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"index history response missing fields: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount", "pe_ttm"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def fetch_index_valuation(code: str) -> pd.DataFrame:
    ak = _akshare()
    raw = _without_proxy(lambda: ak.stock_zh_index_value_csindex(symbol=code))
    out = raw.rename(
        columns={
            "日期": "date",
            "指数代码": "index_code",
            "市盈率1": "pe_total_shares",
            "市盈率2": "pe_calculation_shares",
            "股息率1": "dividend_yield_total_shares_pct",
            "股息率2": "dividend_yield_calculation_shares_pct",
        }
    ).copy()
    required = {"date", "dividend_yield_calculation_shares_pct"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"index valuation response missing fields: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in [
        "pe_total_shares",
        "pe_calculation_shares",
        "dividend_yield_total_shares_pct",
        "dividend_yield_calculation_shares_pct",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def total_return_adjusted_prices(
    raw_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    asset: str,
    as_of: pd.Timestamp,
    share_conversion_factors: dict[pd.Timestamp, float] | None = None,
) -> pd.DataFrame:
    """Create a current-price-anchored total-return OHLC series."""
    prices = raw_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices[(prices["date"] <= as_of) & prices[numeric].notna().all(axis=1)].sort_values("date")
    prices = prices.drop_duplicates("date", keep="last").reset_index(drop=True)
    if prices.empty or (prices[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"invalid ETF price history for {asset}")
    prices["source_raw_close"] = prices["close"]
    prices["share_adjustment_factor"] = 1.0
    applied_share_actions: list[dict[str, Any]] = []
    governed_factors = {
        pd.Timestamp(date).normalize(): float(factor)
        for date, factor in (share_conversion_factors or {}).items()
    }
    if any(not np.isfinite(factor) or factor <= 0 for factor in governed_factors.values()):
        raise ValueError(f"invalid governed ETF share-conversion factor for {asset}")

    events = dividends.copy()
    if events.empty:
        events = pd.DataFrame(columns=["date", "cumulative_dividend"])
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["cumulative_dividend"] = pd.to_numeric(events["cumulative_dividend"], errors="coerce")
    events = events[(events["date"] <= as_of) & events["cumulative_dividend"].notna()].sort_values("date")
    events["cash_distribution"] = events["cumulative_dividend"].diff().fillna(events["cumulative_dividend"])
    if (events["cash_distribution"] < -1e-10).any():
        raise ValueError(f"cumulative ETF dividends decreased for {asset}")

    prices["source_cash_distribution"] = 0.0
    price_dates = pd.Index(prices["date"])
    distribution_alignments: list[dict[str, Any]] = []
    for _, event in events[events["cash_distribution"] > 0].iterrows():
        position = int(price_dates.searchsorted(event["date"], side="left"))
        alignment_lag = (
            int((prices.loc[position, "date"] - event["date"]).days)
            if position < len(prices)
            else None
        )
        if (
            position == 0
            or position >= len(prices)
            or alignment_lag is None
            or alignment_lag > MAX_DISTRIBUTION_ALIGNMENT_CALENDAR_DAYS
        ):
            raise ValueError(f"ETF distribution cannot be aligned to a trading date: {asset} {event['date']}")
        prices.loc[position, "source_cash_distribution"] += float(event["cash_distribution"])
        distribution_alignments.append(
            {
                "event_date": pd.Timestamp(event["date"]).date().isoformat(),
                "aligned_price_date": pd.Timestamp(prices.loc[position, "date"]).date().isoformat(),
                "calendar_lag_days": alignment_lag,
                "cash_distribution": float(event["cash_distribution"]),
            }
        )

    # Sina uses a zero-increment distribution row for some ETF share
    # conversions. A large raw price drop can also be a cash ex-date, so cash
    # explanations must be tested before conversion detection.
    jump_threshold = 0.21
    factor_match_tolerance = 0.12
    common_factors = np.array([0.1, 0.2, 0.25, 0.5, 0.8, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0])

    def apply_share_action(position: int, factor: float, evidence_basis: str, observed_ratio: float) -> None:
        event_date = pd.Timestamp(prices.loc[position, "date"])
        prices.loc[: position - 1, ["open", "high", "low", "close"]] /= factor
        prices.loc[: position - 1, "share_adjustment_factor"] /= factor
        applied_share_actions.append(
            {
                "price_effective_date": event_date.date().isoformat(),
                "shares_after_per_share_before": float(factor),
                "observed_price_ratio": float(observed_ratio),
                "evidence_basis": evidence_basis,
            }
        )

    # Apply registered actions even when the raw jump is below the generic
    # anomaly threshold. Otherwise a moderate split can silently survive as a
    # false market loss.
    for event_date, factor in sorted(governed_factors.items()):
        matches = prices.index[prices["date"].eq(event_date)].tolist()
        if len(matches) != 1 or matches[0] == 0:
            raise ValueError(f"governed ETF share-conversion date is not actionable: {asset} {event_date.date()}")
        position = int(matches[0])
        ratio = float(prices.loc[position - 1, "close"] / prices.loc[position, "close"])
        if abs(ratio / factor - 1.0) > factor_match_tolerance:
            raise ValueError(
                f"unresolved ETF corporate-action jump: {asset} {event_date.date()} "
                f"ratio={ratio:.6f} factor={factor:.6f}"
            )
        apply_share_action(position, factor, "governed_registry", ratio)

    cash_explained_positions: set[int] = set()
    for _ in range(10):
        jumps = prices["close"].pct_change().abs()
        large = jumps[(jumps > jump_threshold) & ~jumps.index.to_series().isin(cash_explained_positions)]
        if large.empty:
            break
        position = int(large.index[0])
        event_date = pd.Timestamp(prices.loc[position, "date"])
        source_cash = float(prices.loc[position, "source_cash_distribution"])
        cash_total_return = (float(prices.loc[position, "close"]) + source_cash) / float(
            prices.loc[position - 1, "close"]
        ) - 1.0
        if source_cash > 0 and abs(cash_total_return) <= jump_threshold:
            cash_explained_positions.add(position)
            continue
        ratio = float(prices.loc[position - 1, "close"] / prices.loc[position, "close"])
        zero_increment_markers = events[
            events["cash_distribution"].abs().le(1e-12)
            & events["date"].between(event_date - pd.Timedelta(days=7), event_date + pd.Timedelta(days=7))
        ]
        factor = float(common_factors[np.argmin(np.abs(common_factors - ratio))])
        evidence_available = not zero_increment_markers.empty
        if not evidence_available or abs(ratio / factor - 1.0) > factor_match_tolerance:
            raise ValueError(f"unresolved ETF corporate-action jump: {asset} {event_date.date()} return={large.iloc[0]:.4f}")
        apply_share_action(position, factor, "zero_marker_common_factor_inference", ratio)
    else:
        raise ValueError(f"too many ETF corporate-action jumps: {asset}")

    # Cash paid before a share conversion must be restated onto the same share
    # basis as the adjusted historical prices.
    prices["cash_distribution"] = prices["source_cash_distribution"] * prices["share_adjustment_factor"]

    previous_close = prices["close"].shift(1)
    total_return = (prices["close"] + prices["cash_distribution"]) / previous_close
    total_return.iloc[0] = 1.0
    if (total_return <= 0).any() or not np.isfinite(total_return).all():
        raise ValueError(f"invalid ETF total-return path for {asset}")
    if ((total_return - 1.0).abs() > jump_threshold).any():
        raise ValueError(f"unresolved ETF total-return jump for {asset}")
    wealth = total_return.cumprod()
    adjusted_close = float(prices["close"].iloc[-1]) * wealth / float(wealth.iloc[-1])
    scale = adjusted_close / prices["close"]
    out = pd.DataFrame({"date": prices["date"]})
    for column in ["open", "high", "low", "close"]:
        out[column] = prices[column] * scale
    out["volume"] = prices["volume"]
    out["amount"] = prices["amount"]
    out["raw_close"] = prices["source_raw_close"]
    out["source_cash_distribution"] = prices["source_cash_distribution"]
    out["cash_distribution"] = prices["cash_distribution"]
    out["share_adjustment_factor"] = prices["share_adjustment_factor"]
    out["adjustment_factor"] = adjusted_close / prices["source_raw_close"]
    out["asset"] = asset
    out["asset_type"] = "etf"
    out["return_basis"] = "total_return"
    out["available_date"] = out["date"]
    out["data_source"] = "sina ETF OHLC + sina cumulative distributions"
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.attrs["applied_share_actions"] = applied_share_actions
    out.attrs["distribution_alignments"] = distribution_alignments
    return out


def _tracking_error(nav: pd.DataFrame, index_total_return: pd.DataFrame, as_of: pd.Timestamp) -> float:
    required = {"date", "daily_growth_pct"}
    if not required.issubset(nav.columns):
        return float("nan")
    fund = nav[["date", "daily_growth_pct"]].copy()
    benchmark = index_total_return[["date", "close"]].copy()
    fund["date"] = pd.to_datetime(fund["date"], errors="coerce")
    fund["fund_return"] = pd.to_numeric(fund["daily_growth_pct"], errors="coerce") / 100.0
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    merged = fund.merge(benchmark, on="date", how="inner").sort_values("date")
    merged = merged[merged["date"] <= as_of].tail(253)
    differences = merged["fund_return"] - merged["close"].pct_change()
    differences = differences.replace([np.inf, -np.inf], np.nan).dropna()
    if len(differences) < 200:
        return float("nan")
    return float(differences.std(ddof=1) * math.sqrt(252.0))


def _index_metrics(index_price: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    data = index_price.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["pe_ttm"] = pd.to_numeric(data["pe_ttm"], errors="coerce")
    data = data[(data["date"] <= as_of) & (data["close"] > 0)].sort_values("date")
    valid_pe = data[data["pe_ttm"] > 0].copy()
    if data.empty or valid_pe.empty:
        raise ValueError("index history has no valid close/PE data")
    latest = valid_pe.iloc[-1]
    five_year = valid_pe[valid_pe["date"] >= pd.Timestamp(latest["date"]) - pd.DateOffset(years=5)]
    if len(five_year) < 1000:
        raise ValueError("index PE history is shorter than five years")
    percentile = float((five_year["pe_ttm"] <= float(latest["pe_ttm"])).mean())
    earnings_proxy = five_year["close"] / five_year["pe_ttm"]
    start = float(earnings_proxy.head(63).median())
    end = float(earnings_proxy.tail(63).median())
    span = (five_year["date"].iloc[-1] - five_year["date"].iloc[0]).days / 365.25
    earnings_cagr = (end / start) ** (1.0 / span) - 1.0 if start > 0 and end > 0 and span >= 4.5 else float("nan")
    actual_history = data[data["open"].notna()] if "open" in data.columns else data
    if actual_history.empty:
        actual_history = data
    return {
        "current_pe": float(latest["pe_ttm"]),
        "pe_percentile_5y": percentile,
        "index_earnings_cagr_5y": float(earnings_cagr),
        "index_available_date": pd.Timestamp(data["date"].iloc[-1]),
        "index_history_years": float(
            (actual_history["date"].iloc[-1] - actual_history["date"].iloc[0]).days / 365.25
        ),
    }


def build_etf_row(
    spot: pd.Series,
    profile: pd.Series,
    nav: pd.DataFrame,
    dividends: pd.DataFrame,
    adjusted_prices: pd.DataFrame,
    index_price: pd.DataFrame,
    index_total_return: pd.DataFrame,
    index_valuation: pd.DataFrame,
    china_10y: float,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    asset = str(spot["asset"]).zfill(6)
    price = adjusted_prices.copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price[price["date"] <= as_of].sort_values("date")
    nav = nav[(pd.to_datetime(nav["date"], errors="coerce") <= as_of) & nav["unit_nav"].notna()].sort_values("date")
    valuation = index_valuation[
        (pd.to_datetime(index_valuation["date"], errors="coerce") <= as_of)
        & index_valuation["dividend_yield_calculation_shares_pct"].notna()
    ].sort_values("date")
    if price.empty or nav.empty or valuation.empty:
        raise ValueError(f"ETF source has no row on or before as_of: {asset}")

    profile_distribution_count = _distribution_count(profile.get("distributions_since_inception_raw"))
    if profile_distribution_count > 0 and dividends.empty:
        raise ValueError(f"ETF dividend endpoint is empty despite profile history: {asset}")
    events = dividends.copy()
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
        events["cumulative_dividend"] = pd.to_numeric(events["cumulative_dividend"], errors="coerce")
        events = events[(events["date"] <= as_of) & events["cumulative_dividend"].notna()].sort_values("date")
        events["cash_distribution"] = events["cumulative_dividend"].diff().fillna(events["cumulative_dividend"])
    five_year_start = as_of - pd.DateOffset(years=5)
    distribution_years = int(events.loc[(events["date"] >= five_year_start) & (events["cash_distribution"] > 0), "date"].dt.year.nunique()) if not events.empty else 0

    recent = price.tail(756)
    returns = recent["close"].pct_change()
    drawdown = recent["close"] / recent["close"].cummax() - 1.0
    latest_price_date = pd.Timestamp(price["date"].iloc[-1])
    latest_nav_date = pd.Timestamp(nav["date"].iloc[-1])
    latest_valuation = valuation.iloc[-1]
    valuation_date = pd.Timestamp(latest_valuation["date"])
    index = _index_metrics(index_price, as_of)
    total_index = index_total_return.copy()
    total_index["date"] = pd.to_datetime(total_index["date"], errors="coerce")
    total_index = total_index[(total_index["date"] <= as_of) & total_index["close"].notna()].sort_values("date")
    if total_index.empty:
        raise ValueError(f"ETF index total-return history is empty: {asset}")

    management_fee = _percent(profile.get("management_fee_raw"))
    custody_fee = _percent(profile.get("custody_fee_raw"))
    sales_fee = _percent(profile.get("sales_service_fee_raw"))
    if not math.isfinite(management_fee) or not math.isfinite(custody_fee):
        raise ValueError(f"ETF operating fees cannot be parsed: {asset}")
    expense_ratio = management_fee + custody_fee + (sales_fee if math.isfinite(sales_fee) else 0.0)

    spot_shares = _number(spot.get("latest_shares"))
    unit_nav = float(nav["unit_nav"].iloc[-1])
    profile_aum = _cn_amount(profile.get("profile_aum_raw"))
    aum = spot_shares * unit_nav if math.isfinite(spot_shares) and spot_shares > 0 else profile_aum
    if not math.isfinite(aum) or aum <= 0:
        raise ValueError(f"ETF AUM cannot be established: {asset}")

    history_years = float((price["date"].iloc[-1] - price["date"].iloc[0]).days / 365.25)
    observed_distribution_count = int((events["cash_distribution"] > 0).sum()) if not events.empty else 0
    dividends_complete = profile_distribution_count == observed_distribution_count
    total_return_ready = bool(history_years >= 5.0 and dividends_complete)
    index_dividend_yield = float(latest_valuation["dividend_yield_calculation_shares_pct"] / 100.0)
    inception_date = _cn_date(profile.get("inception_raw"))
    spot_date = pd.Timestamp(spot["data_date"])
    latest_amount = float(pd.to_numeric(price["amount"], errors="coerce").tail(60).mean())
    available_dates = [as_of, latest_price_date, latest_nav_date, valuation_date, index["index_available_date"]]

    return {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "available_date": max(available_dates).strftime("%Y-%m-%d"),
        "asset": asset,
        "name": str(spot["name"]),
        "asset_type": "etf",
        "sector": "dividend_index",
        "is_tradeable": bool((as_of - latest_price_date).days <= 5 and latest_amount > 0),
        "is_st": False,
        "history_years": history_years,
        "aum_cny": float(aum),
        "avg_daily_amount_cny": latest_amount,
        "expense_ratio": float(expense_ratio),
        "tracking_error_1y": _tracking_error(nav, total_index, as_of),
        "index_history_years": index["index_history_years"],
        "distribution_years_5y": distribution_years,
        "index_dividend_yield": index_dividend_yield,
        "index_earnings_cagr_5y": index["index_earnings_cagr_5y"],
        "current_pe": index["current_pe"],
        "pe_percentile_5y": index["pe_percentile_5y"],
        "china_10y_yield": china_10y,
        "yield_spread_cn10y": index_dividend_yield - china_10y,
        "annual_vol_3y": float(returns.std(ddof=1) * math.sqrt(252.0)),
        "max_drawdown_3y": float(drawdown.min()),
        "total_return_history_ready": total_return_ready,
        "price_available_date": latest_price_date.strftime("%Y-%m-%d"),
        "nav_available_date": latest_nav_date.strftime("%Y-%m-%d"),
        "valuation_available_date": valuation_date.strftime("%Y-%m-%d"),
        "aum_available_date": max(spot_date, latest_nav_date).strftime("%Y-%m-%d"),
        "distribution_available_date": as_of.strftime("%Y-%m-%d"),
        "expense_available_date": as_of.strftime("%Y-%m-%d"),
        "index_available_date": pd.Timestamp(index["index_available_date"]).strftime("%Y-%m-%d"),
        "total_return_available_date": latest_price_date.strftime("%Y-%m-%d"),
        "tracking_index_name": str(profile.get("tracking_index_name")),
        "tracking_index_price_code": INDEX_MAP[str(profile.get("tracking_index_name"))]["price_code"],
        "tracking_index_total_return_code": INDEX_MAP[str(profile.get("tracking_index_name"))]["total_return_code"],
        "tracking_error_basis": "fund reported daily total return vs official index total return",
        "earnings_proxy_method": "63-session median of index close/PE, five-year CAGR",
        "inception_date": inception_date.strftime("%Y-%m-%d") if pd.notna(inception_date) else "",
        "profile_distribution_count": profile_distribution_count,
        "observed_distribution_count": observed_distribution_count,
        "current_universe_only": True,
        "historical_backtest_allowed": False,
        "source_note": "Eastmoney spot/profile/NAV + Sina ETF OHLC/dividends + CSIndex price/TRI/valuation",
        "source_errors": "",
    }


def _asset_data(asset: str, as_of: pd.Timestamp, raw: Path, refresh: bool) -> dict[str, pd.DataFrame]:
    return {
        "profile": _fetch_cached(raw / "profile" / f"{asset}.csv", lambda: fetch_profile(asset), "akshare.fund_overview_em", refresh),
        "nav": _fetch_cached(raw / "nav" / f"{asset}.csv", lambda: fetch_nav(asset, as_of), "eastmoney.f10.lsjz", refresh),
        "dividend": _fetch_cached(
            raw / "dividend" / f"{asset}.csv",
            lambda: fetch_dividends(asset),
            "akshare.fund_etf_dividend_sina",
            refresh,
            allow_empty=True,
        ),
        "price": _fetch_cached(raw / "price" / f"{asset}.csv", lambda: fetch_price(asset), "akshare.fund_etf_hist_sina", refresh),
    }


def _index_data(mapping: dict[str, str], as_of: pd.Timestamp, raw: Path, refresh: bool) -> dict[str, pd.DataFrame]:
    price_code = mapping["price_code"]
    total_code = mapping["total_return_code"]
    return {
        "price": _fetch_cached(
            raw / "index_price" / f"{price_code}.csv",
            lambda: fetch_index_history(price_code, as_of),
            "akshare.stock_zh_index_hist_csindex.price",
            refresh,
        ),
        "total_return": _fetch_cached(
            raw / "index_total_return" / f"{total_code}.csv",
            lambda: fetch_index_history(total_code, as_of),
            "akshare.stock_zh_index_hist_csindex.total_return",
            refresh,
        ),
        "valuation": _fetch_cached(
            raw / "index_valuation" / f"{price_code}.csv",
            lambda: fetch_index_valuation(price_code),
            "akshare.stock_zh_index_value_csindex",
            refresh,
        ),
    }


def run_builder(as_of: pd.Timestamp, max_assets: int, refresh: bool, sleep_seconds: float) -> dict[str, Any]:
    if max_assets <= 0:
        raise ValueError("max_assets must be positive")
    raw = RAW_ROOT / "etf_raw" / as_of.strftime("%Y%m%d")
    spot = _fetch_cached(raw / "spot.csv", fetch_spot, "eastmoney.etf_spot", refresh)
    selected, catalog = build_watchlist(spot, max_assets)
    _write_csv(catalog, WATCHLIST_PATH)
    yield_data = fetch_china_10y(as_of, refresh=refresh)
    china_10y, china_10y_date = china_10y_latest(yield_data, as_of)

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    index_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for position, (_, candidate) in enumerate(selected.iterrows(), start=1):
        asset = str(candidate["asset"]).zfill(6)
        started = time.time()
        status = "error"
        error = ""
        tracking_index = ""
        asset_history_collected = False
        try:
            profile_frame = _fetch_cached(
                raw / "profile" / f"{asset}.csv", lambda: fetch_profile(asset), "akshare.fund_overview_em", refresh
            )
            profile = profile_frame.iloc[-1]
            tracking_index = str(profile.get("tracking_index_name") or "").strip()
            asset_data = _asset_data(asset, as_of, raw, refresh)
            asset_history_collected = True
            mapping = INDEX_MAP.get(tracking_index)
            if mapping is None:
                status = "data_gap"
                error = f"unsupported_tracking_index={tracking_index or 'missing'}"
            else:
                cache_key = mapping["price_code"]
                if cache_key not in index_cache:
                    index_cache[cache_key] = _index_data(mapping, as_of, raw, refresh)
                index_data = index_cache[cache_key]
                adjusted = total_return_adjusted_prices(asset_data["price"], asset_data["dividend"], asset, as_of)
                row = build_etf_row(
                    candidate,
                    profile,
                    asset_data["nav"],
                    asset_data["dividend"],
                    adjusted,
                    index_data["price"],
                    index_data["total_return"],
                    index_data["valuation"],
                    china_10y,
                    as_of,
                )
                _write_csv(adjusted, RAW_ROOT / "prices" / f"{asset}.csv")
                rows.append(row)
                status = "ok"
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        manifest.append(
            {
                "asset": asset,
                "name": candidate["name"],
                "tracking_index_name": tracking_index,
                "status": status,
                "error": error,
                "asset_history_collected": asset_history_collected,
                "seconds": round(time.time() - started, 3),
                "completed": position,
                "total": len(selected),
                "fetched_at": _now(),
            }
        )
        print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)
        if sleep_seconds > 0 and position < len(selected):
            time.sleep(sleep_seconds)

    snapshot = pd.DataFrame(rows).sort_values("asset").reset_index(drop=True) if rows else pd.DataFrame()
    combined = write_snapshot_part("etf", snapshot)
    manifest_frame = pd.DataFrame(manifest)
    manifest_path = RAW_ROOT / "manifests" / f"etf_snapshot_{as_of.strftime('%Y%m%d')}.csv"
    _write_csv(manifest_frame, manifest_path)
    normalized_prices = [RAW_ROOT / "prices" / f"{row['asset']}.csv" for row in rows]
    source_paths = sorted(
        set(raw.rglob("*.csv"))
        | {WATCHLIST_PATH, ETF_SNAPSHOT_PATH, INDEX_REGISTRY_PATH, manifest_path, *normalized_prices}
    )
    source_manifest_path = RAW_ROOT / "manifests" / f"etf_source_manifest_{as_of.strftime('%Y%m%d')}.json"
    source_manifest = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "generated_at": _now(),
        "input_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "code_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in sorted(
                [
                    Path(__file__),
                    Path(__file__).with_name("etf_index_registry.py"),
                    Path(__file__).with_name("snapshot_store.py"),
                    Path(__file__).with_name("stock_snapshot_builder.py"),
                ]
            )
        ],
        "current_universe_only": True,
        "historical_backtest_allowed": False,
    }
    source_manifest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    source_manifest_latest_path = RAW_ROOT / "manifests" / "etf_source_manifest_latest.json"
    source_manifest_latest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status_counts = manifest_frame["status"].value_counts().to_dict() if not manifest_frame.empty else {}
    result = {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "keyword_candidates": len(catalog),
        "selected_for_enrichment": len(selected),
        "snapshot_rows": len(snapshot),
        "combined_snapshot_rows": len(combined),
        "status_counts": status_counts,
        "china_10y": china_10y,
        "china_10y_date": str(china_10y_date.date()),
        "snapshot_path": str(ETF_SNAPSHOT_PATH),
        "combined_snapshot_path": str(RAW_ROOT / "research_snapshot.csv"),
        "watchlist_path": str(WATCHLIST_PATH),
        "manifest_path": str(manifest_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_latest_path": str(source_manifest_latest_path),
        "source_manifest_input_count": len(source_manifest["input_files"]),
        "active_index_mapping_count": len(INDEX_MAP),
        "current_universe_only": True,
        "historical_backtest_allowed": False,
    }
    (RAW_ROOT / "etf_snapshot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--max-assets", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()
    result = run_builder(pd.Timestamp(args.as_of).normalize(), args.max_assets, args.refresh, args.sleep_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
