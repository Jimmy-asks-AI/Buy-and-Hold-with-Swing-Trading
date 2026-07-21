"""Collect a limited JoinQuant ETF price/NAV panel for independent validation.

The trial account exposes only a recent window.  These files can challenge the
Sina/Eastmoney observations, but they are not a full-history source and can
never be promoted directly into the backtest table.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .pit_etf_total_return_collector import load_lifecycles


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
CREDENTIALS_PATH = ROOT / "configs" / "data_credentials.json"
DEFAULT_NAV_DIR = ROOT / "data_raw" / "long_hold_v4" / "etf_raw" / "20260717" / "nav"
OUTPUT_ROOT = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "validation_sources" / "joinquant_etf_price_nav"
)
LATEST_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "joinquant_etf_price_nav_validation_latest.json"
)
DEFAULT_QUERY_START = "2025-05-01"
DEFAULT_QUERY_END = "2026-04-01"
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "money", "paused")
PRICE_COLUMNS = [
    "date",
    "asset",
    "jq_code",
    "exchange",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "paused",
    "price_adjustment",
    "query_start",
    "query_end",
    "source_observed_at",
    "available_date",
    "pit_actionable",
    "data_source",
    "source_vintage",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
NAV_COLUMNS = [
    "date",
    "asset",
    "jq_code",
    "exchange",
    "unit_nav",
    "cumulative_nav",
    "query_start",
    "query_end",
    "source_observed_at",
    "available_date",
    "pit_actionable",
    "data_source",
    "source_vintage",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if gzip else None,
    )
    temporary.replace(path)


def _load_credentials() -> tuple[str, str]:
    payload: dict[str, Any] = {}
    if CREDENTIALS_PATH.is_file():
        payload = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8-sig"))
    section = payload.get("joinquant", {})
    username = str(os.getenv("JQDATA_USERNAME") or section.get("username", "")).strip()
    password = str(os.getenv("JQDATA_PASSWORD") or section.get("password", "")).strip()
    if not username or not password:
        raise ValueError("JoinQuant credentials are unavailable")
    return username, password


def _authenticate_master() -> dict[str, str]:
    manifest = json.loads(MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("historical_backtest_allowed") is not True
        or manifest.get("model_promotion_allowed") is not False
        or Path(str(manifest.get("output_path", ""))).as_posix() != _relative(MASTER_PATH)
        or _sha256(MASTER_PATH) != str(manifest.get("output_sha256", ""))
    ):
        raise ValueError("ETF security master failed source authentication")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF security-master builder hash mismatch")
    return {
        "master_sha256": _sha256(MASTER_PATH),
        "manifest_sha256": _sha256(MASTER_MANIFEST_PATH),
    }


def select_window_assets(
    lifecycles: pd.DataFrame,
    query_start: str | pd.Timestamp,
    query_end: str | pd.Timestamp,
    explicit_assets: Iterable[str] | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(query_start).normalize()
    end = pd.Timestamp(query_end).normalize()
    if start > end:
        raise ValueError("JoinQuant ETF validation query start follows query end")
    selected = lifecycles[
        lifecycles["list_date"].le(end)
        & (lifecycles["delist_date"].isna() | lifecycles["delist_date"].ge(start))
    ].copy()
    if explicit_assets is not None:
        requested = {str(value).strip().replace(".0", "").zfill(6) for value in explicit_assets}
        selected = selected[selected["asset"].isin(requested)].copy()
        missing = sorted(requested.difference(selected["asset"]))
        if missing:
            raise ValueError(f"ETF validation assets are outside the governed query window: {missing}")
    selected["jq_code"] = selected["asset"] + selected["exchange"].map({"SSE": ".XSHG", "SZSE": ".XSHE"})
    if selected["jq_code"].isna().any() or selected["asset"].duplicated().any():
        raise ValueError("ETF validation selection has invalid exchange or duplicate assets")
    return selected.sort_values("asset").reset_index(drop=True)


def normalise_joinquant_prices(
    raw: pd.DataFrame,
    requested_codes: list[str],
    query_start: str | pd.Timestamp,
    query_end: str | pd.Timestamp,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "asset", "jq_code", "exchange", *PRICE_FIELDS[:-1], "amount", "paused"])
    data = raw.copy()
    if "time" not in data.columns:
        data = data.rename_axis("time").reset_index()
    if "code" not in data.columns:
        if len(requested_codes) != 1:
            raise ValueError("JoinQuant ETF price response omitted code for a multi-asset query")
        data["code"] = requested_codes[0]
    missing = sorted({"time", "code", *PRICE_FIELDS}.difference(data.columns))
    if missing:
        raise ValueError(f"JoinQuant ETF price response is incomplete: {missing}")
    data = data.rename(columns={"time": "date", "code": "jq_code", "money": "amount"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["jq_code"] = data["jq_code"].astype(str).str.strip()
    data["asset"] = data["jq_code"].str[:6]
    data["exchange"] = data["jq_code"].str[-4:].map({"XSHG": "SSE", "XSHE": "SZSE"})
    numeric = ["open", "high", "low", "close", "volume", "amount", "paused"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data[~data[["open", "high", "low", "close", "volume", "amount", "paused"]].isna().all(axis=1)].copy()
    requested = set(requested_codes)
    if data["date"].isna().any() or not set(data["jq_code"]).issubset(requested):
        raise ValueError("JoinQuant ETF price response contains invalid dates or unrequested codes")
    start = pd.Timestamp(query_start).normalize()
    end = pd.Timestamp(query_end).normalize()
    if data["date"].lt(start).any() or data["date"].gt(end).any():
        raise ValueError("JoinQuant ETF price response escaped the requested date window")
    traded = data["paused"].fillna(0).eq(0)
    if data.loc[traded, ["open", "high", "low", "close", "volume", "amount"]].isna().any(axis=None):
        raise ValueError("JoinQuant ETF traded rows contain missing market fields")
    if (data.loc[traded, ["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("JoinQuant ETF traded rows contain non-positive prices")
    if (data.loc[traded, ["volume", "amount"]] < 0).any(axis=None):
        raise ValueError("JoinQuant ETF traded rows contain negative activity")
    if data.duplicated(["date", "asset"]).any():
        raise ValueError("JoinQuant ETF price response contains duplicate asset/date rows")
    return data[["date", "asset", "jq_code", "exchange", "open", "high", "low", "close", "volume", "amount", "paused"]].sort_values(
        ["asset", "date"]
    ).reset_index(drop=True)


def normalise_joinquant_nav(
    unit_raw: pd.DataFrame,
    cumulative_raw: pd.DataFrame,
    code_to_exchange: dict[str, str],
) -> pd.DataFrame:
    def melt(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["date", "jq_code", value_name])
        data = frame.copy().rename_axis("date").reset_index()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        return data.melt(id_vars="date", var_name="jq_code", value_name=value_name)

    unit = melt(unit_raw, "unit_nav")
    cumulative = melt(cumulative_raw, "cumulative_nav")
    data = unit.merge(cumulative, on=["date", "jq_code"], how="outer", validate="one_to_one")
    if data.empty:
        return pd.DataFrame(columns=["date", "asset", "jq_code", "exchange", "unit_nav", "cumulative_nav"])
    data[["unit_nav", "cumulative_nav"]] = data[["unit_nav", "cumulative_nav"]].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data[data[["unit_nav", "cumulative_nav"]].notna().any(axis=1)].copy()
    data["jq_code"] = data["jq_code"].astype(str)
    data["asset"] = data["jq_code"].str[:6]
    data["exchange"] = data["jq_code"].map(code_to_exchange)
    if data[["date", "asset", "exchange"]].isna().any(axis=None):
        raise ValueError("JoinQuant ETF NAV response contains invalid keys")
    present = data[["unit_nav", "cumulative_nav"]].notna()
    if ((data[["unit_nav", "cumulative_nav"]] <= 0) & present).any(axis=None):
        raise ValueError("JoinQuant ETF NAV response contains non-positive values")
    if data.duplicated(["date", "asset"]).any():
        raise ValueError("JoinQuant ETF NAV response contains duplicate asset/date rows")
    return data[["date", "asset", "jq_code", "exchange", "unit_nav", "cumulative_nav"]].sort_values(
        ["asset", "date"]
    ).reset_index(drop=True)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def collect(
    *,
    as_of: str | pd.Timestamp,
    query_start: str | pd.Timestamp = DEFAULT_QUERY_START,
    query_end: str | pd.Timestamp = DEFAULT_QUERY_END,
    nav_dir: Path = DEFAULT_NAV_DIR,
    batch_size: int = 100,
    explicit_assets: Iterable[str] | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    if batch_size < 2 or batch_size > 200:
        raise ValueError("JoinQuant ETF validation batch size must be between 2 and 200")
    cutoff = pd.Timestamp(as_of).normalize()
    start = pd.Timestamp(query_start).normalize()
    end = pd.Timestamp(query_end).normalize()
    if end > cutoff:
        raise ValueError("JoinQuant ETF validation query end follows as-of")
    master_auth = _authenticate_master()
    lifecycles = load_lifecycles(MASTER_PATH, cutoff)
    selected = select_window_assets(lifecycles, start, end, explicit_assets)
    if selected.empty:
        raise ValueError("JoinQuant ETF validation selection is empty")

    nav_files = sorted(nav_dir.glob("*.csv")) if nav_dir.is_dir() else []
    nav_assets = {path.stem.zfill(6) for path in nav_files}
    nav_selected = selected[selected["asset"].isin(nav_assets)].copy()
    username, password = _load_credentials()
    import jqdatasdk as jq  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jq.auth(username, password)
    query_before = jq.get_query_count()
    fetched_at = datetime.now().astimezone()
    price_frames: list[pd.DataFrame] = []
    codes = selected["jq_code"].tolist()
    for batch in _chunks(codes, batch_size):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = jq.get_price(
                batch,
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
                frequency="daily",
                fields=list(PRICE_FIELDS),
                skip_paused=False,
                fq=None,
                panel=False,
                fill_paused=False,
                round=False,
            )
        price_frames.append(normalise_joinquant_prices(raw, batch, start, end))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    if prices.empty or prices.duplicated(["date", "asset"]).any():
        raise ValueError("JoinQuant ETF validation returned an empty or duplicate price panel")

    nav = pd.DataFrame(columns=["date", "asset", "jq_code", "exchange", "unit_nav", "cumulative_nav"])
    if not nav_selected.empty:
        nav_codes = nav_selected["jq_code"].tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unit_raw = jq.get_extras(
                "unit_net_value",
                nav_codes,
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
                df=True,
            )
            cumulative_raw = jq.get_extras(
                "acc_net_value",
                nav_codes,
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
                df=True,
            )
        nav = normalise_joinquant_nav(
            unit_raw,
            cumulative_raw,
            dict(zip(nav_selected["jq_code"], nav_selected["exchange"])),
        )
    query_after = jq.get_query_count()

    asset_set_sha256 = hashlib.sha256("\n".join(codes).encode("ascii")).hexdigest()
    source_material = {
        "asset_set_sha256": asset_set_sha256,
        "query_start": start.date().isoformat(),
        "query_end": end.date().isoformat(),
        "frequency": "daily",
        "fields": list(PRICE_FIELDS),
        "fq": None,
        "skip_paused": False,
        "fill_paused": False,
        "round": False,
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
    }
    source_vintage = "joinquant_etf_validation_query_sha256:" + hashlib.sha256(
        json.dumps(source_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    available = fetched_at.replace(tzinfo=None).date().isoformat()
    prices = prices.assign(
        price_adjustment="none",
        query_start=start,
        query_end=end,
        source_observed_at=fetched_at.isoformat(timespec="seconds"),
        available_date=available,
        pit_actionable=False,
        data_source="jqdatasdk.get_price(fq=None); limited trial window",
        source_vintage=source_vintage,
        historical_backtest_allowed=False,
        model_promotion_allowed=False,
    )[PRICE_COLUMNS]
    if not nav.empty:
        nav = nav.assign(
            query_start=start,
            query_end=end,
            source_observed_at=fetched_at.isoformat(timespec="seconds"),
            available_date=available,
            pit_actionable=False,
            data_source="jqdatasdk.get_extras(unit_net_value,acc_net_value); limited trial window",
            source_vintage=source_vintage,
            historical_backtest_allowed=False,
            model_promotion_allowed=False,
        )[NAV_COLUMNS]
    else:
        nav = pd.DataFrame(columns=NAV_COLUMNS)

    status = selected[["asset", "asset_name", "exchange", "list_date", "delist_date", "lifecycle_status"]].copy()
    price_coverage = prices.groupby("asset").agg(
        price_rows=("date", "size"),
        price_start=("date", "min"),
        price_end=("date", "max"),
        paused_rows=("paused", lambda values: int(pd.to_numeric(values, errors="coerce").fillna(0).eq(1).sum())),
    )
    nav_coverage = nav.groupby("asset").agg(
        nav_rows=("date", "size"),
        nav_start=("date", "min"),
        nav_end=("date", "max"),
    )
    status = status.merge(price_coverage, left_on="asset", right_index=True, how="left")
    status = status.merge(nav_coverage, left_on="asset", right_index=True, how="left")
    status["nav_requested"] = status["asset"].isin(nav_selected["asset"])
    status["price_rows"] = status["price_rows"].fillna(0).astype(int)
    status["paused_rows"] = status["paused_rows"].fillna(0).astype(int)
    status["nav_rows"] = status["nav_rows"].fillna(0).astype(int)
    status["collection_status"] = np.where(status["price_rows"].gt(0), "collected", "no_rows_in_trial_window")
    status["historical_backtest_allowed"] = False
    status["model_promotion_allowed"] = False

    run_id = f"{cutoff.strftime('%Y%m%d')}_{fetched_at.strftime('%Y%m%dT%H%M%S%f%z')}_{asset_set_sha256[:12]}"
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    price_path = run_dir / "prices.csv.gz"
    nav_path = run_dir / "nav.csv.gz"
    status_path = run_dir / "asset_status.csv"
    manifest_path = run_dir / "run_manifest.json"
    _atomic_csv(prices, price_path, gzip=True)
    _atomic_csv(nav, nav_path, gzip=True)
    _atomic_csv(status, status_path)
    inputs = [
        {"role": "etf_security_master", "path": _relative(MASTER_PATH), "sha256": master_auth["master_sha256"]},
        {
            "role": "etf_security_master_manifest",
            "path": _relative(MASTER_MANIFEST_PATH),
            "sha256": master_auth["manifest_sha256"],
        },
        *[
            {"role": "eastmoney_nav_asset_selector", "asset": path.stem.zfill(6), "path": _relative(path), "sha256": _sha256(path)}
            for path in nav_files
        ],
    ]
    outputs = [
        {"role": "joinquant_etf_raw_prices", "path": _relative(price_path), "sha256": _sha256(price_path), "rows": len(prices)},
        {"role": "joinquant_etf_raw_nav", "path": _relative(nav_path), "sha256": _sha256(nav_path), "rows": len(nav)},
        {"role": "joinquant_etf_asset_status", "path": _relative(status_path), "sha256": _sha256(status_path), "rows": len(status)},
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": fetched_at.isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": "LIMITED_WINDOW_INDEPENDENT_VALIDATION_SOURCE",
        "query_contract": source_material,
        "source_vintage": source_vintage,
        "inputs": inputs,
        "outputs": outputs,
        "selected_assets": int(len(selected)),
        "selected_delisted_assets": int(selected["lifecycle_status"].eq("delisted").sum()),
        "price_assets_with_rows": int(prices["asset"].nunique()),
        "price_rows": int(len(prices)),
        "nav_requested_assets": int(len(nav_selected)),
        "nav_assets_with_rows": int(nav["asset"].nunique()),
        "nav_rows": int(len(nav)),
        "query_count_before": query_before,
        "query_count_after": query_after,
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "limitations": [
            "JoinQuant trial coverage is limited to the configured recent window",
            "ordinary get_price rows are raw market observations, not certified total-return prices",
            "NAV validation is restricted to assets already present in the current 30-ETF research cache",
            "the source may challenge another provider but cannot establish full-history version completeness",
        ],
    }
    _atomic_json(manifest, manifest_path)
    _atomic_json(manifest, LATEST_MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--query-start", default=DEFAULT_QUERY_START)
    parser.add_argument("--query-end", default=DEFAULT_QUERY_END)
    parser.add_argument("--nav-dir", type=Path, default=DEFAULT_NAV_DIR)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--assets", help="comma-separated governed ETF codes")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit = [item.strip() for item in args.assets.split(",") if item.strip()] if args.assets else None
    result = collect(
        as_of=args.as_of,
        query_start=args.query_start,
        query_end=args.query_end,
        nav_dir=args.nav_dir,
        batch_size=args.batch_size,
        explicit_assets=explicit,
        sleep_seconds=args.sleep_seconds,
    )
    keys = (
        "run_id",
        "selected_assets",
        "selected_delisted_assets",
        "price_assets_with_rows",
        "price_rows",
        "nav_requested_assets",
        "nav_assets_with_rows",
        "nav_rows",
        "qualification_status",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
