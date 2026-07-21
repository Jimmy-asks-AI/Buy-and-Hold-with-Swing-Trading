"""Build an observation-only ETF total-return history from current snapshots.

The source files contain historical market rows, but the ETF selection and the
distribution history are current final snapshots. The outputs are therefore
useful for diagnostics and data-quality work only. They must not be promoted to
the formal point-in-time backtest datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .etf_snapshot_builder import total_return_adjusted_prices
from .etf_index_registry import INDEX_REGISTRY_PATH


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
DEFAULT_SECURITY_MASTER = RAW_ROOT / "pit_history" / "etf_security_master.csv"
DEFAULT_OBSERVATION_DIR = RAW_ROOT / "pit_history" / "observations"
DEFAULT_PRICE_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_total_return_prices_current_universe.csv"
DEFAULT_DIVIDEND_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_dividend_events_current_universe.csv"
DEFAULT_STATUS_OUTPUT = DEFAULT_OBSERVATION_DIR / "etf_history_asset_status_current_universe.csv"
DEFAULT_MANIFEST = RAW_ROOT / "manifests" / "etf_history_observation_latest.json"
QUALIFICATION_STATUS = "OBSERVATION_ONLY_CURRENT_UNIVERSE_FINAL_SNAPSHOT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(path)


def _strict_dates(frame: pd.DataFrame, label: str) -> pd.Series:
    if "date" not in frame.columns:
        raise ValueError(f"{label} is missing date")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    if dates.duplicated().any():
        raise ValueError(f"{label} contains duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"{label} dates are not sorted")
    return dates


def _normalise_prices(raw: pd.DataFrame, asset: str, as_of: pd.Timestamp) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF price source is missing columns for {asset}: {missing}")
    prices = raw[list(required)].copy()
    prices["date"] = _strict_dates(prices, f"ETF price source {asset}")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    prices[numeric] = prices[numeric].apply(pd.to_numeric, errors="coerce")
    prices = prices[prices["date"] <= as_of].copy()
    if prices.empty or prices[numeric].isna().any(axis=None):
        raise ValueError(f"ETF price source has no complete as-of rows for {asset}")
    if (prices[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError(f"ETF price source contains non-positive OHLC for {asset}")
    if (prices[["volume", "amount"]] < 0).any(axis=None):
        raise ValueError(f"ETF price source contains negative volume/amount for {asset}")
    invalid_ohlc = (
        prices["high"].lt(prices[["open", "close", "low"]].max(axis=1))
        | prices["low"].gt(prices[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError(f"ETF price source contains invalid OHLC bounds for {asset}")
    return prices.sort_values("date").reset_index(drop=True)


def _normalise_dividends(raw: pd.DataFrame, asset: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "cumulative_dividend", "cash_distribution"])
    required = {"date", "cumulative_dividend"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF dividend source is missing columns for {asset}: {missing}")
    events = raw[["date", "cumulative_dividend"]].copy()
    events["date"] = _strict_dates(events, f"ETF dividend source {asset}")
    events["cumulative_dividend"] = pd.to_numeric(events["cumulative_dividend"], errors="coerce")
    events = events[events["date"] <= as_of].copy()
    if events["cumulative_dividend"].isna().any() or (events["cumulative_dividend"] < 0).any():
        raise ValueError(f"ETF dividend source contains invalid cumulative values for {asset}")
    events["cash_distribution"] = events["cumulative_dividend"].diff().fillna(events["cumulative_dividend"])
    if (events["cash_distribution"] < -1e-10).any():
        raise ValueError(f"ETF cumulative dividends decreased for {asset}")
    events.loc[events["cash_distribution"].abs() <= 1e-10, "cash_distribution"] = 0.0
    return events.reset_index(drop=True)


def _validate_nav(raw: pd.DataFrame, asset: str, as_of: pd.Timestamp) -> int:
    required = {"date", "unit_nav", "cumulative_nav"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"ETF NAV source is missing columns for {asset}: {missing}")
    nav = raw[list(required)].copy()
    nav["date"] = _strict_dates(nav, f"ETF NAV source {asset}")
    nav[["unit_nav", "cumulative_nav"]] = nav[["unit_nav", "cumulative_nav"]].apply(
        pd.to_numeric, errors="coerce"
    )
    nav = nav[nav["date"] <= as_of]
    if nav.empty or nav[["unit_nav", "cumulative_nav"]].isna().any(axis=None):
        raise ValueError(f"ETF NAV source has no complete as-of rows for {asset}")
    if (nav[["unit_nav", "cumulative_nav"]] <= 0).any(axis=None):
        raise ValueError(f"ETF NAV source contains non-positive values for {asset}")
    return int(len(nav))


def _observed_at(value: Any) -> pd.Timestamp:
    observed = pd.Timestamp(value)
    if pd.isna(observed):
        raise ValueError("ETF source observation timestamp is missing")
    return observed


def build_dividend_observation(
    raw_dividends: pd.DataFrame,
    asset: str,
    asset_name: str,
    tracking_index_name: str,
    as_of: str | pd.Timestamp,
    source_observed_at: Any,
    source_vintage: str,
) -> pd.DataFrame:
    """Convert a final cumulative-dividend snapshot into non-PIT event rows."""

    events = _normalise_dividends(raw_dividends, asset, pd.Timestamp(as_of).normalize())
    columns = [
        "asset",
        "asset_name",
        "asset_type",
        "tracking_index_name",
        "event_date",
        "cumulative_dividend",
        "cash_distribution",
        "event_type",
        "source_observed_at",
        "source_observed_date",
        "available_date",
        "pit_actionable",
        "data_source",
        "source_vintage",
        "qualification_status",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    observed = _observed_at(source_observed_at)
    out = events.rename(columns={"date": "event_date"}).copy()
    out.insert(0, "asset", asset)
    out.insert(1, "asset_name", asset_name)
    out.insert(2, "asset_type", "etf")
    out.insert(3, "tracking_index_name", tracking_index_name)
    out["event_type"] = np.where(out["cash_distribution"] > 0, "cash_distribution", "zero_marker")
    out["source_observed_at"] = observed.isoformat()
    out["source_observed_date"] = observed.normalize()
    out["available_date"] = observed.normalize()
    out["pit_actionable"] = False
    out["data_source"] = "sina cumulative ETF distributions; current final snapshot"
    out["source_vintage"] = source_vintage
    out["qualification_status"] = QUALIFICATION_STATUS
    out["historical_backtest_allowed"] = False
    out["model_promotion_allowed"] = False
    return out[columns]


def build_price_observation(
    raw_prices: pd.DataFrame,
    raw_dividends: pd.DataFrame,
    raw_nav: pd.DataFrame,
    asset: str,
    asset_name: str,
    tracking_index_name: str,
    exchange: str,
    list_date: Any,
    as_of: str | pd.Timestamp,
    source_observed_at: Any,
    source_vintage: str,
) -> tuple[pd.DataFrame, int]:
    """Build explicit adjusted OHLC and a diagnostic total-return index."""

    cutoff = pd.Timestamp(as_of).normalize()
    prices = _normalise_prices(raw_prices, asset, cutoff)
    events = _normalise_dividends(raw_dividends, asset, cutoff)
    nav_rows = _validate_nav(raw_nav, asset, cutoff)
    listed = pd.Timestamp(list_date).normalize()
    if pd.isna(listed) or listed > cutoff or prices["date"].min() < listed:
        raise ValueError(f"ETF price history falls outside the governed lifecycle for {asset}")
    observed = _observed_at(source_observed_at)
    adjusted = total_return_adjusted_prices(prices, events, asset, cutoff)
    adjusted_dates = pd.to_datetime(adjusted["date"], errors="coerce")
    if adjusted_dates.isna().any() or adjusted_dates.duplicated().any() or not adjusted_dates.is_monotonic_increasing:
        raise ValueError(f"ETF adjusted history has invalid dates for {asset}")

    # The adjusted close already embeds each cash distribution. Adding the cash
    # flow again here would double-count dividends.
    period_return = pd.to_numeric(adjusted["close"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    if not np.isfinite(period_return).all() or (period_return <= -1.0).any():
        raise ValueError(f"ETF reconstructed total returns are invalid for {asset}")
    total_return_index = (1.0 + period_return).cumprod()

    out = pd.DataFrame(
        {
            "asset": asset,
            "asset_name": asset_name,
            "asset_type": "etf",
            "exchange": exchange,
            "tracking_index_name": tracking_index_name,
            "list_date": listed,
            "date": adjusted_dates,
            "market_available_date": adjusted_dates,
            "source_observed_at": observed.isoformat(),
            "source_observed_date": observed.normalize(),
            "available_date": observed.normalize(),
            "raw_close": pd.to_numeric(adjusted["raw_close"], errors="coerce"),
            "adjusted_open": pd.to_numeric(adjusted["open"], errors="coerce"),
            "adjusted_high": pd.to_numeric(adjusted["high"], errors="coerce"),
            "adjusted_low": pd.to_numeric(adjusted["low"], errors="coerce"),
            "adjusted_close": pd.to_numeric(adjusted["close"], errors="coerce"),
            "volume": pd.to_numeric(adjusted["volume"], errors="coerce"),
            "amount": pd.to_numeric(adjusted["amount"], errors="coerce"),
            "source_cash_distribution": pd.to_numeric(adjusted["source_cash_distribution"], errors="coerce"),
            "cash_distribution": pd.to_numeric(adjusted["cash_distribution"], errors="coerce"),
            "share_adjustment_factor": pd.to_numeric(adjusted["share_adjustment_factor"], errors="coerce"),
            "adjustment_factor": pd.to_numeric(adjusted["adjustment_factor"], errors="coerce"),
            "period_total_return": period_return,
            "total_return_index": total_return_index,
            "return_basis": "current-final-snapshot total-return adjusted close",
            "pit_actionable": False,
            "data_source": "sina ETF OHLC + sina cumulative distributions; current final snapshot",
            "source_vintage": source_vintage,
            "qualification_status": QUALIFICATION_STATUS,
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
    )
    if out[["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "total_return_index"]].isna().any(
        axis=None
    ):
        raise ValueError(f"ETF reconstructed output contains missing prices for {asset}")
    return out, nav_rows


AssetLoader = Callable[[str], dict[str, pd.DataFrame]]


def process_observation_batch(
    selected: pd.DataFrame,
    loader: AssetLoader,
    as_of: str | pd.Timestamp,
    source_vintage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process assets independently so one malformed history is quarantined."""

    price_frames: list[pd.DataFrame] = []
    dividend_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        asset = str(row.get("asset", "")).zfill(6)
        status: dict[str, Any] = {
            "asset": asset,
            "asset_name": str(row.get("name", row.get("asset_name", ""))),
            "tracking_index_name": str(row.get("tracking_index_name", "")),
            "status": "quarantined",
            "error": "",
            "price_rows": 0,
            "dividend_event_rows": 0,
            "nav_rows_validated": 0,
            "pit_actionable": False,
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
            "qualification_status": QUALIFICATION_STATUS,
        }
        try:
            collected = str(row.get("asset_history_collected", "")).strip().lower()
            if collected not in {"true", "1"} and row.get("asset_history_collected") is not True:
                raise ValueError(f"ETF snapshot did not collect all asset histories for {asset}")
            frames = loader(asset)
            price_output, nav_rows = build_price_observation(
                frames["price"],
                frames["dividend"],
                frames["nav"],
                asset,
                status["asset_name"],
                status["tracking_index_name"],
                str(row.get("exchange", "")),
                row.get("list_date"),
                as_of,
                row.get("fetched_at"),
                source_vintage,
            )
            dividend_output = build_dividend_observation(
                frames["dividend"],
                asset,
                status["asset_name"],
                status["tracking_index_name"],
                as_of,
                row.get("fetched_at"),
                source_vintage,
            )
            price_frames.append(price_output)
            if not dividend_output.empty:
                dividend_frames.append(dividend_output)
            status.update(
                {
                    "status": "ok",
                    "price_rows": len(price_output),
                    "dividend_event_rows": len(dividend_output),
                    "nav_rows_validated": nav_rows,
                }
            )
        except Exception as exc:  # noqa: BLE001 - quarantine is the data contract
            status["error"] = str(exc)
        statuses.append(status)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    dividends = pd.concat(dividend_frames, ignore_index=True) if dividend_frames else pd.DataFrame()
    asset_status = pd.DataFrame(statuses)
    if not prices.empty and prices.duplicated(["asset", "date"]).any():
        raise ValueError("ETF observation output contains duplicate asset/date rows")
    if not dividends.empty and dividends.duplicated(["asset", "event_date"]).any():
        raise ValueError("ETF dividend observation output contains duplicate asset/event rows")
    return prices, dividends, asset_status


def _active_lifecycle(master: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    required = {"asset", "list_date", "event_type", "exchange"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"ETF security master is missing columns: {missing}")
    data = master.copy()
    data["asset"] = data["asset"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    listings = data[data["event_type"].eq("listing")][["asset", "list_date", "exchange"]].copy()
    if listings["asset"].duplicated().any():
        raise ValueError("ETF security master contains duplicate listing events")
    listings["list_date"] = pd.to_datetime(listings["list_date"], errors="coerce")
    exits = set(
        data.loc[
            data["event_type"].eq("delisting")
            & pd.to_datetime(data.get("delist_date"), errors="coerce").le(as_of),
            "asset",
        ]
    )
    return listings[~listings["asset"].isin(exits)].reset_index(drop=True)


def run(
    as_of: str | pd.Timestamp,
    raw_dir: Path,
    snapshot_manifest_path: Path,
    security_master_path: Path,
    price_output_path: Path,
    dividend_output_path: Path,
    status_output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    if not snapshot_manifest_path.is_file() or not security_master_path.is_file():
        raise ValueError("ETF snapshot manifest and security master are required")
    selected = pd.read_csv(snapshot_manifest_path, dtype={"asset": str}, low_memory=False)
    required_manifest = {"asset", "name", "tracking_index_name", "asset_history_collected", "fetched_at"}
    missing = sorted(required_manifest.difference(selected.columns))
    if missing or selected.empty:
        raise ValueError(f"ETF snapshot manifest is incomplete: {missing}")
    selected["asset"] = selected["asset"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if selected["asset"].duplicated().any():
        raise ValueError("ETF snapshot manifest contains duplicate assets")

    master = pd.read_csv(security_master_path, dtype={"asset": str}, low_memory=False)
    lifecycle = _active_lifecycle(master, cutoff)
    selected = selected.merge(lifecycle, on="asset", how="left", validate="one_to_one")

    declared_inputs: list[dict[str, Any]] = [
        {"role": "etf_snapshot_manifest", "path": snapshot_manifest_path},
        {"role": "etf_security_master", "path": security_master_path},
        {"role": "etf_index_registry", "path": INDEX_REGISTRY_PATH},
    ]
    asset_paths: dict[str, dict[str, Path]] = {}
    for asset in selected["asset"]:
        paths = {
            "price": raw_dir / "price" / f"{asset}.csv",
            "dividend": raw_dir / "dividend" / f"{asset}.csv",
            "nav": raw_dir / "nav" / f"{asset}.csv",
        }
        asset_paths[asset] = paths
        declared_inputs.extend({"role": role, "asset": asset, "path": path} for role, path in paths.items())

    input_records: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, str]] = []
    canonical_inputs: list[dict[str, Any]] = []
    for item in declared_inputs:
        path = Path(item["path"])
        record = {key: value for key, value in item.items() if key != "path"}
        record["path"] = _relative(path)
        if path.is_file():
            record["sha256"] = _sha256(path)
            record["bytes"] = path.stat().st_size
            input_records.append(record)
            canonical_inputs.append(record)
        else:
            missing_record = {key: str(value) for key, value in record.items()}
            missing_inputs.append(missing_record)
            canonical_inputs.append({**record, "missing": True})
    bundle_hash = hashlib.sha256(
        json.dumps(canonical_inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_vintage = f"etf_current_universe_final_snapshot_bundle_sha256:{bundle_hash}"

    def loader(asset: str) -> dict[str, pd.DataFrame]:
        paths = asset_paths[asset]
        missing_asset_paths = [str(path) for path in paths.values() if not path.is_file()]
        if missing_asset_paths:
            raise ValueError(f"ETF raw history files are missing for {asset}: {missing_asset_paths}")
        return {
            "price": pd.read_csv(paths["price"], low_memory=False),
            "dividend": pd.read_csv(paths["dividend"], low_memory=False),
            "nav": pd.read_csv(paths["nav"], low_memory=False),
        }

    prices, dividends, asset_status = process_observation_batch(selected, loader, cutoff, source_vintage)
    if prices.empty:
        raise ValueError("all ETF histories were quarantined")
    _atomic_csv(prices, price_output_path)
    _atomic_csv(dividends, dividend_output_path)
    _atomic_csv(asset_status, status_output_path)

    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("etf_index_registry.py").resolve(),
        Path(__file__).with_name("etf_snapshot_builder.py").resolve(),
    ]
    code_records = [{"path": _relative(path), "sha256": _sha256(path)} for path in code_paths]
    ok_assets = set(asset_status.loc[asset_status["status"].eq("ok"), "asset"])
    result: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "market_data_cutoff_date": str(cutoff.date()),
        "inputs": input_records,
        "missing_inputs": missing_inputs,
        "input_bundle_sha256": bundle_hash,
        "source_vintage": source_vintage,
        "code_files": code_records,
        "outputs": [
            {"role": "etf_total_return_prices_observation", "path": _relative(price_output_path), "sha256": _sha256(price_output_path)},
            {"role": "etf_dividend_events_observation", "path": _relative(dividend_output_path), "sha256": _sha256(dividend_output_path)},
            {"role": "etf_history_asset_status", "path": _relative(status_output_path), "sha256": _sha256(status_output_path)},
        ],
        "selected_assets": int(len(selected)),
        "successful_assets": int(len(ok_assets)),
        "quarantined_assets": int(asset_status["status"].eq("quarantined").sum()),
        "quarantined_asset_codes": asset_status.loc[asset_status["status"].eq("quarantined"), "asset"].tolist(),
        "price_rows": int(len(prices)),
        "dividend_event_rows": int(len(dividends)),
        "dividend_event_assets": int(dividends["asset"].nunique()) if not dividends.empty else 0,
        "coverage_start": str(pd.to_datetime(prices["date"]).min().date()),
        "coverage_end": str(pd.to_datetime(prices["date"]).max().date()),
        "source_observed_at_min": str(selected.loc[selected["asset"].isin(ok_assets), "fetched_at"].min()),
        "source_observed_at_max": str(selected.loc[selected["asset"].isin(ok_assets), "fetched_at"].max()),
        "qualification_status": QUALIFICATION_STATUS,
        "pit_actionable": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "limitations": [
            "the ETF population is selected from a current universe snapshot and carries survivorship bias",
            "cumulative distributions are current final snapshots without historical first-publication timestamps",
            "provider dividend dates are not independently verified as exchange ex-dates",
            "adjusted prices are diagnostics only; raw_close and market_available_date preserve the observable market fields",
            "unsupported benchmark mappings do not prevent asset-history observation and remain a separate data gap",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--security-master", type=Path, default=DEFAULT_SECURITY_MASTER)
    parser.add_argument("--price-output", type=Path, default=DEFAULT_PRICE_OUTPUT)
    parser.add_argument("--dividend-output", type=Path, default=DEFAULT_DIVIDEND_OUTPUT)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.as_of).normalize()
    raw_dir = args.raw_dir or RAW_ROOT / "etf_raw" / cutoff.strftime("%Y%m%d")
    snapshot_manifest = args.snapshot_manifest or RAW_ROOT / "manifests" / f"etf_snapshot_{cutoff.strftime('%Y%m%d')}.csv"
    result = run(
        cutoff,
        raw_dir,
        snapshot_manifest,
        args.security_master,
        args.price_output,
        args.dividend_output,
        args.status_output,
        args.manifest,
    )
    summary_keys = (
        "selected_assets",
        "successful_assets",
        "quarantined_assets",
        "price_rows",
        "dividend_event_rows",
        "coverage_start",
        "coverage_end",
        "qualification_status",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
