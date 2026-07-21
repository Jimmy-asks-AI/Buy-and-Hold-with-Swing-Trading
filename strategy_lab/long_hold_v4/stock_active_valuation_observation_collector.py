"""Collect current-final-snapshot valuation history for active A-share candidates.

Eastmoney exposes a convenient historical series, but the response is observed
today and does not prove what was available on each historical date.  This
collector therefore preserves the actual observation timestamp and marks every
row as non-PIT.  The output can support current cross-sectional research and
cross-provider diagnostics; it cannot qualify a historical backtest input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .pit_stock_market_history_builder import load_lifecycles


ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = ROOT / "data_raw" / "long_hold_v4" / "pit_history"
MASTER_PATH = PIT_ROOT / "stock_security_master.csv"
WATCHLIST_PATH = ROOT / "data_catalog" / "long_hold_v4_watchlist.csv"
LEGACY_CACHE_DIR = PIT_ROOT / "validation_sources" / "eastmoney_valuation"
RAW_CACHE_DIR = PIT_ROOT / "observations" / "raw_eastmoney_active_valuation"
OBSERVATION_DIR = PIT_ROOT / "observations"
OBSERVATION_PATH = OBSERVATION_DIR / "stock_active_valuation_history_eastmoney.csv.gz"
STATUS_PATH = OBSERVATION_DIR / "stock_active_valuation_history_eastmoney_status.csv"
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_active_valuation_observation_latest.json"
)
ARCHIVE_DIR = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_active_valuation_observation_runs"
)

SOURCE_NAME = "akshare.stock_value_em/eastmoney"
QUALIFICATION_STATUS = "OBSERVATION_ONLY_CURRENT_FINAL_SNAPSHOT"
MAX_SOURCE_ROWS = 5_000

SOURCE_COLUMNS = {
    "数据日期",
    "当日收盘价",
    "当日涨跌幅",
    "总市值",
    "流通市值",
    "总股本",
    "流通股本",
    "PE(TTM)",
    "PE(静)",
    "市净率",
    "PEG值",
    "市现率",
    "市销率",
}

OBSERVATION_COLUMNS = [
    "date",
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "delist_date",
    "close",
    "pct_change",
    "total_market_cap",
    "float_market_cap",
    "total_shares",
    "free_float_shares",
    "pe_ttm",
    "pe_static",
    "pb_mrq",
    "peg",
    "pcf_ttm",
    "ps_ttm",
    "market_available_date",
    "source_observed_at",
    "available_date",
    "data_source",
    "source_vintage",
    "current_final_snapshot",
    "pit_actionable",
    "qualification_status",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]

STATUS_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "selection_group",
    "list_date",
    "delist_date",
    "collection_status",
    "collection_action",
    "cache_path",
    "cache_sha256",
    "rows",
    "coverage_start",
    "coverage_end",
    "source_observed_at",
    "error",
    "qualification_status",
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
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    temporary.replace(path)


def _normalise_asset(value: Any) -> str:
    asset = str(value).strip()
    if asset.endswith(".0"):
        asset = asset[:-2]
    if not asset.isdigit() or len(asset) > 6:
        raise ValueError(f"invalid A-share asset code: {value}")
    return asset.zfill(6)


def _load_asset_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, dtype=str)
        if "asset" in frame.columns:
            values = frame["asset"].dropna().tolist()
        elif len(frame.columns) == 1:
            values = frame.iloc[:, 0].dropna().tolist()
        else:
            raise ValueError("asset file must contain an asset column")
    except pd.errors.ParserError:
        values = path.read_text(encoding="utf-8-sig").splitlines()
    return {_normalise_asset(value) for value in values if str(value).strip()}


def select_target_assets(
    as_of: str | pd.Timestamp,
    universe: str = "candidates",
    asset_file: Path | None = None,
    master_path: Path = MASTER_PATH,
    candidate_path: Path = WATCHLIST_PATH,
) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of).normalize()
    lifecycles = load_lifecycles(master_path, as_of=as_of_date)
    active = lifecycles[
        lifecycles["delist_date"].isna() | lifecycles["delist_date"].gt(as_of_date)
    ].copy()
    listing_rows = pd.read_csv(master_path, dtype={"asset": str})
    listing_rows = listing_rows[listing_rows["event_type"].eq("listing")][
        ["asset", "asset_name"]
    ].copy()
    listing_rows["asset"] = listing_rows["asset"].map(_normalise_asset)
    active = active.merge(listing_rows, on="asset", how="left", validate="one_to_one")

    if universe == "candidates":
        candidates = pd.read_csv(candidate_path, dtype={"asset": str})
        if "asset" not in candidates.columns:
            raise ValueError("candidate watchlist missing asset column")
        candidates["asset"] = candidates["asset"].map(_normalise_asset)
        if candidates["asset"].duplicated().any():
            raise ValueError("candidate watchlist contains duplicate assets")
        selected = set(candidates["asset"])
        active = active[active["asset"].isin(selected)].copy()
        active["selection_group"] = "current_watchlist_candidates"
    elif universe == "active_all":
        active["selection_group"] = "current_active_all"
    else:
        raise ValueError("universe must be candidates or active_all")

    if asset_file is not None:
        requested = _load_asset_file(asset_file)
        active = active[active["asset"].isin(requested)].copy()
        active["selection_group"] = active["selection_group"] + "_asset_file"
    return active[
        ["asset", "asset_name", "exchange", "list_date", "delist_date", "selection_group"]
    ].sort_values("asset").reset_index(drop=True)


def validate_source_frame(raw: pd.DataFrame, asset: str, as_of: str | pd.Timestamp) -> pd.DataFrame:
    missing = sorted(SOURCE_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Eastmoney valuation response missing columns: {missing}")
    if raw.empty:
        raise ValueError("Eastmoney valuation response is empty")
    if len(raw) >= MAX_SOURCE_ROWS:
        raise ValueError("Eastmoney valuation response reached the 5000-row page cap")
    frame = raw.copy()
    frame["数据日期"] = pd.to_datetime(frame["数据日期"], errors="coerce").dt.normalize()
    if frame["数据日期"].isna().any():
        raise ValueError("Eastmoney valuation response contains invalid dates")
    if frame["数据日期"].duplicated().any():
        raise ValueError("Eastmoney valuation response contains duplicate dates")
    if frame["数据日期"].gt(pd.Timestamp(as_of).normalize()).any():
        raise ValueError("Eastmoney valuation response contains future dates")
    if "asset" in frame.columns:
        observed_assets = frame["asset"].dropna().map(_normalise_asset).unique().tolist()
        if observed_assets and observed_assets != [_normalise_asset(asset)]:
            raise ValueError(f"cached response contains a different asset for {asset}")
    for column in SOURCE_COLUMNS.difference({"数据日期"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["当日收盘价"].isna().any() or not frame["当日收盘价"].gt(0).all():
        raise ValueError("Eastmoney valuation response contains invalid close prices")
    for column in ["总市值", "流通市值", "总股本", "流通股本"]:
        values = frame[column].dropna()
        if not values.empty and not values.gt(0).all():
            raise ValueError(f"Eastmoney valuation response contains invalid {column}")
    return frame.sort_values("数据日期").reset_index(drop=True)


def normalise_observation(
    raw: pd.DataFrame,
    lifecycle: pd.Series,
    as_of: str | pd.Timestamp,
    source_observed_at: str,
) -> pd.DataFrame:
    asset = _normalise_asset(lifecycle["asset"])
    frame = validate_source_frame(raw, asset, as_of)
    start = pd.Timestamp(lifecycle["list_date"]).normalize()
    end = pd.Timestamp(as_of).normalize()
    if pd.notna(lifecycle["delist_date"]):
        end = min(end, pd.Timestamp(lifecycle["delist_date"]).normalize())
    frame = frame[frame["数据日期"].between(start, end)].copy()
    if frame.empty:
        raise ValueError("Eastmoney valuation response has no rows inside the security lifecycle")
    rename = {
        "数据日期": "date",
        "当日收盘价": "close",
        "当日涨跌幅": "pct_change",
        "总市值": "total_market_cap",
        "流通市值": "float_market_cap",
        "总股本": "total_shares",
        "流通股本": "free_float_shares",
        "PE(TTM)": "pe_ttm",
        "PE(静)": "pe_static",
        "市净率": "pb_mrq",
        "PEG值": "peg",
        "市现率": "pcf_ttm",
        "市销率": "ps_ttm",
    }
    output = frame.rename(columns=rename)[list(rename.values())].copy()
    observed_at = pd.Timestamp(source_observed_at)
    if observed_at.tzinfo is None:
        observed_at = observed_at.tz_localize("Asia/Shanghai")
    output["asset"] = asset
    output["asset_name"] = str(lifecycle.get("asset_name", ""))
    output["exchange"] = str(lifecycle["exchange"])
    output["list_date"] = start
    output["delist_date"] = lifecycle["delist_date"]
    output["market_available_date"] = output["date"]
    output["source_observed_at"] = observed_at.isoformat()
    output["available_date"] = observed_at.normalize().date().isoformat()
    output["data_source"] = SOURCE_NAME
    output["source_vintage"] = observed_at.date().isoformat()
    output["current_final_snapshot"] = True
    output["pit_actionable"] = False
    output["qualification_status"] = QUALIFICATION_STATUS
    output["historical_backtest_allowed"] = False
    output["model_promotion_allowed"] = False
    return output[OBSERVATION_COLUMNS].sort_values("date").reset_index(drop=True)


def _cache_path(asset: str, raw_cache_dir: Path = RAW_CACHE_DIR) -> Path:
    return raw_cache_dir / f"{_normalise_asset(asset)}.csv.gz"


def _read_cache(path: Path, asset: str, as_of: str | pd.Timestamp) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, compression="gzip", dtype={"asset": str}, low_memory=False)
    clean = validate_source_frame(frame, asset, as_of)
    if "fetched_at" in frame.columns and frame["fetched_at"].notna().any():
        observed_at = str(frame.loc[frame["fetched_at"].notna(), "fetched_at"].iloc[0])
    else:
        observed_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    return clean, observed_at


def _find_valid_cache(
    asset: str,
    as_of: str | pd.Timestamp,
    raw_cache_dir: Path,
    legacy_cache_dir: Path,
) -> tuple[pd.DataFrame | None, str, Path | None, str]:
    errors: list[str] = []
    for action, path in [
        ("reused_primary_cache", _cache_path(asset, raw_cache_dir)),
        ("reused_legacy_validation_cache", _cache_path(asset, legacy_cache_dir)),
    ]:
        if not path.is_file():
            continue
        try:
            frame, observed_at = _read_cache(path, asset, as_of)
            return frame, observed_at, path, action
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            errors.append(f"{path.name}:{type(exc).__name__}:{str(exc)[:160]}")
    return None, "", None, ";".join(errors)


def _atomic_source_cache(raw: pd.DataFrame, asset: str, fetched_at: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = raw.copy()
    output["asset"] = _normalise_asset(asset)
    output["fetched_at"] = fetched_at
    output["data_source"] = SOURCE_NAME
    temporary = Path(str(path) + ".tmp")
    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    temporary.replace(path)


def _fetch_asset(
    asset: str,
    as_of: str | pd.Timestamp,
    fetcher: Callable[[str], pd.DataFrame],
    attempts: int,
    retry_backoff_seconds: float,
) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            raw = fetcher(asset)
            fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
            return validate_source_frame(raw, asset, as_of), fetched_at
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts and retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def _is_provider_circuit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}:{exc}".lower()
    markers = ["403", "429", "blacklist", "too frequent", "rate limit", "connection aborted"]
    return any(marker in text for marker in markers)


def collect_observation(
    as_of: str,
    universe: str = "candidates",
    asset_file: Path | None = None,
    max_fetch: int = 20,
    sleep_seconds: float = 1.0,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 2.0,
    max_consecutive_failures: int = 3,
    fetcher: Callable[[str], pd.DataFrame] | None = None,
    master_path: Path = MASTER_PATH,
    candidate_path: Path = WATCHLIST_PATH,
    raw_cache_dir: Path = RAW_CACHE_DIR,
    legacy_cache_dir: Path = LEGACY_CACHE_DIR,
    observation_path: Path = OBSERVATION_PATH,
    status_path: Path = STATUS_PATH,
    manifest_path: Path = MANIFEST_PATH,
    archive_dir: Path = ARCHIVE_DIR,
) -> dict[str, Any]:
    if max_fetch < 0 or sleep_seconds < 0 or retry_attempts < 1 or max_consecutive_failures < 1:
        raise ValueError("invalid collection control")
    as_of_date = pd.Timestamp(as_of).normalize()
    targets = select_target_assets(as_of_date, universe, asset_file, master_path, candidate_path)
    if targets.empty:
        raise ValueError("no active target assets selected")
    target_rows = {row.asset: row for row in targets.itertuples(index=False)}
    cache_state: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for asset in targets["asset"]:
        raw, observed_at, path, action = _find_valid_cache(
            asset, as_of_date, raw_cache_dir, legacy_cache_dir
        )
        if raw is not None and path is not None:
            cache_state[asset] = {
                "raw": raw,
                "observed_at": observed_at,
                "path": path,
                "action": action,
                "error": "",
            }
        else:
            pending.append(asset)
            cache_state[asset] = {
                "raw": None,
                "observed_at": "",
                "path": None,
                "action": "pending",
                "error": action,
            }

    selected = pending[:max_fetch]
    if selected and fetcher is None:
        import akshare as ak

        fetcher = ak.stock_value_em
    consecutive_failures = 0
    circuit_open = False
    for position, asset in enumerate(selected):
        if circuit_open:
            cache_state[asset]["action"] = "deferred_provider_circuit_open"
            continue
        try:
            assert fetcher is not None
            raw, fetched_at = _fetch_asset(
                asset,
                as_of_date,
                fetcher,
                retry_attempts,
                retry_backoff_seconds,
            )
            path = _cache_path(asset, raw_cache_dir)
            _atomic_source_cache(raw, asset, fetched_at, path)
            cache_state[asset] = {
                "raw": raw,
                "observed_at": fetched_at,
                "path": path,
                "action": "fetched",
                "error": "",
            }
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            cache_state[asset]["action"] = "failed"
            cache_state[asset]["error"] = f"{type(exc).__name__}:{str(exc)[:300]}"
            if _is_provider_circuit_error(exc) or consecutive_failures >= max_consecutive_failures:
                circuit_open = True
        if sleep_seconds > 0 and position + 1 < len(selected) and not circuit_open:
            time.sleep(sleep_seconds)
    for asset in pending[max_fetch:]:
        cache_state[asset]["action"] = "deferred_fetch_limit"

    observation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_observation = Path(str(observation_path) + ".tmp")
    wrote_header = False
    total_rows = 0
    built_assets = 0
    status_rows: list[dict[str, Any]] = []
    cache_inputs: list[dict[str, str]] = []
    with gzip.open(temporary_observation, "wt", encoding="utf-8-sig", newline="") as handle:
        for asset in targets["asset"]:
            lifecycle = pd.Series(target_rows[asset]._asdict())
            state = cache_state[asset]
            raw = state["raw"]
            path = state["path"]
            observation = pd.DataFrame(columns=OBSERVATION_COLUMNS)
            build_error = str(state["error"])
            if raw is not None and path is not None:
                try:
                    observation = normalise_observation(
                        raw, lifecycle, as_of_date, str(state["observed_at"])
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    build_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            if not observation.empty:
                observation.to_csv(
                    handle,
                    index=False,
                    header=not wrote_header,
                    date_format="%Y-%m-%d",
                    lineterminator="\n",
                )
                wrote_header = True
                total_rows += len(observation)
                built_assets += 1
                cache_inputs.append({"path": _relative(path), "sha256": _sha256(path)})
            status_rows.append(
                {
                    "asset": asset,
                    "asset_name": lifecycle["asset_name"],
                    "exchange": lifecycle["exchange"],
                    "selection_group": lifecycle["selection_group"],
                    "list_date": lifecycle["list_date"],
                    "delist_date": lifecycle["delist_date"],
                    "collection_status": "completed" if not observation.empty else "incomplete",
                    "collection_action": state["action"],
                    "cache_path": _relative(path) if path is not None else "",
                    "cache_sha256": _sha256(path) if path is not None and path.is_file() else "",
                    "rows": len(observation),
                    "coverage_start": observation["date"].min() if not observation.empty else "",
                    "coverage_end": observation["date"].max() if not observation.empty else "",
                    "source_observed_at": state["observed_at"],
                    "error": build_error,
                    "qualification_status": QUALIFICATION_STATUS,
                    "historical_backtest_allowed": False,
                    "model_promotion_allowed": False,
                }
            )
        if not wrote_header:
            pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(handle, index=False, lineterminator="\n")
    temporary_observation.replace(observation_path)

    statuses = pd.DataFrame(status_rows, columns=STATUS_COLUMNS).sort_values("asset").reset_index(drop=True)
    _atomic_csv(statuses, status_path)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    universe_inputs = [{"path": _relative(master_path), "sha256": _sha256(master_path)}]
    if universe == "candidates":
        universe_inputs.append({"path": _relative(candidate_path), "sha256": _sha256(candidate_path)})
    if asset_file is not None:
        universe_inputs.append({"path": _relative(asset_file), "sha256": _sha256(asset_file)})
    manifest = {
        "created_at": created_at,
        "as_of_date": as_of_date.date().isoformat(),
        "dataset_id": "stock_active_valuation_history_eastmoney_observation",
        "schema_version": "v2",
        "source": SOURCE_NAME,
        "universe": universe,
        "target_universe_type": "current_watchlist" if universe == "candidates" else "current_active_master",
        "target_universe_is_current_snapshot": False,
        "target_universe_is_current_watchlist": universe == "candidates",
        "current_universe_only": True,
        "target_assets": int(len(targets)),
        "completed_assets": int(built_assets),
        "incomplete_assets": int(len(targets) - built_assets),
        "observation_rows": int(total_rows),
        "fetched_assets": int(statuses["collection_action"].eq("fetched").sum()),
        "reused_primary_cache_assets": int(
            statuses["collection_action"].eq("reused_primary_cache").sum()
        ),
        "reused_legacy_cache_assets": int(
            statuses["collection_action"].eq("reused_legacy_validation_cache").sum()
        ),
        "failed_assets": int(statuses["collection_action"].eq("failed").sum()),
        "deferred_assets": int(statuses["collection_action"].str.startswith("deferred").sum()),
        "provider_circuit_opened": bool(circuit_open),
        "inputs": [*universe_inputs, *cache_inputs],
        "outputs": [
            {"role": "observation", "path": _relative(observation_path), "sha256": _sha256(observation_path)},
            {"role": "status", "path": _relative(status_path), "sha256": _sha256(status_path)},
        ],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "available_date_policy": "actual source observation date, never historical market date",
        "research_use": "current cross-section and cross-provider diagnostics only",
        "survivorship_bias_for_historical_research": True,
        "qualification_status": QUALIFICATION_STATUS,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(manifest, manifest_path)
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    archive_path = archive_dir / f"{run_id}.json"
    _atomic_json(manifest, archive_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--universe", choices=["candidates", "active_all"], default="candidates")
    parser.add_argument("--asset-file", type=Path)
    parser.add_argument("--max-fetch", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect_observation(
        as_of=args.as_of,
        universe=args.universe,
        asset_file=args.asset_file,
        max_fetch=args.max_fetch,
        sleep_seconds=args.sleep_seconds,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    print(
        json.dumps(
            {
                "target_assets": manifest["target_assets"],
                "completed_assets": manifest["completed_assets"],
                "fetched_assets": manifest["fetched_assets"],
                "failed_assets": manifest["failed_assets"],
                "deferred_assets": manifest["deferred_assets"],
                "observation_rows": manifest["observation_rows"],
                "provider_circuit_opened": manifest["provider_circuit_opened"],
                "qualification_status": manifest["qualification_status"],
                "historical_backtest_allowed": manifest["historical_backtest_allowed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
