"""Collect lifecycle-bounded Sina HFQ factor events for Long Hold V4 Gate E2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
DIVIDEND_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_dividend_events.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_adjustment"
STATUS_PATH = RAW_DIR / "collection_status.json"
FORMAL_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_adjustment_factor.csv"
MIN_DIVIDEND_ALIGNMENT = 0.995
PROGRESS_OUTPUT = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "stock_adjustment_factor_sina_progress.csv"
)
FORMAL_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_adjustment_builder_latest.json"
PROGRESS_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_adjustment_probe_latest.json"
SOURCE_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_lifecycles(path: Path = MASTER_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError("stock_security_master.csv is required before adjustment-factor collection")
    frame = pd.read_csv(path, dtype={"asset": str})
    listed = frame[frame["event_type"].eq("listing")][["asset", "exchange", "list_date"]].copy()
    exits = frame[frame["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listed["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("stock master contains duplicate lifecycle events")
    output = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce")
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce")
    if output[["asset", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("stock master contains incomplete lifecycle keys")
    return output.sort_values("asset").reset_index(drop=True)


def parse_sina_hfq_response(text: str) -> pd.DataFrame:
    try:
        payload = text.split("=", 1)[1].splitlines()[0].strip().rstrip(";")
        rows = ast.literal_eval(payload)["data"]
    except (IndexError, KeyError, SyntaxError, ValueError, TypeError) as exc:
        raise ValueError("invalid Sina HFQ factor response") from exc
    frame = pd.DataFrame(rows)
    if {"d", "f"}.issubset(frame.columns):
        frame = frame.rename(columns={"d": "date", "f": "hfq_factor"})[["date", "hfq_factor"]]
    elif frame.shape[1] == 2:
        frame.columns = ["date", "hfq_factor"]
    else:
        raise ValueError("Sina HFQ factor response has an unsupported row schema")
    if frame.empty:
        raise ValueError("Sina HFQ factor response is empty")
    return frame


def normalise_factor_events(
    raw: pd.DataFrame,
    asset: str,
    list_date: str | pd.Timestamp,
    delist_date: str | pd.Timestamp | None,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    required = {"date", "hfq_factor"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"factor data missing columns: {missing}")
    start = pd.Timestamp(list_date).normalize()
    as_of_date = pd.Timestamp(as_of).normalize()
    end = min(pd.Timestamp(delist_date).normalize(), as_of_date) if pd.notna(delist_date) else as_of_date
    frame = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(raw["date"], errors="coerce"),
            "adj_factor": pd.to_numeric(raw["hfq_factor"], errors="coerce"),
        }
    ).dropna()
    frame = frame[(frame["adj_factor"] > 0) & (frame["effective_date"] <= end)].sort_values("effective_date")
    if frame.empty or frame["effective_date"].duplicated().any():
        raise ValueError(f"{asset} has no valid unique HFQ factor history")
    baseline = frame[frame["effective_date"] <= start].tail(1)
    if baseline.empty:
        raise ValueError(f"{asset} has no factor baseline on or before listing")
    selected = frame[(frame["effective_date"] >= start) & (frame["effective_date"] <= end)].copy()
    if selected.empty or selected.iloc[0]["effective_date"] > start:
        selected = pd.concat(
            [pd.DataFrame([{"effective_date": start, "adj_factor": baseline.iloc[0]["adj_factor"]}]), selected],
            ignore_index=True,
        )
    selected["asset"] = str(asset).zfill(6)
    selected["adjustment_basis"] = "hfq_cumulative"
    selected["available_date"] = selected["effective_date"]
    return selected[["asset", "effective_date", "adj_factor", "adjustment_basis", "available_date"]].reset_index(drop=True)


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.4, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }
    )
    return session


def _symbol(asset: str, exchange: str) -> str:
    prefix = {"SSE": "sh", "SZSE": "sz"}.get(exchange)
    if prefix is None:
        raise ValueError(f"unsupported exchange for {asset}: {exchange}")
    return f"{prefix}{asset}"


def _load_status() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"assets": {}}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def _save_status(status: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _valid_cache(entry: dict[str, Any]) -> bool:
    relative = str(entry.get("path", ""))
    path = ROOT / relative
    if entry.get("status") != "success" or not path.is_file() or _sha256(path) != str(entry.get("sha256", "")):
        return False
    try:
        frame = pd.read_csv(path, nrows=5)
    except (OSError, pd.errors.ParserError):
        return False
    return bool(
        {"date", "hfq_factor"}.issubset(frame.columns)
        and frame["date"].notna().any()
        and pd.to_numeric(frame["hfq_factor"], errors="coerce").notna().any()
    )


def _fetch_and_cache(session: requests.Session, asset: str, exchange: str) -> dict[str, Any]:
    symbol = _symbol(asset, exchange)
    response = session.get(SOURCE_URL.format(symbol=symbol), timeout=15)
    response.raise_for_status()
    raw = parse_sina_hfq_response(response.text)
    payload = raw.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    digest = hashlib.sha256(payload).hexdigest()
    path = RAW_DIR / f"{asset}_{digest[:16]}.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return {
        "status": "success",
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path": _relative(path),
        "sha256": digest,
        "rows": int(len(raw)),
        "source_url": SOURCE_URL.format(symbol=symbol),
    }


def _dividend_alignment(events: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    if not DIVIDEND_PATH.is_file() or events.empty:
        return {"eligible_events": 0, "matched_events": 0, "match_ratio": None, "unmatched_events": []}
    dividends = pd.read_csv(DIVIDEND_PATH, dtype={"asset": str})
    dividends["ex_date"] = pd.to_datetime(dividends["ex_date"], errors="coerce")
    dividends["cash_per_share"] = pd.to_numeric(dividends["cash_per_share"], errors="coerce")
    event_keys = set(zip(events["asset"], pd.to_datetime(events["effective_date"])))
    available_assets = set(events["asset"])
    eligible = dividends[
        dividends["asset"].isin(available_assets)
        & dividends["ex_date"].notna()
        & dividends["ex_date"].le(as_of)
        & dividends["cash_per_share"].gt(0)
    ].drop_duplicates(["asset", "ex_date"])
    matched_mask = pd.Series(
        [(row.asset, row.ex_date) in event_keys for row in eligible.itertuples(index=False)],
        index=eligible.index,
    )
    matched = int(matched_mask.sum())
    unmatched = eligible.loc[~matched_mask, ["asset", "ex_date"]].sort_values(["asset", "ex_date"])
    return {
        "eligible_events": int(len(eligible)),
        "matched_events": matched,
        "match_ratio": round(matched / len(eligible), 6) if len(eligible) else None,
        "unmatched_events": [
            {"asset": str(row.asset), "ex_date": row.ex_date.date().isoformat()}
            for row in unmatched.itertuples(index=False)
        ],
    }


def dividend_alignment_passes(alignment: dict[str, Any]) -> bool:
    ratio = alignment.get("match_ratio")
    return ratio is not None and math.isfinite(float(ratio)) and float(ratio) >= MIN_DIVIDEND_ALIGNMENT


def run_collection(as_of: str, collect_limit: int | None, sleep_seconds: float) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    lifecycles = _load_lifecycles()
    status = _load_status()
    assets_status = status.setdefault("assets", {})
    pending = [row for row in lifecycles.itertuples(index=False) if not _valid_cache(assets_status.get(row.asset, {}))]
    if collect_limit is not None:
        pending = pending[: max(0, collect_limit)]

    session = _session()
    for index, row in enumerate(pending):
        try:
            assets_status[row.asset] = _fetch_and_cache(session, row.asset, row.exchange)
        except (OSError, ValueError, requests.RequestException) as exc:
            assets_status[row.asset] = {
                "status": "failed",
                "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        _save_status(status)
        if sleep_seconds > 0 and index + 1 < len(pending):
            time.sleep(sleep_seconds)

    successful_inputs: list[dict[str, str]] = []
    event_frames: list[pd.DataFrame] = []
    normalisation_failures: dict[str, str] = {}
    for row in lifecycles.itertuples(index=False):
        entry = assets_status.get(row.asset, {})
        if not _valid_cache(entry):
            continue
        path = ROOT / entry["path"]
        try:
            raw = pd.read_csv(path)
            event_frames.append(normalise_factor_events(raw, row.asset, row.list_date, row.delist_date, as_of_date))
            successful_inputs.append({"source_id": row.asset, "source_url": entry["source_url"], "path": entry["path"], "sha256": entry["sha256"]})
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            normalisation_failures[row.asset] = f"{type(exc).__name__}: {str(exc)[:300]}"

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    complete_assets = int(events["asset"].nunique()) if not events.empty else 0
    target_assets = int(len(lifecycles))
    complete = complete_assets == target_assets and not normalisation_failures
    input_hashes = [f"{item['source_id']}:{item['sha256']}" for item in successful_inputs]
    master_hash = _sha256(MASTER_PATH)
    dividend_hash = _sha256(DIVIDEND_PATH) if DIVIDEND_PATH.is_file() else "missing"
    bundle_hash = hashlib.sha256("|".join([master_hash, dividend_hash, *sorted(input_hashes)]).encode()).hexdigest()
    source_vintage = f"sina_hfq_factor_bundle_sha256:{bundle_hash}"
    if not events.empty:
        events["data_source"] = "sina_hfq_factor"
        events["source_vintage"] = source_vintage
        events = events.sort_values(["effective_date", "asset"]).reset_index(drop=True)

    alignment = _dividend_alignment(events, as_of_date)
    alignment_pass = dividend_alignment_passes(alignment)
    formal = complete and alignment_pass
    blockers: list[str] = []
    if not complete:
        blockers.append("incomplete_asset_coverage")
    if not alignment_pass:
        blockers.append("dividend_ex_date_alignment_below_threshold")
    output_path = FORMAL_OUTPUT if formal else PROGRESS_OUTPUT
    manifest_path = FORMAL_MANIFEST if formal else PROGRESS_MANIFEST
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    inputs = [
        {"source_id": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": master_hash},
        *(
            [{"source_id": "stock_dividend_events", "path": _relative(DIVIDEND_PATH), "sha256": dividend_hash}]
            if DIVIDEND_PATH.is_file()
            else []
        ),
        *successful_inputs,
    ]
    code_path = Path(__file__).resolve()
    failed_assets = [asset for asset in lifecycles["asset"] if not _valid_cache(assets_status.get(asset, {}))]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(as_of_date.date()),
        "inputs": inputs,
        "source_vintage": source_vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "rows": int(len(events)),
        "assets": complete_assets,
        "target_assets": target_assets,
        "asset_coverage": round(complete_assets / target_assets, 6) if target_assets else 0.0,
        "failed_assets": failed_assets,
        "normalisation_failures": normalisation_failures,
        "dividend_ex_date_alignment": alignment,
        "minimum_dividend_alignment": MIN_DIVIDEND_ALIGNMENT,
        "asset_coverage_pass": complete,
        "dividend_alignment_pass": alignment_pass,
        "qualification_blockers": blockers,
        "qualification_status": "PASS" if formal else "COLLECTION_IN_PROGRESS",
        "historical_backtest_allowed": bool(formal),
        "model_promotion_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--collect-limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_collection(args.as_of, args.collect_limit, args.sleep_seconds)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "qualification_status",
                    "rows",
                    "assets",
                    "target_assets",
                    "asset_coverage",
                    "dividend_ex_date_alignment",
                    "historical_backtest_allowed",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
