"""Incrementally refresh the permitted Tushare all-stock daily endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = ROOT / "data_raw" / "tushare_daily_only" / "v3_38" / "daily"
CALENDAR_PATH = ROOT / "data_raw" / "akshare" / "calendar" / "trade_calendar.csv"
CREDENTIALS_PATH = ROOT / "configs" / "data_credentials.json"
MANIFEST_DIR = ROOT / "data_raw" / "long_hold_v4" / "manifests"
LATEST_MANIFEST = MANIFEST_DIR / "tushare_daily_refresh_latest.json"
RUN_DIR = MANIFEST_DIR / "tushare_daily_refresh_runs"
STATUS_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "tushare_daily_refresh_status.csv"

RAW_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
OUTPUT_COLUMNS = [
    *RAW_COLUMNS,
    "asset",
    "date",
    "available_date",
    "data_source",
    "fetched_at",
    "price_adjustment",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(path)


def _load_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    if not CREDENTIALS_PATH.is_file():
        raise RuntimeError("Tushare credential is unavailable")
    payload = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8-sig"))
    token = str(payload.get("tushare", {}).get("token", "")).strip()
    if not token:
        raise RuntimeError("Tushare credential is unavailable")
    return token


def _daily_path(trade_date: str) -> Path:
    return DAILY_DIR / f"trade_date={trade_date}.csv"


def load_trade_dates(
    start_date: str,
    as_of: str,
    calendar_path: Path | None = None,
) -> list[str]:
    calendar_path = calendar_path or CALENDAR_PATH
    calendar = pd.read_csv(calendar_path, usecols=["date"])
    dates = pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.normalize()
    start = pd.Timestamp(start_date).normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    if cutoff < start:
        raise ValueError("as_of must not precede start_date")
    selected = dates[dates.between(start, cutoff)].drop_duplicates().sort_values()
    return selected.dt.strftime("%Y%m%d").tolist()


def normalise_daily(raw: pd.DataFrame, trade_date: str, fetched_at: str) -> pd.DataFrame:
    missing = sorted(set(RAW_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Tushare daily response missing columns: {missing}")
    out = raw[RAW_COLUMNS].copy()
    if out.empty:
        raise ValueError(f"Tushare daily response is empty for open date {trade_date}")
    out["ts_code"] = out["ts_code"].astype(str)
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    if not out["trade_date"].eq(trade_date).all():
        raise ValueError(f"Tushare response contains a different trade date for {trade_date}")
    if out["ts_code"].duplicated().any():
        raise ValueError(f"Tushare response contains duplicate securities for {trade_date}")
    numeric = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    required_positive = ["open", "high", "low", "close", "pre_close"]
    if out[required_positive].isna().any().any() or not out[required_positive].gt(0).all().all():
        raise ValueError(f"Tushare response has invalid prices for {trade_date}")
    if out[["vol", "amount"]].isna().any().any() or not out[["vol", "amount"]].ge(0).all().all():
        raise ValueError(f"Tushare response has invalid volume or amount for {trade_date}")
    ohlc_ok = (
        out["high"].ge(out[["open", "close", "low"]].max(axis=1))
        & out["low"].le(out[["open", "close", "high"]].min(axis=1))
    )
    if not ohlc_ok.all():
        raise ValueError(f"Tushare response has inconsistent OHLC values for {trade_date}")
    out["asset"] = out["ts_code"]
    out["date"] = out["trade_date"]
    out["available_date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    out["data_source"] = "tushare.daily"
    out["fetched_at"] = fetched_at
    out["price_adjustment"] = "none_raw"
    return out[OUTPUT_COLUMNS].sort_values("ts_code").reset_index(drop=True)


def validate_daily_file(path: Path, trade_date: str) -> tuple[bool, int, str]:
    if not path.is_file():
        return False, 0, "missing"
    try:
        frame = pd.read_csv(path, low_memory=False)
        normalise_daily(frame, trade_date, str(frame.get("fetched_at", pd.Series([""])).iloc[0]))
    except (OSError, ValueError, KeyError, IndexError, pd.errors.ParserError) as exc:
        return False, 0, f"{type(exc).__name__}:{str(exc)[:200]}"
    return True, int(len(frame)), "validated_existing"


def refresh_daily(
    start_date: str,
    as_of: str,
    max_calls: int = 50,
    sleep_seconds: float = 1.25,
    pro: Any | None = None,
) -> dict[str, Any]:
    if max_calls < 0:
        raise ValueError("max_calls must be non-negative")
    trade_dates = load_trade_dates(start_date, as_of)
    statuses: list[dict[str, Any]] = []
    pending: list[str] = []
    for trade_date in trade_dates:
        path = _daily_path(trade_date)
        valid, rows, detail = validate_daily_file(path, trade_date)
        if valid:
            statuses.append(
                {
                    "trade_date": trade_date,
                    "status": "reused",
                    "rows": rows,
                    "path": _relative(path),
                    "sha256": _sha256(path),
                    "detail": detail,
                }
            )
        else:
            pending.append(trade_date)
    selected = pending[:max_calls]
    deferred = pending[max_calls:]
    if selected and pro is None:
        import tushare as ts

        pro = ts.pro_api(_load_token())
    for position, trade_date in enumerate(selected):
        path = _daily_path(trade_date)
        try:
            raw = pro.daily(trade_date=trade_date)
            fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
            clean = normalise_daily(raw, trade_date, fetched_at)
            _atomic_csv(clean, path)
            statuses.append(
                {
                    "trade_date": trade_date,
                    "status": "fetched",
                    "rows": int(len(clean)),
                    "path": _relative(path),
                    "sha256": _sha256(path),
                    "detail": "validated_and_written",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "trade_date": trade_date,
                    "status": "failed",
                    "rows": 0,
                    "path": _relative(path),
                    "sha256": "",
                    "detail": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
            )
        if sleep_seconds > 0 and position + 1 < len(selected):
            time.sleep(sleep_seconds)

    for trade_date in deferred:
        statuses.append(
            {
                "trade_date": trade_date,
                "status": "deferred",
                "rows": 0,
                "path": _relative(_daily_path(trade_date)),
                "sha256": "",
                "detail": f"max_calls={max_calls}",
            }
        )
    status_columns = ["trade_date", "status", "rows", "path", "sha256", "detail"]
    status = pd.DataFrame(statuses, columns=status_columns)
    if not status.empty:
        status = status.sort_values("trade_date").reset_index(drop=True)
    _atomic_csv(status, STATUS_PATH)
    failed = status[status["status"].eq("failed")]
    deferred_rows = status[status["status"].eq("deferred")]
    ready = bool(len(status) == len(trade_dates) and failed.empty and deferred_rows.empty)
    output_items = [
        {"role": "daily_file", "trade_date": row.trade_date, "path": row.path, "sha256": row.sha256}
        for row in status.itertuples(index=False)
        if row.status in {"reused", "fetched"}
    ]
    code_path = Path(__file__).resolve()
    run_seed = f"{as_of}|{datetime.now().astimezone().isoformat()}|{_sha256(code_path)}"
    run_id = f"{pd.Timestamp(as_of).strftime('%Y%m%d')}_{hashlib.sha256(run_seed.encode()).hexdigest()[:12]}"
    payload = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "as_of_date": pd.Timestamp(as_of).date().isoformat(),
        "target_trade_dates": len(trade_dates),
        "reused_trade_dates": int(status["status"].eq("reused").sum()),
        "fetched_trade_dates": int(status["status"].eq("fetched").sum()),
        "failed_trade_dates": failed["trade_date"].astype(str).tolist(),
        "deferred_trade_dates": deferred_rows["trade_date"].astype(str).tolist(),
        "status_path": _relative(STATUS_PATH),
        "status_sha256": _sha256(STATUS_PATH),
        "inputs": [
            {"role": "trade_calendar", "path": _relative(CALENDAR_PATH), "sha256": _sha256(CALENDAR_PATH)},
        ],
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "outputs": output_items,
        "qualification_status": "REFRESH_COMPLETE_RAW_DAILY_ONLY" if ready else "REFRESH_INCOMPLETE",
        "raw_price_only": True,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(payload, LATEST_MANIFEST)
    immutable_path = RUN_DIR / f"{run_id}.json"
    _atomic_json(payload, immutable_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--max-calls", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_daily(
        start_date=args.start_date,
        as_of=args.as_of,
        max_calls=args.max_calls,
        sleep_seconds=args.sleep_seconds,
    )
    print(
        json.dumps(
            {
                "qualification_status": result["qualification_status"],
                "target_trade_dates": result["target_trade_dates"],
                "reused_trade_dates": result["reused_trade_dates"],
                "fetched_trade_dates": result["fetched_trade_dates"],
                "failed_trade_dates": result["failed_trade_dates"],
                "deferred_trade_dates": result["deferred_trade_dates"],
                "manifest": _relative(LATEST_MANIFEST),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
